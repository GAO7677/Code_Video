#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.proxy_state import read_video_frames
from phys_state_video.schemas import STATE_DIM, StateIndex


DEFAULT_INPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/raw_v1/industrial_s1_pilot")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_pilot_256x144_s8_f16_n6")
SHAPE_INDEX = {"sphere": 0, "box": 1, "cylinder": 2, "capsule": 3, "puck": 4}
ROLE_INDEX = {"dynamic": 0, "support": 1, "occluder": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw simulation samples into phys-state-video episodes.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--context-steps", type=int, default=8)
    parser.add_argument("--future-steps", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--appearance-dim", type=int, default=16)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw.astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_camera(meta: dict, width: int, height: int) -> dict[str, np.ndarray | float]:
    camera = meta["camera"]
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    yfov = math.radians(float(camera.get("yfov_deg", 50.0)))
    aspect = float(width) / float(height)
    fx = 0.5 * width / (math.tan(yfov * 0.5) * aspect)
    fy = 0.5 * height / math.tan(yfov * 0.5)
    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": true_up,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5,
        "cy": height * 0.5,
    }


def object_local_corners(obj: dict) -> np.ndarray:
    shape = obj["shape"]
    size = obj["size"]
    if shape == "sphere":
        radius = float(size["radius"])
        ext = np.asarray([radius, radius, radius], dtype=np.float64)
    elif shape == "box":
        ext = np.asarray([size["hx"], size["hy"], size["hz"]], dtype=np.float64)
    elif shape == "cylinder":
        ext = np.asarray([size["radius"], size["radius"], 0.5 * size["height"]], dtype=np.float64)
    elif shape == "capsule":
        ext = np.asarray([size["radius"], size["radius"], 0.5 * size["height"] + size["radius"]], dtype=np.float64)
    elif shape == "puck":
        ext = np.asarray([size["radius"], size["radius"], 0.5 * size["height"]], dtype=np.float64)
    else:
        raise ValueError(f"unsupported object shape: {shape}")
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * ext[None, :]


def project_points(points_world: np.ndarray, camera: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray]:
    eye = camera["eye"]
    right = camera["right"]
    up = camera["up"]
    forward = camera["forward"]
    delta = points_world - eye[None, :]
    x_cam = delta @ right
    y_cam = delta @ up
    z_cam = delta @ forward
    valid = z_cam > 1e-5
    u = camera["fx"] * (x_cam / np.clip(z_cam, 1e-5, None)) + camera["cx"]
    v = camera["cy"] - camera["fy"] * (y_cam / np.clip(z_cam, 1e-5, None))
    pixels = np.stack([u, v], axis=1)
    return pixels, valid


def projected_box_from_object(position: np.ndarray, quat: np.ndarray, obj: dict, camera: dict[str, np.ndarray | float], width: int, height: int) -> tuple[np.ndarray, np.ndarray, float]:
    corners_local = object_local_corners(obj)
    rot = quat_to_matrix(quat)
    corners_world = corners_local @ rot.T + position[None, :]
    pixels, valid = project_points(corners_world, camera)
    if not np.any(valid):
        return np.zeros((4,), dtype=np.float32), np.zeros((2,), dtype=np.float32), 0.0

    visible_pixels = pixels[valid]
    x0 = float(np.min(visible_pixels[:, 0]))
    y0 = float(np.min(visible_pixels[:, 1]))
    x1 = float(np.max(visible_pixels[:, 0]))
    y1 = float(np.max(visible_pixels[:, 1]))
    unclipped = np.asarray([x0, y0, x1, y1], dtype=np.float32)
    clipped = unclipped.copy()
    clipped[0::2] = np.clip(clipped[0::2], 0.0, float(width))
    clipped[1::2] = np.clip(clipped[1::2], 0.0, float(height))
    center = np.asarray([(unclipped[0] + unclipped[2]) * 0.5 / width, (unclipped[1] + unclipped[3]) * 0.5 / height], dtype=np.float32)
    area = max((unclipped[2] - unclipped[0]) * (unclipped[3] - unclipped[1]), 1e-6)
    return clipped / np.asarray([width, height, width, height], dtype=np.float32), center, float(area)


def object_depth(position: np.ndarray, camera: dict[str, np.ndarray | float]) -> float:
    delta = position.astype(np.float64) - camera["eye"]
    return float(delta @ camera["forward"])


def appearance_vector(obj: dict, appearance_dim: int) -> np.ndarray:
    vec = np.zeros((appearance_dim,), dtype=np.float32)
    shape_idx = SHAPE_INDEX[obj["shape"]]
    role_idx = ROLE_INDEX[obj["role"]]
    vec[shape_idx] = 1.0
    vec[5 + role_idx] = 1.0
    vec[8:11] = np.asarray(obj["color"], dtype=np.float32)

    size = obj["size"]
    if obj["shape"] == "sphere":
        extents = np.asarray([2.0 * size["radius"], 2.0 * size["radius"], 2.0 * size["radius"]], dtype=np.float32)
    elif obj["shape"] == "box":
        extents = np.asarray([2.0 * size["hx"], 2.0 * size["hy"], 2.0 * size["hz"]], dtype=np.float32)
    elif obj["shape"] == "cylinder":
        extents = np.asarray([2.0 * size["radius"], 2.0 * size["radius"], size["height"]], dtype=np.float32)
    elif obj["shape"] == "capsule":
        extents = np.asarray([2.0 * size["radius"], 2.0 * size["radius"], size["height"] + 2.0 * size["radius"]], dtype=np.float32)
    elif obj["shape"] == "puck":
        extents = np.asarray([2.0 * size["radius"], 2.0 * size["radius"], size["height"]], dtype=np.float32)
    else:
        extents = np.zeros((3,), dtype=np.float32)

    vec[11] = float(np.max(extents))
    vec[12] = float(np.min(extents))
    vec[13] = float(np.prod(extents))
    vec[14] = float(obj["mass"])
    vec[15] = float(obj["friction"])
    return vec


def build_prompt(meta: dict) -> str:
    shape_tokens = " ".join(obj["shape"] for obj in meta["objects"])
    return f"{meta['key'].replace('_', ' ')} industrial rigid body simulation {shape_tokens}"


def sample_directories(input_root: Path, split: str) -> list[Path]:
    split_root = input_root / split
    if not split_root.exists():
        return []
    dirs: list[Path] = []
    for family_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        dirs.extend(sorted(path for path in family_dir.iterdir() if path.is_dir()))
    return dirs


def build_camera_vector(camera: dict[str, np.ndarray | float], width: int, height: int) -> np.ndarray:
    return np.asarray(
        [
            camera["fx"] / width,
            camera["fy"] / height,
            camera["cx"] / width,
            camera["cy"] / height,
            *camera["eye"].tolist(),
            *(-camera["forward"]).tolist(),
        ],
        dtype=np.float32,
    )


def process_sample(sample_dir: Path, args: argparse.Namespace, split: str, output_split_root: Path, manifest_records: list[dict]) -> int:
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    states = np.load(sample_dir / "states.npz")
    frames = read_video_frames(sample_dir / "video.mp4", resize_height=args.height, resize_width=args.width)
    frames = frames[:: args.frame_stride]
    total_steps = args.context_steps + args.future_steps
    if frames.shape[0] < total_steps:
        return 0

    positions = states["positions"][:: args.frame_stride]
    quats = states["quats"][:: args.frame_stride]
    object_names = [str(name) for name in states["object_names"]]
    objects = meta["objects"]
    object_lookup = {obj["name"]: obj for obj in objects}
    ordered_objects = [object_lookup[name] for name in object_names]
    if len(ordered_objects) > args.max_objects:
        raise ValueError(f"{sample_dir} has {len(ordered_objects)} objects but max_objects={args.max_objects}")

    camera = make_camera(meta, args.width, args.height)
    num_frames = frames.shape[0]
    num_objects = len(ordered_objects)
    boxes = np.zeros((num_frames, args.max_objects, 4), dtype=np.float32)
    states_arr = np.zeros((num_frames, args.max_objects, STATE_DIM), dtype=np.float32)
    raw_depths = np.zeros((num_frames, args.max_objects), dtype=np.float32)
    raw_centers = np.zeros((num_frames, args.max_objects, 2), dtype=np.float32)
    raw_areas = np.zeros((num_frames, args.max_objects), dtype=np.float32)

    for obj_idx, obj in enumerate(ordered_objects):
        states_arr[:, obj_idx, StateIndex.EXISTENCE] = 1.0
        states_arr[:, obj_idx, StateIndex.CONFIDENCE] = 1.0
        for frame_idx in range(num_frames):
            position = positions[frame_idx, obj_idx]
            quat = quats[frame_idx, obj_idx]
            box, center, area = projected_box_from_object(position, quat, obj, camera, args.width, args.height)
            depth = object_depth(position, camera)
            raw_depths[frame_idx, obj_idx] = max(depth, 1e-4)
            raw_centers[frame_idx, obj_idx] = center
            raw_areas[frame_idx, obj_idx] = area
            boxes[frame_idx, obj_idx] = box
            visible = float(
                depth > 1e-4
                and box[2] > box[0]
                and box[3] > box[1]
                and box[0] < 1.0
                and box[1] < 1.0
                and box[2] > 0.0
                and box[3] > 0.0
            )
            states_arr[frame_idx, obj_idx, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = center
            states_arr[frame_idx, obj_idx, StateIndex.VISIBILITY] = visible

    visible_depths = raw_depths[states_arr[..., StateIndex.VISIBILITY] > 0.5]
    depth_ref = float(np.median(visible_depths)) if visible_depths.size else 1.0
    states_arr[..., StateIndex.DEPTH] = raw_depths / max(depth_ref, 1e-4)
    states_arr[..., StateIndex.LOG_SCALE] = np.log(np.clip(raw_areas / float(args.width * args.height), 1e-6, None))
    states_arr[1:, :, StateIndex.VEL_X:StateIndex.VEL_Y + 1] = raw_centers[1:] - raw_centers[:-1]
    states_arr[1:, :, StateIndex.DEPTH_VEL] = states_arr[1:, :, StateIndex.DEPTH] - states_arr[:-1, :, StateIndex.DEPTH]

    appearance = np.zeros((args.max_objects, args.appearance_dim), dtype=np.float32)
    for obj_idx, obj in enumerate(ordered_objects):
        appearance[obj_idx] = appearance_vector(obj, args.appearance_dim)

    camera_vec = build_camera_vector(camera, args.width, args.height)
    camera_seq = np.repeat(camera_vec[None, :], args.context_steps, axis=0)
    windows_written = 0
    output_split_root.mkdir(parents=True, exist_ok=True)
    max_start = num_frames - total_steps
    for window_idx, start in enumerate(range(0, max_start + 1, args.window_stride)):
        end = start + total_steps
        context_slice = slice(start, start + args.context_steps)
        future_slice = slice(start + args.context_steps, end)
        episode_name = f"{sample_dir.name}_w{window_idx:03d}"
        episode_path = output_split_root / f"{episode_name}.npz"
        meta_path = output_split_root / f"{episode_name}.json"
        if episode_path.exists() and not args.overwrite:
            continue
        np.savez_compressed(
            episode_path,
            context_frames=frames[context_slice].astype(np.float32),
            future_frames=frames[future_slice].astype(np.float32),
            context_states=states_arr[context_slice].astype(np.float32),
            future_states=states_arr[future_slice].astype(np.float32),
            context_boxes=boxes[context_slice].astype(np.float32),
            future_boxes=boxes[future_slice].astype(np.float32),
            appearance=appearance.astype(np.float32),
            camera=camera_seq.astype(np.float32),
        )
        meta_payload = {
            "prompt": build_prompt(meta),
            "sample_dir": str(sample_dir),
            "sample_id": meta.get("sample_id", sample_dir.name),
            "template_key": meta.get("template_key", meta["key"]),
            "split": split,
            "window_index": window_idx,
            "frame_stride": args.frame_stride,
            "window_start": start,
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_records.append({
            "episode": episode_name,
            "split": split,
            "sample_dir": str(sample_dir),
            "template_key": meta.get("template_key", meta["key"]),
            "window_index": window_idx,
            "window_start": start,
        })
        windows_written += 1
    return windows_written


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_root": str(args.input_root),
        "output_root": str(output_root),
        "height": args.height,
        "width": args.width,
        "context_steps": args.context_steps,
        "future_steps": args.future_steps,
        "frame_stride": args.frame_stride,
        "window_stride": args.window_stride,
        "max_objects": args.max_objects,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        sample_dirs = sample_directories(args.input_root, split)
        if args.limit_samples is not None:
            sample_dirs = sample_dirs[: args.limit_samples]
        records: list[dict] = []
        total_episodes = 0
        split_root = output_root / split
        for sample_dir in sample_dirs:
            total_episodes += process_sample(sample_dir, args, split, split_root, records)
        manifest["splits"][split] = {
            "samples": len(sample_dirs),
            "episodes": total_episodes,
            "records": records,
        }
        print(f"[split={split}] samples={len(sample_dirs)} episodes={total_episodes}")

    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved manifest to {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
