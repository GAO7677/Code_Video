#!/usr/bin/env python3
"""Compare CoTracker and DiffTrack Q/K tracks for individual scene regions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "diffusers" / "src"))
sys.path.insert(0, str(REPO_ROOT))

import diffusers
from diffusers import CogVideoXPipeline


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")
DEFAULT_MODEL_PATH = Path("/data/gaoya/agent-data/weights/CogVideoX-2b-modelscope")
DEFAULT_COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
DEFAULT_COTRACKER_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
DEFAULT_OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--case-key", default="case_019_wheel_hits_block")
    parser.add_argument("--sample-type", choices=["base", "background_color", "object_color", "object_shape"], default="base")
    parser.add_argument("--regions", nargs="+", choices=["object_a", "object_b", "background"], default=["object_a", "object_b", "background"])
    parser.add_argument("--points-per-region", type=int, default=32)
    parser.add_argument("--query-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--mask-erode-px", type=int, default=7)
    parser.add_argument("--background-edge-px", type=int, default=15)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--inverse-step", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cotracker-root", type=Path, default=DEFAULT_COTRACKER_ROOT)
    parser.add_argument("--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--trace-length", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_sample(manifest: dict, case_key: str, sample_type: str) -> dict:
    case = next((item for item in manifest["cases"] if item["case_key"] == case_key), None)
    if case is None:
        raise KeyError(f"Case not found in dataset manifest: {case_key}")
    if int(case["object_count"]) != 2:
        raise ValueError(f"The first region experiment requires two objects, got {case['object_count']}")
    if sample_type == "base":
        return case["base"]
    pairs = case.get("pairs", {})
    if isinstance(pairs, list):
        variants = {item["attribute"]: item.get("variant", item) for item in pairs}
    else:
        variants = {key: value.get("variant", value) for key, value in pairs.items()}
    if sample_type not in variants:
        raise KeyError(f"Variant {sample_type!r} is unavailable for {case_key}")
    return variants[sample_type]


def read_video(path: Path, num_frames: int, height: int, width: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != num_frames:
        raise ValueError(f"{path} contains {len(frames)} frames, expected at least {num_frames}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False)
    return F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)


def read_instance_ids(path: Path, num_frames: int, height: int, width: int) -> tuple[np.ndarray, list[str], list[int]]:
    data = np.load(path)
    instance_ids = torch.from_numpy(data["instance_ids"][:num_frames]).unsqueeze(1).float()
    if instance_ids.shape[0] != num_frames:
        raise ValueError(f"{path} contains {instance_ids.shape[0]} masks, expected {num_frames}")
    instance_ids = F.interpolate(instance_ids, size=(256, 256), mode="nearest")
    instance_ids = F.interpolate(instance_ids, size=(height, width), mode="nearest")
    names = [str(name) for name in data["object_names"]]
    ids = [int(value) for value in data["object_ids"]]
    return instance_ids[:, 0].byte().numpy(), names, ids


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def build_region_masks(
    instance_ids: np.ndarray,
    object_names: list[str],
    object_ids: list[int],
    query_frame: int,
    mask_erode_px: int,
    background_edge_px: int,
) -> dict[str, dict]:
    if len(object_ids) != 2:
        raise ValueError(f"Expected two object IDs, got {object_ids}")
    frame_ids = instance_ids[query_frame]
    object_masks = [frame_ids == object_id for object_id in object_ids]
    union = np.logical_or.reduce(object_masks)
    background = erode_mask(~union, background_edge_px)
    border = max(background_edge_px, 16)
    background[:border] = False
    background[-border:] = False
    background[:, :border] = False
    background[:, -border:] = False
    return {
        "object_a": {
            "mask": erode_mask(object_masks[0], mask_erode_px),
            "label": f"object A: {object_names[0]}",
            "object_name": object_names[0],
            "object_id": object_ids[0],
        },
        "object_b": {
            "mask": erode_mask(object_masks[1], mask_erode_px),
            "label": f"object B: {object_names[1]}",
            "object_name": object_names[1],
            "object_id": object_ids[1],
        },
        "background": {
            "mask": background,
            "label": "background",
            "object_name": "background",
            "object_id": 0,
        },
    }


def farthest_point_sample(mask: np.ndarray, count: int) -> np.ndarray:
    yx = np.argwhere(mask)
    if len(yx) < count:
        raise ValueError(f"Mask contains only {len(yx)} valid pixels, cannot sample {count}")
    center = yx.mean(axis=0)
    first = int(np.argmin(np.square(yx - center).sum(axis=1)))
    selected = [first]
    min_distance = np.square(yx - yx[first]).sum(axis=1).astype(np.float64)
    for _ in range(1, count):
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.square(yx - yx[next_index]).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
    sampled_yx = yx[np.asarray(selected)]
    return sampled_yx[:, ::-1].astype(np.float32)


def point_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(count):
        hue = int(round(179 * index / max(count, 1)))
        hsv = np.uint8([[[hue, 210, 255]]])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(int(value) for value in rgb))
    return colors


def draw_mask_points(frame: np.ndarray, mask: np.ndarray, points: np.ndarray, label: str, output_path: Path) -> None:
    image = frame.copy()
    tint = np.zeros_like(image)
    tint[..., 1] = 220
    image[mask] = (0.55 * image[mask] + 0.45 * tint[mask]).astype(np.uint8)
    for index, (x, y) in enumerate(points):
        color = point_colors(len(points))[index]
        cv2.circle(image, (round(float(x)), round(float(y))), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(image, (round(float(x)), round(float(y))), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(image, f"{label} | {len(points)} query points", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{label} | {len(points)} query points", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 1, cv2.LINE_AA)
    imageio.imwrite(output_path, image)


def draw_track_video(
    frames: np.ndarray,
    tracks: np.ndarray,
    output_path: Path,
    label: str,
    method: str,
    fps: float,
    trace_length: int,
    visibility: np.ndarray | None = None,
) -> None:
    colors = point_colors(tracks.shape[1])
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=None) as writer:
        for frame_index, frame in enumerate(frames):
            canvas = frame.copy()
            trace_start = max(0, frame_index - trace_length)
            for point_index, color in enumerate(colors):
                for time_index in range(trace_start + 1, frame_index + 1):
                    if visibility is not None and not (visibility[time_index - 1, point_index] and visibility[time_index, point_index]):
                        continue
                    p0 = tuple(np.rint(tracks[time_index - 1, point_index]).astype(int))
                    p1 = tuple(np.rint(tracks[time_index, point_index]).astype(int))
                    cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
                visible = visibility is None or bool(visibility[frame_index, point_index])
                if not visible:
                    continue
                point = tuple(np.rint(tracks[frame_index, point_index]).astype(int))
                if method == "CoTracker":
                    cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, point, 5, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(canvas, (point[0] - 4, point[1] - 4), (point[0] + 4, point[1] + 4), color, -1, cv2.LINE_AA)
                    cv2.rectangle(canvas, (point[0] - 5, point[1] - 5), (point[0] + 5, point[1] + 5), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{label} | {method} | frame {frame_index:02d}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{label} | {method} | frame {frame_index:02d}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 1, cv2.LINE_AA)
            writer.append_data(canvas)


def draw_comparison_video(
    frames: np.ndarray,
    cotracker: np.ndarray,
    qk: np.ndarray,
    visibility: np.ndarray,
    output_path: Path,
    label: str,
    fps: float,
    trace_length: int,
) -> None:
    colors = point_colors(cotracker.shape[1])
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=None) as writer:
        for frame_index, frame in enumerate(frames):
            canvas = frame.copy()
            trace_start = max(0, frame_index - trace_length)
            for point_index, color in enumerate(colors):
                for time_index in range(trace_start + 1, frame_index + 1):
                    q0 = tuple(np.rint(qk[time_index - 1, point_index]).astype(int))
                    q1 = tuple(np.rint(qk[time_index, point_index]).astype(int))
                    cv2.line(canvas, q0, q1, color, 1, cv2.LINE_AA)
                    if visibility[time_index - 1, point_index] and visibility[time_index, point_index]:
                        c0 = tuple(np.rint(cotracker[time_index - 1, point_index]).astype(int))
                        c1 = tuple(np.rint(cotracker[time_index, point_index]).astype(int))
                        cv2.line(canvas, c0, c1, color, 3, cv2.LINE_AA)
                q_point = tuple(np.rint(qk[frame_index, point_index]).astype(int))
                cv2.rectangle(canvas, (q_point[0] - 4, q_point[1] - 4), (q_point[0] + 4, q_point[1] + 4), color, 2, cv2.LINE_AA)
                if visibility[frame_index, point_index]:
                    c_point = tuple(np.rint(cotracker[frame_index, point_index]).astype(int))
                    cv2.line(canvas, c_point, q_point, (220, 220, 220), 1, cv2.LINE_AA)
                    cv2.circle(canvas, c_point, 4, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, c_point, 5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (10, 8), (408, 61), (0, 0, 0), -1)
            cv2.putText(canvas, f"{label} | frame {frame_index:02d}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "CoTracker: circle/thick | Q/K: square/thin", (18, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            writer.append_data(canvas)


def compute_metrics(qk: np.ndarray, cotracker: np.ndarray, visibility: np.ndarray) -> dict:
    distance = np.linalg.norm(qk - cotracker, axis=-1)
    valid = visibility.astype(bool)
    valid_after_query = valid.copy()
    valid_after_query[0] = False
    values = distance[valid_after_query]
    if values.size == 0:
        raise ValueError("No visible CoTracker points after the query frame")
    metrics = {
        "visible_comparisons": int(values.size),
        "mean_error_px": float(values.mean()),
        "median_error_px": float(np.median(values)),
        "pck4": float((values < 4).mean() * 100),
        "pck8": float((values < 8).mean() * 100),
        "pck16": float((values < 16).mean() * 100),
        "per_frame_mean_error_px": [],
        "per_frame_pck8": [],
    }
    for frame_index in range(len(distance)):
        frame_values = distance[frame_index, valid[frame_index]]
        metrics["per_frame_mean_error_px"].append(float(frame_values.mean()) if frame_values.size else None)
        metrics["per_frame_pck8"].append(float((frame_values < 8).mean() * 100) if frame_values.size else None)
    return metrics


def seed_everything(seed: int, device: str) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def main() -> None:
    args = parse_args()
    if args.query_frame != 0:
        raise ValueError("The DiffTrack Q/K processor currently supports query_frame=0 only")
    expected_diffusers = REPO_ROOT / "diffusers" / "src" / "diffusers"
    if Path(diffusers.__file__).resolve().parent != expected_diffusers:
        raise RuntimeError(f"Expected DiffTrack diffusers at {expected_diffusers}, loaded {diffusers.__file__}")

    manifest = json.loads((args.dataset_root / "dataset_manifest.json").read_text())
    sample = find_sample(manifest, args.case_key, args.sample_type)
    video_path = Path(sample["video"])
    mask_path = Path(sample["mask_ids"])
    video = read_video(video_path, args.num_frames, args.height, args.width)
    instance_ids, object_names, object_ids = read_instance_ids(mask_path, args.num_frames, args.height, args.width)
    region_info = build_region_masks(
        instance_ids,
        object_names,
        object_ids,
        args.query_frame,
        args.mask_erode_px,
        args.background_edge_px,
    )
    regions = {}
    for region_name in args.regions:
        info = region_info[region_name]
        regions[region_name] = {
            **info,
            "points": farthest_point_sample(info["mask"], args.points_per_region),
        }

    output_root = args.output_dir / f"{args.case_key}_{args.sample_type}" / f"layer{args.layer}_step{args.inverse_step}"
    output_root.mkdir(parents=True, exist_ok=True)
    frames = video.permute(0, 2, 3, 1).byte().numpy()
    for region_name, info in regions.items():
        region_dir = output_root / region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        draw_mask_points(frames[args.query_frame], info["mask"], info["points"], info["label"], region_dir / "mask_points.png")

    sys.path.insert(0, str(args.cotracker_root))
    from cotracker.predictor import CoTrackerPredictor

    cotracker_model = CoTrackerPredictor(checkpoint=str(args.cotracker_checkpoint), offline=True).to(args.device).eval()
    video_device = video.unsqueeze(0).to(args.device)
    for region_name, info in regions.items():
        points = torch.from_numpy(info["points"]).to(args.device)
        query_times = torch.full((len(points), 1), float(args.query_frame), device=args.device)
        queries = torch.cat((query_times, points), dim=-1).unsqueeze(0)
        with torch.inference_mode():
            tracks, visibility = cotracker_model(video_device, queries=queries)
        info["cotracker"] = tracks[0].cpu().numpy().astype(np.float32)
        info["visibility"] = visibility[0].cpu().numpy().astype(bool)
        print(f"{region_name}: CoTracker complete, visible={info['visibility'].mean():.4f}")
    del cotracker_model, video_device
    torch.cuda.empty_cache()

    pipe = CogVideoXPipeline.from_pretrained(str(args.model_path), torch_dtype=torch.bfloat16).to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    for region_name, info in regions.items():
        query_coords = torch.from_numpy(info["points"]).unsqueeze(0).to(args.device)
        params = {
            "trajectory": True,
            "attn_weight": False,
            "query_key": False,
            "feature": False,
            "video_mode": "fg",
            "matching_layer": [args.layer],
            "query_coords": query_coords,
        }
        generator = seed_everything(args.seed, args.device)
        with torch.inference_mode():
            _, _, _, trajectory_qk = pipe(
                prompt="",
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                return_dict=False,
                generator=generator,
                vis_timesteps=[args.inverse_step],
                vis_layers=[args.layer],
                output_type="latent",
                params=params,
                video=video.unsqueeze(0),
                inverse_step=args.inverse_step,
            )
        if trajectory_qk is None:
            raise RuntimeError(f"Q/K trajectory was not captured for layer={args.layer}, step={args.inverse_step}")
        info["qk"] = trajectory_qk[0].cpu().numpy().astype(np.float32)
        print(f"{region_name}: Q/K trajectory complete")

    run_manifest = {
        "case_key": args.case_key,
        "sample_type": args.sample_type,
        "video": str(video_path),
        "instance_masks": str(mask_path),
        "mask_source": "renderer_lossless_instance_ids",
        "preprocessing": "source -> 256x256 -> 480x720",
        "query_frame": args.query_frame,
        "points_per_region": args.points_per_region,
        "layer": args.layer,
        "inverse_step": args.inverse_step,
        "regions": {},
    }
    for region_name, info in regions.items():
        region_dir = output_root / region_name
        metrics = compute_metrics(info["qk"], info["cotracker"], info["visibility"])
        np.savez_compressed(
            region_dir / "tracks.npz",
            query_points=info["points"],
            region_mask=info["mask"],
            cotracker_tracks=info["cotracker"],
            cotracker_visibility=info["visibility"],
            qk_tracks=info["qk"],
        )
        (region_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        draw_track_video(
            frames,
            info["cotracker"],
            region_dir / "cotracker_tracks.mp4",
            info["label"],
            "CoTracker",
            args.fps,
            args.trace_length,
            info["visibility"],
        )
        draw_track_video(
            frames,
            info["qk"],
            region_dir / "qk_tracks.mp4",
            info["label"],
            "Q/K",
            args.fps,
            args.trace_length,
        )
        draw_comparison_video(
            frames,
            info["cotracker"],
            info["qk"],
            info["visibility"],
            region_dir / "overlay_comparison.mp4",
            info["label"],
            args.fps,
            args.trace_length,
        )
        run_manifest["regions"][region_name] = {
            "label": info["label"],
            "object_name": info["object_name"],
            "object_id": info["object_id"],
            "valid_mask_pixels": int(info["mask"].sum()),
            "metrics": metrics,
        }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"Results saved to {output_root}")


if __name__ == "__main__":
    main()
