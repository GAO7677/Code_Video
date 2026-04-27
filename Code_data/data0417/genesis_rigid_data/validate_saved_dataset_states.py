"""Build a local browser to validate saved Genesis rigid benchmark samples against GT states.

The script reads saved benchmark samples (for example the flat ``mytest`` folder),
loads the original physics states from each sample's ``source_sample_dir``, and
renders:

- an overlay video on top of the saved ``full_video.mp4``
- a collision-frame strip highlighting start / peak / end frames
- a dataset-level HTML browser grouped by source folder

The main validation target is the primary tracked object whose
``source_object_id`` matches the benchmark sample ``object_id``.
"""

import argparse
import html
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import imageio.v2 as imageio
import numpy as np

from utils_io import ensure_dir, make_json_safe, save_video, to_uint8_rgb, write_json


@dataclass
class ValidationRecord:
    sample_id: str
    scene_id: str
    object_id: str
    rel_dir: str
    source_group: str
    composition: str
    count_bucket: str
    interaction_pattern: str
    motion_category: str
    primary_role: str
    primary_label: str
    num_objects: int
    object_summary: str
    num_frames: int
    visible_frames: int
    collision_frames: list[int]
    collision_summary: str
    frame_size: list[int]
    fps: int
    overlay_video: str
    collision_strip: str
    saved_full_video: str
    saved_future_video: str
    meta_json: str
    warnings: list[str]
    state_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate saved Genesis rigid benchmark samples by overlaying GT states on saved RGB videos."
    )
    parser.add_argument("--dataset_root", type=str, required=True, help="Saved benchmark root, for example .../mytest")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where generated browser assets are written. Defaults to <dataset_root>/state_validation",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing per-sample overlay assets when possible.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples after sorting. 0 means all.")
    parser.add_argument("--serve", action="store_true", help="Serve the generated browser on a local HTTP port.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8043)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)


def choose_primary_object_index(heldout_meta: dict[str, Any], source_meta: dict[str, Any]) -> int:
    objects = source_meta.get("objects", [])
    wanted_object_id = str(heldout_meta.get("object_id", "")).strip()

    for idx, obj in enumerate(objects):
        if str(obj.get("source_object_id", "")).strip() == wanted_object_id:
            return idx

    for idx, obj in enumerate(objects):
        if str(obj.get("role", "")) == "target":
            return idx

    return 0


def compute_track_state(anchor: dict[str, np.ndarray], object_index: int) -> dict[str, np.ndarray]:
    com_uv = np.asarray(anchor["com_uv"][:, object_index], dtype=np.float32)
    bbox_xyxy = np.asarray(anchor["bbox_xyxy"][:, object_index], dtype=np.float32)
    visible = np.asarray(anchor["visibility_mask"][:, object_index] > 0, dtype=bool)
    depth = np.asarray(anchor["center_depth"][:, object_index], dtype=np.float32)

    width = np.where(visible, np.maximum(0.0, bbox_xyxy[:, 2] - bbox_xyxy[:, 0]), 0.0).astype(np.float32)
    height = np.where(visible, np.maximum(0.0, bbox_xyxy[:, 3] - bbox_xyxy[:, 1]), 0.0).astype(np.float32)

    u = com_uv[:, 0].astype(np.float32)
    v = com_uv[:, 1].astype(np.float32)
    u[~visible] = 0.0
    v[~visible] = 0.0
    depth = np.where(visible, depth, 0.0).astype(np.float32)

    du = finite_difference(u, visible)
    dv = finite_difference(v, visible)
    dd = finite_difference(depth, visible)

    return {
        "u": u,
        "v": v,
        "d": depth,
        "w": width,
        "h": height,
        "du": du,
        "dv": dv,
        "dd": dd,
        "vis": visible.astype(np.uint8),
        "bbox_xyxy": bbox_xyxy.astype(np.float32),
    }


def object_palette_rgb(object_index: int) -> list[int]:
    palette = [
        [231, 76, 60],
        [52, 152, 219],
        [46, 204, 113],
        [241, 196, 15],
        [155, 89, 182],
        [230, 126, 34],
        [26, 188, 156],
        [236, 112, 99],
    ]
    return palette[object_index % len(palette)]


def object_display_label(obj: dict[str, Any], object_index: int) -> str:
    role = str(obj.get("role", "")).strip()
    source_id = str(obj.get("source_object_id", f"obj{object_index}")).strip()
    if role:
        return f"{role}:{source_id}"
    return source_id


def summarize_objects(objects: list[dict[str, Any]]) -> str:
    parts = []
    for idx, obj in enumerate(objects):
        parts.append(object_display_label(obj, idx))
    return " | ".join(parts)


def finite_difference(values: np.ndarray, visible: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    vis = np.asarray(visible, dtype=bool)
    out = np.zeros_like(arr, dtype=np.float32)
    num_frames = arr.shape[0]
    for frame_idx in range(num_frames):
        if not vis[frame_idx]:
            continue
        next_idx = frame_idx + 1
        prev_idx = frame_idx - 1
        if next_idx < num_frames and vis[next_idx]:
            out[frame_idx] = arr[next_idx] - arr[frame_idx]
        elif prev_idx >= 0 and vis[prev_idx]:
            out[frame_idx] = arr[frame_idx] - arr[prev_idx]
    return out.astype(np.float32)


def depth_color(depth: float, lo: float, hi: float, visible: bool) -> tuple[int, int, int]:
    if not visible or depth <= 0:
        return (164, 164, 164)
    hi = max(float(hi), float(lo) + 1e-6)
    t = float(np.clip((depth - lo) / (hi - lo), 0.0, 1.0))
    near_rgb = np.array([40, 167, 255], dtype=np.float32)
    far_rgb = np.array([255, 132, 43], dtype=np.float32)
    rgb = (1.0 - t) * near_rgb + t * far_rgb
    return tuple(int(v) for v in rgb.tolist())


def frame_collision_tags(collision_windows: list[dict[str, Any]]) -> dict[int, list[str]]:
    tags: dict[int, list[str]] = defaultdict(list)
    for item in collision_windows:
        label = str(item.get("label", "collision"))
        start_frame = int(item.get("start_frame", -1))
        peak_frame = int(item.get("peak_frame", -1))
        end_frame = int(item.get("end_frame", -1))
        if start_frame >= 0:
            tags[start_frame].append(f"start {label}")
        if peak_frame >= 0:
            tags[peak_frame].append(f"peak {label}")
        if end_frame >= 0 and end_frame not in {start_frame, peak_frame}:
            tags[end_frame].append(f"end {label}")
    return dict(tags)


def label_for_participants(item: dict[str, Any], source_meta: dict[str, Any]) -> str:
    objects = source_meta.get("objects", [])
    index_list = item.get("object_indices", item.get("participant_indices", item.get("participants", [])))
    names: list[str] = []
    for raw_idx in index_list:
        idx = int(raw_idx)
        if idx == -1:
            names.append(str(item.get("environment_name", "environment")))
            continue
        if 0 <= idx < len(objects):
            obj = objects[idx]
            source_id = str(obj.get("source_object_id", f"obj{idx}"))
            role = str(obj.get("role", ""))
            if role:
                names.append(f"{role}:{source_id}")
            else:
                names.append(source_id)
        else:
            names.append(f"obj{idx}")
    return " <-> ".join(names) if names else "collision"


def normalize_collision_item(item: dict[str, Any], source_meta: dict[str, Any]) -> dict[str, Any]:
    participants = item.get("object_indices", item.get("participant_indices", item.get("participants", [])))
    participants = [int(v) for v in participants]
    return {
        "event_id": int(item.get("event_id", item.get("window_id", -1))),
        "window_type": str(item.get("window_type", "")),
        "participant_indices": participants,
        "is_environment": -1 in participants or "environment_name" in item,
        "environment_name": str(item.get("environment_name", "")),
        "start_frame": int(item.get("start_frame", item.get("frame_idx", -1))),
        "peak_frame": int(item.get("peak_frame", item.get("frame_idx", item.get("start_frame", -1)))),
        "end_frame": int(item.get("end_frame", item.get("peak_frame", item.get("frame_idx", item.get("start_frame", -1))))),
        "label": label_for_participants(item, source_meta),
    }


def collect_collision_windows(
    source_meta: dict[str, Any],
    collision_events: list[dict[str, Any]],
    event_windows: list[dict[str, Any]],
    primary_index: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in event_windows:
        participants = item.get("object_indices", item.get("participant_indices", item.get("participants", [])))
        participants = [int(v) for v in participants]
        if primary_index not in participants:
            continue
        normalized.append(
            normalize_collision_item(item, source_meta)
        )

    if not normalized:
        for item in collision_events:
            participants = item.get("object_indices", item.get("participant_indices", item.get("participants", [])))
            participants = [int(v) for v in participants]
            if primary_index not in participants:
                continue
            start_frame = int(item.get("start_frame", item.get("frame_idx", -1)))
            peak_frame = int(item.get("peak_frame", item.get("frame_idx", start_frame)))
            end_frame = int(item.get("end_frame", peak_frame))
            normalized.append(
                normalize_collision_item(
                    {
                        **item,
                        "start_frame": start_frame,
                        "peak_frame": peak_frame,
                        "end_frame": end_frame,
                    },
                    source_meta,
                )
            )

    non_environment = [item for item in normalized if not item["is_environment"]]
    if non_environment:
        return dedupe_collision_windows(non_environment)

    meaningful_environment = [item for item in normalized if int(item["start_frame"]) > 0]
    return dedupe_collision_windows(meaningful_environment)


def collect_all_collision_windows(
    source_meta: dict[str, Any],
    collision_events: list[dict[str, Any]],
    event_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if event_windows:
        normalized.extend(normalize_collision_item(item, source_meta) for item in event_windows)
    else:
        normalized.extend(normalize_collision_item(item, source_meta) for item in collision_events)
    return dedupe_collision_windows(normalized)


def frame_collision_records(collision_windows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in collision_windows:
        label = str(item.get("label", "collision"))
        event_id = int(item.get("event_id", -1))
        start_frame = int(item.get("start_frame", -1))
        peak_frame = int(item.get("peak_frame", -1))
        end_frame = int(item.get("end_frame", -1))
        base = {
            "event_id": event_id,
            "label": label,
            "participant_indices": [int(v) for v in item.get("participant_indices", [])],
            "is_environment": bool(item.get("is_environment", False)),
            "environment_name": str(item.get("environment_name", "")),
        }
        if start_frame >= 0:
            records[start_frame].append({**base, "phase": "start"})
        if peak_frame >= 0:
            records[peak_frame].append({**base, "phase": "peak"})
        if end_frame >= 0 and end_frame not in {start_frame, peak_frame}:
            records[end_frame].append({**base, "phase": "end"})
    return dict(records)


def dedupe_collision_windows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: (x["start_frame"], x["peak_frame"], x["end_frame"], x["label"])):
        key = (int(item["start_frame"]), int(item["peak_frame"]), int(item["end_frame"]), str(item["label"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def summarize_collision_windows(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    parts = []
    for item in items:
        parts.append(
            f"{item['label']} [s={int(item['start_frame'])}, p={int(item['peak_frame'])}, e={int(item['end_frame'])}]"
        )
    return " | ".join(parts)


def draw_banner(frame: np.ndarray, lines: list[str], bg_color: tuple[int, int, int], text_color: tuple[int, int, int]) -> None:
    if not lines:
        return
    line_height = 24
    pad = 12
    box_h = pad * 2 + line_height * len(lines)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], box_h), bg_color, thickness=-1)
    for idx, line in enumerate(lines):
        y = pad + 18 + idx * line_height
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 2, cv2.LINE_AA)


def draw_state_overlay(
    rgb_frame: np.ndarray,
    frame_idx: int,
    state: dict[str, np.ndarray],
    depth_range: tuple[float, float],
    collision_tags: dict[int, list[str]],
) -> np.ndarray:
    frame = np.ascontiguousarray(rgb_frame.copy())
    vis = bool(int(state["vis"][frame_idx]))
    u = float(state["u"][frame_idx])
    v = float(state["v"][frame_idx])
    d = float(state["d"][frame_idx])
    du = float(state["du"][frame_idx])
    dv = float(state["dv"][frame_idx])
    dd = float(state["dd"][frame_idx])
    x1, y1, x2, y2 = [float(vv) for vv in state["bbox_xyxy"][frame_idx]]

    if vis:
        color = depth_color(d, depth_range[0], depth_range[1], visible=True)
        center = (int(round(u)), int(round(v)))
        tl = (int(round(x1)), int(round(y1)))
        br = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, tl, br, color, thickness=3)
        cv2.circle(frame, center, 6, color, thickness=-1)
        arrow_end = (int(round(u + du)), int(round(v + dv)))
        cv2.arrowedLine(frame, center, arrow_end, (255, 255, 255), thickness=3, tipLength=0.25)
        cv2.arrowedLine(frame, center, arrow_end, color, thickness=2, tipLength=0.25)
    else:
        color = (160, 160, 160)

    top_lines = [
        f"frame={frame_idx:02d} vis={int(vis)}  u={u:.1f} v={v:.1f} d={d:.3f}",
        f"bbox(w,h)=({float(state['w'][frame_idx]):.1f}, {float(state['h'][frame_idx]):.1f})  vel(du,dv,dd)=({du:.2f}, {dv:.2f}, {dd:.3f})",
    ]
    if frame_idx in collision_tags:
        top_lines.append("collision: " + " | ".join(collision_tags[frame_idx]))
        draw_banner(frame, top_lines, bg_color=(60, 30, 20), text_color=(245, 245, 245))
    else:
        draw_banner(frame, top_lines, bg_color=(18, 18, 18), text_color=(242, 242, 242))

    chip_color = tuple(int(vv) for vv in color)
    cv2.rectangle(frame, (16, frame.shape[0] - 64), (60, frame.shape[0] - 20), chip_color, thickness=-1)
    cv2.rectangle(frame, (16, frame.shape[0] - 64), (60, frame.shape[0] - 20), (255, 255, 255), thickness=2)
    cv2.putText(
        frame,
        "depth",
        (72, frame.shape[0] - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )
    return frame


def save_collision_strip(
    out_path: Path,
    overlay_frames: list[np.ndarray],
    selected_frames: list[int],
) -> None:
    ensure_dir(out_path.parent)
    if not selected_frames:
        canvas = np.full((180, 720, 3), 28, dtype=np.uint8)
        cv2.putText(canvas, "No primary collision frames selected", (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
        imageio.imwrite(out_path, canvas)
        return

    tiles = [overlay_frames[idx] for idx in selected_frames]
    max_cols = min(4, len(tiles))
    tile_h = 220
    tile_w = int(round(tiles[0].shape[1] * (tile_h / tiles[0].shape[0])))
    rows = int(np.ceil(len(tiles) / max_cols))
    canvas = np.full((rows * tile_h, max_cols * tile_w, 3), 18, dtype=np.uint8)

    for tile_idx, (frame_idx, tile) in enumerate(zip(selected_frames, tiles)):
        row = tile_idx // max_cols
        col = tile_idx % max_cols
        resized = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = resized
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (255, 255, 255), thickness=2)
        cv2.putText(canvas, f"f{frame_idx:02d}", (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    imageio.imwrite(out_path, canvas)


def save_browser_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Prefer browser-friendly H.264 output and fall back to the generic saver."""
    ensure_dir(path.parent)
    frame_list = [to_uint8_rgb(frame) for frame in frames]
    if not frame_list:
        raise ValueError("Cannot save an empty video.")

    try:
        imageio.mimwrite(
            path,
            frame_list,
            fps=int(fps),
            codec="libx264",
            quality=8,
            ffmpeg_log_level="error",
            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        return
    except Exception:
        pass

    save_video(path, frame_list, fps=int(fps))


def run_validation_for_sample(sample_dir: Path, output_root: Path, skip_existing: bool) -> ValidationRecord:
    heldout_meta = load_json(sample_dir / "meta.json")
    source_dir = Path(heldout_meta["source_paths"]["source_sample_dir"])
    source_meta = load_json(source_dir / "metadata.json")

    source_group = source_dir.parent.relative_to(source_dir.parents[2]).as_posix()
    source_group = f"rigid/{source_group}" if not source_group.startswith("rigid/") else source_group
    primary_index = choose_primary_object_index(heldout_meta, source_meta)

    sample_output_dir = output_root / sanitize_name(heldout_meta["sample_id"])
    ensure_dir(sample_output_dir)
    overlay_path = sample_output_dir / "overlay.mp4"
    collision_strip_path = sample_output_dir / "collision_strip.jpg"
    state_json_path = sample_output_dir / "state.json"

    anchor_npz = np.load(source_dir / "physics" / "anchor_targets.npz")
    anchor = {key: anchor_npz[key] for key in anchor_npz.files}
    collision_events = load_json(source_dir / "physics" / "collision_events.json")
    event_windows = load_json(source_dir / "physics" / "event_windows.json")

    state = compute_track_state(anchor, primary_index)
    primary_obj = source_meta.get("objects", [])[primary_index]
    primary_label = str(primary_obj.get("source_object_id", f"obj{primary_index}"))
    primary_role = str(primary_obj.get("role", ""))
    objects = list(source_meta.get("objects", []))

    collision_windows = collect_collision_windows(source_meta, collision_events, event_windows, primary_index)
    collision_tags = frame_collision_tags(collision_windows)
    collision_frames = sorted(collision_tags.keys())
    all_collision_windows = collect_all_collision_windows(source_meta, collision_events, event_windows)
    all_collision_records = frame_collision_records(all_collision_windows)

    saved_full_video_path = sample_dir / "full_video.mp4"
    saved_future_video_path = sample_dir / "future_gt_video.mp4"
    frames = load_video_frames(saved_full_video_path)
    height, width = frames[0].shape[:2]

    warnings: list[str] = []
    expected_frames = int(source_meta.get("frames", len(frames)))
    if len(frames) != expected_frames:
        warnings.append(f"frame_count_mismatch video={len(frames)} source={expected_frames}")
    if state["u"].shape[0] != len(frames):
        warnings.append(f"state_length_mismatch state={state['u'].shape[0]} video={len(frames)}")
    expected_resolution = source_meta.get("resolution", [width, height])
    if [width, height] != [int(expected_resolution[0]), int(expected_resolution[1])]:
        warnings.append(f"resolution_mismatch video={[width, height]} source={expected_resolution}")

    visible = state["vis"] > 0
    for frame_idx in range(min(len(frames), state["vis"].shape[0])):
        if not bool(visible[frame_idx]):
            continue
        u = float(state["u"][frame_idx])
        v = float(state["v"][frame_idx])
        x1, y1, x2, y2 = [float(vv) for vv in state["bbox_xyxy"][frame_idx]]
        if not (0.0 <= u < width and 0.0 <= v < height):
            warnings.append(f"center_out_of_frame@{frame_idx}")
            break
        if x2 < x1 or y2 < y1:
            warnings.append(f"invalid_bbox@{frame_idx}")
            break
        if x1 < -2 or y1 < -2 or x2 > width + 2 or y2 > height + 2:
            warnings.append(f"bbox_out_of_frame@{frame_idx}")
            break

    depth_values = state["d"][visible]
    if depth_values.size > 0:
        depth_range = (float(np.min(depth_values)), float(np.max(depth_values)))
    else:
        depth_range = (0.0, 1.0)

    all_object_payloads: list[dict[str, Any]] = []
    for object_index, obj in enumerate(objects):
        object_state = compute_track_state(anchor, object_index)
        object_visible = object_state["vis"] > 0
        object_depth = object_state["d"][object_visible]
        if object_depth.size > 0:
            object_depth_range = [float(np.min(object_depth)), float(np.max(object_depth))]
        else:
            object_depth_range = [0.0, 1.0]
        all_object_payloads.append(
            {
                "object_index": int(object_index),
                "object_id": int(obj.get("object_id", object_index)),
                "seg_id": int(obj.get("seg_id", -1)),
                "role": str(obj.get("role", "")),
                "source_object_id": str(obj.get("source_object_id", f"obj{object_index}")),
                "label": object_display_label(obj, object_index),
                "color_rgb": object_palette_rgb(object_index),
                "depth_range": object_depth_range,
                "frames": [
                    {
                        "frame_idx": int(frame_idx),
                        "u": float(object_state["u"][frame_idx]),
                        "v": float(object_state["v"][frame_idx]),
                        "d": float(object_state["d"][frame_idx]),
                        "w": float(object_state["w"][frame_idx]),
                        "h": float(object_state["h"][frame_idx]),
                        "du": float(object_state["du"][frame_idx]),
                        "dv": float(object_state["dv"][frame_idx]),
                        "dd": float(object_state["dd"][frame_idx]),
                        "vis": int(object_state["vis"][frame_idx]),
                        "bbox_xyxy": [float(vv) for vv in object_state["bbox_xyxy"][frame_idx].tolist()],
                    }
                    for frame_idx in range(int(min(len(frames), object_state["u"].shape[0])))
                ],
            }
        )

    if not (skip_existing and overlay_path.exists() and collision_strip_path.exists()):
        max_frames = min(len(frames), state["u"].shape[0])
        overlay_frames = [
            draw_state_overlay(frames[frame_idx], frame_idx, state, depth_range, collision_tags)
            for frame_idx in range(max_frames)
        ]
        save_browser_video(overlay_path, overlay_frames, fps=int(heldout_meta.get("fps", source_meta.get("fps", 12))))
        save_collision_strip(collision_strip_path, overlay_frames, collision_frames)

    state_payload = {
        "sample_id": heldout_meta["sample_id"],
        "primary_object_index": int(primary_index),
        "primary_role": primary_role,
        "primary_label": primary_label,
        "source_group": source_group,
        "fps": int(heldout_meta.get("fps", source_meta.get("fps", 12))),
        "depth_range": list(depth_range),
        "collision_windows": collision_windows,
        "all_collision_windows": all_collision_windows,
        "frame_collision_records_all": {
            str(frame_idx): records for frame_idx, records in sorted(all_collision_records.items())
        },
        "warnings": warnings,
        "frames": [
            {
                "frame_idx": int(frame_idx),
                "u": float(state["u"][frame_idx]),
                "v": float(state["v"][frame_idx]),
                "d": float(state["d"][frame_idx]),
                "w": float(state["w"][frame_idx]),
                "h": float(state["h"][frame_idx]),
                "du": float(state["du"][frame_idx]),
                "dv": float(state["dv"][frame_idx]),
                "dd": float(state["dd"][frame_idx]),
                "vis": int(state["vis"][frame_idx]),
                "bbox_xyxy": [float(vv) for vv in state["bbox_xyxy"][frame_idx].tolist()],
                "collision_tags": collision_tags.get(frame_idx, []),
            }
            for frame_idx in range(int(min(len(frames), state["u"].shape[0])))
        ],
        "objects": all_object_payloads,
        "object_summary": summarize_objects(objects),
    }
    write_json(state_json_path, state_payload)

    return ValidationRecord(
        sample_id=str(heldout_meta["sample_id"]),
        scene_id=str(heldout_meta.get("scene_id", sample_dir.name)),
        object_id=str(heldout_meta.get("object_id", "")),
        rel_dir=relative_to_root(sample_dir, sample_dir.parent),
        source_group=source_group,
        composition=str(heldout_meta.get("scene_composition", "")),
        count_bucket=str(source_dir.parent.name),
        interaction_pattern=str(heldout_meta.get("interaction_pattern", "")),
        motion_category=str(source_meta.get("motion_category", "")),
        primary_role=primary_role,
        primary_label=primary_label,
        num_objects=len(objects),
        object_summary=summarize_objects(objects),
        num_frames=int(min(len(frames), state["u"].shape[0])),
        visible_frames=int(np.sum(state["vis"] > 0)),
        collision_frames=collision_frames,
        collision_summary=summarize_collision_windows(collision_windows),
        frame_size=[width, height],
        fps=int(heldout_meta.get("fps", source_meta.get("fps", 12))),
        overlay_video=relative_to_root(overlay_path, sample_dir.parent),
        collision_strip=relative_to_root(collision_strip_path, sample_dir.parent),
        saved_full_video=relative_to_root(saved_full_video_path, sample_dir.parent),
        saved_future_video=relative_to_root(saved_future_video_path, sample_dir.parent),
        meta_json=relative_to_root(sample_dir / "meta.json", sample_dir.parent),
        warnings=warnings,
        state_json=relative_to_root(state_json_path, sample_dir.parent),
    )


def build_index(records: list[ValidationRecord], dataset_root: Path, output_root: Path) -> Path:
    payload = [make_json_safe(record.__dict__) for record in records]
    records_json = (
        json.dumps(payload, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Saved GT State Validator</title>
  <style>
    :root {{
      --bg: #f1eadf;
      --panel: rgba(255, 250, 244, 0.96);
      --panel-soft: rgba(255, 255, 255, 0.56);
      --ink: #1e1913;
      --muted: #6c6255;
      --accent: #ad4f1f;
      --accent-soft: rgba(173, 79, 31, 0.1);
      --accent-strong: #7f3613;
      --border: rgba(30, 25, 19, 0.1);
      --shadow: 0 18px 44px rgba(48, 34, 21, 0.1);
      --warn: #8d2323;
      --ok: #2d6b43;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(173, 79, 31, 0.12), transparent 23rem),
        radial-gradient(circle at top right, rgba(53, 107, 67, 0.1), transparent 24rem),
        linear-gradient(180deg, #f9f6f1 0%, var(--bg) 44%, #e9dece 100%);
    }}
    .shell {{
      width: min(1680px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 22px 0 40px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(340px, 0.8fr);
      gap: 18px;
      padding: 24px;
      border-radius: 28px;
      border: 1px solid var(--border);
      background: linear-gradient(135deg, rgba(255, 251, 246, 0.97), rgba(247, 236, 222, 0.88));
      box-shadow: var(--shadow);
    }}
    .hero-copy {{
      min-width: 0;
    }}
    .hero-side {{
      display: grid;
      gap: 14px;
      align-content: start;
    }}
    .control-panel,
    .dataset-panel {{
      padding: 18px;
      border-radius: 22px;
      border: 1px solid var(--border);
      background: var(--panel-soft);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 3vw, 3.2rem);
      letter-spacing: 0.02em;
    }}
    .sub {{
      margin-top: 12px;
      color: var(--muted);
      line-height: 1.65;
    }}
    .sub code {{
      display: inline;
      padding: 0.12rem 0.34rem;
    }}
    .panel-title {{
      margin: 0 0 12px;
      font-size: 0.95rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--accent-strong);
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .toolbar .search {{
      grid-column: 1 / -1;
    }}
    input, select {{
      width: 100%;
      padding: 13px 15px;
      border-radius: 14px;
      border: 1px solid var(--border);
      font: inherit;
      color: var(--ink);
      background: rgba(255,255,255,0.78);
    }}
    .stats {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .pill {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 14px;
      background: rgba(255,255,255,0.72);
      color: var(--muted);
    }}
    .groups {{
      display: grid;
      gap: 20px;
      margin-top: 24px;
    }}
    .group {{
      padding: 18px;
      border-radius: 26px;
      background: rgba(255,255,255,0.42);
      border: 1px solid var(--border);
      box-shadow: 0 8px 26px rgba(48, 34, 21, 0.06);
    }}
    .group-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .group h2 {{
      margin: 0;
      font-size: 1.16rem;
      line-height: 1.35;
    }}
    .group-sub {{
      margin-top: 5px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .group-pills {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(560px, 1fr));
      gap: 18px;
    }}
    .card {{
      padding: 16px;
      border-radius: 24px;
      border: 1px solid var(--border);
      background: var(--panel);
      box-shadow: 0 10px 26px rgba(48, 34, 21, 0.1);
      display: grid;
      gap: 14px;
    }}
    .card.warn {{
      border-color: rgba(141, 35, 35, 0.24);
      box-shadow: 0 10px 26px rgba(141, 35, 35, 0.12);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }}
    .card-title {{
      min-width: 0;
    }}
    .card h3 {{
      margin: 0;
      font-size: 1.08rem;
      line-height: 1.35;
    }}
    .card-scene {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.94rem;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      max-width: 45%;
    }}
    .chip {{
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid rgba(181, 77, 24, 0.18);
    }}
    .chip.warn {{
      background: rgba(141, 35, 35, 0.08);
      color: var(--warn);
      border-color: rgba(141, 35, 35, 0.22);
    }}
    .card-body {{
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
      gap: 14px;
      align-items: start;
    }}
    .media-col,
    .info-col {{
      display: grid;
      gap: 12px;
      min-width: 0;
    }}
    .media-panel,
    .info-panel {{
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.58);
      padding: 12px;
    }}
    .panel-kicker {{
      margin: 0 0 8px;
      color: var(--accent-strong);
      font-size: 0.8rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .player {{
      position: relative;
      border-radius: 16px;
      overflow: hidden;
      background: #101010;
      aspect-ratio: 4 / 3;
    }}
    video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      border-radius: 16px;
      background: #101010;
      display: block;
    }}
    .overlay-canvas {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }}
    img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.7);
      display: block;
    }}
    .summary-box {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(173, 79, 31, 0.06);
      border: 1px solid rgba(173, 79, 31, 0.12);
      color: var(--ink);
      line-height: 1.55;
    }}
    .summary-box.warn {{
      background: rgba(141, 35, 35, 0.06);
      border-color: rgba(141, 35, 35, 0.12);
    }}
    .legend-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--border);
      color: var(--ink);
      font-size: 0.82rem;
      line-height: 1.2;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(0,0,0,0.16);
      flex: 0 0 auto;
    }}
    .event-list {{
      display: grid;
      gap: 8px;
      max-height: 220px;
      overflow: auto;
    }}
    .event-groups {{
      display: grid;
      gap: 12px;
    }}
    .event-group {{
      display: grid;
      gap: 8px;
    }}
    .event-group-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-size: 0.84rem;
      color: var(--muted);
    }}
    .event-group-head strong {{
      color: var(--ink);
    }}
    .event-empty {{
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px dashed var(--border);
      color: var(--muted);
      background: rgba(255,255,255,0.42);
      font-size: 0.86rem;
    }}
    .event-item {{
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.45;
    }}
    .event-item strong {{
      color: var(--ink);
    }}
    .meta-grid {{
      display: grid;
      gap: 8px;
    }}
    .meta-row {{
      display: grid;
      grid-template-columns: 136px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .meta-row strong {{
      color: var(--ink);
    }}
    .meta-row span {{
      overflow-wrap: anywhere;
      line-height: 1.55;
    }}
    .links {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .links a {{
      text-decoration: none;
      color: var(--accent);
      background: rgba(181, 77, 24, 0.08);
      border: 1px solid rgba(181, 77, 24, 0.18);
      padding: 10px 12px;
      border-radius: 14px;
      font-size: 0.88rem;
      text-align: center;
    }}
    .empty {{
      margin-top: 20px;
      padding: 24px;
      border-radius: 20px;
      border: 1px dashed var(--border);
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.52);
    }}
    code {{
      display: block;
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(30, 25, 19, 0.05);
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.82rem;
      color: var(--ink);
    }}
    @media (max-width: 1200px) {{
      .hero {{
        grid-template-columns: 1fr;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .card-body {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 980px) {{
      .toolbar {{
        grid-template-columns: 1fr;
      }}
      .toolbar .search {{
        grid-column: auto;
      }}
      .group-head,
      .card-head {{
        flex-direction: column;
      }}
      .group-pills,
      .chip-row {{
        justify-content: flex-start;
        max-width: none;
      }}
      .meta-row {{
        grid-template-columns: 1fr;
        gap: 4px;
      }}
      .links {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        width: min(100vw - 18px, 1680px);
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-copy">
        <h1>Saved GT State Validator</h1>
        <p class="sub">
          该页面用于检查保存后的 benchmark 样本是否与源物理 GT 对齐。叠加视频直接画在保存后的
          <code>full_video.mp4</code> 上，验证 <code>u / v / bbox / depth / velocity</code> 是否发生坐标、
          尺度或帧序错位。主跟踪对象默认选择 <code>source_object_id == sample.object_id</code> 的物体。
        </p>
        <p class="sub">
          深度颜色使用主目标可见帧内的相对深度范围；速度箭头使用逐帧有限差分，单位是像素 / 帧。
          页面会同时拆开展示主对象碰撞、其他物体碰撞以及环境接触，避免把全量事件误看成只有主对象。
        </p>
        <div class="stats">
          <div class="pill" id="countPill">加载中</div>
          <div class="pill" id="groupPill"></div>
        </div>
      </div>
      <div class="hero-side">
        <div class="control-panel">
          <div class="panel-title">Filters</div>
          <div class="toolbar">
            <input id="search" class="search" type="search" placeholder="搜索 sample / scene / object / 路径 / 碰撞摘要">
            <select id="group">
              <option value="">全部文件夹</option>
            </select>
            <select id="composition">
              <option value="">全部 composition</option>
            </select>
            <select id="collision">
              <option value="">全部碰撞状态</option>
              <option value="yes">仅有碰撞帧</option>
              <option value="no">仅无碰撞帧</option>
            </select>
            <select id="warning">
              <option value="">全部校验状态</option>
              <option value="warn">仅有 warning</option>
              <option value="ok">仅无 warning</option>
            </select>
          </div>
        </div>
        <div class="dataset-panel">
          <div class="panel-title">Dataset</div>
          <div class="meta-grid">
            <div class="meta-row"><strong>Dataset Root</strong><span>{html.escape(str(dataset_root))}</span></div>
            <div class="meta-row"><strong>Assets Root</strong><span>{html.escape(str(output_root))}</span></div>
          </div>
        </div>
      </div>
    </section>
    <section id="groups" class="groups"></section>
    <section id="empty" class="empty" hidden>没有匹配结果。</section>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const state = {{
      items: [],
      filtered: [],
    }};

    const search = document.getElementById("search");
    const group = document.getElementById("group");
    const composition = document.getElementById("composition");
    const collision = document.getElementById("collision");
    const warning = document.getElementById("warning");
    const groups = document.getElementById("groups");
    const empty = document.getElementById("empty");
    const countPill = document.getElementById("countPill");
    const groupPill = document.getElementById("groupPill");

    function escapeHtml(text) {{
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function uniqSorted(values) {{
      return [...new Set(values.filter(Boolean))].sort();
    }}

    function fillSelect(selectEl, values) {{
      values.forEach((value) => {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        selectEl.appendChild(option);
      }});
    }}

    function formatCollisionFrames(item) {{
      return item.collision_frames.length
        ? item.collision_frames.map((v) => `f${{String(v).padStart(2, "0")}}`).join(", ")
        : "none";
    }}

    function render() {{
      countPill.textContent = `共 ${{state.items.length}} 条，当前匹配 ${{state.filtered.length}} 条`;
      const grouped = new Map();
      state.filtered.forEach((item) => {{
        if (!grouped.has(item.source_group)) {{
          grouped.set(item.source_group, []);
        }}
        grouped.get(item.source_group).push(item);
      }});
      groupPill.textContent = `当前覆盖 ${{grouped.size}} 个文件夹`;

      if (!state.filtered.length) {{
        groups.innerHTML = "";
        empty.hidden = false;
        return;
      }}

      empty.hidden = true;
      const sortedGroups = [...grouped.entries()].sort((a, b) => a[0].localeCompare(b[0]));
      groups.innerHTML = sortedGroups.map(([groupName, items]) => {{
        const warningCount = items.filter((item) => item.warnings.length > 0).length;
        const cards = items.map((item) => {{
          const chips = [
            item.composition,
            item.count_bucket,
            item.primary_role,
            item.primary_label,
          ].filter(Boolean).map((value) => `<span class="chip">${{escapeHtml(value)}}</span>`).join("");
          const warnChip = item.warnings.length
            ? `<span class="chip warn">${{escapeHtml(`warnings:${{item.warnings.length}}`)}}</span>`
            : `<span class="chip">${{escapeHtml("warnings:0")}}</span>`;
          const collisionText = formatCollisionFrames(item);
          return `
            <article class="card ${{item.warnings.length ? "warn" : ""}}">
              <div class="card-head">
                <div class="card-title">
                  <h3>${{escapeHtml(item.sample_id)}}</h3>
                  <div class="card-scene">${{escapeHtml(item.scene_id)}}</div>
                </div>
                <div class="chip-row">${{chips}}${{warnChip}}</div>
              </div>
              <div class="card-body">
                <div class="media-col">
                  <div class="media-panel">
                    <div class="panel-kicker">Interactive Overlay</div>
                    <div
                      class="player"
                      data-player="overlay"
                      data-state-json="${{encodeURI(item.state_json)}}"
                      data-num-frames="${{item.num_frames}}"
                      data-fps="${{item.fps}}"
                    >
                      <video controls preload="none" playsinline src="${{encodeURI(item.saved_full_video)}}"></video>
                      <canvas class="overlay-canvas"></canvas>
                    </div>
                  </div>
                  <div class="media-panel">
                    <div class="panel-kicker">Collision Frames</div>
                    <img src="${{encodeURI(item.collision_strip)}}" alt="collision strip">
                  </div>
                </div>
                <div class="info-col">
                  <div class="info-panel">
                    <div class="panel-kicker">Collision Summary</div>
                    <div class="summary-box ${{item.warnings.length ? "warn" : ""}}">${{escapeHtml(item.collision_summary)}}</div>
                  </div>
                  <div class="info-panel">
                    <div class="panel-kicker">Validation Stats</div>
                    <div class="meta-grid">
                      <div class="meta-row"><strong>Objects</strong><span>${{item.num_objects}}</span></div>
                      <div class="meta-row"><strong>Collision Frames</strong><span>${{escapeHtml(collisionText)}}</span></div>
                      <div class="meta-row"><strong>Frames / Visible</strong><span>${{item.num_frames}} / ${{item.visible_frames}}</span></div>
                      <div class="meta-row"><strong>Resolution</strong><span>${{item.frame_size[0]}} x ${{item.frame_size[1]}}</span></div>
                      <div class="meta-row"><strong>Warnings</strong><span>${{escapeHtml(item.warnings.join(" | ") || "none")}}</span></div>
                      <div class="meta-row"><strong>Saved Sample</strong><span><code>${{escapeHtml(item.rel_dir)}}</code></span></div>
                    </div>
                  </div>
                  <div class="info-panel">
                    <div class="panel-kicker">Object IDs</div>
                    <div>${{escapeHtml(item.object_summary)}}</div>
                  </div>
                  <div class="info-panel">
                    <div class="panel-kicker">Collision Breakdown</div>
                    <div class="event-list" data-collision-events="${{encodeURI(item.state_json)}}"></div>
                  </div>
                  <div class="info-panel">
                    <div class="panel-kicker">Open Assets</div>
                    <div class="links">
                      <a href="${{encodeURI(item.overlay_video)}}" target="_blank" rel="noreferrer">预生成 overlay.mp4</a>
                      <a href="${{encodeURI(item.saved_full_video)}}" target="_blank" rel="noreferrer">原始 full_video</a>
                      <a href="${{encodeURI(item.saved_future_video)}}" target="_blank" rel="noreferrer">future_gt_video</a>
                      <a href="${{encodeURI(item.meta_json)}}" target="_blank" rel="noreferrer">meta.json</a>
                      <a href="${{encodeURI(item.state_json)}}" target="_blank" rel="noreferrer">state.json</a>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          `;
        }}).join("");

        return `
          <section class="group">
            <div class="group-head">
              <div>
                <h2>${{escapeHtml(groupName)}}</h2>
                <div class="group-sub">按源文件夹聚合，便于检查同类生成 case 的一致性。</div>
              </div>
              <div class="group-pills">
                <div class="pill">${{items.length}} samples</div>
                <div class="pill">warnings ${{warningCount}}</div>
              </div>
            </div>
            <div class="grid">${{cards}}</div>
          </section>
        `;
      }}).join("");
      initOverlayPlayers();
    }}

    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      const groupValue = group.value;
      const compositionValue = composition.value;
      const collisionValue = collision.value;
      const warningValue = warning.value;

      state.filtered = state.items.filter((item) => {{
        const haystack = [
          item.sample_id,
          item.scene_id,
          item.object_id,
          item.rel_dir,
          item.source_group,
          item.primary_label,
          item.primary_role,
          item.collision_summary,
          ...(item.warnings || []),
        ].join(" ").toLowerCase();
        const textOk = !q || haystack.includes(q);
        const groupOk = !groupValue || item.source_group === groupValue;
        const compositionOk = !compositionValue || item.composition === compositionValue;
        const collisionOk = !collisionValue
          || (collisionValue === "yes" ? item.collision_frames.length > 0 : item.collision_frames.length === 0);
        const warningOk = !warningValue
          || (warningValue === "warn" ? item.warnings.length > 0 : item.warnings.length === 0);
        return textOk && groupOk && compositionOk && collisionOk && warningOk;
      }});
      render();
    }}

    function init() {{
      state.items = JSON.parse(document.getElementById("records").textContent || "[]");
      state.filtered = [...state.items];
      fillSelect(group, uniqSorted(state.items.map((item) => item.source_group)));
      fillSelect(composition, uniqSorted(state.items.map((item) => item.composition)));

      search.addEventListener("input", applyFilters);
      group.addEventListener("change", applyFilters);
      composition.addEventListener("change", applyFilters);
      collision.addEventListener("change", applyFilters);
      warning.addEventListener("change", applyFilters);
      render();
    }}

    const playerStateCache = new Map();

    function depthColor(depth, range, visible) {{
      if (!visible || !(depth > 0)) {{
        return "rgb(164,164,164)";
      }}
      const lo = range[0];
      const hi = Math.max(range[1], lo + 1e-6);
      const t = Math.max(0, Math.min(1, (depth - lo) / (hi - lo)));
      const near = [40, 167, 255];
      const far = [255, 132, 43];
      const rgb = near.map((v, idx) => Math.round((1 - t) * v + t * far[idx]));
      return `rgb(${{rgb[0]}}, ${{rgb[1]}}, ${{rgb[2]}})`;
    }}

    async function loadPlayerState(url) {{
      if (!playerStateCache.has(url)) {{
        playerStateCache.set(url, fetch(url).then((res) => {{
          if (!res.ok) {{
            throw new Error(`failed to load state json: ${{url}}`);
          }}
          return res.json();
        }}));
      }}
      return playerStateCache.get(url);
    }}

    function rgbFromArray(rgb) {{
      const arr = Array.isArray(rgb) ? rgb : [164, 164, 164];
      return `rgb(${{arr[0]}}, ${{arr[1]}}, ${{arr[2]}})`;
    }}

    function collisionParticipants(event) {{
      return Array.isArray(event && event.participant_indices)
        ? event.participant_indices.map((value) => Number(value))
        : [];
    }}

    function categorizeCollisionEvents(payload, events) {{
      const primaryIndex = Number(payload && payload.primary_object_index);
      const buckets = {{
        primaryObjectEvents: [],
        otherObjectEvents: [],
        environmentEvents: [],
      }};
      (Array.isArray(events) ? events : []).forEach((event) => {{
        const participants = collisionParticipants(event);
        if (event && event.is_environment) {{
          buckets.environmentEvents.push(event);
        }} else if (participants.includes(primaryIndex)) {{
          buckets.primaryObjectEvents.push(event);
        }} else {{
          buckets.otherObjectEvents.push(event);
        }}
      }});
      return buckets;
    }}

    function renderEventItems(events) {{
      if (!events.length) {{
        return '<div class="event-empty">none</div>';
      }}
      return events.map((event) => {{
        const env = event.environment_name ? ` env=${{event.environment_name}}` : '';
        const ids = collisionParticipants(event).join(', ');
        const kind = event.is_environment ? 'environment' : 'object-object';
        const typeText = event.window_type || kind;
        return `
          <div class="event-item">
            <div><strong>event_id=${{event.event_id}}</strong> <span>${{escapeHtml(typeText)}}</span></div>
            <div>${{escapeHtml(event.label)}}</div>
            <div>participant_indices=[${{escapeHtml(ids)}}]${{escapeHtml(env)}}</div>
            <div>frames: s=${{event.start_frame}}, p=${{event.peak_frame}}, e=${{event.end_frame}}</div>
          </div>
        `;
      }}).join("");
    }}

    function renderEventGroup(title, events) {{
      return `
        <section class="event-group">
          <div class="event-group-head">
            <strong>${{escapeHtml(title)}}</strong>
            <span>${{events.length}} events</span>
          </div>
          <div class="event-list">${{renderEventItems(events)}}</div>
        </section>
      `;
    }}

    function drawOverlayFrame(canvas, video, payload, fallbackFps) {{
      const ctx = canvas.getContext("2d");
      if (!ctx || !payload || !payload.objects || !payload.objects.length || !video.videoWidth || !video.videoHeight) {{
        return;
      }}

      const fps = Number(payload.fps || fallbackFps || 12);
      const frameIdx = Math.max(0, Math.floor(video.currentTime * fps + 1e-6));
      const primaryFrames = payload.frames || [];
      const primaryFrame = primaryFrames.length ? primaryFrames[Math.max(0, Math.min(primaryFrames.length - 1, frameIdx))] : null;
      const frameCollisionMap = payload.frame_collision_records_all || {{}};
      const allFrameRecords = frameCollisionMap[String(frameIdx)] || [];
      const groupedFrameRecords = categorizeCollisionEvents(payload, allFrameRecords);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();

      const scaleX = canvas.width / video.videoWidth;
      const scaleY = canvas.height / video.videoHeight;
      ctx.scale(scaleX, scaleY);

      payload.objects.forEach((obj) => {{
        const objFrames = obj.frames || [];
        if (!objFrames.length) {{
          return;
        }}
        const objFrame = objFrames[Math.max(0, Math.min(objFrames.length - 1, frameIdx))];
        if (!objFrame || !objFrame.vis) {{
          return;
        }}

        const color = rgbFromArray(obj.color_rgb);
        const [x1, y1, x2, y2] = objFrame.bbox_xyxy;
        ctx.strokeStyle = color;
        ctx.lineWidth = obj.object_index === payload.primary_object_index ? 3.5 : 2.5;
        ctx.strokeRect(x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1));

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(objFrame.u, objFrame.v, 5, 0, Math.PI * 2);
        ctx.fill();

        const ex = objFrame.u + objFrame.du;
        const ey = objFrame.v + objFrame.dv;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(objFrame.u, objFrame.v);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(objFrame.u, objFrame.v);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        const angle = Math.atan2(objFrame.dv, objFrame.du || 1e-6);
        const headLen = 8;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - headLen * Math.cos(angle - Math.PI / 6), ey - headLen * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - headLen * Math.cos(angle + Math.PI / 6), ey - headLen * Math.sin(angle + Math.PI / 6));
        ctx.stroke();

        const label = obj.label || `obj${{obj.object_index}}`;
        ctx.font = "14px sans-serif";
        const textW = ctx.measureText(label).width + 14;
        const tx = Math.max(2, x1);
        const ty = Math.max(20, y1 - 6);
        ctx.fillStyle = "rgba(18,18,18,0.72)";
        ctx.fillRect(tx, ty - 16, textW, 18);
        ctx.fillStyle = color;
        ctx.fillText(label, tx + 7, ty - 3);
      }});

      const lines = [];
      lines.push(`frame=${{String(Math.max(0, frameIdx)).padStart(2, "0")}} objects=${{payload.objects.length}} primary=${{payload.primary_label || "main"}}`);
      if (primaryFrame) {{
        lines.push(`primary vis=${{primaryFrame.vis}} u=${{primaryFrame.u.toFixed(1)}} v=${{primaryFrame.v.toFixed(1)}} d=${{primaryFrame.d.toFixed(3)}} vel=(${{primaryFrame.du.toFixed(2)}}, ${{primaryFrame.dv.toFixed(2)}}, ${{primaryFrame.dd.toFixed(3)}})`);
      }}
      if (primaryFrame && primaryFrame.collision_tags && primaryFrame.collision_tags.length) {{
        lines.push(`primary-collisions: ${{primaryFrame.collision_tags.join(" | ")}}`);
      }}
      if (groupedFrameRecords.otherObjectEvents.length) {{
        const detail = groupedFrameRecords.otherObjectEvents
          .map((rec) => `id=${{rec.event_id}} ${{rec.phase}} ${{rec.label}}`)
          .join(" | ");
        lines.push(`other-object-collisions: ${{detail}}`);
      }}
      if (groupedFrameRecords.environmentEvents.length) {{
        const detail = groupedFrameRecords.environmentEvents
          .map((rec) => `id=${{rec.event_id}} ${{rec.phase}} ${{rec.label}}`)
          .join(" | ");
        lines.push(`environment-contacts: ${{detail}}`);
      }}

      const bannerHeight = 16 + lines.length * 22;
      ctx.fillStyle = primaryFrame && primaryFrame.collision_tags && primaryFrame.collision_tags.length ? "rgba(60,30,20,0.9)" : "rgba(18,18,18,0.85)";
      ctx.fillRect(0, 0, video.videoWidth, bannerHeight);
      ctx.fillStyle = "#f3f3f3";
      ctx.font = "16px sans-serif";
      lines.forEach((line, idx) => {{
        ctx.fillText(line, 12, 22 + idx * 22);
      }});
      ctx.restore();
    }}

    function initOverlayPlayers() {{
      document.querySelectorAll('[data-player="overlay"]').forEach((root) => {{
        if (root.dataset.bound === "1") {{
          return;
        }}
        root.dataset.bound = "1";

        const video = root.querySelector("video");
        const canvas = root.querySelector("canvas");
        const panel = root.parentElement;
        const stateJsonUrl = root.dataset.stateJson;
        const cardBody = root.closest(".card-body");
        const collisionPanel = cardBody ? cardBody.querySelector(`[data-collision-events="${{CSS.escape(stateJsonUrl)}}"]`) : null;
        const fallbackFps = Number(root.dataset.fps || "12");
        let payload = null;
        let rafId = 0;

        function renderCollisionEvents() {{
          if (!collisionPanel || !payload) {{
            return;
          }}
          const items = payload.all_collision_windows || [];
          const grouped = categorizeCollisionEvents(payload, items);
          collisionPanel.innerHTML = `
            <div class="event-groups">
              ${{renderEventGroup("Primary Object Collisions", grouped.primaryObjectEvents)}}
              ${{renderEventGroup("Other Object Collisions", grouped.otherObjectEvents)}}
              ${{renderEventGroup("Environment Contacts", grouped.environmentEvents)}}
            </div>
          `;
        }}

        function renderLegend() {{
          if (!payload || !panel) {{
            return;
          }}
          const existing = panel.querySelector(".legend-list");
          if (existing) {{
            existing.remove();
          }}
          const legend = document.createElement("div");
          legend.className = "legend-list";
          (payload.objects || []).forEach((obj) => {{
            const chip = document.createElement("div");
            chip.className = "legend-chip";
            const swatch = document.createElement("span");
            swatch.className = "legend-swatch";
            swatch.style.background = rgbFromArray(obj.color_rgb);
            const text = document.createElement("span");
            text.textContent = obj.label || `obj${{obj.object_index}}`;
            chip.appendChild(swatch);
            chip.appendChild(text);
            legend.appendChild(chip);
          }});
          panel.appendChild(legend);
        }}

        function resizeCanvas() {{
          if (!video.videoWidth || !video.videoHeight) {{
            return;
          }}
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          drawOnce();
        }}

        function drawOnce() {{
          if (!payload) {{
            return;
          }}
          drawOverlayFrame(canvas, video, payload, fallbackFps);
        }}

        function loop() {{
          drawOnce();
          if (!video.paused && !video.ended) {{
            rafId = requestAnimationFrame(loop);
          }}
        }}

        loadPlayerState(stateJsonUrl)
          .then((obj) => {{
            payload = obj;
            renderLegend();
            renderCollisionEvents();
            drawOnce();
          }})
          .catch((err) => {{
            console.error(err);
          }});

        video.addEventListener("loadedmetadata", resizeCanvas);
        video.addEventListener("loadeddata", drawOnce);
        video.addEventListener("seeked", drawOnce);
        video.addEventListener("timeupdate", drawOnce);
        video.addEventListener("pause", drawOnce);
        video.addEventListener("play", () => {{
          cancelAnimationFrame(rafId);
          rafId = requestAnimationFrame(loop);
        }});
      }});
    }}

    try {{
      init();
    }} catch (err) {{
      console.error(err);
      countPill.textContent = "加载失败";
      groupPill.textContent = "";
      empty.hidden = false;
      empty.textContent = `页面初始化失败: ${{err}}`;
    }}
  </script>
</body>
</html>
"""
    out_path = dataset_root / "state_validation_browser.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


class BrowserHandler(SimpleHTTPRequestHandler):
    range_re = re.compile(r"bytes=(\d*)-(\d*)$")

    def __init__(self, *args, directory: str, index_name: str, **kwargs):
        self.index_name = str(index_name)
        self._range: tuple[int, int] | None = None
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path in {"", "/"}:
            self.send_response(302)
            self.send_header("Location", f"/{self.index_name}")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            return
        super().do_GET()

    def send_head(self):
        self._range = None
        request_path = urlsplit(self.path).path
        path = self.translate_path(request_path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None

        ctype = self.guess_type(path)
        f = open(path, "rb")
        try:
            fs = os.fstat(f.fileno())
            file_size = int(fs.st_size)
            start = 0
            end = max(0, file_size - 1)
            range_header = self.headers.get("Range")
            if range_header:
                match = self.range_re.match(range_header.strip())
                if match:
                    start_text, end_text = match.groups()
                    if start_text:
                        start = int(start_text)
                    if end_text:
                        end = int(end_text)
                    if not start_text and end_text:
                        length = int(end_text)
                        start = max(0, file_size - length)
                        end = max(0, file_size - 1)
                    start = max(0, min(start, max(0, file_size - 1)))
                    end = max(start, min(end, max(0, file_size - 1)))
                    self._range = (start, end)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(end - start + 1))
                else:
                    self.send_error(416, "Invalid Range")
                    f.close()
                    return None
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(file_size))

            self.send_header("Content-type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile) -> None:
        if self._range is None:
            shutil.copyfileobj(source, outputfile)
            return

        start, end = self._range
        source.seek(start)
        remaining = end - start + 1
        bufsize = 64 * 1024
        while remaining > 0:
            chunk = source.read(min(bufsize, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_dir).resolve() if args.output_dir else (dataset_root / "state_validation")
    ensure_dir(output_root)

    sample_dirs = sorted(path.parent for path in dataset_root.rglob("meta.json"))
    if int(args.limit) > 0:
        sample_dirs = sample_dirs[: int(args.limit)]
    if not sample_dirs:
        raise FileNotFoundError(f"No meta.json files found under {dataset_root}")

    records: list[ValidationRecord] = []
    total = len(sample_dirs)
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        print(f"[{idx}/{total}] validate {sample_dir.name}")
        record = run_validation_for_sample(sample_dir, output_root, skip_existing=bool(args.skip_existing))
        records.append(record)

    manifest_path = output_root / "validation_manifest.json"
    write_json(manifest_path, [record.__dict__ for record in records])
    index_path = build_index(records, dataset_root, output_root)
    print(f"[DONE] samples={len(records)}")
    print(f"[DONE] manifest={manifest_path}")
    print(f"[DONE] browser={index_path}")

    if not args.serve:
        return

    handler = partial(BrowserHandler, directory=str(dataset_root), index_name=index_path.name)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[INFO] browse: http://127.0.0.1:{args.port}/{index_path.name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] stopped server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
