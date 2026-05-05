#!/usr/bin/env python3
"""Build Genesis stage1adapter train packages from raw pre-collision clips."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from build_stage1_subsets import load_raw_state, normalize_state, resolve_main_object_index
from motion_complexity import infer_motion_complexity
from window_interactions import load_interaction_episodes


DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
)
STAGE1ADAPTER_ROOT = DATASET_ROOT / "stage1adapter"

RATIO_SPECS = {
    "ratio11": (1, 1),
    "ratio12": (1, 2),
}

SUPPORT_ENV_NAMES = {"ground", "floor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=STAGE1ADAPTER_ROOT)
    parser.add_argument("--source_split", type=str, default="train")
    parser.add_argument("--count_buckets", type=str, default="count_01")
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument("--ratios", type=str, default="ratio11,ratio12")
    parser.add_argument("--min_context_frames", type=int, default=2)
    parser.add_argument("--min_future_frames", type=int, default=2)
    parser.add_argument("--min_window_frames", type=int, default=10)
    parser.add_argument("--max_window_frames", type=int, default=12)
    parser.add_argument("--max_source_samples", type=int, default=0)
    parser.add_argument("--include_invalid_by_qa", action="store_true")
    parser.add_argument("--rebuild_genesis_train", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_source_meta_path(sample_dir: Path) -> Path:
    for name in ("metadata.json", "meta.json"):
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing metadata.json/meta.json under {sample_dir}")


def load_source_metadata(sample_dir: Path) -> tuple[dict[str, Any], Path]:
    meta_path = find_source_meta_path(sample_dir)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload, meta_path


def infer_prompt(sample_dir: Path, metadata: dict[str, Any]) -> str:
    caption_json_path = sample_dir / "caption.json"
    if caption_json_path.exists():
        try:
            payload = json.loads(caption_json_path.read_text(encoding="utf-8"))
            for key in ("simple_caption", "caption"):
                text = str(payload.get(key, "")).strip()
                if text:
                    return text
        except Exception:
            pass
    for name in ("caption_simple.txt", "caption.txt"):
        path = sample_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    motion = str(metadata.get("motion_category", "")).strip() or "unknown_motion"
    scene = str(metadata.get("scene_composition", "")).strip() or "rigid_scene"
    num_objects = int(metadata.get("num_objects", 0) or 0)
    return f"{scene} with {num_objects} object(s), motion={motion}."


def extract_object_text_meta(sample_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    objects = metadata.get("objects", [])
    merged: list[dict[str, Any]] = []
    if isinstance(objects, list):
        merged = [dict(obj) for obj in objects if isinstance(obj, dict)]

    caption_json_path = sample_dir / "caption.json"
    if not caption_json_path.exists():
        return merged
    try:
        payload = json.loads(caption_json_path.read_text(encoding="utf-8"))
    except Exception:
        return merged
    caption_objects = payload.get("objects", [])
    if not isinstance(caption_objects, list):
        return merged

    by_object_id = {
        int(obj["object_id"]): obj
        for obj in merged
        if isinstance(obj, dict) and obj.get("object_id") is not None
    }
    for obj in caption_objects:
        if not isinstance(obj, dict) or obj.get("object_id") is None:
            continue
        object_id = int(obj["object_id"])
        base = by_object_id.get(object_id, {"object_id": object_id})
        base.update({"name": obj.get("name"), "category": obj.get("category")})
        by_object_id[object_id] = base
    return [by_object_id[key] for key in sorted(by_object_id)]


def build_video(frames: list[np.ndarray], dst: Path, fps: float = 12.0) -> None:
    if not frames:
        return
    ensure_dir(dst.parent)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {dst}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def load_rgb_frames_by_indices(sample_dir: Path, frame_indices: Sequence[int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_idx in frame_indices:
        frame_path = sample_dir / "rgb" / f"frame_{int(frame_idx):03d}.png"
        with Image.open(frame_path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    return frames


def save_local_rgb_frames(out_dir: Path, frames: list[np.ndarray]) -> list[str]:
    rgb_dir = out_dir / "rgb"
    ensure_dir(rgb_dir)
    frame_paths: list[str] = []
    for idx, frame in enumerate(frames):
        frame_path = rgb_dir / f"frame_{idx:03d}.png"
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(str(frame_path))
    return frame_paths


def copy_frame_sequence_by_indices(src_dir: Path, dst_dir: Path, frame_indices: Sequence[int]) -> list[str]:
    if not src_dir.exists():
        return []
    ensure_dir(dst_dir)
    copied: list[str] = []
    for local_idx, orig_idx in enumerate(frame_indices):
        src = src_dir / f"frame_{int(orig_idx):03d}.png"
        if not src.exists():
            continue
        dst = dst_dir / f"frame_{local_idx:03d}.png"
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def load_depth_preview_frames_by_indices(sample_dir: Path, frame_indices: Sequence[int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    depth_dir = sample_dir / "depth"
    for frame_idx in frame_indices:
        frame_path = depth_dir / f"frame_{int(frame_idx):03d}.png"
        if not frame_path.exists():
            continue
        with Image.open(frame_path) as image:
            arr = np.asarray(image)
        if arr.ndim == 2:
            depth = arr.astype(np.float32)
            positive = depth[depth > 0]
            if positive.size > 0:
                lo = float(np.percentile(positive, 2.0))
                hi = float(np.percentile(positive, 98.0))
            else:
                lo = float(depth.min()) if depth.size else 0.0
                hi = float(depth.max()) if depth.size else lo
            if hi <= lo:
                vis = np.zeros_like(depth, dtype=np.uint8)
            else:
                vis = ((np.clip(depth, lo, hi) - lo) / (hi - lo) * 255.0).astype(np.uint8)
            rgb = np.stack([vis, vis, vis], axis=-1)
        else:
            rgb = np.asarray(Image.fromarray(arr).convert("RGB"), dtype=np.uint8)
        frames.append(rgb)
    return frames


def bbox_xyxy_from_state(state_raw: np.ndarray) -> np.ndarray:
    u = state_raw[..., 0]
    v = state_raw[..., 1]
    w = state_raw[..., 3]
    h = state_raw[..., 4]
    x1 = u - 0.5 * w
    y1 = v - 0.5 * h
    x2 = u + 0.5 * w
    y2 = v + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)


def write_local_physics_stub(out_dir: Path, full_state_raw: np.ndarray, object_ids: np.ndarray, seg_ids: np.ndarray) -> None:
    physics_dir = out_dir / "physics"
    ensure_dir(physics_dir)
    visibility_mask = (full_state_raw[..., -1] > 0.5).astype(np.uint8)
    np.savez_compressed(
        physics_dir / "anchor_targets.npz",
        object_ids=np.asarray(object_ids, dtype=np.int32),
        seg_ids=np.asarray(seg_ids, dtype=np.int32),
        com_uv=full_state_raw[..., 0:2].astype(np.float32),
        bbox_xyxy=bbox_xyxy_from_state(full_state_raw),
        visibility_mask=visibility_mask.astype(np.uint8),
        center_depth=full_state_raw[..., 2].astype(np.float32),
    )
    np.save(physics_dir / "state_9d.npy", full_state_raw.astype(np.float32))
    write_json(physics_dir / "event_windows.json", [])


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def collect_source_samples(source_root: Path, sample_filter: str, count_buckets: Sequence[str]) -> list[Path]:
    wanted_buckets = {bucket.strip() for bucket in count_buckets if bucket.strip()}
    seen: set[Path] = set()
    samples: list[Path] = []
    for name in ("metadata.json", "meta.json"):
        for meta_path in source_root.rglob(name):
            sample_dir = meta_path.parent
            if sample_dir in seen:
                continue
            if not (sample_dir / "physics" / "anchor_targets.npz").exists():
                continue
            if wanted_buckets and not any(bucket in sample_dir.parts for bucket in wanted_buckets):
                continue
            if sample_filter and sample_filter not in str(sample_dir):
                continue
            seen.add(sample_dir)
            samples.append(sample_dir)
    return sorted(samples)


def is_support_episode(episode: dict[str, Any]) -> bool:
    return (
        str(episode.get("kind", "")) == "object_environment"
        and int(episode.get("start_frame", -1)) == 0
        and str(episode.get("environment_name", "")).strip().lower() in SUPPORT_ENV_NAMES
    )


def first_meaningful_collision_frame(sample_dir: Path) -> int | None:
    episodes = load_interaction_episodes(sample_dir)
    immediate_collision = False
    candidates: list[int] = []
    for episode in episodes:
        start_frame = int(episode.get("start_frame", -1))
        if start_frame < 0:
            continue
        if is_support_episode(episode):
            continue
        if start_frame == 0:
            immediate_collision = True
            continue
        candidates.append(start_frame)
    if immediate_collision:
        return 0
    if not candidates:
        return None
    return min(candidates)


def choose_split_lengths(
    full_len: int,
    ratio_key: str,
    min_context_frames: int,
    min_future_frames: int,
) -> tuple[int, int] | None:
    if full_len <= 0:
        return None
    left, right = RATIO_SPECS[ratio_key]
    context_len = int(full_len * left // (left + right))
    future_len = int(full_len - context_len)
    if context_len < int(min_context_frames) or future_len < int(min_future_frames):
        return None
    return context_len, future_len


def choose_window_frame_indices(
    *,
    total_frames: int,
    source_collision_frame: int | None,
    ratio_key: str,
    min_context_frames: int,
    min_future_frames: int,
    min_window_frames: int,
    max_window_frames: int,
) -> tuple[list[int], list[int], list[int]] | None:
    end_exclusive = int(source_collision_frame) if source_collision_frame is not None else int(total_frames)
    min_total = int(min_context_frames) + int(min_future_frames)
    if end_exclusive < min_total:
        return None

    full_len = end_exclusive
    if int(max_window_frames) > 0:
        full_len = min(full_len, int(max_window_frames))
    min_required = max(min_total, int(min_window_frames))
    if full_len < min_required:
        return None

    split_lens = choose_split_lengths(
        full_len=full_len,
        ratio_key=ratio_key,
        min_context_frames=min_context_frames,
        min_future_frames=min_future_frames,
    )
    if split_lens is None:
        return None

    context_len, future_len = split_lens
    start_index = end_exclusive - full_len
    orig_full_indices = list(range(start_index, end_exclusive))
    if len(orig_full_indices) != full_len:
        return None
    orig_context_indices = orig_full_indices[:context_len]
    orig_future_indices = orig_full_indices[context_len : context_len + future_len]
    if len(orig_context_indices) < int(min_context_frames) or len(orig_future_indices) < int(min_future_frames):
        return None
    return orig_context_indices, orig_future_indices, orig_full_indices


def context_covers_all_objects(visibility_full: np.ndarray, orig_context_indices: Sequence[int]) -> bool:
    if not orig_context_indices:
        return False
    context_visibility = np.asarray(visibility_full[orig_context_indices], dtype=np.uint8)
    if context_visibility.ndim != 2 or context_visibility.shape[1] <= 0:
        return False
    seen_mask = context_visibility.max(axis=0) > 0
    return bool(np.all(seen_mask))


def collision_count_bucket(count: int) -> str:
    if int(count) <= 0:
        return "c0"
    if int(count) == 1:
        return "c1"
    return "c2plus"


def collision_type_bucket(obj_env_count: int, obj_obj_count: int) -> str:
    if int(obj_env_count) <= 0 and int(obj_obj_count) <= 0:
        return "none"
    if int(obj_env_count) > 0 and int(obj_obj_count) <= 0:
        return "env_only"
    if int(obj_env_count) <= 0 and int(obj_obj_count) > 0:
        return "obj_obj_only"
    return "mixed"


def summarize_local_event_window(events: Sequence[dict[str, Any]], frame_start: int, frame_end_exclusive: int) -> dict[str, Any]:
    overlapping: list[dict[str, Any]] = []
    for event in events:
        start_frame = int(event.get("start_frame", -1))
        end_frame = int(event.get("end_frame", start_frame))
        if max(frame_start, start_frame) <= min(frame_end_exclusive - 1, end_frame):
            overlapping.append(event)
    obj_env = [event for event in overlapping if str(event.get("kind", "")) == "object_environment"]
    obj_obj = [event for event in overlapping if str(event.get("kind", "")) == "object_object"]
    return {
        "frame_start": int(frame_start),
        "frame_end_exclusive": int(frame_end_exclusive),
        "collision_episode_count": int(len(overlapping)),
        "object_environment_count": int(len(obj_env)),
        "object_object_count": int(len(obj_obj)),
        "collision_type_bucket": collision_type_bucket(len(obj_env), len(obj_obj)),
        "collision_count_bucket": collision_count_bucket(len(overlapping)),
        "episodes": [
            {
                "kind": str(event.get("kind", "")),
                "participants": [int(x) for x in event.get("participants", [])],
                "object_indices": [int(x) for x in event.get("object_indices", [])],
                "environment_name": str(event.get("environment_name", "")),
                "window_type": str(event.get("window_type", "")),
                "start_frame": int(event.get("start_frame", 0)),
                "end_frame": int(event.get("end_frame", event.get("start_frame", 0))),
            }
            for event in overlapping
        ],
    }


def build_window_interactions_from_local_events(
    events: Sequence[dict[str, Any]],
    object_count: int,
    context_len: int,
    future_len: int,
) -> dict[str, Any]:
    full_end = int(context_len) + int(future_len)
    full_summary = summarize_local_event_window(events, 0, full_end)
    future_summary = summarize_local_event_window(events, int(context_len), full_end)
    return {
        "object_count": int(object_count),
        "full_window": full_summary,
        "future_window": future_summary,
        "future_bucket": f"obj{int(object_count)}__{future_summary['collision_count_bucket']}__{future_summary['collision_type_bucket']}",
        "source_event_episode_count": int(len(events)),
    }


def build_no_collision_interactions(object_count: int, start_index: int, context_len: int, future_len: int) -> dict[str, Any]:
    future_start = int(start_index) + int(context_len)
    future_end = future_start + int(future_len)
    full_end = future_end
    empty_window = lambda a, b: {
        "frame_start": int(a),
        "frame_end_exclusive": int(b),
        "collision_episode_count": 0,
        "object_environment_count": 0,
        "object_object_count": 0,
        "collision_type_bucket": "none",
        "collision_count_bucket": "c0",
        "collision_subtypes": [],
        "episodes": [],
    }
    return {
        "object_count": int(object_count),
        "full_window": empty_window(start_index, full_end),
        "future_window": empty_window(future_start, future_end),
        "future_bucket": f"obj{int(object_count)}__c0__none",
        "source_event_episode_count": 0,
    }


def trim_range_records(records: Sequence[dict[str, Any]], frame_indices: Sequence[int]) -> list[dict[str, Any]]:
    if not frame_indices:
        return []
    sorted_unique = sorted({int(idx) for idx in frame_indices})
    index_map = {orig: local for local, orig in enumerate(sorted_unique)}
    local_records: list[dict[str, Any]] = []
    for record in records:
        start_frame = int(record.get("start_frame", record.get("frame_idx", -1)))
        end_frame = int(record.get("end_frame", start_frame))
        overlapping = [orig for orig in sorted_unique if start_frame <= orig <= end_frame]
        explicit_hits = []
        for key in ("frame_idx", "peak_frame"):
            if key in record:
                value = int(record.get(key, -1))
                if value in index_map:
                    explicit_hits.append(value)
        relevant = sorted(set(overlapping + explicit_hits))
        if not relevant:
            continue
        local = dict(record)
        local_start = index_map[int(relevant[0])]
        local_end = index_map[int(relevant[-1])]
        if "start_frame" in local:
            local["start_frame"] = int(local_start)
        if "end_frame" in local:
            local["end_frame"] = int(local_end)
        if "frame_idx" in local:
            orig_frame = int(record.get("frame_idx", relevant[0]))
            local["frame_idx"] = int(index_map.get(orig_frame, local_start))
        if "peak_frame" in local:
            orig_peak = int(record.get("peak_frame", record.get("frame_idx", relevant[0])))
            if orig_peak in index_map:
                local["peak_frame"] = int(index_map[orig_peak])
            elif orig_peak < relevant[0]:
                local["peak_frame"] = int(local_start)
            else:
                local["peak_frame"] = int(local_end)
        local_records.append(local)
    return local_records


def save_trimmed_npy(src_path: Path, dst_path: Path, total_frames: int, frame_indices: Sequence[int]) -> bool:
    if not src_path.exists():
        return False
    data = np.load(src_path)
    trimmed = data[list(frame_indices)] if data.ndim > 0 and int(data.shape[0]) == int(total_frames) else data
    ensure_dir(dst_path.parent)
    np.save(dst_path, trimmed)
    return True


def save_trimmed_npz(src_path: Path, dst_path: Path, total_frames: int, frame_indices: Sequence[int]) -> bool:
    if not src_path.exists():
        return False
    payload = np.load(src_path)
    trimmed: dict[str, np.ndarray] = {}
    for key in payload.files:
        value = np.asarray(payload[key])
        if value.ndim > 0 and int(value.shape[0]) == int(total_frames):
            trimmed[key] = value[list(frame_indices)]
        else:
            trimmed[key] = value
    ensure_dir(dst_path.parent)
    np.savez_compressed(dst_path, **trimmed)
    return True


def copy_raw_like_window_assets(
    *,
    sample_dir: Path,
    out_dir: Path,
    orig_full_indices: Sequence[int],
    total_frames: int,
    fps: float,
    full_video_path: Path,
) -> dict[str, bool]:
    copied: dict[str, bool] = {}
    videos_dir = out_dir / "videos"
    ensure_dir(videos_dir)
    shutil.copy2(full_video_path, videos_dir / "rgb.mp4")
    copied["rgb_video"] = True

    depth_paths = copy_frame_sequence_by_indices(sample_dir / "depth", out_dir / "depth", orig_full_indices)
    copied["depth_frames"] = bool(depth_paths)
    if depth_paths:
        depth_frames = load_depth_preview_frames_by_indices(sample_dir, orig_full_indices)
        if depth_frames:
            build_video(depth_frames, videos_dir / "depth.mp4", fps=fps)
            copied["depth_video"] = True
            ensure_dir(out_dir / "visualizations")
            shutil.copy2(videos_dir / "depth.mp4", out_dir / "visualizations" / "depth_vis.mp4")
            copied["depth_visualization_video"] = True
        else:
            copied["depth_video"] = False
            copied["depth_visualization_video"] = False
    else:
        copied["depth_video"] = False
        copied["depth_visualization_video"] = False

    physics_src = sample_dir / "physics"
    physics_dst = out_dir / "physics"
    ensure_dir(physics_dst)
    copied["depth_metric"] = save_trimmed_npy(
        physics_src / "depth_metric.npy", physics_dst / "depth_metric.npy", total_frames, orig_full_indices
    )
    copied["segmentation"] = save_trimmed_npy(
        physics_src / "seg.npy", physics_dst / "seg.npy", total_frames, orig_full_indices
    )
    copied["contact_graph"] = save_trimmed_npy(
        physics_src / "contact_graph.npy", physics_dst / "contact_graph.npy", total_frames, orig_full_indices
    )
    copied["contact_impulse"] = save_trimmed_npy(
        physics_src / "contact_impulse.npy", physics_dst / "contact_impulse.npy", total_frames, orig_full_indices
    )
    copied["frame_phase"] = save_trimmed_npy(
        physics_src / "frame_phase.npy", physics_dst / "frame_phase.npy", total_frames, orig_full_indices
    )
    copied["rigid_kinematics"] = save_trimmed_npz(
        physics_src / "rigid_kinematics.npz", physics_dst / "rigid_kinematics.npz", total_frames, orig_full_indices
    )
    copied["energy"] = save_trimmed_npz(
        physics_src / "energy.npz", physics_dst / "energy.npz", total_frames, orig_full_indices
    )
    if (physics_src / "properties.json").exists():
        shutil.copy2(physics_src / "properties.json", physics_dst / "properties.json")
        copied["properties"] = True
    else:
        copied["properties"] = False
    if (sample_dir / "scene_input.json").exists():
        shutil.copy2(sample_dir / "scene_input.json", out_dir / "scene_input.json")
        copied["scene_input"] = True
    else:
        copied["scene_input"] = False
    return copied


def export_package(
    *,
    sample_dir: Path,
    out_dir: Path,
    output_root: Path,
    sample_id: str,
    sample_label: str,
    split_name: str,
    prompt: str,
    object_text_meta: list[dict[str, Any]],
    metadata: dict[str, Any],
    state_raw_local: np.ndarray,
    state_norm_local: np.ndarray,
    visibility_local: np.ndarray,
    object_ids: np.ndarray,
    seg_ids: np.ndarray,
    dt: np.ndarray,
    orig_context_indices: list[int],
    orig_future_indices: list[int],
    orig_full_indices: list[int],
    main_object_index: int,
    source_collision_frame: int | None,
    ratio_key: str,
    source_meta_json_path: str,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    source_total_frames = int(metadata.get("frames", metadata.get("num_frames", state_raw_local.shape[0])) or state_raw_local.shape[0])
    fps_value = float(metadata.get("fps", metadata.get("video_fps", 12.0)) or 12.0)
    full_frames = load_rgb_frames_by_indices(sample_dir, orig_full_indices)
    local_frame_paths = save_local_rgb_frames(out_dir, full_frames)
    context_video_path = out_dir / "context_video.mp4"
    future_video_path = out_dir / "future_gt_video.mp4"
    full_video_path = out_dir / "full_video.mp4"
    build_video(full_frames[: len(orig_context_indices)], context_video_path, fps=fps_value)
    build_video(full_frames[len(orig_context_indices) :], future_video_path, fps=fps_value)
    build_video(full_frames, full_video_path, fps=fps_value)
    if full_frames:
        Image.fromarray(full_frames[0]).save(out_dir / "first_frame.png")

    x_end = len(orig_context_indices)
    x_state_raw = state_raw_local[:x_end].astype(np.float32)
    y_state_raw = state_raw_local[x_end:].astype(np.float32)
    x_state_norm = state_norm_local[:x_end].astype(np.float32)
    y_state_norm = state_norm_local[x_end:].astype(np.float32)
    x_visibility = visibility_local[:x_end].astype(np.uint8)
    y_visibility = visibility_local[x_end:].astype(np.uint8)

    np.savez_compressed(
        out_dir / "state_pair.npz",
        object_ids=np.asarray(object_ids, dtype=np.int32),
        seg_ids=np.asarray(seg_ids, dtype=np.int32),
        visibility_mask=visibility_local.astype(np.uint8),
        state_raw=state_raw_local.astype(np.float32),
        state_norm=state_norm_local.astype(np.float32),
        x_state_raw=x_state_raw,
        y_state_raw=y_state_raw,
        x_state_norm=x_state_norm,
        y_state_norm=y_state_norm,
        x_visibility=x_visibility,
        y_visibility=y_visibility,
        x_frame_indices=np.arange(x_end, dtype=np.int32),
        y_frame_indices=np.arange(x_end, len(orig_full_indices), dtype=np.int32),
        orig_frame_indices=np.asarray(orig_full_indices, dtype=np.int32),
        dt=np.asarray(dt, dtype=np.float32),
    )
    np.savez_compressed(
        out_dir / "segment_state.npz",
        object_ids=np.asarray(object_ids, dtype=np.int32),
        seg_ids=np.asarray(seg_ids, dtype=np.int32),
        frame_indices=np.asarray(orig_full_indices, dtype=np.int32),
        context_frame_indices=np.asarray(orig_context_indices, dtype=np.int32),
        future_frame_indices=np.asarray(orig_future_indices, dtype=np.int32),
        state_raw=state_raw_local.astype(np.float32),
        state_norm=state_norm_local.astype(np.float32),
        visibility_mask=visibility_local.astype(np.uint8),
    )
    write_local_physics_stub(out_dir, state_raw_local, object_ids, seg_ids)
    copied_assets = copy_raw_like_window_assets(
        sample_dir=sample_dir,
        out_dir=out_dir,
        orig_full_indices=orig_full_indices,
        total_frames=source_total_frames,
        fps=fps_value,
        full_video_path=full_video_path,
    )

    local_events = trim_range_records(load_interaction_episodes(sample_dir), orig_full_indices)
    write_json(out_dir / "physics" / "event_windows.json", local_events)

    future_main_visibility_ratio = 0.0
    if y_visibility.size > 0 and 0 <= int(main_object_index) < int(y_visibility.shape[1]):
        future_main_visibility_ratio = float(y_visibility[:, int(main_object_index)].mean())
    motion_complexity = infer_motion_complexity(
        state_norm=y_state_norm.astype(np.float32),
        visibility_mask=y_visibility.astype(np.uint8),
    )
    start_index = int(orig_full_indices[0]) if orig_full_indices else 0
    window_interactions = build_window_interactions_from_local_events(
        events=local_events,
        object_count=len(object_text_meta) if object_text_meta else int(metadata.get("num_objects", 0) or 0),
        context_len=len(orig_context_indices),
        future_len=len(orig_future_indices),
    )
    segment_kind = "precollision_segment" if source_collision_frame is not None else "full_no_collision_clip"
    pair_meta = {
        "prompt": prompt,
        "source_scene_id": str(metadata.get("scene_id", sample_dir.name)),
        "source_sample_dir": str(sample_dir),
        "context_len": int(len(orig_context_indices)),
        "future_len": int(len(orig_future_indices)),
        "start_index": start_index,
        "main_object_index": int(main_object_index),
        "future_main_visibility_ratio": future_main_visibility_ratio,
        "resolution": metadata.get("resolution"),
        "camera_intrinsics": metadata.get("camera_intrinsics"),
        "objects": object_text_meta,
        "x_frame_paths": local_frame_paths[: len(orig_context_indices)],
        "y_frame_paths": local_frame_paths[len(orig_context_indices) :],
        "motion_complexity": motion_complexity,
        "window_interactions": window_interactions,
        "selection_info": {
            "ratio_key": ratio_key,
            "source_collision_frame": None if source_collision_frame is None else int(source_collision_frame),
            "source_segment_kind": segment_kind,
            "source_window_start_index": start_index,
            "source_window_end_exclusive": int(orig_full_indices[-1]) + 1 if orig_full_indices else start_index,
            "orig_context_frame_indices": orig_context_indices,
            "orig_future_frame_indices": orig_future_indices,
            "orig_full_frame_indices": orig_full_indices,
            "local_context_frame_indices": list(range(len(orig_context_indices))),
            "local_future_frame_indices": list(range(len(orig_context_indices), len(orig_full_indices))),
            "local_full_frame_indices": list(range(len(orig_full_indices))),
        },
    }
    write_json(out_dir / "pair_meta.json", pair_meta)

    meta_json = dict(metadata)
    meta_json["scene_id"] = sample_id
    meta_json["sample_id"] = sample_id
    meta_json["caption"] = prompt
    meta_json["description"] = prompt
    meta_json["dataset"] = "GenesisRigid"
    meta_json["split"] = split_name
    meta_json["view_type"] = "window"
    meta_json["frames"] = int(len(orig_full_indices))
    meta_json["fps"] = fps_value
    meta_json["sample_label"] = sample_label
    meta_json["sample_dir"] = str(out_dir)
    meta_json["source_sample_dir"] = str(sample_dir)
    meta_json["source_scene_id"] = str(metadata.get("scene_id", sample_dir.name))
    meta_json["context_frames"] = int(len(orig_context_indices))
    meta_json["future_frames"] = int(len(orig_future_indices))
    meta_json["raw_frames"] = int(len(orig_full_indices))
    meta_json["window_range"] = {
        "start_index": start_index,
        "end_exclusive": int(orig_full_indices[-1]) + 1 if orig_full_indices else start_index,
        "orig_context_frame_indices": orig_context_indices,
        "orig_future_frame_indices": orig_future_indices,
        "orig_full_frame_indices": orig_full_indices,
        "local_context_frame_indices": list(range(len(orig_context_indices))),
        "local_future_frame_indices": list(range(len(orig_context_indices), len(orig_full_indices))),
        "local_full_frame_indices": list(range(len(orig_full_indices))),
        "ratio_key": ratio_key,
        "segment_kind": segment_kind,
        "source_collision_frame": None if source_collision_frame is None else int(source_collision_frame),
    }
    meta_json["paths"] = {
        "sample_dir": str(out_dir),
        "rgb_video": str(out_dir / "videos" / "rgb.mp4"),
        "depth_video": str(out_dir / "videos" / "depth.mp4"),
        "future_gt_video_path": str(future_video_path),
        "full_video_path": str(full_video_path),
        "context_video_path": str(context_video_path),
        "first_frame_path": str(out_dir / "first_frame.png"),
        "meta_json_path": str(out_dir / "meta.json"),
    }
    meta_json["source_paths"] = {
        "meta_json_path": str(out_dir / "meta.json"),
        "pair_meta_json_path": str(out_dir / "pair_meta.json"),
        "segment_state_npz_path": str(out_dir / "segment_state.npz"),
        "state_pair_npz_path": str(out_dir / "state_pair.npz"),
        "source_sample_dir": str(sample_dir),
        "source_window_dir": "",
        "source_meta_json_path": str(source_meta_json_path),
    }
    meta_json["adapter_window"] = {
        "dataset": "genesis",
        "collision_bucket": str(window_interactions.get("future_window", {}).get("collision_type_bucket", "none")),
        "motion_complexity": str(motion_complexity.get("label", "unknown")),
        "segment_kind": segment_kind,
        "ratio_key": ratio_key,
        "orig_context_frame_indices": orig_context_indices,
        "orig_future_frame_indices": orig_future_indices,
        "orig_full_frame_indices": orig_full_indices,
    }
    outputs = dict(meta_json.get("outputs") or {})
    outputs.update(
        {
            "metadata": "meta.json",
            "rgb_video": "videos/rgb.mp4",
            "depth_video": "videos/depth.mp4" if copied_assets.get("depth_video") else "",
            "depth_metric": "physics/depth_metric.npy" if copied_assets.get("depth_metric") else "",
            "depth_normalized": "",
            "segmentation": "physics/seg.npy" if copied_assets.get("segmentation") else "",
            "flow": "",
            "anchor_targets": "physics/anchor_targets.npz",
            "rigid_kinematics": "physics/rigid_kinematics.npz" if copied_assets.get("rigid_kinematics") else "",
            "energy": "physics/energy.npz" if copied_assets.get("energy") else "",
            "properties": "physics/properties.json" if copied_assets.get("properties") else "",
            "contact_graph": "physics/contact_graph.npy" if copied_assets.get("contact_graph") else "",
            "contact_impulse": "physics/contact_impulse.npy" if copied_assets.get("contact_impulse") else "",
            "frame_phase": "physics/frame_phase.npy" if copied_assets.get("frame_phase") else "",
            "event_windows": "physics/event_windows.json",
            "depth_visualization_video": "visualizations/depth_vis.mp4" if copied_assets.get("depth_visualization_video") else "",
        }
    )
    meta_json["outputs"] = outputs
    meta_json["has_depth_metric"] = bool(copied_assets.get("depth_metric"))
    meta_json["has_seg"] = bool(copied_assets.get("segmentation"))
    meta_json["has_contact_graph"] = bool(copied_assets.get("contact_graph"))
    write_json(out_dir / "meta.json", meta_json)
    segment_info = {
        "sample_id": sample_id,
        "dataset": "genesis",
        "split": split_name,
        "context_frames": int(len(orig_context_indices)),
        "future_frames": int(len(orig_future_indices)),
        "full_frames": int(len(orig_full_indices)),
        "text": prompt,
        "selection_info": pair_meta["selection_info"],
    }
    write_json(out_dir / "segment_info.json", segment_info)
    return {
        "sample_id": sample_id,
        "split": split_name,
        "ratio_key": ratio_key,
        "sample_dir": str(out_dir),
        "rel_dir": str(out_dir.relative_to(output_root)),
        "source_sample_dir": str(sample_dir),
        "frames": int(len(orig_full_indices)),
        "context_frames": int(len(orig_context_indices)),
        "future_frames": int(len(orig_future_indices)),
        "source_collision_frame": None if source_collision_frame is None else int(source_collision_frame),
        "segment_kind": segment_kind,
        "motion_complexity": str(motion_complexity.get("label", "unknown")),
    }


def collect_manifest_entries(output_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for meta_path in sorted((output_root / "train").rglob("meta.json")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        source_paths = payload.get("source_paths") or {}
        adapter = payload.get("adapter_window") or {}
        entries.append(
            {
                "title": f"genesis / {payload.get('sample_id', meta_path.parent.name)}",
                "split": str(payload.get("split", "train")),
                "dataset": "genesis",
                "dataset_slug": "genesis",
                "segment_kind": str(adapter.get("segment_kind", "")),
                "bucket_key": f"genesis_none_{adapter.get('motion_complexity', 'unknown')}",
                "bucket_label": f"Genesis / no-collision / {adapter.get('motion_complexity', 'unknown')}",
                "motion": str(adapter.get("motion_complexity", "unknown")),
                "collision": "none",
                "rel_dir": str(meta_path.parent.relative_to(output_root)),
                "frames": int(payload.get("raw_frames", 0)),
                "context_frames": int(payload.get("context_frames", 0)),
                "future_frames": int(payload.get("future_frames", 0)),
                "pre_collision_future_frames": int(payload.get("future_frames", 0)),
                "source_sample_dir": str(source_paths.get("source_sample_dir", "")),
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    source_root = args.dataset_root / str(args.source_split).strip()
    train_genesis_root = args.output_root / "train" / "genesis"
    manifests_root = args.output_root / "manifests"
    ensure_dir(manifests_root)

    if args.rebuild_genesis_train and train_genesis_root.exists():
        shutil.rmtree(train_genesis_root)

    count_buckets = parse_csv(args.count_buckets)
    ratio_keys = parse_csv(args.ratios)
    unknown_ratios = [key for key in ratio_keys if key not in RATIO_SPECS]
    if unknown_ratios:
        raise ValueError(f"Unknown ratios: {unknown_ratios}")

    samples = collect_source_samples(source_root, args.sample_filter, count_buckets)
    if not args.include_invalid_by_qa:
        samples = [sample for sample in samples if "invalid_by_qa" not in sample.parts]
    if int(args.max_source_samples) > 0:
        samples = samples[: int(args.max_source_samples)]
    if not samples:
        raise RuntimeError(f"No source samples found under {source_root}")

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    ratio_counter: Counter[str] = Counter()

    for sample_dir in samples:
        metadata, source_meta_path = load_source_metadata(sample_dir)
        fps = float(metadata.get("fps", metadata.get("video_fps", 12.0)) or 12.0)
        raw = load_raw_state(sample_dir, fps)
        state_raw_full = raw["state_raw"].astype(np.float32)
        visibility_full = raw["visibility_mask"].astype(np.uint8)
        object_ids = raw["object_ids"].astype(np.int32)
        seg_ids = raw["seg_ids"].astype(np.int32)
        width, height = map(float, metadata["resolution"])
        cam = metadata["camera_intrinsics"]
        state_norm_full = normalize_state(
            state_raw=state_raw_full,
            width=width,
            height=height,
            depth_near=float(cam["near"]),
            depth_far=float(cam["far"]),
        ).astype(np.float32)
        main_object_index = resolve_main_object_index(metadata, object_ids)
        source_collision_frame = first_meaningful_collision_frame(sample_dir)
        if source_collision_frame == 0:
            skipped.append({"sample_dir": str(sample_dir), "reason": "collision_at_frame_0"})
            continue

        prompt = infer_prompt(sample_dir, metadata)
        object_text_meta = extract_object_text_meta(sample_dir, metadata)
        exported_any = False
        rel_source = sample_dir.relative_to(source_root)
        base_parent = train_genesis_root / rel_source.parent

        for ratio_key in ratio_keys:
            chosen = choose_window_frame_indices(
                total_frames=int(state_raw_full.shape[0]),
                source_collision_frame=source_collision_frame,
                ratio_key=ratio_key,
                min_context_frames=int(args.min_context_frames),
                min_future_frames=int(args.min_future_frames),
                min_window_frames=int(args.min_window_frames),
                max_window_frames=int(args.max_window_frames),
            )
            if chosen is None:
                continue
            orig_context_indices, orig_future_indices, full_orig_indices = chosen
            if not context_covers_all_objects(visibility_full, orig_context_indices):
                skipped.append(
                    {
                        "sample_dir": str(sample_dir),
                        "reason": "context_missing_some_objects",
                        "ratio_key": ratio_key,
                        "orig_context_frame_indices": orig_context_indices,
                    }
                )
                continue
            state_raw_local = state_raw_full[full_orig_indices].astype(np.float32)
            state_norm_local = state_norm_full[full_orig_indices].astype(np.float32)
            visibility_local = visibility_full[full_orig_indices].astype(np.uint8)
            sample_id = f"{sample_dir.name}__{ratio_key}"
            out_dir = base_parent / sample_id
            record = export_package(
                sample_dir=sample_dir,
                out_dir=out_dir,
                output_root=args.output_root,
                sample_id=sample_id,
                sample_label=f"{rel_source.as_posix()}::{ratio_key}",
                split_name="train",
                prompt=prompt,
                object_text_meta=object_text_meta,
                metadata=metadata,
                state_raw_local=state_raw_local,
                state_norm_local=state_norm_local,
                visibility_local=visibility_local,
                object_ids=object_ids,
                seg_ids=seg_ids,
                dt=np.asarray(raw["dt"], dtype=np.float32),
                orig_context_indices=orig_context_indices,
                orig_future_indices=orig_future_indices,
                orig_full_indices=full_orig_indices,
                main_object_index=main_object_index,
                source_collision_frame=source_collision_frame,
                ratio_key=ratio_key,
                source_meta_json_path=str(source_meta_path),
            )
            selected.append(record)
            ratio_counter[ratio_key] += 1
            exported_any = True

        if not exported_any:
            skipped.append(
                {
                    "sample_dir": str(sample_dir),
                    "reason": "window_too_short_for_requested_ratios",
                    "source_collision_frame": None if source_collision_frame is None else int(source_collision_frame),
                    "total_frames": int(state_raw_full.shape[0]),
                }
            )

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "source_split": str(args.source_split),
        "count_buckets": count_buckets,
        "ratios": ratio_keys,
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "ratio_counts": dict(ratio_counter),
    }
    write_json(manifests_root / "genesis_precollision_train_manifest.json", {"selected": selected, "skipped": skipped, "summary": summary})
    write_json(manifests_root / "genesis_precollision_train_summary.json", summary)

    root_manifest = {"selected": collect_manifest_entries(args.output_root)}
    write_json(args.output_root / "manifest.json", root_manifest)
    print(
        json.dumps(
            {
                "selected_count": len(selected),
                "skipped_count": len(skipped),
                "ratio_counts": dict(ratio_counter),
                "manifest": str(manifests_root / "genesis_precollision_train_manifest.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
