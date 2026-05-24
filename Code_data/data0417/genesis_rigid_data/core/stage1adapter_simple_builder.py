"""Build simple-motion Genesis stage1adapter train packages from raw train samples."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from core.utils_io import ensure_dir, load_json, write_json


STATE_ADAPTER_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/state_adapter")
TRAIN0419_ROOT = STATE_ADAPTER_ROOT.parent
if str(STATE_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(STATE_ADAPTER_ROOT))
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

from build_stage1_subsets import (  # type: ignore  # noqa: E402
    WINDOW_STRIDE,
    future_main_object_visibility_ok,
    load_raw_state,
    normalize_state,
    resolve_main_object_index,
    rgb_frame_paths,
    window_has_visible_object_every_frame,
)
from motion_complexity import infer_motion_complexity  # type: ignore  # noqa: E402
from window_interactions import infer_window_interactions, load_interaction_episodes, summarize_window_range  # type: ignore  # noqa: E402


CONTEXT_LEN = 8
FUTURE_LENGTHS = (5, 9, 13)
FUTURE_MAIN_VISIBILITY_THRESHOLD = 0.5
ALLOWED_COLLISION_BUCKETS = {"none", "env_only"}
ALLOWED_MOTION_LABELS = {"static", "simple"}


@dataclass(frozen=True)
class BuilderConfig:
    raw_root: Path
    output_root: Path
    overwrite: bool = False
    max_samples: int = 0


def resolve_dataset_roots(raw_root: Path) -> tuple[Path, Path]:
    raw_root = raw_root.resolve()
    if (raw_root / "train" / "rigid").exists():
        return raw_root, raw_root / "train"
    if (raw_root / "rigid").exists():
        return raw_root.parent, raw_root
    raise FileNotFoundError(
        f"Expected either <dataset_root>/train/rigid or <train_root>/rigid under {raw_root}"
    )


def strict_record_from_meta(meta: dict[str, Any], window_dir: Path) -> dict[str, Any] | None:
    wi = meta.get("window_interactions") or {}
    future_window = wi.get("future_window") or {}
    motion = meta.get("motion_complexity") or {}
    collision = str(future_window.get("collision_type_bucket", ""))
    motion_label = str(motion.get("label", ""))
    if collision not in ALLOWED_COLLISION_BUCKETS:
        return None
    if motion_label not in ALLOWED_MOTION_LABELS:
        return None
    frame_paths = list(meta.get("x_frame_paths", [])) + list(meta.get("y_frame_paths", []))
    if not frame_paths or any(not Path(str(path)).exists() for path in frame_paths):
        return None

    start_index = int(meta.get("start_index", 0))
    context_len = int(meta.get("context_len", 0))
    future_len = int(meta.get("future_len", 0))
    future_start = start_index + context_len
    future_end = future_start + future_len

    first_hit = first_main_collision_hit(meta)
    if collision == "none":
        segment_end = future_end
        segment_kind = "full_no_collision_window"
    else:
        if first_hit is None:
            segment_end = future_end
            segment_kind = "main_object_clear_full_future"
        else:
            segment_end = first_hit
            segment_kind = "precollision_segment"
    pre_future_frames = segment_end - future_start
    if collision == "env_only" and segment_kind == "precollision_segment" and pre_future_frames < 2:
        return None
    return {
        "window_dir": str(window_dir),
        "source_sample_dir": str(meta.get("source_sample_dir", "")),
        "start_index": start_index,
        "context_len": context_len,
        "future_len": future_len,
        "future_start": future_start,
        "future_end": future_end,
        "segment_end": segment_end,
        "segment_kind": segment_kind,
        "pre_future_frames": pre_future_frames,
        "collision": collision,
        "motion": motion_label,
        "main_object_index": int(meta.get("main_object_index", 0)),
        "pair_meta": meta,
    }


def first_main_collision_hit(meta: dict[str, Any]) -> int | None:
    wi = meta.get("window_interactions") or {}
    future_window = wi.get("future_window") or {}
    future_start = int(future_window.get("frame_start", int(meta.get("start_index", 0)) + int(meta.get("context_len", 0))))
    future_end = int(future_window.get("frame_end_exclusive", future_start + int(meta.get("future_len", 0))))
    main_idx = int(meta.get("main_object_index", 0))
    first: int | None = None
    for episode in future_window.get("episodes", []):
        obj_indices = [int(x) for x in episode.get("object_indices", []) if int(x) >= 0]
        if main_idx not in obj_indices:
            continue
        start_frame = int(episode.get("start_frame", future_end))
        end_frame = int(episode.get("end_frame", start_frame))
        if end_frame < future_start or start_frame >= future_end:
            continue
        hit = future_start if start_frame < future_start else start_frame
        first = hit if first is None else min(first, hit)
    return first


def choose_best_record(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score: tuple[int, int, int, int] | None = None
    for item in records:
        score = (
            int(item["pre_future_frames"]),
            int(item["future_len"]),
            int(item["segment_end"]) - int(item["start_index"]),
            -int(item["start_index"]),
        )
        if best is None or score > best_score:
            best = item
            best_score = score
    return best


def infer_object_count(sample_dir: Path, metadata: dict[str, Any]) -> int:
    object_count = len(metadata.get("objects", []) or [])
    if object_count > 0:
        return int(object_count)
    for bucket, count in (("count_01", 1), ("count_02", 2), ("count_03_04", 3)):
        if bucket in sample_dir.parts:
            return count
    return 0


def first_new_collision_onset(sample_dir: Path) -> int | None:
    first: int | None = None
    for episode in load_interaction_episodes(sample_dir):
        if str(episode.get("window_type", "")).strip() != "contact_onset":
            continue
        start_frame = int(episode.get("start_frame", -1))
        if start_frame < 1:
            continue
        first = start_frame if first is None else min(first, start_frame)
    return first


def iter_raw_samples(train_root: Path) -> list[Path]:
    samples = sorted(path.parent for path in (train_root / "rigid").rglob("metadata.json"))
    return [sample for sample in samples if (sample / "physics" / "anchor_targets.npz").exists()]


def build_strict_candidates_from_raw_sample(sample_dir: Path) -> list[dict[str, Any]]:
    metadata = load_json(sample_dir / "metadata.json")
    fps = float(metadata.get("fps", metadata.get("video_fps", 12.0)) or 12.0)
    raw = load_raw_state(sample_dir, fps)
    state_raw = raw["state_raw"]
    visibility_mask = raw["visibility_mask"]
    object_ids = raw["object_ids"]
    seg_ids = raw["seg_ids"]
    dt = np.asarray(raw["dt"]).astype(np.float32)
    total_frames = int(state_raw.shape[0])
    width, height = map(float, metadata["resolution"])
    cam = metadata["camera_intrinsics"]
    state_norm = normalize_state(
        state_raw=state_raw,
        width=width,
        height=height,
        depth_near=float(cam["near"]),
        depth_far=float(cam["far"]),
    )
    main_object_index = resolve_main_object_index(metadata, object_ids)
    candidates: list[dict[str, Any]] = []
    for future_len in FUTURE_LENGTHS:
        min_total = CONTEXT_LEN + int(future_len)
        if total_frames < min_total:
            continue
        max_start = total_frames - min_total
        for start in range(0, max_start + 1, WINDOW_STRIDE):
            c0 = int(start)
            c1 = c0 + CONTEXT_LEN
            f0 = c1
            f1 = f0 + int(future_len)
            if not window_has_visible_object_every_frame(visibility_mask, c0, c1):
                continue
            future_visible_ok, future_vis_ratio = future_main_object_visibility_ok(
                visibility_mask=visibility_mask,
                start=f0,
                end=f1,
                main_object_index=main_object_index,
                threshold=FUTURE_MAIN_VISIBILITY_THRESHOLD,
            )
            if not future_visible_ok:
                continue
            meta_payload = {
                "prompt": str(metadata.get("prompt", "")).strip()
                or str(metadata.get("caption", "")).strip()
                or "a rigid object motion scene",
                "source_scene_id": str(metadata.get("scene_id", sample_dir.name)),
                "source_sample_dir": str(sample_dir),
                "context_len": CONTEXT_LEN,
                "future_len": int(future_len),
                "start_index": int(start),
                "main_object_index": int(main_object_index),
                "future_main_visibility_ratio": float(future_vis_ratio),
                "resolution": metadata.get("resolution"),
                "camera_intrinsics": metadata.get("camera_intrinsics"),
                "objects": metadata.get("objects", []),
                "x_frame_paths": rgb_frame_paths(sample_dir, np.arange(c0, c1, dtype=np.int32)),
                "y_frame_paths": rgb_frame_paths(sample_dir, np.arange(f0, f1, dtype=np.int32)),
                "motion_complexity": infer_motion_complexity(
                    state_norm=state_norm[f0:f1].astype(np.float32),
                    visibility_mask=visibility_mask[f0:f1].astype(np.uint8),
                ),
            }
            meta_payload["window_interactions"] = infer_window_interactions(meta_payload)
            record = strict_record_from_meta(meta_payload, sample_dir)
            if record is None:
                continue
            record["_state_raw_full"] = state_raw
            record["_state_norm_full"] = state_norm
            record["_visibility_mask_full"] = visibility_mask
            record["_object_ids"] = object_ids
            record["_seg_ids"] = seg_ids
            record["_dt"] = dt
            candidates.append(record)
    return candidates


def build_count02_preonset_record_from_raw_sample(sample_dir: Path) -> dict[str, Any] | None:
    metadata = load_json(sample_dir / "metadata.json")
    fps = float(metadata.get("fps", metadata.get("video_fps", 12.0)) or 12.0)
    raw = load_raw_state(sample_dir, fps)
    state_raw = raw["state_raw"]
    visibility_mask = raw["visibility_mask"]
    object_ids = raw["object_ids"]
    seg_ids = raw["seg_ids"]
    dt = np.asarray(raw["dt"]).astype(np.float32)
    total_frames = int(state_raw.shape[0])
    if total_frames <= CONTEXT_LEN + 1:
        return None

    width, height = map(float, metadata["resolution"])
    cam = metadata["camera_intrinsics"]
    state_norm = normalize_state(
        state_raw=state_raw,
        width=width,
        height=height,
        depth_near=float(cam["near"]),
        depth_far=float(cam["far"]),
    )
    main_object_index = resolve_main_object_index(metadata, object_ids)
    onset_frame = first_new_collision_onset(sample_dir)
    segment_end = int(onset_frame) if onset_frame is not None else total_frames
    if segment_end <= CONTEXT_LEN + 1:
        return None

    context_start = 0
    context_end = CONTEXT_LEN
    future_start = context_end
    future_end = segment_end
    future_visible_ok, future_vis_ratio = future_main_object_visibility_ok(
        visibility_mask=visibility_mask,
        start=future_start,
        end=future_end,
        main_object_index=main_object_index,
        threshold=FUTURE_MAIN_VISIBILITY_THRESHOLD,
    )
    if not future_visible_ok:
        return None
    if not window_has_visible_object_every_frame(visibility_mask, context_start, context_end):
        return None

    meta_payload = {
        "prompt": str(metadata.get("prompt", "")).strip()
        or str(metadata.get("caption", "")).strip()
        or "a rigid object motion scene",
        "source_scene_id": str(metadata.get("scene_id", sample_dir.name)),
        "source_sample_dir": str(sample_dir),
        "context_len": CONTEXT_LEN,
        "future_len": int(future_end - future_start),
        "start_index": 0,
        "main_object_index": int(main_object_index),
        "future_main_visibility_ratio": float(future_vis_ratio),
        "resolution": metadata.get("resolution"),
        "camera_intrinsics": metadata.get("camera_intrinsics"),
        "objects": metadata.get("objects", []),
        "x_frame_paths": rgb_frame_paths(sample_dir, np.arange(context_start, context_end, dtype=np.int32)),
        "y_frame_paths": rgb_frame_paths(sample_dir, np.arange(future_start, future_end, dtype=np.int32)),
        "motion_complexity": infer_motion_complexity(
            state_norm=state_norm[future_start:future_end].astype(np.float32),
            visibility_mask=visibility_mask[future_start:future_end].astype(np.uint8),
        ),
    }
    motion_label = str((meta_payload.get("motion_complexity") or {}).get("label") or "")
    if not motion_label:
        return None
    record = {
        "window_dir": str(sample_dir),
        "source_sample_dir": str(sample_dir),
        "start_index": 0,
        "context_len": CONTEXT_LEN,
        "future_len": int(future_end - future_start),
        "future_start": future_start,
        "future_end": future_end,
        "segment_end": segment_end,
        "segment_kind": "count02_preonset_until_first_onset",
        "pre_future_frames": int(future_end - future_start),
        "collision": "none",
        "motion": motion_label,
        "main_object_index": int(main_object_index),
        "pair_meta": meta_payload,
    }
    record["_state_raw_full"] = state_raw
    record["_state_norm_full"] = state_norm
    record["_visibility_mask_full"] = visibility_mask
    record["_object_ids"] = object_ids
    record["_seg_ids"] = seg_ids
    record["_dt"] = dt
    record["_selection_policy"] = "count02_preonset_from_frame0"
    return record


def diagnose_count02_preonset_sample(sample_dir: Path) -> str:
    metadata = load_json(sample_dir / "metadata.json")
    fps = float(metadata.get("fps", metadata.get("video_fps", 12.0)) or 12.0)
    raw = load_raw_state(sample_dir, fps)
    state_raw = raw["state_raw"]
    visibility_mask = raw["visibility_mask"]
    total_frames = int(state_raw.shape[0])
    if total_frames <= CONTEXT_LEN + 1:
        return "too_short_total"
    object_ids = raw["object_ids"]
    main_object_index = resolve_main_object_index(metadata, object_ids)
    onset_frame = first_new_collision_onset(sample_dir)
    segment_end = int(onset_frame) if onset_frame is not None else total_frames
    if segment_end <= CONTEXT_LEN + 1:
        return f"too_short_preonset_{segment_end}"
    context_start = 0
    context_end = CONTEXT_LEN
    future_start = context_end
    future_end = segment_end
    if not window_has_visible_object_every_frame(visibility_mask, context_start, context_end):
        return "context_visibility_failed"
    future_visible_ok, future_vis_ratio = future_main_object_visibility_ok(
        visibility_mask=visibility_mask,
        start=future_start,
        end=future_end,
        main_object_index=main_object_index,
        threshold=FUTURE_MAIN_VISIBILITY_THRESHOLD,
    )
    if not future_visible_ok:
        return f"future_visibility_failed_{future_vis_ratio:.3f}"
    width, height = map(float, metadata["resolution"])
    cam = metadata["camera_intrinsics"]
    state_norm = normalize_state(
        state_raw=state_raw,
        width=width,
        height=height,
        depth_near=float(cam["near"]),
        depth_far=float(cam["far"]),
    )
    motion = infer_motion_complexity(
        state_norm=state_norm[future_start:future_end].astype(np.float32),
        visibility_mask=visibility_mask[future_start:future_end].astype(np.uint8),
    )
    motion_label = str((motion or {}).get("label") or "")
    if not motion_label:
        return "missing_motion_label"
    return "ok"


def build_video(frames: list[np.ndarray], dst: Path, fps: float = 12.0) -> None:
    if not frames:
        return
    ensure_dir(dst.parent)
    with imageio.get_writer(
        str(dst),
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        quality=8,
        ffmpeg_params=["-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def load_rgb_frames_by_indices(sample_dir: Path, frame_indices: list[int]) -> list[np.ndarray]:
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


def trim_interaction_episodes(source_sample_dir: Path, frame_indices: list[int]) -> list[dict[str, Any]]:
    if not frame_indices:
        return []
    index_map = {int(orig): idx for idx, orig in enumerate(frame_indices)}
    local_events: list[dict[str, Any]] = []
    for episode in load_interaction_episodes(source_sample_dir):
        start_frame = int(episode["start_frame"])
        end_frame = int(episode["end_frame"])
        overlapping = [orig for orig in frame_indices if start_frame <= int(orig) <= end_frame]
        if not overlapping:
            continue
        local_events.append(
            {
                "kind": str(episode["kind"]),
                "participants": [int(x) for x in episode["participants"]],
                "object_indices": [int(x) for x in episode["object_indices"]],
                "environment_name": str(episode["environment_name"]),
                "window_type": str(episode["window_type"]),
                "start_frame": int(index_map[int(overlapping[0])]),
                "end_frame": int(index_map[int(overlapping[-1])]),
            }
        )
    return local_events


def build_local_window_interactions(
    *,
    object_count: int,
    context_len: int,
    future_len: int,
    local_events: list[dict[str, Any]],
) -> dict[str, Any]:
    full_start = 0
    future_start = int(context_len)
    future_end = future_start + int(future_len)
    full_end = future_end
    full_summary = summarize_window_range(local_events, full_start, full_end)
    future_summary = summarize_window_range(local_events, future_start, future_end)
    future_bucket = (
        f"obj{int(object_count)}__{future_summary['collision_count_bucket']}__{future_summary['collision_type_bucket']}"
    )
    return {
        "object_count": int(object_count),
        "full_window": full_summary,
        "future_window": future_summary,
        "future_bucket": future_bucket,
        "source_event_episode_count": int(len(local_events)),
    }


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


def rel_source_path(sample_dir: Path, train_root: Path) -> Path:
    return sample_dir.relative_to(train_root)


def export_window_package(
    *,
    record: dict[str, Any],
    sample_dir: Path,
    out_dir: Path,
    source_meta_json_path: Path,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    source_sample_dir = Path(str(record["source_sample_dir"]))
    pair_meta = dict(record["pair_meta"])

    state_raw = np.asarray(record["_state_raw_full"]).astype(np.float32)
    state_norm = np.asarray(record["_state_norm_full"]).astype(np.float32)
    visibility_mask = np.asarray(record["_visibility_mask_full"]).astype(np.uint8)
    object_ids = np.asarray(record["_object_ids"]).astype(np.int32)
    seg_ids = np.asarray(record["_seg_ids"]).astype(np.int32)
    dt = np.asarray(record.get("_dt", 1.0 / 12.0), dtype=np.float32)

    future_start = int(record["future_start"])
    segment_end = int(record["segment_end"])
    future_orig = list(range(future_start, segment_end))
    context_count = choose_context_frame_count(int(record["context_len"]), len(future_orig))
    context_start = future_start - context_count
    context_orig = list(range(context_start, future_start))
    full_orig = context_orig + future_orig

    full_frames = load_rgb_frames_by_indices(source_sample_dir, full_orig)
    local_frame_paths = save_local_rgb_frames(out_dir, full_frames)
    context_video_path = out_dir / "context_video.mp4"
    future_video_path = out_dir / "future_gt_video.mp4"
    full_video_path = out_dir / "full_video.mp4"
    build_video(full_frames[: len(context_orig)], context_video_path)
    build_video(full_frames[len(context_orig) :], future_video_path)
    build_video(full_frames, full_video_path)
    if full_frames:
        Image.fromarray(full_frames[0]).save(out_dir / "first_frame.png")

    full_state_raw = state_raw[full_orig].astype(np.float32)
    full_state_norm = state_norm[full_orig].astype(np.float32)
    full_visibility = visibility_mask[full_orig].astype(np.uint8)
    x_state_raw = full_state_raw[: len(context_orig)].astype(np.float32)
    y_state_raw = full_state_raw[len(context_orig) :].astype(np.float32)
    x_state_norm = full_state_norm[: len(context_orig)].astype(np.float32)
    y_state_norm = full_state_norm[len(context_orig) :].astype(np.float32)
    x_visibility = full_visibility[: len(context_orig)].astype(np.uint8)
    y_visibility = full_visibility[len(context_orig) :].astype(np.uint8)

    np.savez_compressed(
        out_dir / "state_pair.npz",
        object_ids=object_ids,
        seg_ids=seg_ids,
        visibility_mask=full_visibility,
        state_raw=full_state_raw,
        state_norm=full_state_norm,
        x_state_raw=x_state_raw,
        y_state_raw=y_state_raw,
        x_state_norm=x_state_norm,
        y_state_norm=y_state_norm,
        x_visibility=x_visibility,
        y_visibility=y_visibility,
        x_frame_indices=np.arange(len(context_orig), dtype=np.int32),
        y_frame_indices=np.arange(len(context_orig), len(full_orig), dtype=np.int32),
        orig_frame_indices=np.asarray(full_orig, dtype=np.int32),
        dt=dt,
    )
    np.savez_compressed(
        out_dir / "segment_state.npz",
        object_ids=object_ids,
        seg_ids=seg_ids,
        frame_indices=np.asarray(full_orig, dtype=np.int32),
        context_frame_indices=np.asarray(context_orig, dtype=np.int32),
        future_frame_indices=np.asarray(future_orig, dtype=np.int32),
        state_raw=full_state_raw,
        state_norm=full_state_norm,
        visibility_mask=full_visibility,
    )

    write_local_physics_stub(out_dir, full_state_raw, object_ids, seg_ids)
    local_events = trim_interaction_episodes(source_sample_dir, full_orig)
    if str(record.get("_selection_policy", "")) == "count02_preonset_from_frame0":
        local_events = []
    write_json(out_dir / "physics" / "event_windows.json", local_events)

    local_pair_meta = {
        "prompt": str(pair_meta.get("prompt", "")).strip() or "a rigid object motion scene",
        "source_scene_id": str(pair_meta.get("source_scene_id", sample_dir.name)),
        "source_sample_dir": str(source_sample_dir),
        "context_len": int(len(context_orig)),
        "future_len": int(len(future_orig)),
        "start_index": int(context_start),
        "main_object_index": int(record["main_object_index"]),
        "future_main_visibility_ratio": float(y_visibility[:, int(record["main_object_index"])].mean())
        if y_visibility.size > 0
        else 0.0,
        "resolution": pair_meta.get("resolution"),
        "camera_intrinsics": pair_meta.get("camera_intrinsics"),
        "objects": pair_meta.get("objects", []),
        "x_frame_paths": local_frame_paths[: len(context_orig)],
        "y_frame_paths": local_frame_paths[len(context_orig) :],
        "motion_complexity": infer_motion_complexity(
            state_norm=y_state_norm.astype(np.float32),
            visibility_mask=y_visibility.astype(np.uint8),
        ),
        "selection_info": {
            "source_window_dir": str(source_sample_dir),
            "source_collision_bucket": str(record["collision"]),
            "source_motion_complexity": str(record["motion"]),
            "source_segment_kind": str(record["segment_kind"]),
            "source_start_index": int(record["start_index"]),
            "source_future_len": int(record["future_len"]),
            "source_future_start": int(record["future_start"]),
            "source_segment_end": int(record["segment_end"]),
            "source_pre_future_frames": int(record["pre_future_frames"]),
            "orig_context_frame_indices": context_orig,
            "orig_future_frame_indices": future_orig,
            "orig_full_frame_indices": full_orig,
        },
    }
    if str(record.get("_selection_policy", "")) == "count02_preonset_from_frame0":
        local_pair_meta["window_interactions"] = build_local_window_interactions(
            object_count=len(local_pair_meta.get("objects", []) or []),
            context_len=int(len(context_orig)),
            future_len=int(len(future_orig)),
            local_events=local_events,
        )
    else:
        local_pair_meta["window_interactions"] = infer_window_interactions(local_pair_meta)
    write_json(out_dir / "pair_meta.json", local_pair_meta)

    meta_json = {
        "sample_id": sample_dir.name,
        "caption": local_pair_meta["prompt"],
        "description": local_pair_meta["prompt"],
        "dataset": "GenesisRigid",
        "split": "train",
        "fps": 12,
        "context_frames": int(len(context_orig)),
        "future_frames": int(len(future_orig)),
        "raw_frames": int(len(full_orig)),
        "sample_label": sample_dir.name,
        "paths": {
            "sample_dir": str(out_dir),
            "future_gt_video_path": str(future_video_path),
            "full_video_path": str(full_video_path),
            "context_video_path": str(context_video_path),
            "first_frame_path": str(out_dir / "first_frame.png"),
            "meta_json_path": str(out_dir / "meta.json"),
        },
        "source_paths": {
            "meta_json_path": str(out_dir / "meta.json"),
            "pair_meta_json_path": str(out_dir / "pair_meta.json"),
            "state_pair_npz_path": str(out_dir / "state_pair.npz"),
            "source_sample_dir": str(source_sample_dir),
            "source_window_dir": str(source_sample_dir),
            "source_meta_json_path": str(source_meta_json_path),
        },
        "adapter_window": {
            "dataset": "genesis",
            "collision_bucket": str(record["collision"]),
            "motion_complexity": str(record["motion"]),
            "segment_kind": str(record["segment_kind"]),
            "orig_context_frame_indices": context_orig,
            "orig_future_frame_indices": future_orig,
            "orig_full_frame_indices": full_orig,
        },
    }
    write_json(out_dir / "meta.json", meta_json)
    write_json(
        out_dir / "segment_info.json",
        {
            "sample_id": sample_dir.name,
            "dataset": "genesis",
            "split": "train",
            "context_frames": len(context_orig),
            "future_frames": len(future_orig),
            "full_frames": len(full_orig),
            "text": local_pair_meta["prompt"],
            "selection_info": local_pair_meta["selection_info"],
        },
    )
    return {
        "sample_id": sample_dir.name,
        "source_sample_dir": str(source_sample_dir),
        "sample_dir": str(out_dir),
        "collision_bucket": str(record["collision"]),
        "motion_complexity": str(record["motion"]),
        "segment_kind": str(record["segment_kind"]),
        "context_frames": len(context_orig),
        "future_frames": len(future_orig),
        "full_frames": len(full_orig),
    }


def choose_context_frame_count(context_len: int, future_visible_frames: int) -> int:
    if context_len <= 0:
        return 0
    if context_len < 2:
        return context_len
    if future_visible_frames <= 2:
        return min(context_len, 2)
    return min(context_len, future_visible_frames)


def process_dataset(config: BuilderConfig) -> dict[str, Any]:
    dataset_root, train_root = resolve_dataset_roots(config.raw_root)
    output_root = config.output_root.resolve()
    train_output_root = output_root / "train" / "genesis"
    if config.overwrite and output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    ensure_dir(train_output_root)
    ensure_dir(output_root / "manifests")

    samples = iter_raw_samples(train_root)
    if config.max_samples > 0:
        samples = samples[: int(config.max_samples)]

    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    collision_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()

    for sample_dir in samples:
        metadata = load_json(sample_dir / "metadata.json")
        object_count = infer_object_count(sample_dir, metadata)
        if object_count == 2:
            best = build_count02_preonset_record_from_raw_sample(sample_dir)
            skip_reason = diagnose_count02_preonset_sample(sample_dir) if best is None else ""
        else:
            candidates = build_strict_candidates_from_raw_sample(sample_dir)
            best = choose_best_record(candidates)
            skip_reason = "no_strict_simple_window"
        if best is None:
            skipped.append({"sample_dir": str(sample_dir), "reason": skip_reason})
            continue
        rel_source = rel_source_path(sample_dir, train_root)
        out_dir = train_output_root / rel_source
        item = export_window_package(
            record=best,
            sample_dir=sample_dir,
            out_dir=out_dir,
            source_meta_json_path=sample_dir / "metadata.json",
        )
        items.append(item)
        collision_counts.update([str(item["collision_bucket"])])
        motion_counts.update([str(item["motion_complexity"])])

    summary = {
        "raw_root": str(config.raw_root.resolve()),
        "dataset_root": str(dataset_root),
        "train_root": str(train_root),
        "output_root": str(output_root),
        "num_input_samples": len(samples),
        "num_exported_samples": len(items),
        "num_skipped_samples": len(skipped),
        "collision_buckets": dict(sorted(collision_counts.items())),
        "motion_complexities": dict(sorted(motion_counts.items())),
    }
    write_json(output_root / "manifests" / "train_items.json", items)
    write_json(output_root / "manifests" / "skipped_items.json", skipped)
    write_json(output_root / "manifests" / "summary.json", summary)
    return summary
