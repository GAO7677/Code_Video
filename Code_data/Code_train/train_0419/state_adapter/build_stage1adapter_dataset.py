#!/usr/bin/env python3
"""Build strict simple-motion stage1adapter train/test datasets plus a local preview portal."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import shutil
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from google.protobuf import message_factory as _message_factory
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN0419_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

if not hasattr(_message_factory, "GetMessageClass"):
    def _compat_get_message_class(descriptor):
        return _message_factory.MessageFactory().GetPrototype(descriptor)

    _message_factory.GetMessageClass = _compat_get_message_class

from build_stage1_subsets import (  # noqa: E402
    WINDOW_STRIDE,
    future_main_object_visibility_ok,
    load_raw_state,
    normalize_state,
    resolve_main_object_index,
    rgb_frame_paths,
    window_has_visible_object_every_frame,
)
from motion_complexity import infer_motion_complexity  # noqa: E402
from prepare_movi_d_physics import (  # noqa: E402
    build_contact_tensors,
    build_prompt,
    boxes_from_segmentations,
    center_depth_from_masks,
    choose_main_object_index,
    compute_state_9d,
    convert_record,
    decode_float_tensor,
    decode_image_sequence,
    decode_int_tensor,
    decode_rgb_frames,
    decode_text_feature,
    iter_serialized_records,
    parse_example,
    ragged_to_frame_boxes,
    uint16_to_metric,
)
from window_interactions import infer_window_interactions, load_interaction_episodes  # noqa: E402


GENESIS_ORACLE_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
)
MOVI_TRAIN_ORACLE_ROOT = Path(
    "/data/gaoya/dataset/kubric_tfds_movi-d/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
)
GENESIS_MYTEST_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/mytest"
)
GENESIS_TRAIN_RAW_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train"
)
MOVI_MYTEST_ROOT = Path("/data/gaoya/dataset/kubric_tfds_movi-d/mytest")
MOVI_TFRECORD_ROOT = Path("/data/gaoya/dataset/kubric_tfds_movi-d/test")
STAGE1ADAPTER_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter"
)
MOVI_TEST_CACHE_ROOT = STAGE1ADAPTER_ROOT / "_cache" / "movi_d_test_raw"
DEFAULT_PORT = 8117
FUTURE_LENGTHS = (5, 9, 13)
CONTEXT_LEN = 8
FUTURE_MAIN_VISIBILITY_THRESHOLD = 0.5


def load_portal_module():
    module_path = SCRIPT_DIR / "visualizations" / "build_precollision_segment_portal.py"
    spec = importlib.util.spec_from_file_location("precollision_portal", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PORTAL_MOD = load_portal_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=STAGE1ADAPTER_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing train/test/benchmark outputs first.")
    parser.add_argument(
        "--only_genesis_test",
        action="store_true",
        help="Only rebuild stage1adapter/test/genesis without touching train, movi test, or benchmark.",
    )
    parser.add_argument(
        "--skip_movi_test_cache",
        action="store_true",
        help="Reuse existing MOVI-D test raw cache under stage1adapter/_cache if present.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def dataset_slug(dataset_name: str) -> str:
    return "movi-d" if str(dataset_name) == "movi_d" else str(dataset_name)


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def choose_context_frame_count(context_len: int, future_visible_frames: int) -> int:
    if context_len <= 0:
        return 0
    if context_len < 2:
        return context_len
    if future_visible_frames <= 2:
        return min(context_len, 2)
    return min(context_len, future_visible_frames)


def first_main_collision_hit(meta: dict[str, Any]) -> int | None:
    wi = meta.get("window_interactions") or {}
    fw = wi.get("future_window") or {}
    future_start = int(fw.get("frame_start", int(meta.get("start_index", 0)) + int(meta.get("context_len", 0))))
    future_end = int(fw.get("frame_end_exclusive", future_start + int(meta.get("future_len", 0))))
    main_idx = int(meta.get("main_object_index", 0))
    first: int | None = None
    for episode in fw.get("episodes", []):
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


def strict_record_from_meta(meta: dict[str, Any], window_dir: Path) -> dict[str, Any] | None:
    wi = meta.get("window_interactions") or {}
    fw = wi.get("future_window") or {}
    mc = meta.get("motion_complexity") or {}
    collision = str(fw.get("collision_type_bucket", ""))
    motion = str(mc.get("label", ""))
    if collision not in {"none", "env_only"}:
        return None
    if motion not in {"static", "simple"}:
        return None
    frame_paths = list(meta.get("x_frame_paths", [])) + list(meta.get("y_frame_paths", []))
    if not meta.get("_in_memory_frames"):
        if not frame_paths or any(not Path(str(path)).exists() for path in frame_paths):
            return None
    elif not frame_paths:
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
        "dataset": str(meta.get("dataset_source", "")).lower().replace("-", "_") or "",
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
        "motion": motion,
        "main_object_index": int(meta.get("main_object_index", 0)),
        "pair_meta": meta,
    }


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


def build_train_source_index() -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {"genesis": {}, "movi-d": {}}
    for dataset_name, root in (("genesis", GENESIS_ORACLE_ROOT), ("movi-d", MOVI_TRAIN_ORACLE_ROOT)):
        if root.exists():
            candidates = []
            for pair_meta_path in sorted(root.rglob("pair_meta.json")):
                meta = load_json(pair_meta_path)
                record = strict_record_from_meta(meta, pair_meta_path.parent)
                if record is None:
                    continue
                record["dataset"] = dataset_name if dataset_name == "genesis" else "movi_d"
                candidates.append(record)
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in candidates:
                grouped[str(record["source_sample_dir"])].append(record)
            for source_sample_dir, items in grouped.items():
                best = choose_best_record(items)
                if best is not None:
                    index[dataset_name][source_sample_dir] = best
            continue

        if dataset_name != "genesis" or not GENESIS_TRAIN_RAW_ROOT.exists():
            continue

        seen_genesis_samples: set[Path] = set()
        for meta_name in ("metadata.json", "meta.json"):
            for metadata_path in sorted(GENESIS_TRAIN_RAW_ROOT.rglob(meta_name)):
                sample_dir = metadata_path.parent
                if sample_dir in seen_genesis_samples:
                    continue
                seen_genesis_samples.add(sample_dir)
                if not (sample_dir / "physics" / "anchor_targets.npz").exists():
                    continue
                candidates = build_strict_candidates_from_raw_sample(sample_dir)
                best = choose_best_record(candidates)
                if best is not None:
                    index[dataset_name][str(sample_dir)] = best
    return index


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
        ensure_dir(frame_path.parent)
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(str(frame_path))
    return frame_paths


def copy_frame_sequence_by_indices(src_dir: Path, dst_dir: Path, frame_indices: Iterable[int]) -> list[str]:
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


def load_depth_preview_frames_by_indices(sample_dir: Path, frame_indices: Iterable[int]) -> list[np.ndarray]:
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


def trim_interaction_events(events: list[dict[str, Any]], frame_indices: list[int]) -> list[dict[str, Any]]:
    if not frame_indices:
        return []
    index_map = {int(orig): idx for idx, orig in enumerate(frame_indices)}
    local_events: list[dict[str, Any]] = []
    for episode in events:
        start_frame = int(episode.get("start_frame", 0))
        end_frame = int(episode.get("end_frame", start_frame))
        overlapping = [orig for orig in frame_indices if start_frame <= int(orig) <= end_frame]
        if not overlapping:
            continue
        local_events.append(
            {
                "kind": str(episode.get("kind", "")),
                "participants": [int(x) for x in episode.get("participants", [])],
                "object_indices": [int(x) for x in episode.get("object_indices", []) if int(x) >= 0],
                "environment_name": str(episode.get("environment_name", "")),
                "window_type": str(episode.get("window_type", "")),
                "start_frame": int(index_map[int(overlapping[0])]),
                "end_frame": int(index_map[int(overlapping[-1])]),
            }
        )
    return local_events


def trim_range_records(records: Iterable[dict[str, Any]], frame_indices: list[int]) -> list[dict[str, Any]]:
    if not frame_indices:
        return []
    index_map = {int(orig): idx for idx, orig in enumerate(frame_indices)}
    local_records: list[dict[str, Any]] = []
    for record in records:
        start_frame = int(record.get("start_frame", record.get("frame_idx", -1)))
        end_frame = int(record.get("end_frame", start_frame))
        overlapping = [orig for orig in frame_indices if start_frame <= int(orig) <= end_frame]
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


def save_trimmed_npy(src_path: Path, dst_path: Path, total_frames: int, frame_indices: list[int]) -> bool:
    if not src_path.exists():
        return False
    data = np.load(src_path)
    trimmed = data[frame_indices] if data.ndim > 0 and int(data.shape[0]) == int(total_frames) else data
    ensure_dir(dst_path.parent)
    np.save(dst_path, trimmed)
    return True


def save_trimmed_npz(src_path: Path, dst_path: Path, total_frames: int, frame_indices: list[int]) -> bool:
    if not src_path.exists():
        return False
    payload = np.load(src_path)
    trimmed: dict[str, np.ndarray] = {}
    for key in payload.files:
        value = np.asarray(payload[key])
        if value.ndim > 0 and int(value.shape[0]) == int(total_frames):
            trimmed[key] = value[frame_indices]
        else:
            trimmed[key] = value
    ensure_dir(dst_path.parent)
    np.savez_compressed(dst_path, **trimmed)
    return True


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


def copy_raw_like_window_assets(
    *,
    source_sample_dir: Path,
    out_dir: Path,
    full_orig: list[int],
    total_frames: int,
    fps: float,
    full_video_path: Path,
) -> dict[str, bool]:
    copied: dict[str, bool] = {}
    videos_dir = out_dir / "videos"
    ensure_dir(videos_dir)
    shutil.copy2(full_video_path, videos_dir / "rgb.mp4")
    copied["rgb_video"] = True

    depth_paths = copy_frame_sequence_by_indices(source_sample_dir / "depth", out_dir / "depth", full_orig)
    copied["depth_frames"] = bool(depth_paths)
    if depth_paths:
        depth_frames = load_depth_preview_frames_by_indices(source_sample_dir, full_orig)
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

    physics_src = source_sample_dir / "physics"
    physics_dst = out_dir / "physics"
    ensure_dir(physics_dst)
    copied["depth_metric"] = save_trimmed_npy(
        physics_src / "depth_metric.npy", physics_dst / "depth_metric.npy", total_frames, full_orig
    )
    copied["segmentation"] = save_trimmed_npy(
        physics_src / "seg.npy", physics_dst / "seg.npy", total_frames, full_orig
    )
    copied["contact_graph"] = save_trimmed_npy(
        physics_src / "contact_graph.npy", physics_dst / "contact_graph.npy", total_frames, full_orig
    )
    copied["contact_impulse"] = save_trimmed_npy(
        physics_src / "contact_impulse.npy", physics_dst / "contact_impulse.npy", total_frames, full_orig
    )
    copied["frame_phase"] = save_trimmed_npy(
        physics_src / "frame_phase.npy", physics_dst / "frame_phase.npy", total_frames, full_orig
    )
    copied["rigid_kinematics"] = save_trimmed_npz(
        physics_src / "rigid_kinematics.npz", physics_dst / "rigid_kinematics.npz", total_frames, full_orig
    )
    copied["energy"] = save_trimmed_npz(
        physics_src / "energy.npz", physics_dst / "energy.npz", total_frames, full_orig
    )
    if (physics_src / "properties.json").exists():
        shutil.copy2(physics_src / "properties.json", physics_dst / "properties.json")
        copied["properties"] = True
    else:
        copied["properties"] = False
    if (source_sample_dir / "scene_input.json").exists():
        shutil.copy2(source_sample_dir / "scene_input.json", out_dir / "scene_input.json")
        copied["scene_input"] = True
    else:
        copied["scene_input"] = False
    return copied


def train_rel_source_path(dataset_name: str, sample_dir: Path) -> Path:
    text = str(sample_dir)
    if dataset_name == "genesis":
        marker = "/version_1_genesis_rigid_data_all_cases/train/"
    else:
        marker = "/kubric_tfds_movi-d/mytrain/movi_d_physics/train/"
    if marker in text:
        return Path(text.split(marker, 1)[1])
    return Path(sample_dir.name)


def export_window_package(
    *,
    record: dict[str, Any],
    out_dir: Path,
    sample_id: str,
    split: str,
    dataset_name: str,
    sample_label: str,
    source_meta_json_path: str,
    extra_source_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    window_dir = Path(str(record["window_dir"]))
    source_sample_dir = Path(str(record["source_sample_dir"]))
    pair_meta = dict(record["pair_meta"])
    source_meta_payload: dict[str, Any] = {}
    source_meta_candidate = Path(str(source_meta_json_path))
    if source_meta_candidate.exists():
        source_meta_payload = load_json(source_meta_candidate)

    state_pair_path = window_dir / "state_pair.npz"
    if state_pair_path.exists():
        with np.load(state_pair_path) as payload:
            state_raw = np.asarray(payload["state_raw"]).astype(np.float32)
            state_norm = np.asarray(payload["state_norm"]).astype(np.float32)
            visibility_mask = np.asarray(payload["visibility_mask"]).astype(np.uint8)
            object_ids = np.asarray(payload["object_ids"]).astype(np.int32)
            seg_ids = np.asarray(payload["seg_ids"]).astype(np.int32)
            dt = (
                np.asarray(payload["dt"]).astype(np.float32)
                if "dt" in payload
                else np.asarray(1.0 / 12.0, dtype=np.float32)
            )
    else:
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
    fps_value = float(source_meta_payload.get("fps", source_meta_payload.get("video_fps", 12.0)) or 12.0)
    source_total_frames = int(
        source_meta_payload.get("frames", source_meta_payload.get("num_frames", len(full_orig))) or len(full_orig)
    )

    rgb_frames_full = record.get("_rgb_frames_full")
    if rgb_frames_full is not None:
        full_frames = [np.asarray(rgb_frames_full[int(idx)], dtype=np.uint8) for idx in full_orig]
    else:
        full_frames = load_rgb_frames_by_indices(source_sample_dir, full_orig)
    local_frame_paths = save_local_rgb_frames(out_dir, full_frames)
    context_video_path = out_dir / "context_video.mp4"
    future_video_path = out_dir / "future_gt_video.mp4"
    full_video_path = out_dir / "full_video.mp4"
    build_video(full_frames[: len(context_orig)], context_video_path, fps=fps_value)
    build_video(full_frames[len(context_orig) :], future_video_path, fps=fps_value)
    build_video(full_frames, full_video_path, fps=fps_value)
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
    copied_assets = copy_raw_like_window_assets(
        source_sample_dir=source_sample_dir,
        out_dir=out_dir,
        full_orig=full_orig,
        total_frames=source_total_frames,
        fps=fps_value,
        full_video_path=full_video_path,
    )
    source_event_windows = record.get("_source_event_windows")
    if source_event_windows is not None:
        local_events = trim_interaction_events(list(source_event_windows), full_orig)
    else:
        local_events = trim_interaction_episodes(source_sample_dir, full_orig)
    write_json(out_dir / "physics" / "event_windows.json", local_events)
    raw_collision_path = source_sample_dir / "physics" / "collision_events.json"
    if raw_collision_path.exists():
        raw_collision_payload = json.loads(raw_collision_path.read_text(encoding="utf-8"))
        write_json(out_dir / "physics" / "collision_events.json", trim_range_records(raw_collision_payload, full_orig))

    local_pair_meta = {
        "prompt": str(pair_meta.get("prompt", "")).strip() or "a rigid object motion scene",
        "source_scene_id": str(pair_meta.get("source_scene_id", sample_id)),
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
            "source_window_dir": str(window_dir),
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
    local_pair_meta["window_interactions"] = infer_window_interactions(local_pair_meta)
    write_json(out_dir / "pair_meta.json", local_pair_meta)

    meta_json = dict(source_meta_payload)
    meta_json["scene_id"] = sample_id
    meta_json["sample_id"] = sample_id
    meta_json["caption"] = local_pair_meta["prompt"]
    meta_json["description"] = local_pair_meta["prompt"]
    meta_json["dataset"] = "MOVI-D" if dataset_name == "movi-d" else "GenesisRigid"
    meta_json["split"] = split
    meta_json["fps"] = fps_value
    meta_json["view_type"] = "window"
    meta_json["frames"] = int(len(full_orig))
    meta_json["context_frames"] = int(len(context_orig))
    meta_json["future_frames"] = int(len(future_orig))
    meta_json["raw_frames"] = int(len(full_orig))
    meta_json["sample_label"] = sample_label
    meta_json["sample_dir"] = str(out_dir)
    meta_json["source_sample_dir"] = str(source_sample_dir)
    meta_json["source_scene_id"] = str(local_pair_meta.get("source_scene_id", source_sample_dir.name))
    meta_json["window_range"] = {
        "start_index": int(context_start),
        "end_exclusive": int(segment_end),
        "orig_context_frame_indices": context_orig,
        "orig_future_frame_indices": future_orig,
        "orig_full_frame_indices": full_orig,
        "local_context_frame_indices": list(range(len(context_orig))),
        "local_future_frame_indices": list(range(len(context_orig), len(full_orig))),
        "local_full_frame_indices": list(range(len(full_orig))),
        "segment_kind": str(record["segment_kind"]),
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
        "source_sample_dir": str(source_sample_dir),
        "source_window_dir": str(window_dir),
        "source_meta_json_path": str(source_meta_json_path),
    }
    meta_json["adapter_window"] = {
        "dataset": dataset_name,
        "collision_bucket": str(record["collision"]),
        "motion_complexity": str(record["motion"]),
        "segment_kind": str(record["segment_kind"]),
        "orig_context_frame_indices": context_orig,
        "orig_future_frame_indices": future_orig,
        "orig_full_frame_indices": full_orig,
    }
    if extra_source_paths:
        meta_json["source_paths"].update(extra_source_paths)
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
            "collision_events": "physics/collision_events.json" if (out_dir / "physics" / "collision_events.json").exists() else "",
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
    write_json(
        out_dir / "segment_info.json",
        {
            "sample_id": sample_id,
            "dataset": dataset_name,
            "split": split,
            "context_frames": len(context_orig),
            "future_frames": len(future_orig),
            "full_frames": len(full_orig),
            "text": local_pair_meta["prompt"],
            "selection_info": local_pair_meta["selection_info"],
        },
    )
    return {
        "sample_id": sample_id,
        "dataset": dataset_name,
        "split": split,
        "sample_dir": str(out_dir),
        "rel_dir": str(out_dir.relative_to(STAGE1ADAPTER_ROOT)),
        "caption": local_pair_meta["prompt"],
        "context_frames": len(context_orig),
        "future_frames": len(future_orig),
        "full_frames": len(full_orig),
        "collision_bucket": str(record["collision"]),
        "motion_complexity": str(record["motion"]),
        "segment_kind": str(record["segment_kind"]),
        "context_video_path": str(context_video_path),
        "future_gt_video_path": str(future_video_path),
        "full_video_path": str(full_video_path),
        "meta_json_path": str(out_dir / "meta.json"),
    }


def prepare_train_packages(output_root: Path, train_index: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dataset_name in ("genesis", "movi-d"):
        for source_sample_dir, record in sorted(train_index[dataset_name].items()):
            sample_dir = Path(source_sample_dir)
            rel_source = train_rel_source_path(dataset_name, sample_dir)
            out_dir = output_root / "train" / dataset_name / rel_source
            meta_source = sample_dir / "metadata.json"
            if not meta_source.exists():
                meta_source = sample_dir / "meta.json"
            items.append(
                export_window_package(
                    record=record,
                    out_dir=out_dir,
                    sample_id=sample_dir.name,
                    split="train",
                    dataset_name=dataset_name,
                    sample_label=str(rel_source),
                    source_meta_json_path=str(meta_source),
                )
            )
    return items


def prepare_genesis_test_packages(output_root: Path, train_index: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for meta_path in sorted(GENESIS_MYTEST_ROOT.glob("*/meta.json")):
        meta = load_json(meta_path)
        source_sample_dir = str((meta.get("source_paths") or {}).get("source_sample_dir", ""))
        record = train_index["genesis"].get(source_sample_dir)
        if record is None:
            continue
        sample_id = str(meta.get("sample_id") or meta_path.parent.name)
        out_dir = output_root / "test" / "genesis" / sample_id
        items.append(
            export_window_package(
                record=record,
                out_dir=out_dir,
                sample_id=sample_id,
                split="test",
                dataset_name="genesis",
                sample_label=sample_id,
                source_meta_json_path=str(meta_path),
                extra_source_paths={
                    "heldout_meta_json_path": str(meta_path),
                    "heldout_sample_dir": str(meta_path.parent),
                },
            )
        )
    return items


def iter_movi_mytest_targets() -> dict[str, dict[int, dict[str, Any]]]:
    by_shard: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for meta_path in sorted(MOVI_MYTEST_ROOT.glob("*/meta.json")):
        meta = load_json(meta_path)
        source_paths = meta.get("source_paths") or {}
        shard_path = str(source_paths.get("tfrecord_path", ""))
        record_index = int(source_paths.get("tfrecord_record_index", -1))
        if shard_path and record_index >= 0:
            by_shard[shard_path][record_index] = meta
    return by_shard


def movi_target_source_sample_dir(meta: dict[str, Any]) -> str:
    sample_dir = str((meta.get("paths") or {}).get("sample_dir", ""))
    if sample_dir:
        return sample_dir
    source_meta_json = str((meta.get("source_paths") or {}).get("meta_json_path", ""))
    if source_meta_json:
        return str(Path(source_meta_json).parent)
    return ""


def build_strict_candidates_from_raw_sample(sample_dir: Path) -> list[dict[str, Any]]:
    meta_path = sample_dir / "metadata.json"
    if not meta_path.exists():
        meta_path = sample_dir / "meta.json"
    metadata = load_json(meta_path)
    fps = float(metadata.get("fps", metadata.get("video_fps", 12.0)) or 12.0)
    raw = load_raw_state(sample_dir, fps)
    state_raw = raw["state_raw"]
    visibility_mask = raw["visibility_mask"]
    object_ids = raw["object_ids"]
    T = int(state_raw.shape[0])
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
    dt = np.asarray(raw["dt"]).astype(np.float32)
    candidates: list[dict[str, Any]] = []
    for future_len in FUTURE_LENGTHS:
        min_total = CONTEXT_LEN + int(future_len)
        if T < min_total:
            continue
        max_start = T - min_total
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
                "prompt": str(metadata.get("prompt", "")).strip() or str(metadata.get("caption", "")).strip() or "a rigid object motion scene",
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
            record["dataset"] = "movi_d"
            record["pair_meta"] = meta_payload
            record["window_dir"] = str(sample_dir)
            record["_state_raw_full"] = state_raw
            record["_state_norm_full"] = state_norm
            record["_visibility_mask_full"] = visibility_mask
            record["_object_ids"] = object_ids
            record["_seg_ids"] = raw["seg_ids"]
            record["_dt"] = dt
            candidates.append(record)
    return candidates


def decode_movi_record_to_raw_payload(
    *,
    features,
    split: str,
    shard_path: Path,
    record_index: int,
    include_rgb: bool,
) -> dict[str, Any]:
    num_frames = int(features["metadata/num_frames"].int64_list.value[0])
    height = int(features["metadata/height"].int64_list.value[0])
    width = int(features["metadata/width"].int64_list.value[0])
    num_instances = int(features["metadata/num_instances"].int64_list.value[0])
    video_name = decode_text_feature(features["metadata/video_name"])[0]
    background = decode_text_feature(features["background"])[0]
    asset_ids = decode_text_feature(features["instances/asset_id"])
    is_dynamic = np.asarray(features["instances/is_dynamic"].int64_list.value, dtype=np.uint8)
    dynamic_count = int(is_dynamic.sum())
    prompt = build_prompt(background, asset_ids, num_instances, dynamic_count)
    sample_id = f"movi_d_{split}__video_{video_name}"

    rgb_frames = decode_rgb_frames(features["video"].bytes_list.value) if include_rgb else None
    seg_raw = decode_image_sequence(features["segmentations"].bytes_list.value)
    depth_raw = decode_image_sequence(features["depth"].bytes_list.value).astype(np.uint16)
    segmentations = seg_raw.reshape(num_frames, height, width).astype(np.uint8)
    depth_range = np.asarray(features["metadata/depth_range"].float_list.value, dtype=np.float32)
    depth_metric = uint16_to_metric(
        depth_raw.reshape(num_frames, height, width),
        depth_range,
    )

    com_uv_norm = decode_float_tensor(features["instances/image_positions"], (num_instances, num_frames, 2))
    com_uv = np.transpose(com_uv_norm, (1, 0, 2)).astype(np.float32)
    com_uv[..., 0] *= float(width)
    com_uv[..., 1] *= float(height)

    visibility_pixels = decode_int_tensor(
        features["instances/visibility"],
        (num_instances, num_frames),
        dtype=np.int32,
    ).transpose(1, 0)
    visibility_ratio = np.clip(visibility_pixels.astype(np.float32) / float(width * height), 0.0, 1.0)
    visibility_mask = (visibility_pixels > 0).astype(np.uint8)

    bbox_frames_flat = np.asarray(
        features["instances/bbox_frames/ragged_flat_values"].int64_list.value,
        dtype=np.int32,
    )
    bbox_frames_row_lengths = np.asarray(
        features["instances/bbox_frames/ragged_row_lengths_0"].int64_list.value,
        dtype=np.int32,
    )
    bboxes_flat = np.asarray(
        features["instances/bboxes/ragged_flat_values"].float_list.value,
        dtype=np.float32,
    ).reshape(-1, 4)
    bboxes_row_lengths = np.asarray(
        features["instances/bboxes/ragged_row_lengths_0"].int64_list.value,
        dtype=np.int32,
    )
    bbox_xyxy_ragged = ragged_to_frame_boxes(
        bbox_frames_flat=bbox_frames_flat,
        bbox_frames_row_lengths=bbox_frames_row_lengths,
        bboxes_flat=bboxes_flat,
        bboxes_row_lengths=bboxes_row_lengths,
        num_instances=num_instances,
        num_frames=num_frames,
        width=width,
        height=height,
    )
    bbox_xyxy_seg, visibility_mask_seg = boxes_from_segmentations(segmentations, num_instances)
    bbox_xyxy = bbox_xyxy_ragged.copy()
    missing_boxes = (bbox_xyxy[..., 2] <= bbox_xyxy[..., 0]) | (bbox_xyxy[..., 3] <= bbox_xyxy[..., 1])
    bbox_xyxy[missing_boxes] = bbox_xyxy_seg[missing_boxes]
    visibility_mask = np.maximum(visibility_mask, visibility_mask_seg).astype(np.uint8)

    center_depth = center_depth_from_masks(
        depth_metric=depth_metric,
        segmentations=segmentations,
        visibility_pixels=visibility_pixels,
        num_instances=num_instances,
    )
    state_9d = compute_state_9d(
        com_uv=com_uv,
        center_depth=center_depth,
        bbox_xyxy=bbox_xyxy,
        visibility_pixels=visibility_ratio,
        fps=12.0,
    )

    collision_frames = np.asarray(features["events/collisions/frame"].int64_list.value, dtype=np.int32)
    collision_instances = np.asarray(
        features["events/collisions/instances"].int64_list.value,
        dtype=np.int32,
    ).reshape(-1, 2)
    collision_forces = np.asarray(features["events/collisions/force"].float_list.value, dtype=np.float32)
    _contact_graph, _contact_force, _frame_phase, _raw_events, event_windows = build_contact_tensors(
        frames=collision_frames,
        instances=collision_instances,
        forces=collision_forces,
        num_frames=num_frames,
        num_instances=num_instances,
    )

    category_ids = np.asarray(features["instances/category"].int64_list.value, dtype=np.int32)
    scale = np.asarray(features["instances/scale"].float_list.value, dtype=np.float32)
    friction = np.asarray(features["instances/friction"].float_list.value, dtype=np.float32)
    restitution = np.asarray(features["instances/restitution"].float_list.value, dtype=np.float32)
    mass = np.asarray(features["instances/mass"].float_list.value, dtype=np.float32)
    main_object_index = choose_main_object_index(is_dynamic, visibility_pixels)
    objects = []
    for obj_idx in range(num_instances):
        role = "dynamic" if bool(is_dynamic[obj_idx]) else "static"
        objects.append(
            {
                "object_id": int(obj_idx),
                "seg_id": int(obj_idx + 1),
                "role": "primary" if obj_idx == main_object_index else role,
                "motion_type": role,
                "dataset_source": "MOVI-D",
                "source_object_id": str(asset_ids[obj_idx]),
                "name": str(asset_ids[obj_idx]),
                "category": str(category_ids[obj_idx]) if obj_idx < category_ids.shape[0] else "",
                "is_dynamic": bool(is_dynamic[obj_idx]),
                "scale": float(scale[obj_idx]),
                "friction": float(friction[obj_idx]),
                "restitution": float(restitution[obj_idx]),
                "mass": float(mass[obj_idx]),
            }
        )

    focal_length = float(features["camera/focal_length"].float_list.value[0])
    sensor_width = float(features["camera/sensor_width"].float_list.value[0])
    fx = focal_length / sensor_width * float(width) if sensor_width > 0 else float(width)
    metadata = {
        "scene_id": sample_id,
        "prompt": prompt,
        "dataset_source": "MOVI-D",
        "resolution": [int(width), int(height)],
        "camera_intrinsics": {
            "fx": float(fx),
            "fy": float(fx),
            "cx": float(width) / 2.0,
            "cy": float(height) / 2.0,
            "near": float(depth_range[0]),
            "far": float(depth_range[1]),
        },
        "objects": objects,
        "main_object_index": int(main_object_index),
        "source_paths": {
            "tfrecord_path": str(shard_path),
            "tfrecord_record_index": int(record_index),
        },
    }
    return {
        "sample_id": sample_id,
        "metadata": metadata,
        "rgb_frames": rgb_frames,
        "state_raw": state_9d.astype(np.float32),
        "visibility_mask": visibility_mask.astype(np.uint8),
        "object_ids": np.arange(num_instances, dtype=np.int32),
        "seg_ids": np.arange(1, num_instances + 1, dtype=np.int32),
        "dt": np.asarray(1.0 / 12.0, dtype=np.float32),
        "event_windows": event_windows,
    }


def build_strict_candidates_from_movi_payload(
    payload: dict[str, Any],
    *,
    source_sample_dir: str,
    window_dir: str,
) -> list[dict[str, Any]]:
    metadata = dict(payload["metadata"])
    state_raw = np.asarray(payload["state_raw"]).astype(np.float32)
    visibility_mask = np.asarray(payload["visibility_mask"]).astype(np.uint8)
    object_ids = np.asarray(payload["object_ids"]).astype(np.int32)
    seg_ids = np.asarray(payload["seg_ids"]).astype(np.int32)
    T = int(state_raw.shape[0])
    width, height = map(float, metadata["resolution"])
    cam = metadata["camera_intrinsics"]
    state_norm = normalize_state(
        state_raw=state_raw,
        width=width,
        height=height,
        depth_near=float(cam["near"]),
        depth_far=float(cam["far"]),
    )
    main_object_index = int(metadata.get("main_object_index", 0))
    candidates: list[dict[str, Any]] = []
    for future_len in FUTURE_LENGTHS:
        min_total = CONTEXT_LEN + int(future_len)
        if T < min_total:
            continue
        max_start = T - min_total
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
                "prompt": str(metadata.get("prompt", "")).strip() or "a rigid object motion scene",
                "source_scene_id": str(metadata.get("scene_id", payload["sample_id"])),
                "source_sample_dir": str(source_sample_dir),
                "context_len": CONTEXT_LEN,
                "future_len": int(future_len),
                "start_index": int(start),
                "main_object_index": int(main_object_index),
                "future_main_visibility_ratio": float(future_vis_ratio),
                "resolution": metadata.get("resolution"),
                "camera_intrinsics": metadata.get("camera_intrinsics"),
                "objects": metadata.get("objects", []),
                "x_frame_paths": [f"in_memory/frame_{idx:03d}.png" for idx in range(c0, c1)],
                "y_frame_paths": [f"in_memory/frame_{idx:03d}.png" for idx in range(f0, f1)],
                "_in_memory_frames": True,
                "motion_complexity": infer_motion_complexity(
                    state_norm=state_norm[f0:f1].astype(np.float32),
                    visibility_mask=visibility_mask[f0:f1].astype(np.uint8),
                ),
            }
            meta_payload["window_interactions"] = infer_window_interactions(meta_payload)
            record = strict_record_from_meta(meta_payload, Path(window_dir))
            if record is None:
                continue
            record["dataset"] = "movi_d"
            record["pair_meta"] = meta_payload
            record["window_dir"] = str(window_dir)
            record["source_sample_dir"] = str(source_sample_dir)
            record["_state_raw_full"] = state_raw
            record["_state_norm_full"] = state_norm
            record["_visibility_mask_full"] = visibility_mask
            record["_object_ids"] = object_ids
            record["_seg_ids"] = seg_ids
            record["_dt"] = payload["dt"]
            if payload.get("rgb_frames") is not None:
                record["_rgb_frames_full"] = payload["rgb_frames"]
            record["_source_event_windows"] = payload["event_windows"]
            candidates.append(record)
    return candidates


def prepare_movi_test_packages(output_root: Path, skip_cache: bool) -> list[dict[str, Any]]:
    del skip_cache
    targets_by_shard = iter_movi_mytest_targets()
    best_candidates: dict[str, dict[str, Any]] = {}
    selected_by_shard: dict[str, dict[int, str]] = defaultdict(dict)
    meta_by_sample_id: dict[str, tuple[Path, dict[str, Any]]] = {}

    for meta_path in sorted(MOVI_MYTEST_ROOT.glob("*/meta.json")):
        meta = load_json(meta_path)
        sample_id = str(meta.get("sample_id") or meta_path.parent.name)
        meta_by_sample_id[sample_id] = (meta_path, meta)

    for shard_path_str, target_map in sorted(targets_by_shard.items()):
        shard_path = Path(shard_path_str)
        if not shard_path.exists():
            continue
        for record_index, payload in enumerate(iter_serialized_records(shard_path)):
            target_meta = target_map.get(record_index)
            if target_meta is None:
                continue
            features = parse_example(payload)
            sample_id = str(target_meta.get("sample_id") or f"movi_d_test_{record_index:04d}")
            source_sample_dir = movi_target_source_sample_dir(target_meta)
            raw_payload = decode_movi_record_to_raw_payload(
                features=features,
                split="test",
                shard_path=shard_path,
                record_index=record_index,
                include_rgb=False,
            )
            candidates = build_strict_candidates_from_movi_payload(
                raw_payload,
                source_sample_dir=source_sample_dir,
                window_dir=str(Path(source_sample_dir).parent if source_sample_dir else shard_path),
            )
            best = choose_best_record(candidates)
            if best is None:
                continue
            best_candidates[sample_id] = best
            selected_by_shard[str(shard_path)][int(record_index)] = sample_id

    items: list[dict[str, Any]] = []
    for shard_path_str, selected_records in sorted(selected_by_shard.items()):
        shard_path = Path(shard_path_str)
        if not selected_records:
            continue
        for record_index, payload in enumerate(iter_serialized_records(shard_path)):
            sample_id = selected_records.get(record_index)
            if sample_id is None:
                continue
            stored = best_candidates.get(sample_id)
            meta_pair = meta_by_sample_id.get(sample_id)
            if stored is None or meta_pair is None:
                continue
            meta_path, meta = meta_pair
            features = parse_example(payload)
            record = dict(stored)
            record["_rgb_frames_full"] = decode_rgb_frames(features["video"].bytes_list.value)
            items.append(
                export_window_package(
                    record=record,
                    out_dir=output_root / "test" / "movi-d" / sample_id,
                    sample_id=sample_id,
                    split="test",
                    dataset_name="movi-d",
                    sample_label=sample_id,
                    source_meta_json_path=str(meta_path),
                    extra_source_paths={
                        "mytest_meta_json_path": str(meta_path),
                        "mytest_sample_dir": str(meta_path.parent),
                        "tfrecord_path": str((meta.get("source_paths") or {}).get("tfrecord_path", "")),
                        "tfrecord_record_index": int((meta.get("source_paths") or {}).get("tfrecord_record_index", -1)),
                    },
                )
            )
    return items


def build_benchmark_links(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {"fixed24": [], "validation100": []}
    list_specs = [
        ("fixed24", TRAIN0419_ROOT / "benchmark_meta_json_paths_fixed24.txt"),
        ("validation100", TRAIN0419_ROOT / "benchmark_meta_json_paths_validation100.txt"),
    ]
    for group_name, list_path in list_specs:
        target_root = output_root / "benchmark" / group_name
        ensure_dir(target_root)
        rows = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for meta_path_str in rows:
            meta_path = Path(meta_path_str)
            sample_id = meta_path.parent.name
            if "/version_1_genesis_rigid_data_all_cases/mytest/" in meta_path_str:
                dataset_name = "genesis"
            elif "/kubric_tfds_movi-d/mytest/" in meta_path_str:
                dataset_name = "movi-d"
            elif "/mvp-lab-OpenVidHD-0.4M-720p-48fps/mytest_train_eval/" in meta_path_str:
                dataset_name = "openvid"
            else:
                dataset_name = "other"
            out_dir = target_root / dataset_name / sample_id
            ensure_dir(out_dir.parent)
            if out_dir.exists() or out_dir.is_symlink():
                if out_dir.is_symlink() or out_dir.is_file():
                    out_dir.unlink()
                elif out_dir.is_dir():
                    shutil.rmtree(out_dir)
            out_dir.symlink_to(meta_path.parent, target_is_directory=True)
            meta = load_json(meta_path)
            results[group_name].append(
                {
                    "sample_id": str(meta.get("sample_id", sample_id)),
                    "dataset": dataset_name,
                    "split": group_name,
                    "sample_dir": str(out_dir),
                    "rel_dir": str(out_dir.relative_to(output_root)),
                    "caption": str(meta.get("caption", "")),
                    "context_frames": int(meta.get("context_frames", 0) or 0),
                    "future_frames": int(meta.get("future_frames", 0) or 0),
                    "full_frames": int(meta.get("raw_frames", 0) or 0),
                    "context_video_path": str(out_dir / "context_video.mp4"),
                    "future_gt_video_path": str(out_dir / "future_gt_video.mp4"),
                    "full_video_path": str(out_dir / "full_video.mp4"),
                    "meta_json_path": str(out_dir / "meta.json"),
                }
            )
    return results


def build_portal(output_root: Path, sections: dict[str, list[dict[str, Any]]]) -> None:
    def render_cards(items: list[dict[str, Any]]) -> str:
        cards = []
        for item in items:
            rel = html_escape(item["rel_dir"])
            extra_links = [f'<a href="{rel}/meta.json">meta.json</a>']
            if item["split"] in {"train", "test"}:
                extra_links.append(f'<a href="{rel}/pair_meta.json">pair_meta.json</a>')
                extra_links.append(f'<a href="{rel}/state_pair.npz">state_pair.npz</a>')
            cards.append(
                f"""
<article class="sample-card">
  <div class="media-grid">
    <div class="media-box">
      <div class="media-label">Context</div>
      <video controls muted preload="metadata" src="{rel}/context_video.mp4"></video>
    </div>
    <div class="media-box">
      <div class="media-label">Future GT</div>
      <video controls muted preload="metadata" src="{rel}/future_gt_video.mp4"></video>
    </div>
    <div class="media-box">
      <div class="media-label">Full</div>
      <video controls muted preload="metadata" src="{rel}/full_video.mp4"></video>
    </div>
  </div>
  <div class="sample-body">
    <h3>{html_escape(item['sample_id'])}</h3>
    <div class="badge-row">
      <span class="badge">{html_escape(item['dataset'])}</span>
      <span class="badge">{html_escape(item['split'])}</span>
      <span class="badge">ctx {int(item.get('context_frames', 0))}</span>
      <span class="badge">fut {int(item.get('future_frames', 0))}</span>
      <span class="badge">full {int(item.get('full_frames', 0))}</span>
      {f"<span class='badge'>{html_escape(str(item.get('collision_bucket', '')))}</span>" if item.get('collision_bucket') else ""}
      {f"<span class='badge'>{html_escape(str(item.get('motion_complexity', '')))}</span>" if item.get('motion_complexity') else ""}
    </div>
    <p class="caption">{html_escape(item.get('caption', ''))}</p>
    <p class="links">{' | '.join(extra_links)}</p>
  </div>
</article>
"""
            )
        return "".join(cards)

    section_parts = []
    for section_name, items in sections.items():
        section_parts.append(
            f"""
<section class="dataset-section">
  <div class="section-head">
    <h2>{html_escape(section_name)}</h2>
    <span class="count-badge">{len(items)} samples</span>
  </div>
  <div class="sample-grid">
    {render_cards(items)}
  </div>
</section>
"""
        )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Stage1Adapter Portal</title>
  <style>
    :root {{
      --bg: #efe7dd;
      --panel: #fffaf3;
      --panel2: #f5ecdf;
      --ink: #1f1a15;
      --muted: #6f665c;
      --line: #d8cbb9;
      --accent: #8b3a10;
      --accent2: #0f766e;
      --shadow: rgba(42, 28, 16, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(139,58,16,0.09), transparent 24%),
        radial-gradient(circle at top right, rgba(15,118,110,0.09), transparent 20%),
        var(--bg);
    }}
    .page {{ max-width: 1700px; margin: 0 auto; padding: 24px 20px 60px; }}
    .hero, .dataset-section {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 16px 36px var(--shadow);
    }}
    .hero {{ padding: 24px 28px; margin-bottom: 18px; }}
    .hero h1, .dataset-section h2, .sample-body h3 {{ margin: 0; }}
    .hero p, .caption {{ color: var(--muted); }}
    .dataset-section {{ margin-top: 16px; padding: 14px; }}
    .section-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .count-badge {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      color: var(--muted);
      background: rgba(255,255,255,0.7);
      font-size: 12px;
    }}
    .sample-grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .sample-card {{
      display: grid;
      grid-template-columns: minmax(820px, 1.4fr) minmax(280px, 0.6fr);
      gap: 12px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.5);
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .media-box {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px;
      background: rgba(255,255,255,0.6);
    }}
    .media-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    video {{ width: 100%; border-radius: 10px; background: #0e1115; display: block; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }}
    .badge {{
      border: 1px solid #dcc7aa;
      background: rgba(255,255,255,0.78);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: #694d33;
    }}
    .links a {{ color: var(--accent); }}
    @media (max-width: 1180px) {{
      .sample-card {{ grid-template-columns: 1fr; }}
      .media-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Stage1Adapter Strict Simple-Motion Dataset</h1>
      <p>严格口径：仅保留 <code>collision_type_bucket in {{none, env_only}}</code> 且 <code>motion_complexity in {{static, simple}}</code> 的样本。每个样本展示 context video、future GT video、full video 和文本描述。</p>
    </section>
    {''.join(section_parts)}
  </div>
</body>
</html>
"""
    write_json(output_root / "portal_manifest.json", sections)
    (output_root / "index.html").write_text(html_text, encoding="utf-8")


def start_server(output_dir: Path, host: str, port: int) -> tuple[int, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    process = subprocess.Popen(
        [
            "python3",
            "-m",
            "http.server",
            str(port),
            "--bind",
            host,
            "--directory",
            str(output_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid, f"http://{host}:{port}/index.html"


def summarize(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for item in items:
        counts[f"{item['split']}::{item['dataset']}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.only_genesis_test and args.rebuild:
        target = output_root / "test" / "genesis"
        if target.exists():
            shutil.rmtree(target)
        ensure_dir(output_root)
        ensure_dir(output_root / "test")
        train_index = build_train_source_index()
        genesis_test_items = prepare_genesis_test_packages(output_root, train_index)
        write_json(output_root / "manifests" / "test_genesis_items.json", genesis_test_items)
        print(f"output_root={output_root}")
        print(f"mode=only_genesis_test")
        print(f"test_genesis_items={len(genesis_test_items)}")
        return

    if args.rebuild:
        for name in ("train", "test", "benchmark", "manifests", "index.html", "portal_manifest.json"):
            path = output_root / name if not str(name).endswith(".html") and not str(name).endswith(".json") else output_root / name
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    ensure_dir(output_root)
    ensure_dir(output_root / "manifests")

    train_index = build_train_source_index()
    train_items = prepare_train_packages(output_root, train_index)
    genesis_test_items = prepare_genesis_test_packages(output_root, train_index)
    movi_test_items = prepare_movi_test_packages(output_root, skip_cache=bool(args.skip_movi_test_cache))
    benchmark_items = build_benchmark_links(output_root)

    all_train_test = train_items + genesis_test_items + movi_test_items
    write_json(output_root / "manifests" / "train_items.json", train_items)
    write_json(output_root / "manifests" / "test_genesis_items.json", genesis_test_items)
    write_json(output_root / "manifests" / "test_movi_items.json", movi_test_items)
    write_json(output_root / "manifests" / "benchmark_fixed24_items.json", benchmark_items["fixed24"])
    write_json(output_root / "manifests" / "benchmark_validation100_items.json", benchmark_items["validation100"])
    write_json(
        output_root / "manifests" / "summary.json",
        {
            "train_test_counts": summarize(all_train_test),
            "benchmark_counts": {
                "fixed24": len(benchmark_items["fixed24"]),
                "validation100": len(benchmark_items["validation100"]),
            },
        },
    )

    portal_sections = {
        "Train / Genesis": [item for item in train_items if item["dataset"] == "genesis"],
        "Train / MOVI-D": [item for item in train_items if item["dataset"] == "movi-d"],
        "Test / Genesis": genesis_test_items,
        "Test / MOVI-D": movi_test_items,
        "Benchmark / Fixed24": benchmark_items["fixed24"],
        "Benchmark / Validation100": benchmark_items["validation100"],
    }
    build_portal(output_root, portal_sections)
    pid, url = start_server(output_root, str(args.host), int(args.port))
    print(f"output_root={output_root}")
    print(f"pid={pid}")
    print(f"url={url}")
    print(f"train_items={len(train_items)}")
    print(f"test_genesis_items={len(genesis_test_items)}")
    print(f"test_movi_items={len(movi_test_items)}")


if __name__ == "__main__":
    main()
