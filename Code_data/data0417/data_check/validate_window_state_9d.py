#!/usr/bin/env python3
"""Validate window-sample 9D state supervision against dense observations."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ADAPTER_DIR = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/state_adapter")
if str(STATE_ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(STATE_ADAPTER_DIR))

from prepare_movi_d_physics import (
    compute_state_9d,
    decode_float_tensor,
    decode_image_sequence,
    decode_rgb_frames,
    iter_serialized_records,
    parse_example,
    uint16_to_metric,
)


STATE_COLORS = [
    "#ff6b6b",
    "#4dabf7",
    "#51cf66",
    "#f59f00",
    "#845ef7",
    "#e64980",
    "#12b886",
    "#fd7e14",
    "#228be6",
    "#40c057",
    "#fab005",
    "#7950f2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stage1adapter window 9D states.")
    parser.add_argument(
        "--window_roots",
        nargs="+",
        default=[
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/genesis",
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/test/genesis",
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/movi-d",
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/test/movi-d",
        ],
        help="Window dataset roots containing meta.json + segment_state.npz.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/gaoya/Code_Video/Code_data/data0417/data_check/state_validation_window"),
    )
    parser.add_argument(
        "--visualize_count",
        type=int,
        default=8,
        help="Representative cases per dataset bucket to render.",
    )
    parser.add_argument(
        "--max_samples_per_bucket",
        type=int,
        default=0,
        help="Optional cap per dataset bucket. 0 means all.",
    )
    parser.add_argument(
        "--write_state_9d",
        action="store_true",
        help="Write state_9d.npy into each window sample directory.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_num(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def discover_window_samples(window_root: Path) -> List[Path]:
    samples: List[Path] = []
    for meta_path in sorted(window_root.rglob("meta.json")):
        sample_dir = meta_path.parent
        if (sample_dir / "segment_state.npz").exists() and (sample_dir / "physics" / "anchor_targets.npz").exists():
            samples.append(sample_dir)
    return samples


def bucket_name_from_meta(meta: dict, sample_dir: Path) -> str:
    dataset = str(meta.get("dataset", "")).strip().lower()
    split = str(meta.get("split", "")).strip().lower() or sample_dir.parts[-2]
    if "genesis" in dataset:
        source = "genesis"
    elif "movi" in dataset:
        source = "movi_d"
    else:
        source = dataset.replace("-", "_") or "unknown"
    return f"{split}__{source}"


def choose_samples(sample_dirs: Sequence[Path], max_samples: int) -> List[Path]:
    sample_dirs = list(sample_dirs)
    if max_samples <= 0 or len(sample_dirs) <= max_samples:
        return sample_dirs
    return sample_dirs[:max_samples]


def compute_observation_tables(seg: np.ndarray, depth_metric: np.ndarray, seg_ids: np.ndarray) -> dict:
    num_frames, height, width = seg.shape
    num_objects = int(seg_ids.shape[0])
    bbox_xyxy = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    centroid_uv = np.zeros((num_frames, num_objects, 2), dtype=np.float32)
    visibility_mask = np.zeros((num_frames, num_objects), dtype=np.uint8)
    visibility_ratio = np.zeros((num_frames, num_objects), dtype=np.float32)
    center_depth = np.zeros((num_frames, num_objects), dtype=np.float32)
    pixel_norm = float(height * width)

    for obj_idx, seg_id in enumerate(seg_ids.astype(np.int32).tolist()):
        target = int(seg_id)
        for frame_idx in range(num_frames):
            mask = seg[frame_idx] == target
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            visibility_mask[frame_idx, obj_idx] = 1
            visibility_ratio[frame_idx, obj_idx] = float(xs.size) / pixel_norm
            centroid_uv[frame_idx, obj_idx, 0] = float(xs.mean())
            centroid_uv[frame_idx, obj_idx, 1] = float(ys.mean())
            bbox_xyxy[frame_idx, obj_idx] = np.asarray(
                [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0],
                dtype=np.float32,
            )
            center_depth[frame_idx, obj_idx] = float(depth_metric[frame_idx][mask].mean())

    return {
        "bbox_xyxy": bbox_xyxy,
        "centroid_uv": centroid_uv,
        "visibility_mask": visibility_mask,
        "visibility_ratio": visibility_ratio,
        "center_depth": center_depth,
    }


def bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bx2, by2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return (inter / union).astype(np.float32)


def summarize_metric(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p95": float(np.percentile(flat, 95)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    return np.asarray(values)[mask_bool]


def compute_velocity_smoothness(state_9d: np.ndarray, width: int, height: int, depth_scale: float) -> float:
    velocity = state_9d[..., 5:8].astype(np.float32).copy()
    velocity[..., 0] /= max(float(width), 1.0)
    velocity[..., 1] /= max(float(height), 1.0)
    velocity[..., 2] /= max(float(depth_scale), 1e-6)
    if velocity.shape[0] <= 2:
        return 0.0
    accel = np.diff(velocity, axis=0)
    return float(np.mean(np.linalg.norm(accel, axis=-1)))


def load_rgb_frames_from_window(sample_dir: Path) -> List[Image.Image]:
    rgb_dir = sample_dir / "rgb"
    frame_paths = sorted(rgb_dir.glob("*.png"))
    if not frame_paths:
        return []
    return [Image.open(path).convert("RGB") for path in frame_paths]


def resolve_source_sample_dir(meta: dict, sample_dir: Path) -> Path:
    direct = str(meta.get("source_sample_dir", "")).strip()
    if direct:
        return Path(direct)
    source_paths = meta.get("source_paths", {})
    if isinstance(source_paths, dict):
        candidate = str(source_paths.get("source_sample_dir", "")).strip()
        if candidate:
            return Path(candidate)
    pair_meta_path = sample_dir / "pair_meta.json"
    if pair_meta_path.exists():
        pair_meta = load_json(pair_meta_path)
        candidate = str(pair_meta.get("source_sample_dir", "")).strip()
        if candidate:
            return Path(candidate)
        pair_source_paths = pair_meta.get("source_paths", {})
        if isinstance(pair_source_paths, dict):
            candidate = str(pair_source_paths.get("source_sample_dir", "")).strip()
            if candidate:
                return Path(candidate)
    raise KeyError(f"source_sample_dir not found for {sample_dir}")


def resolve_movi_tfrecord_source(meta: dict, sample_dir: Path) -> Tuple[str, int]:
    source_paths = meta.get("source_paths", {})
    if isinstance(source_paths, dict):
        shard_path = str(source_paths.get("tfrecord_path", "")).strip()
        record_index = source_paths.get("tfrecord_record_index", None)
        if shard_path and record_index is not None:
            return shard_path, int(record_index)

    source_sample_dir = resolve_source_sample_dir(meta, sample_dir)
    metadata_path = source_sample_dir / "metadata.json"
    if metadata_path.exists():
        source_meta = load_json(metadata_path)
        source_paths = source_meta.get("source_paths", {})
        if isinstance(source_paths, dict):
            shard_path = str(source_paths.get("tfrecord_path", "")).strip()
            record_index = source_paths.get("tfrecord_record_index", None)
            if shard_path and record_index is not None:
                return shard_path, int(record_index)

    raise KeyError(f"tfrecord source not found for {sample_dir}")


def load_genesis_window_sample(sample_dir: Path) -> dict:
    meta = load_json(sample_dir / "meta.json")
    segment_state = np.load(sample_dir / "segment_state.npz")
    anchor = np.load(sample_dir / "physics" / "anchor_targets.npz")
    source_sample_dir = resolve_source_sample_dir(meta, sample_dir)
    frame_indices = np.asarray(segment_state["frame_indices"]).astype(np.int32)

    raw_seg = np.load(source_sample_dir / "physics" / "seg.npy")
    raw_depth = np.load(source_sample_dir / "physics" / "depth_metric.npy")
    seg = raw_seg[frame_indices]
    depth_metric = raw_depth[frame_indices]
    obs = compute_observation_tables(seg, depth_metric, anchor["seg_ids"])

    fps = float(meta.get("fps", 12.0))
    state_saved = np.asarray(segment_state["state_raw"]).astype(np.float32)
    state_recomputed = compute_state_9d(
        com_uv=np.asarray(anchor["com_uv"]).astype(np.float32),
        center_depth=np.asarray(anchor["center_depth"]).astype(np.float32),
        bbox_xyxy=np.asarray(anchor["bbox_xyxy"]).astype(np.float32),
        visibility_pixels=obs["visibility_ratio"],
        fps=fps,
    )

    rigid_path = source_sample_dir / "physics" / "rigid_kinematics.npz"
    reference_com_uv = np.asarray(anchor["com_uv"]).astype(np.float32)
    kinetic_proxy = np.linalg.norm(state_recomputed[..., 5:8], axis=-1).astype(np.float32)
    energy_total = None
    if rigid_path.exists():
        rigid = np.load(rigid_path)
        if "com_uv" in rigid.files:
            reference_com_uv = np.asarray(rigid["com_uv"])[frame_indices].astype(np.float32)
        if "total_energy" in rigid.files:
            energy_total = np.asarray(rigid["total_energy"])[frame_indices].astype(np.float32)

    return {
        "dataset_slug": "genesis",
        "dataset_label": "GenesisRigid",
        "bucket_name": bucket_name_from_meta(meta, sample_dir),
        "sample_dir": sample_dir,
        "sample_id": sample_dir.name,
        "meta": meta,
        "segment_state": {k: segment_state[k] for k in segment_state.files},
        "anchor": {k: anchor[k] for k in anchor.files},
        "seg": seg,
        "depth_metric": depth_metric,
        "obs": obs,
        "fps": fps,
        "height": int(seg.shape[1]),
        "width": int(seg.shape[2]),
        "frames": load_rgb_frames_from_window(sample_dir),
        "state_saved": state_saved,
        "state_recomputed": state_recomputed,
        "reference_com_uv": reference_com_uv,
        "kinetic_proxy": kinetic_proxy,
        "energy_total": energy_total,
        "source_sample_dir": source_sample_dir,
    }


def collect_movi_feature_map(sample_dirs: Sequence[Path]) -> Dict[str, object]:
    lookup: Dict[Tuple[str, int], str] = {}
    per_shard: Dict[str, set[int]] = {}
    for sample_dir in sample_dirs:
        meta = load_json(sample_dir / "meta.json")
        shard_path, record_index = resolve_movi_tfrecord_source(meta, sample_dir)
        lookup[(shard_path, record_index)] = str(sample_dir)
        per_shard.setdefault(shard_path, set()).add(record_index)

    feature_map: Dict[str, object] = {}
    for shard_path, target_indices in sorted(per_shard.items()):
        remaining = set(int(x) for x in target_indices)
        for record_index, payload in enumerate(iter_serialized_records(Path(shard_path))):
            if record_index not in remaining:
                continue
            feature_map[lookup[(shard_path, record_index)]] = parse_example(payload)
            remaining.remove(record_index)
            if not remaining:
                break
    return feature_map


def load_movi_window_sample(sample_dir: Path, feature_map: Dict[str, object]) -> dict:
    meta = load_json(sample_dir / "meta.json")
    segment_state = np.load(sample_dir / "segment_state.npz")
    anchor = np.load(sample_dir / "physics" / "anchor_targets.npz")
    frame_indices = np.asarray(segment_state["frame_indices"]).astype(np.int32)
    features = feature_map[str(sample_dir)]

    num_frames = int(features["metadata/num_frames"].int64_list.value[0])
    height = int(features["metadata/height"].int64_list.value[0])
    width = int(features["metadata/width"].int64_list.value[0])
    num_objects = int(features["metadata/num_instances"].int64_list.value[0])

    raw_seg = decode_image_sequence(features["segmentations"].bytes_list.value).reshape(num_frames, height, width).astype(
        np.uint8
    )
    depth_raw = decode_image_sequence(features["depth"].bytes_list.value).astype(np.uint16)
    raw_depth = uint16_to_metric(
        depth_raw.reshape(num_frames, height, width),
        np.asarray(features["metadata/depth_range"].float_list.value, dtype=np.float32),
    )
    seg = raw_seg[frame_indices]
    depth_metric = raw_depth[frame_indices]
    obs = compute_observation_tables(seg, depth_metric, anchor["seg_ids"])

    fps = float(meta.get("fps", 12.0))
    state_saved = np.asarray(segment_state["state_raw"]).astype(np.float32)
    state_recomputed = compute_state_9d(
        com_uv=np.asarray(anchor["com_uv"]).astype(np.float32),
        center_depth=np.asarray(anchor["center_depth"]).astype(np.float32),
        bbox_xyxy=np.asarray(anchor["bbox_xyxy"]).astype(np.float32),
        visibility_pixels=obs["visibility_ratio"],
        fps=fps,
    )

    image_positions = decode_float_tensor(features["instances/image_positions"], (num_objects, num_frames, 2))
    reference_com_uv = np.transpose(image_positions, (1, 0, 2)).astype(np.float32)
    reference_com_uv[..., 0] *= float(width)
    reference_com_uv[..., 1] *= float(height)
    reference_com_uv = reference_com_uv[frame_indices]

    velocities = decode_float_tensor(features["instances/velocities"], (num_objects, num_frames, 3))
    velocities = np.transpose(velocities, (1, 0, 2)).astype(np.float32)[frame_indices]
    mass = np.asarray(features["instances/mass"].float_list.value, dtype=np.float32)
    kinetic_proxy = 0.5 * np.square(np.linalg.norm(velocities, axis=-1)).astype(np.float32)
    if mass.shape[0] == kinetic_proxy.shape[1]:
        kinetic_proxy = kinetic_proxy * mass[None, :]

    return {
        "dataset_slug": "movi_d",
        "dataset_label": "MOVI-D",
        "bucket_name": bucket_name_from_meta(meta, sample_dir),
        "sample_dir": sample_dir,
        "sample_id": sample_dir.name,
        "meta": meta,
        "segment_state": {k: segment_state[k] for k in segment_state.files},
        "anchor": {k: anchor[k] for k in anchor.files},
        "seg": seg,
        "depth_metric": depth_metric,
        "obs": obs,
        "fps": fps,
        "height": height,
        "width": width,
        "frames": load_rgb_frames_from_window(sample_dir),
        "state_saved": state_saved,
        "state_recomputed": state_recomputed,
        "reference_com_uv": reference_com_uv,
        "kinetic_proxy": kinetic_proxy,
        "energy_total": None,
        "source_sample_dir": resolve_source_sample_dir(meta, sample_dir),
    }


def compute_sample_metrics(sample: dict) -> dict:
    anchor = sample["anchor"]
    obs = sample["obs"]
    state_saved = sample["state_saved"]
    state_recomputed = sample["state_recomputed"]
    visible = obs["visibility_mask"].astype(bool)
    depth_visible = visible & (obs["center_depth"] > 0.0)

    center_proj_anchor_vs_seg = np.linalg.norm(
        np.asarray(anchor["com_uv"]).astype(np.float32) - obs["centroid_uv"],
        axis=-1,
    ).astype(np.float32)
    center_proj_anchor_vs_ref = np.linalg.norm(
        np.asarray(anchor["com_uv"]).astype(np.float32) - sample["reference_com_uv"],
        axis=-1,
    ).astype(np.float32)
    bbox_iou = bbox_iou_matrix(np.asarray(anchor["bbox_xyxy"]).astype(np.float32), obs["bbox_xyxy"])
    depth_abs_error = np.abs(np.asarray(anchor["center_depth"]).astype(np.float32) - obs["center_depth"]).astype(np.float32)
    depth_rel_error = depth_abs_error / np.maximum(obs["center_depth"], 1e-6)
    state_abs_error = np.abs(state_saved - state_recomputed).astype(np.float32)
    vis_abs_error = np.abs(state_saved[..., 8] - np.asarray(anchor["visibility_mask"]).astype(np.float32)).astype(np.float32)

    du_gt = np.zeros_like(state_saved[..., 5], dtype=np.float32)
    dv_gt = np.zeros_like(state_saved[..., 6], dtype=np.float32)
    dd_gt = np.zeros_like(state_saved[..., 7], dtype=np.float32)
    dt = 1.0 / max(float(sample["fps"]), 1e-6)
    if state_saved.shape[0] > 1:
        du_gt[1:] = (state_saved[1:, :, 0] - state_saved[:-1, :, 0]) / dt
        dv_gt[1:] = (state_saved[1:, :, 1] - state_saved[:-1, :, 1]) / dt
        dd_gt[1:] = (state_saved[1:, :, 2] - state_saved[:-1, :, 2]) / dt
    vel_diff = np.stack(
        [state_saved[..., 5] - du_gt, state_saved[..., 6] - dv_gt, state_saved[..., 7] - dd_gt],
        axis=-1,
    )
    vel_diff_norm = np.linalg.norm(vel_diff, axis=-1).astype(np.float32)
    vel_eval_mask = visible.copy()
    if vel_eval_mask.shape[0] > 0:
        vel_eval_mask[0] = False

    depth_scale = float(np.nanmax(obs["center_depth"][depth_visible])) if np.any(depth_visible) else 1.0
    metrics = {
        "visible_frame_ratio": float(np.mean(visible.astype(np.float32))) if visible.size else 0.0,
        "center_projection_error_px": summarize_metric(masked_values(center_proj_anchor_vs_seg, visible)),
        "projection_consistency_ref_px": summarize_metric(masked_values(center_proj_anchor_vs_ref, visible)),
        "bbox_iou": summarize_metric(masked_values(bbox_iou, visible)),
        "depth_consistency_abs": summarize_metric(masked_values(depth_abs_error, depth_visible)),
        "depth_consistency_rel": summarize_metric(masked_values(depth_rel_error, depth_visible)),
        "state_abs_error": summarize_metric(state_abs_error.reshape(-1)),
        "velocity_diff_error": summarize_metric(masked_values(vel_diff_norm, vel_eval_mask)),
        "vis_abs_error": summarize_metric(masked_values(vis_abs_error, visible)),
        "velocity_smoothness": compute_velocity_smoothness(
            state_recomputed,
            width=sample["width"],
            height=sample["height"],
            depth_scale=depth_scale,
        ),
    }

    anomaly_reasons: List[str] = []
    if metrics["center_projection_error_px"]["mean"] is not None and metrics["center_projection_error_px"]["mean"] > 2.0:
        anomaly_reasons.append("large_center_projection_error")
    if metrics["bbox_iou"]["mean"] is not None and metrics["bbox_iou"]["mean"] < 0.85:
        anomaly_reasons.append("low_bbox_iou")
    if metrics["depth_consistency_rel"]["mean"] is not None and metrics["depth_consistency_rel"]["mean"] > 0.08:
        anomaly_reasons.append("high_depth_rel_error")
    if metrics["velocity_diff_error"]["mean"] is not None and metrics["velocity_diff_error"]["mean"] > 1e-3:
        anomaly_reasons.append("velocity_diff_mismatch")
    metrics["anomaly"] = bool(anomaly_reasons)
    metrics["anomaly_reasons"] = anomaly_reasons
    return metrics


def aggregate_dataset_metrics(rows: Sequence[dict]) -> dict:
    def collect(metric_name: str, field: str) -> List[float]:
        values: List[float] = []
        for row in rows:
            value = row["metrics"].get(metric_name, {}).get(field)
            if value is not None:
                values.append(float(value))
        return values

    def stat(values: Sequence[float]) -> dict:
        array = np.asarray(list(values), dtype=np.float32)
        return summarize_metric(array) if array.size else summarize_metric(np.asarray([], dtype=np.float32))

    summary = {
        "num_samples": int(len(rows)),
        "anomaly_ratio": float(sum(1 for row in rows if row["metrics"]["anomaly"]) / max(len(rows), 1)),
        "visible_frame_ratio": stat([row["metrics"]["visible_frame_ratio"] for row in rows]),
        "center_projection_error_px": stat(collect("center_projection_error_px", "mean")),
        "projection_consistency_ref_px": stat(collect("projection_consistency_ref_px", "mean")),
        "bbox_iou": stat(collect("bbox_iou", "mean")),
        "depth_consistency_abs": stat(collect("depth_consistency_abs", "mean")),
        "depth_consistency_rel": stat(collect("depth_consistency_rel", "mean")),
        "state_abs_error": stat(collect("state_abs_error", "mean")),
        "velocity_diff_error": stat(collect("velocity_diff_error", "mean")),
        "vis_abs_error": stat(collect("vis_abs_error", "mean")),
        "velocity_smoothness": stat([row["metrics"]["velocity_smoothness"] for row in rows]),
    }
    hist: Dict[str, int] = {}
    for row in rows:
        for reason in row["metrics"]["anomaly_reasons"]:
            hist[reason] = int(hist.get(reason, 0)) + 1
    summary["anomaly_reason_histogram"] = dict(sorted(hist.items()))
    return summary


def compute_risk_score(metrics: dict) -> float:
    center_penalty = float(metrics["center_projection_error_px"]["mean"] or 0.0) / 5.0
    bbox_penalty = 1.0 - float(metrics["bbox_iou"]["mean"] or 1.0)
    depth_penalty = float(metrics["depth_consistency_rel"]["mean"] or 0.0) * 4.0
    velocity_penalty = min(float(metrics["velocity_diff_error"]["mean"] or 0.0) * 1000.0, 5.0)
    return float(center_penalty + bbox_penalty * 10.0 + depth_penalty + velocity_penalty)


def select_visual_rows(rows: Sequence[dict], count: int) -> List[dict]:
    if count <= 0 or not rows:
        return []
    ranked = sorted(rows, key=lambda row: (-row["risk_score"], row["sample_id"]))
    if len(ranked) <= count:
        return ranked
    picks: List[dict] = []
    picks.extend(ranked[: max(1, count // 2)])
    picks.append(ranked[len(ranked) // 2])
    picks.append(ranked[-1])
    dedup: List[dict] = []
    seen = set()
    for row in picks + ranked:
        if row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        dedup.append(row)
        if len(dedup) >= count:
            break
    return dedup


def save_mp4(frames: Sequence[Image.Image], dst: Path, fps: float) -> None:
    if not frames:
        return
    first = np.asarray(frames[0].convert("RGB"))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {dst}")
    try:
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def save_gif(frames: Sequence[Image.Image], dst: Path, duration_ms: int = 120) -> None:
    if not frames:
        return
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def draw_overlay_frames(sample: dict) -> List[Image.Image]:
    frames = sample["frames"]
    seg = sample["seg"]
    anchor = sample["anchor"]
    obs = sample["obs"]
    result: List[Image.Image] = []
    num_objects = int(np.asarray(anchor["seg_ids"]).shape[0])

    for frame_idx, frame in enumerate(frames):
        image = frame.copy().convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for obj_idx in range(num_objects):
            color = hex_to_rgb(STATE_COLORS[obj_idx % len(STATE_COLORS)])
            mask = seg[frame_idx] == int(anchor["seg_ids"][obj_idx])
            if np.any(mask):
                ys, xs = np.nonzero(mask)
                draw.rectangle(
                    (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                    outline=color + (140,),
                    width=2,
                )
            x1, y1, x2, y2 = [float(x) for x in anchor["bbox_xyxy"][frame_idx, obj_idx]]
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)
            u, v = [float(x) for x in anchor["com_uv"][frame_idx, obj_idx]]
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=color + (255,))
            u2, v2 = [float(x) for x in obs["centroid_uv"][frame_idx, obj_idx]]
            draw.ellipse((u2 - 3, v2 - 3, u2 + 3, v2 + 3), fill=(255, 255, 255, 230))
        draw.rounded_rectangle((8, 8, 160, 36), radius=6, fill=(0, 0, 0, 170))
        draw.text((14, 14), f"frame {frame_idx:02d}", fill=(255, 255, 255, 255))
        result.append(Image.alpha_composite(image, overlay).convert("RGB"))
    return result


def save_strip(frames: Sequence[Image.Image], dst: Path, frame_count: int = 6, thumb_height: int = 180) -> None:
    if not frames:
        return
    if len(frames) <= frame_count:
        indices = list(range(len(frames)))
    else:
        indices = np.linspace(0, len(frames) - 1, frame_count).round().astype(int).tolist()
    thumbs: List[Image.Image] = []
    for idx in indices:
        frame = frames[idx]
        scale = thumb_height / float(frame.height)
        size = (max(1, int(round(frame.width * scale))), thumb_height)
        thumbs.append(frame.resize(size, Image.Resampling.BILINEAR))
    total_width = sum(frame.width for frame in thumbs) + 8 * max(0, len(thumbs) - 1)
    canvas = Image.new("RGB", (total_width, thumb_height), color=(18, 18, 20))
    cursor = 0
    for thumb in thumbs:
        canvas.paste(thumb, (cursor, 0))
        cursor += thumb.width + 8
    canvas.save(dst)


def save_curve_plot(sample: dict, out_path: Path) -> None:
    state = sample["state_saved"]
    obs = sample["obs"]
    num_frames, num_objects = state.shape[:2]
    t = np.arange(num_frames, dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for obj_idx in range(num_objects):
        color = STATE_COLORS[obj_idx % len(STATE_COLORS)]
        speed = np.linalg.norm(state[:, obj_idx, 5:8], axis=-1)
        depth = state[:, obj_idx, 2]
        bbox_area = np.maximum(state[:, obj_idx, 3], 0.0) * np.maximum(state[:, obj_idx, 4], 0.0)
        vis = obs["visibility_ratio"][:, obj_idx]
        axes[0, 0].plot(t, speed, color=color, linewidth=2, alpha=0.9, label=f"obj {obj_idx}")
        axes[0, 1].plot(t, depth, color=color, linewidth=2, alpha=0.9)
        axes[1, 0].plot(t, bbox_area, color=color, linewidth=2, alpha=0.9)
        axes[1, 1].plot(t, vis, color=color, linewidth=2, alpha=0.9)
    if sample["energy_total"] is not None:
        ax_energy = axes[0, 0].twinx()
        ax_energy.plot(t, sample["energy_total"], color="#111111", linewidth=2, linestyle="--", label="total energy")
        ax_energy.set_ylabel("total energy")
    elif sample["kinetic_proxy"] is not None:
        proxy = np.sum(sample["kinetic_proxy"], axis=1) if sample["kinetic_proxy"].ndim == 2 else sample["kinetic_proxy"]
        ax_energy = axes[0, 0].twinx()
        ax_energy.plot(t, proxy, color="#111111", linewidth=2, linestyle="--", label="kinetic proxy")
        ax_energy.set_ylabel("kinetic proxy")
    axes[0, 0].set_title("Velocity magnitude")
    axes[0, 1].set_title("Center depth")
    axes[1, 0].set_title("BBox area")
    axes[1, 1].set_title("Visibility ratio")
    for ax in axes.reshape(-1):
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(loc="upper left", ncol=min(3, num_objects))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_compare_plot(sample: dict, metrics: dict, out_path: Path) -> None:
    anchor = sample["anchor"]
    obs = sample["obs"]
    state_saved = sample["state_saved"]
    num_frames, num_objects = state_saved.shape[:2]
    t = np.arange(num_frames, dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    dt = 1.0 / max(float(sample["fps"]), 1e-6)
    du_gt = np.zeros_like(state_saved[..., 5], dtype=np.float32)
    dv_gt = np.zeros_like(state_saved[..., 6], dtype=np.float32)
    dd_gt = np.zeros_like(state_saved[..., 7], dtype=np.float32)
    if num_frames > 1:
        du_gt[1:] = (state_saved[1:, :, 0] - state_saved[:-1, :, 0]) / dt
        dv_gt[1:] = (state_saved[1:, :, 1] - state_saved[:-1, :, 1]) / dt
        dd_gt[1:] = (state_saved[1:, :, 2] - state_saved[:-1, :, 2]) / dt
    for obj_idx in range(num_objects):
        color = STATE_COLORS[obj_idx % len(STATE_COLORS)]
        center_error = np.linalg.norm(anchor["com_uv"][:, obj_idx] - obs["centroid_uv"][:, obj_idx], axis=-1)
        bbox_iou = bbox_iou_matrix(anchor["bbox_xyxy"][:, obj_idx], obs["bbox_xyxy"][:, obj_idx])
        depth_error = np.abs(anchor["center_depth"][:, obj_idx] - obs["center_depth"][:, obj_idx])
        velocity_error = np.linalg.norm(
            np.stack(
                [
                    state_saved[:, obj_idx, 5] - du_gt[:, obj_idx],
                    state_saved[:, obj_idx, 6] - dv_gt[:, obj_idx],
                    state_saved[:, obj_idx, 7] - dd_gt[:, obj_idx],
                ],
                axis=-1,
            ),
            axis=-1,
        )
        if velocity_error.shape[0] > 0:
            velocity_error[0] = 0.0
        axes[0, 0].plot(t, center_error, color=color, linewidth=2, alpha=0.9, label=f"obj {obj_idx}")
        axes[0, 1].plot(t, bbox_iou, color=color, linewidth=2, alpha=0.9)
        axes[1, 0].plot(t, depth_error, color=color, linewidth=2, alpha=0.9)
        axes[1, 1].plot(t, velocity_error, color=color, linewidth=2, alpha=0.9)
    axes[0, 0].set_title("Center projection error vs seg (px)")
    axes[0, 1].set_title("BBox IoU vs seg")
    axes[1, 0].set_title("Depth error vs depth map")
    axes[1, 1].set_title("Velocity error vs finite diff")
    axes[0, 1].set_ylim(-0.02, 1.02)
    for ax in axes.reshape(-1):
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", ncol=min(3, num_objects))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def metrics_table_html(metrics: dict) -> str:
    names = [
        "center_projection_error_px",
        "projection_consistency_ref_px",
        "bbox_iou",
        "depth_consistency_abs",
        "depth_consistency_rel",
        "state_abs_error",
        "velocity_diff_error",
        "vis_abs_error",
    ]
    rows = []
    for name in names:
        values = metrics.get(name, {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{fmt_num(values.get('mean'))}</td>"
            f"<td>{fmt_num(values.get('median'))}</td>"
            f"<td>{fmt_num(values.get('p95'))}</td>"
            "</tr>"
        )
    rows.append(
        "<tr>"
        "<td>velocity_smoothness</td>"
        f"<td>{metrics['velocity_smoothness']:.6f}</td>"
        "<td>n/a</td><td>n/a</td>"
        "</tr>"
    )
    rows.append(
        "<tr>"
        "<td>anomaly</td>"
        f"<td>{str(metrics['anomaly'])}</td>"
        f"<td colspan='2'>{html.escape(', '.join(metrics['anomaly_reasons']) or 'none')}</td>"
        "</tr>"
    )
    return (
        "<table><thead><tr><th>metric</th><th>mean</th><th>median</th><th>p95</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_case_page(sample: dict, metrics: dict, out_dir: Path, rank_label: str) -> dict:
    ensure_dir(out_dir)
    overlay_frames = draw_overlay_frames(sample)
    overlay_mp4 = "overlay.mp4"
    overlay_gif = "overlay.gif"
    strip_png = "trajectory_strip.png"
    curves_png = "curves.png"
    compare_png = "comparisons.png"
    save_mp4(overlay_frames, out_dir / overlay_mp4, fps=sample["fps"])
    save_gif(overlay_frames, out_dir / overlay_gif, duration_ms=max(50, int(round(1000.0 / max(sample["fps"], 1.0)))))
    save_strip(overlay_frames, out_dir / strip_png)
    save_curve_plot(sample, out_dir / curves_png)
    save_compare_plot(sample, metrics, out_dir / compare_png)
    np.save(out_dir / "state_9d.npy", sample["state_recomputed"].astype(np.float32))

    summary_payload = {
        "sample_id": sample["sample_id"],
        "sample_dir": str(sample["sample_dir"]),
        "source_sample_dir": str(sample["source_sample_dir"]),
        "dataset": sample["dataset_label"],
        "rank_label": rank_label,
        "metrics": metrics,
        "caption": sample["meta"].get("caption", ""),
        "detail_caption": sample["meta"].get("detail_caption", ""),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(sample["dataset_label"])} | {html.escape(sample["sample_id"])}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: linear-gradient(180deg, #f6f3eb 0%, #efe9db 100%);
      color: #1f2933;
    }}
    .wrap {{ max-width: 1460px; margin: 0 auto; }}
    .hero, .card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(17,24,39,0.08);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 10px 28px rgba(15,23,42,0.07);
    }}
    .hero {{ margin-bottom: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    img, video {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(17,24,39,0.08);
      background: #111827;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid rgba(17,24,39,0.08); padding: 8px 10px; text-align: left; }}
    code {{ background: rgba(17,24,39,0.06); padding: 2px 6px; border-radius: 6px; word-break: break-all; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div><a href="../index.html">Back to dataset summary</a></div>
      <h1>{html.escape(sample["dataset_label"])} | {html.escape(sample["sample_id"])}</h1>
      <p>{html.escape(rank_label)} sample</p>
      <p>Window sample: <code>{html.escape(str(sample["sample_dir"]))}</code></p>
      <p>Source sample: <code>{html.escape(str(sample["source_sample_dir"]))}</code></p>
      <p><strong>Caption:</strong> {html.escape(str(sample["meta"].get("caption", "")))}</p>
      <p><strong>Detail Caption:</strong> {html.escape(str(sample["meta"].get("detail_caption", "")))}</p>
    </section>
    <section class="grid">
      <section class="card">
        <h2>Trajectory Overlay Video</h2>
        <video src="{overlay_mp4}" controls preload="metadata"></video>
      </section>
      <section class="card">
        <h2>Trajectory Overlay GIF</h2>
        <img src="{overlay_gif}" alt="overlay gif">
      </section>
      <section class="card">
        <h2>Trajectory Strip</h2>
        <img src="{strip_png}" alt="trajectory strip">
      </section>
      <section class="card">
        <h2>State Curves</h2>
        <img src="{curves_png}" alt="state curves">
      </section>
      <section class="card">
        <h2>Consistency Curves</h2>
        <img src="{compare_png}" alt="compare curves">
      </section>
      <section class="card" style="grid-column: 1 / -1;">
        <h2>Metrics</h2>
        {metrics_table_html(metrics)}
      </section>
    </section>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
    return {
        "sample_id": sample["sample_id"],
        "sample_dir": str(sample["sample_dir"]),
        "rank_label": rank_label,
        "page_rel": f"cases/{out_dir.name}/index.html",
        "metrics": metrics,
        "caption": str(sample["meta"].get("caption", "")),
    }


def render_dataset_portal(dataset_label: str, dataset_slug: str, dataset_summary: dict, case_cards: Sequence[dict], out_dir: Path) -> None:
    ensure_dir(out_dir)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cards_html = []
    for card in case_cards:
        metrics = card["metrics"]
        cards_html.append(
            f"""
<article class="case-card">
  <div class="case-top">
    <div>
      <h3>{html.escape(card['sample_id'])}</h3>
      <p>{html.escape(card['rank_label'])}</p>
    </div>
    <a href="{html.escape(card['page_rel'])}">Open</a>
  </div>
  <p>{html.escape(card['caption'])}</p>
  <p><code>{html.escape(card['sample_dir'])}</code></p>
  <ul>
    <li>center err: {fmt_num(metrics['center_projection_error_px']['mean'])}</li>
    <li>bbox IoU: {fmt_num(metrics['bbox_iou']['mean'])}</li>
    <li>depth rel err: {fmt_num(metrics['depth_consistency_rel']['mean'])}</li>
    <li>velocity diff err: {fmt_num(metrics['velocity_diff_error']['mean'])}</li>
    <li>anomaly: {str(metrics['anomaly'])} ({html.escape(', '.join(metrics['anomaly_reasons']) or 'none')})</li>
  </ul>
</article>
"""
        )

    summary = dataset_summary["aggregate"]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(dataset_label)} Window State Validation</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: radial-gradient(circle at top left, #e0f2fe 0%, #f8fafc 45%, #f4f1ea 100%);
      color: #14213d;
    }}
    .wrap {{ max-width: 1480px; margin: 0 auto; }}
    .hero, .metric-card, .case-card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.07);
    }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .metric-card {{ padding: 18px; }}
    .cases {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
    }}
    .case-card {{ padding: 18px; }}
    .case-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    a {{
      color: #0f766e;
      font-weight: 700;
      text-decoration: none;
    }}
    code {{
      background: rgba(17,24,39,0.06);
      padding: 2px 6px;
      border-radius: 6px;
      word-break: break-all;
    }}
    ul {{ padding-left: 18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div><a href="../index.html">Back to all datasets</a></div>
      <h1>{html.escape(dataset_label)} Window State Validation</h1>
      <p>samples: {dataset_summary['num_samples']} | anomaly ratio: {summary['anomaly_ratio']:.6f}</p>
      <p>{html.escape(dataset_summary['conclusion'])}</p>
      <p>Summary file: <code>{html.escape(str(summary_path))}</code></p>
    </section>
    <section class="metric-grid">
      <div class="metric-card"><strong>Center Projection Error</strong><br>{fmt_num(summary['center_projection_error_px']['mean'])}</div>
      <div class="metric-card"><strong>Projection Ref Consistency</strong><br>{fmt_num(summary['projection_consistency_ref_px']['mean'])}</div>
      <div class="metric-card"><strong>BBox IoU</strong><br>{fmt_num(summary['bbox_iou']['mean'])}</div>
      <div class="metric-card"><strong>Depth Rel Error</strong><br>{fmt_num(summary['depth_consistency_rel']['mean'])}</div>
      <div class="metric-card"><strong>Velocity Diff Error</strong><br>{fmt_num(summary['velocity_diff_error']['mean'])}</div>
      <div class="metric-card"><strong>Velocity Smoothness</strong><br>{fmt_num(summary['velocity_smoothness']['mean'])}</div>
    </section>
    <section class="cases">
      {''.join(cards_html)}
    </section>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def render_root_index(dataset_entries: Sequence[dict], out_root: Path) -> None:
    cards = []
    for entry in dataset_entries:
        agg = entry["summary"]["aggregate"]
        cards.append(
            f"""
<article class="card">
  <h2>{html.escape(entry['dataset_label'])}</h2>
  <p>samples: {entry['summary']['num_samples']}</p>
  <p>center err: {fmt_num(agg['center_projection_error_px']['mean'])}</p>
  <p>bbox IoU: {fmt_num(agg['bbox_iou']['mean'])}</p>
  <p>depth rel err: {fmt_num(agg['depth_consistency_rel']['mean'])}</p>
  <p>velocity diff err: {fmt_num(agg['velocity_diff_error']['mean'])}</p>
  <p>anomaly ratio: {agg['anomaly_ratio']:.6f}</p>
  <a href="{html.escape(entry['dataset_slug'])}/index.html">Open dataset report</a>
</article>
"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Window State Validation Portal</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 45%, #f5efe0 100%);
      color: #14213d;
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; }}
    .hero, .card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 18px;
      box-shadow: 0 10px 24px rgba(15,23,42,0.06);
    }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .card {{ padding: 20px; }}
    a {{
      color: #0f766e;
      font-weight: 700;
      text-decoration: none;
    }}
    code {{
      background: rgba(17,24,39,0.06);
      padding: 2px 6px;
      border-radius: 6px;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Window 9D State Validation</h1>
      <p>Validate state_9d supervision on stage1adapter window samples using RGB, seg, depth, anchor_targets and rigid/TFRecord reference trajectories.</p>
      <p>Output root: <code>{html.escape(str(out_root))}</code></p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""
    (out_root / "index.html").write_text(html_text, encoding="utf-8")


def write_sample_csv(rows: Sequence[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fieldnames = [
        "sample_id",
        "sample_dir",
        "risk_score",
        "visible_frame_ratio",
        "center_projection_error_px_mean",
        "projection_consistency_ref_px_mean",
        "bbox_iou_mean",
        "depth_consistency_abs_mean",
        "depth_consistency_rel_mean",
        "state_abs_error_mean",
        "velocity_diff_error_mean",
        "vis_abs_error_mean",
        "velocity_smoothness",
        "anomaly",
        "anomaly_reasons",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "sample_dir": row["sample_dir"],
                    "risk_score": f"{row['risk_score']:.6f}",
                    "visible_frame_ratio": f"{metrics['visible_frame_ratio']:.6f}",
                    "center_projection_error_px_mean": "" if metrics["center_projection_error_px"]["mean"] is None else f"{metrics['center_projection_error_px']['mean']:.6f}",
                    "projection_consistency_ref_px_mean": "" if metrics["projection_consistency_ref_px"]["mean"] is None else f"{metrics['projection_consistency_ref_px']['mean']:.6f}",
                    "bbox_iou_mean": "" if metrics["bbox_iou"]["mean"] is None else f"{metrics['bbox_iou']['mean']:.6f}",
                    "depth_consistency_abs_mean": "" if metrics["depth_consistency_abs"]["mean"] is None else f"{metrics['depth_consistency_abs']['mean']:.6f}",
                    "depth_consistency_rel_mean": "" if metrics["depth_consistency_rel"]["mean"] is None else f"{metrics['depth_consistency_rel']['mean']:.6f}",
                    "state_abs_error_mean": "" if metrics["state_abs_error"]["mean"] is None else f"{metrics['state_abs_error']['mean']:.6f}",
                    "velocity_diff_error_mean": "" if metrics["velocity_diff_error"]["mean"] is None else f"{metrics['velocity_diff_error']['mean']:.6f}",
                    "vis_abs_error_mean": "" if metrics["vis_abs_error"]["mean"] is None else f"{metrics['vis_abs_error']['mean']:.6f}",
                    "velocity_smoothness": f"{metrics['velocity_smoothness']:.6f}",
                    "anomaly": str(metrics["anomaly"]),
                    "anomaly_reasons": ",".join(metrics["anomaly_reasons"]),
                }
            )


def build_conclusion(dataset_label: str, aggregate: dict) -> str:
    bbox = aggregate["bbox_iou"]["mean"]
    center = aggregate["center_projection_error_px"]["mean"]
    depth_rel = aggregate["depth_consistency_rel"]["mean"]
    vel = aggregate["velocity_diff_error"]["mean"]
    anomaly_ratio = aggregate["anomaly_ratio"]
    if bbox is None or center is None or depth_rel is None or vel is None:
        return f"{dataset_label}: insufficient data."
    if bbox >= 0.95 and center <= 1.0 and depth_rel <= 0.03 and vel <= 1e-4 and anomaly_ratio <= 0.05:
        return (
            f"{dataset_label}: 9D state supervision is reliable on current window samples. "
            f"Projected centers, bbox, depth and finite-difference velocities are all consistent with dense observations."
        )
    return (
        f"{dataset_label}: supervision is broadly usable, but flagged outliers remain. "
        f"Inspect low-IoU, high-depth-error or velocity-mismatch samples in the portal."
    )


def maybe_write_state_9d(sample_dir: Path, state_9d: np.ndarray) -> None:
    np.save(sample_dir / "state_9d.npy", state_9d.astype(np.float32))


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)

    bucket_to_samples: Dict[str, List[Path]] = {}
    for root_str in args.window_roots:
        root = Path(root_str)
        for sample_dir in discover_window_samples(root):
            meta = load_json(sample_dir / "meta.json")
            bucket = bucket_name_from_meta(meta, sample_dir)
            bucket_to_samples.setdefault(bucket, []).append(sample_dir)

    bucket_to_samples = {bucket: choose_samples(paths, args.max_samples_per_bucket) for bucket, paths in sorted(bucket_to_samples.items())}

    movi_sample_dirs: List[Path] = []
    for bucket, paths in bucket_to_samples.items():
        if bucket.endswith("__movi_d"):
            movi_sample_dirs.extend(paths)
    movi_feature_map = collect_movi_feature_map(movi_sample_dirs)

    dataset_entries: List[dict] = []
    root_summary = {"datasets": []}

    for bucket, sample_dirs in bucket_to_samples.items():
        rows: List[dict] = []
        sample_cache: Dict[str, dict] = {}
        for sample_dir in sample_dirs:
            meta = load_json(sample_dir / "meta.json")
            dataset = str(meta.get("dataset", "")).lower()
            if "movi" in dataset:
                sample = load_movi_window_sample(sample_dir, movi_feature_map)
            else:
                sample = load_genesis_window_sample(sample_dir)
            if args.write_state_9d:
                maybe_write_state_9d(sample_dir, sample["state_recomputed"])
            metrics = compute_sample_metrics(sample)
            row = {
                "sample_id": sample["sample_id"],
                "sample_dir": str(sample["sample_dir"]),
                "metrics": metrics,
                "risk_score": compute_risk_score(metrics),
            }
            rows.append(row)
            sample_cache[str(sample_dir)] = sample

        dataset_slug = bucket
        dataset_label = bucket.replace("__", " | ")
        aggregate = aggregate_dataset_metrics(rows)
        conclusion = build_conclusion(dataset_label, aggregate)
        dataset_summary = {
            "dataset_slug": dataset_slug,
            "dataset_label": dataset_label,
            "num_samples": int(len(rows)),
            "aggregate": aggregate,
            "conclusion": conclusion,
        }
        bucket_out_dir = args.output_root / dataset_slug
        ensure_dir(bucket_out_dir)
        write_sample_csv(rows, bucket_out_dir / "sample_metrics.csv")

        visual_rows = select_visual_rows(rows, args.visualize_count)
        case_cards: List[dict] = []
        for idx, row in enumerate(visual_rows):
            sample = sample_cache[row["sample_dir"]]
            rank_label = "worst-ranked" if idx == 0 else ("best-ranked" if idx == len(visual_rows) - 1 else "representative")
            case_dir = bucket_out_dir / "cases" / f"{idx:02d}_{sample['sample_id']}"
            case_cards.append(render_case_page(sample, row["metrics"], case_dir, rank_label))

        render_dataset_portal(dataset_label, dataset_slug, dataset_summary, case_cards, bucket_out_dir)
        dataset_entries.append({"dataset_slug": dataset_slug, "dataset_label": dataset_label, "summary": dataset_summary})
        root_summary["datasets"].append(dataset_summary)

    render_root_index(dataset_entries, args.output_root)
    root_summary["portal_path"] = str(args.output_root / "index.html")
    root_summary["portal_url"] = f"http://127.0.0.1:8048{args.output_root / 'index.html'}"
    (args.output_root / "summary.json").write_text(json.dumps(root_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(root_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
