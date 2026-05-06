#!/usr/bin/env python3
"""Compute minimal per-case geometry diagnostics for benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parent
TRAIN0419_ROOT = SCRIPT_ROOT.parent
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

import prepare_mixed_benchmark_mytest as pm  # noqa: E402


BACKGROUND_SEGMENT_ID = 0
GENESIS_BACKGROUND_SEGMENT_ID = 0
GENESIS_GROUND_SPECIAL_ID = -1


@dataclass
class TargetSpec:
    object_label: str
    seg_id: int
    selection_mode: str
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze geometry drift on benchmark sidecars.")
    parser.add_argument(
        "--sidecar",
        type=Path,
        action="append",
        default=[],
        help="Path to one sidecar json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sidecar_dir",
        type=Path,
        default=None,
        help="Directory containing output sidecars.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of sidecars to process.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Root directory for diagnostics outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing diagnostics outputs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def list_sidecars(args: argparse.Namespace) -> list[Path]:
    sidecars = [path.expanduser().resolve() for path in args.sidecar]
    if args.sidecar_dir is not None:
        sidecars.extend(sorted(args.sidecar_dir.expanduser().resolve().glob("*.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in sidecars:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if args.limit is not None:
        unique = unique[: max(int(args.limit), 0)]
    return unique


def sanitize_token(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("._")
    return safe or "item"


def load_video_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path))
    try:
        return [np.asarray(frame, dtype=np.uint8) for frame in reader]
    finally:
        reader.close()


def save_video_frames(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=max(int(fps), 1),
        codec="libx264",
        quality=6,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def resize_map(image_like: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(image_like)
    image = Image.fromarray(array)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized)


def resize_bbox(
    bbox: tuple[float, float, float, float],
    *,
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    sx = float(dst_width) / float(max(src_width, 1))
    sy = float(dst_height) / float(max(src_height, 1))
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def compute_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def mask_fill_ratio(mask_area: int, bbox_area_value: int) -> float:
    if bbox_area_value <= 0:
        return 0.0
    return float(mask_area) / float(bbox_area_value)


def mask_median_depth(depth_map: np.ndarray, mask: np.ndarray) -> float | None:
    values = np.asarray(depth_map, dtype=np.float32)[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def grayscale(frame: np.ndarray) -> np.ndarray:
    frame_f = np.asarray(frame, dtype=np.float32)
    return (0.299 * frame_f[..., 0] + 0.587 * frame_f[..., 1] + 0.114 * frame_f[..., 2]).astype(np.float32)


def draw_rect(frame: np.ndarray, bbox: tuple[int, int, int, int] | None, color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    if bbox is None:
        return frame
    out = np.asarray(frame, dtype=np.uint8).copy()
    x1, y1, x2, y2 = bbox
    h, w = out.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return out
    t = max(int(thickness), 1)
    out[y1 : min(y1 + t, h), x1:x2] = color
    out[max(y2 - t, 0) : y2, x1:x2] = color
    out[y1:y2, x1 : min(x1 + t, w)] = color
    out[y1:y2, max(x2 - t, 0) : x2] = color
    return out


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = np.asarray(frame, dtype=np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def draw_anchor_window(frame: np.ndarray, anchor_bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    return draw_rect(frame, anchor_bbox, color=(255, 210, 0), thickness=2)


def build_case_analysis_text(
    *,
    sidecar: dict[str, Any],
    summary: dict[str, Any],
    target: TargetSpec,
    mode: str,
    gt_available: bool,
) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"Target `{target.object_label}` was selected with `{target.selection_mode}` under mode `{mode}`."
    )
    lines.append(
        "Generated-video curves are computed from a target-window foreground proxy, so they are useful for trend inspection but not yet strict instance-accurate measurements."
    )
    if gt_available:
        lines.append(
            "GT reference curves use synthetic segmentation/depth and provide a stronger baseline for whether the generated size change looks physically plausible."
        )
    root_cause = str(summary.get("root_cause") or "")
    max_area = float(summary.get("max_future_mask_area_ratio") or 0.0)
    max_invariant = float(summary.get("max_future_area_depth2_ratio") or 0.0)
    mean_bg_scale = summary.get("mean_bg_scale")
    if root_cause == "camera_zoom_or_drift":
        lines.append(
            f"Background scale is elevated (`mean_bg_scale={mean_bg_scale}`), so the current heuristic attributes the size change more to camera zoom/drift than object-only growth."
        )
    elif root_cause == "object_scale_drift":
        lines.append(
            f"Foreground area grows strongly (`max_area_ratio={max_area:.3f}`) while background scale stays near 1 (`mean_bg_scale={mean_bg_scale}`), so the current heuristic flags object-only scale drift."
        )
        lines.append(
            f"The projected-size invariant also drifts heavily (`max_area_depth2_ratio={max_invariant:.3f}`), which argues against a pure perspective explanation."
        )
    elif root_cause == "tracking_or_visibility_failure":
        lines.append(
            "The target proxy is not stable enough over time, so this case is better treated as a tracking/visibility failure than a clean geometry diagnosis."
        )
    else:
        lines.append(
            "The current metrics do not show a strong isolated geometry failure under the simple heuristic."
        )
    return lines


def estimate_background_scale(frame_prev: np.ndarray, frame_curr: np.ndarray, fg_mask: np.ndarray) -> float | None:
    try:
        import cv2
    except ImportError:
        return None

    prev_gray = grayscale(frame_prev).astype(np.uint8)
    curr_gray = grayscale(frame_curr).astype(np.uint8)
    h, w = prev_gray.shape
    grid_y, grid_x = np.mgrid[8 : h - 8 : 16, 8 : w - 8 : 16]
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    keep = ~fg_mask[np.clip(points[:, 1].astype(int), 0, h - 1), np.clip(points[:, 0].astype(int), 0, w - 1)]
    points = points[keep]
    if points.shape[0] < 12:
        return None
    next_points, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, points.reshape(-1, 1, 2), None)
    if next_points is None or status is None:
        return None
    points0 = points[status[:, 0] == 1]
    points1 = next_points.reshape(-1, 2)[status[:, 0] == 1]
    if points0.shape[0] < 8:
        return None
    matrix, _inliers = cv2.estimateAffinePartial2D(points0, points1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return None
    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    return math.sqrt(max(a * a + b * b, 0.0))


def classify_root_cause(
    *,
    max_area_ratio: float,
    max_invariant_ratio: float,
    mean_bg_scale: float | None,
    target_visible_ratio: float,
) -> str:
    if target_visible_ratio < 0.5:
        return "tracking_or_visibility_failure"
    if max_area_ratio < 1.2:
        return "stable_or_small_change"
    if mean_bg_scale is not None and mean_bg_scale > 1.05:
        return "camera_zoom_or_drift"
    if max_invariant_ratio > 1.35:
        return "object_scale_drift"
    return "normal_perspective_or_mild_change"


def normalize_series(values: list[float | None]) -> list[float | None]:
    base = next((float(v) for v in values if v is not None and np.isfinite(v) and v > 0), None)
    if base is None:
        return [None for _ in values]
    return [None if v is None or not np.isfinite(v) else float(v) / base for v in values]


def decode_movid_depth(bytes_payload: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(bytes_payload))
    depth_u16 = np.asarray(image, dtype=np.uint16)
    return depth_u16.astype(np.float32)


def decode_movid_segmentation(bytes_payload: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(bytes_payload))
    return np.asarray(image, dtype=np.int32)


def load_movid_gt(sidecar: dict[str, Any]) -> dict[str, Any]:
    meta_path = Path(sidecar["paths"]["meta_json_path"])
    meta = read_json(meta_path)
    tfrecord_path = Path(meta["source_paths"]["tfrecord_path"])
    record_index = int(meta["source_paths"]["tfrecord_record_index"])
    payload = None
    for idx, raw in enumerate(pm.iter_tfrecord_records(tfrecord_path)):
        if idx == record_index:
            payload = raw
            break
    if payload is None:
        raise RuntimeError(f"Failed to locate TFRecord record {record_index} in {tfrecord_path}")
    example = pm.parse_tf_example(payload)
    features = example.features.feature
    seg_frames = [decode_movid_segmentation(item) for item in features["segmentations"].bytes_list.value]
    depth_frames = [decode_movid_depth(item) for item in features["depth"].bytes_list.value]
    visibility = np.asarray(features["instances/visibility"].int64_list.value, dtype=np.int32)
    num_frames = int(features["metadata/num_frames"].int64_list.value[0])
    num_instances = int(features["metadata/num_instances"].int64_list.value[0])
    visibility = visibility.reshape(num_instances, num_frames).T
    return {
        "meta": meta,
        "seg_frames": seg_frames,
        "depth_frames": depth_frames,
        "visibility": visibility,
        "num_frames": num_frames,
        "num_instances": num_instances,
    }


def select_movid_target(gt: dict[str, Any], context_index: int) -> TargetSpec:
    seg = gt["seg_frames"][context_index]
    best_seg_id = None
    best_area = -1
    for seg_id in sorted(int(v) for v in np.unique(seg) if int(v) != BACKGROUND_SEGMENT_ID):
        area = int(np.count_nonzero(seg == seg_id))
        if area > best_area:
            best_area = area
            best_seg_id = seg_id
    if best_seg_id is None:
        raise RuntimeError("No visible foreground object found in MOVI-D context frame.")
    return TargetSpec(
        object_label=f"seg_{best_seg_id}",
        seg_id=int(best_seg_id),
        selection_mode="largest_visible_gt_instance_last_context",
        confidence=1.0,
    )


def load_genesis_gt(sidecar: dict[str, Any]) -> dict[str, Any]:
    meta_path = Path(sidecar["paths"]["meta_json_path"])
    meta = read_json(meta_path)
    source_sample_dir = Path(meta["source_paths"]["source_sample_dir"])
    source_meta_path = Path(meta["source_paths"]["source_metadata_json_path"])
    if not source_meta_path.exists():
        fallback_meta_path = source_sample_dir / "meta.json"
        if fallback_meta_path.exists():
            source_meta_path = fallback_meta_path
        else:
            raise FileNotFoundError(
                f"Genesis source metadata not found. Checked: {source_meta_path} and {fallback_meta_path}"
            )
    source_meta = read_json(source_meta_path)
    seg = np.load(source_sample_dir / "physics" / "seg.npy")
    depth = np.load(source_sample_dir / "physics" / "depth_metric.npy")
    return {
        "meta": meta,
        "source_meta": source_meta,
        "seg_frames": seg,
        "depth_frames": depth,
    }


def select_genesis_target(gt: dict[str, Any], context_index: int) -> TargetSpec:
    source_meta = gt["source_meta"]
    target_seg_id = None
    for obj in source_meta.get("objects", []):
        if str(obj.get("role") or "") == "target":
            target_seg_id = int(obj["seg_id"])
            break
    seg = np.asarray(gt["seg_frames"][context_index])
    if target_seg_id is not None and np.count_nonzero(seg == target_seg_id) > 0:
        return TargetSpec(
            object_label=f"seg_{target_seg_id}",
            seg_id=target_seg_id,
            selection_mode="genesis_role_target",
            confidence=1.0,
        )
    best_seg_id = None
    best_area = -1
    for seg_id in sorted(int(v) for v in np.unique(seg) if int(v) != GENESIS_BACKGROUND_SEGMENT_ID):
        area = int(np.count_nonzero(seg == seg_id))
        if area > best_area:
            best_area = area
            best_seg_id = seg_id
    if best_seg_id is None:
        raise RuntimeError("No visible foreground object found in Genesis context frame.")
    return TargetSpec(
        object_label=f"seg_{best_seg_id}",
        seg_id=int(best_seg_id),
        selection_mode="largest_visible_gt_segment_last_context",
        confidence=0.8,
    )


def detect_dataset_mode(sidecar: dict[str, Any]) -> str:
    dataset = str(sidecar.get("dataset") or "").lower()
    sample_dir = str(sidecar.get("paths", {}).get("sample_dir") or "").lower()
    if "kubric_tfds_movi-d" in dataset or "movi-d" in dataset or "movi_d" in sample_dir:
        return "movi_d_gt"
    if "genesis" in dataset or "genesis_rigid" in dataset or "version_1_genesis" in dataset:
        return "genesis_gt"
    return "generic_approx"


def detect_generated_proxy_mask(
    *,
    frame: np.ndarray,
    reference_frame: np.ndarray,
    anchor_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    diff = np.abs(grayscale(frame) - grayscale(reference_frame))
    if anchor_bbox is None:
        threshold = float(np.percentile(diff, 90))
        return diff > max(threshold, 12.0)
    x1, y1, x2, y2 = anchor_bbox
    roi = diff[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((h, w), dtype=bool)
    threshold = float(np.percentile(roi, 75))
    roi_mask = roi > max(threshold, 8.0)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = roi_mask
    return mask


def build_curves_from_gt(
    *,
    output_frames: list[np.ndarray],
    seg_frames: list[np.ndarray] | np.ndarray,
    depth_frames: list[np.ndarray] | np.ndarray,
    target_spec: TargetSpec,
    context_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    bg_scales: list[float] = []
    first_frame = output_frames[0]
    height, width = first_frame.shape[0], first_frame.shape[1]
    total_visible = 0

    for t, frame in enumerate(output_frames):
        raw_seg = np.asarray(seg_frames[t])
        raw_depth = np.asarray(depth_frames[t], dtype=np.float32)
        mask = resize_mask(raw_seg == target_spec.seg_id, width=width, height=height)
        depth_resized = resize_map(raw_depth, width=width, height=height).astype(np.float32)
        mask_area = int(np.count_nonzero(mask))
        bbox = compute_bbox_from_mask(mask)
        bbox_area_value = bbox_area(bbox)
        fill_ratio = mask_fill_ratio(mask_area, bbox_area_value)
        depth_value = mask_median_depth(depth_resized, mask)
        invariant = None if depth_value is None else float(mask_area) * float(depth_value) * float(depth_value)
        if mask_area > 0:
            total_visible += 1
        if t > 0:
            bg_scale = estimate_background_scale(output_frames[t - 1], frame, mask)
        else:
            bg_scale = None
        if bg_scale is not None:
            bg_scales.append(float(bg_scale))
        curves.append(
            {
                "t": t,
                "is_context": int(t < context_frames),
                "is_future": int(t >= context_frames),
                "mask_area": mask_area,
                "bbox_area": bbox_area_value,
                "fill_ratio": round(fill_ratio, 6),
                "depth_median": None if depth_value is None else round(depth_value, 6),
                "area_depth2": None if invariant is None else round(invariant, 6),
                "bg_scale": None if bg_scale is None else round(float(bg_scale), 6),
            }
        )

    summary = {
        "target_visible_ratio": float(total_visible) / float(max(len(output_frames), 1)),
        "mean_bg_scale": None if not bg_scales else float(np.mean(bg_scales)),
    }
    return curves, summary


def build_generated_proxy_curves(
    *,
    output_frames: list[np.ndarray],
    context_frames: int,
    anchor_bbox: tuple[int, int, int, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    reference = output_frames[max(context_frames - 1, 0)]
    bg_scales: list[float] = []
    visible = 0

    for t, frame in enumerate(output_frames):
        mask = detect_generated_proxy_mask(frame=frame, reference_frame=reference, anchor_bbox=anchor_bbox)
        mask_area = int(np.count_nonzero(mask))
        bbox = compute_bbox_from_mask(mask)
        bbox_area_value = bbox_area(bbox)
        fill_ratio = mask_fill_ratio(mask_area, bbox_area_value)
        gray = grayscale(frame)
        proxy_depth = None
        if mask_area > 0:
            proxy_depth = float(255.0 - np.median(gray[mask]) + 1.0)
            visible += 1
        invariant = None if proxy_depth is None else float(mask_area) * proxy_depth * proxy_depth
        if t > 0:
            bg_scale = estimate_background_scale(output_frames[t - 1], frame, mask)
        else:
            bg_scale = None
        if bg_scale is not None:
            bg_scales.append(float(bg_scale))
        curves.append(
            {
                "t": t,
                "is_context": int(t < context_frames),
                "is_future": int(t >= context_frames),
                "mask_area": mask_area,
                "bbox_area": bbox_area_value,
                "fill_ratio": round(fill_ratio, 6),
                "depth_median": None if proxy_depth is None else round(proxy_depth, 6),
                "area_depth2": None if invariant is None else round(invariant, 6),
                "bg_scale": None if bg_scale is None else round(float(bg_scale), 6),
            }
        )
    summary = {
        "target_visible_ratio": float(visible) / float(max(len(output_frames), 1)),
        "mean_bg_scale": None if not bg_scales else float(np.mean(bg_scales)),
    }
    return curves, summary


def build_generic_proxy_curves(
    *,
    output_frames: list[np.ndarray],
    context_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], TargetSpec]:
    curves: list[dict[str, Any]] = []
    reference = output_frames[max(context_frames - 1, 0)]
    ref_gray = grayscale(reference)
    bg_scales: list[float] = []
    visible = 0

    for t, frame in enumerate(output_frames):
        gray = grayscale(frame)
        diff = np.abs(gray - ref_gray)
        threshold = float(np.percentile(diff, 90))
        mask = diff > max(threshold, 12.0)
        mask_area = int(np.count_nonzero(mask))
        bbox = compute_bbox_from_mask(mask)
        bbox_area_value = bbox_area(bbox)
        fill_ratio = mask_fill_ratio(mask_area, bbox_area_value)
        proxy_depth = None
        if mask_area > 0:
            proxy_depth = float(255.0 - np.median(gray[mask]) + 1.0)
            visible += 1
        invariant = None if proxy_depth is None else float(mask_area) * proxy_depth * proxy_depth
        if t > 0:
            bg_scale = estimate_background_scale(output_frames[t - 1], frame, mask)
        else:
            bg_scale = None
        if bg_scale is not None:
            bg_scales.append(float(bg_scale))
        curves.append(
            {
                "t": t,
                "is_context": int(t < context_frames),
                "is_future": int(t >= context_frames),
                "mask_area": mask_area,
                "bbox_area": bbox_area_value,
                "fill_ratio": round(fill_ratio, 6),
                "depth_median": None if proxy_depth is None else round(proxy_depth, 6),
                "area_depth2": None if invariant is None else round(invariant, 6),
                "bg_scale": None if bg_scale is None else round(float(bg_scale), 6),
            }
        )
    summary = {
        "target_visible_ratio": float(visible) / float(max(len(output_frames), 1)),
        "mean_bg_scale": None if not bg_scales else float(np.mean(bg_scales)),
    }
    target = TargetSpec(
        object_label="foreground_proxy",
        seg_id=-1,
        selection_mode="frame_difference_proxy",
        confidence=0.2,
    )
    return curves, summary, target


def add_normalized_columns(curves: list[dict[str, Any]]) -> None:
    mask_area_norm = normalize_series([float(row["mask_area"]) if row["mask_area"] else None for row in curves])
    bbox_area_norm = normalize_series([float(row["bbox_area"]) if row["bbox_area"] else None for row in curves])
    depth_norm = normalize_series([row["depth_median"] for row in curves])
    invariant_norm = normalize_series([row["area_depth2"] for row in curves])
    for row, a, b, d, inv in zip(curves, mask_area_norm, bbox_area_norm, depth_norm, invariant_norm):
        row["mask_area_norm"] = None if a is None else round(a, 6)
        row["bbox_area_norm"] = None if b is None else round(b, 6)
        row["depth_norm"] = None if d is None else round(d, 6)
        row["area_depth2_norm"] = None if inv is None else round(inv, 6)


def summarize_case(curves: list[dict[str, Any]], helper: dict[str, Any]) -> dict[str, Any]:
    future_rows = [row for row in curves if int(row["is_future"]) == 1]
    mask_ratios = [float(row["mask_area_norm"]) for row in future_rows if row["mask_area_norm"] is not None]
    invariant_ratios = [float(row["area_depth2_norm"]) for row in future_rows if row["area_depth2_norm"] is not None]
    max_area_ratio = max(mask_ratios) if mask_ratios else 0.0
    max_invariant_ratio = max(invariant_ratios) if invariant_ratios else 0.0
    target_visible_ratio = float(helper.get("target_visible_ratio") or 0.0)
    mean_bg_scale = helper.get("mean_bg_scale")
    root_cause = classify_root_cause(
        max_area_ratio=max_area_ratio,
        max_invariant_ratio=max_invariant_ratio,
        mean_bg_scale=None if mean_bg_scale is None else float(mean_bg_scale),
        target_visible_ratio=target_visible_ratio,
    )
    return {
        "max_future_mask_area_ratio": round(max_area_ratio, 6),
        "max_future_area_depth2_ratio": round(max_invariant_ratio, 6),
        "target_visible_ratio": round(target_visible_ratio, 6),
        "mean_bg_scale": None if mean_bg_scale is None else round(float(mean_bg_scale), 6),
        "root_cause": root_cause,
    }


def plot_curves(curves: list[dict[str, Any]], path: Path, title: str) -> None:
    ts = [int(row["t"]) for row in curves]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(ts, [row["mask_area_norm"] for row in curves], label="mask_area_norm")
    axes[0].plot(ts, [row["bbox_area_norm"] for row in curves], label="bbox_area_norm")
    axes[0].legend()
    axes[0].set_ylabel("area")

    axes[1].plot(ts, [row["depth_norm"] for row in curves], label="depth_norm", color="tab:green")
    axes[1].legend()
    axes[1].set_ylabel("depth")

    axes[2].plot(ts, [row["area_depth2_norm"] for row in curves], label="area*depth^2", color="tab:red")
    axes[2].legend()
    axes[2].set_ylabel("invariant")

    axes[3].plot(ts, [row["bg_scale"] for row in curves], label="bg_scale", color="tab:orange")
    axes[3].legend()
    axes[3].set_ylabel("bg")
    axes[3].set_xlabel("frame")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curve_comparison(
    *,
    generated_curves: list[dict[str, Any]],
    gt_curves: list[dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    if not generated_curves or not gt_curves:
        return
    gen_ts = [int(row["t"]) for row in generated_curves]
    gt_ts = [int(row["t"]) for row in gt_curves]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    axes[0].plot(gen_ts, [row["mask_area_norm"] for row in generated_curves], label="gen mask_area", color="tab:red")
    axes[0].plot(gen_ts, [row["bbox_area_norm"] for row in generated_curves], label="gen bbox_area", color="tab:orange")
    axes[0].plot(gt_ts, [row["mask_area_norm"] for row in gt_curves], label="gt mask_area", color="tab:blue", linestyle="--")
    axes[0].plot(gt_ts, [row["bbox_area_norm"] for row in gt_curves], label="gt bbox_area", color="tab:cyan", linestyle="--")
    axes[0].legend()
    axes[0].set_ylabel("area")

    axes[1].plot(gen_ts, [row["depth_norm"] for row in generated_curves], label="gen depth", color="tab:red")
    axes[1].plot(gt_ts, [row["depth_norm"] for row in gt_curves], label="gt depth", color="tab:blue", linestyle="--")
    axes[1].legend()
    axes[1].set_ylabel("depth")

    axes[2].plot(gen_ts, [row["area_depth2_norm"] for row in generated_curves], label="gen invariant", color="tab:red")
    axes[2].plot(gt_ts, [row["area_depth2_norm"] for row in gt_curves], label="gt invariant", color="tab:blue", linestyle="--")
    axes[2].legend()
    axes[2].set_ylabel("invariant")

    axes[3].plot(gen_ts, [row["bg_scale"] for row in generated_curves], label="gen bg_scale", color="tab:red")
    axes[3].plot(gt_ts, [row["bg_scale"] for row in gt_curves], label="gt bg_scale", color="tab:blue", linestyle="--")
    axes[3].legend()
    axes[3].set_ylabel("bg")
    axes[3].set_xlabel("frame")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_annotated_generated_video(
    *,
    frames: list[np.ndarray],
    case_dir: Path,
    anchor_bbox: tuple[int, int, int, int] | None,
    fps: int,
) -> Path:
    annotated: list[np.ndarray] = []
    reference = frames[0]
    for frame in frames:
        mask = detect_generated_proxy_mask(frame=frame, reference_frame=reference, anchor_bbox=anchor_bbox)
        bbox = compute_bbox_from_mask(mask)
        vis = overlay_mask(frame, mask, color=(228, 87, 46), alpha=0.28)
        vis = draw_rect(vis, bbox, color=(228, 87, 46), thickness=2)
        vis = draw_anchor_window(vis, anchor_bbox)
        annotated.append(vis)
    out_path = case_dir / "generated_diagnostic.mp4"
    save_video_frames(out_path, annotated, fps=fps)
    return out_path


def write_annotated_context_video(
    *,
    context_frames: list[np.ndarray],
    gt_masks: list[np.ndarray],
    gt_seg_id: int,
    case_dir: Path,
    fps: int,
    anchor_bbox: tuple[int, int, int, int] | None,
) -> Path:
    annotated: list[np.ndarray] = []
    for frame, seg in zip(context_frames, gt_masks):
        mask = np.asarray(seg) == gt_seg_id
        bbox = compute_bbox_from_mask(mask)
        vis = overlay_mask(frame, mask, color=(94, 154, 255), alpha=0.28)
        vis = draw_rect(vis, bbox, color=(94, 154, 255), thickness=2)
        vis = draw_anchor_window(vis, anchor_bbox)
        annotated.append(vis)
    out_path = case_dir / "context_diagnostic.mp4"
    save_video_frames(out_path, annotated, fps=fps)
    return out_path


def write_annotated_gt_video(
    *,
    gt_frames: list[np.ndarray],
    gt_masks: list[np.ndarray],
    gt_seg_id: int,
    case_dir: Path,
    fps: int,
) -> Path:
    annotated: list[np.ndarray] = []
    for frame, seg in zip(gt_frames, gt_masks):
        mask = np.asarray(seg) == gt_seg_id
        bbox = compute_bbox_from_mask(mask)
        vis = overlay_mask(frame, mask, color=(48, 132, 214), alpha=0.28)
        vis = draw_rect(vis, bbox, color=(48, 132, 214), thickness=2)
        annotated.append(vis)
    out_path = case_dir / "gt_diagnostic.mp4"
    save_video_frames(out_path, annotated, fps=fps)
    return out_path


def analyze_case(sidecar_path: Path, output_root: Path, overwrite: bool) -> dict[str, Any]:
    sidecar = read_json(sidecar_path)
    paths = sidecar.get("paths", {})
    output_video_path = Path(paths["output_video_path"])
    sample_key = sanitize_token(f"{sidecar.get('dataset')}__{sidecar.get('sample_id')}")
    case_dir = output_root / sample_key
    diagnostics_path = case_dir / "diagnostics.json"
    if diagnostics_path.exists() and not overwrite:
        return read_json(diagnostics_path)

    output_frames = load_video_frames(output_video_path)
    if not output_frames:
        raise RuntimeError(f"No frames found in output video: {output_video_path}")
    context_frames = int(sidecar.get("generation_params", {}).get("used_context_frames") or 0)
    context_frames = max(1, min(context_frames, len(output_frames)))
    fps = int(sidecar.get("generation_params", {}).get("fps") or 8)
    mode = detect_dataset_mode(sidecar)
    anchor_bbox = None
    gt_video_frames: list[np.ndarray] = []
    context_video_path = Path(paths["context_video_path"]) if paths.get("context_video_path") else None
    context_video_frames = load_video_frames(context_video_path) if context_video_path and context_video_path.exists() else []
    context_diagnostic_video = None

    if mode == "movi_d_gt":
        gt = load_movid_gt(sidecar)
        target = select_movid_target(gt, context_frames - 1)
        gt_video_path = Path(paths["full_video_path"])
        gt_video_frames = load_video_frames(gt_video_path)[: min(len(gt["seg_frames"]), len(output_frames))]
        gt_frame_count = min(len(output_frames), len(gt["seg_frames"]), len(gt["depth_frames"]))
        gt_curves, gt_helper = build_curves_from_gt(
            output_frames=output_frames[:gt_frame_count],
            seg_frames=gt["seg_frames"],
            depth_frames=gt["depth_frames"],
            target_spec=target,
            context_frames=min(context_frames, gt_frame_count),
        )
        anchor_mask = gt["seg_frames"][min(context_frames - 1, len(gt["seg_frames"]) - 1)] == target.seg_id
        anchor_bbox = compute_bbox_from_mask(anchor_mask)
        if anchor_bbox is not None:
            anchor_bbox = resize_bbox(
                tuple(float(v) for v in anchor_bbox),
                src_width=int(gt["seg_frames"][0].shape[1]),
                src_height=int(gt["seg_frames"][0].shape[0]),
                dst_width=int(output_frames[0].shape[1]),
                dst_height=int(output_frames[0].shape[0]),
            )
            anchor_bbox = clamp_bbox(anchor_bbox, output_frames[0].shape[1], output_frames[0].shape[0])
        curves, helper = build_generated_proxy_curves(
            output_frames=output_frames,
            context_frames=context_frames,
            anchor_bbox=anchor_bbox,
        )
        context_overlay_frames = gt["seg_frames"][: min(len(context_video_frames), len(gt["seg_frames"]))]
    elif mode == "genesis_gt":
        gt = load_genesis_gt(sidecar)
        target = select_genesis_target(gt, context_frames - 1)
        gt_video_path = Path(paths["full_video_path"])
        gt_video_frames = load_video_frames(gt_video_path)[: min(len(gt["seg_frames"]), len(output_frames))]
        gt_frame_count = min(len(output_frames), len(gt["seg_frames"]), len(gt["depth_frames"]))
        gt_curves, gt_helper = build_curves_from_gt(
            output_frames=output_frames[:gt_frame_count],
            seg_frames=gt["seg_frames"],
            depth_frames=gt["depth_frames"],
            target_spec=target,
            context_frames=min(context_frames, gt_frame_count),
        )
        anchor_mask = np.asarray(gt["seg_frames"][min(context_frames - 1, len(gt["seg_frames"]) - 1)]) == target.seg_id
        anchor_bbox = compute_bbox_from_mask(anchor_mask)
        if anchor_bbox is not None:
            anchor_bbox = resize_bbox(
                tuple(float(v) for v in anchor_bbox),
                src_width=int(gt["seg_frames"][0].shape[1]),
                src_height=int(gt["seg_frames"][0].shape[0]),
                dst_width=int(output_frames[0].shape[1]),
                dst_height=int(output_frames[0].shape[0]),
            )
            anchor_bbox = clamp_bbox(anchor_bbox, output_frames[0].shape[1], output_frames[0].shape[0])
        curves, helper = build_generated_proxy_curves(
            output_frames=output_frames,
            context_frames=context_frames,
            anchor_bbox=anchor_bbox,
        )
        context_overlay_frames = list(gt["seg_frames"][: min(len(context_video_frames), len(gt["seg_frames"]))])
    else:
        curves, helper, target = build_generic_proxy_curves(
            output_frames=output_frames,
            context_frames=context_frames,
        )
        gt_curves = []
        gt_helper = {}
        context_overlay_frames = []

    add_normalized_columns(curves)
    if gt_curves:
        add_normalized_columns(gt_curves)
    summary = summarize_case(curves, helper)
    generated_diagnostic_video = write_annotated_generated_video(
        frames=output_frames,
        case_dir=case_dir,
        anchor_bbox=anchor_bbox,
        fps=fps,
    )
    gt_diagnostic_video = None
    if gt_curves and gt_video_frames:
        if mode == "movi_d_gt":
            gt_diagnostic_video = write_annotated_gt_video(
                gt_frames=gt_video_frames[:len(gt_curves)],
                gt_masks=gt["seg_frames"][:len(gt_curves)],
                gt_seg_id=target.seg_id,
                case_dir=case_dir,
                fps=fps,
            )
        elif mode == "genesis_gt":
            gt_diagnostic_video = write_annotated_gt_video(
                gt_frames=gt_video_frames[:len(gt_curves)],
                gt_masks=list(gt["seg_frames"][:len(gt_curves)]),
                gt_seg_id=target.seg_id,
                case_dir=case_dir,
                fps=fps,
            )
    if context_video_frames and context_overlay_frames:
        context_diagnostic_video = write_annotated_context_video(
            context_frames=context_video_frames[:len(context_overlay_frames)],
            gt_masks=context_overlay_frames,
            gt_seg_id=target.seg_id,
            case_dir=case_dir,
            fps=fps,
            anchor_bbox=anchor_bbox,
        )
    analysis_lines = build_case_analysis_text(
        sidecar=sidecar,
        summary=summary,
        target=target,
        mode=mode,
        gt_available=bool(gt_curves),
    )
    diagnostics = {
        "sidecar_path": str(sidecar_path),
        "dataset": sidecar.get("dataset"),
        "sample_id": sidecar.get("sample_id"),
        "mode": mode,
        "target": {
            "object_label": target.object_label,
            "seg_id": target.seg_id,
            "selection_mode": target.selection_mode,
            "confidence": round(target.confidence, 6),
        },
        "summary": summary,
        "analysis": analysis_lines,
        "num_frames": len(output_frames),
        "context_frames": context_frames,
        "gt_reference": {
            "available": bool(gt_curves),
            "num_frames_compared": len(gt_curves),
            "target_visible_ratio": gt_helper.get("target_visible_ratio"),
            "mean_bg_scale": gt_helper.get("mean_bg_scale"),
        },
        "artifacts": {
            "context_video": str(context_video_path) if context_video_path else None,
            "generated_video": str(output_video_path),
            "full_gt_video": str(Path(paths["full_video_path"])) if paths.get("full_video_path") else None,
            "curves_csv": str(case_dir / "curves.csv"),
            "curves_png": str(case_dir / "curves.png"),
            "gt_curves_csv": str(case_dir / "gt_curves.csv") if gt_curves else None,
            "gt_curves_png": str(case_dir / "gt_curves.png") if gt_curves else None,
            "comparison_curves_png": str(case_dir / "comparison_curves.png") if gt_curves else None,
            "context_diagnostic_video": str(context_diagnostic_video) if context_diagnostic_video else None,
            "generated_diagnostic_video": str(generated_diagnostic_video),
            "gt_diagnostic_video": str(gt_diagnostic_video) if gt_diagnostic_video else None,
        },
    }
    write_csv(case_dir / "curves.csv", curves)
    plot_curves(curves, case_dir / "curves.png", title=f"{sidecar.get('dataset')} | {sidecar.get('sample_id')}")
    if gt_curves:
        write_csv(case_dir / "gt_curves.csv", gt_curves)
        plot_curves(gt_curves, case_dir / "gt_curves.png", title=f"GT | {sidecar.get('dataset')} | {sidecar.get('sample_id')}")
        plot_curve_comparison(
            generated_curves=curves,
            gt_curves=gt_curves,
            path=case_dir / "comparison_curves.png",
            title=f"GT vs Generated | {sidecar.get('dataset')} | {sidecar.get('sample_id')}",
        )
    write_json(diagnostics_path, diagnostics)
    return diagnostics


def main() -> None:
    args = parse_args()
    sidecars = list_sidecars(args)
    if not sidecars:
        raise SystemExit("No sidecars found. Provide --sidecar or --sidecar_dir.")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for sidecar_path in sidecars:
        diagnostics = analyze_case(sidecar_path, output_root=output_root, overwrite=args.overwrite)
        results.append(diagnostics)
        print(
            json.dumps(
                {
                    "dataset": diagnostics["dataset"],
                    "sample_id": diagnostics["sample_id"],
                    "mode": diagnostics["mode"],
                    "root_cause": diagnostics["summary"]["root_cause"],
                    "max_future_mask_area_ratio": diagnostics["summary"]["max_future_mask_area_ratio"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_json(output_root / "summary.json", {"num_cases": len(results), "cases": results})


if __name__ == "__main__":
    main()
