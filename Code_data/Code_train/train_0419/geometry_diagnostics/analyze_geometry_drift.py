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
import torch
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parent
TRAIN0419_ROOT = SCRIPT_ROOT.parent
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

import prepare_mixed_benchmark_mytest as pm  # noqa: E402


BACKGROUND_SEGMENT_ID = 0
GENESIS_BACKGROUND_SEGMENT_ID = 0
GENESIS_GROUND_SPECIAL_ID = -1
SAM2_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OVERLAY_MIN_RELATIVE_MOTION = 0.2
OVERLAY_MIN_ABSOLUTE_MOTION = 1e-4
GENERATED_BORN_MIN_AREA = 250
GENERATED_BORN_MIN_DURATION = 2
GENERATED_BORN_IOU_MATCH = 0.3
GENERATED_BORN_APPEARANCE_MATCH = 0.55
GENERATED_BORN_MIN_CONFIDENCE_NEW = 0.62
GENERATED_BORN_MIN_CONFIDENCE_MATCHED = 0.55
GENERATED_BORN_REFINE_MAX_SEEDS = 3
GENERATED_BORN_REFINE_MIN_SEED_COVERAGE = 0.6
GENERATED_BORN_MERGE_GAP = 3
GENERATED_BORN_MERGE_IOU = 0.12
GENERATED_BORN_MERGE_CENTER_DIST = 90.0
GENERATED_BORN_MERGE_CENTER_DIST_WITH_APPEARANCE = 180.0
GENERATED_BORN_TRACK_APPEARANCE_MERGE = 0.68
GENERATED_BORN_OVERLAP_MERGE_BBOX_IOU = 0.28
GENERATED_BORN_OVERLAP_MERGE_MASK_COVERAGE = 0.62
GENERATED_BORN_SUPPRESS_DOMINATED_COVERAGE = 0.72
GENERATED_BORN_SUPPRESS_DOMINATED_CENTER_DIST = 70.0
GENERATED_BORN_FLOOR_BOTTOM_FRAC = 0.82
GENERATED_BORN_FLOOR_MAX_HEIGHT_FRAC = 0.18
GENERATED_BORN_FLOOR_MIN_WIDTH_FRAC = 0.45
GENERATED_BORN_FLOOR_MIN_DURATION = 10
GENERATED_BORN_STATIC_SMALL_MAX_AREA_FRAC = 0.005
GENERATED_BORN_STATIC_SMALL_MAX_MOTION = 40.0
_SAM2_VIDEO_PREDICTOR = None


@dataclass
class TargetSpec:
    object_label: str
    seg_id: int
    selection_mode: str
    confidence: float


@dataclass
class OverlayObjectSpec:
    object_label: str
    seg_id: int
    color: tuple[int, int, int]
    motion_score: float


@dataclass
class GeneratedTrackResult:
    masks_by_seg_id: dict[int, dict[int, np.ndarray]]
    bboxes_by_seg_id: dict[int, dict[int, tuple[int, int, int, int] | None]]
    source: str


@dataclass
class GeneratedBornTrack:
    track_id: int
    masks_by_frame: dict[int, np.ndarray]
    bboxes_by_frame: dict[int, tuple[int, int, int, int] | None]
    areas_by_frame: dict[int, int]
    classification: str
    matched_seg_id: int | None
    matched_object_label: str | None
    appearance_similarity: float
    color: tuple[int, int, int]
    confidence: float
    displayed: bool
    prototype_hist: np.ndarray | None = None


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


def load_sam2_video_predictor():
    global _SAM2_VIDEO_PREDICTOR
    if _SAM2_VIDEO_PREDICTOR is not None:
        return _SAM2_VIDEO_PREDICTOR
    sys.path.insert(0, "/home/gaoya/Code_Video/Wan2.2-main/wan_/modules/animate/preprocess")
    from sam_utils import build_sam2_video_predictor  # noqa: WPS433

    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        ckpt_path=str(SAM2_CHECKPOINT),
        device="cuda",
        mode="eval",
    )
    predictor.fill_hole_area = 0
    _SAM2_VIDEO_PREDICTOR = predictor
    return predictor


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


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = np.count_nonzero(a | b)
    if union <= 0:
        return 0.0
    inter = np.count_nonzero(a & b)
    return float(inter) / float(union)


def mask_coverage(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    denom = min(np.count_nonzero(a), np.count_nonzero(b))
    if denom <= 0:
        return 0.0
    inter = np.count_nonzero(a & b)
    return float(inter) / float(denom)


def bbox_iou(
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
) -> float:
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def compute_track_motion_score(
    coords_xy: np.ndarray,
    visible_mask: np.ndarray,
) -> float:
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    visible_mask = np.asarray(visible_mask, dtype=bool)
    valid_coords = coords_xy[visible_mask]
    if valid_coords.shape[0] < 2:
        return 0.0
    deltas = valid_coords[1:] - valid_coords[:-1]
    step_norms = np.linalg.norm(deltas, axis=1)
    if step_norms.size == 0:
        return 0.0
    return float(np.sum(step_norms))


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


def estimate_mask_color_histogram(frame: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray | None:
    mask = np.asarray(mask, dtype=bool)
    if np.count_nonzero(mask) < 16:
        return None
    pixels = np.asarray(frame, dtype=np.uint8)[mask]
    hist_parts: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=bins, range=(0, 256))
        hist_parts.append(hist.astype(np.float32))
    hist_vec = np.concatenate(hist_parts, axis=0)
    total = float(np.sum(hist_vec))
    if total <= 0:
        return None
    return hist_vec / total


def histogram_intersection(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    return float(np.minimum(a, b).sum())


def compute_generated_born_confidence(
    *,
    areas_by_frame: dict[int, int],
    appearance_similarity: float,
    classification: str,
) -> float:
    if not areas_by_frame:
        return 0.0
    frame_ids = sorted(int(v) for v in areas_by_frame)
    duration = len(frame_ids)
    max_area = max(int(v) for v in areas_by_frame.values())
    duration_score = min(float(duration) / 8.0, 1.0)
    area_score = min(float(max_area) / 2500.0, 1.0)
    appearance_score = max(min(float(appearance_similarity), 1.0), 0.0)
    base = 0.45 * duration_score + 0.25 * area_score + 0.30 * appearance_score
    if classification == "scale_drift_candidate":
        base += 0.1
    return max(0.0, min(base, 1.0))


def bbox_center(bbox: tuple[int, int, int, int] | None) -> tuple[float, float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


def compute_track_pair_overlap_stats(
    track_a: GeneratedBornTrack,
    track_b: GeneratedBornTrack,
) -> dict[str, float]:
    shared_frames = sorted(set(track_a.masks_by_frame) & set(track_b.masks_by_frame))
    if not shared_frames:
        return {
            "shared_frames": 0.0,
            "mean_mask_iou": 0.0,
            "mean_mask_coverage": 0.0,
            "mean_bbox_iou": 0.0,
            "mean_center_dist": float("inf"),
        }
    mask_ious: list[float] = []
    mask_coverages: list[float] = []
    bbox_ious: list[float] = []
    center_dists: list[float] = []
    for frame_idx in shared_frames:
        mask_a = track_a.masks_by_frame.get(frame_idx)
        mask_b = track_b.masks_by_frame.get(frame_idx)
        bbox_a = track_a.bboxes_by_frame.get(frame_idx)
        bbox_b = track_b.bboxes_by_frame.get(frame_idx)
        if mask_a is not None and mask_b is not None:
            mask_ious.append(mask_iou(mask_a, mask_b))
            mask_coverages.append(mask_coverage(mask_a, mask_b))
        bbox_ious.append(bbox_iou(bbox_a, bbox_b))
        center_a = bbox_center(bbox_a)
        center_b = bbox_center(bbox_b)
        if center_a is not None and center_b is not None:
            center_dists.append(math.dist(center_a, center_b))
    return {
        "shared_frames": float(len(shared_frames)),
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "mean_mask_coverage": float(np.mean(mask_coverages)) if mask_coverages else 0.0,
        "mean_bbox_iou": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
        "mean_center_dist": float(np.mean(center_dists)) if center_dists else float("inf"),
    }


def compute_generated_born_track_motion(track: GeneratedBornTrack) -> float:
    frame_ids = sorted(track.bboxes_by_frame)
    centers: list[tuple[float, float]] = []
    for frame_idx in frame_ids:
        center = bbox_center(track.bboxes_by_frame.get(frame_idx))
        if center is not None:
            centers.append(center)
    if len(centers) < 2:
        return 0.0
    motion = 0.0
    for prev, curr in zip(centers[:-1], centers[1:]):
        motion += math.dist(prev, curr)
    return float(motion)


def estimate_track_prototype_histogram(
    *,
    frames: list[np.ndarray],
    masks_by_frame: dict[int, np.ndarray],
    areas_by_frame: dict[int, int],
) -> np.ndarray | None:
    if not masks_by_frame:
        return None
    best_frame_idx = max(
        masks_by_frame,
        key=lambda frame_idx: int(areas_by_frame.get(int(frame_idx), 0)),
    )
    return estimate_mask_color_histogram(
        frames[int(best_frame_idx)],
        np.asarray(masks_by_frame[int(best_frame_idx)], dtype=bool),
    )


def should_merge_generated_born_tracks(track_a: GeneratedBornTrack, track_b: GeneratedBornTrack) -> bool:
    overlap_stats = compute_track_pair_overlap_stats(track_a, track_b)
    same_class = track_a.classification == track_b.classification
    same_match = track_a.matched_seg_id == track_b.matched_seg_id
    track_similarity = histogram_intersection(track_a.prototype_hist, track_b.prototype_hist)
    if overlap_stats["shared_frames"] >= 1.0:
        if overlap_stats["mean_center_dist"] > GENERATED_BORN_MERGE_CENTER_DIST_WITH_APPEARANCE:
            return False
        if overlap_stats["mean_bbox_iou"] >= GENERATED_BORN_OVERLAP_MERGE_BBOX_IOU:
            return True
        if same_class and same_match and overlap_stats["mean_mask_coverage"] >= GENERATED_BORN_OVERLAP_MERGE_MASK_COVERAGE:
            return True
        if (
            same_class
            and track_similarity >= GENERATED_BORN_TRACK_APPEARANCE_MERGE
            and (
                overlap_stats["mean_mask_coverage"] >= GENERATED_BORN_OVERLAP_MERGE_MASK_COVERAGE * 0.5
                or overlap_stats["mean_bbox_iou"] >= GENERATED_BORN_OVERLAP_MERGE_BBOX_IOU * 0.5
            )
        ):
            return True
        return False
    end_a = max(track_a.masks_by_frame)
    start_b = min(track_b.masks_by_frame)
    if start_b <= end_a or start_b - end_a > GENERATED_BORN_MERGE_GAP:
        return False
    bbox_a = track_a.bboxes_by_frame.get(end_a)
    bbox_b = track_b.bboxes_by_frame.get(start_b)
    center_a = bbox_center(bbox_a)
    center_b = bbox_center(bbox_b)
    if center_a is None or center_b is None:
        return False
    center_dist = math.dist(center_a, center_b)
    iou = mask_iou(track_a.masks_by_frame[end_a], track_b.masks_by_frame[start_b])
    if center_dist <= GENERATED_BORN_MERGE_CENTER_DIST and (iou >= GENERATED_BORN_MERGE_IOU or (same_class and same_match)):
        return True
    return bool(
        center_dist <= GENERATED_BORN_MERGE_CENTER_DIST_WITH_APPEARANCE
        and same_class
        and track_similarity >= GENERATED_BORN_TRACK_APPEARANCE_MERGE
    )


def merge_generated_born_track_pair(track_a: GeneratedBornTrack, track_b: GeneratedBornTrack) -> GeneratedBornTrack:
    masks_by_frame = dict(track_a.masks_by_frame)
    masks_by_frame.update(track_b.masks_by_frame)
    bboxes_by_frame = dict(track_a.bboxes_by_frame)
    bboxes_by_frame.update(track_b.bboxes_by_frame)
    areas_by_frame = dict(track_a.areas_by_frame)
    areas_by_frame.update(track_b.areas_by_frame)
    confidence = max(float(track_a.confidence), float(track_b.confidence))
    displayed = bool(track_a.displayed or track_b.displayed)
    appearance_similarity = max(float(track_a.appearance_similarity), float(track_b.appearance_similarity))
    chosen = track_a if float(track_a.confidence) >= float(track_b.confidence) else track_b
    return GeneratedBornTrack(
        track_id=min(int(track_a.track_id), int(track_b.track_id)),
        masks_by_frame=masks_by_frame,
        bboxes_by_frame=bboxes_by_frame,
        areas_by_frame=areas_by_frame,
        classification=chosen.classification,
        matched_seg_id=chosen.matched_seg_id,
        matched_object_label=chosen.matched_object_label,
        appearance_similarity=appearance_similarity,
        color=chosen.color,
        confidence=confidence,
        displayed=displayed,
        prototype_hist=chosen.prototype_hist,
    )


def merge_generated_born_tracks(tracks: list[GeneratedBornTrack]) -> list[GeneratedBornTrack]:
    if not tracks:
        return []
    ordered = sorted(tracks, key=lambda track: min(track.masks_by_frame))
    merged: list[GeneratedBornTrack] = []
    for track in ordered:
        if not merged:
            merged.append(track)
            continue
        merged_any = False
        for idx, prev in enumerate(merged):
            if not should_merge_generated_born_tracks(prev, track):
                continue
            merged[idx] = merge_generated_born_track_pair(prev, track)
            merged_any = True
            break
        if not merged_any:
            merged.append(track)
    return merged


def is_floor_like_generated_born_track(track: GeneratedBornTrack, frame_shape: tuple[int, int]) -> bool:
    if not track.masks_by_frame:
        return False
    height, width = frame_shape
    frame_ids = sorted(track.masks_by_frame)
    if len(frame_ids) < GENERATED_BORN_FLOOR_MIN_DURATION:
        return False
    bottom_hits = 0
    low_height_hits = 0
    wide_hits = 0
    for frame_idx in frame_ids:
        bbox = track.bboxes_by_frame.get(frame_idx)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if float(y2) / float(max(height, 1)) >= GENERATED_BORN_FLOOR_BOTTOM_FRAC:
            bottom_hits += 1
        if float(y2 - y1) / float(max(height, 1)) <= GENERATED_BORN_FLOOR_MAX_HEIGHT_FRAC:
            low_height_hits += 1
        if float(x2 - x1) / float(max(width, 1)) >= GENERATED_BORN_FLOOR_MIN_WIDTH_FRAC:
            wide_hits += 1
    required_hits = int(len(frame_ids) * 0.8)
    return bool(
        bottom_hits >= required_hits
        and low_height_hits >= required_hits
        and wide_hits >= required_hits
    )


def is_static_small_generated_born_track(track: GeneratedBornTrack, frame_shape: tuple[int, int]) -> bool:
    if not track.masks_by_frame:
        return False
    height, width = frame_shape
    frame_area = float(max(height * width, 1))
    max_area = float(max(track.areas_by_frame.values(), default=0))
    if max_area / frame_area > GENERATED_BORN_STATIC_SMALL_MAX_AREA_FRAC:
        return False
    return compute_generated_born_track_motion(track) <= GENERATED_BORN_STATIC_SMALL_MAX_MOTION


def suppress_dominated_generated_born_tracks(
    tracks: list[GeneratedBornTrack],
) -> list[GeneratedBornTrack]:
    if not tracks:
        return []
    ranked = sorted(
        tracks,
        key=lambda track: (
            bool(track.displayed),
            float(track.confidence),
            len(track.masks_by_frame),
            max(track.areas_by_frame.values(), default=0),
        ),
        reverse=True,
    )
    suppressed_ids: set[int] = set()
    for i, dominant in enumerate(ranked):
        if not dominant.displayed or dominant.track_id in suppressed_ids:
            continue
        for weaker in ranked[i + 1 :]:
            if not weaker.displayed or weaker.track_id in suppressed_ids:
                continue
            if dominant.classification != weaker.classification:
                continue
            if dominant.matched_seg_id != weaker.matched_seg_id:
                continue
            overlap_stats = compute_track_pair_overlap_stats(dominant, weaker)
            if overlap_stats["shared_frames"] < 2.0:
                continue
            if overlap_stats["mean_center_dist"] > GENERATED_BORN_SUPPRESS_DOMINATED_CENTER_DIST:
                continue
            if overlap_stats["mean_mask_coverage"] < GENERATED_BORN_SUPPRESS_DOMINATED_COVERAGE:
                continue
            suppressed_ids.add(int(weaker.track_id))
    final_tracks: list[GeneratedBornTrack] = []
    for track in tracks:
        final_tracks.append(
            GeneratedBornTrack(
                track_id=track.track_id,
                masks_by_frame=track.masks_by_frame,
                bboxes_by_frame=track.bboxes_by_frame,
                areas_by_frame=track.areas_by_frame,
                classification=track.classification,
                matched_seg_id=track.matched_seg_id,
                matched_object_label=track.matched_object_label,
                appearance_similarity=track.appearance_similarity,
                color=track.color,
                confidence=track.confidence,
                displayed=bool(track.displayed and int(track.track_id) not in suppressed_ids),
                prototype_hist=track.prototype_hist,
            )
        )
    return final_tracks


def should_display_generated_born_track(classification: str, confidence: float) -> bool:
    if classification == "new_object":
        return float(confidence) >= GENERATED_BORN_MIN_CONFIDENCE_NEW
    return float(confidence) >= GENERATED_BORN_MIN_CONFIDENCE_MATCHED


def color_for_seg_id(seg_id: int) -> tuple[int, int, int]:
    palette = [
        (228, 87, 46),
        (48, 132, 214),
        (62, 160, 96),
        (214, 111, 48),
        (192, 70, 166),
        (130, 112, 218),
        (194, 157, 52),
        (66, 180, 189),
    ]
    return palette[max(int(seg_id), 1) % len(palette)]


def filter_overlay_objects_by_motion(overlay_objects: list[OverlayObjectSpec]) -> list[OverlayObjectSpec]:
    if not overlay_objects:
        return []
    max_motion = max(float(obj.motion_score) for obj in overlay_objects)
    min_required = max(float(max_motion) * OVERLAY_MIN_RELATIVE_MOTION, OVERLAY_MIN_ABSOLUTE_MOTION)
    kept = [obj for obj in overlay_objects if float(obj.motion_score) >= min_required]
    return kept or [max(overlay_objects, key=lambda obj: float(obj.motion_score))]


def collect_visible_context_masks(
    *,
    seg_frames: list[np.ndarray] | np.ndarray,
    target_seg_id: int,
    context_frame_count: int,
    out_width: int,
    out_height: int,
) -> list[tuple[int, np.ndarray]]:
    visible: list[tuple[int, np.ndarray]] = []
    count = min(int(context_frame_count), len(seg_frames))
    for frame_idx in range(count):
        raw_seg = np.asarray(seg_frames[frame_idx])
        mask = resize_mask(raw_seg == int(target_seg_id), width=out_width, height=out_height)
        if np.count_nonzero(mask) == 0:
            continue
        visible.append((frame_idx, mask))
    return visible


def collect_moving_movid_overlay_objects(gt: dict[str, Any], context_index: int) -> list[OverlayObjectSpec]:
    seg = np.asarray(gt["seg_frames"][context_index])
    context_len = max(int(context_index) + 1, 1)
    objects: list[tuple[float, int, OverlayObjectSpec]] = []
    for seg_id in sorted(int(v) for v in np.unique(seg) if int(v) != BACKGROUND_SEGMENT_ID):
        instance_idx = seg_id - 1
        if instance_idx < 0 or instance_idx >= int(gt["num_instances"]):
            continue
        visible = gt["visibility"][:context_len, instance_idx] > 0
        coords = gt["image_positions"][:context_len, instance_idx]
        motion = compute_track_motion_score(coords, visible)
        area = int(np.count_nonzero(seg == seg_id))
        if area <= 0 or motion <= 0.0:
            continue
        objects.append(
            (
                motion,
                area,
                OverlayObjectSpec(
                    object_label=f"seg_{seg_id}",
                    seg_id=seg_id,
                    color=color_for_seg_id(seg_id),
                    motion_score=float(motion),
                ),
            )
        )
    objects.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return filter_overlay_objects_by_motion([item[2] for item in objects])


def collect_moving_genesis_overlay_objects(gt: dict[str, Any], context_index: int) -> list[OverlayObjectSpec]:
    source_meta = gt["source_meta"]
    seg = np.asarray(gt["seg_frames"][context_index])
    kin = gt["kinematics"]
    seg_ids = np.asarray(kin["seg_ids"], dtype=np.int32)
    com_uv = np.asarray(kin["com_uv"], dtype=np.float32)
    visibility = np.asarray(kin["visibility_mask"], dtype=np.uint8)
    context_len = max(int(context_index) + 1, 1)
    objects: list[tuple[float, int, OverlayObjectSpec]] = []
    for obj_idx, seg_id in enumerate(seg_ids.tolist()):
        seg_id = int(seg_id)
        if seg_id <= 0:
            continue
        visible = visibility[:context_len, obj_idx] > 0
        coords = com_uv[:context_len, obj_idx]
        motion = compute_track_motion_score(coords, visible)
        area = int(np.count_nonzero(seg == seg_id))
        if area <= 0 or motion <= 0.0:
            continue
        role = None
        for obj in source_meta.get("objects", []):
            if int(obj.get("seg_id", -1)) == seg_id:
                role = str(obj.get("role") or "")
                break
        label = f"seg_{seg_id}" if not role else f"seg_{seg_id}_{role}"
        objects.append(
            (
                motion,
                area,
                OverlayObjectSpec(
                    object_label=label,
                    seg_id=seg_id,
                    color=color_for_seg_id(seg_id),
                    motion_score=float(motion),
                ),
            )
        )
    objects.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return filter_overlay_objects_by_motion([item[2] for item in objects])


def collect_visible_context_masks_by_object(
    *,
    seg_frames: list[np.ndarray] | np.ndarray,
    overlay_objects: list[OverlayObjectSpec],
    context_frame_count: int,
    out_width: int,
    out_height: int,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    for obj in overlay_objects:
        visible = collect_visible_context_masks(
            seg_frames=seg_frames,
            target_seg_id=obj.seg_id,
            context_frame_count=context_frame_count,
            out_width=out_width,
            out_height=out_height,
        )
        if visible:
            out[int(obj.seg_id)] = visible
    return out


def run_generated_multi_object_track_with_sam2(
    *,
    frames: list[np.ndarray],
    context_masks_by_seg_id: dict[int, list[tuple[int, np.ndarray]]],
) -> GeneratedTrackResult | None:
    if not context_masks_by_seg_id:
        return None
    predictor = load_sam2_video_predictor()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state_v2(frames=frames)
        predictor.reset_state(inference_state)
        for seg_id, context_masks in context_masks_by_seg_id.items():
            for frame_idx, mask in context_masks:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=int(frame_idx),
                    obj_id=int(seg_id),
                    mask=np.asarray(mask, dtype=np.uint8),
                )
        masks_by_seg_id: dict[int, dict[int, np.ndarray]] = {int(seg_id): {} for seg_id in context_masks_by_seg_id}
        bboxes_by_seg_id: dict[int, dict[int, tuple[int, int, int, int] | None]] = {
            int(seg_id): {} for seg_id in context_masks_by_seg_id
        }
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            for i, out_obj_id in enumerate(out_obj_ids):
                seg_id = int(out_obj_id)
                if seg_id not in masks_by_seg_id:
                    continue
                mask = (out_mask_logits[i] > 0.0).detach().float().cpu().numpy().squeeze(0) > 0
                masks_by_seg_id[seg_id][int(out_frame_idx)] = mask
                bboxes_by_seg_id[seg_id][int(out_frame_idx)] = compute_bbox_from_mask(mask)
    nonempty = any(frames_by_seg for frames_by_seg in masks_by_seg_id.values())
    if not nonempty:
        return None
    return GeneratedTrackResult(
        masks_by_seg_id=masks_by_seg_id,
        bboxes_by_seg_id=bboxes_by_seg_id,
        source="sam2_context_multi_frame_mask_init",
    )


def refine_generated_born_track_with_sam2(
    *,
    frames: list[np.ndarray],
    track: GeneratedBornTrack,
) -> GeneratedBornTrack:
    if not track.displayed or not track.masks_by_frame:
        return track
    frame_ids = sorted(int(v) for v in track.masks_by_frame)
    seed_frame_ids = sorted(
        frame_ids,
        key=lambda frame_idx: int(track.areas_by_frame.get(int(frame_idx), 0)),
        reverse=True,
    )[:GENERATED_BORN_REFINE_MAX_SEEDS]
    predictor = load_sam2_video_predictor()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state_v2(frames=frames)
        predictor.reset_state(inference_state)
        for frame_idx in seed_frame_ids:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=int(frame_idx),
                obj_id=1,
                mask=np.asarray(track.masks_by_frame[int(frame_idx)], dtype=np.uint8),
            )
        refined_masks_by_frame: dict[int, np.ndarray] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            for i, out_obj_id in enumerate(out_obj_ids):
                if int(out_obj_id) != 1:
                    continue
                if int(out_frame_idx) < frame_ids[0] or int(out_frame_idx) > frame_ids[-1]:
                    continue
                refined_masks_by_frame[int(out_frame_idx)] = (
                    (out_mask_logits[i] > 0.0).detach().float().cpu().numpy().squeeze(0) > 0
                )
    if not refined_masks_by_frame:
        return track
    seed_coverages: list[float] = []
    for frame_idx in seed_frame_ids:
        refined_mask = refined_masks_by_frame.get(int(frame_idx))
        original_mask = np.asarray(track.masks_by_frame[int(frame_idx)], dtype=bool)
        if refined_mask is None:
            continue
        original_area = int(np.count_nonzero(original_mask))
        if original_area <= 0:
            continue
        overlap = np.count_nonzero(np.asarray(refined_mask, dtype=bool) & original_mask)
        seed_coverages.append(float(overlap) / float(original_area))
    if not seed_coverages or float(np.mean(seed_coverages)) < GENERATED_BORN_REFINE_MIN_SEED_COVERAGE:
        return track
    merged_masks_by_frame: dict[int, np.ndarray] = {}
    merged_bboxes_by_frame: dict[int, tuple[int, int, int, int] | None] = {}
    merged_areas_by_frame: dict[int, int] = {}
    for frame_idx in frame_ids:
        mask = np.asarray(refined_masks_by_frame.get(int(frame_idx), track.masks_by_frame[int(frame_idx)]), dtype=bool)
        merged_masks_by_frame[int(frame_idx)] = mask
        merged_bboxes_by_frame[int(frame_idx)] = compute_bbox_from_mask(mask)
        merged_areas_by_frame[int(frame_idx)] = int(np.count_nonzero(mask))
    return GeneratedBornTrack(
        track_id=track.track_id,
        masks_by_frame=merged_masks_by_frame,
        bboxes_by_frame=merged_bboxes_by_frame,
        areas_by_frame=merged_areas_by_frame,
        classification=track.classification,
        matched_seg_id=track.matched_seg_id,
        matched_object_label=track.matched_object_label,
        appearance_similarity=track.appearance_similarity,
        color=track.color,
        confidence=track.confidence,
        displayed=track.displayed,
        prototype_hist=track.prototype_hist,
    )


def detect_generated_born_candidates(
    *,
    frame: np.ndarray,
    reference_frame: np.ndarray,
    known_masks: list[np.ndarray],
) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError:
        return []
    diff = np.abs(grayscale(frame) - grayscale(reference_frame))
    threshold = float(np.percentile(diff, 92))
    raw_mask = diff > max(threshold, 14.0)
    known_union = np.zeros(raw_mask.shape, dtype=bool)
    for mask in known_masks:
        known_union |= np.asarray(mask, dtype=bool)
    residual = raw_mask & ~known_union
    kernel = np.ones((5, 5), dtype=np.uint8)
    residual_u8 = cv2.morphologyEx(residual.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(residual_u8, connectivity=8)
    proposals: list[np.ndarray] = []
    for label in range(1, int(num_labels)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < GENERATED_BORN_MIN_AREA:
            continue
        proposals.append(labels == label)
    return proposals


def classify_generated_born_track(
    *,
    track_masks: dict[int, np.ndarray],
    context_objects: list[OverlayObjectSpec],
    output_frames: list[np.ndarray],
    context_reference_hist_by_seg_id: dict[int, np.ndarray | None],
) -> tuple[str, int | None, str | None, float]:
    if not track_masks:
        return "new_object", None, None, 0.0
    first_frame_idx = min(track_masks)
    first_mask = np.asarray(track_masks[first_frame_idx], dtype=bool)
    hist = estimate_mask_color_histogram(output_frames[first_frame_idx], first_mask)
    best_obj = None
    best_score = 0.0
    for obj in context_objects:
        score = histogram_intersection(hist, context_reference_hist_by_seg_id.get(int(obj.seg_id)))
        if score > best_score:
            best_score = score
            best_obj = obj
    frame_ids = sorted(int(v) for v in track_masks)
    areas = [int(np.count_nonzero(track_masks[t])) for t in frame_ids]
    growth_ratio = 1.0
    if areas and areas[0] > 0:
        growth_ratio = float(max(areas)) / float(max(areas[0], 1))
    if best_obj is None or best_score < GENERATED_BORN_APPEARANCE_MATCH:
        return "new_object", None, None, float(best_score)
    if growth_ratio > 1.6:
        return "scale_drift_candidate", int(best_obj.seg_id), best_obj.object_label, float(best_score)
    return "duplicate_of_known_object", int(best_obj.seg_id), best_obj.object_label, float(best_score)


def track_generated_born_objects(
    *,
    output_frames: list[np.ndarray],
    context_frames: int,
    known_tracks: dict[int, dict[int, np.ndarray]],
    context_objects: list[OverlayObjectSpec],
    context_reference_hist_by_seg_id: dict[int, np.ndarray | None],
) -> list[GeneratedBornTrack]:
    active_tracks: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    next_track_id = 1
    reference_frame = output_frames[max(context_frames - 1, 0)]
    for frame_idx in range(context_frames, len(output_frames)):
        known_masks = [frames_by_seg[frame_idx] for frames_by_seg in known_tracks.values() if frame_idx in frames_by_seg]
        proposals = detect_generated_born_candidates(
            frame=output_frames[frame_idx],
            reference_frame=reference_frame,
            known_masks=known_masks,
        )
        matched_tracks: set[int] = set()
        for proposal_mask in proposals:
            best_idx = None
            best_iou = 0.0
            for idx, track in enumerate(active_tracks):
                if int(track["last_frame"]) != frame_idx - 1:
                    continue
                iou = mask_iou(proposal_mask, track["last_mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= GENERATED_BORN_IOU_MATCH:
                track = active_tracks[best_idx]
                track["masks_by_frame"][frame_idx] = proposal_mask
                track["bboxes_by_frame"][frame_idx] = compute_bbox_from_mask(proposal_mask)
                track["areas_by_frame"][frame_idx] = int(np.count_nonzero(proposal_mask))
                track["last_mask"] = proposal_mask
                track["last_frame"] = frame_idx
                matched_tracks.add(best_idx)
            else:
                active_tracks.append(
                    {
                        "track_id": next_track_id,
                        "masks_by_frame": {frame_idx: proposal_mask},
                        "bboxes_by_frame": {frame_idx: compute_bbox_from_mask(proposal_mask)},
                        "areas_by_frame": {frame_idx: int(np.count_nonzero(proposal_mask))},
                        "last_mask": proposal_mask,
                        "last_frame": frame_idx,
                    }
                )
                matched_tracks.add(len(active_tracks) - 1)
                next_track_id += 1
        still_active: list[dict[str, Any]] = []
        for idx, track in enumerate(active_tracks):
            if idx in matched_tracks or int(track["last_frame"]) == frame_idx:
                still_active.append(track)
            else:
                finished.append(track)
        active_tracks = still_active
    finished.extend(active_tracks)
    results: list[GeneratedBornTrack] = []
    for track in finished:
        if len(track["masks_by_frame"]) < GENERATED_BORN_MIN_DURATION:
            continue
        classification, matched_seg_id, matched_label, appearance_similarity = classify_generated_born_track(
            track_masks=track["masks_by_frame"],
            context_objects=context_objects,
            output_frames=output_frames,
            context_reference_hist_by_seg_id=context_reference_hist_by_seg_id,
        )
        confidence = compute_generated_born_confidence(
            areas_by_frame=track["areas_by_frame"],
            appearance_similarity=appearance_similarity,
            classification=classification,
        )
        displayed = should_display_generated_born_track(classification, confidence)
        color = (255, 0, 255) if classification == "new_object" else (255, 255, 0)
        results.append(
            GeneratedBornTrack(
                track_id=int(track["track_id"]),
                masks_by_frame=dict(track["masks_by_frame"]),
                bboxes_by_frame=dict(track["bboxes_by_frame"]),
                areas_by_frame=dict(track["areas_by_frame"]),
                classification=classification,
                matched_seg_id=matched_seg_id,
                matched_object_label=matched_label,
                appearance_similarity=float(appearance_similarity),
                color=color,
                confidence=float(confidence),
                displayed=bool(displayed),
                prototype_hist=estimate_track_prototype_histogram(
                    frames=output_frames,
                    masks_by_frame=dict(track["masks_by_frame"]),
                    areas_by_frame=dict(track["areas_by_frame"]),
                ),
            )
        )
    refined_results: list[GeneratedBornTrack] = []
    for track in results:
        if track.displayed:
            refined_results.append(
                refine_generated_born_track_with_sam2(
                    frames=output_frames,
                    track=track,
                )
            )
        else:
            refined_results.append(track)
    merged_results = merge_generated_born_tracks(refined_results)
    final_results: list[GeneratedBornTrack] = []
    frame_shape = output_frames[0].shape[:2]
    for track in merged_results:
        displayed = bool(track.displayed)
        if displayed and is_floor_like_generated_born_track(track, frame_shape):
            displayed = False
        if displayed and is_static_small_generated_born_track(track, frame_shape):
            displayed = False
        final_results.append(
            GeneratedBornTrack(
                track_id=track.track_id,
                masks_by_frame=track.masks_by_frame,
                bboxes_by_frame=track.bboxes_by_frame,
                areas_by_frame=track.areas_by_frame,
                classification=track.classification,
                matched_seg_id=track.matched_seg_id,
                matched_object_label=track.matched_object_label,
                appearance_similarity=track.appearance_similarity,
                color=track.color,
                confidence=track.confidence,
                displayed=displayed,
                prototype_hist=track.prototype_hist,
            )
        )
    return suppress_dominated_generated_born_tracks(final_results)


def build_case_analysis_text(
    *,
    sidecar: dict[str, Any],
    summary: dict[str, Any],
    target: TargetSpec,
    mode: str,
    gt_available: bool,
    generated_born_tracks: list[GeneratedBornTrack] | None = None,
) -> list[str]:
    lines: list[str] = []
    lines.append(
        f"Target `{target.object_label}` was selected with `{target.selection_mode}` under mode `{mode}`."
    )
    lines.append(
        "Generated-video analysis prefers context-initialized object tracking. When that fails or is unavailable, it falls back to a target-window foreground proxy."
    )
    lines.append(
        "Overlay videos render motion-selected context objects with stable per-object colors so the same object keeps the same color across context, generated, and GT videos. Low-motion objects are filtered automatically."
    )
    if gt_available:
        lines.append(
            "GT reference curves use synthetic segmentation/depth and provide a stronger baseline for whether the generated size change looks physically plausible."
        )
    born_tracks = generated_born_tracks or []
    if born_tracks:
        new_count = sum(1 for track in born_tracks if track.classification == "new_object")
        duplicate_count = sum(1 for track in born_tracks if track.classification == "duplicate_of_known_object")
        drift_count = sum(1 for track in born_tracks if track.classification == "scale_drift_candidate")
        displayed_count = sum(1 for track in born_tracks if track.displayed)
        lines.append(
            f"Generated-only discovery found {len(born_tracks)} extra track(s): new={new_count}, duplicate={duplicate_count}, scale_drift_candidate={drift_count}."
        )
        lines.append(
            f"Only {displayed_count} high-confidence generated-only track(s) are shown in overlay and single-track panels; lower-confidence flicker-like tracks remain in diagnostics JSON only."
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
    image_positions = np.asarray(features["instances/image_positions"].float_list.value, dtype=np.float32)
    image_positions = image_positions.reshape(num_instances, num_frames, 2).transpose(1, 0, 2)
    is_dynamic = np.asarray(features["instances/is_dynamic"].int64_list.value, dtype=np.int32)
    return {
        "meta": meta,
        "seg_frames": seg_frames,
        "depth_frames": depth_frames,
        "visibility": visibility,
        "image_positions": image_positions,
        "is_dynamic": is_dynamic,
        "num_frames": num_frames,
        "num_instances": num_instances,
    }


def select_movid_target(gt: dict[str, Any], context_index: int) -> TargetSpec:
    seg = gt["seg_frames"][context_index]
    context_len = max(int(context_index) + 1, 1)
    best_seg_id = None
    best_motion = -1.0
    best_area = -1
    for seg_id in sorted(int(v) for v in np.unique(seg) if int(v) != BACKGROUND_SEGMENT_ID):
        instance_idx = seg_id - 1
        if instance_idx < 0 or instance_idx >= int(gt["num_instances"]):
            continue
        visible = gt["visibility"][:context_len, instance_idx] > 0
        coords = gt["image_positions"][:context_len, instance_idx]
        motion = compute_track_motion_score(coords, visible)
        area = int(np.count_nonzero(seg == seg_id))
        dynamic_bonus = 1.0 if int(gt["is_dynamic"][instance_idx]) > 0 else 0.0
        score = motion + dynamic_bonus * 1e-3
        if score > best_motion or (math.isclose(score, best_motion) and area > best_area):
            best_motion = score
            best_area = area
            best_seg_id = seg_id
    if best_seg_id is None:
        raise RuntimeError("No visible foreground object found in MOVI-D context frame.")
    return TargetSpec(
        object_label=f"seg_{best_seg_id}",
        seg_id=int(best_seg_id),
        selection_mode="highest_motion_gt_instance_in_context",
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
    kinematics = np.load(source_sample_dir / "physics" / "rigid_kinematics.npz")
    return {
        "meta": meta,
        "source_meta": source_meta,
        "seg_frames": seg,
        "depth_frames": depth,
        "kinematics": kinematics,
    }


def select_genesis_target(gt: dict[str, Any], context_index: int) -> TargetSpec:
    source_meta = gt["source_meta"]
    seg = np.asarray(gt["seg_frames"][context_index])
    kin = gt["kinematics"]
    seg_ids = np.asarray(kin["seg_ids"], dtype=np.int32)
    com_uv = np.asarray(kin["com_uv"], dtype=np.float32)
    visibility = np.asarray(kin["visibility_mask"], dtype=np.uint8)
    context_len = max(int(context_index) + 1, 1)
    best_seg_id = None
    best_motion = -1.0
    best_area = -1
    for obj_idx, seg_id in enumerate(seg_ids.tolist()):
        seg_id = int(seg_id)
        if seg_id <= 0:
            continue
        visible = visibility[:context_len, obj_idx] > 0
        coords = com_uv[:context_len, obj_idx]
        motion = compute_track_motion_score(coords, visible)
        area = int(np.count_nonzero(seg == seg_id))
        if area <= 0:
            continue
        if motion > best_motion or (math.isclose(motion, best_motion) and area > best_area):
            best_motion = motion
            best_area = area
            best_seg_id = seg_id
    if best_seg_id is not None and best_area > 0:
        role = None
        for obj in source_meta.get("objects", []):
            if int(obj.get("seg_id", -1)) == int(best_seg_id):
                role = str(obj.get("role") or "")
                break
        label = f"seg_{best_seg_id}" if not role else f"seg_{best_seg_id}_{role}"
        return TargetSpec(
            object_label=label,
            seg_id=int(best_seg_id),
            selection_mode="highest_motion_gt_object_in_context",
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
        selection_mode="largest_visible_gt_segment_fallback",
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


def build_generated_curves_from_tracked_masks(
    *,
    output_frames: list[np.ndarray],
    context_frames: int,
    tracked_masks: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    bg_scales: list[float] = []
    visible = 0
    for t, frame in enumerate(output_frames):
        mask = tracked_masks.get(t)
        if mask is None:
            mask = np.zeros(frame.shape[:2], dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
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


def draw_overlay_objects(
    *,
    frame: np.ndarray,
    frame_idx: int,
    masks_by_seg_id: dict[int, dict[int, np.ndarray]],
    overlay_objects: list[OverlayObjectSpec],
) -> np.ndarray:
    vis = np.asarray(frame, dtype=np.uint8).copy()
    for obj in overlay_objects:
        mask = masks_by_seg_id.get(int(obj.seg_id), {}).get(int(frame_idx))
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        bbox = compute_bbox_from_mask(mask)
        vis = overlay_mask(vis, mask, color=obj.color, alpha=0.24)
        vis = draw_rect(vis, bbox, color=obj.color, thickness=2)
    return vis


def draw_generated_born_tracks(
    *,
    frame: np.ndarray,
    frame_idx: int,
    born_tracks: list[GeneratedBornTrack],
) -> np.ndarray:
    vis = np.asarray(frame, dtype=np.uint8).copy()
    for track in born_tracks:
        if not track.displayed:
            continue
        mask = track.masks_by_frame.get(int(frame_idx))
        if mask is None:
            continue
        bbox = track.bboxes_by_frame.get(int(frame_idx))
        vis = overlay_mask(vis, np.asarray(mask, dtype=bool), color=track.color, alpha=0.18)
        vis = draw_rect(vis, bbox, color=track.color, thickness=2)
    return vis


def write_single_track_video(
    *,
    frames: list[np.ndarray],
    masks_by_frame: dict[int, np.ndarray],
    case_dir: Path,
    output_name: str,
    fps: int,
    color: tuple[int, int, int],
    anchor_bbox: tuple[int, int, int, int] | None = None,
) -> Path:
    annotated: list[np.ndarray] = []
    for frame_idx, frame in enumerate(frames):
        vis = np.asarray(frame, dtype=np.uint8).copy()
        mask = masks_by_frame.get(int(frame_idx))
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            bbox = compute_bbox_from_mask(mask)
            vis = overlay_mask(vis, mask, color=color, alpha=0.24)
            vis = draw_rect(vis, bbox, color=color, thickness=2)
        if anchor_bbox is not None:
            vis = draw_anchor_window(vis, anchor_bbox)
        annotated.append(vis)
    out_path = case_dir / output_name
    save_video_frames(out_path, annotated, fps=fps)
    return out_path


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
    tracked_masks: dict[int, np.ndarray] | None = None,
    overlay_objects: list[OverlayObjectSpec] | None = None,
    tracked_masks_by_seg_id: dict[int, dict[int, np.ndarray]] | None = None,
    generated_born_tracks: list[GeneratedBornTrack] | None = None,
) -> Path:
    annotated: list[np.ndarray] = []
    reference = frames[0]
    for frame_idx, frame in enumerate(frames):
        if overlay_objects and tracked_masks_by_seg_id:
            vis = draw_overlay_objects(
                frame=frame,
                frame_idx=frame_idx,
                masks_by_seg_id=tracked_masks_by_seg_id,
                overlay_objects=overlay_objects,
            )
        elif tracked_masks is not None and frame_idx in tracked_masks:
            mask = np.asarray(tracked_masks[frame_idx], dtype=bool)
            bbox = compute_bbox_from_mask(mask)
            vis = overlay_mask(frame, mask, color=(228, 87, 46), alpha=0.28)
            vis = draw_rect(vis, bbox, color=(228, 87, 46), thickness=2)
        else:
            mask = detect_generated_proxy_mask(frame=frame, reference_frame=reference, anchor_bbox=anchor_bbox)
            bbox = compute_bbox_from_mask(mask)
            vis = overlay_mask(frame, mask, color=(228, 87, 46), alpha=0.28)
            vis = draw_rect(vis, bbox, color=(228, 87, 46), thickness=2)
        if generated_born_tracks:
            vis = draw_generated_born_tracks(
                frame=vis,
                frame_idx=frame_idx,
                born_tracks=generated_born_tracks,
            )
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
    overlay_objects: list[OverlayObjectSpec] | None = None,
) -> Path:
    annotated: list[np.ndarray] = []
    for frame_idx, (frame, seg) in enumerate(zip(context_frames, gt_masks)):
        vis = np.asarray(frame, dtype=np.uint8).copy()
        if overlay_objects:
            for obj in overlay_objects:
                mask = np.asarray(seg) == int(obj.seg_id)
                if np.count_nonzero(mask) == 0:
                    continue
                bbox = compute_bbox_from_mask(mask)
                vis = overlay_mask(vis, mask, color=obj.color, alpha=0.24)
                vis = draw_rect(vis, bbox, color=obj.color, thickness=2)
        else:
            mask = np.asarray(seg) == gt_seg_id
            bbox = compute_bbox_from_mask(mask)
            vis = overlay_mask(vis, mask, color=(94, 154, 255), alpha=0.28)
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
    overlay_objects: list[OverlayObjectSpec] | None = None,
) -> Path:
    annotated: list[np.ndarray] = []
    for frame, seg in zip(gt_frames, gt_masks):
        vis = np.asarray(frame, dtype=np.uint8).copy()
        if overlay_objects:
            for obj in overlay_objects:
                mask = np.asarray(seg) == int(obj.seg_id)
                if np.count_nonzero(mask) == 0:
                    continue
                bbox = compute_bbox_from_mask(mask)
                vis = overlay_mask(vis, mask, color=obj.color, alpha=0.24)
                vis = draw_rect(vis, bbox, color=obj.color, thickness=2)
        else:
            mask = np.asarray(seg) == gt_seg_id
            bbox = compute_bbox_from_mask(mask)
            vis = overlay_mask(vis, mask, color=(48, 132, 214), alpha=0.28)
            vis = draw_rect(vis, bbox, color=(48, 132, 214), thickness=2)
        annotated.append(vis)
    out_path = case_dir / "gt_diagnostic.mp4"
    save_video_frames(out_path, annotated, fps=fps)
    return out_path


def analyze_case(sidecar_path: Path, output_root: Path, overwrite: bool) -> dict[str, Any]:
    sidecar = read_json(sidecar_path)
    paths = sidecar.get("paths", {})
    output_video_path = Path(paths["output_video_path"])
    sample_key = sanitize_token(
        f"{sidecar.get('model_name')}__{sidecar.get('dataset')}__{sidecar.get('sample_id')}"
    )
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
    generated_anchor_bbox = None
    context_anchor_bbox = None
    gt_video_frames: list[np.ndarray] = []
    context_video_path = Path(paths["context_video_path"]) if paths.get("context_video_path") else None
    context_video_frames = load_video_frames(context_video_path) if context_video_path and context_video_path.exists() else []
    context_diagnostic_video = None
    generated_track_result: GeneratedTrackResult | None = None
    overlay_objects: list[OverlayObjectSpec] = []
    generated_born_tracks: list[GeneratedBornTrack] = []
    context_single_track_videos: list[dict[str, Any]] = []
    generated_single_track_videos: list[dict[str, Any]] = []
    generated_born_single_track_videos: list[dict[str, Any]] = []

    if mode == "movi_d_gt":
        gt = load_movid_gt(sidecar)
        target = select_movid_target(gt, context_frames - 1)
        overlay_objects = collect_moving_movid_overlay_objects(gt, context_frames - 1)
        if not any(int(obj.seg_id) == int(target.seg_id) for obj in overlay_objects):
            overlay_objects.insert(
                0,
                OverlayObjectSpec(
                    object_label=target.object_label,
                    seg_id=int(target.seg_id),
                    color=color_for_seg_id(target.seg_id),
                    motion_score=float("inf"),
                ),
            )
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
            context_anchor_bbox = clamp_bbox(
                anchor_bbox,
                context_video_frames[0].shape[1] if context_video_frames else int(gt["seg_frames"][0].shape[1]),
                context_video_frames[0].shape[0] if context_video_frames else int(gt["seg_frames"][0].shape[0]),
            )
            generated_anchor_bbox = resize_bbox(
                tuple(float(v) for v in anchor_bbox),
                src_width=int(gt["seg_frames"][0].shape[1]),
                src_height=int(gt["seg_frames"][0].shape[0]),
                dst_width=int(output_frames[0].shape[1]),
                dst_height=int(output_frames[0].shape[0]),
            )
            generated_anchor_bbox = clamp_bbox(
                generated_anchor_bbox,
                output_frames[0].shape[1],
                output_frames[0].shape[0],
            )
        context_masks_by_seg_id = collect_visible_context_masks_by_object(
            seg_frames=gt["seg_frames"],
            overlay_objects=overlay_objects,
            context_frame_count=context_frames,
            out_width=output_frames[0].shape[1],
            out_height=output_frames[0].shape[0],
        )
        generated_track_result = run_generated_multi_object_track_with_sam2(
            frames=output_frames,
            context_masks_by_seg_id=context_masks_by_seg_id,
        )
        target_tracked_masks = None if generated_track_result is None else generated_track_result.masks_by_seg_id.get(int(target.seg_id))
        if target_tracked_masks:
            curves, helper = build_generated_curves_from_tracked_masks(
                output_frames=output_frames,
                context_frames=context_frames,
                tracked_masks=target_tracked_masks,
            )
        else:
            curves, helper = build_generated_proxy_curves(
                output_frames=output_frames,
                context_frames=context_frames,
                anchor_bbox=generated_anchor_bbox,
            )
        context_reference_hist_by_seg_id = {}
        if context_video_frames:
            for obj in overlay_objects:
                hist = None
                for frame_idx in range(min(context_frames, len(context_video_frames), len(gt["seg_frames"]))):
                    mask = np.asarray(gt["seg_frames"][frame_idx]) == int(obj.seg_id)
                    if np.count_nonzero(mask) <= 0:
                        continue
                    hist = estimate_mask_color_histogram(context_video_frames[frame_idx], mask)
                    if hist is not None:
                        break
                context_reference_hist_by_seg_id[int(obj.seg_id)] = hist
        if generated_track_result is not None:
            generated_born_tracks = track_generated_born_objects(
                output_frames=output_frames,
                context_frames=context_frames,
                known_tracks=generated_track_result.masks_by_seg_id,
                context_objects=overlay_objects,
                context_reference_hist_by_seg_id=context_reference_hist_by_seg_id,
            )
        context_overlay_frames = gt["seg_frames"][: min(len(context_video_frames), len(gt["seg_frames"]))]
    elif mode == "genesis_gt":
        gt = load_genesis_gt(sidecar)
        target = select_genesis_target(gt, context_frames - 1)
        overlay_objects = collect_moving_genesis_overlay_objects(gt, context_frames - 1)
        if not any(int(obj.seg_id) == int(target.seg_id) for obj in overlay_objects):
            overlay_objects.insert(
                0,
                OverlayObjectSpec(
                    object_label=target.object_label,
                    seg_id=int(target.seg_id),
                    color=color_for_seg_id(target.seg_id),
                    motion_score=float("inf"),
                ),
            )
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
            context_anchor_bbox = clamp_bbox(
                anchor_bbox,
                context_video_frames[0].shape[1] if context_video_frames else int(gt["seg_frames"][0].shape[1]),
                context_video_frames[0].shape[0] if context_video_frames else int(gt["seg_frames"][0].shape[0]),
            )
            generated_anchor_bbox = resize_bbox(
                tuple(float(v) for v in anchor_bbox),
                src_width=int(gt["seg_frames"][0].shape[1]),
                src_height=int(gt["seg_frames"][0].shape[0]),
                dst_width=int(output_frames[0].shape[1]),
                dst_height=int(output_frames[0].shape[0]),
            )
            generated_anchor_bbox = clamp_bbox(
                generated_anchor_bbox,
                output_frames[0].shape[1],
                output_frames[0].shape[0],
            )
        context_masks_by_seg_id = collect_visible_context_masks_by_object(
            seg_frames=gt["seg_frames"],
            overlay_objects=overlay_objects,
            context_frame_count=context_frames,
            out_width=output_frames[0].shape[1],
            out_height=output_frames[0].shape[0],
        )
        generated_track_result = run_generated_multi_object_track_with_sam2(
            frames=output_frames,
            context_masks_by_seg_id=context_masks_by_seg_id,
        )
        target_tracked_masks = None if generated_track_result is None else generated_track_result.masks_by_seg_id.get(int(target.seg_id))
        if target_tracked_masks:
            curves, helper = build_generated_curves_from_tracked_masks(
                output_frames=output_frames,
                context_frames=context_frames,
                tracked_masks=target_tracked_masks,
            )
        else:
            curves, helper = build_generated_proxy_curves(
                output_frames=output_frames,
                context_frames=context_frames,
                anchor_bbox=generated_anchor_bbox,
            )
        context_reference_hist_by_seg_id = {}
        if context_video_frames:
            for obj in overlay_objects:
                hist = None
                for frame_idx in range(min(context_frames, len(context_video_frames), len(gt["seg_frames"]))):
                    mask = np.asarray(gt["seg_frames"][frame_idx]) == int(obj.seg_id)
                    if np.count_nonzero(mask) <= 0:
                        continue
                    hist = estimate_mask_color_histogram(context_video_frames[frame_idx], mask)
                    if hist is not None:
                        break
                context_reference_hist_by_seg_id[int(obj.seg_id)] = hist
        if generated_track_result is not None:
            generated_born_tracks = track_generated_born_objects(
                output_frames=output_frames,
                context_frames=context_frames,
                known_tracks=generated_track_result.masks_by_seg_id,
                context_objects=overlay_objects,
                context_reference_hist_by_seg_id=context_reference_hist_by_seg_id,
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
        anchor_bbox=generated_anchor_bbox,
        fps=fps,
        tracked_masks=None
        if generated_track_result is None
        else generated_track_result.masks_by_seg_id.get(int(target.seg_id)),
        overlay_objects=overlay_objects if generated_track_result is not None else None,
        tracked_masks_by_seg_id=None if generated_track_result is None else generated_track_result.masks_by_seg_id,
        generated_born_tracks=generated_born_tracks,
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
                overlay_objects=overlay_objects,
            )
        elif mode == "genesis_gt":
            gt_diagnostic_video = write_annotated_gt_video(
                gt_frames=gt_video_frames[:len(gt_curves)],
                gt_masks=list(gt["seg_frames"][:len(gt_curves)]),
                gt_seg_id=target.seg_id,
                case_dir=case_dir,
                fps=fps,
                overlay_objects=overlay_objects,
            )
    if context_video_frames and context_overlay_frames:
        context_diagnostic_video = write_annotated_context_video(
            context_frames=context_video_frames[:len(context_overlay_frames)],
            gt_masks=context_overlay_frames,
            gt_seg_id=target.seg_id,
            case_dir=case_dir,
            fps=fps,
            anchor_bbox=context_anchor_bbox,
            overlay_objects=overlay_objects,
        )
        for obj in overlay_objects:
            masks_by_frame: dict[int, np.ndarray] = {}
            frame_count = min(len(context_video_frames), len(context_overlay_frames))
            for frame_idx in range(frame_count):
                mask = np.asarray(context_overlay_frames[frame_idx]) == int(obj.seg_id)
                if np.count_nonzero(mask) > 0:
                    masks_by_frame[frame_idx] = mask
            if not masks_by_frame:
                continue
            out_path = write_single_track_video(
                frames=context_video_frames[:frame_count],
                masks_by_frame=masks_by_frame,
                case_dir=case_dir,
                output_name=f"context_track_seg_{int(obj.seg_id)}.mp4",
                fps=fps,
                color=obj.color,
                anchor_bbox=context_anchor_bbox if int(obj.seg_id) == int(target.seg_id) else None,
            )
            context_single_track_videos.append(
                {
                    "seg_id": int(obj.seg_id),
                    "object_label": obj.object_label,
                    "path": str(out_path),
                }
            )
    if generated_track_result is not None:
        for obj in overlay_objects:
            masks_by_frame = generated_track_result.masks_by_seg_id.get(int(obj.seg_id), {})
            if not masks_by_frame:
                continue
            out_path = write_single_track_video(
                frames=output_frames,
                masks_by_frame=masks_by_frame,
                case_dir=case_dir,
                output_name=f"generated_track_seg_{int(obj.seg_id)}.mp4",
                fps=fps,
                color=obj.color,
                anchor_bbox=generated_anchor_bbox if int(obj.seg_id) == int(target.seg_id) else None,
            )
            generated_single_track_videos.append(
                {
                    "seg_id": int(obj.seg_id),
                    "object_label": obj.object_label,
                    "path": str(out_path),
                }
            )
    for track in generated_born_tracks:
        if not track.displayed:
            continue
        out_path = write_single_track_video(
            frames=output_frames,
            masks_by_frame=track.masks_by_frame,
            case_dir=case_dir,
            output_name=f"generated_born_track_{int(track.track_id)}.mp4",
            fps=fps,
            color=track.color,
        )
        generated_born_single_track_videos.append(
            {
                "track_id": int(track.track_id),
                "classification": track.classification,
                "matched_seg_id": track.matched_seg_id,
                "matched_object_label": track.matched_object_label,
                "path": str(out_path),
            }
        )
    analysis_lines = build_case_analysis_text(
        sidecar=sidecar,
        summary=summary,
        target=target,
        mode=mode,
        gt_available=bool(gt_curves),
        generated_born_tracks=generated_born_tracks,
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
        "generated_tracking": None
        if generated_track_result is None
        else {
            "source": generated_track_result.source,
            "num_tracked_frames": max(
                (len(frame_masks) for frame_masks in generated_track_result.masks_by_seg_id.values()),
                default=0,
            ),
            "tracked_seg_ids": sorted(int(seg_id) for seg_id in generated_track_result.masks_by_seg_id),
        },
        "generated_born_tracks": [
            {
                "track_id": int(track.track_id),
                "classification": track.classification,
                "matched_seg_id": track.matched_seg_id,
                "matched_object_label": track.matched_object_label,
                "appearance_similarity": round(float(track.appearance_similarity), 6),
                "confidence": round(float(track.confidence), 6),
                "displayed": bool(track.displayed),
                "num_frames": len(track.masks_by_frame),
                "start_frame": min(track.masks_by_frame) if track.masks_by_frame else None,
                "end_frame": max(track.masks_by_frame) if track.masks_by_frame else None,
                "max_area": max(track.areas_by_frame.values()) if track.areas_by_frame else 0,
            }
            for track in generated_born_tracks
        ],
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
            "context_single_track_videos": context_single_track_videos,
            "generated_single_track_videos": generated_single_track_videos,
            "generated_born_single_track_videos": generated_born_single_track_videos,
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
