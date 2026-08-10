#!/usr/bin/env python3
"""Compute all non-neural metrics, merge cached metrics, and render audit overlays."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.analyze_legacy_ti2v_object_ablation_raft_motion import (  # noqa: E402
    compare_flow,
    flow_to_bgr,
    load_dynamic_rois,
)
from AAA_my_test.object_query_ablation_metrics.common import (  # noqa: E402
    BASELINE_TRACKS,
    CASE,
    FRAME_COUNT,
    HEIGHT,
    MODE_IDS,
    OBJECTS,
    OBJECT_LABELS,
    OUTPUT_ROOT,
    RAFT_ROOT,
    SEED,
    SOURCE_ROOT,
    SOURCE_STATES,
    SOURCE_VIDEO,
    WIDTH,
    atomic_json,
    load_inventory,
    load_video_frames,
    safe_id,
    video_manifest,
)
from AAA_my_test.object_query_ablation_metrics.metric_definitions import (  # noqa: E402
    METRIC_DEFINITIONS,
)


VBENCH_FIELDS = (
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
COLORS = ((239, 112, 55), (43, 189, 190))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-overlays", action="store_true")
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="compute the complete metric report without rendering per-seed audit videos",
    )
    parser.add_argument("--only", default="", help="optional exact ablation id")
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rounded(value: Any, digits: int = 8) -> float | None:
    number = finite(value)
    return None if number is None else round(number, digits)


def numeric_difference(left: Any, right: Any) -> float | None:
    """Subtract two optional measurements without turning missing tracking into zero."""
    left_number, right_number = finite(left), finite(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def load_tracks(video_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = OUTPUT_ROOT / "tracks" / f"{safe_id(video_id)}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        return arrays["tracks"].astype(np.float32), arrays["visibility"].astype(bool)


def load_masks(video_id: str) -> np.ndarray:
    path = OUTPUT_ROOT / "masks" / f"{safe_id(video_id)}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        return arrays["masks"].astype(bool)


def dynamic_rois_from_tracks(
    tracks: np.ndarray,
    width: int,
    height: int,
    dilate_px: int,
    reference_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build reference-frozen flow ROIs from the same 8 points per object."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    rois: dict[str, np.ndarray] = {}
    for object_index, object_name in enumerate(OBJECTS):
        part = slice(object_index * 8, (object_index + 1) * 8)
        masks = np.zeros((FRAME_COUNT - 1, height, width), dtype=bool)
        for frame_index in range(FRAME_COUNT - 1):
            points = tracks[frame_index, part].copy()
            points[:, 0] *= width / WIDTH
            points[:, 1] *= height / HEIGHT
            points = np.rint(points).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            hull = cv2.convexHull(points.reshape(-1, 1, 2))
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 1)
            masks[frame_index] = cv2.dilate(mask, kernel, iterations=1) > 0
        rois[object_name] = masks
    rois["all_objects"] = np.logical_or(rois["object_A"], rois["object_B"])
    return rois, {
        "definition": (
            f"{reference_id}-frozen CoTracker point convex hull per frame, "
            f"dilated {dilate_px}px at {width}x{height}"
        ),
        "reference_id": reference_id,
        "source_resolution": [HEIGHT, WIDTH],
        "flow_resolution": [height, width],
        "mean_area_pixels": {
            name: round(float(mask.sum(axis=(1, 2)).mean()), 3)
            for name, mask in rois.items()
        },
    }


def robust_centers(tracks: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = np.full((FRAME_COUNT, 2, 2), np.nan, dtype=np.float32)
    valid = np.zeros((FRAME_COUNT, 2), dtype=bool)
    for object_index in range(2):
        part = slice(object_index * 8, (object_index + 1) * 8)
        for frame_index in range(FRAME_COUNT):
            use = visibility[frame_index, part] & np.isfinite(tracks[frame_index, part]).all(axis=1)
            if int(use.sum()) >= 4:
                centers[frame_index, object_index] = np.median(tracks[frame_index, part][use], axis=0)
                valid[frame_index, object_index] = True
    return centers, valid


def projected_gt() -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    states = np.load(SOURCE_STATES, allow_pickle=False)
    positions = states["positions"][:FRAME_COUNT].astype(np.float64)
    eye = states["camera_eye"].astype(np.float64)
    target = states["camera_target"].astype(np.float64)
    up = states["camera_up"].astype(np.float64)
    source_width = int(states["frame_width"][0])
    source_height = int(states["frame_height"][0])
    meta = json.loads((SOURCE_ROOT / "meta.json").read_text(encoding="utf-8"))
    focal = source_height / (2.0 * math.tan(math.radians(float(meta["camera"]["yfov_deg"])) / 2.0))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    centers = np.zeros((FRAME_COUNT, 2, 2), dtype=np.float32)
    for frame_index in range(FRAME_COUNT):
        for object_index in range(2):
            relative = positions[frame_index, object_index] - eye
            depth = float(np.dot(relative, forward))
            x = source_width / 2 + focal * float(np.dot(relative, right)) / depth
            y = source_height / 2 - focal * float(np.dot(relative, camera_up)) / depth
            centers[frame_index, object_index] = (x * WIDTH / source_width, y * HEIGHT / source_height)

    ball = positions[:, 0]
    box = positions[:, 1]
    quaternions = states["quats"][:FRAME_COUNT, 1].astype(np.float64)
    half = np.asarray([meta["objects"][1]["size"][key] for key in ("hx", "hy", "hz")], dtype=np.float64)
    radius = float(meta["objects"][0]["size"]["radius"])
    contact = np.zeros(FRAME_COUNT, dtype=bool)
    distances = np.zeros(FRAME_COUNT, dtype=np.float64)
    for frame_index, quaternion in enumerate(quaternions):
        x, y, z, w = quaternion
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        local = rotation.T @ (ball[frame_index] - box[frame_index])
        closest = np.clip(local, -half, half)
        distance = float(np.linalg.norm(local - closest))
        distances[frame_index] = max(0.0, distance - radius)
        contact[frame_index] = distance <= radius + 1e-3
    contact_frame = first_sustained(contact, 2)
    audit = {
        "source_resolution": [source_height, source_width],
        "output_resolution": [HEIGHT, WIDTH],
        "mapping": "pinhole projection from states.npz camera, then independent x/y stretch",
        "focal_px_source": round(focal, 6),
        "gt_contact_definition": "sphere-to-oriented-box surface distance <= 1e-3m",
        "gt_contact_frame": contact_frame,
        "gt_surface_distance_m": [round(float(value), 7) for value in distances],
    }
    return centers, contact, contact_frame, audit


def first_sustained(flags: np.ndarray, count: int) -> int | None:
    for start in range(0, len(flags) - count + 1):
        if bool(np.all(flags[start : start + count])):
            return start
    return None


def bbox_diagonal(mask: np.ndarray) -> float:
    y, x = np.where(mask)
    if not len(x):
        return 1.0
    return float(math.hypot(float(x.max() - x.min() + 1), float(y.max() - y.min() + 1)))


def trajectory_metrics(
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_center_valid: np.ndarray,
    reference_tracks: np.ndarray,
    reference_visibility: np.ndarray,
    reference_centers: np.ndarray,
    reference_center_valid: np.ndarray,
    object_index: int,
    diagonal: float,
) -> dict[str, Any]:
    part = slice(object_index * 8, (object_index + 1) * 8)
    point_valid = (
        candidate_visibility[:, part]
        & reference_visibility[:, part]
        & np.isfinite(candidate_tracks[:, part]).all(axis=-1)
        & np.isfinite(reference_tracks[:, part]).all(axis=-1)
    )
    point_distance = np.linalg.norm(candidate_tracks[:, part] - reference_tracks[:, part], axis=-1)
    point_values = point_distance[point_valid]
    center_valid = candidate_center_valid[:, object_index] & reference_center_valid[:, object_index]
    center_distance = np.linalg.norm(
        candidate_centers[:, object_index] - reference_centers[:, object_index], axis=-1
    )
    center_values = center_distance[center_valid]
    last_frame = int(np.where(center_valid)[0][-1]) if center_valid.any() else None
    delta = 4
    velocity_valid = center_valid[:-delta] & center_valid[delta:]
    candidate_velocity = (candidate_centers[delta:, object_index] - candidate_centers[:-delta, object_index]) / delta
    reference_velocity = (reference_centers[delta:, object_index] - reference_centers[:-delta, object_index]) / delta
    vector_error = np.linalg.norm(candidate_velocity - reference_velocity, axis=-1)
    candidate_speed = np.linalg.norm(candidate_velocity, axis=-1)
    reference_speed = np.linalg.norm(reference_velocity, axis=-1)
    speed_error = np.abs(candidate_speed - reference_speed)
    direction_valid = velocity_valid & (candidate_speed >= 0.25) & (reference_speed >= 0.25)
    cosine = np.sum(candidate_velocity * reference_velocity, axis=-1) / (
        candidate_speed * reference_speed + 1e-8
    )
    direction = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return {
        "center_ade_px": rounded(center_values.mean() if center_values.size else None),
        "center_ade_norm": rounded(center_values.mean() / diagonal if center_values.size else None),
        "center_fde_px": rounded(center_distance[last_frame] if last_frame is not None else None),
        "center_fde_norm": rounded(center_distance[last_frame] / diagonal if last_frame is not None else None),
        "last_common_visible_frame": last_frame,
        "center_valid_frames": int(center_valid.sum()),
        "point_ade_px": rounded(point_values.mean() if point_values.size else None),
        "point_ade_norm": rounded(point_values.mean() / diagonal if point_values.size else None),
        "point_valid_count": int(point_valid.sum()),
        "pck_native": {
            str(threshold): rounded(np.mean(point_values < threshold) if point_values.size else None)
            for threshold in (16, 32, 64)
        },
        "pck_normalized": {
            str(alpha): rounded(np.mean(point_values < alpha * diagonal) if point_values.size else None)
            for alpha in (0.05, 0.10, 0.20)
        },
        "velocity_vector_error_px_per_frame": rounded(vector_error[velocity_valid].mean() if velocity_valid.any() else None),
        "velocity_speed_error_px_per_frame": rounded(speed_error[velocity_valid].mean() if velocity_valid.any() else None),
        "velocity_direction_error_deg": rounded(direction[direction_valid].mean() if direction_valid.any() else None),
        "velocity_valid_count": int(velocity_valid.sum()),
        "direction_valid_count": int(direction_valid.sum()),
        "series": {
            "center_distance_px": [rounded(value, 5) if valid else None for value, valid in zip(center_distance, center_valid, strict=True)],
            "velocity_vector_error_px_per_frame": [rounded(value, 5) if valid else None for value, valid in zip(vector_error, velocity_valid, strict=True)],
        },
    }


def exact_gt_trajectory_metrics(
    candidate_centers: np.ndarray,
    candidate_valid: np.ndarray,
    gt_centers: np.ndarray,
    object_index: int,
    diagonal: float,
) -> dict[str, Any]:
    # states.npz provides exact rigid-body centers, but no correspondence for the
    # eight CoTracker surface points.  Keep this comparison center-only rather
    # than manufacturing eight identical pseudo-points (which would make a
    # misleading "GT PCK").
    center_valid = candidate_valid[:, object_index] & np.isfinite(
        candidate_centers[:, object_index]
    ).all(axis=-1)
    center_distance = np.linalg.norm(
        candidate_centers[:, object_index] - gt_centers[:, object_index], axis=-1
    )
    center_values = center_distance[center_valid]
    last_frame = int(np.where(center_valid)[0][-1]) if center_valid.any() else None

    delta = 4
    velocity_valid = center_valid[:-delta] & center_valid[delta:]
    candidate_velocity = (
        candidate_centers[delta:, object_index]
        - candidate_centers[:-delta, object_index]
    ) / delta
    gt_velocity = (
        gt_centers[delta:, object_index] - gt_centers[:-delta, object_index]
    ) / delta
    vector_error = np.linalg.norm(candidate_velocity - gt_velocity, axis=-1)
    candidate_speed = np.linalg.norm(candidate_velocity, axis=-1)
    gt_speed = np.linalg.norm(gt_velocity, axis=-1)
    speed_error = np.abs(candidate_speed - gt_speed)
    direction_valid = velocity_valid & (candidate_speed >= 0.25) & (gt_speed >= 0.25)
    cosine = np.sum(candidate_velocity * gt_velocity, axis=-1) / (
        candidate_speed * gt_speed + 1e-8
    )
    direction = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return {
        "center_ade_px": rounded(center_values.mean() if center_values.size else None),
        "center_ade_norm": rounded(center_values.mean() / diagonal if center_values.size else None),
        "center_fde_px": rounded(center_distance[last_frame] if last_frame is not None else None),
        "center_fde_norm": rounded(center_distance[last_frame] / diagonal if last_frame is not None else None),
        "last_common_visible_frame": last_frame,
        "center_valid_frames": int(center_valid.sum()),
        "velocity_vector_error_px_per_frame": rounded(
            vector_error[velocity_valid].mean() if velocity_valid.any() else None
        ),
        "velocity_speed_error_px_per_frame": rounded(
            speed_error[velocity_valid].mean() if velocity_valid.any() else None
        ),
        "velocity_direction_error_deg": rounded(
            direction[direction_valid].mean() if direction_valid.any() else None
        ),
        "velocity_valid_count": int(velocity_valid.sum()),
        "direction_valid_count": int(direction_valid.sum()),
        "series": {
            "center_distance_px": [
                rounded(value, 5) if valid else None
                for value, valid in zip(center_distance, center_valid, strict=True)
            ],
            "velocity_vector_error_px_per_frame": [
                rounded(value, 5) if valid else None
                for value, valid in zip(vector_error, velocity_valid, strict=True)
            ],
        },
        "scope_note": (
            "center/velocity only: simulator GT has rigid-body centers but no "
            "CoTracker surface-point correspondence"
        ),
    }


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    y, x = np.where(mask)
    return None if not len(x) else (float(x.mean()), float(y.mean()))


def shifted_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(mask.astype(np.uint8), matrix, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST) > 0


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def mask_geometry(mask: np.ndarray) -> tuple[float, float | None, float | None]:
    area = float(mask.sum())
    if area <= 0:
        return 0.0, None, None
    y, x = np.where(mask)
    aspect = float((x.max() - x.min() + 1) / max(y.max() - y.min() + 1, 1))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = float(4 * math.pi * area / max(perimeter * perimeter, 1e-8))
    return area, aspect, circularity


def shape_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    raw, aligned, area_error, aspect_error, circularity_error = [], [], [], [], []
    for left, right in zip(candidate, reference, strict=True):
        raw.append(iou(left, right))
        lc, rc = mask_centroid(left), mask_centroid(right)
        if lc is None or rc is None:
            aligned.append(1.0 if lc is None and rc is None else 0.0)
        else:
            aligned.append(iou(shifted_mask(left, rc[0] - lc[0], rc[1] - lc[1]), right))
        la, laspect, lcirc = mask_geometry(left)
        ra, raspect, rcirc = mask_geometry(right)
        area_error.append(abs(math.log((la + 1.0) / (ra + 1.0))))
        aspect_error.append(None if laspect is None or raspect is None else abs(math.log(laspect / raspect)))
        circularity_error.append(None if lcirc is None or rcirc is None else abs(lcirc - rcirc))
    aspect_values = [value for value in aspect_error if value is not None]
    circularity_values = [value for value in circularity_error if value is not None]
    return {
        "raw_iou_mean": rounded(np.mean(raw)),
        "center_aligned_iou_mean": rounded(np.mean(aligned)),
        "area_log_ratio_error_mean": rounded(np.mean(area_error)),
        "aspect_log_ratio_error_mean": rounded(np.mean(aspect_values) if aspect_values else None),
        "circularity_error_mean": rounded(np.mean(circularity_values) if circularity_values else None),
        "candidate_nonempty_rate": rounded(np.mean(candidate.any(axis=(1, 2)))),
        "reference_nonempty_rate": rounded(np.mean(reference.any(axis=(1, 2)))),
        "series": {
            "raw_iou": [rounded(value, 5) for value in raw],
            "center_aligned_iou": [rounded(value, 5) for value in aligned],
            "area_log_ratio_error": [rounded(value, 5) for value in area_error],
        },
    }


def mask_contact(masks: np.ndarray, threshold_px: float = 2.0) -> tuple[np.ndarray, list[float | None], int | None]:
    flags = np.zeros(FRAME_COUNT, dtype=bool)
    distances: list[float | None] = []
    for frame_index in range(FRAME_COUNT):
        first, second = masks[frame_index]
        if not first.any() or not second.any():
            distances.append(None)
            continue
        if np.logical_and(first, second).any():
            distance = 0.0
        else:
            distance_map = cv2.distanceTransform((~second).astype(np.uint8), cv2.DIST_L2, 5)
            distance = float(distance_map[first].min())
        distances.append(distance)
        flags[frame_index] = distance <= threshold_px
    return flags, distances, first_sustained(flags, 2)


def pixel_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_small = np.stack([cv2.resize(frame, (320, 176), interpolation=cv2.INTER_AREA) for frame in left])
    right_small = np.stack([cv2.resize(frame, (320, 176), interpolation=cv2.INTER_AREA) for frame in right])
    difference = left_small.astype(np.float32) - right_small.astype(np.float32)
    mse_by_frame = np.mean(np.square(difference), axis=(1, 2, 3), dtype=np.float64)
    mae_by_frame = np.mean(np.abs(difference), axis=(1, 2, 3), dtype=np.float64) / 255.0
    ssim_by_frame = np.asarray(
        [structural_similarity(a, b, channel_axis=-1, data_range=255) for a, b in zip(left_small, right_small, strict=True)]
    )
    temporal = np.mean(
        np.abs(np.diff(left_small.astype(np.int16), axis=0) - np.diff(right_small.astype(np.int16), axis=0)),
        axis=(1, 2, 3), dtype=np.float64,
    ) / 255.0
    mse = float(mse_by_frame.mean())
    return {
        "ssim_mean": rounded(ssim_by_frame.mean()),
        "psnr_db": None if mse == 0 else rounded(10 * math.log10(255.0**2 / mse)),
        "mae_0_1": rounded(mae_by_frame.mean()),
        "temporal_delta_mae_0_1": rounded(temporal.mean()),
        "series": {
            "ssim": [rounded(value, 6) for value in ssim_by_frame],
            "mae_0_1": [rounded(value, 6) for value in mae_by_frame],
            "temporal_delta_mae_0_1": [rounded(value, 6) for value in temporal],
        },
    }


def vbench_scores(video: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
    manifest = video_manifest(video)
    result = {}
    for field in VBENCH_FIELDS:
        payload = manifest.get(field)
        score = finite(payload.get("score")) if isinstance(payload, dict) else None
        result[field] = {
            "score": rounded(score),
            "baseline": rounded(baseline.get(field)),
            "delta": rounded(score - baseline[field]) if score is not None and field in baseline else None,
        }
    return result


def post_contact_velocity_error(metrics: dict[str, Any], start: int | None, window: int = 8) -> float | None:
    if start is None:
        return None
    series = metrics["series"]["velocity_vector_error_px_per_frame"]
    begin, end = min(start, len(series)), min(start + window, len(series))
    values = [float(value) for value in series[begin:end] if value is not None]
    return None if not values else float(np.mean(values))


def draw_label(image: np.ndarray, text: str, y: int = 22) -> None:
    cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame_index: int,
    gt_centers: np.ndarray,
) -> np.ndarray:
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for object_index, color_rgb in enumerate(COLORS):
        color = tuple(reversed(color_rgb))
        part = range(object_index * 8, (object_index + 1) * 8)
        for point_index in part:
            for previous in range(max(0, frame_index - 20), frame_index):
                if visibility[previous, point_index] and visibility[previous + 1, point_index]:
                    p0 = tuple(np.rint(tracks[previous, point_index]).astype(int))
                    p1 = tuple(np.rint(tracks[previous + 1, point_index]).astype(int))
                    cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
            if visibility[frame_index, point_index]:
                point = tuple(np.rint(tracks[frame_index, point_index]).astype(int))
                cv2.circle(canvas, point, 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)
        gt_path = np.rint(gt_centers[: frame_index + 1, object_index]).astype(np.int32)
        if len(gt_path) > 1:
            cv2.polylines(canvas, [gt_path], False, (210, 60, 210), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, tuple(gt_path[-1]), (210, 60, 210), cv2.MARKER_CROSS, 12, 2)
    return canvas


def tint_masks(frame: np.ndarray, masks: np.ndarray, gt_centers: np.ndarray, frame_index: int) -> np.ndarray:
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()
    for object_index, color_rgb in enumerate(COLORS):
        mask = masks[object_index]
        color = np.asarray(tuple(reversed(color_rgb)), dtype=np.uint8)
        overlay[mask] = color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(v) for v in color), 2)
        center = mask_centroid(mask)
        if center is not None:
            cv2.circle(canvas, tuple(np.rint(center).astype(int)), 5, tuple(int(v) for v in color), -1)
        gt = tuple(np.rint(gt_centers[frame_index, object_index]).astype(int))
        cv2.drawMarker(canvas, gt, (210, 60, 210), cv2.MARKER_CROSS, 12, 2)
    return cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)


def resize_panel(frame: np.ndarray, width: int = 426, height: int = 235) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    with imageio.get_writer(temporary, fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
        for frame in frames:
            # libx264/yuv420p requires even spatial dimensions.  Audit layouts
            # can be odd after concatenating a title strip, so pad only the
            # bottom/right edge without resampling any measured pixels.
            pad_bottom = frame.shape[0] % 2
            pad_right = frame.shape[1] % 2
            if pad_bottom or pad_right:
                frame = cv2.copyMakeBorder(
                    frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE
                )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    temporary.replace(path)


def render_trajectory_overlay(
    path: Path,
    candidate_id: str,
    baseline_frames: np.ndarray,
    candidate_frames: np.ndarray,
    source_frames: np.ndarray,
    baseline_tracks: tuple[np.ndarray, np.ndarray],
    candidate_tracks: tuple[np.ndarray, np.ndarray],
    source_tracks: tuple[np.ndarray, np.ndarray],
    gt_centers: np.ndarray,
) -> None:
    outputs = []
    for frame_index in range(FRAME_COUNT):
        panels = []
        for label, frames, tracks in (
            ("Baseline", baseline_frames, baseline_tracks),
            ("Ablation", candidate_frames, candidate_tracks),
            ("Source / simulator GT", source_frames, source_tracks),
        ):
            panel = draw_tracks(frames[frame_index], tracks[0], tracks[1], frame_index, gt_centers)
            panel = resize_panel(panel)
            draw_label(panel, f"{label} | F{frame_index:02d}")
            panels.append(panel)
        body = np.concatenate(panels, axis=1)
        header = np.full((42, body.shape[1], 3), (221, 232, 226), np.uint8)
        draw_label(header, f"{candidate_id} | circles/trails=actual CoTracker inputs | magenta=projected simulator center", 27)
        outputs.append(np.concatenate([header, body], axis=0))
    write_video(path, outputs)


def render_mask_overlay(
    path: Path,
    candidate_id: str,
    baseline_frames: np.ndarray,
    candidate_frames: np.ndarray,
    source_frames: np.ndarray,
    baseline_masks: np.ndarray,
    candidate_masks: np.ndarray,
    source_masks: np.ndarray,
    gt_centers: np.ndarray,
    distances: dict[str, list[float | None]],
) -> None:
    outputs = []
    for frame_index in range(FRAME_COUNT):
        panels = []
        for label, frames, masks, key in (
            ("Baseline SAM2", baseline_frames, baseline_masks, "baseline"),
            ("Ablation SAM2", candidate_frames, candidate_masks, "candidate"),
            ("Source SAM2 + GT", source_frames, source_masks, "source"),
        ):
            panel = tint_masks(frames[frame_index], masks[frame_index], gt_centers, frame_index)
            panel = resize_panel(panel)
            distance = distances[key][frame_index]
            draw_label(panel, f"{label} | F{frame_index:02d} | mask gap {distance if distance is not None else 'N/A'}", 22)
            panels.append(panel)
        body = np.concatenate(panels, axis=1)
        header = np.full((42, body.shape[1], 3), (232, 224, 211), np.uint8)
        draw_label(header, f"{candidate_id} | orange=sphere mask | cyan=box mask | actual masks used by shape/contact metrics", 27)
        outputs.append(np.concatenate([header, body], axis=0))
    write_video(path, outputs)


def render_pixel_overlay(
    path: Path,
    candidate_id: str,
    baseline_frames: np.ndarray,
    candidate_frames: np.ndarray,
    source_frames: np.ndarray,
    candidate_masks: np.ndarray,
    baseline_masks: np.ndarray,
) -> None:
    outputs = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    for frame_index in range(FRAME_COUNT):
        baseline = cv2.cvtColor(resize_panel(baseline_frames[frame_index]), cv2.COLOR_RGB2BGR)
        candidate = cv2.cvtColor(resize_panel(candidate_frames[frame_index]), cv2.COLOR_RGB2BGR)
        source = cv2.cvtColor(resize_panel(source_frames[frame_index]), cv2.COLOR_RGB2BGR)
        difference = cv2.absdiff(candidate, baseline)
        difference = cv2.applyColorMap(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
        union = np.logical_or(candidate_masks[frame_index].any(axis=0), baseline_masks[frame_index].any(axis=0)).astype(np.uint8)
        outside = cv2.dilate(union, kernel, iterations=1) == 0
        outside = cv2.resize(outside.astype(np.uint8), (426, 235), interpolation=cv2.INTER_NEAREST)
        outside_view = candidate.copy()
        outside_view[outside == 0] = 127
        panels = [baseline, candidate, source, difference, outside_view]
        labels = ["Baseline", "Ablation", "Source GT render", "|Ablation-Baseline|", "Outside-object input"]
        for panel, label in zip(panels, labels, strict=True):
            draw_label(panel, f"{label} | F{frame_index:02d}")
        body = np.concatenate(panels, axis=1)
        header = np.full((42, body.shape[1], 3), (225, 224, 236), np.uint8)
        draw_label(header, f"{candidate_id} | exact decoded frames and outside mask used by pixel/LPIPS diagnostics", 27)
        outputs.append(np.concatenate([header, body], axis=0))
    write_video(path, outputs)


def render_raft_overlay(
    path: Path,
    candidate_id: str,
    baseline_flow: np.ndarray,
    candidate_flow: np.ndarray,
    source_flow: np.ndarray,
    baseline_rois: dict[str, np.ndarray],
    source_rois: dict[str, np.ndarray],
    max_magnitude: float,
) -> None:
    outputs = []
    for frame_index in range(48):
        base = flow_to_bgr(baseline_flow[frame_index], max_magnitude)
        cand = flow_to_bgr(candidate_flow[frame_index], max_magnitude)
        source = flow_to_bgr(source_flow[frame_index], max_magnitude)
        diff_base = np.linalg.norm(candidate_flow[frame_index].astype(np.float32) - baseline_flow[frame_index].astype(np.float32), axis=0)
        diff_source = np.linalg.norm(candidate_flow[frame_index].astype(np.float32) - source_flow[frame_index].astype(np.float32), axis=0)
        diff_base = cv2.applyColorMap(np.uint8(np.clip(diff_base / max(max_magnitude, 1e-6), 0, 1) * 255), cv2.COLORMAP_INFERNO)
        diff_source = cv2.applyColorMap(np.uint8(np.clip(diff_source / max(max_magnitude, 1e-6), 0, 1) * 255), cv2.COLORMAP_INFERNO)
        panels = [base, cand, diff_base, source, cand.copy(), diff_source]
        for panel, roi_set in zip(
            panels,
            (baseline_rois, baseline_rois, baseline_rois, source_rois, source_rois, source_rois),
            strict=True,
        ):
            for object_index, object_name in enumerate(OBJECTS):
                contours, _ = cv2.findContours(
                    roi_set[object_name][frame_index].astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                color = tuple(reversed(COLORS[object_index]))
                cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)
        labels = ["Baseline flow", "Ablation flow", "EPE vs Baseline", "Source flow", "Ablation flow", "EPE vs Source"]
        rows = []
        for row_index in range(2):
            row = []
            for panel, label in zip(panels[row_index * 3 : row_index * 3 + 3], labels[row_index * 3 : row_index * 3 + 3], strict=True):
                tile = cv2.resize(panel, (320, 176), interpolation=cv2.INTER_AREA)
                draw_label(tile, f"{label} | F{frame_index:02d}->{frame_index+1:02d}")
                row.append(tile)
            rows.append(np.concatenate(row, axis=1))
        body = np.concatenate(rows, axis=0)
        header = np.full((42, body.shape[1], 3), (218, 235, 230), np.uint8)
        draw_label(header, f"{candidate_id} | RAFT C_T_SKHT_V2 | outlines=reference-frozen ROIs | scale {max_magnitude:.3f}px", 27)
        outputs.append(np.concatenate([header, body], axis=0))
    write_video(path, outputs)


def main() -> None:
    args = parse_args()
    videos = load_inventory(include_source=True)
    video_map = {row["id"]: row for row in videos}
    candidates = [row for row in videos if row["id"] not in {"baseline", "source_gt_video"}]
    if args.only:
        candidates = [row for row in candidates if row["id"] == args.only]
        if not candidates:
            raise ValueError(f"unknown ablation id: {args.only}")
    perceptual_path = OUTPUT_ROOT / "perceptual/perceptual_metrics.json"
    perceptual_payload = json.loads(perceptual_path.read_text(encoding="utf-8"))
    perceptual = {row["id"]: row for row in perceptual_payload["records"]}

    baseline_frames, _ = load_video_frames(Path(video_map["baseline"]["path"]))
    source_frames, _ = load_video_frames(SOURCE_VIDEO)
    baseline_tracks = load_tracks("baseline")
    source_tracks = load_tracks("source_gt_video")
    baseline_centers, baseline_center_valid = robust_centers(*baseline_tracks)
    source_centers, source_center_valid = robust_centers(*source_tracks)
    baseline_masks = load_masks("baseline")
    source_masks = load_masks("source_gt_video")
    diagonals = [bbox_diagonal(baseline_masks[0, index]) for index in range(2)]
    gt_centers, gt_contact_flags, gt_contact_frame, gt_audit = projected_gt()
    gt_center_valid = np.ones((FRAME_COUNT, 2), dtype=bool)
    baseline_gt = [
        exact_gt_trajectory_metrics(
            baseline_centers, baseline_center_valid, gt_centers, object_index, diagonals[object_index]
        )
        for object_index in range(2)
    ]
    source_gt = [
        exact_gt_trajectory_metrics(
            source_centers, source_center_valid, gt_centers, object_index, diagonals[object_index]
        )
        for object_index in range(2)
    ]
    _, baseline_contact_distances, baseline_contact_frame = mask_contact(baseline_masks)
    _, source_contact_distances, source_contact_frame = mask_contact(source_masks)

    baseline_vbench_manifest = video_manifest(video_map["baseline"])
    baseline_vbench = {
        field: float(baseline_vbench_manifest[field]["score"])
        for field in VBENCH_FIELDS
        if isinstance(baseline_vbench_manifest.get(field), dict)
        and finite(baseline_vbench_manifest[field].get("score")) is not None
    }

    baseline_flow = np.load(RAFT_ROOT / "flows/baseline.npy", mmap_mode="r")
    source_flow = np.load(OUTPUT_ROOT / "raft/source_gt_video.npy", mmap_mode="r")
    baseline_rois, baseline_roi_audit = load_dynamic_rois(BASELINE_TRACKS, 640, 352, 6)
    source_rois, source_roi_audit = dynamic_rois_from_tracks(
        source_tracks[0], 640, 352, 6, "source_gt_video"
    )
    max_flow_magnitude = max(
        float(np.percentile(np.linalg.norm(baseline_flow.astype(np.float32), axis=1), 99.5)),
        float(np.percentile(np.linalg.norm(source_flow.astype(np.float32), axis=1), 99.5)),
    )
    overlay_root = OUTPUT_ROOT / "overlays"
    records = []
    csv_rows = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate["id"])
        safe = safe_id(candidate_id)
        candidate_frames, _ = load_video_frames(Path(candidate["path"]))
        candidate_tracks = load_tracks(candidate_id)
        candidate_centers, candidate_center_valid = robust_centers(*candidate_tracks)
        candidate_masks = load_masks(candidate_id)
        candidate_flow = np.load(RAFT_ROOT / "flows" / f"{safe}.npy", mmap_mode="r")
        object_records = {}
        for object_index, object_name in enumerate(OBJECTS):
            baseline_reference = trajectory_metrics(
                candidate_tracks[0], candidate_tracks[1], candidate_centers, candidate_center_valid,
                baseline_tracks[0], baseline_tracks[1], baseline_centers, baseline_center_valid,
                object_index, diagonals[object_index],
            )
            source_reference = trajectory_metrics(
                candidate_tracks[0], candidate_tracks[1], candidate_centers, candidate_center_valid,
                source_tracks[0], source_tracks[1], source_centers, source_center_valid,
                object_index, diagonals[object_index],
            )
            gt_reference = exact_gt_trajectory_metrics(
                candidate_centers, candidate_center_valid, gt_centers,
                object_index, diagonals[object_index],
            )
            gt_reference["center_ade_change_vs_baseline_norm"] = rounded(
                numeric_difference(
                    gt_reference["center_ade_norm"],
                    baseline_gt[object_index]["center_ade_norm"],
                )
            )
            gt_reference["center_fde_change_vs_baseline_norm"] = rounded(
                numeric_difference(
                    gt_reference["center_fde_norm"],
                    baseline_gt[object_index]["center_fde_norm"],
                )
            )
            gt_reference["velocity_vector_error_change_vs_baseline_px_per_frame"] = rounded(
                numeric_difference(
                    gt_reference["velocity_vector_error_px_per_frame"],
                    baseline_gt[object_index]["velocity_vector_error_px_per_frame"],
                )
            )
            object_records[object_name] = {
                "label": OBJECT_LABELS[object_name],
                "normalizer_f00_bbox_diagonal_px": rounded(diagonals[object_index]),
                "baseline_reference": baseline_reference,
                "source_video_reference": source_reference,
                "simulator_gt_reference": gt_reference,
                "shape_vs_baseline": shape_metrics(candidate_masks[:, object_index], baseline_masks[:, object_index]),
                "shape_vs_source": shape_metrics(candidate_masks[:, object_index], source_masks[:, object_index]),
                "perceptual": perceptual[candidate_id]["objects"][object_name],
            }

        candidate_contact_flags, candidate_contact_distances, candidate_contact_frame = mask_contact(candidate_masks)
        contact_error_change = None
        if candidate_contact_frame is not None and baseline_contact_frame is not None and gt_contact_frame is not None:
            contact_error_change = abs(candidate_contact_frame - gt_contact_frame) - abs(baseline_contact_frame - gt_contact_frame)
        post_candidate = post_contact_velocity_error(object_records["object_A"]["simulator_gt_reference"], gt_contact_frame)
        post_baseline = post_contact_velocity_error(baseline_gt[0], gt_contact_frame)
        post_change = None if post_candidate is None or post_baseline is None else post_candidate - post_baseline

        scope_metrics = {}
        for scope_name, roi_key in (("object_A", "object_A"), ("object_B", "object_B"), ("all_objects", "all_objects")):
            scope_metrics[scope_name] = {
                "vs_baseline": compare_flow(
                    baseline_flow, candidate_flow, baseline_rois[roi_key], 0.25
                ),
                "vs_source": compare_flow(
                    source_flow, candidate_flow, source_rois[roi_key], 0.25
                ),
            }

        target_region = str(candidate.get("region") or "")
        other_object = None
        if candidate.get("target_scope") == "single_object" and target_region in OBJECTS:
            other_name = "object_B" if target_region == "object_A" else "object_A"
            other_object = {
                "object": other_name,
                "center_ade_norm": object_records[other_name]["baseline_reference"]["center_ade_norm"],
                "center_ade_px": object_records[other_name]["baseline_reference"]["center_ade_px"],
            }

        assets = {
            "trajectory": f"overlays/trajectory/{safe}.mp4",
            "mask": f"overlays/mask/{safe}.mp4",
            "pixel": f"overlays/pixel/{safe}.mp4",
            "raft": f"overlays/raft/{safe}.mp4",
            "input_video": str(candidate["path"]),
            "perceptual": {
                object_name: {
                    reference_id: perceptual[candidate_id]["objects"][object_name][reference_id]["montage"]
                    for reference_id in ("baseline", "source_gt_video")
                }
                for object_name in OBJECTS
            },
        }
        if not args.skip_overlays and (
            args.overwrite_overlays or not (OUTPUT_ROOT / assets["trajectory"]).is_file()
        ):
            render_trajectory_overlay(
                OUTPUT_ROOT / assets["trajectory"], candidate_id,
                baseline_frames, candidate_frames, source_frames,
                baseline_tracks, candidate_tracks, source_tracks, gt_centers,
            )
        if not args.skip_overlays and (
            args.overwrite_overlays or not (OUTPUT_ROOT / assets["mask"]).is_file()
        ):
            render_mask_overlay(
                OUTPUT_ROOT / assets["mask"], candidate_id,
                baseline_frames, candidate_frames, source_frames,
                baseline_masks, candidate_masks, source_masks, gt_centers,
                {"baseline": baseline_contact_distances, "candidate": candidate_contact_distances, "source": source_contact_distances},
            )
        if not args.skip_overlays and (
            args.overwrite_overlays or not (OUTPUT_ROOT / assets["pixel"]).is_file()
        ):
            render_pixel_overlay(
                OUTPUT_ROOT / assets["pixel"], candidate_id,
                baseline_frames, candidate_frames, source_frames,
                candidate_masks, baseline_masks,
            )
        if not args.skip_overlays and (
            args.overwrite_overlays or not (OUTPUT_ROOT / assets["raft"]).is_file()
        ):
            render_raft_overlay(
                OUTPUT_ROOT / assets["raft"], candidate_id,
                baseline_flow, candidate_flow, source_flow,
                baseline_rois, source_rois, max_flow_magnitude,
            )

        record = {
            "id": candidate_id,
            "protocol": candidate["protocol"],
            "target_scope": candidate.get("target_scope"),
            "region": candidate.get("region"),
            "mask_mode": candidate.get("mask_mode"),
            "operator_id": MODE_IDS.get(str(candidate.get("mask_mode")), str(candidate.get("mask_mode"))),
            "objects": object_records,
            "other_object": other_object,
            "interaction": {
                "candidate_contact_frame": candidate_contact_frame,
                "baseline_contact_frame": baseline_contact_frame,
                "source_mask_contact_frame": source_contact_frame,
                "simulator_gt_contact_frame": gt_contact_frame,
                "contact_time_error_change_frames": contact_error_change,
                "post_contact_velocity_error_change_px_per_frame": rounded(post_change),
                "candidate_contact_by_frame": candidate_contact_flags.tolist(),
                "candidate_mask_gap_px": [rounded(value, 5) for value in candidate_contact_distances],
            },
            "raft": scope_metrics,
            "pixel": {
                "vs_baseline": pixel_metrics(candidate_frames, baseline_frames),
                "vs_source": pixel_metrics(candidate_frames, source_frames),
            },
            "outside_object_lpips": perceptual[candidate_id]["references"],
            "vbench": vbench_scores(candidate, baseline_vbench),
            "assets": assets,
        }
        records.append(record)
        csv_rows.append(
            {
                "id": candidate_id,
                "protocol": candidate["protocol"],
                "target_scope": candidate.get("target_scope"),
                "region": candidate.get("region"),
                "operator_id": record["operator_id"],
                "mask_mode": candidate.get("mask_mode"),
                "object_A_center_ade_vs_baseline_norm": object_records["object_A"]["baseline_reference"]["center_ade_norm"],
                "object_A_gt_center_ade_change_norm": object_records["object_A"]["simulator_gt_reference"]["center_ade_change_vs_baseline_norm"],
                "object_B_center_ade_vs_baseline_norm": object_records["object_B"]["baseline_reference"]["center_ade_norm"],
                "object_B_gt_center_ade_change_norm": object_records["object_B"]["simulator_gt_reference"]["center_ade_change_vs_baseline_norm"],
                "contact_time_error_change_frames": contact_error_change,
                "ssim_vs_baseline": record["pixel"]["vs_baseline"]["ssim_mean"],
                "ssim_vs_source": record["pixel"]["vs_source"]["ssim_mean"],
            }
        )
        print(f"[{index:02d}/{len(candidates):02d}] metrics + overlays {candidate_id}", flush=True)

    source_asset = OUTPUT_ROOT / "source_gt_49f.mp4"
    if not source_asset.is_file() or args.overwrite_overlays:
        write_video(
            source_asset,
            [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in source_frames],
        )
    baseline_summary = {
        "video_path": str(video_map["baseline"]["path"]),
        "simulator_gt": dict(zip(OBJECTS, baseline_gt, strict=True)),
        "source_video_gt": dict(zip(OBJECTS, source_gt, strict=True)),
        "vbench": vbench_scores(video_map["baseline"], baseline_vbench),
        "contact_frame": baseline_contact_frame,
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": CASE,
        "seed": SEED,
        "video_count": len(videos) - 1,
        "ablation_count": len(records),
        "references": {
            "baseline": str(video_map["baseline"]["path"]),
            "source_gt_video": str(source_asset),
            "simulator_states": str(SOURCE_STATES),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "protocol": {
            "trajectory_center": "median of at least four visible CoTracker points per object/frame",
            "point_tracker": "CoTracker3 offline scaled checkpoint, same F00 points for every video",
            "mask": "SAM2.1 Hiera Large video propagation from the same F00 points and boxes",
            "velocity_delta_frames": 4,
            "contact_mask_threshold_px": 2.0,
            "contact_sustain_frames": 2,
            "gt_projection": gt_audit,
            "raft_roi": {
                "vs_baseline": baseline_roi_audit,
                "vs_source_gt_video": source_roi_audit,
            },
            "perceptual": {key: perceptual_payload[key] for key in ("dino", "dino_frames", "lpips", "lpips_frames", "alignment", "crop_sides_px")},
        },
        "baseline": baseline_summary,
        "records": records,
    }
    atomic_json(OUTPUT_ROOT / "report.json", report)
    csv_path = OUTPUT_ROOT / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"ready: {OUTPUT_ROOT / 'report.json'}")


if __name__ == "__main__":
    main()
