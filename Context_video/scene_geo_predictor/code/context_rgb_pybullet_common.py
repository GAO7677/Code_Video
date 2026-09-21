"""Shared math and contracts for the RGB-context to PyBullet pilot.

The vision process is allowed to consume RGB0--RGB7, timestamps, calibrated
camera parameters, the pilot family, and the fixed 0.11 m sphere-radius prior.
Per-episode simulator state and collision blueprints are evaluation-only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


OBSERVED_FRAMES = 8
FUTURE_FRAMES = 41
FPS = 30
SIM_HZ = 240
BALL_RADIUS_M = 0.11


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(rgb: np.ndarray) -> str:
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError("RGB bundle must be uint8 [T,H,W,3]")
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def camera_calibration(camera: Any, width: int, height: int) -> dict[str, Any]:
    """Return an OpenCV world-to-camera calibration for the fixed renderer."""
    eye = np.asarray(camera.eye, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up_world = np.asarray(camera.up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    rotation = np.stack((right, -camera_up, forward))
    world_to_camera = np.column_stack((rotation, -rotation @ eye))
    focal = float(height) / (2.0 * math.tan(math.radians(float(camera.yfov_deg)) / 2.0))
    intrinsic = np.asarray(
        [[focal, 0.0, (width - 1.0) / 2.0],
         [0.0, focal, (height - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
        raise ValueError("camera rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999999:
        raise ValueError("camera rotation is not right handed")
    return {
        "schema": "calibrated_fixed_context_camera_v1",
        "resolution_wh": [int(width), int(height)],
        "intrinsic_K": intrinsic.tolist(),
        "world_to_camera_3x4": world_to_camera.tolist(),
        "camera_center_world": eye.tolist(),
        "yfov_deg": float(camera.yfov_deg),
        "coordinate_convention": "OpenCV: +x image-right, +y image-down, +z forward",
        "source": "fixed sensor calibration; not estimated object state or scene geometry",
    }


def crop_transform(source_hw: tuple[int, int]) -> dict[str, Any]:
    """Match official VGGT crop preprocessing for a common source resolution."""
    height, width = map(int, source_hw)
    if height < 2 or width < 2:
        raise ValueError("source image is too small")
    new_width = 518
    new_height = int(round(height * new_width / width / 14) * 14)
    if new_height < 1:
        raise ValueError("unsupported source aspect ratio")
    crop_top = max(0, (new_height - 518) // 2)
    output_height = min(new_height, 518)
    scale_x = new_width / width
    scale_y = new_height / height
    affine = np.asarray(
        [[scale_x, 0.0, 0.5 * scale_x - 0.5],
         [0.0, scale_y, 0.5 * scale_y - 0.5 - crop_top],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return {
        "source_hw": [height, width],
        "resized_hw": [new_height, new_width],
        "processed_hw": [output_height, new_width],
        "crop_top": int(crop_top),
        "affine": affine,
    }


def resize_crop_mask(mask: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    """Apply VGGT crop preprocessing to one source-resolution boolean mask."""
    import cv2

    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or list(mask.shape) != transform["source_hw"]:
        raise ValueError("mask does not match the declared source transform")
    new_height, new_width = transform["resized_hw"]
    resized = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    crop_top = int(transform["crop_top"])
    output_height, output_width = transform["processed_hw"]
    cropped = resized[crop_top:crop_top + output_height, :output_width]
    if cropped.shape != (output_height, output_width):
        raise ValueError("processed mask shape mismatch")
    return cropped.astype(bool)


def rgb_motion_circle_prompt(frames: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Find the moving sphere at RGB7 using pixels only.

    Circular edge candidates are ranked by RGB6->RGB7 change, deviation from
    the eight-frame temporal median, and center/rim contrast.  The returned
    box is deliberately a little wider than the detected circle for SAM2.
    """
    import cv2

    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != OBSERVED_FRAMES or frames.shape[-1] != 3:
        raise ValueError("expected uint8 RGB frames [8,H,W,3]")
    height, width = frames.shape[1:3]
    gray = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=16,
        param1=80,
        param2=14,
        minRadius=6,
        maxRadius=22,
    )
    if circles is None or circles.shape[1] == 0:
        raise ValueError("RGB-only moving-sphere detector found no circular candidate")

    adjacent_change = np.mean(
        np.abs(frames[-1].astype(np.float32) - frames[-2].astype(np.float32)), axis=2
    )
    temporal_median = np.median(frames.astype(np.float32), axis=0)
    median_change = np.mean(np.abs(frames[-1].astype(np.float32) - temporal_median), axis=2)
    rows, cols = np.ogrid[:height, :width]
    candidates: list[dict[str, float]] = []
    for x, y, radius in circles[0]:
        distance2 = (cols - float(x)) ** 2 + (rows - float(y)) ** 2
        disk = distance2 <= (1.15 * float(radius)) ** 2
        core = distance2 <= (0.8 * float(radius)) ** 2
        rim = disk & ~core
        motion = float(np.mean(adjacent_change[disk]))
        temporal = float(np.mean(median_change[disk]))
        contrast = abs(float(np.mean(gray[core])) - float(np.mean(gray[rim]))) if np.any(rim) else 0.0
        score = motion + 0.5 * temporal + 0.15 * contrast
        candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
                "adjacent_motion": motion,
                "temporal_change": temporal,
                "center_rim_contrast": contrast,
                "score": score,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    prompt_radius = 1.36 * best["radius"]
    box = np.asarray(
        [
            best["x"] - prompt_radius,
            best["y"] - prompt_radius,
            best["x"] + prompt_radius,
            best["y"] + prompt_radius,
        ],
        dtype=np.float32,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    return box, {
        "method": "RGB7_Hough_circle_ranked_by_RGB_temporal_motion_v1",
        "uses_rgb_frames": [0, 1, 2, 3, 4, 5, 6, 7],
        "selected": best,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "sam2_prompt_box_xyxy": box.tolist(),
        "gt_or_future_used": False,
    }


def estimate_metric_centers(
    depth: np.ndarray,
    masks: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    radius_m: float = BALL_RADIUS_M,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Anchor VGGT depth with a known sphere silhouette and recover 3D centers.

    VGGT depth has an arbitrary global scale.  Each observed sphere silhouette
    supplies an independent metric front-surface depth estimate; their robust
    median is the only fitted scale.  No simulator position is consumed.
    """
    import cv2

    depth = np.asarray(depth, dtype=np.float64)
    masks = np.asarray(masks)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != masks.shape or depth.shape[0] != OBSERVED_FRAMES:
        raise ValueError("depth and masks must have shape [8,H,W]")
    if masks.dtype != np.bool_ or intrinsic.shape != (3, 3) or world_to_camera.shape != (3, 4):
        raise ValueError("invalid mask or camera arrays")
    if not np.isfinite(depth).all() or np.any(depth <= 0) or radius_m <= 0:
        raise ValueError("depth and radius must be finite and positive")

    inverse_k = np.linalg.inv(intrinsic)
    rotation = world_to_camera[:, :3]
    translation = world_to_camera[:, 3]
    per_frame: list[dict[str, Any]] = []
    scales = []
    raw_front_depths = []
    rays = []
    for frame_index, mask in enumerate(masks):
        area = int(mask.sum())
        if area < 20:
            raise ValueError(f"RGB{frame_index} mask has only {area} pixels")
        rows, cols = np.where(mask)
        center_uv = np.asarray([float(cols.mean()), float(rows.mean()), 1.0])
        ray = inverse_k @ center_uv
        ray /= ray[2]
        unit_ray = ray / np.linalg.norm(ray)
        equivalent_radius_px = math.sqrt(area / math.pi)
        focal = math.sqrt(float(intrinsic[0, 0] * intrinsic[1, 1]))
        center_z_silhouette = focal * float(radius_m) / equivalent_radius_px
        front_z_silhouette = center_z_silhouette - float(radius_m) * float(unit_ray[2])

        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        central = mask & (distance >= max(1.0, 0.45 * float(distance.max())))
        if int(central.sum()) < 4:
            central = mask
        raw_front = float(np.median(depth[frame_index][central]))
        scale = front_z_silhouette / raw_front
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid metric scale at RGB{frame_index}")
        scales.append(scale)
        raw_front_depths.append(raw_front)
        rays.append((ray, unit_ray))
        per_frame.append(
            {
                "frame": frame_index,
                "mask_area_pixels": area,
                "mask_center_uv": center_uv[:2].tolist(),
                "equivalent_radius_px": equivalent_radius_px,
                "silhouette_center_z_m": center_z_silhouette,
                "silhouette_front_z_m": front_z_silhouette,
                "raw_vggt_front_depth": raw_front,
                "scale_candidate": scale,
                "central_depth_pixels": int(central.sum()),
            }
        )
    scales_array = np.asarray(scales, dtype=np.float64)
    median_scale = float(np.median(scales_array))
    mad = float(np.median(np.abs(scales_array - median_scale)))
    threshold = max(3.0 * 1.4826 * mad, 0.12 * median_scale)
    inliers = np.abs(scales_array - median_scale) <= threshold
    if int(inliers.sum()) < 5:
        raise ValueError(f"only {int(inliers.sum())}/8 depth-scale anchors agree")
    metric_scale = float(np.median(scales_array[inliers]))

    centers_camera = []
    centers_world = []
    for frame_index, (ray, unit_ray) in enumerate(rays):
        surface = ray * (raw_front_depths[frame_index] * metric_scale)
        center_camera = surface + unit_ray * float(radius_m)
        center_world = rotation.T @ (center_camera - translation)
        centers_camera.append(center_camera)
        centers_world.append(center_world)
        per_frame[frame_index]["scale_inlier"] = bool(inliers[frame_index])
        per_frame[frame_index]["estimated_center_camera_m"] = center_camera.tolist()
        per_frame[frame_index]["estimated_center_world_m"] = center_world.tolist()
    centers = np.asarray(centers_world, dtype=np.float64)
    diagnostics = {
        "method": "VGGT_depth_scaled_by_known_sphere_silhouette_v1",
        "known_radius_m": float(radius_m),
        "metric_scale": metric_scale,
        "scale_candidates": scales_array.tolist(),
        "scale_inliers": inliers.tolist(),
        "scale_relative_mad": mad / median_scale,
        "per_frame": per_frame,
        "centers_camera_m": np.asarray(centers_camera).tolist(),
        "depth_semantics": "VGGT raw camera-Z treated as relative; one global multiplicative scale fitted",
    }
    return centers, diagnostics


def static_point_cloud(
    depth: np.ndarray,
    confidence: np.ndarray,
    processed_rgb: np.ndarray,
    dynamic_masks: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    metric_scale: float,
    *,
    max_points: int = 80000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fuse eight fixed-camera depth maps into a bounded static point sample."""
    import cv2

    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    confidence = np.asarray(confidence, dtype=np.float64)
    processed_rgb = np.asarray(processed_rgb)
    dynamic_masks = np.asarray(dynamic_masks)
    if not (depth.shape == confidence.shape == dynamic_masks.shape):
        raise ValueError("static fusion arrays must share [8,H,W]")
    if processed_rgb.shape != (*depth.shape, 3):
        raise ValueError("processed RGB must have shape [8,H,W,3]")
    if dynamic_masks.dtype != np.bool_ or metric_scale <= 0:
        raise ValueError("invalid dynamic masks or metric scale")
    height, width = depth.shape[1:]
    excluded = np.zeros((height, width), dtype=bool)
    kernel = np.ones((9, 9), dtype=np.uint8)
    for mask in dynamic_masks:
        excluded |= cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    median_depth = np.median(depth, axis=0) * float(metric_scale)
    median_confidence = np.median(confidence, axis=0)
    median_rgb = np.median(processed_rgb.astype(np.float32), axis=0).astype(np.uint8)
    confidence_threshold = float(np.quantile(median_confidence[~excluded], 0.20))
    valid = (
        ~excluded
        & np.isfinite(median_depth)
        & (median_depth > 0.05)
        & (median_depth < 20.0)
        & np.isfinite(median_confidence)
        & (median_confidence >= confidence_threshold)
    )
    rows, cols = np.where(valid)
    camera_rays = np.stack([cols, rows, np.ones_like(cols)], axis=1) @ np.linalg.inv(intrinsic).T
    camera_points = camera_rays * median_depth[rows, cols, None]
    rotation = world_to_camera[:, :3]
    translation = world_to_camera[:, 3]
    world_points = (camera_points - translation) @ rotation
    colors = median_rgb[rows, cols]
    point_confidence = median_confidence[rows, cols]
    finite = np.isfinite(world_points).all(axis=1)
    world_points, colors, point_confidence = world_points[finite], colors[finite], point_confidence[finite]
    if len(world_points) > max_points:
        indices = np.linspace(0, len(world_points) - 1, max_points, dtype=np.int64)
        world_points, colors, point_confidence = (
            world_points[indices], colors[indices], point_confidence[indices]
        )
    report = {
        "method": "median_8_frame_VGGT_depth_dynamic_mask_excluded",
        "confidence_quantile": 0.20,
        "confidence_threshold": confidence_threshold,
        "excluded_dynamic_pixels": int(excluded.sum()),
        "valid_static_pixels": int(valid.sum()),
        "saved_points": int(len(world_points)),
        "max_points": int(max_points),
    }
    return (
        world_points.astype(np.float32),
        colors.astype(np.uint8),
        point_confidence.astype(np.float32),
        report,
    )


def robust_terminal_velocity(
    centers: np.ndarray,
    times: np.ndarray,
    *,
    degree: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit all observed 3D centers and evaluate the derivative at RGB7.

    Deterministic minimal-subset fits identify depth/mask outliers, followed by
    a recency-weighted fit over every admitted frame.  This avoids silently
    reducing the estimate to the final two frames.
    """
    centers = np.asarray(centers, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if centers.shape != (OBSERVED_FRAMES, 3) or times.shape != (OBSERVED_FRAMES,):
        raise ValueError("expected centers [8,3] and times [8]")
    if not np.isfinite(centers).all() or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("nonfinite centers or invalid observation times")
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")

    shifted = times - times[-1]
    design = np.stack([shifted ** power for power in range(degree + 1)], axis=1)
    subset_size = degree + 1
    best_score: tuple[float, float, tuple[int, ...]] | None = None
    best_residual = None
    for subset in itertools.combinations(range(OBSERVED_FRAMES), subset_size):
        indices = np.asarray(subset, dtype=np.int64)
        candidate, *_ = np.linalg.lstsq(design[indices], centers[indices], rcond=None)
        residual = np.linalg.norm(centers - design @ candidate, axis=1)
        score = (float(np.median(residual)), float(np.mean(np.sort(residual)[:6])), subset)
        if best_score is None or score < best_score:
            best_score = score
            best_residual = residual
    assert best_score is not None and best_residual is not None
    median = float(np.median(best_residual))
    mad = float(np.median(np.abs(best_residual - median)))
    inlier_threshold = max(median + 3.0 * 1.4826 * mad, 0.005)
    inliers = best_residual <= inlier_threshold
    if int(inliers.sum()) < 5:
        order = np.argsort(best_residual)
        inliers = np.zeros(OBSERVED_FRAMES, dtype=bool)
        inliers[order[:5]] = True
    weights = np.linspace(0.65, 1.0, OBSERVED_FRAMES) * inliers.astype(np.float64)
    root_w = np.sqrt(weights[inliers])[:, None]
    coefficients, *_ = np.linalg.lstsq(
        design[inliers] * root_w,
        centers[inliers] * root_w,
        rcond=None,
    )
    fitted = design @ coefficients
    residual_norm = np.linalg.norm(centers - fitted, axis=1)
    velocity = coefficients[1]
    diagnostics = {
        "method": f"deterministic_subset_robust_polynomial_degree_{degree}_over_8_frames",
        "weights": weights.tolist(),
        "inlier_frames": np.flatnonzero(inliers).astype(int).tolist(),
        "excluded_frames": np.flatnonzero(~inliers).astype(int).tolist(),
        "inlier_threshold_m": inlier_threshold,
        "selected_seed_subset": list(best_score[2]),
        "fit_rmse_m": float(np.sqrt(np.mean(residual_norm ** 2))),
        "inlier_fit_rmse_m": float(np.sqrt(np.mean(residual_norm[inliers] ** 2))),
        "fit_max_error_m": float(residual_norm.max()),
        "fitted_p7": coefficients[0].tolist(),
        "finite_difference_p6_p7_mps": ((centers[-1] - centers[-2]) / (times[-1] - times[-2])).tolist(),
    }
    return velocity.astype(np.float64), diagnostics


def velocity_errors(estimated: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    estimated = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    est_norm = float(np.linalg.norm(estimated))
    target_norm = float(np.linalg.norm(target))
    if est_norm <= 1e-10 or target_norm <= 1e-10:
        direction = None
    else:
        cosine = float(np.clip(np.dot(estimated, target) / (est_norm * target_norm), -1.0, 1.0))
        direction = float(np.degrees(np.arccos(cosine)))
    return {
        "vector_error_mps": float(np.linalg.norm(estimated - target)),
        "magnitude_error_mps": abs(est_norm - target_norm),
        "direction_error_deg": direction,
        "estimated_speed_mps": est_norm,
        "target_speed_mps": target_norm,
    }


def rolling_omega(linear_velocity: np.ndarray, support_normal: np.ndarray | None = None) -> np.ndarray:
    """Return the explicit pure-rolling hypothesis omega=(n x v)/r."""
    velocity = np.asarray(linear_velocity, dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0] if support_normal is None else support_normal, dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    return np.cross(normal, velocity) / BALL_RADIUS_M


def yaw_quaternion(yaw_rad: float) -> list[float]:
    return [0.0, 0.0, math.sin(0.5 * float(yaw_rad)), math.cos(0.5 * float(yaw_rad))]


def wrap_angle_deg(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)
