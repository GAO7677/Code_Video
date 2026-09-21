"""Shared math and contracts for the RGB-context to PyBullet pilot.

The vision process is allowed to consume RGB0--RGB7, timestamps, calibrated
camera parameters, the pilot family, and the fixed 0.11 m sphere-radius prior.
Per-episode simulator state and collision blueprints are evaluation-only.
"""
from __future__ import annotations

import hashlib
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


def robust_terminal_velocity(
    centers: np.ndarray,
    times: np.ndarray,
    *,
    degree: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit all observed 3D centers and evaluate the derivative at RGB7.

    A two-pass Tukey-style reweighting limits a single bad depth/mask frame
    without silently reducing the estimate to the final two frames.
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
    weights = np.linspace(0.65, 1.0, OBSERVED_FRAMES)
    coefficients = None
    residual_norm = None
    for _ in range(3):
        root_w = np.sqrt(weights)[:, None]
        coefficients, *_ = np.linalg.lstsq(design * root_w, centers * root_w, rcond=None)
        fitted = design @ coefficients
        residual_norm = np.linalg.norm(centers - fitted, axis=1)
        median = float(np.median(residual_norm))
        mad = float(np.median(np.abs(residual_norm - median)))
        scale = max(1.4826 * mad, 1e-5)
        normalized = residual_norm / (4.685 * scale)
        robust = np.square(np.clip(1.0 - normalized * normalized, 0.0, None))
        weights = np.linspace(0.65, 1.0, OBSERVED_FRAMES) * np.maximum(robust, 0.05)
    assert coefficients is not None and residual_norm is not None
    velocity = coefficients[1]
    diagnostics = {
        "method": f"robust_polynomial_degree_{degree}_all_8_frames",
        "weights": weights.tolist(),
        "fit_rmse_m": float(np.sqrt(np.mean(residual_norm ** 2))),
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

