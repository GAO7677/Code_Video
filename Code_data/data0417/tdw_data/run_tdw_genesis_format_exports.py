from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import os
import subprocess
import time

import imageio.v2 as imageio
import numpy as np

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.obi import Obi
from tdw.controller import Controller
from tdw.obi_data.cloth.volume_type import ClothVolumeType
from tdw.obi_data.collision_materials.collision_material import CollisionMaterial
from tdw.obi_data.fluids.disk_emitter import DiskEmitter
from tdw.obi_data.fluids.fluid import Fluid
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera

OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_genesis_format_exports")
HTML_PATH = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_genesis_format_exports.html")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = int(os.environ.get("TDW_PORT", "1181"))
BUILD_ADDRESS = "127.0.0.1"
BUILD_BOOT_WAIT = int(os.environ.get("TDW_BUILD_BOOT_WAIT", "18"))
EXPORT_RESOLUTION = (960, 720)
FPS = 24
GRAVITY_YUP = np.array([0.0, -9.81, 0.0], dtype=np.float32)
GRAVITY_ZUP = np.array([0.0, 0.0, -9.81], dtype=np.float32)
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
YUP_TO_ZUP_ROT = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)


SCENES: Dict[str, Dict[str, Any]] = {
    "building_site": {
        "name": "building_site",
        "skybox": "bergen_4k",
        "camera_position": {"x": -4.8, "y": 2.4, "z": 4.8},
        "look_at": {"x": 0.0, "y": 0.6, "z": 0.0},
        "field_of_view": 72,
    },
    "suburb_scene_2023": {
        "name": "suburb_scene_2023",
        "skybox": "sunset_fairway_4k",
        "camera_position": {"x": -3.4, "y": 1.7, "z": -0.3},
        "look_at": {"x": 0.0, "y": 1.0, "z": 0.0},
        "field_of_view": 72,
    },
    "mm_craftroom_1a": {
        "name": "mm_craftroom_1a",
        "skybox": "kiara_1_dawn_4k",
        "camera_position": {"x": -0.6156362579862034, "y": 1.85, "z": -1.6914467174146353},
        "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
        "field_of_view": 78,
    },
}


CASES: List[Dict[str, Any]] = [
    {
        "kind": "rigid",
        "case_name": "rigid_box_high_drop",
        "scene_key": "building_site",
        "primary_name": "iron_box",
        "scene_label": "rigid",
        "frames": 120,
        "warmup": 6,
        "object": {
            "model_name": "iron_box",
            "position": {"x": 0.0, "y": 1.7, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.3, "y": 0.4, "z": 0.2},
        },
    },
    {
        "kind": "rigid",
        "case_name": "rigid_camera_box_center_tumble",
        "scene_key": "mm_craftroom_1a",
        "primary_name": "camera_box",
        "scene_label": "rigid",
        "frames": 110,
        "warmup": 6,
        "camera_override": {
            "camera_position": {"x": -0.74, "y": 1.94, "z": -2.06},
            "look_at": {"x": 0.05, "y": 0.95, "z": -0.08},
            "field_of_view": 74,
        },
        "object": {
            "model_name": "camera_box",
            "position": {"x": 0.0, "y": 0.95, "z": 0.0},
            "rotation": {"x": 0.0, "y": 18.0, "z": 0.0},
            "velocity": {"x": 0.38, "y": 0.15, "z": 0.12},
            "angular_velocity": {"x": 0.3, "y": 0.75, "z": 0.2},
        },
    },
    {
        "kind": "rigid",
        "case_name": "rigid_serving_bowl_high_drop",
        "scene_key": "building_site",
        "primary_name": "serving_bowl",
        "scene_label": "rigid",
        "frames": 124,
        "warmup": 6,
        "object": {
            "model_name": "serving_bowl",
            "position": {"x": 0.0, "y": 1.65, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.25, "y": 0.2, "z": 0.15},
        },
    },
    {
        "kind": "rigid",
        "case_name": "rigid_hiker_backpack_arc_left",
        "scene_key": "suburb_scene_2023",
        "primary_name": "hiker_backpack",
        "scene_label": "rigid",
        "frames": 124,
        "warmup": 6,
        "camera_override": {
            "camera_position": {"x": -2.65, "y": 1.62, "z": -0.55},
            "look_at": {"x": 0.08, "y": 0.92, "z": 0.02},
            "field_of_view": 68,
        },
        "object": {
            "model_name": "hiker_backpack",
            "position": {"x": -0.55, "y": 1.15, "z": -0.2},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.72, "y": 0.72, "z": 0.44},
            "angular_velocity": {"x": 0.45, "y": 0.3, "z": 0.65},
        },
    },
    {
        "kind": "rigid",
        "case_name": "rigid_cardboard_box_entry_right",
        "scene_key": "building_site",
        "primary_name": "box_18inx18inx12in_cardboard",
        "scene_label": "rigid",
        "frames": 120,
        "warmup": 6,
        "object": {
            "model_name": "box_18inx18inx12in_cardboard",
            "position": {"x": 1.25, "y": 0.55, "z": 0.05},
            "rotation": {"x": 0.0, "y": 30.0, "z": 0.0},
            "velocity": {"x": -1.2, "y": 0.02, "z": -0.06},
            "angular_velocity": {"x": 0.08, "y": 0.18, "z": 0.36},
        },
    },
    {
        "kind": "cloth",
        "case_name": "cloth_drop_ground",
        "scene_key": "suburb_scene_2023",
        "primary_name": "cloth",
        "scene_label": "cloth",
        "frames": 180,
        "warmup": 8,
        "cloth": {
            "material": "cotton",
            "position": {"x": 0.0, "y": 2.2, "z": 0.0},
            "rotation": {"x": 10.0, "y": 0.0, "z": 0.0},
        },
        "support": {
            "model_name": "sphere",
            "library": "models_flex.json",
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale_factor": {"x": 0.7, "y": 0.7, "z": 0.7},
            "mass": 40.0,
            "dynamic_friction": 0.85,
            "static_friction": 0.9,
            "bounciness": 0.02,
        },
    },
    {
        "kind": "cloth",
        "case_name": "cloth_drop_box_craftroom",
        "scene_key": "mm_craftroom_1a",
        "primary_name": "cloth",
        "scene_label": "cloth",
        "frames": 180,
        "warmup": 8,
        "camera_override": {
            "camera_position": {"x": -0.78, "y": 1.92, "z": -2.08},
            "look_at": {"x": 0.02, "y": 0.95, "z": 0.0},
            "field_of_view": 72,
        },
        "cloth": {
            "material": "cotton",
            "position": {"x": 0.02, "y": 2.05, "z": 0.02},
            "rotation": {"x": 18.0, "y": 0.0, "z": 12.0},
        },
        "support": {
            "model_name": "camera_box",
            "library": "models_core.json",
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 16.0, "z": 0.0},
            "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
            "mass": 24.0,
            "dynamic_friction": 0.9,
            "static_friction": 0.95,
            "bounciness": 0.01,
        },
    },
    {
        "kind": "cloth",
        "case_name": "cloth_drop_sphere_building_site",
        "scene_key": "building_site",
        "primary_name": "cloth",
        "scene_label": "cloth",
        "frames": 180,
        "warmup": 8,
        "cloth": {
            "material": "cotton",
            "position": {"x": 0.0, "y": 2.1, "z": 0.0},
            "rotation": {"x": 8.0, "y": 0.0, "z": -10.0},
        },
        "support": {
            "model_name": "sphere",
            "library": "models_flex.json",
            "position": {"x": 0.18, "y": 0.0, "z": -0.08},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale_factor": {"x": 0.62, "y": 0.62, "z": 0.62},
            "mass": 38.0,
            "dynamic_friction": 0.85,
            "static_friction": 0.9,
            "bounciness": 0.02,
        },
    },
    {
        "kind": "soft_volume",
        "case_name": "soft_volume_canvas_drop",
        "scene_key": "mm_craftroom_1a",
        "primary_name": "soft_volume",
        "scene_label": "soft_volume",
        "frames": 180,
        "warmup": 8,
        "volume": {
            "cloth_material": "canvas",
            "volume_type": ClothVolumeType.sphere,
            "position": {"x": -0.05, "y": 1.28, "z": -0.05},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale_factor": {"x": 0.42, "y": 0.42, "z": 0.42},
            "pressure": 2.6,
        },
        "support": {
            "model_name": "camera_box",
            "library": "models_core.json",
            "position": {"x": 0.22, "y": 0.0, "z": 0.12},
            "rotation": {"x": 0.0, "y": -10.0, "z": 0.0},
            "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
            "mass": 18.0,
            "dynamic_friction": 0.85,
            "static_friction": 0.9,
            "bounciness": 0.02,
        },
    },
    {
        "kind": "soft_volume",
        "case_name": "soft_volume_canvas_drop_suburb",
        "scene_key": "suburb_scene_2023",
        "primary_name": "soft_volume",
        "scene_label": "soft_volume",
        "frames": 180,
        "warmup": 8,
        "camera_override": {
            "camera_position": {"x": -2.15, "y": 1.55, "z": 1.05},
            "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
            "field_of_view": 68,
        },
        "volume": {
            "cloth_material": "canvas",
            "volume_type": ClothVolumeType.sphere,
            "position": {"x": 0.0, "y": 1.32, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale_factor": {"x": 0.44, "y": 0.44, "z": 0.44},
            "pressure": 2.4,
        },
        "support": {
            "model_name": "shoebox_fused",
            "library": "models_core.json",
            "position": {"x": 0.12, "y": 0.0, "z": -0.04},
            "rotation": {"x": 0.0, "y": -12.0, "z": 0.0},
            "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
            "mass": 12.0,
            "dynamic_friction": 0.88,
            "static_friction": 0.92,
            "bounciness": 0.02,
        },
    },
    {
        "kind": "soft_volume",
        "case_name": "soft_volume_canvas_drop_building_site",
        "scene_key": "building_site",
        "primary_name": "soft_volume",
        "scene_label": "soft_volume",
        "frames": 180,
        "warmup": 8,
        "volume": {
            "cloth_material": "canvas",
            "volume_type": ClothVolumeType.sphere,
            "position": {"x": -0.08, "y": 1.34, "z": 0.06},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale_factor": {"x": 0.4, "y": 0.4, "z": 0.4},
            "pressure": 2.8,
        },
        "support": {
            "model_name": "camera_box",
            "library": "models_core.json",
            "position": {"x": 0.18, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 20.0, "z": 0.0},
            "scale_factor": {"x": 1.0, "y": 1.0, "z": 1.0},
            "mass": 18.0,
            "dynamic_friction": 0.85,
            "static_friction": 0.9,
            "bounciness": 0.02,
        },
    },
]


def sanitize_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_case_output_dirs(case_dir: Path) -> None:
    if case_dir.exists():
        subprocess.run(["rm", "-rf", str(case_dir)], check=True)
    ensure_dir(case_dir / "videos")
    ensure_dir(case_dir / "physics")
    ensure_dir(case_dir / "rgb")
    ensure_dir(case_dir / "depth")
    ensure_dir(case_dir / "visualizations")


def launch_build(log_path: Path) -> subprocess.Popen:
    ensure_dir(log_path.parent)
    log_fp = open(log_path, "ab")
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return subprocess.Popen([str(BUILD_PATH), f"-port={PORT}", f"-address={BUILD_ADDRESS}", "-force-glcore42"],
                            stdout=log_fp,
                            stderr=subprocess.STDOUT,
                            env=env)


def yup_to_zup_vec(v: Sequence[float]) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(3)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v, dtype=np.float64)
    return v / n


def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(a) for a in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([[1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                     [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                     [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]], dtype=np.float64)


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q


def convert_quat_yup_to_zup(q_xyzw: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    return q


def pca_orientation(points_zup: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_zup, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 4:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    x_axis = normalize(axes[:, 0])
    y_axis = normalize(axes[:, 2])
    z_axis = normalize(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1e-6:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = normalize(np.cross(z_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-6:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(rot) < 0.0:
        rot[:, 2] *= -1.0
    return matrix_to_quat_xyzw(rot)


def get_camera_cfg_zup(scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pos": yup_to_zup_vec([scene_cfg["camera_position"]["x"], scene_cfg["camera_position"]["y"], scene_cfg["camera_position"]["z"]]).tolist(),
        "lookat": yup_to_zup_vec([scene_cfg["look_at"]["x"], scene_cfg["look_at"]["y"], scene_cfg["look_at"]["z"]]).tolist(),
        "up": [0.0, 1.0, 0.0],
        "fov": float(scene_cfg["field_of_view"]),
        "res": [int(EXPORT_RESOLUTION[0]), int(EXPORT_RESOLUTION[1])],
    }


def camera_intrinsics_dict(camera_cfg: Dict[str, Any]) -> Dict[str, float]:
    width = int(camera_cfg["res"][0])
    height = int(camera_cfg["res"][1])
    fov_deg = float(camera_cfg["fov"])
    fy = 0.5 * float(height) / math.tan(math.radians(fov_deg) / 2.0)
    fx = fy
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": 0.5 * float(width),
        "cy": 0.5 * float(height),
        "near": 0.1,
        "far": 100.0,
    }


def rgb_to_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        return x
    if np.issubdtype(x.dtype, np.floating):
        if x.size > 0 and float(np.nanmax(x)) <= 1.0:
            x = x * 255.0
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def metric_depth_from_images(images, pass_mask: str = "_depth") -> np.ndarray:
    for i in range(images.get_num_passes()):
        if images.get_pass_mask(i) == pass_mask:
            depth = TDWUtils.get_depth_values(images.get_image(i),
                                              depth_pass=pass_mask,
                                              width=images.get_width(),
                                              height=images.get_height()).astype(np.float32)
            # TDWUtils.get_depth_values() flips rows internally, but on this build the resulting
            # metric depth ends up vertically inverted relative to the RGB frame. Flip it back so
            # saved depth products align with RGB/segmentation coordinates.
            return np.flipud(depth).copy()
    raise RuntimeError(f"Missing pass {pass_mask}")


def rgb_from_images(images) -> np.ndarray:
    for i in range(images.get_num_passes()):
        if images.get_pass_mask(i) == "_img":
            return rgb_to_uint8(np.array(TDWUtils.get_pil_image(images=images, index=i)))
    raise RuntimeError("Missing _img pass")


def depth_norm(depth_metric: np.ndarray, near: float, far: float) -> np.ndarray:
    arr = np.asarray(depth_metric, dtype=np.float32)
    denom = max(float(far) - float(near), 1e-6)
    out = np.zeros(arr.shape + (1,), dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    out[..., 0][valid] = np.clip((arr[valid] - float(near)) / denom, 0.0, 1.0)
    return out


def depth_to_uint8(depth_normalized: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth_normalized, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def depth_to_vis(depth_metric: np.ndarray, near: float, far: float) -> np.ndarray:
    vis = depth_to_uint8(depth_norm(depth_metric, near=near, far=far))
    return np.repeat(vis[..., None], 3, axis=2)


def compute_depth_display_range(depth_metric_frames: Sequence[np.ndarray], default_near: float, default_far: float) -> Tuple[float, float]:
    vals = []
    for frame in depth_metric_frames:
        arr = np.asarray(frame, dtype=np.float32)
        valid = np.isfinite(arr) & (arr > 0)
        if np.any(valid):
            vals.append(arr[valid].reshape(-1))
    if not vals:
        return float(default_near), float(default_far)
    merged = np.concatenate(vals, axis=0)
    near = float(np.percentile(merged, 2.0))
    far = float(np.percentile(merged, 98.0))
    if not np.isfinite(near) or not np.isfinite(far) or far <= near + 1e-4:
        return float(default_near), float(default_far)
    near = max(float(default_near), near)
    far = min(float(default_far), far)
    return near, max(far, near + 0.25)


def camera_axes_from_cfg(camera_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam_pos = np.asarray(camera_cfg["pos"], dtype=np.float64)
    lookat = np.asarray(camera_cfg["lookat"], dtype=np.float64)
    up_hint = np.asarray(camera_cfg.get("up", [0.0, 1.0, 0.0]), dtype=np.float64)
    forward = normalize(lookat - cam_pos)
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    return cam_pos, right, up, forward


def project_points_to_image(points_world: np.ndarray, camera_cfg: Dict[str, Any], cam_intrinsics: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    cam_pos, cam_right, cam_up, cam_forward = camera_axes_from_cfg(camera_cfg)
    rel = points_world - cam_pos[None, :]
    x_cam = np.sum(rel * cam_right[None, :], axis=1)
    y_cam = np.sum(rel * cam_up[None, :], axis=1)
    z_cam = np.sum(rel * cam_forward[None, :], axis=1)
    safe_z = np.where(z_cam > 1e-8, z_cam, np.nan)
    u = float(cam_intrinsics["fx"]) * (x_cam / safe_z) + float(cam_intrinsics["cx"])
    v = float(cam_intrinsics["cy"]) - float(cam_intrinsics["fy"]) * (y_cam / safe_z)
    return np.stack([u, v], axis=-1).astype(np.float32), z_cam.astype(np.float32)


def bbox_xyxy_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.asarray([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def compute_anchor_targets(seg_frames: np.ndarray,
                           depth_metric_frames: np.ndarray,
                           com_pos_frames: np.ndarray,
                           object_ids: np.ndarray,
                           seg_ids: np.ndarray,
                           camera_cfg: Dict[str, Any],
                           cam_intrinsics: Dict[str, float]) -> Dict[str, np.ndarray]:
    seg_frames = np.asarray(seg_frames, dtype=np.int32)
    depth_metric_frames = np.asarray(depth_metric_frames, dtype=np.float32)
    com_pos_frames = np.asarray(com_pos_frames, dtype=np.float32)
    object_ids = np.asarray(object_ids, dtype=np.int32).reshape(-1)
    seg_ids = np.asarray(seg_ids, dtype=np.int32).reshape(-1)
    num_frames, num_objects = com_pos_frames.shape[:2]
    com_uv, _ = project_points_to_image(com_pos_frames.reshape(-1, 3), camera_cfg=camera_cfg, cam_intrinsics=cam_intrinsics)
    com_uv = com_uv.reshape(num_frames, num_objects, 2).astype(np.float32)
    bbox_xyxy = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    visibility_mask = np.zeros((num_frames, num_objects), dtype=np.uint8)
    center_depth = np.zeros((num_frames, num_objects), dtype=np.float32)
    for frame_idx in range(num_frames):
        frame_seg = seg_frames[frame_idx]
        frame_depth = depth_metric_frames[frame_idx]
        for obj_idx, seg_id in enumerate(seg_ids):
            mask = frame_seg == int(seg_id)
            if not np.any(mask):
                continue
            visibility_mask[frame_idx, obj_idx] = 1
            bbox_xyxy[frame_idx, obj_idx] = bbox_xyxy_from_mask(mask)
            center_depth[frame_idx, obj_idx] = float(np.median(frame_depth[mask]))
    return {
        "object_ids": object_ids.astype(np.int32),
        "seg_ids": seg_ids.astype(np.int32),
        "com_uv": com_uv.astype(np.float32),
        "bbox_xyxy": bbox_xyxy.astype(np.float32),
        "visibility_mask": visibility_mask.astype(np.uint8),
        "center_depth": center_depth.astype(np.float32),
    }


def pairwise_contact_from_aabbs(aabbs: Sequence[Optional[np.ndarray]], clearance: float = 0.01) -> np.ndarray:
    num_objects = len(aabbs)
    graph = np.zeros((num_objects, num_objects), dtype=np.uint8)
    for i in range(num_objects):
        if aabbs[i] is None:
            continue
        amin, amax = aabbs[i][0], aabbs[i][1]
        for j in range(i + 1, num_objects):
            if aabbs[j] is None:
                continue
            bmin, bmax = aabbs[j][0], aabbs[j][1]
            overlap = np.all((amin - clearance) <= bmax) and np.all((bmin - clearance) <= amax)
            if overlap:
                graph[i, j] = 1
                graph[j, i] = 1
    return graph


def summarize_contact_windows(contact_graph_arr: np.ndarray, object_ids: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    contact_graph_arr = np.asarray(contact_graph_arr, dtype=np.uint8)
    object_ids = np.asarray(object_ids, dtype=np.int32)
    num_frames = contact_graph_arr.shape[0]
    frame_phase = np.zeros((num_frames,), dtype=np.int8)
    event_windows: List[Dict[str, Any]] = []
    collision_events: List[Dict[str, Any]] = []
    event_id = 0
    for i in range(contact_graph_arr.shape[1]):
        for j in range(i + 1, contact_graph_arr.shape[2]):
            active = contact_graph_arr[:, i, j].astype(bool)
            if not np.any(active):
                continue
            frame_phase[active] = 1
            start = None
            for frame_idx, value in enumerate(active.tolist() + [False]):
                if value and start is None:
                    start = frame_idx
                elif (not value) and start is not None:
                    end = frame_idx - 1
                    event = {
                        "event_id": int(event_id),
                        "participants": [int(object_ids[i]), int(object_ids[j])],
                        "object_indices": [int(i), int(j)],
                        "frame_idx": int(start),
                        "start_frame": int(start),
                        "peak_frame": int(start),
                        "end_frame": int(end),
                        "impulse_peak": 0.0,
                        "contact_duration": int(end - start + 1),
                    }
                    event_windows.append(event)
                    collision_events.append(dict(event))
                    event_id += 1
                    start = None
    return frame_phase, event_windows, collision_events


def summarize_environment_contact_windows(env_contact_frames: List[np.ndarray],
                                          object_ids: np.ndarray,
                                          environment_id: int = -1) -> List[Dict[str, Any]]:
    if not env_contact_frames:
        return []
    arr = np.asarray(env_contact_frames, dtype=np.uint8)
    object_ids = np.asarray(object_ids, dtype=np.int32)
    windows: List[Dict[str, Any]] = []
    event_id = 0
    for obj_idx in range(arr.shape[1]):
        active = arr[:, obj_idx].astype(bool)
        start = None
        for frame_idx, value in enumerate(active.tolist() + [False]):
            if value and start is None:
                start = frame_idx
            elif (not value) and start is not None:
                end = frame_idx - 1
                windows.append({
                    "event_id": int(event_id),
                    "participants": [int(object_ids[obj_idx]), int(environment_id)],
                    "object_indices": [int(obj_idx), -1],
                    "frame_idx": int(start),
                    "start_frame": int(start),
                    "peak_frame": int(start),
                    "end_frame": int(end),
                    "impulse_peak": 0.0,
                    "contact_duration": int(end - start + 1),
                    "environment_name": "ground",
                })
                event_id += 1
                start = None
    return windows


def get_bbox_corners(aabb: np.ndarray) -> np.ndarray:
    aabb = np.asarray(aabb, dtype=np.float64).reshape(2, 3)
    mn, mx = aabb[0], aabb[1]
    return np.asarray([[mn[0], mn[1], mn[2]],
                       [mn[0], mn[1], mx[2]],
                       [mn[0], mx[1], mn[2]],
                       [mn[0], mx[1], mx[2]],
                       [mx[0], mn[1], mn[2]],
                       [mx[0], mn[1], mx[2]],
                       [mx[0], mx[1], mn[2]],
                       [mx[0], mx[1], mx[2]]], dtype=np.float64)


def rasterize_segmentation(objects_state: List[Dict[str, Any]], camera_cfg: Dict[str, Any], cam_intrinsics: Dict[str, float]) -> np.ndarray:
    width, height = int(EXPORT_RESOLUTION[0]), int(EXPORT_RESOLUTION[1])
    seg = np.zeros((height, width), dtype=np.int32)
    draw_order = sorted(range(len(objects_state)),
                        key=lambda idx: float(objects_state[idx].get("center_depth", np.inf)),
                        reverse=True)
    for idx in draw_order:
        state = objects_state[idx]
        pts = state.get("seg_points")
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            continue
        uv, z = project_points_to_image(pts, camera_cfg=camera_cfg, cam_intrinsics=cam_intrinsics)
        valid = np.isfinite(uv).all(axis=1) & (z > 0.05)
        if not np.any(valid):
            continue
        uv_valid = uv[valid]
        x0 = max(int(np.floor(np.min(uv_valid[:, 0]))), 0)
        y0 = max(int(np.floor(np.min(uv_valid[:, 1]))), 0)
        x1 = min(int(np.ceil(np.max(uv_valid[:, 0]))), width - 1)
        y1 = min(int(np.ceil(np.max(uv_valid[:, 1]))), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        seg[y0:y1 + 1, x0:x1 + 1] = int(state["seg_id"])
    return seg


def estimate_aabb_zup_from_rigid_bound(bound) -> np.ndarray:
    pts = np.asarray([bound.front, bound.back, bound.left, bound.right, bound.top, bound.bottom, bound.center], dtype=np.float64)
    pts_y = np.asarray([yup_to_zup_vec(p) for p in pts], dtype=np.float64)
    return np.stack([pts_y.min(axis=0), pts_y.max(axis=0)], axis=0).astype(np.float32)


def build_track_states(case: Dict[str, Any], track_specs: List[Dict[str, Any]], om: ObjectManager, obi: Optional[Obi]) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for spec in track_specs:
        if spec["track_type"] == "rigid":
            object_id = int(spec["object_id"])
            if object_id not in om.bounds or object_id not in om.transforms:
                raise RuntimeError(f"Missing rigid state for object {object_id}")
            bound = om.bounds[object_id]
            transform = om.transforms[object_id]
            rigidbody = om.rigidbodies.get(object_id, None)
            aabb = estimate_aabb_zup_from_rigid_bound(bound)
            center = aabb.mean(axis=0)
            rot = convert_quat_yup_to_zup(transform.rotation)
            lin = np.zeros((3,), dtype=np.float32) if rigidbody is None else np.asarray(yup_to_zup_vec(rigidbody.velocity), dtype=np.float32)
            ang = np.zeros((3,), dtype=np.float32) if rigidbody is None else np.asarray(yup_to_zup_vec(rigidbody.angular_velocity), dtype=np.float32)
            mass = 1.0
            restitution = None
            if object_id in om.objects_static:
                mass = float(om.objects_static[object_id].mass)
                restitution = float(om.objects_static[object_id].bounciness)
            states.append({
                "object_id": object_id,
                "name": spec["name"],
                "seg_id": int(spec["seg_id"]),
                "track_type": "rigid",
                "entity_type": spec["entity_type"],
                "role": spec["role"],
                "motion_type": spec["motion_type"],
                "motion_group": spec["motion_group"],
                "source_tag": spec["source_tag"],
                "dataset_source": "TDW",
                "source_object_id": spec["name"],
                "com_pos": center.astype(np.float32),
                "orientation_quat": rot.astype(np.float32),
                "linear_vel": lin.astype(np.float32),
                "angular_vel": ang.astype(np.float32),
                "aabb": aabb.astype(np.float32),
                "seg_points": get_bbox_corners(aabb),
                "center_depth": 0.0,
                "mass": float(mass),
                "restitution": restitution,
            })
        else:
            assert obi is not None
            object_id = int(spec["object_id"])
            if object_id not in obi.actors:
                raise RuntimeError(f"Missing Obi actor state for object {object_id}")
            actor = obi.actors[object_id]
            positions = np.asarray(actor.positions, dtype=np.float64)
            velocities = np.asarray(actor.velocities, dtype=np.float64)
            if positions.size == 0:
                positions_z = np.zeros((0, 3), dtype=np.float32)
                center = np.zeros((3,), dtype=np.float32)
                aabb = np.zeros((2, 3), dtype=np.float32)
            else:
                positions_z = np.asarray([yup_to_zup_vec(p) for p in positions], dtype=np.float32)
                center = positions_z.mean(axis=0).astype(np.float32)
                aabb = np.stack([positions_z.min(axis=0), positions_z.max(axis=0)], axis=0).astype(np.float32)
            if velocities.size == 0:
                linear_vel = np.zeros((3,), dtype=np.float32)
            else:
                linear_vel = np.asarray([yup_to_zup_vec(v) for v in velocities], dtype=np.float32).mean(axis=0).astype(np.float32)
            if spec["kind"] in {"cloth", "soft_volume"} and positions.size > 0:
                orientation = pca_orientation(positions_z)
            else:
                orientation = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            states.append({
                "object_id": object_id,
                "name": spec["name"],
                "seg_id": int(spec["seg_id"]),
                "track_type": "obi",
                "entity_type": spec["entity_type"],
                "role": spec["role"],
                "motion_type": spec["motion_type"],
                "motion_group": spec["motion_group"],
                "source_tag": spec["source_tag"],
                "dataset_source": "TDW",
                "source_object_id": spec["name"],
                "com_pos": center.astype(np.float32),
                "orientation_quat": orientation.astype(np.float32),
                "linear_vel": linear_vel.astype(np.float32),
                "angular_vel": np.zeros((3,), dtype=np.float32),
                "aabb": aabb.astype(np.float32),
                "seg_points": positions_z if positions_z.size > 0 else np.zeros((0, 3), dtype=np.float32),
                "center_depth": 0.0,
                "mass": float(spec.get("mass", 1.0)),
                "restitution": None,
            })
    return states


def get_support_commands(c: Controller, support: Dict[str, Any], object_id: int) -> List[dict]:
    return c.get_add_physics_object(model_name=str(support["model_name"]),
                                    object_id=object_id,
                                    library=str(support["library"]),
                                    position=support["position"],
                                    rotation=support["rotation"],
                                    scale_factor=support["scale_factor"],
                                    kinematic=False,
                                    gravity=True,
                                    default_physics_values=False,
                                    mass=float(support.get("mass", 10.0)),
                                    dynamic_friction=float(support.get("dynamic_friction", 0.8)),
                                    static_friction=float(support.get("static_friction", 0.85)),
                                    bounciness=float(support.get("bounciness", 0.02)))


def setup_case(case: Dict[str, Any], c: Controller, obi: Optional[Obi]) -> Tuple[List[dict], List[Dict[str, Any]]]:
    commands: List[dict] = []
    track_specs: List[Dict[str, Any]] = []
    next_seg_id = 1
    if case["kind"] == "rigid":
        object_id = c.get_unique_id()
        obj = case["object"]
        commands.extend(c.get_add_physics_object(model_name=str(obj["model_name"]),
                                                 object_id=object_id,
                                                 position=obj["position"],
                                                 rotation=obj["rotation"]))
        commands.extend([{"$type": "set_velocity", "id": object_id, "velocity": obj["velocity"]},
                         {"$type": "set_angular_velocity", "id": object_id, "angular_velocity": obj["angular_velocity"]}])
        track_specs.append({
            "object_id": object_id,
            "name": str(obj["model_name"]),
            "seg_id": next_seg_id,
            "track_type": "rigid",
            "entity_type": "rigid_body",
            "role": "dynamic",
            "motion_type": "rigid",
            "motion_group": "rigid",
            "source_tag": "tdw_builtin",
            "kind": case["kind"],
        })
    elif case["kind"] == "cloth":
        assert obi is not None
        cloth_id = c.get_unique_id()
        support_id = c.get_unique_id()
        obi.set_solver(substeps=4)
        obi.create_cloth_sheet(cloth_material=str(case["cloth"]["material"]),
                               object_id=cloth_id,
                               position=case["cloth"]["position"],
                               rotation=case["cloth"]["rotation"])
        commands.extend(c.get_add_physics_object(model_name=str(case["support"]["model_name"]),
                                                 object_id=support_id,
                                                 library=str(case["support"]["library"]),
                                                 position=case["support"]["position"],
                                                 rotation=case["support"]["rotation"],
                                                 scale_factor=case["support"]["scale_factor"],
                                                 kinematic=False,
                                                 gravity=True,
                                                 default_physics_values=False,
                                                 mass=float(case["support"].get("mass", 40.0)),
                                                 dynamic_friction=float(case["support"].get("dynamic_friction", 0.85)),
                                                 static_friction=float(case["support"].get("static_friction", 0.9)),
                                                 bounciness=float(case["support"].get("bounciness", 0.02))))
        track_specs.extend([
            {"object_id": cloth_id, "name": "cloth", "seg_id": 1, "track_type": "obi", "entity_type": "cloth", "role": "dynamic", "motion_type": "cloth", "motion_group": "cloth", "source_tag": "tdw_obi", "kind": case["kind"], "mass": 1.0},
            {"object_id": support_id, "name": "sphere_support", "seg_id": 2, "track_type": "rigid", "entity_type": "support", "role": "dynamic", "motion_type": "dynamic_support", "motion_group": "support", "source_tag": "tdw_builtin", "kind": case["kind"]},
        ])
    elif case["kind"] == "soft_volume":
        assert obi is not None
        volume_id = c.get_unique_id()
        support_id = c.get_unique_id()
        obi.set_solver(substeps=4)
        obi.create_cloth_volume(cloth_material=str(case["volume"]["cloth_material"]),
                                object_id=volume_id,
                                volume_type=case["volume"]["volume_type"],
                                position=case["volume"]["position"],
                                rotation=case["volume"]["rotation"],
                                scale_factor=case["volume"]["scale_factor"],
                                pressure=float(case["volume"]["pressure"]))
        commands.extend(get_support_commands(c, case["support"], support_id))
        track_specs.extend([
            {"object_id": volume_id, "name": "soft_volume", "seg_id": 1, "track_type": "obi", "entity_type": "soft_volume", "role": "dynamic", "motion_type": "soft_volume", "motion_group": "soft_volume", "source_tag": "tdw_obi", "kind": case["kind"], "mass": 1.0},
            {"object_id": support_id, "name": "box_support", "seg_id": 2, "track_type": "rigid", "entity_type": "support", "role": "dynamic", "motion_type": "dynamic_support", "motion_group": "support", "source_tag": "tdw_builtin", "kind": case["kind"]},
        ])
    elif case["kind"] in {"granular", "liquid"}:
        assert obi is not None
        fluid_id = c.get_unique_id()
        support_id = c.get_unique_id()
        obi.set_solver(substeps=int(case["substeps"]))
        fluid_cfg = case["fluid"]
        obi.create_fluid(fluid=fluid_cfg["name"],
                         shape=fluid_cfg["shape"],
                         object_id=fluid_id,
                         position=fluid_cfg["position"],
                         rotation=fluid_cfg["rotation"],
                         speed=float(fluid_cfg["speed"]),
                         lifespan=float(fluid_cfg["lifespan"]))
        commands.extend(get_support_commands(c, case["support"], support_id))
        track_specs.extend([
            {"object_id": fluid_id, "name": case["primary_name"], "seg_id": 1, "track_type": "obi", "entity_type": case["kind"], "role": "dynamic", "motion_type": case["kind"], "motion_group": case["kind"], "source_tag": "tdw_obi", "kind": case["kind"], "mass": 1.0},
            {"object_id": support_id, "name": "receptacle", "seg_id": 2, "track_type": "rigid", "entity_type": "support", "role": "dynamic", "motion_type": "dynamic_support", "motion_group": "support", "source_tag": "tdw_builtin", "kind": case["kind"]},
        ])
    else:
        raise ValueError(f"Unsupported kind: {case['kind']}")
    return commands, track_specs


def record_case(case: Dict[str, Any]) -> Path:
    sanitize_proxy_env()
    scene_cfg = SCENES[case["scene_key"]]
    if "camera_override" in case:
        scene_cfg = {**scene_cfg, **case["camera_override"]}
    scene_composition = str(scene_cfg["name"])
    object_count_bucket = f"count_{1 if case['kind'] == 'rigid' else 2}"
    sample_name = f"{case['primary_name']}__{case['case_name']}"
    case_dir = OUTPUT_ROOT / "train" / "rigid" / scene_composition / object_count_bucket / sample_name
    existing_meta = case_dir / "meta.json"
    if existing_meta.exists():
        print(f"SKIP {existing_meta}", flush=True)
        return existing_meta

    build_log = OUTPUT_ROOT / "logs" / f"{sample_name}.build.log"
    print(f"[{sample_name}] launch build on {DISPLAY} port={PORT} address={BUILD_ADDRESS}", flush=True)
    build_holder: Dict[str, subprocess.Popen] = {"proc": launch_build(build_log)}
    time.sleep(BUILD_BOOT_WAIT)
    print(f"[{sample_name}] connect controller", flush=True)
    c = Controller(launch_build=False, check_version=False, port=PORT)
    try:
        prepare_case_output_dirs(case_dir)
        print(f"[{sample_name}] output dir ready {case_dir}", flush=True)
        scene_camera = ThirdPersonCamera(avatar_id="a",
                                         position=scene_cfg["camera_position"],
                                         look_at=scene_cfg["look_at"],
                                         field_of_view=int(scene_cfg["field_of_view"]))
        capture = ImageCapture(path=case_dir / "_tmp", avatar_ids=["a"], pass_masks=["_img", "_depth"])
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img", "_depth"], save=False)
        lighting = InteriorSceneLighting(hdri_skybox=scene_cfg["skybox"],
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        om = ObjectManager(transforms=True, rigidbodies=True, bounds=True)
        obi = None
        add_ons: List[Any] = [lighting, scene_camera, capture, om]
        if case["kind"] != "rigid":
            obi = Obi(output_data=True,
                      floor_material=CollisionMaterial(dynamic_friction=0.55,
                                                       static_friction=0.6,
                                                       stickiness=0.0,
                                                       stick_distance=0.0))
            add_ons.append(obi)
        c.add_ons.extend(add_ons)

        commands = [{"$type": "set_screen_size", "width": int(EXPORT_RESOLUTION[0]), "height": int(EXPORT_RESOLUTION[1])},
                    {"$type": "set_physics_solver_iterations", "iterations": 16},
                    Controller.get_add_scene(scene_name=scene_cfg["name"]),
                    Controller.get_add_hdri_skybox(skybox_name=scene_cfg["skybox"])]

        setup_commands, track_specs = setup_case(case, c, obi)
        commands.extend(setup_commands)
        print(f"[{sample_name}] initial communicate scene={scene_cfg['name']} kind={case['kind']}", flush=True)
        c.communicate(commands)
        print(f"[{sample_name}] warmup frames={int(case.get('warmup', 0))}", flush=True)
        for _ in range(int(case.get("warmup", 0))):
            c.communicate([])

        camera_cfg = get_camera_cfg_zup(scene_cfg)
        cam_intrinsics = camera_intrinsics_dict(camera_cfg)

        rgb_frames: List[np.ndarray] = []
        depth_metric_frames: List[np.ndarray] = []
        seg_frames: List[np.ndarray] = []
        com_pos_frames: List[np.ndarray] = []
        orientation_frames: List[np.ndarray] = []
        linear_vel_frames: List[np.ndarray] = []
        angular_vel_frames: List[np.ndarray] = []
        kinetic_frames: List[float] = []
        kinetic_trans_frames: List[float] = []
        kinetic_rot_frames: List[float] = []
        potential_frames: List[float] = []
        total_frames: List[float] = []
        aabb_frames: List[List[Optional[np.ndarray]]] = []
        env_contact_frames: List[np.ndarray] = []

        total_frames_to_capture = int(case["frames"])
        print(f"[{sample_name}] capture frames={total_frames_to_capture}", flush=True)
        for frame_idx in range(total_frames_to_capture):
            c.communicate([])
            images = capture.images.get("a", None)
            if images is None:
                raise RuntimeError("ImageCapture did not receive images for avatar a")
            rgb = rgb_from_images(images)
            depth_metric = metric_depth_from_images(images, pass_mask="_depth")
            objects_state = build_track_states(case=case, track_specs=track_specs, om=om, obi=obi)
            for state in objects_state:
                uv, z_cam = project_points_to_image(np.asarray([state["com_pos"]], dtype=np.float64), camera_cfg=camera_cfg, cam_intrinsics=cam_intrinsics)
                state["center_depth"] = float(z_cam[0]) if np.isfinite(z_cam[0]) else float("inf")
            seg = rasterize_segmentation(objects_state, camera_cfg=camera_cfg, cam_intrinsics=cam_intrinsics)
            rgb_frames.append(rgb.astype(np.uint8))
            depth_metric_frames.append(depth_metric.astype(np.float32))
            seg_frames.append(seg.astype(np.int32))
            com_pos = np.stack([state["com_pos"] for state in objects_state], axis=0).astype(np.float32)
            orientation = np.stack([state["orientation_quat"] for state in objects_state], axis=0).astype(np.float32)
            linear_vel = np.stack([state["linear_vel"] for state in objects_state], axis=0).astype(np.float32)
            angular_vel = np.stack([state["angular_vel"] for state in objects_state], axis=0).astype(np.float32)
            com_pos_frames.append(com_pos)
            orientation_frames.append(orientation)
            linear_vel_frames.append(linear_vel)
            angular_vel_frames.append(angular_vel)
            masses = np.asarray([float(state["mass"]) for state in objects_state], dtype=np.float32)
            kinetic_trans = float(np.sum(0.5 * masses * np.sum(np.square(linear_vel), axis=1)))
            kinetic_rot = float(np.sum(0.5 * masses * np.sum(np.square(angular_vel), axis=1) * 0.05))
            potential = float(np.sum(masses * 9.81 * np.maximum(com_pos[:, 1], 0.0)))
            total = kinetic_trans + kinetic_rot + potential
            kinetic_frames.append(kinetic_trans + kinetic_rot)
            kinetic_trans_frames.append(kinetic_trans)
            kinetic_rot_frames.append(kinetic_rot)
            potential_frames.append(potential)
            total_frames.append(total)
            aabbs = [state["aabb"] if np.asarray(state["aabb"]).size > 0 else None for state in objects_state]
            aabb_frames.append(aabbs)
            env_contact_frames.append(np.asarray([1 if (aabb is not None and float(aabb[0][1]) <= 0.02) else 0 for aabb in aabbs], dtype=np.uint8))
            if (frame_idx + 1) % 30 == 0 or frame_idx + 1 == total_frames_to_capture:
                print(f"[{sample_name}] captured {frame_idx + 1}/{total_frames_to_capture}", flush=True)

        object_ids = np.asarray([int(s["object_id"]) for s in objects_state], dtype=np.int32)
        seg_ids = np.asarray([int(s["seg_id"]) for s in objects_state], dtype=np.int32)
        object_names = [str(s["name"]) for s in objects_state]
        object_sources = [str(s["source_tag"]) for s in objects_state]
        com_pos_arr = np.stack(com_pos_frames, axis=0).astype(np.float32)
        orientation_arr = np.stack(orientation_frames, axis=0).astype(np.float32)
        linear_vel_arr = np.stack(linear_vel_frames, axis=0).astype(np.float32)
        angular_vel_arr = np.stack(angular_vel_frames, axis=0).astype(np.float32)
        depth_metric_arr = np.stack(depth_metric_frames, axis=0).astype(np.float32)
        seg_arr = np.stack(seg_frames, axis=0).astype(np.int32)
        contact_graph_arr = np.stack([pairwise_contact_from_aabbs(a) for a in aabb_frames], axis=0).astype(np.uint8)
        frame_phase_arr, event_windows, collision_events = summarize_contact_windows(contact_graph_arr, object_ids)
        env_windows = summarize_environment_contact_windows(env_contact_frames, object_ids, environment_id=-1)
        collision_events.extend(env_windows)
        anchor_targets = compute_anchor_targets(seg_frames=seg_arr,
                                                depth_metric_frames=depth_metric_arr,
                                                com_pos_frames=com_pos_arr,
                                                object_ids=object_ids,
                                                seg_ids=seg_ids,
                                                camera_cfg=camera_cfg,
                                                cam_intrinsics=cam_intrinsics)

        scene_input = {
            "object_id": str(case["primary_name"]),
            "sample_name": sample_name,
            "case_name": case["case_name"],
            "case_id": int(CASES.index(case)),
            "case_variant_index": 0,
            "scene_label": str(case["scene_label"]),
            "simulator_mode": str(case["kind"]),
            "simulator_type": str(case["kind"]),
            "scene_composition": scene_composition,
            "interaction_pattern": str(case["case_name"]),
            "object_count_bucket": object_count_bucket,
            "camera": camera_cfg,
            "camera_tag": None,
            "entry_linear_velocity": linear_vel_arr[0, 0].tolist(),
            "entry_angular_velocity": angular_vel_arr[0, 0].tolist(),
            "use_entry_motion": True,
            "object_fixed": False,
            "gravity": GRAVITY_YUP.tolist(),
            "striker_speed_mps": 0.0,
            "counterfactual": None,
            "rigid_restitution_override": None,
        }
        print(f"[{sample_name}] write outputs", flush=True)
        (case_dir / "scene_input.json").write_text(json.dumps(scene_input, ensure_ascii=False, indent=2), encoding="utf-8")

        for frame_idx, rgb in enumerate(rgb_frames):
            imageio.imwrite(case_dir / "rgb" / f"frame_{frame_idx:03d}.png", rgb)
        display_depth_near, display_depth_far = compute_depth_display_range(depth_metric_frames=depth_metric_arr,
                                                                            default_near=cam_intrinsics["near"],
                                                                            default_far=cam_intrinsics["far"])
        for frame_idx, depth_metric in enumerate(depth_metric_arr):
            imageio.imwrite(case_dir / "depth" / f"frame_{frame_idx:03d}.png", depth_to_uint8(depth_norm(depth_metric, near=display_depth_near, far=display_depth_far)))

        np.save(case_dir / "physics" / "depth_metric.npy", depth_metric_arr)
        np.save(case_dir / "physics" / "seg.npy", seg_arr)
        np.save(case_dir / "physics" / "contact_graph.npy", contact_graph_arr)
        np.save(case_dir / "physics" / "contact_impulse.npy", np.zeros_like(contact_graph_arr, dtype=np.float32))
        np.save(case_dir / "physics" / "frame_phase.npy", frame_phase_arr.astype(np.int8))
        np.savez_compressed(case_dir / "physics" / "anchor_targets.npz", **anchor_targets)
        np.savez_compressed(case_dir / "physics" / "rigid_kinematics.npz",
                            object_ids=object_ids.astype(np.int32),
                            seg_ids=seg_ids.astype(np.int32),
                            com_pos=com_pos_arr,
                            orientation_quat=orientation_arr,
                            linear_vel=linear_vel_arr,
                            angular_vel=angular_vel_arr,
                            com_uv=anchor_targets["com_uv"],
                            bbox_xyxy=anchor_targets["bbox_xyxy"],
                            visibility_mask=anchor_targets["visibility_mask"],
                            kinetic_energy=np.asarray(kinetic_frames, dtype=np.float32),
                            potential_energy=np.asarray(potential_frames, dtype=np.float32),
                            total_energy=np.asarray(total_frames, dtype=np.float32))
        np.savez_compressed(case_dir / "physics" / "energy.npz",
                            kinetic_trans=np.asarray(kinetic_trans_frames, dtype=np.float32),
                            kinetic_rot=np.asarray(kinetic_rot_frames, dtype=np.float32),
                            potential_gravity=np.asarray(potential_frames, dtype=np.float32),
                            mechanical_total=np.asarray(total_frames, dtype=np.float32))
        imageio.mimwrite(case_dir / "videos" / "rgb.mp4", rgb_frames, fps=FPS, quality=8)
        imageio.mimwrite(case_dir / "videos" / "depth.mp4", [depth_to_uint8(depth_norm(d, near=display_depth_near, far=display_depth_far)) for d in depth_metric_arr], fps=FPS, quality=8)
        imageio.mimwrite(case_dir / "visualizations" / "depth_vis.mp4", [depth_to_vis(d, near=display_depth_near, far=display_depth_far) for d in depth_metric_arr], fps=FPS, quality=8)
        (case_dir / "physics" / "collision_events.json").write_text(json.dumps(collision_events, ensure_ascii=False, indent=2), encoding="utf-8")
        (case_dir / "physics" / "event_windows.json").write_text(json.dumps(event_windows + env_windows, ensure_ascii=False, indent=2), encoding="utf-8")

        properties_payload = {
            "object_ids": object_ids.astype(np.int32).tolist(),
            "sampled_restitution": [state.get("restitution", None) for state in objects_state],
            "effective_restitution_used": [state.get("restitution", None) for state in objects_state],
            "counterfactual": None,
        }
        (case_dir / "physics" / "properties.json").write_text(json.dumps(properties_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata_payload = {
            "scene_id": sample_name,
            "object_id": str(case["primary_name"]),
            "case_id": int(CASES.index(case)),
            "case_variant_index": 0,
            "case_name": str(case["case_name"]),
            "seed": 0,
            "split": "train",
            "family": "tdw_single_case",
            "simulator_type": str(case["kind"]),
            "scene_composition": scene_composition,
            "interaction_pattern": str(case["case_name"]),
            "object_count_bucket": object_count_bucket,
            "num_objects": int(object_ids.shape[0]),
            "frames": int(com_pos_arr.shape[0]),
            "resolution": [int(EXPORT_RESOLUTION[0]), int(EXPORT_RESOLUTION[1])],
            "motion_category": str(case["scene_label"]),
            "sample_role": "factual",
            "counterfactual": None,
            "convention": {
                "length_unit": "meter",
                "mass_unit": "kg",
                "time_unit": "second",
                "coordinate_system": "right-handed",
                "gravity_axis": "y_negative",
            },
            "simulation": {
                "engine": "TDW",
                "engine_version": "v1.13.0",
                "dt": 1.0 / 60.0,
                "substeps": int(case.get("substeps", 1)),
                "steps_per_frame": 1,
                "frame_dt": 1.0 / 60.0,
                "video_fps": float(FPS),
                "gravity": GRAVITY_YUP.tolist(),
            },
            "camera": camera_cfg,
            "camera_tag": None,
            "camera_intrinsics": cam_intrinsics,
            "objects": [{
                "object_id": int(object_ids[idx]),
                "seg_id": int(seg_ids[idx]),
                "entity_type": str(objects_state[idx]["entity_type"]),
                "role": str(objects_state[idx]["role"]),
                "object_motion_type": str(objects_state[idx]["motion_type"]),
                "object_motion_group": str(objects_state[idx]["motion_group"]),
                "motion_type": str(objects_state[idx]["motion_type"]),
                "motion_group": str(objects_state[idx]["motion_group"]),
                "source_tag": str(object_sources[idx]),
                "dataset_source": str(objects_state[idx]["dataset_source"]),
                "source_object_id": str(objects_state[idx]["source_object_id"]),
            } for idx in range(object_ids.shape[0])],
            "environment_entities": [{
                "name": "ground",
                "special_id": -1,
                "entity_type": "container",
            }],
            "outputs": {
                "metadata": "meta.json",
                "scene_input": "scene_input.json",
                "rgb_video": "videos/rgb.mp4",
                "depth_video": "videos/depth.mp4",
                "depth_metric": "physics/depth_metric.npy",
                "segmentation": "physics/seg.npy",
                "anchor_targets": "physics/anchor_targets.npz",
                "rigid_kinematics": "physics/rigid_kinematics.npz",
                "energy": "physics/energy.npz",
                "properties": "physics/properties.json",
                "contact_graph": "physics/contact_graph.npy",
                "contact_impulse": "physics/contact_impulse.npy",
                "frame_phase": "physics/frame_phase.npy",
                "event_windows": "physics/event_windows.json",
                "depth_visualization_video": "visualizations/depth_vis.mp4",
            },
            "has_depth_metric": True,
            "has_seg": True,
            "has_contact_graph": True,
            "status": "ok",
        }
        (case_dir / "meta.json").write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"GENERATED {case_dir / 'videos' / 'rgb.mp4'}", flush=True)
        return case_dir / "meta.json"
    finally:
        try:
            c.communicate({"$type": "terminate"})
        except Exception:
            pass
        try:
            build_holder["proc"].wait(timeout=10)
        except Exception:
            if "proc" in build_holder:
                build_holder["proc"].kill()


def build_html() -> None:
    cards: List[str] = []
    meta_paths = sorted(OUTPUT_ROOT.glob("train/rigid/*/*/*/meta.json"))
    for meta_path in meta_paths:
        case_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rel = str(case_dir.joinpath("videos", "rgb.mp4").relative_to(OUTPUT_ROOT.parent))
        cards.append(
            f'''<article class="card">
  <video controls preload="metadata" src="{rel}"></video>
  <div class="meta">
    <span class="pill">Genesis Export</span>
    <span class="pill scene-pill">{meta["scene_composition"]}</span>
    <h3>{meta["case_name"]}</h3>
    <p>kind={meta["simulator_type"]} | objects={meta["num_objects"]} | frames={meta["frames"]}</p>
    <code>{case_dir}</code>
  </div>
</article>'''
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TDW Genesis Format Exports</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: rgba(255,255,255,0.94);
      --ink: #171714;
      --muted: #6a665e;
      --accent: #466b5d;
      --border: rgba(54, 42, 24, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(209, 186, 156, 0.30), transparent 28%),
        radial-gradient(circle at right 18%, rgba(161, 187, 176, 0.24), transparent 24%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1680px; margin: 0 auto; padding: 28px 18px 40px; }}
    .hero, .grid, .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(44, 35, 19, 0.10);
    }}
    .hero {{ padding: 26px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; padding: 18px; background: transparent; border: 0; box-shadow: none; }}
    .card {{ overflow: hidden; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 5vw, 52px); line-height: 0.96; }}
    h3 {{ margin: 0 0 10px; font-size: 24px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.65; }}
    video {{ width: 100%; display: block; background: #000; aspect-ratio: 16 / 9; }}
    .meta {{ padding: 18px 20px 22px; }}
    .pill {{
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 10px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(70, 107, 93, 0.12);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    code {{
      display: block;
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(23, 23, 20, 0.05);
      color: #574f45;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>TDW Genesis Format Exports</h1>
      <p>每个物理种类先跑一个代表 case，并按 Genesis 风格目录与文件名导出。这里展示每个样本的 RGB 预览视频。</p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")
    print(HTML_PATH, flush=True)


def main() -> None:
    sanitize_proxy_env()
    ensure_dir(OUTPUT_ROOT)
    case_filter = {name.strip() for name in os.environ.get("TDW_CASE_FILTER", "").split(",") if name.strip()}
    cases = [case for case in CASES if not case_filter or case["case_name"] in case_filter]
    for case in cases:
        print(f"Running {case['case_name']}", flush=True)
        record_case(case)
        print(f"Completed {case['case_name']}", flush=True)
    build_html()


if __name__ == "__main__":
    main()
