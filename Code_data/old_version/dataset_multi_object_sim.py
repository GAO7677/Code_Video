#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_multi_object_sim.py

多物体（1-5个）、多运动、多仿真类型的 Genesis 数据集生成脚本。

支持的场景类型
──────────────
1. pure_rigid        — 纯刚体仿真
   子模式：
     top_drop         : 全部/大部分从上方自由下坠
     side_throw       : 全部从左侧或右侧抛入
     partial_static   : 部分物体静止在地面，其余做各种动态运动
     multi_motion_mix : 每个物体独立随机选运动模式（下坠/侧抛/对角/滑入）

2. pure_mpm          — 纯 MPM 弹性/弹塑性体仿真
   运动模式同 pure_rigid

3. mixed_rigid_mpm   — 刚体 + MPM 混合

物体数量：每场景 1-5 个
物体来源：随机几何体（box/sphere/cylinder/capsule） + 可选 PhysXNet URDF（刚体）

用法示例
────────
  # 仅几何体，快速测试
  CUDA_VISIBLE_DEVICES=0 python dataset_multi_object_sim.py \\
      --dataset_root /data/.../multi_object_sim \\
      --samples_per_scenario 3 --target_seconds 3.0 --no_physx

  # 启用 PhysXNet 资产
  CUDA_VISIBLE_DEVICES=0 python dataset_multi_object_sim.py \\
      --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \\
      --dataset_root /data/.../multi_object_sim \\
      --samples_per_scenario 3 --target_seconds 3.0 --max_physx_objects 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

# ── Optional PhysXNet helpers ────────────────────────────────────────────────
try:
    from physxnet_articulation_demo import (
        _configure_genesis_rigid_entity_from_metadata,
        _default_entity_rigid_material,
        _make_genesis_rigid_material,
        prepare_physxnet_object,
    )
    _HAS_PHYSXNET = True
except ImportError:
    _HAS_PHYSXNET = False
    print("[WARN] physxnet_articulation_demo not importable; PhysXNet assets disabled.")


# ===========================================================================
# ── Global constants
# ===========================================================================

IMG_W, IMG_H     = 960, 720
PREVIEW_FPS      = 30
MAX_OBJECT_PC    = 2048
OBJECT_PC_STRIDE = 4
CAMERA_PC_STRIDE = 2
STOP_ON_ERROR    = False

OBJECT_COUNT_CHOICES = [1, 2, 3, 4, 5]
TARGET_LONGEST_RANGE = (0.15, 0.40)   # 物体最长边目标尺寸（米）

# ── Scene-family weights
SCENE_FAMILY_WEIGHTS: Dict[str, float] = {
    "pure_rigid":      0.50,
    "pure_mpm":        0.30,
    "mixed_rigid_mpm": 0.20,
}

# ── Rigid submode weights
RIGID_SUBMODE_WEIGHTS: Dict[str, float] = {
    "top_drop":         0.28,
    "side_throw":       0.22,
    "partial_static":   0.25,
    "multi_motion_mix": 0.25,
}

# ── Per-object motion weights  (used by multi_motion_mix / mpm / mixed)
SINGLE_MOTION_WEIGHTS: Dict[str, float] = {
    "top_drop":         0.28,
    "top_toss":         0.18,
    "side_throw_left":  0.12,
    "side_throw_right": 0.12,
    "diagonal_left":    0.08,
    "diagonal_right":   0.08,
    "front_slide":      0.08,
    "static_rest":      0.06,
}

# ── Materials
RIGID_MATERIAL_PRESETS: Dict[str, Dict[str, float]] = {
    "wood":    {"rho": 700.0,  "friction": 0.60, "restitution": 0.12},
    "plastic": {"rho": 1050.0, "friction": 0.42, "restitution": 0.18},
    "rubber":  {"rho": 1150.0, "friction": 1.05, "restitution": 0.65},
    "metal":   {"rho": 2700.0, "friction": 0.24, "restitution": 0.10},
    "foam":    {"rho":   90.0, "friction": 0.78, "restitution": 0.05},
    "glass":   {"rho": 2500.0, "friction": 0.16, "restitution": 0.08},
}
RIGID_MATERIAL_WEIGHTS: Dict[str, float] = {
    "wood": 0.25, "plastic": 0.25, "rubber": 0.20,
    "metal": 0.15, "foam": 0.10,   "glass": 0.05,
}

MPM_MATERIAL_PRESETS: Dict[str, Dict[str, float]] = {
    "soft_foam":   {"E": 3e4,   "nu": 0.25, "rho":  600.0},
    "gel":         {"E": 1.2e5, "nu": 0.30, "rho": 1050.0},
    "rubber_soft": {"E": 5e5,   "nu": 0.38, "rho": 1150.0},
    "clay":        {"E": 2e5,   "nu": 0.35, "rho": 1400.0},
    "elastic_obj": {"E": 8e4,   "nu": 0.28, "rho":  900.0},
}
MPM_MATERIAL_WEIGHTS: Dict[str, float] = {
    "soft_foam": 0.25, "gel": 0.20, "rubber_soft": 0.25,
    "clay": 0.15, "elastic_obj": 0.15,
}

# ── Shape weights
RIGID_SHAPE_WEIGHTS = {"box": 0.34, "sphere": 0.24, "cylinder": 0.24, "capsule": 0.18}
MPM_SHAPE_WEIGHTS   = {"box": 0.55, "sphere": 0.45}

# ── Container default
CONTAINER_DEFAULT: Dict[str, Any] = {
    "half_x": 1.40, "half_y": 1.40,
    "wall_thickness": 0.06, "wall_height": 2.00,
    "floor_thickness": 0.06,
    "center": [0.0, 0.0, 0.0],
    "front_keep_out": 0.40, "back_keep_out": 0.10, "side_keep_out": 0.08,
}

# ── Motion kinematics
TOP_DROP_Z_RANGE       = (1.00, 1.60)
TOP_TOSS_Z_RANGE       = (0.90, 1.45)
FRONT_SLIDE_Z_RANGE    = (0.12, 0.32)
SIDE_THROW_Z_RANGE     = (0.70, 1.25)
DIAGONAL_ENTRY_Z_RANGE = (0.90, 1.55)

TOP_DROP_VXY        = 0.10
TOP_TOSS_VXY_RANGE  = (-0.30,  0.30)
TOP_TOSS_VZ_RANGE   = (-0.85, -0.05)
FRONT_VY_RANGE      = (1.00, 1.80)
FRONT_VX_RANGE      = (-0.20, 0.20)
FRONT_VZ_RANGE      = (-0.05, 0.20)
SIDE_VX_RANGE       = (0.90, 1.45)
SIDE_VY_RANGE       = (0.05, 0.25)
SIDE_VZ_RANGE       = (0.40, 0.90)
DIAG_VX_RANGE       = (0.90, 1.60)
DIAG_VY_RANGE       = (0.95, 1.80)
DIAG_VZ_RANGE       = (-0.10, 0.40)

ANGVEL_MAX  = 3.5   # rad/s maximum for rigid bodies
REST_MARGIN = 0.02  # metres above floor for static objects


# ===========================================================================
# ── Utility helpers
# ===========================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def weighted_choice(d: Dict[str, float]) -> str:
    keys  = list(d.keys())
    probs = np.asarray(list(d.values()), dtype=np.float64)
    probs /= probs.sum()
    return str(np.random.choice(keys, p=probs))


def to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:                   return None
    if isinstance(x, np.ndarray):   return x
    if hasattr(x, "detach"):        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):           return x.cpu().numpy()
    return np.asarray(x)


def safe_subsample(xyz: np.ndarray, max_pts: int = 2048) -> np.ndarray:
    xyz = np.asarray(xyz)
    if len(xyz) <= max_pts:
        return xyz.astype(np.float32, copy=False)
    return xyz[np.random.choice(len(xyz), max_pts, replace=False)].astype(np.float32)


def sample_color(alpha: float = 1.0) -> List[float]:
    return [float(np.random.uniform(0.10, 0.95)) for _ in range(3)] + [float(alpha)]


def random_angvel(scale: float) -> List[float]:
    if scale <= 0:
        return [0.0, 0.0, 0.0]
    axis = np.random.uniform(-1.0, 1.0, 3)
    n    = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    return [float(v) for v in axis * float(np.random.uniform(0.2, scale))]


def safe_scene_destroy(scene: Any) -> None:
    if scene is not None:
        try:
            scene.destroy()
        except Exception:
            pass


def save_depth_vis(depth: np.ndarray, out_path: Path) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    vis   = np.zeros(depth.shape + (3,), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        d  = depth[valid].flatten()
        lo, hi = float(d.min()), float(d.max())
        hi = max(hi, lo + 1e-5)
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        gray = (255.0 * (1.0 - norm)).astype(np.uint8)
        vis[..., 0] = gray
        vis[..., 1] = gray
        vis[..., 2] = gray
    imageio.imwrite(out_path, vis)


def prepare_output_dirs(out_dir: Path) -> None:
    for sub in ["rgb", "depth", "depth_vis", "segmentation",
                "normal", "pointcloud", "object_pointcloud",
                "trajectories", "camera", "video"]:
        ensure_dir(out_dir / sub)


def compute_preview_stride(dt: float, num_steps: int, physics_s: float) -> int:
    target_frames = max(1.0, physics_s * PREVIEW_FPS)
    return max(1, int(round(num_steps / target_frames)))


def try_set(obj: Any, methods: List[str], value: Any) -> bool:
    for m in methods:
        if hasattr(obj, m):
            try:
                getattr(obj, m)(value)
                return True
            except Exception:
                pass
    return False


def make_json_safe(x: Any) -> Any:
    if isinstance(x, dict):            return {str(k): make_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):   return [make_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):      return x.tolist()
    if isinstance(x, (np.floating,)):  return float(x)
    if isinstance(x, (np.integer,)):   return int(x)
    if isinstance(x, (np.bool_,)):     return bool(x)
    return x


# ===========================================================================
# ── Spawn / motion sampling
# ===========================================================================

def _container_xy_bounds(C: Dict[str, Any], obj_hx: float, obj_hy: float
                         ) -> Tuple[float, float, float, float]:
    """Return (x_lo, x_hi, y_lo, y_hi) safe spawn band inside container."""
    x_lo = -C["half_x"] + C["wall_thickness"] + C["side_keep_out"]  + obj_hx
    x_hi = +C["half_x"] - C["wall_thickness"] - C["side_keep_out"]  - obj_hx
    y_lo = -C["half_y"] + C["wall_thickness"] + C["front_keep_out"] + obj_hy
    y_hi = +C["half_y"] - C["wall_thickness"] - C["back_keep_out"]  - obj_hy
    cx, cy, _ = C["center"]
    if x_lo >= x_hi: x_lo, x_hi = cx - 0.05, cx + 0.05
    if y_lo >= y_hi: y_lo, y_hi = cy + 0.05, cy + 0.15
    return float(x_lo), float(x_hi), float(y_lo), float(y_hi)


def sample_spawn_xy(C: Dict[str, Any], obj_hx: float, obj_hy: float,
                    bias_back: bool = False) -> Tuple[float, float]:
    x_lo, x_hi, y_lo, y_hi = _container_xy_bounds(C, obj_hx, obj_hy)
    if bias_back:
        mid_y = (y_lo + y_hi) / 2.0
        y_lo  = max(y_lo, mid_y)
    return float(np.random.uniform(x_lo, x_hi)), float(np.random.uniform(y_lo, y_hi))


def _floor_rest_z(C: Dict[str, Any], obj_hz: float, 