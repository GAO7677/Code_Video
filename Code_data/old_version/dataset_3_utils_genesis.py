#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Genesis dataset generation.
Extracted from dataset_3_mpm_genesis.py and dataset_3_rigid_genesis.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

# Constants
IMG_W, IMG_H = 960, 720
TARGET_LONGEST_SIZE_RANGE = (0.18, 0.42)
STATIC_REST_PROB = 0.38
PREVIEW_FPS = 30

TOP_DROP_Z_RANGE = (1.00, 1.55)
TOP_TOSS_Z_RANGE = (0.95, 1.45)
FRONT_SLIDE_Z_RANGE = (0.16, 0.34)
SIDE_THROW_Z_RANGE = (0.75, 1.25)
DIAGONAL_ENTRY_Z_RANGE = (0.95, 1.60)

STRIKE_SPEED_RANGE = (0.80, 1.35)
TOP_DROP_ANGVEL = 1.6
TOP_TOSS_ANGVEL = 2.2
FRONT_SLIDE_ANGVEL = 1.8
DIAGONAL_THROW_ANGVEL = 2.6
SIDE_THROW_ANGVEL = 2.2

FRONT_SLIDE_VY_RANGE = (1.00, 1.75)
FRONT_SLIDE_VX_RANGE = (-0.18, 0.18)
FRONT_SLIDE_VZ_RANGE = (-0.05, 0.18)
DIAGONAL_THROW_VX_RANGE = (0.95, 1.65)
DIAGONAL_THROW_VY_RANGE = (1.00, 1.85)
DIAGONAL_THROW_VZ_RANGE = (-0.12, 0.42)
SIDE_THROW_VX_RANGE = (0.95, 1.40)
SIDE_THROW_VY_RANGE = (0.08, 0.26)
SIDE_THROW_VZ_RANGE = (0.45, 0.90)

REST_CONTACT_MARGIN = 0.02
SLIDE_CONTACT_MARGIN_RANGE = (0.015, 0.035)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_bound_radius(extents: np.ndarray) -> float:
    extents = np.asarray(extents, dtype=np.float64)
    return float(np.linalg.norm(0.5 * extents))


def sample_container_for_objects(scaled_extents: List[np.ndarray]) -> Dict[str, Any]:
    max_ext = np.max(np.stack(scaled_extents, axis=0), axis=0)
    half_x = float(np.clip(max(0.95, 2.1 * max_ext[0] + 0.30), 0.95, 1.70))
    half_y = float(np.clip(max(1.05, 2.4 * max_ext[1] + 0.42), 1.05, 1.95))
    wall_height = float(np.clip(max(1.00, 2.8 * max_ext[2] + 0.42), 1.00, 2.10))
    return {
        "half_x": half_x,
        "half_y": half_y,
        "wall_thickness": 0.10,
        "wall_height": wall_height,
        "floor_thickness": 0.08,
        "center": [0.0, 0.0, 0.0],
        "front_keep_out": 0.38,
        "back_keep_out": 0.12,
        "side_keep_out": 0.08,
    }


def sample_camera(container_cfg: Dict[str, Any]) -> Dict[str, Any]:
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wh = container_cfg["wall_height"]
    cx, cy, cz = container_cfg["center"]
    return {
        "res": [IMG_W, IMG_H],
        "pos": [
            float(cx + np.random.uniform(-0.10, 0.10)),
            float(cy - hy - 1.80 + np.random.uniform(-0.15, 0.10)),
            float(cz + 0.70 * wh + np.random.uniform(-0.08, 0.10)),
        ],
        "lookat": [
            float(cx + np.random.uniform(-0.06, 0.06)),
            float(cy + np.random.uniform(0.12, 0.28)),
            float(cz + 0.24 * wh + np.random.uniform(-0.03, 0.08)),
        ],
        "fov": float(np.random.uniform(38.0, 46.0)),
        "GUI": False,
    }


def sample_spawn_xy(container_cfg: Dict[str, Any], half_x: float, half_y: float, bias_to_back: bool) -> Tuple[float, float]:
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wt = container_cfg["wall_thickness"]

    x_min = -hx + wt + container_cfg["side_keep_out"] + half_x
    x_max = hx - wt - container_cfg["side_keep_out"] - half_x
    y_front = -hy + wt + container_cfg["front_keep_out"] + half_y
    y_back = hy - wt - container_cfg["back_keep_out"] - half_y

    if bias_to_back:
        y_min = max(y_front, 0.05)
        y_max = y_back
    else:
        y_min = y_front
        y_max = y_back

    if x_min >= x_max:
        x_min, x_max = -0.05, 0.05
    if y_min >= y_max:
        y_min, y_max = 0.05, 0.15

    return float(np.random.uniform(x_min, x_max)), float(np.random.uniform(y_min, y_max))


def _random_angvel(scale: float) -> List[float]:
    axis = np.random.uniform(-1.0, 1.0, size=3)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / n
    mag = float(np.random.uniform(0.2, scale))
    return [float(x) for x in axis * mag]


def sample_motion_for_object(
    pattern: str,
    container_cfg: Dict[str, Any],
    scaled_extents: np.ndarray,
    grounding_offset_z: float,
    target_pos: Optional[np.ndarray] = None,
    index_in_scene: int = 0,
    forced_mode: Optional[str] = None,
) -> Dict[str, Any]:
    half_x, half_y, half_z = [float(x) * 0.5 for x in scaled_extents]
    floor_top_z = float(container_cfg["center"][2]) + float(container_cfg["floor_thickness"])
    rest_z = floor_top_z + float(grounding_offset_z) + REST_CONTACT_MARGIN
    mode = forced_mode

    def _sample_front_band() -> Tuple[float, float]:
        x = float(np.random.uniform(-max(0.12, container_cfg["half_x"] * 0.70), max(0.12, container_cfg["half_x"] * 0.70)))
        y = -container_cfg["half_y"] + 0.06 + half_y
        return x, y

    def _sample_side_band(side: str, y_center: Optional[float] = None) -> Tuple[float, float]:
        x_mag = max(container_cfg["half_x"] - half_x - 0.05, 0.12)
        x = -x_mag if side == "left" else x_mag
        if y_center is None:
            y = float(np.random.uniform(-container_cfg["half_y"] * 0.35, container_cfg["half_y"] * 0.25))
        else:
            y = float(np.clip(y_center + np.random.uniform(-0.12, 0.12), -container_cfg["half_y"] * 0.45, container_cfg["half_y"] * 0.30))
        return x, y

    if pattern == "strike_static" and mode is None:
        mode = "static_rest" if index_in_scene == 0 else "strike_static_left"

    if mode == "static_rest":
        x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=True)
        return {
            "motion_type": "static_rest",
            "init_pos": [x, y, rest_z],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [0.0, 0.0, 0.0],
            "init_angvel": [0.0, 0.0, 0.0],
        }

    if pattern == "strike_static":
        if index_in_scene == 0 and mode is None:
            x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=True)
            return {
                "motion_type": "static_rest",
                "init_pos": [x, y, rest_z],
                "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
                "init_linvel": [0.0, 0.0, 0.0],
                "init_angvel": [0.0, 0.0, 0.0],
            }

        strike_target = np.asarray(target_pos if target_pos is not None else [0.0, 0.20, rest_z], dtype=np.float64)
        side = "left" if mode != "strike_static_right" else "right"
        start_x = -container_cfg["half_x"] + 0.12 + half_x if side == "left" else container_cfg["half_x"] - 0.12 - half_x
        start_y = float(np.clip(strike_target[1] + np.random.uniform(-0.08, 0.08), -0.05, 0.35))
        speed = float(np.random.uniform(*STRIKE_SPEED_RANGE))
        x_vel = speed if side == "left" else -speed
        return {
            "motion_type": "strike_static_left" if side == "left" else "strike_static_right",
            "init_pos": [start_x, start_y, rest_z + 0.005],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [x_vel, float(np.random.uniform(-0.10, 0.10)), 0.0],
            "init_angvel": _random_angvel(FRONT_SLIDE_ANGVEL),
        }

    if mode == "front_slide_in":
        x, y = _sample_front_band()
        return {
            "motion_type": "front_slide_in",
            "init_pos": [x, y, rest_z + float(np.random.uniform(*SLIDE_CONTACT_MARGIN_RANGE))],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [float(np.random.uniform(*FRONT_SLIDE_VX_RANGE)), float(np.random.uniform(*FRONT_SLIDE_VY_RANGE)), float(np.random.uniform(*FRONT_SLIDE_VZ_RANGE))],
            "init_angvel": _random_angvel(FRONT_SLIDE_ANGVEL),
        }

    if mode in {"diagonal_corner_left", "diagonal_corner_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side)
        z = max(float(np.random.uniform(*DIAGONAL_ENTRY_Z_RANGE)), rest_z + 0.15)
        vx = abs(float(np.random.uniform(*DIAGONAL_THROW_VX_RANGE)))
        if side == "right":
            vx = -vx
        return {
            "motion_type": mode,
            "init_pos": [x, y, z],
            "init_euler": [float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [vx, float(np.random.uniform(*DIAGONAL_THROW_VY_RANGE)), float(np.random.uniform(*DIAGONAL_THROW_VZ_RANGE))],
            "init_angvel": _random_angvel(DIAGONAL_THROW_ANGVEL),
        }

    if mode in {"side_throw_left", "side_throw_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side)
        z = max(float(np.random.uniform(*SIDE_THROW_Z_RANGE)), rest_z + 0.10)
        vx = abs(float(np.random.uniform(*SIDE_THROW_VX_RANGE)))
        if side == "right":
            vx = -vx
        return {
            "motion_type": mode,
            "init_pos": [x, y, z],
            "init_euler": [float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [vx, float(np.random.uniform(*SIDE_THROW_VY_RANGE)), float(np.random.uniform(*SIDE_THROW_VZ_RANGE))],
            "init_angvel": _random_angvel(SIDE_THROW_ANGVEL),
        }

    static_rest = mode is None and (np.random.rand() < STATIC_REST_PROB)
    if static_rest:
        x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=True)
        return {
            "motion_type": "static_rest",
            "init_pos": [x, y, rest_z],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [0.0, 0.0, 0.0],
            "init_angvel": [0.0, 0.0, 0.0],
        }

    x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=False)
    z = float(np.random.uniform(*TOP_DROP_Z_RANGE))
    mode = "top_drop"
    linvel = [float(np.random.uniform(-0.10, 0.10)), float(np.random.uniform(-0.10, 0.10)), 0.0]

    if mode == "top_toss" or (mode is None and np.random.rand() < 0.45):
        z = float(np.random.uniform(*TOP_TOSS_Z_RANGE))
        mode = "top_toss"
        linvel = [
            float(np.random.uniform(-0.25, 0.25)),
            float(np.random.uniform(-0.20, 0.20)),
            float(np.random.uniform(-0.65, -0.05)),
        ]

    return {
        "motion_type": str(mode),
        "init_pos": [x, y, z],
        "init_euler": [
            float(np.random.uniform(-0.4, 0.4)),
            float(np.random.uniform(-0.4, 0.4)),
            float(np.random.uniform(-math.pi, math.pi)),
        ],
        "init_linvel": linvel,
        "init_angvel": _random_angvel(TOP_TOSS_ANGVEL if mode == "top_toss" else TOP_DROP_ANGVEL),
    }


def scale_soft_parts(soft_parts: List[Dict[str, Any]], scene_scale: float) -> List[Dict[str, Any]]:
    scaled = []
    for rec in soft_parts:
        item = dict(rec)
        item["scene_scale"] = float(scene_scale)
        scaled.append(item)
    return scaled


def resolve_sim_num_steps(dt: float, target_seconds: Optional[float], target_numsteps: Optional[int]) -> int:
    """Outer step count. If both target_numsteps and target_seconds are set, numsteps wins.

    Realized physics time is T = num_steps * dt (may differ slightly from target_seconds when that is inferred,
    due to rounding to integer steps). Preview MP4 duration is set to exactly T in export_scene.
    """
    if target_numsteps is not None:
        return max(1, int(target_numsteps))
    if target_seconds is not None:
        return max(1, int(round(float(target_seconds) / max(float(dt), 1e-6))))
    raise ValueError("Specify at least one of target_seconds or target_numsteps (total simulation steps).")


# =========================
# 通用工具函数（从 mpm_genesis 和 rigid_genesis 提取）
# =========================

def ensure_dir(path: Path) -> None:
    """创建目录，如果不存在则递归创建"""
    path.mkdir(parents=True, exist_ok=True)


def weighted_choice(d: Dict[str, float]) -> str:
    """按权重随机选择字典中的一个键"""
    keys = list(d.keys())
    probs = np.asarray(list(d.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return str(np.random.choice(keys, p=probs))


def to_numpy(x: Any) -> Optional[np.ndarray]:
    """将各种类型转换为 numpy 数组"""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def safe_subsample_points(xyz: np.ndarray, max_points: int = 2048) -> np.ndarray:
    """安全地对点云进行子采样"""
    xyz = np.asarray(xyz)
    if len(xyz) <= max_points:
        return xyz.astype(np.float32, copy=False)
    idx = np.random.choice(len(xyz), size=max_points, replace=False)
    return xyz[idx].astype(np.float32, copy=False)


def save_depth_vis(depth: np.ndarray, out_path: Path) -> None:
    """保存深度图的可视化版本"""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if np.any(valid):
        d = np.asarray(depth[valid], dtype=np.float32).reshape(-1)
        if d.size == 0:
            imageio.imwrite(out_path, vis)
            return
        lo = float(np.min(d))
        hi = float(np.max(d))
        hi = max(hi, lo + 1e-5)
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        gray = (255.0 * (1.0 - norm)).astype(np.uint8)
        vis[..., 0] = gray
        vis[..., 1] = gray
        vis[..., 2] = gray
    imageio.imwrite(out_path, vis)


def compute_preview_stride(dt: float, num_steps: int, phys_duration_s: float, preview_target_fps: int = 30) -> int:
    """计算预览视频的采样间隔，使预览视频帧率约为 preview_target_fps"""
    dt = float(max(dt, 1e-6))
    n = max(1, int(num_steps))
    phys_duration_s = float(max(phys_duration_s, n * dt))
    target_frames = max(1.0, phys_duration_s * float(preview_target_fps))
    return max(1, int(round(float(n) / target_frames)))


def _try_call_methods(obj: Any, method_names: List[str], value: Any) -> bool:
    """尝试调用对象的多个方法之一，用于兼容不同 Genesis 版本"""
    for name in method_names:
        if hasattr(obj, name):
            fn = getattr(obj, name)
            try:
                fn(value)
                return True
            except Exception:
                try:
                    fn(tuple(np.asarray(value).tolist()))
                    return True
                except Exception:
                    pass
    return False


def apply_initial_motion_to_entity(ent: Any, linvel: List[float], angvel: List[float]) -> None:
    """给实体施加初始线速度和角速度"""
    v = np.asarray(linvel, dtype=np.float32)
    w = np.asarray(angvel, dtype=np.float32)
    if np.linalg.norm(v) > 0:
        _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], v)
    if np.linalg.norm(w) > 0:
        _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], w)


def create_genesis_rigid_material(gs: Any, mat_cfg: Dict[str, Any]):
    """从配置创建 Genesis 刚体材料"""
    kwargs = {
        "rho": float(mat_cfg["rho"]),
        "friction": float(mat_cfg["friction"]),
    }
    if mat_cfg.get("restitution") is not None:
        kwargs["restitution"] = float(mat_cfg["restitution"])
    try:
        return gs.materials.Rigid(**kwargs)
    except TypeError:
        kwargs.pop("restitution", None)
        return gs.materials.Rigid(**kwargs)


def prepare_output_dirs(out_dir: Path, subdirs: Optional[List[str]] = None) -> None:
    """准备输出目录结构"""
    if subdirs is None:
        subdirs = [
            "rgb", "depth", "depth_vis", "segmentation", "normal", "pointcloud",
            "object_pointcloud", "trajectories", "camera", "video"
        ]
    for sub in subdirs:
        ensure_dir(out_dir / sub)


def export_entity_state(ent: Any, state_spec: Dict[str, Any]) -> Dict[str, Any]:
    """导出实体的当前状态（位置、速度等）"""
    state = {
        "object_id": state_spec["object_id"],
        "solver": state_spec["solver"],
        "centroid": None,
        "quat": None,
        "vel": None,
        "ang": None,
        "pointcloud": None,
        "n_points": 0,
    }

    if hasattr(ent, "get_particles_pos"):
        pts = to_numpy(ent.get_particles_pos())
        if pts is not None and pts.size > 0:
            pts = pts.reshape(-1, 3)
            state["pointcloud"] = pts
            state["centroid"] = pts.mean(axis=0)
            state["n_points"] = int(len(pts))
            return state

    if hasattr(ent, "get_verts"):
        verts = to_numpy(ent.get_verts())
        if verts is not None and verts.size > 0:
            verts = verts.reshape(-1, 3)
            state["pointcloud"] = verts
            state["centroid"] = verts.mean(axis=0)
            state["n_points"] = int(len(verts))

    if hasattr(ent, "get_pos"):
        try:
            pos = to_numpy(ent.get_pos()).reshape(-1)
            state["centroid"] = pos[:3]
        except Exception:
            pass

    if hasattr(ent, "get_quat"):
        try:
            quat = to_numpy(ent.get_quat()).reshape(-1)
            state["quat"] = quat[:4]
        except Exception:
            pass

    if hasattr(ent, "get_vel"):
        try:
            vel = to_numpy(ent.get_vel()).reshape(-1)
            state["vel"] = vel[:3]
        except Exception:
            pass

    if hasattr(ent, "get_ang"):
        try:
            ang = to_numpy(ent.get_ang()).reshape(-1)
            state["ang"] = ang[:3]
        except Exception:
            pass

    return state


def add_container(gs: Any, scene: Any, container_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建容器（地面、墙壁等）
    
    Args:
        gs: Genesis 模块
        scene: Genesis 场景对象
        container_cfg: 容器配置字典
    
    Returns:
        container_entities: 容器实体字典
    """
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wt = container_cfg["wall_thickness"]
    wh = container_cfg["wall_height"]
    ft = container_cfg["floor_thickness"]
    cx, cy, cz = container_cfg["center"]

    mat = gs.materials.Rigid(rho=1200.0, friction=0.98)
    container_entities: Dict[str, Dict[str, Any]] = {}
    
    floor_pos = (cx, cy, cz + ft / 2.0)
    container_entities["floor"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(2 * hx, 2 * hy, ft), pos=floor_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.70, 0.72, 0.76, 1.0)),
        ),
        "anchor_pos": floor_pos,
    }
    
    left_wall_pos = (cx - hx + wt / 2.0, cy, cz + ft + wh / 2.0)
    container_entities["left_wall"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(wt, 2 * hy, wh), pos=left_wall_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.80, 0.63, 0.61, 1.0)),
        ),
        "anchor_pos": left_wall_pos,
    }
    
    right_wall_pos = (cx + hx - wt / 2.0, cy, cz + ft + wh / 2.0)
    container_entities["right_wall"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(wt, 2 * hy, wh), pos=right_wall_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.62, 0.80, 0.67, 1.0)),
        ),
        "anchor_pos": right_wall_pos,
    }
    
    back_wall_pos = (cx, cy + hy - wt / 2.0, cz + ft + wh / 2.0)
    container_entities["back_wall"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(2 * hx, wt, wh), pos=back_wall_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.63, 0.71, 0.84, 1.0)),
        ),
        "anchor_pos": back_wall_pos,
    }
    
    return container_entities


def build_scene_generation_plan(samples_per_category: int, motion_category_specs: List[Dict[str, Any]], object_counts: List[int]) -> List[Dict[str, Any]]:
    """
    构建场景生成计划
    
    Args:
        samples_per_category: 每个类别的样本数
        motion_category_specs: 运动类别规格列表
        object_counts: 对象计数列表
    
    Returns:
        plan: 场景生成计划列表
    """
    plan = []
    for category_spec in motion_category_specs:
        if category_spec["scene_builder"] == "uniform_dynamic":
            for object_count in object_counts:
                for sample_idx in range(samples_per_category):
                    plan.append({
                        "category_spec": category_spec,
                        "object_count": int(object_count),
                        "sample_index": int(sample_idx),
                    })
        elif category_spec["scene_builder"] == "ground_static_plus_dynamic":
            for sample_idx in range(samples_per_category):
                plan.append({
                    "category_spec": category_spec,
                    "object_count": None,
                    "sample_index": int(sample_idx),
                })
        else:
            raise ValueError(f"Unknown scene_builder: {category_spec['scene_builder']}")
    return plan


def sample_mpm_scene_cfg(
    scene_id: int,
    scene_plan: Dict[str, Any],
    asset_bank: List[Dict[str, Any]],
    seed: int,
    target_seconds: Optional[float] = None,
    target_numsteps: Optional[int] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    category_spec = scene_plan["category_spec"]
    object_count = scene_plan["object_count"]
    if object_count is None:
        n_objects = random.randint(2, min(4, len(asset_bank)))
    else:
        n_objects = min(int(object_count), len(asset_bank))

    chosen_assets = random.sample(asset_bank, k=max(1, n_objects))
    pattern = "strike_static" if category_spec["name"] == "ground_static_plus_dynamic" else category_spec["name"]

    scene_scales = []
    scaled_extents = []
    for asset in chosen_assets:
        ext = np.asarray(asset["bbox_extents"], dtype=np.float64)
        target_longest = float(np.random.uniform(*TARGET_LONGEST_SIZE_RANGE))
        scene_scale = target_longest / max(float(np.max(ext)), 1e-8)
        scene_scales.append(scene_scale)
        scaled_extents.append(ext * scene_scale)

    container_cfg = sample_container_for_objects(scaled_extents)
    camera_cfg = sample_camera(container_cfg)

    objects = []
    strike_target = None
    motion_modes_present = []
    num_static_objects = 0
    for idx, (asset, scene_scale, extents) in enumerate(zip(chosen_assets, scene_scales, scaled_extents)):
        forced_mode = None
        if category_spec["scene_builder"] == "uniform_dynamic":
            forced_mode = category_spec["motion_modes"][idx % len(category_spec["motion_modes"])]
        elif category_spec["scene_builder"] == "ground_static_plus_dynamic":
            if idx == 0:
                forced_mode = "static_rest"
            else:
                forced_mode = category_spec["motion_modes"][(idx - 1) % len(category_spec["motion_modes"])]

        motion = sample_motion_for_object(
            pattern=pattern,
            container_cfg=container_cfg,
            scaled_extents=np.asarray(extents, dtype=np.float64),
            grounding_offset_z=float(asset["grounding_offset_z"]) * float(scene_scale),
            target_pos=strike_target,
            index_in_scene=idx,
            forced_mode=forced_mode,
        )
        if idx == 0:
            strike_target = np.asarray(motion["init_pos"], dtype=np.float64)
        motion_modes_present.append(str(motion["motion_type"]))
        if motion["motion_type"] == "static_rest":
            num_static_objects += 1

        obj_rec = {
            "scene_object_id": idx,
            "asset_id": asset["asset_id"],
            "source_object_id": asset["object_id"],
            "object_name": asset["object_name"],
            "category": asset["category"],
            "pattern": pattern,
            "motion_type": motion["motion_type"],
            "init_pos": [float(x) for x in motion["init_pos"]],
            "init_euler": [float(x) for x in motion["init_euler"]],
            "init_linvel": [float(x) for x in motion["init_linvel"]],
            "init_angvel": [float(x) for x in motion["init_angvel"]],
            "prepared_asset_dir": asset["asset_dir"],
            "prepared_metadata_path": asset["metadata_path"],
            "grounding_offset_z": float(asset["grounding_offset_z"]) * float(scene_scale),
            "bbox_extents": np.asarray(extents, dtype=np.float64).tolist(),
            "scene_scale": float(scene_scale),
            "soft_parts": scale_soft_parts(asset["metadata"].get("soft_parts", []), scene_scale),
            "rigid_part_count": int(asset["rigid_part_count"]),
            "soft_part_count": int(asset["soft_part_count"]),
        }

        if asset["dataset_name"] == "physxnet_articulation":
            obj_rec.update(
                {
                    "solver": "ArticulationRigid",
                    "source_type": "physxnet_articulation",
                    "geom": {
                        "shape": "urdf",
                        "urdf_file": asset["urdf_path"],
                        "scale": float(scene_scale),
                        "bbox_extents": np.asarray(extents, dtype=np.float64).tolist(),
                        "bound_radius": compute_bound_radius(np.asarray(extents, dtype=np.float64)),
                    },
                    "material": dict(asset["material_override"]),
                }
            )
        else:
            obj_rec.update(
                {
                    "solver": "SoftMPM",
                    "source_type": "sophy",
                    "geom": {
                        "shape": "mesh_collection",
                        "scale": float(scene_scale),
                        "bbox_extents": np.asarray(extents, dtype=np.float64).tolist(),
                        "bound_radius": compute_bound_radius(np.asarray(extents, dtype=np.float64)),
                    },
                    "material": None,
                }
            )

        objects.append(obj_rec)

    num_dynamic_objects = len(objects) - num_static_objects
    if object_count is None:
        count_dir = f"count_mixed_s{num_static_objects}_d{num_dynamic_objects}"
        scene_name_suffix = f"s{num_static_objects}_d{num_dynamic_objects}"
    else:
        count_dir = f"count_{int(object_count):02d}"
        scene_name_suffix = f"n{int(object_count):02d}"
    scene_id_str = f"{category_spec['name']}__{scene_name_suffix}__sample_{int(scene_plan['sample_index']):04d}"
    output_relpath = Path("train") / category_spec["name"] / count_dir / scene_id_str

    dt = 1e-3
    sim_steps = resolve_sim_num_steps(dt=dt, target_seconds=target_seconds, target_numsteps=target_numsteps)

    return {
        "scene_id": scene_id_str,
        "seed": seed,
        "family": "mpm_mixed_assets",
        "mpm_motion_category": category_spec["name"],
        "mpm_motion_label_zh": category_spec.get("label_zh", category_spec["name"]),
        "scene_builder": category_spec["scene_builder"],
        "object_count_bucket": object_count,
        "sample_index_in_bucket": int(scene_plan["sample_index"]),
        "num_static_objects": int(num_static_objects),
        "num_dynamic_objects": int(num_dynamic_objects),
        "motion_modes_present": sorted(set(motion_modes_present)),
        "output_relpath": str(output_relpath),
        "pattern": pattern,
        "container": container_cfg,
        "camera": camera_cfg,
        "background": {
            "name": "plain_open_container",
            "background_color": [0.96, 0.97, 0.99, 1.0],
            "ambient_light": [0.65, 0.65, 0.65],
        },
        "sim_options": {
            "gravity": [0.0, 0.0, -9.81],
            "dt": dt,
            "substeps": 10,
            "num_steps": sim_steps,
        },
        "target_seconds": target_seconds,
        "target_numsteps": target_numsteps,
        "objects": objects,
    }
