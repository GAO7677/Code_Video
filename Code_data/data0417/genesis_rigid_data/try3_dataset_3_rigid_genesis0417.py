import json
import csv
import math
import random
import colorsys
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import numpy as np
import imageio.v2 as imageio
import genesis as gs
import trimesh

THIS_DIR = Path(__file__).resolve().parent
CODE_DATA_ROOT = THIS_DIR.parent.parent
if str(CODE_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_DATA_ROOT))

from genesis_energy_utils import rigid_entity_kinematic_snapshot

from old_version.dataset_3_utils_genesis import (
    set_seed,
    compute_bound_radius,
    sample_container_for_objects,
    sample_camera,
    sample_spawn_xy as utils_sample_spawn_xy,
    resolve_sim_num_steps,
    ensure_dir,
    weighted_choice,
    to_numpy,
    safe_subsample_points,
    save_depth_vis,
    compute_preview_stride,
    _try_call_methods,
    apply_initial_motion_to_entity,
    create_genesis_rigid_material,
    prepare_output_dirs,
    export_entity_state,
    add_container,
    build_scene_generation_plan,
)


# =========================
# 基本配置
# =========================
DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/genesis_rigid_multimodal_49f")

EXPORT_FRAMES = 49
EXPORT_RESOLUTION = 512
EXPORT_STEPS_PER_FRAME = 5
IMG_W, IMG_H = EXPORT_RESOLUTION, EXPORT_RESOLUTION
SAMPLES_PER_CATEGORY = 10

MAX_OBJECT_PC = 2048
OBJECT_PC_STRIDE = 5
CAMERA_PC_STRIDE = 2
PREVIEW_FPS = 30

STOP_ON_ERROR = False

# 可选强制场景类型 / rigid pattern。
# 例如：FORCE_SCENE_FAMILY = "rigid_mix"; FORCE_RIGID_PATTERN = None
FORCE_SCENE_FAMILY = "rigid_only"
FORCE_RIGID_PATTERN = None

SCENE_FAMILY_WEIGHTS = {
    "rigid_only": 1.0,
}

# =========================
# 数据集 asset 配置
# 说明：
# 1) 仍然以当前这份“场景脚本”为主：容器 / 相机 / 运动模式 / 导出流程都不变
# 2) 这里只给“物体资产来源”留接口，可在 SOPHY / PhysX-3D / mixed 之间切换
# =========================

# 可选：
# - "sophy"     : 只使用 SOPHY mesh 资产
# - "physx3d"   : 只使用 PhysX-3D 资产
# - "mixed"     : sophy + physx3d
# - "primitive" : 只使用普通几何体“数据集”
# - "all"       : sophy + physx3d + primitive
DATASET_SOURCE = "all"

# =========================
# Primitive 几何体“数据集”配置
# =========================
PRIMITIVE_DATASET_NAME = "primitive"
PRIMITIVE_ASSET_REPEAT = 80

PRIMITIVE_SHAPE_WEIGHTS = {
    "box": 0.34,
    "sphere": 0.24,
    "cylinder": 0.24,
    "capsule": 0.18,
}

PRIMITIVE_MATERIAL_PRESETS = {
    "wood": {
        "rho": 700.0,
        "friction": 0.62,
        "restitution": 0.12,
        "color_range": [[0.45, 0.28, 0.12], [0.76, 0.60, 0.36]],
    },
    "plastic": {
        "rho": 1050.0,
        "friction": 0.42,
        "restitution": 0.18,
        "color_range": [[0.15, 0.15, 0.15], [0.95, 0.95, 0.95]],
    },
    "rubber": {
        "rho": 1150.0,
        "friction": 1.05,
        "restitution": 0.72,
        "color_range": [[0.02, 0.02, 0.02], [0.20, 0.20, 0.20]],
    },
    "metal": {
        "rho": 2700.0,
        "friction": 0.24,
        "restitution": 0.10,
        "color_range": [[0.55, 0.55, 0.58], [0.85, 0.85, 0.90]],
    },
    "foam": {
        "rho": 90.0,
        "friction": 0.78,
        "restitution": 0.05,
        "color_range": [[0.60, 0.60, 0.65], [0.95, 0.95, 1.00]],
    },
    "glass": {
        "rho": 2500.0,
        "friction": 0.16,
        "restitution": 0.08,
        "color_range": [[0.72, 0.85, 0.92], [0.92, 0.98, 1.00]],
    },
}

PRIMITIVE_MATERIAL_WEIGHTS = {
    "wood": 0.22,
    "plastic": 0.22,
    "rubber": 0.18,
    "metal": 0.16,
    "foam": 0.12,
    "glass": 0.10,
}

# ----- SOPHY -----
SOURCE_DATASET_ROOTS = [
    Path("/data/gaoya/dataset/SOPHY_data/bag"),
    Path("/data/gaoya/dataset/SOPHY_data/teddy_bear"),
]

# ----- PhysX-3D -----
PHYSX3D_ROOT = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
PHYSX3D_VERSION = "version_1"
PHYSX3D_MAX_OBJECTS = 50          # None / 0 表示尽量读取全量
PHYSX3D_OBJECT_IDS = None         # 例如 ["000001", "000123"]；None 表示不过滤
PHYSX3D_USE_PART_DENSITY = True   # 尝试从 finaljson 的 parts 字段里估计密度
PHYSX3D_FORCE_COLOR = True        # PhysX-3D 默认无纹理，用随机颜色可视化
PHYSX3D_USE_PART_COLORED_URDF = True   # 用单刚体 URDF + 多 visual part，实现同一物体不同 part 不同颜色
PHYSX3D_URDF_USE_URDF_MATERIAL = True  # 优先使用 URDF 中每个 part 的颜色
PHYSX3D_COLLISION_USE_MERGED_MESH = True

# rigid_mix 场景里，一个物体采样为 dataset mesh 的概率
USE_DATASET_MESH_OBJECTS = True
DATASET_OBJECT_PROB = 1.0

# 调试时可设成小整数；None 表示不截断
MAX_ASSETS_PER_ROOT = None

ASSET_CACHE_DIR = Path("/data/gaoya/AAA_test_video/Dataset_test/genesis_sim_scene_interface_cache")
ASSET_CACHE_DIR = ASSET_CACHE_DIR / "_asset_cache"
ASSET_MANIFEST_PATH = ASSET_CACHE_DIR / "asset_manifest.json"
PHYSX3D_MERGED_CACHE_DIR = ASSET_CACHE_DIR / "_physx3d_merged"
PHYSX3D_PART_URDF_CACHE_DIR = ASSET_CACHE_DIR / "_physx3d_part_urdf"
# SOPHY：按 mat_params + usemtl 切分部件，多 link + fixed joint URDF，刚体侧用各 part 的 rho 算质量/惯性（E/nu/sigma_y 写入元数据供 MPM/后处理；Genesis Rigid 仍只有 rho/摩擦/恢复）
SOPHY_PART_URDF_CACHE_DIR = ASSET_CACHE_DIR / "_sophy_part_urdf"
USE_SOPHY_PART_MATERIALS_RIGID = True
SOPHY_RIGID_PART_DEFAULT_RESTITUTION = 0.10

TARGET_MESH_SIZE_RANGE = (0.2, 0.5)      # 最长边目标尺寸（米）
SIMPLIFY_MESH_FACE_COUNT = 3000           # None 表示不减面；建议 2000~5000
MIN_VALID_MESH_EXTENT = 1e-5

PART_COLOR_STYLE_VERSION = 2
PART_COLOR_BANK = [
    [0.90, 0.24, 0.21],
    [0.08, 0.47, 0.82],
    [0.96, 0.69, 0.16],
    [0.16, 0.64, 0.43],
    [0.57, 0.31, 0.93],
    [0.92, 0.29, 0.56],
    [0.07, 0.71, 0.78],
    [0.86, 0.53, 0.15],
    [0.38, 0.78, 0.22],
    [0.22, 0.34, 0.89],
    [0.73, 0.19, 0.33],
    [0.12, 0.55, 0.31],
]
EXPORT_ENHANCED_RGB = True
ENHANCED_RGB_DIRNAME = "rgb_enhanced"
ENHANCED_RGB_VIDEO_NAME = "rgb_enhanced.mp4"
ENHANCED_RGB_COLOR_EDGE_THRESHOLD = 0.15
ENHANCED_RGB_LUMA_EDGE_THRESHOLD = 0.10
ENHANCED_RGB_DEPTH_EDGE_THRESHOLD = 0.02
ENHANCED_RGB_EDGE_DILATION = 0


# 容器：开口朝 -y，相机放在前方（负 y）看进去
# 调整目标：
# 1) 容器尽量放大，提升“物体留在容器内部”的概率
# 2) 改成真正三面体：地面 + 左右墙 + 后墙，前方完全开口
# 3) 运动模式增加“前方滑入 / 对角入场 / 进入场景撞击静止物体”
CONTAINER = {
    "half_x": 1.50,           # 总宽 3.0m
    "half_y": 1.50,           # 总深 3.0m
    "wall_thickness": 0.04,
    "wall_height": 2.00,
    "front_lip_height": 0.00, # 三面体，不使用前挡板
    "floor_thickness": 0.05,
    "center": [0.0, 0.0, 0.0],
}

# 出生区域安全边距：
# 前开口方向（-y）预留更大 buffer，尽量把物体出生点压到容器中后部
SPAWN_FRONT_KEEP_OUT = 0.42
SPAWN_BACK_KEEP_OUT = 0.10
SPAWN_SIDE_KEEP_OUT = 0.06

# =========================
# 数据集 mesh 的坐标系修正
# =========================
# 外部资产默认按 Y-up 进入；Genesis 场景采用 Z-up。
# 统一先做 source(Y-up) -> scene(Z-up) 的基准旋转，再叠加随机姿态。
YUP_TO_ZUP_EULER_XYZ = [math.pi / 2.0, 0.0, 0.0]

# 外部资产默认按 X-up 进入；Genesis 场景采用 Z-up。
# XUP_TO_ZUP_EULER_XYZ = [0.0, 0.0, math.pi / 2.0]



DATASET_UP_AXIS_BY_DATASET = {
    PRIMITIVE_DATASET_NAME: "z_up",
    "bag": "y_up",
    "teddy_bear": "y_up",
    "physx3d": "y_up",
}

DATASET_EXTRA_BASE_EULER_BY_DATASET = {}


def compute_bound_radius_from_half_extents(half_x: float, half_y: float, half_z: float) -> float:
    """Convert half extents to full extents before delegating to the shared utility."""
    extents = np.array([2.0 * half_x, 2.0 * half_y, 2.0 * half_z], dtype=np.float32)
    return compute_bound_radius(extents)


def sample_spawn_xy(
    half_x: float,
    half_y: float,
    bias_to_back: bool,
    container_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Backward-compatible wrapper around the shared spawn sampler."""
    cfg = dict(container_cfg if container_cfg is not None else CONTAINER)
    cfg.setdefault("front_keep_out", SPAWN_FRONT_KEEP_OUT)
    cfg.setdefault("back_keep_out", SPAWN_BACK_KEEP_OUT)
    cfg.setdefault("side_keep_out", SPAWN_SIDE_KEEP_OUT)
    return utils_sample_spawn_xy(cfg, float(half_x), float(half_y), bool(bias_to_back))


def rgb_to_uint8(rgb: Any) -> np.ndarray:
    array = np.asarray(to_numpy(rgb))
    if array.dtype == np.uint8:
        return array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array[..., :3], 0.0, 1.0) * 255.0
    return np.clip(np.round(array[..., :3]), 0.0, 255.0).astype(np.uint8)


def normalize_depth_map(depth: Any, near: float, far: float) -> np.ndarray:
    array = np.asarray(to_numpy(depth), dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
    valid = np.isfinite(array)
    normalized = np.full(array.shape, 1.0, dtype=np.float32)
    if far <= near:
        return normalized[..., None]
    normalized[valid] = np.clip((array[valid] - near) / max(far - near, 1e-8), 0.0, 1.0)
    return normalized[..., None]


def depth_to_uint8(depth_norm: np.ndarray) -> np.ndarray:
    depth_img = np.asarray(depth_norm, dtype=np.float32)
    if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
        depth_img = depth_img[..., 0]
    return np.clip(np.round(depth_img * 255.0), 0.0, 255.0).astype(np.uint8)


def _compute_neighbor_edge_mask(array: np.ndarray, threshold: float) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        diff_x = np.max(np.abs(arr[:, 1:, :] - arr[:, :-1, :]), axis=-1)
        diff_y = np.max(np.abs(arr[1:, :, :] - arr[:-1, :, :]), axis=-1)
        h, w = arr.shape[:2]
    else:
        diff_x = np.abs(arr[:, 1:] - arr[:, :-1])
        diff_y = np.abs(arr[1:, :] - arr[:-1, :])
        h, w = arr.shape[:2]
    edge = np.zeros((h, w), dtype=bool)
    edge[:, 1:] |= diff_x > float(threshold)
    edge[:, :-1] |= diff_x > float(threshold)
    edge[1:, :] |= diff_y > float(threshold)
    edge[:-1, :] |= diff_y > float(threshold)
    return edge


def _dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    radius = max(int(radius), 0)
    out = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return out
    for _ in range(radius):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        neigh = [padded[1 + dy:1 + dy + out.shape[0], 1 + dx:1 + dx + out.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        out = np.logical_or.reduce(neigh)
    return out


def enhance_part_visibility(rgb_frame: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
    rgb_u8 = np.asarray(rgb_frame, dtype=np.uint8)
    rgb = rgb_u8.astype(np.float32) / 255.0
    depth = np.asarray(depth_frame, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    luma = np.tensordot(rgb, np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=([-1], [0]))
    color_edges = _compute_neighbor_edge_mask(rgb, ENHANCED_RGB_COLOR_EDGE_THRESHOLD)
    luma_edges = _compute_neighbor_edge_mask(luma, ENHANCED_RGB_LUMA_EDGE_THRESHOLD)
    depth_edges = _compute_neighbor_edge_mask(depth, ENHANCED_RGB_DEPTH_EDGE_THRESHOLD)
    edge_mask = _dilate_mask(color_edges | luma_edges | depth_edges, radius=ENHANCED_RGB_EDGE_DILATION)

    out = rgb_u8.astype(np.float32).copy()
    dark_edges = edge_mask & (luma >= 0.45)
    light_edges = edge_mask & ~dark_edges
    out[dark_edges] = out[dark_edges] * 0.70 + np.asarray([20.0, 20.0, 24.0], dtype=np.float32) * 0.30
    out[light_edges] = out[light_edges] * 0.72 + np.asarray([244.0, 244.0, 244.0], dtype=np.float32) * 0.28
    return np.clip(np.round(out), 0.0, 255.0).astype(np.uint8)

# =========================
# rigid 场景模式
# =========================
DATASET_OBJECT_COUNTS = [1, 2, 3,4,5]

RIGID_MOTION_CATEGORY_SPECS = [
    {
        "name": "top_drop_only",
        "label_zh": "下坠",
        "motion_modes": ["top_drop"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [1],
        "force_source": "dataset",
    },
    {
        "name": "top_toss_only",
        "label_zh": "上方抛掷",
        "motion_modes": ["top_toss"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [1],
        "force_source": "dataset",
    },
    {
        "name": "front_slide_only",
        "label_zh": "前向滑入",
        "motion_modes": ["front_slide_in"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
    },
    {
        "name": "diagonal_left_only",
        "label_zh": "左前对角抛入",
        "motion_modes": ["diagonal_corner_left"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
    },
    {
        "name": "diagonal_right_only",
        "label_zh": "右前对角抛入",
        "motion_modes": ["diagonal_corner_right"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
    },
    {
        "name": "side_throw_left_only",
        "label_zh": "左侧抛入",
        "motion_modes": ["side_throw_left"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
    },
    {
        "name": "side_throw_right_only",
        "label_zh": "右侧抛入",
        "motion_modes": ["side_throw_right"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
    },
    {
        "name": "rolling_left_only",
        "label_zh": "左侧滚动",
        "motion_modes": ["rolling_left"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [3],
        "force_source": "dataset",
        "force_shape_map": {"rolling_left": "sphere"},
    },
    {
        "name": "rolling_right_only",
        "label_zh": "右侧滚动",
        "motion_modes": ["rolling_right"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [3],
        "force_source": "dataset",
        "force_shape_map": {"rolling_right": "sphere"},
    },
    {
        "name": "projectile_arc_only",
        "label_zh": "抛物线入场",
        "motion_modes": ["projectile_arc_forward"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
        "force_shape_map": {"projectile_arc_forward": "sphere"},
    },
    {
        "name": "projectile_cross_left_only",
        "label_zh": "左侧抛物线横穿",
        "motion_modes": ["projectile_cross_left"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
        "force_shape_map": {"projectile_cross_left": "sphere"},
    },
    {
        "name": "projectile_cross_right_only",
        "label_zh": "右侧抛物线横穿",
        "motion_modes": ["projectile_cross_right"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
        "force_shape_map": {"projectile_cross_right": "sphere"},
    },
    {
        "name": "swing_drop_left_only",
        "label_zh": "左侧摆入",
        "motion_modes": ["swing_drop_left"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
        "dataset_source_mix": ["primitive"],
    },
    {
        "name": "swing_drop_right_only",
        "label_zh": "右侧摆入",
        "motion_modes": ["swing_drop_right"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [2],
        "force_source": "dataset",
        "dataset_source_mix": ["primitive"],
    },
    {
        "name": "ground_static_cluster",
        "label_zh": "地面静止簇",
        "motion_modes": ["static_rest"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [6],
        "force_source": "dataset",
        "dataset_source_mix": ["physx3d", "sophy", "primitive"],
    },
    {
        "name": "ground_static_with_intruder",
        "label_zh": "地面静止加侧向侵入",
        "motion_modes": ["static_rest", "static_rest", "static_rest", "rolling_left", "front_slide_in"],
        "scene_builder": "uniform_dynamic",
        "object_counts": [5],
        "force_source": "dataset",
        "dataset_source_mix": ["physx3d", "sophy", "primitive"],
    },
    {
        "name": "interaction_pair_multi_motion",
        "label_zh": "单组交互加多运动",
        "motion_modes": [
            "static_rest",
            "strike_static_left",
            "strike_static_right",
            "rolling_left",
            "rolling_right",
            "projectile_arc_forward",
            "projectile_cross_left",
            "projectile_cross_right",
            "swing_drop_left",
            "swing_drop_right",
            "top_drop",
            "top_toss",
        ],
        "scene_builder": "interaction_pair_plus_dynamic",
        "object_counts": [5],
        "force_source": "dataset",
        "dataset_source_mix": ["physx3d", "sophy", "primitive"],
    },
    {
        "name": "dual_interaction_groups",
        "label_zh": "双组交互场景",
        "motion_modes": [
            "static_rest",
            "strike_static_left",
            "strike_static_right",
            "rolling_left",
            "rolling_right",
            "projectile_arc_forward",
            "projectile_cross_left",
            "projectile_cross_right",
            "swing_drop_left",
            "swing_drop_right",
            "top_drop",
            "top_toss",
        ],
        "scene_builder": "dual_interaction_groups",
        "object_counts": [8],
        "force_source": "dataset",
        "dataset_source_mix": ["physx3d", "sophy", "primitive"],
    },
    {
        "name": "omni_showcase_all_modes",
        "label_zh": "全模式综合展示",
        "motion_modes": [
            "static_rest",
            "strike_static_left",
            "strike_static_right",
            "front_slide_in",
            "diagonal_corner_left",
            "diagonal_corner_right",
            "side_throw_left",
            "side_throw_right",
            "rolling_left",
            "rolling_right",
            "projectile_arc_forward",
            "projectile_cross_left",
            "projectile_cross_right",
            "swing_drop_left",
            "swing_drop_right",
        ],
        "scene_builder": "omni_showcase",
        "object_counts": [12],
        "force_source": "dataset",
        "dataset_source_mix": ["physx3d", "sophy", "primitive"],
    },
]

# =========================
# rigid 运动模式
# 说明：
# - top_drop / top_toss: 上方入场
# - front_slide_in: 从前开口低位滑入/冲入
# - diagonal_corner_*: 从前侧上方向对角线打进容器
# - side_throw_*: 从左右外侧扔入
# - rolling_*: 低位滚入
# - projectile_*: 抛物线入场
# - swing_drop_*: 高位摆入（无关节，初速度近似摆入）
# - static_rest: 初始静止在容器内部
# - strike_static_*: 针对静止目标定向撞击
# =========================
RIGID_MOTION_WEIGHTS = {
    "top_drop": 0.07,
    "top_toss": 0.05,
    "front_slide_in": 0.12,
    "diagonal_corner_left": 0.08,
    "diagonal_corner_right": 0.08,
    "side_throw_left": 0.05,
    "side_throw_right": 0.05,
    "rolling_left": 0.10,
    "rolling_right": 0.10,
    "projectile_arc_forward": 0.08,
    "projectile_cross_left": 0.04,
    "projectile_cross_right": 0.04,
    "swing_drop_left": 0.05,
    "swing_drop_right": 0.05,
    "static_rest": 0.04,
}


# =========================
# rigid 场景中的“地面静止物体”混入控制
# 这些物体一开始就静止放在容器地面上，后续仍可被撞动
# =========================
STATIC_PROP_PROB_BY_PATTERN = {
    "drop_cluster": 1.00,
    "opposed_lanes": 1.00,
    "strike_static": 1.00,
    "chain_reaction": 1.00,
}

STATIC_PROP_COUNT_RANGE_BY_PATTERN = {
    "drop_cluster": (1, 2),
    "opposed_lanes": (1, 2),
    "strike_static": (1, 2),   # strike_static 本身已有 1 个静止目标
    "chain_reaction": (1, 2),  # chain_reaction 本身已有 2 个静止目标
}

# 保证每种 pattern 至少还剩下这些会动的关键物体
MIN_DYNAMIC_COUNT_BY_PATTERN = {
    "drop_cluster": 2,
    "opposed_lanes": 2,
    "strike_static": 2,   # 1 target + 1 striker
    "chain_reaction": 3,  # 2 target + 1 striker
}

TOP_DROP_Z_RANGE = (1.10, 1.70)
TOP_TOSS_Z_RANGE = (1.05, 1.60)
SIDE_THROW_Z_RANGE = (0.75, 1.25)
FRONT_SLIDE_Z_RANGE = (0.10, 0.35)
DIAGONAL_ENTRY_Z_RANGE = (0.95, 1.60)

TOP_DROP_VXY = 0.08
TOP_TOSS_VX = 0.38
TOP_TOSS_VY = 0.22
TOP_TOSS_VZ_RANGE = (-1.00, -0.20)

FRONT_SLIDE_VY_RANGE = (1.10, 1.85)
FRONT_SLIDE_VX_RANGE = (-0.22, 0.22)
FRONT_SLIDE_VZ_RANGE = (-0.05, 0.20)

DIAGONAL_THROW_VX_RANGE = (0.95, 1.65)
DIAGONAL_THROW_VY_RANGE = (1.00, 1.85)
DIAGONAL_THROW_VZ_RANGE = (-0.12, 0.42)

SIDE_THROW_VX_RANGE = (0.95, 1.40)
SIDE_THROW_VY_RANGE = (0.08, 0.26)
SIDE_THROW_VZ_RANGE = (0.45, 0.90)

ROLLING_VX_RANGE = (0.85, 1.55)
ROLLING_VY_RANGE = (-0.12, 0.35)
ROLLING_VZ_RANGE = (-0.03, 0.03)
ROLLING_SPIN_NOISE = 0.8

PROJECTILE_ARC_VX_RANGE = (-0.30, 0.30)
PROJECTILE_ARC_VY_RANGE = (1.75, 2.95)
PROJECTILE_ARC_VZ_RANGE = (2.20, 3.80)

PROJECTILE_CROSS_VX_RANGE = (1.05, 1.85)
PROJECTILE_CROSS_VY_RANGE = (1.30, 2.20)
PROJECTILE_CROSS_VZ_RANGE = (2.00, 3.40)

SWING_DROP_VX_RANGE = (0.95, 1.85)
SWING_DROP_VY_RANGE = (0.10, 0.60)
SWING_DROP_VZ_RANGE = (-0.35, 0.20)
SWING_DROP_Z_RANGE = (0.95, 1.55)

STRIKE_SPEED_RANGE = (1.45, 2.25)

TOP_DROP_ANGVEL = 2.8
TOP_TOSS_ANGVEL = 4.2
FRONT_SLIDE_ANGVEL = 3.5
DIAGONAL_THROW_ANGVEL = 5.0
SIDE_THROW_ANGVEL = 4.0
STATIC_REST_ANGVEL = 0.0

USE_DATASET_MESH_OBJECTS = True
DATASET_OBJECT_PROB = 1.0   # 让 rigid_mix 场景尽量都用 SOURCE_DATASET_ROOTS 里的物体
MAX_ASSETS_PER_ROOT = None  # 不再只取前5个，真正把 bag / teddy_bear 都放进来

USE_TEXTURED_DATASET_MESH = True
N_BACKGROUND_PROPS_RANGE = (2, 5)
BACKGROUND_PANEL_Y = 1.10
BACKGROUND_SIDE_X = 1.05
BACKGROUND_Z_RANGE = (0.05, 0.55)


# =========================
# 通用工具
# =========================
def make_json_safe(x):
    if isinstance(x, dict):
        return {str(k): make_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [make_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def safe_scene_destroy(scene):
    if scene is not None:
        try:
            scene.destroy()
        except Exception:
            pass


def sample_background():
    presets = [
        {"name": "white_studio", "background_color": [1.0, 1.0, 1.0], "ambient_light": [0.36, 0.36, 0.36]},
        {"name": "light_gray_studio", "background_color": [0.92, 0.92, 0.92], "ambient_light": [0.32, 0.32, 0.32]},
        {"name": "sky_soft", "background_color": [0.80, 0.88, 1.00], "ambient_light": [0.30, 0.31, 0.35]},
        {"name": "warm_paper", "background_color": [0.98, 0.95, 0.90], "ambient_light": [0.34, 0.32, 0.30]},
        {"name": "dark_studio", "background_color": [0.10, 0.10, 0.12], "ambient_light": [0.20, 0.20, 0.20]},
        {"name": "mint", "background_color": [0.87, 0.95, 0.92], "ambient_light": [0.30, 0.32, 0.31]},
        {"name": "lavender", "background_color": [0.90, 0.88, 0.97], "ambient_light": [0.31, 0.30, 0.34]},
        {"name": "peach", "background_color": [0.99, 0.90, 0.84], "ambient_light": [0.34, 0.31, 0.29]},
    ]
    bg = random.choice(presets)
    bg["n_props"] = random.randint(*N_BACKGROUND_PROPS_RANGE)
    return bg


def sample_color(alpha=1.0):
    return [
        float(np.random.uniform(0.08, 0.95)),
        float(np.random.uniform(0.08, 0.95)),
        float(np.random.uniform(0.08, 0.95)),
        float(alpha),
    ]



def _indent_xml(elem, level=0):
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def _deterministic_rng_from_key(key: str):
    seed = 0
    for ch in str(key):
        seed = (seed * 131 + ord(ch)) % (2 ** 32 - 1)
    return np.random.RandomState(seed)


def pick_distinct_part_colors(n: int, key: str = ""):
    rng = _deterministic_rng_from_key(key)
    colors = []
    if n <= 0:
        return colors

    palette = [np.asarray(c, dtype=np.float32) for c in PART_COLOR_BANK]
    palette_len = len(palette)
    offset = int(rng.randint(0, palette_len))
    stride = 5 if math.gcd(5, palette_len) == 1 else 1
    for i in range(n):
        base_rgb = palette[(offset + i * stride) % palette_len].copy()
        cycle_idx = i // palette_len
        h, s, v = colorsys.rgb_to_hsv(float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2]))
        h = (h + 0.07 * cycle_idx + float(rng.uniform(-0.015, 0.015))) % 1.0
        s = float(np.clip(s + (0.06 if (i % 2) == 0 else -0.03), 0.60, 0.95))
        if cycle_idx == 0:
            v = float(np.clip(v + (0.10 if (i % 2) == 0 else -0.08), 0.45, 0.98))
        else:
            v = float(np.clip(v + (0.08 if (cycle_idx % 2) == 0 else -0.12), 0.40, 0.98))
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append([float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0])
    return colors


def sanitize_trimesh_preserve_scale(mesh: trimesh.Trimesh):
    mesh = mesh.copy()
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    return mesh


def write_multi_visual_single_collision_urdf(
    urdf_path: Path,
    robot_name: str,
    visual_mesh_paths,
    visual_colors,
    collision_mesh_path: Path,
    bbox_extents_m,
    mass_kg: float = 1.0,
):
    ensure_dir(urdf_path.parent)

    robot = ET.Element("robot", attrib={"name": robot_name})
    link = ET.SubElement(robot, "link", attrib={"name": "base_link"})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", attrib={"value": f"{float(max(mass_kg, 1e-4)):.8f}"})

    ex = [max(float(x), 1e-4) for x in bbox_extents_m]
    ixx = max(mass_kg * (ex[1] ** 2 + ex[2] ** 2) / 12.0, 1e-8)
    iyy = max(mass_kg * (ex[0] ** 2 + ex[2] ** 2) / 12.0, 1e-8)
    izz = max(mass_kg * (ex[0] ** 2 + ex[1] ** 2) / 12.0, 1e-8)
    ET.SubElement(
        inertial,
        "inertia",
        attrib={
            "ixx": f"{ixx:.8f}",
            "ixy": "0",
            "ixz": "0",
            "iyy": f"{iyy:.8f}",
            "iyz": "0",
            "izz": f"{izz:.8f}",
        },
    )

    for idx, (visual_mesh_path, rgba) in enumerate(zip(visual_mesh_paths, visual_colors)):
        visual_rel = Path(os.path.relpath(visual_mesh_path, urdf_path.parent)).as_posix()
        visual = ET.SubElement(link, "visual", attrib={"name": f"visual_part_{idx:03d}"})
        ET.SubElement(visual, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        vgeom = ET.SubElement(visual, "geometry")
        ET.SubElement(vgeom, "mesh", attrib={"filename": visual_rel, "scale": "1 1 1"})
        material = ET.SubElement(visual, "material", attrib={"name": f"part_color_{idx:03d}"})
        ET.SubElement(material, "color", attrib={"rgba": " ".join(f"{float(x):.6f}" for x in rgba)})

    collision_rel = Path(os.path.relpath(collision_mesh_path, urdf_path.parent)).as_posix()
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    cgeom = ET.SubElement(collision, "geometry")
    ET.SubElement(cgeom, "mesh", attrib={"filename": collision_rel, "scale": "1 1 1"})

    _indent_xml(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path, {
        "urdf_type": "single_rigid_multi_visual",
        "base_link": "base_link",
        "visual_meshes": [str(x) for x in visual_mesh_paths],
        "collision_mesh": str(collision_mesh_path),
        "bbox_extents_m": [float(x) for x in bbox_extents_m],
        "num_visual_parts": len(list(visual_mesh_paths)),
    }






def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def parse_density_to_kgm3(v):
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", "")
    try:
        return float(s)
    except Exception:
        pass
    if "g/cm" in s:
        num = safe_float(s.split()[0], None)
        return None if num is None else float(num) * 1000.0
    if "kg/m" in s:
        num = safe_float(s.split()[0], None)
        return None if num is None else float(num)
    return None


def infer_physx3d_friction(material_names):
    vals = []
    for name in material_names:
        s = str(name).lower()
        if "metal" in s:
            vals.append(0.25)
        elif "glass" in s:
            vals.append(0.18)
        elif "plastic" in s:
            vals.append(0.45)
        elif "wood" in s:
            vals.append(0.62)
        elif "rubber" in s:
            vals.append(1.00)
        elif "foam" in s:
            vals.append(0.78)
        elif "fabric" in s or "cloth" in s or "leather" in s:
            vals.append(0.72)
        elif "ceramic" in s or "stone" in s:
            vals.append(0.38)
    return float(np.mean(vals)) if len(vals) > 0 else None




def get_dataset_base_euler(dataset_name: str):
    base = np.zeros(3, dtype=np.float32)
    up_axis = str(DATASET_UP_AXIS_BY_DATASET.get(dataset_name, "y_up")).lower()
    if up_axis == "y_up":
        base += np.asarray(YUP_TO_ZUP_EULER_XYZ, dtype=np.float32)
    extra = DATASET_EXTRA_BASE_EULER_BY_DATASET.get(dataset_name, [0.0, 0.0, 0.0])
    base += np.asarray(extra, dtype=np.float32)
    return base


def get_dataset_up_axis(dataset_name: str):
    return str(DATASET_UP_AXIS_BY_DATASET.get(dataset_name, "y_up")).lower()


def convert_bbox_extents_to_scene_frame(extents, dataset_name: str):
    ext = np.asarray(extents, dtype=np.float64).reshape(3)
    if get_dataset_up_axis(dataset_name) == "y_up":
        return ext[[0, 2, 1]]
    return ext


def clamp_float(x, lo, hi):
    return float(max(lo, min(hi, x)))


def euler_xyz_to_matrix(euler_xyz):
    rx, ry, rz = [float(v) for v in euler_xyz]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def compute_vertical_half_extent(half_x: float, half_y: float, half_z: float, euler_xyz):
    R = np.abs(euler_xyz_to_matrix(euler_xyz))
    half_sizes = np.array([half_x, half_y, half_z], dtype=np.float32)
    return float(R[2].dot(half_sizes))


POSE_DELTA_ORTHO = [
    [0.0, 0.0, 0.0],
    [math.pi / 2.0, 0.0, 0.0],
    [-math.pi / 2.0, 0.0, 0.0],
    [0.0, math.pi / 2.0, 0.0],
    [0.0, -math.pi / 2.0, 0.0],
    [0.0, 0.0, math.pi / 2.0],
    [0.0, 0.0, -math.pi / 2.0],
]


def sample_pose_delta_for_mode(mode: str):
    # static_rest: 只采样较稳定的正交姿态 + 很小扰动，确保真正“落地静止”
    if mode == "static_rest":
        base = np.asarray(random.choice(POSE_DELTA_ORTHO), dtype=np.float32)
        jitter = np.array([
            np.random.uniform(-0.08, 0.08),
            np.random.uniform(-0.08, 0.08),
            np.random.uniform(-math.pi, math.pi),
        ], dtype=np.float32)
        return (base + jitter).tolist()

    # 动态物体：
    # 1) 一部分是比较明显的“竖直/侧立/横躺”姿态
    # 2) 一部分是倾斜姿态
    # 3) 一部分完全随机
    p = np.random.rand()
    if p < 0.38:
        base = np.asarray(random.choice(POSE_DELTA_ORTHO), dtype=np.float32)
        jitter = np.array([
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-math.pi, math.pi),
        ], dtype=np.float32)
        return (base + jitter).tolist()
    elif p < 0.75:
        base = np.asarray(random.choice(POSE_DELTA_ORTHO), dtype=np.float32)
        jitter = np.array([
            np.random.uniform(-0.45, 0.45),
            np.random.uniform(-0.45, 0.45),
            np.random.uniform(-math.pi, math.pi),
        ], dtype=np.float32)
        return (base + jitter).tolist()
    else:
        return [
            float(np.random.uniform(-math.pi, math.pi)),
            float(np.random.uniform(-math.pi, math.pi)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]


def sample_non_overlapping_xy(bound_r: float, occupied_slots=None, bias_to_back=False, max_trials=80):
    occupied_slots = occupied_slots if occupied_slots is not None else []
    for _ in range(max_trials):
        x, y = sample_spawn_xy(bound_r + 0.04, bound_r + 0.04, bias_to_back=bias_to_back)
        ok = True
        for ox, oy, orad in occupied_slots:
            if (x - ox) ** 2 + (y - oy) ** 2 < (bound_r + orad + 0.08) ** 2:
                ok = False
                break
        if ok:
            return x, y
    return sample_spawn_xy(bound_r + 0.02, bound_r + 0.02, bias_to_back=bias_to_back)


def register_spawn_slot(occupied_slots, x: float, y: float, bound_r: float):
    if occupied_slots is not None:
        occupied_slots.append((float(x), float(y), float(bound_r)))


def sample_targeted_velocity(start_xyz, target_xyz, speed_range, lateral_noise=0.08, z_noise=0.06):
    start_xyz = np.asarray(start_xyz, dtype=np.float32)
    target_xyz = np.asarray(target_xyz, dtype=np.float32).copy()
    target_xyz[0] += np.random.uniform(-lateral_noise, lateral_noise)
    target_xyz[1] += np.random.uniform(-lateral_noise, lateral_noise)
    target_xyz[2] += np.random.uniform(-z_noise, z_noise)

    delta = target_xyz - start_xyz
    norm = float(np.linalg.norm(delta))
    if norm < 1e-6:
        delta = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        norm = 1.0
    direction = delta / norm
    speed = float(np.random.uniform(*speed_range))
    return (direction * speed).astype(np.float32).tolist()


def sample_rigid_motion(
    half_x: float,
    half_y: float,
    half_z: float,
    bound_r: float = None,
    forced_mode: str = None,
    target_pos=None,
    occupied_slots=None,
    base_euler=None,
):
    """
    返回 rigid 物体的初始位置、初始姿态增量、线速度、角速度、运动类型。

    修正点：
    1) 所有 rigid 物体的初始位置都限制在三面体容器内部；
    2) 保留“撞击静止物体 / 前向进入 / 对角进入”等运动语义，但不再从容器外出生；
    3) 初始姿态仍允许任意 x / y / z 欧拉角。
    """
    mode = forced_mode or weighted_choice(RIGID_MOTION_WEIGHTS)

    hx = CONTAINER["half_x"]
    hy = CONTAINER["half_y"]
    wh = CONTAINER["wall_height"]
    ft = CONTAINER["floor_thickness"]
    wt = CONTAINER["wall_thickness"]

    if bound_r is None:
        bound_r = compute_bound_radius_from_half_extents(half_x, half_y, half_z)

    base_euler = np.asarray(base_euler if base_euler is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    pose_delta = sample_pose_delta_for_mode(mode)
    final_euler = base_euler + np.asarray(pose_delta, dtype=np.float32)
    vertical_half_extent = compute_vertical_half_extent(half_x, half_y, half_z, final_euler)
    safe_floor_z = ft + vertical_half_extent + 0.004
    init_pos = [0.0, 0.0, safe_floor_z]
    linvel = [0.0, 0.0, 0.0]
    angvel = [0.0, 0.0, 0.0]

    # 容器内部可出生范围：严格限制在三面体内部，不允许 x / y 初始在墙外。
    x_lo = -hx + wt + SPAWN_SIDE_KEEP_OUT + bound_r
    x_hi = +hx - wt - SPAWN_SIDE_KEEP_OUT - bound_r
    y_lo = -hy + wt + max(0.14, 0.45 * SPAWN_FRONT_KEEP_OUT) + bound_r
    y_hi = +hy - wt - SPAWN_BACK_KEEP_OUT - bound_r

    if x_lo >= x_hi:
        x_mid = 0.0
        x_lo, x_hi = x_mid - 0.03, x_mid + 0.03
    if y_lo >= y_hi:
        y_mid = 0.10
        y_lo, y_hi = y_mid - 0.03, y_mid + 0.03

    def _sample_front_band(side="center"):
        front_hi = min(y_hi, y_lo + max(0.22, min(0.48, 0.28 * (y_hi - y_lo) + 0.12)))
        y = float(np.random.uniform(y_lo, front_hi))

        width = x_hi - x_lo
        if side == "left":
            lx_hi = min(x_hi, x_lo + max(0.22, 0.32 * width))
            x = float(np.random.uniform(x_lo, lx_hi))
        elif side == "right":
            rx_lo = max(x_lo, x_hi - max(0.22, 0.32 * width))
            x = float(np.random.uniform(rx_lo, x_hi))
        else:
            cx_half = max(0.12, 0.20 * width)
            x = float(np.random.uniform(max(x_lo, -cx_half), min(x_hi, cx_half)))
        return x, y

    def _sample_side_band(side="left", y_center=None):
        width = x_hi - x_lo
        band_w = max(0.18, 0.22 * width)

        if side == "left":
            sx_lo = x_lo
            sx_hi = min(x_hi, x_lo + band_w)
        else:
            sx_lo = max(x_lo, x_hi - band_w)
            sx_hi = x_hi

        if y_center is None:
            sy_lo = y_lo + 0.05
            sy_hi = min(y_hi, y_lo + max(0.30, 0.42 * (y_hi - y_lo)))
        else:
            sy_lo = clamp_float(y_center - 0.12, y_lo, y_hi)
            sy_hi = clamp_float(y_center + 0.12, y_lo, y_hi)
            if sy_lo >= sy_hi:
                sy_lo, sy_hi = y_lo, min(y_hi, y_lo + 0.20)

        x = float(np.random.uniform(sx_lo, sx_hi))
        y = float(np.random.uniform(sy_lo, sy_hi))
        return x, y

    if mode == "static_rest":
        x, y = sample_non_overlapping_xy(bound_r, occupied_slots=occupied_slots, bias_to_back=False)
        y = clamp_float(y, 0.05, y_hi)
        init_pos = [float(x), float(y), float(safe_floor_z)]
        linvel = [0.0, 0.0, 0.0]
        angvel = [0.0, 0.0, 0.0]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode == "top_drop":
        x, y = sample_non_overlapping_xy(bound_r, occupied_slots=occupied_slots, bias_to_back=True)
        z = max(float(np.random.uniform(*TOP_DROP_Z_RANGE)), safe_floor_z + 0.35)
        init_pos = [float(x), float(y), float(z)]
        linvel = [
            float(np.random.uniform(-TOP_DROP_VXY, TOP_DROP_VXY)),
            float(np.random.uniform(-TOP_DROP_VXY, TOP_DROP_VXY)),
            float(np.random.uniform(-0.18, -0.03)),
        ]
        angvel = [
            float(np.random.uniform(-TOP_DROP_ANGVEL, TOP_DROP_ANGVEL)),
            float(np.random.uniform(-TOP_DROP_ANGVEL, TOP_DROP_ANGVEL)),
            float(np.random.uniform(-TOP_DROP_ANGVEL, TOP_DROP_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode == "top_toss":
        x, y = sample_non_overlapping_xy(bound_r, occupied_slots=occupied_slots, bias_to_back=True)
        z = max(float(np.random.uniform(*TOP_TOSS_Z_RANGE)), safe_floor_z + 0.32)
        init_pos = [float(x), float(y), float(z)]
        linvel = [
            float(np.random.uniform(-TOP_TOSS_VX, TOP_TOSS_VX)),
            float(np.random.uniform(-TOP_TOSS_VY, TOP_TOSS_VY)),
            float(np.random.uniform(*TOP_TOSS_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-TOP_TOSS_ANGVEL, TOP_TOSS_ANGVEL)),
            float(np.random.uniform(-TOP_TOSS_ANGVEL, TOP_TOSS_ANGVEL)),
            float(np.random.uniform(-TOP_TOSS_ANGVEL, TOP_TOSS_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode == "front_slide_in":
        x, y = _sample_front_band(side="center")
        z = max(float(np.random.uniform(*FRONT_SLIDE_Z_RANGE)), safe_floor_z + 0.01)
        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(*FRONT_SLIDE_VX_RANGE)),
            float(np.random.uniform(*FRONT_SLIDE_VY_RANGE)),
            float(np.random.uniform(*FRONT_SLIDE_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-FRONT_SLIDE_ANGVEL, FRONT_SLIDE_ANGVEL)),
            float(np.random.uniform(-FRONT_SLIDE_ANGVEL, FRONT_SLIDE_ANGVEL)),
            float(np.random.uniform(-FRONT_SLIDE_ANGVEL, FRONT_SLIDE_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode in {"diagonal_corner_left", "diagonal_corner_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_front_band(side=side)
        z = max(float(np.random.uniform(*DIAGONAL_ENTRY_Z_RANGE)), safe_floor_z + 0.28)
        init_pos = [x, y, z]
        vx = float(np.random.uniform(*DIAGONAL_THROW_VX_RANGE))
        vy = float(np.random.uniform(*DIAGONAL_THROW_VY_RANGE))
        if side == "left":
            vx = abs(vx)
        else:
            vx = -abs(vx)
        linvel = [
            vx,
            vy,
            float(np.random.uniform(*DIAGONAL_THROW_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode == "side_throw_left":
        x, y = _sample_side_band(side="left")
        z = max(float(np.random.uniform(*SIDE_THROW_Z_RANGE)), safe_floor_z + 0.20)
        init_pos = [x, y, z]
        linvel = [
            abs(float(np.random.uniform(*SIDE_THROW_VX_RANGE))),
            float(np.random.uniform(*SIDE_THROW_VY_RANGE)),
            float(np.random.uniform(*SIDE_THROW_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode == "side_throw_right":
        x, y = _sample_side_band(side="right")
        z = max(float(np.random.uniform(*SIDE_THROW_Z_RANGE)), safe_floor_z + 0.20)
        init_pos = [x, y, z]
        linvel = [
            -abs(float(np.random.uniform(*SIDE_THROW_VX_RANGE))),
            float(np.random.uniform(*SIDE_THROW_VY_RANGE)),
            float(np.random.uniform(*SIDE_THROW_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode in {"strike_static_left", "strike_static_right"}:
        if target_pos is None:
            tx, ty, tz = 0.0, 0.35, safe_floor_z
        else:
            tx, ty, tz = [float(v) for v in target_pos[:3]]

        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side, y_center=ty)
        z = clamp_float(
            tz + np.random.uniform(0.02, 0.20),
            safe_floor_z + 0.01,
            max(safe_floor_z + 0.02, wh - 0.18),
        )
        init_pos = [x, y, z]
        linvel = sample_targeted_velocity(init_pos, [tx, ty, tz + 0.03], STRIKE_SPEED_RANGE)
        angvel = [
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode in {"rolling_left", "rolling_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side, y_center=float(np.random.uniform(0.12, 0.62)))
        z = safe_floor_z + float(np.random.uniform(0.0, 0.015))
        init_pos = [x, y, z]

        vx_mag = float(np.random.uniform(*ROLLING_VX_RANGE))
        vx = vx_mag if side == "left" else -vx_mag
        vy = float(np.random.uniform(*ROLLING_VY_RANGE))
        vz = float(np.random.uniform(*ROLLING_VZ_RANGE))
        linvel = [vx, vy, vz]

        roll_radius = max(float(half_z), float(bound_r), 0.04)
        spin_y = -vx / roll_radius
        angvel = [
            float(np.random.uniform(-0.4, 0.4)),
            float(spin_y + np.random.uniform(-ROLLING_SPIN_NOISE, ROLLING_SPIN_NOISE)),
            float(np.random.uniform(-0.4, 0.4)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode == "projectile_arc_forward":
        x, y = _sample_front_band(side="center")
        z = safe_floor_z + float(np.random.uniform(0.03, 0.10))
        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(*PROJECTILE_ARC_VX_RANGE)),
            float(np.random.uniform(*PROJECTILE_ARC_VY_RANGE)),
            float(np.random.uniform(*PROJECTILE_ARC_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-TOP_TOSS_ANGVEL, TOP_TOSS_ANGVEL)),
            float(np.random.uniform(-TOP_TOSS_ANGVEL, TOP_TOSS_ANGVEL)),
            float(np.random.uniform(-TOP_TOSS_ANGVEL, TOP_TOSS_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode in {"projectile_cross_left", "projectile_cross_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side, y_center=float(np.random.uniform(0.0, 0.35)))
        z = safe_floor_z + float(np.random.uniform(0.02, 0.08))
        init_pos = [x, y, z]

        vx = float(np.random.uniform(*PROJECTILE_CROSS_VX_RANGE))
        if side == "right":
            vx = -vx
        linvel = [
            vx,
            float(np.random.uniform(*PROJECTILE_CROSS_VY_RANGE)),
            float(np.random.uniform(*PROJECTILE_CROSS_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    elif mode in {"swing_drop_left", "swing_drop_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side, y_center=float(np.random.uniform(0.10, 0.55)))
        z = max(float(np.random.uniform(*SWING_DROP_Z_RANGE)), safe_floor_z + 0.42)
        init_pos = [x, y, z]

        vx = float(np.random.uniform(*SWING_DROP_VX_RANGE))
        if side == "right":
            vx = -vx
        linvel = [
            vx,
            float(np.random.uniform(*SWING_DROP_VY_RANGE)),
            float(np.random.uniform(*SWING_DROP_VZ_RANGE)),
        ]
        angvel = [
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
            float(np.random.uniform(-DIAGONAL_THROW_ANGVEL, DIAGONAL_THROW_ANGVEL)),
        ]
        register_spawn_slot(occupied_slots, x, y, bound_r)

    else:
        raise ValueError(mode)

    return {
        "motion_type": mode,
        "init_pos": init_pos,
        "pose_delta": pose_delta,
        "init_linvel": linvel,
        "init_angvel": angvel,
    }



def _sample_color_in_range(lo_rgb, hi_rgb, alpha=1.0):
    return [
        float(np.random.uniform(lo_rgb[0], hi_rgb[0])),
        float(np.random.uniform(lo_rgb[1], hi_rgb[1])),
        float(np.random.uniform(lo_rgb[2], hi_rgb[2])),
        float(alpha),
    ]


def _sample_primitive_dims(shape: str):
    if shape == "box":
        sx = float(np.random.uniform(0.10, 0.32))
        sy = float(np.random.uniform(0.10, 0.32))
        sz = float(np.random.uniform(0.10, 0.32))
        bbox = [sx, sy, sz]
        geom_template = {
            "shape": "box",
            "size": [sx, sy, sz],
        }
        return geom_template, bbox

    if shape == "sphere":
        r = float(np.random.uniform(0.06, 0.16))
        bbox = [2 * r, 2 * r, 2 * r]
        geom_template = {
            "shape": "sphere",
            "radius": r,
        }
        return geom_template, bbox

    if shape == "cylinder":
        r = float(np.random.uniform(0.05, 0.14))
        h = float(np.random.uniform(0.12, 0.34))
        bbox = [2 * r, 2 * r, h]
        geom_template = {
            "shape": "cylinder",
            "radius": r,
            "height": h,
        }
        return geom_template, bbox

    if shape == "capsule":
        r = float(np.random.uniform(0.04, 0.11))
        h = float(np.random.uniform(0.12, 0.30))
        bbox = [2 * r, 2 * r, h + 2 * r]
        geom_template = {
            "shape": "capsule",
            "radius": r,
            "height": h,
        }
        return geom_template, bbox

    raise ValueError(shape)


def build_primitive_asset_bank():
    bank = []

    for idx in range(PRIMITIVE_ASSET_REPEAT):
        shape = weighted_choice(PRIMITIVE_SHAPE_WEIGHTS)
        mat_name = weighted_choice(PRIMITIVE_MATERIAL_WEIGHTS)
        mat_cfg = PRIMITIVE_MATERIAL_PRESETS[mat_name]

        geom_template, bbox_extents = _sample_primitive_dims(shape)
        color = _sample_color_in_range(
            mat_cfg["color_range"][0],
            mat_cfg["color_range"][1],
            alpha=1.0,
        )

        asset_id = f"{PRIMITIVE_DATASET_NAME}__{shape}__{mat_name}__{idx:04d}"

        bank.append({
            "asset_id": asset_id,
            "dataset_name": PRIMITIVE_DATASET_NAME,
            "dataset_root": "builtin",
            "sample_dir": f"builtin://{asset_id}",
            "mesh_path": None,
            "mat_json": None,
            "ply_path": None,
            "unit_mesh_path": None,
            "render_mesh_path": None,
            "render_urdf_path": None,
            "raw_bbox_extents": list(bbox_extents),
            "unit_bbox_extents": list(bbox_extents),
            "n_vertices": None,
            "n_faces": None,
            "has_texture": False,
            "primitive_geom_template": geom_template,
            "primitive_color": color,
            "material_override": {
                "family": "Rigid",
                "name": f"primitive_{mat_name}",
                "rho": float(mat_cfg["rho"]),
                "friction": float(mat_cfg["friction"]),
                "restitution": float(mat_cfg["restitution"]),
            },
            "primitive_material_name": mat_name,
            "primitive_shape_name": shape,
        })

    print(f"[INFO] usable primitive assets: {len(bank)}")
    return bank


# =========================
# asset 扫描与预处理
# =========================
def find_candidate_mesh(sample_dir: Path):
    p = sample_dir / "material.obj"
    if p.exists():
        return p

    obj_files = sorted(sample_dir.glob("*.obj"))
    if len(obj_files) == 1:
        return obj_files[0]

    if len(obj_files) > 1:
        for x in obj_files:
            if x.name.lower() == "material.obj":
                return x
        return obj_files[0]

    return None


def try_find_material_json(sample_dir: Path):
    candidates = [
        "mat_params_new_v3.4.json",
        "mat_params_new.json",
        "mat_params.json",
        "material_params.json",
        "material.json",
    ]
    for name in candidates:
        p = sample_dir / name
        if p.exists():
            return p
    return None



def _merge_trimesh_list(mesh_paths, out_path: Path):
    ensure_dir(out_path.parent)
    if out_path.exists():
        return out_path

    geoms = []
    for mp in mesh_paths:
        try:
            obj = trimesh.load(mp, process=False)
            if isinstance(obj, trimesh.Scene):
                sub_geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
                geoms.extend(sub_geoms)
            elif isinstance(obj, trimesh.Trimesh):
                geoms.append(obj)
        except Exception:
            continue

    if len(geoms) == 0:
        raise ValueError(f"No valid trimesh geometry found in: {mesh_paths}")

    merged = trimesh.util.concatenate(geoms)
    merged.export(out_path)
    return out_path


def _list_physx3d_object_ids():
    finaljson_dir = PHYSX3D_ROOT / PHYSX3D_VERSION / "finaljson"
    if not finaljson_dir.exists():
        print(f"[WARN] PhysX-3D finaljson dir not found: {finaljson_dir}")
        return []

    if PHYSX3D_OBJECT_IDS:
        ids = [str(x) for x in PHYSX3D_OBJECT_IDS]
    else:
        ids = sorted([p.stem for p in finaljson_dir.glob("*.json")])

    if PHYSX3D_MAX_OBJECTS not in (None, 0):
        ids = ids[:int(PHYSX3D_MAX_OBJECTS)]
    return ids


def _build_physx3d_material_override(meta_json_path: Path):
    default_mat = sample_rigid_material()
    try:
        with open(meta_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default_mat

    parts = data.get("parts", [])
    densities = []
    material_names = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if PHYSX3D_USE_PART_DENSITY:
            rho = parse_density_to_kgm3(part.get("density"))
            if rho is not None and rho > 0:
                densities.append(float(rho))
        material_names.append(str(part.get("material", "")))

    rho = float(np.mean(densities)) if len(densities) > 0 else default_mat["rho"]
    friction = infer_physx3d_friction(material_names)
    if friction is None:
        friction = default_mat["friction"]

    obj_name = str(data.get("object_name", meta_json_path.stem))
    return {
        "family": "Rigid",
        "name": f"physx3d_{obj_name}",
        "rho": float(rho),
        "friction": float(np.clip(friction, 1e-2, 5.0)),
        "restitution": 0.10,
    }



def build_physx3d_asset_bank():
    base_dir = PHYSX3D_ROOT / PHYSX3D_VERSION
    finaljson_dir = base_dir / "finaljson"
    partseg_dir = base_dir / "partseg"

    if not finaljson_dir.exists() or not partseg_dir.exists():
        print(f"[WARN] PhysX-3D dirs missing under: {base_dir}")
        return []

    bank = []
    failed = []

    for obj_id in _list_physx3d_object_ids():
        try:
            meta_json = finaljson_dir / f"{obj_id}.json"
            objs_dir = partseg_dir / obj_id / "objs"
            if not meta_json.exists() or not objs_dir.exists():
                raise FileNotFoundError(f"missing meta or objs dir for {obj_id}")

            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)

            object_name = str(meta.get("object_name", obj_id))
            category = str(meta.get("category", "unknown"))
            parts_meta = meta.get("parts", [])

            part_infos = []
            for part in sorted(parts_meta, key=lambda x: int(x.get("label", 0))):
                part_id = int(part.get("label", len(part_infos)))
                mesh_path = objs_dir / f"{part_id}.obj"
                if not mesh_path.exists():
                    continue
                part_infos.append({
                    "part_id": part_id,
                    "name": str(part.get("name", f"part_{part_id}")),
                    "material_name": str(part.get("material", "")),
                    "mesh_path": str(mesh_path),
                })

            if len(part_infos) == 0:
                for idx, mesh_path in enumerate(sorted(objs_dir.glob("*.obj"))):
                    part_infos.append({
                        "part_id": idx,
                        "name": mesh_path.stem,
                        "material_name": "",
                        "mesh_path": str(mesh_path),
                    })

            if len(part_infos) == 0:
                raise FileNotFoundError(f"no part meshes for {obj_id}")

            raw_part_mesh_paths = [Path(x["mesh_path"]) for x in part_infos]
            merged_mesh_path = PHYSX3D_MERGED_CACHE_DIR / obj_id / "merged_raw.obj"
            _merge_trimesh_list(raw_part_mesh_paths, merged_mesh_path)

            bank.append({
                "asset_id": f"physx3d__{obj_id}",
                "dataset_name": "physx3d",
                "dataset_root": str(PHYSX3D_ROOT),
                "sample_dir": str(objs_dir),
                "mesh_path": str(merged_mesh_path),
                "mat_json": None,
                "ply_path": None,
                "meta_json": str(meta_json),
                "physx3d_object_id": str(obj_id),
                "physx3d_object_name": object_name,
                "physx3d_category": category,
                "physx3d_part_infos": part_infos,
                "material_override": _build_physx3d_material_override(meta_json),
                "has_texture": False,
            })
        except Exception as e:
            failed.append({"object_id": obj_id, "error": str(e)})
            print(f"[WARN] skip physx3d asset {obj_id}: {e}")

    print(f"[INFO] usable PhysX-3D assets: {len(bank)} | failed: {len(failed)}")
    return bank


def prepare_physx3d_asset_cache(asset):
    obj_id = str(asset["physx3d_object_id"])
    cache_dir = PHYSX3D_MERGED_CACHE_DIR / obj_id
    unit_parts_dir = cache_dir / "unit_parts"
    unit_mesh_path = cache_dir / "merged_unit.obj"
    cache_json = cache_dir / "unit_meta.json"
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in f"{asset.get('physx3d_object_name', obj_id)}_{obj_id}")
    urdf_path = PHYSX3D_PART_URDF_CACHE_DIR / obj_id / f"{safe_name}.urdf"

    if cache_json.exists():
        try:
            meta = json.loads(cache_json.read_text(encoding="utf-8"))
            if (
                int(meta.get("part_color_style_version", 0)) == PART_COLOR_STYLE_VERSION
                and Path(meta["unit_mesh_path"]).exists()
                and Path(meta["render_urdf_path"]).exists()
            ):
                asset["unit_mesh_path"] = meta["unit_mesh_path"]
                asset["render_mesh_path"] = meta["unit_mesh_path"]
                asset["render_urdf_path"] = meta["render_urdf_path"]
                raw_bbox_scene = meta.get(
                    "raw_bbox_extents_scene",
                    convert_bbox_extents_to_scene_frame(meta["raw_bbox_extents"], asset["dataset_name"]),
                )
                unit_bbox_scene = meta.get(
                    "unit_bbox_extents_scene",
                    convert_bbox_extents_to_scene_frame(meta["unit_bbox_extents"], asset["dataset_name"]),
                )
                asset["raw_bbox_extents_local"] = meta["raw_bbox_extents"]
                asset["unit_bbox_extents_local"] = meta["unit_bbox_extents"]
                asset["raw_bbox_extents"] = np.asarray(raw_bbox_scene, dtype=np.float64).tolist()
                asset["unit_bbox_extents"] = np.asarray(unit_bbox_scene, dtype=np.float64).tolist()
                asset["n_vertices"] = meta.get("n_vertices", None)
                asset["n_faces"] = meta.get("n_faces", None)
                asset["has_texture"] = False
                asset["physx3d_unit_part_mesh_paths"] = meta.get("unit_part_mesh_paths", [])
                asset["physx3d_part_colors"] = meta.get("part_colors", [])
                asset["physx3d_part_materials"] = meta.get("part_materials", [])
                asset["coordinate_transform"] = meta.get(
                    "coordinate_transform",
                    {
                        "source_up_axis": get_dataset_up_axis(asset["dataset_name"]),
                        "target_up_axis": "z_up",
                        "base_euler_xyz_rad": get_dataset_base_euler(asset["dataset_name"]).tolist(),
                    },
                )
                return asset
        except Exception:
            pass

    ensure_dir(unit_parts_dir)
    ensure_dir(urdf_path.parent)

    valid_part_records = []
    merged_geoms = []

    for part in asset.get("physx3d_part_infos", []):
        try:
            mesh = sanitize_trimesh_preserve_scale(load_trimesh_any(Path(part["mesh_path"])))
            extents = np.asarray(mesh.extents, dtype=np.float64)
            if np.any(~np.isfinite(extents)) or float(np.max(extents)) < MIN_VALID_MESH_EXTENT:
                continue
            valid_part_records.append({
                "part_id": int(part["part_id"]),
                "name": str(part.get("name", f"part_{part['part_id']}")),
                "material_name": str(part.get("material_name", "")),
                "mesh": mesh,
            })
            merged_geoms.append(mesh.copy())
        except Exception:
            continue

    if len(valid_part_records) == 0:
        raise ValueError(f"No valid PhysX-3D parts remain after load/sanitize: {obj_id}")

    merged_raw = trimesh.util.concatenate(merged_geoms)
    raw_extents = np.asarray(merged_raw.extents, dtype=np.float64)
    if np.any(~np.isfinite(raw_extents)) or float(np.max(raw_extents)) < MIN_VALID_MESH_EXTENT:
        raise ValueError(f"Invalid merged raw extents for PhysX-3D object {obj_id}: {raw_extents}")

    center = np.asarray(merged_raw.bounding_box.centroid, dtype=np.float64)
    unit_scale = 1.0 / max(float(np.max(raw_extents)), 1e-8)

    part_colors = pick_distinct_part_colors(len(valid_part_records), key=obj_id)
    unit_part_mesh_paths = []
    unit_geoms = []

    for rec in valid_part_records:
        mesh = rec["mesh"].copy()
        mesh.apply_translation(-center)
        mesh.apply_scale(unit_scale)
        out_path = unit_parts_dir / f"part_{rec['part_id']:03d}.obj"
        mesh.export(out_path)
        unit_part_mesh_paths.append(str(out_path))
        unit_geoms.append(mesh)

    merged_unit = trimesh.util.concatenate(unit_geoms)
    merged_unit.export(unit_mesh_path)
    unit_extents = np.asarray(merged_unit.extents, dtype=np.float64)

    rho = float(asset["material_override"].get("rho", 1000.0)) if asset.get("material_override") else 1000.0
    est_mass = float(max(np.prod(np.maximum(unit_extents, 1e-4)) * rho, 1e-4))

    write_multi_visual_single_collision_urdf(
        urdf_path=urdf_path,
        robot_name=safe_name,
        visual_mesh_paths=[Path(x) for x in unit_part_mesh_paths],
        visual_colors=part_colors,
        collision_mesh_path=unit_mesh_path,
        bbox_extents_m=unit_extents.tolist(),
        mass_kg=est_mass,
    )

    raw_extents_scene = convert_bbox_extents_to_scene_frame(raw_extents, asset["dataset_name"])
    unit_extents_scene = convert_bbox_extents_to_scene_frame(unit_extents, asset["dataset_name"])

    meta = {
        "asset_id": asset["asset_id"],
        "unit_mesh_path": str(unit_mesh_path),
        "render_urdf_path": str(urdf_path),
        "raw_bbox_extents": raw_extents.tolist(),
        "unit_bbox_extents": unit_extents.tolist(),
        "raw_bbox_extents_scene": raw_extents_scene.tolist(),
        "unit_bbox_extents_scene": unit_extents_scene.tolist(),
        "n_vertices": int(len(merged_unit.vertices)),
        "n_faces": int(len(merged_unit.faces)),
        "unit_part_mesh_paths": unit_part_mesh_paths,
        "part_colors": part_colors,
        "part_color_style_version": PART_COLOR_STYLE_VERSION,
        "coordinate_transform": {
            "source_up_axis": get_dataset_up_axis(asset["dataset_name"]),
            "target_up_axis": "z_up",
            "base_euler_xyz_rad": get_dataset_base_euler(asset["dataset_name"]).tolist(),
        },
        "part_materials": [
            {
                "part_id": int(rec["part_id"]),
                "name": str(rec["name"]),
                "material_name": str(rec["material_name"]),
            }
            for rec in valid_part_records
        ],
    }
    cache_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    asset["unit_mesh_path"] = str(unit_mesh_path)
    asset["render_mesh_path"] = str(unit_mesh_path)
    asset["render_urdf_path"] = str(urdf_path)
    asset["raw_bbox_extents_local"] = raw_extents.tolist()
    asset["unit_bbox_extents_local"] = unit_extents.tolist()
    asset["raw_bbox_extents"] = raw_extents_scene.tolist()
    asset["unit_bbox_extents"] = unit_extents_scene.tolist()
    asset["n_vertices"] = int(len(merged_unit.vertices))
    asset["n_faces"] = int(len(merged_unit.faces))
    asset["has_texture"] = False
    asset["physx3d_unit_part_mesh_paths"] = unit_part_mesh_paths
    asset["physx3d_part_colors"] = part_colors
    asset["physx3d_part_materials"] = meta["part_materials"]
    asset["coordinate_transform"] = meta["coordinate_transform"]
    return asset

def load_trimesh_any(mesh_path: Path):
    obj = trimesh.load(mesh_path, process=False)

    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        raise ValueError(f"Unsupported mesh type: {type(obj)}")

    return mesh


def maybe_simplify_mesh(mesh: trimesh.Trimesh):
    if SIMPLIFY_MESH_FACE_COUNT is None:
        return mesh
    try:
        if len(mesh.faces) > SIMPLIFY_MESH_FACE_COUNT:
            mesh = mesh.simplify_quadric_decimation(SIMPLIFY_MESH_FACE_COUNT)
    except Exception:
        pass
    return mesh


def sanitize_mesh(mesh: trimesh.Trimesh):
    mesh = mesh.copy()

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.merge_vertices()
    except Exception:
        pass

    mesh = maybe_simplify_mesh(mesh)

    extents = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(~np.isfinite(extents)) or float(np.max(extents)) < MIN_VALID_MESH_EXTENT:
        raise ValueError(f"Invalid mesh extents: {extents}")

    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    mesh.apply_translation(-center)

    extents = np.asarray(mesh.extents, dtype=np.float64)
    scale = 1.0 / max(float(np.max(extents)), 1e-8)
    mesh.apply_scale(scale)

    unit_extents = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(~np.isfinite(unit_extents)) or float(np.max(unit_extents)) < MIN_VALID_MESH_EXTENT:
        raise ValueError(f"Invalid unit extents after normalize: {unit_extents}")

    return mesh, extents, unit_extents, scale


def build_asset_id(root: Path, sample_dir: Path):
    rel = sample_dir.relative_to(root)
    rel_str = "__".join(rel.parts)
    return f"{root.name}__{rel_str}"


def find_asset_dirs_from_roots(asset_roots):
    assets = []

    for root in asset_roots:
        if not root.exists():
            print(f"[WARN] asset root not found: {root}")
            continue

        sample_dirs = sorted({p.parent for p in root.rglob("material.obj")})

        if len(sample_dirs) == 0:
            sample_dirs = sorted({p.parent for p in root.rglob("*.obj")})

        if MAX_ASSETS_PER_ROOT is not None:
            sample_dirs = sample_dirs[:MAX_ASSETS_PER_ROOT]

        print(f"[INFO] scanning root={root} | found candidate dirs={len(sample_dirs)}")

        for d in sample_dirs:
            mesh_path = find_candidate_mesh(d)
            if mesh_path is None:
                continue

            mat_json = try_find_material_json(d)
            ply_path = d / "sampled_points.ply"
            if not ply_path.exists():
                ply_path = None

            asset_id = build_asset_id(root, d)

            assets.append({
                "asset_id": asset_id,
                "dataset_name": root.name,
                "dataset_root": str(root),
                "sample_dir": str(d),
                "mesh_path": str(mesh_path),
                "mat_json": str(mat_json) if mat_json is not None else None,
                "ply_path": str(ply_path) if ply_path is not None else None,
            })

    return assets


def _write_obj_mesh_rigid(out_path: Path, vertices: List[List[float]], faces: List[List[int]]) -> None:
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for face in faces:
        lines.append("f " + " ".join(str(int(i)) for i in face) + "\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _extract_obj_submeshes_by_material_rigid(obj_path: Path, out_dir: Path) -> Dict[str, Path]:
    vertices: List[List[float]] = []
    faces_by_material: Dict[str, List[List[int]]] = {}
    current_material = "material_0"

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if line.startswith("usemtl "):
                current_material = line.strip().split(maxsplit=1)[1]
                faces_by_material.setdefault(current_material, [])
                continue

            if not line.startswith("f "):
                continue

            tokens = line.strip().split()[1:]
            face_idx: List[int] = []
            for tok in tokens:
                idx_str = tok.split("/")[0]
                if not idx_str:
                    continue
                idx = int(idx_str)
                if idx < 0:
                    idx = len(vertices) + idx + 1
                face_idx.append(idx)

            if len(face_idx) < 3:
                continue

            tri_faces = []
            if len(face_idx) == 3:
                tri_faces.append(face_idx)
            else:
                for j in range(1, len(face_idx) - 1):
                    tri_faces.append([face_idx[0], face_idx[j], face_idx[j + 1]])
            faces_by_material.setdefault(current_material, []).extend(tri_faces)

    ensure_dir(out_dir)
    out_paths: Dict[str, Path] = {}
    safe_map = str.maketrans({c: "_" for c in '/\\:*?"<>|'})
    for material_name, faces in faces_by_material.items():
        used = sorted({idx for face in faces for idx in face})
        if not used:
            continue
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used, start=1)}
        sub_vertices = [vertices[old_idx - 1] for old_idx in used]
        sub_faces = [[remap[idx] for idx in face] for face in faces]
        slot_key = material_name.translate(safe_map)
        out_path = out_dir / f"{slot_key}.obj"
        _write_obj_mesh_rigid(out_path, sub_vertices, sub_faces)
        out_paths[material_name] = out_path
    return out_paths


def _mesh_volume_safe(m: trimesh.Trimesh) -> float:
    try:
        if m.is_watertight:
            v = float(m.volume)
            if v > 1e-12:
                return v
    except Exception:
        pass
    try:
        return float(m.convex_hull.volume)
    except Exception:
        ex = np.maximum(np.asarray(m.extents, dtype=np.float64), 1e-6)
        return float(ex[0] * ex[1] * ex[2])


def _box_inertia_diag(mass: float, extents: np.ndarray) -> Tuple[float, float, float]:
    ex, ey, ez = [max(float(x), 1e-4) for x in np.asarray(extents).reshape(3).tolist()]
    ixx = max(mass * (ey ** 2 + ez ** 2) / 12.0, 1e-8)
    iyy = max(mass * (ex ** 2 + ez ** 2) / 12.0, 1e-8)
    izz = max(mass * (ex ** 2 + ey ** 2) / 12.0, 1e-8)
    return ixx, iyy, izz


def write_sophy_multipart_fixed_chain_urdf(urdf_path: Path, robot_name: str, part_entries: List[dict]) -> Path:
    ensure_dir(urdf_path.parent)
    robot = ET.Element("robot", attrib={"name": robot_name})
    part_entries = list(part_entries)
    for i, pe in enumerate(part_entries):
        link = ET.SubElement(robot, "link", attrib={"name": pe["link_name"]})
        com = pe["com"]
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(
            inertial,
            "origin",
            attrib={"xyz": f"{float(com[0]):.8f} {float(com[1]):.8f} {float(com[2]):.8f}", "rpy": "0 0 0"},
        )
        ET.SubElement(inertial, "mass", attrib={"value": f"{float(pe['mass']):.8f}"})
        ET.SubElement(
            inertial,
            "inertia",
            attrib={
                "ixx": f"{float(pe['ixx']):.8e}",
                "ixy": "0",
                "ixz": "0",
                "iyy": f"{float(pe['iyy']):.8e}",
                "iyz": "0",
                "izz": f"{float(pe['izz']):.8e}",
            },
        )

        visual = ET.SubElement(link, "visual", attrib={"name": "visual"})
        ET.SubElement(visual, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        vg = ET.SubElement(visual, "geometry")
        ET.SubElement(vg, "mesh", attrib={"filename": str(pe["rel_mesh"]), "scale": "1 1 1"})
        mat = ET.SubElement(visual, "material", attrib={"name": f"sophy_mat_{i:03d}"})
        rgba = pe["rgba"]
        ET.SubElement(mat, "color", attrib={"rgba": " ".join(f"{float(x):.6f}" for x in rgba)})

        collision = ET.SubElement(link, "collision", attrib={"name": "collision"})
        ET.SubElement(collision, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        cg = ET.SubElement(collision, "geometry")
        ET.SubElement(cg, "mesh", attrib={"filename": str(pe["rel_mesh"]), "scale": "1 1 1"})

    for i in range(1, len(part_entries)):
        j_el = ET.SubElement(
            robot,
            "joint",
            attrib={"name": f"sophy_fixed_{i - 1:03d}_to_{i:03d}", "type": "fixed"},
        )
        ET.SubElement(j_el, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(j_el, "parent", attrib={"link": part_entries[i - 1]["link_name"]})
        ET.SubElement(j_el, "child", attrib={"link": part_entries[i]["link_name"]})

    _indent_xml(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path


def infer_friction_from_part_info(part_info):
    mat_name = str(part_info.get("mat_name", "")).lower()
    mat_sub = str(part_info.get("mat_sub_type", "")).lower()

    if "metal" in mat_name:
        return 0.25
    if "fabric" in mat_name or "polyester" in mat_sub:
        return 0.75
    if "plastic" in mat_name or "polyamide" in mat_sub or "polypropylene" in mat_sub:
        return 0.45
    if "leather" in mat_name or "leather" in mat_sub:
        return 0.50
    return None


def _hydrate_sophy_multipart_asset_from_meta(asset: dict, meta: dict, cache_obj: Path) -> None:
    asset["unit_mesh_path"] = str(cache_obj)
    asset["render_mesh_path"] = meta.get("render_mesh_path", asset["mesh_path"])
    asset["render_urdf_path"] = str(meta["render_urdf_path"])
    asset["sophy_multipart_urdf"] = True
    asset["sophy_part_colors"] = meta.get("sophy_part_colors", [])
    asset["sophy_part_material_records"] = meta.get("sophy_part_material_records", [])
    asset["sophy_part_rigid_specs"] = meta.get("sophy_part_rigid_specs", [])
    asset["material_override"] = dict(meta["sophy_composite_rigid_material"])
    raw_bbox_scene = meta.get(
        "raw_bbox_extents_scene",
        convert_bbox_extents_to_scene_frame(meta["raw_bbox_extents"], asset["dataset_name"]),
    )
    unit_bbox_scene = meta.get(
        "unit_bbox_extents_scene",
        convert_bbox_extents_to_scene_frame(meta["unit_bbox_extents"], asset["dataset_name"]),
    )
    asset["raw_bbox_extents_local"] = meta["raw_bbox_extents"]
    asset["unit_bbox_extents_local"] = meta["unit_bbox_extents"]
    asset["raw_bbox_extents"] = np.asarray(raw_bbox_scene, dtype=np.float64).tolist()
    asset["unit_bbox_extents"] = np.asarray(unit_bbox_scene, dtype=np.float64).tolist()
    asset["n_vertices"] = meta.get("n_vertices", None)
    asset["n_faces"] = meta.get("n_faces", None)
    asset["has_texture"] = bool(meta.get("has_texture", False))
    asset["coordinate_transform"] = meta.get(
        "coordinate_transform",
        {
            "source_up_axis": get_dataset_up_axis(asset["dataset_name"]),
            "target_up_axis": "z_up",
            "base_euler_xyz_rad": get_dataset_base_euler(asset["dataset_name"]).tolist(),
        },
    )


def build_sophy_multipart_rigid_asset(asset: dict) -> dict:
    """SOPHY 资产：部件级 rho（质量）+ 推断摩擦；数据集 E/nu/sigma_y 写入 sophy_part_rigid_specs。"""
    asset_id = asset["asset_id"]
    mat_json_path = Path(asset["mat_json"])
    mesh_path = Path(asset["mesh_path"])
    if not mat_json_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError("mat_json or mesh missing for SOPHY multipart rigid")

    cache_dir = SOPHY_PART_URDF_CACHE_DIR / asset_id
    parts_dir = cache_dir / "unit_parts"
    submesh_dir = cache_dir / "_submesh_raw"
    ensure_dir(parts_dir)
    ensure_dir(submesh_dir)

    mat_params = json.loads(mat_json_path.read_text(encoding="utf-8"))
    material_items = [(str(k), dict(v)) for k, v in mat_params.items() if isinstance(v, dict)]

    submeshes = _extract_obj_submeshes_by_material_rigid(mesh_path, submesh_dir)
    if len(submeshes) == 0:
        raise ValueError("no submeshes (usemtl groups) from mesh")

    ordered_submeshes = sorted(
        submeshes.items(),
        key=lambda kv: int(kv[0].split("_")[-1]) if kv[0].split("_")[-1].isdigit() else kv[0],
    )
    n_pair = min(len(ordered_submeshes), len(material_items))
    if n_pair < len(ordered_submeshes) or n_pair < len(material_items):
        print(
            f"[WARN] SOPHY rigid part pairing asset={asset_id}: "
            f"submeshes={len(ordered_submeshes)} mat_params_parts={len(material_items)} -> using {n_pair}"
        )

    default_mat = sample_rigid_material()
    pair_geoms: List[trimesh.Trimesh] = []
    pair_meta: List[dict] = []

    for i in range(n_pair):
        _, sub_path = ordered_submeshes[i]
        material_name, cfg = material_items[i]
        mesh = sanitize_trimesh_preserve_scale(load_trimesh_any(Path(sub_path)))
        if float(np.max(mesh.extents)) < MIN_VALID_MESH_EXTENT:
            continue
        fr = infer_friction_from_part_info(cfg)
        if fr is None:
            fr = float(default_mat["friction"])
        fr = float(np.clip(fr, 1e-2, 5.0))
        rho = float(cfg.get("rho", default_mat["rho"]))
        pair_geoms.append(mesh)
        pair_meta.append(
            {
                "slot": ordered_submeshes[i][0],
                "material_part_name": material_name,
                "cfg": cfg,
                "rho": rho,
                "friction": fr,
            }
        )

    if len(pair_geoms) == 0:
        raise ValueError("no valid SOPHY parts after mesh load")

    merged_raw = trimesh.util.concatenate([g.copy() for g in pair_geoms])
    center = np.asarray(merged_raw.bounding_box.centroid, dtype=np.float64)
    raw_extents = np.asarray(merged_raw.extents, dtype=np.float64)
    unit_scale_factor = 1.0 / max(float(np.max(raw_extents)), 1e-8)

    part_entries_urdf: List[dict] = []
    sophy_part_rigid_specs: List[dict] = []
    sophy_part_material_records: List[dict] = []
    masses: List[float] = []
    volumes: List[float] = []
    colors = pick_distinct_part_colors(len(pair_geoms), key=asset_id)

    unit_geoms: List[trimesh.Trimesh] = []
    for i, (mesh_raw, meta) in enumerate(zip(pair_geoms, pair_meta)):
        m = mesh_raw.copy()
        m.apply_translation(-center)
        m.apply_scale(unit_scale_factor)
        vol = _mesh_volume_safe(m)
        mass = max(float(meta["rho"]) * vol, 1e-6)
        masses.append(mass)
        volumes.append(vol)
        com = np.asarray(m.center_mass, dtype=np.float64)
        ext = np.maximum(np.asarray(m.extents, dtype=np.float64), 1e-6)
        ixx, iyy, izz = _box_inertia_diag(mass, ext)
        out_obj = parts_dir / f"part_{i:03d}.obj"
        m.export(out_obj)
        rel = Path(os.path.relpath(str(out_obj), str(cache_dir))).as_posix()
        link_name = f"sophy_part_{i:03d}"
        part_entries_urdf.append(
            {
                "link_name": link_name,
                "rel_mesh": rel,
                "mass": mass,
                "com": com.tolist(),
                "ixx": ixx,
                "iyy": iyy,
                "izz": izz,
                "rgba": colors[i],
            }
        )
        cfg = meta["cfg"]
        sophy_part_rigid_specs.append(
            {
                "link_name": link_name,
                "material_slot": meta["slot"],
                "material_part_name": meta["material_part_name"],
                "rho_kgm3": meta["rho"],
                "friction": meta["friction"],
                "restitution": SOPHY_RIGID_PART_DEFAULT_RESTITUTION,
                "youngs_pa": cfg.get("E"),
                "poisson": cfg.get("nu"),
                "sigma_y_pa": cfg.get("sigma_y"),
                "elasticity": cfg.get("elasticity"),
                "plasticity": cfg.get("plasticity"),
                "volume_unit_mesh": vol,
                "mass_kg_approx": mass,
            }
        )
        sophy_part_material_records.append(
            {
                "part_index": i,
                "name": meta["material_part_name"],
                "material_name": str(cfg.get("mat_name", "")),
            }
        )
        unit_geoms.append(m)

    merged_unit = trimesh.util.concatenate(unit_geoms)
    unit_extents = np.asarray(merged_unit.extents, dtype=np.float64)
    raw_extents_scene = convert_bbox_extents_to_scene_frame(raw_extents, asset["dataset_name"])
    unit_extents_scene = convert_bbox_extents_to_scene_frame(unit_extents, asset["dataset_name"])

    total_mass = float(sum(masses))
    total_vol = float(sum(volumes))
    rho_eff = total_mass / max(total_vol, 1e-12)
    fric_eff = sum(m * pair_meta[i]["friction"] for i, m in enumerate(masses)) / max(total_mass, 1e-9)
    sophy_composite_rigid_material = {
        "family": "Rigid",
        "name": f"{asset['dataset_name']}_sophy_multipart",
        "rho": float(rho_eff),
        "friction": float(np.clip(fric_eff, 1e-2, 5.0)),
        "restitution": float(SOPHY_RIGID_PART_DEFAULT_RESTITUTION),
    }

    safe_robot = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in asset_id)[:120]
    urdf_path = cache_dir / f"{safe_robot}.urdf"
    write_sophy_multipart_fixed_chain_urdf(urdf_path, safe_robot, part_entries_urdf)

    asset_id_key = asset["asset_id"]
    cache_obj = ASSET_CACHE_DIR / f"{asset_id_key}_unit.obj"
    cache_json = ASSET_CACHE_DIR / f"{asset_id_key}_unit_meta.json"
    merged_unit.export(cache_obj)

    sample_dir = Path(asset["sample_dir"])
    has_texture = (sample_dir / "material.mtl").exists() and len(list(sample_dir.glob("material_*.png"))) > 0

    meta = {
        "asset_id": asset_id_key,
        "mesh_path": asset["mesh_path"],
        "render_mesh_path": asset["mesh_path"],
        "raw_bbox_extents": raw_extents.tolist(),
        "unit_bbox_extents": unit_extents.tolist(),
        "raw_bbox_extents_scene": raw_extents_scene.tolist(),
        "unit_bbox_extents_scene": unit_extents_scene.tolist(),
        "unit_scale_from_raw": float(unit_scale_factor),
        "n_vertices": int(len(merged_unit.vertices)),
        "n_faces": int(len(merged_unit.faces)),
        "has_texture": bool(has_texture),
        "coordinate_transform": {
            "source_up_axis": get_dataset_up_axis(asset["dataset_name"]),
            "target_up_axis": "z_up",
            "base_euler_xyz_rad": get_dataset_base_euler(asset["dataset_name"]).tolist(),
        },
        "part_color_style_version": PART_COLOR_STYLE_VERSION,
        "sophy_multipart_urdf": True,
        "render_urdf_path": str(urdf_path),
        "sophy_part_colors": colors,
        "sophy_part_material_records": sophy_part_material_records,
        "sophy_part_rigid_specs": sophy_part_rigid_specs,
        "sophy_composite_rigid_material": sophy_composite_rigid_material,
    }
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    asset["unit_mesh_path"] = str(cache_obj)
    asset["render_mesh_path"] = asset["mesh_path"]
    asset["render_urdf_path"] = str(urdf_path)
    asset["sophy_multipart_urdf"] = True
    asset["sophy_part_colors"] = colors
    asset["sophy_part_material_records"] = sophy_part_material_records
    asset["sophy_part_rigid_specs"] = sophy_part_rigid_specs
    asset["material_override"] = sophy_composite_rigid_material
    asset["raw_bbox_extents_local"] = raw_extents.tolist()
    asset["unit_bbox_extents_local"] = unit_extents.tolist()
    asset["raw_bbox_extents"] = raw_extents_scene.tolist()
    asset["unit_bbox_extents"] = unit_extents_scene.tolist()
    asset["n_vertices"] = int(meta["n_vertices"])
    asset["n_faces"] = int(meta["n_faces"])
    asset["has_texture"] = bool(has_texture)
    asset["coordinate_transform"] = meta["coordinate_transform"]
    return asset


def prepare_asset_cache(asset):
    ensure_dir(ASSET_CACHE_DIR)

    if asset.get("dataset_name") == "physx3d" and asset.get("physx3d_part_infos"):
        return prepare_physx3d_asset_cache(asset)

    asset_id = asset["asset_id"]
    cache_obj = ASSET_CACHE_DIR / f"{asset_id}_unit.obj"
    cache_json = ASSET_CACHE_DIR / f"{asset_id}_unit_meta.json"

    want_sophy_multipart = (
        bool(USE_SOPHY_PART_MATERIALS_RIGID)
        and bool(asset.get("mat_json"))
        and asset.get("dataset_name") != PRIMITIVE_DATASET_NAME
    )

    if cache_obj.exists() and cache_json.exists():
        with open(cache_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if (
            want_sophy_multipart
            and meta.get("sophy_multipart_urdf")
            and int(meta.get("part_color_style_version", 0)) == PART_COLOR_STYLE_VERSION
        ):
            urdf_p = Path(meta["render_urdf_path"])
            if urdf_p.is_file():
                _hydrate_sophy_multipart_asset_from_meta(asset, meta, cache_obj)
                return asset

        if want_sophy_multipart and not meta.get("sophy_multipart_urdf"):
            pass
        else:
            asset["unit_mesh_path"] = str(cache_obj)
            asset["render_mesh_path"] = meta.get("render_mesh_path", asset["mesh_path"])
            raw_bbox_scene = meta.get(
                "raw_bbox_extents_scene",
                convert_bbox_extents_to_scene_frame(meta["raw_bbox_extents"], asset["dataset_name"]),
            )
            unit_bbox_scene = meta.get(
                "unit_bbox_extents_scene",
                convert_bbox_extents_to_scene_frame(meta["unit_bbox_extents"], asset["dataset_name"]),
            )
            asset["raw_bbox_extents_local"] = meta["raw_bbox_extents"]
            asset["unit_bbox_extents_local"] = meta["unit_bbox_extents"]
            asset["raw_bbox_extents"] = np.asarray(raw_bbox_scene, dtype=np.float64).tolist()
            asset["unit_bbox_extents"] = np.asarray(unit_bbox_scene, dtype=np.float64).tolist()
            asset["n_vertices"] = meta.get("n_vertices", None)
            asset["n_faces"] = meta.get("n_faces", None)
            asset["has_texture"] = bool(meta.get("has_texture", False))
            asset["coordinate_transform"] = meta.get(
                "coordinate_transform",
                {
                    "source_up_axis": get_dataset_up_axis(asset["dataset_name"]),
                    "target_up_axis": "z_up",
                    "base_euler_xyz_rad": get_dataset_base_euler(asset["dataset_name"]).tolist(),
                },
            )
            return asset

    if want_sophy_multipart:
        try:
            return build_sophy_multipart_rigid_asset(asset)
        except Exception as e:
            print(f"[WARN] SOPHY multipart rigid build failed, fallback single mesh: {asset_id} err={e}")

    mesh = load_trimesh_any(Path(asset["mesh_path"]))
    mesh, raw_extents, unit_extents, unit_scale = sanitize_mesh(mesh)

    mesh.export(cache_obj)

    sample_dir = Path(asset["sample_dir"])
    has_texture = (sample_dir / "material.mtl").exists() and len(list(sample_dir.glob("material_*.png"))) > 0

    raw_extents_scene = convert_bbox_extents_to_scene_frame(raw_extents, asset["dataset_name"])
    unit_extents_scene = convert_bbox_extents_to_scene_frame(unit_extents, asset["dataset_name"])

    meta = {
        "asset_id": asset_id,
        "mesh_path": asset["mesh_path"],
        "render_mesh_path": asset["mesh_path"],
        "raw_bbox_extents": raw_extents.tolist(),
        "unit_bbox_extents": unit_extents.tolist(),
        "raw_bbox_extents_scene": raw_extents_scene.tolist(),
        "unit_bbox_extents_scene": unit_extents_scene.tolist(),
        "unit_scale_from_raw": float(unit_scale),
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "has_texture": bool(has_texture),
        "coordinate_transform": {
            "source_up_axis": get_dataset_up_axis(asset["dataset_name"]),
            "target_up_axis": "z_up",
            "base_euler_xyz_rad": get_dataset_base_euler(asset["dataset_name"]).tolist(),
        },
    }
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    asset["unit_mesh_path"] = str(cache_obj)
    asset["render_mesh_path"] = asset["mesh_path"]
    asset["raw_bbox_extents_local"] = raw_extents.tolist()
    asset["unit_bbox_extents_local"] = unit_extents.tolist()
    asset["raw_bbox_extents"] = raw_extents_scene.tolist()
    asset["unit_bbox_extents"] = unit_extents_scene.tolist()
    asset["n_vertices"] = meta["n_vertices"]
    asset["n_faces"] = meta["n_faces"]
    asset["has_texture"] = bool(has_texture)
    asset["coordinate_transform"] = meta["coordinate_transform"]
    return asset


def load_asset_material_or_default(asset):
    if asset.get("material_override", None) is not None:
        return asset["material_override"]

    default_mat = sample_rigid_material()
    mat_json = asset.get("mat_json", None)
    if mat_json is None:
        return default_mat

    try:
        with open(mat_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容旧格式：顶层直接是 rho / friction
        if "rho" in data:
            return {
                "family": "Rigid",
                "name": data.get("name", f"{asset['dataset_name']}_mesh"),
                "rho": float(data.get("rho", default_mat["rho"])),
                "friction": float(np.clip(data.get("friction", default_mat["friction"]), 1e-2, 5.0)),
                "restitution": float(np.clip(data.get("restitution", default_mat.get("restitution", 0.10)), 0.0, 1.2)),
            }

        # 兼容你现在的 part-level JSON
        rho_list = []
        fric_list = []
        for _, v in data.items():
            if not isinstance(v, dict):
                continue
            if "rho" in v:
                rho_list.append(float(v["rho"]))
            fr = infer_friction_from_part_info(v)
            if fr is not None:
                fric_list.append(fr)

        rho = float(np.mean(rho_list)) if len(rho_list) > 0 else default_mat["rho"]
        friction = float(np.mean(fric_list)) if len(fric_list) > 0 else default_mat["friction"]

        return {
            "family": "Rigid",
            "name": f"{asset['dataset_name']}_mesh",
            "rho": rho,
            "friction": float(np.clip(friction, 1e-2, 5.0)),
            "restitution": float(np.clip(SOPHY_RIGID_PART_DEFAULT_RESTITUTION, 0.0, 1.2)),
        }
    except Exception:
        return default_mat

def build_asset_bank():
    raw_assets = []

    if DATASET_SOURCE in {"sophy", "mixed", "all"}:
        raw_assets.extend(find_asset_dirs_from_roots(SOURCE_DATASET_ROOTS))

    if DATASET_SOURCE in {"physx3d", "mixed", "all"}:
        raw_assets.extend(build_physx3d_asset_bank())

    if DATASET_SOURCE in {"primitive", "all"}:
        raw_assets.extend(build_primitive_asset_bank())

    print(f"[INFO] total raw assets found: {len(raw_assets)} | dataset_source={DATASET_SOURCE}")

    bank = []
    failed = []

    for a in raw_assets:
        try:
            if a.get("dataset_name") == PRIMITIVE_DATASET_NAME:
                bank.append(a)
            else:
                bank.append(prepare_asset_cache(a))
        except Exception as e:
            failed.append({
                "sample_dir": a.get("sample_dir"),
                "mesh_path": a.get("mesh_path"),
                "asset_id": a.get("asset_id"),
                "error": str(e),
            })
            print(f"[WARN] skip asset: {a.get('asset_id', a.get('sample_dir'))} | err={e}")

    manifest = {
        "dataset_source": DATASET_SOURCE,
        "n_raw_assets": len(raw_assets),
        "n_usable_assets": len(bank),
        "n_failed_assets": len(failed),
        "usable_assets": bank,
        "failed_assets": failed,
    }
    with open(ASSET_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(manifest), f, ensure_ascii=False, indent=2)

    print(f"[INFO] usable assets: {len(bank)}")
    if len(bank) == 0 and USE_DATASET_MESH_OBJECTS:
        print("[WARN] no usable dataset mesh assets found; will fallback to procedural primitives only.")

    return bank


# =========================
# 容器与相机
# =========================

def sample_rigid_material():
    libs = {
        "light_plastic": {"rho": (300, 800), "friction": (0.20, 0.80), "restitution": (0.10, 0.22)},
        "wood_like": {"rho": (500, 1000), "friction": (0.30, 0.90), "restitution": (0.08, 0.18)},
        "metal_like": {"rho": (1500, 3000), "friction": (0.18, 0.55), "restitution": (0.05, 0.14)},
        "rubber_like": {"rho": (900, 1300), "friction": (0.85, 1.40), "restitution": (0.45, 0.78)},
    }
    name = random.choice(list(libs.keys()))
    conf = libs[name]
    rho = float(np.random.uniform(*conf["rho"]))
    friction = float(np.clip(np.random.uniform(*conf["friction"]), 1e-2, 5.0))
    restitution = float(np.clip(np.random.uniform(*conf["restitution"]), 0.0, 1.2))
    return {
        "family": "Rigid",
        "name": name,
        "rho": rho,
        "friction": friction,
        "restitution": restitution,
    }


def sample_mpm_material():
    material_name = random.choice(["soft_foam", "gel", "rubbery"])
    if material_name == "soft_foam":
        E = float(np.random.uniform(2e4, 8e4))
        nu = float(np.random.uniform(0.15, 0.30))
        rho = float(np.random.uniform(500, 900))
    elif material_name == "gel":
        E = float(np.random.uniform(8e4, 2.5e5))
        nu = float(np.random.uniform(0.20, 0.35))
        rho = float(np.random.uniform(900, 1200))
    else:
        E = float(np.random.uniform(2.5e5, 8e5))
        nu = float(np.random.uniform(0.25, 0.40))
        rho = float(np.random.uniform(950, 1400))

    return {
        "family": "MPM",
        "name": material_name,
        "type": "Elastic",
        "E": E,
        "nu": nu,
        "rho": rho,
        "sampler": random.choice(["pbs", "regular", "random"]),
        "model": random.choice(["corotation", "neohooken"]),
    }


def sample_sph_material():
    return {
        "family": "SPH",
        "name": random.choice(["water_like", "viscous_liquid", "high_tension_liquid"]),
        "rho": float(np.random.uniform(900, 1200)),
        "stiffness": float(np.random.uniform(3e4, 8e4)),
        "exponent": float(np.random.uniform(5.0, 8.0)),
        "mu": float(np.random.uniform(0.002, 0.03)),
        "gamma": float(np.random.uniform(0.005, 0.03)),
        "sampler": random.choice(["regular", "pbs"]),
    }


# =========================
# 几何辅助
# =========================

def rigid_geom_and_margins(force_shape: Optional[str] = None):
    if force_shape is None:
        shape = random.choice(["box", "sphere", "cylinder", "capsule"])
    else:
        shape = str(force_shape).strip().lower()
        if shape not in {"box", "sphere", "cylinder", "capsule"}:
            raise ValueError(f"Unsupported forced shape: {force_shape}")

    if shape == "box":
        size = [
            float(np.random.uniform(0.08, 0.24)),
            float(np.random.uniform(0.08, 0.24)),
            float(np.random.uniform(0.08, 0.24)),
        ]
        geom = {"shape": "box", "size": size}
        half_x = size[0] / 2.0
        half_y = size[1] / 2.0
        half_z = size[2] / 2.0

    elif shape == "sphere":
        radius = float(np.random.uniform(0.05, 0.12))
        geom = {"shape": "sphere", "radius": radius}
        half_x = radius
        half_y = radius
        half_z = radius

    elif shape == "cylinder":
        radius = float(np.random.uniform(0.04, 0.10))
        height = float(np.random.uniform(0.10, 0.24))
        geom = {"shape": "cylinder", "radius": radius, "height": height}
        half_x = radius
        half_y = radius
        half_z = height / 2.0

    else:
        radius = float(np.random.uniform(0.04, 0.09))
        height = float(np.random.uniform(0.10, 0.22))
        geom = {"shape": "capsule", "radius": radius, "height": height}
        half_x = radius
        half_y = radius
        half_z = height / 2.0 + radius

    bound_r = compute_bound_radius_from_half_extents(half_x, half_y, half_z)
    return geom, half_x, half_y, half_z, bound_r


SOURCE_KIND_ALIASES = {
    "physx3d": "physx3d",
    "physxnet": "physx3d",
    "physx": "physx3d",
    "sophy": "sophy",
    "primitive": "primitive",
}


def get_asset_source_kind(asset: Dict[str, Any]) -> str:
    dataset_name = str(asset.get("dataset_name", "")).strip().lower()
    if dataset_name == "physx3d":
        return "physx3d"
    if dataset_name == PRIMITIVE_DATASET_NAME:
        return "primitive"
    return "sophy"


def normalize_source_kind_request(request: Any) -> List[str]:
    if request is None:
        return []
    values = request if isinstance(request, (list, tuple, set)) else [request]
    out = []
    for x in values:
        key = str(x).strip().lower()
        mapped = SOURCE_KIND_ALIASES.get(key, None)
        if mapped is not None and mapped not in out:
            out.append(mapped)
    return out


# =========================
# 物体采样
# =========================

def sample_dataset_rigid_object(obj_id: int, asset_bank, pattern="drop_cluster", motion_request=None):
    motion_request = motion_request or {}
    force_source_kinds = normalize_source_kind_request(motion_request.get("force_dataset_source", None))
    if force_source_kinds:
        source_filtered_assets = [a for a in asset_bank if get_asset_source_kind(a) in set(force_source_kinds)]
        if source_filtered_assets:
            asset = random.choice(source_filtered_assets)
        else:
            asset = random.choice(asset_bank)
    else:
        asset = random.choice(asset_bank)

    if asset.get("dataset_name") == PRIMITIVE_DATASET_NAME:
        geom_template = dict(asset["primitive_geom_template"])
        bbox_ext = np.asarray(asset["unit_bbox_extents"], dtype=np.float32)

        half_x = float(bbox_ext[0] / 2.0)
        half_y = float(bbox_ext[1] / 2.0)
        half_z = float(bbox_ext[2] / 2.0)
        bound_r = compute_bound_radius_from_half_extents(half_x, half_y, half_z)

        base_euler = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        motion = sample_rigid_motion(
            half_x,
            half_y,
            half_z,
            bound_r=bound_r,
            forced_mode=motion_request.get("forced_mode", None),
            target_pos=motion_request.get("target_pos", None),
            occupied_slots=motion_request.get("occupied_slots", None),
            base_euler=base_euler,
        )
        final_euler = (base_euler + np.asarray(motion["pose_delta"], dtype=np.float32)).tolist()

        geom_dict = dict(geom_template)
        geom_dict["bbox_extents"] = bbox_ext.tolist()
        geom_dict["bound_radius"] = float(bound_r)
        geom_dict["asset_id"] = asset["asset_id"]
        geom_dict["dataset_name"] = asset["dataset_name"]
        geom_dict["sample_dir"] = asset["sample_dir"]
        geom_dict["primitive_material_name"] = asset.get("primitive_material_name")
        geom_dict["primitive_shape_name"] = asset.get("primitive_shape_name")

        return {
            "object_id": obj_id,
            "solver": "Rigid",
            "source_type": "dataset_mesh",
            "motion_type": motion["motion_type"],
            "pattern": pattern,
            "geom": geom_dict,
            "material": dict(asset["material_override"]),
            "coordinate_transform": {
                "source_up_axis": "z_up",
                "target_up_axis": "z_up",
                "base_euler_xyz_rad": [0.0, 0.0, 0.0],
            },
            "init_pos": [float(x) for x in motion["init_pos"]],
            "init_euler": [float(x) for x in final_euler],
            "init_linvel": [float(x) for x in motion["init_linvel"]],
            "init_angvel": [float(x) for x in motion["init_angvel"]],
            "color": list(asset["primitive_color"]),
            "surface_vis_mode": "visual",
        }

    unit_ext = np.array(asset["unit_bbox_extents"], dtype=np.float32)
    target_size = float(np.random.uniform(0.20, 0.50))
    mesh_scale = target_size / max(float(np.max(unit_ext)), 1e-8)

    scaled_ext = unit_ext * mesh_scale
    half_x = float(scaled_ext[0] / 2.0)
    half_y = float(scaled_ext[1] / 2.0)
    half_z = float(scaled_ext[2] / 2.0)
    bound_r = compute_bound_radius_from_half_extents(half_x, half_y, half_z)

    base_euler = get_dataset_base_euler(asset["dataset_name"])

    motion = sample_rigid_motion(
        half_x,
        half_y,
        half_z,
        bound_r=bound_r,
        forced_mode=motion_request.get("forced_mode", None),
        target_pos=motion_request.get("target_pos", None),
        occupied_slots=motion_request.get("occupied_slots", None),
        base_euler=base_euler,
    )
    final_euler = (base_euler + np.asarray(motion["pose_delta"], dtype=np.float32)).tolist()

    use_sophy_multipart_urdf = bool(
        USE_SOPHY_PART_MATERIALS_RIGID
        and asset.get("sophy_multipart_urdf")
        and asset.get("render_urdf_path")
    )
    use_part_colored_urdf = bool(
        asset["dataset_name"] == "physx3d"
        and PHYSX3D_USE_PART_COLORED_URDF
        and asset.get("render_urdf_path", None) is not None
    ) or use_sophy_multipart_urdf
    use_texture = bool(asset.get("has_texture", False) and USE_TEXTURED_DATASET_MESH and asset["dataset_name"] != "physx3d")

    if use_part_colored_urdf:
        geom_dict = {
            "shape": "urdf",
            "urdf_file": asset["render_urdf_path"],
            "scale": float(mesh_scale),
            "bbox_extents": scaled_ext.tolist(),
            "bound_radius": float(bound_r),
            "asset_id": asset["asset_id"],
            "dataset_name": asset["dataset_name"],
            "sample_dir": asset["sample_dir"],
            "n_vertices": asset.get("n_vertices", None),
            "n_faces": asset.get("n_faces", None),
            "use_urdf_material": bool(
                (asset["dataset_name"] == "physx3d" and PHYSX3D_URDF_USE_URDF_MATERIAL)
                or use_sophy_multipart_urdf
            ),
            "part_colors": (
                asset.get("sophy_part_colors", [])
                if use_sophy_multipart_urdf
                else asset.get("physx3d_part_colors", [])
            ),
            "part_materials": (
                asset.get("sophy_part_material_records", [])
                if use_sophy_multipart_urdf
                else asset.get("physx3d_part_materials", [])
            ),
            "unit_part_mesh_paths": (
                []
                if use_sophy_multipart_urdf
                else asset.get("physx3d_unit_part_mesh_paths", [])
            ),
        }
        mesh_color = None
    else:
        render_mesh_file = asset["render_mesh_path"] if use_texture else asset["unit_mesh_path"]
        geom_dict = {
            "shape": "mesh",
            "mesh_file": render_mesh_file,
            "scale": float(mesh_scale),
            "bbox_extents": scaled_ext.tolist(),
            "bound_radius": float(bound_r),
            "asset_id": asset["asset_id"],
            "dataset_name": asset["dataset_name"],
            "sample_dir": asset["sample_dir"],
            "n_vertices": asset.get("n_vertices", None),
            "n_faces": asset.get("n_faces", None),
            "use_texture": use_texture,
            "part_colors": asset.get("physx3d_part_colors", []),
            "part_materials": asset.get("physx3d_part_materials", []),
        }
        mesh_color = None if use_texture else sample_color()

    return {
        "object_id": obj_id,
        "solver": "Rigid",
        "source_type": "dataset_mesh",
        "motion_type": motion["motion_type"],
        "pattern": pattern,
        "geom": geom_dict,
        "material": load_asset_material_or_default(asset),
        "coordinate_transform": asset.get(
            "coordinate_transform",
            {
                "source_up_axis": get_dataset_up_axis(asset["dataset_name"]),
                "target_up_axis": "z_up",
                "base_euler_xyz_rad": base_euler.tolist(),
            },
        ),
        "init_pos": [float(x) for x in motion["init_pos"]],
        "init_euler": [float(x) for x in final_euler],
        "init_linvel": [float(x) for x in motion["init_linvel"]],
        "init_angvel": [float(x) for x in motion["init_angvel"]],
        "color": mesh_color,
        "surface_vis_mode": "visual",
        "sophy_part_rigid_specs": asset.get("sophy_part_rigid_specs"),
    }


def sample_procedural_rigid_object(obj_id: int, pattern="drop_cluster", motion_request=None):
    motion_request = motion_request or {}
    force_shape = motion_request.get("force_shape", None)
    geom, half_x, half_y, half_z, bound_r = rigid_geom_and_margins(force_shape=force_shape)
    material = sample_rigid_material()
    motion = sample_rigid_motion(
        half_x,
        half_y,
        half_z,
        bound_r=bound_r,
        forced_mode=motion_request.get("forced_mode", None),
        target_pos=motion_request.get("target_pos", None),
        occupied_slots=motion_request.get("occupied_slots", None),
        base_euler=[0.0, 0.0, 0.0],
    )

    final_euler = [
        float(motion["pose_delta"][0]),
        float(motion["pose_delta"][1]),
        float(motion["pose_delta"][2]),
    ]

    geom = dict(geom)
    geom["bound_radius"] = float(bound_r)

    return {
        "object_id": obj_id,
        "solver": "Rigid",
        "source_type": "procedural",
        "motion_type": motion["motion_type"],
        "pattern": pattern,
        "geom": geom,
        "material": material,
        "coordinate_transform": {
            "source_up_axis": "z_up",
            "target_up_axis": "z_up",
            "base_euler_xyz_rad": [0.0, 0.0, 0.0],
        },
        "init_pos": [float(x) for x in motion["init_pos"]],
        "init_euler": final_euler,
        "init_linvel": [float(x) for x in motion["init_linvel"]],
        "init_angvel": [float(x) for x in motion["init_angvel"]],
        "color": sample_color(),
        "surface_vis_mode": "visual",
    }


def sample_rigid_object(obj_id: int, pattern="drop_cluster", asset_bank=None, motion_request=None):
    motion_request = motion_request or {}
    force_source = str(motion_request.get("force_source", "")).strip().lower()
    if force_source == "procedural":
        return sample_procedural_rigid_object(obj_id, pattern=pattern, motion_request=motion_request)
    if force_source == "dataset":
        if asset_bank is not None and len(asset_bank) > 0:
            return sample_dataset_rigid_object(obj_id, asset_bank, pattern=pattern, motion_request=motion_request)
        return sample_procedural_rigid_object(obj_id, pattern=pattern, motion_request=motion_request)

    if (
        USE_DATASET_MESH_OBJECTS
        and asset_bank is not None
        and len(asset_bank) > 0
        and (np.random.rand() < DATASET_OBJECT_PROB)
    ):
        return sample_dataset_rigid_object(obj_id, asset_bank, pattern=pattern, motion_request=motion_request)

    return sample_procedural_rigid_object(obj_id, pattern=pattern, motion_request=motion_request)


DEFAULT_FORCE_SHAPE_BY_MODE = {
    "rolling_left": "sphere",
    "rolling_right": "sphere",
    "projectile_arc_forward": "sphere",
    "projectile_cross_left": "sphere",
    "projectile_cross_right": "sphere",
}

DEFAULT_FORCE_DATASET_SOURCE_BY_MODE = {
    "swing_drop_left": "primitive",
    "swing_drop_right": "primitive",
}


def _pick_force_shape_for_mode(mode: str, force_shape_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    if force_shape_map and mode in force_shape_map:
        return force_shape_map[mode]
    return DEFAULT_FORCE_SHAPE_BY_MODE.get(mode)


def _pick_force_dataset_source_for_mode(mode: str) -> Optional[str]:
    return DEFAULT_FORCE_DATASET_SOURCE_BY_MODE.get(mode)


def build_uniform_dynamic_rigid_objects(
    category_name: str,
    n_obj: int,
    motion_modes,
    asset_bank=None,
    force_source: Optional[str] = None,
    force_shape_map: Optional[Dict[str, str]] = None,
    dataset_source_mix: Optional[List[str]] = None,
):
    occupied_slots = []
    objects = []
    for obj_id in range(n_obj):
        forced_mode = motion_modes[obj_id % len(motion_modes)]
        motion_request = {
            "forced_mode": forced_mode,
            "occupied_slots": occupied_slots,
        }
        force_shape = _pick_force_shape_for_mode(forced_mode, force_shape_map)
        if force_shape is not None:
            motion_request["force_shape"] = force_shape
        if force_source:
            motion_request["force_source"] = force_source
        if force_source == "dataset":
            mode_source = _pick_force_dataset_source_for_mode(forced_mode)
            if mode_source is not None:
                motion_request["force_dataset_source"] = [mode_source]
            elif dataset_source_mix:
                motion_request["force_dataset_source"] = [dataset_source_mix[obj_id % len(dataset_source_mix)]]
        objects.append(
            sample_rigid_object(
                obj_id,
                pattern=category_name,
                asset_bank=asset_bank,
                motion_request=motion_request,
            )
        )

    random.shuffle(objects)
    for new_id, obj in enumerate(objects):
        obj["object_id"] = new_id
        obj["pattern"] = category_name
    return objects


def build_ground_static_plus_dynamic_objects(category_name: str, asset_bank=None):
    occupied_slots = []
    objects = []
    next_obj_id = 0

    n_static = random.randint(1, 3)
    n_dynamic = random.randint(1, 3)

    for _ in range(n_static):
        objects.append(
            sample_rigid_object(
                next_obj_id,
                pattern=category_name,
                asset_bank=asset_bank,
                motion_request={
                    "forced_mode": "static_rest",
                    "occupied_slots": occupied_slots,
                },
            )
        )
        next_obj_id += 1

    dynamic_modes = [
        "top_drop",
        "top_toss",
        "front_slide_in",
        "diagonal_corner_left",
        "diagonal_corner_right",
        "side_throw_left",
        "side_throw_right",
    ]
    sampled_dynamic_modes = [random.choice(dynamic_modes) for _ in range(n_dynamic)]
    for forced_mode in sampled_dynamic_modes:
        objects.append(
            sample_rigid_object(
                next_obj_id,
                pattern=category_name,
                asset_bank=asset_bank,
                motion_request={
                    "forced_mode": forced_mode,
                    "occupied_slots": occupied_slots,
                },
            )
        )
        next_obj_id += 1

    return {
        "objects": objects,
        "static_count": n_static,
        "dynamic_count": n_dynamic,
        "dynamic_modes": sampled_dynamic_modes,
    }


def get_sim_steps_for_motion_modes(motion_modes):
    motion_modes = set(motion_modes)
    if motion_modes & {
        "projectile_arc_forward",
        "projectile_cross_left",
        "projectile_cross_right",
        "swing_drop_left",
        "swing_drop_right",
    }:
        return 240
    if motion_modes & {"rolling_left", "rolling_right"}:
        return 225
    if motion_modes & {"side_throw_left", "side_throw_right"}:
        return 210
    if motion_modes & {"diagonal_corner_left", "diagonal_corner_right"}:
        return 200
    if motion_modes & {"top_toss"}:
        return 190
    return 180


def compute_preview_stride(dt: float, num_steps: int, phys_duration_s: float, preview_target_fps: int = PREVIEW_FPS) -> int:
    """Subsampling stride so preview has ~preview_target_fps frames per second of simulation time.

    phys_duration_s should be num_steps * dt. Preview MP4 encoding FPS is set in export so playback duration equals phys_duration_s.
    """
    dt = float(max(dt, 1e-6))
    n = max(1, int(num_steps))
    phys_duration_s = float(max(phys_duration_s, n * dt))
    target_frames = max(1.0, phys_duration_s * float(preview_target_fps))
    return max(1, int(round(float(n) / target_frames)))


def build_interaction_pair_plus_dynamic_objects(category_name: str, n_obj: int, asset_bank=None):
    occupied_slots = []
    objects = []
    next_obj_id = 0
    n_obj = max(5, int(n_obj))

    source_mix = ["physx3d", "sophy", "primitive"]
    source_cursor = 0

    def add_obj(
        forced_mode: str,
        *,
        target_pos: Optional[List[float]] = None,
        force_shape: Optional[str] = None,
        force_source: str = "dataset",
        force_dataset_source: Optional[str] = None,
    ):
        nonlocal next_obj_id, source_cursor
        req = {
            "forced_mode": forced_mode,
            "occupied_slots": occupied_slots,
            "force_source": force_source,
        }
        if force_source == "dataset":
            chosen_source = force_dataset_source
            if chosen_source is None:
                chosen_source = _pick_force_dataset_source_for_mode(forced_mode)
            if chosen_source is None:
                chosen_source = source_mix[source_cursor % len(source_mix)]
            req["force_dataset_source"] = [chosen_source]
            source_cursor += 1
        if target_pos is not None:
            req["target_pos"] = target_pos
        if force_shape is not None:
            req["force_shape"] = force_shape
        obj = sample_rigid_object(
            next_obj_id,
            pattern=category_name,
            asset_bank=asset_bank,
            motion_request=req,
        )
        objects.append(obj)
        next_obj_id += 1
        return obj

    target = add_obj("static_rest", force_shape="box")
    strike_mode = random.choice(["strike_static_left", "strike_static_right"])
    add_obj(strike_mode, target_pos=target["init_pos"], force_shape="sphere")
    add_obj(random.choice(["rolling_left", "rolling_right"]), force_shape="sphere")
    add_obj(
        random.choice(["projectile_arc_forward", "projectile_cross_left", "projectile_cross_right"]),
        force_shape="sphere",
    )
    add_obj(random.choice(["swing_drop_left", "swing_drop_right"]))

    extra_modes = [
        "front_slide_in",
        "diagonal_corner_left",
        "diagonal_corner_right",
        "side_throw_left",
        "side_throw_right",
        "rolling_left",
        "rolling_right",
        "projectile_arc_forward",
        "projectile_cross_left",
        "projectile_cross_right",
        "swing_drop_left",
        "swing_drop_right",
    ]
    while len(objects) < n_obj:
        mode = random.choice(extra_modes)
        add_obj(mode, force_shape=_pick_force_shape_for_mode(mode))

    return {
        "objects": objects,
        "static_count": int(sum(1 for obj in objects if obj["motion_type"] == "static_rest")),
        "dynamic_count": int(sum(1 for obj in objects if obj["motion_type"] != "static_rest")),
        "dynamic_modes": [str(obj["motion_type"]) for obj in objects if obj["motion_type"] != "static_rest"],
    }


def build_dual_interaction_group_objects(category_name: str, n_obj: int, asset_bank=None):
    occupied_slots = []
    objects = []
    next_obj_id = 0
    n_obj = max(8, int(n_obj))

    source_mix = ["physx3d", "sophy", "primitive"]
    source_cursor = 0

    def add_obj(
        forced_mode: str,
        *,
        target_pos: Optional[List[float]] = None,
        force_shape: Optional[str] = None,
        force_source: str = "dataset",
        force_dataset_source: Optional[str] = None,
    ):
        nonlocal next_obj_id, source_cursor
        req = {
            "forced_mode": forced_mode,
            "occupied_slots": occupied_slots,
            "force_source": force_source,
        }
        if force_source == "dataset":
            chosen_source = force_dataset_source
            if chosen_source is None:
                chosen_source = _pick_force_dataset_source_for_mode(forced_mode)
            if chosen_source is None:
                chosen_source = source_mix[source_cursor % len(source_mix)]
            req["force_dataset_source"] = [chosen_source]
            source_cursor += 1
        if target_pos is not None:
            req["target_pos"] = target_pos
        if force_shape is not None:
            req["force_shape"] = force_shape
        obj = sample_rigid_object(
            next_obj_id,
            pattern=category_name,
            asset_bank=asset_bank,
            motion_request=req,
        )
        objects.append(obj)
        next_obj_id += 1
        return obj

    target_a = add_obj("static_rest", force_shape="box")
    target_b = add_obj("static_rest", force_shape="box")
    add_obj(random.choice(["strike_static_left", "strike_static_right"]), target_pos=target_a["init_pos"], force_shape="sphere")
    add_obj(random.choice(["strike_static_left", "strike_static_right"]), target_pos=target_b["init_pos"], force_shape="sphere")
    add_obj("rolling_left", force_shape="sphere")
    add_obj("rolling_right", force_shape="sphere")
    add_obj("projectile_cross_left", force_shape="sphere")
    add_obj("projectile_cross_right", force_shape="sphere")

    extra_modes = [
        "projectile_arc_forward",
        "swing_drop_left",
        "swing_drop_right",
        "front_slide_in",
        "diagonal_corner_left",
        "diagonal_corner_right",
    ]
    while len(objects) < n_obj:
        mode = random.choice(extra_modes)
        add_obj(mode, force_shape=_pick_force_shape_for_mode(mode))

    return {
        "objects": objects,
        "static_count": int(sum(1 for obj in objects if obj["motion_type"] == "static_rest")),
        "dynamic_count": int(sum(1 for obj in objects if obj["motion_type"] != "static_rest")),
        "dynamic_modes": [str(obj["motion_type"]) for obj in objects if obj["motion_type"] != "static_rest"],
    }


def build_omni_showcase_objects(category_name: str, n_obj: int, asset_bank=None):
    occupied_slots = []
    objects = []
    next_obj_id = 0

    source_mix = ["physx3d", "sophy", "primitive"]
    source_cursor = 0

    def add_obj(
        forced_mode: str,
        *,
        target_pos: Optional[List[float]] = None,
        force_shape: Optional[str] = None,
        force_source: str = "dataset",
        force_dataset_source: Optional[str] = None,
    ):
        nonlocal next_obj_id, source_cursor
        req = {
            "forced_mode": forced_mode,
            "occupied_slots": occupied_slots,
            "force_source": force_source,
        }
        if force_source == "dataset":
            chosen_source = force_dataset_source
            if chosen_source is None:
                chosen_source = _pick_force_dataset_source_for_mode(forced_mode)
            if chosen_source is None:
                chosen_source = source_mix[source_cursor % len(source_mix)]
            req["force_dataset_source"] = [chosen_source]
            source_cursor += 1
        if target_pos is not None:
            req["target_pos"] = target_pos
        if force_shape is not None:
            req["force_shape"] = force_shape
        obj = sample_rigid_object(
            next_obj_id,
            pattern=category_name,
            asset_bank=asset_bank,
            motion_request=req,
        )
        objects.append(obj)
        next_obj_id += 1
        return obj

    anchor = add_obj("static_rest", force_shape="box")
    add_obj(random.choice(["strike_static_left", "strike_static_right"]), target_pos=anchor["init_pos"], force_shape="sphere")

    showcase_modes = [
        "static_rest",
        "strike_static_left",
        "strike_static_right",
        "front_slide_in",
        "diagonal_corner_left",
        "diagonal_corner_right",
        "side_throw_left",
        "side_throw_right",
        "rolling_left",
        "rolling_right",
        "projectile_arc_forward",
        "projectile_cross_left",
        "projectile_cross_right",
        "swing_drop_left",
        "swing_drop_right",
    ]
    random.shuffle(showcase_modes)

    desired_num = max(int(n_obj), len(showcase_modes) + 2)
    for mode in showcase_modes:
        add_obj(mode, force_shape=_pick_force_shape_for_mode(mode))
        if len(objects) >= desired_num:
            break

    while len(objects) < desired_num:
        mode = random.choice(showcase_modes)
        add_obj(mode, force_shape=_pick_force_shape_for_mode(mode))

    return {
        "objects": objects,
        "static_count": int(sum(1 for obj in objects if obj["motion_type"] == "static_rest")),
        "dynamic_count": int(sum(1 for obj in objects if obj["motion_type"] != "static_rest")),
        "dynamic_modes": [str(obj["motion_type"]) for obj in objects if obj["motion_type"] != "static_rest"],
    }


def build_rigid_scene_from_category(category_spec: dict, object_count: int, asset_bank=None):
    category_name = category_spec["name"]
    builder = category_spec["scene_builder"]

    if builder == "uniform_dynamic":
        objects = build_uniform_dynamic_rigid_objects(
            category_name=category_name,
            n_obj=object_count,
            motion_modes=category_spec["motion_modes"],
            asset_bank=asset_bank,
            force_source=category_spec.get("force_source", None),
            force_shape_map=category_spec.get("force_shape_map", None),
            dataset_source_mix=category_spec.get("dataset_source_mix", None),
        )
        return {
            "objects": objects,
            "object_count_bucket": int(object_count),
            "num_static_objects": 0,
            "num_dynamic_objects": int(object_count),
            "motion_modes_present": sorted({obj["motion_type"] for obj in objects}),
            "sim_steps": get_sim_steps_for_motion_modes(category_spec["motion_modes"]),
        }

    if builder == "ground_static_plus_dynamic":
        mix = build_ground_static_plus_dynamic_objects(
            category_name=category_name,
            asset_bank=asset_bank,
        )
        return {
            "objects": mix["objects"],
            "object_count_bucket": None,
            "num_static_objects": int(mix["static_count"]),
            "num_dynamic_objects": int(mix["dynamic_count"]),
            "motion_modes_present": sorted(set(mix["dynamic_modes"] + ["static_rest"])),
            "sim_steps": get_sim_steps_for_motion_modes(mix["dynamic_modes"]),
            "hybrid_counts": {
                "static_range_sampled": int(mix["static_count"]),
                "dynamic_range_sampled": int(mix["dynamic_count"]),
            },
        }

    if builder == "interaction_pair_plus_dynamic":
        mix = build_interaction_pair_plus_dynamic_objects(
            category_name=category_name,
            n_obj=object_count,
            asset_bank=asset_bank,
        )
        return {
            "objects": mix["objects"],
            "object_count_bucket": int(object_count),
            "num_static_objects": int(mix["static_count"]),
            "num_dynamic_objects": int(mix["dynamic_count"]),
            "motion_modes_present": sorted(set(mix["dynamic_modes"] + ["static_rest"])),
            "sim_steps": get_sim_steps_for_motion_modes(mix["dynamic_modes"]),
        }

    if builder == "dual_interaction_groups":
        mix = build_dual_interaction_group_objects(
            category_name=category_name,
            n_obj=object_count,
            asset_bank=asset_bank,
        )
        return {
            "objects": mix["objects"],
            "object_count_bucket": int(object_count),
            "num_static_objects": int(mix["static_count"]),
            "num_dynamic_objects": int(mix["dynamic_count"]),
            "motion_modes_present": sorted(set(mix["dynamic_modes"] + ["static_rest"])),
            "sim_steps": get_sim_steps_for_motion_modes(mix["dynamic_modes"]),
        }

    if builder == "omni_showcase":
        mix = build_omni_showcase_objects(
            category_name=category_name,
            n_obj=object_count,
            asset_bank=asset_bank,
        )
        return {
            "objects": mix["objects"],
            "object_count_bucket": int(object_count),
            "num_static_objects": int(mix["static_count"]),
            "num_dynamic_objects": int(mix["dynamic_count"]),
            "motion_modes_present": sorted(set(mix["dynamic_modes"] + ["static_rest"])),
            "sim_steps": get_sim_steps_for_motion_modes(mix["dynamic_modes"]),
        }

    raise ValueError(f"Unknown scene_builder: {builder}")


def build_scene_generation_plan(samples_per_category: int, motion_category_specs: List[Dict[str, Any]], object_counts: List[int]) -> List[Dict[str, Any]]:
    plan = []
    default_counts = [int(x) for x in object_counts]
    for category_spec in motion_category_specs:
        builder = str(category_spec["scene_builder"])
        category_counts = category_spec.get("object_counts", None)
        if category_counts is not None:
            category_counts = [max(1, int(x)) for x in category_counts]

        if builder in {"uniform_dynamic", "interaction_pair_plus_dynamic", "dual_interaction_groups", "omni_showcase"}:
            counts_to_use = category_counts if category_counts else default_counts
            for object_count in counts_to_use:
                for sample_idx in range(samples_per_category):
                    plan.append(
                        {
                            "category_spec": category_spec,
                            "object_count": int(object_count),
                            "sample_index": int(sample_idx),
                        }
                    )
        elif builder == "ground_static_plus_dynamic":
            for sample_idx in range(samples_per_category):
                plan.append(
                    {
                        "category_spec": category_spec,
                        "object_count": None,
                        "sample_index": int(sample_idx),
                    }
                )
        else:
            raise ValueError(f"Unknown scene_builder: {builder}")
    return plan


def sample_extra_static_count(pattern: str, n_obj: int) -> int:
    prob = STATIC_PROP_PROB_BY_PATTERN.get(pattern, 0.0)
    if np.random.rand() >= prob:
        return 0

    lo, hi = STATIC_PROP_COUNT_RANGE_BY_PATTERN.get(pattern, (0, 0))
    if hi <= 0:
        return 0

    count = random.randint(lo, hi)
    min_dynamic = MIN_DYNAMIC_COUNT_BY_PATTERN.get(pattern, 2)

    # 总物体数固定为 n_obj，因此额外静止物体不能把动态物体挤没
    count = min(count, max(0, n_obj - min_dynamic))
    return max(0, count)

def build_rigid_objects(pattern: str, n_obj: int, asset_bank=None):
    occupied_slots = []
    objects = []
    next_obj_id = 0
    print(f"✅ pattern {pattern}")

    extra_static_count = sample_extra_static_count(pattern, n_obj)

    def add_static_props(num_static: int):
        nonlocal next_obj_id
        for _ in range(num_static):
            objects.append(
                sample_rigid_object(
                    next_obj_id,
                    pattern=pattern,
                    asset_bank=asset_bank,
                    motion_request={
                        "forced_mode": "static_rest",
                        "occupied_slots": occupied_slots,
                    },
                )
            )
            next_obj_id += 1

    if pattern == "drop_cluster":
        # 先放一些本来就在地上的静止物体
        add_static_props(extra_static_count)

        dynamic_count = n_obj - len(objects)
        generic_modes = [
            "top_drop",
            "top_toss",
            "front_slide_in",
            "diagonal_corner_left",
            "diagonal_corner_right",
        ]
        for _ in range(dynamic_count):
            forced_mode = random.choice(generic_modes)
            objects.append(
                sample_rigid_object(
                    next_obj_id,
                    pattern=pattern,
                    asset_bank=asset_bank,
                    motion_request={
                        "forced_mode": forced_mode,
                        "occupied_slots": occupied_slots,
                    },
                )
            )
            next_obj_id += 1

    elif pattern == "opposed_lanes":
        # 先混入一些静止地面物体
        add_static_props(extra_static_count)

        dynamic_count = n_obj - len(objects)
        lane_modes = [
            "diagonal_corner_left",
            "diagonal_corner_right",
            "side_throw_left",
            "side_throw_right",
            "front_slide_in",
            "top_toss",
        ]
        for k in range(dynamic_count):
            forced_mode = lane_modes[k % len(lane_modes)]
            if k >= 4 and np.random.rand() < 0.5:
                forced_mode = random.choice(["top_drop", "top_toss", "front_slide_in"])

            objects.append(
                sample_rigid_object(
                    next_obj_id,
                    pattern=pattern,
                    asset_bank=asset_bank,
                    motion_request={
                        "forced_mode": forced_mode,
                        "occupied_slots": occupied_slots,
                    },
                )
            )
            next_obj_id += 1

    elif pattern == "strike_static":
        # 主静止目标（必须有）
        target_obj = sample_rigid_object(
            next_obj_id,
            pattern=pattern,
            asset_bank=asset_bank,
            motion_request={
                "forced_mode": "static_rest",
                "occupied_slots": occupied_slots,
            },
        )
        objects.append(target_obj)
        next_obj_id += 1

        # 撞击者（必须有）
        strike_mode = random.choice(["strike_static_left", "strike_static_right", "front_slide_in"])
        strike_motion_request = {
            "forced_mode": strike_mode,
            "occupied_slots": occupied_slots,
        }
        if strike_mode.startswith("strike_static"):
            strike_motion_request["target_pos"] = target_obj["init_pos"]

        striker_obj = sample_rigid_object(
            next_obj_id,
            pattern=pattern,
            asset_bank=asset_bank,
            motion_request=strike_motion_request,
        )
        objects.append(striker_obj)
        next_obj_id += 1

        # 再额外随机加一些“原本就静止”的物体
        add_static_props(extra_static_count)

        # 剩余物体继续作为动态干扰项
        while len(objects) < n_obj:
            forced_mode = random.choice([
                "top_drop",
                "top_toss",
                "front_slide_in",
                "diagonal_corner_left",
                "diagonal_corner_right",
            ])
            objects.append(
                sample_rigid_object(
                    next_obj_id,
                    pattern=pattern,
                    asset_bank=asset_bank,
                    motion_request={
                        "forced_mode": forced_mode,
                        "occupied_slots": occupied_slots,
                    },
                )
            )
            next_obj_id += 1

    elif pattern == "chain_reaction":
        # 两个主静止目标（必须有）
        first_target = sample_rigid_object(
            next_obj_id,
            pattern=pattern,
            asset_bank=asset_bank,
            motion_request={
                "forced_mode": "static_rest",
                "occupied_slots": occupied_slots,
            },
        )
        objects.append(first_target)
        next_obj_id += 1

        second_target = sample_rigid_object(
            next_obj_id,
            pattern=pattern,
            asset_bank=asset_bank,
            motion_request={
                "forced_mode": "static_rest",
                "occupied_slots": occupied_slots,
            },
        )
        objects.append(second_target)
        next_obj_id += 1

        avg_target = (
            (first_target["init_pos"][0] + second_target["init_pos"][0]) * 0.5,
            (first_target["init_pos"][1] + second_target["init_pos"][1]) * 0.5,
            (first_target["init_pos"][2] + second_target["init_pos"][2]) * 0.5,
        )

        # 撞击者（必须有）
        striker_obj = sample_rigid_object(
            next_obj_id,
            pattern=pattern,
            asset_bank=asset_bank,
            motion_request={
                "forced_mode": random.choice(["strike_static_left", "strike_static_right"]),
                "target_pos": avg_target,
                "occupied_slots": occupied_slots,
            },
        )
        objects.append(striker_obj)
        next_obj_id += 1

        # 再额外加一些本来静止在地上的物体
        add_static_props(extra_static_count)

        # 剩余动态物体
        while len(objects) < n_obj:
            forced_mode = random.choice([
                "top_drop",
                "top_toss",
                "front_slide_in",
                "diagonal_corner_left",
                "diagonal_corner_right",
            ])
            objects.append(
                sample_rigid_object(
                    next_obj_id,
                    pattern=pattern,
                    asset_bank=asset_bank,
                    motion_request={
                        "forced_mode": forced_mode,
                        "occupied_slots": occupied_slots,
                    },
                )
            )
            next_obj_id += 1

    else:
        raise ValueError(f"Unknown rigid pattern: {pattern}")

    return objects


def sample_mpm_object(obj_id: int):
    shape = random.choice(["box", "sphere"])
    material = sample_mpm_material()

    if shape == "box":
        geom = {
            "shape": "box",
            "size": [
                float(np.random.uniform(0.10, 0.18)),
                float(np.random.uniform(0.10, 0.18)),
                float(np.random.uniform(0.10, 0.18)),
            ],
        }
        half_x = geom["size"][0] / 2.0
        half_y = geom["size"][1] / 2.0
        half_z = geom["size"][2] / 2.0
    else:
        geom = {
            "shape": "sphere",
            "radius": float(np.random.uniform(0.06, 0.10)),
        }
        half_x = geom["radius"]
        half_y = geom["radius"]
        half_z = geom["radius"]

    x, y = sample_spawn_xy(half_x + 0.03, half_y + 0.03, bias_to_back=True)
    z = float(np.random.uniform(0.30, 0.62))
    z = max(z, CONTAINER["floor_thickness"] + half_z + 0.05)

    return {
        "object_id": obj_id,
        "solver": "MPM",
        "source_type": "procedural",
        "geom": geom,
        "material": material,
        "init_pos": [float(x), float(y), float(z)],
        "color": sample_color(),
        "surface_vis_mode": random.choice(["visual", "particle"]),
    }


def sample_sph_object(obj_id: int):
    material = sample_sph_material()
    size = [
        float(np.random.uniform(0.18, 0.28)),
        float(np.random.uniform(0.18, 0.28)),
        float(np.random.uniform(0.14, 0.22)),
    ]
    half_x = size[0] / 2.0
    half_y = size[1] / 2.0
    half_z = size[2] / 2.0

    x, y = sample_spawn_xy(half_x + 0.03, half_y + 0.03, bias_to_back=True)
    z = float(np.random.uniform(0.34, 0.58))
    z = max(z, CONTAINER["floor_thickness"] + half_z + 0.03)

    return {
        "object_id": obj_id,
        "solver": "SPH",
        "source_type": "procedural",
        "geom": {
            "shape": "box",
            "size": size,
        },
        "material": material,
        "init_pos": [float(x), float(y), float(z)],
        "color": sample_color(),
        "surface_vis_mode": "particle",
    }



def sample_scene_cfg(
    scene_id: int, 
    scene_plan: dict, 
    asset_bank=None, 
    rigid_sim_cfg=None):
    seed = 100000 + scene_id
    set_seed(seed)
    rigid_sim_cfg = rigid_sim_cfg or {}

    bg = sample_background()
    cam = sample_camera(CONTAINER)
    # Keep renderer resolution aligned with export config by default.
    # The exporter still has a runtime shape guard for safety.
    cam["res"] = [int(IMG_W), int(IMG_H)]

    category_spec = scene_plan["category_spec"]
    scene_bundle = build_rigid_scene_from_category(
        category_spec=category_spec,
        object_count=scene_plan["object_count"],
        asset_bank=asset_bank,
    )
    user_target_numsteps = rigid_sim_cfg.get("target_numsteps")
    if user_target_numsteps is None:
        user_target_numsteps = rigid_sim_cfg.get("num_steps")
    sim_steps = resolve_sim_num_steps(
        dt=float(rigid_sim_cfg.get("dt", 4e-3)),
        target_seconds=rigid_sim_cfg.get("target_seconds", None),
        target_numsteps=user_target_numsteps,
    )
    object_count_bucket = scene_bundle["object_count_bucket"]

    if object_count_bucket is None:
        count_dir = f"count_mixed_s{scene_bundle['num_static_objects']}_d{scene_bundle['num_dynamic_objects']}"
        scene_name_suffix = f"s{scene_bundle['num_static_objects']}_d{scene_bundle['num_dynamic_objects']}"
    else:
        count_dir = f"count_{object_count_bucket:02d}"
        scene_name_suffix = f"n{object_count_bucket:02d}"

    scene_id_str = f"{category_spec['name']}__{scene_name_suffix}__sample_{scene_plan['sample_index']:04d}"
    output_relpath = Path("train") / category_spec["name"] / count_dir / scene_id_str

    return {
        "scene_id": scene_id_str,
        "seed": seed,
        "family": "rigid_only",
        "rigid_pattern": category_spec["name"],
        "rigid_motion_category": category_spec["name"],
        "rigid_motion_label_zh": category_spec.get("label_zh", category_spec["name"]),
        "scene_builder": category_spec["scene_builder"],
        "object_count_bucket": object_count_bucket,
        "sample_index_in_bucket": int(scene_plan["sample_index"]),
        "num_static_objects": int(scene_bundle["num_static_objects"]),
        "num_dynamic_objects": int(scene_bundle["num_dynamic_objects"]),
        "motion_modes_present": scene_bundle["motion_modes_present"],
        "hybrid_counts": scene_bundle.get("hybrid_counts", None),
        "output_relpath": str(output_relpath),
        "background": bg,
        "container": CONTAINER,
        "camera": cam,
        "sim_options": {
            "gravity": [0.0, 0.0, -9.81],
            "dt": float(rigid_sim_cfg.get("dt", 4e-3)),
            "substeps": int(rigid_sim_cfg.get("substeps", 8)),
            "num_steps": sim_steps,
        },
        "target_seconds": rigid_sim_cfg.get("target_seconds", None),
        "target_numsteps": int(user_target_numsteps) if user_target_numsteps is not None else None,
        "objects": scene_bundle["objects"],
    }

def add_background_set(scene, bg_cfg):
    # 远处地面
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(3.8, 3.8, 0.03),
            pos=(0.0, 0.25, -0.015),
            fixed=True,
        ),
        material=gs.materials.Rigid(rho=1200.0, friction=0.9),
        surface=gs.surfaces.Default(color=(0.82, 0.82, 0.84, 1.0)),
    )

    # 背景大挡板
    panel_color_bank = [
        (0.86, 0.88, 0.92, 1.0),
        (0.92, 0.90, 0.86, 1.0),
        (0.80, 0.86, 0.90, 1.0),
        (0.25, 0.25, 0.28, 1.0),
    ]
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(3.2, 0.05, 1.8),
            pos=(0.0, BACKGROUND_PANEL_Y, 0.9),
            fixed=True,
        ),
        material=gs.materials.Rigid(rho=1200.0, friction=0.95),
        surface=gs.surfaces.Default(color=random.choice(panel_color_bank)),
    )

    # 左右辅助挡板
    for sign in (-1.0, 1.0):
        scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.05, 2.5, 1.4),
                pos=(sign * BACKGROUND_SIDE_X, 0.25, 0.7),
                fixed=True,
            ),
            material=gs.materials.Rigid(rho=1200.0, friction=0.95),
            surface=gs.surfaces.Default(color=(0.78, 0.80, 0.84, 1.0)),
        )

    # 随机背景道具
    for k in range(bg_cfg.get("n_props", 0)):
        shape = random.choice(["box", "sphere", "cylinder"])
        x = float(np.random.uniform(-0.95, 0.95))
        y = float(np.random.uniform(0.78, 1.35))
        z = float(np.random.uniform(*BACKGROUND_Z_RANGE))
        color = tuple(sample_color())

        if shape == "box":
            size = (
                float(np.random.uniform(0.10, 0.30)),
                float(np.random.uniform(0.08, 0.18)),
                float(np.random.uniform(0.12, 0.35)),
            )
            scene.add_entity(
                morph=gs.morphs.Box(size=size, pos=(x, y, z + size[2] / 2.0), fixed=True),
                material=gs.materials.Rigid(rho=900.0, friction=0.95),
                surface=gs.surfaces.Default(color=color),
            )

        elif shape == "sphere":
            r = float(np.random.uniform(0.06, 0.14))
            scene.add_entity(
                morph=gs.morphs.Sphere(radius=r, pos=(x, y, z + r), fixed=True),
                material=gs.materials.Rigid(rho=900.0, friction=0.95),
                surface=gs.surfaces.Default(color=color),
            )

        else:
            r = float(np.random.uniform(0.05, 0.11))
            h = float(np.random.uniform(0.12, 0.30))
            scene.add_entity(
                morph=gs.morphs.Cylinder(radius=r, height=h, pos=(x, y, z + h / 2.0), fixed=True),
                material=gs.materials.Rigid(rho=900.0, friction=0.95),
                surface=gs.surfaces.Default(color=color),
            )
def build_scene(scene_cfg: dict):
    vis_options = gs.options.VisOptions(
        show_world_frame=False,
        show_link_frame=False,
        background_color=tuple(scene_cfg["background"]["background_color"]),
        ambient_light=tuple(scene_cfg["background"]["ambient_light"]),
        segmentation_level="entity",
        render_particle_as="sphere",
        particle_size_scale=1.0,
    )

    sim_options = gs.options.SimOptions(
        gravity=tuple(scene_cfg["sim_options"]["gravity"]),
        dt=scene_cfg["sim_options"]["dt"],
        substeps=scene_cfg["sim_options"]["substeps"],
    )

    family = scene_cfg["family"]
    if family != "rigid_only":
        raise ValueError(f"dataset_3_rigid_genesis only supports rigid_only scenes, got family={family}")

    scene_kwargs = dict(
        sim_options=sim_options,
        vis_options=vis_options,
        show_viewer=False,
    )

    try:
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(
            dt=scene_cfg["sim_options"]["dt"],
            enable_collision=True,
            use_gjk_collision=True,
        )
    except Exception:
        try:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(
                dt=scene_cfg["sim_options"]["dt"],
            )
        except Exception:
            pass

    scene = gs.Scene(**scene_kwargs)

    # add_background_set(scene, scene_cfg["background"])
    container_entities = add_container(gs, scene, scene_cfg["container"])

    def _add_fallback_rigid_entity(obj, mat, pos, euler, surface_override=None):
        geom = obj.get("geom", {})
        bound_r = float(geom.get("bound_radius", 0.10))
        bound_r = max(0.035, min(0.24, bound_r))
        fallback_surface = surface_override
        if fallback_surface is None:
            fallback_color = obj.get("color", None)
            if fallback_color is None:
                fallback_color = sample_color()
            fallback_surface = gs.surfaces.Default(
                color=tuple(fallback_color),
                vis_mode=obj.get("surface_vis_mode", "visual"),
            )
        try:
            return scene.add_entity(
                morph=gs.morphs.Sphere(radius=bound_r, pos=pos, euler=euler),
                material=mat,
                surface=fallback_surface,
            )
        except Exception:
            side = max(0.05, bound_r * 1.25)
            return scene.add_entity(
                morph=gs.morphs.Box(size=(side, side, side), pos=pos, euler=euler),
                material=mat,
                surface=fallback_surface,
            )

    entities = []

    for obj in scene_cfg["objects"]:
        ent = None

        is_textured_dataset_mesh = (
            obj["solver"] == "Rigid"
            and obj.get("source_type") == "dataset_mesh"
            and obj["geom"]["shape"] == "mesh"
            and obj["geom"].get("use_texture", False)
        )
        is_part_colored_urdf = (
            obj["solver"] == "Rigid"
            and obj.get("source_type") == "dataset_mesh"
            and obj["geom"]["shape"] == "urdf"
            and obj["geom"].get("use_urdf_material", False)
        )

        surface = None
        if not is_textured_dataset_mesh and not is_part_colored_urdf:
            surface = gs.surfaces.Default(
                color=tuple(obj["color"]),
                vis_mode=obj.get("surface_vis_mode", "visual"),
            )

        if obj["solver"] == "Rigid":
            mat = create_genesis_rigid_material(gs, obj["material"])
            euler = tuple(obj["init_euler"])
            pos = tuple(obj["init_pos"])

            if obj["geom"]["shape"] == "mesh":
                kwargs = dict(
                    morph=gs.morphs.Mesh(
                        file=obj["geom"]["mesh_file"],
                        scale=obj["geom"].get("scale", 1.0),
                        pos=pos,
                        euler=euler,
                    ),
                    material=mat,
                )
                if surface is not None:
                    kwargs["surface"] = surface
                try:
                    ent = scene.add_entity(**kwargs)
                except Exception as mesh_err:
                    print(
                        f"[WARN] fallback primitive for object_id={obj.get('object_id')} "
                        f"asset={obj['geom'].get('asset_id')} shape=mesh err={mesh_err}"
                    )
                    ent = _add_fallback_rigid_entity(obj, mat, pos, euler, surface_override=surface)

            elif obj["geom"]["shape"] == "urdf":
                urdf_kwargs = dict(
                    file=obj["geom"]["urdf_file"],
                    scale=obj["geom"].get("scale", 1.0),
                    pos=pos,
                    euler=euler,
                    visualization=True,
                    collision=True,
                    fixed=False,
                    merge_fixed_links=True,
                    prioritize_urdf_material=obj["geom"].get("use_urdf_material", False),
                )
                kwargs = dict(
                    morph=gs.morphs.URDF(**urdf_kwargs),
                    material=mat,
                )
                if surface is not None:
                    kwargs["surface"] = surface
                try:
                    try:
                        ent = scene.add_entity(**kwargs)
                    except TypeError:
                        fallback_keys = ["file", "scale", "pos", "euler", "visualization", "collision", "fixed"]
                        kwargs["morph"] = gs.morphs.URDF(**{k: urdf_kwargs[k] for k in fallback_keys})
                        ent = scene.add_entity(**kwargs)
                except Exception as urdf_err:
                    print(
                        f"[WARN] fallback primitive for object_id={obj.get('object_id')} "
                        f"asset={obj['geom'].get('asset_id')} shape=urdf err={urdf_err}"
                    )
                    ent = _add_fallback_rigid_entity(obj, mat, pos, euler, surface_override=surface)

            elif obj["geom"]["shape"] == "box":
                ent = scene.add_entity(
                    morph=gs.morphs.Box(size=tuple(obj["geom"]["size"]), pos=pos, euler=euler),
                    material=mat,
                    surface=surface,
                )

            elif obj["geom"]["shape"] == "sphere":
                ent = scene.add_entity(
                    morph=gs.morphs.Sphere(radius=obj["geom"]["radius"], pos=pos, euler=euler),
                    material=mat,
                    surface=surface,
                )

            elif obj["geom"]["shape"] == "cylinder":
                ent = scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        radius=obj["geom"]["radius"],
                        height=obj["geom"]["height"],
                        pos=pos,
                        euler=euler,
                    ),
                    material=mat,
                    surface=surface,
                )
            elif obj["geom"]["shape"] == "capsule":
                try:
                    ent = scene.add_entity(
                        morph=gs.morphs.Capsule(
                            radius=obj["geom"]["radius"],
                            height=obj["geom"]["height"],
                            pos=pos,
                            euler=euler,
                        ),
                        material=mat,
                        surface=surface,
                    )
                except Exception:
                    ent = scene.add_entity(
                        morph=gs.morphs.Cylinder(
                            radius=obj["geom"]["radius"],
                            height=obj["geom"]["height"] + 2.0 * obj["geom"]["radius"],
                            pos=pos,
                            euler=euler,
                        ),
                        material=mat,
                        surface=surface,
                    )
            else:
                raise ValueError(obj["geom"]["shape"])

        else:
            raise ValueError(f"dataset_3_rigid_genesis only supports rigid objects, got solver={obj['solver']}")

        entities.append(ent)

    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )

    scene.build()

    # rigid 初始速度
    for obj, ent in zip(scene_cfg["objects"], entities):
        if obj["solver"] == "Rigid":
            apply_initial_motion_to_entity(ent, obj["init_linvel"], obj["init_angvel"])

    return scene, cam, entities, container_entities




def prepare_output_dirs(out_dir: Path):
    subdirs = ["rgb", "depth", "videos", "physics"]
    if EXPORT_ENHANCED_RGB:
        subdirs.append(ENHANCED_RGB_DIRNAME)
    for s in subdirs:
        ensure_dir(out_dir / s)


def write_dataset_format_description(dataset_root: Path):
    doc = f"""# Genesis Sim Merge Dataset Format

## Coordinate Convention

- World frame: `z-up`, right-handed.
- External assets: default `y-up` on input, converted at runtime with base Euler `[{YUP_TO_ZUP_EULER_XYZ[0]:.6f}, {YUP_TO_ZUP_EULER_XYZ[1]:.6f}, {YUP_TO_ZUP_EULER_XYZ[2]:.6f}]` (XYZ radians) when the dataset source is marked as `y_up`.
- Camera point clouds and object trajectories are exported in world coordinates.

## Directory Layout

Each scene lives at `train/<motion_category>/<count_bucket>/<scene_id>/` and contains:

- `scene_input.json`: sampled scene config before simulation.
- `scene_metadata.json`: simulation summary, material summary, coordinate-transform summary, and export schema.
- `rgb/<frame>.png`: RGB image.
- `rgb_enhanced/<frame>.png`: optional edge-enhanced RGB visualization for human inspection.
- `depth/<frame>.npy`: depth map in meters, shape `(H, W)`.
- `depth_vis/<frame>.png`: visualization of the depth map.
- `segmentation/<frame>.npy`: segmentation buffer from Genesis, shape depends on renderer output.
- `normal/<frame>.npy`: world-space normal image, shape `(H, W, 3)`.
- `pointcloud/<frame>.npz`: scene point cloud in world frame with keys `xyz`, `mask`.
- `object_pointcloud/<frame>_obj<id>.npz`: per-object 3D point cloud trajectory sample in world frame with keys `xyz`, `object_id`, `frame`, `solver`, `centroid`, `quat`, `vel`, `ang`, `n_points_raw`, `n_points_saved`, `coordinate_frame`.
- `trajectories/objects_world.csv`: object centroid / quaternion / velocity / angular velocity trajectory.
- `trajectories/frame_index.csv`: per-frame file index for RGB / depth / segmentation / normals / scene point cloud.
- `trajectories/object_pointcloud_index.csv`: per-frame file index for object-level 3D point-cloud trajectories.
- `camera/intrinsics.npy`: camera intrinsics.
- `camera/extrinsics.npy`: camera extrinsics.
- `videos/rgb.mp4`: raw RGB video.
- `videos/rgb_enhanced.mp4`: optional edge-enhanced RGB video for part visibility.

## Core Files

### `trajectories/objects_world.csv`

Columns:

- `frame`: frame index after one `scene.step()`.
- `object_id`: object id in `scene_input.json`.
- `solver`: solver family, e.g. `Rigid`, `MPM`, `SPH`, `PBD`.
- `cx, cy, cz`: object centroid in world frame.
- `qx, qy, qz, qw`: object orientation quaternion in world frame if available.
- `vx, vy, vz`: linear velocity in world frame if available.
- `wx, wy, wz`: angular velocity in world frame if available.
- `n_points`: raw number of geometry points returned by the solver for this frame.

### `trajectories/object_pointcloud_index.csv`

Each row points to one exported object point-cloud sample:

- `frame`, `object_id`, `solver`
- `pointcloud_path`: filename under `object_pointcloud/`
- `cx, cy, cz`: centroid in world frame
- `qx, qy, qz, qw`: quaternion in world frame if available
- `vx, vy, vz`: linear velocity in world frame if available
- `wx, wy, wz`: angular velocity in world frame if available
- `n_points_raw`: solver-returned raw point count
- `n_points_saved`: count after subsampling to `MAX_OBJECT_PC={MAX_OBJECT_PC}`
- `coordinate_frame`: currently always `world`

The ordered sequence of `object_pointcloud/*.npz` files for the same `object_id` forms that object's 3D point-cloud trajectory.

### `pointcloud/<frame>.npz`

- `xyz`: scene-level point cloud in world frame
- `mask`: renderer validity mask

### `depth/<frame>.npy`

- Float depth map in meters.
- Invalid pixels follow the renderer output and should be filtered with `np.isfinite(depth)`.

## Notes

- `frame_index.csv` stores filenames relative to the scene subfolders, not absolute paths.
- Bounding-box statistics in metadata are written in scene `z-up` convention for sampling consistency.
- Local mesh / URDF asset files keep their original mesh-local coordinates; the runtime base Euler records the Y-up to Z-up conversion explicitly.
"""
    (dataset_root / "DATASET_FORMAT.md").write_text(doc, encoding="utf-8")


def export_entity_state(ent, obj_meta):
    state = {
        "object_id": obj_meta["object_id"],
        "solver": obj_meta["solver"],
        "centroid": None,
        "quat": None,
        "vel": None,
        "ang": None,
        "pointcloud": None,
        "n_points": 0,
    }

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

    if state["centroid"] is None and hasattr(ent, "get_particles_pos"):
        try:
            pts = to_numpy(ent.get_particles_pos())
            if pts is not None and pts.size > 0:
                pts = pts.reshape(-1, 3)
                state["pointcloud"] = pts
                state["centroid"] = pts.mean(axis=0)
                state["n_points"] = int(len(pts))
                return state
        except Exception:
            pass

    if state["centroid"] is None and hasattr(ent, "get_verts"):
        try:
            verts = to_numpy(ent.get_verts())
            if verts is not None and verts.size > 0:
                verts = verts.reshape(-1, 3)
                state["pointcloud"] = verts
                state["centroid"] = verts.mean(axis=0)
                state["n_points"] = int(len(verts))
        except Exception:
            pass

    return state


def get_entity_mass(ent, obj_meta) -> float:
    if hasattr(ent, "get_mass"):
        try:
            mass = float(ent.get_mass())
            if np.isfinite(mass) and mass > 0:
                return mass
        except Exception:
            pass
    return float(obj_meta.get("material", {}).get("rho", 1.0))


def collect_physics_state(entities, objects):
    positions = np.full((len(objects), 3), np.nan, dtype=np.float32)
    velocities = np.full((len(objects), 3), np.nan, dtype=np.float32)
    momenta = np.full((len(objects), 3), np.nan, dtype=np.float32)
    masses = np.full((len(objects),), np.nan, dtype=np.float32)

    for idx, (obj_meta, ent) in enumerate(zip(objects, entities)):
        state = export_entity_state(ent, obj_meta)
        mass = get_entity_mass(ent, obj_meta)
        masses[idx] = mass
        if state["centroid"] is not None:
            positions[idx] = np.asarray(state["centroid"], dtype=np.float32)[:3]
        if state["vel"] is not None:
            velocities[idx] = np.asarray(state["vel"], dtype=np.float32)[:3]
            momenta[idx] = velocities[idx] * mass

    return positions, velocities, momenta, masses


def contact_is_active(entity_a, entity_b) -> bool:
    try:
        contacts = entity_a.get_contacts(with_entity=entity_b)
        penetration = np.asarray(to_numpy(contacts.get("penetration", np.zeros((0,), dtype=np.float32))), dtype=np.float32)
        return bool(penetration.size > 0)
    except Exception:
        return False


def build_collision_monitors(scene_cfg: dict, entities):
    monitors = []
    objects = scene_cfg["objects"]
    for idx_a in range(len(objects)):
        for idx_b in range(idx_a + 1, len(objects)):
            monitors.append(
                {
                    "pair_name": f"obj{objects[idx_a]['object_id']}_obj{objects[idx_b]['object_id']}",
                    "entity_a": entities[idx_a],
                    "entity_b": entities[idx_b],
                    "object_idx_a": idx_a,
                    "object_idx_b": idx_b,
                    "object_id_a": int(objects[idx_a]["object_id"]),
                    "object_id_b": int(objects[idx_b]["object_id"]),
                }
            )
    return monitors


def update_collision_events(monitors, previous_active, pre_momenta, post_momenta, frame_idx, events):
    for monitor in monitors:
        is_active = contact_is_active(monitor["entity_a"], monitor["entity_b"])
        was_active = bool(previous_active.get(monitor["pair_name"], False))
        if is_active and not was_active:
            delta_pair = np.zeros((2, 3), dtype=np.float32)
            delta_pair[0] = post_momenta[monitor["object_idx_a"]] - pre_momenta[monitor["object_idx_a"]]
            delta_pair[1] = post_momenta[monitor["object_idx_b"]] - pre_momenta[monitor["object_idx_b"]]
            events.append(
                {
                    "pair_name": monitor["pair_name"],
                    "frame_idx": int(frame_idx),
                    "object_ids": [monitor["object_id_a"], monitor["object_id_b"]],
                    "object_indices": [monitor["object_idx_a"], monitor["object_idx_b"]],
                    "delta_p": delta_pair.tolist(),
                }
            )
        previous_active[monitor["pair_name"]] = is_active


def export_scene(scene_cfg: dict):
    out_dir = DATASET_ROOT / Path(scene_cfg.get("output_relpath", str(Path("train") / scene_cfg["scene_id"])))
    prepare_output_dirs(out_dir)

    with open(out_dir / "scene_input.json", "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, ensure_ascii=False, indent=2)

    scene, cam, entities, container_entities = None, None, None, None

    try:
        scene, cam, entities, container_entities = build_scene(scene_cfg)
        objects = list(scene_cfg["objects"])
        num_frames = int(EXPORT_FRAMES)
        planned_steps = int(scene_cfg["sim_options"]["num_steps"])
        steps_per_frame = max(
            int(EXPORT_STEPS_PER_FRAME),
            int(math.ceil(planned_steps / float(max(1, num_frames - 1)))),
        )
        num_steps = steps_per_frame * max(1, num_frames - 1)
        sim_dt = float(scene_cfg["sim_options"]["dt"])
        physics_duration_s = float(num_steps) * sim_dt

        rgb_frames = None
        enhanced_rgb_frames = None
        depth_frames = None
        trajectory = np.empty((num_frames, len(objects), 6), dtype=np.float32)
        masses = np.empty((len(objects),), dtype=np.float32)
        rigid_com_frames: List[np.ndarray] = []
        rigid_linear_vel_frames: List[np.ndarray] = []
        rigid_angular_vel_frames: List[np.ndarray] = []
        rigid_kinetic_frames: List[np.float32] = []
        rigid_potential_frames: List[np.float32] = []
        rigid_total_frames: List[np.float32] = []

        monitors = build_collision_monitors(scene_cfg, entities)
        previous_active = {monitor["pair_name"]: False for monitor in monitors}
        collision_events = []
        step_counter = 0
        gravity_vec = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        for frame_idx in range(num_frames):
            positions, velocities, momenta, masses = collect_physics_state(entities, objects)
            frame_com = []
            frame_linear_vel = []
            frame_angular_vel = []
            frame_kinetic = 0.0
            frame_potential = 0.0
            for obj_meta, ent, mass in zip(objects, entities, masses):
                snap = rigid_entity_kinematic_snapshot(ent, gravity=gravity_vec)
                frame_com.append(np.asarray(snap.com_pos, dtype=np.float32))
                frame_linear_vel.append(np.asarray(snap.linear_vel, dtype=np.float32))
                frame_angular_vel.append(np.asarray(snap.angular_vel, dtype=np.float32))
                frame_kinetic += float(snap.kinetic)
                frame_potential += -float(mass) * float(np.dot(gravity_vec, np.asarray(snap.com_pos, dtype=np.float64)))
            rigid_com_frames.append(np.stack(frame_com, axis=0).astype(np.float32))
            rigid_linear_vel_frames.append(np.stack(frame_linear_vel, axis=0).astype(np.float32))
            rigid_angular_vel_frames.append(np.stack(frame_angular_vel, axis=0).astype(np.float32))
            rigid_kinetic_frames.append(np.float32(frame_kinetic))
            rigid_potential_frames.append(np.float32(frame_potential))
            rigid_total_frames.append(np.float32(frame_kinetic + frame_potential))
            rendered = cam.render(rgb=True, depth=True, segmentation=False, normal=False)
            if not isinstance(rendered, tuple) or len(rendered) < 2:
                raise RuntimeError("Unexpected camera render output.")
            rgb_raw, depth_raw = rendered[0], rendered[1]
            rgb_frame = rgb_to_uint8(rgb_raw)
            depth_frame = normalize_depth_map(
                depth_raw,
                near=float(getattr(cam, "near", 0.05)),
                far=float(getattr(cam, "far", 50.0)),
            )
            enhanced_rgb_frame = enhance_part_visibility(rgb_frame, depth_frame) if EXPORT_ENHANCED_RGB else None
            if rgb_frame.ndim != 3 or rgb_frame.shape[-1] != 3:
                raise RuntimeError(f"Unexpected RGB frame shape: {rgb_frame.shape}")
            if depth_frame.ndim != 3 or depth_frame.shape[-1] != 1:
                raise RuntimeError(f"Unexpected depth frame shape: {depth_frame.shape}")
            if depth_frame.shape[:2] != rgb_frame.shape[:2]:
                raise RuntimeError(
                    f"RGB/depth resolution mismatch: rgb={rgb_frame.shape[:2]}, depth={depth_frame.shape[:2]}"
                )
            if rgb_frames is None:
                frame_h, frame_w = rgb_frame.shape[:2]
                rgb_frames = np.empty((num_frames, frame_h, frame_w, 3), dtype=np.uint8)
                depth_frames = np.empty((num_frames, frame_h, frame_w, 1), dtype=np.float32)
                if EXPORT_ENHANCED_RGB:
                    enhanced_rgb_frames = np.empty((num_frames, frame_h, frame_w, 3), dtype=np.uint8)
            rgb_frames[frame_idx] = rgb_frame
            depth_frames[frame_idx] = depth_frame
            if enhanced_rgb_frames is not None and enhanced_rgb_frame is not None:
                enhanced_rgb_frames[frame_idx] = enhanced_rgb_frame
            trajectory[frame_idx, :, :3] = positions
            trajectory[frame_idx, :, 3:] = momenta

            if frame_idx == num_frames - 1:
                break

            for _ in range(steps_per_frame):
                _, _, pre_momenta, _ = collect_physics_state(entities, objects)
                scene.step()
                step_counter += 1
                _, _, post_momenta, _ = collect_physics_state(entities, objects)
                output_frame_idx = min(num_frames - 1, int(math.ceil(step_counter / float(steps_per_frame))))
                update_collision_events(monitors, previous_active, pre_momenta, post_momenta, output_frame_idx, collision_events)

        if rgb_frames is None or depth_frames is None:
            raise RuntimeError("No frames were rendered; cannot export scene outputs.")

        for frame_idx, rgb_frame in enumerate(rgb_frames):
            imageio.imwrite(out_dir / "rgb" / f"frame_{frame_idx:03d}.png", rgb_frame)
        if enhanced_rgb_frames is not None:
            for frame_idx, rgb_frame in enumerate(enhanced_rgb_frames):
                imageio.imwrite(out_dir / ENHANCED_RGB_DIRNAME / f"frame_{frame_idx:03d}.png", rgb_frame)
        for frame_idx, depth_frame in enumerate(depth_frames):
            imageio.imwrite(out_dir / "depth" / f"frame_{frame_idx:03d}.png", depth_to_uint8(depth_frame))

        fps = max(1, int(round(1.0 / max(sim_dt * steps_per_frame, 1e-8))))
        imageio.mimsave(out_dir / "videos" / "rgb.mp4", list(rgb_frames), fps=fps)
        if enhanced_rgb_frames is not None:
            imageio.mimsave(out_dir / "videos" / ENHANCED_RGB_VIDEO_NAME, list(enhanced_rgb_frames), fps=fps)
        imageio.mimsave(out_dir / "videos" / "depth.mp4", [depth_to_uint8(frame) for frame in depth_frames], fps=fps)

        np.save(out_dir / "physics" / "trajectory.npy", trajectory.astype(np.float32))
        np.save(out_dir / "physics" / "depth_normalized.npy", depth_frames.astype(np.float32))
        np.savez_compressed(
            out_dir / "physics" / "rigid_kinematics.npz",
            com_pos=np.stack(rigid_com_frames, axis=0).astype(np.float32),
            linear_vel=np.stack(rigid_linear_vel_frames, axis=0).astype(np.float32),
            angular_vel=np.stack(rigid_angular_vel_frames, axis=0).astype(np.float32),
            kinetic_energy=np.asarray(rigid_kinetic_frames, dtype=np.float32),
            potential_energy=np.asarray(rigid_potential_frames, dtype=np.float32),
            total_energy=np.asarray(rigid_total_frames, dtype=np.float32),
        )

        properties_payload = {
            "object_ids": [int(obj["object_id"]) for obj in objects],
            "mass": masses.astype(np.float32).tolist(),
            "restitution": [float(obj.get("material", {}).get("restitution", 0.0)) for obj in objects],
            "friction": [float(obj.get("material", {}).get("friction", 0.0)) for obj in objects],
            "impulse_vector": [
                (np.asarray(obj.get("init_linvel", [0.0, 0.0, 0.0]), dtype=np.float32) * masses[idx]).tolist()
                for idx, obj in enumerate(objects)
            ],
            "init_linear_velocity": [
                np.asarray(obj.get("init_linvel", [0.0, 0.0, 0.0]), dtype=np.float32).tolist()
                for obj in objects
            ],
            "init_angular_velocity": [
                np.asarray(obj.get("init_angvel", [0.0, 0.0, 0.0]), dtype=np.float32).tolist()
                for obj in objects
            ],
        }
        with open(out_dir / "physics" / "properties.json", "w", encoding="utf-8") as f:
            json.dump(properties_payload, f, ensure_ascii=False, indent=2)

        with open(out_dir / "physics" / "collision_events.json", "w", encoding="utf-8") as f:
            json.dump(collision_events, f, ensure_ascii=False, indent=2)

        scene_metadata = {
            "scene_id": scene_cfg["scene_id"],
            "output_relpath": scene_cfg.get("output_relpath"),
            "seed": scene_cfg["seed"],
            "family": scene_cfg["family"],
            "rigid_motion_category": scene_cfg.get("rigid_motion_category"),
            "object_count_bucket": scene_cfg.get("object_count_bucket"),
            "num_objects": len(objects),
            "num_dataset_mesh_objects": int(sum(1 for obj in objects if obj.get("source_type") == "dataset_mesh")),
            "frames": num_frames,
            "resolution": [int(rgb_frames.shape[2]), int(rgb_frames.shape[1])],
            "dt": sim_dt,
            "substeps": scene_cfg["sim_options"]["substeps"],
            "steps_per_frame": steps_per_frame,
            "sim_steps_planned": planned_steps,
            "sim_steps_executed": num_steps,
            "physics_duration_s": physics_duration_s,
            "video_fps": fps,
            "enhanced_rgb_export": bool(enhanced_rgb_frames is not None),
            "enhanced_rgb_video": f"videos/{ENHANCED_RGB_VIDEO_NAME}" if enhanced_rgb_frames is not None else None,
            "trajectory_definition": "COM position xyz + linear momentum xyz",
            "rigid_kinematics_npz": "physics/rigid_kinematics.npz",
            "energy_definition": "kinetic + gravitational potential",
            "status": "ok",
        }

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(scene_metadata, f, ensure_ascii=False, indent=2)

        return scene_metadata

    finally:
        safe_scene_destroy(scene)


# =========================
# 主程序
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate rigid-only Genesis dataset grouped by motion category and object count."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=SAMPLES_PER_CATEGORY,
        help="Number of samples to generate for each motion-category/count bucket.",
    )
    parser.add_argument(
        "--rigid-dt",
        type=float,
        default=4e-3,
        help="Rigid simulation dt.",
    )
    parser.add_argument(
        "--rigid-substeps",
        type=int,
        default=8,
        help="Rigid simulation substeps.",
    )
    parser.add_argument(
        "--target-numsteps",
        type=int,
        default=None,
        dest="target_numsteps",
        help="Total outer simulation steps. If both --target-seconds and --target-numsteps are set, steps win. "
        "At least one of --target-seconds or --target-numsteps is required (--rigid-num-steps counts as target_numsteps).",
    )
    parser.add_argument(
        "--rigid-num-steps",
        type=int,
        default=None,
        help="Alias for --target-numsteps. If both --target-numsteps and --rigid-num-steps are set, --target-numsteps wins.",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=None,
        help="Target physical duration in seconds (derives num_steps = round(seconds/dt)). Preview length when set. "
        "At least one of --target-seconds or --target-numsteps is required.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Only generate the first N planned scenes. Useful for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    samples_per_category = max(1, int(args.samples_per_category))
    target_numsteps = args.target_numsteps
    if target_numsteps is None:
        target_numsteps = args.rigid_num_steps
    if args.target_seconds is None and target_numsteps is None:
        target_numsteps = (EXPORT_FRAMES - 1) * EXPORT_STEPS_PER_FRAME
    rigid_sim_cfg = {
        "dt": float(args.rigid_dt),
        "substeps": int(args.rigid_substeps),
        "target_seconds": float(args.target_seconds) if args.target_seconds is not None else None,
    }
    if target_numsteps is not None:
        rigid_sim_cfg["target_numsteps"] = int(target_numsteps)
    scene_plan_list = build_scene_generation_plan(
        samples_per_category,
        RIGID_MOTION_CATEGORY_SPECS,
        DATASET_OBJECT_COUNTS
    )
    if args.max_scenes is not None:
        scene_plan_list = scene_plan_list[: max(0, int(args.max_scenes))]

    global DATASET_ROOT
    DATASET_ROOT = dataset_root

    ensure_dir(DATASET_ROOT)
    ensure_dir(DATASET_ROOT / "train")
    ensure_dir(DATASET_ROOT / "failed_configs")
    ensure_dir(ASSET_CACHE_DIR)
    write_dataset_format_description(DATASET_ROOT)

    asset_bank = build_asset_bank()

    backend_used = "gpu"
    try:
        gs.init(backend=gs.gpu)
        backend_used = "gpu"
    except Exception:
        gs.init(backend=gs.cpu)
        backend_used = "cpu"

    manifest = {
        "dataset_name": "genesis_sim_v4_rigid_motion_buckets",
        "split": "train",
        "n_scenes_requested": len(scene_plan_list),
        "samples_per_category": samples_per_category,
        "rigid_sim_cli": rigid_sim_cfg,
        "target_seconds": rigid_sim_cfg.get("target_seconds", None),
        "target_numsteps": rigid_sim_cfg.get("target_numsteps", None),
        "image_size": [IMG_W, IMG_H],
        "backend_used": backend_used,
        "scene_families": SCENE_FAMILY_WEIGHTS,
        "object_count_buckets": DATASET_OBJECT_COUNTS,
        "rigid_motion_category_specs": RIGID_MOTION_CATEGORY_SPECS,
        "dataset_source": DATASET_SOURCE,
        "source_dataset_roots": [str(x) for x in SOURCE_DATASET_ROOTS],
        "physx3d_root": str(PHYSX3D_ROOT),
        "physx3d_version": PHYSX3D_VERSION,
        "primitive_dataset_name": PRIMITIVE_DATASET_NAME,
        "primitive_asset_repeat": PRIMITIVE_ASSET_REPEAT,
        "primitive_shape_weights": PRIMITIVE_SHAPE_WEIGHTS,
        "primitive_material_weights": PRIMITIVE_MATERIAL_WEIGHTS,
        "use_dataset_mesh_objects": USE_DATASET_MESH_OBJECTS,
        "dataset_object_prob": DATASET_OBJECT_PROB,
        "n_usable_dataset_assets": len(asset_bank),
        "notes": [
            "Uses z-up convention.",
            "Container is a true three-sided open container: floor + left wall + right wall + back wall.",
            "This dataset only generates rigid-body simulations.",
            "Samples are organized by motion category and object-count bucket.",
            "Uniform dynamic buckets cover object counts 1, 2, 3, 4, 5, and 10.",
            "An extra hybrid category mixes 1-3 static ground objects with 1-3 dynamic objects.",
            "PhysX-3D objects can be rendered as single-rigid-body URDFs with per-part colors.",
            "Rigid objects can start with arbitrary xyz Euler angles instead of only near-flat poses.",
            "Single-scene failure is recorded instead of aborting the whole run.",
            "This script keeps the current scene definition and exposes dataset-source interfaces for SOPHY / PhysX-3D / primitive / all."
        ],
        "scenes": [],
        "failed_scenes": [],
    }

    try:
        for sid, scene_plan in enumerate(scene_plan_list):
            scene_cfg = sample_scene_cfg(
                sid,
                scene_plan=scene_plan,
                asset_bank=asset_bank,
                rigid_sim_cfg=rigid_sim_cfg,
            )

            try:
                print(
                    f"[RUN ] {scene_cfg['scene_id']} | category={scene_cfg['rigid_motion_category']} "
                    f"| count_bucket={scene_cfg['object_count_bucket']}"
                )
                meta = export_scene(scene_cfg)
                manifest["scenes"].append(meta)
                print(
                    f"[ OK ] {scene_cfg['scene_id']} | category={scene_cfg['rigid_motion_category']} "
                    f"| dataset_mesh={meta.get('num_dataset_mesh_objects', 0)}/{meta.get('num_objects', len(scene_cfg.get('objects', [])))}"
                )

            except Exception as e:
                err_info = {
                    "scene_id": scene_cfg["scene_id"],
                    "family": scene_cfg["family"],
                    "rigid_motion_category": scene_cfg.get("rigid_motion_category"),
                    "object_count_bucket": scene_cfg.get("object_count_bucket"),
                    "seed": scene_cfg["seed"],
                    "error": str(e),
                }
                manifest["failed_scenes"].append(err_info)

                with open(DATASET_ROOT / "failed_configs" / f"{scene_cfg['scene_id']}.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "scene_cfg": scene_cfg,
                            "error": str(e),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                print(f"[FAIL] {scene_cfg['scene_id']} | family={scene_cfg['family']} | err={e}")

                if STOP_ON_ERROR:
                    raise

    finally:
        with open(DATASET_ROOT / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        try:
            gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()

'''




'''













'''







1. substeps 的作用
- dt 是每个外层 simulation step 的物理时长。
- substeps 是每个外层 step 再细分成多少个子步。
- 因此子步长为：substep_dt = dt / substeps。
- substeps 调大时，单个子步更小，接触、碰撞、关节约束通常更稳定，更不容易穿透、抖动或出现 NaN；但计算也会更慢。:contentReference[oaicite:0]{index=0}

2. 物理仿真总时长
- 设总共推进 num_steps 个外层 step，则物理真实仿真时长为：
  T = dt × num_steps
- 例如 dt = 1e-3, num_steps = 320，则：
  T = 0.001 × 320 = 0.32 秒。:contentReference[oaicite:1]{index=1}

3. 视频帧数和时长
- 如果视频按 fps 播放，想得到 t 秒视频，至少需要：
  帧数 >= t × fps
- 例如 fps = 30，t = 5 秒，则至少需要：
  5 × 30 = 150 帧。

4. stride 的含义
- stride 表示每隔多少个 physics step 保存 1 帧视频。
- 因此大约可得到的视频帧数为：
  stride = 1/（fps x dt）= 1/(30 x 0.001) = 33
  num_steps = N_frames（150）x stride = 5000
 



'''


'''
刚体运动仿真，就算用了sophy和physxnet数据集但是好像没有将部件级材料参数用于仿真
可以用primitive普通数据集做一下简单物体仿真



python /home/gaoya/Code_Video/Code_data/1_localshow.py \
  --root /data/gaoya/AAA_test_video/Dataset_test/Genesis_rigid/train \
  --host 0.0.0.0 \
  --port 8001

• physics 目录当前导出 4 个文件：
  - trajectory.npy：形状是 (frames, num_objects, 6)，前 3 维是质心位置 xyz，后 3 维是线动量 px,py,pz（见 Code_Video/Code_data/dataset_3_rigid_genesis.py:3669, Code_Video/
    Code_data/dataset_3_rigid_genesis.py:3643）。
  - depth_normalized.npy：形状是 (frames, H, W, 1)，值为归一化深度 0~1（见 Code_Video/Code_data/dataset_3_rigid_genesis.py:3670）。
  - properties.json：object_ids / mass / restitution / friction / impulse_vector，其中 impulse_vector = init_linvel * mass（见 Code_Video/Code_data/
    dataset_3_rigid_genesis.py:3672）。
  - collision_events.json：碰撞事件列表；每条含 pair_name, frame_idx, object_ids, object_indices, delta_p（见 Code_Video/Code_data/dataset_3_rigid_genesis.py:3565, Code_Video/
    Code_data/dataset_3_rigid_genesis.py:3685）。




rm -rf /data/gaoya/AAA_test_video/Dataset_physV/Genesis_rigid
python /home/gaoya/Code_Video/Code_data/try3_dataset_3_rigid_genesis.py \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/Genesis_rigid \
  --samples-per-category 1


python3 /home/gaoya/Code_Video/Code_data/1_localshow.py \
  --root /data/gaoya/AAA_test_video/Dataset_physV/Genesis_rigid \
  --host 0.0.0.0 \
  --port 8000 \
  --title "Genesis Rigid Viewer" \
  --profile genesis_rigid
'''
