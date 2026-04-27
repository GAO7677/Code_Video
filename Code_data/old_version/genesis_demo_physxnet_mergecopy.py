import json
import csv
import math
import random
import colorsys
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import genesis as gs
import trimesh


# =========================
# 基本配置
# =========================
DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_test/genesis_sim_merge")
IMG_W, IMG_H = 640, 480
N_SCENES = 10

MAX_OBJECT_PC = 2048
OBJECT_PC_STRIDE = 5
CAMERA_PC_STRIDE = 2

ENABLE_CLOTH = False
CLOTH_MESH_PATH = None
STOP_ON_ERROR = False

# 可选强制场景类型 / rigid pattern。
# 例如：FORCE_SCENE_FAMILY = "rigid_mix"; FORCE_RIGID_PATTERN = None
FORCE_SCENE_FAMILY = None
FORCE_RIGID_PATTERN = None

SCENE_FAMILY_WEIGHTS = {
    "rigid_mix": 0.4,
    "mpm_mix": 0.3,
    "sph_liquid": 0.3,
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

TARGET_MESH_SIZE_RANGE = (0.2, 0.5)      # 最长边目标尺寸（米）
SIMPLIFY_MESH_FACE_COUNT = 3000           # None 表示不减面；建议 2000~5000
MIN_VALID_MESH_EXTENT = 1e-5


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

# =========================
# rigid 场景模式
# drop_cluster      : 多个物体从上方或前方进入
# opposed_lanes     : 左右/对角双向入场
# strike_static     : 一个运动物体进入并撞击一个静止物体
# chain_reaction    : 一个运动物体撞击两个近邻静止物体
# =========================
RIGID_SCENE_PATTERN_WEIGHTS = {
    "drop_cluster": 0.06,
    "opposed_lanes": 0.20,
    "strike_static": 0.58,
    "chain_reaction": 0.16,
}

# =========================
# rigid 运动模式
# 说明：
# - top_drop / top_toss: 上方入场
# - front_slide_in: 从前开口低位滑入/冲入
# - diagonal_corner_*: 从前侧上方向对角线打进容器
# - side_throw_*: 从左右外侧扔入
# - static_rest: 初始静止在容器内部
# - strike_static_*: 针对静止目标定向撞击
# =========================
RIGID_MOTION_WEIGHTS = {
    "top_drop": 0.34,
    "top_toss": 0.22,
    "front_slide_in": 0.18,
    "diagonal_corner_left": 0.10,
    "diagonal_corner_right": 0.10,
    "side_throw_left": 0.03,
    "side_throw_right": 0.03,
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
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def weighted_choice(d: dict):
    keys = list(d.keys())
    probs = np.array(list(d.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


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


def safe_subsample_points(xyz: np.ndarray, max_points=2048):
    if xyz is None:
        return None
    xyz = np.asarray(xyz)
    if xyz.ndim == 3:
        xyz = xyz.reshape(-1, 3)
    if len(xyz) <= max_points:
        return xyz
    idx = np.random.choice(len(xyz), size=max_points, replace=False)
    return xyz[idx]


def save_depth_vis(depth: np.ndarray, path: Path):
    d = depth.copy()
    valid = np.isfinite(d)
    vis = np.zeros_like(d, dtype=np.uint8)
    if valid.any():
        dmin, dmax = d[valid].min(), d[valid].max()
        denom = max(dmax - dmin, 1e-8)
        vis[valid] = ((d[valid] - dmin) / denom * 255).astype(np.uint8)
    imageio.imwrite(path, vis)


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

    base_h = float(rng.uniform(0.0, 1.0))
    for i in range(n):
        h = (base_h + i / max(n, 1) + rng.uniform(-0.03, 0.03)) % 1.0
        s = float(rng.uniform(0.55, 0.88))
        v = float(rng.uniform(0.68, 0.96))
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


def compute_bound_radius(half_x: float, half_y: float, half_z: float):
    return float(math.sqrt(half_x ** 2 + half_y ** 2 + half_z ** 2))


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
        bound_r = compute_bound_radius(half_x, half_y, half_z)

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

    else:
        raise ValueError(mode)

    return {
        "motion_type": mode,
        "init_pos": init_pos,
        "pose_delta": pose_delta,
        "init_linvel": linvel,
        "init_angvel": angvel,
    }



def _try_call_methods(obj, method_names, value):
    """
    尽量兼容不同 Genesis 版本的 API。
    """
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


def apply_initial_motion_to_rigid_entity(ent, obj_meta):
    """
    在 scene.build() 之后给 rigid 物体施加初始线速度和角速度。
    """
    if obj_meta.get("solver") != "Rigid":
        return

    v = np.asarray(obj_meta.get("init_linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
    w = np.asarray(obj_meta.get("init_angvel", [0.0, 0.0, 0.0]), dtype=np.float32)

    if np.linalg.norm(v) > 0:
        _try_call_methods(
            ent,
            ["set_vel", "set_velocity", "set_linear_velocity"],
            v,
        )

    if np.linalg.norm(w) > 0:
        _try_call_methods(
            ent,
            ["set_ang", "set_angvel", "set_angular_velocity"],
            w,
        )



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
            if Path(meta["unit_mesh_path"]).exists() and Path(meta["render_urdf_path"]).exists():
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

def prepare_asset_cache(asset):
    ensure_dir(ASSET_CACHE_DIR)

    if asset.get("dataset_name") == "physx3d" and asset.get("physx3d_part_infos"):
        return prepare_physx3d_asset_cache(asset)

    asset_id = asset["asset_id"]
    cache_obj = ASSET_CACHE_DIR / f"{asset_id}_unit.obj"
    cache_json = ASSET_CACHE_DIR / f"{asset_id}_unit_meta.json"

    if cache_obj.exists() and cache_json.exists():
        with open(cache_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

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

    mesh = load_trimesh_any(Path(asset["mesh_path"]))
    mesh, raw_extents, unit_extents, unit_scale = sanitize_mesh(mesh)

    # 继续导出 unit.obj，仅用于几何尺度统计 / debug
    mesh.export(cache_obj)

    sample_dir = Path(asset["sample_dir"])
    has_texture = (sample_dir / "material.mtl").exists() and len(list(sample_dir.glob("material_*.png"))) > 0

    raw_extents_scene = convert_bbox_extents_to_scene_frame(raw_extents, asset["dataset_name"])
    unit_extents_scene = convert_bbox_extents_to_scene_frame(unit_extents, asset["dataset_name"])

    meta = {
        "asset_id": asset_id,
        "mesh_path": asset["mesh_path"],
        "render_mesh_path": asset["mesh_path"],   # 真正渲染时优先用原始 material.obj
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
def infer_friction_from_part_info(part_info):
    mat_name = str(part_info.get("mat_name", "")).lower()
    mat_sub = str(part_info.get("mat_sub_type", "")).lower()

    if "metal" in mat_name:
        return 0.25
    if "fabric" in mat_name or "polyester" in mat_sub:
        return 0.75
    if "plastic" in mat_name or "polyamide" in mat_sub or "polypropylene" in mat_sub:
        return 0.45
    return None


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
            "restitution": float(np.clip(default_mat.get("restitution", 0.10), 0.0, 1.2)),
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

def sample_camera(container_cfg: dict):
    """
    相机放在容器开口外侧（负 y），略微拉远、略微降低，
    让放大的三面体容器和内部物体都更容易完整进入画面。
    """
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wh = container_cfg["wall_height"]
    cx, cy, cz = container_cfg["center"]

    return {
        "res": [IMG_W, IMG_H],
        "pos": [
            float(cx + np.random.uniform(-0.10, 0.10)),
            float(cy - hy - 2.10 + np.random.uniform(-0.18, 0.12)),
            float(cz + 0.72 * wh + np.random.uniform(-0.08, 0.12)),
        ],
        "lookat": [
            float(cx + np.random.uniform(-0.06, 0.06)),
            float(cy + np.random.uniform(0.15, 0.35)),
            float(cz + 0.22 + np.random.uniform(-0.04, 0.10)),
        ],
        "fov": float(np.random.uniform(40, 48)),
        "GUI": False,
    }


def add_container(scene, container_cfg: dict):
    """
    真三面体容器：
    - floor
    - left wall
    - right wall
    - back wall
    前方完全开口，开口朝 -y，相机从前方看进去。
    """
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wt = container_cfg["wall_thickness"]
    wh = container_cfg["wall_height"]
    ft = container_cfg["floor_thickness"]
    cx, cy, cz = container_cfg["center"]

    wall_mat = gs.materials.Rigid(rho=1200.0, friction=0.98)

    floor_surface = gs.surfaces.Default(color=(0.68, 0.70, 0.74, 1.0))
    left_surface = gs.surfaces.Default(color=(0.78, 0.60, 0.60, 1.0))
    right_surface = gs.surfaces.Default(color=(0.60, 0.78, 0.64, 1.0))
    back_surface = gs.surfaces.Default(color=(0.60, 0.68, 0.82, 1.0))

    container_entities = {}

    container_entities["floor"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2 * hx, 2 * hy, ft),
            pos=(cx, cy, cz + ft / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=floor_surface,
    )

    container_entities["left_wall"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(wt, 2 * hy, wh),
            pos=(cx - hx + wt / 2.0, cy, cz + ft + wh / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=left_surface,
    )

    container_entities["right_wall"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(wt, 2 * hy, wh),
            pos=(cx + hx - wt / 2.0, cy, cz + ft + wh / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=right_surface,
    )

    container_entities["back_wall"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2 * hx, wt, wh),
            pos=(cx, cy + hy - wt / 2.0, cz + ft + wh / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=back_surface,
    )

    return container_entities


def sample_spawn_xy(margin_x: float, margin_y: float, bias_to_back=False):
    """
    在容器内部采样 x/y，确保不贴墙。
    关键策略：
    - 左右方向留出安全边距
    - 前开口方向（-y）留出更大安全边距
    - bias_to_back=True 时，进一步偏向容器中后部
    """
    hx = CONTAINER["half_x"]
    hy = CONTAINER["half_y"]
    wt = CONTAINER["wall_thickness"]

    x_min = -hx + wt + SPAWN_SIDE_KEEP_OUT + margin_x
    x_max = +hx - wt - SPAWN_SIDE_KEEP_OUT - margin_x

    safe_front_y = -hy + wt + SPAWN_FRONT_KEEP_OUT + margin_y
    safe_back_y = +hy - wt - SPAWN_BACK_KEEP_OUT - margin_y

    if bias_to_back:
        y_min = max(safe_front_y, 0.02)
        y_max = safe_back_y
    else:
        y_min = max(safe_front_y, -0.08)
        y_max = safe_back_y

    if x_min >= x_max:
        x_min, x_max = -0.03, 0.03
    if y_min >= y_max:
        y_min, y_max = 0.02, 0.08

    x = float(np.random.uniform(x_min, x_max))
    y = float(np.random.uniform(y_min, y_max))
    return x, y



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


def create_genesis_rigid_material(mat_cfg):
    kwargs = {
        "rho": float(mat_cfg["rho"]),
        "friction": float(mat_cfg["friction"]),
    }
    restitution = mat_cfg.get("restitution", None)
    if restitution is not None:
        kwargs["restitution"] = float(restitution)

    try:
        return gs.materials.Rigid(**kwargs)
    except TypeError:
        kwargs.pop("restitution", None)
        return gs.materials.Rigid(**kwargs)


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

def rigid_geom_and_margins():
    shape = random.choice(["box", "sphere", "cylinder", "capsule"])

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

    bound_r = compute_bound_radius(half_x, half_y, half_z)
    return geom, half_x, half_y, half_z, bound_r


# =========================
# 物体采样
# =========================

def sample_dataset_rigid_object(obj_id: int, asset_bank, pattern="drop_cluster", motion_request=None):
    asset = random.choice(asset_bank)
    motion_request = motion_request or {}

    if asset.get("dataset_name") == PRIMITIVE_DATASET_NAME:
        geom_template = dict(asset["primitive_geom_template"])
        bbox_ext = np.asarray(asset["unit_bbox_extents"], dtype=np.float32)

        half_x = float(bbox_ext[0] / 2.0)
        half_y = float(bbox_ext[1] / 2.0)
        half_z = float(bbox_ext[2] / 2.0)
        bound_r = compute_bound_radius(half_x, half_y, half_z)

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
    bound_r = compute_bound_radius(half_x, half_y, half_z)

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

    use_part_colored_urdf = bool(
        asset["dataset_name"] == "physx3d"
        and PHYSX3D_USE_PART_COLORED_URDF
        and asset.get("render_urdf_path", None) is not None
    )
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
            "use_urdf_material": bool(PHYSX3D_URDF_USE_URDF_MATERIAL),
            "part_colors": asset.get("physx3d_part_colors", []),
            "part_materials": asset.get("physx3d_part_materials", []),
            "unit_part_mesh_paths": asset.get("physx3d_unit_part_mesh_paths", []),
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
    }


def sample_procedural_rigid_object(obj_id: int, pattern="drop_cluster", motion_request=None):
    geom, half_x, half_y, half_z, bound_r = rigid_geom_and_margins()
    material = sample_rigid_material()

    motion_request = motion_request or {}
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
    if (
        USE_DATASET_MESH_OBJECTS
        and asset_bank is not None
        and len(asset_bank) > 0
        and (np.random.rand() < DATASET_OBJECT_PROB)
    ):
        return sample_dataset_rigid_object(obj_id, asset_bank, pattern=pattern, motion_request=motion_request)

    return sample_procedural_rigid_object(obj_id, pattern=pattern, motion_request=motion_request)



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



def sample_scene_cfg(scene_id: int, asset_bank=None):
    seed = 100000 + scene_id
    set_seed(seed)

    family = FORCE_SCENE_FAMILY or weighted_choice(SCENE_FAMILY_WEIGHTS)
    if FORCE_RIGID_PATTERN is not None:
        family = "rigid_mix"
    if family == "cloth_drop" and not (ENABLE_CLOTH and CLOTH_MESH_PATH and Path(CLOTH_MESH_PATH).exists()):
        family = "rigid_mix"

    bg = sample_background()
    cam = sample_camera(CONTAINER)
    rigid_pattern = None

    if family == "rigid_mix":
        rigid_pattern = FORCE_RIGID_PATTERN or weighted_choice(RIGID_SCENE_PATTERN_WEIGHTS)
        if rigid_pattern == "strike_static":
            n_obj = random.randint(5, 7)
            sim_steps = 210
        elif rigid_pattern == "chain_reaction":
            n_obj = random.randint(6, 8)
            sim_steps = 220
        elif rigid_pattern == "opposed_lanes":
            n_obj = random.randint(5, 7)
            sim_steps = 190
        else:
            n_obj = random.randint(5, 8)
            sim_steps = 180

        objects = build_rigid_objects(rigid_pattern, n_obj, asset_bank=asset_bank)
        sim_options = {
            "gravity": [0.0, 0.0, -9.81],
            "dt": 4e-3,
            "substeps": 8,
            "num_steps": sim_steps,
        }

    elif family == "mpm_mix":
        n_obj = random.randint(3, 5)
        objects = [sample_mpm_object(i) for i in range(n_obj)]
        sim_options = {
            "gravity": [0.0, 0.0, -9.81],
            "dt": 4e-3,
            "substeps": 10,
            "num_steps": 220,
        }

    elif family == "sph_liquid":
        n_obj = random.randint(2, 3)
        objects = [sample_sph_object(i) for i in range(n_obj)]
        sim_options = {
            "gravity": [0.0, 0.0, -9.81],
            "dt": 4e-3,
            "substeps": 10,
            "num_steps": 220,
        }

    else:
        raise ValueError(f"Unknown family: {family}")

    return {
        "scene_id": f"train_scene_{scene_id:06d}",
        "seed": seed,
        "family": family,
        "rigid_pattern": rigid_pattern,
        "background": bg,
        "container": CONTAINER,
        "camera": cam,
        "sim_options": sim_options,
        "objects": objects,
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
    scene_kwargs = dict(
        sim_options=sim_options,
        vis_options=vis_options,
        show_viewer=False,
    )

    if family == "mpm_mix":
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(
            lower_bound=(-0.9, -0.9, -0.1),
            upper_bound=(0.9, 0.9, 1.8),
        )
    elif family == "sph_liquid":
        scene_kwargs["sph_options"] = gs.options.SPHOptions(
            lower_bound=(-0.9, -0.9, -0.1),
            upper_bound=(0.9, 0.9, 1.8),
            particle_size=0.01,
        )
    elif family == "cloth_drop":
        scene_kwargs["pbd_options"] = gs.options.PBDOptions()

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
    container_entities = add_container(scene, scene_cfg["container"])

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
            mat = create_genesis_rigid_material(obj["material"])
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
                ent = scene.add_entity(**kwargs)

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
                    ent = scene.add_entity(**kwargs)
                except TypeError:
                    fallback_keys = ["file", "scale", "pos", "euler", "visualization", "collision", "fixed"]
                    kwargs["morph"] = gs.morphs.URDF(**{k: urdf_kwargs[k] for k in fallback_keys})
                    ent = scene.add_entity(**kwargs)

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

        elif obj["solver"] == "MPM":
            mat = gs.materials.MPM.Elastic(
                E=obj["material"]["E"],
                nu=obj["material"]["nu"],
                rho=obj["material"]["rho"],
                sampler=obj["material"]["sampler"],
                model=obj["material"]["model"],
            )
            pos = tuple(obj["init_pos"])

            if obj["geom"]["shape"] == "box":
                ent = scene.add_entity(
                    morph=gs.morphs.Box(size=tuple(obj["geom"]["size"]), pos=pos),
                    material=mat,
                    surface=surface,
                )
            elif obj["geom"]["shape"] == "sphere":
                ent = scene.add_entity(
                    morph=gs.morphs.Sphere(radius=obj["geom"]["radius"], pos=pos),
                    material=mat,
                    surface=surface,
                )
            else:
                raise ValueError(obj["geom"]["shape"])

        elif obj["solver"] == "SPH":
            mat = gs.materials.SPH.Liquid(
                rho=obj["material"]["rho"],
                stiffness=obj["material"]["stiffness"],
                exponent=obj["material"]["exponent"],
                mu=obj["material"]["mu"],
                gamma=obj["material"]["gamma"],
                sampler=obj["material"]["sampler"],
            )
            pos = tuple(obj["init_pos"])

            if obj["geom"]["shape"] == "box":
                ent = scene.add_entity(
                    morph=gs.morphs.Box(size=tuple(obj["geom"]["size"]), pos=pos),
                    material=mat,
                    surface=surface,
                )
            else:
                raise ValueError(obj["geom"]["shape"])

        elif obj["solver"] == "PBD":
            ent = scene.add_entity(
                material=gs.materials.PBD.Cloth(),
                morph=gs.morphs.Mesh(
                    file=CLOTH_MESH_PATH,
                    scale=obj.get("scale", 1.0),
                    pos=tuple(obj["init_pos"]),
                    euler=tuple(obj.get("init_euler", [0.0, 0.0, 0.0])),
                ),
                surface=surface,
            )

        else:
            raise ValueError(obj["solver"])

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
            apply_initial_motion_to_rigid_entity(ent, obj)

    # MPM 给一点轻微扰动（可选）
    for obj, ent in zip(scene_cfg["objects"], entities):
        if obj["solver"] == "MPM" and hasattr(ent, "set_velocity"):
            try:
                pts = to_numpy(ent.get_particles_pos()).reshape(-1, 3)
                if len(pts) > 0:
                    vel = np.zeros((len(pts), 3), dtype=np.float32)
                    vel[:, 0] = np.random.uniform(-0.25, 0.25)
                    vel[:, 1] = np.random.uniform(-0.20, 0.20)
                    vel[:, 2] = np.random.uniform(-0.08, 0.08)
                    ent.set_velocity(vel)
            except Exception:
                pass

    return scene, cam, entities, container_entities




def prepare_output_dirs(out_dir: Path):
    subdirs = [
        "rgb", "depth", "depth_vis", "segmentation", "normal", "pointcloud",
        "object_pointcloud", "trajectories", "camera", "video"
    ]
    for s in subdirs:
        ensure_dir(out_dir / s)


def write_dataset_format_description(dataset_root: Path):
    doc = f"""# Genesis Sim Merge Dataset Format

## Coordinate Convention

- World frame: `z-up`, right-handed.
- External assets: default `y-up` on input, converted at runtime with base Euler `[{YUP_TO_ZUP_EULER_XYZ[0]:.6f}, {YUP_TO_ZUP_EULER_XYZ[1]:.6f}, {YUP_TO_ZUP_EULER_XYZ[2]:.6f}]` (XYZ radians) when the dataset source is marked as `y_up`.
- Camera point clouds and object trajectories are exported in world coordinates.

## Directory Layout

Each scene lives at `train/<scene_id>/` and contains:

- `scene_input.json`: sampled scene config before simulation.
- `scene_metadata.json`: simulation summary, material summary, coordinate-transform summary, and export schema.
- `rgb/<frame>.png`: RGB image.
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
- `video/preview.mp4`: preview video.

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

    if hasattr(ent, "get_particles_pos"):
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

    if hasattr(ent, "get_verts"):
        try:
            verts = to_numpy(ent.get_verts())
            if verts is not None and verts.size > 0:
                verts = verts.reshape(-1, 3)
                state["pointcloud"] = verts
                state["centroid"] = verts.mean(axis=0)
                state["n_points"] = int(len(verts))
        except Exception:
            pass

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


def export_scene(scene_cfg: dict):
    out_dir = DATASET_ROOT / "train" / scene_cfg["scene_id"]
    prepare_output_dirs(out_dir)

    with open(out_dir / "scene_input.json", "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, ensure_ascii=False, indent=2)

    scene, cam, entities, container_entities = None, None, None, None
    traj_csv, frame_csv, object_pc_csv = None, None, None

    try:
        scene, cam, entities, container_entities = build_scene(scene_cfg)

        try:
            np.save(out_dir / "camera" / "intrinsics.npy", to_numpy(cam.intrinsics))
        except Exception:
            pass
        try:
            np.save(out_dir / "camera" / "extrinsics.npy", to_numpy(cam.extrinsics))
        except Exception:
            pass

        traj_path = out_dir / "trajectories" / "objects_world.csv"
        frame_index_path = out_dir / "trajectories" / "frame_index.csv"
        object_pc_index_path = out_dir / "trajectories" / "object_pointcloud_index.csv"

        traj_csv = open(traj_path, "w", newline="", encoding="utf-8")
        traj_writer = csv.writer(traj_csv)
        traj_writer.writerow([
            "frame", "object_id", "solver",
            "cx", "cy", "cz",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
            "n_points"
        ])

        frame_csv = open(frame_index_path, "w", newline="", encoding="utf-8")
        frame_writer = csv.writer(frame_csv)
        frame_writer.writerow([
            "frame", "rgb_path", "depth_path", "depth_vis_path",
            "seg_path", "normal_path", "pointcloud_path"
        ])

        object_pc_csv = open(object_pc_index_path, "w", newline="", encoding="utf-8")
        object_pc_writer = csv.writer(object_pc_csv)
        object_pc_writer.writerow([
            "frame", "object_id", "solver", "pointcloud_path",
            "cx", "cy", "cz",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
            "n_points_raw", "n_points_saved", "coordinate_frame",
        ])

        preview_frames = []
        collision_detected = False
        num_steps = scene_cfg["sim_options"]["num_steps"]

        for t in range(num_steps):
            scene.step()

            rgb, depth, seg, normal = cam.render(
                rgb=True,
                depth=True,
                segmentation=True,
                normal=True,
            )

            rgb_path = out_dir / "rgb" / f"{t:06d}.png"
            depth_path = out_dir / "depth" / f"{t:06d}.npy"
            depth_vis_path = out_dir / "depth_vis" / f"{t:06d}.png"
            seg_path = out_dir / "segmentation" / f"{t:06d}.npy"
            normal_path = out_dir / "normal" / f"{t:06d}.npy"

            imageio.imwrite(rgb_path, rgb)
            np.save(depth_path, depth)
            save_depth_vis(depth, depth_vis_path)
            np.save(seg_path, seg)
            np.save(normal_path, normal)

            pc_name = ""
            if (t % CAMERA_PC_STRIDE) == 0:
                try:
                    pc, mask = cam.render_pointcloud(world_frame=True)
                    pc_path = out_dir / "pointcloud" / f"{t:06d}.npz"
                    np.savez_compressed(pc_path, xyz=pc, mask=mask)
                    pc_name = pc_path.name
                except Exception:
                    pc_name = ""

            frame_writer.writerow([
                t,
                rgb_path.name,
                depth_path.name,
                depth_vis_path.name,
                seg_path.name,
                normal_path.name,
                pc_name,
            ])

            if t % 3 == 0:
                preview_frames.append(rgb)

            for obj_meta, ent in zip(scene_cfg["objects"], entities):
                if obj_meta["solver"] == "Rigid" and hasattr(ent, "detect_collision"):
                    try:
                        if bool(ent.detect_collision()):
                            collision_detected = True
                    except Exception:
                        pass

                state = export_entity_state(ent, obj_meta)

                c = state["centroid"] if state["centroid"] is not None else [np.nan, np.nan, np.nan]
                q = state["quat"] if state["quat"] is not None else [np.nan] * 4
                v = state["vel"] if state["vel"] is not None else [np.nan] * 3
                w = state["ang"] if state["ang"] is not None else [np.nan] * 3

                traj_writer.writerow([
                    t, obj_meta["object_id"], obj_meta["solver"],
                    float(c[0]), float(c[1]), float(c[2]),
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                    float(v[0]), float(v[1]), float(v[2]),
                    float(w[0]), float(w[1]), float(w[2]),
                    int(state["n_points"]),
                ])

                if (t % OBJECT_PC_STRIDE) == 0 and state["pointcloud"] is not None:
                    xyz = safe_subsample_points(state["pointcloud"], max_points=MAX_OBJECT_PC)
                    object_pc_path = out_dir / "object_pointcloud" / f"{t:06d}_obj{obj_meta['object_id']:02d}.npz"
                    np.savez_compressed(
                        object_pc_path,
                        xyz=xyz,
                        solver=obj_meta["solver"],
                        object_id=obj_meta["object_id"],
                        frame=t,
                        centroid=np.asarray(c, dtype=np.float32),
                        quat=np.asarray(q, dtype=np.float32),
                        vel=np.asarray(v, dtype=np.float32),
                        ang=np.asarray(w, dtype=np.float32),
                        n_points_raw=int(state["n_points"]),
                        n_points_saved=int(len(xyz)),
                        coordinate_frame="world",
                    )
                    object_pc_writer.writerow([
                        t, obj_meta["object_id"], obj_meta["solver"], object_pc_path.name,
                        float(c[0]), float(c[1]), float(c[2]),
                        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                        float(v[0]), float(v[1]), float(v[2]),
                        float(w[0]), float(w[1]), float(w[2]),
                        int(state["n_points"]), int(len(xyz)), "world",
                    ])

        if len(preview_frames) > 0:
            imageio.mimsave(out_dir / "video" / "preview.mp4", preview_frames, fps=20)

        material_summary = []
        for obj in scene_cfg["objects"]:
            record = {
                "object_id": obj["object_id"],
                "solver": obj["solver"],
                "source_type": obj.get("source_type", "unknown"),
                "motion_type": obj.get("motion_type", "unknown"),
                "material": obj["material"],
                "coordinate_transform": obj.get("coordinate_transform", None),
            }
            if obj["solver"] == "Rigid" and obj["geom"]["shape"] in {"mesh", "urdf"}:
                record["asset"] = {
                    "asset_id": obj["geom"].get("asset_id"),
                    "dataset_name": obj["geom"].get("dataset_name"),
                    "sample_dir": obj["geom"].get("sample_dir"),
                    "mesh_file": obj["geom"].get("mesh_file"),
                    "urdf_file": obj["geom"].get("urdf_file"),
                    "scale": obj["geom"].get("scale"),
                    "bbox_extents": obj["geom"].get("bbox_extents"),
                    "n_vertices": obj["geom"].get("n_vertices"),
                    "n_faces": obj["geom"].get("n_faces"),
                    "part_colors": obj["geom"].get("part_colors", []),
                    "part_materials": obj["geom"].get("part_materials", []),
                }
            else:
                record["geom"] = obj["geom"]
                if obj["geom"].get("dataset_name") == PRIMITIVE_DATASET_NAME:
                    record["asset"] = {
                        "asset_id": obj["geom"].get("asset_id"),
                        "dataset_name": obj["geom"].get("dataset_name"),
                        "sample_dir": obj["geom"].get("sample_dir"),
                        "primitive_shape_name": obj["geom"].get("primitive_shape_name"),
                        "primitive_material_name": obj["geom"].get("primitive_material_name"),
                        "bbox_extents": obj["geom"].get("bbox_extents"),
                    }
            material_summary.append(record)

        scene_metadata = {
            "scene_id": scene_cfg["scene_id"],
            "seed": scene_cfg["seed"],
            "family": scene_cfg["family"],
            "rigid_pattern": scene_cfg.get("rigid_pattern", None),
            "num_objects": len(scene_cfg["objects"]),
            "num_dataset_mesh_objects": int(sum(
                1 for x in scene_cfg["objects"]
                if x.get("source_type") == "dataset_mesh"
            )),
            "sim_steps": num_steps,
            "dt": scene_cfg["sim_options"]["dt"],
            "substeps": scene_cfg["sim_options"]["substeps"],
            "collision_detected": collision_detected,
            "background_name": scene_cfg["background"]["name"],
            "container": scene_cfg["container"],
            "material_summary": material_summary,
            "exports": {
                "coordinate_convention": {
                    "world_up_axis": "z_up",
                    "camera_pointcloud_frame": "world",
                    "object_trajectory_frame": "world",
                    "external_asset_default_up_axis": "y_up",
                },
                "files": {
                    "rgb": "rgb/<frame:06d>.png",
                    "depth": "depth/<frame:06d>.npy",
                    "depth_vis": "depth_vis/<frame:06d>.png",
                    "segmentation": "segmentation/<frame:06d>.npy",
                    "normal": "normal/<frame:06d>.npy",
                    "scene_pointcloud": "pointcloud/<frame:06d>.npz",
                    "object_pointcloud": "object_pointcloud/<frame:06d>_obj<object_id:02d>.npz",
                    "frame_index_csv": "trajectories/frame_index.csv",
                    "object_trajectory_csv": "trajectories/objects_world.csv",
                    "object_pointcloud_index_csv": "trajectories/object_pointcloud_index.csv",
                    "camera_intrinsics": "camera/intrinsics.npy",
                    "camera_extrinsics": "camera/extrinsics.npy",
                    "preview_video": "video/preview.mp4",
                },
                "object_pointcloud_npz_keys": [
                    "xyz", "object_id", "frame", "solver", "centroid",
                    "quat", "vel", "ang", "n_points_raw", "n_points_saved", "coordinate_frame",
                ],
            },
            "status": "ok",
        }

        with open(out_dir / "scene_metadata.json", "w", encoding="utf-8") as f:
            json.dump(scene_metadata, f, ensure_ascii=False, indent=2)

        return scene_metadata

    finally:
        if traj_csv is not None:
            traj_csv.close()
        if frame_csv is not None:
            frame_csv.close()
        if object_pc_csv is not None:
            object_pc_csv.close()
        safe_scene_destroy(scene)


# =========================
# 主程序
# =========================
def main():
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
        "dataset_name": "genesis_sim_v3",
        "split": "train",
        "n_scenes_requested": N_SCENES,
        "image_size": [IMG_W, IMG_H],
        "backend_used": backend_used,
        "scene_families": SCENE_FAMILY_WEIGHTS,
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
            "Rigid scenes support scene-level motion patterns including strike_static and chain_reaction.",
            "PhysX-3D objects can be rendered as single-rigid-body URDFs with per-part colors.",
            "Rigid objects can start with arbitrary xyz Euler angles instead of only near-flat poses.",
            "MPM/SPH scenes remain procedural for stability.",
            "Single-scene failure is recorded instead of aborting the whole run.",
            "This script keeps the current scene definition and exposes dataset-source interfaces for SOPHY / PhysX-3D / primitive / all."
        ],
        "scenes": [],
        "failed_scenes": [],
    }

    try:
        for sid in range(N_SCENES):
            scene_cfg = sample_scene_cfg(sid, asset_bank=asset_bank)

            try:
                print(f"[RUN ] {scene_cfg['scene_id']} | family={scene_cfg['family']}")
                meta = export_scene(scene_cfg)
                manifest["scenes"].append(meta)
                print(
                    f"[ OK ] {scene_cfg['scene_id']} | family={scene_cfg['family']} "
                    f"| dataset_mesh={meta['num_dataset_mesh_objects']}/{meta['num_objects']}"
                )

            except Exception as e:
                err_info = {
                    "scene_id": scene_cfg["scene_id"],
                    "family": scene_cfg["family"],
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

rm -r /data/gaoya/AAA_test_video/Dataset_test/genesis_sim_merge
CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/Code_data/genesis_demo_physxnet_urdf_loader_merge.py 





'''


'''
刚体运动仿真，就算用了sophy和physxnet数据集但是好像没有将部件级材料参数用于仿真
可以用primitive普通数据集做一下简单物体仿真



python /home/gaoya/Code_Video/Code_data/1_localshow.py \
  --root /data/gaoya/AAA_test_video/Dataset_test//train \
  --host 0.0.0.0 \
  --port 8001



'''
