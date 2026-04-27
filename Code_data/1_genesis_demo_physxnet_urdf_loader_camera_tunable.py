from __future__ import annotations

import os
import csv
import copy
import json
import math
import random
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import trimesh

try:
    import genesis as gs
except Exception:
    gs = None


_GENESIS_INIT_LOCK = threading.Lock()
_GENESIS_INITIALIZED = False
_GENESIS_BACKEND_USED = "none"


def ensure_genesis_initialized(prefer_gpu: bool = True) -> str:
    """Lazily initialize Genesis once so imported helper paths also work."""
    global _GENESIS_INITIALIZED, _GENESIS_BACKEND_USED

    if gs is None:
        raise RuntimeError("genesis is not importable in this environment")

    if _GENESIS_INITIALIZED:
        return _GENESIS_BACKEND_USED

    with _GENESIS_INIT_LOCK:
        if _GENESIS_INITIALIZED:
            return _GENESIS_BACKEND_USED

        init_candidates = []
        if prefer_gpu and hasattr(gs, "gpu"):
            init_candidates.append(("gpu", gs.gpu))
        if hasattr(gs, "cpu"):
            init_candidates.append(("cpu", gs.cpu))

        last_err = None
        if not init_candidates:
            try:
                gs.init()
                _GENESIS_INITIALIZED = True
                _GENESIS_BACKEND_USED = "default"
                return _GENESIS_BACKEND_USED
            except Exception as e:
                msg = str(e).lower()
                if "already" in msg and "initial" in msg:
                    _GENESIS_INITIALIZED = True
                    _GENESIS_BACKEND_USED = "existing"
                    return _GENESIS_BACKEND_USED
                raise

        for backend_name, backend_value in init_candidates:
            try:
                gs.init(backend=backend_value)
                _GENESIS_INITIALIZED = True
                _GENESIS_BACKEND_USED = backend_name
                return _GENESIS_BACKEND_USED
            except Exception as e:
                msg = str(e).lower()
                if "already" in msg and "initial" in msg:
                    _GENESIS_INITIALIZED = True
                    _GENESIS_BACKEND_USED = backend_name
                    return _GENESIS_BACKEND_USED
                last_err = e

        raise RuntimeError(f"failed to initialize Genesis: {last_err}") from last_err


# =========================================================
# 用户可调参数（优先改这里）
# =========================================================
PHYSXNET_ROOT = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
PHYSXNET_VERSION = "version_1"
OUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_test/physxnet_urdf_dataset_v2_camera_fixed")

IMG_W, IMG_H = 960, 720
OUTPUT_FPS = 30
FPS = OUTPUT_FPS
PHYSICS_FPS = 360
PHYSICS_STEPS_PER_FRAME = max(1, PHYSICS_FPS // OUTPUT_FPS)
N_OUTPUT_FRAMES = 96
N_SCENES = 10

# 读取多少个 PhysXNet 物体进候选池；设为 0 / None 表示尽量读取全量，最大化利用原始数据集。
MAX_DATASET_OBJECTS_TO_READ = 20

# 同一场景物体数量控制
MIN_OBJECTS_PER_SCENE = 2
MAX_OBJECTS_PER_SCENE = 6
MAX_VOLUME_RATIO_IN_SCENE = 4.5
MAX_SCENE_SAMPLING_RETRIES = 120
STATIC_OBJECT_PROB = 0.30

# 仿真稳定性控制
# 注意：scene.step() 与视频帧不是同一个概念。这里把“物理步频”和“输出视频帧率”分开，
# 避免 dt 很小但每一步都直接写成一帧，导致视频看起来像“悬停/慢动作”。
SIM_DT = 1.0 / PHYSICS_FPS
SIM_SUBSTEPS = 12
SIM_NUM_STEPS = N_OUTPUT_FRAMES * PHYSICS_STEPS_PER_FRAME
WARMUP_STEPS = 0
PREVIEW_FRAME_STRIDE = 1

# 相机远近控制
CAMERA_DISTANCE_MULT_MIN = 5.5
CAMERA_DISTANCE_MULT_MAX = 10.5
CAMERA_FOV_MIN = 32.0
CAMERA_FOV_MAX = 48.0
CAMERA_ELEVATION_MIN = 36.0
CAMERA_ELEVATION_MAX = 68.0
CAMERA_AZIMUTH_MIN = 10.0
CAMERA_AZIMUTH_MAX = 42.0

# 更合理的小场景摄影棚尺寸：按“运动包围盒 + 物体尺寸”而不是按单个最大边长粗暴放大
SCENE_PANEL_SIZE_MIN = 2.4
SCENE_PANEL_SIZE_MAX = 5.2
SCENE_PANEL_MOTION_SCALE = 1.55
SCENE_PANEL_OBJECT_SCALE = 1.85
SCENE_SIDE_MARGIN = 0.38
SCENE_FRONT_MARGIN = 0.78
SCENE_BACK_MARGIN = 0.58
SCENE_WALL_THICKNESS_BASE = 0.05
SCENE_WALL_THICKNESS_SCALE = 0.06

# 镜头不要太高、不要太远、物体在画面里更大
CAMERA_BASE_DIST_SCALE = 1.40
CAMERA_HEIGHT_SCALE = 0.00
CAMERA_MIN_DIST = 2.90
CAMERA_MAX_DIST = 5.00
CAMERA_OUTSIDE_FRONT_GAP = 0.34

CAMERA_OCCUPANCY_TARGET = 0.26
CAMERA_OCCUPANCY_MIN = 0.10
CAMERA_OCCUPANCY_MAX = 0.42
CAMERA_MIN_PER_OBJECT_AREA = 0.0035



# =========================
# 手动/交互式相机覆盖
# =========================
# 默认仍走自动相机；当 USE_MANUAL_CAMERA=True 时，直接使用 MANUAL_CAMERA
USE_MANUAL_CAMERA = False

# 可选：从外部 JSON 加载手动相机；优先级高于上面的 MANUAL_CAMERA
# JSON 格式示例：
# {
#   "pos": [0.0, -3.2, 0.78],
#   "lookat": [0.0, 0.25, 0.28],
#   "fov": 35.0,
#   "res": [960, 720]
# }
MANUAL_CAMERA_JSON_PATH: Optional[str] = None

MANUAL_CAMERA = {
    "pos": [0.0, -3.2, 0.78],
    "lookat": [0.0, 0.25, 0.28],
    "fov": 35.0,
    "res": [IMG_W, IMG_H],
    "GUI": False,
    "camera_style": "manual_override",
}

# 目标尺度控制：把 PhysXNet 的原始物体缩放到更稳的仿真范围
TARGET_MAX_DIM_MIN = 0.5
TARGET_MAX_DIM_MAX = 0.6
MIN_VALID_PROXY_EXTENT = 0.25

MERGED_CACHE_DIR = PHYSXNET_ROOT / PHYSXNET_VERSION / "_merged_for_genesis"
PROXY_CACHE_DIR = PHYSXNET_ROOT / PHYSXNET_VERSION / "_collision_proxy_for_genesis"
URDF_CACHE_DIR = PHYSXNET_ROOT / PHYSXNET_VERSION / "_urdf_for_genesis"
EXPORT_MERGED_WHEN_LOADING = True
EXPORT_PROXY_WHEN_LOADING = True
EXPORT_URDF_WHEN_LOADING = True

# 推荐 auto / part_hulls / convex_hull
PROXY_MODE = "auto"  # choices: auto, merged, convex_hull, bbox_mesh, part_hulls, part_boxes
STOP_ON_ERROR = False

# 这次改成 PhysX-3D -> URDF -> Genesis。
# 注意：这里导出的是“单刚体 URDF”，即把 render mesh 和 collision proxy 写入同一个 base_link。
# 好处是最稳、最容易直接接 Genesis；局限是不会自动恢复复杂 articulated 关节树。
USE_URDF_ASSET = True
URDF_MERGE_FIXED_LINKS = True
URDF_PRIORITIZE_URDF_MATERIAL = False
URDF_RECOMPUTE_INERTIA = False

# 更偏向复杂物体，而不是随机从少量候选里抽到简单几何体。
PREFER_COMPLEX_OBJECTS = True
COMPLEX_OBJECT_TOP_RATIO = 0.45
COMPLEX_OBJECT_WEIGHT_POWER = 1.35

# 复杂外观渲染：物理上仍用 collision proxy，画面里单独同步一个高细节 visual mesh。
USE_COMPLEX_VISUAL_MESH = True
VISUAL_MESH_DECIMATE = False
VISUAL_MESH_CONVEXIFY = False
VISUAL_MESH_FIXED = True
COLLISION_PROXY_DECIMATE = False
COLLISION_PROXY_CONVEXIFY = False

# 更多运动类型，不只下坠
SCENE_FAMILY_WEIGHTS = {
    # "free_drop": 0.05,
    # "vertical_bounce": 0.03,
    # "upward_toss": 0.05,
    # "underhand_arc": 0.05,
    # "oblique_throw": 0.05,
    # "side_throw": 0.05,
    # "bank_shot": 0.04,
    # "rolling_push": 0.04,
    # "floor_slide": 0.04,
    # "diagonal_sweep": 0.04,
    # "rest_then_hit": 0.05,
    # "line_chain_collision": 0.04,
    # "cross_fire": 0.04,
    # "stack_drop": 0.03,
    # "late_entry": 0.05,
    # "staggered_rain": 0.04,
    # "sequential_entry": 0.04,
    # "static_then_dual_hit": 0.04,
    "static_then_dual_hit": 1,
    # "orbit_mix": 0.03,
    # "mixed_multi": 0.04,
    # "front_entry_arc": 0.05,
    # "front_entry_slide": 0.04,
    # "left_right_pingpong": 0.04,
    # "back_wall_rebound": 0.03,
}

MAX_LINVEL_NORM = 1.8
MAX_ANGVEL_NORM = 7.0
LINEAR_SPEED_SCALE = 0.72
ANGULAR_SPEED_SCALE = 0.85

CORNER_BASE = {"center": [0.0, 0.0, 0.0]}

OBJECT_COLOR_PALETTE = [
    (0.90, 0.35, 0.35, 1.0),
    (0.35, 0.75, 0.40, 1.0),
    (0.30, 0.50, 0.90, 1.0),
    (0.90, 0.75, 0.30, 1.0),
    (0.75, 0.35, 0.85, 1.0),
    (0.25, 0.78, 0.82, 1.0),
    (0.95, 0.55, 0.25, 1.0),
    (0.55, 0.55, 0.60, 1.0),
    (0.45, 0.65, 0.20, 1.0),
    (0.20, 0.55, 0.65, 1.0),
]


# =========================
# 墙面条纹参数
# =========================
ENABLE_STRIPED_WALLS = True



# 条纹宽度、间隔（沿 z 方向一条条叠上去）
WALL_STRIPE_BAND = 0.16
WALL_STRIPE_GAP = 0.10

# 条纹层厚度：非常薄，只做显示
WALL_STRIPE_DEPTH = 0.008

# 距边缘留一点空，避免太贴边不好看
WALL_STRIPE_MARGIN = 0.04

# 是否让三面墙用不同条纹底色
CONTAINER_FACE_COLORS = {
    "floor": (0.88, 0.89, 0.91, 1.0),
    "wall_left": (0.95, 0.95, 0.96, 1.0),
    "wall_right": (0.95, 0.95, 0.96, 1.0),
    "wall_back": (0.95, 0.95, 0.96, 1.0),
}

# 墙面统一改成浅色条纹
WALL_STRIPE_COLOR_A = (0.98, 0.98, 0.985, 1.0)
WALL_STRIPE_COLOR_B = (0.90, 0.91, 0.93, 1.0)

# 不再按每面墙底色去混深色，直接保持统一浅色条纹
WALL_STRIPE_USE_FACE_BASE = False



# 墙面相关安全边距：
# 1) 可见墙面保持较薄，保证画面自然；
# 2) 在可见墙后面再叠一层更厚的 guard wall，减少高速穿墙。
# 3) 所有生成位置和运行时进入位置，都尽量与墙面保持 clearance。
VISIBLE_WALL_THICKNESS_SCALE = 1.0
GUARD_WALL_THICKNESS_SCALE = 3.0
MIN_VISIBLE_WALL_THICKNESS = 0.05
MIN_GUARD_WALL_THICKNESS = 0.18
WALL_CLEARANCE_BASE = 0.14
WALL_CLEARANCE_EXTRA = 0.06
MAX_RUNTIME_PUSH_FROM_WALL = 0.42
FRONT_OPEN_MARGIN = 0.26
RUNTIME_PARKING_OFFSET = 0.65
EARLY_ENTRY_FRAME_MAX = 10


# 基于材质名的粗略物理先验。PhysXNet 本身提供 dimension / material / density / kinematics 等信息，
# 这里仅在 Genesis 侧补一个更稳的 friction / restitution / damping 先验，尽量让同场景物体更有差异。
MATERIAL_PRIORS = {
    "metal": {"friction": (0.15, 0.40), "restitution": (0.05, 0.18), "linear_damping": (0.00, 0.02)},
    "glass": {"friction": (0.08, 0.22), "restitution": (0.02, 0.10), "linear_damping": (0.00, 0.02)},
    "plastic": {"friction": (0.30, 0.65), "restitution": (0.08, 0.22), "linear_damping": (0.01, 0.04)},
    "wood": {"friction": (0.35, 0.80), "restitution": (0.02, 0.12), "linear_damping": (0.01, 0.05)},
    "rubber": {"friction": (0.70, 1.10), "restitution": (0.35, 0.70), "linear_damping": (0.01, 0.04)},
    "foam": {"friction": (0.45, 0.90), "restitution": (0.12, 0.30), "linear_damping": (0.03, 0.08)},
    "fabric": {"friction": (0.45, 0.95), "restitution": (0.02, 0.12), "linear_damping": (0.03, 0.10)},
    "leather": {"friction": (0.45, 0.85), "restitution": (0.03, 0.10), "linear_damping": (0.02, 0.06)},
    "ceramic": {"friction": (0.20, 0.45), "restitution": (0.03, 0.12), "linear_damping": (0.00, 0.03)},
    "stone": {"friction": (0.30, 0.60), "restitution": (0.02, 0.10), "linear_damping": (0.00, 0.03)},
    "default": {"friction": (0.28, 0.78), "restitution": (0.04, 0.18), "linear_damping": (0.01, 0.05)},
}

def _mix_rgba(c1, c2, a=0.5):
    return tuple(float((1 - a) * c1[i] + a * c2[i]) for i in range(4))


def _add_striped_wall_overlay(
    scene: Any,
    wall_mat: Any,
    plane: str,               # "x" or "y"
    face_pos: float,          # 内侧可见面的坐标
    span_a0: float,           # 另一水平轴起点
    span_a1: float,           # 另一水平轴终点
    z0: float,                # 高度起点
    z1: float,                # 高度终点
    base_color: Tuple[float, float, float, float],
    inward_sign: float,       # 朝容器内部的方向；左墙=+1，右墙=-1，后墙=-1
) -> None:
    if not ENABLE_STRIPED_WALLS:
        return

    stripe_a = WALL_STRIPE_COLOR_A
    stripe_b = WALL_STRIPE_COLOR_B
    if WALL_STRIPE_USE_FACE_BASE:
        stripe_b = _mix_rgba(base_color, WALL_STRIPE_COLOR_B, 0.45)

    z = z0 + WALL_STRIPE_MARGIN
    idx = 0

    usable_span = max(0.0, (span_a1 - span_a0) - 2.0 * WALL_STRIPE_MARGIN)
    if usable_span <= 1e-6:
        return

    while z < z1 - WALL_STRIPE_MARGIN:
        h = min(WALL_STRIPE_BAND, z1 - WALL_STRIPE_MARGIN - z)
        if h <= 1e-6:
            break

        color = stripe_a if (idx % 2 == 0) else stripe_b

        if plane == "x":
            # x 固定，沿 y-z 展开（左右墙）
            size = (WALL_STRIPE_DEPTH, usable_span, h)
            pos = (
                face_pos + inward_sign * (WALL_STRIPE_DEPTH * 0.5),
                0.5 * (span_a0 + span_a1),
                z + 0.5 * h,
            )
        else:
            # y 固定，沿 x-z 展开（后墙）
            size = (usable_span, WALL_STRIPE_DEPTH, h)
            pos = (
                0.5 * (span_a0 + span_a1),
                face_pos + inward_sign * (WALL_STRIPE_DEPTH * 0.5),
                z + 0.5 * h,
            )

        scene.add_entity(
            morph=gs.morphs.Box(
                size=size,
                pos=pos,
                fixed=True,
                collision=False,   # 只显示，不参与碰撞
            ),
            material=wall_mat,
            surface=gs.surfaces.Default(color=color),
        )

        z += WALL_STRIPE_BAND + WALL_STRIPE_GAP
        idx += 1
# =========================================================
# 数据结构
# =========================================================
@dataclass
class PartSpec:
    part_id: int
    name: str
    mesh_path: str
    image_path: Optional[str]
    material_name: str
    density_kgm3: Optional[float]
    youngs_modulus_pa: Optional[float]
    poisson_ratio: Optional[float]
    priority_rank: Optional[int]
    basic_description: str
    functional_description: str
    movement_description: str
    joint_type: str


@dataclass
class GenesisObjectSpec:
    object_id: str
    object_name: str
    category: str
    dimension_cm: List[float]
    dimension_m: List[float]
    render_mesh_path: str
    collision_mesh_path: str
    urdf_path: str
    proxy_mode: str
    bbox_extents_m: List[float]
    part_mesh_paths: List[str]
    genesis_rigid: Dict[str, Any]
    genesis_parts: List[Dict[str, Any]]
    parts: List[PartSpec]
    proxy_stats: Dict[str, Any]
    urdf_stats: Dict[str, Any]


# =========================================================
# 通用工具
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def weighted_choice(weight_dict: Dict[str, float]) -> str:
    keys = list(weight_dict.keys())
    probs = np.asarray(list(weight_dict.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return str(np.random.choice(keys, p=probs))


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_density_to_kgm3(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", "")
    try:
        return float(s)
    except Exception:
        pass
    if "g/cm" in s:
        num = safe_float(s.split()[0])
        return None if num is None else float(num) * 1000.0
    if "kg/m" in s:
        num = safe_float(s.split()[0])
        return None if num is None else float(num)
    return None


def parse_youngs_to_pa(v: Optional[float]) -> Optional[float]:
    return None if v is None else float(v) * 1e9


def parse_dimension_to_cm(raw: str) -> List[float]:
    raw = (raw or "").strip().lower().replace("×", "x")
    if not raw:
        return [0.0, 0.0, 0.0]
    nums = []
    cur = ""
    for ch in raw:
        if ch.isdigit() or ch in ".-":
            cur += ch
        else:
            if cur:
                nums.append(cur)
                cur = ""
    if cur:
        nums.append(cur)
    vals = [float(x) for x in nums[:3]] if nums else [0.0, 0.0, 0.0]
    while len(vals) < 3:
        vals.append(0.0)
    if "mm" in raw:
        vals = [v / 10.0 for v in vals]
    elif "m" in raw and "cm" not in raw:
        vals = [v * 100.0 for v in vals]
    return vals


def cm_to_m(vals_cm: Sequence[float]) -> List[float]:
    return [float(x) / 100.0 for x in vals_cm]


def infer_joint_type(movement_description: str) -> str:
    if not movement_description:
        return "unknown"
    s = movement_description.lower()
    if any(k in s for k in ["rigidly fixed", "fixed to", "no movement", "rigidly attached"]):
        return "fixed"
    if any(k in s for k in ["rotate", "rotates", "hinge", "revolve", "swivel"]):
        return "revolute"
    if any(k in s for k in ["slide", "slides", "sliding", "pull out", "push in", "translate"]):
        return "prismatic"
    return "unknown"


def clamp_vec_norm(v: Sequence[float], max_norm: float) -> List[float]:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n > max_norm and n > 1e-8:
        arr = arr / n * max_norm
    return arr.astype(float).tolist()


def to_numpy_host(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def to_uint8_image(img: Any) -> np.ndarray:
    arr = to_numpy_host(img)
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_preview_video(preview_frames: List[np.ndarray], out_video_path: Path, fps: int = FPS) -> Optional[Path]:
    if len(preview_frames) == 0:
        return None
    ensure_dir(out_video_path.parent)
    preview_frames = [to_uint8_image(fr) for fr in preview_frames]
    try:
        writer = imageio.get_writer(
            str(out_video_path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,
        )
        for fr in preview_frames:
            writer.append_data(fr)
        writer.close()
        return out_video_path
    except Exception:
        pass
    gif_path = out_video_path.with_suffix(".gif")
    imageio.mimsave(str(gif_path), preview_frames, fps=fps)
    return gif_path


def safe_scene_destroy(scene: Any) -> None:
    if scene is None:
        return
    try:
        scene.destroy()
    except Exception:
        pass


def pick_distinct_colors(n: int) -> List[Tuple[float, float, float, float]]:
    palette = OBJECT_COLOR_PALETTE.copy()
    random.shuffle(palette)
    if n <= len(palette):
        return palette[:n]
    out = []
    while len(out) < n:
        out.extend(palette)
    return out[:n]


def object_bbox_volume(obj: Dict[str, Any]) -> float:
    ex = obj["geom"]["bbox_extents"]
    return float(ex[0] * ex[1] * ex[2])


def compute_scene_volume_ratio(objects: List[Dict[str, Any]]) -> float:
    vols = [object_bbox_volume(o) for o in objects if object_bbox_volume(o) > 1e-8]
    if len(vols) <= 1:
        return 1.0
    return float(max(vols) / min(vols))


def scene_volume_ratio_ok(objects: List[Dict[str, Any]], max_ratio: float = MAX_VOLUME_RATIO_IN_SCENE) -> bool:
    return compute_scene_volume_ratio(objects) <= max_ratio


def aabb_overlap_3d(
    c1: Sequence[float],
    e1: Sequence[float],
    c2: Sequence[float],
    e2: Sequence[float],
    margin: float = 0.03,
) -> bool:
    return (
        abs(c1[0] - c2[0]) < (e1[0] + e2[0]) / 2.0 + margin
        and abs(c1[1] - c2[1]) < (e1[1] + e2[1]) / 2.0 + margin
        and abs(c1[2] - c2[2]) < (e1[2] + e2[2]) / 2.0 + margin
    )


def floor_spawn_z(bbox_extents: Sequence[float], extra: float = 0.02) -> float:
    return float(max(bbox_extents[2] / 2.0 + extra, 0.08))


# =========================================================
# mesh / proxy 处理
# =========================================================
def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as trimesh.Trimesh: {mesh_path}")
    return mesh


def sanitize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    try:
        mesh.remove_infinite_values()
    except Exception:
        pass
    try:
        if not mesh.is_watertight and len(mesh.faces) < 300000:
            trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    return mesh


def merge_meshes(mesh_paths: List[Path], export_path: Path) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    meshes = []
    for p in mesh_paths:
        if not p.exists():
            continue
        meshes.append(sanitize_mesh(load_mesh(p)))
    if not meshes:
        raise FileNotFoundError(f"No valid meshes found for merge: {mesh_paths}")
    merged = trimesh.util.concatenate(meshes)
    merged = sanitize_mesh(merged)
    merged, _ = center_mesh_at_bbox(merged)
    merged.export(export_path)
    return export_path


def mesh_bbox_extents(mesh: trimesh.Trimesh) -> List[float]:
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float32)
    ext = np.maximum(ext, MIN_VALID_PROXY_EXTENT)
    return ext.astype(float).tolist()


def safe_mesh_volume(mesh: trimesh.Trimesh) -> Optional[float]:
    try:
        return float(abs(mesh.volume))
    except Exception:
        return None


def center_mesh_at_bbox(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, List[float]]:
    mesh = mesh.copy()
    center = ((mesh.bounds[0] + mesh.bounds[1]) / 2.0).astype(np.float32)
    mesh.apply_translation(-center)
    mesh = sanitize_mesh(mesh)
    return mesh, center.astype(float).tolist()


def export_bbox_proxy(mesh: trimesh.Trimesh, export_path: Path) -> Tuple[Path, List[float], Dict[str, Any]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    mesh, local_center = center_mesh_at_bbox(mesh)
    ext = mesh_bbox_extents(mesh)
    box = trimesh.creation.box(extents=ext)
    box.export(export_path)
    return export_path, ext, {
        "proxy_mode": "bbox_mesh",
        "num_proxy_parts": 1,
        "mesh_is_watertight": bool(mesh.is_watertight),
        "approx_volume": safe_mesh_volume(box),
        "local_bbox_center_before_centering": local_center,
    }


def export_convex_hull_proxy(mesh: trimesh.Trimesh, export_path: Path) -> Tuple[Path, List[float], Dict[str, Any]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    hull = sanitize_mesh(mesh.convex_hull)
    hull, local_center = center_mesh_at_bbox(hull)
    ext = mesh_bbox_extents(hull)
    hull.export(export_path)
    return export_path, ext, {
        "proxy_mode": "convex_hull",
        "num_proxy_parts": 1,
        "mesh_is_watertight": bool(hull.is_watertight),
        "approx_volume": safe_mesh_volume(hull),
        "local_bbox_center_before_centering": local_center,
    }


def export_part_hull_proxy(part_mesh_paths: List[Path], export_path: Path) -> Tuple[Path, List[float], Dict[str, Any]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    hulls = []
    for p in part_mesh_paths:
        if not p.exists():
            continue
        try:
            m = sanitize_mesh(load_mesh(p))
            h = sanitize_mesh(m.convex_hull)
            hulls.append(h)
        except Exception:
            continue
    if not hulls:
        raise FileNotFoundError("No valid part hulls can be exported.")
    merged = sanitize_mesh(trimesh.util.concatenate(hulls))
    merged, local_center = center_mesh_at_bbox(merged)
    ext = mesh_bbox_extents(merged)
    merged.export(export_path)
    return export_path, ext, {
        "proxy_mode": "part_hulls",
        "num_proxy_parts": len(hulls),
        "mesh_is_watertight": bool(merged.is_watertight),
        "approx_volume": safe_mesh_volume(merged),
        "local_bbox_center_before_centering": local_center,
    }


def export_part_box_proxy(part_mesh_paths: List[Path], export_path: Path) -> Tuple[Path, List[float], Dict[str, Any]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    boxes = []
    for p in part_mesh_paths:
        if not p.exists():
            continue
        try:
            m = sanitize_mesh(load_mesh(p))
            ext = mesh_bbox_extents(m)
            center = ((m.bounds[0] + m.bounds[1]) / 2.0).astype(float)
            b = trimesh.creation.box(extents=ext, transform=trimesh.transformations.translation_matrix(center))
            boxes.append(b)
        except Exception:
            continue
    if not boxes:
        raise FileNotFoundError("No valid part boxes can be exported.")
    merged = sanitize_mesh(trimesh.util.concatenate(boxes))
    merged, local_center = center_mesh_at_bbox(merged)
    ext = mesh_bbox_extents(merged)
    merged.export(export_path)
    return export_path, ext, {
        "proxy_mode": "part_boxes",
        "num_proxy_parts": len(boxes),
        "mesh_is_watertight": bool(merged.is_watertight),
        "approx_volume": safe_mesh_volume(merged),
        "local_bbox_center_before_centering": local_center,
    }


def choose_auto_proxy(part_mesh_paths: List[Path], merged_mesh_path: Path) -> str:
    try:
        mesh = sanitize_mesh(load_mesh(merged_mesh_path))
        ext = np.asarray(mesh_bbox_extents(mesh), dtype=np.float32)
        aspect = float(np.max(ext) / max(np.min(ext), 1e-6))
        if aspect > 8.0:
            return "bbox_mesh"
        # 默认更偏向稳定的单体 convex hull；
        # part_hulls 虽然更贴近原始几何，但在高速碰撞和离散步进下更容易出现穿透或抖动。
        if len(part_mesh_paths) >= 2:
            return "convex_hull"
        return "convex_hull"
    except Exception:
        return "bbox_mesh"


def build_collision_proxy(
    merged_mesh_path: Path,
    part_mesh_paths: List[Path],
    proxy_dir: Path,
    proxy_mode: str,
) -> Tuple[Path, List[float], Dict[str, Any]]:
    mesh = sanitize_mesh(load_mesh(merged_mesh_path))
    chosen = choose_auto_proxy(part_mesh_paths, merged_mesh_path) if proxy_mode == "auto" else proxy_mode
    if chosen == "merged":
        return merged_mesh_path, mesh_bbox_extents(mesh), {
            "proxy_mode": "merged",
            "num_proxy_parts": 1,
            "mesh_is_watertight": bool(mesh.is_watertight),
            "approx_volume": safe_mesh_volume(mesh),
        }
    if chosen == "convex_hull":
        return export_convex_hull_proxy(mesh, proxy_dir / "collision_convex_hull.obj")
    if chosen == "bbox_mesh":
        return export_bbox_proxy(mesh, proxy_dir / "collision_bbox.obj")
    if chosen == "part_hulls":
        return export_part_hull_proxy(part_mesh_paths, proxy_dir / "collision_part_hulls.obj")
    if chosen == "part_boxes":
        return export_part_box_proxy(part_mesh_paths, proxy_dir / "collision_part_boxes.obj")
    raise ValueError(f"Unknown proxy_mode: {proxy_mode}")


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def _urdf_relpath(target_path: Path, urdf_dir: Path) -> str:
    try:
        return target_path.resolve().relative_to(urdf_dir.resolve()).as_posix()
    except Exception:
        try:
            return Path(os.path.relpath(target_path, urdf_dir)).as_posix()
        except Exception:
            return str(target_path)


def write_single_rigid_urdf(
    urdf_path: Path,
    robot_name: str,
    render_mesh_path: Path,
    collision_mesh_path: Path,
    proxy_mode: str,
    bbox_extents_m: Sequence[float],
    color_rgba: Sequence[float] = (0.75, 0.75, 0.78, 1.0),
    mass_kg: float = 1.0,
) -> Tuple[Path, Dict[str, Any]]:
    ensure_dir(urdf_path.parent)
    render_rel = Path(os.path.relpath(render_mesh_path, urdf_path.parent)).as_posix()
    collision_rel = Path(os.path.relpath(collision_mesh_path, urdf_path.parent)).as_posix()

    robot = ET.Element("robot", attrib={"name": robot_name})
    link = ET.SubElement(robot, "link", attrib={"name": "base_link"})

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", attrib={"value": f"{float(max(mass_kg, 1e-4)):.8f}"})
    # 用 bbox 粗略给一个对角惯量，Genesis 若使用外部 material 也能继续工作。
    ex = [max(float(x), 1e-4) for x in bbox_extents_m]
    ixx = max(mass_kg * (ex[1] ** 2 + ex[2] ** 2) / 12.0, 1e-8)
    iyy = max(mass_kg * (ex[0] ** 2 + ex[2] ** 2) / 12.0, 1e-8)
    izz = max(mass_kg * (ex[0] ** 2 + ex[1] ** 2) / 12.0, 1e-8)
    ET.SubElement(
        inertial,
        "inertia",
        attrib={
            "ixx": f"{ixx:.8f}", "ixy": "0", "ixz": "0",
            "iyy": f"{iyy:.8f}", "iyz": "0", "izz": f"{izz:.8f}",
        },
    )

    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    vgeom = ET.SubElement(visual, "geometry")
    ET.SubElement(vgeom, "mesh", attrib={"filename": render_rel, "scale": "1 1 1"})
    material = ET.SubElement(visual, "material", attrib={"name": "obj_color"})
    ET.SubElement(material, "color", attrib={"rgba": " ".join(str(float(x)) for x in color_rgba)})

    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    cgeom = ET.SubElement(collision, "geometry")
    ET.SubElement(cgeom, "mesh", attrib={"filename": collision_rel, "scale": "1 1 1"})

    _indent_xml(robot)
    tree = ET.ElementTree(robot)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path, {
        "urdf_type": "single_rigid_body",
        "base_link": "base_link",
        "visual_mesh": str(render_mesh_path),
        "collision_mesh": str(collision_mesh_path),
        "proxy_mode": str(proxy_mode),
        "bbox_extents_m": [float(x) for x in bbox_extents_m],
    }


def build_object_urdf(
    obj_id: str,
    object_name: str,
    render_mesh_path: Path,
    collision_mesh_path: Path,
    bbox_extents_m: Sequence[float],
    proxy_mode: str,
    urdf_root: Path,
    mass_kg: float,
) -> Tuple[Path, Dict[str, Any]]:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in f"{object_name}_{obj_id}")
    urdf_dir = urdf_root / obj_id
    urdf_path = urdf_dir / f"{safe_name}.urdf"
    return write_single_rigid_urdf(
        urdf_path=urdf_path,
        robot_name=safe_name,
        render_mesh_path=render_mesh_path,
        collision_mesh_path=collision_mesh_path,
        proxy_mode=proxy_mode,
        bbox_extents_m=bbox_extents_m,
        mass_kg=mass_kg,
    )


def estimate_mesh_complexity(
    merged_mesh_path: Path,
    part_mesh_paths: List[Path],
    proxy_mode: str,
    n_parts: int,
) -> Dict[str, float]:
    face_count = 0
    vert_count = 0
    surface_area = 0.0
    watertight = 0.0
    try:
        mesh = sanitize_mesh(load_mesh(merged_mesh_path))
        face_count = int(len(mesh.faces))
        vert_count = int(len(mesh.vertices))
        surface_area = float(getattr(mesh, "area", 0.0) or 0.0)
        watertight = 1.0 if bool(mesh.is_watertight) else 0.0
    except Exception:
        pass

    part_count = int(max(n_parts, len(part_mesh_paths)))
    proxy_penalty = {
        "bbox_mesh": 1.00,
        "convex_hull": 0.70,
        "part_boxes": 0.45,
        "part_hulls": 0.18,
        "merged": 0.00,
    }.get(str(proxy_mode), 0.35)

    score = (
        1.35 * math.log1p(max(face_count, 0))
        + 0.55 * math.log1p(max(vert_count, 0))
        + 0.90 * math.log1p(max(part_count, 0))
        + 0.08 * math.log1p(max(surface_area, 0.0) * 1e4)
        + 0.35 * watertight
        - proxy_penalty
    )
    return {
        "render_face_count": float(face_count),
        "render_vertex_count": float(vert_count),
        "render_surface_area": float(surface_area),
        "complexity_score": float(score),
    }


def sample_complex_objects_for_scene(
    object_bank: List[Dict[str, Any]],
    n_obj: int,
) -> List[Dict[str, Any]]:
    if n_obj >= len(object_bank):
        return list(object_bank)

    if not PREFER_COMPLEX_OBJECTS:
        return random.sample(object_bank, k=n_obj)

    ordered = sorted(
        object_bank,
        key=lambda x: float(x.get("complexity_score", 0.0)),
        reverse=True,
    )
    topk = max(n_obj, int(math.ceil(len(ordered) * COMPLEX_OBJECT_TOP_RATIO)))
    candidate_pool = ordered[:topk]

    scores = np.asarray([max(float(x.get("complexity_score", 0.0)), 1e-3) for x in candidate_pool], dtype=np.float64)
    scores = np.power(scores / max(scores.max(), 1e-8), COMPLEX_OBJECT_WEIGHT_POWER)
    probs = scores / max(scores.sum(), 1e-12)
    chosen_idx = np.random.choice(len(candidate_pool), size=n_obj, replace=False, p=probs)
    return [candidate_pool[int(i)] for i in chosen_idx]


# =========================================================
# Loader
# =========================================================
class PhysXNetGenesisLoader:
    def __init__(
        self,
        root: str,
        version: str = "version_1",
        merged_cache_dir: Optional[str] = None,
        proxy_cache_dir: Optional[str] = None,
        urdf_cache_dir: Optional[str] = None,
        merge_ext: str = ".obj",
        proxy_mode: str = PROXY_MODE,
    ):
        self.root = Path(root)
        self.version = version
        self.base_dir = self.root / version
        self.finaljson_dir = self.base_dir / "finaljson"
        self.partseg_dir = self.base_dir / "partseg"
        self.merged_cache_dir = Path(merged_cache_dir) if merged_cache_dir else self.base_dir / "_merged_for_genesis"
        self.proxy_cache_dir = Path(proxy_cache_dir) if proxy_cache_dir else self.base_dir / "_collision_proxy_for_genesis"
        self.urdf_cache_dir = Path(urdf_cache_dir) if urdf_cache_dir else self.base_dir / "_urdf_for_genesis"
        self.merge_ext = merge_ext
        self.proxy_mode = proxy_mode
        if not self.finaljson_dir.exists():
            raise FileNotFoundError(f"finaljson dir not found: {self.finaljson_dir}")
        if not self.partseg_dir.exists():
            raise FileNotFoundError(f"partseg dir not found: {self.partseg_dir}")

    def __len__(self) -> int:
        return len(list(self.finaljson_dir.glob("*.json")))

    def list_object_ids(self) -> List[str]:
        return sorted([p.stem for p in self.finaljson_dir.glob("*.json")])

    def _find_img_for_part(self, imgs_dir: Path, part_id: int) -> Optional[Path]:
        if not imgs_dir.exists():
            return None
        cands = sorted(imgs_dir.glob(f"{part_id}_*.png"))
        if len(cands) == 0:
            cands = sorted(imgs_dir.glob(f"{part_id}_*"))
        return cands[0] if cands else None

    def _build_part_spec(self, obj_id: str, part_info: Dict[str, Any], objs_dir: Path, imgs_dir: Path) -> PartSpec:
        part_id = int(part_info["label"])
        mesh_path = objs_dir / f"{part_id}.obj"
        image_path = self._find_img_for_part(imgs_dir, part_id)
        material_name = str(part_info.get("material", "Unknown"))
        density_kgm3 = parse_density_to_kgm3(part_info.get("density"))
        young_pa = parse_youngs_to_pa(safe_float(part_info.get("Young's Modulus (GPa)")))
        poisson = safe_float(part_info.get("Poisson's Ratio"))
        basic_desc = str(part_info.get("Basic_description", ""))
        func_desc = str(part_info.get("Functional_description", ""))
        move_desc = str(part_info.get("Movement_description", ""))
        return PartSpec(
            part_id=part_id,
            name=str(part_info.get("name", f"part_{part_id}")),
            mesh_path=str(mesh_path),
            image_path=str(image_path) if image_path else None,
            material_name=material_name,
            density_kgm3=density_kgm3,
            youngs_modulus_pa=young_pa,
            poisson_ratio=poisson,
            priority_rank=safe_int(part_info.get("priority_rank")),
            basic_description=basic_desc,
            functional_description=func_desc,
            movement_description=move_desc,
            joint_type=infer_joint_type(move_desc),
        )

    def _build_genesis_part_dict(self, part: PartSpec) -> Dict[str, Any]:
        return {
            "name": part.name,
            "part_id": part.part_id,
            "entity_type": "rigid",
            "morph": {"type": "mesh", "file": part.mesh_path},
            "material": {
                "type": "rigid",
                "density": part.density_kgm3,
                "youngs_modulus": part.youngs_modulus_pa,
                "poisson_ratio": part.poisson_ratio,
                "material_name": part.material_name,
            },
            "semantic": {
                "priority_rank": part.priority_rank,
                "joint_type": part.joint_type,
                "basic_description": part.basic_description,
                "functional_description": part.functional_description,
                "movement_description": part.movement_description,
            },
        }

    def _build_genesis_rigid_dict(
        self,
        obj_id: str,
        object_name: str,
        render_mesh_path: Path,
        collision_mesh_path: Path,
        urdf_path: Path,
        proxy_mode: str,
        bbox_extents_m: List[float],
        proxy_stats: Dict[str, Any],
        urdf_stats: Dict[str, Any],
        parts: List[PartSpec],
    ) -> Dict[str, Any]:
        densities = [p.density_kgm3 for p in parts if p.density_kgm3 is not None]
        youngs = [p.youngs_modulus_pa for p in parts if p.youngs_modulus_pa is not None]
        poissons = [p.poisson_ratio for p in parts if p.poisson_ratio is not None]
        avg_density = float(np.mean(densities)) if densities else None
        avg_young = float(np.mean(youngs)) if youngs else None
        avg_poisson = float(np.mean(poissons)) if poissons else None
        materials = sorted(set(p.material_name for p in parts))
        return {
            "name": f"{object_name}_{obj_id}",
            "entity_type": "rigid",
            "morph": {"type": "urdf", "file": str(urdf_path)},
            "render": {"type": "mesh", "file": str(render_mesh_path)},
            "collision": {
                "proxy_mode": proxy_mode,
                "file": str(collision_mesh_path),
                "bbox_extents_m": bbox_extents_m,
                "proxy_stats": proxy_stats,
            },
            "urdf": urdf_stats,
            "material": {
                "type": "rigid",
                "density": avg_density,
                "youngs_modulus": avg_young,
                "poisson_ratio": avg_poisson,
                "material_names": materials,
            },
            "source": {
                "object_id": obj_id,
                "num_parts": len(parts),
                "note": "Rigid single-link URDF generated from PhysX-3D merged visual mesh + collision proxy mesh.",
            },
        }

    def get_object(
        self,
        obj_id: str,
        export_merged: bool = True,
        export_proxy: bool = True,
        export_urdf: bool = True,
    ) -> GenesisObjectSpec:
        json_path = self.finaljson_dir / f"{obj_id}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        part_dir = self.partseg_dir / obj_id
        objs_dir = part_dir / "objs"
        imgs_dir = part_dir / "imgs"
        if not objs_dir.exists():
            raise FileNotFoundError(f"objs dir not found: {objs_dir}")

        object_name = str(data.get("object_name", obj_id))
        category = str(data.get("category", "Unknown"))
        dim_cm = parse_dimension_to_cm(str(data.get("dimension", "")))
        dim_m = cm_to_m(dim_cm)

        parts_info = data.get("parts", [])
        parts = [self._build_part_spec(obj_id, pinfo, objs_dir, imgs_dir) for pinfo in sorted(parts_info, key=lambda x: int(x["label"]))]
        part_mesh_paths = [Path(p.mesh_path) for p in parts if Path(p.mesh_path).exists()]
        if len(part_mesh_paths) == 0:
            raise FileNotFoundError(f"No part meshes found for object {obj_id}")

        merged_mesh_path = self.merged_cache_dir / obj_id / f"merged{self.merge_ext}"
        if export_merged and not merged_mesh_path.exists():
            merge_meshes(part_mesh_paths, merged_mesh_path)
        ensure_dir(merged_mesh_path.parent)

        proxy_dir = self.proxy_cache_dir / self.proxy_mode / obj_id
        collision_mesh_path, bbox_extents_m, proxy_stats = build_collision_proxy(merged_mesh_path, part_mesh_paths, proxy_dir, self.proxy_mode)
        if not export_proxy:
            collision_mesh_path = merged_mesh_path
            proxy_stats = {"proxy_mode": "merged", "num_proxy_parts": 1}

        densities = [p.density_kgm3 for p in parts if p.density_kgm3 is not None and p.density_kgm3 > 0]
        mean_density = float(np.mean(densities)) if densities else 1000.0
        bbox = np.asarray(bbox_extents_m, dtype=np.float32)
        est_mass = float(max(np.prod(np.maximum(bbox, 1e-4)) * mean_density, 1e-4))

        urdf_path, urdf_stats = build_object_urdf(
            obj_id=obj_id,
            object_name=object_name,
            render_mesh_path=merged_mesh_path,
            collision_mesh_path=collision_mesh_path,
            bbox_extents_m=bbox_extents_m,
            proxy_mode=proxy_stats.get("proxy_mode", self.proxy_mode),
            urdf_root=self.urdf_cache_dir,
            mass_kg=est_mass,
        )

        genesis_parts = [self._build_genesis_part_dict(p) for p in parts]
        genesis_rigid = self._build_genesis_rigid_dict(
            obj_id=obj_id,
            object_name=object_name,
            render_mesh_path=merged_mesh_path,
            collision_mesh_path=collision_mesh_path,
            urdf_path=urdf_path,
            proxy_mode=proxy_stats.get("proxy_mode", self.proxy_mode),
            bbox_extents_m=bbox_extents_m,
            proxy_stats=proxy_stats,
            urdf_stats=urdf_stats,
            parts=parts,
        )

        return GenesisObjectSpec(
            object_id=obj_id,
            object_name=object_name,
            category=category,
            dimension_cm=dim_cm,
            dimension_m=dim_m,
            render_mesh_path=str(merged_mesh_path),
            collision_mesh_path=str(collision_mesh_path),
            urdf_path=str(urdf_path),
            proxy_mode=proxy_stats.get("proxy_mode", self.proxy_mode),
            bbox_extents_m=bbox_extents_m,
            part_mesh_paths=[str(p) for p in part_mesh_paths],
            genesis_rigid=genesis_rigid,
            genesis_parts=genesis_parts,
            parts=parts,
            proxy_stats=proxy_stats,
            urdf_stats=urdf_stats,
        )

    def iter_objects(self, export_merged: bool = True, export_proxy: bool = True, export_urdf: bool = True) -> Iterator[GenesisObjectSpec]:
        for obj_id in self.list_object_ids():
            try:
                yield self.get_object(
                    obj_id=obj_id,
                    export_merged=export_merged,
                    export_proxy=export_proxy,
                    export_urdf=export_urdf,
                )
            except Exception as e:
                print(f"[WARN] skip object {obj_id}: {e}")


# =========================================================
# 候选池与场景采样
# =========================================================
def infer_material_bucket(material_names: Sequence[str]) -> str:
    joined = " ".join(str(x).lower() for x in material_names)
    for key in ["metal", "glass", "plastic", "wood", "rubber", "foam", "fabric", "leather", "ceramic", "stone"]:
        if key in joined:
            return key
    return "default"


def sample_material_runtime_params(material_names: Sequence[str], density_hint: Optional[float]) -> Dict[str, float]:
    bucket = infer_material_bucket(material_names)
    prior = MATERIAL_PRIORS[bucket]
    friction = float(np.random.uniform(*prior["friction"]))
    restitution = float(np.random.uniform(*prior["restitution"]))
    linear_damping = float(np.random.uniform(*prior["linear_damping"]))
    rho = float(density_hint) if density_hint is not None else float(np.random.uniform(500.0, 2500.0))
    return {
        "bucket": bucket,
        "rho": rho,
        "friction": friction,
        "restitution": restitution,
        "linear_damping": linear_damping,
    }


def build_physxnet_object_bank(loader: PhysXNetGenesisLoader, max_objects_to_read: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_ids = loader.list_object_ids()
    if max_objects_to_read is not None and int(max_objects_to_read) > 0 and len(all_ids) > int(max_objects_to_read):
        # 先做一次固定随机子集裁剪，随后在 usable bank 内再按复杂度偏置采样；
        # 这样既控制预处理成本，又不会每个 scene 都只落到简单物体。
        rnd = random.Random(0)
        all_ids = sorted(rnd.sample(all_ids, k=int(max_objects_to_read)))

    bank, failed = [], []
    for i, obj_id in enumerate(all_ids, 1):
        try:
            obj = loader.get_object(obj_id, export_merged=EXPORT_MERGED_WHEN_LOADING, export_proxy=EXPORT_PROXY_WHEN_LOADING, export_urdf=EXPORT_URDF_WHEN_LOADING)
            dim_m = np.asarray(obj.dimension_m, dtype=np.float32)
            bbox_m = np.asarray(obj.bbox_extents_m, dtype=np.float32)
            ref = max(float(np.max(dim_m)), float(np.max(bbox_m)), 1e-6)
            if ref > 5.0:
                failed.append({"object_id": obj_id, "error": "too_large_raw_dimension"})
                continue
            ratio = float(np.max(bbox_m) / max(np.min(bbox_m), 1e-6))
            if ratio > 12.0:
                failed.append({"object_id": obj_id, "error": "extreme_aspect_ratio"})
                continue

            part_material_names = [p.material_name for p in obj.parts]
            avg_density = obj.genesis_rigid["material"].get("density", None)
            runtime_mat = sample_material_runtime_params(part_material_names, avg_density)
            complexity = estimate_mesh_complexity(
                merged_mesh_path=Path(obj.render_mesh_path),
                part_mesh_paths=[Path(p) for p in obj.part_mesh_paths],
                proxy_mode=obj.proxy_mode,
                n_parts=len(obj.parts),
            )
            bank.append({
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "category": obj.category,
                "dimension_m": obj.dimension_m,
                "bbox_extents_m": obj.bbox_extents_m,
                "render_mesh_path": obj.render_mesh_path,
                "collision_mesh_path": obj.collision_mesh_path,
                "urdf_path": obj.urdf_path,
                "proxy_mode": obj.proxy_mode,
                "proxy_stats": obj.proxy_stats,
                "genesis_rigid": obj.genesis_rigid,
                "urdf_stats": obj.urdf_stats,
                "parts": obj.parts,
                "n_parts": len(obj.parts),
                "material_names": part_material_names,
                "runtime_material_prior": runtime_mat,
                **complexity,
            })
            if i % 50 == 0 or i == len(all_ids):
                print(f"[INFO] loaded {i}/{len(all_ids)}")
        except Exception as e:
            failed.append({"object_id": obj_id, "error": str(e)})
            print(f"[WARN] skip {obj_id}: {e}")

    bank.sort(key=lambda x: float(x.get("complexity_score", 0.0)), reverse=True)
    return bank, failed


def split_object_bank_by_id(object_bank: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    ids = sorted([x["object_id"] for x in object_bank])
    rnd = random.Random(12345)
    rnd.shuffle(ids)
    n = len(ids)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    out = {"train": [], "val": [], "test": []}
    for x in object_bank:
        oid = x["object_id"]
        if oid in train_ids:
            out["train"].append(x)
        elif oid in val_ids:
            out["val"].append(x)
        else:
            out["test"].append(x)
    return out


def sample_background() -> Dict[str, Any]:
    presets = [
        {"name": "deep_slate", "background_color": [0.08, 0.10, 0.12], "ambient_light": [0.38, 0.38, 0.40]},
        {"name": "blue_stage", "background_color": [0.07, 0.10, 0.16], "ambient_light": [0.36, 0.38, 0.42]},
        {"name": "charcoal", "background_color": [0.10, 0.10, 0.11], "ambient_light": [0.36, 0.36, 0.36]},
        {"name": "green_room", "background_color": [0.08, 0.11, 0.10], "ambient_light": [0.36, 0.40, 0.38]},
        {"name": "purple_stage", "background_color": [0.09, 0.08, 0.12], "ambient_light": [0.38, 0.36, 0.40]},
    ]
    return random.choice(presets)


def build_corner_cfg_from_objects(objects: List[Dict[str, Any]], corner_base: Dict[str, Any]) -> Dict[str, Any]:
    points = collect_camera_interest_points(objects, corner_cfg=None)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    scene_center = 0.5 * (pmin + pmax)
    span_x = float(max(pmax[0] - pmin[0], 0.35))
    span_y = float(max(pmax[1] - pmin[1], 0.35))
    max_extent = max(float(np.max(obj["geom"]["bbox_extents"])) for obj in objects)

    required_w = span_x + 2.0 * max(SCENE_SIDE_MARGIN, 0.55 * max_extent)
    required_d = span_y + max(SCENE_FRONT_MARGIN, 0.95 * max_extent) + max(SCENE_BACK_MARGIN, 0.75 * max_extent)
    panel_size = float(np.clip(
        max(
            SCENE_PANEL_MOTION_SCALE * max(span_x, span_y, 0.65) + SCENE_PANEL_OBJECT_SCALE * max_extent,
            required_w,
            required_d,
        ),
        SCENE_PANEL_SIZE_MIN,
        SCENE_PANEL_SIZE_MAX,
    ))
    thickness = float(np.clip(
        SCENE_WALL_THICKNESS_BASE + SCENE_WALL_THICKNESS_SCALE * max_extent,
        0.045,
        0.12,
    ))

    center_x = float(scene_center[0] - 0.5 * panel_size)
    center_y = float(pmin[1] - max(SCENE_FRONT_MARGIN, 0.95 * max_extent))
    center_z = float((corner_base.get("center") or [0.0, 0.0, 0.0])[2])
    return {
        "center": [center_x, center_y, center_z],
        "panel_size": panel_size,
        "thickness": thickness,
        "layout": "u_three_walls_open_front",
        "open_side": "front",
        "motion_bbox": {
            "min": pmin.astype(float).tolist(),
            "max": pmax.astype(float).tolist(),
            "span": [span_x, span_y, float(max(pmax[2] - pmin[2], 0.20))],
        },
    }


def wall_clearance_for_extent(extent: float) -> float:
    extent = max(float(extent), 0.0)
    return float(WALL_CLEARANCE_BASE + WALL_CLEARANCE_EXTRA * extent)


def _corner_center(corner_cfg: Optional[Dict[str, Any]]) -> Tuple[float, float, float]:
    if corner_cfg is None:
        return 0.0, 0.0, 0.0
    center = corner_cfg.get("center") or corner_cfg.get("origin") or corner_cfg.get("corner_origin")
    if isinstance(center, (list, tuple)) and len(center) >= 3:
        return float(center[0]), float(center[1]), float(center[2])
    return 0.0, 0.0, 0.0


def _corner_panel_size(corner_cfg: Optional[Dict[str, Any]], default: float = 2.45) -> float:
    if corner_cfg is None:
        return float(default)
    for key in ("panel_size", "size", "scene_size", "room_size", "wall_span", "inner_size", "inner_span", "span"):
        if key in corner_cfg:
            try:
                return max(float(corner_cfg[key]), 0.6)
            except Exception:
                pass
    if all(k in corner_cfg for k in ("x_min", "x_max", "y_min", "y_max")):
        try:
            sx = float(corner_cfg["x_max"]) - float(corner_cfg["x_min"])
            sy = float(corner_cfg["y_max"]) - float(corner_cfg["y_min"])
            return max(sx, sy, 0.6)
        except Exception:
            pass
    bounds_xy = corner_cfg.get("bounds_xy")
    if isinstance(bounds_xy, (list, tuple)) and len(bounds_xy) >= 4:
        try:
            sx = float(bounds_xy[2]) - float(bounds_xy[0])
            sy = float(bounds_xy[3]) - float(bounds_xy[1])
            return max(sx, sy, 0.6)
        except Exception:
            pass
    return float(default)


def _corner_thickness(corner_cfg: Optional[Dict[str, Any]], default: float = 0.08) -> float:
    if corner_cfg is None:
        return float(default)
    for key in ("thickness", "wall_thickness", "panel_thickness"):
        if key in corner_cfg:
            try:
                return max(float(corner_cfg[key]), 0.02)
            except Exception:
                pass
    return float(default)


def corner_inner_bounds_xy(
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    edge_margin: float = 0.18,
) -> Tuple[float, float, float, float]:
    ex = [float(x) for x in bbox_extents]
    hx, hy = ex[0] / 2.0, ex[1] / 2.0
    clear_x = wall_clearance_for_extent(ex[0])
    clear_y = wall_clearance_for_extent(ex[1])
    if corner_cfg is None:
        x_min = clear_x + hx
        y_min = FRONT_OPEN_MARGIN + hy
        x_max = max(x_min + 0.45, 2.45 - clear_x - hx)
        y_max = max(y_min + 0.45, 1.95 - clear_y - hy)
        return float(x_min), float(y_min), float(x_max), float(y_max)

    cx, cy, _ = _corner_center(corner_cfg)
    big = _corner_panel_size(corner_cfg)
    visible_thick = max(MIN_VISIBLE_WALL_THICKNESS, _corner_thickness(corner_cfg) * VISIBLE_WALL_THICKNESS_SCALE)
    x_min = float(cx + visible_thick + clear_x + hx)
    x_max = float(cx + big - visible_thick - clear_x - hx)
    y_min = float(cy + FRONT_OPEN_MARGIN + hy)
    y_max = float(cy + big - visible_thick - edge_margin - clear_y - hy)
    if x_max <= x_min:
        x_mid = 0.5 * (x_min + x_max)
        x_min = x_mid - 0.025
        x_max = x_mid + 0.025
    if y_max <= y_min:
        y_mid = 0.5 * (y_min + y_max)
        y_min = y_mid - 0.025
        y_max = y_mid + 0.025
    return x_min, y_min, x_max, y_max

def keep_inside_corner_xy(
    xy: Sequence[float],
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
) -> List[float]:
    x_min, y_min, x_max, y_max = corner_inner_bounds_xy(bbox_extents, corner_cfg)
    x = float(np.clip(float(xy[0]), x_min, x_max))
    y = float(np.clip(float(xy[1]), y_min, y_max))
    return [x, y]


def keep_position_inside_corner(
    pos: Sequence[float],
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    floor_extra: float = 0.02,
) -> List[float]:
    ex = [float(x) for x in bbox_extents]
    xy = keep_inside_corner_xy(pos[:2], ex, corner_cfg)
    floor_z = floor_spawn_z(ex, extra=floor_extra)
    z = float(max(float(pos[2]), floor_z))
    return [xy[0], xy[1], z]


def make_velocity_wall_safe(
    pos: Sequence[float],
    bbox_extents: Sequence[float],
    linvel: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    toward_wall_cap_x: float = -1.10,
    toward_wall_cap_y: float = -1.00,
) -> List[float]:
    vel = np.asarray(linvel, dtype=np.float32).copy()
    x_min, y_min, _, _ = corner_inner_bounds_xy(bbox_extents, corner_cfg)
    wall_zone_x = x_min + max(0.08, 0.30 * float(bbox_extents[0]))
    wall_zone_y = y_min + max(0.08, 0.30 * float(bbox_extents[1]))
    if float(pos[0]) <= wall_zone_x and float(vel[0]) < 0.0:
        vel[0] = max(float(vel[0]), toward_wall_cap_x)
    if float(pos[1]) <= wall_zone_y and float(vel[1]) < 0.0:
        vel[1] = max(float(vel[1]), toward_wall_cap_y)
    return vel.astype(float).tolist()


def push_out_of_overlap(
    pos: Sequence[float],
    ex: Sequence[float],
    other_pos: Sequence[float],
    other_ex: Sequence[float],
    margin: float,
    corner_cfg: Optional[Dict[str, Any]] = None,
) -> List[float]:
    p = np.asarray(pos, dtype=np.float32)
    q = np.asarray(other_pos, dtype=np.float32)
    ex = np.asarray(ex, dtype=np.float32)
    oe = np.asarray(other_ex, dtype=np.float32)
    delta = p - q
    min_sep = 0.5 * (ex + oe) + float(margin)
    overlap = min_sep - np.abs(delta)
    if np.any(overlap <= 0):
        return keep_position_inside_corner(p.tolist(), ex.tolist(), corner_cfg)

    candidate_axes = [0, 1, 2]
    candidate_axes.sort(key=lambda a: float(overlap[a]))
    pushed = p.copy()
    for axis in candidate_axes:
        sign = 1.0 if float(delta[axis]) >= 0.0 else -1.0
        if abs(float(delta[axis])) < 1e-4:
            sign = 1.0 if axis != 2 else 1.0
        trial = pushed.copy()
        trial[axis] += sign * float(overlap[axis] + 0.025)
        trial = np.asarray(keep_position_inside_corner(trial.tolist(), ex.tolist(), corner_cfg), dtype=np.float32)
        if not aabb_overlap_3d(trial.tolist(), ex.tolist(), q.tolist(), oe.tolist(), margin=margin * 0.60):
            return trial.astype(float).tolist()
        pushed = trial

    pushed[2] = max(float(pushed[2]), float(q[2] + 0.5 * oe[2] + 0.5 * ex[2] + margin + 0.03))
    pushed = np.asarray(keep_position_inside_corner(pushed.tolist(), ex.tolist(), corner_cfg), dtype=np.float32)
    return pushed.astype(float).tolist()


def refine_scene_layout(objects: List[Dict[str, Any]], corner_cfg: Dict[str, Any], max_passes: int = 8) -> None:
    for obj in objects:
        ex = obj["geom"]["bbox_extents"]
        if not obj.get("start_offstage", False):
            obj["init_pos"] = keep_position_inside_corner(obj["init_pos"], ex, corner_cfg)
            obj["init_linvel"] = make_velocity_wall_safe(obj["init_pos"], ex, obj.get("init_linvel", [0.0, 0.0, 0.0]), corner_cfg)
        for ev in obj.get("script_events", []):
            if ev.get("type") == "teleport_and_set_motion":
                ev["request_pos"] = keep_position_inside_corner(ev["request_pos"], ex, corner_cfg, floor_extra=0.04)
                ev["linvel"] = make_velocity_wall_safe(ev["request_pos"], ex, ev.get("linvel", [0.0, 0.0, 0.0]), corner_cfg)
            elif ev.get("type") == "set_motion" and not obj.get("start_offstage", False):
                ev["linvel"] = make_velocity_wall_safe(obj["init_pos"], ex, ev.get("linvel", [0.0, 0.0, 0.0]), corner_cfg)
        if obj.get("camera_ref_pos") is not None:
            obj["camera_ref_pos"] = keep_position_inside_corner(obj["camera_ref_pos"], ex, corner_cfg, floor_extra=0.04)

    active_ids = [i for i, obj in enumerate(objects) if not obj.get("start_offstage", False)]
    for _ in range(max_passes):
        moved = False
        for idx_i, i in enumerate(active_ids):
            oi = objects[i]
            ex_i = oi["geom"]["bbox_extents"]
            pos_i = keep_position_inside_corner(oi["init_pos"], ex_i, corner_cfg)
            for j in active_ids[:idx_i]:
                oj = objects[j]
                margin = float(max(oi.get("placement_margin", 0.12), oj.get("placement_margin", 0.12)))
                if aabb_overlap_3d(pos_i, ex_i, oj["init_pos"], oj["geom"]["bbox_extents"], margin=margin):
                    pos_i = push_out_of_overlap(pos_i, ex_i, oj["init_pos"], oj["geom"]["bbox_extents"], margin=margin, corner_cfg=corner_cfg)
                    moved = True
            oi["init_pos"] = keep_position_inside_corner(pos_i, ex_i, corner_cfg)
            oi["init_linvel"] = make_velocity_wall_safe(oi["init_pos"], ex_i, oi.get("init_linvel", [0.0, 0.0, 0.0]), corner_cfg)
        if not moved:
            break

def collect_camera_interest_points(
    objects: List[Dict[str, Any]],
    corner_cfg: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    pts: List[List[float]] = []
    for obj in objects:
        ex = obj["geom"]["bbox_extents"]
        base_refs = [obj.get("camera_ref_pos"), obj.get("init_pos")]
        for ref in base_refs:
            if ref is not None:
                pts.append(keep_position_inside_corner(ref, ex, corner_cfg, floor_extra=0.02))

        init_pos = obj.get("init_pos")
        init_linvel = obj.get("init_linvel", [0.0, 0.0, 0.0])
        if init_pos is not None:
            p0 = np.asarray(init_pos, dtype=np.float32)
            v0 = np.asarray(init_linvel, dtype=np.float32)
            for horizon in (0.12, 0.28, 0.50, 0.80, 1.10):
                pf = p0 + v0 * float(horizon)
                pts.append(keep_position_inside_corner(pf.tolist(), ex, corner_cfg, floor_extra=0.02))

        for ev in obj.get("script_events", []):
            if ev.get("type") == "teleport_and_set_motion" and ev.get("request_pos") is not None:
                request = keep_position_inside_corner(ev["request_pos"], ex, corner_cfg, floor_extra=0.04)
                pts.append(request)
                rv = np.asarray(ev.get("linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
                rp = np.asarray(request, dtype=np.float32)
                for horizon in (0.10, 0.24, 0.45, 0.80):
                    pf = rp + rv * float(horizon)
                    pts.append(keep_position_inside_corner(pf.tolist(), ex, corner_cfg, floor_extra=0.03))
            elif ev.get("type") == "set_motion" and init_pos is not None:
                sv = np.asarray(ev.get("linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
                sp = np.asarray(init_pos, dtype=np.float32)
                for horizon in (0.12, 0.26, 0.50, 0.85):
                    pf = sp + sv * float(horizon)
                    pts.append(keep_position_inside_corner(pf.tolist(), ex, corner_cfg, floor_extra=0.02))

    if not pts:
        pts = [[0.0, 0.0, 0.25]]
    return np.asarray(pts, dtype=np.float32)



def _camera_basis(cam_pos: np.ndarray, lookat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = lookat - cam_pos
    f_norm = float(np.linalg.norm(forward))
    if f_norm < 1e-6:
        forward = np.asarray([0.0, 1.0, -0.1], dtype=np.float32)
        f_norm = float(np.linalg.norm(forward))
    forward = forward / f_norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    r_norm = float(np.linalg.norm(right))
    if r_norm < 1e-6:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        r_norm = float(np.linalg.norm(right))
    right = right / max(r_norm, 1e-6)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-6)
    return forward, right, up



def _bbox_corners(center: Sequence[float], bbox_extents: Sequence[float]) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    half = 0.5 * np.asarray(bbox_extents, dtype=np.float32)
    signs = np.asarray([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=np.float32)
    return c[None, :] + signs * half[None, :]



def collect_camera_projection_groups(
    objects: List[Dict[str, Any]],
    corner_cfg: Optional[Dict[str, Any]] = None,
) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for obj in objects:
        ex = np.asarray(obj["geom"]["bbox_extents"], dtype=np.float32)
        refs: List[np.ndarray] = []
        for ref in (obj.get("camera_ref_pos"), obj.get("init_pos")):
            if ref is not None:
                refs.append(np.asarray(keep_position_inside_corner(ref, ex.tolist(), corner_cfg, floor_extra=0.02), dtype=np.float32))

        init_pos = obj.get("init_pos")
        init_linvel = np.asarray(obj.get("init_linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
        if init_pos is not None:
            p0 = np.asarray(init_pos, dtype=np.float32)
            for horizon in (0.18, 0.40, 0.80):
                refs.append(np.asarray(keep_position_inside_corner((p0 + init_linvel * float(horizon)).tolist(), ex.tolist(), corner_cfg, floor_extra=0.02), dtype=np.float32))

        for ev in obj.get("script_events", []):
            if ev.get("type") == "teleport_and_set_motion" and ev.get("request_pos") is not None:
                request = np.asarray(keep_position_inside_corner(ev["request_pos"], ex.tolist(), corner_cfg, floor_extra=0.04), dtype=np.float32)
                refs.append(request)
                rv = np.asarray(ev.get("linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
                for horizon in (0.18, 0.45, 0.80):
                    refs.append(np.asarray(keep_position_inside_corner((request + rv * float(horizon)).tolist(), ex.tolist(), corner_cfg, floor_extra=0.03), dtype=np.float32))
            elif ev.get("type") == "set_motion" and init_pos is not None:
                p0 = np.asarray(init_pos, dtype=np.float32)
                sv = np.asarray(ev.get("linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
                for horizon in (0.18, 0.40, 0.80):
                    refs.append(np.asarray(keep_position_inside_corner((p0 + sv * float(horizon)).tolist(), ex.tolist(), corner_cfg, floor_extra=0.02), dtype=np.float32))

        if not refs:
            refs = [np.asarray([0.0, 0.0, max(0.2, 0.5 * float(ex[2]))], dtype=np.float32)]

        pts = np.concatenate([_bbox_corners(ref, ex.tolist()) for ref in refs], axis=0)
        groups.append(pts.astype(np.float32))
    return groups



def project_points_to_view(
    cam_pos: Sequence[float],
    lookat: Sequence[float],
    fov_deg: float,
    points: np.ndarray,
    aspect: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam = np.asarray(cam_pos, dtype=np.float32)
    target = np.asarray(lookat, dtype=np.float32)
    forward, right, up = _camera_basis(cam, target)
    rel = np.asarray(points, dtype=np.float32) - cam[None, :]
    z = rel @ forward
    x = rel @ right
    y = rel @ up
    tan_v = math.tan(math.radians(float(fov_deg)) * 0.5)
    tan_h = tan_v * float(aspect)
    nx = x / np.maximum(z * tan_h, 1e-6)
    ny = y / np.maximum(z * tan_v, 1e-6)
    u = 0.5 * (nx + 1.0)
    v = 0.5 * (1.0 - ny)
    return u, v, z, nx, ny



def _intersection_area(xmin: float, ymin: float, xmax: float, ymax: float) -> float:
    ix0 = max(0.0, float(xmin))
    iy0 = max(0.0, float(ymin))
    ix1 = min(1.0, float(xmax))
    iy1 = min(1.0, float(ymax))
    return max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)



def camera_candidate_metrics(
    cam_pos: Sequence[float],
    lookat: Sequence[float],
    fov_deg: float,
    points: np.ndarray,
    projection_groups: List[np.ndarray],
    aspect: float,
) -> Dict[str, float]:
    cam = np.asarray(cam_pos, dtype=np.float32)
    target = np.asarray(lookat, dtype=np.float32)
    u, v, z, nx, ny = project_points_to_view(cam, target, fov_deg, points, aspect)
    positive = z > 0.08
    if not np.any(positive):
        return {"score": -1e9, "occupancy": 0.0, "visible_ratio": 0.0, "mean_area": 0.0}

    hard_in = (np.abs(nx[positive]) <= 1.0) & (np.abs(ny[positive]) <= 1.0)
    soft_in = (np.abs(nx[positive]) <= 0.84) & (np.abs(ny[positive]) <= 0.84)
    edge_penalty = float(np.mean(np.maximum(np.abs(nx[positive]) - 0.94, 0.0) + np.maximum(np.abs(ny[positive]) - 0.94, 0.0)))
    center_penalty = float(abs(np.median(nx[positive])) + 0.7 * abs(np.median(ny[positive])))
    depth_penalty = float(np.std(z[positive]) / max(np.mean(z[positive]), 1e-6))

    scene_occ = _intersection_area(float(np.min(u[positive])), float(np.min(v[positive])), float(np.max(u[positive])), float(np.max(v[positive])))

    obj_areas: List[float] = []
    obj_visible: List[float] = []
    obj_edge_penalty = 0.0
    for grp in projection_groups:
        gu, gv, gz, _, _ = project_points_to_view(cam, target, fov_deg, grp, aspect)
        gpos = gz > 0.08
        if not np.any(gpos):
            obj_areas.append(0.0)
            obj_visible.append(0.0)
            obj_edge_penalty += 1.0
            continue
        xmin, ymin = float(np.min(gu[gpos])), float(np.min(gv[gpos]))
        xmax, ymax = float(np.max(gu[gpos])), float(np.max(gv[gpos]))
        area = _intersection_area(xmin, ymin, xmax, ymax)
        obj_areas.append(area)
        visible = 1.0 if area >= CAMERA_MIN_PER_OBJECT_AREA else max(area / max(CAMERA_MIN_PER_OBJECT_AREA, 1e-6), 0.0)
        obj_visible.append(float(np.clip(visible, 0.0, 1.0)))
        obj_edge_penalty += max(0.0, 0.02 - xmin) + max(0.0, xmax - 0.98) + max(0.0, 0.02 - ymin) + max(0.0, ymax - 0.98)

    visible_ratio = float(np.mean(obj_visible)) if obj_visible else 0.0
    mean_area = float(np.mean(obj_areas)) if obj_areas else 0.0
    occupancy_penalty = 3.5 * abs(scene_occ - CAMERA_OCCUPANCY_TARGET)
    if scene_occ < CAMERA_OCCUPANCY_MIN:
        occupancy_penalty += 8.5 * (CAMERA_OCCUPANCY_MIN - scene_occ)
    if scene_occ > CAMERA_OCCUPANCY_MAX:
        occupancy_penalty += 8.5 * (scene_occ - CAMERA_OCCUPANCY_MAX)
    tiny_penalty = float(np.mean([max(0.0, CAMERA_MIN_PER_OBJECT_AREA - a) for a in obj_areas])) * 18.0 if obj_areas else 0.0

    score = (
        4.5 * float(np.mean(hard_in))
        + 6.5 * float(np.mean(soft_in))
        + 5.5 * visible_ratio
        - 1.8 * edge_penalty
        - 1.1 * center_penalty
        - 0.6 * depth_penalty
        - occupancy_penalty
        - 0.7 * obj_edge_penalty
        - tiny_penalty
    )
    return {
        "score": float(score),
        "occupancy": float(scene_occ),
        "visible_ratio": float(visible_ratio),
        "mean_area": float(mean_area),
        "edge_penalty": float(edge_penalty + obj_edge_penalty),
    }





def normalize_camera_cfg(camera: Dict[str, Any]) -> Dict[str, Any]:
    cam = copy.deepcopy(camera)
    cam.setdefault("res", [IMG_W, IMG_H])
    cam.setdefault("GUI", False)
    cam.setdefault("camera_style", "manual_override")
    cam.setdefault("camera_metrics", {})
    pos = np.asarray(cam["pos"], dtype=np.float32)
    lookat = np.asarray(cam["lookat"], dtype=np.float32)
    cam["pos"] = pos.astype(float).tolist()
    cam["lookat"] = lookat.astype(float).tolist()
    cam["fov"] = float(cam["fov"])
    cam["distance"] = float(np.linalg.norm(pos - lookat))
    return cam


def load_manual_camera_cfg(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manual camera json not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return normalize_camera_cfg(data)


def dump_camera_cfg(camera: Dict[str, Any], out_path: Path) -> Path:
    ensure_dir(out_path.parent)
    cam = normalize_camera_cfg(camera)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cam, f, ensure_ascii=False, indent=2)
    return out_path


def build_manual_camera_from_scene(
    objects: List[Dict[str, Any]],
    corner_cfg: Dict[str, Any],
    manual_cam: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if manual_cam is None:
        points = collect_camera_interest_points(objects, corner_cfg)
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        center = 0.5 * (pmin + pmax)
        max_extent = max(float(np.max(obj["geom"]["bbox_extents"])) for obj in objects)
        x_min, y_min, x_max, y_max = corner_inner_bounds_xy([max_extent, max_extent, max_extent], corner_cfg)
        centerline_x = 0.5 * (x_min + x_max)
        lookat = np.asarray([
            float(centerline_x),
            float(np.clip(center[1], y_min + 0.18, y_max - 0.18)),
            float(max(0.18, center[2])),
        ], dtype=np.float32)
        manual_cam = {
            "pos": [float(centerline_x), float(y_min - 0.28), float(max(0.55, center[2] + 0.32))],
            "lookat": lookat.astype(float).tolist(),
            "fov": 35.0,
            "res": [IMG_W, IMG_H],
            "GUI": False,
            "camera_style": "manual_override_default",
        }
    return normalize_camera_cfg(manual_cam)


def resolve_camera_override(
    objects: Optional[List[Dict[str, Any]]] = None,
    corner_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if MANUAL_CAMERA_JSON_PATH:
        return load_manual_camera_cfg(MANUAL_CAMERA_JSON_PATH)
    if USE_MANUAL_CAMERA:
        if MANUAL_CAMERA is not None:
            return normalize_camera_cfg(MANUAL_CAMERA)
        if objects is not None and corner_cfg is not None:
            return build_manual_camera_from_scene(objects, corner_cfg, None)
    return None


def clone_scene_cfg_with_camera(scene_cfg: Dict[str, Any], camera_override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(scene_cfg)
    if camera_override is not None:
        cfg["camera"] = normalize_camera_cfg(camera_override)
    return cfg

def sample_camera_from_objects(objects: List[Dict[str, Any]], corner_cfg: Dict[str, Any]) -> Dict[str, Any]:
    camera_override = resolve_camera_override(objects, corner_cfg)
    if camera_override is not None:
        return normalize_camera_cfg(camera_override)

    points = collect_camera_interest_points(objects, corner_cfg)
    projection_groups = collect_camera_projection_groups(objects, corner_cfg)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    scene_center = 0.5 * (pmin + pmax)
    span_x = float(max(pmax[0] - pmin[0], 1e-4))
    span_y = float(max(pmax[1] - pmin[1], 1e-4))
    span_z = float(max(pmax[2] - pmin[2], 1e-4))
    motion_span = max(span_x, span_y, 0.65)
    aspect = float(IMG_W) / max(float(IMG_H), 1.0)

    max_extent = max(float(np.max(obj["geom"]["bbox_extents"])) for obj in objects)
    x_min, y_min, x_max, y_max = corner_inner_bounds_xy([max_extent, max_extent, max_extent], corner_cfg)
    room_w = max(x_max - x_min, 0.8)
    room_d = max(y_max - y_min, 0.8)
    centerline_x = 0.5 * (x_min + x_max)

    focus_center = np.asarray([
        float(centerline_x),
        float(np.clip(pmin[1] + 0.64 * span_y, y_min + 0.26 * room_d, y_max - 0.12 * room_d)),
        float(np.clip(pmin[2] + 0.16 * span_z, 0.12, max(0.22, pmin[2] + 0.30 * span_z))),
    ], dtype=np.float32)

    base_dist = float(np.clip(
        CAMERA_BASE_DIST_SCALE * motion_span + CAMERA_HEIGHT_SCALE * span_z + 0.58 * max_extent,
        CAMERA_MIN_DIST,
        CAMERA_MAX_DIST,
    ))
    cam_y = float(y_min - np.clip(CAMERA_OUTSIDE_FRONT_GAP + 0.05 * motion_span, 0.16, 0.34))
    z_base = float(np.clip(
        focus_center[2] + 0.03 * base_dist + 0.03 * max_extent,
        0.42, 1.05,
    ))

    candidate_specs = [
        {"x_mul": 0.00, "z_mul": -0.08, "look_z_mul": -0.02, "fov": 33.0},
        {"x_mul": -0.03, "z_mul": -0.05, "look_z_mul": -0.02, "fov": 34.0},
        {"x_mul": 0.00, "z_mul": -0.02, "look_z_mul": -0.01, "fov": 35.0},
        {"x_mul": 0.03, "z_mul": 0.00, "look_z_mul": -0.01, "fov": 36.0},
        {"x_mul": 0.00, "z_mul": 0.03, "look_z_mul": 0.00, "fov": 37.0},
    ]

    best = None
    for spec in candidate_specs:
        lookat = focus_center.copy()
        lookat[0] = float(centerline_x)
        lookat[2] = float(max(0.16, lookat[2] + spec["look_z_mul"] * base_dist))

        cam_x = float(centerline_x + spec["x_mul"] * room_w)
        cam_z = float(max(z_base + spec["z_mul"] * base_dist, lookat[2] + 0.03, 0.42))
        cam_pos = np.asarray([cam_x, cam_y, cam_z], dtype=np.float32)

        metrics = camera_candidate_metrics(cam_pos, lookat, spec["fov"], points, projection_groups, aspect)
        dist = float(np.linalg.norm(cam_pos - lookat))
        metrics["score"] -= 0.12 * abs(dist - base_dist)
        metrics["score"] -= 0.05 * abs(cam_x - float(scene_center[0])) / max(room_w, 1e-4)

        if best is None or metrics["score"] > best[0]:
            best = (metrics["score"], cam_pos, lookat, spec["fov"], metrics)

    _, cam_pos, lookat, fov, metrics = best
    return normalize_camera_cfg({
        "res": [IMG_W, IMG_H],
        "pos": cam_pos.astype(float).tolist(),
        "lookat": lookat.astype(float).tolist(),
        "fov": float(fov),
        "GUI": False,
        "camera_style": "outside_open_front_object_coverage_scored",
        "camera_metrics": metrics,
    })


    candidate_specs = [
    {"z_mul": -0.05, "look_z_mul": -0.02, "fov": 34.0},
    {"z_mul": -0.02, "look_z_mul": -0.01, "fov": 35.0},
    {"z_mul":  0.00, "look_z_mul": -0.01, "fov": 36.0},
]

    best = None
    for spec in candidate_specs:
        lookat = focus_center.copy()
        lookat[0] = float(centerline_x)
        lookat[2] = float(max(0.16, lookat[2] + spec["look_z_mul"] * base_dist))

        cam_x = float(centerline_x)
        cam_z = float(max(z_base + spec["z_mul"] * base_dist, lookat[2] + 0.03, 0.42))
        cam_pos = np.asarray([cam_x, cam_y, cam_z], dtype=np.float32)

        metrics = camera_candidate_metrics(cam_pos, lookat, spec["fov"], points, projection_groups, aspect)
        dist = float(np.linalg.norm(cam_pos - lookat))
        metrics["score"] -= 0.12 * abs(dist - base_dist)
        metrics["score"] -= 0.05 * abs(cam_x - float(scene_center[0])) / max(room_w, 1e-4)

        if best is None or metrics["score"] > best[0]:
            best = (metrics["score"], cam_pos, lookat, spec["fov"], metrics)

    _, cam_pos, lookat, fov, metrics = best
    dist = float(np.linalg.norm(cam_pos - lookat))
    return {
        "res": [IMG_W, IMG_H],
        "pos": cam_pos.astype(float).tolist(),
        "lookat": lookat.astype(float).tolist(),
        "fov": float(fov),
        "GUI": False,
        "distance": dist,
        "camera_style": "outside_open_front_object_coverage_scored",
        "camera_metrics": metrics,
    }




def family_prefers_floor_motion(family: str) -> bool:
    return family in {
        # "rolling_push",
        # "floor_slide",
        # "diagonal_sweep",
        # "rest_then_hit",
        # "line_chain_collision",
        # "cross_fire",
        # "stack_drop",
        # "late_entry",
        # "sequential_entry",
        "static_then_dual_hit",
        # "mixed_multi",
        # "front_entry_slide",
        # "left_right_pingpong",
    }


def sample_fractional_corner_position(
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    x_frac_range: Tuple[float, float] = (0.15, 0.85),
    y_frac_range: Tuple[float, float] = (0.15, 0.85),
    z_frac_range: Tuple[float, float] = (0.00, 0.00),
    floor_only: bool = False,
    floor_extra: float = 0.02,
) -> List[float]:
    ex = [float(x) for x in bbox_extents]
    x_min, y_min, x_max, y_max = corner_inner_bounds_xy(ex, corner_cfg)
    x_lo, x_hi = sorted((float(x_frac_range[0]), float(x_frac_range[1])))
    y_lo, y_hi = sorted((float(y_frac_range[0]), float(y_frac_range[1])))
    x_lo = float(np.clip(x_lo, 0.0, 1.0))
    x_hi = float(np.clip(x_hi, x_lo, 1.0))
    y_lo = float(np.clip(y_lo, 0.0, 1.0))
    y_hi = float(np.clip(y_hi, y_lo, 1.0))
    x = float(np.random.uniform(x_min + x_lo * (x_max - x_min), x_min + x_hi * (x_max - x_min)))
    y = float(np.random.uniform(y_min + y_lo * (y_max - y_min), y_min + y_hi * (y_max - y_min)))
    floor_z = floor_spawn_z(ex, extra=floor_extra)
    if floor_only:
        z = floor_z
    else:
        scene_h = float(_corner_panel_size(corner_cfg, default=2.9) * 0.62) if corner_cfg is not None else 1.8
        z_lo, z_hi = sorted((float(z_frac_range[0]), float(z_frac_range[1])))
        z_lo = float(np.clip(z_lo, 0.0, 1.0))
        z_hi = float(np.clip(z_hi, z_lo, 1.25))
        z = float(np.random.uniform(floor_z + z_lo * scene_h, floor_z + z_hi * scene_h))
    return keep_position_inside_corner([x, y, z], ex, corner_cfg, floor_extra=floor_extra)


def sample_non_overlapping_position(
    existing_objects: List[Dict[str, Any]],
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    x_frac_range: Tuple[float, float] = (0.15, 0.85),
    y_frac_range: Tuple[float, float] = (0.15, 0.85),
    z_frac_range: Tuple[float, float] = (0.00, 0.00),
    floor_only: bool = False,
    floor_extra: float = 0.02,
    max_trials: int = 320,
) -> List[float]:
    ex = list(map(float, bbox_extents))
    for _ in range(max_trials):
        pos = sample_fractional_corner_position(
            ex,
            corner_cfg=corner_cfg,
            x_frac_range=x_frac_range,
            y_frac_range=y_frac_range,
            z_frac_range=z_frac_range,
            floor_only=floor_only,
            floor_extra=floor_extra,
        )
        collide = False
        for obj in existing_objects:
            margin = float(max(0.10, obj.get("placement_margin", 0.12)))
            if aabb_overlap_3d(pos, ex, obj["init_pos"], obj["geom"]["bbox_extents"], margin=margin):
                collide = True
                break
        if not collide:
            return pos
    fallback = sample_fractional_corner_position(
        ex,
        corner_cfg=corner_cfg,
        x_frac_range=(0.22, 0.78),
        y_frac_range=(0.22, 0.78),
        z_frac_range=(0.10, 0.50),
        floor_only=floor_only,
        floor_extra=floor_extra,
    )
    return keep_position_inside_corner(fallback, ex, corner_cfg, floor_extra=floor_extra)


def center_xy_from_corner(bbox_extents: Sequence[float], corner_cfg: Optional[Dict[str, Any]] = None) -> List[float]:
    x_min, y_min, x_max, y_max = corner_inner_bounds_xy(bbox_extents, corner_cfg)
    return [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)]


def frame_safe_bounds_xy(
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    margin_frac: float = 0.14,
) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = corner_inner_bounds_xy(bbox_extents, corner_cfg)
    room_w = max(x_max - x_min, 1e-4)
    room_d = max(y_max - y_min, 1e-4)
    mx = min(room_w * 0.30, max(0.05, room_w * float(margin_frac)))
    my = min(room_d * 0.30, max(0.05, room_d * float(margin_frac)))
    sx0, sy0, sx1, sy1 = x_min + mx, y_min + my, x_max - mx, y_max - my
    if sx0 >= sx1:
        mid = 0.5 * (x_min + x_max)
        span = max(0.03, 0.18 * room_w)
        sx0, sx1 = mid - span, mid + span
    if sy0 >= sy1:
        mid = 0.5 * (y_min + y_max)
        span = max(0.03, 0.18 * room_d)
        sy0, sy1 = mid - span, mid + span
    return float(sx0), float(sy0), float(sx1), float(sy1)


def clamp_xy_to_frame_safe(
    xy: Sequence[float],
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    margin_frac: float = 0.14,
) -> List[float]:
    sx0, sy0, sx1, sy1 = frame_safe_bounds_xy(bbox_extents, corner_cfg, margin_frac=margin_frac)
    return [float(np.clip(float(xy[0]), sx0, sx1)), float(np.clip(float(xy[1]), sy0, sy1))]


def keep_linvel_in_frame(
    pos: Sequence[float],
    bbox_extents: Sequence[float],
    linvel: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]] = None,
    horizon_sec: float = 0.85,
    margin_frac: float = 0.14,
) -> List[float]:
    if corner_cfg is None:
        return [float(x) for x in linvel]
    v = np.asarray(linvel, dtype=np.float32).copy()
    speed_xy = float(np.linalg.norm(v[:2]))
    if speed_xy < 1e-6:
        return v.astype(float).tolist()
    pos_xy = np.asarray(pos[:2], dtype=np.float32)
    future_xy = pos_xy + v[:2] * float(horizon_sec)
    clamped_xy = np.asarray(clamp_xy_to_frame_safe(future_xy.tolist(), bbox_extents, corner_cfg, margin_frac=margin_frac), dtype=np.float32)
    if float(np.linalg.norm(future_xy - clamped_xy)) < 1e-4:
        return v.astype(float).tolist()
    delta = clamped_xy - pos_xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        v[0] = 0.0
        v[1] = 0.0
        return v.astype(float).tolist()
    new_speed_xy = min(speed_xy, dist / max(float(horizon_sec), 1e-4))
    v[:2] = delta / dist * new_speed_xy
    return v.astype(float).tolist()


def runtime_parking_position(
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]],
    lane: str = "back",
    slot: float = 0.5,
    floor_extra: float = 0.02,
) -> List[float]:
    ex = list(map(float, bbox_extents))
    if corner_cfg is None:
        return [0.0, -4.0, floor_spawn_z(ex, extra=floor_extra)]
    cx, cy, _ = _corner_center(corner_cfg)
    big = _corner_panel_size(corner_cfg)
    visible_thick = max(MIN_VISIBLE_WALL_THICKNESS, _corner_thickness(corner_cfg) * VISIBLE_WALL_THICKNESS_SCALE)
    guard_thick = max(MIN_GUARD_WALL_THICKNESS, _corner_thickness(corner_cfg) * GUARD_WALL_THICKNESS_SCALE)
    slot = float(np.clip(slot, 0.08, 0.92))
    offset = visible_thick + guard_thick + RUNTIME_PARKING_OFFSET + 1.2 * max(ex[0], ex[1], ex[2])
    floor_z = floor_spawn_z(ex, extra=floor_extra)
    if lane == "left":
        x = cx - offset
        y = cy + slot * big
    elif lane == "right":
        x = cx + big + offset
        y = cy + slot * big
    else:
        x = cx + slot * big
        y = cy + big + offset
    return [float(x), float(y), float(floor_z)]


def edge_request_position(
    bbox_extents: Sequence[float],
    corner_cfg: Optional[Dict[str, Any]],
    side: str,
    slot: float,
    z_frac: float = 0.04,
    floor_extra: float = 0.03,
) -> List[float]:
    ex = list(map(float, bbox_extents))
    x_min, y_min, x_max, y_max = frame_safe_bounds_xy(ex, corner_cfg, margin_frac=0.10)
    slot = float(np.clip(slot, 0.08, 0.92))
    floor_z = floor_spawn_z(ex, extra=floor_extra)
    scene_h = float(_corner_panel_size(corner_cfg, default=2.9) * 0.55) if corner_cfg is not None else 1.6
    z = float(floor_z + z_frac * scene_h)
    side = {"top": "back", "diag": "back_right"}.get(side, side)
    if side == "right":
        x = x_max
        y = y_min + slot * (y_max - y_min)
    elif side == "left":
        x = x_min
        y = y_min + slot * (y_max - y_min)
    elif side == "front":
        x = x_min + slot * (x_max - x_min)
        y = y_min
    elif side == "back":
        x = x_min + slot * (x_max - x_min)
        y = y_max
    elif side == "back_left":
        x = x_min
        y = y_max - slot * 0.24 * (y_max - y_min)
    else:
        x = x_max
        y = y_max - slot * 0.24 * (y_max - y_min)
    return keep_position_inside_corner([x, y, z], ex, corner_cfg, floor_extra=floor_extra)

def make_toward_point_velocity(
    start_pos: Sequence[float],
    target_xy: Sequence[float],
    speed_xy: float,
    lift_z: float = 0.0,
    tangent_jitter: float = 0.10,
) -> List[float]:
    start_xy = np.asarray(start_pos[:2], dtype=np.float32)
    target_xy = np.asarray(target_xy[:2], dtype=np.float32)
    delta = target_xy - start_xy
    norm = float(np.linalg.norm(delta))
    if norm < 1e-5:
        theta = float(np.random.uniform(-math.pi, math.pi))
        delta = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
        norm = 1.0
    direction = delta / norm
    tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    mixed = direction + tangent * float(np.random.uniform(-tangent_jitter, tangent_jitter))
    mixed_norm = float(np.linalg.norm(mixed))
    if mixed_norm > 1e-6:
        mixed = mixed / mixed_norm
    vel_xy = mixed * float(speed_xy)
    return [float(vel_xy[0]), float(vel_xy[1]), float(lift_z)]


def make_spin(scale: float = 1.0, biased_axis: Optional[int] = None) -> List[float]:
    spin = np.asarray([np.random.uniform(-3.5, 3.5) for _ in range(3)], dtype=np.float32) * float(scale)
    if biased_axis is not None:
        spin[biased_axis] += float(np.random.uniform(2.0, 5.0)) * np.sign(np.random.uniform(-1.0, 1.0) + 1e-4)
    return spin.astype(float).tolist()


def sample_motion_for_family(
    family: str,
    obj_rank: int,
    total_objects: int,
    bbox_extents: Sequence[float],
    existing_objects: List[Dict[str, Any]],
    corner_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ex = list(map(float, bbox_extents))
    init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg=corner_cfg)
    init_euler = [
        float(np.random.uniform(-0.25, 0.25)),
        float(np.random.uniform(-0.25, 0.25)),
        float(np.random.uniform(-math.pi, math.pi)),
    ]
    linvel = [0.0, 0.0, 0.0]
    angvel = make_spin(scale=0.55)
    events: List[Dict[str, Any]] = []
    mode_name = family
    center_xy = center_xy_from_corner(ex, corner_cfg)
    x_min, y_min, x_max, y_max = corner_inner_bounds_xy(ex, corner_cfg)
    n = max(total_objects - 1, 1)
    slot = obj_rank / n if total_objects > 1 else 0.5
    alt = -1.0 if obj_rank % 2 == 0 else 1.0
    start_offstage = False
    activation_frame = 0
    camera_ref_pos = init_pos

    def register_runtime_entry(
        request_pos: Sequence[float],
        target_xy: Sequence[float],
        frame_idx: int,
        speed_xy_range: Tuple[float, float],
        lift_range: Tuple[float, float],
        tangent_jitter: float,
        ang_scale: float,
        park_lane: str = "back",
        park_slot: Optional[float] = None,
    ) -> None:
        nonlocal init_pos, linvel, angvel, events, start_offstage, activation_frame, camera_ref_pos
        request = [float(x) for x in request_pos]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        request[:2] = clamp_xy_to_frame_safe(request[:2], ex, corner_cfg, margin_frac=0.10)
        pslot = slot if park_slot is None else float(park_slot)
        init_pos = runtime_parking_position(ex, corner_cfg, lane=park_lane, slot=pslot, floor_extra=0.02)
        linvel = [0.0, 0.0, 0.0]
        angvel = [0.0, 0.0, 0.0]
        events.append({
            "type": "teleport_and_set_motion",
            "frame_idx": int(max(0, frame_idx)),
            "request_pos": request,
            "linvel": make_toward_point_velocity(request, target_xy, speed_xy=float(np.random.uniform(*speed_xy_range)), lift_z=float(np.random.uniform(*lift_range)), tangent_jitter=tangent_jitter),
            "angvel": make_spin(scale=ang_scale),
        })
        start_offstage = True
        activation_frame = int(max(0, frame_idx))
        floor_z = floor_spawn_z(ex, extra=0.03)
        camera_ref_pos = [
            float(0.45 * request[0] + 0.55 * target_xy[0]),
            float(0.45 * request[1] + 0.55 * target_xy[1]),
            float(max(request[2], floor_z + 0.12)),
        ]

    if family == "free_drop":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.08, 0.92), (0.18, 0.88), (0.55, 1.02), floor_only=False)
        angvel = make_spin(scale=0.40)
        mode_name = "free_drop_wide"
    elif family == "vertical_bounce":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.18, 0.82), (0.20, 0.84), (0.58, 1.05), floor_only=False)
        linvel = [float(np.random.uniform(-0.08, 0.08)), float(np.random.uniform(-0.08, 0.08)), float(np.random.uniform(-0.55, -0.15))]
        angvel = make_spin(scale=0.35)
        mode_name = "vertical_bounce_drop"
    elif family == "upward_toss":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.22, 0.78), (0.18, 0.40), floor_only=True, floor_extra=0.03)
        linvel = [float(np.random.uniform(-0.30, 0.30)), float(np.random.uniform(0.50, 1.15)), float(np.random.uniform(1.25, 2.10))]
        angvel = make_spin(scale=0.95)
        mode_name = "upward_toss_front"
    elif family == "underhand_arc":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.10, 0.30), (0.16 + 0.10 * slot, 0.30 + 0.10 * slot), floor_only=True, floor_extra=0.03)
        target_xy = [center_xy[0] + 0.08 * (x_max - x_min), center_xy[1] + alt * 0.08 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.15, 1.95)), lift_z=float(np.random.uniform(0.95, 1.65)), tangent_jitter=0.06)
        angvel = make_spin(scale=0.80)
        mode_name = "underhand_arc"
    elif family == "oblique_throw":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.76, 0.96), (0.18, 0.86), (0.18, 0.55), floor_only=False)
        target_xy = [center_xy[0] - 0.06 * (x_max - x_min), center_xy[1] + alt * 0.18 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.35, 2.25)), lift_z=float(np.random.uniform(0.55, 1.45)), tangent_jitter=0.10)
        angvel = make_spin(scale=1.05)
        mode_name = "oblique_throw_to_center"
    elif family == "side_throw":
        side_x = (0.06, 0.18) if obj_rank % 2 == 0 else (0.82, 0.96)
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, side_x, (0.18 + 0.08 * slot, 0.34 + 0.10 * slot), (0.10, 0.34), floor_only=False)
        target_xy = [center_xy[0] + alt * 0.04 * (x_max - x_min), center_xy[1] + 0.12 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.35, 2.05)), lift_z=float(np.random.uniform(0.35, 0.95)), tangent_jitter=0.05)
        angvel = make_spin(scale=1.10)
        mode_name = "side_throw_low_arc"
    elif family == "bank_shot":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.18, 0.82), (0.12, 0.24), (0.04, 0.20), floor_only=False)
        target_xy = [center_xy[0], y_max - 0.10 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.25, 1.95)), lift_z=float(np.random.uniform(0.12, 0.40)), tangent_jitter=0.03)
        angvel = make_spin(scale=0.85)
        mode_name = "bank_shot_backwall"
    elif family == "rolling_push":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.10, 0.26), (0.18 + 0.12 * slot, 0.32 + 0.12 * slot), floor_only=True, floor_extra=0.01)
        target_xy = [x_max - 0.12 * (x_max - x_min), init_pos[1] + alt * 0.05 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(0.90, 1.75)), lift_z=0.0, tangent_jitter=0.02)
        angvel = make_spin(scale=0.90, biased_axis=1)
        mode_name = "rolling_push"
    elif family == "floor_slide":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.12, 0.24), (0.14 + 0.10 * slot, 0.30 + 0.12 * slot), floor_only=True, floor_extra=0.01)
        target_xy = [x_max - 0.18 * (x_max - x_min), y_min + 0.76 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(0.85, 1.60)), lift_z=0.0, tangent_jitter=0.03)
        angvel = [0.0, 0.0, float(np.random.uniform(-1.2, 1.2))]
        mode_name = "floor_slide_diag"
    elif family == "diagonal_sweep":
        if obj_rank % 2 == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.08, 0.20), (0.14, 0.28), floor_only=True, floor_extra=0.02)
            target_xy = [x_max - 0.18 * (x_max - x_min), y_max - 0.10 * (y_max - y_min)]
        else:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.80, 0.94), (0.14, 0.28), floor_only=True, floor_extra=0.02)
            target_xy = [x_min + 0.16 * (x_max - x_min), y_max - 0.16 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.00, 1.80)), lift_z=0.0, tangent_jitter=0.02)
        angvel = make_spin(scale=0.55, biased_axis=2)
        mode_name = "diagonal_sweep"
    elif family == "rest_then_hit":
        if obj_rank == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.34, 0.56), (0.42, 0.70), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "rest_target"
        else:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.10, 0.24), (0.18 + 0.12 * slot, 0.36 + 0.12 * slot), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            target_xy = center_xy_from_corner(ex, corner_cfg)
            events.append({
                "type": "set_motion",
                "frame_idx": int(np.random.randint(8, 22) + 3 * obj_rank),
                "linvel": make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.25, 2.00)), lift_z=float(np.random.uniform(0.05, 0.35)), tangent_jitter=0.03),
                "angvel": make_spin(scale=0.85),
            })
            camera_ref_pos = init_pos
            mode_name = "late_hit_actor"
    elif family == "line_chain_collision":
        x_line = np.linspace(0.24, 0.76, total_objects)
        x_frac = float(x_line[obj_rank])
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (x_frac - 0.03, x_frac + 0.03), (0.48, 0.60), floor_only=True, floor_extra=0.02)
        linvel = [0.0, 0.0, 0.0]
        angvel = [0.0, 0.0, 0.0]
        if obj_rank == 0:
            target_xy = [x_max - 0.08 * (x_max - x_min), center_xy[1]]
            target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
            linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.35, 1.95)), lift_z=0.0, tangent_jitter=0.00)
            mode_name = "chain_striker"
        else:
            mode_name = f"chain_target_{obj_rank}"
    elif family == "cross_fire":
        if obj_rank % 2 == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.10, 0.20), (0.22 + 0.12 * slot, 0.40 + 0.12 * slot), floor_only=True, floor_extra=0.02)
            target_xy = [x_max - 0.16 * (x_max - x_min), y_min + 0.70 * (y_max - y_min)]
        else:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.80, 0.92), (0.22 + 0.12 * slot, 0.40 + 0.12 * slot), floor_only=True, floor_extra=0.02)
            target_xy = [x_min + 0.16 * (x_max - x_min), y_min + 0.70 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.20, 1.95)), lift_z=float(np.random.uniform(0.00, 0.28)), tangent_jitter=0.02)
        angvel = make_spin(scale=0.65)
        mode_name = "cross_fire"
    elif family == "stack_drop":
        stack_x = x_min + (0.42 + 0.12 * np.random.uniform(-1.0, 1.0)) * (x_max - x_min)
        stack_y = y_min + (0.58 + 0.10 * np.random.uniform(-1.0, 1.0)) * (y_max - y_min)
        base_z = floor_spawn_z(ex, extra=0.02)
        init_pos = keep_position_inside_corner([stack_x, stack_y, base_z + obj_rank * (0.22 + 0.55 * ex[2])], ex, corner_cfg)
        linvel = [0.0, 0.0, 0.0]
        angvel = [0.0, 0.0, 0.0]
        mode_name = "stack_drop"
    elif family == "late_entry":
        if obj_rank < max(1, total_objects // 2):
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.34, 0.62), (0.44, 0.74), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "late_entry_static"
        else:
            side = random.choice(["left", "right", "front"])
            request_pos = edge_request_position(ex, corner_cfg, side=side, slot=0.16 + 0.20 * (obj_rank % 3), z_frac=float(np.random.uniform(0.02, 0.12)))
            register_runtime_entry(
                request_pos,
                center_xy,
                frame_idx=int(np.random.randint(0, EARLY_ENTRY_FRAME_MAX) + 6 * (obj_rank - max(1, total_objects // 2))),
                speed_xy_range=(1.20, 1.95),
                lift_range=(0.05, 0.55),
                tangent_jitter=0.05,
                ang_scale=0.75,
                park_lane="back" if side != "back" else "left",
            )
            mode_name = f"late_entry_actor_{side}"
    elif family == "staggered_rain":
        if obj_rank == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.30, 0.70), (0.42, 0.74), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "rain_target"
        else:
            request_pos = sample_fractional_corner_position(ex, corner_cfg, x_frac_range=(0.12, 0.88), y_frac_range=(0.34, 0.90), z_frac_range=(0.72, 1.08), floor_only=False, floor_extra=0.05)
            register_runtime_entry(
                request_pos,
                [request_pos[0] + np.random.uniform(-0.08, 0.08), request_pos[1] + np.random.uniform(-0.08, 0.08)],
                frame_idx=int((obj_rank - 1) * np.random.randint(4, 8)),
                speed_xy_range=(0.02, 0.26),
                lift_range=(-0.55, -0.08),
                tangent_jitter=0.00,
                ang_scale=0.55,
                park_lane="back",
            )
            # Override vertical rain velocity after helper.
            events[-1]["linvel"] = [float(np.random.uniform(-0.18, 0.18)), float(np.random.uniform(0.00, 0.18)), float(np.random.uniform(-0.55, -0.08))]
            mode_name = "staggered_rain_entry"
    elif family == "sequential_entry":
        if obj_rank == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.34, 0.60), (0.42, 0.70), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "anchor_static"
        else:
            side = ["left", "right", "front"][(obj_rank - 1) % 3]
            request_pos = edge_request_position(ex, corner_cfg, side=side, slot=0.12 + 0.18 * obj_rank, z_frac=float(np.random.uniform(0.01, 0.10)))
            target_xy = [center_xy[0] + alt * 0.12 * (x_max - x_min), center_xy[1] + 0.12 * (y_max - y_min)]
            register_runtime_entry(
                request_pos,
                target_xy,
                frame_idx=int((obj_rank - 1) * np.random.randint(6, 10)),
                speed_xy_range=(1.15, 1.90),
                lift_range=(0.05, 0.42),
                tangent_jitter=0.04,
                ang_scale=0.70,
                park_lane="back" if side != "back" else "right",
            )
            mode_name = f"sequential_entry_actor_{side}"
    elif family == "static_then_dual_hit":
        if obj_rank == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.38, 0.58), (0.46, 0.70), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "central_static"
        elif obj_rank == 1:
            request_pos = edge_request_position(ex, corner_cfg, side="right", slot=0.24, z_frac=float(np.random.uniform(0.02, 0.08)))
            register_runtime_entry(request_pos, center_xy, frame_idx=int(np.random.randint(0, 8)), speed_xy_range=(1.25, 1.95), lift_range=(0.04, 0.28), tangent_jitter=0.02, ang_scale=0.72, park_lane="right")
            mode_name = "dual_hit_right"
        else:
            request_pos = edge_request_position(ex, corner_cfg, side="left", slot=0.74, z_frac=float(np.random.uniform(0.02, 0.10)))
            register_runtime_entry(request_pos, center_xy, frame_idx=int(np.random.randint(10, 20)), speed_xy_range=(1.20, 1.85), lift_range=(0.03, 0.26), tangent_jitter=0.02, ang_scale=0.72, park_lane="left")
            mode_name = "dual_hit_left"
    elif family == "orbit_mix":
        sector = obj_rank % 4
        if sector == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.84, 0.96), (0.20, 0.36), floor_only=False, z_frac_range=(0.10, 0.35))
        elif sector == 1:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.70, 0.90), (0.76, 0.92), floor_only=False, z_frac_range=(0.08, 0.28))
        elif sector == 2:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.18, 0.34), (0.60, 0.88), floor_only=True, floor_extra=0.02)
        else:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.20, 0.36), (0.18, 0.34), floor_only=False, z_frac_range=(0.42, 0.88))
        target_xy = [center_xy[0] + alt * 0.15 * (x_max - x_min), center_xy[1] + 0.10 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(0.95, 1.80)), lift_z=float(np.random.uniform(0.00, 0.52)), tangent_jitter=0.12)
        angvel = make_spin(scale=0.95)
        mode_name = "orbit_mix"
    elif family == "mixed_multi":
        cycle = obj_rank % 5
        if cycle == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.12, 0.88), (0.28, 0.88), (0.58, 1.00), floor_only=False)
            angvel = make_spin(scale=0.45)
            mode_name = "mixed_drop"
        elif cycle == 1:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.08, 0.18), (0.24, 0.40), floor_only=True, floor_extra=0.02)
            target_xy = [x_max - 0.18 * (x_max - x_min), center_xy[1]]
            target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
            linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.10, 1.75)), lift_z=0.0, tangent_jitter=0.02)
            angvel = make_spin(scale=0.65, biased_axis=1)
            mode_name = "mixed_roll"
        elif cycle == 2:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.32, 0.58), (0.42, 0.68), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "mixed_rest"
        elif cycle == 3:
            side = random.choice(["left", "right", "front"])
            request_pos = edge_request_position(ex, corner_cfg, side=side, slot=0.18 + 0.14 * min(obj_rank, 4), z_frac=0.04)
            register_runtime_entry(request_pos, center_xy, frame_idx=int(np.random.randint(0, 18)), speed_xy_range=(1.20, 1.90), lift_range=(0.08, 0.42), tangent_jitter=0.04, ang_scale=0.68, park_lane="back")
            mode_name = f"mixed_entry_{side}"
        else:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.72, 0.96), (0.28, 0.84), (0.12, 0.44), floor_only=False)
            target_xy = [center_xy[0] - 0.10 * (x_max - x_min), center_xy[1] + alt * 0.14 * (y_max - y_min)]
            target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
            linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.25, 2.05)), lift_z=float(np.random.uniform(0.25, 0.95)), tangent_jitter=0.05)
            angvel = make_spin(scale=0.92)
            mode_name = "mixed_throw"
    elif family == "front_entry_arc":
        request_pos = edge_request_position(ex, corner_cfg, side="front", slot=0.20 + 0.18 * (obj_rank % 4), z_frac=float(np.random.uniform(0.02, 0.10)))
        target_xy = [center_xy[0] + alt * 0.10 * (x_max - x_min), y_min + 0.66 * (y_max - y_min)]
        if obj_rank < 2:
            frame_idx = int(obj_rank * np.random.randint(0, 4))
        else:
            frame_idx = int(4 + obj_rank * np.random.randint(4, 8))
        register_runtime_entry(request_pos, target_xy, frame_idx=frame_idx, speed_xy_range=(1.20, 2.00), lift_range=(0.25, 1.10), tangent_jitter=0.03, ang_scale=0.72, park_lane="back")
        mode_name = "front_entry_arc"
    elif family == "front_entry_slide":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.12, 0.88), (0.00, 0.06), floor_only=True, floor_extra=0.01)
        target_xy = [center_xy[0] + alt * 0.08 * (x_max - x_min), y_max - 0.10 * (y_max - y_min)]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.00, 1.85)), lift_z=0.0, tangent_jitter=0.02)
        angvel = make_spin(scale=0.48, biased_axis=2)
        mode_name = "front_entry_slide"
    elif family == "left_right_pingpong":
        if obj_rank == 0:
            init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.42, 0.58), (0.46, 0.72), floor_only=True, floor_extra=0.02)
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "pingpong_anchor"
        else:
            side = "left" if obj_rank % 2 == 1 else "right"
            request_pos = edge_request_position(ex, corner_cfg, side=side, slot=0.22 + 0.14 * min(obj_rank, 4), z_frac=float(np.random.uniform(0.01, 0.08)))
            target_xy = [center_xy[0], center_xy[1] + alt * 0.10 * (y_max - y_min)]
            register_runtime_entry(request_pos, target_xy, frame_idx=int((obj_rank - 1) * np.random.randint(4, 8)), speed_xy_range=(1.25, 2.05), lift_range=(0.02, 0.24), tangent_jitter=0.02, ang_scale=0.65, park_lane=side)
            mode_name = f"pingpong_{side}"
    elif family == "back_wall_rebound":
        init_pos = sample_non_overlapping_position(existing_objects, ex, corner_cfg, (0.18, 0.82), (0.02, 0.12), floor_only=False, z_frac_range=(0.06, 0.20))
        target_xy = [center_xy[0] + alt * 0.08 * (x_max - x_min), y_max]
        target_xy = clamp_xy_to_frame_safe(target_xy, ex, corner_cfg, margin_frac=0.14)
        linvel = make_toward_point_velocity(init_pos, target_xy, speed_xy=float(np.random.uniform(1.10, 1.85)), lift_z=float(np.random.uniform(0.02, 0.35)), tangent_jitter=0.01)
        angvel = make_spin(scale=0.62)
        mode_name = "back_wall_rebound"

    if not start_offstage:
        init_pos = keep_position_inside_corner(init_pos, ex, corner_cfg)
        linvel = make_velocity_wall_safe(init_pos, ex, linvel, corner_cfg, toward_wall_cap_x=-1.35, toward_wall_cap_y=-1.25)
    else:
        for ev in events:
            if ev.get("type") == "teleport_and_set_motion":
                ev["request_pos"][:2] = clamp_xy_to_frame_safe(ev["request_pos"][:2], ex, corner_cfg, margin_frac=0.10)
                ev["linvel"] = make_velocity_wall_safe(ev["request_pos"], ex, ev["linvel"], corner_cfg, toward_wall_cap_x=-1.35, toward_wall_cap_y=-1.25)
    linvel = clamp_vec_norm((np.asarray(linvel, dtype=np.float32) * LINEAR_SPEED_SCALE).tolist(), MAX_LINVEL_NORM)
    linvel = clamp_vec_norm(keep_linvel_in_frame(init_pos, ex, linvel, corner_cfg, horizon_sec=0.85, margin_frac=0.14), MAX_LINVEL_NORM)
    angvel = clamp_vec_norm((np.asarray(angvel, dtype=np.float32) * ANGULAR_SPEED_SCALE).tolist(), MAX_ANGVEL_NORM)
    for ev in events:
        if "linvel" in ev:
            scaled = clamp_vec_norm((np.asarray(ev["linvel"], dtype=np.float32) * LINEAR_SPEED_SCALE).tolist(), MAX_LINVEL_NORM)
            ref_pos = ev.get("request_pos", init_pos)
            ev["linvel"] = clamp_vec_norm(keep_linvel_in_frame(ref_pos, ex, scaled, corner_cfg, horizon_sec=0.85, margin_frac=0.14), MAX_LINVEL_NORM)
        if "angvel" in ev:
            ev["angvel"] = clamp_vec_norm((np.asarray(ev["angvel"], dtype=np.float32) * ANGULAR_SPEED_SCALE).tolist(), MAX_ANGVEL_NORM)
    return {
        "motion_type": mode_name,
        "init_pos": [float(x) for x in init_pos],
        "init_euler": init_euler,
        "init_linvel": linvel,
        "init_angvel": angvel,
        "script_events": events,
        "start_offstage": bool(start_offstage),
        "activation_frame": int(activation_frame),
        "camera_ref_pos": [float(x) for x in (camera_ref_pos if camera_ref_pos is not None else init_pos)],
    }

def build_physxnet_rigid_object_template(
    obj_idx: int,
    bank_item: Dict[str, Any],
    scene_colors: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Dict[str, Any]:
    dim_m = np.asarray(bank_item["dimension_m"], dtype=np.float32)
    if np.max(dim_m) <= 1e-6:
        dim_m = np.asarray([0.25, 0.25, 0.25], dtype=np.float32)
    max_dim = float(np.max(dim_m))
    target_max_dim = float(np.random.uniform(TARGET_MAX_DIM_MIN, TARGET_MAX_DIM_MAX))
    mesh_scale = target_max_dim / max(max_dim, 1e-8)
    proxy_bbox = np.asarray(bank_item.get("bbox_extents_m", bank_item["dimension_m"]), dtype=np.float32)
    scaled_dim = np.maximum(proxy_bbox * mesh_scale, MIN_VALID_PROXY_EXTENT)
    runtime_prior = bank_item["runtime_material_prior"]
    render_color = scene_colors[obj_idx] if scene_colors is not None else pick_distinct_colors(obj_idx + 1)[-1]
    return {
        "scene_object_id": obj_idx,
        "solver": "Rigid",
        "source_type": "physxnet_urdf",
        "asset_mode": "urdf_single_rigid",
        "motion_type": "pending",
        "object_id": bank_item["object_id"],
        "object_name": bank_item["object_name"],
        "category": bank_item["category"],
        "proxy_mode": bank_item["proxy_mode"],
        "proxy_stats": bank_item.get("proxy_stats", {}),
        "urdf_stats": bank_item.get("urdf_stats", {}),
        "n_parts": bank_item.get("n_parts", 1),
        "material_names": bank_item.get("material_names", []),
        "complexity_score": float(bank_item.get("complexity_score", 0.0)),
        "geom": {
            "shape": "urdf",
            "urdf_file": bank_item["urdf_path"],
            "render_mesh_file": bank_item["render_mesh_path"],
            "collision_mesh_file": bank_item["collision_mesh_path"],
            "scale": float(mesh_scale),
            "bbox_extents": scaled_dim.astype(float).tolist(),
            "render_face_count": float(bank_item.get("render_face_count", 0.0)),
            "render_vertex_count": float(bank_item.get("render_vertex_count", 0.0)),
        },
        "material": {
            "family": "Rigid",
            "rho": runtime_prior["rho"],
            "friction": runtime_prior["friction"],
            "restitution": runtime_prior["restitution"],
            "linear_damping": runtime_prior["linear_damping"],
            "bucket": runtime_prior["bucket"],
        },
        "render_color": render_color,
        "placement_margin": float(max(0.10, 0.34 * float(np.max(scaled_dim)) + 0.02 * np.random.uniform(0.0, 1.0))),
        "init_pos": [0.0, 0.0, 0.0],
        "init_euler": [0.0, 0.0, 0.0],
        "init_linvel": [0.0, 0.0, 0.0],
        "init_angvel": [0.0, 0.0, 0.0],
        "script_events": [],
    }


def apply_motion_to_object_template(obj: Dict[str, Any], motion: Dict[str, Any]) -> Dict[str, Any]:
    obj["motion_type"] = motion["motion_type"]
    obj["init_pos"] = motion["init_pos"]
    obj["init_euler"] = motion["init_euler"]
    obj["init_linvel"] = motion["init_linvel"]
    obj["init_angvel"] = motion["init_angvel"]
    obj["script_events"] = motion.get("script_events", [])
    obj["start_offstage"] = bool(motion.get("start_offstage", False))
    obj["activation_frame"] = int(motion.get("activation_frame", 0))
    obj["camera_ref_pos"] = motion.get("camera_ref_pos", motion["init_pos"])
    return obj


def sample_physxnet_rigid_object(
    obj_idx: int,
    bank_item: Dict[str, Any],
    family: str,
    total_objects: int,
    existing_objects: Optional[List[Dict[str, Any]]] = None,
    scene_colors: Optional[List[Tuple[float, float, float, float]]] = None,
    corner_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    existing_objects = existing_objects or []
    obj = build_physxnet_rigid_object_template(obj_idx, bank_item, scene_colors=scene_colors)
    motion = sample_motion_for_family(
        family,
        obj_rank=obj_idx,
        total_objects=total_objects,
        bbox_extents=obj["geom"]["bbox_extents"],
        existing_objects=existing_objects,
        corner_cfg=corner_cfg,
    )
    return apply_motion_to_object_template(obj, motion)


def sample_scene_cfg(scene_id: int, split: str, object_bank: List[Dict[str, Any]]) -> Dict[str, Any]:
    seed = 100000 + scene_id
    set_seed(seed)
    family = weighted_choice(SCENE_FAMILY_WEIGHTS)
    bg = sample_background()
    if family in ["line_chain_collision", "cross_fire", "stack_drop", "mixed_multi", "orbit_mix", "left_right_pingpong"]:
        n_obj = random.randint(max(3, MIN_OBJECTS_PER_SCENE), min(MAX_OBJECTS_PER_SCENE, 6))
    elif family in ["rest_then_hit", "late_entry", "sequential_entry", "static_then_dual_hit", "staggered_rain", "front_entry_arc"]:
        n_obj = random.randint(2, min(MAX_OBJECTS_PER_SCENE, 5))
    else:
        n_obj = random.randint(MIN_OBJECTS_PER_SCENE, min(MAX_OBJECTS_PER_SCENE, 5))
    n_obj = min(n_obj, len(object_bank))

    best_objects = None
    for _ in range(MAX_SCENE_SAMPLING_RETRIES):
        chosen = sample_complex_objects_for_scene(object_bank, n_obj)
        scene_colors = pick_distinct_colors(n_obj)
        proto_objects = [build_physxnet_rigid_object_template(i, bank_item, scene_colors=scene_colors) for i, bank_item in enumerate(chosen)]
        corner_cfg = build_corner_cfg_from_objects(proto_objects, CORNER_BASE)
        objects: List[Dict[str, Any]] = []
        for i, base_obj in enumerate(proto_objects):
            motion = sample_motion_for_family(
                family,
                obj_rank=i,
                total_objects=n_obj,
                bbox_extents=base_obj["geom"]["bbox_extents"],
                existing_objects=objects,
                corner_cfg=corner_cfg,
            )
            obj = apply_motion_to_object_template(base_obj, motion)
            objects.append(obj)
        refine_scene_layout(objects, corner_cfg)
        if scene_volume_ratio_ok(objects, MAX_VOLUME_RATIO_IN_SCENE):
            best_objects = objects
            break
        if best_objects is None:
            best_objects = objects
    objects = best_objects
    corner_cfg = build_corner_cfg_from_objects(objects, CORNER_BASE)
    refine_scene_layout(objects, corner_cfg)
    camera = sample_camera_from_objects(objects, corner_cfg)
    return {
        "scene_id": f"{split}_scene_{scene_id:06d}",
        "split": split,
        "seed": seed,
        "family": family,
        "background": bg,
        "corner": corner_cfg,
        "camera": camera,
        "sim_options": {
            "gravity": [0.0, 0.0, -9.81],
            "dt": SIM_DT,
            "substeps": SIM_SUBSTEPS,
            "physics_fps": PHYSICS_FPS,
            "physics_steps_per_frame": PHYSICS_STEPS_PER_FRAME,
            "num_steps": SIM_NUM_STEPS,
            "num_frames": N_OUTPUT_FRAMES,
            "fps": FPS,
        },
        "objects": objects,
        "volume_ratio_limit": MAX_VOLUME_RATIO_IN_SCENE,
        "actual_volume_ratio": compute_scene_volume_ratio(objects),
        "proxy_mode": PROXY_MODE,
        "asset_loader_mode": "PhysX-3D -> URDF -> Genesis",
        "object_complexity": {
            "mean_score": float(np.mean([o.get("complexity_score", 0.0) for o in chosen])) if len(chosen) > 0 else 0.0,
            "max_score": float(np.max([o.get("complexity_score", 0.0) for o in chosen])) if len(chosen) > 0 else 0.0,
            "min_score": float(np.min([o.get("complexity_score", 0.0) for o in chosen])) if len(chosen) > 0 else 0.0,
        },
    }


# =========================================================
# Genesis 兼容辅助
# =========================================================
def _try_call_methods(obj: Any, method_names: Sequence[str], value: Any) -> bool:
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


def _try_get_methods(obj: Any, method_names: Sequence[str]) -> Any:
    for name in method_names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)()
            except Exception:
                pass
    return None


def query_entity_state(ent: Any) -> Dict[str, Optional[List[float]]]:
    def _to_list(x: Any) -> Optional[List[float]]:
        if x is None:
            return None
        return to_numpy_host(x).astype(float).reshape(-1).tolist()
    return {
        "position": _to_list(_try_get_methods(ent, ["get_pos", "get_position"])),
        "euler": _to_list(_try_get_methods(ent, ["get_euler", "get_rotation_euler"])),
        "linear_velocity": _to_list(_try_get_methods(ent, ["get_vel", "get_velocity", "get_linear_velocity"])),
        "angular_velocity": _to_list(_try_get_methods(ent, ["get_angvel", "get_angular_velocity"])),
    }


def apply_initial_motion_to_rigid_entity(ent: Any, obj_meta: Dict[str, Any]) -> None:
    v = np.asarray(obj_meta.get("init_linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
    w = np.asarray(obj_meta.get("init_angvel", [0.0, 0.0, 0.0]), dtype=np.float32)
    if np.linalg.norm(v) > 0:
        _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], v)
    if np.linalg.norm(w) > 0:
        _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], w)


def find_safe_runtime_spawn_position(
    request_pos: Sequence[float],
    bbox_extents: Sequence[float],
    live_states: List[Dict[str, Any]],
    corner_cfg: Optional[Dict[str, Any]] = None,
    margin: float = 0.18,
) -> List[float]:
    req = np.asarray(request_pos, dtype=np.float32)
    ex = np.asarray(bbox_extents, dtype=np.float32)
    req_xy = keep_inside_corner_xy(req[:2], ex.tolist(), corner_cfg)
    req = np.asarray([req_xy[0], req_xy[1], max(float(req[2]), floor_spawn_z(ex.tolist(), extra=0.04))], dtype=np.float32)
    offsets = [
        np.asarray([0.0, 0.0, 0.18], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.32], dtype=np.float32),
        np.asarray([0.0, 0.18, 0.18], dtype=np.float32),
        np.asarray([0.0, -0.18, 0.18], dtype=np.float32),
        np.asarray([0.18, 0.0, 0.18], dtype=np.float32),
        np.asarray([-0.18, 0.0, 0.18], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.48], dtype=np.float32),
        np.asarray([0.24, 0.20, 0.24], dtype=np.float32),
        np.asarray([-0.24, -0.20, 0.24], dtype=np.float32),
        np.asarray([MAX_RUNTIME_PUSH_FROM_WALL, 0.0, 0.26], dtype=np.float32),
        np.asarray([0.0, MAX_RUNTIME_PUSH_FROM_WALL, 0.26], dtype=np.float32),
    ]
    for off in offsets:
        pos = req + off
        pos = np.asarray(keep_position_inside_corner(pos.tolist(), ex.tolist(), corner_cfg), dtype=np.float32)
        ok = True
        for st in live_states:
            p = st.get("position")
            e = st.get("bbox_extents")
            if p is None or e is None:
                continue
            if aabb_overlap_3d(pos.tolist(), ex.tolist(), p, e, margin=margin):
                ok = False
                break
        if ok:
            return pos.astype(float).tolist()
    fallback = req + np.asarray([MAX_RUNTIME_PUSH_FROM_WALL, MAX_RUNTIME_PUSH_FROM_WALL, 0.45], dtype=np.float32)
    fallback = np.asarray(keep_position_inside_corner(fallback.tolist(), ex.tolist(), corner_cfg), dtype=np.float32)
    return fallback.astype(float).tolist()


def apply_script_events(frame_idx: int, scene_cfg: Dict[str, Any], entities: List[Any]) -> List[Dict[str, Any]]:
    live_states = []
    for obj, ent in zip(scene_cfg["objects"], entities):
        st = query_entity_state(ent)
        live_states.append({"position": st["position"], "bbox_extents": obj["geom"]["bbox_extents"]})
    applied = []
    for obj, ent in zip(scene_cfg["objects"], entities):
        for ev in obj.get("script_events", []):
            if int(ev.get("frame_idx", -1)) != frame_idx:
                continue
            if ev["type"] == "set_motion":
                _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], np.asarray(ev["linvel"], dtype=np.float32))
                _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], np.asarray(ev["angvel"], dtype=np.float32))
                applied.append({"type": "set_motion", "frame_idx": frame_idx, "scene_object_id": obj["scene_object_id"]})
            elif ev["type"] == "teleport_and_set_motion":
                safe_pos = find_safe_runtime_spawn_position(ev["request_pos"], obj["geom"]["bbox_extents"], live_states, corner_cfg=scene_cfg.get("corner"))
                _try_call_methods(ent, ["set_pos", "set_position"], np.asarray(safe_pos, dtype=np.float32))
                _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], np.asarray(ev["linvel"], dtype=np.float32))
                _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], np.asarray(ev["angvel"], dtype=np.float32))
                obj.setdefault("runtime_spawn_log", []).append({"frame_idx": frame_idx, "requested": ev["request_pos"], "applied": safe_pos})
                applied.append({
                    "type": "teleport_and_set_motion",
                    "frame_idx": frame_idx,
                    "scene_object_id": obj["scene_object_id"],
                    "applied_pos": safe_pos,
                })
    return applied


# def add_large_corner(scene: Any, corner_cfg: Dict[str, Any]) -> None:
#     cx, cy, cz = _corner_center(corner_cfg)
#     big = _corner_panel_size(corner_cfg)
#     wall_thickness = _corner_thickness(corner_cfg)
#     visible_thick = max(MIN_VISIBLE_WALL_THICKNESS, wall_thickness * VISIBLE_WALL_THICKNESS_SCALE)
#     guard_thick = max(MIN_GUARD_WALL_THICKNESS, wall_thickness * GUARD_WALL_THICKNESS_SCALE)
#     floor_thick = max(visible_thick, 0.08)
#     wall_mat = gs.materials.Rigid(rho=1200.0, friction=0.95)
#     floor_size = (big, big, floor_thick)
#     scene.add_entity(
#         morph=gs.morphs.Box(size=floor_size, pos=(cx + big * 0.5, cy + big * 0.5, cz - floor_thick * 0.5), fixed=True),
#         material=wall_mat,
#         surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["floor"]),
#     )

#     # U 形三面墙：左 / 右 / 后，前方保持开口，画面更像摄影棚或测试台。
#     scene.add_entity(
#         morph=gs.morphs.Box(size=(visible_thick, big, big), pos=(cx - visible_thick * 0.5, cy + big * 0.5, cz + big * 0.5), fixed=True),
#         material=wall_mat,
#         surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_left"]),
#     )
#     scene.add_entity(
#         morph=gs.morphs.Box(size=(visible_thick, big, big), pos=(cx + big + visible_thick * 0.5, cy + big * 0.5, cz + big * 0.5), fixed=True),
#         material=wall_mat,
#         surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_right"]),
#     )
#     scene.add_entity(
#         morph=gs.morphs.Box(size=(big, visible_thick, big), pos=(cx + big * 0.5, cy + big + visible_thick * 0.5, cz + big * 0.5), fixed=True),
#         material=wall_mat,
#         surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_back"]),
#     )

#     # guard walls：放在可见墙外侧，减少高速穿墙，但不进入主要画面。
#     scene.add_entity(
#         morph=gs.morphs.Box(size=(guard_thick, big, big), pos=(cx - visible_thick - guard_thick * 0.5, cy + big * 0.5, cz + big * 0.5), fixed=True),
#         material=wall_mat,
#         surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_left"]),
#     )
#     scene.add_entity(
#         morph=gs.morphs.Box(size=(guard_thick, big, big), pos=(cx + big + visible_thick + guard_thick * 0.5, cy + big * 0.5, cz + big * 0.5), fixed=True),
#         material=wall_mat,
#         surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_right"]),
#     )
#     scene.add_entity(
#         morph=gs.morphs.Box(size=(big, guard_thick, big), pos=(cx + big * 0.5, cy + big + visible_thick + guard_thick * 0.5, cz + big * 0.5), fixed=True),
#         material=wall_mat,
#         surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_back"]),
#     )


def add_large_corner(scene: Any, corner_cfg: Dict[str, Any]) -> None:
    cx, cy, cz = _corner_center(corner_cfg)
    big = _corner_panel_size(corner_cfg)
    wall_thickness = _corner_thickness(corner_cfg)

    visible_thick = max(MIN_VISIBLE_WALL_THICKNESS, wall_thickness * VISIBLE_WALL_THICKNESS_SCALE)
    guard_thick = max(MIN_GUARD_WALL_THICKNESS, wall_thickness * GUARD_WALL_THICKNESS_SCALE)
    floor_thick = max(visible_thick, 0.08)

    wall_mat = gs.materials.Rigid(rho=1200.0, friction=0.95)

    # floor
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(big, big, floor_thick),
            pos=(cx + big * 0.5, cy + big * 0.5, cz - floor_thick * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["floor"]),
    )

    # =========================
    # 可见三面墙（纯色底层）
    # =========================
    # 左墙
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(visible_thick, big, big),
            pos=(cx - visible_thick * 0.5, cy + big * 0.5, cz + big * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_left"]),
    )

    # 右墙
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(visible_thick, big, big),
            pos=(cx + big + visible_thick * 0.5, cy + big * 0.5, cz + big * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_right"]),
    )

    # 后墙
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(big, visible_thick, big),
            pos=(cx + big * 0.5, cy + big + visible_thick * 0.5, cz + big * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_back"]),
    )

    # =========================
    # 条纹装饰层：只放在“可见内侧”
    # =========================
    if ENABLE_STRIPED_WALLS:
        # 左墙内侧面：x = cx
        _add_striped_wall_overlay(
            scene=scene,
            wall_mat=wall_mat,
            plane="x",
            face_pos=cx,
            span_a0=cy,
            span_a1=cy + big,
            z0=cz,
            z1=cz + big,
            base_color=CONTAINER_FACE_COLORS["wall_left"],
            inward_sign=+1.0,
        )

        # 右墙内侧面：x = cx + big
        _add_striped_wall_overlay(
            scene=scene,
            wall_mat=wall_mat,
            plane="x",
            face_pos=cx + big,
            span_a0=cy,
            span_a1=cy + big,
            z0=cz,
            z1=cz + big,
            base_color=CONTAINER_FACE_COLORS["wall_right"],
            inward_sign=-1.0,
        )

        # 后墙内侧面：y = cy + big
        _add_striped_wall_overlay(
            scene=scene,
            wall_mat=wall_mat,
            plane="y",
            face_pos=cy + big,
            span_a0=cx,
            span_a1=cx + big,
            z0=cz,
            z1=cz + big,
            base_color=CONTAINER_FACE_COLORS["wall_back"],
            inward_sign=-1.0,
        )

    # =========================
    # guard walls：放在可见墙外侧，减少高速穿墙
    # =========================
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(guard_thick, big, big),
            pos=(cx - visible_thick - guard_thick * 0.5, cy + big * 0.5, cz + big * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_left"]),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(guard_thick, big, big),
            pos=(cx + big + visible_thick + guard_thick * 0.5, cy + big * 0.5, cz + big * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_right"]),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(big, guard_thick, big),
            pos=(cx + big * 0.5, cy + big + visible_thick + guard_thick * 0.5, cz + big * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_back"]),
    )

def sync_visual_entities(collision_entities: List[Any], visual_entities: List[Any]) -> None:
    if not USE_COMPLEX_VISUAL_MESH:
        return
    for cent, vent in zip(collision_entities, visual_entities):
        if vent is None:
            continue
        st = query_entity_state(cent)
        pos = st.get("position")
        euler = st.get("euler")
        if pos is not None:
            _try_call_methods(vent, ["set_pos", "set_position"], np.asarray(pos, dtype=np.float32))
        if euler is not None:
            _try_call_methods(vent, ["set_euler", "set_rotation_euler"], np.asarray(euler, dtype=np.float32))


def build_scene(scene_cfg: Dict[str, Any]) -> Tuple[Any, Any, List[Any], List[Optional[Any]]]:
    ensure_genesis_initialized()
    vis_options = gs.options.VisOptions(
        show_world_frame=False,
        show_link_frame=False,
        background_color=tuple(scene_cfg["background"]["background_color"]),
        ambient_light=tuple(scene_cfg["background"]["ambient_light"]),
        segmentation_level="entity",
    )
    sim_options = gs.options.SimOptions(
        gravity=tuple(scene_cfg["sim_options"]["gravity"]),
        dt=scene_cfg["sim_options"]["dt"],
        substeps=scene_cfg["sim_options"]["substeps"],
    )
    scene_kwargs = dict(sim_options=sim_options, vis_options=vis_options, show_viewer=False)
    try:
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(
            dt=scene_cfg["sim_options"]["dt"],
            enable_collision=True,
            use_gjk_collision=True,
        )
    except Exception:
        pass
    scene = gs.Scene(**scene_kwargs)
    add_large_corner(scene, scene_cfg["corner"])

    collision_entities: List[Any] = []
    visual_entities: List[Optional[Any]] = []

    for obj in scene_cfg["objects"]:
        mat_kwargs = {"rho": obj["material"]["rho"], "friction": obj["material"]["friction"]}
        try:
            if obj["material"].get("restitution") is not None:
                mat_kwargs["restitution"] = obj["material"]["restitution"]
        except Exception:
            pass
        try:
            if obj["material"].get("linear_damping") is not None:
                mat_kwargs["linear_damping"] = obj["material"]["linear_damping"]
        except Exception:
            pass
        try:
            mat = gs.materials.Rigid(**mat_kwargs)
        except Exception:
            fallback_kwargs = {k: mat_kwargs[k] for k in ["rho", "friction"] if k in mat_kwargs}
            mat = gs.materials.Rigid(**fallback_kwargs)

        surface = gs.surfaces.Default(color=obj["render_color"])

        urdf_kwargs = dict(
            file=obj["geom"]["urdf_file"],
            scale=obj["geom"]["scale"],
            pos=tuple(obj["init_pos"]),
            euler=tuple(obj["init_euler"]),
            visualization=True,
            collision=True,
            fixed=False,
            merge_fixed_links=URDF_MERGE_FIXED_LINKS,
            prioritize_urdf_material=URDF_PRIORITIZE_URDF_MATERIAL,
            recompute_inertia=URDF_RECOMPUTE_INERTIA,
            decimate=COLLISION_PROXY_DECIMATE,
            convexify=COLLISION_PROXY_CONVEXIFY,
        )
        try:
            collision_entity = scene.add_entity(
                morph=gs.morphs.URDF(**urdf_kwargs),
                material=mat,
                surface=surface,
            )
        except TypeError:
            # 老版本 Genesis 可能不支持较新的 URDF 参数。
            fallback_keys = ["file", "scale", "pos", "euler", "visualization", "collision", "fixed"]
            collision_entity = scene.add_entity(
                morph=gs.morphs.URDF(**{k: urdf_kwargs[k] for k in fallback_keys}),
                material=mat,
                surface=surface,
            )
        collision_entities.append(collision_entity)
        visual_entities.append(None)

    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )
    scene.build()
    for _ in range(WARMUP_STEPS):
        scene.step()
    for obj, ent in zip(scene_cfg["objects"], collision_entities):
        apply_initial_motion_to_rigid_entity(ent, obj)
    sync_visual_entities(collision_entities, visual_entities)
    return scene, cam, collision_entities, visual_entities



def build_scene_from_cfg(scene_cfg: Dict[str, Any], camera_override: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any, List[Any], List[Optional[Any]]]:
    cfg = clone_scene_cfg_with_camera(scene_cfg, camera_override)
    return build_scene(cfg)


def render_preview_image_with_camera(scene_cfg: Dict[str, Any], camera_override: Optional[Dict[str, Any]] = None) -> np.ndarray:
    scene = cam = None
    collision_entities = visual_entities = None
    try:
        scene, cam, collision_entities, visual_entities = build_scene_from_cfg(scene_cfg, camera_override=camera_override)
        render_out = cam.render()
        rgb, _, _ = parse_render_output(render_out)
        if rgb is None:
            raise RuntimeError("camera render returned empty rgb")
        return rgb
    finally:
        safe_scene_destroy(scene)

def parse_render_output(render_out: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    rgb, depth, seg = None, None, None
    if isinstance(render_out, tuple):
        if len(render_out) >= 1:
            rgb = render_out[0]
        if len(render_out) >= 2:
            depth = render_out[1]
        if len(render_out) >= 3:
            seg = render_out[2]
    else:
        rgb = render_out
    rgb = None if rgb is None else to_uint8_image(rgb)
    depth = None if depth is None else to_numpy_host(depth)
    seg = None if seg is None else to_numpy_host(seg)
    return rgb, depth, seg


def extract_events(frame_states: Dict[int, List[Dict[str, Any]]], scene_cfg: Dict[str, Any], runtime_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    floor_z = 0.0
    for obj_idx, obj in enumerate(scene_cfg["objects"]):
        hz = obj["geom"]["bbox_extents"][2] / 2.0
        grounded = False
        staticed = False
        peak_speed = 0.0
        active_from = int(obj.get("activation_frame", 0))
        for t in sorted(frame_states.keys()):
            if t < active_from:
                continue
            st = frame_states[t][obj_idx]
            pos = st["position"]
            vel = st["linear_velocity"]
            if pos is not None and not grounded and pos[2] - hz <= floor_z + 0.02:
                events.append({"event_type": "first_ground_contact", "frame_idx": t, "scene_object_id": obj["scene_object_id"]})
                grounded = True
            if vel is not None:
                vnorm = float(np.linalg.norm(np.asarray(vel, dtype=np.float32)))
                peak_speed = max(peak_speed, vnorm)
                if not staticed and vnorm < 0.05 and t > 10:
                    events.append({"event_type": "near_static", "frame_idx": t, "scene_object_id": obj["scene_object_id"]})
                    staticed = True
        events.append({"event_type": "peak_speed", "scene_object_id": obj["scene_object_id"], "value": peak_speed})
        if obj.get("runtime_spawn_log"):
            for r in obj["runtime_spawn_log"]:
                events.append({
                    "event_type": "runtime_spawn_adjusted",
                    "frame_idx": r["frame_idx"],
                    "scene_object_id": obj["scene_object_id"],
                    "requested": r["requested"],
                    "applied": r["applied"],
                })

    seen_collision = set()
    for t in sorted(frame_states.keys()):
        poses = [x["position"] for x in frame_states[t]]
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                if poses[i] is None or poses[j] is None:
                    continue
                pi = np.asarray(poses[i], dtype=np.float32)
                pj = np.asarray(poses[j], dtype=np.float32)
                di = np.asarray(scene_cfg["objects"][i]["geom"]["bbox_extents"], dtype=np.float32)
                dj = np.asarray(scene_cfg["objects"][j]["geom"]["bbox_extents"], dtype=np.float32)
                thresh = 0.40 * float(np.linalg.norm(di) + np.linalg.norm(dj))
                if np.linalg.norm(pi - pj) < thresh:
                    key = (i, j)
                    if key not in seen_collision:
                        seen_collision.add(key)
                        events.append({
                            "event_type": "approx_object_collision",
                            "frame_idx": t,
                            "obj_a": scene_cfg["objects"][i]["scene_object_id"],
                            "obj_b": scene_cfg["objects"][j]["scene_object_id"],
                        })
    events.extend(runtime_events)
    return events


# =========================================================
# 导出：更规范的输出格式
# =========================================================
def write_frame_csv(frame_states: Dict[int, List[Dict[str, Any]]], out_csv_path: Path) -> None:
    ensure_dir(out_csv_path.parent)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "scene_object_id",
            "px", "py", "pz",
            "ex", "ey", "ez",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
        ])
        for t in sorted(frame_states.keys()):
            for row in frame_states[t]:
                pos = row.get("position") or [None, None, None]
                eul = row.get("euler") or [None, None, None]
                vel = row.get("linear_velocity") or [None, None, None]
                ang = row.get("angular_velocity") or [None, None, None]
                writer.writerow([t, row["scene_object_id"], *pos[:3], *eul[:3], *vel[:3], *ang[:3]])


def export_scene(scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = OUT_ROOT / scene_cfg["split"] / scene_cfg["scene_id"]
    ensure_dir(out_dir / "rgb")
    ensure_dir(out_dir / "depth")
    ensure_dir(out_dir / "seg")
    ensure_dir(out_dir / "video")
    ensure_dir(out_dir / "ann")

    with open(out_dir / "ann" / "scene_input.json", "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, ensure_ascii=False, indent=2)
    dump_camera_cfg(scene_cfg["camera"], out_dir / "ann" / "camera_selected.json")

    prompt = (
        f"A physics simulation scene with {len(scene_cfg['objects'])} rigid objects. "
        f"Scene family: {scene_cfg['family']}. Collision uses proxy mode {scene_cfg['proxy_mode']}. "
        f"Same-scene volume ratio is limited to <= {scene_cfg['volume_ratio_limit']}, actual ratio {scene_cfg['actual_volume_ratio']:.3f}."
    )
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    scene = cam = collision_entities = visual_entities = None
    preview_frames: List[np.ndarray] = []
    frame_states: Dict[int, List[Dict[str, Any]]] = {}
    runtime_events: List[Dict[str, Any]] = []
    try:
        scene, cam, collision_entities, visual_entities = build_scene(scene_cfg)
        num_frames = int(scene_cfg["sim_options"]["num_frames"])
        physics_steps_per_frame = int(scene_cfg["sim_options"].get("physics_steps_per_frame", 1))
        for t in range(num_frames):
            runtime_events.extend(apply_script_events(t, scene_cfg, collision_entities))
            for _ in range(physics_steps_per_frame):
                scene.step()
            sync_visual_entities(collision_entities, visual_entities)
            try:
                render_out = cam.render(rgb=True, depth=True, segmentation=True)
            except Exception:
                render_out = cam.render(rgb=True)
            rgb, depth, seg = parse_render_output(render_out)
            if rgb is None:
                raise RuntimeError(f"render returned rgb=None at frame {t}")
            imageio.imwrite(out_dir / "rgb" / f"{t:06d}.png", rgb)
            if depth is not None:
                np.save(out_dir / "depth" / f"{t:06d}.npy", depth)
            if seg is not None:
                np.save(out_dir / "seg" / f"{t:06d}.npy", seg)
            if t % PREVIEW_FRAME_STRIDE == 0:
                preview_frames.append(rgb)

            per_obj_states = []
            for obj, ent in zip(scene_cfg["objects"], collision_entities):
                st = query_entity_state(ent)
                per_obj_states.append({
                    "scene_object_id": obj["scene_object_id"],
                    "position": st["position"],
                    "euler": st["euler"],
                    "linear_velocity": st["linear_velocity"],
                    "angular_velocity": st["angular_velocity"],
                })
            frame_states[t] = per_obj_states

        preview_path = save_preview_video(preview_frames, out_dir / "video" / "preview.mp4", fps=FPS)

        objects_json = []
        for obj in scene_cfg["objects"]:
            objects_json.append({
                "scene_object_id": obj["scene_object_id"],
                "object_id": obj["object_id"],
                "object_name": obj["object_name"],
                "category": obj["category"],
                "asset_mode": obj.get("asset_mode", "urdf_single_rigid"),
                "urdf_file": obj["geom"].get("urdf_file"),
                "render_mesh_file": obj["geom"]["render_mesh_file"],
                "collision_mesh_file": obj["geom"]["collision_mesh_file"],
                "proxy_mode": obj["proxy_mode"],
                "proxy_stats": obj.get("proxy_stats", {}),
                "urdf_stats": obj.get("urdf_stats", {}),
                "scale": obj["geom"]["scale"],
                "bbox_extents": obj["geom"]["bbox_extents"],
                "render_face_count": obj["geom"].get("render_face_count"),
                "render_vertex_count": obj["geom"].get("render_vertex_count"),
                "render_color": obj["render_color"],
                "complexity_score": obj.get("complexity_score", 0.0),
                "motion_type": obj["motion_type"],
                "material_bucket": obj["material"]["bucket"],
                "rho": obj["material"]["rho"],
                "friction": obj["material"]["friction"],
                "restitution": obj["material"].get("restitution"),
                "linear_damping": obj["material"].get("linear_damping"),
                "material_names": obj.get("material_names", []),
                "init_pos": obj["init_pos"],
                "init_euler": obj["init_euler"],
                "init_linvel": obj["init_linvel"],
                "init_angvel": obj["init_angvel"],
                "script_events": obj.get("script_events", []),
                "start_offstage": obj.get("start_offstage", False),
                "activation_frame": obj.get("activation_frame", 0),
                "camera_ref_pos": obj.get("camera_ref_pos"),
            })
        with open(out_dir / "ann" / "objects.json", "w", encoding="utf-8") as f:
            json.dump(objects_json, f, ensure_ascii=False, indent=2)

        with open(out_dir / "ann" / "frames.jsonl", "w", encoding="utf-8") as f:
            for t in range(num_frames):
                row = {"frame_idx": t, "timestamp_sec": t / float(scene_cfg["sim_options"]["fps"]), "objects": frame_states[t]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_frame_csv(frame_states, out_dir / "ann" / "trajectories.csv")

        events = extract_events(frame_states, scene_cfg, runtime_events)
        with open(out_dir / "ann" / "events.json", "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        scene_metadata = {
            "scene_id": scene_cfg["scene_id"],
            "split": scene_cfg["split"],
            "seed": scene_cfg["seed"],
            "family": scene_cfg["family"],
            "proxy_mode": scene_cfg["proxy_mode"],
            "volume_ratio_limit": scene_cfg["volume_ratio_limit"],
            "actual_volume_ratio": scene_cfg["actual_volume_ratio"],
            "camera": scene_cfg["camera"],
            "corner": scene_cfg["corner"],
            "background": scene_cfg["background"],
            "sim_options": scene_cfg["sim_options"],
            "num_objects": len(scene_cfg["objects"]),
            "num_frames": num_frames,
            "preview_file": preview_path.name if preview_path is not None else None,
            "asset_loader_mode": "PhysX-3D -> URDF -> Genesis",
            "container_face_colors": CONTAINER_FACE_COLORS,
            "use_complex_visual_mesh": USE_COMPLEX_VISUAL_MESH,
            "object_complexity": scene_cfg.get("object_complexity", {}),
            "status": "ok",
        }
        with open(out_dir / "ann" / "scene_metadata.json", "w", encoding="utf-8") as f:
            json.dump(scene_metadata, f, ensure_ascii=False, indent=2)
        return scene_metadata
    finally:
        safe_scene_destroy(scene)


# =========================================================
# 主程序
# =========================================================
def main(out_dir: Path) -> None:
    ensure_dir(out_dir)
    ensure_dir(MERGED_CACHE_DIR)
    ensure_dir(PROXY_CACHE_DIR)
    ensure_dir(URDF_CACHE_DIR)
    for sp in ["train", "val", "test"]:
        ensure_dir(OUT_ROOT / sp)

    loader = PhysXNetGenesisLoader(
        root=str(PHYSXNET_ROOT),
        version=PHYSXNET_VERSION,
        merged_cache_dir=str(MERGED_CACHE_DIR),
        proxy_cache_dir=str(PROXY_CACHE_DIR),
        urdf_cache_dir=str(URDF_CACHE_DIR),
        proxy_mode=PROXY_MODE,
    )
    object_bank, failed = build_physxnet_object_bank(loader=loader, max_objects_to_read=MAX_DATASET_OBJECTS_TO_READ)
    if len(object_bank) == 0:
        raise RuntimeError("No usable PhysXNet objects loaded.")
    split_bank = split_object_bank_by_id(object_bank)

    with open(OUT_ROOT / "object_bank_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "physxnet_root": str(PHYSXNET_ROOT),
            "version": PHYSXNET_VERSION,
            "proxy_mode": PROXY_MODE,
            "asset_loader_mode": "PhysX-3D -> URDF -> Genesis",
            "urdf_cache_dir": str(URDF_CACHE_DIR),
            "max_dataset_objects_to_read": MAX_DATASET_OBJECTS_TO_READ,
            "n_usable_objects": len(object_bank),
            "n_failed_objects": len(failed),
            "split_stats": {k: len(v) for k, v in split_bank.items()},
            "complexity_summary": {
                "top10_object_ids": [x["object_id"] for x in object_bank[:10]],
                "top10_scores": [float(x.get("complexity_score", 0.0)) for x in object_bank[:10]],
            },
            "failed": failed,
        }, f, ensure_ascii=False, indent=2)

    backend_used = "none"
    if gs is not None:
        backend_used = ensure_genesis_initialized()

    manifest = {
        "dataset_name": "physxnet_urdf_dataset_v1",
        "proxy_mode": PROXY_MODE,
        "n_scenes_requested": N_SCENES,
        "image_size": [IMG_W, IMG_H],
        "backend_used": backend_used,
        "sim_dt": SIM_DT,
        "sim_substeps": SIM_SUBSTEPS,
        "max_volume_ratio_in_scene": MAX_VOLUME_RATIO_IN_SCENE,
        "controls": {
            "min_objects_per_scene": MIN_OBJECTS_PER_SCENE,
            "max_objects_per_scene": MAX_OBJECTS_PER_SCENE,
            "camera_distance_mult_min": CAMERA_DISTANCE_MULT_MIN,
            "camera_distance_mult_max": CAMERA_DISTANCE_MULT_MAX,
            "target_max_dim_min": TARGET_MAX_DIM_MIN,
            "target_max_dim_max": TARGET_MAX_DIM_MAX,
            "prefer_complex_objects": PREFER_COMPLEX_OBJECTS,
            "use_complex_visual_mesh": USE_COMPLEX_VISUAL_MESH,
        },
        "scenes": {"train": [], "val": [], "test": []},
        "failed_scenes": [],
    }

    scene_rows = []
    try:
        split_order = ["train", "val", "test"]
        split_weights = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
        split_weights = split_weights / split_weights.sum()
        for sid in range(N_SCENES):
            split = str(np.random.choice(split_order, p=split_weights))
            bank = split_bank[split] if len(split_bank[split]) >= MIN_OBJECTS_PER_SCENE else split_bank["train"]
            scene_cfg = sample_scene_cfg(sid, split=split, object_bank=bank)
            try:
                print(
                    f"[RUN ] {scene_cfg['scene_id']} | family={scene_cfg['family']} | "
                    f"n_obj={len(scene_cfg['objects'])} | vol_ratio={scene_cfg['actual_volume_ratio']:.3f} | proxy={scene_cfg['proxy_mode']}"
                )
                meta = export_scene(scene_cfg)
                manifest["scenes"][split].append(meta)
                scene_rows.append([
                    meta["scene_id"],
                    meta["split"],
                    meta["family"],
                    meta["num_objects"],
                    meta["num_frames"],
                    f"{meta['actual_volume_ratio']:.6f}",
                    meta["preview_file"],
                ])
                print(f"[ OK ] {scene_cfg['scene_id']} | preview={meta['preview_file']}")
            except Exception as e:
                err = {"scene_id": scene_cfg["scene_id"], "split": split, "family": scene_cfg["family"], "error": str(e)}
                manifest["failed_scenes"].append(err)
                print(f"[FAIL] {scene_cfg['scene_id']} | err={e}")
                if STOP_ON_ERROR:
                    raise
    finally:
        with open(OUT_ROOT / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(OUT_ROOT / "scene_index.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scene_id", "split", "family", "num_objects", "num_frames", "volume_ratio", "preview_file"])
            writer.writerows(scene_rows)
        if gs is not None:
            try:
                gs.destroy()
            except Exception:
                pass


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="/data/gaoya/AAA_test_video/Dataset_test/physxnet_proxy_dataset_v23333")
    parser.add_argument("--use_manual_camera", action="store_true", help="Directly override auto camera with MANUAL_CAMERA or --manual_camera_json")
    parser.add_argument("--manual_camera_json", type=str, default="", help="Path to a camera json produced by camera_tuner_gradio.py")
    args = parser.parse_args()

    OUT_ROOT = Path(args.out_root)
    if args.use_manual_camera:
        USE_MANUAL_CAMERA = True
    if args.manual_camera_json:
        MANUAL_CAMERA_JSON_PATH = args.manual_camera_json
        USE_MANUAL_CAMERA = True

    main(OUT_ROOT)
