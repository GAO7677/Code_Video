import json
import csv
import math
import random
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import genesis as gs
import trimesh


# =========================
# 基本配置
# =========================
DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_test/genesis_sim_sophy_0322444")
IMG_W, IMG_H = 640, 480
N_SCENES = 5

MAX_OBJECT_PC = 2048
OBJECT_PC_STRIDE = 5
CAMERA_PC_STRIDE = 2

ENABLE_CLOTH = False
CLOTH_MESH_PATH = None
STOP_ON_ERROR = False

SCENE_FAMILY_WEIGHTS = {
    "rigid_mix": 0.4,
    "mpm_mix": 0.3,
    "sph_liquid": 0.3,
}

# =========================
# 数据集 asset 配置
# =========================
SOURCE_DATASET_ROOTS = [
    Path("/data/gaoya/dataset/SOPHY_data/bag"),
    Path("/data/gaoya/dataset/SOPHY_data/teddy_bear"),
]

# USE_DATASET_MESH_OBJECTS = True
# DATASET_OBJECT_PROB = 0.8          # rigid_mix 场景里，一个物体采样为 dataset mesh 的概率
MAX_ASSETS_PER_ROOT = 5          # 调试时可设成小整数
ASSET_CACHE_DIR = DATASET_ROOT / "_asset_cache"
ASSET_MANIFEST_PATH = DATASET_ROOT / "asset_manifest.json"

TARGET_MESH_SIZE_RANGE = (0.2, 0.5)   # 最长边目标尺寸（米）
SIMPLIFY_MESH_FACE_COUNT = 3000         # None 表示不减面；建议 2000~5000
MIN_VALID_MESH_EXTENT = 1e-5

# 容器：开口朝 -y，相机放在前方（负 y）看进去
# 调整目标：
# 1) 容器尽量放大，提升“物体落在容器内部”的概率
# 2) 改成真正三面体：地面 + 左右墙 + 后墙，前方完全开口
# 3) 配合更保守的出生区和初速度，减少物体从前方飞出
# 增大容器尺寸，提升“物体落在容器内部”的概率
CONTAINER = {

    "half_x": 1.5,# 宽
    "half_y": 1.5,# 深
    "wall_thickness": 0.04,
    "wall_height": 2,# 高
    "front_lip_height": 0.00,
    "floor_thickness": 0.05,
    "center": [0.0, 0.0, 0.0],
}

# 出生区域安全边距：
# 前开口方向（-y）预留更大 buffer，尽量把物体出生点压到容器中后部
SPAWN_FRONT_KEEP_OUT = 0.42
SPAWN_BACK_KEEP_OUT = 0.10
SPAWN_SIDE_KEEP_OUT = 0.06

# =========================
# 数据集 mesh 的朝向修正
# =========================
# 说明：
# teddy_bear 很可能是 y-up 的模型，在 z-up 世界里会平躺。
# 这里先用绕 x 轴 -90° 进行扶正。
# 如果你发现扶正后仍然不对，把 teddy_bear 改成 [0.0, -math.pi / 2.0, 0.0] 试一下。
DATASET_BASE_EULER_BY_DATASET = {
    "teddy_bear":  [0.0, -math.pi / 2.0, 0.0] ,
    "bag": [0.0, 0.0, 0.0],
}

# =========================
# rigid 运动模式
# 目标：优先保证物体尽量落在容器内部，因此：
# - 上方下落 / 上方轻抛占绝大多数
# - 左右侧抛显著减少且速度收敛
# =========================
RIGID_MOTION_WEIGHTS = {
    "top_drop": 0.70,
    "top_toss": 0.24,
    "side_throw_left": 0.03,
    "side_throw_right": 0.03,
}

TOP_DROP_Z_RANGE = (1.05, 1.45)
TOP_TOSS_Z_RANGE = (1.00, 1.38)
SIDE_THROW_Z_RANGE = (0.62, 0.88)

TOP_DROP_VXY = 0.06
TOP_TOSS_VX = 0.30
TOP_TOSS_VY = 0.16
TOP_TOSS_VZ_RANGE = (-0.90, -0.25)    # 向下

SIDE_THROW_VX_RANGE = (0.95, 1.40)    # 朝容器中心，但更保守
SIDE_THROW_VY_RANGE = (0.08, 0.26)    # 统一偏向 +y，把物体往容器后部送
SIDE_THROW_VZ_RANGE = (0.45, 0.90)    # 减少过高抛射

TOP_DROP_ANGVEL = 1.5
TOP_TOSS_ANGVEL = 3.0
SIDE_THROW_ANGVEL = 3.5

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





def get_dataset_base_euler(dataset_name: str):
    arr = DATASET_BASE_EULER_BY_DATASET.get(dataset_name, [0.0, 0.0, 0.0])
    return np.asarray(arr, dtype=np.float32)


def sample_rigid_motion(half_x: float, half_y: float, half_z: float):
    """
    返回 rigid 物体的初始位置、初始姿态增量、线速度、角速度、运动类型。
    设计目标：
    - 绝大多数物体从容器上方落下或轻微抛下
    - 少量侧抛保留多样性，但速度明显减小
    - 出生点整体偏向容器中后部，尽量减少从前方开口飞出的概率
    """
    mode = weighted_choice(RIGID_MOTION_WEIGHTS)

    hx = CONTAINER["half_x"]
    hy = CONTAINER["half_y"]
    wh = CONTAINER["wall_height"]

    init_pos = [0.0, 0.0, 1.0]
    linvel = [0.0, 0.0, 0.0]
    angvel = [0.0, 0.0, 0.0]
    pose_delta = [0.0, 0.0, 0.0]

    if mode == "top_drop":
        x, y = sample_spawn_xy(half_x + 0.02, half_y + 0.02, bias_to_back=True)
        z = float(np.random.uniform(*TOP_DROP_Z_RANGE))
        z = max(z, CONTAINER["floor_thickness"] + half_z + 0.34)

        init_pos = [float(x), float(y), float(z)]

        linvel = [
            float(np.random.uniform(-TOP_DROP_VXY, TOP_DROP_VXY)),
            float(np.random.uniform(-TOP_DROP_VXY, TOP_DROP_VXY)),
            float(np.random.uniform(-0.18, -0.03)),
        ]

        angvel = [
            float(np.random.uniform(-TOP_DROP_ANGVEL, TOP_DROP_ANGVEL)),
            float(np.random.uniform(-TOP_DROP_ANGVEL, TOP_DROP_ANGVEL)),
            float(np.random.uniform(-1.0, 1.0)),
        ]

        pose_delta = [
            float(np.random.uniform(-0.06, 0.06)),
            float(np.random.uniform(-0.06, 0.06)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]

    elif mode == "top_toss":
        x, y = sample_spawn_xy(half_x + 0.02, half_y + 0.02, bias_to_back=True)
        z = float(np.random.uniform(*TOP_TOSS_Z_RANGE))
        z = max(z, CONTAINER["floor_thickness"] + half_z + 0.34)

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

        pose_delta = [
            float(np.random.uniform(-0.14, 0.14)),
            float(np.random.uniform(-0.14, 0.14)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]

    elif mode == "side_throw_left":
        x = float(-hx - half_x - np.random.uniform(0.05, 0.10))
        y = float(np.random.uniform(0.06, 0.30))
        z = float(np.random.uniform(*SIDE_THROW_Z_RANGE))
        z = max(z, wh + 0.04)

        init_pos = [x, y, z]

        linvel = [
            float(np.random.uniform(*SIDE_THROW_VX_RANGE)),
            float(np.random.uniform(*SIDE_THROW_VY_RANGE)),
            float(np.random.uniform(*SIDE_THROW_VZ_RANGE)),
        ]

        angvel = [
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
        ]

        pose_delta = [
            float(np.random.uniform(-0.22, 0.22)),
            float(np.random.uniform(-0.22, 0.22)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]

    elif mode == "side_throw_right":
        x = float(+hx + half_x + np.random.uniform(0.05, 0.10))
        y = float(np.random.uniform(0.06, 0.30))
        z = float(np.random.uniform(*SIDE_THROW_Z_RANGE))
        z = max(z, wh + 0.04)

        init_pos = [x, y, z]

        linvel = [
            float(-np.random.uniform(*SIDE_THROW_VX_RANGE)),
            float(np.random.uniform(*SIDE_THROW_VY_RANGE)),
            float(np.random.uniform(*SIDE_THROW_VZ_RANGE)),
        ]

        angvel = [
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
            float(np.random.uniform(-SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)),
        ]

        pose_delta = [
            float(np.random.uniform(-0.22, 0.22)),
            float(np.random.uniform(-0.22, 0.22)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]

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

    asset_id = asset["asset_id"]
    cache_obj = ASSET_CACHE_DIR / f"{asset_id}_unit.obj"
    cache_json = ASSET_CACHE_DIR / f"{asset_id}_unit_meta.json"

    if cache_obj.exists() and cache_json.exists():
        with open(cache_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        asset["unit_mesh_path"] = str(cache_obj)
        asset["render_mesh_path"] = meta.get("render_mesh_path", asset["mesh_path"])
        asset["raw_bbox_extents"] = meta["raw_bbox_extents"]
        asset["unit_bbox_extents"] = meta["unit_bbox_extents"]
        asset["n_vertices"] = meta.get("n_vertices", None)
        asset["n_faces"] = meta.get("n_faces", None)
        asset["has_texture"] = bool(meta.get("has_texture", False))
        return asset

    mesh = load_trimesh_any(Path(asset["mesh_path"]))
    mesh, raw_extents, unit_extents, unit_scale = sanitize_mesh(mesh)

    # 继续导出 unit.obj，仅用于几何尺度统计 / debug
    mesh.export(cache_obj)

    sample_dir = Path(asset["sample_dir"])
    has_texture = (sample_dir / "material.mtl").exists() and len(list(sample_dir.glob("material_*.png"))) > 0

    meta = {
        "asset_id": asset_id,
        "mesh_path": asset["mesh_path"],
        "render_mesh_path": asset["mesh_path"],   # 真正渲染时优先用原始 material.obj
        "raw_bbox_extents": raw_extents.tolist(),
        "unit_bbox_extents": unit_extents.tolist(),
        "unit_scale_from_raw": float(unit_scale),
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "has_texture": bool(has_texture),
    }
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    asset["unit_mesh_path"] = str(cache_obj)
    asset["render_mesh_path"] = asset["mesh_path"]
    asset["raw_bbox_extents"] = raw_extents.tolist()
    asset["unit_bbox_extents"] = unit_extents.tolist()
    asset["n_vertices"] = meta["n_vertices"]
    asset["n_faces"] = meta["n_faces"]
    asset["has_texture"] = bool(has_texture)
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
        }
    except Exception:
        return default_mat

def build_asset_bank():
    raw_assets = find_asset_dirs_from_roots(SOURCE_DATASET_ROOTS)
    print(f"[INFO] total raw assets found: {len(raw_assets)}")

    bank = []
    failed = []

    for a in raw_assets:
        try:
            bank.append(prepare_asset_cache(a))
        except Exception as e:
            failed.append({
                "sample_dir": a.get("sample_dir"),
                "mesh_path": a.get("mesh_path"),
                "error": str(e),
            })
            print(f"[WARN] skip asset: {a.get('sample_dir')} | err={e}")

    manifest = {
        "n_raw_assets": len(raw_assets),
        "n_usable_assets": len(bank),
        "n_failed_assets": len(failed),
        "usable_assets": bank,
        "failed_assets": failed,
    }
    with open(ASSET_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

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
            float(cx + np.random.uniform(-0.08, 0.08)),
            float(cy - hy - 1.35 + np.random.uniform(-0.10, 0.08)),
            float(cz + wh + 0.30 + np.random.uniform(-0.05, 0.10)),
        ],
        "lookat": [
            float(cx + np.random.uniform(-0.04, 0.04)),
            float(cy + np.random.uniform(0.04, 0.20)),
            float(cz + 0.22 + np.random.uniform(-0.02, 0.10)),
        ],
        "fov": float(np.random.uniform(34, 42)),
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


# =========================
# 材质采样
# =========================
def sample_rigid_material():
    libs = {
        "light_plastic": {"rho": (300, 800), "friction": (0.20, 0.80)},
        "wood_like": {"rho": (500, 1000), "friction": (0.30, 0.90)},
        "metal_like": {"rho": (1500, 3000), "friction": (0.18, 0.55)},
        "rubber_like": {"rho": (900, 1300), "friction": (0.85, 1.40)},
    }
    name = random.choice(list(libs.keys()))
    conf = libs[name]
    rho = float(np.random.uniform(*conf["rho"]))
    friction = float(np.clip(np.random.uniform(*conf["friction"]), 1e-2, 5.0))
    return {
        "family": "Rigid",
        "name": name,
        "rho": rho,
        "friction": friction,
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
def rigid_geom_and_margins():
    shape = random.choice(["box", "sphere", "cylinder"])

    if shape == "box":
        size = [
            float(np.random.uniform(0.05, 0.18)),
            float(np.random.uniform(0.05, 0.18)),
            float(np.random.uniform(0.05, 0.18)),
        ]
        geom = {"shape": "box", "size": size}
        half_x = size[0] / 2.0
        half_y = size[1] / 2.0
        half_z = size[2] / 2.0

    elif shape == "sphere":
        radius = float(np.random.uniform(0.04, 0.10))
        geom = {"shape": "sphere", "radius": radius}
        half_x = radius
        half_y = radius
        half_z = radius

    else:
        radius = float(np.random.uniform(0.03, 0.08))
        height = float(np.random.uniform(0.06, 0.18))
        geom = {"shape": "cylinder", "radius": radius, "height": height}
        half_x = radius
        half_y = radius
        half_z = height / 2.0

    return geom, half_x, half_y, half_z


# =========================
# 物体采样
# =========================
def sample_dataset_rigid_object(obj_id: int, asset_bank, pattern="drop_cluster"):
    asset = random.choice(asset_bank)

    unit_ext = np.array(asset["unit_bbox_extents"], dtype=np.float32)
    target_size = float(np.random.uniform(*TARGET_MESH_SIZE_RANGE))
    mesh_scale = target_size / max(float(np.max(unit_ext)), 1e-8)

    scaled_ext = unit_ext * mesh_scale
    half_x = float(scaled_ext[0] / 2.0)
    half_y = float(scaled_ext[1] / 2.0)
    half_z = float(scaled_ext[2] / 2.0)

    motion = sample_rigid_motion(half_x, half_y, half_z)

    base_euler = get_dataset_base_euler(asset["dataset_name"])
    final_euler = (base_euler + np.asarray(motion["pose_delta"], dtype=np.float32)).tolist()

    render_mesh_file = asset["render_mesh_path"] if USE_TEXTURED_DATASET_MESH else asset["unit_mesh_path"]

    return {
        "object_id": obj_id,
        "solver": "Rigid",
        "source_type": "dataset_mesh",
        "motion_type": motion["motion_type"],
        "geom": {
            "shape": "mesh",
            "mesh_file": render_mesh_file,
            "scale": float(mesh_scale),
            "bbox_extents": scaled_ext.tolist(),
            "asset_id": asset["asset_id"],
            "dataset_name": asset["dataset_name"],
            "sample_dir": asset["sample_dir"],
            "n_vertices": asset.get("n_vertices", None),
            "n_faces": asset.get("n_faces", None),
            "use_texture": bool(asset.get("has_texture", False) and USE_TEXTURED_DATASET_MESH),
        },
        "material": load_asset_material_or_default(asset),
        "init_pos": [float(x) for x in motion["init_pos"]],
        "init_euler": [float(x) for x in final_euler],
        "init_linvel": [float(x) for x in motion["init_linvel"]],
        "init_angvel": [float(x) for x in motion["init_angvel"]],
        "color": None,   # 贴图物体不再需要随机色
        "surface_vis_mode": "visual",
    }

def sample_procedural_rigid_object(obj_id: int, pattern="drop_cluster"):
    geom, half_x, half_y, half_z = rigid_geom_and_margins()
    material = sample_rigid_material()

    motion = sample_rigid_motion(half_x, half_y, half_z)

    # 程序化物体不需要专门扶正，直接用 motion 给出的姿态扰动
    final_euler = [
        float(motion["pose_delta"][0]),
        float(motion["pose_delta"][1]),
        float(motion["pose_delta"][2]),
    ]

    return {
        "object_id": obj_id,
        "solver": "Rigid",
        "source_type": "procedural",
        "motion_type": motion["motion_type"],
        "geom": geom,
        "material": material,
        "init_pos": [float(x) for x in motion["init_pos"]],
        "init_euler": final_euler,
        "init_linvel": [float(x) for x in motion["init_linvel"]],
        "init_angvel": [float(x) for x in motion["init_angvel"]],
        "color": sample_color(),
        "surface_vis_mode": "visual",
    }

def sample_rigid_object(obj_id: int, pattern="drop_cluster", asset_bank=None):
    if (
        USE_DATASET_MESH_OBJECTS
        and asset_bank is not None
        and len(asset_bank) > 0
        and (np.random.rand() < DATASET_OBJECT_PROB)
    ):
        return sample_dataset_rigid_object(obj_id, asset_bank, pattern=pattern)

    return sample_procedural_rigid_object(obj_id, pattern=pattern)


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

    family = weighted_choice(SCENE_FAMILY_WEIGHTS)
    if family == "cloth_drop" and not (ENABLE_CLOTH and CLOTH_MESH_PATH and Path(CLOTH_MESH_PATH).exists()):
        family = "rigid_mix"

    bg = sample_background()
    cam = sample_camera(CONTAINER)

    if family == "rigid_mix":
        pattern = random.choice(["drop_cluster", "opposed_lanes"])
        n_obj = random.randint(4, 7)
        objects = [sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank) for i in range(n_obj)]
        sim_options = {
            "gravity": [0.0, 0.0, -9.81],
            "dt": 4e-3,
            "substeps": 8,
            "num_steps": 180,
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

        surface = None
        if not is_textured_dataset_mesh:
            surface = gs.surfaces.Default(
                color=tuple(obj["color"]),
                vis_mode=obj.get("surface_vis_mode", "visual"),
            )

        if obj["solver"] == "Rigid":
            mat = gs.materials.Rigid(
                rho=obj["material"]["rho"],
                friction=obj["material"]["friction"],
            )
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
    traj_csv, frame_csv = None, None

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
                    np.savez_compressed(
                        out_dir / "object_pointcloud" / f"{t:06d}_obj{obj_meta['object_id']:02d}.npz",
                        xyz=xyz,
                        solver=obj_meta["solver"],
                        object_id=obj_meta["object_id"],
                    )

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
            }
            if obj["solver"] == "Rigid" and obj["geom"]["shape"] == "mesh":
                record["asset"] = {
                    "asset_id": obj["geom"].get("asset_id"),
                    "dataset_name": obj["geom"].get("dataset_name"),
                    "sample_dir": obj["geom"].get("sample_dir"),
                    "mesh_file": obj["geom"].get("mesh_file"),
                    "scale": obj["geom"].get("scale"),
                    "bbox_extents": obj["geom"].get("bbox_extents"),
                    "n_vertices": obj["geom"].get("n_vertices"),
                    "n_faces": obj["geom"].get("n_faces"),
                }
            else:
                record["geom"] = obj["geom"]
            material_summary.append(record)

        scene_metadata = {
            "scene_id": scene_cfg["scene_id"],
            "seed": scene_cfg["seed"],
            "family": scene_cfg["family"],
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
        safe_scene_destroy(scene)


# =========================
# 主程序
# =========================
def main():
    ensure_dir(DATASET_ROOT)
    ensure_dir(DATASET_ROOT / "train")
    ensure_dir(DATASET_ROOT / "failed_configs")
    ensure_dir(ASSET_CACHE_DIR)

    asset_bank = build_asset_bank()

    backend_used = "cpu"
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
        "source_dataset_roots": [str(x) for x in SOURCE_DATASET_ROOTS],
        "use_dataset_mesh_objects": USE_DATASET_MESH_OBJECTS,
        "dataset_object_prob": DATASET_OBJECT_PROB,
        "n_usable_dataset_assets": len(asset_bank),
        "notes": [
            "Uses z-up convention.",
            "Container has back/side walls and a low front lip, so the camera is not occluded.",
            "Rigid scenes can mix dataset mesh objects and procedural primitives.",
            "MPM/SPH scenes remain procedural for stability.",
            "Single-scene failure is recorded instead of aborting the whole run."
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
