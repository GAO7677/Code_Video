import json
import csv
import math
import random
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import genesis as gs


# =========================
# 基本配置
# =========================
import trimesh

ASSET_DATA_ROOTS = [
    Path("/data/gaoya/dataset/SOPHY_data/bag/simulation_data/train/bag"),
    # Path("/data/gaoya/dataset/another_dataset/train"),
    # Path("/data/gaoya/dataset/xxx"),
]

USE_DATASET_MESH_OBJECTS = True
DATASET_OBJECT_PROB = 0.7   # 刚开始建议 0.5~0.8
MAX_ASSETS_PER_ROOT = None  # 可限制扫描数量，调试时很有用

ASSET_CACHE_DIR = DATASET_ROOT / "_asset_cache"
TARGET_MESH_SIZE_RANGE = (0.08, 0.20)   # 物体最长边目标尺寸范围（米）



DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_test/genesis_sim_v3")
IMG_W, IMG_H = 640, 480
N_SCENES = 100

MAX_OBJECT_PC = 2048
OBJECT_PC_STRIDE = 5
CAMERA_PC_STRIDE = 2

ENABLE_CLOTH = False
CLOTH_MESH_PATH = None
STOP_ON_ERROR = False

SCENE_FAMILY_WEIGHTS = {
    "rigid_mix": 0.45,
    "mpm_mix": 0.30,
    "sph_liquid": 0.25,
}

# 容器：开口朝 -y，相机放在前方（负 y）看进去
CONTAINER = {
    "half_x": 0.50,            # 容器半宽（x 方向）
    "half_y": 0.62,            # 容器半深（y 方向）
    "wall_thickness": 0.035,
    "wall_height": 0.56,       # 左右/后墙高度
    "front_lip_height": 0.09,  # 前挡板高度（低，保证不遮挡）
    "floor_thickness": 0.04,
    "center": [0.0, 0.0, 0.0],
}


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
        {"name": "white", "background_color": [1.0, 1.0, 1.0], "ambient_light": [0.36, 0.36, 0.36]},
        {"name": "light_gray", "background_color": [0.92, 0.92, 0.92], "ambient_light": [0.32, 0.32, 0.32]},
        {"name": "sky_blue", "background_color": [0.80, 0.88, 1.00], "ambient_light": [0.30, 0.31, 0.35]},
        {"name": "warm", "background_color": [0.98, 0.95, 0.90], "ambient_light": [0.34, 0.32, 0.30]},
        {"name": "dark", "background_color": [0.10, 0.10, 0.12], "ambient_light": [0.20, 0.20, 0.20]},
    ]
    return random.choice(presets)


def sample_color(alpha=1.0):
    return [
        float(np.random.uniform(0.08, 0.95)),
        float(np.random.uniform(0.08, 0.95)),
        float(np.random.uniform(0.08, 0.95)),
        float(alpha),
    ]


# =========================
# 容器与相机
# =========================
def sample_camera(container_cfg: dict):
    """
    相机放在容器开口外侧（负 y），略高俯视，保证不会被前墙遮挡。
    """
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wh = container_cfg["wall_height"]
    cx, cy, cz = container_cfg["center"]

    return {
        "res": [IMG_W, IMG_H],
        "pos": [
            float(cx + np.random.uniform(-0.10, 0.10)),             # x
            float(cy - hy - 1.05 + np.random.uniform(-0.10, 0.08)), # y：在容器前方外侧
            float(cz + wh + 0.45 + np.random.uniform(-0.06, 0.12)), # z：高于挡板很多
        ],
        "lookat": [
            float(cx + np.random.uniform(-0.05, 0.05)),
            float(cy + np.random.uniform(-0.02, 0.12)),
            float(cz + 0.24 + np.random.uniform(-0.02, 0.12)),
        ],
        "fov": float(np.random.uniform(32, 40)),
        "GUI": False,
    }


def add_container(scene, container_cfg: dict):
    """
    容器结构：
    - floor
    - left wall
    - right wall
    - back wall
    - front low lip（低挡板）
    开口朝 -y，相机从前方看进去。
    """
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wt = container_cfg["wall_thickness"]
    wh = container_cfg["wall_height"]
    lip_h = container_cfg["front_lip_height"]
    ft = container_cfg["floor_thickness"]
    cx, cy, cz = container_cfg["center"]

    wall_mat = gs.materials.Rigid(rho=1200.0, friction=0.95)

    floor_surface = gs.surfaces.Default(color=(0.72, 0.72, 0.74, 1.0))
    wall_surface = gs.surfaces.Default(color=(0.76, 0.80, 0.86, 1.0))
    lip_surface = gs.surfaces.Default(color=(0.70, 0.76, 0.82, 1.0))

    container_entities = {}

    # floor
    container_entities["floor"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2 * hx, 2 * hy, ft),
            pos=(cx, cy, cz + ft / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=floor_surface,
    )

    # left wall (x negative)
    container_entities["left_wall"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(wt, 2 * hy, wh),
            pos=(cx - hx + wt / 2.0, cy, cz + ft + wh / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=wall_surface,
    )

    # right wall (x positive)
    container_entities["right_wall"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(wt, 2 * hy, wh),
            pos=(cx + hx - wt / 2.0, cy, cz + ft + wh / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=wall_surface,
    )

    # back wall (y positive)
    container_entities["back_wall"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2 * hx, wt, wh),
            pos=(cx, cy + hy - wt / 2.0, cz + ft + wh / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=wall_surface,
    )

    # front low lip (y negative) —— 低挡板，不挡视线
    container_entities["front_lip"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(2 * hx, wt, lip_h),
            pos=(cx, cy - hy + wt / 2.0, cz + ft + lip_h / 2.0),
            fixed=True,
        ),
        material=wall_mat,
        surface=lip_surface,
    )

    return container_entities


def sample_spawn_xy(margin_x: float, margin_y: float, bias_to_back=False):
    """
    在容器内部采样 x/y，确保不贴墙。
    bias_to_back=True 时，物体更偏向容器中后部，减少从前方挡板掉出去的概率。
    """
    hx = CONTAINER["half_x"]
    hy = CONTAINER["half_y"]
    wt = CONTAINER["wall_thickness"]

    x_min = -hx + wt + margin_x
    x_max = +hx - wt - margin_x

    # y 方向前方是开口 + 低挡板，所以出生点尽量别贴太前
    if bias_to_back:
        y_min = -0.05
        y_max = +hy - wt - margin_y
    else:
        y_min = -hy + wt + 0.14 + margin_y
        y_max = +hy - wt - margin_y

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
def sample_rigid_object(obj_id: int, pattern="drop_cluster"):
    geom, half_x, half_y, half_z = rigid_geom_and_margins()
    material = sample_rigid_material()

    ground_margin = 0.02

    if pattern == "drop_cluster":
        x, y = sample_spawn_xy(half_x + 0.02, half_y + 0.02, bias_to_back=False)
        z = float(np.random.uniform(0.70, 1.20))
        z = max(z, CONTAINER["floor_thickness"] + half_z + ground_margin + 0.20)

    elif pattern == "opposed_lanes":
        lane_x = np.random.uniform(0.16, 0.26)
        x = float(-lane_x if (obj_id % 2 == 0) else lane_x)
        y = float(np.random.uniform(-0.05, 0.22))
        z = float(np.random.uniform(0.65, 1.10))
        z = max(z, CONTAINER["floor_thickness"] + half_z + ground_margin + 0.20)

    else:
        x, y = sample_spawn_xy(half_x + 0.02, half_y + 0.02, bias_to_back=False)
        z = float(np.random.uniform(0.70, 1.20))
        z = max(z, CONTAINER["floor_thickness"] + half_z + ground_margin + 0.20)

    return {
        "object_id": obj_id,
        "solver": "Rigid",
        "geom": geom,
        "material": material,
        "init_pos": [float(x), float(y), float(z)],
        "init_euler": [
            float(np.random.uniform(-0.06, 0.06)),
            float(np.random.uniform(-0.06, 0.06)),
            float(np.random.uniform(-math.pi, math.pi)),
        ],
        "color": sample_color(),
        "surface_vis_mode": "visual",
    }


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
        "geom": {
            "shape": "box",
            "size": size,
        },
        "material": material,
        "init_pos": [float(x), float(y), float(z)],
        "color": sample_color(),
        "surface_vis_mode": "particle",
    }


def sample_scene_cfg(scene_id: int):
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
        objects = [sample_rigid_object(i, pattern=pattern) for i in range(n_obj)]
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
        "scene_id": f"scene_{scene_id:06d}",
        "seed": seed,
        "family": family,
        "background": bg,
        "container": CONTAINER,
        "camera": cam,
        "sim_options": sim_options,
        "objects": objects,
    }


# =========================
# 场景构建
# =========================
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

    # 给软体/流体足够包围盒
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

    # 兼容不同 Genesis 版本
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

    # 容器
    container_entities = add_container(scene, scene_cfg["container"])

    entities = []
    entity_roles = []

    for obj in scene_cfg["objects"]:
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

            if obj["geom"]["shape"] == "box":
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
            ent = scene.add_entity(
                morph=gs.morphs.Box(size=tuple(obj["geom"]["size"]), pos=pos),
                material=mat,
                surface=surface,
            )

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
        entity_roles.append("object")

    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )

    scene.build()

    # 给 MPM 加一点轻微初速度
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


# =========================
# 导出
# =========================
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

    # 粒子类优先
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

    # mesh 顶点
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
            material_summary.append({
                "object_id": obj["object_id"],
                "solver": obj["solver"],
                "material": obj["material"],
            })

        scene_metadata = {
            "scene_id": scene_cfg["scene_id"],
            "seed": scene_cfg["seed"],
            "family": scene_cfg["family"],
            "num_objects": len(scene_cfg["objects"]),
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
    ensure_dir(DATASET_ROOT / "train")
    ensure_dir(DATASET_ROOT / "failed_configs")

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
        "notes": [
            "Uses z-up convention.",
            "Container has back/side walls and a low front lip, so the camera is not occluded.",
            "Each scene contains multiple objects with independently sampled material/physics parameters.",
            "Single-scene failure is recorded instead of aborting the whole run."
        ],
        "scenes": [],
        "failed_scenes": [],
    }

    try:
        for sid in range(N_SCENES):
            scene_cfg = sample_scene_cfg(sid)

            try:
                print(f"[RUN ] {scene_cfg['scene_id']} | family={scene_cfg['family']}")
                meta = export_scene(scene_cfg)
                manifest["scenes"].append(meta)
                print(f"[ OK ] {scene_cfg['scene_id']} | family={scene_cfg['family']}")

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