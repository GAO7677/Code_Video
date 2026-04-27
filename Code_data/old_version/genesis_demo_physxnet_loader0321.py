import json
import math
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import genesis as gs

from Code_Video.Code_data.old_version.physxnet_genesis_loader import PhysXNetGenesisLoader


# =========================
# 基本配置
# =========================
PHYSXNET_ROOT = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
PHYSXNET_VERSION = "version_1"

OUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_test/physxnet_genesis_colored_corner0321")
IMG_W, IMG_H = 960, 720
N_SCENES = 3

MAX_DATASET_OBJECTS_TO_READ = 500
MIN_OBJECTS_PER_SCENE = 2
MAX_OBJECTS_PER_SCENE = 4

MERGED_CACHE_DIR = PHYSXNET_ROOT / PHYSXNET_VERSION / "_merged_for_genesis"
EXPORT_MERGED_WHEN_LOADING = True
STOP_ON_ERROR = False

CORNER_BASE = {
    "center": [0.0, 0.0, 0.0],
}

RIGID_MOTION_WEIGHTS = {
    "top_drop": 0.58,
    "top_toss": 0.24,
    "side_throw_x": 0.09,
    "side_throw_y": 0.09,
}


# =========================
# 配色
# =========================
OBJECT_COLOR_PALETTE = [
    (0.90, 0.35, 0.35, 1.0),
    (0.35, 0.75, 0.40, 1.0),
    (0.30, 0.50, 0.90, 1.0),
    (0.90, 0.75, 0.30, 1.0),
    (0.75, 0.35, 0.85, 1.0),
    (0.25, 0.78, 0.82, 1.0),
    (0.95, 0.55, 0.25, 1.0),
    (0.55, 0.55, 0.60, 1.0),
]

CONTAINER_FACE_COLORS = {
    "floor":  (0.82, 0.82, 0.84, 1.0),
    "wall_x": (0.72, 0.80, 0.92, 1.0),
    "wall_y": (0.88, 0.82, 0.90, 1.0),
}


# =========================
# 工具
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def weighted_choice(d: dict):
    keys = list(d.keys())
    probs = np.array(list(d.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs)


def to_uint8_image(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return (img * 255.0).clip(0, 255).astype(np.uint8)
    return img.clip(0, 255).astype(np.uint8)


def safe_scene_destroy(scene):
    if scene is not None:
        try:
            scene.destroy()
        except Exception:
            pass


def sample_background():
    presets = [
        {
            "name": "soft_studio",
            "background_color": [0.98, 0.98, 0.99],
            "ambient_light": [0.52, 0.52, 0.52],   # 提高环境光，减弱阴影生硬感
        },
        {
            "name": "light_gray_studio",
            "background_color": [0.94, 0.94, 0.95],
            "ambient_light": [0.50, 0.50, 0.50],
        },
        {
            "name": "warm_soft",
            "background_color": [0.98, 0.96, 0.93],
            "ambient_light": [0.50, 0.48, 0.46],
        },
    ]
    return random.choice(presets)


def pick_color(idx: int):
    return OBJECT_COLOR_PALETTE[idx % len(OBJECT_COLOR_PALETTE)]


# =========================
# 相机
# =========================


def sample_camera_from_objects(objects, corner_cfg):
    cx, cy, cz = corner_cfg["center"]
    
    # 计算物体包围盒
    max_x = max(float(obj["geom"]["bbox_extents"][0]) for obj in objects)
    max_y = max(float(obj["geom"]["bbox_extents"][1]) for obj in objects)
    max_z = max(float(obj["geom"]["bbox_extents"][2]) for obj in objects)
    obj_ref = max(max_x, max_y, max_z, 0.25)

    # --- 1. 调整观察重心 (Focus) ---
    # 将重心从墙根向外平移，使物体落点位于画面中心附近
    focus_x = cx + max(0.8, 2.0 * obj_ref)
    focus_y = cy + max(0.8, 2.0 * obj_ref)
    focus_z = cz + obj_ref * 0.2

    # --- 2. 距离控制 ---
    # 增加基础距离系数（从原来的 2.8 提升到 4.5+）
    # 如果觉得物体还是太大，可以调大下面这个 5.0
    dist = max(20.0 * obj_ref, 4.5) * np.random.uniform(1.1, 1.4)

    # --- 3. 极坐标位置采样 ---
    # theta: 俯仰角 (45-60度，偏高一点可以减少墙面遮挡并增加空间感)
    # phi: 水平旋转 (15-35度，确保从开口侧斜着看进去)
    theta = np.deg2rad(np.random.uniform(45, 60))
    phi = np.deg2rad(np.random.uniform(15, 35))

    # 计算相机坐标 (确保在 x > focus_x, y > focus_y 区域，即开口侧)
    cam_x = focus_x + dist * np.sin(theta) * np.cos(phi)
    cam_y = focus_y + dist * np.sin(theta) * np.sin(phi)
    cam_z = focus_z + dist * np.cos(theta)

    # --- 4. 镜头参数 ---
    # 使用较小的 FOV (35-45) 可以获得更自然的“长焦”效果，减少畸变
    fov = float(np.random.uniform(35, 42))

    return {
        "res": [IMG_W, IMG_H],
        "pos": [float(cam_x), float(cam_y), float(cam_z)],
        "lookat": [float(focus_x), float(focus_y), float(focus_z)],
        "fov": fov,
        "GUI": False,
    }




def build_corner_cfg_from_objects(objects, corner_base):
    max_x = max(float(obj["geom"]["bbox_extents"][0]) for obj in objects)
    max_y = max(float(obj["geom"]["bbox_extents"][1]) for obj in objects)
    max_z = max(float(obj["geom"]["bbox_extents"][2]) for obj in objects)

    ref = max(max_x, max_y, max_z, 0.25)

    panel_size = float(np.random.uniform(16.5 * ref, 20.0 * ref))
    thickness = max(0.05, 0.07 * ref)

    return {
        "center": list(corner_base["center"]),
        "panel_size": panel_size,
        "thickness": thickness,
    }
def add_large_corner(scene, corner_cfg: dict):
    cx, cy, cz = corner_cfg["center"]
    BIG = corner_cfg["panel_size"]
    THICK = corner_cfg["thickness"]

    wall_mat = gs.materials.Rigid(rho=1200.0, friction=0.95)

    floor_surface = gs.surfaces.Default(color=CONTAINER_FACE_COLORS["floor"])
    wall_x_surface = gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_x"])
    wall_y_surface = gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_y"])

    entities = {}

    # floor: z = 0 附近
    entities["floor"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BIG, BIG, THICK),
            pos=(cx + BIG * 0.5, cy + BIG * 0.5, cz - THICK * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=floor_surface,
    )

    # wall_x: x = 0 附近
    entities["wall_x"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(THICK, BIG, BIG),
            pos=(cx - THICK * 0.5, cy + BIG * 0.5, cz + BIG * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=wall_x_surface,
    )

    # wall_y: y = 0 附近
    entities["wall_y"] = scene.add_entity(
        morph=gs.morphs.Box(
            size=(BIG, THICK, BIG),
            pos=(cx + BIG * 0.5, cy - THICK * 0.5, cz + BIG * 0.5),
            fixed=True,
        ),
        material=wall_mat,
        surface=wall_y_surface,
    )

    return entities


# =========================
# 运动采样
# =========================
def sample_rigid_motion_in_corner(half_x: float, half_y: float, half_z: float):
    mode = weighted_choice(RIGID_MOTION_WEIGHTS)

    safe_x_min = 0.45 + half_x
    safe_y_min = 0.45 + half_y

    if mode == "top_drop":
        x = float(np.random.uniform(safe_x_min, 1.35))
        y = float(np.random.uniform(safe_y_min, 1.35))
        z = float(np.random.uniform(1.00, 1.90))

        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-0.22, -0.04)),
        ]
        angvel = [
            float(np.random.uniform(-2.0, 2.0)),
            float(np.random.uniform(-2.0, 2.0)),
            float(np.random.uniform(-1.2, 1.2)),
        ]
        pose_delta = [
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]

    elif mode == "top_toss":
        x = float(np.random.uniform(safe_x_min, 1.35))
        y = float(np.random.uniform(safe_y_min, 1.35))
        z = float(np.random.uniform(0.95, 1.70))

        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(-0.45, 0.45)),
            float(np.random.uniform(-0.35, 0.35)),
            float(np.random.uniform(-0.95, -0.30)),
        ]
        angvel = [
            float(np.random.uniform(-4.0, 4.0)),
            float(np.random.uniform(-4.0, 4.0)),
            float(np.random.uniform(-4.0, 4.0)),
        ]
        pose_delta = [
            float(np.random.uniform(-0.18, 0.18)),
            float(np.random.uniform(-0.18, 0.18)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]

    elif mode == "side_throw_x":
        x = float(np.random.uniform(1.50, 2.20))
        y = float(np.random.uniform(safe_y_min, 1.35))
        z = float(np.random.uniform(0.55, 1.30))

        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(-2.10, -1.25)),
            float(np.random.uniform(-0.18, 0.14)),
            float(np.random.uniform(0.75, 1.45)),
        ]
        angvel = [
            float(np.random.uniform(-6.0, 6.0)),
            float(np.random.uniform(-6.0, 6.0)),
            float(np.random.uniform(-6.0, 6.0)),
        ]
        pose_delta = [
            float(np.random.uniform(-0.30, 0.30)),
            float(np.random.uniform(-0.30, 0.30)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]

    elif mode == "side_throw_y":
        x = float(np.random.uniform(safe_x_min, 1.35))
        y = float(np.random.uniform(1.50, 2.20))
        z = float(np.random.uniform(0.55, 1.30))

        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(-0.14, 0.18)),
            float(np.random.uniform(-2.10, -1.25)),
            float(np.random.uniform(0.75, 1.45)),
        ]
        angvel = [
            float(np.random.uniform(-6.0, 6.0)),
            float(np.random.uniform(-6.0, 6.0)),
            float(np.random.uniform(-6.0, 6.0)),
        ]
        pose_delta = [
            float(np.random.uniform(-0.30, 0.30)),
            float(np.random.uniform(-0.30, 0.30)),
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
    v = np.asarray(obj_meta.get("init_linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
    w = np.asarray(obj_meta.get("init_angvel", [0.0, 0.0, 0.0]), dtype=np.float32)

    if np.linalg.norm(v) > 0:
        _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], v)
    if np.linalg.norm(w) > 0:
        _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], w)


# =========================
# 候选池
# =========================
def build_physxnet_object_bank(loader, max_objects_to_read: int):
    all_ids = loader.list_object_ids()
    if max_objects_to_read is not None:
        all_ids = all_ids[:max_objects_to_read]

    bank = []
    failed = []

    total = len(all_ids)
    print(f"[INFO] start loading candidate pool: {total} objects")

    for i, obj_id in enumerate(all_ids, 1):
        try:
            obj = loader.get_object(obj_id, export_merged=EXPORT_MERGED_WHEN_LOADING)

            dim_m = np.array(obj.dimension_m, dtype=np.float32)
            if np.max(dim_m) > 5.0:
                failed.append({"object_id": obj_id, "error": "too_large_raw_dimension"})
                continue

            bank.append({
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "category": obj.category,
                "dimension_m": obj.dimension_m,
                "merged_mesh_path": obj.merged_mesh_path,
                "genesis_rigid": obj.genesis_rigid,
                "parts": obj.parts,
            })

            if i % 50 == 0 or i == total:
                print(f"[INFO] loaded {i}/{total}")

        except Exception as e:
            failed.append({"object_id": obj_id, "error": str(e)})
            print(f"[WARN] skip {obj_id}: {e}")

    print(f"[INFO] usable objects in bank: {len(bank)} | failed: {len(failed)}")
    return bank, failed


# =========================
# 物体采样：带颜色
# =========================
def sample_physxnet_rigid_object(obj_idx: int, bank_item: dict):
    dim_m = np.array(bank_item["dimension_m"], dtype=np.float32)

    if np.max(dim_m) <= 1e-6:
        dim_m = np.array([0.25, 0.25, 0.25], dtype=np.float32)

    max_dim = float(np.max(dim_m))
    target_max_dim = float(np.random.uniform(0.18, 0.45))
    mesh_scale = target_max_dim / max(max_dim, 1e-8)

    scaled_dim = dim_m * mesh_scale
    half_x = float(max(scaled_dim[0] / 2.0, 0.03))
    half_y = float(max(scaled_dim[1] / 2.0, 0.03))
    half_z = float(max(scaled_dim[2] / 2.0, 0.03))

    motion = sample_rigid_motion_in_corner(half_x, half_y, half_z)

    rigid = bank_item["genesis_rigid"]
    material = rigid["material"]

    rho = material.get("density", 1200.0)
    young = material.get("youngs_modulus", None)
    poisson = material.get("poisson_ratio", None)
    friction = float(np.random.uniform(0.25, 0.85))

    color = pick_color(obj_idx)

    final_euler = [
        float(motion["pose_delta"][0]),
        float(motion["pose_delta"][1]),
        float(motion["pose_delta"][2]),
    ]

    return {
        "scene_object_id": obj_idx,
        "solver": "Rigid",
        "source_type": "physxnet_merged_mesh",
        "motion_type": motion["motion_type"],
        "object_id": bank_item["object_id"],
        "object_name": bank_item["object_name"],
        "category": bank_item["category"],
        "geom": {
            "shape": "mesh",
            "mesh_file": rigid["morph"]["file"],
            "scale": float(mesh_scale),
            "bbox_extents": scaled_dim.tolist(),
        },
        "material": {
            "family": "Rigid",
            "rho": float(rho) if rho is not None else 1200.0,
            "friction": friction,
            "young": young,
            "poisson": poisson,
        },
        "render_color": color,
        "init_pos": [float(x) for x in motion["init_pos"]],
        "init_euler": final_euler,
        "init_linvel": [float(x) for x in motion["init_linvel"]],
        "init_angvel": [float(x) for x in motion["init_angvel"]],
    }


# =========================
# 场景采样
# =========================
def sample_scene_cfg(scene_id: int, object_bank: list):
    seed = 100000 + scene_id
    set_seed(seed)

    bg = sample_background()

    n_obj = random.randint(MIN_OBJECTS_PER_SCENE, MAX_OBJECTS_PER_SCENE)
    n_obj = min(n_obj, len(object_bank))
    chosen = random.sample(object_bank, k=n_obj)

    objects = [
        sample_physxnet_rigid_object(i, bank_item)
        for i, bank_item in enumerate(chosen)
    ]

    corner_cfg = build_corner_cfg_from_objects(objects, CORNER_BASE)
    cam = sample_camera_from_objects(objects, corner_cfg)

    sim_options = {
        "gravity": [0.0, 0.0, -9.81],
        "dt": 4e-3,
        "substeps": 8,
        "num_steps": 180,
    }

    return {
        "scene_id": f"scene_{scene_id:06d}",
        "seed": seed,
        "family": "rigid_mix",
        "background": bg,
        "corner": corner_cfg,
        "camera": cam,
        "sim_options": sim_options,
        "objects": objects,
    }


# =========================
# 构建场景
# =========================
def build_scene(scene_cfg: dict):
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
    # scene.add_light(
    #         pos=(5.0, 5.0, 10.0),
    #         color=(1.0, 1.0, 1.0),
    #         intensity=800.0, # 如果画面太亮或太暗，调整这个值
    #         lookat=(0.0, 0.0, 0.0)
    #     )
    add_large_corner(scene, scene_cfg["corner"])

    entities = []
    for obj in scene_cfg["objects"]:
        mat = gs.materials.Rigid(
            rho=obj["material"]["rho"],
            friction=obj["material"]["friction"],
        )

        surface = gs.surfaces.Default(color=obj["render_color"])

        ent = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=obj["geom"]["mesh_file"],
                scale=obj["geom"]["scale"],
                pos=tuple(obj["init_pos"]),
                euler=tuple(obj["init_euler"]),
            ),
            material=mat,
            surface=surface,
        )
        entities.append(ent)

    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )

    scene.build()

    for obj, ent in zip(scene_cfg["objects"], entities):
        apply_initial_motion_to_rigid_entity(ent, obj)

    return scene, cam, entities


# =========================
# 导出
# =========================
def export_scene(scene_cfg: dict):
    out_dir = OUT_ROOT / "train" / scene_cfg["scene_id"]
    ensure_dir(out_dir / "rgb")
    ensure_dir(out_dir / "video")

    with open(out_dir / "scene_input.json", "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, ensure_ascii=False, indent=2)

    scene, cam, entities = None, None, None

    try:
        scene, cam, entities = build_scene(scene_cfg)

        preview_frames = []
        num_steps = scene_cfg["sim_options"]["num_steps"]

        for t in range(num_steps):
            scene.step()

            render_out = cam.render(rgb=True)
            rgb = render_out[0] if isinstance(render_out, tuple) else render_out

            if rgb is None:
                raise RuntimeError(f"cam.render returned None at step {t}")

            rgb = to_uint8_image(rgb)

            rgb_path = out_dir / "rgb" / f"{t:06d}.png"
            imageio.imwrite(rgb_path, rgb)

            if t % 3 == 0:
                preview_frames.append(rgb)

        if len(preview_frames) > 0:
            imageio.mimsave(out_dir / "video" / "preview.mp4", preview_frames, fps=20)

        scene_metadata = {
            "scene_id": scene_cfg["scene_id"],
            "seed": scene_cfg["seed"],
            "family": scene_cfg["family"],
            "corner": scene_cfg["corner"],
            "camera": scene_cfg["camera"],
            "num_objects": len(scene_cfg["objects"]),
            "objects": [
                {
                    "scene_object_id": x["scene_object_id"],
                    "object_id": x["object_id"],
                    "object_name": x["object_name"],
                    "category": x["category"],
                    "mesh_file": x["geom"]["mesh_file"],
                    "scale": x["geom"]["scale"],
                    "bbox_extents": x["geom"]["bbox_extents"],
                    "motion_type": x["motion_type"],
                    "render_color": x["render_color"],
                }
                for x in scene_cfg["objects"]
            ],
            "container_face_colors": CONTAINER_FACE_COLORS,
            "status": "ok",
        }

        with open(out_dir / "scene_metadata.json", "w", encoding="utf-8") as f:
            json.dump(scene_metadata, f, ensure_ascii=False, indent=2)

        return scene_metadata

    finally:
        safe_scene_destroy(scene)


# =========================
# 主程序
# =========================
def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_ROOT / "train")
    ensure_dir(MERGED_CACHE_DIR)

    loader = PhysXNetGenesisLoader(
        root=str(PHYSXNET_ROOT),
        version=PHYSXNET_VERSION,
        merged_cache_dir=str(MERGED_CACHE_DIR),
    )

    object_bank, failed = build_physxnet_object_bank(
        loader=loader,
        max_objects_to_read=MAX_DATASET_OBJECTS_TO_READ,
    )

    if len(object_bank) == 0:
        raise RuntimeError("No usable PhysXNet objects loaded.")

    with open(OUT_ROOT / "object_bank_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "physxnet_root": str(PHYSXNET_ROOT),
                "version": PHYSXNET_VERSION,
                "max_dataset_objects_to_read": MAX_DATASET_OBJECTS_TO_READ,
                "n_usable_objects": len(object_bank),
                "n_failed_objects": len(failed),
                "usable_object_ids": [x["object_id"] for x in object_bank],
                "failed": failed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    backend_used = "cpu"
    try:
        gs.init(backend=gs.gpu)
        backend_used = "gpu"
    except Exception:
        gs.init(backend=gs.cpu)
        backend_used = "cpu"

    manifest = {
        "dataset_name": "physxnet_genesis_colored_corner",
        "split": "train",
        "n_scenes_requested": N_SCENES,
        "image_size": [IMG_W, IMG_H],
        "backend_used": backend_used,
        "physxnet_root": str(PHYSXNET_ROOT),
        "physxnet_version": PHYSXNET_VERSION,
        "max_dataset_objects_to_read": MAX_DATASET_OBJECTS_TO_READ,
        "min_objects_per_scene": MIN_OBJECTS_PER_SCENE,
        "max_objects_per_scene": MAX_OBJECTS_PER_SCENE,
        "n_usable_objects": len(object_bank),
        "container_face_colors": CONTAINER_FACE_COLORS,
        "scenes": [],
        "failed_scenes": [],
    }

    try:
        for sid in range(N_SCENES):
            scene_cfg = sample_scene_cfg(sid, object_bank=object_bank)

            try:
                max_extent = max(max(obj["geom"]["bbox_extents"]) for obj in scene_cfg["objects"])
                print(
                    f"[RUN ] {scene_cfg['scene_id']} | "
                    f"n_obj={len(scene_cfg['objects'])} | "
                    f"max_obj_extent={max_extent:.3f} | "
                    f"corner_size={scene_cfg['corner']['panel_size']:.3f}"
                )
                meta = export_scene(scene_cfg)
                manifest["scenes"].append(meta)
                print(f"[ OK ] {scene_cfg['scene_id']} | n_obj={meta['num_objects']}")

            except Exception as e:
                err_info = {
                    "scene_id": scene_cfg["scene_id"],
                    "seed": scene_cfg["seed"],
                    "error": str(e),
                }
                manifest["failed_scenes"].append(err_info)
                print(f"[FAIL] {scene_cfg['scene_id']} | err={e}")

                if STOP_ON_ERROR:
                    raise

    finally:
        with open(OUT_ROOT / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        try:
            gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
'''


CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/Code_data/genesis_demo_physxnet_loader.py

'''