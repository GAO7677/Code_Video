# 用途：提供场景模板与采样辅助函数。
"""该模块用于定义 Genesis 刚体场景模板与采样辅助函数；输入为对象数量、布局和物理参数，输出为可直接用于数据生成的场景配置字典。"""
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CODE_DATA_DIR = THIS_DIR.parents[1]
if str(CODE_DATA_DIR) not in sys.path:
    sys.path.append(str(CODE_DATA_DIR))

from old_version.dataset_3_utils_genesis import set_seed, weighted_choice


SCENE_TYPE_WEIGHTS = {
    "drop": 1.0,
    "slide": 1.0,
    "collision": 1.0,
    "stack_topple": 1.0,
    "roll": 1.0,
}

SCENE_TYPE_MIN_COUNTS = {
    "drop": 1,
    "slide": 1,
    "collision": 2,
    "stack_topple": 2,
    "roll": 1,
}

TRAY_CFG = {
    "half_x": 1.25,
    "half_y": 1.35,
    "wall_thickness": 0.05,
    "wall_height": 0.50,
    "floor_thickness": 0.06,
    "center": [0.0, 0.0, 0.0],
}

GROUND_PRESETS = [
    {"name": "rubber_floor", "rho": 1100.0, "friction": 0.95, "restitution": 0.06, "color": [0.76, 0.77, 0.80, 1.0]},
    {"name": "wood_floor", "rho": 760.0, "friction": 0.62, "restitution": 0.10, "color": [0.73, 0.63, 0.50, 1.0]},
    {"name": "metal_floor", "rho": 2500.0, "friction": 0.28, "restitution": 0.08, "color": [0.70, 0.72, 0.76, 1.0]},
    {"name": "plastic_floor", "rho": 960.0, "friction": 0.42, "restitution": 0.14, "color": [0.82, 0.84, 0.86, 1.0]},
]

SHAPE_WEIGHTS = {
    "cube": 0.26,
    "cuboid": 0.30,
    "sphere": 0.22,
    "cylinder": 0.22,
}

OBJECT_COLORS = [
    [0.90, 0.28, 0.22, 1.0],
    [0.20, 0.50, 0.88, 1.0],
    [0.93, 0.72, 0.20, 1.0],
    [0.25, 0.72, 0.52, 1.0],
    [0.60, 0.42, 0.92, 1.0],
    [0.90, 0.45, 0.72, 1.0],
]


def _material_name(friction: float, restitution: float, rho: float) -> str:
    if rho > 2200:
        return "metal_like"
    if friction > 0.82:
        return "rubber_like"
    if restitution > 0.18:
        return "bouncy_plastic"
    if rho < 500:
        return "foam_like"
    return "plastic_like"


def sample_camera_cfg(width: int, height: int) -> Dict[str, Any]:
    radius = random.uniform(2.3, 2.8)
    azimuth = random.uniform(-0.40, 0.40)
    elevation = random.uniform(0.95, 1.35)
    pos = [
        float(radius * math.sin(azimuth)),
        float(-radius * math.cos(azimuth)),
        float(elevation),
    ]
    lookat = [
        float(random.uniform(-0.08, 0.08)),
        float(random.uniform(0.00, 0.25)),
        float(random.uniform(0.16, 0.34)),
    ]
    return {
        "res": [int(width), int(height)],
        "pos": pos,
        "lookat": lookat,
        "up": [0.0, 0.0, 1.0],
        "fov": float(random.uniform(34.0, 46.0)),
        "GUI": False,
    }


def _rotation_matrix_xyz(euler_xyz: List[float]) -> np.ndarray:
    rx, ry, rz = [float(v) for v in euler_xyz]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz_m @ ry_m @ rx_m).astype(np.float32)


def _shape_local_half_extents(shape: str, geom: Dict[str, Any]) -> np.ndarray:
    if shape in {"cube", "cuboid"}:
        size = np.asarray(geom["size"], dtype=np.float32)
        return 0.5 * size
    if shape == "sphere":
        radius = float(geom["radius"])
        return np.asarray([radius, radius, radius], dtype=np.float32)
    if shape == "cylinder":
        radius = float(geom["radius"])
        half_height = 0.5 * float(geom["height"])
        return np.asarray([radius, radius, half_height], dtype=np.float32)
    raise ValueError(f"Unsupported shape: {shape}")


def compute_vertical_half_extent(shape: str, geom: Dict[str, Any], euler_xyz: List[float]) -> float:
    local_half_extents = _shape_local_half_extents(shape, geom)
    rot = np.abs(_rotation_matrix_xyz(euler_xyz))
    return float(rot[2].dot(local_half_extents))


def estimate_volume_m3(shape: str, geom: Dict[str, Any]) -> float:
    if shape in {"cube", "cuboid"}:
        sx, sy, sz = [float(v) for v in geom["size"]]
        return float(sx * sy * sz)
    if shape == "sphere":
        radius = float(geom["radius"])
        return float((4.0 / 3.0) * math.pi * (radius ** 3))
    if shape == "cylinder":
        radius = float(geom["radius"])
        height = float(geom["height"])
        return float(math.pi * (radius ** 2) * height)
    raise ValueError(f"Unsupported shape: {shape}")


def sample_shape(scene_type: str) -> str:
    if scene_type == "roll":
        return random.choice(["sphere", "cylinder"])
    if scene_type == "stack_topple":
        return random.choice(["cube", "cuboid", "cylinder"])
    return weighted_choice(SHAPE_WEIGHTS)


def sample_geometry(shape: str) -> Dict[str, Any]:
    if shape == "cube":
        edge = float(random.uniform(0.14, 0.24))
        return {"size": [edge, edge, edge]}
    if shape == "cuboid":
        return {
            "size": [
                float(random.uniform(0.14, 0.30)),
                float(random.uniform(0.12, 0.24)),
                float(random.uniform(0.10, 0.22)),
            ]
        }
    if shape == "sphere":
        return {"radius": float(random.uniform(0.08, 0.14))}
    if shape == "cylinder":
        return {
            "radius": float(random.uniform(0.07, 0.12)),
            "height": float(random.uniform(0.14, 0.28)),
        }
    raise ValueError(f"Unsupported shape: {shape}")


def sample_object_material() -> Dict[str, Any]:
    rho = float(random.uniform(250.0, 3200.0))
    friction = float(random.uniform(0.12, 1.05))
    restitution = float(random.uniform(0.02, 0.35))
    return {
        "name": _material_name(friction, restitution, rho),
        "rho": rho,
        "friction": friction,
        "sampled_restitution": restitution,
        "effective_restitution_used": None,
    }


def sample_ground_material(scene_type: str) -> Dict[str, Any]:
    preset = dict(random.choice(GROUND_PRESETS))
    if scene_type == "slide":
        preset["friction"] = float(random.uniform(0.08, 0.28))
    elif scene_type == "roll":
        preset["friction"] = float(random.uniform(0.10, 0.45))
    else:
        preset["friction"] = float(np.clip(preset["friction"] + random.uniform(-0.12, 0.12), 0.08, 1.10))
    sampled_restitution = float(np.clip(preset["restitution"] + random.uniform(-0.04, 0.05), 0.0, 0.35))
    preset["sampled_restitution"] = sampled_restitution
    preset["effective_restitution_used"] = None
    preset.pop("restitution", None)
    return preset


def _base_object_dict(
    object_id: int,
    shape: str,
    geom: Dict[str, Any],
    euler: List[float],
    pos: List[float],
    linvel: List[float],
    angvel: List[float],
    color: List[float],
    material: Dict[str, Any],
) -> Dict[str, Any]:
    volume = estimate_volume_m3(shape, geom)
    bbox_extent = (2.0 * _shape_local_half_extents(shape, geom)).astype(np.float32)
    return {
        "object_id": int(object_id),
        "shape": shape,
        "geometry": geom,
        "canonical_size": {
            "bbox_extent": [float(v) for v in bbox_extent.tolist()],
        },
        "euler": [float(v) for v in euler],
        "pos": [float(v) for v in pos],
        "linvel": [float(v) for v in linvel],
        "angvel": [float(v) for v in angvel],
        "color": [float(v) for v in color],
        "material": {
            **material,
            "approx_mass_kg": float(max(material["rho"] * volume, 1e-4)),
        },
    }


def _sample_xy(slot_radius: float, occupied_xy: List[Tuple[float, float]], bias_to_back: bool) -> Tuple[float, float]:
    x_min = -TRAY_CFG["half_x"] + TRAY_CFG["wall_thickness"] + slot_radius + 0.05
    x_max = TRAY_CFG["half_x"] - TRAY_CFG["wall_thickness"] - slot_radius - 0.05
    y_min = -TRAY_CFG["half_y"] + slot_radius + 0.10
    y_max = TRAY_CFG["half_y"] - TRAY_CFG["wall_thickness"] - slot_radius - 0.08
    if bias_to_back:
        y_min = max(y_min, 0.20)
    for _ in range(80):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if all(math.hypot(x - ox, y - oy) >= slot_radius + 0.10 for ox, oy in occupied_xy):
            return float(x), float(y)
    return float(random.uniform(x_min, x_max)), float(random.uniform(y_min, y_max))


def _sample_drop_objects(num_objects: int, ground_top_z: float) -> List[Dict[str, Any]]:
    occupied_xy: List[Tuple[float, float]] = []
    objects = []
    for idx in range(num_objects):
        shape = sample_shape("drop")
        geom = sample_geometry(shape)
        euler = [
            float(random.uniform(-0.35, 0.35)),
            float(random.uniform(-0.35, 0.35)),
            float(random.uniform(-math.pi, math.pi)),
        ]
        slot_radius = float(np.linalg.norm(_shape_local_half_extents(shape, geom)[:2]))
        x, y = _sample_xy(slot_radius, occupied_xy, bias_to_back=False)
        occupied_xy.append((x, y))
        rest_half_z = compute_vertical_half_extent(shape, geom, euler)
        z = ground_top_z + rest_half_z + random.uniform(0.45, 1.20)
        linvel = [float(random.uniform(-0.08, 0.08)), float(random.uniform(-0.08, 0.12)), float(random.uniform(-0.20, 0.0))]
        angvel = [float(random.uniform(-3.0, 3.0)), float(random.uniform(-3.0, 3.0)), float(random.uniform(-3.0, 3.0))]
        objects.append(
            _base_object_dict(idx, shape, geom, euler, [x, y, z], linvel, angvel, random.choice(OBJECT_COLORS), sample_object_material())
        )
    return objects


def _sample_slide_objects(num_objects: int, ground_top_z: float) -> List[Dict[str, Any]]:
    occupied_xy: List[Tuple[float, float]] = []
    objects = []
    for idx in range(num_objects):
        shape = random.choice(["cube", "cuboid", "cylinder", "sphere"])
        geom = sample_geometry(shape)
        euler = [0.0, 0.0, float(random.uniform(-0.35, 0.35))]
        if shape == "cylinder" and random.random() < 0.6:
            euler[0] = math.pi / 2.0
        slot_radius = float(np.linalg.norm(_shape_local_half_extents(shape, geom)[:2]))
        x, y = _sample_xy(slot_radius, occupied_xy, bias_to_back=False)
        y = min(y, -0.20 + 0.18 * idx)
        occupied_xy.append((x, y))
        rest_half_z = compute_vertical_half_extent(shape, geom, euler)
        z = ground_top_z + rest_half_z + 0.004
        linvel = [float(random.uniform(-0.10, 0.10)), float(random.uniform(0.95, 1.60)), 0.0]
        angvel = [0.0, 0.0, float(random.uniform(-1.0, 1.0))]
        objects.append(
            _base_object_dict(idx, shape, geom, euler, [x, y, z], linvel, angvel, random.choice(OBJECT_COLORS), sample_object_material())
        )
    return objects


def _sample_collision_objects(num_objects: int, ground_top_z: float) -> List[Dict[str, Any]]:
    objects = []
    primary_shapes = [sample_shape("collision"), sample_shape("collision")]
    for idx, start_x, vx in [(0, -0.55, 1.45), (1, 0.55, -1.45)]:
        shape = primary_shapes[idx]
        geom = sample_geometry(shape)
        euler = [
            float(random.uniform(-0.25, 0.25)),
            float(random.uniform(-0.25, 0.25)),
            float(random.uniform(-math.pi, math.pi)),
        ]
        rest_half_z = compute_vertical_half_extent(shape, geom, euler)
        z = ground_top_z + rest_half_z + 0.004
        linvel = [float(vx + random.uniform(-0.18, 0.18)), float(random.uniform(-0.08, 0.08)), 0.0]
        angvel = [float(random.uniform(-2.2, 2.2)), float(random.uniform(-2.2, 2.2)), float(random.uniform(-2.2, 2.2))]
        objects.append(
            _base_object_dict(
                idx,
                shape,
                geom,
                euler,
                [start_x + random.uniform(-0.04, 0.04), random.uniform(-0.04, 0.12), z],
                linvel,
                angvel,
                OBJECT_COLORS[idx % len(OBJECT_COLORS)],
                sample_object_material(),
            )
        )

    occupied_xy = [(objects[0]["pos"][0], objects[0]["pos"][1]), (objects[1]["pos"][0], objects[1]["pos"][1])]
    for idx in range(2, num_objects):
        shape = random.choice(["cube", "cuboid", "sphere"])
        geom = sample_geometry(shape)
        euler = [0.0, 0.0, float(random.uniform(-math.pi, math.pi))]
        slot_radius = float(np.linalg.norm(_shape_local_half_extents(shape, geom)[:2]))
        x, y = _sample_xy(slot_radius, occupied_xy, bias_to_back=True)
        occupied_xy.append((x, y))
        rest_half_z = compute_vertical_half_extent(shape, geom, euler)
        z = ground_top_z + rest_half_z + 0.004
        objects.append(
            _base_object_dict(
                idx,
                shape,
                geom,
                euler,
                [x, y, z],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                OBJECT_COLORS[idx % len(OBJECT_COLORS)],
                sample_object_material(),
            )
        )
    return objects


def _sample_stack_topple_objects(num_objects: int, ground_top_z: float) -> List[Dict[str, Any]]:
    objects = []
    tower_x = float(random.uniform(-0.08, 0.08))
    tower_y = float(random.uniform(0.18, 0.38))
    z_cursor = ground_top_z
    for idx in range(num_objects):
        shape = random.choice(["cube", "cuboid", "cylinder"])
        geom = sample_geometry(shape)
        euler = [0.0, 0.0, float(random.uniform(-0.08, 0.08))]
        if shape == "cylinder" and random.random() < 0.25:
            euler[0] = math.pi / 2.0
        half_z = compute_vertical_half_extent(shape, geom, euler)
        lateral_offset = 0.012 * idx + random.uniform(-0.006, 0.006)
        pos = [tower_x + lateral_offset, tower_y + 0.006 * idx, z_cursor + half_z + 0.003]
        z_cursor = pos[2] + half_z + 0.003
        linvel = [0.0, 0.0, 0.0]
        angvel = [0.0, 0.0, 0.0]
        if idx == num_objects - 1:
            linvel = [float(random.uniform(0.08, 0.18)), float(random.uniform(-0.04, 0.04)), 0.0]
        objects.append(
            _base_object_dict(idx, shape, geom, euler, pos, linvel, angvel, OBJECT_COLORS[idx % len(OBJECT_COLORS)], sample_object_material())
        )
    return objects


def _sample_roll_objects(num_objects: int, ground_top_z: float) -> List[Dict[str, Any]]:
    occupied_xy: List[Tuple[float, float]] = []
    objects = []
    for idx in range(num_objects):
        shape = random.choice(["sphere", "cylinder"])
        geom = sample_geometry(shape)
        euler = [0.0, 0.0, 0.0]
        if shape == "cylinder":
            euler[0] = math.pi / 2.0
            euler[2] = float(random.uniform(-0.10, 0.10))
        slot_radius = float(np.linalg.norm(_shape_local_half_extents(shape, geom)[:2]))
        x = -0.62 + 0.34 * idx + random.uniform(-0.03, 0.03)
        y = -0.15 + random.uniform(-0.10, 0.18)
        if occupied_xy:
            x = max(x, occupied_xy[-1][0] + slot_radius + 0.18)
        occupied_xy.append((x, y))
        rest_half_z = compute_vertical_half_extent(shape, geom, euler)
        z = ground_top_z + rest_half_z + 0.004
        vx = float(random.uniform(0.95, 1.55))
        linvel = [vx, float(random.uniform(-0.05, 0.14)), 0.0]
        if shape == "sphere":
            angvel = [float(random.uniform(-1.6, 1.6)), float(random.uniform(-9.0, -4.5)), float(random.uniform(-1.2, 1.2))]
        else:
            angvel = [0.0, float(random.uniform(-9.5, -4.5)), 0.0]
        objects.append(
            _base_object_dict(idx, shape, geom, euler, [x, y, z], linvel, angvel, OBJECT_COLORS[idx % len(OBJECT_COLORS)], sample_object_material())
        )
    return objects


def sample_scene_template(
    seed: int,
    width: int,
    height: int,
    frames: int,
    fps: int,
    scene_type: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(int(seed))
    scene_type = scene_type or weighted_choice(SCENE_TYPE_WEIGHTS)
    num_objects = max(SCENE_TYPE_MIN_COUNTS[scene_type], random.randint(1, 4))
    ground_material = sample_ground_material(scene_type)
    ground_top_z = float(TRAY_CFG["floor_thickness"])

    if scene_type == "drop":
        objects = _sample_drop_objects(num_objects, ground_top_z)
    elif scene_type == "slide":
        objects = _sample_slide_objects(num_objects, ground_top_z)
    elif scene_type == "collision":
        objects = _sample_collision_objects(num_objects, ground_top_z)
    elif scene_type == "stack_topple":
        objects = _sample_stack_topple_objects(num_objects, ground_top_z)
    elif scene_type == "roll":
        objects = _sample_roll_objects(num_objects, ground_top_z)
    else:
        raise ValueError(f"Unsupported scene type: {scene_type}")

    return {
        "seed": int(seed),
        "scene_type": scene_type,
        "num_objects": int(len(objects)),
        "frames": int(frames),
        "fps": int(fps),
        "tray": dict(TRAY_CFG),
        "ground_material": ground_material,
        "camera": sample_camera_cfg(width=width, height=height),
        "objects": objects,
    }
