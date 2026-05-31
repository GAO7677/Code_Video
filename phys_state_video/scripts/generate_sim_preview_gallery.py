#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np

np.infty = np.inf

import pybullet as p
import pybullet_data
import pyrender
import trimesh
from pyrender.constants import RenderFlags


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/sim_objstate_rigid_v3_preview")
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
VIDEO_DIR = OUTPUT_ROOT / "videos"
META_DIR = OUTPUT_ROOT / "meta"
DEFAULT_PORT = 18823

IMG_W = 960
IMG_H = 540
SIM_HZ = 240
FPS = 30
SIM_DURATION = 3.0
RECORD_EVERY = max(1, SIM_HZ // FPS)

CAM_EYE = np.array([0.0, -3.0, 1.42], dtype=np.float64)
CAM_TARGET = np.array([0.0, 0.28, 0.38], dtype=np.float64)
CAM_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
EARTH_GRAVITY = 9.81

ASSET_ROOTS = {
    "kaolin": Path("/home/gaoya/Code_Video/kaolin-master/sample_data/meshes"),
    "genesis_mesh": Path("/home/gaoya/Code_Video/Genesis_main/genesis/assets/meshes"),
    "genesis_wheel": Path("/home/gaoya/Code_Video/Genesis_main/genesis/assets/urdf/wheel"),
    "demo": Path("/home/gaoya/Code_Video/Code_data/demo_outputs"),
}

LOCAL_ASSETS = {
    "avocado": ASSET_ROOTS["kaolin"] / "avocado.obj",
    "pizza": ASSET_ROOTS["kaolin"] / "pizza.obj",
    "fox": ASSET_ROOTS["kaolin"] / "fox.obj",
    "armchair": ASSET_ROOTS["kaolin"] / "armchair.obj",
    "duck": ASSET_ROOTS["genesis_mesh"] / "duck.obj",
    "bunny": ASSET_ROOTS["genesis_mesh"] / "bunny.obj",
    "fancy_wheel": ASSET_ROOTS["genesis_wheel"] / "fancy_wheel.obj",
    "bowl": ASSET_ROOTS["demo"] / "raw_bowl_mesh" / "raw_bowl_mesh.obj",
}


@dataclass
class ObjectSpec:
    name: str
    shape: str
    color: List[float]
    mass: float
    position: List[float]
    size: Dict[str, float]
    dynamic: bool = True
    restitution: float = 0.55
    friction: float = 0.55
    linear_damping: float = 0.03
    angular_damping: float = 0.03
    orientation_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    linear_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    role: str = "dynamic"
    render_mode: str = "primitive"
    texture_style: str = "solid"
    mesh_path: str = ""
    mesh_target_extents: List[float] = field(default_factory=list)
    mesh_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    collision_proxy: Dict[str, object] | None = None


@dataclass
class ScenarioSpec:
    key: str
    family: str
    title: str
    description: str
    gravity: float
    floor_friction: float
    objects: List[ObjectSpec]
    seed: int
    pre_roll_s: float = 0.75
    sim_type: str = "rigid"


def _quat_from_euler_deg(values: List[float]) -> List[float]:
    radians = [math.radians(v) for v in values]
    return list(p.getQuaternionFromEuler(radians))


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def _tr(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return pose


def _pb_pose(pos: List[float], quat: List[float]) -> np.ndarray:
    rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    return pose


def _apply_euler_transform(mesh: trimesh.Trimesh, euler_deg: List[float]) -> trimesh.Trimesh:
    mesh = mesh.copy()
    quat = _quat_from_euler_deg(euler_deg)
    rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    mesh.apply_transform(transform)
    return mesh


def _normalize_mesh(mesh: trimesh.Trimesh, target_extents: List[float]) -> trimesh.Trimesh:
    mesh = mesh.copy()
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) * 0.5
    mesh.apply_translation(-center)
    extents = np.maximum(mesh.extents.astype(np.float64), 1e-6)
    if target_extents:
        target = np.asarray(target_extents, dtype=np.float64)
        scale = float(np.min(target / extents))
        mesh.apply_scale(scale)
    return mesh


def _make_primitive_mesh(obj: ObjectSpec) -> trimesh.Trimesh:
    if obj.shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=obj.size["radius"])
    elif obj.shape == "box":
        mesh = trimesh.creation.box(extents=[
            obj.size["hx"] * 2.0,
            obj.size["hy"] * 2.0,
            obj.size["hz"] * 2.0,
        ])
    elif obj.shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=obj.size["radius"], height=obj.size["height"], sections=40)
    elif obj.shape == "capsule":
        mesh = trimesh.creation.capsule(radius=obj.size["radius"], height=obj.size["height"], count=[16, 24])
    else:
        raise ValueError(f"unsupported primitive shape: {obj.shape}")
    _apply_procedural_material(mesh, obj)
    return mesh


def _apply_procedural_material(mesh: trimesh.Trimesh, obj: ObjectSpec) -> None:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    base = np.tile(np.asarray(obj.color, dtype=np.float32), (len(verts), 1))
    style = obj.texture_style
    if style == "solid":
        colors = base
    elif style == "wood":
        grain = 0.5 + 0.5 * np.sin(verts[:, 2] * 26.0 + verts[:, 0] * 4.0)
        colors = base.copy()
        colors[:, 0] += grain * 0.10
        colors[:, 1] += grain * 0.06
    elif style == "metal":
        sheen = 0.35 + 0.65 * (verts[:, 2] - verts[:, 2].min()) / max(float(np.ptp(verts[:, 2])), 1e-6)
        colors = base * (0.8 + 0.3 * sheen[:, None])
    elif style == "rubber":
        speckle = 0.5 + 0.5 * np.sin(verts[:, 0] * 55.0) * np.cos(verts[:, 1] * 40.0)
        colors = base * (0.82 + 0.18 * speckle[:, None])
    elif style == "checker":
        pattern = (((verts[:, 0] * 10).astype(int) + (verts[:, 1] * 10).astype(int)) % 2).astype(np.float32)
        colors = base * (0.72 + 0.28 * pattern[:, None])
    else:
        colors = base
    noise = np.random.default_rng(42).uniform(-0.03, 0.03, colors.shape).astype(np.float32)
    mesh.visual.vertex_colors = np.clip(colors + noise, 0.0, 1.0)


def _load_render_mesh(obj: ObjectSpec) -> trimesh.Trimesh:
    if obj.render_mode == "primitive":
        return _make_primitive_mesh(obj)
    mesh_path = Path(obj.mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh = _apply_euler_transform(mesh, obj.mesh_euler_deg)
    mesh = _normalize_mesh(mesh, obj.mesh_target_extents)
    if not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        colors = np.tile(np.asarray(obj.color, dtype=np.float32), (len(mesh.vertices), 1))
        mesh.visual.vertex_colors = colors
    return mesh


def _collision_proxy_for_object(obj: ObjectSpec) -> Dict[str, object]:
    if obj.collision_proxy is not None:
        return obj.collision_proxy
    if obj.shape == "sphere":
        return {"shape": "sphere", "radius": obj.size["radius"]}
    if obj.shape == "box":
        return {"shape": "box", "hx": obj.size["hx"], "hy": obj.size["hy"], "hz": obj.size["hz"]}
    if obj.shape == "cylinder":
        return {"shape": "cylinder", "radius": obj.size["radius"], "height": obj.size["height"]}
    if obj.shape == "capsule":
        return {"shape": "capsule", "radius": obj.size["radius"], "height": obj.size["height"]}
    raise ValueError(f"no collision proxy for shape={obj.shape}")


def _create_collision_shape(proxy: Dict[str, object]) -> int:
    shape = proxy["shape"]
    if shape == "sphere":
        return p.createCollisionShape(p.GEOM_SPHERE, radius=float(proxy["radius"]))
    if shape == "box":
        return p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[float(proxy["hx"]), float(proxy["hy"]), float(proxy["hz"])],
        )
    if shape == "cylinder":
        return p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=float(proxy["radius"]),
            height=float(proxy["height"]),
        )
    if shape == "capsule":
        return p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=float(proxy["radius"]),
            height=float(proxy["height"]),
        )
    raise ValueError(f"unsupported collision proxy: {shape}")


def _floor_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    verts = mesh.vertices
    colors = np.zeros((len(verts), 3), dtype=np.float32)
    rng = np.random.default_rng(13)
    plank_width = 0.55
    for idx, vert in enumerate(verts):
        plank_id = int((vert[0] + 7.0) / plank_width)
        base = np.array([0.50, 0.37, 0.23], dtype=np.float32)
        base += rng.uniform(-0.04, 0.04, 3).astype(np.float32)
        seam_pos = -7.0 + (plank_id + 1) * plank_width
        if abs(vert[0] - seam_pos) < 0.025:
            base *= 0.6
        colors[idx] = np.clip(base, 0.0, 1.0)
    return colors


def _wall_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    rng = np.random.default_rng(17)
    base = np.tile(np.array([0.52, 0.51, 0.48], dtype=np.float32), (len(mesh.vertices), 1))
    base += rng.uniform(-0.02, 0.02, base.shape).astype(np.float32)
    return np.clip(base, 0.0, 1.0)


class PreviewRenderer:
    def __init__(self) -> None:
        self.scene = pyrender.Scene(bg_color=[0.56, 0.53, 0.49], ambient_light=[0.05, 0.05, 0.05])
        floor = trimesh.creation.box(extents=[14.0, 12.0, 0.04])
        floor.visual.vertex_colors = _floor_colors(floor)
        self.scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False), pose=_tr(0.0, 2.0, -0.04))

        wall = trimesh.creation.box(extents=[14.0, 0.03, 3.6])
        wall.visual.vertex_colors = _wall_colors(wall)
        self.scene.add(pyrender.Mesh.from_trimesh(wall, smooth=False), pose=_tr(0.0, 5.0, 1.8))

        left_fill = pyrender.SpotLight(color=[1.0, 0.96, 0.92], intensity=55.0, innerConeAngle=0.4, outerConeAngle=1.0)
        right_fill = pyrender.SpotLight(color=[0.92, 0.96, 1.0], intensity=36.0, innerConeAngle=0.45, outerConeAngle=1.0)
        self.scene.add(left_fill, pose=_look_at(np.array([-1.7, -1.5, 2.6]), np.array([0.2, 0.4, 0.3]), np.array([0, 0, 1])))
        self.scene.add(right_fill, pose=_look_at(np.array([1.9, -0.8, 2.2]), np.array([-0.1, 0.8, 0.4]), np.array([0, 0, 1])))

        camera = pyrender.PerspectiveCamera(yfov=np.radians(50.0), aspectRatio=IMG_W / IMG_H)
        self.scene.add(camera, pose=_look_at(CAM_EYE, CAM_TARGET, CAM_UP))
        self.renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)
        self.nodes: Dict[str, pyrender.Node] = {}

    def add_object(self, obj: ObjectSpec) -> None:
        mesh = _load_render_mesh(obj)
        node = self.scene.add(
            pyrender.Mesh.from_trimesh(mesh, smooth=True),
            pose=_tr(obj.position[0], obj.position[1], obj.position[2]),
        )
        self.nodes[obj.name] = node

    def update_pose(self, name: str, pos: List[float], quat: List[float]) -> None:
        self.scene.set_pose(self.nodes[name], pose=_pb_pose(pos, quat))

    def render(self) -> np.ndarray:
        color, _ = self.renderer.render(self.scene, flags=RenderFlags.SHADOWS_SPOT)
        return color

    def cleanup(self) -> None:
        self.renderer.delete()


def primitive(
    *,
    name: str,
    shape: str,
    color: List[float],
    position: List[float],
    size: Dict[str, float],
    mass: float = 1.0,
    dynamic: bool = True,
    texture_style: str = "solid",
    restitution: float = 0.55,
    friction: float = 0.55,
    orientation_euler_deg: List[float] | None = None,
    linear_velocity: List[float] | None = None,
    angular_velocity: List[float] | None = None,
    role: str = "dynamic",
) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        shape=shape,
        color=color,
        mass=mass,
        position=position,
        size=size,
        dynamic=dynamic,
        restitution=restitution,
        friction=friction,
        orientation_euler_deg=orientation_euler_deg or [0.0, 0.0, 0.0],
        linear_velocity=linear_velocity or [0.0, 0.0, 0.0],
        angular_velocity=angular_velocity or [0.0, 0.0, 0.0],
        role=role,
        texture_style=texture_style,
    )


def textured_mesh(
    *,
    name: str,
    asset_key: str,
    position: List[float],
    target_extents: List[float],
    collision_proxy: Dict[str, object],
    mass: float = 1.0,
    dynamic: bool = True,
    color: List[float] | None = None,
    restitution: float = 0.45,
    friction: float = 0.55,
    orientation_euler_deg: List[float] | None = None,
    mesh_euler_deg: List[float] | None = None,
    linear_velocity: List[float] | None = None,
    angular_velocity: List[float] | None = None,
    role: str = "dynamic",
) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        shape="mesh",
        color=color or [0.75, 0.75, 0.75],
        mass=mass,
        position=position,
        size={"fit_x": target_extents[0], "fit_y": target_extents[1], "fit_z": target_extents[2]},
        dynamic=dynamic,
        restitution=restitution,
        friction=friction,
        orientation_euler_deg=orientation_euler_deg or [0.0, 0.0, 0.0],
        linear_velocity=linear_velocity or [0.0, 0.0, 0.0],
        angular_velocity=angular_velocity or [0.0, 0.0, 0.0],
        role=role,
        render_mode="mesh",
        texture_style="mesh_texture",
        mesh_path=str(LOCAL_ASSETS[asset_key]),
        mesh_target_extents=target_extents,
        mesh_euler_deg=mesh_euler_deg or [0.0, 0.0, 0.0],
        collision_proxy=collision_proxy,
    )


def build_preview_scenarios() -> List[ScenarioSpec]:
    scenarios = [
        ScenarioSpec(
            key="rigid_f1_avocado_rollin",
            family="F1 单物体运动",
            title="牛油果连续入镜并落地滚动",
            description="物体在录制开始前已经存在于相机外，通过 pre-roll 连续进入画面，并在可见段中完成明显落地后滚动。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.92,
            seed=1101,
            pre_roll_s=0.28,
            objects=[
                textured_mesh(
                    name="avocado",
                    asset_key="avocado",
                    position=[-2.15, -0.12, 0.75],
                    target_extents=[0.26, 0.18, 0.18],
                    collision_proxy={"shape": "sphere", "radius": 0.15},
                    restitution=0.18,
                    friction=0.92,
                    mesh_euler_deg=[-90.0, 0.0, 0.0],
                    linear_velocity=[2.95, 0.05, -0.25],
                    angular_velocity=[0.0, 11.0, 0.0],
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f1_textured_wheel",
            family="F1 单物体运动",
            title="复杂轮胎滚动",
            description="使用本地复杂 mesh 轮胎进行渲染，物理侧仍使用稳定的圆柱代理碰撞。",
            gravity=EARTH_GRAVITY,
            floor_friction=1.05,
            seed=1102,
            pre_roll_s=0.80,
            objects=[
                textured_mesh(
                    name="wheel",
                    asset_key="fancy_wheel",
                    position=[-3.10, -0.34, 0.27],
                    target_extents=[0.22, 0.54, 0.54],
                    collision_proxy={"shape": "cylinder", "radius": 0.27, "height": 0.16},
                    restitution=0.14,
                    friction=1.05,
                    orientation_euler_deg=[90.0, 0.0, 0.0],
                    mesh_euler_deg=[0.0, 0.0, 0.0],
                    linear_velocity=[2.95, 0.03, 0.0],
                    angular_velocity=[12.0, 0.0, 0.0],
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f2_duck_hits_crate",
            family="F2 双体交互",
            title="小鸭撞木箱",
            description="复杂 mesh 动态物体与木纹箱体碰撞，检查 mesh 渲染下的身份保持和碰撞后速度传递。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.74,
            seed=2101,
            pre_roll_s=0.45,
            objects=[
                textured_mesh(
                    name="duck",
                    asset_key="duck",
                    position=[-2.40, -0.10, 0.22],
                    target_extents=[0.28, 0.22, 0.30],
                    collision_proxy={"shape": "box", "hx": 0.13, "hy": 0.10, "hz": 0.15},
                    restitution=0.34,
                    friction=0.58,
                    linear_velocity=[3.45, 0.08, 0.20],
                    angular_velocity=[0.0, 0.0, 1.6],
                ),
                primitive(
                    name="crate",
                    shape="box",
                    color=[0.49, 0.33, 0.18],
                    position=[0.20, -0.02, 0.22],
                    size={"hx": 0.23, "hy": 0.20, "hz": 0.22},
                    mass=1.6,
                    texture_style="wood",
                    restitution=0.20,
                    friction=0.78,
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f2_pizza_glancing",
            family="F2 双体交互",
            title="披萨盘擦碰金属块",
            description="贴图 mesh 与程序金属材质方块发生斜向擦碰，用于检查复杂轮廓物体的偏转。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.52,
            seed=2102,
            pre_roll_s=0.12,
            objects=[
                textured_mesh(
                    name="pizza",
                    asset_key="pizza",
                    position=[-1.65, 0.48, 0.12],
                    target_extents=[0.44, 0.44, 0.08],
                    collision_proxy={"shape": "cylinder", "radius": 0.21, "height": 0.06},
                    restitution=0.16,
                    friction=0.58,
                    mesh_euler_deg=[-90.0, 0.0, 0.0],
                    linear_velocity=[4.65, -0.22, 0.02],
                    angular_velocity=[0.0, 0.0, 6.2],
                ),
                primitive(
                    name="metal_cube",
                    shape="box",
                    color=[0.55, 0.59, 0.64],
                    position=[0.45, 0.20, 0.22],
                    size={"hx": 0.22, "hy": 0.22, "hz": 0.22},
                    mass=1.2,
                    texture_style="metal",
                    restitution=0.14,
                    friction=0.66,
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f3_avocado_duck_chain",
            family="F3 多体连锁",
            title="牛油果触发小鸭再推动木块",
            description="链式传播中同时混入贴图 mesh 和程序木纹物体，检查多物体因果传递。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.88,
            seed=3101,
            pre_roll_s=0.55,
            objects=[
                textured_mesh(
                    name="lead_avocado",
                    asset_key="avocado",
                    position=[-2.50, -0.18, 0.16],
                    target_extents=[0.24, 0.17, 0.17],
                    collision_proxy={"shape": "sphere", "radius": 0.15},
                    restitution=0.18,
                    friction=0.90,
                    mesh_euler_deg=[-90.0, 0.0, 0.0],
                    linear_velocity=[4.10, 0.0, 0.0],
                    angular_velocity=[0.0, 10.5, 0.0],
                ),
                textured_mesh(
                    name="duck_mid",
                    asset_key="duck",
                    position=[-0.45, -0.18, 0.22],
                    target_extents=[0.27, 0.22, 0.30],
                    collision_proxy={"shape": "box", "hx": 0.13, "hy": 0.10, "hz": 0.15},
                    restitution=0.28,
                    friction=0.62,
                ),
                primitive(
                    name="end_block",
                    shape="box",
                    color=[0.46, 0.30, 0.16],
                    position=[0.62, -0.18, 0.24],
                    size={"hx": 0.20, "hy": 0.18, "hz": 0.24},
                    mass=1.0,
                    texture_style="wood",
                    restitution=0.18,
                    friction=0.76,
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f3_bunny_wheel_chain",
            family="F3 多体连锁",
            title="轮胎触发兔子和木柱",
            description="多体链式传播中加入更高轮廓复杂度的兔子 mesh，观察倒伏和后续接触。",
            gravity=EARTH_GRAVITY,
            floor_friction=1.00,
            seed=3102,
            pre_roll_s=0.65,
            objects=[
                textured_mesh(
                    name="lead_wheel",
                    asset_key="fancy_wheel",
                    position=[-2.85, 0.14, 0.27],
                    target_extents=[0.22, 0.54, 0.54],
                    collision_proxy={"shape": "cylinder", "radius": 0.27, "height": 0.16},
                    restitution=0.10,
                    friction=1.00,
                    orientation_euler_deg=[90.0, 0.0, 0.0],
                    linear_velocity=[3.30, 0.0, 0.0],
                    angular_velocity=[12.0, 0.0, 0.0],
                ),
                textured_mesh(
                    name="bunny_mid",
                    asset_key="bunny",
                    position=[-0.45, 0.14, 0.21],
                    target_extents=[0.24, 0.18, 0.24],
                    collision_proxy={"shape": "box", "hx": 0.10, "hy": 0.08, "hz": 0.12},
                    restitution=0.08,
                    friction=0.70,
                    linear_velocity=[0.0, 0.0, 0.0],
                ),
                primitive(
                    name="wood_post",
                    shape="box",
                    color=[0.57, 0.39, 0.21],
                    position=[0.52, 0.14, 0.29],
                    size={"hx": 0.09, "hy": 0.18, "hz": 0.29},
                    mass=0.75,
                    texture_style="wood",
                    restitution=0.10,
                    friction=0.80,
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f4_ball_behind_armchair",
            family="F4 遮挡与重现",
            title="球体经过扶手椅后方",
            description="使用贴图扶手椅作为静态遮挡物，动态球体在 pre-roll 后从画外连续入镜并完成遮挡-重现。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.68,
            seed=4101,
            pre_roll_s=0.85,
            objects=[
                primitive(
                    name="moving_ball",
                    shape="sphere",
                    color=[0.86, 0.29, 0.20],
                    position=[-3.05, 0.76, 0.18],
                    size={"radius": 0.18},
                    mass=1.0,
                    texture_style="rubber",
                    restitution=0.68,
                    friction=0.42,
                    linear_velocity=[3.70, 0.0, 0.0],
                    angular_velocity=[0.0, 6.0, 0.0],
                ),
                textured_mesh(
                    name="armchair_occ",
                    asset_key="armchair",
                    position=[0.0, -0.08, 0.42],
                    target_extents=[0.95, 0.72, 0.82],
                    collision_proxy={"shape": "box", "hx": 0.32, "hy": 0.28, "hz": 0.42},
                    mass=0.0,
                    dynamic=False,
                    restitution=0.05,
                    friction=0.85,
                    role="occluder",
                    mesh_euler_deg=[-90.0, 0.0, 0.0],
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f4_duck_cross_chair",
            family="F4 遮挡与重现",
            title="小鸭与牛油果在扶手椅后方交叉",
            description="两个不同 mesh 物体在大型静态物体后方交叉，主要看 identity 是否稳定。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.72,
            seed=4102,
            pre_roll_s=0.75,
            objects=[
                textured_mesh(
                    name="duck_cross",
                    asset_key="duck",
                    position=[-2.55, 0.62, 0.22],
                    target_extents=[0.27, 0.22, 0.30],
                    collision_proxy={"shape": "box", "hx": 0.13, "hy": 0.10, "hz": 0.15},
                    restitution=0.24,
                    friction=0.58,
                    linear_velocity=[3.25, 0.0, 0.0],
                ),
                textured_mesh(
                    name="avocado_cross",
                    asset_key="avocado",
                    position=[2.65, 0.90, 0.16],
                    target_extents=[0.24, 0.17, 0.17],
                    collision_proxy={"shape": "sphere", "radius": 0.15},
                    restitution=0.14,
                    friction=0.88,
                    mesh_euler_deg=[-90.0, 0.0, 0.0],
                    linear_velocity=[-2.85, 0.0, 0.0],
                    angular_velocity=[0.0, -8.0, 0.0],
                ),
                textured_mesh(
                    name="chair_occ",
                    asset_key="armchair",
                    position=[0.0, -0.06, 0.42],
                    target_extents=[0.95, 0.72, 0.82],
                    collision_proxy={"shape": "box", "hx": 0.32, "hy": 0.28, "hz": 0.42},
                    mass=0.0,
                    dynamic=False,
                    restitution=0.05,
                    friction=0.85,
                    role="occluder",
                    mesh_euler_deg=[-90.0, 0.0, 0.0],
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f5_avocado_into_bowl",
            family="F5 支撑与跌落",
            title="牛油果跌入碗中",
            description="用 bowl mesh 做支撑容器，顶部跌落的牛油果在复杂形状附近发生离散事件切换。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.84,
            seed=5101,
            pre_roll_s=0.10,
            objects=[
                textured_mesh(
                    name="drop_avocado",
                    asset_key="avocado",
                    position=[-0.18, 0.0, 1.30],
                    target_extents=[0.24, 0.17, 0.17],
                    collision_proxy={"shape": "sphere", "radius": 0.15},
                    restitution=0.26,
                    friction=0.84,
                    mesh_euler_deg=[-90.0, 0.0, 0.0],
                    linear_velocity=[0.22, 0.0, -0.05],
                ),
                textured_mesh(
                    name="bowl_support",
                    asset_key="bowl",
                    position=[0.10, 0.0, 0.07],
                    target_extents=[0.62, 0.62, 0.22],
                    collision_proxy={"shape": "cylinder", "radius": 0.28, "height": 0.12},
                    mass=0.0,
                    dynamic=False,
                    color=[0.83, 0.80, 0.76],
                    restitution=0.05,
                    friction=0.92,
                    role="support",
                ),
            ],
        ),
        ScenarioSpec(
            key="rigid_f5_bunny_topple",
            family="F5 支撑与跌落",
            title="兔子在窄台上失稳倒下",
            description="复杂 mesh 放在窄支撑台上，通过轻微初始偏置和角速度触发失稳倒伏。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.78,
            seed=5102,
            pre_roll_s=0.05,
            objects=[
                primitive(
                    name="pedestal",
                    shape="box",
                    color=[0.45, 0.32, 0.18],
                    position=[0.0, 0.0, 0.12],
                    size={"hx": 0.13, "hy": 0.13, "hz": 0.12},
                    mass=0.0,
                    dynamic=False,
                    texture_style="wood",
                    role="support",
                    friction=0.82,
                ),
                textured_mesh(
                    name="topple_bunny",
                    asset_key="bunny",
                    position=[0.17, 0.0, 0.33],
                    target_extents=[0.28, 0.20, 0.28],
                    collision_proxy={"shape": "box", "hx": 0.10, "hy": 0.08, "hz": 0.14},
                    restitution=0.08,
                    friction=0.64,
                    orientation_euler_deg=[0.0, 8.0, 16.0],
                    linear_velocity=[0.03, 0.0, 0.0],
                    angular_velocity=[0.0, 0.45, 0.18],
                ),
            ],
        ),
    ]
    return scenarios


def _overlay_text(frame_bgr: np.ndarray, lines: List[str]) -> None:
    for idx, line in enumerate(lines):
        y = 30 + idx * 24
        cv2.putText(frame_bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (244, 242, 237), 1, cv2.LINE_AA)


def run_scenario(renderer: PreviewRenderer, scenario: ScenarioSpec) -> dict:
    np.random.seed(scenario.seed)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")
    p.setGravity(0.0, 0.0, -scenario.gravity)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0 / SIM_HZ, numSolverIterations=120, numSubSteps=1)
    p.changeDynamics(
        plane_id,
        -1,
        lateralFriction=scenario.floor_friction,
        restitution=0.02,
        activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
    )
    if abs(float(scenario.gravity) - EARTH_GRAVITY) > 1e-6:
        raise ValueError(f"rigid preview requires Earth gravity {EARTH_GRAVITY}, got {scenario.gravity}")

    bodies: List[dict] = []
    for obj in scenario.objects:
        renderer.add_object(obj)
        quat = _quat_from_euler_deg(obj.orientation_euler_deg)
        body_id = p.createMultiBody(
            baseMass=obj.mass if obj.dynamic else 0.0,
            baseCollisionShapeIndex=_create_collision_shape(_collision_proxy_for_object(obj)),
            basePosition=obj.position,
            baseOrientation=quat,
        )
        p.changeDynamics(
            body_id,
            -1,
            restitution=obj.restitution,
            lateralFriction=obj.friction,
            linearDamping=obj.linear_damping,
            angularDamping=obj.angular_damping,
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        )
        p.resetBaseVelocity(body_id, linearVelocity=obj.linear_velocity, angularVelocity=obj.angular_velocity)
        bodies.append({"spec": obj, "body_id": body_id})

    total_steps = int(SIM_DURATION * SIM_HZ)
    pre_roll_steps = int(scenario.pre_roll_s * SIM_HZ)
    max_frames = math.ceil(total_steps / RECORD_EVERY)
    object_count = len(bodies)
    positions = np.zeros((max_frames, object_count, 3), dtype=np.float32)
    quats = np.zeros((max_frames, object_count, 4), dtype=np.float32)
    linvels = np.zeros((max_frames, object_count, 3), dtype=np.float32)
    angvels = np.zeros((max_frames, object_count, 3), dtype=np.float32)
    frames: List[np.ndarray] = []

    frame_index = 0
    for step in range(pre_roll_steps + total_steps):
        p.stepSimulation()
        if step < pre_roll_steps:
            continue
        record_step = step - pre_roll_steps
        if record_step % RECORD_EVERY != 0:
            continue
        visible_time = record_step / SIM_HZ
        for obj_index, body in enumerate(bodies):
            body_id = body["body_id"]
            pos, quat = p.getBasePositionAndOrientation(body_id)
            linvel, angvel = p.getBaseVelocity(body_id)
            positions[frame_index, obj_index] = np.asarray(pos, dtype=np.float32)
            quats[frame_index, obj_index] = np.asarray(quat, dtype=np.float32)
            linvels[frame_index, obj_index] = np.asarray(linvel, dtype=np.float32)
            angvels[frame_index, obj_index] = np.asarray(angvel, dtype=np.float32)
            renderer.update_pose(body["spec"].name, pos, quat)

        frame_rgb = renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _overlay_text(
            frame_bgr,
            [
                f"{scenario.family} | {scenario.title}",
                f"{scenario.key} | t={visible_time:0.2f}s | pre-roll={scenario.pre_roll_s:0.2f}s | floor_mu={scenario.floor_friction:0.2f} | objects={object_count}",
            ],
        )
        frames.append(frame_bgr)
        frame_index += 1

    for body in bodies:
        p.removeBody(body["body_id"])

    output_mp4 = VIDEO_DIR / f"{scenario.key}.mp4"
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H))
    for frame in frames:
        writer.write(frame)
    writer.release()

    output_npz = META_DIR / f"{scenario.key}_states.npz"
    np.savez_compressed(
        output_npz,
        positions=positions[:frame_index],
        quats=quats[:frame_index],
        linear_velocities=linvels[:frame_index],
        angular_velocities=angvels[:frame_index],
        frame_times=np.arange(frame_index, dtype=np.float32) / FPS,
        object_names=np.asarray([body["spec"].name for body in bodies]),
        object_roles=np.asarray([body["spec"].role for body in bodies]),
    )

    meta = {
        "key": scenario.key,
        "family": scenario.family,
        "title": scenario.title,
        "description": scenario.description,
        "seed": scenario.seed,
        "gravity": scenario.gravity,
        "sim_type": scenario.sim_type,
        "floor_friction": scenario.floor_friction,
        "spawn_policy": "no mid-scene spawn; pre-roll allowed for off-screen continuity",
        "video": str(output_mp4),
        "states": str(output_npz),
        "fps": FPS,
        "resolution": [IMG_W, IMG_H],
        "duration_s": len(frames) / FPS,
        "pre_roll_s": scenario.pre_roll_s,
        "objects": [asdict(obj) for obj in scenario.objects],
    }
    (META_DIR / f"{scenario.key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def generate_html(report_items: List[dict], port: int) -> Path:
    by_family: Dict[str, List[dict]] = {}
    for item in report_items:
        by_family.setdefault(item["family"], []).append(item)

    sections: List[str] = []
    for family, items in by_family.items():
        cards: List[str] = []
        for item in items:
            object_badges = []
            for obj in item["objects"]:
                role = obj["role"]
                role_text = "遮挡物" if role == "occluder" else ("支撑物" if role == "support" else "动态物体")
                render_text = "mesh" if obj["render_mode"] == "mesh" else obj["shape"]
                object_badges.append(
                    f"<span class='badge'>{html.escape(obj['name'])}: {html.escape(render_text)} / {role_text}</span>"
                )
            cards.append(
                f"""
                <article class="card">
                  <video controls autoplay loop muted playsinline preload="metadata">
                    <source src="videos/{html.escape(item['key'])}.mp4" type="video/mp4">
                  </video>
                  <div class="content">
                    <div class="eyebrow">{html.escape(item['key'])}</div>
                    <h3>{html.escape(item['title'])}</h3>
                    <p class="desc">{html.escape(item['description'])}</p>
                    <div class="meta-line">
                      <span>type={html.escape(item['sim_type'])}</span>
                      <span>g={item['gravity']}</span>
                      <span>floor_mu={item['floor_friction']}</span>
                      <span>pre-roll={item['pre_roll_s']}s</span>
                      <span>objects={len(item['objects'])}</span>
                    </div>
                    <div class="badge-row">{''.join(object_badges)}</div>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="family">
              <div class="family-head">
                <h2>{html.escape(family)}</h2>
                <p>{len(items)} 个代表 case</p>
              </div>
              <div class="grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rigid Sim ObjState Preview v2</title>
  <style>
    :root {{
      --bg: #181612;
      --panel: rgba(32, 29, 24, 0.92);
      --line: rgba(255, 255, 255, 0.10);
      --text: #f2ede4;
      --muted: #b8ae9b;
      --accent: #de7f39;
      --accent-2: #73a8c7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(222, 127, 57, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(115, 168, 199, 0.16), transparent 28%),
        linear-gradient(180deg, #171410 0%, #11100d 100%);
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: 0.01em;
    }}
    .lead {{
      margin: 0;
      max-width: 1020px;
      color: var(--muted);
      line-height: 1.75;
      font-size: 15px;
    }}
    .hero {{
      margin-bottom: 28px;
      padding: 22px 24px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      backdrop-filter: blur(8px);
    }}
    .hero-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    .pill {{
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      font-size: 13px;
      color: var(--text);
    }}
    .family {{
      margin-top: 26px;
    }}
    .family-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .family-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .family-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      backdrop-filter: blur(8px);
      box-shadow: 0 18px 30px rgba(0, 0, 0, 0.18);
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
    }}
    .content {{
      padding: 16px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .desc {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.65;
      font-size: 14px;
    }}
    .meta-line {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      color: #ddd4c6;
      font-size: 13px;
    }}
    .badge-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.03);
      color: #e8dfd0;
      font-size: 12px;
    }}
    .footer {{
      margin-top: 30px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    a {{ color: var(--accent-2); }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Rigid 仿真 Object-State 预览集 v3</h1>
      <p class="lead">
        这一版只包含 rigid-body 场景，不和 MPM 混合。所有动态物体都在录制开始前已经存在，
        通过 pre-roll 从画外连续入镜，避免“凭空出现”。同时引入了本地 textured mesh 资源和 render mesh / collision proxy 分离，
        让外观更复杂、碰撞仍然稳定。当前 rigid 预览统一采用地球重力 `9.81 m/s²`，并把地面摩擦系数作为显式场景参数纳入变化范围。
      </p>
      <div class="hero-row">
        <span class="pill">simulation type = rigid</span>
        <span class="pill">分辨率 {IMG_W}×{IMG_H}</span>
        <span class="pill">FPS {FPS}</span>
        <span class="pill">{len(report_items)} 个 preview case</span>
        <span class="pill">输出目录 {html.escape(str(OUTPUT_ROOT))}</span>
      </div>
    </section>
    {''.join(sections)}
    <div class="footer">
      本地访问地址：
      <a href="http://127.0.0.1:{port}">http://127.0.0.1:{port}</a>
    </div>
  </div>
</body>
</html>
"""
    html_path = OUTPUT_ROOT / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


def start_server(port: int) -> int:
    log_path = OUTPUT_ROOT / f"http_{port}.log"
    pid_path = OUTPUT_ROOT / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            return pid
        except Exception:
            pid_path.unlink(missing_ok=True)
    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(OUTPUT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a rigid-body simulation preview gallery.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    global OUTPUT_ROOT, VIDEO_DIR, META_DIR
    OUTPUT_ROOT = args.output_root
    VIDEO_DIR = OUTPUT_ROOT / "videos"
    META_DIR = OUTPUT_ROOT / "meta"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for path in [VIDEO_DIR, META_DIR]:
            if path.exists():
                shutil.rmtree(path)
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        META_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = build_preview_scenarios()
    manifest: List[dict]
    if not args.serve_only:
        p.connect(p.DIRECT)
        try:
            manifest = []
            for scenario in scenarios:
                renderer = PreviewRenderer()
                try:
                    print(f"[generate] {scenario.key} :: {scenario.title}")
                    manifest.append(run_scenario(renderer, scenario))
                finally:
                    renderer.cleanup()
        finally:
            p.disconnect()
        (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))

    html_path = generate_html(manifest, args.port)
    pid = start_server(args.port)
    print(f"gallery: {html_path}")
    print(f"server: http://127.0.0.1:{args.port}")
    print(f"pid: {pid}")


if __name__ == "__main__":
    main()
