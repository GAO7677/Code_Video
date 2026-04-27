#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hamiltonian-Rigid-Dynamics dataset generator built on Genesis.

This script generates a rigid-body dataset for symplectic-aligned video diffusion
training. Each scene contains four simultaneous rigid-body phenomena:

1. Collision between a moving striker and a target block
2. Rolling sphere on an inclined plane
3. Projectile motion with floor impact
4. Single pendulum oscillation

Per scene, the script writes one sample folder containing:

- rgb PNG frames and `videos/rgb.mp4`
- normalized depth PNG frames and `videos/depth.mp4`
- `physics/trajectory.npy` with shape [49, num_objects, 6]
- `physics/properties.json` with mass / restitution / friction / impulse vector
- `physics/collision_events.json` with impact frame indices and momentum jumps

The generator uses a fixed camera viewpoint by default and rejects samples when:

- any tracked body leaves the camera frustum,
- any tracked state becomes invalid,
- bodies fly outside a safe world-space envelope,
- or contacts exhibit excessive penetration.

Example:

/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/try3_SLD.py \
  --num-scenes 10 \
  --output-dir /home/gaoya/Code_Video/Code_data/hamiltonian_rigid_h5
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import numpy as np
except Exception:
    np = None

os.environ.setdefault("MUJOCO_GL", "0")
os.environ.setdefault("TQDM_DISABLE", "1")
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

gs = None


DATASET_NAME = "Hamiltonian-Rigid-Dynamics"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "hamiltonian_rigid_dataset"
DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / "_hamiltonian_rigid_assets"
DEFAULT_FRAMES = 49
DEFAULT_RESOLUTION = 640
DEFAULT_BACKGROUND_COLOR = (0.94, 0.95, 0.93)
DEFAULT_AMBIENT_LIGHT = (0.34, 0.34, 0.32)

GENESIS_INIT_LOCK = threading.Lock()
GENESIS_INITIALIZED = False
GENESIS_BACKEND_USED = "none"


class SceneGenerationError(RuntimeError):
    """Raised when a sampled scene is invalid and must be retried."""


@dataclass
class ObjectPhysicalSpec:
    name: str
    phenomenon: str
    mass: float
    restitution: float
    friction: float
    impulse_vector: np.ndarray


@dataclass
class SceneParams:
    scene_index: int
    object_specs: Dict[str, ObjectPhysicalSpec]
    collision_sphere_radius: float
    collision_box_size: np.ndarray
    rolling_sphere_radius: float
    projectile_sphere_radius: float
    ramp_angle_deg: float
    ramp_center: np.ndarray
    ramp_size: np.ndarray
    pendulum_length: float
    pendulum_bob_radius: float
    pendulum_angle_deg: float
    pendulum_anchor: np.ndarray


@dataclass
class DynamicBody:
    name: str
    phenomenon: str
    entity: Any
    link_local_idx: int
    mass: float
    restitution: float
    friction: float
    impulse_vector: np.ndarray


@dataclass
class ContactMonitor:
    pair_name: str
    entity_a: Any
    entity_b: Any
    object_idx_a: int
    object_idx_b: int
    partner_name: str


@dataclass
class SceneBundle:
    scene: Any
    camera: Any
    floor: Any
    ramp: Any
    dynamic_bodies: List[DynamicBody]
    contact_monitors: List[ContactMonitor]


def _get_default_genesis_repo() -> Path:
    return Path(__file__).resolve().parents[1] / "Genesis_main"


def ensure_python_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if imageio is None:
        missing.append("imageio")
    if missing:
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + ". Please install them in the runtime environment before running this script."
        )


def _import_genesis():
    global gs
    if gs is not None:
        return gs

    genesis_repo = _get_default_genesis_repo()
    if genesis_repo.exists():
        sys.path.insert(0, str(genesis_repo))

    import genesis as _gs  # type: ignore

    gs = _gs
    return gs


def ensure_genesis_initialized(prefer_gpu: bool = True) -> str:
    global GENESIS_INITIALIZED, GENESIS_BACKEND_USED

    gs_mod = _import_genesis()
    if GENESIS_INITIALIZED:
        return GENESIS_BACKEND_USED

    with GENESIS_INIT_LOCK:
        if GENESIS_INITIALIZED:
            return GENESIS_BACKEND_USED

        init_candidates: List[Tuple[str, Any]] = []
        if prefer_gpu and hasattr(gs_mod, "gpu"):
            init_candidates.append(("gpu", gs_mod.gpu))
        if hasattr(gs_mod, "cpu"):
            init_candidates.append(("cpu", gs_mod.cpu))

        last_error = None
        if not init_candidates:
            try:
                gs_mod.init()
                GENESIS_INITIALIZED = True
                GENESIS_BACKEND_USED = "default"
                return GENESIS_BACKEND_USED
            except Exception as exc:
                msg = str(exc).lower()
                if "already" in msg and "initial" in msg:
                    GENESIS_INITIALIZED = True
                    GENESIS_BACKEND_USED = "existing"
                    return GENESIS_BACKEND_USED
                raise

        for backend_name, backend_value in init_candidates:
            try:
                gs_mod.init(backend=backend_value)
                GENESIS_INITIALIZED = True
                device_text = str(getattr(gs_mod, "device", backend_name)).lower()
                if "cpu" in device_text:
                    GENESIS_BACKEND_USED = "cpu"
                elif "cuda" in device_text or "gpu" in device_text:
                    GENESIS_BACKEND_USED = "gpu"
                else:
                    GENESIS_BACKEND_USED = backend_name
                return GENESIS_BACKEND_USED
            except Exception as exc:
                msg = str(exc).lower()
                if "already" in msg and "initial" in msg:
                    GENESIS_INITIALIZED = True
                    GENESIS_BACKEND_USED = backend_name
                    return GENESIS_BACKEND_USED
                last_error = exc

        raise RuntimeError(f"Failed to initialize Genesis: {last_error}") from last_error


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def rgb_to_uint8(rgb: Any) -> np.ndarray:
    array = to_numpy(rgb)
    if array.dtype == np.uint8:
        return array
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.max(initial=0.0) <= 1.0 + 1e-6:
        array = array * 255.0
    return np.clip(np.round(array), 0.0, 255.0).astype(np.uint8)


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
    normalized[valid] = np.clip((array[valid] - near) / (far - near), 0.0, 1.0)
    return normalized[..., None]


def depth_to_uint8(depth_norm: np.ndarray) -> np.ndarray:
    depth_img = np.asarray(depth_norm, dtype=np.float32)
    if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
        depth_img = depth_img[..., 0]
    return np.clip(np.round(depth_img * 255.0), 0.0, 255.0).astype(np.uint8)


def spherical_camera(radius: float, azimuth_deg: float, elevation_deg: float, lookat: np.ndarray) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    xy = radius * math.cos(elevation)
    return lookat + np.array(
        [
            xy * math.cos(azimuth),
            xy * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )


def make_rigid_material(gs_mod: Any, *, rho: float, friction: float, restitution: Optional[float] = None):
    kwargs = {
        "rho": float(rho),
        "friction": float(np.clip(friction, 1e-2, 5.0)),
    }
    if restitution is not None:
        try:
            return gs_mod.materials.Rigid(
                restitution=float(np.clip(restitution, 0.0, 1.0)),
                **kwargs,
            )
        except TypeError:
            pass
    return gs_mod.materials.Rigid(**kwargs)


def camera_basis(pos: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = lookat - pos
    norm_forward = np.linalg.norm(forward)
    if norm_forward < 1e-8:
        raise SceneGenerationError("Camera forward vector is degenerate.")
    forward = forward / norm_forward

    right = np.cross(forward, up)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-8:
        raise SceneGenerationError("Camera right vector is degenerate.")
    right = right / norm_right
    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-8)
    return right, true_up, forward


def points_in_camera_frustum(
    points: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_lookat: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    near: float,
    far: float,
    margin: float = 0.92,
) -> np.ndarray:
    right, true_up, forward = camera_basis(camera_pos, camera_lookat, camera_up)
    rel = points - camera_pos[None, :]
    x_cam = rel @ right
    y_cam = rel @ true_up
    z_cam = rel @ forward

    tan_half = math.tan(math.radians(fov_deg) * 0.5)
    visible = z_cam > near
    visible &= z_cam < far
    visible &= np.abs(x_cam) <= (margin * tan_half * z_cam)
    visible &= np.abs(y_cam) <= (margin * tan_half * z_cam)
    return visible


def interpolate_keyframes(values: Sequence[float], frame_idx: int, total_frames: int) -> float:
    if not values:
        raise ValueError("At least one keyframe value is required.")
    if len(values) == 1 or total_frames <= 1:
        return float(values[0])

    phase = float(frame_idx) / float(total_frames - 1)
    scaled = phase * float(len(values) - 1)
    left = int(math.floor(scaled))
    right = min(left + 1, len(values) - 1)
    alpha = scaled - left
    return float((1.0 - alpha) * float(values[left]) + alpha * float(values[right]))


def write_pendulum_urdf(
    out_path: Path,
    *,
    length: float,
    bob_radius: float,
    bob_mass: float,
    joint_damping: float,
) -> None:
    ensure_dir(out_path.parent)

    support_size = np.array([0.12, 0.12, 0.08], dtype=np.float64)
    inertia_scalar = 2.0 / 5.0 * bob_mass * (bob_radius ** 2)

    robot = ET.Element("robot", name="simple_pendulum")

    base_link = ET.SubElement(robot, "link", name="base")
    base_visual = ET.SubElement(base_link, "visual")
    ET.SubElement(base_visual, "origin", xyz="0 0 0", rpy="0 0 0")
    base_visual_geom = ET.SubElement(base_visual, "geometry")
    ET.SubElement(
        base_visual_geom,
        "box",
        size=f"{support_size[0]:.6f} {support_size[1]:.6f} {support_size[2]:.6f}",
    )
    base_collision = ET.SubElement(base_link, "collision")
    ET.SubElement(base_collision, "origin", xyz="0 0 0", rpy="0 0 0")
    base_collision_geom = ET.SubElement(base_collision, "geometry")
    ET.SubElement(
        base_collision_geom,
        "box",
        size=f"{support_size[0]:.6f} {support_size[1]:.6f} {support_size[2]:.6f}",
    )
    base_inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(base_inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(base_inertial, "mass", value="0.05")
    ET.SubElement(
        base_inertial,
        "inertia",
        ixx="1e-4",
        ixy="0",
        ixz="0",
        iyy="1e-4",
        iyz="0",
        izz="1e-4",
    )

    bob_link = ET.SubElement(robot, "link", name="bob")
    bob_inertial = ET.SubElement(bob_link, "inertial")
    ET.SubElement(bob_inertial, "origin", xyz=f"0 0 {-length:.6f}", rpy="0 0 0")
    ET.SubElement(bob_inertial, "mass", value=f"{bob_mass:.6f}")
    ET.SubElement(
        bob_inertial,
        "inertia",
        ixx=f"{inertia_scalar:.6f}",
        ixy="0",
        ixz="0",
        iyy=f"{inertia_scalar:.6f}",
        iyz="0",
        izz=f"{inertia_scalar:.6f}",
    )

    rod_visual = ET.SubElement(bob_link, "visual", name="rod_visual")
    ET.SubElement(rod_visual, "origin", xyz=f"0 0 {-0.5 * length:.6f}", rpy="0 0 0")
    rod_geometry = ET.SubElement(rod_visual, "geometry")
    ET.SubElement(rod_geometry, "cylinder", radius="0.015", length=f"{length:.6f}")

    bob_visual = ET.SubElement(bob_link, "visual", name="bob_visual")
    ET.SubElement(bob_visual, "origin", xyz=f"0 0 {-length:.6f}", rpy="0 0 0")
    bob_visual_geom = ET.SubElement(bob_visual, "geometry")
    ET.SubElement(bob_visual_geom, "sphere", radius=f"{bob_radius:.6f}")

    bob_collision = ET.SubElement(bob_link, "collision")
    ET.SubElement(bob_collision, "origin", xyz=f"0 0 {-length:.6f}", rpy="0 0 0")
    bob_collision_geom = ET.SubElement(bob_collision, "geometry")
    ET.SubElement(bob_collision_geom, "sphere", radius=f"{bob_radius:.6f}")

    joint = ET.SubElement(robot, "joint", name="pendulum_joint", type="revolute")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="bob")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(joint, "axis", xyz="0 1 0")
    ET.SubElement(joint, "limit", lower="-2.7", upper="2.7", effort="200", velocity="12")
    ET.SubElement(joint, "dynamics", damping=f"{joint_damping:.6f}", friction="0.0")

    tree = ET.ElementTree(robot)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


class HamiltonianRigidDatasetGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.frames = int(args.frames)
        self.resolution = int(args.resolution)
        self.dt = float(args.dt)
        self.substeps = int(args.substeps)
        self.steps_per_frame = int(args.steps_per_frame)
        self.camera_fov = float(args.camera_fov)
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.fixed_camera_pos = np.asarray(args.fixed_camera_pos, dtype=np.float64)
        self.fixed_camera_lookat = np.asarray(args.fixed_camera_lookat, dtype=np.float64)
        self.background_color = tuple(float(x) for x in args.background_color)
        self.ambient_light = tuple(float(x) for x in args.ambient_light)
        self.world_bounds_min = np.array([-3.4, -3.1, -0.3], dtype=np.float64)
        self.world_bounds_max = np.array([3.4, 3.1, 4.2], dtype=np.float64)

    def sample_scene_params(self, scene_index: int, rng: np.random.Generator) -> SceneParams:
        collision_striker_mass = float(rng.uniform(0.5, 5.0))
        collision_target_mass = float(rng.uniform(0.5, 5.0))
        rolling_mass = float(rng.uniform(0.5, 5.0))
        projectile_mass = float(rng.uniform(0.5, 5.0))
        pendulum_mass = float(rng.uniform(0.5, 5.0))

        collision_speed = float(rng.uniform(1.9, 3.1))
        collision_impulse = np.array(
            [
                float(rng.uniform(-0.08, 0.08) * collision_striker_mass),
                collision_speed * collision_striker_mass,
                0.0,
            ],
            dtype=np.float32,
        )

        rolling_speed = float(rng.uniform(0.18, 0.38))
        rolling_impulse = np.array(
            [
                0.0,
                -rolling_speed * rolling_mass,
                0.0,
            ],
            dtype=np.float32,
        )

        projectile_velocity = np.array(
            [
                float(rng.uniform(0.25, 0.90)),
                float(rng.uniform(2.2, 3.0)),
                float(rng.uniform(3.0, 3.8)),
            ],
            dtype=np.float32,
        )
        projectile_impulse = projectile_mass * projectile_velocity

        object_specs = {
            "collision_striker": ObjectPhysicalSpec(
                name="collision_striker",
                phenomenon="collision",
                mass=collision_striker_mass,
                restitution=float(rng.uniform(0.1, 0.9)),
                friction=float(rng.uniform(0.01, 0.5)),
                impulse_vector=collision_impulse,
            ),
            "collision_target": ObjectPhysicalSpec(
                name="collision_target",
                phenomenon="collision",
                mass=collision_target_mass,
                restitution=float(rng.uniform(0.1, 0.9)),
                friction=float(rng.uniform(0.01, 0.5)),
                impulse_vector=np.zeros(3, dtype=np.float32),
            ),
            "rolling_sphere": ObjectPhysicalSpec(
                name="rolling_sphere",
                phenomenon="rolling",
                mass=rolling_mass,
                restitution=float(rng.uniform(0.1, 0.9)),
                friction=float(rng.uniform(0.01, 0.5)),
                impulse_vector=rolling_impulse,
            ),
            "projectile_sphere": ObjectPhysicalSpec(
                name="projectile_sphere",
                phenomenon="projectile",
                mass=projectile_mass,
                restitution=float(rng.uniform(0.1, 0.9)),
                friction=float(rng.uniform(0.01, 0.5)),
                impulse_vector=projectile_impulse.astype(np.float32),
            ),
            "pendulum_bob": ObjectPhysicalSpec(
                name="pendulum_bob",
                phenomenon="pendulum",
                mass=pendulum_mass,
                restitution=float(rng.uniform(0.1, 0.9)),
                friction=float(rng.uniform(0.01, 0.5)),
                impulse_vector=np.zeros(3, dtype=np.float32),
            ),
        }

        collision_sphere_radius = float(rng.uniform(0.15, 0.19))
        collision_box_edge = float(rng.uniform(0.26, 0.34))
        rolling_sphere_radius = float(rng.uniform(0.13, 0.17))
        projectile_sphere_radius = float(rng.uniform(0.13, 0.16))
        pendulum_bob_radius = float(rng.uniform(0.11, 0.15))

        return SceneParams(
            scene_index=scene_index,
            object_specs=object_specs,
            collision_sphere_radius=collision_sphere_radius,
            collision_box_size=np.array([collision_box_edge, collision_box_edge, collision_box_edge], dtype=np.float64),
            rolling_sphere_radius=rolling_sphere_radius,
            projectile_sphere_radius=projectile_sphere_radius,
            ramp_angle_deg=float(rng.uniform(16.0, 26.0)),
            ramp_center=np.array([-1.00, 1.20, 0.34], dtype=np.float64),
            ramp_size=np.array([1.10, 1.50, 0.08], dtype=np.float64),
            pendulum_length=float(rng.uniform(0.92, 1.18)),
            pendulum_bob_radius=pendulum_bob_radius,
            pendulum_angle_deg=float(rng.uniform(20.0, 34.0)) * float(rng.choice([-1.0, 1.0])),
            pendulum_anchor=np.array([1.62, 0.92, 1.92], dtype=np.float64),
        )

    def build_scene(self, params: SceneParams) -> SceneBundle:
        gs_mod = _import_genesis()

        initial_lookat = self.fixed_camera_lookat.copy()
        initial_camera_pos = self.fixed_camera_pos.copy()

        scene = gs_mod.Scene(
            sim_options=gs_mod.options.SimOptions(
                dt=self.dt,
                substeps=self.substeps,
                gravity=(0.0, 0.0, -9.81),
                floor_height=0.0,
            ),
            rigid_options=gs_mod.options.RigidOptions(
                integrator=gs_mod.integrator.implicitfast,
                constraint_solver=gs_mod.constraint_solver.Newton,
                iterations=int(self.args.rigid_iterations),
                ls_iterations=int(self.args.rigid_ls_iterations),
                noslip_iterations=int(self.args.rigid_noslip_iterations),
                tolerance=float(self.args.rigid_tolerance),
                ls_tolerance=float(self.args.rigid_ls_tolerance),
                max_collision_pairs=int(self.args.max_collision_pairs),
                constraint_timeconst=float(self.args.constraint_timeconst),
                use_hibernation=False,
                enable_self_collision=False,
                enable_adjacent_collision=False,
                enable_multi_contact=True,
                use_gjk_collision=True,
            ),
            viewer_options=gs_mod.options.ViewerOptions(
                camera_pos=tuple(initial_camera_pos.tolist()),
                camera_lookat=tuple(initial_lookat.tolist()),
                camera_fov=float(self.camera_fov),
                max_FPS=120,
            ),
            vis_options=gs_mod.options.VisOptions(
                background_color=self.background_color,
                ambient_light=self.ambient_light,
            ),
            show_viewer=bool(self.args.show_viewer),
        )

        low_friction_material = make_rigid_material(gs_mod, rho=1200.0, friction=0.01, restitution=0.05)

        floor = scene.add_entity(
            morph=gs_mod.morphs.Plane(pos=(0.0, 0.0, 0.0), fixed=True),
            material=low_friction_material,
            surface=gs_mod.surfaces.Default(color=(0.88, 0.88, 0.90, 1.0)),
        )

        ramp = scene.add_entity(
            morph=gs_mod.morphs.Box(
                pos=tuple(params.ramp_center.tolist()),
                size=tuple(params.ramp_size.tolist()),
                euler=(params.ramp_angle_deg, 0.0, 0.0),
                fixed=True,
            ),
            material=low_friction_material,
            surface=gs_mod.surfaces.Default(color=(0.72, 0.66, 0.52, 1.0)),
        )

        collision_striker = scene.add_entity(
            morph=gs_mod.morphs.Sphere(
                pos=(-1.92, -1.20, params.collision_sphere_radius + 0.002),
                radius=params.collision_sphere_radius,
            ),
            material=make_rigid_material(
                gs_mod,
                rho=1000.0,
                friction=float(params.object_specs["collision_striker"].friction),
                restitution=float(params.object_specs["collision_striker"].restitution),
            ),
            surface=gs_mod.surfaces.Default(color=(0.92, 0.30, 0.23, 1.0)),
        )

        collision_target = scene.add_entity(
            morph=gs_mod.morphs.Box(
                pos=(-1.92, -0.52, 0.5 * params.collision_box_size[2] + 0.002),
                size=tuple(params.collision_box_size.tolist()),
                euler=(0.0, 0.0, 7.0),
            ),
            material=make_rigid_material(
                gs_mod,
                rho=950.0,
                friction=float(params.object_specs["collision_target"].friction),
                restitution=float(params.object_specs["collision_target"].restitution),
            ),
            surface=gs_mod.surfaces.Default(color=(0.18, 0.56, 0.91, 1.0)),
        )

        high_side_y = params.ramp_center[1] + 0.28
        rolling_top_z = (
            params.ramp_center[2]
            + 0.5 * params.ramp_size[2]
            + 0.28 * math.sin(math.radians(params.ramp_angle_deg))
            + params.rolling_sphere_radius
            + 0.02
        )
        rolling_sphere = scene.add_entity(
            morph=gs_mod.morphs.Sphere(
                pos=(params.ramp_center[0], high_side_y, rolling_top_z),
                radius=params.rolling_sphere_radius,
            ),
            material=make_rigid_material(
                gs_mod,
                rho=1000.0,
                friction=float(params.object_specs["rolling_sphere"].friction),
                restitution=float(params.object_specs["rolling_sphere"].restitution),
            ),
            surface=gs_mod.surfaces.Default(color=(0.17, 0.72, 0.38, 1.0)),
        )

        projectile_sphere = scene.add_entity(
            morph=gs_mod.morphs.Sphere(
                pos=(0.18, -1.55, params.projectile_sphere_radius + 0.02),
                radius=params.projectile_sphere_radius,
            ),
            material=make_rigid_material(
                gs_mod,
                rho=1000.0,
                friction=float(params.object_specs["projectile_sphere"].friction),
                restitution=float(params.object_specs["projectile_sphere"].restitution),
            ),
            surface=gs_mod.surfaces.Default(color=(0.96, 0.78, 0.23, 1.0)),
        )

        pendulum_urdf = Path(self.args.asset_dir) / f"pendulum_scene_{params.scene_index:05d}.urdf"
        write_pendulum_urdf(
            pendulum_urdf,
            length=params.pendulum_length,
            bob_radius=params.pendulum_bob_radius,
            bob_mass=float(params.object_specs["pendulum_bob"].mass),
            joint_damping=float(self.args.pendulum_joint_damping),
        )
        pendulum = scene.add_entity(
            morph=gs_mod.morphs.URDF(
                file=str(pendulum_urdf),
                pos=tuple(params.pendulum_anchor.tolist()),
                fixed=True,
                merge_fixed_links=False,
            ),
            material=make_rigid_material(
                gs_mod,
                rho=1000.0,
                friction=float(params.object_specs["pendulum_bob"].friction),
                restitution=float(params.object_specs["pendulum_bob"].restitution),
            ),
            surface=gs_mod.surfaces.Default(color=(0.60, 0.32, 0.72, 1.0)),
        )

        camera = scene.add_camera(
            res=(self.resolution, self.resolution),
            pos=tuple(initial_camera_pos.tolist()),
            lookat=tuple(initial_lookat.tolist()),
            up=(0.0, 0.0, 1.0),
            fov=float(self.camera_fov),
            near=0.05,
            far=14.0,
            GUI=bool(self.args.show_viewer),
        )

        scene.build()

        collision_striker.set_mass(float(params.object_specs["collision_striker"].mass))
        collision_target.set_mass(float(params.object_specs["collision_target"].mass))
        rolling_sphere.set_mass(float(params.object_specs["rolling_sphere"].mass))
        projectile_sphere.set_mass(float(params.object_specs["projectile_sphere"].mass))
        pendulum_bob_link = pendulum.get_link("bob")
        pendulum_bob_link.set_mass(float(params.object_specs["pendulum_bob"].mass))

        striker_velocity = np.concatenate(
            [
                params.object_specs["collision_striker"].impulse_vector.astype(np.float64)
                / float(params.object_specs["collision_striker"].mass),
                np.zeros(3, dtype=np.float64),
            ]
        )
        collision_striker.set_dofs_velocity(striker_velocity.tolist())

        rolling_linear_velocity = (
            params.object_specs["rolling_sphere"].impulse_vector.astype(np.float64)
            / float(params.object_specs["rolling_sphere"].mass)
        )
        rolling_angular_velocity = np.array(
            [abs(rolling_linear_velocity[1]) / max(params.rolling_sphere_radius, 1e-6), 0.0, 0.0],
            dtype=np.float64,
        )
        rolling_sphere.set_dofs_velocity(np.concatenate([rolling_linear_velocity, rolling_angular_velocity]).tolist())

        projectile_velocity = (
            params.object_specs["projectile_sphere"].impulse_vector.astype(np.float64)
            / float(params.object_specs["projectile_sphere"].mass)
        )
        projectile_sphere.set_dofs_velocity(np.concatenate([projectile_velocity, np.zeros(3, dtype=np.float64)]).tolist())

        pendulum.set_dofs_position([math.radians(params.pendulum_angle_deg)], zero_velocity=True)

        dynamic_bodies = [
            DynamicBody(
                name="collision_striker",
                phenomenon="collision",
                entity=collision_striker,
                link_local_idx=0,
                mass=float(params.object_specs["collision_striker"].mass),
                restitution=float(params.object_specs["collision_striker"].restitution),
                friction=float(params.object_specs["collision_striker"].friction),
                impulse_vector=params.object_specs["collision_striker"].impulse_vector.copy(),
            ),
            DynamicBody(
                name="collision_target",
                phenomenon="collision",
                entity=collision_target,
                link_local_idx=0,
                mass=float(params.object_specs["collision_target"].mass),
                restitution=float(params.object_specs["collision_target"].restitution),
                friction=float(params.object_specs["collision_target"].friction),
                impulse_vector=params.object_specs["collision_target"].impulse_vector.copy(),
            ),
            DynamicBody(
                name="rolling_sphere",
                phenomenon="rolling",
                entity=rolling_sphere,
                link_local_idx=0,
                mass=float(params.object_specs["rolling_sphere"].mass),
                restitution=float(params.object_specs["rolling_sphere"].restitution),
                friction=float(params.object_specs["rolling_sphere"].friction),
                impulse_vector=params.object_specs["rolling_sphere"].impulse_vector.copy(),
            ),
            DynamicBody(
                name="projectile_sphere",
                phenomenon="projectile",
                entity=projectile_sphere,
                link_local_idx=0,
                mass=float(params.object_specs["projectile_sphere"].mass),
                restitution=float(params.object_specs["projectile_sphere"].restitution),
                friction=float(params.object_specs["projectile_sphere"].friction),
                impulse_vector=params.object_specs["projectile_sphere"].impulse_vector.copy(),
            ),
            DynamicBody(
                name="pendulum_bob",
                phenomenon="pendulum",
                entity=pendulum,
                link_local_idx=int(pendulum_bob_link.idx_local),
                mass=float(params.object_specs["pendulum_bob"].mass),
                restitution=float(params.object_specs["pendulum_bob"].restitution),
                friction=float(params.object_specs["pendulum_bob"].friction),
                impulse_vector=params.object_specs["pendulum_bob"].impulse_vector.copy(),
            ),
        ]

        contact_monitors = [
            ContactMonitor(
                pair_name="collision_impact",
                entity_a=collision_striker,
                entity_b=collision_target,
                object_idx_a=0,
                object_idx_b=1,
                partner_name="collision_target",
            ),
            ContactMonitor(
                pair_name="rolling_contact",
                entity_a=rolling_sphere,
                entity_b=ramp,
                object_idx_a=2,
                object_idx_b=-1,
                partner_name="ramp",
            ),
            ContactMonitor(
                pair_name="projectile_impact",
                entity_a=projectile_sphere,
                entity_b=floor,
                object_idx_a=3,
                object_idx_b=-1,
                partner_name="floor",
            ),
        ]

        return SceneBundle(
            scene=scene,
            camera=camera,
            floor=floor,
            ramp=ramp,
            dynamic_bodies=dynamic_bodies,
            contact_monitors=contact_monitors,
        )

    def query_body_state(self, body: DynamicBody) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = to_numpy(body.entity.get_links_pos([body.link_local_idx], ref="link_com")).astype(np.float32)
        vel = to_numpy(body.entity.get_links_vel([body.link_local_idx], ref="link_com")).astype(np.float32)
        if pos.ndim == 3:
            pos = pos[0]
        if vel.ndim == 3:
            vel = vel[0]
        pos = pos[0]
        vel = vel[0]
        momentum = float(body.mass) * vel
        return pos, vel, momentum.astype(np.float32)

    def collect_body_arrays(self, bodies: Sequence[DynamicBody]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.empty((len(bodies), 3), dtype=np.float32)
        velocities = np.empty((len(bodies), 3), dtype=np.float32)
        momenta = np.empty((len(bodies), 3), dtype=np.float32)
        for idx, body in enumerate(bodies):
            pos, vel, momentum = self.query_body_state(body)
            positions[idx] = pos
            velocities[idx] = vel
            momenta[idx] = momentum
        return positions, velocities, momenta

    def compute_camera_pose(
        self,
        params: SceneParams,
        frame_idx: int,
        body_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.fixed_camera_pos.copy(), self.fixed_camera_lookat.copy()

    def validate_runtime_state(
        self,
        bundle: SceneBundle,
        body_positions: np.ndarray,
        body_velocities: np.ndarray,
        camera_pos: np.ndarray,
        camera_lookat: np.ndarray,
    ) -> None:
        if not np.isfinite(body_positions).all() or not np.isfinite(body_velocities).all():
            raise SceneGenerationError("Body state contains NaN or Inf.")

        if np.any(body_positions < self.world_bounds_min[None, :]) or np.any(body_positions > self.world_bounds_max[None, :]):
            raise SceneGenerationError("Tracked body escaped world bounds.")

        visibility = points_in_camera_frustum(
            body_positions.astype(np.float64),
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_up=self.camera_up,
            fov_deg=self.camera_fov,
            near=float(bundle.camera.near),
            far=float(bundle.camera.far),
            margin=float(self.args.visibility_margin),
        )
        if not bool(np.all(visibility)):
            raise SceneGenerationError("Tracked body left the camera frustum.")

        seen_entities = set()
        for body in bundle.dynamic_bodies:
            token = id(body.entity)
            if token in seen_entities:
                continue
            seen_entities.add(token)
            contacts = body.entity.get_contacts(exclude_self_contact=True)
            penetration = np.asarray(to_numpy(contacts.get("penetration", np.zeros((0,), dtype=np.float32))), dtype=np.float32)
            if penetration.size > 0 and float(np.max(penetration)) > float(self.args.max_penetration):
                raise SceneGenerationError("Detected excessive rigid penetration.")

    def contact_is_active(self, monitor: ContactMonitor) -> bool:
        contacts = monitor.entity_a.get_contacts(with_entity=monitor.entity_b)
        penetration = np.asarray(to_numpy(contacts.get("penetration", np.zeros((0,), dtype=np.float32))), dtype=np.float32)
        return bool(penetration.size > 0)

    def update_collision_events(
        self,
        *,
        monitors: Sequence[ContactMonitor],
        previous_active: Dict[str, bool],
        pre_momenta: np.ndarray,
        post_momenta: np.ndarray,
        frame_idx: int,
        events: List[Dict[str, Any]],
    ) -> None:
        for monitor in monitors:
            is_active = self.contact_is_active(monitor)
            was_active = bool(previous_active.get(monitor.pair_name, False))
            if is_active and not was_active:
                delta_pair = np.zeros((2, 3), dtype=np.float32)
                delta_pair[0] = post_momenta[monitor.object_idx_a] - pre_momenta[monitor.object_idx_a]
                if monitor.object_idx_b >= 0:
                    delta_pair[1] = post_momenta[monitor.object_idx_b] - pre_momenta[monitor.object_idx_b]

                events.append(
                    {
                        "pair_name": monitor.pair_name,
                        "frame_idx": int(frame_idx),
                        "object_indices": np.array([monitor.object_idx_a, monitor.object_idx_b], dtype=np.int32),
                        "partner_name": monitor.partner_name,
                        "delta_p": delta_pair,
                    }
                )
            previous_active[monitor.pair_name] = is_active

    def validate_scene_payload(
        self,
        trajectory: np.ndarray,
        events: Sequence[Dict[str, Any]],
    ) -> None:
        event_names = {str(event["pair_name"]) for event in events}
        if "collision_impact" not in event_names:
            raise SceneGenerationError("Striker-target collision was not observed.")
        if "projectile_impact" not in event_names:
            raise SceneGenerationError("Projectile-floor impact was not observed.")

        positions = trajectory[..., :3]
        target_disp = float(np.linalg.norm(positions[-1, 1] - positions[0, 1]))
        rolling_disp = float(np.linalg.norm(positions[-1, 2] - positions[0, 2]))
        projectile_z = positions[:, 3, 2]
        pendulum_x = positions[:, 4, 0]

        if target_disp < 0.03:
            raise SceneGenerationError("Collision target barely moved after impact.")
        if rolling_disp < 0.20:
            raise SceneGenerationError("Rolling sphere displacement is too small.")
        if float(np.max(projectile_z) - projectile_z[0]) < 0.22:
            raise SceneGenerationError("Projectile arc is too weak.")
        if float(np.max(projectile_z) - projectile_z[-1]) < 0.18:
            raise SceneGenerationError("Projectile did not fall back after apex.")
        if float(np.ptp(pendulum_x)) < 0.12:
            raise SceneGenerationError("Pendulum lateral motion is too small.")

    def generate_scene(self, params: SceneParams) -> Dict[str, Any]:
        bundle = self.build_scene(params)
        scene = bundle.scene
        camera = bundle.camera
        bodies = bundle.dynamic_bodies

        rgb_frames = np.empty((self.frames, self.resolution, self.resolution, 3), dtype=np.uint8)
        depth_frames = np.empty((self.frames, self.resolution, self.resolution, 1), dtype=np.float32)
        trajectory = np.empty((self.frames, len(bodies), 6), dtype=np.float32)
        camera_positions = np.empty((self.frames, 3), dtype=np.float32)
        camera_lookats = np.empty((self.frames, 3), dtype=np.float32)

        previous_active = {monitor.pair_name: False for monitor in bundle.contact_monitors}
        collision_events: List[Dict[str, Any]] = []
        step_counter = 0

        for frame_idx in range(self.frames):
            body_positions, body_velocities, body_momenta = self.collect_body_arrays(bodies)
            camera_pos, camera_lookat = self.compute_camera_pose(params, frame_idx, body_positions)
            camera.set_pose(pos=tuple(camera_pos.tolist()), lookat=tuple(camera_lookat.tolist()), up=(0.0, 0.0, 1.0))
            self.validate_runtime_state(bundle, body_positions, body_velocities, camera_pos, camera_lookat)

            rendered = camera.render(rgb=True, depth=True, segmentation=False, normal=False)
            if not isinstance(rendered, tuple) or len(rendered) < 2:
                raise SceneGenerationError("Unexpected camera render output.")
            rgb_raw, depth_raw = rendered[0], rendered[1]

            rgb_frames[frame_idx] = rgb_to_uint8(rgb_raw)
            depth_frames[frame_idx] = normalize_depth_map(depth_raw, near=float(camera.near), far=float(camera.far))
            trajectory[frame_idx, :, :3] = body_positions
            trajectory[frame_idx, :, 3:] = body_momenta
            camera_positions[frame_idx] = camera_pos.astype(np.float32)
            camera_lookats[frame_idx] = camera_lookat.astype(np.float32)

            if frame_idx == self.frames - 1:
                break

            for _ in range(self.steps_per_frame):
                _, _, pre_momenta = self.collect_body_arrays(bodies)
                scene.step()
                step_counter += 1
                _, _, post_momenta = self.collect_body_arrays(bodies)
                output_frame_idx = min(self.frames - 1, int(math.ceil(step_counter / float(self.steps_per_frame))))
                self.update_collision_events(
                    monitors=bundle.contact_monitors,
                    previous_active=previous_active,
                    pre_momenta=pre_momenta,
                    post_momenta=post_momenta,
                    frame_idx=output_frame_idx,
                    events=collision_events,
                )

        self.validate_scene_payload(trajectory, collision_events)

        return {
            "rgb": rgb_frames,
            "depth": depth_frames,
            "trajectory": trajectory,
            "camera_pos": camera_positions,
            "camera_lookat": camera_lookats,
            "dynamic_bodies": bodies,
            "collision_events": collision_events,
        }

    def write_scene_folder(
        self,
        scene_dir: Path,
        payload: Dict[str, Any],
        params: SceneParams,
        *,
        per_scene_seed: int,
    ) -> None:
        ensure_dir(scene_dir)
        rgb_dir = scene_dir / "rgb"
        depth_dir = scene_dir / "depth"
        videos_dir = scene_dir / "videos"
        physics_dir = scene_dir / "physics"
        ensure_dir(rgb_dir)
        ensure_dir(depth_dir)
        ensure_dir(videos_dir)
        ensure_dir(physics_dir)

        bodies: List[DynamicBody] = list(payload["dynamic_bodies"])
        collision_events: List[Dict[str, Any]] = list(payload["collision_events"])
        rgb_frames = np.asarray(payload["rgb"], dtype=np.uint8)
        depth_frames = np.asarray(payload["depth"], dtype=np.float32)

        scene_metadata = {
            "dataset_name": DATASET_NAME,
            "scene_index": int(params.scene_index),
            "seed": int(per_scene_seed),
            "dt": float(self.dt),
            "substeps": int(self.substeps),
            "steps_per_frame": int(self.steps_per_frame),
            "frames": int(self.frames),
            "resolution": int(self.resolution),
            "camera_fov": float(self.camera_fov),
            "fixed_camera_pos": self.fixed_camera_pos.astype(np.float32).tolist(),
            "fixed_camera_lookat": self.fixed_camera_lookat.astype(np.float32).tolist(),
            "background_color": [float(x) for x in self.background_color],
            "ambient_light": [float(x) for x in self.ambient_light],
            "trajectory_definition": "COM position xyz + linear momentum xyz",
        }
        with open(scene_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(scene_metadata, f, ensure_ascii=False, indent=2)

        for frame_idx, rgb_frame in enumerate(rgb_frames):
            imageio.imwrite(rgb_dir / f"frame_{frame_idx:03d}.png", rgb_frame)
        for frame_idx, depth_frame in enumerate(depth_frames):
            imageio.imwrite(depth_dir / f"frame_{frame_idx:03d}.png", depth_to_uint8(depth_frame))

        fps = max(1, int(round(1.0 / max(self.dt * self.steps_per_frame, 1e-8))))
        imageio.mimsave(videos_dir / "rgb.mp4", list(rgb_frames), fps=fps)
        depth_video_frames = [depth_to_uint8(frame) for frame in depth_frames]
        imageio.mimsave(videos_dir / "depth.mp4", depth_video_frames, fps=fps)

        np.save(physics_dir / "trajectory.npy", np.asarray(payload["trajectory"], dtype=np.float32))
        np.save(physics_dir / "camera_position.npy", np.asarray(payload["camera_pos"], dtype=np.float32))
        np.save(physics_dir / "camera_lookat.npy", np.asarray(payload["camera_lookat"], dtype=np.float32))
        np.save(physics_dir / "depth_normalized.npy", depth_frames)

        properties_payload = {
            "object_names": [body.name for body in bodies],
            "phenomena": [body.phenomenon for body in bodies],
            "mass": [float(body.mass) for body in bodies],
            "restitution": [float(body.restitution) for body in bodies],
            "friction": [float(body.friction) for body in bodies],
            "impulse_vector": [body.impulse_vector.astype(np.float32).tolist() for body in bodies],
        }
        with open(physics_dir / "properties.json", "w", encoding="utf-8") as f:
            json.dump(properties_payload, f, ensure_ascii=False, indent=2)

        collision_payload = []
        for event in collision_events:
            collision_payload.append(
                {
                    "pair_name": str(event["pair_name"]),
                    "frame_idx": int(event["frame_idx"]),
                    "partner_name": str(event["partner_name"]),
                    "object_indices": np.asarray(event["object_indices"], dtype=np.int32).tolist(),
                    "delta_p": np.asarray(event["delta_p"], dtype=np.float32).tolist(),
                }
            )
        with open(physics_dir / "collision_events.json", "w", encoding="utf-8") as f:
            json.dump(collision_payload, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Hamiltonian-Rigid-Dynamics dataset with Genesis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-scenes", type=int, default=1, help="Number of scene folders to generate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for per-scene folders.")
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR, help="Directory for generated helper URDF assets.")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Frames saved per sample. Keep 49 for the target dataset.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Square render resolution. Use 512 if you need the earlier lower-resolution setting.",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Simulation step size in seconds.")
    parser.add_argument("--substeps", type=int, default=4, help="Genesis rigid substeps per scene step.")
    parser.add_argument("--steps-per-frame", type=int, default=5, help="Simulation steps between saved frames.")
    parser.add_argument("--camera-fov", type=float, default=60.0, help="Camera field-of-view in degrees.")
    parser.add_argument(
        "--fixed-camera-pos",
        type=float,
        nargs=3,
        default=(7.2, -7.2, 4.2),
        metavar=("X", "Y", "Z"),
        help="Fixed world-space camera position.",
    )
    parser.add_argument(
        "--fixed-camera-lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Fixed world-space camera lookat point.",
    )
    parser.add_argument("--visibility-margin", type=float, default=0.995, help="Projection margin used by the FOV rejection logic.")
    parser.add_argument(
        "--background-color",
        type=float,
        nargs=3,
        default=DEFAULT_BACKGROUND_COLOR,
        metavar=("R", "G", "B"),
        help="Scene background color in normalized RGB.",
    )
    parser.add_argument(
        "--ambient-light",
        type=float,
        nargs=3,
        default=DEFAULT_AMBIENT_LIGHT,
        metavar=("R", "G", "B"),
        help="Ambient light color in normalized RGB.",
    )
    parser.add_argument("--max-penetration", type=float, default=0.05, help="Reject sample when a contact penetration exceeds this threshold in meters.")
    parser.add_argument("--max-scene-retries", type=int, default=12, help="Maximum retries per requested output sample.")
    parser.add_argument("--rigid-iterations", type=int, default=96, help="Newton solver iterations for Genesis rigid dynamics.")
    parser.add_argument("--rigid-ls-iterations", type=int, default=80, help="Line-search iterations for the rigid solver.")
    parser.add_argument("--rigid-noslip-iterations", type=int, default=8, help="Noslip post-processing iterations.")
    parser.add_argument("--rigid-tolerance", type=float, default=1e-7, help="Rigid solver tolerance.")
    parser.add_argument("--rigid-ls-tolerance", type=float, default=1e-3, help="Rigid solver line-search tolerance.")
    parser.add_argument("--constraint-timeconst", type=float, default=0.006, help="Rigid contact time constant.")
    parser.add_argument("--max-collision-pairs", type=int, default=256, help="Maximum rigid collision pairs.")
    parser.add_argument("--pendulum-joint-damping", type=float, default=0.02, help="URDF revolute-joint damping for the pendulum.")
    parser.add_argument(
        "--pyopengl-platform",
        choices=["egl", "pyglet", "osmesa"],
        default=None,
        help="Override offscreen OpenGL backend before Genesis import.",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force Genesis CPU backend.")
    parser.add_argument("--show-viewer", action="store_true", help="Enable the Genesis viewer for debugging.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.pyopengl_platform:
        os.environ["PYOPENGL_PLATFORM"] = str(args.pyopengl_platform)
        if args.pyopengl_platform == "pyglet" and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("PYGLET_HEADLESS", "1")

    if args.frames != DEFAULT_FRAMES:
        print(f"[warn] --frames={args.frames} differs from the requested dataset spec of 49.", flush=True)
    if args.resolution != DEFAULT_RESOLUTION:
        print(
            f"[warn] --resolution={args.resolution} differs from the current default of {DEFAULT_RESOLUTION}.",
            flush=True,
        )

    ensure_python_dependencies()
    ensure_dir(args.output_dir)
    ensure_dir(args.asset_dir)
    backend = ensure_genesis_initialized(prefer_gpu=not args.cpu_only)
    print(f"[init] Genesis backend: {backend}", flush=True)

    generator = HamiltonianRigidDatasetGenerator(args)
    master_rng = np.random.default_rng(int(args.seed))

    generated = 0
    while generated < int(args.num_scenes):
        success = False
        for attempt_idx in range(int(args.max_scene_retries)):
            per_scene_seed = int(master_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(per_scene_seed)
            random.seed(per_scene_seed)
            params = generator.sample_scene_params(scene_index=generated, rng=rng)
            scene_dir = Path(args.output_dir) / f"scene_{generated:05d}"

            try:
                payload = generator.generate_scene(params)
                generator.write_scene_folder(scene_dir, payload, params, per_scene_seed=per_scene_seed)
                print(
                    f"[scene {generated:05d}] saved {scene_dir} "
                    f"(attempt {attempt_idx + 1}, seed {per_scene_seed})",
                    flush=True,
                )
                success = True
                break
            except SceneGenerationError as exc:
                print(
                    f"[scene {generated:05d}] reject attempt {attempt_idx + 1}/{args.max_scene_retries}: {exc}",
                    flush=True,
                )
                gc.collect()

        if not success:
            raise RuntimeError(
                f"Failed to generate scene {generated:05d} after {args.max_scene_retries} attempts."
            )
        generated += 1

    print(f"[done] Generated {generated} scene(s) for {DATASET_NAME}.", flush=True)


if __name__ == "__main__":
    main()
'''


/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/try3_SLD.py \
  --num-scenes 1 \
  --output-dir /data/gaoya/AAA_test_video/Dataset_physV/hamiltonian_rigid_h5/hamiltonian_rigid_h5_test \
  --asset-dir /data/gaoya/AAA_test_video/Dataset_physV/hamiltonian_rigid_h5/_hamiltonian_rigid_assets \
  --seed 42
'''
