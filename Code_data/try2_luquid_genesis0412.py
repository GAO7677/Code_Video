#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis liquid-object interaction dataset generator.

This script builds a Lagrangian fluid-object interaction dataset where an emitter
pours liquid into random container meshes. It supports:

- Genesis SPH or PBD liquid solver
- PhysXNet container discovery and on-the-fly merged OBJ caching
- Headless rendering of RGB and normalized depth
- Stable particle tracking for a fixed set of particle IDs across 49 frames
- Automatic scene rejection and retry when tracked particles are lost or leave bounds
- Per-scene folder export with RGB/depth frames, particle arrays, metadata, and preview videos

Example:

python3 /home/gaoya/Code_Video/Code_data/luquid_genesis0412.py \
  --num-scenes 10 \
  --solver sph \
  --output-dir /home/gaoya/Code_Video/Code_data/liquid_dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:
    np = None
try:
    import h5py
except Exception:
    h5py = None
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import trimesh
except Exception:
    trimesh = None

gs = None


DEFAULT_PHYSX_ROOT = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
DEFAULT_PHYSX_VERSION = "version_1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "liquid_dataset"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "_liquid_container_cache"
CONTAINER_CACHE_VERSION = "proc_v7_diverse_shapes"

CONTAINER_KEYWORDS = (
    "bowl",
    "cup",
    "mug",
    "container",
    "bucket",
    "pot",
    "jar",
    "basin",
    "box",
    "bin",
)
CONTAINER_CATEGORY_KEYWORDS = (
    "container",
    "storagecontainer",
    "storage container",
    "storagebox",
    "drinkware",
    "kitchenware",
    "cookware",
    "tableware",
    "household container",
)
DRINKING_GLASS_KEYWORDS = (
    "drinking glass",
    "water glass",
    "glass cup",
    "glass mug",
    "glass tumbler",
    "glass jar",
)
NON_CONTAINER_KEYWORDS = (
    "table",
    "coffee table",
    "desk",
    "cabinet",
    "drawer",
    "wardrobe",
    "shelf",
    "rack",
    "lamp",
    "spotlight",
    "faucet",
    "plumbing fixture",
    "chair",
    "stool",
    "bench",
    "sofa",
    "bed",
    "cart",
    "stand",
    "frame",
    "mirror",
    "door",
)
LIQUID_KEYWORDS = (
    "liquid",
    "water",
    "drink",
    "beverage",
    "soup",
    "oil",
    "juice",
    "sauce",
)
CLOSURE_KEYWORDS = (
    "lid",
    "cover",
    "cap",
    "stopper",
    "plug",
)

GENESIS_INIT_LOCK = threading.Lock()
GENESIS_INITIALIZED = False
GENESIS_BACKEND_USED = "none"
GENESIS_RENDER_PLATFORM: Optional[str] = None
GENESIS_EGL_DEVICE_ID: Optional[int] = None

LIQUID_PRESET_SEQUENCE = (
    "calm_single",
    "splash_single",
    "viscous_dual",
    "dual_collision",
    "triple_arc",
)

LIQUID_PRESETS: Dict[str, Dict[str, Any]] = {
    "calm_single": {
        "cluster_range": (1, 1),
        "viscosity": (0.0025, 0.0060),
        "stiffness": (36000.0, 52000.0),
        "exponent": (6.2, 7.0),
        "gamma": (0.0012, 0.0038),
        "speed": (0.50, 0.82),
        "diameter_scale": (5.2, 6.2),
        "budget_scale": (1.30, 1.80),
        "volume_tiers": (0.35, 1.10, 2.80),
    },
    "splash_single": {
        "cluster_range": (1, 1),
        "viscosity": (0.0012, 0.0040),
        "stiffness": (56000.0, 86000.0),
        "exponent": (6.8, 7.6),
        "gamma": (0.0008, 0.0030),
        "speed": (0.95, 1.40),
        "diameter_scale": (5.8, 6.8),
        "budget_scale": (1.40, 2.00),
        "volume_tiers": (0.30, 1.00, 3.20),
    },
    "viscous_dual": {
        "cluster_range": (2, 2),
        "viscosity": (0.0035, 0.0075),
        "stiffness": (24000.0, 36000.0),
        "exponent": (5.8, 6.5),
        "gamma": (0.0015, 0.0048),
        "speed": (0.34, 0.56),
        "diameter_scale": (5.0, 6.0),
        "budget_scale": (1.35, 2.10),
        "volume_tiers": (0.45, 1.20, 3.60),
    },
    "dual_collision": {
        "cluster_range": (2, 2),
        "viscosity": (0.0025, 0.0065),
        "stiffness": (28000.0, 42000.0),
        "exponent": (6.0, 6.8),
        "gamma": (0.0012, 0.0042),
        "speed": (0.48, 0.78),
        "diameter_scale": (5.0, 6.0),
        "budget_scale": (1.45, 2.20),
        "volume_tiers": (0.35, 1.15, 3.00),
    },
    "triple_arc": {
        "cluster_range": (3, 3),
        "viscosity": (0.0022, 0.0058),
        "stiffness": (26000.0, 38000.0),
        "exponent": (6.0, 6.7),
        "gamma": (0.0010, 0.0038),
        "speed": (0.40, 0.68),
        "diameter_scale": (4.8, 5.8),
        "budget_scale": (1.40, 2.10),
        "volume_tiers": (0.28, 1.00, 2.60),
    },
}

CONTAINER_COLOR_PALETTES: Tuple[Dict[str, Any], ...] = (
    {"name": "neutral_ceramic", "rgba": (0.72, 0.72, 0.70, 1.0)},
)

FLUID_COLOR_PALETTES: Tuple[Dict[str, Any], ...] = (
    {"name": "reference_water", "rgba": (0.34, 0.56, 0.86, 1.0)},
)


class SceneGenerationError(RuntimeError):
    """Raised when a sampled scene is invalid and must be retried."""


@dataclass
class ContainerCandidate:
    kind: str
    source_id: str
    label: str
    obj_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    source_up: str = "z"


@dataclass
class PreparedContainer:
    cache_path: Path
    candidate: ContainerCandidate
    canonical_extents: np.ndarray
    scale_to_scene: float

    @property
    def scene_extents(self) -> np.ndarray:
        return self.canonical_extents * self.scale_to_scene


@dataclass
class ContainerPlacement:
    prepared_container: PreparedContainer
    pos: np.ndarray
    quat_wxyz: Tuple[float, float, float, float]
    yaw_deg: float
    color_name: str
    surface_rgba: Tuple[float, float, float, float]
    emissive_rgb: Tuple[float, float, float]


@dataclass
class LiquidCluster:
    cluster_index: int
    target_container_index: int
    volume_tier: str
    direction_name: str
    pos: np.ndarray
    direction: np.ndarray
    speed: float
    diameter: float
    length: float
    target_particles: int


@dataclass
class SceneParams:
    scene_index: int
    preset_name: str
    scene_variant_name: str
    viscosity: float
    stiffness: float
    exponent: float
    gamma: float
    emitter_speed: float
    emitter_diameter: float
    emitter_length: float
    camera_azimuth_deg: float
    camera_elevation_deg: float
    camera_radius: float
    camera_fov_deg: float
    gravity: Tuple[float, float, float]
    prepared_container: PreparedContainer
    solver_name: str
    solver_substeps: int
    scene_bounds_lower: np.ndarray
    scene_bounds_upper: np.ndarray
    emitter_pos: np.ndarray
    emitter_direction: np.ndarray
    emission_interval_steps: int
    camera_pos: np.ndarray
    camera_lookat: np.ndarray
    container_pos: np.ndarray
    emission_steps_total: int
    clusters_per_container: int
    tracking_bounds_lower: np.ndarray
    tracking_bounds_upper: np.ndarray
    estimated_particles_per_emit: int
    target_particle_budget: int
    containers: List[ContainerPlacement]
    liquid_clusters: List[LiquidCluster]
    source_container: Optional[ContainerPlacement]
    source_anchor: Optional[np.ndarray]
    source_mode: str
    source_delay_steps: int
    fluid_color_name: str
    fluid_rgba: Tuple[float, float, float, float]
    source_tilt_schedule_deg: Optional[np.ndarray] = None
    source_rest_quat_wxyz: Optional[Tuple[float, float, float, float]] = None
    source_collision_proxy_size: Optional[Tuple[float, float, float]] = None
    source_collision_proxy_offset: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class EmissionPulse:
    emit_step: int
    cluster_index: int
    pulse_count: int
    pulse_index: int
    estimated_particles: int


def _get_default_genesis_repo() -> Path:
    return Path(__file__).resolve().parents[1] / "Genesis_main"


def ensure_python_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if imageio is None:
        missing.append("imageio")
    if trimesh is None:
        missing.append("trimesh")
    if missing:
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + ". Please install them in the runtime environment before generating the dataset."
        )


def _import_genesis():
    global gs
    if gs is not None:
        return gs

    if GENESIS_RENDER_PLATFORM is not None:
        os.environ["PYOPENGL_PLATFORM"] = GENESIS_RENDER_PLATFORM
    if GENESIS_EGL_DEVICE_ID is not None:
        os.environ["EGL_DEVICE_ID"] = str(GENESIS_EGL_DEVICE_ID)

    genesis_repo = _get_default_genesis_repo()
    if genesis_repo.exists():
        sys.path.insert(0, str(genesis_repo))

    import genesis as _gs  # type: ignore

    gs = _gs
    return gs


def configure_render_backend(render_platform: str, egl_device_id: Optional[int]) -> None:
    global GENESIS_RENDER_PLATFORM, GENESIS_EGL_DEVICE_ID

    GENESIS_RENDER_PLATFORM = None if render_platform == "auto" else render_platform
    GENESIS_EGL_DEVICE_ID = egl_device_id

    if GENESIS_RENDER_PLATFORM is None:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    else:
        os.environ["PYOPENGL_PLATFORM"] = GENESIS_RENDER_PLATFORM

    if GENESIS_EGL_DEVICE_ID is None:
        os.environ.pop("EGL_DEVICE_ID", None)
    else:
        os.environ["EGL_DEVICE_ID"] = str(GENESIS_EGL_DEVICE_ID)


def validate_runtime_configuration(args: argparse.Namespace) -> None:
    if sys.platform.startswith("linux") and args.render_platform == "pyglet":
        has_x11 = bool(os.environ.get("DISPLAY"))
        has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
        if not has_x11 and not has_wayland:
            raise ValueError(
                "--render-platform pyglet requires a live desktop display on Linux. "
                "This machine does not expose DISPLAY/WAYLAND_DISPLAY, so pyglet cannot create the hidden OpenGL "
                "window used by Genesis offscreen rendering. Use --render-platform egl, leave it at --render-platform "
                "auto, or add --skip-render for physics-only generation."
            )
    if float(args.liquid_volume_scale) <= 0.0:
        raise ValueError("--liquid-volume-scale must be > 0.")
    if float(args.camera_distance_scale) <= 0.0:
        raise ValueError("--camera-distance-scale must be > 0.")
    if not (10.0 <= float(args.camera_fov) <= 140.0):
        raise ValueError("--camera-fov must be between 10 and 140 degrees.")


def ensure_genesis_initialized(prefer_gpu: bool = True) -> str:
    global GENESIS_INITIALIZED, GENESIS_BACKEND_USED

    gs_mod = _import_genesis()
    if GENESIS_INITIALIZED:
        return GENESIS_BACKEND_USED

    with GENESIS_INIT_LOCK:
        if GENESIS_INITIALIZED:
            return GENESIS_BACKEND_USED

        init_candidates = []
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
            except Exception as exc:  # pragma: no cover - defensive
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def parse_dim_string(dim_text: Any) -> Optional[np.ndarray]:
    if dim_text is None:
        return None
    if isinstance(dim_text, (list, tuple)) and len(dim_text) == 3:
        values = np.asarray(dim_text, dtype=np.float64)
    else:
        text = str(dim_text).strip().lower().replace("cm", "").replace(" ", "")
        parts = text.split("*")
        if len(parts) != 3:
            return None
        try:
            values = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        except Exception:
            return None
    if np.any(values <= 0):
        return None
    return values / 100.0


def parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def parse_csv_choices(value: str) -> List[str]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated non-empty list.")
    return parts


def parse_int_csv(value: str) -> List[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated non-empty integer list.")
    parsed: List[int] = []
    for part in parts:
        try:
            parsed.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid integer value: {part}") from exc
    return parsed


def parse_float_pair(value: str) -> Tuple[float, float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated floats.")
    try:
        low = float(parts[0])
        high = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float range: {value}") from exc
    if low <= 0.0 or high <= 0.0:
        raise argparse.ArgumentTypeError("Volume multipliers must be positive.")
    if low > high:
        raise argparse.ArgumentTypeError("Expected low <= high for the liquid volume range.")
    return (low, high)


def is_container_like(text: str) -> bool:
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in NON_CONTAINER_KEYWORDS):
        return False
    if any(keyword in text_lower for keyword in DRINKING_GLASS_KEYWORDS):
        return True
    if any(keyword in text_lower for keyword in CONTAINER_CATEGORY_KEYWORDS):
        return True
    return any(keyword in text_lower for keyword in CONTAINER_KEYWORDS)


def is_preferred_container_like(text: str) -> bool:
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in NON_CONTAINER_KEYWORDS):
        return False
    preferred = ("bowl", "cup", "mug", "container", "bucket", "jar", "basin", "pot")
    if any(keyword in text_lower for keyword in DRINKING_GLASS_KEYWORDS):
        return True
    return any(keyword in text_lower for keyword in preferred)


def is_liquid_like(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in LIQUID_KEYWORDS)


def is_closure_like(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CLOSURE_KEYWORDS)


def normalize_vector(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return arr / norm


def z_yaw_quat_wxyz(yaw_deg: float) -> Tuple[float, float, float, float]:
    half = 0.5 * math.radians(float(yaw_deg))
    return (float(math.cos(half)), 0.0, 0.0, float(math.sin(half)))


def axis_angle_quat_wxyz(axis: Sequence[float], angle_deg: float) -> Tuple[float, float, float, float]:
    axis_arr = normalize_vector(axis)
    half = 0.5 * math.radians(float(angle_deg))
    sin_half = math.sin(half)
    return (
        float(math.cos(half)),
        float(axis_arr[0] * sin_half),
        float(axis_arr[1] * sin_half),
        float(axis_arr[2] * sin_half),
    )


def quat_mul_wxyz(
    q0: Sequence[float],
    q1: Sequence[float],
) -> Tuple[float, float, float, float]:
    w0, x0, y0, z0 = [float(v) for v in q0]
    w1, x1, y1, z1 = [float(v) for v in q1]
    return (
        float(w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1),
        float(w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1),
        float(w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1),
        float(w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1),
    )


def y_up_to_z_up_transform() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_trimesh_single(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom.copy()
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No valid meshes found in scene asset: {mesh_path}")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError(f"Empty mesh asset: {mesh_path}")
        return mesh
    raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(loaded)}")


def canonicalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    source_up: str,
    physical_dims_m: Optional[np.ndarray],
) -> trimesh.Trimesh:
    mesh = mesh.copy()
    if source_up == "y":
        mesh.apply_transform(y_up_to_z_up_transform())

    extents = np.maximum(mesh.extents.astype(np.float64), 1e-8)
    if physical_dims_m is not None:
        scale = float(np.median(np.asarray(physical_dims_m, dtype=np.float64) / extents))
        if np.isfinite(scale) and scale > 1e-8:
            mesh.apply_scale(scale)

    bounds = mesh.bounds.astype(np.float64)
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    z_min = bounds[0, 2]
    mesh.apply_translation(np.array([-center_xy[0], -center_xy[1], -z_min], dtype=np.float64))
    return mesh


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb_np = to_numpy(rgb)
    if rgb_np.ndim != 3:
        raise ValueError(f"Expected RGB image with 3 dims, got shape {rgb_np.shape}")
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    if rgb_np.dtype == np.uint8:
        return rgb_np
    rgb_np = rgb_np.astype(np.float32)
    if rgb_np.max(initial=0.0) <= 1.0:
        rgb_np = rgb_np * 255.0
    rgb_np = np.clip(rgb_np, 0.0, 255.0)
    return rgb_np.astype(np.uint8)


def normalize_depth_map(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth_np = to_numpy(depth).astype(np.float32)
    valid = np.isfinite(depth_np) & (depth_np > near) & (depth_np < far)
    denom = max(far - near, 1e-6)
    depth_norm = np.zeros_like(depth_np, dtype=np.float32)
    depth_norm[valid] = np.clip((depth_np[valid] - near) / denom, 0.0, 1.0)
    return depth_norm[..., None]


def depth_to_uint8(depth_norm: np.ndarray) -> np.ndarray:
    depth_np = np.asarray(depth_norm, dtype=np.float32)
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=1.0, neginf=0.0)
    depth_np = np.clip(depth_np, 0.0, 1.0)
    return np.round(depth_np * 255.0).astype(np.uint8)


def write_mp4_rgb(path: Path, frames: np.ndarray, fps: int) -> None:
    imageio.mimwrite(
        path,
        frames,
        fps=fps,
        codec="libx264rgb",
        quality=8,
        ffmpeg_params=["-pix_fmt", "rgb24"],
    )


def get_effective_solver_viscosity(solver_name: str, requested_viscosity: float) -> float:
    if solver_name == "sph":
        return float(np.clip(requested_viscosity, 0.0015, 0.020))
    return float(requested_viscosity)


def estimate_particles_per_emit(diameter: float, length: float, particle_size: float) -> int:
    radius = max(0.5 * float(diameter), 1e-6)
    cyl_volume = math.pi * radius * radius * max(float(length), 1e-6)
    particle_volume = max(float(particle_size) ** 3, 1e-9)
    estimate = int(math.ceil(cyl_volume / particle_volume))
    return max(1, estimate)


def estimate_droplet_length_for_particles(diameter: float, target_particles: int, particle_size: float) -> float:
    radius = max(0.5 * float(diameter), 1e-6)
    cross_section = math.pi * radius * radius
    particle_volume = max(float(particle_size) ** 3, 1e-9)
    target_volume = max(int(target_particles), 1) * particle_volume
    return float(target_volume / max(cross_section, 1e-9))


def build_revolved_open_container(profile_points: Sequence[Sequence[float]], sections: int = 64) -> trimesh.Trimesh:
    profile = np.asarray(profile_points, dtype=np.float64)
    mesh = trimesh.creation.revolve(profile, sections=sections)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def build_revolved_solid_open_container(profile_points: Sequence[Sequence[float]], sections: int = 64) -> trimesh.Trimesh:
    profile = np.asarray(profile_points, dtype=np.float64)
    if profile.shape[0] < 3:
        raise ValueError("profile_points must contain at least 3 points.")
    if not np.allclose(profile[0], profile[-1]):
        profile = np.vstack([profile, profile[0]])
    mesh = trimesh.creation.revolve(profile, sections=sections)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def voxel_fill_mesh(mesh: trimesh.Trimesh, pitch: float) -> trimesh.Trimesh:
    voxel_grid = mesh.voxelized(float(pitch))
    filled = voxel_grid.fill()
    rebuilt = filled.marching_cubes
    rebuilt.remove_unreferenced_vertices()
    rebuilt.merge_vertices()
    return rebuilt


def finalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def make_box_mesh(
    extents: Sequence[float],
    center: Sequence[float],
    rotation: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=tuple(float(v) for v in extents))
    transform = trimesh.transformations.translation_matrix(tuple(float(v) for v in center))
    if rotation is not None:
        transform = transform @ rotation
    mesh.apply_transform(transform)
    return mesh


def combine_meshes(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    if not meshes:
        raise ValueError("Expected at least one mesh to combine.")
    return finalize_mesh(trimesh.util.concatenate(list(meshes)))


def build_rectangular_shell(
    *,
    width: float,
    depth: float,
    height: float,
    wall: float,
    bottom: float,
    include_front: bool = True,
    include_back: bool = True,
    include_left: bool = True,
    include_right: bool = True,
    include_top: bool = False,
    top: Optional[float] = None,
) -> trimesh.Trimesh:
    wall = float(max(1e-4, min(wall, 0.45 * min(width, depth))))
    bottom = float(max(1e-4, min(bottom, 0.45 * height)))
    top_thickness = float(max(1e-4, top if top is not None else wall))
    usable_height = float(height - bottom - (top_thickness if include_top else 0.0))
    wall_height = max(usable_height, wall)
    z_center = bottom + 0.5 * wall_height
    meshes: List[trimesh.Trimesh] = [
        make_box_mesh((width, depth, bottom), (0.0, 0.0, 0.5 * bottom)),
    ]

    if include_front:
        meshes.append(make_box_mesh((width, wall, wall_height), (0.0, 0.5 * (depth - wall), z_center)))
    if include_back:
        meshes.append(make_box_mesh((width, wall, wall_height), (0.0, -0.5 * (depth - wall), z_center)))

    side_y_min = -0.5 * depth + (wall if include_back else 0.0)
    side_y_max = 0.5 * depth - (wall if include_front else 0.0)
    side_depth = max(wall, side_y_max - side_y_min)
    side_center_y = 0.5 * (side_y_min + side_y_max)
    if include_right:
        meshes.append(make_box_mesh((wall, side_depth, wall_height), (0.5 * (width - wall), side_center_y, z_center)))
    if include_left:
        meshes.append(make_box_mesh((wall, side_depth, wall_height), (-0.5 * (width - wall), side_center_y, z_center)))

    if include_top:
        meshes.append(make_box_mesh((width, depth, top_thickness), (0.0, 0.0, height - 0.5 * top_thickness)))

    return combine_meshes(meshes)


def build_ramp_channel(
    *,
    width: float,
    depth: float,
    height: float,
    wall: float,
    floor: float,
    high_end: str = "back",
    include_side_rails: bool = True,
    include_front_wall: bool = False,
    include_back_wall: bool = True,
) -> trimesh.Trimesh:
    wall = float(max(1e-4, min(wall, 0.45 * min(width, depth))))
    floor = float(max(1e-4, min(floor, 0.35 * height)))
    span_y_min = -0.5 * depth + (wall if include_back_wall else 0.0)
    span_y_max = 0.5 * depth - (wall if include_front_wall else 0.0)
    run = max(0.5 * depth, span_y_max - span_y_min)
    low_z = 0.5 * floor
    high_z = max(low_z + 0.028, height - 0.5 * floor)
    angle = math.atan2(high_z - low_z, run)
    if high_end == "back":
        angle = -angle
    elif high_end != "front":
        raise ValueError(f"Unsupported ramp high_end: {high_end}")

    ramp_center_y = 0.5 * (span_y_min + span_y_max)
    ramp_center_z = 0.5 * (low_z + high_z)
    ramp_length = math.hypot(run, high_z - low_z)
    inner_width = max(0.45 * width, width - 2.0 * wall)

    meshes: List[trimesh.Trimesh] = [
        make_box_mesh(
            (inner_width, ramp_length, floor),
            (0.0, ramp_center_y, ramp_center_z),
            rotation=trimesh.transformations.rotation_matrix(angle, (1.0, 0.0, 0.0)),
        )
    ]

    if include_side_rails:
        rail_height = max(height, high_z + 0.5 * floor)
        meshes.append(make_box_mesh((wall, depth, rail_height), (0.5 * (width - wall), 0.0, 0.5 * rail_height)))
        meshes.append(make_box_mesh((wall, depth, rail_height), (-0.5 * (width - wall), 0.0, 0.5 * rail_height)))

    if include_back_wall:
        meshes.append(make_box_mesh((width, wall, height), (0.0, -0.5 * (depth - wall), 0.5 * height)))
    if include_front_wall:
        meshes.append(make_box_mesh((width, wall, height), (0.0, 0.5 * (depth - wall), 0.5 * height)))

    return combine_meshes(meshes)


class FluidSimulator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.cache_dir = Path(args.cache_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.cache_dir)

        self.frames = int(args.frames)
        self.resolution = int(args.resolution)
        self.output_fps = int(args.fps)
        self.physics_fps = int(args.physics_fps)
        self.steps_per_frame = max(1, self.physics_fps // self.output_fps)
        self.dt = 1.0 / float(self.physics_fps)
        self.substeps = int(args.substeps)
        self.tracked_particle_count = int(args.tracked_particles)
        self.max_particles = int(args.max_particles)
        self.scene_attempts = int(args.max_retries_per_scene)
        self.random_seed = int(args.seed)
        self.rng = np.random.default_rng(self.random_seed)
        self._candidate_cache: Optional[List[ContainerCandidate]] = None
        # Keep rendering on one conservative path only. The recon branch has repeatedly
        # produced unstable colors across environments, so all RGB exports are forced to
        # use the same particle-based rendering path.
        self.render_rgb_modes = ["particle"]

        if self.tracked_particle_count > self.max_particles:
            raise ValueError(
                f"tracked_particles ({self.tracked_particle_count}) cannot exceed max_particles ({self.max_particles})."
            )
        if int(self.args.min_containers) > int(self.args.max_containers):
            raise ValueError("--min-containers cannot exceed --max-containers.")

    def _choose_liquid_preset_name(self, scene_index: int, attempt_index: int = 0) -> str:
        preset_name = str(self.args.liquid_preset)
        if preset_name == "auto":
            return LIQUID_PRESET_SEQUENCE[(scene_index + attempt_index) % len(LIQUID_PRESET_SEQUENCE)]
        if preset_name not in LIQUID_PRESETS:
            raise ValueError(f"Unknown liquid preset: {preset_name}")
        return preset_name

    def _sample_container_styles(
        self,
        rng: np.random.Generator,
        count: int,
    ) -> List[Tuple[str, Tuple[float, float, float, float], Tuple[float, float, float]]]:
        if count <= 0:
            return []
        styles: List[Tuple[str, Tuple[float, float, float, float], Tuple[float, float, float]]] = []
        for _ in range(count):
            palette_entry = CONTAINER_COLOR_PALETTES[0]
            base_rgba = np.asarray(palette_entry["rgba"], dtype=np.float64)
            jitter = np.concatenate([rng.uniform(-0.015, 0.015, size=3), np.zeros(1, dtype=np.float64)])
            rgba = np.clip(base_rgba + jitter, [0.62, 0.62, 0.60, 1.0], [0.78, 0.78, 0.76, 1.0])
            emissive = np.zeros(3, dtype=np.float64)
            styles.append(
                (
                    str(palette_entry["name"]),
                    tuple(float(v) for v in rgba.tolist()),
                    tuple(float(v) for v in emissive.tolist()),
                )
            )
        return styles

    def _sample_fluid_style(
        self,
        rng: np.random.Generator,
    ) -> Tuple[str, Tuple[float, float, float, float]]:
        palette_entry = FLUID_COLOR_PALETTES[0]
        base_rgba = np.asarray(palette_entry["rgba"], dtype=np.float64)
        jitter = np.concatenate([rng.uniform(-0.010, 0.010, size=3), np.zeros(1, dtype=np.float64)])
        rgba = np.clip(base_rgba + jitter, [0.28, 0.50, 0.80, 1.0], [0.40, 0.62, 0.92, 1.0])
        return str(palette_entry["name"]), tuple(float(v) for v in rgba.tolist())

    def _container_flow_profile(self, source_id: str) -> Dict[str, Any]:
        profile: Dict[str, Any] = {
            "receives_liquid": True,
            "entry_mode": "top",
            "speed_scale": 1.0,
            "diameter_scale": 1.0,
            "length_scale": 1.0,
            "height_range": (0.055, 0.095),
            "lateral_radius_scale": 1.0,
            "aim_z_range": (0.005, 0.030),
        }
        overrides: Dict[str, Dict[str, Any]] = {
            "open_ended_trough": {
                "entry_mode": "end_y",
                "speed_scale": 0.58,
                "diameter_scale": 0.90,
                "length_scale": 0.80,
                "height_range": (0.050, 0.082),
                "lateral_radius_scale": 0.42,
                "aim_z_range": (0.002, 0.018),
            },
            "roofed_tunnel": {
                "entry_mode": "end_y",
                "speed_scale": 0.42,
                "diameter_scale": 0.82,
                "length_scale": 0.70,
                "height_range": (0.046, 0.078),
                "lateral_radius_scale": 0.30,
                "aim_z_range": (0.000, 0.014),
            },
            "ramp_channel": {
                "entry_mode": "ramp",
                "speed_scale": 0.56,
                "diameter_scale": 0.88,
                "length_scale": 0.76,
                "height_range": (0.050, 0.082),
                "lateral_radius_scale": 0.44,
                "aim_z_range": (0.008, 0.026),
            },
            "spill_ramp": {
                "entry_mode": "ramp",
                "speed_scale": 0.48,
                "diameter_scale": 0.82,
                "length_scale": 0.62,
                "height_range": (0.042, 0.072),
                "lateral_radius_scale": 0.38,
                "aim_z_range": (0.006, 0.020),
            },
            "side_cut_box": {
                "entry_mode": "side_x",
                "speed_scale": 0.60,
                "diameter_scale": 0.90,
                "length_scale": 0.78,
                "height_range": (0.042, 0.074),
                "lateral_radius_scale": 0.42,
                "aim_z_range": (0.002, 0.018),
            },
            "sealed_box": {
                "receives_liquid": False,
            },
            "sealed_canister": {
                "receives_liquid": False,
            },
        }
        profile.update(overrides.get(source_id, {}))
        return profile

    def _select_liquid_target_indices(self, containers: Sequence[ContainerPlacement]) -> List[int]:
        target_indices = [
            idx
            for idx, container in enumerate(containers)
            if bool(self._container_flow_profile(container.prepared_container.candidate.source_id)["receives_liquid"])
        ]
        return target_indices if target_indices else list(range(len(containers)))

    def _sample_source_container(
        self,
        rng: np.random.Generator,
        target_containers: Sequence[ContainerPlacement],
    ) -> ContainerPlacement:
        candidates = [
            candidate
            for candidate in self.get_container_candidates()
            if bool(self._container_flow_profile(candidate.source_id).get("receives_liquid", True))
        ]
        # Prefer deeper source containers so the initial liquid stays contained
        # while the source is upright, instead of spilling before the pour starts.
        preferred_source_ids = {"mixing_bowl", "cylinder_cup", "tapered_beaker"}
        preferred_candidates = [candidate for candidate in candidates if str(candidate.source_id) in preferred_source_ids]
        if preferred_candidates:
            candidates = preferred_candidates
        if not candidates:
            raise ValueError("No valid source-container candidates available.")
        candidate = candidates[int(rng.integers(0, len(candidates)))]
        prepared = self.prepare_container(candidate, rng)
        source_styles = self._sample_container_styles(rng, 1)
        color_name, surface_rgba, emissive_rgb = source_styles[0]

        target_centers = np.stack([container.pos for container in target_containers], axis=0)
        scene_center = np.mean(target_centers[:, :2], axis=0)
        max_target_height = max(float(container.prepared_container.scene_extents[2]) for container in target_containers)
        max_target_radius = max(
            float(np.linalg.norm(container.pos[:2] - scene_center))
            + 0.6 * float(np.max(container.prepared_container.scene_extents[:2]))
            for container in target_containers
        )
        # Keep the source clearly visible, but do not push it so far outward that it sits
        # visually against the simulation walls when scene bounds are rendered.
        source_radius = max(0.32, max_target_radius + 0.10 + 0.32 * float(np.max(prepared.scene_extents[:2])))
        source_angle = float(rng.uniform(-0.95, 0.95))
        source_xy = scene_center + source_radius * np.array([math.cos(source_angle), math.sin(source_angle)], dtype=np.float64)
        source_z = max_target_height + float(rng.uniform(0.05, 0.09))
        yaw_deg = math.degrees(math.atan2(scene_center[1] - source_xy[1], scene_center[0] - source_xy[0]))

        return ContainerPlacement(
            prepared_container=prepared,
            pos=np.array([float(source_xy[0]), float(source_xy[1]), float(source_z)], dtype=np.float64),
            quat_wxyz=z_yaw_quat_wxyz(yaw_deg),
            yaw_deg=float(yaw_deg),
            color_name=color_name,
            surface_rgba=surface_rgba,
            emissive_rgb=emissive_rgb,
        )

    def _sample_offscreen_source_anchor(
        self,
        rng: np.random.Generator,
        target_containers: Sequence[ContainerPlacement],
    ) -> np.ndarray:
        target_centers = np.stack([container.pos for container in target_containers], axis=0)
        scene_center = np.mean(target_centers[:, :2], axis=0)
        max_target_height = max(float(container.prepared_container.scene_extents[2]) for container in target_containers)
        max_target_radius = max(
            float(np.linalg.norm(container.pos[:2] - scene_center))
            + 0.8 * float(np.max(container.prepared_container.scene_extents[:2]))
            for container in target_containers
        )
        radius = max(0.75, max_target_radius + float(rng.uniform(0.45, 0.70)))
        angle = float(rng.uniform(-1.15, 1.15))
        xy = scene_center + radius * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        z = max_target_height + float(rng.uniform(0.08, 0.14))
        return np.array([float(xy[0]), float(xy[1]), float(z)], dtype=np.float64)

    def _sample_spout_source_anchor(
        self,
        rng: np.random.Generator,
        target_containers: Sequence[ContainerPlacement],
    ) -> np.ndarray:
        target_centers = np.stack([container.pos for container in target_containers], axis=0)
        scene_center = np.mean(target_centers[:, :2], axis=0)
        max_target_height = max(float(container.prepared_container.scene_extents[2]) for container in target_containers)
        radius = max(
            0.52,
            max(float(np.linalg.norm(container.pos[:2] - scene_center)) for container in target_containers)
            + float(rng.uniform(0.24, 0.38)),
        )
        angle = float(rng.uniform(-0.75, 0.75))
        xy = scene_center + radius * np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
        z = max_target_height + float(rng.uniform(0.12, 0.18))
        return np.array([float(xy[0]), float(xy[1]), float(z)], dtype=np.float64)

    def _sample_liquid_clusters(
        self,
        *,
        rng: np.random.Generator,
        preset_name: str,
        particle_size: float,
        containers: Sequence[ContainerPlacement],
        target_container_indices: Sequence[int],
        total_particle_budget: int,
        source_container: Optional[ContainerPlacement],
        source_anchor: Optional[np.ndarray],
        source_mode: str,
        source_delay_steps: int = 0,
        attempt_index: int = 0,
    ) -> Tuple[str, float, float, float, float, int, List[LiquidCluster]]:
        preset = LIQUID_PRESETS[preset_name]
        viscosity = float(rng.uniform(*preset["viscosity"]))
        stiffness = float(rng.uniform(*preset["stiffness"]))
        exponent = float(rng.uniform(*preset["exponent"]))
        gamma = float(rng.uniform(*preset["gamma"]))
        retry_speed_scale = float(max(0.45, 1.0 - 0.10 * max(0, attempt_index)))
        retry_budget_scale = float(max(0.55, 1.0 - 0.08 * max(0, attempt_index)))

        angle_offset = float(rng.uniform(0.0, 2.0 * math.pi))
        preset_volume_tiers = tuple(float(v) for v in preset.get("volume_tiers", (0.55, 1.00, 1.70)))
        volume_tiers = [
            ("small", preset_volume_tiers[0]),
            ("medium", preset_volume_tiers[1]),
            ("large", preset_volume_tiers[2]),
        ]
        direction_modes = [
            ("top", np.array([0.00, 0.00], dtype=np.float64)),
            ("front_left", np.array([-0.10, 0.06], dtype=np.float64)),
            ("front_right", np.array([0.10, -0.06], dtype=np.float64)),
        ]
        if source_mode == "spout":
            direction_modes = [("stream", np.array([0.00, 0.00], dtype=np.float64))]
        cluster_low, cluster_high = preset["cluster_range"]
        clusters_per_container = int(rng.integers(cluster_low, cluster_high + 1))
        clusters_per_container = int(np.clip(clusters_per_container, 1, len(volume_tiers)))
        if source_mode == "spout":
            clusters_per_container = 1
        if source_mode == "spout":
            cluster_mode_indices = [0]
        elif clusters_per_container == 1:
            cluster_mode_indices = [int(rng.integers(0, len(volume_tiers)))]
        elif clusters_per_container == 2:
            cluster_mode_indices = [0, 2]
        else:
            cluster_mode_indices = list(range(len(volume_tiers)))

        active_volume_tiers = [volume_tiers[idx] for idx in cluster_mode_indices]
        active_direction_modes = [direction_modes[idx] for idx in cluster_mode_indices]
        scene_variant_name = (
            f"{len(containers)}container_{len(target_container_indices)}target_"
            f"{clusters_per_container}cluster_{len(active_direction_modes)}direction"
        )

        num_clusters = len(target_container_indices) * clusters_per_container
        min_particles_per_cluster = max(128, self.tracked_particle_count // max(num_clusters, 1))
        total_particle_budget = int(math.ceil(float(total_particle_budget) * retry_budget_scale))
        total_particle_budget = max(total_particle_budget, min_particles_per_cluster * num_clusters)

        weight_list: List[float] = []
        for _container_index in target_container_indices:
            for _volume_name, volume_scale in active_volume_tiers:
                weight_list.append(volume_scale)
        cluster_weights = np.asarray(weight_list, dtype=np.float64)
        cluster_weights /= np.sum(cluster_weights)
        raw_alloc = cluster_weights * float(total_particle_budget)
        particle_alloc = np.floor(raw_alloc).astype(np.int32)
        particle_alloc = np.maximum(min_particles_per_cluster, particle_alloc)
        alloc_gap = int(total_particle_budget - int(np.sum(particle_alloc)))
        if alloc_gap > 0:
            frac_order = np.argsort(-(raw_alloc - np.floor(raw_alloc)), kind="stable")
            for idx in frac_order[:alloc_gap]:
                particle_alloc[int(idx)] += 1
        elif alloc_gap < 0:
            remaining = -alloc_gap
            reducible = np.argsort(-particle_alloc, kind="stable")
            for idx in reducible:
                headroom = int(particle_alloc[int(idx)] - min_particles_per_cluster)
                if headroom <= 0:
                    continue
                delta = min(headroom, remaining)
                particle_alloc[int(idx)] -= delta
                remaining -= delta
                if remaining <= 0:
                    break

        clusters: List[LiquidCluster] = []
        alloc_index = 0
        cluster_index = 0
        source_pos = None if source_container is None else np.asarray(source_container.pos, dtype=np.float64)
        source_extents = (
            None
            if source_container is None
            else np.asarray(source_container.prepared_container.scene_extents, dtype=np.float64)
        )
        anchor_pos = None if source_anchor is None else np.asarray(source_anchor, dtype=np.float64)
        for target_container_index in target_container_indices:
            container = containers[target_container_index]
            flow_profile = self._container_flow_profile(container.prepared_container.candidate.source_id)
            container_extents = container.prepared_container.scene_extents
            opening_radius = float(max(0.03, 0.36 * min(container_extents[0], container_extents[1])))
            top_z = float(container.pos[2] + container_extents[2])
            lateral_radius = float(
                max(0.015, min(0.70 * opening_radius, 0.060)) * float(flow_profile["lateral_radius_scale"])
            )
            complexity_scale = 1.0 + 0.18 * max(0, clusters_per_container - 1) + 0.10 * max(0, len(containers) - 1)
            for local_idx, ((volume_name, volume_scale), (direction_name, direction_bias)) in enumerate(
                zip(active_volume_tiers, active_direction_modes)
            ):
                angle = angle_offset + local_idx * (2.0 * math.pi / max(len(volume_tiers), 1))
                radius_scale = 0.34 + 0.16 * float(rng.uniform(0.0, 1.0))
                offset_xy = radius_scale * lateral_radius * np.array(
                    [math.cos(angle), math.sin(angle)],
                    dtype=np.float64,
                )
                speed = float(rng.uniform(*preset["speed"])) * (0.72 + 0.05 * local_idx)
                speed /= math.sqrt(complexity_scale)
                speed *= float(flow_profile["speed_scale"])
                speed *= retry_speed_scale
                diameter = float(rng.uniform(*preset["diameter_scale"]) * particle_size * float(flow_profile["diameter_scale"]))
                target_particles = int(max(64, particle_alloc[alloc_index]))
                alloc_index += 1
                length = estimate_droplet_length_for_particles(
                    diameter=diameter,
                    target_particles=target_particles,
                    particle_size=particle_size,
                )
                length *= float(rng.uniform(0.78, 0.92)) * float(flow_profile["length_scale"])

                entry_mode = str(flow_profile["entry_mode"])
                height_offset = float(rng.uniform(*flow_profile["height_range"])) + 0.015 * local_idx
                aim_z_offset = float(rng.uniform(*flow_profile["aim_z_range"]))
                end_sign = -1.0 if ((target_container_index + local_idx) % 2 == 0) else 1.0

                source_anchor = None
                if source_mode in {"offscreen_stream", "offscreen_delayed", "spout"} and anchor_pos is not None:
                    source_anchor = np.asarray(anchor_pos, dtype=np.float64).copy()
                elif source_pos is not None and source_extents is not None:
                    source_anchor = np.array(
                        [
                            source_pos[0],
                            source_pos[1],
                            source_pos[2] + 0.15 * source_extents[2],
                        ],
                        dtype=np.float64,
                    )

                if source_anchor is not None:
                    used_prefilled_source = False
                    if source_mode == "spout":
                        early_pour_scale = 0.10
                        speed *= early_pour_scale
                        diameter *= 1.55
                        length *= 1.85
                        lateral_spread = max(0.016, 0.22 * opening_radius)
                        interior_xy = source_anchor[:2] + lateral_spread * np.array(
                            [float(rng.uniform(-0.35, 0.35)), float(rng.uniform(-0.35, 0.35))],
                            dtype=np.float64,
                        )
                        pos = np.array(
                            [
                                float(interior_xy[0]),
                                float(interior_xy[1]),
                                float(source_anchor[2] + 0.012 * local_idx),
                            ],
                            dtype=np.float64,
                        )
                        aim_point = np.array(
                            [
                                float(container.pos[0]),
                                float(container.pos[1]),
                                float(top_z - 0.018 + 0.20 * aim_z_offset),
                            ],
                            dtype=np.float64,
                        )
                    elif source_mode in {"offscreen_stream", "offscreen_delayed"}:
                        if local_idx == 0:
                            early_pour_scale = 0.34
                        elif local_idx == 1:
                            early_pour_scale = 0.42
                        else:
                            early_pour_scale = 0.56
                    else:
                        if local_idx == 0:
                            early_pour_scale = 0.42
                        elif local_idx == 1:
                            early_pour_scale = 0.52
                        else:
                            early_pour_scale = 0.68
                    if not used_prefilled_source:
                        speed *= early_pour_scale
                        diameter *= 1.18 if source_mode == "spout" else (0.84 if source_mode in {"offscreen_stream", "offscreen_delayed"} else 0.88)
                        length *= 1.24 if source_mode == "spout" else (0.52 if source_mode in {"offscreen_stream", "offscreen_delayed"} else 0.68)
                        aim_point = np.array(
                            [
                                container.pos[0] + direction_bias[0] * (0.04 if source_mode == "spout" else 0.10) * opening_radius,
                                container.pos[1] + direction_bias[1] * (0.04 if source_mode == "spout" else 0.10) * opening_radius,
                                top_z - (0.016 if source_mode == "spout" else 0.018) + 0.25 * aim_z_offset,
                            ],
                            dtype=np.float64,
                        )
                        lateral_jitter = (
                            0.0012 if source_mode == "spout" else (0.010 if source_mode in {"offscreen_stream", "offscreen_delayed"} else 0.015)
                        ) * opening_radius * np.array(
                            [
                                float(rng.uniform(-1.0, 1.0)),
                                float(rng.uniform(-1.0, 1.0)),
                                0.0,
                            ],
                            dtype=np.float64,
                        )
                        pos = source_anchor + lateral_jitter
                        if source_mode in {"offscreen_stream", "offscreen_delayed", "spout"}:
                            pos[2] += 0.003 * local_idx
                        else:
                            pos[2] -= 0.10 * source_extents[2]
                            pos[2] += 0.004 * local_idx
                elif entry_mode == "end_y":
                    pos = np.array(
                        [
                            container.pos[0] + 0.25 * offset_xy[0],
                            container.pos[1] + end_sign * (0.55 * container_extents[1] + 0.28 * length + 0.03),
                            container.pos[2] + 0.46 * container_extents[2] + height_offset,
                        ],
                        dtype=np.float64,
                    )
                    aim_point = np.array(
                        [
                            container.pos[0] + 0.18 * direction_bias[0] * opening_radius,
                            container.pos[1] + end_sign * 0.30 * container_extents[1],
                            container.pos[2] + 0.42 * container_extents[2] + aim_z_offset,
                        ],
                        dtype=np.float64,
                    )
                elif entry_mode == "side_x":
                    pos = np.array(
                        [
                            container.pos[0] + 0.55 * container_extents[0] + 0.25 * length + 0.03,
                            container.pos[1] + 0.25 * offset_xy[1],
                            container.pos[2] + 0.48 * container_extents[2] + height_offset,
                        ],
                        dtype=np.float64,
                    )
                    aim_point = np.array(
                        [
                            container.pos[0] + 0.18 * container_extents[0],
                            container.pos[1] + direction_bias[1] * 0.20 * opening_radius,
                            container.pos[2] + 0.42 * container_extents[2] + aim_z_offset,
                        ],
                        dtype=np.float64,
                    )
                elif entry_mode == "ramp":
                    pos = np.array(
                        [
                            container.pos[0] + 0.20 * offset_xy[0],
                            container.pos[1] - 0.28 * container_extents[1],
                            top_z + 0.5 * length + height_offset,
                        ],
                        dtype=np.float64,
                    )
                    aim_point = np.array(
                        [
                            container.pos[0] + direction_bias[0] * 0.18 * opening_radius,
                            container.pos[1] + 0.10 * container_extents[1],
                            container.pos[2] + 0.45 * container_extents[2] + aim_z_offset,
                        ],
                        dtype=np.float64,
                    )
                else:
                    height = top_z + 0.5 * length + height_offset
                    pos = np.array(
                        [container.pos[0] + offset_xy[0], container.pos[1] + offset_xy[1], height],
                        dtype=np.float64,
                    )
                    aim_point = np.array(
                        [
                            container.pos[0] + direction_bias[0] * opening_radius,
                            container.pos[1] + direction_bias[1] * opening_radius,
                            top_z + aim_z_offset,
                        ],
                        dtype=np.float64,
                    )

                direction = aim_point - pos
                jitter_xy = 0.02 / complexity_scale
                direction[0] += float(rng.uniform(-jitter_xy, jitter_xy))
                direction[1] += float(rng.uniform(-jitter_xy, jitter_xy))
                if source_anchor is not None:
                    direction[2] = min(direction[2], -0.10)
                elif entry_mode == "top":
                    direction[2] = min(direction[2], -0.12)
                else:
                    direction[2] = min(direction[2], -0.02)
                direction = normalize_vector(direction)

                clusters.append(
                    LiquidCluster(
                        cluster_index=cluster_index,
                        target_container_index=target_container_index,
                        volume_tier=volume_name,
                        direction_name=direction_name,
                        pos=pos,
                        direction=direction,
                        speed=speed,
                        diameter=diameter,
                        length=length,
                        target_particles=target_particles,
                    )
                )
                cluster_index += 1

        return scene_variant_name, viscosity, stiffness, exponent, gamma, clusters_per_container, clusters

    def _repair_container_mesh_for_render(self, mesh: trimesh.Trimesh, candidate: ContainerCandidate) -> trimesh.Trimesh:
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()

        # Use raw procedural meshes directly in Genesis. In particular, the bowl preview
        # we just inspected should go into simulation unchanged rather than being voxel-filled.
        if candidate.kind == "procedural":
            return mesh

        if mesh.is_watertight:
            return mesh

        extents = np.maximum(mesh.extents.astype(np.float64), 1e-6)
        min_extent = float(np.min(extents))
        particle_size = float(self.args.particle_size)
        voxel_pitch = float(np.clip(min(0.40 * particle_size, 0.04 * min_extent), 0.0015, 0.0060))

        try:
            repaired = voxel_fill_mesh(mesh, voxel_pitch)
        except Exception as exc:
            print(
                f"[Warn] voxel_fill_failed container={candidate.source_id} "
                f"pitch={voxel_pitch:.4f} reason={exc}"
            )
            return mesh

        if len(repaired.faces) == 0 or len(repaired.vertices) == 0:
            print(
                f"[Warn] voxel_fill_empty container={candidate.source_id} "
                f"pitch={voxel_pitch:.4f}"
            )
            return mesh

        print(
            f"[Info] voxel_filled container={candidate.source_id} "
            f"pitch={voxel_pitch:.4f} watertight_before={mesh.is_watertight} "
            f"watertight_after={repaired.is_watertight}"
        )
        return repaired

    def run(self) -> None:
        ensure_genesis_initialized(prefer_gpu=not self.args.force_cpu)

        written = 0
        for scene_index in range(self.args.num_scenes):
            scene_path = self.output_dir / f"scene_{scene_index:05d}"
            if scene_path.exists() and not self.args.overwrite:
                print(f"[Skip] scene {scene_index:05d} already exists: {scene_path}")
                written += 1
                continue

            success = False
            for attempt in range(self.scene_attempts):
                params: Optional[SceneParams] = None
                try:
                    per_scene_seed = self.random_seed + scene_index * 997 + attempt
                    scene_rng = np.random.default_rng(per_scene_seed)
                    params = self.sample_scene_params(scene_index, scene_rng, attempt_index=attempt)
                    payload = self.generate_scene(params, scene_rng)
                    rgb_variants: Dict[str, np.ndarray] = {str(self.args.fluid_vis_mode): payload["rgb"]}
                    if not self.args.skip_render:
                        for render_mode in self.render_rgb_modes:
                            if render_mode in rgb_variants:
                                continue
                            rgb_variants[render_mode] = self.render_scene_rgb_variant(params, fluid_vis_mode=render_mode)
                    payload["rgb_variants"] = rgb_variants
                    self.write_scene_folder(scene_path, payload, params, per_scene_seed)
                    print(
                        f"[OK] scene={scene_index:05d} attempt={attempt + 1} "
                        f"containers={len(params.containers)} "
                        f"labels={[container.prepared_container.candidate.label for container in params.containers]} "
                        f"file={scene_path}"
                    )
                    success = True
                    written += 1
                    break
                except SceneGenerationError as exc:
                    debug_context = ""
                    if params is not None:
                        debug_context = (
                            f" seed={per_scene_seed}"
                            f" containers={[container.prepared_container.candidate.source_id for container in params.containers]}"
                            f" viscosity={params.viscosity:.5f}"
                            f" emitter_speed={params.emitter_speed:.4f}"
                        )
                    print(
                        f"[Retry] scene={scene_index:05d} attempt={attempt + 1}/{self.scene_attempts} "
                        f"reason={exc}{debug_context}"
                    )
                except Exception as exc:
                    debug_context = ""
                    if params is not None:
                        debug_context = (
                            f" seed={per_scene_seed}"
                            f" containers={[container.prepared_container.candidate.source_id for container in params.containers]}"
                        )
                    print(
                        f"[Retry] scene={scene_index:05d} attempt={attempt + 1}/{self.scene_attempts} "
                        f"unexpected_error={exc}{debug_context}"
                    )
            if not success:
                raise RuntimeError(f"Failed to generate scene {scene_index:05d} after {self.scene_attempts} attempts.")

        print(f"[Done] Wrote {written} scene folders to {self.output_dir}")

    def get_container_candidates(self) -> List[ContainerCandidate]:
        if self._candidate_cache is not None:
            return self._candidate_cache

        mode = self.args.container_mode
        if mode == "auto":
            mode = "direct" if self.args.container_obj_dir else "procedural"

        candidates: List[ContainerCandidate] = []
        if mode == "procedural":
            candidates = self.scan_procedural_candidates()
        elif mode == "direct":
            candidates = self.scan_direct_obj_dir(Path(self.args.container_obj_dir))
        elif mode == "physxnet":
            candidates = self.scan_physxnet_candidates(Path(self.args.physx_root), self.args.physx_version)
        else:
            raise ValueError(f"Unsupported container mode: {mode}")

        if not candidates:
            raise RuntimeError("No usable container meshes were found.")

        self._candidate_cache = candidates
        print(f"[Info] Loaded {len(candidates)} container candidates from mode={mode}")
        return candidates

    def scan_procedural_candidates(self) -> List[ContainerCandidate]:
        return [
            ContainerCandidate(kind="procedural", source_id="wide_bowl", label="Wide Bowl"),
            ContainerCandidate(kind="procedural", source_id="mixing_bowl", label="Mixing Bowl"),
            ContainerCandidate(kind="procedural", source_id="cylinder_cup", label="Cylinder Cup"),
            ContainerCandidate(kind="procedural", source_id="tapered_beaker", label="Tapered Beaker"),
            ContainerCandidate(kind="procedural", source_id="open_box", label="Open Box"),
            ContainerCandidate(kind="procedural", source_id="rect_basin", label="Rect Basin"),
            ContainerCandidate(kind="procedural", source_id="open_ended_trough", label="Open Ended Trough"),
            ContainerCandidate(kind="procedural", source_id="side_cut_box", label="Side Cut Box"),
            ContainerCandidate(kind="procedural", source_id="ramp_channel", label="Ramp Channel"),
            ContainerCandidate(kind="procedural", source_id="spill_ramp", label="Spill Ramp"),
            ContainerCandidate(kind="procedural", source_id="roofed_tunnel", label="Roofed Tunnel"),
            ContainerCandidate(kind="procedural", source_id="sealed_box", label="Sealed Box"),
            ContainerCandidate(kind="procedural", source_id="sealed_canister", label="Sealed Canister"),
        ]

    def scan_direct_obj_dir(self, obj_dir: Path) -> List[ContainerCandidate]:
        if not obj_dir.exists():
            raise FileNotFoundError(f"Container OBJ directory does not exist: {obj_dir}")
        candidates = []
        for mesh_path in sorted(obj_dir.rglob("*.obj")):
            candidates.append(
                ContainerCandidate(
                    kind="direct",
                    source_id=mesh_path.stem,
                    label=mesh_path.stem,
                    obj_path=mesh_path,
                    source_up=self.args.container_mesh_up,
                )
            )
        return candidates

    def scan_physxnet_candidates(self, physx_root: Path, version: str) -> List[ContainerCandidate]:
        version_root = physx_root / version
        meta_dir = version_root / "finaljson"
        if not meta_dir.exists():
            raise FileNotFoundError(f"PhysXNet metadata directory not found: {meta_dir}")

        candidates: List[ContainerCandidate] = []
        for json_path in sorted(meta_dir.glob("*.json")):
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            object_name = str(metadata.get("object_name", "")).strip()
            category = str(metadata.get("category", "")).strip()
            text = f"{object_name} {category}".lower()
            if not is_container_like(text):
                continue

            source_id = json_path.stem
            part_dir = version_root / "partseg" / source_id / "objs"
            if not part_dir.exists():
                continue

            kept_labels = self._select_physxnet_container_part_labels(metadata)
            if not kept_labels:
                continue

            if not any((part_dir / f"{label}.obj").exists() for label in kept_labels):
                continue

            label = object_name or f"physxnet_{source_id}"
            candidates.append(
                ContainerCandidate(
                    kind="physxnet",
                    source_id=source_id,
                    label=label,
                    metadata_path=json_path,
                    source_up="y",
                )
            )

        preferred = [candidate for candidate in candidates if is_preferred_container_like(candidate.label)]
        return preferred if preferred else candidates

    def _select_physxnet_container_part_labels(self, metadata: Dict[str, Any]) -> List[int]:
        kept_labels: List[int] = []
        for part in metadata.get("parts", []):
            try:
                label = int(part.get("label"))
            except Exception:
                continue

            fields = [
                str(part.get("name", "")),
                str(part.get("material", "")),
                str(part.get("Basic_description", "")),
                str(part.get("Functional_description", "")),
                str(part.get("Movement_description", "")),
            ]
            text = " ".join(fields).lower()
            if is_liquid_like(text):
                continue
            if is_closure_like(text):
                continue
            kept_labels.append(label)

        return kept_labels

    def prepare_container(self, candidate: ContainerCandidate, rng: np.random.Generator) -> PreparedContainer:
        cache_key = f"{CONTAINER_CACHE_VERSION}_{candidate.kind}_{candidate.source_id}_{candidate.source_up}"
        cache_stub = stable_hash(cache_key)
        cache_mesh_path = self.cache_dir / f"{candidate.kind}_{candidate.source_id}_{cache_stub}.obj"
        cache_meta_path = self.cache_dir / f"{candidate.kind}_{candidate.source_id}_{cache_stub}.json"

        if cache_mesh_path.exists() and cache_meta_path.exists():
            meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            canonical_extents = np.asarray(meta["canonical_extents"], dtype=np.float64)
        else:
            mesh, physical_dims = self._load_candidate_mesh(candidate)
            mesh = canonicalize_mesh(mesh, source_up=candidate.source_up, physical_dims_m=physical_dims)
            mesh = self._repair_container_mesh_for_render(mesh, candidate)
            canonical_extents = np.maximum(mesh.extents.astype(np.float64), 1e-6)
            mesh.export(cache_mesh_path)
            cache_meta_path.write_text(
                json.dumps(
                    {
                        "source_id": candidate.source_id,
                        "label": candidate.label,
                        "canonical_extents": canonical_extents.tolist(),
                        "mesh_is_watertight": bool(mesh.is_watertight),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        target_max_extent = float(rng.uniform(0.16, 0.28))
        scale_to_scene = target_max_extent / max(float(canonical_extents.max()), 1e-6)
        return PreparedContainer(
            cache_path=cache_mesh_path,
            candidate=candidate,
            canonical_extents=canonical_extents,
            scale_to_scene=scale_to_scene,
        )

    def _load_candidate_mesh(
        self,
        candidate: ContainerCandidate,
    ) -> Tuple[trimesh.Trimesh, Optional[np.ndarray]]:
        if candidate.kind == "procedural":
            return self._build_procedural_container_mesh(candidate.source_id), None

        if candidate.kind == "direct":
            if candidate.obj_path is None:
                raise ValueError("Direct container candidate is missing obj_path.")
            return load_trimesh_single(candidate.obj_path), None

        if candidate.kind != "physxnet":
            raise ValueError(f"Unsupported candidate kind: {candidate.kind}")

        if candidate.metadata_path is None:
            raise ValueError("PhysXNet candidate is missing metadata_path.")

        metadata = json.loads(candidate.metadata_path.read_text(encoding="utf-8"))
        partseg_dir = Path(self.args.physx_root) / self.args.physx_version / "partseg" / candidate.source_id / "objs"
        kept_labels = self._select_physxnet_container_part_labels(metadata)
        meshes: List[trimesh.Trimesh] = []
        for label in kept_labels:
            mesh_path = partseg_dir / f"{label}.obj"
            if mesh_path.exists():
                try:
                    meshes.append(load_trimesh_single(mesh_path))
                except Exception:
                    continue

        if not meshes:
            raise SceneGenerationError(f"No rigid container parts could be loaded for PhysXNet object {candidate.source_id}")

        merged = trimesh.util.concatenate(meshes)
        physical_dims = parse_dim_string(metadata.get("dimension"))
        return merged, physical_dims

    def _build_procedural_container_mesh(self, source_id: str) -> trimesh.Trimesh:
        particle_size = float(self.args.particle_size)
        wall = max(0.018, 1.35 * particle_size)
        bottom = max(0.018, 1.35 * particle_size)

        if source_id == "cylinder_cup":
            profile = [
                [0.040, 0.118],
                [0.038, 0.014],
                [0.018, 0.014],
                [0.000, 0.000],
                [0.049, 0.000],
                [0.051, 0.118],
            ]
            return build_revolved_solid_open_container(profile, sections=56)

        if source_id == "tapered_beaker":
            profile = [
                [0.034, 0.125],
                [0.031, 0.014],
                [0.014, 0.014],
                [0.000, 0.000],
                [0.044, 0.000],
                [0.050, 0.125],
            ]
            return build_revolved_solid_open_container(profile, sections=56)

        if source_id == "wide_bowl":
            profile = [
                [0.082, 0.078],
                [0.062, 0.054],
                [0.040, 0.024],
                [0.018, 0.017],
                [0.000, 0.000],
                [0.056, 0.000],
                [0.074, 0.020],
                [0.094, 0.047],
                [0.108, 0.078],
            ]
            return build_revolved_solid_open_container(profile, sections=72)

        if source_id == "mixing_bowl":
            profile = [
                [0.095, 0.072],
                [0.071, 0.050],
                [0.046, 0.025],
                [0.020, 0.016],
                [0.000, 0.000],
                [0.060, 0.000],
                [0.082, 0.022],
                [0.104, 0.048],
                [0.120, 0.072],
            ]
            return build_revolved_solid_open_container(profile, sections=72)

        if source_id == "open_box":
            return build_rectangular_shell(
                width=0.124,
                depth=0.124,
                height=0.102,
                wall=wall,
                bottom=bottom,
            )

        if source_id == "rect_basin":
            return build_rectangular_shell(
                width=0.172,
                depth=0.132,
                height=0.070,
                wall=wall,
                bottom=bottom,
            )

        if source_id == "open_ended_trough":
            return build_rectangular_shell(
                width=0.164,
                depth=0.126,
                height=0.084,
                wall=wall,
                bottom=bottom,
                include_front=False,
                include_back=False,
            )

        if source_id == "side_cut_box":
            return build_rectangular_shell(
                width=0.142,
                depth=0.118,
                height=0.108,
                wall=wall,
                bottom=bottom,
                include_right=False,
            )

        if source_id == "ramp_channel":
            return build_ramp_channel(
                width=0.164,
                depth=0.142,
                height=0.108,
                wall=wall,
                floor=bottom,
                high_end="back",
                include_side_rails=True,
                include_front_wall=False,
                include_back_wall=True,
            )

        if source_id == "spill_ramp":
            return build_ramp_channel(
                width=0.156,
                depth=0.152,
                height=0.086,
                wall=wall,
                floor=bottom,
                high_end="back",
                include_side_rails=True,
                include_front_wall=False,
                include_back_wall=False,
            )

        if source_id == "roofed_tunnel":
            return build_rectangular_shell(
                width=0.154,
                depth=0.160,
                height=0.096,
                wall=wall,
                bottom=bottom,
                include_front=False,
                include_back=False,
                include_top=True,
                top=wall,
            )

        if source_id == "sealed_box":
            return finalize_mesh(trimesh.creation.box(extents=(0.118, 0.118, 0.118)))

        if source_id == "sealed_canister":
            return finalize_mesh(trimesh.creation.cylinder(radius=0.050, height=0.124, sections=56))

        raise ValueError(f"Unsupported procedural container source_id: {source_id}")

    def sample_scene_params(self, scene_index: int, rng: np.random.Generator, attempt_index: int = 0) -> SceneParams:
        candidates = self.get_container_candidates()
        source_mode = str(self.args.source_mode)
        preset_name = self._choose_liquid_preset_name(scene_index, attempt_index=attempt_index)
        allowed_container_counts = sorted(
            {
                int(count)
                for count in self.args.container_counts
                if int(self.args.min_containers) <= int(count) <= int(self.args.max_containers)
            }
        )
        allowed_container_counts = [count for count in allowed_container_counts if 1 <= count <= min(len(candidates), 3)]
        if not allowed_container_counts:
            raise ValueError("No valid container counts remain after applying --container-counts and min/max limits.")
        container_count = int(allowed_container_counts[int(rng.integers(0, len(allowed_container_counts)))])
        if len(candidates) >= container_count:
            selection_indices = rng.choice(len(candidates), size=container_count, replace=False)
            selected_candidates = [candidates[int(idx)] for idx in np.asarray(selection_indices).tolist()]
        else:
            selected_candidates = [candidates[int(rng.integers(0, len(candidates)))] for _ in range(container_count)]

        if source_mode == "spout":
            preferred_target_ids = {"wide_bowl", "mixing_bowl", "cylinder_cup", "tapered_beaker", "open_box", "rect_basin"}
            if not any(candidate.source_id in preferred_target_ids for candidate in selected_candidates):
                preferred_candidates = [candidate for candidate in candidates if candidate.source_id in preferred_target_ids]
                if preferred_candidates:
                    replace_idx = int(rng.integers(0, len(selected_candidates)))
                    selected_candidates[replace_idx] = preferred_candidates[int(rng.integers(0, len(preferred_candidates)))]
        prepared_list = [self.prepare_container(candidate, rng) for candidate in selected_candidates]

        max_container_extent = max(float(np.max(prepared.scene_extents)) for prepared in prepared_list)
        base_spacing = max(0.26, 2.2 * max_container_extent)
        if container_count == 1:
            x_positions = [0.0]
            y_positions = [0.0]
        else:
            x_positions = np.linspace(
                -0.5 * base_spacing * (container_count - 1),
                0.5 * base_spacing * (container_count - 1),
                container_count,
            )
            y_positions = [float(rng.uniform(-0.05, 0.05)) for _ in range(container_count)]

        container_styles = self._sample_container_styles(rng, container_count)
        containers: List[ContainerPlacement] = []
        for idx, prepared in enumerate(prepared_list):
            color_name, surface_rgba, emissive_rgb = container_styles[idx]
            candidate_kind = str(prepared.candidate.kind)
            yaw_deg = float(rng.uniform(-180.0, 180.0)) if candidate_kind == "physxnet" else 0.0
            containers.append(
                ContainerPlacement(
                    prepared_container=prepared,
                    pos=np.array([float(x_positions[idx]), float(y_positions[idx]), 0.0], dtype=np.float64),
                    quat_wxyz=z_yaw_quat_wxyz(yaw_deg),
                    yaw_deg=yaw_deg,
                    color_name=color_name,
                    surface_rgba=surface_rgba,
                    emissive_rgb=emissive_rgb,
                )
            )
        target_container_indices = self._select_liquid_target_indices(containers)
        source_container = None
        source_delay_steps = 0
        source_collision_proxy_size = None
        source_collision_proxy_offset = None
        if source_mode == "offscreen_stream":
            source_anchor = self._sample_offscreen_source_anchor(rng, [containers[idx] for idx in target_container_indices])
        elif source_mode == "offscreen_delayed":
            source_anchor = self._sample_offscreen_source_anchor(rng, [containers[idx] for idx in target_container_indices])
            source_delay_steps = int(max(1, self.steps_per_frame * int(self.args.source_delay_frames)))
        elif source_mode == "spout":
            primary_target = containers[int(target_container_indices[0])]
            primary_extents = np.asarray(primary_target.prepared_container.scene_extents, dtype=np.float64)
            lateral_dir = np.array(
                [
                    float(rng.uniform(-0.85, 0.85)),
                    float(rng.uniform(-0.45, 0.45)),
                ],
                dtype=np.float64,
            )
            if float(np.linalg.norm(lateral_dir)) < 1e-8:
                lateral_dir = np.array([1.0, 0.0], dtype=np.float64)
            lateral_dir /= float(np.linalg.norm(lateral_dir))
            source_anchor = np.array(
                [
                    float(primary_target.pos[0] - lateral_dir[0] * max(0.26, 1.05 * primary_extents[0])),
                    float(primary_target.pos[1] - lateral_dir[1] * max(0.20, 0.90 * primary_extents[1])),
                    float(primary_target.pos[2] + 1.15 * primary_extents[2]),
                ],
                dtype=np.float64,
            )
            source_delay_steps = int(max(1, self.steps_per_frame * int(self.args.source_delay_frames)))
        else:
            source_anchor = self._sample_offscreen_source_anchor(rng, [containers[idx] for idx in target_container_indices])

        prepared = containers[0].prepared_container
        container_extents = prepared.scene_extents
        max_extent = max(float(np.max(container.prepared_container.scene_extents)) for container in containers)
        top_z = max(
            max(float(container.pos[2] + container.prepared_container.scene_extents[2]) for container in containers),
            float(source_anchor[2]),
        )
        bowl_radius = max(float(0.5 * np.linalg.norm(container.prepared_container.scene_extents[:2])) for container in containers)
        preset = LIQUID_PRESETS[preset_name]
        volume_low, volume_high = tuple(float(v) for v in self.args.liquid_volume_range)
        budget_scale = (
            float(rng.uniform(*preset["budget_scale"]))
            * float(rng.uniform(volume_low, volume_high))
            * float(self.args.liquid_volume_scale)
        )
        minimum_scene_budget = int(
            math.ceil(max(1, len(target_container_indices)) * 3 * 320 * float(self.args.liquid_volume_scale))
        )
        target_particle_budget = max(
            self.tracked_particle_count,
            int(math.ceil(self.tracked_particle_count * budget_scale)),
            minimum_scene_budget,
        )
        target_particle_budget = min(target_particle_budget, max(self.tracked_particle_count, self.max_particles - 512))

        particle_size = float(self.args.particle_size)
        (
            scene_variant_name,
            viscosity,
            stiffness,
            exponent,
            gamma,
            clusters_per_container,
            liquid_clusters,
        ) = self._sample_liquid_clusters(
            rng=rng,
            preset_name=preset_name,
            particle_size=particle_size,
            containers=containers,
            target_container_indices=target_container_indices,
            total_particle_budget=target_particle_budget,
            source_container=source_container,
            source_anchor=source_anchor,
            source_mode=source_mode,
            source_delay_steps=source_delay_steps,
            attempt_index=attempt_index,
        )
        emission_interval_steps = max(1, min(4, self.steps_per_frame // 3))
        emitter_speed = float(np.mean([cluster.speed for cluster in liquid_clusters]))
        emitter_diameter = float(np.mean([cluster.diameter for cluster in liquid_clusters]))
        emitter_length = float(np.mean([cluster.length for cluster in liquid_clusters]))
        estimate_length = emitter_length
        estimated_particles_per_emit = estimate_particles_per_emit(
            diameter=emitter_diameter,
            length=estimate_length,
            particle_size=particle_size,
        )

        max_cluster_height = max(float(cluster.pos[2] + 0.5 * cluster.length) for cluster in liquid_clusters)
        emitter_pos = np.array(
            [
                float(np.mean([cluster.pos[0] for cluster in liquid_clusters])),
                float(np.mean([cluster.pos[1] for cluster in liquid_clusters])),
                float(np.mean([cluster.pos[2] for cluster in liquid_clusters])),
            ],
            dtype=np.float64,
        )
        emitter_direction = normalize_vector(np.mean([cluster.direction for cluster in liquid_clusters], axis=0))

        all_scene_xy = np.stack([container.pos[:2] for container in containers] + [source_anchor[:2]], axis=0)
        scene_center_xy = np.mean(all_scene_xy, axis=0)
        spread_xy = max(
            0.0,
            float(
                np.max(
                    np.linalg.norm(
                        all_scene_xy - scene_center_xy[None, :],
                        axis=1,
                    )
                )
            ),
        )
        camera_azimuth_deg = float(rng.uniform(0.0, 360.0))
        camera_elevation_deg = float(rng.uniform(26.0, 35.0))
        camera_radius = float(
            float(self.args.camera_distance_scale) * max(1.10, spread_xy + rng.uniform(3.2, 4.0) * max_extent)
        )
        camera_fov_deg = float(self.args.camera_fov)
        lookat = np.array([float(scene_center_xy[0]), float(scene_center_xy[1]), 0.48 * top_z], dtype=np.float64)
        camera_pos = self._spherical_camera(camera_radius, camera_azimuth_deg, camera_elevation_deg, lookat)

        if source_mode == "spout" and source_anchor is not None and target_container_indices:
            primary_target = containers[int(target_container_indices[0])]
            target_top = float(primary_target.pos[2] + primary_target.prepared_container.scene_extents[2])
            target_focus = np.array(
                [float(primary_target.pos[0]), float(primary_target.pos[1]), target_top - 0.02],
                dtype=np.float64,
            )
            source_to_target = target_focus - source_anchor
            horizontal_dir = np.array([float(source_to_target[0]), float(source_to_target[1]), 0.0], dtype=np.float64)
            horizontal_norm = float(np.linalg.norm(horizontal_dir))
            if horizontal_norm < 1e-8:
                horizontal_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                horizontal_dir /= horizontal_norm
            side_dir = np.array([-horizontal_dir[1], horizontal_dir[0], 0.0], dtype=np.float64)
            side_dir *= -1.0 if float(rng.uniform(0.0, 1.0)) < 0.5 else 1.0
            lookat = 0.82 * target_focus + 0.18 * np.asarray(source_anchor, dtype=np.float64)
            lookat[2] = max(0.09, 0.82 * target_top + 0.18 * float(source_anchor[2]))
            source_target_span = float(np.linalg.norm(target_focus - source_anchor))
            camera_elevation_deg = float(rng.uniform(18.0, 24.0))
            camera_radius = float(
                float(self.args.camera_distance_scale)
                * max(0.58, 0.72 * source_target_span + rng.uniform(1.10, 1.40) * max_extent)
            )
            horizontal_radius = camera_radius * math.cos(math.radians(camera_elevation_deg))
            vertical_radius = camera_radius * math.sin(math.radians(camera_elevation_deg))
            camera_pos = lookat + side_dir * horizontal_radius + np.array([0.0, 0.0, vertical_radius], dtype=np.float64)
            camera_azimuth_deg = float(math.degrees(math.atan2(camera_pos[1] - lookat[1], camera_pos[0] - lookat[0])))
            camera_fov_deg = min(max(camera_fov_deg, 42.0), 52.0)

        source_tilt_schedule_deg = None
        source_rest_quat_wxyz = None
        if source_mode == "spout":
            source_tilt_schedule_deg = None
            source_rest_quat_wxyz = None

        furthest_container_xy = max(
            [float(np.linalg.norm(container.pos[:2] - scene_center_xy)) for container in containers]
            + [float(np.linalg.norm(source_anchor[:2] - scene_center_xy))]
        )
        scene_radius = max(1.00, furthest_container_xy + 4.5 * max_extent)
        scene_height = max(1.25, max_cluster_height + 3.0 * max_extent)
        scene_bounds_lower = np.array(
            [float(scene_center_xy[0] - scene_radius), float(scene_center_xy[1] - scene_radius), -0.12],
            dtype=np.float64,
        )
        scene_bounds_upper = np.array(
            [float(scene_center_xy[0] + scene_radius), float(scene_center_xy[1] + scene_radius), scene_height],
            dtype=np.float64,
        )

        cluster_extent_points: List[np.ndarray] = [np.asarray(source_anchor, dtype=np.float64)]
        if source_container is not None:
            source_center = np.asarray(source_container.pos, dtype=np.float64)
            source_size = np.asarray(source_container.prepared_container.scene_extents, dtype=np.float64)
            cluster_extent_points.extend(
                [
                    source_center,
                    source_center + np.array([source_size[0], 0.0, 0.0], dtype=np.float64),
                    source_center - np.array([source_size[0], 0.0, 0.0], dtype=np.float64),
                    source_center + np.array([0.0, source_size[1], 0.0], dtype=np.float64),
                    source_center - np.array([0.0, source_size[1], 0.0], dtype=np.float64),
                    source_center + np.array([0.0, 0.0, source_size[2]], dtype=np.float64),
                ]
            )
        for cluster in liquid_clusters:
            cluster_pos = np.asarray(cluster.pos, dtype=np.float64)
            cluster_dir = normalize_vector(np.asarray(cluster.direction, dtype=np.float64))
            half_length = 0.5 * float(max(cluster.length, cluster.diameter))
            radial_pad = 2.5 * float(cluster.diameter)
            cluster_extent_points.extend(
                [
                    cluster_pos,
                    cluster_pos - half_length * cluster_dir,
                    cluster_pos + half_length * cluster_dir,
                    cluster_pos + np.array([radial_pad, 0.0, 0.0], dtype=np.float64),
                    cluster_pos - np.array([radial_pad, 0.0, 0.0], dtype=np.float64),
                    cluster_pos + np.array([0.0, radial_pad, 0.0], dtype=np.float64),
                    cluster_pos - np.array([0.0, radial_pad, 0.0], dtype=np.float64),
                ]
            )
        cluster_extent_points_np = np.stack(cluster_extent_points, axis=0)
        scene_margin_xy = max(0.12, 2.5 * max_extent)
        scene_bounds_lower[:2] = np.minimum(
            scene_bounds_lower[:2],
            np.min(cluster_extent_points_np[:, :2], axis=0) - scene_margin_xy,
        )
        scene_bounds_upper[:2] = np.maximum(
            scene_bounds_upper[:2],
            np.max(cluster_extent_points_np[:, :2], axis=0) + scene_margin_xy,
        )
        scene_bounds_upper[2] = max(
            float(scene_bounds_upper[2]),
            float(np.max(cluster_extent_points_np[:, 2]) + max(0.18, 1.6 * max_extent)),
        )
        # Also expand the Z lower bound from cluster extent points.
        # The original code only hardcodes -0.12 and never updates it from clusters.
        # Long droplets with near-horizontal directions can place particles well below -0.12.
        scene_bounds_lower[2] = min(
            float(scene_bounds_lower[2]),
            float(np.min(cluster_extent_points_np[:, 2]) - max(0.18, 1.6 * max_extent)),
        )
        if source_container is not None:
            source_center = np.asarray(source_container.pos, dtype=np.float64)
            source_size = np.asarray(source_container.prepared_container.scene_extents, dtype=np.float64)
            # Give the source bowl/cup extra room so it does not look embedded in the
            # rendered scene-bound walls at the first frame.
            source_wall_gap_xy = max(0.22, 1.15 * float(np.max(source_size[:2])) + 0.06)
            source_min_xy = source_center[:2] - source_size[:2] - source_wall_gap_xy
            source_max_xy = source_center[:2] + source_size[:2] + source_wall_gap_xy
            scene_bounds_lower[:2] = np.minimum(scene_bounds_lower[:2], source_min_xy)
            scene_bounds_upper[:2] = np.maximum(scene_bounds_upper[:2], source_max_xy)
        # Safety margin: account for droplet radial extent along Z (non-vertical directions
        # project diameter onto Z axis) and multi-pulse position spread beyond half_length.
        _bounds_safety = max(0.20, float(particle_size) * 12.0)
        scene_bounds_lower -= _bounds_safety
        scene_bounds_upper += _bounds_safety
        tracking_bounds_lower = np.array(
            [
                float(scene_center_xy[0] - max(0.55, furthest_container_xy + 2.0 * bowl_radius)),
                float(scene_center_xy[1] - max(0.55, furthest_container_xy + 2.0 * bowl_radius)),
                -0.06,
            ],
            dtype=np.float64,
        )
        tracking_bounds_upper = np.array(
            [
                float(scene_center_xy[0] + max(0.55, furthest_container_xy + 2.0 * bowl_radius)),
                float(scene_center_xy[1] + max(0.55, furthest_container_xy + 2.0 * bowl_radius)),
                max(0.75, max_cluster_height + 1.5 * max_extent),
            ],
            dtype=np.float64,
        )

        emission_steps_total = max(1, len(liquid_clusters) * emission_interval_steps)
        solver_substeps = int(self.substeps)
        if self.args.solver == "sph":
            solver_substeps = max(
                solver_substeps,
                32 + 8 * max(0, clusters_per_container - 1) + 6 * max(0, len(containers) - 1),
            )
        gravity = (0.0, 0.0, -9.81)
        fluid_color_name, fluid_rgba = self._sample_fluid_style(rng)

        return SceneParams(
            scene_index=scene_index,
            preset_name=preset_name,
            scene_variant_name=scene_variant_name,
            viscosity=viscosity,
            stiffness=stiffness,
            exponent=exponent,
            gamma=gamma,
            emitter_speed=emitter_speed,
            emitter_diameter=emitter_diameter,
            emitter_length=emitter_length,
            camera_azimuth_deg=camera_azimuth_deg,
            camera_elevation_deg=camera_elevation_deg,
            camera_radius=camera_radius,
            camera_fov_deg=camera_fov_deg,
            gravity=gravity,
            prepared_container=prepared,
            solver_name=self.args.solver,
            solver_substeps=solver_substeps,
            scene_bounds_lower=scene_bounds_lower,
            scene_bounds_upper=scene_bounds_upper,
            emitter_pos=emitter_pos,
            emitter_direction=emitter_direction,
            emission_interval_steps=emission_interval_steps,
            camera_pos=camera_pos,
            camera_lookat=lookat,
            container_pos=np.array([float(scene_center_xy[0]), float(scene_center_xy[1]), 0.0], dtype=np.float64),
            emission_steps_total=emission_steps_total,
            clusters_per_container=clusters_per_container,
            tracking_bounds_lower=tracking_bounds_lower,
            tracking_bounds_upper=tracking_bounds_upper,
            estimated_particles_per_emit=estimated_particles_per_emit,
            target_particle_budget=target_particle_budget,
            containers=containers,
            liquid_clusters=liquid_clusters,
            source_container=source_container,
            source_anchor=source_anchor,
            source_mode=source_mode,
            source_delay_steps=source_delay_steps,
            fluid_color_name=fluid_color_name,
            fluid_rgba=fluid_rgba,
            source_tilt_schedule_deg=source_tilt_schedule_deg,
            source_rest_quat_wxyz=source_rest_quat_wxyz,
            source_collision_proxy_size=source_collision_proxy_size,
            source_collision_proxy_offset=source_collision_proxy_offset,
        )

    def _spherical_camera(
        self,
        radius: float,
        azimuth_deg: float,
        elevation_deg: float,
        lookat: np.ndarray,
    ) -> np.ndarray:
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

    def _make_scene(self, params: SceneParams, fluid_vis_mode: Optional[str] = None):
        gs_mod = _import_genesis()
        max_extent = max(float(np.max(container.prepared_container.scene_extents)) for container in params.containers)

        scene_kwargs: Dict[str, Any] = dict(
            sim_options=gs_mod.options.SimOptions(
                dt=self.dt,
                substeps=int(params.solver_substeps),
                gravity=params.gravity,
                floor_height=-0.003,
            ),
            vis_options=gs_mod.options.VisOptions(
                background_color=(0.95, 0.95, 0.95),
                ambient_light=(0.35, 0.35, 0.35),
                visualize_sph_boundary=False,
                visualize_pbd_boundary=False,
            ),
            viewer_options=gs_mod.options.ViewerOptions(
                camera_pos=tuple(params.camera_pos.tolist()),
                camera_lookat=tuple(params.camera_lookat.tolist()),
                camera_fov=float(params.camera_fov_deg),
                max_FPS=120,
            ),
            profiling_options=gs_mod.options.ProfilingOptions(
                show_FPS=False,
            ),
            rigid_options=gs_mod.options.RigidOptions(
                use_gjk_collision=True,
            ),
            show_viewer=bool(self.args.show_viewer),
        )
        if params.solver_name == "sph":
            scene_kwargs["sph_options"] = gs_mod.options.SPHOptions(
                lower_bound=tuple(params.scene_bounds_lower.tolist()),
                upper_bound=tuple(params.scene_bounds_upper.tolist()),
                particle_size=float(self.args.particle_size),
                pressure_solver="DFSPH",
            )
        elif params.solver_name == "pbd":
            scene_kwargs["pbd_options"] = gs_mod.options.PBDOptions(
                lower_bound=tuple(params.scene_bounds_lower.tolist()),
                upper_bound=tuple(params.scene_bounds_upper.tolist()),
                particle_size=float(self.args.particle_size),
                max_density_solver_iterations=10,
                max_viscosity_solver_iterations=2,
            )
        else:  # pragma: no cover - parser blocks this
            raise ValueError(f"Unsupported solver: {params.solver_name}")

        scene = gs_mod.Scene(**scene_kwargs)

        scene.add_entity(
            morph=gs_mod.morphs.Plane(pos=(0.0, 0.0, -0.003), fixed=True),
            material=gs_mod.materials.Rigid(friction=1.0, needs_coup=True),
            surface=gs_mod.surfaces.Default(color=(0.84, 0.84, 0.84, 1.0), vis_mode="visual"),
        )

        wall_thickness = max(0.02, 2.5 * float(self.args.particle_size))
        bounds_center = 0.5 * (params.scene_bounds_lower + params.scene_bounds_upper)
        bounds_size = np.maximum(params.scene_bounds_upper - params.scene_bounds_lower, 1e-3)
        if bool(self.args.render_scene_bounds):
            visual_wall_margin = max(6.0, 3.0 * float(np.max(bounds_size[:2])))
            visual_wall_center = np.array(
                [bounds_center[0], bounds_center[1], 0.55 * bounds_size[2]],
                dtype=np.float64,
            )
            visual_wall_height = max(float(bounds_size[2]), 1.6)
            visual_wall_span_x = float(bounds_size[0] + 2.0 * visual_wall_margin)
            visual_wall_span_y = float(bounds_size[1] + 2.0 * visual_wall_margin)
            wall_specs = [
                ((wall_thickness, visual_wall_span_y, visual_wall_height), (params.scene_bounds_lower[0] - visual_wall_margin, visual_wall_center[1], visual_wall_center[2])),
                ((wall_thickness, visual_wall_span_y, visual_wall_height), (params.scene_bounds_upper[0] + visual_wall_margin, visual_wall_center[1], visual_wall_center[2])),
                ((visual_wall_span_x, wall_thickness, visual_wall_height), (visual_wall_center[0], params.scene_bounds_lower[1] - visual_wall_margin, visual_wall_center[2])),
                ((visual_wall_span_x, wall_thickness, visual_wall_height), (visual_wall_center[0], params.scene_bounds_upper[1] + visual_wall_margin, visual_wall_center[2])),
            ]
            wall_surface = gs_mod.surfaces.Default(color=(0.88, 0.88, 0.88, 1.0), vis_mode="visual")
            for wall_size, wall_pos in wall_specs:
                scene.add_entity(
                    morph=gs_mod.morphs.Box(
                        size=tuple(float(v) for v in wall_size),
                        pos=tuple(float(v) for v in wall_pos),
                        fixed=True,
                    ),
                    material=gs_mod.materials.Rigid(
                        friction=0.9,
                        needs_coup=True,
                        coup_friction=0.2,
                    ),
                    surface=wall_surface,
                )

        for container in params.containers:
            scene.add_entity(
                morph=gs_mod.morphs.Mesh(
                    file=str(container.prepared_container.cache_path),
                    pos=tuple(container.pos.tolist()),
                    quat=container.quat_wxyz,
                    scale=float(container.prepared_container.scale_to_scene),
                    fixed=True,
                    convexify=False,
                    merge_submeshes_for_collision=True,
                    file_meshes_are_zup=True,
                ),
                material=gs_mod.materials.Rigid(
                    friction=0.9,
                    needs_coup=True,
                    coup_friction=0.2,
                    sdf_cell_size=max(0.0025, 0.02 * max_extent),
                ),
                surface=gs_mod.surfaces.Default(
                    color=container.surface_rgba,
                    vis_mode="visual",
                ),
            )

        source_entity = None
        if params.source_container is not None:
            source = params.source_container
            source_entity = scene.add_entity(
                morph=gs_mod.morphs.Mesh(
                    file=str(source.prepared_container.cache_path),
                    pos=tuple(source.pos.tolist()),
                    quat=source.quat_wxyz,
                    scale=float(source.prepared_container.scale_to_scene),
                    fixed=True,
                    convexify=False,
                    merge_submeshes_for_collision=True,
                    file_meshes_are_zup=True,
                ),
                material=gs_mod.materials.Rigid(
                    friction=0.9,
                    needs_coup=True,
                    coup_friction=0.2,
                    sdf_cell_size=max(0.0025, 0.02 * max_extent),
                ),
                surface=gs_mod.surfaces.Default(
                    color=source.surface_rgba,
                    vis_mode="visual",
                ),
            )

        fluid_material = self._make_fluid_material(params)
        fluid_surface = self._make_fluid_surface(gs_mod, params, fluid_vis_mode="particle")
        emitter = None
        fluid_entity = None
        emitter = scene.add_emitter(
            material=fluid_material,
            max_particles=self.max_particles,
            surface=fluid_surface,
        )
        fluid_entity = emitter.entity

        camera = None
        if not self.args.skip_render:
            camera = scene.add_camera(
                res=(self.resolution, self.resolution),
                pos=tuple(params.camera_pos.tolist()),
                lookat=tuple(params.camera_lookat.tolist()),
                up=(0.0, 0.0, 1.0),
                fov=float(params.camera_fov_deg),
                near=0.05,
                far=max(3.0, float(params.scene_bounds_upper[2] + 1.5)),
                GUI=bool(self.args.show_viewer),
            )

        if self.args.skip_render:
            # Genesis builds the visualizer unconditionally inside `scene.build()`.
            # Skip that stage on headless machines when we only want physics validation.
            scene._visualizer.build = lambda: None

        scene.build()
        return scene, emitter, fluid_entity, camera, source_entity

    def _make_fluid_surface(self, gs_mod, params: SceneParams, fluid_vis_mode: Optional[str] = None):
        return gs_mod.surfaces.Default(
            color=(
                float(params.fluid_rgba[0]),
                float(params.fluid_rgba[1]),
                float(params.fluid_rgba[2]),
                1.0,
            ),
            vis_mode="particle",
        )

    def _make_fluid_material(self, params: SceneParams):
        gs_mod = _import_genesis()
        if params.solver_name == "sph":
            effective_mu = get_effective_solver_viscosity(params.solver_name, params.viscosity)
            return gs_mod.materials.SPH.Liquid(
                rho=1000.0,
                stiffness=float(params.stiffness),
                exponent=float(params.exponent),
                mu=effective_mu,
                gamma=float(params.gamma),
                sampler="regular",
            )
        return gs_mod.materials.PBD.Liquid(
            rho=1000.0,
            sampler="random",
            density_relaxation=0.8,
            viscosity_relaxation=params.viscosity,
        )

    def generate_scene(self, params: SceneParams, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        scene, emitter, fluid, camera, source_entity = self._make_scene(params)
        pulse_schedule: Sequence[EmissionPulse] = ()
        if emitter is not None:
            pulse_schedule = self._build_emission_schedule(params)
        capture_start_step = (
            max(1, int(params.source_delay_steps) - self.steps_per_frame)
            if emitter is None and params.source_mode == "spout" and params.source_container is not None
            else self._estimate_capture_start_step(params, pulse_schedule)
        )
        global_step = self._advance_scene_to_step(
            scene=scene,
            emitter=emitter,
            fluid=fluid,
            source_entity=source_entity,
            params=params,
            pulse_schedule=pulse_schedule,
            start_step=0,
            end_step_exclusive=capture_start_step,
            emit_enabled=emitter is not None,
        )

        tracked_ids = self._choose_tracked_ids(fluid, params, rng)
        while tracked_ids is None and global_step < max(capture_start_step + self.steps_per_frame * 4, max(len(pulse_schedule), 1) * 2):
            global_step = self._advance_scene_to_step(
                scene=scene,
                emitter=emitter,
                fluid=fluid,
                source_entity=source_entity,
                params=params,
                pulse_schedule=pulse_schedule,
                start_step=global_step,
                end_step_exclusive=global_step + max(1, params.emission_interval_steps),
                emit_enabled=emitter is not None,
            )
            tracked_ids = self._choose_tracked_ids(fluid, params, rng)
        if tracked_ids is None:
            raise SceneGenerationError("Unable to lock stable fluid particles during the early pouring phase.")

        rgb_frames = np.empty(
            (self.frames, self.resolution, self.resolution, 3),
            dtype=np.uint8,
        )
        depth_frames = np.empty(
            (self.frames, self.resolution, self.resolution, 1),
            dtype=np.float32,
        )
        pos_frames = np.empty((self.frames, self.tracked_particle_count, 3), dtype=np.float32)
        vel_frames = np.empty((self.frames, self.tracked_particle_count, 3), dtype=np.float32)

        self._capture_frame(
            frame_idx=0,
            fluid=fluid,
            tracked_ids=tracked_ids,
            camera=camera,
            rgb_frames=rgb_frames,
            depth_frames=depth_frames,
            pos_frames=pos_frames,
            vel_frames=vel_frames,
            params=params,
        )

        for frame_idx in range(1, self.frames):
            global_step = self._advance_scene_to_step(
                scene=scene,
                emitter=emitter,
                fluid=fluid,
                source_entity=source_entity,
                params=params,
                pulse_schedule=pulse_schedule,
                start_step=global_step,
                end_step_exclusive=global_step + self.steps_per_frame,
                emit_enabled=emitter is not None,
            )
            self._capture_frame(
                frame_idx=frame_idx,
                fluid=fluid,
                tracked_ids=tracked_ids,
                camera=camera,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                pos_frames=pos_frames,
                vel_frames=vel_frames,
                params=params,
            )

        return {
            "rgb": rgb_frames,
            "depth": depth_frames,
            "particles_pos": pos_frames,
            "particles_vel": vel_frames,
            "particles_ids": tracked_ids.astype(np.int32),
        }

    def render_scene_rgb_variant(self, params: SceneParams, fluid_vis_mode: str) -> np.ndarray:
        scene, emitter, fluid, camera, source_entity = self._make_scene(params, fluid_vis_mode="particle")
        if camera is None:
            return np.zeros((self.frames, self.resolution, self.resolution, 3), dtype=np.uint8)

        pulse_schedule: Sequence[EmissionPulse] = ()
        if emitter is not None:
            pulse_schedule = self._build_emission_schedule(params)
        capture_start_step = (
            max(1, int(params.source_delay_steps) - self.steps_per_frame)
            if emitter is None and params.source_mode == "spout" and params.source_container is not None
            else self._estimate_capture_start_step(params, pulse_schedule)
        )
        global_step = self._advance_scene_to_step(
            scene=scene,
            emitter=emitter,
            fluid=fluid,
            source_entity=source_entity,
            params=params,
            pulse_schedule=pulse_schedule,
            start_step=0,
            end_step_exclusive=capture_start_step,
            emit_enabled=emitter is not None,
        )
        rgb_frames = np.empty((self.frames, self.resolution, self.resolution, 3), dtype=np.uint8)

        rgb_raw, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
        rgb_frames[0] = self._decorate_rgb_frame(rgb_to_uint8(to_numpy(rgb_raw)), params)

        for frame_idx in range(1, self.frames):
            global_step = self._advance_scene_to_step(
                scene=scene,
                emitter=emitter,
                fluid=fluid,
                source_entity=source_entity,
                params=params,
                pulse_schedule=pulse_schedule,
                start_step=global_step,
                end_step_exclusive=global_step + self.steps_per_frame,
                emit_enabled=emitter is not None,
            )
            rgb_raw, _, _, _ = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
            rgb_frames[frame_idx] = self._decorate_rgb_frame(rgb_to_uint8(to_numpy(rgb_raw)), params)

        return rgb_frames

    def _should_emit(self, global_step: int, params: SceneParams) -> bool:
        if params.emission_interval_steps <= 0:
            return False
        if global_step >= params.emission_steps_total:
            return False
        return (global_step % params.emission_interval_steps) == 0

    def _emit_clusters_once(self, emitter, params: SceneParams) -> None:
        for cluster in params.liquid_clusters:
            self._emit_cluster(emitter, cluster)

    def _cluster_emit_pulse_count(self, cluster: LiquidCluster) -> int:
        diameter = max(float(cluster.diameter), 1e-6)
        length = max(float(cluster.length), 0.0)
        max_pulse_length = float(np.clip(2.4 * diameter, 0.05, 0.12))
        length_pulses = max(1, int(math.ceil(length / max_pulse_length)))
        particle_pulses = max(1, int(math.ceil(float(cluster.target_particles) / 420.0)))
        return max(length_pulses, particle_pulses)

    def _emit_cluster_pulse(self, emitter, cluster: LiquidCluster, pulse_count: int, pulse_index: int) -> None:
        pulse_count = max(1, int(pulse_count))
        if pulse_count == 1:
            self._emit_cluster(emitter, cluster)
            return

        pulse_length = float(cluster.length) / float(pulse_count)
        pulse_spacing = 0.85 * pulse_length
        center_offset = (pulse_index - 0.5 * (pulse_count - 1)) * pulse_spacing
        pulse_pos = cluster.pos - center_offset * cluster.direction
        pulse_speed = float(cluster.speed) * (0.94 if pulse_count > 2 else 0.97)

        emit_kwargs = dict(
            droplet_shape="circle",
            droplet_size=float(cluster.diameter),
            pos=tuple(np.asarray(pulse_pos, dtype=np.float64).tolist()),
            direction=tuple(cluster.direction.tolist()),
            speed=pulse_speed,
            p_size=float(self.args.particle_size),
        )
        if pulse_length > 0.0:
            emit_kwargs["droplet_length"] = pulse_length
        emitter.emit(**emit_kwargs)

    def _build_emission_schedule(self, params: SceneParams) -> List[EmissionPulse]:
        cluster_pulses: List[List[EmissionPulse]] = []
        base_stride = max(2, int(params.emission_interval_steps))
        for cluster in params.liquid_clusters:
            pulse_count = self._cluster_emit_pulse_count(cluster)
            cluster_events: List[EmissionPulse] = []
            for pulse_idx in range(pulse_count):
                estimated_particles = int(math.ceil(float(cluster.target_particles) / float(pulse_count)))
                cluster_events.append(
                    EmissionPulse(
                        emit_step=0,
                        cluster_index=int(cluster.cluster_index),
                        pulse_count=pulse_count,
                        pulse_index=pulse_idx,
                        estimated_particles=estimated_particles,
                    )
                )
            cluster_pulses.append(cluster_events)

        pulse_schedule: List[EmissionPulse] = []
        max_rounds = max((len(events) for events in cluster_pulses), default=0)
        for round_idx in range(max_rounds):
            for cluster_events in cluster_pulses:
                if round_idx >= len(cluster_events):
                    continue
                event = cluster_events[round_idx]
                pulse_schedule.append(
                    EmissionPulse(
                        emit_step=len(pulse_schedule) * base_stride,
                        cluster_index=event.cluster_index,
                        pulse_count=event.pulse_count,
                        pulse_index=event.pulse_index,
                        estimated_particles=event.estimated_particles,
                    )
                )
        return pulse_schedule

    def _estimate_capture_start_step(self, params: SceneParams, pulse_schedule: Sequence[EmissionPulse]) -> int:
        if not pulse_schedule:
            return max(1, self.steps_per_frame)
        particle_threshold = max(
            int(math.ceil(2.25 * float(self.tracked_particle_count))),
            min(self.max_particles // 6, 2048),
        )
        cumulative_particles = 0
        capture_step = pulse_schedule[-1].emit_step
        for event in pulse_schedule:
            if int(event.emit_step) < int(params.source_delay_steps):
                continue
            cumulative_particles += int(event.estimated_particles)
            if cumulative_particles >= particle_threshold:
                capture_step = int(event.emit_step)
                break
        settle_steps = max(self.steps_per_frame + 2, 2 * self.steps_per_frame)
        return max(1, capture_step + settle_steps)

    def _advance_scene_to_step(
        self,
        *,
        scene,
        emitter,
        fluid,
        source_entity=None,
        params: SceneParams,
        pulse_schedule: Sequence[EmissionPulse],
        start_step: int,
        end_step_exclusive: int,
        emit_enabled: bool = True,
    ) -> int:
        pulse_idx = 0
        while pulse_idx < len(pulse_schedule) and int(pulse_schedule[pulse_idx].emit_step) < start_step:
            pulse_idx += 1

        for global_step in range(start_step, max(start_step, end_step_exclusive)):
            if source_entity is not None and params.source_mode == "spout" and params.source_tilt_schedule_deg is not None:
                tilt_cfg = np.asarray(params.source_tilt_schedule_deg, dtype=np.float64)
                # Rotate around the local lip axis perpendicular to the pour direction.
                # The previous sign choice tipped the container the wrong way, so the
                # source effectively started in an already-pouring configuration.
                tilt_axis = np.array([tilt_cfg[1], -tilt_cfg[0], 0.0], dtype=np.float64)
                # Keep the source upright briefly so the visible particle clump starts
                # above the lip, then begins falling before the pour rotation ramps up.
                tilt_delay_steps = float(max(int(params.source_delay_steps), self.steps_per_frame * 6))
                tilt_progress = min(
                    1.0,
                    max(0.0, float(global_step) - tilt_delay_steps) / max(1.0, float(self.steps_per_frame * 12)),
                )
                tilt_deg = float((1.0 - tilt_progress) * tilt_cfg[2] + tilt_progress * tilt_cfg[3])
                base_yaw = float(params.source_container.yaw_deg if params.source_container is not None else 0.0)
                base_quat = z_yaw_quat_wxyz(base_yaw)
                tilt_quat = axis_angle_quat_wxyz(tilt_axis, tilt_deg)
                source_entity.set_quat(quat_mul_wxyz(tilt_quat, base_quat), zero_velocity=True)
            while emit_enabled and pulse_idx < len(pulse_schedule) and int(pulse_schedule[pulse_idx].emit_step) == global_step:
                event = pulse_schedule[pulse_idx]
                if global_step >= int(params.source_delay_steps):
                    cluster = params.liquid_clusters[int(event.cluster_index)]
                    self._emit_cluster_pulse(
                        emitter,
                        cluster,
                        pulse_count=int(event.pulse_count),
                        pulse_index=int(event.pulse_index),
                    )
                pulse_idx += 1
            if not emit_enabled:
                while pulse_idx < len(pulse_schedule) and int(pulse_schedule[pulse_idx].emit_step) == global_step:
                    pulse_idx += 1
            scene.step()
            if global_step < int(params.source_delay_steps):
                continue
            if self._active_ids(fluid).size == 0:
                if not emit_enabled:
                    continue
                future_emit_exists = any(
                    int(event.emit_step) >= int(params.source_delay_steps) and int(event.emit_step) > global_step
                    for event in pulse_schedule[pulse_idx:]
                )
                if future_emit_exists:
                    continue
            self._validate_active_particle_state(fluid, params)
        return max(start_step, end_step_exclusive)

    def _camera_basis(self, params: SceneParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye = np.asarray(params.camera_pos, dtype=np.float64)
        lookat = np.asarray(params.camera_lookat, dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        forward = normalize_vector(lookat - eye)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = normalize_vector(right)
        up = normalize_vector(np.cross(right, forward))
        return forward, right, up

    def _project_world_to_image(self, point: np.ndarray, params: SceneParams) -> Optional[Tuple[int, int]]:
        eye = np.asarray(params.camera_pos, dtype=np.float64)
        forward, right, up = self._camera_basis(params)
        rel = np.asarray(point, dtype=np.float64) - eye
        cam_x = float(np.dot(rel, right))
        cam_y = float(np.dot(rel, up))
        cam_z = float(np.dot(rel, forward))
        if cam_z <= 1e-4:
            return None
        focal = 0.5 * float(self.resolution) / math.tan(0.5 * math.radians(float(params.camera_fov_deg)))
        px = int(round(0.5 * float(self.resolution) + focal * cam_x / cam_z))
        py = int(round(0.5 * float(self.resolution) - focal * cam_y / cam_z))
        return px, py

    def _draw_line(self, image: np.ndarray, p0: Tuple[int, int], p1: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        x0, y0 = p0
        x1, y1 = p1
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for idx in range(steps + 1):
            t = float(idx) / float(steps)
            x = int(round((1.0 - t) * x0 + t * x1))
            y = int(round((1.0 - t) * y0 + t * y1))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] = color

    def _decorate_rgb_frame(self, rgb_frame: np.ndarray, params: SceneParams) -> np.ndarray:
        return np.asarray(rgb_frame, dtype=np.uint8)

    def _emit_cluster(self, emitter, cluster: LiquidCluster) -> None:
        emit_kwargs = dict(
            droplet_shape="circle",
            droplet_size=cluster.diameter,
            pos=tuple(cluster.pos.tolist()),
            direction=tuple(cluster.direction.tolist()),
            speed=cluster.speed,
            p_size=float(self.args.particle_size),
        )
        if cluster.length > 0.0:
            emit_kwargs["droplet_length"] = cluster.length
        emitter.emit(**emit_kwargs)

    def _active_ids(self, fluid) -> np.ndarray:
        active = to_numpy(fluid.get_particles_active()).astype(bool)
        return np.flatnonzero(active).astype(np.int32)

    def _validate_active_particle_state(
        self,
        fluid,
        params: SceneParams,
    ) -> None:
        active_ids = self._active_ids(fluid)
        if active_ids.size == 0:
            raise SceneGenerationError("No active particles remain in the scene.")

        positions_all = to_numpy(fluid.get_particles_pos()).astype(np.float32)
        velocities_all = to_numpy(fluid.get_particles_vel()).astype(np.float32)
        active_pos = positions_all[active_ids]
        active_vel = velocities_all[active_ids]

        if not np.isfinite(active_pos).all() or not np.isfinite(active_vel).all():
            raise SceneGenerationError("Active particle state contains NaN or Inf.")

        margin = max(6.0 * float(self.args.particle_size), 0.03)
        inside = np.all(active_pos >= (params.scene_bounds_lower[None, :] - margin), axis=1) & np.all(
            active_pos <= (params.scene_bounds_upper[None, :] + margin),
            axis=1,
        )
        escaped_count = int(np.count_nonzero(~inside))
        escaped_limit = max(8, int(math.ceil(0.005 * float(active_ids.size))))
        if escaped_count > escaped_limit:
            raise SceneGenerationError(
                f"Too many active particles escaped the scene bounds ({escaped_count}/{active_ids.size})."
            )

    def _choose_tracked_ids(
        self,
        fluid,
        params: SceneParams,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        active_ids = self._active_ids(fluid)
        min_required = self.tracked_particle_count
        if params.source_mode == "spout":
            min_required = max(24, min(self.tracked_particle_count, max(24, self.tracked_particle_count // 32)))
        if active_ids.size < min_required:
            return None

        positions_all = to_numpy(fluid.get_particles_pos()).astype(np.float32)
        velocities_all = to_numpy(fluid.get_particles_vel()).astype(np.float32)
        active_pos = positions_all[active_ids]
        active_vel = velocities_all[active_ids]
        valid = np.isfinite(active_pos).all(axis=1) & np.isfinite(active_vel).all(axis=1)
        if not np.any(valid):
            return None

        active_ids = active_ids[valid]
        active_pos = active_pos[valid]
        active_vel = active_vel[valid]

        inside_scene = np.all(
            active_pos >= params.scene_bounds_lower[None, :] - 1e-4,
            axis=1,
        ) & np.all(
            active_pos <= params.scene_bounds_upper[None, :] + 1e-4,
            axis=1,
        )
        if not np.any(inside_scene):
            return None

        active_ids = active_ids[inside_scene]
        active_pos = active_pos[inside_scene]
        active_vel = active_vel[inside_scene]
        if active_ids.size < min_required:
            return None

        inside_tracking_roi = np.all(
            active_pos >= params.tracking_bounds_lower[None, :],
            axis=1,
        ) & np.all(
            active_pos <= params.tracking_bounds_upper[None, :],
            axis=1,
        )
        scene_margin = max(4.0 * float(self.args.particle_size), 0.02)
        inside_safe_scene = np.all(
            active_pos >= (params.scene_bounds_lower[None, :] + scene_margin),
            axis=1,
        ) & np.all(
            active_pos <= (params.scene_bounds_upper[None, :] - scene_margin),
            axis=1,
        )
        speed = np.linalg.norm(active_vel, axis=1)
        container_centers_xy = np.stack([container.pos[:2] for container in params.containers], axis=0).astype(np.float32)
        dist_xy = np.min(
            np.linalg.norm(active_pos[:, None, :2] - container_centers_xy[None, :, :], axis=2),
            axis=1,
        )
        target_z = np.full_like(speed, params.camera_lookat[2], dtype=np.float32)
        z_cost = np.abs(active_pos[:, 2] - target_z)

        # Prefer particles already inside the pouring region and comfortably away from the scene bounds.
        # This reduces false scene rejection caused by tracking splash-out particles instead of representative fluid mass.
        stable_mask = inside_tracking_roi & inside_safe_scene
        if np.count_nonzero(stable_mask) >= min_required:
            active_ids = active_ids[stable_mask]
            active_pos = active_pos[stable_mask]
            active_vel = active_vel[stable_mask]
            inside_tracking_roi = inside_tracking_roi[stable_mask]
            inside_safe_scene = inside_safe_scene[stable_mask]
            speed = speed[stable_mask]
            dist_xy = dist_xy[stable_mask]
            z_cost = z_cost[stable_mask]

        score = (
            dist_xy
            + 0.20 * z_cost
            + 0.12 * speed
            + np.where(inside_tracking_roi, 0.0, 2.0).astype(np.float32)
            + np.where(inside_safe_scene, 0.0, 4.0).astype(np.float32)
        )
        order = np.argsort(score, kind="stable")
        chosen_count = min(self.tracked_particle_count, active_ids.size)
        if params.source_mode == "spout":
            chosen_count = max(min_required, min(chosen_count, self.tracked_particle_count))
        chosen = active_ids[order[: chosen_count]]
        return chosen.astype(np.int32)

    def _capture_frame(
        self,
        *,
        frame_idx: int,
        fluid,
        tracked_ids: np.ndarray,
        camera,
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray,
        pos_frames: np.ndarray,
        vel_frames: np.ndarray,
        params: SceneParams,
    ) -> None:
        positions_all = to_numpy(fluid.get_particles_pos()).astype(np.float32)
        velocities_all = to_numpy(fluid.get_particles_vel()).astype(np.float32)
        active_all = to_numpy(fluid.get_particles_active()).astype(bool)

        max_tracked_id = int(np.max(tracked_ids)) if tracked_ids.size > 0 else -1
        if positions_all.shape[0] < max_tracked_id + 1:
            raise SceneGenerationError("Tracked particle IDs exceed current particle buffer size.")

        tracked_active = active_all[tracked_ids]
        if not np.any(tracked_active):
            raise SceneGenerationError("Tracked particle ID became inactive.")

        valid_ids = tracked_ids[tracked_active]
        tracked_pos = positions_all[valid_ids]
        tracked_vel = velocities_all[valid_ids]

        if not np.isfinite(tracked_pos).all() or not np.isfinite(tracked_vel).all():
            raise SceneGenerationError("Tracked particle state contains NaN or Inf.")

        inside = np.all(tracked_pos >= params.scene_bounds_lower[None, :] - 1e-4, axis=1) & np.all(
            tracked_pos <= params.scene_bounds_upper[None, :] + 1e-4,
            axis=1,
        )
        if not np.all(inside):
            raise SceneGenerationError("Tracked particles escaped the simulation bounds.")

        if camera is None:
            rgb_frames[frame_idx].fill(0)
            depth_frames[frame_idx].fill(0.0)
        else:
            rgb_raw, depth_raw, _, _ = camera.render(rgb=True, depth=True, segmentation=False, normal=False)
            rgb_frames[frame_idx] = self._decorate_rgb_frame(rgb_to_uint8(to_numpy(rgb_raw)), params)
            depth_frames[frame_idx] = normalize_depth_map(
                to_numpy(depth_raw),
                near=float(camera.near),
                far=float(camera.far),
            )
        padded_pos = np.zeros((self.tracked_particle_count, 3), dtype=np.float32)
        padded_vel = np.zeros((self.tracked_particle_count, 3), dtype=np.float32)
        copy_count = min(self.tracked_particle_count, tracked_pos.shape[0])
        if copy_count > 0:
            padded_pos[:copy_count] = tracked_pos[:copy_count]
            padded_vel[:copy_count] = tracked_vel[:copy_count]
            if copy_count < self.tracked_particle_count:
                padded_pos[copy_count:] = tracked_pos[copy_count - 1]
                padded_vel[copy_count:] = tracked_vel[copy_count - 1]
        pos_frames[frame_idx] = padded_pos
        vel_frames[frame_idx] = padded_vel

    def write_scene_folder(
        self,
        scene_path: Path,
        payload: Dict[str, np.ndarray],
        params: SceneParams,
        per_scene_seed: int,
    ) -> None:
        ensure_dir(scene_path.parent)
        if scene_path.exists():
            shutil.rmtree(scene_path)
        ensure_dir(scene_path)

        rgb_dir = scene_path / "rgb"
        depth_dir = scene_path / "depth"
        particles_dir = scene_path / "particles"
        metadata_dir = scene_path / "metadata"
        video_dir = scene_path / "video"
        for path in (rgb_dir, depth_dir, particles_dir, metadata_dir, video_dir):
            ensure_dir(path)

        rgb_frames = payload["rgb"]
        depth_frames = payload["depth"]
        rgb_variants = payload.get("rgb_variants", {str(self.args.fluid_vis_mode): rgb_frames})
        depth_uint8_frames = np.empty((self.frames, self.resolution, self.resolution), dtype=np.uint8)

        for frame_idx in range(self.frames):
            rgb_frame_path = rgb_dir / f"{frame_idx:04d}.png"
            depth_frame_path = depth_dir / f"{frame_idx:04d}.png"
            imageio.imwrite(rgb_frame_path, rgb_frames[frame_idx])
            depth_uint8 = depth_to_uint8(depth_frames[frame_idx])
            depth_uint8_frames[frame_idx] = depth_uint8
            imageio.imwrite(depth_frame_path, depth_uint8)

        np.save(depth_dir / "depth.npy", depth_frames)
        np.save(particles_dir / "pos.npy", payload["particles_pos"])
        np.save(particles_dir / "vel.npy", payload["particles_vel"])
        np.save(particles_dir / "ids.npy", payload["particles_ids"].astype(np.int32))

        depth_video_frames = np.repeat(depth_uint8_frames[..., None], 3, axis=-1)
        write_mp4_rgb(video_dir / "rgb_preview.mp4", rgb_frames, fps=self.output_fps)
        imageio.mimwrite(video_dir / "depth_preview.mp4", depth_video_frames, fps=self.output_fps, quality=8)
        variant_video_files: Dict[str, str] = {}
        for render_mode, variant_frames in rgb_variants.items():
            variant_name = f"rgb_{render_mode}_preview.mp4"
            write_mp4_rgb(video_dir / variant_name, variant_frames, fps=self.output_fps)
            variant_video_files[render_mode] = f"video/{variant_name}"

        metadata = {
            "scene_index": int(params.scene_index),
            "seed": int(per_scene_seed),
            "solver": params.solver_name,
            "solver_substeps": int(params.solver_substeps),
            "liquid_preset": params.preset_name,
            "scene_variant_name": params.scene_variant_name,
            "fps": int(self.output_fps),
            "frames": int(self.frames),
            "resolution": [int(self.resolution), int(self.resolution)],
            "tracked_particle_count": int(self.tracked_particle_count),
            "particle_id_semantics": "Genesis particle buffer indices, sampled once and kept fixed across all frames.",
            "particle_sample_strategy": "score_based_selection_from_active_particles_in_stable_roi",
            "particle_size": float(self.args.particle_size),
            "fluid_vis_mode": str(self.args.fluid_vis_mode),
            "render_rgb_modes": list(self.render_rgb_modes),
            "container_double_sided": bool(self.args.container_double_sided),
            "render_scene_bounds": bool(self.args.render_scene_bounds),
            "skip_render": bool(self.args.skip_render),
            "fluid_color_name": str(params.fluid_color_name),
            "fluid_rgba": [float(v) for v in params.fluid_rgba],
            "viscosity": float(params.viscosity),
            "stiffness": float(params.stiffness),
            "exponent": float(params.exponent),
            "gamma": float(params.gamma),
            "solver_effective_viscosity": float(
                get_effective_solver_viscosity(params.solver_name, params.viscosity)
            ),
            "gravity": np.asarray(params.gravity, dtype=np.float32).tolist(),
            "recording_start": "after_staggered_finite_emit_and_short_settle",
            "emission_schedule": "staggered_finite_packets",
            "emitter_speed": float(params.emitter_speed),
            "emitter_diameter": float(params.emitter_diameter),
            "emitter_length": float(params.emitter_length),
            "emission_interval_steps": int(params.emission_interval_steps),
            "estimated_particles_per_emit": int(params.estimated_particles_per_emit),
            "target_particle_budget": int(params.target_particle_budget),
            "clusters_per_container": int(params.clusters_per_container),
            "volume_tiers_per_container": ["small", "medium", "large"],
            "direction_modes_available": ["top", "front_left", "front_right"],
            "liquid_clusters": [
                {
                    "cluster_index": int(cluster.cluster_index),
                    "target_container_index": int(cluster.target_container_index),
                    "volume_tier": str(cluster.volume_tier),
                    "direction_name": str(cluster.direction_name),
                    "pos": np.asarray(cluster.pos, dtype=np.float32).tolist(),
                    "direction": np.asarray(cluster.direction, dtype=np.float32).tolist(),
                    "speed": float(cluster.speed),
                    "diameter": float(cluster.diameter),
                    "length": float(cluster.length),
                    "target_particles": int(cluster.target_particles),
                }
                for cluster in params.liquid_clusters
            ],
            "source_mode": str(params.source_mode),
            "source_delay_steps": int(params.source_delay_steps),
            "source_tilt_schedule_deg": (
                None
                if params.source_tilt_schedule_deg is None
                else np.asarray(params.source_tilt_schedule_deg, dtype=np.float32).tolist()
            ),
            "camera_azimuth_deg": float(params.camera_azimuth_deg),
            "camera_elevation_deg": float(params.camera_elevation_deg),
            "camera_radius": float(params.camera_radius),
            "camera_fov_deg": float(params.camera_fov_deg),
            "camera_pos": np.asarray(params.camera_pos, dtype=np.float32).tolist(),
            "camera_lookat": np.asarray(params.camera_lookat, dtype=np.float32).tolist(),
            "liquid_volume_scale": float(self.args.liquid_volume_scale),
            "container_source_id": params.prepared_container.candidate.source_id,
            "container_label": params.prepared_container.candidate.label,
            "container_kind": params.prepared_container.candidate.kind,
            "container_mesh": str(params.prepared_container.cache_path),
            "containers": [
                {
                    "container_index": int(idx),
                    "source_id": container.prepared_container.candidate.source_id,
                    "label": container.prepared_container.candidate.label,
                    "kind": container.prepared_container.candidate.kind,
                    "mesh": str(container.prepared_container.cache_path),
                    "pos": np.asarray(container.pos, dtype=np.float32).tolist(),
                    "scene_extents": container.prepared_container.scene_extents.astype(np.float32).tolist(),
                    "quat_wxyz": [float(v) for v in container.quat_wxyz],
                    "yaw_deg": float(container.yaw_deg),
                    "color_name": str(container.color_name),
                    "surface_rgba": [float(v) for v in container.surface_rgba],
                    "emissive_rgb": [float(v) for v in container.emissive_rgb],
                }
                for idx, container in enumerate(params.containers)
            ],
            "source_container": (
                None
                if params.source_container is None
                else {
                    "source_id": params.source_container.prepared_container.candidate.source_id,
                    "label": params.source_container.prepared_container.candidate.label,
                    "kind": params.source_container.prepared_container.candidate.kind,
                    "mesh": str(params.source_container.prepared_container.cache_path),
                    "pos": np.asarray(params.source_container.pos, dtype=np.float32).tolist(),
                    "scene_extents": params.source_container.prepared_container.scene_extents.astype(np.float32).tolist(),
                    "quat_wxyz": [float(v) for v in params.source_container.quat_wxyz],
                    "yaw_deg": float(params.source_container.yaw_deg),
                    "color_name": str(params.source_container.color_name),
                    "surface_rgba": [float(v) for v in params.source_container.surface_rgba],
                    "emissive_rgb": [float(v) for v in params.source_container.emissive_rgb],
                }
            ),
            "source_anchor": (
                None if params.source_anchor is None else np.asarray(params.source_anchor, dtype=np.float32).tolist()
            ),
            "scene_bounds_lower": params.scene_bounds_lower.astype(np.float32).tolist(),
            "scene_bounds_upper": params.scene_bounds_upper.astype(np.float32).tolist(),
            "genesis_backend": GENESIS_BACKEND_USED,
            "files": {
                "rgb_frames_dir": "rgb",
                "depth_frames_dir": "depth",
                "depth_array": "depth/depth.npy",
                "particles_pos": "particles/pos.npy",
                "particles_vel": "particles/vel.npy",
                "particles_ids": "particles/ids.npy",
                "rgb_video": "video/rgb_preview.mp4",
                "depth_video": "video/depth_preview.mp4",
                "rgb_variant_videos": variant_video_files,
            },
        }
        (metadata_dir / "scene.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Genesis liquid-object interaction dataset and save each scene as a folder."
    )
    parser.add_argument("--solver", choices=("sph", "pbd"), default="sph", help="Fluid solver family.")
    parser.add_argument("--num-scenes", type=int, default=1, help="Number of scene folders to generate.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for scene_xxxxx folders.",
    )
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Cache directory for prepared meshes.")
    parser.add_argument("--seed", type=int, default=20260412, help="Global random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scene files.")
    parser.add_argument("--frames", type=int, default=49, help="Frames per scene.")
    parser.add_argument("--fps", type=int, default=24, help="Output frame rate.")
    parser.add_argument("--resolution", type=int, default=512, help="Rendered image resolution.")
    parser.add_argument("--physics-fps", type=int, default=300, help="Internal simulation frequency.")
    parser.add_argument("--substeps", type=int, default=10, help="Genesis substeps per scene.step().")
    parser.add_argument("--particle-size", type=float, default=0.010, help="Fluid particle diameter in meters.")
    parser.add_argument("--tracked-particles", type=int, default=1024, help="Tracked particle count.")
    parser.add_argument("--max-particles", type=int, default=24576, help="Emitter particle buffer size.")
    parser.add_argument(
        "--liquid-volume-scale",
        type=float,
        default=1.0,
        help="Extra multiplier on liquid particle budget. Values > 1 create more dramatic large-volume pours.",
    )
    parser.add_argument(
        "--liquid-volume-range",
        type=parse_float_pair,
        default=(0.70, 3.40),
        help="Extra low,high multiplier range applied on top of --liquid-volume-scale to widen liquid volume sampling.",
    )
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=1.0,
        help="Extra multiplier on the sampled camera radius. Values > 1 pull the camera farther back.",
    )
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=46.0,
        help="Camera field of view in degrees. Larger values include more containers in frame.",
    )
    parser.add_argument(
        "--liquid-preset",
        type=str,
        default="auto",
        choices=("auto",) + LIQUID_PRESET_SEQUENCE,
        help="Liquid parameter preset. 'auto' cycles through a diverse preset family across scenes.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=3,
        help="Deprecated compatibility flag. Actual emitted cluster count is num_containers x 3 volume tiers.",
    )
    parser.add_argument(
        "--max-containers",
        type=int,
        default=3,
        help="Maximum number of container instances placed in one scene. Each container receives 3 liquid packets.",
    )
    parser.add_argument(
        "--min-containers",
        type=int,
        default=1,
        help="Minimum number of container instances placed in one scene. Use the same value as --max-containers to fix it.",
    )
    parser.add_argument(
        "--container-counts",
        type=parse_int_csv,
        default=[1, 2, 3],
        help="Comma-separated allowed container counts to sample from. Example: 1,2,3 or 2 or 1,3",
    )
    parser.add_argument(
        "--fluid-vis-mode",
        choices=("recon", "particle"),
        default="particle",
        help="Fluid rendering mode for RGB preview generation. The current renderer forces particle mode for stable colors.",
    )
    parser.add_argument(
        "--render-rgb-modes",
        type=parse_csv_choices,
        default=["particle"],
        help="Comma-separated RGB render modes to export in video/. The current renderer exports particle only.",
    )
    parser.add_argument(
        "--container-double-sided",
        type=parse_bool_flag,
        default=False,
        help="Whether to render both sides of the container mesh. Accepts true/false. Default: false.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("offscreen_stream", "offscreen_delayed", "spout"),
        default="spout",
        help="Liquid source style: continuous flow from offscreen, delayed offscreen entry, or a visible in-frame source container that pours.",
    )
    parser.add_argument(
        "--source-delay-frames",
        type=int,
        default=6,
        help="How many output frames to delay the start of the source flow for delayed-entry source modes.",
    )
    parser.add_argument(
        "--render-scene-bounds",
        type=parse_bool_flag,
        default=True,
        help="Overlay the simulation bounds as visible guide lines in rendered RGB frames.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip camera rendering and write blank RGB/depth frames. Useful on headless machines without EGL.",
    )
    parser.add_argument(
        "--render-platform",
        choices=("auto", "egl", "pyglet", "osmesa"),
        default="auto",
        help="PyOpenGL backend for Genesis rasterization. 'auto' leaves backend selection to Genesis.",
    )
    parser.add_argument(
        "--egl-device-id",
        type=int,
        default=None,
        help="Optional EGL device index when using --render-platform egl.",
    )
    parser.add_argument(
        "--max-retries-per-scene",
        type=int,
        default=8,
        help="How many times to retry a failed scene before aborting.",
    )
    parser.add_argument(
        "--container-mode",
        choices=("auto", "procedural", "direct", "physxnet"),
        default="procedural",
        help="Container source mode. 'auto' uses direct OBJ dir if provided, otherwise procedural.",
    )
    parser.add_argument(
        "--container-obj-dir",
        type=str,
        default=None,
        help="Directory containing standalone container .obj files for direct mode.",
    )
    parser.add_argument(
        "--container-mesh-up",
        choices=("z", "y"),
        default="z",
        help="Up-axis convention for direct OBJ containers.",
    )
    parser.add_argument("--physx-root", type=str, default=str(DEFAULT_PHYSX_ROOT), help="PhysXNet root directory.")
    parser.add_argument("--physx-version", type=str, default=DEFAULT_PHYSX_VERSION, help="PhysXNet version folder.")
    parser.add_argument("--show-viewer", action="store_true", help="Show the Genesis viewer for debugging.")
    parser.add_argument("--force-cpu", action="store_true", help="Force Genesis CPU backend initialization.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.container_mode == "direct" and not args.container_obj_dir:
        parser.error("--container-mode direct requires --container-obj-dir.")

    if args.egl_device_id is not None and args.render_platform != "egl":
        parser.error("--egl-device-id can only be used together with --render-platform egl.")

    valid_container_counts = sorted({int(count) for count in args.container_counts})
    if any(count not in (1, 2, 3) for count in valid_container_counts):
        parser.error("--container-counts only supports values from {1,2,3}.")
    args.container_counts = valid_container_counts

    try:
        validate_runtime_configuration(args)
    except ValueError as exc:
        parser.error(str(exc))

    configure_render_backend(args.render_platform, args.egl_device_id)
    ensure_python_dependencies()
    simulator = FluidSimulator(args)
    simulator.run()


if __name__ == "__main__":
    main()
'''
rm -rf /data/gaoya/AAA_test_video/Dataset_physV/liquid_cache
rm -rf /data/gaoya/AAA_test_video/Dataset_physV/liquid_dataset

python /home/gaoya/Code_Video/Code_data/try2_luquid_genesis0412.py \
    --num-scenes 12 \
    --solver sph \
    --container-mode procedural \
    --container-counts 1,2,3 \
    --min-containers 1 \
    --max-containers 3 \
    --particle-size 0.008 \
    --tracked-particles 1024 \
    --max-particles 65536 \
    --liquid-preset triple_arc \
    --liquid-volume-scale 3.0 \
    --liquid-volume-range 1.4,6.0 \
    --camera-distance-scale 1.35 \
    --camera-fov 58 \
    --output-dir /data/gaoya/AAA_test_video/Dataset_physV/liquid_dataset \
    --cache-dir /data/gaoya/AAA_test_video/Dataset_physV/liquid_cache \
    --render-platform egl \
    --fluid-vis-mode particle \
    --render-rgb-modes particle,recon \
    --container-double-sided false \
    --render-scene-bounds false \
    --source-mode spout \
    --source-delay-frames 6 \
    --overwrite

  如果之后要切回液体表面重建，只改这一项：--fluid-vis-mode recon 

    
可视化

python3 /home/gaoya/Code_Video/Code_data/1_localshow.py \
    --root /data/gaoya/AAA_test_video/Dataset_physV/liquid_dataset \
    --profile scene_video \
    --video-name "rgb_particle_preview.mp4" \
    --video-name "rgb_recon_preview.mp4" \
    --title "Genesis Liquid Viewer" \
    --port 8018

'''
