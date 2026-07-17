#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import site
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import trimesh


def _sanitize_user_site_for_genesis() -> None:
    user_site = site.getusersitepackages()
    if not user_site:
        return
    user_site = os.path.abspath(user_site)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != user_site]


_sanitize_user_site_for_genesis()

import genesis as gs

try:
    from dataset_new_0705.material_catalog_0705 import (
        build_hdri_catalog,
        build_lighting_catalog,
        build_material_catalog,
        build_surface_catalog,
    )
except ImportError:
    from material_catalog_0705 import (
        build_hdri_catalog,
        build_lighting_catalog,
        build_material_catalog,
        build_surface_catalog,
    )


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview")
DEFAULT_FPS = 30
DEFAULT_RES = (960, 544)


@dataclass(frozen=True)
class CameraSpec:
    res: tuple[int, int]
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float


@dataclass(frozen=True)
class SimSpec:
    dt: float
    substeps: int
    horizon: int
    gravity: tuple[float, float, float]
    mpm_lower_bound: tuple[float, float, float]
    mpm_upper_bound: tuple[float, float, float]
    grid_density: int


@dataclass(frozen=True)
class MaterialPalette:
    floor_color: tuple[float, float, float, float]
    wall_color: tuple[float, float, float, float]
    wall_alt_color: tuple[float, float, float, float]
    ceiling_color: tuple[float, float, float, float]
    trim_color: tuple[float, float, float, float]
    wood_mid_color: tuple[float, float, float, float]
    wood_dark_color: tuple[float, float, float, float]
    fabric_color: tuple[float, float, float, float]
    metal_color: tuple[float, float, float, float]
    clutter_color: tuple[float, float, float, float]
    soft_color: tuple[float, float, float, float]
    rigid_color: tuple[float, float, float, float]
    accent_color: tuple[float, float, float, float]


@dataclass(frozen=True)
class CaseSpec:
    key: str
    family: str
    title: str
    description: str
    sim: SimSpec
    camera: CameraSpec
    mpm_vis_mode: str
    palette: MaterialPalette
    scene_theme: str
    surface_key: str
    lighting_key: str
    soft_material_key: str
    rigid_material_key: str
    motion_profile: str
    soft_secondary_material_key: str = ""


_GS_INITIALIZED = False
_MATERIAL_CATALOG = build_material_catalog()
_LIGHTING_CATALOG = build_lighting_catalog()
_SURFACE_CATALOG = build_surface_catalog()
_HDRI_CATALOG = build_hdri_catalog()
_UV_ASSET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0713pybullet/assets/genesis_uv_primitives")

_UNIT_BOX_UV_OBJ = """# unit box with per-face UVs
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
v 0.5 -0.5 -0.5
v -0.5 -0.5 -0.5
v -0.5 0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 -0.5 -0.5
v -0.5 -0.5 0.5
v -0.5 0.5 0.5
v -0.5 0.5 -0.5
v 0.5 -0.5 0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
v 0.5 0.5 0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 -0.5 0.5
v -0.5 -0.5 0.5
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 -1.0
vn -1.0 0.0 0.0
vn 1.0 0.0 0.0
vn 0.0 1.0 0.0
vn 0.0 -1.0 0.0
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
f 5/1/2 6/2/2 7/3/2
f 5/1/2 7/3/2 8/4/2
f 9/1/3 10/2/3 11/3/3
f 9/1/3 11/3/3 12/4/3
f 13/1/4 14/2/4 15/3/4
f 13/1/4 15/3/4 16/4/4
f 17/1/5 18/2/5 19/3/5
f 17/1/5 19/3/5 20/4/5
f 21/1/6 22/2/6 23/3/6
f 21/1/6 23/3/6 24/4/6
"""


def _palette_residential_warm() -> MaterialPalette:
    return MaterialPalette(
        floor_color=(0.44, 0.35, 0.27, 1.0),
        wall_color=(0.88, 0.85, 0.79, 1.0),
        wall_alt_color=(0.76, 0.72, 0.65, 1.0),
        ceiling_color=(0.95, 0.94, 0.91, 1.0),
        trim_color=(0.37, 0.26, 0.18, 1.0),
        wood_mid_color=(0.64, 0.50, 0.34, 1.0),
        wood_dark_color=(0.30, 0.21, 0.15, 1.0),
        fabric_color=(0.76, 0.72, 0.67, 1.0),
        metal_color=(0.48, 0.56, 0.58, 1.0),
        clutter_color=(0.61, 0.54, 0.42, 1.0),
        soft_color=(0.24, 0.50, 0.76, 1.0),
        rigid_color=(0.86, 0.67, 0.21, 1.0),
        accent_color=(0.84, 0.67, 0.26, 1.0),
    )


def _palette_industrial_cool() -> MaterialPalette:
    return MaterialPalette(
        floor_color=(0.32, 0.34, 0.35, 1.0),
        wall_color=(0.84, 0.84, 0.82, 1.0),
        wall_alt_color=(0.70, 0.72, 0.73, 1.0),
        ceiling_color=(0.84, 0.86, 0.87, 1.0),
        trim_color=(0.27, 0.30, 0.31, 1.0),
        wood_mid_color=(0.47, 0.37, 0.27, 1.0),
        wood_dark_color=(0.21, 0.20, 0.18, 1.0),
        fabric_color=(0.73, 0.74, 0.74, 1.0),
        metal_color=(0.48, 0.51, 0.53, 1.0),
        clutter_color=(0.56, 0.50, 0.38, 1.0),
        soft_color=(0.30, 0.51, 0.74, 1.0),
        rigid_color=(0.93, 0.76, 0.22, 1.0),
        accent_color=(0.78, 0.62, 0.21, 1.0),
    )


def _palette_soft_daylight() -> MaterialPalette:
    return MaterialPalette(
        floor_color=(0.50, 0.42, 0.31, 1.0),
        wall_color=(0.91, 0.90, 0.85, 1.0),
        wall_alt_color=(0.80, 0.79, 0.73, 1.0),
        ceiling_color=(0.97, 0.96, 0.94, 1.0),
        trim_color=(0.45, 0.33, 0.22, 1.0),
        wood_mid_color=(0.72, 0.58, 0.39, 1.0),
        wood_dark_color=(0.34, 0.25, 0.17, 1.0),
        fabric_color=(0.81, 0.78, 0.73, 1.0),
        metal_color=(0.56, 0.62, 0.64, 1.0),
        clutter_color=(0.67, 0.58, 0.44, 1.0),
        soft_color=(0.18, 0.52, 0.78, 1.0),
        rigid_color=(0.83, 0.64, 0.18, 1.0),
        accent_color=(0.86, 0.71, 0.28, 1.0),
    )


def _palette_loft_neutral() -> MaterialPalette:
    return MaterialPalette(
        floor_color=(0.38, 0.31, 0.25, 1.0),
        wall_color=(0.82, 0.81, 0.78, 1.0),
        wall_alt_color=(0.63, 0.63, 0.62, 1.0),
        ceiling_color=(0.92, 0.92, 0.90, 1.0),
        trim_color=(0.29, 0.27, 0.24, 1.0),
        wood_mid_color=(0.58, 0.48, 0.36, 1.0),
        wood_dark_color=(0.26, 0.22, 0.18, 1.0),
        fabric_color=(0.73, 0.72, 0.69, 1.0),
        metal_color=(0.50, 0.54, 0.57, 1.0),
        clutter_color=(0.60, 0.54, 0.46, 1.0),
        soft_color=(0.77, 0.34, 0.27, 1.0),
        rigid_color=(0.22, 0.47, 0.70, 1.0),
        accent_color=(0.80, 0.62, 0.24, 1.0),
    )


CASE_LIBRARY: dict[str, CaseSpec] = {
    "F1_elastic_drop_residential": CaseSpec(
        key="F1_elastic_drop_residential",
        family="F1",
        title="Elastic block drop in residential room",
        description="A soft block drops in a warm indoor room with layered structure and nearby clutter.",
        sim=SimSpec(
            dt=4e-3,
            substeps=14,
            horizon=180,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.3, -1.2, -0.1),
            mpm_upper_bound=(1.3, 1.2, 1.8),
            grid_density=96,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.45, -2.10, 1.40),
            lookat=(0.05, 0.08, 0.34),
            fov=36.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_residential_warm(),
        scene_theme="residential_warm",
        surface_key="residential_wood_floor",
        lighting_key="studio_warm",
        soft_material_key="leather_brown",
        rigid_material_key="painted_metal_teal",
        motion_profile="drop",
    ),
    "F2_sphere_impact_studio": CaseSpec(
        key="F2_sphere_impact_studio",
        family="F2",
        title="Sphere impact on soft block",
        description="A rigid sphere impacts a soft block in a daylight indoor scene to validate rigid-MPM coupling.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=220,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.55, -2.25, 1.46),
            lookat=(0.20, 0.02, 0.26),
            fov=34.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_soft_daylight(),
        scene_theme="soft_daylight",
        surface_key="studio_wood_floor",
        lighting_key="studio_soft",
        soft_material_key="cardboard_kraft",
        rigid_material_key="painted_metal_teal",
        motion_profile="sphere_impact",
    ),
    "F3_double_soft_stack": CaseSpec(
        key="F3_double_soft_stack",
        family="F3",
        title="Two soft blocks with offset drop",
        description="A smaller soft block drops onto a larger resting soft block to create soft-soft interaction.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=220,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.48, -2.05, 1.48),
            lookat=(0.06, 0.04, 0.30),
            fov=35.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_residential_warm(),
        scene_theme="residential_warm",
        surface_key="residential_wood_floor",
        lighting_key="studio_warm",
        soft_material_key="leather_brown",
        rigid_material_key="painted_metal_yellow",
        soft_secondary_material_key="cardboard_kraft",
        motion_profile="soft_stack",
    ),
    "F4_side_swipe_cylinder": CaseSpec(
        key="F4_side_swipe_cylinder",
        family="F4",
        title="Cylinder side-swipe on soft block",
        description="A rolling rigid cylinder side-swipes a soft block for a more lateral motion pattern.",
        sim=SimSpec(
            dt=4e-3,
            substeps=24,
            horizon=240,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.72, -2.35, 1.30),
            lookat=(0.22, -0.02, 0.26),
            fov=33.0,
        ),
        # The Genesis rasterizer currently drops the box-like MPM visual mesh in this shot, while
        # particle mode renders reliably and makes deformation easier to read.
        mpm_vis_mode="particle",
        palette=_palette_industrial_cool(),
        scene_theme="industrial_cool",
        surface_key="painted_concrete_floor",
        lighting_key="hall_neutral",
        soft_material_key="cardboard_kraft",
        rigid_material_key="painted_metal_yellow",
        motion_profile="cylinder_swipe",
    ),
    "F5_corner_drop_oblique": CaseSpec(
        key="F5_corner_drop_oblique",
        family="F5",
        title="Oblique corner drop",
        description="A soft block starts rotated and drops near a support block for richer contact and camera occlusion.",
        sim=SimSpec(
            dt=4e-3,
            substeps=16,
            horizon=210,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=104,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.38, -1.95, 1.22),
            lookat=(0.18, 0.18, 0.24),
            fov=39.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_soft_daylight(),
        scene_theme="soft_daylight",
        surface_key="soft_wall_floor",
        lighting_key="studio_soft",
        soft_material_key="leather_brown",
        rigid_material_key="painted_metal_teal",
        motion_profile="corner_drop",
    ),
    "F6_tall_slab_flip_loft": CaseSpec(
        key="F6_tall_slab_flip_loft",
        family="F6",
        title="Tall soft slab flip in loft room",
        description="A tall soft slab drops with a stronger initial tilt in a loft-like room for more dramatic squash and rebound.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=230,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 2.0),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.86, -2.48, 1.10),
            lookat=(0.12, 0.02, 0.32),
            fov=31.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_loft_neutral(),
        scene_theme="loft_neutral",
        surface_key="dark_wood_floor",
        lighting_key="hall_neutral",
        soft_material_key="cardboard_kraft",
        rigid_material_key="painted_metal_teal",
        motion_profile="tall_flip",
    ),
    "F7_dual_sphere_press_gallery": CaseSpec(
        key="F7_dual_sphere_press_gallery",
        family="F7",
        title="Dual sphere squeeze in gallery-like interior",
        description="Two rigid spheres compress a soft block from offset directions to create richer coupled motion and more asymmetric deformation.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=250,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.5, -1.2, -0.1),
            mpm_upper_bound=(1.5, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.58, -2.12, 1.52),
            lookat=(0.04, 0.02, 0.26),
            fov=35.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_soft_daylight(),
        scene_theme="soft_daylight",
        surface_key="studio_wood_floor",
        lighting_key="studio_soft",
        soft_material_key="leather_brown",
        rigid_material_key="painted_metal_teal",
        motion_profile="dual_sphere_press",
    ),
    "F8_glancing_arc_impact": CaseSpec(
        key="F8_glancing_arc_impact",
        family="F8",
        title="Glancing arc impact",
        description="A rigid sphere descends diagonally from above, clipping the upper edge of a soft block for a more glancing contact.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=230,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 2.0),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.42, -2.00, 1.70),
            lookat=(0.10, 0.06, 0.36),
            fov=37.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_residential_warm(),
        scene_theme="residential_warm",
        surface_key="residential_wood_floor",
        lighting_key="studio_warm",
        soft_material_key="cardboard_kraft",
        rigid_material_key="painted_metal_yellow",
        motion_profile="glancing_arc",
    ),
    "F9_reverse_swipe_workshop": CaseSpec(
        key="F9_reverse_swipe_workshop",
        family="F9",
        title="Reverse cylinder swipe in workshop room",
        description="A rigid cylinder approaches from the opposite side with a shallower line, giving a more oblique lateral shove.",
        sim=SimSpec(
            dt=4e-3,
            substeps=24,
            horizon=240,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(-1.60, -2.08, 1.28),
            lookat=(0.05, -0.02, 0.24),
            fov=34.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_industrial_cool(),
        scene_theme="industrial_cool",
        surface_key="painted_concrete_floor",
        lighting_key="hall_neutral",
        soft_material_key="leather_brown",
        rigid_material_key="painted_metal_yellow",
        motion_profile="reverse_swipe",
    ),
    "F10_ramp_slide_study": CaseSpec(
        key="F10_ramp_slide_study",
        family="F10",
        title="Ramp slide in study-like interior",
        description="A soft block starts on a shallow ramp and slides into a tumble, adding a gravity-driven motion family distinct from direct impact.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=230,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.42, -1.38, 1.18),
            lookat=(-0.16, 0.10, 0.30),
            fov=36.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_soft_daylight(),
        scene_theme="soft_daylight",
        surface_key="studio_wood_floor",
        lighting_key="studio_soft",
        soft_material_key="leather_brown",
        rigid_material_key="painted_metal_teal",
        motion_profile="ramp_slide",
    ),
    "F11_overhead_press_atelier": CaseSpec(
        key="F11_overhead_press_atelier",
        family="F11",
        title="Overhead press in atelier room",
        description="A rigid sphere drops from above with a slight lateral bias, producing a more vertical compression event on a tall soft block.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=240,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 2.0),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(0.98, -1.30, 1.34),
            lookat=(0.06, 0.02, 0.38),
            fov=35.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_residential_warm(),
        scene_theme="residential_warm",
        surface_key="residential_wood_floor",
        lighting_key="studio_warm",
        soft_material_key="cardboard_kraft",
        rigid_material_key="painted_metal_yellow",
        motion_profile="overhead_press",
    ),
    "F12_ledge_topple_loft": CaseSpec(
        key="F12_ledge_topple_loft",
        family="F12",
        title="Ledge topple in loft room",
        description="A soft block is perched on a narrow ledge and topples under gravity, creating delayed contact and asymmetric landing.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=220,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(-1.02, -1.42, 1.06),
            lookat=(0.56, 0.20, 0.36),
            fov=37.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_loft_neutral(),
        scene_theme="loft_neutral",
        surface_key="dark_wood_floor",
        lighting_key="hall_neutral",
        soft_material_key="leather_brown",
        rigid_material_key="painted_metal_teal",
        motion_profile="ledge_topple",
    ),
    "F13_wall_pinch_workroom": CaseSpec(
        key="F13_wall_pinch_workroom",
        family="F13",
        title="Wall pinch in workroom interior",
        description="A rigid sphere drives a soft block toward a nearby support wall, producing a more constrained compression and rebound pattern.",
        sim=SimSpec(
            dt=4e-3,
            substeps=18,
            horizon=480,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.4, -1.2, -0.1),
            mpm_upper_bound=(1.4, 1.2, 1.9),
            grid_density=112,
        ),
        camera=CameraSpec(
            res=DEFAULT_RES,
            pos=(1.18, -1.34, 0.96),
            lookat=(0.28, 0.10, 0.24),
            fov=34.0,
        ),
        mpm_vis_mode="visual",
        palette=_palette_industrial_cool(),
        scene_theme="industrial_cool",
        surface_key="painted_concrete_floor",
        lighting_key="hall_neutral",
        soft_material_key="cardboard_kraft",
        rigid_material_key="painted_metal_yellow",
        motion_profile="wall_pinch",
    ),
}


def build_family_case_catalog() -> dict[str, list[CaseSpec]]:
    grouped: dict[str, list[CaseSpec]] = {}
    for case in sorted(CASE_LIBRARY.values(), key=lambda item: (item.family, item.key)):
        grouped.setdefault(case.family, []).append(case)
    return grouped


def _init_genesis() -> None:
    global _GS_INITIALIZED
    if _GS_INITIALIZED:
        return
    gs.init(backend=gs.gpu)
    _GS_INITIALIZED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a realism-oriented Genesis MPM preview case.")
    parser.add_argument(
        "--case-key",
        default="F2_sphere_impact_studio",
        choices=sorted(CASE_LIBRARY.keys()),
        help="Which preview case to simulate.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="", help="Optional explicit output folder name.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Video playback speed relative to real simulation time. 1.0 is real-time, values below 1.0 are slower.",
    )
    parser.add_argument("--width", type=int, default=0, help="Optional override for camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional override for camera height.")
    parser.add_argument(
        "--mpm-vis-mode",
        choices=["visual", "particle"],
        default="",
        help="Override how MPM bodies are rendered.",
    )
    parser.add_argument(
        "--save-every-frame",
        action="store_true",
        help="If set, write all RGB frames to disk instead of only preview keyframes.",
    )
    return parser.parse_args()


def _camera_spec(case: CaseSpec, width: int, height: int) -> CameraSpec:
    if width > 0 and height > 0:
        return CameraSpec(
            res=(int(width), int(height)),
            pos=case.camera.pos,
            lookat=case.camera.lookat,
            fov=case.camera.fov,
        )
    return case.camera


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_uv_box_mesh_path() -> Path:
    _ensure_dir(_UV_ASSET_ROOT)
    path = _UV_ASSET_ROOT / "unit_box_uv.obj"
    if not path.exists():
        path.write_text(_UNIT_BOX_UV_OBJ, encoding="utf-8")
    return path


def _build_output_dirs(case_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": case_dir,
        "rgb": case_dir / "rgb",
        "video": case_dir / "video",
        "debug": case_dir / "debug",
    }
    for path in dirs.values():
        _ensure_dir(path)
    return dirs


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _rgb(color: tuple[float, ...]) -> tuple[float, float, float]:
    return (float(color[0]), float(color[1]), float(color[2]))


def _rgba(color: tuple[float, ...], alpha: float = 1.0) -> tuple[float, float, float, float]:
    return (float(color[0]), float(color[1]), float(color[2]), float(alpha))


def _mix_rgb(a: tuple[float, float, float], b: tuple[float, float, float], weight_b: float) -> tuple[float, float, float]:
    wa = float(np.clip(1.0 - weight_b, 0.0, 1.0))
    wb = float(np.clip(weight_b, 0.0, 1.0))
    return tuple(float(np.clip(wa * a[i] + wb * b[i], 0.0, 1.0)) for i in range(3))


def _safe_image_texture(path: str, *, encoding: str, image_color: tuple[float, ...] | None = None) -> Any | None:
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    kwargs: dict[str, Any] = {
        "image_path": str(src),
        "encoding": encoding,
    }
    if image_color is not None:
        kwargs["image_color"] = image_color
    return gs.textures.ImageTexture(**kwargs)


def _material_surface(
    material_key: str,
    *,
    tint_rgb: tuple[float, float, float] | None = None,
    opacity: float = 1.0,
    roughness_override: float | None = None,
    metallic_override: float | None = None,
    vis_mode: str | None = None,
    double_sided: bool | None = None,
    smooth: bool = True,
) -> Any:
    material = _MATERIAL_CATALOG[material_key]
    texture_tint = _mix_rgb((1.0, 1.0, 1.0), tint_rgb, 0.52) if tint_rgb is not None else None
    diffuse_tex = _safe_image_texture(material.texture_path, encoding="srgb", image_color=texture_tint)
    roughness_tex = _safe_image_texture(material.roughness_path, encoding="linear")
    normal_tex = _safe_image_texture(material.normal_path, encoding="linear")
    kwargs: dict[str, Any] = {
        "vis_mode": vis_mode,
        "double_sided": double_sided,
        "smooth": smooth,
    }

    if diffuse_tex is not None:
        kwargs["diffuse_texture"] = diffuse_tex
        if float(opacity) < 0.999:
            kwargs["opacity"] = float(opacity)
    else:
        base_rgb = _rgb(material.base_color)
        if tint_rgb is not None:
            base_rgb = _mix_rgb(base_rgb, tint_rgb, 0.62)
        kwargs["color"] = _rgba(base_rgb, alpha=opacity)

    if roughness_tex is not None:
        kwargs["roughness_texture"] = roughness_tex
    else:
        kwargs["roughness"] = float(roughness_override if roughness_override is not None else material.roughness)

    if normal_tex is not None:
        kwargs["normal_texture"] = normal_tex
        kwargs["normal_diff_clamp"] = 85.0

    metallic_value = metallic_override if metallic_override is not None else material.metallic
    if metallic_value > 0.0:
        kwargs["metallic"] = float(metallic_value)

    return gs.surfaces.Default(**{k: v for k, v in kwargs.items() if v is not None})


def _scene_material_keys(case: CaseSpec) -> dict[str, str]:
    surface = _SURFACE_CATALOG[case.surface_key]
    keys = {
        "floor": surface.floor_material_key,
        "floor_alt": "floor_wood_weathered",
        "stage": "leather_brown",
        "wall": surface.wall_material_key,
        "wall_alt": "wall_beige" if surface.wall_material_key == "wall_cream" else "wall_cream",
        "trim": "wood_dark",
        "window_frame": "wood_dark",
        "window_glow": "plastic_white",
        "fabric": "fabric_curtain",
        "cabinet": "wall_cream",
        "wood_mid": "wood_plywood",
        "wood_dark": "wood_dark",
        "desk_top": "wood_dark",
        "desk_leg": "concrete_painted",
        "metal": "painted_metal_teal",
        "metal_accent": "painted_metal_yellow",
        "clutter": "cardboard_kraft",
        "soft": case.soft_material_key,
        "soft_secondary": case.soft_secondary_material_key or case.soft_material_key,
        "rigid": case.rigid_material_key,
    }
    if case.scene_theme == "industrial_cool":
        keys.update(
            floor="concrete_clean_floor",
            floor_alt="concrete_clean_floor",
            stage="wood_dark",
            wall="wall_cream",
            wall_alt="concrete_clean_wall",
            cabinet="wall_beige",
            desk_top="wood_dark",
            desk_leg="concrete_clean_wall",
            metal="painted_metal_teal",
            metal_accent="painted_metal_yellow",
        )
    elif case.scene_theme == "soft_daylight":
        keys.update(
            floor="floor_wood",
            floor_alt="floor_wood_weathered",
            stage="leather_brown",
            wall="wall_cream",
            wall_alt="wall_beige",
            cabinet="wall_cream",
            desk_top="wood_plywood",
            metal="painted_metal_teal",
        )
    elif case.scene_theme == "loft_neutral":
        keys.update(
            floor="wood_dark",
            floor_alt="floor_wood_weathered",
            stage="leather_brown",
            wall="wall_beige",
            wall_alt="concrete_painted",
            cabinet="concrete_painted",
            desk_top="wood_dark",
            desk_leg="concrete_painted",
        )
    return keys


def _scene_lights(case: CaseSpec) -> list[dict[str, Any]]:
    lighting = _LIGHTING_CATALOG[case.lighting_key]
    key_scale = float(lighting.key_light_intensity)
    fill_scale = float(lighting.fill_light_intensity)
    rim_scale = float(lighting.rim_light_intensity)
    lights = [
        {
            "type": "directional",
            "dir": (-0.86, 0.22, -0.92),
            "color": (1.00, 0.95, 0.88),
            "intensity": 5.8 * key_scale,
        },
        {
            "type": "directional",
            "dir": (0.58, -0.18, -0.62),
            "color": (0.82, 0.90, 1.00),
            "intensity": 3.1 * fill_scale,
        },
        {
            "type": "point",
            "pos": (1.55, 2.10, 2.30),
            "color": (1.00, 0.97, 0.92),
            "intensity": 16.0 * rim_scale,
        },
    ]
    if case.scene_theme == "industrial_cool":
        lights[-1]["pos"] = (0.10, -2.30, 1.95)
        lights[-1]["color"] = (1.00, 0.92, 0.82)
        lights[-1]["intensity"] = 10.5 * key_scale
    elif case.scene_theme == "soft_daylight":
        lights[-1]["pos"] = (1.95, 2.55, 2.05)
        lights[-1]["color"] = (0.92, 0.96, 1.00)
        lights[-1]["intensity"] = 14.0 * rim_scale
    return lights


def _soft_surface(case: CaseSpec, *, secondary: bool = False, vis_mode: str | None = None) -> Any:
    material_key = case.soft_secondary_material_key if secondary and case.soft_secondary_material_key else case.soft_material_key
    material = _MATERIAL_CATALOG[material_key]
    tint = _mix_rgb(_rgb(material.base_color), _rgb(case.palette.soft_color), 0.78 if not secondary else 0.62)
    if vis_mode == "visual":
        lift = 0.24 if case.scene_theme == "industrial_cool" else 0.18
        tint = _mix_rgb(tint, (0.96, 0.97, 0.98), lift if not secondary else lift + 0.06)
    return gs.surfaces.Default(
        color=_rgba(tint),
        roughness=max(0.58, float(material.roughness)),
        metallic=0.0,
        smooth=True,
        vis_mode=vis_mode,
    )


def _soft_marker_rgba(
    points: np.ndarray,
    *,
    tint: tuple[float, float, float],
    accent: tuple[float, float, float],
) -> np.ndarray:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    coords = (points - bbox_min) / extent
    u = coords[:, 0]
    v = coords[:, 1]
    w = coords[:, 2]

    base_rgb = _mix_rgb(tint, (0.97, 0.98, 0.99), 0.58)
    cool_rgb = _mix_rgb(tint, (0.00, 0.92, 1.00), 0.72)
    warm_rgb = _mix_rgb(accent, (1.00, 0.46, 0.10), 0.62)
    deep_rgb = _mix_rgb(tint, (0.05, 0.10, 0.24), 0.78)
    dark_rgb = (0.06, 0.07, 0.09)
    light_rgb = (0.99, 0.99, 0.99)

    colors = np.tile(np.array(base_rgb, dtype=np.float32), (len(points), 1))
    cool = np.tile(np.array(cool_rgb, dtype=np.float32), (len(points), 1))
    warm = np.tile(np.array(warm_rgb, dtype=np.float32), (len(points), 1))
    deep = np.tile(np.array(deep_rgb, dtype=np.float32), (len(points), 1))
    dark = np.tile(np.array(dark_rgb, dtype=np.float32), (len(points), 1))
    light = np.tile(np.array(light_rgb, dtype=np.float32), (len(points), 1))

    left_panel = u < 0.34
    right_panel = u > 0.68
    low_panel = v < 0.24
    top_cap = w > 0.76
    front_band = np.abs(v - 0.52) < 0.08
    spine = np.abs(u - 0.50) < 0.05
    edge_emphasis = (u < 0.06) | (u > 0.94) | (v < 0.06) | (v > 0.94) | (w < 0.06) | (w > 0.94)

    colors[left_panel] = cool[left_panel]
    colors[right_panel] = warm[right_panel]
    colors[low_panel] = deep[low_panel] * 0.80 + colors[low_panel] * 0.20
    colors[top_cap] = light[top_cap]
    colors[front_band] = colors[front_band] * 0.35 + light[front_band] * 0.65
    colors[spine] = dark[spine]

    coarse_checker = (np.mod(np.floor(u * 4.0) + np.floor(w * 4.0), 2.0) < 0.5)
    colors[coarse_checker] = colors[coarse_checker] * 0.78 + light[coarse_checker] * 0.22

    diagonal = np.abs((u * 1.15 + w * 0.85) - 0.92) < 0.09
    colors[diagonal] = warm[diagonal] * 0.68 + light[diagonal] * 0.32

    colors[edge_emphasis] = colors[edge_emphasis] * 0.30 + dark[edge_emphasis] * 0.70

    rgba = np.concatenate([np.clip(colors, 0.0, 1.0), np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return np.round(rgba * 255.0).astype(np.uint8)


def _apply_soft_visual_markers(case: CaseSpec, entities: dict[str, Any]) -> None:
    for name, entity in entities.items():
        if not hasattr(entity, "get_particles_pos") or not hasattr(entity, "vmesh"):
            continue
        if entity.vmesh is None or not hasattr(entity.vmesh, "trimesh"):
            continue

        secondary = "top" in name
        tint = _rgb(case.palette.soft_color if not secondary else _mix_rgb(_rgb(case.palette.soft_color), _rgb(case.palette.accent_color), 0.24))
        accent = _mix_rgb(_rgb(case.palette.accent_color), (1.0, 1.0, 1.0), 0.18 if not secondary else 0.32)
        mesh = entity.vmesh.trimesh
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        vertex_colors = _soft_marker_rgba(vertices, tint=tint, accent=accent)
        face_colors = None
        if hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0:
            faces = np.asarray(mesh.faces, dtype=np.int64)
            face_points = vertices[faces].mean(axis=1)
            face_colors = _soft_marker_rgba(face_points, tint=tint, accent=accent)
        entity.vmesh.trimesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            face_colors=face_colors,
            vertex_colors=vertex_colors,
        )


def _rigid_surface(case: CaseSpec) -> Any:
    tint = _mix_rgb(_rgb(_MATERIAL_CATALOG[case.rigid_material_key].base_color), _rgb(case.palette.rigid_color), 0.72)
    return _material_surface(
        case.rigid_material_key,
        tint_rgb=tint,
        roughness_override=0.34,
        metallic_override=max(0.18, _MATERIAL_CATALOG[case.rigid_material_key].metallic),
        smooth=True,
    )


def _record_frame_indices(num_steps: int) -> list[int]:
    if num_steps <= 1:
        return [0]
    mid = num_steps // 2
    return sorted({0, mid, num_steps - 1})


def _resample_video_frames_to_fps(
    raw_frames: list[np.ndarray],
    *,
    sim_dt: float,
    target_fps: int,
    playback_speed: float = 1.0,
) -> list[np.ndarray]:
    if not raw_frames:
        return []
    speed = max(float(playback_speed), 1e-6)
    sim_duration = float(sim_dt) * len(raw_frames) / speed
    target_frame_count = max(1, min(len(raw_frames), int(round(sim_duration * float(target_fps)))))
    if target_frame_count == len(raw_frames):
        return raw_frames
    sampled_positions = np.linspace(0, len(raw_frames) - 1, num=target_frame_count)
    sampled_indices = np.clip(np.rint(sampled_positions).astype(int), 0, len(raw_frames) - 1)
    return [raw_frames[idx] for idx in sampled_indices.tolist()]


def _render_rgb(camera: Any) -> np.ndarray:
    rgb = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _soft_block_state(block: Any) -> dict[str, Any]:
    pts = _to_numpy(block.get_particles_pos()).reshape(-1, 3)
    centroid = pts.mean(axis=0)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    return {
        "n_particles": int(pts.shape[0]),
        "centroid": centroid.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_size": bbox_size.tolist(),
    }


def _rigid_body_state(body: Any) -> dict[str, Any]:
    pos = _to_numpy(body.get_pos()).reshape(-1)[:3]
    quat = _to_numpy(body.get_quat()).reshape(-1)[:4]
    vel = _to_numpy(body.get_vel()).reshape(-1)[:3]
    ang = _to_numpy(body.get_ang()).reshape(-1)[:3]
    return {
        "pos": pos.tolist(),
        "quat": quat.tolist(),
        "vel": vel.tolist(),
        "ang": ang.tolist(),
    }


def _bsdf_surface(
    color: tuple[float, float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    double_sided: bool | None = None,
    vis_mode: str | None = None,
) -> Any:
    return gs.surfaces.Default(
        color=color,
        roughness=float(roughness),
        metallic=float(metallic),
        double_sided=double_sided,
        vis_mode=vis_mode,
    )


def _rigid_material(*, rho: float, friction: float, coup_friction: float) -> Any:
    return gs.materials.Rigid(
        rho=float(rho),
        friction=float(friction),
        coup_friction=float(coup_friction),
    )


def _add_fixed_box(
    scene: Any,
    *,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float, float] | None = None,
    roughness: float = 0.8,
    metallic: float = 0.0,
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision: bool = True,
    surface: Any | None = None,
    material: Any | None = None,
) -> Any:
    rigid_material = material or _rigid_material(rho=720.0, friction=0.82, coup_friction=0.98)
    if surface is not None and hasattr(surface, "requires_uv") and surface.requires_uv():
        if collision:
            scene.add_entity(
                material=rigid_material,
                morph=gs.morphs.Box(
                    pos=pos,
                    size=size,
                    euler=euler,
                    fixed=True,
                    collision=True,
                    visualization=False,
                ),
            )
        return scene.add_entity(
            material=_rigid_material(rho=720.0, friction=0.82, coup_friction=0.98),
            morph=gs.morphs.Mesh(
                file=str(_ensure_uv_box_mesh_path()),
                scale=size,
                pos=pos,
                euler=euler,
                fixed=True,
                collision=False,
                decimate=False,
                convexify=False,
            ),
            surface=surface,
        )

    return scene.add_entity(
        material=rigid_material,
        morph=gs.morphs.Box(
            pos=pos,
            size=size,
            euler=euler,
            fixed=True,
            collision=collision,
            visualization=True,
        ),
        surface=surface or _bsdf_surface(color or (0.7, 0.7, 0.7, 1.0), roughness=roughness, metallic=metallic),
    )


def _add_fixed_cylinder(
    scene: Any,
    *,
    pos: tuple[float, float, float],
    radius: float,
    height: float,
    color: tuple[float, float, float, float] | None = None,
    roughness: float = 0.8,
    metallic: float = 0.0,
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision: bool = True,
    surface: Any | None = None,
    material: Any | None = None,
) -> Any:
    return scene.add_entity(
        material=material or _rigid_material(rho=735.0, friction=0.80, coup_friction=0.96),
        morph=gs.morphs.Cylinder(
            pos=pos,
            radius=radius,
            height=height,
            euler=euler,
            fixed=True,
            collision=collision,
            visualization=True,
        ),
        surface=surface or _bsdf_surface(color or (0.7, 0.7, 0.7, 1.0), roughness=roughness, metallic=metallic),
    )


def _should_add_front_portal(case: CaseSpec) -> bool:
    # The shallow dataset cameras mostly sit just outside the room and look inward. For those
    # shots, the decorative front portal reads as a hard occluder instead of a framing element.
    # Keep it only for genuinely interior viewpoints that are already well inside the doorway.
    cam_y = float(case.camera.pos[1])
    lookat_y = float(case.camera.lookat[1])
    return cam_y > -0.95 and lookat_y > -0.08


def _should_add_industrial_floor_clutter(case: CaseSpec) -> bool:
    # The low industrial clutter block is visual-only. For the lateral cylinder swipe families it
    # sits directly in the action corridor and reads like a collision object, creating a false
    # "tunneling" impression even when the solver is behaving as configured.
    return case.motion_profile not in {"cylinder_swipe", "reverse_swipe"}


def _add_room_shell(scene: Any, case: CaseSpec) -> None:
    p = case.palette
    mats = _scene_material_keys(case)
    floor_surface = _material_surface(mats["floor"], tint_rgb=_rgb(p.floor_color), smooth=True)
    floor_alt_surface = _material_surface(mats["floor_alt"], tint_rgb=_mix_rgb(_rgb(p.floor_color), _rgb(p.wood_mid_color), 0.28), smooth=True)
    stage_tint = _mix_rgb(_rgb(p.floor_color), _rgb(p.fabric_color), 0.34)
    if case.scene_theme == "soft_daylight":
        stage_tint = _mix_rgb(_rgb(p.floor_color), _rgb(p.ceiling_color), 0.22)
    elif case.scene_theme == "industrial_cool":
        stage_tint = _mix_rgb(_rgb(p.floor_color), _rgb(p.wall_alt_color), 0.18)
    elif case.scene_theme == "loft_neutral":
        stage_tint = _mix_rgb(_rgb(p.floor_color), _rgb(p.wall_color), 0.16)
    if case.motion_profile == "overhead_press":
        stage_tint = _mix_rgb(_rgb(p.floor_color), _rgb(p.ceiling_color), 0.28)
    elif case.motion_profile == "wall_pinch":
        stage_tint = _mix_rgb(_rgb(p.floor_color), _rgb(p.ceiling_color), 0.24)
    elif case.motion_profile == "ledge_topple":
        stage_tint = _mix_rgb(_rgb(p.floor_color), _rgb(p.ceiling_color), 0.20)
    stage_surface = _material_surface(mats["stage"], tint_rgb=stage_tint, smooth=True)
    wall_surface = _material_surface(mats["wall"], tint_rgb=_rgb(p.wall_color), smooth=True)
    wall_alt_surface = _material_surface(mats["wall_alt"], tint_rgb=_rgb(p.wall_alt_color), smooth=True)
    ceiling_surface = _material_surface("plastic_white", tint_rgb=_rgb(p.ceiling_color), roughness_override=0.28, smooth=True)
    trim_surface = _material_surface(mats["trim"], tint_rgb=_rgb(p.trim_color), roughness_override=0.74, smooth=True)
    window_frame_surface = _material_surface(mats["window_frame"], tint_rgb=_rgb(p.wood_dark_color), roughness_override=0.58, smooth=True)
    window_glow_surface = _material_surface(mats["window_glow"], tint_rgb=(1.0, 1.0, 1.0), roughness_override=0.12, smooth=True)
    fabric_surface = _material_surface(mats["fabric"], tint_rgb=_rgb(p.fabric_color), roughness_override=0.96, double_sided=True, smooth=True)
    wood_mid_surface = _material_surface(mats["wood_mid"], tint_rgb=_rgb(p.wood_mid_color), smooth=True)
    wood_dark_surface = _material_surface(mats["wood_dark"], tint_rgb=_rgb(p.wood_dark_color), roughness_override=0.62, smooth=True)
    cabinet_surface = _material_surface(mats["cabinet"], tint_rgb=_mix_rgb(_rgb(p.wall_color), _rgb(p.wood_mid_color), 0.12), smooth=True)
    desk_top_surface = _material_surface(mats["desk_top"], tint_rgb=_rgb(p.wood_dark_color), smooth=True)
    desk_leg_surface = _material_surface(mats["desk_leg"], tint_rgb=_rgb(p.wall_alt_color), smooth=True)
    metal_surface = _material_surface(mats["metal"], tint_rgb=_rgb(p.metal_color), roughness_override=0.34, metallic_override=0.42, smooth=True)
    metal_accent_surface = _material_surface(mats["metal_accent"], tint_rgb=_rgb(p.accent_color), roughness_override=0.28, metallic_override=0.36, smooth=True)
    clutter_surface = _material_surface(mats["clutter"], tint_rgb=_rgb(p.clutter_color), roughness_override=0.86, smooth=True)
    pale_panel_surface = _material_surface(mats["wall"], tint_rgb=_mix_rgb(_rgb(p.wall_color), _rgb(p.ceiling_color), 0.18), smooth=True)
    dark_panel_surface = _material_surface(mats["wall_alt"], tint_rgb=_mix_rgb(_rgb(p.wall_alt_color), _rgb(p.trim_color), 0.30), smooth=True)
    book_surface = _material_surface(mats["metal_accent"], tint_rgb=_mix_rgb(_rgb(p.accent_color), _rgb(p.fabric_color), 0.22), roughness_override=0.72, metallic_override=0.0, smooth=True)
    neutral_book_surface = _material_surface(mats["cabinet"], tint_rgb=_mix_rgb(_rgb(p.fabric_color), _rgb(p.wall_color), 0.36), roughness_override=0.74, smooth=True)

    stage_size = (1.32, 1.04, 0.010)
    stage_pos = (0.12, 0.16, 0.005)
    if case.motion_profile in {"tall_flip", "overhead_press"}:
        stage_size = (1.18, 0.92, 0.010)
        stage_pos = (0.08, 0.10, 0.005)
    elif case.motion_profile in {"ramp_slide", "ledge_topple"}:
        stage_size = (1.10, 0.86, 0.010)
        stage_pos = (-0.06, 0.08, 0.005)

    # Main room shell and a slightly elevated central stage to focus the interaction area.
    _add_fixed_box(scene, pos=(0.0, 0.60, -0.04), size=(5.4, 4.8, 0.08), surface=floor_surface)
    _add_fixed_box(scene, pos=stage_pos, size=stage_size, surface=stage_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.35, 0.55, 0.003), size=(1.10, 1.55, 0.010), surface=floor_alt_surface, collision=False)
    _add_fixed_box(scene, pos=(1.55, 1.65, 0.003), size=(1.00, 1.35, 0.010), surface=floor_alt_surface, collision=False)
    _add_fixed_box(scene, pos=(0.98, -0.24, 0.002), size=(0.78, 0.22, 0.006), surface=floor_alt_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.86, 1.48, 0.002), size=(0.52, 0.86, 0.006), surface=floor_alt_surface, collision=False)
    _add_fixed_box(scene, pos=(0.0, 2.88, 1.55), size=(5.4, 0.08, 3.10), surface=wall_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.68, 0.60, 1.55), size=(0.08, 4.8, 3.10), surface=wall_alt_surface, collision=False)
    _add_fixed_box(scene, pos=(2.68, 0.60, 1.55), size=(0.08, 4.8, 3.10), surface=wall_alt_surface, collision=False)
    _add_fixed_box(scene, pos=(0.0, 0.60, 3.13), size=(5.4, 4.8, 0.08), surface=ceiling_surface, collision=False)
    _add_fixed_box(scene, pos=(0.0, 2.83, 0.07), size=(5.18, 0.04, 0.14), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.64, 0.60, 0.07), size=(0.04, 4.66, 0.14), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(2.64, 0.60, 0.07), size=(0.04, 4.66, 0.14), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(0.0, 2.82, 0.74), size=(5.16, 0.05, 0.72), surface=wall_alt_surface, collision=False)
    _add_fixed_box(scene, pos=(0.00, 2.84, 2.55), size=(5.12, 0.03, 0.22), surface=pale_panel_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.18, 2.81, 1.56), size=(0.72, 0.02, 0.94), surface=pale_panel_surface, collision=False)
    _add_fixed_box(scene, pos=(2.12, 2.81, 1.42), size=(0.54, 0.02, 0.72), surface=dark_panel_surface, collision=False)
    # Keep the near-camera portal only for deep interior views. The default preview cameras are
    # mostly doorway-adjacent or slightly outside the room, where the portal blocks the action.
    if _should_add_front_portal(case):
        _add_fixed_box(scene, pos=(-1.92, -1.76, 1.10), size=(1.28, 0.04, 2.20), surface=wall_surface, collision=False)
        _add_fixed_box(scene, pos=(1.92, -1.76, 1.10), size=(1.28, 0.04, 2.20), surface=wall_surface, collision=False)
        _add_fixed_box(scene, pos=(0.00, -1.76, 2.42), size=(2.72, 0.04, 0.42), surface=wall_surface, collision=False)
        _add_fixed_box(scene, pos=(0.00, -1.75, 0.08), size=(2.80, 0.06, 0.16), surface=trim_surface, collision=False)

    # Window, curtain, and beam structure on the rear wall.
    _add_fixed_box(scene, pos=(1.52, 2.80, 2.00), size=(1.56, 0.03, 1.22), surface=window_frame_surface, collision=False)
    _add_fixed_box(scene, pos=(1.52, 2.77, 2.00), size=(1.30, 0.02, 1.02), surface=window_glow_surface, collision=False)
    _add_fixed_cylinder(scene, pos=(1.52, 2.74, 2.57), radius=0.025, height=1.78, surface=metal_surface, euler=(0.0, 0.0, 90.0), collision=False)
    _add_fixed_box(scene, pos=(0.88, 2.74, 1.94), size=(0.44, 0.02, 1.22), surface=fabric_surface, collision=False)
    _add_fixed_box(scene, pos=(2.16, 2.74, 1.94), size=(0.44, 0.02, 1.22), surface=fabric_surface, collision=False)
    _add_fixed_box(scene, pos=(1.52, 2.73, 1.30), size=(1.36, 0.10, 0.05), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(1.54, 2.70, 0.44), size=(0.92, 0.12, 0.54), surface=pale_panel_surface, collision=False)
    _add_fixed_box(scene, pos=(1.22, 2.66, 0.44), size=(0.05, 0.04, 0.46), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(1.52, 2.66, 0.44), size=(0.05, 0.04, 0.46), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(1.82, 2.66, 0.44), size=(0.05, 0.04, 0.46), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(0.00, 2.05, 3.02), size=(5.00, 0.10, 0.08), surface=wood_dark_surface, collision=False)

    # Left-side door frame and cabinet cluster.
    _add_fixed_box(scene, pos=(-2.62, 2.12, 1.02), size=(0.05, 0.74, 2.04), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.50, 2.12, 2.01), size=(0.22, 0.70, 0.08), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.48, 2.08, 1.00), size=(0.02, 0.58, 1.90), surface=wall_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.84, 1.95, 0.58), size=(1.24, 0.38, 1.16), surface=cabinet_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.84, 1.94, 1.10), size=(1.12, 0.31, 0.08), surface=wood_dark_surface, collision=False)
    _add_fixed_cylinder(scene, pos=(-1.28, 1.78, 0.66), radius=0.015, height=0.36, surface=metal_accent_surface, euler=(0.0, 0.0, 90.0), collision=False)
    _add_fixed_box(scene, pos=(-1.34, 2.30, 0.20), size=(0.54, 0.26, 0.26), surface=clutter_surface, euler=(0.0, 0.0, -6.0), collision=False)
    _add_fixed_box(scene, pos=(-2.18, 2.80, 1.78), size=(0.58, 0.06, 1.42), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.40, 2.72, 1.00), size=(0.24, 0.12, 0.32), surface=clutter_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.82, 1.74, 1.46), size=(1.00, 0.16, 0.06), surface=wood_mid_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.10, 1.72, 1.56), size=(0.18, 0.10, 0.18), surface=neutral_book_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.88, 1.72, 1.58), size=(0.10, 0.12, 0.22), surface=book_surface, collision=False)
    _add_fixed_box(scene, pos=(-1.70, 1.72, 1.54), size=(0.14, 0.10, 0.16), surface=clutter_surface, collision=False)
    _add_fixed_box(scene, pos=(-0.96, 2.80, 1.70), size=(0.52, 0.03, 0.72), surface=trim_surface, collision=False)
    _add_fixed_box(scene, pos=(-0.96, 2.78, 1.70), size=(0.40, 0.02, 0.58), surface=pale_panel_surface, collision=False)

    # Right-side desk, stool, and foreground clutter.
    _add_fixed_box(scene, pos=(1.86, 0.98, 0.78), size=(1.42, 0.74, 0.08), surface=desk_top_surface, collision=False)
    _add_fixed_box(scene, pos=(1.22, 0.64, 0.38), size=(0.08, 0.08, 0.76), surface=desk_leg_surface, collision=False)
    _add_fixed_box(scene, pos=(2.50, 0.64, 0.38), size=(0.08, 0.08, 0.76), surface=desk_leg_surface, collision=False)
    _add_fixed_cylinder(scene, pos=(1.76, 0.62, 0.52), radius=0.20, height=0.06, surface=stage_surface, collision=False)
    _add_fixed_cylinder(scene, pos=(1.76, 0.62, 0.26), radius=0.05, height=0.50, surface=metal_surface, collision=False)
    _add_fixed_box(scene, pos=(2.06, -0.02, 0.09), size=(0.36, 0.22, 0.18), surface=clutter_surface, euler=(0.0, 0.0, 8.0), collision=False)
    _add_fixed_cylinder(scene, pos=(1.72, 0.20, 0.11), radius=0.06, height=0.22, surface=metal_accent_surface, collision=False)
    _add_fixed_box(scene, pos=(2.12, 1.02, 0.90), size=(0.24, 0.14, 0.06), surface=neutral_book_surface, collision=False)
    _add_fixed_box(scene, pos=(2.34, 0.96, 0.92), size=(0.16, 0.10, 0.10), surface=book_surface, collision=False)
    _add_fixed_box(scene, pos=(2.50, -0.20, 0.44), size=(0.28, 0.28, 0.70), surface=dark_panel_surface, collision=False)
    _add_fixed_box(scene, pos=(2.50, -0.20, 0.82), size=(0.20, 0.22, 0.06), surface=wood_dark_surface, collision=False)

    # Ceiling/floor pipes and small structure elements to avoid the empty-simulator look.
    _add_fixed_cylinder(scene, pos=(-2.28, 2.20, 2.48), radius=0.03, height=1.78, surface=metal_surface, euler=(90.0, 0.0, 0.0), collision=False)
    _add_fixed_cylinder(scene, pos=(2.24, 0.16, 1.98), radius=0.03, height=2.12, surface=metal_surface, collision=False)
    _add_fixed_cylinder(scene, pos=(-2.18, -0.36, 2.76), radius=0.018, height=1.98, surface=metal_surface, collision=False)
    _add_fixed_box(scene, pos=(2.38, 2.84, 2.46), size=(0.28, 0.03, 0.22), surface=ceiling_surface, collision=False)
    _add_fixed_box(scene, pos=(-0.92, 2.83, 1.32), size=(0.56, 0.04, 0.32), surface=wood_mid_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.44, 0.20, 2.86), size=(0.18, 2.30, 0.10), surface=metal_surface, collision=False)
    _add_fixed_box(scene, pos=(-2.44, -0.76, 2.70), size=(0.18, 0.18, 0.18), surface=desk_leg_surface, collision=False)
    _add_fixed_box(scene, pos=(0.52, 2.84, 2.46), size=(0.54, 0.03, 0.18), surface=ceiling_surface, collision=False)

    if case.scene_theme == "industrial_cool":
        _add_fixed_box(scene, pos=(0.00, 2.82, 1.20), size=(0.96, 0.04, 0.56), surface=desk_leg_surface, collision=False)
        if _should_add_industrial_floor_clutter(case):
            _add_fixed_box(scene, pos=(-0.25, -0.18, 0.05), size=(0.50, 0.18, 0.10), surface=clutter_surface, collision=False)
        _add_fixed_box(scene, pos=(-2.16, 0.45, 2.30), size=(0.34, 0.14, 0.22), surface=metal_surface, collision=False)
        _add_fixed_box(scene, pos=(-1.92, 1.30, 1.36), size=(0.52, 0.16, 0.52), surface=wall_alt_surface, collision=False)
        _add_fixed_box(scene, pos=(-2.00, 1.18, 1.12), size=(0.14, 0.10, 0.08), surface=metal_accent_surface, collision=False)
        _add_fixed_box(scene, pos=(2.34, 1.88, 1.12), size=(0.22, 0.22, 1.10), surface=pale_panel_surface, collision=False)
    elif case.scene_theme == "soft_daylight":
        _add_fixed_box(scene, pos=(2.30, 2.76, 2.45), size=(0.28, 0.02, 0.22), surface=ceiling_surface, collision=False)
        _add_fixed_box(scene, pos=(-1.88, 2.30, 1.30), size=(0.24, 0.08, 0.36), surface=wood_mid_surface, collision=False)
        _add_fixed_box(scene, pos=(2.16, 1.62, 0.14), size=(0.24, 0.24, 0.28), surface=clutter_surface, collision=False)
        _add_fixed_box(scene, pos=(-2.06, 0.04, 0.28), size=(0.20, 0.42, 0.56), surface=pale_panel_surface, collision=False)
        _add_fixed_box(scene, pos=(-2.06, 0.04, 0.60), size=(0.12, 0.12, 0.08), surface=wood_mid_surface, collision=False)
    elif case.scene_theme == "loft_neutral":
        _add_fixed_box(scene, pos=(-0.10, 2.84, 2.62), size=(1.10, 0.05, 0.18), surface=wood_dark_surface, collision=False)
        _add_fixed_box(scene, pos=(-2.12, 0.10, 1.10), size=(0.12, 0.42, 2.20), surface=desk_leg_surface, collision=False)
        _add_fixed_box(scene, pos=(2.36, 1.86, 0.40), size=(0.26, 0.26, 0.80), surface=clutter_surface, collision=False)
        _add_fixed_box(scene, pos=(-1.86, 1.30, 0.42), size=(0.68, 0.24, 0.84), surface=dark_panel_surface, collision=False)
        _add_fixed_box(scene, pos=(-1.86, 1.28, 0.86), size=(0.56, 0.18, 0.06), surface=wood_dark_surface, collision=False)
        _add_fixed_box(scene, pos=(-1.84, 1.22, 0.96), size=(0.12, 0.10, 0.18), surface=book_surface, collision=False)


def _scene_common(case: CaseSpec, camera: CameraSpec, mpm_vis_mode: str) -> tuple[Any, Any]:
    lighting = _LIGHTING_CATALOG[case.lighting_key]
    ambient = 0.16 + float(lighting.ambient_boost)
    bg_color = _mix_rgb(_rgb(case.palette.wall_color), (0.93, 0.94, 0.95), 0.20)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=case.sim.dt,
            substeps=case.sim.substeps,
            gravity=case.sim.gravity,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=case.sim.mpm_lower_bound,
            upper_bound=case.sim.mpm_upper_bound,
            grid_density=case.sim.grid_density,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=case.sim.dt,
            gravity=case.sim.gravity,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera.pos,
            camera_lookat=camera.lookat,
            camera_fov=camera.fov,
            res=camera.res,
            max_FPS=max(DEFAULT_FPS, 60),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            visualize_mpm_boundary=False,
            shadow=True,
            ambient_light=(ambient, ambient, ambient + 0.01),
            background_color=bg_color,
            plane_reflection=False,
            lights=_scene_lights(case),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    cam = scene.add_camera(
        res=camera.res,
        pos=camera.pos,
        lookat=camera.lookat,
        fov=camera.fov,
        GUI=False,
    )
    _add_room_shell(scene, case)

    return scene, cam


def _add_soft_box(
    scene: Any,
    *,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    euler: tuple[float, float, float],
    color: tuple[float, float, float, float] | None,
    vis_mode: str,
    E: float,
    nu: float,
    rho: float,
    surface: Any | None = None,
) -> Any:
    return scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=E,
            nu=nu,
            rho=rho,
            sampler="pbs-16",
            model="neohooken",
        ),
        morph=gs.morphs.Box(
            pos=pos,
            size=size,
            euler=euler,
        ),
        surface=surface or _bsdf_surface(color or (0.72, 0.48, 0.36, 1.0), roughness=0.68, metallic=0.02, vis_mode=vis_mode),
    )


def _build_case_entities(scene: Any, case: CaseSpec, mpm_vis_mode: str) -> dict[str, Any]:
    p = case.palette
    mats = _scene_material_keys(case)
    entities: dict[str, Any] = {}
    primary_soft_surface = _soft_surface(case, vis_mode=mpm_vis_mode)
    secondary_soft_surface = _soft_surface(case, secondary=True, vis_mode=mpm_vis_mode)
    rigid_surface = _rigid_surface(case)
    support_surface = _material_surface(mats["wood_mid"], tint_rgb=_rgb(p.wood_mid_color), smooth=True)
    support_rigid_material = _rigid_material(rho=860.0, friction=0.58, coup_friction=0.90)

    if case.motion_profile == "drop":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.08, 0.05, 0.60),
            size=(0.28, 0.28, 0.28),
            euler=(12.0, 0.0, 18.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.8e4,
            nu=0.25,
            rho=240.0,
            surface=primary_soft_surface,
        )
        return entities

    if case.motion_profile == "sphere_impact":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.18, 0.00, 0.22),
            size=(0.32, 0.24, 0.24),
            euler=(0.0, 0.0, 10.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.3e4,
            nu=0.24,
            rho=220.0,
            surface=primary_soft_surface,
        )
        entities["rigid_sphere"] = scene.add_entity(
            material=gs.materials.Rigid(rho=900.0, friction=0.45, coup_friction=0.9),
            morph=gs.morphs.Sphere(
                pos=(-0.72, -0.02, 0.48),
                radius=0.10,
            ),
            surface=rigid_surface,
        )
        return entities

    if case.motion_profile == "soft_stack":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.16, 0.02, 0.18),
            size=(0.34, 0.26, 0.20),
            euler=(0.0, 0.0, -6.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.1e4,
            nu=0.23,
            rho=215.0,
            surface=primary_soft_surface,
        )
        entities["soft_block_top"] = _add_soft_box(
            scene,
            pos=(-0.08, 0.06, 0.64),
            size=(0.20, 0.20, 0.20),
            euler=(18.0, 0.0, 22.0),
            color=(p.accent_color[0], p.accent_color[1] * 0.92, p.accent_color[2] * 0.92, 1.0),
            vis_mode=mpm_vis_mode,
            E=5.2e4,
            nu=0.26,
            rho=245.0,
            surface=secondary_soft_surface,
        )
        return entities

    if case.motion_profile == "cylinder_swipe":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.26, -0.02, 0.20),
            size=(0.34, 0.22, 0.22),
            euler=(0.0, 0.0, 8.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.4e4,
            nu=0.24,
            rho=225.0,
            surface=primary_soft_surface,
        )
        entities["rigid_cylinder"] = scene.add_entity(
            material=_rigid_material(rho=965.0, friction=0.34, coup_friction=0.78),
            morph=gs.morphs.Cylinder(
                pos=(-0.86, -0.24, 0.135),
                radius=0.09,
                height=0.22,
                euler=(90.0, 0.0, 0.0),
            ),
            surface=rigid_surface,
        )
        return entities

    if case.motion_profile == "corner_drop":
        _add_fixed_box(
            scene,
            pos=(0.62, 0.42, 0.11),
            size=(0.38, 0.26, 0.22),
            euler=(0.0, 0.0, -8.0),
            surface=support_surface,
            material=support_rigid_material,
            collision=True,
        )
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.12, 0.22, 0.58),
            size=(0.26, 0.28, 0.26),
            euler=(24.0, 4.0, 28.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.9e4,
            nu=0.25,
            rho=235.0,
            surface=primary_soft_surface,
        )
        return entities

    if case.motion_profile == "tall_flip":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.10, -0.02, 0.54),
            size=(0.22, 0.22, 0.42),
            euler=(34.0, 8.0, 26.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.7e4,
            nu=0.24,
            rho=230.0,
            surface=primary_soft_surface,
        )
        return entities

    if case.motion_profile == "dual_sphere_press":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.04, 0.02, 0.22),
            size=(0.34, 0.24, 0.24),
            euler=(0.0, 0.0, -8.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.2e4,
            nu=0.24,
            rho=220.0,
            surface=primary_soft_surface,
        )
        entities["rigid_sphere_left"] = scene.add_entity(
            material=_rigid_material(rho=910.0, friction=0.44, coup_friction=0.88),
            morph=gs.morphs.Sphere(
                pos=(-0.82, -0.06, 0.30),
                radius=0.10,
            ),
            surface=rigid_surface,
        )
        entities["rigid_sphere_right"] = scene.add_entity(
            material=_rigid_material(rho=940.0, friction=0.39, coup_friction=0.84),
            morph=gs.morphs.Sphere(
                pos=(0.88, 0.16, 0.50),
                radius=0.09,
            ),
            surface=_material_surface(mats["metal_accent"], tint_rgb=_rgb(p.accent_color), roughness_override=0.28, metallic_override=0.34, smooth=True),
        )
        return entities

    if case.motion_profile == "glancing_arc":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.06, 0.08, 0.22),
            size=(0.32, 0.24, 0.24),
            euler=(0.0, 0.0, 12.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.4e4,
            nu=0.24,
            rho=220.0,
            surface=primary_soft_surface,
        )
        entities["rigid_sphere"] = scene.add_entity(
            material=gs.materials.Rigid(rho=950.0, friction=0.42, coup_friction=0.88),
            morph=gs.morphs.Sphere(
                pos=(-0.58, -0.28, 0.90),
                radius=0.10,
            ),
            surface=rigid_surface,
        )
        return entities

    if case.motion_profile == "reverse_swipe":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(-0.18, 0.02, 0.20),
            size=(0.34, 0.22, 0.22),
            euler=(0.0, 0.0, -10.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.3e4,
            nu=0.24,
            rho=225.0,
            surface=primary_soft_surface,
        )
        entities["rigid_cylinder"] = scene.add_entity(
            material=gs.materials.Rigid(rho=980.0, friction=0.48, coup_friction=0.95),
            morph=gs.morphs.Cylinder(
                pos=(0.82, 0.18, 0.16),
                radius=0.09,
                height=0.22,
                euler=(90.0, 0.0, 0.0),
            ),
            surface=rigid_surface,
        )
        return entities

    if case.motion_profile == "ramp_slide":
        _add_fixed_box(
            scene,
            pos=(-0.10, 0.10, 0.18),
            size=(0.86, 0.46, 0.12),
            euler=(0.0, -18.0, 10.0),
            surface=support_surface,
            material=support_rigid_material,
            collision=True,
        )
        _add_fixed_box(
            scene,
            pos=(-0.44, 0.26, 0.09),
            size=(0.20, 0.28, 0.18),
            surface=support_surface,
            material=_rigid_material(rho=900.0, friction=0.52, coup_friction=0.86),
            collision=True,
        )
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(-0.22, 0.12, 0.42),
            size=(0.28, 0.22, 0.22),
            euler=(12.0, -10.0, 18.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.4e4,
            nu=0.24,
            rho=225.0,
            surface=primary_soft_surface,
        )
        return entities

    if case.motion_profile == "overhead_press":
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.10, 0.04, 0.28),
            size=(0.24, 0.24, 0.34),
            euler=(6.0, 0.0, 12.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.8e4,
            nu=0.25,
            rho=230.0,
            surface=primary_soft_surface,
        )
        entities["rigid_sphere"] = scene.add_entity(
            material=gs.materials.Rigid(rho=980.0, friction=0.42, coup_friction=0.90),
            morph=gs.morphs.Sphere(
                pos=(-0.10, -0.08, 1.08),
                radius=0.11,
            ),
            surface=rigid_surface,
        )
        return entities

    if case.motion_profile == "ledge_topple":
        _add_fixed_box(
            scene,
            pos=(0.28, 0.20, 0.16),
            size=(0.44, 0.28, 0.16),
            surface=support_surface,
            collision=True,
        )
        _add_fixed_box(
            scene,
            pos=(0.54, 0.20, 0.30),
            size=(0.10, 0.26, 0.10),
            surface=support_surface,
            collision=True,
        )
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.66, 0.20, 0.48),
            size=(0.22, 0.22, 0.28),
            euler=(28.0, 10.0, 38.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.6e4,
            nu=0.25,
            rho=228.0,
            surface=primary_soft_surface,
        )
        return entities

    if case.motion_profile == "wall_pinch":
        _add_fixed_box(
            scene,
            pos=(0.78, 0.14, 0.31),
            size=(0.08, 0.40, 0.62),
            surface=support_surface,
            collision=True,
        )
        entities["soft_block"] = _add_soft_box(
            scene,
            pos=(0.24, 0.10, 0.24),
            size=(0.30, 0.24, 0.24),
            euler=(0.0, 0.0, 8.0),
            color=p.soft_color,
            vis_mode=mpm_vis_mode,
            E=4.3e4,
            nu=0.24,
            rho=220.0,
            surface=primary_soft_surface,
        )
        entities["rigid_sphere"] = scene.add_entity(
            material=gs.materials.Rigid(rho=960.0, friction=0.42, coup_friction=0.90),
            morph=gs.morphs.Sphere(
                pos=(-0.70, 0.04, 0.30),
                radius=0.10,
            ),
            surface=rigid_surface,
        )
        return entities

    raise ValueError(f"unsupported motion_profile: {case.motion_profile}")


def _apply_case_initial_conditions(case: CaseSpec, entities: dict[str, Any]) -> None:
    if case.motion_profile == "sphere_impact":
        entities["rigid_sphere"].set_dofs_velocity((2.55, 0.02, -0.10, 0.0, 6.0, 0.0))
        return
    if case.motion_profile == "cylinder_swipe":
        entities["rigid_cylinder"].set_dofs_velocity((2.08, 0.18, 0.0, 6.2, 0.0, 0.0))
        return
    if case.motion_profile == "dual_sphere_press":
        entities["rigid_sphere_left"].set_dofs_velocity((2.45, 0.22, 0.04, 0.0, 6.8, 0.0))
        entities["rigid_sphere_right"].set_dofs_velocity((-2.05, -0.36, -0.12, 0.0, -5.5, 0.0))
        return
    if case.motion_profile == "glancing_arc":
        entities["rigid_sphere"].set_dofs_velocity((1.95, 0.84, -1.65, 0.0, 5.0, 2.0))
        return
    if case.motion_profile == "reverse_swipe":
        entities["rigid_cylinder"].set_dofs_velocity((-2.10, -0.34, 0.0, -8.5, 0.0, 0.0))
        return
    if case.motion_profile == "overhead_press":
        entities["rigid_sphere"].set_dofs_velocity((0.42, 0.18, -2.35, 2.4, 0.0, 1.0))
        return
    if case.motion_profile == "wall_pinch":
        entities["rigid_sphere"].set_dofs_velocity((2.65, 0.10, -0.06, 0.0, 5.4, 0.0))
        return


def _collect_entity_states(entities: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, entity in entities.items():
        if hasattr(entity, "get_particles_pos"):
            state[name] = _soft_block_state(entity)
        else:
            state[name] = _rigid_body_state(entity)
    return state


def render_case(
    *,
    case_key: str,
    output_root: Path,
    run_name: str,
    fps: int,
    playback_speed: float,
    width: int,
    height: int,
    mpm_vis_mode_override: str,
    save_every_frame: bool,
) -> dict[str, Any]:
    _init_genesis()
    case = CASE_LIBRARY[case_key]
    camera = _camera_spec(case, width=width, height=height)
    mpm_vis_mode = mpm_vis_mode_override or case.mpm_vis_mode
    run_label = run_name or case_key
    case_dir = output_root / run_label
    dirs = _build_output_dirs(case_dir)

    scene, cam = _scene_common(case, camera, mpm_vis_mode)
    entities = _build_case_entities(scene, case, mpm_vis_mode)
    scene.build()
    if mpm_vis_mode == "visual":
        _apply_soft_visual_markers(case, entities)
    _apply_case_initial_conditions(case, entities)

    preview_frames: list[np.ndarray] = []
    saved_keyframes: list[str] = []
    record_steps = _record_frame_indices(case.sim.horizon)

    initial_state = _collect_entity_states(entities)

    for step_idx in range(case.sim.horizon):
        scene.step()
        rgb = _render_rgb(cam)
        if save_every_frame:
            frame_path = dirs["rgb"] / f"{step_idx:06d}.png"
            imageio.imwrite(frame_path, rgb)
        elif step_idx in record_steps:
            frame_path = dirs["rgb"] / f"{step_idx:06d}.png"
            imageio.imwrite(frame_path, rgb)
            saved_keyframes.append(frame_path.name)
        preview_frames.append(rgb)

    video_frames = _resample_video_frames_to_fps(
        preview_frames,
        sim_dt=case.sim.dt,
        target_fps=fps,
        playback_speed=playback_speed,
    )
    video_path = dirs["video"] / "preview.mp4"
    imageio.mimwrite(video_path, video_frames, fps=fps, quality=8)

    final_state = _collect_entity_states(entities)
    sim_duration_s = float(case.sim.dt) * float(case.sim.horizon)

    manifest = {
        "case_key": case.key,
        "family": case.family,
        "title": case.title,
        "description": case.description,
        "scene_theme": case.scene_theme,
        "motion_profile": case.motion_profile,
        "output_root": str(case_dir),
        "video": str(video_path),
        "keyframes": saved_keyframes,
        "fps": int(fps),
        "playback_speed": float(playback_speed),
        "num_frames": int(len(video_frames)),
        "raw_sim_frames": int(len(preview_frames)),
        "sim_duration_s": sim_duration_s,
        "video_duration_s": float(len(video_frames)) / float(fps),
        "mpm_vis_mode": mpm_vis_mode,
        "camera": asdict(camera),
        "sim": asdict(case.sim),
        "initial_state": initial_state,
        "final_state": final_state,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "genesis_backend": "gpu",
        },
        "notes": [
            "This preview adds a more structured indoor scene using fixed rigid decor to improve realism without interfering with the main MPM interaction area.",
            "Genesis in the current environment warns about torch 2.7.x being outside its supported range; preview success does not fully de-risk larger runs.",
        ],
    }
    manifest_path = dirs["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(_to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = render_case(
        case_key=args.case_key,
        output_root=args.output_root,
        run_name=args.run_name,
        fps=args.fps,
        playback_speed=args.playback_speed,
        width=args.width,
        height=args.height,
        mpm_vis_mode_override=args.mpm_vis_mode,
        save_every_frame=args.save_every_frame,
    )
    print(json.dumps(_to_jsonable(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
