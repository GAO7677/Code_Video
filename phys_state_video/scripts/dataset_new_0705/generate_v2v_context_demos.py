"""Generate short-context physical demos for video continuation experiments.

Each control group keeps the object, materials, camera, and simulation settings
fixed while changing one visible geometric or initial-state variable.  The
first eight 30-fps frames are exported separately so the continuation boundary
can be reviewed alongside the complete simulated trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError("PyBullet is required for V2V context demos") from exc

from .common_specs import CameraSpec, ObjectInstanceSpec, ScenarioBlueprint
from .material_catalog_0705 import build_material_catalog, build_surface_catalog
from .render_sim_0705 import (
    RealismPreviewRenderer,
    blueprint_to_legacy_scenario,
    override_legacy_runtime,
    register_material_assets,
)
from .scene_generators_0705 import EARTH_GRAVITY, _collision_vertical_extent

try:
    from .. import generate_sim_preview_gallery as legacy
except ImportError:  # pragma: no cover - direct script fallback
    import generate_sim_preview_gallery as legacy


FPS = 30
SIM_HZ = 240
SIM_DURATION_S = 3.0
CONTEXT_FRAMES = 8
CONTEXT_FRAME_OPTIONS = (8, 16)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
SCENE_STYLE = "indoor_natural"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_context_demos_20260819"
)

V2V_QUESTION = (
    "Use the selected context segment as the visual context for a video continuation task. "
    "Identify the visible geometry, object state, contact or support relation, "
    "and the single physical condition that constrains what happens next. "
    "Separate observations in the context from later simulated events, and "
    "focus on motion, timing, direction, and interaction rather than appearance."
)


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    family_key: str
    family_title: str
    family_description: str
    level: str
    title: str
    description: str
    controlled_variable: str
    controlled_value: float
    controlled_value_label: str
    units: str
    blueprint: ScenarioBlueprint
    event_rule: str
    setup_constraints: Callable[[dict[str, int]], None] | None = None


def _camera(
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    yfov_deg: float = 50.0,
    hdri_key: str = "hall_bright",
) -> CameraSpec:
    return CameraSpec(
        eye=eye,
        target=target,
        up=(0.0, 0.0, 1.0),
        yfov_deg=yfov_deg,
        jitter_eye_xyz=(0.0, 0.0, 0.0),
        jitter_target_xyz=(0.0, 0.0, 0.0),
        jitter_fov_deg=0.0,
        hdri_key=hdri_key,
    )


def _object(
    *,
    name: str,
    family_key: str,
    shape: str,
    size: dict[str, float],
    material_key: str,
    position: tuple[float, float, float],
    dynamic: bool,
    mass: float,
    friction: float,
    restitution: float,
    role: str = "dynamic",
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    linear_damping: float = 0.02,
    angular_damping: float = 0.04,
    metadata: dict[str, object] | None = None,
) -> ObjectInstanceSpec:
    materials = build_material_catalog()
    if not dynamic:
        mass = 0.0
        role = role if role.startswith("anchored_") else "anchored_fixture"
    return ObjectInstanceSpec(
        name=name,
        family_key=family_key,
        shape=shape,
        semantic_role=role,
        size=size,
        mass=float(mass),
        friction=float(friction),
        restitution=float(restitution),
        linear_damping=float(linear_damping),
        angular_damping=float(angular_damping),
        material_key=material_key,
        color=materials[material_key].base_color,
        dynamic=dynamic,
        role=role,
        position=position,
        orientation_euler_deg=orientation,
        linear_velocity=velocity,
        angular_velocity=angular_velocity,
        metadata=dict(metadata or {}),
    )


def _blueprint(
    *,
    family_key: str,
    sample_key: str,
    title: str,
    description: str,
    objects: Iterable[ObjectInstanceSpec],
    camera: CameraSpec,
    surface_key: str,
    tags: tuple[str, ...],
    metadata: dict[str, object],
) -> ScenarioBlueprint:
    blueprint = ScenarioBlueprint(
        family_key=family_key,
        sample_key=sample_key,
        title=title,
        description=description,
        gravity=EARTH_GRAVITY,
        pre_roll_s=0.04,
        camera_key=f"v2v_{family_key.lower()}",
        surface_key=surface_key,
        lighting_key=camera.hdri_key,
        camera=camera,
        objects=tuple(objects),
        tags=tags,
        metadata={
            **metadata,
            "fps": FPS,
            "sim_hz": SIM_HZ,
            "duration_s": SIM_DURATION_S,
            "context_frames": CONTEXT_FRAMES,
            "context_duration_s": CONTEXT_FRAMES / FPS,
            "context_frame_options": list(CONTEXT_FRAME_OPTIONS),
            "scene_style": SCENE_STYLE,
        },
    )
    _validate_blueprint(blueprint)
    return blueprint


def _validate_blueprint(blueprint: ScenarioBlueprint) -> None:
    if not math.isclose(blueprint.gravity, EARTH_GRAVITY, abs_tol=1e-9):
        raise ValueError(f"{blueprint.sample_key}: non-Earth gravity")
    names = [obj.name for obj in blueprint.objects]
    if len(names) != len(set(names)):
        raise ValueError(f"{blueprint.sample_key}: duplicate object names")
    for obj in blueprint.objects:
        if obj.dynamic:
            if obj.mass <= 0.0:
                raise ValueError(f"{blueprint.sample_key}/{obj.name}: dynamic mass must be positive")
        elif obj.mass != 0.0:
            raise ValueError(f"{blueprint.sample_key}/{obj.name}: static mass must be zero")
        if obj.role.startswith("anchored_"):
            continue
        if not obj.dynamic:
            expected = _collision_vertical_extent(obj)
            if not math.isclose(obj.position[2], expected, abs_tol=1e-6):
                raise ValueError(
                    f"{blueprint.sample_key}/{obj.name}: static object is not grounded"
                )


def _make_gap_case(sample_key: str, gap_width: float) -> DemoCase:
    left_hx = 0.72
    right_hx = 0.72
    left_center = -0.78
    left_edge = left_center + left_hx
    right_center = left_edge + gap_width + right_hx
    platform_top = 0.48
    platform_hz = 0.07
    ball_radius = 0.11
    objects = [
        _object(
            name="gap_ball",
            family_key="ball",
            shape="sphere",
            size={"radius": ball_radius},
            material_key="rubber_red",
            position=(-0.56, 0.0, platform_top + ball_radius),
            dynamic=True,
            mass=1.2,
            friction=0.26,
            restitution=0.46,
            velocity=(1.85, 0.0, 0.0),
            angular_velocity=(0.0, -1.85 / ball_radius, 0.0),
            linear_damping=0.01,
            angular_damping=0.01,
            metadata={"appearance_group": "v2v_gap_red_rubber_ball_v1"},
        ),
        _object(
            name="left_platform",
            family_key="platform_block",
            shape="box",
            size={"hx": left_hx, "hy": 0.52, "hz": platform_hz},
            material_key="wood_dark",
            position=(left_center, 0.0, platform_top - platform_hz),
            dynamic=False,
            mass=0.0,
            friction=0.82,
            restitution=0.02,
        ),
        _object(
            name="right_platform",
            family_key="platform_block",
            shape="box",
            size={"hx": right_hx, "hy": 0.52, "hz": platform_hz},
            material_key="wood_plywood",
            position=(right_center, 0.0, platform_top - platform_hz),
            dynamic=False,
            mass=0.0,
            friction=0.82,
            restitution=0.02,
        ),
        _object(
            name="left_platform_support",
            family_key="table_leg",
            shape="box",
            size={"hx": 0.24, "hy": 0.28, "hz": (platform_top - 2.0 * platform_hz) * 0.5},
            material_key="concrete_painted",
            position=(left_center, 0.0, (platform_top - 2.0 * platform_hz) * 0.5),
            dynamic=False,
            mass=0.0,
            friction=0.84,
            restitution=0.02,
        ),
        _object(
            name="right_platform_support",
            family_key="table_leg",
            shape="box",
            size={"hx": 0.24, "hy": 0.28, "hz": (platform_top - 2.0 * platform_hz) * 0.5},
            material_key="concrete_painted",
            position=(right_center, 0.0, (platform_top - 2.0 * platform_hz) * 0.5),
            dynamic=False,
            mass=0.0,
            friction=0.84,
            restitution=0.02,
        ),
    ]
    camera = _camera(
        eye=(0.15, -4.85, 1.30),
        target=(0.30, 0.0, 0.62),
        yfov_deg=50.0,
    )
    blueprint = _blueprint(
        family_key="V2V_GAP",
        sample_key=sample_key,
        title=f"Ball crossing a {gap_width:.2f} m platform gap",
        description="A rolling ball approaches a visible gap between two platforms; only the gap width changes.",
        objects=objects,
        camera=camera,
        surface_key="residential_wood_floor",
        tags=("v2v", "platform_gap", "gravity", "left_to_right"),
        metadata={
            "controlled_variable": "gap_width_m",
            "gap_width_m": gap_width,
            "left_platform_edge_x_m": left_edge,
            "platform_height_m": platform_top,
            "platform_top_half_thickness_m": platform_hz,
            "initial_speed_mps": 1.85,
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_GAP",
        family_title="平台间隙",
        family_description="球在左侧平台上接近可见间隙，间隙宽度决定后续是否落到右侧平台或掉到地面。",
        level="L2",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="gap_width_m",
        controlled_value=gap_width,
        controlled_value_label=f"{gap_width:.2f} m",
        units="m",
        blueprint=blueprint,
        event_rule="ball_crosses_left_platform_edge",
    )


def _make_obstacle_case(sample_key: str, obstacle_x: float) -> DemoCase:
    ball_radius = 0.11
    barrier_hz = 0.24
    objects = [
        _object(
            name="obstacle_ball",
            family_key="ball",
            shape="sphere",
            size={"radius": ball_radius},
            material_key="rubber_red",
            position=(-0.66, 0.0, ball_radius + 0.004),
            dynamic=True,
            mass=1.0,
            friction=0.22,
            restitution=0.40,
            velocity=(2.15, 0.0, 0.0),
            angular_velocity=(0.0, -2.15 / ball_radius, 0.0),
            linear_damping=0.01,
            angular_damping=0.01,
            metadata={"appearance_group": "v2v_obstacle_red_rubber_ball_v1"},
        ),
        _object(
            name="obstacle_barrier",
            family_key="barrier_box",
            shape="box",
            size={"hx": 0.075, "hy": 0.32, "hz": barrier_hz},
            material_key="painted_metal_teal",
            position=(obstacle_x, 0.0, barrier_hz),
            dynamic=False,
            mass=0.0,
            friction=0.75,
            restitution=0.16,
        ),
    ]
    camera = _camera(
        eye=(0.05, -4.55, 1.12),
        target=(0.05, 0.0, 0.48),
        yfov_deg=48.0,
    )
    blueprint = _blueprint(
        family_key="V2V_OBSTACLE",
        sample_key=sample_key,
        title=f"Ball and obstacle at x={obstacle_x:.2f} m",
        description="A rolling ball approaches a visible barrier; only the barrier position changes.",
        objects=objects,
        camera=camera,
        surface_key="studio_wood_floor",
        tags=("v2v", "obstacle_distance", "collision", "left_to_right"),
        metadata={
            "controlled_variable": "obstacle_x_m",
            "obstacle_x_m": obstacle_x,
            "initial_speed_mps": 2.15,
            "barrier_half_x_m": 0.075,
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_OBSTACLE",
        family_title="障碍物位置",
        family_description="球朝可见障碍物滚动，障碍物距离决定碰撞发生的时间和碰撞后的轨迹。",
        level="L2",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="obstacle_x_m",
        controlled_value=obstacle_x,
        controlled_value_label=f"x={obstacle_x:.2f} m",
        units="m",
        blueprint=blueprint,
        event_rule="ball_contacts_barrier",
    )


def _bowl_segment(
    index: int,
    radius: float,
    x: float,
    span: float,
    bottom_z: float,
    segment_dx: float,
) -> ObjectInstanceSpec:
    root = max(radius * radius - x * x, 1e-8)
    slope = x / math.sqrt(root)
    theta = math.atan(slope)
    surface_z = bottom_z + radius - math.sqrt(root)
    thickness = 0.065
    normal = (-math.sin(theta), 0.0, math.cos(theta))
    center = (
        # The curve describes the interior (upper) contact face.  Put the
        # solid slab below that face rather than growing it into the bowl.
        x - normal[0] * thickness * 0.5,
        0.0,
        surface_z - normal[2] * thickness * 0.5,
    )
    return _object(
        name=f"bowl_segment_{index:02d}",
        family_key="slab_box",
        shape="box",
        # Leave a physical gap between adjacent tangent slabs.  The previous
        # 0.56 multiplier made neighbouring collision boxes overlap.
        size={"hx": segment_dx * 0.40, "hy": 0.48, "hz": thickness * 0.5},
        material_key="wood_plywood",
        position=center,
        dynamic=False,
        mass=0.0,
        friction=0.72,
        restitution=0.03,
        orientation=(0.0, -math.degrees(theta), 0.0),
    )


def _make_bowl_case(sample_key: str, radius: float) -> DemoCase:
    span = 0.86
    bottom_z = 0.16
    xs = np.linspace(-span, span, 15)
    segment_dx = float(xs[1] - xs[0])
    ball_radius = 0.115
    ball_surface_x = float(xs[2])
    root = math.sqrt(radius * radius - ball_surface_x * ball_surface_x)
    surface_slope = ball_surface_x / root
    surface_theta = math.atan(surface_slope)
    surface_normal = (-math.sin(surface_theta), 0.0, math.cos(surface_theta))
    ball_surface_z = bottom_z + radius - root
    ball_clearance = 0.012
    ball_position = (
        ball_surface_x + surface_normal[0] * (ball_radius + ball_clearance),
        0.0,
        ball_surface_z + surface_normal[2] * (ball_radius + ball_clearance),
    )
    objects: list[ObjectInstanceSpec] = [
        _object(
            name="bowl_base",
            family_key="platform_block",
            shape="box",
            size={"hx": 1.16, "hy": 0.56, "hz": 0.045},
            material_key="wood_dark",
            position=(0.0, 0.0, 0.045),
            dynamic=False,
            mass=0.0,
            friction=0.82,
            restitution=0.02,
        ),
        _object(
            name="bowl_ball",
            family_key="ball",
            shape="sphere",
            size={"radius": ball_radius},
            material_key="rubber_blue",
            position=ball_position,
            dynamic=True,
            mass=1.1,
            friction=0.28,
            restitution=0.22,
            velocity=(0.42, 0.0, 0.0),
            angular_velocity=(0.0, -0.42 / ball_radius, 0.0),
            linear_damping=0.01,
            angular_damping=0.01,
            metadata={"appearance_group": "v2v_bowl_blue_rubber_ball_v1"},
        ),
    ]
    objects.extend(
        _bowl_segment(index, radius, float(x), span, bottom_z, segment_dx)
        for index, x in enumerate(xs)
    )
    camera = _camera(
        eye=(0.05, -4.55, 1.30),
        target=(0.05, 0.0, 0.68),
        yfov_deg=48.0,
        hdri_key="studio_warm",
    )
    blueprint = _blueprint(
        family_key="V2V_BOWL",
        sample_key=sample_key,
        title=f"Ball in a bowl with radius {radius:.2f} m",
        description="A blue rubber ball enters a visible curved container; only the bowl curvature changes.",
        objects=objects,
        camera=camera,
        surface_key="residential_wood_floor",
        tags=("v2v", "curved_container", "gravity", "rolling"),
        metadata={
            "controlled_variable": "bowl_radius_m",
            "bowl_radius_m": radius,
            "bowl_span_m": span,
            "bowl_bottom_z_m": bottom_z,
            "initial_speed_mps": 0.42,
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_BOWL",
        family_title="容器曲率",
        family_description="球进入可见的弧形容器，曲率决定后续爬升、反向、振荡和趋于稳定的过程。",
        level="L2",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="bowl_radius_m",
        controlled_value=radius,
        controlled_value_label=f"R={radius:.2f} m",
        units="m",
        blueprint=blueprint,
        event_rule="ball_passes_bowl_center_and_reverses",
    )


def _make_pendulum_case(sample_key: str, length: float) -> DemoCase:
    anchor_x = -0.82
    anchor_z = 1.70
    angle_deg = 18.0
    angle = math.radians(angle_deg)
    bob_radius = 0.13
    bob_x = anchor_x + length * math.sin(angle)
    bob_z = anchor_z - length * math.cos(angle)
    rope_clearance = 0.022
    rope_length = max(0.04, length - bob_radius - rope_clearance)
    rope_end_x = anchor_x + rope_length * math.sin(angle)
    rope_end_z = anchor_z - rope_length * math.cos(angle)
    rope_center = ((anchor_x + rope_end_x) * 0.5, 0.0, (anchor_z + rope_end_z) * 0.5)
    rope_vec = (rope_end_x - anchor_x, 0.0, rope_end_z - anchor_z)
    post_half_height = 0.76
    post_center_z = 0.18 + post_half_height
    objects = [
        _object(
            name="pendulum_base",
            family_key="platform_block",
            shape="box",
            size={"hx": 0.55, "hy": 0.40, "hz": 0.09},
            material_key="wood_dark",
            position=(anchor_x, 0.0, 0.09),
            dynamic=False,
            mass=0.0,
            friction=0.82,
            restitution=0.02,
        ),
        _object(
            name="pendulum_post",
            family_key="table_leg",
            shape="box",
            size={"hx": 0.065, "hy": 0.05, "hz": post_half_height},
            material_key="painted_metal_teal",
            position=(anchor_x, 0.26, post_center_z),
            dynamic=False,
            mass=0.0,
            friction=0.80,
            restitution=0.02,
        ),
        _object(
            name="pendulum_crossbar",
            family_key="platform_block",
            shape="box",
            size={"hx": 0.14, "hy": 0.15, "hz": 0.03},
            material_key="painted_metal_teal",
            position=(anchor_x, 0.11, 1.77),
            dynamic=False,
            mass=0.0,
            friction=0.80,
            restitution=0.02,
        ),
        _object(
            name="pendulum_rope",
            family_key="table_leg",
            shape="cylinder",
            size={"radius": 0.018, "height": rope_length},
            material_key="painted_metal_yellow",
            position=rope_center,
            dynamic=False,
            mass=0.0,
            friction=0.40,
            restitution=0.0,
            role="anchored_visual",
            orientation=_quat_vector_as_euler(rope_vec),
            metadata={
                "visual_only": True,
                "visual_anchor": (anchor_x, 0.0, anchor_z),
                "visual_target": "pendulum_bob",
                "visual_target_surface_offset_m": bob_radius + rope_clearance,
            },
        ),
        _object(
            name="pendulum_bob",
            family_key="ball",
            shape="sphere",
            size={"radius": bob_radius},
            material_key="rubber_red",
            position=(bob_x, 0.0, bob_z),
            dynamic=True,
            mass=1.2,
            friction=0.42,
            restitution=0.18,
            orientation=(0.0, -angle_deg, 0.0),
            linear_damping=0.015,
            angular_damping=0.02,
            metadata={"appearance_group": "v2v_pendulum_red_rubber_bob_v1"},
        ),
    ]
    camera = _camera(
        eye=(0.38, -4.55, 1.36),
        target=(-0.40, 0.0, 0.96),
        yfov_deg=48.0,
        hdri_key="hall_bright",
    )
    blueprint = _blueprint(
        family_key="V2V_PENDULUM",
        sample_key=sample_key,
        title=f"Pendulum with length {length:.2f} m",
        description="A suspended bob begins at a visible angle; only the pendulum length changes.",
        objects=objects,
        camera=camera,
        surface_key="painted_concrete_floor",
        tags=("v2v", "pendulum", "gravity", "constrained_motion"),
        metadata={
            "controlled_variable": "pendulum_length_m",
            "pendulum_length_m": length,
            "initial_angle_deg": angle_deg,
            "anchor": (anchor_x, 0.0, anchor_z),
            "constraints": [
                {
                    "type": "point2point",
                    "body": "pendulum_bob",
                    "parent_body": "pendulum_post",
                    "parent_frame": (0.0, -0.26, anchor_z - post_center_z),
                    "child_frame": (
                        -length * math.sin(angle),
                        0.0,
                        length * math.cos(angle),
                    ),
                }
            ],
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_PENDULUM",
        family_title="摆长",
        family_description="画面显示悬挂点、初始偏角和摆长，摆长决定后续摆动周期和运动范围。",
        level="L3",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="pendulum_length_m",
        controlled_value=length,
        controlled_value_label=f"{length:.2f} m",
        units="m",
        blueprint=blueprint,
        event_rule="bob_crosses_anchor_vertical",
    )


def _make_seesaw_case(sample_key: str, load_x: float) -> DemoCase:
    # Keep a small visible clearance between the fulcrum and the board at
    # initialization; the hinge constraint supplies the shared pivot point.
    pivot_height = 0.44
    pivot_center_z = 0.5 * pivot_height
    board_center_z = 0.50
    board_hx = 1.02
    board_hy = 0.26
    board_hz = 0.045
    board_angle = 0.0
    theta = math.radians(board_angle)
    load_hx, load_hy, load_hz = 0.14, 0.14, 0.12
    local_z = board_hz + load_hz + 0.012
    load_position = (
        load_x * math.cos(theta) + local_z * math.sin(theta),
        0.0,
        board_center_z - load_x * math.sin(theta) + local_z * math.cos(theta),
    )
    objects = [
        _object(
            name="seesaw_pivot",
            family_key="platform_block",
            shape="box",
            size={"hx": 0.12, "hy": 0.30, "hz": pivot_center_z},
            material_key="concrete_painted",
            position=(0.0, 0.0, pivot_center_z),
            dynamic=False,
            mass=0.0,
            friction=0.90,
            restitution=0.02,
        ),
        _object(
            name="seesaw_board",
            family_key="slab_box",
            shape="box",
            size={"hx": board_hx, "hy": board_hy, "hz": board_hz},
            material_key="wood_plywood",
            position=(0.0, 0.0, board_center_z),
            dynamic=True,
            mass=8.0,
            friction=0.78,
            restitution=0.04,
            role="dynamic_hinged_board",
            orientation=(0.0, board_angle, 0.0),
            linear_damping=0.04,
            angular_damping=0.08,
        ),
        _object(
            name="seesaw_load",
            family_key="wood_block",
            shape="box",
            size={"hx": load_hx, "hy": load_hy, "hz": load_hz},
            material_key="rubber_blue",
            position=load_position,
            dynamic=True,
            mass=1.4,
            friction=0.70,
            restitution=0.10,
            orientation=(0.0, board_angle, 0.0),
            linear_damping=0.03,
            angular_damping=0.05,
        ),
    ]
    camera = _camera(
        eye=(0.0, -4.75, 1.30),
        target=(0.0, 0.0, 0.62),
        yfov_deg=48.0,
        hdri_key="studio_warm",
    )
    blueprint = _blueprint(
        family_key="V2V_SEESAW",
        sample_key=sample_key,
        title=f"Seesaw load at x={load_x:.2f} m",
        description="A block rests on a hinged board; only its distance from the pivot changes.",
        objects=objects,
        camera=camera,
        surface_key="residential_wood_floor",
        tags=("v2v", "seesaw", "hinge", "gravity"),
        metadata={
            "controlled_variable": "load_position_x_m",
            "load_position_x_m": load_x,
            "pivot_z_m": pivot_height,
            "initial_board_angle_deg": board_angle,
            "constraints": [
                {
                    "type": "point2point",
                    "body": "seesaw_board",
                    "parent_body": "seesaw_pivot",
                    "parent_frame": (0.0, -0.20, pivot_center_z),
                    "child_frame": (0.0, -0.20, pivot_height - board_center_z),
                },
                {
                    "type": "point2point",
                    "body": "seesaw_board",
                    "parent_body": "seesaw_pivot",
                    "parent_frame": (0.0, 0.20, pivot_center_z),
                    "child_frame": (0.0, 0.20, pivot_height - board_center_z),
                },
            ],
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_SEESAW",
        family_title="跷跷板载荷位置",
        family_description="画面显示支点、板面和载荷位置，载荷距支点的距离决定后续转动方向和幅度。",
        level="L3",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="load_position_x_m",
        controlled_value=load_x,
        controlled_value_label=f"x={load_x:.2f} m",
        units="m",
        blueprint=blueprint,
        event_rule="hinged_board_rotates_after_load_settles",
    )


def _make_domino_case(sample_key: str, spacing: float) -> DemoCase:
    hx, hy, hz = 0.045, 0.12, 0.24
    trigger_ball_radius = 0.085
    trigger_ball_x = -1.45
    trigger_ball_speed = 1.80
    start_x = -0.55
    center_step = 2.0 * hx + spacing
    objects: list[ObjectInstanceSpec] = [
        _object(
            name="domino_trigger_ball",
            family_key="trigger_ball",
            shape="sphere",
            size={"radius": trigger_ball_radius},
            material_key="rubber_red",
            position=(trigger_ball_x, 0.0, trigger_ball_radius),
            dynamic=True,
            mass=1.20,
            friction=0.30,
            restitution=0.45,
            velocity=(trigger_ball_speed, 0.0, 0.0),
            linear_damping=0.01,
            angular_damping=0.01,
            role="dynamic_trigger",
            metadata={"appearance_group": "v2v_domino_red_rubber_trigger_v1"},
        ),
        _object(
            name="domino_0",
            family_key="tall_box",
            shape="box",
            size={"hx": hx, "hy": hy, "hz": hz},
            material_key="rubber_red",
            position=(start_x, 0.0, hz),
            dynamic=True,
            mass=0.42,
            friction=0.72,
            restitution=0.08,
            linear_damping=0.015,
            angular_damping=0.035,
        )
    ]
    for index in range(1, 5):
        objects.append(
            _object(
                name=f"domino_{index}",
                family_key="tall_box",
                shape="box",
                size={"hx": hx, "hy": hy, "hz": hz},
                material_key="painted_metal_teal",
                position=(start_x + index * center_step, 0.0, hz),
                dynamic=True,
                mass=0.42,
                friction=0.72,
                restitution=0.08,
                linear_damping=0.015,
                angular_damping=0.035,
            )
        )
    camera = _camera(
        eye=(0.0, -4.45, 1.02),
        target=(-0.25, 0.0, 0.48),
        yfov_deg=46.0,
        hdri_key="hall_bright",
    )
    blueprint = _blueprint(
        family_key="V2V_DOMINO",
        sample_key=sample_key,
        title=f"Domino chain with {spacing:.2f} m spacing",
        description="A moving red ball strikes the first upright domino; only the spacing changes.",
        objects=objects,
        camera=camera,
        surface_key="painted_concrete_floor",
        tags=("v2v", "domino_chain", "contact_transfer", "left_to_right"),
        metadata={
            "controlled_variable": "domino_gap_m",
            "domino_gap_m": spacing,
            "domino_center_step_m": center_step,
            "trigger_ball_start_x_m": trigger_ball_x,
            "trigger_ball_speed_mps": trigger_ball_speed,
            "trigger_ball_radius_m": trigger_ball_radius,
            "trigger_ball_mass_kg": 1.20,
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_DOMINO",
        family_title="多米诺间距",
        family_description="前 8 帧显示第一块的倾斜、排列方向和物体间距，间距决定后续碰撞能否连续传播。",
        level="L3",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="domino_gap_m",
        controlled_value=spacing,
        controlled_value_label=f"{spacing:.2f} m",
        units="m",
        blueprint=blueprint,
        event_rule="second_domino_starts_to_fall",
    )


def _quat_vector_as_euler(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return a Y-axis pitch that points a cylinder's local Z axis at vector."""
    x, _, z = vector
    return (0.0, math.degrees(math.atan2(x, z)), 0.0)


def build_demo_cases(seed_base: int = 20260819) -> list[DemoCase]:
    del seed_base  # Geometry is deterministic; seeds are assigned by the runner.
    cases: list[DemoCase] = []
    for value in (0.12, 0.30, 0.48):
        cases.append(_make_gap_case(f"v2v_gap_{int(value * 100):03d}", value))
    for value in (0.20, 0.55, 0.90):
        cases.append(_make_obstacle_case(f"v2v_obstacle_{int(value * 100):03d}", value))
    for value in (1.10, 1.50, 2.00):
        cases.append(_make_bowl_case(f"v2v_bowl_r{int(value * 100):03d}", value))
    for value in (0.70, 1.00, 1.30):
        cases.append(_make_pendulum_case(f"v2v_pendulum_l{int(value * 100):03d}", value))
    for value in (0.28, 0.48, 0.68):
        cases.append(_make_seesaw_case(f"v2v_seesaw_x{int(value * 100):03d}", value))
    for value in (0.02, 0.06, 0.12):
        cases.append(_make_domino_case(f"v2v_domino_g{int(value * 100):03d}", value))
    return cases


def _object_payload(obj: ObjectInstanceSpec) -> dict[str, object]:
    return {
        "name": obj.name,
        "family_key": obj.family_key,
        "shape": obj.shape,
        "dynamic": bool(obj.dynamic),
        "role": obj.role,
        "mass_kg": round(float(obj.mass), 5),
        "friction": round(float(obj.friction), 5),
        "restitution": round(float(obj.restitution), 5),
        "position": [round(float(value), 5) for value in obj.position],
        "orientation_euler_deg": [round(float(value), 5) for value in obj.orientation_euler_deg],
        "linear_velocity": [round(float(value), 5) for value in obj.linear_velocity],
        "angular_velocity": [round(float(value), 5) for value in obj.angular_velocity],
        "size": {key: round(float(value), 5) for key, value in obj.size.items()},
        "material_key": obj.material_key,
        "metadata": obj.metadata,
    }


def _json_safe(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _state_summary(
    positions: np.ndarray,
    linear_velocities: np.ndarray,
    angular_velocities: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    objects: list[dict[str, object]] = []
    max_linear_speed = 0.0
    any_motion = False
    for index, name in enumerate(names):
        pos = positions[:, index]
        linear = linear_velocities[:, index]
        angular = angular_velocities[:, index]
        speed = np.linalg.norm(linear, axis=1)
        angular_speed = np.linalg.norm(angular, axis=1)
        path_length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
        max_speed = float(speed.max(initial=0.0))
        final_speed = float(speed[-1])
        max_angular = float(angular_speed.max(initial=0.0))
        final_angular = float(angular_speed[-1])
        height_range = float(pos[:, 2].max() - pos[:, 2].min())
        moving = max(max_speed, max_angular * 0.05, height_range) > 0.05
        any_motion = any_motion or moving
        max_linear_speed = max(max_linear_speed, max_speed)
        objects.append(
            {
                "name": name,
                "path_length_m": round(path_length, 5),
                "net_displacement_m": round(float(np.linalg.norm(pos[-1] - pos[0])), 5),
                "max_speed_mps": round(max_speed, 5),
                "final_speed_mps": round(final_speed, 5),
                "max_angular_speed_radps": round(max_angular, 5),
                "final_angular_speed_radps": round(final_angular, 5),
                "height_range_m": round(height_range, 5),
                "derived_motion_present": bool(moving),
            }
        )
    final_motion = "moving" if any(
        item["final_speed_mps"] > 0.08 or item["final_angular_speed_radps"] > 0.35
        for item in objects
    ) else "nearly_stationary"
    return {
        "frames": int(len(positions)),
        "duration_s": round(max(0, len(positions) - 1) / FPS, 4),
        "objects": objects,
        "max_linear_speed_mps": round(max_linear_speed, 5),
        "final_motion_state": final_motion,
        "derived_motion_present": bool(any_motion),
        "derivation": "state_trajectory_summary_only; not a semantic event label",
    }


def _quat_to_y_angle(quat: np.ndarray) -> float:
    return float(legacy.p.getEulerFromQuaternion([float(value) for value in quat])[1])


def _first_event_frame(
    case: DemoCase,
    positions: np.ndarray,
    quats: np.ndarray,
) -> int | None:
    names = [obj.name for obj in case.blueprint.objects]
    index = {name: idx for idx, name in enumerate(names)}
    metadata = case.blueprint.metadata
    if case.family_key == "V2V_GAP":
        threshold = float(metadata["left_platform_edge_x_m"]) + 0.02
        frames = np.flatnonzero(positions[:, index["gap_ball"], 0] > threshold)
    elif case.family_key == "V2V_OBSTACLE":
        ball = positions[:, index["obstacle_ball"], 0]
        barrier = float(metadata["obstacle_x_m"])
        contact_distance = float(metadata.get("barrier_half_x_m", 0.075)) + 0.115 + 0.015
        frames = np.flatnonzero(np.abs(ball - barrier) <= contact_distance)
    elif case.family_key == "V2V_BOWL":
        ball_x = positions[:, index["bowl_ball"], 0]
        frames = np.flatnonzero(ball_x >= -0.04)
    elif case.family_key == "V2V_PENDULUM":
        anchor_x = float(metadata["anchor"][0])
        bob_x = positions[:, index["pendulum_bob"], 0]
        frames = np.flatnonzero(bob_x <= anchor_x)
    elif case.family_key == "V2V_SEESAW":
        board_idx = index["seesaw_board"]
        angles = np.asarray([_quat_to_y_angle(quat) for quat in quats[:, board_idx]])
        frames = np.flatnonzero(np.abs(angles) > math.radians(10.0))
    elif case.family_key == "V2V_DOMINO":
        domino_idx = index["domino_1"]
        angles = np.asarray([abs(_quat_to_y_angle(quat)) for quat in quats[:, domino_idx]])
        frames = np.flatnonzero(angles > math.radians(3.0))
    else:  # pragma: no cover - defensive guard
        return None
    return int(frames[0]) if len(frames) else None


def _visual_pose(
    obj: ObjectInstanceSpec,
    positions: dict[str, np.ndarray],
) -> tuple[tuple[float, float, float], list[float]]:
    anchor = np.asarray(obj.metadata["visual_anchor"], dtype=np.float64)
    target = positions[str(obj.metadata["visual_target"])]
    vector = target - anchor
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        vector = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        length = 1.0
    surface_offset = float(obj.metadata.get("visual_target_surface_offset_m", 0.0))
    endpoint = target - vector * min(surface_offset / length, 0.95)
    vector = endpoint - anchor
    midpoint = (anchor + endpoint) * 0.5
    euler = _quat_vector_as_euler((float(vector[0]), float(vector[1]), float(vector[2])))
    quat = legacy._quat_from_euler_deg(list(euler))
    return (tuple(float(value) for value in midpoint), quat)


def _make_constraint(descriptor: dict[str, object], body_ids: dict[str, int]) -> int:
    body_name = str(descriptor["body"])
    body_id = body_ids[body_name]
    parent_body_name = descriptor.get("parent_body")
    parent_body_id = -1 if parent_body_name is None else body_ids[str(parent_body_name)]
    parent_frame = [
        float(value)
        for value in descriptor.get("parent_frame", descriptor.get("parent_anchor", (0.0, 0.0, 0.0)))
    ]
    child_frame = [float(value) for value in descriptor["child_frame"]]
    constraint_type = str(descriptor["type"])
    if constraint_type == "point2point":
        joint_type = p.JOINT_POINT2POINT
        axis = [0.0, 0.0, 0.0]
    elif constraint_type == "revolute":
        # The generic PyBullet constraint API is portable here as a
        # point-to-point anchor; the board can rotate around the shared axis.
        joint_type = p.JOINT_POINT2POINT
        axis = [float(value) for value in descriptor["axis"]]
    else:
        raise ValueError(f"unsupported V2V constraint type: {constraint_type}")
    constraint_id = p.createConstraint(
        parent_body_id,
        -1,
        body_id,
        -1,
        joint_type,
        axis,
        parent_frame,
        child_frame,
    )
    p.changeConstraint(constraint_id, maxForce=500.0)
    return int(constraint_id)


def _create_visual_audit_bodies(
    visual_objects: list[ObjectInstanceSpec],
    legacy_by_name: dict[str, legacy.ObjectSpec],
) -> dict[str, int]:
    """Create non-interacting collision proxies for visual-only geometry."""
    body_ids: dict[str, int] = {}
    for obj in visual_objects:
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=legacy._collision_shape(legacy_by_name[obj.name]),
            basePosition=list(obj.position),
            baseOrientation=legacy._quat_from_euler_deg(list(obj.orientation_euler_deg)),
        )
        # Proxies stay queryable by getClosestPoints while contributing no
        # collision response to the actual simulation.
        p.setCollisionFilterGroupMask(body_id, -1, 0, 0)
        body_ids[obj.name] = int(body_id)
    return body_ids


def _physical_positions(body_ids: dict[str, int]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(p.getBasePositionAndOrientation(body_id)[0], dtype=np.float64)
        for name, body_id in body_ids.items()
    }


def _update_visual_audit_bodies(
    visual_objects: list[ObjectInstanceSpec],
    audit_body_ids: dict[str, int],
    physical_positions: dict[str, np.ndarray],
) -> None:
    for obj in visual_objects:
        pose_pos, pose_quat = _visual_pose(obj, physical_positions)
        p.resetBasePositionAndOrientation(
            audit_body_ids[obj.name],
            list(pose_pos),
            pose_quat,
        )


def audit_v2v_case_initialization(
    case: DemoCase,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Run the mandatory initialization checks without rendering a video."""
    blueprint = case.blueprint
    scenario = blueprint_to_legacy_scenario(blueprint, seed=seed)
    all_objects = list(blueprint.objects)
    visual_only_names = {
        obj.name for obj in all_objects if bool(obj.metadata.get("visual_only"))
    }
    physical_objects = [obj for obj in all_objects if obj.name not in visual_only_names]
    visual_objects = [obj for obj in all_objects if obj.name in visual_only_names]
    legacy_by_name = {obj.name: obj for obj in scenario.objects}

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("could not connect to PyBullet DIRECT for V2V initialization audit")
    constraint_ids: list[int] = []
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -EARTH_GRAVITY)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / SIM_HZ,
            numSolverIterations=120,
            numSubSteps=1,
        )
        plane_id = p.loadURDF("plane.urdf")
        surface = build_surface_catalog()[blueprint.surface_key]
        floor_mu = float(np.clip(surface.floor_friction_range.midpoint(), 0.05, 1.20))
        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=floor_mu,
            restitution=0.02,
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        )

        body_ids: dict[str, int] = {}
        for obj in physical_objects:
            legacy_obj = legacy_by_name[obj.name]
            body_id = p.createMultiBody(
                baseMass=float(obj.mass) if obj.dynamic else 0.0,
                baseCollisionShapeIndex=legacy._collision_shape(legacy_obj),
                basePosition=list(obj.position),
                baseOrientation=legacy._quat_from_euler_deg(list(obj.orientation_euler_deg)),
            )
            p.changeDynamics(
                body_id,
                -1,
                restitution=float(obj.restitution),
                lateralFriction=float(obj.friction),
                linearDamping=float(obj.linear_damping),
                angularDamping=float(obj.angular_damping),
                activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            )
            p.resetBaseVelocity(
                body_id,
                linearVelocity=list(obj.linear_velocity),
                angularVelocity=list(obj.angular_velocity),
            )
            body_ids[obj.name] = int(body_id)

        for descriptor in blueprint.metadata.get("constraints", []):
            constraint_ids.append(_make_constraint(dict(descriptor), body_ids))
        for left_name, right_name in blueprint.metadata.get("disable_collision_pairs", []):
            p.setCollisionFilterPair(
                body_ids[str(left_name)], body_ids[str(right_name)], -1, -1, 0
            )

        visual_audit_body_ids = _create_visual_audit_bodies(
            visual_objects,
            legacy_by_name,
        )
        audit_body_ids = {**body_ids, **visual_audit_body_ids}
        stages = [
            legacy.assert_initialization_contacts(
                audit_body_ids,
                plane_id,
                stage="post_creation",
            )
        ]
        for _ in range(int(round(blueprint.pre_roll_s * SIM_HZ))):
            p.stepSimulation()
        positions = _physical_positions(body_ids)
        _update_visual_audit_bodies(
            visual_objects,
            visual_audit_body_ids,
            positions,
        )
        stages.append(
            legacy.assert_initialization_contacts(
                audit_body_ids,
                plane_id,
                stage="post_pre_roll",
            )
        )
        stages.append(
            legacy.assert_initialization_contacts(
                audit_body_ids,
                plane_id,
                stage="video_frame_0",
            )
        )
        return {
            "case_id": case.case_id,
            "passed": True,
            "penetration_tolerance_m": legacy.INITIALIZATION_PENETRATION_TOLERANCE_M,
            "stages": stages,
        }
    finally:
        for constraint_id in constraint_ids:
            try:
                p.removeConstraint(constraint_id)
            except Exception:
                pass
        p.disconnect(client)


def _appearance_seed(obj: ObjectInstanceSpec) -> int:
    appearance_group = str(obj.metadata.get("appearance_group", obj.name))
    return sum(
        (index + 1) * ord(character)
        for index, character in enumerate(appearance_group)
    ) % (2**32 - 1)


def _render_case(
    case: DemoCase,
    *,
    seed: int,
    output_root: Path,
    width: int,
    height: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    materials = build_material_catalog()
    register_material_assets(materials)
    blueprint = case.blueprint
    scenario = blueprint_to_legacy_scenario(blueprint, seed=seed)
    all_objects = list(blueprint.objects)
    object_by_name = {obj.name: obj for obj in all_objects}
    visual_only_names = {
        obj.name for obj in all_objects if bool(obj.metadata.get("visual_only"))
    }
    physical_objects = [obj for obj in all_objects if obj.name not in visual_only_names]
    visual_objects = [obj for obj in all_objects if obj.name in visual_only_names]
    legacy_by_name = {obj.name: obj for obj in scenario.objects}

    with override_legacy_runtime(
        output_root=output_root,
        camera=blueprint.camera,
        width=width,
        height=height,
    ):
        client = p.connect(p.DIRECT)
        if client < 0:
            raise RuntimeError("could not connect to PyBullet DIRECT")
        renderer = None
        body_ids: dict[str, int] = {}
        visual_audit_body_ids: dict[str, int] = {}
        constraint_ids: list[int] = []
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetSimulation()
            p.setGravity(0.0, 0.0, -EARTH_GRAVITY)
            p.setPhysicsEngineParameter(
                fixedTimeStep=1.0 / SIM_HZ,
                numSolverIterations=120,
                numSubSteps=1,
            )
            plane_id = p.loadURDF("plane.urdf")
            surface = build_surface_catalog()[blueprint.surface_key]
            floor_mu = float(np.clip(surface.floor_friction_range.midpoint(), 0.05, 1.20))
            p.changeDynamics(
                plane_id,
                -1,
                lateralFriction=floor_mu,
                restitution=0.02,
                activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            )
            renderer = RealismPreviewRenderer(
                camera=blueprint.camera,
                surface_key=blueprint.surface_key,
                lighting_key=blueprint.lighting_key,
                width=width,
                height=height,
                scene_style=SCENE_STYLE,
                capture_instance_masks=True,
            )
            renderer.object_materials = {
                obj.name: materials[obj.material_key] for obj in all_objects
            }
            renderer.object_texture_seeds = {
                obj.name: _appearance_seed(obj) for obj in all_objects
            }
            for obj in all_objects:
                renderer.add_object(legacy_by_name[obj.name])
            for obj in physical_objects:
                legacy_obj = legacy_by_name[obj.name]
                shape_id = legacy._collision_shape(legacy_obj)
                body_id = p.createMultiBody(
                    baseMass=float(obj.mass) if obj.dynamic else 0.0,
                    baseCollisionShapeIndex=shape_id,
                    basePosition=list(obj.position),
                    baseOrientation=legacy._quat_from_euler_deg(list(obj.orientation_euler_deg)),
                )
                p.changeDynamics(
                    body_id,
                    -1,
                    restitution=float(obj.restitution),
                    lateralFriction=float(obj.friction),
                    linearDamping=float(obj.linear_damping),
                    angularDamping=float(obj.angular_damping),
                    activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
                )
                p.resetBaseVelocity(
                    body_id,
                    linearVelocity=list(obj.linear_velocity),
                    angularVelocity=list(obj.angular_velocity),
                )
                body_ids[obj.name] = int(body_id)
            for descriptor in blueprint.metadata.get("constraints", []):
                constraint_ids.append(_make_constraint(dict(descriptor), body_ids))
            for left_name, right_name in blueprint.metadata.get("disable_collision_pairs", []):
                p.setCollisionFilterPair(
                    body_ids[str(left_name)], body_ids[str(right_name)], -1, -1, 0
                )

            visual_audit_body_ids = _create_visual_audit_bodies(
                visual_objects,
                legacy_by_name,
            )
            audit_body_ids = {**body_ids, **visual_audit_body_ids}
            initialization_qa = [
                legacy.assert_initialization_contacts(
                    audit_body_ids,
                    plane_id,
                    stage="post_creation",
                )
            ]

            pre_roll_steps = int(round(blueprint.pre_roll_s * SIM_HZ))
            for _ in range(pre_roll_steps):
                p.stepSimulation()
            pre_roll_positions = _physical_positions(body_ids)
            _update_visual_audit_bodies(
                visual_objects,
                visual_audit_body_ids,
                pre_roll_positions,
            )
            initialization_qa.append(
                legacy.assert_initialization_contacts(
                    audit_body_ids,
                    plane_id,
                    stage="post_pre_roll",
                )
            )

            object_names = [obj.name for obj in all_objects]
            object_index = {name: idx for idx, name in enumerate(object_names)}
            total_steps = int(SIM_DURATION_S * SIM_HZ)
            record_every = SIM_HZ // FPS
            frame_count = math.ceil(total_steps / record_every)
            positions = np.zeros((frame_count, len(all_objects), 3), dtype=np.float32)
            quats = np.zeros((frame_count, len(all_objects), 4), dtype=np.float32)
            linear_velocities = np.zeros((frame_count, len(all_objects), 3), dtype=np.float32)
            angular_velocities = np.zeros((frame_count, len(all_objects), 3), dtype=np.float32)
            frames: list[np.ndarray] = []
            for step in range(total_steps):
                if step % record_every != 0:
                    p.stepSimulation()
                    continue
                frame_index = step // record_every
                current_positions: dict[str, np.ndarray] = {}
                for obj in physical_objects:
                    body_id = body_ids[obj.name]
                    pos, quat = p.getBasePositionAndOrientation(body_id)
                    linvel, angvel = p.getBaseVelocity(body_id)
                    pos_arr = np.asarray(pos, dtype=np.float32)
                    quat_arr = np.asarray(quat, dtype=np.float32)
                    current_positions[obj.name] = pos_arr
                    idx = object_index[obj.name]
                    positions[frame_index, idx] = pos_arr
                    quats[frame_index, idx] = quat_arr
                    linear_velocities[frame_index, idx] = np.asarray(linvel, dtype=np.float32)
                    angular_velocities[frame_index, idx] = np.asarray(angvel, dtype=np.float32)
                    renderer.update_pose(obj.name, list(pos), list(quat))
                if frame_index == 0:
                    _update_visual_audit_bodies(
                        visual_objects,
                        visual_audit_body_ids,
                        current_positions,
                    )
                    initialization_qa.append(
                        legacy.assert_initialization_contacts(
                            audit_body_ids,
                            plane_id,
                            stage="video_frame_0",
                        )
                    )
                for obj in all_objects:
                    if obj.name not in visual_only_names:
                        continue
                    pose_pos, pose_quat = _visual_pose(obj, current_positions)
                    idx = object_index[obj.name]
                    positions[frame_index, idx] = np.asarray(pose_pos, dtype=np.float32)
                    quats[frame_index, idx] = np.asarray(pose_quat, dtype=np.float32)
                    if frame_index > 0:
                        linear_velocities[frame_index, idx] = (
                            positions[frame_index, idx] - positions[frame_index - 1, idx]
                        ) * FPS
                    renderer.update_pose(obj.name, list(pose_pos), pose_quat)
                frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                p.stepSimulation()
            positions = positions[: len(frames)]
            quats = quats[: len(frames)]
            linear_velocities = linear_velocities[: len(frames)]
            angular_velocities = angular_velocities[: len(frames)]
            video_path = output_root / "videos" / f"{case.case_id}.mp4"
            context_path = output_root / "context" / f"{case.case_id}_context8f.mp4"
            context16_path = output_root / "context" / f"{case.case_id}_context16f.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.parent.mkdir(parents=True, exist_ok=True)
            legacy._write_video_h264(video_path, frames)
            legacy._write_video_h264(context_path, frames[:CONTEXT_FRAMES])
            legacy._write_video_h264(context16_path, frames[:CONTEXT_FRAME_OPTIONS[-1]])
            states_path = output_root / "meta" / f"{case.case_id}_states.npz"
            states_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                states_path,
                positions=positions,
                quats=quats,
                linear_velocities=linear_velocities,
                angular_velocities=angular_velocities,
                frame_times=np.arange(len(frames), dtype=np.float32) / FPS,
                object_names=np.asarray(object_names, dtype=np.str_),
                object_roles=np.asarray([obj.role for obj in all_objects], dtype=np.str_),
                frame_width=np.asarray([width], dtype=np.int32),
                frame_height=np.asarray([height], dtype=np.int32),
            )
            first_event_frame = _first_event_frame(case, positions, quats)
            mask_ids = np.stack(renderer.mask_frames).astype(np.uint8, copy=False)
            mask_path = output_root / "masks" / f"{case.case_id}_instance_ids.npz"
            mask_video_path = output_root / "masks" / f"{case.case_id}_instance_mask.mp4"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                mask_path,
                instance_ids=mask_ids,
                object_names=np.asarray(list(renderer.instance_ids), dtype=np.str_),
                object_ids=np.asarray(list(renderer.instance_ids.values()), dtype=np.uint8),
            )
            palette_bgr = np.asarray(
                [[0, 0, 0], [76, 76, 230], [204, 153, 51], [102, 204, 76]],
                dtype=np.uint8,
            )
            legacy._write_video_h264(
                mask_video_path,
                [palette_bgr[np.minimum(frame, len(palette_bgr) - 1)] for frame in mask_ids],
            )
            all_visible = bool(
                all(
                    np.all(np.any(mask_ids == object_id, axis=(1, 2)))
                    for object_id in renderer.instance_ids.values()
                )
            )
            state_summary = _state_summary(
                positions, linear_velocities, angular_velocities, object_names
            )
            qa = {
                "context_frames": CONTEXT_FRAMES,
                "context_duration_s": CONTEXT_FRAMES / FPS,
                "context_frame_options": list(CONTEXT_FRAME_OPTIONS),
                "context16_duration_s": CONTEXT_FRAME_OPTIONS[-1] / FPS,
                "first_event_frame": first_event_frame,
                "first_event_time_s": (
                    round(first_event_frame / FPS, 4) if first_event_frame is not None else None
                ),
                "event_after_context": bool(
                    first_event_frame is not None and first_event_frame >= CONTEXT_FRAMES
                ),
                "all_objects_visible_every_frame": all_visible,
                "frame_count": len(frames),
                "initialization": {
                    "passed": True,
                    "penetration_tolerance_m": legacy.INITIALIZATION_PENETRATION_TOLERANCE_M,
                    "stages": initialization_qa,
                },
            }
            meta = {
                "case_id": case.case_id,
                "family_key": case.family_key,
                "family_title": case.family_title,
                "family_description": case.family_description,
                "title": case.title,
                "description": case.description,
                "seed": seed,
                "gravity": EARTH_GRAVITY,
                "fps": FPS,
                "sim_hz": SIM_HZ,
                "duration_s": SIM_DURATION_S,
                "context_frames": CONTEXT_FRAMES,
                "context_duration_s": CONTEXT_FRAMES / FPS,
                "video": str(video_path),
                "context_video": str(context_path),
                "context16_video": str(context16_path),
                "states": str(states_path),
                "mask_video": str(mask_video_path),
                "mask_ids": str(mask_path),
                "camera": {
                    "eye": list(blueprint.camera.eye),
                    "target": list(blueprint.camera.target),
                    "up": list(blueprint.camera.up),
                    "yfov_deg": blueprint.camera.yfov_deg,
                },
                "surface_key": blueprint.surface_key,
                "lighting_key": blueprint.lighting_key,
                "scene_style": SCENE_STYLE,
                "controlled_variable": case.controlled_variable,
                "controlled_value": case.controlled_value,
                "controlled_value_label": case.controlled_value_label,
                "event_rule": case.event_rule,
                "scenario_spec": _json_safe(blueprint.metadata),
                "objects": [_json_safe(_object_payload(obj)) for obj in all_objects],
                "state_summary": state_summary,
                "qa": qa,
            }
            meta_path = output_root / "meta" / f"{case.case_id}.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest = {
                "sample_key": case.case_id,
                "family_key": case.family_key,
                "video": str(video_path),
                "context_video": str(context_path),
                "context16_video": str(context16_path),
                "states": str(states_path),
                "meta": str(meta_path),
                "mask_video": str(mask_video_path),
                "mask_ids": str(mask_path),
                "instance_id_map": {name: idx + 1 for idx, name in enumerate(object_names)},
                "width": width,
                "height": height,
                "caption": case.description,
                "context_caption": f"First {CONTEXT_FRAMES} frames before {case.event_rule}.",
                "context16_caption": f"First {CONTEXT_FRAME_OPTIONS[-1]} frames before {case.event_rule}.",
            }
            (output_root / "cases" / case.family_key / case.case_id).mkdir(
                parents=True, exist_ok=True
            )
            # Keep the case directory as a lightweight index while artifacts live
            # in the family-specific output root for simple viewer serving.
            manifest_path = output_root / "cases" / case.family_key / case.case_id / "case_manifest.json"
            manifest_path.write_text(
                json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return {
                "case_id": case.case_id,
                "status": "rendered",
                "difficulty": {
                    "level": case.level,
                    "title": "短上下文物理条件",
                    "description": "前 8 帧可观察决定后续运动的几何或状态条件。",
                    "priority": 2 if case.level == "L2" else 3,
                },
                "scene_family": {
                    "family_key": case.family_key,
                    "title": case.family_title,
                    "description": case.family_description,
                    "target_event_types": [case.event_rule],
                },
                "scene_title": case.title,
                "scene_description": case.description,
                "scene_style": SCENE_STYLE,
                "scenario_spec": _json_safe(
                    {
                        **blueprint.metadata,
                        "controlled_variable": case.controlled_variable,
                        "controlled_value": case.controlled_value,
                        "controlled_value_label": case.controlled_value_label,
                        "units": case.units,
                        "objects": [_object_payload(obj) for obj in all_objects],
                    }
                ),
                "state_summary": state_summary,
                "video": str(video_path),
                "context_video": str(context_path),
                "context16_video": str(context16_path),
                "video_url": "/media/" + case.case_id,
                "context_video_url": "/media-context/" + case.case_id,
                "context16_video_url": "/media-context16/" + case.case_id,
                "meta": str(meta_path),
                "states": str(states_path),
                "mask_video": str(mask_video_path),
                "mask_ids": str(mask_path),
                "question": V2V_QUESTION,
                "response_final": "仅记录仿真条件与轨迹，尚未运行 VLM。",
                "response_raw": "",
                "answer_source": "simulation_v2v_context_demo",
                "caption_intent_only": case.description,
                "v2v": {
                    "context_frames": CONTEXT_FRAMES,
                    "context_duration_s": CONTEXT_FRAMES / FPS,
                    "controlled_variable": case.controlled_variable,
                    "controlled_value": case.controlled_value,
                    "controlled_value_label": case.controlled_value_label,
                    "event_rule": case.event_rule,
                    **qa,
                },
            }
        finally:
            for constraint_id in constraint_ids:
                try:
                    p.removeConstraint(constraint_id)
                except Exception:
                    pass
            if renderer is not None:
                renderer.cleanup()
            p.disconnect(client)


def generate_demo_batch(
    *,
    output_root: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed_base: int = 20260819,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, case in enumerate(build_demo_cases(seed_base)):
        seed = seed_base + index * 1009
        case_root = output_root / "artifacts" / case.family_key / case.case_id
        try:
            row = _render_case(
                case,
                seed=seed,
                output_root=case_root,
                width=width,
                height=height,
            )
            rows.append(row)
            print(f"rendered {case.case_id}", flush=True)
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append(
                {
                    "case_id": case.case_id,
                    "family_key": case.family_key,
                    "error": repr(exc),
                }
            )
            print(f"failed {case.case_id}: {exc!r}", flush=True)
    results_path = output_root / "cases.jsonl"
    results_path.write_text(
        "".join(json.dumps(_json_safe(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "failure_report.json").write_text(
        json.dumps(_json_safe(failures), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    qa_summary = {
        "requested": len(build_demo_cases(seed_base)),
        "rendered": len(rows),
        "failed": len(failures),
        "all_events_after_context": bool(
            rows and all(bool(row.get("v2v", {}).get("event_after_context")) for row in rows)
        ),
        "all_objects_visible_every_frame": bool(
            rows and all(bool(row.get("v2v", {}).get("all_objects_visible_every_frame")) for row in rows)
        ),
    }
    (output_root / "reports" / "summary.json").write_text(
        json.dumps(_json_safe(qa_summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed-base", type=int, default=20260819)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_demo_batch(
        output_root=args.output_root,
        width=args.width,
        height=args.height,
        seed_base=args.seed_base,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
