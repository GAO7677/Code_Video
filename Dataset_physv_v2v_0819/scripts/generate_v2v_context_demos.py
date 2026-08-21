"""Generate short-context physical demos for video continuation experiments.

Each control group keeps the object, materials, camera, and simulation settings
fixed while changing one visible geometric or initial-state variable.  The
first 8 and 16 30-fps frames are exported separately so two continuation
boundaries can be reviewed alongside the complete simulated trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, replace
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
from .initialization_contracts_0819 import (
    build_contact_contract,
    validate_color_separation,
)
from .material_catalog_0705 import build_material_catalog, build_surface_catalog
from .render_sim_0705 import (
    RealismPreviewRenderer,
    blueprint_to_legacy_scenario,
    override_legacy_runtime,
    register_material_assets,
    write_ground_truth_capture,
)
from .scene_generators_0705 import EARTH_GRAVITY, _collision_vertical_extent
from .taxonomy_0819 import taxonomy_for_family

try:
    from . import generate_sim_preview_gallery as legacy
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
BOWL_BALL_CENTER_HEIGHT_ABOVE_BOTTOM_M = 0.42
BOWL_BALL_CLEARANCE_M = 0.0
OBSTACLE_BALL_REFERENCE_RADIUS_M = 0.11
OBSTACLE_BALL_REFERENCE_MASS_KG = 1.0
OBSTACLE_BALL_DENSITY_KG_M3 = OBSTACLE_BALL_REFERENCE_MASS_KG / (
    (4.0 / 3.0) * math.pi * OBSTACLE_BALL_REFERENCE_RADIUS_M**3
)
PUCK_BARRIER_NORMAL_ANGLES_DEG = (30.0, 45.0, 60.0, 75.0, 90.0)
DOOR_FRAME_OPENING_WIDTHS_M = (0.38, 0.46, 0.54, 0.62, 0.74)
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
            "taxonomy": taxonomy_for_family(family_key),
        },
    )
    color_qa = validate_color_separation(blueprint)
    blueprint = replace(
        blueprint,
        metadata={
            **blueprint.metadata,
            "initialization_contract": build_contact_contract(blueprint),
            "color_separation_qa": color_qa,
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
            # Leave enough visible approach distance for both context lengths.
            position=(-1.16, 0.0, platform_top + ball_radius),
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
        eye=(0.25, -4.25, 1.30),
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


def _make_obstacle_case(sample_key: str, initial_speed: float) -> DemoCase:
    ball_radius = 0.11
    barrier_hz = 0.24
    barrier_x = 0.80
    ball_start_x = -1.55
    objects = [
        _object(
            name="obstacle_ball",
            family_key="ball",
            shape="sphere",
            size={"radius": ball_radius},
            material_key="rubber_red",
            # The barrier and release point are fixed; only the initial speed
            # changes across this control group.
            position=(ball_start_x, 0.0, ball_radius),
            dynamic=True,
            mass=1.0,
            friction=0.22,
            restitution=0.85,
            velocity=(initial_speed, 0.0, 0.0),
            # For motion along +X, positive Y rotation makes the bottom
            # contact point stationary relative to the floor.
            angular_velocity=(0.0, initial_speed / ball_radius, 0.0),
            linear_damping=0.045,
            angular_damping=0.03,
            metadata={
                "appearance_group": "v2v_obstacle_red_rubber_ball_v1",
                "rolling_friction": 0.006,
                "spinning_friction": 0.006,
                "ccd_swept_sphere_radius_m": ball_radius * 0.95,
            },
        ),
        _object(
            name="obstacle_barrier",
            family_key="barrier_box",
            shape="box",
            # Slightly thicken the fixed barrier so the high-speed ball cannot
            # tunnel through its collision shape between simulation steps.
            size={"hx": 0.12, "hy": 0.32, "hz": barrier_hz},
            material_key="painted_metal_blue",
            position=(barrier_x, 0.0, barrier_hz),
            dynamic=False,
            mass=0.0,
            friction=0.12,
            # A high-restitution blue barrier makes the incoming-speed
            # difference observable in the rebound while leaving the floor
            # restitution unchanged.
            restitution=0.90,
        ),
    ]
    camera = _camera(
        eye=(0.05, -4.00, 1.12),
        target=(0.05, 0.0, 0.48),
        yfov_deg=48.0,
    )
    blueprint = _blueprint(
        family_key="V2V_OBSTACLE",
        sample_key=sample_key,
        title=f"Ball rolls at {initial_speed:.1f} m/s toward a fixed obstacle",
        description="A red ball starts from the same position in every case and approaches the same blue barrier at a controlled initial speed.",
        objects=objects,
        camera=camera,
        surface_key="studio_wood_floor",
        tags=("v2v", "initial_speed", "collision", "left_to_right", "rebound"),
        metadata={
            "controlled_variable": "initial_speed_mps",
            "ball_start_x_m": ball_start_x,
            "obstacle_x_m": barrier_x,
            "barrier_restitution": 0.90,
            "initial_speed_mps": initial_speed,
            "barrier_half_x_m": 0.12,
            "ball_radius_m": ball_radius,
            "contact_margin_m": 0.05,
            "physics_sub_steps": 8,
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_OBSTACLE",
        family_title="障碍碰撞：初速度",
        family_description="蓝色挡板和小球起点固定，小球以不同初速度出发，碰撞时速度、反弹强度和后续运动距离因此不同。",
        level="L2",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="initial_speed_mps",
        controlled_value=initial_speed,
        controlled_value_label=f"v={initial_speed:.1f} m/s",
        units="m/s",
        blueprint=blueprint,
        event_rule="ball_contacts_barrier",
    )


def _make_obstacle_size_case(sample_key: str, ball_radius: float) -> DemoCase:
    """Create a radius-control obstacle case with constant ball density.

    The collision scene, start position, linear speed, angular speed, contact
    parameters, barrier, camera, and material stay fixed.  Only the sphere
    radius changes; its grounded z position and mass are derived from radius.
    """
    barrier_hz = 0.24
    barrier_x = 0.80
    ball_start_x = -1.30
    initial_speed = 3.00
    initial_angular_speed = initial_speed / OBSTACLE_BALL_REFERENCE_RADIUS_M
    ball_mass = OBSTACLE_BALL_DENSITY_KG_M3 * (4.0 / 3.0) * math.pi * ball_radius**3
    objects = [
        _object(
            name="obstacle_ball",
            family_key="ball",
            shape="sphere",
            size={"radius": ball_radius},
            material_key="rubber_red",
            position=(ball_start_x, 0.0, ball_radius),
            dynamic=True,
            mass=ball_mass,
            friction=0.22,
            restitution=0.85,
            velocity=(initial_speed, 0.0, 0.0),
            # Keep angular speed identical across the radius controls.  The
            # resulting radius-dependent slip is part of the controlled scene.
            angular_velocity=(0.0, initial_angular_speed, 0.0),
            linear_damping=0.045,
            angular_damping=0.03,
            metadata={
                "appearance_group": "v2v_obstacle_red_rubber_ball_v1",
                "rolling_friction": 0.006,
                "spinning_friction": 0.006,
                "ccd_swept_sphere_radius_m": ball_radius * 0.95,
                "density_kg_m3": OBSTACLE_BALL_DENSITY_KG_M3,
                "mass_from_volume": True,
                "reference_radius_m": OBSTACLE_BALL_REFERENCE_RADIUS_M,
            },
        ),
        _object(
            name="obstacle_barrier",
            family_key="barrier_box",
            shape="box",
            size={"hx": 0.12, "hy": 0.32, "hz": barrier_hz},
            material_key="painted_metal_teal",
            position=(barrier_x, 0.0, barrier_hz),
            dynamic=False,
            mass=0.0,
            friction=0.12,
            restitution=0.90,
        ),
    ]
    camera = _camera(
        eye=(0.05, -4.00, 1.12),
        target=(0.05, 0.0, 0.48),
        yfov_deg=48.0,
    )
    blueprint = _blueprint(
        family_key="V2V_OBSTACLE_SIZE",
        sample_key=sample_key,
        title=f"Obstacle collision with ball radius {ball_radius:.3f} m",
        description="The same rolling ball scene is repeated with a different sphere radius; density and all other physical settings remain fixed.",
        objects=objects,
        camera=camera,
        surface_key="studio_wood_floor",
        tags=("v2v", "ball_radius", "constant_density", "collision", "left_to_right", "rebound"),
        metadata={
            "controlled_variable": "ball_radius_m",
            "ball_radius_m": ball_radius,
            "ball_density_kg_m3": OBSTACLE_BALL_DENSITY_KG_M3,
            "ball_mass_kg": ball_mass,
            "ball_volume_m3": (4.0 / 3.0) * math.pi * ball_radius**3,
            "ball_start_x_m": ball_start_x,
            "obstacle_x_m": barrier_x,
            "barrier_restitution": 0.90,
            "initial_speed_mps": initial_speed,
            "initial_angular_speed_radps": initial_angular_speed,
            "barrier_half_x_m": 0.12,
            "contact_margin_m": 0.05,
            "physics_sub_steps": 8,
            "constant_parameters": [
                "ball_start_x_m",
                "initial_speed_mps",
                "initial_angular_speed_radps",
                "friction",
                "restitution",
                "linear_damping",
                "angular_damping",
                "barrier_geometry",
                "barrier_material",
                "camera",
            ],
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="V2V_OBSTACLE_SIZE",
        family_title="障碍碰撞：球体积与质量",
        family_description="障碍物、初始位置、速度和接触参数固定，球密度相同，仅改变球半径，因此质量随体积变化。",
        level="L2",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="ball_radius_m",
        controlled_value=ball_radius,
        controlled_value_label=f"r={ball_radius:.3f} m",
        units="m",
        blueprint=blueprint,
        event_rule="ball_contacts_barrier",
    )


def _bowl_curve_size(radius: float, span: float, bottom_z: float) -> dict[str, float]:
    """Parameters for the shared continuous physical/rendering bowl shell."""
    return {
        "radius": float(radius),
        "span": float(span),
        "bottom_z": float(bottom_z),
        "thickness": 0.065,
        "half_y": 0.48,
        "segments": 96.0,
    }


def _bowl_inner_wall_x_for_ball_height(
    *,
    radius: float,
    ball_center_height_above_bottom: float,
    ball_surface_offset: float,
) -> float:
    """Return the left inner-wall contact coordinate for a target ball height.

    The contact face is a circular arc.  Solving the sphere-center height
    analytically keeps the ball's gravitational potential fixed while radius
    is the only controlled variable.
    """
    if radius <= ball_surface_offset:
        raise ValueError("bowl radius must exceed ball surface offset")
    if not ball_surface_offset < ball_center_height_above_bottom < radius:
        raise ValueError("target ball center height must lie inside the bowl radius")
    root = radius * (radius - ball_center_height_above_bottom) / (radius - ball_surface_offset)
    root = float(np.clip(root, 0.0, radius))
    return -math.sqrt(max(radius * radius - root * root, 0.0))


def _make_bowl_case(sample_key: str, radius: float) -> DemoCase:
    bottom_z = 0.16
    ball_radius = 0.115
    ball_clearance = BOWL_BALL_CLEARANCE_M
    ball_surface_offset = ball_radius + ball_clearance
    ball_surface_x = _bowl_inner_wall_x_for_ball_height(
        radius=radius,
        ball_center_height_above_bottom=BOWL_BALL_CENTER_HEIGHT_ABOVE_BOTTOM_M,
        ball_surface_offset=ball_surface_offset,
    )
    # Keep a small physical rim beyond the release point while using one
    # continuous curved shell for both rendering and collision.
    span = abs(ball_surface_x) / 0.90
    root = math.sqrt(radius * radius - ball_surface_x * ball_surface_x)
    surface_slope = ball_surface_x / root
    surface_theta = math.atan(surface_slope)
    surface_normal = (-math.sin(surface_theta), 0.0, math.cos(surface_theta))
    ball_surface_z = bottom_z + radius - root
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
            size={"hx": span + 0.18, "hy": 0.56, "hz": 0.045},
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
            # The release has no imposed kinetic energy.  Motion is driven by
            # the fixed, visible gravitational potential on the inner wall.
            velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_damping=0.01,
            angular_damping=0.01,
            metadata={"appearance_group": "v2v_bowl_blue_rubber_ball_v1"},
        ),
        _object(
            name="bowl_surface",
            family_key="curved_container",
            shape="bowl_curve",
            size=_bowl_curve_size(radius, span, bottom_z),
            material_key="wood_plywood",
            position=(0.0, 0.0, 0.0),
            dynamic=False,
            mass=0.0,
            friction=0.72,
            restitution=0.03,
            role="anchored_static",
        ),
    ]
    camera = _camera(
        eye=(0.10, -3.85, 1.35),
        target=(0.00, 0.0, 0.60),
        yfov_deg=46.0,
        hdri_key="studio_warm",
    )
    blueprint = _blueprint(
        family_key="V2V_BOWL",
        sample_key=sample_key,
        title=f"Ball in a bowl with radius {radius:.2f} m",
        description="A blue rubber ball is released from the same visible height on the inner wall of a curved container; only the bowl curvature changes.",
        objects=objects,
        camera=camera,
        surface_key="residential_wood_floor",
        tags=("v2v", "curved_container", "gravity", "rolling"),
        metadata={
            "controlled_variable": "bowl_radius_m",
            "bowl_radius_m": radius,
            "bowl_span_m": span,
            "bowl_bottom_z_m": bottom_z,
            "ball_center_height_above_bottom_m": BOWL_BALL_CENTER_HEIGHT_ABOVE_BOTTOM_M,
            "ball_initial_surface_x_m": ball_surface_x,
            "ball_contact_segment_name": "bowl_surface",
            "bowl_surface_segments": 96,
            "bowl_surface_thickness_m": 0.065,
            "initial_speed_mps": 0.0,
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
    anchor_z = 2.55
    angle_deg = 18.0
    angle = math.radians(angle_deg)
    bob_radius = 0.18
    bob_x = anchor_x + length * math.sin(angle)
    bob_z = anchor_z - length * math.cos(angle)
    rope_clearance = 0.0
    rope_radius = 0.018
    rope_anchor = (anchor_x, 0.0, anchor_z - rope_radius * math.sin(angle))
    rope_end_x = bob_x - bob_radius * math.sin(angle)
    rope_end_z = bob_z + bob_radius * math.cos(angle)
    rope_vec = (rope_end_x - rope_anchor[0], 0.0, rope_end_z - rope_anchor[2])
    rope_length = max(0.04, math.sqrt(sum(value * value for value in rope_vec)) - rope_clearance)
    rope_center = (
        (rope_anchor[0] + rope_end_x) * 0.5,
        0.0,
        (rope_anchor[2] + rope_end_z) * 0.5,
    )
    post_half_height = (anchor_z - 0.18) * 0.5
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
            position=(anchor_x, 0.11, anchor_z + 0.03),
            dynamic=False,
            mass=0.0,
            friction=0.80,
            restitution=0.02,
        ),
        _object(
            name="pendulum_rope",
            family_key="table_leg",
            shape="cylinder",
            size={"radius": rope_radius, "height": rope_length},
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
                "visual_anchor": rope_anchor,
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
        eye=(0.42, -3.85, 1.68),
        target=(-0.48, 0.0, 1.25),
        yfov_deg=48.0,
        hdri_key="hall_bright",
    )
    blueprint = _blueprint(
        family_key="V2V_PENDULUM",
        sample_key=sample_key,
        title=f"Pendulum with length {length:.2f} m",
        description="A larger red bob hangs from a thick, high-contrast rope and begins at a raised visible angle; only the pendulum length changes.",
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
    # A horizontal cylindrical fulcrum replaces the old rectangular block.
    # The board is tangent to its top and uses two point constraints separated
    # along the shaft, so only rotation around the shaft remains free.
    pivot_radius = 0.10
    pivot_height = pivot_radius
    pivot_center_z = pivot_radius
    pivot_axis_z = pivot_center_z + pivot_radius
    board_center_z = pivot_axis_z + 0.045
    board_hx = 1.35
    board_hy = 0.26
    board_hz = 0.045
    board_angle = 0.0
    theta = math.radians(board_angle)
    load_hx, load_hy, load_hz = 0.14, 0.14, 0.12
    hinge_half_span = 0.20
    local_z = board_hz + load_hz
    load_position = (
        load_x * math.cos(theta) + local_z * math.sin(theta),
        0.0,
        board_center_z - load_x * math.sin(theta) + local_z * math.cos(theta),
    )
    objects = [
        _object(
            name="seesaw_pivot",
            family_key="cylindrical_fulcrum",
            shape="cylinder",
            size={"radius": pivot_radius, "height": 0.64},
            material_key="concrete_painted",
            position=(0.0, 0.0, pivot_center_z),
            dynamic=False,
            mass=0.0,
            friction=0.90,
            restitution=0.02,
            orientation=(90.0, 0.0, 0.0),
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
        # The longer board needs a wider physical framing so both endpoints
        # remain visible throughout the rotation.
        eye=(0.0, -5.25, 1.40),
        target=(0.0, 0.0, 0.46),
        yfov_deg=48.0,
        hdri_key="studio_warm",
    )
    blueprint = _blueprint(
        family_key="V2V_SEESAW",
        sample_key=sample_key,
        title=f"Seesaw load at x={load_x:.2f} m on a 2.70 m board",
        description=(
            "A block rests on a longer hinged board; its position is sampled uniformly "
            "from the center to the safe edge, and only that position changes."
        ),
        objects=objects,
        camera=camera,
        surface_key="residential_wood_floor",
        tags=("v2v", "seesaw", "hinge", "gravity"),
        metadata={
            "controlled_variable": "load_position_x_m",
            "load_position_x_m": load_x,
            "pivot_z_m": pivot_height,
            "pivot_axis_z_m": pivot_axis_z,
            "pivot_shape": "horizontal_cylinder",
            "pivot_radius_m": pivot_radius,
            "board_length_m": 2.0 * board_hx,
            # Disable direct shaft-board collision. Their meshes remain
            # tangent, while two separated shaft constraints define the hinge
            # axis without allowing an unintended roll toward the camera.
            "disable_collision_pairs": [("seesaw_pivot", "seesaw_board")],
            "initial_board_angle_deg": board_angle,
            "hinge_half_span_m": hinge_half_span,
            "constraints": [
                {
                    "type": "point2point",
                    "body": "seesaw_board",
                    "parent_body": "seesaw_pivot",
                    "axis": (0.0, 0.0, 1.0),
                    # The pivot is rotated 90 degrees around X: local Z maps
                    # to the world-Y shaft direction with the opposite sign.
                    "parent_frame": (0.0, pivot_radius, -hinge_half_span),
                    "child_frame": (0.0, hinge_half_span, pivot_axis_z - board_center_z),
                },
                {
                    "type": "point2point",
                    "body": "seesaw_board",
                    "parent_body": "seesaw_pivot",
                    "axis": (0.0, 0.0, 1.0),
                    "parent_frame": (0.0, pivot_radius, hinge_half_span),
                    "child_frame": (0.0, -hinge_half_span, pivot_axis_z - board_center_z),
                }
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


def _make_puck_barrier_case(sample_key: str, normal_angle_deg: float) -> DemoCase:
    """Create a scene-control case with a fixed puck and rotated barrier."""
    puck_radius = 0.16
    puck_height = 0.06
    puck_start_x = -1.55
    barrier_x = 0.65
    barrier_hx = 0.045
    barrier_hy = 0.72
    barrier_hz = 0.06
    angle = math.radians(normal_angle_deg)
    barrier_orientation_z_deg = 90.0 - normal_angle_deg
    objects = [
        _object(
            name="puck",
            family_key="ice_puck",
            shape="puck",
            size={"radius": puck_radius, "height": puck_height},
            material_key="wood_dark",
            # Convex-mesh collision margins require a 1 mm initialization
            # clearance; gravity settles the puck onto the floor in pre-roll.
            position=(puck_start_x, 0.0, puck_height * 0.5 + 0.001),
            dynamic=True,
            mass=0.30,
            friction=0.0,
            restitution=0.90,
            velocity=(1.40, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_damping=0.004,
            angular_damping=0.01,
            metadata={
                "appearance_group": "scene_puck_barrier_dark_ice_puck_v1",
                "rolling_friction": 0.001,
                "spinning_friction": 0.001,
            },
        ),
        _object(
            name="puck_barrier",
            family_key="rigid_barrier",
            shape="box",
            size={"hx": barrier_hx, "hy": barrier_hy, "hz": barrier_hz},
            material_key="painted_metal_blue",
            position=(barrier_x, 0.0, barrier_hz),
            dynamic=False,
            mass=0.0,
            friction=0.0,
            restitution=0.95,
            orientation=(0.0, 0.0, barrier_orientation_z_deg),
        ),
    ]
    camera = _camera(
        eye=(0.20, -4.25, 1.18),
        target=(0.20, -0.40, 0.24),
        yfov_deg=45.0,
        hdri_key="hall_bright",
    )
    blueprint = _blueprint(
        family_key="SCENE_PUCK_BARRIER",
        sample_key=sample_key,
        title=f"Ice puck barrier collision with {normal_angle_deg:.0f} degree normal",
        description=(
            "A disk-shaped ice puck slides in a straight line across a low-friction floor into a fixed rigid barrier. "
            "The barrier center, puck, speed, and physical parameters are fixed; only the barrier plane normal direction changes."
        ),
        objects=objects,
        camera=camera,
        surface_key="painted_concrete_floor",
        tags=("scene", "puck", "barrier", "reflection", "low_friction", "left_to_right"),
        metadata={
            "controlled_variable": "barrier_normal_angle_deg",
            "barrier_normal_angle_deg": normal_angle_deg,
            "barrier_normal_reference": "angle from +Y transverse direction; 90 degrees is head-on",
            "barrier_normal_xy": (math.sin(angle), math.cos(angle)),
            "barrier_center_x_m": barrier_x,
            "barrier_half_x_m": barrier_hx,
            "barrier_half_y_m": barrier_hy,
            "barrier_restitution": 0.95,
            "puck_radius_m": puck_radius,
            "puck_height_m": puck_height,
            "initial_speed_mps": 1.40,
            "floor_friction": 0.04,
            "rebound_solver": "normal_impulse_correction_for_thin_disk_floor_wall_contact",
            "physics_sub_steps": 8,
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="SCENE_PUCK_BARRIER",
        family_title="冰球撞固定挡板",
        family_description="冰球、挡板中心、速度和材质固定，只改变挡板平面法线方向，观察碰撞后的反射方向和轨迹。",
        level="L2",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="barrier_normal_angle_deg",
        controlled_value=normal_angle_deg,
        controlled_value_label=f"normal={normal_angle_deg:.0f} degrees",
        units="deg",
        blueprint=blueprint,
        event_rule="puck_contacts_barrier",
    )


def _make_door_frame_case(sample_key: str, opening_width: float) -> DemoCase:
    """Create a scene-control case with a fixed crate and variable doorway."""
    crate_hx, crate_hy, crate_hz = 0.34, 0.24, 0.28
    crate_start_x = -1.55
    crate_start_y = 0.10
    crate_speed = 1.20
    frame_x = 0.72
    frame_hx = 0.08
    frame_side_hy = 0.10
    frame_side_hz = 0.50
    frame_height = 2.0 * frame_side_hz
    lintel_hz = 0.10
    outer_half_y = opening_width * 0.5 + frame_side_hy
    wall_outer_half_y = 1.55
    wall_inner_edge_y = outer_half_y + frame_side_hy
    wall_half_y = 0.5 * (wall_outer_half_y - wall_inner_edge_y)
    wall_center_y = 0.5 * (wall_outer_half_y + wall_inner_edge_y)
    wall_half_x = 0.14
    wall_half_z = 0.5 * (frame_height + 2.0 * lintel_hz)
    objects = [
        _object(
            name="door_crate",
            family_key="wooden_crate",
            shape="box",
            size={"hx": crate_hx, "hy": crate_hy, "hz": crate_hz},
            material_key="wood_red",
            position=(crate_start_x, crate_start_y, crate_hz),
            dynamic=True,
            mass=2.40,
            friction=0.10,
            restitution=0.08,
            velocity=(crate_speed, 0.0, 0.0),
            linear_damping=0.025,
            angular_damping=0.07,
            metadata={"appearance_group": "scene_door_frame_red_wood_crate_v1"},
        ),
        _object(
            name="door_frame_left",
            family_key="door_frame",
            shape="box",
            size={"hx": frame_hx, "hy": frame_side_hy, "hz": frame_side_hz},
            material_key="painted_metal_teal",
            position=(frame_x, -outer_half_y, frame_side_hz),
            dynamic=False,
            mass=0.0,
            friction=0.46,
            restitution=0.05,
        ),
        _object(
            name="door_frame_right",
            family_key="door_frame",
            shape="box",
            size={"hx": frame_hx, "hy": frame_side_hy, "hz": frame_side_hz},
            material_key="painted_metal_teal",
            position=(frame_x, outer_half_y, frame_side_hz),
            dynamic=False,
            mass=0.0,
            friction=0.46,
            restitution=0.05,
        ),
        _object(
            name="door_frame_lintel",
            family_key="door_frame",
            shape="box",
            size={"hx": frame_hx, "hy": outer_half_y + frame_side_hy, "hz": lintel_hz},
            material_key="painted_metal_teal",
            position=(frame_x, 0.0, frame_height + lintel_hz),
            dynamic=False,
            mass=0.0,
            friction=0.46,
            restitution=0.05,
            role="anchored_static",
        ),
        _object(
            name="door_wall_left",
            family_key="door_wall",
            shape="box",
            size={"hx": wall_half_x, "hy": wall_half_y, "hz": wall_half_z},
            material_key="wall_beige",
            position=(frame_x, -wall_center_y, wall_half_z),
            dynamic=False,
            mass=0.0,
            friction=0.50,
            restitution=0.02,
            role="anchored_static",
        ),
        _object(
            name="door_wall_right",
            family_key="door_wall",
            shape="box",
            size={"hx": wall_half_x, "hy": wall_half_y, "hz": wall_half_z},
            material_key="wall_beige",
            position=(frame_x, wall_center_y, wall_half_z),
            dynamic=False,
            mass=0.0,
            friction=0.50,
            restitution=0.02,
            role="anchored_static",
        ),
    ]
    camera = _camera(
        # Look across the doorway plane from the crate's approach side so
        # both posts and the lintel remain legible in the same view.
        eye=(-4.10, -2.35, 1.35),
        target=(0.25, 0.0, 0.82),
        yfov_deg=46.0,
        hdri_key="studio_warm",
    )
    blueprint = _blueprint(
        family_key="SCENE_DOOR_FRAME",
        sample_key=sample_key,
        title=f"Wooden crate through a {opening_width:.2f} m doorway",
        description=(
            "A rectangular wooden crate moves forward with fixed size, pose, speed, and lateral offset toward a fixed-thickness door frame. "
            "Only the doorway opening width changes, producing different passage and contact outcomes."
        ),
        objects=objects,
        camera=camera,
        surface_key="residential_wood_floor",
        tags=("scene", "door_frame", "crate", "clearance", "contact", "left_to_right"),
        metadata={
            "controlled_variable": "door_opening_width_m",
            "door_opening_width_m": opening_width,
            "door_frame_center_x_m": frame_x,
            "door_frame_half_x_m": frame_hx,
            "door_frame_thickness_m": 2.0 * frame_hx,
            "door_frame_height_m": frame_height + 2.0 * lintel_hz,
            "door_wall_outer_half_y_m": wall_outer_half_y,
            "door_wall_height_m": 2.0 * wall_half_z,
            "door_wall_thickness_m": 2.0 * wall_half_x,
            "crate_size_m": {"length_x": 2.0 * crate_hx, "width_y": 2.0 * crate_hy, "height_z": 2.0 * crate_hz},
            "crate_half_x_m": crate_hx,
            "crate_initial_y_m": crate_start_y,
            "initial_speed_mps": crate_speed,
            "floor_friction": 0.10,
            "physics_sub_steps": 8,
        },
    )
    return DemoCase(
        case_id=sample_key,
        family_key="SCENE_DOOR_FRAME",
        family_title="木箱穿过门框",
        family_description="木箱的尺寸、姿态、初始位置和速度固定，只改变门框开口宽度，观察通过、擦碰、卡住和旋转。",
        level="L3",
        title=blueprint.title,
        description=blueprint.description,
        controlled_variable="door_opening_width_m",
        controlled_value=opening_width,
        controlled_value_label=f"opening={opening_width:.2f} m",
        units="m",
        blueprint=blueprint,
        event_rule="crate_reaches_door_frame",
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
            material_key="painted_metal_yellow",
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
        eye=(0.20, -3.90, 1.02),
        target=(0.00, 0.0, 0.48),
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
        family_description="短上下文显示第一块的倾斜、排列方向和物体间距，间距决定后续碰撞能否连续传播。",
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
    for value in (0.06, 0.22, 0.38, 0.54, 0.70):
        cases.append(_make_gap_case(f"v2v_gap_{int(value * 100):03d}", value))
    for value in (1.2, 1.4, 1.6, 1.8, 5.2):
        cases.append(_make_obstacle_case(f"v2v_obstacle_v{int(round(value * 100)):03d}", value))
    for value in (0.08, 0.11, 0.14, 0.17, 0.20):
        cases.append(_make_obstacle_size_case(f"v2v_obstacle_size_r{int(value * 1000):03d}", value))
    for value in (0.80, 1.30, 1.80, 2.30, 2.80):
        cases.append(_make_bowl_case(f"v2v_bowl_r{int(value * 100):03d}", value))
    for value in (0.55, 0.83, 1.10, 1.38, 1.65):
        cases.append(_make_pendulum_case(f"v2v_pendulum_l{int(value * 100):03d}", value))
    # The load travels uniformly from the board center to a small safety
    # margin before the edge, so no case starts with the block overhanging.
    seesaw_edge_x = 1.35 - 0.14 - 0.04
    for value in np.linspace(0.0, seesaw_edge_x, 5):
        value = float(value)
        cases.append(_make_seesaw_case(f"v2v_seesaw_x{int(round(value * 100)):03d}", value))
    for value in (0.00, 0.045, 0.09, 0.135, 0.18):
        cases.append(_make_domino_case(f"v2v_domino_g{int(round(value * 1000)):03d}", value))
    for value in PUCK_BARRIER_NORMAL_ANGLES_DEG:
        cases.append(_make_puck_barrier_case(f"scene_puck_barrier_n{int(value):03d}", value))
    for value in DOOR_FRAME_OPENING_WIDTHS_M:
        cases.append(_make_door_frame_case(f"scene_door_frame_w{int(round(value * 100)):03d}", value))
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
        "rolling_friction": round(float(obj.metadata.get("rolling_friction", 0.0)), 5),
        "spinning_friction": round(float(obj.metadata.get("spinning_friction", 0.0)), 5),
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
    elif case.family_key in {"V2V_OBSTACLE", "V2V_OBSTACLE_SIZE"}:
        ball = positions[:, index["obstacle_ball"], 0]
        barrier = float(metadata["obstacle_x_m"])
        contact_distance = (
            float(metadata.get("barrier_half_x_m", 0.12))
            + float(metadata.get("ball_radius_m", 0.11))
            + float(metadata.get("contact_margin_m", 0.03))
        )
        frames = np.flatnonzero(np.abs(ball - barrier) <= contact_distance)
    elif case.family_key == "SCENE_PUCK_BARRIER":
        puck = positions[:, index["puck"], 0]
        barrier = float(metadata["barrier_center_x_m"])
        contact_distance = (
            float(metadata.get("barrier_half_x_m", 0.045))
            + float(metadata.get("puck_radius_m", 0.12))
            + float(metadata.get("contact_margin_m", 0.04))
        )
        frames = np.flatnonzero(np.abs(puck - barrier) <= contact_distance)
    elif case.family_key == "SCENE_DOOR_FRAME":
        crate = positions[:, index["door_crate"], 0]
        frame = float(metadata["door_frame_center_x_m"])
        contact_distance = (
            float(metadata.get("door_frame_half_x_m", 0.08))
            + float(metadata.get("crate_half_x_m", 0.28))
            + float(metadata.get("contact_margin_m", 0.05))
        )
        frames = np.flatnonzero(np.abs(crate - frame) <= contact_distance)
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
        # The cylindrical fulcrum and paired axis anchors produce a compact
        # rotation, so use a 1-degree threshold for the first visible motion.
        frames = np.flatnonzero(np.abs(angles) > math.radians(1.0))
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
        # PyBullet's generic constraint API does not expose a reliable
        # revolute constraint between two independent bodies in this setup.
        # A centered point anchor leaves the intended gravity-driven rotation
        # free while the shaft provides the visible hinge geometry.
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
            numSolverIterations=legacy.PHYSICS_SOLVER_ITERATIONS,
            numSubSteps=int(blueprint.metadata.get("physics_sub_steps", legacy.PHYSICS_SUB_STEPS)),
            contactERP=legacy.PHYSICS_CONTACT_ERP,
            erp=legacy.PHYSICS_CONTACT_ERP,
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
            dynamics_kwargs = dict(
                restitution=float(obj.restitution),
                lateralFriction=float(obj.friction),
                rollingFriction=float(obj.metadata.get("rolling_friction", 0.0)),
                spinningFriction=float(obj.metadata.get("spinning_friction", 0.0)),
                linearDamping=float(obj.linear_damping),
                angularDamping=float(obj.angular_damping),
                activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            )
            if "ccd_swept_sphere_radius_m" in obj.metadata:
                dynamics_kwargs.update(
                    ccdSweptSphereRadius=float(obj.metadata["ccd_swept_sphere_radius_m"]),
                    contactProcessingThreshold=0.0,
                )
            p.changeDynamics(body_id, -1, **dynamics_kwargs)
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
            ),
            legacy.assert_initialization_contact_contract(
                audit_body_ids,
                plane_id,
                blueprint.metadata.get("initialization_contract"),
                stage="post_creation_contract",
            ),
        ]
        for _ in range(int(round(blueprint.pre_roll_s * SIM_HZ))):
            p.stepSimulation()
        positions = _physical_positions(body_ids)
        _update_visual_audit_bodies(
            visual_objects,
            visual_audit_body_ids,
            positions,
        )
        stages.extend(
            [
                legacy.assert_initialization_contacts(
                    audit_body_ids,
                    plane_id,
                    stage="post_pre_roll",
                ),
                legacy.assert_initialization_contact_contract(
                    audit_body_ids,
                    plane_id,
                    blueprint.metadata.get("initialization_contract"),
                    stage="post_pre_roll_contract",
                ),
                legacy.assert_initialization_contacts(
                    audit_body_ids,
                    plane_id,
                    stage="video_frame_0",
                ),
                legacy.assert_initialization_contact_contract(
                    audit_body_ids,
                    plane_id,
                    blueprint.metadata.get("initialization_contract"),
                    stage="video_frame_0_contract",
                ),
            ]
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


def _step_v2v_simulation(
    body_ids: dict[str, int],
    blueprint: ScenarioBlueprint,
) -> dict[str, object] | None:
    """Advance PyBullet and preserve a visible rebound for a thin puck.

    A thin disk resting on the floor can remain in a persistent wall/floor
    contact manifold.  In that configuration Bullet resolves the first
    oblique impact as a zero-normal-speed sliding contact even when both
    bodies have restitution.  Apply the missing normal impulse once, only
    when the pre-step velocity is approaching the barrier; tangential motion,
    gravity, and all other contacts remain solver-controlled.
    """
    if blueprint.family_key != "SCENE_PUCK_BARRIER":
        p.stepSimulation()
        return None

    puck_id = body_ids.get("puck")
    barrier_id = body_ids.get("puck_barrier")
    if puck_id is None or barrier_id is None:
        p.stepSimulation()
        return None

    previous_velocity = np.asarray(p.getBaseVelocity(puck_id)[0], dtype=np.float64)
    p.stepSimulation()
    contacts = p.getContactPoints(puck_id, barrier_id)
    if not contacts:
        return None

    contact = max(
        contacts,
        key=lambda point: math.hypot(float(point[7][0]), float(point[7][1])),
    )
    normal = np.asarray(contact[7], dtype=np.float64)
    normal[2] = 0.0
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-8:
        return None
    normal /= normal_length
    current_velocity = np.asarray(p.getBaseVelocity(puck_id)[0], dtype=np.float64)
    incoming_normal_speed = float(np.dot(previous_velocity, normal))
    solved_normal_speed = float(np.dot(current_velocity, normal))
    objects = {obj.name: obj for obj in blueprint.objects}
    effective_restitution = min(
        float(objects["puck"].restitution),
        float(objects["puck_barrier"].restitution),
    )
    if incoming_normal_speed >= -0.08 or solved_normal_speed >= 0.08:
        return None

    desired_normal_speed = -effective_restitution * incoming_normal_speed
    correction = desired_normal_speed - solved_normal_speed
    current_velocity += correction * normal
    angular_velocity = p.getBaseVelocity(puck_id)[1]
    p.resetBaseVelocity(
        puck_id,
        linearVelocity=current_velocity.tolist(),
        angularVelocity=angular_velocity,
    )
    return {
        "incoming_normal_speed_mps": round(incoming_normal_speed, 6),
        "solver_normal_speed_mps": round(solved_normal_speed, 6),
        "outgoing_normal_speed_mps": round(desired_normal_speed, 6),
        "effective_restitution": round(effective_restitution, 6),
        "normal_xy": [round(float(value), 6) for value in normal[:2]],
        "corrected_velocity_xy_mps": [
            round(float(value), 6) for value in current_velocity[:2]
        ],
    }


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
    ground_truth_output_dir: Path | None = None,
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
        rebound_corrections: list[dict[str, object]] = []
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetSimulation()
            p.setGravity(0.0, 0.0, -EARTH_GRAVITY)
            p.setPhysicsEngineParameter(
                fixedTimeStep=1.0 / SIM_HZ,
                numSolverIterations=legacy.PHYSICS_SOLVER_ITERATIONS,
                numSubSteps=int(blueprint.metadata.get("physics_sub_steps", legacy.PHYSICS_SUB_STEPS)),
                contactERP=legacy.PHYSICS_CONTACT_ERP,
                erp=legacy.PHYSICS_CONTACT_ERP,
            )
            plane_id = p.loadURDF("plane.urdf")
            surface = build_surface_catalog()[blueprint.surface_key]
            floor_mu = float(
                np.clip(
                    blueprint.metadata.get("floor_friction", surface.floor_friction_range.midpoint()),
                    0.01,
                    1.20,
                )
            )
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
                capture_rgb_frames_dir=(
                    ground_truth_output_dir / "frames"
                    if ground_truth_output_dir is not None
                    else None
                ),
                capture_depth_frames=ground_truth_output_dir is not None,
            )
            if ground_truth_output_dir is not None:
                renderer.capture_contact_records = True
                renderer.contact_dynamic_names = {
                    obj.name for obj in physical_objects if obj.dynamic
                }
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
                dynamics_kwargs = dict(
                    restitution=float(obj.restitution),
                    lateralFriction=float(obj.friction),
                    rollingFriction=float(obj.metadata.get("rolling_friction", 0.0)),
                    spinningFriction=float(obj.metadata.get("spinning_friction", 0.0)),
                    linearDamping=float(obj.linear_damping),
                    angularDamping=float(obj.angular_damping),
                    activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
                )
                if "ccd_swept_sphere_radius_m" in obj.metadata:
                    dynamics_kwargs.update(
                        ccdSweptSphereRadius=float(obj.metadata["ccd_swept_sphere_radius_m"]),
                        contactProcessingThreshold=0.0,
                    )
                p.changeDynamics(body_id, -1, **dynamics_kwargs)
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
                ),
                legacy.assert_initialization_contact_contract(
                    audit_body_ids,
                    plane_id,
                    blueprint.metadata.get("initialization_contract"),
                    stage="post_creation_contract",
                ),
            ]

            physics_step_index = 0

            def step_simulation() -> None:
                nonlocal physics_step_index
                correction = _step_v2v_simulation(body_ids, blueprint)
                if correction is not None:
                    rebound_corrections.append(
                        {
                            "substep_index": physics_step_index,
                            "time_s": round(physics_step_index / SIM_HZ, 6),
                            **correction,
                        }
                    )
                physics_step_index += 1

            pre_roll_steps = int(round(blueprint.pre_roll_s * SIM_HZ))
            for _ in range(pre_roll_steps):
                step_simulation()
            pre_roll_positions = _physical_positions(body_ids)
            _update_visual_audit_bodies(
                visual_objects,
                visual_audit_body_ids,
                pre_roll_positions,
            )
            initialization_qa.extend(
                [
                    legacy.assert_initialization_contacts(
                        audit_body_ids,
                        plane_id,
                        stage="post_pre_roll",
                    ),
                    legacy.assert_initialization_contact_contract(
                        audit_body_ids,
                        plane_id,
                        blueprint.metadata.get("initialization_contract"),
                        stage="post_pre_roll_contract",
                    ),
                ]
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
                    step_simulation()
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
                    initialization_qa.extend(
                        [
                            legacy.assert_initialization_contacts(
                                audit_body_ids,
                                plane_id,
                                stage="video_frame_0",
                            ),
                            legacy.assert_initialization_contact_contract(
                                audit_body_ids,
                                plane_id,
                                blueprint.metadata.get("initialization_contract"),
                                stage="video_frame_0_contract",
                            ),
                        ]
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
                if renderer.capture_contact_records:
                    renderer.contact_records.extend(
                        legacy.collect_frame_contacts(
                            body_ids,
                            plane_id,
                            frame_index=frame_index,
                            time_s=frame_index / FPS,
                            dynamic_names=renderer.contact_dynamic_names,
                        )
                    )
                frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
                step_simulation()
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
            if ground_truth_output_dir is not None:
                ground_truth_capture = write_ground_truth_capture(
                    output_dir=ground_truth_output_dir,
                    renderer=renderer,
                )
            else:
                ground_truth_capture = None
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
                "event_after_context16": bool(
                    first_event_frame is not None
                    and first_event_frame >= CONTEXT_FRAME_OPTIONS[-1]
                ),
                "all_objects_visible_every_frame": all_visible,
                "frame_count": len(frames),
                "initialization": {
                    "passed": True,
                    "penetration_tolerance_m": legacy.INITIALIZATION_PENETRATION_TOLERANCE_M,
                    "stages": initialization_qa,
                },
                "puck_rebound_corrections": rebound_corrections,
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
                "context_frame_options": list(CONTEXT_FRAME_OPTIONS),
                "video": str(video_path),
                "context_video": str(context_path),
                "context16_video": str(context16_path),
                "states": str(states_path),
                "mask_video": str(mask_video_path),
                "mask_ids": str(mask_path),
                "ground_truth_capture": ground_truth_capture,
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
                "initialization_qa": qa["initialization"],
                "qa": qa,
                "puck_rebound_corrections": rebound_corrections,
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
                "initialization_qa": qa["initialization"],
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
                        "description": "短上下文中可观察决定后续运动的几何或状态条件。",
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
                "initialization_qa": qa["initialization"],
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
