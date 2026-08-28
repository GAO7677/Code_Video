from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

import numpy as np

from .common_specs import (
    CameraSpec,
    ObjectFamilySpec,
    ObjectInstanceSpec,
    RangeSpec,
    ScenarioBlueprint,
    ScenarioFamilySpec,
)
from .material_catalog_0705 import build_material_catalog
from .initialization_contracts_0819 import (
    build_contact_contract,
    validate_color_separation,
)
from .object_catalog_0705 import build_object_family_catalog


EARTH_GRAVITY = 9.81
NOMINAL_RENDER_WIDTH = 1280
NOMINAL_RENDER_HEIGHT = 720
DIRECTION_MODES = {"left_to_right", "right_to_left", "vertical"}
# Generic scenes keep the established conservative framing.  Dedicated F11,
# F12, and V2V cameras below are tightened independently after their motion
# envelopes are known.
DEFAULT_CAMERA_DISTANCE_SCALE = 0.88
F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG = -48.0
INITIAL_GROUND_CLEARANCE_M = 0.006


def build_camera_catalog() -> dict[str, CameraSpec]:
    cameras = [
        CameraSpec(
            eye=(0.0, -3.05, 1.46),
            target=(0.0, 0.24, 0.36),
            yfov_deg=48.0,
            jitter_eye_xyz=(0.16, 0.14, 0.08),
            jitter_target_xyz=(0.18, 0.20, 0.08),
            jitter_fov_deg=4.0,
            hdri_key="studio_soft",
        ),
        CameraSpec(
            eye=(0.22, -2.90, 1.34),
            target=(0.02, 0.20, 0.30),
            yfov_deg=52.0,
            jitter_eye_xyz=(0.14, 0.12, 0.06),
            jitter_target_xyz=(0.16, 0.18, 0.08),
            jitter_fov_deg=3.0,
            hdri_key="hall_neutral",
        ),
        CameraSpec(
            eye=(-0.18, -3.10, 1.52),
            target=(0.0, 0.28, 0.40),
            yfov_deg=46.0,
            jitter_eye_xyz=(0.18, 0.14, 0.08),
            jitter_target_xyz=(0.18, 0.18, 0.08),
            jitter_fov_deg=4.0,
            hdri_key="studio_warm",
        ),
        CameraSpec(
            eye=(-0.45, -2.65, 1.62),
            target=(0.12, 0.22, 0.34),
            yfov_deg=42.0,
            jitter_eye_xyz=(0.18, 0.12, 0.08),
            jitter_target_xyz=(0.18, 0.15, 0.06),
            jitter_fov_deg=3.0,
            hdri_key="studio_soft",
        ),
        CameraSpec(
            eye=(0.38, -3.35, 1.28),
            target=(-0.08, 0.26, 0.30),
            yfov_deg=56.0,
            jitter_eye_xyz=(0.20, 0.16, 0.06),
            jitter_target_xyz=(0.18, 0.16, 0.06),
            jitter_fov_deg=4.0,
            hdri_key="hall_neutral",
        ),
        CameraSpec(
            eye=(0.0, -2.35, 1.82),
            target=(0.0, 0.18, 0.58),
            yfov_deg=36.0,
            jitter_eye_xyz=(0.16, 0.10, 0.08),
            jitter_target_xyz=(0.16, 0.12, 0.08),
            jitter_fov_deg=2.0,
            hdri_key="studio_warm",
        ),
        CameraSpec(
            eye=(0.55, -2.55, 1.18),
            target=(0.02, 0.30, 0.28),
            yfov_deg=39.0,
            jitter_eye_xyz=(0.12, 0.10, 0.05),
            jitter_target_xyz=(0.12, 0.12, 0.05),
            jitter_fov_deg=2.0,
            hdri_key="hall_neutral",
        ),
        CameraSpec(
            eye=(-0.62, -2.85, 1.26),
            target=(0.08, 0.18, 0.30),
            yfov_deg=44.0,
            jitter_eye_xyz=(0.14, 0.12, 0.05),
            jitter_target_xyz=(0.14, 0.12, 0.05),
            jitter_fov_deg=2.5,
            hdri_key="studio_soft",
        ),
        CameraSpec(
            eye=(0.20, -2.15, 1.68),
            target=(0.02, 0.16, 0.54),
            yfov_deg=35.0,
            jitter_eye_xyz=(0.10, 0.08, 0.05),
            jitter_target_xyz=(0.10, 0.08, 0.05),
            jitter_fov_deg=1.5,
            hdri_key="studio_warm",
        ),
        CameraSpec(
            # F11: a mostly frontal view that includes both the high table
            # surface and the floor-side rebound without a top-down angle.
            eye=(0.0, -4.05, 1.20),
            target=(0.25, 0.0, 0.82),
            yfov_deg=54.0,
            jitter_eye_xyz=(0.0, 0.0, 0.0),
            jitter_target_xyz=(0.0, 0.0, 0.0),
            jitter_fov_deg=0.0,
            hdri_key="studio_warm",
        ),
        CameraSpec(
            # F12 runs from the raised left end of the ramp to the right-side
            # floor. Center the full simulated rollout rather than only the
            # support, with a shallow downward view that keeps the ramp
            # surface and block-floor contact visually legible.
            eye=(2.00, -6.40, 1.25),
            target=(2.00, 0.0, 0.62),
            yfov_deg=50.0,
            jitter_eye_xyz=(0.0, 0.0, 0.0),
            jitter_target_xyz=(0.0, 0.0, 0.0),
            jitter_fov_deg=0.0,
            hdri_key="hall_bright",
        ),
    ]
    return {f"cam_{idx:02d}": camera for idx, camera in enumerate(cameras)}


def build_scenario_family_catalog() -> dict[str, ScenarioFamilySpec]:
    families = [
        ScenarioFamilySpec(
            key="F1",
            title="Single Object Motion",
            description="Single-object rolling, sliding, tumbling and bouncing.",
            family_slug="F1_single_object",
            min_dynamic_objects=1,
            max_dynamic_objects=1,
            min_total_objects=1,
            max_total_objects=2,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("first_bounce", "peak_speed", "stop"),
            preferred_surface_keys=("studio_wood_floor", "residential_wood_floor", "dark_wood_floor"),
            preferred_camera_keys=("cam_00", "cam_02", "cam_03", "cam_06"),
            motion_modes=("roll", "slide", "bounce", "spin", "glance"),
            speed_range=(1.0, 5.2),
            spin_range=(0.0, 12.0),
            angle_range_deg=(0.0, 28.0),
        ),
        ScenarioFamilySpec(
            key="F2",
            title="Two-Object Interaction",
            description="Impact transfer between one driver and one target.",
            family_slug="F2_two_object",
            min_dynamic_objects=2,
            max_dynamic_objects=2,
            min_total_objects=2,
            max_total_objects=3,
            supports_occlusion=False,
            supports_support_objects=False,
            target_event_types=("first_contact", "max_impulse", "post_impact_turn"),
            preferred_surface_keys=("studio_wood_floor", "residential_wood_floor", "painted_concrete_floor"),
            preferred_camera_keys=("cam_00", "cam_01", "cam_04", "cam_05"),
            motion_modes=("head_on", "glance", "crossing", "offset_push"),
            speed_range=(1.8, 5.5),
            spin_range=(0.0, 10.0),
            angle_range_deg=(2.0, 36.0),
        ),
        ScenarioFamilySpec(
            key="F3",
            title="Chain Reaction",
            description="Three-body causal propagation with at least two contacts.",
            family_slug="F3_chain_reaction",
            min_dynamic_objects=3,
            max_dynamic_objects=3,
            min_total_objects=3,
            max_total_objects=4,
            supports_occlusion=False,
            supports_support_objects=False,
            target_event_types=("first_contact", "second_contact", "peak_chain_motion"),
            preferred_surface_keys=("studio_wood_floor", "residential_wood_floor", "painted_concrete_floor"),
            preferred_camera_keys=("cam_00", "cam_01", "cam_03", "cam_06"),
            motion_modes=("domino", "push_chain", "rolling_chain", "offset_chain"),
            speed_range=(1.4, 4.8),
            spin_range=(0.0, 10.0),
            angle_range_deg=(-48.0, -12.0),
        ),
        ScenarioFamilySpec(
            key="F4",
            title="Occlusion and Reappearance",
            description="Visible-to-hidden and hidden-to-visible transitions with identity ambiguity.",
            family_slug="F4_occlusion",
            min_dynamic_objects=1,
            max_dynamic_objects=2,
            min_total_objects=2,
            max_total_objects=4,
            supports_occlusion=True,
            supports_support_objects=False,
            target_event_types=("enter_occlusion", "full_occlusion", "reappear"),
            preferred_surface_keys=("studio_wood_floor", "residential_wood_floor", "dark_wood_floor"),
            preferred_camera_keys=("cam_00", "cam_02", "cam_04", "cam_05"),
            motion_modes=("left_pass", "right_pass", "cross", "double_pass"),
            speed_range=(1.2, 4.2),
            spin_range=(0.0, 8.0),
            angle_range_deg=(0.0, 18.0),
        ),
        ScenarioFamilySpec(
            key="F5",
            title="Support and Drop",
            description="Support loss, ramp exit and controlled topple.",
            family_slug="F5_drop_support",
            min_dynamic_objects=1,
            max_dynamic_objects=2,
            min_total_objects=2,
            max_total_objects=4,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("support_loss", "drop_start", "land"),
            preferred_surface_keys=("dark_wood_floor", "residential_wood_floor", "painted_concrete_floor"),
            preferred_camera_keys=("cam_01", "cam_02", "cam_04", "cam_06"),
            motion_modes=("drop", "topple", "slide_off", "roll_off"),
            speed_range=(0.2, 1.6),
            spin_range=(0.0, 6.0),
            angle_range_deg=(0.0, 22.0),
        ),
        ScenarioFamilySpec(
            key="F6",
            title="Ramp Slide",
            description="Objects sliding down a visible incline with variable roll-out behavior.",
            family_slug="F6_ramp_slide",
            min_dynamic_objects=1,
            max_dynamic_objects=2,
            min_total_objects=2,
            max_total_objects=4,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("ramp_entry", "ramp_exit", "land"),
            preferred_surface_keys=("residential_wood_floor", "painted_concrete_floor"),
            preferred_camera_keys=("cam_02", "cam_04", "cam_06"),
            motion_modes=("shallow_slide", "steep_slide", "rollout"),
            speed_range=(0.8, 3.6),
            spin_range=(0.0, 8.0),
            angle_range_deg=(8.0, 35.0),
        ),
        ScenarioFamilySpec(
            key="F7",
            title="Spin Dominant Motion",
            description="Strong angular motion with translation secondary.",
            family_slug="F7_spin_dominant",
            min_dynamic_objects=1,
            max_dynamic_objects=2,
            min_total_objects=1,
            max_total_objects=3,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("peak_spin", "orientation_change", "stop"),
            preferred_surface_keys=("studio_wood_floor", "dark_wood_floor"),
            preferred_camera_keys=("cam_00", "cam_03", "cam_06"),
            motion_modes=("high_spin", "reverse_spin", "wobble_spin"),
            speed_range=(0.6, 3.2),
            spin_range=(6.0, 16.0),
            angle_range_deg=(0.0, 22.0),
        ),
        ScenarioFamilySpec(
            key="F8",
            title="Bounce Heavy Motion",
            description="Repeated rebound and settling with high restitution objects.",
            family_slug="F8_bounce_heavy",
            min_dynamic_objects=1,
            max_dynamic_objects=2,
            min_total_objects=1,
            max_total_objects=3,
            supports_occlusion=False,
            supports_support_objects=False,
            target_event_types=("first_bounce", "second_bounce", "settle"),
            preferred_surface_keys=("studio_wood_floor", "residential_wood_floor"),
            preferred_camera_keys=("cam_00", "cam_01", "cam_05"),
            motion_modes=("vertical_drop", "oblique_drop", "multi_bounce"),
            speed_range=(1.0, 5.0),
            spin_range=(0.0, 8.0),
            angle_range_deg=(0.0, 20.0),
        ),
        ScenarioFamilySpec(
            key="F9",
            title="Clutter Interaction",
            description="Local interaction among multiple objects in a room-like cluttered area.",
            family_slug="F9_clutter_interaction",
            min_dynamic_objects=2,
            max_dynamic_objects=3,
            min_total_objects=3,
            max_total_objects=6,
            supports_occlusion=True,
            supports_support_objects=True,
            target_event_types=("contact", "occlusion", "settle"),
            preferred_surface_keys=("residential_wood_floor", "painted_concrete_floor"),
            preferred_camera_keys=("cam_02", "cam_04", "cam_06"),
            motion_modes=("crowded_slide", "offset_collision", "spill"),
            speed_range=(0.8, 4.0),
            spin_range=(0.0, 8.0),
            angle_range_deg=(0.0, 26.0),
        ),
        ScenarioFamilySpec(
            key="F10",
            title="Edge and Boundary",
            description="Edge-aware motion around table, platform or floor boundary.",
            family_slug="F10_edge_boundary",
            min_dynamic_objects=1,
            max_dynamic_objects=2,
            min_total_objects=2,
            max_total_objects=4,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("edge_approach", "fall_off", "land"),
            preferred_surface_keys=("dark_wood_floor", "painted_concrete_floor"),
            preferred_camera_keys=("cam_01", "cam_04", "cam_06"),
            motion_modes=("edge_roll", "fall_off", "boundary_slide"),
            speed_range=(0.6, 3.8),
            spin_range=(0.0, 10.0),
            angle_range_deg=(0.0, 30.0),
        ),
        ScenarioFamilySpec(
            key="F11",
            title="Table Roll-Off",
            description="A fixed-speed object rolls across a table and drops from the edge at different table heights.",
            family_slug="F11_table_rolloff",
            min_dynamic_objects=1,
            max_dynamic_objects=1,
            min_total_objects=6,
            max_total_objects=6,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("table_entry", "edge_drop", "land"),
            preferred_surface_keys=("residential_wood_floor", "studio_wood_floor", "dark_wood_floor"),
            preferred_camera_keys=("cam_09",),
            motion_modes=("table_rolloff",),
            speed_range=(1.25, 1.25),
            spin_range=(0.0, 0.0),
            angle_range_deg=(0.0, 24.0),
        ),
        ScenarioFamilySpec(
            key="F12",
            title="Inclined Ramp Control",
            description="A red wooden block is released from rest on a floor-supported ramp while only the incline angle changes.",
            family_slug="F12_inclined_ramp_control",
            min_dynamic_objects=4,
            max_dynamic_objects=4,
            min_total_objects=4,
            max_total_objects=4,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("release", "ramp_slide", "ramp_exit"),
            preferred_surface_keys=("painted_concrete_floor",),
            preferred_camera_keys=("cam_10",),
            motion_modes=("gravity_slide",),
            speed_range=(0.0, 0.0),
            spin_range=(0.0, 0.0),
            angle_range_deg=(8.0, 32.0),
        ),
        ScenarioFamilySpec(
            key="V2V_RAMP_PLATFORM",
            title="Incline-to-Platform Length Control",
            description=(
                "A wooden block leaves a fixed incline, crosses a horizontal platform, "
                "and then falls; only the horizontal platform length changes."
            ),
            family_slug="V2V_ramp_platform_length",
            min_dynamic_objects=1,
            max_dynamic_objects=1,
            min_total_objects=9,
            max_total_objects=9,
            supports_occlusion=False,
            supports_support_objects=True,
            target_event_types=("release", "ramp_exit", "platform_exit", "land"),
            preferred_surface_keys=("residential_wood_floor", "studio_wood_floor", "dark_wood_floor"),
            preferred_camera_keys=("cam_10",),
            motion_modes=("ramp_platform",),
            speed_range=(0.0, 0.0),
            spin_range=(0.0, 0.0),
            angle_range_deg=(25.0, 25.0),
            notes="Formal strict-CYCLES control family; horizontal platform length is the only scene variable.",
        ),
    ]
    return {family.key: family for family in families}


def _sample_range(rng: np.random.Generator, spec: RangeSpec) -> float:
    if spec.low == spec.high:
        return float(spec.low)
    return float(rng.uniform(spec.low, spec.high))


def _pick_material_key(
    rng: np.random.Generator,
    family: ObjectFamilySpec,
    material_keys_by_category: dict[str, list[str]],
) -> str:
    choices: list[str] = []
    for category in family.allowed_material_categories:
        choices.extend(material_keys_by_category.get(category, []))
    if not choices:
        raise KeyError(f"no materials available for categories={family.allowed_material_categories}")
    return str(rng.choice(choices))


def _color_from_material_key(material_key: str) -> tuple[float, float, float]:
    materials = build_material_catalog()
    material = materials[material_key]
    return material.base_color


def _family_sizes(rng: np.random.Generator, family: ObjectFamilySpec) -> dict[str, float]:
    return {name: _sample_range(rng, spec) for name, spec in family.size_ranges.items()}


def _scale_size_dict(size: dict[str, float], size_scale: float) -> dict[str, float]:
    if math.isclose(size_scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
        return dict(size)
    return {name: float(value) * size_scale for name, value in size.items()}


def _sample_motion_profile(rng: np.random.Generator, family: ScenarioFamilySpec) -> dict[str, float | str]:
    speed_min, speed_max = family.speed_range
    spin_min, spin_max = family.spin_range
    angle_min, angle_max = family.angle_range_deg
    modes = family.motion_modes or ("default",)
    return {
        "motion_mode": str(rng.choice(modes)),
        "speed": float(rng.uniform(speed_min, speed_max)) if speed_max > speed_min else float(speed_min),
        "spin": float(rng.uniform(spin_min, spin_max)) if spin_max > spin_min else float(spin_min),
        "angle_deg": float(rng.uniform(angle_min, angle_max)) if angle_max > angle_min else float(angle_min),
    }


def _sample_heading_deg(
    rng: np.random.Generator,
    *,
    axis: str,
    angle_deg: float,
    lateral_jitter: float,
) -> float:
    spread_deg = max(angle_deg, 8.0)
    if axis == "diag":
        base_deg = float(rng.choice([-42.0, -28.0, 28.0, 42.0]))
        jitter_deg = spread_deg * float(rng.uniform(0.30, 1.15))
        return base_deg + float(rng.choice([-1.0, 1.0])) * jitter_deg
    if axis == "y":
        base_deg = float(rng.choice([90.0, -90.0]))
        jitter_deg = spread_deg * float(rng.uniform(0.15, max(0.30, lateral_jitter + 0.20)))
        return base_deg + float(rng.choice([-1.0, 1.0])) * jitter_deg
    mode = str(rng.choice(["straight", "bias", "wide"]))
    if mode == "straight":
        offset_deg = float(rng.uniform(-0.45, 0.45)) * spread_deg
    elif mode == "bias":
        offset_deg = float(rng.choice([-1.0, 1.0])) * spread_deg * float(rng.uniform(0.35, 1.00))
    else:
        offset_deg = float(rng.choice([-1.0, 1.0])) * min(55.0, spread_deg * float(rng.uniform(0.85, 1.45)))
    return offset_deg


def _sample_angular_velocity(
    rng: np.random.Generator,
    *,
    spin: float,
) -> tuple[float, float, float]:
    if spin <= 0.0:
        return (0.0, 0.0, 0.0)
    dominant_axis = int(rng.integers(0, 3))
    dominant_sign = float(rng.choice([-1.0, 1.0]))
    dominant_mag = spin * float(rng.uniform(0.55, 1.00))
    secondary_scale = spin * float(rng.uniform(0.10, 0.42))
    components = [float(rng.uniform(-secondary_scale, secondary_scale)) for _ in range(3)]
    components[dominant_axis] = dominant_sign * dominant_mag
    if rng.random() < 0.45:
        coupled_axis = int((dominant_axis + int(rng.integers(1, 3))) % 3)
        coupled_mag = spin * float(rng.uniform(0.18, 0.65))
        components[coupled_axis] += float(rng.choice([-1.0, 1.0])) * coupled_mag
    return tuple(float(np.clip(value, -spin, spin)) for value in components)


def _sample_orientation_euler(
    rng: np.random.Generator,
    *,
    angle_deg: float,
) -> tuple[float, float, float]:
    if angle_deg <= 0.0:
        return (0.0, 0.0, 0.0)
    dominant_axis = int(rng.integers(0, 3))
    dominant_sign = float(rng.choice([-1.0, 1.0]))
    dominant_mag = angle_deg * float(rng.uniform(0.30, 1.00))
    secondary_mag = angle_deg * float(rng.uniform(0.12, 0.45))
    components = [float(rng.uniform(-secondary_mag, secondary_mag)) for _ in range(3)]
    components[dominant_axis] = dominant_sign * dominant_mag
    return tuple(components)


def _motion_vectors(
    rng: np.random.Generator,
    *,
    speed: float,
    spin: float,
    angle_deg: float,
    axis: str = "x",
    lateral_jitter: float = 0.35,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    heading_deg = _sample_heading_deg(rng, axis=axis, angle_deg=angle_deg, lateral_jitter=lateral_jitter)
    heading = math.radians(heading_deg)
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    vz = float(rng.uniform(-0.10, 0.06))
    linear_velocity = (vx, vy, vz)
    angular_velocity = _sample_angular_velocity(rng, spin=spin)
    orientation = _sample_orientation_euler(rng, angle_deg=angle_deg)
    return linear_velocity, angular_velocity, orientation


def _collision_vertical_extent(obj: ObjectInstanceSpec) -> float:
    """Return the world-space vertical half-extent of the PyBullet collision shape."""
    size = obj.size
    roll, pitch, _ = (math.radians(value) for value in obj.orientation_euler_deg)
    vertical_axis = (
        -math.sin(pitch),
        math.cos(pitch) * math.sin(roll),
        math.cos(pitch) * math.cos(roll),
    )
    ax, ay, az = (abs(value) for value in vertical_axis)

    if obj.shape in {"sphere", "ellipsoid"}:
        if "radius" in size:
            return float(size["radius"])
        return float(max(size["rx"], size["ry"], size["rz"]))
    if obj.shape in {"box", "rounded_box", "wedge"}:
        return float(ax * size["hx"] + ay * size["hy"] + az * size["hz"])
    if obj.shape in {"cylinder", "puck", "wheel_thick", "spool", "cone_frustum"}:
        radius = size.get(
            "radius",
            size.get("flange_radius", max(size.get("r_top", 0.0), size.get("r_base", 0.0))),
        )
        height = size.get("height", size.get("width"))
        radial_extent = float(radius) * math.sqrt(ax * ax + ay * ay)
        return radial_extent + 0.5 * float(height) * az
    if obj.shape == "capsule":
        return float(size["radius"]) + 0.5 * float(size["height"]) * az
    if obj.shape == "dumbbell":
        return float(size["weight_radius"]) + 0.5 * float(size["length"]) * az
    raise ValueError(f"unsupported collision shape for grounding: {obj.shape}")


def _place_dynamic_on_ground(
    obj: ObjectInstanceSpec,
    *,
    clearance_m: float = INITIAL_GROUND_CLEARANCE_M,
) -> ObjectInstanceSpec:
    """Place a dynamic rigid body just above the floor using its true extent."""
    if not obj.dynamic:
        raise ValueError(f"expected a dynamic object, got {obj.name}")
    return replace(
        obj,
        position=(
            obj.position[0],
            obj.position[1],
            _collision_vertical_extent(obj) + float(clearance_m),
        ),
    )


def _horizontal_clearance_radius(obj: ObjectInstanceSpec) -> float:
    """Conservative radius used to separate sampled floor objects at spawn."""
    size = obj.size
    if obj.shape in {"sphere", "ellipsoid"}:
        return float(size.get("radius", max(size.get("rx", 0.0), size.get("ry", 0.0))))
    if obj.shape in {"box", "rounded_box", "wedge"}:
        return float(math.hypot(size["hx"], size["hy"]))
    if obj.shape in {"cylinder", "puck", "wheel_thick", "spool", "cone_frustum"}:
        return float(
            max(
                size.get("radius", 0.0),
                size.get("flange_radius", 0.0),
                size.get("r_top", 0.0),
                size.get("r_base", 0.0),
                0.5 * size.get("height", size.get("width", 0.0)),
            )
        )
    if obj.shape == "capsule":
        return float(size["radius"] + 0.5 * size["height"])
    if obj.shape == "dumbbell":
        return float(size["weight_radius"] + 0.5 * size["length"])
    raise ValueError(f"unsupported collision shape for horizontal clearance: {obj.shape}")


def validate_blueprint_physics(blueprint: ScenarioBlueprint) -> None:
    if not math.isclose(float(blueprint.gravity), EARTH_GRAVITY, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{blueprint.sample_key}: all dynamic objects require gravity {EARTH_GRAVITY}, "
            f"got {blueprint.gravity}"
        )

    for obj in blueprint.objects:
        if obj.dynamic:
            if obj.mass <= 0.0:
                raise ValueError(f"{blueprint.sample_key}/{obj.name}: dynamic object mass must be positive")
            continue

        if obj.role == "support":
            raise ValueError(
                f"{blueprint.sample_key}/{obj.name}: visible support must be dynamic; "
                "use role='anchored_occluder' only for fixed scene geometry"
            )

        if obj.role.startswith("anchored_"):
            if obj.mass != 0.0:
                raise ValueError(f"{blueprint.sample_key}/{obj.name}: static object mass must be zero")
            continue

        expected_z = _collision_vertical_extent(obj)
        if obj.mass != 0.0:
            raise ValueError(f"{blueprint.sample_key}/{obj.name}: static object mass must be zero")
        if not math.isclose(float(obj.position[2]), expected_z, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError(
                f"{blueprint.sample_key}/{obj.name}: static object must touch the ground; "
                f"center_z={obj.position[2]:.8f}, required={expected_z:.8f}"
            )


def _set_blueprint_direction(blueprint: ScenarioBlueprint, direction_mode: str) -> ScenarioBlueprint:
    if direction_mode not in DIRECTION_MODES:
        raise ValueError(f"unsupported direction_mode={direction_mode}")
    if direction_mode == "vertical" and blueprint.family_key not in {"F5", "F8"}:
        raise ValueError(f"vertical direction is only supported for F5/F8, got {blueprint.family_key}")

    objects = blueprint.objects
    if direction_mode == "right_to_left":
        objects = tuple(
            replace(
                obj,
                position=(-obj.position[0], obj.position[1], obj.position[2]),
                orientation_euler_deg=(
                    obj.orientation_euler_deg[0],
                    -obj.orientation_euler_deg[1],
                    -obj.orientation_euler_deg[2],
                ),
                linear_velocity=(-obj.linear_velocity[0], obj.linear_velocity[1], obj.linear_velocity[2]),
                angular_velocity=(obj.angular_velocity[0], -obj.angular_velocity[1], -obj.angular_velocity[2]),
            )
            for obj in objects
        )
    elif direction_mode == "vertical":
        objects = tuple(
            replace(obj, linear_velocity=(0.0, 0.0, obj.linear_velocity[2])) if obj.dynamic else obj
            for obj in objects
        )

    direction_tag = f"direction_{direction_mode}"
    tags = blueprint.tags[:-1] + (direction_tag,) + blueprint.tags[-1:] if blueprint.tags else (direction_tag,)
    metadata = {**blueprint.metadata, "direction_mode": direction_mode}
    return replace(blueprint, objects=objects, tags=tags, metadata=metadata)


def _sample_object(
    rng: np.random.Generator,
    family: ObjectFamilySpec,
    name: str,
    material_keys_by_category: dict[str, list[str]],
    *,
    role: str | None = None,
    dynamic: bool | None = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    forced_material_key: str | None = None,
    size_scale: float = 1.0,
) -> ObjectInstanceSpec:
    material_key = forced_material_key or _pick_material_key(rng, family, material_keys_by_category)
    is_dynamic = family.dynamic_default if dynamic is None else dynamic
    obj = ObjectInstanceSpec(
        name=name,
        family_key=family.key,
        shape=family.shape,
        semantic_role=family.semantic_role,
        size=_scale_size_dict(_family_sizes(rng, family), size_scale),
        mass=_sample_range(rng, family.mass_range),
        friction=_sample_range(rng, family.friction_range),
        restitution=_sample_range(rng, family.restitution_range),
        linear_damping=_sample_range(rng, family.linear_damping_range),
        angular_damping=_sample_range(rng, family.angular_damping_range),
        material_key=material_key,
        color=_color_from_material_key(material_key),
        dynamic=is_dynamic,
        role=role or ("dynamic" if is_dynamic else family.semantic_role),
        position=position,
        orientation_euler_deg=orientation_euler_deg,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )
    if obj.dynamic:
        return obj

    grounded_position = (obj.position[0], obj.position[1], _collision_vertical_extent(obj))
    return replace(obj, mass=0.0, position=grounded_position)


def _support_volume_m3(obj: ObjectInstanceSpec) -> float:
    """Estimate a visible support volume for assigning a rigid-body mass."""
    if obj.shape == "wedge":
        return 4.0 * obj.size["hx"] * obj.size["hy"] * obj.size["hz"]
    if obj.shape in {"box", "rounded_box"}:
        return 8.0 * obj.size["hx"] * obj.size["hy"] * obj.size["hz"]
    raise ValueError(f"unsupported dynamic support shape: {obj.shape}")


def _dynamicize_support(obj: ObjectInstanceSpec) -> ObjectInstanceSpec:
    """Turn a visible support into a grounded dynamic body.

    The floor and walls remain environment geometry. A support listed in a
    blueprint, however, is a visible physical object and must have mass.
    """
    if obj.role != "support":
        raise ValueError(f"expected a support object, got role={obj.role!r}")
    density_by_material = {
        "wood": 500.0,
        "cardboard": 300.0,
        "concrete": 1600.0,
    }
    density = next(
        (value for prefix, value in density_by_material.items() if obj.material_key.startswith(prefix)),
        500.0,
    )
    mass = max(0.25, _support_volume_m3(obj) * density)
    grounded_z = _collision_vertical_extent(obj)
    return replace(
        obj,
        dynamic=True,
        role="dynamic_support",
        mass=float(mass),
        position=(obj.position[0], obj.position[1], grounded_z),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        linear_damping=max(float(obj.linear_damping), 0.02),
        angular_damping=max(float(obj.angular_damping), 0.04),
    )


def _sample_camera(rng: np.random.Generator, key: str) -> CameraSpec:
    return _sample_camera_with_distance_scale(rng, key, camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE)


def _sample_camera_with_distance_scale(
    rng: np.random.Generator,
    key: str,
    *,
    camera_distance_scale: float = DEFAULT_CAMERA_DISTANCE_SCALE,
) -> CameraSpec:
    camera = build_camera_catalog()[key]
    eye = tuple(float(base + rng.uniform(-jitter, jitter)) for base, jitter in zip(camera.eye, camera.jitter_eye_xyz))
    target = tuple(
        float(base + rng.uniform(-jitter, jitter))
        for base, jitter in zip(camera.target, camera.jitter_target_xyz)
    )
    if not math.isclose(camera_distance_scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
        eye = _scale_camera_eye_towards_target(eye, target, camera_distance_scale)
    yfov_deg = float(camera.yfov_deg + rng.uniform(-camera.jitter_fov_deg, camera.jitter_fov_deg))
    return replace(camera, eye=eye, target=target, yfov_deg=yfov_deg)


def _scale_camera_eye_towards_target(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    camera_distance_scale: float,
) -> tuple[float, float, float]:
    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    scaled_eye = target_arr + (eye_arr - target_arr) * float(camera_distance_scale)
    return tuple(float(value) for value in scaled_eye)


def _make_projection_camera(camera: CameraSpec, width: int, height: int) -> dict[str, np.ndarray | float]:
    eye = np.asarray(camera.eye, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    yfov = math.radians(float(camera.yfov_deg))
    aspect = float(width) / float(height)
    fx = 0.5 * width / (math.tan(yfov * 0.5) * aspect)
    fy = 0.5 * height / math.tan(yfov * 0.5)
    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": true_up,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5,
        "cy": height * 0.5,
    }


def _project_world_point(point_world: tuple[float, float, float] | np.ndarray, camera: dict[str, np.ndarray | float]) -> tuple[np.ndarray | None, float]:
    point = np.asarray(point_world, dtype=np.float64)
    delta = point - np.asarray(camera["eye"], dtype=np.float64)
    x_cam = float(delta @ np.asarray(camera["right"], dtype=np.float64))
    y_cam = float(delta @ np.asarray(camera["up"], dtype=np.float64))
    z_cam = float(delta @ np.asarray(camera["forward"], dtype=np.float64))
    if z_cam <= 1e-6:
        return None, z_cam
    u = float(camera["fx"]) * (x_cam / z_cam) + float(camera["cx"])
    v = float(camera["cy"]) - float(camera["fy"]) * (y_cam / z_cam)
    return np.asarray([u, v], dtype=np.float32), z_cam


def _visibility_score_for_object(
    obj: ObjectInstanceSpec,
    camera: CameraSpec,
    *,
    width: int = NOMINAL_RENDER_WIDTH,
    height: int = NOMINAL_RENDER_HEIGHT,
    horizon_s: float = 1.5,
) -> float:
    projection = _make_projection_camera(camera, width=width, height=height)
    sample_times = (0.0, 0.20, 0.45, 0.75, 1.10, horizon_s)
    margin_x = width * 0.06
    margin_y = height * 0.12
    center_x0 = width * 0.12
    center_x1 = width * 0.88
    center_y0 = height * 0.16
    center_y1 = height * 0.88
    pos0 = np.asarray(obj.position, dtype=np.float64)
    vel = np.asarray(obj.linear_velocity, dtype=np.float64)
    score = 0.0
    for idx, sample_t in enumerate(sample_times):
        point = pos0 + vel * float(sample_t)
        pixel, depth = _project_world_point(point, projection)
        if pixel is None or depth <= 0.0:
            score -= 6.0 if idx == 0 else 3.0
            continue
        u = float(pixel[0])
        v = float(pixel[1])
        in_frame = 0.0 <= u <= width and 0.0 <= v <= height
        in_safe = margin_x <= u <= (width - margin_x) and margin_y <= v <= (height - margin_y)
        in_center = center_x0 <= u <= center_x1 and center_y0 <= v <= center_y1
        if in_frame:
            score += 1.0
        else:
            score -= 4.0 if idx == 0 else 2.0
        if in_safe:
            score += 1.0
        elif idx <= 2:
            score -= 0.8
        if in_center:
            score += 0.6
    return score


def _select_best_camera_for_motion(
    rng: np.random.Generator,
    preferred_camera_keys: tuple[str, ...],
    dynamic_objects: tuple[ObjectInstanceSpec, ...],
    *,
    camera_distance_scale: float = 1.0,
) -> tuple[str, CameraSpec]:
    camera_keys = list(preferred_camera_keys)
    rng.shuffle(camera_keys)
    best_key = camera_keys[0]
    best_camera = _sample_camera_with_distance_scale(rng, best_key, camera_distance_scale=camera_distance_scale)
    best_score = sum(_visibility_score_for_object(obj, best_camera) for obj in dynamic_objects)
    for key in camera_keys[1:]:
        candidate_camera = _sample_camera_with_distance_scale(rng, key, camera_distance_scale=camera_distance_scale)
        candidate_score = sum(_visibility_score_for_object(obj, candidate_camera) for obj in dynamic_objects)
        if candidate_score > best_score:
            best_key = key
            best_camera = candidate_camera
            best_score = candidate_score
    return best_key, best_camera


def _camera_envelope_extent(obj: ObjectInstanceSpec) -> np.ndarray:
    """Return a conservative world-space half extent for camera fitting."""
    size = obj.size
    if obj.shape == "sphere":
        radius = float(size.get("radius", 0.10))
        return np.asarray([radius, radius, radius], dtype=np.float64)
    if obj.shape in {"box", "slab"}:
        return np.asarray(
            [
                float(size.get("hx", 0.10)),
                float(size.get("hy", 0.10)),
                float(size.get("hz", 0.10)),
            ],
            dtype=np.float64,
        )
    radius = float(size.get("radius", 0.10))
    height = float(size.get("height", 2.0 * radius))
    return np.asarray([radius, radius, max(radius, 0.5 * height)], dtype=np.float64)


def _fit_generic_camera_to_motion_envelope(blueprint: ScenarioBlueprint) -> ScenarioBlueprint:
    """Frame every generic scene's plausible 3-second motion envelope.

    Sampling only the first 1.5 seconds when selecting cameras allowed fast
    objects to leave the view late in a clip. F1-F10 use this conservative
    envelope fit after direction mirroring; dedicated control scenes retain
    their explicit cameras.
    """
    if not blueprint.family_key.startswith("F") or blueprint.family_key in {"F11", "F12"}:
        return blueprint
    # Most floor interactions lose their initial velocity before the clip
    # ends. A two-second envelope is wide enough for the observed rollout
    # while avoiding the severe over-zoom-out caused by linear extrapolation
    # over the entire three-second duration.
    duration_s = 2.0 + float(blueprint.pre_roll_s)
    sample_times = np.linspace(0.0, duration_s, 13, dtype=np.float64)
    low = np.full(3, np.inf, dtype=np.float64)
    high = np.full(3, -np.inf, dtype=np.float64)
    for obj in blueprint.objects:
        extent = _camera_envelope_extent(obj) + np.asarray([0.08, 0.08, 0.05])
        pos0 = np.asarray(obj.position, dtype=np.float64)
        if obj.dynamic:
            velocity = np.asarray(obj.linear_velocity, dtype=np.float64)
            points = pos0[None, :] + sample_times[:, None] * velocity[None, :]
            points[:, 2] = np.maximum(points[:, 2], extent[2])
        else:
            points = pos0[None, :]
        low = np.minimum(low, np.min(points - extent[None, :], axis=0))
        high = np.maximum(high, np.max(points + extent[None, :], axis=0))

    center = 0.5 * (low + high)
    camera = blueprint.camera
    forward = np.asarray(camera.target, dtype=np.float64) - np.asarray(camera.eye, dtype=np.float64)
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, np.asarray(camera.up, dtype=np.float64))
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    half_extents = 0.5 * (high - low)
    corners = np.asarray(
        [
            center + np.asarray([sx, sy, sz]) * half_extents
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    relative = corners - center[None, :]
    horizontal = np.abs(relative @ right)
    vertical = np.abs(relative @ true_up)
    depth = relative @ forward
    vertical_tan = math.tan(math.radians(float(camera.yfov_deg)) * 0.5)
    horizontal_tan = vertical_tan * (NOMINAL_RENDER_WIDTH / NOMINAL_RENDER_HEIGHT)
    safe_fraction = 0.80
    required_distance = max(
        float(np.max(horizontal / max(horizontal_tan * safe_fraction, 1e-6) - depth)),
        float(np.max(vertical / max(vertical_tan * safe_fraction, 1e-6) - depth)),
        float(np.max(-depth)) + 0.45,
        1.5,
    )
    fitted_camera = replace(
        camera,
        eye=tuple(float(value) for value in center - forward * required_distance),
        target=tuple(float(value) for value in center),
    )
    if blueprint.family_key == "F8":
        # The bouncer settles near the floor after its rebound. Shift the
        # camera and target together so the late floor motion stays visible
        # without changing the viewing angle or the simulated trajectory.
        frame_shift = np.asarray([0.0, 0.0, -0.32], dtype=np.float64)
        fitted_camera = replace(
            fitted_camera,
            eye=tuple(float(value) for value in np.asarray(fitted_camera.eye) + frame_shift),
            target=tuple(float(value) for value in np.asarray(fitted_camera.target) + frame_shift),
        )
    return replace(
        blueprint,
        camera=fitted_camera,
        metadata={
            **blueprint.metadata,
            "auto_motion_envelope_camera": True,
            "camera_motion_envelope_duration_s": round(duration_s, 4),
            "camera_motion_envelope_safe_fraction": safe_fraction,
        },
    )


def _material_keys_by_category() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, material in build_material_catalog().items():
        out.setdefault(material.category, []).append(key)
    return out


def _allowed_material_keys(family: ObjectFamilySpec, material_keys_by_category: dict[str, list[str]]) -> list[str]:
    keys: list[str] = []
    for category in family.allowed_material_categories:
        keys.extend(material_keys_by_category.get(category, []))
    return keys


def _make_f1(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F1"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    driver_key = str(rng.choice(["ball", "capsule_can", "flat_puck", "wheel", "spool", "bobbin", "drum_barrel", "roller_drum", "dumbbell"]))
    driver_family = object_families[driver_key]
    linear_velocity, angular_velocity, orientation = _motion_vectors(
        rng,
        speed=motion["speed"],
        spin=motion["spin"],
        angle_deg=motion["angle_deg"],
        axis="x",
    )
    driver = _sample_object(
        rng,
        driver_family,
        name="driver_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-2.0 + rng.uniform(-0.25, 0.10), rng.uniform(-0.55, 0.55), 0.18 + rng.uniform(-0.03, 0.10)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(orientation, driver_family.orientation_jitter_deg)),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )
    driver = _place_dynamic_on_ground(driver)
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (driver,))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"{driver_family.display_name} motion",
        description=f"Single-object motion with {driver_family.display_name.lower()} under varied material and camera settings.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.08, 0.30)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(driver,),
        tags=("diverse_object", "appearance_randomized", "single_motion", motion["motion_mode"]),
    )


def _make_f2(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F2"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    driver_family = object_families[str(rng.choice(["flat_puck", "ball", "capsule_can", "wheel", "bobbin", "drum_barrel", "roller_drum"]))]
    target_family = object_families[str(rng.choice(["crate_box", "upright_cylinder", "tall_box", "cone_frustum", "tool_case", "shipping_box"]))]
    driver_linear, driver_angular, driver_orientation = _motion_vectors(
        rng,
        speed=motion["speed"],
        spin=motion["spin"],
        angle_deg=motion["angle_deg"],
        axis=str(rng.choice(["x", "diag"])),
    )
    driver = _sample_object(
        rng,
        driver_family,
        name="driver_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-2.10 + rng.uniform(-0.25, 0.08), rng.uniform(-0.50, 0.50), 0.16 + rng.uniform(-0.04, 0.06)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(driver_orientation, driver_family.orientation_jitter_deg)),
        linear_velocity=driver_linear,
        angular_velocity=driver_angular,
    )
    driver = _place_dynamic_on_ground(driver)
    target = _sample_object(
        rng,
        target_family,
        name="target_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(rng.uniform(-0.10, 0.40), rng.uniform(-0.16, 0.18), 0.18 + rng.uniform(-0.02, 0.06)),
        orientation_euler_deg=(0.0, 0.0, rng.uniform(-12.0, 12.0)),
    )
    target = _place_dynamic_on_ground(target)
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (driver, target))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"{driver_family.display_name} impacts {target_family.display_name}",
        description="Parameterized two-body interaction with widened object and material diversity.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.10, 0.26)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(driver, target),
        tags=("diverse_object", "appearance_randomized", "two_body_contact", motion["motion_mode"]),
    )


def _make_f3(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F3"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    lead_family = object_families[str(rng.choice(["ball", "capsule_can", "wheel", "spool", "bobbin", "drum_barrel"]))]
    mid_family = object_families[str(rng.choice(["crate_box", "upright_cylinder", "tall_box", "tool_case", "shipping_box"]))]
    tail_family = object_families[str(rng.choice(["crate_box", "upright_cylinder", "cone_frustum", "tool_case", "shipping_box", "roller_drum"]))]
    lead_linear, lead_angular, lead_orientation = _motion_vectors(
        rng,
        speed=motion["speed"],
        spin=motion["spin"],
        angle_deg=motion["angle_deg"],
        axis="x",
    )
    lead = _sample_object(
        rng,
        lead_family,
        name="lead_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-2.20 + rng.uniform(-0.18, 0.06), rng.uniform(-0.18, 0.18), 0.17 + rng.uniform(-0.03, 0.04)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(lead_orientation, lead_family.orientation_jitter_deg)),
        linear_velocity=lead_linear,
        angular_velocity=lead_angular,
    )
    lead = _place_dynamic_on_ground(lead)
    mid = _sample_object(
        rng,
        mid_family,
        name="mid_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(rng.uniform(-0.30, -0.02), rng.uniform(-0.18, 0.18), 0.18),
        orientation_euler_deg=(0.0, 0.0, rng.uniform(-10.0, 10.0)),
    )
    mid = _place_dynamic_on_ground(mid)
    tail = _sample_object(
        rng,
        tail_family,
        name="tail_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(rng.uniform(0.58, 0.92), rng.uniform(-0.18, 0.18), 0.18),
        orientation_euler_deg=(0.0, 0.0, rng.uniform(-10.0, 10.0)),
    )
    tail = _place_dynamic_on_ground(tail)
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (lead, mid, tail))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title="Three-body chain reaction",
        description="Lead object drives a two-step contact chain with diversified geometry and appearance.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.18, 0.34)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(lead, mid, tail),
        tags=("diverse_object", "appearance_randomized", "chain_reaction", motion["motion_mode"]),
    )


def _make_f4(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F4"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    mover_family = object_families[str(rng.choice(["ball", "wheel", "capsule_can"]))]
    occluder_family = object_families["pillar_occluder"]
    mover_count = int(rng.choice([1, 2]))
    movers: list[ObjectInstanceSpec] = []
    shared_allowed = _allowed_material_keys(mover_family, material_keys)
    forced_mover_materials: list[str] = []
    if mover_count == 2 and len(shared_allowed) >= 2:
        forced_mover_materials = list(rng.choice(shared_allowed, size=2, replace=False))
    elif mover_count == 2 and shared_allowed:
        forced_mover_materials = [shared_allowed[0], shared_allowed[0]]
    for idx in range(mover_count):
        direction = 1.0 if idx == 0 else -1.0
        start_x = -2.70 if idx == 0 else 2.55
        mover_linear, mover_angular, mover_orientation = _motion_vectors(
            rng,
            speed=motion["speed"],
            spin=motion["spin"],
            angle_deg=motion["angle_deg"],
            axis="x",
        )
        mover = _sample_object(
                rng,
                mover_family,
                name=f"mover_{idx}",
                material_keys_by_category=material_keys,
                size_scale=size_scale,
                position=(start_x + rng.uniform(-0.18, 0.12), rng.uniform(0.50, 0.92), 0.17),
                orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
                linear_velocity=(direction * abs(mover_linear[0]), mover_linear[1], mover_linear[2]),
                angular_velocity=mover_angular,
                forced_material_key=forced_mover_materials[idx] if idx < len(forced_mover_materials) else None,
            )
        movers.append(_place_dynamic_on_ground(mover))
    occluders = [
        _sample_object(
            rng,
            occluder_family,
            name="occluder_left",
            material_keys_by_category=material_keys,
            size_scale=size_scale,
            dynamic=False,
            role="anchored_occluder",
            position=(-0.18, -0.05, 0.50),
            orientation_euler_deg=(0.0, 0.0, 0.0),
        )
    ]
    if mover_count == 1 or rng.random() > 0.4:
        occluders.append(
            _sample_object(
                rng,
                occluder_family,
                name="occluder_right",
                material_keys_by_category=material_keys,
                size_scale=size_scale,
                dynamic=False,
                role="anchored_occluder",
                position=(0.18, -0.05, 0.50),
                orientation_euler_deg=(0.0, 0.0, 0.0),
            )
        )
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, tuple(movers))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title="Occlusion and reappearance",
        description="Parameterized occlusion scene with variable mover count, occluder width and camera viewpoint.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.50, 0.82)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=tuple(movers + occluders),
        tags=("diverse_object", "appearance_randomized", "occlusion", motion["motion_mode"]),
    )


def _make_f5(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F5"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    dynamic_family = object_families[str(rng.choice(["ball", "upright_cylinder", "crate_box", "cone_frustum", "tool_case", "shipping_box", "drum_barrel"]))]
    support_family = object_families["crate_box"]
    support = _sample_object(
        rng,
        support_family,
        name="support_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        dynamic=False,
        role="support",
        position=(rng.uniform(-0.08, 0.20), 0.0, 0.16),
        forced_material_key=str(rng.choice(["wood_plywood", "wood_dark", "cardboard_kraft"])),
    )
    support = _dynamicize_support(support)
    drop_linear, drop_angular, drop_orientation = _motion_vectors(
        rng,
        speed=motion["speed"],
        spin=motion["spin"],
        angle_deg=motion["angle_deg"],
        axis="x",
    )
    dynamic = _sample_object(
        rng,
        dynamic_family,
        name="drop_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(rng.uniform(-0.28, 0.00), 0.0, rng.uniform(0.64, 1.18)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(drop_orientation, dynamic_family.orientation_jitter_deg)),
        linear_velocity=drop_linear,
        angular_velocity=drop_angular,
    )
    dynamic = replace(
        dynamic,
        position=(
            dynamic.position[0],
            dynamic.position[1],
            max(
                dynamic.position[2],
                support.position[2]
                + _collision_vertical_extent(support)
                + _collision_vertical_extent(dynamic)
                + INITIAL_GROUND_CLEARANCE_M,
            ),
        ),
    )
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (dynamic,))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"{dynamic_family.display_name} support-loss",
        description="Support/drop scene with variable support geometry and dynamic object family.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.02, 0.12)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(dynamic, support),
        tags=("diverse_object", "appearance_randomized", "support_drop", motion["motion_mode"]),
    )


def _make_f6(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F6"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    mover_family = object_families[str(rng.choice(["ball", "capsule_can", "crate_box", "shipping_box", "wheel"]))]
    support_family = object_families[str(rng.choice(["slab_box", "wedge_ramp"]))]
    support = _sample_object(
        rng,
        support_family,
        name="ramp_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        dynamic=False,
        role="support",
        position=(0.0, 0.0, 0.12),
        forced_material_key=str(rng.choice(["wood_plywood", "wood_dark", "concrete_painted"])),
        orientation_euler_deg=(0.0, rng.uniform(0.0, 14.0), rng.uniform(-6.0, 6.0)),
    )
    support = _dynamicize_support(support)
    mover_linear, mover_angular, mover_orientation = _motion_vectors(
        rng,
        speed=motion["speed"],
        spin=motion["spin"],
        angle_deg=motion["angle_deg"],
        axis="x",
    )
    mover = _sample_object(
        rng,
        mover_family,
        name="slider_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-1.6 + rng.uniform(-0.2, 0.1), rng.uniform(-0.08, 0.16), 0.55 + rng.uniform(0.0, 0.22)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
        linear_velocity=(abs(mover_linear[0]), mover_linear[1], mover_linear[2]),
        angular_velocity=mover_angular,
    )
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (mover,))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"{mover_family.display_name} ramp slide",
        description="Inclined-plane motion with visible support geometry and more natural indoor view.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.08, 0.20)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(mover, support),
        tags=("diverse_object", "appearance_randomized", "ramp_motion", motion["motion_mode"]),
    )


def _make_f7(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F7"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    mover_family = object_families[str(rng.choice(["wheel", "spool", "drum_barrel", "capsule_can", "dumbbell"]))]
    mover_linear, mover_angular, mover_orientation = _motion_vectors(
        rng,
        speed=motion["speed"],
        spin=motion["spin"],
        angle_deg=motion["angle_deg"],
        axis="x",
    )
    mover = _sample_object(
        rng,
        mover_family,
        name="spinner_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-1.3 + rng.uniform(-0.15, 0.15), rng.uniform(-0.10, 0.10), 0.22 + rng.uniform(-0.04, 0.06)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
        linear_velocity=(mover_linear[0] * 0.55, mover_linear[1], mover_linear[2]),
        angular_velocity=(mover_angular[0], mover_angular[1] * 1.2, mover_angular[2]),
    )
    mover = _place_dynamic_on_ground(mover)
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (mover,))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"{mover_family.display_name} spin dominant",
        description="Rotation-heavy motion where angular momentum dominates translation.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.04, 0.16)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(mover,),
        tags=("diverse_object", "appearance_randomized", "spin_dominant", motion["motion_mode"]),
    )


def _make_f8(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F8"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    mover_family = object_families[str(rng.choice(["ball", "capsule_can", "cone_frustum", "upright_cylinder"]))]
    mover_linear, mover_angular, mover_orientation = _motion_vectors(
        rng,
        speed=motion["speed"],
        spin=motion["spin"],
        angle_deg=motion["angle_deg"],
        axis="x",
    )
    mover = _sample_object(
        rng,
        mover_family,
        name="bouncer_0",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-1.0 + rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 1.05 + rng.uniform(0.0, 0.30)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
        linear_velocity=(abs(mover_linear[0]), mover_linear[1], -abs(mover_linear[0]) * 0.08),
        angular_velocity=mover_angular,
    )
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (mover,))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"{mover_family.display_name} heavy bounce",
        description="Repeated rebound and settling with a visually clean indoor backdrop.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.02, 0.10)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(mover,),
        tags=("diverse_object", "appearance_randomized", "bounce_heavy", motion["motion_mode"]),
    )


def _make_f9(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F9"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    mover_a_family = object_families[str(rng.choice(["ball", "crate_box", "capsule_can", "wheel", "spool"]))]
    mover_b_family = object_families[str(rng.choice(["shipping_box", "tool_case", "upright_cylinder", "cone_frustum"]))]
    mover_a_lin, mover_a_ang, mover_a_ori = _motion_vectors(rng, speed=motion["speed"], spin=motion["spin"], angle_deg=motion["angle_deg"], axis="x")
    mover_b_lin, mover_b_ang, mover_b_ori = _motion_vectors(rng, speed=motion["speed"] * 0.85, spin=motion["spin"], angle_deg=motion["angle_deg"], axis="diag")
    mover_a = _sample_object(
        rng,
        mover_a_family,
        name="clutter_a",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-1.6 + rng.uniform(-0.2, 0.1), rng.uniform(-0.25, 0.25), 0.18 + rng.uniform(-0.05, 0.08)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_a_ori, mover_a_family.orientation_jitter_deg)),
        linear_velocity=mover_a_lin,
        angular_velocity=mover_a_ang,
    )
    mover_a = _place_dynamic_on_ground(mover_a)
    mover_b = _sample_object(
        rng,
        mover_b_family,
        name="clutter_b",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(rng.uniform(-0.3, 0.5), rng.uniform(-0.2, 0.2), 0.18 + rng.uniform(-0.05, 0.08)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_b_ori, mover_b_family.orientation_jitter_deg)),
        linear_velocity=mover_b_lin,
        angular_velocity=mover_b_ang,
    )
    mover_b = _place_dynamic_on_ground(mover_b)
    support = _sample_object(
        rng,
        object_families[str(rng.choice(["slab_box", "stack_box"]))],
        name="clutter_support",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        dynamic=False,
        role="support",
        position=(rng.uniform(-0.20, 0.20), rng.uniform(-0.08, 0.08), 0.12),
        forced_material_key=str(rng.choice(["wood_plywood", "cardboard_kraft", "concrete_painted"])),
    )
    support = _dynamicize_support(support)
    side = 1.0 if mover_b.position[1] >= support.position[1] else -1.0
    mover_b = replace(
        mover_b,
        position=(
            mover_b.position[0],
            support.position[1]
            + side * (
                _horizontal_clearance_radius(support)
                + _horizontal_clearance_radius(mover_b)
                + 0.08
            ),
            mover_b.position[2],
        ),
    )
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (mover_a, mover_b))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title="Clutter interaction",
        description="Room-like local clutter with multiple interacting objects and partial occlusion.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.12, 0.30)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(mover_a, mover_b, support),
        tags=("diverse_object", "appearance_randomized", "clutter_interaction", motion["motion_mode"]),
    )


def _make_f10(rng: np.random.Generator, sample_key: str, size_scale: float = 1.0) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F10"]
    material_keys = _material_keys_by_category()
    motion = _sample_motion_profile(rng, family)
    mover_family = object_families[str(rng.choice(["ball", "wheel", "drum_barrel", "shipping_box", "tool_case"]))]
    mover_lin, mover_ang, mover_ori = _motion_vectors(rng, speed=motion["speed"], spin=motion["spin"], angle_deg=motion["angle_deg"], axis="x")
    mover = _sample_object(
        rng,
        mover_family,
        name="edge_mover",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        position=(-1.5 + rng.uniform(-0.2, 0.1), rng.uniform(-0.10, 0.10), 0.18 + rng.uniform(-0.04, 0.06)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_ori, mover_family.orientation_jitter_deg)),
        linear_velocity=mover_lin,
        angular_velocity=mover_ang,
    )
    mover = _place_dynamic_on_ground(mover)
    support = _sample_object(
        rng,
        object_families[str(rng.choice(["slab_box", "stack_box", "platform_block"]))],
        name="edge_support",
        material_keys_by_category=material_keys,
        size_scale=size_scale,
        dynamic=False,
        role="support",
        position=(rng.uniform(-0.15, 0.15), 0.0, 0.10),
        forced_material_key=str(rng.choice(["wood_dark", "wood_plywood", "cardboard_kraft"])),
    )
    support = _dynamicize_support(support)
    camera_key, camera = _select_best_camera_for_motion(rng, family.preferred_camera_keys, (mover,))
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title="Edge and boundary",
        description="Motion near an edge or boundary with fall-off potential.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=float(rng.uniform(0.08, 0.22)),
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(mover, support),
        tags=("diverse_object", "appearance_randomized", "edge_boundary", motion["motion_mode"]),
    )


def _make_f11(
    rng: np.random.Generator,
    sample_key: str,
    size_scale: float = 1.0,
    *,
    table_height_m: float | None = None,
    initial_speed_mps: float | None = None,
    travel_angle_deg: float | None = None,
) -> ScenarioBlueprint:
    family = build_scenario_family_catalog()["F11"]
    materials = build_material_catalog()

    table_height = float(table_height_m if table_height_m is not None else rng.uniform(0.35, 1.25))
    table_height = float(np.clip(table_height, 0.30, 1.40))
    speed = float(initial_speed_mps if initial_speed_mps is not None else rng.uniform(*family.speed_range))
    speed = float(np.clip(speed, 0.65, 2.4))
    if travel_angle_deg is None:
        travel_angle = F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG
    else:
        travel_angle = float(np.clip(travel_angle_deg, -60.0, 180.0))
    travel_heading = math.radians(travel_angle)
    travel_direction = (math.cos(travel_heading), math.sin(travel_heading))

    table_top_thickness = 0.05 * size_scale
    table_top_half = 0.60 * size_scale
    table_depth_half = 0.36 * size_scale
    leg_half = 0.035 * size_scale
    leg_height = max(0.20 * size_scale, table_height - table_top_thickness)
    top_center_z = leg_height + 0.5 * table_top_thickness
    leg_center_z = 0.5 * leg_height
    ball_radius = 0.14 * size_scale
    ball_start_distance = 0.36 * size_scale
    ball_start_x = -ball_start_distance * travel_direction[0]
    ball_start_y = -ball_start_distance * travel_direction[1]
    ball_velocity = (speed * travel_direction[0], speed * travel_direction[1], 0.0)
    ball_angular_velocity = (
        ball_velocity[1] / max(ball_radius, 1e-6),
        -ball_velocity[0] / max(ball_radius, 1e-6),
        0.0,
    )
    table_top_material_key = str(rng.choice(["wood_plywood", "wood_dark", "concrete_painted"]))

    table_top = ObjectInstanceSpec(
        name="table_top_0",
        family_key="table_top",
        shape="box",
        semantic_role="support",
        size={"hx": table_top_half, "hy": table_depth_half, "hz": 0.5 * table_top_thickness},
        mass=0.0,
        friction=0.88,
        restitution=0.01,
        linear_damping=0.0,
        angular_damping=0.0,
        material_key=table_top_material_key,
        color=materials[table_top_material_key].base_color,
        dynamic=False,
        role="anchored_fixture",
        position=(0.0, 0.0, top_center_z),
        orientation_euler_deg=(0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )

    leg_material = str(rng.choice(["wood_dark", "painted_metal_teal", "concrete_painted"]))
    leg_color = materials[leg_material].base_color
    leg_dx = table_top_half - 0.08 * size_scale
    leg_dy = table_depth_half - 0.08 * size_scale
    leg_positions = [
        (-leg_dx, -leg_dy, leg_center_z),
        (leg_dx, -leg_dy, leg_center_z),
        (-leg_dx, leg_dy, leg_center_z),
        (leg_dx, leg_dy, leg_center_z),
    ]
    legs = [
        ObjectInstanceSpec(
            name=f"table_leg_{index}",
            family_key="table_leg",
            shape="box",
            semantic_role="support",
            size={"hx": leg_half, "hy": leg_half, "hz": 0.5 * leg_height},
            mass=0.0,
            friction=0.90,
            restitution=0.01,
            linear_damping=0.0,
            angular_damping=0.0,
            material_key=leg_material,
            color=leg_color,
            dynamic=False,
            role="anchored_fixture",
            position=position,
            orientation_euler_deg=(0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )
        for index, position in enumerate(leg_positions)
    ]

    mover = ObjectInstanceSpec(
        name="roller_0",
        family_key="ball",
        shape="sphere",
        semantic_role="rolling_dynamic",
        size={"radius": ball_radius},
        mass=2.50,
        friction=0.40,
        restitution=1.0,
        linear_damping=0.0,
        angular_damping=0.0,
        material_key="rubber_red",
        color=materials["rubber_red"].base_color,
        dynamic=True,
        role="dynamic",
        position=(ball_start_x, ball_start_y, table_height + ball_radius),
        orientation_euler_deg=(0.0, 0.0, 0.0),
        linear_velocity=ball_velocity,
        angular_velocity=ball_angular_velocity,
        metadata={
            "appearance_group": "f11_table_rolloff_red_rubber_ball_v1",
            "appearance_contract": "same_ball_material_and_texture_mapping_for_all_f11_controls",
        },
    )

    camera_key = "cam_09"
    camera = build_camera_catalog()[camera_key]

    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"Table roll-off at {table_height:.2f}m",
        description="A fixed-speed ball rolls across a table of varying height and drops from the edge under gravity.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=0.04,
        camera_key=camera_key,
        surface_key=str(rng.choice(family.preferred_surface_keys)),
        lighting_key=build_camera_catalog()[camera_key].hdri_key,
        camera=camera,
        objects=(mover, table_top, *legs),
        tags=("table_rolloff", "fixed_speed", f"table_height_{table_height:.2f}", "gravity_drop"),
        metadata={
            "table_height_m": round(table_height, 5),
            "initial_speed_mps": round(speed, 5),
            "travel_angle_deg": round(travel_angle, 5),
            "travel_direction_xy": (
                round(float(travel_direction[0]), 5),
                round(float(travel_direction[1]), 5),
            ),
            "floor_restitution": 0.62,
            "table_top_thickness_m": round(table_top_thickness, 5),
            "table_top_half_width_m": round(table_top_half, 5),
            "table_top_half_depth_m": round(table_depth_half, 5),
        },
    )


def _make_f12(
    rng: np.random.Generator,
    sample_key: str,
    size_scale: float = 1.0,
    *,
    ramp_angle_deg: float | None = None,
    ramp_length_m: float | None = None,
    ramp_support_height_m: float | None = None,
) -> ScenarioBlueprint:
    """Build a gravity-only inclined-ramp control scene.

    The block, board, and both risers are dynamic PyBullet bodies.  The board
    rests on the floor at its lower edge and on two grounded risers at the high
    end, so the visible ramp is supported by contact rather than a fixed or
    suspended body.  A fixed support height can be supplied for a length
    control group; in that mode the slope angle is solved from the support
    contact instead of being held fixed.
    """
    del rng  # The control geometry intentionally has no appearance randomness.
    family = build_scenario_family_catalog()["F12"]
    materials = build_material_catalog()

    ramp_length = float(ramp_length_m if ramp_length_m is not None else 1.80)
    ramp_length = float(np.clip(ramp_length, 0.60, 2.40))

    support_half_x = 0.015 * size_scale
    support_local_fraction = 0.63 / 0.90
    fixed_support_height = (
        None
        if ramp_support_height_m is None
        else float(ramp_support_height_m) * size_scale
    )
    if fixed_support_height is not None:
        if fixed_support_height <= 0.10 * size_scale:
            raise ValueError(
                "ramp_support_height_m must be above the minimum riser height"
            )

        def support_top_z_for_angle(angle_deg: float) -> float:
            theta = math.radians(angle_deg)
            half_length = 0.50 * ramp_length * size_scale
            board_thickness = 0.035 * size_scale
            board_center = board_thickness * math.cos(theta) + half_length * math.sin(theta)
            support_local_x = -support_local_fraction * half_length
            support_footprint_drop = support_half_x * math.tan(theta)
            return (
                board_center
                - support_local_x * math.sin(theta)
                - board_thickness * math.cos(theta)
                - support_footprint_drop
            )

        low_angle, high_angle = 8.0, 42.0
        low_height = support_top_z_for_angle(low_angle)
        high_height = support_top_z_for_angle(high_angle)
        if not low_height <= fixed_support_height <= high_height:
            raise ValueError(
                "ramp_support_height_m is outside the supported angle range: "
                f"{low_height:.4f}..{high_height:.4f} m"
            )
        for _ in range(64):
            middle_angle = 0.5 * (low_angle + high_angle)
            if support_top_z_for_angle(middle_angle) < fixed_support_height:
                low_angle = middle_angle
            else:
                high_angle = middle_angle
        ramp_angle = 0.5 * (low_angle + high_angle)
    else:
        ramp_angle = float(ramp_angle_deg if ramp_angle_deg is not None else 20.0)
        ramp_angle = float(np.clip(ramp_angle, 8.0, 42.0))
    theta = math.radians(ramp_angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    board_half_length = 0.50 * ramp_length * size_scale
    board_half_width = 0.34 * size_scale
    board_half_thickness = 0.035 * size_scale
    block_half_x = 0.20 * size_scale
    block_half_y = 0.16 * size_scale
    block_half_z = 0.14 * size_scale

    # The lower underside touches the ground at x=+half_length.  The board and
    # risers remain dynamic, but start in a mechanically supported arrangement.
    board_center_z = board_half_thickness * cos_theta + board_half_length * sin_theta
    board = ObjectInstanceSpec(
        name="incline_board_0",
        family_key="incline_board",
        shape="box",
        semantic_role="ramp_support",
        size={"hx": board_half_length, "hy": board_half_width, "hz": board_half_thickness},
        mass=48.0,
        friction=0.88,
        restitution=0.01,
        linear_damping=0.05,
        angular_damping=0.12,
        material_key="painted_metal_yellow",
        color=materials["painted_metal_yellow"].base_color,
        dynamic=True,
        role="dynamic_ramp",
        position=(0.0, 0.0, board_center_z),
        orientation_euler_deg=(0.0, ramp_angle, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )

    support_local_x = -support_local_fraction * board_half_length
    support_half_y = 0.10 * size_scale
    support_clearance = 0.0
    # The board underside falls across the horizontal footprint of a riser.
    # Clear the lowest point of that footprint, not only its center, so the
    # dynamic riser starts below the board rather than cutting through it.
    support_footprint_drop = support_half_x * math.tan(theta)
    support_top_z = (
        board_center_z
        - support_local_x * sin_theta
        - board_half_thickness * cos_theta
        - support_footprint_drop
        - support_clearance
    )
    support_height = (
        fixed_support_height
        if fixed_support_height is not None
        else max(0.10 * size_scale, support_top_z)
    )
    support_x = support_local_x * cos_theta - board_half_thickness * sin_theta
    risers = tuple(
        ObjectInstanceSpec(
            name=f"incline_riser_{index}",
            family_key="incline_riser",
            shape="box",
            semantic_role="ramp_support",
            size={"hx": support_half_x, "hy": support_half_y, "hz": 0.5 * support_height},
            mass=120.0,
            friction=0.92,
            restitution=0.01,
            linear_damping=0.10,
            angular_damping=0.16,
            material_key="painted_metal_yellow",
            color=materials["painted_metal_yellow"].base_color,
            dynamic=True,
            role="dynamic_support",
            position=(support_x, side * 0.30 * size_scale, 0.5 * support_height),
            orientation_euler_deg=(0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
        )
        for index, side in enumerate((-1.0, 1.0))
    )

    block_local_x = -(0.50 / 0.90) * board_half_length
    block_local_z = board_half_thickness + block_half_z
    block = ObjectInstanceSpec(
        name="block_0",
        family_key="wood_block",
        shape="box",
        semantic_role="sliding_dynamic",
        size={"hx": block_half_x, "hy": block_half_y, "hz": block_half_z},
        mass=2.50,
        friction=0.12,
        restitution=0.08,
        linear_damping=0.02,
        angular_damping=0.04,
        material_key="wood_red",
        color=materials["wood_red"].base_color,
        dynamic=True,
        role="dynamic",
        position=(
            block_local_x * cos_theta + block_local_z * sin_theta,
            0.0,
            board_center_z - block_local_x * sin_theta + block_local_z * cos_theta,
        ),
        orientation_euler_deg=(0.0, ramp_angle, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    )

    camera_key = "cam_10"
    camera = build_camera_catalog()[camera_key]
    return ScenarioBlueprint(
        family_key=family.key,
        sample_key=sample_key,
        title=f"Red wooden block release on a {ramp_angle:.0f} degree incline",
        description="A red wooden block is released from rest on a floor-supported ramp; only the ramp incline changes.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=0.04,
        camera_key=camera_key,
        surface_key="painted_concrete_floor",
        lighting_key=camera.hdri_key,
        camera=camera,
        objects=(block, board, *risers),
        tags=("inclined_ramp_control", "gravity_release", f"ramp_angle_{ramp_angle:.0f}"),
        metadata={
            "ramp_angle_deg": round(ramp_angle, 5),
            "ramp_length_m": round(2.0 * board_half_length, 5),
            "ramp_width_m": round(2.0 * board_half_width, 5),
            "ramp_thickness_m": round(2.0 * board_half_thickness, 5),
            "released_from_rest": True,
            "initial_speed_mps": 0.0,
            "floor_restitution": 0.02,
            "support_mode": "dynamic_floor_supported_risers",
            "ramp_support_height_m": round(float(support_height), 5),
            "controlled_variable": "ramp_angle_deg",
        },
    )


FAMILY_GENERATORS = {
    "F1": _make_f1,
    "F2": _make_f2,
    "F3": _make_f3,
    "F4": _make_f4,
    "F5": _make_f5,
    "F6": _make_f6,
    "F7": _make_f7,
    "F8": _make_f8,
    "F9": _make_f9,
    "F10": _make_f10,
    "F11": _make_f11,
    "F12": _make_f12,
}


def generate_scenario_blueprint(
    family_key: str,
    sample_key: str,
    seed: int,
    direction_mode: str = "auto",
    size_scale: float = 1.0,
    camera_distance_scale: float = DEFAULT_CAMERA_DISTANCE_SCALE,
    table_height_m: float | None = None,
    initial_speed_mps: float | None = None,
    travel_angle_deg: float | None = None,
    ramp_angle_deg: float | None = None,
    ramp_length_m: float | None = None,
    ramp_support_height_m: float | None = None,
) -> ScenarioBlueprint:
    if family_key not in FAMILY_GENERATORS:
        raise KeyError(f"unsupported family_key={family_key}")
    if size_scale <= 0.0:
        raise ValueError(f"size_scale must be positive, got {size_scale}")
    if camera_distance_scale <= 0.0:
        raise ValueError(f"camera_distance_scale must be positive, got {camera_distance_scale}")
    if direction_mode == "auto":
        direction_mode = "left_to_right" if seed % 2 == 0 else "right_to_left"
    rng = np.random.default_rng(seed)
    if family_key == "F11":
        blueprint = FAMILY_GENERATORS[family_key](
            rng,
            sample_key,
            size_scale,
            table_height_m=table_height_m,
            initial_speed_mps=initial_speed_mps,
            travel_angle_deg=travel_angle_deg,
        )
    elif family_key == "F12":
        blueprint = FAMILY_GENERATORS[family_key](
            rng,
            sample_key,
            size_scale,
            ramp_angle_deg=ramp_angle_deg,
            ramp_length_m=ramp_length_m,
            ramp_support_height_m=ramp_support_height_m,
        )
    else:
        blueprint = FAMILY_GENERATORS[family_key](rng, sample_key, size_scale)
    blueprint = replace(
        blueprint,
        metadata={
            **blueprint.metadata,
            "size_scale": float(size_scale),
            "camera_distance_scale": float(camera_distance_scale),
        },
    )
    if not math.isclose(camera_distance_scale, 1.0, rel_tol=0.0, abs_tol=1e-9):
        adjusted_camera = replace(
            blueprint.camera,
            eye=_scale_camera_eye_towards_target(
                blueprint.camera.eye,
                blueprint.camera.target,
                camera_distance_scale,
            ),
        )
        blueprint = replace(blueprint, camera=adjusted_camera)
    blueprint = _set_blueprint_direction(blueprint, direction_mode)
    blueprint = _fit_generic_camera_to_motion_envelope(blueprint)
    if family_key in {"F11", "F12"}:
        color_qa = validate_color_separation(blueprint)
        blueprint = replace(
            blueprint,
            metadata={
                **blueprint.metadata,
                "initialization_contract": build_contact_contract(blueprint),
                "color_separation_qa": color_qa,
            },
        )
    validate_blueprint_physics(blueprint)
    return blueprint


def preview_diversity_report(
    num_samples_per_family: int = 6,
    *,
    size_scale: float = 1.0,
    camera_distance_scale: float = DEFAULT_CAMERA_DISTANCE_SCALE,
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for family_key in sorted(FAMILY_GENERATORS):
        object_keys: set[str] = set()
        material_keys: set[str] = set()
        camera_keys: set[str] = set()
        shape_keys: set[str] = set()
        for idx in range(num_samples_per_family):
            blueprint = generate_scenario_blueprint(
                family_key=family_key,
                sample_key=f"{family_key.lower()}_preview_{idx:03d}",
                seed=20260705 + idx * 1009,
                size_scale=size_scale,
                camera_distance_scale=camera_distance_scale,
            )
            camera_keys.add(blueprint.camera_key)
            for obj in blueprint.objects:
                object_keys.add(obj.family_key)
                material_keys.add(obj.material_key)
                shape_keys.add(obj.shape)
        report[family_key] = {
            "unique_object_families": sorted(object_keys),
            "unique_materials": sorted(material_keys),
            "unique_shapes": sorted(shape_keys),
            "unique_cameras": sorted(camera_keys),
        }
    return report
