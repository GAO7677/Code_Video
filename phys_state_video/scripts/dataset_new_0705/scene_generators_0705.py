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
from .object_catalog_0705 import build_object_family_catalog


EARTH_GRAVITY = 9.81
NOMINAL_RENDER_WIDTH = 1280
NOMINAL_RENDER_HEIGHT = 720


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
            angle_range_deg=(0.0, 24.0),
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
) -> ObjectInstanceSpec:
    material_key = forced_material_key or _pick_material_key(rng, family, material_keys_by_category)
    return ObjectInstanceSpec(
        name=name,
        family_key=family.key,
        shape=family.shape,
        semantic_role=family.semantic_role,
        size=_family_sizes(rng, family),
        mass=_sample_range(rng, family.mass_range),
        friction=_sample_range(rng, family.friction_range),
        restitution=_sample_range(rng, family.restitution_range),
        linear_damping=_sample_range(rng, family.linear_damping_range),
        angular_damping=_sample_range(rng, family.angular_damping_range),
        material_key=material_key,
        color=_color_from_material_key(material_key),
        dynamic=family.dynamic_default if dynamic is None else dynamic,
        role=role or ("dynamic" if (family.dynamic_default if dynamic is None else dynamic) else family.semantic_role),
        position=position,
        orientation_euler_deg=orientation_euler_deg,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )


def _sample_camera(rng: np.random.Generator, key: str) -> CameraSpec:
    camera = build_camera_catalog()[key]
    eye = tuple(float(base + rng.uniform(-jitter, jitter)) for base, jitter in zip(camera.eye, camera.jitter_eye_xyz))
    target = tuple(
        float(base + rng.uniform(-jitter, jitter))
        for base, jitter in zip(camera.target, camera.jitter_target_xyz)
    )
    yfov_deg = float(camera.yfov_deg + rng.uniform(-camera.jitter_fov_deg, camera.jitter_fov_deg))
    return replace(camera, eye=eye, target=target, yfov_deg=yfov_deg)


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
) -> tuple[str, CameraSpec]:
    camera_keys = list(preferred_camera_keys)
    rng.shuffle(camera_keys)
    best_key = camera_keys[0]
    best_camera = _sample_camera(rng, best_key)
    best_score = sum(_visibility_score_for_object(obj, best_camera) for obj in dynamic_objects)
    for key in camera_keys[1:]:
        candidate_camera = _sample_camera(rng, key)
        candidate_score = sum(_visibility_score_for_object(obj, candidate_camera) for obj in dynamic_objects)
        if candidate_score > best_score:
            best_key = key
            best_camera = candidate_camera
            best_score = candidate_score
    return best_key, best_camera


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


def _make_f1(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        position=(-2.0 + rng.uniform(-0.25, 0.10), rng.uniform(-0.55, 0.55), 0.18 + rng.uniform(-0.03, 0.10)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(orientation, driver_family.orientation_jitter_deg)),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )
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


def _make_f2(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        position=(-2.10 + rng.uniform(-0.25, 0.08), rng.uniform(-0.50, 0.50), 0.16 + rng.uniform(-0.04, 0.06)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(driver_orientation, driver_family.orientation_jitter_deg)),
        linear_velocity=driver_linear,
        angular_velocity=driver_angular,
    )
    target = _sample_object(
        rng,
        target_family,
        name="target_0",
        material_keys_by_category=material_keys,
        position=(rng.uniform(-0.10, 0.40), rng.uniform(-0.16, 0.18), 0.18 + rng.uniform(-0.02, 0.06)),
        orientation_euler_deg=(0.0, 0.0, rng.uniform(-12.0, 12.0)),
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(driver, target),
        tags=("diverse_object", "appearance_randomized", "two_body_contact", motion["motion_mode"]),
    )


def _make_f3(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        position=(-2.20 + rng.uniform(-0.18, 0.06), rng.uniform(-0.18, 0.18), 0.17 + rng.uniform(-0.03, 0.04)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(lead_orientation, lead_family.orientation_jitter_deg)),
        linear_velocity=lead_linear,
        angular_velocity=lead_angular,
    )
    mid = _sample_object(
        rng,
        mid_family,
        name="mid_0",
        material_keys_by_category=material_keys,
        position=(rng.uniform(-0.30, -0.02), rng.uniform(-0.18, 0.18), 0.18),
        orientation_euler_deg=(0.0, 0.0, rng.uniform(-10.0, 10.0)),
    )
    tail = _sample_object(
        rng,
        tail_family,
        name="tail_0",
        material_keys_by_category=material_keys,
        position=(rng.uniform(0.58, 0.92), rng.uniform(-0.18, 0.18), 0.18),
        orientation_euler_deg=(0.0, 0.0, rng.uniform(-10.0, 10.0)),
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(lead, mid, tail),
        tags=("diverse_object", "appearance_randomized", "chain_reaction", motion["motion_mode"]),
    )


def _make_f4(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        movers.append(
            _sample_object(
                rng,
                mover_family,
                name=f"mover_{idx}",
                material_keys_by_category=material_keys,
                position=(start_x + rng.uniform(-0.18, 0.12), rng.uniform(0.50, 0.92), 0.17),
                orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
                linear_velocity=(direction * abs(mover_linear[0]), mover_linear[1], mover_linear[2]),
                angular_velocity=mover_angular,
                forced_material_key=forced_mover_materials[idx] if idx < len(forced_mover_materials) else None,
            )
        )
    occluders = [
        _sample_object(
            rng,
            occluder_family,
            name="occluder_left",
            material_keys_by_category=material_keys,
            dynamic=False,
            role="occluder",
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
                dynamic=False,
                role="occluder",
                position=(0.18, -0.05, 0.50),
                orientation_euler_deg=(0.0, 0.0, 0.0),
            )
        )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=tuple(movers + occluders),
        tags=("diverse_object", "appearance_randomized", "occlusion", motion["motion_mode"]),
    )


def _make_f5(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        dynamic=False,
        role="support",
        position=(rng.uniform(-0.08, 0.20), 0.0, 0.16),
        forced_material_key=str(rng.choice(["wood_plywood", "wood_dark", "cardboard_kraft"])),
    )
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
        position=(rng.uniform(-0.28, 0.00), 0.0, rng.uniform(0.64, 1.18)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(drop_orientation, dynamic_family.orientation_jitter_deg)),
        linear_velocity=drop_linear,
        angular_velocity=drop_angular,
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(dynamic, support),
        tags=("diverse_object", "appearance_randomized", "support_drop", motion["motion_mode"]),
    )


def _make_f6(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        dynamic=False,
        role="support",
        position=(0.0, 0.0, 0.12),
        forced_material_key=str(rng.choice(["wood_plywood", "wood_dark", "concrete_painted"])),
        orientation_euler_deg=(0.0, rng.uniform(0.0, 14.0), rng.uniform(-6.0, 6.0)),
    )
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
        position=(-1.6 + rng.uniform(-0.2, 0.1), rng.uniform(-0.08, 0.16), 0.55 + rng.uniform(0.0, 0.22)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
        linear_velocity=(abs(mover_linear[0]), mover_linear[1], mover_linear[2]),
        angular_velocity=mover_angular,
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(mover, support),
        tags=("diverse_object", "appearance_randomized", "ramp_motion", motion["motion_mode"]),
    )


def _make_f7(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        position=(-1.3 + rng.uniform(-0.15, 0.15), rng.uniform(-0.10, 0.10), 0.22 + rng.uniform(-0.04, 0.06)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
        linear_velocity=(mover_linear[0] * 0.55, mover_linear[1], mover_linear[2]),
        angular_velocity=(mover_angular[0], mover_angular[1] * 1.2, mover_angular[2]),
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(mover,),
        tags=("diverse_object", "appearance_randomized", "spin_dominant", motion["motion_mode"]),
    )


def _make_f8(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        position=(-1.0 + rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 1.05 + rng.uniform(0.0, 0.30)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_orientation, mover_family.orientation_jitter_deg)),
        linear_velocity=(abs(mover_linear[0]), mover_linear[1], -abs(mover_linear[0]) * 0.08),
        angular_velocity=mover_angular,
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(mover,),
        tags=("diverse_object", "appearance_randomized", "bounce_heavy", motion["motion_mode"]),
    )


def _make_f9(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        position=(-1.6 + rng.uniform(-0.2, 0.1), rng.uniform(-0.25, 0.25), 0.18 + rng.uniform(-0.05, 0.08)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_a_ori, mover_a_family.orientation_jitter_deg)),
        linear_velocity=mover_a_lin,
        angular_velocity=mover_a_ang,
    )
    mover_b = _sample_object(
        rng,
        mover_b_family,
        name="clutter_b",
        material_keys_by_category=material_keys,
        position=(rng.uniform(-0.3, 0.5), rng.uniform(-0.2, 0.2), 0.18 + rng.uniform(-0.05, 0.08)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_b_ori, mover_b_family.orientation_jitter_deg)),
        linear_velocity=mover_b_lin,
        angular_velocity=mover_b_ang,
    )
    support = _sample_object(
        rng,
        object_families[str(rng.choice(["slab_box", "stack_box"]))],
        name="clutter_support",
        material_keys_by_category=material_keys,
        dynamic=False,
        role="support",
        position=(rng.uniform(-0.20, 0.20), rng.uniform(-0.08, 0.08), 0.12),
        forced_material_key=str(rng.choice(["wood_plywood", "cardboard_kraft", "concrete_painted"])),
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(mover_a, mover_b, support),
        tags=("diverse_object", "appearance_randomized", "clutter_interaction", motion["motion_mode"]),
    )


def _make_f10(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
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
        position=(-1.5 + rng.uniform(-0.2, 0.1), rng.uniform(-0.10, 0.10), 0.18 + rng.uniform(-0.04, 0.06)),
        orientation_euler_deg=tuple(float(o + rng.uniform(-a, a)) for o, a in zip(mover_ori, mover_family.orientation_jitter_deg)),
        linear_velocity=mover_lin,
        angular_velocity=mover_ang,
    )
    support = _sample_object(
        rng,
        object_families[str(rng.choice(["slab_box", "stack_box", "platform_block"]))],
        name="edge_support",
        material_keys_by_category=material_keys,
        dynamic=False,
        role="support",
        position=(rng.uniform(-0.15, 0.15), 0.0, 0.10),
        forced_material_key=str(rng.choice(["wood_dark", "wood_plywood", "cardboard_kraft"])),
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(mover, support),
        tags=("diverse_object", "appearance_randomized", "edge_boundary", motion["motion_mode"]),
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
}


def generate_scenario_blueprint(
    family_key: str,
    sample_key: str,
    seed: int,
) -> ScenarioBlueprint:
    if family_key not in FAMILY_GENERATORS:
        raise KeyError(f"unsupported family_key={family_key}")
    rng = np.random.default_rng(seed)
    return FAMILY_GENERATORS[family_key](rng, sample_key)


def preview_diversity_report(num_samples_per_family: int = 6) -> dict[str, dict[str, object]]:
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
