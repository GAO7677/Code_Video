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
    driver_key = str(rng.choice(["ball", "capsule_can", "flat_puck", "wheel", "spool", "bobbin", "drum_barrel", "roller_drum", "dumbbell"]))
    driver_family = object_families[driver_key]
    driver = _sample_object(
        rng,
        driver_family,
        name="driver_0",
        material_keys_by_category=material_keys,
        position=(-2.0 + rng.uniform(-0.25, 0.10), rng.uniform(-0.55, 0.55), 0.18 + rng.uniform(-0.03, 0.10)),
        orientation_euler_deg=tuple(float(rng.uniform(-a, a)) for a in driver_family.orientation_jitter_deg),
        linear_velocity=(rng.uniform(2.7, 4.9), rng.uniform(-0.45, 0.45), rng.uniform(-0.08, 0.04)),
        angular_velocity=(rng.uniform(-2.0, 10.0), rng.uniform(-8.0, 8.0), rng.uniform(-3.0, 5.0)),
    )
    camera_key = str(rng.choice(family.preferred_camera_keys))
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
        camera=_sample_camera(rng, camera_key),
        objects=(driver,),
        tags=("diverse_object", "appearance_randomized", "single_motion"),
    )


def _make_f2(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F2"]
    material_keys = _material_keys_by_category()
    driver_family = object_families[str(rng.choice(["flat_puck", "ball", "capsule_can", "wheel", "bobbin", "drum_barrel", "roller_drum"]))]
    target_family = object_families[str(rng.choice(["crate_box", "upright_cylinder", "tall_box", "cone_frustum", "tool_case", "shipping_box"]))]
    driver = _sample_object(
        rng,
        driver_family,
        name="driver_0",
        material_keys_by_category=material_keys,
        position=(-2.10 + rng.uniform(-0.25, 0.08), rng.uniform(-0.50, 0.50), 0.16 + rng.uniform(-0.04, 0.06)),
        orientation_euler_deg=tuple(float(rng.uniform(-a, a)) for a in driver_family.orientation_jitter_deg),
        linear_velocity=(rng.uniform(3.2, 5.0), rng.uniform(-0.35, 0.35), 0.0),
        angular_velocity=(rng.uniform(-2.0, 8.0), rng.uniform(-8.0, 8.0), rng.uniform(-3.0, 7.0)),
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
        tags=("diverse_object", "appearance_randomized", "two_body_contact"),
    )


def _make_f3(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F3"]
    material_keys = _material_keys_by_category()
    lead_family = object_families[str(rng.choice(["ball", "capsule_can", "wheel", "spool", "bobbin", "drum_barrel"]))]
    mid_family = object_families[str(rng.choice(["crate_box", "upright_cylinder", "tall_box", "tool_case", "shipping_box"]))]
    tail_family = object_families[str(rng.choice(["crate_box", "upright_cylinder", "cone_frustum", "tool_case", "shipping_box", "roller_drum"]))]
    lead = _sample_object(
        rng,
        lead_family,
        name="lead_0",
        material_keys_by_category=material_keys,
        position=(-2.20 + rng.uniform(-0.18, 0.06), rng.uniform(-0.18, 0.18), 0.17 + rng.uniform(-0.03, 0.04)),
        orientation_euler_deg=tuple(float(rng.uniform(-a, a)) for a in lead_family.orientation_jitter_deg),
        linear_velocity=(rng.uniform(3.4, 4.8), rng.uniform(-0.18, 0.18), 0.0),
        angular_velocity=(rng.uniform(-1.0, 9.0), rng.uniform(-7.0, 7.0), rng.uniform(-2.0, 6.0)),
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
        tags=("diverse_object", "appearance_randomized", "chain_reaction"),
    )


def _make_f4(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F4"]
    material_keys = _material_keys_by_category()
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
        movers.append(
            _sample_object(
                rng,
                mover_family,
                name=f"mover_{idx}",
                material_keys_by_category=material_keys,
                position=(start_x + rng.uniform(-0.18, 0.12), rng.uniform(0.50, 0.92), 0.17),
                linear_velocity=(direction * rng.uniform(2.8, 3.8), rng.uniform(-0.08, 0.08), 0.0),
                angular_velocity=(0.0, rng.uniform(-7.0, 7.0), 0.0),
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
        tags=("diverse_object", "appearance_randomized", "occlusion"),
    )


def _make_f5(rng: np.random.Generator, sample_key: str) -> ScenarioBlueprint:
    object_families = build_object_family_catalog()
    family = build_scenario_family_catalog()["F5"]
    material_keys = _material_keys_by_category()
    dynamic_family = object_families[str(rng.choice(["ball", "upright_cylinder", "crate_box", "cone_frustum", "tool_case", "shipping_box", "drum_barrel"]))]
    support_family = object_families[str(rng.choice(["platform_block", "wedge_ramp"]))]
    support = _sample_object(
        rng,
        support_family,
        name="support_0",
        material_keys_by_category=material_keys,
        dynamic=False,
        role="support",
        position=(rng.uniform(-0.08, 0.20), 0.0, 0.12),
    )
    dynamic = _sample_object(
        rng,
        dynamic_family,
        name="drop_0",
        material_keys_by_category=material_keys,
        position=(rng.uniform(-0.28, 0.00), 0.0, rng.uniform(0.64, 1.18)),
        orientation_euler_deg=(rng.uniform(-10.0, 14.0), rng.uniform(-10.0, 14.0), rng.uniform(-18.0, 18.0)),
        linear_velocity=(rng.uniform(0.10, 0.80), 0.0, rng.uniform(-0.16, 0.02)),
        angular_velocity=(rng.uniform(-3.0, 4.0), rng.uniform(-5.0, 5.0), rng.uniform(-3.0, 4.0)),
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
        tags=("diverse_object", "appearance_randomized", "support_drop"),
    )


FAMILY_GENERATORS = {
    "F1": _make_f1,
    "F2": _make_f2,
    "F3": _make_f3,
    "F4": _make_f4,
    "F5": _make_f5,
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
