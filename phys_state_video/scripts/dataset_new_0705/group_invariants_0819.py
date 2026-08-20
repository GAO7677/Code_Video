"""Semantic invariant checks for controlled PhysV case groups."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Iterable


_METADATA_VARIABLE_KEYS = {
    "F11": {"table_height_m"},
    "F12": {"ramp_angle_deg", "ramp_support_height_m"},
    "F12_RAMP_LENGTH": {"ramp_length_m", "ramp_angle_deg"},
    "V2V_GAP": {"gap_width_m", "left_platform_edge_x_m"},
    "V2V_OBSTACLE": {"initial_speed_mps"},
    "V2V_OBSTACLE_SIZE": {
        "ball_radius_m",
        "ball_mass_kg",
        "ball_volume_m3",
    },
    "V2V_BOWL": {
        "bowl_radius_m",
        "bowl_span_m",
        "ball_initial_surface_x_m",
        "ball_contact_segment_name",
    },
    "V2V_PENDULUM": {"pendulum_length_m", "constraints"},
    "V2V_SEESAW": {"load_position_x_m"},
    "V2V_DOMINO": {"domino_gap_m", "domino_center_step_m"},
}

_GEOMETRY_VARYING_FAMILIES = {
    "F11",
    "F12",
    "F12_RAMP_LENGTH",
    "V2V_GAP",
    "V2V_BOWL",
    "V2V_PENDULUM",
    "V2V_SEESAW",
    "V2V_DOMINO",
}


def _freeze(value: Any) -> Any:
    if is_dataclass(value):
        return _freeze(asdict(value))
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float):
        return round(value, 8)
    return value


def _filtered_metadata(blueprint: Any, family_key: str) -> Any:
    ignored = _METADATA_VARIABLE_KEYS.get(family_key, set()) | {
        "controlled_variable",
        "initialization_contract",
        "color_separation_qa",
    }
    return {
        key: value
        for key, value in blueprint.metadata.items()
        if key not in ignored
    }


def _varying_object_fields(family_key: str, name: str) -> set[str]:
    fields: set[str] = set()
    if family_key in _GEOMETRY_VARYING_FAMILIES:
        fields.update({"position", "orientation_euler_deg"})
    if family_key in {"F11", "F12_RAMP_LENGTH", "V2V_BOWL"}:
        fields.add("size")
    if family_key == "F12":
        fields.update({"position", "orientation_euler_deg", "size"})
    if family_key == "V2V_OBSTACLE" and name == "obstacle_ball":
        fields.update({"linear_velocity", "angular_velocity"})
    if family_key == "V2V_OBSTACLE_SIZE" and name == "obstacle_ball":
        fields.update({"position", "size", "mass"})
    if family_key == "V2V_PENDULUM" and name == "pendulum_rope":
        fields.add("size")
    return fields


def _object_signature(obj: Any, family_key: str) -> Any:
    fields = {
        "name": obj.name,
        "family_key": obj.family_key,
        "shape": obj.shape,
        "semantic_role": obj.semantic_role,
        "size": obj.size,
        "mass": obj.mass,
        "friction": obj.friction,
        "restitution": obj.restitution,
        "linear_damping": obj.linear_damping,
        "angular_damping": obj.angular_damping,
        "material_key": obj.material_key,
        "dynamic": obj.dynamic,
        "role": obj.role,
        "position": obj.position,
        "orientation_euler_deg": obj.orientation_euler_deg,
        "linear_velocity": obj.linear_velocity,
        "angular_velocity": obj.angular_velocity,
        "metadata": obj.metadata,
    }
    for field_name in _varying_object_fields(family_key, obj.name):
        fields.pop(field_name, None)
    if family_key == "V2V_OBSTACLE_SIZE" and obj.name == "obstacle_ball":
        fields["metadata"] = {
            key: value
            for key, value in obj.metadata.items()
            if key not in {"ccd_swept_sphere_radius_m"}
        }
    return _freeze(fields)


def invariant_signature(case: Any) -> Any:
    blueprint = case.blueprint
    family_key = str(case.family_key)
    camera = blueprint.camera
    camera_fields = {
        "eye": camera.eye,
        "target": camera.target,
        "up": camera.up,
        "yfov_deg": camera.yfov_deg,
        "jitter_eye_xyz": camera.jitter_eye_xyz,
        "jitter_target_xyz": camera.jitter_target_xyz,
        "jitter_fov_deg": camera.jitter_fov_deg,
        "hdri_key": camera.hdri_key,
        "exposure_range": camera.exposure_range,
    }
    return _freeze(
        {
            "gravity": blueprint.gravity,
            "pre_roll_s": blueprint.pre_roll_s,
            "camera_key": blueprint.camera_key,
            "surface_key": blueprint.surface_key,
            "lighting_key": blueprint.lighting_key,
            "camera": camera_fields,
            "metadata": _filtered_metadata(blueprint, family_key),
            "objects": {
                obj.name: _object_signature(obj, family_key)
                for obj in blueprint.objects
            },
        }
    )


def audit_group_invariants(cases: Iterable[Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    for case in cases:
        grouped.setdefault(str(case.family_key), []).append(case)
    reports: dict[str, dict[str, Any]] = {}
    for family_key, family_cases in sorted(grouped.items()):
        reference = invariant_signature(family_cases[0])
        mismatches = [
            {
                "case_id": case.case_id,
                "matches_reference": invariant_signature(case) == reference,
            }
            for case in family_cases[1:]
            if invariant_signature(case) != reference
        ]
        reports[family_key] = {
            "case_count": len(family_cases),
            "passed": not mismatches,
            "reference_case_id": family_cases[0].case_id,
            "mismatches": mismatches,
        }
    failures = [key for key, report in reports.items() if not report["passed"]]
    if failures:
        details = "; ".join(
            f"{key}: {[item['case_id'] for item in reports[key]['mismatches']]}"
            for key in failures
        )
        raise ValueError(f"controlled-group invariant mismatch: {details}")
    return reports
