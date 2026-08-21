"""Initialization and appearance contracts for the PhysV 0819 controls.

The simulation already rejects deep overlaps.  This module adds the stricter
part of the dataset contract: explicitly connected parts must be touching,
and visually distinct objects in one scene must not collapse to near-identical
base colors.
"""

from __future__ import annotations

import math
from typing import Iterable

from .common_specs import ObjectInstanceSpec, ScenarioBlueprint
from .material_catalog_0705 import build_material_catalog


CONTACT_GAP_TOLERANCE_M = 0.001
COLOR_DISTANCE_MIN = 0.18


def _pairs(names: set[str], candidates: Iterable[tuple[str, str]]) -> list[list[str]]:
    return [
        [left, right]
        for left, right in candidates
        if left in names and right in names and left != right
    ]


def build_contact_contract(blueprint: ScenarioBlueprint) -> dict[str, object]:
    """Return the expected touching relationships for a supported scene."""
    names = {obj.name for obj in blueprint.objects}
    family = blueprint.family_key
    touching_pairs: list[list[str]] = []
    touching_ground: list[str] = []

    if family == "F11":
        touching_pairs = _pairs(
            names,
            [
                ("table_top_0", "table_leg_0"),
                ("table_top_0", "table_leg_1"),
                ("table_top_0", "table_leg_2"),
                ("table_top_0", "table_leg_3"),
                ("roller_0", "table_top_0"),
            ],
        )
        touching_ground = [name for name in names if name.startswith("table_leg_")]
    elif family == "F12":
        touching_pairs = _pairs(
            names,
            [
                ("incline_board_0", "incline_riser_0"),
                ("incline_board_0", "incline_riser_1"),
                ("incline_board_0", "block_0"),
            ],
        )
        touching_ground = [
            name
            for name in ("incline_board_0", "incline_riser_0", "incline_riser_1")
            if name in names
        ]
    elif family == "V2V_GAP":
        touching_pairs = _pairs(
            names,
            [
                ("gap_ball", "left_platform"),
                ("left_platform", "left_platform_support"),
                ("right_platform", "right_platform_support"),
            ],
        )
        touching_ground = [
            name
            for name in ("left_platform_support", "right_platform_support")
            if name in names
        ]
    elif family in {"V2V_OBSTACLE", "V2V_OBSTACLE_SIZE"}:
        touching_pairs = []
        touching_ground = [
            name
            for name in ("obstacle_ball", "obstacle_barrier")
            if name in names
        ]
    elif family == "V2V_BOWL":
        touching_ground = [name for name in ("bowl_base",) if name in names]
        segment_name = str(blueprint.metadata.get("ball_contact_segment_name", ""))
        if segment_name:
            touching_pairs = _pairs(names, [("bowl_ball", segment_name)])
    elif family == "V2V_PENDULUM":
        touching_pairs = _pairs(
            names,
            [
                ("pendulum_base", "pendulum_post"),
                ("pendulum_post", "pendulum_crossbar"),
                ("pendulum_crossbar", "pendulum_rope"),
                ("pendulum_rope", "pendulum_bob"),
            ],
        )
        touching_ground = [name for name in ("pendulum_base",) if name in names]
    elif family == "V2V_SEESAW":
        touching_pairs = _pairs(
            names,
            [
                ("seesaw_pivot", "seesaw_board"),
                ("seesaw_board", "seesaw_load"),
            ],
        )
        touching_ground = [name for name in ("seesaw_pivot",) if name in names]
    elif family == "V2V_DOMINO":
        touching_ground = [
            name
            for name in names
            if name == "domino_trigger_ball" or name.startswith("domino_")
        ]
    elif family == "SCENE_PUCK_BARRIER":
        touching_ground = [name for name in ("puck", "puck_barrier") if name in names]
    elif family == "SCENE_DOOR_FRAME":
        touching_pairs = _pairs(
            names,
            [
                ("door_frame_left", "door_frame_lintel"),
                ("door_frame_right", "door_frame_lintel"),
            ],
        )
        touching_ground = [
            name
            for name in ("door_crate", "door_frame_left", "door_frame_right")
            if name in names
        ]

    return {
        "touching_pairs": touching_pairs,
        "touching_ground": sorted(touching_ground),
        "gap_tolerance_m": CONTACT_GAP_TOLERANCE_M,
        "policy": "listed connected parts must be tangent within the tolerance at initialization",
    }


def _connected_pairs(contract: dict[str, object]) -> set[frozenset[str]]:
    return {
        frozenset((str(pair[0]), str(pair[1])))
        for pair in contract.get("touching_pairs", [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    }


def color_separation_report(
    blueprint: ScenarioBlueprint,
    *,
    minimum_distance: float = COLOR_DISTANCE_MIN,
) -> dict[str, object]:
    """Check base-color contrast for independently visible scene objects.

    Repeated static fixtures and directly connected structural parts may share
    a material.  Separate dynamic actors are still checked even when their
    material key is the same, because that is a likely visual ambiguity.
    """
    materials = build_material_catalog()
    contract = build_contact_contract(blueprint)
    connected = _connected_pairs(contract)
    violations: list[dict[str, object]] = []
    minimum_seen = math.inf
    objects = list(blueprint.objects)
    for index, first in enumerate(objects):
        color_a = materials[first.material_key].base_color
        for second in objects[index + 1 :]:
            color_b = materials[second.material_key].base_color
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(color_a, color_b)))
            minimum_seen = min(minimum_seen, distance)
            pair = frozenset((first.name, second.name))
            same_material = first.material_key == second.material_key
            both_static = not first.dynamic and not second.dynamic
            same_family = first.family_key == second.family_key
            directly_connected = pair in connected
            if (same_material and (both_static or same_family or directly_connected)) or directly_connected:
                continue
            if distance < minimum_distance:
                violations.append(
                    {
                        "object_a": first.name,
                        "object_b": second.name,
                        "material_a": first.material_key,
                        "material_b": second.material_key,
                        "base_color_distance": round(distance, 6),
                    }
                )
    return {
        "passed": not violations,
        "minimum_base_color_distance": (
            None if math.isinf(minimum_seen) else round(minimum_seen, 6)
        ),
        "minimum_required_distance": minimum_distance,
        "violations": violations,
    }


def validate_color_separation(blueprint: ScenarioBlueprint) -> dict[str, object]:
    report = color_separation_report(blueprint)
    if not report["passed"]:
        details = "; ".join(
            f"{item['object_a']} vs {item['object_b']}={item['base_color_distance']:.4f}"
            for item in report["violations"]
        )
        raise ValueError(f"{blueprint.sample_key}: similar scene colors: {details}")
    return report
