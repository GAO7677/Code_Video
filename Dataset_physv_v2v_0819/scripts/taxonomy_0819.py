"""Controlled-physics taxonomy for the PhysV V2V 0819 dataset."""

from __future__ import annotations


TAXONOMY_DEFINITIONS = {
    "Scene": "Static environment geometry changes; the moving object's geometry and physical parameters stay fixed.",
    "Object": "The environment stays fixed; only the moving object's geometry or initial state changes.",
    "Relation": "The object and environment stay fixed; only their relative position, direction, or support relation changes.",
}


FAMILY_TAXONOMY = {
    "F11": "Scene",
    "F12": "Scene",
    "F12_RAMP_LENGTH": "Scene",
    "V2V_GAP": "Scene",
    "V2V_BOWL": "Scene",
    "V2V_DOMINO": "Relation",
    "V2V_OBSTACLE": "Object",
    "V2V_OBSTACLE_SIZE": "Object",
    "V2V_PENDULUM": "Object",
    "V2V_SEESAW": "Relation",
    "SCENE_PUCK_BARRIER": "Scene",
    "SCENE_DOOR_FRAME": "Scene",
}


def taxonomy_for_family(family_key: str) -> str:
    try:
        return FAMILY_TAXONOMY[family_key]
    except KeyError as exc:
        raise KeyError(f"no taxonomy mapping for family {family_key!r}") from exc
