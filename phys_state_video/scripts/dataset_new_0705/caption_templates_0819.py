"""Metadata-driven caption templates for the PhysV V2V dataset."""

from __future__ import annotations

from typing import Mapping


CAPTION_SCHEMA_VERSION = "physv_caption_v1"
CAPTION_FILES = {
    "specific": "captions/caption_specific.txt",
    "abstract": "captions/caption_abstract.txt",
    "bundle": "captions/captions.json",
}


def _control(metadata: Mapping[str, object]) -> Mapping[str, object]:
    value = metadata.get("control", {})
    return value if isinstance(value, Mapping) else {}


def _value_label(metadata: Mapping[str, object]) -> str:
    control = _control(metadata)
    label = str(control.get("value_label", "")).strip()
    if label:
        return label.replace(" deg", " degrees")
    value = control.get("value")
    units = str(control.get("units", "")).strip()
    if isinstance(value, (int, float)):
        suffix = " degrees" if units == "deg" else f" {units}" if units else ""
        return f"{float(value):.2f}{suffix}"
    return "the configured value"


def _specific_caption(metadata: Mapping[str, object]) -> str:
    family_key = str(metadata.get("family_key", ""))
    task_type = str(metadata.get("task_type", ""))
    value = _value_label(metadata)
    if family_key == "F11" or task_type == "table_rolloff":
        return (
            f"A ball rolls across a table that is {value} high, leaves the right edge, "
            "and falls under gravity."
        )
    if family_key == "F12_RAMP_LENGTH" or task_type == "incline_length_release":
        scenario = metadata.get("scenario_spec", {})
        scenario = scenario if isinstance(scenario, Mapping) else {}
        angle = scenario.get("ramp_angle_deg")
        angle_text = f"{float(angle):.1f} degrees" if isinstance(angle, (int, float)) else "a derived slope"
        return (
            f"A red wooden block is released from rest on an incline that is {value} long "
            f"with a {angle_text} slope; the high-end support height is fixed, and the block moves along the incline."
        )
    if family_key == "F12" or task_type == "incline_release":
        return (
            f"A red wooden block is released from rest on a ramp inclined at {value} "
            "and slides down the incline."
        )
    if family_key == "V2V_GAP" or task_type == "gap_rolloff":
        return (
            f"A ball rolls toward a visible platform gap that is {value} wide, "
            "leaves the first platform, and moves through the gap under gravity."
        )
    if family_key == "V2V_OBSTACLE_SIZE":
        return (
            f"A red ball with radius {value} rolls rightward toward a fixed blue barrier, "
            "collides with it, rebounds, and travels onward with a speed determined by its incoming motion."
        )
    if family_key == "V2V_OBSTACLE" or task_type == "obstacle_collision":
        return (
            f"A red ball starts from the same position and rolls rightward at {value} toward a fixed blue barrier, "
            "collides with it, rebounds, and travels onward with a speed determined by its incoming motion."
        )
    if family_key == "V2V_BOWL" or task_type == "bowl_descent":
        return (
            f"A blue rubber ball starts on the inner wall of a bowl with radius {value}, "
            "rolls downward under gravity, passes through the low point, reverses direction, "
            "and continues along the curved surface."
        )
    if family_key == "V2V_PENDULUM" or task_type == "pendulum_swing":
        return (
            f"A bob suspended from a fixed support by a pendulum of length {value} "
            "starts at an angle, swings through the vertical, and continues oscillating."
        )
    if family_key == "V2V_SEESAW" or task_type == "seesaw_rotation":
        return (
            f"A block rests on a hinged board at {value} from the pivot; the load shifts "
            "the balance and the board rotates."
        )
    if family_key == "V2V_DOMINO" or task_type == "domino_chain":
        return (
            f"A red ball rolls into a row of dominoes spaced {value} apart, knocks the first "
            "domino over, and begins a contact-transfer sequence through the row."
        )
    description = str(metadata.get("scene_description_simulator_only", "")).strip()
    return description or "Rigid objects move through a simulated physical scene."


def _abstract_caption(metadata: Mapping[str, object]) -> str:
    family_key = str(metadata.get("family_key", ""))
    task_type = str(metadata.get("task_type", ""))
    if family_key == "F11" or task_type == "table_rolloff":
        return "A ball rolls across a raised table, leaves the right edge, and falls under gravity."
    if family_key == "F12_RAMP_LENGTH" or task_type == "incline_length_release":
        return "A red wooden block is released from rest on an incline with a fixed high-end support height; its length changes the slope, and the block moves along the incline."
    if family_key == "F12" or task_type == "incline_release":
        return "A red wooden block is released from rest on an inclined surface and slides down the slope."
    if family_key == "V2V_GAP" or task_type == "gap_rolloff":
        return "A ball rolls across a platform toward a visible gap, leaves the edge, and moves through the opening under gravity."
    if family_key == "V2V_OBSTACLE_SIZE":
        return "A red ball rolls rightward toward a fixed blue barrier, rebounds after contact, and continues moving."
    if family_key == "V2V_OBSTACLE" or task_type == "obstacle_collision":
        return "A red ball rolls rightward from the same starting position at a different initial speed toward a fixed blue barrier, rebounds after contact, and continues moving."
    if family_key == "V2V_BOWL" or task_type == "bowl_descent":
        return "A blue rubber ball starts on the inner wall of a curved bowl, rolls downward under gravity, reverses after passing through the low point, and continues along the wall."
    if family_key == "V2V_PENDULUM" or task_type == "pendulum_swing":
        return "A bob suspended from a fixed support starts at an angle, swings through the vertical, and continues oscillating."
    if family_key == "V2V_SEESAW" or task_type == "seesaw_rotation":
        return "A block rests on a hinged board, shifts the balance, and causes the board to rotate."
    if family_key == "V2V_DOMINO" or task_type == "domino_chain":
        return "A red ball rolls into a row of upright dominoes, knocks the first one over, and starts contact transfer through the row."
    return "Rigid objects move through a simulated physical scene."


def build_caption_bundle(metadata: Mapping[str, object]) -> dict[str, str]:
    """Build explicit and control-variable-blind captions from sample metadata."""
    return {
        "specific": _specific_caption(metadata),
        "abstract": _abstract_caption(metadata),
    }


def attach_caption_metadata(metadata: dict[str, object]) -> dict[str, str]:
    bundle = build_caption_bundle(metadata)
    metadata["caption_schema_version"] = CAPTION_SCHEMA_VERSION
    metadata["captions"] = {
        "specific": {
            "text": bundle["specific"],
            "file": CAPTION_FILES["specific"],
            "control_variable_exposed": True,
        },
        "abstract": {
            "text": bundle["abstract"],
            "file": CAPTION_FILES["abstract"],
            "control_variable_exposed": False,
        },
        "bundle_file": CAPTION_FILES["bundle"],
    }
    return bundle
