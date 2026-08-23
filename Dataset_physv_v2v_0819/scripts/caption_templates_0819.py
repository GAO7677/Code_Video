"""Observed-outcome caption templates for the PhysV V2V dataset."""

from __future__ import annotations

from typing import Mapping


CAPTION_SCHEMA_VERSION = "physv_caption_v2_observed"
CAPTION_FILES = {
    "specific": "captions/caption_specific.txt",
    "abstract": "captions/caption_abstract.txt",
    "bundle": "captions/captions.json",
}


def _control(metadata: Mapping[str, object]) -> Mapping[str, object]:
    value = metadata.get("control", {})
    return value if isinstance(value, Mapping) else {}


def _scenario(metadata: Mapping[str, object]) -> Mapping[str, object]:
    value = metadata.get("scenario_spec", {})
    return value if isinstance(value, Mapping) else {}


def _observations(metadata: Mapping[str, object]) -> Mapping[str, object]:
    value = metadata.get("caption_observations", {})
    return value if isinstance(value, Mapping) else {}


def _details(metadata: Mapping[str, object]) -> Mapping[str, object]:
    value = _observations(metadata).get("details", {})
    return value if isinstance(value, Mapping) else {}


def _outcome(metadata: Mapping[str, object]) -> str:
    return str(_observations(metadata).get("outcome_code", ""))


def _value_label(metadata: Mapping[str, object]) -> str:
    control = _control(metadata)
    label = str(control.get("value_label", "")).strip()
    if label:
        if label.startswith("v="):
            return label[2:]
        if label.startswith("r=") or label.startswith("R="):
            return label[2:]
        if label.startswith("anchor="):
            return label[8:]
        if label.startswith("opening="):
            return label[9:]
        if label.startswith("normal="):
            return label[7:]
        if label.endswith(" deg"):
            return label[:-4] + " degrees"
        return label
    value = control.get("value")
    units = str(control.get("units", "")).strip()
    if isinstance(value, (int, float)):
        suffix = " degrees" if units == "deg" else f" {units}" if units else ""
        return f"{float(value):.2f}{suffix}"
    return "the configured value"


def _number_text(value: object, fallback: str) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return fallback


def _f11_caption(metadata: Mapping[str, object], specific: bool) -> str:
    value = _value_label(metadata) if specific else "a raised"
    outcome = _outcome(metadata)
    if specific:
        opening = f"A ball rolls across a table that is {value} high and leaves its right edge."
    else:
        opening = "A ball rolls across a raised table and leaves its right edge."
    if outcome == "table_rolloff_lands_and_rolls":
        return opening + " It lands on the ground and continues rolling at the end of the video."
    if outcome == "table_rolloff_bounces_back_airborne":
        return opening + " It briefly contacts the ground, bounces back into the air, and is airborne again at the end."
    return opening + " It remains airborne after the drop at the end of the video."


def _ramp_caption(metadata: Mapping[str, object], specific: bool) -> str:
    details = _details(metadata)
    outcome = _outcome(metadata)
    angle_fallback = _value_label(metadata) if str(metadata.get("family_key", "")) == "F12" else "the configured"
    angle = _number_text(details.get("ramp_angle_deg"), angle_fallback)
    length = _number_text(details.get("ramp_length_m"), "the configured")
    if specific:
        if str(metadata.get("family_key", "")) == "F12_RAMP_LENGTH":
            opening = f"A red wooden block is released from rest on a {length} m ramp inclined at {angle} degrees."
        else:
            opening = f"A red wooden block is released from rest on a ramp inclined at {angle} degrees."
    elif str(metadata.get("family_key", "")) == "F12_RAMP_LENGTH":
        opening = "A red wooden block is released from rest on an inclined surface with the high-end support height fixed."
    else:
        opening = "A red wooden block is released from rest on an inclined surface."
    if not outcome:
        return opening + " It slides down the incline."
    if outcome == "ramp_block_remains_on_ramp":
        return opening + " It slides along the ramp and is still on the ramp at the end."
    if outcome == "ramp_block_lands_and_slides_on_floor":
        return opening + " It leaves the ramp, lands on the floor, and continues sliding to the right at the end."
    return opening + " It leaves the ramp, lands on the floor, and comes to rest by the end."


def _gap_caption(metadata: Mapping[str, object], specific: bool) -> str:
    value = _value_label(metadata) if specific else ""
    outcome = _outcome(metadata)
    opening = (
        f"A red ball rolls from the left platform toward a {value} gap between two platforms."
        if specific
        else "A red ball rolls from the left platform toward a gap between two platforms."
    )
    if outcome == "gap_crosses_to_right_platform":
        return opening + " It crosses the gap, lands on the opposite platform, and continues rolling there at the end."
    if outcome == "gap_brief_right_edge_contact_then_ground":
        return opening + " It drops through the gap, briefly contacts the far platform edge, then reaches the ground and is still rolling at the end."
    if outcome == "gap_drops_to_ground_and_reaches_support":
        return opening + " It drops through the gap to the ground, rolls toward the far platform support, and comes to rest there by the end."
    if outcome == "gap_drops_to_ground":
        return opening + " It drops through the gap to the ground and continues moving there at the end."
    return opening + " It leaves the platform and remains airborne at the end."


def _obstacle_caption(metadata: Mapping[str, object], specific: bool, size_control: bool) -> str:
    details = _details(metadata)
    outcome = _outcome(metadata)
    if size_control:
        value = _number_text(details.get("ball_radius_m"), _value_label(metadata)) if specific else None
        opening = (
            f"A red ball with radius {value} m rolls rightward toward a fixed blue barrier."
            if specific
            else "A red ball rolls rightward toward a fixed blue barrier."
        )
    else:
        value = _value_label(metadata) if specific else None
        opening = (
            f"A red ball starts from the same position and rolls rightward at {value} toward a fixed blue barrier."
            if specific
            else "A red ball starts from the same position and rolls rightward at a controlled speed toward a fixed blue barrier."
        )
    if outcome == "obstacle_ball_slows_before_collision":
        return opening + " It slows before reaching the barrier and remains on the approach side at the end; no rebound is observed."
    if outcome == "obstacle_ball_short_rebound":
        return opening + " It reverses near the barrier and is nearly stopped close to it at the end."
    if outcome == "obstacle_ball_rebounds_without_returning_to_start":
        return opening + " It reverses near the barrier and travels back left without reaching its release position by the end."
    if outcome == "obstacle_ball_strong_rebound_left":
        return opening + " It rebounds strongly and travels far back to the left while still moving at the end."
    if outcome == "obstacle_size_rebounds_and_stops_before_start":
        return opening + " It rebounds but comes to rest before returning to its release region."
    if outcome == "obstacle_size_rebounds_near_start":
        return opening + " It rebounds toward the left, reaches the release region, and is still moving slowly at the end."
    if outcome == "obstacle_size_rebounds_past_start":
        return opening + " It rebounds and travels past its release position while still moving left at the end."
    return opening + " It reaches the barrier region but does not show a rebound."


def _bowl_caption(metadata: Mapping[str, object], specific: bool) -> str:
    details = _details(metadata)
    if specific:
        radius = _number_text(details.get("bowl_radius_m"), _value_label(metadata))
        opening = f"A blue rubber ball starts on the inner wall of a bowl with radius {radius} m."
    else:
        opening = "A blue rubber ball starts on the inner wall of a curved bowl."
    return opening + " It rolls down the wall, passes through the low region, reverses direction, and remains in motion along the curved surface at the end."


def _pendulum_caption(metadata: Mapping[str, object], specific: bool) -> str:
    if specific:
        length = _number_text(_details(metadata).get("pendulum_length_m"), _value_label(metadata))
        opening = f"A bob suspended by a pendulum {length} m long is released from an angle."
    else:
        opening = "A bob suspended by a pendulum is released from an angle."
    return opening + " It swings through the low point, reverses direction, and is still moving at the end."


def _pendulum_cabinet_caption(metadata: Mapping[str, object], specific: bool) -> str:
    details = _details(metadata)
    if specific:
        anchor = _number_text(details.get("anchor_height_m"), _value_label(metadata))
        impact_height = _number_text(details.get("impact_height_m"), "the observed")
        impact_speed = _number_text(details.get("impact_speed_mps"), "the observed")
        opening = f"A fixed-length pendulum with its suspension point {anchor} m high swings toward a tall cabinet."
        return opening + f" The bob contacts the cabinet at about {impact_height} m height with an impact speed of {impact_speed} m/s, then continues swinging."
    return "A fixed-length pendulum swings toward a tall cabinet; changing the suspension height changes the contact height while the release geometry and impact speed stay matched. The bob continues swinging after contact."


def _seesaw_caption(metadata: Mapping[str, object], specific: bool) -> str:
    details = _details(metadata)
    load_x = _number_text(details.get("load_position_x_m"), _value_label(metadata))
    centered = _outcome(metadata) == "seesaw_centered_load_stays_level"
    if specific:
        opening = f"A block is placed at x={load_x} m on a hinged seesaw."
    else:
        opening = "A block is placed on a hinged seesaw."
    if centered:
        return opening + " The board stays level and stationary through the end of the video."
    angle = _number_text(details.get("final_board_angle_deg"), "an observed")
    return opening + f" The off-center load tips the board by about {angle} degrees, then the board and block settle by the end."


def _domino_caption(metadata: Mapping[str, object], specific: bool) -> str:
    value = _value_label(metadata) if specific else None
    opening = (
        f"A red trigger ball rolls into five dominoes spaced {value} apart."
        if specific
        else "A red trigger ball rolls into a row of five dominoes."
    )
    outcome = _outcome(metadata)
    if outcome == "domino_chain_reaches_all_five":
        return opening + " The contact sequence propagates through the row, and all five dominoes are toppled at the end."
    if outcome == "domino_chain_does_not_start":
        return opening + " The dominoes remain upright at the end after only small disturbances."
    if outcome == "domino_chain_only_nudges_row":
        return opening + " The first three dominoes tilt slightly, while the last two remain upright at the end."
    if outcome == "domino_last_only_topples":
        return opening + " The first four dominoes remain upright while only the last domino is lying down at the end."
    return opening + " The first four dominoes remain upright while the last domino is lying down at the end."


def _puck_caption(metadata: Mapping[str, object], specific: bool) -> str:
    value = _value_label(metadata) if specific else None
    opening = (
        f"An ice puck slides across a low-friction floor toward a fixed rigid barrier oriented at {value}."
        if specific
        else "An ice puck slides across a low-friction floor toward a fixed rigid barrier."
    )
    endings = {
        "puck_rebounds_forward_transverse": "After reaching the barrier, it continues forward with a strong transverse deflection.",
        "puck_rebounds_almost_sideways": "It turns almost sideways after the impact and continues moving.",
        "puck_rebounds_diagonally_back": "It rebounds diagonally back across the floor and remains moving.",
        "puck_rebounds_mostly_back_with_lateral_deflection": "It rebounds mostly back toward its approach side with a lateral component.",
        "puck_rebounds_straight_back": "It reverses almost straight back along its incoming line.",
    }
    return opening + " " + endings.get(_outcome(metadata), "It changes direction after reaching the barrier and remains moving.")


def _door_caption(metadata: Mapping[str, object], specific: bool, ball: bool) -> str:
    details = _details(metadata)
    value = _number_text(details.get("opening_width_m"), _value_label(metadata))
    object_name = "blue rubber ball" if ball else "wooden crate"
    opening = (
        f"A {object_name} moves toward a fixed door frame with a {value} m-wide opening."
        if specific
        else f"A {object_name} moves toward a fixed door frame."
    )
    outcome = _outcome(metadata)
    if outcome == "door_object_passes_clear":
        return opening + " It passes through without contacting the frame and continues moving at the end."
    if outcome == "door_ball_touches_both_posts_then_passes":
        return opening + " It contacts both side posts, squeezes through, and continues moving beyond the doorway at the end."
    if outcome == "door_ball_deflects_then_passes":
        return opening + " It touches a side post, deflects laterally, passes through, and continues moving at the end."
    if outcome == "door_ball_stays_in_frame_and_rotates":
        return opening + " It contacts the side posts, remains in the doorway, and is still rotating at the end."
    if outcome == "door_crate_contacts_both_posts_and_stays_blocked":
        return opening + " It rubs both side posts, rotates, and remains blocked in the doorway at the end."
    if outcome == "door_crate_contacts_post_and_stays_blocked":
        return opening + " It contacts a side post, rotates or deflects, and remains blocked in the doorway at the end."
    if outcome == "door_crate_contacts_post_and_stops":
        return opening + " It contacts the right post and stops in the doorway at the end."
    return opening + " It does not reach the doorway during the observed clip."


def _specific_caption(metadata: Mapping[str, object]) -> str:
    family_key = str(metadata.get("family_key", ""))
    task_type = str(metadata.get("task_type", ""))
    if family_key == "F11" or task_type == "table_rolloff":
        return _f11_caption(metadata, True)
    if family_key == "F12_RAMP_LENGTH" or task_type == "incline_length_release":
        return _ramp_caption(metadata, True)
    if family_key == "F12" or task_type == "incline_release":
        return _ramp_caption(metadata, True)
    if family_key == "V2V_GAP" or task_type == "gap_rolloff":
        return _gap_caption(metadata, True)
    if family_key == "V2V_OBSTACLE_SIZE":
        return _obstacle_caption(metadata, True, True)
    if family_key == "V2V_OBSTACLE" or task_type == "obstacle_collision":
        return _obstacle_caption(metadata, True, False)
    if family_key == "V2V_BOWL" or task_type == "bowl_descent":
        return _bowl_caption(metadata, True)
    if family_key == "V2V_PENDULUM_CABINET" or task_type == "pendulum_cabinet_collision":
        return _pendulum_cabinet_caption(metadata, True)
    if family_key == "V2V_PENDULUM" or task_type == "pendulum_swing":
        return _pendulum_caption(metadata, True)
    if family_key == "V2V_SEESAW" or task_type == "seesaw_rotation":
        return _seesaw_caption(metadata, True)
    if family_key == "V2V_DOMINO" or task_type == "domino_chain":
        return _domino_caption(metadata, True)
    if family_key == "SCENE_PUCK_BARRIER" or task_type == "puck_barrier_collision":
        return _puck_caption(metadata, True)
    if family_key == "SCENE_DOOR_FRAME_BALL" or task_type == "door_frame_clearance_ball":
        return _door_caption(metadata, True, True)
    if family_key == "SCENE_DOOR_FRAME" or task_type == "door_frame_clearance":
        return _door_caption(metadata, True, False)
    description = str(metadata.get("scene_description_simulator_only", "")).strip()
    return description or "Rigid objects move through a simulated physical scene."


def _abstract_caption(metadata: Mapping[str, object]) -> str:
    family_key = str(metadata.get("family_key", ""))
    task_type = str(metadata.get("task_type", ""))
    if family_key == "F11" or task_type == "table_rolloff":
        return _f11_caption(metadata, False)
    if family_key in {"F12", "F12_RAMP_LENGTH"} or task_type in {"incline_release", "incline_length_release"}:
        return _ramp_caption(metadata, False)
    if family_key == "V2V_GAP" or task_type == "gap_rolloff":
        return _gap_caption(metadata, False)
    if family_key == "V2V_OBSTACLE_SIZE":
        return _obstacle_caption(metadata, False, True)
    if family_key == "V2V_OBSTACLE" or task_type == "obstacle_collision":
        return _obstacle_caption(metadata, False, False)
    if family_key == "V2V_BOWL" or task_type == "bowl_descent":
        return _bowl_caption(metadata, False)
    if family_key == "V2V_PENDULUM_CABINET" or task_type == "pendulum_cabinet_collision":
        return _pendulum_cabinet_caption(metadata, False)
    if family_key == "V2V_PENDULUM" or task_type == "pendulum_swing":
        return _pendulum_caption(metadata, False)
    if family_key == "V2V_SEESAW" or task_type == "seesaw_rotation":
        return _seesaw_caption(metadata, False)
    if family_key == "V2V_DOMINO" or task_type == "domino_chain":
        return _domino_caption(metadata, False)
    if family_key == "SCENE_PUCK_BARRIER" or task_type == "puck_barrier_collision":
        return _puck_caption(metadata, False)
    if family_key == "SCENE_DOOR_FRAME_BALL" or task_type == "door_frame_clearance_ball":
        return _door_caption(metadata, False, True)
    if family_key == "SCENE_DOOR_FRAME" or task_type == "door_frame_clearance":
        return _door_caption(metadata, False, False)
    return "Rigid objects move through a simulated physical scene."


def build_caption_bundle(metadata: Mapping[str, object]) -> dict[str, str]:
    """Build captions from observed outcomes attached to metadata."""
    return {
        "specific": _specific_caption(metadata),
        "abstract": _abstract_caption(metadata),
    }


def attach_caption_metadata(
    metadata: dict[str, object],
    observations: Mapping[str, object] | None = None,
) -> dict[str, str]:
    if observations is not None:
        metadata["caption_observations"] = dict(observations)
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
