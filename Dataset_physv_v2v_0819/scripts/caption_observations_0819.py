"""Extract caption-worthy outcomes from exported physics supervision.

The simulator metadata describes the intended control, but it does not say
what actually happened in the rendered rollout.  This module keeps caption
generation grounded in the exported trajectories and contact records.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OBSERVATION_SCHEMA_VERSION = "physv_caption_observations_v1"
MOTION_THRESHOLD_MPS = 0.03


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: object, digits: int = 4) -> float:
    return round(_float(value), digits)


def _first_true(values: Sequence[bool] | np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return int(indices[0]) if len(indices) else None


def _last_true(values: Sequence[bool] | np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return int(indices[-1]) if len(indices) else None


def _frame_time(frame: int | None, frame_times: np.ndarray) -> float | None:
    if frame is None or frame < 0 or frame >= len(frame_times):
        return None
    return _round(frame_times[frame], 6)


def _pair_frames(
    contacts: Sequence[Mapping[str, object]],
    target: str,
    other: str | None = None,
) -> list[int]:
    frames: list[int] = []
    for record in contacts:
        left = str(record.get("obj_a", ""))
        right = str(record.get("obj_b", ""))
        if target not in (left, right):
            continue
        if other is not None and other not in (left, right):
            continue
        frames.append(int(record.get("frame", -1)))
    return sorted(set(frame for frame in frames if frame >= 0))


def _first_pair_frame(
    contacts: Sequence[Mapping[str, object]],
    target: str,
    other: str | None = None,
    minimum_frame: int = 0,
) -> int | None:
    frames = [frame for frame in _pair_frames(contacts, target, other) if frame >= minimum_frame]
    return frames[0] if frames else None


def _dynamic_index(names: Sequence[str], dynamic_mask: np.ndarray, preferred: str) -> int:
    if preferred in names:
        return names.index(preferred)
    candidates = np.flatnonzero(dynamic_mask)
    if len(candidates):
        return int(candidates[0])
    return 0


def _base_observation(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    outcome_code: str,
    event_frame: int | None,
    end_state: str,
    details: Mapping[str, object],
) -> dict[str, object]:
    event_frame = int(event_frame) if event_frame is not None else None
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "outcome_code": outcome_code,
        "event_frame": event_frame,
        "event_time_s": _frame_time(event_frame, frame_times),
        "end_state": end_state,
        "details": dict(details),
    }


def _common_motion_details(
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    index: int,
) -> dict[str, object]:
    return {
        "actor": "",
        "initial_speed_mps": _round(speeds[0, index]),
        "final_speed_mps": _round(speeds[-1, index]),
        "max_speed_mps": _round(np.max(speeds[:, index])),
        "initial_position_m": [_round(value) for value in positions[0, index]],
        "final_position_m": [_round(value) for value in positions[-1, index]],
        "final_velocity_mps": [_round(value) for value in velocities[-1, index]],
    }


def _reverse_frame(values: np.ndarray, start: int = 5, threshold: float = 0.03) -> int | None:
    if len(values) <= start:
        return None
    initial = float(np.median(values[: min(start, len(values))]))
    if abs(initial) < threshold:
        nonzero = values[np.abs(values) >= threshold]
        initial = float(nonzero[0]) if len(nonzero) else 0.0
    if abs(initial) < threshold:
        return None
    sign = 1.0 if initial > 0 else -1.0
    candidates = np.flatnonzero((np.arange(len(values)) >= start) & (values * sign < -threshold))
    return int(candidates[0]) if len(candidates) else None


def _derive_f11(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "roller_0")
    actor = names[index]
    table_frames = _pair_frames(contacts, actor, "table_top_0")
    ground_frames = _pair_frames(contacts, actor, "ground")
    conditioning = metadata.get("conditioning", {})
    conditioning = conditioning if isinstance(conditioning, Mapping) else {}
    event_frame = conditioning.get("first_event_frame")
    event_frame = int(event_frame) if isinstance(event_frame, (int, float)) and int(event_frame) >= 0 else None
    final_speed = float(speeds[-1, index])
    if ground_frames and ground_frames[-1] >= len(frame_times) - 3 and final_speed > MOTION_THRESHOLD_MPS:
        end_state = "moving_on_ground"
        outcome = "table_rolloff_lands_and_rolls"
    elif ground_frames and ground_frames[-1] < len(frame_times) - 3:
        end_state = "airborne_after_ground_contact"
        outcome = "table_rolloff_bounces_back_airborne"
    else:
        end_state = "airborne_after_table_edge"
        outcome = "table_rolloff_still_airborne"
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "table_contact_first_frame": table_frames[0] if table_frames else None,
            "table_contact_last_frame": table_frames[-1] if table_frames else None,
            "ground_contact_first_frame": ground_frames[0] if ground_frames else None,
            "ground_contact_last_frame": ground_frames[-1] if ground_frames else None,
        }
    )
    return _base_observation(metadata, frame_times, outcome, event_frame, end_state, details)


def _derive_ramp(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "block_0")
    actor = names[index]
    ramp_frames = _pair_frames(contacts, actor, "incline_board_0")
    ground_frames = _pair_frames(contacts, actor, "ground")
    final_speed = float(speeds[-1, index])
    still_on_ramp = bool(ramp_frames and ramp_frames[-1] >= len(frame_times) - 3 and not ground_frames)
    if still_on_ramp:
        outcome = "ramp_block_remains_on_ramp"
        end_state = "moving_on_ramp"
        event_frame = None
    elif final_speed > MOTION_THRESHOLD_MPS:
        outcome = "ramp_block_lands_and_slides_on_floor"
        end_state = "moving_on_floor"
        event_frame = ground_frames[0] if ground_frames else (ramp_frames[-1] + 1 if ramp_frames else None)
    else:
        outcome = "ramp_block_lands_and_stops_on_floor"
        end_state = "stopped_on_floor"
        event_frame = ground_frames[0] if ground_frames else (ramp_frames[-1] + 1 if ramp_frames else None)
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "ramp_contact_first_frame": ramp_frames[0] if ramp_frames else None,
            "ramp_contact_last_frame": ramp_frames[-1] if ramp_frames else None,
            "ground_contact_first_frame": ground_frames[0] if ground_frames else None,
            "ground_contact_last_frame": ground_frames[-1] if ground_frames else None,
            "ramp_angle_deg": _round(scenario.get("ramp_angle_deg"), 2),
            "ramp_length_m": _round(scenario.get("ramp_length_m"), 3),
        }
    )
    return _base_observation(metadata, frame_times, outcome, event_frame, end_state, details)


def _derive_ramp_platform(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Extract support-departure and landing events for the ramp-platform family."""
    index = _dynamic_index(names, dynamic_mask, "block_0")
    actor = names[index]
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}

    ramp_frames = _pair_frames(contacts, actor, "incline_board_0")
    platform_frames = _pair_frames(contacts, actor, "horizontal_platform_0")
    ground_frames = _pair_frames(contacts, actor, "ground")
    ramp_exit_x = _float(scenario.get("ramp_exit_x_m"), 0.0)
    platform_edge_x = _float(scenario.get("platform_edge_x_m"), 0.0)
    block_half_x = _float(scenario.get("block_half_x_m"), 0.16)

    ramp_candidates = np.flatnonzero(positions[:, index, 0] >= ramp_exit_x)
    ramp_exit_frame = int(ramp_candidates[0]) if len(ramp_candidates) else None
    if ramp_exit_frame is None and ramp_frames:
        ramp_exit_frame = min(ramp_frames[-1] + 1, len(frame_times) - 1)

    departure_candidates = np.flatnonzero(
        positions[:, index, 0] > platform_edge_x + block_half_x
    )
    platform_departure_frame = int(departure_candidates[0]) if len(departure_candidates) else None
    if platform_departure_frame is None and platform_frames:
        platform_departure_frame = min(platform_frames[-1] + 1, len(frame_times) - 1)

    landing_candidates = [
        frame for frame in ground_frames
        if platform_departure_frame is None or frame >= platform_departure_frame
    ]
    landing_frame = landing_candidates[0] if landing_candidates else None
    if landing_frame is not None:
        outcome = "ramp_platform_lands_on_ground"
        end_state = "moving_on_ground" if speeds[-1, index] > MOTION_THRESHOLD_MPS else "stopped_on_ground"
    elif platform_departure_frame is not None:
        outcome = "ramp_platform_departure_observed"
        end_state = "airborne_after_platform"
    elif ramp_exit_frame is not None:
        outcome = "ramp_platform_on_horizontal_platform"
        end_state = "moving_on_platform"
    else:
        outcome = "ramp_platform_remains_on_ramp"
        end_state = "moving_on_ramp"

    event_frame = ramp_exit_frame
    details = _common_motion_details(positions, velocities, speeds, index)
    landing_point = (
        [_round(value, 5) for value in positions[landing_frame, index]]
        if landing_frame is not None
        else None
    )
    details.update(
        {
            "actor": actor,
            "ramp_contact_first_frame": ramp_frames[0] if ramp_frames else None,
            "ramp_contact_last_frame": ramp_frames[-1] if ramp_frames else None,
            "platform_contact_first_frame": platform_frames[0] if platform_frames else None,
            "platform_contact_last_frame": platform_frames[-1] if platform_frames else None,
            "ground_contact_first_frame": ground_frames[0] if ground_frames else None,
            "ground_contact_last_frame": ground_frames[-1] if ground_frames else None,
            "ramp_exit_frame": ramp_exit_frame,
            "ramp_exit_time_s": _frame_time(ramp_exit_frame, frame_times),
            "platform_departure_frame": platform_departure_frame,
            "platform_departure_time_s": _frame_time(platform_departure_frame, frame_times),
            "landing_frame": landing_frame,
            "landing_time_s": _frame_time(landing_frame, frame_times),
            "landing_point_m": landing_point,
            "ramp_angle_deg": _round(scenario.get("ramp_angle_deg"), 2),
            "ramp_length_m": _round(scenario.get("ramp_length_m"), 3),
            "horizontal_platform_length_m": _round(scenario.get("horizontal_platform_length_m"), 3),
            "table_height_m": _round(scenario.get("table_height_m"), 3),
            "ramp_exit_x_m": _round(ramp_exit_x, 4),
            "platform_edge_x_m": _round(platform_edge_x, 4),
        }
    )
    return _base_observation(metadata, frame_times, outcome, event_frame, end_state, details)


def _derive_gap(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "gap_ball")
    actor = names[index]
    left_frames = _pair_frames(contacts, actor, "left_platform")
    right_frames = _pair_frames(contacts, actor, "right_platform")
    support_frames = _pair_frames(contacts, actor, "right_platform_support")
    ground_frames = _pair_frames(contacts, actor, "ground")
    conditioning = metadata.get("conditioning", {})
    conditioning = conditioning if isinstance(conditioning, Mapping) else {}
    raw_event = conditioning.get("first_event_frame")
    event_frame = int(raw_event) if isinstance(raw_event, (int, float)) and int(raw_event) >= 0 else None
    if event_frame is None and left_frames:
        left_set = set(left_frames)
        candidates = [frame for frame in range(left_frames[0], len(frame_times)) if frame not in left_set]
        event_frame = candidates[0] if candidates else None

    if right_frames and not ground_frames:
        outcome = "gap_crosses_to_right_platform"
        end_state = "moving_on_right_platform"
    elif ground_frames and right_frames:
        outcome = "gap_brief_right_edge_contact_then_ground"
        end_state = "moving_on_ground" if speeds[-1, index] > MOTION_THRESHOLD_MPS else "stopped_on_ground"
    elif ground_frames and support_frames:
        outcome = "gap_drops_to_ground_and_reaches_support"
        end_state = "moving_at_right_support" if speeds[-1, index] > MOTION_THRESHOLD_MPS else "stopped_at_right_support"
    elif ground_frames:
        outcome = "gap_drops_to_ground"
        end_state = "moving_on_ground" if speeds[-1, index] > MOTION_THRESHOLD_MPS else "stopped_on_ground"
    else:
        outcome = "gap_leaves_platform_and_remains_airborne"
        end_state = "airborne_after_gap"
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "left_platform_last_frame": left_frames[-1] if left_frames else None,
            "right_platform_first_frame": right_frames[0] if right_frames else None,
            "right_platform_last_frame": right_frames[-1] if right_frames else None,
            "right_support_first_frame": support_frames[0] if support_frames else None,
            "ground_contact_first_frame": ground_frames[0] if ground_frames else None,
            "gap_width_m": _round(scenario.get("gap_width_m"), 3),
        }
    )
    return _base_observation(metadata, frame_times, outcome, event_frame, end_state, details)


def _derive_obstacle(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "obstacle_ball")
    actor = names[index]
    vx = velocities[:, index, 0]
    reverse_frame = _reverse_frame(vx)
    barrier_frames = _pair_frames(contacts, actor, "obstacle_barrier")
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    start_x = float(positions[0, index, 0])
    max_x_frame = int(np.argmax(positions[:, index, 0]))
    max_x = float(positions[max_x_frame, index, 0])
    end_x = float(positions[-1, index, 0])
    if reverse_frame is None:
        outcome = "obstacle_ball_slows_before_collision"
        end_state = "slow_on_approach_side"
    elif max_x - end_x < 0.25:
        outcome = "obstacle_ball_short_rebound"
        end_state = "near_barrier_after_rebound"
    elif end_x > start_x:
        outcome = "obstacle_ball_rebounds_without_returning_to_start"
        end_state = "moving_back_left_near_barrier"
    else:
        outcome = "obstacle_ball_strong_rebound_left"
        end_state = "moving_far_back_left"
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "barrier_contact_first_frame": barrier_frames[0] if barrier_frames else None,
            "reversal_frame": reverse_frame,
            "max_x_m": _round(max_x),
            "max_x_frame": max_x_frame,
            "start_x_m": _round(start_x),
            "end_x_m": _round(end_x),
            "obstacle_x_m": _round(scenario.get("obstacle_x_m"), 3),
        }
    )
    return _base_observation(metadata, frame_times, outcome, reverse_frame, end_state, details)


def _derive_obstacle_size(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "obstacle_ball")
    actor = names[index]
    vx = velocities[:, index, 0]
    reverse_frame = _reverse_frame(vx)
    start_x = float(positions[0, index, 0])
    max_x_frame = int(np.argmax(positions[:, index, 0]))
    max_x = float(positions[max_x_frame, index, 0])
    end_x = float(positions[-1, index, 0])
    end_speed = float(speeds[-1, index])
    if reverse_frame is None:
        outcome = "obstacle_size_no_rebound"
        end_state = "no_rebound_observed"
    elif end_speed <= MOTION_THRESHOLD_MPS and end_x > start_x:
        outcome = "obstacle_size_rebounds_and_stops_before_start"
        end_state = "stopped_before_release_region"
    elif end_x > start_x:
        outcome = "obstacle_size_rebounds_near_start"
        end_state = "moving_left_near_release_region"
    else:
        outcome = "obstacle_size_rebounds_past_start"
        end_state = "moving_left_past_release_region"
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "reversal_frame": reverse_frame,
            "max_x_m": _round(max_x),
            "start_x_m": _round(start_x),
            "end_x_m": _round(end_x),
            "ball_radius_m": _round(scenario.get("ball_radius_m"), 3),
        }
    )
    return _base_observation(metadata, frame_times, outcome, reverse_frame, end_state, details)


def _derive_bowl(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "bowl_ball")
    actor = names[index]
    surface_frames = _pair_frames(contacts, actor, "bowl_surface")
    max_x_frame = int(np.argmax(positions[:, index, 0]))
    low_frame = int(np.argmin(positions[:, index, 2]))
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "surface_contact_first_frame": surface_frames[0] if surface_frames else None,
            "surface_contact_last_frame": surface_frames[-1] if surface_frames else None,
            "low_point_frame": low_frame,
            "first_turn_frame": max_x_frame,
            "bowl_radius_m": _round(scenario.get("bowl_radius_m"), 3),
        }
    )
    return _base_observation(
        metadata,
        frame_times,
        "bowl_descends_passes_low_point_and_reverses",
        max_x_frame,
        "moving_along_bowl_after_reversal" if speeds[-1, index] > MOTION_THRESHOLD_MPS else "settled_in_bowl",
        details,
    )


def _derive_pendulum(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "pendulum_bob")
    actor = names[index]
    # Initial velocity samples contain small constraint-solver transients;
    # use the first clear horizontal extremum as the swing reversal.
    turning = int(np.argmin(positions[:, index, 0]))
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "first_horizontal_reversal_frame": turning,
            "pendulum_length_m": _round(scenario.get("pendulum_length_m"), 3),
        }
    )
    return _base_observation(
        metadata,
        frame_times,
        "pendulum_swings_through_low_point",
        turning,
        "swinging_at_end" if speeds[-1, index] > MOTION_THRESHOLD_MPS else "settled_at_end",
        details,
    )


def _derive_pendulum_cabinet(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "pendulum_bob")
    actor = names[index]
    contact_frame = _first_pair_frame(contacts, actor, "pendulum_cabinet_body")
    impact_frame = contact_frame
    impact_position = positions[impact_frame, index] if impact_frame is not None else positions[-1, index]
    impact_speed = speeds[impact_frame, index] if impact_frame is not None else speeds[-1, index]
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "cabinet_contact_frame": contact_frame,
            "impact_position_m": [_round(value) for value in impact_position],
            "impact_height_m": _round(impact_position[2]),
            "impact_speed_mps": _round(impact_speed),
            "anchor_height_m": _round(scenario.get("pendulum_anchor_height_m"), 3),
        }
    )
    return _base_observation(
        metadata,
        frame_times,
        "pendulum_hits_cabinet_at_controlled_height",
        impact_frame,
        "swinging_after_cabinet_contact" if speeds[-1, index] > MOTION_THRESHOLD_MPS else "settled_after_cabinet_contact",
        details,
    )


def _derive_seesaw(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    rotations: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    board_index = names.index("seesaw_board") if "seesaw_board" in names else _dynamic_index(names, dynamic_mask, "seesaw_board")
    load_index = names.index("seesaw_load") if "seesaw_load" in names else board_index
    # The generated board rotates about world Y; this is robust to tiny solver noise.
    q = rotations[:, board_index]
    board_angles = np.degrees(2.0 * np.arctan2(q[:, 2], q[:, 0]))
    max_angle = float(np.max(np.abs(board_angles)))
    end_angle = float(board_angles[-1])
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    control = metadata.get("control", {})
    control = control if isinstance(control, Mapping) else {}
    load_x = _float(control.get("value"), _float(scenario.get("load_position_x_m")))
    if abs(load_x) < 0.01 and max_angle < 1.0:
        outcome = "seesaw_centered_load_stays_level"
        end_state = "level_and_stationary"
        event_frame = None
    else:
        outcome = "seesaw_offcenter_load_tips_and_settles"
        end_state = "tilted_and_stationary" if speeds[-1, load_index] <= MOTION_THRESHOLD_MPS else "tilted_and_moving"
        event_frame = int(np.argmax(np.abs(board_angles)))
    details = _common_motion_details(positions, velocities, speeds, load_index)
    details.update(
        {
            "actor": names[load_index],
            "board_actor": names[board_index],
            "load_position_x_m": _round(load_x, 4),
            "maximum_board_angle_deg": _round(max_angle, 3),
            "final_board_angle_deg": _round(end_angle, 3),
        }
    )
    return _base_observation(metadata, frame_times, outcome, event_frame, end_state, details)


def _derive_domino(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    rotations: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    domino_indices = [index for index, name in enumerate(names) if name.startswith("domino_") and name != "domino_trigger_ball"]
    final_angles = []
    for index in domino_indices:
        q = rotations[-1, index]
        final_angles.append(abs(math.degrees(2.0 * math.atan2(float(q[2]), float(q[0])))))
    toppled = [angle >= 25.0 for angle in final_angles]
    count = sum(toppled)
    if count == len(domino_indices):
        outcome = "domino_chain_reaches_all_five"
        end_state = "all_dominoes_toppled"
    elif count == 0 and max(final_angles, default=0.0) < 3.0:
        outcome = "domino_chain_does_not_start"
        end_state = "dominoes_remain_upright"
    elif count == 0:
        outcome = "domino_chain_only_nudges_row"
        end_state = "dominoes_only_slightly_tilted"
    elif count == 1 and toppled[-1]:
        outcome = "domino_last_only_topples"
        end_state = "last_domino_toppled_only"
    else:
        outcome = "domino_chain_partial_or_anomalous"
        end_state = "partial_topple"
    ball_index = names.index("domino_trigger_ball") if "domino_trigger_ball" in names else domino_indices[0]
    details = _common_motion_details(positions, velocities, speeds, ball_index)
    details.update(
        {
            "actor": names[ball_index],
            "toppled_count": int(count),
            "domino_count": len(domino_indices),
            "final_domino_tilt_angles_deg": [_round(angle, 2) for angle in final_angles],
            "first_domino_contact_frame": _first_pair_frame(contacts, names[domino_indices[0]], "domino_trigger_ball") if domino_indices else None,
        }
    )
    return _base_observation(metadata, frame_times, outcome, details["first_domino_contact_frame"], end_state, details)


def _derive_puck(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    index = _dynamic_index(names, dynamic_mask, "puck")
    actor = names[index]
    vx = velocities[:, index, 0]
    vy = velocities[:, index, 1]
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    barrier_frames = _pair_frames(contacts, actor, "puck_barrier")
    impact_candidates = np.flatnonzero(positions[:, index, 0] >= 0.25)
    event_frame = barrier_frames[0] if barrier_frames else (int(impact_candidates[0]) if len(impact_candidates) else None)
    end_vx = float(vx[-1])
    end_vy = float(vy[-1])
    if end_vx > 0.4:
        outcome = "puck_rebounds_forward_transverse"
        end_state = "moving_forward_with_strong_transverse_deflection"
    elif abs(end_vx) < 0.25:
        outcome = "puck_rebounds_almost_sideways"
        end_state = "moving_almost_transversely"
    elif end_vx < -1.15 and abs(end_vy) < 0.25:
        outcome = "puck_rebounds_straight_back"
        end_state = "moving_back_along_incoming_line"
    elif end_vx < -0.85:
        outcome = "puck_rebounds_mostly_back_with_lateral_deflection"
        end_state = "moving_back_with_lateral_deflection"
    else:
        outcome = "puck_rebounds_diagonally_back"
        end_state = "moving_diagonally_back"
    angle = math.degrees(math.atan2(end_vy, end_vx))
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "barrier_contact_first_frame": barrier_frames[0] if barrier_frames else None,
            "rebound_angle_world_xy_deg": _round(angle, 2),
            "final_velocity_xy_mps": [_round(end_vx), _round(end_vy)],
            "barrier_normal_angle_deg": _round(scenario.get("barrier_normal_angle_deg"), 2),
        }
    )
    return _base_observation(metadata, frame_times, outcome, event_frame, end_state, details)


def _derive_door(
    metadata: Mapping[str, object],
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    rotations: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
    ball: bool,
) -> dict[str, object]:
    preferred = "door_ball" if ball else "door_crate"
    index = _dynamic_index(names, dynamic_mask, preferred)
    actor = names[index]
    left_frames = _pair_frames(contacts, actor, "door_frame_left")
    right_frames = _pair_frames(contacts, actor, "door_frame_right")
    frame_contacts = sorted(set(left_frames + right_frames))
    scenario = metadata.get("scenario_spec", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    center_x = _float(scenario.get("door_frame_center_x_m"), 0.72)
    passed = float(positions[-1, index, 0]) > center_x + 0.65
    final_speed = float(speeds[-1, index])
    if passed:
        if frame_contacts:
            if ball and len(left_frames) and len(right_frames):
                outcome = "door_ball_touches_both_posts_then_passes"
                end_state = "passed_after_two_post_contacts"
            elif ball and len(frame_contacts):
                outcome = "door_ball_deflects_then_passes"
                end_state = "passed_after_post_deflection"
            else:
                outcome = "door_crate_passes_after_frame_contact"
                end_state = "passed_after_frame_contact"
        else:
            outcome = "door_object_passes_clear"
            end_state = "clear_passage"
    elif frame_contacts:
        if ball:
            outcome = "door_ball_stays_in_frame_and_rotates"
            end_state = "stuck_rotating_in_doorway" if final_speed > MOTION_THRESHOLD_MPS else "stuck_in_doorway"
        elif len(left_frames) and len(right_frames):
            outcome = "door_crate_contacts_both_posts_and_stays_blocked"
            end_state = "blocked_after_both_post_contacts"
        elif final_speed > MOTION_THRESHOLD_MPS:
            outcome = "door_crate_contacts_post_and_stays_blocked"
            end_state = "blocked_and_slowly_moving"
        else:
            outcome = "door_crate_contacts_post_and_stops"
            end_state = "blocked_and_stopped"
    else:
        outcome = "door_object_does_not_reach_frame"
        end_state = "before_doorway"
    event_frame = frame_contacts[0] if frame_contacts else None
    details = _common_motion_details(positions, velocities, speeds, index)
    details.update(
        {
            "actor": actor,
            "left_post_contact_first_frame": left_frames[0] if left_frames else None,
            "right_post_contact_first_frame": right_frames[0] if right_frames else None,
            "frame_contact_last_frame": frame_contacts[-1] if frame_contacts else None,
            "passed_frame": bool(passed),
            "opening_width_m": _round(scenario.get("door_opening_width_m"), 3),
            "final_x_m": _round(positions[-1, index, 0]),
            "final_y_m": _round(positions[-1, index, 1]),
        }
    )
    return _base_observation(metadata, frame_times, outcome, event_frame, end_state, details)


def derive_caption_observations_from_arrays(
    metadata: Mapping[str, object],
    *,
    frame_times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    speeds: np.ndarray,
    rotations: np.ndarray,
    names: Sequence[str],
    dynamic_mask: np.ndarray,
    contacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    family = str(metadata.get("family_key", ""))
    task_type = str(metadata.get("task_type", ""))
    if family == "F11" or task_type == "table_rolloff":
        return _derive_f11(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family in {"F12", "F12_RAMP_LENGTH"} or task_type in {"incline_release", "incline_length_release"}:
        return _derive_ramp(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_RAMP_PLATFORM" or task_type == "incline_to_platform":
        return _derive_ramp_platform(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_GAP" or task_type == "gap_rolloff":
        return _derive_gap(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_OBSTACLE_SIZE":
        return _derive_obstacle_size(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_OBSTACLE" or task_type == "obstacle_collision":
        return _derive_obstacle(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_BOWL" or task_type == "bowl_descent":
        return _derive_bowl(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_PENDULUM_CABINET" or task_type == "pendulum_cabinet_collision":
        return _derive_pendulum_cabinet(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_PENDULUM" or task_type == "pendulum_swing":
        return _derive_pendulum(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "V2V_SEESAW" or task_type == "seesaw_rotation":
        return _derive_seesaw(metadata, frame_times, positions, velocities, speeds, rotations, names, dynamic_mask, contacts)
    if family == "V2V_DOMINO" or task_type == "domino_chain":
        return _derive_domino(metadata, frame_times, positions, velocities, speeds, rotations, names, dynamic_mask, contacts)
    if family == "SCENE_PUCK_BARRIER" or task_type == "puck_barrier_collision":
        return _derive_puck(metadata, frame_times, positions, velocities, speeds, names, dynamic_mask, contacts)
    if family == "SCENE_DOOR_FRAME_BALL" or task_type == "door_frame_clearance_ball":
        return _derive_door(metadata, frame_times, positions, velocities, speeds, rotations, names, dynamic_mask, contacts, True)
    if family == "SCENE_DOOR_FRAME" or task_type == "door_frame_clearance":
        return _derive_door(metadata, frame_times, positions, velocities, speeds, rotations, names, dynamic_mask, contacts, False)
    return _base_observation(metadata, frame_times, "unclassified_motion", None, "observed", {})


def derive_caption_observations(sample_dir: Path, metadata: Mapping[str, object] | None = None) -> dict[str, object]:
    """Load exported supervision and return structured, JSON-safe observations."""
    if metadata is None:
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    payload = np.load(sample_dir / "physics_supervision.npz", allow_pickle=False)
    contacts = json.loads((sample_dir / "contacts.json").read_text(encoding="utf-8"))
    names = [_decode(value) for value in payload["object_names"]]
    observations = derive_caption_observations_from_arrays(
        metadata,
        frame_times=np.asarray(payload["frame_times_s"]),
        positions=np.asarray(payload["positions_m"]),
        velocities=np.asarray(payload["linear_velocity_mps"]),
        speeds=np.asarray(payload["speed_mps"]),
        rotations=np.asarray(payload["rotations_wxyz"]),
        names=names,
        dynamic_mask=np.asarray(payload["dynamic_mask"], dtype=bool),
        contacts=contacts,
    )
    return observations
