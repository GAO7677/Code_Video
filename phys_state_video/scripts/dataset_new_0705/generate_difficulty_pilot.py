"""Render a small scene-first difficulty pilot for visual inspection.

The difficulty level describes scene complexity. Motion labels are derived from
the saved simulator states and are deliberately kept separate from the scene
family's intended mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from .common_specs import ScenarioBlueprint
from .render_sim_0705 import render_blueprint_case
from .scene_generators_0705 import (
    DEFAULT_CAMERA_DISTANCE_SCALE,
    F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG,
    build_scenario_family_catalog,
    generate_scenario_blueprint,
)

try:
    from .. import generate_sim_preview_gallery as legacy
except ImportError:  # pragma: no cover - direct script fallback
    import generate_sim_preview_gallery as legacy


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_difficulty_pilot_20260817"
)
ANALYSIS_QUESTION = (
    "Describe this video with emphasis on its physical dynamics. Focus on "
    "object translation, rotation, contacts, interactions, and how motion "
    "changes over time, including acceleration, deceleration, bouncing, "
    "rolling, sliding, rocking, oscillation, and settling when present. "
    "Describe events in temporal order and include the motion state near the "
    "end of the video. Minimize appearance details unrelated to the physical "
    "behavior. Only describe visually observed events and do not predict what "
    "happens after the video ends."
)


DIFFICULTY_LEVELS = {
    "L1": {
        "title": "基础场景",
        "description": "单物体或单一环境机制，主要考察可见的平移、旋转和基础接触。",
        "families": ("F1", "F6", "F7"),
    },
    "L2": {
        "title": "多阶段场景",
        "description": "支撑、边界或弹性接触造成多个连续阶段，但交互对象数量有限。",
        "families": ("F5", "F8", "F10"),
    },
    "L3": {
        "title": "交互场景",
        "description": "多个动态物体、遮挡或局部拥挤环境，要求理解因果和事件顺序。",
        "families": ("F2", "F3", "F4", "F9"),
    },
}

TABLE_HEIGHT_CONTROL_CASES = (
    {"table_height_m": 0.30, "height_label": "low", "travel_angle_deg": F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG, "angle_label": "sr048"},
    {"table_height_m": 0.58, "height_label": "low_mid", "travel_angle_deg": F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG, "angle_label": "sr048"},
    {"table_height_m": 0.85, "height_label": "mid", "travel_angle_deg": F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG, "angle_label": "sr048"},
    {"table_height_m": 1.12, "height_label": "high_mid", "travel_angle_deg": F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG, "angle_label": "sr048"},
    {"table_height_m": 1.40, "height_label": "high", "travel_angle_deg": F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG, "angle_label": "sr048"},
)

# Direction variants remain a separate F11 view and are intentionally not
# included in the V2V short-context overview.
TABLE_DIRECTION_VARIANT_CASES = (
    {"table_height_m": 0.85, "height_label": "mid", "travel_angle_deg": -12.0, "angle_label": "sr012"},
    {"table_height_m": 0.85, "height_label": "mid", "travel_angle_deg": -18.0, "angle_label": "sr018"},
    {"table_height_m": 0.85, "height_label": "mid", "travel_angle_deg": -24.0, "angle_label": "sr024"},
    {"table_height_m": 0.85, "height_label": "mid", "travel_angle_deg": -30.0, "angle_label": "sr030"},
    {"table_height_m": 0.85, "height_label": "mid", "travel_angle_deg": -36.0, "angle_label": "sr036"},
)

TABLE_ROLLOFF_CASES = TABLE_HEIGHT_CONTROL_CASES + TABLE_DIRECTION_VARIANT_CASES

RAMP_INCLINE_CASES = (
    {"ramp_angle_deg": 8.0, "angle_label": "a008", "angle_name": "shallow"},
    {"ramp_angle_deg": 16.0, "angle_label": "a016", "angle_name": "moderate_shallow"},
    {"ramp_angle_deg": 24.0, "angle_label": "a024", "angle_name": "moderate"},
    {"ramp_angle_deg": 33.0, "angle_label": "a033", "angle_name": "steep"},
    {"ramp_angle_deg": 42.0, "angle_label": "a042", "angle_name": "very_steep"},
)

RAMP_LENGTH_CONTROL_CASES = (
    {"ramp_length_m": 0.80, "length_label": "l080", "ramp_angle_deg": 24.0},
    {"ramp_length_m": 1.10, "length_label": "l110", "ramp_angle_deg": 24.0},
    {"ramp_length_m": 1.40, "length_label": "l140", "ramp_angle_deg": 24.0},
    {"ramp_length_m": 1.70, "length_label": "l170", "ramp_angle_deg": 24.0},
    {"ramp_length_m": 2.00, "length_label": "l200", "ramp_angle_deg": 24.0},
)


def _decode_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _direction_reversals(velocities: np.ndarray, threshold: float = 0.05) -> int:
    planar = np.asarray(velocities[:, :2], dtype=np.float64)
    speed = np.linalg.norm(planar, axis=1)
    valid = speed >= threshold
    directions = planar[valid]
    if len(directions) < 3:
        return 0
    dots = np.sum(directions[:-1] * directions[1:], axis=1)
    return int(np.count_nonzero(dots < -0.15))


def _summarize_states(states_path: Path) -> dict[str, object]:
    payload = np.load(states_path, allow_pickle=True)
    positions = np.asarray(payload["positions"], dtype=np.float64)
    linear = np.asarray(payload["linear_velocities"], dtype=np.float64)
    angular = np.asarray(payload["angular_velocities"], dtype=np.float64)
    frame_times = np.asarray(payload["frame_times"], dtype=np.float64)
    names = [_decode_name(value) for value in payload["object_names"]]

    if len(frame_times) > 1:
        duration_s = float(frame_times[-1] - frame_times[0])
    else:
        duration_s = 0.0

    objects: list[dict[str, object]] = []
    all_max_speed = 0.0
    any_motion = False
    for index, name in enumerate(names):
        object_positions = positions[:, index]
        object_linear = linear[:, index]
        object_angular = angular[:, index]
        speed = np.linalg.norm(object_linear, axis=1)
        angular_speed = np.linalg.norm(object_angular, axis=1)
        segment_lengths = np.linalg.norm(np.diff(object_positions, axis=0), axis=1)
        path_length = float(segment_lengths.sum())
        displacement = float(np.linalg.norm(object_positions[-1] - object_positions[0]))
        max_speed = float(speed.max(initial=0.0))
        final_speed = float(speed[-1])
        max_angular_speed = float(angular_speed.max(initial=0.0))
        final_angular_speed = float(angular_speed[-1])
        direction_reversals = _direction_reversals(object_linear)
        height_range = float(object_positions[:, 2].max() - object_positions[:, 2].min())
        moving = max(max_speed, max_angular_speed * 0.05, height_range) > 0.05
        any_motion = any_motion or moving
        all_max_speed = max(all_max_speed, max_speed)
        objects.append(
            {
                "name": name,
                "path_length_m": round(path_length, 5),
                "net_displacement_m": round(displacement, 5),
                "max_speed_mps": round(max_speed, 5),
                "final_speed_mps": round(final_speed, 5),
                "max_angular_speed_radps": round(max_angular_speed, 5),
                "final_angular_speed_radps": round(final_angular_speed, 5),
                "height_range_m": round(height_range, 5),
                "direction_reversals": direction_reversals,
                "derived_motion_present": bool(moving),
            }
        )

    final_motion = "moving" if any(
        float(item["final_speed_mps"]) > 0.08
        or float(item["final_angular_speed_radps"]) > 0.35
        for item in objects
    ) else "nearly_stationary"
    return {
        "frames": int(len(frame_times)),
        "duration_s": round(duration_s, 4),
        "objects": objects,
        "max_linear_speed_mps": round(all_max_speed, 5),
        "final_motion_state": final_motion,
        "derived_motion_present": bool(any_motion),
        "derivation": "state_trajectory_summary_only; not a semantic event label",
    }


def _family_summary(family_key: str) -> dict[str, object]:
    family = build_scenario_family_catalog()[family_key]
    return {
        "family_key": family_key,
        "title": family.title,
        "description": family.description,
        "target_event_types": list(family.target_event_types),
        "target_event_types_are": "scene_mechanisms_for_sampling_only",
    }


def _scenario_objects(blueprint: ScenarioBlueprint) -> list[dict[str, object]]:
    return [
        {
            "name": obj.name,
            "family_key": obj.family_key,
            "shape": obj.shape,
            "dynamic": bool(obj.dynamic),
            "role": obj.role,
            "mass_kg": round(float(obj.mass), 5),
            "friction": round(float(obj.friction), 5),
            "restitution": round(float(obj.restitution), 5),
            "position": [round(float(value), 5) for value in obj.position],
            "linear_velocity": [round(float(value), 5) for value in obj.linear_velocity],
            "angular_velocity": [round(float(value), 5) for value in obj.angular_velocity],
        }
        for obj in blueprint.objects
    ]


def _export_context_videos(video_path: Path, case_root: Path, case_id: str) -> tuple[Path, Path]:
    """Export the same first 8/16 frames used by the V2V viewer controls."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open rendered video for context export: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < 16:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 16:
        raise RuntimeError(
            f"{case_id}: expected at least 16 frames for short-context controls, got {len(frames)}"
        )
    context_root = case_root / "context"
    context_root.mkdir(parents=True, exist_ok=True)
    context_path = context_root / f"{case_id}_context8f.mp4"
    context16_path = context_root / f"{case_id}_context16f.mp4"
    legacy._write_video_h264(context_path, frames[:8])
    legacy._write_video_h264(context16_path, frames[:16])
    return context_path, context16_path


def _control_first_event_frame(blueprint: ScenarioBlueprint, states_path: Path) -> int | None:
    payload = np.load(states_path, allow_pickle=True)
    positions = np.asarray(payload["positions"], dtype=np.float64)
    names = [_decode_name(value) for value in payload["object_names"]]
    index = {name: idx for idx, name in enumerate(names)}
    if blueprint.family_key == "F11":
        mover = next(obj for obj in blueprint.objects if obj.name == "roller_0")
        threshold = float(blueprint.metadata["table_top_half_width_m"])
        frames = np.flatnonzero(positions[:, index[mover.name], 0] > threshold)
    elif blueprint.family_key == "F12":
        block = next(obj for obj in blueprint.objects if obj.name == "block_0")
        theta = math.radians(float(blueprint.metadata["ramp_angle_deg"]))
        ramp_edge_x = 0.5 * float(blueprint.metadata["ramp_length_m"]) * math.cos(theta)
        frames = np.flatnonzero(positions[:, index[block.name], 0] > ramp_edge_x - 0.02)
    else:
        return None
    return int(frames[0]) if len(frames) else None


def _control_visibility_contract(
    blueprint: ScenarioBlueprint,
    mask_ids_path: Path,
) -> dict[str, object]:
    payload = np.load(mask_ids_path, allow_pickle=True)
    mask_ids = np.asarray(payload["instance_ids"])
    object_names = [_decode_name(value) for value in payload["object_names"]]
    object_ids = [int(value) for value in payload["object_ids"]]
    visible_by_name = {
        name: bool(np.all(np.any(mask_ids == object_id, axis=(1, 2))))
        for name, object_id in zip(object_names, object_ids)
    }
    dynamic_names = [obj.name for obj in blueprint.objects if obj.dynamic]
    dynamic_visibility = {
        name: bool(visible_by_name.get(name, False)) for name in dynamic_names
    }
    return {
        # The training-relevant contract is that every moving object remains
        # in frame; static fixture pixel visibility is reported separately.
        "all_objects_visible_every_frame": bool(dynamic_visibility and all(dynamic_visibility.values())),
        "dynamic_object_visibility": dynamic_visibility,
        "rendered_object_visibility": visible_by_name,
        "rendered_objects_visible_every_frame": bool(visible_by_name and all(visible_by_name.values())),
    }


def _control_context_artifacts(
    blueprint: ScenarioBlueprint,
    render_manifest: dict[str, object],
    case_root: Path,
    case_id: str,
) -> dict[str, object]:
    context_path, context16_path = _export_context_videos(
        Path(str(render_manifest["video"])), case_root, case_id
    )
    first_event_frame = _control_first_event_frame(
        blueprint, Path(str(render_manifest["states"]))
    )
    visibility = _control_visibility_contract(
        blueprint, Path(str(render_manifest["mask_ids"]))
    )
    controlled_variable = blueprint.metadata.get("controlled_variable")
    if not controlled_variable:
        controlled_variable = {
            "F11": "table_height_m",
            "F12": "ramp_angle_deg",
        }.get(blueprint.family_key)
    return {
        "context_video": str(context_path),
        "context16_video": str(context16_path),
        "v2v": {
            "short_context_control_group": True,
            "context_frames": 8,
            "context_duration_s": round(8 / 30, 4),
            "context_frame_options": [8, 16],
            "context16_duration_s": round(16 / 30, 4),
            "controlled_variable": controlled_variable,
            "controlled_value": blueprint.metadata.get(
                "table_height_m", blueprint.metadata.get("ramp_angle_deg")
            ),
            "controlled_value_label": (
                f"{float(blueprint.metadata['table_height_m']):.2f} m"
                if blueprint.family_key == "F11"
                else f"{float(blueprint.metadata['ramp_angle_deg']):.0f}°"
            ),
            "event_rule": (
                "ball_crosses_table_right_edge"
                if blueprint.family_key == "F11"
                else "block_exits_ramp_lower_edge"
            ),
            "first_event_frame": first_event_frame,
            "first_event_time_s": round(first_event_frame / 30, 4) if first_event_frame is not None else None,
            "event_after_context": bool(first_event_frame is not None and first_event_frame >= 8),
            "event_after_context16": bool(first_event_frame is not None and first_event_frame >= 16),
            **visibility,
        },
    }


def _load_existing_rows(results_path: Path) -> list[dict[str, object]]:
    if not results_path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid existing results at {results_path}:{line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError(f"invalid result row at {results_path}:{line_number}")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scene-first difficulty pilot cases.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per-level", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed-base", type=int, default=20260817)
    parser.add_argument("--size-scale", type=float, default=1.0)
    parser.add_argument("--camera-distance-scale", type=float, default=DEFAULT_CAMERA_DISTANCE_SCALE)
    parser.add_argument("--scene-style", type=str, default="indoor_realistic")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-control-group",
        choices=("f11", "f12"),
        help="Render only one deterministic control group instead of the full pilot.",
    )
    parser.add_argument(
        "--append-existing-results",
        action="store_true",
        help="Merge newly rendered control cases into an existing cases.jsonl at --output-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_level <= 0:
        raise ValueError("--per-level must be positive")
    if args.overwrite and args.append_existing_results:
        raise ValueError("--overwrite cannot be combined with --append-existing-results")
    output_root = args.output_root
    results_path = output_root / "cases.jsonl"
    existing_rows = _load_existing_rows(results_path) if args.append_existing_results else []
    if args.only_control_group and results_path.exists() and not args.append_existing_results:
        raise FileExistsError(
            f"{results_path} already exists; pass --append-existing-results to preserve its rows"
        )
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    family_catalog = build_scenario_family_catalog()
    shared_table_speed = 1.25

    case_index = 0
    base_levels = DIFFICULTY_LEVELS.items() if args.only_control_group is None else ()
    for level_key, level in base_levels:
        families = list(level["families"])
        for local_index in range(args.per_level):
            family_key = families[local_index % len(families)]
            case_id = f"difficulty_{level_key.lower()}_{family_key.lower()}_{local_index:03d}"
            seed = int(args.seed_base + case_index * 1009)
            case_root = output_root / "cases" / level_key / family_key / case_id
            case_index += 1
            try:
                if case_root.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"case exists; pass --overwrite for a fresh pilot: {case_root}"
                    )
                blueprint = generate_scenario_blueprint(
                    family_key=family_key,
                    sample_key=case_id,
                    seed=seed,
                    size_scale=args.size_scale,
                    camera_distance_scale=args.camera_distance_scale,
                )
                render_manifest = render_blueprint_case(
                    blueprint=blueprint,
                    seed=seed,
                    output_root=case_root,
                    width=args.width,
                    height=args.height,
                    scene_style=args.scene_style,
                    export_instance_masks=True,
                    preserve_states=True,
                )
                state_summary = _summarize_states(Path(render_manifest["states"]))
                family_summary = _family_summary(family_key)
                difficulty = {
                    "level": level_key,
                    "title": level["title"],
                    "description": level["description"],
                    "priority": ["L1", "L2", "L3"].index(level_key) + 1,
                }
                pilot_metadata = {
                    "difficulty": difficulty,
                    "scene_family": family_summary,
                    "state_summary": state_summary,
                    "initialization_qa": render_manifest["initialization_qa"],
                    "scene_style": args.scene_style,
                    "scenario_spec": {
                        "surface_key": blueprint.surface_key,
                        "camera_key": blueprint.camera_key,
                        "lighting_key": blueprint.lighting_key,
                        "scene_style": args.scene_style,
                        "floor_restitution": round(float(blueprint.metadata.get("floor_restitution", 0.02)), 5),
                        "initialization_qa": render_manifest["initialization_qa"],
                        "objects": [
                            {
                                "name": obj.name,
                                "family_key": obj.family_key,
                                "shape": obj.shape,
                                "dynamic": bool(obj.dynamic),
                                "role": obj.role,
                                "mass_kg": round(float(obj.mass), 5),
                                "friction": round(float(obj.friction), 5),
                                "restitution": round(float(obj.restitution), 5),
                                "position": [round(float(value), 5) for value in obj.position],
                                "linear_velocity": [round(float(value), 5) for value in obj.linear_velocity],
                                "angular_velocity": [round(float(value), 5) for value in obj.angular_velocity],
                            }
                            for obj in blueprint.objects
                        ],
                    },
                    "label_policy": "scene mechanism is not an observed motion label",
                }
                meta_path = Path(render_manifest["meta"])
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_payload["difficulty_pilot"] = pilot_metadata
                meta_path.write_text(
                    json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                manifest_path = case_root / "case_manifest.json"
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_payload["difficulty_pilot"] = pilot_metadata
                manifest_path.write_text(
                    json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                rows.append(
                    {
                        "case_id": case_id,
                        "status": "rendered",
                        "difficulty": difficulty,
                        "scene_family": family_summary,
                        "scene_title": blueprint.title,
                        "scene_description": blueprint.description,
                        "scene_style": args.scene_style,
                        "scenario_spec": pilot_metadata["scenario_spec"],
                        "state_summary": state_summary,
                        "initialization_qa": render_manifest["initialization_qa"],
                        "video": render_manifest["video"],
                        "video_url": "/media/" + case_id,
                        "meta": render_manifest["meta"],
                        "states": render_manifest["states"],
                        "mask_video": render_manifest["mask_video"],
                        "mask_ids": render_manifest["mask_ids"],
                        "question": ANALYSIS_QUESTION,
                        "response_final": "本 pilot 只展示仿真视频和状态摘要，尚未运行 VLM。",
                        "response_raw": "",
                        "answer_source": "simulation_pilot",
                        "caption_intent_only": render_manifest["caption"],
                    }
                )
                print(f"rendered {case_id} -> {render_manifest['video']}", flush=True)
            except Exception as exc:  # pragma: no cover - batch guard
                failures.append(
                    {
                        "case_id": case_id,
                        "difficulty": level_key,
                        "family_key": family_key,
                        "seed": seed,
                        "error": repr(exc),
                    }
                )
                print(f"failed {case_id}: {exc!r}", flush=True)

    f11_cases = TABLE_ROLLOFF_CASES if args.only_control_group in (None, "f11") else ()
    for extra in f11_cases:
        family_key = "F11"
        table_height_m = float(extra["table_height_m"])
        travel_angle_deg = float(extra["travel_angle_deg"])
        case_id = f"difficulty_l2_f11_h{int(round(table_height_m * 100)):03d}_{extra['angle_label']}"
        seed = int(args.seed_base + 88000)
        case_root = output_root / "cases" / "L2" / family_key / case_id
        try:
            if case_root.exists() and not args.overwrite:
                raise FileExistsError(
                    f"case exists; pass --overwrite for a fresh pilot: {case_root}"
                )
            blueprint = generate_scenario_blueprint(
                family_key=family_key,
                sample_key=case_id,
                seed=seed,
                direction_mode="left_to_right",
                size_scale=args.size_scale,
                camera_distance_scale=args.camera_distance_scale,
                table_height_m=table_height_m,
                initial_speed_mps=shared_table_speed,
                travel_angle_deg=travel_angle_deg,
            )
            render_manifest = render_blueprint_case(
                blueprint=blueprint,
                seed=seed,
                output_root=case_root,
                width=args.width,
                height=args.height,
                scene_style="indoor_realistic",
                export_instance_masks=True,
                preserve_states=True,
            )
            context_artifacts = (
                _control_context_artifacts(blueprint, render_manifest, case_root, case_id)
                if extra["angle_label"] == "sr048"
                else None
            )
            state_summary = _summarize_states(Path(render_manifest["states"]))
            family_summary = _family_summary(family_key)
            difficulty = {
                "level": "L2",
                "title": "桌面滚落",
                "description": "同物体、同初速度在不同桌高和不同屏幕右向斜率上滚动并越过近侧桌缘，重点观察落体时机、反弹和轨迹方向变化。",
                "priority": 2,
            }
            pilot_metadata = {
                "difficulty": difficulty,
                "scene_family": family_summary,
                "state_summary": state_summary,
                "initialization_qa": render_manifest["initialization_qa"],
                "scene_style": "indoor_realistic",
                "scenario_spec": {
                    "surface_key": blueprint.surface_key,
                    "camera_key": blueprint.camera_key,
                    "lighting_key": blueprint.lighting_key,
                    "scene_style": "indoor_realistic",
                    "table_height_m": round(table_height_m, 5),
                    "initial_speed_mps": round(shared_table_speed, 5),
                    "travel_angle_deg": round(travel_angle_deg, 5),
                    "travel_direction_xy": blueprint.metadata.get("travel_direction_xy"),
                    "floor_restitution": round(float(blueprint.metadata.get("floor_restitution", 0.02)), 5),
                    "controlled_variable": "table_height_m",
                    "controlled_value": round(table_height_m, 5),
                    "controlled_value_label": f"{table_height_m:.2f} m",
                    "initialization_qa": render_manifest["initialization_qa"],
                    "table_height_label": extra["height_label"],
                    "angle_label": extra["angle_label"],
                    "objects": _scenario_objects(blueprint),
                },
                "label_policy": "scene mechanism is not an observed motion label",
            }
            if context_artifacts is not None:
                pilot_metadata["v2v"] = context_artifacts["v2v"]
            meta_path = Path(render_manifest["meta"])
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_payload["difficulty_pilot"] = pilot_metadata
            meta_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest_path = case_root / "case_manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["difficulty_pilot"] = pilot_metadata
            manifest_path.write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            rows.append(
                {
                    "case_id": case_id,
                    "status": "rendered",
                    "difficulty": difficulty,
                    "scene_family": family_summary,
                    "scene_title": blueprint.title,
                    "scene_description": blueprint.description,
                    "scene_style": "indoor_realistic",
                    "scenario_spec": pilot_metadata["scenario_spec"],
                    "state_summary": state_summary,
                    "initialization_qa": render_manifest["initialization_qa"],
                        "video": render_manifest["video"],
                        "video_url": "/media/" + case_id,
                        **(
                            {
                                "context_video": context_artifacts["context_video"],
                                "context16_video": context_artifacts["context16_video"],
                                "context_video_url": "/media-context/" + case_id,
                                "context16_video_url": "/media-context16/" + case_id,
                                "v2v": context_artifacts["v2v"],
                            }
                            if context_artifacts is not None
                            else {}
                        ),
                        "meta": render_manifest["meta"],
                    "states": render_manifest["states"],
                    "mask_video": render_manifest["mask_video"],
                    "mask_ids": render_manifest["mask_ids"],
                    "question": ANALYSIS_QUESTION,
                    "response_final": "本 pilot 只展示仿真视频和状态摘要，尚未运行 VLM。",
                    "response_raw": "",
                    "answer_source": "simulation_pilot",
                    "caption_intent_only": render_manifest["caption"],
                }
            )
            print(f"rendered {case_id} -> {render_manifest['video']}", flush=True)
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append(
                {
                    "case_id": case_id,
                    "difficulty": "L2",
                    "family_key": family_key,
                    "seed": seed,
                    "error": repr(exc),
                }
            )
            print(f"failed {case_id}: {exc!r}", flush=True)

    f12_cases = RAMP_INCLINE_CASES if args.only_control_group in (None, "f12") else ()
    for extra in f12_cases:
        family_key = "F12"
        ramp_angle_deg = float(extra["ramp_angle_deg"])
        case_id = f"difficulty_l2_f12_{extra['angle_label']}"
        seed = int(args.seed_base + 99000)
        case_root = output_root / "cases" / "L2" / family_key / case_id
        try:
            if case_root.exists() and not args.overwrite:
                raise FileExistsError(
                    f"case exists; pass --overwrite for a fresh pilot: {case_root}"
                )
            blueprint = generate_scenario_blueprint(
                family_key=family_key,
                sample_key=case_id,
                seed=seed,
                direction_mode="left_to_right",
                size_scale=args.size_scale,
                camera_distance_scale=args.camera_distance_scale,
                ramp_angle_deg=ramp_angle_deg,
            )
            render_manifest = render_blueprint_case(
                blueprint=blueprint,
                seed=seed,
                output_root=case_root,
                width=args.width,
                height=args.height,
                scene_style="indoor_realistic",
                export_instance_masks=True,
                preserve_states=True,
            )
            context_artifacts = _control_context_artifacts(
                blueprint, render_manifest, case_root, case_id
            )
            state_summary = _summarize_states(Path(render_manifest["states"]))
            family_summary = _family_summary(family_key)
            difficulty = {
                "level": "L2",
                "title": "斜面释放",
                "description": "同一红色木块从静止在不同坡角的动态支撑斜板上释放，重点观察重力驱动的滑动、翻滚、斜面退出和末态。",
                "priority": 2,
            }
            pilot_metadata = {
                "difficulty": difficulty,
                "scene_family": family_summary,
                "state_summary": state_summary,
                "initialization_qa": render_manifest["initialization_qa"],
                "scene_style": "indoor_realistic",
                "scenario_spec": {
                    "surface_key": blueprint.surface_key,
                    "camera_key": blueprint.camera_key,
                    "lighting_key": blueprint.lighting_key,
                    "scene_style": "indoor_realistic",
                    "ramp_angle_deg": round(ramp_angle_deg, 5),
                    "ramp_angle_label": extra["angle_label"],
                    "ramp_angle_name": extra["angle_name"],
                    "ramp_length_m": round(float(blueprint.metadata["ramp_length_m"]), 5),
                    "initial_speed_mps": 0.0,
                    "released_from_rest": True,
                    "controlled_variable": "ramp_angle_deg",
                    "controlled_value": round(ramp_angle_deg, 5),
                    "controlled_value_label": f"{ramp_angle_deg:.0f}°",
                    "support_mode": blueprint.metadata["support_mode"],
                    "floor_restitution": round(float(blueprint.metadata.get("floor_restitution", 0.02)), 5),
                    "initialization_qa": render_manifest["initialization_qa"],
                    "objects": _scenario_objects(blueprint),
                },
                "label_policy": "scene mechanism is not an observed motion label",
                "v2v": context_artifacts["v2v"],
            }
            meta_path = Path(render_manifest["meta"])
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_payload["difficulty_pilot"] = pilot_metadata
            meta_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest_path = case_root / "case_manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["difficulty_pilot"] = pilot_metadata
            manifest_path.write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            rows.append(
                {
                    "case_id": case_id,
                    "status": "rendered",
                    "difficulty": difficulty,
                    "scene_family": family_summary,
                    "scene_title": blueprint.title,
                    "scene_description": blueprint.description,
                    "scene_style": "indoor_realistic",
                    "scenario_spec": pilot_metadata["scenario_spec"],
                    "state_summary": state_summary,
                    "initialization_qa": render_manifest["initialization_qa"],
                    "video": render_manifest["video"],
                    "video_url": "/media/" + case_id,
                    "context_video": context_artifacts["context_video"],
                    "context16_video": context_artifacts["context16_video"],
                    "context_video_url": "/media-context/" + case_id,
                    "context16_video_url": "/media-context16/" + case_id,
                    "meta": render_manifest["meta"],
                    "states": render_manifest["states"],
                    "mask_video": render_manifest["mask_video"],
                    "mask_ids": render_manifest["mask_ids"],
                    "question": ANALYSIS_QUESTION,
                    "response_final": "本 pilot 只展示仿真视频和状态摘要，尚未运行 VLM。",
                    "response_raw": "",
                    "answer_source": "simulation_pilot",
                    "caption_intent_only": render_manifest["caption"],
                    "v2v": context_artifacts["v2v"],
                }
            )
            print(f"rendered {case_id} -> {render_manifest['video']}", flush=True)
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append(
                {
                    "case_id": case_id,
                    "difficulty": "L2",
                    "family_key": family_key,
                    "seed": seed,
                    "error": repr(exc),
                }
            )
            print(f"failed {case_id}: {exc!r}", flush=True)

    merged_by_case_id = {str(row["case_id"]): row for row in existing_rows}
    merged_by_case_id.update({str(row["case_id"]): row for row in rows})
    all_rows = list(merged_by_case_id.values())
    requested = (
        (3 * args.per_level if args.only_control_group is None else 0)
        + (len(TABLE_ROLLOFF_CASES) if args.only_control_group in (None, "f11") else 0)
        + (len(RAMP_INCLINE_CASES) if args.only_control_group in (None, "f12") else 0)
    )

    _write_json(output_root / "pilot_manifest.json", all_rows)
    _write_json(output_root / "reports" / "failure_report.json", failures)
    _write_json(
        output_root / "reports" / "summary.json",
        {
            "total_requested": requested,
            "rendered": len(all_rows),
            "rendered_this_run": len(rows),
            "existing_rows_preserved": len(existing_rows),
            "failures": len(failures),
            "difficulty_levels": DIFFICULTY_LEVELS,
            "state_summary_policy": "derived from preserved PyBullet states; not a semantic event annotation",
        },
    )
    with (output_root / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "requested": requested,
                "rendered": len(all_rows),
                "rendered_this_run": len(rows),
                "failures": len(failures),
                "results": str(output_root / "cases.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
