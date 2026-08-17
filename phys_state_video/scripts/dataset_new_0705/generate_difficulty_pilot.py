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

import numpy as np

from .render_sim_0705 import render_blueprint_case
from .scene_generators_0705 import (
    DEFAULT_CAMERA_DISTANCE_SCALE,
    build_scenario_family_catalog,
    generate_scenario_blueprint,
)


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

TABLE_ROLLOFF_CASES = (
    {"table_height_m": 0.46, "height_label": "low", "travel_angle_deg": 0.0, "angle_label": "a000"},
    {"table_height_m": 0.68, "height_label": "mid", "travel_angle_deg": 0.0, "angle_label": "a000"},
    {"table_height_m": 0.92, "height_label": "high", "travel_angle_deg": 0.0, "angle_label": "a000"},
    {"table_height_m": 0.68, "height_label": "mid", "travel_angle_deg": -24.0, "angle_label": "am24"},
    {"table_height_m": 0.68, "height_label": "mid", "travel_angle_deg": -12.0, "angle_label": "am12"},
    {"table_height_m": 0.68, "height_label": "mid", "travel_angle_deg": 12.0, "angle_label": "ap12"},
    {"table_height_m": 0.68, "height_label": "mid", "travel_angle_deg": 24.0, "angle_label": "ap24"},
    {"table_height_m": 0.68, "height_label": "mid", "travel_angle_deg": 180.0, "angle_label": "a180"},
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_level <= 0:
        raise ValueError("--per-level must be positive")
    output_root = args.output_root
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    family_catalog = build_scenario_family_catalog()
    shared_table_speed = 1.25

    case_index = 0
    for level_key, level in DIFFICULTY_LEVELS.items():
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
                    "scene_style": args.scene_style,
                    "scenario_spec": {
                        "surface_key": blueprint.surface_key,
                        "camera_key": blueprint.camera_key,
                        "lighting_key": blueprint.lighting_key,
                        "scene_style": args.scene_style,
                        "floor_restitution": round(float(blueprint.metadata.get("floor_restitution", 0.02)), 5),
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

    for extra in TABLE_ROLLOFF_CASES:
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
            state_summary = _summarize_states(Path(render_manifest["states"]))
            family_summary = _family_summary(family_key)
            difficulty = {
                "level": "L2",
                "title": "桌面滚落",
                "description": "同物体、同初速度在不同桌高和不同速度方向上滚动并越过桌缘，重点观察落体时机、反弹和轨迹方向变化。",
                "priority": 2,
            }
            pilot_metadata = {
                "difficulty": difficulty,
                "scene_family": family_summary,
                "state_summary": state_summary,
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
                    "table_height_label": extra["height_label"],
                    "angle_label": extra["angle_label"],
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
                    "scene_style": "indoor_realistic",
                    "scenario_spec": pilot_metadata["scenario_spec"],
                    "state_summary": state_summary,
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
                    "difficulty": "L2",
                    "family_key": family_key,
                    "seed": seed,
                    "error": repr(exc),
                }
            )
            print(f"failed {case_id}: {exc!r}", flush=True)

    _write_json(output_root / "pilot_manifest.json", rows)
    _write_json(output_root / "reports" / "failure_report.json", failures)
    _write_json(
        output_root / "reports" / "summary.json",
        {
            "total_requested": 3 * args.per_level + len(TABLE_ROLLOFF_CASES),
            "rendered": len(rows),
            "failures": len(failures),
            "difficulty_levels": DIFFICULTY_LEVELS,
            "state_summary_policy": "derived from preserved PyBullet states; not a semantic event annotation",
        },
    )
    with (output_root / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "requested": 3 * args.per_level + len(TABLE_ROLLOFF_CASES),
                "rendered": len(rows),
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
