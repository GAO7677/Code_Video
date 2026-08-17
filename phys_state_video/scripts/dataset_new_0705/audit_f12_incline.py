"""Audit deterministic F12 incline-control renders and create frame strips."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CASE_SPECS = (
    ("difficulty_l2_f12_a008", 8.0),
    ("difficulty_l2_f12_a016", 16.0),
    ("difficulty_l2_f12_a024", 24.0),
    ("difficulty_l2_f12_a032", 32.0),
)


def _case_root(output_root: Path, case_id: str) -> Path:
    return output_root / "cases" / "L2" / "F12" / case_id


def _write_timeline(video_path: Path, output_path: Path, angle_deg: float) -> int:
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 3:
            raise RuntimeError(f"expected at least three frames: {video_path}")
        frame_indices = (0, frame_count // 2, frame_count - 1)
        frames: list[np.ndarray] = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not read frame {frame_index}: {video_path}")
            label = f"{angle_deg:g} deg | frame {frame_index + 1}/{frame_count}"
            cv2.putText(
                frame,
                label,
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (30, 30, 30),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            frames.append(frame)
    finally:
        capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), np.concatenate(frames, axis=1)):
        raise RuntimeError(f"could not write timeline: {output_path}")
    return frame_count


def _camera_downward_angle_deg(camera: dict[str, object]) -> float:
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    horizontal_distance = float(np.linalg.norm((eye - target)[:2]))
    return math.degrees(math.atan2(abs(float(eye[2] - target[2])), horizontal_distance))


def _load_case(output_root: Path, case_id: str, angle_deg: float) -> dict[str, object]:
    case_root = _case_root(output_root, case_id)
    meta_path = case_root / "meta" / f"{case_id}.json"
    states_path = case_root / "meta" / f"{case_id}_states.npz"
    mask_path = case_root / "masks" / f"{case_id}_instance_ids.npz"
    video_path = case_root / "videos" / f"{case_id}.mp4"

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    state_payload = np.load(states_path, allow_pickle=True)
    mask_payload = np.load(mask_path, allow_pickle=True)
    positions = np.asarray(state_payload["positions"], dtype=np.float64)
    linear_velocities = np.asarray(state_payload["linear_velocities"], dtype=np.float64)
    names = [str(value) for value in state_payload["object_names"]]
    instance_ids = np.asarray(mask_payload["instance_ids"])
    mask_names = [str(value) for value in mask_payload["object_names"]]
    mask_object_ids = [int(value) for value in mask_payload["object_ids"]]
    visibility = {
        name: bool(np.any(instance_ids == object_id, axis=(1, 2)).all())
        for name, object_id in zip(mask_names, mask_object_ids)
    }
    object_specs = {str(item["name"]): item for item in metadata["objects"]}
    wheel_index = names.index("wheel_0")
    wheel = object_specs["wheel_0"]
    board_index = names.index("incline_board_0")
    riser_indices = [index for index, name in enumerate(names) if name.startswith("incline_riser_")]
    wheel_speeds = np.linalg.norm(linear_velocities[:, wheel_index], axis=1)
    configured_initial_speed = float(
        np.linalg.norm(np.asarray(wheel["linear_velocity"], dtype=np.float64))
    )
    board_displacement = float(np.linalg.norm(positions[-1, board_index] - positions[0, board_index]))
    riser_displacement = max(
        float(np.linalg.norm(positions[-1, index] - positions[0, index]))
        for index in riser_indices
    )
    frame_count = _write_timeline(
        video_path,
        output_root / "reports" / f"f12_a{int(angle_deg):03d}_timeline.jpg",
        angle_deg,
    )

    return {
        "case_id": case_id,
        "ramp_angle_deg": angle_deg,
        "frames": frame_count,
        "lighting_key": metadata["lighting_key"],
        "camera_downward_angle_deg": round(_camera_downward_angle_deg(metadata["camera"]), 4),
        "all_dynamic_objects": bool(all(bool(spec["dynamic"]) for spec in object_specs.values())),
        "all_objects_visible_every_frame": bool(all(visibility.values())),
        "visibility_by_object": visibility,
        "wheel": {
            "mass_kg": float(wheel["mass"]),
            "radius_m": float(wheel["size"]["radius"]),
            "width_m": float(wheel["size"]["width"]),
            "friction": float(wheel["friction"]),
            "restitution": float(wheel["restitution"]),
            "configured_initial_speed_mps": round(configured_initial_speed, 6),
            "first_saved_frame_speed_mps": round(float(wheel_speeds[0]), 6),
            "max_speed_mps": round(float(wheel_speeds.max()), 6),
            "final_speed_mps": round(float(wheel_speeds[-1]), 6),
        },
        "board_final_displacement_m": round(board_displacement, 6),
        "max_riser_final_displacement_m": round(riser_displacement, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    cases = [_load_case(args.output_root, case_id, angle_deg) for case_id, angle_deg in CASE_SPECS]
    speeds = [float(case["wheel"]["max_speed_mps"]) for case in cases]
    wheel_signatures = [
        (
            case["wheel"]["mass_kg"],
            case["wheel"]["radius_m"],
            case["wheel"]["width_m"],
            case["wheel"]["friction"],
            case["wheel"]["restitution"],
        )
        for case in cases
    ]
    checks = {
        "four_angle_cases": [case["ramp_angle_deg"] for case in cases]
        == [angle for _, angle in CASE_SPECS],
        "all_use_hall_bright": all(case["lighting_key"] == "hall_bright" for case in cases),
        "identical_enlarged_wheel_parameters": wheel_signatures == [wheel_signatures[0]] * len(cases),
        "released_from_rest": all(
            float(case["wheel"]["configured_initial_speed_mps"]) == 0.0 for case in cases
        ),
        "all_camera_downward_angles_between_5_and_6deg": all(
            5.0 < float(case["camera_downward_angle_deg"]) < 6.0 for case in cases
        ),
        "all_visible_all_frames": all(bool(case["all_objects_visible_every_frame"]) for case in cases),
        "all_visible_bodies_dynamic": all(bool(case["all_dynamic_objects"]) for case in cases),
        "max_speed_strictly_increases_with_angle": all(
            later > earlier for earlier, later in zip(speeds, speeds[1:])
        ),
        "ramp_settles_within_2cm": all(
            float(case["board_final_displacement_m"]) < 0.02 for case in cases
        ),
        "risers_settle_within_2mm": all(
            float(case["max_riser_final_displacement_m"]) < 0.002 for case in cases
        ),
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "F12 incline-control render audit",
        "camera_view": "front-facing shallow downward view",
        "checks": checks,
        "pass": bool(all(checks.values())),
        "cases": cases,
    }
    report_path = args.output_root / "reports" / "f12_incline_physics_qa.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "pass": report["pass"]}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
