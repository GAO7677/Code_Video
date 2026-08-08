#!/usr/bin/env python3
"""Render controlled ball-block speed, direction, and start-distance variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from simulate_ball_block import Scenario, run_scenario


DEFAULT_VIDEO_DIR = Path(
    "/home/gaoya/data/agent-data/datasets/physv-ball-block-motion-controlled"
)
BASE_VELOCITY = np.asarray([3.5, 0.0, 1.8], dtype=np.float64)
BASE_BALL_START_X = -1.0
BLOCK_START_X = 0.3
BASE_CENTER_DISTANCE_X = BLOCK_START_X - BASE_BALL_START_X


def scaled_velocity(scale: float) -> tuple[float, float, float]:
    return tuple(float(value) for value in BASE_VELOCITY * scale)


def yaw_velocity(degrees: float) -> tuple[float, float, float]:
    radians = math.radians(degrees)
    vx, _, vz = BASE_VELOCITY
    return float(vx * math.cos(radians)), float(vx * math.sin(radians)), float(vz)


SCENARIOS = (
    Scenario(
        "motion_speed_050x",
        "speed=0.50x, baseline direction",
        0.7,
        0.5,
        1.0,
        scaled_velocity(0.50),
    ),
    Scenario(
        "motion_speed_075x",
        "speed=0.75x, baseline direction",
        0.7,
        0.5,
        1.0,
        scaled_velocity(0.75),
    ),
    Scenario(
        "motion_speed_125x",
        "speed=1.25x, baseline direction",
        0.7,
        0.5,
        1.0,
        scaled_velocity(1.25),
    ),
    Scenario(
        "motion_speed_150x",
        "speed=1.50x, baseline direction",
        0.7,
        0.5,
        1.0,
        scaled_velocity(1.50),
    ),
    Scenario(
        "motion_direction_yaw_m10",
        "yaw=-10deg, baseline speed",
        0.7,
        0.5,
        1.0,
        yaw_velocity(-10.0),
    ),
    Scenario(
        "motion_direction_yaw_p10",
        "yaw=+10deg, baseline speed",
        0.7,
        0.5,
        1.0,
        yaw_velocity(10.0),
    ),
    Scenario(
        "motion_distance_050x",
        "x center distance=0.50x, baseline velocity",
        0.7,
        0.5,
        1.0,
        tuple(float(value) for value in BASE_VELOCITY),
        ball_start_x=BLOCK_START_X - 0.50 * BASE_CENTER_DISTANCE_X,
    ),
    Scenario(
        "motion_distance_075x",
        "x center distance=0.75x, baseline velocity",
        0.7,
        0.5,
        1.0,
        tuple(float(value) for value in BASE_VELOCITY),
        ball_start_x=BLOCK_START_X - 0.75 * BASE_CENTER_DISTANCE_X,
    ),
    Scenario(
        "motion_distance_125x",
        "x center distance=1.25x, baseline velocity",
        0.7,
        0.5,
        1.0,
        tuple(float(value) for value in BASE_VELOCITY),
        ball_start_x=BLOCK_START_X - 1.25 * BASE_CENTER_DISTANCE_X,
    ),
    Scenario(
        "motion_distance_150x",
        "x center distance=1.50x, baseline velocity",
        0.7,
        0.5,
        1.0,
        tuple(float(value) for value in BASE_VELOCITY),
        ball_start_x=BLOCK_START_X - 1.50 * BASE_CENTER_DISTANCE_X,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_manifest(video_dir: Path) -> None:
    base_speed = float(np.linalg.norm(BASE_VELOCITY))
    cases = []
    for scenario in SCENARIOS:
        velocity = np.asarray(scenario.ball_velocity, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        center_distance_x = BLOCK_START_X - scenario.ball_start_x
        if scenario.name.startswith("motion_speed"):
            changed_variable = "speed"
        elif scenario.name.startswith("motion_direction"):
            changed_variable = "direction_yaw"
        else:
            changed_variable = "initial_distance_x"
        cases.append(
            {
                "scenario": scenario.name,
                "ball_start_x_m": scenario.ball_start_x,
                "center_distance_x_m": center_distance_x,
                "distance_scale": center_distance_x / BASE_CENTER_DISTANCE_X,
                "velocity_ms": velocity.tolist(),
                "speed_ms": speed,
                "speed_scale": speed / base_speed,
                "yaw_deg": math.degrees(math.atan2(velocity[1], velocity[0])),
                "changed_variable": changed_variable,
            }
        )
    manifest = {
        "baseline_scenario": "e07_mu05_m1",
        "baseline_velocity_ms": BASE_VELOCITY.tolist(),
        "baseline_speed_ms": base_speed,
        "baseline_ball_start_x_m": BASE_BALL_START_X,
        "block_start_x_m": BLOCK_START_X,
        "baseline_center_distance_x_m": BASE_CENTER_DISTANCE_X,
        "fixed_parameters": {
            "restitution": 0.7,
            "lateral_friction": 0.5,
            "ball_mass_kg": 1.0,
            "block_mass_kg": 1.5,
            "gravity_ms2": [0.0, 0.0, -9.81],
        },
        "cases": cases,
    }
    (video_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.video_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.scenario) if args.scenario else None
    scenarios = [scenario for scenario in SCENARIOS if selected is None or scenario.name in selected]
    if selected is not None and {scenario.name for scenario in scenarios} != selected:
        missing = sorted(selected - {scenario.name for scenario in scenarios})
        raise ValueError(f"Unknown scenario(s): {missing}")

    for scenario in scenarios:
        output_mp4 = args.video_dir / f"{scenario.name}.mp4"
        output_json = output_mp4.with_suffix(".json")
        if not args.overwrite and (output_mp4.exists() or output_json.exists()):
            raise FileExistsError(f"Refusing to overwrite existing case: {scenario.name}")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    try:
        for index, scenario in enumerate(scenarios, 1):
            print(f"[{index}/{len(scenarios)}] {scenario.name}: {scenario.label}", flush=True)
            run_scenario(
                scenario,
                args.video_dir / f"{scenario.name}.mp4",
                record_every=4,
                sim_duration=2.5,
                show_stats_overlay=False,
                show_scenario_label=False,
                codec="h264",
                export_sample_root=None,
            )
    finally:
        p.disconnect()
    write_manifest(args.video_dir)
    print(f"Done -> {args.video_dir}", flush=True)


if __name__ == "__main__":
    main()
