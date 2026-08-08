#!/usr/bin/env python3
"""Render extreme single-variable ball-block controls and contact diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from simulate_ball_block import Scenario, run_scenario


DEFAULT_VIDEO_DIR = (
    Path(os.environ.get("AGENT_DATA_ROOT", "/data/gaoya/agent-data"))
    / "datasets/physv-ball-block-extreme-controlled"
)
BASE_VELOCITY = np.asarray([3.5, 0.0, 1.8], dtype=np.float64)
BASE_BALL_START_X = -1.0
BLOCK_START_X = 0.3
BASE_CENTER_DISTANCE_X = BLOCK_START_X - BASE_BALL_START_X
SIM_HZ = 240
RECORD_EVERY = 4
SIM_DURATION = 2.5


def scaled_velocity(scale: float) -> tuple[float, float, float]:
    return tuple(float(value) for value in BASE_VELOCITY * scale)


def yaw_velocity(degrees: float) -> tuple[float, float, float]:
    radians = math.radians(degrees)
    vx, _, vz = BASE_VELOCITY
    return float(vx * math.cos(radians)), float(vx * math.sin(radians)), float(vz)


def distance_start_x(scale: float) -> float:
    return BLOCK_START_X - scale * BASE_CENTER_DISTANCE_X


@dataclass(frozen=True)
class ExtremeCase:
    scenario: Scenario
    changed_variable: str
    value: float
    baseline_value: float
    unit: str
    expected_collision: bool = True


CASES = (
    ExtremeCase(Scenario("extreme_restitution_000", "e=0.0", 0.0, 0.5, 1.0), "restitution", 0.0, 0.7, ""),
    ExtremeCase(Scenario("extreme_restitution_100", "e=1.0", 1.0, 0.5, 1.0), "restitution", 1.0, 0.7, ""),
    ExtremeCase(Scenario("extreme_friction_000", "mu=0.0", 0.7, 0.0, 1.0), "lateral_friction", 0.0, 0.5, ""),
    ExtremeCase(Scenario("extreme_friction_200", "mu=2.0", 0.7, 2.0, 1.0), "lateral_friction", 2.0, 0.5, ""),
    ExtremeCase(Scenario("extreme_mass_005x", "mass=0.05x", 0.7, 0.5, 0.05), "ball_mass_scale", 0.05, 1.0, "x"),
    ExtremeCase(Scenario("extreme_mass_10x", "mass=10x", 0.7, 0.5, 10.0), "ball_mass_scale", 10.0, 1.0, "x"),
    ExtremeCase(Scenario("extreme_speed_025x", "speed=0.25x", 0.7, 0.5, 1.0, scaled_velocity(0.25)), "speed_scale", 0.25, 1.0, "x"),
    ExtremeCase(Scenario("extreme_speed_200x", "speed=2.0x", 0.7, 0.5, 1.0, scaled_velocity(2.0)), "speed_scale", 2.0, 1.0, "x"),
    ExtremeCase(Scenario("extreme_direction_yaw_m25", "yaw=-25deg", 0.7, 0.5, 1.0, yaw_velocity(-25.0)), "direction_yaw", -25.0, 0.0, "deg", False),
    ExtremeCase(Scenario("extreme_direction_yaw_p25", "yaw=+25deg", 0.7, 0.5, 1.0, yaw_velocity(25.0)), "direction_yaw", 25.0, 0.0, "deg", False),
    ExtremeCase(Scenario("extreme_distance_035x", "distance=0.35x", 0.7, 0.5, 1.0, tuple(BASE_VELOCITY), distance_start_x(0.35)), "initial_distance_scale", 0.35, 1.0, "x"),
    ExtremeCase(Scenario("extreme_distance_200x", "distance=2.0x", 0.7, 0.5, 1.0, tuple(BASE_VELOCITY), distance_start_x(2.0)), "initial_distance_scale", 2.0, 1.0, "x"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def probe_contact(scenario: Scenario) -> dict[str, object]:
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0 / SIM_HZ, numSolverIterations=100, numSubSteps=1)
    ball_radius = 0.18
    block_half = (0.25, 0.20, 0.30)
    ball_id = p.createMultiBody(
        scenario.ball_mass,
        p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius),
        basePosition=(scenario.ball_start_x, 0.0, 0.20),
    )
    block_id = p.createMultiBody(
        1.5,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=block_half),
        basePosition=(BLOCK_START_X, 0.0, block_half[2]),
    )
    for body, spinning, damping in ((ball_id, 0.003, 0.03), (block_id, 0.008, 0.06)):
        p.changeDynamics(
            body,
            -1,
            restitution=scenario.restitution,
            lateralFriction=scenario.lateral_friction,
            spinningFriction=spinning,
            linearDamping=damping,
            angularDamping=damping,
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        )
    p.resetBaseVelocity(ball_id, linearVelocity=scenario.ball_velocity)
    for _ in range(10):
        p.stepSimulation()

    initial_block = np.asarray(p.getBasePositionAndOrientation(block_id)[0], dtype=np.float64)
    first_contact_step = None
    contact_steps = 0
    finite = True
    for step in range(round(SIM_DURATION * SIM_HZ)):
        p.stepSimulation()
        if p.getContactPoints(ball_id, block_id):
            contact_steps += 1
            if first_contact_step is None:
                first_contact_step = step
        positions = (
            p.getBasePositionAndOrientation(ball_id)[0],
            p.getBasePositionAndOrientation(block_id)[0],
        )
        finite = finite and bool(np.isfinite(np.asarray(positions)).all())

    final_block = np.asarray(p.getBasePositionAndOrientation(block_id)[0], dtype=np.float64)
    p.removeBody(ball_id)
    p.removeBody(block_id)
    return {
        "collision_detected": first_contact_step is not None,
        "first_contact_step": first_contact_step,
        "first_contact_time_s": None if first_contact_step is None else first_contact_step / SIM_HZ,
        "first_contact_frame_60fps": None if first_contact_step is None else int(math.ceil(first_contact_step / RECORD_EVERY)),
        "contact_physics_steps": contact_steps,
        "block_displacement_m": float(np.linalg.norm(final_block - initial_block)),
        "finite_state": finite,
    }


def append_metadata(path: Path, case: ExtremeCase, probe: dict[str, object]) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["controlled_variable"] = case.changed_variable
    metadata["control_value"] = case.value
    metadata["baseline_value"] = case.baseline_value
    metadata["control_unit"] = case.unit
    metadata["expected_collision"] = case.expected_collision
    metadata["contact_probe"] = probe
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.video_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.scenario) if args.scenario else None
    cases = [case for case in CASES if selected is None or case.scenario.name in selected]
    available = {case.scenario.name for case in CASES}
    if selected is not None and selected - available:
        raise ValueError(f"Unknown scenarios: {sorted(selected - available)}")
    for case in cases:
        paths = (args.video_dir / f"{case.scenario.name}.mp4", args.video_dir / f"{case.scenario.name}.json")
        if not args.overwrite and any(path.exists() for path in paths):
            raise FileExistsError(f"Refusing to overwrite existing case: {case.scenario.name}")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    manifest_cases = []
    try:
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case.scenario.name}", flush=True)
            probe = probe_contact(case.scenario)
            if not probe["finite_state"]:
                raise RuntimeError(f"Non-finite physics state: {case.scenario.name}")
            output = args.video_dir / f"{case.scenario.name}.mp4"
            run_scenario(
                case.scenario,
                output,
                record_every=RECORD_EVERY,
                sim_duration=SIM_DURATION,
                show_stats_overlay=False,
                show_scenario_label=False,
                codec="h264",
                export_sample_root=None,
            )
            append_metadata(output.with_suffix(".json"), case, probe)
            manifest_cases.append(
                {
                    "scenario": case.scenario.name,
                    "changed_variable": case.changed_variable,
                    "value": case.value,
                    "baseline_value": case.baseline_value,
                    "unit": case.unit,
                    "expected_collision": case.expected_collision,
                    "contact_probe": probe,
                }
            )
            print(f"  contact={probe['collision_detected']} first_frame={probe['first_contact_frame_60fps']}", flush=True)
    finally:
        p.disconnect()

    manifest = {
        "baseline": "e07_mu05_m1",
        "control_policy": "one variable changed per case",
        "rendering": {"fps": 60, "frames": 150, "resolution": [1280, 720]},
        "cases": manifest_cases,
    }
    (args.video_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Done -> {args.video_dir}", flush=True)


if __name__ == "__main__":
    main()
