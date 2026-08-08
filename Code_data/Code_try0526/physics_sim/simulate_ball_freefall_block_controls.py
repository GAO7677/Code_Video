#!/usr/bin/env python3
"""Render controlled free-fall height and block-restitution variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from simulate_ball_block import IMG_H, IMG_W, SIM_HZ, SceneRenderer, write_video


DEFAULT_VIDEO_DIR = Path(
    "/data/gaoya/agent-data/datasets/physv-ball-freefall-block-controlled"
)
FPS = 60
RECORD_EVERY = SIM_HZ // FPS
SIM_DURATION_S = 2.5
BALL_RADIUS_M = 0.18
BALL_MASS_KG = 1.0
BALL_RESTITUTION = 1.0
BLOCK_HALF_EXTENTS_M = (0.25, 0.20, 0.30)
BLOCK_MASS_KG = 0.0
LATERAL_FRICTION = 0.5
GRAVITY_MS2 = (0.0, 0.0, -9.81)
BLOCK_CENTER_XYZ = (0.3, 0.0, BLOCK_HALF_EXTENTS_M[2])
BLOCK_TOP_Z_M = BLOCK_CENTER_XYZ[2] + BLOCK_HALF_EXTENTS_M[2]
CAMERA_EYE = np.asarray([0.3, -3.2, 1.8], dtype=np.float64)
CAMERA_TARGET = np.asarray([0.3, 0.0, 0.9], dtype=np.float64)
CAMERA_UP = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class DropScenario:
    name: str
    label: str
    drop_height_m: float
    block_restitution: float
    changed_variable: str


SCENARIOS = (
    DropScenario("freefall_baseline_h080_e05", "baseline: h=0.80m, block e=0.5", 0.8, 0.5, "baseline"),
    DropScenario("freefall_height_h030_e05", "height: h=0.30m, block e=0.5", 0.3, 0.5, "drop_height"),
    DropScenario("freefall_height_h130_e05", "height: h=1.30m, block e=0.5", 1.3, 0.5, "drop_height"),
    DropScenario("freefall_restitution_h080_e01", "restitution: h=0.80m, block e=0.1", 0.8, 0.1, "block_restitution"),
    DropScenario("freefall_restitution_h080_e09", "restitution: h=0.80m, block e=0.9", 0.8, 0.9, "block_restitution"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ball_start_xyz(drop_height_m: float) -> tuple[float, float, float]:
    return (
        BLOCK_CENTER_XYZ[0],
        BLOCK_CENTER_XYZ[1],
        BLOCK_TOP_Z_M + BALL_RADIUS_M + drop_height_m,
    )


def render_scenario(scenario: DropScenario, output_mp4: Path) -> None:
    ball_start = ball_start_xyz(scenario.drop_height_m)
    ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS_M)
    ball_id = p.createMultiBody(BALL_MASS_KG, ball_collision, basePosition=ball_start)
    block_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=BLOCK_HALF_EXTENTS_M)
    block_id = p.createMultiBody(BLOCK_MASS_KG, block_collision, basePosition=BLOCK_CENTER_XYZ)

    p.changeDynamics(
        ball_id,
        -1,
        restitution=BALL_RESTITUTION,
        lateralFriction=LATERAL_FRICTION,
        spinningFriction=0.003,
        linearDamping=0.03,
        angularDamping=0.03,
    )
    p.changeDynamics(
        block_id,
        -1,
        restitution=scenario.block_restitution,
        lateralFriction=LATERAL_FRICTION,
        spinningFriction=0.008,
    )
    p.resetBaseVelocity(ball_id, linearVelocity=(0.0, 0.0, 0.0))

    renderer = SceneRenderer(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)
    frames: list[np.ndarray] = []
    try:
        for step in range(int(SIM_DURATION_S * SIM_HZ)):
            if step % RECORD_EVERY == 0:
                ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id)
                block_pos, block_quat = p.getBasePositionAndOrientation(block_id)
                renderer.set_ball(ball_pos, ball_quat, BALL_RADIUS_M)
                renderer.set_block(block_pos, block_quat, BLOCK_HALF_EXTENTS_M)
                frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            p.stepSimulation()
    finally:
        renderer.cleanup()

    write_video(output_mp4, frames, FPS, "h264")
    metadata = {
        "video": str(output_mp4),
        "combination": "ball_freefall_block",
        "caption": "A ball freely falling onto a static wooden block",
        "scenario": scenario.name,
        "parameters": {
            "restitution": scenario.block_restitution,
            "ball_restitution": BALL_RESTITUTION,
            "block_restitution": scenario.block_restitution,
            "lateral_friction": LATERAL_FRICTION,
            "ball_mass_kg": BALL_MASS_KG,
            "block_mass_kg": BLOCK_MASS_KG,
        },
        "initial_conditions": {
            "drop_height_above_block_m": scenario.drop_height_m,
            "ball_radius_m": BALL_RADIUS_M,
            "ball_start_xyz": list(ball_start),
            "ball_velocity_ms": [0.0, 0.0, 0.0],
            "block_half_extents_m": list(BLOCK_HALF_EXTENTS_M),
            "block_start_xyz": list(BLOCK_CENTER_XYZ),
            "gravity_ms2": list(GRAVITY_MS2),
        },
        "rendering": {
            "engine": "pyrender",
            "fps": FPS,
            "duration_s": SIM_DURATION_S,
            "resolution": [IMG_W, IMG_H],
            "frames": len(frames),
            "codec": "h264",
            "overlay_stats": False,
            "overlay_scenario_label": False,
            "camera_eye": CAMERA_EYE.tolist(),
            "camera_target": CAMERA_TARGET.tolist(),
            "camera_up": CAMERA_UP.tolist(),
        },
        "physics": {
            "engine": "pybullet",
            "timestep_s": 1.0 / SIM_HZ,
            "solver_iterations": 100,
            "block_is_static": True,
        },
        "experiment": {"changed_variable": scenario.changed_variable},
    }
    output_mp4.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    p.removeBody(ball_id)
    p.removeBody(block_id)
    print(f"wrote {output_mp4.name}: {len(frames)} frames", flush=True)


def write_manifest(video_dir: Path) -> None:
    manifest = {
        "experiment": "ball_freefall_block_controlled",
        "design": "one_factor_at_a_time",
        "baseline": {"drop_height_m": 0.8, "block_restitution": 0.5},
        "fixed_parameters": {
            "ball_initial_velocity_ms": [0.0, 0.0, 0.0],
            "ball_restitution": BALL_RESTITUTION,
            "ball_mass_kg": BALL_MASS_KG,
            "ball_radius_m": BALL_RADIUS_M,
            "block_mass_kg": BLOCK_MASS_KG,
            "block_is_static": True,
            "lateral_friction": LATERAL_FRICTION,
            "gravity_ms2": list(GRAVITY_MS2),
            "fps": FPS,
            "duration_s": SIM_DURATION_S,
            "resolution": [IMG_W, IMG_H],
        },
        "height_definition": "vertical clearance from ball bottom to block top",
        "cases": [asdict(scenario) for scenario in SCENARIOS],
    }
    (video_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    selected_names = set(args.scenario) if args.scenario else None
    selected = [s for s in SCENARIOS if selected_names is None or s.name in selected_names]
    if selected_names is not None and {s.name for s in selected} != selected_names:
        raise ValueError(f"Unknown scenarios: {sorted(selected_names - {s.name for s in selected})}")

    args.video_dir.mkdir(parents=True, exist_ok=True)
    for scenario in selected:
        outputs = [args.video_dir / f"{scenario.name}{suffix}" for suffix in (".mp4", ".json")]
        if not args.overwrite and any(path.exists() for path in outputs):
            raise FileExistsError(f"Refusing to overwrite existing case: {scenario.name}")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(*GRAVITY_MS2)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / SIM_HZ,
        numSolverIterations=100,
        numSubSteps=1,
    )
    p.loadURDF("plane.urdf")
    try:
        for index, scenario in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] {scenario.label}", flush=True)
            render_scenario(scenario, args.video_dir / f"{scenario.name}.mp4")
    finally:
        p.disconnect()

    write_manifest(args.video_dir)
    print(f"Done -> {args.video_dir}", flush=True)


if __name__ == "__main__":
    main()
