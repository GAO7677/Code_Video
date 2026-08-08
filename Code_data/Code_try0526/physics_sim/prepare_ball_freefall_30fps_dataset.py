#!/usr/bin/env python3
"""Create exact even-frame 30 FPS inputs and aligned PyBullet GT states."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pybullet as p
import pybullet_data

from simulate_ball_block import IMG_H, IMG_W, SIM_HZ


SOURCE_FPS = 60
TARGET_FPS = 30
SIM_DURATION_S = 2.5
TARGET_FRAME_COUNT = int(TARGET_FPS * SIM_DURATION_S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_even_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames: list[np.ndarray] = []
    source_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_count % 2 == 0:
            frames.append(frame)
        source_count += 1
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if source_count != 150 or abs(source_fps - SOURCE_FPS) > 1e-6:
        raise ValueError(f"Unexpected source video: {path}, frames={source_count}, fps={source_fps}")
    if len(frames) != TARGET_FRAME_COUNT:
        raise ValueError(f"Unexpected output frame count for {path}: {len(frames)}")
    return frames


def write_all_intra_video(path: Path, frames: list[np.ndarray]) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{IMG_W}x{IMG_H}",
        "-r",
        str(TARGET_FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-g",
        "1",
        "-bf",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def simulate_states(metadata: dict[str, object]) -> dict[str, np.ndarray]:
    parameters = metadata["parameters"]
    initial = metadata["initial_conditions"]
    ball_radius = float(initial["ball_radius_m"])
    block_half_extents = tuple(float(x) for x in initial["block_half_extents_m"])
    ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
    ball_id = p.createMultiBody(
        float(parameters["ball_mass_kg"]),
        ball_collision,
        basePosition=initial["ball_start_xyz"],
    )
    block_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_half_extents)
    block_id = p.createMultiBody(
        float(parameters["block_mass_kg"]),
        block_collision,
        basePosition=initial["block_start_xyz"],
    )
    p.changeDynamics(
        ball_id,
        -1,
        restitution=float(parameters["ball_restitution"]),
        lateralFriction=float(parameters["lateral_friction"]),
        spinningFriction=0.003,
        linearDamping=0.03,
        angularDamping=0.03,
    )
    p.changeDynamics(
        block_id,
        -1,
        restitution=float(parameters["block_restitution"]),
        lateralFriction=float(parameters["lateral_friction"]),
        spinningFriction=0.008,
    )
    p.resetBaseVelocity(ball_id, linearVelocity=initial["ball_velocity_ms"])

    ball_positions: list[tuple[float, float, float]] = []
    ball_velocities: list[tuple[float, float, float]] = []
    block_positions: list[tuple[float, float, float]] = []
    block_velocities: list[tuple[float, float, float]] = []
    contact: list[bool] = []
    source_indices: list[int] = []
    first_contact_step = -1
    for step in range(int(SIM_DURATION_S * SIM_HZ)):
        if step % (SIM_HZ // TARGET_FPS) == 0:
            ball_positions.append(p.getBasePositionAndOrientation(ball_id)[0])
            ball_velocities.append(p.getBaseVelocity(ball_id)[0])
            block_positions.append(p.getBasePositionAndOrientation(block_id)[0])
            block_velocities.append(p.getBaseVelocity(block_id)[0])
            contact.append(bool(p.getContactPoints(bodyA=ball_id, bodyB=block_id)))
            source_indices.append(step // (SIM_HZ // SOURCE_FPS))
        p.stepSimulation()
        if first_contact_step < 0 and p.getContactPoints(bodyA=ball_id, bodyB=block_id):
            first_contact_step = step + 1

    p.removeBody(ball_id)
    p.removeBody(block_id)
    return {
        "ball_positions_m": np.asarray(ball_positions, dtype=np.float32),
        "ball_velocities_ms": np.asarray(ball_velocities, dtype=np.float32),
        "block_positions_m": np.asarray(block_positions, dtype=np.float32),
        "block_velocities_ms": np.asarray(block_velocities, dtype=np.float32),
        "contact": np.asarray(contact, dtype=np.bool_),
        "frame_times_s": np.arange(TARGET_FRAME_COUNT, dtype=np.float32) / TARGET_FPS,
        "source_frame_indices_60fps": np.asarray(source_indices, dtype=np.int16),
        "first_contact_step_240hz": np.asarray(first_contact_step, dtype=np.int32),
        "first_contact_time_s": np.asarray(first_contact_step / SIM_HZ, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    source_manifest = json.loads((args.source_dir / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True)

    canonical_name = "freefall_height_h080_e07"
    canonical_frames = read_even_frames(args.source_dir / f"{canonical_name}.mp4")
    canonical_metadata = json.loads((args.source_dir / f"{canonical_name}.json").read_text())

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / SIM_HZ,
        numSolverIterations=100,
        numSubSteps=1,
    )
    p.loadURDF("plane.urdf")
    try:
        canonical_states = simulate_states(canonical_metadata)
        first_contact_time = float(canonical_states["first_contact_time_s"])
        if first_contact_time <= 0:
            raise RuntimeError("Canonical h=0.8 case has no ball-block contact")
        first_contact_frame = int(math.ceil(first_contact_time * TARGET_FPS))
        for case in source_manifest["cases"]:
            name = case["name"]
            source_video = args.source_dir / f"{name}.mp4"
            source_json = args.source_dir / f"{name}.json"
            output_video = args.output_dir / source_video.name
            frames = read_even_frames(source_video)
            if case["changed_variable"] == "block_restitution":
                frames[:first_contact_frame] = canonical_frames[:first_contact_frame]
            write_all_intra_video(output_video, frames)

            metadata = json.loads(source_json.read_text())
            states_name = f"{name}.states.npz"
            np.savez_compressed(args.output_dir / states_name, **simulate_states(metadata))
            metadata["video"] = str(output_video)
            metadata["states"] = states_name
            metadata["rendering"].update(
                {
                    "fps": TARGET_FPS,
                    "frames": TARGET_FRAME_COUNT,
                    "duration_s": SIM_DURATION_S,
                    "source_fps": SOURCE_FPS,
                    "source_frames": 150,
                    "temporal_downsample": "exact_even_source_frames",
                    "source_frame_indices": list(range(0, 150, 2)),
                    "encoding": "H.264 CRF 0 all-intra",
                    "precontact_visual_canonicalization": {
                        "enabled": case["changed_variable"] == "block_restitution",
                        "canonical_case": canonical_name,
                        "frames": list(range(first_contact_frame)),
                        "reason": "remove codec-only differences before restitution becomes observable",
                    },
                }
            )
            output_video.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {name}: 75 frames @ 30 FPS + states", flush=True)
    finally:
        p.disconnect()

    source_manifest["fixed_parameters"].update(
        {
            "fps": TARGET_FPS,
            "frames": TARGET_FRAME_COUNT,
            "duration_s": SIM_DURATION_S,
        }
    )
    source_manifest["source_video_dir"] = str(args.source_dir)
    source_manifest["temporal_downsample"] = {
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "source_frame_indices": list(range(0, 150, 2)),
    }
    source_manifest["precontact_visual_canonicalization"] = {
        "group": "block_restitution",
        "canonical_case": canonical_name,
        "frames": list(range(first_contact_frame)),
        "encoding": "H.264 CRF 0 all-intra",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
