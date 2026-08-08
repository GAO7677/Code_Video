#!/usr/bin/env python3
"""球撞击木块物理仿真 — PyBullet 物理 + Pyrender 阴影渲染"""

from __future__ import annotations

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import argparse
import subprocess

# Fix NumPy 2.0 compat with pyrender
np.infty = np.inf

import pybullet as p
import pybullet_data
import pyrender
import trimesh
from pyrender.constants import RenderFlags

OUTPUT_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
VIDEO_DIR = OUTPUT_DIR / "videos" / "ball_block"
DEFAULT_FPS = 60
DEFAULT_SIM_DURATION = 2.5
SIM_HZ = 240
IMG_W, IMG_H = 1280, 720

CAM_EYE = np.array([0.05, -2.2, 1.2])
CAM_TARGET = np.array([0.05, 0.3, 0.45])
CAM_UP = np.array([0, 0, 1])


@dataclass
class Scenario:
    name: str
    label: str
    restitution: float
    lateral_friction: float
    ball_mass: float
    ball_velocity: tuple[float, float, float] = (3.5, 0.0, 1.8)
    ball_start_x: float = -1.0


SCENARIOS = [
    Scenario("e03_mu05_m1",   "e=0.3  u=0.5  m=1.0kg  plastic",    0.3, 0.5, 1.0),
    Scenario("e05_mu05_m1",   "e=0.5  u=0.5  m=1.0kg  medium",     0.5, 0.5, 1.0),
    Scenario("e07_mu05_m1",   "e=0.7  u=0.5  m=1.0kg  bouncy",     0.7, 0.5, 1.0),
    Scenario("e09_mu05_m1",   "e=0.9  u=0.5  m=1.0kg  superball",  0.9, 0.5, 1.0),
    Scenario("e07_mu01_m1",   "e=0.7  u=0.1  m=1.0kg  low-fric",   0.7, 0.1, 1.0),
    Scenario("e07_mu10_m1",   "e=0.7  u=1.0  m=1.0kg  high-fric",  0.7, 1.0, 1.0),
    Scenario("e07_mu05_m01",  "e=0.7  u=0.5  m=0.1kg  light-ball", 0.7, 0.5, 0.1),
    Scenario("e07_mu05_m5",   "e=0.7  u=0.5  m=5.0kg  heavy-ball", 0.7, 0.5, 5.0),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render ball-block physics simulation videos.")
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=None,
        help="Optional scenario name(s) to export. Defaults to all built-in scenarios.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Target recorded FPS. Must divide the 240 Hz physics step exactly.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=None,
        help="Record one frame every N physics steps. Overrides --fps when set.",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=DEFAULT_SIM_DURATION,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--export-sample-root",
        type=Path,
        default=None,
        help=(
            "Optional raw-sample export root. When set, the script also writes "
            "train/ball_block_dense/<scenario>/ with video.mp4, meta.json, and states.npz."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Optional directory for the rendered mp4/json preview files. Defaults to the original ball_block video dir.",
    )
    parser.add_argument(
        "--show-stats-overlay",
        action="store_true",
        help="Overlay left-top time and speed stats on frames.",
    )
    parser.add_argument(
        "--hide-scenario-label",
        action="store_true",
        help="Do not overlay the top-right scenario label.",
    )
    parser.add_argument(
        "--codec",
        default="h264",
        choices=["h264", "mp4v"],
        help="Output video codec.",
    )
    return parser.parse_args()


def resolve_record_every(args: argparse.Namespace) -> int:
    if args.record_every is not None:
        record_every = int(args.record_every)
        if record_every <= 0:
            raise ValueError("--record-every must be positive")
        if SIM_HZ % record_every != 0:
            raise ValueError(f"--record-every must divide {SIM_HZ}, got {record_every}")
        expected_fps = SIM_HZ // record_every
        if args.fps != expected_fps:
            raise ValueError(
                f"--fps={args.fps} is inconsistent with --record-every={record_every}; "
                f"expected fps={expected_fps} for {SIM_HZ} Hz physics"
            )
        return record_every

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if SIM_HZ % args.fps != 0:
        raise ValueError(f"--fps must divide {SIM_HZ}, got {args.fps}")
    return SIM_HZ // args.fps


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x; pose[:3, 1] = y; pose[:3, 2] = z; pose[:3, 3] = eye
    return pose


def _tr(x, y, z):
    return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]], dtype=np.float64)


def _pb_pose(pos, quat):
    """PyBullet pos+quat to 4x4 matrix."""
    mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = mat
    pose[:3, 3] = pos
    return pose


def _vc_basketball(vertices):
    n = len(vertices)
    np.random.seed(42)
    colors = np.tile([0.85, 0.42, 0.18], (n, 1)).astype(np.float32)
    colors += np.random.uniform(-0.04, 0.04, (n, 3)).astype(np.float32)
    y = vertices[:, 2]; x = vertices[:, 0]; z = vertices[:, 1]
    r2d = np.sqrt(x**2 + z**2)
    colors[np.abs(y) < 0.03] = [0.06, 0.06, 0.08]
    for s in [-1, 1]:
        arc = np.abs(x - s * (0.28 + 0.12 * np.sin(y * np.pi * 1.3)) * r2d)
        colors[(arc < 0.02) & (r2d > 0.05)] = [0.06, 0.06, 0.08]
    colors[np.abs(x) < 0.02] = [0.06, 0.06, 0.08]
    return np.clip(colors, 0, 1)


def _vc_wood(n):
    """Wood-like vertex colors with grain."""
    np.random.seed(7)
    base = np.tile([0.52, 0.34, 0.20], (n, 1)).astype(np.float32)
    # Add stripe grain based on vertex Y position (vertical grain)
    return np.clip(base + np.random.uniform(-0.05, 0.05, (n, 3)).astype(np.float32), 0, 1)


def _vc_floor(n, verts):
    """Floor with plank-like wood colors based on X position."""
    colors = np.zeros((n, 3), dtype=np.float32)
    plank_w = 0.7  # 70cm planks
    half_ext = 10.0
    np.random.seed(3)
    for i in range(n):
        px = verts[i, 0]
        plank_idx = int((px + half_ext) / plank_w)
        base = np.array([0.48, 0.35, 0.22]) + np.random.uniform(-0.04, 0.04, 3)
        seam_pos = (plank_idx + 1) * plank_w - half_ext
        if abs(px - seam_pos) < 0.03:
            base *= 0.55
        colors[i] = base
    return np.clip(colors + np.random.uniform(-0.03, 0.03, (n, 3)).astype(np.float32), 0, 1)


def _vc_wall(n):
    """Warm neutral wall color."""
    base = np.tile([0.55, 0.52, 0.48], (n, 1)).astype(np.float32)
    return np.clip(base + np.random.uniform(-0.02, 0.02, (n, 3)).astype(np.float32), 0, 1)


def _vc_grey(n):
    return np.tile([0.48, 0.50, 0.55], (n, 1)).astype(np.float32)


class SceneRenderer:
    """Pyrender renderer with SpotLight shadow mapping."""

    def __init__(self, camera_eye=None, camera_target=None, camera_up=None):
        self.scene = pyrender.Scene(bg_color=[0.55, 0.52, 0.48],
                                     ambient_light=[0.04, 0.04, 0.05])

        # Infinite floor — very large, wood-colored
        floor = trimesh.creation.box(extents=[20, 10, 0.04])
        floor.visual.vertex_colors = _vc_floor(len(floor.vertices), floor.vertices)
        self.scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False),
                       pose=_tr(0, 2.0, -0.04))

        # Back wall only — warm neutral
        wall = trimesh.creation.box(extents=[20, 0.02, 3.0])
        wall.visual.vertex_colors = _vc_wall(len(wall.vertices))
        self.scene.add(pyrender.Mesh.from_trimesh(wall, smooth=False), pose=_tr(0, 5.0, 1.5))

        # Key light: SpotLight with shadow mapping
        spot = pyrender.SpotLight(color=[1.0, 0.95, 0.90], intensity=70.0,
                                  innerConeAngle=0.4, outerConeAngle=1.0)
        self.scene.add(spot, pose=_look_at([-1.5, -1.5, 2.5], [0.3, 0.5, 0], [0, 0, 1]))

        # Camera
        self.cam = pyrender.PerspectiveCamera(yfov=np.radians(55), aspectRatio=IMG_W/IMG_H)
        self.scene.add(
            self.cam,
            pose=_look_at(
                CAM_EYE if camera_eye is None else camera_eye,
                CAM_TARGET if camera_target is None else camera_target,
                CAM_UP if camera_up is None else camera_up,
            ),
        )

        self.renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)
        self.ball_node = None
        self.block_node = None

    def set_ball(self, pos, quat, radius=0.18):
        if self.ball_node is not None:
            self.scene.remove_node(self.ball_node)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        sphere.visual.vertex_colors = _vc_basketball(sphere.vertices)
        self.ball_node = self.scene.add(
            pyrender.Mesh.from_trimesh(sphere, smooth=True),
            pose=_pb_pose(pos, quat))

    def set_block(self, pos, quat, half_ext=(0.25, 0.20, 0.30)):
        if self.block_node is not None:
            self.scene.remove_node(self.block_node)
        box = trimesh.creation.box(extents=[2*half_ext[0], 2*half_ext[1], 2*half_ext[2]])
        # Wood-like vertex colors: vertical grain + random variation
        n = len(box.vertices)
        vc = np.tile([0.48, 0.32, 0.18], (n, 1)).astype(np.float32)
        # Stripe based on Z (height) for horizontal grain bands
        z = box.vertices[:, 2]
        stripes = 0.5 + 0.5 * np.sin(z * 20.0)
        vc[:, 0] += stripes * 0.08
        vc[:, 1] += stripes * 0.05
        vc = np.clip(vc + np.random.uniform(-0.04, 0.04, (n, 3)).astype(np.float32), 0, 1)
        box.visual.vertex_colors = vc
        self.block_node = self.scene.add(
            pyrender.Mesh.from_trimesh(box, smooth=False),
            pose=_pb_pose(pos, quat))

    def render(self):
        color, _ = self.renderer.render(self.scene, flags=RenderFlags.SHADOWS_SPOT)
        return color

    def cleanup(self):
        self.renderer.delete()


def write_video(path: Path, frames: list[np.ndarray], fps: int, codec: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if codec == "mp4v":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (IMG_W, IMG_H))
        for frame in frames:
            out.write(frame)
        out.release()
        return

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
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
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret} for {path}")
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()


def run_scenario(
    sc: Scenario,
    output_mp4: Path,
    *,
    record_every: int,
    sim_duration: float,
    show_stats_overlay: bool,
    show_scenario_label: bool,
    codec: str,
    export_sample_root: Path | None,
) -> None:
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSolverIterations=100, numSubSteps=1)

    ball_r, ball_z = 0.18, 0.20
    block_h = (0.25, 0.20, 0.30)

    col_b = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_r)
    ball_id = p.createMultiBody(
        sc.ball_mass, col_b, basePosition=(sc.ball_start_x, 0.0, ball_z)
    )
    col_bk = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_h)
    block_id = p.createMultiBody(1.5, col_bk, basePosition=(0.3, 0.0, block_h[2]))

    p.changeDynamics(ball_id, -1, restitution=sc.restitution,
                     lateralFriction=sc.lateral_friction,
                     spinningFriction=0.003, linearDamping=0.03, angularDamping=0.03)
    p.changeDynamics(block_id, -1, restitution=sc.restitution,
                     lateralFriction=sc.lateral_friction,
                     spinningFriction=0.008, linearDamping=0.06, angularDamping=0.06,
                     activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
    p.resetBaseVelocity(ball_id, linearVelocity=sc.ball_velocity)

    for _ in range(10):
        p.stepSimulation()

    renderer = SceneRenderer()
    frames = []
    positions = []
    quats = []
    linvels = []
    angvels = []
    times = []

    sim_steps = int(sim_duration * SIM_HZ)
    for step in range(sim_steps):
        p.stepSimulation()
        if step % record_every != 0:
            continue
        elapsed = step / float(SIM_HZ)

        ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id)
        ball_vel, _ = p.getBaseVelocity(ball_id)
        block_pos, block_quat = p.getBasePositionAndOrientation(block_id)
        block_vel, _ = p.getBaseVelocity(block_id)

        positions.append(np.asarray([ball_pos, block_pos], dtype=np.float32))
        quats.append(np.asarray([ball_quat, block_quat], dtype=np.float32))
        linvels.append(np.asarray([ball_vel, block_vel], dtype=np.float32))
        angvels.append(
            np.asarray(
                [
                    p.getBaseVelocity(ball_id)[1],
                    p.getBaseVelocity(block_id)[1],
                ],
                dtype=np.float32,
            )
        )
        times.append(float(elapsed))

        renderer.set_ball(ball_pos, ball_quat, ball_r)
        renderer.set_block(block_pos, block_quat, block_h)
        color = renderer.render()

        frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        bs, bks = float(np.linalg.norm(ball_vel)), float(np.linalg.norm(block_vel))
        if show_stats_overlay:
            for i, line in enumerate([
                f"t = {elapsed:.2f}s",
                f"Ball |v| = {bs:.2f} m/s",
                f"Block |v| = {bks:.2f} m/s",
            ]):
                y = 36 + i * 34
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (10, 10, 10), 2, cv2.LINE_AA)
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (250, 245, 235), 1, cv2.LINE_AA)
        if show_scenario_label:
            label = sc.label.split("\n")[0]
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            cv2.putText(frame, label, (IMG_W - tw - 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (IMG_W - tw - 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250, 245, 235), 1, cv2.LINE_AA)
        frames.append(frame)

    renderer.cleanup()
    p.removeBody(ball_id)
    p.removeBody(block_id)

    recorded_fps = SIM_HZ // record_every
    write_video(output_mp4, frames, recorded_fps, codec)

    # Write metadata JSON
    import json
    meta = {
        "video": str(output_mp4),
        "combination": "ball_block",
        "caption": "Ball colliding with a wooden block",
        "scenario": sc.name,
        "parameters": {
            "restitution": sc.restitution,
            "lateral_friction": sc.lateral_friction,
            "ball_mass_kg": sc.ball_mass,
            "block_mass_kg": 1.5,
        },
        "initial_conditions": {
            "ball_radius_m": ball_r,
            "ball_start_xyz": [sc.ball_start_x, 0.0, ball_z],
            "ball_velocity_ms": list(sc.ball_velocity),
            "block_half_extents_m": list(block_h),
            "block_start_xyz": [0.3, 0.0, block_h[2]],
            "gravity_ms2": [0, 0, -9.81],
        },
        "rendering": {
            "engine": "pyrender",
            "light": "SpotLight, intensity=70, cone=0.4/1.0, pos=(-1.5,-1.5,2.5)",
            "shadows": "SHADOWS_SPOT",
            "fps": recorded_fps,
            "duration_s": sim_duration,
            "resolution": [IMG_W, IMG_H],
            "frames": len(frames),
            "codec": codec,
            "overlay_stats": show_stats_overlay,
            "overlay_scenario_label": show_scenario_label,
        },
        "physics": {
            "engine": "pybullet",
            "timestep_s": 1.0/float(SIM_HZ),
            "solver_iterations": 100,
            "spinning_friction_ball": 0.003,
            "spinning_friction_block": 0.008,
            "linear_damping_ball": 0.03,
            "linear_damping_block": 0.06,
        },
    }
    json_path = output_mp4.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if export_sample_root is not None:
        sample_dir = export_sample_root / "train" / "ball_block_dense" / sc.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_video = sample_dir / "video.mp4"
        sample_json = sample_dir / "meta.json"
        sample_npz = sample_dir / "states.npz"
        sample_video.write_bytes(output_mp4.read_bytes())
        np.savez_compressed(
            sample_npz,
            positions=np.stack(positions, axis=0),
            quats=np.stack(quats, axis=0),
            linear_velocities=np.stack(linvels, axis=0),
            angular_velocities=np.stack(angvels, axis=0),
            frame_times=np.asarray(times, dtype=np.float32),
            object_names=np.asarray(["ball", "block"]),
            object_roles=np.asarray(["dynamic", "dynamic"]),
        )
        sample_json.write_text(
            json.dumps(
                {
                    "scenario": sc.name,
                    "caption": "Ball colliding with a wooden block",
                    "video": str(sample_video),
                    "states": str(sample_npz),
                    "fps": recorded_fps,
                    "duration_s": sim_duration,
                    "frames": len(frames),
                    "camera": {
                        "eye": CAM_EYE.tolist(),
                        "target": CAM_TARGET.tolist(),
                        "up": CAM_UP.tolist(),
                        "yfov_deg": 55.0,
                    },
                    "objects": [
                        {
                            "name": "ball",
                            "shape": "sphere",
                            "role": "dynamic",
                            "position": [-1.0, 0.0, ball_z],
                            "size": {"radius": ball_r},
                            "mass": sc.ball_mass,
                            "friction": sc.lateral_friction,
                        },
                        {
                            "name": "block",
                            "shape": "box",
                            "role": "dynamic",
                            "position": [0.3, 0.0, block_h[2]],
                            "size": {"hx": block_h[0], "hy": block_h[1], "hz": block_h[2]},
                            "mass": 1.5,
                            "friction": sc.lateral_friction,
                        },
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if export_sample_root is not None:
        print(f"  -> dense sample: {sample_dir}")
    print(f"  -> {output_mp4.name} ({len(frames)} frames) + {json_path.name}")


def main():
    args = parse_args()
    video_dir = args.video_dir or VIDEO_DIR
    video_dir.mkdir(parents=True, exist_ok=True)
    record_every = resolve_record_every(args)
    selected = set(args.scenario) if args.scenario else None
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    print(f"Pyrender shadow rendering — {len(SCENARIOS)} scenarios\n")
    for i, sc in enumerate(SCENARIOS, 1):
        if selected is not None and sc.name not in selected:
            continue
        out = video_dir / f"{sc.name}.mp4"
        print(f"[{i}/{len(SCENARIOS)}] {sc.label}")
        run_scenario(
            sc,
            out,
            record_every=record_every,
            sim_duration=float(args.sim_duration),
            show_stats_overlay=args.show_stats_overlay,
            show_scenario_label=not args.hide_scenario_label,
            codec=args.codec,
            export_sample_root=args.export_sample_root,
        )

    p.disconnect()
    print(f"\nDone -> {video_dir}")


if __name__ == "__main__":
    main()
