#!/usr/bin/env python3
"""球撞击木块物理仿真 — PyBullet 物理 + Pyrender PBR 渲染"""

from __future__ import annotations

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import pyrender
import trimesh

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
VIDEO_DIR = OUTPUT_DIR / "videos"
FPS = 60
SIM_DURATION = 2.5
SIM_STEPS = int(SIM_DURATION * 240)
RECORD_EVERY = 4
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


SCENARIOS = [
    Scenario("e03_mu05_m1",   "e=0.3  u=0.5  m=1.0kg  plastic",   0.3, 0.5, 1.0),
    Scenario("e05_mu05_m1",   "e=0.5  u=0.5  m=1.0kg  medium",    0.5, 0.5, 1.0),
    Scenario("e07_mu05_m1",   "e=0.7  u=0.5  m=1.0kg  bouncy",    0.7, 0.5, 1.0),
    Scenario("e09_mu05_m1",   "e=0.9  u=0.5  m=1.0kg  superball", 0.9, 0.5, 1.0),
    Scenario("e07_mu01_m1",   "e=0.7  u=0.1  m=1.0kg  low-fric",  0.7, 0.1, 1.0),
    Scenario("e07_mu10_m1",   "e=0.7  u=1.0  m=1.0kg  high-fric", 0.7, 1.0, 1.0),
    Scenario("e07_mu05_m01",  "e=0.7  u=0.5  m=0.1kg  light-ball",0.7, 0.5, 0.1),
    Scenario("e07_mu05_m5",   "e=0.7  u=0.5  m=5.0kg  heavy-ball",0.7, 0.5, 5.0),
]


# ── vertex color textures ─────────────────────────────────────────

def _make_wood_color(n_verts: int) -> np.ndarray:
    """Generate wood-like vertex colors for a mesh."""
    np.random.seed(7)
    base = np.array([0.52, 0.34, 0.20], dtype=np.float32)
    colors = np.tile(base, (n_verts, 1))
    colors += np.random.uniform(-0.06, 0.06, (n_verts, 3)).astype(np.float32)
    return np.clip(colors, 0, 1)


def _make_basketball_color(n_verts: int, vertices: np.ndarray) -> np.ndarray:
    """Generate basketball-like vertex colors for a sphere."""
    np.random.seed(42)
    base = np.array([0.85, 0.42, 0.18], dtype=np.float32)
    colors = np.tile(base, (n_verts, 1))
    colors += np.random.uniform(-0.04, 0.04, (n_verts, 3)).astype(np.float32)
    # Simulate black ribs at equator and two arcs
    y = vertices[:, 2]  # Z is up in trimesh sphere
    x = vertices[:, 0]
    z = vertices[:, 1]
    r2d = np.sqrt(x**2 + z**2)
    # Equator band
    near_equator = np.abs(y) < 0.03
    colors[near_equator] = [0.06, 0.06, 0.08]
    # Two arc bands
    for s in [-1, 1]:
        arc_dist = np.abs(x - s * (0.28 + 0.12 * np.sin(y * np.pi * 1.3)) * r2d)
        near_arc = (arc_dist < 0.02) & (r2d > 0.05)
        colors[near_arc] = [0.06, 0.06, 0.08]
    # Vertical band
    near_vert = np.abs(x) < 0.02
    colors[near_vert] = [0.06, 0.06, 0.08]
    return np.clip(colors, 0, 1)


def _make_concrete_color(n_verts: int) -> np.ndarray:
    """Warm grey vertex colors for floor."""
    np.random.seed(3)
    base = np.array([0.62, 0.55, 0.48], dtype=np.float32)
    colors = np.tile(base, (n_verts, 1))
    colors += np.random.uniform(-0.03, 0.03, (n_verts, 3)).astype(np.float32)
    return np.clip(colors, 0, 1)


# ── Pyrender scene and renderer (reused across frames) ────────────

def _look_at_pose(eye, target, up):
    """Build a 4x4 camera-to-world pose from look-at parameters."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


class SceneRenderer:
    """Reusable Pyrender renderer with shadows."""

    def __init__(self):
        self.scene = pyrender.Scene(bg_color=[0.25, 0.28, 0.35], ambient_light=[0.15, 0.15, 0.18])

        # ── floor ──
        floor = trimesh.creation.box(extents=[5, 3, 0.04])
        floor.visual.vertex_colors = _make_concrete_color(len(floor.vertices))
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor, smooth=False)
        floor_node = self.scene.add(self.floor_mesh, pose=self._tr([0, 0.5, -0.04]))

        # ── back wall ──
        wall = trimesh.creation.box(extents=[5, 0.02, 2.0])
        wall.visual.vertex_colors = np.tile([0.55, 0.58, 0.62], (len(wall.vertices), 1))
        self.wall_mesh = pyrender.Mesh.from_trimesh(wall, smooth=False)
        self.scene.add(self.wall_mesh, pose=self._tr([0, 2.5, 2.0]))

        # ── side walls ──
        sw = trimesh.creation.box(extents=[0.02, 3, 2.0])
        sw.visual.vertex_colors = np.tile([0.50, 0.53, 0.57], (len(sw.vertices), 1))
        self.sw_mesh = pyrender.Mesh.from_trimesh(sw, smooth=False)
        self.scene.add(self.sw_mesh, pose=self._tr([-5, 0.5, 2.0]))
        self.scene.add(self.sw_mesh, pose=self._tr([5, 0.5, 2.0]))

        # ── key light (warm, casts shadows) ──
        key_light = pyrender.SpotLight(
            color=[1.0, 0.95, 0.85], intensity=18.0,
            innerConeAngle=np.pi/6, outerConeAngle=np.pi/4,
        )
        light_pose = _look_at_pose([-2.5, -1.5, 3.5], [0, 0.3, 0.3], [0, 0, 1])
        self.scene.add(key_light, pose=light_pose)

        # ── fill light (cool, no shadow for performance) ──
        # fill = pyrender.PointLight(color=[0.7, 0.8, 1.0], intensity=4.0)
        # self.scene.add(fill, pose=self._tr([2.0, -1.2, 2.5]))

        # ── camera ──
        self.cam = pyrender.PerspectiveCamera(yfov=np.radians(55), aspectRatio=IMG_W/IMG_H)
        cam_pose = _look_at_pose(CAM_EYE, CAM_TARGET, CAM_UP)
        self.cam_node = self.scene.add(self.cam, pose=cam_pose)

        # ── renderer ──
        self.renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)

        # Dynamic object slots
        self.ball_node = None
        self.block_node = None

    @staticmethod
    def _tr(pos):
        return np.array([[1,0,0,pos[0]],[0,1,0,pos[1]],[0,0,1,pos[2]],[0,0,0,1]], dtype=np.float64)

    def set_ball(self, pos, quat, radius=0.18):
        if self.ball_node is not None:
            self.scene.remove_node(self.ball_node)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        sphere.visual.vertex_colors = _make_basketball_color(len(sphere.vertices), sphere.vertices)
        mesh = pyrender.Mesh.from_trimesh(sphere, smooth=True)
        pose = self._pose_from_pybullet(pos, quat)
        self.ball_node = self.scene.add(mesh, pose=pose)

    def set_block(self, pos, quat, half_ext=(0.25, 0.20, 0.30)):
        if self.block_node is not None:
            self.scene.remove_node(self.block_node)
        box = trimesh.creation.box(extents=[2*half_ext[0], 2*half_ext[1], 2*half_ext[2]])
        box.visual.vertex_colors = _make_wood_color(len(box.vertices))
        mesh = pyrender.Mesh.from_trimesh(box, smooth=False)
        pose = self._pose_from_pybullet(pos, quat)
        self.block_node = self.scene.add(mesh, pose=pose)

    @staticmethod
    def _pose_from_pybullet(pos, quat):
        """PyBullet quaternion (x,y,z,w) to 4x4 transform."""
        from pybullet import getMatrixFromQuaternion
        mat = np.array(getMatrixFromQuaternion(quat)).reshape(3, 3)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = mat
        pose[:3, 3] = pos
        return pose

    def render(self):
        color, _ = self.renderer.render(self.scene)
        return color

    def cleanup(self):
        self.renderer.delete()


# ── simulation ────────────────────────────────────────────────────

def run_scenario(sc: Scenario, output_mp4: Path) -> None:
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSolverIterations=100, numSubSteps=1)

    ball_r, ball_z = 0.18, 0.20
    block_h = (0.25, 0.20, 0.30)

    col_b = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_r)
    ball_id = p.createMultiBody(sc.ball_mass, col_b, basePosition=(-1.0, 0.0, ball_z))
    col_bk = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_h)
    block_id = p.createMultiBody(1.5, col_bk, basePosition=(0.3, 0.0, block_h[2]))

    p.changeDynamics(ball_id, -1, restitution=sc.restitution,
                     lateralFriction=sc.lateral_friction,
                     spinningFriction=0.003, linearDamping=0.03, angularDamping=0.03)
    p.changeDynamics(block_id, -1, restitution=sc.restitution,
                     lateralFriction=sc.lateral_friction,
                     spinningFriction=0.008, linearDamping=0.06, angularDamping=0.06,
                     activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
    p.resetBaseVelocity(ball_id, linearVelocity=[3.5, 0.0, 1.8])

    for _ in range(10):
        p.stepSimulation()

    renderer = SceneRenderer()
    frames = []

    for step in range(SIM_STEPS):
        p.stepSimulation()
        if step % RECORD_EVERY != 0:
            continue
        elapsed = step / 240.0

        ball_pos, ball_quat = p.getBasePositionAndOrientation(ball_id)
        ball_vel, _ = p.getBaseVelocity(ball_id)
        block_pos, block_quat = p.getBasePositionAndOrientation(block_id)
        block_vel, _ = p.getBaseVelocity(block_id)

        renderer.set_ball(ball_pos, ball_quat, ball_r)
        renderer.set_block(block_pos, block_quat, block_h)
        color = renderer.render()

        frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        bs, bks = float(np.linalg.norm(ball_vel)), float(np.linalg.norm(block_vel))
        for i, line in enumerate([
            f"t = {elapsed:.2f}s",
            f"Ball |v| = {bs:.2f} m/s",
            f"Block |v| = {bks:.2f} m/s",
        ]):
            y = 36 + i*34
            cv2.putText(frame, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (10,10,10), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (250,245,235), 1, cv2.LINE_AA)
        label = sc.label.split("\n")[0]
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.putText(frame, label, (IMG_W-tw-20,36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10,10,10), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (IMG_W-tw-20,36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250,245,235), 1, cv2.LINE_AA)
        frames.append(frame)

    renderer.cleanup()
    p.removeBody(ball_id)
    p.removeBody(block_id)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (IMG_W, IMG_H))
    for f in frames:
        out.write(f)
    out.release()
    print(f"  -> {output_mp4.name} ({len(frames)} frames)")


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")  # physics ground, loaded once

    print(f"Running {len(SCENARIOS)} scenarios with Pyrender...\n")
    for i, sc in enumerate(SCENARIOS, 1):
        out = VIDEO_DIR / f"{sc.name}.mp4"
        print(f"[{i}/{len(SCENARIOS)}] {sc.label}")
        run_scenario(sc, out)

    p.disconnect()
    print(f"\nDone -> {VIDEO_DIR}")


if __name__ == "__main__":
    main()
