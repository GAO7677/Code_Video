#!/usr/bin/env python3
"""球撞击木块物理仿真 — PyBullet 渲染 + 纹理 + 单点光照"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
VIDEO_DIR = OUTPUT_DIR / "videos"
TEX_DIR = OUTPUT_DIR / "textures"
FPS = 60
SIM_DURATION = 2.5
SIM_STEPS = int(SIM_DURATION * 240)
RECORD_EVERY = 4
IMG_W, IMG_H = 1280, 720

CAM_EYE = (0.05, -2.2, 1.2)
CAM_TARGET = (0.05, 0.3, 0.45)
CAM_UP = (0, 0, 1)


@dataclass
class Scenario:
    name: str
    label: str
    restitution: float
    lateral_friction: float
    ball_mass: float


SCENARIOS = [
    Scenario("e03_mu05_m1",   "e=0.3  u=0.5  m=1.0kg  塑性碰撞",    0.3, 0.5, 1.0),
    Scenario("e05_mu05_m1",   "e=0.5  u=0.5  m=1.0kg  中等弹性",    0.5, 0.5, 1.0),
    Scenario("e07_mu05_m1",   "e=0.7  u=0.5  m=1.0kg  高弹性",      0.7, 0.5, 1.0),
    Scenario("e09_mu05_m1",   "e=0.9  u=0.5  m=1.0kg  超高弹性",    0.9, 0.5, 1.0),
    Scenario("e07_mu01_m1",   "e=0.7  u=0.1  m=1.0kg  低摩擦打滑",  0.7, 0.1, 1.0),
    Scenario("e07_mu10_m1",   "e=0.7  u=1.0  m=1.0kg  高摩擦咬合",  0.7, 1.0, 1.0),
    Scenario("e07_mu05_m01",  "e=0.7  u=0.5  m=0.1kg  轻球弹飞",    0.7, 0.5, 0.1),
    Scenario("e07_mu05_m5",   "e=0.7  u=0.5  m=5.0kg  重球推动",    0.7, 0.5, 5.0),
]


# ── textures ──────────────────────────────────────────────────────

def _make_basketball(size: int = 512) -> np.ndarray:
    x = np.linspace(-1, 1, size); y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    orange = np.array([0.85, 0.42, 0.18])
    light = np.array([0.92, 0.52, 0.26])
    edge = np.clip(r, 0.3, 0.98)
    base = orange + (light - orange) * (1 - edge).reshape(size, size, 1)
    rib = np.zeros((size, size), np.float32)
    rib[np.abs(yy) < 0.015] = 1
    for s in [-1, 1]:
        rib[np.abs(xx - s * (0.28 + 0.12 * np.sin(yy * np.pi * 1.3))) < 0.012] = 1
    rib[np.abs(xx) < 0.012] = 1
    rib = cv2.GaussianBlur(rib, (3, 3), 0.5)
    np.random.seed(42)
    grain = cv2.GaussianBlur(np.random.rand(size, size).astype(np.float32) * 0.06 - 0.03, (3, 3), 0.8)
    base = np.clip(base + grain[..., None], 0, 1)
    out = base * (1 - rib[..., None] * 0.85) + np.array([0.06, 0.06, 0.08]) * rib[..., None] * 0.85
    out[r > 0.97] = [0, 0, 0]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _make_hardwood(size: int = 512) -> np.ndarray:
    np.random.seed(7)
    base = np.array([0.52, 0.33, 0.20])
    plank_w = size // 7
    tex = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(7):
        x0, x1 = i * plank_w, (i + 1) * plank_w if i < 6 else size
        tone = base + np.random.uniform(-0.05, 0.07, 3)
        plank = np.tile(tone.reshape(1, 1, 3), (size, x1 - x0, 1))
        yg = np.linspace(0, 6 * np.pi, size).reshape(-1, 1).astype(np.float32)
        grain = (np.sin(yg * 1.7 + np.random.uniform(0, 3)) * 0.05
                 + np.sin(yg * 3.5 + np.random.uniform(0, 3)) * 0.03
                 + np.random.randn(size, x1 - x0).astype(np.float32) * 0.018)
        grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 3), 0.6)
        plank = np.clip(plank + grain[..., None], 0, 1)
        tex[:, x0:x1] = plank
    tex = np.clip(tex + np.random.randn(size, size, 1).astype(np.float32) * 0.012, 0, 1)
    for i in range(1, 7):
        sx = i * plank_w
        tex[:, sx-2:sx+2] *= 0.50
    return (tex * 255).astype(np.uint8)


def _make_neutral_wall(size: int = 128) -> np.ndarray:
    y = np.linspace(0.72, 0.62, size).reshape(-1, 1).astype(np.float32)
    val = np.tile(y, (1, size))
    tex = np.stack([val, val * 0.97, val * 0.92], axis=-1)
    tex = np.clip(tex + np.random.randn(size, size, 1).astype(np.float32) * 0.008, 0, 1)
    return (tex * 255).astype(np.uint8)


def load_textures() -> dict[str, int]:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    specs = {
        "basketball": ("basketball.png", _make_basketball()),
        "hardwood":   ("hardwood.png",   _make_hardwood()),
        "neutral":    ("neutral.png",    _make_neutral_wall()),
    }
    uids = {}
    for name, (fname, arr) in specs.items():
        path = TEX_DIR / fname
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        uids[name] = p.loadTexture(str(path))
    return uids


# ── scene ──────────────────────────────────────────────────────────

def _box(half_ext, pos, tex_uid, spec=(0.03, 0.03, 0.03)):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                              rgbaColor=[1, 1, 1, 1], specularColor=spec)
    body = p.createMultiBody(0, col, vis, basePosition=pos)
    p.changeVisualShape(body, -1, textureUniqueId=tex_uid)
    return body


def build_static_scene(tex: dict[str, int]) -> list[int]:
    bodies = []
    bodies.append(_box([5, 3, 0.04], [0, 0.5, -0.04], tex["hardwood"], spec=[0.12, 0.10, 0.07]))
    bodies.append(_box([5, 0.02, 2.0], [0, 2.5, 2.0], tex["neutral"], spec=[0.01, 0.01, 0.01]))
    bodies.append(_box([0.02, 3, 2.0], [-5, 0.5, 2.0], tex["neutral"], spec=[0.01, 0.01, 0.01]))
    bodies.append(_box([0.02, 3, 2.0], [5, 0.5, 2.0], tex["neutral"], spec=[0.01, 0.01, 0.01]))
    return bodies


def add_light():
    """Single warm light panel from upper-left."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.02, 0.35])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.02, 0.35],
                              rgbaColor=[1.0, 0.95, 0.85, 1.0],
                              specularColor=[0.9, 0.85, 0.75])
    p.createMultiBody(0, col, vis, basePosition=[-2.2, -1.5, 2.4])


def create_ball(mass: float, radius: float, tex_uid: int, pos: tuple) -> int:
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                              rgbaColor=[1, 1, 1, 1], specularColor=[0.30, 0.25, 0.18])
    body = p.createMultiBody(mass, col, vis, basePosition=pos)
    p.changeVisualShape(body, -1, textureUniqueId=tex_uid)
    return body


def create_block(mass: float, half_ext: tuple, tex_uid: int, pos: tuple) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                              rgbaColor=[1, 1, 1, 1], specularColor=[0.08, 0.06, 0.04])
    body = p.createMultiBody(mass, col, vis, basePosition=pos)
    p.changeVisualShape(body, -1, textureUniqueId=tex_uid)
    return body


# ── sim ────────────────────────────────────────────────────────────

def run_scenario(sc: Scenario, tex: dict[str, int], output_mp4: Path) -> None:
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0, numSolverIterations=100, numSubSteps=1)

    static_bodies = build_static_scene(tex)
    add_light()

    ball_r, ball_z = 0.18, 0.20
    block_h = (0.25, 0.20, 0.30)
    ball_id = create_ball(sc.ball_mass, ball_r, tex["basketball"], (-1.0, 0.0, ball_z))
    block_id = create_block(1.5, block_h, tex["hardwood"], (0.3, 0.0, block_h[2]))

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

    view_m = p.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP)
    proj_m = p.computeProjectionMatrixFOV(fov=55, aspect=IMG_W / IMG_H, nearVal=0.05, farVal=30.0)

    frames = []
    for step in range(SIM_STEPS):
        p.stepSimulation()
        if step % RECORD_EVERY != 0:
            continue
        elapsed = step / 240.0
        ball_vel, _ = p.getBaseVelocity(ball_id)
        block_vel, _ = p.getBaseVelocity(block_id)

        _, _, rgba, _, _ = p.getCameraImage(
            IMG_W, IMG_H, view_m, proj_m,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
        frame = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bs, bks = float(np.linalg.norm(ball_vel)), float(np.linalg.norm(block_vel))
        for i, line in enumerate([
            f"t = {elapsed:.2f}s",
            f"Ball |v| = {bs:.2f} m/s",
            f"Block |v| = {bks:.2f} m/s",
        ]):
            y = 36 + i * 34
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (10, 10, 10), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (250, 245, 235), 1, cv2.LINE_AA)
        label = sc.label.split("\n")[0]
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.putText(frame, label, (IMG_W - tw - 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (IMG_W - tw - 20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250, 245, 235), 1, cv2.LINE_AA)

        frames.append(frame)

    for body in [ball_id, block_id] + static_bodies:
        p.removeBody(body)

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
    tex = load_textures()

    print(f"Running {len(SCENARIOS)} scenarios...\n")
    for i, sc in enumerate(SCENARIOS, 1):
        out = VIDEO_DIR / f"{sc.name}.mp4"
        print(f"[{i}/{len(SCENARIOS)}] {sc.label}")
        run_scenario(sc, tex, out)

    p.disconnect()
    print(f"\nDone -> {VIDEO_DIR}")


if __name__ == "__main__":
    main()
