#!/usr/bin/env python3
"""球撞击木块物理仿真 — 逼真纹理 + 光照"""

from __future__ import annotations

import math
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

# Closer camera for tighter framing
CAM_EYE = (0.05, -3.5, 1.8)
CAM_TARGET = (0.05, 0, 0.55)
CAM_UP = (0, 0, 1)


@dataclass
class Scenario:
    name: str
    label: str
    restitution: float
    lateral_friction: float
    ball_mass: float


SCENARIOS = [
    Scenario("e03_mu05_m1",   "恢复系数 e=0.3  摩擦 μ=0.5  球质量 1.0kg\n塑性碰撞 — 球几乎不反弹",           0.3, 0.5, 1.0),
    Scenario("e05_mu05_m1",   "恢复系数 e=0.5  摩擦 μ=0.5  球质量 1.0kg\n中等弹性 — 部分动能损失",             0.5, 0.5, 1.0),
    Scenario("e07_mu05_m1",   "恢复系数 e=0.7  摩擦 μ=0.5  球质量 1.0kg\n高弹性 — 球明显反弹",                 0.7, 0.5, 1.0),
    Scenario("e09_mu05_m1",   "恢复系数 e=0.9  摩擦 μ=0.5  球质量 1.0kg\n超高弹性 — 球快速弹飞",               0.9, 0.5, 1.0),
    Scenario("e07_mu01_m1",   "恢复系数 e=0.7  摩擦 μ=0.1  球质量 1.0kg\n低摩擦 — 碰撞打滑，切向力小",         0.7, 0.1, 1.0),
    Scenario("e07_mu10_m1",   "恢复系数 e=0.7  摩擦 μ=1.0  球质量 1.0kg\n高摩擦 — 咬合，木块被带转",           0.7, 1.0, 1.0),
    Scenario("e07_mu05_m01",  "恢复系数 e=0.7  摩擦 μ=0.5  球质量 0.1kg\n轻球 — 自己弹飞，木块几乎不动",       0.7, 0.5, 0.1),
    Scenario("e07_mu05_m5",   "恢复系数 e=0.7  摩擦 μ=0.5  球质量 5.0kg\n重球 — 推动木块滑行",                 0.7, 0.5, 5.0),
]


# ── procedural textures ────────────────────────────────────────────

def _make_wood_texture(size: int = 512) -> np.ndarray:
    """Generate wood-grain texture using sinusoidal rings + noise."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    # Rings
    grain = np.sin(r * 28 + 0.3 * np.sin(xx * 15) * np.cos(yy * 18))
    grain = (grain + 1.0) * 0.5
    # Fine noise
    noise = np.random.randn(size, size).astype(np.float32) * 0.04
    grain = np.clip(grain + noise, 0, 1)
    # Color: warm wood tones
    base = np.array([0.55, 0.32, 0.17], dtype=np.float32)  # dark wood
    light = np.array([0.78, 0.52, 0.28], dtype=np.float32)  # light grain
    tex = base + (light - base) * grain[..., None]
    tex = np.clip(tex + np.random.randn(size, size, 1).astype(np.float32) * 0.03, 0, 1)
    return (tex * 255).astype(np.uint8)


def _make_metal_texture(size: int = 512) -> np.ndarray:
    """Brushed metal texture."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    # Brushed lines
    brush = np.sin(yy * 200 + np.sin(xx * 40) * 3.0) * 0.5 + 0.5
    brush = brush * 0.15 + 0.55  # mid-grey base
    # Noise
    noise = np.random.randn(size, size).astype(np.float32) * 0.03
    val = np.clip(brush + noise, 0, 1)
    # Slight bluish tint
    tex = np.stack([val, val, val * 1.05], axis=-1)
    tex = np.clip(tex, 0, 1)
    return (tex * 255).astype(np.uint8)


def _make_concrete_texture(size: int = 512) -> np.ndarray:
    """Concrete/stone ground texture."""
    noise1 = np.random.randn(size // 8, size // 8).astype(np.float32)
    noise1 = cv2.resize(noise1, (size, size), interpolation=cv2.INTER_CUBIC)
    noise2 = np.random.randn(size // 2, size // 2).astype(np.float32)
    noise2 = cv2.resize(noise2, (size, size), interpolation=cv2.INTER_CUBIC)
    val = noise1 * 0.12 + noise2 * 0.05 + 0.55
    val = np.clip(val, 0, 1)
    # Warm grey
    tex = np.stack([val, val * 0.96, val * 0.90], axis=-1)
    tex = np.clip(tex + np.random.randn(size, size, 1).astype(np.float32) * 0.02, 0, 1)
    return (tex * 255).astype(np.uint8)


def _make_sky_texture(size: int = 512) -> np.ndarray:
    """Soft gradient for background / sky."""
    y = np.linspace(0, 1, size).reshape(-1, 1).astype(np.float32)
    # Top: warm sky, bottom: slightly darker
    top = np.array([0.45, 0.62, 0.78])
    bot = np.array([0.65, 0.60, 0.52])
    tex = bot + (top - bot) * y
    tex = np.clip(tex + np.random.randn(size, 1, 3).astype(np.float32) * 0.01, 0, 1)
    tex_big = np.tile(tex, (1, size, 1))
    return (tex_big * 255).astype(np.uint8)


def generate_textures() -> dict[str, int]:
    """Generate procedural textures, save to disk, return name→uid map."""
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    textures = {
        "wood": ("wood.png", _make_wood_texture()),
        "metal": ("metal.png", _make_metal_texture()),
        "concrete": ("concrete.png", _make_concrete_texture()),
        "sky": ("sky.png", _make_sky_texture()),
    }
    uids: dict[str, int] = {}
    for name, (fname, arr) in textures.items():
        path = TEX_DIR / fname
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        uid = p.loadTexture(str(path))
        uids[name] = uid
    return uids


# ── scene construction ─────────────────────────────────────────────

def _box_body(half_ext, pos, tex_uid: int, specular=(0.05, 0.05, 0.05)):
    """Helper: create a textured box body."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                              rgbaColor=[1, 1, 1, 1], specularColor=specular)
    body = p.createMultiBody(0, col, vis, basePosition=pos)
    p.changeVisualShape(body, -1, textureUniqueId=tex_uid)
    return body


def build_scene(tex: dict[str, int]):
    """Build the full 3D scene with textured objects and walls."""
    _box_body([8, 4, 0.05], [0, 0, -0.05], tex["concrete"], specular=[0.04, 0.04, 0.04])   # ground
    _box_body([8, 0.02, 2.5], [0, -3.5, 2.5], tex["sky"], specular=[0.01, 0.01, 0.01])     # back wall
    _box_body([0.02, 4, 2.5], [-8, 0, 2.5], tex["sky"], specular=[0.01, 0.01, 0.01])        # left wall
    _box_body([0.02, 4, 2.5], [8, 0, 2.5], tex["sky"], specular=[0.01, 0.01, 0.01])         # right wall


def create_ball(mass: float, radius: float, tex_uid: int, start_pos: tuple) -> int:
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                              rgbaColor=[1, 1, 1, 1], specularColor=[0.55, 0.55, 0.6])
    body = p.createMultiBody(mass, col, vis, basePosition=start_pos)
    p.changeVisualShape(body, -1, textureUniqueId=tex_uid)
    return body


def create_block(mass: float, half_ext: tuple, tex_uid: int, start_pos: tuple) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_ext)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_ext,
                              rgbaColor=[1, 1, 1, 1], specularColor=[0.06, 0.04, 0.03])
    body = p.createMultiBody(mass, col, vis, basePosition=start_pos)
    p.changeVisualShape(body, -1, textureUniqueId=tex_uid)
    return body


# ── main simulation ────────────────────────────────────────────────

def run_scenario(scenario: Scenario, tex: dict[str, int], output_mp4: Path) -> None:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 240.0,
        numSolverIterations=100,
        numSubSteps=1,
    )

    build_scene(tex)

    ball_radius = 0.18
    ball_start = (-1.6, 0.0, ball_radius + 0.02)
    block_half = (0.25, 0.20, 0.30)
    block_start = (0.3, 0.0, block_half[2])

    ball_id = create_ball(scenario.ball_mass, ball_radius, tex["metal"], ball_start)
    block_id = create_block(1.5, block_half, tex["wood"], block_start)

    p.changeDynamics(ball_id, -1, restitution=scenario.restitution,
                     lateralFriction=scenario.lateral_friction,
                     spinningFriction=0.003, linearDamping=0.03, angularDamping=0.03)
    p.changeDynamics(block_id, -1, restitution=scenario.restitution,
                     lateralFriction=scenario.lateral_friction,
                     spinningFriction=0.008, linearDamping=0.06, angularDamping=0.06,
                     activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)

    p.resetBaseVelocity(ball_id, linearVelocity=[3.5, 0.0, 1.8])

    for _ in range(10):
        p.stepSimulation()

    view_m = p.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP)
    proj_m = p.computeProjectionMatrixFOV(fov=50, aspect=IMG_W / IMG_H, nearVal=0.05, farVal=30.0)

    frames = []
    for step in range(SIM_STEPS):
        p.stepSimulation()
        if step % RECORD_EVERY != 0:
            continue

        elapsed = step / 240.0
        ball_pos, _ = p.getBasePositionAndOrientation(ball_id)
        ball_vel, _ = p.getBaseVelocity(ball_id)
        block_pos, _ = p.getBasePositionAndOrientation(block_id)
        block_vel, _ = p.getBaseVelocity(block_id)

        _, _, rgba, _, _ = p.getCameraImage(
            IMG_W, IMG_H, view_m, proj_m,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
        frame = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ── HUD ──
        ball_speed = float(np.linalg.norm(ball_vel))
        block_speed = float(np.linalg.norm(block_vel))
        lines_info = [
            f"t = {elapsed:.2f}s",
            f"Ball  |v| = {ball_speed:.2f} m/s",
            f"Block |v| = {block_speed:.2f} m/s",
        ]
        for i, line in enumerate(lines_info):
            y = 42 + i * 36
            cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 240, 230), 1, cv2.LINE_AA)

        param_lines = scenario.label.split("\n")
        for i, line in enumerate(param_lines):
            y = 40 + i * 26
            text_w = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            x = IMG_W - text_w - 24
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 240, 230), 1, cv2.LINE_AA)

        frames.append(frame)

    p.disconnect()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_mp4), fourcc, FPS, (IMG_W, IMG_H))
    for f in frames:
        out.write(f)
    out.release()
    print(f"  → {output_mp4.name} ({len(frames)} frames)")


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Generate textures once (need a temp connection for loadTexture)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    tex = generate_textures()
    p.disconnect()

    print(f"Running {len(SCENARIOS)} scenarios...")
    print(f"Gravity: 9.81 m/s^2  |  Initial velocity: (3.5, 0, 1.8) m/s")
    print(f"FPS: {FPS}  |  Duration: {SIM_DURATION}s\n")

    for i, sc in enumerate(SCENARIOS, 1):
        out = VIDEO_DIR / f"{sc.name}.mp4"
        print(f"[{i}/{len(SCENARIOS)}] {sc.label.split(chr(10))[0]}")
        run_scenario(sc, tex, out)

    print(f"\nDone → {VIDEO_DIR}")


if __name__ == "__main__":
    main()
