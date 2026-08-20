#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import numpy as np

np.infty = np.inf

import pybullet as p
import pybullet_data
import pyrender
import trimesh
from pyrender.constants import RenderFlags


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/preview_v1/industrial")
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
VIDEO_DIR = OUTPUT_ROOT / "videos"
META_DIR = OUTPUT_ROOT / "meta"
DEFAULT_PORT = 18825
DEFAULT_THEME = "industrial"

IMG_W = 960
IMG_H = 540
SIM_HZ = 240
FPS = 30
SIM_DURATION = 3.0
RECORD_EVERY = max(1, SIM_HZ // FPS)
PHYSICS_SOLVER_ITERATIONS = 120
PHYSICS_SUB_STEPS = 2
PHYSICS_CONTACT_ERP = 0.20

CAM_EYE = np.array([0.0, -3.0, 1.42], dtype=np.float64)
CAM_TARGET = np.array([0.0, 0.28, 0.38], dtype=np.float64)
CAM_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
EARTH_GRAVITY = 9.81
ACTIVE_THEME = DEFAULT_THEME
TEXTURE_ROOT = Path("/data/gaoya/dataset/textures/polyhaven_wood")
WOOD_TEXTURES = {
    "wood_planks": TEXTURE_ROOT / "wood_planks_diff_4k.jpg",
    "plywood": TEXTURE_ROOT / "plywood_diff_4k.jpg",
    "dark_wood": TEXTURE_ROOT / "dark_wood_diff_4k.jpg",
    "weathered_planks": TEXTURE_ROOT / "weathered_brown_planks_diff_4k.jpg",
    "wood_floor": Path("/data/gaoya/dataset/blender_render_assets/polyhaven_v1/textures/wood_floor/wood_floor_diff_2k.jpg"),
    "rubber_grain": Path("/data/gaoya/dataset/blender_render_assets/polyhaven_v1/textures/brown_leather/brown_leather_albedo_2k.jpg"),
}
TEXTURE_CACHE: Dict[str, np.ndarray] = {}
FFMPEG_CANDIDATES = [
    "ffmpeg",
    "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
    "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
]

INITIALIZATION_PENETRATION_TOLERANCE_M = 0.001
INITIALIZATION_NEAR_CONTACT_DISTANCE_M = 0.003


class InitializationPenetrationError(RuntimeError):
    """Raised when a generated scene starts with a measurable overlap."""


def inspect_initialization_contacts(
    body_ids: Dict[str, int],
    plane_id: int,
    *,
    stage: str,
    penetration_tolerance_m: float = INITIALIZATION_PENETRATION_TOLERANCE_M,
    near_contact_distance_m: float = INITIALIZATION_NEAR_CONTACT_DISTANCE_M,
) -> dict:
    """Inspect every object pair and object-floor pair at one simulator stage.

    The check deliberately includes pairs whose collision response is disabled
    by a constraint.  A constraint may remove a response, but it must not hide
    an invalid initial geometry.
    """
    names = list(body_ids)
    pair_records: list[dict] = []

    def collect(left_name: str, left_id: int, right_name: str, right_id: int) -> None:
        points = p.getClosestPoints(
            int(left_id),
            int(right_id),
            distance=float(near_contact_distance_m),
        )
        if not points:
            return
        distance = min(float(point[8]) for point in points)
        if distance <= float(near_contact_distance_m):
            pair_records.append(
                {
                    "body_a": left_name,
                    "body_b": right_name,
                    "distance_m": round(distance, 6),
                    "penetrating": bool(distance < -float(penetration_tolerance_m)),
                }
            )

    for name in names:
        collect(name, body_ids[name], "ground", plane_id)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            collect(left_name, body_ids[left_name], right_name, body_ids[right_name])

    penetrations = [record for record in pair_records if record["penetrating"]]
    return {
        "stage": stage,
        "passed": not penetrations,
        "penetration_tolerance_m": float(penetration_tolerance_m),
        "near_contact_distance_m": float(near_contact_distance_m),
        "checked_object_count": len(names),
        "checked_pair_count": len(names) + (len(names) * (len(names) - 1)) // 2,
        "near_contacts": pair_records,
        "penetrations": penetrations,
    }


def assert_initialization_contacts(
    body_ids: Dict[str, int],
    plane_id: int,
    *,
    stage: str,
) -> dict:
    report = inspect_initialization_contacts(body_ids, plane_id, stage=stage)
    if not report["passed"]:
        details = "; ".join(
            f"{item['body_a']} vs {item['body_b']}={item['distance_m']:.6f}m"
            for item in report["penetrations"]
        )
        raise InitializationPenetrationError(
            f"initialization penetration at {stage}: {details} "
            f"(tolerance={INITIALIZATION_PENETRATION_TOLERANCE_M:.4f}m)"
        )
    return report


def assert_initialization_contact_contract(
    body_ids: Dict[str, int],
    plane_id: int,
    contract: dict | None,
    *,
    stage: str,
) -> dict:
    """Require declared connected parts to be tangent at an init stage.

    ``assert_initialization_contacts`` catches overlaps for every pair.  This
    companion check catches the opposite failure for the small set of pairs
    that are declared as one connected support or assembly.
    """
    contract = contract or {}
    gap_tolerance_m = float(contract.get("gap_tolerance_m", 0.001))
    penetration_tolerance_m = float(
        contract.get("penetration_tolerance_m", INITIALIZATION_PENETRATION_TOLERANCE_M)
    )
    checks: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    query_distance_m = max(INITIALIZATION_NEAR_CONTACT_DISTANCE_M, gap_tolerance_m * 2.0)

    def check_pair(left_name: str, left_id: int, right_name: str, right_id: int) -> None:
        points = p.getClosestPoints(
            int(left_id),
            int(right_id),
            distance=query_distance_m,
        )
        distance = min((float(point[8]) for point in points), default=float("inf"))
        record = {
            "body_a": left_name,
            "body_b": right_name,
            "distance_m": None if not math.isfinite(distance) else round(distance, 6),
            "gap_tolerance_m": gap_tolerance_m,
            "passed": bool(
                math.isfinite(distance)
                and distance <= gap_tolerance_m
                and distance >= -penetration_tolerance_m
            ),
        }
        checks.append(record)
        if not record["passed"]:
            failures.append(record)

    def require_body(name: str) -> int:
        if name not in body_ids:
            raise InitializationPenetrationError(
                f"initialization contact contract references missing body {name!r} at {stage}"
            )
        return int(body_ids[name])

    for pair in contract.get("touching_pairs", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise InitializationPenetrationError(f"invalid touching pair {pair!r} at {stage}")
        left_name, right_name = str(pair[0]), str(pair[1])
        check_pair(left_name, require_body(left_name), right_name, require_body(right_name))
    for name_value in contract.get("touching_ground", []):
        name = str(name_value)
        check_pair(name, require_body(name), "ground", int(plane_id))

    if failures:
        details = "; ".join(
            f"{item['body_a']} vs {item['body_b']}={item['distance_m']}m"
            for item in failures
        )
        raise InitializationPenetrationError(
            f"initialization contact gap/penetration at {stage}: {details}"
        )
    return {
        "stage": stage,
        "passed": True,
        "contract_checks": checks,
        "gap_tolerance_m": gap_tolerance_m,
        "penetration_tolerance_m": penetration_tolerance_m,
    }


def collect_frame_contacts(
    body_ids: Dict[str, int],
    plane_id: int,
    *,
    frame_index: int,
    time_s: float,
    dynamic_names: set[str] | None = None,
    max_points_per_pair: int = 5,
) -> list[dict]:
    """Return sampled PyBullet contacts in a RigidBench-compatible form.

    The normal follows PyBullet's convention: it points from ``obj_b`` toward
    ``obj_a``.  Static-static fixture contacts are omitted when
    ``dynamic_names`` is supplied so exported labels focus on motion-relevant
    interactions.
    """
    if max_points_per_pair <= 0:
        raise ValueError("max_points_per_pair must be positive")
    name_by_body_id = {int(body_id): name for name, body_id in body_ids.items()}
    name_by_body_id[int(plane_id)] = "ground"
    grouped: dict[tuple[str, str], list[dict]] = {}
    for point in p.getContactPoints():
        body_a = int(point[1])
        body_b = int(point[2])
        obj_a = name_by_body_id.get(body_a)
        obj_b = name_by_body_id.get(body_b)
        if obj_a is None or obj_b is None:
            continue
        if dynamic_names is not None and obj_a not in dynamic_names and obj_b not in dynamic_names:
            continue
        key = (obj_a, obj_b)
        contacts = grouped.setdefault(key, [])
        if len(contacts) >= max_points_per_pair:
            continue
        contacts.append(
            {
                # Match the compact RigidBench fields while retaining raw
                # solver quantities needed for contact-supervised training.
                "point": [float(value) for value in point[6]],
                "normal": [float(value) for value in point[7]],
                "position_on_a": [float(value) for value in point[5]],
                "position_on_b": [float(value) for value in point[6]],
                "distance_m": float(point[8]),
                "normal_force_n": float(point[9]),
            }
        )
    return [
        {
            "frame": int(frame_index),
            "time_s": float(time_s),
            "obj_a": obj_a,
            "obj_b": obj_b,
            "contacts": contacts,
        }
        for (obj_a, obj_b), contacts in grouped.items()
    ]


@dataclass
class ObjectSpec:
    name: str
    shape: str
    color: List[float]
    mass: float
    position: List[float]
    size: Dict[str, float]
    dynamic: bool = True
    restitution: float = 0.55
    friction: float = 0.55
    linear_damping: float = 0.03
    angular_damping: float = 0.03
    orientation_euler_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    linear_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    role: str = "dynamic"
    texture_style: str = "solid"
    texture_asset: str = ""


@dataclass
class ScenarioSpec:
    key: str
    family: str
    title: str
    description: str
    gravity: float
    floor_friction: float
    objects: List[ObjectSpec]
    seed: int
    pre_roll_s: float = 0.75
    sim_type: str = "rigid_simple"
    floor_restitution: float = 0.02


THEME_LABELS = {
    "industrial": "工业训练数据",
    "daily_objects": "日常物体",
}


def _quat_from_euler_deg(values: List[float]) -> List[float]:
    radians = [math.radians(v) for v in values]
    return list(p.getQuaternionFromEuler(radians))


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def _tr(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return pose


def _pb_pose(pos: List[float], quat: List[float]) -> np.ndarray:
    rot = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    return pose


def _load_texture_rgb(texture_asset: str) -> np.ndarray | None:
    if not texture_asset:
        return None
    if texture_asset in TEXTURE_CACHE:
        return TEXTURE_CACHE[texture_asset]
    path = WOOD_TEXTURES.get(texture_asset)
    if path is None or not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    TEXTURE_CACHE[texture_asset] = rgb
    return rgb


def _sample_texture(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = texture.shape[:2]
    uv = uv - np.floor(uv)
    x = np.clip((uv[:, 0] * (w - 1)).astype(np.int32), 0, w - 1)
    y = np.clip(((1.0 - uv[:, 1]) * (h - 1)).astype(np.int32), 0, h - 1)
    return texture[y, x]


def _resolve_ffmpeg() -> str:
    for candidate in FFMPEG_CANDIDATES:
        resolved = shutil.which(candidate) if os.path.sep not in candidate else candidate
        if resolved and Path(resolved).exists():
            return resolved
    raise RuntimeError(
        "ffmpeg with libx264 support is required to export H.264 videos, but no usable binary was found"
    )


def _write_video_h264(output_mp4: Path, frames: List[np.ndarray]) -> None:
    if not frames:
        raise ValueError(f"cannot encode an empty video: {output_mp4}")
    frame_height, frame_width = frames[0].shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"invalid video frame shape for {output_mp4}: {frames[0].shape}")
    if any(frame.shape[:2] != (frame_height, frame_width) for frame in frames):
        raise ValueError(f"inconsistent frame sizes while encoding {output_mp4}")
    tmp_mp4v = output_mp4.with_suffix(".tmp_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(tmp_mp4v),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open temporary mp4v writer for {tmp_mp4v}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    ffmpeg_bin = _resolve_ffmpeg()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(tmp_mp4v),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg H.264 transcode failed for {output_mp4}: {exc.stderr}") from exc
    finally:
        if tmp_mp4v.exists():
            tmp_mp4v.unlink()


def _textured_vertex_colors(mesh: trimesh.Trimesh, texture_asset: str, tint: np.ndarray, repeat: float = 2.2) -> np.ndarray | None:
    texture = _load_texture_rgb(texture_asset)
    if texture is None:
        return None
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if len(verts) == 0 or len(normals) == 0:
        return None

    abs_normals = np.abs(normals)
    dominant = np.argmax(abs_normals, axis=1)
    mins = verts.min(axis=0)
    spans = np.maximum(verts.max(axis=0) - mins, 1e-6)

    uv_xy = np.stack([(verts[:, 0] - mins[0]) / spans[0], (verts[:, 1] - mins[1]) / spans[1]], axis=1) * repeat
    uv_xz = np.stack([(verts[:, 0] - mins[0]) / spans[0], (verts[:, 2] - mins[2]) / spans[2]], axis=1) * repeat
    uv_yz = np.stack([(verts[:, 1] - mins[1]) / spans[1], (verts[:, 2] - mins[2]) / spans[2]], axis=1) * repeat

    colors = np.zeros((len(verts), 3), dtype=np.float32)
    idx_x = dominant == 0
    idx_y = dominant == 1
    idx_z = dominant == 2
    if np.any(idx_x):
        colors[idx_x] = _sample_texture(texture, uv_yz[idx_x])
    if np.any(idx_y):
        colors[idx_y] = _sample_texture(texture, uv_xz[idx_y])
    if np.any(idx_z):
        colors[idx_z] = _sample_texture(texture, uv_xy[idx_z])

    brightness = 0.88 + 0.12 * (verts[:, 2] - mins[2]) / spans[2]
    colors = colors * brightness[:, None]
    colors = np.clip(colors * (0.82 + 0.18 * tint[None, :]), 0.0, 1.0)
    return colors


def _apply_procedural_material(mesh: trimesh.Trimesh, obj: ObjectSpec) -> None:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    base = np.tile(np.asarray(obj.color, dtype=np.float32), (len(verts), 1))
    style = obj.texture_style
    if style == "solid":
        colors = base
    elif style == "wood_real":
        sampled = _textured_vertex_colors(mesh, obj.texture_asset, np.asarray(obj.color, dtype=np.float32))
        if sampled is None:
            grain = 0.5 + 0.5 * np.sin(verts[:, 2] * 24.0 + verts[:, 0] * 5.0)
            colors = base.copy()
            colors[:, 0] += grain * 0.10
            colors[:, 1] += grain * 0.06
        else:
            colors = sampled
    elif style in {"fabric_real", "paper_real"}:
        sampled = _textured_vertex_colors(mesh, obj.texture_asset, np.asarray(obj.color, dtype=np.float32))
        if sampled is None:
            warp = 0.5 + 0.5 * np.sin(verts[:, 0] * 18.0 + verts[:, 1] * 12.0)
            weft = 0.5 + 0.5 * np.cos(verts[:, 1] * 16.0 + verts[:, 2] * 10.0)
            colors = base.copy()
            colors[:, 0] = np.clip(colors[:, 0] * (0.86 + 0.10 * warp), 0.0, 1.0)
            colors[:, 1] = np.clip(colors[:, 1] * (0.84 + 0.10 * weft), 0.0, 1.0)
            colors[:, 2] = np.clip(colors[:, 2] * (0.88 + 0.08 * warp), 0.0, 1.0)
        else:
            colors = sampled
    elif style == "rubber_real":
        sampled = _textured_vertex_colors(
            mesh,
            obj.texture_asset,
            np.asarray(obj.color, dtype=np.float32),
            repeat=2.2,
        )
        if sampled is None:
            grain = 0.5 + 0.5 * np.sin(verts[:, 0] * 18.0 + verts[:, 1] * 13.0 + verts[:, 2] * 9.0)
        else:
            grain = np.mean(sampled, axis=1)
            grain = (grain - float(grain.min())) / max(float(np.ptp(grain)), 1e-6)
        colors = base * (0.76 + 0.22 * grain[:, None])
        colors += np.asarray(obj.color, dtype=np.float32)[None, :] * (0.02 * grain[:, None])
    elif style == "painted":
        band = 0.5 + 0.5 * np.sin(verts[:, 2] * 11.0 + verts[:, 0] * 2.3)
        streak = 0.5 + 0.5 * np.sin(verts[:, 0] * 9.0 + verts[:, 1] * 5.0 + verts[:, 2] * 3.0)
        topcoat = np.array([0.96, 0.95, 0.92], dtype=np.float32)
        colors = base * (0.74 + 0.16 * band[:, None])
        colors = colors * (0.88 + 0.08 * streak[:, None])
        colors = colors * 0.82 + topcoat[None, :] * (0.18 * band[:, None])
    elif style == "stripe":
        stripe = (np.sin(verts[:, 0] * 22.0 + verts[:, 1] * 4.0) > 0).astype(np.float32)
        stripe = 0.25 + 0.75 * stripe
        accent = np.clip(np.asarray(obj.color, dtype=np.float32) * np.array([1.10, 0.92, 0.82], dtype=np.float32), 0.0, 1.0)
        light = np.array([0.95, 0.94, 0.90], dtype=np.float32)
        colors = accent[None, :] * stripe[:, None] + light[None, :] * (1.0 - stripe[:, None])
        colors = colors * (0.88 + 0.08 * (0.5 + 0.5 * np.sin(verts[:, 2] * 4.0)))
    elif style == "two_tone":
        split = (verts[:, 1] > np.median(verts[:, 1])).astype(np.float32)
        alt = np.clip(
            np.array(
                [
                    min(1.0, obj.color[2] * 0.90 + 0.20),
                    min(1.0, obj.color[0] * 0.70 + 0.18),
                    min(1.0, obj.color[1] * 0.75 + 0.18),
                ],
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        colors = base * split[:, None] + alt[None, :] * (1.0 - split[:, None])
    elif style == "label":
        label = (np.abs(verts[:, 2]) < 0.045).astype(np.float32)
        label *= (np.cos(verts[:, 0] * 18.0) > -0.15).astype(np.float32)
        colors = base * 0.85
        colors += label[:, None] * np.array([0.92, 0.90, 0.82], dtype=np.float32) * 0.55
    elif style == "wood":
        grain = 0.5 + 0.5 * np.sin(verts[:, 2] * 24.0 + verts[:, 0] * 5.0)
        colors = base.copy()
        colors[:, 0] += grain * 0.10
        colors[:, 1] += grain * 0.06
    elif style == "metal":
        sheen = 0.32 + 0.68 * (verts[:, 2] - verts[:, 2].min()) / max(float(np.ptp(verts[:, 2])), 1e-6)
        micro = 0.5 + 0.5 * np.sin(verts[:, 0] * 48.0 + verts[:, 1] * 17.0)
        colors = base * (0.78 + 0.24 * sheen[:, None])
        colors = colors * (0.95 + 0.05 * micro[:, None])
    elif style == "rubber":
        speckle = 0.5 + 0.5 * np.sin(verts[:, 0] * 33.0) * np.cos(verts[:, 1] * 24.0)
        blotch = 0.5 + 0.5 * np.sin(verts[:, 0] * 7.0 + verts[:, 1] * 6.0 + verts[:, 2] * 2.0)
        colors = base * (0.80 + 0.16 * speckle[:, None])
        colors = colors * (0.95 + 0.05 * blotch[:, None])
    elif style == "plastic":
        soften = 0.55 + 0.45 * (verts[:, 2] - verts[:, 2].min()) / max(float(np.ptp(verts[:, 2])), 1e-6)
        micro = 0.5 + 0.5 * np.sin(verts[:, 0] * 22.0 + verts[:, 1] * 19.0)
        colors = base * (0.86 + 0.08 * soften[:, None])
        colors = colors * (0.96 + 0.04 * micro[:, None])
        colors += np.array([0.02, 0.02, 0.015], dtype=np.float32)
    elif style == "checker":
        pattern = (((verts[:, 0] * 8).astype(int) + (verts[:, 1] * 8).astype(int)) % 2).astype(np.float32)
        colors = base * (0.72 + 0.28 * pattern[:, None])
    else:
        colors = base
    noise = np.random.default_rng(42).uniform(-0.03, 0.03, colors.shape).astype(np.float32)
    mesh.visual.vertex_colors = np.clip(colors + noise, 0.0, 1.0)


def _bowl_curve_geometry(size: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Build a closed, continuous thin shell for a cylindrical bowl section."""
    radius = float(size["radius"])
    span = float(size["span"])
    bottom_z = float(size["bottom_z"])
    thickness = float(size["thickness"])
    half_y = float(size["half_y"])
    segments = max(8, int(round(float(size.get("segments", 96.0)))))
    xs = np.linspace(-span, span, segments + 1, dtype=np.float64)
    vertices: list[list[float]] = []
    for x in xs:
        root = math.sqrt(max(radius * radius - x * x, 1e-10))
        surface_z = bottom_z + radius - root
        normal = np.asarray([-x / radius, 0.0, root / radius], dtype=np.float64)
        inner = np.asarray([x, 0.0, surface_z], dtype=np.float64)
        outer = inner - normal * thickness
        vertices.extend(
            [
                [float(inner[0]), -half_y, float(inner[2])],
                [float(inner[0]), half_y, float(inner[2])],
                [float(outer[0]), -half_y, float(outer[2])],
                [float(outer[0]), half_y, float(outer[2])],
            ]
        )

    faces: list[list[int]] = []
    for index in range(segments):
        current = 4 * index
        following = 4 * (index + 1)
        # Inner and outer curved faces.
        faces.extend(
            [
                [current, following, following + 1],
                [current, following + 1, current + 1],
                [current + 2, following + 3, following + 2],
                [current + 2, current + 3, following + 3],
                # Front and back longitudinal side walls.
                [current, current + 2, following + 2],
                [current, following + 2, following],
                [current + 1, following + 1, following + 3],
                [current + 1, following + 3, current + 3],
            ]
        )
    # Close the two end caps so the static collision mesh is watertight.
    left = 0
    right = 4 * segments
    faces.extend(
        [
            [left, left + 1, left + 3],
            [left, left + 3, left + 2],
            [right, right + 3, right + 1],
            [right, right + 2, right + 3],
        ]
    )
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _make_mesh(obj: ObjectSpec) -> trimesh.Trimesh:
    s = obj.size
    if obj.shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=s["radius"])
    elif obj.shape == "box":
        mesh = trimesh.creation.box(extents=[2 * s["hx"], 2 * s["hy"], 2 * s["hz"]])
        mesh = mesh.subdivide().subdivide()
    elif obj.shape == "rounded_box":
        core = trimesh.creation.box(extents=[2 * s["hx"], 2 * s["hy"], 2 * s["hz"]])
        bumps = []
        r = s["corner_radius"]
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    sp = trimesh.creation.icosphere(subdivisions=2, radius=r)
                    sp.apply_translation([
                        sx * max(s["hx"] - r, 0.0),
                        sy * max(s["hy"] - r, 0.0),
                        sz * max(s["hz"] - r, 0.0),
                    ])
                    bumps.append(sp)
        mesh = trimesh.util.concatenate([core] + bumps)
    elif obj.shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=s["radius"], height=s["height"], sections=40)
    elif obj.shape == "capsule":
        mesh = trimesh.creation.capsule(radius=s["radius"], height=s["height"], count=[16, 24])
    elif obj.shape == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale([s["rx"], s["ry"], s["rz"]])
    elif obj.shape == "puck":
        mesh = trimesh.creation.cylinder(radius=s["radius"], height=s["height"], sections=40)
    elif obj.shape == "cone_frustum":
        sections = 40
        angles = np.linspace(0.0, 2.0 * np.pi, num=sections, endpoint=False)
        bottom = np.stack([s["r_base"] * np.cos(angles), s["r_base"] * np.sin(angles), np.full_like(angles, -0.5 * s["height"])], axis=1)
        top = np.stack([s["r_top"] * np.cos(angles), s["r_top"] * np.sin(angles), np.full_like(angles, 0.5 * s["height"])], axis=1)
        vertices = np.concatenate([bottom, top], axis=0)
        faces = []
        for i in range(sections):
            j = (i + 1) % sections
            faces.append([i, j, sections + j])
            faces.append([i, sections + j, sections + i])
        bottom_center = len(vertices)
        top_center = len(vertices) + 1
        vertices = np.concatenate([vertices, [[0.0, 0.0, -0.5 * s["height"]], [0.0, 0.0, 0.5 * s["height"]]]], axis=0)
        for i in range(sections):
            j = (i + 1) % sections
            faces.append([bottom_center, j, i])
            faces.append([top_center, sections + i, sections + j])
        mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    elif obj.shape == "wedge":
        hx, hy, hz = s["hx"], s["hy"], s["hz"]
        vertices = np.array([
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 5],
            [0, 3, 4],
            [3, 2, 5], [3, 5, 4],
            [0, 4, 3],
        ], dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    elif obj.shape == "wheel_thick":
        outer = trimesh.creation.cylinder(radius=s["radius"], height=s["width"], sections=48)
        band = trimesh.creation.cylinder(radius=s["radius"] * 0.78, height=s["width"] * 0.72, sections=48)
        band.apply_scale([1.0, 1.0, 1.0])
        mesh = trimesh.util.concatenate([outer, band])
    elif obj.shape == "spool":
        core = trimesh.creation.cylinder(radius=s["core_radius"], height=s["width"], sections=40)
        flange_a = trimesh.creation.cylinder(radius=s["flange_radius"], height=s["flange_width"], sections=40)
        flange_b = flange_a.copy()
        flange_a.apply_translation([0.0, 0.0, -0.5 * (s["width"] - s["flange_width"])])
        flange_b.apply_translation([0.0, 0.0, 0.5 * (s["width"] - s["flange_width"])])
        mesh = trimesh.util.concatenate([core, flange_a, flange_b])
    elif obj.shape == "dumbbell":
        bar = trimesh.creation.cylinder(radius=s["bar_radius"], height=s["length"], sections=32)
        left = trimesh.creation.icosphere(subdivisions=2, radius=s["weight_radius"])
        right = left.copy()
        left.apply_translation([0.0, 0.0, -0.5 * s["length"]])
        right.apply_translation([0.0, 0.0, 0.5 * s["length"]])
        mesh = trimesh.util.concatenate([bar, left, right])
    elif obj.shape == "bowl_curve":
        vertices, faces = _bowl_curve_geometry(s)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        raise ValueError(f"unsupported shape: {obj.shape}")

    _apply_procedural_material(mesh, obj)
    return mesh


def _collision_shape(obj: ObjectSpec) -> int:
    s = obj.size
    if obj.shape in {"sphere", "ellipsoid"}:
        radius = s["radius"] if "radius" in s else max(s["rx"], s["ry"], s["rz"])
        return p.createCollisionShape(p.GEOM_SPHERE, radius=float(radius))
    if obj.shape in {"box", "rounded_box", "wedge"}:
        return p.createCollisionShape(p.GEOM_BOX, halfExtents=[float(s["hx"]), float(s["hy"]), float(s["hz"])])
    if obj.shape in {"cylinder", "puck", "wheel_thick", "spool"}:
        radius = s["radius"] if "radius" in s else s.get("flange_radius", s.get("core_radius"))
        height = s["height"] if "height" in s else s.get("width")
        return p.createCollisionShape(p.GEOM_CYLINDER, radius=float(radius), height=float(height))
    if obj.shape == "capsule":
        return p.createCollisionShape(p.GEOM_CAPSULE, radius=float(s["radius"]), height=float(s["height"]))
    if obj.shape == "cone_frustum":
        return p.createCollisionShape(p.GEOM_CYLINDER, radius=float(max(s["r_top"], s["r_base"])), height=float(s["height"]))
    if obj.shape == "dumbbell":
        return p.createCollisionShape(p.GEOM_CAPSULE, radius=float(s["weight_radius"]), height=float(s["length"]))
    if obj.shape == "bowl_curve":
        vertices, faces = _bowl_curve_geometry(s)
        return p.createCollisionShape(
            p.GEOM_MESH,
            vertices=vertices.tolist(),
            indices=faces.reshape(-1).tolist(),
        )
    raise ValueError(f"unsupported collision shape: {obj.shape}")


def _floor_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    verts = mesh.vertices
    colors = np.zeros((len(verts), 3), dtype=np.float32)
    rng = np.random.default_rng(13)
    plank_width = 0.55
    for idx, vert in enumerate(verts):
        plank_id = int((vert[0] + 7.0) / plank_width)
        base = np.array([0.50, 0.37, 0.23], dtype=np.float32)
        base += rng.uniform(-0.04, 0.04, 3).astype(np.float32)
        seam_pos = -7.0 + (plank_id + 1) * plank_width
        if abs(vert[0] - seam_pos) < 0.025:
            base *= 0.6
        colors[idx] = np.clip(base, 0.0, 1.0)
    return colors


def _wall_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    rng = np.random.default_rng(17)
    base = np.tile(np.array([0.52, 0.51, 0.48], dtype=np.float32), (len(mesh.vertices), 1))
    base += rng.uniform(-0.02, 0.02, base.shape).astype(np.float32)
    return np.clip(base, 0.0, 1.0)


class PreviewRenderer:
    def __init__(self) -> None:
        self.scene = pyrender.Scene(bg_color=[0.56, 0.53, 0.49], ambient_light=[0.05, 0.05, 0.05])
        floor = trimesh.creation.box(extents=[14.0, 12.0, 0.04])
        floor.visual.vertex_colors = _floor_colors(floor)
        self.scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False), pose=_tr(0.0, 2.0, -0.04))

        wall = trimesh.creation.box(extents=[14.0, 0.03, 3.6])
        wall.visual.vertex_colors = _wall_colors(wall)
        self.scene.add(pyrender.Mesh.from_trimesh(wall, smooth=False), pose=_tr(0.0, 5.0, 1.8))

        left_fill = pyrender.SpotLight(color=[1.0, 0.96, 0.92], intensity=55.0, innerConeAngle=0.4, outerConeAngle=1.0)
        right_fill = pyrender.SpotLight(color=[0.92, 0.96, 1.0], intensity=36.0, innerConeAngle=0.45, outerConeAngle=1.0)
        self.scene.add(left_fill, pose=_look_at(np.array([-1.7, -1.5, 2.6]), np.array([0.2, 0.4, 0.3]), np.array([0, 0, 1])))
        self.scene.add(right_fill, pose=_look_at(np.array([1.9, -0.8, 2.2]), np.array([-0.1, 0.8, 0.4]), np.array([0, 0, 1])))

        camera = pyrender.PerspectiveCamera(yfov=np.radians(50.0), aspectRatio=IMG_W / IMG_H)
        self.scene.add(camera, pose=_look_at(CAM_EYE, CAM_TARGET, CAM_UP))
        self.renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)
        self.nodes: Dict[str, pyrender.Node] = {}

    def add_object(self, obj: ObjectSpec) -> None:
        mesh = _make_mesh(obj)
        node = self.scene.add(
            pyrender.Mesh.from_trimesh(mesh, smooth=True),
            pose=_tr(obj.position[0], obj.position[1], obj.position[2]),
        )
        self.nodes[obj.name] = node

    def update_pose(self, name: str, pos: List[float], quat: List[float]) -> None:
        self.scene.set_pose(self.nodes[name], pose=_pb_pose(pos, quat))

    def render(self) -> np.ndarray:
        color, _ = self.renderer.render(self.scene, flags=RenderFlags.SHADOWS_SPOT)
        return color

    def cleanup(self) -> None:
        self.renderer.delete()


def make_obj(
    *,
    name: str,
    shape: str,
    color: List[float],
    position: List[float],
    size: Dict[str, float],
    mass: float = 1.0,
    dynamic: bool = True,
    texture_style: str = "solid",
    restitution: float = 0.55,
    friction: float = 0.55,
    orientation_euler_deg: List[float] | None = None,
    linear_velocity: List[float] | None = None,
    angular_velocity: List[float] | None = None,
    role: str = "dynamic",
    texture_asset: str = "",
) -> ObjectSpec:
    return ObjectSpec(
        name=name,
        shape=shape,
        color=color,
        mass=mass,
        position=position,
        size=size,
        dynamic=dynamic,
        restitution=restitution,
        friction=friction,
        orientation_euler_deg=orientation_euler_deg or [0.0, 0.0, 0.0],
        linear_velocity=linear_velocity or [0.0, 0.0, 0.0],
        angular_velocity=angular_velocity or [0.0, 0.0, 0.0],
        role=role,
        texture_style=texture_style,
        texture_asset=texture_asset,
    )


def apply_theme_to_scenarios(scenarios: List[ScenarioSpec], theme: str) -> List[ScenarioSpec]:
    if theme not in THEME_LABELS:
        raise ValueError(f"unsupported theme: {theme}")
    if theme == "industrial":
        return _apply_industrial_theme(scenarios)
    return _apply_daily_theme(scenarios)


def _apply_industrial_theme(scenarios: List[ScenarioSpec]) -> List[ScenarioSpec]:
    styled: List[ScenarioSpec] = []
    for scenario in scenarios:
        objects: List[ObjectSpec] = []
        for obj in scenario.objects:
            new_obj = replace(obj)
            if obj.role == "occluder":
                new_obj.color = [0.78, 0.76, 0.72] if "left" in obj.name or "occ" in obj.name else [0.68, 0.72, 0.78]
                new_obj.texture_style = "painted"
            elif obj.role == "support":
                new_obj.color = [0.92, 0.88, 0.82]
                new_obj.texture_style = "wood_real"
                new_obj.texture_asset = "plywood"
            elif obj.shape == "sphere":
                new_obj.color = [0.88, 0.34, 0.16]
                new_obj.texture_style = "rubber"
            elif obj.shape == "capsule":
                new_obj.color = [0.92, 0.62, 0.20]
                new_obj.texture_style = "stripe"
            elif obj.shape == "box":
                if "target" in obj.name or "mid" in obj.name or "push" in obj.name or "tail" in obj.name:
                    new_obj.color = [0.96, 0.92, 0.88]
                    new_obj.texture_style = "wood_real"
                    new_obj.texture_asset = "wood_planks" if "target" in obj.name else "weathered_planks"
                else:
                    new_obj.color = [0.84, 0.70, 0.34]
                    new_obj.texture_style = "painted"
            elif obj.shape == "cylinder":
                if "rolling" in obj.name or "topple" in obj.name or "tail" in obj.name:
                    new_obj.color = [0.24, 0.60, 0.82]
                    new_obj.texture_style = "label"
                else:
                    new_obj.color = [0.95, 0.90, 0.84]
                    new_obj.texture_style = "wood_real"
                    new_obj.texture_asset = "dark_wood"
            elif obj.shape == "puck":
                new_obj.color = [0.84, 0.46, 0.18]
                new_obj.texture_style = "rubber"
            objects.append(new_obj)
        styled.append(replace(scenario, objects=objects, sim_type="rigid_simple_industrial"))
    return styled


def _apply_daily_theme(scenarios: List[ScenarioSpec]) -> List[ScenarioSpec]:
    styled: List[ScenarioSpec] = []
    for scenario in scenarios:
        objects: List[ObjectSpec] = []
        title = scenario.title
        description = scenario.description
        for obj in scenario.objects:
            new_obj = replace(obj)
            if obj.role == "occluder":
                new_obj.color = [0.93, 0.88, 0.74] if "left" in obj.name or "occ" in obj.name else [0.78, 0.86, 0.93]
                new_obj.texture_style = "painted"
            elif obj.role == "support":
                new_obj.color = [0.96, 0.93, 0.88]
                new_obj.texture_style = "wood_real"
                new_obj.texture_asset = "wood_floor"
            elif obj.shape == "sphere":
                new_obj.color = [0.93, 0.30, 0.25] if "left" in obj.name or "lead" in obj.name or "drop" in obj.name else [0.29, 0.61, 0.90]
                new_obj.texture_style = "plastic"
            elif obj.shape == "capsule":
                new_obj.color = [0.97, 0.73, 0.27]
                new_obj.texture_style = "two_tone"
            elif obj.shape == "box":
                if "target" in obj.name or "mid" in obj.name or "push" in obj.name or "tail" in obj.name:
                    new_obj.color = [0.95, 0.92, 0.86] if "target" in obj.name else [0.90, 0.86, 0.80]
                    new_obj.texture_style = "wood_real"
                    new_obj.texture_asset = "plywood" if "target" in obj.name else "wood_planks"
                else:
                    new_obj.color = [0.96, 0.82, 0.34]
                    new_obj.texture_style = "two_tone"
            elif obj.shape == "cylinder":
                new_obj.color = [0.29, 0.72, 0.56] if "tail" in obj.name or "rolling" in obj.name else [0.98, 0.57, 0.31]
                new_obj.texture_style = "stripe"
            elif obj.shape == "puck":
                new_obj.color = [0.97, 0.50, 0.21]
                new_obj.texture_style = "label"
            objects.append(new_obj)

        daily_title = title
        daily_desc = description
        if scenario.key == "simple_f1_sphere_bounce_roll":
            daily_title = "训练球入镜后弹跳滚动"
            daily_desc = "像儿童训练球一样的彩色球体从画外连续入镜，在重力下弹跳后滚动。"
        elif scenario.key == "simple_f1_capsule_slide_spin":
            daily_title = "胶囊收纳盒滑行并自旋"
            daily_desc = "像日常塑料收纳盒一样的胶囊体滑入画面，在摩擦作用下滑行并转动。"
        elif scenario.key == "simple_f1_capsule_low_spin_long_slide":
            daily_title = "收纳盒低速自旋长距离滑行"
            daily_desc = "胶囊收纳盒以较小角速度和中等线速度入镜，主要观察长距离滑行与缓慢转姿。"
        elif scenario.key == "simple_f1_capsule_fast_spin_glancing":
            daily_title = "收纳盒高速自旋斜向滑入"
            daily_desc = "胶囊收纳盒带更大的横向分量和更高角速度斜向滑入，观察快速姿态耦合变化。"
        elif scenario.key == "simple_f1_capsule_upright_tumble_slide":
            daily_title = "收纳盒大倾角翻滚滑行"
            daily_desc = "胶囊收纳盒以更大的初始倾角入镜，在滑行中发生明显翻滚和轴向姿态切换。"
        elif scenario.key == "simple_f1_capsule_highspeed_lowspin":
            daily_title = "收纳盒高速低自旋掠过"
            daily_desc = "胶囊收纳盒以更高线速度但较低角速度快速掠过画面，观察平移主导的运动形态。"
        elif scenario.key == "simple_f1_capsule_reverse_spin_sweep":
            daily_title = "收纳盒反向自旋横向扫过"
            daily_desc = "胶囊收纳盒带反向角速度与较强横向速度分量横扫画面，观察反向滚滑耦合。"
        elif scenario.key == "simple_f2_puck_hits_box":
            daily_title = "塑料圆盘撞击积木盒"
            daily_desc = "扁平塑料圆盘斜向滑入并撞击彩色积木盒，观察速度传递和偏转。"
        elif scenario.key == "simple_f2_cylinder_hits_cylinder":
            daily_title = "卷筒撞击杯罐"
            daily_desc = "卧放卷筒样物体滚入后撞击立起的杯罐样物体，观察碰撞后的平移与转动耦合。"
        elif scenario.key == "simple_f3_sphere_chain_reaction":
            daily_title = "玩具球触发收纳盒连锁"
            daily_desc = "玩具球先撞第一个收纳盒，再带动第二个收纳盒，测试简单几何下的因果传播。"
        elif scenario.key == "simple_f3_capsule_box_cylinder_chain":
            daily_title = "收纳盒推动纸罐连锁"
            daily_desc = "胶囊收纳盒推动彩色方盒，再由方盒碰到纸罐样圆柱，形成三体连锁传播。"
        elif scenario.key == "simple_f4_ball_behind_pillars":
            daily_title = "玩具球经过路障柱后方"
            daily_desc = "使用两个彩色路障柱做遮挡，玩具球从画外连续入镜并完成遮挡-重现。"
        elif scenario.key == "simple_f4_dual_sphere_cross_occlusion":
            daily_title = "双球在路障柱后交叉"
            daily_desc = "两个彩色玩具球从左右两侧入镜，在路障柱后交叉并重现，检查 identity 保持。"
        elif scenario.key == "simple_f5_sphere_drop_on_platform":
            daily_title = "训练球落到矮台后滚下"
            daily_desc = "彩色训练球落到矮平台后继续滚动并离开支撑面，测试支撑切换。"
        elif scenario.key == "simple_f5_cylinder_topple":
            daily_title = "罐状物在窄底座上失稳倒下"
            daily_desc = "像饮料罐一样的圆柱立在窄底座上并带轻微倾斜，在重力作用下自然失稳倒伏。"

        styled.append(
            replace(
                scenario,
                title=daily_title,
                description=daily_desc,
                objects=objects,
                sim_type="rigid_simple_daily_objects",
            )
        )
    return styled


def build_preview_scenarios() -> List[ScenarioSpec]:
    return [
        ScenarioSpec(
            key="simple_f1_sphere_bounce_roll",
            family="F1 单物体运动",
            title="球体入镜后弹跳滚动",
            description="球体从画外连续入镜，在真实重力下落地后发生弹跳和滚动。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.74,
            seed=1001,
            pre_roll_s=0.26,
            objects=[
                make_obj(
                    name="lead_sphere",
                    shape="sphere",
                    color=[0.84, 0.35, 0.22],
                    position=[-1.85, -0.18, 0.82],
                    size={"radius": 0.16},
                    texture_style="rubber",
                    restitution=0.58,
                    friction=0.62,
                    linear_velocity=[2.95, 0.10, -0.05],
                    angular_velocity=[0.0, 7.0, 0.0],
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f1_capsule_slide_spin",
            family="F1 单物体运动",
            title="胶囊体滑行并自旋",
            description="胶囊体以初速度入镜，在地面摩擦作用下滑行并伴随姿态变化。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.88,
            seed=1002,
            pre_roll_s=0.18,
            objects=[
                make_obj(
                    name="slide_capsule",
                    shape="capsule",
                    color=[0.91, 0.62, 0.24],
                    position=[-1.95, 0.32, 0.17],
                    size={"radius": 0.10, "height": 0.28},
                    texture_style="stripe",
                    restitution=0.10,
                    friction=0.84,
                    orientation_euler_deg=[90.0, 10.0, 0.0],
                    linear_velocity=[3.25, -0.10, 0.0],
                    angular_velocity=[9.0, 0.0, 2.0],
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f1_capsule_low_spin_long_slide",
            family="F1 单物体运动",
            title="胶囊体低速自旋长距离滑行",
            description="胶囊体以较小角速度和中等线速度入镜，主要观察长距离滑行与缓慢转姿。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.84,
            seed=1011,
            pre_roll_s=0.16,
            objects=[
                make_obj(
                    name="slide_capsule",
                    shape="capsule",
                    color=[0.91, 0.62, 0.24],
                    position=[-2.15, 0.24, 0.17],
                    size={"radius": 0.10, "height": 0.28},
                    texture_style="stripe",
                    restitution=0.08,
                    friction=0.76,
                    orientation_euler_deg=[90.0, 4.0, -8.0],
                    linear_velocity=[2.65, -0.04, 0.0],
                    angular_velocity=[4.0, 0.0, 0.8],
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f1_capsule_fast_spin_glancing",
            family="F1 单物体运动",
            title="胶囊体高速自旋斜向滑入",
            description="胶囊体带更大的横向分量和更高角速度斜向滑入，观察快速姿态耦合变化。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.82,
            seed=1012,
            pre_roll_s=0.20,
            objects=[
                make_obj(
                    name="slide_capsule",
                    shape="capsule",
                    color=[0.91, 0.62, 0.24],
                    position=[-2.10, 0.56, 0.17],
                    size={"radius": 0.10, "height": 0.28},
                    texture_style="stripe",
                    restitution=0.10,
                    friction=0.82,
                    orientation_euler_deg=[90.0, 18.0, 12.0],
                    linear_velocity=[3.95, -0.34, 0.0],
                    angular_velocity=[12.0, 0.0, 3.2],
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f1_capsule_upright_tumble_slide",
            family="F1 单物体运动",
            title="胶囊体大倾角翻滚滑行",
            description="胶囊体以更大的初始倾角入镜，在滑行中发生明显翻滚和轴向姿态切换。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.90,
            seed=1013,
            pre_roll_s=0.12,
            objects=[
                make_obj(
                    name="slide_capsule",
                    shape="capsule",
                    color=[0.91, 0.62, 0.24],
                    position=[-1.90, -0.08, 0.17],
                    size={"radius": 0.10, "height": 0.28},
                    texture_style="stripe",
                    restitution=0.09,
                    friction=0.88,
                    orientation_euler_deg=[78.0, 24.0, 28.0],
                    linear_velocity=[3.10, 0.08, 0.0],
                    angular_velocity=[7.5, 1.6, 4.0],
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f1_capsule_highspeed_lowspin",
            family="F1 单物体运动",
            title="胶囊体高速低自旋掠过",
            description="胶囊体以更高线速度但较低角速度快速掠过画面，观察平移主导的运动形态。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.80,
            seed=1014,
            pre_roll_s=0.24,
            objects=[
                make_obj(
                    name="slide_capsule",
                    shape="capsule",
                    color=[0.91, 0.62, 0.24],
                    position=[-2.35, -0.26, 0.17],
                    size={"radius": 0.10, "height": 0.28},
                    texture_style="stripe",
                    restitution=0.08,
                    friction=0.72,
                    orientation_euler_deg=[90.0, -6.0, 5.0],
                    linear_velocity=[4.55, 0.18, 0.0],
                    angular_velocity=[3.4, 0.0, 0.6],
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f1_capsule_reverse_spin_sweep",
            family="F1 单物体运动",
            title="胶囊体反向自旋横向扫过",
            description="胶囊体带反向角速度与较强横向速度分量横扫画面，观察反向滚滑耦合。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.86,
            seed=1015,
            pre_roll_s=0.18,
            objects=[
                make_obj(
                    name="slide_capsule",
                    shape="capsule",
                    color=[0.91, 0.62, 0.24],
                    position=[-2.05, 0.62, 0.17],
                    size={"radius": 0.10, "height": 0.28},
                    texture_style="stripe",
                    restitution=0.10,
                    friction=0.80,
                    orientation_euler_deg=[90.0, -14.0, -18.0],
                    linear_velocity=[3.45, -0.48, 0.0],
                    angular_velocity=[-8.0, 0.0, -2.6],
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f2_puck_hits_box",
            family="F2 双体交互",
            title="圆盘撞击方块",
            description="扁平圆盘斜向滑入并撞击方块，观察速度传递和偏转。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.61,
            seed=2001,
            pre_roll_s=0.12,
            objects=[
                make_obj(
                    name="impact_puck",
                    shape="puck",
                    color=[0.76, 0.44, 0.20],
                    position=[-1.76, 0.42, 0.05],
                    size={"radius": 0.21, "height": 0.06},
                    texture_style="rubber",
                    restitution=0.08,
                    friction=0.55,
                    linear_velocity=[4.40, -0.28, 0.0],
                    angular_velocity=[0.0, 0.0, 7.0],
                ),
                make_obj(
                    name="target_box",
                    shape="box",
                    color=[0.30, 0.63, 0.76],
                    position=[0.30, 0.06, 0.18],
                    size={"hx": 0.18, "hy": 0.18, "hz": 0.18},
                    mass=1.2,
                    texture_style="painted",
                    restitution=0.12,
                    friction=0.72,
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f2_cylinder_hits_cylinder",
            family="F2 双体交互",
            title="卧圆柱撞立圆柱",
            description="卧放圆柱滚入后撞击立柱，观察碰撞后的平移与转动耦合。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.70,
            seed=2002,
            pre_roll_s=0.20,
            objects=[
                make_obj(
                    name="rolling_cylinder",
                    shape="cylinder",
                    color=[0.22, 0.66, 0.72],
                    position=[-2.10, -0.16, 0.12],
                    size={"radius": 0.12, "height": 0.30},
                    texture_style="label",
                    restitution=0.10,
                    friction=0.74,
                    orientation_euler_deg=[90.0, 0.0, 0.0],
                    linear_velocity=[3.85, 0.12, 0.0],
                    angular_velocity=[11.0, 0.0, 0.0],
                ),
                make_obj(
                    name="upright_cylinder",
                    shape="cylinder",
                    color=[0.94, 0.82, 0.42],
                    position=[0.18, 0.02, 0.22],
                    size={"radius": 0.11, "height": 0.44},
                    mass=1.1,
                    texture_style="two_tone",
                    restitution=0.10,
                    friction=0.72,
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f3_sphere_chain_reaction",
            family="F3 多体连锁",
            title="球体触发双方块连锁",
            description="球体先撞第一个方块，再带动第二个方块，测试简单几何下的因果传播。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.67,
            seed=3001,
            pre_roll_s=0.24,
            objects=[
                make_obj(
                    name="lead_ball",
                    shape="sphere",
                    color=[0.84, 0.31, 0.22],
                    position=[-2.10, -0.14, 0.16],
                    size={"radius": 0.16},
                    texture_style="rubber",
                    restitution=0.42,
                    friction=0.48,
                    linear_velocity=[4.05, 0.0, 0.0],
                    angular_velocity=[0.0, 7.0, 0.0],
                ),
                make_obj(
                    name="mid_box",
                    shape="box",
                    color=[0.84, 0.54, 0.28],
                    position=[-0.20, -0.14, 0.17],
                    size={"hx": 0.17, "hy": 0.17, "hz": 0.17},
                    mass=0.95,
                    texture_style="painted",
                    restitution=0.08,
                    friction=0.74,
                ),
                make_obj(
                    name="box_tail",
                    shape="box",
                    color=[0.28, 0.58, 0.76],
                    position=[0.74, -0.14, 0.18],
                    size={"hx": 0.18, "hy": 0.18, "hz": 0.18},
                    mass=0.95,
                    texture_style="two_tone",
                    restitution=0.08,
                    friction=0.76,
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f3_capsule_box_cylinder_chain",
            family="F3 多体连锁",
            title="胶囊推动方块再碰圆柱",
            description="胶囊体推动方块，再由方块碰到圆柱，形成三体链式传播。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.79,
            seed=3002,
            pre_roll_s=0.30,
            objects=[
                make_obj(
                    name="lead_capsule",
                    shape="capsule",
                    color=[0.83, 0.47, 0.26],
                    position=[-2.20, 0.12, 0.13],
                    size={"radius": 0.09, "height": 0.26},
                    texture_style="label",
                    restitution=0.12,
                    friction=0.78,
                    orientation_euler_deg=[90.0, 0.0, 0.0],
                    linear_velocity=[3.75, 0.0, 0.0],
                    angular_velocity=[8.0, 0.0, 0.0],
                ),
                make_obj(
                    name="push_box",
                    shape="box",
                    color=[0.88, 0.72, 0.36],
                    position=[-0.18, 0.12, 0.16],
                    size={"hx": 0.16, "hy": 0.16, "hz": 0.16},
                    mass=0.85,
                    texture_style="painted",
                    restitution=0.08,
                    friction=0.78,
                ),
                make_obj(
                    name="tail_cylinder",
                    shape="cylinder",
                    color=[0.36, 0.70, 0.58],
                    position=[0.78, 0.12, 0.18],
                    size={"radius": 0.10, "height": 0.36},
                    mass=0.90,
                    texture_style="stripe",
                    restitution=0.10,
                    friction=0.70,
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f4_ball_behind_pillars",
            family="F4 遮挡与重现",
            title="球体经过双柱后方",
            description="使用两个静态柱体做遮挡，动态球体从画外连续入镜并完成遮挡-重现。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.68,
            seed=4001,
            pre_roll_s=0.78,
            objects=[
                make_obj(
                    name="moving_ball",
                    shape="sphere",
                    color=[0.86, 0.29, 0.20],
                    position=[-2.90, 0.78, 0.18],
                    size={"radius": 0.18},
                    texture_style="rubber",
                    restitution=0.68,
                    friction=0.42,
                    linear_velocity=[3.70, 0.0, 0.0],
                    angular_velocity=[0.0, 6.0, 0.0],
                ),
                make_obj(
                    name="pillar_left_occ",
                    shape="cylinder",
                    color=[0.76, 0.74, 0.68],
                    position=[-0.18, -0.06, 0.48],
                    size={"radius": 0.16, "height": 0.96},
                    dynamic=False,
                    mass=0.0,
                    role="occluder",
                    texture_style="painted",
                    friction=0.84,
                ),
                make_obj(
                    name="pillar_right_occ",
                    shape="cylinder",
                    color=[0.66, 0.70, 0.76],
                    position=[0.18, -0.06, 0.48],
                    size={"radius": 0.16, "height": 0.96},
                    dynamic=False,
                    mass=0.0,
                    role="occluder",
                    texture_style="painted",
                    friction=0.84,
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f4_dual_sphere_cross_occlusion",
            family="F4 遮挡与重现",
            title="双球在柱体后交叉",
            description="两个球体从左右两侧入镜，在遮挡柱后交叉并重现，检查 identity 保持。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.72,
            seed=4002,
            pre_roll_s=0.66,
            objects=[
                make_obj(
                    name="cross_ball_left",
                    shape="sphere",
                    color=[0.86, 0.33, 0.24],
                    position=[-2.55, 0.72, 0.16],
                    size={"radius": 0.16},
                    texture_style="rubber",
                    restitution=0.56,
                    friction=0.50,
                    linear_velocity=[3.10, 0.0, 0.0],
                    angular_velocity=[0.0, 6.0, 0.0],
                ),
                make_obj(
                    name="cross_ball_right",
                    shape="sphere",
                    color=[0.28, 0.48, 0.80],
                    position=[2.40, 0.92, 0.16],
                    size={"radius": 0.16},
                    texture_style="rubber",
                    restitution=0.56,
                    friction=0.50,
                    linear_velocity=[-2.85, 0.0, 0.0],
                    angular_velocity=[0.0, -6.0, 0.0],
                ),
                make_obj(
                    name="occ_column",
                    shape="cylinder",
                    color=[0.74, 0.72, 0.67],
                    position=[0.0, -0.04, 0.50],
                    size={"radius": 0.18, "height": 1.00},
                    dynamic=False,
                    mass=0.0,
                    role="occluder",
                    texture_style="painted",
                    friction=0.84,
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f5_sphere_drop_on_platform",
            family="F5 支撑与跌落",
            title="球体落到平台后滚下",
            description="球体落到平台后继续滚动并离开支撑面，测试支撑切换。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.84,
            seed=5001,
            pre_roll_s=0.05,
            objects=[
                make_obj(
                    name="drop_ball",
                    shape="sphere",
                    color=[0.84, 0.35, 0.22],
                    position=[-0.30, 0.0, 1.10],
                    size={"radius": 0.15},
                    texture_style="rubber",
                    restitution=0.34,
                    friction=0.74,
                    linear_velocity=[0.55, 0.0, -0.08],
                    angular_velocity=[0.0, 3.0, 0.0],
                ),
                make_obj(
                    name="support_platform",
                    shape="box",
                    color=[0.83, 0.63, 0.30],
                    position=[0.12, 0.0, 0.16],
                    size={"hx": 0.34, "hy": 0.26, "hz": 0.16},
                    dynamic=False,
                    mass=0.0,
                    role="support",
                    texture_style="painted",
                    friction=0.88,
                ),
            ],
        ),
        ScenarioSpec(
            key="simple_f5_cylinder_topple",
            family="F5 支撑与跌落",
            title="圆柱在窄底座上失稳倒下",
            description="圆柱立在窄底座上并带轻微倾斜，开始后在重力作用下自然失稳倒伏。",
            gravity=EARTH_GRAVITY,
            floor_friction=0.78,
            seed=5002,
            pre_roll_s=0.05,
            objects=[
                make_obj(
                    name="pedestal",
                    shape="box",
                    color=[0.82, 0.58, 0.26],
                    position=[0.0, 0.0, 0.11],
                    size={"hx": 0.13, "hy": 0.13, "hz": 0.11},
                    dynamic=False,
                    mass=0.0,
                    role="support",
                    texture_style="two_tone",
                    friction=0.82,
                ),
                make_obj(
                    name="topple_cylinder",
                    shape="cylinder",
                    color=[0.24, 0.61, 0.84],
                    position=[0.12, 0.0, 0.43],
                    size={"radius": 0.12, "height": 0.42},
                    texture_style="stripe",
                    restitution=0.08,
                    friction=0.66,
                    orientation_euler_deg=[0.0, 8.0, 14.0],
                    linear_velocity=[0.03, 0.0, 0.0],
                    angular_velocity=[0.0, 0.40, 0.16],
                ),
            ],
        ),
    ]


def _overlay_text(frame_bgr: np.ndarray, lines: List[str]) -> None:
    for idx, line in enumerate(lines):
        y = 30 + idx * 24
        cv2.putText(frame_bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (244, 242, 237), 1, cv2.LINE_AA)


def run_scenario(renderer: PreviewRenderer, scenario: ScenarioSpec, overlay_text: bool = True) -> dict:
    np.random.seed(scenario.seed)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")
    p.setGravity(0.0, 0.0, -scenario.gravity)
    # Use the same contact solve for pre-roll, QA, and recorded frames.  Two
    # substeps keep fast floor contact from accumulating a visible overlap.
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / SIM_HZ,
        numSolverIterations=PHYSICS_SOLVER_ITERATIONS,
        numSubSteps=PHYSICS_SUB_STEPS,
        contactERP=PHYSICS_CONTACT_ERP,
        erp=PHYSICS_CONTACT_ERP,
    )
    p.changeDynamics(
        plane_id,
        -1,
        lateralFriction=scenario.floor_friction,
        restitution=scenario.floor_restitution,
        activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
    )
    if abs(float(scenario.gravity) - EARTH_GRAVITY) > 1e-6:
        raise ValueError(f"rigid preview requires Earth gravity {EARTH_GRAVITY}, got {scenario.gravity}")

    bodies: List[dict] = []
    for obj in scenario.objects:
        renderer.add_object(obj)
        quat = _quat_from_euler_deg(obj.orientation_euler_deg)
        body_id = p.createMultiBody(
            baseMass=obj.mass if obj.dynamic else 0.0,
            baseCollisionShapeIndex=_collision_shape(obj),
            basePosition=obj.position,
            baseOrientation=quat,
        )
        p.changeDynamics(
            body_id,
            -1,
            restitution=obj.restitution,
            lateralFriction=obj.friction,
            linearDamping=obj.linear_damping,
            angularDamping=obj.angular_damping,
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        )
        p.resetBaseVelocity(body_id, linearVelocity=obj.linear_velocity, angularVelocity=obj.angular_velocity)
        bodies.append({"spec": obj, "body_id": body_id})

    body_ids = {body["spec"].name: int(body["body_id"]) for body in bodies}
    initialization_qa = [
        assert_initialization_contacts(
            body_ids,
            plane_id,
            stage="post_creation",
        )
    ]

    total_steps = int(SIM_DURATION * SIM_HZ)
    pre_roll_steps = int(scenario.pre_roll_s * SIM_HZ)
    max_frames = math.ceil(total_steps / RECORD_EVERY)
    object_count = len(bodies)
    positions = np.zeros((max_frames, object_count, 3), dtype=np.float32)
    quats = np.zeros((max_frames, object_count, 4), dtype=np.float32)
    linvels = np.zeros((max_frames, object_count, 3), dtype=np.float32)
    angvels = np.zeros((max_frames, object_count, 3), dtype=np.float32)
    frames: List[np.ndarray] = []

    for _ in range(pre_roll_steps):
        p.stepSimulation()
    initialization_qa.append(
        assert_initialization_contacts(
            body_ids,
            plane_id,
            stage="post_pre_roll",
        )
    )

    frame_index = 0
    for record_step in range(total_steps):
        if record_step % RECORD_EVERY != 0:
            p.stepSimulation()
            continue
        if frame_index == 0:
            initialization_qa.append(
                assert_initialization_contacts(
                    body_ids,
                    plane_id,
                    stage="video_frame_0",
                )
            )
        visible_time = record_step / SIM_HZ
        for obj_index, body in enumerate(bodies):
            body_id = body["body_id"]
            pos, quat = p.getBasePositionAndOrientation(body_id)
            linvel, angvel = p.getBaseVelocity(body_id)
            positions[frame_index, obj_index] = np.asarray(pos, dtype=np.float32)
            quats[frame_index, obj_index] = np.asarray(quat, dtype=np.float32)
            linvels[frame_index, obj_index] = np.asarray(linvel, dtype=np.float32)
            angvels[frame_index, obj_index] = np.asarray(angvel, dtype=np.float32)
            renderer.update_pose(body["spec"].name, pos, quat)

        if getattr(renderer, "capture_contact_records", False):
            renderer.contact_records.extend(
                collect_frame_contacts(
                    body_ids,
                    plane_id,
                    frame_index=frame_index,
                    time_s=visible_time,
                    dynamic_names=getattr(renderer, "contact_dynamic_names", None),
                )
            )

        frame_rgb = renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if overlay_text:
            _overlay_text(
                frame_bgr,
                [
                    f"{scenario.family} | {scenario.title}",
                    f"{scenario.key} | t={visible_time:0.2f}s | pre-roll={scenario.pre_roll_s:0.2f}s | floor_mu={scenario.floor_friction:0.2f} | objects={object_count}",
                ],
        )
        frames.append(frame_bgr)
        frame_index += 1
        p.stepSimulation()

    for body in bodies:
        p.removeBody(body["body_id"])

    output_mp4 = VIDEO_DIR / f"{scenario.key}.mp4"
    _write_video_h264(output_mp4, frames)

    output_npz = META_DIR / f"{scenario.key}_states.npz"
    np.savez_compressed(
        output_npz,
        positions=positions[:frame_index],
        quats=quats[:frame_index],
        linear_velocities=linvels[:frame_index],
        angular_velocities=angvels[:frame_index],
        frame_times=np.arange(frame_index, dtype=np.float32) / FPS,
        object_names=np.asarray([body["spec"].name for body in bodies]),
        object_roles=np.asarray([body["spec"].role for body in bodies]),
        camera_eye=CAM_EYE.astype(np.float32),
        camera_target=CAM_TARGET.astype(np.float32),
        camera_up=CAM_UP.astype(np.float32),
        frame_width=np.asarray([IMG_W], dtype=np.int32),
        frame_height=np.asarray([IMG_H], dtype=np.int32),
    )

    meta = {
        "key": scenario.key,
        "family": scenario.family,
        "title": scenario.title,
        "description": scenario.description,
        "seed": scenario.seed,
        "gravity": scenario.gravity,
        "sim_type": scenario.sim_type,
        "floor_friction": scenario.floor_friction,
        "floor_restitution": scenario.floor_restitution,
        "spawn_policy": "no mid-scene spawn; pre-roll allowed for off-screen continuity",
        "video": str(output_mp4),
        "states": str(output_npz),
        "fps": FPS,
        "resolution": [IMG_W, IMG_H],
        "duration_s": len(frames) / FPS,
        "pre_roll_s": scenario.pre_roll_s,
        "camera": {
            "eye": CAM_EYE.tolist(),
            "target": CAM_TARGET.tolist(),
            "up": CAM_UP.tolist(),
            "yfov_deg": 50.0,
        },
        "objects": [asdict(obj) for obj in scenario.objects],
        "initialization_qa": {
            "passed": True,
            "penetration_tolerance_m": INITIALIZATION_PENETRATION_TOLERANCE_M,
            "stages": initialization_qa,
        },
    }
    (META_DIR / f"{scenario.key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def generate_html(report_items: List[dict], port: int) -> Path:
    theme_label = THEME_LABELS.get(ACTIVE_THEME, ACTIVE_THEME)
    by_family: Dict[str, List[dict]] = {}
    for item in report_items:
        by_family.setdefault(item["family"], []).append(item)

    sections: List[str] = []
    for family, items in by_family.items():
        cards: List[str] = []
        for item in items:
            object_badges = []
            for obj in item["objects"]:
                role = obj["role"]
                role_text = "遮挡物" if role == "occluder" else ("支撑物" if role == "support" else "动态物体")
                object_badges.append(
                    f"<span class='badge'>{html.escape(obj['name'])}: {html.escape(obj['shape'])} / {role_text}</span>"
                )
            cards.append(
                f"""
                <article class="card">
                  <video controls autoplay loop muted playsinline preload="metadata">
                    <source src="videos/{html.escape(item['key'])}.mp4" type="video/mp4">
                  </video>
                  <div class="content">
                    <div class="eyebrow">{html.escape(item['key'])}</div>
                    <h3>{html.escape(item['title'])}</h3>
                    <p class="desc">{html.escape(item['description'])}</p>
                    <div class="meta-line">
                      <span>type={html.escape(item['sim_type'])}</span>
                      <span>g={item['gravity']}</span>
                      <span>floor_mu={item['floor_friction']}</span>
                      <span>pre-roll={item['pre_roll_s']}s</span>
                      <span>objects={len(item['objects'])}</span>
                    </div>
                    <div class="badge-row">{''.join(object_badges)}</div>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="family">
              <div class="family-head">
                <h2>{html.escape(family)}</h2>
                <p>{len(items)} 个代表 case</p>
              </div>
              <div class="grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rigid Simple Sim Preview v1 - {html.escape(theme_label)}</title>
  <style>
    :root {{
      --bg: #181612;
      --panel: rgba(32, 29, 24, 0.92);
      --line: rgba(255, 255, 255, 0.10);
      --text: #f2ede4;
      --muted: #b8ae9b;
      --accent: #de7f39;
      --accent-2: #73a8c7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(222, 127, 57, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(115, 168, 199, 0.16), transparent 28%),
        linear-gradient(180deg, #171410 0%, #11100d 100%);
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: 0.01em;
    }}
    .lead {{
      margin: 0;
      max-width: 1040px;
      color: var(--muted);
      line-height: 1.75;
      font-size: 15px;
    }}
    .hero {{
      margin-bottom: 28px;
      padding: 22px 24px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      backdrop-filter: blur(8px);
    }}
    .hero-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    .pill {{
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      font-size: 13px;
      color: var(--text);
    }}
    .family {{
      margin-top: 26px;
    }}
    .family-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .family-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .family-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      backdrop-filter: blur(8px);
      box-shadow: 0 18px 30px rgba(0, 0, 0, 0.18);
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
    }}
    .content {{
      padding: 16px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    h3 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .desc {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.65;
      font-size: 14px;
    }}
    .meta-line {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      color: #ddd4c6;
      font-size: 13px;
    }}
    .badge-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.03);
      color: #e8dfd0;
      font-size: 12px;
    }}
    .footer {{
      margin-top: 30px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    a {{ color: var(--accent-2); }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Rigid Simple 仿真预览集 v1 · {html.escape(theme_label)}</h1>
      <p class="lead">
        这一版只保留碰撞形状和视觉形状高度一致的简单刚体：sphere、box、cylinder、capsule、puck。
        所有场景固定使用地球重力 `9.81 m/s²`，并把地面摩擦系数作为显式场景参数纳入变化范围，避免引入大幅 mesh-collision mismatch。
        当前展示主题为“{html.escape(theme_label)}”，物理配置保持一致，仅切换对象的视觉语义、配色和程序化材质。
      </p>
      <div class="hero-row">
        <span class="pill">simulation type = rigid_simple</span>
        <span class="pill">theme = {html.escape(ACTIVE_THEME)}</span>
        <span class="pill">分辨率 {IMG_W}×{IMG_H}</span>
        <span class="pill">FPS {FPS}</span>
        <span class="pill">{len(report_items)} 个 preview case</span>
        <span class="pill">输出目录 {html.escape(str(OUTPUT_ROOT))}</span>
      </div>
    </section>
    {''.join(sections)}
    <div class="footer">
      本地访问地址：
      <a href="http://127.0.0.1:{port}">http://127.0.0.1:{port}</a>
    </div>
  </div>
</body>
</html>
"""
    html_path = OUTPUT_ROOT / "index.html"
    html_path.write_text(page, encoding="utf-8")
    return html_path


def start_server(port: int) -> int:
    log_path = OUTPUT_ROOT / f"http_{port}.log"
    pid_path = OUTPUT_ROOT / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            return pid
        except Exception:
            pid_path.unlink(missing_ok=True)
    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(OUTPUT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a rigid-core simulation preview gallery.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--theme", type=str, default=DEFAULT_THEME, choices=sorted(THEME_LABELS.keys()))
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    global OUTPUT_ROOT, VIDEO_DIR, META_DIR, ACTIVE_THEME
    OUTPUT_ROOT = args.output_root
    VIDEO_DIR = OUTPUT_ROOT / "videos"
    META_DIR = OUTPUT_ROOT / "meta"
    ACTIVE_THEME = args.theme

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for path in [VIDEO_DIR, META_DIR]:
            if path.exists():
                shutil.rmtree(path)
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        META_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = apply_theme_to_scenarios(build_preview_scenarios(), args.theme)
    manifest: List[dict]
    if not args.serve_only:
        p.connect(p.DIRECT)
        try:
            manifest = []
            for scenario in scenarios:
                renderer = PreviewRenderer()
                try:
                    print(f"[generate] {scenario.key} :: {scenario.title}")
                    manifest.append(run_scenario(renderer, scenario))
                finally:
                    renderer.cleanup()
        finally:
            p.disconnect()
        (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))

    html_path = generate_html(manifest, args.port)
    pid = start_server(args.port)
    print(f"gallery: {html_path}")
    print(f"server: http://127.0.0.1:{args.port}")
    print(f"pid: {pid}")


if __name__ == "__main__":
    main()
