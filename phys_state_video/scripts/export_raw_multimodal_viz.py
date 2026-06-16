#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PB_NEAR = 0.05
PB_FAR = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multimodal visualizations for raw simulation cases.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500"),
        help="Raw dataset root containing train/val/test family folders.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/_viz/0613pybullet_multimodal"),
    )
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--panel-width", type=int, default=None, help="Optional resize width for exports; default uses original video width.")
    parser.add_argument("--panel-height", type=int, default=None, help="Optional resize height for exports; default uses original video height.")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


@dataclass
class CaseData:
    sample_dir: Path
    meta: dict
    states: dict[str, np.ndarray]
    frames: np.ndarray
    frame_width: int
    frame_height: int


def clean_dir(path: Path) -> None:
    if path.exists():
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            if child.is_dir() and not child.is_symlink():
                clean_dir(child)
                child.rmdir()
            else:
                child.unlink()


def read_video_frames(video_path: Path, width: int | None = None, height: int | None = None) -> tuple[np.ndarray, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = int(width or source_width)
    out_height = int(height or source_height)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if out_width != source_width or out_height != source_height:
                frame_bgr = cv2.resize(frame_bgr, (out_width, out_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.astype(np.float32) / 255.0)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return np.stack(frames, axis=0), out_width, out_height


def load_case(sample_dir: Path, panel_width: int, panel_height: int) -> CaseData:
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    states_npz = np.load(sample_dir / "states.npz")
    states = {key: states_npz[key] for key in states_npz.files}
    resize_width = panel_width if panel_width and panel_width > 0 else None
    resize_height = panel_height if panel_height and panel_height > 0 else None
    frames, frame_width, frame_height = read_video_frames(sample_dir / "video.mp4", resize_width, resize_height)
    return CaseData(sample_dir=sample_dir, meta=meta, states=states, frames=frames, frame_width=frame_width, frame_height=frame_height)


def family_priority(family_dir: Path) -> tuple[int, str]:
    name = family_dir.name
    priority = {
        "F1_single_object": 0,
        "F2_two_object": 1,
        "F3_chain_reaction": 2,
        "F4_occlusion": 3,
        "F5_drop_support": 4,
    }.get(name, 99)
    return priority, name


def choose_cases(split_root: Path, max_cases: int) -> list[Path]:
    selected: list[Path] = []
    families = sorted((path for path in split_root.iterdir() if path.is_dir()), key=family_priority)
    for family_dir in families:
        sample_dirs = sorted((path for path in family_dir.iterdir() if path.is_dir()), key=lambda p: p.name)
        if not sample_dirs:
            continue
        if family_dir.name == "F5_drop_support":
            preferred = next((p for p in sample_dirs if p.name == "sample_000273"), None)
            selected.append(preferred or sample_dirs[0])
        else:
            selected.append(sample_dirs[0])
        if len(selected) >= max_cases:
            return selected[:max_cases]

    if len(selected) < max_cases:
        for family_dir in families:
            for sample_dir in sorted((path for path in family_dir.iterdir() if path.is_dir()), key=lambda p: p.name):
                if sample_dir in selected:
                    continue
                selected.append(sample_dir)
                if len(selected) >= max_cases:
                    return selected[:max_cases]
    return selected[:max_cases]


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw.astype(np.float64)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def make_camera(meta: dict, width: int, height: int) -> dict[str, np.ndarray | float]:
    camera = meta["camera"]
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    yfov = math.radians(float(camera.get("yfov_deg", 50.0)))
    aspect = float(width) / float(height)
    fx = 0.5 * width / (math.tan(yfov * 0.5) * aspect)
    fy = 0.5 * height / math.tan(yfov * 0.5)
    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": true_up,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5,
        "cy": height * 0.5,
    }


def project_point(point_world: np.ndarray, camera: dict[str, np.ndarray | float]) -> tuple[np.ndarray | None, float]:
    delta = point_world.astype(np.float64) - camera["eye"]
    x_cam = float(delta @ camera["right"])
    y_cam = float(delta @ camera["up"])
    z_cam = float(delta @ camera["forward"])
    if z_cam <= 1e-6:
        return None, z_cam
    u = float(camera["fx"]) * (x_cam / z_cam) + float(camera["cx"])
    v = float(camera["cy"]) - float(camera["fy"]) * (y_cam / z_cam)
    return np.asarray([u, v], dtype=np.float32), z_cam


def compute_pybullet_view_projection(meta: dict, width: int, height: int) -> tuple[list[float], list[float]]:
    camera = meta["camera"]
    eye = camera["eye"]
    target = camera["target"]
    up = camera["up"]
    yfov_deg = float(camera.get("yfov_deg", 50.0))
    aspect = float(width) / float(height)
    view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=up)
    projection = p.computeProjectionMatrixFOV(fov=yfov_deg, aspect=aspect, nearVal=PB_NEAR, farVal=PB_FAR)
    return view, projection


def zbuffer_to_metric_depth(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = far * near / np.clip(far - (far - near) * depth_buffer, 1e-8, None)
    return depth.astype(np.float32)


def rotation_matrix_to_quaternion(rot: np.ndarray) -> list[float]:
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def make_backdrop_pose(meta: dict, distance: float = 7.0) -> tuple[list[float], list[float]]:
    camera = meta["camera"]
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera["target"], dtype=np.float64)
    up_hint = np.asarray(camera["up"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-8

    # Box local axes: x=width, y=thickness(normal), z=height.
    rot = np.eye(3, dtype=np.float64)
    rot[:, 0] = right
    rot[:, 1] = -forward
    rot[:, 2] = up
    center = target + forward * distance
    quat = rotation_matrix_to_quaternion(rot)
    return center.astype(np.float32).tolist(), quat


def create_collision_shape(obj: dict, client_id: int) -> int:
    size = obj["size"]
    shape = obj["shape"]
    if shape == "sphere":
        return p.createCollisionShape(p.GEOM_SPHERE, radius=float(size["radius"]), physicsClientId=client_id)
    if shape == "box":
        return p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[float(size["hx"]), float(size["hy"]), float(size["hz"])],
            physicsClientId=client_id,
        )
    if shape in {"cylinder", "puck"}:
        return p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=float(size["radius"]),
            height=float(size["height"]),
            physicsClientId=client_id,
        )
    if shape == "capsule":
        return p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=float(size["radius"]),
            height=float(size["height"]),
            physicsClientId=client_id,
        )
    raise ValueError(f"unsupported pybullet replay shape: {shape}")


def create_visual_shape(obj: dict, client_id: int) -> int:
    size = obj["size"]
    rgba = list(np.asarray(obj.get("color", [0.7, 0.7, 0.7]), dtype=np.float32)) + [1.0]
    shape = obj["shape"]
    if shape == "sphere":
        return p.createVisualShape(p.GEOM_SPHERE, radius=float(size["radius"]), rgbaColor=rgba, physicsClientId=client_id)
    if shape == "box":
        return p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[float(size["hx"]), float(size["hy"]), float(size["hz"])],
            rgbaColor=rgba,
            physicsClientId=client_id,
        )
    if shape in {"cylinder", "puck"}:
        return p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=float(size["radius"]),
            length=float(size["height"]),
            rgbaColor=rgba,
            physicsClientId=client_id,
        )
    if shape == "capsule":
        return p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=float(size["radius"]),
            length=float(size["height"]),
            rgbaColor=rgba,
            physicsClientId=client_id,
        )
    raise ValueError(f"unsupported pybullet replay visual shape: {shape}")


def render_pybullet_groundtruth(case: CaseData, width: int, height: int, out_dir: Path) -> dict[str, np.ndarray | float]:
    client_id = p.connect(p.DIRECT)
    body_ids: list[int] = []
    object_body_ids: list[int] = []
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.resetSimulation(physicsClientId=client_id)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
        body_ids.append(plane_id)
        p.changeDynamics(plane_id, -1, lateralFriction=float(case.meta.get("floor_friction", 0.7)), restitution=0.02, physicsClientId=client_id)

        backdrop_half_extents = [10.0, 0.03, 6.0]
        backdrop_pos, backdrop_quat = make_backdrop_pose(case.meta, distance=7.5)
        backdrop_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=backdrop_half_extents,
            physicsClientId=client_id,
        )
        backdrop_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=backdrop_half_extents,
            rgbaColor=[0.78, 0.77, 0.74, 1.0],
            physicsClientId=client_id,
        )
        backdrop_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=backdrop_collision,
            baseVisualShapeIndex=backdrop_visual,
            basePosition=backdrop_pos,
            baseOrientation=backdrop_quat,
            physicsClientId=client_id,
        )
        body_ids.append(backdrop_id)

        for obj in case.meta["objects"]:
            collision = create_collision_shape(obj, client_id)
            visual = create_visual_shape(obj, client_id)
            quat = p.getQuaternionFromEuler([math.radians(v) for v in obj.get("orientation_euler_deg", [0.0, 0.0, 0.0])])
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=obj["position"],
                baseOrientation=quat,
                physicsClientId=client_id,
            )
            body_ids.append(body_id)
            object_body_ids.append(body_id)

        view, projection = compute_pybullet_view_projection(case.meta, width, height)
        positions = case.states["positions"][: case.frames.shape[0]]
        quats = case.states["quats"][: case.frames.shape[0]]
        num_frames = positions.shape[0]

        depth_meters = np.zeros((num_frames, height, width), dtype=np.float32)
        depth_zbuffer = np.zeros((num_frames, height, width), dtype=np.float32)
        segmentation_raw = np.zeros((num_frames, height, width), dtype=np.int32)
        segmentation_object_idx = np.full((num_frames, height, width), -1, dtype=np.int16)
        valid_mask = np.zeros((num_frames, height, width), dtype=bool)

        for frame_idx in range(num_frames):
            for obj_idx, body_id in enumerate(object_body_ids):
                p.resetBasePositionAndOrientation(
                    body_id,
                    positions[frame_idx, obj_idx].tolist(),
                    quats[frame_idx, obj_idx].tolist(),
                    physicsClientId=client_id,
                )

            camera_out = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view,
                projectionMatrix=projection,
                renderer=p.ER_TINY_RENDERER,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                physicsClientId=client_id,
            )
            depth_buffer = np.asarray(camera_out[3], dtype=np.float32).reshape(height, width)
            segmentation = np.asarray(camera_out[4], dtype=np.int32).reshape(height, width)
            metric_depth = zbuffer_to_metric_depth(depth_buffer, PB_NEAR, PB_FAR)

            depth_meters[frame_idx] = metric_depth
            depth_zbuffer[frame_idx] = depth_buffer
            segmentation_raw[frame_idx] = segmentation
            valid_mask[frame_idx] = depth_buffer < (1.0 - 1e-6)
            for obj_idx, body_id in enumerate(object_body_ids):
                segmentation_object_idx[frame_idx][segmentation == body_id] = obj_idx
    finally:
        if p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)

    valid_depths = depth_meters[valid_mask]
    depth_near = float(np.min(valid_depths)) if valid_depths.size else 0.0
    depth_far = float(np.max(valid_depths)) if valid_depths.size else 1.0
    if abs(depth_far - depth_near) < 1e-6:
        depth_far = depth_near + 1.0
    depth_vis_near, depth_vis_far = robust_depth_range(depth_meters, valid_mask)

    np.savez_compressed(
        out_dir / "pybullet_depth_gt.npz",
        depth_meters=depth_meters,
        depth_zbuffer=depth_zbuffer,
        segmentation_raw=segmentation_raw,
        segmentation_object_idx=segmentation_object_idx,
        valid_mask=valid_mask.astype(np.uint8),
        near=np.asarray([depth_near], dtype=np.float32),
        far=np.asarray([depth_far], dtype=np.float32),
        vis_near=np.asarray([depth_vis_near], dtype=np.float32),
        vis_far=np.asarray([depth_vis_far], dtype=np.float32),
        camera_near=np.asarray([PB_NEAR], dtype=np.float32),
        camera_far=np.asarray([PB_FAR], dtype=np.float32),
    )
    return {
        "depth_meters": depth_meters,
        "depth_zbuffer": depth_zbuffer,
        "segmentation_object_idx": segmentation_object_idx,
        "valid_mask": valid_mask,
        "depth_near": depth_near,
        "depth_far": depth_far,
        "depth_vis_near": depth_vis_near,
        "depth_vis_far": depth_vis_far,
    }


def object_local_corners(obj: dict) -> np.ndarray:
    shape = obj["shape"]
    size = obj["size"]
    if shape == "sphere":
        ext = np.asarray([size["radius"], size["radius"], size["radius"]], dtype=np.float64)
    elif shape == "box":
        ext = np.asarray([size["hx"], size["hy"], size["hz"]], dtype=np.float64)
    elif shape == "cylinder":
        ext = np.asarray([size["radius"], size["radius"], 0.5 * size["height"]], dtype=np.float64)
    elif shape == "capsule":
        ext = np.asarray([size["radius"], size["radius"], 0.5 * size["height"] + size["radius"]], dtype=np.float64)
    elif shape == "puck":
        ext = np.asarray([size["radius"], size["radius"], 0.5 * size["height"]], dtype=np.float64)
    else:
        raise ValueError(f"unsupported object shape: {shape}")
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return signs * ext[None, :]


def polygon_mask(
    obj: dict,
    position: np.ndarray,
    quat: np.ndarray,
    camera: dict[str, np.ndarray | float],
    width: int,
    height: int,
) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    center_px, depth = project_point(position, camera)
    if center_px is None:
        return None, depth, None

    shape = obj["shape"]
    if shape == "sphere":
        radius = float(obj["size"]["radius"])
        right_px, _ = project_point(position + camera["right"] * radius, camera)
        up_px, _ = project_point(position + camera["up"] * radius, camera)
        if right_px is None or up_px is None:
            return None, depth, center_px
        rx = max(int(round(abs(right_px[0] - center_px[0]))), 2)
        ry = max(int(round(abs(up_px[1] - center_px[1]))), 2)
        return None, depth, center_px, np.asarray([rx, ry], dtype=np.int32)

    corners_local = object_local_corners(obj)
    rot = quat_to_matrix(quat)
    corners_world = corners_local @ rot.T + position[None, :]
    pixels: list[np.ndarray] = []
    for corner in corners_world:
        px, _ = project_point(corner, camera)
        if px is not None:
            pixels.append(px)
    if len(pixels) < 3:
        return None, depth, center_px
    pts = np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    return hull, depth, center_px


def draw_text_box(canvas: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 18)) -> None:
    x, y = origin
    for idx, line in enumerate(lines):
        yy = y + idx * 18
        cv2.putText(canvas, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (18, 18, 18), 1, cv2.LINE_AA)


def depth_to_grayscale(gray01: np.ndarray) -> np.ndarray:
    gray_uint8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(gray_uint8[..., None], 3, axis=2)


def robust_depth_range(depth_meters: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    valid_depths = depth_meters[valid_mask]
    if valid_depths.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(valid_depths, 2.0))
    hi = float(np.percentile(valid_depths, 98.0))
    abs_lo = float(np.min(valid_depths))
    abs_hi = float(np.max(valid_depths))
    lo = max(lo, abs_lo)
    hi = min(hi, abs_hi)
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    return lo, hi


def flow_hsv_color(dx: float, dy: float, ref_mag: float) -> tuple[int, int, int]:
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-8:
        return 255, 255, 255
    hue = (math.atan2(-dy, dx) + math.pi) / (2.0 * math.pi)
    sat = max(0.15, min(1.0, mag / max(ref_mag, 1e-6)))
    val = 1.0
    hsv = np.asarray([[[int(hue * 179.0), int(sat * 255.0), int(val * 255.0)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def blank_panel(width: int, height: int, color: tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:, :] = np.asarray(color, dtype=np.uint8)[None, None, :]
    return panel


def save_animated_gif(path: Path, frames_rgb: list[np.ndarray], fps: int) -> None:
    if not frames_rgb:
        raise ValueError("cannot save empty animation")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame.astype(np.uint8), mode="RGB") for frame in frames_rgb]
    duration_ms = max(int(round(1000.0 / max(fps, 1))), 1)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def render_case(case: CaseData, out_dir: Path, panel_width: int | None, panel_height: int | None, fps: int) -> dict:
    objects = case.meta["objects"]
    object_lookup = {obj["name"]: obj for obj in objects}
    object_names = [str(name) for name in case.states["object_names"]]
    ordered_objects = [object_lookup[name] for name in object_names]
    num_objects = len(ordered_objects)
    num_frames = min(
        case.frames.shape[0],
        int(case.states["positions"].shape[0]),
        int(case.states["quats"].shape[0]),
        int(case.states["linear_velocities"].shape[0]),
    )
    if num_frames <= 0:
        raise RuntimeError(f"no aligned frames for {case.sample_dir}")

    positions = case.states["positions"][:num_frames]
    quats = case.states["quats"][:num_frames]
    linear_velocities = case.states["linear_velocities"][:num_frames]
    angular_velocities = case.states["angular_velocities"][:num_frames]
    masses = np.asarray([float(obj.get("mass", 1.0)) for obj in ordered_objects], dtype=np.float32)

    panel_width = panel_width if panel_width and panel_width > 0 else case.frame_width
    panel_height = panel_height if panel_height and panel_height > 0 else case.frame_height
    camera = make_camera(case.meta, panel_width, panel_height)
    pybullet_gt = render_pybullet_groundtruth(case, panel_width, panel_height, out_dir)
    all_depths: list[float] = []
    all_flow_mags: list[float] = []
    all_momentum_mags: list[float] = []
    frame_cache: list[dict] = []

    for frame_idx in range(num_frames):
        per_object: list[dict] = []
        for obj_idx, obj in enumerate(ordered_objects):
            mask_repr = polygon_mask(
                obj=obj,
                position=positions[frame_idx, obj_idx],
                quat=quats[frame_idx, obj_idx],
                camera=camera,
                width=panel_width,
                height=panel_height,
            )
            if len(mask_repr) == 4:
                hull, depth, center_px, ellipse_axes = mask_repr
            else:
                hull, depth, center_px = mask_repr  # type: ignore[misc]
                ellipse_axes = None

            next_idx = min(frame_idx + 1, num_frames - 1)
            next_center_px, _ = project_point(positions[next_idx, obj_idx], camera)
            if next_center_px is None:
                flow_vec = np.zeros((2,), dtype=np.float32)
            else:
                flow_vec = next_center_px - center_px
            momentum_mag = float(masses[obj_idx] * np.linalg.norm(linear_velocities[frame_idx, obj_idx]))
            flow_mag = float(np.linalg.norm(flow_vec))
            if math.isfinite(depth):
                all_depths.append(depth)
            all_flow_mags.append(flow_mag)
            all_momentum_mags.append(momentum_mag)
            per_object.append(
                {
                    "obj_idx": obj_idx,
                    "name": obj["name"],
                    "shape": obj["shape"],
                    "role": obj["role"],
                    "center_px": center_px,
                    "next_center_px": next_center_px,
                    "depth": depth,
                    "flow_vec": flow_vec,
                    "momentum_mag": momentum_mag,
                    "mass": masses[obj_idx],
                    "linear_velocity": linear_velocities[frame_idx, obj_idx],
                    "angular_velocity": angular_velocities[frame_idx, obj_idx],
                    "mask_repr": (hull, ellipse_axes),
                }
            )
        frame_cache.append({"objects": per_object})

    finite_depths = np.asarray([value for value in all_depths if math.isfinite(value)], dtype=np.float32)
    near_depth = float(np.min(finite_depths)) if finite_depths.size else 0.0
    far_depth = float(np.max(finite_depths)) if finite_depths.size else 1.0
    if abs(far_depth - near_depth) < 1e-6:
        far_depth = near_depth + 1.0
    max_flow_mag = max(float(np.max(np.asarray(all_flow_mags, dtype=np.float32))), 1.0)
    max_momentum_mag = max(float(np.max(np.asarray(all_momentum_mags, dtype=np.float32))), 1.0)

    palette = np.asarray(
        [
            [239, 83, 80],
            [129, 199, 132],
            [79, 195, 247],
            [255, 202, 40],
            [171, 71, 188],
            [141, 110, 99],
        ],
        dtype=np.uint8,
    )

    modality_dirs = {
        "rgb": out_dir / "rgb",
        "depth": out_dir / "depth",
        "mask": out_dir / "mask",
        "flow": out_dir / "flow",
        "momentum": out_dir / "momentum",
        "montage": out_dir / "montage",
    }
    for path in modality_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    def make_writer(path: Path) -> imageio.core.format.Writer:
        return imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_params=["-movflags", "+faststart"],
        )

    writers = {
        name: make_writer(path / f"{name}.mp4")
        for name, path in modality_dirs.items()
        if name != "montage"
    }
    writers["montage"] = make_writer(modality_dirs["montage"] / "montage.mp4")
    montage_frames_rgb: list[np.ndarray] = []

    try:
        for frame_idx in range(num_frames):
            rgb = np.clip(case.frames[frame_idx] * 255.0, 0, 255).astype(np.uint8)
            depth_map = np.asarray(pybullet_gt["depth_meters"][frame_idx], dtype=np.float32)
            valid_depth_mask = np.asarray(pybullet_gt["valid_mask"][frame_idx], dtype=bool)
            mask_idx = np.asarray(pybullet_gt["segmentation_object_idx"][frame_idx], dtype=np.int32)
            flow_panel = blank_panel(panel_width, panel_height, (255, 255, 255))
            momentum_panel = blank_panel(panel_width, panel_height, (15, 16, 18))

            # Use PyBullet-rendered per-pixel segmentation as the support for dense modality overlays.
            for obj in sorted(frame_cache[frame_idx]["objects"], key=lambda item: float(item["depth"]), reverse=True):
                obj_idx = int(obj["obj_idx"])
                update = mask_idx == obj_idx
                if not np.any(update):
                    continue
                flow_vec = obj["flow_vec"]
                flow_color_rgb = flow_hsv_color(float(flow_vec[0]), float(flow_vec[1]), max_flow_mag)
                flow_panel[update] = np.asarray(flow_color_rgb, dtype=np.uint8)

                momentum_mag = float(obj["momentum_mag"])
                momentum_rgb = flow_hsv_color(float(flow_vec[0]), float(flow_vec[1]), max_momentum_mag)
                momentum_panel[update] = np.asarray(momentum_rgb, dtype=np.uint8)

            # Build depth and mask panels.
            gt_vis_near = float(pybullet_gt["depth_vis_near"])
            gt_vis_far = float(pybullet_gt["depth_vis_far"])
            depth_clamped = np.where(valid_depth_mask, np.minimum(depth_map, gt_vis_far), gt_vis_far)
            depth_norm = (depth_clamped - gt_vis_near) / max(gt_vis_far - gt_vis_near, 1e-6)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            depth_vis = depth_to_grayscale(1.0 - depth_norm)
            depth_panel = depth_vis.copy()

            mask_panel = blank_panel(panel_width, panel_height, (8, 8, 8))
            for obj_idx in range(num_objects):
                color = palette[obj_idx % len(palette)]
                mask_panel[mask_idx == obj_idx] = color
            rgb_panel = rgb.copy()

            montage_top = np.concatenate([rgb_panel, depth_panel], axis=1)
            montage_bottom = np.concatenate([mask_panel, flow_panel], axis=1)
            montage = np.concatenate([montage_top, montage_bottom], axis=0)
            montage_frames_rgb.append(montage.copy())

            writers["rgb"].append_data(rgb_panel)
            writers["depth"].append_data(depth_panel)
            writers["mask"].append_data(mask_panel)
            writers["flow"].append_data(flow_panel)
            writers["momentum"].append_data(momentum_panel)
            writers["montage"].append_data(montage)

    finally:
        for writer in writers.values():
            writer.close()
    save_animated_gif(modality_dirs["montage"] / "montage.gif", montage_frames_rgb, fps)

    summary = {
        "sample_dir": str(case.sample_dir),
        "sample_id": case.sample_dir.name,
        "title": case.meta.get("title"),
        "description": case.meta.get("description"),
        "family_slug": case.meta.get("family_slug"),
        "split": case.meta.get("split"),
        "num_frames": num_frames,
        "frame_size": [case.frame_width, case.frame_height],
        "panel_size": [panel_width, panel_height],
        "depth_source": "pybullet_zbuffer_groundtruth",
        "files": {name: f"{name}/{name}.mp4" for name in writers.keys()},
        "preview": "montage/montage.gif",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def render_html(report: dict) -> str:
    cards = []
    for case in report["cases"]:
        links = "".join(
            f'<a href="{html.escape(case["files"][key])}">{html.escape(key)}.mp4</a>'
            for key in ["rgb", "depth", "mask", "flow", "momentum"]
        )
        cards.append(
            f"""
            <article class="card">
              <div class="card-head">
                <div>
                  <div class="eyebrow">{html.escape(case["family_slug"])}</div>
                  <h2>{html.escape(case["sample_id"])}</h2>
                </div>
                <div class="chip">{html.escape(case["split"])}</div>
              </div>
              <p class="title">{html.escape(case["title"] or case["sample_id"])}</p>
              <p class="desc">{html.escape(case["description"] or "")}</p>
              <img class="preview" src="{html.escape(case["preview"])}" alt="{html.escape(case["sample_id"])} preview">
              <div class="links">
                <a href="{html.escape(case["preview"])}">montage.gif</a>
                <a href="{html.escape(case["files"]["montage"])}">montage.mp4</a>
                {links}
                <a href="{html.escape(case["sample_rel_meta"])}">meta.json</a>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>0613pybullet multimodal viz</title>
  <style>
    :root {{
      --bg0:#0f1720;
      --bg1:#182430;
      --panel:rgba(18,24,33,0.92);
      --line:#2f3e4f;
      --ink:#eef3f7;
      --muted:#9eb0bf;
      --accent:#5ed0c2;
      --accent2:#f3a85d;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"IBM Plex Sans","Noto Sans SC","Source Han Sans SC",sans-serif;
      background:
        radial-gradient(circle at top left, rgba(94,208,194,0.14), transparent 26%),
        radial-gradient(circle at top right, rgba(243,168,93,0.12), transparent 22%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
    }}
    .page {{ max-width:1700px; margin:0 auto; padding:24px; }}
    .hero,.card {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:22px;
      box-shadow:0 16px 36px rgba(0,0,0,0.22);
    }}
    .hero {{ padding:20px; margin-bottom:18px; }}
    .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.08em; font-size:12px; }}
    .intro {{ color:var(--muted); line-height:1.65; max-width:980px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .card {{ padding:16px; }}
    .card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
    .chip {{ padding:8px 12px; border-radius:999px; background:rgba(255,255,255,0.05); border:1px solid var(--line); color:var(--muted); }}
    .title {{ margin:12px 0 8px; font-weight:700; }}
    .desc {{ color:var(--muted); line-height:1.6; min-height:42px; }}
    .preview {{ width:100%; display:block; margin-top:12px; border-radius:14px; background:#000; }}
    .links {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:10px; }}
    .links a {{ color:var(--accent); text-decoration:none; font-weight:600; }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">0613pybullet</div>
      <h1>Raw multimodal simulation visualizations</h1>
      <p class="intro">
        这些 case 都是从 `video.mp4 + meta.json + states.npz` 重建出来的可视化：
        RGB、深度、实例 mask、flow 和动量。这里的 depth / mask / flow 不是原始导出文件，
        而是根据仿真状态和相机参数做的可视化重建，适合快速审查几类典型场景。
      </p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"missing split root: {split_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_dir(args.output_dir)

    selected = choose_cases(split_root, args.max_cases)
    if not selected:
        raise RuntimeError(f"no sample cases found under {split_root}")

    case_reports: list[dict] = []
    for sample_dir in selected:
        case = load_case(sample_dir, args.panel_width, args.panel_height)
        case.meta.setdefault("split", args.split)
        case.meta.setdefault("family_slug", sample_dir.parent.name)
        out_dir = args.output_dir / sample_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = render_case(case, out_dir, args.panel_width, args.panel_height, args.fps)
        summary_files = {
            key: os.path.relpath(out_dir / rel_path, args.output_dir)
            for key, rel_path in summary["files"].items()
        }
        summary_preview = os.path.relpath(out_dir / summary["preview"], args.output_dir)
        case_reports.append(
            {
                **summary,
                "files": summary_files,
                "preview": summary_preview,
                "sample_rel_meta": os.path.relpath(sample_dir / "meta.json", args.output_dir),
            }
        )

    report = {
        "data_root": str(args.data_root),
        "split": args.split,
        "output_dir": str(args.output_dir),
        "cases": case_reports,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"index": str(args.output_dir / "index.html"), **report}, ensure_ascii=False, indent=2))

    if args.serve:
        print(
            f"[serve] cd {args.output_dir} && python3 -m http.server {args.port} --bind 127.0.0.1",
            flush=True,
        )
        subprocess.run(["python3", "-m", "http.server", str(args.port), "--bind", "127.0.0.1"], cwd=str(args.output_dir), check=True)


if __name__ == "__main__":
    main()
