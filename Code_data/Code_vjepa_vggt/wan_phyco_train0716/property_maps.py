from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


CHANNEL_NAMES = (
    "restitution",
    "friction",
    "rigid_valid",
    "neo_hookean_lambda",
    "neo_hookean_mu",
    "deformation_valid",
    "motion_strength",
    "move_direction_x",
    "move_direction_y",
)
BRANCH_SLICES = ((0, 3), (3, 6), (6, 9))


@dataclass(frozen=True)
class PropertyMapResult:
    maps: torch.Tensor
    branch_valid: torch.Tensor
    diagnostics: dict[str, Any]


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-8 else np.zeros_like(vector)


def _draw_circle(array: np.ndarray, center: tuple[float, float], radius: float, values: dict[int, float]) -> None:
    x, y = int(round(center[0])), int(round(center[1]))
    r = max(2, int(round(radius)))
    mask = np.zeros(array.shape[-2:], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 1, thickness=-1)
    selector = mask > 0
    for channel, value in values.items():
        array[channel, selector] = float(value)


def _camera_projector(metadata: dict[str, Any], height: int, width: int):
    camera = metadata.get("camera", {})
    eye = np.asarray(camera.get("eye", [0.0, -3.0, 1.4]), dtype=np.float32)
    target = np.asarray(camera.get("target", [0.0, 0.0, 0.4]), dtype=np.float32)
    up_hint = np.asarray(camera.get("up", [0.0, 0.0, 1.0]), dtype=np.float32)
    forward = _normalized(target - eye)
    right = _normalized(np.cross(forward, up_hint))
    up = _normalized(np.cross(right, forward))
    tan_half_y = math.tan(math.radians(float(camera.get("yfov_deg", 50.0))) / 2.0)
    aspect = float(width) / max(float(height), 1.0)

    def project(position: Any) -> tuple[float, float, float] | None:
        relative = np.asarray(position, dtype=np.float32) - eye
        depth = float(np.dot(relative, forward))
        if depth <= 1.0e-5:
            return None
        ndc_x = float(np.dot(relative, right)) / (depth * tan_half_y * aspect)
        ndc_y = float(np.dot(relative, up)) / (depth * tan_half_y)
        px = (0.5 * ndc_x + 0.5) * width
        py = (0.5 - 0.5 * ndc_y) * height
        return px, py, depth

    return project, right, up, tan_half_y


def _object_radius_world(item: dict[str, Any]) -> float:
    size = item.get("size", {})
    candidates = [
        float(value)
        for key, value in size.items()
        if key in {"radius", "height", "hx", "hy", "hz", "length"}
        and isinstance(value, (int, float))
    ]
    return max(candidates, default=0.12)


def build_pybullet_property_map(
    metadata: dict[str, Any],
    *,
    height: int,
    width: int,
) -> PropertyMapResult:
    maps = np.zeros((len(CHANNEL_NAMES), height, width), dtype=np.float32)
    project, camera_right, camera_up, tan_half_y = _camera_projector(metadata, height, width)
    rigid_count = 0
    motion_count = 0
    for item in metadata.get("objects", []):
        if not isinstance(item, dict):
            continue
        projected = project(item.get("position", [0.0, 0.0, 0.0]))
        if projected is None:
            continue
        px, py, depth = projected
        if px < -width or px > 2 * width or py < -height or py > 2 * height:
            continue
        radius_px = 0.5 * height * _object_radius_world(item) / max(depth * tan_half_y, 1.0e-5)
        velocity = np.asarray(item.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        speed = float(np.linalg.norm(velocity))
        direction_x = float(np.dot(velocity, camera_right))
        direction_y = float(-np.dot(velocity, camera_up))
        direction_norm = math.hypot(direction_x, direction_y)
        if direction_norm > 1.0e-8:
            direction_x /= direction_norm
            direction_y /= direction_norm
            motion_count += 1
        values = {
            0: float(np.clip(item.get("restitution", 0.0), 0.0, 1.0)),
            1: float(np.clip(item.get("friction", metadata.get("floor_friction", 0.0)), 0.0, 1.0)),
            2: 1.0,
            6: float(np.clip(speed / 2.0, 0.0, 1.0)),
            7: direction_x,
            8: direction_y,
        }
        _draw_circle(maps, (px, py), radius_px, values)
        rigid_count += 1
    branch_valid = torch.tensor([rigid_count > 0, False, motion_count > 0], dtype=torch.bool)
    return PropertyMapResult(
        maps=torch.from_numpy(maps).unsqueeze(1),
        branch_valid=branch_valid,
        diagnostics={
            "source": "pybullet",
            "rigid_object_count": rigid_count,
            "motion_object_count": motion_count,
            "force_supervision_kind": "initial_velocity_proxy",
            "deformation_supervision": False,
        },
    )


def _first_video_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read first frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _sequence_value(container: dict[str, Any], key: str, index: int, default: Any = None) -> Any:
    values = container.get(key)
    if isinstance(values, list) and index < len(values):
        return values[index]
    return default


def _soft_value(value: Any, scale: float) -> float:
    if value is None:
        return 0.0
    return float(np.clip(math.log1p(max(float(value), 0.0)) / math.log1p(scale), 0.0, 1.0))


def build_kubric_property_map(
    sample_dir: str | Path,
    *,
    height: int,
    width: int,
) -> PropertyMapResult:
    sample_dir = Path(sample_dir)
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    object_data = metadata.get("object_data", {})
    segmentation = _first_video_frame(sample_dir / "segmentation.mp4")
    colors = object_data.get("segmentation_color", [])
    object_names = [str(value) for value in object_data.get("type", [])]
    force_by_name: dict[str, tuple[float, np.ndarray]] = {}
    image_entries = metadata.get("applied_velocities_image", [])
    simulator_entries = metadata.get("applied_velocities_simulator", [])
    for entry_index, entry in enumerate(simulator_entries if isinstance(simulator_entries, list) else []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("object_name", metadata.get("velocity_applied_block", "")))
        magnitude = float(entry.get("velocity_magnitude", metadata.get("velocity_magnitude", 0.0)) or 0.0)
        direction = entry.get("velocity_vector_world", metadata.get("velocity_direction_xy", [0.0, 0.0]))
        if entry_index < len(image_entries) and isinstance(image_entries[entry_index], dict):
            direction = image_entries[entry_index].get("velocity_arrow_unit_vector", direction)
        vector = _normalized(np.asarray(direction[:2], dtype=np.float32))
        if name:
            force_by_name[name] = (magnitude, vector)
    fallback_name = str(metadata.get("velocity_applied_block", ""))
    if fallback_name and fallback_name not in force_by_name:
        vector = _normalized(
            np.asarray(
                metadata.get("velocity_direction_xy", metadata.get("velocity_direction", [0.0, 0.0]))[:2],
                dtype=np.float32,
            )
        )
        force_by_name[fallback_name] = (float(metadata.get("velocity_magnitude", 0.0) or 0.0), vector)
    maps = np.zeros((len(CHANNEL_NAMES), height, width), dtype=np.float32)
    rigid_count = deform_count = force_count = 0
    for index, color in enumerate(colors):
        target = np.asarray(color, dtype=np.int16)
        distance = np.abs(segmentation.astype(np.int16) - target.reshape(1, 1, 3)).max(axis=-1)
        mask = _resize_mask(distance <= 12, height, width)
        if not bool(mask.any()):
            continue
        restitution = _sequence_value(object_data, "restitution", index, 0.0)
        friction = _sequence_value(object_data, "friction", index, 0.0)
        maps[0, mask] = float(np.clip(restitution or 0.0, 0.0, 1.0))
        maps[1, mask] = float(np.clip(friction or 0.0, 0.0, 1.0))
        maps[2, mask] = 1.0
        rigid_count += 1

        lam = _sequence_value(object_data, "neo_hookean_lambda", index)
        mu = _sequence_value(object_data, "neo_hookean_mu", index)
        damping = _sequence_value(object_data, "neo_hookean_damping", index)
        deformable = any(value is not None for value in (lam, mu, damping)) or bool(
            _sequence_value(object_data, "use_neo_hookean", index, False)
        )
        if deformable:
            maps[3, mask] = _soft_value(lam, 1.0e6)
            maps[4, mask] = _soft_value(mu, 1.0e6)
            maps[5, mask] = 1.0
            deform_count += 1

        force = _sequence_value(object_data, "force", index)
        if force is None:
            force = _sequence_value(object_data, "applied_force", index)
        direction = _sequence_value(object_data, "move_dir", index)
        if direction is None:
            direction = _sequence_value(object_data, "force_direction", index)
        if isinstance(force, (list, tuple)) and len(force) >= 2:
            vector = np.asarray(force[:2], dtype=np.float32)
            magnitude = float(np.linalg.norm(vector))
            direction = vector
        else:
            magnitude = float(force or 0.0)
        object_name = object_names[index] if index < len(object_names) else ""
        if object_name in force_by_name:
            magnitude, vector = force_by_name[object_name]
            direction = vector.tolist()
        if isinstance(direction, (list, tuple, np.ndarray)) and len(direction) >= 2:
            vector = _normalized(np.asarray(direction[:2], dtype=np.float32))
            maps[6, mask] = float(np.clip(magnitude / 100.0, 0.0, 1.0))
            maps[7, mask] = float(vector[0])
            maps[8, mask] = float(vector[1])
            force_count += 1
    branch_valid = torch.tensor([rigid_count > 0, deform_count > 0, force_count > 0], dtype=torch.bool)
    return PropertyMapResult(
        maps=torch.from_numpy(maps).unsqueeze(1),
        branch_valid=branch_valid,
        diagnostics={
            "source": "kubric",
            "rigid_object_count": rigid_count,
            "deformable_object_count": deform_count,
            "force_object_count": force_count,
            "force_supervision_kind": "metadata",
        },
    )


def build_null_property_map(*, height: int, width: int, source: str = "openvid") -> PropertyMapResult:
    return PropertyMapResult(
        maps=torch.zeros(len(CHANNEL_NAMES), 1, height, width, dtype=torch.float32),
        branch_valid=torch.zeros(3, dtype=torch.bool),
        diagnostics={"source": source, "null_control": True},
    )
