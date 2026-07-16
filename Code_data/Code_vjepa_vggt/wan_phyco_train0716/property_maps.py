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
    "neo_hookean_mu_norm",
    "neo_hookean_lambda_norm",
    "neo_hookean_damping_norm",
    "deformation_valid",
    "action_magnitude_norm",
    "action_direction_x",
    "action_direction_y",
    "action_type",
    "action_valid",
)
BRANCH_SLICES = ((0, 3), (3, 7), (7, 12))

VELOCITY_MAGNITUDE_MAX = 5.0
NEO_HOOKEAN_MU_RANGE = (60.0, 600.0)
NEO_HOOKEAN_LAMBDA_RANGE = (100.0, 600.0)
ACTION_TYPE_VELOCITY = -1.0
ACTION_TYPE_FORCE = 1.0


@dataclass(frozen=True)
class PropertyMapResult:
    maps: torch.Tensor
    branch_valid: torch.Tensor
    diagnostics: dict[str, Any]


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-8 else np.zeros_like(vector)


def _minmax(value: Any, minimum: float, maximum: float) -> float:
    if value is None or maximum <= minimum:
        return 0.0
    return float(np.clip((float(value) - minimum) / (maximum - minimum), 0.0, 1.0))


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
            7: float(np.clip(speed / VELOCITY_MAGNITUDE_MAX, 0.0, 1.0)),
            8: direction_x,
            9: direction_y,
            10: ACTION_TYPE_VELOCITY if direction_norm > 1.0e-8 else 0.0,
            11: 1.0 if direction_norm > 1.0e-8 else 0.0,
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
            "action_supervision_kind": "initial_velocity",
            "velocity_magnitude_max": VELOCITY_MAGNITUDE_MAX,
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


def _resolve_action_object(
    object_data: dict[str, Any],
    entry: dict[str, Any],
    *,
    point_key: str,
) -> tuple[int | None, str]:
    explicit_names = [str(value) for value in object_data.get("object_name", [])]
    types = [str(value) for value in object_data.get("type", [])]
    target = str(entry.get("object_name", ""))
    if target in explicit_names:
        return explicit_names.index(target), "object_name"
    if target in types:
        return types.index(target), "object_type"

    point = entry.get(point_key)
    positions = object_data.get("position", [])
    if isinstance(point, (list, tuple)) and len(point) >= 3 and positions:
        point_array = np.asarray(point[:3], dtype=np.float32)
        distances = [
            float(np.linalg.norm(np.asarray(position[:3], dtype=np.float32) - point_array))
            for position in positions
        ]
        best = int(np.argmin(distances))
        if distances[best] <= 0.05:
            return best, "action_point"

    background_types = {
        "dome",
        "ground",
        "ground_plane",
        "cube_platform",
        "platform",
        "pool_table",
        "wall",
    }
    candidates = [index for index, value in enumerate(types) if value not in background_types]
    if len(candidates) == 1:
        return candidates[0], "unique_foreground"
    return None, "unresolved"


def _collect_kubric_actions(
    metadata: dict[str, Any],
    object_data: dict[str, Any],
) -> tuple[dict[int, tuple[float, np.ndarray, float]], dict[str, int]]:
    actions: dict[int, tuple[float, np.ndarray, float]] = {}
    diagnostics: dict[str, int] = {
        "velocity_entries": 0,
        "force_entries": 0,
        "resolved_entries": 0,
        "unresolved_entries": 0,
    }
    specs = (
        (
            "velocity",
            "applied_velocities_simulator",
            "applied_velocities_image",
            "velocity_magnitude",
            "velocity_vector_world",
            "velocity_arrow_unit_vector",
            "velocity_point_world",
            ACTION_TYPE_VELOCITY,
        ),
        (
            "force",
            "applied_forces_simulator",
            "applied_forces_image",
            "force_magnitude",
            "force_vector_world",
            "force_arrow_unit_vector",
            "force_point_world",
            ACTION_TYPE_FORCE,
        ),
    )
    for kind, simulator_key, image_key, magnitude_key, vector_key, arrow_key, point_key, action_type in specs:
        simulator_entries = metadata.get(simulator_key, [])
        image_entries = metadata.get(image_key, [])
        if not isinstance(simulator_entries, list):
            continue
        diagnostics[f"{kind}_entries"] += len(simulator_entries)
        for entry_index, entry in enumerate(simulator_entries):
            if not isinstance(entry, dict):
                continue
            object_index, method = _resolve_action_object(object_data, entry, point_key=point_key)
            if object_index is None:
                diagnostics["unresolved_entries"] += 1
                continue
            image_entry = image_entries[entry_index] if entry_index < len(image_entries) else None
            direction = entry.get(vector_key, [0.0, 0.0])
            if isinstance(image_entry, dict):
                direction = image_entry.get(arrow_key, direction)
                if arrow_key == "force_arrow_unit_vector" and arrow_key not in image_entry:
                    start = image_entry.get("image_coordinates")
                    end = image_entry.get("force_end_image_coordinates")
                    if isinstance(start, list) and isinstance(end, list):
                        direction = [end[0] - start[0], end[1] - start[1]]
            vector = _normalized(np.asarray(direction[:2], dtype=np.float32))
            magnitude = float(entry.get(magnitude_key, 0.0) or 0.0)
            if kind == "velocity":
                magnitude_norm = float(np.clip(magnitude / VELOCITY_MAGNITUDE_MAX, 0.0, 1.0))
            else:
                minimum = float(metadata.get("min_force", metadata.get("force_magnitude_min", 0.0)))
                maximum = float(metadata.get("max_force", metadata.get("force_magnitude_max", 450.0)))
                magnitude_norm = _minmax(magnitude, minimum, maximum)
            actions[object_index] = (magnitude_norm, vector, action_type)
            diagnostics["resolved_entries"] += 1
            diagnostics[f"binding_{method}"] = diagnostics.get(f"binding_{method}", 0) + 1
    return actions, diagnostics


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
    actions, action_diagnostics = _collect_kubric_actions(metadata, object_data)
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
            maps[3, mask] = _minmax(mu, *NEO_HOOKEAN_MU_RANGE)
            maps[4, mask] = _minmax(lam, *NEO_HOOKEAN_LAMBDA_RANGE)
            maps[5, mask] = float(np.clip(float(damping or 0.0), 0.0, 1.0))
            maps[6, mask] = 1.0
            deform_count += 1

        if index in actions:
            magnitude_norm, vector, action_type = actions[index]
            maps[7, mask] = magnitude_norm
            maps[8, mask] = float(vector[0])
            maps[9, mask] = float(vector[1])
            maps[10, mask] = action_type
            maps[11, mask] = 1.0
            force_count += 1
    branch_valid = torch.tensor([rigid_count > 0, deform_count > 0, force_count > 0], dtype=torch.bool)
    return PropertyMapResult(
        maps=torch.from_numpy(maps).unsqueeze(1),
        branch_valid=branch_valid,
        diagnostics={
            "source": "kubric",
            "rigid_object_count": rigid_count,
            "deformable_object_count": deform_count,
            "action_object_count": force_count,
            "action_supervision_kind": "typed_velocity_or_force",
            "velocity_magnitude_max": VELOCITY_MAGNITUDE_MAX,
            "neo_hookean_mu_range": list(NEO_HOOKEAN_MU_RANGE),
            "neo_hookean_lambda_range": list(NEO_HOOKEAN_LAMBDA_RANGE),
            **action_diagnostics,
        },
    )


def build_null_property_map(*, height: int, width: int, source: str = "openvid") -> PropertyMapResult:
    return PropertyMapResult(
        maps=torch.zeros(len(CHANNEL_NAMES), 1, height, width, dtype=torch.float32),
        branch_valid=torch.zeros(3, dtype=torch.bool),
        diagnostics={"source": source, "null_control": True},
    )
