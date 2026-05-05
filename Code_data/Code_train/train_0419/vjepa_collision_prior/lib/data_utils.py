from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCENE_RE = re.compile(r"^(?P<object_id>[^_]+)__case(?P<case_id>\d+)(?P<suffix>.*)$")


@dataclass(frozen=True)
class SampleMeta:
    scene_id: str
    object_id: str
    case_id: str
    scene_composition: str
    interaction_pattern: str
    num_objects: int
    frames: int
    fps: int
    source_dir: str
    rgb_dir: str
    physics_dir: str
    metadata_path: str


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_scene_id(scene_id: str) -> tuple[str, str]:
    match = SCENE_RE.match(scene_id)
    if not match:
        raise ValueError(f"Could not parse scene id: {scene_id}")
    return match.group("object_id"), match.group("case_id")


def discover_source_catalog(dataset_root: str | Path) -> dict[str, SampleMeta]:
    dataset_root = Path(dataset_root)
    catalog: dict[str, SampleMeta] = {}
    for metadata_path in dataset_root.glob("train/rigid/*/*/*/metadata.json"):
        payload = read_json(metadata_path)
        scene_id = payload["scene_id"]
        object_id, case_id = parse_scene_id(scene_id)
        source_dir = metadata_path.parent
        dt = float(payload["simulation"]["dt"])
        steps_per_frame = int(payload["simulation"]["steps_per_frame"])
        fps = int(round(1.0 / (dt * steps_per_frame)))
        catalog[scene_id] = SampleMeta(
            scene_id=scene_id,
            object_id=object_id,
            case_id=case_id,
            scene_composition=payload["scene_composition"],
            interaction_pattern=payload["interaction_pattern"],
            num_objects=int(payload["num_objects"]),
            frames=int(payload["frames"]),
            fps=fps,
            source_dir=str(source_dir),
            rgb_dir=str(source_dir / "rgb"),
            physics_dir=str(source_dir / "physics"),
            metadata_path=str(metadata_path),
        )
    return catalog


@lru_cache(maxsize=4096)
def load_collision_events(source_dir: str, include_environment: bool = False) -> list[dict[str, Any]]:
    physics_dir = Path(source_dir) / "physics"
    events_path = physics_dir / "event_windows.json"
    if not events_path.exists():
        events_path = physics_dir / "collision_events.json"
    if not events_path.exists():
        return []
    rows = read_json(events_path)
    if include_environment:
        return rows
    filtered = []
    for event in rows:
        participants = event.get("object_indices", event.get("participants", []))
        if len(participants) >= 2 and all(int(x) >= 0 for x in participants[:2]):
            filtered.append(event)
    return filtered


def primary_collision_event(source_dir: str, include_environment: bool = False) -> dict[str, Any] | None:
    events = load_collision_events(source_dir, include_environment=include_environment)
    if not events:
        return None
    return sorted(events, key=lambda item: (int(item.get("start_frame", 10**9)), int(item.get("peak_frame", 10**9))))[0]


@lru_cache(maxsize=4096)
def load_kinematics(source_dir: str) -> dict[str, np.ndarray]:
    npz_path = Path(source_dir) / "physics" / "rigid_kinematics.npz"
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def collision_scene_ids(catalog: dict[str, SampleMeta], include_environment: bool = False) -> list[str]:
    scene_ids = []
    for scene_id, meta in catalog.items():
        if primary_collision_event(meta.source_dir, include_environment=include_environment) is not None:
            scene_ids.append(scene_id)
    return scene_ids


def grouped_scene_ids(catalog: dict[str, SampleMeta]) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    for scene_id, meta in catalog.items():
        grouped.setdefault(meta.object_id, {}).setdefault(meta.scene_composition, []).append(scene_id)
    return grouped


def candidate_frame_range(start: int, width: int) -> list[int]:
    return list(range(start, start + width))


def is_valid_window(total_frames: int, start: int, width: int) -> bool:
    return 0 <= start and (start + width) <= total_frames


def frame_paths(rgb_dir: str | Path, frame_indices: list[int]) -> list[Path]:
    rgb_dir = Path(rgb_dir)
    return [rgb_dir / f"frame_{idx:03d}.png" for idx in frame_indices]


def load_rgb_clip(rgb_dir: str | Path, frame_indices: list[int]) -> list[np.ndarray]:
    frames = []
    for frame_path in frame_paths(rgb_dir, frame_indices):
        with Image.open(frame_path) as image:
            frames.append(np.array(image.convert("RGB")))
    return frames


def transform_bbox_xyxy(
    bbox_xyxy: np.ndarray,
    image_hw: tuple[int, int],
    crop_size: int,
    short_side_size: int | None = None,
) -> np.ndarray:
    if short_side_size is None:
        short_side_size = int(crop_size * 256 / 224)

    height, width = image_hw
    scale = short_side_size / min(height, width)
    resized_h = int(round(height * scale))
    resized_w = int(round(width * scale))
    offset_y = max(0, (resized_h - crop_size) // 2)
    offset_x = max(0, (resized_w - crop_size) // 2)

    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    x1 = x1 * scale - offset_x
    x2 = x2 * scale - offset_x
    y1 = y1 * scale - offset_y
    y2 = y2 * scale - offset_y

    transformed = np.array(
        [
            np.clip(x1, 0, crop_size),
            np.clip(y1, 0, crop_size),
            np.clip(x2, 0, crop_size),
            np.clip(y2, 0, crop_size),
        ],
        dtype=np.float32,
    )
    return transformed


def temporal_union_bboxes(
    source_dir: str,
    frame_indices: list[int],
    object_indices: list[int],
    tubelet_size: int,
    crop_size: int,
    image_hw: tuple[int, int] = (720, 960),
) -> dict[int, np.ndarray]:
    kin = load_kinematics(source_dir)
    bbox_xyxy = kin["bbox_xyxy"][frame_indices]
    visibility = kin["visibility_mask"][frame_indices]
    grouped: dict[int, list[np.ndarray]] = {}
    for obj_idx in object_indices:
        grouped[obj_idx] = []
        for start in range(0, len(frame_indices), tubelet_size):
            stop = min(len(frame_indices), start + tubelet_size)
            vis = visibility[start:stop, obj_idx]
            box_group = bbox_xyxy[start:stop, obj_idx]
            valid_boxes = [transform_bbox_xyxy(box, image_hw=image_hw, crop_size=crop_size) for box, keep in zip(box_group, vis) if keep]
            if not valid_boxes:
                grouped[obj_idx].append(np.array([0, 0, 0, 0], dtype=np.float32))
                continue
            stack = np.stack(valid_boxes)
            union = np.array(
                [stack[:, 0].min(), stack[:, 1].min(), stack[:, 2].max(), stack[:, 3].max()],
                dtype=np.float32,
            )
            grouped[obj_idx].append(union)
        grouped[obj_idx] = np.stack(grouped[obj_idx], axis=0)
    return grouped


def token_indices_from_bboxes(
    bboxes_by_object: dict[int, np.ndarray],
    grid_size: int,
    crop_size: int,
) -> dict[int, np.ndarray]:
    patch_size = crop_size / grid_size
    token_ids: dict[int, np.ndarray] = {}
    for obj_idx, seq_boxes in bboxes_by_object.items():
        per_time_ids: list[np.ndarray] = []
        for time_idx, box in enumerate(seq_boxes):
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                per_time_ids.append(np.empty((0,), dtype=np.int64))
                continue
            left = int(np.clip(math.floor(x1 / patch_size), 0, grid_size - 1))
            right = int(np.clip(math.ceil(x2 / patch_size), 1, grid_size))
            top = int(np.clip(math.floor(y1 / patch_size), 0, grid_size - 1))
            bottom = int(np.clip(math.ceil(y2 / patch_size), 1, grid_size))
            ids = []
            for row in range(top, bottom):
                for col in range(left, right):
                    ids.append(time_idx * grid_size * grid_size + row * grid_size + col)
            per_time_ids.append(np.array(ids, dtype=np.int64))
        token_ids[obj_idx] = np.concatenate(per_time_ids, axis=0) if per_time_ids else np.empty((0,), dtype=np.int64)
    return token_ids


def constant_velocity_rollout(source_dir: str, frame_indices: list[int], future_width: int, object_indices: list[int]) -> np.ndarray:
    kin = load_kinematics(source_dir)
    com_pos = kin["com_pos"]
    linear_vel = kin["linear_vel"]
    last_context = frame_indices[-1]
    dt = 1.0 / 12.0
    pred = []
    for step in range(1, future_width + 1):
        current = []
        for obj_idx in object_indices:
            pos = com_pos[last_context, obj_idx]
            vel = linear_vel[last_context, obj_idx]
            current.append(pos + vel * dt * step)
        pred.append(np.stack(current, axis=0))
    return np.stack(pred, axis=0)


def actual_future_positions(source_dir: str, frame_indices: list[int], object_indices: list[int]) -> np.ndarray:
    kin = load_kinematics(source_dir)
    com_pos = kin["com_pos"][frame_indices]
    current = []
    for obj_idx in object_indices:
        if obj_idx >= com_pos.shape[1]:
            current.append(np.zeros((len(frame_indices), 3), dtype=np.float32))
        else:
            current.append(com_pos[:, obj_idx])
    return np.stack(current, axis=1)


def rng_for_key(seed: int, key: str) -> random.Random:
    return random.Random(f"{seed}:{key}")
