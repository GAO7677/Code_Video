from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


COUNT_BUCKET_ORDER = ("count_01", "count_02", "count_03_04")
COLLISION_PROFILE_ORDER = (
    "no_collision",
    "env_only",
    "obj_obj_only_c1",
    "obj_obj_only_c2plus",
    "mixed_c1",
    "mixed_c2plus",
)
COLLISION_PROFILE_LABELS = {
    "no_collision": "No Collision",
    "env_only": "Env Only",
    "obj_obj_only_c1": "Obj-Obj x1",
    "obj_obj_only_c2plus": "Obj-Obj x2+",
    "mixed_c1": "Mixed x1",
    "mixed_c2plus": "Mixed x2+",
}
DERIVED_TAG_VERSION = "v1"


def collision_type_bucket(events: Sequence[Dict[str, Any]]) -> str:
    obj_obj = sum(1 for event in events if -1 not in list(event.get("participants", [])))
    obj_env = sum(1 for event in events if -1 in list(event.get("participants", [])))
    if obj_obj == 0 and obj_env == 0:
        return "none"
    if obj_obj == 0:
        return "env_only"
    if obj_env == 0:
        return "obj_obj_only"
    return "mixed"


def collision_profile_bucket(obj_obj_count: int, obj_env_count: int) -> str:
    if obj_obj_count <= 0 and obj_env_count <= 0:
        return "no_collision"
    if obj_obj_count <= 0:
        return "env_only"
    if obj_env_count <= 0:
        return "obj_obj_only_c1" if obj_obj_count == 1 else "obj_obj_only_c2plus"
    return "mixed_c1" if obj_obj_count == 1 else "mixed_c2plus"


def bucket_display_label(count_bucket: str, collision_profile: str) -> str:
    collision_label = COLLISION_PROFILE_LABELS.get(collision_profile, collision_profile)
    return f"{count_bucket} | {collision_label}"


def collision_count_bucket(obj_obj_count: int) -> str:
    if obj_obj_count <= 0:
        return "c0"
    if obj_obj_count == 1:
        return "c1"
    return "c2plus"


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(np.float32), q))


def infer_sample_motion_complexity(
    linear_vel: np.ndarray,
    visibility_mask: np.ndarray,
    obj_obj_event_count: int,
) -> Dict[str, Any]:
    vel = np.asarray(linear_vel, dtype=np.float32)
    vis = np.asarray(visibility_mask).astype(bool)
    speed = np.linalg.norm(vel, axis=-1)
    valid_speed = speed[vis]

    pair_visible = vis[1:] & vis[:-1] if vel.shape[0] > 1 else np.zeros((0,) + vis.shape[1:], dtype=bool)
    accel = np.diff(vel, axis=0) if vel.shape[0] > 1 else np.zeros((0,) + vel.shape[1:], dtype=np.float32)
    accel_mag = np.linalg.norm(accel, axis=-1) if accel.size > 0 else np.zeros(pair_visible.shape, dtype=np.float32)
    valid_accel = accel_mag[pair_visible] if accel_mag.size > 0 else np.zeros((0,), dtype=np.float32)

    moving_threshold = 0.03
    moving_mask = vis & (speed >= moving_threshold)
    moving_frame_ratio = float(moving_mask.sum()) / float(max(int(vis.sum()), 1))
    moving_object_count = int(np.sum(np.any(moving_mask, axis=0)))

    speed_mean = float(valid_speed.mean()) if valid_speed.size else 0.0
    speed_p90 = safe_percentile(valid_speed, 90.0)
    accel_p90 = safe_percentile(valid_accel, 90.0)

    speed_score = min(speed_p90 / 4.0, 1.0)
    accel_score = min(accel_p90 / 3.0, 1.0)
    moving_ratio_score = min(moving_frame_ratio / 0.75, 1.0)
    moving_object_score = min(max(moving_object_count - 1, 0) / 2.0, 1.0)
    collision_score = min(float(obj_obj_event_count) / 2.0, 1.0)
    complexity_score = float(
        0.35 * speed_score
        + 0.25 * accel_score
        + 0.15 * moving_ratio_score
        + 0.15 * moving_object_score
        + 0.10 * collision_score
    )

    if speed_p90 < 0.02 and moving_frame_ratio < 0.08:
        label = "static"
    elif complexity_score < 0.28 and moving_object_count <= 1 and obj_obj_event_count == 0:
        label = "simple"
    elif complexity_score < 0.68 and moving_object_count <= 2:
        label = "moderate"
    else:
        label = "complex"

    return {
        "label": label,
        "score": float(complexity_score),
        "metrics": {
            "speed_mean": float(speed_mean),
            "speed_p90": float(speed_p90),
            "accel_p90": float(accel_p90),
            "moving_frame_ratio": float(moving_frame_ratio),
            "moving_object_count": int(moving_object_count),
        },
    }


def compute_derived_tags(
    metadata: Dict[str, Any],
    events: Sequence[Dict[str, Any]],
    linear_vel: np.ndarray,
    visibility_mask: np.ndarray,
    com_pos: Optional[np.ndarray] = None,
    bbox_xyxy: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    obj_obj_events = [event for event in events if -1 not in list(event.get("participants", []))]
    obj_env_events = [event for event in events if -1 in list(event.get("participants", []))]
    fused_obj_obj_count = int(len(obj_obj_events))
    fused_obj_env_count = 0
    if com_pos is not None:
        inferred = infer_collision_counts_from_kinematics(
            com_pos=np.asarray(com_pos, dtype=np.float32),
            linear_vel=np.asarray(linear_vel, dtype=np.float32),
            visibility_mask=np.asarray(visibility_mask).astype(bool),
            bbox_xyxy=None if bbox_xyxy is None else np.asarray(bbox_xyxy, dtype=np.float32),
        )
        fused_obj_obj_count = max(fused_obj_obj_count, int(inferred["obj_obj_event_count"]))
        # Environment events in old exports mix support contact with true impacts.
        # For bucketing, trust the kinematics-based onset count instead.
        fused_obj_env_count = int(inferred["obj_env_event_count"])
    else:
        fused_obj_env_count = int(len(obj_env_events))
    motion = infer_sample_motion_complexity(
        linear_vel=linear_vel,
        visibility_mask=visibility_mask,
        obj_obj_event_count=fused_obj_obj_count,
    )
    count_bucket = str(metadata.get("object_count_bucket", ""))
    collision_profile = collision_profile_bucket(fused_obj_obj_count, fused_obj_env_count)
    return {
        "derived_tag_version": DERIVED_TAG_VERSION,
        "motion_label": str(motion["label"]),
        "motion_score": float(motion["score"]),
        "motion_metrics": dict(motion["metrics"]),
        "collision_type_bucket": (
            "none"
            if fused_obj_obj_count <= 0 and fused_obj_env_count <= 0
            else "env_only"
            if fused_obj_obj_count <= 0
            else "obj_obj_only"
            if fused_obj_env_count <= 0
            else "mixed"
        ),
        "collision_profile_bucket": collision_profile,
        "collision_count_bucket": collision_count_bucket(fused_obj_obj_count),
        "obj_obj_event_count": int(fused_obj_obj_count),
        "obj_env_event_count": int(fused_obj_env_count),
        "bucket_key": f"{count_bucket}__{collision_profile}",
        "bucket_label": bucket_display_label(count_bucket, collision_profile),
    }


def load_sample_arrays(sample_dir: Path) -> Dict[str, Any]:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    physics_dir = sample_dir / "physics"
    event_path = physics_dir / "event_windows.json"
    if not event_path.exists():
        event_path = physics_dir / "collision_events.json"
    events = json.loads(event_path.read_text(encoding="utf-8"))
    kin = np.load(sample_dir / "physics" / "rigid_kinematics.npz")
    anchor = np.load(sample_dir / "physics" / "anchor_targets.npz")
    return {
        "metadata": metadata,
        "events": events,
        "linear_vel": np.asarray(kin["linear_vel"], dtype=np.float32),
        "com_pos": np.asarray(kin["com_pos"], dtype=np.float32),
        "bbox_xyxy": np.asarray(anchor["bbox_xyxy"], dtype=np.float32),
        "visibility_mask": np.asarray(anchor["visibility_mask"]) > 0.5,
    }


def _group_contiguous_true_ranges(mask: np.ndarray) -> List[tuple[int, int]]:
    ranges: List[tuple[int, int]] = []
    start: Optional[int] = None
    for idx in range(int(mask.shape[0])):
        is_active = bool(mask[idx])
        if is_active and start is None:
            start = idx
        elif (not is_active) and start is not None:
            ranges.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        ranges.append((int(start), int(mask.shape[0] - 1)))
    return ranges


def _bbox_intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_edge_gap(box_a: np.ndarray, box_b: np.ndarray) -> float:
    gap_x = max(float(box_b[0] - box_a[2]), float(box_a[0] - box_b[2]), 0.0)
    gap_y = max(float(box_b[1] - box_a[3]), float(box_a[1] - box_b[3]), 0.0)
    return max(gap_x, gap_y)


def infer_collision_counts_from_kinematics(
    com_pos: np.ndarray,
    linear_vel: np.ndarray,
    visibility_mask: np.ndarray,
    bbox_xyxy: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    pos = np.asarray(com_pos, dtype=np.float32)
    vel = np.asarray(linear_vel, dtype=np.float32)
    vis = np.asarray(visibility_mask).astype(bool)
    if pos.ndim != 3 or vel.ndim != 3 or pos.shape != vel.shape:
        return {"obj_obj_event_count": 0, "obj_env_event_count": 0}

    num_frames, num_objects, _ = pos.shape
    if num_frames <= 1 or num_objects <= 0:
        return {"obj_obj_event_count": 0, "obj_env_event_count": 0}

    speed = np.linalg.norm(vel, axis=-1)
    vel_jump = np.zeros((num_frames, num_objects), dtype=np.float32)
    vel_jump[1:] = np.linalg.norm(np.diff(vel, axis=0), axis=-1)

    obj_obj_event_count = 0
    if num_objects >= 2:
        for idx_a in range(num_objects):
            for idx_b in range(idx_a + 1, num_objects):
                pair_visible = vis[:, idx_a] & vis[:, idx_b]
                if not np.any(pair_visible):
                    continue
                dist = np.linalg.norm(pos[:, idx_a, :] - pos[:, idx_b, :], axis=-1)
                jump_pair = np.maximum(vel_jump[:, idx_a], vel_jump[:, idx_b]) > 0.35
                transfer_pair = np.zeros((num_frames,), dtype=bool)
                speed_a = speed[:, idx_a]
                speed_b = speed[:, idx_b]
                transfer_pair[1:] = (
                    ((speed_a[:-1] < 0.15) & (speed_a[1:] > 0.45) & (speed_b[:-1] > 0.8))
                    | ((speed_b[:-1] < 0.15) & (speed_b[1:] > 0.45) & (speed_a[:-1] > 0.8))
                )
                overlap_mask = np.zeros((num_frames,), dtype=bool)
                near_touch_mask = np.zeros((num_frames,), dtype=bool)
                if bbox_xyxy is not None:
                    for frame_idx in range(num_frames):
                        if not pair_visible[frame_idx]:
                            continue
                        box_a = bbox_xyxy[frame_idx, idx_a]
                        box_b = bbox_xyxy[frame_idx, idx_b]
                        overlap_mask[frame_idx] = _bbox_intersection_area(box_a, box_b) > 0.0
                        near_touch_mask[frame_idx] = _bbox_edge_gap(box_a, box_b) <= 4.0
                local_min_mask = np.zeros((num_frames,), dtype=bool)
                if num_frames >= 3:
                    local_min_mask[1:-1] = (
                        (dist[1:-1] <= dist[:-2] + 1e-4)
                        & (dist[1:-1] <= dist[2:] + 1e-4)
                        & (dist[1:-1] <= 0.32)
                    )
                candidate = pair_visible & (jump_pair | transfer_pair) & (
                    overlap_mask | (near_touch_mask & local_min_mask)
                )
                candidate[:1] = False
                obj_obj_event_count += len(_group_contiguous_true_ranges(candidate))

    obj_env_event_count = 0
    for obj_idx in range(num_objects):
        valid = vis[:, obj_idx]
        if not np.any(valid):
            continue
        z = pos[:, obj_idx, 2]
        z_valid = z[valid]
        z_min = float(np.min(z_valid))
        z_span = float(np.max(z_valid) - z_min)
        near_floor_margin = max(0.06, min(0.20, 0.12 * z_span + 0.02))
        near_floor = valid & (z <= z_min + near_floor_margin)
        vz = vel[:, obj_idx, 2]
        near_ranges = _group_contiguous_true_ranges(near_floor)
        for start, _end in near_ranges:
            if start <= 0:
                # Initial support contact is not a collision event.
                continue
            lookback_start = max(0, start - 4)
            recent_off_floor = valid[lookback_start:start] & (~near_floor[lookback_start:start])
            if not np.any(recent_off_floor):
                continue
            recent_z = z[lookback_start:start]
            airborne_height = float(np.max(recent_z) - z_min) if recent_z.size else 0.0
            descending = bool(np.min(vz[lookback_start:start]) < -0.15) if start > lookback_start else False
            strong_jump = bool(np.max(vel_jump[lookback_start : start + 1, obj_idx]) > 0.8)
            if airborne_height > near_floor_margin * 1.2 and (descending or strong_jump):
                obj_env_event_count += 1

    return {
        "obj_obj_event_count": int(obj_obj_event_count),
        "obj_env_event_count": int(obj_env_event_count),
    }
