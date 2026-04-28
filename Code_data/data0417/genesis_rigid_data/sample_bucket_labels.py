from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

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
) -> Dict[str, Any]:
    obj_obj_events = [event for event in events if -1 not in list(event.get("participants", []))]
    obj_env_events = [event for event in events if -1 in list(event.get("participants", []))]
    motion = infer_sample_motion_complexity(
        linear_vel=linear_vel,
        visibility_mask=visibility_mask,
        obj_obj_event_count=len(obj_obj_events),
    )
    count_bucket = str(metadata.get("object_count_bucket", ""))
    collision_profile = collision_profile_bucket(len(obj_obj_events), len(obj_env_events))
    return {
        "derived_tag_version": DERIVED_TAG_VERSION,
        "motion_label": str(motion["label"]),
        "motion_score": float(motion["score"]),
        "motion_metrics": dict(motion["metrics"]),
        "collision_type_bucket": collision_type_bucket(events),
        "collision_profile_bucket": collision_profile,
        "collision_count_bucket": collision_count_bucket(len(obj_obj_events)),
        "obj_obj_event_count": int(len(obj_obj_events)),
        "obj_env_event_count": int(len(obj_env_events)),
        "bucket_key": f"{count_bucket}__{collision_profile}",
        "bucket_label": bucket_display_label(count_bucket, collision_profile),
    }


def load_sample_arrays(sample_dir: Path) -> Dict[str, Any]:
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    events = json.loads((sample_dir / "physics" / "collision_events.json").read_text(encoding="utf-8"))
    kin = np.load(sample_dir / "physics" / "rigid_kinematics.npz")
    anchor = np.load(sample_dir / "physics" / "anchor_targets.npz")
    return {
        "metadata": metadata,
        "events": events,
        "linear_vel": np.asarray(kin["linear_vel"], dtype=np.float32),
        "visibility_mask": np.asarray(anchor["visibility_mask"]) > 0.5,
    }

