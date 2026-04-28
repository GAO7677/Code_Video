"""Utilities for classifying rigid-motion windows by motion complexity."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


MOTION_COMPLEXITY_LABELS = ("static", "simple", "moderate", "complex")
MOTION_COMPLEXITY_TO_ID = {name: idx for idx, name in enumerate(MOTION_COMPLEXITY_LABELS)}


def parse_motion_complexity_filter(value: str | Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    labels = {item for item in items if item}
    unknown = sorted(label for label in labels if label not in MOTION_COMPLEXITY_TO_ID)
    if unknown:
        raise ValueError(
            f"Unknown motion complexity label(s): {unknown}. "
            f"Expected one of {list(MOTION_COMPLEXITY_LABELS)}."
        )
    return labels


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(np.float32), q))


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.astype(np.float32).mean())


def infer_motion_complexity(
    state_norm: np.ndarray,
    visibility_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    state_norm = np.asarray(state_norm, dtype=np.float32)
    if state_norm.ndim != 3 or state_norm.shape[-1] < 9:
        raise ValueError(f"Expected state_norm with shape [T, N, >=9], got {state_norm.shape}")

    if visibility_mask is None:
        visible = state_norm[..., 8] > 0.5
    else:
        visible = np.asarray(visibility_mask).astype(np.float32) > 0.5
        if visible.shape != state_norm.shape[:2]:
            raise ValueError(
                f"visibility_mask shape {visible.shape} does not match state_norm {state_norm.shape[:2]}"
            )

    velocity = state_norm[..., 5:8]
    speed = np.linalg.norm(velocity, axis=-1)
    valid_speed = speed[visible]

    pair_visible = visible[1:] & visible[:-1] if state_norm.shape[0] > 1 else np.zeros((0,) + visible.shape[1:], dtype=bool)
    accel = np.diff(velocity, axis=0) if state_norm.shape[0] > 1 else np.zeros((0,) + velocity.shape[1:], dtype=np.float32)
    accel_mag = np.linalg.norm(accel, axis=-1) if accel.size > 0 else np.zeros(pair_visible.shape, dtype=np.float32)
    valid_accel = accel_mag[pair_visible] if accel_mag.size > 0 else np.zeros((0,), dtype=np.float32)

    moving_threshold = 0.01
    moving_mask = visible & (speed >= moving_threshold)
    moving_frame_ratio = float(moving_mask.sum()) / float(max(int(visible.sum()), 1))
    moving_object_count = int(np.sum(np.any(moving_mask, axis=0)))
    visible_object_count_mean = float(np.asarray(visible.sum(axis=1), dtype=np.float32).mean()) if visible.shape[0] > 0 else 0.0

    speed_mean = _safe_mean(valid_speed)
    speed_p90 = _safe_percentile(valid_speed, 90.0)
    accel_mean = _safe_mean(valid_accel)
    accel_p90 = _safe_percentile(valid_accel, 90.0)

    speed_score = min(speed_p90 / 0.08, 1.0)
    accel_score = min(accel_p90 / 0.04, 1.0)
    moving_ratio_score = min(moving_frame_ratio / 0.60, 1.0)
    moving_object_score = min(max(moving_object_count - 1, 0) / 2.0, 1.0)
    complexity_score = float(
        0.45 * speed_score
        + 0.25 * accel_score
        + 0.15 * moving_ratio_score
        + 0.15 * moving_object_score
    )

    if speed_p90 < 0.005 and moving_frame_ratio < 0.08:
        label = "static"
    elif complexity_score < 0.22 and moving_object_count <= 1:
        label = "simple"
    elif complexity_score < 0.55 and moving_object_count <= 2:
        label = "moderate"
    else:
        label = "complex"

    return {
        "label": label,
        "bucket_id": int(MOTION_COMPLEXITY_TO_ID[label]),
        "score": float(complexity_score),
        "metrics": {
            "speed_mean": float(speed_mean),
            "speed_p90": float(speed_p90),
            "accel_mean": float(accel_mean),
            "accel_p90": float(accel_p90),
            "moving_frame_ratio": float(moving_frame_ratio),
            "moving_object_count": int(moving_object_count),
            "visible_object_count_mean": float(visible_object_count_mean),
            "moving_speed_threshold": float(moving_threshold),
        },
    }


def build_inverse_frequency_weights(
    labels: Sequence[str],
    strength: float = 1.0,
) -> List[float]:
    counts = Counter(labels)
    weights: List[float] = []
    for label in labels:
        count = max(int(counts.get(label, 1)), 1)
        weights.append(float(1.0 / (float(count) ** float(strength))))
    return weights


def summarize_motion_complexity(labels: Iterable[str]) -> Dict[str, int]:
    counts = Counter(str(label) for label in labels)
    return {label: int(counts.get(label, 0)) for label in MOTION_COMPLEXITY_LABELS}
