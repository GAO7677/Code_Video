from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


class StateIndex:
    CENTER_X = 0
    CENTER_Y = 1
    DEPTH = 2
    LOG_SCALE = 3
    VEL_X = 4
    VEL_Y = 5
    DEPTH_VEL = 6
    VISIBILITY = 7
    EXISTENCE = 8
    CONFIDENCE = 9


STATE_DIM = 10


@dataclass(slots=True)
class FrameObjectState:
    track_id: int
    center: np.ndarray
    depth: float
    log_scale: float
    velocity: np.ndarray
    depth_velocity: float
    visibility: float
    existence: float
    confidence: float
    bbox: np.ndarray
    appearance: Optional[np.ndarray] = None
    scale_depth_consistency: Optional[float] = None

    def to_vector(self) -> np.ndarray:
        vector = np.zeros((STATE_DIM,), dtype=np.float32)
        vector[StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = self.center.astype(np.float32)
        vector[StateIndex.DEPTH] = np.float32(self.depth)
        vector[StateIndex.LOG_SCALE] = np.float32(self.log_scale)
        vector[StateIndex.VEL_X:StateIndex.VEL_Y + 1] = self.velocity.astype(np.float32)
        vector[StateIndex.DEPTH_VEL] = np.float32(self.depth_velocity)
        vector[StateIndex.VISIBILITY] = np.float32(self.visibility)
        vector[StateIndex.EXISTENCE] = np.float32(self.existence)
        vector[StateIndex.CONFIDENCE] = np.float32(self.confidence)
        return vector


@dataclass(slots=True)
class EpisodeArrays:
    context_frames: np.ndarray
    future_frames: np.ndarray
    context_states: np.ndarray
    future_states: np.ndarray
    context_boxes: np.ndarray
    future_boxes: np.ndarray
    appearance: np.ndarray
    camera: np.ndarray
    prompt: str = ""

    def validate(self) -> None:
        if self.context_states.shape[-1] != STATE_DIM:
            raise ValueError(f"context state dim must be {STATE_DIM}, got {self.context_states.shape[-1]}")
        if self.future_states.shape[-1] != STATE_DIM:
            raise ValueError(f"future state dim must be {STATE_DIM}, got {self.future_states.shape[-1]}")
        if self.context_boxes.shape[-1] != 4 or self.future_boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [..., 4]")


def stack_state_vectors(states: Iterable[FrameObjectState]) -> np.ndarray:
    return np.stack([state.to_vector() for state in states], axis=0)


def compute_box_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((4,), dtype=np.float32)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def box_log_scale(box: np.ndarray, eps: float = 1e-6) -> float:
    width = max(float(box[2] - box[0]), 0.0)
    height = max(float(box[3] - box[1]), 0.0)
    area = max(width * height, eps)
    return float(np.log(area))


def normalize_box(box: np.ndarray, height: int, width: int) -> np.ndarray:
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return box.astype(np.float32) / scale


def denormalize_box(box: np.ndarray, height: int, width: int) -> np.ndarray:
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return box.astype(np.float32) * scale
