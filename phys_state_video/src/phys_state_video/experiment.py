from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conditioning import ConditionBundle
from .utils import require_torch

torch = require_torch()


@dataclass(slots=True)
class EvalAverages:
    loss: float
    recon: float
    state_aux: float
    center_error: float
    log_scale_error: float
    visibility_error: float


def apply_condition_mode(bundle: ConditionBundle, mode: str) -> ConditionBundle:
    mode = mode.lower()
    if mode == "state":
        return bundle
    if mode == "maps_only":
        return ConditionBundle(maps=bundle.maps,
                               memory_tokens=torch.zeros_like(
                                   bundle.memory_tokens))
    if mode == "memory_only":
        return ConditionBundle(maps=torch.zeros_like(bundle.maps),
                               memory_tokens=bundle.memory_tokens)
    if mode == "none":
        return ConditionBundle(maps=torch.zeros_like(bundle.maps),
                               memory_tokens=torch.zeros_like(
                                   bundle.memory_tokens))
    raise ValueError(f"unsupported condition mode: {mode}")


def perturb_condition_bundle(bundle: ConditionBundle,
                             center_shift: float = 0.05,
                             scale: float = 0.85) -> ConditionBundle:
    pixel_shift_y = max(1, int(round(bundle.maps.shape[-2] * center_shift)))
    pixel_shift_x = max(1, int(round(bundle.maps.shape[-1] * center_shift)))
    shifted_maps = torch.roll(bundle.maps, shifts=1, dims=1)
    shifted_maps = torch.roll(shifted_maps,
                              shifts=(pixel_shift_y, pixel_shift_x),
                              dims=(-2, -1))
    shifted_maps = shifted_maps * scale
    shifted_memory = bundle.memory_tokens.clone()
    shifted_memory = shifted_memory * scale
    return ConditionBundle(maps=shifted_maps, memory_tokens=shifted_memory)


def compute_state_metrics(predicted_states: np.ndarray,
                          target_states: np.ndarray) -> dict[str, float]:
    center_error = np.linalg.norm(predicted_states[..., 0:2] -
                                  target_states[..., 0:2],
                                  axis=-1).mean()
    log_scale_error = np.abs(predicted_states[..., 3] -
                             target_states[..., 3]).mean()
    visibility_error = np.abs(predicted_states[..., 7] -
                              target_states[..., 7]).mean()
    return {
        "center_error": float(center_error),
        "log_scale_error": float(log_scale_error),
        "visibility_error": float(visibility_error),
    }
