"""Identity calibration that never rematches after the observed prefix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class AlignmentResult:
    slot_to_object: torch.Tensor
    matched_iou: torch.Tensor
    mean_matched_iou: float
    coverage: float


def hard_slot_masks(attention: torch.Tensor) -> torch.Tensor:
    """Convert `[T,K,H,W]` soft attention to exclusive boolean masks."""
    if attention.ndim != 4:
        raise ValueError(f"attention must be [T,K,H,W], got {attention.shape}")
    labels = attention.argmax(dim=1)
    return functional.one_hot(labels, num_classes=attention.shape[1]).permute(
        0, 3, 1, 2
    ).bool()


def pairwise_prefix_iou(
    slot_masks: torch.Tensor,
    object_masks: torch.Tensor,
    object_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Aggregate intersections/unions over time before computing IoU."""
    if slot_masks.ndim != 4 or object_masks.ndim != 4:
        raise ValueError("slot_masks and object_masks must be [T,N,H,W]")
    if slot_masks.shape[0] != object_masks.shape[0]:
        raise ValueError("slot/object masks have different time lengths")
    if slot_masks.shape[-2:] != object_masks.shape[-2:]:
        object_masks = functional.interpolate(
            object_masks.float(), size=slot_masks.shape[-2:], mode="nearest"
        ).bool()
    slot = slot_masks.bool()[:, :, None]
    obj = object_masks.bool()[:, None]
    intersection = (slot & obj).sum(dim=(0, 3, 4)).float()
    union = (slot | obj).sum(dim=(0, 3, 4)).float()
    iou = intersection / union.clamp_min(1)
    if object_valid is not None:
        iou[:, ~object_valid.bool()] = 0
    return iou


def calibrate_identity(
    attention: torch.Tensor,
    object_masks: torch.Tensor,
    object_valid: torch.Tensor,
    calibration_states: int = 4,
    mode: str = "prefix_oracle",
    minimum_iou: float = 0.05,
    dummy_cost: float = 0.80,
) -> AlignmentResult:
    """Solve one prefix mapping and return a frozen slot-to-object assignment.

    `prefix_oracle` uses all observed calibration states. `boundary_frozen`
    uses only the final observed state. Neither mode reads a future mask.
    """
    if mode not in {"prefix_oracle", "boundary_frozen"}:
        raise ValueError(f"Unsupported alignment mode: {mode}")
    if not 1 <= calibration_states <= attention.shape[0]:
        raise ValueError("calibration_states is outside the trajectory")
    if mode == "prefix_oracle":
        time_slice = slice(0, calibration_states)
    else:
        time_slice = slice(calibration_states - 1, calibration_states)

    iou = pairwise_prefix_iou(
        hard_slot_masks(attention[time_slice]),
        object_masks[time_slice],
        object_valid,
    )
    num_slots, num_objects = iou.shape
    cost = np.full(
        (num_slots, num_objects + num_slots), dummy_cost, dtype=np.float64
    )
    cost[:, :num_objects] = 1.0 - iou.detach().cpu().numpy()

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError("SciPy is required for identity calibration") from error
    rows, columns = linear_sum_assignment(cost)

    mapping = torch.full((num_slots,), -1, dtype=torch.long)
    matched_iou = torch.zeros(num_slots, dtype=torch.float32)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= num_objects or not bool(object_valid[column]):
            continue
        score = float(iou[row, column])
        if score < minimum_iou:
            continue
        mapping[row] = column
        matched_iou[row] = score

    valid_objects = int(object_valid.sum())
    matched_objects = int(torch.unique(mapping[mapping >= 0]).numel())
    selected = matched_iou[mapping >= 0]
    return AlignmentResult(
        slot_to_object=mapping,
        matched_iou=matched_iou,
        mean_matched_iou=float(selected.mean()) if selected.numel() else 0.0,
        coverage=matched_objects / max(valid_objects, 1),
    )


def per_frame_oracle_assignments(
    attention: torch.Tensor,
    object_masks: torch.Tensor,
    object_valid: torch.Tensor,
    minimum_iou: float = 0.05,
) -> torch.Tensor:
    """Return the labelled future-cheating ceiling for diagnostics only."""
    assignments = []
    for time_index in range(attention.shape[0]):
        result = calibrate_identity(
            attention[time_index : time_index + 1],
            object_masks[time_index : time_index + 1],
            object_valid,
            calibration_states=1,
            mode="boundary_frozen",
            minimum_iou=minimum_iou,
        )
        assignments.append(result.slot_to_object)
    return torch.stack(assignments)


def assignment_switch_rate(assignments: torch.Tensor) -> float:
    """Fraction of adjacent valid slot identities that change."""
    if assignments.ndim != 2:
        raise ValueError("assignments must be [T,K]")
    previous = assignments[:-1]
    current = assignments[1:]
    valid = (previous >= 0) & (current >= 0)
    if not bool(valid.any()):
        return 0.0
    return float((previous[valid] != current[valid]).float().mean())

