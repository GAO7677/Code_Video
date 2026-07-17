from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MaskLossWeights:
    union: float = 0.20
    instance: float = 0.10
    static: float = 0.02
    background: float = 0.01
    unused: float = 0.01
    focal_bce: float = 0.25


def _soft_dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().reshape(-1)
    target = target.float().reshape(-1)
    intersection = (prediction * target).sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (
        prediction.sum() + target.sum() + 1e-6
    )


def _focal_bce_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    bce = F.binary_cross_entropy(prediction, target, reduction="none")
    probability = target * prediction + (1.0 - target) * (1.0 - prediction)
    alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha)
    return (alpha_factor * (1.0 - probability).pow(gamma) * bce).mean()


def _segmentation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    focal_bce_weight: float,
) -> torch.Tensor:
    return _soft_dice_loss(prediction, target) + focal_bce_weight * _focal_bce_loss(
        prediction, target
    )


def _pairwise_dice_cost(
    slots: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return clip-level [S,K] Dice cost without detaching model gradients."""
    slot_flat = slots.float().reshape(slots.shape[0], -1)
    target_flat = targets.float().reshape(targets.shape[0], -1)
    intersection = slot_flat @ target_flat.transpose(0, 1)
    denominator = slot_flat.sum(dim=1, keepdim=True) + target_flat.sum(
        dim=1, keepdim=True
    ).transpose(0, 1)
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def _hungarian_slots(slots: torch.Tensor, targets: torch.Tensor) -> list[int]:
    cost = _pairwise_dice_cost(slots, targets).detach().cpu().numpy()
    rows, columns = linear_sum_assignment(cost)
    assignment = [-1] * targets.shape[0]
    for row, column in zip(rows.tolist(), columns.tolist()):
        assignment[column] = row
    if any(index < 0 for index in assignment):
        raise RuntimeError(
            f"Could not assign {targets.shape[0]} targets to {slots.shape[0]} slots"
        )
    return assignment


def compute_mask_loss(
    predicted_masks: torch.Tensor,
    dynamic_instance_masks: torch.Tensor,
    dynamic_instance_valid: torch.Tensor,
    dynamic_union_mask: torch.Tensor,
    static_geometry_mask: torch.Tensor,
    mask_supervision_valid: torch.Tensor,
    instance_supervision_valid: torch.Tensor,
    weights: MaskLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Compute clip-level object-mask supervision on the V-JEPA latent grid.

    Args:
        predicted_masks: SAVi masks [B,T,S,H,W,1] or [B,T,S,H,W].
        dynamic_instance_masks: Soft patch occupancy [B,T,K,H,W].
        dynamic_instance_valid: Valid dynamic instances [B,K].
        dynamic_union_mask: Dynamic foreground occupancy [B,T,1,H,W].
        static_geometry_mask: Static physical-geometry occupancy [B,T,1,H,W].
        mask_supervision_valid: Whether any reliable mask exists for each sample [B].
        instance_supervision_valid: Whether K fits available object slots [B].

    Matching is performed once over the complete clip. This keeps a single
    predicted slot assigned to the same object across all latent timesteps.
    """
    weights = weights or MaskLossWeights()
    if predicted_masks.ndim == 6:
        if predicted_masks.shape[-1] != 1:
            raise ValueError(
                f"Expected singleton mask channel, got {tuple(predicted_masks.shape)}"
            )
        predicted_masks = predicted_masks.squeeze(-1)
    if predicted_masks.ndim != 5:
        raise ValueError(
            f"Expected predicted masks [B,T,S,H,W], got {tuple(predicted_masks.shape)}"
        )
    batch, time, num_slots, height, width = predicted_masks.shape
    expected_instance_prefix = (batch, time)
    if dynamic_instance_masks.shape[:2] != expected_instance_prefix:
        raise ValueError(
            "Dynamic mask batch/time mismatch: "
            f"predicted={tuple(predicted_masks.shape)}, "
            f"target={tuple(dynamic_instance_masks.shape)}"
        )
    if dynamic_instance_masks.shape[-2:] != (height, width):
        raise ValueError(
            f"Dynamic mask spatial shape {tuple(dynamic_instance_masks.shape[-2:])} "
            f"does not match predicted {(height, width)}"
        )

    zero = predicted_masks.float().sum(dim=(1, 2, 3, 4)) * 0.0
    result = {
        "mask_total": zero.clone(),
        "mask_union": zero.clone(),
        "mask_instance": zero.clone(),
        "mask_static": zero.clone(),
        "mask_background": zero.clone(),
        "mask_unused": zero.clone(),
        "mask_supervision_rate": mask_supervision_valid.float(),
        "mask_instance_supervision_rate": (
            mask_supervision_valid.bool() & instance_supervision_valid.bool()
        ).float(),
    }

    for sample_index in range(batch):
        if not bool(mask_supervision_valid[sample_index].item()):
            continue
        slots = predicted_masks[sample_index].float().permute(1, 0, 2, 3)
        dynamic_union = dynamic_union_mask[sample_index, :, 0].float().clamp(0, 1)
        static_target = static_geometry_mask[sample_index, :, 0].float().clamp(0, 1)
        background_target = (1.0 - dynamic_union - static_target).clamp(0, 1)

        valid_instances = dynamic_instance_valid[sample_index].bool()
        use_instances = bool(instance_supervision_valid[sample_index].item()) and bool(
            valid_instances.any().item()
        )
        targets = []
        target_roles = []
        if use_instances:
            instances = dynamic_instance_masks[sample_index, :, valid_instances]
            instances = instances.float().permute(1, 0, 2, 3).clamp(0, 1)
            targets.extend(instances.unbind(dim=0))
            target_roles.extend(["dynamic"] * instances.shape[0])
        if float(static_target.sum().item()) > 1e-6:
            targets.append(static_target)
            target_roles.append("static")
        targets.append(background_target)
        target_roles.append("background")
        if len(targets) > num_slots:
            raise ValueError(
                f"Mask targets={len(targets)} exceed num_slots={num_slots}; "
                "reduce max supervised instances"
            )
        target_tensor = torch.stack(targets, dim=0)
        assigned_slots = _hungarian_slots(slots, target_tensor)
        role_to_slots: dict[str, list[int]] = {
            "dynamic": [],
            "static": [],
            "background": [],
        }
        instance_losses = []
        static_losses = []
        background_losses = []
        for target_index, (role, slot_index) in enumerate(
            zip(target_roles, assigned_slots)
        ):
            role_to_slots[role].append(slot_index)
            value = _segmentation_loss(
                slots[slot_index], target_tensor[target_index], weights.focal_bce
            )
            if role == "dynamic":
                instance_losses.append(value)
            elif role == "static":
                static_losses.append(value)
            else:
                background_losses.append(value)

        if use_instances:
            dynamic_slot_indices = role_to_slots["dynamic"]
            predicted_union = slots[dynamic_slot_indices].sum(dim=0).clamp(0, 1)
        else:
            nondynamic = set(role_to_slots["static"] + role_to_slots["background"])
            dynamic_slot_indices = [index for index in range(num_slots) if index not in nondynamic]
            predicted_union = slots[dynamic_slot_indices].sum(dim=0).clamp(0, 1)
        union_loss = _segmentation_loss(
            predicted_union, dynamic_union, weights.focal_bce
        )
        instance_loss = (
            torch.stack(instance_losses).mean() if instance_losses else zero[sample_index]
        )
        static_loss = (
            torch.stack(static_losses).mean() if static_losses else zero[sample_index]
        )
        background_loss = (
            torch.stack(background_losses).mean()
            if background_losses
            else zero[sample_index]
        )
        assigned = set(assigned_slots)
        unused_indices = [index for index in range(num_slots) if index not in assigned]
        unused_loss = (
            slots[unused_indices].mean() if use_instances and unused_indices else zero[sample_index]
        )
        total = (
            weights.union * union_loss
            + weights.instance * instance_loss
            + weights.static * static_loss
            + weights.background * background_loss
            + weights.unused * unused_loss
        )
        result["mask_total"][sample_index] = total
        result["mask_union"][sample_index] = union_loss
        result["mask_instance"][sample_index] = instance_loss
        result["mask_static"][sample_index] = static_loss
        result["mask_background"][sample_index] = background_loss
        result["mask_unused"][sample_index] = unused_loss
    return result
