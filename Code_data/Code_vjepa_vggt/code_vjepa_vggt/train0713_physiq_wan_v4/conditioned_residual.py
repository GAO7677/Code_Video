from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LocalizedResidualConfig:
    scale: float = 1.0
    dilation_ratio: float = 0.15
    active_step_start: float = 0.0
    active_step_end: float = 1.0
    condition_confidence: float = 1.0

    def validate(self) -> None:
        if self.scale < 0.0:
            raise ValueError("scale must be non-negative")
        if self.dilation_ratio < 0.0:
            raise ValueError("dilation_ratio must be non-negative")
        if not 0.0 <= self.active_step_start <= self.active_step_end <= 1.0:
            raise ValueError("active step range must satisfy 0 <= start <= end <= 1")
        if not 0.0 <= self.condition_confidence <= 1.0:
            raise ValueError("condition_confidence must be in [0, 1]")


def rasterize_observed_box_union(
    boxes_xyxy_px: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    latent_hw: tuple[int, int],
    dilation_ratio: float,
) -> torch.Tensor:
    """Rasterize a conservative 2D union from prefix-only box tubes."""
    if boxes_xyxy_px.ndim == 2:
        boxes_xyxy_px = boxes_xyxy_px.unsqueeze(0)
    if boxes_xyxy_px.ndim != 3 or boxes_xyxy_px.shape[-1] != 4:
        raise ValueError(f"boxes must have shape [T,O,4] or [O,4], got {list(boxes_xyxy_px.shape)}")
    image_h, image_w = (int(value) for value in image_hw)
    latent_h, latent_w = (int(value) for value in latent_hw)
    if min(image_h, image_w, latent_h, latent_w) <= 0:
        raise ValueError("image and latent dimensions must be positive")

    boxes = torch.nan_to_num(boxes_xyxy_px.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
    support = torch.zeros((latent_h, latent_w), dtype=torch.float32, device=boxes.device)
    for box in boxes[valid]:
        pad_x = float(dilation_ratio) * float(box[2] - box[0])
        pad_y = float(dilation_ratio) * float(box[3] - box[1])
        x1 = max(0, min(latent_w, int(torch.floor((box[0] - pad_x) / image_w * latent_w).item())))
        y1 = max(0, min(latent_h, int(torch.floor((box[1] - pad_y) / image_h * latent_h).item())))
        x2 = max(0, min(latent_w, int(torch.ceil((box[2] + pad_x) / image_w * latent_w).item())))
        y2 = max(0, min(latent_h, int(torch.ceil((box[3] + pad_y) / image_h * latent_h).item())))
        if x2 > x1 and y2 > y1:
            support[y1:y2, x1:x2] = 1.0
    return support


def rasterize_observed_mask_union(
    masks: torch.Tensor,
    *,
    latent_hw: tuple[int, int],
    dilation_ratio: float,
) -> torch.Tensor:
    """Reduce prefix object masks to a latent-resolution spatial union."""
    if masks.ndim < 2:
        raise ValueError(f"masks must end with [H,W], got {list(masks.shape)}")
    latent_h, latent_w = (int(value) for value in latent_hw)
    mask = torch.nan_to_num(masks.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if mask.ndim > 2:
        mask = mask.reshape(-1, int(mask.shape[-2]), int(mask.shape[-1])).amax(dim=0)
    mask = (mask > 0.5).float()
    if dilation_ratio > 0.0 and bool(mask.any()):
        equivalent_radius = float(mask.sum().sqrt().item())
        radius = max(1, int(round(float(dilation_ratio) * equivalent_radius)))
        mask = F.max_pool2d(
            mask.view(1, 1, *mask.shape),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[0, 0]
    return F.interpolate(
        mask.view(1, 1, *mask.shape),
        size=(latent_h, latent_w),
        mode="nearest",
    )[0, 0]


def step_gate(step_index: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = float(step_index) / float(max(total_steps - 1, 1))
    return 1.0 if float(start) <= progress <= float(end) else 0.0


class LocalizedConditionedResidual:
    """Blend a conditioned CFG prediction into a branch-disabled CFG prediction."""

    def __init__(
        self,
        boxes_xyxy_px: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        config: LocalizedResidualConfig,
        observed_masks: torch.Tensor | None = None,
    ) -> None:
        config.validate()
        self.boxes_xyxy_px = boxes_xyxy_px.detach().float().cpu()
        self.image_hw = tuple(int(value) for value in image_hw)
        self.config = config
        self.observed_masks = None if observed_masks is None else observed_masks.detach().float().cpu()
        self.step_records: list[dict[str, float | int | bool]] = []

    def blend(
        self,
        base_prediction: torch.Tensor,
        conditioned_prediction: torch.Tensor,
        *,
        step_index: int,
        total_steps: int,
        prefix_latent_frames: int,
    ) -> torch.Tensor:
        if base_prediction.shape != conditioned_prediction.shape or base_prediction.ndim != 5:
            raise ValueError("predictions must have matching [B,C,T,H,W] shapes")
        gate = step_gate(
            step_index,
            total_steps,
            self.config.active_step_start,
            self.config.active_step_end,
        )
        residual = conditioned_prediction - base_prediction
        if self.observed_masks is not None and bool(self.observed_masks.any()):
            support_2d = rasterize_observed_mask_union(
                self.observed_masks,
                latent_hw=(int(residual.shape[-2]), int(residual.shape[-1])),
                dilation_ratio=self.config.dilation_ratio,
            )
            support_source = "mask_union"
        else:
            support_2d = rasterize_observed_box_union(
                self.boxes_xyxy_px,
                image_hw=self.image_hw,
                latent_hw=(int(residual.shape[-2]), int(residual.shape[-1])),
                dilation_ratio=self.config.dilation_ratio,
            )
            support_source = "box_union_fallback"
        support_2d = support_2d.to(device=residual.device, dtype=residual.dtype)
        support = support_2d.view(1, 1, 1, *support_2d.shape).expand(
            int(residual.shape[0]), 1, int(residual.shape[2]), -1, -1
        ).clone()
        prefix = max(0, min(int(prefix_latent_frames), int(residual.shape[2])))
        support[:, :, :prefix] = 0
        alpha = float(self.config.scale * self.config.condition_confidence * gate)
        localized = torch.nan_to_num(residual * support * alpha)
        output = base_prediction + localized
        with torch.no_grad():
            base_l2 = float(torch.linalg.vector_norm(base_prediction.float()).item())
            residual_l2 = float(torch.linalg.vector_norm(residual.float()).item())
            localized_l2 = float(torch.linalg.vector_norm(localized.float()).item())
            self.step_records.append(
                {
                    "step_index": int(step_index),
                    "active": bool(gate > 0.0),
                    "alpha": alpha,
                    "support_coverage": float(support.float().mean().item()),
                    "residual_to_base_l2": residual_l2 / max(base_l2, 1.0e-12),
                    "localized_to_base_l2": localized_l2 / max(base_l2, 1.0e-12),
                    "finite": bool(torch.isfinite(output).all().item()),
                    "support_source": support_source,
                }
            )
        return output

    def summary(self) -> dict[str, object]:
        return {
            "config": {
                "scale": self.config.scale,
                "dilation_ratio": self.config.dilation_ratio,
                "active_step_start": self.config.active_step_start,
                "active_step_end": self.config.active_step_end,
                "condition_confidence": self.config.condition_confidence,
            },
            "image_hw": list(self.image_hw),
            "observed_boxes_shape": list(self.boxes_xyxy_px.shape),
            "observed_masks_shape": None if self.observed_masks is None else list(self.observed_masks.shape),
            "steps": list(self.step_records),
        }
