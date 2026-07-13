from __future__ import annotations

import torch

from .conditioned_residual import (
    LocalizedConditionedResidual,
    LocalizedResidualConfig,
    rasterize_observed_box_union,
    rasterize_observed_mask_union,
    step_gate,
)


def _controller(**config_kwargs) -> LocalizedConditionedResidual:
    return LocalizedConditionedResidual(
        torch.tensor([[[20.0, 10.0, 40.0, 30.0]]]),
        image_hw=(40, 80),
        config=LocalizedResidualConfig(**config_kwargs),
    )


def test_box_union_rasterization_and_dilation() -> None:
    boxes = torch.tensor([[[20.0, 10.0, 40.0, 30.0]]])
    support = rasterize_observed_box_union(
        boxes, image_hw=(40, 80), latent_hw=(4, 8), dilation_ratio=0.0
    )
    dilated = rasterize_observed_box_union(
        boxes, image_hw=(40, 80), latent_hw=(4, 8), dilation_ratio=0.50
    )
    assert tuple(support.shape) == (4, 8)
    assert 0 < support.sum() < dilated.sum() <= support.numel()


def test_step_gate_uses_normalized_denoising_progress() -> None:
    assert step_gate(0, 5, 0.0, 0.5) == 1.0
    assert step_gate(2, 5, 0.0, 0.5) == 1.0
    assert step_gate(3, 5, 0.0, 0.5) == 0.0


def test_mask_union_reduces_time_and_objects_and_dilates() -> None:
    masks = torch.zeros(2, 2, 8, 8)
    masks[0, 0, 2:4, 2:4] = 1
    masks[1, 1, 4:6, 4:6] = 1
    support = rasterize_observed_mask_union(masks, latent_hw=(4, 4), dilation_ratio=0.0)
    dilated = rasterize_observed_mask_union(masks, latent_hw=(4, 4), dilation_ratio=0.5)
    assert tuple(support.shape) == (4, 4)
    assert 0 < support.sum() < dilated.sum() <= support.numel()


def test_controller_prefers_mask_support_over_boxes() -> None:
    masks = torch.zeros(1, 1, 40, 80)
    masks[..., 10:20, 20:30] = 1
    controller = LocalizedConditionedResidual(
        torch.tensor([[[0.0, 0.0, 80.0, 40.0]]]),
        image_hw=(40, 80),
        config=LocalizedResidualConfig(scale=1.0, dilation_ratio=0.0),
        observed_masks=masks,
    )
    base = torch.zeros(1, 1, 2, 4, 8)
    output = controller.blend(base, torch.ones_like(base), step_index=0, total_steps=2, prefix_latent_frames=0)
    assert 0 < torch.count_nonzero(output) < output.numel()
    assert controller.step_records[0]["support_source"] == "mask_union"


def test_zero_scale_is_exact_base_equivalence() -> None:
    base = torch.randn(1, 2, 4, 4, 8)
    conditioned = torch.randn_like(base)
    output = _controller(scale=0.0).blend(
        base, conditioned, step_index=0, total_steps=4, prefix_latent_frames=1
    )
    assert torch.equal(output, base)


def test_localization_preserves_prefix_and_exterior() -> None:
    base = torch.zeros(1, 1, 3, 4, 8)
    conditioned = torch.ones_like(base)
    output = _controller(scale=1.0, dilation_ratio=0.0).blend(
        base, conditioned, step_index=0, total_steps=4, prefix_latent_frames=1
    )
    assert torch.count_nonzero(output[:, :, 0]) == 0
    assert torch.count_nonzero(output[:, :, 1:, :, :2]) == 0
    assert torch.count_nonzero(output[:, :, 1:]) > 0


def test_non_finite_residual_is_sanitized() -> None:
    base = torch.zeros(1, 1, 2, 4, 8)
    conditioned = torch.ones_like(base)
    conditioned[..., 1, 2, 3] = float("nan")
    output = _controller(scale=1.0).blend(
        base, conditioned, step_index=0, total_steps=4, prefix_latent_frames=0
    )
    assert torch.isfinite(output).all()
