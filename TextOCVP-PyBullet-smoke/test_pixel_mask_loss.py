from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parent
TEXTOCVP_SRC = Path("/home/gaoya/Code_Video/TextOCVP-master/src")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(TEXTOCVP_SRC))

from data.Stage1Indexed import Stage1Indexed  # noqa: E402
from feature_space_stage1.mask_loss import compute_mask_loss  # noqa: E402


def _targets(batch: int, time: int, height: int, width: int) -> dict[str, torch.Tensor]:
    dynamic = torch.zeros(batch, time, 1, height, width)
    dynamic[..., : width // 2] = 1.0
    return {
        "dynamic_instance_masks": dynamic.clone(),
        "dynamic_instance_valid": torch.ones(batch, 1, dtype=torch.bool),
        "dynamic_union_mask": dynamic.clone(),
        "static_geometry_mask": torch.zeros(batch, time, 1, height, width),
        "mask_supervision_valid": torch.ones(batch, dtype=torch.bool),
        "instance_supervision_valid": torch.ones(batch, dtype=torch.bool),
    }


def test_channel_first_pixel_masks_are_supported() -> None:
    target = _targets(batch=1, time=2, height=4, width=6)
    dynamic = target["dynamic_union_mask"]
    background = 1.0 - dynamic
    unused = torch.zeros_like(dynamic)
    predicted = torch.stack((dynamic, background, unused), dim=2)
    result = compute_mask_loss(predicted_masks=predicted, **target)
    assert result["mask_total"].shape == (1,)
    assert torch.isfinite(result["mask_total"]).all()
    assert result["mask_union"].item() < 1.0e-4
    assert result["mask_instance"].item() < 1.0e-4


def test_unsupervised_sample_has_zero_mask_gradient() -> None:
    logits = torch.randn(1, 2, 3, 1, 4, 6, requires_grad=True)
    predicted = logits.softmax(dim=2)
    target = _targets(batch=1, time=2, height=4, width=6)
    target["mask_supervision_valid"].zero_()
    target["instance_supervision_valid"].zero_()
    result = compute_mask_loss(predicted_masks=predicted, **target)
    result["mask_total"].sum().backward()
    assert result["mask_total"].item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


def test_pixel_dataset_mask_target_shape_matches_savi_decoder() -> None:
    dataset = Stage1Indexed(
        index_root="/data/gaoya/AAA_test_video/0623_savi/indices",
        dataset_mode="mixed",
        split="train",
        num_frames=10,
        img_size=(216, 384),
        preprocess_mode="resize",
        load_masks=True,
        max_mask_instances=6,
        mask_temporal_stride=1,
        mask_spatial_stride=1,
        max_samples=1,
    )
    targets = dataset._empty_mask_targets()
    assert targets["dynamic_instance_masks"].shape == (10, 6, 216, 384)
    assert targets["dynamic_union_mask"].shape == (10, 1, 216, 384)
    assert targets["static_geometry_mask"].shape == (10, 1, 216, 384)
