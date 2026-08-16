from __future__ import annotations

import torch

from noise_gated_correspondence import (
    cross_frame_mask_terms,
    masks_to_token_occupancy,
    token_occupancy_to_pixel,
    uniform_object_region_correspondence_objective,
)


def test_mask_mapping_uses_area_occupancy_and_latent_aligned_frames():
    masks = torch.zeros(1, 49, 512, 896)
    masks[:, :, 32:64, 64:96] = 1.0
    masks[:, 4, 32:48, 64:96] = 0.0
    frame_indices = torch.arange(13) * 4

    occupancy = masks_to_token_occupancy(
        masks.index_select(1, frame_indices),
        token_hw=(16, 28),
    )

    assert occupancy.shape == (1, 13, 16, 28)
    assert torch.allclose(occupancy[0, 0, 1, 2], torch.tensor(1.0))
    assert torch.allclose(occupancy[0, 1, 1, 2], torch.tensor(0.5))
    assert torch.allclose(occupancy[0, 2:, 1, 2], torch.ones(11))
    assert torch.count_nonzero(occupancy) == 13


def test_any_occupancy_reverse_mapping_contains_every_gt_pixel():
    masks = torch.zeros(1, 2, 512, 896)
    masks[0, 0, 31:66, 63:98] = 1.0
    masks[0, 1, 220:287, 410:491] = 1.0
    occupancy = masks_to_token_occupancy(masks, token_hw=(16, 28))
    reverse_support = token_occupancy_to_pixel(
        occupancy > 0,
        pixel_hw=(512, 896),
    ).bool()

    assert reverse_support.shape == masks.shape
    assert not bool((masks.bool() & ~reverse_support).any())
    assert bool((reverse_support & ~masks.bool()).any())


def test_mask_region_attention_is_normalized_and_reaches_query_and_key():
    torch.manual_seed(9)
    q = torch.randn(1, 3 * 4, 2, 5, requires_grad=True)
    k = torch.randn(1, 3 * 4, 2, 5, requires_grad=True)
    occupancy = torch.zeros(2, 3, 2, 2)
    occupancy[0, 0, 0, 0] = 1.0
    occupancy[0, 1, 0, 1] = 0.75
    occupancy[0, 2, 1, 1] = 1.0
    occupancy[1, 0, 1, 0] = 0.25
    occupancy[1, 0, 1, 1] = 0.75
    occupancy[1, 1, 1, 0] = 1.0
    occupancy[1, 2, 0, 0] = 0.5
    occupancy[1, 2, 0, 1] = 0.5

    terms = cross_frame_mask_terms(
        q,
        k,
        object_token_occupancy_othw=occupancy,
        source_frame=0,
    )

    assert terms["attention"].shape == (1, 2, 3, 2, 4)
    assert terms["target"].shape == (2, 3, 4)
    assert terms["valid"].tolist() == [[False, True, True], [False, True, True]]
    assert torch.allclose(
        terms["attention"].sum(dim=-1),
        torch.ones(1, 2, 3, 2),
        atol=1.0e-6,
    )
    assert torch.allclose(terms["target"].sum(dim=-1), torch.ones(2, 3))

    aggregate = terms["attention"].mean(dim=3).squeeze(0)
    objective = uniform_object_region_correspondence_objective(
        aggregate,
        terms["target"],
        terms["valid"],
        lambda_corr=0.01,
    )
    objective["loss"].backward()
    assert q.grad is not None and float(q.grad.norm()) > 0.0
    assert k.grad is not None and float(k.grad.norm()) > 0.0


def test_region_loss_averages_frames_within_each_object_then_objects():
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ],
        requires_grad=True,
    )
    probability = logits.softmax(dim=-1)
    target = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ]
    )
    valid = torch.tensor([[False, True, False], [False, True, True]])

    objective = uniform_object_region_correspondence_objective(
        probability,
        target,
        valid,
        lambda_corr=0.02,
    )

    assert objective["valid_per_object"].tolist() == [1, 2]
    assert torch.allclose(
        objective["raw_soft_ce"],
        objective["raw_soft_ce_per_object"].mean(),
    )
    assert torch.allclose(objective["loss"], 0.02 * objective["raw_soft_ce"])
