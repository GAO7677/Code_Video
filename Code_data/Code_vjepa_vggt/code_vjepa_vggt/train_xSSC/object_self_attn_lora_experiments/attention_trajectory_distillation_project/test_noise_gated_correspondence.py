from __future__ import annotations

import torch

from noise_gated_correspondence import (
    cross_frame_point_terms,
    gaussian_soft_targets,
    noise_reliability_gate,
    points_to_token_coordinates,
)


def test_gaussian_target_is_normalized_and_continuous_across_token_boundary():
    centers = torch.tensor([[[3.49, 2.0]], [[3.51, 2.0]]])
    target = gaussian_soft_targets(centers, token_hw=(4, 8), sigma_tokens=1.0)
    assert torch.allclose(target.sum(dim=-1), torch.ones(2, 1), atol=1.0e-6)
    assert float((target[0] - target[1]).abs().sum()) < 0.03


def test_noise_gate_decreases_and_disables_high_noise():
    values = noise_reliability_gate(
        torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]),
        gamma=1.0,
        cutoff=0.75,
    )
    assert torch.all(values[:-2] > values[1:-1])
    assert 0.0 < float(values[3]) < float(values[2])
    assert float(values[4]) == 0.0


def test_pixel_points_map_to_continuous_token_coordinates():
    points = torch.tensor([[[0.0, 0.0], [895.0, 511.0], [447.5, 255.5]]])
    mapped = points_to_token_coordinates(
        points,
        pixel_hw=(512, 896),
        token_hw=(16, 28),
    )
    assert torch.allclose(mapped[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(mapped[0, 1], torch.tensor([27.0, 15.0]))
    assert torch.allclose(mapped[0, 2], torch.tensor([13.5, 7.5]), atol=1.0e-5)


def test_soft_correspondence_loss_reaches_query_and_key():
    torch.manual_seed(4)
    q = torch.randn(1, 3 * 4, 2, 5, requires_grad=True)
    k = torch.randn(1, 3 * 4, 2, 5, requires_grad=True)
    coordinates = torch.tensor(
        [
            [[0.4, 0.3], [1.6, 0.8]],
            [[0.8, 0.4], [1.2, 0.9]],
            [[1.1, 0.5], [0.8, 1.0]],
        ]
    )
    visible = torch.ones(3, 2, dtype=torch.bool)
    terms = cross_frame_point_terms(
        q,
        k,
        point_coordinates_tn2=coordinates,
        point_visibility_tn=visible,
        token_hw=(2, 2),
        source_frame=0,
        sigma_tokens=0.8,
        coordinate_huber_beta=0.5,
    )
    valid = terms["valid"][None, :, None]
    loss = (
        terms["ce"].masked_select(valid).mean()
        + terms["coordinate_huber"].masked_select(valid).mean()
    )
    loss.backward()
    assert q.grad is not None and float(q.grad.norm()) > 0.0
    assert k.grad is not None and float(k.grad.norm()) > 0.0
