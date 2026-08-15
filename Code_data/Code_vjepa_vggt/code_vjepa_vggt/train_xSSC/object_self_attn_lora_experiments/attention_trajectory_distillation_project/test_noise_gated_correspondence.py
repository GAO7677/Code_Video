from __future__ import annotations

import torch

from noise_gated_correspondence import (
    bilinear_sample_token_features,
    conditional_correspondence_objective,
    cross_frame_point_terms,
    gaussian_soft_targets,
    noise_reliability_gate,
    points_to_token_coordinates,
    uniform_object_correspondence_objective,
)


def test_gaussian_target_is_normalized_and_continuous_across_token_boundary():
    centers = torch.tensor([[[3.49, 2.0]], [[3.51, 2.0]]])
    target = gaussian_soft_targets(centers, token_hw=(4, 8), sigma_tokens=1.0)
    assert torch.allclose(target.sum(dim=-1), torch.ones(2, 1), atol=1.0e-6)
    assert float((target[0] - target[1]).abs().sum()) < 0.03


def test_noise_gate_decreases_smoothly_without_hard_cutoff():
    values = noise_reliability_gate(
        torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]),
        gamma=1.0,
    )
    assert torch.all(values[:-1] > values[1:])
    assert torch.all(values > 0.0)


def test_pixel_points_map_to_token_cell_centers():
    points = torch.tensor(
        [[
            [0.0, 0.0],
            [15.5, 15.5],
            [47.5, 47.5],
            [879.5, 495.5],
            [895.0, 511.0],
            [447.5, 255.5],
        ]]
    )
    mapped = points_to_token_coordinates(
        points,
        pixel_hw=(512, 896),
        token_hw=(16, 28),
    )
    assert torch.allclose(mapped[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(mapped[0, 1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(mapped[0, 2], torch.tensor([1.0, 1.0]))
    assert torch.allclose(mapped[0, 3], torch.tensor([27.0, 15.0]))
    assert torch.allclose(mapped[0, 4], torch.tensor([27.0, 15.0]))
    assert torch.allclose(mapped[0, 5], torch.tensor([13.5, 7.5]), atol=1.0e-5)


def test_source_query_is_bilinearly_sampled():
    features = torch.tensor([0.0, 2.0, 4.0, 6.0]).reshape(1, 4, 1, 1)
    coordinates = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    sampled = bilinear_sample_token_features(
        features,
        coordinates,
        token_hw=(2, 2),
    )
    assert sampled.shape == (1, 2, 1, 1)
    assert torch.allclose(sampled.flatten(), torch.tensor([3.0, 3.5]))


def test_aggregate_soft_correspondence_loss_reaches_query_and_key():
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
    )
    head_weights = torch.tensor([0.7, 0.3]).reshape(1, 1, 2, 1, 1)
    aggregate = (terms["attention"] * head_weights).sum(dim=2)
    target = terms["target"][None]
    objective = conditional_correspondence_objective(
        aggregate,
        target,
        terms["valid"][None],
        scheduler_sigma=0.5,
        gate_gamma=1.0,
        lambda_corr=0.01,
    )
    assert torch.allclose(objective["noise_gate"], torch.tensor(0.5))
    assert torch.allclose(
        objective["loss"],
        0.005 * objective["raw_soft_ce"],
    )
    objective["loss"].backward()
    assert q.grad is not None and float(q.grad.norm()) > 0.0
    assert k.grad is not None and float(k.grad.norm()) > 0.0


def test_uniform_object_loss_averages_objects_after_valid_pairs():
    logits = torch.tensor(
        [
            [[[3.0, 0.0], [0.0, 3.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
        ],
        requires_grad=True,
    )
    probability = logits.softmax(dim=-1)
    target = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    valid = torch.tensor([[[True, False]], [[True, True]]])
    objective = uniform_object_correspondence_objective(
        probability,
        target,
        valid,
        lambda_corr=0.01,
    )
    expected = objective["raw_soft_ce_per_object"].mean()
    assert torch.allclose(objective["raw_soft_ce"], expected)
    assert objective["valid_per_object"].tolist() == [1, 2]
    assert torch.allclose(objective["loss"], 0.01 * expected)
    objective["loss"].backward()
    assert logits.grad is not None and float(logits.grad.norm()) > 0.0


def test_uniform_object_loss_rejects_object_without_valid_pairs():
    probability = torch.full((2, 1, 1, 2), 0.5)
    target = torch.full((2, 1, 1, 2), 0.5)
    valid = torch.tensor([[[True]], [[False]]])
    try:
        uniform_object_correspondence_objective(
            probability,
            target,
            valid,
            lambda_corr=0.01,
        )
    except RuntimeError as error:
        assert "objects contain no valid" in str(error)
    else:
        raise AssertionError("expected empty-object validation failure")
