from __future__ import annotations

import numpy as np
import torch

from AAA_my_test.wan_context_point_guidance.protocol_core import (
    fixed_mutable_rms_delta,
    forward_attention_audit_sums,
    global_context_point_loss,
    global_forward_point_loss,
    transform_points_stretch_to_cover_crop,
)
from AAA_my_test.wan_context_point_guidance.render_constraint_diagnostics import (
    _select_source_anchors,
)


def test_source_anchor_selection_supports_short_source_clips() -> None:
    frames = np.arange(30, dtype=np.int64)[:, None]
    anchors = np.array([0, 2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 27, 29])
    selected = _select_source_anchors(frames, anchors)
    assert selected.shape == (13, 1)
    assert selected[:, 0].tolist() == anchors.tolist()


def test_fixed_mutable_rms_delta_equal_budget_and_frozen_context() -> None:
    gradient_a = torch.randn(1, 4, 13, 3, 5)
    gradient_b = 17.0 * torch.randn_like(gradient_a)
    delta_a, audit_a = fixed_mutable_rms_delta(gradient_a, 2, 0.01)
    delta_b, audit_b = fixed_mutable_rms_delta(gradient_b, 2, 0.01)
    assert not bool(delta_a[:, :, :2].any())
    assert not bool(delta_b[:, :, :2].any())
    assert abs(audit_a["actual_mutable_update_rms"] - 0.01) < 1.0e-6
    assert abs(audit_b["actual_mutable_update_rms"] - 0.01) < 1.0e-6


def test_global_loss_uses_full_time_denominator() -> None:
    # One point: future Query at t=2 should select the same point at context t=0.
    # Raising a distractor logit at t=1 must increase loss if the denominator is
    # genuinely global rather than normalized only inside frame 0.
    q = torch.zeros(1, 3 * 4, 1, 2)
    k = torch.zeros_like(q)
    rows = torch.tensor([[0], [0], [0]])
    visibility = torch.ones(3, 1, dtype=torch.bool)
    q[:, 8, 0] = torch.tensor([1.0, 0.0])
    k[:, 0, 0] = torch.tensor([2.0, 0.0])
    base = global_context_point_loss(
        q, k, rows, visibility, (2, 2), (2,), (0,), 0.25
    )
    k[:, 4, 0] = torch.tensor([8.0, 0.0])
    distracted = global_context_point_loss(
        q, k, rows, visibility, (2, 2), (2,), (0,), 0.25
    )
    assert float(distracted) > float(base) + 1.0


def test_forward_loss_uses_context_query_and_future_key() -> None:
    q = torch.zeros(1, 3 * 4, 1, 2)
    k = torch.zeros_like(q)
    rows = torch.tensor([[0], [1], [3]])
    visibility = torch.ones(3, 1, dtype=torch.bool)
    q[:, 0, 0] = torch.tensor([1.0, 0.0])
    k[:, 11, 0] = torch.tensor([16.0, 0.0])
    aligned = global_forward_point_loss(
        q, k, rows, visibility, (2, 2), (0,), (2,), 0.25
    )
    q[:, 0, 0] = 0
    q[:, 11, 0] = torch.tensor([1.0, 0.0])
    future_query_only = global_forward_point_loss(
        q, k, rows, visibility, (2, 2), (0,), (2,), 0.25
    )
    assert float(aligned) + 1.0 < float(future_query_only)


def test_forward_loss_uses_full_spatiotemporal_denominator() -> None:
    q = torch.zeros(1, 3 * 4, 1, 2)
    k = torch.zeros_like(q)
    rows = torch.tensor([[0], [1], [3]])
    visibility = torch.ones(3, 1, dtype=torch.bool)
    q[:, 0, 0] = torch.tensor([1.0, 0.0])
    k[:, 11, 0] = torch.tensor([4.0, 0.0])
    base = global_forward_point_loss(
        q, k, rows, visibility, (2, 2), (0,), (2,), 0.25
    )
    k[:, 5, 0] = torch.tensor([8.0, 0.0])
    distracted = global_forward_point_loss(
        q, k, rows, visibility, (2, 2), (0,), (2,), 0.25
    )
    assert float(distracted) > float(base) + 1.0


def test_negative_gradient_update_reduces_forward_loss() -> None:
    q = torch.randn(1, 3 * 4, 1, 3, requires_grad=True)
    k = torch.randn_like(q)
    rows = torch.tensor([[0], [1], [2]])
    visibility = torch.ones(3, 1, dtype=torch.bool)
    loss = global_forward_point_loss(
        q, k, rows, visibility, (2, 2), (0, 1), (2,), 0.6
    )
    gradient = torch.autograd.grad(loss, q)[0]
    q_updated = (q.detach() - 1.0e-3 * gradient).requires_grad_(False)
    updated_loss = global_forward_point_loss(
        q_updated, k.detach(), rows, visibility, (2, 2), (0, 1), (2,), 0.6
    )
    assert float(updated_loss) < float(loss)


def test_forward_attention_audit_preserves_global_mass_and_localization() -> None:
    q = torch.zeros(1, 3 * 4, 1, 2)
    k = torch.zeros_like(q)
    rows = torch.tensor([[0], [1], [3]])
    visibility = torch.ones(3, 1, dtype=torch.bool)
    q[:, 0, 0] = torch.tensor([1.0, 0.0])
    k[:, 11, 0] = torch.tensor([16.0, 0.0])
    audit = forward_attention_audit_sums(
        q, k, rows, visibility, (2, 2), (0,), (1, 2), 0.25
    )
    # One Query/head pair has unit probability over all 12 spatiotemporal Keys.
    assert torch.allclose(audit["heatmap_sum"].sum(), torch.tensor(1.0))
    assert audit["heatmap_pair_count"].tolist() == [1.0, 1.0, 1.0]
    assert int(audit["heatmap_sum"][2].argmax()) == 3
    assert float(audit["localized_mass_sum"][2]) > 0.99
    assert float(audit["peak_distance_sum"][2]) == 0.0
    assert float(audit["peak_hit_sum"][2]) == 1.0


def test_negative_gradient_update_reduces_global_loss() -> None:
    q = torch.randn(1, 3 * 4, 1, 3, requires_grad=True)
    k = torch.randn_like(q)
    rows = torch.tensor([[0], [1], [2]])
    visibility = torch.ones(3, 1, dtype=torch.bool)
    loss = global_context_point_loss(
        q, k, rows, visibility, (2, 2), (2,), (0, 1), 0.6
    )
    gradient = torch.autograd.grad(loss, q)[0]
    q_updated = (q.detach() - 1.0e-3 * gradient).requires_grad_(False)
    updated_loss = global_context_point_loss(
        q_updated, k.detach(), rows, visibility, (2, 2), (2,), (0, 1), 0.6
    )
    assert float(updated_loss) < float(loss)


def test_identity_geometry_transform() -> None:
    points = np.asarray([[[10.0, 20.0], [100.0, 50.0]]], dtype=np.float32)
    transformed, in_frame, metadata = transform_points_stretch_to_cover_crop(
        points,
        source_hw=(100, 200),
        stretched_hw=(100, 200),
        crop_hw=(100, 200),
    )
    np.testing.assert_allclose(transformed, points, atol=1.0e-5)
    assert bool(in_frame.all())
    assert metadata["crop_top"] == 0
    assert metadata["crop_left"] == 0
