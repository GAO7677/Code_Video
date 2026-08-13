from __future__ import annotations

import numpy as np
import torch

from AAA_my_test.wan_context_point_guidance.protocol_core import (
    fixed_mutable_rms_delta,
    global_context_point_loss,
    transform_points_stretch_to_cover_crop,
)


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
