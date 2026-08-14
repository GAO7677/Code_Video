from __future__ import annotations

import torch

from object_trajectory_loss import object_trajectory_loss


def test_zero_loss_for_equal_tracks() -> None:
    tracks = torch.tensor(
        [[[[1.0, 2.0]], [[2.0, 4.0]], [[4.0, 8.0]]]],
        requires_grad=True,
    )
    visibility = torch.ones((1, 3, 1), dtype=torch.bool)
    loss, diagnostics = object_trajectory_loss(
        tracks,
        tracks.detach().clone(),
        visibility,
        height=9,
        width=9,
        anchor_frame=0,
        future_start_frame=1,
        huber_delta=0.01,
    )
    assert float(loss) == 0.0
    assert float(diagnostics["normalized_ade"]) == 0.0
    loss.backward()
    assert tracks.grad is not None


def test_loss_has_gradient_to_prediction() -> None:
    pred = torch.zeros((1, 4, 2, 2), requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[:, 2:, :, 0] = 4.0
    visibility = torch.ones((1, 4, 2), dtype=torch.bool)
    loss, diagnostics = object_trajectory_loss(
        pred,
        gt,
        visibility,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
    )
    loss.backward()
    assert float(loss) > 0.0
    assert float(diagnostics["normalized_ade"]) > 0.0
    assert pred.grad is not None
    assert float(pred.grad.abs().sum()) > 0.0


def test_training_loss_is_raw_huber_divided_by_beta() -> None:
    pred = torch.zeros((1, 3, 1, 2), requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[:, 2, :, 0] = 2.0
    visibility = torch.ones((1, 3, 1), dtype=torch.bool)
    beta = 0.01
    loss, diagnostics = object_trajectory_loss(
        pred,
        gt,
        visibility,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=beta,
    )
    torch.testing.assert_close(diagnostics["raw_huber"], loss * beta)
    torch.testing.assert_close(diagnostics["per_frame_loss"], loss.reshape(1, 1))


def test_gt_visibility_is_the_only_validity_gate() -> None:
    pred = torch.zeros((1, 4, 2, 2), requires_grad=True)
    gt = torch.zeros_like(pred)
    pred.data[:, 3, 0] = 10.0
    pred.data[:, 3, 1] = 100.0
    visibility = torch.zeros((1, 4, 2), dtype=torch.bool)
    visibility[:, 3, 0] = True
    loss, diagnostics = object_trajectory_loss(
        pred,
        gt,
        visibility,
        height=101,
        width=101,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
    )
    loss.backward()
    assert int(diagnostics["valid_count"]) == 1
    assert pred.grad is not None
    assert float(pred.grad[:, 3, 0].abs().sum()) > 0.0
    assert float(pred.grad[:, 3, 1].abs().sum()) == 0.0
