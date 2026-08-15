from __future__ import annotations

import torch

from object_trajectory_loss import (
    object_equal_visibility_aware_trajectory_loss,
    object_trajectory_loss,
    visibility_aware_trajectory_loss,
)


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


def test_multiobject_loss_is_an_equal_object_mean() -> None:
    from run_training_case_diagnostics import aggregate_object_trajectory_losses

    pred = torch.zeros((1, 3, 4, 2), requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[:, 2, 2:, 0] = 2.0
    visibility = torch.ones((1, 3, 4), dtype=torch.bool)
    aggregate, _, rows = aggregate_object_trajectory_losses(
        pred,
        gt,
        visibility,
        object_count=2,
        points_per_object=2,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
    )
    expected = torch.stack([row["loss"] for row in rows]).mean()
    torch.testing.assert_close(aggregate, expected)
    assert len(rows) == 2
    assert float(rows[0]["loss"]) == 0.0
    assert float(rows[1]["loss"]) > 0.0


def test_visibility_aware_loss_ignores_occluded_coordinates() -> None:
    pred = torch.zeros((1, 3, 2, 2), requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[:, 2, 1, 0] = 10.0
    gt_visibility = torch.ones((1, 3, 2))
    gt_visibility[:, 2, 1] = 0.1
    gt_confidence = torch.ones_like(gt_visibility)
    pred_visibility = torch.ones_like(gt_visibility) * 0.95
    loss, diagnostics = visibility_aware_trajectory_loss(
        pred,
        gt,
        gt_visibility,
        gt_confidence,
        pred_visibility,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
        visibility_threshold=0.9,
        visibility_loss_weight=0.0,
    )
    assert float(loss) == 0.0
    assert int(diagnostics["valid_count"]) == 1


def test_prediction_visibility_penalizes_without_masking_coordinate_loss() -> None:
    pred = torch.zeros((1, 3, 1, 2), requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[:, 2, 0, 0] = 2.0
    gt_visibility = torch.ones((1, 3, 1))
    gt_confidence = torch.ones_like(gt_visibility)
    pred_visibility = torch.full_like(gt_visibility, 0.1, requires_grad=True)
    total, diagnostics = visibility_aware_trajectory_loss(
        pred,
        gt,
        gt_visibility,
        gt_confidence,
        pred_visibility,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
        visibility_threshold=0.9,
        visibility_loss_weight=0.05,
    )
    assert float(diagnostics["coordinate_loss"]) > 0.0
    assert float(diagnostics["visibility_loss"]) > 0.0
    total.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0
    assert pred_visibility.grad is not None
    assert float(pred_visibility.grad.abs().sum()) > 0.0


def test_visibility_aware_multiobject_loss_is_object_equal() -> None:
    pred = torch.zeros((1, 3, 4, 2), requires_grad=True)
    gt = torch.zeros_like(pred)
    gt[:, 2, 2:, 0] = 2.0
    scores = torch.ones((1, 3, 4))
    total, diagnostics = object_equal_visibility_aware_trajectory_loss(
        pred,
        gt,
        scores,
        scores,
        scores * 0.95,
        scores.bool(),
        object_count=2,
        points_per_object=2,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
        visibility_threshold=0.9,
        visibility_loss_weight=0.0,
    )
    object_one, _ = visibility_aware_trajectory_loss(
        pred[:, :, :2],
        gt[:, :, :2],
        scores[:, :, :2],
        scores[:, :, :2],
        scores[:, :, :2] * 0.95,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
        visibility_threshold=0.9,
        visibility_loss_weight=0.0,
        gt_geometric_visibility=scores[:, :, :2].bool(),
    )
    object_two, _ = visibility_aware_trajectory_loss(
        pred[:, :, 2:],
        gt[:, :, 2:],
        scores[:, :, 2:],
        scores[:, :, 2:],
        scores[:, :, 2:] * 0.95,
        height=11,
        width=21,
        anchor_frame=1,
        future_start_frame=2,
        huber_delta=0.01,
        visibility_threshold=0.9,
        visibility_loss_weight=0.0,
        gt_geometric_visibility=scores[:, :, 2:].bool(),
    )
    torch.testing.assert_close(total, (object_one + object_two) / 2)
    torch.testing.assert_close(total, diagnostics["coordinate_loss"])


def test_query_score_replacement_preserves_autograd() -> None:
    from run_training_case_diagnostics import replace_query_predictions

    tracks = torch.zeros((1, 3, 1, 2), requires_grad=True)
    visibility_logits = torch.zeros((1, 3, 1), requires_grad=True)
    confidence_logits = torch.zeros((1, 3, 1), requires_grad=True)
    queries = torch.tensor([[[1.0, 4.0, 5.0]]])
    replaced_tracks, replaced_visibility, replaced_confidence = (
        replace_query_predictions(
            tracks,
            visibility_logits.sigmoid(),
            confidence_logits.sigmoid(),
            queries,
        )
    )
    assert float(replaced_visibility[0, 1, 0]) == 1.0
    assert float(replaced_confidence[0, 1, 0]) == 1.0
    torch.testing.assert_close(replaced_tracks[0, 1, 0], queries[0, 0, 1:])
    loss = (
        replaced_tracks[:, 2:].sum()
        + replaced_visibility[:, 2:].sum()
        + replaced_confidence[:, 2:].sum()
    )
    loss.backward()
    assert tracks.grad is not None
    assert visibility_logits.grad is not None
    assert confidence_logits.grad is not None


def test_geometric_visibility_uses_corresponding_object_mask() -> None:
    import numpy as np
    from run_training_case_diagnostics import tracks_inside_object_masks

    masks = np.zeros((2, 3, 256, 448), dtype=bool)
    masks[0, :, 20:40, 30:50] = True
    masks[1, :, 100:120, 200:220] = True
    tracks = torch.tensor(
        [[
            [[40.0, 30.0], [40.0, 30.0], [300.0, 200.0], [205.0, 110.0]],
            [[40.0, 30.0], [40.0, 30.0], [300.0, 200.0], [205.0, 110.0]],
            [[40.0, 30.0], [40.0, 30.0], [300.0, 200.0], [205.0, 110.0]],
        ]]
    )
    visible = tracks_inside_object_masks(tracks, masks, points_per_object=2)
    assert visible.shape == (1, 3, 4)
    assert bool(visible[:, :, 0].all())
    assert bool(visible[:, :, 1].all())
    assert not bool(visible[:, :, 2].any())
    assert bool(visible[:, :, 3].all())
