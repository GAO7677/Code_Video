from __future__ import annotations

import torch

from AAA_my_test.wan_context_point_guidance.direct_attention import (
    intervene_attention_rows,
    point_attention_targets,
)


def test_future_to_context_targets_use_same_tracked_point() -> None:
    rows = torch.tensor([[0, 3], [1, 2], [2, 1]])
    visibility = torch.ones(3, 2, dtype=torch.bool)
    query_rows, targets = point_attention_targets(
        rows,
        visibility,
        (2, 2),
        context_times=(0,),
        future_times=(1, 2),
        direction="future_to_context",
        sigma_tokens=0.1,
        device=torch.device("cpu"),
    )
    assert query_rows.tolist() == [5, 6, 9, 10]
    assert torch.allclose(targets.sum(dim=-1), torch.ones(4))
    # Future point 0 at R1 reads point 0 at context token 0.
    assert int(targets[0].argmax()) == 0
    # Future point 1 at R1 reads point 1 at context token 3.
    assert int(targets[1].argmax()) == 3


def test_bidirectional_targets_include_context_and_future_query_rows() -> None:
    rows = torch.tensor([[0], [1], [2]])
    visibility = torch.ones(3, 1, dtype=torch.bool)
    query_rows, targets = point_attention_targets(
        rows,
        visibility,
        (2, 2),
        context_times=(0,),
        future_times=(1, 2),
        direction="bidirectional",
        sigma_tokens=0.1,
        device=torch.device("cpu"),
    )
    assert query_rows.tolist() == [0, 5, 10]
    assert set(torch.nonzero(targets[0] > 1.0e-4)[:, 0].tolist()) == {5, 10}
    assert int(targets[1].argmax()) == 0
    assert int(targets[2].argmax()) == 0


def test_attention_intervention_matches_tv_budget_and_reduces_target_ce() -> None:
    torch.manual_seed(7)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 11, 4, 8)
    v = torch.randn_like(k)
    targets = torch.zeros(3, 11)
    targets[:, 2] = 1.0
    output, metrics = intervene_attention_rows(q, k, v, targets, 0.1)
    assert output.shape == q.shape
    assert torch.allclose(
        metrics["after"].sum(dim=-1),
        torch.ones_like(metrics["actual_tv"]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        metrics["actual_tv"],
        torch.full_like(metrics["actual_tv"], 0.1),
        atol=1.0e-5,
    )
    assert bool((metrics["target_ce_after"] < metrics["target_ce_before"]).all())
