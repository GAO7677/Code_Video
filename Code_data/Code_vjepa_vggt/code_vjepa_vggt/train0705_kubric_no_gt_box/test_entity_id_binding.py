from __future__ import annotations

import torch

from code_vjepa_vggt.models.object_condition_adapter import ObjectConditionAdapter
from code_vjepa_vggt.models.object_entity_id_binder import (
    EntityIDBindingObjectConditionAdapter,
    find_subsequence_spans,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve_entity_id_binding import (
    _phrase_candidates,
)


def _adapter(*, zero_output: bool) -> EntityIDBindingObjectConditionAdapter:
    adapter = EntityIDBindingObjectConditionAdapter(
        dim=4,
        num_slots=2,
        max_time_steps=2,
        entity_text_dim=4,
        entity_bottleneck_dim=4,
        entity_gate_init=1.0,
        entity_dropout_prob=0.0,
        entity_residual_max_ratio=0.0,
    )
    if not zero_output:
        with torch.no_grad():
            adapter.entity_text_down.weight.copy_(torch.eye(4))
            adapter.entity_text_up.weight.copy_(torch.eye(4))
            adapter.entity_id_embed.weight.copy_(
                torch.tensor(
                    [[0.5, 0.0, 0.0, 0.0], [0.0, 0.75, 0.0, 0.0]]
                )
            )
    adapter.eval()
    return adapter


def test_find_subsequence_spans_returns_repeated_mentions() -> None:
    assert find_subsequence_spans([1, 2, 3, 2, 3, 4], [2, 3]) == [(1, 3), (3, 5)]


def test_phrase_candidates_extract_multiword_alias_before_action_suffix() -> None:
    candidates = _phrase_candidates("round rigid object, enters")
    assert candidates[0] == "round rigid object enters"
    assert "round rigid object" in candidates
    assert "ball" in candidates


def test_zero_initialized_binding_preserves_old_adapter_output() -> None:
    torch.manual_seed(0)
    old = ObjectConditionAdapter(dim=4, num_slots=2, max_time_steps=2)
    bound = _adapter(zero_output=True)
    load_info = bound.load_state_dict(old.state_dict(), strict=False)
    assert not load_info.unexpected_keys
    old.eval()
    bound.eval()

    tokens = torch.randn((1, 2, 2, 4))
    boxes = torch.rand((1, 2, 2, 4))
    valid = torch.ones((1, 2))
    bound.set_entity_binding_context(
        entity_text_by_id=torch.randn((1, 2, 4)),
        entity_text_match_mask=torch.ones((1, 2), dtype=torch.bool),
        slot_entity_ids=torch.tensor([[1, 0]]),
    )
    expected = old(tokens, object_valid_mask=valid, bbox_xyxy=boxes)
    actual = bound(tokens, object_valid_mask=valid, bbox_xyxy=boxes)
    torch.testing.assert_close(actual, expected)


def test_hard_entity_id_routing_swaps_slot_residuals() -> None:
    adapter = _adapter(zero_output=False)
    tokens = torch.ones((1, 1, 2, 4))
    entity_text = torch.tensor(
        [[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]]
    )
    matched = torch.ones((1, 2), dtype=torch.bool)
    valid = torch.ones((1, 2))

    adapter.set_entity_binding_context(
        entity_text_by_id=entity_text,
        entity_text_match_mask=matched,
        slot_entity_ids=torch.tensor([[0, 1]]),
    )
    direct = adapter.apply_entity_binding(tokens, object_valid_mask=valid) - tokens
    adapter.set_entity_binding_context(
        entity_text_by_id=entity_text,
        entity_text_match_mask=matched,
        slot_entity_ids=torch.tensor([[1, 0]]),
    )
    swapped = adapter.apply_entity_binding(tokens, object_valid_mask=valid) - tokens

    torch.testing.assert_close(direct[:, :, 0], swapped[:, :, 1])
    torch.testing.assert_close(direct[:, :, 1], swapped[:, :, 0])
    assert not torch.allclose(direct[:, :, 0], direct[:, :, 1])


def test_two_same_noun_entities_get_distinct_id_conditioning_without_collision() -> None:
    adapter = _adapter(zero_output=False)
    tokens = torch.zeros((1, 2, 2, 4))
    same_ball_text = torch.tensor(
        [[[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]]
    )
    adapter.set_entity_binding_context(
        entity_text_by_id=same_ball_text,
        entity_text_match_mask=torch.ones((1, 2), dtype=torch.bool),
        slot_entity_ids=torch.tensor([[1, 0]]),
    )
    output = adapter.apply_entity_binding(
        tokens,
        object_valid_mask=torch.ones((1, 2)),
    )
    metrics = adapter.pop_entity_binding_metrics()

    assert not torch.allclose(output[:, :, 0], output[:, :, 1])
    assert metrics["train/entity_binding_matched_slot_count"] == 2.0
    assert metrics["train/entity_binding_unique_id_count"] == 2.0
    assert metrics["train/entity_binding_id_collision_count"] == 0.0


def test_zero_initialized_projection_has_finite_backward_and_optimizer_step() -> None:
    torch.manual_seed(1)
    adapter = _adapter(zero_output=True)
    adapter.train()
    adapter.entity_residual_max_ratio = 0.1
    adapter.set_entity_binding_context(
        entity_text_by_id=torch.randn((1, 2, 4)),
        entity_text_match_mask=torch.ones((1, 2), dtype=torch.bool),
        slot_entity_ids=torch.tensor([[0, 1]]),
    )
    tokens = torch.randn((1, 2, 2, 4), requires_grad=True)
    output = adapter(
        tokens,
        object_valid_mask=torch.ones((1, 2)),
    )
    output[..., 0].square().sum().backward()

    entity_parameters = {
        name: parameter
        for name, parameter in adapter.named_parameters()
        if name.startswith("entity_")
    }
    for name, parameter in entity_parameters.items():
        assert torch.isfinite(parameter).all(), f"non-finite parameter before step: {name}"
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
    assert float(adapter.entity_text_up.weight.grad.abs().sum().item()) > 0.0

    optimizer = torch.optim.AdamW(entity_parameters.values(), lr=1.0e-3)
    optimizer.step()
    for name, parameter in entity_parameters.items():
        assert torch.isfinite(parameter).all(), f"non-finite parameter after step: {name}"
    assert float(adapter.entity_text_up.weight.abs().sum().item()) > 0.0
