from __future__ import annotations

import numpy as np
import torch
from torch import nn

from code_vjepa_vggt.models.object_condition_adapter import ObjectConditionAdapter
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    DetectedObjectTrack,
    dedupe_object_tracks,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_context_only_no_gt_box_v_newtrain_kubric import (
    ContextOnlyNoGTBoxWanModule,
    compact_object_context_valid_slots,
)


def _module_stub() -> ContextOnlyNoGTBoxWanModule:
    module = ContextOnlyNoGTBoxWanModule.__new__(ContextOnlyNoGTBoxWanModule)
    module.aux_max_objects = 4
    module.object_slot_dropout_prob = 1.0
    module.full_slot_loss_weight = 1.5
    return module


def test_slot_dropout_keeps_nonempty_subset() -> None:
    module = _module_stub()
    mask = torch.ones((1, 4), dtype=torch.float32)
    for seed in range(16):
        torch.manual_seed(seed)
        sampled, metrics = module._apply_object_slot_dropout(mask)
        count = int(sampled.sum().item())
        assert 1 <= count <= 3
        assert metrics["train/object_count_before_dropout"] == 4.0
        assert metrics["train/object_slot_dropout_applied"] == 1.0
        assert metrics["train/object_full_slot_sample"] == 0.0
        assert metrics["train/object_main_loss_weight"] == 1.0


def test_full_slot_sample_gets_weight_without_dropout() -> None:
    module = _module_stub()
    module.object_slot_dropout_prob = 0.0
    mask = torch.ones((1, 4), dtype=torch.float32)
    sampled, metrics = module._apply_object_slot_dropout(mask)
    assert torch.equal(sampled, mask)
    assert metrics["train/object_full_slot_sample"] == 1.0
    assert metrics["train/object_main_loss_weight"] == 1.5


def test_gate_regularizer_is_active_below_old_target() -> None:
    module = _module_stub()
    module.object_gate_reg_target = 0.08
    block = nn.Module()
    block.object_gate = nn.Parameter(torch.full((1,), 0.10))
    pipe = type(
        "Pipe",
        (),
        {
            "dit": type("DiT", (), {"blocks": [block]})(),
            "device": torch.device("cpu"),
            "torch_dtype": torch.float32,
        },
    )()
    loss, metrics = module._compute_object_gate_regularizer(pipe)
    loss.backward()
    assert float(loss) > 0.0
    assert block.object_gate.grad is not None and float(block.object_gate.grad.abs().sum()) > 0.0
    assert metrics["train/object_gate_tanh_abs_max"] > 0.08


def test_adapter_mlp_cap_and_regularizer_keep_gradients() -> None:
    torch.manual_seed(0)
    adapter = ObjectConditionAdapter(dim=16, num_slots=4, max_time_steps=4)
    adapter.mlp_residual_max_ratio = 2.0
    with torch.no_grad():
        for layer in adapter.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.mul_(20.0)
    tokens = torch.randn((1, 2, 4, 16), requires_grad=True)
    output = adapter(tokens, object_valid_mask=torch.ones((1, 4)))
    ratio, diagnostics = adapter.pop_mlp_diagnostics()
    assert ratio is not None
    assert diagnostics["max_ratio"] > 2.0
    assert diagnostics["cap_applied_fraction"] > 0.0
    assert diagnostics["cap_scale_min"] < 1.0
    loss = output.square().mean() + torch.relu(ratio - 1.0).square().mean()
    loss.backward()
    assert tokens.grad is not None and float(tokens.grad.abs().sum()) > 0.0
    assert adapter.mlp[0].weight.grad is not None
    assert float(adapter.mlp[0].weight.grad.abs().sum()) > 0.0


def test_adapter_mlp_diagnostics_filter_invalid_slots() -> None:
    torch.manual_seed(1)
    adapter = ObjectConditionAdapter(dim=8, num_slots=4, max_time_steps=2)
    tokens = torch.randn((1, 1, 4, 8))
    adapter(tokens, object_valid_mask=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    ratio, diagnostics = adapter.pop_mlp_diagnostics(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    assert ratio is not None
    assert ratio.numel() == 1
    assert diagnostics["mean_ratio"] == float(ratio.detach().float().mean().item())


def test_compact_object_context_physically_removes_invalid_slots() -> None:
    context = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).view(1, 8, 3)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    compacted = compact_object_context_valid_slots(context, mask)
    assert compacted is not None
    expected = context.view(1, 2, 4, 3)[:, :, [0, 2], :].reshape(1, 4, 3)
    assert torch.equal(compacted, expected)


def test_compact_object_context_returns_none_without_valid_slots() -> None:
    context = torch.zeros((1, 8, 3), dtype=torch.float32)
    mask = torch.zeros((1, 4), dtype=torch.float32)
    assert compact_object_context_valid_slots(context, mask) is None


def test_grounding_dedupe_removes_same_phrase_nested_boxes() -> None:
    masks = np.ones((2, 16, 16), dtype=np.uint8)
    large = DetectedObjectTrack(
        box_prompt_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
        masks_thw=masks,
        boxes_t4=np.array([[10, 10, 50, 50], [11, 10, 51, 50]], dtype=np.float32),
        score=0.45,
        phrase="ball",
    )
    nested = DetectedObjectTrack(
        box_prompt_xyxy=np.array([18, 18, 43, 43], dtype=np.float32),
        masks_thw=masks,
        boxes_t4=np.array([[18, 18, 43, 43], [19, 18, 44, 43]], dtype=np.float32),
        score=0.40,
        phrase="ball",
    )
    kept = dedupe_object_tracks([large, nested], iou_threshold=0.75)
    assert len(kept) == 1


def test_grounding_dedupe_keeps_nested_boxes_with_different_phrases() -> None:
    masks = np.ones((2, 16, 16), dtype=np.uint8)
    outer = DetectedObjectTrack(
        box_prompt_xyxy=np.array([10, 10, 50, 50], dtype=np.float32),
        masks_thw=masks,
        boxes_t4=np.array([[10, 10, 50, 50], [10, 10, 50, 50]], dtype=np.float32),
        score=0.45,
        phrase="gripper",
    )
    inner = DetectedObjectTrack(
        box_prompt_xyxy=np.array([18, 18, 43, 43], dtype=np.float32),
        masks_thw=masks,
        boxes_t4=np.array([[18, 18, 43, 43], [18, 18, 43, 43]], dtype=np.float32),
        score=0.40,
        phrase="ball",
    )
    kept = dedupe_object_tracks([outer, inner], iou_threshold=0.75)
    assert len(kept) == 2
