from __future__ import annotations

import torch
from torch import nn

from code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_context_only_no_gt_box_v_newtrain_kubric import (
    ContextOnlyNoGTBoxWanModule,
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
