#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn

from dit_ablation import DiTAblationSpec, install_dit_ablation


class AddOne(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x + 1


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = AddOne()
        self.object_cross_attn = AddOne()

    def forward(self, x, *args, **kwargs):
        x = x + self.self_attn(x)
        x = x + self.object_cross_attn(x)
        return x + 1


class FakeDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(30)])


def run_block(dit):
    x = torch.ones(1, 2, 3)
    return dit.blocks[7](x, None, None, None)


def main() -> None:
    baseline = run_block(FakeDiT())

    whole = FakeDiT()
    install_dit_ablation(
        whole,
        DiTAblationSpec("whole_block", 7),
    )
    assert torch.equal(run_block(whole), torch.ones(1, 2, 3))

    self_attn = FakeDiT()
    install_dit_ablation(
        self_attn,
        DiTAblationSpec("self_attn", 7),
    )
    self_attn_out = run_block(self_attn)
    assert torch.all(self_attn_out < baseline)

    object_attn = FakeDiT()
    install_dit_ablation(
        object_attn,
        DiTAblationSpec("object_cross_attn", 7),
    )
    object_attn_out = run_block(object_attn)
    assert torch.all(object_attn_out < baseline)

    untouched = FakeDiT()
    metadata = install_dit_ablation(untouched, DiTAblationSpec())
    assert torch.equal(run_block(untouched), baseline)
    assert metadata["disabled_module"] is None
    print("dit_ablation tests passed")


if __name__ == "__main__":
    main()
