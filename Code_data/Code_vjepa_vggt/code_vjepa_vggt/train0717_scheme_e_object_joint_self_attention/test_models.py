from __future__ import annotations

import torch
import torch.nn as nn

from code_vjepa_vggt.train0717_scheme_e_object_joint_self_attention.models import (
    BottleneckObjectJointSelfAttention,
    install_bottleneck_object_joint_self_attention,
)


def test_joint_attention_shape_and_object_gradients() -> None:
    torch.manual_seed(3)
    module = BottleneckObjectJointSelfAttention(
        video_dim=32,
        object_dim=12,
        inner_dim=16,
        num_heads=4,
    ).train()
    video = torch.randn(2, 11, 32, requires_grad=True)
    objects = torch.randn(2, 5, 12, requires_grad=True)
    output = module(video, objects)
    assert output.shape == video.shape
    trace = module.pop_trace()
    assert trace["object_update_attention_pairs"] == 5 * (11 + 5)
    assert trace["video_read_attention_pairs"] == 11 * 5
    assert trace["prohibited_video_video_attention_pairs"] == 0
    output.square().mean().backward()
    assert video.grad is not None and torch.isfinite(video.grad).all()
    assert objects.grad is not None and torch.isfinite(objects.grad).all()


def test_zero_object_context_produces_exact_zero_residual() -> None:
    module = BottleneckObjectJointSelfAttention(
        video_dim=32,
        object_dim=12,
        inner_dim=16,
        num_heads=4,
    ).eval()
    output = module(torch.randn(2, 9, 32), torch.zeros(2, 4, 12))
    assert torch.count_nonzero(output) == 0


def test_object_tokens_change_joint_attention_output() -> None:
    torch.manual_seed(5)
    module = BottleneckObjectJointSelfAttention(
        video_dim=32,
        object_dim=12,
        inner_dim=16,
        num_heads=4,
    ).eval()
    video = torch.randn(1, 9, 32)
    first = module(video, torch.randn(1, 4, 12))
    second = module(video, torch.randn(1, 4, 12))
    assert not torch.equal(first, second)


class _SelfAttention(nn.Module):
    def forward(self, x, freqs):
        return 0.1 * x


class _ZeroModule(nn.Module):
    def forward(self, x, *args):
        return torch.zeros_like(x)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.self_attn = _SelfAttention()
        self.cross_attn = _ZeroModule()
        self.ffn = _ZeroModule()
        self.object_cross_attn = nn.Linear(dim, dim)
        self.object_gate = nn.Parameter(torch.ones(1, 1, dim))

    @staticmethod
    def gate(x, gate, delta):
        return x + delta


class _Dit(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(4)])
        self.object_embedding = nn.Linear(dim, dim)
        self._object_branch_residual_scale = 1.0
        self._object_branch_ratio_guard_max_ratio = None
        self._object_branch_ratio_guard_max_block_id = None
        self._object_branch_trace_collect = False
        self._object_branch_trace_buffer = None


def test_install_prunes_blocks_and_uses_scalar_zero_gate() -> None:
    dit = _Dit()
    report = install_bottleneck_object_joint_self_attention(
        dit,
        (1, 3),
        object_dim=8,
        inner_dim=8,
        num_heads=2,
        gate_init=0.0,
    )
    assert report["active_block_ids"] == [1, 3]
    assert report["injection_type"] == "gated_masked_object_joint_attention"
    for block_id, block in enumerate(dit.blocks):
        if block_id in (1, 3):
            assert isinstance(block.object_cross_attn, BottleneckObjectJointSelfAttention)
            assert block.object_cross_attn.object_update_norm is not None
            assert tuple(block.object_gate.shape) == (1,)
            assert float(block.object_gate.item()) == 0.0
        else:
            assert block.object_cross_attn is None
            assert block.object_gate is None


def test_zero_gate_strictly_preserves_block_output() -> None:
    torch.manual_seed(7)
    dit = _Dit()
    install_bottleneck_object_joint_self_attention(
        dit,
        (1,),
        object_dim=8,
        inner_dim=8,
        num_heads=2,
        gate_init=0.0,
    )
    block = dit.blocks[1]
    x = torch.randn(1, 6, 16)
    context = torch.randn(1, 3, 16)
    t_mod = torch.zeros(1, 6, 16)
    first = block(x, context, t_mod, None, torch.randn(1, 2, 8))
    second = block(x, context, t_mod, None, torch.randn(1, 2, 8))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_nonzero_gate_backpropagates_to_joint_adapter() -> None:
    dit = _Dit()
    install_bottleneck_object_joint_self_attention(
        dit,
        (1,),
        object_dim=8,
        inner_dim=8,
        num_heads=2,
        gate_init=0.1,
    )
    block = dit.blocks[1]
    output = block(
        torch.randn(1, 6, 16),
        torch.randn(1, 3, 16),
        torch.zeros(1, 6, 16),
        None,
        torch.randn(1, 2, 8),
    )
    output.square().mean().backward()
    assert block.object_cross_attn.object_in.weight.grad is not None
    assert torch.isfinite(block.object_cross_attn.object_in.weight.grad).all()
