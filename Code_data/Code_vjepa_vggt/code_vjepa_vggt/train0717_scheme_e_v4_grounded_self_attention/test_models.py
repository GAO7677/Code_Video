from __future__ import annotations

import torch
import torch.nn as nn

from code_vjepa_vggt.train0717_scheme_e_v4_grounded_self_attention.models import (
    GroundedObjectCondition,
    TrainableGroupedGroundedAttention,
    install_grouped_grounded_self_attention,
)


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
        self.object_gate = nn.Parameter(torch.zeros(1))

    @staticmethod
    def gate(x, gate, delta):
        return x + delta


class _Dit(nn.Module):
    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(3)])
        self.object_embedding = nn.Linear(dim, dim)


def _condition(*, object_scale: float = 1.0) -> GroundedObjectCondition:
    masks = torch.zeros(1, 6, 2)
    masks[:, :2, 0] = 1.0
    masks[:, 2:4, 1] = 1.0
    known = torch.tensor([[True, True, True, True, False, False]])
    targets = torch.tensor([[0, 0, 1, 1, -100, -100]])
    return GroundedObjectCondition(
        content_delta=object_scale * torch.randn(1, 2, 4, 8),
        valid_mask=torch.tensor([[True, True]]),
        evidence_confidence=torch.ones(1, 2),
        noun_features=torch.randn(1, 2, 12),
        noun_matched_mask=torch.tensor([[True, True]]),
        spatial_bias=(2.0 * masks - 1.0) * 0.5 * known[:, :, None],
        known_token_mask=known,
        assignment_targets=targets,
    )


def test_grounded_block_shape_assignment_loss_and_gradients() -> None:
    torch.manual_seed(7)
    dit = _Dit()
    report = install_grouped_grounded_self_attention(
        dit,
        (1,),
        object_dim=8,
        text_dim=12,
        inner_dim=8,
        gate_init=0.01,
        assignment_loss_weight=0.1,
        spatial_bias_dropout_p=0.0,
    )
    assert report["active_block_ids"] == [1]
    block = dit.blocks[1]
    assert isinstance(block.object_cross_attn, TrainableGroupedGroundedAttention)
    video = torch.randn(1, 6, 16, requires_grad=True)
    output = block(
        video,
        torch.randn(1, 3, 16),
        torch.zeros(1, 6, 16),
        None,
        _condition(),
    )
    assignment_loss = block.object_cross_attn.pop_assignment_loss()
    assert output.shape == video.shape
    assert assignment_loss is not None and torch.isfinite(assignment_loss)
    (output.square().mean() + assignment_loss).backward()
    assert block.object_cross_attn.video_q.weight.grad is not None
    assert block.object_cross_attn.object_v.weight.grad is not None
    assert block.object_cross_attn.noun_k.weight.grad is not None


def test_zero_content_strictly_preserves_native_block() -> None:
    torch.manual_seed(11)
    dit = _Dit()
    install_grouped_grounded_self_attention(
        dit,
        (1,),
        object_dim=8,
        text_dim=12,
        inner_dim=8,
        gate_init=0.01,
        spatial_bias_dropout_p=0.0,
    )
    block = dit.blocks[1]
    x = torch.randn(1, 6, 16)
    condition = _condition(object_scale=0.0)
    output = block(x, torch.randn(1, 3, 16), torch.zeros(1, 6, 16), None, condition)
    expected = x + 0.1 * block.norm1(x)
    torch.testing.assert_close(output, expected, rtol=1.0e-6, atol=1.0e-6)
