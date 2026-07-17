#!/usr/bin/env python3
"""Minimal grouped grounding attention prototype for a future Scheme-E v4.

The module is called after Wan's native video self-attention and before its
text cross-attention. Video queries read grouped object keys/values, while
object tokens never read video tokens. This preserves the pretrained native
self-attention and removes the Scheme-E v3 video -> object -> video bypass.

This file intentionally does not patch Wan or define a training entrypoint.
It only implements and smoke-tests the key tensor operations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GroundedAttentionOutput:
    hidden_states: torch.Tensor
    residual: torch.Tensor
    assignment: torch.Tensor
    residual_to_hidden_rms: torch.Tensor


def pool_routed_noun_spans(
    text_context: torch.Tensor,
    noun_spans: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool one hard-routed text span per object.

    Args:
        text_context: [B, L, D_text] frozen T5 token embeddings.
        noun_spans: [B, O, 2] half-open [start, end) token spans.
            A negative start marks an unmatched object.

    Returns:
        Pooled noun features [B, O, D_text] and matched mask [B, O].
    """
    if text_context.ndim != 3:
        raise ValueError("text_context must be [B,L,D_text]")
    if noun_spans.ndim != 3 or int(noun_spans.shape[-1]) != 2:
        raise ValueError("noun_spans must be [B,O,2]")
    if int(text_context.shape[0]) != int(noun_spans.shape[0]):
        raise ValueError("text_context and noun_spans batch dimensions differ")

    batch, text_length, text_dim = text_context.shape
    objects = int(noun_spans.shape[1])
    pooled = text_context.new_zeros(batch, objects, text_dim)
    matched = torch.zeros(
        batch,
        objects,
        dtype=torch.bool,
        device=text_context.device,
    )
    for batch_id in range(batch):
        for object_id in range(objects):
            start, end = (
                int(value)
                for value in noun_spans[batch_id, object_id].detach().cpu().tolist()
            )
            if start < 0:
                continue
            if not 0 <= start < end <= text_length:
                raise ValueError(
                    f"invalid noun span {(start, end)} for text length {text_length}"
                )
            pooled[batch_id, object_id] = text_context[
                batch_id, start:end
            ].mean(dim=0)
            matched[batch_id, object_id] = True
    return pooled, matched


def masks_to_spatial_bias(
    aligned_object_masks: torch.Tensor,
    known_token_mask: torch.Tensor,
    *,
    strength: float = 2.0,
) -> torch.Tensor:
    """Convert aligned context masks to a soft [B,N,O] attention bias.

    Future/unknown tokens receive zero bias. This function expects masks to
    already be resized and flattened to the Wan token grid.
    """
    if aligned_object_masks.ndim != 3:
        raise ValueError("aligned_object_masks must be [B,N,O]")
    if tuple(known_token_mask.shape) != tuple(aligned_object_masks.shape[:2]):
        raise ValueError("known_token_mask must be [B,N]")
    masks = aligned_object_masks.clamp(0.0, 1.0)
    signed_bias = (2.0 * masks - 1.0) * float(strength)
    return signed_bias * known_token_mask[:, :, None].to(signed_bias.dtype)


def masks_to_assignment_targets(
    aligned_object_masks: torch.Tensor,
    object_valid_mask: torch.Tensor,
    *,
    known_token_mask: torch.Tensor | None = None,
    foreground_threshold: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Create object/background labels for assignment supervision.

    The background class index is O.
    """
    if aligned_object_masks.ndim != 3:
        raise ValueError("aligned_object_masks must be [B,N,O]")
    batch, _, objects = aligned_object_masks.shape
    if tuple(object_valid_mask.shape) != (batch, objects):
        raise ValueError("object_valid_mask must be [B,O]")

    valid = object_valid_mask[:, None, :].to(dtype=torch.bool)
    masks = aligned_object_masks.clamp(0.0, 1.0).masked_fill(~valid, 0.0)
    confidence, object_ids = masks.max(dim=-1)
    background = torch.full_like(object_ids, objects)
    targets = torch.where(
        confidence >= float(foreground_threshold),
        object_ids,
        background,
    )
    if known_token_mask is not None:
        if tuple(known_token_mask.shape) != tuple(targets.shape):
            raise ValueError("known_token_mask must be [B,N]")
        targets = targets.masked_fill(
            ~known_token_mask.to(device=targets.device).bool(),
            int(ignore_index),
        )
    return targets


def remove_null_object_baseline(
    object_tokens: torch.Tensor,
    null_object_tokens: torch.Tensor,
) -> torch.Tensor:
    """Remove content-independent learned-query output from a token resampler."""
    if tuple(object_tokens.shape) != tuple(null_object_tokens.shape):
        raise ValueError("object_tokens and null_object_tokens must have equal shapes")
    return object_tokens - null_object_tokens


def assignment_nll_loss(
    assignment: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over head-averaged object/background assignments."""
    if assignment.ndim != 3:
        raise ValueError("assignment must be [B,N,O+1]")
    if tuple(targets.shape) != tuple(assignment.shape[:2]):
        raise ValueError("targets must be [B,N]")
    log_probs = assignment.float().clamp_min(1.0e-8).log()
    return F.nll_loss(
        log_probs.flatten(0, 1),
        targets.long().flatten(),
        ignore_index=int(ignore_index),
    )


class GroupedGroundedSelfAttention(nn.Module):
    """Video-to-object attention placed beside Wan's native self-attention.

    Object tokens remain grouped as [B,O,K,D_object]. A fixed zero background
    logit participates in assignment normalization but has no value, so it
    cannot become an unconditional generation path.
    """

    def __init__(
        self,
        *,
        video_dim: int,
        object_dim: int,
        text_dim: int,
        inner_dim: int = 256,
        num_heads: int = 8,
        residual_gate_init: float = 0.01,
        noun_key_gate_init: float = 0.1,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError("inner_dim must be divisible by num_heads")
        self.video_dim = int(video_dim)
        self.object_dim = int(object_dim)
        self.text_dim = int(text_dim)
        self.inner_dim = int(inner_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.inner_dim // self.num_heads
        self.eps = float(eps)

        self.video_norm = nn.LayerNorm(
            self.video_dim,
            eps=self.eps,
            elementwise_affine=False,
        )
        self.object_norm = nn.LayerNorm(
            self.object_dim,
            eps=self.eps,
            elementwise_affine=False,
        )
        self.video_q = nn.Linear(self.video_dim, self.inner_dim, bias=False)
        self.object_k = nn.Linear(self.object_dim, self.inner_dim, bias=False)
        self.object_v = nn.Linear(self.object_dim, self.inner_dim, bias=False)
        self.noun_k = nn.Linear(self.text_dim, self.inner_dim, bias=False)
        self.video_out = nn.Linear(self.inner_dim, self.video_dim, bias=False)
        self.residual_gate = nn.Parameter(
            torch.tensor(float(residual_gate_init), dtype=torch.float32)
        )
        self.noun_key_gate = nn.Parameter(
            torch.tensor(float(noun_key_gate_init), dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.video_q,
            self.object_k,
            self.object_v,
            self.noun_k,
            self.video_out,
        ):
            nn.init.xavier_uniform_(module.weight)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(
            batch,
            tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _rms_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        rms_squared = tensor.float().square().mean(dim=-1, keepdim=True)
        return tensor * torch.rsqrt(rms_squared + self.eps).to(tensor.dtype)

    def forward(
        self,
        video_hidden: torch.Tensor,
        object_tokens: torch.Tensor,
        object_valid_mask: torch.Tensor,
        *,
        noun_features: torch.Tensor | None = None,
        noun_matched_mask: torch.Tensor | None = None,
        spatial_bias: torch.Tensor | None = None,
        null_object_tokens: torch.Tensor | None = None,
    ) -> GroundedAttentionOutput:
        """Inject a grouped object residual into post-native-SA video hidden."""
        if video_hidden.ndim != 3 or int(video_hidden.shape[-1]) != self.video_dim:
            raise ValueError(f"video_hidden must be [B,N,{self.video_dim}]")
        if object_tokens.ndim != 4 or int(object_tokens.shape[-1]) != self.object_dim:
            raise ValueError(f"object_tokens must be [B,O,K,{self.object_dim}]")
        batch, video_tokens, _ = video_hidden.shape
        object_batch, objects, _, _ = object_tokens.shape
        if object_batch != batch:
            raise ValueError("video and object batch dimensions differ")
        if tuple(object_valid_mask.shape) != (batch, objects):
            raise ValueError("object_valid_mask must be [B,O]")
        if (noun_features is None) != (noun_matched_mask is None):
            raise ValueError(
                "noun_features and noun_matched_mask must be provided together"
            )
        if noun_features is not None and tuple(noun_features.shape) != (
            batch,
            objects,
            self.text_dim,
        ):
            raise ValueError(f"noun_features must be [B,O,{self.text_dim}]")
        if noun_matched_mask is not None and tuple(noun_matched_mask.shape) != (
            batch,
            objects,
        ):
            raise ValueError("noun_matched_mask must be [B,O]")
        if spatial_bias is not None and tuple(spatial_bias.shape) != (
            batch,
            video_tokens,
            objects,
        ):
            raise ValueError("spatial_bias must be [B,N,O]")

        if null_object_tokens is not None:
            object_tokens = remove_null_object_baseline(
                object_tokens,
                null_object_tokens.to(
                    device=object_tokens.device,
                    dtype=object_tokens.dtype,
                ),
            )
        valid = object_valid_mask.to(device=object_tokens.device).bool()
        visual_evidence = object_tokens.detach().abs().amax(dim=(2, 3)) > 0
        active = valid & visual_evidence

        # K-token mean pooling is intentional for the first prototype.
        normalized_objects = self.object_norm(object_tokens)
        pooled_objects = normalized_objects.mean(dim=2)
        pooled_objects = pooled_objects * active[:, :, None].to(
            pooled_objects.dtype
        )

        object_keys = self.object_k(pooled_objects)
        if noun_features is not None and noun_matched_mask is not None:
            noun_delta = self.noun_k(
                noun_features.to(
                    device=self.noun_k.weight.device,
                    dtype=self.noun_k.weight.dtype,
                )
            )
            noun_active = active & noun_matched_mask.to(device=active.device).bool()
            noun_delta = noun_delta.to(
                device=object_keys.device,
                dtype=object_keys.dtype,
            )
            noun_delta = noun_delta * noun_active[:, :, None].to(noun_delta.dtype)
            noun_gate = torch.tanh(self.noun_key_gate.float()).to(
                device=object_keys.device,
                dtype=object_keys.dtype,
            )
            object_keys = object_keys + noun_gate * noun_delta

        # Object values contain visual/tube evidence only. Text can change
        # routing keys but cannot independently generate a video residual.
        object_values = self.object_v(pooled_objects)
        object_keys = object_keys * active[:, :, None].to(object_keys.dtype)
        object_values = object_values * active[:, :, None].to(object_values.dtype)

        queries = self._rms_normalize(
            self._split_heads(self.video_q(self.video_norm(video_hidden)))
        )
        keys = self._rms_normalize(self._split_heads(object_keys))
        values = self._split_heads(object_values)
        logits = torch.einsum("bhnd,bhod->bhno", queries, keys)
        logits = logits * (1.0 / math.sqrt(self.head_dim))
        if spatial_bias is not None:
            logits = logits + spatial_bias[:, None].to(
                device=logits.device,
                dtype=logits.dtype,
            )
        logits = logits.masked_fill(
            ~active[:, None, None, :],
            torch.finfo(logits.dtype).min,
        )

        background_logits = torch.zeros(
            batch,
            self.num_heads,
            video_tokens,
            1,
            device=logits.device,
            dtype=logits.dtype,
        )
        assignment_per_head = torch.softmax(
            torch.cat([logits, background_logits], dim=-1),
            dim=-1,
        )
        object_assignment = assignment_per_head[..., :objects]
        attended = torch.einsum(
            "bhno,bhod->bhnd",
            object_assignment,
            values,
        )
        attended = attended.transpose(1, 2).contiguous().flatten(start_dim=2)
        residual = self.video_out(attended)

        any_active = active.any(dim=1)
        residual = residual * any_active[:, None, None].to(residual.dtype)
        gate = torch.tanh(self.residual_gate.float()).to(
            device=residual.device,
            dtype=residual.dtype,
        )
        gated_residual = gate * residual
        hidden_rms = video_hidden.float().square().mean().sqrt().clamp_min(self.eps)
        residual_rms = gated_residual.float().square().mean().sqrt()
        return GroundedAttentionOutput(
            hidden_states=video_hidden + gated_residual,
            residual=gated_residual,
            assignment=assignment_per_head.mean(dim=1),
            residual_to_hidden_rms=residual_rms / hidden_rms,
        )


def _smoke() -> None:
    torch.manual_seed(17)
    module = GroupedGroundedSelfAttention(
        video_dim=32,
        object_dim=16,
        text_dim=24,
        inner_dim=16,
        num_heads=4,
    ).train()
    video = torch.randn(1, 12, 32, requires_grad=True)
    objects = torch.randn(1, 3, 4, 16, requires_grad=True)
    valid = torch.tensor([[True, True, False]])
    text = torch.randn(1, 8, 24)
    spans = torch.tensor([[[1, 3], [4, 6], [-1, -1]]])
    noun_features, noun_matched = pool_routed_noun_spans(text, spans)
    aligned_masks = torch.zeros(1, 12, 3)
    aligned_masks[:, :4, 0] = 1.0
    aligned_masks[:, 4:8, 1] = 1.0
    known = torch.tensor([[True] * 8 + [False] * 4])
    spatial_bias = masks_to_spatial_bias(aligned_masks, known)

    output = module(
        video,
        objects,
        valid,
        noun_features=noun_features,
        noun_matched_mask=noun_matched,
        spatial_bias=spatial_bias,
    )
    assert output.hidden_states.shape == video.shape
    assert output.assignment.shape == (1, 12, 4)
    torch.testing.assert_close(
        output.assignment.sum(dim=-1),
        torch.ones(1, 12),
    )
    assert torch.count_nonzero(output.assignment[..., 2]) == 0

    targets = masks_to_assignment_targets(
        aligned_masks,
        valid,
        known_token_mask=known,
    )
    assert torch.count_nonzero(targets[:, 8:] != -100) == 0
    loss = output.hidden_states.square().mean() + assignment_nll_loss(
        output.assignment,
        targets,
    )
    loss.backward()
    for name in ("video_q", "object_k", "object_v", "noun_k", "video_out"):
        weight = getattr(module, name).weight
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert module.residual_gate.grad is not None
    assert module.noun_key_gate.grad is not None

    all_invalid = module(video.detach(), objects.detach(), torch.zeros_like(valid))
    torch.testing.assert_close(
        all_invalid.hidden_states,
        video.detach(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(all_invalid.residual) == 0

    text_only = module(
        video.detach(),
        torch.zeros_like(objects.detach()),
        valid,
        noun_features=noun_features,
        noun_matched_mask=noun_matched,
    )
    torch.testing.assert_close(
        text_only.hidden_states,
        video.detach(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(text_only.residual) == 0

    null_tokens = torch.randn_like(objects.detach())
    null_centered = module(
        video.detach(),
        null_tokens,
        valid,
        null_object_tokens=null_tokens,
    )
    torch.testing.assert_close(
        null_centered.hidden_states,
        video.detach(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(null_centered.residual) == 0

    with torch.no_grad():
        old_gate = module.residual_gate.clone()
        module.residual_gate.zero_()
        zero_gate = module(video.detach(), objects.detach(), valid)
        module.residual_gate.copy_(old_gate)
    torch.testing.assert_close(
        zero_gate.hidden_states,
        video.detach(),
        rtol=0.0,
        atol=0.0,
    )

    print(
        {
            "video_shape": list(video.shape),
            "object_shape": list(objects.shape),
            "assignment_shape": list(output.assignment.shape),
            "assignment_sum_min": float(output.assignment.sum(dim=-1).min()),
            "assignment_sum_max": float(output.assignment.sum(dim=-1).max()),
            "residual_to_hidden_rms": float(
                output.residual_to_hidden_rms.detach()
            ),
            "all_invalid_exact_identity": True,
            "text_only_exact_identity": True,
            "null_baseline_exact_identity": True,
            "zero_gate_exact_identity": True,
        }
    )


if __name__ == "__main__":
    _smoke()
