#!/usr/bin/env python3
"""Minimal grouped grounding attention prototype for a future Scheme-E v4.

The module is called after Wan's native video self-attention and before its
text cross-attention. A shared video-to-object routing map reads grouped object
values, while object tokens never read video tokens. This preserves the
pretrained native self-attention and removes the Scheme-E v3
video -> object -> video bypass.

The caller must provide content deltas from the object resampler, not raw
learned-query outputs. For classifier-free guidance, conditional and
unconditional branches should receive the same visual content deltas; noun
features are supplied only to the conditional branch.

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
    evidence_gate: torch.Tensor
    residual_to_hidden_rms: torch.Tensor
    residual_ratio_mean: torch.Tensor
    residual_ratio_p95: torch.Tensor
    residual_ratio_max: torch.Tensor
    context_residual_to_hidden_rms: torch.Tensor | None
    future_residual_to_hidden_rms: torch.Tensor | None
    per_object_assignment_mass: torch.Tensor
    background_assignment_mass: torch.Tensor
    assignment_entropy_mean: torch.Tensor
    content_logit_std: torch.Tensor
    spatial_bias_std: torch.Tensor | None


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
    strength: float = 0.5,
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


def assignment_nll_loss(
    assignment: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over the shared object/background assignment map."""
    if assignment.ndim != 3:
        raise ValueError("assignment must be [B,N,O+1]")
    if tuple(targets.shape) != tuple(assignment.shape[:2]):
        raise ValueError("targets must be [B,N]")
    log_probs = assignment.float().clamp_min(1.0e-8).log()
    flat_targets = targets.long().flatten()
    supervised = flat_targets != int(ignore_index)
    if not bool(supervised.any()):
        return log_probs.sum() * 0.0
    return F.nll_loss(
        log_probs.flatten(0, 1),
        flat_targets,
        ignore_index=int(ignore_index),
    )


class GroupedGroundedSelfAttention(nn.Module):
    """Shared video-to-object routing placed beside Wan's native self-attention.

    Object tokens remain grouped as [B,O,K,D_object]. A fixed zero background
    logit participates in assignment normalization but has no value, so it
    cannot become an unconditional generation path. The routing assignment is
    shared across the projected value channels, which makes its supervision
    directly identifiable instead of hiding collapsed attention heads.
    """

    def __init__(
        self,
        *,
        video_dim: int,
        object_dim: int,
        text_dim: int,
        inner_dim: int = 256,
        residual_gate_init: float = 0.01,
        noun_key_gate_init: float = 0.1,
        evidence_rms_reference: float = 0.01,
        evidence_active_threshold: float = 1.0e-3,
        spatial_bias_dropout_p: float = 0.25,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if evidence_rms_reference <= 0:
            raise ValueError("evidence_rms_reference must be positive")
        if not 0 <= evidence_active_threshold < 1:
            raise ValueError("evidence_active_threshold must be in [0,1)")
        if not 0 <= spatial_bias_dropout_p <= 1:
            raise ValueError("spatial_bias_dropout_p must be in [0,1]")
        self.video_dim = int(video_dim)
        self.object_dim = int(object_dim)
        self.text_dim = int(text_dim)
        self.inner_dim = int(inner_dim)
        self.evidence_rms_reference = float(evidence_rms_reference)
        self.evidence_active_threshold = float(evidence_active_threshold)
        self.spatial_bias_dropout_p = float(spatial_bias_dropout_p)
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

    def _rms_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        rms_squared = tensor.float().square().mean(dim=-1, keepdim=True)
        return tensor * torch.rsqrt(rms_squared + self.eps).to(tensor.dtype)

    def _drop_spatial_bias(self, spatial_bias: torch.Tensor) -> torch.Tensor:
        if not self.training or self.spatial_bias_dropout_p == 0:
            return spatial_bias
        if self.spatial_bias_dropout_p == 1:
            return torch.zeros_like(spatial_bias)
        keep = torch.rand(
            spatial_bias.shape[0],
            1,
            spatial_bias.shape[2],
            device=spatial_bias.device,
        ) >= self.spatial_bias_dropout_p
        return spatial_bias * keep.to(spatial_bias.dtype)

    def _masked_rms_ratio(
        self,
        residual: torch.Tensor,
        hidden: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        selected_tokens = int(token_mask.sum().item())
        if selected_tokens == 0:
            return None
        mask = token_mask[:, :, None].to(device=residual.device, dtype=torch.float32)
        denominator = float(selected_tokens * residual.shape[-1])
        residual_rms = (
            (residual.float().square() * mask).sum() / denominator
        ).sqrt()
        hidden_rms = (
            (hidden.float().square() * mask).sum() / denominator
        ).sqrt().clamp_min(self.eps)
        return residual_rms / hidden_rms

    def forward(
        self,
        video_hidden: torch.Tensor,
        object_content_delta: torch.Tensor,
        object_valid_mask: torch.Tensor,
        *,
        object_evidence_confidence: torch.Tensor | None = None,
        noun_features: torch.Tensor | None = None,
        noun_matched_mask: torch.Tensor | None = None,
        spatial_bias: torch.Tensor | None = None,
        known_token_mask: torch.Tensor | None = None,
    ) -> GroundedAttentionOutput:
        """Inject content-grounded object residuals into post-native-SA hidden.

        ``object_content_delta`` must exclude the resampler's learned-query
        baseline. ``object_evidence_confidence`` is an optional detached
        detector/tracker confidence in [0,1].
        """
        if video_hidden.ndim != 3 or int(video_hidden.shape[-1]) != self.video_dim:
            raise ValueError(f"video_hidden must be [B,N,{self.video_dim}]")
        if (
            object_content_delta.ndim != 4
            or int(object_content_delta.shape[-1]) != self.object_dim
        ):
            raise ValueError(
                f"object_content_delta must be [B,O,K,{self.object_dim}]"
            )
        batch, video_tokens, _ = video_hidden.shape
        object_batch, objects, _, _ = object_content_delta.shape
        if object_batch != batch:
            raise ValueError("video and object batch dimensions differ")
        if tuple(object_valid_mask.shape) != (batch, objects):
            raise ValueError("object_valid_mask must be [B,O]")
        if object_evidence_confidence is not None and tuple(
            object_evidence_confidence.shape
        ) != (batch, objects):
            raise ValueError("object_evidence_confidence must be [B,O]")
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
        if known_token_mask is not None and tuple(known_token_mask.shape) != (
            batch,
            video_tokens,
        ):
            raise ValueError("known_token_mask must be [B,N]")

        valid = object_valid_mask.to(device=object_content_delta.device).bool()
        delta_rms = (
            object_content_delta.detach().float().square().mean(dim=(2, 3)).sqrt()
        )
        numerical_evidence = delta_rms / (
            delta_rms + self.evidence_rms_reference
        )
        if object_evidence_confidence is None:
            external_confidence = torch.ones_like(numerical_evidence)
        else:
            external_confidence = object_evidence_confidence.detach().to(
                device=numerical_evidence.device,
                dtype=numerical_evidence.dtype,
            ).clamp(0.0, 1.0)
        evidence_gate = (
            numerical_evidence
            * external_confidence
            * valid.to(numerical_evidence.dtype)
        )
        active = valid & (evidence_gate >= self.evidence_active_threshold)

        # K-token mean pooling is intentional for the first prototype.
        normalized_objects = self.object_norm(object_content_delta)
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
        object_values = object_values * evidence_gate[:, :, None].to(
            object_values.dtype
        )

        queries = self._rms_normalize(self.video_q(self.video_norm(video_hidden)))
        keys = self._rms_normalize(object_keys)
        content_logits = torch.einsum("bnd,bod->bno", queries, keys)
        content_logits = content_logits * (1.0 / math.sqrt(self.inner_dim))
        logits = content_logits + evidence_gate.clamp_min(1.0e-8).log()[:, None]
        applied_spatial_bias = None
        if spatial_bias is not None:
            applied_spatial_bias = self._drop_spatial_bias(spatial_bias).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            logits = logits + applied_spatial_bias
        logits = logits.masked_fill(
            ~active[:, None, :],
            torch.finfo(logits.dtype).min,
        )

        background_logits = torch.zeros(
            batch,
            video_tokens,
            1,
            device=logits.device,
            dtype=logits.dtype,
        )
        assignment = torch.softmax(
            torch.cat([logits, background_logits], dim=-1),
            dim=-1,
        )
        object_assignment = assignment[..., :objects]
        attended = torch.einsum(
            "bno,bod->bnd",
            object_assignment,
            object_values,
        )
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
        token_hidden_rms = (
            video_hidden.float().square().mean(dim=-1).sqrt().clamp_min(self.eps)
        )
        token_residual_rms = gated_residual.float().square().mean(dim=-1).sqrt()
        token_residual_ratio = token_residual_rms / token_hidden_rms

        context_ratio = None
        future_ratio = None
        if known_token_mask is not None:
            known = known_token_mask.to(device=video_hidden.device).bool()
            context_ratio = self._masked_rms_ratio(
                gated_residual,
                video_hidden,
                known,
            )
            future_ratio = self._masked_rms_ratio(
                gated_residual,
                video_hidden,
                ~known,
            )

        active_content_logits = content_logits.masked_select(active[:, None, :])
        if active_content_logits.numel() == 0:
            content_logit_std = content_logits.new_zeros((), dtype=torch.float32)
        else:
            content_logit_std = active_content_logits.float().std(unbiased=False)
        spatial_bias_std = (
            None
            if applied_spatial_bias is None
            else applied_spatial_bias.float().std(unbiased=False)
        )
        assignment_entropy = -(
            assignment.float()
            * assignment.float().clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        return GroundedAttentionOutput(
            hidden_states=video_hidden + gated_residual,
            residual=gated_residual,
            assignment=assignment,
            evidence_gate=evidence_gate,
            residual_to_hidden_rms=residual_rms / hidden_rms,
            residual_ratio_mean=token_residual_ratio.mean(),
            residual_ratio_p95=torch.quantile(
                token_residual_ratio.flatten(),
                0.95,
            ),
            residual_ratio_max=token_residual_ratio.max(),
            context_residual_to_hidden_rms=context_ratio,
            future_residual_to_hidden_rms=future_ratio,
            per_object_assignment_mass=object_assignment.float().mean(dim=1),
            background_assignment_mass=assignment[..., -1].float().mean(dim=1),
            assignment_entropy_mean=assignment_entropy.mean(dim=1),
            content_logit_std=content_logit_std,
            spatial_bias_std=spatial_bias_std,
        )


def _smoke() -> None:
    torch.manual_seed(17)
    module = GroupedGroundedSelfAttention(
        video_dim=32,
        object_dim=16,
        text_dim=24,
        inner_dim=16,
        spatial_bias_dropout_p=0.0,
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
        known_token_mask=known,
    )
    assert output.hidden_states.shape == video.shape
    assert output.assignment.shape == (1, 12, 4)
    torch.testing.assert_close(
        output.assignment.sum(dim=-1),
        torch.ones(1, 12),
    )
    assert torch.count_nonzero(output.assignment[..., 2]) == 0
    assert output.context_residual_to_hidden_rms is not None
    assert output.future_residual_to_hidden_rms is not None
    assert tuple(output.per_object_assignment_mass.shape) == (1, 3)
    assert tuple(output.background_assignment_mass.shape) == (1,)

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
    all_ignored_loss = assignment_nll_loss(
        output.assignment,
        torch.full_like(targets, -100),
    )
    assert float(all_ignored_loss) == 0.0
    assert torch.isfinite(all_ignored_loss)
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

    tiny_delta = module(
        video.detach(),
        torch.full_like(objects.detach(), 1.0e-8),
        valid,
    )
    torch.testing.assert_close(
        tiny_delta.hidden_states,
        video.detach(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(tiny_delta.residual) == 0

    zero_confidence = module(
        video.detach(),
        objects.detach(),
        valid,
        object_evidence_confidence=torch.zeros_like(valid, dtype=torch.float32),
        noun_features=noun_features,
        noun_matched_mask=noun_matched,
    )
    torch.testing.assert_close(
        zero_confidence.hidden_states,
        video.detach(),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(zero_confidence.residual) == 0

    full_bias_dropout = GroupedGroundedSelfAttention(
        video_dim=32,
        object_dim=16,
        text_dim=24,
        inner_dim=16,
        spatial_bias_dropout_p=1.0,
    ).train()
    dropped_bias_output = full_bias_dropout(
        video.detach(),
        objects.detach(),
        valid,
        spatial_bias=spatial_bias,
    )
    assert dropped_bias_output.spatial_bias_std is not None
    assert float(dropped_bias_output.spatial_bias_std) == 0.0

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
            "residual_ratio_p95": float(output.residual_ratio_p95.detach()),
            "context_residual_to_hidden_rms": float(
                output.context_residual_to_hidden_rms.detach()
            ),
            "future_residual_to_hidden_rms": float(
                output.future_residual_to_hidden_rms.detach()
            ),
            "background_assignment_mass": float(
                output.background_assignment_mass.mean().detach()
            ),
            "assignment_entropy_mean": float(
                output.assignment_entropy_mean.mean().detach()
            ),
            "content_logit_std": float(output.content_logit_std.detach()),
            "spatial_bias_std": float(output.spatial_bias_std.detach()),
            "all_invalid_exact_identity": True,
            "text_only_exact_identity": True,
            "tiny_delta_exact_identity": True,
            "zero_confidence_exact_identity": True,
            "full_spatial_bias_dropout": True,
            "zero_gate_exact_identity": True,
        }
    )


if __name__ == "__main__":
    _smoke()
