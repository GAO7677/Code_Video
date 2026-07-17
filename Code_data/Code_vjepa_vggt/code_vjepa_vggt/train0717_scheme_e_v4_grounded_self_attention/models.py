from __future__ import annotations

from dataclasses import dataclass, replace
import types

import torch
import torch.nn as nn

from diffsynth.models.wan_video_dit import modulate

from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler.models import (
    parse_block_ids,
    prune_object_cross_attention_blocks,
)
from code_vjepa_vggt.train0717_scheme_e_v4_grounded_self_attention.prototype_grouped_grounded_self_attention import (
    GroupedGroundedSelfAttention,
    GroundedAttentionOutput,
    assignment_nll_loss,
)


@dataclass(frozen=True)
class GroundedObjectCondition:
    content_delta: torch.Tensor
    valid_mask: torch.Tensor
    evidence_confidence: torch.Tensor
    noun_features: torch.Tensor
    noun_matched_mask: torch.Tensor
    spatial_bias: torch.Tensor
    known_token_mask: torch.Tensor
    assignment_targets: torch.Tensor

    def to(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> GroundedObjectCondition:
        float_dtype = self.content_delta.dtype if dtype is None else dtype
        return replace(
            self,
            content_delta=self.content_delta.to(device=device, dtype=float_dtype),
            valid_mask=self.valid_mask.to(device=device, dtype=torch.bool),
            evidence_confidence=self.evidence_confidence.to(
                device=device, dtype=torch.float32
            ),
            noun_features=self.noun_features.to(device=device, dtype=float_dtype),
            noun_matched_mask=self.noun_matched_mask.to(
                device=device, dtype=torch.bool
            ),
            spatial_bias=self.spatial_bias.to(device=device, dtype=float_dtype),
            known_token_mask=self.known_token_mask.to(device=device, dtype=torch.bool),
            assignment_targets=self.assignment_targets.to(
                device=device, dtype=torch.long
            ),
        )


class TrainableGroupedGroundedAttention(GroupedGroundedSelfAttention):
    def __init__(self, *args, assignment_loss_weight: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.assignment_loss_weight = float(assignment_loss_weight)
        self._last_assignment_loss: torch.Tensor | None = None
        self._last_trace: dict[str, object] | None = None

    @property
    def q(self) -> nn.Linear:
        """Read-only compatibility alias for Scheme-D shape tracing."""
        return self.video_q

    @property
    def k(self) -> nn.Linear:
        """Read-only compatibility alias for Scheme-D shape tracing."""
        return self.object_k

    def forward_condition(
        self,
        video_hidden: torch.Tensor,
        condition: GroundedObjectCondition,
    ) -> GroundedAttentionOutput:
        output = super().forward(
            video_hidden,
            condition.content_delta,
            condition.valid_mask,
            object_evidence_confidence=condition.evidence_confidence,
            noun_features=condition.noun_features,
            noun_matched_mask=condition.noun_matched_mask,
            spatial_bias=condition.spatial_bias,
            known_token_mask=condition.known_token_mask,
        )
        assignment_loss = assignment_nll_loss(
            output.assignment,
            condition.assignment_targets,
        )
        self._last_assignment_loss = self.assignment_loss_weight * assignment_loss
        self._last_trace = {
            "video_shape": list(video_hidden.shape),
            "object_shape": list(condition.content_delta.shape),
            "assignment_shape": list(output.assignment.shape),
            "valid_objects": int(condition.valid_mask.sum().detach().item()),
            "matched_nouns": int(condition.noun_matched_mask.sum().detach().item()),
            "evidence_gate_mean": float(output.evidence_gate.detach().mean().item()),
            "background_assignment_mass": float(
                output.background_assignment_mass.detach().mean().item()
            ),
            "assignment_entropy_mean": float(
                output.assignment_entropy_mean.detach().mean().item()
            ),
            "content_logit_std": float(output.content_logit_std.detach().item()),
            "spatial_bias_std": (
                None
                if output.spatial_bias_std is None
                else float(output.spatial_bias_std.detach().item())
            ),
            "residual_ratio_mean": float(output.residual_ratio_mean.detach().item()),
            "residual_ratio_p95": float(output.residual_ratio_p95.detach().item()),
            "residual_ratio_max": float(output.residual_ratio_max.detach().item()),
            "context_residual_ratio": (
                None
                if output.context_residual_to_hidden_rms is None
                else float(output.context_residual_to_hidden_rms.detach().item())
            ),
            "future_residual_ratio": (
                None
                if output.future_residual_to_hidden_rms is None
                else float(output.future_residual_to_hidden_rms.detach().item())
            ),
        }
        return output

    def pop_assignment_loss(self) -> torch.Tensor | None:
        value = self._last_assignment_loss
        self._last_assignment_loss = None
        return value

    def pop_trace(self) -> dict[str, object] | None:
        value = self._last_trace
        self._last_trace = None
        return value


def install_grounded_self_attention_forward(dit: nn.Module) -> None:
    def block_forward(self, x, context, t_mod, freqs, object_context=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        if (
            isinstance(object_context, GroundedObjectCondition)
            and isinstance(
                getattr(self, "object_cross_attn", None),
                TrainableGroupedGroundedAttention,
            )
        ):
            x = self.object_cross_attn.forward_condition(x, object_context).hidden_states
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return self.gate(x, gate_mlp, self.ffn(input_x))

    for block_id, block in enumerate(dit.blocks):
        block._codex_object_block_id = int(block_id)
        block.forward = types.MethodType(block_forward, block)


def install_grouped_grounded_self_attention(
    dit: nn.Module,
    active_block_ids: tuple[int, ...],
    *,
    object_dim: int,
    text_dim: int,
    inner_dim: int = 256,
    gate_init: float = 0.01,
    noun_key_gate_init: float = 0.1,
    assignment_loss_weight: float = 0.1,
    evidence_rms_reference: float = 0.01,
    evidence_active_threshold: float = 1.0e-3,
    spatial_bias_dropout_p: float = 0.25,
) -> dict[str, int | float | list[int] | str]:
    blocks = list(getattr(dit, "blocks", []))
    active = parse_block_ids(
        ",".join(map(str, active_block_ids)),
        num_blocks=len(blocks),
    )
    layout = prune_object_cross_attention_blocks(dit, active)
    video_dim = int(getattr(dit, "dim"))
    for block_id in active:
        block = blocks[block_id]
        old_module = getattr(block, "object_cross_attn", None)
        old_param = next(old_module.parameters(), None)
        if old_param is None:
            raise RuntimeError(f"object attention missing at block {block_id}")
        block.object_cross_attn = TrainableGroupedGroundedAttention(
            video_dim=video_dim,
            object_dim=int(object_dim),
            text_dim=int(text_dim),
            inner_dim=int(inner_dim),
            residual_gate_init=float(gate_init),
            noun_key_gate_init=float(noun_key_gate_init),
            evidence_rms_reference=float(evidence_rms_reference),
            evidence_active_threshold=float(evidence_active_threshold),
            spatial_bias_dropout_p=float(spatial_bias_dropout_p),
            assignment_loss_weight=float(assignment_loss_weight),
        ).to(device=old_param.device, dtype=torch.float32)
        block.object_gate = None
        block.norm4 = None
    dit.object_embedding = None
    install_grounded_self_attention_forward(dit)
    return {
        **layout,
        "injection_type": "grouped_grounded_self_attention_stage_adapter",
        "injection_position": "after_wan_self_attention_before_text_cross_attention",
        "object_dim": int(object_dim),
        "text_dim": int(text_dim),
        "attention_inner_dim": int(inner_dim),
        "gate_init": float(gate_init),
        "noun_key_gate_init": float(noun_key_gate_init),
        "assignment_loss_weight": float(assignment_loss_weight),
        "routing_policy": "shared_video_to_object_with_fixed_background",
    }
