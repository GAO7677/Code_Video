from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsynth.models.wan_video_dit import modulate

from code_vjepa_vggt.context_wan_v_newtrain import (
    ObjectBranchInstabilityError,
    _tensor_l2_rms_stats,
    _tensor_numeric_stats,
)
from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler.models import (
    parse_block_ids,
    prune_object_cross_attention_blocks,
)


class MaskedBottleneckObjectJointAttention(nn.Module):
    """Block-sparse joint attention with no added video-to-video path.

    Object queries first read ``[video; object]``. Video queries then read only
    the updated object memory. This is equivalent to a masked joint-attention
    pattern, but avoids materializing the prohibited ``N x N`` video block.
    """

    def __init__(
        self,
        *,
        video_dim: int,
        object_dim: int,
        inner_dim: int = 256,
        num_heads: int = 8,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError("joint attention inner_dim must be divisible by num_heads")
        self.video_dim = int(video_dim)
        self.object_dim = int(object_dim)
        self.inner_dim = int(inner_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.inner_dim // self.num_heads
        self.eps = float(eps)

        self.video_in = nn.Linear(self.video_dim, self.inner_dim, bias=False)
        self.object_in = nn.Linear(self.object_dim, self.inner_dim, bias=False)
        self.video_norm = nn.LayerNorm(self.inner_dim, eps=self.eps)
        self.object_norm = nn.LayerNorm(self.inner_dim, eps=self.eps)
        self.object_update_norm = nn.LayerNorm(self.inner_dim, eps=self.eps)
        self.modality_embed = nn.Parameter(torch.zeros(2, self.inner_dim))
        self.q = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.k = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.v = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.video_out = nn.Linear(self.inner_dim, self.video_dim, bias=False)
        self._last_trace: dict[str, object] | None = None

        for module in (
            self.video_in,
            self.object_in,
            self.q,
            self.k,
            self.v,
            self.o,
            self.video_out,
        ):
            nn.init.xavier_uniform_(module.weight)
        nn.init.normal_(self.modality_embed, std=self.inner_dim**-0.5)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _rms_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = tensor.float().square().mean(dim=-1, keepdim=True)
        return tensor * torch.rsqrt(scale + self.eps).to(dtype=tensor.dtype)

    def _attend(
        self,
        queries: torch.Tensor,
        key_values: torch.Tensor,
    ) -> torch.Tensor:
        q = self._rms_normalize(self._split_heads(self.q(queries)))
        k = self._rms_normalize(self._split_heads(self.k(key_values)))
        v = self._split_heads(self.v(key_values))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().flatten(start_dim=2)
        return self.o(attended)

    def forward(self, video: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
        if video.ndim != 3 or int(video.shape[-1]) != self.video_dim:
            raise ValueError(
                f"video must be [B,N,{self.video_dim}], got {list(video.shape)}"
            )
        if objects.ndim != 3 or int(objects.shape[-1]) != self.object_dim:
            raise ValueError(
                f"objects must be [B,M,{self.object_dim}], got {list(objects.shape)}"
            )
        if int(video.shape[0]) != int(objects.shape[0]):
            raise ValueError("video and object batch dimensions differ")

        object_present = objects.detach().abs().amax(dim=(1, 2)) > 0
        active_batch_items = int(object_present.sum().item())
        if active_batch_items == 0:
            output = torch.zeros_like(video)
            self._last_trace = {
                "video_shape": list(video.shape),
                "object_shape": list(objects.shape),
                "joint_shape": [
                    int(video.shape[0]),
                    int(video.shape[1]) + int(objects.shape[1]),
                    self.inner_dim,
                ],
                "output_shape": list(output.shape),
                "active_batch_items": 0,
                "object_update_attention_pairs": 0,
                "video_read_attention_pairs": 0,
                "prohibited_video_video_attention_pairs": 0,
            }
            return output

        video_tokens = self.video_norm(self.video_in(video))
        object_tokens = self.object_norm(self.object_in(objects))
        video_tokens = video_tokens + self.modality_embed[0].view(1, 1, -1)
        object_tokens = object_tokens + self.modality_embed[1].view(1, 1, -1)
        joint = torch.cat([video_tokens, object_tokens], dim=1)

        object_update = self._attend(object_tokens, joint)
        updated_objects = self.object_update_norm(object_tokens + object_update)
        video_from_objects = self._attend(video_tokens, updated_objects)
        video_delta = self.video_out(video_from_objects)

        # Full object dropout supplies zero memory. Suppress the complete joint
        # adapter in that case so it cannot become an object-independent video
        # branch. Detaching only the binary presence decision preserves normal
        # gradients for non-zero object contexts.
        video_delta = video_delta * object_present[:, None, None].to(video_delta.dtype)
        video_tokens_count = int(video.shape[1])
        object_tokens_count = int(objects.shape[1])
        self._last_trace = {
            "video_shape": list(video.shape),
            "object_shape": list(objects.shape),
            "joint_shape": list(joint.shape),
            "output_shape": list(video_delta.shape),
            "active_batch_items": active_batch_items,
            "object_update_attention_pairs": object_tokens_count
            * (video_tokens_count + object_tokens_count),
            "video_read_attention_pairs": video_tokens_count * object_tokens_count,
            "prohibited_video_video_attention_pairs": 0,
        }
        return video_delta

    def pop_trace(self) -> dict[str, object] | None:
        trace = self._last_trace
        self._last_trace = None
        return trace


def _add_gated_object_residual(
    dit: nn.Module,
    block: nn.Module,
    x: torch.Tensor,
    object_delta: torch.Tensor,
) -> torch.Tensor:
    block_id = int(getattr(block, "_codex_object_block_id", -1))
    object_gate_tanh = torch.tanh(block.object_gate).to(dtype=object_delta.dtype)
    gated_object_delta = object_gate_tanh * object_delta
    residual_scale = float(getattr(dit, "_object_branch_residual_scale", 1.0))
    gated_object_delta = gated_object_delta * residual_scale

    guard_max_ratio = getattr(dit, "_object_branch_ratio_guard_max_ratio", None)
    guard_max_block_id = getattr(dit, "_object_branch_ratio_guard_max_block_id", None)
    guard_enabled = (
        guard_max_ratio is not None
        and float(guard_max_ratio) > 0.0
        and block_id >= 0
        and (
            guard_max_block_id is None
            or int(guard_max_block_id) < 0
            or block_id <= int(guard_max_block_id)
        )
    )
    trace_enabled = bool(getattr(dit, "_object_branch_trace_collect", False))
    guard_applied = False
    guard_scale = 1.0
    x_stats = object_delta_stats = gated_stats = None
    pre_guard_ratio = None
    if guard_enabled or trace_enabled:
        x_stats = _tensor_l2_rms_stats(x)
        object_delta_stats = _tensor_l2_rms_stats(object_delta)
        gated_stats = _tensor_l2_rms_stats(gated_object_delta)
        pre_guard_ratio = float(
            float(gated_stats["l2"] or 0.0) / max(float(x_stats["l2"] or 0.0), 1.0e-12)
        )
        if guard_enabled and pre_guard_ratio > float(guard_max_ratio):
            guard_scale = float(float(guard_max_ratio) / max(pre_guard_ratio, 1.0e-12))
            gated_object_delta = gated_object_delta * guard_scale
            guard_applied = True
            abort_after = getattr(dit, "_object_branch_guard_abort_after_count", None)
            if abort_after is not None and int(abort_after) > 0:
                count = int(getattr(dit, "_object_branch_guard_abort_count", 0)) + 1
                dit._object_branch_guard_abort_count = count
                if count >= int(abort_after):
                    raise ObjectBranchInstabilityError(
                        "object joint self-attention guard triggered repeatedly: "
                        f"count={count} block={block_id} "
                        f"pre_guard_ratio={pre_guard_ratio:.6f}"
                    )

    if trace_enabled:
        trace_buffer = getattr(dit, "_object_branch_trace_buffer", None)
        if isinstance(trace_buffer, list):
            final_stats = _tensor_l2_rms_stats(gated_object_delta)
            trace_buffer.append(
                {
                    "block_id": block_id,
                    "injection_type": "gated_masked_object_joint_attention",
                    "injection_position": "after_wan_self_attention_before_text_cross_attention",
                    "x_before_object": x_stats,
                    "object_delta": object_delta_stats,
                    "pre_guard_gated_object_delta": gated_stats,
                    "pre_guard_gated_to_x_ratio_l2": pre_guard_ratio,
                    "gated_object_delta": final_stats,
                    "gated_to_x_ratio_l2": float(
                        float(final_stats["l2"] or 0.0)
                        / max(float((x_stats or {}).get("l2") or 0.0), 1.0e-12)
                    ),
                    "object_gate_raw": _tensor_numeric_stats(block.object_gate),
                    "object_gate_tanh": _tensor_numeric_stats(object_gate_tanh),
                    "object_branch_residual_scale": residual_scale,
                    "object_ratio_guard": {
                        "enabled": bool(guard_enabled),
                        "applied": bool(guard_applied),
                        "max_ratio": None if guard_max_ratio is None else float(guard_max_ratio),
                        "max_block_id": (
                            None if guard_max_block_id is None else int(guard_max_block_id)
                        ),
                        "scale": float(guard_scale),
                    },
                }
            )
    return x + gated_object_delta


def install_joint_self_attention_forward(dit: nn.Module) -> None:
    """Place the object adapter between native self- and text cross-attention."""

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
        if object_context is not None and getattr(self, "object_cross_attn", None) is not None:
            object_delta = self.object_cross_attn(self.norm4(x), object_context)
            x = _add_gated_object_residual(dit, self, x, object_delta)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x

    for block_id, block in enumerate(dit.blocks):
        block._codex_object_block_id = int(block_id)
        block.forward = types.MethodType(block_forward, block)


def install_bottleneck_object_joint_self_attention(
    dit: nn.Module,
    active_block_ids: tuple[int, ...],
    *,
    object_dim: int,
    inner_dim: int = 256,
    num_heads: int = 8,
    gate_init: float = 0.0,
) -> dict[str, int | float | list[int] | str]:
    """Replace Scheme-D cross-attention with gated masked joint attention."""
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
        block.object_cross_attn = MaskedBottleneckObjectJointAttention(
            video_dim=video_dim,
            object_dim=int(object_dim),
            inner_dim=int(inner_dim),
            num_heads=int(num_heads),
        ).to(device=old_param.device, dtype=torch.float32)
        block.object_gate = nn.Parameter(
            torch.full((1,), float(gate_init), device=old_param.device, dtype=torch.float32)
        )
    dit.object_embedding = None
    install_joint_self_attention_forward(dit)
    return {
        **layout,
        "injection_type": "gated_masked_object_joint_attention",
        "injection_position": "after_wan_self_attention_before_text_cross_attention",
        "object_dim": int(object_dim),
        "attention_inner_dim": int(inner_dim),
        "attention_heads": int(num_heads),
        "gate_init": float(gate_init),
        "attention_mask_policy": "object_reads_video_and_object_then_video_reads_object_only",
    }


# Keep the old import name for lightweight tooling while checkpoints use the
# v3-only object_update_norm key to reject the previous unrestricted adapter.
BottleneckObjectJointSelfAttention = MaskedBottleneckObjectJointAttention
