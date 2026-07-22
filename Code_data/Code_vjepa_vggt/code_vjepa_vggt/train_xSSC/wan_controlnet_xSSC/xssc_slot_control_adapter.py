"""xSSC slot-assignment ControlNet-style adapter for Wan experiments.

This module is intentionally small and local to ``train_xSSC``.  It does not
modify DiffSynth in place.  The intended hook point is DiffSynth's
``model_fn_wan_video`` after Wan patchifies latents into tokens and before /
inside the DiT block loop, mirroring the existing VACE hint injection pattern.

Core idea:

    latent query tokens [B, T_lat*H*W, C_wan]
    xSSC slots         [B, T_slot, K, 256]
        -> temporal align to [B, T_lat, K, 256]
        -> project slots to C_wan
        -> each latent position softmaxes over K slots at the same latent time
        -> weighted slot vector becomes a dense condition token
        -> zero-init residual projections produce per-layer hints

The adapter trains only its small projections/gates by default.  Wan and xSSC
should remain frozen in the oracle stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentGrid:
    """Patch-token grid after Wan patchification."""

    frames: int
    height: int
    width: int

    @property
    def num_tokens(self) -> int:
        return int(self.frames) * int(self.height) * int(self.width)


def align_slots_to_latent_time(
    slots: torch.Tensor,
    target_frames: int,
    mode: str = "linear",
) -> torch.Tensor:
    """Align frame-level xSSC slots to Wan latent time.

    Args:
        slots: Tensor shaped ``[B, T_slot, K, D]``.
        target_frames: Wan latent temporal length after VAE/patchification.
        mode: ``linear`` uses temporal interpolation, ``window_mean`` averages
            source-frame windows centered on each latent time.

    Returns:
        Tensor shaped ``[B, target_frames, K, D]``.
    """
    if slots.ndim != 4:
        raise ValueError(f"slots must be [B,T,K,D], got {tuple(slots.shape)}")
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if int(slots.shape[1]) == int(target_frames):
        return slots

    if mode == "linear":
        # Interpolate over time independently for every slot/channel.
        x = slots.permute(0, 2, 3, 1).reshape(-1, 1, slots.shape[1])
        x = F.interpolate(x.float(), size=int(target_frames), mode="linear", align_corners=True)
        x = x.to(dtype=slots.dtype).reshape(slots.shape[0], slots.shape[2], slots.shape[3], target_frames)
        return x.permute(0, 3, 1, 2).contiguous()

    if mode == "window_mean":
        bsz, source_frames, num_slots, dim = slots.shape
        edges = torch.linspace(0, source_frames, steps=target_frames + 1, device=slots.device)
        aligned = []
        for i in range(target_frames):
            lo = int(torch.floor(edges[i]).item())
            hi = int(torch.ceil(edges[i + 1]).item())
            hi = max(lo + 1, min(hi, source_frames))
            aligned.append(slots[:, lo:hi].mean(dim=1))
        return torch.stack(aligned, dim=1).reshape(bsz, target_frames, num_slots, dim)

    raise ValueError(f"Unsupported temporal alignment mode: {mode!r}")


class ZeroLinear(nn.Linear):
    """Linear layer initialized as a no-op residual producer."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class XSSCSlotAssignmentControlAdapter(nn.Module):
    """Build block-wise residual hints from xSSC slots and Wan latent queries."""

    def __init__(
        self,
        *,
        slot_dim: int = 256,
        wan_dim: int = 3072,
        num_slots: int = 7,
        control_layers: Sequence[int] = (11, 29),
        temporal_align: str = "linear",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.wan_dim = int(wan_dim)
        self.num_slots = int(num_slots)
        self.control_layers = tuple(int(i) for i in control_layers)
        self.layer_to_hint = {layer: idx for idx, layer in enumerate(self.control_layers)}
        self.temporal_align = str(temporal_align)
        self.temperature = float(temperature)

        self.query_norm = nn.LayerNorm(self.wan_dim)
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.wan_dim),
            nn.GELU(),
            nn.LayerNorm(self.wan_dim),
        )
        self.query_proj = nn.Linear(self.wan_dim, self.wan_dim, bias=False)
        self.key_proj = nn.Linear(self.wan_dim, self.wan_dim, bias=False)
        self.value_proj = nn.Linear(self.wan_dim, self.wan_dim, bias=False)
        self.hint_projs = nn.ModuleList([ZeroLinear(self.wan_dim, self.wan_dim) for _ in self.control_layers])
        self.gates = nn.Parameter(torch.zeros(len(self.control_layers)))

    def forward(
        self,
        *,
        latent_tokens: torch.Tensor,
        latent_grid: LatentGrid,
        xssc_slots: torch.Tensor,
        return_assignment: bool = False,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """Return per-layer hints shaped like Wan hidden tokens.

        Args:
            latent_tokens: Wan hidden tokens ``[B, T_lat*H*W, C_wan]``.
            latent_grid: Token grid metadata.
            xssc_slots: Frozen xSSC slots ``[B, T_slot, K, 256]``.
            return_assignment: Also return slot probabilities
                ``[B,T_lat,H,W,K]`` for visualization/diagnostics.
        """
        if latent_tokens.ndim != 3:
            raise ValueError(f"latent_tokens must be [B,N,C], got {tuple(latent_tokens.shape)}")
        bsz, num_tokens, dim = latent_tokens.shape
        if dim != self.wan_dim:
            raise ValueError(f"latent dim mismatch: expected {self.wan_dim}, got {dim}")
        if num_tokens != latent_grid.num_tokens:
            raise ValueError(f"token/grid mismatch: {num_tokens} vs {latent_grid.num_tokens}")
        if xssc_slots.shape[0] != bsz:
            raise ValueError(f"batch mismatch: tokens B={bsz}, slots B={xssc_slots.shape[0]}")
        if xssc_slots.shape[2] != self.num_slots:
            raise ValueError(f"slot count mismatch: expected {self.num_slots}, got {xssc_slots.shape[2]}")

        slots = align_slots_to_latent_time(
            xssc_slots,
            target_frames=latent_grid.frames,
            mode=self.temporal_align,
        )
        slots = self.slot_proj(slots)

        q = latent_tokens.reshape(bsz, latent_grid.frames, latent_grid.height, latent_grid.width, dim)
        q = self.query_proj(self.query_norm(q))
        k = self.key_proj(slots)
        v = self.value_proj(slots)

        scale = (dim ** -0.5) / max(self.temperature, 1e-6)
        logits = torch.einsum("bthwc,btkc->bthwk", q, k) * scale
        assignment = torch.softmax(logits, dim=-1)
        dense = torch.einsum("bthwk,btkc->bthwc", assignment, v)
        dense = dense.reshape(bsz, num_tokens, dim)

        hints = []
        for proj, gate in zip(self.hint_projs, self.gates):
            hints.append(proj(dense) * gate)
        return hints, assignment if return_assignment else None

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.named_parameters() if param.requires_grad]


def freeze_except_xssc_control_adapter(modules: Iterable[nn.Module]) -> None:
    """Freeze all parameters in the provided modules."""
    for module in modules:
        module.requires_grad_(False)
