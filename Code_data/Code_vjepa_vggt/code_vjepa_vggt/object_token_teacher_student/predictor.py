from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FuturePredictorOutput:
    pred_future_tokens: torch.Tensor


class FutureObjectTokenPredictor(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        max_context_latent_frames: int = 8,
        max_future_latent_frames: int = 8,
        num_slots: int = 4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_context_latent_frames = int(max_context_latent_frames)
        self.max_future_latent_frames = int(max_future_latent_frames)
        self.num_slots = int(num_slots)
        self.slot_embed = nn.Embedding(self.num_slots, self.dim)
        self.context_time_embed = nn.Embedding(self.max_context_latent_frames, self.dim)
        self.future_time_embed = nn.Embedding(self.max_future_latent_frames, self.dim)
        self.in_norm = nn.LayerNorm(self.dim)
        self.mixer = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        *,
        future_latent_frames: int,
        object_valid_mask: torch.Tensor | None = None,
    ) -> FuturePredictorOutput:
        if context_tokens.ndim != 4:
            raise ValueError(f"context_tokens must have shape [B,T,O,D], got {list(context_tokens.shape)}")
        batch, context_t, slots, dim = context_tokens.shape
        if dim != self.dim:
            raise ValueError(f"context token dim mismatch: got {dim}, expected {self.dim}")
        if slots > self.num_slots:
            raise ValueError(f"slot count {slots} exceeds configured num_slots={self.num_slots}")
        if context_t > self.max_context_latent_frames:
            raise ValueError(
                f"context latent frames {context_t} exceeds configured max_context_latent_frames={self.max_context_latent_frames}"
            )
        future_latent_frames = int(future_latent_frames)
        if future_latent_frames <= 0 or future_latent_frames > self.max_future_latent_frames:
            raise ValueError(
                f"future_latent_frames must be in [1, {self.max_future_latent_frames}], got {future_latent_frames}"
            )
        x = torch.nan_to_num(context_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        slot_ids = torch.arange(slots, device=x.device)
        time_ids = torch.arange(context_t, device=x.device)
        x = x + self.slot_embed(slot_ids).view(1, 1, slots, self.dim)
        x = x + self.context_time_embed(time_ids).view(1, context_t, 1, self.dim)
        if object_valid_mask is not None:
            x = x * object_valid_mask[:, None, :, None].to(dtype=x.dtype, device=x.device)
        summary = self.in_norm(x).mean(dim=1, keepdim=True)
        future_ids = torch.arange(future_latent_frames, device=x.device)
        future = summary.expand(batch, future_latent_frames, slots, self.dim).contiguous()
        future = future + self.future_time_embed(future_ids).view(1, future_latent_frames, 1, self.dim)
        future = future + self.slot_embed(slot_ids).view(1, 1, slots, self.dim)
        future = future + self.mixer(future)
        future = self.readout(future)
        if object_valid_mask is not None:
            future = future * object_valid_mask[:, None, :, None].to(dtype=future.dtype, device=future.device)
        return FuturePredictorOutput(pred_future_tokens=future)
