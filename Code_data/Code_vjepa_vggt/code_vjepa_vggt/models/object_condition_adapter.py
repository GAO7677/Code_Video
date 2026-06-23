from __future__ import annotations

import torch
import torch.nn as nn


class ObjectConditionAdapter(nn.Module):
    def __init__(self, dim: int = 4096, num_slots: int = 8, max_time_steps: int = 64) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.max_time_steps = int(max_time_steps)
        self.slot_embed = nn.Embedding(self.num_slots, self.dim)
        self.time_embed = nn.Embedding(self.max_time_steps, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, object_latent_tokens: torch.Tensor) -> torch.Tensor:
        batch, time_steps, slots, dim = object_latent_tokens.shape
        if int(slots) > self.num_slots:
            raise ValueError(f"slots={slots} exceeds configured num_slots={self.num_slots}")
        if int(time_steps) > self.max_time_steps:
            raise ValueError(f"time_steps={time_steps} exceeds configured max_time_steps={self.max_time_steps}")
        slot_ids = torch.arange(slots, device=object_latent_tokens.device)
        time_ids = torch.arange(time_steps, device=object_latent_tokens.device)
        slot_bias = self.slot_embed(slot_ids).view(1, 1, slots, dim)
        time_bias = self.time_embed(time_ids).view(1, time_steps, 1, dim)
        x = object_latent_tokens + slot_bias + time_bias
        x = self.norm(x)
        x = x + self.mlp(x)
        return x.view(batch, time_steps * slots, dim)
