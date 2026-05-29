from __future__ import annotations

from typing import Dict

from .config import AdapterConfig
from .schemas import STATE_DIM
from .utils import require_torch

torch = require_torch()
nn = torch.nn


class StateCrossAttentionAdapter(nn.Module):
    def __init__(self, latent_dim: int, memory_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.memory_proj = nn.LazyLinear(memory_dim)
        self.query_proj = nn.Linear(latent_dim, memory_dim)
        self.attn = nn.MultiheadAttention(memory_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(memory_dim, latent_dim)

    def forward(self, latent_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.memory_proj(memory_tokens)
        query = self.query_proj(latent_tokens)
        attended, _ = self.attn(query, memory, memory, need_weights=False)
        return self.out_proj(attended)


class TinyVideoBackbone(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.context_encoder = nn.Sequential(
            nn.LazyConv2d(config.latent_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.latent_dim // 2, config.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.cond_encoder = nn.Sequential(
            nn.LazyConv2d(config.latent_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.latent_dim // 2, config.latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.latent_dim, config.latent_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(config.latent_dim // 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.adapter = StateCrossAttentionAdapter(
            latent_dim=config.latent_dim,
            memory_dim=config.memory_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, STATE_DIM),
        )

        if config.freeze_backbone:
            for module in [self.context_encoder, self.cond_encoder, self.decoder]:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(
        self,
        context_frames: torch.Tensor,
        cond_maps: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, _, _, height, width = context_frames.shape
        future_steps = cond_maps.shape[1]
        context_summary = context_frames.mean(dim=1)
        context_latent = self.context_encoder(context_summary)
        latent_h, latent_w = context_latent.shape[-2:]

        output_frames = []
        state_logits = []
        for step in range(future_steps):
            cond_latent = self.cond_encoder(cond_maps[:, step])
            fused = context_latent + cond_latent
            tokens = fused.flatten(2).transpose(1, 2)
            tokens = tokens + self.adapter(tokens, memory_tokens)
            pooled = tokens.mean(dim=1)
            state_logits.append(self.state_head(pooled))
            fused = tokens.transpose(1, 2).reshape(batch, self.config.latent_dim, latent_h, latent_w)
            frame = self.decoder(fused)
            if frame.shape[-2:] != (height, width):
                frame = torch.nn.functional.interpolate(frame, size=(height, width), mode="bilinear", align_corners=False)
            output_frames.append(frame)

        return {
            "frames": torch.stack(output_frames, dim=1),
            "state_logits": torch.stack(state_logits, dim=1),
        }


def adapter_loss(
    predicted_frames: torch.Tensor,
    target_frames: torch.Tensor,
    predicted_state_logits: torch.Tensor,
    target_states: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    recon = torch.mean(torch.abs(predicted_frames - target_frames))
    pooled_target = target_states.mean(dim=2)
    state_aux = torch.mean((predicted_state_logits - pooled_target) ** 2)
    total = recon + 0.1 * state_aux
    return {"loss": total, "recon": recon, "state_aux": state_aux}
