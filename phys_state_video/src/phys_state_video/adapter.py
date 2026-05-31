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
        temporal_padding = config.temporal_kernel_size // 2
        self.context_temporal = nn.Sequential(
            nn.Conv3d(
                config.latent_dim,
                config.latent_dim,
                kernel_size=(config.temporal_kernel_size, 1, 1),
                padding=(temporal_padding, 0, 0),
                groups=config.latent_dim,
            ),
            nn.GELU(),
            nn.Conv3d(config.latent_dim, config.latent_dim, kernel_size=1),
        )
        self.context_frame_score = nn.Linear(config.latent_dim, 1)
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
        self.spatial_head = nn.Sequential(
            nn.Conv2d(config.latent_dim, config.latent_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.latent_dim // 2, 2, kernel_size=1),
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

        nn.init.zeros_(self.context_temporal[-1].weight)
        nn.init.zeros_(self.context_temporal[-1].bias)
        nn.init.zeros_(self.context_frame_score.weight)
        nn.init.zeros_(self.context_frame_score.bias)

    def encode_context(self, context_frames: torch.Tensor) -> torch.Tensor:
        batch, context_steps, channels, height, width = context_frames.shape
        frame_latents = self.context_encoder(context_frames.reshape(batch * context_steps, channels, height, width))
        latent_h, latent_w = frame_latents.shape[-2:]
        frame_latents = frame_latents.reshape(batch, context_steps, self.config.latent_dim, latent_h, latent_w)
        frame_latents = frame_latents.permute(0, 2, 1, 3, 4)
        temporal_latents = frame_latents + self.context_temporal(frame_latents)
        frame_descriptors = temporal_latents.mean(dim=(-1, -2)).transpose(1, 2)
        frame_weights = torch.softmax(self.context_frame_score(frame_descriptors), dim=1)
        return torch.sum(
            temporal_latents * frame_weights.transpose(1, 2).unsqueeze(-1).unsqueeze(-1),
            dim=2,
        )

    def forward(
        self,
        context_frames: torch.Tensor,
        cond_maps: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, _, _, height, width = context_frames.shape
        future_steps = cond_maps.shape[1]
        context_latent = self.encode_context(context_frames)
        latent_h, latent_w = context_latent.shape[-2:]

        output_frames = []
        state_logits = []
        spatial_logits = []
        for step in range(future_steps):
            cond_latent = self.cond_encoder(cond_maps[:, step])
            fused = context_latent + cond_latent
            tokens = fused.flatten(2).transpose(1, 2)
            tokens = tokens + self.adapter(tokens, memory_tokens)
            pooled = tokens.mean(dim=1)
            state_logits.append(self.state_head(pooled))
            fused = tokens.transpose(1, 2).reshape(batch, self.config.latent_dim, latent_h, latent_w)
            spatial = self.spatial_head(fused)
            if spatial.shape[-2:] != (height, width):
                spatial = torch.nn.functional.interpolate(
                    spatial,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            spatial_logits.append(spatial)
            frame = self.decoder(fused)
            if frame.shape[-2:] != (height, width):
                frame = torch.nn.functional.interpolate(frame, size=(height, width), mode="bilinear", align_corners=False)
            output_frames.append(frame)

        return {
            "frames": torch.stack(output_frames, dim=1),
            "state_logits": torch.stack(state_logits, dim=1),
            "spatial_logits": torch.stack(spatial_logits, dim=1),
        }


def adapter_loss(
    predicted_frames: torch.Tensor,
    target_frames: torch.Tensor,
    predicted_state_logits: torch.Tensor,
    target_states: torch.Tensor,
    state_loss_weights: torch.Tensor | None = None,
    state_loss_scale: float = 0.1,
    predicted_spatial_logits: torch.Tensor | None = None,
    target_spatial_maps: torch.Tensor | None = None,
    spatial_loss_scale: float = 0.0,
    spatial_foreground_weight: float = 4.0,
) -> Dict[str, torch.Tensor]:
    recon = torch.mean(torch.abs(predicted_frames - target_frames))
    pooled_target = target_states.mean(dim=2)
    state_error = (predicted_state_logits - pooled_target) ** 2
    if state_loss_weights is not None:
        view_shape = [1] * (state_error.ndim - 1) + [state_error.shape[-1]]
        state_error = state_error * state_loss_weights.view(*view_shape)
    state_aux = torch.mean(state_error)
    spatial_aux = torch.zeros((), device=predicted_frames.device, dtype=predicted_frames.dtype)
    if predicted_spatial_logits is not None and target_spatial_maps is not None:
        spatial_prob = torch.sigmoid(predicted_spatial_logits)
        spatial_weight = 1.0 + spatial_foreground_weight * target_spatial_maps
        spatial_aux = torch.mean(((spatial_prob - target_spatial_maps) ** 2) * spatial_weight)
    total = recon + state_loss_scale * state_aux + spatial_loss_scale * spatial_aux
    return {"loss": total, "recon": recon, "state_aux": state_aux, "spatial_aux": spatial_aux}
