from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .schemas import STATE_DIM, StateIndex
from .utils import hash_prompt_tokens, require_torch

torch = require_torch()
nn = torch.nn
F = torch.nn.functional


GEOM_STATE_INDICES = (
    StateIndex.CENTER_X,
    StateIndex.CENTER_Y,
    StateIndex.DEPTH,
    StateIndex.LOG_SCALE,
)
MOTION_STATE_INDICES = (
    StateIndex.VEL_X,
    StateIndex.VEL_Y,
    StateIndex.DEPTH_VEL,
)
VIS_STATE_INDICES = (
    StateIndex.VISIBILITY,
    StateIndex.EXISTENCE,
    StateIndex.CONFIDENCE,
)


@dataclass(slots=True)
class WanStateLatentPredictorV2Config:
    latent_channels: int = 16
    latent_pool_side: int = 2
    camera_dim: int = 8
    prompt_vocab_size: int = 4096
    prompt_embed_dim: int = 64
    hidden_dim: int = 256
    state_latent_dim: int = 128
    max_context_latent_steps: int = 16
    max_future_latent_steps: int = 16
    max_objects: int = 6
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dropout: float = 0.1


def resample_temporal_features(features: torch.Tensor, target_steps: int) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"expected temporal features [B, T, D], got {tuple(features.shape)}")
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if features.shape[1] == target_steps:
        return features
    resized = F.interpolate(
        features.transpose(1, 2),
        size=target_steps,
        mode="linear",
        align_corners=False,
    )
    return resized.transpose(1, 2).contiguous()


def resample_temporal_states(states: torch.Tensor, target_steps: int) -> torch.Tensor:
    if states.ndim != 4:
        raise ValueError(f"expected temporal states [B, T, N, D], got {tuple(states.shape)}")
    batch, _, num_objects, state_dim = states.shape
    flattened = states.permute(0, 2, 3, 1).contiguous().view(batch, num_objects * state_dim, states.shape[1])
    resized = F.interpolate(
        flattened,
        size=target_steps,
        mode="linear",
        align_corners=False,
    )
    return resized.view(batch, num_objects, state_dim, target_steps).permute(0, 3, 1, 2).contiguous()


class PromptEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")

    def forward(self, prompts: Sequence[str], device) -> torch.Tensor:
        token_ids: List[int] = []
        offsets = [0]
        for prompt in prompts:
            ids = hash_prompt_tokens(prompt, self.vocab_size)
            token_ids.extend(ids)
            offsets.append(offsets[-1] + len(ids))
        flat = torch.tensor(token_ids, dtype=torch.long, device=device)
        offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return self.embedding(flat, offsets_tensor)

    def forward_tokens(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding.weight[token_ids]
        masked = embeddings * token_mask.unsqueeze(-1)
        denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return masked.sum(dim=1) / denom


class GroupedStateHeads(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, max_objects: int):
        super().__init__()
        self.max_objects = max_objects
        self.geom_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_objects * len(GEOM_STATE_INDICES)),
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_objects * len(MOTION_STATE_INDICES)),
        )
        self.vis_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_objects * len(VIS_STATE_INDICES)),
        )

    def forward(self, state_latents: torch.Tensor, num_objects: int) -> Dict[str, torch.Tensor]:
        if num_objects > self.max_objects:
            raise ValueError(f"num_objects={num_objects} exceeds max_objects={self.max_objects}")
        batch, steps = state_latents.shape[:2]
        geom = self.geom_head(state_latents).view(batch, steps, self.max_objects, len(GEOM_STATE_INDICES))
        motion = self.motion_head(state_latents).view(batch, steps, self.max_objects, len(MOTION_STATE_INDICES))
        vis = self.vis_head(state_latents).view(batch, steps, self.max_objects, len(VIS_STATE_INDICES))
        geom = geom[:, :, :num_objects]
        motion = motion[:, :, :num_objects]
        vis = vis[:, :, :num_objects]

        merged = state_latents.new_zeros((batch, steps, num_objects, STATE_DIM))
        merged[..., list(GEOM_STATE_INDICES)] = geom
        merged[..., list(MOTION_STATE_INDICES)] = motion
        merged[..., list(VIS_STATE_INDICES)] = vis
        return {
            "geom": geom,
            "motion": motion,
            "vis": vis,
            "state": merged,
        }

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)


class WanStateLatentPredictorV2(nn.Module):
    def __init__(self, config: WanStateLatentPredictorV2Config | None = None):
        super().__init__()
        self.config = config or WanStateLatentPredictorV2Config()
        latent_feature_dim = self.config.latent_channels * (self.config.latent_pool_side ** 2 + 2)

        self.prompt_encoder = PromptEncoder(self.config.prompt_vocab_size, self.config.prompt_embed_dim)
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(self.config.prompt_embed_dim),
            nn.Linear(self.config.prompt_embed_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        encoder_dim = latent_feature_dim + self.config.camera_dim + self.config.hidden_dim
        self.context_input_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.context_pos_embed = nn.Parameter(
            torch.randn(1, self.config.max_context_latent_steps, self.config.hidden_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=4 * self.config.hidden_dim,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_encoder_layers)
        self.context_state_latent_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.state_latent_dim),
        )

        self.future_queries = nn.Parameter(
            torch.randn(1, self.config.max_future_latent_steps, self.config.hidden_dim) * 0.02
        )
        self.future_pos_embed = nn.Parameter(
            torch.randn(1, self.config.max_future_latent_steps, self.config.hidden_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=4 * self.config.hidden_dim,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.future_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_decoder_layers)
        self.future_state_latent_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.state_latent_dim),
        )
        self.state_heads = GroupedStateHeads(
            latent_dim=self.config.state_latent_dim,
            hidden_dim=self.config.hidden_dim,
            max_objects=self.config.max_objects,
        )

    def freeze_state_heads(self) -> None:
        self.state_heads.freeze()

    def unfreeze_state_heads(self) -> None:
        self.state_heads.unfreeze()

    def _encode_prompt(
        self,
        prompts: Sequence[str] | None,
        prompt_token_ids: torch.Tensor | None,
        prompt_token_mask: torch.Tensor | None,
        device,
    ) -> torch.Tensor:
        if prompt_token_ids is not None and prompt_token_mask is not None:
            prompt_embed = self.prompt_encoder.forward_tokens(prompt_token_ids, prompt_token_mask)
        elif prompts is not None:
            prompt_embed = self.prompt_encoder(prompts, device)
        else:
            raise ValueError("either prompts or prompt_token_ids/prompt_token_mask must be provided")
        return self.prompt_proj(prompt_embed)

    def _latent_features(self, context_latents: torch.Tensor) -> torch.Tensor:
        if context_latents.ndim != 5:
            raise ValueError(
                f"expected context latents [B, T, C, H, W], got {tuple(context_latents.shape)}"
            )
        batch, steps, channels, height, width = context_latents.shape
        pooled = F.adaptive_avg_pool2d(
            context_latents.reshape(batch * steps, channels, height, width),
            output_size=(self.config.latent_pool_side, self.config.latent_pool_side),
        )
        pooled = pooled.reshape(batch, steps, channels * self.config.latent_pool_side ** 2)
        mean = context_latents.mean(dim=(-1, -2))
        std = context_latents.var(dim=(-1, -2), unbiased=False).add(1e-6).sqrt()
        return torch.cat([pooled, mean, std], dim=-1)

    def forward(
        self,
        context_latents: torch.Tensor,
        camera: torch.Tensor,
        prompts: Sequence[str] | None = None,
        prompt_token_ids: torch.Tensor | None = None,
        prompt_token_mask: torch.Tensor | None = None,
        future_latent_steps: int | None = None,
        num_objects: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch, context_steps = context_latents.shape[:2]
        future_latent_steps = future_latent_steps or self.config.max_future_latent_steps
        num_objects = num_objects or self.config.max_objects

        if context_steps > self.config.max_context_latent_steps:
            raise ValueError(
                f"context_steps={context_steps} exceeds max_context_latent_steps={self.config.max_context_latent_steps}"
            )
        if future_latent_steps > self.config.max_future_latent_steps:
            raise ValueError(
                f"future_latent_steps={future_latent_steps} exceeds max_future_latent_steps={self.config.max_future_latent_steps}"
            )
        if camera.shape[:2] != (batch, context_steps):
            raise ValueError(
                f"camera shape {tuple(camera.shape)} does not match context latents batch/steps {(batch, context_steps)}"
            )

        prompt_context = self._encode_prompt(
            prompts,
            prompt_token_ids,
            prompt_token_mask,
            context_latents.device,
        ).unsqueeze(1).expand(-1, context_steps, -1)
        latent_features = self._latent_features(context_latents)
        encoder_input = torch.cat([latent_features, camera, prompt_context], dim=-1)
        encoder_input = self.context_input_proj(encoder_input)
        encoder_input = encoder_input + self.context_pos_embed[:, :context_steps]

        context_hidden = self.context_encoder(encoder_input)
        context_state_latents = self.context_state_latent_proj(context_hidden)

        global_context = context_hidden.mean(dim=1, keepdim=True)
        future_queries = self.future_queries[:, :future_latent_steps] + self.future_pos_embed[:, :future_latent_steps]
        future_queries = future_queries.expand(batch, -1, -1) + global_context
        future_hidden = self.future_decoder(future_queries, context_hidden)
        future_state_latents = self.future_state_latent_proj(future_hidden)

        context_grouped = self.state_heads(context_state_latents, num_objects=num_objects)
        future_grouped = self.state_heads(future_state_latents, num_objects=num_objects)
        return {
            "context_state_latents": context_state_latents,
            "future_state_latents": future_state_latents,
            "context_geom_predictions": context_grouped["geom"],
            "context_motion_predictions": context_grouped["motion"],
            "context_vis_predictions": context_grouped["vis"],
            "context_state_predictions": context_grouped["state"],
            "future_geom_predictions": future_grouped["geom"],
            "future_motion_predictions": future_grouped["motion"],
            "future_vis_predictions": future_grouped["vis"],
            "future_state_predictions": future_grouped["state"],
        }


def _group_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)


def _compute_grouped_losses(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    geom_loss = _group_loss(predicted[..., list(GEOM_STATE_INDICES)], target[..., list(GEOM_STATE_INDICES)])
    motion_loss = _group_loss(predicted[..., list(MOTION_STATE_INDICES)], target[..., list(MOTION_STATE_INDICES)])
    vis_loss = _group_loss(predicted[..., list(VIS_STATE_INDICES)], target[..., list(VIS_STATE_INDICES)])
    total = geom_loss + motion_loss + 0.5 * vis_loss
    return {
        "loss": total,
        "geom": geom_loss,
        "motion": motion_loss,
        "vis": vis_loss,
    }


def wan_state_predictor_v2_loss(
    outputs: Dict[str, torch.Tensor],
    context_target: torch.Tensor,
    future_target: torch.Tensor,
    train_stage: str,
    latent_smooth_scale: float = 0.05,
) -> Dict[str, torch.Tensor]:
    context_losses = _compute_grouped_losses(outputs["context_state_predictions"], context_target)
    future_losses = _compute_grouped_losses(outputs["future_state_predictions"], future_target)

    future_state_latents = outputs["future_state_latents"]
    if future_state_latents.shape[1] > 1:
        latent_smooth = torch.mean((future_state_latents[:, 1:] - future_state_latents[:, :-1]) ** 2)
    else:
        latent_smooth = torch.zeros((), device=future_state_latents.device, dtype=future_state_latents.dtype)

    if train_stage == "context_only":
        total = context_losses["loss"]
    elif train_stage == "future_only":
        total = future_losses["loss"] + latent_smooth_scale * latent_smooth
    elif train_stage == "joint_finetune":
        total = context_losses["loss"] + future_losses["loss"] + latent_smooth_scale * latent_smooth
    else:
        raise ValueError(f"unsupported train_stage={train_stage}")

    return {
        "loss": total,
        "context_loss": context_losses["loss"],
        "context_geom": context_losses["geom"],
        "context_motion": context_losses["motion"],
        "context_vis": context_losses["vis"],
        "future_loss": future_losses["loss"],
        "future_geom": future_losses["geom"],
        "future_motion": future_losses["motion"],
        "future_vis": future_losses["vis"],
        "latent_smooth": latent_smooth,
    }
