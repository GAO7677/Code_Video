from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .schemas import STATE_DIM, StateIndex
from .utils import hash_prompt_tokens, require_torch

torch = require_torch()
nn = torch.nn
F = torch.nn.functional


@dataclass(slots=True)
class WanStateLatentPredictorConfig:
    latent_channels: int = 16
    latent_pool_side: int = 2
    camera_dim: int = 8
    prompt_vocab_size: int = 4096
    prompt_embed_dim: int = 64
    hidden_dim: int = 256
    state_latent_dim: int = 128
    future_steps: int = 12
    max_objects: int = 6
    max_context_steps: int = 16
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dropout: float = 0.1


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


class WanStateLatentPredictor(nn.Module):
    def __init__(self, config: WanStateLatentPredictorConfig | None = None):
        super().__init__()
        self.config = config or WanStateLatentPredictorConfig()

        latent_feature_dim = self.config.latent_channels * (self.config.latent_pool_side ** 2 + 2)
        encoder_dim = latent_feature_dim + self.config.camera_dim + self.config.prompt_embed_dim

        self.prompt_encoder = PromptEncoder(self.config.prompt_vocab_size, self.config.prompt_embed_dim)
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(self.config.prompt_embed_dim),
            nn.Linear(self.config.prompt_embed_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.context_input_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.context_pos_embed = nn.Parameter(
            torch.randn(1, self.config.max_context_steps, self.config.hidden_dim) * 0.02
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
            torch.randn(1, self.config.future_steps, self.config.hidden_dim) * 0.02
        )
        self.future_pos_embed = nn.Parameter(
            torch.randn(1, self.config.future_steps, self.config.hidden_dim) * 0.02
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

        self.object_state_head = nn.Sequential(
            nn.LayerNorm(self.config.state_latent_dim),
            nn.Linear(self.config.state_latent_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.max_objects * STATE_DIM),
        )

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
                f"expected context latents with shape [B, K, C, H, W], got {tuple(context_latents.shape)}"
            )
        batch, context_steps, channels, height, width = context_latents.shape
        pooled = F.adaptive_avg_pool2d(
            context_latents.reshape(batch * context_steps, channels, height, width),
            output_size=(self.config.latent_pool_side, self.config.latent_pool_side),
        )
        pooled = pooled.reshape(batch, context_steps, channels * self.config.latent_pool_side ** 2)
        mean = context_latents.mean(dim=(-1, -2))
        std = context_latents.var(dim=(-1, -2), unbiased=False).add(1e-6).sqrt()
        return torch.cat([pooled, mean, std], dim=-1)

    def _decode_object_states(self, state_latents: torch.Tensor, num_objects: int) -> torch.Tensor:
        if num_objects > self.config.max_objects:
            raise ValueError(f"num_objects={num_objects} exceeds max_objects={self.config.max_objects}")
        logits = self.object_state_head(state_latents)
        logits = logits.view(state_latents.shape[0], state_latents.shape[1], self.config.max_objects, STATE_DIM)
        return logits[:, :, :num_objects]

    def forward(
        self,
        context_latents: torch.Tensor,
        camera: torch.Tensor,
        prompts: Sequence[str] | None = None,
        prompt_token_ids: torch.Tensor | None = None,
        prompt_token_mask: torch.Tensor | None = None,
        future_steps: int | None = None,
        num_objects: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch, context_steps = context_latents.shape[:2]
        future_steps = future_steps or self.config.future_steps
        num_objects = num_objects or self.config.max_objects
        if future_steps > self.config.future_steps:
            raise ValueError(f"future_steps={future_steps} exceeds configured future_steps={self.config.future_steps}")
        if context_steps > self.config.max_context_steps:
            raise ValueError(
                f"context_steps={context_steps} exceeds configured max_context_steps={self.config.max_context_steps}"
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
        )
        prompt_context = prompt_context.unsqueeze(1).expand(-1, context_steps, -1)

        latent_features = self._latent_features(context_latents)
        encoder_input = torch.cat([latent_features, camera, prompt_context], dim=-1)
        encoder_input = self.context_input_proj(encoder_input)
        encoder_input = encoder_input + self.context_pos_embed[:, :context_steps]
        context_hidden = self.context_encoder(encoder_input)
        context_state_latents = self.context_state_latent_proj(context_hidden)

        global_context = context_hidden.mean(dim=1, keepdim=True)
        future_queries = self.future_queries[:, :future_steps] + self.future_pos_embed[:, :future_steps]
        future_queries = future_queries.expand(batch, -1, -1) + global_context
        future_hidden = self.future_decoder(future_queries, context_hidden)
        future_state_latents = self.future_state_latent_proj(future_hidden)

        context_state_predictions = self._decode_object_states(context_state_latents, num_objects)
        future_state_predictions = self._decode_object_states(future_state_latents, num_objects)
        return {
            "context_state_latents": context_state_latents,
            "future_state_latents": future_state_latents,
            "context_state_predictions": context_state_predictions,
            "future_state_predictions": future_state_predictions,
        }


def _state_supervision_loss(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    cont_idx = [
        StateIndex.CENTER_X,
        StateIndex.CENTER_Y,
        StateIndex.DEPTH,
        StateIndex.LOG_SCALE,
        StateIndex.VEL_X,
        StateIndex.VEL_Y,
        StateIndex.DEPTH_VEL,
    ]
    cont_loss = torch.mean((predicted[..., cont_idx] - target[..., cont_idx]) ** 2)
    visibility_loss = torch.mean((predicted[..., StateIndex.VISIBILITY] - target[..., StateIndex.VISIBILITY]) ** 2)
    existence_loss = torch.mean((predicted[..., StateIndex.EXISTENCE] - target[..., StateIndex.EXISTENCE]) ** 2)
    confidence_loss = torch.mean((predicted[..., StateIndex.CONFIDENCE] - target[..., StateIndex.CONFIDENCE]) ** 2)
    total = cont_loss + 0.5 * visibility_loss + 0.5 * existence_loss + 0.25 * confidence_loss
    return {
        "loss": total,
        "continuous": cont_loss,
        "visibility": visibility_loss,
        "existence": existence_loss,
        "confidence": confidence_loss,
    }


def wan_state_predictor_loss(
    outputs: Dict[str, torch.Tensor],
    context_target: torch.Tensor,
    future_target: torch.Tensor,
    context_loss_scale: float = 0.5,
    latent_smooth_scale: float = 0.05,
) -> Dict[str, torch.Tensor]:
    context_losses = _state_supervision_loss(outputs["context_state_predictions"], context_target)
    future_losses = _state_supervision_loss(outputs["future_state_predictions"], future_target)

    future_state_latents = outputs["future_state_latents"]
    if future_state_latents.shape[1] > 1:
        latent_smooth = torch.mean((future_state_latents[:, 1:] - future_state_latents[:, :-1]) ** 2)
    else:
        latent_smooth = torch.zeros((), device=future_state_latents.device, dtype=future_state_latents.dtype)

    total = future_losses["loss"] + context_loss_scale * context_losses["loss"] + latent_smooth_scale * latent_smooth
    return {
        "loss": total,
        "future_loss": future_losses["loss"],
        "future_continuous": future_losses["continuous"],
        "future_visibility": future_losses["visibility"],
        "future_existence": future_losses["existence"],
        "future_confidence": future_losses["confidence"],
        "context_loss": context_losses["loss"],
        "context_continuous": context_losses["continuous"],
        "context_visibility": context_losses["visibility"],
        "context_existence": context_losses["existence"],
        "context_confidence": context_losses["confidence"],
        "latent_smooth": latent_smooth,
    }
