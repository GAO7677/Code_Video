from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .schemas import STATE_DIM, StateIndex
from .utils import hash_prompt_tokens, require_torch

torch = require_torch()
nn = torch.nn


@dataclass(slots=True)
class VisualLatentPredictorV4Config:
    context_channels: int = 3
    frame_height: int = 144
    frame_width: int = 256
    prompt_vocab_size: int = 4096
    prompt_embed_dim: int = 64
    hidden_dim: int = 192
    future_latent_dim: int = 128
    future_steps: int = 12
    max_objects: int = 6
    dropout: float = 0.1
    uncertainty_bias: float = 0.05
    center_delta_scale: float = 0.25
    depth_delta_scale: float = 0.10
    log_scale_delta_scale: float = 0.25


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


class VisualContextLatentPredictorV4(nn.Module):
    def __init__(self, config: VisualLatentPredictorV4Config | None = None):
        super().__init__()
        self.config = config or VisualLatentPredictorV4Config()
        self.prompt_encoder = PromptEncoder(self.config.prompt_vocab_size, self.config.prompt_embed_dim)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(self.config.context_channels, self.config.hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.hidden_dim // 2, self.config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.lat_mu = nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, kernel_size=1)
        self.lat_logvar = nn.Conv2d(self.config.hidden_dim, self.config.hidden_dim, kernel_size=1)
        self.temporal_encoder = nn.GRU(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.config.dropout,
        )
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(self.config.prompt_embed_dim),
            nn.Linear(self.config.prompt_embed_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(STATE_DIM),
            nn.Linear(STATE_DIM, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.state_to_token = nn.Sequential(
            nn.LayerNorm(STATE_DIM),
            nn.Linear(STATE_DIM, self.config.future_latent_dim),
        )
        self.slot_queries = nn.Parameter(torch.randn(1, self.config.max_objects, self.config.hidden_dim) * 0.02)
        self.decoder_cell = nn.GRUCell(
            input_size=self.config.future_latent_dim + 2 * self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
        )
        self.future_token_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.future_latent_dim),
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.config.future_latent_dim),
            nn.Linear(self.config.future_latent_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, STATE_DIM),
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

    def _encode_context(self, context_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, context_steps, channels, height, width = context_frames.shape
        encoded = self.frame_encoder(context_frames.reshape(batch * context_steps, channels, height, width))
        mu = self.lat_mu(encoded)
        logvar = self.lat_logvar(encoded).clamp(-8.0, 8.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        pooled = z.mean(dim=(-1, -2)).reshape(batch, context_steps, self.config.hidden_dim)
        _, hidden = self.temporal_encoder(pooled)
        kl = 0.5 * torch.mean(torch.exp(logvar) + mu * mu - 1.0 - logvar)
        return hidden[-1], kl

    def forward(
        self,
        context_frames: torch.Tensor,
        context_states: torch.Tensor,
        prompts: Sequence[str] | None = None,
        prompt_token_ids: torch.Tensor | None = None,
        prompt_token_mask: torch.Tensor | None = None,
        future_steps: int | None = None,
        num_objects: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch = context_frames.shape[0]
        future_steps = future_steps or self.config.future_steps
        num_objects = num_objects or self.config.max_objects
        if num_objects > self.config.max_objects:
            raise ValueError(f"num_objects={num_objects} exceeds max_objects={self.config.max_objects}")
        if context_states.shape[-1] != STATE_DIM:
            raise ValueError(f"expected context state dim {STATE_DIM}, got {context_states.shape[-1]}")

        prompt_context = self._encode_prompt(
            prompts,
            prompt_token_ids,
            prompt_token_mask,
            context_frames.device,
        )
        history_token, kl = self._encode_context(context_frames)
        history_token = history_token + prompt_context

        current = context_states[:, -1, :num_objects]
        state_context = self.state_encoder(current)
        slot_queries = self.slot_queries[:, :num_objects].expand(batch, num_objects, -1)
        hidden = history_token.unsqueeze(1).expand(-1, num_objects, -1) + slot_queries + state_context
        prev_token = self.state_to_token(current)

        outputs = []
        confidences = []
        latents = []
        motions = []
        history_expand = history_token.unsqueeze(1).expand(-1, num_objects, -1)
        for _ in range(future_steps):
            current_state_embed = self.state_encoder(current)
            decoder_input = torch.cat([prev_token, history_expand, current_state_embed], dim=-1)
            hidden = self.decoder_cell(
                decoder_input.reshape(batch * num_objects, -1),
                hidden.reshape(batch * num_objects, -1),
            ).reshape(batch, num_objects, -1)
            future_token = self.future_token_proj(hidden)
            raw_state = self.state_head(future_token)

            center_delta = self.config.center_delta_scale * torch.tanh(
                raw_state[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
            )
            depth_delta = self.config.depth_delta_scale * raw_state[..., StateIndex.DEPTH]
            log_scale_delta = self.config.log_scale_delta_scale * torch.tanh(raw_state[..., StateIndex.LOG_SCALE])

            next_state = current.clone()
            next_state[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = torch.clamp(
                current[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] + center_delta,
                0.0,
                1.0,
            )
            next_state[..., StateIndex.DEPTH] = torch.clamp_min(
                current[..., StateIndex.DEPTH] + depth_delta,
                1e-3,
            )
            next_state[..., StateIndex.LOG_SCALE] = current[..., StateIndex.LOG_SCALE] + log_scale_delta
            next_state[..., StateIndex.VEL_X:StateIndex.VEL_Y + 1] = center_delta
            next_state[..., StateIndex.DEPTH_VEL] = depth_delta
            next_state[..., StateIndex.VISIBILITY] = torch.sigmoid(raw_state[..., StateIndex.VISIBILITY])
            next_state[..., StateIndex.EXISTENCE] = torch.sigmoid(raw_state[..., StateIndex.EXISTENCE])
            next_state[..., StateIndex.CONFIDENCE] = torch.sigmoid(
                raw_state[..., StateIndex.CONFIDENCE] + self.config.uncertainty_bias
            )

            outputs.append(next_state)
            confidences.append(next_state[..., StateIndex.CONFIDENCE])
            latents.append(future_token)
            motions.append(
                torch.cat(
                    [center_delta, depth_delta.unsqueeze(-1)],
                    dim=-1,
                )
            )
            prev_token = future_token
            current = next_state

        predicted = torch.stack(outputs, dim=1)
        confidence = torch.stack(confidences, dim=1)
        future_latents = torch.stack(latents, dim=1)
        future_motion = torch.stack(motions, dim=1)
        return {
            "states": predicted,
            "confidence": confidence,
            "latents": future_latents,
            "motion": future_motion,
            "kl": kl,
            "anchor_state": context_states[:, -1, :num_objects],
        }


def predictor_visual_v4_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    kl_scale: float = 1e-4,
    first_step_delta_scale: float = 1.0,
    first_step_abs_scale: float = 0.5,
) -> Dict[str, torch.Tensor]:
    predicted = outputs["states"]
    future_latents = outputs["latents"]
    future_motion = outputs["motion"]
    anchor_state = outputs["anchor_state"]

    cont_idx = [
        StateIndex.CENTER_X,
        StateIndex.CENTER_Y,
        StateIndex.DEPTH,
        StateIndex.LOG_SCALE,
        StateIndex.VEL_X,
        StateIndex.VEL_Y,
        StateIndex.DEPTH_VEL,
    ]
    cont_pred = predicted[..., cont_idx]
    cont_target = target[..., cont_idx]
    mse = torch.mean((cont_pred - cont_target) ** 2)

    vis_loss = torch.mean((predicted[..., StateIndex.VISIBILITY] - target[..., StateIndex.VISIBILITY]) ** 2)
    exist_loss = torch.mean((predicted[..., StateIndex.EXISTENCE] - target[..., StateIndex.EXISTENCE]) ** 2)

    pred_vel = predicted[:, 1:, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] - predicted[:, :-1, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
    tgt_vel = target[:, 1:, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] - target[:, :-1, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
    smooth = torch.mean((pred_vel - tgt_vel) ** 2)

    scale_depth_pred = predicted[..., StateIndex.LOG_SCALE] + 2.0 * torch.log(predicted[..., StateIndex.DEPTH].clamp_min(1e-4))
    scale_depth_tgt = target[..., StateIndex.LOG_SCALE] + 2.0 * torch.log(target[..., StateIndex.DEPTH].clamp_min(1e-4))
    scale_depth = torch.mean((scale_depth_pred - scale_depth_tgt) ** 2)

    motion_target = torch.stack(
        [
            target[..., StateIndex.VEL_X],
            target[..., StateIndex.VEL_Y],
            target[..., StateIndex.DEPTH_VEL],
        ],
        dim=-1,
    )
    motion_aux = torch.mean((future_motion - motion_target) ** 2)

    velocity_alignment = torch.relu(
        -(future_motion[..., 0:2] * motion_target[..., 0:2]).sum(dim=-1)
    ).mean()
    latent_smooth = torch.mean((future_latents[:, 1:] - future_latents[:, :-1]) ** 2)

    anchor_idx = [
        StateIndex.CENTER_X,
        StateIndex.CENTER_Y,
        StateIndex.DEPTH,
        StateIndex.LOG_SCALE,
    ]
    pred_first_delta = predicted[:, 0, :, anchor_idx] - anchor_state[..., anchor_idx]
    tgt_first_delta = target[:, 0, :, anchor_idx] - anchor_state[..., anchor_idx]
    first_step_delta = torch.mean((pred_first_delta - tgt_first_delta) ** 2)
    first_step_abs = torch.mean((predicted[:, 0, :, anchor_idx] - target[:, 0, :, anchor_idx]) ** 2)

    kl = outputs.get("kl")
    if kl is None:
        kl = torch.zeros((), device=target.device, dtype=target.dtype)
    elif kl.ndim > 0:
        kl = kl.mean()

    total = (
        mse
        + 0.5 * vis_loss
        + 0.5 * exist_loss
        + 0.25 * smooth
        + 0.1 * scale_depth
        + 0.5 * motion_aux
        + 0.25 * velocity_alignment
        + 0.05 * latent_smooth
        + first_step_delta_scale * first_step_delta
        + first_step_abs_scale * first_step_abs
        + kl_scale * kl
    )
    return {
        "loss": total,
        "mse": mse,
        "visibility": vis_loss,
        "existence": exist_loss,
        "smoothness": smooth,
        "scale_depth": scale_depth,
        "motion_aux": motion_aux,
        "velocity_align": velocity_alignment,
        "latent_smooth": latent_smooth,
        "first_step_delta": first_step_delta,
        "first_step_abs": first_step_abs,
        "kl": kl,
    }
