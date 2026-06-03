from __future__ import annotations

from typing import Dict, List, Sequence

from .config import PredictorConfig
from .interaction import compute_interaction_features
from .schemas import STATE_DIM, StateIndex
from .utils import hash_prompt_tokens, require_torch

torch = require_torch()
nn = torch.nn


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


class FutureStatePredictor(nn.Module):
    def __init__(self, config: PredictorConfig | None = None):
        super().__init__()
        self.config = config or PredictorConfig()
        input_dim = self.config.state_dim + self.config.appearance_dim + self.config.camera_dim + self.config.prompt_embed_dim
        decoder_input_dim = input_dim + 3
        self.prompt_encoder = PromptEncoder(self.config.prompt_vocab_size, self.config.prompt_embed_dim)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.decoder_cell = nn.GRUCell(decoder_input_dim, self.config.hidden_dim)
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
        self.motion_head = nn.Sequential(
            nn.LayerNorm(self.config.future_latent_dim),
            nn.Linear(self.config.future_latent_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 3),
        )

    def forward(
        self,
        context_states: torch.Tensor,
        appearance: torch.Tensor,
        camera: torch.Tensor,
        prompts: Sequence[str] | None = None,
        prompt_token_ids: torch.Tensor | None = None,
        prompt_token_mask: torch.Tensor | None = None,
        future_steps: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch, context_steps, num_objects, state_dim = context_states.shape
        if state_dim != STATE_DIM:
            raise ValueError(f"expected state dim {STATE_DIM}, got {state_dim}")
        future_steps = future_steps or self.config.future_steps

        if prompt_token_ids is not None and prompt_token_mask is not None:
            prompt_embed = self.prompt_encoder.forward_tokens(prompt_token_ids, prompt_token_mask)
        elif prompts is not None:
            prompt_embed = self.prompt_encoder(prompts, context_states.device)
        else:
            raise ValueError("either prompts or prompt_token_ids/prompt_token_mask must be provided")

        prompt_expand = prompt_embed[:, None, None, :].expand(batch, context_steps, num_objects, -1)
        appearance_expand = appearance[:, None, :, :].expand(batch, context_steps, num_objects, -1)
        camera_expand = camera[:, :, None, :].expand(batch, context_steps, num_objects, -1)
        encoder_input = torch.cat([context_states, appearance_expand, camera_expand, prompt_expand], dim=-1)
        encoder_input = encoder_input.reshape(batch * num_objects, context_steps, -1)

        _, hidden = self.encoder(encoder_input)
        hidden = hidden[-1]
        current = context_states[:, -1]
        outputs = []
        confidences = []
        latents = []
        motions = []

        prompt_last = prompt_embed[:, None, :].expand(batch, num_objects, -1)
        appearance_last = appearance
        camera_last = camera[:, -1, None, :].expand(batch, num_objects, -1)

        for _ in range(future_steps):
            interaction = compute_interaction_features(
                current[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1],
                current[..., StateIndex.EXISTENCE],
            )
            decoder_input = torch.cat([current, appearance_last, camera_last, prompt_last, interaction], dim=-1)
            hidden = self.decoder_cell(decoder_input.reshape(batch * num_objects, -1), hidden)
            future_token = self.future_token_proj(hidden).reshape(batch, num_objects, self.config.future_latent_dim)
            raw_state = self.state_head(future_token)
            motion = self.motion_head(future_token)

            next_state = current.clone()
            next_state[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = (
                current[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] + motion[..., 0:2]
            )
            next_state[..., StateIndex.DEPTH] = current[..., StateIndex.DEPTH] + motion[..., 2]
            next_state[..., StateIndex.LOG_SCALE] = current[..., StateIndex.LOG_SCALE] + 0.1 * raw_state[..., StateIndex.LOG_SCALE]
            next_state[..., StateIndex.VEL_X:StateIndex.VEL_Y + 1] = motion[..., 0:2]
            next_state[..., StateIndex.DEPTH_VEL] = motion[..., 2]
            next_state[..., StateIndex.VISIBILITY] = torch.sigmoid(raw_state[..., StateIndex.VISIBILITY])
            next_state[..., StateIndex.EXISTENCE] = torch.sigmoid(raw_state[..., StateIndex.EXISTENCE])
            next_state[..., StateIndex.CONFIDENCE] = torch.sigmoid(
                raw_state[..., StateIndex.CONFIDENCE] + self.config.uncertainty_bias
            )
            current = next_state
            outputs.append(next_state)
            confidences.append(next_state[..., StateIndex.CONFIDENCE])
            latents.append(future_token)
            motions.append(motion)

        predicted = torch.stack(outputs, dim=1)
        confidence = torch.stack(confidences, dim=1)
        future_latents = torch.stack(latents, dim=1)
        future_motion = torch.stack(motions, dim=1)
        return {
            "states": predicted,
            "confidence": confidence,
            "latents": future_latents,
            "motion": future_motion,
        }


def predictor_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
    predicted = outputs["states"]
    future_latents = outputs["latents"]
    future_motion = outputs["motion"]

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

    total = (
        mse
        + 0.5 * vis_loss
        + 0.5 * exist_loss
        + 0.25 * smooth
        + 0.1 * scale_depth
        + 0.5 * motion_aux
        + 0.25 * velocity_alignment
        + 0.05 * latent_smooth
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
    }
