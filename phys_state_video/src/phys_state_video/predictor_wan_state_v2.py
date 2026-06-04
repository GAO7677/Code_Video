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
    state_map_height: int = 2
    state_map_width: int = 2
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
            nn.Linear(hidden_dim, len(GEOM_STATE_INDICES)),
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(MOTION_STATE_INDICES)),
        )
        self.vis_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(VIS_STATE_INDICES)),
        )

    def forward(self, state_slots: torch.Tensor, num_objects: int) -> Dict[str, torch.Tensor]:
        if state_slots.ndim != 4:
            raise ValueError(f"expected state slots [B, T, N, D], got {tuple(state_slots.shape)}")
        if num_objects > self.max_objects:
            raise ValueError(f"num_objects={num_objects} exceeds max_objects={self.max_objects}")
        slots = state_slots[:, :, :num_objects]
        geom = self.geom_head(slots)
        motion = self.motion_head(slots)
        vis_logits = self.vis_head(slots)
        vis = torch.sigmoid(vis_logits)
        merged = state_slots.new_zeros((slots.shape[0], slots.shape[1], num_objects, STATE_DIM))
        merged[..., list(GEOM_STATE_INDICES)] = geom
        merged[..., list(MOTION_STATE_INDICES)] = motion
        merged[..., list(VIS_STATE_INDICES)] = vis
        return {
            "geom": geom,
            "motion": motion,
            "vis": vis,
            "vis_logits": vis_logits,
            "state": merged,
        }

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)


class SpatialObjectQueryDecoder(nn.Module):
    def __init__(self, hidden_dim: int, max_objects: int, num_heads: int, dropout: float):
        super().__init__()
        self.max_objects = max_objects
        self.object_queries = nn.Parameter(torch.randn(1, max_objects, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state_maps: torch.Tensor, num_objects: int) -> torch.Tensor:
        if state_maps.ndim != 5:
            raise ValueError(f"expected state maps [B, T, H, W, D], got {tuple(state_maps.shape)}")
        if num_objects > self.max_objects:
            raise ValueError(f"num_objects={num_objects} exceeds max_objects={self.max_objects}")
        batch, steps, height, width, hidden_dim = state_maps.shape
        memory = state_maps.view(batch * steps, height * width, hidden_dim)
        queries = self.object_queries[:, :num_objects].expand(batch * steps, -1, -1)
        attended, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        slots = self.out_norm(queries + attended)
        slots = self.out_proj(slots)
        return slots.view(batch, steps, num_objects, hidden_dim)


class AdapterTokenHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state_maps: torch.Tensor, object_slots: torch.Tensor) -> torch.Tensor:
        spatial_summary = state_maps.mean(dim=(2, 3))
        slot_summary = object_slots.mean(dim=2)
        return self.proj(torch.cat([spatial_summary, slot_summary], dim=-1))


class WanStateLatentPredictorV2(nn.Module):
    def __init__(self, config: WanStateLatentPredictorV2Config | None = None):
        super().__init__()
        self.config = config or WanStateLatentPredictorV2Config()
        self.model_dim = self.config.state_latent_dim

        self.prompt_encoder = PromptEncoder(self.config.prompt_vocab_size, self.config.prompt_embed_dim)
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(self.config.prompt_embed_dim),
            nn.Linear(self.config.prompt_embed_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.camera_proj = nn.Sequential(
            nn.LayerNorm(self.config.camera_dim),
            nn.Linear(self.config.camera_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.visual_stem = nn.Sequential(
            nn.Conv2d(self.config.latent_channels, self.model_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.model_dim, self.model_dim, kernel_size=3, padding=1),
        )
        self.context_pos_embed = nn.Parameter(
            torch.randn(1, self.config.max_context_latent_steps, 1, 1, self.model_dim) * 0.02
        )
        self.future_time_queries = nn.Parameter(
            torch.randn(1, self.config.max_future_latent_steps, 1, 1, self.model_dim) * 0.02
        )
        self.future_spatial_pos_embed = nn.Parameter(
            torch.randn(
                1,
                1,
                self.config.state_map_height,
                self.config.state_map_width,
                self.model_dim,
            )
            * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.config.num_heads,
            dim_feedforward=4 * self.config.hidden_dim,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=self.config.num_heads,
            dim_feedforward=4 * self.config.hidden_dim,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_encoder_layers)
        self.future_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_decoder_layers)
        self.object_query_decoder = SpatialObjectQueryDecoder(
            hidden_dim=self.model_dim,
            max_objects=self.config.max_objects,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.adapter_token_head = AdapterTokenHead(hidden_dim=self.model_dim)
        self.state_heads = GroupedStateHeads(
            latent_dim=self.model_dim,
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

    def _build_context_state_maps(
        self,
        context_latents: torch.Tensor,
        camera: torch.Tensor,
        prompt_embed: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, _, height, width = context_latents.shape
        visual = self.visual_stem(context_latents.view(batch * steps, context_latents.shape[2], height, width))
        visual = F.adaptive_avg_pool2d(
            visual,
            output_size=(self.config.state_map_height, self.config.state_map_width),
        )
        visual = visual.view(
            batch,
            steps,
            self.model_dim,
            self.config.state_map_height,
            self.config.state_map_width,
        ).permute(0, 1, 3, 4, 2).contiguous()
        camera_embed = self.camera_proj(camera).view(batch, steps, 1, 1, self.model_dim)
        prompt_embed = prompt_embed.view(batch, 1, 1, 1, self.model_dim)
        state_maps = visual + camera_embed + prompt_embed + self.context_pos_embed[:, :steps]
        grid_h, grid_w = self.config.state_map_height, self.config.state_map_width
        encoded = state_maps.permute(0, 2, 3, 1, 4).contiguous().view(batch * grid_h * grid_w, steps, self.model_dim)
        encoded = self.context_encoder(encoded)
        encoded = encoded.view(batch, grid_h, grid_w, steps, self.model_dim).permute(0, 3, 1, 2, 4).contiguous()
        return encoded

    def _build_future_state_maps(
        self,
        context_state_maps: torch.Tensor,
        future_latent_steps: int,
    ) -> torch.Tensor:
        batch, context_steps, grid_h, grid_w, hidden_dim = context_state_maps.shape
        memory = context_state_maps.view(batch, context_steps * grid_h * grid_w, hidden_dim)
        future_queries = self.future_time_queries[:, :future_latent_steps] + self.future_spatial_pos_embed
        future_queries = future_queries.expand(batch, -1, -1, -1, -1).contiguous()
        future_tokens = future_queries.view(batch, future_latent_steps * grid_h * grid_w, hidden_dim)
        future_hidden = self.future_decoder(future_tokens, memory)
        return future_hidden.view(batch, future_latent_steps, grid_h, grid_w, hidden_dim)

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
        if context_latents.ndim != 5:
            raise ValueError(
                f"expected context latents [B, T, C, H, W], got {tuple(context_latents.shape)}"
            )
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

        prompt_embed = self._encode_prompt(
            prompts,
            prompt_token_ids,
            prompt_token_mask,
            context_latents.device,
        )
        context_state_maps = self._build_context_state_maps(context_latents, camera, prompt_embed)
        future_state_maps = self._build_future_state_maps(context_state_maps, future_latent_steps)

        context_object_slots = self.object_query_decoder(context_state_maps, num_objects=num_objects)
        future_object_slots = self.object_query_decoder(future_state_maps, num_objects=num_objects)
        future_adapter_tokens = self.adapter_token_head(future_state_maps, future_object_slots)

        context_grouped = self.state_heads(context_object_slots, num_objects=num_objects)
        future_grouped = self.state_heads(future_object_slots, num_objects=num_objects)
        return {
            "context_state_latents": context_state_maps,
            "future_state_latents": future_state_maps,
            "context_state_maps": context_state_maps,
            "future_state_maps": future_state_maps,
            "context_object_slots": context_object_slots,
            "future_object_slots": future_object_slots,
            "future_adapter_tokens": future_adapter_tokens,
            "context_geom_predictions": context_grouped["geom"],
            "context_motion_predictions": context_grouped["motion"],
            "context_vis_predictions": context_grouped["vis"],
            "context_vis_logits": context_grouped["vis_logits"],
            "context_state_predictions": context_grouped["state"],
            "future_geom_predictions": future_grouped["geom"],
            "future_motion_predictions": future_grouped["motion"],
            "future_vis_predictions": future_grouped["vis"],
            "future_vis_logits": future_grouped["vis_logits"],
            "future_state_predictions": future_grouped["state"],
        }


def _group_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)


def wan_state_predictor_v2_loss(
    outputs: Dict[str, torch.Tensor],
    context_target: torch.Tensor,
    future_target: torch.Tensor,
    train_stage: str,
    latent_smooth_scale: float = 0.05,
) -> Dict[str, torch.Tensor]:
    context_geom_loss = _group_loss(
        outputs["context_state_predictions"][..., list(GEOM_STATE_INDICES)],
        context_target[..., list(GEOM_STATE_INDICES)],
    )
    context_motion_loss = _group_loss(
        outputs["context_state_predictions"][..., list(MOTION_STATE_INDICES)],
        context_target[..., list(MOTION_STATE_INDICES)],
    )
    context_vis_loss = F.binary_cross_entropy_with_logits(
        outputs["context_vis_logits"],
        context_target[..., list(VIS_STATE_INDICES)].clamp(0.0, 1.0),
    )
    context_losses = {
        "loss": context_geom_loss + context_motion_loss + 0.5 * context_vis_loss,
        "geom": context_geom_loss,
        "motion": context_motion_loss,
        "vis": context_vis_loss,
    }
    future_geom_loss = _group_loss(
        outputs["future_state_predictions"][..., list(GEOM_STATE_INDICES)],
        future_target[..., list(GEOM_STATE_INDICES)],
    )
    future_motion_loss = _group_loss(
        outputs["future_state_predictions"][..., list(MOTION_STATE_INDICES)],
        future_target[..., list(MOTION_STATE_INDICES)],
    )
    future_vis_loss = F.binary_cross_entropy_with_logits(
        outputs["future_vis_logits"],
        future_target[..., list(VIS_STATE_INDICES)].clamp(0.0, 1.0),
    )
    future_losses = {
        "loss": future_geom_loss + future_motion_loss + 0.5 * future_vis_loss,
        "geom": future_geom_loss,
        "motion": future_motion_loss,
        "vis": future_vis_loss,
    }

    future_adapter_tokens = outputs["future_adapter_tokens"]
    if future_adapter_tokens.shape[1] > 1:
        latent_smooth = torch.mean((future_adapter_tokens[:, 1:] - future_adapter_tokens[:, :-1]) ** 2)
    else:
        latent_smooth = torch.zeros((), device=future_adapter_tokens.device, dtype=future_adapter_tokens.dtype)

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
