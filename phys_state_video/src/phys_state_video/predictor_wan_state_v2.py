from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .schemas import STATE_DIM, StateIndex
from .utils import require_torch

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
    camera_dim: int = 8
    prompt_context_dim: int = 4096
    hidden_dim: int = 256
    state_latent_dim: int = 128
    state_map_height: int = 2
    state_map_width: int = 2
    max_context_latent_steps: int = 16
    max_future_latent_steps: int = 16
    max_prompt_tokens: int = 512
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


class ConditionMapHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state_maps: torch.Tensor) -> torch.Tensor:
        if state_maps.ndim != 5:
            raise ValueError(f"expected state maps [B, T, H, W, D], got {tuple(state_maps.shape)}")
        return self.proj(state_maps)


class MemoryTokenHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, context_object_slots: torch.Tensor) -> torch.Tensor:
        if context_object_slots.ndim != 4:
            raise ValueError(
                f"expected context object slots [B, T, N, D], got {tuple(context_object_slots.shape)}"
            )
        slots_bntd = context_object_slots.permute(0, 2, 1, 3).contiguous()
        weights = self.score(slots_bntd).squeeze(-1)
        weights = torch.softmax(weights, dim=-1)
        pooled = torch.sum(slots_bntd * weights.unsqueeze(-1), dim=2)
        return self.proj(pooled)


class WanStateLatentPredictorV2(nn.Module):
    def __init__(self, config: WanStateLatentPredictorV2Config | None = None):
        super().__init__()
        self.config = config or WanStateLatentPredictorV2Config()
        self.model_dim = self.config.state_latent_dim

        self.prompt_token_proj = nn.Sequential(
            nn.LayerNorm(self.config.prompt_context_dim),
            nn.Linear(self.config.prompt_context_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.prompt_summary_proj = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
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
        self.base_future_query = nn.Parameter(
            torch.randn(
                1,
                1,
                self.config.state_map_height,
                self.config.state_map_width,
                self.model_dim,
            )
            * 0.02
        )
        self.tail_query_proj = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.future_delta_proj = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(
                1,
                1,
                self.config.state_map_height,
                self.config.state_map_width,
                self.model_dim,
            )
            * 0.02
        )
        self.context_prompt_cross_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.context_fusion_norm = nn.LayerNorm(self.model_dim)
        self.context_prompt_norm = nn.LayerNorm(self.model_dim)
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
        self.condition_map_head = ConditionMapHead(hidden_dim=self.model_dim)
        self.memory_token_head = MemoryTokenHead(hidden_dim=self.model_dim)
        self.state_heads = GroupedStateHeads(
            latent_dim=self.model_dim,
            hidden_dim=self.config.hidden_dim,
            max_objects=self.config.max_objects,
        )
        nn.init.zeros_(self.tail_query_proj[-1].weight)
        nn.init.zeros_(self.tail_query_proj[-1].bias)
        nn.init.zeros_(self.future_delta_proj[-1].weight)
        nn.init.zeros_(self.future_delta_proj[-1].bias)

    def freeze_state_heads(self) -> None:
        self.state_heads.freeze()

    def unfreeze_state_heads(self) -> None:
        self.state_heads.unfreeze()

    def _encode_prompt_context(
        self,
        prompt_context: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_context.ndim != 3:
            raise ValueError(
                f"expected prompt_context [B, L_prompt, D_prompt], got {tuple(prompt_context.shape)}"
            )
        if prompt_mask.ndim != 2 or prompt_mask.shape[:2] != prompt_context.shape[:2]:
            raise ValueError(
                f"expected prompt_mask [B, L_prompt] matching prompt_context, got {tuple(prompt_mask.shape)}"
            )
        prompt_tokens = self.prompt_token_proj(prompt_context)
        masked = prompt_tokens * prompt_mask.unsqueeze(-1)
        denom = prompt_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        prompt_summary = self.prompt_summary_proj(masked.sum(dim=1) / denom)
        return prompt_tokens, prompt_summary

    def _build_time_embedding(
        self,
        steps: int,
        *,
        device,
        dtype,
        offset: int = 0,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        half_dim = self.model_dim // 2
        positions = torch.arange(offset, offset + steps, device=device, dtype=torch.float32)
        if half_dim <= 0:
            return positions.view(1, steps, 1, 1, 1).to(dtype=dtype)
        freq_exponent = -torch.arange(half_dim, device=device, dtype=torch.float32) / max(half_dim, 1)
        inv_freq = torch.exp(freq_exponent * torch.log(positions.new_tensor(10000.0)))
        angles = positions[:, None] * inv_freq[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.model_dim:
            padding = embedding.new_zeros((steps, self.model_dim - embedding.shape[-1]))
            embedding = torch.cat([embedding, padding], dim=-1)
        return embedding[:, : self.model_dim].view(1, steps, 1, 1, self.model_dim).to(dtype=dtype)

    def _build_context_state_maps(
        self,
        context_latents: torch.Tensor,
        camera: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        prompt_summary: torch.Tensor,
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
        prompt_embed = prompt_summary.view(batch, 1, 1, 1, self.model_dim)
        context_time_embed = self._build_time_embedding(
            steps,
            device=context_latents.device,
            dtype=visual.dtype,
        )
        state_maps = (
            visual
            + camera_embed
            + prompt_embed
            + context_time_embed
            + self.spatial_pos_embed
        )
        state_maps = self.context_fusion_norm(state_maps)
        grid_h, grid_w = self.config.state_map_height, self.config.state_map_width
        context_tokens = state_maps.view(batch, steps * grid_h * grid_w, self.model_dim)
        context_tokens = self.context_encoder(context_tokens)
        prompt_padding_mask = prompt_mask <= 0
        attended_prompt, _ = self.context_prompt_cross_attn(
            self.context_prompt_norm(context_tokens),
            prompt_tokens,
            prompt_tokens,
            key_padding_mask=prompt_padding_mask,
            need_weights=False,
        )
        context_tokens = context_tokens + attended_prompt
        return context_tokens.view(batch, steps, grid_h, grid_w, self.model_dim)

    def _build_future_state_maps(
        self,
        context_state_maps: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        future_latent_steps: int,
        future_camera: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, context_steps, grid_h, grid_w, hidden_dim = context_state_maps.shape
        context_tail = context_state_maps[:, -1:]
        context_memory = context_state_maps.view(batch, context_steps * grid_h * grid_w, hidden_dim)
        memory = torch.cat([context_memory, prompt_tokens], dim=1)
        context_padding_mask = torch.zeros(
            batch,
            context_memory.shape[1],
            dtype=torch.bool,
            device=context_state_maps.device,
        )
        prompt_padding_mask = prompt_mask <= 0
        memory_padding_mask = torch.cat([context_padding_mask, prompt_padding_mask], dim=1)
        future_time_embed = self._build_time_embedding(
            future_latent_steps,
            device=context_state_maps.device,
            dtype=context_state_maps.dtype,
            offset=context_steps,
        )
        tail_query = context_tail + self.tail_query_proj(context_tail)
        future_queries = tail_query + self.base_future_query + self.spatial_pos_embed + future_time_embed
        if future_camera is not None:
            if future_camera.shape[:2] != (batch, future_latent_steps):
                raise ValueError(
                    f"future_camera shape {tuple(future_camera.shape)} does not match "
                    f"(batch={batch}, future_latent_steps={future_latent_steps})"
                )
            future_camera_embed = self.camera_proj(future_camera).view(batch, future_latent_steps, 1, 1, hidden_dim)
            future_queries = future_queries + future_camera_embed
        future_queries = future_queries.expand(batch, -1, -1, -1, -1).contiguous()
        future_tokens = future_queries.view(batch, future_latent_steps * grid_h * grid_w, hidden_dim)
        future_hidden = self.future_decoder(
            future_tokens,
            memory,
            memory_key_padding_mask=memory_padding_mask,
        )
        future_hidden = future_hidden.view(batch, future_latent_steps, grid_h, grid_w, hidden_dim)
        # Predict latent deltas from the context tail, then integrate over future time.
        future_delta = self.future_delta_proj(future_hidden)
        return context_tail + torch.cumsum(future_delta, dim=1)

    def _predict_future_world(
        self,
        context_latents: torch.Tensor,
        camera: torch.Tensor,
        prompt_context: torch.Tensor,
        prompt_mask: torch.Tensor,
        future_latent_steps: int,
        future_camera: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        prompt_tokens, prompt_summary = self._encode_prompt_context(prompt_context, prompt_mask)
        context_state_maps = self._build_context_state_maps(
            context_latents,
            camera,
            prompt_tokens,
            prompt_mask,
            prompt_summary,
        )
        future_state_maps = self._build_future_state_maps(
            context_state_maps,
            prompt_tokens,
            prompt_mask,
            future_latent_steps,
            future_camera=future_camera,
        )
        return {
            "context_state_maps": context_state_maps,
            "future_state_maps": future_state_maps,
            "prompt_tokens": prompt_tokens,
        }

    def _readout_object_states(
        self,
        context_state_maps: torch.Tensor,
        future_state_maps: torch.Tensor,
        *,
        num_objects: int,
    ) -> Dict[str, torch.Tensor]:
        context_object_slots = self.object_query_decoder(context_state_maps, num_objects=num_objects)
        future_object_slots = self.object_query_decoder(future_state_maps, num_objects=num_objects)
        context_grouped = self.state_heads(context_object_slots, num_objects=num_objects)
        future_grouped = self.state_heads(future_object_slots, num_objects=num_objects)
        return {
            "context_object_slots": context_object_slots,
            "future_object_slots": future_object_slots,
            "context_state_predictions": context_grouped["state"],
            "future_state_predictions": future_grouped["state"],
            "context_geom_predictions": context_grouped["geom"],
            "context_motion_predictions": context_grouped["motion"],
            "context_vis_predictions": context_grouped["vis"],
            "context_vis_logits": context_grouped["vis_logits"],
            "future_geom_predictions": future_grouped["geom"],
            "future_motion_predictions": future_grouped["motion"],
            "future_vis_predictions": future_grouped["vis"],
            "future_vis_logits": future_grouped["vis_logits"],
        }

    def _project_wan_conditions(
        self,
        future_state_maps: torch.Tensor,
        context_object_slots: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        projected_condition_maps = self.condition_map_head(future_state_maps)
        condition_maps = projected_condition_maps.permute(0, 1, 4, 2, 3).contiguous()
        memory_tokens = self.memory_token_head(context_object_slots)
        return {
            "projected_condition_maps": projected_condition_maps,
            "condition_maps": condition_maps,
            "memory_tokens": memory_tokens,
        }

    def forward(
        self,
        context_latents: torch.Tensor,
        camera: torch.Tensor,
        prompt_context: torch.Tensor,
        prompt_mask: torch.Tensor,
        future_latent_steps: int | None = None,
        num_objects: int | None = None,
        future_camera: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if context_latents.ndim != 5:
            raise ValueError(
                f"expected context latents [B, T, C, H, W], got {tuple(context_latents.shape)}"
            )
        batch, context_steps = context_latents.shape[:2]
        future_latent_steps = future_latent_steps or self.config.max_future_latent_steps
        num_objects = num_objects or self.config.max_objects

        if camera.shape[:2] != (batch, context_steps):
            raise ValueError(
                f"camera shape {tuple(camera.shape)} does not match context latents batch/steps {(batch, context_steps)}"
            )
        if future_camera is not None and future_camera.shape[0] != batch:
            raise ValueError(
                f"future_camera batch size must match context batch {batch}, got {tuple(future_camera.shape)}"
            )
        if prompt_context.shape[0] != batch or prompt_mask.shape[0] != batch:
            raise ValueError(
                f"prompt batch size must match context batch {batch}, got {tuple(prompt_context.shape)} and "
                f"{tuple(prompt_mask.shape)}"
            )

        world_outputs = self._predict_future_world(
            context_latents,
            camera,
            prompt_context,
            prompt_mask,
            future_latent_steps,
            future_camera=future_camera,
        )
        readout_outputs = self._readout_object_states(
            world_outputs["context_state_maps"],
            world_outputs["future_state_maps"],
            num_objects=num_objects,
        )
        condition_outputs = self._project_wan_conditions(
            world_outputs["future_state_maps"],
            readout_outputs["context_object_slots"],
        )
        return {
            "context_state_maps": world_outputs["context_state_maps"],
            "future_state_maps": world_outputs["future_state_maps"],
            "condition_maps": condition_outputs["condition_maps"],
            "memory_tokens": condition_outputs["memory_tokens"],
            "context_state_predictions": readout_outputs["context_state_predictions"],
            "future_state_predictions": readout_outputs["future_state_predictions"],
            "context_geom_predictions": readout_outputs["context_geom_predictions"],
            "context_motion_predictions": readout_outputs["context_motion_predictions"],
            "context_vis_predictions": readout_outputs["context_vis_predictions"],
            "context_vis_logits": readout_outputs["context_vis_logits"],
            "future_geom_predictions": readout_outputs["future_geom_predictions"],
            "future_motion_predictions": readout_outputs["future_motion_predictions"],
            "future_vis_predictions": readout_outputs["future_vis_predictions"],
            "future_vis_logits": readout_outputs["future_vis_logits"],
            "context_object_slots": readout_outputs["context_object_slots"],
            "future_object_slots": readout_outputs["future_object_slots"],
            "projected_condition_maps": condition_outputs["projected_condition_maps"],
            "prompt_tokens": world_outputs["prompt_tokens"],
            "debug_context_object_slots": readout_outputs["context_object_slots"],
            "debug_future_object_slots": readout_outputs["future_object_slots"],
            "debug_projected_future_state_maps": condition_outputs["projected_condition_maps"],
            "debug_prompt_tokens": world_outputs["prompt_tokens"],
        }


def load_wan_state_predictor_v2_state_dict(
    model: WanStateLatentPredictorV2,
    state_dict: dict[str, torch.Tensor],
) -> None:
    adapted = dict(state_dict)
    old_future_queries = adapted.pop("future_time_queries", None)
    adapted.pop("context_time_pos_embed", None)
    if old_future_queries is not None and "base_future_query" not in adapted:
        if old_future_queries.ndim != 5:
            raise ValueError(
                f"expected legacy future_time_queries with shape [1, T, H, W, D], got {tuple(old_future_queries.shape)}"
            )
        base_future_query = old_future_queries.mean(dim=1, keepdim=True)
        target_shape = model.base_future_query.shape
        if base_future_query.shape != target_shape:
            if base_future_query.shape[0] != 1 or base_future_query.shape[1] != 1 or base_future_query.shape[-1] != target_shape[-1]:
                raise ValueError(
                    "legacy future_time_queries cannot be adapted to base_future_query: "
                    f"legacy_shape={tuple(base_future_query.shape)} target_shape={tuple(target_shape)}"
                )
            base_query = base_future_query.permute(0, 1, 4, 2, 3).contiguous().view(
                1,
                target_shape[-1],
                base_future_query.shape[2],
                base_future_query.shape[3],
            )
            resized = F.interpolate(
                base_query,
                size=(target_shape[2], target_shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            base_future_query = resized.view(1, 1, target_shape[-1], target_shape[2], target_shape[3]).permute(
                0, 1, 3, 4, 2
            ).contiguous()
        adapted["base_future_query"] = base_future_query
    incompatible = model.load_state_dict(adapted, strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    allowed_missing = {
        "base_future_query",
        "tail_query_proj.0.weight",
        "tail_query_proj.0.bias",
        "tail_query_proj.1.weight",
        "tail_query_proj.1.bias",
        "tail_query_proj.3.weight",
        "tail_query_proj.3.bias",
        "future_delta_proj.0.weight",
        "future_delta_proj.0.bias",
        "future_delta_proj.1.weight",
        "future_delta_proj.1.bias",
        "future_delta_proj.3.weight",
        "future_delta_proj.3.bias",
    }
    if missing_keys - allowed_missing or unexpected_keys:
        raise RuntimeError(
            "failed to load WanStateLatentPredictorV2 state_dict: "
            f"missing_keys={sorted(missing_keys)} unexpected_keys={sorted(unexpected_keys)}"
        )


def _group_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target) ** 2)


def _masked_group_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        return _group_loss(predicted, target)
    if mask.ndim != predicted.ndim - 1:
        raise ValueError(
            f"expected mask ndim {predicted.ndim - 1} for predicted {tuple(predicted.shape)}, got {tuple(mask.shape)}"
        )
    error = (predicted - target) ** 2
    weighted = error * mask.unsqueeze(-1)
    denom = (mask.sum() * predicted.shape[-1]).clamp_min(1.0)
    return weighted.sum() / denom


def _state_presence_mask(states: torch.Tensor) -> torch.Tensor:
    existence = states[..., StateIndex.EXISTENCE]
    visibility = states[..., StateIndex.VISIBILITY]
    return ((existence > 0.5) | (visibility > 0.2)).to(dtype=states.dtype)


def _boundary_object_mask(
    context_target: torch.Tensor,
    future_target: torch.Tensor,
) -> torch.Tensor:
    context_tail_mask = _state_presence_mask(context_target[:, -1])
    future_head_mask = _state_presence_mask(future_target[:, 0])
    return torch.maximum(context_tail_mask, future_head_mask)


def _boundary_head_loss(
    predicted_future: torch.Tensor,
    future_target: torch.Tensor,
    state_indices: tuple[int, ...],
    object_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted_head = predicted_future[:, 0, :, list(state_indices)]
    target_head = future_target[:, 0, :, list(state_indices)]
    return _masked_group_loss(predicted_head, target_head, object_mask)


def _boundary_rollout_loss(
    predicted_future: torch.Tensor,
    future_target: torch.Tensor,
    state_indices: tuple[int, ...],
    rollout_steps: int,
    rollout_decay: float,
) -> torch.Tensor:
    horizon = min(int(rollout_steps), int(future_target.shape[1]))
    if horizon <= 0:
        return predicted_future.new_zeros(())
    predicted = predicted_future[:, :horizon, :, list(state_indices)]
    target = future_target[:, :horizon, :, list(state_indices)]
    mask = _state_presence_mask(future_target[:, :horizon])
    weights = predicted.new_tensor(
        [float(rollout_decay) ** step_idx for step_idx in range(horizon)],
        dtype=predicted.dtype,
    ).view(1, horizon, 1)
    error = ((predicted - target) ** 2).mean(dim=-1)
    weighted = error * mask * weights
    denom = (mask * weights).sum().clamp_min(1.0)
    return weighted.sum() / denom


def _boundary_delta_loss(
    predicted_context: torch.Tensor,
    predicted_future: torch.Tensor,
    context_target: torch.Tensor,
    future_target: torch.Tensor,
    state_indices: tuple[int, ...],
    object_mask: torch.Tensor | None = None,
    detach_context_tail: bool = False,
) -> torch.Tensor:
    context_tail = predicted_context[:, -1, :, list(state_indices)]
    if detach_context_tail:
        context_tail = context_tail.detach()
    predicted_delta = (
        predicted_future[:, 0, :, list(state_indices)] - context_tail
    )
    target_delta = context_target.new_zeros(predicted_delta.shape)
    target_delta = future_target[:, 0, :, list(state_indices)] - context_target[:, -1, :, list(state_indices)]
    return _masked_group_loss(predicted_delta, target_delta, object_mask)


def _boundary_curvature_loss(
    predicted_context: torch.Tensor,
    predicted_future: torch.Tensor,
    context_target: torch.Tensor,
    future_target: torch.Tensor,
    state_indices: tuple[int, ...],
    object_mask: torch.Tensor | None = None,
    detach_context_tail: bool = False,
) -> torch.Tensor:
    if future_target.shape[1] < 2:
        return predicted_future.new_zeros(())
    context_tail = predicted_context[:, -1, :, list(state_indices)]
    if detach_context_tail:
        context_tail = context_tail.detach()
    predicted_second_delta = (
        predicted_future[:, 1, :, list(state_indices)]
        - 2.0 * predicted_future[:, 0, :, list(state_indices)]
        + context_tail
    )
    target_second_delta = (
        future_target[:, 1, :, list(state_indices)]
        - 2.0 * future_target[:, 0, :, list(state_indices)]
        + context_target[:, -1, :, list(state_indices)]
    )
    return _masked_group_loss(predicted_second_delta, target_second_delta, object_mask)


def wan_state_predictor_v2_loss(
    outputs: Dict[str, torch.Tensor],
    context_target: torch.Tensor,
    future_target: torch.Tensor,
    train_stage: str,
    latent_smooth_scale: float = 0.05,
    boundary_continuity_scale: float = 0.0,
    boundary_head_scale: float = 0.0,
    boundary_rollout_scale: float = 0.0,
    boundary_rollout_steps: int = 3,
    boundary_rollout_decay: float = 0.5,
    boundary_curvature_scale: float = 0.0,
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
    boundary_mask = _boundary_object_mask(context_target, future_target)
    boundary_head_geom_loss = _boundary_head_loss(
        outputs["future_state_predictions"],
        future_target,
        GEOM_STATE_INDICES,
        object_mask=boundary_mask,
    )
    boundary_head_motion_loss = _boundary_head_loss(
        outputs["future_state_predictions"],
        future_target,
        MOTION_STATE_INDICES,
        object_mask=boundary_mask,
    )
    boundary_head = boundary_head_geom_loss + boundary_head_motion_loss
    boundary_rollout_geom_loss = _boundary_rollout_loss(
        outputs["future_state_predictions"],
        future_target,
        GEOM_STATE_INDICES,
        rollout_steps=boundary_rollout_steps,
        rollout_decay=boundary_rollout_decay,
    )
    boundary_rollout_motion_loss = _boundary_rollout_loss(
        outputs["future_state_predictions"],
        future_target,
        MOTION_STATE_INDICES,
        rollout_steps=boundary_rollout_steps,
        rollout_decay=boundary_rollout_decay,
    )
    boundary_rollout = boundary_rollout_geom_loss + boundary_rollout_motion_loss

    boundary_geom_loss = _boundary_delta_loss(
        outputs["context_state_predictions"],
        outputs["future_state_predictions"],
        context_target,
        future_target,
        GEOM_STATE_INDICES,
        object_mask=boundary_mask,
        detach_context_tail=True,
    )
    boundary_motion_loss = _boundary_delta_loss(
        outputs["context_state_predictions"],
        outputs["future_state_predictions"],
        context_target,
        future_target,
        MOTION_STATE_INDICES,
        object_mask=boundary_mask,
        detach_context_tail=True,
    )
    boundary_continuity = boundary_geom_loss + boundary_motion_loss
    boundary_curvature_geom_loss = _boundary_curvature_loss(
        outputs["context_state_predictions"],
        outputs["future_state_predictions"],
        context_target,
        future_target,
        GEOM_STATE_INDICES,
        object_mask=boundary_mask,
        detach_context_tail=True,
    )
    boundary_curvature_motion_loss = _boundary_curvature_loss(
        outputs["context_state_predictions"],
        outputs["future_state_predictions"],
        context_target,
        future_target,
        MOTION_STATE_INDICES,
        object_mask=boundary_mask,
        detach_context_tail=True,
    )
    boundary_curvature = boundary_curvature_geom_loss + boundary_curvature_motion_loss

    future_state_maps = outputs["future_state_maps"]
    if future_state_maps.shape[1] > 1:
        latent_smooth = torch.mean((future_state_maps[:, 1:] - future_state_maps[:, :-1]) ** 2)
    else:
        latent_smooth = torch.zeros((), device=future_state_maps.device, dtype=future_state_maps.dtype)

    if train_stage == "context_only":
        total = context_losses["loss"]
    elif train_stage == "future_only":
        total = (
            future_losses["loss"]
            + latent_smooth_scale * latent_smooth
            + boundary_head_scale * boundary_head
            + boundary_rollout_scale * boundary_rollout
            + boundary_continuity_scale * boundary_continuity
            + boundary_curvature_scale * boundary_curvature
        )
    elif train_stage == "joint_finetune":
        total = (
            context_losses["loss"]
            + future_losses["loss"]
            + latent_smooth_scale * latent_smooth
            + boundary_head_scale * boundary_head
            + boundary_rollout_scale * boundary_rollout
            + boundary_continuity_scale * boundary_continuity
            + boundary_curvature_scale * boundary_curvature
        )
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
        "boundary_head": boundary_head,
        "boundary_head_geom": boundary_head_geom_loss,
        "boundary_head_motion": boundary_head_motion_loss,
        "boundary_rollout": boundary_rollout,
        "boundary_rollout_geom": boundary_rollout_geom_loss,
        "boundary_rollout_motion": boundary_rollout_motion_loss,
        "boundary_continuity": boundary_continuity,
        "boundary_geom": boundary_geom_loss,
        "boundary_motion": boundary_motion_loss,
        "boundary_curvature": boundary_curvature,
        "boundary_curvature_geom": boundary_curvature_geom_loss,
        "boundary_curvature_motion": boundary_curvature_motion_loss,
    }
