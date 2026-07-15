from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from code_vjepa_vggt.models.object_tokens import ObjectTokenOutput


@dataclass(frozen=True)
class TubeResamplerDiagnostics:
    source_tokens_per_object: int
    output_tokens_per_object: int
    valid_objects: int
    jepa_frames: int
    latent_frames: int
    track_frames: int


class ObjectTubeResampler(nn.Module):
    """Compress per-object visual and trajectory tubes into fixed learned tokens.

    Resampling is independent across objects. Source tokens contain local VAE
    features, local V-JEPA features, and all tracked point observations. The
    output layout is ``[B, K, O, D]`` so existing object adapters can treat K
    as a learned token-type axis without exposing raw context time to Wan.
    """

    def __init__(
        self,
        *,
        jepa_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_output_tokens: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        max_objects: int = 4,
        max_points: int = 16,
        modality_dropout_prob: float = 0.10,
        min_box_px: float = 16.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_output_tokens <= 0 or num_layers <= 0:
            raise ValueError("num_output_tokens and num_layers must be positive")
        if not 0.0 <= modality_dropout_prob < 1.0:
            raise ValueError("modality_dropout_prob must be in [0, 1)")

        self.jepa_dim = int(jepa_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_output_tokens = int(num_output_tokens)
        self.max_objects = int(max_objects)
        self.max_points = int(max_points)
        self.modality_dropout_prob = float(modality_dropout_prob)
        self.min_box_px = float(min_box_px)

        self.jepa_proj = nn.Linear(self.jepa_dim, self.hidden_dim)
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.track_proj = nn.Sequential(
            nn.Linear(6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.modality_embed = nn.Embedding(3, self.hidden_dim)
        self.point_embed = nn.Embedding(self.max_points, self.hidden_dim)
        self.slot_embed = nn.Embedding(self.max_objects, self.hidden_dim)
        self.output_queries = nn.Parameter(
            torch.randn(self.num_output_tokens, self.hidden_dim)
            / self.hidden_dim**0.5
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(self.hidden_dim),
                        "source_norm": nn.LayerNorm(self.hidden_dim),
                        "cross_attn": nn.MultiheadAttention(
                            self.hidden_dim,
                            num_heads,
                            dropout=0.0,
                            batch_first=True,
                        ),
                        "ffn_norm": nn.LayerNorm(self.hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
                            nn.GELU(),
                            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                        ),
                    }
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self._last_diagnostics: TubeResamplerDiagnostics | None = None
        self._last_shape_trace: dict[str, list[int]] | None = None

    @staticmethod
    def _group_tracks(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tracks.ndim == 4:
            return tracks.unsqueeze(3), visibility.unsqueeze(3), confidence.unsqueeze(3)
        if tracks.ndim != 5 or visibility.ndim != 4 or confidence.ndim != 4:
            raise ValueError(
                "tracks must be [B,T,O,P,2] with visibility/confidence [B,T,O,P]"
            )
        return tracks, visibility, confidence

    @staticmethod
    def _time_indices(src_frames: int, dst_frames: int, device: torch.device) -> torch.Tensor:
        if dst_frames <= 1:
            return torch.zeros(1, device=device, dtype=torch.long)
        return torch.linspace(
            0, max(src_frames - 1, 0), dst_frames, device=device
        ).round().long()

    @staticmethod
    def _sample_feature_grid(
        feature_grid: torch.Tensor,
        tracks: torch.Tensor,
        *,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Sample ``[B,T,H,W,C]`` features at ``[B,T,O,P,2]`` pixels."""
        if feature_grid.ndim != 5 or tracks.ndim != 5:
            raise ValueError("feature_grid and tracks must both have five dimensions")
        batch, frames, grid_h, grid_w, channels = feature_grid.shape
        if tuple(tracks.shape[:2]) != (batch, frames):
            raise ValueError("feature and track batch/time dimensions differ")
        objects, points = int(tracks.shape[2]), int(tracks.shape[3])
        height, width = int(image_hw[0]), int(image_hw[1])
        x = tracks[..., 0] / max(float(width - 1), 1.0)
        y = tracks[..., 1] / max(float(height - 1), 1.0)
        grid = torch.stack(
            [x.clamp(0.0, 1.0) * 2.0 - 1.0, y.clamp(0.0, 1.0) * 2.0 - 1.0],
            dim=-1,
        ).reshape(batch * frames, objects * points, 1, 2)
        feature_map = feature_grid.permute(0, 1, 4, 2, 3).reshape(
            batch * frames, channels, grid_h, grid_w
        )
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).permute(0, 2, 1).reshape(
            batch, frames, objects, points, channels
        )

    def _source_tokens(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        *,
        projection: nn.Module,
        modality_id: int,
        slot_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects, points = values.shape[:4]
        if points > self.max_points:
            raise ValueError(f"points={points} exceeds max_points={self.max_points}")
        projected = projection(values)
        times = torch.linspace(0.0, 1.0, frames, device=values.device, dtype=values.dtype)
        time_bias = self.time_proj(times[:, None]).view(1, frames, 1, 1, self.hidden_dim)
        point_ids = torch.arange(points, device=values.device)
        point_bias = self.point_embed(point_ids).view(1, 1, 1, points, self.hidden_dim)
        modality_bias = self.modality_embed.weight[modality_id].view(
            1, 1, 1, 1, self.hidden_dim
        )
        projected = projected + time_bias + point_bias + slot_bias + modality_bias
        # Keep trajectory observations as an always-available fallback. Only
        # the two visual modalities are independently dropped.
        if (
            self.training
            and modality_id in (0, 1)
            and self.modality_dropout_prob > 0.0
        ):
            keep = torch.rand(batch, 1, objects, 1, device=values.device) >= self.modality_dropout_prob
            valid = valid & keep
        tokens = projected.permute(0, 2, 1, 3, 4).reshape(
            batch, objects, frames * points, self.hidden_dim
        )
        token_valid = valid.permute(0, 2, 1, 3).reshape(
            batch, objects, frames * points
        )
        return tokens, token_valid

    @staticmethod
    def _last_track_state(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        box_prior_xyxy: torch.Tensor | None,
        min_box_px: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (visibility * confidence).clamp_min(0.0)
        denom = weights.sum(dim=3, keepdim=True).clamp_min(1.0e-6)
        centers = (tracks * weights.unsqueeze(-1)).sum(dim=3) / denom
        height, width = int(image_hw[0]), int(image_hw[1])
        centers_norm = torch.stack(
            [
                centers[..., 0] / max(float(width - 1), 1.0),
                centers[..., 1] / max(float(height - 1), 1.0),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        delta = centers_norm[:, -1] - centers_norm[:, 0]
        last_vis = visibility[:, -1].mean(dim=2, keepdim=True)
        last_conf = confidence[:, -1].mean(dim=2, keepdim=True)
        summary = torch.cat([centers_norm[:, -1], delta, last_vis, last_conf], dim=-1)

        last_tracks = tracks[:, -1]
        last_valid = weights[:, -1] > 1.0e-6
        x = last_tracks[..., 0] / max(float(width - 1), 1.0)
        y = last_tracks[..., 1] / max(float(height - 1), 1.0)
        x_min = torch.where(last_valid, x, torch.inf).amin(dim=2)
        y_min = torch.where(last_valid, y, torch.inf).amin(dim=2)
        x_max = torch.where(last_valid, x, -torch.inf).amax(dim=2)
        y_max = torch.where(last_valid, y, -torch.inf).amax(dim=2)
        any_valid = last_valid.any(dim=2)
        min_wh = last_tracks.new_tensor(
            [
                float(min_box_px) / max(float(width - 1), 1.0),
                float(min_box_px) / max(float(height - 1), 1.0),
            ]
        )
        center = torch.stack([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dim=-1)
        wh = torch.stack([x_max - x_min, y_max - y_min], dim=-1)
        wh = torch.maximum(torch.nan_to_num(wh, nan=0.0), min_wh)
        boxes = torch.cat([(center - 0.5 * wh).clamp(0.0, 1.0), (center + 0.5 * wh).clamp(0.0, 1.0)], dim=-1)
        if box_prior_xyxy is not None:
            prior = box_prior_xyxy.to(device=boxes.device, dtype=boxes.dtype).clamp(0.0, 1.0)
            boxes = torch.where(any_valid.unsqueeze(-1), boxes, prior)
        else:
            fallback = boxes.new_tensor([0.45, 0.45, 0.55, 0.55])
            boxes = torch.where(any_valid.unsqueeze(-1), boxes, fallback)
        return summary, boxes

    def pop_diagnostics(self) -> TubeResamplerDiagnostics | None:
        diagnostics = self._last_diagnostics
        self._last_diagnostics = None
        return diagnostics

    def pop_shape_trace(self) -> dict[str, list[int]] | None:
        trace = self._last_shape_trace
        self._last_shape_trace = None
        return trace

    def forward(
        self,
        jepa_patch_tokens: torch.Tensor,
        context_latents: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        track_image_hw: tuple[int, int],
        object_valid_mask: torch.Tensor | None = None,
        box_prior_xyxy: torch.Tensor | None = None,
        **_: object,
    ) -> ObjectTokenOutput:
        tracks, visibility, confidence = self._group_tracks(
            tracks.float(), visibility.float(), confidence.float()
        )
        jepa_patch_tokens = torch.nan_to_num(jepa_patch_tokens.float())
        context_latents = torch.nan_to_num(context_latents.float())
        tracks = torch.nan_to_num(tracks)
        visibility = torch.nan_to_num(visibility).clamp(0.0, 1.0)
        confidence = torch.nan_to_num(confidence).clamp(0.0, 1.0)
        batch, track_frames, objects, points = tracks.shape[:4]
        if objects > self.max_objects:
            raise ValueError(f"objects={objects} exceeds max_objects={self.max_objects}")
        if object_valid_mask is None:
            object_valid_mask = torch.ones(
                batch, objects, device=tracks.device, dtype=tracks.dtype
            )
        object_valid = object_valid_mask.to(device=tracks.device) > 0.5
        slot_ids = torch.arange(objects, device=tracks.device)
        slot_bias = self.slot_embed(slot_ids).view(1, 1, objects, 1, self.hidden_dim)

        latent_grid = context_latents.permute(0, 2, 3, 4, 1).contiguous()
        latent_frames = int(latent_grid.shape[1])
        latent_ids = self._time_indices(track_frames, latent_frames, tracks.device)
        latent_tracks = tracks[:, latent_ids]
        latent_visibility = visibility[:, latent_ids]
        latent_confidence = confidence[:, latent_ids]
        latent_values = self._sample_feature_grid(
            latent_grid, latent_tracks, image_hw=track_image_hw
        )

        jepa_frames = int(jepa_patch_tokens.shape[1])
        jepa_ids = self._time_indices(track_frames, jepa_frames, tracks.device)
        jepa_tracks = tracks[:, jepa_ids]
        jepa_visibility = visibility[:, jepa_ids]
        jepa_confidence = confidence[:, jepa_ids]
        jepa_values = self._sample_feature_grid(
            jepa_patch_tokens, jepa_tracks, image_hw=track_image_hw
        )

        xy = torch.stack(
            [
                tracks[..., 0] / max(float(track_image_hw[1] - 1), 1.0),
                tracks[..., 1] / max(float(track_image_hw[0] - 1), 1.0),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        delta = torch.zeros_like(xy)
        delta[:, 1:] = xy[:, 1:] - xy[:, :-1]
        track_values = torch.cat(
            [xy, delta, visibility.unsqueeze(-1), confidence.unsqueeze(-1)], dim=-1
        )

        slot_valid = object_valid[:, None, :, None]
        latent_valid = (latent_visibility * latent_confidence > 1.0e-5) & slot_valid
        jepa_valid = (jepa_visibility * jepa_confidence > 1.0e-5) & slot_valid
        track_valid = (visibility * confidence > 1.0e-5) & slot_valid
        latent_tokens, latent_token_valid = self._source_tokens(
            latent_values,
            latent_valid,
            projection=self.latent_proj,
            modality_id=0,
            slot_bias=slot_bias,
        )
        jepa_tokens, jepa_token_valid = self._source_tokens(
            jepa_values,
            jepa_valid,
            projection=self.jepa_proj,
            modality_id=1,
            slot_bias=slot_bias,
        )
        track_tokens, track_token_valid = self._source_tokens(
            track_values,
            track_valid,
            projection=self.track_proj,
            modality_id=2,
            slot_bias=slot_bias,
        )
        source = torch.cat([latent_tokens, jepa_tokens, track_tokens], dim=2)
        source_valid = torch.cat(
            [latent_token_valid, jepa_token_valid, track_token_valid], dim=2
        )
        source = source.reshape(batch * objects, source.shape[2], self.hidden_dim)
        source_valid = source_valid.reshape(batch * objects, source_valid.shape[2])
        object_valid_flat = object_valid.reshape(batch * objects)
        all_masked = ~source_valid.any(dim=1)
        if bool(all_masked.any()):
            source_valid = source_valid.clone()
            source_valid[all_masked, 0] = True
            source = source.clone()
            source[all_masked, 0] = 0.0

        queries = self.output_queries.unsqueeze(0).expand(batch * objects, -1, -1)
        for layer in self.layers:
            delta_query, _ = layer["cross_attn"](
                layer["query_norm"](queries),
                layer["source_norm"](source),
                layer["source_norm"](source),
                key_padding_mask=~source_valid,
                need_weights=False,
            )
            queries = queries + delta_query
            queries = queries + layer["ffn"](layer["ffn_norm"](queries))
        output = self.output_proj(self.output_norm(queries))
        output = output * object_valid_flat[:, None, None].to(output.dtype)
        output = output.view(batch, objects, self.num_output_tokens, self.output_dim)
        output = output.permute(0, 2, 1, 3).contiguous()

        track_summary, last_boxes = self._last_track_state(
            tracks,
            visibility,
            confidence,
            image_hw=track_image_hw,
            box_prior_xyxy=box_prior_xyxy,
            min_box_px=self.min_box_px,
        )
        track_summary = track_summary[:, None].expand(
            -1, self.num_output_tokens, -1, -1
        )
        active_boxes = last_boxes[:, None].expand(
            -1, self.num_output_tokens, -1, -1
        )
        output_mask = object_valid_mask[:, None, :, None].to(output.dtype)
        track_summary = track_summary * output_mask
        active_boxes = active_boxes * output_mask
        self._last_diagnostics = TubeResamplerDiagnostics(
            source_tokens_per_object=int(source.shape[1]),
            output_tokens_per_object=self.num_output_tokens,
            valid_objects=int(object_valid.sum().item()),
            jepa_frames=jepa_frames,
            latent_frames=latent_frames,
            track_frames=track_frames,
        )
        self._last_shape_trace = {
            "01_jepa_patch_tokens": list(jepa_patch_tokens.shape),
            "02_context_latents": list(context_latents.shape),
            "03_grouped_tracks": list(tracks.shape),
            "04_visibility": list(visibility.shape),
            "05_confidence": list(confidence.shape),
            "06_object_valid_mask": list(object_valid_mask.shape),
            "07_latent_tracks": list(latent_tracks.shape),
            "08_latent_samples": list(latent_values.shape),
            "09_jepa_tracks": list(jepa_tracks.shape),
            "10_jepa_samples": list(jepa_values.shape),
            "11_track_state_features": list(track_values.shape),
            "12_latent_source_tokens_BO_S_H": list(latent_tokens.shape),
            "13_jepa_source_tokens_BO_S_H": list(jepa_tokens.shape),
            "14_track_source_tokens_BO_S_H": list(track_tokens.shape),
            "15_joined_source_tokens_BO_S_H": [
                batch,
                objects,
                int(source.shape[1]),
                self.hidden_dim,
            ],
            "16_isolated_object_batch_BO_S_H": list(source.shape),
            "17_learned_queries_BO_K_H": [
                batch * objects,
                self.num_output_tokens,
                self.hidden_dim,
            ],
            "18_resampled_object_tokens_B_K_O_D": list(output.shape),
            "19_boundary_track_summary_B_K_O_6": list(track_summary.shape),
            "20_boundary_boxes_B_K_O_4": list(active_boxes.shape),
        }

        pooled = output.mean(dim=1)
        zeros = torch.zeros_like(output)
        return ObjectTokenOutput(
            object_tokens=pooled,
            object_latent_tokens=output,
            jepa_tokens=pooled,
            jepa_latent_tokens=zeros,
            latent_tokens=pooled,
            latent_latent_tokens=zeros,
            geom_tokens=pooled,
            track_geom_latent_tokens=zeros,
            vggt_geom_tokens=None,
            depth_latent_tokens=None,
            world_latent_tokens=None,
            motion_latent_tokens=pooled,
            active_track_summary=track_summary,
            active_box_xyxy=active_boxes,
        )


def parse_block_ids(raw_value: str, *, num_blocks: int) -> tuple[int, ...]:
    block_ids = tuple(
        sorted({int(part.strip()) for part in str(raw_value).split(",") if part.strip()})
    )
    if not block_ids:
        raise ValueError("at least one object cross-attention block is required")
    invalid = [block_id for block_id in block_ids if not 0 <= block_id < num_blocks]
    if invalid:
        raise ValueError(f"object block IDs outside [0,{num_blocks - 1}]: {invalid}")
    return block_ids


def prune_object_cross_attention_blocks(
    dit: nn.Module,
    active_block_ids: tuple[int, ...],
) -> dict[str, int | list[int]]:
    """Remove object-only modules from inactive DiT blocks in-place."""
    blocks = list(getattr(dit, "blocks", []))
    active = set(parse_block_ids(",".join(map(str, active_block_ids)), num_blocks=len(blocks)))
    removed = 0
    for block_id, block in enumerate(blocks):
        if block_id in active:
            continue
        for name in ("object_cross_attn", "norm4", "object_gate"):
            if getattr(block, name, None) is not None:
                setattr(block, name, None)
        removed += 1
    return {
        "num_blocks": len(blocks),
        "active_block_ids": sorted(active),
        "active_count": len(active),
        "removed_count": removed,
    }
