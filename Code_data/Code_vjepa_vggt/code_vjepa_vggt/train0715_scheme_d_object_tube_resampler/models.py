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
    motion_tokens_per_object: int
    valid_objects: int
    jepa_frames: int
    latent_frames: int
    track_frames: int


def fourier_encode(
    values: torch.Tensor,
    *,
    num_bands: int,
    max_frequency: float = 8.0,
) -> torch.Tensor:
    """Append fixed sinusoidal bands to scalar coordinates."""
    if num_bands <= 0:
        return values
    frequencies = torch.logspace(
        0.0,
        torch.log10(values.new_tensor(float(max_frequency))),
        int(num_bands),
        device=values.device,
        dtype=values.dtype,
    )
    angles = values.unsqueeze(-1) * frequencies * torch.pi
    encoded = torch.cat([values.unsqueeze(-1), angles.sin(), angles.cos()], dim=-1)
    return encoded.flatten(start_dim=-2)


def sinusoidal_time_embedding(times: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a fixed time embedding without a learned 1 -> D expansion."""
    if dim <= 0:
        raise ValueError("time embedding dimension must be positive")
    half = dim // 2
    if half == 0:
        return times.unsqueeze(-1)
    frequencies = torch.exp(
        torch.arange(half, device=times.device, dtype=times.dtype)
        * (-torch.log(times.new_tensor(10_000.0)) / max(half - 1, 1))
    )
    angles = times.unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class TrajectoryTokenEncoder(nn.Module):
    """Compress all tracked point observations into a few motion tokens/object."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_motion_tokens: int = 4,
        num_heads: int = 8,
        fourier_bands: int = 4,
        max_points: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by motion num_heads")
        if num_motion_tokens <= 0:
            raise ValueError("num_motion_tokens must be positive")
        self.hidden_dim = int(hidden_dim)
        self.num_motion_tokens = int(num_motion_tokens)
        self.fourier_bands = int(fourier_bands)
        self.max_points = int(max_points)
        # Five geometric scalars each retain their raw value and 2F Fourier
        # values; visibility and confidence remain explicit scalar channels.
        input_dim = 5 * (1 + 2 * self.fourier_bands) + 2
        bottleneck_dim = max(self.hidden_dim // 2, 64)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.hidden_dim),
        )
        self.motion_queries = nn.Parameter(
            torch.randn(self.num_motion_tokens, self.hidden_dim)
            / self.hidden_dim**0.5
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.source_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            self.hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        track_state: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
        """
        Args:
            track_state: ``[B,T,O,P,7]`` containing x/y, dx/dy, t, vis, conf.
            valid: ``[B,T,O,P]``.

        Returns:
            Motion tokens ``[B,O,M,H]`` and validity ``[B,O,M]``.
        """
        if track_state.ndim != 5 or int(track_state.shape[-1]) != 7:
            raise ValueError("track_state must be [B,T,O,P,7]")
        batch, frames, objects, points = track_state.shape[:4]
        if points > self.max_points:
            raise ValueError(f"points={points} exceeds max_points={self.max_points}")
        geometry = fourier_encode(
            track_state[..., :5],
            num_bands=self.fourier_bands,
        )
        encoded_input = torch.cat([geometry, track_state[..., 5:]], dim=-1)
        encoded = self.input_proj(self.input_norm(encoded_input))
        encoded = encoded.permute(0, 2, 1, 3, 4).reshape(
            batch * objects,
            frames * points,
            self.hidden_dim,
        )
        source_valid = valid.permute(0, 2, 1, 3).reshape(
            batch * objects,
            frames * points,
        )
        object_has_motion = source_valid.any(dim=1)
        if bool((~object_has_motion).any()):
            source_valid = source_valid.clone()
            encoded = encoded.clone()
            source_valid[~object_has_motion, 0] = True
            encoded[~object_has_motion, 0] = 0.0

        queries = self.motion_queries.unsqueeze(0).expand(batch * objects, -1, -1)
        delta, _ = self.cross_attn(
            self.query_norm(queries),
            self.source_norm(encoded),
            self.source_norm(encoded),
            key_padding_mask=~source_valid,
            need_weights=False,
        )
        queries = queries + delta
        queries = queries + self.ffn(self.ffn_norm(queries))
        queries = self.output_norm(queries)
        queries = queries * object_has_motion[:, None, None].to(queries.dtype)
        tokens = queries.view(
            batch,
            objects,
            self.num_motion_tokens,
            self.hidden_dim,
        )
        token_valid = object_has_motion.view(batch, objects, 1).expand(
            -1, -1, self.num_motion_tokens
        )
        trace = {
            "motion_fourier_features_B_T_O_P_F": list(encoded_input.shape),
            "motion_encoded_observations_BO_TP_H": list(encoded.shape),
            "motion_queries_BO_M_H": [
                batch * objects,
                self.num_motion_tokens,
                self.hidden_dim,
            ],
            "motion_tokens_B_O_M_H": list(tokens.shape),
        }
        return tokens, token_valid, trace


class ObjectTubeResampler(nn.Module):
    """Compress per-object visual and trajectory tubes into fixed learned tokens.

    Resampling is independent across objects. Source tokens contain local VAE
    features, local V-JEPA features, and compressed trajectory tokens. The
    output layout is ``[B, K, O, D]`` so existing object adapters can treat K
    as a learned token-type axis without exposing raw context time to Wan.
    """

    def __init__(
        self,
        *,
        jepa_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_output_tokens: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        num_motion_tokens: int = 4,
        motion_fourier_bands: int = 4,
        spatial_fourier_bands: int = 4,
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
        self.num_motion_tokens = int(num_motion_tokens)
        self.spatial_fourier_bands = int(spatial_fourier_bands)
        self.max_objects = int(max_objects)
        self.max_points = int(max_points)
        self.modality_dropout_prob = float(modality_dropout_prob)
        self.min_box_px = float(min_box_px)

        self.jepa_norm = nn.LayerNorm(self.jepa_dim)
        self.jepa_proj = nn.Sequential(
            nn.Linear(self.jepa_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.latent_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.motion_encoder = TrajectoryTokenEncoder(
            hidden_dim=self.hidden_dim,
            num_motion_tokens=self.num_motion_tokens,
            num_heads=num_heads,
            fourier_bands=int(motion_fourier_bands),
            max_points=self.max_points,
        )
        spatial_dim = 2 * (1 + 2 * self.spatial_fourier_bands)
        self.spatial_norm = nn.LayerNorm(spatial_dim)
        self.spatial_proj = nn.Linear(spatial_dim, self.hidden_dim)
        self.modality_embed = nn.Embedding(3, self.hidden_dim)
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
        self.output_proj = (
            nn.Identity()
            if self.hidden_dim == self.output_dim
            else nn.Linear(self.hidden_dim, self.output_dim)
        )
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

    def _visual_source_tokens(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        positions_norm: torch.Tensor,
        *,
        normalization: nn.Module,
        projection: nn.Module,
        modality_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, objects, points = values.shape[:4]
        if points > self.max_points:
            raise ValueError(f"points={points} exceeds max_points={self.max_points}")
        if positions_norm.shape != values.shape[:-1] + (2,):
            raise ValueError("positions_norm must match values [B,T,O,P,2]")
        projected = projection(normalization(values))
        times = torch.linspace(0.0, 1.0, frames, device=values.device, dtype=values.dtype)
        time_bias = sinusoidal_time_embedding(times, self.hidden_dim).view(
            1, frames, 1, 1, self.hidden_dim
        )
        spatial_features = fourier_encode(
            positions_norm,
            num_bands=self.spatial_fourier_bands,
        )
        spatial_bias = self.spatial_proj(self.spatial_norm(spatial_features))
        modality_bias = self.modality_embed.weight[modality_id].view(
            1, 1, 1, 1, self.hidden_dim
        )
        projected = projected + time_bias + spatial_bias + modality_bias
        if (
            self.training
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

        latent_grid = context_latents.permute(0, 2, 3, 4, 1).contiguous()
        latent_frames = int(latent_grid.shape[1])
        latent_ids = self._time_indices(track_frames, latent_frames, tracks.device)
        latent_tracks = tracks[:, latent_ids]
        latent_visibility = visibility[:, latent_ids]
        latent_confidence = confidence[:, latent_ids]
        latent_values = self._sample_feature_grid(
            latent_grid, latent_tracks, image_hw=track_image_hw
        )
        latent_positions = torch.stack(
            [
                latent_tracks[..., 0]
                / max(float(track_image_hw[1] - 1), 1.0),
                latent_tracks[..., 1]
                / max(float(track_image_hw[0] - 1), 1.0),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

        jepa_frames = int(jepa_patch_tokens.shape[1])
        jepa_ids = self._time_indices(track_frames, jepa_frames, tracks.device)
        jepa_tracks = tracks[:, jepa_ids]
        jepa_visibility = visibility[:, jepa_ids]
        jepa_confidence = confidence[:, jepa_ids]
        jepa_values = self._sample_feature_grid(
            jepa_patch_tokens, jepa_tracks, image_hw=track_image_hw
        )
        jepa_positions = torch.stack(
            [
                jepa_tracks[..., 0]
                / max(float(track_image_hw[1] - 1), 1.0),
                jepa_tracks[..., 1]
                / max(float(track_image_hw[0] - 1), 1.0),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

        xy = torch.stack(
            [
                tracks[..., 0] / max(float(track_image_hw[1] - 1), 1.0),
                tracks[..., 1] / max(float(track_image_hw[0] - 1), 1.0),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        delta = torch.zeros_like(xy)
        delta[:, 1:] = xy[:, 1:] - xy[:, :-1]
        normalized_time = torch.linspace(
            0.0,
            1.0,
            track_frames,
            device=tracks.device,
            dtype=tracks.dtype,
        ).view(1, track_frames, 1, 1, 1)
        normalized_time = normalized_time.expand(batch, -1, objects, points, -1)
        track_values = torch.cat(
            [
                xy,
                delta,
                normalized_time,
                visibility.unsqueeze(-1),
                confidence.unsqueeze(-1),
            ],
            dim=-1,
        )

        slot_valid = object_valid[:, None, :, None]
        latent_valid = (latent_visibility * latent_confidence > 1.0e-5) & slot_valid
        jepa_valid = (jepa_visibility * jepa_confidence > 1.0e-5) & slot_valid
        track_valid = (visibility * confidence > 1.0e-5) & slot_valid
        latent_tokens, latent_token_valid = self._visual_source_tokens(
            latent_values,
            latent_valid,
            latent_positions,
            normalization=self.latent_norm,
            projection=self.latent_proj,
            modality_id=0,
        )
        jepa_tokens, jepa_token_valid = self._visual_source_tokens(
            jepa_values,
            jepa_valid,
            jepa_positions,
            normalization=self.jepa_norm,
            projection=self.jepa_proj,
            modality_id=1,
        )
        motion_tokens, motion_token_valid, motion_trace = self.motion_encoder(
            track_values,
            track_valid,
        )
        motion_tokens = motion_tokens + self.modality_embed.weight[2].view(
            1, 1, 1, self.hidden_dim
        )
        source = torch.cat([latent_tokens, jepa_tokens, motion_tokens], dim=2)
        source_valid = torch.cat(
            [latent_token_valid, jepa_token_valid, motion_token_valid], dim=2
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
            motion_tokens_per_object=self.num_motion_tokens,
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
            "08b_latent_sample_positions": list(latent_positions.shape),
            "09_jepa_tracks": list(jepa_tracks.shape),
            "10_jepa_samples": list(jepa_values.shape),
            "10b_jepa_sample_positions": list(jepa_positions.shape),
            "11_track_state_xy_dxdy_t_vis_conf": list(track_values.shape),
            "12_latent_source_tokens_BO_S_H": list(latent_tokens.shape),
            "13_jepa_source_tokens_BO_S_H": list(jepa_tokens.shape),
            "14_motion_source_tokens_B_O_M_H": list(motion_tokens.shape),
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
        self._last_shape_trace.update(motion_trace)

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


class BottleneckObjectCrossAttention(nn.Module):
    """Cross-attend Wan video queries to compact object memory.

    Video tokens stay at Wan's hidden width while attention runs at a much
    smaller width. This keeps condition-injection capacity proportional to the
    256-dimensional object memory instead of allocating a full 3072-wide
    attention branch at every selected DiT block.
    """

    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        inner_dim: int = 256,
        num_heads: int = 8,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if inner_dim % num_heads != 0:
            raise ValueError("object attention inner_dim must be divisible by num_heads")
        self.query_dim = int(query_dim)
        self.context_dim = int(context_dim)
        self.inner_dim = int(inner_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.inner_dim // self.num_heads
        self.eps = float(eps)
        self.q = nn.Linear(self.query_dim, self.inner_dim, bias=False)
        self.k = nn.Linear(self.context_dim, self.inner_dim, bias=False)
        self.v = nn.Linear(self.context_dim, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.query_dim, bias=False)
        for module in (self.q, self.k, self.v, self.o):
            nn.init.xavier_uniform_(module.weight)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _rms_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = tensor.float().square().mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(scale + self.eps).to(dtype=tensor.dtype)
        return tensor * scale

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 3 or int(query.shape[-1]) != self.query_dim:
            raise ValueError(
                f"query must be [B,N,{self.query_dim}], got {list(query.shape)}"
            )
        if context.ndim != 3 or int(context.shape[-1]) != self.context_dim:
            raise ValueError(
                f"context must be [B,L,{self.context_dim}], got {list(context.shape)}"
            )
        q = self._rms_normalize(self._split_heads(self.q(query)))
        k = self._rms_normalize(self._split_heads(self.k(context)))
        v = self._split_heads(self.v(context))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().flatten(start_dim=2)
        return self.o(attended)


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


def install_bottleneck_object_cross_attention(
    dit: nn.Module,
    active_block_ids: tuple[int, ...],
    *,
    object_dim: int,
    inner_dim: int = 256,
    num_heads: int = 8,
) -> dict[str, int | list[int]]:
    """Replace active full-width object attention with compact adapters."""
    layout = prune_object_cross_attention_blocks(dit, active_block_ids)
    query_dim = int(getattr(dit, "dim"))
    blocks = list(getattr(dit, "blocks", []))
    for block_id in layout["active_block_ids"]:
        block = blocks[int(block_id)]
        old_module = getattr(block, "object_cross_attn", None)
        old_param = next(old_module.parameters(), None)
        if old_param is None:
            raise RuntimeError(f"object cross-attention missing at block {block_id}")
        block.object_cross_attn = BottleneckObjectCrossAttention(
            query_dim=query_dim,
            context_dim=int(object_dim),
            inner_dim=int(inner_dim),
            num_heads=int(num_heads),
        ).to(device=old_param.device, dtype=torch.float32)
    # The custom attention consumes 256-dimensional memory directly.
    dit.object_embedding = None
    return {
        **layout,
        "object_dim": int(object_dim),
        "attention_inner_dim": int(inner_dim),
        "attention_heads": int(num_heads),
    }
