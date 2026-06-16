from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObjectTokenOutput:
    object_tokens: torch.Tensor
    jepa_tokens: torch.Tensor
    latent_tokens: torch.Tensor
    geom_tokens: torch.Tensor
    vggt_geom_tokens: torch.Tensor | None = None


class ObjectTubeProjector(nn.Module):
    def __init__(
        self,
        jepa_dim: int,
        latent_dim: int,
        out_dim: int,
        jepa_window_radius: int = 1,
        latent_window_radius: int = 1,
    ) -> None:
        super().__init__()
        self.jepa_window_radius = jepa_window_radius
        self.latent_window_radius = latent_window_radius
        self.out_dim = out_dim
        self.jepa_proj = nn.Linear(jepa_dim, out_dim)
        self.latent_proj = nn.Linear(latent_dim, out_dim)
        self.geom_proj = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.vggt_geom_proj = nn.Sequential(
            nn.Linear(5, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self.vggt_world_clip = 16.0
        self.vggt_depth_clip = 16.0

    def _ensure_latent_proj(self, latent_dim: int, device: torch.device) -> None:
        if self.latent_proj.in_features == latent_dim:
            return
        self.latent_proj = nn.Linear(latent_dim, self.out_dim).to(device)

    def _ensure_jepa_proj(self, jepa_dim: int, device: torch.device) -> None:
        if self.jepa_proj.in_features == jepa_dim:
            return
        self.jepa_proj = nn.Linear(jepa_dim, self.out_dim).to(device)

    @staticmethod
    def _time_indices(src_frames: int, dst_frames: int, device: torch.device) -> torch.Tensor:
        if dst_frames <= 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return torch.linspace(0, src_frames - 1, dst_frames, device=device).round().long()

    @staticmethod
    def _normalized_tracks(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        height, width = image_hw
        tracks = torch.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
        visibility = torch.nan_to_num(visibility, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
        x = tracks[..., 0] / max(width - 1, 1)
        y = tracks[..., 1] / max(height - 1, 1)
        x = x.clamp(0.0, 1.0)
        y = y.clamp(0.0, 1.0)
        return torch.stack([x, y, visibility, confidence], dim=-1)

    def _pool_feature_grid(
        self,
        features: torch.Tensor,
        tracks: torch.Tensor,
        image_hw: tuple[int, int],
        window_radius: int,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim == 4:
            features = features.unsqueeze(-1)
        batch, frames, grid_h, grid_w, dim = features.shape
        _, _, objects, _ = tracks.shape
        height, width = image_hw
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        tracks = torch.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
        feature_map = features.permute(0, 1, 4, 2, 3).reshape(batch * frames, dim, grid_h, grid_w)
        if window_radius > 0:
            kernel = 2 * window_radius + 1
            feature_map = F.avg_pool2d(feature_map, kernel_size=kernel, stride=1, padding=window_radius)

        x = tracks[..., 0] / max(float(width - 1), 1.0)
        y = tracks[..., 1] / max(float(height - 1), 1.0)
        x = x.clamp(0.0, float(width - 1)) / max(float(width - 1), 1.0)
        y = y.clamp(0.0, float(height - 1)) / max(float(height - 1), 1.0)
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).view(batch * frames, objects, 1, 2)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 1).reshape(batch, frames, objects, dim)
        if frame_valid_mask is None:
            weights = sampled.new_ones(batch, frames, objects, 1)
        else:
            weights = frame_valid_mask[:, :, None, None].to(dtype=sampled.dtype, device=sampled.device)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (sampled * weights).sum(dim=1) / denom

    def forward(
        self,
        jepa_patch_tokens: torch.Tensor,
        context_latents: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        track_image_hw: tuple[int, int],
        vggt_world_points: torch.Tensor | None = None,
        vggt_world_points_conf: torch.Tensor | None = None,
        vggt_depth: torch.Tensor | None = None,
        vggt_depth_conf: torch.Tensor | None = None,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> ObjectTokenOutput:
        # Keep the object-token path in fp32 so bf16/autocast cannot inject
        # occasional non-finite values into the conditioning stream.
        with torch.autocast(device_type=jepa_patch_tokens.device.type, enabled=False):
            jepa_patch_tokens = torch.nan_to_num(jepa_patch_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
            context_latents = torch.nan_to_num(context_latents.float(), nan=0.0, posinf=0.0, neginf=0.0)
            tracks = torch.nan_to_num(tracks.float(), nan=0.0, posinf=0.0, neginf=0.0)
            visibility = torch.nan_to_num(visibility.float(), nan=0.0, posinf=0.0, neginf=0.0)
            confidence = torch.nan_to_num(confidence.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if vggt_world_points is not None:
                vggt_world_points = torch.nan_to_num(vggt_world_points.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if vggt_world_points_conf is not None:
                vggt_world_points_conf = torch.nan_to_num(vggt_world_points_conf.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if vggt_depth is not None:
                vggt_depth = torch.nan_to_num(vggt_depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if vggt_depth_conf is not None:
                vggt_depth_conf = torch.nan_to_num(vggt_depth_conf.float(), nan=0.0, posinf=0.0, neginf=0.0)
            jepa_time_idx = self._time_indices(tracks.shape[1], jepa_patch_tokens.shape[1], tracks.device)
            latent_time_idx = self._time_indices(tracks.shape[1], context_latents.shape[2], tracks.device)

            jepa_tracks = tracks[:, jepa_time_idx]
            latent_tracks = tracks[:, latent_time_idx]
            geom_steps = self._normalized_tracks(tracks, visibility, confidence, track_image_hw)
            jepa_valid = frame_valid_mask[:, jepa_time_idx] if frame_valid_mask is not None else None
            latent_valid = frame_valid_mask[:, latent_time_idx] if frame_valid_mask is not None else None

            jepa_local = self._pool_feature_grid(
                jepa_patch_tokens,
                jepa_tracks,
                image_hw=track_image_hw,
                window_radius=self.jepa_window_radius,
                frame_valid_mask=jepa_valid,
            )

            latent_grid = context_latents.permute(0, 2, 3, 4, 1).contiguous()
            latent_local = self._pool_feature_grid(
                latent_grid,
                latent_tracks,
                image_hw=track_image_hw,
                window_radius=self.latent_window_radius,
                frame_valid_mask=latent_valid,
            )

            expected_jepa_dim = int(self.jepa_proj.in_features)
            if int(jepa_local.shape[-1]) != expected_jepa_dim:
                actual_jepa_dim = int(jepa_local.shape[-1])
                if actual_jepa_dim % expected_jepa_dim == 0:
                    fold = actual_jepa_dim // expected_jepa_dim
                    jepa_local = jepa_local.reshape(*jepa_local.shape[:-1], fold, expected_jepa_dim).mean(dim=-2)
                else:
                    self._ensure_jepa_proj(actual_jepa_dim, jepa_local.device)
            jepa_tokens = self.jepa_proj(torch.nan_to_num(jepa_local, nan=0.0, posinf=0.0, neginf=0.0))
            self._ensure_latent_proj(latent_local.shape[-1], latent_local.device)
            latent_tokens = self.latent_proj(torch.nan_to_num(latent_local, nan=0.0, posinf=0.0, neginf=0.0))
            geom_feat = self.geom_proj(torch.nan_to_num(geom_steps, nan=0.0, posinf=0.0, neginf=0.0))
            geom_weights = torch.nan_to_num((visibility * confidence).unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
            if frame_valid_mask is not None:
                geom_weights = geom_weights * frame_valid_mask[:, :, None, None].to(dtype=geom_weights.dtype, device=geom_weights.device)
            geom_denom = geom_weights.sum(dim=1).clamp_min(1.0)
            geom_tokens = (geom_feat * geom_weights).sum(dim=1) / geom_denom
            vggt_geom_tokens = None
            if vggt_world_points is not None and vggt_depth is not None:
                world_local = self._pool_feature_grid(
                    vggt_world_points,
                    tracks,
                    image_hw=track_image_hw,
                    window_radius=0,
                    frame_valid_mask=frame_valid_mask,
                )
                depth_local = self._pool_feature_grid(
                    vggt_depth,
                    tracks,
                    image_hw=track_image_hw,
                    window_radius=0,
                    frame_valid_mask=frame_valid_mask,
                )
                world_local = torch.nan_to_num(world_local, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                    -self.vggt_world_clip, self.vggt_world_clip
                )
                depth_local = torch.nan_to_num(depth_local, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                    -self.vggt_depth_clip, self.vggt_depth_clip
                )
                world_conf_local = None
                depth_conf_local = None
                if vggt_world_points_conf is not None:
                    world_conf_local = self._pool_feature_grid(
                        vggt_world_points_conf,
                        tracks,
                        image_hw=track_image_hw,
                        window_radius=0,
                        frame_valid_mask=frame_valid_mask,
                    )
                if vggt_depth_conf is not None:
                    depth_conf_local = self._pool_feature_grid(
                        vggt_depth_conf.unsqueeze(-1),
                        tracks,
                        image_hw=track_image_hw,
                        window_radius=0,
                        frame_valid_mask=frame_valid_mask,
                    )
                if world_conf_local is None:
                    world_conf_local = torch.ones_like(depth_local)
                if depth_conf_local is None:
                    depth_conf_local = torch.ones_like(depth_local)
                world_conf_local = torch.nan_to_num(world_conf_local, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
                depth_conf_local = torch.nan_to_num(depth_conf_local, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
                vggt_geom_feat = torch.nan_to_num(torch.cat(
                    [
                        world_local,
                        depth_local,
                        0.5 * (world_conf_local + depth_conf_local),
                    ],
                    dim=-1,
                ), nan=0.0, posinf=0.0, neginf=0.0)
                vggt_geom_tokens = self.vggt_geom_proj(torch.nan_to_num(vggt_geom_feat, nan=0.0, posinf=0.0, neginf=0.0))

            fused_geom = geom_tokens if vggt_geom_tokens is None else (geom_tokens + vggt_geom_tokens)
            object_tokens = self.out_norm(torch.nan_to_num(jepa_tokens + latent_tokens + fused_geom, nan=0.0, posinf=0.0, neginf=0.0))
            object_tokens = torch.nan_to_num(object_tokens, nan=0.0, posinf=0.0, neginf=0.0)
            return ObjectTokenOutput(
                object_tokens=object_tokens.to(dtype=jepa_patch_tokens.dtype),
                jepa_tokens=jepa_tokens.to(dtype=jepa_patch_tokens.dtype),
                latent_tokens=latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                geom_tokens=geom_tokens.to(dtype=jepa_patch_tokens.dtype),
                vggt_geom_tokens=None if vggt_geom_tokens is None else vggt_geom_tokens.to(dtype=jepa_patch_tokens.dtype),
            )


def box_centers_to_tracks(
    boxes: torch.Tensor,
    image_hw: tuple[int, int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # boxes: [B,T,K,4], normalized xyxy in [0,1], zero box means invalid
    x0 = boxes[..., 0]
    y0 = boxes[..., 1]
    x1 = boxes[..., 2]
    y1 = boxes[..., 3]
    valid = (x1 - x0 > eps) & (y1 - y0 > eps)
    cx = 0.5 * (x0 + x1) * image_hw[1]
    cy = 0.5 * (y0 + y1) * image_hw[0]
    tracks = torch.stack([cx, cy], dim=-1)
    vis = valid.float()
    conf = valid.float()
    return tracks, vis, conf
