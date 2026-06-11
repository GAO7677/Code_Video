from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ObjectTokenOutput:
    object_tokens: torch.Tensor
    jepa_tokens: torch.Tensor
    latent_tokens: torch.Tensor
    geom_tokens: torch.Tensor


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
        self.jepa_proj = nn.Linear(jepa_dim, out_dim)
        self.latent_proj = nn.Linear(latent_dim, out_dim)
        self.geom_proj = nn.Sequential(
            nn.Linear(4, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

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
        x = tracks[..., 0] / max(width - 1, 1)
        y = tracks[..., 1] / max(height - 1, 1)
        return torch.stack([x, y, visibility, confidence], dim=-1)

    def _pool_feature_grid(
        self,
        features: torch.Tensor,
        tracks: torch.Tensor,
        image_hw: tuple[int, int],
        window_radius: int,
    ) -> torch.Tensor:
        batch, frames, grid_h, grid_w, dim = features.shape
        _, _, objects, _ = tracks.shape
        pooled = features.new_zeros(batch, objects, dim)
        height, width = image_hw

        for b in range(batch):
            for k in range(objects):
                token_list = []
                for t in range(frames):
                    x = tracks[b, t, k, 0]
                    y = tracks[b, t, k, 1]
                    gx = int(torch.clamp(torch.round(x / max(width, 1) * (grid_w - 1)), 0, grid_w - 1).item())
                    gy = int(torch.clamp(torch.round(y / max(height, 1) * (grid_h - 1)), 0, grid_h - 1).item())
                    x0 = max(0, gx - window_radius)
                    x1 = min(grid_w, gx + window_radius + 1)
                    y0 = max(0, gy - window_radius)
                    y1 = min(grid_h, gy + window_radius + 1)
                    token_list.append(features[b, t, y0:y1, x0:x1].reshape(-1, dim).mean(dim=0))
                pooled[b, k] = torch.stack(token_list, dim=0).mean(dim=0)
        return pooled

    def forward(
        self,
        jepa_patch_tokens: torch.Tensor,
        context_latents: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        track_image_hw: tuple[int, int],
    ) -> ObjectTokenOutput:
        jepa_time_idx = self._time_indices(tracks.shape[1], jepa_patch_tokens.shape[1], tracks.device)
        latent_time_idx = self._time_indices(tracks.shape[1], context_latents.shape[2], tracks.device)

        jepa_tracks = tracks[:, jepa_time_idx]
        latent_tracks = tracks[:, latent_time_idx]
        geom_steps = self._normalized_tracks(tracks, visibility, confidence, track_image_hw)

        jepa_local = self._pool_feature_grid(
            jepa_patch_tokens,
            jepa_tracks,
            image_hw=track_image_hw,
            window_radius=self.jepa_window_radius,
        )

        latent_grid = context_latents.permute(0, 2, 3, 4, 1).contiguous()
        latent_local = self._pool_feature_grid(
            latent_grid,
            latent_tracks,
            image_hw=track_image_hw,
            window_radius=self.latent_window_radius,
        )

        jepa_tokens = self.jepa_proj(jepa_local)
        latent_tokens = self.latent_proj(latent_local)
        geom_tokens = self.geom_proj(geom_steps).mean(dim=1)
        object_tokens = self.out_norm(jepa_tokens + latent_tokens + geom_tokens)
        return ObjectTokenOutput(
            object_tokens=object_tokens,
            jepa_tokens=jepa_tokens,
            latent_tokens=latent_tokens,
            geom_tokens=geom_tokens,
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
