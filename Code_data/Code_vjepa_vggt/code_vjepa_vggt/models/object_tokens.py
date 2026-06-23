from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObjectTokenOutput:
    object_tokens: torch.Tensor
    object_latent_tokens: torch.Tensor
    jepa_tokens: torch.Tensor
    jepa_latent_tokens: torch.Tensor
    latent_tokens: torch.Tensor
    latent_latent_tokens: torch.Tensor
    geom_tokens: torch.Tensor
    track_geom_latent_tokens: torch.Tensor
    vggt_geom_tokens: torch.Tensor | None = None
    depth_latent_tokens: torch.Tensor | None = None
    world_latent_tokens: torch.Tensor | None = None
    active_track_summary: torch.Tensor | None = None


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
        self.jepa_window_radius = int(jepa_window_radius)
        self.latent_window_radius = int(latent_window_radius)
        self.out_dim = int(out_dim)
        self.jepa_proj = nn.Linear(int(jepa_dim), self.out_dim)
        self.latent_proj = nn.Linear(int(latent_dim), self.out_dim)
        self.track_geom_proj = nn.Sequential(
            nn.Linear(6, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(2, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.world_proj = nn.Sequential(
            nn.Linear(4, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.out_norm = nn.LayerNorm(self.out_dim)
        self.vggt_world_clip = 16.0
        self.vggt_depth_clip = 16.0

    def _ensure_latent_proj(self, latent_dim: int, device: torch.device) -> None:
        if self.latent_proj.in_features == int(latent_dim):
            return
        self.latent_proj = nn.Linear(int(latent_dim), self.out_dim).to(device)

    def _ensure_jepa_proj(self, jepa_dim: int, device: torch.device) -> None:
        if self.jepa_proj.in_features == int(jepa_dim):
            return
        self.jepa_proj = nn.Linear(int(jepa_dim), self.out_dim).to(device)

    @staticmethod
    def _time_indices(src_frames: int, dst_frames: int, device: torch.device) -> torch.Tensor:
        if int(dst_frames) <= 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return torch.linspace(0, int(src_frames) - 1, int(dst_frames), device=device).round().long()

    @staticmethod
    def _resize_tracks_xy(
        tracks: torch.Tensor,
        *,
        src_hw: tuple[int, int],
        dst_hw: tuple[int, int],
        align_corners: bool,
    ) -> torch.Tensor:
        if tuple(int(v) for v in src_hw) == tuple(int(v) for v in dst_hw):
            return tracks
        out = tracks.clone()
        src_h, src_w = int(src_hw[0]), int(src_hw[1])
        dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
        if align_corners:
            scale_x = float(max(dst_w - 1, 1)) / max(float(src_w - 1), 1.0)
            scale_y = float(max(dst_h - 1, 1)) / max(float(src_h - 1), 1.0)
            out[..., 0] *= scale_x
            out[..., 1] *= scale_y
            return out
        out[..., 0] = ((out[..., 0] + 0.5) * float(dst_w) / max(float(src_w), 1.0)) - 0.5
        out[..., 1] = ((out[..., 1] + 0.5) * float(dst_h) / max(float(src_h), 1.0)) - 0.5
        return out

    @staticmethod
    def _pool_feature_grid(
        features: torch.Tensor,
        tracks: torch.Tensor,
        image_hw: tuple[int, int],
        window_radius: int,
    ) -> torch.Tensor:
        if features.ndim == 4:
            features = features.unsqueeze(-1)
        batch, frames, grid_h, grid_w, dim = features.shape
        _, _, objects, _ = tracks.shape
        height, width = image_hw
        feature_map = features.permute(0, 1, 4, 2, 3).reshape(batch * frames, dim, grid_h, grid_w)
        if int(window_radius) > 0:
            kernel = 2 * int(window_radius) + 1
            feature_map = F.avg_pool2d(feature_map, kernel_size=kernel, stride=1, padding=int(window_radius))

        x = tracks[..., 0] / max(float(width - 1), 1.0)
        y = tracks[..., 1] / max(float(height - 1), 1.0)
        x = x.clamp(0.0, 1.0) * 2.0 - 1.0
        y = y.clamp(0.0, 1.0) * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).view(batch * frames, objects, 1, 2)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).permute(0, 2, 1).reshape(batch, frames, objects, dim)

    @staticmethod
    def _temporal_group_mean(values: torch.Tensor, target_frames: int, *, frame_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, src_frames, objects, dim = values.shape
        if int(src_frames) == int(target_frames):
            return values
        if int(src_frames) % int(target_frames) != 0:
            raise ValueError(f"cannot evenly group src_frames={src_frames} into target_frames={target_frames}")
        group = int(src_frames) // int(target_frames)
        values = values.view(batch, int(target_frames), group, objects, dim)
        if frame_valid_mask is None:
            return values.mean(dim=2)
        weights = frame_valid_mask.view(batch, int(target_frames), group, 1, 1).to(dtype=values.dtype, device=values.device)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (values * weights).sum(dim=2) / denom

    @staticmethod
    def _track_summary(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        image_hw: tuple[int, int],
        target_frames: int,
        *,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        height, width = int(image_hw[0]), int(image_hw[1])
        x = tracks[..., 0] / max(float(width - 1), 1.0)
        y = tracks[..., 1] / max(float(height - 1), 1.0)
        xy = torch.stack([x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)], dim=-1)
        batch, src_frames, objects, _ = xy.shape
        if int(src_frames) % int(target_frames) != 0:
            raise ValueError(f"cannot evenly summarize src_frames={src_frames} into target_frames={target_frames}")
        group = int(src_frames) // int(target_frames)
        xy = xy.view(batch, int(target_frames), group, objects, 2)
        vis = visibility.view(batch, int(target_frames), group, objects, 1)
        conf = confidence.view(batch, int(target_frames), group, objects, 1)
        if frame_valid_mask is None:
            weights = torch.ones(batch, int(target_frames), group, objects, 1, device=xy.device, dtype=xy.dtype)
        else:
            weights = frame_valid_mask.view(batch, int(target_frames), group, 1, 1).to(dtype=xy.dtype, device=xy.device)
            weights = weights.expand(-1, -1, -1, objects, -1)
        mean_xy = (xy * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        delta_xy = xy[:, :, -1] - xy[:, :, 0]
        mean_vis = (vis * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        mean_conf = (conf * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return torch.cat([mean_xy, delta_xy, mean_vis, mean_conf], dim=-1)

    @staticmethod
    def _confidence_group_mean(
        values: torch.Tensor,
        target_frames: int,
        *,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, src_frames, objects, dim = values.shape
        if int(src_frames) == int(target_frames):
            return values
        if int(src_frames) % int(target_frames) != 0:
            raise ValueError(f"cannot evenly group src_frames={src_frames} into target_frames={target_frames}")
        group = int(src_frames) // int(target_frames)
        values = values.view(batch, int(target_frames), group, objects, dim)
        if frame_valid_mask is None:
            return values.mean(dim=2)
        weights = frame_valid_mask.view(batch, int(target_frames), group, 1, 1).to(dtype=values.dtype, device=values.device)
        weights = weights.expand(-1, -1, -1, objects, -1)
        return (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

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
        vggt_geometry_image_hw: tuple[int, int] | None = None,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> ObjectTokenOutput:
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

            batch, src_frames, objects, _ = tracks.shape
            latent_frames = int(context_latents.shape[2])
            if int(src_frames) % max(latent_frames, 1) != 0:
                raise ValueError(
                    f"track frames ({src_frames}) must be divisible by latent frames ({latent_frames}) "
                    "for latent-time conditioning"
                )

            jepa_time_idx = self._time_indices(src_frames, int(jepa_patch_tokens.shape[1]), tracks.device)
            latent_time_idx = self._time_indices(src_frames, latent_frames, tracks.device)
            jepa_tracks = tracks[:, jepa_time_idx]
            latent_tracks = tracks[:, latent_time_idx]

            jepa_local = self._pool_feature_grid(
                jepa_patch_tokens,
                jepa_tracks,
                image_hw=track_image_hw,
                window_radius=self.jepa_window_radius,
            )
            if int(jepa_local.shape[1]) % max(latent_frames, 1) != 0:
                raise ValueError(
                    f"JEPA frames ({jepa_local.shape[1]}) must be divisible by latent frames ({latent_frames})"
                )
            jepa_valid = frame_valid_mask[:, jepa_time_idx] if frame_valid_mask is not None else None
            jepa_local = self._temporal_group_mean(jepa_local, latent_frames, frame_valid_mask=jepa_valid)
            expected_jepa_dim = int(self.jepa_proj.in_features)
            if int(jepa_local.shape[-1]) != expected_jepa_dim:
                actual_jepa_dim = int(jepa_local.shape[-1])
                if actual_jepa_dim % expected_jepa_dim == 0:
                    fold = actual_jepa_dim // expected_jepa_dim
                    jepa_local = jepa_local.reshape(*jepa_local.shape[:-1], fold, expected_jepa_dim).mean(dim=-2)
                else:
                    self._ensure_jepa_proj(actual_jepa_dim, jepa_local.device)
            jepa_latent_tokens = self.jepa_proj(jepa_local)

            latent_grid = context_latents.permute(0, 2, 3, 4, 1).contiguous()
            latent_local = self._pool_feature_grid(
                latent_grid,
                latent_tracks,
                image_hw=track_image_hw,
                window_radius=self.latent_window_radius,
            )
            self._ensure_latent_proj(int(latent_local.shape[-1]), latent_local.device)
            latent_latent_tokens = self.latent_proj(latent_local)

            active_track_summary = self._track_summary(
                tracks,
                visibility,
                confidence,
                image_hw=track_image_hw,
                target_frames=latent_frames,
                frame_valid_mask=frame_valid_mask,
            )
            track_geom_latent_tokens = self.track_geom_proj(active_track_summary)

            depth_latent_tokens = None
            world_latent_tokens = None
            if vggt_world_points is not None and vggt_depth is not None:
                geometry_image_hw = (
                    tuple(int(v) for v in vggt_geometry_image_hw)
                    if vggt_geometry_image_hw is not None
                    else tuple(int(v) for v in track_image_hw)
                )
                geometry_tracks = self._resize_tracks_xy(
                    tracks,
                    src_hw=track_image_hw,
                    dst_hw=geometry_image_hw,
                    align_corners=False,
                )
                world_local = self._pool_feature_grid(
                    vggt_world_points,
                    geometry_tracks,
                    image_hw=geometry_image_hw,
                    window_radius=0,
                ).clamp(-self.vggt_world_clip, self.vggt_world_clip)
                depth_local = self._pool_feature_grid(
                    vggt_depth,
                    geometry_tracks,
                    image_hw=geometry_image_hw,
                    window_radius=0,
                ).clamp(-self.vggt_depth_clip, self.vggt_depth_clip)
                depth_conf_local = None
                world_conf_local = None
                if vggt_depth_conf is not None:
                    depth_conf_local = self._pool_feature_grid(
                        vggt_depth_conf.unsqueeze(-1),
                        geometry_tracks,
                        image_hw=geometry_image_hw,
                        window_radius=0,
                    ).clamp(0.0, 1.0)
                if vggt_world_points_conf is not None:
                    world_conf_local = self._pool_feature_grid(
                        vggt_world_points_conf,
                        geometry_tracks,
                        image_hw=geometry_image_hw,
                        window_radius=0,
                    ).clamp(0.0, 1.0)
                if depth_conf_local is None:
                    depth_conf_local = torch.ones_like(depth_local)
                if world_conf_local is None:
                    world_conf_local = torch.ones_like(depth_local)
                depth_lat = self._confidence_group_mean(depth_local, latent_frames, frame_valid_mask=frame_valid_mask)
                depth_conf_lat = self._confidence_group_mean(depth_conf_local, latent_frames, frame_valid_mask=frame_valid_mask)
                world_lat = self._confidence_group_mean(world_local, latent_frames, frame_valid_mask=frame_valid_mask)
                world_conf_lat = self._confidence_group_mean(world_conf_local, latent_frames, frame_valid_mask=frame_valid_mask)
                depth_latent_tokens = self.depth_proj(torch.cat([depth_lat, depth_conf_lat], dim=-1))
                world_latent_tokens = self.world_proj(torch.cat([world_lat, world_conf_lat], dim=-1))

            fused = jepa_latent_tokens + latent_latent_tokens + track_geom_latent_tokens
            if depth_latent_tokens is not None:
                fused = fused + depth_latent_tokens
            if world_latent_tokens is not None:
                fused = fused + world_latent_tokens
            object_latent_tokens = self.out_norm(torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0))
            object_tokens = object_latent_tokens.mean(dim=1)
            jepa_tokens = jepa_latent_tokens.mean(dim=1)
            latent_tokens = latent_latent_tokens.mean(dim=1)
            geom_tokens = track_geom_latent_tokens.mean(dim=1)
            vggt_geom_tokens = None
            if depth_latent_tokens is not None or world_latent_tokens is not None:
                parts = []
                if depth_latent_tokens is not None:
                    parts.append(depth_latent_tokens)
                if world_latent_tokens is not None:
                    parts.append(world_latent_tokens)
                vggt_geom_tokens = torch.stack(parts, dim=0).sum(dim=0).mean(dim=1)
            return ObjectTokenOutput(
                object_tokens=object_tokens.to(dtype=jepa_patch_tokens.dtype),
                object_latent_tokens=object_latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                jepa_tokens=jepa_tokens.to(dtype=jepa_patch_tokens.dtype),
                jepa_latent_tokens=jepa_latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                latent_tokens=latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                latent_latent_tokens=latent_latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                geom_tokens=geom_tokens.to(dtype=jepa_patch_tokens.dtype),
                track_geom_latent_tokens=track_geom_latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                vggt_geom_tokens=None if vggt_geom_tokens is None else vggt_geom_tokens.to(dtype=jepa_patch_tokens.dtype),
                depth_latent_tokens=None if depth_latent_tokens is None else depth_latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                world_latent_tokens=None if world_latent_tokens is None else world_latent_tokens.to(dtype=jepa_patch_tokens.dtype),
                active_track_summary=active_track_summary.to(dtype=jepa_patch_tokens.dtype),
            )


def box_centers_to_tracks(
    boxes: torch.Tensor,
    image_hw: tuple[int, int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
