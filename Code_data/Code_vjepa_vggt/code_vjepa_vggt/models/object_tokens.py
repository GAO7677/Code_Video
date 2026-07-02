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
    motion_latent_tokens: torch.Tensor | None = None
    active_track_summary: torch.Tensor | None = None
    active_box_xyxy: torch.Tensor | None = None


class ObjectTubeProjector(nn.Module):
    def __init__(
        self,
        jepa_dim: int,
        latent_dim: int,
        out_dim: int,
        vggt_dense_dim: int = 2048,
        jepa_window_radius: int = 1,
        latent_window_radius: int = 1,
        min_box_px: float = 16.0,
    ) -> None:
        super().__init__()
        self.jepa_window_radius = int(jepa_window_radius)
        self.latent_window_radius = int(latent_window_radius)
        self.min_box_px = float(min_box_px)
        self.out_dim = int(out_dim)
        self.jepa_proj = nn.Linear(int(jepa_dim), self.out_dim)
        self.latent_proj = nn.Linear(int(latent_dim), self.out_dim)
        self.track_geom_proj = nn.Sequential(
            nn.Linear(6, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.vggt_geom_point_proj = nn.Sequential(
            nn.Linear(int(vggt_dense_dim) + 1 + 6, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.motion_point_proj = nn.Sequential(
            nn.Linear(6, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.motion_router_score = nn.Linear(self.out_dim, 1)
        self.geom_router_score = nn.Linear(self.out_dim, 1)
        self.jepa_router_score = nn.Linear(self.out_dim, 1)
        self.latent_router_score = nn.Linear(self.out_dim, 1)
        self.track_geometry_router_score = nn.Linear(self.out_dim, 1)
        self.appearance_router_score = nn.Linear(self.out_dim, 1)
        self.modal_refine = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
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

    @staticmethod
    def _rebuild_linear_preserving_state(
        module: nn.Linear,
        *,
        in_features: int,
        out_features: int,
        device: torch.device,
    ) -> nn.Linear:
        new_module = nn.Linear(
            int(in_features),
            int(out_features),
            bias=module.bias is not None,
        ).to(device=device, dtype=module.weight.dtype)
        new_module.train(module.training)
        requires_grad = any(param.requires_grad for param in module.parameters())
        new_module.requires_grad_(requires_grad)
        return new_module

    def _ensure_latent_proj(self, latent_dim: int, device: torch.device) -> None:
        if self.latent_proj.in_features == int(latent_dim):
            return
        self.latent_proj = self._rebuild_linear_preserving_state(
            self.latent_proj,
            in_features=int(latent_dim),
            out_features=self.out_dim,
            device=device,
        )

    def _ensure_jepa_proj(self, jepa_dim: int, device: torch.device) -> None:
        if self.jepa_proj.in_features == int(jepa_dim):
            return
        self.jepa_proj = self._rebuild_linear_preserving_state(
            self.jepa_proj,
            in_features=int(jepa_dim),
            out_features=self.out_dim,
            device=device,
        )

    @staticmethod
    def _grid_feature_hw(feature_grid: torch.Tensor) -> tuple[int, int]:
        if feature_grid.ndim != 5:
            raise ValueError(f"feature_grid must have shape [B,T,H,W,C], got {list(feature_grid.shape)}")
        return int(feature_grid.shape[2]), int(feature_grid.shape[3])

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
        scale_x = float(max(dst_w - 1, 1)) / max(float(src_w - 1), 1.0)
        scale_y = float(max(dst_h - 1, 1)) / max(float(src_h - 1), 1.0)
        out[..., 0] *= scale_x
        out[..., 1] *= scale_y
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
    def _ensure_grouped_tracks(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tracks.ndim == 4:
            tracks = tracks.unsqueeze(3)
            visibility = visibility.unsqueeze(3)
            confidence = confidence.unsqueeze(3)
            return tracks, visibility, confidence
        if tracks.ndim != 5:
            raise ValueError(
                f"tracks must have shape [B,T,O,2] or [B,T,O,P,2], got {list(tracks.shape)}"
            )
        if visibility.ndim != 4 or confidence.ndim != 4:
            raise ValueError(
                "visibility/confidence must match grouped tracks shape [B,T,O,P], "
                f"got visibility={list(visibility.shape)}, confidence={list(confidence.shape)}"
            )
        return tracks, visibility, confidence

    @staticmethod
    def _flatten_point_axis(values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, frames, objects, points = values.shape[:4]
        suffix = values.shape[4:]
        flat = values.reshape(batch, frames, objects * points, *suffix)
        return flat, objects, points

    @staticmethod
    def _restore_point_axis(values: torch.Tensor, objects: int, points: int) -> torch.Tensor:
        batch, frames = values.shape[:2]
        suffix = values.shape[3:]
        return values.reshape(batch, frames, objects, points, *suffix)

    @staticmethod
    def _point_weights(
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        *,
        target_frames: int | None = None,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = (visibility * confidence).clamp_min(0.0).unsqueeze(-1)
        if target_frames is None or int(weights.shape[1]) == int(target_frames):
            return weights
        batch, src_frames, objects, points, _ = weights.shape
        if int(src_frames) % max(int(target_frames), 1) != 0:
            raise ValueError(f"cannot evenly group src_frames={src_frames} into target_frames={target_frames}")
        group = int(src_frames) // int(target_frames)
        weights = weights.view(batch, int(target_frames), group, objects, points, 1)
        if frame_valid_mask is not None:
            time_weights = frame_valid_mask.view(batch, int(target_frames), group, 1, 1, 1).to(
                dtype=weights.dtype,
                device=weights.device,
            )
            weights = weights * time_weights
        return weights.mean(dim=2)

    @staticmethod
    def _aggregate_points(
        values: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(f"values must have shape [B,T,O,P,D], got {list(values.shape)}")
        if weights is None:
            return values.mean(dim=3)
        if weights.ndim != 5:
            raise ValueError(f"weights must have shape [B,T,O,P,1], got {list(weights.shape)}")
        denom = weights.sum(dim=3).clamp_min(1.0e-6)
        return (values * weights).sum(dim=3) / denom

    def _point_attention_pool(
        self,
        point_tokens: torch.Tensor,
        score_head: nn.Module,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if point_tokens.ndim != 5:
            raise ValueError(
                f"point_tokens must have shape [B,T,O,P,D], got {list(point_tokens.shape)}"
            )
        logits = score_head(point_tokens)
        if weights is not None:
            valid = weights > 1.0e-6
            logits = logits.masked_fill(~valid, -1.0e4)
        attn = torch.softmax(logits, dim=3)
        if weights is not None:
            attn = attn * weights
            attn = attn / attn.sum(dim=3, keepdim=True).clamp_min(1.0e-6)
        return (point_tokens * attn).sum(dim=3)

    def _modality_fuse(
        self,
        track_geometry_tokens: torch.Tensor,
        appearance_tokens: torch.Tensor,
    ) -> torch.Tensor:
        gate_logits = torch.cat(
            [
                self.track_geometry_router_score(track_geometry_tokens),
                self.appearance_router_score(appearance_tokens),
            ],
            dim=-1,
        )
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused = (
            gate_weights[..., 0:1] * track_geometry_tokens
            + gate_weights[..., 1:2] * appearance_tokens
        )
        return self.out_norm(fused + self.modal_refine(fused))

    @staticmethod
    def _pair_fuse(
        tokens_a: torch.Tensor,
        tokens_b: torch.Tensor,
        score_a: nn.Module,
        score_b: nn.Module,
    ) -> torch.Tensor:
        gate_logits = torch.cat(
            [
                score_a(tokens_a),
                score_b(tokens_b),
            ],
            dim=-1,
        )
        gate_weights = torch.softmax(gate_logits, dim=-1)
        fused = (
            gate_weights[..., 0:1] * tokens_a
            + gate_weights[..., 1:2] * tokens_b
        )
        return fused

    @staticmethod
    def _temporal_group_mean_grouped(
        values: torch.Tensor,
        target_frames: int,
        *,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(f"values must have shape [B,T,O,P,D], got {list(values.shape)}")
        batch, src_frames, objects, points, dim = values.shape
        if int(src_frames) == int(target_frames):
            return values
        if int(src_frames) % int(target_frames) != 0:
            raise ValueError(f"cannot evenly group src_frames={src_frames} into target_frames={target_frames}")
        group = int(src_frames) // int(target_frames)
        values = values.view(batch, int(target_frames), group, objects, points, dim)
        if frame_valid_mask is None:
            return values.mean(dim=2)
        weights = frame_valid_mask.view(batch, int(target_frames), group, 1, 1, 1).to(
            dtype=values.dtype,
            device=values.device,
        )
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (values * weights).sum(dim=2) / denom

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
        weights = (vis * conf).clamp_min(0.0)
        if frame_valid_mask is not None:
            weights = weights * frame_valid_mask.view(batch, int(target_frames), group, 1, 1).to(dtype=xy.dtype, device=xy.device)
        valid = weights.squeeze(-1) > 1.0e-6
        if not bool(valid.any().item()):
            valid = torch.ones_like(valid)
        point_weights = valid.to(dtype=xy.dtype).unsqueeze(-1)
        trimmed_xy = torch.where(valid.unsqueeze(-1), xy, torch.nan)
        if int(xy.shape[2]) >= 4:
            sorted_x = torch.nan_to_num(trimmed_xy[..., 0], nan=1.0).sort(dim=2).values
            sorted_y = torch.nan_to_num(trimmed_xy[..., 1], nan=1.0).sort(dim=2).values
            valid_count = valid.sum(dim=2)
            trim = torch.clamp((valid_count - 4) // 2, min=0, max=max(int(xy.shape[2] // 2), 0))
            min_idx = trim.unsqueeze(-1).clamp(max=int(xy.shape[2]) - 1)
            max_idx = (valid_count - 1 - trim).clamp(min=0, max=int(xy.shape[2]) - 1).unsqueeze(-1)
            center_x = 0.5 * (
                torch.gather(sorted_x, dim=2, index=min_idx).squeeze(-1)
                + torch.gather(sorted_x, dim=2, index=max_idx).squeeze(-1)
            )
            center_y = 0.5 * (
                torch.gather(sorted_y, dim=2, index=min_idx).squeeze(-1)
                + torch.gather(sorted_y, dim=2, index=max_idx).squeeze(-1)
            )
            center_xy = torch.stack([center_x, center_y], dim=-1)
        else:
            center_xy = (xy * point_weights).sum(dim=2) / point_weights.sum(dim=2).clamp_min(1.0e-6)

        valid_group = valid.permute(0, 1, 3, 2)  # [B,T,O,G]
        first_idx = valid_group.float().argmax(dim=-1)
        last_idx = group - 1 - valid_group.flip(dims=[-1]).float().argmax(dim=-1)
        no_valid = valid_group.any(dim=-1) == 0
        first_idx = torch.where(no_valid, torch.zeros_like(first_idx), first_idx)
        last_idx = torch.where(no_valid, torch.zeros_like(last_idx), last_idx)

        xy_perm = xy.permute(0, 1, 3, 2, 4)  # [B,T,O,G,2]
        gather_first = first_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2).long()
        gather_last = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 2).long()
        first_xy = torch.gather(xy_perm, dim=3, index=gather_first).squeeze(3)
        last_xy = torch.gather(xy_perm, dim=3, index=gather_last).squeeze(3)
        delta_xy = last_xy - first_xy
        mean_vis = (vis * point_weights).sum(dim=2) / point_weights.sum(dim=2).clamp_min(1.0e-6)
        mean_conf = (conf * point_weights).sum(dim=2) / point_weights.sum(dim=2).clamp_min(1.0e-6)
        # Keep the summary center aligned with the last valid observation so it
        # matches the training-side GT grouping semantics.
        return torch.cat([center_xy, delta_xy, mean_vis, mean_conf], dim=-1)

    @staticmethod
    def _boxes_from_summary(
        active_track_summary: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        radius_px: float = 12.0,
        box_prior_xyxy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if box_prior_xyxy is not None:
            return torch.nan_to_num(box_prior_xyxy.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        height, width = int(image_hw[0]), int(image_hw[1])
        center_xy = active_track_summary[..., :2]
        radius_x = float(radius_px) / max(float(width - 1), 1.0)
        radius_y = float(radius_px) / max(float(height - 1), 1.0)
        half_wh = center_xy.new_tensor([radius_x, radius_y]).view(1, 1, 1, 2)
        return torch.cat(
            [
                (center_xy - half_wh).clamp(0.0, 1.0),
                (center_xy + half_wh).clamp(0.0, 1.0),
            ],
            dim=-1,
        )

    @staticmethod
    def _boxes_from_tracks(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        target_frames: int | None = None,
        box_prior_xyxy: torch.Tensor | None = None,
        expand_ratio: float = 0.15,
        min_box_px: float = 16.0,
    ) -> torch.Tensor:
        prior = None
        if target_frames is not None and int(tracks.shape[1]) != int(target_frames):
            if int(tracks.shape[1]) % int(target_frames) != 0:
                raise ValueError(
                    f"track frames ({int(tracks.shape[1])}) must be divisible by target_frames ({int(target_frames)})"
                )
            group = int(tracks.shape[1]) // int(target_frames)
            tracks = tracks.view(tracks.shape[0], int(target_frames), group, tracks.shape[2], tracks.shape[3], tracks.shape[4]).mean(dim=2)
            visibility = visibility.view(visibility.shape[0], int(target_frames), group, visibility.shape[2], visibility.shape[3]).mean(dim=2)
            confidence = confidence.view(confidence.shape[0], int(target_frames), group, confidence.shape[2], confidence.shape[3]).mean(dim=2)
        height, width = int(image_hw[0]), int(image_hw[1])
        x = tracks[..., 0] / max(float(width - 1), 1.0)
        y = tracks[..., 1] / max(float(height - 1), 1.0)
        valid = (visibility * confidence).clamp_min(0.0) > 1.0e-6
        valid_any = valid.any(dim=3)
        if box_prior_xyxy is not None:
            prior = torch.nan_to_num(box_prior_xyxy.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            if prior.ndim == 3:
                prior = prior[:, None].expand(-1, tracks.shape[1], -1, -1)
            elif prior.ndim == 4:
                prior = prior
            else:
                raise ValueError(f"box_prior_xyxy must have shape [B,O,4] or [B,T,O,4], got {list(prior.shape)}")
        # NOTE: previously this returned `prior` directly when a box prior existed,
        # which broadcast a single (context) box to ALL latent frames -> a STATIC
        # anchor that cannot track object motion into future frames (box aux loss
        # then plateaus, dominated by future frames). We now fall through to the
        # track-derived per-frame box below, which uses the dynamic track centers
        # and fuses the prior only as a width/height floor (box_wh = max(dynamic, prior)).
        if not bool(valid_any.any().item()):
            if prior is not None:
                return prior
            center_xy = tracks.new_zeros(tracks.shape[0], tracks.shape[1], tracks.shape[2], 2)
            half_wh = center_xy.new_tensor([0.02, 0.02]).view(1, 1, 1, 2)
            return torch.cat(
                [
                    (center_xy - half_wh).clamp(0.0, 1.0),
                    (center_xy + half_wh).clamp(0.0, 1.0),
                ],
                dim=-1,
            )
        sort_x = torch.where(valid, x, torch.full_like(x, float("inf"))).sort(dim=3).values
        sort_y = torch.where(valid, y, torch.full_like(y, float("inf"))).sort(dim=3).values
        valid_count = valid.sum(dim=3)
        trim = torch.clamp((valid_count - 2) // 2, min=0, max=max(int(tracks.shape[3] // 2), 0))
        min_idx = trim.clamp(max=int(tracks.shape[3]) - 1).unsqueeze(-1)
        max_idx = (valid_count - 1 - trim).clamp(min=0, max=int(tracks.shape[3]) - 1).unsqueeze(-1)
        x_min = torch.gather(sort_x, dim=3, index=min_idx).squeeze(-1)
        y_min = torch.gather(sort_y, dim=3, index=min_idx).squeeze(-1)
        x_max = torch.gather(sort_x, dim=3, index=max_idx).squeeze(-1)
        y_max = torch.gather(sort_y, dim=3, index=max_idx).squeeze(-1)
        # When a per-(frame,object) slot has no valid track points, sort filled it
        # with inf -> x_min=inf, x_max=inf -> span=nan. Replace with prior center
        # (if available) or a safe fallback so the anchor stays finite.
        no_valid_slot = ~valid_any  # [B, Lf, O]
        if prior is not None:
            prior_cx = 0.5 * (prior[..., 0] + prior[..., 2])
            prior_cy = 0.5 * (prior[..., 1] + prior[..., 3])
            x_min = torch.where(no_valid_slot, prior_cx, x_min)
            x_max = torch.where(no_valid_slot, prior_cx, x_max)
            y_min = torch.where(no_valid_slot, prior_cy, y_min)
            y_max = torch.where(no_valid_slot, prior_cy, y_max)
        else:
            x_min = torch.nan_to_num(x_min, nan=0.5, posinf=0.5)
            x_max = torch.nan_to_num(x_max, nan=0.5, posinf=0.5)
            y_min = torch.nan_to_num(y_min, nan=0.5, posinf=0.5)
            y_max = torch.nan_to_num(y_max, nan=0.5, posinf=0.5)
        span_x = (x_max - x_min).clamp_min(1.0e-4)
        span_y = (y_max - y_min).clamp_min(1.0e-4)
        pad_x = span_x * float(expand_ratio)
        pad_y = span_y * float(expand_ratio)
        min_box_wh = x.new_tensor(
            [
                float(min_box_px) / max(float(width - 1), 1.0),
                float(min_box_px) / max(float(height - 1), 1.0),
            ]
        )
        dynamic_wh = torch.stack([span_x + 2.0 * pad_x, span_y + 2.0 * pad_y], dim=-1)
        dynamic_wh = torch.maximum(dynamic_wh, min_box_wh.view(1, 1, 1, 2))
        if prior is not None:
            prior_wh = (prior[..., 2:] - prior[..., :2]).clamp_min(1.0e-4)
            box_wh = torch.maximum(dynamic_wh, prior_wh)
        else:
            box_wh = dynamic_wh
        half_wh = 0.5 * box_wh
        center_xy = torch.stack([0.5 * (x_min + x_max), 0.5 * (y_min + y_max)], dim=-1)
        active_box_xyxy = torch.stack(
            [
                (center_xy[..., 0] - half_wh[..., 0]).clamp(0.0, 1.0),
                (center_xy[..., 1] - half_wh[..., 1]).clamp(0.0, 1.0),
                (center_xy[..., 0] + half_wh[..., 0]).clamp(0.0, 1.0),
                (center_xy[..., 1] + half_wh[..., 1]).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return active_box_xyxy

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

    def _track_summary_grouped(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        image_hw: tuple[int, int],
        target_frames: int,
        *,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, src_frames, objects, points, _ = tracks.shape
        flat_tracks, _, _ = self._flatten_point_axis(tracks)
        flat_visibility, _, _ = self._flatten_point_axis(visibility.unsqueeze(-1))
        flat_confidence, _, _ = self._flatten_point_axis(confidence.unsqueeze(-1))
        point_summary = self._track_summary(
            flat_tracks,
            flat_visibility.squeeze(-1),
            flat_confidence.squeeze(-1),
            image_hw=image_hw,
            target_frames=target_frames,
            frame_valid_mask=frame_valid_mask,
        )
        point_summary = point_summary.view(batch, target_frames, objects, points, point_summary.shape[-1])
        point_weights = self._point_weights(
            visibility,
            confidence,
            target_frames=target_frames,
            frame_valid_mask=frame_valid_mask,
        )
        return self._aggregate_points(point_summary, point_weights)

    @staticmethod
    def _center_tracks_from_grouped(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (visibility * confidence).clamp_min(0.0)
        denom = weights.sum(dim=3, keepdim=True).clamp_min(1.0e-6)
        centers = (tracks * weights.unsqueeze(-1)).sum(dim=3) / denom
        valid = weights.sum(dim=3) > 1.0e-6
        return centers, valid

    @staticmethod
    def _select_first_last_valid(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = int(values.shape[2])
        valid_group = valid_mask.bool()
        any_valid = valid_group.any(dim=2)
        first_idx = valid_group.float().argmax(dim=2)
        last_idx = group - 1 - valid_group.flip(dims=[2]).float().argmax(dim=2)
        first_idx = torch.where(any_valid, first_idx, torch.zeros_like(first_idx))
        last_idx = torch.where(any_valid, last_idx, torch.zeros_like(last_idx))

        gather_shape = [*values.shape[:2], values.shape[3], 1, values.shape[-1]]
        values_perm = values.permute(0, 1, 3, 2, 4)
        first_gather = first_idx.unsqueeze(2).unsqueeze(-1).expand(*gather_shape).long()
        last_gather = last_idx.unsqueeze(2).unsqueeze(-1).expand(*gather_shape).long()
        first_values = torch.gather(values_perm, dim=3, index=first_gather).squeeze(3).permute(0, 1, 2, 3)
        last_values = torch.gather(values_perm, dim=3, index=last_gather).squeeze(3).permute(0, 1, 2, 3)
        return first_values, last_values

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
        vggt_world_points: torch.Tensor | None = None,
        vggt_world_points_conf: torch.Tensor | None = None,
        vggt_depth: torch.Tensor | None = None,
        vggt_depth_conf: torch.Tensor | None = None,
        vggt_dense_patch_tokens: torch.Tensor | None = None,
        vggt_patch_grid_hw: tuple[int, int] | None = None,
        vggt_geometry_image_hw: tuple[int, int] | None = None,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> ObjectTokenOutput:
        with torch.autocast(device_type=jepa_patch_tokens.device.type, enabled=False):
            jepa_patch_tokens = torch.nan_to_num(jepa_patch_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
            context_latents = torch.nan_to_num(context_latents.float(), nan=0.0, posinf=0.0, neginf=0.0)
            tracks = torch.nan_to_num(tracks.float(), nan=0.0, posinf=0.0, neginf=0.0)
            visibility = torch.nan_to_num(visibility.float(), nan=0.0, posinf=0.0, neginf=0.0)
            confidence = torch.nan_to_num(confidence.float(), nan=0.0, posinf=0.0, neginf=0.0)
            tracks, visibility, confidence = self._ensure_grouped_tracks(tracks, visibility, confidence)
            feature_device = tracks.device
            if object_valid_mask is not None:
                object_valid_mask = torch.nan_to_num(
                    object_valid_mask.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_world_points is not None:
                vggt_world_points = torch.nan_to_num(
                    vggt_world_points.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_world_points_conf is not None:
                vggt_world_points_conf = torch.nan_to_num(
                    vggt_world_points_conf.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_depth is not None:
                vggt_depth = torch.nan_to_num(vggt_depth.float(), nan=0.0, posinf=0.0, neginf=0.0).to(
                    device=feature_device
                )
            if vggt_depth_conf is not None:
                vggt_depth_conf = torch.nan_to_num(
                    vggt_depth_conf.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_dense_patch_tokens is not None:
                vggt_dense_patch_tokens = torch.nan_to_num(
                    vggt_dense_patch_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)

            batch, src_frames, objects, points, _ = tracks.shape
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
            flat_jepa_tracks, _, _ = self._flatten_point_axis(jepa_tracks)
            flat_latent_tracks, _, _ = self._flatten_point_axis(latent_tracks)

            jepa_local = self._pool_feature_grid(
                jepa_patch_tokens,
                flat_jepa_tracks,
                image_hw=track_image_hw,
                window_radius=self.jepa_window_radius,
            )
            if int(jepa_local.shape[1]) % max(latent_frames, 1) != 0:
                raise ValueError(
                    f"JEPA frames ({jepa_local.shape[1]}) must be divisible by latent frames ({latent_frames})"
                )
            jepa_valid = frame_valid_mask[:, jepa_time_idx] if frame_valid_mask is not None else None
            jepa_local = self._temporal_group_mean(jepa_local, latent_frames, frame_valid_mask=jepa_valid)
            jepa_local = self._restore_point_axis(jepa_local, objects, points)
            point_weights_lat = self._point_weights(
                visibility,
                confidence,
                target_frames=latent_frames,
                frame_valid_mask=frame_valid_mask,
            )
            jepa_local = self._aggregate_points(jepa_local, point_weights_lat)
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
                flat_latent_tracks,
                image_hw=track_image_hw,
                window_radius=self.latent_window_radius,
            )
            latent_local = self._restore_point_axis(latent_local, objects, points)
            latent_local = self._aggregate_points(latent_local, point_weights_lat)
            self._ensure_latent_proj(int(latent_local.shape[-1]), latent_local.device)
            latent_latent_tokens = self.latent_proj(latent_local)

            motion_xy = tracks[:, latent_time_idx]
            motion_xy = self._resize_tracks_xy(
                motion_xy,
                src_hw=track_image_hw,
                dst_hw=track_image_hw,
                align_corners=False,
            )
            motion_xy_norm = torch.stack(
                [
                    motion_xy[..., 0] / max(float(track_image_hw[1] - 1), 1.0),
                    motion_xy[..., 1] / max(float(track_image_hw[0] - 1), 1.0),
                ],
                dim=-1,
            ).clamp(0.0, 1.0)
            motion_delta = motion_xy_norm.clone()
            motion_delta[:, 1:] = motion_xy_norm[:, 1:] - motion_xy_norm[:, :-1]
            motion_delta[:, 0] = 0.0
            motion_local = torch.cat(
                [
                    motion_xy_norm,
                    motion_delta,
                    visibility[:, latent_time_idx].unsqueeze(-1),
                    confidence[:, latent_time_idx].unsqueeze(-1),
                ],
                dim=-1,
            )
            motion_point_tokens = self.motion_point_proj(motion_local)
            motion_latent_tokens = self._point_attention_pool(
                motion_point_tokens,
                self.motion_router_score,
                point_weights_lat,
            )

            center_tracks, center_track_valid = self._center_tracks_from_grouped(
                tracks,
                visibility,
                confidence,
            )
            active_track_summary = self._track_summary(
                center_tracks,
                center_track_valid.to(dtype=center_tracks.dtype),
                center_track_valid.to(dtype=center_tracks.dtype),
                image_hw=track_image_hw,
                target_frames=latent_frames,
                frame_valid_mask=frame_valid_mask,
            )
            active_box_xyxy = self._boxes_from_tracks(
                tracks,
                visibility,
                confidence,
                image_hw=track_image_hw,
                target_frames=latent_frames,
                box_prior_xyxy=box_prior_xyxy,
                min_box_px=self.min_box_px,
            )
            track_geom_latent_tokens = self.track_geom_proj(active_track_summary)

            depth_latent_tokens = None
            world_latent_tokens = None
            geom_latent_tokens = track_geom_latent_tokens
            vggt_geom_tokens = None
            if vggt_dense_patch_tokens is not None:
                geometry_patch_hw = (
                    tuple(int(v) for v in vggt_patch_grid_hw)
                    if vggt_patch_grid_hw is not None
                    else self._grid_feature_hw(vggt_dense_patch_tokens)
                )
                geometry_tracks = self._resize_tracks_xy(
                    tracks.reshape(batch, src_frames, objects * points, 2),
                    src_hw=track_image_hw,
                    dst_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                    align_corners=False,
                ).view(batch, src_frames, objects, points, 2)
                patch_tracks = self._resize_tracks_xy(
                    geometry_tracks.reshape(batch, src_frames, objects * points, 2),
                    src_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                    dst_hw=geometry_patch_hw,
                    align_corners=False,
                ).view(batch, src_frames, objects, points, 2)
                flat_patch_tracks, _, _ = self._flatten_point_axis(patch_tracks)
                geom_local = self._pool_feature_grid(
                    vggt_dense_patch_tokens,
                    flat_patch_tracks,
                    image_hw=geometry_patch_hw,
                    window_radius=0,
                )
                geom_local = self._restore_point_axis(geom_local, objects, points)
                geom_local = self._temporal_group_mean_grouped(
                    geom_local,
                    latent_frames,
                    frame_valid_mask=frame_valid_mask,
                )
                flat_geometry_tracks, _, _ = self._flatten_point_axis(geometry_tracks)
                depth_local = None
                if vggt_depth is not None:
                    depth_local = self._pool_feature_grid(
                        vggt_depth,
                        flat_geometry_tracks,
                        image_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                        window_radius=0,
                    ).clamp(-self.vggt_depth_clip, self.vggt_depth_clip)
                    depth_local = self._restore_point_axis(depth_local, objects, points)
                    depth_local = self._temporal_group_mean_grouped(
                        depth_local,
                        latent_frames,
                        frame_valid_mask=frame_valid_mask,
                    )
                else:
                    depth_local = geom_local.new_zeros(*geom_local.shape[:-1], 1)
                motion_local_lat = motion_local
                if int(motion_local_lat.shape[1]) != int(latent_frames):
                    motion_local_lat = self._temporal_group_mean_grouped(
                        motion_local_lat,
                        latent_frames,
                        frame_valid_mask=frame_valid_mask,
                    )
                geom_point_features = torch.cat(
                    [geom_local, depth_local, motion_local_lat],
                    dim=-1,
                )
                geom_point_tokens = self.vggt_geom_point_proj(geom_point_features)
                geom_latent_tokens = self._point_attention_pool(
                    geom_point_tokens,
                    self.geom_router_score,
                    point_weights_lat,
                )
                vggt_geom_tokens = geom_latent_tokens.mean(dim=1)

            track_geom_latent_tokens = self._pair_fuse(
                motion_latent_tokens,
                geom_latent_tokens,
                self.motion_router_score,
                self.geom_router_score,
            )
            appearance_latent_tokens = self._pair_fuse(
                jepa_latent_tokens,
                latent_latent_tokens,
                self.jepa_router_score,
                self.latent_router_score,
            )
            if vggt_depth is not None:
                geometry_image_hw = (
                    tuple(int(v) for v in vggt_geometry_image_hw)
                    if vggt_geometry_image_hw is not None
                    else tuple(int(v) for v in track_image_hw)
                )
                geometry_tracks = self._resize_tracks_xy(
                    tracks.reshape(batch, src_frames, objects * points, 2),
                    src_hw=track_image_hw,
                    dst_hw=geometry_image_hw,
                    align_corners=False,
                )
                geometry_tracks = geometry_tracks.view(batch, src_frames, objects, points, 2)
                flat_geometry_tracks, _, _ = self._flatten_point_axis(geometry_tracks)
                depth_local = self._pool_feature_grid(
                    vggt_depth,
                    flat_geometry_tracks,
                    image_hw=geometry_image_hw,
                    window_radius=0,
                ).clamp(-self.vggt_depth_clip, self.vggt_depth_clip)
                depth_local = self._restore_point_axis(depth_local, objects, points)
                if vggt_depth_conf is not None:
                    depth_conf_local = self._pool_feature_grid(
                        vggt_depth_conf.unsqueeze(-1),
                        flat_geometry_tracks,
                        image_hw=geometry_image_hw,
                        window_radius=0,
                    ).clamp(0.0, 1.0)
                    depth_conf_local = self._restore_point_axis(depth_conf_local, objects, points)
                else:
                    depth_conf_local = torch.ones_like(depth_local)
                depth_local = self._temporal_group_mean_grouped(
                    depth_local,
                    latent_frames,
                    frame_valid_mask=frame_valid_mask,
                )
                depth_conf_local = self._temporal_group_mean_grouped(
                    depth_conf_local,
                    latent_frames,
                    frame_valid_mask=frame_valid_mask,
                )
                depth_local = self._aggregate_points(depth_local, point_weights_lat)
                depth_conf_local = self._aggregate_points(depth_conf_local, point_weights_lat)
                depth_latent_tokens = self.depth_proj(torch.cat([depth_local, depth_conf_local], dim=-1))

            object_latent_tokens = self._modality_fuse(
                track_geom_latent_tokens,
                appearance_latent_tokens,
            )
            if object_valid_mask is not None:
                slot_mask = object_valid_mask[:, None, :, None].to(dtype=object_latent_tokens.dtype, device=object_latent_tokens.device)
                object_latent_tokens = object_latent_tokens * slot_mask
                jepa_latent_tokens = jepa_latent_tokens * slot_mask
                latent_latent_tokens = latent_latent_tokens * slot_mask
                motion_latent_tokens = motion_latent_tokens * slot_mask
                geom_latent_tokens = geom_latent_tokens * slot_mask
                track_geom_latent_tokens = track_geom_latent_tokens * slot_mask
                active_track_summary = active_track_summary * slot_mask
                active_box_xyxy = active_box_xyxy * slot_mask
                if depth_latent_tokens is not None:
                    depth_latent_tokens = depth_latent_tokens * slot_mask
            object_tokens = object_latent_tokens.mean(dim=1)
            jepa_tokens = jepa_latent_tokens.mean(dim=1)
            latent_tokens = latent_latent_tokens.mean(dim=1)
            geom_tokens = geom_latent_tokens.mean(dim=1)
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
                motion_latent_tokens=motion_latent_tokens.mean(dim=1).to(dtype=jepa_patch_tokens.dtype),
                active_track_summary=active_track_summary.to(dtype=jepa_patch_tokens.dtype),
                active_box_xyxy=active_box_xyxy.to(dtype=jepa_patch_tokens.dtype),
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
