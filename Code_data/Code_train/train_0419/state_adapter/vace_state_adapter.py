"""State-conditioned adapter that turns future 9D object states into VACE context volumes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class VaceStateAdapter(torch.nn.Module):
    def __init__(
        self,
        state_dim: int = 9,
        pose_dim: int = 9,
        spatial_feature_dim: int = 8,
        hidden_dim: int = 128,
        vace_in_dim: int = 96,
        condition_dropout: float = 0.1,
        state_is_normalized: bool = True,
        use_temporal_encoding: bool = True,
        temporal_embed_dim: int = 32,
        depth_log_scale: float = 4.0,
        velocity_clip: float = 0.5,
        depth_velocity_clip: float = 0.1,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.pose_dim = int(pose_dim)
        self.spatial_feature_dim = int(spatial_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.vace_in_dim = int(vace_in_dim)
        self.condition_dropout = float(condition_dropout)
        self.state_is_normalized = bool(state_is_normalized)
        self.use_temporal_encoding = bool(use_temporal_encoding)
        self.temporal_embed_dim = int(temporal_embed_dim)
        self.depth_log_scale = float(depth_log_scale)
        self.velocity_clip = float(velocity_clip)
        self.depth_velocity_clip = float(depth_velocity_clip)

        self.spatial_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(self.spatial_feature_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(self.hidden_dim, self.vace_in_dim, kernel_size=3, padding=1),
        )
        self.temporal_projection = torch.nn.Sequential(
            torch.nn.Linear(self.temporal_embed_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.spatial_feature_dim),
        )
        self.pose_projection = torch.nn.Sequential(
            torch.nn.Linear(self.pose_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.spatial_feature_dim),
        )
        self.context_phase_embedding = torch.nn.Parameter(torch.zeros(self.spatial_feature_dim))
        self.future_phase_embedding = torch.nn.Parameter(torch.zeros(self.spatial_feature_dim))
        torch.nn.init.zeros_(self.spatial_encoder[-1].weight)
        torch.nn.init.zeros_(self.spatial_encoder[-1].bias)

    def _normalize_geometry(
        self,
        oracle_state: torch.Tensor,
        frame_width: int,
        frame_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = oracle_state[..., 0]
        v = oracle_state[..., 1]
        w = oracle_state[..., 3]
        h = oracle_state[..., 4]
        if self.state_is_normalized:
            return u, v, w, h
        width = max(int(frame_width), 1)
        height = max(int(frame_height), 1)
        return u / width, v / height, w / width, h / height

    def _maybe_dropout(self, maps: torch.Tensor) -> torch.Tensor:
        if self.training and self.condition_dropout > 0.0:
            if torch.rand((), device=maps.device) < self.condition_dropout:
                return torch.zeros_like(maps)
        return maps

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.clamp(min=0.0)
        if self.depth_log_scale <= 0.0:
            return depth
        scale = depth.new_tensor(self.depth_log_scale)
        return torch.log1p(depth * scale) / math.log1p(float(self.depth_log_scale))

    def _normalize_signed(self, value: torch.Tensor, clip: float) -> torch.Tensor:
        if clip <= 0.0:
            return value
        clip_value = value.new_tensor(float(clip))
        return value.clamp(min=-clip_value, max=clip_value) / clip_value

    def _build_temporal_features(
        self,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_frames <= 0:
            return torch.zeros((0, self.spatial_feature_dim), device=device, dtype=dtype)
        if not self.use_temporal_encoding:
            return torch.zeros((num_frames, self.spatial_feature_dim), device=device, dtype=dtype)

        position = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
        half_dim = max(self.temporal_embed_dim // 2, 1)
        freq_ids = torch.arange(half_dim, device=device, dtype=dtype)
        denom = torch.pow(position.new_tensor(10000.0), freq_ids / max(half_dim - 1, 1))
        angles = position.unsqueeze(1) / denom.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < self.temporal_embed_dim:
            emb = F.pad(emb, (0, self.temporal_embed_dim - emb.shape[1]))
        elif emb.shape[1] > self.temporal_embed_dim:
            emb = emb[:, : self.temporal_embed_dim]
        return self.temporal_projection(emb)

    def _build_spatial_state_maps(
        self,
        oracle_state: torch.Tensor,
        latent_height: int,
        latent_width: int,
        frame_height: int,
        frame_width: int,
        oracle_visibility: torch.Tensor | None = None,
        phase_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if oracle_state.dim() == 3:
            oracle_state = oracle_state.unsqueeze(0)
        if oracle_visibility is not None and oracle_visibility.dim() == 2:
            oracle_visibility = oracle_visibility.unsqueeze(0)

        device = oracle_state.device
        dtype = oracle_state.dtype
        batch, frames, num_objects, _ = oracle_state.shape

        u, v, w, h = self._normalize_geometry(
            oracle_state=oracle_state,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        d = self._normalize_depth(oracle_state[..., 2])
        du = self._normalize_signed(oracle_state[..., 5], self.velocity_clip)
        dv = self._normalize_signed(oracle_state[..., 6], self.velocity_clip)
        dd = self._normalize_signed(oracle_state[..., 7], self.depth_velocity_clip)
        vis = oracle_state[..., 8] if oracle_visibility is None else oracle_visibility
        vis = vis.clamp(0.0, 1.0)

        grid_x = torch.linspace(0.0, 1.0, latent_width, device=device, dtype=dtype).view(1, 1, 1, 1, latent_width)
        grid_y = torch.linspace(0.0, 1.0, latent_height, device=device, dtype=dtype).view(1, 1, 1, latent_height, 1)

        u = u.unsqueeze(-1).unsqueeze(-1)
        v = v.unsqueeze(-1).unsqueeze(-1)
        w = w.clamp(min=1e-4, max=1.0).unsqueeze(-1).unsqueeze(-1)
        h = h.clamp(min=1e-4, max=1.0).unsqueeze(-1).unsqueeze(-1)
        d = d.unsqueeze(-1).unsqueeze(-1)
        du = du.unsqueeze(-1).unsqueeze(-1)
        dv = dv.unsqueeze(-1).unsqueeze(-1)
        dd = dd.unsqueeze(-1).unsqueeze(-1)
        vis = vis.unsqueeze(-1).unsqueeze(-1)

        x1 = (u - 0.5 * w).clamp(0.0, 1.0)
        x2 = (u + 0.5 * w).clamp(0.0, 1.0)
        y1 = (v - 0.5 * h).clamp(0.0, 1.0)
        y2 = (v + 0.5 * h).clamp(0.0, 1.0)

        box = ((grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)).to(dtype=dtype) * vis
        sigma_x = torch.clamp(w * 0.25, min=1.0 / max(latent_width, 4))
        sigma_y = torch.clamp(h * 0.25, min=1.0 / max(latent_height, 4))
        gauss = torch.exp(
            -0.5
            * (
                ((grid_x - u) / sigma_x) ** 2
                + ((grid_y - v) / sigma_y) ** 2
            )
        ) * vis

        object_maps = torch.stack(
            [
                box,
                gauss,
                gauss * d,
                gauss * du,
                gauss * dv,
                gauss * dd,
                box * w,
                box * h,
            ],
            dim=3,
        )
        maps = object_maps.sum(dim=2)
        if maps.shape[2] != self.spatial_feature_dim:
            raise RuntimeError(
                f"Expected {self.spatial_feature_dim} spatial channels, got {maps.shape[3]}."
            )
        maps = maps.permute(0, 2, 1, 3, 4).contiguous()
        if phase_ids is not None:
            if phase_ids.dim() == 1:
                phase_ids = phase_ids.unsqueeze(0)
            phase_ids = phase_ids.to(device=device, dtype=torch.long)
            phase_bank = torch.stack(
                [self.context_phase_embedding, self.future_phase_embedding],
                dim=0,
            ).to(device=device, dtype=dtype)
            phase_feats = phase_bank[phase_ids].permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            maps = maps + phase_feats
        temporal = self._build_temporal_features(
            num_frames=maps.shape[2],
            device=device,
            dtype=dtype,
        )
        temporal = temporal.transpose(0, 1).view(1, self.spatial_feature_dim, maps.shape[2], 1, 1)
        return maps + temporal

    def _build_pose_features(
        self,
        oracle_pose: torch.Tensor | None,
        oracle_visibility: torch.Tensor | None,
        phase_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if oracle_pose is None:
            return None
        if oracle_pose.dim() == 3:
            oracle_pose = oracle_pose.unsqueeze(0)
        if oracle_visibility is not None and oracle_visibility.dim() == 2:
            oracle_visibility = oracle_visibility.unsqueeze(0)

        device = oracle_pose.device
        dtype = oracle_pose.dtype
        pose = oracle_pose
        if pose.shape[-1] != self.pose_dim:
            raise RuntimeError(f"Expected pose dim {self.pose_dim}, got {pose.shape[-1]}.")

        if oracle_visibility is None:
            weights = torch.ones(pose.shape[:-1], device=device, dtype=dtype)
        else:
            weights = oracle_visibility.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        weights_sum = weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        pooled_pose = (pose * weights.unsqueeze(-1)).sum(dim=2) / weights_sum
        pose_feats = self.pose_projection(pooled_pose)

        if phase_ids is not None:
            if phase_ids.dim() == 1:
                phase_ids = phase_ids.unsqueeze(0)
            phase_ids = phase_ids.to(device=device, dtype=torch.long)
            phase_bank = torch.stack(
                [self.context_phase_embedding, self.future_phase_embedding],
                dim=0,
            ).to(device=device, dtype=dtype)
            pose_feats = pose_feats + phase_bank[phase_ids]

        temporal = self._build_temporal_features(
            num_frames=pose_feats.shape[1],
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        pose_feats = pose_feats + temporal
        return pose_feats.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).contiguous()

    def _prepare_state_sequences(
        self,
        context_state: torch.Tensor | None,
        future_state: torch.Tensor | None,
        context_visibility: torch.Tensor | None,
        future_visibility: torch.Tensor | None,
        context_pose: torch.Tensor | None = None,
        future_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        if future_state is None:
            raise ValueError("future_state must not be None when preparing state sequences.")

        if future_state.dim() == 3:
            future_state = future_state.unsqueeze(0)
        if context_state is not None and context_state.dim() == 3:
            context_state = context_state.unsqueeze(0)
        if future_pose is not None and future_pose.dim() == 3:
            future_pose = future_pose.unsqueeze(0)
        if context_pose is not None and context_pose.dim() == 3:
            context_pose = context_pose.unsqueeze(0)
        if future_visibility is not None and future_visibility.dim() == 2:
            future_visibility = future_visibility.unsqueeze(0)
        if context_visibility is not None and context_visibility.dim() == 2:
            context_visibility = context_visibility.unsqueeze(0)

        state_chunks = []
        pose_chunks = []
        visibility_chunks = []
        phase_chunks = []

        if context_state is not None and context_state.shape[1] > 0:
            state_chunks.append(context_state)
            if context_pose is not None:
                pose_chunks.append(context_pose)
            phase_chunks.append(torch.zeros(context_state.shape[:2], device=context_state.device, dtype=torch.long))
            if context_visibility is not None:
                visibility_chunks.append(context_visibility)
            elif future_visibility is not None:
                visibility_chunks.append(context_state[..., 8].clamp(0.0, 1.0))

        if future_state.shape[1] > 0:
            state_chunks.append(future_state)
            if future_pose is not None:
                pose_chunks.append(future_pose)
            phase_chunks.append(torch.ones(future_state.shape[:2], device=future_state.device, dtype=torch.long))
            if future_visibility is not None:
                visibility_chunks.append(future_visibility)
            elif context_visibility is not None:
                visibility_chunks.append(future_state[..., 8].clamp(0.0, 1.0))

        full_state = torch.cat(state_chunks, dim=1)
        phase_ids = torch.cat(phase_chunks, dim=1)
        full_visibility = None
        full_pose = None
        if len(visibility_chunks) == len(state_chunks) and visibility_chunks:
            full_visibility = torch.cat(visibility_chunks, dim=1)
        if len(pose_chunks) == len(state_chunks) and pose_chunks:
            full_pose = torch.cat(pose_chunks, dim=1)
        return full_state, full_visibility, phase_ids, full_pose

    def build_vace_context(
        self,
        oracle_state: torch.Tensor,
        total_latent_frames: int,
        clean_prefix_len: int,
        latent_height: int,
        latent_width: int,
        frame_height: int,
        frame_width: int,
        oracle_visibility: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
        context_visibility: torch.Tensor | None = None,
        oracle_pose: torch.Tensor | None = None,
        context_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if oracle_state.dim() == 3:
            oracle_state = oracle_state.unsqueeze(0)
        if total_latent_frames <= 0:
            raise ValueError(f"total_latent_frames must be positive, got {total_latent_frames}.")

        if context_state is not None:
            full_state, full_visibility, phase_ids, full_pose = self._prepare_state_sequences(
                context_state=context_state,
                future_state=oracle_state,
                context_visibility=context_visibility,
                future_visibility=oracle_visibility,
                context_pose=context_pose,
                future_pose=oracle_pose,
            )
            maps = self._build_spatial_state_maps(
                oracle_state=full_state,
                latent_height=latent_height,
                latent_width=latent_width,
                frame_height=frame_height,
                frame_width=frame_width,
                oracle_visibility=full_visibility,
                phase_ids=phase_ids,
            )
            pose_feats = self._build_pose_features(
                oracle_pose=full_pose,
                oracle_visibility=full_visibility,
                phase_ids=phase_ids,
            )
            if pose_feats is not None:
                maps = maps + pose_feats
            if maps.shape[2] != int(total_latent_frames):
                maps = F.interpolate(
                    maps,
                    size=(int(total_latent_frames), latent_height, latent_width),
                    mode="trilinear",
                    align_corners=False,
                )
            maps = self._maybe_dropout(maps)
            return self.spatial_encoder(maps)

        future_frames = max(int(total_latent_frames) - int(clean_prefix_len), 0)
        if future_frames <= 0:
            batch = oracle_state.shape[0]
            return oracle_state.new_zeros(
                (batch, self.vace_in_dim, total_latent_frames, latent_height, latent_width)
            )

        maps = self._build_spatial_state_maps(
            oracle_state=oracle_state,
            latent_height=latent_height,
            latent_width=latent_width,
            frame_height=frame_height,
            frame_width=frame_width,
            oracle_visibility=oracle_visibility,
        )
        pose_feats = self._build_pose_features(
            oracle_pose=oracle_pose,
            oracle_visibility=oracle_visibility,
            phase_ids=None,
        )
        if pose_feats is not None:
            maps = maps + pose_feats
        if maps.shape[2] != future_frames:
            maps = F.interpolate(
                maps,
                size=(future_frames, latent_height, latent_width),
                mode="trilinear",
                align_corners=False,
            )
        maps = self._maybe_dropout(maps)
        future_context = self.spatial_encoder(maps)

        if clean_prefix_len > 0:
            prefix = future_context.new_zeros(
                (future_context.shape[0], future_context.shape[1], clean_prefix_len, latent_height, latent_width)
            )
            return torch.cat([prefix, future_context], dim=2)
        return future_context
