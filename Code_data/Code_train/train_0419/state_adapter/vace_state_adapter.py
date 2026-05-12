"""State-conditioned adapter that turns future 9D object states into VACE context volumes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class VaceStateAdapter(torch.nn.Module):
    def __init__(
        self,
        state_dim: int = 9,
        spatial_feature_dim: int = 8,
        hidden_dim: int = 128,
        vace_in_dim: int = 96,
        condition_dropout: float = 0.1,
        state_is_normalized: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.spatial_feature_dim = int(spatial_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.vace_in_dim = int(vace_in_dim)
        self.condition_dropout = float(condition_dropout)
        self.state_is_normalized = bool(state_is_normalized)

        self.spatial_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(self.spatial_feature_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv3d(self.hidden_dim, self.vace_in_dim, kernel_size=3, padding=1),
        )
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

    def _build_spatial_state_maps(
        self,
        oracle_state: torch.Tensor,
        latent_height: int,
        latent_width: int,
        frame_height: int,
        frame_width: int,
        oracle_visibility: torch.Tensor | None = None,
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
        d = oracle_state[..., 2]
        du = oracle_state[..., 5]
        dv = oracle_state[..., 6]
        dd = oracle_state[..., 7]
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
        return maps.permute(0, 2, 1, 3, 4).contiguous()

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
    ) -> torch.Tensor:
        if oracle_state.dim() == 3:
            oracle_state = oracle_state.unsqueeze(0)
        if total_latent_frames <= 0:
            raise ValueError(f"total_latent_frames must be positive, got {total_latent_frames}.")

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
