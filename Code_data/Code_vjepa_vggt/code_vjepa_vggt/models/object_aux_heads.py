from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ObjectAuxHeadOutput:
    pred_track_summary: torch.Tensor
    pred_box_wh: torch.Tensor
    pred_box_xyxy: torch.Tensor
    pred_depth: torch.Tensor


class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObjectAuxHeads(nn.Module):
    def __init__(self, dim: int = 4096, track_delta_scale: float = 0.25) -> None:
        super().__init__()
        self.track_delta_scale = float(track_delta_scale)
        self.track_head = _ResidualMLP(dim, 4)
        self.box_head = _ResidualMLP(dim, 2)
        self.depth_head = _ResidualMLP(dim, 1)

    def forward(
        self,
        object_latent_tokens: torch.Tensor,
        active_track_summary: torch.Tensor,
    ) -> ObjectAuxHeadOutput:
        track_delta = self.track_delta_scale * torch.tanh(self.track_head(object_latent_tokens))
        pred_track_summary = active_track_summary + track_delta
        pred_box_wh = torch.sigmoid(self.box_head(object_latent_tokens))
        center_xy = pred_track_summary[..., :2]
        half_wh = 0.5 * pred_box_wh
        pred_box_xyxy = torch.cat(
            [
                (center_xy - half_wh).clamp(0.0, 1.0),
                (center_xy + half_wh).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        pred_depth = self.depth_head(object_latent_tokens)
        return ObjectAuxHeadOutput(
            pred_track_summary=pred_track_summary,
            pred_box_wh=pred_box_wh,
            pred_box_xyxy=pred_box_xyxy,
            pred_depth=pred_depth,
        )
