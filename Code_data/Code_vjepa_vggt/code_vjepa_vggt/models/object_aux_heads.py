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
    def __init__(
        self,
        dim: int = 4096,
        track_delta_scale: float = 0.06,
        box_delta_scale: float = 0.06,
        box_wh_log_scale: float = 1.25,
        box_wh_max_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.track_delta_scale = float(track_delta_scale)
        self.box_delta_scale = float(box_delta_scale)
        self.box_wh_log_scale = float(box_wh_log_scale)
        self.box_wh_max_scale = float(box_wh_max_scale)
        self.track_head = _ResidualMLP(dim, 4)
        self.box_head = _ResidualMLP(dim, 4)
        self.depth_head = _ResidualMLP(dim, 1)

    def set_scales(
        self,
        *,
        track_delta_scale: float | None = None,
        box_delta_scale: float | None = None,
        box_wh_log_scale: float | None = None,
        box_wh_max_scale: float | None = None,
    ) -> None:
        if track_delta_scale is not None:
            self.track_delta_scale = float(track_delta_scale)
        if box_delta_scale is not None:
            self.box_delta_scale = float(box_delta_scale)
        if box_wh_log_scale is not None:
            self.box_wh_log_scale = float(box_wh_log_scale)
        if box_wh_max_scale is not None:
            self.box_wh_max_scale = float(box_wh_max_scale)

    def forward(
        self,
        object_latent_tokens: torch.Tensor,
        active_track_summary: torch.Tensor,
        active_box_xyxy: torch.Tensor | None = None,
    ) -> ObjectAuxHeadOutput:
        if int(active_track_summary.shape[-1]) < 4:
            raise ValueError(
                f"active_track_summary must have at least 4 channels, got {list(active_track_summary.shape)}"
            )
        track_base = active_track_summary[..., :4]
        track_delta = self.track_delta_scale * torch.tanh(self.track_head(object_latent_tokens))
        pred_track_summary = track_base + track_delta
        # Anchor the box center to the current track summary so the box can
        # follow motion, while still letting the prior box control the scale.
        base_center_xy = pred_track_summary[..., :2]
        if active_box_xyxy is None:
            base_half_wh = base_center_xy.new_tensor([0.03, 0.03]).view(1, 1, 1, 2)
            base_box_xyxy = torch.cat(
                [
                    (base_center_xy - base_half_wh).clamp(0.0, 1.0),
                    (base_center_xy + base_half_wh).clamp(0.0, 1.0),
                ],
                dim=-1,
            )
        else:
            base_wh = (active_box_xyxy[..., 2:] - active_box_xyxy[..., :2]).clamp_min(1.0e-4)
            base_box_xyxy = torch.cat(
                [
                    (base_center_xy - 0.5 * base_wh).clamp(0.0, 1.0),
                    (base_center_xy + 0.5 * base_wh).clamp(0.0, 1.0),
                ],
                dim=-1,
            )
        box_delta = self.box_head(object_latent_tokens)
        base_center_xy = 0.5 * (base_box_xyxy[..., :2] + base_box_xyxy[..., 2:])
        base_wh = (base_box_xyxy[..., 2:] - base_box_xyxy[..., :2]).clamp_min(1.0e-4)
        center_delta = self.box_delta_scale * torch.tanh(box_delta[..., :2]) * base_wh
        wh_log_scale = self.box_wh_log_scale * torch.tanh(box_delta[..., 2:])
        wh_scale = torch.exp(wh_log_scale).clamp(0.75, self.box_wh_max_scale)

        pred_center_xy = (base_center_xy + center_delta).clamp(0.0, 1.0)
        pred_box_wh = (base_wh * wh_scale).clamp(1.0e-4, 1.0)
        half_wh = 0.5 * pred_box_wh
        pred_box_xyxy = torch.cat(
            [
                (pred_center_xy - half_wh).clamp(0.0, 1.0),
                (pred_center_xy + half_wh).clamp(0.0, 1.0),
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
