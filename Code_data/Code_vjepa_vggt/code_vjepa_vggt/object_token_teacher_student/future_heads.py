from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from code_vjepa_vggt.models.object_aux_heads import ObjectAuxHeads


@dataclass
class FutureHeadOutput:
    pred_track_summary: torch.Tensor
    pred_box_xyxy: torch.Tensor
    pred_box_wh: torch.Tensor
    pred_depth: torch.Tensor


class FutureObjectHeads(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        track_delta_scale: float = 1.0,
        box_delta_scale: float = 1.0,
        box_wh_log_scale: float = 1.25,
        box_wh_max_scale: float = 2.0,
        track_gate_init: float = 0.5,
        box_center_gate_init: float = 0.5,
        box_size_gate_init: float = 0.5,
    ) -> None:
        super().__init__()
        # Unlike the Stage1 aux heads (which refine an already-good per-frame
        # base), the Stage2 future heads predict displacement from the *last
        # context* anchor, so the residual must be able to span a much larger
        # range. We therefore default to wide delta scales and open gates.
        self.heads = ObjectAuxHeads(
            dim=dim,
            track_delta_scale=track_delta_scale,
            box_delta_scale=box_delta_scale,
            box_wh_log_scale=box_wh_log_scale,
            box_wh_max_scale=box_wh_max_scale,
            track_gate_init=track_gate_init,
            box_center_gate_init=box_center_gate_init,
            box_size_gate_init=box_size_gate_init,
        )

    def forward(
        self,
        future_tokens: torch.Tensor,
        base_track_summary: torch.Tensor,
        base_box_xyxy: torch.Tensor,
    ) -> FutureHeadOutput:
        out = self.heads(
            future_tokens,
            active_track_summary=base_track_summary,
            active_box_xyxy=base_box_xyxy,
        )
        return FutureHeadOutput(
            pred_track_summary=out.pred_track_summary,
            pred_box_xyxy=out.pred_box_xyxy,
            pred_box_wh=out.pred_box_wh,
            pred_depth=out.pred_depth,
        )
