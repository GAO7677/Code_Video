from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from code_vjepa_vggt.utils.paths import ensure_upstream_paths

ensure_upstream_paths()

from vggt.models.vggt import VGGT  # type: ignore


@dataclass
class VGGTTrackOutput:
    query_points: torch.Tensor
    tracks: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    image_hw: tuple[int, int]
    used_model: bool


class VGGTTrackAdapter(nn.Module):
    def __init__(
        self,
        model_path: str | None,
        num_queries: int = 8,
        device: str = "cuda",
        input_hw: tuple[int, int] = (420, 728),
    ) -> None:
        super().__init__()
        self.device_obj = torch.device(device)
        self.model_path = model_path
        self.num_queries = num_queries
        self.input_hw = input_hw
        self.model = None
        if model_path and Path(model_path).exists():
            self.model = VGGT.from_pretrained(model_path).eval().requires_grad_(False).to(self.device_obj)

    def _make_uniform_queries(self, batch_size: int, image_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
        height, width = image_hw
        cols = max(1, int(torch.ceil(torch.sqrt(torch.tensor(float(self.num_queries)))).item()))
        rows = max(1, (self.num_queries + cols - 1) // cols)
        xs = torch.linspace(0.2 * width, 0.8 * width, cols, device=device)
        ys = torch.linspace(0.2 * height, 0.8 * height, rows, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)[: self.num_queries]
        return points.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    @staticmethod
    def _resize_query_points(
        query_points: torch.Tensor,
        *,
        src_hw: tuple[int, int],
        dst_hw: tuple[int, int],
    ) -> torch.Tensor:
        scale_x = float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
        scale_y = float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
        out = query_points.clone()
        out[..., 0] *= scale_x
        out[..., 1] *= scale_y
        return out

    def forward(
        self,
        frames_bthwc_01: torch.Tensor,
        *,
        query_points_prior: torch.Tensor | None = None,
        query_image_hw: tuple[int, int] | None = None,
    ) -> VGGTTrackOutput:
        batch_size, frames, height, width, _ = frames_bthwc_01.shape
        if self.model is None:
            if query_points_prior is not None:
                query_points = query_points_prior.to(device=frames_bthwc_01.device, dtype=frames_bthwc_01.dtype)
            else:
                query_points = self._make_uniform_queries(batch_size, (height, width), frames_bthwc_01.device)
            tracks = query_points.unsqueeze(1).expand(-1, frames, -1, -1).clone()
            vis = torch.ones(batch_size, frames, self.num_queries, device=frames_bthwc_01.device)
            conf = torch.ones(batch_size, frames, self.num_queries, device=frames_bthwc_01.device)
            return VGGTTrackOutput(
                query_points=query_points,
                tracks=tracks,
                visibility=vis,
                confidence=conf,
                image_hw=(height, width),
                used_model=False,
            )

        frames_bchw = frames_bthwc_01.permute(0, 1, 4, 2, 3)
        resized = F.interpolate(
            frames_bchw.reshape(-1, 3, height, width),
            size=self.input_hw,
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, frames, 3, self.input_hw[0], self.input_hw[1])
        if query_points_prior is not None:
            query_points = query_points_prior.to(device=resized.device, dtype=resized.dtype)
            src_hw = query_image_hw if query_image_hw is not None else (height, width)
            query_points = self._resize_query_points(
                query_points,
                src_hw=src_hw,
                dst_hw=self.input_hw,
            )
        else:
            query_points = self._make_uniform_queries(batch_size, self.input_hw, resized.device)
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.model.shortcut_forward(resized)
            track_list, vis, conf = self.model.track_head(
                aggregated_tokens_list,
                images=resized,
                patch_start_idx=patch_start_idx,
                query_points=query_points,
            )
            tracks = track_list[-1] if isinstance(track_list, (list, tuple)) else track_list

        return VGGTTrackOutput(
            query_points=query_points,
            tracks=tracks,
            visibility=vis,
            confidence=conf,
            image_hw=self.input_hw,
            used_model=True,
        )
