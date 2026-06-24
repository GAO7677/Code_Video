from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


COTRACKER_REPO_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
if str(COTRACKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_REPO_ROOT))

from cotracker.predictor import CoTrackerPredictor  # type: ignore


@dataclass
class CoTrackerOutput:
    query_points: torch.Tensor
    tracks: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    image_hw: tuple[int, int]
    input_hw: tuple[int, int]
    used_model: bool


class CoTrackerAdapter(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None,
        *,
        num_queries: int = 8,
        device: str = "cuda",
        input_hw: tuple[int, int] = (384, 512),
        window_len: int = 60,
    ) -> None:
        super().__init__()
        self.device_obj = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.num_queries = int(num_queries)
        self.input_hw = tuple(int(v) for v in input_hw)
        self.window_len = int(window_len)
        self.model = None
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model = CoTrackerPredictor(
                checkpoint=str(checkpoint_path),
                offline=True,
                v2=False,
                window_len=self.window_len,
            ).to(self.device_obj)
            self.model.eval().requires_grad_(False)

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

    @staticmethod
    def _resize_video_bthwc(frames_bthwc_01: torch.Tensor, dst_hw: tuple[int, int]) -> torch.Tensor:
        b, t, h, w, c = frames_bthwc_01.shape
        frames_bchw = frames_bthwc_01.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        resized = F.interpolate(frames_bchw, size=dst_hw, mode="bilinear", align_corners=True)
        return resized.reshape(b, t, c, dst_hw[0], dst_hw[1]).permute(0, 1, 3, 4, 2).contiguous()

    @staticmethod
    def _scale_tracks(
        tracks: torch.Tensor,
        *,
        src_hw: tuple[int, int],
        dst_hw: tuple[int, int],
    ) -> torch.Tensor:
        out = tracks.clone()
        # CoTracker uses pixel-center style coordinates on the resized input.
        # Match the same corner-aligned resize convention used by the analysis
        # utilities so the roundtrip back to native pixels is consistent.
        scale_x = float(max(dst_hw[1] - 1, 1)) / max(float(src_hw[1] - 1), 1.0)
        scale_y = float(max(dst_hw[0] - 1, 1)) / max(float(src_hw[0] - 1), 1.0)
        out[..., 0] *= scale_x
        out[..., 1] *= scale_y
        return out

    def forward(
        self,
        frames_bthwc_01: torch.Tensor,
        *,
        query_points_prior: torch.Tensor | None = None,
        query_frame_ids: torch.Tensor | int | None = None,
        query_image_hw: tuple[int, int] | None = None,
    ) -> CoTrackerOutput:
        batch_size, frames, height, width, _ = frames_bthwc_01.shape
        native_hw = (height, width)
        if query_points_prior is not None:
            query_points_native = query_points_prior.to(device=frames_bthwc_01.device, dtype=frames_bthwc_01.dtype)
        else:
            raise RuntimeError("query_points_prior is required; uniform query fallback is disabled")

        if self.model is None:
            raise RuntimeError("CoTracker model is required for inference; fallback tracks are disabled")

        src_hw = query_image_hw if query_image_hw is not None else native_hw
        query_points_cot = self._resize_query_points(query_points_native, src_hw=src_hw, dst_hw=self.input_hw)
        cotracker_video = self._resize_video_bthwc(frames_bthwc_01, self.input_hw).permute(0, 1, 4, 2, 3)
        if query_frame_ids is None:
            query_frame_ids = torch.zeros(
                batch_size,
                query_points_cot.shape[1],
                1,
                device=query_points_cot.device,
                dtype=query_points_cot.dtype,
            )
        else:
            query_frame_ids = torch.as_tensor(query_frame_ids, device=query_points_cot.device)
            if query_frame_ids.ndim == 0:
                query_frame_ids = query_frame_ids.view(1, 1, 1).expand(batch_size, query_points_cot.shape[1], 1)
            elif query_frame_ids.ndim == 1:
                if int(query_frame_ids.shape[0]) != int(query_points_cot.shape[1]):
                    raise ValueError(
                        f"query_frame_ids length {int(query_frame_ids.shape[0])} does not match num_queries {int(query_points_cot.shape[1])}"
                    )
                query_frame_ids = query_frame_ids.view(1, -1, 1).expand(batch_size, -1, -1)
            elif query_frame_ids.ndim == 2:
                if tuple(int(v) for v in query_frame_ids.shape) != (batch_size, query_points_cot.shape[1]):
                    raise ValueError(
                        f"query_frame_ids shape {list(query_frame_ids.shape)} does not match batch/num_queries "
                        f"({batch_size}, {int(query_points_cot.shape[1])})"
                    )
                query_frame_ids = query_frame_ids.unsqueeze(-1)
            elif query_frame_ids.ndim == 3:
                if tuple(int(v) for v in query_frame_ids.shape) != (batch_size, query_points_cot.shape[1], 1):
                    raise ValueError(
                        f"query_frame_ids shape {list(query_frame_ids.shape)} does not match "
                        f"({batch_size}, {int(query_points_cot.shape[1])}, 1)"
                    )
            else:
                raise ValueError(f"unsupported query_frame_ids shape: {list(query_frame_ids.shape)}")
            query_frame_ids = query_frame_ids.to(dtype=query_points_cot.dtype)
        queries = torch.cat([query_frame_ids, query_points_cot], dim=-1)

        with torch.no_grad():
            tracks, visibility = self.model(
                cotracker_video.to(self.device_obj),
                queries=queries.to(self.device_obj),
                backward_tracking=False,
            )
        tracks = self._scale_tracks(tracks, src_hw=self.input_hw, dst_hw=native_hw)
        visibility = visibility.to(dtype=tracks.dtype)
        confidence = visibility.clone()
        return CoTrackerOutput(
            query_points=query_points_native,
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            image_hw=native_hw,
            input_hw=self.input_hw,
            used_model=True,
        )
