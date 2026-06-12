from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu


def sample_frame_indices(frame_count: int, num_frames: int) -> np.ndarray:
    idx = np.linspace(0, frame_count - 1, num_frames)
    return np.round(idx).astype(np.int64)


def read_video_uniform(video_path: str | Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    frame_idx = sample_frame_indices(len(vr), num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_idx


def read_video_prefix(video_path: str | Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    frame_idx = np.arange(min(len(vr), num_frames), dtype=np.int64)
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_idx


def preprocess_video_rgb_uint8(
    video_thwc: np.ndarray,
    out_hw: tuple[int, int],
    value_range: str = "minus_one_to_one",
) -> torch.Tensor:
    x = torch.from_numpy(video_thwc).permute(0, 3, 1, 2).float()  # [T,C,H,W]
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    x = x / 255.0
    if value_range == "minus_one_to_one":
        x = x * 2.0 - 1.0
    return x.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
