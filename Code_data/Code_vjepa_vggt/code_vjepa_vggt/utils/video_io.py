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


def _resize_cover_center_crop_video(
    video_tchw: torch.Tensor,
    crop_hw: tuple[int, int],
) -> torch.Tensor:
    target_h, target_w = int(crop_hw[0]), int(crop_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"crop_hw must be positive, got {crop_hw}")
    _, _, src_h, src_w = video_tchw.shape
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"invalid source resolution {(src_h, src_w)}")

    scale = max(target_h / float(src_h), target_w / float(src_w))
    resized_h = max(target_h, int(round(src_h * scale)))
    resized_w = max(target_w, int(round(src_w * scale)))
    resized = F.interpolate(
        video_tchw,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    top = max(0, (resized_h - target_h) // 2)
    left = max(0, (resized_w - target_w) // 2)
    return resized[:, :, top : top + target_h, left : left + target_w]


def preprocess_video_rgb_uint8(
    video_thwc: np.ndarray,
    out_hw: tuple[int, int],
    value_range: str = "minus_one_to_one",
    resize_mode: str = "stretch",
    cover_crop_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    x = torch.from_numpy(video_thwc).permute(0, 3, 1, 2).float()  # [T,C,H,W]
    if resize_mode == "stretch":
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    elif resize_mode == "cover_crop":
        crop_hw = out_hw if cover_crop_hw is None else cover_crop_hw
        x = _resize_cover_center_crop_video(x, crop_hw)
        if tuple(int(v) for v in crop_hw) != tuple(int(v) for v in out_hw):
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"unsupported resize_mode: {resize_mode}")
    x = x / 255.0
    if value_range == "minus_one_to_one":
        x = x * 2.0 - 1.0
    return x.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
