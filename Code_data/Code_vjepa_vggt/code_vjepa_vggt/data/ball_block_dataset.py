from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_uniform


class BallBlockVideoDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        num_frames: int,
        num_context_frames: int,
        resolution: tuple[int, int],
    ) -> None:
        self.root = Path(root)
        self.num_frames = num_frames
        self.num_context_frames = num_context_frames
        self.resolution = resolution
        self.samples = sorted(self.root.glob("*.json"))
        if not self.samples:
            raise RuntimeError(f"no json metadata found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta_path = self.samples[idx]
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        video_path = Path(meta["video"])
        frames, frame_indices = read_video_uniform(video_path, self.num_frames)
        video = preprocess_video_rgb_uint8(frames, self.resolution)  # [C,T,H,W]
        context_video = video[:, : self.num_context_frames].contiguous()

        return {
            "video": video,
            "context_video": context_video,
            "caption": meta["caption"],
            "video_path": str(video_path),
            "frame_indices": torch.from_numpy(frame_indices),
            "num_context_frames": self.num_context_frames,
            "metadata": meta,
        }

