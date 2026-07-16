"""Raw-video adapter for PyBullet SAVi decomposition training."""

from __future__ import annotations

from pathlib import Path

import decord
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PyBullet(Dataset):
    """Load contiguous clips directly from the raw PyBullet H.264 videos."""

    SPLIT_ALIASES = {"valid": "val"}

    def __init__(
        self,
        root,
        split,
        num_frames=10,
        img_size=(64, 112),
        random_start=True,
        frame_stride=1,
        sampling_frame_range=(0, 49),
        max_samples=None,
        **kwargs,
    ):
        del kwargs
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.num_frames = int(num_frames)
        self.img_size = tuple(int(value) for value in img_size)
        self.random_start = bool(random_start) and split == "train"
        self.frame_stride = int(frame_stride)
        self.sampling_frame_range = tuple(int(value) for value in sampling_frame_range)
        if self.num_frames < 1:
            raise ValueError("num_frames must be positive")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if (
            len(self.sampling_frame_range) != 2
            or self.sampling_frame_range[0] < 0
            or self.sampling_frame_range[1] < self.sampling_frame_range[0]
        ):
            raise ValueError(
                "sampling_frame_range must be [first_frame, last_frame] with 0 <= first <= last"
            )

        split_dir = self.SPLIT_ALIASES.get(split, split)
        split_root = self.root / split_dir
        if not split_root.is_dir():
            raise FileNotFoundError(f"PyBullet split directory does not exist: {split_root}")

        family_videos = []
        for family_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            videos = sorted(family_dir.glob("sample_*/video.mp4"))
            if videos:
                family_videos.append(videos)
        self.video_paths = self._round_robin(family_videos, max_samples)
        if not self.video_paths:
            raise FileNotFoundError(f"No sample_*/video.mp4 files found under {split_root}")

    @staticmethod
    def _round_robin(family_videos, max_samples):
        """Keep smoke subsets family-balanced while retaining every formal sample."""
        if max_samples is None:
            return [path for videos in family_videos for path in videos]

        limit = int(max_samples)
        if limit < 1:
            raise ValueError("max_samples must be positive when provided")
        selected = []
        offset = 0
        while len(selected) < limit:
            added = False
            for videos in family_videos:
                if offset < len(videos):
                    selected.append(videos[offset])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            offset += 1
        return selected

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
        required_span = 1 + (self.num_frames - 1) * self.frame_stride
        range_start, configured_range_end = self.sampling_frame_range
        range_end = min(configured_range_end, len(reader) - 1)
        max_start = range_end - required_span + 1
        if max_start < range_start:
            raise ValueError(
                f"Video {video_path} cannot provide {self.num_frames} frames with stride "
                f"{self.frame_stride} inside [{range_start}, {range_end}]"
            )

        if self.random_start:
            start = int(torch.randint(range_start, max_start + 1, size=(1,)).item())
        else:
            start = (range_start + max_start) // 2
        frame_ids = start + np.arange(self.num_frames) * self.frame_stride
        frames = torch.from_numpy(reader.get_batch(frame_ids).asnumpy()).float()
        frames = frames.permute(0, 3, 1, 2).div_(255.0)
        frames = F.interpolate(
            frames,
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        metadata = {
            "video_path": str(video_path),
            "start_frame": start,
            "frame_ids": frame_ids.tolist(),
            "sampling_frame_range": list(self.sampling_frame_range),
        }
        return frames, metadata

    @staticmethod
    def collate_fn(data):
        videos = torch.stack([sample[0] for sample in data], dim=0)
        return videos, {}
