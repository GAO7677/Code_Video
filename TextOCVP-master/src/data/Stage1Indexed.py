"""Fixed-index multi-source dataset for comparable SAVi Stage 1 experiments."""

from __future__ import annotations

import json
from pathlib import Path

import decord
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Stage1Indexed(Dataset):
    SPLIT_FILES = {
        "train": "train.jsonl",
        "valid": "handoff_monitor.jsonl",
        "val": "handoff_monitor.jsonl",
    }

    def __init__(
        self,
        index_root,
        dataset_mode,
        split,
        num_frames=10,
        img_size=(216, 384),
        frame_stride=1,
        random_start=True,
        max_samples=None,
        **kwargs,
    ):
        del kwargs
        if dataset_mode not in {"pybullet", "kubric", "mixed"}:
            raise ValueError(f"Unsupported dataset_mode={dataset_mode!r}")
        if split not in self.SPLIT_FILES:
            raise ValueError(f"Unsupported split={split!r}")
        self.dataset_mode = dataset_mode
        self.split = split
        self.num_frames = int(num_frames)
        self.img_size = tuple(int(value) for value in img_size)
        self.frame_stride = int(frame_stride)
        self.random_start = bool(random_start) and split == "train"
        index_path = Path(index_root).resolve() / dataset_mode / self.SPLIT_FILES[split]
        if not index_path.is_file():
            raise FileNotFoundError(f"Stage 1 index does not exist: {index_path}")
        self.records = [
            json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line
        ]
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if not self.records:
            raise RuntimeError(f"No records loaded from {index_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        video_path = Path(record["video_path"])
        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
        range_start, configured_range_end = record.get("sampling_frame_range", [0, 49])
        range_end = min(int(configured_range_end), len(reader) - 1)
        required_span = 1 + (self.num_frames - 1) * self.frame_stride
        max_start = range_end - required_span + 1
        if max_start < range_start:
            raise ValueError(
                f"Cannot sample {self.num_frames} frames from {video_path} in "
                f"[{range_start}, {range_end}]"
            )
        if self.random_start:
            start = int(torch.randint(int(range_start), max_start + 1, size=(1,)).item())
        else:
            start = (int(range_start) + max_start) // 2
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
            **record,
            "start_frame": start,
            "frame_ids": frame_ids.tolist(),
            "training_shape": [self.num_frames, 3, *self.img_size],
        }
        return frames, metadata

    @staticmethod
    def collate_fn(data):
        videos = torch.stack([sample[0] for sample in data], dim=0)
        metadata = [sample[1] for sample in data]
        return videos, {
            "metadata": metadata,
            "sources": [item["source"] for item in metadata],
        }
