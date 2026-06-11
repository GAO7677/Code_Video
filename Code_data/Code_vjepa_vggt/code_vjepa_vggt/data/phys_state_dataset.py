from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from code_vjepa_vggt.utils.npz_io import load_npz_tensor_dict


class PhysStateEpisodeDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: tuple[int, int],
        num_context_frames: int = 8,
        context_fraction: float = 0.5,
        random_context_frames: bool = True,
        seed: int = 42,
    ) -> None:
        self.root = Path(root) / split
        self.split = split
        self.resolution = resolution
        self.num_context_frames = num_context_frames
        self.context_fraction = context_fraction
        self.random_context_frames = random_context_frames
        self.seed = seed
        self.samples = sorted(self.root.glob("*.json"))
        if not self.samples:
            raise RuntimeError(f"no json metadata found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_video(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        return F.interpolate(frames_tchw, size=self.resolution, mode="bilinear", align_corners=False)

    def _select_context_indices(self, total_frames: int, idx: int) -> torch.Tensor:
        max_context_end = max(self.num_context_frames, int(total_frames * self.context_fraction))
        candidate_count = min(total_frames, max_context_end)
        candidate = torch.arange(candidate_count, dtype=torch.long)
        if candidate_count <= self.num_context_frames:
            return candidate[: self.num_context_frames]
        if self.random_context_frames:
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx)
            perm = torch.randperm(candidate_count, generator=generator)
            chosen = candidate[perm[: self.num_context_frames]]
            return chosen.sort().values
        lin = torch.linspace(0, candidate_count - 1, self.num_context_frames)
        return lin.round().long()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta_path = self.samples[idx]
        npz_path = meta_path.with_suffix(".npz")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        tensors = load_npz_tensor_dict(npz_path)
        context_frames = tensors["context_frames"].float()
        future_frames = tensors["future_frames"].float()
        all_frames = torch.cat([context_frames, future_frames], dim=0)
        total_frames = all_frames.shape[0]
        context_indices = self._select_context_indices(total_frames, idx)

        video_tchw = self._resize_video(all_frames)
        video = video_tchw.permute(1, 0, 2, 3).contiguous()
        video = video * 2.0 - 1.0
        context_video = video[:, context_indices].contiguous()

        all_boxes = torch.cat([tensors["context_boxes"].float(), tensors["future_boxes"].float()], dim=0)
        all_states = torch.cat([tensors["context_states"].float(), tensors["future_states"].float()], dim=0)
        context_boxes = all_boxes[context_indices].contiguous()
        context_states = all_states[context_indices].contiguous()

        return {
            "video": video,
            "context_video": context_video,
            "caption": meta["prompt"],
            "video_path": str(npz_path),
            "frame_indices": torch.arange(total_frames, dtype=torch.long),
            "context_frame_indices": context_indices,
            "num_context_frames": int(context_indices.numel()),
            "metadata": meta,
            "context_boxes": context_boxes,
            "future_boxes": tensors["future_boxes"].float(),
            "context_states": context_states,
            "future_states": tensors["future_states"].float(),
            "appearance": tensors["appearance"].float(),
            "camera": tensors["camera"].float(),
        }
