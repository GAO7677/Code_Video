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
        init_scan_limit: int | None = None,
    ) -> None:
        self.root = Path(root) / split
        self.split = split
        self.resolution = resolution
        self.num_context_frames = num_context_frames
        self.context_fraction = context_fraction
        self.random_context_frames = random_context_frames
        self.seed = seed
        self.init_scan_limit = None if init_scan_limit is None else max(int(init_scan_limit), 1)
        self.samples = sorted(self.root.glob("*.json"))
        if not self.samples:
            raise RuntimeError(f"no json metadata found under {self.root}")
        if self.init_scan_limit is not None:
            self.samples = self.samples[: self.init_scan_limit]
        if not self._can_skip_context_filter(self.samples):
            self.samples = self._filter_samples_with_enough_context(self.samples)
        if not self.samples:
            raise RuntimeError(
                f"no samples under {self.root} can provide fixed num_context_frames={self.num_context_frames} "
                f"within context_fraction={self.context_fraction}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_video(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        return F.interpolate(frames_tchw, size=self.resolution, mode="bilinear", align_corners=False)

    def _max_context_len(self, total_frames: int) -> int:
        return max(1, min(total_frames, int(total_frames * self.context_fraction)))

    def _filter_samples_with_enough_context(self, samples: list[Path]) -> list[Path]:
        filtered: list[Path] = []
        for meta_path in samples:
            npz_path = meta_path.with_suffix(".npz")
            tensors = load_npz_tensor_dict(npz_path)
            total_frames = int(tensors["context_frames"].shape[0] + tensors["future_frames"].shape[0])
            if self._max_context_len(total_frames) >= self.num_context_frames:
                filtered.append(meta_path)
        return filtered

    def _can_skip_context_filter(self, samples: list[Path]) -> bool:
        if not samples:
            return False
        probe_count = min(len(samples), 8)
        total_frames_ref: int | None = None
        for meta_path in samples[:probe_count]:
            npz_path = meta_path.with_suffix(".npz")
            tensors = load_npz_tensor_dict(npz_path)
            total_frames = int(tensors["context_frames"].shape[0] + tensors["future_frames"].shape[0])
            if self._max_context_len(total_frames) < self.num_context_frames:
                return False
            if total_frames_ref is None:
                total_frames_ref = total_frames
                continue
            if total_frames != total_frames_ref:
                return False
        return True

    def _select_context_indices(self, total_frames: int, idx: int) -> torch.Tensor:
        max_context_len = self._max_context_len(total_frames)
        if max_context_len < self.num_context_frames:
            raise RuntimeError(
                f"sample idx={idx} only has max_context_len={max_context_len}, "
                f"smaller than required num_context_frames={self.num_context_frames}"
            )

        if not self.random_context_frames:
            return torch.arange(self.num_context_frames, dtype=torch.long)

        max_start = max_context_len - self.num_context_frames
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item()) if max_start > 0 else 0
        return torch.arange(start, start + self.num_context_frames, dtype=torch.long)

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
