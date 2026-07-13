from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from code_vjepa_vggt.data.kubric_no_gt_box_dataset import KubricNoGTBoxDataset
from code_vjepa_vggt.train0419_reference.dataset import OpenVidParquetDataset
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


class KubricReplayNoGTBoxDataset(KubricNoGTBoxDataset):
    """Reuse the stability-v3 index while decoding a shorter replay clip."""

    def __init__(
        self,
        *args,
        num_frames: int,
        num_context_frames: int,
        index_num_frames: int = 69,
        index_num_context_frames: int = 20,
        **kwargs,
    ) -> None:
        output_num_frames = int(num_frames)
        output_num_context_frames = int(num_context_frames)
        if output_num_frames > int(index_num_frames):
            raise ValueError(
                f"output num_frames={output_num_frames} exceeds index minimum={index_num_frames}"
            )
        super().__init__(
            *args,
            num_frames=int(index_num_frames),
            num_context_frames=int(index_num_context_frames),
            **kwargs,
        )
        self.num_frames = output_num_frames
        self.num_context_frames = output_num_context_frames


class OpenVidNoGTBoxDataset(Dataset):
    """Adapt embedded OpenVid parquet clips to the Stage1B no-GT-box contract."""

    dataset_kind = "openvid_no_gt_box"

    def __init__(
        self,
        root: str | Path,
        resolution: tuple[int, int],
        *,
        num_frames: int,
        num_context_frames: int,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.resolution = tuple(int(value) for value in resolution)
        self.num_frames = int(num_frames)
        self.num_context_frames = int(num_context_frames)
        self.dataset = OpenVidParquetDataset(
            str(self.root),
            height=self.resolution[0],
            width=self.resolution[1],
            num_frames=self.num_frames,
        )
        available = len(self.dataset)
        self.num_samples = available if max_samples is None else min(available, int(max_samples))
        if self.num_samples <= 0:
            raise RuntimeError(f"OpenVid dataset is empty under {self.root}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = int(index) % self.num_samples
        raw = self.dataset[row_index]
        frames = np.stack(
            [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in raw["video"]],
            axis=0,
        )
        video = preprocess_video_rgb_uint8(
            frames,
            self.resolution,
            value_range="minus_one_to_one",
        )
        context_indices = torch.arange(self.num_context_frames, dtype=torch.long)
        return {
            "video": video,
            "context_video": video[:, context_indices].contiguous(),
            "caption": str(raw["prompt"]),
            "video_path": f"openvid-parquet://row/{row_index}",
            "frame_indices": torch.arange(self.num_frames, dtype=torch.long),
            "context_frame_indices": context_indices,
            "num_context_frames": self.num_context_frames,
            "metadata": {
                "dataset_source": "openvid",
                "openvid_row_index": row_index,
                "source_root": str(self.root),
                "sampled_frame_count": self.num_frames,
            },
        }


class WeightedNoGTBoxMixture(Dataset):
    """Concatenate datasets while assigning a fixed total probability per source."""

    dataset_kind = "weighted_no_gt_box_mixture"

    def __init__(
        self,
        datasets: Sequence[Dataset],
        source_names: Sequence[str],
        source_probabilities: Sequence[float],
    ) -> None:
        if not (len(datasets) == len(source_names) == len(source_probabilities)):
            raise ValueError("datasets, source_names, and source_probabilities must have equal length")
        if not datasets:
            raise ValueError("at least one source dataset is required")

        lengths = [len(dataset) for dataset in datasets]
        if any(length <= 0 for length in lengths):
            raise ValueError(f"all source datasets must be non-empty, got lengths={lengths}")
        probabilities = [float(value) for value in source_probabilities]
        if any(value < 0.0 for value in probabilities) or sum(probabilities) <= 0.0:
            raise ValueError(f"source probabilities must be non-negative with positive sum: {probabilities}")
        probability_sum = sum(probabilities)
        probabilities = [value / probability_sum for value in probabilities]

        self.datasets = list(datasets)
        self.source_names = [str(name) for name in source_names]
        self.source_probabilities = probabilities
        self.source_lengths = lengths
        self.cumulative_lengths = np.cumsum(lengths).tolist()
        self.sample_weights = [
            probability / source_length
            for probability, source_length in zip(probabilities, lengths)
            for _ in range(source_length)
        ]
        self.load_from_cache = False
        self.dataset_stats = {
            "name": "PyBullet+Kubric+OpenVid replay mixture",
            "kind": self.dataset_kind,
            "num_samples": len(self),
            "source_lengths": dict(zip(self.source_names, self.source_lengths)),
            "source_probabilities": dict(zip(self.source_names, self.source_probabilities)),
        }

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_id = bisect_right(self.cumulative_lengths, index)
        previous = 0 if source_id == 0 else self.cumulative_lengths[source_id - 1]
        sample = dict(self.datasets[source_id][index - previous])
        source_name = self.source_names[source_id]
        metadata = dict(sample.get("metadata", {}))
        metadata.update(
            {
                "dataset_source": source_name,
                "dataset_source_id": source_id,
                "dataset_source_probability": self.source_probabilities[source_id],
            }
        )
        sample["metadata"] = metadata
        return sample
