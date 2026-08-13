"""Datasets over fixed-shape offline causal trajectory records."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import NUM_SLOTS, NUM_STATES, SLOT_DIM
from .io_utils import read_jsonl


REQUIRED_RECORD_KEYS = {
    "slots",
    "slot_attention",
    "gt_mask",
    "gt_position",
    "gt_velocity",
    "gt_image_position",
    "gt_bbox",
    "gt_visibility",
    "object_valid",
    "prefix_slot_to_object",
    "boundary_slot_to_object",
    "source",
}


def validate_record(record: dict) -> None:
    missing = REQUIRED_RECORD_KEYS.difference(record)
    if missing:
        raise KeyError(f"Trajectory record is missing keys: {sorted(missing)}")
    if tuple(record["slots"].shape) != (NUM_STATES, NUM_SLOTS, SLOT_DIM):
        raise ValueError(f"Unexpected slot shape: {record['slots'].shape}")
    if record["slot_attention"].shape[:2] != (NUM_STATES, NUM_SLOTS):
        raise ValueError("Unexpected slot-attention shape")
    for key in ("prefix_slot_to_object", "boundary_slot_to_object"):
        if tuple(record[key].shape) != (NUM_SLOTS,):
            raise ValueError(f"Unexpected {key} shape: {record[key].shape}")


class TrajectoryDataset(Dataset):
    def __init__(self, cache_root: Path, split: str):
        self.cache_root = Path(cache_root)
        self.split = split
        manifest = self.cache_root / split / "records.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing cache manifest: {manifest}")
        self.entries = read_jsonl(manifest)

    def __len__(self):
        return len(self.entries)

    @lru_cache(maxsize=16)
    def _load(self, index):
        entry = self.entries[index]
        record_path = self.cache_root / self.split / entry["record"]
        record = torch.load(record_path, map_location="cpu", weights_only=True)
        validate_record(record)
        return record

    def __getitem__(self, index):
        return self._load(int(index))


class PredictionWindowDataset(Dataset):
    """One-step windows with common origins for H=1/2/4 comparisons."""

    def __init__(
        self,
        cache_root: Path,
        split: str,
        history: int,
        first_origin: int = 3,
        last_origin: int = 10,
    ):
        if history not in {1, 2, 4}:
            raise ValueError(f"Unsupported history: {history}")
        self.trajectories = TrajectoryDataset(cache_root, split)
        self.history = history
        self.origins = tuple(range(first_origin, last_origin + 1))
        if self.origins[0] < history - 1 or self.origins[-1] >= NUM_STATES - 1:
            raise ValueError("Prediction origins are incompatible with history/states")

    def __len__(self):
        return len(self.trajectories) * len(self.origins)

    def __getitem__(self, index):
        trajectory_index, origin_offset = divmod(index, len(self.origins))
        origin = self.origins[origin_offset]
        record = self.trajectories[trajectory_index]
        return {
            "history": record["slots"][origin - self.history + 1 : origin + 1].float(),
            "target": record["slots"][origin + 1].float(),
            "slot_valid": torch.ones(NUM_SLOTS, dtype=torch.bool),
            "trajectory_index": torch.tensor(trajectory_index),
            "origin": torch.tensor(origin),
        }


def gather_object_targets(
    record: dict,
    mapping_key: str = "prefix_slot_to_object",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Gather fixed GT identities into the slot axis for all states."""
    mapping = record[mapping_key].long()
    clamped = mapping.clamp_min(0)
    object_valid = record["object_valid"].bool()
    mapped_valid = mapping >= 0
    mapped_valid &= object_valid.gather(0, clamped)

    def gather(value):
        index = clamped[None, :, None].expand(value.shape[0], -1, value.shape[-1])
        return value.gather(1, index)

    visibility = record["gt_visibility"].gather(
        1, clamped[None].expand(record["gt_visibility"].shape[0], -1)
    )
    target = {
        "position": gather(record["gt_position"].float()),
        "velocity": gather(record["gt_velocity"].float()),
        "image_position": gather(record["gt_image_position"].float()),
        "bbox": gather(record["gt_bbox"].float()),
        "presence": visibility > 0,
    }
    valid = mapped_valid[None].expand(record["slots"].shape[0], -1)
    return target, valid
