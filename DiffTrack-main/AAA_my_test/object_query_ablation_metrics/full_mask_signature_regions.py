#!/usr/bin/env python3
"""Map frozen per-object video masks to disjoint latent-token signatures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


FRAME_COUNT = 49
ANCHOR_FRAMES = np.arange(13, dtype=np.int64) * 4
PIXEL_HW = (704, 1280)
LATENT_GRID = (13, 22, 40)


@dataclass(frozen=True)
class SignaturePartition:
    """A disjoint partition of all mask-covered latent-video tokens."""

    object_names: tuple[str, ...]
    anchor_frames: tuple[int, ...]
    grid: tuple[int, int, int]
    signature_rows: dict[int, tuple[int, ...]]
    signature_rows_by_time: dict[int, tuple[tuple[int, ...], ...]]
    occupancy_by_time: np.ndarray

    @property
    def union_rows(self) -> tuple[int, ...]:
        return tuple(sorted({row for rows in self.signature_rows.values() for row in rows}))

    def signature_label(self, signature: int) -> str:
        names = [
            name
            for index, name in enumerate(self.object_names)
            if signature & (1 << index)
        ]
        if not names:
            raise ValueError("zero is the background signature")
        return "+".join(names)

    def audit(self) -> dict[str, Any]:
        signature_counts = {
            self.signature_label(signature): len(rows)
            for signature, rows in sorted(self.signature_rows.items())
        }
        shared_counts = {
            label: count for label, count in signature_counts.items() if "+" in label
        }
        return {
            "object_names": list(self.object_names),
            "anchor_frames": list(self.anchor_frames),
            "grid": list(self.grid),
            "membership_rule": "latent cell is active for an object iff any frozen SAM2 mask pixel intersects it",
            "signature_token_counts": signature_counts,
            "shared_signature_token_counts": shared_counts,
            "signature_count": len(signature_counts),
            "shared_signature_count": len(shared_counts),
            "union_token_count": len(self.union_rows),
            "union_token_counts_by_time": [
                len(
                    {
                        row
                        for rows in self.signature_rows_by_time.values()
                        for row in rows[time_index]
                    }
                )
                for time_index in range(self.grid[0])
            ],
            "signature_token_indices": {
                self.signature_label(signature): list(rows)
                for signature, rows in sorted(self.signature_rows.items())
            },
            "signature_token_indices_by_time": {
                self.signature_label(signature): [list(rows) for rows in rows_by_time]
                for signature, rows_by_time in sorted(self.signature_rows_by_time.items())
            },
        }


def unpack_mask_cache(path: Path, expected_video: Path | None = None) -> np.ndarray:
    """Read the packed SAM2 cache written by the metric pipeline."""
    with np.load(path, allow_pickle=False) as arrays:
        shape = tuple(int(value) for value in arrays["mask_shape"].tolist())
        packed = arrays["masks_packed"]
        cached_video = Path(str(arrays["video_path"].item()))
    if len(shape) != 4 or shape[0] != FRAME_COUNT or shape[2:] != PIXEL_HW:
        raise RuntimeError(f"unexpected SAM2 mask shape in {path}: {shape}")
    if expected_video is not None and cached_video.resolve() != expected_video.resolve():
        raise RuntimeError(
            f"mask cache is not frozen from the requested Baseline: {cached_video} != {expected_video}"
        )
    count = int(np.prod(shape))
    return np.unpackbits(packed)[:count].reshape(shape).astype(bool)


def build_signature_partition(
    masks: np.ndarray,
    object_names: Iterable[str],
    *,
    anchor_frames: np.ndarray = ANCHOR_FRAMES,
    grid_hw: tuple[int, int] = LATENT_GRID[1:],
) -> SignaturePartition:
    """Quantize full masks by intersection and group cells by object membership."""
    names = tuple(str(name) for name in object_names)
    masks = np.asarray(masks, dtype=bool)
    anchors = np.asarray(anchor_frames, dtype=np.int64)
    if masks.ndim != 4 or masks.shape[1] != len(names):
        raise ValueError(
            f"expected masks [frames,{len(names)},height,width], got {masks.shape}"
        )
    if not len(names) or len(names) > 62:
        raise ValueError("signature partition supports 1..62 objects")
    if anchors.ndim != 1 or not len(anchors) or anchors.min() < 0 or anchors.max() >= masks.shape[0]:
        raise ValueError(f"invalid anchor frames: {anchors.tolist()}")
    grid_h, grid_w = (int(grid_hw[0]), int(grid_hw[1]))
    pixel_h, pixel_w = masks.shape[-2:]
    if pixel_h % grid_h or pixel_w % grid_w:
        raise ValueError(
            f"pixel grid {(pixel_h, pixel_w)} is not divisible by latent grid {(grid_h, grid_w)}"
        )

    cell_h, cell_w = pixel_h // grid_h, pixel_w // grid_w
    anchored = masks[anchors]
    occupancy = anchored.reshape(
        len(anchors), len(names), grid_h, cell_h, grid_w, cell_w
    ).mean(axis=(3, 5), dtype=np.float32)
    active = occupancy > 0.0
    signatures = np.zeros((len(anchors), grid_h, grid_w), dtype=np.int64)
    for object_index in range(len(names)):
        signatures |= active[:, object_index].astype(np.int64) << object_index

    rows: dict[int, tuple[int, ...]] = {}
    rows_by_time: dict[int, tuple[tuple[int, ...], ...]] = {}
    spatial_size = grid_h * grid_w
    for signature in sorted(int(value) for value in np.unique(signatures) if value):
        by_time: list[tuple[int, ...]] = []
        flat_rows: list[int] = []
        for time_index in range(len(anchors)):
            spatial = np.flatnonzero(signatures[time_index].reshape(-1) == signature)
            global_rows = tuple(
                int(time_index * spatial_size + value) for value in spatial.tolist()
            )
            by_time.append(global_rows)
            flat_rows.extend(global_rows)
        rows[signature] = tuple(flat_rows)
        rows_by_time[signature] = tuple(by_time)

    partition = SignaturePartition(
        object_names=names,
        anchor_frames=tuple(int(value) for value in anchors.tolist()),
        grid=(len(anchors), grid_h, grid_w),
        signature_rows=rows,
        signature_rows_by_time=rows_by_time,
        occupancy_by_time=occupancy,
    )
    _validate_partition(partition)
    return partition


def _validate_partition(partition: SignaturePartition) -> None:
    seen: set[int] = set()
    limit = int(np.prod(partition.grid))
    for signature, rows in partition.signature_rows.items():
        if signature <= 0 or not rows:
            raise RuntimeError(f"invalid empty/non-object signature {signature}")
        overlap = seen.intersection(rows)
        if overlap:
            raise RuntimeError(f"signature rows are not disjoint: {sorted(overlap)[:8]}")
        if min(rows) < 0 or max(rows) >= limit:
            raise RuntimeError(f"signature {signature} contains out-of-grid rows")
        seen.update(rows)
    if not seen:
        raise RuntimeError("full-mask partition contains no object token")


def torch_signature_groups(
    partition: SignaturePartition, device: torch.device
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Return M1 self-block groups and their full object-region union."""
    groups = []
    for rows in partition.signature_rows.values():
        tensor = torch.as_tensor(rows, device=device, dtype=torch.long)
        groups.append((tensor, tensor))
    union = torch.as_tensor(partition.union_rows, device=device, dtype=torch.long)
    return groups, union
