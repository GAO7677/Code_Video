from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, sample_frame_indices


@dataclass(frozen=True)
class RawWindow:
    video_path: Path
    window_start: int


_SHAPE_PHRASES = {
    "sphere": ("ball", "round rigid object"),
    "puck": ("puck", "flat round rigid object"),
    "box": ("block", "box-shaped rigid object"),
    "cylinder": ("cylinder", "cylindrical rigid object"),
}


def _objects_by_shape(metadata: dict[str, Any], shape: str) -> list[dict[str, Any]]:
    return [
        item
        for item in metadata.get("objects", [])
        if isinstance(item, dict) and str(item.get("shape")) == shape
    ]


def _motion_direction(metadata: dict[str, Any]) -> str:
    dynamic = [
        item
        for item in metadata.get("objects", [])
        if isinstance(item, dict) and bool(item.get("dynamic"))
    ]
    if not dynamic:
        return "across the scene"
    velocity = dynamic[0].get("linear_velocity", [0.0, 0.0, 0.0])
    if isinstance(velocity, list) and velocity:
        if float(velocity[0]) > 0.2:
            return "from left to right"
        if float(velocity[0]) < -0.2:
            return "from right to left"
    return "across the scene"


def _english_caption(metadata: dict[str, Any], family_slug: str) -> str:
    direction = _motion_direction(metadata)
    spheres = _objects_by_shape(metadata, "sphere")
    pucks = _objects_by_shape(metadata, "puck")
    boxes = _objects_by_shape(metadata, "box")
    cylinders = _objects_by_shape(metadata, "cylinder")
    if family_slug == "F1_single_object" and spheres:
        return (
            f"A ball, a round rigid object, enters {direction}, falls under gravity, "
            "bounces on the floor, and rolls forward."
        )
    if family_slug == "F2_two_object":
        mover, generic = _SHAPE_PHRASES["puck"] if pucks else _SHAPE_PHRASES["sphere"]
        target, target_generic = (
            _SHAPE_PHRASES["box"] if boxes else ("target", "stationary rigid object")
        )
        return (
            f"A moving {mover}, a {generic}, travels {direction} and collides with a "
            f"{target}, a {target_generic}, transferring momentum and causing deflection."
        )
    if family_slug == "F3_chain_reaction" and spheres and len(boxes) >= 2:
        return (
            f"A moving ball, a round rigid object, travels {direction} and hits the first "
            "block, causing two box-shaped rigid objects to move in sequence in a chain reaction."
        )
    if family_slug == "F4_occlusion" and spheres and cylinders:
        return (
            f"A moving ball, a round rigid object, travels {direction}, passes behind two "
            "stationary cylinders acting as occluders, and continues its motion after reappearing."
        )
    if family_slug == "F5_drop_support" and spheres and boxes:
        return (
            "A ball, a round rigid object, falls under gravity onto a stationary block-shaped "
            "support, then rolls across and leaves the supporting surface."
        )
    shape_terms = [
        _SHAPE_PHRASES[str(item["shape"])][0]
        for item in metadata.get("objects", [])
        if isinstance(item, dict) and str(item.get("shape")) in _SHAPE_PHRASES
    ]
    visible = ", ".join(shape_terms) if shape_terms else "rigid bodies"
    return f"The scene contains {visible}, which move and interact according to rigid-body physics."


class PyBulletRawNoGTBoxDataset(Dataset):
    """Read complete raw PyBullet videos for the no-GT-box Stage1B branch."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: tuple[int, int],
        *,
        num_frames: int = 49,
        num_context_frames: int = 8,
        sampling_strategy: str = "prefix",
        window_starts: tuple[int, ...] = (0,),
        init_scan_limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.resolution = resolution
        self.num_frames = int(num_frames)
        self.num_context_frames = int(num_context_frames)
        self.sampling_strategy = str(sampling_strategy)
        self.window_starts = tuple(dict.fromkeys(int(value) for value in window_starts))
        if self.num_context_frames > self.num_frames:
            raise ValueError("num_context_frames cannot exceed num_frames")
        if self.sampling_strategy not in {"prefix", "uniform"}:
            raise ValueError(f"unsupported sampling_strategy={self.sampling_strategy!r}")
        if not self.window_starts or min(self.window_starts) < 0:
            raise ValueError(f"window_starts must contain non-negative values: {self.window_starts}")
        if self.sampling_strategy == "uniform" and self.window_starts != (0,):
            raise ValueError("window_starts are only supported with prefix sampling")

        split_root = self.root / self.split
        videos = sorted(split_root.glob("*/sample_*/video.mp4"))
        if init_scan_limit is not None:
            videos = videos[: max(1, int(init_scan_limit))]
        self.samples = [
            RawWindow(video_path=video_path, window_start=window_start)
            for video_path in videos
            for window_start in self.window_starts
        ]
        if not self.samples:
            raise RuntimeError(f"no */sample_*/video.mp4 found under {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.samples[idx]
        video_path = record.video_path
        meta_path = video_path.with_name("meta.json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        vr = VideoReader(str(video_path), ctx=cpu(0))
        source_frames = len(vr)
        if source_frames < self.num_frames + record.window_start:
            raise RuntimeError(
                f"{video_path} has {source_frames} frames, requested window "
                f"[{record.window_start}, {record.window_start + self.num_frames})"
            )
        if self.sampling_strategy == "uniform":
            frame_indices_np = sample_frame_indices(source_frames, self.num_frames)
        else:
            frame_indices_np = np.arange(
                record.window_start,
                record.window_start + self.num_frames,
                dtype=np.int64,
            )
        frames = vr.get_batch(frame_indices_np).asnumpy()
        video = preprocess_video_rgb_uint8(
            frames,
            self.resolution,
            value_range="minus_one_to_one",
        )
        context_indices = torch.arange(self.num_context_frames, dtype=torch.long)
        family_slug = video_path.parent.parent.name
        caption = _english_caption(metadata, family_slug)
        metadata = {
            **metadata,
            "source_video_path": str(video_path),
            "source_frame_count": int(source_frames),
            "sampled_frame_indices": frame_indices_np.tolist(),
            "sampling_strategy": self.sampling_strategy,
            "window_start": record.window_start,
            "family_slug": family_slug,
        }
        return {
            "video": video,
            "context_video": video[:, context_indices].contiguous(),
            "caption": caption,
            "video_path": str(video_path),
            "frame_indices": torch.arange(self.num_frames, dtype=torch.long),
            "context_frame_indices": context_indices,
            "num_context_frames": self.num_context_frames,
            "metadata": metadata,
        }
