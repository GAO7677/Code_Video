from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class DepthAnythingCache:
    depth_frames: torch.Tensor
    frame_indices: list[int]
    source_video: str | None = None
    output_file: str | None = None
    q_low: float | None = None
    q_high: float | None = None
    dtype: str | None = None


def _candidate_cache_names(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []
    video_path = sample.get("video_path")
    if isinstance(video_path, str) and video_path.strip():
        path = Path(video_path.strip())
        names.extend(
            [
                f"{path.stem}.depth_anything.pt",
                f"{path.name}.depth_anything.pt",
            ]
        )
    sample_name = sample.get("sample_name")
    if isinstance(sample_name, str) and sample_name.strip():
        names.append(f"{sample_name.strip()}.depth_anything.pt")
    stem = sample.get("stem")
    if isinstance(stem, str) and stem.strip():
        names.append(f"{stem.strip()}.depth_anything.pt")
    return list(dict.fromkeys(names))


def load_depth_anything_cache(
    sample: dict[str, Any],
    cache_root: str | Path,
    *,
    allow_missing: bool = True,
) -> DepthAnythingCache | None:
    cache_root = Path(cache_root).expanduser().resolve()
    candidates = _candidate_cache_names(sample)
    for rel_name in candidates:
        candidate = cache_root / rel_name
        if not candidate.is_file():
            continue
        payload = torch.load(candidate, map_location="cpu")
        depth_frames = payload.get("depth_frames")
        if depth_frames is None:
            raise RuntimeError(f"invalid Depth Anything cache payload: {candidate}")
        if not isinstance(depth_frames, torch.Tensor):
            depth_frames = torch.as_tensor(depth_frames)
        return DepthAnythingCache(
            depth_frames=depth_frames.float(),
            frame_indices=[int(v) for v in payload.get("frame_indices", [])],
            source_video=payload.get("source_video"),
            output_file=payload.get("output_file"),
            q_low=float(payload["q_low"]) if payload.get("q_low") is not None else None,
            q_high=float(payload["q_high"]) if payload.get("q_high") is not None else None,
            dtype=payload.get("dtype"),
        )
    if allow_missing:
        return None
    raise FileNotFoundError(
        f"Depth Anything cache not found under {cache_root}. "
        f"tried: {', '.join(candidates) if candidates else '<none>'}"
    )
