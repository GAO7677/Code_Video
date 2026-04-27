from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class BenchSample:
    sample_id: str
    prompt: str
    video_path: str
    context_frames_dir: str | None = None
    context_frame_paths: list[str] = field(default_factory=list)
    image_path: str | None = None
    gt_video_path: str | None = None
    generated_start_frame: int | None = None
    gt_start_frame: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


_KNOWN_KEYS = {
    "sample_id",
    "id",
    "prompt",
    "video_path",
    "context_frames_dir",
    "context_frame_paths",
    "image_path",
    "gt_video_path",
    "generated_start_frame",
    "gt_start_frame",
}


def _resolve(path_value: str | None) -> str | None:
    if path_value in (None, ""):
        return None
    return str(Path(path_value).expanduser().resolve())


def _parse_sample(raw: dict[str, Any], idx: int) -> BenchSample:
    sample_id = str(raw.get("sample_id") or raw.get("id") or f"sample_{idx:05d}")
    prompt = str(raw["prompt"])
    video_path = _resolve(raw["video_path"])
    context_frames_dir = _resolve(raw.get("context_frames_dir"))
    context_frame_paths = [_resolve(p) for p in raw.get("context_frame_paths", [])]
    image_path = _resolve(raw.get("image_path"))
    gt_video_path = _resolve(raw.get("gt_video_path"))
    generated_start_frame = raw.get("generated_start_frame")
    gt_start_frame = raw.get("gt_start_frame")
    extras = {k: v for k, v in raw.items() if k not in _KNOWN_KEYS}
    return BenchSample(
        sample_id=sample_id,
        prompt=prompt,
        video_path=video_path,
        context_frames_dir=context_frames_dir,
        context_frame_paths=context_frame_paths,
        image_path=image_path,
        gt_video_path=gt_video_path,
        generated_start_frame=generated_start_frame,
        gt_start_frame=gt_start_frame,
        extras=extras,
    )


def load_manifest(path: str) -> list[BenchSample]:
    manifest_path = Path(path).expanduser().resolve()
    suffix = manifest_path.suffix.lower()

    rows: list[dict[str, Any]]
    if suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            rows = data.get("samples", [])
        else:
            rows = data
    elif suffix == ".jsonl":
        rows = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    samples = [_parse_sample(raw, idx) for idx, raw in enumerate(rows)]
    if not samples:
        raise ValueError(f"No samples found in manifest: {manifest_path}")
    return samples

