from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
from pathlib import Path

from .manifest import BenchSample


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".gif", ".mov", ".avi", ".mkv"}


@dataclass
class StagedDataset:
    root_dir: str
    video_dir: str
    prompt_map: dict[str, str]
    sample_to_video_name: dict[str, str]
    image_dir: str | None = None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(sample: BenchSample, max_len: int = 140) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.sample_id).strip("._") or "sample"
    prompt = sample.prompt.replace("/", " ").replace("\\", " ").replace("\n", " ")
    prompt = re.sub(r"\s+", " ", prompt).strip()
    prompt = re.sub(r"[^A-Za-z0-9 _.,()'\-]+", "_", prompt)
    if not prompt:
        prompt = "prompt"
    stem = f"{prefix}__{prompt}"
    return stem[:max_len].rstrip(" ._")


def _link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _choose_last_context_image(sample: BenchSample) -> Path:
    if sample.image_path:
        path = Path(sample.image_path)
        if not path.exists():
            raise FileNotFoundError(f"image_path does not exist for sample {sample.sample_id}: {path}")
        return path

    if sample.context_frame_paths:
        paths = [Path(p) for p in sample.context_frame_paths]
    elif sample.context_frames_dir:
        ctx_dir = Path(sample.context_frames_dir)
        if not ctx_dir.exists():
            raise FileNotFoundError(f"context_frames_dir does not exist for sample {sample.sample_id}: {ctx_dir}")
        paths = sorted(
            [p for p in ctx_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )
    else:
        raise ValueError(
            f"Sample {sample.sample_id} does not have context frames or image_path, "
            "but I2V metrics require one of them."
        )

    if not paths:
        raise ValueError(f"No context frames found for sample {sample.sample_id}")
    return paths[-1]


def stage_custom_vbench_dataset(
    samples: list[BenchSample],
    staging_root: str,
    use_symlink: bool = True,
    with_images: bool = False,
) -> StagedDataset:
    root_dir = _ensure_dir(Path(staging_root).expanduser().resolve())
    video_dir = _ensure_dir(root_dir / "videos")
    image_dir = _ensure_dir(root_dir / "images") if with_images else None

    prompt_map: dict[str, str] = {}
    sample_to_video_name: dict[str, str] = {}

    for sample in samples:
        src_video = Path(sample.video_path)
        if not src_video.exists():
            raise FileNotFoundError(f"video_path does not exist for sample {sample.sample_id}: {src_video}")
        if src_video.suffix.lower() not in VIDEO_EXTS:
            raise ValueError(f"Unsupported video suffix for {src_video}: {src_video.suffix}")

        stem = safe_stem(sample)
        dst_video_name = f"{stem}{src_video.suffix.lower()}"
        dst_video = video_dir / dst_video_name
        _link_or_copy(src_video, dst_video, use_symlink=use_symlink)

        prompt_map[dst_video_name] = sample.prompt
        sample_to_video_name[sample.sample_id] = dst_video_name

        if image_dir is not None:
            src_image = _choose_last_context_image(sample)
            dst_image = image_dir / f"{stem}{src_image.suffix.lower()}"
            _link_or_copy(src_image, dst_image, use_symlink=use_symlink)

    return StagedDataset(
        root_dir=str(root_dir),
        video_dir=str(video_dir),
        image_dir=str(image_dir) if image_dir is not None else None,
        prompt_map=prompt_map,
        sample_to_video_name=sample_to_video_name,
    )

