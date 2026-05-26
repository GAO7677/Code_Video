from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_video_frames(path: Path) -> list[np.ndarray]:
    try:
        reader = imageio.get_reader(str(path))
        try:
            return [np.asarray(frame, dtype=np.uint8) for frame in reader]
        finally:
            reader.close()
    except Exception:
        cap = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if frames:
            return frames
        raise


def save_video_frames(path: Path, frames: list[np.ndarray], fps: int, quality: int = 5) -> None:
    ensure_dir(path.parent)
    with imageio.get_writer(
        str(path),
        fps=max(int(fps), 1),
        codec="libx264",
        quality=int(quality),
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def pil_list_to_numpy(frames: list[Image.Image]) -> list[np.ndarray]:
    return [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]


def extract_first_frame(video_path: Path) -> Image.Image:
    frames = load_video_frames(video_path)
    if not frames:
        raise ValueError(f"Video has no frames: {video_path}")
    return Image.fromarray(frames[0]).convert("RGB")


def resize_crop_frame(frame: Image.Image, width: int, height: int) -> Image.Image:
    src_width, src_height = frame.size
    src_ratio = src_width / max(src_height, 1)
    dst_ratio = width / max(height, 1)
    if src_ratio > dst_ratio:
        new_height = height
        new_width = int(round(src_width * (height / src_height)))
    else:
        new_width = width
        new_height = int(round(src_height * (width / src_width)))
    resized = frame.resize((new_width, new_height), Image.Resampling.BILINEAR)
    left = max(0, (new_width - width) // 2)
    top = max(0, (new_height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def resize_pad_frame(frame: Image.Image, width: int, height: int) -> Image.Image:
    src_width, src_height = frame.size
    scale = min(width / max(src_width, 1), height / max(src_height, 1))
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))
    resized = frame.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - new_width) // 2, (height - new_height) // 2))
    return canvas


def load_context_frames(
    video_path: Path,
    *,
    context_frames: int,
    width: int,
    height: int,
    resize_mode: str = "crop",
) -> list[Image.Image]:
    raw_frames = load_video_frames(video_path)
    if not raw_frames:
        raise ValueError(f"Context video has no frames: {video_path}")
    keep = min(max(int(context_frames), 1), len(raw_frames))
    frames: list[Image.Image] = []
    for frame in raw_frames[:keep]:
        image = Image.fromarray(frame).convert("RGB")
        if resize_mode == "pad":
            image = resize_pad_frame(image, width=width, height=height)
        else:
            image = resize_crop_frame(image, width=width, height=height)
        frames.append(image)
    return frames


def uniform_subsample_indices(total: int, keep: int) -> np.ndarray:
    if keep >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, keep, dtype=np.int64)


def uniform_subsample_frames(frames: list[np.ndarray], keep: int) -> list[np.ndarray]:
    if keep <= 0 or len(frames) <= keep:
        return list(frames)
    indices = uniform_subsample_indices(len(frames), keep)
    return [frames[int(index)] for index in indices]


def symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def detect_video_fps(video_path: Path, fallback: int = 16) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if fps is None or fps <= 0:
        return int(fallback)
    return int(round(float(fps)))
