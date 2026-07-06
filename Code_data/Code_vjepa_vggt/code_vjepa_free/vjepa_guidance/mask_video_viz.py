from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg",
        "/home/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("ffmpeg is required to write H.264 mp4 output")


def write_mp4_h264(path: Path, frames_thwc_uint8: np.ndarray, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    tmp_path = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {tmp_path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    try:
        ffmpeg = find_ffmpeg()
    except RuntimeError:
        tmp_path.replace(path)
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(tmp_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    tmp_path.unlink(missing_ok=True)


def ensure_browser_video(source_path: Path) -> Path:
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if out_path.exists() and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns and out_path.stat().st_size > 0:
        return out_path
    try:
        ffmpeg = find_ffmpeg()
    except RuntimeError:
        return source_path
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def render_binary_mask_video(
    mask_thw: np.ndarray,
    *,
    fg_color: tuple[int, int, int] = (70, 220, 120),
    bg_color: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    mask = np.clip(mask_thw.astype(np.float32), 0.0, 1.0)
    frames, height, width = mask.shape
    out = np.zeros((frames, height, width, 3), dtype=np.uint8)
    out[...] = np.asarray(bg_color, dtype=np.uint8)
    binary = mask > 0.5
    out[binary] = np.asarray(fg_color, dtype=np.uint8)
    return out


def render_motion_overlay_video(
    video_thwc_u8: np.ndarray,
    mask_thw: np.ndarray,
    *,
    color: tuple[int, int, int] = (70, 220, 120),
    alpha: float = 0.45,
) -> np.ndarray:
    video = video_thwc_u8.astype(np.float32)
    mask = np.clip(mask_thw.astype(np.float32), 0.0, 1.0)[..., None]
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 1, 3)
    out = video * (1.0 - alpha * mask) + tint * (alpha * mask)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def render_background_overlay_video(
    video_thwc_u8: np.ndarray,
    motion_mask_thw: np.ndarray,
    *,
    color: tuple[int, int, int] = (70, 140, 245),
) -> np.ndarray:
    video = video_thwc_u8.astype(np.float32)
    motion = np.clip(motion_mask_thw.astype(np.float32), 0.0, 1.0)[..., None]
    background = 1.0 - motion
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 1, 3)
    out = video * (1.0 - 0.40 * background) + tint * (0.40 * background)
    out = out * (0.85 + 0.15 * motion)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
