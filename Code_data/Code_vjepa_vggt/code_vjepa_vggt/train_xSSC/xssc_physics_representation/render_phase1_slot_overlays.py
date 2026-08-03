#!/usr/bin/env python3
"""Render all-slot xSSC attention overlays for every Phase-1 frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


DEFAULT_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_physics_representation/phase1")
PALETTE = np.asarray(
    [
        [230, 57, 70],
        [45, 156, 219],
        [52, 199, 89],
        [246, 166, 35],
        [153, 102, 204],
        [0, 188, 174],
        [240, 98, 146],
        [173, 125, 64],
        [128, 140, 153],
        [205, 220, 57],
        [74, 84, 201],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No readable frames: {path}")
    return np.stack(frames), fps


def labels_to_source(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    size = 256
    scale = min(size / width, size / height)
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    labels_256 = cv2.resize(labels.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    labels_valid = labels_256[top : top + resized_h, left : left + resized_w]
    return cv2.resize(labels_valid, (width, height), interpolation=cv2.INTER_NEAREST)


def render_overlay(frames: np.ndarray, attention: np.ndarray, alpha: float) -> np.ndarray:
    if attention.ndim != 4 or attention.shape[-2:] != (16, 16):
        raise RuntimeError(f"Unexpected attention shape: {attention.shape}")
    frame_count = min(len(frames), len(attention))
    output = np.empty_like(frames[:frame_count])
    for frame_id in range(frame_count):
        labels = attention[frame_id].argmax(axis=0)
        labels = labels_to_source(labels, frames.shape[1], frames.shape[2])
        colors = PALETTE[labels % len(PALETTE)]
        output[frame_id] = np.clip(
            frames[frame_id].astype(np.float32) * (1.0 - alpha) + colors.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
    return output


def write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    output_root = root / "report" / "slot_overlays"
    records = []
    total = len(manifest["cases"]) * len(manifest["models"])
    position = 0
    for case in manifest["cases"]:
        frames, fps = read_video(Path(case["video"]))
        for model in manifest["models"]:
            position += 1
            feature_path = root / "features" / model["name"] / f"{case['case_id']}.npz"
            output_path = output_root / model["name"] / f"{case['case_id']}.mp4"
            if args.force or not output_path.is_file():
                with np.load(feature_path) as item:
                    attention = item["attention"].astype(np.float32)
                overlay = render_overlay(frames, attention, args.alpha)
                write_video(output_path, overlay, fps)
                state = "rendered"
            else:
                state = "cached"
            records.append(
                {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "model": model["name"],
                    "short_name": model["short_name"],
                    "video": str(output_path.relative_to(root / "report")),
                    "frames": min(len(frames), 150),
                    "fps": fps,
                }
            )
            print(f"[{position:03d}/{total:03d}] {state} {model['name']} / {case['case_id']}", flush=True)
    payload = {
        "definition": "argmax over all xSSC slots at each 16x16 patch, nearest-resized through the 256x256 letterbox transform and overlaid on every source frame",
        "alpha": args.alpha,
        "palette_rgb": PALETTE.tolist(),
        "records": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[complete] overlays={len(records)} manifest={output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
