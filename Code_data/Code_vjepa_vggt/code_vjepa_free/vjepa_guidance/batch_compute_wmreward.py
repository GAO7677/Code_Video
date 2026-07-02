#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader

try:
    from .vjepa_surprise import VJEPASurpriseEnergy
except ImportError:
    from vjepa_surprise import VJEPASurpriseEnergy


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def set_deterministic(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def find_videos(input_dir: Path) -> list[Path]:
    videos = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos


def sample_video_frames(video_path: Path, max_frames: int) -> np.ndarray:
    reader = VideoReader(str(video_path))
    num_frames = len(reader)
    frame_count = min(max_frames, num_frames)
    frame_indices = np.linspace(0, num_frames - 1, frame_count, dtype=int)
    return reader.get_batch(frame_indices).asnumpy()


def load_video_as_tensor(video_path: Path, max_frames: int) -> tuple[torch.Tensor, int]:
    video_np = sample_video_frames(video_path, max_frames=max_frames)
    sampled_frames = int(video_np.shape[0])
    video = torch.from_numpy(video_np).float()  # [T,H,W,C]
    video = video.permute(0, 3, 1, 2) / 255.0  # [T,C,H,W]
    video = (video * 2.0) - 1.0
    video = video.permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
    return video.unsqueeze(0), sampled_frames


def choose_window_params(
    *,
    num_frames: int,
    base_window: int,
    base_context: int,
    base_stride: int,
) -> tuple[int, int, int]:
    if num_frames < 2:
        raise ValueError(f"Video has too few frames: {num_frames}")

    window_size = min(base_window, num_frames)
    if window_size < 2:
        raise ValueError(f"window_size became invalid: {window_size}")
    if window_size % 2 == 1:
        window_size -= 1
    if window_size < 2:
        raise ValueError(f"window_size became invalid after even adjustment: {window_size}")

    context_frames = min(base_context, window_size - 2)
    if context_frames < 2:
        context_frames = max(2, window_size // 2)
    if context_frames >= window_size:
        context_frames = window_size - 2
    if context_frames % 2 == 1:
        context_frames -= 1
    if context_frames < 2:
        raise ValueError(
            f"context_frames became invalid for num_frames={num_frames}, window_size={window_size}"
        )

    stride = min(base_stride, window_size)
    stride = max(1, stride)
    return window_size, context_frames, stride


def write_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_path",
                "relative_path",
                "basename",
                "surprise_score",
                "similarity_score",
                "sampled_frames",
                "num_windows",
                "window_size",
                "context_frames",
                "stride",
                "checkpoint_path",
                "model_name",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch compute WMReward-style V-JEPA surprise scores for videos.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"),
    )
    parser.add_argument("--model_name", type=str, default="vitg384", choices=["vith", "vitg", "vitg384"])
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=49)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    set_deterministic(args.seed)
    videos = find_videos(args.input_dir)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"No video files found under {args.input_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    energy_fn = VJEPASurpriseEnergy(
        model_name=args.model_name,
        device=device,
        local_torchhub=True,
        checkpoint_path=args.checkpoint_path,
    )

    rows: list[dict[str, object]] = []
    for idx, video_path in enumerate(videos, start=1):
        relative_path = str(video_path.relative_to(args.input_dir))
        print(f"[{idx}/{len(videos)}] {relative_path}", flush=True)
        try:
            video_tensor, sampled_frames = load_video_as_tensor(video_path, max_frames=args.max_frames)
            window_size, context_frames, stride = choose_window_params(
                num_frames=sampled_frames,
                base_window=args.window_size,
                base_context=args.context_frames,
                base_stride=args.stride,
            )
            with torch.no_grad():
                surprise = float(
                    energy_fn(
                        video_tensor,
                        window_size=window_size,
                        context_frames=context_frames,
                        stride=stride,
                        reduction="mean",
                    ).item()
                )
            similarity = 1.0 - surprise
            num_windows = max(1, (sampled_frames - window_size) // stride + 1)
            print(
                f"  surprise={surprise:.6f} similarity={similarity:.6f} sampled_frames={sampled_frames} "
                f"window={window_size} context={context_frames} stride={stride} windows={num_windows}",
                flush=True,
            )
            rows.append(
                {
                    "video_path": str(video_path),
                    "relative_path": relative_path,
                    "basename": video_path.name,
                    "surprise_score": f"{surprise:.8f}",
                    "similarity_score": f"{similarity:.8f}",
                    "sampled_frames": sampled_frames,
                    "num_windows": num_windows,
                    "window_size": window_size,
                    "context_frames": context_frames,
                    "stride": stride,
                    "checkpoint_path": str(args.checkpoint_path),
                    "model_name": args.model_name,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            rows.append(
                {
                    "video_path": str(video_path),
                    "relative_path": relative_path,
                    "basename": video_path.name,
                    "surprise_score": "",
                    "similarity_score": "",
                    "sampled_frames": "",
                    "num_windows": "",
                    "window_size": "",
                    "context_frames": "",
                    "stride": "",
                    "checkpoint_path": str(args.checkpoint_path),
                    "model_name": args.model_name,
                    "status": "error",
                    "error": repr(exc),
                }
            )
        write_rows(args.output_csv, rows)

    print(f"Saved CSV to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
