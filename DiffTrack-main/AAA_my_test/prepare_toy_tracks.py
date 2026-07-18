#!/usr/bin/env python3
"""Build CoTracker pseudo-ground-truth tracks for the 0718 toy videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cotracker-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video(path: Path, num_frames: int, height: int, width: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != num_frames:
        raise ValueError(f"{path} contains {len(frames)} frames, expected at least {num_frames}")

    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    # Match DiffTrack's TAP-Vid preprocessing exactly: source -> 256 -> 480x720.
    video = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False)
    return F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)


def make_queries(grid_size: int, height: int, width: int, device: str) -> torch.Tensor:
    margin_x = width / (2 * grid_size)
    margin_y = height / (2 * grid_size)
    xs = torch.linspace(margin_x, width - margin_x, grid_size, device=device)
    ys = torch.linspace(margin_y, height - margin_y, grid_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    return torch.cat((torch.zeros_like(xy[:, :1]), xy), dim=-1).unsqueeze(0)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.cotracker_root))
    from cotracker.predictor import CoTrackerPredictor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.dataset_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under {args.dataset_root}")

    by_digest: dict[str, list[Path]] = {}
    for path in videos:
        by_digest.setdefault(sha256(path), []).append(path)

    model = CoTrackerPredictor(checkpoint=str(args.checkpoint), offline=True).to(args.device).eval()
    queries = make_queries(args.grid_size, args.height, args.width, args.device)
    manifest = {"dataset_root": str(args.dataset_root), "samples": [], "aliases": {}}

    for index, (digest, paths) in enumerate(sorted(by_digest.items(), key=lambda item: str(item[1][0]))):
        sample_id = f"{index:03d}_{paths[0].stem}"
        output_path = args.output_dir / f"{sample_id}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"Reuse {output_path}")
        else:
            video = read_video(paths[0], args.num_frames, args.height, args.width)
            with torch.inference_mode():
                tracks, visibility = model(video.unsqueeze(0).to(args.device), queries=queries)
            np.savez_compressed(
                output_path,
                tracks=tracks[0].cpu().numpy().astype(np.float32),
                visibility=visibility[0].cpu().numpy().astype(np.bool_),
                queries=queries[0].cpu().numpy().astype(np.float32),
            )
            print(f"Saved {output_path}: tracks={tuple(tracks.shape)}")

        relative_paths = [str(path.relative_to(args.dataset_root)) for path in paths]
        manifest["samples"].append(
            {
                "sample_id": sample_id,
                "sha256": digest,
                "canonical_video": relative_paths[0],
                "all_video_paths": relative_paths,
                "tracks": output_path.name,
            }
        )
        for path in relative_paths:
            manifest["aliases"][path] = sample_id

    manifest.update(
        {
            "num_manifest_videos": len(videos),
            "num_unique_videos": len(by_digest),
            "num_frames": args.num_frames,
            "height": args.height,
            "width": args.width,
            "grid_size": args.grid_size,
            "cotracker_checkpoint": str(args.checkpoint),
        }
    )
    manifest_path = args.output_dir / "tracks_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
