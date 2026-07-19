#!/usr/bin/env python3
"""Prepare CoTracker pseudo-GT tracks for 0718ToyDataset base case videos only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from AAA_my_test.sam2_region_query_utils import DEFAULT_CACHE_ROOT, load_region_cache


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
    parser.add_argument("--region-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--source-start-frame", type=int, default=4)
    parser.add_argument("--points-per-region", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_video(
    path: Path, num_frames: int, height: int, width: int, source_start_frame: int
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_index >= source_start_frame:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"{path} has no frames at or after {source_start_frame}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    video = F.interpolate(video, size=(256, 256), mode="bilinear", align_corners=False)
    return F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)


def make_region_queries(
    case_key: str,
    cache_root: Path,
    height: int,
    width: int,
    points_per_region: int,
    device: str,
) -> tuple[torch.Tensor, list[dict]]:
    cache = load_region_cache(cache_root, case_key)
    source_h, source_w = cache.context_frame_rgb.shape[:2]
    xy = torch.from_numpy(cache.query_points.copy()).float().to(device)
    xy[:, 0] *= width / source_w
    xy[:, 1] *= height / source_h
    regions = []
    for region in cache.regions:
        if region.point_end - region.point_start != points_per_region:
            raise ValueError(
                f"{case_key}/{region.region_name}: expected {points_per_region} cached points"
            )
        regions.append(region.__dict__.copy())
    return torch.cat((torch.zeros_like(xy[:, :1]), xy), dim=-1).unsqueeze(0)


def load_case_videos(dataset_root: Path) -> list[dict[str, str]]:
    cases = []
    for manifest_path in sorted((dataset_root / "cases").glob("case_*/case_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = payload["base"]
        video_path = Path(base["video"])
        if not video_path.is_file():
            raise FileNotFoundError(f"missing base video: {video_path}")
        cases.append(
            {
                "case_key": str(payload["case_key"]),
                "manifest": str(manifest_path),
                "canonical_video": str(video_path.relative_to(dataset_root)),
            }
        )
    if not cases:
        raise RuntimeError(f"no case manifests found under {dataset_root}")
    return cases


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.cotracker_root))
    from cotracker.predictor import CoTrackerPredictor

    cases = load_case_videos(args.dataset_root.resolve())
    model = CoTrackerPredictor(checkpoint=str(args.checkpoint), offline=True).to(args.device).eval()
    manifest = {
        "dataset_root": str(args.dataset_root.resolve()),
        "samples": [],
        "aliases": {},
        "num_manifest_videos": len(cases),
        "num_unique_videos": len(cases),
        "num_frames": int(args.num_frames),
        "height": int(args.height),
        "width": int(args.width),
        "query_mode": "sam2_regions",
        "region_cache_root": str(args.region_cache_root.resolve()),
        "source_start_frame": int(args.source_start_frame),
        "points_per_region": int(args.points_per_region),
        "cotracker_checkpoint": str(args.checkpoint),
    }

    for case in cases:
        sample_id = str(case["case_key"])
        queries, regions = make_region_queries(
            sample_id,
            args.region_cache_root,
            args.height,
            args.width,
            args.points_per_region,
            args.device,
        )
        output_path = args.output_dir / f"{sample_id}.npz"
        video_path = args.dataset_root / case["canonical_video"]
        if output_path.exists() and not args.overwrite:
            print(f"Reuse {output_path}", flush=True)
        else:
            video = read_video(
                video_path,
                args.num_frames,
                args.height,
                args.width,
                args.source_start_frame,
            )
            with torch.inference_mode():
                tracks, visibility = model(video.unsqueeze(0).to(args.device), queries=queries)
            np.savez_compressed(
                output_path,
                tracks=tracks[0].cpu().numpy().astype(np.float32),
                visibility=visibility[0].cpu().numpy().astype(np.bool_),
                queries=queries[0].cpu().numpy().astype(np.float32),
            )
            print(f"Saved {output_path}: tracks={tuple(tracks.shape)}", flush=True)
        manifest["samples"].append(
            {
                "sample_id": sample_id,
                "case_key": str(case["case_key"]),
                "case_manifest": str(case["manifest"]),
                "canonical_video": case["canonical_video"],
                "all_video_paths": [case["canonical_video"]],
                "tracks": output_path.name,
                "regions": regions,
            }
        )
        manifest["aliases"][case["canonical_video"]] = sample_id

    manifest_path = args.output_dir / "tracks_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
