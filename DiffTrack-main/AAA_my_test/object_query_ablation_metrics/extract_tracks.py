#!/usr/bin/env python3
"""Run the same CoTracker3 point protocol on all 49 videos and the source render."""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for path in (ROOT, COTRACKER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from AAA_my_test.object_query_ablation_metrics.common import (  # noqa: E402
    BASELINE_TRACKS,
    FRAME_COUNT,
    OUTPUT_ROOT,
    atomic_json,
    atomic_npz,
    load_inventory,
    load_query_data,
    load_video_frames,
    safe_id,
    sha256_file,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (  # noqa: E402
    load_cotracker,
    run_cotracker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", default="", help="optional exact video id")
    return parser.parse_args()


def cache_valid(path: Path, video_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as arrays:
            return (
                tuple(arrays["tracks"].shape) == (FRAME_COUNT, 16, 2)
                and tuple(arrays["visibility"].shape) == (FRAME_COUNT, 16)
                and str(arrays["video_sha256"].item()) == video_sha256
            )
    except (OSError, KeyError, ValueError):
        return False


def main() -> None:
    args = parse_args()
    videos = load_inventory(include_source=True)
    if args.only:
        videos = [row for row in videos if row["id"] == args.only]
        if not videos:
            raise ValueError(f"unknown video id: {args.only}")
    output = OUTPUT_ROOT / "tracks"
    output.mkdir(parents=True, exist_ok=True)
    query_points, _slices, _masks = load_query_data()
    model = None
    records = []
    try:
        for index, video in enumerate(videos, start=1):
            video_id = str(video["id"])
            video_path = Path(video["path"])
            digest = sha256_file(video_path)
            track_path = output / f"{safe_id(video_id)}.npz"
            if cache_valid(track_path, digest) and not args.overwrite:
                print(f"[{index:02d}/{len(videos):02d}] reuse {video_id}", flush=True)
            elif video_id == "baseline" and BASELINE_TRACKS.is_file() and not args.overwrite:
                with np.load(BASELINE_TRACKS, allow_pickle=False) as arrays:
                    atomic_npz(
                        track_path,
                        tracks=arrays["tracks"].astype(np.float32),
                        visibility=arrays["visibility"].astype(bool),
                        query_points=query_points,
                        video_id=np.asarray(video_id),
                        video_path=np.asarray(str(video_path)),
                        video_sha256=np.asarray(digest),
                        tracker=np.asarray("CoTracker3 offline scaled checkpoint"),
                        query_frame=np.int32(0),
                    )
                print(f"[{index:02d}/{len(videos):02d}] import frozen baseline", flush=True)
            else:
                if model is None:
                    model = load_cotracker(args.device)
                frames, _fps = load_video_frames(video_path)
                tracks, visibility = run_cotracker(
                    model, frames, query_points.copy(), args.device
                )
                if tracks.shape != (FRAME_COUNT, 16, 2) or not np.isfinite(tracks).all():
                    raise RuntimeError(f"invalid tracks for {video_id}: {tracks.shape}")
                atomic_npz(
                    track_path,
                    tracks=tracks.astype(np.float32),
                    visibility=visibility.astype(bool),
                    query_points=query_points,
                    video_id=np.asarray(video_id),
                    video_path=np.asarray(str(video_path)),
                    video_sha256=np.asarray(digest),
                    tracker=np.asarray("CoTracker3 offline scaled checkpoint"),
                    query_frame=np.int32(0),
                )
                del frames, tracks, visibility
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[{index:02d}/{len(videos):02d}] tracked {video_id}", flush=True)
            with np.load(track_path, allow_pickle=False) as arrays:
                visibility_rate = float(arrays["visibility"].mean())
            records.append(
                {
                    "id": video_id,
                    "track_file": str(track_path.relative_to(OUTPUT_ROOT)),
                    "visibility_rate": round(visibility_rate, 6),
                    "video_sha256": digest,
                }
            )
    finally:
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tracker": "CoTracker3 offline scaled checkpoint",
            "query_frame": 0,
            "frame_count": FRAME_COUNT,
            "video_count": len(records),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
