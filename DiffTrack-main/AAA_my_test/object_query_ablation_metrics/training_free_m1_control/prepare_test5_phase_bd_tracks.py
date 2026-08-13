#!/usr/bin/env python3
"""Prepare all frozen baseline object tracks with one CoTracker model load."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (
    load_cotracker,
    object_queries,
    run_cotracker,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    atomic_npz,
    tracks_root,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ready_path(tracks_base: Path, sample: dict) -> Path | None:
    output = tracks_root(
        tracks_base, str(sample["case"]), int(sample["seed"])
    )
    track_path = output / "tracks.npz"
    if track_path.is_file() and (output / "manifest.json").is_file():
        return track_path
    return None


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest.get("samples") or [])
    if not samples:
        raise RuntimeError("manifest contains no samples")
    missing = [sample for sample in samples if args.overwrite or ready_path(args.tracks_root, sample) is None]
    if not missing:
        print(f"all {len(samples)} frozen track files already exist", flush=True)
        return

    model = load_cotracker(str(args.device))
    completed = 0
    try:
        for index, sample in enumerate(missing, start=1):
            case, seed = str(sample["case"]), int(sample["seed"])
            output = tracks_root(args.tracks_root, case, seed)
            output.mkdir(parents=True, exist_ok=True)
            baseline = Path(str(sample["baseline_video"]))
            if not baseline.is_file():
                raise FileNotFoundError(f"{case}/seed_{seed:05d}: missing baseline {baseline}")
            cache_dir = Path(str(sample["query_cache_dir"]))
            cache = load_region_cache(cache_dir.parent, cache_dir.name)
            if int(cache.metadata.get("query_context_frame", -1)) != 0:
                raise RuntimeError(f"{case}: expected query frame zero")
            points, query_regions = object_queries(cache)
            frames = np.asarray(iio.imread(baseline))
            if frames.ndim != 4 or frames.shape[-1] < 3:
                raise RuntimeError(f"{case}: unexpected baseline shape {frames.shape}")
            frames = frames[..., :3].astype(np.uint8, copy=False)
            print(
                f"[{index}/{len(missing)}] CoTracker {case}/seed_{seed:05d}",
                flush=True,
            )
            tracks, visibility = run_cotracker(model, frames, points, str(args.device))
            anchors = np.arange(13, dtype=np.int64) * 4
            if int(anchors[-1]) >= len(frames):
                anchors = np.rint(np.linspace(0, len(frames) - 1, 13)).astype(np.int64)
            if not np.isfinite(tracks[anchors]).all():
                raise RuntimeError(f"{case}/seed_{seed:05d}: non-finite anchor tracks")
            starts = np.asarray([part.start for _, part in query_regions], dtype=np.int32)
            ends = np.asarray([part.stop for _, part in query_regions], dtype=np.int32)
            names = np.asarray([region.region_name for region, _ in query_regions])
            atomic_npz(
                output / "tracks.npz",
                tracks=tracks.astype(np.float32),
                visibility=visibility.astype(np.bool_),
                anchor_pixel_frames=anchors,
                query_points=points.astype(np.float32),
                region_names=names,
                point_starts=starts,
                point_ends=ends,
                source_video=np.asarray(str(baseline)),
                source_json=np.asarray(str(sample["input_json"])),
                pixel_height=np.int32(frames.shape[1]),
                pixel_width=np.int32(frames.shape[2]),
                seed=np.int32(seed),
            )
            atomic_json(
                output / "manifest.json",
                {
                    "case": case,
                    "seed": seed,
                    "source_video": str(baseline),
                    "source_json": str(sample["input_json"]),
                    "query_cache_dir": str(cache_dir),
                    "tracker": "CoTracker3 offline scaled checkpoint",
                    "query_pixel_frame": 0,
                    "latent_anchor_pixel_frames": anchors.tolist(),
                    "point_count": int(len(points)),
                    "regions": [
                        {
                            "region_name": str(name),
                            "point_start": int(start),
                            "point_end": int(end),
                            "anchor_visibility_rate": float(
                                visibility[anchors, start:end].mean()
                            ),
                        }
                        for name, start, end in zip(
                            names.tolist(), starts.tolist(), ends.tolist()
                        )
                    ],
                    "selection_policy": (
                        "use every finite CoTracker coordinate at all 13 latent anchors; "
                        "visibility is retained for audit"
                    ),
                    "frozen_before_intervention": True,
                },
            )
            completed += 1
            del frames, tracks, visibility
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    atomic_json(
        args.tracks_root / "test5_tracks_complete.json",
        {
            "manifest": str(args.manifest_path),
            "sample_count": len(samples),
            "newly_completed": completed,
        },
    )
    print(f"frozen tracks ready: {len(samples)}/{len(samples)}", flush=True)


if __name__ == "__main__":
    main()
