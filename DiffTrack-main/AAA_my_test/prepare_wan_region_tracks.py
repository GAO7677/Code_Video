#!/usr/bin/env python3
"""Prepare compact object/background CoTracker tracks for Wan motion analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from AAA_my_test.wan_motion_utils import (
    COTRACKER_CHECKPOINT,
    COTRACKER_ROOT,
    DATASET_ROOT,
    LATENT_ANCHOR_FRAMES,
    NUM_FRAMES,
    OUTPUT_ROOT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    atomic_write_json,
    build_regions,
    classify_region_tracks,
    enumerate_samples,
    farthest_point_sample,
    find_sample,
    free_space_gib,
    load_manifest,
    read_instance_ids,
    read_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "tracks_base")
    parser.add_argument("--case-key")
    parser.add_argument(
        "--sample-types",
        nargs="+",
        choices=["base", "background_color", "object_color", "object_shape"],
        default=["base"],
    )
    parser.add_argument("--points-per-region", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cotracker-root", type=Path, default=COTRACKER_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=COTRACKER_CHECKPOINT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    return parser.parse_args()


def prepare_sample(model, sample: dict, args: argparse.Namespace) -> dict:
    sample_key = sample["sample_key"]
    output_path = args.output_dir / f"{sample_key}.npz"
    metadata_path = args.output_dir / f"{sample_key}.json"
    if output_path.exists() and metadata_path.exists() and not args.overwrite:
        return {
            "sample_key": sample_key,
            "case_key": sample["case_key"],
            "sample_type": sample["sample_type"],
            "tracks": str(output_path),
            "metadata": str(metadata_path),
            "status": "reused",
        }

    if free_space_gib(args.output_dir) < args.min_free_gib:
        raise RuntimeError(f"Free space below {args.min_free_gib:.1f} GiB; refusing to create a new sample")

    video = read_video(Path(sample["video"]), NUM_FRAMES)
    instance_ids, object_names, object_ids = read_instance_ids(Path(sample["mask_ids"]), NUM_FRAMES)
    regions = build_regions(instance_ids, object_names, object_ids)
    included_object_names = {region["region_name"] for region in regions if region["region_type"] == "object"}
    excluded_regions = [
        {
            "region_name": name,
            "region_type": "object",
            "object_id": object_id,
            "reason": "not_visible_at_query_frame",
        }
        for name, object_id in zip(object_names, object_ids)
        if name not in included_object_names
    ]

    query_parts = []
    region_records = []
    point_offset = 0
    for region in regions:
        points = farthest_point_sample(region["mask"], args.points_per_region)
        query_parts.append(points)
        region_records.append(
            {
                "region_name": region["region_name"],
                "region_type": region["region_type"],
                "object_id": region["object_id"],
                "point_start": point_offset,
                "point_end": point_offset + len(points),
            }
        )
        point_offset += len(points)

    query_points = np.concatenate(query_parts, axis=0).astype(np.float32)
    query_tensor = torch.from_numpy(query_points).to(args.device)
    queries = torch.cat((torch.zeros_like(query_tensor[:, :1]), query_tensor), dim=1).unsqueeze(0)
    with torch.inference_mode():
        tracks, visibility = model(video.unsqueeze(0).to(args.device), queries=queries)
    tracks_np = tracks[0].cpu().numpy().astype(np.float32)
    visibility_np = visibility[0].cpu().numpy().astype(np.bool_)

    for record in region_records:
        point_slice = slice(record["point_start"], record["point_end"])
        record["motion_class"] = classify_region_tracks(
            tracks_np[:, point_slice], visibility_np[:, point_slice], record["region_type"]
        )
        record["visibility_rate"] = float(visibility_np[:, point_slice].mean())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_points=query_points,
            tracks=tracks_np,
            visibility=visibility_np,
            anchor_frames=LATENT_ANCHOR_FRAMES,
            anchor_tracks=tracks_np[LATENT_ANCHOR_FRAMES],
            anchor_visibility=visibility_np[LATENT_ANCHOR_FRAMES],
        )
    temporary.replace(output_path)

    caption = sample.get("caption", "")
    if sample.get("meta") and Path(sample["meta"]).exists():
        import json

        caption = json.loads(Path(sample["meta"]).read_text()).get("caption", caption)
    metadata = {
        "sample_key": sample_key,
        "case_key": sample["case_key"],
        "sample_type": sample["sample_type"],
        "video": str(sample["video"]),
        "mask_ids": str(sample["mask_ids"]),
        "caption": caption,
        "preprocessing": f"aspect-preserving resize and center crop to {TARGET_WIDTH}x{TARGET_HEIGHT}",
        "num_frames": NUM_FRAMES,
        "anchor_frames": LATENT_ANCHOR_FRAMES.tolist(),
        "points_per_region": args.points_per_region,
        "regions": region_records,
        "excluded_regions": excluded_regions,
    }
    atomic_write_json(metadata_path, metadata)
    del video, tracks, visibility, queries, query_tensor
    torch.cuda.empty_cache()
    return {
        "sample_key": sample_key,
        "case_key": sample["case_key"],
        "sample_type": sample["sample_type"],
        "tracks": str(output_path),
        "metadata": str(metadata_path),
        "status": "created",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.dataset_root)
    if args.case_key:
        if len(args.sample_types) != 1:
            raise ValueError("--case-key preparation accepts exactly one sample type")
        samples = [find_sample(manifest, args.case_key, args.sample_types[0])]
    else:
        samples = enumerate_samples(manifest, args.sample_types)[args.start : args.end]
    if not samples:
        raise ValueError("No samples selected")

    sys.path.insert(0, str(args.cotracker_root))
    from cotracker.predictor import CoTrackerPredictor

    model = CoTrackerPredictor(checkpoint=str(args.checkpoint), offline=True).to(args.device).eval()
    records = []
    for index, sample in enumerate(samples, start=1):
        record = prepare_sample(model, sample, args)
        records.append(record)
        print(f"[{index}/{len(samples)}] {record['sample_key']}: {record['status']}", flush=True)

    atomic_write_json(
        args.output_dir / f"manifest_{args.start}_{args.end if args.end is not None else 'end'}.json",
        {
            "dataset_root": str(args.dataset_root),
            "height": TARGET_HEIGHT,
            "width": TARGET_WIDTH,
            "num_frames": NUM_FRAMES,
            "cotracker_checkpoint": str(args.checkpoint),
            "samples": records,
        },
    )


if __name__ == "__main__":
    main()
