#!/usr/bin/env python3
"""Extract RAFT and CoTracker features with a distinct SAM2 region cache per case."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from extract_stc_motion_features import (
    atomic_save_npz,
    extract_cotracker,
    extract_raft,
    load_cotracker,
    load_raft,
    load_region_queries,
    read_video,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_OUTPUT_ROOT / "inventory.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--flow-store-height", type=int, default=64)
    parser.add_argument("--flow-store-width", type=int, default=112)
    parser.add_argument("--raft-batch-size", type=int, default=4)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--cotracker-grid-size", type=int, default=20)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def feature_is_current(
    metadata_path: Path,
    entry: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if args.overwrite or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    settings = metadata.get("settings", {})
    return (
        metadata.get("status") == "complete"
        and metadata.get("source", {}).get("cache_key") == entry["source"]["cache_key"]
        and settings.get("region_cache_key") == entry["region_cache"]["cache_key"]
        and settings.get("num_frames") == args.num_frames
        and settings.get("context_frames") == args.context_frames
        and settings.get("height") == args.height
        and settings.get("width") == args.width
        and settings.get("query_frame") == args.context_frames - 1
        and settings.get("raft_iters") == args.raft_iters
        and metadata_path.with_name("features.npz").is_file()
    )


def main() -> None:
    args = parse_args()
    if args.context_frames < 1 or args.context_frames >= args.num_frames:
        raise ValueError("context_frames must be in [1, num_frames)")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    entries = [
        entry
        for index, entry in enumerate(inventory["entries"])
        if index % args.num_shards == args.shard_id
    ]
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise RuntimeError("This shard has no inventory entries")

    raft = load_raft(args.device)
    cotracker = load_cotracker(args.device)
    query_frame = args.context_frames - 1
    region_cache: dict[
        str,
        tuple[Any, Any, int, list[dict[str, Any]]],
    ] = {}
    print(
        f"[extract-s-motion] entries={len(entries)} device={args.device} "
        f"shard={args.shard_id}/{args.num_shards}",
        flush=True,
    )
    completed = 0
    reused = 0
    for index, entry in enumerate(entries, start=1):
        feature_dir = args.output_root / "features" / entry["source"]["cache_key"]
        metadata_path = feature_dir / "metadata.json"
        if feature_is_current(metadata_path, entry, args):
            reused += 1
            print(f"[{index}/{len(entries)}] reuse {entry['entry_id']}", flush=True)
            continue

        case_id = str(entry["case_id"])
        if case_id not in region_cache:
            loaded = load_region_queries(
                Path(entry["region_cache"]["path"]),
                args.height,
                args.width,
            )
            if loaded[2] != query_frame:
                raise ValueError(
                    f"{case_id}: region query frame {loaded[2]} != expected {query_frame}"
                )
            region_cache[case_id] = loaded
        query_points, region_ids, region_source_frame, track_regions = region_cache[case_id]

        started = time.time()
        frames, fps, source_frame_count = read_video(
            Path(entry["source"]["path"]),
            args.num_frames,
            args.height,
            args.width,
        )
        arrays = extract_raft(
            raft,
            frames,
            args.device,
            args.raft_batch_size,
            args.raft_iters,
            args.flow_store_height,
            args.flow_store_width,
        )
        arrays.update(
            extract_cotracker(
                cotracker,
                frames,
                args.device,
                args.cotracker_grid_size,
                query_frame,
                query_points,
                region_ids,
            )
        )
        atomic_save_npz(feature_dir / "features.npz", **arrays)
        metadata = {
            "schema_version": 3,
            "status": "complete",
            "entry": entry,
            "source": entry["source"],
            "fps": fps,
            "source_frame_count": source_frame_count,
            "elapsed_seconds": time.time() - started,
            "settings": {
                "num_frames": args.num_frames,
                "context_frames": args.context_frames,
                "query_frame": query_frame,
                "height": args.height,
                "width": args.width,
                "flow_store_height": args.flow_store_height,
                "flow_store_width": args.flow_store_width,
                "raft_iters": args.raft_iters,
                "raft_batch_size": args.raft_batch_size,
                "cotracker_grid_size": args.cotracker_grid_size,
                "track_query_mode": "per_case_sam2_amg_regions",
                "track_regions": track_regions,
                "region_source_frame": region_source_frame,
                "region_cache_path": entry["region_cache"]["path"],
                "region_cache_key": entry["region_cache"]["cache_key"],
                "spatial_preprocess": "center_crop_to_7:4_then_resize",
                "flow_units": "normalized_image_width_and_height_per_frame",
                "track_units": "normalized_xy",
            },
        }
        atomic_write_json(metadata_path, metadata)
        completed += 1
        print(
            f"[{index}/{len(entries)}] complete {entry['entry_id']} "
            f"{metadata['elapsed_seconds']:.1f}s",
            flush=True,
        )
    print(f"[extract-s-motion] complete={completed} reused={reused}", flush=True)


if __name__ == "__main__":
    main()
