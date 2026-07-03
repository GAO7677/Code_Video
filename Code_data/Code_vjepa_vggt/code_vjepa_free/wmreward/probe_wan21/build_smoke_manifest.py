#!/usr/bin/env python3
"""Build a smoke-test probe manifest by pairing samples across surprise scores.

Unlike build_generation_manifest.py which pairs different models on the same input,
this script pairs different input samples by ranking all available samples by
surprise_score and assigning low/high roles to the bottom/top halves.
Used when only one model is available (single-model smoke test).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline_common import (
    DEFAULT_SMOKE_PIPELINE_ROOT,
    read_csv_rows,
    safe_float,
    write_csv_rows,
    write_json,
    write_jsonl,
)


PAIR_MANIFEST_FIELDS = [
    "subset",
    "pair_id",
    "group_rank",
    "role",
    "basename",
    "group_size",
    "group_gap",
    "json_path",
    "video_path",
    "relative_path",
    "surprise_score",
    "similarity_score",
    "source_tag",
    "model_key",
    "model_name",
    "input_json_path",
    "input_caption",
    "input_video_path",
    "input_image_path",
    "gt_video_path",
    "output_json_exists",
    "output_video_exists",
    "wmreward_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build smoke probe manifest by pairing samples across surprise scores."
    )
    parser.add_argument("--smoke-name", default="wan21_smoke_test")
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=None,
        help="Override pipeline root (default: DEFAULT_SMOKE_PIPELINE_ROOT/<smoke-name>)",
    )
    parser.add_argument("--subset-name", default="smoke_probe_pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pipeline_root is not None:
        pipeline_root = args.pipeline_root.expanduser().resolve()
    else:
        pipeline_root = (DEFAULT_SMOKE_PIPELINE_ROOT / args.smoke_name).resolve()

    registry_csv = pipeline_root / "manifests" / "generation_registry_all.csv"
    rows = read_csv_rows(registry_csv)

    # Keep only rows with valid WMReward scores
    eligible = [
        row for row in rows
        if row.get("wmreward_status") == "ok" and safe_float(row.get("surprise_score")) is not None
    ]
    if len(eligible) < 2:
        raise ValueError(f"Need at least 2 scored samples to build pairs, found {len(eligible)}")

    # Sort ascending by surprise_score
    eligible.sort(key=lambda r: (safe_float(r["surprise_score"]), r["basename"]))

    # Pair bottom half (low) with top half (high) by index
    n_pairs = len(eligible) // 2
    low_rows = eligible[:n_pairs]
    high_rows = list(reversed(eligible[len(eligible) - n_pairs:]))

    pair_rows: list[dict[str, Any]] = []
    for idx, (low, high) in enumerate(zip(low_rows, high_rows), start=1):
        low_score = safe_float(low["surprise_score"])
        high_score = safe_float(high["surprise_score"])
        assert low_score is not None and high_score is not None
        gap = high_score - low_score
        pair_id = f"{args.subset_name}_pair_{idx:04d}"
        for role, row in (("low", low), ("high", high)):
            pair_rows.append(
                {
                    "subset": args.subset_name,
                    "pair_id": pair_id,
                    "group_rank": idx,
                    "role": role,
                    "basename": row["basename"],
                    "group_size": 2,
                    "group_gap": f"{gap:.8f}",
                    "json_path": row["output_json_path"],
                    "video_path": row["output_video_path"],
                    "relative_path": row["relative_path"],
                    "surprise_score": row["surprise_score"],
                    "similarity_score": row["similarity_score"],
                    "source_tag": row["model_key"],
                    "model_key": row["model_key"],
                    "model_name": row["model_name"],
                    "input_json_path": row["input_json_path"],
                    "input_caption": row["input_caption"],
                    "input_video_path": row["input_video_path"],
                    "input_image_path": row["input_image_path"],
                    "gt_video_path": row["gt_video_path"],
                    "output_json_exists": row["output_json_exists"],
                    "output_video_exists": row["output_video_exists"],
                    "wmreward_status": row["wmreward_status"],
                }
            )

    manifest_dir = pipeline_root / "manifests"
    manifest_csv = manifest_dir / f"{args.subset_name}.csv"
    manifest_jsonl = manifest_dir / f"{args.subset_name}.jsonl"
    summary_json = manifest_dir / f"{args.subset_name}_summary.json"

    write_csv_rows(manifest_csv, pair_rows, PAIR_MANIFEST_FIELDS)
    write_jsonl(manifest_jsonl, pair_rows)
    write_json(
        summary_json,
        {
            "subset_name": args.subset_name,
            "pipeline_root": str(pipeline_root),
            "pair_count": n_pairs,
            "sample_count": len(pair_rows),
            "pairing_strategy": "cross_sample_low_high",
            "total_eligible": len(eligible),
        },
    )
    print(f"Wrote {n_pairs} pairs ({len(pair_rows)} rows) to {manifest_csv}")
    print(manifest_csv)


if __name__ == "__main__":
    main()
