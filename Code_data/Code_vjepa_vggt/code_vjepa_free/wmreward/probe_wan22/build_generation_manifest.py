#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import (
    DEFAULT_PIPELINE_ROOT,
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
        description=(
            "Build a probe-ready paired manifest from generated outputs by grouping rows with the same basename."
        )
    )
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--subset-name", default="generated_probe_pairs")
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--drop-ties", action="store_true")
    return parser.parse_args()


def sort_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            safe_float(row["surprise_score"]),
            row["model_key"],
            row["output_video_path"],
        ),
    )


def make_pair_rows(
    *,
    basename: str,
    ranked_rows: list[dict[str, str]],
    pair_rank: int,
    subset_name: str,
) -> list[dict[str, Any]]:
    low_row = ranked_rows[0]
    high_row = ranked_rows[-1]
    low_score = safe_float(low_row["surprise_score"])
    high_score = safe_float(high_row["surprise_score"])
    if low_score is None or high_score is None:
        raise ValueError(f"Missing surprise score in ranked rows for basename={basename}")

    group_gap = high_score - low_score
    pair_id = f"{subset_name}_pair_{pair_rank:04d}"
    output_rows: list[dict[str, Any]] = []
    for role, row in (("low", low_row), ("high", high_row)):
        output_rows.append(
            {
                "subset": subset_name,
                "pair_id": pair_id,
                "group_rank": pair_rank,
                "role": role,
                "basename": basename,
                "group_size": len(ranked_rows),
                "group_gap": f"{group_gap:.8f}",
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
    return output_rows


def main() -> None:
    args = parse_args()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    input_csv = args.input_csv.expanduser().resolve() if args.input_csv else pipeline_root / "manifests" / "generation_registry_all.csv"
    rows = read_csv_rows(input_csv)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        score = safe_float(row.get("surprise_score"))
        if row.get("wmreward_status") != "ok" or score is None:
            continue
        grouped[row["basename"]].append(row)

    pair_candidates: list[tuple[float, str, list[dict[str, str]]]] = []
    skipped_groups: list[dict[str, Any]] = []
    for basename, items in sorted(grouped.items()):
        if len(items) < args.min_group_size:
            skipped_groups.append(
                {
                    "basename": basename,
                    "reason": f"group_size_lt_{args.min_group_size}",
                    "group_size": len(items),
                    "available_models": [item["model_key"] for item in items],
                }
            )
            continue
        ranked_rows = sort_candidates(items)
        low_score = safe_float(ranked_rows[0]["surprise_score"])
        high_score = safe_float(ranked_rows[-1]["surprise_score"])
        if low_score is None or high_score is None:
            continue
        if args.drop_ties and low_score == high_score:
            skipped_groups.append(
                {
                    "basename": basename,
                    "reason": "tied_scores",
                    "group_size": len(items),
                    "available_models": [item["model_key"] for item in ranked_rows],
                }
            )
            continue
        pair_candidates.append((high_score - low_score, basename, ranked_rows))

    pair_candidates.sort(key=lambda item: (-item[0], item[1]))
    pair_rows: list[dict[str, Any]] = []
    pair_summary_rows: list[dict[str, Any]] = []
    for idx, (gap, basename, ranked_rows) in enumerate(pair_candidates, start=1):
        pair_rows.extend(
            make_pair_rows(
                basename=basename,
                ranked_rows=ranked_rows,
                pair_rank=idx,
                subset_name=args.subset_name,
            )
        )
        pair_summary_rows.append(
            {
                "pair_id": f"{args.subset_name}_pair_{idx:04d}",
                "group_rank": idx,
                "basename": basename,
                "group_size": len(ranked_rows),
                "group_gap": f"{gap:.8f}",
                "ordered_models_low_to_high": [row["model_key"] for row in ranked_rows],
                "ordered_scores_low_to_high": [row["surprise_score"] for row in ranked_rows],
            }
        )

    manifest_csv = pipeline_root / "manifests" / f"{args.subset_name}.csv"
    manifest_jsonl = pipeline_root / "manifests" / f"{args.subset_name}.jsonl"
    summary_json = pipeline_root / "manifests" / f"{args.subset_name}_summary.json"
    write_csv_rows(manifest_csv, pair_rows, PAIR_MANIFEST_FIELDS)
    write_jsonl(manifest_jsonl, pair_rows)
    write_json(
        summary_json,
        {
            "subset_name": args.subset_name,
            "input_csv": str(input_csv),
            "pipeline_root": str(pipeline_root),
            "pair_count": len(pair_summary_rows),
            "sample_count": len(pair_rows),
            "min_group_size": args.min_group_size,
            "drop_ties": bool(args.drop_ties),
            "pair_rule": {
                "group_by": "basename",
                "eligible_rows": "wmreward_status == ok and surprise_score is not empty",
                "ranking": "ascending surprise_score within each basename group",
                "low_role": "lowest surprise_score row",
                "high_role": "highest surprise_score row",
                "ties": "kept by default using stable order (model_key, output_video_path); dropped only with --drop-ties",
                "group_gap": "high_surprise_score - low_surprise_score",
                "pair_order": "descending group_gap, then basename ascending",
            },
            "pairs": pair_summary_rows,
            "skipped_groups": skipped_groups,
        },
    )
    print(manifest_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
