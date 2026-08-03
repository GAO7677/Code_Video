#!/usr/bin/env python3
"""Join all-step object tracking results with the validated head-role classes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
DEFAULT_CLASSES = Path(
    "/data/gaoya/agent-data/outputs/head_classification_csv/common22_public_stable/"
    "head_classification_all_720.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/head_classification_csv/allsteps_objects"
)
ROLES = ("S", "T", "P", "C", "G", "M")
CLASS_FIELDS = (
    "head_id",
    "depth",
    "final_class",
    "s_subtype",
    "model_role_signature",
    "raw_score_winner_signature",
    "raw_score_consensus_candidate",
    "in_training_s_same_full59",
    "in_training_t_common_full70",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_classes(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {(int(row["block"]), int(row["head"])): row for row in rows}
    if len(rows) != 720 or len(result) != 720:
        raise ValueError(f"Expected 720 unique classified heads, found {len(result)}")
    return result


def load_objects(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row["scope"] == "objects"]
    if len(rows) != 28800:
        raise ValueError(f"Expected 28,800 object rows, found {len(rows)}")
    positions = Counter((int(row["block"]), int(row["head"])) for row in rows)
    if len(positions) != 720 or set(positions.values()) != {40}:
        raise ValueError("Each of the 720 heads must have exactly 40 object-step rows")
    ranks = sorted(int(row["rank_macro_pck32"]) for row in rows)
    if ranks != list(range(1, 28801)):
        raise ValueError("Object combination ranks must contain every integer from 1 to 28,800")
    return source_fields, rows


def mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def pstdev(rows: list[dict[str, str]], field: str) -> float:
    return statistics.pstdev(float(row[field]) for row in rows)


def main() -> None:
    args = parse_args()
    classes = load_classes(args.classes)
    source_fields, object_rows = load_objects(args.source)

    merged_rows: list[dict[str, Any]] = []
    for source_row in object_rows:
        key = (int(source_row["block"]), int(source_row["head"]))
        classified = classes[key]
        merged_rows.append(
            {
                **source_row,
                **{field: classified[field] for field in CLASS_FIELDS},
            }
        )
    merged_rows.sort(key=lambda row: int(row["rank_macro_pck32"]))
    write_csv(
        args.output_dir / "object_step_head_rankings_with_classes.csv",
        merged_rows,
        [*source_fields, *CLASS_FIELDS],
    )

    population = Counter(row["final_class"] for row in merged_rows)
    combination_enrichment: list[dict[str, Any]] = []
    for top_k in (10, 20, 50, 100, 500, 1000, 5000, 28800):
        selected = merged_rows[:top_k]
        counts = Counter(row["final_class"] for row in selected)
        unique_by_role: dict[str, set[str]] = defaultdict(set)
        for row in selected:
            unique_by_role[row["final_class"]].add(row["head_id"])
        for role in ROLES:
            population_share = population[role] / len(merged_rows)
            combination_enrichment.append(
                {
                    "top_k_step_head_combinations": top_k,
                    "final_class": role,
                    "combination_count": counts[role],
                    "combination_share_percent": 100 * counts[role] / top_k,
                    "unique_head_count": len(unique_by_role[role]),
                    "population_share_percent": 100 * population_share,
                    "expected_count_if_random": top_k * population_share,
                    "enrichment_fold": (counts[role] / top_k) / population_share,
                }
            )
    write_csv(
        args.output_dir / "object_step_head_class_enrichment.csv",
        combination_enrichment,
        [
            "top_k_step_head_combinations",
            "final_class",
            "combination_count",
            "combination_share_percent",
            "unique_head_count",
            "population_share_percent",
            "expected_count_if_random",
            "enrichment_fold",
        ],
    )

    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in object_rows:
        grouped[(int(row["block"]), int(row["head"]))].append(row)

    head_rows: list[dict[str, Any]] = []
    for (block, head), rows in grouped.items():
        rows_by_metric = sorted(
            rows,
            key=lambda row: (
                -float(row["macro_pck32"]),
                float(row["macro_mean_error_px"]),
                -float(row["worst_model_macro_pck32"]),
            ),
        )
        best = rows_by_metric[0]
        source_ranks = [int(row["rank_macro_pck32"]) for row in rows]
        top5 = rows_by_metric[:5]
        classified = classes[(block, head)]
        head_rows.append(
            {
                "block": block,
                "head": head,
                **{field: classified[field] for field in CLASS_FIELDS},
                "num_steps": len(rows),
                "mean_macro_pck8": mean(rows, "macro_pck8"),
                "mean_macro_pck16": mean(rows, "macro_pck16"),
                "mean_macro_pck32": mean(rows, "macro_pck32"),
                "std_across_steps_macro_pck32": pstdev(rows, "macro_pck32"),
                "min_step_macro_pck32": min(float(row["macro_pck32"]) for row in rows),
                "max_step_macro_pck32": max(float(row["macro_pck32"]) for row in rows),
                "top5_step_mean_macro_pck32": statistics.fmean(
                    float(row["macro_pck32"]) for row in top5
                ),
                "mean_macro_error_px": mean(rows, "macro_mean_error_px"),
                "mean_worst_model_macro_pck32": mean(rows, "worst_model_macro_pck32"),
                "min_worst_model_macro_pck32": min(
                    float(row["worst_model_macro_pck32"]) for row in rows
                ),
                "mean_std_model_macro_pck32": mean(rows, "std_model_macro_pck32"),
                "best_step": int(best["step"]),
                "best_step_timestep": float(best["timestep"]),
                "best_step_sigma": float(best["sigma"]),
                "best_step_macro_pck32": float(best["macro_pck32"]),
                "best_combination_rank": min(source_ranks),
                "median_combination_rank": statistics.median(source_ranks),
                "mean_combination_rank": statistics.fmean(source_ranks),
                "count_combination_top100": sum(rank <= 100 for rank in source_ranks),
                "count_combination_top500": sum(rank <= 500 for rank in source_ranks),
                "count_combination_top1000": sum(rank <= 1000 for rank in source_ranks),
                "count_combination_top5000": sum(rank <= 5000 for rank in source_ranks),
            }
        )

    head_rows.sort(
        key=lambda row: (
            -row["mean_macro_pck32"],
            row["mean_macro_error_px"],
            -row["mean_worst_model_macro_pck32"],
            row["block"],
            row["head"],
        )
    )
    for rank, row in enumerate(head_rows, start=1):
        row["allsteps_head_rank"] = rank

    head_fields = [
        "allsteps_head_rank",
        "block",
        "head",
        *CLASS_FIELDS,
        "num_steps",
        "mean_macro_pck8",
        "mean_macro_pck16",
        "mean_macro_pck32",
        "std_across_steps_macro_pck32",
        "min_step_macro_pck32",
        "max_step_macro_pck32",
        "top5_step_mean_macro_pck32",
        "mean_macro_error_px",
        "mean_worst_model_macro_pck32",
        "min_worst_model_macro_pck32",
        "mean_std_model_macro_pck32",
        "best_step",
        "best_step_timestep",
        "best_step_sigma",
        "best_step_macro_pck32",
        "best_combination_rank",
        "median_combination_rank",
        "mean_combination_rank",
        "count_combination_top100",
        "count_combination_top500",
        "count_combination_top1000",
        "count_combination_top5000",
    ]
    write_csv(args.output_dir / "object_head_allsteps_statistics.csv", head_rows, head_fields)

    head_enrichment: list[dict[str, Any]] = []
    head_population = Counter(row["final_class"] for row in head_rows)
    for top_k in (10, 20, 50, 100, 200, 360, 720):
        counts = Counter(row["final_class"] for row in head_rows[:top_k])
        for role in ROLES:
            population_share = head_population[role] / 720
            head_enrichment.append(
                {
                    "top_k_heads": top_k,
                    "final_class": role,
                    "count": counts[role],
                    "share_percent": 100 * counts[role] / top_k,
                    "population_count": head_population[role],
                    "population_share_percent": 100 * population_share,
                    "expected_count_if_random": top_k * population_share,
                    "enrichment_fold": (counts[role] / top_k) / population_share,
                }
            )
    write_csv(
        args.output_dir / "object_head_class_enrichment.csv",
        head_enrichment,
        [
            "top_k_heads",
            "final_class",
            "count",
            "share_percent",
            "population_count",
            "population_share_percent",
            "expected_count_if_random",
            "enrichment_fold",
        ],
    )

    class_statistics: list[dict[str, Any]] = []
    for role in ROLES:
        selected = [row for row in head_rows if row["final_class"] == role]
        ranks = [row["allsteps_head_rank"] for row in selected]
        class_statistics.append(
            {
                "final_class": role,
                "head_count": len(selected),
                "best_head_rank": min(ranks),
                "median_head_rank": statistics.median(ranks),
                "mean_head_rank": statistics.fmean(ranks),
                "count_top20": sum(rank <= 20 for rank in ranks),
                "count_top50": sum(rank <= 50 for rank in ranks),
                "count_top100": sum(rank <= 100 for rank in ranks),
                "class_mean_head_mean_macro_pck32": statistics.fmean(
                    row["mean_macro_pck32"] for row in selected
                ),
                "class_mean_head_top5_step_pck32": statistics.fmean(
                    row["top5_step_mean_macro_pck32"] for row in selected
                ),
                "class_mean_head_step_std_pck32": statistics.fmean(
                    row["std_across_steps_macro_pck32"] for row in selected
                ),
                "class_mean_worst_model_pck32": statistics.fmean(
                    row["mean_worst_model_macro_pck32"] for row in selected
                ),
            }
        )
    write_csv(
        args.output_dir / "object_head_class_statistics.csv",
        class_statistics,
        [
            "final_class",
            "head_count",
            "best_head_rank",
            "median_head_rank",
            "mean_head_rank",
            "count_top20",
            "count_top50",
            "count_top100",
            "class_mean_head_mean_macro_pck32",
            "class_mean_head_top5_step_pck32",
            "class_mean_head_step_std_pck32",
            "class_mean_worst_model_pck32",
        ],
    )

    metadata = {
        "source": str(args.source),
        "classification_source": str(args.classes),
        "scope": "objects",
        "step_head_combinations": len(merged_rows),
        "unique_heads": len(head_rows),
        "steps_per_head": 40,
        "source_combination_ranking_rule": (
            "mean macro PCK@32 across GT/LoRA/Baseline descending, mean error ascending, "
            "worst-model macro PCK@32 descending"
        ),
        "allsteps_head_ranking_rule": (
            "40-step mean macro PCK@32 descending, 40-step mean error ascending, "
            "40-step mean worst-model PCK@32 descending"
        ),
        "top20_allsteps_heads": [
            {
                "rank": row["allsteps_head_rank"],
                "head_id": row["head_id"],
                "class": row["final_class"],
                "mean_macro_pck32": row["mean_macro_pck32"],
            }
            for row in head_rows[:20]
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "object_step_head_combinations": len(merged_rows),
                "unique_heads": len(head_rows),
                "steps_per_head": 40,
                "top20_class_counts": dict(
                    Counter(row["final_class"] for row in head_rows[:20])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
