#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate probe results across extraction presets into a single leaderboard."
    )
    parser.add_argument(
        "--results_root",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/probe_results/wan21_t2v_1p3b_final",
    )
    parser.add_argument("--top_k", type=int, default=30)
    return parser.parse_args()


def safe_float(value: str | None, default: float) -> float:
    if value in (None, "", "nan"):
        return default
    return float(value)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    leaderboard_rows: list[dict[str, object]] = []

    for preset_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        grid_csv = preset_dir / "grid_search" / "probe_grid_metrics.csv"
        ridge_csv = preset_dir / "ridge_single_features_mean" / "probe_metrics.csv"

        if grid_csv.is_file():
            rows = list(csv.DictReader(grid_csv.open(newline="", encoding="utf-8")))
            for row in rows:
                if row.get("status") != "ok":
                    continue
                leaderboard_rows.append(
                    {
                        "preset": preset_dir.name,
                        "source": "grid_search",
                        "step_idx": row["step_idx"],
                        "layer_idx": row["layer_idx"],
                        "frame_reduce": row.get("frame_reduce", ""),
                        "feature_spec": row.get("feature_group", ""),
                        "pearson": safe_float(row.get("pearson"), float("-inf")),
                        "spearman": safe_float(row.get("spearman"), float("-inf")),
                        "r2": safe_float(row.get("r2"), float("-inf")),
                        "mae": safe_float(row.get("mae"), float("inf")),
                        "n_samples": int(row["n_samples"]),
                        "n_groups": int(row["n_groups"]),
                    }
                )

        if ridge_csv.is_file():
            rows = list(csv.DictReader(ridge_csv.open(newline="", encoding="utf-8")))
            for row in rows:
                if row.get("status") != "ok":
                    continue
                leaderboard_rows.append(
                    {
                        "preset": preset_dir.name,
                        "source": "ridge_single_features_mean",
                        "step_idx": row["step_idx"],
                        "layer_idx": row["layer_idx"],
                        "frame_reduce": "mean",
                        "feature_spec": row.get("feature_key", ""),
                        "pearson": safe_float(row.get("pearson"), float("-inf")),
                        "spearman": safe_float(row.get("spearman"), float("-inf")),
                        "r2": safe_float(row.get("r2"), float("-inf")),
                        "mae": safe_float(row.get("mae"), float("inf")),
                        "n_samples": int(row["n_samples"]),
                        "n_groups": int(row["n_groups"]),
                    }
                )

    leaderboard_rows.sort(
        key=lambda row: (
            -row["pearson"],
            -row["spearman"],
            -row["r2"],
            row["mae"],
            row["preset"],
            row["source"],
            row["step_idx"],
            row["layer_idx"],
            row["feature_spec"],
        )
    )

    output_csv = results_root / "suite_leaderboard.csv"
    fieldnames = [
        "preset",
        "source",
        "step_idx",
        "layer_idx",
        "frame_reduce",
        "feature_spec",
        "pearson",
        "spearman",
        "r2",
        "mae",
        "n_samples",
        "n_groups",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(leaderboard_rows)

    output_json = results_root / "suite_leaderboard_summary.json"
    output_json.write_text(
        json.dumps(
            {
                "results_root": str(results_root),
                "num_rows": len(leaderboard_rows),
                "top_k": args.top_k,
                "top_results": leaderboard_rows[: args.top_k],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_csv)
    print(output_json)


if __name__ == "__main__":
    main()
