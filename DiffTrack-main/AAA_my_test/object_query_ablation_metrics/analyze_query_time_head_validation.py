#!/usr/bin/env python3
"""Aggregate all-latent query validation and apply the frozen decision rule."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BASE = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage1_query_time_validation"
)
DEFAULT_RUNS = BASE / "runs"
DEFAULT_THRESHOLDS = BASE / "frozen_thresholds.json"
DEFAULT_OUTPUT = BASE / "analysis"
DEFAULT_HEAD_SCOPES = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/pck_head_scopes_s039_latest3350.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--head-scopes", type=Path, default=DEFAULT_HEAD_SCOPES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-runs", type=int, default=15)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def head_pairs(payload: dict[str, Any], scope: str) -> list[tuple[int, int]]:
    definition = payload["head_scopes"][scope]
    start, end = int(definition["rank_start"]) - 1, int(definition["rank_end"])
    return [
        (int(row["block"]), int(row["head"])) for row in payload["entries"][start:end]
    ]


def rank_heads(
    correct: np.ndarray, comparisons: np.ndarray, error_sum: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        pck = np.where(comparisons > 0, 100.0 * correct / comparisons, np.nan)
        error = np.where(comparisons > 0, error_sum / comparisons, np.nan)
    rows = [
        {
            "block": block,
            "head": head,
            "pck32": None if not np.isfinite(pck[block, head]) else float(pck[block, head]),
            "mean_error_px": (
                None if not np.isfinite(error[block, head]) else float(error[block, head])
            ),
            "comparisons": int(comparisons[block, head]),
        }
        for block in range(30)
        for head in range(24)
    ]
    rows.sort(
        key=lambda row: (
            row["pck32"] is None,
            -(row["pck32"] if row["pck32"] is not None else -1.0),
            row["mean_error_px"]
            if row["mean_error_px"] is not None
            else float("inf"),
            row["block"],
            row["head"],
        )
    )
    ranks = np.empty(720, dtype=np.float64)
    for rank, row in enumerate(rows):
        ranks[int(row["block"]) * 24 + int(row["head"])] = rank
    return rows, ranks


def group_pck(
    correct: np.ndarray, comparisons: np.ndarray, pairs: list[tuple[int, int]]
) -> float:
    numerator = sum(int(correct[block, head]) for block, head in pairs)
    denominator = sum(int(comparisons[block, head]) for block, head in pairs)
    return float(100.0 * numerator / denominator) if denominator else float("nan")


def main() -> None:
    args = parse_args()
    complete_paths = sorted(args.runs_root.glob("*/seed_*/complete.json"))
    if len(complete_paths) != args.require_runs:
        raise RuntimeError(
            f"expected {args.require_runs} complete runs, found {len(complete_paths)}"
        )
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    scopes = json.loads(args.head_scopes.read_text(encoding="utf-8"))
    fixed_top = head_pairs(scopes, "top100")
    fixed_bottom = head_pairs(scopes, "bottom100")

    runs: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for complete_path in complete_paths:
        root = complete_path.parent
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        with np.load(root / "metrics.npz", allow_pickle=False) as arrays:
            row = {
                "case": str(manifest["case"]),
                "seed": int(manifest["seed"]),
                "correct": arrays["correct32"].astype(np.int64),
                "comparisons": arrays["comparisons"].astype(np.int64),
                "error_sum": arrays["error_sum"].astype(np.float64),
            }
        runs.append(row)
        by_case[row["case"]].append(row)

    correct = sum((row["correct"] for row in runs), np.zeros((13, 30, 24), dtype=np.int64))
    comparisons = sum(
        (row["comparisons"] for row in runs), np.zeros((13, 30, 24), dtype=np.int64)
    )
    error_sum = sum(
        (row["error_sum"] for row in runs), np.zeros((13, 30, 24), dtype=np.float64)
    )

    rankings, rank_vectors, top_sets = [], [], []
    per_anchor = []
    for query_time in range(13):
        ranking, rank_vector = rank_heads(
            correct[query_time], comparisons[query_time], error_sum[query_time]
        )
        rankings.append(ranking)
        rank_vectors.append(rank_vector)
        top_sets.append({(row["block"], row["head"]) for row in ranking[:100]})
        top_pck = group_pck(correct[query_time], comparisons[query_time], fixed_top)
        bottom_pck = group_pck(correct[query_time], comparisons[query_time], fixed_bottom)
        per_anchor.append(
            {
                "query_time": query_time,
                "pixel_frame": query_time * 4,
                "fixed_top100_pck32": top_pck,
                "fixed_bottom100_pck32": bottom_pck,
                "top_minus_bottom_pck32": top_pck - bottom_pck,
                "query_specific_top100": ranking[:100],
            }
        )

    pairwise = []
    for first in range(13):
        for second in range(first + 1, 13):
            intersection = len(top_sets[first] & top_sets[second])
            union = len(top_sets[first] | top_sets[second])
            pairwise.append(
                {
                    "first_query_time": first,
                    "second_query_time": second,
                    "top100_intersection": intersection,
                    "top100_jaccard": intersection / union,
                    "spearman": float(np.corrcoef(rank_vectors[first], rank_vectors[second])[0, 1]),
                }
            )
    median_jaccard = float(np.median([row["top100_jaccard"] for row in pairwise]))
    median_spearman = float(np.median([row["spearman"] for row in pairwise]))
    anchor_fraction = float(
        np.mean([row["top_minus_bottom_pck32"] > 0 for row in per_anchor])
    )

    case_effects = []
    for case, case_runs in sorted(by_case.items()):
        c_correct = sum(
            (row["correct"] for row in case_runs), np.zeros((13, 30, 24), dtype=np.int64)
        )
        c_comparisons = sum(
            (row["comparisons"] for row in case_runs),
            np.zeros((13, 30, 24), dtype=np.int64),
        )
        deltas = [
            group_pck(c_correct[q], c_comparisons[q], fixed_top)
            - group_pck(c_correct[q], c_comparisons[q], fixed_bottom)
            for q in range(13)
        ]
        case_effects.append(
            {
                "case": case,
                "seed_count": len(case_runs),
                "mean_top_minus_bottom_pck32": float(np.nanmean(deltas)),
                "positive_anchor_fraction": float(np.mean(np.asarray(deltas) > 0)),
            }
        )
    values = np.asarray(
        [row["mean_top_minus_bottom_pck32"] for row in case_effects], dtype=np.float64
    )
    rng = np.random.default_rng(int(thresholds["bootstrap"]["seed"]))
    bootstrap = np.mean(
        values[rng.integers(0, len(values), size=(int(thresholds["bootstrap"]["resamples"]), len(values)))],
        axis=1,
    )
    bootstrap_lcb = float(np.quantile(bootstrap, 0.025))
    bootstrap_mean = float(values.mean())

    practical = thresholds["practical_thresholds"]
    null = thresholds["null"]
    checks = {
        "median_jaccard_practical": median_jaccard
        >= float(practical["median_pairwise_top100_jaccard"]),
        "median_spearman_practical": median_spearman
        >= float(practical["median_pairwise_spearman"]),
        "top_beats_bottom_anchor_fraction": anchor_fraction
        >= float(practical["fixed_top100_beats_fixed_bottom100_anchor_fraction"]),
        "top_minus_bottom_case_bootstrap_lcb": bootstrap_lcb
        > float(practical["case_cluster_bootstrap_lcb_top_minus_bottom_pck32"]),
    }
    exceeds_null = (
        median_jaccard > float(null["top100_jaccard_q99"])
        and median_spearman > float(null["spearman_q99"])
    )
    if all(checks.values()):
        decision = "pass"
    elif exceeds_null:
        decision = "conditional"
    else:
        decision = "fail"

    tube_correct = correct.sum(axis=0)
    tube_comparisons = comparisons.sum(axis=0)
    tube_error = error_sum.sum(axis=0)
    tube_ranking, _ = rank_heads(tube_correct, tube_comparisons, tube_error)
    tube_scopes = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.runs_root),
        "query_times": list(range(13)),
        "ranking_metric": "micro PCK@32 over all Qt to all Kt except same-time",
        "activated_by_decision": decision != "pass",
        "head_scopes": {
            "tube_top100": {"rank_start": 1, "rank_end": 100},
            "tube_bottom100": {"rank_start": 621, "rank_end": 720},
        },
        "entries": tube_ranking,
    }

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "case_count": len(by_case),
        "seeds_by_case": {case: sorted(row["seed"] for row in rows) for case, rows in by_case.items()},
        "thresholds": str(args.thresholds),
        "summary": {
            "median_pairwise_top100_jaccard": median_jaccard,
            "median_pairwise_spearman": median_spearman,
            "fixed_top100_beats_bottom100_anchor_fraction": anchor_fraction,
            "case_mean_top_minus_bottom_pck32": bootstrap_mean,
            "case_cluster_bootstrap_lcb_top_minus_bottom_pck32": bootstrap_lcb,
            "exceeds_permutation_null": exceeds_null,
            "checks": checks,
            "decision": decision,
        },
        "per_anchor": per_anchor,
        "pairwise_query_time_stability": pairwise,
        "case_effects": case_effects,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "report.json", result)
    atomic_json(args.output_dir / "tube_head_scopes.json", tube_scopes)

    with (args.output_dir / "per_anchor.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_time",
                "pixel_frame",
                "fixed_top100_pck32",
                "fixed_bottom100_pck32",
                "top_minus_bottom_pck32",
            ],
        )
        writer.writeheader()
        for row in per_anchor:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# Stage 1 Query-time Ranking Validation",
        "",
        f"- Runs: **{len(runs)}**; cases: **{len(by_case)}**.",
        f"- Decision: **{decision.upper()}**.",
        f"- Median pairwise Top100 Jaccard: **{median_jaccard:.4f}**.",
        f"- Median pairwise Spearman: **{median_spearman:.4f}**.",
        f"- Fixed Top100 beats Bottom100 anchors: **{anchor_fraction:.1%}**.",
        f"- Case-level Top−Bottom PCK@32: **{bootstrap_mean:.3f} pp**; "
        f"bootstrap lower 95% bound **{bootstrap_lcb:.3f} pp**.",
        "",
        "| check | pass |",
        "|---|---|",
        *[f"| {key} | {'yes' if value else 'no'} |" for key, value in checks.items()],
        "",
        "If the decision is CONDITIONAL or FAIL, `tube_head_scopes.json` contains the "
        "pre-specified TubeTop100/TubeBottom100 alternative while retaining fixed latest3350 controls.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
