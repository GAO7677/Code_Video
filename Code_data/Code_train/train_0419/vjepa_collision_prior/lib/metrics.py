from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_ranking_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_queries": 0,
            "top1_accuracy": 0.0,
            "mean_gt_rank": None,
            "mean_positive_negative_margin": None,
            "per_horizon": {},
        }

    top1 = sum(1 for row in rows if row["gt_rank"] == 1)
    mean_rank = sum(row["gt_rank"] for row in rows) / len(rows)
    mean_margin = sum(row["positive_negative_margin"] for row in rows) / len(rows)

    per_horizon_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_horizon_rows[int(row["horizon"])].append(row)

    per_horizon = {}
    for horizon, group in sorted(per_horizon_rows.items()):
        per_horizon[horizon] = {
            "num_queries": len(group),
            "top1_accuracy": sum(1 for row in group if row["gt_rank"] == 1) / len(group),
            "mean_gt_rank": sum(row["gt_rank"] for row in group) / len(group),
            "mean_positive_negative_margin": sum(row["positive_negative_margin"] for row in group) / len(group),
        }

    return {
        "num_queries": len(rows),
        "top1_accuracy": top1 / len(rows),
        "mean_gt_rank": mean_rank,
        "mean_positive_negative_margin": mean_margin,
        "per_horizon": per_horizon,
    }
