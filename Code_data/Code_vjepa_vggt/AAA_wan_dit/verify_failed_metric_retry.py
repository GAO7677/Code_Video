#!/usr/bin/env python3
"""Verify retry summaries and the final strict plot statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--stats-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Retry manifest has no tasks list")

    failures: list[dict[str, Any]] = []
    for task in tasks:
        task_id = task["task_id"]
        summary_path = args.summary_dir / f"{task_id}.json"
        if not summary_path.is_file():
            failures.append({"task_id": task_id, "error": "missing_summary"})
            continue
        summary = load_json(summary_path)
        status = summary.get("metric_status", {})
        num_cases = int(status.get("num_cases", -1))
        num_success = int(status.get("num_success", -1))
        num_failed = int(status.get("num_failed", -1))
        if (
            num_cases != args.expected_cases
            or num_success != args.expected_cases
            or num_failed != 0
        ):
            failures.append(
                {
                    "task_id": task_id,
                    "metric": task["metric"],
                    "result_root": task["result_root"],
                    "num_cases": num_cases,
                    "num_success": num_success,
                    "num_failed": num_failed,
                }
            )

    incomplete_stats: list[dict[str, str]] = []
    num_stats = 0
    if args.stats_csv is not None:
        with args.stats_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                num_stats += 1
                if row["score_count"] != row["expected_count"]:
                    incomplete_stats.append(
                        {
                            "method_id": row["method_id"],
                            "metric": row["metric"],
                            "score_count": row["score_count"],
                            "expected_count": row["expected_count"],
                        }
                    )

    report = {
        "complete": not failures and not incomplete_stats,
        "num_retry_tasks": len(tasks),
        "num_verified_retry_tasks": len(tasks) - len(failures),
        "retry_failures": failures,
        "num_plot_stats": num_stats,
        "num_incomplete_plot_stats": len(incomplete_stats),
        "incomplete_plot_stats": incomplete_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "complete": report["complete"],
        "num_retry_tasks": report["num_retry_tasks"],
        "num_verified_retry_tasks": report["num_verified_retry_tasks"],
        "num_incomplete_plot_stats": report["num_incomplete_plot_stats"],
    }, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
