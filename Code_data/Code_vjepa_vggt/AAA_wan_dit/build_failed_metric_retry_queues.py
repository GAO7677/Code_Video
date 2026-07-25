#!/usr/bin/env python3
"""Build retry queues from metric summaries containing failed cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


QUEUE_GROUPS = {
    "videophy2": "videophy2",
    "cosmos_reason1": "cosmos",
}


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    queue_rows: dict[str, list[tuple[str, str, str, str, int]]] = {
        "videophy2": [],
        "cosmos": [],
        "gpu_common": [],
    }
    metric_counts: Counter[str] = Counter()
    failed_case_count = 0

    for path in sorted(args.summary_dir.glob("*.json")):
        payload = load_summary(path)
        status = payload.get("metric_status")
        if not isinstance(status, dict):
            continue
        num_failed = int(status.get("num_failed", 0))
        if num_failed <= 0:
            continue

        metric = payload.get("metric")
        result_root = payload.get("result_root")
        if not isinstance(metric, str) or not isinstance(result_root, str):
            raise ValueError(f"Missing metric/result_root in {path}")

        group = QUEUE_GROUPS.get(metric, "gpu_common")
        index = len(queue_rows[group])
        task_id = f"retry-{group}-{index:04d}"
        queue_rows[group].append(
            (task_id, metric, result_root, path.stem, num_failed)
        )
        metric_counts[metric] += 1
        failed_case_count += num_failed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group, rows in queue_rows.items():
        queue_path = args.output_dir / f"{group}.tsv"
        queue_path.write_text(
            "".join(
                f"{task_id}\t{metric}\t{result_root}\n"
                for task_id, metric, result_root, _, _ in rows
            ),
            encoding="utf-8",
        )

    report = {
        "source_summary_dir": str(args.summary_dir.resolve()),
        "num_retry_tasks": sum(len(rows) for rows in queue_rows.values()),
        "num_failed_case_evaluations": failed_case_count,
        "queue_task_counts": {
            group: len(rows) for group, rows in queue_rows.items()
        },
        "metric_task_counts": dict(sorted(metric_counts.items())),
        "tasks": [
            {
                "task_id": task_id,
                "metric": metric,
                "result_root": result_root,
                "source_task_id": source_task_id,
                "previous_failed_cases": num_failed,
                "queue": group,
            }
            for group, rows in queue_rows.items()
            for task_id, metric, result_root, source_task_id, num_failed in rows
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "num_retry_tasks",
        "num_failed_case_evaluations",
        "queue_task_counts",
        "metric_task_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()
