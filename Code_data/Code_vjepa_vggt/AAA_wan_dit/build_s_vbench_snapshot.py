#!/usr/bin/env python3
"""Build a VBench-only queue for complete result groups in a motion snapshot."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=20)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    args = parse_args()
    inventory_path = args.inventory.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in inventory["entries"]:
        if entry["family"] == "gt":
            continue
        key = (
            entry["family"],
            entry["model"],
            int(entry["seed"]),
            entry.get("variant"),
            entry.get("subset_id"),
        )
        groups[key].append(entry)

    complete_groups = []
    incomplete_groups = []
    queue_lines = []
    skipped_complete = 0
    task_index = 0
    for key, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(entries) != int(args.expected_cases):
            incomplete_groups.append({"key": list(key), "cases": len(entries)})
            continue
        sources = [Path(entry["source"]["path"]).resolve() for entry in entries]
        result_root = Path(os.path.commonpath([str(path) for path in sources]))
        if result_root.suffix == ".mp4":
            result_root = result_root.parent
        payloads = [read_result(path.with_suffix(".json")) for path in sources]
        missing_metrics = []
        for metric in METRICS:
            if all(payload.get(metric) is not None for payload in payloads):
                skipped_complete += 1
                continue
            task_id = f"vbench-{task_index:05d}"
            queue_lines.append(f"{task_id}\t{metric}\t{result_root}")
            missing_metrics.append(metric)
            task_index += 1
        complete_groups.append(
            {
                "key": list(key),
                "cases": len(entries),
                "result_root": str(result_root),
                "missing_metrics": missing_metrics,
            }
        )

    for child in ("queues", "logs", "state", "task_summaries"):
        (output / child).mkdir(parents=True, exist_ok=True)
    (output / "queues" / "gpu_common.tsv").write_text(
        "\n".join(queue_lines) + ("\n" if queue_lines else ""),
        encoding="utf-8",
    )
    (output / "queues" / "gpu_common.cursor").write_text("1\n", encoding="utf-8")
    (output / "queues" / "gpu_common.lock").touch()
    (output / "completed_tasks.tsv").write_text("", encoding="utf-8")
    (output / "failed_tasks.tsv").write_text("", encoding="utf-8")
    (output / "leaf_folders.txt").write_text(
        "\n".join(group["result_root"] for group in complete_groups) + "\n",
        encoding="utf-8",
    )
    plan = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": str(inventory_path),
        "metrics": list(METRICS),
        "expected_cases": int(args.expected_cases),
        "complete_groups": complete_groups,
        "incomplete_groups": incomplete_groups,
        "result_roots": len(complete_groups),
        "queued_tasks": len(queue_lines),
        "skipped_complete_tasks": skipped_complete,
    }
    atomic_json(output / "plan.json", plan)
    print(
        f"[s-vbench-snapshot] roots={len(complete_groups)} "
        f"queued={len(queue_lines)} already_complete={skipped_complete} "
        f"incomplete_groups={len(incomplete_groups)}"
    )


if __name__ == "__main__":
    main()
