#!/usr/bin/env python3
"""Build metric queues for generation jobs that are already complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manage_remaining_block_pipeline import METRIC_GROUPS, find_leaf, read_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = set(read_paths(args.input_list))
    if len(allowed) != 67:
        raise ValueError(f"Expected 67 input cases, found {len(allowed)}")

    config_roots: list[Path] = []
    for line in args.completed.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            raise ValueError(f"Malformed completed row: {line}")
        config_roots.append(Path(fields[4]).expanduser().resolve())
    if not config_roots or len(config_roots) != len(set(config_roots)):
        raise ValueError("Completed config roots are empty or duplicated")

    leaves = [find_leaf(root, allowed) for root in config_roots]
    queue_dir = args.run_root / "queues"
    queue_dir.mkdir(parents=True, exist_ok=True)
    (args.run_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.run_root / "state").mkdir(parents=True, exist_ok=True)
    (args.run_root / "task_summaries").mkdir(parents=True, exist_ok=True)
    (args.run_root / "completed_roots.txt").write_text(
        "".join(f"{leaf}\n" for leaf in leaves), encoding="utf-8"
    )

    task_index = 0
    counts: dict[str, int] = {}
    for group, metrics in METRIC_GROUPS.items():
        target_group = "cpu" if group == "cpu" else "gpu"
        queue = queue_dir / f"{target_group}.tsv"
        mode = "a" if queue.exists() else "w"
        rows: list[str] = []
        for metric in metrics:
            for leaf in leaves:
                rows.append(
                    f"incremental-{task_index:05d}\t{metric}\t{leaf}\n"
                )
                task_index += 1
        with queue.open(mode, encoding="utf-8") as handle:
            handle.writelines(rows)
        counts[target_group] = counts.get(target_group, 0) + len(rows)

    for group in ("cpu", "gpu"):
        (queue_dir / f"{group}.cursor").write_text("1\n", encoding="utf-8")
        (queue_dir / f"{group}.lock").touch()
    (args.run_root / "completed_tasks.tsv").touch()
    (args.run_root / "failed_tasks.tsv").touch()
    report = {
        "num_roots": len(leaves),
        "num_tasks": task_index,
        "group_counts": counts,
        "roots": [str(path) for path in leaves],
    }
    (args.run_root / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
