#!/usr/bin/env python3
"""Create a recovery queue containing only incomplete generation configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from manage_remaining_block_pipeline import find_leaf, read_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = set(read_paths(args.input_list))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pending: list[dict[str, object]] = []
    complete: list[dict[str, object]] = []
    for index, job in enumerate(manifest["jobs"]):
        root = Path(job["config_root"]).expanduser().resolve()
        try:
            leaf = find_leaf(root, allowed)
        except ValueError:
            pending.append(job)
        else:
            complete.append({**job, "leaf_root": str(leaf)})

    rows = [
        "\t".join(
            (
                f"recovery-gen-{index:04d}",
                str(job["model"]),
                str(job["mode"]),
                str(job["block"]),
                str(job["config_root"]),
            )
        )
        + "\n"
        for index, job in enumerate(pending)
    ]
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.queue.write_text("".join(rows), encoding="utf-8")
    report = {
        "expected_configs": len(manifest["jobs"]),
        "complete_configs": len(complete),
        "pending_configs": len(pending),
        "complete": complete,
        "pending": pending,
    }
    args.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "expected_configs": report["expected_configs"],
                "complete_configs": report["complete_configs"],
                "pending_configs": report["pending_configs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
