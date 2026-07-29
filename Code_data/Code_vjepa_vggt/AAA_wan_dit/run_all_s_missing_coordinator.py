#!/usr/bin/env python3
"""Monitor missing All-S jobs and publish a verified completion marker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_head_role_dose_control_pilot_worker import (
    _job_root,
    _load_config,
    _tasks,
)

PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
CASE_GALLERY = Path(__file__).with_name(
    "build_head_role_dose_control_case_gallery.py"
)
DOSE_CONFIG = Path(__file__).with_name("head_role_dose_control_pilot.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_states(root: Path) -> tuple[Counter[str], list[dict[str, Any]]]:
    records = []
    for path in sorted((root / "state").glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return Counter(str(record.get("status", "unknown")) for record in records), records


def refresh_case_gallery(config_path: Path) -> None:
    result = subprocess.run(
        [
            str(PYTHON),
            str(CASE_GALLERY),
            "--config",
            str(DOSE_CONFIG),
            "--all-s-config",
            str(config_path),
        ],
        check=False,
    )
    if result.returncode:
        print(
            f"[all-s-coordinator] case gallery refresh failed: "
            f"{result.returncode}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config, root, manifest, cases, subset_ids = _load_config(config_path)
    tasks = _tasks(config, subset_ids)
    expected = len(tasks)
    poll_seconds = int(config["execution"]["poll_seconds"])
    max_attempts = int(config["execution"]["max_attempts_per_task"])
    complete_marker = root / "generation.complete"
    failed_marker = root / "generation.failed"
    last_gallery_complete = -1

    while True:
        counts, records = read_states(root)
        atomic_json(
            root / "progress.json",
            {
                "phase": "generation",
                "expected_tasks": expected,
                "expected_videos": expected * len(cases),
                "state_counts": dict(sorted(counts.items())),
                "updated_at_unix": time.time(),
            },
        )
        print(
            f"[all-s-coordinator] {dict(counts)} / expected={expected}",
            flush=True,
        )
        if counts["complete"] != last_gallery_complete:
            refresh_case_gallery(config_path)
            last_gallery_complete = counts["complete"]
        if counts["complete"] == expected:
            break
        exhausted = [
            record
            for record in records
            if record.get("status") == "failed"
            and int(record.get("attempt", 0)) >= max_attempts
        ]
        if exhausted:
            atomic_json(root / "generation_failure.json", {"tasks": exhausted})
            failed_marker.touch()
            raise RuntimeError(f"{len(exhausted)} tasks exhausted retries")
        time.sleep(poll_seconds)

    state_by_id = {record["task_id"]: record for record in records}
    manifest_records = []
    for model, seed, subset_id, start, end in tasks:
        task_id = (
            f"{model}__seed-{seed:06d}__{subset_id}"
            f"__steps{start:02d}_{end:02d}"
        )
        record = state_by_id[task_id]
        manifest_records.append(
            {
                "task_id": task_id,
                "model": model,
                "seed": seed,
                "subset_id": subset_id,
                "step_range": [start, end],
                "k": record["k"],
                "job_root": str(
                    _job_root(root, model, seed, subset_id, start, end)
                ),
                "videos": record["videos"],
            }
        )
    atomic_json(
        root / "generation_manifest.json",
        {
            "status": "complete",
            "config": str(config_path),
            "subset_manifest": str(manifest),
            "tasks": expected,
            "videos": expected * len(cases),
            "records": manifest_records,
        },
    )
    complete_marker.touch()
    refresh_case_gallery(config_path)
    print(
        f"[all-s-coordinator] complete: {expected} tasks, "
        f"{expected * len(cases)} videos",
        flush=True,
    )


if __name__ == "__main__":
    main()
