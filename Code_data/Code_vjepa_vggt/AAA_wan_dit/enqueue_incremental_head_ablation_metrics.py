#!/usr/bin/env python3
"""Continuously enqueue metrics for newly completed head-ablation configs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import TextIO

from configured_head_ablation import load_config, result_config_count, run_root


KINDS = ("cpu", "gpu_common", "gpu_heavy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def append_locked(path: Path, lock: TextIO, lines: list[str]) -> None:
    if not lines:
        return
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    generation_root = run_root(config) / "generation"
    output = args.output_root.expanduser().resolve()
    for child in ("queues", "logs", "state", "task_summaries"):
        (output / child).mkdir(parents=True, exist_ok=True)
    queue_paths = {kind: output / "queues" / f"{kind}.tsv" for kind in KINDS}
    queue_locks = {
        kind: (output / "queues" / f"{kind}.lock").open("a+", encoding="utf-8")
        for kind in KINDS
    }
    for kind, path in queue_paths.items():
        path.touch()
        cursor = output / "queues" / f"{kind}.cursor"
        if not cursor.exists():
            atomic_text(cursor, "1\n")
    for path in (output / "completed_tasks.tsv", output / "failed_tasks.tsv"):
        path.touch()

    ledger_path = output / "enqueued_configs.txt"
    enqueued = (
        {
            line.strip()
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if ledger_path.is_file()
        else set()
    )
    expected = result_config_count(config)
    metric_groups = config["metrics"]["groups"]
    metrics = {
        "cpu": list(metric_groups["cpu"]),
        "gpu_common": list(metric_groups["gpu_common"]),
        "gpu_heavy": [
            *metric_groups["videophy2"],
            *metric_groups["cosmos"],
        ],
    }

    while True:
        complete_states = sorted(
            (generation_root / "task_state").glob("*.complete")
        )
        new_task_ids = [
            path.stem for path in complete_states if path.stem not in enqueued
        ]
        added = 0
        for task_id in new_task_ids:
            validation_path = generation_root / "validations" / f"{task_id}.json"
            if not validation_path.is_file():
                continue
            validation = json.loads(
                validation_path.read_text(encoding="utf-8")
            )
            if int(validation.get("num_cases", -1)) != int(
                config["input"]["expected_unique_cases"]
            ):
                raise RuntimeError(f"invalid validation: {validation_path}")
            result_root = str(Path(validation["result_root"]).resolve())
            for kind in KINDS:
                lines = [
                    f"inc-{task_id}-{metric}\t{metric}\t{result_root}\t0\n"
                    for metric in metrics[kind]
                ]
                append_locked(queue_paths[kind], queue_locks[kind], lines)
            enqueued.add(task_id)
            added += 1
            atomic_text(ledger_path, "\n".join(sorted(enqueued)) + "\n")

        leaf_paths = []
        for task_id in sorted(enqueued):
            validation_path = generation_root / "validations" / f"{task_id}.json"
            if validation_path.is_file():
                payload = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
                leaf_paths.append(str(Path(payload["result_root"]).resolve()))
        atomic_text(
            output / "leaf_folders_incremental.txt",
            "\n".join(leaf_paths) + ("\n" if leaf_paths else ""),
        )
        status = {
            "expected_configs": expected,
            "generation_complete_configs": len(complete_states),
            "enqueued_configs": len(enqueued),
            "newly_enqueued": added,
            "updated_at_unix": time.time(),
        }
        atomic_text(
            output / "enqueue_status.json",
            json.dumps(status, indent=2) + "\n",
        )
        print(
            "[incremental-enqueue] "
            f"generation={len(complete_states)}/{expected} "
            f"enqueued={len(enqueued)} added={added}",
            flush=True,
        )
        if len(enqueued) == expected:
            (output / "enqueue.complete").touch()
            break
        time.sleep(args.poll_seconds)

    for lock in queue_locks.values():
        lock.close()


if __name__ == "__main__":
    main()
