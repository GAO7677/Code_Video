#!/usr/bin/env python3
"""Run filtered test_5 CPU metrics with the checkpoint watcher's state format."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import xssc_lora_checkpoint_watch as checkpoint_watch


def parse_csv_strings(value: str) -> set[str]:
    result = {item.strip() for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("at least one method is required")
    return result


def parse_csv_ints(value: str) -> set[int]:
    result = {int(item.strip()) for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("at least one step is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--methods", type=parse_csv_strings, required=True)
    parser.add_argument("--steps", type=parse_csv_ints, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def run_task(config: dict[str, Any], task: dict[str, Any]) -> None:
    manifest = task["manifest"]
    metric = str(task["metric"])
    method_key = str(manifest["method_key"])
    step = int(manifest["step"])
    marker = checkpoint_watch.metric_marker_path(config, method_key, step, metric)
    if marker.is_file():
        return
    lock = (
        checkpoint_watch.state_paths(config)["state"]
        / "metric_locks"
        / "cpu"
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.lock"
    )
    with checkpoint_watch.exclusive_lock(lock):
        if not marker.is_file():
            checkpoint_watch.run_metric_task(config, "cpu", task)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    checkpoint_watch.prepare_directories(config)
    tasks = [
        task
        for task in checkpoint_watch.metric_tasks(config, "cpu")
        if str(task["manifest"]["method_key"]) in args.methods
        and int(task["manifest"]["step"]) in args.steps
    ]
    checkpoint_watch.log(
        f"filtered test_5 CPU metric queue tasks={len(tasks)} workers={args.workers}"
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_task, config, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    if args.refresh:
        checkpoint_watch.refresh_site(config)
    checkpoint_watch.log("filtered test_5 CPU metric queue complete")


if __name__ == "__main__":
    main()
