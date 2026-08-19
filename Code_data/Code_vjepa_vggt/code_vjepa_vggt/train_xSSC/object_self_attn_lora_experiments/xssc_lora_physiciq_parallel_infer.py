#!/usr/bin/env python3
"""Parallel PhysicIQ inference runner for xSSC LoRA checkpoints.

This keeps the same inference script and manifest format as
xssc_lora_physiciq_watch.py, but assigns pending checkpoint tasks to multiple
explicit GPUs. It is meant for one-off catch-up runs when several checkpoints
already exist on disk.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

from xssc_lora_checkpoint_watch import (
    atomic_write_json,
    checkpoint_complete,
    discover_checkpoints,
    load_json,
    log,
    timestamp,
    validate_result_root,
)
from xssc_lora_physiciq_watch import (
    append_leaf_folder,
    method_name,
    phys_manifest_path,
    phys_state_root,
)


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if 4 in result:
        raise argparse.ArgumentTypeError("GPU 4 is prohibited by workspace rules")
    return result


def parse_csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one value")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_csv_ints, required=True)
    parser.add_argument("--steps", type=parse_csv_ints, required=True)
    parser.add_argument("--methods", type=parse_csv_strings, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--adopt-existing", action="store_true")
    return parser.parse_args()


def active_command_contains(needle: str) -> bool:
    process = subprocess.run(
        ["ps", "-eo", "cmd"],
        check=True,
        capture_output=True,
        text=True,
    )
    return needle in process.stdout


def phys_result_root(config: dict[str, Any], task: dict[str, Any]) -> Path:
    return Path(config["physiciq"]["output_root"]).resolve() / method_name(config, task)


def selected_tasks(
    config: dict[str, Any],
    steps: set[int],
    methods: set[str],
) -> list[dict[str, Any]]:
    phys_methods = set(config["physiciq"]["method_keys"])
    tasks = []
    for task in discover_checkpoints(config):
        step = int(task["step"])
        method_key = str(task["method_key"])
        checkpoint_dir = Path(task["checkpoint_dir"]).resolve()
        if step not in steps or method_key not in methods or method_key not in phys_methods:
            continue
        method_config = next(
            item for item in config["methods"] if item["key"] == method_key
        )
        if not checkpoint_complete(checkpoint_dir, method_config):
            continue
        if phys_manifest_path(config, method_key, step).is_file():
            continue
        tasks.append({**task, "checkpoint_dir": str(checkpoint_dir)})
    return sorted(tasks, key=lambda row: (int(row["step"]), str(row["method_key"])))


def write_pending(config: dict[str, Any], remaining: list[dict[str, Any]]) -> None:
    path = phys_state_root(config) / "inference.pending"
    if not remaining:
        path.unlink(missing_ok=True)
        return
    atomic_write_json(
        path,
        {
            "updated_utc": timestamp(),
            "num_pending": len(remaining),
            "runner": "parallel",
            "tasks": [
                {
                    "method_key": task["method_key"],
                    "step": int(task["step"]),
                    "checkpoint_dir": task["checkpoint_dir"],
                }
                for task in remaining
            ],
        },
    )


def register_phys_manifest(
    config: dict[str, Any],
    task: dict[str, Any],
    result_root: Path,
) -> None:
    validation = validate_result_root(
        config,
        result_root,
        input_list=Path(config["physiciq"]["input_list"]).resolve(),
        expected_cases=int(config["physiciq"]["expected_cases"]),
    )
    payload = {
        "method_key": task["method_key"],
        "method_label": task["method_label"],
        "step": int(task["step"]),
        "checkpoint_dir": task["checkpoint_dir"],
        "result_root": str(result_root.resolve()),
        "input_list": str(Path(config["physiciq"]["input_list"]).resolve()),
        "num_inference_steps": int(config["physiciq"]["num_inference_steps"]),
        "completed_utc": timestamp(),
        "validation": validation,
        "runner": "parallel",
    }
    atomic_write_json(
        phys_manifest_path(config, task["method_key"], int(task["step"])),
        payload,
    )
    meta_root = (
        Path(config["physiciq"]["output_root"]).resolve()
        / "_run_meta"
        / method_name(config, task)
    )
    atomic_write_json(meta_root / "batch_manifest.json", payload)
    append_leaf_folder(config, result_root)


def wait_for_existing_result(
    config: dict[str, Any],
    task: dict[str, Any],
    result_root: Path,
    poll_seconds: int,
) -> bool:
    name = method_name(config, task)
    while active_command_contains(name):
        try:
            register_phys_manifest(config, task, result_root)
            log(
                f"adopted running PhysicIQ result method={task['method_key']} "
                f"step={task['step']}"
            )
            return True
        except Exception:
            time.sleep(poll_seconds)
    try:
        register_phys_manifest(config, task, result_root)
        log(
            f"adopted existing PhysicIQ result method={task['method_key']} "
            f"step={task['step']}"
        )
        return True
    except Exception:
        return False


def run_task(
    config: dict[str, Any],
    task: dict[str, Any],
    gpu_id: int,
    adopt_existing: bool,
    poll_seconds: int,
) -> None:
    result_root = phys_result_root(config, task)
    if adopt_existing and result_root.exists():
        if wait_for_existing_result(config, task, result_root, poll_seconds):
            return

    phys = config["physiciq"]
    output_root = Path(phys["output_root"]).resolve()
    name = method_name(config, task)
    meta_root = output_root / "_run_meta" / name
    shard_root = meta_root / "shards"
    log_root = meta_root / "logs"
    trace_root = meta_root / "numeric_traces" / "shard_00"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    input_list = Path(phys["input_list"]).resolve()
    shard_file = shard_root / "shard_00.txt"
    shard_file.write_text(input_list.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = log_root / "shard_00.log"

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "TEST_LIST": str(shard_file),
            "NUM_INFERENCE_STEPS": str(phys["num_inference_steps"]),
            "STEP_OUTPUT_DIR_NAME": name,
            "SHARD_TAG": "shard_00",
            "TRACE_ROOT": str(trace_root),
        }
    )
    log(
        f"parallel PhysicIQ inference start method={task['method_key']} "
        f"step={task['step']} gpu={gpu_id}"
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        subprocess.run(
            [
                "bash",
                next(
                    item for item in config["methods"]
                    if item["key"] == task["method_key"]
                ).get("run_infer_script", config["paths"]["run_infer_script"]),
                task["checkpoint_dir"],
                str(gpu_id),
                str(output_root),
            ],
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    register_phys_manifest(config, task, result_root)
    log(
        f"parallel PhysicIQ inference complete method={task['method_key']} "
        f"step={task['step']}"
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    phys_state_root(config).mkdir(parents=True, exist_ok=True)

    tasks = selected_tasks(config, set(args.steps), set(args.methods))
    if not tasks:
        write_pending(config, [])
        log("parallel PhysicIQ inference: no pending tasks")
        return
    tasks = sorted(
        tasks,
        key=lambda task: (
            phys_result_root(config, task).exists()
            and active_command_contains(method_name(config, task)),
            int(task["step"]),
            str(task["method_key"]),
        ),
    )

    remaining_lock = threading.Lock()
    remaining = tasks.copy()
    write_pending(config, remaining)

    def worker(task: dict[str, Any], gpu_id: int) -> None:
        try:
            run_task(
                config,
                task,
                gpu_id,
                adopt_existing=args.adopt_existing,
                poll_seconds=args.poll_seconds,
            )
        finally:
            with remaining_lock:
                remaining[:] = [
                    item
                    for item in remaining
                    if not (
                        item["method_key"] == task["method_key"]
                        and int(item["step"]) == int(task["step"])
                    )
                ]
                write_pending(config, remaining)

    queues = [[] for _ in args.gpus]
    for index, task in enumerate(tasks):
        queues[index % len(args.gpus)].append(task)

    def gpu_worker(gpu_id: int, queue: list[dict[str, Any]]) -> None:
        for task in queue:
            worker(task, gpu_id)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(gpu_worker, gpu_id, queue)
            for gpu_id, queue in zip(args.gpus, queues)
            if queue
        ]
        for future in as_completed(futures):
            future.result()

    write_pending(config, [])
    log("parallel PhysicIQ inference complete")


if __name__ == "__main__":
    main()
