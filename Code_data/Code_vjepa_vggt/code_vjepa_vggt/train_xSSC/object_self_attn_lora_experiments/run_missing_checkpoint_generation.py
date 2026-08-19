#!/usr/bin/env python3
"""Fill missing test_5 and PhysicIQ generations on explicitly assigned GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import xssc_lora_checkpoint_watch as checkpoint_watch


def parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def parse_gpu_ids(value: str) -> list[int]:
    result = parse_csv_ints(value)
    if 4 in result:
        raise argparse.ArgumentTypeError("GPU4 is prohibited by workspace rules")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("GPU ids must be unique")
    return result


def parse_csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one method is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_gpu_ids, required=True)
    parser.add_argument("--methods", type=parse_csv_strings, default=None)
    parser.add_argument("--steps", type=parse_csv_ints, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test5-only", action="store_true")
    mode.add_argument("--physiciq-only", action="store_true")
    return parser.parse_args()


def complete_tasks(config: dict) -> list[dict]:
    return [
        task
        for task in checkpoint_watch.discover_checkpoints(config)
        if checkpoint_watch.checkpoint_complete(
            Path(task["checkpoint_dir"]),
            checkpoint_watch.method_config(config, task["method_key"]),
        )
    ]


def missing_test5_tasks(config: dict, tasks: list[dict]) -> list[dict]:
    return [
        task
        for task in tasks
        if not checkpoint_watch.manifest_path(
            config, task["method_key"], int(task["step"])
        ).is_file()
    ]


def run_test5(config: dict, tasks: list[dict], gpus: list[int]) -> None:
    def worker(gpu: int, queue: list[dict]) -> None:
        for task in queue:
            state = checkpoint_watch.state_paths(config)["state"]
            lock = (
                state
                / "inference_locks"
                / task["method_key"]
                / f"step-{int(task['step']):06d}.lock"
            )
            with checkpoint_watch.try_exclusive_lock(lock) as acquired:
                if not acquired:
                    manifest = checkpoint_watch.manifest_path(
                        config, task["method_key"], int(task["step"])
                    )
                    checkpoint_watch.log(
                        f"test_5 task already running; waiting for manifest: {task}"
                    )
                    while not manifest.is_file():
                        time.sleep(int(config["runtime"]["gpu_poll_seconds"]))
                    continue
                if checkpoint_watch.manifest_path(
                    config, task["method_key"], int(task["step"])
                ).is_file():
                    continue
                checkpoint_watch.log(
                    f"manual parallel test_5 start method={task['method_key']} "
                    f"step={task['step']} requested_gpu={gpu}"
                )
                with checkpoint_watch.reserve_available_gpu(config) as gpu_id:
                    checkpoint_watch.run_inference_task(config, task, gpu_id)

    queues = [[] for _ in gpus]
    for index, task in enumerate(tasks):
        queues[index % len(gpus)].append(task)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(worker, gpu, queue)
            for gpu, queue in zip(gpus, queues)
            if queue
        ]
        for future in as_completed(futures):
            future.result()
    checkpoint_watch.refresh_site(config)


def run_physiciq(config_path: Path, config: dict, tasks: list[dict], gpus: list[int]) -> None:
    phys_keys = set(config["physiciq"]["method_keys"])
    pending = []
    phys_root = Path(config["paths"]["watch_root"]) / "state" / "physiciq" / "inference"
    for task in tasks:
        method = str(task["method_key"])
        step = int(task["step"])
        if method not in phys_keys:
            continue
        if not (phys_root / method / f"step-{step:06d}.json").is_file():
            pending.append(task)
    if not pending:
        print("[missing-generation] no missing PhysicIQ tasks", flush=True)
        return

    methods = sorted({str(task["method_key"]) for task in pending})
    steps = sorted({int(task["step"]) for task in pending})
    subprocess.run(
        [
            config["paths"]["python"],
            str(ROOT / "xssc_lora_physiciq_parallel_infer.py"),
            "--config",
            str(config_path),
            "--gpus",
            ",".join(map(str, gpus)),
            "--methods",
            ",".join(methods),
            "--steps",
            ",".join(map(str, steps)),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["runtime"]["gpu_ids"] = args.gpus
    tasks = complete_tasks(config)
    if args.methods is not None:
        methods = set(args.methods)
        tasks = [task for task in tasks if task["method_key"] in methods]
    if args.steps is not None:
        steps = set(args.steps)
        tasks = [task for task in tasks if int(task["step"]) in steps]
    test5 = missing_test5_tasks(config, tasks)
    print(
        f"[missing-generation] complete_checkpoints={len(tasks)} "
        f"missing_test5={len(test5)} gpus={args.gpus}",
        flush=True,
    )
    for task in test5:
        print(
            f"  test_5: {task['method_key']} step={int(task['step'])}",
            flush=True,
        )
    if test5 and not args.physiciq_only:
        run_test5(config, test5, args.gpus)
    if not args.test5_only:
        run_physiciq(config_path, config, tasks, args.gpus)
    checkpoint_watch.refresh_site(config)
    print("[missing-generation] all generation gaps filled", flush=True)


if __name__ == "__main__":
    main()
