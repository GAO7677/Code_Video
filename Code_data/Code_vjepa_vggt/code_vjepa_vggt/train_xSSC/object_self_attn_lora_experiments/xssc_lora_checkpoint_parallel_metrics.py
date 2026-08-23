#!/usr/bin/env python3
"""Run xSSC LoRA checkpoint GPU metrics on multiple idle GPUs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import subprocess
import threading
from typing import Any

from xssc_lora_checkpoint_watch import (
    atomic_write_json,
    exclusive_lock,
    load_json,
    load_manifests,
    log,
    metric_marker_path,
    prepare_directories,
    refresh_site,
    state_paths,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated GPU ids.")
    parser.add_argument("--methods", default="", help="Optional comma-separated method keys.")
    parser.add_argument("--steps", default="", help="Optional comma-separated optimizer steps.")
    parser.add_argument(
        "--metrics",
        default="",
        help="Optional comma-separated metric names.",
    )
    parser.add_argument("--kind", choices=["gpu"], default="gpu")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def parse_csv_strings(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def parse_csv_ints(value: str) -> set[int] | None:
    items = {int(item.strip()) for item in value.split(",") if item.strip()}
    return items or None


def metric_tasks(
    config: dict[str, Any],
    kind: str,
    method_filter: set[str] | None,
    step_filter: set[int] | None,
    metric_filter: set[str] | None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    metrics = config["metrics"][kind]
    for manifest in load_manifests(config):
        method_key = str(manifest["method_key"])
        step = int(manifest["step"])
        if method_filter is not None and method_key not in method_filter:
            continue
        if step_filter is not None and step not in step_filter:
            continue
        result_root = Path(manifest["result_root"])
        if not result_root.is_dir():
            continue
        for metric in metrics:
            if metric_filter is not None and metric not in metric_filter:
                continue
            marker = metric_marker_path(config, method_key, step, metric)
            if marker.is_file():
                continue
            tasks.append({"metric": metric, "manifest": manifest})
    return tasks


def run_metric_task_on_gpu(
    config: dict[str, Any],
    kind: str,
    task: dict[str, Any],
    gpu_id: int,
) -> None:
    metric = task["metric"]
    manifest = task["manifest"]
    method_key = str(manifest["method_key"])
    step = int(manifest["step"])
    marker_path = metric_marker_path(config, method_key, step, metric)
    if marker_path.is_file():
        return

    lock_path = (
        state_paths(config)["root"]
        / "state"
        / "metric_locks"
        / kind
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.lock"
    )
    with exclusive_lock(lock_path):
        if marker_path.is_file():
            return
        run_metric_task_on_gpu_unlocked(config, kind, task, gpu_id)


def run_metric_task_on_gpu_unlocked(
    config: dict[str, Any],
    kind: str,
    task: dict[str, Any],
    gpu_id: int,
) -> None:
    metric = task["metric"]
    manifest = task["manifest"]
    method_key = str(manifest["method_key"])
    step = int(manifest["step"])
    marker_path = metric_marker_path(config, method_key, step, metric)
    paths = state_paths(config)
    summary_path = (
        paths["summaries"]
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        paths["logs"]
        / "metrics_parallel"
        / kind
        / f"gpu{gpu_id}"
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        config["paths"]["python"],
        config["paths"]["bench_script"],
        "--metric",
        metric,
        "--result-root",
        str(manifest["result_root"]),
        "--input-json-allowlist",
        config["paths"]["input_list"],
        "--output-summary",
        str(summary_path),
    ]
    if metric == "wmreward":
        command.extend(["--wmreward-reset-interval", "1000000"])

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = (
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
        "/home/gaoya/Code_Video/Code_data/Code_try0526"
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log(
        f"parallel metric start gpu={gpu_id} method={method_key} "
        f"step={step} metric={metric}"
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            [
                config["paths"]["python"],
                config["paths"]["metric_validator_script"],
                str(summary_path),
                "--expected-cases",
                str(config["runtime"]["expected_cases"]),
            ],
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    marker = {
        "completed_utc": timestamp(),
        "kind": kind,
        "method_key": method_key,
        "step": step,
        "metric": metric,
        "result_root": str(manifest["result_root"]),
        "summary_path": str(summary_path),
        "worker_gpu": gpu_id,
    }
    atomic_write_json(marker_path, marker)
    log(
        f"parallel metric complete gpu={gpu_id} method={method_key} "
        f"step={step} metric={metric}"
    )


def worker(
    config: dict[str, Any],
    kind: str,
    task_queue: "queue.Queue[dict[str, Any]]",
    gpu_id: int,
    failures: list[str],
    failure_lock: threading.Lock,
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_metric_task_on_gpu(config, kind, task, gpu_id)
        except Exception as exc:  # noqa: BLE001 - keep long metric queue moving.
            manifest = task["manifest"]
            message = (
                f"gpu={gpu_id} method={manifest['method_key']} "
                f"step={manifest['step']} metric={task['metric']}: {exc}"
            )
            log(f"parallel metric failed {message}")
            with failure_lock:
                failures.append(message)
        finally:
            task_queue.task_done()


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    config["_config_path"] = str(args.config.resolve())
    prepare_directories(config)

    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if 4 in gpus:
        raise SystemExit("GPU4 is forbidden by workspace rules; remove it from --gpus.")
    if not gpus:
        raise SystemExit("No GPU ids provided.")
    if int(args.workers_per_gpu) < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")

    tasks = metric_tasks(
        config=config,
        kind=args.kind,
        method_filter=parse_csv_strings(args.methods),
        step_filter=parse_csv_ints(args.steps),
        metric_filter=parse_csv_strings(args.metrics),
    )
    log(f"parallel metric queue kind={args.kind} gpus={gpus} tasks={len(tasks)}")
    if not tasks:
        if args.refresh:
            refresh_site(config)
        return

    task_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    failures: list[str] = []
    failure_lock = threading.Lock()
    threads = []
    for gpu in gpus:
        for worker_index in range(int(args.workers_per_gpu)):
            threads.append(
                threading.Thread(
                    target=worker,
                    args=(config, args.kind, task_queue, gpu, failures, failure_lock),
                    name=f"gpu{gpu}-worker{worker_index}",
                    daemon=False,
                )
            )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if args.refresh:
        refresh_site(config)

    if failures:
        raise SystemExit("\n".join(failures))
    log("parallel metric queue complete")


if __name__ == "__main__":
    main()
