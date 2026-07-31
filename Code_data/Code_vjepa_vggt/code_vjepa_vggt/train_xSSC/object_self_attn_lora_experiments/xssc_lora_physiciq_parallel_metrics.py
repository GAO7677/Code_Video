#!/usr/bin/env python3
"""Run formal PhysicIQ metrics for xSSC LoRA checkpoints in parallel."""

from __future__ import annotations

import argparse
import fcntl
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
    log,
    timestamp,
)
from xssc_lora_physiciq_watch import (
    load_phys_manifests,
    phys_metric_marker_path,
    phys_state_root,
    refresh_plots_if_complete,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids for GPU metrics.")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--methods", default="", help="Optional comma-separated method keys.")
    parser.add_argument("--steps", default="", help="Optional comma-separated optimizer steps.")
    parser.add_argument("--metrics", default="", help="Optional comma-separated metric names.")
    parser.add_argument("--refresh-plots", action="store_true")
    parser.add_argument(
        "--skip-locked",
        action="store_true",
        help="Skip a metric if another worker already holds its lock.",
    )
    return parser.parse_args()


def parse_csv_strings(value: str) -> set[str] | None:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def parse_csv_ints(value: str) -> set[int] | None:
    items = {int(item.strip()) for item in value.split(",") if item.strip()}
    return items or None


def metric_tasks(
    config: dict[str, Any],
    *,
    kind: str,
    method_filter: set[str] | None,
    step_filter: set[int] | None,
    metric_filter: set[str] | None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    metrics = [
        metric
        for metric in config["metrics"][kind]
        if metric_filter is None or metric in metric_filter
    ]
    for manifest in load_phys_manifests(config):
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
            marker = phys_metric_marker_path(config, method_key, step, metric)
            if marker.is_file():
                continue
            tasks.append({"manifest": manifest, "metric": metric})
    return tasks


def run_metric_unlocked(
    config: dict[str, Any],
    *,
    kind: str,
    task: dict[str, Any],
    worker_label: str,
    cuda_visible_devices: str,
) -> None:
    manifest = task["manifest"]
    metric = str(task["metric"])
    method_key = str(manifest["method_key"])
    step = int(manifest["step"])
    root = Path(config["paths"]["watch_root"]).resolve()
    summary_path = (
        root
        / "physiciq_metric_task_summaries"
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.json"
    )
    log_path = (
        root
        / "logs"
        / "physiciq_metrics_parallel"
        / kind
        / worker_label
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.log"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        config["paths"]["python"],
        config["paths"]["bench_script"],
        "--metric",
        metric,
        "--result-root",
        str(manifest["result_root"]),
        "--input-json-allowlist",
        config["physiciq"]["input_list"],
        "--output-summary",
        str(summary_path),
    ]
    if metric == "wmreward":
        command.extend(["--wmreward-reset-interval", "1000000"])

    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
        "/home/gaoya/Code_Video/Code_data/Code_try0526"
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    log(
        f"PhysicIQ parallel metric start worker={worker_label} kind={kind} "
        f"method={method_key} step={step} metric={metric}"
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
                str(config["physiciq"]["expected_cases"]),
            ],
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

    atomic_write_json(
        phys_metric_marker_path(config, method_key, step, metric),
        {
            "completed_utc": timestamp(),
            "kind": kind,
            "method_key": method_key,
            "step": step,
            "metric": metric,
            "result_root": str(manifest["result_root"]),
            "summary_path": str(summary_path),
            "worker": worker_label,
        },
    )
    log(
        f"PhysicIQ parallel metric complete worker={worker_label} "
        f"method={method_key} step={step} metric={metric}"
    )
    refresh_plots_if_complete(config, manifest)


def run_metric(
    config: dict[str, Any],
    *,
    kind: str,
    task: dict[str, Any],
    worker_label: str,
    cuda_visible_devices: str,
    skip_locked: bool,
) -> None:
    manifest = task["manifest"]
    metric = str(task["metric"])
    method_key = str(manifest["method_key"])
    step = int(manifest["step"])
    marker_path = phys_metric_marker_path(config, method_key, step, metric)
    if marker_path.is_file():
        return
    lock_path = (
        phys_state_root(config)
        / "metric_locks"
        / kind
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.lock"
    )
    if skip_locked:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log(
                    f"PhysicIQ parallel metric skip locked worker={worker_label} "
                    f"method={method_key} step={step} metric={metric}"
                )
                return
            if marker_path.is_file():
                return
            run_metric_unlocked(
                config,
                kind=kind,
                task=task,
                worker_label=worker_label,
                cuda_visible_devices=cuda_visible_devices,
            )
        return
    with exclusive_lock(lock_path):
        if marker_path.is_file():
            return
        run_metric_unlocked(
            config,
            kind=kind,
            task=task,
            worker_label=worker_label,
            cuda_visible_devices=cuda_visible_devices,
        )


def worker(
    config: dict[str, Any],
    *,
    kind: str,
    worker_label: str,
    cuda_visible_devices: str,
    task_queue: "queue.Queue[dict[str, Any]]",
    failures: list[str],
    failure_lock: threading.Lock,
    skip_locked: bool,
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_metric(
                config,
                kind=kind,
                task=task,
                worker_label=worker_label,
                cuda_visible_devices=cuda_visible_devices,
                skip_locked=skip_locked,
            )
        except Exception as exc:  # noqa: BLE001 - keep queue moving.
            manifest = task["manifest"]
            message = (
                f"worker={worker_label} kind={kind} method={manifest['method_key']} "
                f"step={manifest['step']} metric={task['metric']}: {exc}"
            )
            log(f"PhysicIQ parallel metric failed {message}")
            with failure_lock:
                failures.append(message)
        finally:
            task_queue.task_done()


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    config["_config_path"] = str(args.config.resolve())
    if not config.get("physiciq", {}).get("enabled"):
        raise SystemExit("PhysicIQ watcher is disabled in config")
    if int(args.workers_per_gpu) < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")
    if int(args.cpu_workers) < 1:
        raise SystemExit("--cpu-workers must be >= 1")

    task_list = metric_tasks(
        config,
        kind=args.kind,
        method_filter=parse_csv_strings(args.methods),
        step_filter=parse_csv_ints(args.steps),
        metric_filter=parse_csv_strings(args.metrics),
    )
    log(f"PhysicIQ parallel metric queue kind={args.kind} tasks={len(task_list)}")
    if not task_list:
        return

    task_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
    for task in task_list:
        task_queue.put(task)

    worker_specs: list[tuple[str, str]] = []
    if args.kind == "gpu":
        gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
        if not gpus:
            raise SystemExit("--gpus is required for --kind gpu")
        if 4 in gpus:
            raise SystemExit("GPU4 is forbidden by workspace rules; remove it from --gpus.")
        for gpu_id in gpus:
            for worker_index in range(int(args.workers_per_gpu)):
                worker_specs.append((f"gpu{gpu_id}-worker{worker_index}", str(gpu_id)))
    else:
        worker_specs = [(f"cpu-worker{index}", "") for index in range(int(args.cpu_workers))]

    failures: list[str] = []
    failure_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(config,),
            kwargs={
                "kind": args.kind,
                "worker_label": label,
                "cuda_visible_devices": cuda_value,
                "task_queue": task_queue,
                "failures": failures,
                "failure_lock": failure_lock,
                "skip_locked": bool(args.skip_locked),
            },
            name=label,
            daemon=False,
        )
        for label, cuda_value in worker_specs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if args.refresh_plots:
        for manifest in load_phys_manifests(config):
            refresh_plots_if_complete(config, manifest)

    if failures:
        raise SystemExit("\n".join(failures))
    log("PhysicIQ parallel metric queue complete")


if __name__ == "__main__":
    main()
