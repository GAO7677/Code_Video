#!/usr/bin/env python3
"""Drain missing test5 VBench metrics with resumable shared-GPU workers.

This queue deliberately runs only the seven VBench metrics.  It uses the
checkpoint watcher's per-task and per-GPU locks, so it can coexist with the
older metric queues and resume safely after an interruption.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import xssc_lora_checkpoint_watch as watch  # noqa: E402
import xssc_lora_checkpoint_parallel_metrics as parallel  # noqa: E402


VBENCH_METRICS = {
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,5,6,7",
        help="Candidate physical GPUs; GPU4 is intentionally excluded.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--idle-rounds",
        type=int,
        default=3,
        help="Exit after this many consecutive empty scans.",
    )
    return parser.parse_args()


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    if 4 in gpus:
        raise ValueError("GPU4 is forbidden by workspace rules")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"duplicate GPU ids: {gpus}")
    return gpus


def missing_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = parallel.metric_tasks(
        config,
        "gpu",
        method_filter=None,
        step_filter=None,
        metric_filter=VBENCH_METRICS,
    )
    # Inference manifests are validated before registration.  Keep this guard
    # so a stale/incomplete manifest can never start a VBench subprocess.
    return [
        task
        for task in tasks
        if Path(task["manifest"]["result_root"]).is_dir()
    ]


def task_lock_path(config: dict[str, Any], task: dict[str, Any]) -> Path:
    manifest = task["manifest"]
    return (
        watch.state_paths(config)["state"]
        / "metric_locks"
        / "gpu"
        / str(manifest["method_key"])
        / f"step-{int(manifest['step']):06d}"
        / f"{task['metric']}.lock"
    )


def run_one(config: dict[str, Any], task: dict[str, Any]) -> bool:
    manifest = task["manifest"]
    method = str(manifest["method_key"])
    step = int(manifest["step"])
    metric = str(task["metric"])
    marker = watch.metric_marker_path(config, method, step, metric)
    if marker.is_file():
        return False

    with watch.try_exclusive_lock(task_lock_path(config, task)) as acquired:
        if not acquired or marker.is_file():
            return False
        with watch.reserve_metric_gpu(config, allow_parallel=True) as gpu_id:
            watch.run_metric_task(config, "gpu", task, gpu_id)
        watch.log(
            f"vbench queue complete gpu={gpu_id} method={method} "
            f"step={step} metric={metric}"
        )
        return True


def ready_gpus(config: dict[str, Any], gpu_ids: list[int]) -> list[int]:
    threshold = int(config["runtime"]["gpu_ready_max_used_mib"])
    usage = {gpu_id: watch.gpu_memory_used(gpu_id) for gpu_id in gpu_ids}
    ready = [gpu_id for gpu_id in gpu_ids if usage[gpu_id] <= threshold]
    watch.log(
        "vbench GPU scan "
        + ", ".join(f"GPU{gpu_id}={usage[gpu_id]}MiB" for gpu_id in gpu_ids)
        + f" threshold={threshold} ready={ready}"
    )
    return ready


def main() -> None:
    args = parse_args()
    if args.workers_per_gpu < 1:
        raise SystemExit("--workers-per-gpu must be >= 1")
    if args.poll_seconds < 1 or args.idle_rounds < 1:
        raise SystemExit("poll and idle rounds must be positive")

    config_path = args.config.resolve()
    config = watch.load_json(config_path)
    config["_config_path"] = str(config_path)
    gpu_ids = parse_gpus(args.gpus)
    config["runtime"]["gpu_ids"] = gpu_ids
    config["runtime"]["gpu_metric_workers_per_gpu"] = int(args.workers_per_gpu)
    watch.prepare_directories(config)

    parallelism = len(gpu_ids) * int(args.workers_per_gpu)
    empty_scans = 0
    while True:
        tasks = missing_tasks(config)
        watch.log(
            f"vbench queue scan missing={len(tasks)} "
            f"candidate_gpus={gpu_ids} workers_per_gpu={args.workers_per_gpu}"
        )
        if not tasks:
            empty_scans += 1
            if empty_scans >= args.idle_rounds:
                watch.log("vbench queue drained")
                return
            time.sleep(args.poll_seconds)
            continue

        empty_scans = 0
        ready = ready_gpus(config, gpu_ids)
        if not ready:
            time.sleep(args.poll_seconds)
            continue
        # Freeze the low-memory candidate set for this batch.  This prevents
        # the watcher's shared-slot allowance from attaching to an unrelated
        # high-memory inference process that already occupies a GPU.
        batch_config = dict(config)
        batch_config["runtime"] = dict(config["runtime"])
        batch_config["runtime"]["gpu_ids"] = ready
        batch_config["runtime"]["gpu_metric_workers_per_gpu"] = int(
            args.workers_per_gpu
        )
        batch_parallelism = len(ready) * int(args.workers_per_gpu)
        batch = tasks[:batch_parallelism]
        handled = 0
        with ThreadPoolExecutor(
            max_workers=batch_parallelism,
            thread_name_prefix="vbench-worker",
        ) as executor:
            futures = {
                executor.submit(run_one, batch_config, task): task for task in batch
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    handled += int(future.result())
                except Exception as exc:  # keep the resumable queue alive
                    manifest = task["manifest"]
                    watch.log(
                        f"vbench queue failed method={manifest['method_key']} "
                        f"step={manifest['step']} metric={task['metric']}: {exc}"
                    )
        if handled == 0:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
