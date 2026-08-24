#!/usr/bin/env python3
"""Run the local strict P0 metrics through the existing ``bench.py``.

This queue is resumable and uses the workspace's shared GPU lock files.  By
default it waits for an idle GPU; ``--stack-busy`` enables shared metric slots
for explicitly approved GPUs so metric models can coexist with other work.
GPU 4 is never a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physicsiq-verified-strict-metrics"
)
BENCH = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py"
)
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
SHARED_WATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch"
)
METRICS = (
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "videophy2",
    "cosmos_reason1",
)
VBench_METRICS = set(METRICS[:7])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--gpus",
        default="2,3,5",
        help="Physical GPU candidates; GPU4 is rejected. Defaults to 2,3,5.",
    )
    parser.add_argument(
        "--gpu-ready-max-used-mib",
        type=int,
        default=512,
        help="Only use a candidate below this memory watermark.",
    )
    parser.add_argument(
        "--stack-busy",
        action="store_true",
        help=(
            "Use shared metric slots and permit the selected GPUs to be busy. "
            "Use only for explicitly approved GPU candidates."
        ),
    )
    parser.add_argument(
        "--gpu-workers-per-gpu",
        type=int,
        default=2,
        help="Shared metric slots per GPU when --stack-busy is enabled.",
    )
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_gpu_ids(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Invalid --gpus value: {raw!r}")
    if any(value < 0 for value in values) or 4 in values:
        raise ValueError("GPU4 is forbidden and GPU ids must be non-negative")
    return values


def shared_lock_config(
    gpus: list[int],
    threshold: int,
    poll_seconds: int,
    *,
    stack_busy: bool,
    workers_per_gpu: int,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "gpu_ids": gpus,
        "gpu_ready_max_used_mib": threshold,
        "gpu_poll_seconds": poll_seconds,
        "gpu_metric_workers_per_gpu": workers_per_gpu,
    }
    if stack_busy:
        # The shared reservation helper normally only admits a busy GPU when
        # another metric slot is already occupied.  This queue's explicit
        # --stack-busy opt-in makes the memory watermark permissive for the
        # selected physical GPUs; the caller remains responsible for choosing
        # safe candidates.
        runtime["gpu_metric_parallel_metrics"] = list(METRICS)
    return {
        "paths": {"watch_root": str(SHARED_WATCH_ROOT)},
        "runtime": runtime,
    }


def reserve_gpu(config: dict[str, Any], *, allow_parallel: bool):
    # Import lazily so preparation/inspection remains usable without loading
    # the large watcher dependency tree.
    sys.path.insert(
        0,
        str(
            Path(
                "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
                "code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
            )
        ),
    )
    from xssc_lora_checkpoint_watch import reserve_metric_gpu

    return reserve_metric_gpu(config, allow_parallel=allow_parallel)


def summary_complete(path: Path, expected_cases: int) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing summary"
    try:
        payload = load_json(path)
        status = payload.get("metric_status") or {}
        cases = int(status.get("num_cases") or 0)
        success = int(status.get("num_success") or 0)
        failed = int(status.get("num_failed") or 0)
        completed = int(status.get("completed") or 0)
        errors = payload.get("errors")
        error_count = len(errors) if isinstance(errors, list) else 0
    except Exception as exc:  # noqa: BLE001
        return False, f"invalid summary: {exc}"
    if cases == expected_cases and success == expected_cases and failed == 0 and completed == expected_cases and error_count == 0:
        return True, f"{success}/{cases}"
    return False, f"cases={cases} success={success} failed={failed} completed={completed} errors={error_count}"


def task_log(root: Path, method_key: str, metric: str) -> Path:
    return root / "logs" / method_key / f"{metric}.log"


def run_task(
    *,
    root: Path,
    method: dict[str, Any],
    metric: str,
    lock_config: dict[str, Any],
    retries: int,
    status_lock: Lock,
    stack_busy: bool,
) -> dict[str, Any]:
    method_key = str(method["key"])
    result_root = Path(method["result_root"])
    summary_path = root / "summaries" / method_key / f"{metric}.json"
    log_path = task_log(root, method_key, metric)
    expected_cases = int(method.get("num_cases", 198))
    complete, detail = summary_complete(summary_path, expected_cases)
    if complete:
        with status_lock:
            print(f"[{timestamp()}] skip {method_key}/{metric}: {detail}", flush=True)
        return {"method": method_key, "metric": metric, "status": "skipped"}

    command = [
        str(PYTHON),
        str(BENCH),
        "--metric",
        metric,
        "--result-root",
        str(result_root),
        "--input-json-allowlist",
        str(root / "strict_input_allowlist.txt"),
        "--output-summary",
        str(summary_path),
    ]
    if metric == "videophy2":
        command.extend(["--videophy2-task", "generated_only_sa_pc_joint"])
    if metric in VBench_METRICS:
        command.extend(
            [
                "--vbench-output-root",
                str(root / "vbench_outputs" / method_key / metric),
            ]
        )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": ":".join(
                (
                    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt",
                    "/home/gaoya/Code_Video/Code_data/Code_try0526",
                )
            ),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 2):
        with status_lock:
            print(
                f"[{timestamp()}] waiting GPU for {method_key}/{metric} "
                f"attempt={attempt}",
                flush=True,
            )
        with reserve_gpu(lock_config, allow_parallel=stack_busy) as gpu_id:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with status_lock:
                print(
                    f"[{timestamp()}] start {method_key}/{metric} on GPU{gpu_id}",
                    flush=True,
                )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n[{timestamp()}] command attempt={attempt} gpu={gpu_id}\n"
                    + " ".join(command)
                    + "\n"
                )
                completed_process = subprocess.run(
                    command,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                handle.write(
                    f"[{timestamp()}] exit={completed_process.returncode}\n"
                )
        complete, detail = summary_complete(summary_path, expected_cases)
        if complete:
            with status_lock:
                print(
                    f"[{timestamp()}] done {method_key}/{metric}: {detail}",
                    flush=True,
                )
            return {"method": method_key, "metric": metric, "status": "complete"}
        with status_lock:
            print(
                f"[{timestamp()}] incomplete {method_key}/{metric}: {detail}",
                flush=True,
            )
    return {"method": method_key, "metric": metric, "status": "failed", "detail": detail}


def main() -> None:
    args = parse_args()
    if args.gpu_ready_max_used_mib < 0 or args.poll_seconds < 1 or args.retries < 0:
        raise ValueError("watermark, poll interval, and retries must be non-negative")
    root = args.root.expanduser().resolve()
    manifest = load_json(root / "manifest.json")
    methods = list(manifest["methods"])
    if args.gpu_workers_per_gpu < 1:
        raise ValueError("--gpu-workers-per-gpu must be positive")
    gpus = parse_gpu_ids(args.gpus)
    effective_threshold = args.gpu_ready_max_used_mib
    if args.stack_busy:
        # In stacking mode the explicit opt-in supersedes the idle watermark;
        # lock sharing still prevents two exclusive metric reservations from
        # colliding on the same physical GPU.
        effective_threshold = max(effective_threshold, 1_000_000)
    lock_config = shared_lock_config(
        gpus,
        effective_threshold,
        args.poll_seconds,
        stack_busy=args.stack_busy,
        workers_per_gpu=args.gpu_workers_per_gpu,
    )
    task_list = [
        (method, metric)
        for method in methods
        for metric in METRICS
    ]
    state_path = root / "queue_state.json"
    atomic_write(
        state_path,
        {
            "started_at": timestamp(),
            "root": str(root),
            "gpus": gpus,
            "gpu_ready_max_used_mib": effective_threshold,
            "stack_busy": args.stack_busy,
            "tasks": [
                {"method": method["key"], "metric": metric, "status": "queued"}
                for method, metric in task_list
            ],
        },
    )
    print(
        f"[{timestamp()}] strict metric queue tasks={len(task_list)} "
        f"gpus={gpus} threshold={effective_threshold}MiB "
        f"stack_busy={args.stack_busy}",
        flush=True,
    )
    status_lock = Lock()
    worker_count = args.workers or len(gpus)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        futures = [
            executor.submit(
                run_task,
                root=root,
                method=method,
                metric=metric,
                lock_config=lock_config,
                retries=args.retries,
                status_lock=status_lock,
                stack_busy=args.stack_busy,
            )
            for method, metric in task_list
        ]
        for future in as_completed(futures):
            results.append(future.result())
            complete_count = sum(item["status"] in {"complete", "skipped"} for item in results)
            with status_lock:
                print(
                    f"[{timestamp()}] queue progress {complete_count}/{len(task_list)}",
                    flush=True,
                )
    final_state = {
        "finished_at": timestamp(),
        "tasks": sorted(results, key=lambda item: (item["method"], item["metric"])),
    }
    atomic_write(state_path, final_state)
    failed = [item for item in results if item["status"] == "failed"]
    print(
        f"[{timestamp()}] queue finished complete={len(results) - len(failed)} "
        f"failed={len(failed)}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
