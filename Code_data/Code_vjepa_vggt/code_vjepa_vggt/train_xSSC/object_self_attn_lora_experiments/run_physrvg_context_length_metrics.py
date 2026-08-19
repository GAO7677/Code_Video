#!/usr/bin/env python3
"""Run the existing AAAinfer metrics over context-length sweep outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BENCH = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py")
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
PYTHONPATH = ":".join(
    (
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt",
        "/home/gaoya/Code_Video/Code_data/Code_try0526",
    )
)

CPU_METRICS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
)
GPU_METRICS = (
    "wmreward",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("test5", "physiciq"), required=True)
    parser.add_argument(
        "--gpus",
        default="1,2,3,5,6,7",
        help="Physical GPUs for metric workers; GPU4 is rejected.",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise ValueError(f"invalid GPU list: {value!r}")
    if 4 in gpus:
        raise ValueError("GPU4 is prohibited")
    return gpus


def make_allowlist(result_root: Path, destination: Path) -> int:
    paths = []
    for result_json in sorted(result_root.glob("*.json")):
        if result_json.name.endswith(".input.json"):
            continue
        try:
            payload = load_json(result_json)
        except Exception:
            continue
        input_json = payload.get("input_json")
        if isinstance(input_json, str) and input_json.strip():
            paths.append(str(Path(input_json).resolve()))
    if not paths:
        raise RuntimeError(f"no result JSON inputs found under {result_root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(f"{path}\n" for path in sorted(set(paths))), encoding="utf-8")
    return len(set(paths))


def task_done(summary_path: Path, result_root: Path, expected: int) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
    except Exception:
        return False
    status = summary.get("metric_status", {})
    completed = int(status.get("completed", 0)) if isinstance(status, dict) else 0
    return completed == expected and int(summary.get("num_result_jsons", 0)) == expected


def run_one(
    *,
    dataset: str,
    context_length: int,
    result_root: Path,
    metric: str,
    output_root: Path,
    gpu: int | None,
    force: bool,
) -> dict[str, Any]:
    metric_root = output_root / dataset / f"ctx{context_length:02d}" / metric
    metric_root.mkdir(parents=True, exist_ok=True)
    allowlist = metric_root / "input_allowlist.txt"
    expected = make_allowlist(result_root, allowlist)
    summary_path = metric_root / "summary.json"
    if not force and task_done(summary_path, result_root, expected):
        return {"status": "skipped", "dataset": dataset, "ctx": context_length, "metric": metric}
    log_path = metric_root / "run.log"
    command = [
        str(PYTHON),
        str(BENCH),
        "--metric",
        metric,
        "--result-root",
        str(result_root),
        "--input-json-allowlist",
        str(allowlist),
        "--output-summary",
        str(summary_path),
    ]
    if metric == "wmreward":
        command.extend(["--wmreward-reset-interval", "1000000"])
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = PYTHONPATH
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(command, check=True, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    return {
        "status": "complete",
        "dataset": dataset,
        "ctx": context_length,
        "metric": metric,
        "gpu": gpu,
        "summary": str(summary_path),
        "expected_cases": expected,
    }


def main() -> None:
    args = parse_args()
    if args.workers_per_gpu < 1 or args.cpu_workers < 1:
        raise ValueError("worker counts must be positive")
    gpus = parse_gpus(args.gpus)
    sweep_root = args.sweep_root.expanduser().resolve()
    manifest = load_json(sweep_root / "sweep_manifest.json")
    records = [row for row in manifest.get("records", []) if row.get("dataset") == args.dataset]
    records = sorted(records, key=lambda row: int(row["context_frames"]))
    if not args.include_baseline:
        records = [row for row in records if not row.get("reused_existing_reference")]
    tasks = []
    for record in records:
        context_length = int(record["context_frames"])
        result_root = Path(record["result_root"]).resolve()
        if not result_root.is_dir():
            print(f"[skip] missing result root: {result_root}", flush=True)
            continue
        for metric in CPU_METRICS:
            tasks.append((context_length, result_root, metric, None))
        for metric in GPU_METRICS:
            for gpu_index in range(args.workers_per_gpu):
                # Replicate each metric task only once; workers-per-GPU controls
                # scheduling capacity below, not duplicate evaluation.
                if gpu_index == 0:
                    tasks.append((context_length, result_root, metric, None))
    if not tasks:
        print("[done] no pending metric tasks", flush=True)
        return

    cpu_tasks = [task for task in tasks if task[2] in CPU_METRICS]
    gpu_tasks = [task for task in tasks if task[2] in GPU_METRICS]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.cpu_workers) as pool:
        futures = [
            pool.submit(
                run_one,
                dataset=args.dataset,
                context_length=ctx,
                result_root=root,
                metric=metric,
                output_root=sweep_root / "metrics",
                gpu=None,
                force=args.force,
            )
            for ctx, root, metric, _ in cpu_tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())

    gpu_slots = [gpu for gpu in gpus for _ in range(args.workers_per_gpu)]
    with ThreadPoolExecutor(max_workers=max(1, len(gpu_slots))) as pool:
        futures = []
        for index, (ctx, root, metric, _) in enumerate(gpu_tasks):
            gpu = gpu_slots[index % len(gpu_slots)]
            futures.append(
                pool.submit(
                    run_one,
                    dataset=args.dataset,
                    context_length=ctx,
                    result_root=root,
                    metric=metric,
                    output_root=sweep_root / "metrics",
                    gpu=gpu,
                    force=args.force,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())
    write_json(
        sweep_root / "metrics" / f"{args.dataset}_metrics_manifest.json",
        {"dataset": args.dataset, "results": results},
    )
    print(f"[done] dataset={args.dataset} tasks={len(results)}", flush=True)


if __name__ == "__main__":
    main()
