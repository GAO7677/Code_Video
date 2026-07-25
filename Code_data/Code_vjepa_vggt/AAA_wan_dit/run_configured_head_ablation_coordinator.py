#!/usr/bin/env python3
"""Coordinate generation completion, metric queues, and final verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from configured_head_ablation import (
    METRIC_KINDS,
    load_config,
    metric_count,
    metric_worker_count,
    read_unique_inputs,
    result_config_count,
    run_root,
)


SUMMARY = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_vjepa_vggt/train0705_kubric_no_gt_box/"
    "summarize_benchmark_txt_metrics.py"
)
VERIFY_METRICS = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_vjepa_vggt/train0705_kubric_no_gt_box/"
    "verify_bench_physiq_metrics.py"
)
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def _wait_for_generation(config: dict) -> None:
    root = run_root(config)
    generation = root / "generation"
    expected_workers = (
        len(config["experiment"]["gpus"])
        * int(config["experiment"]["generation_workers_per_gpu"])
    )
    expected_jobs = result_config_count(config)
    while True:
        states = len(list((generation / "state").glob("*.json")))
        completed = _line_count(generation / "completed.tsv")
        failed = _line_count(generation / "failed.tsv")
        print(
            f"[coordinator] generation workers={states}/{expected_workers} "
            f"configs={completed}/{expected_jobs} failed={failed}",
            flush=True,
        )
        if states >= expected_workers:
            break
        time.sleep(30)
    if failed or completed != expected_jobs:
        (root / "generation.failed").touch()
        raise RuntimeError(
            f"generation incomplete: completed={completed}, failed={failed}, "
            f"expected={expected_jobs}"
        )


def _prepare_metric_queues(config: dict) -> list[Path]:
    root = run_root(config)
    generation = root / "generation"
    expected_jobs = result_config_count(config)
    result_roots: list[Path] = []
    manifest = []
    for line in (generation / "queue.tsv").read_text(encoding="utf-8").splitlines():
        task_id, model, block, head = line.split("\t")
        validation_path = generation / "validations" / f"{task_id}.json"
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        result_path = Path(payload["result_root"]).expanduser().resolve()
        result_roots.append(result_path)
        manifest.append(
            {
                "task_id": task_id,
                "model": model,
                "block": int(block),
                "head": int(head),
                "result_root": str(result_path),
                "validation": str(validation_path),
            }
        )
    if len(result_roots) != expected_jobs or len(set(result_roots)) != expected_jobs:
        raise RuntimeError("generation validations do not contain unique result roots")

    leaf_list = root / "leaf_folders.txt"
    leaf_list.write_text(
        "\n".join(str(path) for path in result_roots) + "\n",
        encoding="utf-8",
    )
    (root / "generation_manifest.json").write_text(
        json.dumps(
            {
                "num_configs": expected_jobs,
                "num_cases_per_config": len(read_unique_inputs(config)),
                "configs": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics_root = root / "metrics"
    for child in ("queues", "logs", "state", "task_summaries"):
        (metrics_root / child).mkdir(parents=True, exist_ok=True)
    (metrics_root / "completed_tasks.tsv").write_text("", encoding="utf-8")
    (metrics_root / "failed_tasks.tsv").write_text("", encoding="utf-8")
    for kind in METRIC_KINDS:
        queue_path = metrics_root / "queues" / f"{kind}.tsv"
        lines = []
        task_index = 0
        for result_path in result_roots:
            for metric in config["metrics"]["groups"][kind]:
                lines.append(
                    f"{kind}-{task_index:05d}\t{metric}\t{result_path}"
                )
                task_index += 1
        queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (metrics_root / "queues" / f"{kind}.cursor").write_text(
            "1\n", encoding="utf-8"
        )
    return result_roots


def _wait_for_metrics(config: dict) -> None:
    root = run_root(config)
    metrics_root = root / "metrics"
    expected_workers = metric_worker_count(config)
    expected_tasks = result_config_count(config) * metric_count(config)
    while True:
        states = len(list((metrics_root / "state").glob("*.complete")))
        completed = _line_count(metrics_root / "completed_tasks.tsv")
        failed = _line_count(metrics_root / "failed_tasks.tsv")
        print(
            f"[coordinator] metrics workers={states}/{expected_workers} "
            f"tasks={completed}/{expected_tasks} failed={failed}",
            flush=True,
        )
        if states >= expected_workers:
            break
        time.sleep(30)
    if failed or completed != expected_tasks:
        (root / "metrics.failed").touch()
        raise RuntimeError(
            f"metrics incomplete: completed={completed}, failed={failed}, "
            f"expected={expected_tasks}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = run_root(config)
    _wait_for_generation(config)
    _prepare_metric_queues(config)
    (root / "metrics.ready").touch()
    print("[coordinator] metric queues released", flush=True)
    _wait_for_metrics(config)

    metrics_root = root / "metrics"
    subprocess.run(
        [
            str(PYTHON),
            str(SUMMARY),
            "--input-txt",
            str(root / "leaf_folders.txt"),
            "--output-csv",
            str(metrics_root / "metric_summary.csv"),
            "--input-json-allowlist",
            str(root / "input_unique.txt"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(PYTHON),
            str(VERIFY_METRICS),
            "--baseline-list",
            str(root / "leaf_folders.txt"),
            "--output",
            str(metrics_root / "verification.json"),
            "--input-json-allowlist",
            str(root / "input_unique.txt"),
        ],
        check=True,
    )
    (root / "pipeline.complete").write_text(
        json.dumps(
            {
                "configs": result_config_count(config),
                "cases_per_config": len(read_unique_inputs(config)),
                "metric_tasks": result_config_count(config) * metric_count(config),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[coordinator] pipeline complete", flush=True)


if __name__ == "__main__":
    main()

