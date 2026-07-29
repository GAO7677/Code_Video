#!/usr/bin/env python3
"""Coordinate pilot generation, metric queues, summaries, and completion."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_head_role_dose_control_pilot_worker import (
    _input_cases,
    _load_config,
    _tasks,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
SUMMARY = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py"
)
VERIFY = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "train0705_kubric_no_gt_box/verify_bench_physiq_metrics.py"
)
DOSE_SUMMARY = SCRIPT_DIR / "summarize_head_role_dose_control.py"
DOSE_REPORT = SCRIPT_DIR / "render_head_role_dose_control_report.py"
DOSE_GALLERY = SCRIPT_DIR / "build_head_role_dose_control_gallery.py"
CASE_GALLERY = SCRIPT_DIR / "build_head_role_dose_control_case_gallery.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _states(root: Path) -> tuple[Counter[str], list[dict[str, Any]]]:
    records = []
    for path in sorted((root / "state").glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            records.append({"status": "invalid", "path": str(path)})
    return Counter(str(record.get("status", "invalid")) for record in records), records


def _common_video_parent(videos: dict[str, str]) -> Path:
    paths = [str(Path(path).expanduser().resolve()) for path in videos.values()]
    if not paths:
        raise ValueError("Task state has no videos")
    parent = Path(os.path.commonpath(paths))
    if parent.suffix.lower() == ".mp4":
        parent = parent.parent
    return parent


def _baseline_root(base: Path, model: str, seed: int) -> Path:
    path = (
        base
        / model
        / f"seed-{seed:06d}"
        / "generated"
        / model
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _prepare_metric_inputs(
    *,
    config: dict[str, Any],
    root: Path,
    records: list[dict[str, Any]],
    expected_tasks: int,
) -> tuple[list[Path], int]:
    complete = [record for record in records if record.get("status") == "complete"]
    if len(complete) != expected_tasks:
        raise RuntimeError(f"Expected {expected_tasks} completed tasks, got {len(complete)}")
    cases = _input_cases(Path(config["input_list"]).expanduser().resolve())
    manifest_records = []
    roots: list[Path] = []
    for record in complete:
        videos = record.get("videos", {})
        if set(videos) != cases:
            raise RuntimeError(f"Incomplete video map in {record.get('task_id')}")
        result_root = _common_video_parent(videos)
        roots.append(result_root)
        manifest_records.append(
            {
                **record,
                "kind": "ablation",
                "result_root": str(result_root),
            }
        )
    baseline_base = Path(config["metrics"]["baseline_root"]).expanduser().resolve()
    for seed in config["seeds"]:
        for model in config["models"]:
            result_root = _baseline_root(baseline_base, str(model), int(seed))
            roots.append(result_root)
            manifest_records.append(
                {
                    "kind": "baseline",
                    "model": str(model),
                    "seed": int(seed),
                    "variant": "baseline",
                    "result_root": str(result_root),
                }
            )
    if len(roots) != len(set(roots)):
        raise RuntimeError("Metric result roots are not unique")
    leaf_list = root / "leaf_folders.txt"
    leaf_list.write_text(
        "\n".join(str(path) for path in roots) + "\n",
        encoding="utf-8",
    )
    _atomic_json(
        root / "generation_manifest.json",
        {
            "schema_version": 1,
            "ablation_roots": expected_tasks,
            "baseline_roots": len(config["models"]) * len(config["seeds"]),
            "cases_per_root": len(cases),
            "entries": manifest_records,
        },
    )
    return roots, len(cases)


def _prepare_metric_queues(
    config: dict[str, Any],
    root: Path,
    roots: list[Path],
) -> int:
    metric_root = root / "metrics"
    queues = metric_root / "queues"
    for path in (queues, metric_root / "logs", metric_root / "state", metric_root / "task_summaries"):
        path.mkdir(parents=True, exist_ok=True)
    (metric_root / "completed_tasks.tsv").write_text("", encoding="utf-8")
    (metric_root / "failed_tasks.tsv").write_text("", encoding="utf-8")
    total = 0
    for kind, metrics in config["metrics"]["groups"].items():
        lines = []
        task_index = 0
        for result_root in roots:
            for metric in metrics:
                lines.append(
                    f"{kind}-{task_index:05d}\t{metric}\t{result_root}"
                )
                task_index += 1
        (queues / f"{kind}.tsv").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        (queues / f"{kind}.cursor").write_text("1\n", encoding="utf-8")
        (queues / f"{kind}.lock").touch()
        total += len(lines)
    return total


def _progress(
    *,
    root: Path,
    phase: str,
    states: Counter[str],
    expected: int,
    extra: dict[str, Any] | None = None,
) -> None:
    disk = shutil.disk_usage(root.parent)
    payload = {
        "phase": phase,
        "expected_generation_tasks": expected,
        "generation_states": dict(states),
        "output_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        ),
        "data_disk_free_bytes": disk.free,
        "updated_at_unix": time.time(),
        **(extra or {}),
    }
    _atomic_json(root / "progress.json", payload)
    print(f"[dose-coordinator] {json.dumps(payload, sort_keys=True)}", flush=True)


def _stop_incremental_metrics(root: Path, poll_seconds: int) -> None:
    incremental = root / "incremental_metrics_live"
    plan_path = incremental / "plan.json"
    if not (incremental / "started").is_file() or not plan_path.is_file():
        return
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = int(plan["expected_workers"])
    (incremental / "stop").touch()
    deadline = time.time() + 1800
    while time.time() < deadline:
        complete = len(list((incremental / "state").glob("*.complete")))
        if complete >= expected:
            break
        print(
            f"[dose-coordinator] draining incremental metrics "
            f"{complete}/{expected}",
            flush=True,
        )
        time.sleep(min(poll_seconds, 20))
    _atomic_json(
        incremental / "drain_status.json",
        {
            "expected_workers": expected,
            "workers_complete": len(
                list((incremental / "state").glob("*.complete"))
            ),
            "drained_at_unix": time.time(),
        },
    )


def _refresh_case_gallery(config_path: Path) -> None:
    result = subprocess.run(
        [
            str(PYTHON),
            str(CASE_GALLERY),
            "--config",
            str(config_path),
        ],
        check=False,
    )
    if result.returncode:
        print(
            f"[dose-coordinator] case gallery refresh failed: {result.returncode}",
            flush=True,
        )


def _wait_for_metric_dependency(
    config: dict[str, Any],
    root: Path,
    expected: int,
    poll_seconds: int,
) -> None:
    value = config.get("metrics", {}).get("defer_until_file")
    if not value:
        return
    dependency = Path(value).expanduser().resolve()
    while not dependency.is_file():
        counts, _ = _states(root)
        _progress(
            root=root,
            phase="waiting_for_priority_generation",
            states=counts,
            expected=expected,
            extra={"waiting_for_file": str(dependency)},
        )
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config, root, _, _, subset_ids = _load_config(config_path)
    expected = len(_tasks(config, subset_ids))
    last_gallery_complete = -1
    while True:
        counts, records = _states(root)
        _progress(
            root=root,
            phase="generation",
            states=counts,
            expected=expected,
        )
        if counts["complete"] != last_gallery_complete:
            _refresh_case_gallery(config_path)
            last_gallery_complete = counts["complete"]
        if counts["complete"] == expected:
            break
        if counts["failed"] and not counts["running"]:
            attempts = [
                int(record.get("attempt", 0))
                for record in records
                if record.get("status") == "failed"
            ]
            if attempts and min(attempts) >= int(
                config["execution"]["max_attempts_per_task"]
            ):
                (root / "generation.failed").touch()
                raise RuntimeError("Generation has exhausted retry attempts")
        time.sleep(int(args.poll_seconds))

    _stop_incremental_metrics(root, int(args.poll_seconds))
    (root / "generation.complete").touch()
    _wait_for_metric_dependency(
        config,
        root,
        expected,
        int(args.poll_seconds),
    )
    ready = root / "metrics.ready"
    if not ready.is_file():
        roots, cases_per_root = _prepare_metric_inputs(
            config=config,
            root=root,
            records=records,
            expected_tasks=expected,
        )
        metric_tasks = _prepare_metric_queues(config, root, roots)
        _atomic_json(
            root / "metric_plan.json",
            {
                "result_roots": len(roots),
                "cases_per_root": cases_per_root,
                "metric_tasks": metric_tasks,
                "groups": config["metrics"]["groups"],
            },
        )
        ready.touch()
    metric_worker_count = (
        len(config["execution"]["gpus"])
        * sum(int(value) for value in config["metrics"]["workers_per_gpu"].values())
    )
    metric_root = root / "metrics"
    while True:
        worker_done = len(list((metric_root / "state").glob("*.complete")))
        completed = sum(
            1
            for line in (metric_root / "completed_tasks.tsv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        failed = sum(
            1
            for line in (metric_root / "failed_tasks.tsv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        _progress(
            root=root,
            phase="metrics",
            states=Counter({"complete": expected}),
            expected=expected,
            extra={
                "metric_workers_complete": worker_done,
                "metric_workers_expected": metric_worker_count,
                "metric_tasks_complete": completed,
                "metric_tasks_failed": failed,
            },
        )
        if worker_done == metric_worker_count:
            break
        time.sleep(int(args.poll_seconds))

    subprocess.run(
        [
            str(PYTHON),
            str(SUMMARY),
            "--input-txt",
            str(root / "leaf_folders.txt"),
            "--output-csv",
            str(metric_root / "metric_summary.csv"),
            "--input-json-allowlist",
            str(config["input_list"]),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(PYTHON),
            str(VERIFY),
            "--baseline-list",
            str(root / "leaf_folders.txt"),
            "--output",
            str(metric_root / "verification.json"),
            "--input-json-allowlist",
            str(config["input_list"]),
        ],
        check=True,
    )
    failed = [
        line
        for line in (metric_root / "failed_tasks.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if failed:
        (root / "metrics.failed").touch()
        raise RuntimeError(f"{len(failed)} metric tasks failed")
    subprocess.run(
        [
            str(PYTHON),
            str(DOSE_SUMMARY),
            "--root",
            str(root),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(PYTHON),
            str(DOSE_REPORT),
            "--root",
            str(root),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(PYTHON),
            str(DOSE_GALLERY),
            "--root",
            str(root),
        ],
        check=True,
    )
    (root / "pipeline.complete").touch()
    _progress(
        root=root,
        phase="complete",
        states=Counter({"complete": expected}),
        expected=expected,
    )


if __name__ == "__main__":
    main()
