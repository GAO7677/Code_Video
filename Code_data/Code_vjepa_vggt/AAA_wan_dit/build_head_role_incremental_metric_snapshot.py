#!/usr/bin/env python3
"""Freeze metric queues for generation tasks that are already complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_head_role_dose_control_coordinator import (
    _baseline_root,
    _common_video_parent,
)
from run_head_role_dose_control_pilot_worker import _input_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    output = root / "incremental_metrics_live"
    if (output / "started").exists():
        raise RuntimeError(f"Incremental snapshot already started: {output}")
    cases = _input_cases(Path(config["input_list"]).expanduser().resolve())
    roots: list[Path] = []
    entries: list[dict[str, Any]] = []
    for state_path in sorted((root / "state").glob("*.json")):
        record = json.loads(state_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete":
            continue
        videos = record.get("videos", {})
        if set(videos) != cases:
            raise RuntimeError(f"Incomplete video map: {state_path}")
        result_root = _common_video_parent(videos)
        roots.append(result_root)
        entries.append(
            {
                "kind": "ablation",
                "task_id": record["task_id"],
                "result_root": str(result_root),
            }
        )
    baseline_base = Path(config["metrics"]["baseline_root"]).expanduser().resolve()
    for seed in config["seeds"]:
        for model in config["models"]:
            result_root = _baseline_root(baseline_base, str(model), int(seed))
            roots.append(result_root)
            entries.append(
                {
                    "kind": "baseline",
                    "model": model,
                    "seed": int(seed),
                    "result_root": str(result_root),
                }
            )
    if len(roots) != len(set(roots)):
        raise RuntimeError("Incremental metric roots are not unique")
    for path in (
        output / "queues",
        output / "logs",
        output / "state",
        output / "task_summaries",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (output / "completed_tasks.tsv").write_text("", encoding="utf-8")
    (output / "failed_tasks.tsv").write_text("", encoding="utf-8")
    task_counts = {}
    for kind in ("cpu", "gpu_common"):
        lines = []
        for index, result_root in enumerate(roots):
            for metric in config["metrics"]["groups"][kind]:
                lines.append(f"{kind}-{index:05d}-{metric}\t{metric}\t{result_root}")
        (output / "queues" / f"{kind}.tsv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        (output / "queues" / f"{kind}.cursor").write_text("1\n", encoding="utf-8")
        (output / "queues" / f"{kind}.lock").touch()
        task_counts[kind] = len(lines)
    atomic_json(
        output / "plan.json",
        {
            "schema_version": 1,
            "generation_tasks_snapshotted": len(entries) - 6,
            "baseline_roots": 6,
            "result_roots": len(roots),
            "cases_per_root": len(cases),
            "task_counts": task_counts,
            "expected_workers": int(args.workers),
            "entries": entries,
        },
    )
    print(
        f"[incremental-snapshot] roots={len(roots)} "
        f"cpu={task_counts['cpu']} gpu_common={task_counts['gpu_common']}"
    )


if __name__ == "__main__":
    main()
