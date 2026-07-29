#!/usr/bin/env python3
"""Freeze missing physics-metric tasks for currently complete S-head runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_head_role_dose_control_coordinator import (
    _baseline_root,
    _common_video_parent,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/"
    "physics_metric_snapshots"
)
PREWARM_PLAN = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/"
    "metric_prewarm_current/plan.json"
)
BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/pass1"
)
EXPERIMENT_ROOTS = {
    "s_feature": Path(
        "/data/gaoya/agent-data/outputs/wan_dit_s_feature_split/pilot"
    ),
    "s_feature_union": Path(
        "/data/gaoya/agent-data/outputs/wan_dit_s_feature_union/pilot"
    ),
    "s_feature_phased": Path(
        "/data/gaoya/agent-data/outputs/wan_dit_s_feature_phased/pilot"
    ),
    "s_depth": Path(
        "/data/gaoya/agent-data/outputs/wan_dit_head_role_depth_strata/s_only"
    ),
}
CPU_METRICS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
)
HEAVY_METRICS = ("videophy2", "cosmos_reason1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--prewarm-plan", type=Path, default=PREWARM_PLAN)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_queue(path: Path, rows: list[tuple[str, str, Path]]) -> None:
    path.write_text(
        "".join(f"{task}\t{metric}\t{root}\n" for task, metric, root in rows),
        encoding="utf-8",
    )
    path.with_suffix(".cursor").write_text("1\n", encoding="utf-8")
    path.with_suffix(".lock").touch()


def complete_roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for family, experiment_root in EXPERIMENT_ROOTS.items():
        for state_path in sorted((experiment_root / "state").glob("*.json")):
            record = read_json(state_path)
            if record.get("status") != "complete":
                continue
            videos = record.get("videos", {})
            if len(videos) != 20:
                raise RuntimeError(f"Incomplete video map: {state_path}")
            roots.append((family, _common_video_parent(videos)))
    for seed in (851, 3278):
        for model in ("physrvg", "wan_lora", "xssc"):
            roots.append(("baseline", _baseline_root(BASELINE_ROOT, model, seed)))
    return roots


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_base.expanduser().resolve() / timestamp
    for path in (
        output / "queues",
        output / "logs",
        output / "state",
        output / "task_summaries",
    ):
        path.mkdir(parents=True, exist_ok=False)

    prior_roots = {
        str(Path(item).expanduser().resolve())
        for item in read_json(args.prewarm_plan.expanduser().resolve())["roots"]
    }
    candidates = complete_roots()
    root_to_family: dict[Path, str] = {}
    for family, root in candidates:
        resolved = root.expanduser().resolve()
        if resolved in root_to_family:
            raise RuntimeError(f"Duplicate result root: {resolved}")
        root_to_family[resolved] = family

    selected = [
        (family, root)
        for root, family in root_to_family.items()
        if str(root) not in prior_roots
    ]
    cpu_rows: list[tuple[str, str, Path]] = []
    heavy_rows: list[tuple[str, str, Path]] = []
    for index, (_, root) in enumerate(selected):
        for metric in CPU_METRICS:
            cpu_rows.append((f"cpu-{index:04d}-{metric}", metric, root))
        for metric in HEAVY_METRICS:
            heavy_rows.append((f"heavy-{index:04d}-{metric}", metric, root))

    write_queue(output / "queues" / "cpu.tsv", cpu_rows)
    write_queue(output / "queues" / "heavy.tsv", heavy_rows)
    (output / "completed_tasks.tsv").write_text("", encoding="utf-8")
    (output / "failed_tasks.tsv").write_text("", encoding="utf-8")
    plan = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "result_roots_seen": len(candidates),
        "result_roots_excluded_as_prewarmed": len(candidates) - len(selected),
        "result_roots_selected": len(selected),
        "selected_by_family": dict(Counter(family for family, _ in selected)),
        "metrics": {
            "cpu": list(CPU_METRICS),
            "heavy": list(HEAVY_METRICS),
        },
        "task_counts": {"cpu": len(cpu_rows), "heavy": len(heavy_rows)},
        "roots": [
            {"family": family, "result_root": str(root)}
            for family, root in selected
        ],
    }
    (output / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_base.mkdir(parents=True, exist_ok=True)
    (args.output_base / "latest").write_text(str(output) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
