#!/usr/bin/env python3
"""Build queues for every missing metric in the shared S-head gallery."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from summarize_head_role_dose_control import METRICS


DEFAULT_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/head-role-dose-control-pilot/manifest.json"
)
DEFAULT_OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/"
    "full_metric_snapshots"
)
TASK_SPECS = (
    ("physics_iq_with_context", "cpu", ("physics_iq_with_context",)),
    ("physics_iq_without_context", "cpu", ("physics_iq_without_context",)),
    ("pmf_with_context", "cpu", ("pmf_with_context",)),
    ("pmf_without_context", "cpu", ("pmf_without_context",)),
    ("wmreward", "gpu", ("wmreward_surprise",)),
    (
        "vbench_subject_consistency",
        "gpu",
        ("vbench_subject_consistency",),
    ),
    (
        "vbench_background_consistency",
        "gpu",
        ("vbench_background_consistency",),
    ),
    (
        "vbench_temporal_flickering",
        "gpu",
        ("vbench_temporal_flickering",),
    ),
    (
        "vbench_motion_smoothness",
        "gpu",
        ("vbench_motion_smoothness",),
    ),
    ("vbench_dynamic_degree", "gpu", ("vbench_dynamic_degree",)),
    (
        "vbench_aesthetic_quality",
        "gpu",
        ("vbench_aesthetic_quality",),
    ),
    (
        "vbench_imaging_quality",
        "gpu",
        ("vbench_imaging_quality",),
    ),
    (
        "videophy2",
        "gpu",
        (
            "videophy2_sa",
            "videophy2_pc",
            "videophy2_joint_rate",
            "videophy2_pc_raw",
        ),
    ),
    ("cosmos_reason1", "gpu", ("cosmos_reason1",)),
)
METRIC_PATHS = {metric.name: metric.path for metric in METRICS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--expected-cases", type=int, default=20)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def media_path(gallery: Path, url: str) -> Path:
    prefix = "/head-role-dose-control-pilot/"
    if not url.startswith(prefix):
        raise ValueError(f"Unexpected gallery URL: {url}")
    return (gallery / url[len(prefix) :]).resolve()


def group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("kind"),
        record.get("model"),
        int(record.get("seed", -1)),
        record.get("subset_id"),
        int(record.get("start", -1)),
        int(record.get("end", -1)),
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    gallery = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in payload["records"]:
        if record.get("video"):
            groups[group_key(record)].append(record)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_base.expanduser().resolve() / f"final_{stamp}"
    for child in ("queues", "logs", "state", "task_summaries"):
        (output / child).mkdir(parents=True, exist_ok=False)

    rows: dict[str, list[tuple[str, str, Path]]] = {
        "cpu": [],
        "gpu": [],
    }
    incomplete = []
    complete_groups = []
    missing_by_metric: Counter[str] = Counter()
    skipped_complete = 0
    task_index = 0
    roots = []
    for key, records in sorted(groups.items(), key=lambda item: str(item[0])):
        if len(records) != args.expected_cases:
            incomplete.append({"key": list(key), "cases": len(records)})
            continue
        videos = [media_path(gallery, str(record["video"])) for record in records]
        if any(not video.is_file() for video in videos):
            incomplete.append(
                {
                    "key": list(key),
                    "cases": len(records),
                    "missing_videos": [
                        str(video) for video in videos if not video.is_file()
                    ],
                }
            )
            continue
        result_root = Path(os.path.commonpath([str(video) for video in videos]))
        if result_root.suffix.lower() == ".mp4":
            result_root = result_root.parent
        sidecars = []
        for video in videos:
            sidecar = video.with_suffix(".json")
            if not sidecar.is_file():
                break
            sidecars.append(json.loads(sidecar.read_text(encoding="utf-8")))
        if len(sidecars) != args.expected_cases:
            incomplete.append(
                {"key": list(key), "cases": len(sidecars), "missing_sidecars": True}
            )
            continue

        group_missing = []
        for metric, queue, outputs in TASK_SPECS:
            complete = all(
                all(
                    nested(sidecar, METRIC_PATHS[output_name]) is not None
                    for output_name in outputs
                )
                for sidecar in sidecars
            )
            if complete:
                skipped_complete += 1
                continue
            task_id = f"{queue}-{task_index:05d}-{metric}"
            rows[queue].append((task_id, metric, result_root))
            group_missing.append(metric)
            missing_by_metric[metric] += 1
            task_index += 1
        roots.append(result_root)
        complete_groups.append(
            {
                "key": list(key),
                "result_root": str(result_root),
                "missing_metrics": group_missing,
            }
        )

    if incomplete and not args.allow_incomplete:
        raise RuntimeError(
            f"Gallery has {len(incomplete)} incomplete 20-case metric groups"
        )
    for queue, queue_rows in rows.items():
        path = output / "queues" / f"{queue}.tsv"
        path.write_text(
            "".join(
                f"{task}\t{metric}\t{root}\n"
                for task, metric, root in queue_rows
            ),
            encoding="utf-8",
        )
        path.with_suffix(".cursor").write_text("1\n", encoding="utf-8")
        path.with_suffix(".lock").touch()
    (output / "completed_tasks.tsv").write_text("", encoding="utf-8")
    (output / "failed_tasks.tsv").write_text("", encoding="utf-8")
    (output / "leaf_folders.txt").write_text(
        "\n".join(str(root) for root in roots) + "\n",
        encoding="utf-8",
    )
    plan = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "expected_cases": args.expected_cases,
        "metric_outputs": {
            metric: list(outputs) for metric, _, outputs in TASK_SPECS
        },
        "result_groups": len(complete_groups),
        "incomplete_groups": incomplete,
        "task_counts": {queue: len(queue_rows) for queue, queue_rows in rows.items()},
        "missing_by_metric": dict(missing_by_metric),
        "skipped_complete_tasks": skipped_complete,
        "groups": complete_groups,
    }
    atomic_json(output / "plan.json", plan)
    args.output_base.mkdir(parents=True, exist_ok=True)
    (args.output_base / "latest").write_text(str(output) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
