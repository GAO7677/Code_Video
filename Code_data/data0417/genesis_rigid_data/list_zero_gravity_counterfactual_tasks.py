#!/usr/bin/env python3
"""Collect zero-gravity rigid counterfactual cases that need regeneration."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CASE_NAME_TO_INDEX = {
    "case000_static_center": 0,
    "case001_static_left": 1,
    "case002_static_right": 2,
    "case003_static_highdrop": 3,
    "case005_entry_left": 5,
    "case006_entry_right": 6,
    "case007_entry_fast_center": 7,
    "case000_static_center_v2": 100,
    "case001_static_left_v2": 101,
    "case002_static_right_v2": 102,
    "case900_random_parabola": 900,
    "case901_high_drop": 901,
}

BUCKET_TO_TARGET_COUNT = {
    ("single_object_preview", "count_01"): 1,
    ("interaction_pair_plus_dynamic", "count_02"): 2,
}


def infer_target_count(scene_composition: str, count_bucket: str, metadata_path: Path) -> int:
    fixed = BUCKET_TO_TARGET_COUNT.get((scene_composition, count_bucket))
    if fixed is not None:
        return fixed
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        num_objects = int(meta.get("num_objects", 0))
        if num_objects > 0:
            return num_objects
    except Exception:
        pass
    raise ValueError(
        f"Unable to infer target count for scene={scene_composition} bucket={count_bucket} metadata={metadata_path}"
    )


def collect_tasks(output_root: Path) -> tuple[list[dict[str, object]], Counter]:
    tasks_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    stats = Counter()
    for scene_input_path in sorted(output_root.rglob("scene_input.json")):
        try:
            with scene_input_path.open("r", encoding="utf-8") as f:
                scene_input = json.load(f)
        except Exception:
            stats["read_error"] += 1
            continue

        gravity = scene_input.get("gravity")
        if not (isinstance(gravity, list) and len(gravity) == 3):
            continue
        try:
            gravity_z = float(gravity[2])
        except Exception:
            continue
        if abs(gravity_z) > 1e-9:
            continue

        cf = dict(scene_input.get("counterfactual") or {})
        parent_case_name = str(cf.get("parent_case_name") or "").strip()
        if not parent_case_name:
            stats["missing_parent_case_name"] += 1
            continue
        parent_case_index = CASE_NAME_TO_INDEX.get(parent_case_name)
        if parent_case_index is None:
            stats["unknown_parent_case_name"] += 1
            continue

        parts = scene_input_path.parts
        try:
            rigid_idx = parts.index("rigid")
            scene_composition = str(parts[rigid_idx + 1])
            count_bucket = str(parts[rigid_idx + 2])
        except Exception:
            stats["path_parse_error"] += 1
            continue

        metadata_path = scene_input_path.with_name("metadata.json")
        try:
            target_count = infer_target_count(scene_composition, count_bucket, metadata_path)
        except Exception:
            stats["target_count_error"] += 1
            continue

        object_id = str(scene_input.get("object_id") or "").strip()
        case_name = str(scene_input.get("case_name") or "").strip()
        key = (object_id, scene_composition, count_bucket, target_count, parent_case_index, parent_case_name)
        record = tasks_by_key.setdefault(
            key,
            {
                "object_id": object_id,
                "scene_composition": scene_composition,
                "count_bucket": count_bucket,
                "target_count": target_count,
                "parent_case_index": parent_case_index,
                "parent_case_name": parent_case_name,
                "example_case_name": case_name,
                "example_scene_input": str(scene_input_path),
                "counterfactual_kind": str(cf.get("kind") or ""),
            },
        )
        record.setdefault("paths", []).append(str(scene_input_path.parent))
        stats["zero_gravity_cases"] += 1

    tasks = list(tasks_by_key.values())
    tasks.sort(
        key=lambda item: (
            str(item["scene_composition"]),
            str(item["count_bucket"]),
            str(item["object_id"]),
            int(item["parent_case_index"]),
        )
    )
    stats["unique_regen_tasks"] = len(tasks)
    return tasks, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="List zero-gravity counterfactual scenes that should be regenerated.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--tasks_tsv", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    tasks_tsv = args.tasks_tsv.resolve()
    summary_json = args.summary_json.resolve()
    tasks_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    tasks, stats = collect_tasks(output_root)

    with tasks_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "object_id",
                "scene_composition",
                "count_bucket",
                "target_count",
                "parent_case_index",
                "parent_case_name",
                "example_case_name",
                "counterfactual_kind",
                "example_scene_input",
            ]
        )
        for item in tasks:
            writer.writerow(
                [
                    item["object_id"],
                    item["scene_composition"],
                    item["count_bucket"],
                    item["target_count"],
                    item["parent_case_index"],
                    item["parent_case_name"],
                    item["example_case_name"],
                    item["counterfactual_kind"],
                    item["example_scene_input"],
                ]
            )

    summary = {
        "output_root": str(output_root),
        "stats": dict(stats),
        "tasks_preview": tasks[:20],
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"zero_gravity_cases={stats['zero_gravity_cases']}")
    print(f"unique_regen_tasks={stats['unique_regen_tasks']}")
    print(f"tasks_tsv={tasks_tsv}")
    print(f"summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
