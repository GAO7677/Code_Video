#!/usr/bin/env python3
"""Build simple path lists organized by view/split/complexity."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_unified_dataset_catalog import (
    build_summary,
    collect_genesis_raw_records,
    collect_movi_mytest_preview_records,
    collect_movi_raw_train_records,
    collect_stage1adapter_benchmark_records,
    collect_stage1adapter_window_records,
    ensure_dir,
    slugify,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/organized_view_split_complexity_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_txt(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def collect_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(collect_genesis_raw_records())
    records.extend(collect_movi_raw_train_records())
    records.extend(collect_movi_mytest_preview_records())
    records.extend(collect_stage1adapter_window_records())
    records.extend(collect_stage1adapter_benchmark_records())
    return sorted(
        records,
        key=lambda item: (
            str(item.get("view_group", "")),
            str(item.get("split_group", "")),
            str(item.get("complexity_bucket", "")),
            str(item.get("dataset", "")),
            str(item.get("sample_id", "")),
        ),
    )


def write_path_list(path_stem: Path, records: list[dict[str, Any]]) -> None:
    paths = [str(item["sample_dir"]) for item in records]
    write_txt(path_stem.with_suffix(".txt"), paths)
    write_json(path_stem.with_suffix(".json"), paths)


def normalize_complexity_bucket(value: str) -> str:
    text = str(value or "unknown")
    for prefix in ("pair_2", "few_3_4", "many_5plus"):
        if text in {f"{prefix}__env_only", f"{prefix}__mixed", f"{prefix}__obj_obj_only"}:
            return f"{prefix}__collision"
    return text


def split_parts(split_group: str) -> tuple[str, ...]:
    text = str(split_group)
    if text == "train":
        return ("train",)
    if text == "test":
        return ("test",)
    if text == "benchmark_fixed24":
        return ("benchmark", "fixed24")
    if text == "benchmark_validation100":
        return ("benchmark", "validation100")
    return ("other", slugify(text))


def write_lists(records: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    node_counts: dict[str, Counter[str]] = defaultdict(Counter)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        view = slugify(str(record.get("view_group", "unknown")))
        parts = split_parts(str(record.get("split_group", "unknown")))
        complexity = slugify(normalize_complexity_bucket(str(record.get("complexity_bucket", "unknown"))))
        grouped[(view, *parts, complexity)].append(record)
        node_counts["/".join((view, *parts))][complexity] += 1

    for key, items in sorted(grouped.items()):
        *dir_parts, complexity = key
        base_dir = output_root.joinpath(*dir_parts)
        write_path_list(base_dir / complexity, items)

    by_node_records: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        view = slugify(str(record.get("view_group", "unknown")))
        parts = split_parts(str(record.get("split_group", "unknown")))
        by_node_records[(view, *parts)].append(record)
    for dir_parts, items in sorted(by_node_records.items()):
        base_dir = output_root.joinpath(*dir_parts)
        write_path_list(base_dir / "_all_samples", items)

    return {
        "node_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(node_counts.items())
        }
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    records = collect_records()
    write_path_list(output_root / "all_sample_dirs", records)
    hierarchy_summary = write_lists(records, output_root)
    summary = build_summary(records)
    summary.update(hierarchy_summary)
    write_json(output_root / "summary.json", summary)

    print(json.dumps({"output_root": str(output_root), "total_records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
