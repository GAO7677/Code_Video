#!/usr/bin/env python3
"""Build simple path-only catalogs grouped by split/view/complexity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


DEFAULT_OUTPUT_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/unified_path_lists_v1")


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
            str(item.get("split_group", "")),
            str(item.get("view_group", "")),
            str(item.get("dataset", "")),
            str(item.get("sample_id", "")),
        ),
    )


def write_path_list(path_stem: Path, records: list[dict[str, Any]]) -> None:
    paths = [str(item["sample_dir"]) for item in records]
    write_txt(path_stem.with_suffix(".txt"), paths)
    write_json(path_stem.with_suffix(".json"), paths)


def write_grouped_lists(records: list[dict[str, Any]], output_root: Path) -> None:
    by_field_names = [
        "split_group",
        "view_group",
        "source_group",
        "object_count_group",
        "collision_bucket",
        "complexity_bucket",
    ]
    for field_name in by_field_names:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            groups[str(item.get(field_name, "unknown"))].append(item)
        for value, items in groups.items():
            stem = output_root / "by_field" / field_name / slugify(value)
            write_path_list(stem, items)

    combo_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        key = "__".join(
            [
                f"split_{slugify(str(item.get('split_group', 'unknown')))}",
                f"view_{slugify(str(item.get('view_group', 'unknown')))}",
                f"complexity_{slugify(str(item.get('complexity_bucket', 'unknown')))}",
            ]
        )
        combo_groups[key].append(item)
    for key, items in combo_groups.items():
        stem = output_root / "by_combo" / key
        write_path_list(stem, items)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    records = collect_records()
    write_path_list(output_root / "all_sample_dirs", records)
    write_grouped_lists(records, output_root)
    write_json(output_root / "summary.json", build_summary(records))

    print(json.dumps({"output_root": str(output_root), "total_records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
