#!/usr/bin/env python3
"""Build unified json/txt catalogs for Genesis raw, stage1adapter, and MOVI-D data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GENESIS_RAW_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train"
)
STAGE1ADAPTER_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter"
)
MOVI_ROOT = Path("/data/gaoya/dataset/kubric_tfds_movi-d")
MOVI_RAW_TRAIN_ROOT = MOVI_ROOT / "mytrain" / "movi_d_physics" / "train"
MOVI_MYTEST_ROOT = MOVI_ROOT / "mytest"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/dataset_catalog_unified_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_txt(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def slugify(text: str) -> str:
    parts: list[str] = []
    for ch in str(text).strip().lower():
        if ch.isalnum():
            parts.append(ch)
        else:
            parts.append("_")
    slug = "".join(parts)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def normalize_dataset_name(name: str) -> str:
    text = str(name).strip().lower()
    if text in {"genesisrigid", "genesis", "genesis_rigid"}:
        return "genesis"
    if text in {"movi-d", "movi_d", "movi"}:
        return "movi-d"
    if text in {"openvid"}:
        return "openvid"
    return text or "unknown"


def parse_num_objects_from_caption(text: str) -> int | None:
    match = re.search(r"(\d+)\s+object\(s\)", str(text), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"with\s+(\d+)\s+rigid\s+object\(s\)", str(text), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def object_count_group(num_objects: int | None) -> str:
    if num_objects is None or int(num_objects) <= 0:
        return "unknown"
    if int(num_objects) == 1:
        return "single_1"
    if int(num_objects) == 2:
        return "pair_2"
    if int(num_objects) in {3, 4}:
        return "few_3_4"
    return "many_5plus"


def collision_rank(bucket: str) -> int:
    order = {
        "none": 0,
        "env_only": 1,
        "obj_obj_only": 2,
        "mixed": 3,
        "unknown": 4,
    }
    return order.get(str(bucket), 4)


def derive_complexity_bucket(num_objects: int | None, collision_bucket: str) -> str:
    group = object_count_group(num_objects)
    bucket = str(collision_bucket or "unknown")
    if bucket not in {"none", "env_only", "obj_obj_only", "mixed"}:
        bucket = "unknown"
    return f"{group}__{bucket}"


def make_record(
    *,
    dataset: str,
    split_group: str,
    view_group: str,
    sample_dir: Path,
    meta_path: Path,
    sample_id: str,
    source_group: str,
    caption: str,
    num_objects: int | None,
    collision_bucket: str,
    motion_label: str,
    has_physics_labels: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "dataset": normalize_dataset_name(dataset),
        "split_group": str(split_group),
        "view_group": str(view_group),
        "sample_dir": str(sample_dir),
        "meta_path": str(meta_path),
        "sample_id": str(sample_id),
        "source_group": str(source_group),
        "caption": str(caption or ""),
        "num_objects": None if num_objects is None else int(num_objects),
        "object_count_group": object_count_group(num_objects),
        "collision_bucket": str(collision_bucket or "unknown"),
        "motion_label": str(motion_label or "unknown"),
        "complexity_bucket": derive_complexity_bucket(num_objects, collision_bucket),
        "complexity_rank": collision_rank(collision_bucket),
        "has_physics_labels": bool(has_physics_labels),
    }
    if extra:
        record.update(extra)
    return record


def collect_genesis_raw_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for meta_path in sorted(GENESIS_RAW_ROOT.rglob("metadata.json")):
        if "invalid_by_qa" in meta_path.parts:
            continue
        meta = load_json(meta_path)
        sample_dir = meta_path.parent
        records.append(
            make_record(
                dataset="genesis",
                split_group=str(meta.get("split", "train") or "train"),
                view_group="raw",
                sample_dir=sample_dir,
                meta_path=meta_path,
                sample_id=str(meta.get("scene_id", sample_dir.name)),
                source_group="genesis_raw",
                caption=str(meta.get("prompt", "")),
                num_objects=int(meta["num_objects"]) if meta.get("num_objects") is not None else None,
                collision_bucket=str(meta.get("collision_type_bucket", "unknown")),
                motion_label=str(meta.get("motion_label", meta.get("motion_category", "unknown"))),
                has_physics_labels=True,
                extra={
                    "family": str(meta.get("family", "")),
                    "scene_composition": str(meta.get("scene_composition", "")),
                    "interaction_pattern": str(meta.get("interaction_pattern", "")),
                    "object_count_bucket": str(meta.get("object_count_bucket", "")),
                    "obj_env_event_count": int(meta.get("obj_env_event_count", 0) or 0),
                    "obj_obj_event_count": int(meta.get("obj_obj_event_count", 0) or 0),
                },
            )
        )
    return records


def collect_movi_raw_train_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for meta_path in sorted(MOVI_RAW_TRAIN_ROOT.rglob("metadata.json")):
        meta = load_json(meta_path)
        sample_dir = meta_path.parent
        records.append(
            make_record(
                dataset="movi-d",
                split_group=str(meta.get("split", "train") or "train"),
                view_group="raw",
                sample_dir=sample_dir,
                meta_path=meta_path,
                sample_id=str(meta.get("scene_id", sample_dir.name)),
                source_group="movi_raw_train",
                caption=str(meta.get("prompt", "")),
                num_objects=int(meta["num_objects"]) if meta.get("num_objects") is not None else None,
                collision_bucket=str(meta.get("collision_type_bucket", "unknown")),
                motion_label="unknown",
                has_physics_labels=True,
                extra={
                    "background": str(meta.get("background", "")),
                    "object_count_bucket": str(meta.get("object_count_bucket", "")),
                    "obj_env_event_count": int(meta.get("obj_env_event_count", 0) or 0),
                    "obj_obj_event_count": int(meta.get("obj_obj_event_count", 0) or 0),
                },
            )
        )
    return records


def collect_movi_mytest_preview_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for meta_path in sorted(MOVI_MYTEST_ROOT.glob("*/meta.json")):
        meta = load_json(meta_path)
        sample_dir = meta_path.parent
        caption = str(meta.get("caption", ""))
        num_objects = parse_num_objects_from_caption(caption)
        records.append(
            make_record(
                dataset="movi-d",
                split_group="test",
                view_group="raw",
                sample_dir=sample_dir,
                meta_path=meta_path,
                sample_id=str(meta.get("sample_id", sample_dir.name)),
                source_group="movi_raw_test_preview",
                caption=caption,
                num_objects=num_objects,
                collision_bucket="unknown",
                motion_label="unknown",
                has_physics_labels=False,
                extra={
                    "preview_only": True,
                    "fps": int(meta.get("fps", 0) or 0),
                    "context_frames": int(meta.get("context_frames", 0) or 0),
                    "future_frames": int(meta.get("future_frames", 0) or 0),
                    "raw_frames": int(meta.get("raw_frames", 0) or 0),
                },
            )
        )
    return records


def collect_stage1adapter_window_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split_dir_name in ("train", "test"):
        split_root = STAGE1ADAPTER_ROOT / split_dir_name
        if not split_root.exists():
            continue
        for meta_path in sorted(split_root.rglob("meta.json")):
            meta = load_json(meta_path)
            sample_dir = meta_path.parent
            sample_id = str(meta.get("sample_id", sample_dir.name))
            dataset = normalize_dataset_name(str(meta.get("dataset", "")))
            pair_meta_path = sample_dir / "pair_meta.json"
            pair_meta = load_json(pair_meta_path) if pair_meta_path.exists() else {}
            adapter_window = meta.get("adapter_window") or {}
            num_objects = None
            if isinstance(pair_meta.get("objects"), list):
                num_objects = len(pair_meta.get("objects") or [])
            if num_objects in (None, 0):
                num_objects = parse_num_objects_from_caption(str(meta.get("caption", "")))
            records.append(
                make_record(
                    dataset=dataset,
                    split_group=split_dir_name,
                    view_group="window",
                    sample_dir=sample_dir,
                    meta_path=meta_path,
                    sample_id=sample_id,
                    source_group=f"stage1adapter_{split_dir_name}",
                    caption=str(meta.get("caption", "")),
                    num_objects=num_objects,
                    collision_bucket=str(adapter_window.get("collision_bucket", "unknown")),
                    motion_label=str(adapter_window.get("motion_complexity", "unknown")),
                    has_physics_labels=True,
                    extra={
                        "context_frames": int(meta.get("context_frames", 0) or 0),
                        "future_frames": int(meta.get("future_frames", 0) or 0),
                        "raw_frames": int(meta.get("raw_frames", 0) or 0),
                        "segment_kind": str(adapter_window.get("segment_kind", "")),
                    },
                )
            )
    return records


def collect_stage1adapter_benchmark_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    benchmark_root = STAGE1ADAPTER_ROOT / "benchmark"
    if not benchmark_root.exists():
        return records
    for group_name in ("fixed24", "validation100"):
        group_root = benchmark_root / group_name
        if not group_root.exists():
            continue
        for dataset_dir in sorted(group_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for sample_dir in sorted(dataset_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                meta_path = sample_dir / "meta.json"
                if not meta_path.exists():
                    continue
                meta = load_json(meta_path)
                caption = str(meta.get("caption", ""))
                num_objects = parse_num_objects_from_caption(caption)
                dataset = normalize_dataset_name(str(meta.get("dataset", "")))
                records.append(
                    make_record(
                        dataset=dataset,
                        split_group=f"benchmark_{group_name}",
                        view_group="window",
                        sample_dir=sample_dir,
                        meta_path=meta_path,
                        sample_id=str(meta.get("sample_id", sample_dir.name)),
                        source_group=f"stage1adapter_benchmark_{group_name}",
                        caption=caption,
                        num_objects=num_objects,
                        collision_bucket="unknown",
                        motion_label="unknown",
                        has_physics_labels=False,
                        extra={
                            "context_frames": int(meta.get("context_frames", 0) or 0),
                            "future_frames": int(meta.get("future_frames", 0) or 0),
                            "raw_frames": int(meta.get("raw_frames", 0) or 0),
                        },
                    )
                )
    return records


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "dataset",
        "split_group",
        "view_group",
        "source_group",
        "object_count_group",
        "collision_bucket",
        "complexity_bucket",
        "has_physics_labels",
    ]
    summary: dict[str, Any] = {"total_records": len(records), "counts": {}}
    for field in fields:
        counter = Counter(str(record.get(field, "")) for record in records)
        summary["counts"][field] = dict(sorted(counter.items()))
    combo_counter = Counter(
        (
            str(record.get("split_group", "")),
            str(record.get("view_group", "")),
            str(record.get("complexity_bucket", "")),
        )
        for record in records
    )
    summary["counts"]["split_view_complexity"] = {
        "__".join(key): value for key, value in sorted(combo_counter.items())
    }
    return summary


def write_group_manifests(records: list[dict[str, Any]], output_root: Path) -> None:
    fields = [
        "dataset",
        "split_group",
        "view_group",
        "source_group",
        "object_count_group",
        "collision_bucket",
        "complexity_bucket",
    ]
    for field in fields:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(field, "unknown"))].append(record)
        for value, items in groups.items():
            slug = slugify(value)
            base = output_root / "by_field" / field / slug
            write_json(base.with_suffix(".json"), items)
            write_txt(base.with_suffix(".txt"), [str(item["sample_dir"]) for item in items])

    combo_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = "__".join(
            [
                f"split_{slugify(str(record.get('split_group', 'unknown')))}",
                f"view_{slugify(str(record.get('view_group', 'unknown')))}",
                f"complexity_{slugify(str(record.get('complexity_bucket', 'unknown')))}",
            ]
        )
        combo_groups[key].append(record)
    for key, items in combo_groups.items():
        base = output_root / "by_combo" / key
        write_json(base.with_suffix(".json"), items)
        write_txt(base.with_suffix(".txt"), [str(item["sample_dir"]) for item in items])


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    records: list[dict[str, Any]] = []
    records.extend(collect_genesis_raw_records())
    records.extend(collect_movi_raw_train_records())
    records.extend(collect_movi_mytest_preview_records())
    records.extend(collect_stage1adapter_window_records())
    records.extend(collect_stage1adapter_benchmark_records())
    records = sorted(
        records,
        key=lambda item: (
            str(item.get("split_group", "")),
            str(item.get("view_group", "")),
            str(item.get("dataset", "")),
            str(item.get("sample_id", "")),
        ),
    )

    write_json(output_root / "all_records.json", records)
    write_txt(output_root / "all_sample_dirs.txt", [str(item["sample_dir"]) for item in records])
    write_txt(
        output_root / "all_meta_paths.txt",
        [str(item["meta_path"]) for item in records],
    )
    ensure_dir(output_root)
    (output_root / "all_records.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    write_json(output_root / "summary.json", build_summary(records))
    write_group_manifests(records, output_root)

    print(json.dumps({"output_root": str(output_root), "total_records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
