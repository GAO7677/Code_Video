#!/usr/bin/env python3
"""Rebuild data_summary/sum0504 from current raw/window sample directories."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GENESIS_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
)
GENESIS_TRAIN_RAW_ROOT = GENESIS_ROOT / "train"
STAGE1ADAPTER_ROOT = GENESIS_ROOT / "stage1adapter"
MOVI_ROOT = Path("/data/gaoya/dataset/kubric_tfds_movi-d")
MOVI_TRAIN_RAW_ROOT = MOVI_ROOT / "mytrain" / "movi_d_physics" / "train"
OUTPUT_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504")

ALLOWED_SPLITS = ("train", "val", "test")
ALLOWED_SIMULATORS = ("rigid",)
ALLOWED_COUNT_BUCKETS = ("count_01", "count_02", "count_03_04")
ALLOWED_COLLISION_BUCKETS = (
    "no_collision",
    "env_only",
    "obj_obj_only_c1",
    "obj_obj_only_c2plus",
    "mixed_c1",
    "mixed_c2plus",
)


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


def find_meta_path(sample_dir: Path) -> Path | None:
    for name in ("meta.json", "metadata.json"):
        path = sample_dir / name
        if path.exists():
            return path
    return None


def count_bucket_from_num_objects(num_objects: int | None) -> str | None:
    if num_objects is None:
        return None
    if int(num_objects) == 1:
        return "count_01"
    if int(num_objects) == 2:
        return "count_02"
    if int(num_objects) in (3, 4):
        return "count_03_04"
    return None


def collision_leaf(collision_type_bucket: str | None, collision_count_bucket: str | None) -> str | None:
    collision_type = str(collision_type_bucket or "").strip().lower()
    collision_count = str(collision_count_bucket or "").strip().lower()
    if collision_type == "none":
        return "no_collision"
    if collision_type == "env_only":
        return "env_only"
    if collision_type == "obj_obj_only":
        if collision_count == "c1":
            return "obj_obj_only_c1"
        if collision_count == "c2plus":
            return "obj_obj_only_c2plus"
        return None
    if collision_type == "mixed":
        if collision_count == "c1":
            return "mixed_c1"
        if collision_count == "c2plus":
            return "mixed_c2plus"
        return None
    return None


def source_raw_meta_from_window(meta: dict[str, Any]) -> dict[str, Any] | None:
    source_paths = meta.get("source_paths") or {}
    candidates: list[Path] = []
    for key in ("source_meta_json_path", "source_metadata_json_path"):
        value = str(source_paths.get(key, "")).strip()
        if value:
            candidates.append(Path(value))
    source_sample_dir = Path(str(source_paths.get("source_sample_dir", "")))
    if source_sample_dir.exists():
        for name in ("meta.json", "metadata.json"):
            candidates.append(source_sample_dir / name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            payload = load_json(candidate)
            if payload.get("collision_type_bucket") is not None or payload.get("object_count_bucket") is not None:
                return payload
    return None


def classify_sample(sample_dir: Path, meta: dict[str, Any]) -> tuple[str, str, str] | None:
    path_text = str(sample_dir)

    if "/stage1adapter/benchmark/" in path_text:
        split_name = "val"
    else:
        split_value = str(meta.get("split", "")).strip().lower()
        if split_value == "train":
            split_name = "train"
        elif split_value == "test":
            split_name = "test"
        else:
            return None

    simulator_type = "rigid"
    effective_meta = meta
    if "/stage1adapter/" in path_text:
        source_meta = source_raw_meta_from_window(meta)
        if source_meta is not None:
            effective_meta = source_meta

    count_bucket = str(effective_meta.get("object_count_bucket", "")).strip()
    if count_bucket not in ALLOWED_COUNT_BUCKETS:
        count_bucket = count_bucket_from_num_objects(effective_meta.get("num_objects"))
    if count_bucket not in ALLOWED_COUNT_BUCKETS:
        return None

    if "/stage1adapter/" in path_text:
        leaf = collision_leaf(
            effective_meta.get("collision_type_bucket"),
            effective_meta.get("collision_count_bucket"),
        )
        if leaf is None:
            adapter_window = meta.get("adapter_window") or {}
            leaf = collision_leaf(
                adapter_window.get("collision_bucket"),
                effective_meta.get("collision_count_bucket"),
            )
    else:
        leaf = collision_leaf(
            effective_meta.get("collision_type_bucket"),
            effective_meta.get("collision_count_bucket"),
        )
    if leaf not in ALLOWED_COLLISION_BUCKETS:
        return None

    return split_name, simulator_type, count_bucket, leaf


def gather_raw_samples() -> list[Path]:
    samples: list[Path] = []
    for root in (GENESIS_TRAIN_RAW_ROOT, MOVI_TRAIN_RAW_ROOT):
        if not root.exists():
            continue
        for name in ("meta.json", "metadata.json"):
            for meta_path in root.rglob(name):
                samples.append(meta_path.parent)
    return sorted(set(samples))


def gather_stage1adapter_samples() -> list[Path]:
    samples: list[Path] = []
    for branch in (STAGE1ADAPTER_ROOT / "train", STAGE1ADAPTER_ROOT / "test"):
        if branch.exists():
            for meta_path in branch.rglob("meta.json"):
                samples.append(meta_path.parent)
    benchmark_root = STAGE1ADAPTER_ROOT / "benchmark"
    if benchmark_root.exists():
        for split_dir in benchmark_root.iterdir():
            if not split_dir.is_dir():
                continue
            for dataset_dir in split_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                for sample_dir in dataset_dir.iterdir():
                    if sample_dir.is_dir() or sample_dir.is_symlink():
                        samples.append(sample_dir)
    return sorted(set(samples))


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    ensure_dir(OUTPUT_ROOT)

    included: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    included_by_leaf: Counter[str] = Counter()
    excluded_breakdown: Counter[str] = Counter()
    all_samples = gather_raw_samples() + gather_stage1adapter_samples()

    for sample_dir in sorted(set(all_samples)):
        meta_path = find_meta_path(sample_dir)
        if meta_path is None:
            excluded_breakdown["missing_meta"] += 1
            continue
        try:
            meta = load_json(meta_path)
        except Exception:
            excluded_breakdown["unreadable_meta"] += 1
            continue
        bucket = classify_sample(sample_dir, meta)
        dataset_name = str(meta.get("dataset", "unknown"))
        if bucket is None:
            split_name = str(meta.get("split", "unknown")).strip().lower() or "unknown"
            excluded_breakdown[f"{split_name}/unmapped/{dataset_name}"] += 1
            continue
        split_name, simulator_type, count_bucket, collision_bucket = bucket
        leaf_key = f"{split_name}/{simulator_type}/{count_bucket}/{collision_bucket}"
        included[(split_name, simulator_type, count_bucket, collision_bucket)].append(str(sample_dir))
        included_by_leaf[leaf_key] += 1

    split_counts: dict[str, Counter[str]] = {split_name: Counter() for split_name in ALLOWED_SPLITS}
    for key, paths in included.items():
        split_name, simulator_type, count_bucket, collision_bucket = key
        leaf_dir = OUTPUT_ROOT / split_name / simulator_type / count_bucket / collision_bucket
        sorted_paths = sorted(set(paths))
        write_txt(leaf_dir / "samples.txt", sorted_paths)
        write_json(
            leaf_dir / "summary.json",
            {
                "split": split_name,
                "simulator_type": simulator_type,
                "object_count_bucket": count_bucket,
                "collision_bucket": collision_bucket,
                "num_samples": len(sorted_paths),
            },
        )
        split_counts[split_name][f"{simulator_type}/{count_bucket}/{collision_bucket}"] = len(sorted_paths)

    for split_name, counter in split_counts.items():
        write_json(
            OUTPUT_ROOT / split_name / "summary.json",
            {
                "split": split_name,
                "num_samples": int(sum(counter.values())),
                "leaf_counts": dict(sorted(counter.items())),
            },
        )

    root_summary = {
        "schema": "<split>/<simulator_type>/<object_count_bucket>/<collision_bucket>/samples.txt",
        "notes": [
            "Only path index files are created; raw data folders are not moved.",
            "Samples are included only if they can be stably mapped to count_01/count_02/count_03_04 and to one of the six collision buckets.",
            "Window samples inherit collision type from adapter metadata and collision count from source raw metadata when available.",
        ],
        "included_samples": int(sum(included_by_leaf.values())),
        "excluded_samples": int(sum(excluded_breakdown.values())),
        "included_by_leaf": dict(sorted(included_by_leaf.items())),
        "excluded_breakdown": dict(sorted(excluded_breakdown.items())),
    }
    write_json(OUTPUT_ROOT / "summary.json", root_summary)
    write_txt(
        OUTPUT_ROOT / "README.md",
        [
            "# sum0504",
            "",
            "目录结构：`<split>/<simulator_type>/<object_count_bucket>/<collision_bucket>/samples.txt`",
            "",
            "当前仅整理可稳定映射到以下规则的样本：",
            "- split: `train / val / test`",
            "- simulator_type: `rigid`",
            "- object_count_bucket: `count_01 / count_02 / count_03_04`",
            "- collision_bucket: `no_collision / env_only / obj_obj_only_c1 / obj_obj_only_c2plus / mixed_c1 / mixed_c2plus`",
            "",
            "说明：",
            "- 不移动真实样本文件夹，仅记录绝对路径。",
            "- 每个叶子目录只保留 `samples.txt` 和 `summary.json`。",
            "- 根目录和 split 目录下提供汇总 `summary.json`。",
            "- 无法稳定映射到这套规则的样本不会被纳入，会记录在根目录 `summary.json` 的 `excluded_breakdown` 中。",
        ],
    )


if __name__ == "__main__":
    main()
