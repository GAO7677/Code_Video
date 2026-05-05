#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from sample_bucket_labels import (
    COLLISION_PROFILE_ORDER,
    COUNT_BUCKET_ORDER,
    compute_derived_tags,
    find_sample_meta_path,
    load_sample_arrays,
)


DEFAULT_OUTPUT_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504")
RAW_TRAIN_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)
STAGE1_TRAIN_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/genesis/rigid"
)
STAGE1_TEST_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/test/genesis"
)
STAGE1_BENCHMARK_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/benchmark"
)
VAL_SUBSETS = ("fixed24", "validation100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild sum0504 path-only indices from Genesis raw/window samples.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, payload: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path: Path, lines: Iterable[str], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def scan_leaf_sample_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    sample_dirs: set[Path] = set()
    for meta_name in ("meta.json", "metadata.json"):
        for meta_path in root.glob("*/*/*/" + meta_name):
            sample_dirs.add(meta_path.parent)
    return sorted(sample_dirs)


def scan_test_or_val_sample_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    sample_dirs: set[Path] = set()
    for meta_name in ("meta.json", "metadata.json"):
        for meta_path in root.rglob(meta_name):
            sample_dirs.add(meta_path.parent)
    if not sample_dirs:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            try:
                sample_dirs.add(find_sample_meta_path(child).parent)
            except Exception:
                continue
    return sorted(sample_dirs)


def choose_split_roots() -> Dict[str, List[Path]]:
    return {
        "train": [RAW_TRAIN_ROOT, STAGE1_TRAIN_ROOT],
        "test": [STAGE1_TEST_ROOT],
        "val": [STAGE1_BENCHMARK_ROOT / name / "genesis" for name in VAL_SUBSETS],
    }


def classify_sample(sample_dir: Path) -> Tuple[str | None, Dict[str, Any] | None, str | None]:
    try:
        meta_path = find_sample_meta_path(sample_dir)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, f"meta_error:{type(exc).__name__}"

    count_bucket = str(metadata.get("object_count_bucket", "") or "")
    if count_bucket not in COUNT_BUCKET_ORDER:
        return None, metadata, f"bad_count_bucket:{count_bucket or 'missing'}"

    try:
        payload = load_sample_arrays(sample_dir)
        derived_tags = compute_derived_tags(
            metadata=payload["metadata"],
            events=payload["events"],
            linear_vel=payload["linear_vel"],
            visibility_mask=payload["visibility_mask"],
            com_pos=payload["com_pos"],
            bbox_xyxy=payload["bbox_xyxy"],
        )
    except Exception as exc:
        return None, metadata, f"derived_error:{type(exc).__name__}"

    collision_bucket = str(derived_tags.get("collision_profile_bucket", "") or "")
    if collision_bucket not in COLLISION_PROFILE_ORDER:
        return None, metadata, f"bad_collision_bucket:{collision_bucket or 'missing'}"

    return collision_bucket, metadata, None


def iter_split_samples(split: str, roots: List[Path]) -> List[Path]:
    sample_dirs: List[Path] = []
    for root in roots:
        if split == "train":
            if root == RAW_TRAIN_ROOT or root == STAGE1_TRAIN_ROOT:
                sample_dirs.extend(scan_leaf_sample_dirs(root))
            else:
                sample_dirs.extend(scan_test_or_val_sample_dirs(root))
        else:
            sample_dirs.extend(scan_test_or_val_sample_dirs(root))
    return sorted(set(sample_dirs))


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    roots_by_split = choose_split_roots()

    grouped: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    included_by_leaf: Dict[str, int] = {}
    excluded_breakdown: Counter[str] = Counter()
    included_samples = 0
    excluded_samples = 0

    for split, roots in roots_by_split.items():
        for sample_dir in iter_split_samples(split, roots):
            collision_bucket, metadata, error_key = classify_sample(sample_dir)
            if collision_bucket is None:
                dataset_name = "GenesisRigid"
                if metadata is not None:
                    dataset_name = str(metadata.get("dataset") or metadata.get("dataset_name") or "GenesisRigid")
                excluded_breakdown[f"{split}/unmapped/{dataset_name}:{error_key}"] += 1
                excluded_samples += 1
                continue

            key = (
                split,
                "rigid",
                str(metadata.get("object_count_bucket")),
                str(collision_bucket),
            )
            grouped[key].append(str(sample_dir))
            included_samples += 1

    if not args.dry_run and output_root.exists():
        for path in sorted(output_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        output_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        leaf_counts: Dict[str, int] = {}
        for count_bucket in COUNT_BUCKET_ORDER:
            for collision_bucket in COLLISION_PROFILE_ORDER:
                key = (split, "rigid", count_bucket, collision_bucket)
                lines = sorted(grouped.get(key, []))
                leaf_rel = Path(split) / "rigid" / count_bucket / collision_bucket
                write_lines(output_root / leaf_rel / "samples.txt", lines, args.dry_run)
                summary = {
                    "split": split,
                    "simulator_type": "rigid",
                    "object_count_bucket": count_bucket,
                    "collision_bucket": collision_bucket,
                    "num_samples": len(lines),
                }
                json_dump(output_root / leaf_rel / "summary.json", summary, args.dry_run)
                included_by_leaf[str(leaf_rel)] = len(lines)
                leaf_counts[f"rigid/{count_bucket}/{collision_bucket}"] = len(lines)
                split_counts[split][f"rigid/{count_bucket}/{collision_bucket}"] = len(lines)

        split_summary = {
            "split": split,
            "num_samples": int(sum(leaf_counts.values())),
            "leaf_counts": leaf_counts,
        }
        json_dump(output_root / split / "summary.json", split_summary, args.dry_run)

    root_summary = {
        "schema": "<split>/<simulator_type>/<object_count_bucket>/<collision_bucket>/samples.txt",
        "notes": [
            "Only path index files are created; raw data folders are not moved.",
            "train includes Genesis raw train/rigid plus stage1adapter/train genesis windows.",
            "test includes stage1adapter/test genesis windows; val includes benchmark/fixed24 and benchmark/validation100 genesis windows.",
            "Samples are included only if they can be stably mapped to count_01/count_02/count_03_04 and to one of the six collision buckets.",
        ],
        "included_samples": int(included_samples),
        "excluded_samples": int(excluded_samples),
        "included_by_leaf": included_by_leaf,
        "excluded_breakdown": dict(sorted(excluded_breakdown.items())),
    }
    json_dump(output_root / "summary.json", root_summary, args.dry_run)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "included_samples": included_samples,
                "excluded_samples": excluded_samples,
                "excluded_breakdown": dict(sorted(excluded_breakdown.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
