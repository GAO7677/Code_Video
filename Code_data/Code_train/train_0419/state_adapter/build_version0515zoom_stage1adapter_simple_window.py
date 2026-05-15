#!/usr/bin/env python3
"""Build strict simple-motion Genesis stage1adapter windows directly from version0515zoom_genesis_rigid/train."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_stage1adapter_dataset import (  # noqa: E402
    build_strict_candidates_from_raw_sample,
    choose_best_record,
    ensure_dir,
    export_window_package,
    write_json,
)


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "stage1adapter_simple_window"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def find_raw_samples(dataset_root: Path, split: str, sample_filter: str) -> list[Path]:
    split_root = (dataset_root / split).resolve()
    if not split_root.is_dir():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    samples: list[Path] = []
    for meta_path in sorted(split_root.rglob("metadata.json")):
        sample_dir = meta_path.parent
        if "invalid_by_qa" in sample_dir.parts:
            continue
        if not (sample_dir / "physics" / "anchor_targets.npz").is_file():
            continue
        if sample_filter and sample_filter not in str(sample_dir):
            continue
        samples.append(sample_dir)
    return samples


def rel_source_path(dataset_root: Path, split: str, sample_dir: Path) -> Path:
    return sample_dir.resolve().relative_to((dataset_root / split).resolve())


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    collision_counter = Counter(str(item.get("collision_bucket", "")) for item in items)
    motion_counter = Counter(str(item.get("motion_complexity", "")) for item in items)
    source_counter = Counter()
    for item in items:
        rel_dir = Path(str(item.get("rel_dir", "")))
        parts = rel_dir.parts
        key = "/".join(parts[:4]) if len(parts) >= 4 else str(rel_dir.parent)
        source_counter[key] += 1
    return {
        "total": len(items),
        "collision_buckets": dict(sorted(collision_counter.items())),
        "motion_complexity": dict(sorted(motion_counter.items())),
        "source_groups": dict(sorted(source_counter.items())),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    if args.rebuild and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)
    ensure_dir(output_root / "manifests")

    raw_samples = find_raw_samples(dataset_root, args.split, args.sample_filter)
    if args.limit > 0:
        raw_samples = raw_samples[: int(args.limit)]

    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for sample_dir in raw_samples:
        candidates = build_strict_candidates_from_raw_sample(sample_dir)
        best = choose_best_record(candidates)
        if best is None:
            skipped.append({"sample_dir": str(sample_dir), "reason": "no_strict_simple_window"})
            continue
        rel_source = rel_source_path(dataset_root, args.split, sample_dir)
        out_dir = output_root / args.split / "genesis" / rel_source
        items.append(
            export_window_package(
                record=best,
                out_dir=out_dir,
                package_root=output_root,
                sample_id=sample_dir.name,
                split=args.split,
                dataset_name="genesis",
                sample_label=str(rel_source),
                source_meta_json_path=str(sample_dir / "metadata.json"),
            )
        )

    write_json(output_root / "manifests" / f"{args.split}_items.json", items)
    write_json(output_root / "manifests" / f"{args.split}_skipped.json", skipped)
    write_json(
        output_root / "manifests" / "summary.json",
        {
            "dataset_root": str(dataset_root),
            "output_root": str(output_root),
            "split": args.split,
            "sample_filter": str(args.sample_filter),
            "scanned_raw_samples": len(raw_samples),
            "accepted_windows": len(items),
            "skipped_samples": len(skipped),
            "accepted_summary": summarize(items),
        },
    )
    (output_root / f"{args.split}_samples.txt").write_text(
        "".join(f"{item['sample_dir']}\n" for item in items),
        encoding="utf-8",
    )
    print(json.dumps({
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "split": args.split,
        "scanned_raw_samples": len(raw_samples),
        "accepted_windows": len(items),
        "skipped_samples": len(skipped),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
