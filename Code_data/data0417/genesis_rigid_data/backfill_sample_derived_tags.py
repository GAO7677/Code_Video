#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from sample_bucket_labels import compute_derived_tags, load_sample_arrays


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill derived bucket tags into sample metadata.json files.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def iter_sample_dirs(dataset_root: Path, sample_filter: str) -> List[Path]:
    sample_dirs: List[Path] = []
    for meta_path in sorted(dataset_root.glob("*/*/*/metadata.json")):
        sample_dir = meta_path.parent
        if sample_filter and sample_filter not in str(sample_dir):
            continue
        if not (sample_dir / "physics" / "collision_events.json").exists():
            continue
        if not (sample_dir / "physics" / "rigid_kinematics.npz").exists():
            continue
        if not (sample_dir / "physics" / "anchor_targets.npz").exists():
            continue
        sample_dirs.append(sample_dir)
    return sample_dirs


def merge_metadata(metadata: Dict[str, Any], derived_tags: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(metadata)
    merged["derived_tags"] = dict(derived_tags)
    merged["motion_label"] = str(derived_tags["motion_label"])
    merged["motion_score"] = float(derived_tags["motion_score"])
    merged["motion_metrics"] = dict(derived_tags["motion_metrics"])
    merged["collision_type_bucket"] = str(derived_tags["collision_type_bucket"])
    merged["collision_profile_bucket"] = str(derived_tags["collision_profile_bucket"])
    merged["collision_count_bucket"] = str(derived_tags["collision_count_bucket"])
    merged["obj_obj_event_count"] = int(derived_tags["obj_obj_event_count"])
    merged["obj_env_event_count"] = int(derived_tags["obj_env_event_count"])
    merged["bucket_key"] = str(derived_tags["bucket_key"])
    merged["bucket_label"] = str(derived_tags["bucket_label"])
    merged["derived_tag_version"] = str(derived_tags["derived_tag_version"])
    return merged


def main() -> None:
    args = parse_args()
    sample_dirs = iter_sample_dirs(args.dataset_root, args.sample_filter)
    if int(args.limit) > 0:
        sample_dirs = sample_dirs[: int(args.limit)]

    updated = 0
    skipped = 0
    failed = 0

    for sample_dir in sample_dirs:
        meta_path = sample_dir / "metadata.json"
        try:
            payload = load_sample_arrays(sample_dir)
            derived_tags = compute_derived_tags(
                metadata=payload["metadata"],
                events=payload["events"],
                linear_vel=payload["linear_vel"],
                visibility_mask=payload["visibility_mask"],
            )
            merged = merge_metadata(payload["metadata"], derived_tags)
            old_text = meta_path.read_text(encoding="utf-8")
            new_text = json.dumps(merged, ensure_ascii=False, indent=2) + "\n"
            if old_text == new_text:
                skipped += 1
                continue
            if not args.dry_run:
                meta_path.write_text(new_text, encoding="utf-8")
            updated += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {sample_dir}: {exc}")

    print(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "sample_filter": args.sample_filter,
                "limit": int(args.limit),
                "dry_run": bool(args.dry_run),
                "sample_count": len(sample_dirs),
                "updated": updated,
                "skipped": skipped,
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
