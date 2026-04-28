#!/usr/bin/env python3
"""Backfill motion_complexity into oracle-state window pair_meta.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motion_complexity import infer_motion_complexity
from window_interactions import infer_window_interactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill motion_complexity into oracle-state window pair_meta.json files."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
        ),
    )
    parser.add_argument(
        "--sample_filter",
        type=str,
        default="",
        help="Optional substring filter on window directory path.",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=0,
        help="0 means process all windows.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite motion_complexity even if pair_meta.json already has it.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would change without writing files.",
    )
    return parser.parse_args()


def iter_window_dirs(dataset_root: Path, sample_filter: str, max_windows: int):
    count = 0
    for pair_meta_path in sorted(dataset_root.rglob("pair_meta.json")):
        window_dir = pair_meta_path.parent
        if sample_filter and sample_filter not in str(window_dir):
            continue
        yield window_dir
        count += 1
        if int(max_windows) > 0 and count >= int(max_windows):
            break


def compute_motion_complexity(window_dir: Path) -> dict:
    with np.load(window_dir / "state_pair.npz") as payload:
        if "y_state_norm" in payload:
            state_norm = np.asarray(payload["y_state_norm"]).astype(np.float32)
        elif "y_state" in payload:
            state_norm = np.asarray(payload["y_state"]).astype(np.float32)
        else:
            raise KeyError(f"No y_state_norm/y_state found in {window_dir / 'state_pair.npz'}")
        visibility = payload["y_visibility"] if "y_visibility" in payload else None
        return infer_motion_complexity(state_norm=state_norm, visibility_mask=visibility)


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {args.dataset_root}")

    total = 0
    updated = 0
    skipped_existing = 0
    label_counts: dict[str, int] = {}

    for window_dir in iter_window_dirs(args.dataset_root, args.sample_filter, args.max_windows):
        total += 1
        pair_meta_path = window_dir / "pair_meta.json"
        meta = json.loads(pair_meta_path.read_text(encoding="utf-8"))
        has_motion = isinstance(meta.get("motion_complexity"), dict) and "label" in meta["motion_complexity"]
        has_interactions = isinstance(meta.get("window_interactions"), dict) and "future_bucket" in meta["window_interactions"]
        if (not args.force) and has_motion and has_interactions:
            skipped_existing += 1
            label = str(meta["motion_complexity"]["label"])
            label_counts[label] = int(label_counts.get(label, 0)) + 1
            continue

        motion_complexity = meta.get("motion_complexity") if has_motion and not args.force else compute_motion_complexity(window_dir)
        meta["motion_complexity"] = motion_complexity
        meta["window_interactions"] = infer_window_interactions(meta)
        label = str(meta["motion_complexity"]["label"])
        label_counts[label] = int(label_counts.get(label, 0)) + 1
        if not args.dry_run:
            pair_meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        updated += 1
        if updated % 200 == 0:
            print(
                f"progress updated={updated} total_seen={total} "
                f"last_label={label} window_dir={window_dir}"
            )

    print(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "sample_filter": args.sample_filter,
                "max_windows": int(args.max_windows),
                "force": bool(args.force),
                "dry_run": bool(args.dry_run),
                "total_seen": int(total),
                "updated": int(updated),
                "skipped_existing": int(skipped_existing),
                "label_counts": label_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
