#!/usr/bin/env python3
# 用途：为 stage1adapter simple train 数据构建路径索引 summary。
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.utils_io import load_json, write_json, write_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a path-only summary for Genesis stage1adapter simple train packages."
    )
    parser.add_argument("--dataset_root", type=Path, required=True, help="Stage1adapter dataset root")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_group_keys(sample_dir: Path, pair_meta: dict[str, Any]) -> tuple[str, str, str, str]:
    parts = sample_dir.parts
    try:
        rigid_idx = parts.index("rigid")
    except ValueError as exc:
        raise RuntimeError(f"Could not locate rigid path marker for {sample_dir}") from exc
    scene_composition = parts[rigid_idx + 1] if rigid_idx + 1 < len(parts) else "unknown_scene"
    count_bucket = parts[rigid_idx + 2] if rigid_idx + 2 < len(parts) else "unknown_count"
    future_window = (pair_meta.get("window_interactions") or {}).get("future_window") or {}
    collision_bucket = str(future_window.get("collision_type_bucket") or "unknown_collision")
    motion_label = str((pair_meta.get("motion_complexity") or {}).get("label") or "unknown_motion")
    return scene_composition, count_bucket, collision_bucket, motion_label


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    train_root = dataset_root / "train" / "genesis" / "rigid"
    manifest_path = dataset_root / "manifests" / "train_items.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    if args.overwrite and output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    items = load_json(manifest_path)
    if not isinstance(items, list):
        raise RuntimeError(f"Bad manifest payload: {manifest_path}")

    grouped: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    scene_counts: Counter[str] = Counter()
    count_counts: Counter[str] = Counter()
    collision_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()

    for item in items:
        if not isinstance(item, dict):
            continue
        sample_dir = Path(str(item.get("sample_dir", ""))).resolve()
        pair_meta_path = sample_dir / "pair_meta.json"
        if not pair_meta_path.exists():
            continue
        pair_meta = load_json(pair_meta_path)
        scene_composition, count_bucket, collision_bucket, motion_label = infer_group_keys(sample_dir, pair_meta)
        grouped[(scene_composition, count_bucket, collision_bucket, motion_label)].append(str(sample_dir))
        scene_counts.update([scene_composition])
        count_counts.update([count_bucket])
        collision_counts.update([collision_bucket])
        motion_counts.update([motion_label])

    leaf_rows: list[dict[str, Any]] = []
    all_samples: list[str] = []
    for key in sorted(grouped):
        scene_composition, count_bucket, collision_bucket, motion_label = key
        samples = sorted(set(grouped[key]))
        leaf_dir = output_root / "train" / "rigid" / scene_composition / count_bucket / collision_bucket / motion_label
        write_lines(leaf_dir / "samples.txt", samples)
        leaf_summary = {
            "split": "train",
            "scene_composition": scene_composition,
            "object_count_bucket": count_bucket,
            "collision_bucket": collision_bucket,
            "motion_complexity": motion_label,
            "num_samples": len(samples),
            "relative_dir": str(leaf_dir.relative_to(output_root)),
        }
        write_json(leaf_dir / "summary.json", leaf_summary)
        leaf_rows.append(leaf_summary)
        all_samples.extend(samples)

    all_samples = sorted(set(all_samples))
    write_lines(output_root / "train" / "samples.txt", all_samples)
    write_json(
        output_root / "train" / "summary.json",
        {
            "split": "train",
            "dataset": "version0515zoom_genesis_rigid_stage1adapter_simple_train",
            "num_samples": len(all_samples),
            "leaf_groups": leaf_rows,
        },
    )
    write_lines(output_root / "all_samples.txt", all_samples)
    root_summary = {
        "dataset": "version0515zoom_genesis_rigid_stage1adapter_simple_train",
        "dataset_root": str(dataset_root),
        "source_train_root": str(train_root),
        "manifest_path": str(manifest_path),
        "num_samples": len(all_samples),
        "scenes": dict(sorted(scene_counts.items())),
        "count_buckets": dict(sorted(count_counts.items())),
        "collision_buckets": dict(sorted(collision_counts.items())),
        "motion_complexities": dict(sorted(motion_counts.items())),
        "leaf_groups": leaf_rows,
        "notes": [
            "Only path index files are created; stage1adapter sample folders are not moved.",
            "Grouping is based on exported stage1adapter sample paths plus pair_meta.json fields.",
            "Samples under invalid_by_qa are preserved as a separate count-bucket-like path component.",
        ],
    }
    write_json(output_root / "summary.json", root_summary)

    readme_lines = [
        "# version0515zoom_genesis_rigid_stage1adapter_simple_train",
        "",
        "- 这里只记录路径，不移动原始 stage1adapter 数据。",
        "- 来源是 `version0515zoom_genesis_rigid/stage1adapter/train/genesis/rigid`。",
        "- 叶子分组按 `scene_composition / count_bucket_path / collision_bucket / motion_complexity`。",
        "- `invalid_by_qa` 路径会被原样保留，不会并入正常 count bucket。",
        "",
        "## 数量",
        "",
        f"- train: {len(all_samples)}",
        "",
        "## 叶子分类",
        "",
    ]
    for row in leaf_rows:
        readme_lines.append(f"- {row['relative_dir']}: {row['num_samples']}")
    write_lines(output_root / "README.md", readme_lines)


if __name__ == "__main__":
    main()
