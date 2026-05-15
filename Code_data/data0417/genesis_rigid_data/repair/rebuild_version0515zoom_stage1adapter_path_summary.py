#!/usr/bin/env python3
"""Build a path-only train/test/val summary for version0515zoom stage1adapter simple windows."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


WINDOW_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid/stage1adapter_simple_window/train/genesis"
)
RAW_ASSIGNMENTS_PATH = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid/raw_split_assignments.json"
)
OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid_stage1adapter_simple_window"
)

SPLITS = ("train", "test", "val")
COUNT_BUCKETS = ("count_01", "count_02", "count_03_04")
COLLISION_BUCKETS = ("no_collision", "env_only")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def load_raw_assignments() -> dict[str, dict]:
    payload = read_json(RAW_ASSIGNMENTS_PATH)
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise RuntimeError(f"Bad raw assignments payload: {RAW_ASSIGNMENTS_PATH}")
    return assignments


def iter_window_sample_dirs() -> list[Path]:
    result: list[Path] = []
    for pair_meta_path in sorted(WINDOW_ROOT.rglob("pair_meta.json")):
        sample_dir = pair_meta_path.parent
        if sample_dir.is_dir():
            result.append(sample_dir.resolve())
    return result


def resolve_raw_source(sample_dir: Path) -> Path | None:
    meta_path = sample_dir / "meta.json"
    pair_meta_path = sample_dir / "pair_meta.json"
    candidates: list[str] = []
    if meta_path.exists():
        meta = read_json(meta_path)
        source_paths = meta.get("source_paths") or {}
        candidates.extend(
            [
                str(source_paths.get("source_sample_dir", "")).strip(),
                str(source_paths.get("source_window_dir", "")).strip(),
            ]
        )
    if pair_meta_path.exists():
        pair_meta = read_json(pair_meta_path)
        candidates.extend(
            [
                str(pair_meta.get("source_sample_dir", "")).strip(),
                str((pair_meta.get("selection_info") or {}).get("source_window_dir", "")).strip(),
            ]
        )
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None


def infer_count_bucket(sample_dir: Path, pair_meta: dict) -> str | None:
    for path in (sample_dir, Path(str(pair_meta.get("source_sample_dir", "")))):
        parts = set(path.parts)
        for bucket in COUNT_BUCKETS:
            if bucket in parts:
                return bucket
    object_count = int(((pair_meta.get("window_interactions") or {}).get("object_count")) or len(pair_meta.get("objects", []) or []))
    if object_count <= 1:
        return "count_01"
    if object_count == 2:
        return "count_02"
    if object_count in (3, 4):
        return "count_03_04"
    return None


def infer_collision_bucket(pair_meta: dict) -> str | None:
    future = (pair_meta.get("window_interactions") or {}).get("future_window") or {}
    collision = str(future.get("collision_type_bucket", "")).strip()
    if collision == "none":
        return "no_collision"
    if collision == "env_only":
        return "env_only"
    return None


def main() -> None:
    raw_assignments = load_raw_assignments()
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str], list[str]] = {}
    skipped: list[dict[str, str]] = []

    for sample_dir in iter_window_sample_dirs():
        pair_meta_path = sample_dir / "pair_meta.json"
        pair_meta = read_json(pair_meta_path)
        raw_source = resolve_raw_source(sample_dir)
        if raw_source is None:
            skipped.append({"sample_dir": str(sample_dir), "reason": "missing_raw_source"})
            continue
        assignment = raw_assignments.get(str(raw_source))
        if assignment is None:
            skipped.append({"sample_dir": str(sample_dir), "reason": "raw_source_not_in_assignments"})
            continue
        split = str(assignment.get("split", "")).strip()
        count_bucket = infer_count_bucket(sample_dir, pair_meta)
        collision_bucket = infer_collision_bucket(pair_meta)
        if split not in SPLITS or count_bucket not in COUNT_BUCKETS or collision_bucket not in COLLISION_BUCKETS:
            skipped.append(
                {
                    "sample_dir": str(sample_dir),
                    "reason": f"bad_bucket split={split} count={count_bucket} collision={collision_bucket}",
                }
            )
            continue
        grouped.setdefault((split, count_bucket, collision_bucket), []).append(str(sample_dir))

    leaf_rows: list[dict] = []
    split_counts: dict[str, int] = {}
    all_samples: list[str] = []

    for split in SPLITS:
        split_samples: list[str] = []
        split_leaf_rows: list[dict] = []
        for count_bucket in COUNT_BUCKETS:
            for collision_bucket in COLLISION_BUCKETS:
                samples = sorted(set(grouped.get((split, count_bucket, collision_bucket), [])))
                if not samples:
                    continue
                out_dir = OUTPUT_ROOT / split / "rigid" / count_bucket / collision_bucket
                write_lines(out_dir / "samples.txt", samples)
                row = {
                    "split": split,
                    "simulator_type": "rigid",
                    "object_count_bucket": count_bucket,
                    "collision_bucket": collision_bucket,
                    "num_samples": len(samples),
                    "relative_dir": str(out_dir.relative_to(OUTPUT_ROOT)),
                }
                write_json(out_dir / "summary.json", row)
                split_leaf_rows.append(row)
                leaf_rows.append(row)
                split_samples.extend(samples)
        split_samples = sorted(set(split_samples))
        split_counts[split] = len(split_samples)
        all_samples.extend(split_samples)
        write_lines(OUTPUT_ROOT / split / "samples.txt", split_samples)
        write_json(
            OUTPUT_ROOT / split / "summary.json",
            {
                "split": split,
                "simulator_type": "rigid",
                "dataset": "version0515zoom_stage1adapter_simple_window",
                "num_samples": len(split_samples),
                "leaf_groups": split_leaf_rows,
            },
        )

    all_samples = sorted(set(all_samples))
    write_lines(OUTPUT_ROOT / "all_samples.txt", all_samples)
    write_json(
        OUTPUT_ROOT / "summary.json",
        {
            "dataset": "version0515zoom_stage1adapter_simple_window",
            "num_samples": len(all_samples),
            "window_root": str(WINDOW_ROOT),
            "raw_assignments_path": str(RAW_ASSIGNMENTS_PATH),
            "splits": split_counts,
            "leaf_groups": leaf_rows,
            "skipped": skipped,
        },
    )

    readme_lines = [
        "# version0515zoom_genesis_rigid_stage1adapter_simple_window",
        "",
        "- 这里只记录路径，不移动原始 window 数据。",
        "- 来源是 `version0515zoom_genesis_rigid/stage1adapter_simple_window/train/genesis`。",
        "- split 继承自对应 raw source sample 在 `version0515zoom_genesis_rigid/raw_split_assignments.json` 中的 heldout 结果。",
        "- collision bucket 读取 window 自身的 `pair_meta.json -> window_interactions.future_window.collision_type_bucket`。",
        "- 当前只保留非空分类目录。",
        "",
        "## 数量",
        "",
        f"- train: {split_counts.get('train', 0)}",
        f"- test: {split_counts.get('test', 0)}",
        f"- val: {split_counts.get('val', 0)}",
        f"- total: {len(all_samples)}",
        "",
        "## 叶子分类",
        "",
    ]
    for row in leaf_rows:
        readme_lines.append(f"- {row['relative_dir']}: {row['num_samples']}")
    write_lines(OUTPUT_ROOT / "README.md", readme_lines)


if __name__ == "__main__":
    main()
