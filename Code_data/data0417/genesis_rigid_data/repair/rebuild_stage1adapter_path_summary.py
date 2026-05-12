# 用途：从 sum0504 重建 stage1adapter 路径汇总。
from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SUM0504_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504"
)
OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/stage1adapter_simple_window"
)
STAGE1_WINDOW_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/genesis/rigid"
)
RAW_ASSIGNMENTS_PATH = SUM0504_ROOT / "raw_split_assignments.json"

SPLITS = ["train", "test", "val"]
COUNTS = ["count_01", "count_02", "count_03_04"]
BUCKETS = ["no_collision", "env_only"]

from repair.rebuild_sum0504_index import classify_sample, scan_leaf_sample_dirs


def read_paths(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_raw_assignments() -> dict[str, dict]:
    if not RAW_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(f"missing raw assignments: {RAW_ASSIGNMENTS_PATH}")
    payload = json.loads(RAW_ASSIGNMENTS_PATH.read_text(encoding="utf-8"))
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise RuntimeError(f"bad assignments payload: {RAW_ASSIGNMENTS_PATH}")
    return assignments


def resolve_raw_source_from_meta(sample_dir: Path) -> Path | None:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    candidates = [
        meta.get("source_sample_dir"),
        meta.get("source_window_dir"),
        (meta.get("source_paths") or {}).get("source_sample_dir"),
        (meta.get("source_paths") or {}).get("source_window_dir"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).resolve()
        if path.exists():
            return path
    return None


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_assignments = load_raw_assignments()

    split_summaries: dict[str, dict] = {}
    root_all_samples: list[str] = []
    leaf_rows: list[dict] = []
    grouped_samples: dict[tuple[str, str, str], list[str]] = {}

    for sample_dir in scan_leaf_sample_dirs(STAGE1_WINDOW_ROOT):
        raw_source = resolve_raw_source_from_meta(sample_dir)
        if raw_source is None:
            continue
        raw_info = raw_assignments.get(str(raw_source.resolve()))
        if raw_info is None:
            continue

        collision_bucket, metadata, _ = classify_sample(sample_dir)
        if collision_bucket not in BUCKETS or metadata is None:
            continue
        count_bucket = str(metadata.get("object_count_bucket") or "")
        if count_bucket not in COUNTS:
            continue
        split = str(raw_info.get("split") or "")
        if split not in SPLITS:
            continue
        grouped_samples.setdefault((split, count_bucket, str(collision_bucket)), []).append(str(sample_dir.resolve()))

    for split in SPLITS:
        split_samples: list[str] = []
        split_leaf_rows: list[dict] = []

        for count_bucket in COUNTS:
            for collision_bucket in BUCKETS:
                samples = sorted(set(grouped_samples.get((split, count_bucket, collision_bucket), [])))
                if not samples:
                    continue

                out_dir = OUTPUT_ROOT / split / "rigid" / count_bucket / collision_bucket
                write_text(out_dir / "samples.txt", samples)
                write_json(
                    out_dir / "summary.json",
                    {
                        "split": split,
                        "simulator_type": "rigid",
                        "object_count_bucket": count_bucket,
                        "collision_bucket": collision_bucket,
                        "num_samples": len(samples),
                    },
                )

                row = {
                    "split": split,
                    "simulator_type": "rigid",
                    "object_count_bucket": count_bucket,
                    "collision_bucket": collision_bucket,
                    "num_samples": len(samples),
                    "relative_dir": str(out_dir.relative_to(OUTPUT_ROOT)),
                }
                split_leaf_rows.append(row)
                leaf_rows.append(row)
                split_samples.extend(samples)

        split_samples = sorted(split_samples)
        write_text(OUTPUT_ROOT / split / "samples.txt", split_samples)
        split_summary = {
            "split": split,
            "simulator_type": "rigid",
            "dataset": "stage1adapter_simple_window_genesis",
            "collision_buckets": BUCKETS,
            "num_samples": len(split_samples),
            "leaf_groups": split_leaf_rows,
        }
        write_json(OUTPUT_ROOT / split / "summary.json", split_summary)

        split_summaries[split] = split_summary
        root_all_samples.extend(split_samples)

    root_all_samples = sorted(root_all_samples)
    write_text(OUTPUT_ROOT / "all_samples.txt", root_all_samples)
    write_json(
        OUTPUT_ROOT / "summary.json",
        {
            "dataset": "stage1adapter_simple_window_genesis",
            "source_summary_root": str(SUM0504_ROOT),
            "raw_assignments_path": str(RAW_ASSIGNMENTS_PATH),
            "source_window_root": str(STAGE1_WINDOW_ROOT),
            "note": "Only Genesis no_collision/env_only window samples are included; split follows the held-out raw source sample.",
            "num_samples": len(root_all_samples),
            "splits": {
                split: split_summaries[split]["num_samples"] for split in SPLITS
            },
            "leaf_groups": leaf_rows,
        },
    )

    readme_lines = [
        "# stage1adapter_simple_window",
        "",
        "- 这里只记录路径，不移动原始数据。",
        "- 来源是 Genesis stage1adapter/train 下的 window 样本。",
        "- 只保留 `no_collision` 和 `env_only`。",
        "- split 继承自对应 raw source sample 在 `sum0504/raw_split_assignments.json` 中的 heldout 结果。",
        "- 当前只保留非空分类目录。",
        "",
        "## 数量",
        "",
        f"- train: {split_summaries['train']['num_samples']}",
        f"- test: {split_summaries['test']['num_samples']}",
        f"- val: {split_summaries['val']['num_samples']}",
        f"- total: {len(root_all_samples)}",
        "",
        "## 叶子分类",
        "",
    ]
    for row in leaf_rows:
        readme_lines.append(
            f"- {row['relative_dir']}: {row['num_samples']}"
        )
    write_text(OUTPUT_ROOT / "README.md", readme_lines)


if __name__ == "__main__":
    main()
