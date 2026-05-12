# 用途：从 sum0504 重建 stage1adapter 路径汇总。
from __future__ import annotations

import json
import shutil
from pathlib import Path


SUM0504_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504"
)
OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/stage1adapter_simple_window"
)

SPLITS = ["train", "test", "val"]
COUNTS = ["count_01", "count_02", "count_03_04"]
BUCKETS = ["no_collision", "env_only"]


def read_paths(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def keep_stage1adapter_genesis_window(sample_path: str) -> bool:
    return "/stage1adapter/" in sample_path and "/genesis/" in sample_path


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict] = {}
    root_all_samples: list[str] = []
    leaf_rows: list[dict] = []

    for split in SPLITS:
        split_samples: list[str] = []
        split_leaf_rows: list[dict] = []

        for count_bucket in COUNTS:
            for collision_bucket in BUCKETS:
                src = (
                    SUM0504_ROOT
                    / split
                    / "rigid"
                    / count_bucket
                    / collision_bucket
                    / "samples.txt"
                )
                samples = [
                    sample
                    for sample in read_paths(src)
                    if keep_stage1adapter_genesis_window(sample) and Path(sample).exists()
                ]
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
            "note": "Only Genesis window samples in no_collision/env_only are included.",
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
        "- 来源是 `sum0504` 中属于 Genesis `stage1adapter` window 的 `no_collision` 和 `env_only` 样本。",
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
