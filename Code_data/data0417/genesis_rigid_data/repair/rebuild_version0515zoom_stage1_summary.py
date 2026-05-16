#!/usr/bin/env python3
# 用途：为 version0515zoom_genesis_rigid 的 stage1_subsets_v1 重建路径索引与简要统计。
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from core.utils_io import load_json, write_json, write_lines

SUBSET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid/preprocess_v1/stage1_subsets_v1"
)
OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid_stage1"
)
STAGE_NAMES = ["stage1a_precontact_strict", "stage1b_simple_dynamics"]

def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    root_summary: dict[str, Any] = {
        "dataset": "version0515zoom_genesis_rigid_stage1",
        "subset_root": str(SUBSET_ROOT),
        "stages": {},
        "notes": [
            "Only path index files are created; source stage1 windows are not moved.",
            "This summary is built from version0515zoom_genesis_rigid/preprocess_v1/stage1_subsets_v1.",
        ],
    }

    readme_lines = [
        "# version0515zoom_genesis_rigid_stage1",
        "",
        "- 这里只记录路径，不移动原始 stage1 window 数据。",
        "- 来源是 `version0515zoom_genesis_rigid/preprocess_v1/stage1_subsets_v1`。",
        "- 当前按 `stage1a_precontact_strict` 和 `stage1b_simple_dynamics` 两组记录。",
        "",
        "## 数量",
        "",
    ]

    all_samples: list[str] = []

    for stage_name in STAGE_NAMES:
        manifest_path = SUBSET_ROOT / stage_name / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        accepted = list(manifest.get("accepted", []) or [])
        skipped = list(manifest.get("skipped", []) or [])
        sample_dirs = sorted(
            {
                str(Path(str(item.get("out_dir", ""))).resolve())
                for item in accepted
                if str(item.get("out_dir", "")).strip()
            }
        )
        all_samples.extend(sample_dirs)

        write_lines(OUTPUT_ROOT / stage_name / "samples.txt", sample_dirs)
        write_json(
            OUTPUT_ROOT / stage_name / "summary.json",
            {
                "stage": stage_name,
                "num_windows": len(sample_dirs),
                "num_skipped": len(skipped),
                "manifest_path": str(manifest_path),
            },
        )
        root_summary["stages"][stage_name] = {
            "num_windows": len(sample_dirs),
            "num_skipped": len(skipped),
            "manifest_path": str(manifest_path),
        }
        readme_lines.append(f"- {stage_name}: {len(sample_dirs)}")

    all_samples = sorted(set(all_samples))
    write_lines(OUTPUT_ROOT / "all_samples.txt", all_samples)
    root_summary["num_windows"] = len(all_samples)
    write_json(OUTPUT_ROOT / "summary.json", root_summary)

    readme_lines.extend(
        [
            f"- total: {len(all_samples)}",
            "",
            "## 目录",
            "",
            "- stage1a_precontact_strict/samples.txt",
            "- stage1b_simple_dynamics/samples.txt",
        ]
    )
    write_lines(OUTPUT_ROOT / "README.md", readme_lines)


if __name__ == "__main__":
    main()
