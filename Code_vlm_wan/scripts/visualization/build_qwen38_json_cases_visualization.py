#!/usr/bin/env python3
"""Build exact-input viewer assets for JSON-list Qwen3.8 inference results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoProcessor

from build_qwen38_multi_visualization import build_case, load_rows


DEFAULT_RESULTS = Path(
    "/data/gaoya/agent-data/outputs/physv_qwen3_8/test_5_source_video_gpu7_fla.jsonl"
)
DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3.8-27B-FP8"
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physv_qwen3_8/viewer_test_5_source_video_gpu7_fla"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dataset_name(row: dict[str, Any]) -> str:
    case_id = row["case_id"]
    if case_id.startswith("0613pybullet_"):
        return "0613pybullet"
    if case_id.startswith("physicIQ_"):
        return "Physics-IQ"
    if case_id.startswith("phyco_kubric_"):
        return "Phyco-Kubric"
    return "Other"


def family_name(row: dict[str, Any]) -> str:
    return row["case_id"].replace("phyco_kubric_", "").replace("0613pybullet_", "")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    failed = [row["case_id"] for row in rows if row.get("status") != "ok"]
    if not rows or failed:
        raise ValueError(f"Expected only successful rows; failures: {failed}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    public_rows = []
    for row in rows:
        public = dict(row)
        public["dataset"] = dataset_name(row)
        public["family"] = family_name(row)
        public["sample"] = row["case_id"]
        public.update(build_case(processor, row, args.output_dir))
        public_rows.append(public)
        print(f"built={row['case_id']}", flush=True)
    with (args.output_dir / "viewer_data.json").open("w", encoding="utf-8") as handle:
        json.dump({"cases": public_rows}, handle, ensure_ascii=False, indent=2)
    print(f"viewer_data={args.output_dir / 'viewer_data.json'}")


if __name__ == "__main__":
    main()
