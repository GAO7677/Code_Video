#!/usr/bin/env python3
"""Build benchmark-style meta.json files from stage1adapter_simple_window summary paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/stage1adapter_simple_window")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage1adapter/Genesis_simple_window/tools/meta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark-style meta json files for stage1adapter simple-window split.")
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_paths(sample_dir: Path) -> dict[str, str]:
    return {
        "sample_dir": str(sample_dir),
        "context_video_path": str(sample_dir / "context_video.mp4"),
        "future_gt_video_path": str(sample_dir / "future_gt_video.mp4"),
        "full_video_path": str(sample_dir / "full_video.mp4"),
        "first_frame_path": str(sample_dir / "first_frame.png"),
        "meta_json_path": str(sample_dir / "meta.json"),
    }


def build_benchmark_meta(sample_dir: Path, split: str) -> dict[str, Any]:
    meta = load_json(sample_dir / "meta.json")
    pair_meta = load_json(sample_dir / "pair_meta.json")
    paths = make_paths(sample_dir)
    sample_id = str(meta.get("sample_id") or sample_dir.name)
    context_frames = int(meta.get("context_frames") or pair_meta.get("context_len") or 0)
    future_frames = int(meta.get("future_frames") or pair_meta.get("future_len") or 0)
    raw_frames = int(meta.get("raw_frames") or (context_frames + future_frames))
    return {
        "dataset": "version_1_genesis_rigid_data_all_cases",
        "sample_id": sample_id,
        "caption": str(meta.get("caption") or meta.get("description") or ""),
        "description": str(meta.get("detail_caption") or meta.get("description") or ""),
        "scenario": str(meta.get("bucket_label") or meta.get("sample_label") or ""),
        "split": split,
        "view_type": "window",
        "context_frames": context_frames,
        "future_frames": future_frames,
        "raw_frames": raw_frames,
        "fps": float(meta.get("fps") or 12.0),
        "paths": paths,
        "source_meta": {
            "scene_composition": meta.get("scene_composition"),
            "object_count_bucket": meta.get("object_count_bucket"),
            "collision_type_bucket": meta.get("collision_type_bucket"),
            "collision_profile_bucket": meta.get("collision_profile_bucket"),
            "collision_count_bucket": meta.get("collision_count_bucket"),
            "window_range": meta.get("window_range"),
        },
    }


def main() -> None:
    args = parse_args()
    samples_txt = args.summary_root / args.split / "samples.txt"
    if not samples_txt.is_file():
        raise FileNotFoundError(f"Missing samples list: {samples_txt}")

    output_root = args.output_root / f"stage1adapter_simple_window_{args.split}"
    meta_root = output_root / "meta_jsons"
    meta_root.mkdir(parents=True, exist_ok=True)
    meta_list_path = output_root / f"benchmark_meta_json_paths_{args.split}.txt"

    records: list[str] = []
    for raw_line in samples_txt.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sample_dir = Path(line)
        if not sample_dir.is_dir():
            continue
        if not (sample_dir / "meta.json").is_file():
            continue
        out_path = meta_root / f"{sample_dir.name}.json"
        if out_path.exists() and not args.overwrite:
            records.append(str(out_path))
            continue
        payload = build_benchmark_meta(sample_dir, args.split)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records.append(str(out_path))

    meta_list_path.write_text("\n".join(records) + ("\n" if records else ""), encoding="utf-8")
    summary = {
        "split": args.split,
        "num_meta_jsons": len(records),
        "meta_list_path": str(meta_list_path),
        "meta_root": str(meta_root),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(meta_list_path)


if __name__ == "__main__":
    main()
