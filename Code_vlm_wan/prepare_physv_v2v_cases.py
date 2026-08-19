#!/usr/bin/env python3
"""Prepare JSON case files for the generic Qwen3.8 video-caption runner.

The PhysV V2V export stores metadata and videos in each sample directory, while
``run_qwen38_json_cases.py`` consumes a text file of JSON paths.  This adapter
bridges those two schemas without copying any video data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--one-per-family", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.one_per_family:
        selected = []
        seen = set()
        for sample in samples:
            family = sample["family_key"]
            if family not in seen:
                selected.append(sample)
                seen.add(family)
        samples = selected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_list = args.output_dir / "cases.txt"
    lines = []
    for sample in samples:
        sample_dir = Path(sample["sample_dir"])
        video = sample_dir / "raw/source_video.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        case_json = args.output_dir / f"{sample['sample_id']}.json"
        payload = {
            "case_id": sample["sample_id"],
            "input_caption": (
                f"{sample['task_type']} / {sample['controlled_variable']}="
                f"{sample['controlled_value']}"
            ),
            "source_video": str(video),
        }
        case_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines.append(str(case_json))
    case_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"selected={len(samples)}")
    print(f"case_list={case_list}")


if __name__ == "__main__":
    main()
