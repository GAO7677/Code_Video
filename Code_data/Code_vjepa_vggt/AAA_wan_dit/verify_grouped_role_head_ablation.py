#!/usr/bin/env python3
"""Validate one grouped Head-category ablation output."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from grouped_head_targets import targets_for_category


CASE = "0613pybullet_sample_001460_w002"
FFPROBE = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffprobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("wan_lora", "xssc", "physrvg"), required=True
    )
    parser.add_argument("--category", choices=("S", "T", "P", "C", "G"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    candidates = []
    for path in root.rglob(f"{CASE}.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = payload.get("output_video")
        if video and Path(video).is_file():
            candidates.append((path, payload, Path(video)))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one case result under {root}, found {len(candidates)}")
    path, payload, video = candidates[0]
    key = "physrvg_ablation" if args.model == "physrvg" else "dit_ablation"
    metadata = payload.get(key)
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{path} has no {key}")
    expected_targets = [
        {"block_id": block, "head_id": head}
        for block, head in targets_for_category(args.category)
    ]
    actual_targets = [
        {"block_id": int(item["block_id"]), "head_id": int(item["head_id"])}
        for item in metadata.get("targets", [])
    ]
    expected_calls = 240 if args.model == "physrvg" else 480
    checks = {
        "mode": metadata.get("mode") == "self_attn_grouped_head_zero",
        "category": metadata.get("category") == args.category,
        "targets": actual_targets == expected_targets,
        "num_targets": metadata.get("num_targets") == 6,
        "observed_calls": (
            metadata.get("observed_target_forward_calls") == expected_calls
        ),
        "expected_calls": (
            metadata.get("expected_target_forward_calls") == expected_calls
        ),
        "call_count_ok": metadata.get("target_forward_call_count_ok") is True,
    }
    probe = subprocess.check_output(
        [
            str(FFPROBE), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames",
            "-of", "csv=p=0", str(video),
        ],
        text=True,
    ).strip()
    checks["video_shape"] = probe == "896,512,49"
    if not all(checks.values()):
        raise RuntimeError(f"validation failed: {checks}")
    print(
        json.dumps(
            {
                "model": args.model,
                "category": args.category,
                "result_json": str(path),
                "video": str(video),
                "probe": probe,
                "checks": checks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
