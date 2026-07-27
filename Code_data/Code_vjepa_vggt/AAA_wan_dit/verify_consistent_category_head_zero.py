#!/usr/bin/env python3
"""Validate one protocol-consistent Head-category ablation video."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from consistent_head_targets import CATEGORIES, targets_for_category


CASE = "0613pybullet_sample_001460_w002"
FFPROBE = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffprobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classification-metadata", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    expected_targets, source = targets_for_category(
        args.classification_metadata,
        args.category,
    )
    candidates = []
    for path in root.glob(f"{CASE}.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        output_video = payload.get("output_video")
        if output_video and Path(output_video).is_file():
            candidates.append((path, payload, Path(output_video)))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one case result under {root}, found {len(candidates)}"
        )
    path, payload, video = candidates[0]
    metadata = payload.get("dit_ablation")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{path} has no dit_ablation metadata")
    actual_targets = [
        (int(item["block_id"]), int(item["head_id"]))
        for item in metadata.get("targets", [])
    ]
    inference_steps = int(payload.get("step", 40))
    expected_calls = len(expected_targets) * inference_steps * 2
    target_selection = metadata.get("target_selection", {})
    checks = {
        "mode": metadata.get("mode") == "self_attn_grouped_head_zero",
        "category": metadata.get("category") == args.category,
        "targets": actual_targets == expected_targets,
        "num_targets": metadata.get("num_targets") == len(expected_targets),
        "classification_sha256": (
            target_selection.get("sha256") == source["sha256"]
        ),
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
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "csv=p=0",
            str(video),
        ],
        text=True,
    ).strip()
    checks["video_shape"] = probe == "896,512,49"
    if not all(checks.values()):
        raise RuntimeError(f"Validation failed: {checks}")
    print(
        json.dumps(
            {
                "category": args.category,
                "num_targets": len(expected_targets),
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
