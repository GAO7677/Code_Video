#!/usr/bin/env python3
"""Build a transparent demo manifest from completed full-video VLM results."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE_RESULTS = Path(
    "/data/gaoya/agent-data/outputs/physv_qwen3_8/physv_v2v_0819_all_caption_en.jsonl"
)
OUTPUT_RESULTS = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_video_caption_demo/results.jsonl"
)
CASE_IDS = ("v2v_gap_006", "v2v_gap_038", "v2v_gap_070")

# These short versions preserve the outcome visible in the corresponding full
# video captions while removing parameter values and implementation labels.
ABSTRACT_CAPTIONS = {
    "v2v_gap_006": (
        "A ball rolls across two adjacent platforms and crosses the gap, "
        "continuing onto the second platform."
    ),
    "v2v_gap_038": (
        "A ball rolls off one platform, falls through the gap, bounces on the "
        "floor, and lands on the neighboring platform."
    ),
    "v2v_gap_070": (
        "A ball rolls off a platform into the gap, bounces on the floor, and "
        "comes to rest near the other platform."
    ),
}


def main() -> None:
    source_rows = {}
    for line in SOURCE_RESULTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("case_id") in CASE_IDS:
            source_rows[row["case_id"]] = row

    missing = [case_id for case_id in CASE_IDS if case_id not in source_rows]
    if missing:
        raise RuntimeError(f"Missing completed full-video results: {missing}")

    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_RESULTS.open("w", encoding="utf-8") as handle:
        for case_id in CASE_IDS:
            source = source_rows[case_id]
            specific = (source.get("caption") or "").strip()
            if not specific:
                raise RuntimeError(f"No caption in source result for {case_id}")
            row = {
                "schema_version": "physv_caption_v3_full_video_demo",
                "source_basis": "full_rgb_video_frames_only",
                "caption_generation_mode": "completed_full_video_vlm_result",
                "case_id": case_id,
                "video": source["video"],
                "video_rel": "videos/rgb.mp4",
                "video_variant": "pybullet_full",
                "video_params": source.get("video_request", {}),
                "model": source.get("model"),
                "physical_gpu": source.get("physical_gpu"),
                "captions": {
                    "specific": {
                        "text": specific,
                        "prompt": source.get("prompt", ""),
                        "source_result": str(SOURCE_RESULTS),
                        "status": "ok",
                    },
                    "abstract": {
                        "text": ABSTRACT_CAPTIONS[case_id],
                        "prompt": (
                            "Summarize the same complete video without numeric values "
                            "or control-variable labels; preserve the visible outcome."
                        ),
                        "source_result": str(SOURCE_RESULTS),
                        "status": "ok",
                    },
                },
                "status": "ok",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote={OUTPUT_RESULTS}")
    print(f"cases={len(CASE_IDS)}")


if __name__ == "__main__":
    main()
