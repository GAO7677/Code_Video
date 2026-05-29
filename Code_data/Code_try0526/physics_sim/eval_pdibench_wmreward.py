#!/usr/bin/env python3
"""Recompute WMReward for PDI-Bench outputs via the official CLI."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from physv_eval.paths import A_OUTPUT
from physv_eval.records import save_payload, set_wmreward
from physv_eval.wmreward_official import WMRewardRunner


def main() -> None:
    runner = WMRewardRunner(cuda_visible_devices="2")
    videos = sorted(A_OUTPUT.rglob("*.mp4"))
    print(f"Found {len(videos)} videos\n")
    for index, video_path in enumerate(videos, start=1):
        json_path = video_path.with_suffix(".json")
        if not json_path.exists():
            continue
        rel_path = video_path.relative_to(A_OUTPUT)
        print(f"[{index}/{len(videos)}] {rel_path}...", end=" ", flush=True)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        result = runner.score(video_path)
        set_wmreward(payload, result)
        save_payload(json_path, payload)
        print(
            f"surprise={result['surprise']:.6f} similarity={result['similarity']:.6f}",
            flush=True,
        )
    print("\nDone")


if __name__ == "__main__":
    main()
