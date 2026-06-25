#!/usr/bin/env python3
"""Recompute WMReward for PhysV simulation videos via the official CLI."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from physv_eval.paths import DATA_ROOT
from physv_eval.records import save_payload, set_wmreward
from physv_eval.single_case.wmreward import score_case
from physv_eval.wmreward_official import WMRewardRunner


VIDEO_DIRS = [
    DATA_ROOT / "videos" / "ball_block",
    DATA_ROOT / "videos" / "jepa_sensitivity",
]


def main() -> None:
    runner = WMRewardRunner(cuda_visible_devices="2")
    for video_dir in VIDEO_DIRS:
        if not video_dir.exists():
            continue
        videos = sorted(video_dir.glob("*.mp4"))
        print(f"[{video_dir.name}] {len(videos)} videos")
        for video_path in videos:
            json_path = video_path.with_suffix(".json")
            if not json_path.exists():
                continue
            print(f"  {video_path.stem}...", end=" ", flush=True)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            result = score_case(video_path, runner=runner)
            set_wmreward(payload, result)
            save_payload(json_path, payload)
            print(
                f"surprise={result['surprise']:.6f} similarity={result['similarity']:.6f}",
                flush=True,
            )
    print("\nDone")


if __name__ == "__main__":
    main()
