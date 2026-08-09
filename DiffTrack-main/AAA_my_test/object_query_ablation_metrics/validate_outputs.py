#!/usr/bin/env python3
"""Fail-fast integrity audit for the 49-video metric report and media assets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.object_query_ablation_metrics.common import (  # noqa: E402
    FRAME_COUNT,
    OUTPUT_ROOT,
    RAFT_ROOT,
    safe_id,
)
from AAA_my_test.object_query_ablation_metrics.metric_definitions import (  # noqa: E402
    METRIC_DEFINITIONS,
)


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unreadable video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def main() -> None:
    report_path = OUTPUT_ROOT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report["records"]
    assert report["video_count"] == 49
    assert report["ablation_count"] == 48 == len(records)
    assert [row["rank"] for row in METRIC_DEFINITIONS] == list(range(1, 26))
    assert len({row["id"] for row in METRIC_DEFINITIONS}) == 25

    for video_id in ["baseline", "source_gt_video", *[row["id"] for row in records]]:
        with np.load(OUTPUT_ROOT / "tracks" / f"{safe_id(video_id)}.npz") as arrays:
            assert arrays["tracks"].shape == (FRAME_COUNT, 16, 2)
            assert arrays["visibility"].shape == (FRAME_COUNT, 16)
        with np.load(OUTPUT_ROOT / "masks" / f"{safe_id(video_id)}.npz") as arrays:
            assert arrays["masks"].shape == (FRAME_COUNT, 2, 704, 1280)

    assert np.load(OUTPUT_ROOT / "raft/source_gt_video.npy", mmap_mode="r").shape == (
        FRAME_COUNT - 1, 2, 352, 640
    )
    for row in records:
        identifier = row["id"]
        assert np.load(
            RAFT_ROOT / "flows" / f"{safe_id(identifier)}.npy", mmap_mode="r"
        ).shape == (FRAME_COUNT - 1, 2, 352, 640)
        for kind, expected in (("trajectory", 49), ("mask", 49), ("pixel", 49), ("raft", 48)):
            path = OUTPUT_ROOT / row["assets"][kind]
            actual = video_frames(path)
            if actual != expected:
                raise RuntimeError(f"{identifier} {kind}: expected {expected} frames, got {actual}")
        for object_name in ("object_A", "object_B"):
            for reference in ("baseline", "source_gt_video"):
                path = OUTPUT_ROOT / row["assets"]["perceptual"][object_name][reference]
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"missing perceptual montage: {path}")

    print(
        f"validated {report['video_count']} videos, {len(records)} ablations, "
        f"{len(METRIC_DEFINITIONS)} metrics and {len(records) * 8} audit assets"
    )


if __name__ == "__main__":
    main()
