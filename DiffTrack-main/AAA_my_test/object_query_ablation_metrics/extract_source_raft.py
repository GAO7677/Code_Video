#!/usr/bin/env python3
"""Extract source-video RAFT flow with the exact settings used by the existing cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.analyze_legacy_ti2v_object_ablation_raft_motion import (  # noqa: E402
    DEFAULT_WEIGHT,
    extract_flow,
    load_model,
    sha256_file,
    write_flow_video,
)
from AAA_my_test.object_query_ablation_metrics.common import (  # noqa: E402
    FRAME_COUNT,
    OUTPUT_ROOT,
    RAFT_ROOT,
    SOURCE_VIDEO,
    atomic_json,
)


def decode_first_frames(path: Path, width: int, height: int) -> np.ndarray:
    """Decode exactly the first generated-length window from the longer source render."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while len(frames) < FRAME_COUNT:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != FRAME_COUNT:
        raise RuntimeError(f"expected at least {FRAME_COUNT} frames in {path}, got {len(frames)}")
    return np.stack(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = OUTPUT_ROOT / "raft"
    output.mkdir(parents=True, exist_ok=True)
    flow_path = output / "source_gt_video.npy"
    metadata_path = output / "source_gt_video.json"
    settings = {
        "width": 640,
        "height": 352,
        "flow_updates": 12,
        "weights": "C_T_SKHT_V2",
        "weight_sha256": sha256_file(DEFAULT_WEIGHT),
    }
    valid = False
    if flow_path.is_file() and metadata_path.is_file() and not args.overwrite:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            flow = np.load(flow_path, mmap_mode="r")
            valid = (
                metadata.get("settings") == settings
                and metadata.get("source_sha256") == sha256_file(SOURCE_VIDEO)
                and flow.shape == (48, 2, 352, 640)
                and flow.dtype == np.float16
            )
        except (OSError, ValueError, json.JSONDecodeError):
            valid = False
    if not valid:
        device = torch.device(args.device)
        model, transforms = load_model(DEFAULT_WEIGHT, device)
        frames = decode_first_frames(SOURCE_VIDEO, 640, 352)
        flow = extract_flow(frames, model, transforms, device, 4, 12)
        temporary = flow_path.with_name(flow_path.stem + ".tmp.npy")
        np.save(temporary, flow.astype(np.float16))
        temporary.replace(flow_path)
        atomic_json(
            metadata_path,
            {
                "video_id": "source_gt_video",
                "source_video": str(SOURCE_VIDEO),
                "source_sha256": sha256_file(SOURCE_VIDEO),
                "settings": settings,
            },
        )
    baseline_flow = np.load(RAFT_ROOT / "flows/baseline.npy", mmap_mode="r")
    max_magnitude = float(
        np.percentile(np.linalg.norm(baseline_flow.astype(np.float32), axis=1), 99.5)
    )
    flow = np.load(flow_path, mmap_mode="r")
    write_flow_video(output / "source_gt_video.mp4", flow, max_magnitude)
    print(f"ready: {flow_path}")


if __name__ == "__main__":
    main()
