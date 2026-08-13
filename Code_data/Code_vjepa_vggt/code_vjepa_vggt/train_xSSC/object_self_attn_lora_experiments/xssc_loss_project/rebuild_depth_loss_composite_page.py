#!/usr/bin/env python3
"""Rebuild the depth-loss page as temporally locked composite videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import diagnose_xssc_loss as common
import visualize_depth_loss_sigma_sweep as demo


DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/full_sa_no_object_depth_loss_sigma_demo"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_rgb_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode any frames from {path}")
    return np.stack(frames)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gt_rgb = read_rgb_video(output_dir / "gt.mp4")
    gt_depth_rgb = read_rgb_video(output_dir / "gt_depth.mp4")
    future_start = int(metadata["future_frames"][0])
    fps = int(metadata["fps"])
    near_values = []
    far_values = []

    for record in metadata["levels"]:
        level_dir = output_dir / record["folder"]
        pred_rgb = read_rgb_video(level_dir / "pred_x0.mp4")
        pred_depth_rgb = read_rgb_video(level_dir / "pred_depth.mp4")
        loss_rgb = read_rgb_video(level_dir / "depth_loss_map.mp4")
        overlay_rgb = read_rgb_video(level_dir / "depth_loss_overlay.mp4")
        composite = demo.composite_grid(
            gt_rgb=gt_rgb,
            pred_rgb=pred_rgb,
            gt_depth_rgb=gt_depth_rgb,
            pred_depth_rgb=pred_depth_rgb,
            loss_rgb=loss_rgb,
            overlay_rgb=overlay_rgb,
        )
        common._write_mp4(level_dir / "composite_overlay.mp4", composite, fps)

        with np.load(level_dir / "depth_loss_maps.npz") as data:
            loss_map = data["depth_loss"]
        height = int(loss_map.shape[-2])
        far_floor = loss_map[
            future_start:,
            int(0.35 * height) : int(0.55 * height),
        ]
        near_floor = loss_map[
            future_start:,
            int(0.78 * height) :,
        ]
        far_mean = float(far_floor.mean())
        near_mean = float(near_floor.mean())
        record["loss_far_floor"] = far_mean
        record["loss_near_floor"] = near_mean
        record["loss_near_far_floor_ratio"] = near_mean / max(far_mean, 1e-12)
        near_values.append(near_mean)
        far_values.append(far_mean)
        print(
            f"[composite] {record['folder']}: "
            f"near/far={record['loss_near_far_floor_ratio']:.3f}",
            flush=True,
        )

    metadata["floor_region_analysis"] = {
        "far_floor_y_fraction": [0.35, 0.55],
        "near_floor_y_fraction": [0.78, 1.0],
        "mean_far_floor_loss_across_sigma": float(np.mean(far_values)),
        "mean_near_floor_loss_across_sigma": float(np.mean(near_values)),
        "aggregate_near_far_ratio": float(
            np.mean(near_values) / max(np.mean(far_values), 1e-12)
        ),
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    demo.build_page(
        output_dir,
        metadata["levels"],
        metadata["sample"],
        float(metadata["loss_global_p99"]),
    )
    print(f"[composite] rebuilt {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
