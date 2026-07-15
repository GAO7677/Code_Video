#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def slice_metrics(delta: np.ndarray) -> dict[str, float]:
    mae = float(np.abs(delta).mean())
    mse = float(np.square(delta).mean())
    return {
        "mae_0_1": mae,
        "rmse_0_1": math.sqrt(mse),
        "psnr_db": float("inf") if mse == 0.0 else -10.0 * math.log10(mse),
        "changed_pixel_fraction_gt_5_255": float(
            (np.abs(delta).max(axis=-1) > (5.0 / 255.0)).mean()
        ),
    }


def compare_pair(
    baseline: np.ndarray,
    variant: np.ndarray,
    *,
    context_frames: int,
) -> dict[str, object]:
    if baseline.shape != variant.shape:
        raise ValueError(f"video shape mismatch: {baseline.shape} vs {variant.shape}")
    delta = variant - baseline
    per_frame_mae = np.abs(delta).mean(axis=(1, 2, 3))
    temporal_motion = np.abs(np.diff(variant, axis=0)).mean(axis=(1, 2, 3))
    baseline_motion = np.abs(np.diff(baseline, axis=0)).mean(axis=(1, 2, 3))
    return {
        "shape_T_H_W_C": list(baseline.shape),
        "all_frames": slice_metrics(delta),
        "context_frames": slice_metrics(delta[:context_frames]),
        "future_frames": slice_metrics(delta[context_frames:]),
        "per_frame_mae_0_1": [float(value) for value in per_frame_mae],
        "variant_temporal_motion_mean_0_1": float(temporal_motion.mean()),
        "baseline_temporal_motion_mean_0_1": float(baseline_motion.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_paths = sorted(args.baseline_dir.glob("*.mp4"))
    if not baseline_paths:
        raise ValueError(f"no baseline mp4 files in {args.baseline_dir}")
    report: dict[str, object] = {
        "baseline_dir": str(args.baseline_dir.resolve()),
        "context_frames": int(args.context_frames),
        "cases": {},
    }
    for baseline_path in baseline_paths:
        baseline = read_video(baseline_path)
        case_report: dict[str, object] = {}
        for variant_name, raw_dir in args.variant:
            variant_path = Path(raw_dir) / baseline_path.name
            if not variant_path.is_file():
                raise FileNotFoundError(variant_path)
            case_report[variant_name] = compare_pair(
                baseline,
                read_video(variant_path),
                context_frames=int(args.context_frames),
            )
        report["cases"][baseline_path.stem] = case_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
