#!/usr/bin/env python3
"""Evaluate one CYCLES reference video against its own aligned GT.

This is an oracle/self-consistency check, not the full RigidBench tracker
pipeline.  It deliberately uses the aligned CYCLES masks, depth, and
simulator trajectories as both prediction and reference so that the result
measures the upper bound of the exported GT package without SAM2,
CoTracker, VDA, DINO, or LPIPS model error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rigidbench.eval.score.depth import compute_si_mse
from rigidbench.eval.score.mask import chamfer_per_frame, iou_per_frame, l2_per_frame
from rigidbench.eval.score.track import compute_ate_scalar
from rigidbench.eval.track.gt import compute_gt_trajectories


def _read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode {path}")
    return np.stack(frames)


def _nanmean(value: np.ndarray) -> float:
    return float(np.nanmean(value)) if np.any(np.isfinite(value)) else float("nan")


def evaluate(case_dir: Path, output_dir: Path) -> dict:
    adapter = case_dir / "rigidbench"
    metadata = json.loads((adapter / "metadata.json").read_text())
    video_path = adapter / "video.mp4"
    frames = _read_video(video_path)

    mask_data = np.load(adapter / "masks.npz")
    masks = mask_data["masks"].astype(bool)
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"Expected one active actor mask, got {masks.shape}")

    # The exact same arrays are used on both sides of each metric.
    iou = _nanmean(iou_per_frame(masks, masks))
    l2 = _nanmean(l2_per_frame(masks, masks))
    chamfer = _nanmean(chamfer_per_frame(masks, masks))

    gt = compute_gt_trajectories(adapter, "roller_0")
    ate = compute_ate_scalar(
        gt["trajectories"],
        gt["trajectories"],
        int(metadata["camera"]["intrinsics"]["height"]),
        visibility=gt["visibility"],
    )["ate"]

    depth = np.load(adapter / "depth.npz")["depth"].astype(np.float32)
    # RigidBench's SI-MSE receives disparity, while the CYCLES export stores Z.
    valid_depth = np.isfinite(depth) & (depth > 0)
    disparity = np.zeros_like(depth, dtype=np.float32)
    disparity[valid_depth] = 1.0 / depth[valid_depth]
    si_mse = compute_si_mse(depth, disparity)

    positions = np.load(adapter / "trajectories.npz")["roller_0_positions"]
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))

    # For identical RGB/tracks/background, these self-distances are exactly
    # zero by definition.  They are recorded explicitly rather than invoking
    # learned DINO/CoTracker/LPIPS models, which would measure model error.
    rgb_equal = bool(np.array_equal(frames, frames))
    metrics = {
        "iou": iou,
        "l2": l2,
        "chamfer": chamfer,
        "ate": float(ate),
        "si_mse": float(si_mse),
        "lpips": 0.0,
        "ssim": 1.0,
        "ate3d": 0.0,
        "iddrift": 0.0,
        "bgdrift": 0.0,
    }
    report = {
        "evaluation_kind": "oracle_gt_vs_gt",
        "official_rigidbench_score": False,
        "sample_id": metadata["sample_id"],
        "source_video": str(video_path),
        "source_video_semantics": "rgb_cycles.mp4 / CYCLES renderer reference",
        "case_dir": str(case_dir),
        "resolution": [int(frames.shape[2]), int(frames.shape[1])],
        "fps": float(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS)),
        "frame_count": int(frames.shape[0]),
        "active_actor": "roller_0",
        "query_points": int(gt["query_points"].shape[0]),
        "visible_query_fraction": float(gt["visibility"].mean()),
        "gt_trajectory_displacement_m": displacement,
        "rgb_decode_self_equal": rgb_equal,
        "metrics": metrics,
        "interpretation": (
            "Deterministic upper-bound/self-consistency result. It validates the "
            "CYCLES-aligned GT package; it is not a learned-tracker RigidBench run."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# CYCLES GT single-case self-evaluation",
        "",
        f"- sample: `{report['sample_id']}`",
        f"- video: `{report['source_video']}`",
        f"- resolution/FPS/frames: `{report['resolution']} / {report['fps']} / {report['frame_count']}`",
        f"- evaluation kind: `{report['evaluation_kind']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in metrics.items():
        lines.append(f"| `{name}` | `{value:.10g}` |")
    lines += [
        "",
        "> This is the GT-vs-GT oracle check. It is not an official full RigidBench score and does not run SAM2/CoTracker/VDA/DINO/LPIPS.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.case_dir, args.output_dir)
    print(json.dumps(report["metrics"], indent=2))
    print(f"Results: {args.output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
