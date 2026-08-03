#!/usr/bin/env python3
"""Validate frame-0 GT-box initialization and report temporal slot retention."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default="dinov3_movic_step044000")
    parser.add_argument("--case", default="e07_mu05_m1")
    parser.add_argument("--min-recall", type=float, default=0.8)
    args = parser.parse_args()
    path = args.root / "features" / args.model / f"{args.case}.npz"
    with np.load(path) as item:
        slots = item["slots"].astype(np.float32)
        attention = item["attention"].astype(np.float32)
        selected = item["selected_slots"].astype(np.int64)
        recall = item["recall_matrix"].astype(np.float32)
    manifest_path = args.root / "manifest.json"
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = next(item for item in manifest["cases"] if item["case_id"] == args.case)
    with np.load(case["role_masks"]) as item:
        role_masks = item["masks"].astype(np.float32)
    per_frame_recall = (attention[:, None] * role_masks[:, :, None]).sum(axis=(-1, -2))
    per_frame_recall /= role_masks.sum(axis=(-1, -2))[:, :, None] + 1.0e-8
    frame0_best = per_frame_recall[0].argmax(axis=1)
    frame0_recall = per_frame_recall[0, np.arange(2), frame0_best]
    selected_recall = recall[np.arange(2), selected]
    unconstrained = recall.argmax(axis=1)
    if slots.shape != (150, 11, 512) or attention.shape != (150, 11, 16, 16):
        raise RuntimeError(f"Unexpected smoke shapes: slots={slots.shape}, attention={attention.shape}")
    if not np.isfinite(slots).all() or not np.isfinite(attention).all():
        raise RuntimeError("Smoke output contains non-finite values")
    if len(set(frame0_best.tolist())) != 2 or np.any(frame0_recall < args.min_recall):
        raise RuntimeError(
            f"Frame-0 bbox initialization failed: slots={frame0_best.tolist()}, "
            f"recall={frame0_recall.tolist()}"
        )
    print(
        f"[smoke-pass] frame0_slots={frame0_best.tolist()} "
        f"frame0_recall={[round(float(value), 4) for value in frame0_recall]} "
        f"full_video_slots={unconstrained.tolist()} "
        f"full_video_recall={[round(float(value), 4) for value in recall.max(axis=1)]}"
    )


if __name__ == "__main__":
    main()
