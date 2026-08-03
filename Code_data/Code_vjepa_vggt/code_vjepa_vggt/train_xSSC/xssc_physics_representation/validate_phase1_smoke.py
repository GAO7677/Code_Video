#!/usr/bin/env python3
"""Fail fast when the Phase-1 GT-box smoke does not separate ball and block."""

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
    selected_recall = recall[np.arange(2), selected]
    unconstrained = recall.argmax(axis=1)
    if slots.shape != (150, 11, 512) or attention.shape != (150, 11, 16, 16):
        raise RuntimeError(f"Unexpected smoke shapes: slots={slots.shape}, attention={attention.shape}")
    if not np.isfinite(slots).all() or not np.isfinite(attention).all():
        raise RuntimeError("Smoke output contains non-finite values")
    if len(set(selected.tolist())) != 2 or len(set(unconstrained.tolist())) != 2:
        raise RuntimeError(
            f"Ball/block are not independently represented: selected={selected.tolist()}, "
            f"unconstrained={unconstrained.tolist()}, recall={selected_recall.tolist()}"
        )
    if np.any(selected_recall < args.min_recall):
        raise RuntimeError(
            f"Smoke recall below {args.min_recall}: selected={selected.tolist()}, "
            f"recall={selected_recall.tolist()}"
        )
    print(
        f"[smoke-pass] selected={selected.tolist()} "
        f"recall={[round(float(value), 4) for value in selected_recall]}"
    )


if __name__ == "__main__":
    main()
