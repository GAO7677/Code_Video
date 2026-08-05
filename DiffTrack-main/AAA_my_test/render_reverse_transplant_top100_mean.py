#!/usr/bin/env python3
"""Render Top100 mean before/after overlays for reverse attention transplantation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from render_object_query_step_alignment_head_overlays import (
    PANEL_WIDTH,
    header,
    normalize_delta,
    normalize_pair,
    read_frames,
    strip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = read_frames(args.video)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for capture_path in sorted(args.capture_root.glob("step_*.npz")):
        with np.load(capture_path, allow_pickle=False) as data:
            before = data["before"].astype(np.float32).mean(axis=0)
            after = data["after"].astype(np.float32).mean(axis=0)
            matched = data["matched_step10"].astype(np.int64)
            names = data["region_names"].astype(str)
            step40 = int(data["step40"].item())
            branch = str(data["branch"].item())
            sigma40 = float(data["sigma40"].item()) if "sigma40" in data else None
            matched_sigma = data["matched_sigma10"].astype(np.float32) if "matched_sigma10" in data else None
        for region_index, name in enumerate(names):
            before_map = before[region_index].reshape(13, 16, 28)
            after_map = after[region_index].reshape(13, 16, 28)
            before_normalized, after_normalized = normalize_pair(before_map, after_map)
            delta_normalized = normalize_delta(np.abs(after_map - before_map))
            unique, counts = np.unique(matched[:, region_index], return_counts=True)
            distribution = " ".join(
                f"S{int(step):02d}:{int(count)}" for step, count in zip(unique, counts)
            )
            sigma_note = ""
            if sigma40 is not None and matched_sigma is not None:
                sigma10 = float(matched_sigma[0, region_index])
                step10 = int(matched[0, region_index])
                distance = abs(np.log(max(sigma40, 1e-12)) - np.log(max(sigma10, 1e-12)))
                sigma_note = (
                    f" | S40={step40:02d} sigma40={sigma40:.6g} -> "
                    f"S10={step10:02d} sigma10={sigma10:.6g} | log-distance={distance:.6g}"
                )
            width = PANEL_WIDTH * len(frames)
            canvas = np.concatenate(
                [
                    header(
                        width,
                        f"Top100 Mean | 40-step S{step40:02d} | {branch} | {name} | donor distribution {distribution}{sigma_note}",
                    ),
                    strip(frames, before_normalized, f"BEFORE A40 S{step40:02d}", cv2.COLORMAP_TURBO),
                    strip(frames, after_normalized, "AFTER matched A10", cv2.COLORMAP_TURBO),
                    strip(frames, delta_normalized, "ABS DELTA", cv2.COLORMAP_MAGMA),
                ],
                axis=0,
            )
            output = args.output_root / (
                f"step{step40:02d}__{branch}__{name}__top100_mean_before_after.jpg"
            )
            if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed to write {output}")
            print(output)


if __name__ == "__main__":
    main()
