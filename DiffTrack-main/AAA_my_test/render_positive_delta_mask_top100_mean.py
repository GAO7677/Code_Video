#!/usr/bin/env python3
"""Render frozen positive-delta removal-mask diagnostics for Top100 object-query heads."""

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


def normalized_single(values: np.ndarray) -> np.ndarray:
    normalized, _ = normalize_pair(values, values)
    return normalized


def main() -> None:
    args = parse_args()
    frames = read_frames(args.video)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for capture_path in sorted(args.capture_root.glob("step_*.npz")):
        with np.load(capture_path, allow_pickle=False) as data:
            arrays = {
                key: data[key].astype(np.float32).mean(axis=0)
                for key in (
                    "before",
                    "after",
                    "reference10",
                    "target40",
                    "positive_delta",
                    "raw_mask",
                    "dilated_mask",
                    "removed",
                )
            }
            names = data["region_names"].astype(str)
            step40 = int(data["step40"].item())
            branch = str(data["branch"].item())
            sigma40 = float(data["sigma40"].item())
            matched_steps = data["matched_step10"].astype(np.int64)
            matched_sigmas = data["matched_sigma10"].astype(np.float32)
        for region_index, name in enumerate(names):
            maps = {key: value[region_index].reshape(13, 16, 28) for key, value in arrays.items()}
            before_norm, after_norm = normalize_pair(maps["before"], maps["after"])
            unique, counts = np.unique(matched_steps[:, region_index], return_counts=True)
            distribution = " ".join(f"S{int(item):02d}:{int(count)}" for item, count in zip(unique, counts))
            sigma_values = matched_sigmas[:, region_index]
            sigma_note = f"sigma10=[{sigma_values.min():.6g},{sigma_values.max():.6g}]"
            width = PANEL_WIDTH * len(frames)
            canvas = np.concatenate(
                [
                    header(
                        width,
                        f"Positive-Delta P95 + Dilate1 | {branch} | {name} | "
                        f"A40 S{step40:02d} sigma={sigma40:.6g} - matched A10 ({distribution}) {sigma_note}",
                    ),
                    strip(frames, normalized_single(maps["target40"]), "FROZEN NO-INTERVENTION A40", cv2.COLORMAP_TURBO),
                    strip(frames, normalized_single(maps["reference10"]), "10-STEP REFERENCE", cv2.COLORMAP_TURBO),
                    strip(frames, normalize_delta(maps["positive_delta"]), "POSITIVE DELTA A40-A10", cv2.COLORMAP_MAGMA),
                    strip(frames, maps["raw_mask"].clip(0, 1), "RAW P95 REMOVE MASK", cv2.COLORMAP_HOT),
                    strip(frames, maps["dilated_mask"].clip(0, 1), "DILATED REMOVE MASK r=1", cv2.COLORMAP_HOT),
                    strip(frames, before_norm, "LIVE A40 BEFORE", cv2.COLORMAP_TURBO),
                    strip(frames, after_norm, "AFTER REMOVE + RENORMALIZE", cv2.COLORMAP_TURBO),
                    strip(frames, normalize_delta(maps["removed"]), "REMOVED ATTENTION MASS", cv2.COLORMAP_MAGMA),
                ],
                axis=0,
            )
            output = args.output_root / f"step{step40:02d}__{branch}__{name}__top100_mean_before_after_mask.jpg"
            if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed to write {output}")
            print(output)


if __name__ == "__main__":
    main()
