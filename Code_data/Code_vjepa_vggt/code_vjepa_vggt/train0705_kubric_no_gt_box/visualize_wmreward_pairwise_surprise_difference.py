#!/usr/bin/env python3
"""Render pairwise spatial differences between WMReward patch-surprise maps."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import cv2
import decord
import numpy as np


DEFAULT_INPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "wmreward_patch_surprise_30f_physiq025_x0_remaining35_vs01_vs_gt_20260714"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--difference-quantile", type=float, default=0.90)
    return parser.parse_args()


def load_frames(record: dict, count: int) -> np.ndarray:
    reader = decord.VideoReader(record["path"], ctx=decord.cpu(0))
    indices = np.asarray(record["sampled_source_frame_indices"], dtype=np.int64)
    if len(indices) != count:
        raise ValueError(f"expected {count} frame indices, got {len(indices)}")
    frames = reader.get_batch(indices).asnumpy()
    crop_top = int(record.get("crop_top", 0))
    if crop_top:
        frames = frames[:, crop_top:]
    return np.stack(
        [cv2.resize(frame, (384, 384), interpolation=cv2.INTER_LINEAR) for frame in frames]
    )


def header(image: np.ndarray, text: str, height: int = 46) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (9, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def signed_heatmap(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values / scale, -1.0, 1.0)
    magnitude = np.abs(normalized)
    heat = np.full((*normalized.shape, 3), 255.0, dtype=np.float32)
    positive = normalized >= 0
    heat[..., 1] = 255.0 * (1.0 - magnitude)
    heat[..., 0] = np.where(positive, 255.0, 255.0 * (1.0 - magnitude))
    heat[..., 2] = np.where(positive, 255.0 * (1.0 - magnitude), 255.0)
    return cv2.resize(
        heat.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST
    )


def scalar_heatmap(values: np.ndarray, scale: float) -> np.ndarray:
    encoded = np.clip(values / scale, 0.0, 1.0)
    heat = cv2.applyColorMap((encoded * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(
        cv2.resize(heat, (384, 384), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_BGR2RGB,
    )


def union_overlay(
    frame: np.ndarray,
    union_patch: np.ndarray,
    direction_patch: np.ndarray,
) -> np.ndarray:
    union = cv2.resize(
        union_patch.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    direction = cv2.resize(
        direction_patch.astype(np.int8), (384, 384), interpolation=cv2.INTER_NEAREST
    )
    colors = np.zeros_like(frame)
    colors[direction > 0] = (255, 0, 0)
    colors[direction < 0] = (0, 80, 255)
    mixed = cv2.addWeighted(frame, 0.35, colors, 0.65, 0)
    overlay = frame.copy()
    overlay[union] = mixed[union]
    contours, _ = cv2.findContours(
        union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def main() -> None:
    args = parse_args()
    if not 0.0 < args.difference_quantile < 1.0:
        raise ValueError("--difference-quantile must be in (0,1)")
    output_dir = args.output_dir or (args.input_dir / "pairwise_surprise_difference_q90")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.input_dir / "result.json").read_text())
    archive = np.load(args.input_dir / "patch_surprise_maps_fp16.npz")
    names = list(result["videos"])
    maps = {name: archive[name].astype(np.float32) for name in names}
    for value in maps.values():
        value[value < 0] = np.nan
    num_frames = int(result["method"]["num_frames"])
    frames = {name: load_frames(result["videos"][name], num_frames) for name in names}

    comparisons = []
    all_abs_values = []
    for name_a, name_b in combinations(names, 2):
        windows = result["videos"][name_a]["windows"]
        tubelet = int(result["videos"][name_a]["tubelet_size"])
        for window_index, window in enumerate(windows):
            start, end = [int(value) for value in window["target_frame_range"]]
            token_start = start // tubelet
            token_end = (end + 1) // tubelet
            difference = maps[name_a][token_start:token_end] - maps[name_b][token_start:token_end]
            comparisons.append((name_a, name_b, window_index, start, end, difference))
            all_abs_values.append(np.abs(difference[np.isfinite(difference)]))
    shared_scale = max(float(np.quantile(np.concatenate(all_abs_values), 0.99)), 1.0e-8)

    rows = []
    for name_a, name_b, window_index, start, end, difference in comparisons:
        absolute = np.abs(difference)
        finite = absolute[np.isfinite(absolute)]
        threshold = float(np.quantile(finite, args.difference_quantile))
        union_patch = np.any(absolute >= threshold, axis=0)
        temporal_mean = np.nanmean(difference, axis=0)
        temporal_max_abs = np.nanmax(absolute, axis=0)
        max_indices = np.nanargmax(absolute, axis=0)
        direction = np.take_along_axis(difference, max_indices[None], axis=0)[0]
        direction_patch = np.sign(direction).astype(np.int8)
        representative = (start + end) // 2

        signed = signed_heatmap(temporal_mean, shared_scale)
        max_abs = scalar_heatmap(temporal_max_abs, shared_scale)
        overlay_a = union_overlay(frames[name_a][representative], union_patch, direction_patch)
        overlay_b = union_overlay(frames[name_b][representative], union_patch, direction_patch)
        union_signed = signed_heatmap(np.where(union_patch, direction, 0.0), shared_scale)

        top_row = np.concatenate(
            [
                header(frames[name_a][representative], f"A={name_a} | representative f{representative:02d}"),
                header(signed, "temporal mean delta | red=A higher, blue=B higher"),
                header(frames[name_b][representative], f"B={name_b} | representative f{representative:02d}"),
            ],
            axis=1,
        )
        bottom_row = np.concatenate(
            [
                header(overlay_a, "A + top-10% absolute-difference union"),
                header(max_abs, "temporal max absolute surprise difference"),
                header(overlay_b, "B + top-10% absolute-difference union"),
            ],
            axis=1,
        )
        union_row = np.concatenate(
            [
                header(union_signed, "signed difference inside union"),
                header(
                    cv2.resize(
                        (union_patch.astype(np.uint8) * 255),
                        (384, 384),
                        interpolation=cv2.INTER_NEAREST,
                    )[..., None].repeat(3, axis=2),
                    "binary spatial union",
                ),
                header(
                    np.full((384, 384, 3), 255, dtype=np.uint8),
                    f"red: {name_a} higher | blue: {name_b} higher",
                ),
            ],
            axis=1,
        )
        image = np.concatenate([top_row, bottom_row, union_row], axis=0)
        image = header(
            image,
            f"{name_a} - {name_b} | target {start:02d}-{end:02d} | q90={threshold:.6f} "
            f"| union={union_patch.mean() * 100:.2f}% | shared scale +/-{shared_scale:.6f}",
        )
        pair_dir = output_dir / f"{name_a}__vs__{name_b}"
        path = pair_dir / f"window_{window_index}_target_{start:02d}-{end:02d}.jpg"
        save_rgb(path, image)

        positive = union_patch & (direction_patch > 0)
        negative = union_patch & (direction_patch < 0)
        rows.append(
            {
                "video_a": name_a,
                "video_b": name_b,
                "window_index": window_index,
                "target_frame_start": start,
                "target_frame_end": end,
                "absolute_difference_q90": threshold,
                "union_area_ratio": float(union_patch.mean()),
                "a_higher_area_ratio": float(positive.mean()),
                "b_higher_area_ratio": float(negative.mean()),
                "mean_signed_difference": float(np.nanmean(difference)),
                "mean_absolute_difference": float(np.nanmean(absolute)),
                "image_path": str(path),
            }
        )

    with (output_dir / "pairwise_difference_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "definition": "delta = patch_surprise_A - patch_surprise_B",
                "color": "red means A has higher surprise; blue means B has higher surprise",
                "difference_quantile": args.difference_quantile,
                "shared_symmetric_scale": [-shared_scale, shared_scale],
                "comparisons": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
