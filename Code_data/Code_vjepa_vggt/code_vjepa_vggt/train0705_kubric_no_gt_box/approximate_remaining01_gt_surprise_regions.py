#!/usr/bin/env python3
"""Approximate remaining-01 error regions using remaining-35 as a pseudo-reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import decord
import numpy as np


DEFAULT_INPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "wmreward_patch_surprise_30f_physiq025_x0_remaining35_vs01_vs_gt_20260714"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "remaining01_surprise_proxy_from_remaining35_20260715"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--shared-scale", type=float, default=0.69384765625)
    return parser.parse_args()


def load_frames(record: dict, count: int) -> np.ndarray:
    reader = decord.VideoReader(record["path"], ctx=decord.cpu(0))
    frames = reader.get_batch(np.asarray(record["sampled_source_frame_indices"])).asnumpy()
    crop_top = int(record.get("crop_top", 0))
    if crop_top:
        frames = frames[:, crop_top:]
    return np.stack(
        [cv2.resize(frame, (384, 384), interpolation=cv2.INTER_LINEAR) for frame in frames]
    )


def header(image: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, 46, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (9, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def signed_heat(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values / scale, -1.0, 1.0)
    magnitude = np.abs(normalized)
    heat = np.full((*values.shape, 3), 255.0, dtype=np.float32)
    positive = normalized >= 0
    heat[..., 1] = 255.0 * (1.0 - magnitude)
    heat[..., 0] = np.where(positive, 255.0, 255.0 * (1.0 - magnitude))
    heat[..., 2] = np.where(positive, 255.0 * (1.0 - magnitude), 255.0)
    return cv2.resize(heat.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST)


def mask_to_pixels(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST).astype(bool)


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    red = np.zeros_like(frame)
    red[..., 0] = 255
    mixed = cv2.addWeighted(frame, 0.35, red, 0.65, 0)
    output = frame.copy()
    output[mask] = mixed[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, (255, 255, 255), 2)
    return output


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.input_dir / "result.json").read_text())
    archive = np.load(args.input_dir / "patch_surprise_maps_fp16.npz")
    s01 = archive["x0_remaining_01"].astype(np.float32)
    s35 = archive["x0_remaining_35"].astype(np.float32)
    s01[s01 < 0] = np.nan
    s35[s35 < 0] = np.nan
    frames = load_frames(
        result["videos"]["x0_remaining_01"], int(result["method"]["num_frames"])
    )
    tubelet = int(result["videos"]["x0_remaining_01"]["tubelet_size"])
    records = []

    for window_index, window in enumerate(result["videos"]["x0_remaining_01"]["windows"]):
        start, end = [int(value) for value in window["target_frame_range"]]
        token_start, token_end = start // tubelet, (end + 1) // tubelet
        clip01 = s01[token_start:token_end]
        clip35 = s35[token_start:token_end]
        aligned_delta = clip01 - clip35
        temporal_mean_delta = np.nanmean(aligned_delta, axis=0)
        max_map_delta = np.nanmax(clip01, axis=0) - np.nanmax(clip35, axis=0)

        aligned_threshold = float(np.quantile(aligned_delta[np.isfinite(aligned_delta)], args.quantile))
        aligned_union = np.any(aligned_delta >= aligned_threshold, axis=0)
        max_threshold = float(np.quantile(max_map_delta[np.isfinite(max_map_delta)], args.quantile))
        max_mask = max_map_delta >= max_threshold
        aligned_pixels = mask_to_pixels(aligned_union)
        max_pixels = mask_to_pixels(max_mask)
        representative = (start + end) // 2

        panels = [
            header(frames[representative], f"remaining-01 | representative f{representative:02d}"),
            header(
                signed_heat(temporal_mean_delta, args.shared_scale),
                "aligned temporal mean: S01-S35 | red=proxy anomaly",
            ),
            header(
                signed_heat(max_map_delta, args.shared_scale),
                "requested proxy: max(S01)-max(S35)",
            ),
            header(
                overlay(frames[representative], aligned_pixels),
                f"aligned positive q{args.quantile:.2f} temporal union",
            ),
            header(
                overlay(frames[representative], max_pixels),
                f"max-map subtraction positive q{args.quantile:.2f}",
            ),
        ]
        image = np.concatenate(panels, axis=1)
        image = header(
            image,
            f"GT-free approximate localization | target {start:02d}-{end:02d} | remaining-35=pseudo-reference "
            f"| unified scale +/-{args.shared_scale:.6f}",
        )
        image_path = args.output_dir / f"window_{window_index}_target_{start:02d}-{end:02d}.jpg"
        save_rgb(image_path, image)

        aligned_mask_path = args.output_dir / f"window_{window_index}_aligned_positive_union_mask.png"
        max_mask_path = args.output_dir / f"window_{window_index}_maxmap_positive_mask.png"
        save_rgb(aligned_mask_path, np.repeat((aligned_pixels * 255)[..., None], 3, axis=2).astype(np.uint8))
        save_rgb(max_mask_path, np.repeat((max_pixels * 255)[..., None], 3, axis=2).astype(np.uint8))
        records.append(
            {
                "window_index": window_index,
                "target_frame_range": [start, end],
                "aligned_delta_threshold": aligned_threshold,
                "aligned_union_area_ratio": float(aligned_union.mean()),
                "max_map_delta_threshold": max_threshold,
                "max_map_mask_area_ratio": float(max_mask.mean()),
                "mean_aligned_delta": float(np.nanmean(aligned_delta)),
                "image_path": str(image_path),
                "aligned_union_mask_path": str(aligned_mask_path),
                "max_map_mask_path": str(max_mask_path),
            }
        )

    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "uses_ground_truth": False,
                "pseudo_reference": "x0_remaining_35",
                "target": "x0_remaining_01",
                "recommended_proxy": "S01[t,y,x] - S35[t,y,x], positive q90 temporal union",
                "requested_proxy": "max_t(S01)-max_t(S35), positive q90 spatial mask",
                "shared_signed_scale": [-args.shared_scale, args.shared_scale],
                "windows": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
