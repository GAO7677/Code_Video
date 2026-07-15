#!/usr/bin/env python3
"""Redraw single-video surprise and pairwise differences with one magnitude scale."""

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
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "wmreward_surprise_and_pairwise_unified_scale_20260715"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale-quantile", type=float, default=0.99)
    parser.add_argument("--region-quantile", type=float, default=0.90)
    return parser.parse_args()


def load_frames(record: dict, count: int) -> np.ndarray:
    reader = decord.VideoReader(record["path"], ctx=decord.cpu(0))
    indices = np.asarray(record["sampled_source_frame_indices"], dtype=np.int64)
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
        canvas, text, (9, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def positive_heat(values: np.ndarray, scale: float) -> np.ndarray:
    encoded = np.clip(values / scale, 0.0, 1.0)
    heat = cv2.applyColorMap((encoded * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(
        cv2.resize(heat, (384, 384), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_BGR2RGB,
    )


def signed_heat(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values / scale, -1.0, 1.0)
    magnitude = np.abs(normalized)
    heat = np.full((*normalized.shape, 3), 255.0, dtype=np.float32)
    positive = normalized >= 0
    heat[..., 1] = 255.0 * (1.0 - magnitude)
    heat[..., 0] = np.where(positive, 255.0, 255.0 * (1.0 - magnitude))
    heat[..., 2] = np.where(positive, 255.0 * (1.0 - magnitude), 255.0)
    return cv2.resize(heat.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST)


def resize_mask(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST).astype(bool)


def overlay_positive(frame: np.ndarray, heat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def overlay_signed(frame: np.ndarray, mask: np.ndarray, direction: np.ndarray) -> np.ndarray:
    colors = np.zeros_like(frame)
    colors[direction > 0] = (255, 0, 0)
    colors[direction < 0] = (0, 80, 255)
    mixed = cv2.addWeighted(frame, 0.35, colors, 0.65, 0)
    output = frame.copy()
    output[mask] = mixed[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 255, 255), 2)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.input_dir / "result.json").read_text())
    archive = np.load(args.input_dir / "patch_surprise_maps_fp16.npz")
    names = list(source["videos"])
    maps = {name: archive[name].astype(np.float32) for name in names}
    for value in maps.values():
        value[value < 0] = np.nan
    frames = {
        name: load_frames(source["videos"][name], int(source["method"]["num_frames"]))
        for name in names
    }

    surprise_values = np.concatenate([value[np.isfinite(value)] for value in maps.values()])
    difference_values = []
    for name_a, name_b in combinations(names, 2):
        diff = maps[name_a] - maps[name_b]
        difference_values.append(np.abs(diff[np.isfinite(diff)]))
    scale_values = np.concatenate([surprise_values, *difference_values])
    shared_scale = max(float(np.quantile(scale_values, args.scale_quantile)), 1.0e-8)
    rows = []

    for name in names:
        record = source["videos"][name]
        tubelet = int(record["tubelet_size"])
        for window_index, window in enumerate(record["windows"]):
            start, end = [int(value) for value in window["target_frame_range"]]
            token_start, token_end = start // tubelet, (end + 1) // tubelet
            clip = maps[name][token_start:token_end]
            mean_map = np.nanmean(clip, axis=0)
            max_map = np.nanmax(clip, axis=0)
            threshold = float(np.quantile(clip[np.isfinite(clip)], args.region_quantile))
            union = resize_mask(np.any(clip >= threshold, axis=0))
            representative = (start + end) // 2
            mean_heat = positive_heat(mean_map, shared_scale)
            max_heat = positive_heat(max_map, shared_scale)
            panels = [
                header(frames[name][representative], f"{name} | representative f{representative:02d}"),
                header(mean_heat, f"temporal mean surprise | range [0,{shared_scale:.6f}]"),
                header(max_heat, f"temporal max surprise | range [0,{shared_scale:.6f}]"),
                header(
                    overlay_positive(frames[name][representative], max_heat, union),
                    f"overlay + q{args.region_quantile:.2f} temporal union",
                ),
            ]
            image = np.concatenate(panels, axis=1)
            image = header(
                image,
                f"single surprise | {name} | target {start:02d}-{end:02d} "
                f"| official={window['official_chunk_surprise']:.6f} | unified magnitude={shared_scale:.6f}",
            )
            path = args.output_dir / "single_surprise" / name / f"window_{window_index}_target_{start:02d}-{end:02d}.jpg"
            save_rgb(path, image)
            rows.append(
                {
                    "type": "single_surprise",
                    "item_a": name,
                    "item_b": "",
                    "window_index": window_index,
                    "target_start": start,
                    "target_end": end,
                    "mean_value": float(np.nanmean(clip)),
                    "mean_absolute_value": float(np.nanmean(np.abs(clip))),
                    "image_path": str(path),
                }
            )

    for name_a, name_b in combinations(names, 2):
        record = source["videos"][name_a]
        tubelet = int(record["tubelet_size"])
        for window_index, window in enumerate(record["windows"]):
            start, end = [int(value) for value in window["target_frame_range"]]
            token_start, token_end = start // tubelet, (end + 1) // tubelet
            diff = maps[name_a][token_start:token_end] - maps[name_b][token_start:token_end]
            absolute = np.abs(diff)
            mean_diff = np.nanmean(diff, axis=0)
            max_abs = np.nanmax(absolute, axis=0)
            threshold = float(np.quantile(absolute[np.isfinite(absolute)], args.region_quantile))
            union_patch = np.any(absolute >= threshold, axis=0)
            union = resize_mask(union_patch)
            max_t = np.nanargmax(absolute, axis=0)
            direction_patch = np.sign(np.take_along_axis(diff, max_t[None], axis=0)[0])
            direction = cv2.resize(direction_patch.astype(np.int8), (384, 384), interpolation=cv2.INTER_NEAREST)
            representative = (start + end) // 2
            signed = signed_heat(mean_diff, shared_scale)
            abs_heat = positive_heat(max_abs, shared_scale)
            panels = [
                header(
                    overlay_signed(frames[name_a][representative], union, direction),
                    f"A={name_a} | red=A higher, blue=B higher",
                ),
                header(signed, f"temporal mean delta | range [-{shared_scale:.6f},+{shared_scale:.6f}]"),
                header(abs_heat, f"temporal max |delta| | range [0,{shared_scale:.6f}]"),
                header(
                    overlay_signed(frames[name_b][representative], union, direction),
                    f"B={name_b} | q{args.region_quantile:.2f} difference union",
                ),
            ]
            image = np.concatenate(panels, axis=1)
            image = header(
                image,
                f"pairwise delta=A-B | {name_a} vs {name_b} | target {start:02d}-{end:02d} "
                f"| mean delta={np.nanmean(diff):+.6f} | unified magnitude={shared_scale:.6f}",
            )
            path = args.output_dir / "pairwise_difference" / f"{name_a}__vs__{name_b}" / f"window_{window_index}_target_{start:02d}-{end:02d}.jpg"
            save_rgb(path, image)
            rows.append(
                {
                    "type": "pairwise_difference",
                    "item_a": name_a,
                    "item_b": name_b,
                    "window_index": window_index,
                    "target_start": start,
                    "target_end": end,
                    "mean_value": float(np.nanmean(diff)),
                    "mean_absolute_value": float(np.nanmean(absolute)),
                    "image_path": str(path),
                }
            )

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.input_dir),
                "scale_quantile": args.scale_quantile,
                "region_quantile": args.region_quantile,
                "unified_magnitude": shared_scale,
                "single_surprise_range": [0.0, shared_scale],
                "pairwise_signed_difference_range": [-shared_scale, shared_scale],
                "pairwise_absolute_difference_range": [0.0, shared_scale],
                "artifacts": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[scale] {shared_scale:.9f}")
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
