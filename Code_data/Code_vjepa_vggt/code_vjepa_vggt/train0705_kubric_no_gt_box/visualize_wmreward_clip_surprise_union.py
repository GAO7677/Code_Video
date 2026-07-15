#!/usr/bin/env python3
"""Visualize the temporal union of high-surprise patches for each WMReward clip."""

from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--high-quantile", type=float, default=0.90)
    return parser.parse_args()


def load_aligned_frames(record: dict, count: int, size: int = 384) -> np.ndarray:
    reader = decord.VideoReader(record["path"], ctx=decord.cpu(0))
    indices = np.asarray(record["sampled_source_frame_indices"], dtype=np.int64)
    if len(indices) != count:
        raise ValueError(f"expected {count} sampled frames, got {len(indices)}")
    frames = reader.get_batch(indices).asnumpy()
    crop_top = int(record.get("crop_top", 0))
    if crop_top:
        frames = frames[:, crop_top:]
    return np.stack(
        [cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR) for frame in frames]
    )


def add_header(image: np.ndarray, text: str, height: int = 46) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (9, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def overlay_union(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    red = np.zeros_like(frame)
    red[..., 0] = 255
    overlay[mask] = cv2.addWeighted(frame, 0.35, red, 0.65, 0)[mask]
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return overlay


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {path}")


def make_contact(frames: list[np.ndarray], columns: int = 4) -> np.ndarray:
    rows = []
    for start in range(0, len(frames), columns):
        row = frames[start : start + columns]
        if len(row) < columns:
            row.extend([np.full_like(row[0], 255) for _ in range(columns - len(row))])
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.high_quantile < 1.0:
        raise ValueError("--high-quantile must be in (0,1)")
    output_dir = args.output_dir or (args.input_dir / "clip_high_surprise_union_q90")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads((args.input_dir / "result.json").read_text())
    stored = np.load(args.input_dir / "patch_surprise_maps_fp16.npz")
    num_frames = int(result["method"]["num_frames"])
    rows = []
    output_records = {}

    for video_name, record in result["videos"].items():
        frames = load_aligned_frames(record, num_frames)
        surprise = stored[video_name].astype(np.float32)
        surprise[surprise < 0] = np.nan
        tubelet_size = int(record["tubelet_size"])
        video_records = []

        for window_index, window in enumerate(record["windows"]):
            target_start, target_end = [int(value) for value in window["target_frame_range"]]
            token_start = target_start // tubelet_size
            token_end = (target_end + 1) // tubelet_size
            clip_map = surprise[token_start:token_end]
            finite_values = clip_map[np.isfinite(clip_map)]
            threshold = float(np.quantile(finite_values, args.high_quantile))
            high = clip_map >= threshold
            union_patch = np.any(high, axis=0)
            temporal_max = np.nanmax(clip_map, axis=0)
            union_mask = cv2.resize(
                union_patch.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            clip_name = (
                f"window_{window_index}_target_{target_start:02d}-{target_end:02d}"
            )
            clip_dir = output_dir / video_name / clip_name
            clip_dir.mkdir(parents=True, exist_ok=True)

            mask_image = np.zeros((384, 384, 3), dtype=np.uint8)
            mask_image[union_mask] = (255, 255, 255)
            mask_path = clip_dir / "high_surprise_union_mask.png"
            save_rgb(mask_path, mask_image)

            max_norm = np.clip(
                temporal_max / max(float(np.nanquantile(temporal_max, 0.99)), 1.0e-8), 0, 1
            )
            heat = cv2.applyColorMap((max_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            heat = cv2.cvtColor(
                cv2.resize(heat, (384, 384), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_BGR2RGB,
            )
            contours, _ = cv2.findContours(
                union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(heat, contours, -1, (255, 255, 255), 2)
            heat_path = clip_dir / "temporal_max_surprise_with_union.png"
            save_rgb(
                heat_path,
                add_header(
                    heat,
                    f"{video_name} | {clip_name} | q{args.high_quantile:.2f} threshold={threshold:.4f}",
                ),
            )

            contact_frames = []
            for frame_index in range(target_start, target_end + 1):
                contact_frames.append(
                    add_header(
                        overlay_union(frames[frame_index], union_mask),
                        f"f{frame_index:02d} | union high-surprise region",
                    )
                )
            contact = make_contact(contact_frames)
            contact = add_header(
                contact,
                f"{video_name} | {clip_name} | official={window['official_chunk_surprise']:.6f} "
                f"| threshold={threshold:.6f} | union={union_patch.mean() * 100:.2f}%",
            )
            contact_path = clip_dir / "target_frames_union_overlay_contact.jpg"
            save_rgb(contact_path, contact)

            clip_record = {
                "video": video_name,
                "window_index": window_index,
                "target_frame_start": target_start,
                "target_frame_end": target_end,
                "official_window_surprise": float(window["official_chunk_surprise"]),
                "high_quantile": args.high_quantile,
                "threshold": threshold,
                "union_patch_count": int(union_patch.sum()),
                "total_spatial_patches": int(union_patch.size),
                "union_area_ratio": float(union_patch.mean()),
                "mask_path": str(mask_path),
                "heatmap_path": str(heat_path),
                "contact_path": str(contact_path),
            }
            rows.append(clip_record)
            video_records.append(clip_record)
        output_records[video_name] = video_records

    with (output_dir / "clip_union_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "definition": (
                    "Within each WMReward target clip, select the top 10% finite tokenwise "
                    "surprise values, then take the spatial union over target tubelets."
                ),
                "high_quantile": args.high_quantile,
                "videos": output_records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
