#!/usr/bin/env python3
"""Render eight Head-wise PCK query attention maps over generated RGB frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


LATENT_FRAMES = 13
TILE_WIDTH = 160
TILE_HEIGHT = 90
HEADER_HEIGHT = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def video_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 49:
        raise RuntimeError(f"Expected 49 frames in {path}, got {len(frames)}")
    return [frames[index] for index in range(0, 49, 4)]


def overlay(frame, values, vmax, delta=False):
    base = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    heat = cv2.resize(values.astype(np.float32), (TILE_WIDTH, TILE_HEIGHT))
    norm = np.clip(heat / max(float(vmax), 1e-12), 0.0, 1.0)
    colored = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_TURBO)
    alpha = (0.18 + 0.62 * norm)[..., None]
    mixed = base * (1.0 - alpha) + colored * alpha
    if delta:
        cv2.putText(mixed, "|delta|", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1)
    return np.uint8(np.clip(mixed, 0, 255))


def query_tile(context_frame, mask, points, label, stripe_color):
    tile = cv2.resize(context_frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask.astype(np.uint8), (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_NEAREST) > 0
    tint = np.zeros_like(tile)
    tint[:] = (45, 145, 210)
    tile[resized_mask] = (0.72 * tile[resized_mask] + 0.28 * tint[resized_mask]).astype(np.uint8)
    scale_x = TILE_WIDTH / context_frame.shape[1]
    scale_y = TILE_HEIGHT / context_frame.shape[0]
    for x, y in points:
        center = (int(round(float(x) * scale_x)), int(round(float(y) * scale_y)))
        cv2.circle(tile, center, 3, (20, 20, 235), -1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (TILE_WIDTH - 1, 19), (24, 31, 28), -1)
    cv2.rectangle(tile, (0, 0), (5, TILE_HEIGHT - 1), stripe_color, -1)
    cv2.putText(tile, label, (9, 14), cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1)
    return tile


def header(canvas, title):
    cv2.putText(canvas, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 42, 36), 2)
    for key_frame in range(LATENT_FRAMES):
        x = (key_frame + 1) * TILE_WIDTH
        cv2.putText(
            canvas,
            f"K{key_frame:02d}/F{key_frame*4:02d}",
            (x + 38, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            .42,
            (30, 42, 36),
            1,
        )


def matrix_image(frames, values, context_frame, mask, points, vmax, title, delta=False):
    canvas = np.full(
        (HEADER_HEIGHT + TILE_HEIGHT, (LATENT_FRAMES + 1) * TILE_WIDTH, 3),
        245,
        dtype=np.uint8,
    )
    header(canvas, title)
    y = HEADER_HEIGHT
    canvas[y:y + TILE_HEIGHT, :TILE_WIDTH] = query_tile(
        context_frame,
        mask,
        points,
        "SUM P00-P07 Q=F04",
        (92, 130, 35),
    )
    for key_frame in range(LATENT_FRAMES):
        x = (key_frame + 1) * TILE_WIDTH
        canvas[y:y + TILE_HEIGHT, x:x + TILE_WIDTH] = overlay(
            frames[key_frame], values[key_frame], vmax, delta
        )
    return canvas


def paired_image(frames, before, after, context_frame, mask, points, vmax, title):
    canvas = np.full(
        (
            HEADER_HEIGHT + 2 * TILE_HEIGHT,
            (LATENT_FRAMES + 1) * TILE_WIDTH,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    header(canvas, title)
    for row_index, (label, values, stripe_color) in enumerate(
        (("BEFORE", before, (46, 105, 178)), ("AFTER", after, (92, 130, 35)))
    ):
        y = HEADER_HEIGHT + row_index * TILE_HEIGHT
        canvas[y:y + TILE_HEIGHT, :TILE_WIDTH] = query_tile(
            context_frame,
            mask,
            points,
            f"SUM P00-P07 {label}",
            stripe_color,
        )
        for key_frame in range(LATENT_FRAMES):
            x = (key_frame + 1) * TILE_WIDTH
            canvas[y:y + TILE_HEIGHT, x:x + TILE_WIDTH] = overlay(
                frames[key_frame], values[key_frame], vmax
            )
    return canvas


def scalar(payload, key):
    return payload[key].item() if np.asarray(payload[key]).ndim == 0 else payload[key]


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for capture_path in sorted(args.capture_root.glob("*.npz")):
        with np.load(capture_path) as payload:
            name = capture_path.stem
            group = name.split("__")[1]
            suffix = "steps_00_40" if "step39" in name else "steps_00_10"
            video_path = args.video_root / f"{group}_{suffix}.mp4"
            if not video_path.is_file():
                continue
            frames = video_frames(video_path)
            before = payload["before"]
            after = payload["after"]
            before_sum = before.sum(axis=0)
            after_sum = after.sum(axis=0)
            delta_sum = np.abs(payload["delta"]).sum(axis=0)
            context_frame = payload["query_context_frame"]
            mask = payload["query_mask"]
            points = payload["query_points"]
            probability_vmax = float(
                np.percentile(np.concatenate([before_sum.ravel(), after_sum.ravel()]), 99.5)
            )
            delta_vmax = float(np.percentile(delta_sum, 99.5))
            region_name = str(scalar(payload, "region_name"))
            region_phrase = str(scalar(payload, "region_phrase"))
            outputs = {}
            for kind, values, vmax, is_delta in (
                ("before", before_sum, probability_vmax, False),
                ("after", after_sum, probability_vmax, False),
                ("abs_delta", delta_sum, delta_vmax, True),
            ):
                filename = f"{name}__{kind}.jpg"
                image = matrix_image(
                    frames,
                    values,
                    context_frame,
                    mask,
                    points,
                    vmax,
                    f"{name} | {kind} | vmax={vmax:.3e}",
                    is_delta,
                )
                cv2.imwrite(str(args.output_root / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                outputs[kind] = filename
            paired_filename = f"{name}__before_after.jpg"
            paired = paired_image(
                frames,
                before_sum,
                after_sum,
                context_frame,
                mask,
                points,
                probability_vmax,
                f"{name} | SUM of 8 Head-wise PCK queries | vmax={probability_vmax:.3e}",
            )
            cv2.imwrite(
                str(args.output_root / paired_filename),
                paired,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            outputs["before_after"] = paired_filename
            records.append(
                {
                    "capture": capture_path.name,
                    "protocol": "headwise_pck_sam2_context_f04",
                    "group": group,
                    "region_name": region_name,
                    "region_phrase": region_phrase,
                    "query_count": int(len(points)),
                    "query_aggregation": "sum",
                    "query_latent_frame": int(scalar(payload, "query_latent_frame")),
                    "query_pixel_frame": int(scalar(payload, "query_pixel_frame")),
                    "block": int(scalar(payload, "block")),
                    "head": int(scalar(payload, "head")),
                    "step": int(scalar(payload, "step")),
                    "pck32": float(scalar(payload, "pck32")),
                    "video": str(video_path),
                    "images": outputs,
                }
            )
    (args.output_root / "manifest.json").write_text(
        json.dumps(
            {"protocol": "headwise_pck_sam2_context_f04", "records": records},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
