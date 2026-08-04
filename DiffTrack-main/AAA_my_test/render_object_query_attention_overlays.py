#!/usr/bin/env python3
"""Render 13x13 object-query attention matrices over generated RGB frames."""

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


def matrix_image(frames, values, boxes, vmax, title, delta=False):
    canvas = np.full(
        (HEADER_HEIGHT + LATENT_FRAMES * TILE_HEIGHT, (LATENT_FRAMES + 1) * TILE_WIDTH, 3),
        245,
        dtype=np.uint8,
    )
    cv2.putText(canvas, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .58, (30, 42, 36), 2)
    for key_frame in range(LATENT_FRAMES):
        x = (key_frame + 1) * TILE_WIDTH
        cv2.putText(canvas, f"K{key_frame:02d}/F{key_frame*4:02d}", (x + 38, 27), cv2.FONT_HERSHEY_SIMPLEX, .42, (30, 42, 36), 1)
    for query_frame in range(LATENT_FRAMES):
        y = HEADER_HEIGHT + query_frame * TILE_HEIGHT
        query = cv2.resize(frames[query_frame], (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
        scale_x, scale_y = TILE_WIDTH / 896.0, TILE_HEIGHT / 512.0
        x1, y1, x2, y2 = boxes[query_frame]
        cv2.rectangle(
            query,
            (int(x1 * scale_x), int(y1 * scale_y)),
            (int(x2 * scale_x), int(y2 * scale_y)),
            (30, 30, 230),
            2,
        )
        cv2.putText(query, f"Q{query_frame:02d}/F{query_frame*4:02d}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        canvas[y:y + TILE_HEIGHT, :TILE_WIDTH] = query
        for key_frame in range(LATENT_FRAMES):
            x = (key_frame + 1) * TILE_WIDTH
            canvas[y:y + TILE_HEIGHT, x:x + TILE_WIDTH] = overlay(
                frames[key_frame], values[query_frame, key_frame], vmax, delta
            )
    return canvas


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for capture_path in sorted(args.capture_root.glob("*.npz")):
        payload = np.load(capture_path)
        name = capture_path.stem
        group = name.split("__")[1]
        suffix = "steps_00_40" if "step39" in name else "steps_00_10"
        video_path = args.video_root / f"{group}_{suffix}.mp4"
        if not video_path.is_file():
            continue
        frames = video_frames(video_path)
        before = payload["before"]
        after = payload["after"]
        delta = np.abs(payload["delta"])
        probability_vmax = float(np.percentile(np.concatenate([before.ravel(), after.ravel()]), 99.5))
        delta_vmax = float(np.percentile(delta, 99.5))
        outputs = {}
        for kind, values, vmax, is_delta in (
            ("before", before, probability_vmax, False),
            ("after", after, probability_vmax, False),
            ("abs_delta", delta, delta_vmax, True),
        ):
            filename = f"{name}__{kind}.jpg"
            image = matrix_image(
                frames,
                values,
                payload["query_boxes"],
                vmax,
                f"{name} | {kind} | vmax={vmax:.3e}",
                is_delta,
            )
            cv2.imwrite(str(args.output_root / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            outputs[kind] = filename
        records.append(
            {
                "capture": capture_path.name,
                "group": group,
                "block": int(payload["block"]),
                "head": int(payload["head"]),
                "step": int(payload["step"]),
                "pck32": float(payload["pck32"]),
                "video": str(video_path),
                "images": outputs,
            }
        )
    (args.output_root / "manifest.json").write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
