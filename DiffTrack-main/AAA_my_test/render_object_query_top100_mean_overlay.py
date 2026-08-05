#!/usr/bin/env python3
"""Render Object Query attention averaged across all captured Top100 heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FRAMES = 13
TILE_W = 160
TILE_H = 90
LABEL_W = 180
HEADER_H = 42


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    return parser.parse_args()


def video_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 49:
        raise RuntimeError(f"Expected 49 frames, got {len(frames)}: {path}")
    return [frames[index] for index in range(0, 49, 4)]


def query_sum(values):
    values = np.asarray(values)
    if values.ndim == 4:
        return values.sum(axis=0)
    if values.ndim == 3:
        return values
    raise RuntimeError(f"Unexpected query attention shape: {values.shape}")


def query_tile(payload, label):
    frame = cv2.resize(payload["query_context_frame"], (LABEL_W, TILE_H))
    mask = cv2.resize(
        payload["query_mask"].astype(np.uint8), (LABEL_W, TILE_H),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    tint = np.zeros_like(frame)
    tint[:] = (45, 145, 220)
    frame[mask] = np.uint8(0.68 * frame[mask] + 0.32 * tint[mask])
    sx = LABEL_W / payload["query_context_frame"].shape[1]
    sy = TILE_H / payload["query_context_frame"].shape[0]
    for x, y in payload["query_points"]:
        cv2.circle(frame, (int(x * sx), int(y * sy)), 3, (20, 20, 240), -1)
    cv2.rectangle(frame, (0, 0), (LABEL_W - 1, 20), (25, 31, 28), -1)
    cv2.putText(frame, label, (7, 15), cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1)
    return frame


def render(payload, frames, values, vmax, title, label):
    canvas = np.full(
        (HEADER_H + TILE_H, LABEL_W + FRAMES * TILE_W, 3), 244, np.uint8
    )
    cv2.putText(canvas, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .50, (27, 38, 33), 2)
    canvas[HEADER_H:, :LABEL_W] = query_tile(payload, label)
    for frame_index in range(FRAMES):
        x = LABEL_W + frame_index * TILE_W
        cv2.putText(
            canvas, f"K{frame_index:02d}/F{frame_index*4:02d}", (x + 34, 27),
            cv2.FONT_HERSHEY_SIMPLEX, .39, (27, 38, 33), 1,
        )
        base = cv2.resize(frames[frame_index], (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        heat = cv2.resize(values[frame_index].astype(np.float32), (TILE_W, TILE_H))
        norm = np.clip(heat / max(float(vmax), 1e-12), 0, 1)
        color = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_TURBO)
        alpha = (0.12 + 0.72 * norm)[..., None]
        canvas[HEADER_H:, x:x + TILE_W] = np.uint8(
            np.clip(base * (1 - alpha) + color * alpha, 0, 255)
        )
    anchor_x = LABEL_W + TILE_W
    cv2.rectangle(
        canvas, (anchor_x, HEADER_H),
        (anchor_x + TILE_W - 1, HEADER_H + TILE_H - 1), (40, 210, 245), 2,
    )
    return canvas


def scalar(payload, key):
    value = payload[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def main():
    cfg = args()
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    frames = video_frames(cfg.video)
    grouped = {}
    for path in sorted(cfg.capture_root.glob("*.npz")):
        with np.load(path) as data:
            region = str(scalar(data, "region_name"))
            entry = grouped.setdefault(region, {"before": [], "after": [], "heads": [], "payload": {}})
            entry["before"].append(query_sum(data["before"]))
            entry["after"].append(query_sum(data["after"]))
            entry["heads"].append((int(scalar(data, "block")), int(scalar(data, "head"))))
            if not entry["payload"]:
                entry["payload"] = {
                    key: np.asarray(data[key]).copy()
                    for key in ("query_context_frame", "query_mask", "query_points")
                }
                entry["phrase"] = str(scalar(data, "region_phrase"))
                entry["step"] = int(scalar(data, "step"))
                entry["seed"] = int(scalar(data, "seed"))
    records = []
    for region, entry in sorted(grouped.items()):
        heads = sorted(set(entry["heads"]))
        if len(heads) != len(entry["heads"]):
            raise RuntimeError(f"Duplicate heads for {region}")
        before = np.stack(entry["before"]).mean(axis=0)
        after = np.stack(entry["after"]).mean(axis=0)
        shared_max = max(
            float(np.percentile(np.concatenate([before.ravel(), after.ravel()]), 99.5)),
            1e-12,
        )
        images = {}
        is_identity = "identity" in cfg.condition
        kinds = (("original", before),) if is_identity else (("before", before), ("after", after))
        for kind, values in kinds:
            filename = f"{region}__{cfg.condition}__{kind}__top{len(heads)}_mean.jpg"
            image = render(
                entry["payload"], frames, values, shared_max,
                f"Seed {entry['seed']} | {region} | {cfg.condition} | {kind} | mean of {len(heads)} heads",
                f"SUM 8Q / MEAN {len(heads)}H",
            )
            cv2.imwrite(str(cfg.output_root / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            images[kind] = filename
        records.append({
            "region_name": region,
            "region_phrase": entry["phrase"],
            "seed": entry["seed"],
            "step": entry["step"],
            "condition": cfg.condition,
            "num_heads": len(heads),
            "query_aggregation": "sum_8_queries_then_mean_heads",
            "heads": [{"block": block, "head": head} for block, head in heads],
            "shared_vmax": shared_max,
            "images": images,
        })
    (cfg.output_root / "manifest.json").write_text(
        json.dumps({"condition": cfg.condition, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
