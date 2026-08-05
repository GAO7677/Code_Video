#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FRAMES = 13
TILE_W = 160
TILE_H = 90
LABEL_W = 160
HEADER_H = 42


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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


def heat_overlay(frame, values, vmax, removed=False):
    base = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    heat = cv2.resize(values.astype(np.float32), (TILE_W, TILE_H))
    norm = np.clip(heat / max(float(vmax), 1e-12), 0, 1)
    cmap = cv2.COLORMAP_HOT if removed else cv2.COLORMAP_TURBO
    color = cv2.applyColorMap(np.uint8(norm * 255), cmap)
    alpha = (0.12 + 0.72 * norm)[..., None]
    return np.uint8(np.clip(base * (1 - alpha) + color * alpha, 0, 255))


def query_tile(payload, label):
    frame = cv2.resize(payload["query_context_frame"], (TILE_W, TILE_H))
    mask = cv2.resize(
        payload["query_mask"].astype(np.uint8), (TILE_W, TILE_H),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    tint = np.zeros_like(frame)
    tint[:] = (45, 145, 220)
    frame[mask] = np.uint8(0.68 * frame[mask] + 0.32 * tint[mask])
    sx = TILE_W / payload["query_context_frame"].shape[1]
    sy = TILE_H / payload["query_context_frame"].shape[0]
    for x, y in payload["query_points"]:
        cv2.circle(frame, (int(x * sx), int(y * sy)), 3, (20, 20, 240), -1)
    cv2.rectangle(frame, (0, 0), (TILE_W - 1, 19), (25, 31, 28), -1)
    cv2.putText(frame, label, (7, 14), cv2.FONT_HERSHEY_SIMPLEX, .33, (255, 255, 255), 1)
    return frame


def query_sum(values):
    values = np.asarray(values)
    if values.ndim == 4:
        return values.sum(axis=0)
    if values.ndim == 3:
        return values
    raise RuntimeError(f"Expected [Q,F,H,W] or [F,H,W], got {values.shape}")


def robust_positive_max(values):
    positive = np.asarray(values)[np.asarray(values) > 0]
    if not positive.size:
        return 1e-12
    return max(float(np.percentile(positive, 99.5)), 1e-12)


def p90_mask(payload, before):
    if "p90_frequency" in payload:
        return (np.asarray(payload["p90_frequency"]) >= 0.5).astype(np.float32), "captured_majority"
    thresholds = np.quantile(before.reshape(FRAMES, -1), 0.90, axis=1)
    return (before >= thresholds[:, None, None]).astype(np.float32), "legacy_reconstructed"


def main_component_mask(payload, before):
    if "main_component_frequency" in payload:
        return (
            np.asarray(payload["main_component_frequency"]) >= 0.5
        ).astype(np.float32), "captured_majority"
    output = np.zeros_like(before, dtype=np.float32)
    top_k = 5
    for frame_index in range(FRAMES):
        frame = before[frame_index]
        candidate = np.zeros(frame.size, dtype=np.uint8)
        candidate[np.argpartition(frame.ravel(), -top_k)[-top_k:]] = 1
        candidate = candidate.reshape(frame.shape)
        _, labels = cv2.connectedComponents(candidate, connectivity=8)
        peak = np.unravel_index(np.argmax(frame), frame.shape)
        peak_label = labels[peak]
        if peak_label:
            output[frame_index] = labels == peak_label
    return output, "legacy_reconstructed"


def render(payload, frames, title):
    before = query_sum(payload["before"])
    after = query_sum(payload["after"])
    removed = query_sum(payload["removed"])
    shared_max = float(np.percentile(np.concatenate([before.ravel(), after.ravel()]), 99.5))
    removed_max = robust_positive_max(removed)
    canvas = np.full((HEADER_H + 3 * TILE_H, LABEL_W + FRAMES * TILE_W, 3), 244, np.uint8)
    cv2.putText(canvas, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .51, (27, 38, 33), 2)
    for frame_index in range(FRAMES):
        x = LABEL_W + frame_index * TILE_W
        cv2.putText(canvas, f"K{frame_index:02d}/F{frame_index*4:02d}", (x + 34, 27), cv2.FONT_HERSHEY_SIMPLEX, .39, (27, 38, 33), 1)
    for row, (label, values, vmax, removed_mode) in enumerate((
        ("BEFORE", before, shared_max, False),
        ("AFTER", after, shared_max, False),
        ("REMOVED", removed, removed_max, True),
    )):
        y = HEADER_H + row * TILE_H
        canvas[y:y + TILE_H, :LABEL_W] = query_tile(payload, label)
        for frame_index in range(FRAMES):
            x = LABEL_W + frame_index * TILE_W
            canvas[y:y + TILE_H, x:x + TILE_W] = heat_overlay(
                frames[frame_index], values[frame_index], vmax, removed_mode
            )
    anchor_x = LABEL_W + TILE_W
    cv2.rectangle(canvas, (anchor_x, HEADER_H), (anchor_x + TILE_W - 1, HEADER_H + 3 * TILE_H - 1), (40, 210, 245), 2)
    return canvas


def render_single(payload, frames, title, label, values, vmax, removed=False):
    canvas = np.full(
        (HEADER_H + TILE_H, LABEL_W + FRAMES * TILE_W, 3), 244, np.uint8
    )
    cv2.putText(canvas, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .51, (27, 38, 33), 2)
    for frame_index in range(FRAMES):
        x = LABEL_W + frame_index * TILE_W
        cv2.putText(
            canvas, f"K{frame_index:02d}/F{frame_index*4:02d}", (x + 34, 27),
            cv2.FONT_HERSHEY_SIMPLEX, .39, (27, 38, 33), 1,
        )
    canvas[HEADER_H:HEADER_H + TILE_H, :LABEL_W] = query_tile(payload, label)
    for frame_index in range(FRAMES):
        x = LABEL_W + frame_index * TILE_W
        canvas[HEADER_H:HEADER_H + TILE_H, x:x + TILE_W] = heat_overlay(
            frames[frame_index], values[frame_index], vmax, removed
        )
    anchor_x = LABEL_W + TILE_W
    cv2.rectangle(
        canvas, (anchor_x, HEADER_H),
        (anchor_x + TILE_W - 1, HEADER_H + TILE_H - 1), (40, 210, 245), 2,
    )
    return canvas


def scalar(payload, key):
    return payload[key].item() if np.asarray(payload[key]).ndim == 0 else payload[key]


def main():
    cfg = args()
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    frames = video_frames(cfg.video)
    records = []
    for path in sorted(cfg.capture_root.glob("*.npz")):
        with np.load(path) as payload:
            region = str(scalar(payload, "region_name"))
            block = int(scalar(payload, "block"))
            head = int(scalar(payload, "head"))
            seed = int(scalar(payload, "seed"))
            step = int(scalar(payload, "step"))
            mode = str(scalar(payload, "mode")) if "mode" in payload else "legacy"
            before = query_sum(payload["before"])
            after = query_sum(payload["after"])
            removed = query_sum(payload["removed"])
            mask90, mask_source = p90_mask(payload, before)
            component_mask, component_source = main_component_mask(payload, before)
            shared_max = max(
                float(np.percentile(np.concatenate([before.ravel(), after.ravel()]), 99.5)),
                1e-12,
            )
            removed_max = robust_positive_max(removed)
            title = (
                f"Seed {seed} | L{block:02d}/H{head:02d} | {region} | "
                f"SUM P00-P{payload['query_points'].shape[0] - 1:02d} | Q=F04 | S{step:03d}"
            )
            images = {}
            if mode == "probability_object_query_identity":
                identity_name = f"{path.stem}__no_intervention.jpg"
                identity = render_single(
                    payload, frames, title, "NO INTERVENTION", before, shared_max
                )
                cv2.imwrite(
                    str(cfg.output_root / identity_name), identity,
                    [cv2.IMWRITE_JPEG_QUALITY, 94],
                )
                images["identity"] = identity_name
            else:
                for kind, label, values, vmax, removed_mode in (
                    ("before", "INTERVENTION BEFORE", before, shared_max, False),
                    ("p90_mask", "P90 MASK", mask90, 1.0, True),
                    (
                        "main_component", "P99 / TOP-5 MAIN CONNECTED COMPONENT",
                        component_mask, 1.0, True,
                    ),
                    ("after", "INTERVENTION AFTER", after, shared_max, False),
                    ("removed", "REMOVED MASS", removed, removed_max, True),
                ):
                    output_name = f"{path.stem}__{kind}.jpg"
                    output = render_single(
                        payload, frames, title, label, values, vmax, removed_mode
                    )
                    cv2.imwrite(
                        str(cfg.output_root / output_name), output,
                        [cv2.IMWRITE_JPEG_QUALITY, 94],
                    )
                    images[kind] = output_name
            filename = f"{path.stem}__before_after_removed.jpg"
            image = render(
                payload,
                frames,
                title,
            )
            cv2.imwrite(str(cfg.output_root / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            images["combined"] = filename
            records.append({
                "capture": path.name,
                "image": filename,
                "region_name": region,
                "region_phrase": str(scalar(payload, "region_phrase")),
                "block": block,
                "head": head,
                "seed": seed,
                "step": step,
                "pck32": float(scalar(payload, "pck32")),
                "high_quantile": float(scalar(payload, "high_quantile")),
                "neighbor_radius": int(scalar(payload, "neighbor_radius")),
                "mode": mode,
                "query_count": int(payload["query_points"].shape[0]),
                "query_aggregation": "sum" if payload["before"].ndim == 4 else "legacy_mean",
                "max_abs_delta": float(np.abs(after - before).max()),
                "mean_abs_delta": float(np.abs(after - before).mean()),
                "removed_nonzero": int(np.count_nonzero(removed)),
                "p90_mask_source": mask_source,
                "main_component_source": component_source,
                "main_component_topk": int(scalar(payload, "main_component_topk"))
                if "main_component_topk" in payload else 5,
                "images": images,
            })
    (cfg.output_root / "manifest.json").write_text(
        json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
