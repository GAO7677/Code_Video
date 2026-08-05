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
ALL_TOKEN_PANEL = 416
ALL_TOKEN_HEADER = 42
TOKEN_GRID_HEIGHT = 16
TOKEN_GRID_WIDTH = 28


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


def object_query_rows(points, query_frame, context_shape, source_tokens, pooled_tokens):
    frame_tokens = TOKEN_GRID_HEIGHT * TOKEN_GRID_WIDTH
    height, width = context_shape[:2]
    rows = []
    for x, y in points:
        token_x = min(TOKEN_GRID_WIDTH - 1, max(0, int(float(x) * TOKEN_GRID_WIDTH / width)))
        token_y = min(TOKEN_GRID_HEIGHT - 1, max(0, int(float(y) * TOKEN_GRID_HEIGHT / height)))
        source_row = int(query_frame) * frame_tokens + token_y * TOKEN_GRID_WIDTH + token_x
        pooled_row = min(
            int(pooled_tokens) - 1,
            max(0, int(source_row * int(pooled_tokens) / int(source_tokens))),
        )
        rows.append(pooled_row)
    return sorted(set(rows))


def all_token_qk_image(before, after, title, query_rows, region_name):
    log_before = np.log10(np.maximum(before.astype(np.float32), 1e-12))
    log_after = np.log10(np.maximum(after.astype(np.float32), 1e-12))
    shared = np.concatenate([log_before.ravel(), log_after.ravel()])
    low, high = np.percentile(shared, [0.5, 99.5])
    delta = np.abs(after.astype(np.float32) - before.astype(np.float32))
    delta_high = float(np.percentile(delta, 99.5)) or 1e-12

    def colored(values, vmin, vmax, colormap):
        scale = max(float(vmax - vmin), 1e-12)
        normalized = np.clip((values - vmin) / scale, 0.0, 1.0)
        return cv2.applyColorMap(np.uint8(normalized * 255), colormap)

    panels = (
        (colored(log_before, low, high, cv2.COLORMAP_MAGMA), "Before log10(A)"),
        (colored(log_after, low, high, cv2.COLORMAP_MAGMA), "After log10(A)"),
        (colored(delta, 0.0, delta_high, cv2.COLORMAP_VIRIDIS), "|After - Before|"),
    )
    canvas = np.full(
        (ALL_TOKEN_HEADER + ALL_TOKEN_PANEL, 3 * ALL_TOKEN_PANEL, 3),
        242,
        dtype=np.uint8,
    )
    for index, (panel, label) in enumerate(panels):
        x = index * ALL_TOKEN_PANEL
        canvas[ALL_TOKEN_HEADER:, x:x + ALL_TOKEN_PANEL] = panel
        cv2.putText(
            canvas, label, (x + 8, 27), cv2.FONT_HERSHEY_SIMPLEX,
            .58, (28, 38, 34), 2,
        )
        line_color = (235, 188, 55) if region_name == "object_A" else (45, 145, 245)
        cv2.putText(
            canvas, f"{region_name} query rows", (x + 245, 27),
            cv2.FONT_HERSHEY_SIMPLEX, .36, line_color, 1,
        )
        for row in query_rows:
            y = ALL_TOKEN_HEADER + int(row)
            cv2.line(canvas, (x, y), (x + ALL_TOKEN_PANEL - 1, y), (12, 18, 16), 3)
            cv2.line(canvas, (x, y), (x + ALL_TOKEN_PANEL - 1, y), line_color, 1)
    cv2.putText(
        canvas, title, (8, ALL_TOKEN_HEADER + ALL_TOKEN_PANEL - 9),
        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1,
    )
    return canvas


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
            all_token_capture = (
                args.capture_root / "all_token" /
                f"{name.split('__', 1)[0]}__{group}"
                f"__step{int(scalar(payload, 'step')):02d}"
                f"__b{int(scalar(payload, 'block')):02d}"
                f"_h{int(scalar(payload, 'head')):02d}.npz"
            )
            if all_token_capture.is_file():
                with np.load(all_token_capture) as all_token:
                    all_token_filename = (
                        f"{all_token_capture.stem}__{region_name}__all_token_qk.jpg"
                    )
                    query_rows = object_query_rows(
                        points,
                        int(scalar(payload, "query_latent_frame")),
                        context_frame.shape,
                        int(scalar(all_token, "source_tokens")),
                        all_token["before"].shape[0],
                    )
                    image = all_token_qk_image(
                        all_token["before"], all_token["after"],
                        f"L{int(scalar(payload, 'block')):02d}/"
                        f"H{int(scalar(payload, 'head')):02d} | 5824 -> 416 bins",
                        query_rows,
                        region_name,
                    )
                    cv2.imwrite(
                        str(args.output_root / all_token_filename), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 94],
                    )
                    outputs["all_token_qk"] = all_token_filename
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
