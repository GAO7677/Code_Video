#!/usr/bin/env python3
"""Draw selected latent query patches on a video frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ball_query_attention import parse_query_coords


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--query-coords", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-height", type=int, default=16)
    parser.add_argument("--grid-width", type=int, default=28)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {args.frame} from {args.video}")

    height, width = frame.shape[:2]
    if height % args.grid_height or width % args.grid_width:
        raise ValueError(
            f"frame {width}x{height} is not divisible by "
            f"{args.grid_width}x{args.grid_height}"
        )
    cell_h = height // args.grid_height
    cell_w = width // args.grid_width
    coords = parse_query_coords(args.query_coords)
    query_time = coords[0][0]
    if any(time != query_time for time, _, _ in coords):
        raise ValueError("preview expects all query patches at one latent time")

    selected_rows = [row for _, row, _ in coords]
    selected_columns = [column for _, _, column in coords]
    annotated = frame.copy()
    for _, row, column in coords:
        start = (column * cell_w, row * cell_h)
        stop = ((column + 1) * cell_w - 1, (row + 1) * cell_h - 1)
        cv2.rectangle(annotated, start, stop, (70, 245, 130), 3)
    label = (
        f"{args.label} | video frame {args.frame} -> latent t={query_time} | "
        f"{len(coords)} ball query patches"
    )
    cv2.rectangle(annotated, (0, 0), (width, 38), (12, 20, 16), -1)
    cv2.putText(
        annotated,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    pad = 32
    x0 = max(0, min(selected_columns) * cell_w - pad)
    x1 = min(width, (max(selected_columns) + 1) * cell_w + pad)
    y0 = max(0, min(selected_rows) * cell_h - pad)
    y1 = min(height, (max(selected_rows) + 1) * cell_h + pad)
    crop = annotated[y0:y1, x0:x1]
    zoom_height = height
    zoom_width = int(round(crop.shape[1] * zoom_height / crop.shape[0]))
    zoom = cv2.resize(crop, (zoom_width, zoom_height), interpolation=cv2.INTER_NEAREST)
    combined = cv2.hconcat((annotated, zoom))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), combined):
        raise RuntimeError(f"failed to write {args.output}")
    print(
        f"preview={args.output} frame_shape={width}x{height} "
        f"query_coords={coords}"
    )


if __name__ == "__main__":
    main()
