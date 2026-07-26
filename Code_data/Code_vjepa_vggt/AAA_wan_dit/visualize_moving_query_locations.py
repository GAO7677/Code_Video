#!/usr/bin/env python3
"""Render all per-latent-frame moving query boxes from a motion query map."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np

from motion_query_map import _center_crop_resize, _read_video
from moving_query_attention import moving_query_coords


def _contact_sheet(case: str, item: dict, output: Path) -> None:
    frames = [
        _center_crop_resize(frame)
        for frame in _read_video(Path(item["source_video"]))
    ]
    coords = moving_query_coords(
        item["trajectory"],
        frame_shape=tuple(int(value) for value in item["frame_shape"]),
    )
    grouped = {
        time: [coord for coord in coords if coord[0] == time]
        for time in range(13)
    }
    panel_w, panel_h = 448, 256
    canvas = np.full((3 * panel_h, 5 * panel_w, 3), 24, dtype=np.uint8)
    for time in range(13):
        frame_index = min(4 * time, len(frames) - 1)
        frame = frames[frame_index].copy()
        rows = [coord[1] for coord in grouped[time]]
        columns = [coord[2] for coord in grouped[time]]
        x0 = min(columns) * frame.shape[1] // 28
        x1 = (max(columns) + 1) * frame.shape[1] // 28
        y0 = min(rows) * frame.shape[0] // 16
        y1 = (max(rows) + 1) * frame.shape[0] // 16
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 4)
        cv2.putText(
            frame,
            f"latent t={time:02d} | video frame={frame_index:02d}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"grid rows {min(rows)}-{max(rows)}, cols {min(columns)}-{max(columns)}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panel = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(time, 5)
        canvas[
            row * panel_h : (row + 1) * panel_h,
            column * panel_w : (column + 1) * panel_w,
        ] = panel
    cv2.putText(
        canvas,
        case,
        (3 * panel_w + 12, 2 * panel_h + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output), canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.query_map.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for case, item in payload["cases"].items():
        name = f"{case}.jpg"
        _contact_sheet(case, item, args.output_dir / name)
        print(f"[moving-query-preview] {case} -> {name}", flush=True)
        entries.append(
            f"<section><h2>{html.escape(case)}</h2>"
            f"<a href='{html.escape(name)}'><img loading='lazy' src='{html.escape(name)}'></a>"
            "</section>"
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Moving query locations</title>
<style>
body{{margin:0;background:#171918;color:#eee;font:14px Arial,sans-serif}}
main{{max-width:1800px;margin:auto;padding:20px}} h1,h2{{letter-spacing:0}}
section{{border-top:1px solid #444;padding-top:14px;margin-top:20px}}
img{{display:block;width:100%;height:auto;background:#222}}
</style></head><body><main>
<h1>test_5 moving-query locations</h1>
<p>Green box: 2x2 Wan tokens at each latent time. Video frame is approximately 4 x latent time.</p>
{''.join(entries)}
</main></body></html>"""
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
