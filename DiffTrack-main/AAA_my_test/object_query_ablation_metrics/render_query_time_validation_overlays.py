#!/usr/bin/env python3
"""Render fixed-Q_t head-group argmax clouds against the frozen GT tube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np


BASE = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage1_query_time_validation"
)
DEFAULT_RUN = BASE / "runs/0613pybullet_sample_001460_w002/seed_47326"
DEFAULT_REPORT = BASE / "analysis/report.json"
DEFAULT_TUBE_SCOPES = BASE / "analysis/tube_head_scopes.json"
DEFAULT_FIXED_SCOPES = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/pck_head_scopes_s039_latest3350.json"
)
DEFAULT_OUTPUT = BASE / "analysis/overlays/0613pybullet_sample_001460_w002/seed_47326"
PANEL_SIZE = (640, 352)
COLORS = [
    (41, 182, 246),
    (238, 130, 62),
    (101, 204, 118),
    (210, 97, 190),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fixed-scopes", type=Path, default=DEFAULT_FIXED_SCOPES)
    parser.add_argument("--tube-scopes", type=Path, default=DEFAULT_TUBE_SCOPES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-times", type=int, nargs="+", default=[0, 6, 12])
    return parser.parse_args()


def pairs(payload: dict[str, Any], scope: str) -> list[tuple[int, int]]:
    definition = payload["head_scopes"][scope]
    start, end = int(definition["rank_start"]) - 1, int(definition["rank_end"])
    return [
        (int(row["block"]), int(row["head"])) for row in payload["entries"][start:end]
    ]


def text(frame: np.ndarray, value: str, y: int, scale: float = 0.55) -> None:
    cv2.putText(
        frame,
        value,
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        value,
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def base_panel(frame: np.ndarray, title: str) -> np.ndarray:
    panel = cv2.resize(frame, PANEL_SIZE, interpolation=cv2.INTER_AREA)
    shade = panel.copy()
    cv2.rectangle(shade, (0, 0), (PANEL_SIZE[0], 38), (0, 0, 0), -1)
    panel = cv2.addWeighted(shade, 0.55, panel, 0.45, 0)
    text(panel, title, 25, 0.62)
    return panel


def scaled(point: np.ndarray, native_hw: tuple[int, int]) -> tuple[int, int]:
    native_h, native_w = native_hw
    return (
        int(round(float(point[0]) * PANEL_SIZE[0] / native_w)),
        int(round(float(point[1]) * PANEL_SIZE[1] / native_h)),
    )


def draw_gt(panel: np.ndarray, tracks: np.ndarray, qt: int, tk: int, native_hw) -> None:
    for point_index in range(tracks.shape[1]):
        color = COLORS[point_index % len(COLORS)]
        path = [scaled(tracks[t, point_index], native_hw) for t in range(13)]
        for first, second in zip(path[:-1], path[1:]):
            cv2.line(panel, first, second, color, 1, cv2.LINE_AA)
        cv2.circle(panel, path[tk], 5, (70, 255, 100), 2, cv2.LINE_AA)
        cv2.circle(panel, path[qt], 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_predictions(
    panel: np.ndarray,
    predictions: np.ndarray,
    selected: list[tuple[int, int]],
    qt: int,
    tk: int,
    native_hw,
) -> None:
    values = np.stack(
        [predictions[block, head, qt, tk].astype(np.float32) for block, head in selected]
    )
    # Draw a deterministic 20-head subset as a response cloud to keep the image readable.
    for head_values in values[::5]:
        for point in head_values:
            cv2.circle(panel, scaled(point, native_hw), 1, (70, 110, 255), -1, cv2.LINE_AA)
    mean_values = np.nanmean(values, axis=0)
    for point_index, point in enumerate(mean_values):
        color = COLORS[point_index % len(COLORS)]
        cv2.drawMarker(
            panel,
            scaled(point, native_hw),
            color,
            cv2.MARKER_CROSS,
            12,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    for path in (
        args.run_dir / "manifest.json",
        args.run_dir / "metrics.npz",
        args.report,
        args.fixed_scopes,
        args.tube_scopes,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    fixed = json.loads(args.fixed_scopes.read_text(encoding="utf-8"))
    tube = json.loads(args.tube_scopes.read_text(encoding="utf-8"))
    with np.load(args.run_dir / "metrics.npz", allow_pickle=False) as arrays:
        predictions = arrays["predictions"]
        anchors = arrays["latent_anchor_pixel_frames"].astype(np.int64)
    track_path = Path(manifest["track_path"])
    with np.load(track_path, allow_pickle=False) as arrays:
        tracks = arrays["tracks"].astype(np.float32)[anchors]
        source_video = Path(str(arrays["source_video"].item()))
    frames = iio.imread(source_video)
    native_hw = (int(frames.shape[1]), int(frames.shape[2]))
    selected_scopes = [
        ("Fixed latest3350 Top100", pairs(fixed, "top100")),
        ("Fixed latest3350 Bottom100", pairs(fixed, "bottom100")),
        ("TubeTop100", pairs(tube, "tube_top100")),
        ("TubeBottom100", pairs(tube, "tube_bottom100")),
    ]
    per_anchor = {int(row["query_time"]): row for row in report["per_anchor"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for qt in args.query_times:
        if not 0 <= qt < 13:
            raise ValueError(f"query time must be in [0, 13): {qt}")
        rendered = []
        for tk, pixel_frame in enumerate(anchors):
            source = base_panel(frames[pixel_frame], f"Baseline + GT tube | K F{pixel_frame:02d}")
            draw_gt(source, tracks, qt, tk, native_hw)
            panels = [source]
            for label, scope_pairs in selected_scopes:
                panel = base_panel(frames[pixel_frame], label)
                draw_gt(panel, tracks, qt, tk, native_hw)
                draw_predictions(panel, predictions, scope_pairs, qt, tk, native_hw)
                panels.append(panel)
            info = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), 24, dtype=np.uint8)
            row = per_anchor[qt]
            text(info, f"Fixed Q latent t={qt} / pixel F{int(anchors[qt]):02d}", 48, 0.75)
            text(info, "white: query point | green ring: GT at K_t", 88)
            text(info, "colored line: GT tube | cross: 100-head mean argmax", 122)
            text(info, "red dots: deterministic 20-head argmax cloud", 156)
            text(info, f"Fixed Top100 PCK@32: {row['fixed_top100_pck32']:.2f}%", 210)
            text(info, f"Fixed Bottom100 PCK@32: {row['fixed_bottom100_pck32']:.2f}%", 244)
            text(info, f"Top-Bottom: {row['top_minus_bottom_pck32']:+.2f} pp", 278)
            text(info, f"Stage1 decision: {report['summary']['decision'].upper()}", 322, 0.68)
            panels.append(info)
            rendered.append(
                np.concatenate(
                    (
                        np.concatenate(panels[:3], axis=1),
                        np.concatenate(panels[3:], axis=1),
                    ),
                    axis=0,
                )
            )
        output = args.output_dir / f"fixed_query_t{qt:02d}_F{int(anchors[qt]):02d}.mp4"
        iio.imwrite(output, np.stack(rendered), fps=4, codec="libx264", quality=8)
        print(output)


if __name__ == "__main__":
    main()
