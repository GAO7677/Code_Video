#!/usr/bin/env python3
"""Render the Stage-5 cross-object latent-token overlap audit onto baseline videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
    "visual_samples/attention_zero_seed47326/cases_other10_6seeds_latest.json"
)
TRACKS_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
    "visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1"
)
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage5_token_overlap_audit"
)
SELECTED_SEEDS = {13248, 47326, 90094}
LATENT_T, GRID_H, GRID_W = 13, 22, 40
FRAME_STRIDE = 4
OBJECT_COLORS_BGR = (
    (255, 212, 0),   # cyan
    (56, 189, 255),  # amber
    (255, 105, 207), # violet
    (139, 214, 91),  # green
    (235, 142, 87),
    (104, 114, 255),
)
OVERLAP_BGR = (70, 55, 255)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _track_path(sample: dict[str, Any]) -> Path:
    return (
        TRACKS_ROOT
        / str(sample["case"])
        / f"seed_{int(sample['seed']):05d}"
        / "frozen_baseline_tracks"
        / "tracks.npz"
    )


def token_partition(track_path: Path) -> dict[str, Any]:
    with np.load(track_path) as arrays:
        tracks = arrays["tracks"][arrays["anchor_pixel_frames"]].astype(np.float32)
        names = [str(value) for value in arrays["region_names"].tolist()]
        starts = arrays["point_starts"].astype(np.int64)
        ends = arrays["point_ends"].astype(np.int64)
        pixel_h = int(arrays["pixel_height"])
        pixel_w = int(arrays["pixel_width"])
        anchor_frames = arrays["anchor_pixel_frames"].astype(np.int64)
    if tracks.shape[0] != LATENT_T:
        raise RuntimeError(f"expected {LATENT_T} latent anchors, got {tracks.shape}")

    by_object: dict[str, list[set[int]]] = {}
    for name, start, end in zip(names, starts, ends):
        points = tracks[:, start:end]
        x = np.floor(points[..., 0] * GRID_W / pixel_w).astype(np.int64)
        y = np.floor(points[..., 1] * GRID_H / pixel_h).astype(np.int64)
        x = np.clip(x, 0, GRID_W - 1)
        y = np.clip(y, 0, GRID_H - 1)
        by_object[name] = [
            {int(row * GRID_W + col) for row, col in zip(y[t], x[t])}
            for t in range(LATENT_T)
        ]

    events: list[dict[str, Any]] = []
    for latent_t in range(LATENT_T):
        owners_by_spatial: dict[int, list[str]] = {}
        for name in names:
            for spatial in by_object[name][latent_t]:
                owners_by_spatial.setdefault(spatial, []).append(name)
        for spatial, owners in sorted(owners_by_spatial.items()):
            if len(owners) < 2:
                continue
            row, col = divmod(spatial, GRID_W)
            events.append(
                {
                    "latent_t": latent_t,
                    "anchor_frame": int(anchor_frames[latent_t]),
                    "grid_y": row,
                    "grid_x": col,
                    "spatial_token": spatial,
                    "global_token": latent_t * GRID_H * GRID_W + spatial,
                    "objects": sorted(owners),
                }
            )
    return {
        "names": names,
        "by_object": by_object,
        "events": events,
        "anchor_frames": anchor_frames.tolist(),
        "pixel_hw": [pixel_h, pixel_w],
    }


def _draw_cell(
    frame: np.ndarray,
    spatial: int,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    fill_alpha: float = 0.10,
) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    row, col = divmod(int(spatial), GRID_W)
    x0, x1 = int(round(col * width / GRID_W)), int(round((col + 1) * width / GRID_W))
    y0, y1 = int(round(row * height / GRID_H)), int(round((row + 1) * height / GRID_H))
    if fill_alpha > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), color, thickness, cv2.LINE_AA)
    return x0, y0, x1, y1


def _label(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = origin
    x = max(4, min(x, frame.shape[1] - width - 8))
    y = max(height + 8, min(y, frame.shape[0] - baseline - 5))
    cv2.rectangle(frame, (x - 3, y - height - 4), (x + width + 3, y + baseline + 3), (5, 9, 15), -1)
    cv2.putText(frame, text, (x, y), font, scale, (248, 251, 255), 1, cv2.LINE_AA)


def render_overlay(
    baseline: Path,
    output: Path,
    partition: dict[str, Any],
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(baseline))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open baseline video: {baseline}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (height, width) != tuple(partition["pixel_hw"]):
        raise RuntimeError(
            f"video is {(height, width)}, track coordinates use {partition['pixel_hw']}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_avi = output.with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(
        str(temporary_avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create temporary overlay video: {temporary_avi}")

    events_by_t: dict[int, list[dict[str, Any]]] = {}
    for event in partition["events"]:
        events_by_t.setdefault(int(event["latent_t"]), []).append(event)
    object_colors = {
        name: OBJECT_COLORS_BGR[index % len(OBJECT_COLORS_BGR)]
        for index, name in enumerate(partition["names"])
    }

    written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            latent_t = min(LATENT_T - 1, max(0, (written + FRAME_STRIDE // 2) // FRAME_STRIDE))
            events = events_by_t.get(latent_t, [])
            overlap_spatial = {int(event["spatial_token"]) for event in events}
            for name in partition["names"]:
                for spatial in sorted(partition["by_object"][name][latent_t]):
                    if spatial in overlap_spatial:
                        continue
                    _draw_cell(frame, spatial, object_colors[name])
            for event in events:
                x0, y0, x1, y1 = _draw_cell(
                    frame,
                    int(event["spatial_token"]),
                    OVERLAP_BGR,
                    thickness=3,
                    fill_alpha=0.52,
                )
                cv2.line(frame, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (x1 - 3, y0 + 2), (x0 + 2, y1 - 3), (255, 255, 255), 2, cv2.LINE_AA)
                _label(
                    frame,
                    f"T{event['global_token']} {'+'.join(event['objects'])}",
                    (x0, y0 - 5),
                    scale=0.46,
                )

            anchor = int(partition["anchor_frames"][latent_t])
            _label(
                frame,
                f"F{written:02d} -> R{latent_t:02d} / anchor F{anchor:02d} | overlap {len(events)}",
                (16, 31),
                scale=0.64,
            )
            legend_x = 16
            for name in partition["names"]:
                cv2.rectangle(frame, (legend_x, 45), (legend_x + 18, 63), object_colors[name], -1)
                _label(frame, name, (legend_x + 24, 61), scale=0.46)
                legend_x += 126
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
    if written != frame_count:
        raise RuntimeError(f"decoded {written}/{frame_count} frames from {baseline}")

    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(temporary_avi),
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    temporary_avi.unlink(missing_ok=True)
    return {"fps": fps, "frame_count": written, "height": height, "width": width}


def build(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = [
        row
        for row in payload["samples"]
        if int(row["seed"]) in SELECTED_SEEDS
    ]
    units: list[dict[str, Any]] = []
    total_events = 0
    for sample in samples:
        track_path = _track_path(sample)
        if not track_path.is_file():
            raise FileNotFoundError(track_path)
        partition = token_partition(track_path)
        if not partition["events"]:
            continue
        case, seed = str(sample["case"]), int(sample["seed"])
        output = args.output_root / "overlays" / case / f"seed_{seed:05d}" / "token_overlap.mp4"
        if args.overwrite or not output.is_file():
            video_meta = render_overlay(Path(sample["baseline_video"]), output, partition)
        else:
            capture = cv2.VideoCapture(str(output))
            video_meta = {
                "fps": float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
                "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            }
            capture.release()
        counts_by_t = [0] * LATENT_T
        for event in partition["events"]:
            counts_by_t[int(event["latent_t"])] += 1
        pair_counts: dict[str, int] = {}
        for event in partition["events"]:
            for left, right in combinations(event["objects"], 2):
                key = f"{left}+{right}"
                pair_counts[key] = pair_counts.get(key, 0) + 1
        total_events += len(partition["events"])
        units.append(
            {
                "case": case,
                "seed": seed,
                "baseline_video": str(Path(sample["baseline_video"])),
                "overlay_video": str(output),
                "tracks_npz": str(track_path),
                "objects": partition["names"],
                "object_colors_rgb": {
                    name: list(reversed(OBJECT_COLORS_BGR[index % len(OBJECT_COLORS_BGR)]))
                    for index, name in enumerate(partition["names"])
                },
                "overlap_event_count": len(partition["events"]),
                "overlap_latent_count": sum(value > 0 for value in counts_by_t),
                "counts_by_latent": counts_by_t,
                "pair_counts": pair_counts,
                "events": partition["events"],
                "anchor_frames": partition["anchor_frames"],
                "video": video_meta,
            }
        )
    units.sort(key=lambda row: (row["case"], row["seed"]))
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "latent_grid": [LATENT_T, GRID_H, GRID_W],
            "token_id": "global_token = latent_t * (22 * 40) + grid_y * 40 + grid_x",
            "pixel_cell": "each 22x40 spatial token maps to an exact 32x32 output-pixel cell",
            "video_mapping": "each RGB frame uses the nearest frozen latent anchor F00,F04,...,F48",
            "overlap": "two or more tracked object regions quantize to the same latent-video token",
        },
        "selected_case_seed_units": len(samples),
        "overlap_case_seed_units": len(units),
        "overlap_events": total_events,
        "units": units,
    }
    atomic_json(args.output_root / "catalog.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(
        json.dumps(
            {
                "overlap_case_seed_units": result["overlap_case_seed_units"],
                "overlap_events": result["overlap_events"],
                "catalog": str(OUTPUT_ROOT / "catalog.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
