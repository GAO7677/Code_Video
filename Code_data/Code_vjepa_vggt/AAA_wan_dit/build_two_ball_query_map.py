#!/usr/bin/env python3
"""Build two identity-locked Wan query tracks from saved SAM2 candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


TIMES, GRID_H, GRID_W = 13, 16, 28
FRAME_H, FRAME_W = 512, 896


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-masks", type=Path, required=True)
    parser.add_argument("--generated-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--overlap-threshold", type=float, default=0.10)
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def _query_tokens(
    masks: np.ndarray, overlap_threshold: float, max_tokens: int
) -> tuple[list[list[list[int]]], list[dict[str, float | int | bool | None]]]:
    coords_per_time = []
    trajectory = []
    for latent_time in range(TIMES):
        frame_index = 4 * latent_time
        mask = masks[frame_index].astype(np.float32)
        if int(mask.sum()) == 0:
            coords_per_time.append([])
            trajectory.append(
                {
                    "valid": False,
                    "video_frame": frame_index,
                    "cx": None,
                    "cy": None,
                    "area": 0,
                }
            )
            continue
        pooled = cv2.resize(
            mask, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA
        )
        candidates = np.argwhere(pooled >= float(overlap_threshold))
        if len(candidates) == 0:
            candidates = np.asarray(
                [np.unravel_index(int(np.argmax(pooled)), pooled.shape)]
            )
        ordered = sorted(
            candidates.tolist(),
            key=lambda rc: float(pooled[int(rc[0]), int(rc[1])]),
            reverse=True,
        )[: int(max_tokens)]
        coords_per_time.append(
            [
                [latent_time, int(row), int(column)]
                for row, column in ordered
            ]
        )
        ys, xs = np.nonzero(mask)
        trajectory.append(
            {
                "valid": True,
                "video_frame": frame_index,
                "cx": float(xs.mean()),
                "cy": float(ys.mean()),
                "area": int(mask.sum()),
                "radius": math.sqrt(float(mask.sum()) / math.pi),
            }
        )
    return coords_per_time, trajectory


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (FRAME_H, FRAME_W):
            raise ValueError(f"unexpected frame shape: {frame.shape}")
        frames.append(frame)
    capture.release()
    if len(frames) != 49:
        raise ValueError(f"expected 49 video frames, found {len(frames)}")
    return frames


def _rect(coords: list[list[int]]) -> tuple[int, int, int, int] | None:
    if not coords:
        return None
    rows = [int(item[1]) for item in coords]
    columns = [int(item[2]) for item in coords]
    return (
        min(columns) * 32,
        min(rows) * 32,
        (max(columns) + 1) * 32,
        (max(rows) + 1) * 32,
    )


def _write_preview(
    frames: list[np.ndarray],
    tracks: list[dict],
    output_path: Path,
) -> None:
    tiles = []
    colors = ((70, 230, 70), (220, 80, 230))
    for latent_time in range(TIMES):
        frame_index = 4 * latent_time
        tile = cv2.resize(frames[frame_index], (448, 256))
        for track_index, track in enumerate(tracks):
            rect = _rect(track["query_coords_per_time"][latent_time])
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            cv2.rectangle(
                tile,
                (x0 // 2, y0 // 2),
                (x1 // 2, y1 // 2),
                colors[track_index],
                2,
            )
        cv2.putText(
            tile,
            f"latent t{latent_time:02d} / source frame {frame_index:02d}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % 4:
        tiles.append(blank)
    sheet = np.vstack(
        [np.hstack(tiles[index : index + 4]) for index in range(0, len(tiles), 4)]
    )
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"failed to write {output_path}")


def main() -> None:
    args = parse_args()
    masks_path = args.candidate_masks.expanduser().resolve()
    video_path = args.generated_video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(masks_path) as arrays:
        candidate_masks = arrays["candidate_masks"].astype(np.uint8)
    if candidate_masks.shape != (2, 49, FRAME_H, FRAME_W):
        raise ValueError(f"unexpected candidate mask shape: {candidate_masks.shape}")

    tracks = []
    for candidate_index, name in enumerate(("ball_A", "ball_B")):
        coords, trajectory = _query_tokens(
            candidate_masks[candidate_index],
            args.overlap_threshold,
            args.max_tokens,
        )
        tracks.append(
            {
                "name": name,
                "candidate_index": candidate_index,
                "identity_policy": "single SAM2 candidate; no cross-instance fallback",
                "query_coords_per_time": coords,
                "query_tokens_per_time": [len(items) for items in coords],
                "valid_query_times": [
                    time for time, items in enumerate(coords) if items
                ],
                "trajectory": trajectory,
            }
        )
    if tracks[0]["valid_query_times"] != list(range(TIMES)):
        raise ValueError("ball A must be valid at all latent times")
    if tracks[1]["valid_query_times"] != list(range(3, TIMES)):
        raise ValueError(
            f"ball B expected at t3..t12, got {tracks[1]['valid_query_times']}"
        )

    preview_path = output_dir / "two_ball_query_preview.jpg"
    _write_preview(_read_video(video_path), tracks, preview_path)
    payload = {
        "case": args.case,
        "generated_video": str(video_path),
        "candidate_masks": str(masks_path),
        "grid": [TIMES, GRID_H, GRID_W],
        "query_source_frame": "frame=4*latent_time",
        "overlap_threshold": float(args.overlap_threshold),
        "max_tokens": int(args.max_tokens),
        "tracks": tracks,
        "preview": str(preview_path),
    }
    output_path = output_dir / "query_map.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[two-ball-query] wrote {output_path}")


if __name__ == "__main__":
    main()
