#!/usr/bin/env python3
"""Track SAM2 object points and map them to per-frame Wan latent query tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


FRAMES = 49
ANCHORS = np.arange(0, FRAMES, 4, dtype=np.int64)
HEIGHT = 512
WIDTH = 896
LATENT_HEIGHT = 16
LATENT_WIDTH = 28
SPATIAL_TOKENS = LATENT_HEIGHT * LATENT_WIDTH
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
COTRACKER_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--video40", type=Path, required=True)
    parser.add_argument("--video10", type=Path, required=True)
    parser.add_argument("--region-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_video(path: Path) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    bgr = []
    while len(bgr) < FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (HEIGHT, WIDTH):
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        bgr.append(frame)
    capture.release()
    if len(bgr) != FRAMES:
        raise RuntimeError(f"Expected {FRAMES} frames, got {len(bgr)}: {path}")
    bgr_array = np.stack(bgr)
    rgb_tensor = np.ascontiguousarray(bgr_array[..., ::-1].transpose(0, 3, 1, 2))
    return bgr_array, rgb_tensor


def color_for_point(index: int, count: int) -> tuple[int, int, int]:
    hue = int(round(179 * index / max(count, 1)))
    hsv = np.uint8([[[hue, 205, 245]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def render_query_tokens(
    frames: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    region_name: str,
    schedule: str,
) -> np.ndarray:
    tiles = []
    point_count = tracks.shape[1]
    for latent_index, pixel_index in enumerate(ANCHORS):
        tile = cv2.resize(frames[pixel_index], (320, 183), interpolation=cv2.INTER_AREA)
        sx, sy = 320.0 / WIDTH, 183.0 / HEIGHT
        visible_count = 0
        token_set = set()
        for point_index in range(point_count):
            if not bool(visibility[pixel_index, point_index]):
                continue
            x, y = tracks[pixel_index, point_index]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            x = float(np.clip(x, 0, WIDTH - 1))
            y = float(np.clip(y, 0, HEIGHT - 1))
            token_x = int(np.clip(np.floor(x * LATENT_WIDTH / WIDTH), 0, LATENT_WIDTH - 1))
            token_y = int(np.clip(np.floor(y * LATENT_HEIGHT / HEIGHT), 0, LATENT_HEIGHT - 1))
            token_set.add((token_y, token_x))
            visible_count += 1
            px, py = int(round(x * sx)), int(round(y * sy))
            x0 = int(round(token_x * 320 / LATENT_WIDTH))
            x1 = int(round((token_x + 1) * 320 / LATENT_WIDTH)) - 1
            y0 = int(round(token_y * 183 / LATENT_HEIGHT))
            y1 = int(round((token_y + 1) * 183 / LATENT_HEIGHT)) - 1
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            color = color_for_point(point_index, point_count)
            cv2.rectangle(tile, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
            cv2.line(tile, (px, py), (cx, cy), color, 1, cv2.LINE_AA)
            cv2.circle(tile, (px, py), 3, color, -1, cv2.LINE_AA)
            cv2.drawMarker(tile, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA)
            cv2.putText(
                tile,
                f"Q({token_y},{token_x})",
                (min(cx + 3, 255), max(cy - 3, 34)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.27,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.rectangle(tile, (0, 0), (319, 29), (244, 240, 230), -1)
        cv2.putText(
            tile,
            f"Q{latent_index:02d}/F{pixel_index:02d}  points {visible_count}  tokens {len(token_set)}",
            (7, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (25, 31, 29),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    body = np.concatenate(tiles, axis=1)
    header = np.full((45, body.shape[1], 3), (237, 232, 219), np.uint8)
    cv2.putText(
        header,
        f"{schedule} CoTracker Dynamic Query Tokens | {region_name} | circle=pixel track, box/cross=16x28 query token",
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.69,
        (22, 38, 31),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, body], axis=0)


def token_indices(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    result = np.full((len(ANCHORS), tracks.shape[1]), -1, dtype=np.int64)
    for latent_index, pixel_index in enumerate(ANCHORS):
        for point_index, (x, y) in enumerate(tracks[pixel_index]):
            if not bool(visibility[pixel_index, point_index]) or not np.isfinite(x + y):
                continue
            token_x = int(np.clip(np.floor(x * LATENT_WIDTH / WIDTH), 0, LATENT_WIDTH - 1))
            token_y = int(np.clip(np.floor(y * LATENT_HEIGHT / HEIGHT), 0, LATENT_HEIGHT - 1))
            result[latent_index, point_index] = (
                latent_index * SPATIAL_TOKENS + token_y * LATENT_WIDTH + token_x
            )
    return result


def main() -> None:
    args = parse_args()
    os.environ["OBJECT_QUERY_REGION_CACHE"] = str(args.region_cache)
    from AAA_my_test.object_query_attention_capture_headwise_pck import pck_query_regions

    regions, _context = pck_query_regions()
    starts, ends, query_parts = [], [], []
    offset = 0
    for region in regions:
        points = np.asarray(region["points"], dtype=np.float32)
        starts.append(offset)
        offset += len(points)
        ends.append(offset)
        query_parts.append(points)
    query_points = np.concatenate(query_parts, axis=0)

    sys.path.insert(0, str(COTRACKER_ROOT))
    from cotracker.predictor import CoTrackerPredictor

    device = torch.device(args.device)
    model = CoTrackerPredictor(
        checkpoint=str(COTRACKER_CHECKPOINT), offline=True
    ).to(device).eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for schedule, path in (("steps40", args.video40), ("steps10", args.video10)):
        frames, video = read_video(path)
        video_tensor = torch.from_numpy(video).float().unsqueeze(0).to(device)
        points = torch.from_numpy(query_points).float().to(device)
        query_times = torch.full((len(points), 1), 4.0, device=device)
        queries = torch.cat((query_times, points), dim=-1).unsqueeze(0)
        with torch.inference_mode():
            tracks_tensor, visibility_tensor = model(video_tensor, queries=queries)
        tracks = tracks_tensor[0].float().cpu().numpy().astype(np.float32)
        visibility = visibility_tensor[0].cpu().numpy().astype(np.bool_)
        schedule_root = args.output_root / schedule
        overlay_root = schedule_root / "query_token_overlays"
        overlay_root.mkdir(parents=True, exist_ok=True)
        all_tokens = token_indices(tracks, visibility)
        np.savez_compressed(
            schedule_root / "dynamic_query_tracks.npz",
            tracks=tracks,
            visibility=visibility,
            anchor_frames=ANCHORS,
            anchor_token_indices=all_tokens,
            query_points=query_points,
            region_names=np.asarray([str(region["name"]) for region in regions]),
            region_phrases=np.asarray([str(region["phrase"]) for region in regions]),
            point_starts=np.asarray(starts, dtype=np.int32),
            point_ends=np.asarray(ends, dtype=np.int32),
            query_pixel_frame=np.int32(4),
            latent_height=np.int32(LATENT_HEIGHT),
            latent_width=np.int32(LATENT_WIDTH),
            source_video=np.asarray(str(path)),
            schedule=np.asarray(schedule),
            seed=np.int32(args.seed),
        )
        for region, start, end in zip(regions, starts, ends):
            name = str(region["name"])
            image_name = f"seed{args.seed:06d}__{schedule}__{name}__query_tokens.jpg"
            image = render_query_tokens(
                frames, tracks[:, start:end], visibility[:, start:end], name, schedule
            )
            if not cv2.imwrite(
                str(overlay_root / image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 94]
            ):
                raise RuntimeError(f"Failed to write {overlay_root / image_name}")
            records.append(
                {
                    "schedule": schedule,
                    "region_name": name,
                    "region_phrase": str(region["phrase"]),
                    "image": image_name,
                    "visible_rate": float(visibility[:, start:end].mean()),
                }
            )
        del video_tensor, tracks_tensor, visibility_tensor
        torch.cuda.empty_cache()
    (args.output_root / "tracks_manifest.json").write_text(
        json.dumps({"seed": args.seed, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
