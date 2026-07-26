#!/usr/bin/env python3
"""Build per-case motion queries and trajectories for attention analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TARGET_HEIGHT = 512
TARGET_WIDTH = 896
GRID_HEIGHT = 16
GRID_WIDTH = 28
TEMPORAL_TOKENS = 13


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"cannot read video: {path}")
    return frames


def _center_crop_resize(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = max(TARGET_WIDTH / width, TARGET_HEIGHT / height)
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y0 = max(0, (resized.shape[0] - TARGET_HEIGHT) // 2)
    x0 = max(0, (resized.shape[1] - TARGET_WIDTH) // 2)
    return resized[y0 : y0 + TARGET_HEIGHT, x0 : x0 + TARGET_WIDTH]


def _motion_map(frames: list[np.ndarray], index: int) -> np.ndarray:
    current = cv2.GaussianBlur(frames[index], (5, 5), 0)
    offsets = [value for value in (index - 4, index + 4) if 0 <= value < len(frames)]
    maps = []
    for other_index in offsets:
        other = cv2.GaussianBlur(frames[other_index], (5, 5), 0)
        maps.append(cv2.cvtColor(cv2.absdiff(current, other), cv2.COLOR_BGR2GRAY))
    if not maps:
        return np.zeros(current.shape[:2], dtype=np.float32)
    motion = np.maximum.reduce(maps).astype(np.float32)
    return cv2.GaussianBlur(motion, (9, 9), 0)


def _component_candidates(motion: np.ndarray) -> list[dict[str, float]]:
    interior = motion.copy()
    interior[:16] = 0
    interior[-16:] = 0
    interior[:, :16] = 0
    interior[:, -16:] = 0
    positive = interior[interior > 0]
    if positive.size == 0:
        return []
    threshold = max(8.0, float(np.percentile(positive, 91)))
    binary = (interior >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8)
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates: list[dict[str, float]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < 40 or area > 0.28 * TARGET_HEIGHT * TARGET_WIDTH:
            continue
        cx, cy = (float(value) for value in centroids[label])
        patch = interior[y : y + height, x : x + width]
        energy = float(patch.sum())
        candidates.append(
            {
                "cx": cx,
                "cy": cy,
                "radius": 0.5 * float(max(width, height)),
                "area": float(area),
                "energy": energy,
            }
        )
    return candidates


def _fallback_center(motion: np.ndarray) -> dict[str, float]:
    pooled = cv2.resize(motion, (GRID_WIDTH, GRID_HEIGHT), interpolation=cv2.INTER_AREA)
    row, column = np.unravel_index(int(np.argmax(pooled)), pooled.shape)
    return {
        "cx": (column + 0.5) * TARGET_WIDTH / GRID_WIDTH,
        "cy": (row + 0.5) * TARGET_HEIGHT / GRID_HEIGHT,
        "radius": 32.0,
        "area": 1024.0,
        "energy": float(pooled[row, column]),
    }


def _select_trajectory(frames: list[np.ndarray]) -> list[dict[str, float]]:
    indices = [min(4 * time, len(frames) - 1) for time in range(TEMPORAL_TOKENS)]
    all_candidates = [_component_candidates(_motion_map(frames, index)) for index in indices]
    query_time = 2
    query_options = all_candidates[query_time]
    query = (
        max(query_options, key=lambda item: item["energy"])
        if query_options
        else _fallback_center(_motion_map(frames, indices[query_time]))
    )
    trajectory: list[dict[str, float] | None] = [None] * TEMPORAL_TOKENS
    trajectory[query_time] = query
    gray_frames = [cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY) for index in indices]
    half = 32
    qx = int(round(query["cx"]))
    qy = int(round(query["cy"]))
    qx = min(max(qx, half), TARGET_WIDTH - half)
    qy = min(max(qy, half), TARGET_HEIGHT - half)
    template = gray_frames[query_time][qy - half : qy + half, qx - half : qx + half]
    for direction in (-1, 1):
        previous = dict(query)
        times = (
            range(query_time - 1, -1, -1)
            if direction < 0
            else range(query_time + 1, TEMPORAL_TOKENS)
        )
        for time in times:
            center_x = int(round(previous["cx"]))
            center_y = int(round(previous["cy"]))
            radius = 112
            x0 = max(0, center_x - radius - half)
            y0 = max(0, center_y - radius - half)
            x1 = min(TARGET_WIDTH, center_x + radius + half)
            y1 = min(TARGET_HEIGHT, center_y + radius + half)
            search = gray_frames[time][y0:y1, x0:x1]
            if search.shape[0] < 2 * half or search.shape[1] < 2 * half:
                trajectory[time] = dict(previous)
                continue
            response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, match_score, _, match = cv2.minMaxLoc(response)
            matched_x = float(x0 + match[0] + half)
            matched_y = float(y0 + match[1] + half)
            options = all_candidates[time]
            if options:
                nearby = min(
                    options,
                    key=lambda item: np.hypot(
                        item["cx"] - matched_x, item["cy"] - matched_y
                    ),
                )
                if np.hypot(
                    nearby["cx"] - matched_x, nearby["cy"] - matched_y
                ) <= 48:
                    matched_x = 0.6 * matched_x + 0.4 * nearby["cx"]
                    matched_y = 0.6 * matched_y + 0.4 * nearby["cy"]
            previous = {
                "cx": matched_x,
                "cy": matched_y,
                "radius": float(query["radius"]),
                "area": float(query["area"]),
                "energy": float(match_score),
            }
            trajectory[time] = previous
    known = [index for index, item in enumerate(trajectory) if item is not None]
    for time, item in enumerate(trajectory):
        if item is not None:
            continue
        lower = max((index for index in known if index < time), default=known[0])
        upper = min((index for index in known if index > time), default=known[-1])
        weight = 0.0 if lower == upper else (time - lower) / (upper - lower)
        trajectory[time] = {
            key: float(
                trajectory[lower][key] * (1.0 - weight)
                + trajectory[upper][key] * weight
            )
            for key in ("cx", "cy", "radius", "area", "energy")
        }
    return [item for item in trajectory if item is not None]


def _query_coords(center: dict[str, float]) -> list[list[int]]:
    row = int(center["cy"] * GRID_HEIGHT / TARGET_HEIGHT)
    column = int(center["cx"] * GRID_WIDTH / TARGET_WIDTH)
    row0 = min(max(row - 1, 0), GRID_HEIGHT - 2)
    column0 = min(max(column - 1, 0), GRID_WIDTH - 2)
    return [
        [2, row0, column0],
        [2, row0, column0 + 1],
        [2, row0 + 1, column0],
        [2, row0 + 1, column0 + 1],
    ]


def _draw_preview(
    frame: np.ndarray,
    query_coords: list[list[int]],
    trajectory: list[dict[str, float]],
    label: str,
) -> np.ndarray:
    output = frame.copy()
    points = np.asarray(
        [[round(item["cx"]), round(item["cy"])] for item in trajectory],
        dtype=np.int32,
    )
    cv2.polylines(output, [points], False, (0, 220, 255), 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(output, (int(x), int(y)), 4, (0, 220, 255), -1)
    rows = [item[1] for item in query_coords]
    columns = [item[2] for item in query_coords]
    x0 = min(columns) * TARGET_WIDTH // GRID_WIDTH
    x1 = (max(columns) + 1) * TARGET_WIDTH // GRID_WIDTH
    y0 = min(rows) * TARGET_HEIGHT // GRID_HEIGHT
    y1 = (max(rows) + 1) * TARGET_HEIGHT // GRID_HEIGHT
    cv2.rectangle(output, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.putText(
        output,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def build_query_map(input_list: Path, output_dir: Path) -> dict[str, Any]:
    json_paths = [
        Path(line.strip()).expanduser().resolve()
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    json_paths = list(dict.fromkeys(json_paths))
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    cases: dict[str, Any] = {}
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source_video = Path(payload["source_video"]).expanduser().resolve()
        frames = [_center_crop_resize(frame) for frame in _read_video(source_video)]
        trajectory = _select_trajectory(frames)
        coords = _query_coords(trajectory[2])
        preview = _draw_preview(
            frames[min(8, len(frames) - 1)], coords, trajectory, json_path.stem
        )
        preview_path = preview_dir / f"{json_path.stem}.jpg"
        cv2.imwrite(str(preview_path), preview)
        cases[json_path.stem] = {
            "input_json": str(json_path),
            "source_video": str(source_video),
            "query_coords": coords,
            "query_coords_text": ",".join(":".join(map(str, item)) for item in coords),
            "query_video_frame": 8,
            "trajectory": trajectory,
            "frame_shape": [TARGET_HEIGHT, TARGET_WIDTH],
            "preview": str(preview_path),
        }
    unique_list = output_dir / "test_5_unique.txt"
    unique_list.write_text(
        "".join(f"{path}\n" for path in json_paths), encoding="utf-8"
    )
    result = {
        "input_list": str(input_list),
        "unique_input_list": str(unique_list),
        "target_shape": [TARGET_HEIGHT, TARGET_WIDTH],
        "grid": [TEMPORAL_TOKENS, GRID_HEIGHT, GRID_WIDTH],
        "cases": cases,
    }
    (output_dir / "motion_query_map.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_query_map(
        args.input_list.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "cases": len(result["cases"]),
                "query_map": str(Path(args.output_dir).resolve() / "motion_query_map.json"),
                "unique_list": result["unique_input_list"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
