#!/usr/bin/env python3
"""Render the best 10-step/40-step object-query head matches on video frames."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PANEL_WIDTH = 240
PANEL_HEIGHT = 137
LABEL_HEIGHT = 24
PAIR_HEADER_HEIGHT = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ranking-csv", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step10", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    anchors = np.rint(np.linspace(0, len(frames) - 1, 13)).astype(np.int64)
    return [frames[int(index)] for index in anchors]


def read_matches(args: argparse.Namespace) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    with args.ranking_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["seed"]) != args.seed
                or row["branch"] != args.branch
                or int(row["step10"]) != args.step10
            ):
                continue
            record: dict[str, object] = {
                "object": row["object"],
                "rank": int(row["rank"]),
                "block": int(row["block"]),
                "head": int(row["head"]),
                "score": float(row.get("lora_pck32", row.get("baseline_pck32", "nan"))),
                "step40": int(row["best_step40"]),
                "cosine": float(row["cosine"]),
            }
            grouped.setdefault(row["object"], []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: float(item["cosine"]), reverse=True)
        del records[args.top_k :]
    return grouped


def load_capture(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def head_map(capture: dict[str, np.ndarray], block: int, head: int, object_name: str) -> np.ndarray:
    head_indices = np.flatnonzero(
        (capture["blocks"].astype(np.int64) == block)
        & (capture["heads"].astype(np.int64) == head)
    )
    object_indices = np.flatnonzero(capture["region_names"].astype(str) == object_name)
    if head_indices.size != 1 or object_indices.size != 1:
        raise RuntimeError(f"Missing unique L{block:02d}/H{head:02d} {object_name}")
    return capture["attention"][int(head_indices[0]), int(object_indices[0])].astype(np.float32)


def normalize_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized_first = np.zeros_like(first, dtype=np.float32)
    normalized_second = np.zeros_like(second, dtype=np.float32)
    for frame_index in range(first.shape[0]):
        values = np.concatenate((first[frame_index].ravel(), second[frame_index].ravel()))
        vmax = float(np.quantile(values, 0.99))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(values.max(initial=0.0))
        if vmax > 0:
            normalized_first[frame_index] = np.clip(first[frame_index] / vmax, 0.0, 1.0)
            normalized_second[frame_index] = np.clip(second[frame_index] / vmax, 0.0, 1.0)
    return normalized_first, normalized_second


def normalize_delta(delta: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(delta, dtype=np.float32)
    for frame_index in range(delta.shape[0]):
        vmax = float(np.quantile(delta[frame_index], 0.99))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(delta[frame_index].max(initial=0.0))
        if vmax > 0:
            normalized[frame_index] = np.clip(delta[frame_index] / vmax, 0.0, 1.0)
    return normalized


def overlay(frame: np.ndarray, values: np.ndarray, label: str, colormap: int) -> np.ndarray:
    base = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    heat = cv2.resize(values, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(np.uint8(np.clip(heat, 0.0, 1.0) * 255), colormap)
    alpha = (0.18 + 0.62 * heat)[..., None]
    blended = np.uint8(np.clip(base * (1.0 - alpha) + color * alpha, 0, 255))
    panel = np.full((PANEL_HEIGHT + LABEL_HEIGHT, PANEL_WIDTH, 3), 245, np.uint8)
    panel[LABEL_HEIGHT:] = blended
    cv2.putText(panel, label, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 35, 30), 1, cv2.LINE_AA)
    return panel


def strip(frames: list[np.ndarray], maps: np.ndarray, prefix: str, colormap: int) -> np.ndarray:
    return np.concatenate(
        [overlay(frame, values, f"{prefix} K{index:02d}", colormap) for index, (frame, values) in enumerate(zip(frames, maps))],
        axis=1,
    )


def header(width: int, text: str) -> np.ndarray:
    canvas = np.full((PAIR_HEADER_HEIGHT, width, 3), (228, 221, 208), np.uint8)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 55, 45), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    frames = read_frames(args.video)
    matches = read_matches(args)
    seed_root = args.root / "seeds" / f"seed_{args.seed:06d}"
    cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    def capture(steps: int, step: int) -> dict[str, np.ndarray]:
        key = (steps, step)
        if key not in cache:
            cache[key] = load_capture(
                seed_root / f"steps{steps}" / "captures" / f"step_{step:02d}__{args.branch}.npz"
            )
        return cache[key]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ten_capture = capture(10, args.step10)
    for object_name, records in matches.items():
        sections = []
        for match_index, record in enumerate(records, start=1):
            block = int(record["block"])
            head = int(record["head"])
            step40 = int(record["step40"])
            ten = head_map(ten_capture, block, head, object_name)
            forty = head_map(capture(40, step40), block, head, object_name)
            ten_normalized, forty_normalized = normalize_pair(ten, forty)
            delta_normalized = normalize_delta(np.abs(ten - forty))
            width = PANEL_WIDTH * len(frames)
            title = (
                f"#{match_index} L{block:02d}/H{head:02d} | 10-step S{args.step10:02d} -> "
                f"40-step S{step40:02d} | cosine={float(record['cosine']):.6f} | "
                f"PCK@32={float(record['score']):.3f}"
            )
            sections.extend(
                [
                    header(width, title),
                    strip(frames, ten_normalized, f"10S S{args.step10:02d}", cv2.COLORMAP_TURBO),
                    strip(frames, forty_normalized, f"40S S{step40:02d}", cv2.COLORMAP_TURBO),
                    strip(frames, delta_normalized, "ABS DELTA", cv2.COLORMAP_MAGMA),
                ]
            )
        if not sections:
            continue
        output = np.concatenate(sections, axis=0)
        output_path = args.output_dir / (
            f"seed{args.seed:06d}__step{args.step10:02d}__{args.branch}__{object_name}"
            "__best_head_matches.jpg"
        )
        if not cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Failed to write {output_path}")
        print(output_path)


if __name__ == "__main__":
    main()
