#!/usr/bin/env python3
"""Render Top-K reverse 40-step -> 10-step head matches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from render_object_query_step_alignment_head_overlays import (
    PANEL_WIDTH,
    head_map,
    header,
    load_capture,
    normalize_delta,
    normalize_pair,
    read_frames,
    strip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ranking-csv", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    with args.ranking_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) != args.seed or row["branch"] != args.branch:
                continue
            record: dict[str, object] = {
                "block": int(row["block"]),
                "head": int(row["head"]),
                "score": float(row["lora_pck32"]),
                "step10": int(row["best_step10"]),
                "cosine": float(row["cosine"]),
            }
            grouped.setdefault((int(row["step40"]), row["object"]), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: float(item["cosine"]), reverse=True)
        del records[args.top_k :]

    frames = read_frames(args.video)
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
    for step40 in range(40):
        forty_capture = capture(40, step40)
        for object_name in ("object_A", "object_B"):
            sections = []
            for match_index, record in enumerate(grouped.get((step40, object_name), []), start=1):
                block = int(record["block"])
                head = int(record["head"])
                step10 = int(record["step10"])
                forty = head_map(forty_capture, block, head, object_name)
                ten = head_map(capture(10, step10), block, head, object_name)
                forty_normalized, ten_normalized = normalize_pair(forty, ten)
                delta_normalized = normalize_delta(np.abs(forty - ten))
                width = PANEL_WIDTH * len(frames)
                title = (
                    f"#{match_index} L{block:02d}/H{head:02d} | 40-step S{step40:02d} -> "
                    f"10-step S{step10:02d} | cosine={float(record['cosine']):.6f} | "
                    f"PCK@32={float(record['score']):.3f}"
                )
                sections.extend(
                    [
                        header(width, title),
                        strip(frames, forty_normalized, f"40S S{step40:02d}", cv2.COLORMAP_TURBO),
                        strip(frames, ten_normalized, f"10S S{step10:02d}", cv2.COLORMAP_TURBO),
                        strip(frames, delta_normalized, "ABS DELTA", cv2.COLORMAP_MAGMA),
                    ]
                )
            if not sections:
                continue
            output = np.concatenate(sections, axis=0)
            output_path = args.output_dir / (
                f"seed{args.seed:06d}__step{step40:02d}__{args.branch}__{object_name}"
                "__reverse_best_head_matches.jpg"
            )
            if not cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed to write {output_path}")
            print(output_path)


if __name__ == "__main__":
    main()
