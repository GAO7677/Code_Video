#!/usr/bin/env python3
"""Extract representative GT/reconstruction frames from a SAVi audit gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", default=[0, 4, 9])
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def read_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    wanted = set(indices)
    frames = {}
    index = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
            wanted.remove(index)
        index += 1
    capture.release()
    if wanted:
        raise RuntimeError(f"Missing frames {sorted(wanted)} in {path}")
    return frames


def title_bar(width: int, lines: list[str], height: int = 56) -> np.ndarray:
    bar = np.full((height, width, 3), 247, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            bar,
            line,
            (10, 20 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    return bar


def extract_panels(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    body = frame[66:]
    panel_width = body.shape[1] // 4
    return tuple(
        body[:, index * panel_width : (index + 1) * panel_width]
        for index in range(4)
    )


def make_sheet(
    root: Path,
    samples: list[dict],
    frame_indices: list[int],
    scale: int,
    gt_recon_only: bool,
) -> np.ndarray:
    rows = []
    for sample in samples:
        frames = read_frames(root / sample["video"], frame_indices)
        cells = []
        for frame_index in frame_indices:
            gt, reconstruction, error, slots = extract_panels(frames[frame_index])
            panels = (gt, reconstruction) if gt_recon_only else (gt, reconstruction, error, slots)
            body = np.concatenate(panels, axis=1)
            body = cv2.resize(
                body,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            heading = title_bar(
                body.shape[1],
                [f"t={frame_index}: " + ("GT | reconstruction" if gt_recon_only else "GT | reconstruction | MSE | slots")],
                height=34,
            )
            cells.append(np.concatenate([heading, body], axis=0))
        row_body = np.concatenate(cells, axis=1)
        label = title_bar(
            row_body.shape[1],
            [f"{sample['kind']} / {sample['sample_id']}", f"global MSE={sample['metrics']['global_loss']:.6f}"],
        )
        rows.append(np.concatenate([label, row_body], axis=0))
    return np.concatenate(rows, axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    root = args.summary.parent
    for kind in ("kubric_val", "physiq"):
        samples = [sample for sample in summary["samples"] if sample["kind"] == kind]
        for gt_recon_only, suffix in ((True, "gt_reconstruction"), (False, "full_audit")):
            sheet = make_sheet(root, samples, args.frame_indices, args.scale, gt_recon_only)
            output_path = args.output_dir / f"{kind}_{suffix}_contact_sheet.jpg"
            if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Failed to write {output_path}")
            print(output_path)


if __name__ == "__main__":
    main()
