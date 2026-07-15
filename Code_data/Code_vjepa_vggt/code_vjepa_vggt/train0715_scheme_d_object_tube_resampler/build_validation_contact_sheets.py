#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VARIANTS = (
    ("baseline 1.0x", "baseline"),
    ("zero object", "no_object_context"),
    ("object 0.75x", "object_residual_0p75x"),
    ("object 1.5x", "object_residual_1p5x"),
)
FRAME_INDICES = (0, 7, 8, 16, 24, 32, 40, 48)


def read_selected_frames(path: Path, indices: tuple[int, ...]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != 49:
        capture.release()
        raise ValueError(f"expected 49 frames in {path}, got {frame_count}")
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot read frame {index} from {path}")
        frames.append(frame)
    capture.release()
    return frames


def compose_sheet(
    rows: list[tuple[str, list[np.ndarray]]],
    *,
    frame_indices: tuple[int, ...],
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    label_width = 180
    header_height = 34
    canvas = np.full(
        (
            header_height + len(rows) * tile_height,
            label_width + len(frame_indices) * tile_width,
            3,
        ),
        24,
        dtype=np.uint8,
    )
    for column, frame_index in enumerate(frame_indices):
        x = label_width + column * tile_width
        cv2.putText(
            canvas,
            f"frame {frame_index}",
            (x + max((tile_width - 78) // 2, 5), 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    for row_index, (label, frames) in enumerate(rows):
        if len(frames) != len(frame_indices):
            raise ValueError(f"{label} has {len(frames)} frames, expected {len(frame_indices)}")
        y = header_height + row_index * tile_height
        cv2.putText(
            canvas,
            label,
            (12, y + tile_height // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        for column, frame in enumerate(frames):
            x = label_width + column * tile_width
            resized = cv2.resize(
                frame,
                (tile_width, tile_height),
                interpolation=cv2.INTER_AREA,
            )
            canvas[y : y + tile_height, x : x + tile_width] = resized
            if frame_indices[column] == 7:
                cv2.line(
                    canvas,
                    (x + tile_width - 1, y),
                    (x + tile_width - 1, y + tile_height - 1),
                    (0, 215, 255),
                    2,
                )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tile-width", type=int, default=224)
    parser.add_argument("--tile-height", type=int, default=128)
    args = parser.parse_args()

    root = args.validation_root.expanduser().resolve()
    output_dir = (
        root / "_analysis_contact_sheets_combined"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = root / "baseline" / "results"
    baseline_paths = sorted(baseline_dir.glob("*.mp4"))
    if not baseline_paths:
        raise ValueError(f"no baseline videos in {baseline_dir}")

    manifest: dict[str, object] = {
        "validation_root": str(root),
        "frame_indices": list(FRAME_INDICES),
        "variants": [directory for _, directory in VARIANTS],
        "contact_sheets": [],
    }
    for baseline_path in baseline_paths:
        rows: list[tuple[str, list[np.ndarray]]] = []
        for label, directory in VARIANTS:
            video_path = root / directory / "results" / baseline_path.name
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            rows.append((label, read_selected_frames(video_path, FRAME_INDICES)))
        sheet = compose_sheet(
            rows,
            frame_indices=FRAME_INDICES,
            tile_width=int(args.tile_width),
            tile_height=int(args.tile_height),
        )
        output_path = output_dir / f"{baseline_path.stem}.jpg"
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write {output_path}")
        manifest["contact_sheets"].append(str(output_path))
        print(output_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
