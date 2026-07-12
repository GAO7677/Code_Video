from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from build_physiq_failure_contact_sheets import annotate, letterbox, read_video_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build object-context scale comparison sheets.")
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--method-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--frames", type=int, default=8)
    return parser.parse_args()


def scale_label(path: Path) -> str:
    marker = "temporal_scale_"
    if marker not in path.name:
        return path.name
    value = path.name.rsplit(marker, 1)[1]
    return f"object scale {int(value) / 100:.2f}"


def main() -> None:
    args = parse_args()
    cases = [line.strip() for line in args.case_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    methods = [Path(line.strip()).resolve() for line in args.method_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.label and len(args.label) != len(methods):
        raise ValueError(f"expected {len(methods)} labels, received {len(args.label)}")
    labels = args.label or [scale_label(method) for method in methods]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        rows = []
        for method, label in zip(methods, labels):
            frames = read_video_samples(method / f"{case}.mp4", int(args.frames))
            tiles = [
                annotate(
                    letterbox(frame, 280, 160),
                    label,
                    index / max(1, len(frames) - 1),
                )
                for index, frame in enumerate(frames)
            ]
            rows.append(np.concatenate(tiles, axis=1))
        output_path = output_dir / f"{case}_scale_comparison.jpg"
        if not cv2.imwrite(str(output_path), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {output_path}")
        print(output_path)


if __name__ == "__main__":
    main()
