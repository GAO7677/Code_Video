from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


METRICS = (
    ("wmreward_surprise", ("wmreward", "surprise")),
    ("physics_iq_with_context", ("physics_iq_with_context", "score")),
    ("physics_iq_without_context", ("physics_iq_without_context", "score")),
    ("pmf_with_context", ("pmf_with_context", "score")),
    ("pmf_without_context", ("pmf_without_context", "score")),
    ("videophy2", ("videophy2", "score")),
    ("cosmos_reason1", ("cosmos_reason1", "score")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned failure-analysis contact sheets.")
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--physiq-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=280)
    parser.add_argument("--tile-height", type=int, default=160)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def nested_number(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def read_video_samples(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = np.linspace(0, frame_count - 1, count).round().astype(int)
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {index}/{frame_count} from {path}")
        frames.append(frame)
    capture.release()
    return frames


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def annotate(frame: np.ndarray, label: str, time_fraction: float) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 25), (0, 0, 0), thickness=-1)
    cv2.putText(output, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        output,
        f"t={time_fraction:.2f}",
        (output.shape[1] - 58, output.shape[0] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def method_dirs(root: Path) -> dict[str, Path]:
    kubric = root / "train_stage1b_kubric0708"
    prefix = "train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z_step-"
    suffix = "_steps40_512x896_ctx08_49f_defaultnegprompt"
    return {
        "PhysRVG": root / "physRVG_steps40_512x896_08_49f",
        "v3-3500 legacy": kubric / f"{prefix}003500{suffix}",
        "v3-3500 temporal": kubric / f"{prefix}003500{suffix}_temporal_sam2",
        "v3-4000 temporal": kubric / f"{prefix}004000{suffix}_temporal_sam2",
        "v3-4500 temporal": kubric / f"{prefix}004500{suffix}_temporal_sam2",
    }


def main() -> None:
    args = parse_args()
    root = args.physiq_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = method_dirs(root)
    cases = [line.strip() for line in args.case_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    metric_rows: list[dict[str, Any]] = []

    for case in cases:
        reference_json = methods["v3-3500 temporal"] / f"{case}.json"
        reference_payload = load_json(reference_json)
        source_video = Path(str(reference_payload["source_video"])).expanduser().resolve()
        videos: list[tuple[str, Path]] = [("GT", source_video)]
        videos.extend((label, directory / f"{case}.mp4") for label, directory in methods.items())
        overlay = methods["v3-3500 temporal"] / f"{case}_input_prepipe_overlay.mp4"
        if overlay.is_file():
            videos.append(("3500 temporal query overlay", overlay))

        rows: list[np.ndarray] = []
        for label, video_path in videos:
            sampled = read_video_samples(video_path, int(args.frames))
            tiles = [
                annotate(
                    letterbox(frame, int(args.tile_width), int(args.tile_height)),
                    label,
                    index / max(1, len(sampled) - 1),
                )
                for index, frame in enumerate(sampled)
            ]
            rows.append(np.concatenate(tiles, axis=1))

        sheet = np.concatenate(rows, axis=0)
        sheet_path = output_dir / f"{case}_contact_sheet.jpg"
        if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {sheet_path}")

        for label, directory in methods.items():
            payload = load_json(directory / f"{case}.json")
            row: dict[str, Any] = {"case": case, "method": label}
            for metric_name, metric_path in METRICS:
                row[metric_name] = nested_number(payload, metric_path)
            metric_rows.append(row)

        print(sheet_path)

    csv_path = output_dir / "selected_case_metrics.csv"
    fieldnames = ["case", "method", *(name for name, _ in METRICS)]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)
    print(csv_path)


if __name__ == "__main__":
    main()
