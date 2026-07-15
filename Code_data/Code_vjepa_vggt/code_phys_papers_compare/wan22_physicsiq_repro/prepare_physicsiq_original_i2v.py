from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


DEFAULT_SMOKE_IDS = (1, 37, 64, 91, 127, 187)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare last-frame I2V inputs for Physics-IQ Original."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--descriptions-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs="*", default=DEFAULT_SMOKE_IDS)
    parser.add_argument(
        "--image-source",
        choices=("conditioning-last-frame", "switch-frame"),
        default="conditioning-last-frame",
    )
    return parser.parse_args()


def extract_last_frame(video_path: Path, image_path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open conditioning video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(
            f"invalid conditioning metadata: fps={fps}, frames={frame_count}, path={video_path}"
        )

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"failed to read last frame: {video_path}")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"failed to write conditioning image: {image_path}")

    return {
        "conditioning_fps": fps,
        "conditioning_frames": frame_count,
        "conditioning_duration_seconds": frame_count / fps,
        "conditioning_width": width,
        "conditioning_height": height,
        "conditioning_frame_index": frame_count - 1,
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    descriptions_file = args.descriptions_file.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    images_root = output_root / "images"
    manifest_path = output_root / "input_manifest.jsonl"
    summary_path = output_root / "preparation_summary.json"

    conditioning_root = dataset_root / "split-videos" / "conditioning" / "30FPS"
    if not conditioning_root.is_dir():
        raise FileNotFoundError(f"conditioning root not found: {conditioning_root}")

    with descriptions_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows_by_id = {int(row["generated_video_name"][:4]): row for row in rows}

    selected_ids = list(dict.fromkeys(args.ids))
    entries: list[dict[str, object]] = []
    for benchmark_id in selected_ids:
        if benchmark_id not in rows_by_id:
            raise KeyError(f"benchmark id missing from descriptions: {benchmark_id:04d}")

        row = rows_by_id[benchmark_id]
        scenario = row["scenario"]
        _, scenario_suffix = scenario.split("_", 1)
        conditioning_name = (
            f"{benchmark_id:04d}_conditioning-videos_30FPS_{scenario_suffix}"
        )
        conditioning_path = conditioning_root / conditioning_name
        if not conditioning_path.is_file():
            raise FileNotFoundError(f"conditioning video not found: {conditioning_path}")

        if args.image_source == "switch-frame":
            switch_matches = sorted(
                (dataset_root / "switch-frames").glob(f"{benchmark_id:04d}_*")
            )
            if len(switch_matches) != 1:
                raise RuntimeError(
                    f"expected one switch frame for {benchmark_id:04d}, got {switch_matches}"
                )
            image_path = switch_matches[0]
            capture = cv2.VideoCapture(str(conditioning_path))
            metadata = {
                "conditioning_fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "conditioning_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "conditioning_duration_seconds": float(
                    capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    / capture.get(cv2.CAP_PROP_FPS)
                ),
                "conditioning_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "conditioning_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "conditioning_frame_index": int(
                    capture.get(cv2.CAP_PROP_FRAME_COUNT)
                )
                - 1,
            }
            capture.release()
        else:
            image_path = images_root / f"{benchmark_id:04d}_last_frame.png"
            metadata = extract_last_frame(conditioning_path, image_path)
        entry = {
            "benchmark_id": f"{benchmark_id:04d}",
            "scenario": scenario,
            "category": row["category"],
            "prompt": row["description"],
            "prompt_setting": "op",
            "image_source": args.image_source,
            "conditioning_video": str(conditioning_path),
            "conditioning_image": str(image_path),
            "generated_video_name": row["generated_video_name"],
            **metadata,
        }
        entries.append(entry)
        print(
            f"[prepared] {benchmark_id:04d} frames={metadata['conditioning_frames']} "
            f"fps={metadata['conditioning_fps']} image={image_path.name}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "dataset_root": str(dataset_root),
        "descriptions_file": str(descriptions_file),
        "prompt_setting": "op",
        "image_source": args.image_source,
        "num_items": len(entries),
        "benchmark_ids": [entry["benchmark_id"] for entry in entries],
        "manifest": str(manifest_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[summary] wrote {len(entries)} inputs to {manifest_path}")


if __name__ == "__main__":
    main()
