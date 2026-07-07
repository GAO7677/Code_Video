from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from code_vjepa_vggt.data.wisa_no_gt_box_dataset import WisaNoGTBoxDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample WISA-80K mp4 cases and emit input jsons compatible with "
            "inspect_stage1b_prepipe_overlay.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/gaoya/dataset/qihoo360-WISA-80K"),
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=None,
        help="Optional override for the actual WISA mp4 directory. Default: <dataset-root>/videos",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--split-train-ratio", type=float, default=0.9)
    parser.add_argument("--split-val-ratio", type=float, default=0.05)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/wisa_prepipe_overlay/input_jsons"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.count <= 0:
        raise ValueError(f"--count must be positive, got {args.count}")

    dataset = WisaNoGTBoxDataset(
        root=args.dataset_root.expanduser().resolve(),
        videos_root=None if args.videos_root is None else args.videos_root.expanduser().resolve(),
        split=args.split,
        resolution=(512, 896),
        num_frames=int(args.num_frames),
        num_context_frames=int(args.num_context_frames),
        sampling_strategy="prefix",
        labels=args.label,
        split_train_ratio=float(args.split_train_ratio),
        split_val_ratio=float(args.split_val_ratio),
    )

    records = list(dataset.samples)
    if len(records) < int(args.count):
        raise RuntimeError(
            f"only found {len(records)} candidate samples for split={args.split}, "
            f"smaller than requested count={args.count}"
        )

    rng = random.Random(int(args.seed))
    rng.shuffle(records)
    selected = records[: int(args.count)]

    json_paths: list[str] = []
    for index, record in enumerate(selected):
        stem = f"{index:02d}_{record.label or 'nolabel'}_{record.video_name.rsplit('.', 1)[0]}"
        json_path = output_dir / f"{stem}.json"
        payload = {
            "input_video": record.video_path,
            "input_caption": record.prompt,
            "sample_key": record.key,
            "video_name": record.video_name,
            "label": record.label,
            "metadata_index": record.metadata_index,
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        json_paths.append(str(json_path))

    manifest = {
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "videos_root": str(
            (args.videos_root.expanduser().resolve() if args.videos_root is not None else (args.dataset_root / "videos").expanduser().resolve())
        ),
        "split": str(args.split),
        "count": int(args.count),
        "seed": int(args.seed),
        "num_frames": int(args.num_frames),
        "num_context_frames": int(args.num_context_frames),
        "split_train_ratio": float(args.split_train_ratio),
        "split_val_ratio": float(args.split_val_ratio),
        "labels": [item for item in (args.label or []) if item],
        "json_paths": json_paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
