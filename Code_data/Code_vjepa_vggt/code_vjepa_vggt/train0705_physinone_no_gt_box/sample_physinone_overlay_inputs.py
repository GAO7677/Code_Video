from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from code_vjepa_vggt.data.phisinone_no_gt_box_dataset import PhysInOneNoGTBoxDataset
from code_vjepa_vggt.inspect_cotracker_vggt_geometry import write_mp4


def _video_cthw_to_uint8_thwc(video_cthw: torch.Tensor) -> torch.Tensor:
    video = video_cthw.detach().cpu().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample PhysInOne cases and emit local mp4/json inputs compatible with "
            "train0705/inspect_stage1b_prepipe_overlay.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/gaoya/dataset/vLAR-PhysInOne/PhysInOneP01-PhysInOneP01"),
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-strategy", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--camera-name", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/phisinone_prepipe_overlay/input_jsons"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if args.count <= 0:
        raise ValueError(f"--count must be positive, got {args.count}")

    dataset = PhysInOneNoGTBoxDataset(
        root=dataset_root,
        split=str(args.split),
        resolution=(int(args.height), int(args.width)),
        num_frames=int(args.num_frames),
        num_context_frames=int(args.num_context_frames),
        sampling_strategy=str(args.sampling_strategy),
        camera_names=args.camera_name,
    )
    if len(dataset) < int(args.count):
        raise RuntimeError(
            f"only found {len(dataset)} candidate samples for split={args.split}, "
            f"smaller than requested count={args.count}"
        )

    all_indices = list(range(len(dataset)))
    rng = random.Random(int(args.seed))
    rng.shuffle(all_indices)
    selected_indices = all_indices[: int(args.count)]

    json_paths: list[str] = []
    exported_cases: list[dict[str, object]] = []
    for rank, dataset_index in enumerate(selected_indices):
        sample = dataset[int(dataset_index)]
        metadata = dict(sample.get("metadata", {}))
        sample_name = str(sample.get("sample_name") or metadata.get("sample_name") or f"sample_{dataset_index:06d}")
        stem = f"{rank:02d}_{sample_name}"

        video_path = videos_dir / f"{stem}.mp4"
        full_video = _video_cthw_to_uint8_thwc(sample["video"]).numpy()
        write_mp4(video_path, full_video, fps=int(args.fps))

        json_path = output_dir / f"{stem}.json"
        payload = {
            "input_video": str(video_path.resolve()),
            "input_caption": str(sample["caption"]),
            "sample_key": str(metadata.get("sample_key", sample_name)),
            "sample_name": sample_name,
            "source_zip": str(metadata.get("source_zip", "")),
            "camera_name": str(metadata.get("camera_name", "")),
            "physics_group": str(metadata.get("physics_group", "")),
            "scene_name": str(metadata.get("scene_name", "")),
            "sampled_frame_indices": metadata.get("sampled_frame_indices", []),
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        json_paths.append(str(json_path))
        exported_cases.append(payload)

    manifest = {
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "count": int(args.count),
        "seed": int(args.seed),
        "num_frames": int(args.num_frames),
        "num_context_frames": int(args.num_context_frames),
        "resolution": [int(args.height), int(args.width)],
        "sampling_strategy": str(args.sampling_strategy),
        "json_paths": json_paths,
        "cases": exported_cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
