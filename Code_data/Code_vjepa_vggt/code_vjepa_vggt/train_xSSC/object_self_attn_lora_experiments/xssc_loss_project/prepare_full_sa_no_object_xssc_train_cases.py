#!/usr/bin/env python3
"""Export deterministic samples from the exact three-source training mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from code_vjepa_vggt.data.mixed_replay_no_gt_box_dataset import (
    KubricReplayNoGTBoxDataset,
    OpenVidNoGTBoxDataset,
)
from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
    PyBullet0713NoGTBoxDataset,
)


DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "full_sa_no_object_xssc_loss_train_cases"
)
FFMPEG = "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
HEIGHT = 512
WIDTH = 896
NUM_FRAMES = 49
CONTEXT_FRAMES = 8
FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_datasets() -> dict[str, object]:
    resolution = (HEIGHT, WIDTH)
    return {
        "pybullet": PyBullet0713NoGTBoxDataset(
            root="/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5",
            split="train",
            resolution=resolution,
            num_frames=NUM_FRAMES,
            num_context_frames=CONTEXT_FRAMES,
            sampling_strategy="prefix",
        ),
        "kubric": KubricReplayNoGTBoxDataset(
            root="/data/gaoya/dataset/nnsriram97-phyco_kubric",
            split="train",
            resolution=resolution,
            num_frames=NUM_FRAMES,
            num_context_frames=CONTEXT_FRAMES,
            index_num_frames=69,
            index_num_context_frames=20,
            sampling_strategy="prefix",
            seed=42,
            cache_root="/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset",
        ),
        "openvid": OpenVidNoGTBoxDataset(
            root="/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train",
            resolution=resolution,
            num_frames=NUM_FRAMES,
            num_context_frames=CONTEXT_FRAMES,
        ),
    }


def to_uint8(video) -> np.ndarray:
    frames = video.detach().float().cpu().permute(1, 2, 3, 0).numpy()
    frames = np.clip((frames + 1.0) * 127.5, 0, 255)
    return np.rint(frames).astype(np.uint8)


def write_video(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.run(command, input=frames.tobytes(), check=True)
    if process.returncode != 0 or not path.is_file():
        raise RuntimeError(f"Failed to write {path}")


def sample_key(source: str, index: int, metadata: dict) -> str:
    raw = str(metadata.get("sample_key") or f"row_{index:06d}")
    leaf = raw.replace("/", "_").replace(" ", "_")
    return f"{source}_{index:06d}_{leaf}"[:180]


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    input_root = output_root / "inputs"
    media_root = output_root / "training_media"
    manifest_path = output_root / "cases.json"
    if manifest_path.is_file() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inference"]["gpu"] = int(args.gpu)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(manifest_path)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets()
    selected = {
        "pybullet": [0, 539, 1078],
        "kubric": [0, 38092, 76184],
        "openvid": [0, 17833, 35666],
    }
    records = []
    json_paths = []
    for source, indices in selected.items():
        dataset = datasets[source]
        for index in indices:
            sample = dataset[index]
            metadata = dict(sample.get("metadata", {}))
            case_id = sample_key(source, index, metadata)
            frames = to_uint8(sample["video"])
            case_media = media_root / case_id
            gt_path = case_media / "gt_49f.mp4"
            context_path = case_media / "context_08f.mp4"
            image_path = case_media / "first_frame.jpg"
            write_video(gt_path, frames)
            write_video(context_path, frames[:CONTEXT_FRAMES])
            Image.fromarray(frames[0]).save(image_path, quality=95)

            payload = {
                "source_video": str(gt_path),
                "input_video": str(context_path),
                "input_image": str(image_path),
                "input_caption": str(sample["caption"]),
                "training_dataset_source": source,
                "training_dataset_index": int(index),
                "original_video_path": str(sample["video_path"]),
            }
            input_json = input_root / f"{case_id}.json"
            input_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            json_paths.append(input_json)
            records.append(
                {
                    "case_id": case_id,
                    "source": source,
                    "source_index": int(index),
                    "prompt": str(sample["caption"]),
                    "original_video_path": str(sample["video_path"]),
                    "input_json": str(input_json),
                    "gt_video": str(gt_path),
                    "context_video": str(context_path),
                    "frame_indices": metadata.get(
                        "sampled_frame_indices", list(range(NUM_FRAMES))
                    ),
                }
            )

    (input_root / "cases.txt").write_text(
        "".join(f"{path}\n" for path in json_paths), encoding="utf-8"
    )
    manifest = {
        "experiment": "full_sa_no_object_xssc_loss_dinov3_movic_step50000",
        "selection_policy": "three deterministic, spread indices per active training source",
        "training_mixture": {"pybullet": 0.3, "kubric": 0.3, "openvid": 0.4},
        "inference": {
            "checkpoints": [500, 1000],
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "context_frames": CONTEXT_FRAMES,
            "num_inference_steps": 40,
            "fps": FPS,
            "gpu": int(args.gpu),
        },
        "cases": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
