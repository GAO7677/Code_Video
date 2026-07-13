from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from code_vjepa_vggt.data.pybullet_raw_no_gt_box_dataset import (
    PyBulletRawNoGTBoxDataset,
)


FFMPEG = "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"


def write_h264(path: Path, frames: np.ndarray, fps: int = 30) -> None:
    height, width = frames.shape[1:3]
    command = [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.run(command, input=frames.tobytes(), check=True)


def tensor_to_uint8(sample: dict) -> np.ndarray:
    return (
        (sample["video"].permute(1, 2, 3, 0).float().numpy() + 1.0) * 127.5
    ).clip(0, 255).astype(np.uint8)


def annotated(frames: np.ndarray, context_frames: int) -> np.ndarray:
    result = frames.copy()
    for frame_idx, frame in enumerate(result):
        phase = "CONTEXT" if frame_idx < context_frames else "TARGET"
        text = f"train frame {frame_idx:02d} | {phase}"
        cv2.rectangle(frame, (0, 0), (330, 38), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return result


def contact_sheet(frames: np.ndarray, context_frames: int) -> np.ndarray:
    indices = [0, 7, 8, 14, 21, 28, 35, 42, 48]
    marked = frames[indices].copy()
    for frame, frame_idx in zip(marked, indices):
        phase = "CONTEXT" if frame_idx < context_frames else "TARGET"
        text = f"train frame {frame_idx:02d} | {phase}"
        cv2.rectangle(frame, (0, 0), (330, 38), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    tiles = [
        cv2.resize(frame, (448, 256), interpolation=cv2.INTER_AREA)
        for frame in marked
    ]
    return np.concatenate(
        [
            np.concatenate(tiles[0:3], axis=1),
            np.concatenate(tiles[3:6], axis=1),
            np.concatenate(tiles[6:9], axis=1),
        ],
        axis=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/"
        "industrial_s1_scale2_merged_h264_batch1500",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PyBulletRawNoGTBoxDataset(
        args.root,
        "train",
        (512, 896),
        num_frames=49,
        num_context_frames=8,
        sampling_strategy="prefix",
        window_starts=(0,),
    )

    selected: list[int] = []
    seen_families: set[str] = set()
    for idx, record in enumerate(dataset.samples):
        family = record.video_path.parent.parent.name
        if family not in seen_families and record.window_start == 0:
            selected.append(idx)
            seen_families.add(family)
        if len(seen_families) == 5:
            break

    manifest = []
    for idx in selected:
        sample = dataset[idx]
        frames = tensor_to_uint8(sample)
        metadata = sample["metadata"]
        case_name = (
            f"{metadata['family_slug']}_{metadata['sample_id']}_"
            f"start{metadata['window_start']:02d}"
        )
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        write_h264(case_dir / "training_video_49f_h264.mp4", frames)
        write_h264(case_dir / "context_video_8f_h264.mp4", frames[:8])
        write_h264(case_dir / "annotated_timeline_49f_h264.mp4", annotated(frames, 8))
        sheet = contact_sheet(frames, 8)
        cv2.imwrite(
            str(case_dir / "contact_sheet.jpg"),
            cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        )
        payload = {
            "case": case_name,
            "source_video": sample["video_path"],
            "source_resolution": [960, 540],
            "training_tensor_shape": list(sample["video"].shape),
            "training_resolution": [896, 512],
            "context_frames": 8,
            "target_frames": 41,
            "window_start": metadata["window_start"],
            "raw_frame_indices": metadata["sampled_frame_indices"],
            "caption": sample["caption"],
            "encoding": "H.264 / yuv420p",
            "fps": 30,
            "resize_mode": "stretch",
            "value_range": "[-1, 1]",
            "video_role": "frames 0-7 are context; frames 8-48 are diffusion targets",
        }
        (case_dir / "sample.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(payload)

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# PyBullet Raw 49-Frame Training Data Demo\n\n"
        "## Basic Parameters\n\n"
        "- Source: raw_v1 H.264 videos, 960x540, 90 frames, 30 FPS.\n"
        "- Window: 49 consecutive frames at raw start 0 (raw frames 0-48).\n"
        "- Training resolution: 896x512 (width x height), stretch resize.\n"
        "- Tensor: [C,T,H,W] = [3,49,512,896], normalized to [-1,1].\n"
        "- Context: local frames 0-7 (8 frames).\n"
        "- Diffusion target timeline: local frames 8-48 (41 frames).\n"
        "- Preview encoding: H.264, yuv420p, CRF 18, 30 FPS.\n"
        "- Train split: 1200 raw videos x 1 start-0 window = 1200 samples.\n"
        "- Caption: metadata-driven category + geometry + role + action relation.\n\n"
        "Each case contains the clean 49-frame training video, the exact 8-frame "
        "context video, an annotated timeline, a contact sheet, and sample.json.\n",
        encoding="utf-8",
    )
    print(f"exported {len(manifest)} cases to {output_dir}")


if __name__ == "__main__":
    main()
