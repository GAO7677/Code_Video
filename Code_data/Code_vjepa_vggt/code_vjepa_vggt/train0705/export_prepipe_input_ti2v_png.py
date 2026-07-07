from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
    read_video_uniform,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_image_as_single_frame(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    frame = np.asarray(image, dtype=np.uint8)
    return np.expand_dims(frame, axis=0), np.array([0], dtype=np.int64)


def _load_context_video_for_mode(
    *,
    video_path: Path,
    target_context_frames: int,
    sampling_mode: str,
):
    if video_path.suffix.lower() in _IMAGE_SUFFIXES:
        frames, frame_indices = _load_image_as_single_frame(video_path)
    elif sampling_mode == "uniform":
        frames, frame_indices = read_video_uniform(video_path, target_context_frames)
    else:
        frames, frame_indices = read_video_prefix(video_path, target_context_frames)
    if int(frames.shape[0]) < int(target_context_frames):
        raise RuntimeError(
            f"context video {video_path} only provides {int(frames.shape[0])} frames, "
            f"smaller than required num_context_frames={int(target_context_frames)}"
        )
    if int(frames.shape[0]) > int(target_context_frames):
        frames = frames[:target_context_frames]
        frame_indices = frame_indices[:target_context_frames]
    return frames, frame_indices


def _tensor_video_to_pil_list(context_video_single):
    frames = context_video_single.detach().cpu().permute(1, 2, 3, 0)
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(dtype=torch.uint8).numpy()
    return [Image.fromarray(frame) for frame in frames]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk the same pre-pipe TI2V input preprocessing path as train0705 V2V, "
            "without loading any model, and export the actual frame fed to pipe."
        )
    )
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--context-frames", type=int, default=1)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--output-name",
        type=str,
        default="input_ti2v_video.png",
        help="Filename written next to the source input_video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = core._read_list_file(args.input_json_list_path.expanduser().resolve())
    if args.limit is not None:
        json_paths = json_paths[: max(0, int(args.limit))]
    if not json_paths:
        raise RuntimeError(f"no input jsons found in {args.input_json_list_path}")

    for input_json_path in json_paths:
        payload = core._load_input_json(input_json_path)
        input_video = core._resolve_input_video(payload, input_json_path)
        context_video_path = Path(input_video).expanduser().resolve()
        frames, frame_indices = _load_context_video_for_mode(
            video_path=context_video_path,
            target_context_frames=int(args.context_frames),
            sampling_mode=str(args.sampling_mode),
        )
        context_video_single = preprocess_video_rgb_uint8(
            frames,
            (int(args.height), int(args.width)),
        )
        context_pil = _tensor_video_to_pil_list(context_video_single)
        if len(context_pil) < 1:
            raise RuntimeError(f"no pre-pipe frames produced for {context_video_path}")

        output_path = context_video_path.parent / str(args.output_name)
        context_pil[0].save(output_path)

        summary = {
            "input_json": str(input_json_path),
            "input_video": str(context_video_path),
            "saved_png": str(output_path),
            "context_frames_requested": int(args.context_frames),
            "sampling_mode": str(args.sampling_mode),
            "frame_indices": frame_indices.tolist(),
            "saved_frame_index": int(frame_indices[0]),
            "saved_png_wh": [int(context_pil[0].width), int(context_pil[0].height)],
            "num_prepipe_frames": len(context_pil),
        }
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
