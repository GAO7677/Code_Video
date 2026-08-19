#!/usr/bin/env python3
"""Build visual artifacts for the Qwen3.8 GPU7 demo's exact video input."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from imageio_ffmpeg import get_ffmpeg_exe
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


DEFAULT_RESULT = Path(
    "/data/gaoya/agent-data/outputs/physv_qwen3_8/f1_demo_gpu7_fla.json"
)
DEFAULT_PREVIOUS = Path(
    "/data/gaoya/agent-data/outputs/physv_qwen3_8/f1_fixed_gpu7.json"
)
DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3.8-27B-FP8"
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/physv_qwen3_8/viewer_gpu7_fla")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def unpack_video_patches(
    pixel_values: torch.Tensor, grid_thw: list[int], video_processor: Any
) -> torch.Tensor:
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
    merge_size = int(video_processor.merge_size)
    temporal_patch_size = int(video_processor.temporal_patch_size)
    patch_size = int(video_processor.patch_size)
    expected_tokens = grid_t * grid_h * grid_w
    expected_features = 3 * temporal_patch_size * patch_size * patch_size
    if list(pixel_values.shape) != [expected_tokens, expected_features]:
        raise ValueError(f"Unexpected patch shape {list(pixel_values.shape)} for grid {grid_thw}")
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(f"Grid is not divisible by merge size: {grid_thw}")
    patches = pixel_values.reshape(
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        3,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    return (
        patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        .contiguous()
        .reshape(grid_t * temporal_patch_size, 3, grid_h * patch_size, grid_w * patch_size)
    )


def to_rgb_uint8(frames: torch.Tensor, video_processor: Any | None = None) -> np.ndarray:
    frames = frames.detach().cpu()
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames, got {list(frames.shape)}")
    if frames.shape[1] == 3:
        frames = frames.permute(0, 2, 3, 1)
    elif frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB channels, got {list(frames.shape)}")
    if video_processor is not None:
        mean = torch.tensor(video_processor.image_mean, dtype=torch.float32).view(1, 1, 1, 3)
        std = torch.tensor(video_processor.image_std, dtype=torch.float32).view(1, 1, 1, 3)
        frames = (frames.float() * std + mean) / float(video_processor.rescale_factor)
    return frames.float().clamp(0, 255).round().to(torch.uint8).numpy()


def write_video(frames: np.ndarray, output: Path, fps: float) -> None:
    frame_count, height, width, channels = frames.shape
    if channels != 3 or width % 2 or height % 2:
        raise ValueError(f"Invalid RGB video shape: {list(frames.shape)}")
    ffmpeg = get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-framerate", f"{fps:.12g}",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(stderr.strip())
        temporary.replace(output)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if temporary.exists():
            temporary.unlink()


def make_contact_sheet(frames: np.ndarray, indices: list[int], source_fps: float, output: Path) -> None:
    thumb_width = 160
    thumb_height = max(round(frames.shape[1] * thumb_width / frames.shape[2]), 1)
    columns = 4
    label_height = 28
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#132326")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for position, (frame, index) in enumerate(zip(frames, indices)):
        row, col = divmod(position, columns)
        x, y = col * thumb_width, row * (thumb_height + label_height)
        image = Image.fromarray(frame).resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y + thumb_height, x + thumb_width, y + thumb_height + label_height), fill="#132326")
        draw.text((x + 7, y + thumb_height + 7), f"#{index:02d}  {index / source_fps:.2f}s", fill="#d9ecea", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    args = parse_args()
    result = load_json(args.result)
    previous = load_json(args.previous) if args.previous.is_file() else None
    request = result["video_request"]
    video = result["video"]
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video, **request},
        {"type": "text", "text": result["prompt"]},
    ]}]
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    _, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    raw_video, metadata = video_inputs[0]
    inputs = processor(
        text=[prompt_text], videos=[raw_video], video_metadata=[metadata], padding=True,
        return_tensors="pt", **video_kwargs,
    )
    grid_thw = [int(value) for value in inputs["video_grid_thw"][0].tolist()]
    processed = unpack_video_patches(
        inputs["pixel_values_videos"], grid_thw, processor.video_processor
    )
    raw_rgb = to_rgb_uint8(raw_video)
    processed_rgb = to_rgb_uint8(processed, processor.video_processor)
    indices = [int(value) for value in metadata["frames_indices"]]
    source_fps = float(metadata["fps"])
    display_fps = len(indices) * source_fps / int(metadata["total_num_frames"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_video(raw_rgb, args.output_dir / "sampled_frames.mp4", display_fps)
    write_video(processed_rgb, args.output_dir / "processor_frames.mp4", display_fps)
    make_contact_sheet(raw_rgb, indices, source_fps, args.output_dir / "sampled_contact_sheet.jpg")
    make_contact_sheet(processed_rgb, indices, source_fps, args.output_dir / "processor_contact_sheet.jpg")
    record = {
        "run": result,
        "previous_run": previous,
        "source_video_url": "/media/source-video",
        "sampled_frames_url": "/media/sampled-frames",
        "processor_frames_url": "/media/processor-frames",
        "sampled_contact_sheet_url": "/media/sampled-sheet",
        "processor_contact_sheet_url": "/media/processor-sheet",
        "input_audit": {
            "source_fps": source_fps,
            "source_total_frames": int(metadata["total_num_frames"]),
            "frames_indices": indices,
            "sampled_frame_count": len(indices),
            "input_replay_fps": display_fps,
            "raw_video_shape": list(raw_video.shape),
            "processor_frame_shape": list(processed.shape),
            "pixel_values_videos_shape": list(inputs["pixel_values_videos"].shape),
            "video_grid_thw": grid_thw,
            "input_ids_shape": list(inputs["input_ids"].shape),
            "attention_mask_shape": list(inputs["attention_mask"].shape),
            "raw_dtype": str(raw_video.dtype),
            "processor_dtype": str(processed.dtype),
            "video_backend": str(metadata.get("video_backend", "unknown")),
        },
    }
    with (args.output_dir / "viewer_data.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print(f"output_dir={args.output_dir}")
    print(f"sampled_frames={len(indices)} raw_shape={list(raw_video.shape)} processed_shape={list(processed.shape)}")
    print(f"pixel_values_shape={list(inputs['pixel_values_videos'].shape)} grid_thw={grid_thw}")


if __name__ == "__main__":
    main()
