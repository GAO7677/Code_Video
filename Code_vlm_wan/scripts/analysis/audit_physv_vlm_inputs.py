#!/usr/bin/env python3
"""Rebuild and audit the visual video inputs supplied to Qwen3-VL via vLLM."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import torch
from decord import VideoReader
from imageio_ffmpeg import get_ffmpeg_exe
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8"
DEFAULT_RESULTS = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "0613_phyco_frame_compare_chinese_prompt.jsonl"
)
DEFAULT_FULL_RESULTS = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "0613_phyco_chinese_caption_prompt.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "vlm_input_replays_chinese_prompt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--full-results", type=Path, default=Path(DEFAULT_FULL_RESULTS))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--max-pixels", type=int, default=160000)
    parser.add_argument("--processor-shortest-edge", type=int, default=4096)
    parser.add_argument("--processor-longest-edge", type=int, default=160000)
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--variant-key", action="append", dest="variant_keys")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomically(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def canonical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "fps": float(metadata["fps"]),
        "frames_indices": [int(index) for index in metadata["frames_indices"]],
        "total_num_frames": int(metadata["total_num_frames"]),
        "video_backend": str(metadata.get("video_backend", "unknown")),
    }


def require_saved_stage_one(variant: dict[str, Any], raw: torch.Tensor, metadata: dict[str, Any]) -> None:
    saved_info = variant.get("video_info") or {}
    if saved_info.get("shape") != list(raw.shape):
        raise ValueError(
            f"Stage-one shape mismatch: saved={saved_info.get('shape')} rebuilt={list(raw.shape)}"
        )
    saved_metadata = saved_info.get("metadata") or {}
    actual_metadata = canonical_metadata(metadata)
    for key, actual_value in actual_metadata.items():
        expected_value = saved_metadata.get(key)
        if key == "fps":
            matches = expected_value is not None and abs(float(expected_value) - actual_value) < 1e-6
        elif key == "frames_indices":
            matches = [int(value) for value in expected_value or []] == actual_value
        elif key == "total_num_frames":
            matches = expected_value is not None and int(expected_value) == actual_value
        else:
            matches = expected_value == actual_value
        if not matches:
            raise ValueError(
                f"Stage-one metadata mismatch for {key}: "
                f"saved={expected_value!r} rebuilt={actual_value!r}"
            )


def build_messages(
    row: dict[str, Any],
    variant_key: str,
    variant: dict[str, Any],
    full_row: dict[str, Any] | None,
    max_pixels: int,
) -> list[dict[str, Any]]:
    video_content: dict[str, Any] = {
        "type": "video",
        "video": str(variant["video"]),
        "max_pixels": max_pixels,
    }
    params = variant.get("video_params")
    if params is None and variant_key == "full":
        if full_row is None:
            raise ValueError(f"Missing full-video result for {row['case_id']}")
        params = full_row.get("video_params")
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError(f"Invalid full-video parameters for {row['case_id']} / {variant_key}")
        if variant_key == "full":
            if full_row is None:
                raise ValueError(f"Missing full-video result for {row['case_id']}")
            if str(full_row.get("video")) != str(variant["video"]):
                raise ValueError(f"Full-video path mismatch for {row['case_id']}")
            if full_row.get("video_params") != params:
                raise ValueError(f"Saved full-video parameters do not match for {row['case_id']}")
        required = ("fps", "max_frames", "max_pixels")
        if any(name not in params for name in required):
            raise ValueError(f"Missing saved full-video parameters for {row['case_id']}")
        if int(params["max_pixels"]) != max_pixels:
            raise ValueError(
                f"Unexpected full-video max_pixels for {row['case_id']}: {params['max_pixels']}"
            )
        video_content["fps"] = float(params["fps"])
        video_content["max_frames"] = int(params["max_frames"])
    else:
        frame_count = variant.get("frame_count")
        if not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError(f"Missing prefix frame count for {row['case_id']} / {variant_key}")
        video_content["nframes"] = frame_count

    return [
        {
            "role": "user",
            "content": [video_content, {"type": "text", "text": row["question"]}],
        }
    ]


def extract_raw_video_input(
    messages: list[dict[str, Any]], processor: Any
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    _, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if not video_inputs or len(video_inputs) != 1:
        raise ValueError("Expected exactly one video input")
    item = video_inputs[0]
    if not isinstance(item, tuple) or len(item) != 2:
        raise ValueError("Expected video input with metadata")
    raw, metadata = item
    if not isinstance(metadata, dict):
        metadata = dict(metadata)
    if video_kwargs.get("do_sample_frames") is not False:
        raise ValueError(f"Unexpected Qwen video kwargs: {video_kwargs}")
    return raw.detach().cpu().contiguous(), metadata, dict(video_kwargs)


def unpatch_qwen3_vl_video(
    pixel_values: torch.Tensor, grid_thw: list[int], video_processor: Any
) -> torch.Tensor:
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
    merge_size = int(video_processor.merge_size)
    temporal_patch_size = int(video_processor.temporal_patch_size)
    patch_size = int(video_processor.patch_size)
    channels = 3
    expected_tokens = grid_t * grid_h * grid_w
    expected_features = channels * temporal_patch_size * patch_size * patch_size
    if list(pixel_values.shape) != [expected_tokens, expected_features]:
        raise ValueError(
            f"Unexpected patch shape {list(pixel_values.shape)} for grid {grid_thw}"
        )
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(f"Grid is not divisible by merge size: {grid_thw}")

    patches = pixel_values.reshape(
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    return (
        patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        .contiguous()
        .reshape(
            grid_t * temporal_patch_size,
            channels,
            grid_h * patch_size,
            grid_w * patch_size,
        )
    )


def repatch_qwen3_vl_video(
    frames: torch.Tensor, grid_thw: list[int], video_processor: Any
) -> torch.Tensor:
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
    merge_size = int(video_processor.merge_size)
    temporal_patch_size = int(video_processor.temporal_patch_size)
    patch_size = int(video_processor.patch_size)
    expected_shape = [
        grid_t * temporal_patch_size,
        3,
        grid_h * patch_size,
        grid_w * patch_size,
    ]
    if list(frames.shape) != expected_shape:
        raise ValueError(f"Unexpected reconstructed video shape: {list(frames.shape)}")
    return (
        frames.reshape(
            grid_t,
            temporal_patch_size,
            3,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        .contiguous()
        .reshape(-1, 3 * temporal_patch_size * patch_size * patch_size)
    )


def denormalize_to_rgb(frames: torch.Tensor, video_processor: Any) -> np.ndarray:
    mean = torch.tensor(video_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(video_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    rescale_factor = float(video_processor.rescale_factor)
    rgb = ((frames.float() * std + mean) / rescale_factor).clamp(0, 255).round()
    return rgb.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


def replay_fps(metadata: dict[str, Any], input_frame_count: int) -> float:
    source_fps = float(metadata["fps"])
    total_frames = int(metadata["total_num_frames"])
    if source_fps <= 0 or total_frames <= 0:
        raise ValueError(f"Invalid source timing metadata: {metadata}")
    return input_frame_count * source_fps / total_frames


def replay_name(case_id: str, variant_key: str) -> str:
    return f"{case_id.replace('/', '__')}__{variant_key}__vlm_input.mp4"


def write_replay_video(
    frames: np.ndarray, destination: Path, fps: float, ffmpeg_bin: str
) -> None:
    frame_count, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames, got shape {list(frames.shape)}")
    if width % 2 or height % 2:
        raise ValueError(f"H.264 replay dimensions must be even, got {width}x{height}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.12g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {destination}: {stderr.strip()}")
        temporary.replace(destination)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if temporary.exists():
            temporary.unlink()

    reader = VideoReader(str(destination))
    if len(reader) != frame_count:
        raise RuntimeError(f"Replay frame count mismatch for {destination}: {len(reader)} != {frame_count}")
    decoded_shape = list(reader[0].asnumpy().shape)
    if decoded_shape[:2] != [height, width] or decoded_shape[2] != 3:
        raise RuntimeError(f"Replay dimensions mismatch for {destination}: {decoded_shape}")


def model_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def audit_variant(
    row: dict[str, Any],
    variant_key: str,
    variant: dict[str, Any],
    full_row: dict[str, Any] | None,
    processor: Any,
    args: argparse.Namespace,
    ffmpeg_bin: str,
) -> dict[str, Any]:
    messages = build_messages(row, variant_key, variant, full_row, args.max_pixels)
    raw, metadata, video_kwargs = extract_raw_video_input(messages, processor)
    require_saved_stage_one(variant, raw, metadata)

    processor_size = {
        "shortest_edge": args.processor_shortest_edge,
        "longest_edge": args.processor_longest_edge,
    }
    processor_output = processor.video_processor(
        raw,
        do_sample_frames=False,
        size=processor_size,
        return_tensors="pt",
    )
    precast_patches = processor_output["pixel_values_videos"].detach().cpu().contiguous()
    grid_thw = [int(value) for value in processor_output["video_grid_thw"][0].tolist()]
    patches = precast_patches.to(dtype=model_dtype(args.model_dtype)).contiguous()
    normalized_frames = unpatch_qwen3_vl_video(patches, grid_thw, processor.video_processor)
    if not torch.equal(
        repatch_qwen3_vl_video(normalized_frames, grid_thw, processor.video_processor), patches
    ):
        raise RuntimeError(f"Patch round trip failed for {row['case_id']} / {variant_key}")

    rgb_frames = denormalize_to_rgb(normalized_frames, processor.video_processor)
    destination = args.output_dir / replay_name(row["case_id"], variant_key)
    fps = replay_fps(metadata, raw.shape[0])
    if not args.dry_run:
        write_replay_video(rgb_frames, destination, fps, ffmpeg_bin)

    canonical = canonical_metadata(metadata)
    return {
        "status": "verified",
        "source_video": str(variant["video"]),
        "source_fps": canonical["fps"],
        "source_total_frames": canonical["total_num_frames"],
        "source_frame_indices": canonical["frames_indices"],
        "source_video_backend": canonical["video_backend"],
        "qwen_video_kwargs": video_kwargs,
        "stage_one_shape": list(raw.shape),
        "stage_one_dtype": str(raw.dtype),
        "stage_one_tensor_sha256": tensor_sha256(raw),
        "vllm_processor_kwargs": {
            "do_sample_frames": False,
            "size": processor_size,
        },
        "vllm_patch_precast_dtype": str(precast_patches.dtype),
        "vllm_patch_precast_sha256": tensor_sha256(precast_patches),
        "vllm_patch_shape": list(patches.shape),
        "vllm_patch_dtype": str(patches.dtype),
        "vllm_patch_sha256": tensor_sha256(patches),
        "vllm_grid_thw": grid_thw,
        "visual_replay_shape": list(normalized_frames.shape),
        "visual_replay_rgb_sha256": hashlib.sha256(rgb_frames.tobytes()).hexdigest(),
        "visual_replay_fps": fps,
        "visual_replay_encoding": "h264_yuv420p_rgb_replay_from_denormalized_vllm_patches",
        "visual_replay_note": (
            "The MP4 is a display replay from the exact final patch tensor after "
            "denormalization and 8-bit RGB encoding; tensor hashes identify the "
            "actual vLLM model input."
        ),
        "tool_versions": {
            "transformers": importlib.metadata.version("transformers"),
            "qwen_vl_utils": importlib.metadata.version("qwen-vl-utils"),
            "vllm_effective_dtype": args.model_dtype,
        },
        "replay_video": str(destination),
    }


def main() -> None:
    args = parse_args()
    if not args.results.is_file():
        raise FileNotFoundError(args.results)
    if not args.full_results.is_file():
        raise FileNotFoundError(args.full_results)
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(args.model_path)

    rows = load_jsonl(args.results)
    full_rows = load_jsonl(args.full_results)
    full_by_case = {row["case_id"]: row for row in full_rows}
    if len(full_by_case) != len(full_rows):
        raise ValueError("Duplicate case IDs in full-video results")
    selected_case_ids = set(args.case_ids or [row["case_id"] for row in rows])
    variant_keys = args.variant_keys or ["prefix_8", "prefix_16", "prefix_24", "full"]
    if len(set(variant_keys)) != len(variant_keys):
        raise ValueError(f"Duplicate variant keys: {variant_keys}")
    missing_case_ids = selected_case_ids - {row["case_id"] for row in rows}
    if missing_case_ids:
        raise ValueError(f"Unknown case IDs: {sorted(missing_case_ids)}")

    print(f"results={args.results}")
    print(f"output_dir={args.output_dir}")
    print(f"cases={len(selected_case_ids)}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    print("processor_loaded")
    ffmpeg_bin = get_ffmpeg_exe()

    audited = 0
    for row in rows:
        if row["case_id"] not in selected_case_ids:
            continue
        full_row = full_by_case.get(row["case_id"])
        if full_row is None:
            raise ValueError(f"No full-video result for {row['case_id']}")
        variants = row.get("variants") or {}
        for variant_key in variant_keys:
            variant = variants.get(variant_key)
            if variant is None:
                raise ValueError(f"Missing {variant_key} for {row['case_id']}")
            audit = audit_variant(
                row, variant_key, variant, full_row, processor, args, ffmpeg_bin
            )
            if not args.dry_run:
                variant["vlm_input_video"] = audit["replay_video"]
                variant["vlm_input_video_fps"] = audit["visual_replay_fps"]
                variant["vlm_input_audit"] = audit
            audited += 1
            print(
                f"verified={row['case_id']} variant={variant_key} "
                f"visual_shape={audit['visual_replay_shape']} "
                f"patch_shape={audit['vllm_patch_shape']}",
                flush=True,
            )

    if args.dry_run:
        print(f"dry_run_verified_variants={audited}")
    else:
        write_jsonl_atomically(args.results, rows)
        print(f"audited_variants={audited}")
        print(f"updated_results={args.results}")


if __name__ == "__main__":
    main()
