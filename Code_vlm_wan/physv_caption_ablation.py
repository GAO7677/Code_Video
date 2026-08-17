#!/usr/bin/env python3
"""Run auditable input and prompt ablations for one physical video caption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def dense_start_indices(total_frames: int, dense_frames: int, target_frames: int) -> list[int]:
    """Keep the earliest frames contiguous, then cover the remaining duration."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")

    target_frames = min(target_frames, total_frames)
    dense_frames = min(dense_frames, target_frames)
    head = list(range(dense_frames))
    remaining = target_frames - dense_frames
    if remaining == 0:
        return head

    import numpy as np

    tail = np.linspace(dense_frames, total_frames - 1, remaining).round().astype(int).tolist()
    indices = sorted(set(head + tail))
    if len(indices) != target_frames:
        raise RuntimeError(
            f"Expected {target_frames} unique indices, got {len(indices)}: {indices}"
        )
    return indices


def early_storyboard_indices(total_frames: int) -> list[int]:
    candidates = [0, 1, 2, 3, 4, 6]
    indices = [min(index, total_frames - 1) for index in candidates]
    return indices


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from decord import VideoReader  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from infer_physv_qwen3vl import final_answer  # noqa: E402


DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8"
DEFAULT_VIDEO = (
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/"
    "industrial_s1_scale2_merged_h264_batch1500/val/F1_single_object/"
    "sample_000301/video.mp4"
)
DEFAULT_OUTPUT = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "f1_caption_input_prompt_ablation.json"
)
DEFAULT_INPUT_DIR = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "f1_caption_input_prompt_ablation_inputs"
)
DEFAULT_BASELINE_PROMPT = "prompts/physv_concise_temporal_caption_zh.txt"
DEFAULT_EVIDENCE_PROMPT = "prompts/physv_evidence_temporal_caption_zh.txt"
MIN_PROCESSOR_PIXELS = 4096


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    fps: float
    prompt_kind: str
    thinking: bool
    input_profile: str


VARIANTS = (
    Variant("baseline", "当前基线 / FPS 15", 15.0, "baseline", False, "native"),
    Variant("prompt_only", "仅替换证据提示词 / FPS 15", 15.0, "evidence", False, "native"),
    Variant("thinking_only", "仅启用 Thinking / FPS 15", 15.0, "baseline", True, "native"),
    Variant("fps30_only", "仅提高至 FPS 30", 30.0, "baseline", False, "native"),
    Variant("dense_start_only", "仅开头密集采样", 30.0, "baseline", False, "dense_start"),
    Variant("storyboard_only", "仅追加早期局部证据图", 15.0, "baseline", False, "storyboard"),
    Variant(
        "recommended_combo",
        "证据提示词 + Thinking + 密集采样 + 局部证据图",
        30.0,
        "evidence",
        True,
        "dense_start_storyboard",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--video", type=Path, default=Path(DEFAULT_VIDEO))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--baseline-prompt-file", type=Path, default=Path(DEFAULT_BASELINE_PROMPT))
    parser.add_argument("--evidence-prompt-file", type=Path, default=Path(DEFAULT_EVIDENCE_PROMPT))
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--dense-frames", type=int, default=12)
    parser.add_argument("--max-pixels", type=int, default=6500000)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.79)
    parser.add_argument("--answer-max-tokens", type=int, default=160)
    parser.add_argument("--thinking-max-tokens", type=int, default=512)
    parser.add_argument("--skip-mm-profiling", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def video_metadata(reader: VideoReader, indices: list[int]) -> dict[str, Any]:
    return {
        "fps": float(reader.get_avg_fps()),
        "frames_indices": [int(index) for index in indices],
        "total_num_frames": int(len(reader)),
        "video_backend": "decord_custom_indices",
    }


def read_custom_video(reader: VideoReader, indices: list[int]) -> torch.Tensor:
    frames = reader.get_batch(indices).asnumpy()
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().contiguous()


def foreground_box(frame: np.ndarray, background: np.ndarray) -> tuple[int, int, int, int] | None:
    difference = np.abs(frame.astype(np.int16) - background.astype(np.int16)).mean(axis=2)
    threshold = max(16.0, float(np.percentile(difference, 99.3)))
    mask = (difference >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [stats[index] for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 24]
    if not candidates:
        return None

    x, y, box_width, box_height, _ = max(candidates, key=lambda value: int(value[cv2.CC_STAT_AREA]))
    return int(x), int(y), int(box_width), int(box_height)


def shared_crop_box(frames: np.ndarray, background: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frames.shape[1:3]
    boxes = [box for frame in frames if (box := foreground_box(frame, background)) is not None]
    if not boxes:
        return 0, 0, width, height

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    largest_object = max(max(box[2], box[3]) for box in boxes)
    margin = largest_object * 2
    roi_height = max(bottom - top + margin * 2, largest_object * 6)
    roi_width = max(right - left + margin * 2, roi_height * 1.5)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    roi_width = min(width, int(round(roi_width)))
    roi_height = min(height, int(round(roi_height)))
    crop_left = max(0, min(width - roi_width, int(round(center_x - roi_width / 2))))
    crop_top = max(0, min(height - roi_height, int(round(center_y - roi_height / 2))))
    return crop_left, crop_top, crop_left + roi_width, crop_top + roi_height


def build_storyboard(reader: VideoReader, indices: list[int], destination: Path) -> tuple[Image.Image, dict[str, Any]]:
    all_indices = list(range(len(reader)))
    background = np.median(reader.get_batch(all_indices).asnumpy(), axis=0).astype(np.uint8)
    frames = reader.get_batch(indices).asnumpy()
    crop_left, crop_top, crop_right, crop_bottom = shared_crop_box(frames, background)
    tile_width = 384
    tile_height = 256
    board = Image.new("RGB", (tile_width * 3, tile_height * 2), color=(0, 0, 0))
    draw = ImageDraw.Draw(board)
    source_fps = float(reader.get_avg_fps())
    for offset, (index, frame) in enumerate(zip(indices, frames)):
        crop = Image.fromarray(frame[crop_top:crop_bottom, crop_left:crop_right]).resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        x = (offset % 3) * tile_width
        y = (offset // 3) * tile_height
        board.paste(crop, (x, y))
        label = f"t={index / source_fps:.2f}s"
        draw.rectangle((x, y, x + 92, y + 22), fill=(0, 0, 0))
        draw.text((x + 5, y + 4), label, fill=(255, 255, 255))

    destination.parent.mkdir(parents=True, exist_ok=True)
    board.save(destination)
    return board, {
        "path": str(destination),
        "frame_indices": indices,
        "source_fps": source_fps,
        "shape": [board.height, board.width, 3],
        "shared_source_crop": [crop_left, crop_top, crop_right, crop_bottom],
        "rgb_sha256": hashlib.sha256(np.asarray(board).tobytes()).hexdigest(),
    }


def build_messages(
    video: Path,
    fps: float,
    max_frames: int,
    max_pixels: int,
    question: str,
    storyboard: Image.Image | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "video",
            "video": str(video),
            "fps": fps,
            "max_frames": max_frames,
            "max_pixels": max_pixels,
        }
    ]
    if storyboard is not None:
        content.extend(
            [
                {"type": "image", "image": storyboard},
                {
                    "type": "text",
                    "text": (
                        "附图是同一视频开头的局部放大帧，按从左到右、从上到下的时间顺序排列；"
                        "图中只标记时间，不包含事件标签。\n\n"
                        + question
                    ),
                },
            ]
        )
    else:
        content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def video_audit(
    raw: torch.Tensor,
    metadata: dict[str, Any],
    processor: Any,
    max_pixels: int,
) -> dict[str, Any]:
    output = processor.video_processor(
        raw,
        do_sample_frames=False,
        size={"shortest_edge": MIN_PROCESSOR_PIXELS, "longest_edge": max_pixels},
        return_tensors="pt",
    )
    grid = [int(value) for value in output["video_grid_thw"][0].tolist()]
    final_shape = [
        grid[0] * processor.video_processor.temporal_patch_size,
        3,
        grid[1] * processor.video_processor.patch_size,
        grid[2] * processor.video_processor.patch_size,
    ]
    return {
        "source_fps": float(metadata["fps"]),
        "source_frame_indices": [int(index) for index in metadata["frames_indices"]],
        "source_total_frames": int(metadata["total_num_frames"]),
        "source_video_backend": metadata["video_backend"],
        "stage_one_shape": list(raw.shape),
        "stage_one_tensor_sha256": tensor_sha256(raw),
        "vllm_grid_thw": grid,
        "visual_replay_shape": final_shape,
        "llm_visual_tokens": int(grid[0] * grid[1] * grid[2] // (processor.video_processor.merge_size**2)),
    }


def prepare_native_input(
    messages: list[dict[str, Any]], processor: Any, max_pixels: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if not video_inputs or len(video_inputs) != 1:
        raise RuntimeError("Expected exactly one native video input")
    raw, metadata = video_inputs[0]
    metadata = dict(metadata)
    mm_data: dict[str, Any] = {"video": video_inputs}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    return (
        {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        },
        video_audit(raw, metadata, processor, max_pixels),
    )


def prepare_dense_input(
    video: Path,
    messages: list[dict[str, Any]],
    processor: Any,
    max_frames: int,
    dense_frames: int,
    max_pixels: int,
    storyboard: Image.Image | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reader = VideoReader(str(video))
    indices = dense_start_indices(len(reader), dense_frames, max_frames)
    raw = read_custom_video(reader, indices)
    metadata = video_metadata(reader, indices)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    mm_data: dict[str, Any] = {"video": [(raw, metadata)]}
    if storyboard is not None:
        mm_data["image"] = [storyboard]
    return (
        {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {"do_sample_frames": False},
        },
        video_audit(raw, metadata, processor, max_pixels),
    )


def disable_thinking(vllm_input: dict[str, Any]) -> None:
    thinking_prompt = "<|im_start|>assistant\n<think>\n"
    answer_prompt = "<|im_start|>assistant\n<think>\n</think>\n\n"
    if vllm_input["prompt"].endswith(thinking_prompt):
        vllm_input["prompt"] = vllm_input["prompt"][: -len(thinking_prompt)] + answer_prompt


def make_variant_input(
    variant: Variant,
    video: Path,
    prompts: dict[str, str],
    processor: Any,
    args: argparse.Namespace,
    storyboard: Image.Image,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    use_storyboard = variant.input_profile in {"storyboard", "dense_start_storyboard"}
    messages = build_messages(
        video,
        variant.fps,
        args.max_frames,
        args.max_pixels,
        prompts[variant.prompt_kind],
        storyboard if use_storyboard else None,
    )
    if variant.input_profile in {"dense_start", "dense_start_storyboard"}:
        vllm_input, audit = prepare_dense_input(
            video,
            messages,
            processor,
            args.max_frames,
            args.dense_frames,
            args.max_pixels,
            storyboard if use_storyboard else None,
        )
    else:
        vllm_input, audit = prepare_native_input(messages, processor, args.max_pixels)
    if not variant.thinking:
        disable_thinking(vllm_input)
    return vllm_input, audit, None


def sampling_params(variant: Variant, args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.thinking_max_tokens if variant.thinking else args.answer_max_tokens,
    )


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(args.model_path)
    prompts = {
        "baseline": read_text(args.baseline_prompt_file),
        "evidence": read_text(args.evidence_prompt_file),
    }
    args.input_dir.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(str(args.video))
    storyboard, storyboard_audit = build_storyboard(
        reader,
        early_storyboard_indices(len(reader)),
        args.input_dir / "early_contact_storyboard.png",
    )

    results: dict[str, Any] = {
        "model": args.model_path,
        "video": str(args.video),
        "source": {
            "fps": float(reader.get_avg_fps()),
            "frames": int(len(reader)),
        },
        "max_pixels": args.max_pixels,
        "max_model_len": args.max_model_len,
        "storyboard": storyboard_audit,
        "variants": [],
    }
    print(f"video={args.video}")
    print(f"source_fps={results['source']['fps']}")
    print(f"source_frames={results['source']['frames']}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    print("processor_loaded")
    prepared: list[tuple[Variant, dict[str, Any], dict[str, Any]]] = []
    for variant in VARIANTS:
        vllm_input, audit, _ = make_variant_input(
            variant, args.video, prompts, processor, args, storyboard
        )
        prepared.append((variant, vllm_input, audit))
        print(
            f"prepared={variant.key} frames={len(audit['source_frame_indices'])} "
            f"grid={audit['vllm_grid_thw']} visual_tokens={audit['llm_visual_tokens']}",
            flush=True,
        )
        results["variants"].append(
            {
                **asdict(variant),
                "prompt": prompts[variant.prompt_kind],
                "prompt_sha256": hashlib.sha256(
                    prompts[variant.prompt_kind].encode("utf-8")
                ).hexdigest(),
                "input_audit": audit,
                "status": "prepared",
            }
        )
    write_json_atomically(args.output, results)
    if args.dry_run:
        print(f"results={args.output}")
        return

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        limit_mm_per_prompt={"video": 1, "image": 1},
        mm_processor_kwargs={
            "size": {
                "longest_edge": args.max_pixels,
                "shortest_edge": MIN_PROCESSOR_PIXELS,
            }
        },
        skip_mm_profiling=args.skip_mm_profiling,
        enforce_eager=True,
        seed=0,
    )
    print("model_loaded")
    for result, (variant, vllm_input, _) in zip(results["variants"], prepared):
        started = time.time()
        try:
            output = llm.generate(
                [vllm_input], sampling_params=sampling_params(variant, args), use_tqdm=False
            )[0]
            raw_text = output.outputs[0].text
            result.update(
                {
                    "response_raw": raw_text,
                    "response_final": final_answer(raw_text),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "status": "ok",
                }
            )
            print(f"response={variant.key} {result['response_final']!r}", flush=True)
        except Exception as exc:
            result.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "status": "error",
                }
            )
            print(f"error={variant.key} {type(exc).__name__}: {exc}", flush=True)
        write_json_atomically(args.output, results)
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
