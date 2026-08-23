#!/usr/bin/env python3
"""Probe a two-stage, evidence-first physical caption input on one video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from decord import VideoReader  # noqa: E402
from PIL import Image  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from infer_physv_qwen3vl import final_answer  # noqa: E402
from physv_caption_ablation import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_VIDEO,
    MIN_PROCESSOR_PIXELS,
    build_storyboard,
    disable_thinking,
    early_storyboard_indices,
    read_text,
    write_json_atomically,
)


DEFAULT_OUTPUT = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "f1_caption_evidence_pipeline_probe.json"
)
DEFAULT_INPUT_DIR = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "f1_caption_evidence_pipeline_inputs"
)
DEFAULT_EVIDENCE_PROMPT = "prompts/physv_evidence_temporal_caption_zh.txt"

STATE_PROBE_PROMPT = """以下六张图是同一视频在标注时间的连续局部帧，顺序为从左到右、从上到下。只根据图像可见内容逐项核验，不使用物理先验。

只输出一行 JSON，键依次为 t0.00、t0.03、t0.07、t0.10、t0.13、t0.20。每个值只能是："明显离开支撑面"、"与支撑面接触"、"无法可靠判断"。

只有当球体下缘与可见支撑面之间存在明显空隙时才选择“明显离开支撑面”；只有下缘与支撑面相连时才选择“与支撑面接触”。阴影或投影本身不是支撑面。不要描述颜色、运动原因或视频其余部分。"""

EARLY_CAPTION_PROMPT = """以下六张图是同一视频开头的连续局部帧，顺序为从左到右、从上到下。请只用不超过两句中文描述这段早期过程。

必须按时间顺序说明球体最初相对支撑面的状态，以及随后可见的高度或接触状态变化。只陈述图像直接支持的事实；阴影不是接触证据。"""


def split_storyboard(board: Image.Image) -> list[Image.Image]:
    width, height = board.size
    if width % 3 or height % 2:
        raise ValueError(f"Expected a 3x2 storyboard, got {width}x{height}")
    tile_width = width // 3
    tile_height = height // 2
    return [
        board.crop(
            (
                (index % 3) * tile_width,
                (index // 3) * tile_height,
                (index % 3 + 1) * tile_width,
                (index // 3 + 1) * tile_height,
            )
        )
        for index in range(6)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--video", type=Path, default=Path(DEFAULT_VIDEO))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--evidence-prompt-file", type=Path, default=Path(DEFAULT_EVIDENCE_PROMPT))
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=6500000)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.79)
    parser.add_argument("--skip-mm-profiling", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_messages(
    frames: list[Image.Image],
    frame_indices: list[int],
    source_fps: float,
    question: str,
    video: Path | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if video is not None:
        content.append(
            {
                "type": "video",
                "video": str(video),
                "fps": 15.0,
                "max_frames": 64,
                "max_pixels": 6500000,
            }
        )
    for frame, index in zip(frames, frame_indices):
        content.extend(
            [
                {"type": "text", "text": f"局部帧时间：t={index / source_fps:.2f}s"},
                {"type": "image", "image": frame},
            ]
        )
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def prepare_input(messages: list[dict[str, Any]], processor: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    audit: dict[str, Any] = {
        "image_count": len(image_inputs or []),
        "image_sizes": [list(image.size) for image in image_inputs or []],
        "video_kwargs": video_kwargs,
    }
    if video_inputs:
        raw, metadata = video_inputs[0]
        audit.update(
            {
                "video_stage_one_shape": list(raw.shape),
                "video_source_fps": float(metadata["fps"]),
                "video_source_frame_indices": [int(index) for index in metadata["frames_indices"]],
            }
        )
    return (
        {"prompt": prompt, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs},
        audit,
    )


def generate(
    llm: LLM, vllm_input: dict[str, Any], max_tokens: int, thinking: bool
) -> tuple[str, str, float]:
    request = dict(vllm_input)
    if not thinking:
        disable_thinking(request)
    started = time.time()
    output = llm.generate(
        [request],
        sampling_params=SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens),
        use_tqdm=False,
    )[0]
    raw = output.outputs[0].text
    return raw, final_answer(raw), round(time.time() - started, 3)


def evidence_augmented_prompt(
    caption_prompt: str, state_probe: str, early_caption: str
) -> str:
    return (
        caption_prompt
        + "\n\n以下是对同一视频前 0.20 秒局部帧的视觉核验结果。它不是数据集标注，"
        "而是需要在最终描述中保留的已观察事实；不要因为后续持续运动而省略该短暂状态变化。\n"
        + f"逐帧核验：{state_probe}\n"
        + f"早期核验摘要：{early_caption}\n\n"
        "先按这些已核验事实描述开头和首次状态变化，再描述完整视频后续过程。"
    )


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(args.model_path)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(str(args.video))
    indices = early_storyboard_indices(len(reader))
    board, storyboard_audit = build_storyboard(reader, indices, args.input_dir / "fixed_early_storyboard.png")
    frames = split_storyboard(board)
    source_fps = float(reader.get_avg_fps())
    evidence_prompt = read_text(args.evidence_prompt_file)
    messages = {
        "state_probe": build_messages(frames, indices, source_fps, STATE_PROBE_PROMPT),
        "early_caption": build_messages(frames, indices, source_fps, EARLY_CAPTION_PROMPT),
        "final_caption": build_messages(
            frames,
            indices,
            source_fps,
            evidence_prompt,
            video=args.video,
        ),
    }

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    prepared = {key: prepare_input(value, processor) for key, value in messages.items()}
    result: dict[str, Any] = {
        "model": args.model_path,
        "video": str(args.video),
        "source": {"fps": source_fps, "frames": int(len(reader))},
        "storyboard": storyboard_audit,
        "stages": {
            key: {"input_audit": audit, "status": "prepared"}
            for key, (_, audit) in prepared.items()
        },
    }
    write_json_atomically(args.output, result)
    print(f"video={args.video}")
    for key, (_, audit) in prepared.items():
        print(f"prepared={key} images={audit['image_count']} video={'video_stage_one_shape' in audit}")
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
        limit_mm_per_prompt={"video": 1, "image": 6},
        mm_processor_kwargs={
            "size": {"longest_edge": args.max_pixels, "shortest_edge": MIN_PROCESSOR_PIXELS}
        },
        skip_mm_profiling=args.skip_mm_profiling,
        enforce_eager=True,
        seed=0,
    )
    print("model_loaded")
    stages = (
        ("state_probe", 256, False),
        ("early_caption", 512, True),
        ("final_caption", 1024, True),
    )
    for key, max_tokens, thinking in stages:
        try:
            raw, answer, elapsed = generate(llm, prepared[key][0], max_tokens, thinking)
            result["stages"][key].update(
                {
                    "response_raw": raw,
                    "response_final": answer,
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                }
            )
            print(f"response={key} {answer!r}", flush=True)
        except Exception as exc:
            result["stages"][key].update(
                {"error_type": type(exc).__name__, "error": str(exc), "status": "error"}
            )
            print(f"error={key} {type(exc).__name__}: {exc}", flush=True)
        write_json_atomically(args.output, result)

    state_probe = result["stages"]["state_probe"].get("response_final")
    early_caption = result["stages"]["early_caption"].get("response_final")
    if isinstance(state_probe, str) and isinstance(early_caption, str):
        augmented_messages = build_messages(
            frames,
            indices,
            source_fps,
            evidence_augmented_prompt(evidence_prompt, state_probe, early_caption),
            video=args.video,
        )
        augmented_input, augmented_audit = prepare_input(augmented_messages, processor)
        result["stages"]["final_caption_with_evidence"] = {
            "input_audit": augmented_audit,
            "state_probe_source": state_probe,
            "early_caption_source": early_caption,
            "status": "prepared",
        }
        try:
            raw, answer, elapsed = generate(llm, augmented_input, 1024, True)
            result["stages"]["final_caption_with_evidence"].update(
                {
                    "response_raw": raw,
                    "response_final": answer,
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                }
            )
            print(f"response=final_caption_with_evidence {answer!r}", flush=True)
        except Exception as exc:
            result["stages"]["final_caption_with_evidence"].update(
                {"error_type": type(exc).__name__, "error": str(exc), "status": "error"}
            )
            print(f"error=final_caption_with_evidence {type(exc).__name__}: {exc}", flush=True)
        write_json_atomically(args.output, result)
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
