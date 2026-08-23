#!/usr/bin/env python3
"""Caption physical videos with a pixel-selected evidence window and Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from decord import VideoReader  # noqa: E402
from PIL import Image  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from infer_physv_qwen3vl import final_answer  # noqa: E402
from physv_caption_ablation import (  # noqa: E402
    DEFAULT_MODEL,
    MIN_PROCESSOR_PIXELS,
    build_storyboard,
    disable_thinking,
    read_text,
    video_audit,
)


DEFAULT_SOURCE_RESULTS = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "0613_phyco_chinese_caption_prompt_fps15_maxpixels6500000.jsonl"
)
DEFAULT_OUTPUT = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "0613_phyco_evidence_pipeline_caption_fps15_maxpixels6500000.jsonl"
)
DEFAULT_INPUT_DIR = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "evidence_pipeline_inputs_fps15_maxpixels6500000"
)
DEFAULT_PROMPT = "prompts/physv_evidence_temporal_caption_zh.txt"

def event_window_indices(total_frames: int, center: int, window_size: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    window_size = min(window_size, total_frames)
    start = max(0, min(center - window_size // 2, total_frames - window_size))
    return list(range(start, start + window_size))


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


def event_time_keys(frame_indices: list[int], source_fps: float) -> list[str]:
    return [f"t={index / source_fps:.2f}s" for index in frame_indices]


def event_probe_prompt(frame_indices: list[int], source_fps: float) -> str:
    keys = event_time_keys(frame_indices, source_fps)
    return (
        "以下六张图是同一视频在标注时间的连续局部帧，顺序为从左到右、从上到下。"
        "只根据图像可见内容逐项核验，不使用物理先验。\n\n"
        "只输出一行 JSON，键必须依次为："
        + ", ".join(json.dumps(key, ensure_ascii=False) for key in keys)
        + "。每个值必须是少于24个汉字的客观可见状态。"
        "优先描述主要物体相对可见表面的悬空或接触、与其他物体的相碰、位置或方向变化。\n\n"
        "不要根据阴影判断接触，不要描述颜色、运动原因、视频其余部分或不可见细节。"
        "不要使用“停止”或“静止”，除非最后至少三张连续图中该物体位置都清楚未变；"
        "不要使用“滚动”“滑动”“反弹”或“碰撞”，除非连续图清楚支持。"
    )


def parse_event_probe(text: str, frame_indices: list[int], source_fps: float) -> dict[str, str]:
    keys = event_time_keys(frame_indices, source_fps)
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or list(parsed) != keys:
        raise ValueError(f"Event probe must contain exactly these time keys: {keys}")
    if not all(isinstance(value, str) and 0 < len(value) <= 48 for value in parsed.values()):
        raise ValueError("Event probe values must be non-empty short strings")
    return parsed


def select_event_center(reader: VideoReader) -> tuple[int, dict[str, Any]]:
    all_indices = list(range(len(reader)))
    frames = reader.get_batch(all_indices).asnumpy()
    grayscale = np.stack(
        [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (160, 90)) for frame in frames]
    ).astype(np.float32)
    if len(grayscale) == 1:
        return 0, {"method": "single_frame", "score": 0.0}

    difference = np.abs(grayscale[1:] - grayscale[:-1]).mean(axis=(1, 2))
    change = np.abs(np.diff(difference, prepend=difference[0]))
    score = difference + 3.0 * change
    center = int(np.argmax(score)) + 1
    top = np.argsort(score)[-5:][::-1]
    return center, {
        "method": "downscaled_grayscale_difference_plus_change",
        "score": float(score[center - 1]),
        "top_frames": [
            {
                "frame": int(index + 1),
                "difference": round(float(difference[index]), 5),
                "score": round(float(score[index]), 5),
            }
            for index in top
        ],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomically(path: Path, rows: list[dict[str, Any]]) -> None:
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
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def slug(case_id: str) -> str:
    return case_id.replace("/", "__")


def build_messages(
    frames: list[Image.Image],
    frame_indices: list[int],
    source_fps: float,
    question: str,
    video: str | None = None,
    fps: float = 15.0,
    max_frames: int = 64,
    max_pixels: int = 6500000,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if video is not None:
        content.append(
            {
                "type": "video",
                "video": video,
                "fps": fps,
                "max_frames": max_frames,
                "max_pixels": max_pixels,
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


def prepare_input(
    messages: list[dict[str, Any]], processor: Any, max_pixels: int
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "qwen_video_kwargs": video_kwargs,
    }
    if video_inputs:
        raw, metadata = video_inputs[0]
        metadata = dict(metadata)
        audit["video"] = video_audit(raw, metadata, processor, max_pixels)
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
    caption_prompt: str,
    event_states: dict[str, str],
    first_time: float,
    last_time: float,
) -> str:
    return (
        caption_prompt
        + "\n\n以下是从同一视频 t="
        + f"{first_time:.2f}s 至 t={last_time:.2f}s 的连续局部帧得到的视觉核验摘要。"
        + "它不是数据集标注，而是最终描述必须保留的已观察事实；不要因后续持续运动而省略这段短暂状态变化。\n"
        + f"逐帧视觉核验：{json.dumps(event_states, ensure_ascii=False)}\n\n"
        + "先按该核验描述相关状态变化，再描述完整视频的后续过程。"
        "不要把短窗口中的位置相近误写为停止或静止。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--source-results", type=Path, default=Path(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--prompt-file", type=Path, default=Path(DEFAULT_PROMPT))
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=6500000)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.79)
    parser.add_argument("--event-window-size", type=int, default=6)
    parser.add_argument("--evidence-width", type=int, default=576)
    parser.add_argument("--evidence-height", type=int, default=384)
    parser.add_argument("--probe-max-tokens", type=int, default=512)
    parser.add_argument("--caption-max-tokens", type=int, default=1024)
    parser.add_argument("--probe-thinking", action="store_true")
    parser.add_argument("--caption-thinking", action="store_true")
    parser.add_argument("--skip-mm-profiling", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.event_window_size != 6:
        raise ValueError("The evidence prompt and storyboard require --event-window-size=6")
    if not args.source_results.is_file():
        raise FileNotFoundError(args.source_results)
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(args.model_path)
    caption_prompt = read_text(args.prompt_file)
    source_rows = load_jsonl(args.source_results)
    wanted = set(args.case_ids or [row["case_id"] for row in source_rows])
    rows = [row for row in source_rows if row["case_id"] in wanted]
    missing = wanted - {row["case_id"] for row in rows}
    if missing:
        raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    if not rows:
        raise ValueError("No rows selected")

    args.input_dir.mkdir(parents=True, exist_ok=True)
    print(f"cases={len(rows)}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    print("processor_loaded")
    llm = None
    if not args.dry_run:
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
            limit_mm_per_prompt={"video": 1, "image": args.event_window_size},
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

    output_rows: list[dict[str, Any]] = []
    for position, source in enumerate(rows, start=1):
        case_id = source["case_id"]
        video = str(source["video"])
        started = time.time()
        result: dict[str, Any] = {
            "case_id": case_id,
            "dataset": source.get("dataset"),
            "video": video,
            "baseline_caption": source.get("response_final"),
            "caption_prompt": caption_prompt,
            "caption_prompt_source": str(args.prompt_file),
            "video_params": {
                "fps": 15.0,
                "max_frames": args.max_frames,
                "max_pixels": args.max_pixels,
            },
            "evidence_params": {
                "frame_count": args.event_window_size,
                "tile_width": args.evidence_width,
                "tile_height": args.evidence_height,
            },
            "thinking_enabled": {
                "event_probe": args.probe_thinking,
                "caption": args.caption_thinking,
            },
            "status": "prepared",
        }
        try:
            reader = VideoReader(video)
            source_fps = float(reader.get_avg_fps())
            center, selection = select_event_center(reader)
            indices = event_window_indices(len(reader), center, args.event_window_size)
            storyboard_path = args.input_dir / slug(case_id) / "event_storyboard.png"
            board, storyboard_audit = build_storyboard(
                reader,
                indices,
                storyboard_path,
                tile_width=args.evidence_width,
                tile_height=args.evidence_height,
            )
            frames = split_storyboard(board)
            result["event_window"] = {
                "center_frame": center,
                "center_time": center / source_fps,
                "frame_indices": indices,
                "start_time": indices[0] / source_fps,
                "end_time": indices[-1] / source_fps,
                "selection": selection,
                "storyboard": storyboard_audit,
            }

            probe_messages = build_messages(
                frames, indices, source_fps, event_probe_prompt(indices, source_fps)
            )
            probe_input, probe_audit = prepare_input(probe_messages, processor, args.max_pixels)
            result["event_probe"] = {"input_audit": probe_audit, "status": "prepared"}
            if args.dry_run:
                result["final_input"] = {"status": "pending_event_probe"}
            else:
                assert llm is not None
                probe_raw, probe_final, probe_elapsed = generate(
                    llm, probe_input, args.probe_max_tokens, args.probe_thinking
                )
                result["event_probe"].update(
                    {
                        "response_raw": probe_raw,
                        "response_final": probe_final,
                        "elapsed_seconds": probe_elapsed,
                        "status": "ok",
                    }
                )
                event_states = parse_event_probe(probe_final, indices, source_fps)
                result["event_probe"]["states"] = event_states
                final_question = evidence_augmented_prompt(
                    caption_prompt,
                    event_states,
                    indices[0] / source_fps,
                    indices[-1] / source_fps,
                )
                final_messages = build_messages(
                    frames,
                    indices,
                    source_fps,
                    final_question,
                    video=video,
                    fps=15.0,
                    max_frames=args.max_frames,
                    max_pixels=args.max_pixels,
                )
                final_input, final_audit = prepare_input(final_messages, processor, args.max_pixels)
                result["final_input"] = {"input_audit": final_audit, "status": "prepared"}
                final_raw, final_caption, final_elapsed = generate(
                    llm, final_input, args.caption_max_tokens, args.caption_thinking
                )
                result.update(
                    {
                        "response_raw": final_raw,
                        "response_final": final_caption,
                        "final_elapsed_seconds": final_elapsed,
                        "status": "ok",
                    }
                )
                result["final_input"]["status"] = "ok"
            print(f"[{position}/{len(rows)}] {case_id} status={result['status']}", flush=True)
            if result.get("response_final"):
                print(f"caption={result['response_final']!r}", flush=True)
        except Exception as exc:
            result.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "status": "error",
                }
            )
            print(f"[{position}/{len(rows)}] {case_id} error={type(exc).__name__}: {exc}", flush=True)
        result["elapsed_seconds"] = round(time.time() - started, 3)
        output_rows.append(result)
        write_jsonl_atomically(args.output, output_rows)
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
