#!/usr/bin/env python3
"""Run prefix-frame comparisons against the existing full-video answers."""

import argparse
import gc
import json
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from decord import VideoReader  # noqa: E402
from imageio_ffmpeg import get_ffmpeg_exe  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from infer_physv_qwen3vl import (  # noqa: E402
    final_answer,
    get_video_info,
    is_oom_error,
    prepare_vllm_input,
)


DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8"
DEFAULT_FULL_RESULTS = "/data/gaoya/agent-data/outputs/physv_qwen3vl/0613_phyco_caption_prompt.jsonl"
DEFAULT_PROMPT = "/home/gaoya/Code_Video/Code_vlm_wan/prompts/physv_concise_temporal_caption.txt"
DEFAULT_CLIP_ROOT = "/data/gaoya/agent-data/outputs/physv_qwen3vl/frame_prefix_clips"
DEFAULT_OUTPUT = "/data/gaoya/agent-data/outputs/physv_qwen3vl/0613_phyco_frame_compare.jsonl"
FRAME_COUNTS = (8, 16, 24)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--full-results", type=Path, default=Path(DEFAULT_FULL_RESULTS))
    parser.add_argument("--prompt-file", type=Path, default=Path(DEFAULT_PROMPT))
    parser.add_argument("--clip-root", type=Path, default=Path(DEFAULT_CLIP_ROOT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--max-pixels", type=int, default=360 * 640)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument(
        "--normalize-existing",
        action="store_true",
        help="Normalize response_final fields in an existing comparison JSONL and exit.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clip_name(case_id: str, frame_count: int) -> str:
    safe_case_id = case_id.replace("/", "__")
    return f"{safe_case_id}__prefix_{frame_count:02d}.mp4"


def make_prefix_clip(source: Path, destination: Path, frame_count: int, ffmpeg_bin: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-vsync",
        "0",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    actual_frames = len(VideoReader(str(destination)))
    if actual_frames != frame_count:
        raise RuntimeError(
            f"Prefix clip has {actual_frames} frames, expected {frame_count}: {destination}"
        )


def build_prefix_messages(video_path: Path, prompt: str, frame_count: int, max_pixels: int):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": frame_count,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def answer_prompt(vllm_input: dict):
    thinking_prompt = "<|im_start|>assistant\n<think>\n"
    answer_prompt = "<|im_start|>assistant\n<think>\n</think>\n\n"
    if vllm_input["prompt"].endswith(thinking_prompt):
        vllm_input["prompt"] = (
            vllm_input["prompt"][: -len(thinking_prompt)] + answer_prompt
        )


def normalize_caption(text: str, max_sentences: int = 4, max_words: int = 120) -> str:
    """Keep the visible caption within the prompt's sentence and word limits."""
    caption = re.sub(r"\s+", " ", final_answer(text)).strip()
    if not caption:
        return caption

    selected = []
    word_count = 0
    for sentence in re.split(r"(?<=[.!?])\s+", caption):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = re.findall(r"\b[\w'-]+\b", sentence)
        if len(selected) >= max_sentences or word_count + len(sentence_words) > max_words:
            break
        selected.append(sentence)
        word_count += len(sentence_words)

    if selected:
        return " ".join(selected)

    words = re.findall(r"\S+", caption)[:max_words]
    return " ".join(words).rstrip(".!?") + "."


def normalize_existing_results(output_path: Path):
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    rows = load_rows(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            for variant in row.get("variants", {}).values():
                raw_text = variant.get("response_raw") or variant.get("response_final", "")
                if raw_text:
                    variant["response_final"] = normalize_caption(raw_text)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temp_path.replace(output_path)
    print(f"normalized_results={output_path}")


def run_prefix_case(
    video_path: Path,
    prompt: str,
    frame_count: int,
    max_pixels: int,
    processor,
    llm,
    sampling_params,
):
    result = {
        "frame_count": frame_count,
        "video": str(video_path),
        "thinking_disabled": True,
    }
    messages = build_prefix_messages(video_path, prompt, frame_count, max_pixels)
    try:
        vllm_input, video_inputs = prepare_vllm_input(messages, processor)
        answer_prompt(vllm_input)
        result["video_info"] = get_video_info(video_inputs)
        outputs = llm.generate(
            [vllm_input], sampling_params=sampling_params, use_tqdm=False
        )
        raw_text = outputs[0].outputs[0].text
        result.update(
            {
                "response_raw": raw_text,
                "response_final": normalize_caption(raw_text),
                "status": "ok",
            }
        )
        print(f"response={raw_text!r}", flush=True)
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "oom": is_oom_error(exc),
                "status": "error",
            }
        )
        print(f"error={type(exc).__name__}: {exc}", flush=True)
    return result


def full_variant(row: dict):
    return {
        "label": "完整视频",
        "frame_count": None,
        "video": row["video"],
        "video_info": row.get("video_info"),
        "response_raw": row.get("response_raw", ""),
        "response_final": row.get("response_final", ""),
        "status": row.get("status", "unknown"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "thinking_disabled": row.get("thinking_disabled", True),
        "source": "existing_full_result",
    }


def main():
    args = parse_args()
    if args.normalize_existing:
        normalize_existing_results(args.output)
        return
    if not args.full_results.is_file():
        raise FileNotFoundError(args.full_results)
    if not args.prompt_file.is_file():
        raise FileNotFoundError(args.prompt_file)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    full_rows = load_rows(args.full_results)
    if not full_rows:
        raise ValueError(f"No rows in {args.full_results}")
    if any(row.get("status") != "ok" for row in full_rows):
        raise ValueError("Full results contain non-ok rows")

    ffmpeg_bin = args.ffmpeg_bin or get_ffmpeg_exe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.clip_root.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model_path}")
    print(f"cases={len(full_rows)}")
    print(f"frame_counts={FRAME_COUNTS}")
    print(f"ffmpeg={ffmpeg_bin}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    clip_paths = {}
    for row in full_rows:
        source = Path(row["video"])
        case_clips = {}
        for frame_count in FRAME_COUNTS:
            destination = args.clip_root / clip_name(row["case_id"], frame_count)
            print(f"clip={row['case_id']} frames={frame_count} path={destination}")
            make_prefix_clip(source, destination, frame_count, ffmpeg_bin)
            case_clips[frame_count] = destination
        clip_paths[row["case_id"]] = case_clips

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    print("processor_loaded")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        limit_mm_per_prompt={"video": 1},
        enforce_eager=True,
        seed=0,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
    )

    with args.output.open("w", encoding="utf-8") as output_file:
        for case_index, row in enumerate(full_rows, start=1):
            print(f"[{case_index}/{len(full_rows)}] {row['case_id']}", flush=True)
            variants = {"full": full_variant(row)}
            for frame_count in FRAME_COUNTS:
                key = f"prefix_{frame_count}"
                print(f"variant={key}", flush=True)
                variant = run_prefix_case(
                    clip_paths[row["case_id"]][frame_count],
                    prompt,
                    frame_count,
                    args.max_pixels,
                    processor,
                    llm,
                    sampling_params,
                )
                variant["label"] = f"前 {frame_count} 帧"
                variant["source"] = "generated_prefix_clip"
                variants[key] = variant
                gc.collect()

            comparison = {
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "source_video": row["video"],
                "question": prompt,
                "prompt_source": str(args.prompt_file),
                "original_prompt": row.get("original_prompt", ""),
                "original_prompt_source": row.get("original_prompt_source"),
                "variants": variants,
                "frame_counts": list(FRAME_COUNTS),
            }
            output_file.write(json.dumps(comparison, ensure_ascii=False, default=str) + "\n")
            output_file.flush()

    print(f"results={args.output}")


if __name__ == "__main__":
    main()
