#!/usr/bin/env python3
"""Run Qwen3-VL video QA on a small PhysV sample."""

import argparse
import gc
import glob
import json
import os
import time
from pathlib import Path

# Keep the inference process on the PyTorch path in environments that also
# contain TensorFlow or Flax installations.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
# Avoid a runtime FlashInfer JIT build in this environment; vLLM's native
# sampler is sufficient for the single-sequence probe.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8"
DEFAULT_DATASET = "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
DEFAULT_OUTPUT = "/data/gaoya/agent-data/outputs/physv_qwen3vl/cases.jsonl"
ANALYSIS_PROMPT = """你是一个视频物理动态分析器。

分析这段视频中可观察到的物理动态，重点关注物体的平移、旋转、接触和相互作用，以及运动方向、速度、强度和运动模式随时间的变化。特别关注碰撞、反弹、滑动、滚动、摇摆、加速、减速、反向和趋于稳定等动态过程，以及视频结束时仍然存在的运动状态。

弱化与物理行为无关的颜色、纹理和语义外观信息，只依据视频中实际观察到的内容进行分析，不要预测视频结束后的事件。"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=360 * 640)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    return parser.parse_args()


def find_videos(dataset_root, num_cases):
    pattern = str(Path(dataset_root) / "cases" / "F*" / "*" / "videos" / "*.mp4")
    videos = sorted(Path(path) for path in glob.glob(pattern))
    if not videos:
        raise FileNotFoundError(f"No PhysV videos found under {dataset_root}")
    return videos[:num_cases]


def load_original_prompt(video_path):
    case_dir = video_path.parent.parent
    metadata_paths = [
        case_dir / "case_manifest.json",
        case_dir / "meta" / f"{video_path.stem}.json",
    ]
    for metadata_path in metadata_paths:
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        prompt = (
            metadata.get("caption")
            or metadata.get("input_caption")
            or metadata.get("short_caption")
        )
        if prompt:
            return {"text": prompt, "source": str(metadata_path)}
    return {"text": "", "source": None}


def build_question(original_prompt):
    prompt_text = original_prompt["text"]
    if not prompt_text:
        return ANALYSIS_PROMPT
    return (
        f"{ANALYSIS_PROMPT}\n\n"
        "本视频数据集的原始场景描述（仅作辅助上下文，必须以视频实际观察为准）：\n"
        f"{prompt_text}\n\n"
        "请核对上述描述与视频内容；如果不一致，以视频中实际观察到的内容为准。"
    )


def build_messages(video_path, args, question):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "fps": args.fps,
                    "max_frames": args.max_frames,
                    "max_pixels": args.max_pixels,
                },
                {"type": "text", "text": question},
            ],
        }
    ]


def prepare_vllm_input(messages, processor):
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    multimodal_data = {}
    if image_inputs is not None:
        multimodal_data["image"] = image_inputs
    if video_inputs is not None:
        multimodal_data["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": multimodal_data,
        "mm_processor_kwargs": video_kwargs,
    }, video_inputs


def get_video_info(video_inputs):
    if not video_inputs:
        return None
    item = video_inputs[0]
    if isinstance(item, tuple):
        tensor, metadata = item
    else:
        tensor, metadata = item, None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "metadata": metadata,
    }


def final_answer(text):
    for marker in ("</think>", "<｜end▁of▁thinking｜>"):
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return text.strip()


def is_oom_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def main():
    args = parse_args()
    videos = find_videos(args.dataset_root, args.num_cases)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model_path}")
    print(f"videos={len(videos)}")
    print(
        "video_params="
        f"fps={args.fps}, max_frames={args.max_frames}, max_pixels={args.max_pixels}"
    )
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    print("processor_loaded")

    try:
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
    except Exception as exc:
        print(f"model_load_error={type(exc).__name__}: {exc}")
        if is_oom_error(exc):
            print("model_load_oom=true; video parameters cannot fix a model-load OOM")
        raise

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_new_tokens,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, video_path in enumerate(videos, start=1):
            case_id = video_path.parent.parent.name
            original_prompt = load_original_prompt(video_path)
            question = build_question(original_prompt)
            messages = build_messages(video_path, args, question)
            print(f"[{index}/{len(videos)}] preparing {case_id}", flush=True)
            print(
                f"[{index}/{len(videos)}] original_prompt="
                f"{original_prompt['text']!r} source={original_prompt['source']}",
                flush=True,
            )
            started = time.time()
            try:
                vllm_input, video_inputs = prepare_vllm_input(messages, processor)
                video_info = get_video_info(video_inputs)
                print(f"[{index}/{len(videos)}] video={video_info}", flush=True)
                outputs = llm.generate(
                    [vllm_input], sampling_params=sampling_params, use_tqdm=False
                )
                raw_text = outputs[0].outputs[0].text
                result = {
                    "case_id": case_id,
                    "video": str(video_path),
                    "question": question,
                    "analysis_prompt": ANALYSIS_PROMPT,
                    "original_prompt": original_prompt["text"],
                    "original_prompt_source": original_prompt["source"],
                    "video_params": {
                        "fps": args.fps,
                        "max_frames": args.max_frames,
                        "max_pixels": args.max_pixels,
                    },
                    "video_info": video_info,
                    "response_raw": raw_text,
                    "response_final": final_answer(raw_text),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "status": "ok",
                }
                print(f"[{index}/{len(videos)}] response={raw_text!r}", flush=True)
            except Exception as exc:
                result = {
                    "case_id": case_id,
                    "video": str(video_path),
                    "question": question,
                    "analysis_prompt": ANALYSIS_PROMPT,
                    "original_prompt": original_prompt["text"],
                    "original_prompt_source": original_prompt["source"],
                    "video_params": {
                        "fps": args.fps,
                        "max_frames": args.max_frames,
                        "max_pixels": args.max_pixels,
                    },
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "oom": is_oom_error(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                    "status": "error",
                }
                print(f"[{index}/{len(videos)}] error={type(exc).__name__}: {exc}", flush=True)
                if is_oom_error(exc):
                    print("input_or_generation_oom=true", flush=True)
            output_file.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            output_file.flush()
            gc.collect()

    print(f"results={output_path}")


if __name__ == "__main__":
    main()
