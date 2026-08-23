#!/usr/bin/env python3
"""Run the six established PhysV examples with Qwen3.8 on one GPU process."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import kernels
import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35


MODEL_PATH = "/data/gaoya/ckpt/Qwen-Qwen3.8-27B-FP8"
PROMPT_PATH = Path("/home/gaoya/Code_Video/Code_vlm_wan/prompts/physv_evidence_temporal_caption_zh.txt")
OUTPUT_PATH = Path("/data/gaoya/agent-data/outputs/physv_qwen3_8/six_cases_gpu7_fla.jsonl")

CASES = (
    ("0613pybullet", "F1_single_object", "sample_000301", "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F1_single_object/sample_000301/video.mp4"),
    ("0613pybullet", "F3_chain_reaction", "sample_000321", "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F3_chain_reaction/sample_000321/video.mp4"),
    ("0613pybullet", "F5_drop_support", "sample_000335", "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/video.mp4"),
    ("phyco_kubric", "ball_drop_soft_v4", "0144a4", "/data/gaoya/dataset/nnsriram97-phyco_kubric/ball_drop_soft_v4/2025-09-05/0144a4/rgba.mp4"),
    ("phyco_kubric", "ball_wall_collision", "00ac15", "/data/gaoya/dataset/nnsriram97-phyco_kubric/ball_wall_collision/2025-08-08/00ac15/rgba.mp4"),
    ("phyco_kubric", "friction_slide_flat_force_v3", "003c2c", "/data/gaoya/dataset/nnsriram97-phyco_kubric/friction_slide_flat_force_v3/2025-10-07/003c2c/rgba.mp4"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--prompt-file", type=Path, default=PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=6_500_000)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    return parser.parse_args()


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def loading_audit(info: dict[str, Any]) -> dict[str, list[str]]:
    return {key: sorted(value) if isinstance(value, (set, list, tuple)) else value for key, value in info.items()}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    print(f"physical_gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"cases={len(CASES)} output={args.output}", flush=True)
    print(f"qwen_fast_path={qwen35.is_fast_path_available}", flush=True)
    started = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model, info = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        max_memory={0: "46GiB"},
        trust_remote_code=True,
        output_loading_info=True,
    )
    model.eval()
    loader = loading_audit(info)
    print(f"model_loaded_seconds={time.time() - started:.3f}", flush=True)
    print(f"loading_counts={{k: len(v) for k, v in loader.items() if isinstance(v, list)}}", flush=True)

    for number, (dataset, family, sample, video) in enumerate(CASES, start=1):
        case_id = f"{dataset}/{family}/{sample}"
        case_started = time.time()
        record: dict[str, Any] = {
            "case_number": number,
            "case_id": case_id,
            "dataset": dataset,
            "family": family,
            "sample": sample,
            "video": video,
            "model": args.model_path,
            "physical_gpu": 7,
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "kernels": kernels.__version__,
                "qwen_fast_path": qwen35.is_fast_path_available,
                "causal_conv_available": qwen35.causal_conv1d_fn is not None,
                "fla_available": qwen35.chunk_gated_delta_rule is not None,
            },
            "prompt": prompt,
            "video_request": {"fps": args.fps, "max_frames": args.max_frames, "max_pixels": args.max_pixels},
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": args.max_new_tokens},
            "loading": loader,
        }
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video, "fps": args.fps, "max_frames": args.max_frames, "max_pixels": args.max_pixels},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            _, video_inputs, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
            )
            raw_video, metadata = video_inputs[0]
            inputs = processor(
                text=[text], videos=[raw_video], video_metadata=[metadata], padding=True,
                return_tensors="pt", **video_kwargs,
            )
            inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
            record["video_metadata"] = str(metadata)
            record["input_shapes"] = {key: list(value.shape) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.80,
                top_k=20,
                repetition_penalty=1.0,
                use_cache=True,
            )
            new_ids = generated[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
            generated_tokens = generated[:, inputs["input_ids"].shape[1]:]
            record["new_token_ids"] = new_ids
            record["raw_output"] = processor.batch_decode(
                generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
            record["caption"] = processor.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            record["status"] = "ok"
        except Exception as error:
            record["status"] = "error"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
        finally:
            record["elapsed_seconds"] = round(time.time() - case_started, 3)
            append_record(args.output, record)
            torch.cuda.empty_cache()
            gc.collect()
            print(
                f"case={number}/{len(CASES)} status={record['status']} "
                f"elapsed_seconds={record['elapsed_seconds']} caption={record.get('caption', record.get('error', ''))!r}",
                flush=True,
            )

    print(f"total_seconds={time.time() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
