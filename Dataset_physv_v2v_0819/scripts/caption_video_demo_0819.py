#!/usr/bin/env python3
"""Generate frame-only captions for a small PhysV V2V demo set."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_MODEL = Path("/data/gaoya/ckpt/Qwen-Qwen3.8-27B-FP8")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_video_caption_demo/results.jsonl"
)
DEFAULT_CASES = (
    "v2v_gap_006",
    "v2v_gap_038",
    "scene_puck_barrier_n060",
)

SPECIFIC_PROMPT = """Describe this complete video using only what is visually observable in its RGB frames.

Write one concise, temporally ordered caption focused on physical dynamics. Identify the moving objects and relevant scene geometry, then describe translation, rotation, contacts, collisions, changes of direction or motion mode, and the state near the end. Distinguish the actual outcome shown in this particular video, such as passing through, falling into, bouncing from, sliding along, rotating, or becoming blocked. Use concrete spatial descriptions when they are visible. Do not read or infer metadata, simulation parameters, object IDs, trajectories, contact annotations, or events outside the video. Do not invent numeric values unless a number is visibly written in the video. Do not predict what happens after the video ends. Output only the caption, in no more than 5 sentences and 120 English words."""

ABSTRACT_PROMPT = """Describe this complete video using only what is visually observable in its RGB frames.

Write one concise, temporally ordered caption focused on the physical dynamics. Keep the description general and do not mention numeric values, control-variable names, case IDs, metadata, simulation parameters, object IDs, trajectories, contact annotations, or events outside the video. Still preserve the actual outcome visible in this video, including whether an object passes, falls, bounces, slides, rotates, or becomes blocked. Mention relevant contacts and the motion state near the end when clear. Do not predict what happens after the video ends. Output only the caption, in no more than 4 sentences and 90 English words."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Case ID to include; may be repeated. Defaults to three representative demos.",
    )
    parser.add_argument("--case-list", type=Path, default=None)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=6_500_000)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.79)
    parser.add_argument("--skip-mm-profiling", action="store_true")
    return parser.parse_args()


def load_case_ids(args: argparse.Namespace) -> list[str]:
    if args.case_list is not None:
        case_ids = [
            line.strip()
            for line in args.case_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif args.case_ids:
        case_ids = list(args.case_ids)
    else:
        case_ids = list(DEFAULT_CASES)
    if not case_ids:
        raise ValueError("No case IDs were provided")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique")
    return case_ids


def case_video(dataset_root: Path, case_id: str) -> Path:
    if Path(case_id).name != case_id or case_id in {"", ".", ".."}:
        raise ValueError(f"Invalid case ID: {case_id!r}")
    video = dataset_root / "samples" / case_id / "videos" / "rgb_cycles.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    return video


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message and "memory" in message


def run() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")
    import kernels  # noqa: WPS433
    import torch  # noqa: WPS433
    import transformers  # noqa: WPS433
    from qwen_vl_utils import process_vision_info  # noqa: WPS433
    from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: WPS433
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35  # noqa: WPS433

    case_ids = load_case_ids(args)
    videos = [(case_id, case_video(args.dataset_root, case_id)) for case_id in case_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model_path}", flush=True)
    print(f"cases={len(videos)}", flush=True)
    print(
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(
        f"video_params=fps:{args.fps},max_frames:{args.max_frames},"
        f"max_pixels:{args.max_pixels}",
        flush=True,
    )
    for case_id, video in videos:
        print(f"selected={case_id} {video}", flush=True)

    processor = AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=True
    )
    print("processor_loaded", flush=True)
    started = time.time()
    model, loading_info = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path),
        torch_dtype="auto",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        max_memory={0: "42GiB"},
        trust_remote_code=True,
    )
    model.eval()
    print(
        f"model_loaded_seconds={time.time() - started:.3f} "
        f"fast_path={qwen35.is_fast_path_available}",
        flush=True,
    )

    prompts = {"specific": SPECIFIC_PROMPT, "abstract": ABSTRACT_PROMPT}
    with args.output.open("w", encoding="utf-8") as handle:
        for index, (case_id, video) in enumerate(videos, start=1):
            result = {
                "schema_version": "physv_caption_v3_full_video_demo",
                "source_basis": "full_rgb_video_frames_only",
                "case_id": case_id,
                "video": str(video),
                "video_variant": "cycles_pbr_full",
                "video_params": {
                    "fps": args.fps,
                    "max_frames": args.max_frames,
                    "max_pixels": args.max_pixels,
                },
                "model": str(args.model_path),
                "physical_gpu": args.physical_gpu,
                "captions": {},
                "status": "ok",
            }
            print(f"[{index}/{len(videos)}] {case_id}", flush=True)
            for variant, question in prompts.items():
                started = time.time()
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": str(video),
                                    "fps": args.fps,
                                    "max_frames": args.max_frames,
                                    "max_pixels": args.max_pixels,
                                },
                                {"type": "text", "text": question},
                            ],
                        }
                    ]
                    chat_text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    _, video_inputs, video_kwargs = process_vision_info(
                        messages,
                        image_patch_size=16,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )
                    raw_video, metadata = video_inputs[0]
                    inputs = processor(
                        text=[chat_text],
                        videos=[raw_video],
                        video_metadata=[metadata],
                        padding=True,
                        return_tensors="pt",
                        **video_kwargs,
                    )
                    inputs = {
                        key: value.to("cuda:0") if hasattr(value, "to") else value
                        for key, value in inputs.items()
                    }
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.8,
                        top_k=20,
                        repetition_penalty=1.0,
                        use_cache=True,
                    )
                    generated_tokens = generated[:, inputs["input_ids"].shape[1] :]
                    raw = processor.batch_decode(
                        generated_tokens,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    text = processor.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    result["captions"][variant] = {
                        "prompt": question,
                        "response_raw": raw,
                        "text": text,
                        "video_metadata": str(metadata),
                        "input_shapes": {
                            key: list(value.shape)
                            for key, value in inputs.items()
                            if isinstance(value, torch.Tensor)
                        },
                        "elapsed_seconds": round(time.time() - started, 3),
                        "status": "ok",
                    }
                    print(f"  {variant}: {result['captions'][variant]['text']}", flush=True)
                except Exception as exc:  # keep later demos runnable after one failure
                    result["status"] = "error"
                    result["captions"][variant] = {
                        "prompt": question,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "oom": is_oom_error(exc),
                        "elapsed_seconds": round(time.time() - started, 3),
                        "status": "error",
                    }
                    print(f"  {variant}: {type(exc).__name__}: {exc}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            handle.flush()
    print(f"results={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
