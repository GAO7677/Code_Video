#!/usr/bin/env python3
"""Run the existing Qwen3-VL physics prompt on two dataset layouts."""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from infer_physv_qwen3vl import (  # noqa: E402
    ANALYSIS_PROMPT,
    build_messages,
    build_question,
    final_answer,
    get_video_info,
    is_oom_error,
    prepare_vllm_input,
)
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8"
DEFAULT_0613_ROOT = "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet"
DEFAULT_PHYCO_ROOT = "/data/gaoya/dataset/nnsriram97-phyco_kubric"
DEFAULT_OUTPUT = "/data/gaoya/agent-data/outputs/physv_qwen3vl/0613_phyco_smoke.jsonl"
ANSWER_CONSTRAINT = """

请只输出最终观察结论，不要展示思考过程，也不要重复场景描述。严格限制为不超过 4 句话、120 个英文单词；按时间顺序说明主要运动、接触/碰撞和视频末状态。无法确认的细节不要臆测。
""".strip()


@dataclass(frozen=True)
class Case:
    dataset: str
    case_id: str
    video_path: Path
    context_text: str
    context_source: str | None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def select_0613_cases(root: Path, count: int) -> list[Case]:
    videos = sorted(root.glob("raw_v1/*/val/*/sample_*/video.mp4"))
    if not videos:
        raise FileNotFoundError(f"No 0613 raw val videos found under {root}")

    preferred_families = [
        "F1_single_object",
        "F3_chain_reaction",
        "F5_drop_support",
    ]
    by_family = {}
    for video_path in videos:
        family = video_path.parent.parent.name
        by_family.setdefault(family, video_path)

    selected = []
    for family in preferred_families:
        if family in by_family:
            selected.append(by_family[family])
    for video_path in videos:
        if video_path not in selected:
            selected.append(video_path)
        if len(selected) >= count:
            break

    cases = []
    for video_path in selected[:count]:
        case_dir = video_path.parent
        metadata_path = case_dir / "meta.json"
        metadata = read_json(metadata_path)
        context_parts = [
            value
            for value in (metadata.get("title"), metadata.get("description"))
            if value
        ]
        relative = video_path.relative_to(root)
        case_id = "0613pybullet/" + "/".join(relative.parts[:-1])
        cases.append(
            Case(
                dataset="0613pybullet",
                case_id=case_id,
                video_path=video_path,
                context_text=". ".join(context_parts),
                context_source=str(metadata_path) if context_parts else None,
            )
        )
    return cases


def select_phyco_cases(root: Path, count: int) -> list[Case]:
    split_path = root / "test_500_balanced.txt"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    preferred_scenarios = [
        "ball_drop_soft_v4",
        "ball_wall_collision",
        "friction_slide_flat_force_v3",
    ]
    candidates = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        case_dir = Path(line.strip())
        video_path = case_dir / "rgba.mp4"
        if video_path.is_file():
            candidates.append(video_path)

    by_scenario = {}
    for video_path in candidates:
        scenario = video_path.parents[2].name
        by_scenario.setdefault(scenario, video_path)

    selected = []
    for scenario in preferred_scenarios:
        if scenario in by_scenario:
            selected.append(by_scenario[scenario])
    for video_path in candidates:
        if video_path not in selected:
            selected.append(video_path)
        if len(selected) >= count:
            break

    cases = []
    for video_path in selected[:count]:
        scenario_dir = video_path.parents[2]
        scenario = scenario_dir.name
        context_path = scenario_dir / "common_caption_cosmos.txt"
        context_text = ""
        if context_path.is_file():
            context_text = context_path.read_text(encoding="utf-8").strip()
        relative = video_path.relative_to(root)
        case_id = "phyco/test_500_balanced/" + "/".join(relative.parts[:-1])
        cases.append(
            Case(
                dataset="phyco_kubric",
                case_id=case_id,
                video_path=video_path,
                context_text=context_text,
                context_source=str(context_path) if context_text else None,
            )
        )
    return cases


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--0613-root", default=DEFAULT_0613_ROOT)
    parser.add_argument("--phyco-root", default=DEFAULT_PHYCO_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Use this exact user prompt instead of the default context-augmented prompt.",
    )
    parser.add_argument("--num-cases-per-dataset", type=int, default=3)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-pixels", type=int, default=360 * 640)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Use the answer-only assistant prompt for Thinking checkpoints.",
    )
    return parser.parse_args()


def run_case(case: Case, args, processor, llm, sampling_params):
    original_prompt = {
        "text": case.context_text,
        "source": case.context_source,
    }
    question = args.prompt_text or (
        build_question(original_prompt) + "\n\n" + ANSWER_CONSTRAINT
    )
    messages = build_messages(case.video_path, args, question)
    started = time.time()
    result = {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "video": str(case.video_path),
        "question": question,
        "analysis_prompt": args.prompt_text or ANALYSIS_PROMPT,
        "prompt_source": str(args.prompt_file) if args.prompt_file else "default",
        "original_prompt": case.context_text,
        "original_prompt_source": case.context_source,
        "video_params": {
            "fps": args.fps,
            "max_frames": args.max_frames,
            "max_pixels": args.max_pixels,
        },
        "thinking_disabled": args.disable_thinking,
    }
    try:
        vllm_input, video_inputs = prepare_vllm_input(messages, processor)
        if args.disable_thinking:
            thinking_prompt = "<|im_start|>assistant\n<think>\n"
            answer_prompt = "<|im_start|>assistant\n<think>\n</think>\n\n"
            if vllm_input["prompt"].endswith(thinking_prompt):
                vllm_input["prompt"] = (
                    vllm_input["prompt"][: -len(thinking_prompt)] + answer_prompt
                )
        result["video_info"] = get_video_info(video_inputs)
        outputs = llm.generate(
            [vllm_input], sampling_params=sampling_params, use_tqdm=False
        )
        raw_text = outputs[0].outputs[0].text
        result.update(
            {
                "response_raw": raw_text,
                "response_final": final_answer(raw_text),
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
    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def main():
    args = parse_args()
    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            raise FileNotFoundError(args.prompt_file)
        args.prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    else:
        args.prompt_text = None
    count = max(args.num_cases_per_dataset, 1)
    cases = select_0613_cases(Path(args.__dict__["0613_root"]), count)
    cases += select_phyco_cases(Path(args.phyco_root), count)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"model={args.model_path}")
    print(f"cases={len(cases)}")
    for case in cases:
        print(f"selected={case.dataset} {case.case_id} {case.video_path}")
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
            print("model_load_oom=true")
        raise

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] preparing {case.case_id}", flush=True)
            print(f"context_source={case.context_source}", flush=True)
            result = run_case(case, args, processor, llm, sampling_params)
            output_file.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            output_file.flush()
            gc.collect()

    print(f"results={output_path}")


if __name__ == "__main__":
    main()
