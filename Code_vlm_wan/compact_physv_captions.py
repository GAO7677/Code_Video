#!/usr/bin/env python3
"""Enforce concise physical captions without changing the visual inference inputs."""

from __future__ import annotations

import argparse
import json
import os
import re
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

DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8"
DEFAULT_INPUT = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "0613_phyco_evidence_pipeline_final_fps15_maxpixels6500000.jsonl"
)
DEFAULT_OUTPUT = (
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/"
    "0613_phyco_evidence_pipeline_final_compact_fps15_maxpixels6500000.jsonl"
)

COMPACTION_PROMPT = """将下面的候选物理视频描述压缩为最多4句中文。

只能合并连续的重复描述或删除冗余描述；不得增加、替换、否定或解释候选描述中没有的物体、事件、接触关系、运动、方向、姿态或时间顺序。保留候选描述中最早和最后的可见状态。不要添加标题、标签、分析或说明，只输出压缩后的 caption。

候选描述：
{caption}"""


# Keep text-only validation importable without the video inference stack.
def final_answer(text: str) -> str:
    for marker in ("</think>", "<｜end▁of▁thinking｜>"):
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return text.strip()


def disable_thinking(vllm_input: dict[str, Any]) -> None:
    thinking_prompt = "<|im_start|>assistant\n<think>\n"
    answer_prompt = "<|im_start|>assistant\n<think>\n</think>\n\n"
    if vllm_input["prompt"].endswith(thinking_prompt):
        vllm_input["prompt"] = vllm_input["prompt"][: -len(thinking_prompt)] + answer_prompt


def sentence_count(text: str) -> int:
    return len(sentence_parts(text))


def sentence_parts(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", text.strip()) if part.strip()]


def deduplicate_exact_sentences(text: str) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for sentence in sentence_parts(text):
        normalized = re.sub(r"\s+", "", sentence)
        if normalized not in seen:
            unique.append(sentence)
            seen.add(normalized)
    return "".join(unique)


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


def make_input(processor: Any, caption: str) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": COMPACTION_PROMPT.format(caption=caption)}],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    request = {"prompt": prompt}
    disable_thinking(request)
    return request


def compact_caption(llm: Any, request: dict[str, Any], max_tokens: int) -> tuple[str, str, float]:
    from vllm import SamplingParams

    started = time.time()
    output = llm.generate(
        [request],
        sampling_params=SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_tokens),
        use_tqdm=False,
    )[0]
    raw = output.outputs[0].text
    return raw, final_answer(raw), round(time.time() - started, 3)


def shutdown_llm(llm: Any) -> None:
    llm.llm_engine.engine_core.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.79)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    from transformers import AutoProcessor
    from vllm import LLM

    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(args.model_path)
    rows = load_jsonl(args.input)
    wanted = set(args.case_ids or [row["case_id"] for row in rows])
    selected = [row for row in rows if row["case_id"] in wanted]
    missing = wanted - {row["case_id"] for row in selected}
    if missing:
        raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    if not selected:
        raise ValueError("No cases selected")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("Output must not overwrite input")

    print(f"cases={len(selected)}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    print("processor_loaded")
    if args.dry_run:
        for row in selected:
            caption = str(row.get("response_final") or "")
            print(f"prepared={row['case_id']} sentences={sentence_count(caption)}")
        return

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        skip_mm_profiling=True,
        enforce_eager=True,
        seed=0,
    )
    print("model_loaded")
    try:
        output_rows: list[dict[str, Any]] = []
        for position, source in enumerate(selected, start=1):
            result = dict(source)
            original = str(source.get("response_final") or "").strip()
            if not original:
                raise ValueError(f"Missing response_final for {source['case_id']}")
            deduplicated = deduplicate_exact_sentences(original)
            input_sentences = sentence_count(original)
            deduplicated_sentences = sentence_count(deduplicated)
            raw = None
            elapsed = 0.0
            if deduplicated_sentences <= 4:
                compact = deduplicated
                method = "deterministic_exact_sentence_deduplicate"
            else:
                raw, compact, elapsed = compact_caption(
                    llm, make_input(processor, deduplicated), args.max_tokens
                )
                method = "text_only_deduplicate_and_merge"
                if sentence_count(compact) > 4:
                    compact = deduplicated
                    method = "deterministic_deduplicate_fallback"
            final_sentences = sentence_count(compact)
            if final_sentences > 4:
                raise ValueError(
                    f"Compaction exceeded four sentences for {source['case_id']}: {final_sentences}"
                )
            result["response_before_compaction"] = original
            result["response_final"] = compact
            result["caption_compaction"] = {
                "method": method,
                "prompt": COMPACTION_PROMPT,
                "input_sentence_count": input_sentences,
                "deduplicated_sentence_count": deduplicated_sentences,
                "output_sentence_count": final_sentences,
                "response_raw": raw,
                "elapsed_seconds": elapsed,
                "thinking_enabled": False,
                "status": "ok",
            }
            output_rows.append(result)
            write_jsonl_atomically(args.output, output_rows)
            print(
                f"[{position}/{len(selected)}] {source['case_id']} "
                f"sentences={sentence_count(original)}->{final_sentences}",
                flush=True,
            )
            print(f"caption={compact!r}", flush=True)
        print(f"results={args.output}")
    finally:
        shutdown_llm(llm)


if __name__ == "__main__":
    main()
