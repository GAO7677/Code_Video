#!/usr/bin/env python3
"""Run paired Qwen3.8 captions on an 8-frame context and a full Cycles video."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
MODEL_PATH = Path("/data/gaoya/ckpt/Qwen-Qwen3.8-27B-FP8")
CONTEXT_PROMPT_PATH = Path(
    "/home/gaoya/Code_Video/Code_vlm_wan/prompts/physv_qwen_object_contact_geometry_en.txt"
)
FULL_PROMPT_PATH = Path(
    "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/prompts/physv_full_observed_continuation_en.txt"
)
OUTPUT_PATH = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_context8_vs_full/demo_results.jsonl"
)
CASE_IDS = (
    "difficulty_l2_f11_h085_sr048",
    "difficulty_l2_f12_a024",
    "v2v_gap_038",
    "scene_puck_barrier_n060",
    "scene_door_frame_ball_w054",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--context-prompt", type=Path, default=CONTEXT_PROMPT_PATH)
    parser.add_argument("--full-prompt", type=Path, default=FULL_PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--max-pixels", type=int, default=6_500_000)
    parser.add_argument("--full-fps", type=float, default=15.0)
    parser.add_argument("--full-max-frames", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gib", type=int, default=46)
    return parser.parse_args()


def load_case(case_id: str, dataset_root: Path) -> tuple[Path, dict[str, Any], Path, Path]:
    if Path(case_id).name != case_id or case_id in {"", ".", ".."}:
        raise ValueError(f"Invalid case ID: {case_id!r}")
    json_path = (
        dataset_root
        / "testjsons"
        / "v2v_jsons"
        / "physv_v2v_0819_all_cycles"
        / f"{case_id}.json"
    )
    case = json.loads(json_path.read_text(encoding="utf-8"))
    case_root = dataset_root / "samples" / case_id
    context_video = case_root / "context" / "context8_cycles.mp4"
    full_video = case_root / "videos" / "rgb_cycles.mp4"
    for path in (json_path, context_video, full_video):
        if not path.is_file():
            raise FileNotFoundError(path)
    return json_path, case, context_video, full_video


def normalise_loading(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: sorted(value) if isinstance(value, (set, list, tuple)) else value
        for key, value in info.items()
    }


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or ("cuda" in message and "memory" in message)


def infer_one(
    *,
    processor: Any,
    model: Any,
    process_vision_info: Any,
    torch: Any,
    video_path: Path,
    prompt: str,
    window: str,
    max_pixels: int,
    full_fps: float,
    full_max_frames: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    if window == "context8":
        video_request = {"nframes": 8, "max_pixels": max_pixels}
        video_content = {
            "type": "video",
            "video": str(video_path),
            "nframes": 8,
            "max_pixels": max_pixels,
        }
    else:
        video_request = {
            "fps": full_fps,
            "max_frames": full_max_frames,
            "max_pixels": max_pixels,
        }
        video_content = {
            "type": "video",
            "video": str(video_path),
            "fps": full_fps,
            "max_frames": full_max_frames,
            "max_pixels": max_pixels,
        }

    started = time.time()
    result: dict[str, Any] = {
        "window": window,
        "video": str(video_path),
        "video_request": video_request,
        "prompt": prompt,
        "status": "error",
    }
    try:
        messages = [{"role": "user", "content": [video_content, {"type": "text", "text": prompt}]}]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        _, video_inputs, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        raw_video, metadata = video_inputs[0]
        inputs = processor(
            text=[chat_text], videos=[raw_video], video_metadata=[metadata],
            padding=True, return_tensors="pt", **video_kwargs,
        )
        inputs = {
            key: value.to("cuda:0") if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        result["video_metadata"] = str(metadata)
        result["input_shapes"] = {
            key: list(value.shape)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
        generated_tokens = generated[:, inputs["input_ids"].shape[1] :]
        result["response_raw"] = processor.batch_decode(
            generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        result["text"] = processor.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        result["status"] = "ok"
    except Exception as exc:  # Keep the paired record inspectable on failure.
        result.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "oom": is_oom_error(exc),
        })
    finally:
        result["elapsed_seconds"] = round(time.time() - started, 3)
        gc.collect()
        torch.cuda.empty_cache()
    return result


def main() -> int:
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

    case_ids = list(args.case_ids or CASE_IDS)
    loaded_cases = [load_case(case_id, args.dataset_root) for case_id in case_ids]
    context_prompt = args.context_prompt.read_text(encoding="utf-8").strip()
    full_prompt = args.full_prompt.read_text(encoding="utf-8").strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["case_id"]] = row

    print(f"model={args.model_path}", flush=True)
    print(f"physical_gpu={args.physical_gpu} device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"cases={len(loaded_cases)} output={args.output}", flush=True)
    print("caption_source=RGB frames only; metadata and truth files are not model inputs", flush=True)
    print(
        f"context_request=nframes:8,max_pixels:{args.max_pixels}; "
        f"full_request=fps:{args.full_fps},max_frames:{args.full_max_frames},max_pixels:{args.max_pixels}",
        flush=True,
    )

    pending = [item for item in loaded_cases if existing.get(item[0].stem, {}).get("status") != "ok"]
    processor = model = loading = None
    if pending:
        started = time.time()
        processor = AutoProcessor.from_pretrained(
            str(args.model_path), local_files_only=True, trust_remote_code=True
        )
        print("processor_loaded", flush=True)
        model, loading_info = AutoModelForImageTextToText.from_pretrained(
            str(args.model_path), torch_dtype="auto", device_map={"": "cuda:0"},
            low_cpu_mem_usage=True, max_memory={0: f"{args.max_memory_gib}GiB"},
            trust_remote_code=True, output_loading_info=True,
        )
        model.eval()
        loading = normalise_loading(loading_info)
        print(
            f"model_loaded_seconds={time.time() - started:.3f} fast_path={qwen35.is_fast_path_available}",
            flush=True,
        )

    for number, (json_path, case, context_video, full_video) in enumerate(loaded_cases, start=1):
        case_id = json_path.stem
        if existing.get(case_id, {}).get("status") == "ok":
            print(f"skip={number}/{len(loaded_cases)} case={case_id}", flush=True)
            continue
        print(f"[{number}/{len(loaded_cases)}] {case_id}", flush=True)
        row: dict[str, Any] = {
            "schema_version": "physv_context8_vs_full_v1",
            "dataset": "physv_v2v_0819",
            "source_basis": "RGB Cycles video frames only",
            "caption_inputs_exclude": ["physics_supervision.npz", "contacts.json", "raw/trajectories.npz", "raw/masks.npz", "raw/depth.npz"],
            "case_id": case_id,
            "case_json": str(json_path),
            "title": case.get("title"),
            "taxonomy": case.get("taxonomy"),
            "task_type": case.get("task_type"),
            "source_group": case.get("source_group"),
            "control": case.get("control"),
            "video_variant": "cycles_pbr",
            "context": {"video_rel": "context/context8_cycles.mp4"},
            "full": {"video_rel": "videos/rgb_cycles.mp4"},
            "prompts": {"context_file": str(args.context_prompt), "full_file": str(args.full_prompt)},
            "model": str(args.model_path),
            "physical_gpu": args.physical_gpu,
            "runtime": {
                "torch": torch.__version__, "transformers": transformers.__version__,
                "kernels": kernels.__version__, "qwen_fast_path": qwen35.is_fast_path_available,
            },
            "status": "error",
        }
        if loading is not None:
            row["loading"] = loading
        row["context"] = infer_one(
            processor=processor, model=model, process_vision_info=process_vision_info, torch=torch,
            video_path=context_video, prompt=context_prompt, window="context8",
            max_pixels=args.max_pixels, full_fps=args.full_fps, full_max_frames=args.full_max_frames,
            max_new_tokens=args.max_new_tokens,
        )
        row["context"]["video_rel"] = "context/context8_cycles.mp4"
        print(f"  context8={row['context']['status']} elapsed={row['context']['elapsed_seconds']}s", flush=True)
        row["full"] = infer_one(
            processor=processor, model=model, process_vision_info=process_vision_info, torch=torch,
            video_path=full_video, prompt=full_prompt, window="full",
            max_pixels=args.max_pixels, full_fps=args.full_fps, full_max_frames=args.full_max_frames,
            max_new_tokens=args.max_new_tokens,
        )
        row["full"]["video_rel"] = "videos/rgb_cycles.mp4"
        print(f"  full={row['full']['status']} elapsed={row['full']['elapsed_seconds']}s", flush=True)
        row["status"] = "ok" if row["context"]["status"] == "ok" and row["full"]["status"] == "ok" else "partial"
        row["elapsed_seconds"] = round(row["context"]["elapsed_seconds"] + row["full"]["elapsed_seconds"], 3)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        existing[case_id] = row

    print(f"results={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
