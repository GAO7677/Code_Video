#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path
from typing import Any

import torch

from rerank_video.generators import VaceGenerator
from rerank_video.pdi_proxy_eval import VaceTI2VRunner, WanTI2VRunner
from rerank_video.schemas import GeneratorConfig, InputSpec
from rerank_video.video_utils import (
    ensure_dir,
    load_context_frames,
    pil_list_to_numpy,
    save_video_frames,
    write_json,
)


DATASET_ROOT = Path("/data/gaoya/dataset/physics-iq-benchmark/mytest")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark")
BENCHMARK_NAME = "physics-iq-benchmark"
METHODS = ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]

WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")

SEED = 42
FPS = 16
NUM_FRAMES = 49
CONTEXT_FRAMES = 8
NUM_INFERENCE_STEPS = 30
CFG_SCALE = 5.0
QUALITY = 5
WIDTH = 672
HEIGHT = 384
NEGATIVE_PROMPT = ""
DEVICE = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PDI-style method folders for physics-iq-benchmark.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_meta_paths(limit: int | None) -> list[Path]:
    meta_paths = sorted(DATASET_ROOT.glob("*/meta.json"))
    if limit is not None:
        meta_paths = meta_paths[:limit]
    return meta_paths


def load_case(meta_path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    paths = payload.get("paths") or {}
    return {
        "sample_id": str(payload["sample_id"]),
        "caption": str(payload["caption"]),
        "category": str(payload.get("category") or ""),
        "scenario": payload.get("scenario"),
        "sample_dir": meta_path.parent,
        "meta_json_path": meta_path,
        "context_video_path": Path(paths["context_video_path"]),
        "future_gt_video_path": Path(paths["future_gt_video_path"]),
        "full_video_path": Path(paths["full_video_path"]),
        "first_frame_path": Path(paths["first_frame_path"]),
    }


def method_dir(output_root: Path, method: str) -> Path:
    return output_root / "output" / method / BENCHMARK_NAME


def output_paths(output_root: Path, method: str, sample_id: str) -> tuple[Path, Path]:
    base = method_dir(output_root, method) / sample_id
    return base.with_suffix(".mp4"), base.with_suffix(".json")


def is_complete(video_path: Path, json_path: Path, overwrite: bool) -> bool:
    return not overwrite and video_path.is_file() and json_path.is_file()


def build_payload(
    *,
    case: dict[str, Any],
    method: str,
    output_video_path: Path,
    output_json_path: Path,
    conditioning_mode: str,
    context_frames: int,
    num_inference_steps: int,
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset": BENCHMARK_NAME,
        "method": method,
        "task": BENCHMARK_NAME,
        "clip_name": case["sample_id"],
        "sample_id": case["sample_id"],
        "prompt": case["caption"],
        "caption": case["caption"],
        "category": case["category"],
        "scenario": case["scenario"],
        "video": str(output_video_path),
        "video_path": str(output_video_path),
        "context_video": str(case["context_video_path"]),
        "future_gt_video": str(case["future_gt_video_path"]),
        "full_video": str(case["full_video_path"]),
        "first_frame": str(case["first_frame_path"]),
        "seed": SEED,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": num_inference_steps,
        "cfg_scale": CFG_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "conditioning_mode": conditioning_mode,
        "context_frames": context_frames,
        "negative_prompt": NEGATIVE_PROMPT,
        "paths": {
            "sample_dir": str(case["sample_dir"]),
            "context_video_path": str(case["context_video_path"]),
            "future_gt_video_path": str(case["future_gt_video_path"]),
            "full_video_path": str(case["full_video_path"]),
            "first_frame_path": str(case["first_frame_path"]),
            "meta_json_path": str(case["meta_json_path"]),
            "output_video_path": str(output_video_path),
            "output_json_path": str(output_json_path),
        },
    }


def build_gt_video(case: dict[str, Any], output_video_path: Path) -> None:
    context_frames = load_context_frames(
        case["context_video_path"],
        context_frames=CONTEXT_FRAMES,
        width=WIDTH,
        height=HEIGHT,
        resize_mode="crop",
    )
    future_frames = load_context_frames(
        case["future_gt_video_path"],
        context_frames=NUM_FRAMES - CONTEXT_FRAMES,
        width=WIDTH,
        height=HEIGHT,
        resize_mode="crop",
    )
    frames = pil_list_to_numpy(context_frames + future_frames)
    save_video_frames(output_video_path, frames, fps=FPS, quality=QUALITY)


def run_gt(output_root: Path, cases: list[dict[str, Any]], overwrite: bool) -> None:
    for case in cases:
        output_video_path, output_json_path = output_paths(output_root, "GT", case["sample_id"])
        if is_complete(output_video_path, output_json_path, overwrite):
            print(f"[skip] GT {case['sample_id']}", flush=True)
            continue
        print(f"[run] GT {case['sample_id']}", flush=True)
        ensure_dir(output_video_path.parent)
        build_gt_video(case, output_video_path)
        write_json(
            output_json_path,
            build_payload(
                case=case,
                method="GT",
                output_video_path=output_video_path,
                output_json_path=output_json_path,
                conditioning_mode="GT_context08_plus_future41",
                context_frames=CONTEXT_FRAMES,
                num_inference_steps=0,
            ),
        )


def run_wan_ti2v(output_root: Path, cases: list[dict[str, Any]], overwrite: bool) -> None:
    runner = WanTI2VRunner(model_root=WAN_ROOT, device=DEVICE)
    for case in cases:
        output_video_path, output_json_path = output_paths(output_root, "wan22-5B-TI2V", case["sample_id"])
        if is_complete(output_video_path, output_json_path, overwrite):
            print(f"[skip] wan22-5B-TI2V {case['sample_id']}", flush=True)
            continue
        print(f"[run] wan22-5B-TI2V {case['sample_id']} seed={SEED}", flush=True)
        ensure_dir(output_video_path.parent)
        runner.generate(
            first_frame_path=case["first_frame_path"],
            prompt=case["caption"],
            output_path=output_video_path,
            seed=SEED,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            fps=FPS,
            num_inference_steps=NUM_INFERENCE_STEPS,
            cfg_scale=CFG_SCALE,
            quality=QUALITY,
        )
        write_json(
            output_json_path,
            build_payload(
                case=case,
                method="wan22-5B-TI2V",
                output_video_path=output_video_path,
                output_json_path=output_json_path,
                conditioning_mode="TI2V_first_frame",
                context_frames=1,
                num_inference_steps=NUM_INFERENCE_STEPS,
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vace_ti2v(output_root: Path, cases: list[dict[str, Any]], overwrite: bool) -> None:
    runner = VaceTI2VRunner(model_root=VACE_ROOT, device=DEVICE)
    for case in cases:
        output_video_path, output_json_path = output_paths(output_root, "VACE_1p3B_TI2V", case["sample_id"])
        if is_complete(output_video_path, output_json_path, overwrite):
            print(f"[skip] VACE_1p3B_TI2V {case['sample_id']}", flush=True)
            continue
        print(f"[run] VACE_1p3B_TI2V {case['sample_id']} seed={SEED}", flush=True)
        ensure_dir(output_video_path.parent)
        runner.generate(
            first_frame_path=case["first_frame_path"],
            prompt=case["caption"],
            output_path=output_video_path,
            seed=SEED,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            fps=FPS,
            num_inference_steps=NUM_INFERENCE_STEPS,
            cfg_scale=CFG_SCALE,
            quality=QUALITY,
        )
        write_json(
            output_json_path,
            build_payload(
                case=case,
                method="VACE_1p3B_TI2V",
                output_video_path=output_video_path,
                output_json_path=output_json_path,
                conditioning_mode="TI2V_first_frame",
                context_frames=1,
                num_inference_steps=NUM_INFERENCE_STEPS,
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vace_ctx08(output_root: Path, cases: list[dict[str, Any]], overwrite: bool) -> None:
    config = GeneratorConfig(
        key="vace_ctx08",
        type="vace",
        enabled=True,
        device=DEVICE,
        model_root=VACE_ROOT,
        num_candidates=1,
        base_seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        num_frames=NUM_FRAMES,
        context_frames=CONTEXT_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
        quality=QUALITY,
        negative_prompt=NEGATIVE_PROMPT,
    )
    runner = VaceGenerator(config)
    for case in cases:
        output_video_path, output_json_path = output_paths(output_root, "VACE_1p3B_ctx08", case["sample_id"])
        if is_complete(output_video_path, output_json_path, overwrite):
            print(f"[skip] VACE_1p3B_ctx08 {case['sample_id']}", flush=True)
            continue
        print(f"[run] VACE_1p3B_ctx08 {case['sample_id']} seed={SEED}", flush=True)
        ensure_dir(output_video_path.parent)
        tmp_dir = ensure_dir(output_video_path.parent / "_tmp")
        records = runner.generate(
            input_spec=InputSpec(prompt=case["caption"], context_video_path=case["context_video_path"]),
            config=config,
            output_dir=tmp_dir,
        )
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one ctx08 record for {case['sample_id']}, got {len(records)}")
        shutil.copy2(records[0].video_path, output_video_path)
        records[0].video_path.unlink()
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
        write_json(
            output_json_path,
            build_payload(
                case=case,
                method="VACE_1p3B_ctx08",
                output_video_path=output_video_path,
                output_json_path=output_json_path,
                conditioning_mode="V2V_ctx08",
                context_frames=CONTEXT_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_manifest(output_root: Path, cases: list[dict[str, Any]], methods: list[str]) -> None:
    payload = {
        "benchmark": BENCHMARK_NAME,
        "dataset_root": str(DATASET_ROOT),
        "output_root": str(output_root),
        "num_cases": len(cases),
        "methods": methods,
        "sample_ids": [case["sample_id"] for case in cases],
    }
    write_json(output_root / "manifest.json", payload)


def main() -> None:
    args = parse_args()
    cases = [load_case(path) for path in iter_meta_paths(args.limit)]
    write_manifest(args.output_root, cases, args.methods)
    if "GT" in args.methods:
        run_gt(args.output_root, cases, args.overwrite)
    if "wan22-5B-TI2V" in args.methods:
        run_wan_ti2v(args.output_root, cases, args.overwrite)
    if "VACE_1p3B_TI2V" in args.methods:
        run_vace_ti2v(args.output_root, cases, args.overwrite)
    if "VACE_1p3B_ctx08" in args.methods:
        run_vace_ctx08(args.output_root, cases, args.overwrite)
    print(f"Generated methods: {', '.join(args.methods)}", flush=True)


if __name__ == "__main__":
    main()
