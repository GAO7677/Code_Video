#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from rerank_video.generators import VaceGenerator
from rerank_video.pdi_proxy_eval import VaceTI2VRunner, WanTI2VRunner
from rerank_video.schemas import GeneratorConfig, InputSpec
from rerank_video.video_utils import ensure_dir, pil_list_to_numpy, save_video_frames, write_json


PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
PHYGENBENCH_ROOT = Path("/home/gaoya/Code_Video/PhyGenBench-main")
PROMPTS_PATH = PHYGENBENCH_ROOT / "PhyGenBench" / "prompts.json"
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/phygenbench")
BENCHMARK_NAME = "phygenbench"

WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")
FLUX_CACHE_ROOT = Path("/data/gaoya/ckpt")
FLUX_KONTEXT_ROOT = Path(
    os.environ.get(
        "FLUX_KONTEXT_ROOT",
        "/data/luoyang/ckpt/pretrained/models--black-forest-labs--FLUX.1-Kontext-dev",
    )
)

METHODS = ["FLUX_1_Kontext", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]

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


if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PhyGenBench first frames and TI2V/VACE videos.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--flux-embedded-guidance", type=float, default=2.5)
    parser.add_argument("--flux-steps", type=int, default=28)
    return parser.parse_args()


def load_prompts() -> list[dict[str, Any]]:
    payload = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        rows.append(
            {
                "sample_id": f"{idx:03d}",
                "prompt_index": idx,
                "clip_name": f"output_video_{idx}",
                "caption": str(item["caption"]),
                "physical_laws": str(item.get("physical_laws") or ""),
                "sub_category": str(item.get("sub_category") or ""),
                "main_category": str(item.get("main_category") or ""),
            }
        )
    return rows


def slice_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    data = rows
    if args.limit is not None:
        data = data[: args.limit]
    start_index = max(int(args.start_index), 0)
    end_index = len(data) if args.end_index is None else min(int(args.end_index), len(data))
    data = data[start_index:end_index]
    if args.num_shards is not None or args.shard_id is not None:
        if args.num_shards is None or args.shard_id is None:
            raise ValueError("--num-shards and --shard-id must be set together")
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be positive")
        if not 0 <= args.shard_id < args.num_shards:
            raise ValueError("--shard-id must be in [0, num_shards)")
        data = [row for i, row in enumerate(data) if i % args.num_shards == args.shard_id]
    return data


def method_dir(output_root: Path, method: str) -> Path:
    return output_root / "output" / method / BENCHMARK_NAME


def first_frame_path(output_root: Path, sample_id: str) -> Path:
    return method_dir(output_root, "FLUX_1_Kontext") / f"{sample_id}.png"


def first_frame_json_path(output_root: Path, sample_id: str) -> Path:
    return method_dir(output_root, "FLUX_1_Kontext") / f"{sample_id}.json"


def context_video_path(output_root: Path, sample_id: str) -> Path:
    return method_dir(output_root, "FLUX_1_Kontext") / f"{sample_id}.ctx08.mp4"


def output_paths(output_root: Path, method: str, sample_id: str) -> tuple[Path, Path]:
    base = method_dir(output_root, method) / sample_id
    return base.with_suffix(".mp4"), base.with_suffix(".json")


def is_complete(video_path: Path, json_path: Path, overwrite: bool) -> bool:
    return not overwrite and video_path.is_file() and json_path.is_file()


def first_frame_complete(output_root: Path, sample_id: str, overwrite: bool) -> bool:
    return (
        not overwrite
        and first_frame_path(output_root, sample_id).is_file()
        and first_frame_json_path(output_root, sample_id).is_file()
        and context_video_path(output_root, sample_id).is_file()
    )


def build_first_frame_payload(
    *,
    row: dict[str, Any],
    image_path: Path,
    context_path: Path,
    seed: int,
    flux_steps: int,
    flux_embedded_guidance: float,
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset": BENCHMARK_NAME,
        "method": "FLUX_1_Kontext",
        "sample_id": row["sample_id"],
        "prompt_index": row["prompt_index"],
        "clip_name": row["clip_name"],
        "prompt": row["caption"],
        "caption": row["caption"],
        "physical_laws": row["physical_laws"],
        "sub_category": row["sub_category"],
        "main_category": row["main_category"],
        "image_path": str(image_path),
        "first_frame": str(image_path),
        "context_video": str(context_path),
        "seed": seed,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "context_frames": CONTEXT_FRAMES,
        "flux_steps": flux_steps,
        "flux_embedded_guidance": flux_embedded_guidance,
    }


def build_video_payload(
    *,
    row: dict[str, Any],
    method: str,
    output_video_path: Path,
    output_json_path: Path,
    first_frame: Path,
    context_video: Path,
    seed: int,
    conditioning_mode: str,
    context_frames: int,
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "dataset": BENCHMARK_NAME,
        "method": method,
        "sample_id": row["sample_id"],
        "prompt_index": row["prompt_index"],
        "clip_name": row["clip_name"],
        "prompt": row["caption"],
        "caption": row["caption"],
        "physical_laws": row["physical_laws"],
        "sub_category": row["sub_category"],
        "main_category": row["main_category"],
        "video": str(output_video_path),
        "video_path": str(output_video_path),
        "first_frame": str(first_frame),
        "context_video": str(context_video),
        "seed": seed,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "cfg_scale": CFG_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "conditioning_mode": conditioning_mode,
        "context_frames": context_frames,
        "negative_prompt": NEGATIVE_PROMPT,
        "paths": {
            "output_video_path": str(output_video_path),
            "output_json_path": str(output_json_path),
            "first_frame_path": str(first_frame),
            "context_video_path": str(context_video),
        },
    }


def build_repeated_context_video(image_path: Path, out_path: Path, *, fps: int, frames: int) -> None:
    image = Image.open(image_path).convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    np_frames = pil_list_to_numpy([image.copy() for _ in range(max(int(frames), 1))])
    save_video_frames(out_path, np_frames, fps=fps, quality=QUALITY)


class FluxKontextFirstFrameRunner:
    def __init__(self, *, device: str, model_root: Path) -> None:
        from diffusers import FluxKontextPipeline

        self.device = device
        self.pipe = FluxKontextPipeline.from_pretrained(
            str(model_root),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        if str(device).startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        seed: int,
        embedded_guidance: float,
        num_inference_steps: int,
    ) -> Path:
        generator_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        result = self.pipe(
            prompt=prompt,
            guidance_scale=embedded_guidance,
            generator=torch.Generator(device=generator_device).manual_seed(seed),
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=num_inference_steps,
            output_type="pil",
        )
        ensure_dir(output_path.parent)
        image = result.images[0] if hasattr(result, "images") else result[0][0]
        image.save(output_path)
        return output_path


def ensure_first_frames(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    runner: FluxKontextFirstFrameRunner | None = None
    try:
        for row in rows:
            sample_id = row["sample_id"]
            image_path = first_frame_path(output_root, sample_id)
            image_json_path = first_frame_json_path(output_root, sample_id)
            ctx_path = context_video_path(output_root, sample_id)
            seed = SEED + args.seed_offset + row["prompt_index"] - 1
            if first_frame_complete(output_root, sample_id, args.overwrite):
                print(f"[skip] FLUX_1_Kontext {sample_id}", flush=True)
                continue
            if runner is None:
                if not FLUX_KONTEXT_ROOT.is_dir():
                    raise FileNotFoundError(f"Missing FLUX_KONTEXT_ROOT: {FLUX_KONTEXT_ROOT}")
                runner = FluxKontextFirstFrameRunner(device=args.device, model_root=FLUX_KONTEXT_ROOT)
            print(f"[run] FLUX_1_Kontext {sample_id} seed={seed}", flush=True)
            runner.generate(
                prompt=row["caption"],
                output_path=image_path,
                seed=seed,
                embedded_guidance=args.flux_embedded_guidance,
                num_inference_steps=args.flux_steps,
            )
            build_repeated_context_video(image_path, ctx_path, fps=FPS, frames=CONTEXT_FRAMES)
            write_json(
                image_json_path,
                build_first_frame_payload(
                    row=row,
                    image_path=image_path,
                    context_path=ctx_path,
                    seed=seed,
                    flux_steps=args.flux_steps,
                    flux_embedded_guidance=args.flux_embedded_guidance,
                ),
            )
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_wan_ti2v(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    runner = WanTI2VRunner(model_root=WAN_ROOT, device=args.device)
    try:
        for row in rows:
            seed = SEED + args.seed_offset + row["prompt_index"] - 1
            image_path = first_frame_path(output_root, row["sample_id"])
            ctx_path = context_video_path(output_root, row["sample_id"])
            output_video_path, output_json_path = output_paths(output_root, "wan22-5B-TI2V", row["sample_id"])
            if is_complete(output_video_path, output_json_path, args.overwrite):
                print(f"[skip] wan22-5B-TI2V {row['sample_id']}", flush=True)
                continue
            print(f"[run] wan22-5B-TI2V {row['sample_id']} seed={seed}", flush=True)
            ensure_dir(output_video_path.parent)
            runner.generate(
                first_frame_path=image_path,
                prompt=row["caption"],
                output_path=output_video_path,
                seed=seed,
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
                build_video_payload(
                    row=row,
                    method="wan22-5B-TI2V",
                    output_video_path=output_video_path,
                    output_json_path=output_json_path,
                    first_frame=image_path,
                    context_video=ctx_path,
                    seed=seed,
                    conditioning_mode="TI2V_first_frame",
                    context_frames=1,
                ),
            )
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_vace_ti2v(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    runner = VaceTI2VRunner(model_root=VACE_ROOT, device=args.device)
    try:
        for row in rows:
            seed = SEED + args.seed_offset + row["prompt_index"] - 1
            image_path = first_frame_path(output_root, row["sample_id"])
            ctx_path = context_video_path(output_root, row["sample_id"])
            output_video_path, output_json_path = output_paths(output_root, "VACE_1p3B_TI2V", row["sample_id"])
            if is_complete(output_video_path, output_json_path, args.overwrite):
                print(f"[skip] VACE_1p3B_TI2V {row['sample_id']}", flush=True)
                continue
            print(f"[run] VACE_1p3B_TI2V {row['sample_id']} seed={seed}", flush=True)
            ensure_dir(output_video_path.parent)
            runner.generate(
                first_frame_path=image_path,
                prompt=row["caption"],
                output_path=output_video_path,
                seed=seed,
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
                build_video_payload(
                    row=row,
                    method="VACE_1p3B_TI2V",
                    output_video_path=output_video_path,
                    output_json_path=output_json_path,
                    first_frame=image_path,
                    context_video=ctx_path,
                    seed=seed,
                    conditioning_mode="TI2V_first_frame",
                    context_frames=1,
                ),
            )
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_vace_ctx08(output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    config = GeneratorConfig(
        key="vace_ctx08",
        type="vace",
        enabled=True,
        device=args.device,
        model_root=VACE_ROOT,
        num_candidates=1,
        base_seed=SEED + args.seed_offset,
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
    try:
        for row in rows:
            image_path = first_frame_path(output_root, row["sample_id"])
            ctx_path = context_video_path(output_root, row["sample_id"])
            output_video_path, output_json_path = output_paths(output_root, "VACE_1p3B_ctx08", row["sample_id"])
            seed = SEED + args.seed_offset + row["prompt_index"] - 1
            if is_complete(output_video_path, output_json_path, args.overwrite):
                print(f"[skip] VACE_1p3B_ctx08 {row['sample_id']}", flush=True)
                continue
            print(f"[run] VACE_1p3B_ctx08 {row['sample_id']} seed={seed}", flush=True)
            ensure_dir(output_video_path.parent)
            tmp_dir = ensure_dir(output_video_path.parent / "_tmp")
            records = runner.generate(
                input_spec=InputSpec(prompt=row["caption"], context_video_path=ctx_path),
                config=config,
                output_dir=tmp_dir,
            )
            if len(records) != 1:
                raise RuntimeError(f"Expected exactly one ctx08 record for {row['sample_id']}, got {len(records)}")
            shutil.copy2(records[0].video_path, output_video_path)
            records[0].video_path.unlink()
            if not any(tmp_dir.iterdir()):
                tmp_dir.rmdir()
            write_json(
                output_json_path,
                build_video_payload(
                    row=row,
                    method="VACE_1p3B_ctx08",
                    output_video_path=output_video_path,
                    output_json_path=output_json_path,
                    first_frame=image_path,
                    context_video=ctx_path,
                    seed=seed,
                    conditioning_mode="V2V_ctx08_repeat_frame",
                    context_frames=CONTEXT_FRAMES,
                ),
            )
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_manifest(output_root: Path, rows: list[dict[str, Any]], methods: list[str], args: argparse.Namespace) -> None:
    payload = {
        "benchmark": BENCHMARK_NAME,
        "prompts_path": str(PROMPTS_PATH),
        "output_root": str(output_root),
        "methods": methods,
        "num_rows": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
        "slice": {
            "limit": args.limit,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
        },
        "generation": {
            "seed": SEED,
            "seed_offset": args.seed_offset,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "context_frames": CONTEXT_FRAMES,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "cfg_scale": CFG_SCALE,
            "width": WIDTH,
            "height": HEIGHT,
            "negative_prompt": NEGATIVE_PROMPT,
            "device": args.device,
            "flux_steps": args.flux_steps,
            "flux_embedded_guidance": args.flux_embedded_guidance,
        },
    }
    write_json(output_root / "manifest.json", payload)


def main() -> None:
    args = parse_args()
    rows = slice_rows(load_prompts(), args)
    if not rows:
        raise RuntimeError("No prompt rows selected.")
    write_manifest(args.output_root, rows, args.methods, args)
    if any(method in args.methods for method in ("FLUX_1_Kontext", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08")):
        ensure_first_frames(args.output_root, rows, args)
    if "wan22-5B-TI2V" in args.methods:
        run_wan_ti2v(args.output_root, rows, args)
    if "VACE_1p3B_TI2V" in args.methods:
        run_vace_ti2v(args.output_root, rows, args)
    if "VACE_1p3B_ctx08" in args.methods:
        run_vace_ctx08(args.output_root, rows, args)
    print(f"Generated methods: {', '.join(args.methods)}", flush=True)


if __name__ == "__main__":
    main()
