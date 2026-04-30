#!/usr/bin/env python3
"""Run batched VACE inference for benchmark meta lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

import batch_eval_lora as bel
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


DEFAULT_VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched VACE inference for a meta-list benchmark.")
    parser.add_argument("--vace_root", type=Path, default=DEFAULT_VACE_ROOT)
    parser.add_argument("--meta_list_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--runtime_root", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--mode", choices=["ti2v_firstframe", "v2v_clipref"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def build_pipeline(vace_root: Path, device: str) -> WanVideoPipeline:
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=str(vace_root / "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(vace_root / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(vace_root / "Wan2.1_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(
            path=str(vace_root / "google" / "umt5-xxl"),
            skip_download=True,
        ),
        redirect_common_files=False,
    )


def load_first_frame(case: dict[str, Any], *, height: int, width: int) -> Image.Image:
    source_paths = case.get("source_paths", {})
    raw_first = source_paths.get("first_frame_path")
    if isinstance(raw_first, str) and raw_first:
        return Image.open(raw_first).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return bel.load_context_frames(
        context_path=Path(case["context_path"]),
        context_frames=1,
        height=height,
        width=width,
        resize_mode=case.get("context_resize_mode", "crop"),
    )[0]


def build_vace_inputs(
    *,
    case: dict[str, Any],
    mode: str,
    context_frames: int,
    height: int,
    width: int,
    aligned_num_frames: int,
) -> tuple[list[Image.Image], list[Image.Image], int]:
    placeholder = Image.new("RGB", (width, height), (128, 128, 128))
    mask_black = Image.new("RGB", (width, height), (0, 0, 0))
    mask_white = Image.new("RGB", (width, height), (255, 255, 255))

    if mode == "ti2v_firstframe":
        first_frame = load_first_frame(case, height=height, width=width)
        known_frames = [first_frame]
        used_context_frames = 1
    else:
        known_frames = bel.load_context_frames(
            context_path=Path(case["context_path"]),
            context_frames=context_frames,
            height=height,
            width=width,
            resize_mode=case.get("context_resize_mode", "crop"),
        )
        used_context_frames = len(known_frames)

    video_input = known_frames + [placeholder.copy() for _ in range(aligned_num_frames - used_context_frames)]
    video_mask = [mask_black.copy() for _ in range(used_context_frames)] + [
        mask_white.copy() for _ in range(aligned_num_frames - used_context_frames)
    ]
    return video_input, video_mask, used_context_frames


def build_case_payload(
    *,
    args: argparse.Namespace,
    case: dict[str, Any],
    index: int,
    output_path: Path,
    used_context_frames: int,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    sidecar_path = output_path.with_suffix(".json")
    source_paths = case.get("source_paths", {})
    paths_payload = bel.build_paths_payload(
        source_paths=source_paths,
        context_path=case["context_path"],
        output_path=output_path,
        sidecar_path=sidecar_path,
        conditioning_mode=args.mode,
    )
    if args.mode == "ti2v_firstframe":
        first_frame_path = source_paths.get("first_frame_path")
        if isinstance(first_frame_path, str) and first_frame_path:
            paths_payload["input_roles"] = [
                {"role": "vace_video_known_frame", "path": first_frame_path},
            ]
        else:
            paths_payload["input_roles"] = [
                {
                    "role": "vace_video_known_frame",
                    "path": case["context_path"],
                    "note": "first frame extracted from context video at runtime",
                },
            ]
    elif args.mode == "v2v_clipref":
        paths_payload["input_roles"] = [
            {"role": "vace_video_known_frames", "path": case["context_path"]},
        ]

    source_conditions = []
    if args.mode == "ti2v_firstframe":
        first_frame_path = source_paths.get("first_frame_path")
        if isinstance(first_frame_path, str) and first_frame_path:
            source_conditions.append({"role": "vace_video_known_frame", "path": first_frame_path})
        else:
            source_conditions.append(
                {
                    "role": "vace_video_known_frame",
                    "path": case["context_path"],
                    "note": "first frame extracted from context video at runtime",
                }
            )
    else:
        source_conditions.append({"role": "vace_video_known_frames", "path": case["context_path"]})

    payload = {
        "model_name": args.model_name,
        "benchmark_step": None,
        "dataset": case["dataset"],
        "sample_id": case["sample_id"],
        "scenario": case.get("scenario"),
        "seed": args.seed,
        "caption": case["caption"],
        "weights_path": str(args.vace_root),
        "status": status,
        "generation_params": {
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "num_frames": args.requested_output_frames,
            "requested_output_frames": args.requested_output_frames,
            "aligned_generation_num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "negative_prompt": args.negative_prompt,
            "context_frames": args.context_frames,
            "used_context_frames": used_context_frames,
            "conditioning_mode": args.mode,
            "task": "vace_meta_benchmark",
        },
        "runtime": {
            "index_in_sorted_list": index,
            "shard_id": 0,
            "num_shards": 1,
        },
        "paths": paths_payload,
        "model_inputs": {
            "conditioning_mode": args.mode,
            "pipeline_kwargs": ["vace_video", "vace_video_mask"],
            "source_conditions": source_conditions,
            "synthetic_conditions": [
                {
                    "role": "vace_video_placeholder_frames",
                    "count": max(args.num_frames - used_context_frames, 0),
                    "note": "gray placeholder frames appended after known source frames",
                },
                {
                    "role": "vace_video_mask",
                    "known_frames": used_context_frames,
                    "future_frames": max(args.num_frames - used_context_frames, 0),
                    "note": "black mask for known frames, white mask for future frames",
                },
            ],
        },
    }
    if error is not None:
        payload["error"] = error
    return payload


def write_run_manifest(args: argparse.Namespace, metadata_dir: Path) -> None:
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": {
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
    }
    bel.write_json(metadata_dir / f"{args.model_name}_run_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.runtime_root = args.runtime_root.expanduser().resolve()
    args.meta_list_path = args.meta_list_path.expanduser().resolve()
    args.vace_root = args.vace_root.expanduser().resolve()
    assert_exists(args.vace_root, "VACE root")
    assert_exists(args.meta_list_path, "Meta list path")

    args.requested_output_frames = int(args.num_frames)
    args.num_frames = bel.align_generation_num_frames(args.num_frames)

    cases = bel.collect_cases(bel.load_meta_paths(args.meta_list_path), limit=None)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.runtime_root / "metadata" / args.model_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(args, metadata_dir)

    pipe = build_pipeline(args.vace_root, args.device)
    entries: list[dict[str, Any]] = []

    for index, case in enumerate(cases):
        output_path = args.output_root / case["output_name"]
        sidecar_path = output_path.with_suffix(".json")
        if output_path.exists() and not args.overwrite:
            payload = build_case_payload(
                args=args,
                case=case,
                index=index,
                output_path=output_path,
                used_context_frames=1 if args.mode == "ti2v_firstframe" else args.context_frames,
                status="skipped_existing",
            )
            bel.write_json(sidecar_path, payload)
            entries.append(payload)
            continue

        try:
            video_input, video_mask, used_context_frames = build_vace_inputs(
                case=case,
                mode=args.mode,
                context_frames=args.context_frames,
                height=args.height,
                width=args.width,
                aligned_num_frames=args.num_frames,
            )
            with torch.no_grad():
                video = pipe(
                    prompt=case["caption"],
                    negative_prompt=args.negative_prompt,
                    vace_video=video_input,
                    vace_video_mask=video_mask,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    seed=args.seed,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    tiled=True,
                )
            video = video[: args.requested_output_frames]
            bel.save_video(video, str(output_path), fps=args.fps, quality=args.quality)
            payload = build_case_payload(
                args=args,
                case=case,
                index=index,
                output_path=output_path,
                used_context_frames=used_context_frames,
                status="generated",
            )
        except Exception as exc:
            payload = build_case_payload(
                args=args,
                case=case,
                index=index,
                output_path=output_path,
                used_context_frames=0,
                status="failed",
                error=repr(exc),
            )
        bel.write_json(sidecar_path, payload)
        entries.append(payload)

    bel.write_jsonl(metadata_dir / f"{args.model_name}_per_case.jsonl", entries)
    summary = bel.build_summary(entries)
    bel.write_json(
        args.runtime_root / "summary.json",
        {
            "model_name": args.model_name,
            "meta_list_path": str(args.meta_list_path),
            "summary": summary,
        },
    )
    print(args.runtime_root / "summary.json")


if __name__ == "__main__":
    main()
