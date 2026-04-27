#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

import sys

DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

sys.path.append("/home/gaoya/code_my_utils")
from tools.seed import seed_everything  # noqa: E402

from diffsynth import ModelConfig  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402
from context_wan import ContextAwareWanVideoPipeline  # noqa: E402

from physinone_benchmark_common import (  # noqa: E402
    find_tokenizer_path,
    load_jsonl,
    run_all_benchmarks,
    write_json,
    write_jsonl,
)


DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/physInOne_AB")
DEFAULT_BENCH_CONFIG = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/physInOne/bench_paths.local.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan2.2 TV2V baseline on PhysInOne benchmark manifests.")
    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--wan_root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--bench_config", type=Path, default=DEFAULT_BENCH_CONFIG)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model_name", default="wan22_tv2v_pretrain_ctx8_physinone")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_benchmarks", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def build_pipeline(wan_root: Path, device: str) -> ContextAwareWanVideoPipeline:
    return ContextAwareWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                path=[
                    str(wan_root / "diffusion_pytorch_model-00001-of-00003.safetensors"),
                    str(wan_root / "diffusion_pytorch_model-00002-of-00003.safetensors"),
                    str(wan_root / "diffusion_pytorch_model-00003-of-00003.safetensors"),
                ]
            ),
            ModelConfig(path=str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(wan_root / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(find_tokenizer_path(wan_root))),
    )


def load_context_images(context_dir: Path, height: int, width: int, max_frames: int) -> List[Image.Image]:
    paths = sorted(
        p for p in context_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    frames = [Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS) for path in paths[:max_frames]]
    if len(frames) < max_frames:
        raise ValueError(f"Expected at least {max_frames} context frames in {context_dir}, found {len(frames)}")
    return frames


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.source_manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    generated_dir = args.output_root / "generated_videos" / args.model_name
    metadata_dir = args.output_root / "metadata" / args.model_name
    benchmark_manifest_path = metadata_dir / f"{args.model_name}_benchmark_manifest.jsonl"
    summary_path = args.output_root / "benchmarks" / args.model_name / "summary.json"

    generated_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows: List[Dict[str, Any]] = []
    pipe = None if args.skip_generation else build_pipeline(args.wan_root, args.device)

    for row in rows:
        sample_id = str(row["sample_id"])
        out_path = generated_dir / f"{sample_id}.mp4"
        meta_path = metadata_dir / f"{sample_id}.json"

        if not args.skip_generation and (args.overwrite or not out_path.exists()):
            seed_everything(args.seed)
            context = load_context_images(Path(row["context_frames_dir"]), args.height, args.width, args.context_frames)
            with torch.no_grad():
                video = pipe(
                    prompt=row["prompt"],
                    negative_prompt=args.negative_prompt,
                    input_image=context[0],
                    context_video=context,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    seed=args.seed,
                    cfg_scale=args.cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    tiled=True,
                )
            save_video(video[: args.num_frames], str(out_path), fps=args.fps, quality=args.quality)

        case_meta = {
            "sample_id": sample_id,
            "prompt": row["prompt"],
            "context_frames_dir": row["context_frames_dir"],
            "image_path": row["image_path"],
            "output_path": str(out_path),
            "gt_video_path": row["gt_video_path"],
            "group_id": row["group_id"],
            "group_name": row["group_name"],
            "split": row["split"],
            "physics_types": row["physics_types"],
            "task": "tv2v",
            "num_frames": args.num_frames,
            "context_frames": args.context_frames,
            "seed": args.seed,
        }
        write_json(meta_path, case_meta)

        benchmark_rows.append(
            {
                "sample_id": sample_id,
                "prompt": row["prompt"],
                "video_path": str(out_path.resolve()),
                "context_frames_dir": row["context_frames_dir"],
                "image_path": row["image_path"],
                "gt_video_path": row["gt_video_path"],
                "generated_start_frame": 0,
                "gt_start_frame": 0,
                "group_id": row["group_id"],
                "group_name": row["group_name"],
                "split": row["split"],
                "physics_types": row["physics_types"],
                "task": "tv2v",
            }
        )

    write_jsonl(benchmark_manifest_path, benchmark_rows)

    if not args.skip_benchmarks:
        summary = run_all_benchmarks(
            bench_config_path=args.bench_config,
            benchmark_manifest_path=benchmark_manifest_path,
            output_root=args.output_root / "benchmarks" / args.model_name,
            run_prefix=args.model_name,
        )
        print(summary)
        print(summary_path)
    else:
        print(benchmark_manifest_path)


if __name__ == "__main__":
    main()
