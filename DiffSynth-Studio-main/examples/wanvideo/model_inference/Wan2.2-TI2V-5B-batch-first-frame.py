#!/usr/bin/env python3
"""Batch the official Wan2.2-TI2V-5B image-to-video example over case JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

CHECKPOINT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_official_first_frame_seed_sweep"
)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(40, 10))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_pipeline(device: str) -> WanVideoPipeline:
    diffusion_shards = sorted(CHECKPOINT.glob("diffusion_pytorch_model*.safetensors"))
    if not diffusion_shards:
        raise FileNotFoundError(f"No diffusion shards found under {CHECKPOINT}")
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=str(CHECKPOINT / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=[str(path) for path in diffusion_shards]),
            ModelConfig(path=str(CHECKPOINT / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(CHECKPOINT / "google/umt5-xxl")),
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.case_json.expanduser().resolve().read_text(encoding="utf-8"))
    prompt = str(payload.get("input_caption") or payload.get("caption") or "").strip()
    if not prompt:
        raise KeyError(f"Missing input_caption/caption in {args.case_json}")
    image_path = Path(payload["input_image"]).expanduser().resolve()
    input_image = Image.open(image_path).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )
    case_key = args.case_json.stem
    pipe = build_pipeline(args.device)

    for seed in args.seeds:
        for steps in args.steps:
            output_dir = args.output_root / "seeds" / f"seed_{seed:06d}" / f"steps{steps}"
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / "original.mp4"
            if video_path.is_file() and video_path.stat().st_size:
                print(f"skip case={case_key} seed={seed} steps={steps}", flush=True)
                continue
            print(f"start case={case_key} seed={seed} steps={steps}", flush=True)
            with torch.inference_mode():
                video = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    seed=seed,
                    tiled=True,
                    height=args.height,
                    width=args.width,
                    input_image=input_image,
                    num_frames=args.num_frames,
                    num_inference_steps=steps,
                )
            save_video(video, str(video_path), fps=args.fps, quality=args.quality)
            metadata = {
                "case": case_key,
                "case_json": str(args.case_json.expanduser().resolve()),
                "pipeline": "DiffSynth WanVideoPipeline / Wan2.2-TI2V-5B",
                "source_example": str(Path(__file__).with_name("Wan2.2-TI2V-5B.py")),
                "conditioning": "prompt + input_image(first frame)",
                "prompt": prompt,
                "input_image": str(image_path),
                "context_video": None,
                "seed": seed,
                "num_inference_steps": steps,
                "num_frames": args.num_frames,
                "height": args.height,
                "width": args.width,
                "fps": args.fps,
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"complete {video_path}", flush=True)


if __name__ == "__main__":
    main()
