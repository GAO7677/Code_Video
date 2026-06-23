#!/usr/bin/env python3
"""Run Wan2.2 TI2V-5B text-to-video inference with an optional LoRA."""

import argparse
import json
from pathlib import Path

import torch

from diffsynth import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video


DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-to-video inference for Wan2.2 TI2V-5B with LoRA.")
    parser.add_argument("--wan_root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument("--output_video_path", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def find_tokenizer_path(wan_root: Path) -> Path:
    candidates = [
        wan_root / "google" / "umt5-xxl",
        wan_root / "google",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Tokenizer directory not found. Checked: {', '.join(str(path) for path in candidates)}"
    )


def build_model_configs(wan_root: Path) -> list[ModelConfig]:
    dit_shards = [
        wan_root / "diffusion_pytorch_model-00001-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00002-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00003-of-00003.safetensors",
    ]
    t5_path = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = wan_root / "Wan2.2_VAE.pth"
    for path in dit_shards + [t5_path, vae_path]:
        assert_exists(path, "Required model file")
    return [
        ModelConfig(path=[str(path) for path in dit_shards]),
        ModelConfig(path=str(t5_path)),
        ModelConfig(path=str(vae_path)),
    ]


def write_sidecar_json(args: argparse.Namespace) -> None:
    sidecar_path = args.output_video_path.with_suffix(".json")
    payload = {
        "video_path": str(args.output_video_path),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "wan_root": str(args.wan_root),
        "lora_path": str(args.lora_path),
        "device": args.device,
        "mode": "text2video",
    }
    sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.wan_root = args.wan_root.expanduser().resolve()
    args.lora_path = args.lora_path.expanduser().resolve()
    args.output_video_path = args.output_video_path.expanduser().resolve()

    assert_exists(args.wan_root, "Wan root")
    assert_exists(args.lora_path, "LoRA checkpoint")
    if args.output_video_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_video_path}")

    print(f"[t2v] device={args.device}")
    print(f"[t2v] wan_root={args.wan_root}")
    print(f"[t2v] lora_path={args.lora_path}")
    print(f"[t2v] output_video_path={args.output_video_path}")
    print(f"[t2v] seed={args.seed}")
    print(f"[t2v] size={args.width}x{args.height} frames={args.num_frames} fps={args.fps}")
    print(f"[t2v] prompt={args.prompt}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=build_model_configs(args.wan_root),
        tokenizer_config=ModelConfig(path=str(find_tokenizer_path(args.wan_root))),
    )
    pipe.load_lora(pipe.dit, str(args.lora_path), alpha=1.0)

    with torch.no_grad():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
        )

    args.output_video_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(args.output_video_path), fps=args.fps, quality=args.quality)
    write_sidecar_json(args)
    print(f"[t2v] saved_video={args.output_video_path}")
    print(f"[t2v] saved_json={args.output_video_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
