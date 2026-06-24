from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix


def _resolve_device() -> str:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return "cuda:0"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether VAE time compression aligns between full video and context-only video.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--full-video", required=True, help="full training-style video clip")
    parser.add_argument("--context-video", required=True, help="context-only input clip")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = _resolve_device()
    trainer = ContextVideoTrainer(config, build_optimizer=False, device=device)

    total_frames = int(config["data"].get("num_frames", 24))
    context_frames = int(config["data"]["num_context_frames"])
    resolution = tuple(config["data"]["resolution"])

    full_rgb, _ = read_video_prefix(Path(args.full_video), total_frames)
    ctx_rgb, _ = read_video_prefix(Path(args.context_video), context_frames)

    full_video = preprocess_video_rgb_uint8(full_rgb, resolution).unsqueeze(0).to(trainer.device_obj)
    context_video = preprocess_video_rgb_uint8(ctx_rgb, resolution).unsqueeze(0).to(trainer.device_obj)

    with torch.no_grad():
        full_latents = trainer.bundle.vae.encode([full_video[0]])[0].detach().float().cpu()
        context_latents = trainer.bundle.vae.encode([context_video[0]])[0].detach().float().cpu()

    ctx_lat_t = int(context_latents.shape[1])
    ctx_part_from_full = full_latents[:, :ctx_lat_t].contiguous()
    diff = (ctx_part_from_full - context_latents).abs()

    report = {
        "full_video": str(Path(args.full_video).resolve()),
        "context_video": str(Path(args.context_video).resolve()),
        "full_latents_shape": list(full_latents.shape),
        "context_latents_shape": list(context_latents.shape),
        "ctx_lat_t": ctx_lat_t,
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "full_prefix_mean": float(ctx_part_from_full.mean().item()),
        "context_mean": float(context_latents.mean().item()),
        "full_prefix_std": float(ctx_part_from_full.std(unbiased=False).item()),
        "context_std": float(context_latents.std(unbiased=False).item()),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
