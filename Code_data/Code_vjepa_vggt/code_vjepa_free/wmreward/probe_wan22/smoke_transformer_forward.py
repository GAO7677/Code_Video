#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from extract_probe_features import (
    BlockFeatureRecorder,
    build_sample_runtime,
    compute_token_grid,
    load_manifest,
    load_pipe,
    ProbeConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Single-step transformer-only smoke test for Wan2.2 probing.")
    parser.add_argument(
        "--model_root",
        default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers",
    )
    parser.add_argument(
        "--manifest_csv",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/subsets/subset16_smoke.csv",
    )
    parser.add_argument(
        "--output_root",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/smoke_forward_outputs",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--prompt_max_length", type=int, default=256)
    parser.add_argument("--seed_mode", default="source", choices=["source", "fixed"])
    parser.add_argument("--fixed_seed", type=int, default=42)
    parser.add_argument("--capture_layers", default="2,8,14,20,29")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ProbeConfig(
        model_root=args.model_root,
        manifest_csv=args.manifest_csv,
        output_root=args.output_root,
        limit=args.limit,
        overwrite=True,
        device=args.device,
        dtype=args.dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        prompt_max_length=args.prompt_max_length,
        seed_mode=args.seed_mode,
        fixed_seed=args.fixed_seed,
        negative_prompt=None,
        capture_step_indices=[0],
        capture_layers=[int(x) for x in args.capture_layers.split(",") if x.strip()],
        capture_branches="cond",
        save_final_latents=False,
    )

    out_root = Path(config.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(config.manifest_csv, config.limit)
    runtime = build_sample_runtime(manifest_rows[0], config)
    pipe = load_pipe(config)

    recorder = BlockFeatureRecorder(
        capture_layers=config.capture_layers,
        capture_step_indices=[0],
        capture_branches="cond",
    )
    recorder.register(pipe.transformer)

    try:
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=runtime["prompt"],
            negative_prompt=runtime["negative_prompt"],
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=config.prompt_max_length,
            device=pipe._execution_device,
        )
        prompt_embeds = prompt_embeds.to(pipe.transformer.dtype)

        pipe.scheduler.set_timesteps(config.num_inference_steps, device=pipe._execution_device)
        t = pipe.scheduler.timesteps[0]

        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.transformer.config.in_channels,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            dtype=torch.float32,
            device=pipe._execution_device,
            generator=torch.Generator(device=config.device).manual_seed(runtime["seed"]),
            latents=None,
        )
        mask = torch.ones(latents.shape, dtype=torch.float32, device=pipe._execution_device)

        if pipe.config.expand_timesteps:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            timestep = t.expand(latents.shape[0])

        latent_model_input = latents.to(pipe.transformer.dtype)
        token_grid = compute_token_grid(tuple(latent_model_input.shape), tuple(pipe.transformer.config.patch_size))
        timestep_value = int(t.item()) if torch.is_tensor(t) else int(t)
        recorder.activate(0, timestep_value, "cond", token_grid, tuple(latent_model_input.shape))
        with pipe.transformer.cache_context("cond"):
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        recorder.deactivate()

        payload = {
            "meta": {
                "sample_id": runtime["sample_id"],
                "pair_id": runtime["manifest_row"]["pair_id"],
                "role": runtime["manifest_row"]["role"],
                "basename": runtime["manifest_row"]["basename"],
                "prompt": runtime["prompt"],
                "seed": runtime["seed"],
                "source_surprise_score": float(runtime["manifest_row"]["surprise_score"]),
                "height": config.height,
                "width": config.width,
                "num_frames": config.num_frames,
                "num_inference_steps": config.num_inference_steps,
                "timestep": timestep_value,
                "capture_layers": config.capture_layers,
                "latent_shape": list(latent_model_input.shape),
                "noise_pred_shape": list(noise_pred.shape),
            },
            "features": recorder.data,
        }

        sample_dir = out_root / runtime["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, sample_dir / "probe_forward_smoke.pt")
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(payload["meta"], f, indent=2, ensure_ascii=False)
        print(json.dumps({"sample_id": runtime["sample_id"], "status": "ok", "output_dir": str(sample_dir)}, ensure_ascii=False))
    finally:
        recorder.remove()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
