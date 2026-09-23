#!/usr/bin/env python3
"""Offline scheduler, endpoint, and context-noise interpolation checks."""

import json
import math
import sys
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

TRAIN = Path("/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/train")
sys.path.insert(0, str(TRAIN))
import train_full_sa_pybullet_aux as train  # noqa: E402


class DeterministicForward:
    def __call__(self, *, hidden_states, timestep, **kwargs):
        token_bias = timestep.float().mean(dim=1).view(-1, 1, 1, 1, 1)
        return (hidden_states + token_bias.to(hidden_states.dtype),)


def run(mode, scale, clean, prompt, scheduler, noise, fixed_index):
    if mode == "C":
        all_frame_noise, timestep_mode = False, "force_zero"
    else:
        all_frame_noise, timestep_mode = True, "match_noise"
    return train._flow_forward(
        DeterministicForward(), scheduler, clean, prompt,
        context_latent_frames=2, patch_size=(1, 1, 1), fixed_index=fixed_index,
        all_frame_noise=all_frame_noise, full_frame_flow=False,
        context_timestep_mode=timestep_mode, context_noise_scale=scale,
        noise_override=noise,
    )


def maxdiff(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers", subfolder="scheduler"
    )
    scheduler.set_timesteps(1000, device=device)
    assert len(scheduler.timesteps) == 1000
    assert len(scheduler.sigmas) == 1001
    assert float(scheduler.sigmas[-1]) == 0.0
    assert len(scheduler.timesteps) != len(scheduler.sigmas)

    clean = torch.randn((1, 4, 13, 2, 2), device=device, dtype=dtype)
    prompt = torch.randn((1, 4, 8), device=device, dtype=dtype)
    noise = torch.randn_like(clean)
    fixed_index = 333
    with torch.no_grad():
        loss_c, state_c = run("C", 0.0, clean, prompt, scheduler, noise, fixed_index)
        loss_new0, state_new0 = run("new", 0.0, clean, prompt, scheduler, noise, fixed_index)
        loss_a, state_a = train._flow_forward(
            DeterministicForward(), scheduler, clean, prompt,
            context_latent_frames=2, patch_size=(1, 1, 1), fixed_index=fixed_index,
            all_frame_noise=True, full_frame_flow=False,
            context_timestep_mode="match_noise", noise_override=noise,
        )
        loss_new1, state_new1 = run("new", 1.0, clean, prompt, scheduler, noise, fixed_index)

    checks = {
        "lambda0_model_input_max_abs": maxdiff(state_new0["model_input"], state_c["model_input"]),
        "lambda0_token_timestep_max_abs": maxdiff(state_new0["token_timesteps"], state_c["token_timesteps"]),
        "lambda0_forward_max_abs": maxdiff(state_new0["prediction"], state_c["prediction"]),
        "lambda0_loss_abs": abs(float(loss_new0) - float(loss_c)),
        "lambda1_model_input_max_abs": maxdiff(state_new1["model_input"], state_a["model_input"]),
        "lambda1_token_timestep_max_abs": maxdiff(state_new1["token_timesteps"], state_a["token_timesteps"]),
        "lambda1_forward_max_abs": maxdiff(state_new1["prediction"], state_a["prediction"]),
        "lambda1_loss_abs": abs(float(loss_new1) - float(loss_a)),
        "lambda0_context_timestep": float(state_new0["context_timestep"].item()),
        "lambda1_context_timestep": float(state_new1["context_timestep"].item()),
        "lambda1_future_timestep": float(state_new1["future_timestep"].item()),
        "loss_latent_start_C": float(state_c["loss_latent_start"].item()),
        "loss_latent_start_A": float(state_a["loss_latent_start"].item()),
    }
    tolerance = 1e-6 if dtype == torch.float32 else 1e-3
    for key in checks:
        if "lambda0_" in key or "lambda1_" in key:
            if key.endswith("context_timestep") or key.endswith("future_timestep"):
                continue
            assert checks[key] <= tolerance, (key, checks[key], tolerance)
    assert checks["loss_latent_start_C"] == checks["loss_latent_start_A"] == 2.0
    assert checks["lambda0_context_timestep"] == 0.0
    assert checks["lambda1_context_timestep"] == checks["lambda1_future_timestep"]

    stats = {}
    generator = torch.Generator(device=device).manual_seed(42)
    index_samples = torch.randint(0, len(scheduler.timesteps), (1000,), generator=generator, device=device)
    for scale in (0.25, 0.5, 0.75):
        ratios, errors = [], []
        for index in index_samples:
            i = int(index.item())
            future_sigma = scheduler.sigmas[i]
            tc, actual, target = train._context_noise_pair(
                scheduler,
                scheduler.timesteps[i:i + 1], future_sigma.reshape(1), scale,
                device=device,
            )
            ratio = float((actual / future_sigma).item())
            ratios.append(ratio)
            errors.append(abs(ratio - scale))
            assert tc.numel() == actual.numel() == target.numel() == 1
        errors.sort()
        stats[str(scale)] = {
            "mean_ratio": sum(ratios) / len(ratios),
            "std_ratio": float(torch.tensor(ratios).std(unbiased=False)),
            "max_abs_error": max(errors),
            "p95_abs_error": errors[math.ceil(0.95 * len(errors)) - 1],
        }

    print(json.dumps({"device": str(device), "scheduler": {
        "class": type(scheduler).__name__, "diffusers": "0.35.0.dev0",
        "paired_training_entries": len(scheduler.timesteps),
        "terminal_unpaired_sigma": float(scheduler.sigmas[-1]),
    }, "endpoint_checks": checks, "interpolation_stats_1000": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
