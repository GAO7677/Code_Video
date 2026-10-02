"""Wan-5B-compatible flow noise and future-only loss for A/C/N75."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


MODE_SCALES = {"A": 1.0, "C": 0.0, "N75": 0.75}


def training_scheduler(device, num_train_timesteps=1000, shift=1.0):
    # The old trainer instantiated FlowMatch from an inference UniPC config.
    # Its effective training `shift` was 1.0; `flow_shift=5` was not used.
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(num_train_timesteps), shift=float(shift)
    )
    scheduler.set_timesteps(int(num_train_timesteps), device=device)
    return scheduler


def context_noise_pair(scheduler, future_timestep, future_sigma, scale):
    """Use the old trainer's paired sigma nearest-neighbor rule, including ties."""
    if not 0.0 <= float(scale) <= 1.0:
        raise ValueError("context noise scale must be in [0, 1]")
    if scale == 0.0:
        return torch.zeros_like(future_timestep), torch.zeros_like(future_sigma), torch.zeros_like(future_sigma)
    if scale == 1.0:
        return future_timestep, future_sigma, future_sigma
    paired = scheduler.sigmas[:len(scheduler.timesteps)].to(
        device=future_sigma.device, dtype=future_sigma.dtype
    )
    target = future_sigma * float(scale)
    index = torch.argmin((paired - target.reshape(-1)[0]).abs())
    return (
        scheduler.timesteps[index].to(future_timestep).reshape_as(future_timestep),
        paired[index].reshape_as(future_sigma),
        target,
    )


@dataclass
class FlowBatch:
    latents: torch.Tensor
    timesteps: torch.Tensor
    target: torch.Tensor
    future_timestep: torch.Tensor
    future_sigma: torch.Tensor
    context_timestep: torch.Tensor
    context_sigma: torch.Tensor
    context_sigma_target: torch.Tensor
    expert: str


def flow_batch(clean, scheduler, mode, index, context_frames=2, noise=None, boundary=0.875):
    if mode not in MODE_SCALES:
        raise ValueError(f"unsupported training mode: {mode}")
    if clean.ndim != 5 or not 0 < context_frames < clean.shape[2]:
        raise ValueError("expected B,C,T,H,W with nonempty context and future")
    if not 0 <= int(index) < len(scheduler.timesteps):
        raise ValueError("training timestep index out of range")
    noise = torch.randn_like(clean) if noise is None else noise
    if noise.shape != clean.shape or noise.device != clean.device:
        raise ValueError("noise must have the same shape and device as clean latents")
    timestep = scheduler.timesteps[int(index)].reshape(1)
    sigma = scheduler.sigmas[int(index)].to(clean.device)
    # scheduler.scale_noise broadcasts sigma to B,1,1,1,1; this also preserves
    # the old trainer's float32 intermediate/operation order on bf16 caches.
    noisy = scheduler.scale_noise(clean, timestep, noise)
    context_t, context_s, target_s = context_noise_pair(
        scheduler, timestep, sigma, MODE_SCALES[mode]
    )
    if mode == "A":
        model_input = noisy
    elif mode == "C":
        mask = torch.ones_like(clean[:, :1])
        mask[:, :, :context_frames] = 0
        model_input = (1 - mask) * clean + mask * noisy
    else:
        context = (1.0 - context_s) * clean[:, :, :context_frames] + context_s * noise[:, :, :context_frames]
        model_input = torch.cat((context, noisy[:, :, context_frames:]), dim=2)
    frame_t = timestep.reshape(1, 1).expand(clean.shape[0], clean.shape[2]).clone().float()
    frame_t[:, :context_frames] = context_t.float()
    return FlowBatch(
        latents=model_input, timesteps=frame_t, target=noise - clean,
        future_timestep=timestep, future_sigma=sigma,
        context_timestep=context_t, context_sigma=context_s,
        context_sigma_target=target_s,
        expert="high" if float(timestep.item()) >= float(boundary) * 1000 else "low",
    )


def future_loss(prediction, target, context_frames=2):
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if not 0 < context_frames < prediction.shape[2]:
        raise ValueError("invalid future loss range")
    return F.mse_loss(prediction[:, :, context_frames:].float(), target[:, :, context_frames:].float())
