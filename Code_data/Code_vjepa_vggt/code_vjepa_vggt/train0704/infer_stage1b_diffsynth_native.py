#!/usr/bin/env python3
"""
Stage1b DiffSynth-native inference script.

Loads a checkpoint produced by train_stage1b_diffsynth_native.py,
runs context-conditioned video generation using WanVideoPipeline + object branch.

Based on /home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main/examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py
but extended for our training pipeline (context mask injection, object branch).

Run command (foreground, GPU 7 via CUDA_VISIBLE_DEVICES; gpu4 is faulty, do not use):

    # step-000050
    cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt && \
    CUDA_VISIBLE_DEVICES=7 /data/gaoya/miniconda3/envs/vjepa2/bin/python \
      code_vjepa_vggt/train0704/infer_stage1b_diffsynth_native.py \
      --checkpoint /data/gaoya/AAA_test_video/stage1b_diffsynth_native/test_run/checkpoints/step-000050/checkpoint.safetensors \
      --context-video /data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center_trimmed/source_video/context_video_8f.mp4 \
      --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
      --output-dir /tmp/infer_stage1b_step050 \
      --gpu 0

    # step-000100
    cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt && \
    CUDA_VISIBLE_DEVICES=7 /data/gaoya/miniconda3/envs/vjepa2/bin/python \
      code_vjepa_vggt/train0704/infer_stage1b_diffsynth_native.py \
      --checkpoint /data/gaoya/AAA_test_video/stage1b_diffsynth_native/test_run/checkpoints/step-000100/checkpoint.safetensors \
      --context-video /data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center_trimmed/source_video/context_video_8f.mp4 \
      --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
      --output-dir /tmp/infer_stage1b_step100 \
      --gpu 0

Note: --gpu 0 refers to the first *visible* device, so it maps to physical GPU 7
via CUDA_VISIBLE_DEVICES=7.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange
from safetensors.torch import load_file

# Paths
DIFFSYNTH_PATH = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
VJEPA_PATH = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for p in [str(DIFFSYNTH_PATH), str(VJEPA_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

WAN_CKPT = "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
STAGE1A_LORA = "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint.safetensors")
    p.add_argument("--context-video", required=True, help="Context video mp4")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu", type=int, default=7)
    p.add_argument("--num-frames", type=int, default=16, help="Total frames to generate")
    p.add_argument("--num-context-frames", type=int, default=8)
    p.add_argument("--sampling-steps", type=int, default=20)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=896)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pipeline(device):
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    print("Loading WanVideoPipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=[
                WAN_CKPT + "/diffusion_pytorch_model-00001-of-00003.safetensors",
                WAN_CKPT + "/diffusion_pytorch_model-00002-of-00003.safetensors",
                WAN_CKPT + "/diffusion_pytorch_model-00003-of-00003.safetensors",
            ]),
            ModelConfig(path=WAN_CKPT + "/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=WAN_CKPT + "/Wan2.2_VAE.pth"),
        ],
    )
    return pipe


def setup_lora(pipe):
    from peft import LoraConfig, inject_adapter_in_model
    print("Injecting LoRA (rank=32)...")
    lora_config = LoraConfig(r=32, lora_alpha=32, target_modules=["q", "k", "v", "o"], bias="none")
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)

    # Load stage1a LoRA weights
    state = load_file(STAGE1A_LORA)
    lora_keys = {k: v for k, v in state.items() if "lora_" in k}
    print(f"  Loading {len(lora_keys)} LoRA keys from stage1a checkpoint...")
    pipe.dit.load_state_dict(lora_keys, strict=False)

    # Freeze everything in DiT
    for param in pipe.dit.parameters():
        param.requires_grad = False


def setup_object_branch(pipe):
    from code_vjepa_vggt.models.diffsynth_object_injection import inject_object_branch_to_dit
    print("Injecting object branch (after LoRA)...")
    inject_object_branch_to_dit(pipe.dit, object_cross_attn_dim=4096, object_gate_init=0.1)

    # Freeze object branch too (inference only)
    for name, param in pipe.dit.named_parameters():
        if "object_" in name or "norm4" in name:
            param.requires_grad = False


def load_checkpoint(pipe, ckpt_path):
    """Load object branch weights from our training checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    state = load_file(ckpt_path)

    # Extract dit.* keys and strip prefix
    dit_keys = {k[len("dit."):]: v for k, v in state.items() if k.startswith("dit.")}
    print(f"  Found {len(dit_keys)} DiT object branch keys")

    missing, unexpected = pipe.dit.load_state_dict(dit_keys, strict=False)
    obj_missing = [k for k in missing if "object_" in k or "norm4" in k]
    print(f"  Object branch: {len(obj_missing)} missing, {len(unexpected)} unexpected")
    if obj_missing:
        print(f"  WARNING: Missing object keys: {obj_missing[:5]}")


def read_video_frames(video_path, num_frames):
    """Read first num_frames from video, return [T, H, W, C] uint8."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())
    return np.stack(frames[:num_frames])


def preprocess_frames(frames_thwc, height, width):
    """Resize and normalize frames to [-1, 1], return [C, T, H, W] bfloat16."""
    resized = []
    for f in frames_thwc:
        f = cv2.resize(f, (width, height), interpolation=cv2.INTER_LANCZOS4)
        resized.append(f)
    arr = np.stack(resized).astype(np.float32) / 127.5 - 1.0  # [T, H, W, C] in [-1, 1]
    t = torch.from_numpy(arr).permute(3, 0, 1, 2)  # [C, T, H, W]
    return t.to(torch.bfloat16)


def find_ffmpeg():
    for candidate in [shutil.which("ffmpeg"),
                      "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
                      "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
                      "/usr/bin/ffmpeg"]:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("ffmpeg not found")


def write_mp4(path, frames_thwc, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = int(frames_thwc.shape[1]), int(frames_thwc.shape[2])
    ffmpeg = find_ffmpeg()
    tmp = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in frames_thwc:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run([ffmpeg, "-y", "-i", str(tmp), "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp.unlink(missing_ok=True)


def model_forward(pipe, latents, timestep, prompt_emb, object_context):
    """Manual DiT forward matching trainer._model_forward."""
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
    device = latents.device
    B, C, T, H, W = latents.shape

    # sinusoidal_embedding_1d expects a 1-D timestep tensor (matches training)
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)

    t = pipe.dit.time_embedding(
        sinusoidal_embedding_1d(pipe.dit.freq_dim, timestep).to(latents.dtype)
    )
    t_mod = pipe.dit.time_projection(t).unflatten(1, (6, pipe.dit.dim))
    context = pipe.dit.text_embedding(prompt_emb)

    x = pipe.dit.patch_embedding(latents)
    _, D, T_p, H_p, W_p = x.shape
    x = x.flatten(2).transpose(1, 2)

    freqs = torch.cat([
        pipe.dit.freqs[0][:T_p].view(T_p, 1, 1, -1).expand(T_p, H_p, W_p, -1),
        pipe.dit.freqs[1][:H_p].view(1, H_p, 1, -1).expand(T_p, H_p, W_p, -1),
        pipe.dit.freqs[2][:W_p].view(1, 1, W_p, -1).expand(T_p, H_p, W_p, -1),
    ], dim=-1).reshape(T_p * H_p * W_p, 1, -1).to(device)

    pipe.dit._object_context_holder["context"] = object_context
    try:
        for block in pipe.dit.blocks:
            x = block(x, context, t_mod, freqs)
    finally:
        pipe.dit._object_context_holder["context"] = None

    x = pipe.dit.head(x, t)
    x = rearrange(
        x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
        f=T_p, h=H_p, w=W_p,
        x=pipe.dit.patch_size[0], y=pipe.dit.patch_size[1], z=pipe.dit.patch_size[2]
    )
    return x


def run_sampling(pipe, prompt_emb, context_latents, num_frames, num_context_frames,
                 num_steps, device, dtype):
    """Run denoising loop with context mask injection."""
    from code_vjepa_vggt.utils.masks import latent_frame_mask, broadcast_latent_mask
    from diffsynth.diffusion.flow_match import FlowMatchScheduler

    vae_stride_t = 4
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(num_steps, shift=5.0)

    C_lat = context_latents.shape[0]
    H_lat = context_latents.shape[2]
    W_lat = context_latents.shape[3]
    total_lat_t = max(1, (num_frames - 1) // vae_stride_t + 1)

    # Build clean latent buffer
    latent_clean = torch.zeros(1, C_lat, total_lat_t, H_lat, W_lat, device=device, dtype=dtype)
    ctx_lat_t = context_latents.shape[1]
    copy_t = min(ctx_lat_t, total_lat_t)
    latent_clean[0, :, :copy_t] = context_latents[:, :copy_t]

    noise = torch.randn_like(latent_clean)
    sigma_0 = scheduler.sigmas[0].to(device=device, dtype=dtype)
    x_t = (1.0 - sigma_0) * latent_clean + sigma_0 * noise

    # Context mask
    context_mask_t, future_mask_t = latent_frame_mask(
        num_video_frames=num_frames,
        num_context_frames=num_context_frames,
        vae_stride_t=vae_stride_t,
        device=device,
    )
    context_mask = broadcast_latent_mask(context_mask_t, latent_clean)
    future_mask = broadcast_latent_mask(future_mask_t, latent_clean)

    # Keep context frames clean
    x_t = context_mask * latent_clean + (1.0 - context_mask) * x_t

    print(f"  Sampling {num_steps} steps, total_lat_t={total_lat_t}, ctx_lat_t={copy_t}")
    pipe.dit.eval()
    with torch.no_grad():
        for step_idx, timestep in enumerate(scheduler.timesteps):
            timestep_d = timestep.to(device=device, dtype=dtype)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model_forward(pipe, x_t, timestep_d, prompt_emb, object_context=None)
            pred_norm = float(pred.norm().item())
            print(f"    step {step_idx:02d}/{num_steps} | pred_norm={pred_norm:.1f}", flush=True)
            x_t = scheduler.step(pred, timestep, x_t)
            x_t = context_mask * latent_clean + (1.0 - context_mask) * x_t

    return x_t[0]  # [C, T, H, W]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and setup pipeline
    pipe = load_pipeline(device)
    setup_lora(pipe)
    setup_object_branch(pipe)
    load_checkpoint(pipe, args.checkpoint)

    pipe.dit.eval()
    for param in pipe.dit.parameters():
        param.requires_grad = False

    # Encode text
    print("Encoding prompt...")
    ids, mask = pipe.tokenizer([args.prompt], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        prompt_emb = pipe.text_encoder(ids, mask).to(dtype=dtype, device=device)
    print(f"  prompt_emb shape: {list(prompt_emb.shape)}")

    # Load context video
    print(f"Loading context video: {args.context_video}")
    frames = read_video_frames(args.context_video, args.num_context_frames)
    print(f"  Loaded {len(frames)} frames, shape: {frames.shape}")
    context_tensor = preprocess_frames(frames, args.height, args.width)  # [C, T, H, W]

    # Encode context with VAE
    print("Encoding context with VAE...")
    with torch.no_grad():
        context_latents_list = pipe.vae.encode(
            [context_tensor.to(device=device, dtype=dtype)],
            device=device
        )
    context_latents = context_latents_list[0]  # [C, T_lat, H_lat, W_lat]
    print(f"  context_latents shape: {list(context_latents.shape)}")

    # Free text encoder and VAE from GPU before DiT inference
    gc.collect()
    torch.cuda.empty_cache()

    # Run denoising
    print(f"Running denoising ({args.sampling_steps} steps)...")
    pred_latents = run_sampling(
        pipe=pipe,
        prompt_emb=prompt_emb,
        context_latents=context_latents,
        num_frames=args.num_frames,
        num_context_frames=args.num_context_frames,
        num_steps=args.sampling_steps,
        device=device,
        dtype=dtype,
    )
    print(f"  pred_latents shape: {list(pred_latents.shape)}")

    # Free DiT
    del pipe.dit
    pipe.dit = None
    gc.collect()
    torch.cuda.empty_cache()

    # Decode
    print("Decoding with VAE...")
    with torch.no_grad():
        decoded = pipe.vae.decode([pred_latents.to(device=device, dtype=dtype)], device=device)
    if isinstance(decoded, list):
        decoded = decoded[0]
    print(f"  decoded shape: {list(decoded.shape)}")

    # [B, C, T, H, W] or [C, T, H, W] -> [T, H, W, C] uint8
    video = decoded.detach().cpu().float()
    if video.dim() == 5:
        video = video[0]  # drop batch dim -> [C, T, H, W]
    video = video.permute(1, 2, 3, 0)  # [T, H, W, C]
    video = ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).numpy()

    # Save
    ckpt_name = Path(args.checkpoint).parent.name  # e.g. step-000100
    out_path = out_dir / f"{ckpt_name}_pred.mp4"
    write_mp4(out_path, video, args.fps)
    print(f"Saved: {out_path}")

    result = {
        "checkpoint": args.checkpoint,
        "context_video": args.context_video,
        "prompt": args.prompt,
        "output_video": str(out_path),
        "num_frames": args.num_frames,
        "num_context_frames": args.num_context_frames,
        "pred_latents_shape": list(pred_latents.shape),
    }
    with open(out_dir / f"{ckpt_name}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
