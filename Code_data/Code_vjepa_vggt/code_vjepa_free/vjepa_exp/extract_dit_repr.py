"""
Extract Wan2.1 DiT multi-layer hidden-state features and Gram matrices from a video.

The script:
  1. VAE-encodes the video to get latents
  2. Adds noise at a chosen timestep (default t=500)
  3. Runs one forward pass of the Wan transformer with forward hooks on each block
  4. Spatially aligns the captured hidden states to the same T×H×W grid as V-JEPA
     (T=24, H=10, W=15)  via trilinear interpolation
  5. Computes Gram matrices and saves everything as .npz

Usage:
    python extract_dit_repr.py \
        --video /path/to/video.mp4 \
        --model-path /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers \
        --out-dir ./repr_cache \
        [--timestep 500] [--layers 0 9 14 19 24 29]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── aligned grid (same as V-JEPA output) ──────────────────────────────────────
ALIGN_T = 24   # matches V-JEPA T tokens
ALIGN_H = 10   # matches V-JEPA H patches
ALIGN_W = 15   # matches V-JEPA W patches

# ── Wan 1.3B geometry ─────────────────────────────────────────────────────────
WAN_NUM_LAYERS = 30
WAN_PATCH_T    = 1
WAN_PATCH_H    = 2
WAN_PATCH_W    = 2
WAN_HIDDEN     = 1536

# ── VAE constants ──────────────────────────────────────────────────────────────
VAE_TEMPORAL_COMPRESS = 4   # VAE downsamples temporal by 4  (49→13)
VAE_SPATIAL_COMPRESS  = 8   # VAE downsamples spatial by 8   (480→60)


def load_video_frames(video_path: str, num_frames: int = 49) -> torch.Tensor:
    """Return [C, F, H, W] float32 in [-1, 1]."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).float()   # [F, H, W, C]
        frames = frames.permute(3, 0, 1, 2)      # [C, F, H, W]
        return frames / 127.5 - 1.0
    except ImportError:
        pass

    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_set = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frame = __import__("cv2").cvtColor(frame, __import__("cv2").COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).float())
        i += 1
    cap.release()
    frames = torch.stack(frames[:num_frames]).permute(3, 0, 1, 2)
    return frames / 127.5 - 1.0


def preprocess_for_wan(frames: torch.Tensor, target_h: int = 480, target_w: int = 480) -> torch.Tensor:
    """
    frames: [C, F, H, W] in [-1, 1]
    Returns [1, C, F, H', W'] ready for Wan VAE (default 480×480).
    """
    C, F, H, W = frames.shape
    flat = frames.permute(1, 0, 2, 3)  # [F, C, H, W]
    flat = F.interpolate(flat, (target_h, target_w), mode="bicubic", align_corners=False)
    out = flat.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H', W']
    return out


@torch.no_grad()
def encode_video_with_vae(vae, frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """frames: [1, C, F, H, W]; returns latents [1, C_lat, T_lat, H_lat, W_lat]."""
    frames = frames.to(device, dtype=torch.float16)
    # WanVAE expects input in the format [B, C, F, H, W]
    latents = vae.encode(frames).latent_dist.sample()
    latents = latents * vae.config.scaling_factor
    return latents


def compute_gram(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """
    tokens: [T, H, W, D]  (already L2-normalized)
    mode: 'spatial'  → [T, HW, HW]
          'temporal' → [HW, T, T]
    """
    T, H, W, D = tokens.shape
    HW = H * W
    tok = F.normalize(tokens, dim=-1)

    if mode == "spatial":
        flat = tok.reshape(T, HW, D)
        return torch.bmm(flat, flat.transpose(1, 2))         # [T, HW, HW]
    elif mode == "temporal":
        flat = tok.reshape(T, HW, D).permute(1, 0, 2)       # [HW, T, D]
        return torch.bmm(flat, flat.transpose(1, 2))         # [HW, T, T]
    else:
        raise ValueError(f"unknown mode {mode}")


class HiddenStateCapture:
    """Register forward hooks on WanTransformerBlock to capture output hidden states."""

    def __init__(self, layer_indices: list[int]):
        self.layer_indices = set(layer_indices)
        self.captured: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def register(self, transformer):
        for i, block in enumerate(transformer.blocks):
            if i in self.layer_indices:
                h = block.register_forward_hook(self._make_hook(i))
                self._hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output shape: [B, N, D]
            self.captured[layer_idx] = output.detach().cpu().float()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.captured.clear()


def align_hidden_to_vjepa_grid(
    hidden: torch.Tensor,
    lat_T: int,
    lat_H: int,
    lat_W: int,
) -> torch.Tensor:
    """
    hidden: [N, D] where N = lat_T * (lat_H//p_h) * (lat_W//p_w)
    Returns [ALIGN_T, ALIGN_H, ALIGN_W, D]

    Alignment strategy (mirrors VideoREPA):
      - reshape to [D, lat_T, H_patch, W_patch] (DiT patch grid after patch embedding)
      - trilinear interpolate to [D, ALIGN_T, ALIGN_H, ALIGN_W]
    """
    H_patch = lat_H // WAN_PATCH_H
    W_patch = lat_W // WAN_PATCH_W
    T_patch = lat_T // WAN_PATCH_T

    N, D = hidden.shape
    assert N == T_patch * H_patch * W_patch, f"N={N} vs {T_patch*H_patch*W_patch}"

    # [N, D] → [D, T, H, W]
    x = hidden.reshape(T_patch, H_patch, W_patch, D).permute(3, 0, 1, 2).unsqueeze(0)
    # → [1, D, T, H, W]

    # trilinear to [1, D, ALIGN_T, ALIGN_H, ALIGN_W]
    x = F.interpolate(x, size=(ALIGN_T, ALIGN_H, ALIGN_W), mode="trilinear", align_corners=False)

    # [D, T, H, W] → [T, H, W, D]
    x = x.squeeze(0).permute(1, 2, 3, 0)   # [T, H, W, D]
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-path",
                        default="/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--out-dir", default="./repr_cache")
    parser.add_argument("--timestep", type=int, default=500,
                        help="noise level t ∈ [0, 1000]; 500 = moderate noise")
    parser.add_argument("--layers", nargs="+", type=int,
                        default=list(range(WAN_NUM_LAYERS)),
                        help="DiT block indices to capture (0-29)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gram-modes", nargs="+", default=["spatial", "temporal"],
                        choices=["spatial", "temporal"])
    parser.add_argument("--num-video-frames", type=int, default=49,
                        help="frames to sample from video")
    parser.add_argument("--spatial-size", type=int, default=480,
                        help="spatial resolution fed to VAE (square)")
    parser.add_argument("--prompt", type=str, default="",
                        help="text prompt (can be empty, only used to shape encoder_hidden_states)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load models ──────────────────────────────────────────────────────────
    print("[DiT] Loading models from", args.model_path)
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    vae = AutoencoderKLWan.from_pretrained(
        args.model_path, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    vae.eval()

    transformer = WanTransformer3DModel.from_pretrained(
        args.model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_path, subfolder="scheduler"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)
    text_encoder.eval()

    # ── Encode text ──────────────────────────────────────────────────────────
    print("[DiT] Encoding text prompt")
    with torch.no_grad():
        text_inputs = tokenizer(
            [args.prompt],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        enc_out = text_encoder(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )
        encoder_hidden_states = enc_out.last_hidden_state  # [1, seq, D_text]

    # ── Load & preprocess video ───────────────────────────────────────────────
    print(f"[DiT] Loading video: {args.video}")
    frames = load_video_frames(args.video, num_frames=args.num_video_frames)
    video_tensor = preprocess_for_wan(frames, args.spatial_size, args.spatial_size)
    print(f"[DiT] Preprocessed video: {video_tensor.shape}")

    # ── VAE encode ───────────────────────────────────────────────────────────
    print("[DiT] VAE encoding…")
    with torch.no_grad():
        latents = encode_video_with_vae(vae, video_tensor, device)
    print(f"[DiT] Latents: {latents.shape}")
    # shape: [1, C_lat, T_lat, H_lat, W_lat]  e.g. [1, 16, 13, 60, 60]
    _, C_lat, T_lat, H_lat, W_lat = latents.shape

    # ── Add noise ────────────────────────────────────────────────────────────
    scheduler.set_timesteps(1000)
    t_idx = args.timestep  # use t directly as step index
    timesteps_tensor = torch.tensor([args.timestep], device=device, dtype=torch.long)

    latents_bf16 = latents.to(torch.bfloat16)
    noise = torch.randn_like(latents_bf16)
    # Use scheduler sigma at the chosen step for manual noise addition
    sigma = scheduler.sigmas[t_idx].to(device)
    noisy_latents = latents_bf16 + sigma * noise
    print(f"[DiT] Added noise at t={args.timestep}, sigma={sigma.item():.4f}")

    # ── Register hooks ───────────────────────────────────────────────────────
    capture = HiddenStateCapture(args.layers)
    capture.register(transformer)

    # ── Forward pass ─────────────────────────────────────────────────────────
    print("[DiT] Running forward pass…")
    with torch.no_grad():
        _ = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_tensor,
            encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
            return_dict=False,
        )

    capture.remove()
    print(f"[DiT] Captured {len(capture.captured)} layers")

    # ── Align & compute Grams ────────────────────────────────────────────────
    H_patch = H_lat // WAN_PATCH_H
    W_patch = W_lat // WAN_PATCH_W
    T_patch = T_lat // WAN_PATCH_T
    print(f"[DiT] DiT token grid: T={T_patch}, H={H_patch}, W={W_patch}")
    print(f"[DiT] Aligning to:    T={ALIGN_T}, H={ALIGN_H}, W={ALIGN_W}")

    save = {}
    for layer_idx in sorted(capture.captured):
        hidden = capture.captured[layer_idx].squeeze(0)  # [N, D]
        aligned = align_hidden_to_vjepa_grid(hidden, T_lat, H_lat, W_lat)
        # aligned: [T, H, W, D]
        key = f"layer_{layer_idx:02d}"
        save[f"tokens_{key}"] = aligned.half().numpy()

        for mode in args.gram_modes:
            gram = compute_gram(aligned, mode=mode)
            save[f"gram_{mode}_{key}"] = gram.half().numpy()
            print(f"  layer {layer_idx:2d} | gram_{mode}: {gram.shape}")

    # metadata
    save["layers"] = np.array(args.layers)
    save["align_T"] = np.array(ALIGN_T)
    save["align_H"] = np.array(ALIGN_H)
    save["align_W"] = np.array(ALIGN_W)
    save["embed_dim"] = np.array(WAN_HIDDEN)
    save["timestep"] = np.array(args.timestep)
    save["lat_shape"] = np.array([C_lat, T_lat, H_lat, W_lat])

    stem = Path(args.video).stem
    out_path = Path(args.out_dir) / f"{stem}_dit.npz"
    np.savez_compressed(str(out_path), **save)
    print(f"[DiT] Saved to {out_path}")


if __name__ == "__main__":
    main()
