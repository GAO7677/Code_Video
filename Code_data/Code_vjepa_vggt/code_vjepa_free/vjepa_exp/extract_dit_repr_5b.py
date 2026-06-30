"""
Extract Wan2.2 TI2V-5B DiT multi-layer hidden-state features and Gram matrices.

Strategy (matches VideoREPA alignment approach used in extract_vjepa_gram.py):
  1. Load WanTI2V-5B via the official Wan2.2-main runtime (same path used by
     generate_wan_baseline.py / wanti2v.py in this project).
  2. VAE-encode the source video to get clean latents z.
  3. Add noise at a chosen timestep t (default 500/1000).
  4. Run one forward pass of model.model (WanModel, 30 blocks) with hooks on
     each WanAttentionBlock to capture x after that block.
  5. Each hook output is [1, seq_len, D=3072]; trim to real seq_len tokens,
     reshape to [T_p, H_p, W_p, D] using the patch grid, then trilinear-
     interpolate to the canonical V-JEPA grid [T=24, H=10, W=15, D].
  6. Compute gram_spatial [T,HW,HW] and gram_temporal [HW,T,T] (same as
     extract_vjepa_gram.py) and save to <stem>_dit5b.npz.

5B model geometry:
  patch_size   = (1, 2, 2)      # t_p, h_p, w_p
  vae_stride   = (4, 16, 16)    # temporal×4, spatial×16
  num_layers   = 30
  dim          = 3072

For 49-frame 704×1280 video:
  latent shape  : [C=16, T_lat=13, H_lat=44, W_lat=80]
  patch grid    : T_p=13, H_p=22, W_p=40  (seq_len=11440)
  align target  : T=24, H=10, W=15

Usage (in 'wan' environment):
    PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \\
    python extract_dit_repr_5b.py \\
        --video /path/to/video.mp4 \\
        --wan-root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \\
        --out-dir /data/gaoya/agent-data/outputs/vjepa_wan_precheck/repr_cache \\
        --prompt "A red ball rolls down a ramp." \\
        [--timestep 500] [--layers 0 4 9 14 19 24 29]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── canonical alignment grid (same as V-JEPA 2.1 Large output) ───────────────
ALIGN_T = 24
ALIGN_H = 10
ALIGN_W = 15

# ── 5B geometry constants ─────────────────────────────────────────────────────
WAN_PATCH_SIZE = (1, 2, 2)   # t_p, h_p, w_p
WAN_VAE_STRIDE = (4, 16, 16) # temporal, h, w
WAN_NUM_LAYERS = 30
WAN_HIDDEN_DIM = 3072

OFFICIAL_WAN_REPO = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
DEFAULT_WAN_ROOT  = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")


# ── Gram computation (identical to extract_vjepa_gram.py) ────────────────────

def compute_gram(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """tokens: [T, H, W, D]; returns gram_spatial [T,HW,HW] or gram_temporal [HW,T,T]."""
    T, H, W, D = tokens.shape
    HW = H * W
    tok = F.normalize(tokens.reshape(T, HW, D), dim=-1)

    if mode == "spatial":
        return torch.bmm(tok, tok.transpose(1, 2))          # [T, HW, HW]
    elif mode == "temporal":
        tok_t = tok.permute(1, 0, 2)                        # [HW, T, D]
        return torch.bmm(tok_t, tok_t.transpose(1, 2))      # [HW, T, T]
    else:
        raise ValueError(mode)


# ── Hook infrastructure ───────────────────────────────────────────────────────

class BlockOutputCapture:
    """Capture WanAttentionBlock output (x tensor) after each selected block."""

    def __init__(self, layer_indices: list[int]):
        self._target = set(layer_indices)
        self.captured: dict[int, torch.Tensor] = {}  # layer_idx → [1, seq_len, D]
        self._handles: list = []

    def register(self, wan_model) -> None:
        for i, block in enumerate(wan_model.blocks):
            if i in self._target:
                self._handles.append(
                    block.register_forward_hook(self._hook(i))
                )

    def _hook(self, idx: int):
        def fn(module, inputs, output):
            # WanAttentionBlock returns the updated x tensor [B, seq_len, D]
            self.captured[idx] = output.detach().cpu().float()
        return fn

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Alignment helper ──────────────────────────────────────────────────────────

def align_to_vjepa_grid(
    x_flat: torch.Tensor,   # [seq_len_real, D]
    T_p: int, H_p: int, W_p: int,
) -> torch.Tensor:
    """
    Reshape flat patch tokens to spatial grid then trilinear-interpolate to
    [ALIGN_T, ALIGN_H, ALIGN_W, D].
    """
    assert x_flat.shape[0] == T_p * H_p * W_p, \
        f"token count {x_flat.shape[0]} != T_p*H_p*W_p={T_p*H_p*W_p}"
    D = x_flat.shape[-1]

    # [T_p, H_p, W_p, D] → [1, D, T_p, H_p, W_p]
    vol = x_flat.reshape(T_p, H_p, W_p, D).permute(3, 0, 1, 2).unsqueeze(0)
    vol = F.interpolate(
        vol, size=(ALIGN_T, ALIGN_H, ALIGN_W),
        mode="trilinear", align_corners=False,
    )
    # [1, D, T, H, W] → [T, H, W, D]
    return vol.squeeze(0).permute(1, 2, 3, 0)


# ── Video loading (reuse logic from existing scripts) ─────────────────────────

def load_video_frames_cv2(video_path: str, num_frames: int = 49) -> torch.Tensor:
    """Return [C, F, H, W] float32 in [-1, 1]."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).float())
        i += 1
    cap.release()
    frames_t = torch.stack(frames[:num_frames]).permute(3, 0, 1, 2)  # [C, F, H, W]
    return frames_t / 127.5 - 1.0


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Wan2.2 TI2V-5B DiT hidden states and Gram matrices."
    )
    p.add_argument("--video", required=True, help="Source video (mp4/mov).")
    p.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT,
                   help="Wan2.2 TI2V-5B checkpoint directory.")
    p.add_argument("--out-dir", type=Path, default=Path("./repr_cache"))
    p.add_argument("--prompt", default="",
                   help="Text prompt (influences context embeddings).")
    p.add_argument("--negative-prompt", default="",
                   help="Negative prompt (not used in forward pass, included for parity).")
    p.add_argument("--timestep", type=int, default=500,
                   help="Noise level t in [0, 1000]. 500 = mid-noise.")
    p.add_argument("--layers", nargs="+", type=int,
                   default=list(range(WAN_NUM_LAYERS)),
                   help="Block indices to capture (0–29).")
    p.add_argument("--gram-modes", nargs="+", default=["spatial", "temporal"],
                   choices=["spatial", "temporal"])
    p.add_argument("--num-video-frames", type=int, default=49,
                   help="Frames sampled from the video for VAE encoding.")
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--device", default="cuda")
    p.add_argument("--t5-cpu", action="store_true", default=True,
                   help="Keep T5 on CPU (saves VRAM).")
    p.add_argument("--convert-model-dtype", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Import official Wan2.2 runtime ────────────────────────────────────────
    wan_repo = OFFICIAL_WAN_REPO.resolve()
    if str(wan_repo) not in sys.path:
        sys.path.insert(0, str(wan_repo))

    import wan
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel

    # Patch low_cpu_mem_usage default (matches wanti2v_runtime.py)
    if not getattr(WanModel, "_codex_low_cpu_patch", False):
        _orig_fp = WanModel.from_pretrained.__func__
        def _patched(cls, name_or_path, *a, **kw):
            kw.setdefault("low_cpu_mem_usage", False)
            return _orig_fp(cls, name_or_path, *a, **kw)
        WanModel.from_pretrained = classmethod(_patched)
        WanModel._codex_low_cpu_patch = True

    # ── Build WanTI2V pipeline (reuse same construction as wanti2v.py) ────────
    print(f"[5B] Loading WanTI2V from {args.wan_root}")
    cfg = WAN_CONFIGS["ti2v-5B"]
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(args.wan_root.resolve()),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    wan_model = wan_ti2v.model   # WanModel (30 blocks, dim=3072)
    vae       = wan_ti2v.vae
    t5_enc    = wan_ti2v.text_encoder
    wan_model.eval()

    # ── Text encoding ─────────────────────────────────────────────────────────
    print("[5B] Encoding text prompt")
    if args.t5_cpu:
        context = t5_enc([args.prompt], torch.device("cpu"))
        context = [c.to(device) for c in context]
    else:
        t5_enc.model.to(device)
        context = t5_enc([args.prompt], device)

    # ── Load & VAE-encode video ───────────────────────────────────────────────
    print(f"[5B] Loading video: {args.video}")
    frames = load_video_frames_cv2(args.video, num_frames=args.num_video_frames)
    # frames: [C, F, H, W]; resize to target resolution
    C, F_vid, H_vid, W_vid = frames.shape
    if H_vid != args.height or W_vid != args.width:
        flat = frames.permute(1, 0, 2, 3)   # [F, C, H, W]
        flat = F.interpolate(flat, (args.height, args.width),
                             mode="bicubic", align_corners=False)
        frames = flat.permute(1, 0, 2, 3)

    print(f"[5B] Frames shape: {frames.shape}")

    # Wan VAE encode expects list of [C, F, H, W] tensors in range [-1, 1]
    frames_dev = frames.to(device, dtype=torch.float32)
    with torch.no_grad():
        z_list = vae.encode([frames_dev])   # list of [C_lat, T_lat, H_lat, W_lat]
    z = z_list[0]                           # [C_lat=16, T_lat, H_lat, W_lat]
    C_lat, T_lat, H_lat, W_lat = z.shape
    print(f"[5B] Latent shape: {z.shape}")

    # Patch grid dimensions
    t_p, h_p, w_p = WAN_PATCH_SIZE
    T_p = T_lat // t_p
    H_p = H_lat // h_p
    W_p = W_lat // w_p
    real_seq_len = T_p * H_p * W_p
    # seq_len must be divisible by sp_size=1
    seq_len = real_seq_len
    print(f"[5B] Patch grid: T_p={T_p}, H_p={H_p}, W_p={W_p}  seq_len={seq_len}")
    print(f"[5B] Align target: T={ALIGN_T}, H={ALIGN_H}, W={ALIGN_W}")

    # ── Add noise at timestep t ───────────────────────────────────────────────
    # Use a simple linear noise schedule matching Wan's flow formulation:
    # x_t = (1 - sigma) * z + sigma * noise,  sigma = t / num_train_steps
    num_train_steps = getattr(wan_ti2v, "num_train_timesteps", 1000)
    sigma = args.timestep / num_train_steps
    noise = torch.randn_like(z)
    noisy_z = (1.0 - sigma) * z + sigma * noise
    print(f"[5B] Noisy latent at t={args.timestep}, sigma={sigma:.3f}")

    # ── Register hooks ────────────────────────────────────────────────────────
    capture = BlockOutputCapture(args.layers)
    wan_model.to(device)
    capture.register(wan_model)

    # ── Single forward pass (no denoising loop) ───────────────────────────────
    print("[5B] Running single forward pass…")
    # Build per-token timestep tensor matching generate_context format
    t_scalar = torch.tensor([args.timestep], dtype=torch.float32, device=device)
    # Wan expects timestep shape [B, seq_len] (per-token) or [B]
    timestep = t_scalar.expand(1, seq_len)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = wan_model(
            [noisy_z.to(device)],
            t=timestep,
            context=context,
            seq_len=seq_len,
        )

    capture.remove()
    print(f"[5B] Captured {len(capture.captured)} layers")

    # ── Align, compute Grams, save ───────────────────────────────────────────
    save: dict[str, np.ndarray] = {}

    for layer_idx in sorted(capture.captured):
        raw = capture.captured[layer_idx]  # [1, seq_len_padded, D]
        x_flat = raw[0, :real_seq_len, :]  # [real_seq_len, D]

        aligned = align_to_vjepa_grid(x_flat, T_p, H_p, W_p)  # [T, H, W, D]

        key = f"layer_{layer_idx:02d}"
        save[f"tokens_{key}"] = aligned.half().numpy()

        for mode in args.gram_modes:
            gram = compute_gram(aligned, mode)
            save[f"gram_{mode}_{key}"] = gram.half().numpy()
            print(f"  block {layer_idx:2d} | gram_{mode}: {gram.shape}")

    # metadata
    save["layers"]     = np.array(args.layers, dtype=np.int32)
    save["align_T"]    = np.array(ALIGN_T)
    save["align_H"]    = np.array(ALIGN_H)
    save["align_W"]    = np.array(ALIGN_W)
    save["embed_dim"]  = np.array(WAN_HIDDEN_DIM)
    save["num_layers"] = np.array(WAN_NUM_LAYERS)
    save["timestep"]   = np.array(args.timestep)
    save["lat_shape"]  = np.array([C_lat, T_lat, H_lat, W_lat])
    save["patch_grid"] = np.array([T_p, H_p, W_p])

    stem = Path(args.video).stem
    out_path = args.out_dir / f"{stem}_dit5b.npz"
    np.savez_compressed(str(out_path), **save)
    print(f"[5B] Saved → {out_path}")


if __name__ == "__main__":
    main()
