"""
Extract V-JEPA 2.1 ViT-L multi-layer features and Gram matrices from a video.

Usage:
    python extract_vjepa_gram.py \
        --video /path/to/video.mp4 \
        --ckpt /data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt \
        --out-dir ./repr_cache \
        [--layers 0 5 11 17 23]  # default: all 24 layers
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

sys.path.insert(0, str(Path(__file__).parent))
from local_vjepa21_backbone import build_vjepa2_1_vit_large_384_encoder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# V-JEPA 2.1 ViT-L constants
VJEPA_FRAMES = 48      # input temporal dim (first frame removed → 48 used)
VJEPA_H = 160          # spatial height after /3 downsample of 480
VJEPA_W = 240          # spatial width  after /3 downsample of 720 (or W)
PATCH = 16
TUBELET = 2
DEPTH = 24             # ViT-L depth


def load_video_frames(video_path: str, num_frames: int = 49) -> torch.Tensor:
    """Load `num_frames` uniformly sampled frames, return [C, F, H, W] float32 in [-1, 1]."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).float()  # [F, H, W, C] 0-255
        frames = frames.permute(3, 0, 1, 2)     # [C, F, H, W]
        frames = frames / 127.5 - 1.0
        return frames
    except ImportError:
        pass

    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).float())
        idx += 1
    cap.release()
    frames = torch.stack(frames[:num_frames])  # [F, H, W, C]
    frames = frames.permute(3, 0, 1, 2)        # [C, F, H, W]
    frames = frames / 127.5 - 1.0
    return frames


def preprocess_for_vjepa(frames: torch.Tensor) -> torch.Tensor:
    """
    frames: [C, F, H, W] in [-1, 1]
    Returns [1, C, F, H', W'] ready for V-JEPA encoder, where H'=VJEPA_H, W'=VJEPA_W.
    Mirrors VideoREPA lora_trainer.py:168-191.
    """
    C, F, H, W = frames.shape
    # skip first frame then resize → [C, 48, H, W]
    frames = frames[:, 1:]          # drop first frame (VideoREPA convention)
    F2 = frames.shape[1]

    # normalize to ImageNet stats (from [-1,1] → [0,1] first)
    frames_01 = (frames + 1.0) / 2.0   # [C, F, H, W]
    # apply per-channel normalization frame-by-frame via broadcasting
    mean = torch.tensor(IMAGENET_MEAN, device=frames.device).view(3, 1, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=frames.device).view(3, 1, 1, 1)
    frames_norm = (frames_01 - mean) / std  # [C, F, H, W]

    # spatial resize to 160×240 (H//3, W//3 for 480×720) via bicubic interpolation
    frames_flat = frames_norm.permute(1, 0, 2, 3)  # [F, C, H, W]
    frames_flat = F.interpolate(frames_flat, (VJEPA_H, VJEPA_W), mode="bicubic", align_corners=False)
    frames_out = frames_flat.permute(1, 0, 2, 3)   # [C, F, H', W']

    return frames_out.unsqueeze(0)   # [1, C, F, H', W']


@torch.no_grad()
def extract_vjepa_features(
    encoder,
    video_tensor: torch.Tensor,  # [1, C, F, H', W']
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Returns list of tensors, one per layer in encoder.out_layers.
    Each tensor: [T, H_patches, W_patches, D]  (batch dim squeezed)
    """
    video_tensor = video_tensor.to(device, dtype=torch.float32)
    outs = encoder(video_tensor)  # list of [1, N, D]

    T = VJEPA_FRAMES // TUBELET           # 24
    H_p = VJEPA_H // PATCH                # 10
    W_p = VJEPA_W // PATCH                # 15
    results = []
    for feat in outs:
        # feat: [1, T*H_p*W_p, D]
        B, N, D = feat.shape
        assert N == T * H_p * W_p, f"unexpected token count {N} vs {T*H_p*W_p}"
        feat = feat.squeeze(0).reshape(T, H_p, W_p, D)  # [T, H, W, D]
        results.append(feat.cpu())
    return results


def compute_gram(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """
    tokens: [T, H, W, D]  normalized
    mode: 'spatial'  → [T, HW, HW]
          'temporal' → [HW, T, T]
          'joint'    → [T*HW, T*HW]
    """
    T, H, W, D = tokens.shape
    HW = H * W
    tokens_norm = F.normalize(tokens, dim=-1)

    if mode == "spatial":
        # For each frame t: [HW, D] @ [D, HW] → [HW, HW]
        flat = tokens_norm.reshape(T, HW, D)         # [T, HW, D]
        gram = torch.bmm(flat, flat.transpose(1, 2)) # [T, HW, HW]
        return gram

    elif mode == "temporal":
        # For each spatial position s: [T, D] @ [D, T] → [T, T]
        flat = tokens_norm.reshape(T, HW, D).permute(1, 0, 2)  # [HW, T, D]
        gram = torch.bmm(flat, flat.transpose(1, 2))            # [HW, T, T]
        return gram

    elif mode == "joint":
        flat = tokens_norm.reshape(T * HW, D)          # [T*HW, D]
        gram = torch.mm(flat, flat.t())                 # [T*HW, T*HW]
        return gram

    else:
        raise ValueError(f"unknown mode {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--ckpt", default="/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
    parser.add_argument("--out-dir", default="./repr_cache")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(DEPTH)),
                        help="which ViT-L layers to extract (0-23)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gram-modes", nargs="+", default=["spatial", "temporal"],
                        choices=["spatial", "temporal", "joint"])
    parser.add_argument("--num-video-frames", type=int, default=49,
                        help="frames to sample from video (first will be dropped)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[V-JEPA] Loading encoder, layers={args.layers}")
    encoder = build_vjepa2_1_vit_large_384_encoder(
        checkpoint_path=args.ckpt,
        out_layers=args.layers,
        map_location="cpu",
    )
    encoder.eval().to(device)

    print(f"[V-JEPA] Loading video: {args.video}")
    frames = load_video_frames(args.video, num_frames=args.num_video_frames)
    video_tensor = preprocess_for_vjepa(frames)
    print(f"[V-JEPA] Preprocessed video: {video_tensor.shape}")

    print("[V-JEPA] Extracting features...")
    feats = extract_vjepa_features(encoder, video_tensor, device)
    print(f"[V-JEPA] Got {len(feats)} layer outputs, each {feats[0].shape}")

    # Build save dict
    save = {}
    for i, (layer_idx, feat) in enumerate(zip(args.layers, feats)):
        key = f"layer_{layer_idx:02d}"
        # store tokens as float16 to save disk
        save[f"tokens_{key}"] = feat.half().numpy()   # [T, H, W, D]
        for mode in args.gram_modes:
            gram = compute_gram(feat.float(), mode=mode)
            save[f"gram_{mode}_{key}"] = gram.half().numpy()
            print(f"  layer {layer_idx:2d} | gram_{mode}: {gram.shape}")

    # metadata
    save["layers"] = np.array(args.layers)
    save["grid_T"] = np.array(VJEPA_FRAMES // TUBELET)
    save["grid_H"] = np.array(VJEPA_H // PATCH)
    save["grid_W"] = np.array(VJEPA_W // PATCH)
    save["embed_dim"] = np.array(1024)

    stem = Path(args.video).stem
    out_path = Path(args.out_dir) / f"{stem}_vjepa.npz"
    np.savez_compressed(str(out_path), **save)
    print(f"[V-JEPA] Saved to {out_path}")


if __name__ == "__main__":
    main()
