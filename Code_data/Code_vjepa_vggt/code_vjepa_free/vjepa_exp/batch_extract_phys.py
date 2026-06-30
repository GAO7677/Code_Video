"""
Batch V-JEPA + Wan DiT-5B feature extraction for phys_compare_manifest.json.

Loads each model ONCE, then loops over all 150 manifest entries.
Outputs per-video {stem}_vjepa.npz and {stem}_dit5b.npz to --out-dir.
Skips already-computed files (idempotent).

Usage (wan env):
    PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \\
    python batch_extract_phys.py \\
        --manifest /data/gaoya/agent-data/outputs/phys_compare_manifest.json \\
        --out-dir  /data/gaoya/agent-data/outputs/phys_compare/repr_cache \\
        [--vjepa-only | --dit-only] [--max-samples N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Project paths ─────────────────────────────────────────────────────────────
EXP_DIR     = Path(__file__).parent.resolve()
VJEPA2_ROOT = Path("/home/gaoya/Code_Video/vjepa2-main")
WAN_REPO    = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
VJEPA_CKPT  = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
WAN_ROOT    = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(VJEPA2_ROOT))
sys.path.insert(0, str(WAN_REPO))

from local_vjepa21_backbone import build_vjepa2_1_vit_large_384_encoder

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VJEPA_FRAMES  = 48
VJEPA_H, VJEPA_W = 160, 240
PATCH, TUBELET    = 16, 2
VJEPA_DEPTH       = 24

WAN_PATCH_SIZE = (1, 2, 2)
WAN_VAE_STRIDE = (4, 16, 16)
WAN_NUM_LAYERS = 30
WAN_HIDDEN_DIM = 3072

ALIGN_T, ALIGN_H, ALIGN_W = 24, 10, 15

# Default layers to extract (sparse subset to save memory and time)
VJEPA_LAYERS = [0, 3, 5, 8, 11, 14, 17, 20, 23]
DIT_LAYERS   = list(range(30))


# ═══════════════════════════════════════════════════════════════════════════════
# Video loading
# ═══════════════════════════════════════════════════════════════════════════════

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
    except Exception:
        pass

    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_set = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    frames_list, i = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(torch.from_numpy(frame).float())
        i += 1
    cap.release()
    frames_t = torch.stack(frames_list[:num_frames]).permute(3, 0, 1, 2)
    return frames_t / 127.5 - 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Gram computation (shared between both extractors)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gram(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    """tokens: [T, H, W, D]; returns gram_spatial [T,HW,HW] or gram_temporal [HW,T,T]."""
    T, H, W, D = tokens.shape
    HW = H * W
    tok = F.normalize(tokens.reshape(T, HW, D), dim=-1)
    if mode == "spatial":
        return torch.bmm(tok, tok.transpose(1, 2))           # [T, HW, HW]
    else:  # temporal
        tok_t = tok.permute(1, 0, 2)                         # [HW, T, D]
        return torch.bmm(tok_t, tok_t.transpose(1, 2))       # [HW, T, T]


# ═══════════════════════════════════════════════════════════════════════════════
# V-JEPA extractor
# ═══════════════════════════════════════════════════════════════════════════════

def build_vjepa_encoder(layers: list[int], device: torch.device):
    print(f"[V-JEPA] Loading encoder (layers={layers})…")
    enc = build_vjepa2_1_vit_large_384_encoder(
        checkpoint_path=str(VJEPA_CKPT),
        out_layers=layers,
        map_location="cpu",
    )
    return enc.eval().to(device)


def preprocess_vjepa(frames: torch.Tensor) -> torch.Tensor:
    """frames: [C, F, H, W] in [-1,1] → [1, C, 48, H', W']"""
    frames = frames[:, 1:]                                     # drop first
    frames_01 = (frames + 1.0) / 2.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    frames_norm = (frames_01 - mean) / std
    flat = frames_norm.permute(1, 0, 2, 3)                    # [F, C, H, W]
    flat = F.interpolate(flat, (VJEPA_H, VJEPA_W), mode="bicubic", align_corners=False)
    return flat.permute(1, 0, 2, 3).unsqueeze(0)              # [1, C, F, H', W']


@torch.no_grad()
def extract_vjepa(encoder, video_path: str, layers: list[int],
                  device: torch.device, out_path: Path) -> None:
    frames = load_video_frames(video_path, num_frames=49)
    video_tensor = preprocess_vjepa(frames).to(device, dtype=torch.float32)

    outs = encoder(video_tensor)   # list of [1, N, D]
    T  = VJEPA_FRAMES // TUBELET   # 24
    H_p = VJEPA_H // PATCH         # 10
    W_p = VJEPA_W // PATCH         # 15

    save: dict[str, np.ndarray] = {}
    for layer_idx, feat in zip(layers, outs):
        feat = feat.squeeze(0).reshape(T, H_p, W_p, -1).cpu()  # [T, H, W, D]
        key = f"layer_{layer_idx:02d}"
        save[f"tokens_{key}"] = feat.half().numpy()
        for mode in ("spatial", "temporal"):
            gram = compute_gram(feat.float(), mode)
            save[f"gram_{mode}_{key}"] = gram.half().numpy()

    save["layers"]    = np.array(layers, dtype=np.int32)
    save["grid_T"]    = np.array(T)
    save["grid_H"]    = np.array(H_p)
    save["grid_W"]    = np.array(W_p)
    save["embed_dim"] = np.array(1024)
    np.savez_compressed(str(out_path), **save)


# ═══════════════════════════════════════════════════════════════════════════════
# DiT-5B extractor
# ═══════════════════════════════════════════════════════════════════════════════

class BlockOutputCapture:
    def __init__(self, layer_indices: list[int]):
        self._target = set(layer_indices)
        self.captured: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def register(self, wan_model) -> None:
        for i, block in enumerate(wan_model.blocks):
            if i in self._target:
                self._handles.append(block.register_forward_hook(self._hook(i)))

    def _hook(self, idx: int):
        def fn(module, inputs, output):
            self.captured[idx] = output.detach().cpu().float()
        return fn

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        # NOTE: do NOT clear self.captured here — callers read it after remove()


def align_to_vjepa_grid(x_flat: torch.Tensor, T_p: int, H_p: int, W_p: int) -> torch.Tensor:
    D = x_flat.shape[-1]
    vol = x_flat.reshape(T_p, H_p, W_p, D).permute(3, 0, 1, 2).unsqueeze(0)
    vol = F.interpolate(vol, size=(ALIGN_T, ALIGN_H, ALIGN_W),
                        mode="trilinear", align_corners=False)
    return vol.squeeze(0).permute(1, 2, 3, 0)    # [T, H, W, D]


def build_wan_pipeline(device: torch.device):
    import wan
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel

    if not getattr(WanModel, "_codex_low_cpu_patch", False):
        _orig = WanModel.from_pretrained.__func__
        def _patched(cls, name_or_path, *a, **kw):
            kw.setdefault("low_cpu_mem_usage", False)
            return _orig(cls, name_or_path, *a, **kw)
        WanModel.from_pretrained = classmethod(_patched)
        WanModel._codex_low_cpu_patch = True

    print(f"[DiT-5B] Loading WanTI2V from {WAN_ROOT}…")
    cfg = WAN_CONFIGS["ti2v-5B"]
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(WAN_ROOT),
        device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=True, convert_model_dtype=True,
    )
    pipe.model.to(device).eval()
    return pipe


@torch.no_grad()
def extract_dit5b(pipe, video_path: str, prompt: str, layers: list[int],
                  device: torch.device, out_path: Path,
                  timestep: int = 500,
                  height: int = 480, width: int = 720) -> None:
    frames = load_video_frames(video_path, num_frames=49)
    C, F_vid, H_vid, W_vid = frames.shape
    if H_vid != height or W_vid != width:
        flat = F.interpolate(frames.permute(1, 0, 2, 3),
                             (height, width), mode="bicubic", align_corners=False)
        frames = flat.permute(1, 0, 2, 3)

    frames_dev = frames.to(device, dtype=torch.float32)
    z_list = pipe.vae.encode([frames_dev])
    z = z_list[0]                          # [C_lat, T_lat, H_lat, W_lat]
    C_lat, T_lat, H_lat, W_lat = z.shape

    t_p, h_p, w_p = WAN_PATCH_SIZE
    T_p = T_lat // t_p
    H_p = H_lat // h_p
    W_p = W_lat // w_p
    seq_len = T_p * H_p * W_p

    sigma = timestep / 1000.0
    noisy_z = (1.0 - sigma) * z + sigma * torch.randn_like(z)

    # text context
    context = pipe.text_encoder([prompt], torch.device("cpu"))
    context = [c.to(device) for c in context]

    capture = BlockOutputCapture(layers)
    capture.register(pipe.model)

    t_tensor = torch.tensor([timestep], dtype=torch.float32, device=device).expand(1, seq_len)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = pipe.model([noisy_z.to(device)], t=t_tensor, context=context, seq_len=seq_len)

    capture.remove()

    save: dict[str, np.ndarray] = {}
    for layer_idx in sorted(capture.captured):
        raw = capture.captured[layer_idx]   # [1, seq_len_padded, D]
        x_flat = raw[0, :seq_len, :]        # [seq_len, D]
        aligned = align_to_vjepa_grid(x_flat, T_p, H_p, W_p)  # [T, H, W, D]
        key = f"layer_{layer_idx:02d}"
        save[f"tokens_{key}"] = aligned.half().numpy()
        for mode in ("spatial", "temporal"):
            gram = compute_gram(aligned, mode)
            save[f"gram_{mode}_{key}"] = gram.half().numpy()

    save["layers"]     = np.array(layers, dtype=np.int32)
    save["align_T"]    = np.array(ALIGN_T)
    save["align_H"]    = np.array(ALIGN_H)
    save["align_W"]    = np.array(ALIGN_W)
    save["embed_dim"]  = np.array(WAN_HIDDEN_DIM)
    save["timestep"]   = np.array(timestep)
    save["lat_shape"]  = np.array([C_lat, T_lat, H_lat, W_lat])
    save["patch_grid"] = np.array([T_p, H_p, W_p])
    np.savez_compressed(str(out_path), **save)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/data/gaoya/agent-data/outputs/phys_compare_manifest.json")
    p.add_argument("--out-dir", default="/data/gaoya/agent-data/outputs/phys_compare/repr_cache")
    p.add_argument("--vjepa-only", action="store_true")
    p.add_argument("--dit-only",   action="store_true")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Process first N entries only (for testing)")
    p.add_argument("--timestep", type=int, default=500)
    p.add_argument("--dit-height", type=int, default=480,
                   help="Resize height for DiT VAE (smaller = faster, 480 recommended)")
    p.add_argument("--dit-width",  type=int, default=720)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vjepa-layers", nargs="+", type=int, default=VJEPA_LAYERS)
    p.add_argument("--dit-layers",   nargs="+", type=int, default=DIT_LAYERS)
    p.add_argument("--shard-id",   type=int, default=0,
                   help="Which shard to process (0-indexed)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards (processes running in parallel)")
    return p.parse_args()


def make_stem(idx: int, record: dict) -> str:
    return f"{record['source']}_{idx:04d}"


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    manifest = json.load(open(args.manifest))
    if args.max_samples:
        manifest = manifest[:args.max_samples]

    # shard: each process handles a contiguous slice of the manifest
    if args.num_shards > 1:
        shard_size = math.ceil(len(manifest) / args.num_shards)
        start = args.shard_id * shard_size
        end   = min(start + shard_size, len(manifest))
        manifest_slice = manifest[start:end]
        print(f"[shard {args.shard_id}/{args.num_shards}] processing indices {start}–{end-1} ({len(manifest_slice)} items)")
    else:
        start = 0
        manifest_slice = manifest

    # only shard 0 writes the full stem index (once, before extraction)
    if args.shard_id == 0:
        stem_map = {make_stem(i, r): r for i, r in enumerate(manifest)}
        json.dump(
            {stem: {"source": r["source"], "scenario": r["scenario"],
                    "caption": r["caption"], "video_path": r["video_path"]}
             for stem, r in stem_map.items()},
            open(os.path.join(args.out_dir, "stem_index.json"), "w"),
            indent=2, ensure_ascii=False,
        )

    # ── V-JEPA pass ───────────────────────────────────────────────────────────
    if not args.dit_only:
        encoder = build_vjepa_encoder(args.vjepa_layers, device)
        for i, rec in enumerate(manifest_slice, start=start):
            stem = make_stem(i, rec)
            out_path = Path(args.out_dir) / f"{stem}_vjepa.npz"
            if out_path.exists():
                print(f"[V-JEPA] skip {stem} (cached)")
                continue
            print(f"[V-JEPA] {i+1}/{len(manifest)}  {stem}  {Path(rec['video_path']).name}")
            try:
                extract_vjepa(encoder, rec["video_path"], args.vjepa_layers,
                              device, out_path)
            except Exception as e:
                print(f"  WARN: {e}")
        del encoder
        torch.cuda.empty_cache()

    # ── DiT-5B pass ───────────────────────────────────────────────────────────
    if not args.vjepa_only:
        pipe = build_wan_pipeline(device)
        for i, rec in enumerate(manifest_slice, start=start):
            stem = make_stem(i, rec)
            out_path = Path(args.out_dir) / f"{stem}_dit5b.npz"
            if out_path.exists():
                print(f"[DiT-5B] skip {stem} (cached)")
                continue
            print(f"[DiT-5B] {i+1}/{len(manifest)}  {stem}  {Path(rec['video_path']).name}")
            try:
                extract_dit5b(pipe, rec["video_path"], rec.get("caption", ""),
                              args.dit_layers, device, out_path,
                              timestep=args.timestep,
                              height=args.dit_height, width=args.dit_width)
            except Exception as e:
                print(f"  WARN: {e}")
        del pipe
        torch.cuda.empty_cache()

    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
