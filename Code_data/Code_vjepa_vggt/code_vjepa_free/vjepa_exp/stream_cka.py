"""
流式 CKA：对每个视频跑一次前向，即时计算 gram → CKA，累加到 running sum。
无任何 token/gram 缓存写盘，磁盘占用仅输出图表（~几百 KB）。

用法（wan env）：
    PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
    python stream_cka.py \
        --manifest /data/gaoya/agent-data/outputs/phys_compare_manifest.json \
        --out-dir  /data/gaoya/agent-data/outputs/phys_compare/cka_stream \
        [--max-samples N] [--timestep 500]
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR     = Path(__file__).parent.resolve()
VJEPA2_ROOT = Path("/home/gaoya/Code_Video/vjepa2-main")
WAN_REPO    = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
VJEPA_CKPT  = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")
WAN_ROOT    = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")

sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(VJEPA2_ROOT))
sys.path.insert(0, str(WAN_REPO))

from local_vjepa21_backbone import build_vjepa2_1_vit_large_384_encoder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VJEPA_FRAMES   = 48
VJEPA_H, VJEPA_W = 160, 240
PATCH, TUBELET = 16, 2

WAN_PATCH_SIZE = (1, 2, 2)
WAN_VAE_STRIDE = (4, 16, 16)
WAN_NUM_LAYERS = 30
WAN_HIDDEN_DIM = 3072

# V-JEPA spatial grid dimensions — set dynamically after first forward pass
# H_p = VJEPA_H // PATCH = 10,  W_p = VJEPA_W // PATCH = 15
# T_p is determined by the encoder (depends on tubelet stride and any internal padding)
ALIGN_H = VJEPA_H // PATCH   # 10  — fixed by spatial patch size
ALIGN_W = VJEPA_W // PATCH   # 15  — fixed by spatial patch size
ALIGN_T: int | None = None    # resolved on first vjepa_grams() call

VJEPA_LAYERS = [0, 3, 5, 8, 11, 14, 17, 20, 23]
DIT_LAYERS   = list(range(30))

DATASET_ORDER = ["phyco_kubric", "pybullet", "physics-iq"]
COLORS = {"phyco_kubric": "#1f77b4", "pybullet": "#ff7f0e", "physics-iq": "#2ca02c"}


# ── video loading ──────────────────────────────────────────────────────────────

def load_video_frames(video_path: str, num_frames: int = 49,
                      first_frames: bool = False) -> torch.Tensor:
    """Return [C, F, H, W] float32 in [-1, 1].

    first_frames=True: take the first num_frames frames (no resampling).
    first_frames=False: uniformly sample num_frames across the full video.
    """
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        if first_frames:
            indices = np.arange(min(num_frames, total))
        else:
            indices = np.linspace(0, total - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).float()
        return frames.permute(3, 0, 1, 2) / 127.5 - 1.0
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if first_frames:
        idx_set = set(range(min(num_frames, total)))
    else:
        idx_set = set(np.linspace(0, total - 1, num_frames).astype(int).tolist())
    frames_list, i = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frames_list.append(torch.from_numpy(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).float())
        i += 1
    cap.release()
    return torch.stack(frames_list[:num_frames]).permute(3, 0, 1, 2) / 127.5 - 1.0


# ── gram + CKA ─────────────────────────────────────────────────────────────────

def gram_aligned(grid: torch.Tensor) -> np.ndarray:
    """grid: [T, H, W, D] on the common spatiotemporal grid → gram [T*H*W, T*H*W].

    Tokens are L2-normalised before computing the gram matrix.
    All T*H*W = ALIGN_N_TOK tokens are used (no subsampling).
    """
    T, H, W, D = grid.shape
    tok = F.normalize(grid.reshape(T * H * W, D).float(), dim=-1)  # [N, D]
    gram = tok @ tok.T                                              # [N, N]
    return gram.cpu().numpy().astype(np.float64)


def cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    num   = float(np.sum(Kc * Lc))
    denom = float(np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc)))
    return num / (denom + 1e-12)


# ── V-JEPA forward ─────────────────────────────────────────────────────────────

def preprocess_vjepa(frames: torch.Tensor) -> torch.Tensor:
    frames = frames[:, 1:]   # drop first frame
    frames_01 = (frames + 1.0) / 2.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    frames_norm = (frames_01 - mean) / std
    flat = frames_norm.permute(1, 0, 2, 3)
    flat = F.interpolate(flat, (VJEPA_H, VJEPA_W), mode="bicubic", align_corners=False)
    return flat.permute(1, 0, 2, 3).unsqueeze(0)   # [1, C, F, H', W']


@torch.no_grad()
def vjepa_grams(encoder, frames: torch.Tensor, device: torch.device) -> dict[int, np.ndarray]:
    """Returns {layer_idx: gram_aligned [N, N]}.  N = ALIGN_T * ALIGN_H * ALIGN_W."""
    global ALIGN_T
    video_tensor = preprocess_vjepa(frames).to(device, dtype=torch.float32)
    outs = encoder(video_tensor)   # list of [1, N, D]
    if ALIGN_T is None:
        # resolve actual temporal token count from the encoder output
        N_actual = outs[0].shape[1]   # N = T_p * H_p * W_p
        ALIGN_T = N_actual // (ALIGN_H * ALIGN_W)
        print(f"[V-JEPA] resolved grid: T={ALIGN_T} H={ALIGN_H} W={ALIGN_W} → N={ALIGN_T*ALIGN_H*ALIGN_W}", flush=True)
    result = {}
    for layer_idx, feat in zip(VJEPA_LAYERS, outs):
        grid = feat.squeeze(0).reshape(ALIGN_T, ALIGN_H, ALIGN_W, -1)  # [T, H, W, D]
        result[layer_idx] = gram_aligned(grid)
    return result


# ── DiT-5B forward ─────────────────────────────────────────────────────────────

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


def align_to_vjepa_grid(x_flat: torch.Tensor, T_p: int, H_p: int, W_p: int) -> torch.Tensor:
    """Trilinear-interpolate DiT tokens to V-JEPA grid [ALIGN_T, ALIGN_H, ALIGN_W, D].

    x_flat: [T_p*H_p*W_p, D]  (DiT patch-token sequence, spatial order T,H,W)
    ALIGN_T must be resolved before this is called (happens on first vjepa_grams()).
    Returns: [ALIGN_T, ALIGN_H, ALIGN_W, D]
    """
    assert ALIGN_T is not None, "call vjepa_grams() before dit_grams() to resolve ALIGN_T"
    D = x_flat.shape[-1]
    vol = x_flat.reshape(T_p, H_p, W_p, D).permute(3, 0, 1, 2).unsqueeze(0)
    vol = F.interpolate(vol, size=(ALIGN_T, ALIGN_H, ALIGN_W),
                        mode="trilinear", align_corners=False)
    return vol.squeeze(0).permute(1, 2, 3, 0)   # [ALIGN_T, ALIGN_H, ALIGN_W, D]


@torch.no_grad()
def dit_grams(pipe, frames: torch.Tensor, prompt: str,
              device: torch.device, timesteps: list[int],
              height: int, width: int) -> dict[int, dict[int, np.ndarray]]:
    """Returns {timestep: {layer_idx: gram_aligned [ALIGN_N_TOK, ALIGN_N_TOK]}}.

    DiT tokens are trilinearly interpolated to V-JEPA grid [ALIGN_T, ALIGN_H, ALIGN_W]
    before computing the gram, so both models share identical token coordinates.
    VAE encode is done once; DiT forward is run once per timestep.
    """
    C, F_vid, H_vid, W_vid = frames.shape
    if H_vid != height or W_vid != width:
        flat = F.interpolate(frames.permute(1, 0, 2, 3),
                             (height, width), mode="bicubic", align_corners=False)
        frames = flat.permute(1, 0, 2, 3)

    frames_dev = frames.to(device, dtype=torch.float32)
    z = pipe.vae.encode([frames_dev])[0]
    C_lat, T_lat, H_lat, W_lat = z.shape

    t_p, h_p, w_p = WAN_PATCH_SIZE
    T_p = T_lat // t_p
    H_p = H_lat // h_p
    W_p = W_lat // w_p
    seq_len = T_p * H_p * W_p

    context = pipe.text_encoder([prompt], torch.device("cpu"))
    context = [c.to(device) for c in context]

    result: dict[int, dict[int, np.ndarray]] = {}
    for timestep in timesteps:
        sigma = timestep / 1000.0
        torch.manual_seed(0)
        noisy_z = (1.0 - sigma) * z + sigma * torch.randn_like(z)

        capture = BlockOutputCapture(DIT_LAYERS)
        capture.register(pipe.model)
        t_tensor = torch.tensor([timestep], dtype=torch.float32, device=device).expand(1, seq_len)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = pipe.model([noisy_z.to(device)], t=t_tensor, context=context, seq_len=seq_len)
        capture.remove()

        ts_result: dict[int, np.ndarray] = {}
        for layer_idx in sorted(capture.captured):
            raw = capture.captured[layer_idx]
            x_flat = raw[0, :seq_len, :]                              # [T_p*H_p*W_p, D]
            grid = align_to_vjepa_grid(x_flat, T_p, H_p, W_p)        # [ALIGN_T, ALIGN_H, ALIGN_W, D]
            ts_result[layer_idx] = gram_aligned(grid)
        result[timestep] = ts_result
    return result


# ── running accumulators ───────────────────────────────────────────────────────

class CKAAccumulator:
    """Maintains a running sum of per-video CKA grids per dataset.

    Grid shape: [n_vj, n_dt, T] where T = number of timesteps.
    Legacy [n_vj, n_dt] files (T=1) are expanded automatically on load.
    """
    def __init__(self):
        self.sum:   dict[str, np.ndarray] = {}
        self.count: dict[str, int] = defaultdict(int)
        self.vj_layers:  list[int] | None = None
        self.dt_layers:  list[int] | None = None
        self.timesteps:  list[int] | None = None

    def update(self, dataset: str,
               vj_grams: dict[int, np.ndarray],
               dt_grams_multi: dict[int, dict[int, np.ndarray]],
               precomputed_grid: np.ndarray | None = None) -> np.ndarray:
        """Returns [n_vj, n_dt, T] grid for this video."""
        if precomputed_grid is not None:
            grid = precomputed_grid.astype(np.float32)
            if grid.ndim == 2:              # legacy [9,30] → [9,30,1]
                grid = grid[:, :, np.newaxis]
            if self.vj_layers is None:
                self.vj_layers = VJEPA_LAYERS
                self.dt_layers = DIT_LAYERS
                self.timesteps = [500]
        else:
            vj_layers  = sorted(vj_grams)
            timesteps  = sorted(dt_grams_multi)
            dt_layers  = sorted(dt_grams_multi[timesteps[0]])
            if self.vj_layers is None:
                self.vj_layers = vj_layers
                self.dt_layers = dt_layers
                self.timesteps = timesteps

            grid = np.zeros((len(vj_layers), len(dt_layers), len(timesteps)),
                            dtype=np.float32)
            for vi, vl in enumerate(vj_layers):
                Kv = vj_grams[vl]
                for di, dl in enumerate(dt_layers):
                    for ti, ts in enumerate(timesteps):
                        Kd = dt_grams_multi[ts][dl]
                        n = min(Kv.shape[0], Kd.shape[0])
                        grid[vi, di, ti] = cka_from_grams(Kv[:n, :n], Kd[:n, :n])

        if dataset not in self.sum:
            self.sum[dataset] = grid.copy()
        else:
            self.sum[dataset] += grid
        self.count[dataset] += 1
        return grid

    def means(self) -> dict[str, np.ndarray]:
        return {ds: self.sum[ds] / self.count[ds] for ds in self.sum}


# ── model loaders ──────────────────────────────────────────────────────────────

def build_vjepa_encoder(device: torch.device):
    print(f"[V-JEPA] Loading encoder…")
    from local_vjepa21_backbone import build_vjepa2_1_vit_large_384_encoder
    enc = build_vjepa2_1_vit_large_384_encoder(
        checkpoint_path=str(VJEPA_CKPT),
        out_layers=VJEPA_LAYERS,
        map_location="cpu",
    )
    return enc.eval().to(device)


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

    print(f"[DiT-5B] Loading WanTI2V…")
    cfg = WAN_CONFIGS["ti2v-5B"]
    pipe = wan.WanTI2V(
        config=cfg, checkpoint_dir=str(WAN_ROOT),
        device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False,
        t5_cpu=True, convert_model_dtype=True,
    )
    pipe.model.to(device).eval()
    return pipe


# ── plotting ───────────────────────────────────────────────────────────────────

def save_plots(dataset_grids: dict[str, np.ndarray],
               vj_layers: list[int], dt_layers: list[int],
               timesteps: list[int],
               out_dir: Path, suffix: str = "") -> None:
    """dataset_grids values are [n_vj, n_dt, T]; plots collapse over T."""
    datasets = [d for d in DATASET_ORDER if d in dataset_grids]
    if not datasets:
        datasets = list(dataset_grids.keys())
    if not datasets:
        return

    # collapse over timesteps for legacy-style plots
    grids_t = {ds: g.mean(axis=2) for ds, g in dataset_grids.items()}   # [9,30]
    vmax = max(g.max() for g in grids_t.values())

    # per-dataset heatmaps (mean over timesteps)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        g = grids_t[ds]
        im = ax.imshow(g, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"{ds}\nmax={g.max():.3f}", fontsize=10)
        ax.set_xlabel("DiT-5B layer"); ax.set_ylabel("V-JEPA layer")
        ax.set_xticks(range(len(dt_layers))); ax.set_xticklabels(dt_layers, fontsize=7, rotation=45)
        ax.set_yticks(range(len(vj_layers))); ax.set_yticklabels(vj_layers, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("V-JEPA vs Wan DiT-5B — Linear CKA (mean over timesteps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"cka_per_dataset{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"Saved cka_per_dataset{suffix}.png")

    # layer curves
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds in datasets:
        ax.plot(vj_layers, grids_t[ds].mean(axis=1),
                label=ds, color=COLORS.get(ds), marker="o", markersize=4)
    ax.set_xlabel("V-JEPA layer"); ax.set_ylabel("Mean CKA (over DiT layers & timesteps)")
    ax.set_title("V-JEPA–DiT CKA vs V-JEPA depth")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"cka_per_layer_curve{suffix}.png"), dpi=150)
    plt.close(fig)
    print(f"Saved cka_per_layer_curve{suffix}.png")

    # per-timestep heatmaps (first dataset only, or GT if present)
    ref_ds = next((d for d in ["pybullet_test100", datasets[0]] if d in dataset_grids), datasets[0])
    if len(timesteps) > 1:
        g_ts = dataset_grids[ref_ds]   # [9, 30, T]
        fig, axes = plt.subplots(1, len(timesteps), figsize=(5*len(timesteps), 4), sharey=True)
        if len(timesteps) == 1:
            axes = [axes]
        vmax_ts = g_ts.max()
        for ax, (ti, ts) in zip(axes, enumerate(timesteps)):
            im = ax.imshow(g_ts[:, :, ti], aspect="auto", origin="lower",
                           vmin=0, vmax=vmax_ts, cmap="viridis")
            ax.set_title(f"τ={ts}", fontsize=10)
            ax.set_xlabel("DiT-5B layer")
            ax.set_xticks(range(len(dt_layers))); ax.set_xticklabels(dt_layers, fontsize=6, rotation=45)
            ax.set_yticks(range(len(vj_layers))); ax.set_yticklabels(vj_layers, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"CKA per timestep — {ref_ds}", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(out_dir / f"cka_per_timestep{suffix}.png"), dpi=150)
        plt.close(fig)
        print(f"Saved cka_per_timestep{suffix}.png")

    # save matrices
    save_dict = {ds.replace("-", "_"): g for ds, g in dataset_grids.items()}
    save_dict["vj_layers"]  = np.array(vj_layers)
    save_dict["dt_layers"]  = np.array(dt_layers)
    save_dict["timesteps"]  = np.array(timesteps)
    np.savez_compressed(str(out_dir / f"cka_matrices{suffix}.npz"), **save_dict)
    print(f"Saved cka_matrices{suffix}.npz")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None,
                   help="JSON manifest with {video_path, source, caption} entries")
    p.add_argument("--video-dir", default=None,
                   help="Recursively find all video.mp4 under this dir (dataset name = dir name)")
    p.add_argument("--dataset-name", default="pybullet",
                   help="Dataset label used when --video-dir is given")
    p.add_argument("--out-dir",  default="/data/gaoya/agent-data/outputs/phys_compare/cka_stream")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--shard-id",   type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--timestep",  type=int, default=None,
                   help="Single timestep (legacy alias; overrides --timesteps)")
    p.add_argument("--timesteps", type=int, nargs="+", default=[500],
                   help="One or more DiT noise timesteps, e.g. --timesteps 100 300 500 700 900")
    p.add_argument("--num-frames", type=int, default=49,
                   help="Number of frames to load per video")
    p.add_argument("--dit-height", type=int, default=480)
    p.add_argument("--dit-width",  type=int, default=720)
    p.add_argument("--case-dir", default="/data/gaoya/AAA_test_video/0626vjepa_free/GT_check/0613pybullet",
                   help="Per-case [9,30] CKA grids saved as <sample_name>.npy here")
    p.add_argument("--device", default="cuda")
    p.add_argument("--first-frames", action="store_true",
                   help="Take the first num_frames frames instead of uniform sampling")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.case_dir = Path(args.case_dir)
    args.case_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # build record list
    if args.video_dir:
        video_files = sorted(Path(args.video_dir).rglob("video.mp4"))
        records = [{"video_path": str(f), "source": args.dataset_name, "caption": ""}
                   for f in video_files]
    else:
        records = json.load(open(args.manifest))

    if args.max_samples:
        records = records[:args.max_samples]

    # shard
    import math
    if args.num_shards > 1:
        shard_size = math.ceil(len(records) / args.num_shards)
        start = args.shard_id * shard_size
        records = records[start: start + shard_size]
        print(f"[shard {args.shard_id}/{args.num_shards}] {len(records)} videos", flush=True)

    print(f"Processing {len(records)} videos (no disk cache)")

    encoder = build_vjepa_encoder(device)
    pipe    = build_wan_pipeline(device)

    # resolve timesteps
    timesteps = [args.timestep] if args.timestep is not None else args.timesteps
    timesteps = sorted(set(timesteps))
    print(f"Timesteps: {timesteps}", flush=True)

    # write timesteps marker so analysis scripts can read it
    np.save(str(args.case_dir / "timesteps.npy"), np.array(timesteps))

    accum = CKAAccumulator()

    for i, rec in enumerate(records):
        video_path = rec["video_path"]
        dataset    = rec["source"]
        prompt     = rec.get("caption", "")
        sample_name = rec.get("sample_name") or Path(video_path).parent.name
        print(f"  [{i+1}/{len(records)}] {dataset} — {sample_name}", flush=True)
        try:
            out_file = args.case_dir / f"{sample_name}.npy"
            if out_file.exists():
                print(f"    skip (exists)", flush=True)
                grid = np.load(str(out_file))
                accum.update(dataset, {}, {}, precomputed_grid=grid)
                continue

            frames = load_video_frames(video_path, num_frames=args.num_frames,
                                       first_frames=args.first_frames)
            vj = vjepa_grams(encoder, frames, device)
            dt = dit_grams(pipe, frames, prompt, device,
                           timesteps=timesteps,
                           height=args.dit_height, width=args.dit_width)
            grid = accum.update(dataset, vj, dt)
            np.save(str(out_file), grid)
        except Exception as e:
            print(f"    WARN: {e}", flush=True)

    dataset_grids = accum.means()
    print(f"\nDataset counts: { {ds: accum.count[ds] for ds in accum.count} }")

    # save raw sums + counts so shards can be merged later
    shard_tag = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    np.savez_compressed(
        str(out_dir / f"cka_sums{shard_tag}.npz"),
        **{f"{ds.replace('-','_')}_sum":   accum.sum[ds]   for ds in accum.sum},
        **{f"{ds.replace('-','_')}_count": np.array(accum.count[ds]) for ds in accum.count},
        vj_layers=np.array(accum.vj_layers),
        dt_layers=np.array(accum.dt_layers),
        timesteps=np.array(accum.timesteps),
    )
    print(f"Saved cka_sums{shard_tag}.npz")

    save_plots(dataset_grids, accum.vj_layers, accum.dt_layers,
               accum.timesteps, out_dir, suffix=shard_tag)

    print("\n=== Summary: max CKA per dataset ===")
    for ds in DATASET_ORDER:
        if ds not in dataset_grids:
            continue
        g = dataset_grids[ds]
        vi, di = np.unravel_index(np.argmax(g), g.shape)
        print(f"  {ds:<20} max={g.max():.4f}"
              f"  at V-JEPA L{accum.vj_layers[vi]} / DiT L{accum.dt_layers[di]}"
              f"  (n={accum.count[ds]})")


if __name__ == "__main__":
    main()
