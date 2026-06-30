"""
Visualise the per-layer token relation shape extracted by V-JEPA 2.1.

For every requested layer the script produces three panels:
  (a) Spatial relation matrix   – time-averaged cosine similarity [H·W × H·W]
  (b) Temporal relation matrix  – spatially-averaged cosine similarity [T × T]
  (c) Centre-token affinity map – cosine similarity of the centre token to
                                  every spatial token, averaged over time

Usage – features already extracted by extract_vjepa_features.py:
    python vis_vjepa_token_relations.py \
        --features-dir /data/gaoya/.../out_dir \
        --layers 5 11 17 23 \
        --output-dir ./vis_out

Usage – extract on the fly:
    python vis_vjepa_token_relations.py \
        --video /tmp/gaoya/physics_iq_single_case/sample_001460/scored_source_video.mp4 \
        --layers 5 11 17 23 \
        --output-dir ./vis_out
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

VJEPA2_REPO = Path("/home/gaoya/Code_Video/vjepa2-main")
VJEPA2_1_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")

# Default token grid for ViT-L 384, 64 frames, tubelet=2, patch=16
DEFAULT_T_TOK = 32
DEFAULT_H_TOK = 24
DEFAULT_W_TOK = 24


def _add_repo(repo: Path) -> None:
    s = str(repo)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Feature I/O
# ---------------------------------------------------------------------------

def load_features_from_dir(features_dir: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    """Load pre-extracted .pt files (shape [1, N, D]) from an extraction dir."""
    result = {}
    for layer in layers:
        pt = features_dir / f"layer_{layer}.pt"
        if not pt.exists():
            raise FileNotFoundError(f"Missing feature file: {pt}")
        result[layer] = torch.load(pt, map_location="cpu")
    return result


def extract_features_from_video(
    video_path: Path,
    layers: list[int],
    target_frames: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Run V-JEPA encoder inline and return per-layer token tensors [1, N, D]."""
    _add_repo(VJEPA2_REPO)
    from local_vjepa21_backbone import build_vjepa2_1_vit_large_384_encoder
    from evals.hub.preprocessor import vjepa2_preprocessor
    from decord import VideoReader

    vr = VideoReader(str(video_path))
    n = len(vr)
    idx = np.linspace(0, n - 1, min(target_frames, n), dtype=int)
    frames = vr.get_batch(idx).asnumpy()

    processor = vjepa2_preprocessor(crop_size=384)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)
    x = processor(video)[0].to(device).unsqueeze(0)

    encoder = build_vjepa2_1_vit_large_384_encoder(
        checkpoint_path=str(VJEPA2_1_CKPT),
        map_location="cpu",
        out_layers=layers,
    ).to(device).eval()

    with torch.inference_mode():
        layer_outputs = encoder(x)

    return {layer: tok.detach().cpu() for layer, tok in zip(layers, layer_outputs)}


# ---------------------------------------------------------------------------
# Relation computations
# ---------------------------------------------------------------------------

def cosine_sim_matrix(X: torch.Tensor) -> torch.Tensor:
    """X: [N, D] → [N, N] cosine similarity matrix in [-1, 1]."""
    X = F.normalize(X.float(), dim=-1)
    return (X @ X.T).clamp(-1.0, 1.0)


def spatial_relation(tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
    """
    Time-averaged cosine similarity between spatial token positions.
    tokens: [N, D] (N = T*H*W) → returns [H*W, H*W].
    """
    tok = tokens.view(T, H * W, -1).mean(dim=0)  # [HW, D]
    return cosine_sim_matrix(tok)


def temporal_relation(tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
    """
    Space-averaged cosine similarity between temporal token positions.
    tokens: [N, D] → returns [T, T].
    """
    tok = tokens.view(T, H * W, -1).mean(dim=1)  # [T, D]
    return cosine_sim_matrix(tok)


def centre_affinity(tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
    """
    Cosine similarity from the spatial centre token to every spatial position,
    averaged over time.  tokens: [N, D] → returns [H, W].
    """
    tok = tokens.view(T, H, W, -1)                         # [T, H, W, D]
    ref = tok[:, H // 2, W // 2, :]                        # [T, D]
    ref_n = F.normalize(ref.float(), dim=-1)               # [T, D]
    tok_n = F.normalize(tok.float(), dim=-1)               # [T, H, W, D]
    sim = (tok_n * ref_n[:, None, None, :]).sum(dim=-1)    # [T, H, W]
    return sim.mean(dim=0)                                  # [H, W]


def query_affinity(
    tokens: torch.Tensor, T: int, H: int, W: int, qh: int, qw: int
) -> torch.Tensor:
    """
    Same as centre_affinity but for an arbitrary query position (qh, qw).
    Returns [H, W].
    """
    tok = tokens.view(T, H, W, -1)
    ref_n = F.normalize(tok[:, qh, qw, :].float(), dim=-1)   # [T, D]
    tok_n = F.normalize(tok.float(), dim=-1)
    sim = (tok_n * ref_n[:, None, None, :]).sum(dim=-1)
    return sim.mean(dim=0)


# ---------------------------------------------------------------------------
# Per-layer feature statistics (for title annotation)
# ---------------------------------------------------------------------------

def _layer_stats(tokens: torch.Tensor) -> str:
    x = tokens.float()
    return f"mean={x.mean():.3f}  std={x.std():.3f}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CMAP_REL = "RdYlBu_r"
CMAP_AFF = "hot"


def _imshow_cb(ax, data, vmin, vmax, cmap, title, xlabel, ylabel, fig):
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def make_figure(
    layer_features: dict[int, torch.Tensor],
    T: int,
    H: int,
    W: int,
    title_prefix: str = "",
    query_positions: list[tuple[int, int]] | None = None,
) -> plt.Figure:
    """
    Build a multi-row figure.  Each row = one layer, three panels:
      col 0: spatial relation matrix [H·W × H·W]
      col 1: temporal relation matrix [T × T]
      col 2: centre-token affinity map [H × W]
    If query_positions is given, an extra col per query is added.
    """
    layers = sorted(layer_features.keys())
    n_layers = len(layers)
    n_cols = 3 + (len(query_positions) if query_positions else 0)

    fig, axes = plt.subplots(
        n_layers, n_cols,
        figsize=(5 * n_cols, 4.5 * n_layers),
        squeeze=False,
    )
    fig.suptitle(
        f"{title_prefix}V-JEPA Token Relation Shape  "
        f"(T_tok={T}  H_tok={H}  W_tok={W})",
        fontsize=11,
    )

    for row, layer_idx in enumerate(layers):
        tokens = layer_features[layer_idx].squeeze(0)   # [N, D]
        stats = _layer_stats(tokens)

        sp = spatial_relation(tokens, T, H, W).numpy()
        te = temporal_relation(tokens, T, H, W).numpy()
        ca = centre_affinity(tokens, T, H, W).numpy()

        _imshow_cb(
            axes[row, 0], sp, -1, 1, CMAP_REL,
            f"Layer {layer_idx}  spatial rel [H·W × H·W]\n{stats}",
            "spatial token idx", "spatial token idx", fig,
        )
        _imshow_cb(
            axes[row, 1], te, -1, 1, CMAP_REL,
            f"Layer {layer_idx}  temporal rel [T × T]",
            "time token", "time token", fig,
        )

        ax_ca = axes[row, 2]
        im = ax_ca.imshow(ca, vmin=-1, vmax=1, cmap=CMAP_AFF, aspect="equal", interpolation="nearest")
        ax_ca.set_title(f"Layer {layer_idx}  centre affinity [H × W]", fontsize=8)
        ax_ca.set_xlabel("W patches", fontsize=7)
        ax_ca.set_ylabel("H patches", fontsize=7)
        ax_ca.tick_params(labelsize=6)
        ax_ca.scatter([W // 2], [H // 2], marker="x", color="cyan", s=60, linewidths=1.5)
        fig.colorbar(im, ax=ax_ca, fraction=0.046, pad=0.04)

        for col_offset, (qh, qw) in enumerate(query_positions or []):
            qa = query_affinity(tokens, T, H, W, qh, qw).numpy()
            ax_q = axes[row, 3 + col_offset]
            im_q = ax_q.imshow(qa, vmin=-1, vmax=1, cmap=CMAP_AFF, aspect="equal", interpolation="nearest")
            ax_q.set_title(f"Layer {layer_idx}  query ({qh},{qw}) affinity", fontsize=8)
            ax_q.set_xlabel("W patches", fontsize=7)
            ax_q.set_ylabel("H patches", fontsize=7)
            ax_q.tick_params(labelsize=6)
            ax_q.scatter([qw], [qh], marker="x", color="lime", s=60, linewidths=1.5)
            fig.colorbar(im_q, ax=ax_q, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise per-layer V-JEPA 2.1 token relation heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--features-dir", type=Path,
        help="Dir produced by extract_vjepa_features.py (contains layer_N.pt files).",
    )
    src.add_argument(
        "--video", type=Path,
        help="Input video path — encoder will be run inline.",
    )
    p.add_argument(
        "--layers", nargs="+", type=int,
        default=[5, 11, 17, 23],
        help="Layer indices to visualise.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("vis_token_relations"))
    p.add_argument("--output-name", default="vjepa_token_relations.png")
    p.add_argument("--target-frames", type=int, default=64)
    p.add_argument(
        "--t-tok", type=int, default=None,
        help="Temporal token count = num_frames / tubelet_size. Auto-inferred if omitted.",
    )
    p.add_argument("--h-tok", type=int, default=DEFAULT_H_TOK)
    p.add_argument("--w-tok", type=int, default=DEFAULT_W_TOK)
    p.add_argument(
        "--query", nargs=2, type=int, action="append", metavar=("H", "W"),
        help="Extra query position(s) for affinity maps, e.g. --query 6 6.",
    )
    p.add_argument("--cuda", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.features_dir:
        features_dir = args.features_dir.expanduser().resolve()
        print(f"Loading features from {features_dir}")
        layer_features = load_features_from_dir(features_dir, args.layers)
        title_prefix = f"{features_dir.name} — "
    else:
        video_path = args.video.expanduser().resolve()
        print(f"Extracting features from {video_path} …")
        layer_features = extract_features_from_video(
            video_path, args.layers, args.target_frames, device
        )
        title_prefix = f"{video_path.stem} — "

    # Infer / verify token grid
    sample_tok = next(iter(layer_features.values())).squeeze(0)
    N = sample_tok.shape[0]
    H, W = args.h_tok, args.w_tok
    if args.t_tok is not None:
        T = args.t_tok
        if T * H * W != N:
            raise ValueError(
                f"Token grid mismatch: T={T} H={H} W={W} → {T*H*W} tokens, "
                f"but tensor has N={N}. Adjust --t-tok / --h-tok / --w-tok."
            )
    else:
        if N % (H * W) != 0:
            raise ValueError(
                f"Cannot auto-infer T: N={N} is not divisible by H·W={H*W}. "
                f"Set --t-tok explicitly."
            )
        T = N // (H * W)
        print(f"Auto-inferred T_tok={T} (N={N}, H={H}, W={W})")

    query_positions = [tuple(q) for q in args.query] if args.query else None

    print(f"Computing relations for layers {sorted(layer_features.keys())} …")
    fig = make_figure(
        layer_features, T, H, W,
        title_prefix=title_prefix,
        query_positions=query_positions,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
