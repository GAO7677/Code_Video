"""
Analyze correlation between V-JEPA and Wan2.2 TI2V-5B DiT representations.

Loads cached .npz files from extract_vjepa_gram.py and extract_dit_repr_5b.py,
then computes:
  1. Linear CKA (N_vjepa × 30 heatmap)
  2. Gram matrix Pearson correlation – spatial and temporal separately

Outputs (in --out-dir):
  cka_heatmap.png          CKA between every (vjepa_layer, dit_layer) pair
  gram_corr_spatial.png    Spatial Gram Pearson r per layer pair
  gram_corr_temporal.png   Temporal Gram Pearson r per layer pair
  best_pair_viz.png        Gram matrix visualisation for top CKA pair

Usage:
    python analyze_vjepa_dit_corr.py \
        --repr-dir /data/gaoya/agent-data/outputs/vjepa_wan_precheck/repr_cache \
        --video-stem <video_stem> \
        --out-dir /data/gaoya/agent-data/outputs/vjepa_wan_precheck/corr_vis
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Linear CKA ────────────────────────────────────────────────────────────────

def gram_linear(X: np.ndarray) -> np.ndarray:
    """X: [N, D] → N×N linear kernel matrix."""
    return X @ X.T


def center_gram(K: np.ndarray) -> np.ndarray:
    """Centering of a Gram (kernel) matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices [N, D_x] and [N, D_y].
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    Numerically stable via Frobenius norms of centred Gram matrices.
    """
    K = center_gram(gram_linear(X))
    L = center_gram(gram_linear(Y))
    num = np.sum(K * L)
    denom = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return float(num / (denom + 1e-12))


# ── Gram Pearson ───────────────────────────────────────────────────────────────

def pearson_r_flat(A: np.ndarray, B: np.ndarray) -> float:
    """Flatten both arrays and compute Pearson r."""
    a = A.flatten().astype(np.float32)
    b = B.flatten().astype(np.float32)
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / (denom + 1e-12))


def gram_pearson(gram_v: np.ndarray, gram_d: np.ndarray, mode: str) -> float:
    """
    gram_v, gram_d shapes:
      spatial:  [T, HW, HW]  → average over T first
      temporal: [HW, T, T]   → average over HW first
    Returns scalar Pearson r between the average Gram matrices.
    """
    if mode == "spatial":
        g_v = gram_v.astype(np.float32).mean(axis=0)   # [HW, HW]
        g_d = gram_d.astype(np.float32).mean(axis=0)
    else:
        g_v = gram_v.astype(np.float32).mean(axis=0)   # [T, T]
        g_d = gram_d.astype(np.float32).mean(axis=0)
    return pearson_r_flat(g_v, g_d)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repr-dir", default="./repr_cache")
    parser.add_argument("--video-stem", required=True,
                        help="stem used when saving .npz files, e.g. 'scored_source_video'")
    parser.add_argument("--out-dir", default="./corr_vis")
    parser.add_argument("--cka-subsample", type=int, default=2000,
                        help="max tokens per layer used in CKA (random subsample for speed)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vjepa_path = Path(args.repr_dir) / f"{args.video_stem}_vjepa.npz"
    # prefer 5B cache; fall back to legacy 1.3B cache for backwards compat
    dit_path_5b   = Path(args.repr_dir) / f"{args.video_stem}_dit5b.npz"
    dit_path_legacy = Path(args.repr_dir) / f"{args.video_stem}_dit.npz"
    dit_path = dit_path_5b if dit_path_5b.exists() else dit_path_legacy
    print(f"Loading {vjepa_path}")
    vjepa = np.load(str(vjepa_path), allow_pickle=False)
    print(f"Loading {dit_path}")
    dit   = np.load(str(dit_path),   allow_pickle=False)

    vjepa_layers = sorted(vjepa["layers"].tolist())
    dit_layers   = sorted(dit["layers"].tolist())
    print(f"V-JEPA layers: {len(vjepa_layers)}  DiT layers: {len(dit_layers)}")

    # ── CKA grid ──────────────────────────────────────────────────────────────
    print("Computing CKA…")
    cka_grid = np.zeros((len(vjepa_layers), len(dit_layers)), dtype=np.float32)
    rng = np.random.default_rng(0)

    for vi, vl in enumerate(vjepa_layers):
        v_tokens = vjepa[f"tokens_layer_{vl:02d}"].astype(np.float32)  # [T,H,W,D]
        T, H, W, D = v_tokens.shape
        v_flat = v_tokens.reshape(-1, D)   # [T*H*W, D]
        if v_flat.shape[0] > args.cka_subsample:
            idx = rng.choice(v_flat.shape[0], args.cka_subsample, replace=False)
            v_flat = v_flat[idx]

        for di, dl in enumerate(dit_layers):
            d_tokens = dit[f"tokens_layer_{dl:02d}"].astype(np.float32)   # [T,H,W,D]
            d_flat = d_tokens.reshape(-1, d_tokens.shape[-1])
            if d_flat.shape[0] > args.cka_subsample:
                # use same spatial indices if dims match, else resample
                if d_flat.shape[0] == v_flat.shape[0] + (args.cka_subsample - len(idx)):
                    d_flat_sub = d_flat[idx] if d_flat.shape[0] >= idx.max() + 1 else d_flat
                else:
                    idx2 = rng.choice(d_flat.shape[0], args.cka_subsample, replace=False)
                    d_flat_sub = d_flat[idx2]
            else:
                d_flat_sub = d_flat

            # align N dimension by taking min
            n = min(v_flat.shape[0], d_flat_sub.shape[0])
            cka_grid[vi, di] = linear_cka(v_flat[:n], d_flat_sub[:n])

        if vi % 4 == 0:
            print(f"  V-JEPA layer {vl:2d} done")

    # save raw matrix
    np.save(str(Path(args.out_dir) / f"{args.video_stem}_cka.npy"), cka_grid)

    # plot CKA heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(cka_grid, aspect="auto", origin="lower",
                   vmin=0, vmax=cka_grid.max(), cmap="viridis")
    ax.set_xlabel("DiT layer")
    ax.set_ylabel("V-JEPA layer")
    ax.set_title(f"Linear CKA  (max={cka_grid.max():.3f})")
    ax.set_xticks(range(len(dit_layers)))
    ax.set_xticklabels(dit_layers, rotation=90, fontsize=6)
    ax.set_yticks(range(len(vjepa_layers)))
    ax.set_yticklabels(vjepa_layers, fontsize=7)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    cka_png = Path(args.out_dir) / "cka_heatmap.png"
    fig.savefig(str(cka_png), dpi=150)
    plt.close(fig)
    print(f"Saved {cka_png}")

    # ── Gram Pearson grids ────────────────────────────────────────────────────
    for mode in ("spatial", "temporal"):
        gram_key_v = f"gram_{mode}_layer_"
        gram_key_d = f"gram_{mode}_layer_"

        v_gram_keys = [k for k in vjepa.files if k.startswith(f"gram_{mode}_")]
        d_gram_keys = [k for k in dit.files   if k.startswith(f"gram_{mode}_")]
        if not v_gram_keys or not d_gram_keys:
            print(f"  Skipping {mode} Gram (not found in cache)")
            continue

        print(f"Computing Gram Pearson ({mode})…")
        pearson_grid = np.zeros((len(vjepa_layers), len(dit_layers)), dtype=np.float32)
        for vi, vl in enumerate(vjepa_layers):
            key_v = f"gram_{mode}_layer_{vl:02d}"
            if key_v not in vjepa.files:
                continue
            g_v = vjepa[key_v]
            for di, dl in enumerate(dit_layers):
                key_d = f"gram_{mode}_layer_{dl:02d}"
                if key_d not in dit.files:
                    continue
                g_d = dit[key_d]
                pearson_grid[vi, di] = gram_pearson(g_v, g_d, mode)

        np.save(str(Path(args.out_dir) / f"{args.video_stem}_gram_pearson_{mode}.npy"),
                pearson_grid)

        fig, ax = plt.subplots(figsize=(12, 8))
        vmax = max(abs(pearson_grid.max()), abs(pearson_grid.min()))
        im = ax.imshow(pearson_grid, aspect="auto", origin="lower",
                       vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_xlabel("DiT layer")
        ax.set_ylabel("V-JEPA layer")
        ax.set_title(f"Gram Pearson r  ({mode})  max={pearson_grid.max():.3f}")
        ax.set_xticks(range(len(dit_layers)))
        ax.set_xticklabels(dit_layers, rotation=90, fontsize=6)
        ax.set_yticks(range(len(vjepa_layers)))
        ax.set_yticklabels(vjepa_layers, fontsize=7)
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        out_png = Path(args.out_dir) / f"gram_corr_{mode}.png"
        fig.savefig(str(out_png), dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")

    # ── Best pair visualisation ───────────────────────────────────────────────
    best_vi, best_di = np.unravel_index(np.argmax(cka_grid), cka_grid.shape)
    best_vl = vjepa_layers[best_vi]
    best_dl = dit_layers[best_di]
    print(f"Best CKA pair: V-JEPA layer {best_vl}, DiT layer {best_dl}  "
          f"CKA={cka_grid[best_vi, best_di]:.4f}")

    # spatial Gram average over T
    key_v = f"gram_spatial_layer_{best_vl:02d}"
    key_d = f"gram_spatial_layer_{best_dl:02d}"
    if key_v in vjepa.files and key_d in dit.files:
        g_v = vjepa[key_v].astype(np.float32).mean(axis=0)   # [HW, HW]
        g_d = dit[key_d].astype(np.float32).mean(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, gram, title in zip(axes, [g_v, g_d],
                                   [f"V-JEPA L{best_vl}", f"DiT L{best_dl}"]):
            im = ax.imshow(gram, cmap="hot", aspect="auto")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        fig.suptitle(f"Spatial Gram (avg over T)  |  CKA={cka_grid[best_vi, best_di]:.4f}")
        fig.tight_layout()
        out_png = Path(args.out_dir) / "best_pair_viz.png"
        fig.savefig(str(out_png), dpi=150)
        plt.close(fig)
        print(f"Saved {out_png}")

    print("\nDone. Summary:")
    print(f"  Max CKA = {cka_grid.max():.4f}  at V-JEPA L{best_vl} / DiT L{best_dl}")
    v_sp = np.load(str(Path(args.out_dir) / f"{args.video_stem}_gram_pearson_spatial.npy"), allow_pickle=False) \
        if (Path(args.out_dir) / f"{args.video_stem}_gram_pearson_spatial.npy").exists() else None
    if v_sp is not None:
        print(f"  Max spatial Gram Pearson r = {v_sp.max():.4f}")


if __name__ == "__main__":
    main()
