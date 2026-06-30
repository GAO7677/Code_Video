"""
Per-dataset V-JEPA vs Wan DiT-5B CKA comparison.

Reads all {stem}_vjepa.npz + {stem}_dit5b.npz from --repr-dir,
groups by source dataset (phyco_kubric / pybullet / physics-iq),
computes per-video CKA grids, then averages per dataset.

Outputs (in --out-dir):
  cka_per_dataset.png        3-panel heatmap (one per dataset)
  cka_per_layer_curve.png    per-vjepa-layer mean CKA, 3 curves per dataset
  cka_matrices.npz           raw per-dataset average CKA grids
  cka_dataset_diff.png       heatmap of pairwise dataset differences

Usage:
    python analyze_phys_dataset_cka.py \\
        --repr-dir /data/gaoya/agent-data/outputs/phys_compare/repr_cache \\
        --out-dir  /data/gaoya/agent-data/outputs/phys_compare/cka_analysis \\
        [--cka-subsample 1500]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Linear CKA ────────────────────────────────────────────────────────────────

def center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    K = center_gram(X @ X.T)
    L = center_gram(Y @ Y.T)
    num = np.sum(K * L)
    denom = np.sqrt(np.sum(K * K) * np.sum(L * L))
    return float(num / (denom + 1e-12))


# ── Per-video CKA grid ─────────────────────────────────────────────────────────

def compute_video_cka_grid(
    vjepa_path: Path,
    dit_path: Path,
    subsample: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Returns CKA grid [n_vjepa_layers, n_dit_layers] or None on error."""
    try:
        vj = np.load(str(vjepa_path), allow_pickle=False)
        dt = np.load(str(dit_path),   allow_pickle=False)
    except Exception as e:
        print(f"  load error: {e}")
        return None

    vj_layers = sorted(vj["layers"].tolist())
    dt_layers = sorted(dt["layers"].tolist())
    grid = np.zeros((len(vj_layers), len(dt_layers)), dtype=np.float32)

    for vi, vl in enumerate(vj_layers):
        v_tok = vj[f"tokens_layer_{vl:02d}"].astype(np.float32)   # [T,H,W,D]
        T, H, W, D = v_tok.shape
        v_flat = v_tok.reshape(-1, D)
        if v_flat.shape[0] > subsample:
            idx = rng.choice(v_flat.shape[0], subsample, replace=False)
            v_flat = v_flat[idx]
        else:
            idx = None

        for di, dl in enumerate(dt_layers):
            key = f"tokens_layer_{dl:02d}"
            if key not in dt.files:
                continue
            d_tok = dt[key].astype(np.float32)
            d_flat = d_tok.reshape(-1, d_tok.shape[-1])
            if d_flat.shape[0] > subsample:
                if idx is not None and d_flat.shape[0] >= idx.max() + 1:
                    d_flat = d_flat[idx]
                else:
                    idx2 = rng.choice(d_flat.shape[0], subsample, replace=False)
                    d_flat = d_flat[idx2]

            n = min(v_flat.shape[0], d_flat.shape[0])
            grid[vi, di] = linear_cka(v_flat[:n], d_flat[:n])

    return grid


# ── Plotting helpers ───────────────────────────────────────────────────────────

DATASET_COLORS = {
    "phyco_kubric": "#1f77b4",
    "pybullet":     "#ff7f0e",
    "physics-iq":   "#2ca02c",
}
DATASET_ORDER = ["phyco_kubric", "pybullet", "physics-iq"]


def plot_cka_heatmaps(
    dataset_grids: dict[str, np.ndarray],
    vj_layers: list[int],
    dt_layers: list[int],
    out_path: Path,
) -> None:
    datasets = [d for d in DATASET_ORDER if d in dataset_grids]
    n = len(datasets)
    vmax = max(g.max() for g in dataset_grids.values())

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        g = dataset_grids[ds]
        im = ax.imshow(g, aspect="auto", origin="lower",
                       vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"{ds}\nmax={g.max():.3f}", fontsize=10)
        ax.set_xlabel("DiT-5B layer", fontsize=8)
        ax.set_ylabel("V-JEPA layer", fontsize=8)
        ax.set_xticks(range(0, len(dt_layers), 5))
        ax.set_xticklabels(dt_layers[::5], fontsize=6, rotation=45)
        ax.set_yticks(range(len(vj_layers)))
        ax.set_yticklabels(vj_layers, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("V-JEPA vs Wan DiT-5B  —  Linear CKA (per-dataset average)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_per_layer_curves(
    dataset_grids: dict[str, np.ndarray],
    vj_layers: list[int],
    out_path: Path,
) -> None:
    """Line plot: x = V-JEPA layer, y = mean CKA over all DiT layers."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds in DATASET_ORDER:
        if ds not in dataset_grids:
            continue
        g = dataset_grids[ds]               # [n_vj, n_dt]
        mean_cka = g.mean(axis=1)           # [n_vj]
        ax.plot(vj_layers, mean_cka,
                label=ds, color=DATASET_COLORS[ds],
                marker="o", markersize=4, linewidth=1.5)

    ax.set_xlabel("V-JEPA layer", fontsize=9)
    ax.set_ylabel("Mean CKA (over DiT layers)", fontsize=9)
    ax.set_title("V-JEPA–DiT CKA vs V-JEPA depth  —  per dataset", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_diff_heatmaps(
    dataset_grids: dict[str, np.ndarray],
    vj_layers: list[int],
    dt_layers: list[int],
    out_path: Path,
) -> None:
    """Pairwise difference grids for available dataset pairs."""
    pairs = [
        ("pybullet",   "phyco_kubric"),
        ("physics-iq", "phyco_kubric"),
        ("physics-iq", "pybullet"),
    ]
    pairs = [(a, b) for a, b in pairs if a in dataset_grids and b in dataset_grids]
    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5), sharey=True)
    if len(pairs) == 1:
        axes = [axes]

    for ax, (a, b) in zip(axes, pairs):
        diff = dataset_grids[a] - dataset_grids[b]
        vabs = max(abs(diff.max()), abs(diff.min()), 0.01)
        im = ax.imshow(diff, aspect="auto", origin="lower",
                       vmin=-vabs, vmax=vabs, cmap="RdBu_r")
        ax.set_title(f"{a}\nminus {b}", fontsize=9)
        ax.set_xlabel("DiT-5B layer", fontsize=8)
        ax.set_ylabel("V-JEPA layer", fontsize=8)
        ax.set_xticks(range(0, len(dt_layers), 5))
        ax.set_xticklabels(dt_layers[::5], fontsize=6, rotation=45)
        ax.set_yticks(range(len(vj_layers)))
        ax.set_yticklabels(vj_layers, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("CKA difference between datasets", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repr-dir", default="/data/gaoya/agent-data/outputs/phys_compare/repr_cache")
    p.add_argument("--out-dir",  default="/data/gaoya/agent-data/outputs/phys_compare/cka_analysis")
    p.add_argument("--cka-subsample", type=int, default=1500)
    args = p.parse_args()

    repr_dir = Path(args.repr_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load stem index
    idx_path = repr_dir / "stem_index.json"
    if not idx_path.exists():
        print(f"ERROR: stem_index.json not found in {repr_dir}")
        print("Run batch_extract_phys.py first.")
        return
    stem_index = json.load(open(idx_path))

    rng = np.random.default_rng(0)
    # Group by dataset
    by_dataset: dict[str, list[np.ndarray]] = defaultdict(list)
    vj_layers_ref: list[int] | None = None
    dt_layers_ref: list[int] | None = None

    total = len(stem_index)
    for n_done, (stem, info) in enumerate(stem_index.items()):
        vj_path  = repr_dir / f"{stem}_vjepa.npz"
        dit_path = repr_dir / f"{stem}_dit5b.npz"
        if not vj_path.exists() or not dit_path.exists():
            missing = []
            if not vj_path.exists():  missing.append("vjepa")
            if not dit_path.exists(): missing.append("dit5b")
            print(f"  [{n_done+1}/{total}] skip {stem} (missing: {', '.join(missing)})")
            continue

        print(f"  [{n_done+1}/{total}] {stem}  ({info['source']})")
        grid = compute_video_cka_grid(vj_path, dit_path, args.cka_subsample, rng)
        if grid is None:
            continue

        ds = info["source"]
        by_dataset[ds].append(grid)

        # record layer order from first successfully processed video
        if vj_layers_ref is None:
            vj = np.load(str(vj_path), allow_pickle=False)
            dt = np.load(str(dit_path), allow_pickle=False)
            vj_layers_ref = sorted(vj["layers"].tolist())
            dt_layers_ref = sorted(dt["layers"].tolist())

    if vj_layers_ref is None:
        print("No data processed. Exiting.")
        return

    print(f"\nDataset counts: { {ds: len(v) for ds, v in by_dataset.items()} }")

    # Average per dataset
    dataset_grids: dict[str, np.ndarray] = {}
    for ds, grids in by_dataset.items():
        dataset_grids[ds] = np.stack(grids).mean(axis=0)
        print(f"  {ds}: mean CKA max = {dataset_grids[ds].max():.4f}")

    # Save raw matrices
    np.savez_compressed(
        str(out_dir / "cka_matrices.npz"),
        **{ds.replace("-", "_"): g for ds, g in dataset_grids.items()},
        vj_layers=np.array(vj_layers_ref),
        dt_layers=np.array(dt_layers_ref),
    )
    print(f"Saved cka_matrices.npz")

    # Plots
    plot_cka_heatmaps(dataset_grids, vj_layers_ref, dt_layers_ref,
                      out_dir / "cka_per_dataset.png")
    print("Saved cka_per_dataset.png")

    plot_per_layer_curves(dataset_grids, vj_layers_ref,
                          out_dir / "cka_per_layer_curve.png")
    print("Saved cka_per_layer_curve.png")

    plot_diff_heatmaps(dataset_grids, vj_layers_ref, dt_layers_ref,
                       out_dir / "cka_dataset_diff.png")
    print("Saved cka_dataset_diff.png")

    # Print summary table
    print("\n=== Summary: max CKA per dataset ===")
    for ds in DATASET_ORDER:
        if ds not in dataset_grids:
            continue
        g = dataset_grids[ds]
        best_vi, best_di = np.unravel_index(np.argmax(g), g.shape)
        print(f"  {ds:<20} max={g.max():.4f}  "
              f"at V-JEPA L{vj_layers_ref[best_vi]} / DiT L{dt_layers_ref[best_di]}")


if __name__ == "__main__":
    main()
