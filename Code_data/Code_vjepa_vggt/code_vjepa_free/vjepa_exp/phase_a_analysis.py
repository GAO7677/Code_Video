"""
Phase A: GT-only layer correspondence analysis.

Loads per-video [9, 30, T] CKA grids from a GT case-dir, averages across
videos, then plots [V-JEPA layer, DiT layer] heatmaps per timestep and
identifies the best-matching (vj_layer, dit_layer, timestep) triple.

Output:
  <out-dir>/
    phase_a_heatmaps.png          — one heatmap per τ
    phase_a_layer_curves.png      — mean CKA vs V-JEPA depth, one line per τ
    phase_a_best_layers.json      — {vj_layer, vj_layer_idx, dit_layer,
                                     dit_layer_idx, timestep, timestep_idx,
                                     peak_cka}

Usage:
    python phase_a_analysis.py \
        --gt-dir  /data/gaoya/.../GT_multi_ts \
        --out-dir /data/gaoya/.../phase_a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VJEPA_LAYERS = [0, 3, 5, 8, 11, 14, 17, 20, 23]
DIT_LAYERS   = list(range(30))


def load_grids(case_dir: Path) -> tuple[np.ndarray, list[int]]:
    """Return (mean_grid [9, 30, T], timesteps list)."""
    ts_file = case_dir / "timesteps.npy"
    if ts_file.exists():
        timesteps = np.load(str(ts_file)).tolist()
    else:
        timesteps = [500]

    npys = sorted(f for f in case_dir.glob("*.npy") if f.name != "timesteps.npy")
    if not npys:
        raise FileNotFoundError(f"No .npy files in {case_dir}")

    grids = []
    for p in npys:
        g = np.load(str(p)).astype(np.float32)
        if g.ndim == 2:                   # legacy [9,30] → [9,30,1]
            g = g[:, :, np.newaxis]
        grids.append(g)

    mean_grid = np.stack(grids).mean(axis=0)   # [9, 30, T]
    print(f"Loaded {len(grids)} videos → mean grid {mean_grid.shape}, "
          f"timesteps={timesteps}")
    return mean_grid, timesteps


def plot_heatmaps(mean_grid: np.ndarray, timesteps: list[int],
                  vj_layers: list[int], dt_layers: list[int],
                  out_dir: Path) -> None:
    T = len(timesteps)
    vmax = mean_grid.max()
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 4.5), sharey=True)
    if T == 1:
        axes = [axes]
    for ti, (ax, ts) in enumerate(zip(axes, timesteps)):
        g = mean_grid[:, :, ti]
        im = ax.imshow(g, aspect="auto", origin="lower",
                       vmin=0, vmax=vmax, cmap="viridis")
        peak_vi, peak_di = np.unravel_index(g.argmax(), g.shape)
        ax.set_title(f"τ={ts}\npeak @ vj={vj_layers[peak_vi]}, dit={dt_layers[peak_di]}"
                     f"\n CKA={g.max():.3f}", fontsize=9)
        ax.set_xlabel("DiT-5B layer")
        if ti == 0:
            ax.set_ylabel("V-JEPA layer")
        ax.set_xticks(range(len(dt_layers)))
        ax.set_xticklabels(dt_layers, fontsize=6, rotation=45)
        ax.set_yticks(range(len(vj_layers)))
        ax.set_yticklabels(vj_layers, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Phase A: V-JEPA–DiT CKA on GT videos (per timestep)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_dir / "phase_a_heatmaps.png"), dpi=150)
    plt.close(fig)
    print("Saved phase_a_heatmaps.png")


def plot_layer_curves(mean_grid: np.ndarray, timesteps: list[int],
                      vj_layers: list[int],
                      out_dir: Path) -> None:
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(timesteps) - 1, 1)) for i in range(len(timesteps))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # left: mean over DiT layers, per V-JEPA depth
    ax = axes[0]
    for ti, (ts, c) in enumerate(zip(timesteps, colors)):
        ax.plot(vj_layers, mean_grid[:, :, ti].mean(axis=1),
                marker="o", markersize=4, color=c, label=f"τ={ts}")
    ax.set_xlabel("V-JEPA layer")
    ax.set_ylabel("Mean CKA (over DiT layers)")
    ax.set_title("CKA vs V-JEPA depth")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # right: mean over V-JEPA layers, per DiT depth
    ax = axes[1]
    dt_layers = list(range(mean_grid.shape[1]))
    for ti, (ts, c) in enumerate(zip(timesteps, colors)):
        ax.plot(dt_layers, mean_grid[:, :, ti].mean(axis=0),
                marker="o", markersize=3, color=c, label=f"τ={ts}")
    ax.set_xlabel("DiT-5B layer")
    ax.set_ylabel("Mean CKA (over V-JEPA layers)")
    ax.set_title("CKA vs DiT depth")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Phase A: layer marginals on GT videos", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_dir / "phase_a_layer_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved phase_a_layer_curves.png")


def find_best_layers(mean_grid: np.ndarray, timesteps: list[int],
                     vj_layers: list[int], dt_layers: list[int]) -> dict:
    peak_flat = mean_grid.argmax()
    vi, di, ti = np.unravel_index(peak_flat, mean_grid.shape)
    return {
        "vj_layer":      int(vj_layers[vi]),
        "vj_layer_idx":  int(vi),
        "dit_layer":     int(dt_layers[di]),
        "dit_layer_idx": int(di),
        "timestep":      int(timesteps[ti]),
        "timestep_idx":  int(ti),
        "peak_cka":      float(mean_grid[vi, di, ti]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True,
                   help="case-dir containing per-video [9,30,T] .npy from stream_cka.py")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <gt-dir>/../phase_a)")
    args = p.parse_args()

    gt_dir  = Path(args.gt_dir)
    out_dir = Path(args.out_dir) if args.out_dir else gt_dir.parent / "phase_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_grid, timesteps = load_grids(gt_dir)
    vj_layers = VJEPA_LAYERS[:mean_grid.shape[0]]
    dt_layers = list(range(mean_grid.shape[1]))

    plot_heatmaps(mean_grid, timesteps, vj_layers, dt_layers, out_dir)
    plot_layer_curves(mean_grid, timesteps, vj_layers, out_dir)

    best = find_best_layers(mean_grid, timesteps, vj_layers, dt_layers)
    best_path = out_dir / "phase_a_best_layers.json"
    json.dump(best, open(str(best_path), "w"), indent=2)
    print(f"Saved phase_a_best_layers.json")
    print(f"\nBest layer pair:")
    print(f"  V-JEPA layer : {best['vj_layer']} (idx {best['vj_layer_idx']})")
    print(f"  DiT   layer  : {best['dit_layer']} (idx {best['dit_layer_idx']})")
    print(f"  timestep     : {best['timestep']} (idx {best['timestep_idx']})")
    print(f"  peak CKA     : {best['peak_cka']:.4f}")

    # also save full mean grid for phase B
    np.save(str(out_dir / "gt_mean_grid.npy"), mean_grid)
    print("Saved gt_mean_grid.npy")


if __name__ == "__main__":
    main()
