"""
Phase B: GT vs generated videos — V-JEPA–DiT relation mismatch as physical
anomaly signal.

For each video, D_rel = 1 - CKA(R_V^{l*}, R_D^{m*,τ*}) where (l*, m*, τ*)
comes from Phase A's best_layers.json.

Outputs:
  phase_b_distributions.png  — violin/box of D_rel per group
  phase_b_roc.png            — ROC curve (GT=0 vs generated=1)
  phase_b_stats.json         — AUC, mean/std per group, Mann-Whitney p

Usage:
    python phase_b_analysis.py \
        --best-layers <phase_a_out>/phase_a_best_layers.json \
        --gt-dir      <case-dir with GT npy> \
        --model-dirs  label1:<case-dir1> label2:<case-dir2> ... \
        --out-dir     <output>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── loading ────────────────────────────────────────────────────────────────────

def load_timesteps(case_dir: Path) -> list[int]:
    ts_file = case_dir / "timesteps.npy"
    return np.load(str(ts_file)).tolist() if ts_file.exists() else [500]


def load_per_video_scores(case_dir: Path, vi: int, di: int, ti: int) -> np.ndarray:
    """Return 1-D array of D_rel = 1 - CKA[vi, di, ti] per video."""
    npys = sorted(f for f in case_dir.glob("*.npy") if f.name != "timesteps.npy")
    scores = []
    for p in npys:
        g = np.load(str(p)).astype(np.float32)
        if g.ndim == 2:             # legacy [9,30] → treat ti=0
            cka_val = g[vi, di]
        else:
            cka_val = g[vi, di, ti]
        scores.append(1.0 - float(cka_val))
    return np.array(scores)


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_distributions(groups: dict[str, np.ndarray], out_dir: Path) -> None:
    labels = list(groups.keys())
    data   = [groups[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 5))
    parts = ax.violinplot(data, positions=range(len(labels)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("D_rel  =  1 − CKA(V-JEPA, DiT)")
    ax.set_title("Phase B: relation mismatch distribution\n(lower = more physically consistent)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "phase_b_distributions.png"), dpi=150)
    plt.close(fig)
    print("Saved phase_b_distributions.png")


def plot_roc(gt_scores: np.ndarray,
             model_groups: dict[str, np.ndarray],
             out_dir: Path) -> dict[str, float]:
    """One ROC per model group (GT=0, model=1).  Returns {label: AUC}."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    aucs: dict[str, float] = {}

    cmap = plt.cm.tab10
    for ci, (label, gen_scores) in enumerate(model_groups.items()):
        y_true  = np.concatenate([np.zeros(len(gt_scores)),
                                   np.ones(len(gen_scores))])
        y_score = np.concatenate([gt_scores, gen_scores])
        try:
            auc = roc_auc_score(y_true, y_score)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, color=cmap(ci % 10),
                    label=f"{label}  AUC={auc:.3f}", lw=1.5)
            aucs[label] = float(auc)
        except ValueError as e:
            print(f"  ROC skipped for {label}: {e}")
            aucs[label] = float("nan")

    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Phase B: ROC  (GT=0 vs generated=1)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / "phase_b_roc.png"), dpi=150)
    plt.close(fig)
    print("Saved phase_b_roc.png")
    return aucs


def compute_stats(gt_scores: np.ndarray,
                  model_groups: dict[str, np.ndarray],
                  aucs: dict[str, float]) -> dict:
    from scipy import stats as sp_stats

    out = {
        "GT": {
            "n":    int(len(gt_scores)),
            "mean": float(gt_scores.mean()),
            "std":  float(gt_scores.std()),
            "median": float(np.median(gt_scores)),
        }
    }
    for label, gen_scores in model_groups.items():
        stat, pval = sp_stats.mannwhitneyu(gt_scores, gen_scores,
                                           alternative="less")
        out[label] = {
            "n":         int(len(gen_scores)),
            "mean":      float(gen_scores.mean()),
            "std":       float(gen_scores.std()),
            "median":    float(np.median(gen_scores)),
            "roc_auc":   aucs.get(label, float("nan")),
            "mw_stat":   float(stat),
            "mw_pval":   float(pval),
        }
    return out


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best-layers", required=True,
                   help="JSON from phase_a_analysis.py")
    p.add_argument("--gt-dir", required=True,
                   help="case-dir with per-video GT .npy files")
    p.add_argument("--model-dirs", nargs="+", metavar="LABEL:DIR",
                   help="e.g. basemodel:/path/to/case_dir loramodel:/path/to/case_dir")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    best = json.load(open(args.best_layers))
    vi   = best["vj_layer_idx"]
    di   = best["dit_layer_idx"]
    ti   = best["timestep_idx"]
    print(f"Using V-JEPA layer {best['vj_layer']}, "
          f"DiT layer {best['dit_layer']}, "
          f"timestep {best['timestep']}  (peak CKA={best['peak_cka']:.4f})")

    gt_dir  = Path(args.gt_dir)
    out_dir = Path(args.out_dir) if args.out_dir else gt_dir.parent / "phase_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_scores = load_per_video_scores(gt_dir, vi, di, ti)
    print(f"GT: {len(gt_scores)} videos, "
          f"D_rel mean={gt_scores.mean():.4f} std={gt_scores.std():.4f}")

    model_groups: dict[str, np.ndarray] = {}
    for spec in (args.model_dirs or []):
        label, path = spec.split(":", 1)
        scores = load_per_video_scores(Path(path), vi, di, ti)
        model_groups[label] = scores
        print(f"{label}: {len(scores)} videos, "
              f"D_rel mean={scores.mean():.4f} std={scores.std():.4f}")

    all_groups = {"GT": gt_scores, **model_groups}
    plot_distributions(all_groups, out_dir)

    aucs: dict[str, float] = {}
    if model_groups:
        aucs = plot_roc(gt_scores, model_groups, out_dir)
        stats = compute_stats(gt_scores, model_groups, aucs)
        stats_path = out_dir / "phase_b_stats.json"
        json.dump(stats, open(str(stats_path), "w"), indent=2)
        print("Saved phase_b_stats.json")
        print("\n=== Phase B summary ===")
        for label, s in stats.items():
            line = f"  {label:40s}  mean={s['mean']:.4f}  std={s['std']:.4f}"
            if "roc_auc" in s:
                line += f"  AUC={s['roc_auc']:.3f}  p={s['mw_pval']:.3e}"
            print(line)
    else:
        print("No --model-dirs provided; only distribution plot saved.")


if __name__ == "__main__":
    main()
