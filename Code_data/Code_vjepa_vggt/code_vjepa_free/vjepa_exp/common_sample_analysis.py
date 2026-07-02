"""
按数字 ID 匹配 GT / basemodel / step-002400 的公共样本，计算均值 CKA 并可视化。
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

VJEPA_LAYERS = [0, 3, 5, 8, 11, 14, 17, 20, 23]
DIT_LAYERS   = list(range(30))
TIMESTEPS    = [100, 300, 500, 700, 900]

BASE    = Path("/data/gaoya/AAA_test_video/0626vjepa_free/GT_check/0623test_100")
OUT     = BASE / "plots_common"
OUT.mkdir(exist_ok=True)

GROUPS = {
    "GT":       BASE / "GT_algram",
    "base_ti2v5B": BASE / "basemodel/wan2p2_ti2v5B_algram",
    "step-002400": BASE / "pybullet0624_diffsynth_object_v_newtrain_gpu67/step-002400_algram",
}


def id_map(d: Path) -> dict:
    """stem → numeric_id  (first 6-digit run in stem)"""
    result = {}
    for f in d.glob("*.npy"):
        if f.name == "timesteps.npy":
            continue
        m = re.search(r"(\d{6})", f.stem)
        if m:
            result[m.group(1)] = f
    return result


id_maps = {k: id_map(d) for k, d in GROUPS.items()}

# common IDs
common_ids = sorted(
    set.intersection(*[set(m.keys()) for m in id_maps.values()])
)
print(f"公共样本数: {len(common_ids)}")


def load_grids(id_m: dict, ids: list) -> np.ndarray:
    """返回 [N, 9, 30, 5]"""
    out = []
    for nid in ids:
        g = np.load(str(id_m[nid])).astype(np.float32)
        if g.ndim == 2:          # legacy [9,30]
            g = g[:, :, np.newaxis]
        assert g.shape == (9, 30, 5), f"unexpected shape {g.shape} in {id_m[nid]}"
        out.append(g)
    return np.stack(out)         # [N, 9, 30, 5]


grids = {k: load_grids(id_maps[k], common_ids) for k in GROUPS}
# [N, 9, 30, 5]

means = {k: g.mean(axis=0) for k, g in grids.items()}   # [9, 30, 5]
means_t = {k: g.mean(axis=2) for k, g in means.items()} # [9, 30]

# ── 1. heatmaps (mean over timesteps) ────────────────────────────────────────
n = len(means_t)
vmax = max(g.max() for g in means_t.values())
fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
for ax, (label, g) in zip(axes, means_t.items()):
    im = ax.imshow(g, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="viridis")
    ax.set_title(f"{label}\n(n={len(common_ids)}, max={g.max():.3f})", fontsize=9)
    ax.set_xlabel("DiT layer", fontsize=8)
    ax.set_xticks(range(0, 30, 5)); ax.set_xticklabels(range(0, 30, 5), fontsize=7)
    ax.set_yticks(range(9)); ax.set_yticklabels(VJEPA_LAYERS, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[0].set_ylabel("V-JEPA layer", fontsize=8)
fig.suptitle(f"Cosine-kernel CKA — {len(common_ids)} common samples (mean over τ)", fontsize=11)
fig.tight_layout()
fig.savefig(str(OUT / "heatmap_mean_ts.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved heatmap_mean_ts.png")

# ── 2. per-timestep heatmap for each group ────────────────────────────────────
for label, g in means.items():   # [9, 30, 5]
    T = g.shape[2]
    vmax_l = g.max()
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 4), sharey=True)
    for ti, ax in enumerate(axes):
        im = ax.imshow(g[:, :, ti], aspect="auto", origin="lower",
                       vmin=0, vmax=vmax_l, cmap="viridis")
        ax.set_title(f"τ={TIMESTEPS[ti]}", fontsize=10)
        ax.set_xlabel("DiT layer", fontsize=8)
        ax.set_xticks(range(0, 30, 5)); ax.set_xticklabels(range(0, 30, 5), fontsize=7)
        ax.set_yticks(range(9)); ax.set_yticklabels(VJEPA_LAYERS, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("V-JEPA layer", fontsize=8)
    fig.suptitle(f"{label} — CKA per τ (n={len(common_ids)})", fontsize=11)
    fig.tight_layout()
    fname = f"heatmap_{label.replace('/', '_')}_per_ts.png"
    fig.savefig(str(OUT / fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")

# ── 3. curves: V-JEPA axis ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, g in means_t.items():
    lw = 2.5 if label == "GT" else 1.5
    ax.plot(VJEPA_LAYERS, g.mean(axis=1), label=label, linewidth=lw, marker="o", markersize=4)
ax.set_xlabel("V-JEPA layer"); ax.set_ylabel("Mean CKA (over DiT layers & τ)")
ax.set_title(f"CKA vs V-JEPA depth — {len(common_ids)} common samples")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(OUT / "curve_vjepa_axis.png"), dpi=150)
plt.close(fig)
print("Saved curve_vjepa_axis.png")

# ── 4. curves: DiT axis ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for label, g in means_t.items():
    lw = 2.5 if label == "GT" else 1.5
    ax.plot(DIT_LAYERS, g.mean(axis=0), label=label, linewidth=lw, marker="o", markersize=3)
ax.set_xlabel("DiT layer"); ax.set_ylabel("Mean CKA (over V-JEPA layers & τ)")
ax.set_title(f"CKA vs DiT depth — {len(common_ids)} common samples")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(OUT / "curve_dit_axis.png"), dpi=150)
plt.close(fig)
print("Saved curve_dit_axis.png")

# ── 5. per-timestep curves: DiT axis ─────────────────────────────────────────
fig, axes = plt.subplots(1, len(TIMESTEPS), figsize=(5 * len(TIMESTEPS), 4), sharey=True)
for ti, tau in enumerate(TIMESTEPS):
    ax = axes[ti]
    for label, g in means.items():   # [9, 30, 5]
        lw = 2.5 if label == "GT" else 1.5
        ax.plot(DIT_LAYERS, g[:, :, ti].mean(axis=0), label=label, linewidth=lw, marker="o", markersize=3)
    ax.set_title(f"τ={tau}", fontsize=10)
    ax.set_xlabel("DiT layer", fontsize=8)
    ax.grid(True, alpha=0.3)
    if ti == 0:
        ax.set_ylabel("Mean CKA (over V-JEPA layers)")
        ax.legend(fontsize=7)
fig.suptitle(f"CKA vs DiT depth per timestep — {len(common_ids)} common samples", fontsize=11)
fig.tight_layout()
fig.savefig(str(OUT / "curve_dit_per_ts.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved curve_dit_per_ts.png")

# ── 6. D_rel = 1 - CKA 差值 heatmap (GT - generated) ─────────────────────────
gt_g = means_t["GT"]
fig, axes = plt.subplots(1, n - 1, figsize=(5 * (n - 1), 4), sharey=True)
if n - 1 == 1:
    axes = [axes]
other_labels = [k for k in means_t if k != "GT"]
vmax_diff = max(abs(gt_g - means_t[k]).max() for k in other_labels)
for ax, label in zip(axes, other_labels):
    diff = gt_g - means_t[label]   # positive = GT > generated
    im = ax.imshow(diff, aspect="auto", origin="lower",
                   vmin=-vmax_diff, vmax=vmax_diff, cmap="RdBu_r")
    ax.set_title(f"GT − {label}\nmax_gap={diff.max():.3f}", fontsize=9)
    ax.set_xlabel("DiT layer", fontsize=8)
    ax.set_xticks(range(0, 30, 5)); ax.set_xticklabels(range(0, 30, 5), fontsize=7)
    ax.set_yticks(range(9)); ax.set_yticklabels(VJEPA_LAYERS, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[0].set_ylabel("V-JEPA layer", fontsize=8)
fig.suptitle(f"CKA gap (GT − generated) — {len(common_ids)} common samples", fontsize=11)
fig.tight_layout()
fig.savefig(str(OUT / "diff_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved diff_heatmap.png")

# ── 7. 打印最大区分度的 (vi, di, ti) 组合 ────────────────────────────────────
print("\n=== 各 generated 组 vs GT 最大差值位置 ===")
for label in other_labels:
    diff3d = means["GT"] - means[label]   # [9, 30, 5]
    idx = np.unravel_index(np.argmax(diff3d), diff3d.shape)
    vi, di, ti = idx
    print(f"  {label}: V-JEPA L{VJEPA_LAYERS[vi]}, DiT L{di}, τ={TIMESTEPS[ti]}  gap={diff3d[vi,di,ti]:.4f}")

# ── 8. per-sample D_rel boxplot for best (vi,di,ti) ──────────────────────────
for label in other_labels:
    diff3d = means["GT"] - means[label]
    idx = np.unravel_index(np.argmax(diff3d), diff3d.shape)
    vi, di, ti = idx

    gt_vals  = grids["GT"][:, vi, di, ti]     # [N]
    gen_vals = grids[label][:, vi, di, ti]    # [N]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([gt_vals, gen_vals], labels=["GT", label], patch_artist=True)
    ax.set_ylabel("CKA")
    ax.set_title(f"V-JEPA L{VJEPA_LAYERS[vi]} × DiT L{di} × τ={TIMESTEPS[ti]}\n"
                 f"GT mean={gt_vals.mean():.3f}  {label} mean={gen_vals.mean():.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"boxplot_{label.replace('/', '_')}_best.png"
    fig.savefig(str(OUT / fname), dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")

print(f"\n所有图保存到 {OUT}")
