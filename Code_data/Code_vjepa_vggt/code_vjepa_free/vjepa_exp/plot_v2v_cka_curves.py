"""
可视化所有v2v模型组 vs GT 的CKA曲线。
每条曲线 = 对该组所有样本取均值后，沿 V-JEPA 层 or DiT 层的 marginal mean。
输出两张图：
  - cka_vjepa_axis.png  : x轴 = V-JEPA层，y = mean over DiT层
  - cka_dit_axis.png    : x轴 = DiT层，   y = mean over V-JEPA层
GT 用粗黑线高亮。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict

VJEPA_LAYERS = [0, 3, 5, 8, 11, 14, 17, 20, 23]
DIT_LAYERS   = list(range(30))

BASE   = Path('/data/gaoya/AAA_test_video/0626vjepa_free/GT_check/0623test_100')
OUT    = BASE / 'plots'
OUT.mkdir(exist_ok=True)

import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument('--gt-dir', default=str(BASE / 'GT'))
_args, _ = _parser.parse_known_args()
GT_DIR = Path(_args.gt_dir)

# ── load GT ──────────────────────────────────────────────────────────────────
def load_mean_grid(npy_dir: Path) -> np.ndarray | None:
    """Load all (9,30) npy in dir and return mean grid."""
    npys = list(npy_dir.glob('*.npy'))
    if not npys:
        return None
    grids = [np.load(p) for p in npys]
    return np.stack(grids).mean(axis=0)   # (9, 30)

gt_grid = load_mean_grid(GT_DIR)
assert gt_grid is not None, f"GT npy not found in {GT_DIR}"

# ── load all v2v groups ───────────────────────────────────────────────────────
tasks = json.load(open('/data/gaoya/AAA_test_video/0623/testjsons/v2v_task_list.json'))

groups = {}   # label -> (9,30) mean grid
for t in tasks:
    cd = Path(t['case_dir'])
    label = str(cd.relative_to(BASE))
    g = load_mean_grid(cd)
    if g is not None:
        groups[label] = g

print(f"GT + {len(groups)} v2v groups loaded")

# ── color assignment by top-level model family ───────────────────────────────
families = sorted(set(l.split('/')[0] for l in groups))
palette = cm.get_cmap('tab10', len(families))
fam_color = {f: palette(i) for i, f in enumerate(families)}

def label2color(label):
    return fam_color[label.split('/')[0]]

def short_label(label):
    parts = label.split('/')
    family = parts[0]
    tail   = '/'.join(parts[1:]) if len(parts) > 1 else label

    # family abbreviation
    family = (family
        .replace('basemodel', 'base')
        .replace('loramodel', 'lora')
        .replace('pybullet0624_diffsynth_object_v_newtrain_gpu67', 'py24-newtrain')
        .replace('pybullet0625_diffsynth_object_heads_only_gpu67', 'py25-heads')
        .replace('pybullet0626_diffsynth_object_heads_only_gpu67_fresh500_val', 'py26-heads-fr500')
        .replace('pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun', 'py26-stage2')
        .replace('pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun_chain', 'py26-stage2-chain')
        .replace('pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun_frame49', 'py26-stage2-f49')
        .replace('pybullet0629_stage1b_old8000_cross', 'py29-stage1b-cross')
    )
    tail = (tail
        .replace('wan2p2_ti2v5B', 'ti2v5B')
        .replace('wan_openvid_0613pybullet_lorav2v_step', 's')
        .replace('wan_openvid_lorav2v_step', 's')
    )
    return f"{family}/{tail}" if tail else family

# ── plot helper ───────────────────────────────────────────────────────────────
def make_plot(axis: str):
    """axis: 'vjepa' or 'dit'"""
    fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(16, 6),
                                      gridspec_kw={'width_ratios': [3, 1]})
    ax_leg.axis('off')

    legend_lines = []
    legend_labels = []

    for label, grid in groups.items():
        if axis == 'vjepa':
            y = grid.mean(axis=1)
            x = VJEPA_LAYERS
        else:
            y = grid.mean(axis=0)
            x = DIT_LAYERS
        color = label2color(label)
        sl = short_label(label)
        line, = ax.plot(x, y, color=color, linewidth=1.2, alpha=0.75, linestyle='-')
        legend_lines.append(line)
        legend_labels.append(sl)

    # GT on top
    if axis == 'vjepa':
        gt_y = gt_grid.mean(axis=1)
        x    = VJEPA_LAYERS
        xlabel = 'V-JEPA layer'
        ax.set_xticks(VJEPA_LAYERS)
    else:
        gt_y = gt_grid.mean(axis=0)
        x    = DIT_LAYERS
        xlabel = 'DiT layer'

    gt_line, = ax.plot(x, gt_y, color='black', linewidth=3.0, zorder=10,
                        linestyle='-', marker='o', markersize=4)
    legend_lines = [gt_line] + legend_lines
    legend_labels = ['GT (real video)'] + legend_labels

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Mean CKA', fontsize=12)
    ax.set_title(f'V-JEPA–DiT CKA  ({xlabel} axis)  —  GT vs v2v models', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax_leg.legend(legend_lines, legend_labels,
                  fontsize=7.5, loc='upper left',
                  ncol=1, frameon=False,
                  handlelength=1.5, handletextpad=0.4, labelspacing=0.35)

    fig.tight_layout(pad=1.5)
    out_path = OUT / f'cka_{axis}_axis.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

make_plot('vjepa')
make_plot('dit')
print("Done.")
