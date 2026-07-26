#!/usr/bin/env python3
"""Build static plots and a filterable table for all-block head roles."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle


ROLE_ORDER = ("S", "T", "P", "C", "G")
ROLE_LABELS = {
    "S": "帧内空间",
    "T": "球轨迹传播",
    "P": "固定位置时间对齐",
    "C": "首帧/历史上下文",
    "G": "全局聚合",
}
ROLE_LABELS_EN = {
    "S": "intraframe spatial",
    "T": "moving-ball trajectory",
    "P": "fixed-position alignment",
    "C": "first-frame/history context",
    "G": "global aggregation",
}
ROLE_COLORS = {
    "S": "#2ca25f",
    "T": "#2b8cbe",
    "P": "#f0a202",
    "C": "#d95f8d",
    "G": "#756bb1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-csv", type=Path, required=True)
    parser.add_argument("--block-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="All models")
    return parser.parse_args()


def _class_kind(label: str) -> str:
    if label.startswith("明确"):
        return "clear"
    if label.startswith("不稳定"):
        return "unstable"
    return "mixed"


def _plot_role_grid(
    rows: list[dict[str, str]], output: Path, title: str
) -> None:
    by_key = {(int(row["block"]), int(row["head"])): row for row in rows}
    figure, axis = plt.subplots(figsize=(16, 14), dpi=170)
    for block in range(30):
        for head in range(24):
            row = by_key[(block, head)]
            role = row["aggregate_primary_role"]
            stability = float(row["aggregate_role_stability"])
            kind = _class_kind(row["classification"])
            rectangle = Rectangle(
                (head - 0.5, block - 0.5),
                1,
                1,
                facecolor=ROLE_COLORS[role],
                alpha=0.30 + 0.70 * stability,
                edgecolor="#111111" if kind == "clear" else "#ffffff",
                linewidth=1.0 if kind == "clear" else 0.35,
            )
            axis.add_patch(rectangle)
            if kind == "mixed":
                axis.text(
                    head,
                    block,
                    row["aggregate_secondary_role"],
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="#111111",
                    fontweight="bold",
                )
            elif kind == "unstable":
                axis.text(
                    head,
                    block,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#111111",
                    fontweight="bold",
                )
    axis.set_xlim(-0.5, 23.5)
    axis.set_ylim(29.5, -0.5)
    axis.set_xticks(range(24))
    axis.set_yticks(range(30))
    axis.set_xlabel("Head")
    axis.set_ylabel("DiT Block")
    axis.set_title(
        f"{title}: all 720 heads, exact moving-ball query role classification"
    )
    axis.grid(False)
    legend = [
        Patch(facecolor=ROLE_COLORS[role], label=f"{role} {ROLE_LABELS_EN[role]}")
        for role in ROLE_ORDER
    ]
    legend.extend(
        [
            Patch(facecolor="#eeeeee", edgecolor="#111111", label="black border: clear"),
            Patch(facecolor="#eeeeee", label="letter: mixed secondary role"),
            Patch(facecolor="#eeeeee", label="x: unstable mixed"),
        ]
    )
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(output)
    plt.close(figure)


def _plot_stability(
    rows: list[dict[str, str]], output: Path, title: str
) -> None:
    matrix = np.zeros((30, 24), dtype=np.float64)
    for row in rows:
        matrix[int(row["block"]), int(row["head"])] = float(
            row["aggregate_role_stability"]
        )
    figure, axis = plt.subplots(figsize=(15, 11), dpi=170)
    image = axis.imshow(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )
    axis.set_xticks(range(24))
    axis.set_yticks(range(30))
    axis.set_xlabel("Head")
    axis.set_ylabel("DiT Block")
    axis.set_title(f"{title}: primary-role agreement across 4 denoise steps")
    figure.colorbar(image, ax=axis, label="agreement fraction")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def _plot_role_counts(
    rows: list[dict[str, str]], output: Path, title: str
) -> None:
    counts = np.zeros((30, len(ROLE_ORDER)), dtype=np.int64)
    for row in rows:
        counts[
            int(row["block"]), ROLE_ORDER.index(row["aggregate_primary_role"])
        ] += 1
    figure, axis = plt.subplots(figsize=(15, 8), dpi=170)
    left = np.zeros(30, dtype=np.int64)
    positions = np.arange(30)
    for index, role in enumerate(ROLE_ORDER):
        axis.barh(
            positions,
            counts[:, index],
            left=left,
            color=ROLE_COLORS[role],
            label=f"{role} {ROLE_LABELS_EN[role]}",
        )
        left += counts[:, index]
    axis.set_yticks(positions)
    axis.set_ylim(29.7, -0.7)
    axis.set_xlim(0, 24)
    axis.set_xlabel("Head count")
    axis.set_ylabel("DiT Block")
    axis.set_title(f"{title}: primary-role composition per block")
    axis.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output)
    plt.close(figure)


def _write_html(
    rows: list[dict[str, str]],
    block_rows: list[dict[str, str]],
    output: Path,
    title: str,
) -> None:
    table_rows = []
    for row in rows:
        kind = _class_kind(row["classification"])
        cells = [
            row["block"],
            row["head"],
            row["classification"],
            row["aggregate_primary_role"],
            f"{float(row['aggregate_primary_score']):.3f}",
            row["aggregate_secondary_role"],
            f"{float(row['aggregate_secondary_score']):.3f}",
            f"{float(row['aggregate_role_stability']):.0%}",
            f"{float(row['model_role_consistency']):.0%}",
            f"{float(row['step_role_consistency']):.0%}",
            f"{float(row['same_frame_mass']):.3f}",
            f"{float(row['cross_ball_enrichment']):.2f}",
            f"{float(row['aligned_enrichment']):.2f}",
            f"{float(row['first_frame_mass']):.3f}",
            f"{float(row['entropy']):.3f}",
        ]
        table_rows.append(
            f"<tr data-block='{row['block']}' "
            f"data-role='{row['aggregate_primary_role']}' data-kind='{kind}'>"
            + "".join(f"<td>{html.escape(value)}</td>" for value in cells)
            + "</tr>"
        )

    global_roles = Counter(row["aggregate_primary_role"] for row in rows)
    global_kinds = Counter(_class_kind(row["classification"]) for row in rows)
    summary = " | ".join(
        f"{role}={global_roles[role]}" for role in ROLE_ORDER
    )
    classes = (
        f"clear={global_kinds['clear']} | mixed={global_kinds['mixed']} | "
        f"unstable={global_kinds['unstable']}"
    )
    block_payload = json.dumps(block_rows, ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>All-block attention-head classification</title>
<style>
body{{margin:0;background:#f4f5f3;color:#1e2521;font:14px/1.45 system-ui,sans-serif}}
header,main{{max-width:1800px;margin:auto;padding:18px 22px}}h1,h2{{margin:0 0 10px}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.plots img{{width:100%;display:block;background:#fff;border:1px solid #ccd2ce}}
.wide{{grid-column:1/-1}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}}
select{{padding:7px 9px;border:1px solid #aeb7b1;background:#fff}}
.table-wrap{{overflow:auto;max-height:75vh;border:1px solid #bcc4bf;background:#fff}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:6px 8px;border-bottom:1px solid #e0e4e1;text-align:right}}
th{{position:sticky;top:0;background:#25322b;color:#fff}}td:nth-child(3),th:nth-child(3){{text-align:left}}
@media(max-width:900px){{.plots{{grid-template-columns:1fr}}}}
</style></head><body><header>
<h1>{html.escape(title)} · 30 Blocks × 24 Heads</h1>
<p>{summary}</p><p>{classes}</p>
</header><main>
<div class="plots">
<a class="wide" href="allblock_head_role_grid.png"><img src="allblock_head_role_grid.png"></a>
<a href="allblock_head_stability.png"><img src="allblock_head_stability.png"></a>
<a href="allblock_role_counts.png"><img src="allblock_role_counts.png"></a>
</div>
<div class="controls">
<select id="block"><option value="">All blocks</option>{''.join(f'<option>{i}</option>' for i in range(30))}</select>
<select id="role"><option value="">All roles</option>{''.join(f'<option>{role}</option>' for role in ROLE_ORDER)}</select>
<select id="kind"><option value="">All classes</option><option value="clear">clear</option><option value="mixed">mixed</option><option value="unstable">unstable</option></select>
</div>
<div class="table-wrap"><table><thead><tr>
<th>Block</th><th>Head</th><th>Classification</th><th>Primary</th><th>Primary score</th>
<th>Secondary</th><th>Secondary score</th><th>12-sample stability</th>
<th>Model consistency</th><th>Step consistency</th><th>Same-frame mass</th>
<th>Ball trajectory ×</th><th>Fixed alignment ×</th><th>First-frame mass</th><th>Entropy</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</main><script>
const controls=[document.querySelector('#block'),document.querySelector('#role'),document.querySelector('#kind')];
function filterRows(){{
  const [block,role,kind]=controls.map(x=>x.value);
  document.querySelectorAll('tbody tr').forEach(row=>{{
    row.hidden=(block && row.dataset.block!==block)||(role && row.dataset.role!==role)||(kind && row.dataset.kind!==kind);
  }});
}}
controls.forEach(control=>control.addEventListener('change',filterRows));
const blockMetrics={block_payload};
</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.head_csv.expanduser().resolve().open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.block_csv.expanduser().resolve().open(encoding="utf-8", newline="") as handle:
        block_rows = list(csv.DictReader(handle))
    if len(rows) != 720:
        raise RuntimeError(f"expected 720 head rows, found {len(rows)}")
    _plot_role_grid(rows, output / "allblock_head_role_grid.png", args.title)
    _plot_stability(rows, output / "allblock_head_stability.png", args.title)
    _plot_role_counts(rows, output / "allblock_role_counts.png", args.title)
    _write_html(rows, block_rows, output, args.title)
    print(f"gallery={output / 'index.html'}")


if __name__ == "__main__":
    main()
