#!/usr/bin/env python3
"""Build progress and metric-curve views for the all-block/all-head sweep."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from configured_head_ablation import (
    load_config,
    result_config_count,
    run_root,
    selected_blocks,
    selected_heads,
)


TAG_PATTERN = re.compile(
    r"self_attn_head_zero_block(?P<block>\d{2})_head(?P<head>\d{2})"
)
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MODEL_PATH_NAMES = {
    "wan_lora": "wan_lora",
    "xssc": "xssc",
    "physrvg": "PhyRVG",
}
METRIC_LABELS = {
    "physics_iq_with_context_score_mean": "Physics-IQ with context",
    "physics_iq_without_context_score_mean": "Physics-IQ without context",
    "pmf_with_context_score_mean": "PMF with context",
    "pmf_without_context_score_mean": "PMF without context",
    "wmreward_surprise_mean": "WMReward surprise",
    "videophy2_sa_score_mean": "VideoPhy2 SA",
    "videophy2_pc_score_mean": "VideoPhy2 PC",
    "videophy2_joint_rate_mean": "VideoPhy2 joint rate",
    "cosmos_reason1_score_mean": "Cosmos-Reason1",
    "vbench_subject_consistency_score_mean": "VBench subject consistency",
    "vbench_background_consistency_score_mean": "VBench background consistency",
    "vbench_temporal_flickering_score_mean": "VBench temporal flickering",
    "vbench_motion_smoothness_score_mean": "VBench motion smoothness",
    "vbench_dynamic_degree_score_mean": "VBench dynamic degree",
    "vbench_aesthetic_quality_score_mean": "VBench aesthetic quality",
    "vbench_imaging_quality_score_mean": "VBench imaging quality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-summary", type=Path)
    return parser.parse_args()


def completed_jobs(root: Path) -> set[tuple[str, int, int]]:
    path = root / "generation" / "completed.tsv"
    output = set()
    if not path.is_file():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 4:
            output.add((fields[1], int(fields[2]), int(fields[3])))
    return output


def failed_jobs(root: Path) -> set[tuple[str, int, int]]:
    path = root / "generation" / "failed.tsv"
    output = set()
    if not path.is_file():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 4:
            output.add((fields[1], int(fields[2]), int(fields[3])))
    return output


def plot_progress(
    output: Path,
    models: list[str],
    blocks: list[int],
    heads: list[int],
    completed: set[tuple[str, int, int]],
    failed: set[tuple[str, int, int]],
) -> str:
    figure, axes = plt.subplots(
        len(models),
        1,
        figsize=(14, 3.8 * len(models)),
        dpi=160,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]
    cmap = ListedColormap(("#e7eaee", "#2a9d6f", "#d1495b"))
    for axis, model in zip(axes, models):
        values = np.zeros((len(blocks), len(heads)), dtype=np.int64)
        for block_index, block in enumerate(blocks):
            for head_index, head in enumerate(heads):
                key = (model, block, head)
                if key in completed:
                    values[block_index, head_index] = 1
                elif key in failed:
                    values[block_index, head_index] = 2
        axis.imshow(values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        axis.set_title(
            f"{MODEL_LABELS[model]}: "
            f"{sum(key[0] == model for key in completed)}/"
            f"{len(blocks) * len(heads)} generated configurations"
        )
        axis.set_xlabel("Head")
        axis.set_ylabel("Block")
        axis.set_xticks(np.arange(0, len(heads), 2), heads[::2])
        axis.set_yticks(np.arange(0, len(blocks), 2), blocks[::2])
        axis.tick_params(labelsize=7)
    figure.legend(
        handles=[
            Patch(facecolor="#e7eaee", label="Pending"),
            Patch(facecolor="#2a9d6f", label="Complete"),
            Patch(facecolor="#d1495b", label="Failed attempt; awaiting retry"),
        ],
        loc="upper center",
        ncol=3,
        fontsize=8,
    )
    filename = "generation_progress.png"
    figure.savefig(output / filename, bbox_inches="tight")
    plt.close(figure)
    return filename


def infer_model(path: str) -> str | None:
    parts = {part.lower() for part in Path(path).parts}
    if "wan_lora" in parts:
        return "wan_lora"
    if "xssc" in parts:
        return "xssc"
    if "phyrvg" in parts or "physrvg" in parts:
        return "physrvg"
    return None


def load_metric_data(
    path: Path,
    models: list[str],
    blocks: list[int],
    heads: list[int],
    expected_cases: int,
) -> tuple[list[str], dict[str, dict[str, list[list[float | None]]]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    metric_keys = [
        key
        for key in METRIC_LABELS
        if any(
            int(float(row.get(key.removesuffix("_mean") + "_count", 0) or 0))
            == expected_cases
            for row in rows
        )
    ]
    data = {
        model: {
            metric: [
                [None for _ in heads]
                for _ in blocks
            ]
            for metric in metric_keys
        }
        for model in models
    }
    block_lookup = {value: index for index, value in enumerate(blocks)}
    head_lookup = {value: index for index, value in enumerate(heads)}
    for row in rows:
        result_root = row.get("result_root", "")
        model = infer_model(result_root)
        match = TAG_PATTERN.search(result_root)
        if model not in data or match is None:
            continue
        block = int(match.group("block"))
        head = int(match.group("head"))
        if block not in block_lookup or head not in head_lookup:
            continue
        for metric in metric_keys:
            count_key = metric.removesuffix("_mean") + "_count"
            try:
                count = int(float(row.get(count_key, 0) or 0))
            except (TypeError, ValueError):
                continue
            if count != expected_cases:
                continue
            raw = row.get(metric, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                data[model][metric][block_lookup[block]][head_lookup[head]] = value
    return metric_keys, data


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = run_root(config)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = list(config["sweep"]["models"])
    blocks = selected_blocks(config)
    heads = selected_heads(config)
    completed = completed_jobs(root)
    failed = failed_jobs(root) - completed
    expected = result_config_count(config)
    metric_summary = (
        args.metric_summary.expanduser().resolve()
        if args.metric_summary
        else root / "metrics" / "metric_summary.csv"
    )
    metric_keys: list[str] = []
    metric_data: dict[str, Any] = {}
    if metric_summary.is_file():
        metric_keys, metric_data = load_metric_data(
            metric_summary,
            models,
            blocks,
            heads,
            int(config["input"]["expected_unique_cases"]),
        )
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "models": models,
        "model_labels": MODEL_LABELS,
        "blocks": blocks,
        "heads": heads,
        "metric_labels": {
            key: METRIC_LABELS[key] for key in metric_keys
        },
        "metrics": metric_data,
        "metric_summary": str(metric_summary),
        "generation": {
            "complete": len(completed),
            "failed_unrecovered": len(failed),
            "expected": expected,
        },
    }
    (output / "data.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    model_options = "".join(
        f"<option value='{html.escape(model)}'>"
        f"{html.escape(MODEL_LABELS[model])}</option>"
        for model in models
    )
    metric_options = "".join(
        f"<option value='{html.escape(metric)}'>"
        f"{html.escape(METRIC_LABELS[metric])}</option>"
        for metric in metric_keys
    )
    metric_view = (
        f"""<section id="metric-view">
<div class="toolbar">
<label>Model<select id="model-select">{model_options}</select></label>
<label>Metric<select id="metric-select">{metric_options}</select></label>
<label>Block<select id="block-select"></select></label>
<label>Head<select id="head-select"></select></label>
</div>
<div class="chart-grid">
<div><h3>Selected block: metric across heads</h3><canvas id="head-chart"></canvas></div>
<div><h3>Selected head: metric across blocks</h3><canvas id="block-chart"></canvas></div>
</div>
<p class="note">Every point is the 20-case mean for one self-attention-head-zero
configuration. Missing points remain blank. No 67-case baseline is mixed into this
test_5 experiment.</p>
</section>"""
        if metric_keys
        else """<section><h2>Metric curves</h2>
<p class="pending">The configured metric stage has not produced
<code>metrics/metric_summary.csv</code> yet. Curves will appear here after generation
and benchmark evaluation complete; no placeholder values are fabricated.</p></section>"""
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>All-block/all-head test_5 ablation</title>
<style>
:root{{--ink:#18212a;--muted:#626c76;--line:#d6dce1;--paper:#f5f7f7;--panel:#fff;
--accent:#146b5b}} *{{box-sizing:border-box}} body{{margin:0;background:var(--paper);
color:var(--ink);font:14px/1.45 Arial,sans-serif;letter-spacing:0}}
header{{padding:18px 28px;background:#27343f;color:#fff}} h1{{font-size:22px;margin:0 0 5px}}
header p{{margin:0;color:#dce2e7}} main{{max-width:1500px;margin:0 auto;padding:20px 24px 40px}}
section{{border-top:1px solid var(--line);padding:18px 0 24px}} h2{{font-size:17px;margin:0 0 12px}}
h3{{font-size:13px;margin:0 0 8px}} img{{display:block;width:100%;height:auto;background:#fff;
border:1px solid var(--line)}} .status{{font-size:13px;color:var(--muted);margin:0 0 16px}}
.toolbar{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:14px}} label{{font-size:12px;
font-weight:700}} select{{display:block;min-width:160px;margin-top:4px;padding:7px 9px;
border:1px solid #aeb7bf;background:#fff}} .chart-grid{{display:grid;
grid-template-columns:1fr 1fr;gap:14px}} .chart-grid>div{{background:#fff;
border:1px solid var(--line);padding:12px}} canvas{{width:100%;height:360px;display:block}}
.pending{{padding:26px;background:#fff;border:1px dashed #9ca7b0;color:var(--muted)}}
.note{{font-size:12px;color:var(--muted)}} code{{font-family:monospace}}
@media(max-width:900px){{.chart-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>All-block/all-head test_5 ablation</h1>
<p>Self-attention head output set to zero, one block/head at a time</p></header>
<main>
<p class="status">Metrics updated {html.escape(generated_at)}</p>
{metric_view}
</main>
<script>
const DATA={json.dumps(payload, ensure_ascii=False)};
function fillSelect(id, values) {{
  const el=document.getElementById(id); if(!el) return;
  values.forEach(v=>{{const o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o);}});
}}
function draw(canvasId, labels, values, xLabel, title) {{
  const canvas=document.getElementById(canvasId); if(!canvas) return;
  const ratio=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect();
  canvas.width=Math.max(600,rect.width*ratio); canvas.height=360*ratio;
  const c=canvas.getContext('2d'); c.scale(ratio,ratio);
  const w=canvas.width/ratio,h=canvas.height/ratio,p={{l:58,r:18,t:28,b:48}};
  c.clearRect(0,0,w,h); c.fillStyle='#fff';c.fillRect(0,0,w,h);
  const finite=values.filter(v=>Number.isFinite(v));
  if(!finite.length){{c.fillStyle='#6b747d';c.font='14px Arial';c.fillText('No metric values available',p.l,60);return;}}
  let lo=Math.min(...finite),hi=Math.max(...finite); if(lo===hi){{lo-=0.5;hi+=0.5}}
  const pad=(hi-lo)*0.12;lo-=pad;hi+=pad;
  const X=i=>p.l+i*(w-p.l-p.r)/Math.max(labels.length-1,1);
  const Y=v=>p.t+(hi-v)*(h-p.t-p.b)/(hi-lo);
  c.strokeStyle='#d8dde2';c.lineWidth=1;
  for(let i=0;i<5;i++){{const y=p.t+i*(h-p.t-p.b)/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke();
    const v=hi-i*(hi-lo)/4;c.fillStyle='#59636d';c.font='11px Arial';c.fillText(v.toFixed(3),5,y+4);}}
  c.strokeStyle='#146b5b';c.lineWidth=2;c.beginPath();let started=false;
  values.forEach((v,i)=>{{if(!Number.isFinite(v)){{started=false;return}}const x=X(i),y=Y(v);
    if(!started){{c.moveTo(x,y);started=true}}else c.lineTo(x,y);}});c.stroke();
  values.forEach((v,i)=>{{if(!Number.isFinite(v))return;c.fillStyle='#146b5b';c.beginPath();c.arc(X(i),Y(v),3.2,0,Math.PI*2);c.fill();}});
  c.fillStyle='#26313a';c.font='11px Arial';c.textAlign='center';
  labels.forEach((label,i)=>{{if(i%2===0)c.fillText(label,X(i),h-25);}});
  c.font='12px Arial';c.fillText(xLabel,(p.l+w-p.r)/2,h-7);
  c.textAlign='left';c.font='12px Arial';c.fillText(title,p.l,16);
}}
function update() {{
  const model=document.getElementById('model-select')?.value;
  const metric=document.getElementById('metric-select')?.value;
  if(!model||!metric)return;
  const block=Number(document.getElementById('block-select').value);
  const head=Number(document.getElementById('head-select').value);
  const grid=DATA.metrics[model][metric];
  draw('head-chart',DATA.heads,grid[DATA.blocks.indexOf(block)],'Head',DATA.metric_labels[metric]);
  draw('block-chart',DATA.blocks,grid.map(row=>row[DATA.heads.indexOf(head)]),'Block',DATA.metric_labels[metric]);
}}
fillSelect('block-select',DATA.blocks);fillSelect('head-select',DATA.heads);
['model-select','metric-select','block-select','head-select'].forEach(id=>document.getElementById(id)?.addEventListener('change',update));
window.addEventListener('resize',update);update();
</script></body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
