#!/usr/bin/env python3
"""Build a compact filterable video and metric gallery for the pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from summarize_head_role_dose_control import METRICS


MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def link(path: Path, destination: Path) -> str:
    path = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != path:
            destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace {destination}")
    if not destination.exists():
        destination.symlink_to(path)
    return destination.name


def number(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if pd.notna(value) else None


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    gallery = root / "_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "analysis" / "per_video_metrics.csv")
    paired = pd.read_csv(root / "analysis" / "paired_vs_baseline.csv")
    harm_by_video = {
        row["output_video"]: {
            metric.name: number(row.get(f"{metric.name}_harm"))
            for metric in METRICS
        }
        for row in paired.to_dict(orient="records")
    }
    records = []
    for row in frame.to_dict(orient="records"):
        video = Path(row["output_video"])
        if row["kind"] == "baseline":
            asset_dir = gallery / "baseline-assets" / row["model"] / f"seed-{int(row['seed']):06d}"
            asset_dir.mkdir(parents=True, exist_ok=True)
            name = link(video, asset_dir / f"{row['case_id']}.mp4")
            video_url = f"baseline-assets/{row['model']}/seed-{int(row['seed']):06d}/{name}"
        else:
            video_url = os.path.relpath(video, gallery)
        records.append(
            {
                "kind": row["kind"],
                "model": row["model"],
                "model_label": MODEL_LABELS[row["model"]],
                "seed": int(row["seed"]),
                "case_id": row["case_id"],
                "subset_id": row["subset_id"] if pd.notna(row["subset_id"]) else "",
                "role": row["role"],
                "k": int(row["k"]),
                "replicate": int(row["replicate"]),
                "matching": row["matching"],
                "start": int(row["denoise_start"]),
                "end": int(row["denoise_end"]),
                "caption": row["caption"],
                "video": video_url,
                "metrics": {metric.name: number(row.get(metric.name)) for metric in METRICS},
                "harm": harm_by_video.get(str(video), {}),
            }
        )
    metric_defs = [
        {"name": metric.name, "direction": metric.direction} for metric in METRICS
    ]
    data = {
        "records": records,
        "models": list(MODEL_LABELS),
        "model_labels": MODEL_LABELS,
        "metric_definitions": metric_defs,
        "note": (
            "正 harm 表示该消融相对同模型、同 seed、同 source baseline 使指标变差；"
            "WMReward surprise 已按越低越好转换符号。k=8 是近似深度匹配，"
            "k=5 是 S/T/C 共享 block 上的完全匹配。"
        ),
    }
    atomic_write(
        gallery / "data.json",
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
    )
    atomic_write(gallery / "index.html", HTML)
    print(f"[dose-gallery] records={len(records)} output={gallery / 'index.html'}")


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head Role Dose-Control Pilot</title>
<style>
:root{--bg:#101315;--panel:#181d20;--line:#343b40;--text:#eef1f2;--muted:#a6afb5;--accent:#56bca6;--bad:#ff998f;--good:#72d4a5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#101315f2;border-bottom:1px solid var(--line);padding:12px 16px}
h1,h2,p{margin:0}h1{font-size:20px}.controls{display:flex;flex-wrap:wrap;gap:8px;margin-top:9px}
label{display:grid;gap:3px;color:var(--muted);font-size:11px}select{min-width:120px;padding:6px;background:#242a2e;color:var(--text);border:1px solid var(--line)}
.note{margin-top:8px;color:var(--muted)}main{padding:14px 16px}.model{margin:22px 0;border-top:3px solid var(--accent)}
.links{display:flex;gap:12px;margin-top:8px}.links a{color:var(--accent)}
.model h2{padding:9px 0;font-size:22px}.grid{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:8px}
.card{border:1px solid var(--line);background:var(--panel);min-width:0}.card h3{margin:0;padding:6px 8px;font-size:13px;background:#242a2e}
video{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#050606}.meta{padding:5px 8px;color:var(--muted);font-size:11px}
.missing{display:grid;place-items:center;aspect-ratio:7/4;color:#788289;background:#1c2225}
.metrics{margin-top:22px;overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:5px 7px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
thead th{background:#242a2e}.bad{color:var(--bad)}.good{color:var(--good)}
@media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(220px,1fr))}}@media(max-width:620px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>S/T/C 等数量、深度匹配消融</h1><div class="controls">
<label>Case<select id="case"></select></label><label>Seed<select id="seed"></select></label>
<label>去噪阶段<select id="stage"></select></label><label>匹配<select id="matching"></select></label>
<label>Replicate<select id="rep"></select></label></div><p class="note" id="note"></p>
<nav class="links"><a href="../analysis/conclusions.md">自动结论</a><a href="../analysis/role_harm_case_bootstrap.csv">角色汇总 CSV</a><a href="../analysis/matched_role_contrasts.csv">配对对比 CSV</a><a href="../analysis/per_video_metrics.csv">逐视频指标 CSV</a></nav></header>
<main><section><h2>聚合结果曲线</h2><p class="note">每张图均按模型单独统计；误差条为 source-case cluster bootstrap 95% CI。正值表示消融造成退化。</p>
<div class="grid">
<article class="card"><h3>Wan+LoRA · k=8</h3><img style="width:100%" src="../analysis/plots/wan_lora_approx_depth_primary_harm.png"></article>
<article class="card"><h3>Wan+LoRA · k=5</h3><img style="width:100%" src="../analysis/plots/wan_lora_exact_block_primary_harm.png"></article>
<article class="card"><h3>Wan+xSSC · k=8</h3><img style="width:100%" src="../analysis/plots/xssc_approx_depth_primary_harm.png"></article>
<article class="card"><h3>Wan+xSSC · k=5</h3><img style="width:100%" src="../analysis/plots/xssc_exact_block_primary_harm.png"></article>
<article class="card"><h3>PhysRVG · k=8</h3><img style="width:100%" src="../analysis/plots/physrvg_approx_depth_primary_harm.png"></article>
<article class="card"><h3>PhysRVG · k=5</h3><img style="width:100%" src="../analysis/plots/physrvg_exact_block_primary_harm.png"></article>
</div></section><div id="main"></div></main><script>
let D,R;const q=x=>document.getElementById(x),roles=["baseline","S","T","C"];
function opts(id,values,fmt=x=>x){const e=q(id);e.innerHTML=values.map(x=>`<option value="${x}">${fmt(x)}</option>`).join("")}
function key(r){return [r.case_id,r.seed,`${r.start}-${r.end}`,r.matching,r.replicate].join("|")}
function score(x){return x===null||x===undefined?"NA":Number(x).toPrecision(4)}
function current(){return {case_id:q("case").value,seed:+q("seed").value,stage:q("stage").value,matching:q("matching").value,rep:+q("rep").value}}
function render(){const c=current(), [start,end]=c.stage.split("-").map(Number);let html="";
 for(const model of D.models){const rows=R.filter(r=>r.model===model&&r.case_id===c.case_id&&r.seed===c.seed);
  const base=rows.find(r=>r.kind==="baseline");const selected=rows.filter(r=>r.kind==="ablation"&&r.start===start&&r.end===end&&r.matching===c.matching&&r.replicate===c.rep);
  html+=`<section class="model"><h2>${D.model_labels[model]} · seed ${c.seed}</h2><div class="grid">`;
  for(const role of roles){const r=role==="baseline"?base:selected.find(x=>x.role===role);
   html+=`<article class="card"><h3>${role==="baseline"?"Baseline":`消融 ${role}`}</h3>`;
   html+=r?`<video controls muted playsinline preload="metadata" src="${r.video}"></video><div class="meta">${r.kind==="baseline"?"未消融":`${r.subset_id} · ${r.start}–${r.end}`}</div>`:`<div class="missing">无对应结果</div>`;html+="</article>"}
  html+="</div></section>";
 }
 const metricRows=D.metric_definitions.map(m=>{let cells="";
  for(const model of D.models){const rows=R.filter(r=>r.model===model&&r.case_id===c.case_id&&r.seed===c.seed);
   const base=rows.find(r=>r.kind==="baseline");const selected=rows.filter(r=>r.kind==="ablation"&&r.start===start&&r.end===end&&r.matching===c.matching&&r.replicate===c.rep);
   for(const role of roles){const r=role==="baseline"?base:selected.find(x=>x.role===role);const v=r?(role==="baseline"?r.metrics[m.name]:r.harm[m.name]):null;
    cells+=`<td class="${role!=="baseline"&&v>0?"bad":role!=="baseline"&&v<0?"good":""}">${score(v)}</td>`}}
  return `<tr><td>${m.name}<br><small>${m.direction}</small></td>${cells}</tr>`}).join("");
 let heads="";for(const model of D.models)for(const role of roles)heads+=`<th>${D.model_labels[model]}<br>${role}${role==="baseline"?" score":" harm"}</th>`;
 html+=`<section class="metrics"><h2>全部指标</h2><table><thead><tr><th>指标</th>${heads}</tr></thead><tbody>${metricRows}</tbody></table></section>`;
 q("main").innerHTML=html;
}
fetch("data.json").then(x=>x.json()).then(d=>{D=d;R=d.records;q("note").textContent=d.note;
 opts("case",[...new Set(R.map(x=>x.case_id))].sort());opts("seed",[...new Set(R.map(x=>x.seed))].sort((a,b)=>a-b));
 opts("stage",[...new Set(R.filter(x=>x.kind==="ablation").map(x=>`${x.start}-${x.end}`))].sort(),x=>x.replace("-","–"));
 opts("matching",[...new Set(R.filter(x=>x.kind==="ablation").map(x=>x.matching))]);
 q("matching").value="approx_depth";function reps(){const c=current();const rs=[...new Set(R.filter(x=>x.kind==="ablation"&&x.matching===c.matching).map(x=>x.replicate))].sort();opts("rep",rs)}
 q("matching").onchange=()=>{reps();render()};for(const id of ["case","seed","stage","rep"])q(id).onchange=render;reps();render()});
</script></body></html>
"""


if __name__ == "__main__":
    main()
