#!/usr/bin/env python3
"""Build a filterable gallery for all common stable S/T/C all-token QK maps."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLE_NAMES = {
    "S": "S：同帧局部空间 Head",
    "T": "T：跨帧运动轨迹 Head",
    "C": "C：对象与上下文交互 Head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--heatmap-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def ensure_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"refusing to replace non-link path: {destination}")
    destination.symlink_to(source, target_is_directory=True)


def main() -> None:
    args = parse_args()
    selection = json.loads(
        args.selection.expanduser().resolve().read_text(encoding="utf-8")
    )
    heatmaps = args.heatmap_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ensure_link(heatmaps, output / "figures")

    with args.heads.expanduser().resolve().open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    case = str(selection["case"])
    rows = []
    for item in source_rows:
        role = item["role"]
        block = int(item["block"])
        head = int(item["head"])
        model_data = {}
        for model in MODELS:
            name = f"{role}_{int(item['role_index']):03d}_B{block:02d}H{head:02d}"
            path = heatmaps / model / case / f"{name}_block{block:02d}_head{head:02d}.png"
            model_data[model] = {
                "figure": (
                    f"figures/{path.relative_to(heatmaps).as_posix()}"
                    if path.is_file()
                    else None
                ),
                "score": float(item[f"{model}_score"]),
                "margin": float(item[f"{model}_margin"]),
                "support": float(item[f"{model}_support"]),
            }
        rows.append(
            {
                "role": role,
                "role_index": int(item["role_index"]),
                "block": block,
                "head": head,
                "mean_score": float(item["cross_model_mean_score"]),
                "models": model_data,
            }
        )

    payload = {
        "case": case,
        "seed": int(selection["representative_seed"]),
        "models": list(MODELS),
        "model_names": MODEL_NAMES,
        "role_names": ROLE_NAMES,
        "role_counts": selection["role_counts"],
        "rows": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>公共稳定 S/T/C Head 的 all-token Q@K</title>
<style>
:root{{--bg:#111416;--panel:#1b2023;--line:#3a4247;--text:#f2f4f5;--muted:#abb4b9;--accent:#5fc4ae}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;padding:13px 18px;background:#111416f5;border-bottom:1px solid var(--line)}}
h1,h2,p{{margin:0}}h1{{font-size:21px}}.note{{margin-top:5px;color:var(--muted)}}
.controls{{display:flex;gap:9px;align-items:end;flex-wrap:wrap;margin-top:10px}}label{{display:grid;gap:3px;color:var(--muted);font-size:11px}}
select{{min-width:150px;padding:7px 9px;border:1px solid var(--line);background:#252c30;color:var(--text)}}.count{{margin-left:auto;color:var(--accent);font-weight:750}}
main{{padding:16px 18px}}.role-title{{padding:10px 0;border-bottom:3px solid var(--accent);font-size:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:10px;margin-top:12px}}
article{{border:1px solid var(--line);background:var(--panel)}}article h2{{padding:7px 9px;background:#252c30;font-size:14px}}
img{{display:block;width:100%;height:auto;background:#07090a}}.missing{{display:grid;place-items:center;aspect-ratio:11/12;color:#c7a96a}}
.meta{{display:flex;flex-wrap:wrap;gap:5px 12px;padding:7px 9px;color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}
@media(max-width:600px){{.grid{{grid-template-columns:1fr}}.count{{width:100%;margin-left:0}}}}
</style></head><body>
<header><h1>公共稳定 S/T/C Head：all-token Q@K</h1>
<p class="note">同一 case、seed 851；每个模型一次前向；去噪步 5/15/25/35。每张图左列为 raw QK/√d，右列为精确 softmax attention；5,824 tokens 池化到 512×512。</p>
<p class="note">S/T/C 是 22 seeds × 20 cases 聚合后在三个模型中类别一致的公共稳定 head。筛选只改变展示，不改变捕获数据。</p>
<div class="controls"><label>模型<select id="model"></select></label><label>类别<select id="role"></select></label><label>Block<select id="block"></select></label><label>排序<select id="sort"><option value="position">Block / Head</option><option value="score">类别分数</option></select></label><span class="count" id="count"></span></div>
</header><main><h2 class="role-title" id="title"></h2><div class="grid" id="grid"></div></main>
<script>
const D={data},q=id=>document.getElementById(id);
const esc=x=>String(x).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
q("model").innerHTML=D.models.map(x=>`<option value="${{x}}">${{D.model_names[x]}}</option>`).join("");
q("role").innerHTML=["S","T","C"].map(x=>`<option value="${{x}}">${{D.role_names[x]}} (${{D.role_counts[x]}})</option>`).join("");
q("block").innerHTML=`<option value="all">全部 Block</option>`+Array.from({{length:30}},(_,x)=>`<option value="${{x}}">Block ${{String(x).padStart(2,"0")}}</option>`).join("");
function render(){{
  const model=q("model").value,role=q("role").value,block=q("block").value;
  let rows=D.rows.filter(x=>x.role===role&&(block==="all"||x.block===Number(block)));
  rows.sort(q("sort").value==="score"?(a,b)=>b.models[model].score-a.models[model].score:(a,b)=>a.block-b.block||a.head-b.head);
  q("title").textContent=`${{D.model_names[model]}} · ${{D.role_names[role]}}`;
  q("count").textContent=`显示 ${{rows.length}} / ${{D.role_counts[role]}} heads`;
  q("grid").innerHTML=rows.map(x=>{{const m=x.models[model],media=m.figure?`<a href="${{esc(m.figure)}}"><img loading="lazy" src="${{esc(m.figure)}}" alt="B${{x.block}} H${{x.head}} QK"></a>`:`<div class="missing">等待该模型前向与渲染</div>`;return`<article><h2>${{role}} #${{x.role_index}} · Block ${{String(x.block).padStart(2,"0")}} · Head ${{String(x.head).padStart(2,"0")}}</h2>${{media}}<div class="meta"><span>score ${{m.score.toFixed(4)}}</span><span>margin ${{m.margin.toFixed(4)}}</span><span>support ${{m.support.toFixed(3)}}</span><span>三模型均值 ${{x.mean_score.toFixed(4)}}</span></div></article>`}}).join("");
}}
for(const id of["model","role","block","sort"])q(id).onchange=render;render();
</script></body></html>"""
    temporary = output / f".index.html.{os.getpid()}.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output / "index.html")
    print(output / "index.html")


if __name__ == "__main__":
    main()
