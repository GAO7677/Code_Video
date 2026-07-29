#!/usr/bin/env python3
"""Build a gallery for aggregate-T heads rejected by confidence filters."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
GROUP_NAMES = {
    "margin_only_boundary": "仅 margin 未通过：边界样本",
    "support_only_boundary": "仅 support 未通过：边界样本",
    "margin_and_support_boundary": "margin 与 support 均未通过：边界样本",
    "high_score_rejected": "高 score_T 但被筛掉",
    "stable_reference": "稳定 T 参考",
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

    with args.heads.expanduser().resolve().open(
        encoding="utf-8", newline=""
    ) as handle:
        source_rows = list(csv.DictReader(handle))

    case = str(selection["case"])
    rows = []
    for item in source_rows:
        model = item["model"]
        block = int(item["block"])
        head = int(item["head"])
        key = item["selection_key"]
        figure = (
            heatmaps
            / model
            / case
            / f"{key}_block{block:02d}_head{head:02d}.png"
        )
        rows.append(
            {
                "model": model,
                "selection_key": key,
                "group": item["selection_reason"],
                "failure": item["failure"],
                "block": block,
                "head": head,
                "aggregate_role": item["aggregate_role"],
                "runner_up_role": item["runner_up_role"],
                "score_T": float(item["score_T"]),
                "runner_up_score": float(item["runner_up_score"]),
                "margin": float(item["margin"]),
                "support": float(item["support"]),
                "support_ci95_low": float(item["support_ci95_low"]),
                "support_ci95_high": float(item["support_ci95_high"]),
                "valid_trajectory_samples": int(
                    item["valid_trajectory_samples"]
                ),
                "total_samples": int(item["total_samples"]),
                "figure": (
                    f"figures/{figure.relative_to(heatmaps).as_posix()}"
                    if figure.is_file()
                    else None
                ),
            }
        )

    available_models = [
        model
        for model in MODEL_NAMES
        if any(row["model"] == model and row["figure"] for row in rows)
    ]
    pending_models = [
        model for model in MODEL_NAMES if model not in available_models
    ]
    if not available_models:
        raise RuntimeError("no rendered heatmaps are available")
    payload = {
        "case": case,
        "seed": int(selection["representative_seed"]),
        "models": available_models,
        "pending_models": pending_models,
        "model_names": MODEL_NAMES,
        "group_names": GROUP_NAMES,
        "thresholds": selection["thresholds"],
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
<title>被筛选 T 候选的 all-token Q@K</title>
<style>
:root{{--bg:#111416;--panel:#1b2023;--line:#3a4247;--text:#f2f4f5;--muted:#abb4b9;--good:#54b88e;--warn:#d7a44a;--bad:#d36b64}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;padding:13px 18px;background:#111416f5;border-bottom:1px solid var(--line)}}
h1,h2,p{{margin:0}}h1{{font-size:21px}}.note{{margin-top:5px;color:var(--muted)}}
.controls{{display:flex;gap:9px;align-items:end;flex-wrap:wrap;margin-top:10px}}
label{{display:grid;gap:3px;color:var(--muted);font-size:11px}}
select{{min-width:190px;padding:7px 9px;border:1px solid var(--line);background:#252c30;color:var(--text)}}
.count{{margin-left:auto;color:#70c9b5;font-weight:750}}main{{padding:16px 18px}}
.summary{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-bottom:14px}}
.summary div{{border-left:3px solid var(--warn);background:#1b2023;padding:8px 10px}}
.summary b{{display:block;font-size:15px}}.summary span{{color:var(--muted);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:10px}}
article{{border:1px solid var(--line);background:var(--panel)}}article h2{{padding:8px 10px;background:#252c30;font-size:14px}}
img{{display:block;width:100%;height:auto;background:#07090a}}.missing{{display:grid;place-items:center;aspect-ratio:11/12;color:#c7a96a}}
.meta{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 12px;padding:8px 10px;color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}}
.reason{{grid-column:1/-1;color:var(--warn);font-weight:700}}.pass .reason{{color:var(--good)}}.failed{{color:var(--bad)}}.passed{{color:var(--good)}}
@media(max-width:760px){{.summary{{grid-template-columns:1fr}}.grid{{grid-template-columns:1fr}}.count{{width:100%;margin-left:0}}}}
</style></head><body>
<header><h1>被置信度规则筛掉的聚合 T 候选：all-token Q@K</h1>
<p class="note">不做跨模型交集；同一 case、seed 851，去噪步 5/15/25/35。左列是 raw QK/√d，右列是精确 softmax attention；5,824 tokens 池化为 512×512。</p>
<p class="note">“聚合 T 候选”表示平均 score_T 为五类最高；最终仍需 margin ≥ 0.08 且 support ≥ 0.50。页面按失败原因选择边界样本，并补充高 score_T 淘汰样本和稳定 T 参考；模型下拉框只列出已经完成前向的结果。</p>
<div class="controls"><label>模型<select id="model"></select></label><label>筛选组<select id="group"></select></label><label>排序<select id="sort"><option value="selection">选择顺序</option><option value="score">score_T 高到低</option><option value="margin">margin 高到低</option><option value="support">support 高到低</option><option value="position">Block / Head</option></select></label><span class="count" id="count"></span></div>
</header><main><div class="summary"><div><b>候选条件</b><span>score_T 是聚合最高类别分数</span></div><div><b>margin 门槛</b><span>score_T − 第二名 ≥ 0.08</span></div><div><b>support 门槛</b><span>有效轨迹样本中硬标签 T 的比例 ≥ 0.50</span></div></div><div class="grid" id="grid"></div></main>
<script>
const D={data},q=id=>document.getElementById(id);
const esc=x=>String(x).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
q("model").innerHTML=D.models.map(x=>`<option value="${{x}}">${{D.model_names[x]}}</option>`).join("");
q("group").innerHTML=`<option value="all">全部筛选组</option>`+Object.entries(D.group_names).map(([k,v])=>`<option value="${{k}}">${{v}}</option>`).join("");
function verdict(value,threshold){{return value>=threshold?`<span class="passed">通过</span>`:`<span class="failed">未通过</span>`}}
function render(){{
  const model=q("model").value,group=q("group").value,mode=q("sort").value;
  let rows=D.rows.filter(x=>x.model===model&&(group==="all"||x.group===group));
  if(mode==="score")rows.sort((a,b)=>b.score_T-a.score_T);
  else if(mode==="margin")rows.sort((a,b)=>b.margin-a.margin);
  else if(mode==="support")rows.sort((a,b)=>b.support-a.support);
  else if(mode==="position")rows.sort((a,b)=>a.block-b.block||a.head-b.head);
  q("count").textContent=`${{D.model_names[model]}} · 显示 ${{rows.length}} 个 head`;
  q("grid").innerHTML=rows.map(x=>{{const media=x.figure?`<a href="${{esc(x.figure)}}"><img loading="lazy" src="${{esc(x.figure)}}" alt="B${{x.block}} H${{x.head}} QK"></a>`:`<div class="missing">等待该模型前向与渲染</div>`;return`<article class="${{x.group==="stable_reference"?"pass":""}}"><h2>Block ${{String(x.block).padStart(2,"0")}} · Head ${{String(x.head).padStart(2,"0")}}</h2>${{media}}<div class="meta"><span class="reason">${{D.group_names[x.group]}}</span><span>score_T ${{x.score_T.toFixed(4)}}</span><span>第二名 ${{x.runner_up_role}} · ${{x.runner_up_score.toFixed(4)}}</span><span>margin ${{x.margin.toFixed(4)}} · ${{verdict(x.margin,D.thresholds.aggregate_margin)}}</span><span>support ${{x.support.toFixed(3)}} · ${{verdict(x.support,D.thresholds.aggregate_support)}}</span><span>support 95% CI [${{x.support_ci95_low.toFixed(3)}}, ${{x.support_ci95_high.toFixed(3)}}]</span><span>轨迹有效样本 ${{x.valid_trajectory_samples}} / ${{x.total_samples}}</span></div></article>`}}).join("");
}}
for(const id of["model","group","sort"])q(id).onchange=render;render();
</script></body></html>"""
    temporary = output / f".index.html.{os.getpid()}.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output / "index.html")
    print(output / "index.html")


if __name__ == "__main__":
    main()
