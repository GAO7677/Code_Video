#!/usr/bin/env python3
"""Build a Solid Mechanics raw-metric-gap video comparison page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_SOURCE_PAGE = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physiciq/index.html"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "solid-mechanics-xssc-vs-physrvg"
)

METRICS = [
    {"key": "videophy2_pc_raw", "label": "VideoPhy2 PC raw", "direction": "↑"},
    {"key": "cosmos_reason1", "label": "Cosmos Reason", "direction": "↑"},
    {
        "key": "physics_iq_with_context",
        "label": "Physics-IQ ctx",
        "direction": "↑",
    },
    {
        "key": "physics_iq_without_context",
        "label": "Physics-IQ no ctx",
        "direction": "↑",
    },
]

METHODS = {
    "xssc": {
        "key": "full_sa_no_object_xssc_loss_dinov3_movic_step50000",
        "step": 500,
        "label": "Full-SA + No-Object + xSSC Loss · step500",
    },
    "physrvg_off": {
        "key": "physrvg_test5_lora_off",
        "step": 40,
        "label": "PhysRVG finetuned DiT · LoRA OFF",
    },
    "physrvg_on": {
        "key": "physrvg_test5_lora_on",
        "step": 40,
        "label": "PhysRVG finetuned DiT + LoRA",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-page", type=Path, default=DEFAULT_SOURCE_PAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_dashboard_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});\s*const caseSelect=", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find dashboard payload in {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dashboard object in {path}")
    return payload


def find_record(
    payload: dict[str, Any], method_key: str, step: int
) -> dict[str, Any]:
    matches = [
        record
        for record in payload["records"]
        if record["method_key"] == method_key and int(record["step"]) == step
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one record for method={method_key} step={step}, "
            f"found {len(matches)}"
        )
    return matches[0]


def metric_value(record: dict[str, Any], stem: str, key: str) -> float:
    value = record.get("metrics", {}).get(stem, {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Missing numeric metric method={record['method_key']} "
            f"step={record['step']} case={stem} metric={key}"
        )
    return float(value)


def build_data(payload: dict[str, Any]) -> dict[str, Any]:
    records = {
        name: find_record(payload, spec["key"], int(spec["step"]))
        for name, spec in METHODS.items()
    }
    cases: list[dict[str, Any]] = []
    for case in payload["cases"]:
        stem = str(case["stem"])
        if "Solid_Mechanics" not in stem:
            continue
        values: dict[str, dict[str, float | str]] = {}
        for metric in METRICS:
            key = metric["key"]
            xssc = metric_value(records["xssc"], stem, key)
            physrvg_off = metric_value(records["physrvg_off"], stem, key)
            physrvg_on = metric_value(records["physrvg_on"], stem, key)
            delta_off = abs(xssc - physrvg_off)
            delta_on = abs(xssc - physrvg_on)
            values[key] = {
                "xssc": xssc,
                "physrvg_off": physrvg_off,
                "physrvg_on": physrvg_on,
                "delta_off": delta_off,
                "delta_on": delta_on,
                "rank_gap": max(delta_off, delta_on),
                "max_reference": (
                    "PhysRVG LoRA OFF" if delta_off >= delta_on else "PhysRVG +LoRA"
                ),
            }
        cases.append(
            {
                "stem": stem,
                "prompt": case.get("prompt", ""),
                "gt": case["gt"],
                "videos": {
                    name: record["videos"][stem]
                    for name, record in records.items()
                },
                "metrics": values,
            }
        )
    if len(cases) != 39:
        raise ValueError(f"Expected 39 Solid Mechanics cases, found {len(cases)}")
    return {"metrics": METRICS, "methods": METHODS, "cases": cases}


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Solid Mechanics · xSSC Loss vs PhysRVG</title>
  <style>
    :root{--bg:#f2f4f3;--paper:#fff;--ink:#172126;--muted:#657278;
      --line:#d5dcde;--xssc:#6f4ead;--off:#315c87;--on:#0b6e4f;--best:#fff2c7}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}
    .toolbar{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:9px;
      min-height:60px;padding:10px 16px;background:rgba(255,255,255,.97);
      border-bottom:1px solid var(--line)}
    .toolbar a{font-weight:800;color:#006d77;text-decoration:none}.title{margin-right:auto;
      font-size:15px;font-weight:900}select,button{height:38px;border:1px solid var(--line);
      border-radius:6px;background:#fff;color:var(--ink);font:inherit}
    select{max-width:370px;padding:0 9px}button{min-width:39px;padding:0 10px;cursor:pointer}
    main{max-width:1800px;margin:auto;padding:18px}.intro{margin-bottom:15px;padding:15px 17px;
      background:var(--paper);border:1px solid var(--line);border-radius:8px}
    h1{margin:0 0 6px;font-size:22px}.intro p,.prompt,.note{margin:0;color:var(--muted);
      font-size:13px;line-height:1.55}.summary-grid{display:grid;
      grid-template-columns:minmax(560px,1.05fr) minmax(620px,1fr);gap:14px;align-items:start}
    .panel{background:var(--paper);border:1px solid var(--line);border-radius:8px;overflow:hidden}
    .panel-head{padding:12px 14px;border-bottom:1px solid var(--line)}
    .panel-head h2{margin:0 0 4px;font-size:16px}.rank-wrap{max-height:410px;overflow:auto}
    table{width:100%;border-collapse:separate;border-spacing:0;font-size:12px;
      font-variant-numeric:tabular-nums}th,td{padding:7px 8px;border-right:1px solid #e4e9ea;
      border-bottom:1px solid #e4e9ea;text-align:right;white-space:nowrap}
    th{position:sticky;top:0;z-index:2;background:#edf1f2;color:#45545b}th.case,td.case{
      text-align:left;max-width:360px;overflow:hidden;text-overflow:ellipsis}tbody tr{cursor:pointer}
    tbody tr:hover{background:#f4f8f8}tbody tr.active{background:var(--best);font-weight:850}
    .case-detail{padding:14px}.case-detail h2{margin:0 0 7px;font-size:16px;overflow-wrap:anywhere}
    .video-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px;margin-top:16px}
    .video-card{padding:9px;background:var(--paper);border:1px solid var(--line);border-radius:8px}
    .video-label{min-height:38px;padding:1px 2px 7px;font-size:12px;font-weight:900;
      display:flex;align-items:center}.video-label.xssc{color:var(--xssc)}
    .video-label.off{color:var(--off)}.video-label.on{color:var(--on)}
    video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111617}
    .metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:13px}
    .metric-card{padding:11px;background:var(--paper);border:1px solid var(--line);border-radius:8px}
    .metric-card.selected{border:2px solid #a06a00;background:#fffaf0}.metric-card h3{
      margin:0 0 8px;font-size:13px}.metric-card dl{display:grid;grid-template-columns:1fr auto;
      gap:5px 9px;margin:0;font-size:12px}.metric-card dt{color:var(--muted)}
    .metric-card dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;font-weight:800}
    .gap{color:#9b3a31}.footer{margin:16px 0 5px;color:var(--muted);font-size:11px}
    @media(max-width:1200px){.summary-grid{grid-template-columns:1fr}.video-grid,
      .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:700px){.toolbar{flex-wrap:wrap}.title{width:100%;order:-1}main{padding:10px}
      select{max-width:calc(100vw - 22px)}.video-grid,.metric-grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="toolbar">
    <a href="../">返回总览</a><div class="title">Solid Mechanics · xSSC Loss vs PhysRVG</div>
    <select id="metric" aria-label="排序指标"></select>
    <select id="case" aria-label="case"></select>
    <button id="play" title="同步播放">▶</button><button id="pause" title="同步暂停">Ⅱ</button>
    <button id="replay" title="全部重新播放">↺</button>
  </div>
  <main>
    <section class="intro"><h1>Solid Mechanics 原始指标差距</h1>
      <p>共 39 个 case。四个指标分别独立排序，不做归一化；排名值为
      max(|xSSC − PhysRVG LoRA OFF|, |xSSC − PhysRVG +LoRA|)。表格同时保留两组原始绝对差值。</p>
    </section>
    <div class="summary-grid">
      <section class="panel"><div class="panel-head"><h2 id="rank-title"></h2>
        <p class="note">点击任意一行切换下方视频；差值越大排名越靠前。</p></div>
        <div class="rank-wrap"><table><thead><tr><th>#</th><th class="case">Case</th>
          <th>xSSC</th><th>LoRA OFF</th><th>|Δ OFF|</th><th>+LoRA</th><th>|Δ ON|</th>
          <th>最大差值</th></tr></thead><tbody id="ranking"></tbody></table></div></section>
      <section class="panel case-detail"><h2 id="case-title"></h2><p class="prompt" id="prompt"></p></section>
    </div>
    <section class="video-grid">
      <div class="video-card"><div class="video-label">GT · 49 frames @ 30 FPS</div><video id="gt" muted playsinline preload="metadata"></video></div>
      <div class="video-card"><div class="video-label xssc">Full-SA + No-Object + xSSC Loss · step500</div><video id="xssc" muted playsinline preload="metadata"></video></div>
      <div class="video-card"><div class="video-label off">PhysRVG finetuned DiT · LoRA OFF</div><video id="physrvg_off" muted playsinline preload="metadata"></video></div>
      <div class="video-card"><div class="video-label on">PhysRVG finetuned DiT + LoRA</div><video id="physrvg_on" muted playsinline preload="metadata"></video></div>
    </section>
    <section class="metric-grid" id="metric-grid"></section>
    <p class="footer">Generated from the existing PhysicIQ dashboard metrics. Built: __GENERATED__</p>
  </main>
  <script>
    const D=__DATA__;
    const metricSelect=document.getElementById('metric');
    const caseSelect=document.getElementById('case');
    const ranking=document.getElementById('ranking');
    let selectedStem='';
    function videos(){return [...document.querySelectorAll('video')]}
    function fmt(value){return Number.isInteger(value)?String(value):value.toFixed(2)}
    function rankedCases(){const key=metricSelect.value;return [...D.cases]
      .sort((a,b)=>b.metrics[key].rank_gap-a.metrics[key].rank_gap||a.stem.localeCompare(b.stem))}
    function renderCase(stem){
      const item=D.cases.find(row=>row.stem===stem);if(!item)return;selectedStem=stem;
      videos().forEach(video=>video.pause());
      document.getElementById('case-title').textContent=item.stem;
      document.getElementById('prompt').textContent=item.prompt;
      document.getElementById('gt').src=item.gt;
      Object.entries(item.videos).forEach(([key,path])=>document.getElementById(key).src=path);
      const grid=document.getElementById('metric-grid');grid.replaceChildren();
      D.metrics.forEach(spec=>{
        const v=item.metrics[spec.key];const card=document.createElement('article');
        card.className='metric-card'+(spec.key===metricSelect.value?' selected':'');
        card.innerHTML=`<h3>${spec.label} ${spec.direction}</h3><dl>
          <dt>xSSC</dt><dd>${fmt(v.xssc)}</dd><dt>PhysRVG LoRA OFF</dt><dd>${fmt(v.physrvg_off)}</dd>
          <dt>|Δ OFF|</dt><dd class="gap">${fmt(v.delta_off)}</dd>
          <dt>PhysRVG +LoRA</dt><dd>${fmt(v.physrvg_on)}</dd>
          <dt>|Δ ON|</dt><dd class="gap">${fmt(v.delta_on)}</dd>
          <dt>最大差值来源</dt><dd>${v.max_reference}</dd></dl>`;grid.append(card);
      });
      [...ranking.querySelectorAll('tr')].forEach(row=>row.classList.toggle('active',row.dataset.stem===stem));
      caseSelect.value=stem;
    }
    function renderMetric(){
      const spec=D.metrics.find(item=>item.key===metricSelect.value);const rows=rankedCases();
      document.getElementById('rank-title').textContent=`${spec.label} ${spec.direction} · 原始差值排名`;
      ranking.replaceChildren();caseSelect.replaceChildren();
      rows.forEach((item,index)=>{
        const v=item.metrics[spec.key];caseSelect.add(new Option(`${String(index+1).padStart(2,'0')} · ${item.stem}`,item.stem));
        const tr=document.createElement('tr');tr.dataset.stem=item.stem;
        tr.innerHTML=`<td>${index+1}</td><td class="case" title="${item.stem}">${item.stem}</td>
          <td>${fmt(v.xssc)}</td><td>${fmt(v.physrvg_off)}</td><td class="gap">${fmt(v.delta_off)}</td>
          <td>${fmt(v.physrvg_on)}</td><td class="gap">${fmt(v.delta_on)}</td><td class="gap">${fmt(v.rank_gap)}</td>`;
        tr.onclick=()=>renderCase(item.stem);ranking.append(tr);
      });
      renderCase(rows[0].stem);
    }
    D.metrics.forEach(spec=>metricSelect.add(new Option(`${spec.label} ${spec.direction}`,spec.key)));
    metricSelect.onchange=renderMetric;caseSelect.onchange=()=>renderCase(caseSelect.value);
    document.getElementById('play').onclick=()=>videos().forEach(video=>video.play().catch(()=>{}));
    document.getElementById('pause').onclick=()=>videos().forEach(video=>video.pause());
    document.getElementById('replay').onclick=()=>videos().forEach(video=>{video.currentTime=0;video.play().catch(()=>{})});
    renderMetric();
  </script>
</body>
</html>
'''


def main() -> None:
    args = parse_args()
    data = build_data(load_dashboard_payload(args.source_page.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    output = HTML_TEMPLATE.replace("__DATA__", encoded).replace(
        "__GENERATED__", "2026-08-11 UTC"
    )
    output_path = args.output_dir / "index.html"
    output_path.write_text(output, encoding="utf-8")
    print(output_path)
    for metric in METRICS:
        key = metric["key"]
        ranked = sorted(
            data["cases"],
            key=lambda case: (-float(case["metrics"][key]["rank_gap"]), case["stem"]),
        )
        print(f"{metric['label']}:")
        for case in ranked[:5]:
            values = case["metrics"][key]
            print(
                f"  {case['stem']} gap={values['rank_gap']:.4f} "
                f"off={values['delta_off']:.4f} on={values['delta_on']:.4f}"
            )


if __name__ == "__main__":
    main()
