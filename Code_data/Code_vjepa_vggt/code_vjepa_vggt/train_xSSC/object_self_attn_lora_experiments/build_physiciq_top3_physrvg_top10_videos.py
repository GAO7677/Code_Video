#!/usr/bin/env python3
"""Build a video-only page for per-metric Top-10 PhysRVG gaps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from build_physiciq_physrvg_worst_case_dashboard import (
    DEFAULT_SOURCE_PAGE,
    load_dashboard_payload,
)
from build_physiciq_top3_physrvg_all_cases_dashboard import build_top3_data


DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physiciq-top3-vs-physrvg-top10-videos"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-page", type=Path, default=DEFAULT_SOURCE_PAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def metric_value(record: dict[str, Any], stem: str, key: str) -> float:
    value = record["metrics"][stem][key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid metric {key} for {stem}")
    return float(value)


def build_video_only_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = build_top3_data(payload)
    cases = {case["stem"]: case for case in data["cases"]}
    references = data["references"]
    sections: list[dict[str, Any]] = []
    for spec in data["metric_specs"]:
        metric_key = str(spec["key"])
        direction = str(spec["direction"])
        ranked_cases: list[dict[str, Any]] = []
        for stem, case in cases.items():
            off = metric_value(references["off"], stem, metric_key)
            on = metric_value(references["on"], stem, metric_key)
            reference_value = min(off, on) if direction == "lower" else max(off, on)
            largest_gap = -1.0
            for record in data["top_records"]:
                candidate = metric_value(record, stem, metric_key)
                largest_gap = max(largest_gap, abs(candidate - reference_value))
            ranked_cases.append(
                {
                    "stem": stem,
                    "sort_gap": largest_gap,
                    "gt": case["gt"],
                    "top_videos": [
                        record["videos"][stem] for record in data["top_records"]
                    ],
                    "physrvg_off": references["off"]["videos"][stem],
                    "physrvg_on": references["on"]["videos"][stem],
                }
            )
        ranked_cases.sort(key=lambda row: (-float(row["sort_gap"]), str(row["stem"])))
        sections.append(
            {
                "key": metric_key,
                "label": str(spec["label"]),
                "cases": [
                    {key: value for key, value in row.items() if key != "sort_gap"}
                    for row in ranked_cases[:10]
                ],
            }
        )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methods": [
            {
                "label": str(row["label"]),
                "step": int(row["step"]),
                "color": str(row["color"]),
            }
            for row in data["top3"]
        ],
        "sections": sections,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PhysicIQ · 四指标差距 Top 10 · 纯视频</title>
  <style>
    :root{--fog:#e9eff0;--paper:#fbfcfc;--ink:#18282e;--muted:#67787f;--line:#ccd8db;
      --deep:#14333e;--off:#315c87;--on:#0b6e4f;--accent:#b9513b;--shadow:0 8px 25px rgba(15,43,54,.09)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--fog);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}a{color:inherit}.hero{padding:22px clamp(14px,4vw,58px) 24px;background:var(--deep);color:#f7fbfc}
    .links{display:flex;flex-wrap:wrap;gap:15px}.links a{text-decoration:none;color:#a9d7da;font-size:12px;font-weight:850}
    h1{margin:18px 0 7px;font:850 clamp(28px,4.3vw,52px)/1 "Arial Narrow","Roboto Condensed",sans-serif;letter-spacing:-.025em}
    .hero p{max-width:980px;margin:0;color:#bfd3d8;font-size:12px;line-height:1.6}.tabs{position:sticky;top:0;z-index:20;display:grid;
      grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border-bottom:1px solid var(--line);box-shadow:0 5px 18px rgba(15,43,54,.1)}
    .tabs a{padding:12px 10px;background:rgba(251,252,252,.97);text-align:center;text-decoration:none;font-size:11px;font-weight:900}
    .tabs a:hover{background:#fff3e8;color:#8d392b}main{max-width:1920px;margin:auto;padding:20px clamp(9px,2vw,29px) 70px}
    .metric-section{margin-bottom:32px;scroll-margin-top:58px}.section-head{display:flex;align-items:end;justify-content:space-between;gap:15px;margin-bottom:10px}
    .section-head h2{margin:0;font:850 23px/1 "Arial Narrow",sans-serif}.section-head span{color:var(--muted);font-size:11px}
    .case-strip{margin-bottom:13px;background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}
    .case-head{display:flex;align-items:center;gap:12px;padding:10px 12px;border-bottom:1px solid var(--line)}.rank{flex:none;width:34px;
      font:900 24px/1 "Arial Narrow",sans-serif;color:#a0adb1}.case-head h3{min-width:0;margin:0;overflow:hidden;text-overflow:ellipsis;
      white-space:nowrap;font-size:12px}.case-head button{margin-left:auto;flex:none;height:30px;padding:0 9px;border:1px solid var(--line);border-radius:4px;
      background:#fff;color:var(--ink);cursor:pointer;font-size:10px;font-weight:850}.film-scroll{overflow-x:auto}.film{display:grid;
      grid-template-columns:repeat(6,minmax(225px,1fr));min-width:1420px;gap:1px;background:var(--line)}.frame{padding:8px;background:var(--paper)}
    .label{display:flex;justify-content:space-between;gap:7px;min-height:39px;padding:1px 2px 6px;font-size:10px;font-weight:900;line-height:1.3}
    .label span{max-width:80%}.label em{color:var(--muted);font-style:normal;text-align:right;font-weight:650}video{display:block;width:100%;aspect-ratio:16/9;
      object-fit:contain;background:#0c171b}.replay-all{position:fixed;right:20px;bottom:18px;z-index:30;height:48px;padding:0 16px;border:0;
      border-radius:24px;background:var(--accent);color:#fff;box-shadow:0 8px 23px rgba(102,39,28,.3);cursor:pointer;font-weight:900}
    .footer{color:var(--muted);font-size:10px}button:focus-visible,a:focus-visible{outline:3px solid #dbac46;outline-offset:2px}
    @media(max-width:760px){.tabs{grid-template-columns:repeat(2,1fr)}.metric-section{scroll-margin-top:104px}.section-head{display:block}.section-head span{display:block;margin-top:5px}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  </style>
</head>
<body>
  <header class="hero"><div class="links"><a href="../">← 返回 8844 总览</a>
    <a href="../physiciq-top3-vs-physrvg-all-cases/">返回完整 67-case 对比</a></div>
    <h1>四项主要指标 · 差距 Top 10</h1>
    <p>每个指标独立选择 10 个不重复 case；每个 case 只展示 GT、综合 Top 3 与 PhysRVG LoRA OFF/+LoRA。指标数值与表格已全部隐藏。</p>
  </header>
  <nav id="tabs" class="tabs"></nav>
  <main id="content"></main>
  <button id="replay-all" class="replay-all">↺ 重播已加载视频</button>
  <script>
    const D=__DATA__;const tabs=document.getElementById('tabs');const content=document.getElementById('content');
    function videoFrame(label,sub,path,color){return `<div class="frame"><div class="label" style="color:${color}"><span>${label}</span><em>${sub}</em></div>
      <video data-src="${path}" muted playsinline controls preload="none"></video></div>`}
    function loadStrip(strip){strip.querySelectorAll('video[data-src]').forEach(video=>{video.src=video.dataset.src;delete video.dataset.src;video.load()})}
    D.sections.forEach(section=>{const anchor=document.createElement('a');anchor.href=`#${section.key}`;anchor.textContent=`${section.label} · Top 10`;tabs.append(anchor);
      const root=document.createElement('section');root.id=section.key;root.className='metric-section';root.innerHTML=`<div class="section-head"><h2>${section.label} · Top 10</h2>
        <span>10 个 case · 仅视频</span></div>`;section.cases.forEach((item,index)=>{const strip=document.createElement('article');strip.className='case-strip';
        const topFrames=item.top_videos.map((path,i)=>videoFrame(D.methods[i].label,`综合 #${i+1} · step ${D.methods[i].step}`,path,D.methods[i].color)).join('');
        strip.innerHTML=`<div class="case-head"><span class="rank">${String(index+1).padStart(2,'0')}</span><h3>${item.stem}</h3><button>↺ 重播本组</button></div>
          <div class="film-scroll"><div class="film">${videoFrame('GT','49f · 30 FPS',item.gt,'var(--ink)')}${topFrames}
          ${videoFrame('PhysRVG finetuned DiT · LoRA OFF','inference 40',item.physrvg_off,'var(--off)')}
          ${videoFrame('PhysRVG finetuned DiT + LoRA','inference 40',item.physrvg_on,'var(--on)')}</div></div>`;
        strip.querySelector('button').onclick=()=>{loadStrip(strip);strip.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})};root.append(strip)});content.append(root)});
    const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){loadStrip(entry.target);observer.unobserve(entry.target)}}),{rootMargin:'700px 0px'});
    document.querySelectorAll('.case-strip').forEach(strip=>observer.observe(strip));document.getElementById('replay-all').onclick=()=>document.querySelectorAll('video:not([data-src])').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});
  </script>
</body>
</html>
'''


def build_dashboard(source_page: Path, output_dir: Path) -> Path:
    data = build_video_only_data(load_dashboard_payload(source_page.resolve()))
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = HTML_TEMPLATE.replace("__DATA__", encoded).replace(
        "__GENERATED__", data["generated_utc"]
    )
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    print(build_dashboard(args.source_page, args.output_dir))


if __name__ == "__main__":
    main()
