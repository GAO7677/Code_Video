from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    path: tuple[str, ...]
    direction: str
    threshold: float
    precision: int


# Thresholds only make metrics with different units comparable for case ranking.
# They are deliberately displayed in the page rather than hidden as a magic score.
METRICS: tuple[Metric, ...] = (
    Metric("piq_ctx", "Physics-IQ · 含条件帧", ("physics_iq_with_context", "score"), "up", 1.0, 2),
    Metric("piq_noctx", "Physics-IQ · 去条件帧", ("physics_iq_without_context", "score"), "up", 1.0, 2),
    Metric("pmf_ctx", "PMF · 含条件帧", ("pmf_with_context", "score"), "up", 0.02, 4),
    Metric("pmf_noctx", "PMF · 去条件帧", ("pmf_without_context", "score"), "up", 0.02, 4),
    Metric("wmreward", "WMReward Surprise", ("wmreward", "surprise"), "down", 0.002, 4),
    Metric("vbench_subject", "VBench · 主体一致性", ("vbench_subject_consistency", "score"), "up", 0.002, 4),
    Metric("vbench_background", "VBench · 背景一致性", ("vbench_background_consistency", "score"), "up", 0.002, 4),
    Metric("vbench_flicker", "VBench · 时序闪烁", ("vbench_temporal_flickering", "score"), "up", 0.0005, 4),
    Metric("vbench_smooth", "VBench · 运动平滑", ("vbench_motion_smoothness", "score"), "up", 0.0005, 4),
    Metric("vbench_dynamic", "VBench · 动态程度", ("vbench_dynamic_degree", "score"), "up", 0.05, 4),
    Metric("vbench_aesthetic", "VBench · 美学质量", ("vbench_aesthetic_quality", "score"), "up", 0.01, 4),
    Metric("vbench_imaging", "VBench · 成像质量", ("vbench_imaging_quality", "score"), "up", 0.01, 4),
    Metric("videophy2_sa", "VideoPhy-2 · SA", ("videophy2", "sa_score"), "up", 1.0, 2),
    Metric("videophy2_pc", "VideoPhy-2 · PC", ("videophy2", "pc_score"), "up", 1.0, 2),
    Metric("videophy2_joint", "VideoPhy-2 · Joint pass", ("videophy2", "joint_rate"), "up", 1.0, 2),
    Metric("cosmos", "Cosmos-Reason1", ("cosmos_reason1", "score"), "up", 1.0, 2),
)

CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("step",),
    ("seed",),
    ("effective_context_frames",),
    ("frame_indices",),
    ("guidance",),
    ("negative_prompt",),
    ("do_cfg",),
    ("model_args", "fps"),
    ("model_args", "num_frames"),
    ("model_args", "height"),
    ("model_args", "width"),
    ("model_args", "dit_checkpoint"),
    ("model_args", "model_id"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PhysRVG LoRA on/off metric-difference portal.")
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--no-lora-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_safe(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def case_stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.mp4") if path.is_file()}


def ensure_link(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    link.symlink_to(target.resolve(), target_is_directory=True)


def metric_record(metric: Metric, with_lora: dict[str, Any], no_lora: dict[str, Any]) -> dict[str, Any]:
    baseline = as_number(nested_get(with_lora, metric.path))
    ablation = as_number(nested_get(no_lora, metric.path))
    delta = None if baseline is None or ablation is None else ablation - baseline
    gap = None if delta is None else abs(delta) / metric.threshold
    quality_delta = None if delta is None else (delta if metric.direction == "up" else -delta)
    return {
        "key": metric.key,
        "label": metric.label,
        "direction": metric.direction,
        "threshold": metric.threshold,
        "precision": metric.precision,
        "with_lora": baseline,
        "no_lora": ablation,
        "delta": delta,
        "quality_delta": quality_delta,
        "gap": gap,
    }


def validate_pair(stem: str, left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for path in CONFIG_PATHS:
        left_value = nested_get(left, path)
        right_value = nested_get(right, path)
        if left_value != right_value:
            mismatches.append(f"{stem}: {'.'.join(path)}: {left_value!r} != {right_value!r}")
    return mismatches


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def build_payload(lora_dir: Path, no_lora_dir: Path, top_k: int) -> dict[str, Any]:
    lora_stems = case_stems(lora_dir)
    no_lora_stems = case_stems(no_lora_dir)
    if lora_stems != no_lora_stems:
        missing_left = sorted(no_lora_stems - lora_stems)
        missing_right = sorted(lora_stems - no_lora_stems)
        raise RuntimeError(f"Unpaired MP4s; missing with-LoRA={missing_left}, missing no-LoRA={missing_right}")
    if not lora_stems:
        raise RuntimeError("No paired MP4 cases found")

    cases: list[dict[str, Any]] = []
    config_mismatches: list[str] = []
    for stem in sorted(lora_stems):
        left_json = lora_dir / f"{stem}.json"
        right_json = no_lora_dir / f"{stem}.json"
        if not left_json.is_file() or not right_json.is_file():
            raise FileNotFoundError(f"Missing per-case JSON for {stem}")
        left = load_json(left_json)
        right = load_json(right_json)
        config_mismatches.extend(validate_pair(stem, left, right))
        metrics = [metric_record(metric, left, right) for metric in METRICS]
        gaps = sorted((row["gap"] for row in metrics if row["gap"] is not None), reverse=True)
        top_gaps = gaps[:5]
        quality_deltas = [row["quality_delta"] for row in metrics if row["quality_delta"] is not None]
        cases.append(
            {
                "id": stem,
                "caption": left.get("input_caption") or right.get("input_caption") or "",
                "source_video": left.get("source_video") or right.get("source_video"),
                "with_lora_video": f"videos/with_lora/{stem}.mp4",
                "no_lora_video": f"videos/no_lora/{stem}.mp4",
                "input_image": f"videos/with_lora/{stem}_input_ctx08.jpg",
                "metrics": metrics,
                "difference_score": mean(top_gaps) or 0.0,
                "significant_count": sum(gap >= 1.0 for gap in gaps),
                "max_gap": gaps[0] if gaps else 0.0,
                "with_lora_wins": sum(value < -1e-12 for value in quality_deltas),
                "no_lora_wins": sum(value > 1e-12 for value in quality_deltas),
            }
        )

    cases.sort(key=lambda row: (-row["difference_score"], -row["significant_count"], -row["max_gap"], row["id"]))
    for rank, case in enumerate(cases, start=1):
        case["rank"] = rank
        case["is_top"] = rank <= min(top_k, len(cases))

    aggregates: list[dict[str, Any]] = []
    for metric in METRICS:
        rows = [next(row for row in case["metrics"] if row["key"] == metric.key) for case in cases]
        left_values = [row["with_lora"] for row in rows if row["with_lora"] is not None]
        right_values = [row["no_lora"] for row in rows if row["no_lora"] is not None]
        left_mean = mean(left_values)
        right_mean = mean(right_values)
        delta = None if left_mean is None or right_mean is None else right_mean - left_mean
        aggregates.append(
            {
                "key": metric.key,
                "label": metric.label,
                "direction": metric.direction,
                "precision": metric.precision,
                "with_lora": left_mean,
                "no_lora": right_mean,
                "delta": delta,
                "quality_delta": None if delta is None else (delta if metric.direction == "up" else -delta),
            }
        )

    first = load_json(lora_dir / f"{cases[0]['id']}.json")
    config = {
        "steps": first.get("step"),
        "seed": first.get("seed"),
        "guidance": first.get("guidance"),
        "context_frames": first.get("effective_context_frames"),
        "fps": nested_get(first, ("model_args", "fps")),
        "num_frames": nested_get(first, ("model_args", "num_frames")),
        "size": f"{nested_get(first, ('model_args', 'height'))}×{nested_get(first, ('model_args', 'width'))}",
        "dit_checkpoint": nested_get(first, ("model_args", "dit_checkpoint")),
        "lora_checkpoint": nested_get(first, ("model_args", "lora_checkpoint")),
        "negative_prompt": first.get("negative_prompt"),
        "do_cfg": first.get("do_cfg"),
    }
    return {
        "case_count": len(cases),
        "top_k": min(top_k, len(cases)),
        "cases": cases,
        "aggregates": aggregates,
        "config": config,
        "config_mismatches": config_mismatches,
        "ranking_note": "综合差异 = 每个 case 最大 5 项 |无LoRA−有LoRA| / 指标敏感阈值 的均值；Joint pass 作为独立展示项参与排序。",
    }


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    metric_keys = [metric.key for metric in METRICS]
    fieldnames = ["rank", "case", "difference_score", "significant_count", "max_gap"]
    for key in metric_keys:
        fieldnames.extend((f"{key}_with_lora", f"{key}_no_lora", f"{key}_delta", f"{key}_gap"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in payload["cases"]:
            row: dict[str, Any] = {
                "rank": case["rank"],
                "case": case["id"],
                "difference_score": case["difference_score"],
                "significant_count": case["significant_count"],
                "max_gap": case["max_gap"],
            }
            for metric in case["metrics"]:
                key = metric["key"]
                row[f"{key}_with_lora"] = metric["with_lora"]
                row[f"{key}_no_lora"] = metric["no_lora"]
                row[f"{key}_delta"] = metric["delta"]
                row[f"{key}_gap"] = metric["gap"]
            writer.writerow(row)


def build_html(payload: dict[str, Any]) -> str:
    data = json_safe(payload)
    title = html.escape("PHYRVG-PhysRVG LoRA 开关：指标差异 Case 对比")
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    :root{{--bg:#eef2f0;--paper:#fffdf8;--ink:#16211d;--muted:#66716c;--line:#d4ddd8;--green:#116c55;--red:#a34135;--blue:#315c87;--amber:#b77a22;--shadow:0 18px 55px rgba(28,49,40,.10)}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:radial-gradient(circle at 8% 0,#dcebe5 0,transparent 28rem),radial-gradient(circle at 95% 12%,#f2e5d3 0,transparent 30rem),var(--bg);color:var(--ink);font-family:Inter,"Noto Sans SC",system-ui,sans-serif}}
    .shell{{max-width:1540px;margin:auto;padding:26px 22px 90px}} .panel{{background:rgba(255,253,248,.94);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow)}}
    .hero{{padding:26px 28px}} .eyebrow{{font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--green)}} h1{{margin:8px 0 10px;font-size:clamp(30px,4vw,52px);line-height:1.04;letter-spacing:-.035em}} p{{line-height:1.65}} .hero p{{max-width:1050px;margin:0;color:var(--muted)}}
    .config{{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}} .pill{{border:1px solid var(--line);border-radius:999px;padding:7px 11px;background:#fff;font-size:12px}} .pill b{{color:var(--green)}}
    .picker{{display:grid;grid-template-columns:minmax(280px,1fr) auto auto;gap:10px;align-items:end;padding:18px 20px;margin-top:18px;position:sticky;top:10px;z-index:20}} label{{display:block;font-size:12px;font-weight:800;color:var(--muted);margin-bottom:7px}} select,button{{font:inherit}} select{{width:100%;border:1px solid var(--line);background:white;border-radius:12px;padding:11px 12px;color:var(--ink)}} button{{border:0;border-radius:12px;padding:11px 15px;font-weight:800;cursor:pointer;background:#e7ece9;color:var(--ink)}} button:hover{{filter:brightness(.97)}}
    .case-head{{margin-top:18px;padding:22px 24px}} .case-title-row{{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}} h2{{font-size:22px;margin:0;overflow-wrap:anywhere}} .caption{{color:var(--muted);margin:9px 0 0;max-width:1100px}} .rank-badge{{flex:0 0 auto;border:1px solid #dcb979;color:#80540f;background:#fff4db;border-radius:999px;padding:8px 11px;font-size:12px;font-weight:900}}
    .stats{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-top:17px}} .stat{{border:1px solid var(--line);background:#fff;border-radius:15px;padding:12px}} .stat span{{display:block;color:var(--muted);font-size:11px;margin-bottom:4px}} .stat strong{{font-size:20px}}
    .videos{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}} .video-card{{overflow:hidden}} .video-head{{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:14px 16px}} .video-head h3{{margin:0;font-size:17px}} .tag{{font-size:11px;font-weight:900;border-radius:999px;padding:6px 9px}} .tag.on{{color:#fff;background:var(--green)}} .tag.off{{color:#fff;background:var(--blue)}} video{{display:block;width:100%;background:#111;aspect-ratio:896/512;object-fit:contain}}
    .metrics{{margin-top:18px;padding:20px;overflow:hidden}} .section-title{{display:flex;justify-content:space-between;gap:18px;align-items:baseline;margin-bottom:13px}} .section-title h3{{margin:0;font-size:19px}} .section-title p{{margin:0;color:var(--muted);font-size:12px}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:15px}} table{{width:100%;border-collapse:collapse;background:#fff;min-width:880px}} th,td{{padding:10px 12px;border-bottom:1px solid #e8ece9;text-align:right;font-variant-numeric:tabular-nums}} th{{position:sticky;top:0;background:#f5f7f5;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}} th:first-child,td:first-child{{text-align:left}} tr.hot td{{background:#fff9e9}} .dir{{font-size:10px;color:var(--muted)}} .delta.pos,.verdict.pos{{color:var(--green);font-weight:800}} .delta.neg,.verdict.neg{{color:var(--red);font-weight:800}} .gapbar{{display:inline-block;min-width:56px;border-radius:999px;background:#edf1ef;padding:4px 7px;font-size:11px}}
    details{{margin-top:18px}} summary{{cursor:pointer;padding:17px 20px;font-weight:800}} details .inner{{padding:0 20px 20px}} .ranking{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}} .rank-item{{text-align:left;background:#fff;border:1px solid var(--line);padding:12px;border-radius:14px}} .rank-item b{{display:block;overflow-wrap:anywhere}} .rank-item span{{display:block;color:var(--muted);font-size:11px;margin-top:5px}}
    .replay{{position:fixed;right:24px;bottom:24px;z-index:50;background:var(--ink);color:#fff;border:1px solid rgba(255,255,255,.2);box-shadow:0 12px 35px rgba(0,0,0,.25);padding:13px 18px;border-radius:999px}}
    .footnote{{font-size:12px;color:var(--muted);margin-top:16px}} .hidden{{display:none}}
    @media(max-width:900px){{.videos{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}.ranking{{grid-template-columns:1fr 1fr}}.picker{{grid-template-columns:1fr auto auto}}}}
    @media(max-width:600px){{.shell{{padding:12px 10px 90px}}.hero{{padding:21px 18px}}.picker{{grid-template-columns:1fr 1fr}}.picker .field{{grid-column:1/-1}}.stats{{grid-template-columns:1fr 1fr}}.ranking{{grid-template-columns:1fr}}.replay{{right:14px;bottom:14px}}}}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero panel">
      <div class="eyebrow">{payload['case_count']} paired cases · LoRA ablation</div>
      <h1>{title}</h1>
      <p>只改变 PHYRVG-PhysRVG LoRA 是否加载；视频并排同步查看。默认按跨指标归一化差异排序，优先展示指标差异最大的 case。</p>
      <div class="config" id="config"></div>
    </section>
    <section class="picker panel">
      <div class="field"><label for="case-select">选择 case（按综合差异从大到小）</label><select id="case-select"></select></div>
      <button id="prev" type="button">← 上一个</button><button id="next" type="button">下一个 →</button>
    </section>
    <section class="case-head panel">
      <div class="case-title-row"><div><h2 id="case-title"></h2><p class="caption" id="caption"></p></div><div class="rank-badge" id="rank"></div></div>
      <div class="stats" id="stats"></div>
    </section>
    <section class="videos">
      <article class="video-card panel"><div class="video-head"><h3>PHYRVG-PhysRVG DIT + LoRA</h3><span class="tag on">LoRA ON</span></div><video id="with-lora" controls loop muted playsinline preload="metadata"></video></article>
      <article class="video-card panel"><div class="video-head"><h3>PHYRVG-PhysRVG finetuned DIT</h3><span class="tag off">LoRA OFF</span></div><video id="no-lora" controls loop muted playsinline preload="metadata"></video></article>
    </section>
    <section class="metrics panel">
      <div class="section-title"><h3>逐项指标差异</h3><p>Δ = 无 LoRA − 有 LoRA；表格按当前 case 的归一化差异降序</p></div>
      <div class="table-wrap"><table><thead><tr><th>指标</th><th>有 LoRA</th><th>无 LoRA</th><th>Δ</th><th>差异倍数</th><th>质量判断</th></tr></thead><tbody id="metric-body"></tbody></table></div>
      <p class="footnote" id="ranking-note"></p>
    </section>
    <details class="panel"><summary>差异最大的代表 case（点击切换）</summary><div class="inner ranking" id="ranking"></div></details>
    <details class="panel"><summary>全部 {payload['case_count']} 个 case 的指标均值</summary><div class="inner table-wrap"><table><thead><tr><th>指标</th><th>有 LoRA 均值</th><th>无 LoRA 均值</th><th>Δ</th><th>质量判断</th></tr></thead><tbody id="aggregate-body"></tbody></table></div></details>
  </main>
  <button class="replay" id="replay" type="button">↻ 全部重新播放</button>
  <script id="payload" type="application/json">{data}</script>
  <script>
    const DATA=JSON.parse(document.getElementById('payload').textContent);
    const $=id=>document.getElementById(id); const select=$('case-select');
    const fmt=(value,precision=3)=>value===null||value===undefined?'NA':Number(value).toFixed(precision).replace(/\\.?0+$/,'');
    const signed=(value,precision)=>value===null||value===undefined?'NA':`${{value>=0?'+':''}}${{fmt(value,precision)}}`;
    const klass=value=>value>1e-12?'pos':value<-1e-12?'neg':'';
    const verdict=value=>value>1e-12?'无 LoRA 更好':value<-1e-12?'有 LoRA 更好':'持平';
    const config=DATA.config;
    $('config').innerHTML=[['Steps',config.steps],['FPS',config.fps],['Frames',config.num_frames],['Context',config.context_frames],['CFG scale',config.guidance],['CFG active',config.do_cfg?'yes':'no'],['Seed',config.seed],['Size',config.size],['Negative prompt',config.negative_prompt]].map(([k,v])=>`<span class="pill"><b>${{k}}</b> ${{v}}</span>`).join('');
    DATA.cases.forEach(c=>{{const o=document.createElement('option');o.value=c.id;o.textContent=`#${{c.rank}} ${{c.is_top?'★ ':''}}${{c.id}} · score ${{fmt(c.difference_score,2)}}`;select.appendChild(o)}});
    function setVideo(video,path){{video.pause();video.removeAttribute('src');video.load();video.src=path;video.load()}}
    function render(id,push=true){{
      const c=DATA.cases.find(x=>x.id===id)||DATA.cases[0]; select.value=c.id;
      $('case-title').textContent=c.id; $('caption').textContent=c.caption||'（无 caption）'; $('rank').textContent=`差异排名 #${{c.rank}} / ${{DATA.case_count}}`;
      $('stats').innerHTML=`<div class="stat"><span>综合差异</span><strong>${{fmt(c.difference_score,2)}}×</strong></div><div class="stat"><span>超过阈值指标</span><strong>${{c.significant_count}}</strong></div><div class="stat"><span>最大单项差异</span><strong>${{fmt(c.max_gap,2)}}×</strong></div><div class="stat"><span>有 LoRA 更优项</span><strong>${{c.with_lora_wins}}</strong></div><div class="stat"><span>无 LoRA 更优项</span><strong>${{c.no_lora_wins}}</strong></div>`;
      setVideo($('with-lora'),c.with_lora_video);setVideo($('no-lora'),c.no_lora_video);
      const sorted=[...c.metrics].sort((a,b)=>(b.gap??-1)-(a.gap??-1));
      $('metric-body').innerHTML=sorted.map((m,i)=>`<tr class="${{i<5?'hot':''}}"><td><b>${{m.label}}</b><br><span class="dir">${{m.direction==='up'?'↑ 越高越好':'↓ 越低越好'}} · 阈值 ${{fmt(m.threshold,4)}}</span></td><td>${{fmt(m.with_lora,m.precision)}}</td><td>${{fmt(m.no_lora,m.precision)}}</td><td class="delta ${{klass(m.quality_delta)}}">${{signed(m.delta,m.precision)}}</td><td><span class="gapbar">${{fmt(m.gap,2)}}×</span></td><td class="verdict ${{klass(m.quality_delta)}}">${{verdict(m.quality_delta)}}</td></tr>`).join('');
      if(push)history.replaceState(null,'',`#case=${{encodeURIComponent(c.id)}}`);
    }}
    select.addEventListener('change',()=>render(select.value));
    function move(delta){{const i=DATA.cases.findIndex(c=>c.id===select.value);render(DATA.cases[(i+delta+DATA.cases.length)%DATA.cases.length].id)}}
    $('prev').addEventListener('click',()=>move(-1));$('next').addEventListener('click',()=>move(1));
    $('replay').addEventListener('click',()=>{{document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}})}});
    $('ranking-note').textContent=DATA.ranking_note;
    $('ranking').innerHTML=DATA.cases.filter(c=>c.is_top).map(c=>`<button class="rank-item" data-case="${{c.id}}"><b>#${{c.rank}} ${{c.id}}</b><span>综合差异 ${{fmt(c.difference_score,2)}}× · ${{c.significant_count}} 项超过阈值</span></button>`).join('');
    $('ranking').addEventListener('click',e=>{{const button=e.target.closest('[data-case]');if(button){{render(button.dataset.case);scrollTo({{top:document.querySelector('.case-head').offsetTop-90,behavior:'smooth'}})}}}});
    $('aggregate-body').innerHTML=DATA.aggregates.map(m=>`<tr><td><b>${{m.label}}</b><br><span class="dir">${{m.direction==='up'?'↑ 越高越好':'↓ 越低越好'}}</span></td><td>${{fmt(m.with_lora,m.precision)}}</td><td>${{fmt(m.no_lora,m.precision)}}</td><td class="delta ${{klass(m.quality_delta)}}">${{signed(m.delta,m.precision)}}</td><td class="verdict ${{klass(m.quality_delta)}}">${{verdict(m.quality_delta)}}</td></tr>`).join('');
    const initial=new URLSearchParams(location.hash.slice(1)).get('case');render(initial,false);
  </script>
</body>
</html>'''


def main() -> None:
    args = parse_args()
    lora_dir = args.lora_dir.expanduser().resolve()
    no_lora_dir = args.no_lora_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(lora_dir, no_lora_dir, args.top_k)
    if payload["config_mismatches"]:
        preview = "\n".join(payload["config_mismatches"][:10])
        raise RuntimeError(f"Inference configurations do not match:\n{preview}")
    videos = output_dir / "videos"
    videos.mkdir(exist_ok=True)
    ensure_link(videos / "with_lora", lora_dir)
    ensure_link(videos / "no_lora", no_lora_dir)
    (output_dir / "index.html").write_text(build_html(payload), encoding="utf-8")
    (output_dir / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "metric_differences.csv", payload)
    print(f"cases={payload['case_count']} top_k={payload['top_k']}")
    for case in payload["cases"][: payload["top_k"]]:
        print(f"#{case['rank']:02d} score={case['difference_score']:.3f} significant={case['significant_count']:02d} {case['id']}")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
