#!/usr/bin/env python3
"""Read-only results dashboard for the frozen GT-STC validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)
SEED = 47326
MODES = ("region", "point", "combined")
MODE_LABELS = {
    "region": "Region · tube mass",
    "point": "Point · tracked correspondence",
    "combined": "Combined · region + point",
}


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _target_metric(path: Path, target: str) -> dict[str, Any] | None:
    for row in _json(path).get("metrics", []):
        if str(row.get("target")) == target:
            return row
    return None


def _variant(case: str, target: str, mode: str) -> dict[str, Any]:
    name = "baseline" if mode == "baseline" else f"{mode}__{target}__lambda0p1"
    directory = ROOT / "generations" / case / f"seed_{SEED:05d}" / name
    video = directory / "generated.mp4"
    metric_path = directory / "trajectory_metrics.json"
    metric = _target_metric(metric_path, target)
    complete = (
        (directory / "complete.json").is_file()
        and video.is_file()
        and video.stat().st_size > 0
    )
    return {
        "name": name,
        "mode": mode,
        "label": "Baseline" if mode == "baseline" else MODE_LABELS[mode],
        "complete": complete,
        "video_ready": complete,
        "metric_ready": metric is not None,
        "metric": metric,
    }


def _representatives(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select auditable best/worst examples from the frozen eligible cohort."""
    rows: list[dict[str, Any]] = []
    for case_row in cases:
        for target_row in case_row["targets"]:
            for variant in target_row["variants"]:
                if variant["mode"] not in MODES or not variant["metric_ready"]:
                    continue
                rows.append(
                    {
                        "case": case_row["case"],
                        "target": target_row["name"],
                        "mode": variant["mode"],
                        "delta_ade_d0": variant.get("delta_ade_d0"),
                        "delta_track_loss": variant.get("delta_track_loss"),
                        "quality_pass": bool((variant.get("metric") or {}).get("quality_pass")),
                    }
                )
    selected: list[dict[str, Any]] = []
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        gated = [
            row
            for row in mode_rows
            if row["quality_pass"] and row["delta_ade_d0"] is not None
        ]
        trackable = [row for row in mode_rows if row["delta_track_loss"] is not None]
        choices = (
            ("最大轨迹改善", min(gated, key=lambda row: row["delta_ade_d0"]) if gated else None),
            ("最大轨迹恶化", max(gated, key=lambda row: row["delta_ade_d0"]) if gated else None),
            (
                "最大可追踪性损失",
                max(trackable, key=lambda row: row["delta_track_loss"]) if trackable else None,
            ),
        )
        for category, row in choices:
            if row is not None:
                selected.append({"category": category, **row})
    return selected


def catalog() -> dict[str, Any]:
    screening_path = ROOT / "screening" / f"seed_{SEED:05d}" / "baseline_eligibility.json"
    screening = _json(screening_path)
    cases = []
    guided_complete = 0
    guided_metrics = 0
    for job in screening.get("eligible_jobs", []):
        case = str(job["case"])
        tube_manifest = _json(ROOT / "gt_tubes" / case / "manifest.json")
        targets = []
        for target in job["targets"]:
            target = str(target)
            variants = [_variant(case, target, "baseline")]
            variants.extend(_variant(case, target, mode) for mode in MODES)
            guided_complete += sum(row["complete"] for row in variants[1:])
            guided_metrics += sum(row["metric_ready"] for row in variants[1:])
            baseline_metric = variants[0]["metric"] or {}
            for variant in variants[1:]:
                metric = variant["metric"] or {}
                variant["delta_ade_d0"] = (
                    float(metric["ade_d0"]) - float(baseline_metric["ade_d0"])
                    if metric.get("ade_d0") is not None
                    and baseline_metric.get("ade_d0") is not None
                    else None
                )
                variant["delta_track_loss"] = (
                    float(metric["future_track_loss_score_0_100"])
                    - float(baseline_metric["future_track_loss_score_0_100"])
                    if metric.get("future_track_loss_score_0_100") is not None
                    and baseline_metric.get("future_track_loss_score_0_100") is not None
                    else None
                )
            targets.append({"name": target, "variants": variants})
        cases.append(
            {
                "case": case,
                "source_video_ready": Path(str(tube_manifest.get("source_video", ""))).is_file(),
                "targets": targets,
            }
        )
    final_report = _json(
        ROOT / "final_analysis" / f"seed_{SEED:05d}" / "frozen_validation_report.json"
    )
    total = int(screening.get("eligible_target_count", 0)) * len(MODES)
    return {
        "protocol": "wan_gt_guidance_frozen_validation_v1",
        "seed": SEED,
        "case_count": int(screening.get("case_count", 0)),
        "eligible_case_count": int(screening.get("eligible_case_count", 0)),
        "eligible_target_count": int(screening.get("eligible_target_count", 0)),
        "missing_case_count": int(screening.get("missing_case_count", 0)),
        "guided_total": total,
        "guided_complete": guided_complete,
        "guided_metrics": guided_metrics,
        "cases": cases,
        "representatives": _representatives(cases),
        "final_report_ready": bool(final_report),
        "final_aggregate": final_report.get("aggregate", []),
        "trigger_modes": final_report.get("trigger_modes", []),
        "definitions": [
            {
                "metric": "Future ADE / D0",
                "calculation": "F04–F48 共同可见 CoTracker 点到 source GT tube 的逐帧平均距离 ÷ F00 bbox 对角线",
                "direction": "越小越接近 GT；Δ<0 表示优于同 seed Baseline",
            },
            {
                "metric": "Future FDE / D0",
                "calculation": "最后共同有效未来 anchor 的点距离 ÷ D0",
                "direction": "越小越接近 GT；只在轨迹门控通过时报告",
            },
            {
                "metric": "PCK@10% D0",
                "calculation": "共同可见未来点中误差 ≤ 0.1D0 的比例",
                "direction": "越大越接近 GT",
            },
            {
                "metric": "Future Track Loss",
                "calculation": "100 × (1 − 共同有效未来 anchor / source 有效未来 anchor)",
                "direction": "越小越好；消失/不可追踪也保留，不能被 ADE 均值丢掉",
            },
            {
                "metric": "Trajectory quality gate",
                "calculation": "未来共同 anchor ≥4 且覆盖率 ≥0.8；F00 condition frame 完全排除",
                "direction": "未通过时 ADE/FDE/PCK 记 N/A",
            },
        ],
    }


def asset(kind: str, case: str, target: str = "", variant: str = "") -> Path | None:
    screening = _json(
        ROOT / "screening" / f"seed_{SEED:05d}" / "baseline_eligibility.json"
    )
    jobs = {
        str(row["case"]): {str(value) for value in row["targets"]}
        for row in screening.get("eligible_jobs", [])
    }
    if case not in jobs:
        return None
    if kind == "source":
        source = Path(str(_json(ROOT / "gt_tubes" / case / "manifest.json").get("source_video", "")))
        return source if source.is_file() else None
    if kind != "generated" or target not in jobs[case]:
        return None
    allowed = {"baseline"} | {
        f"{mode}__{target}__lambda0p1" for mode in MODES
    }
    if variant not in allowed:
        return None
    video = ROOT / "generations" / case / f"seed_{SEED:05d}" / variant / "generated.mp4"
    return video if video.is_file() else None


def page() -> str:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GT-STC Guidance Validation</title><style>
:root{--ink:#152238;--paper:#edf3f7;--panel:#f8fbfd;--line:#b7c7d5;--cobalt:#175c91;--cyan:#1d91a8;--amber:#d88a24;--red:#b64d50;--muted:#60748a;--shadow:0 14px 40px #17345018}*{box-sizing:border-box}body{margin:0;background:linear-gradient(90deg,#dbe7ee 1px,transparent 1px),linear-gradient(#dbe7ee 1px,transparent 1px),var(--paper);background-size:28px 28px;color:var(--ink);font:15px/1.55 "Avenir Next","Segoe UI",sans-serif}header{padding:34px clamp(20px,5vw,72px) 28px;background:#eef5f9eF;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}a{color:var(--cobalt)}.eyebrow,.mono{font:700 11px/1.3 ui-monospace,SFMono-Regular,monospace;letter-spacing:.12em;text-transform:uppercase}.eyebrow{color:var(--cyan);margin-top:18px}h1{max-width:1100px;margin:8px 0 10px;font:700 clamp(34px,6vw,76px)/.94 "Arial Narrow","Avenir Next Condensed",sans-serif;letter-spacing:-.045em}.lead{max-width:970px;color:#3c536b;font-size:17px}.anchor-strip{display:grid;grid-template-columns:repeat(13,1fr);max-width:780px;margin-top:24px;border:1px solid var(--line);background:var(--panel)}.anchor-strip i{height:13px;border-right:1px solid var(--line);background:linear-gradient(90deg,var(--cobalt),var(--cyan));opacity:calc(.25 + var(--n)*.055)}.anchor-strip i:last-child{border:0}main{padding:26px clamp(16px,4vw,64px) 80px;max-width:1900px;margin:auto}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.stat,.section{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow)}.stat{padding:18px}.stat b{display:block;font:700 30px/1 "Arial Narrow",sans-serif;margin-top:7px}.section{margin-top:18px;padding:20px}.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:16px 0}.toolbar label{font-weight:700}.toolbar select,.toolbar button{display:block;margin-top:5px;padding:9px 12px;border:1px solid #8da5b7;background:#fff;color:var(--ink)}.toolbar button{cursor:pointer;background:var(--cobalt);color:#fff}.definitions{overflow:auto}table{width:100%;border-collapse:collapse;min-width:800px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d6e1e8;vertical-align:top}th{font:700 11px ui-monospace,monospace;text-transform:uppercase;color:var(--muted)}.case-title{display:flex;justify-content:space-between;gap:16px;align-items:center}.case-title h2{margin:0;font:700 25px "Arial Narrow",sans-serif}.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:11px;overflow-x:auto;padding-bottom:8px}.card{min-width:220px;border:1px solid var(--line);background:#fff}.card video,.empty{width:100%;aspect-ratio:16/9;background:#102033;display:block}.empty{display:grid;place-items:center;color:#b9c8d4;font:700 12px ui-monospace,monospace}.caption{padding:12px}.caption b{display:block}.bad{color:var(--red)}.good{color:#16785f}.pending{color:var(--amber)}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin-top:9px;font:12px ui-monospace,monospace;color:#40576d}.aggregate{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}.agg{padding:14px;border-left:5px solid var(--cyan);background:#eef6f8}.agg b{display:block}.jump{padding:6px 9px;border:1px solid var(--cobalt);background:#fff;color:var(--cobalt);cursor:pointer}.footer{color:var(--muted);margin-top:28px}@media(max-width:760px){h1{font-size:43px}.section{padding:13px}.grid{grid-template-columns:repeat(5,82vw)}}@media(prefers-reduced-motion:no-preference){.stat,.section{animation:up .35s ease both}@keyframes up{from{opacity:0;transform:translateY(8px)}}}</style></head><body>
<header><a href="/">← 返回 8092 总入口</a> · <a href="/gt-stc-guidance-preflight?v=2">Tube 预检</a><div class="eyebrow">Frozen latest3350 Top100 · source-oracle intervention</div><h1>GT tube 是否真的<br>能拉动生成轨迹？</h1><p class="lead">同 seed Baseline 与 Region / Point / Combined 并排。只报告未来帧；条件帧 F00 不参与轨迹分数。消失不是缺失数据：它进入 Track Loss，并阻止该样本被算作 ADE 改善。</p><div class="anchor-strip" aria-label="13 latent anchors"><i style="--n:0"></i><i style="--n:1"></i><i style="--n:2"></i><i style="--n:3"></i><i style="--n:4"></i><i style="--n:5"></i><i style="--n:6"></i><i style="--n:7"></i><i style="--n:8"></i><i style="--n:9"></i><i style="--n:10"></i><i style="--n:11"></i><i style="--n:12"></i></div></header>
<main><div id="summary" class="summary"></div><section class="section"><h2>计算与判读</h2><div class="definitions"><table><thead><tr><th>指标</th><th>精确计算</th><th>方向</th></tr></thead><tbody id="defs"></tbody></table></div></section><section class="section" id="paired"><div class="case-title"><h2>冻结配对结果</h2><span id="updated" class="mono">读取中</span></div><div class="toolbar"><label>Case<select id="case"></select></label><label>Target<select id="target"></select></label><button id="refresh">刷新现场</button><button id="replay">同步重播</button></div><div id="gallery"></div></section><section class="section"><h2>Case-balanced 汇总</h2><div id="aggregate" class="aggregate"></div></section><section class="section"><h2>代表性样本</h2><p>仅在冻结 eligible cohort 内选择；ADE 排序要求 guided trajectory gate 通过，Track Loss 排序保留破坏性失败。</p><div class="definitions"><table><thead><tr><th>Mode</th><th>选择理由</th><th>Case / Target</th><th>ΔADE/D0</th><th>ΔTrack Loss</th><th>查看</th></tr></thead><tbody id="representatives"></tbody></table></div></section><p class="footer">视频仅在进入视口附近时加载。未生成项显示明确状态，不预留会触发网络请求的空 video。</p></main>
<script>
const api='/api/gt-stc-guidance-results',E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),F=(v,d=3)=>v==null?'N/A':Number(v).toFixed(d);let D;const $=id=>document.getElementById(id);
function lazy(){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting&&e.target.dataset.src){e.target.src=e.target.dataset.src;delete e.target.dataset.src;e.target.load();io.unobserve(e.target)}}),{rootMargin:'500px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function metric(v){const m=v.metric||{},gate=m.quality_pass===true;return `<div class="metrics"><span class="${gate?'good':'bad'}">Gate ${gate?'PASS':'FAIL/N.A.'}</span><span>ADE/D0 ${F(m.ade_d0)}</span><span>FDE/D0 ${F(m.fde_d0)}</span><span>PCK10 ${F(m.pck_10pct_d0)}</span><span>TrackLoss ${F(m.future_track_loss_score_0_100,1)}</span>${v.mode==='baseline'?'':`<span>ΔADE ${F(v.delta_ade_d0)}</span><span>ΔLoss ${F(v.delta_track_loss,1)}</span>`}</div>`}
function video(src,label,v){return `<article class="card">${src?`<video controls muted playsinline preload="none" data-src="${src}"></video>`:`<div class="empty">${v?.complete?'METRICS PENDING':'QUEUED / RUNNING'}</div>`}<div class="caption"><b>${E(label)}</b>${v?metric(v):'<div class="metrics"><span>Source GT tube</span></div>'}</div></article>`}
function render(){const c=D.cases.find(x=>x.case===$('case').value)||D.cases[0];if(!c){$('gallery').innerHTML='<p>Baseline screen 尚未完成。</p>';return}const opts=c.targets.map(t=>`<option>${E(t.name)}</option>`).join('');if($('target').dataset.case!==c.case){$('target').innerHTML=opts;$('target').dataset.case=c.case}const t=c.targets.find(x=>x.name===$('target').value)||c.targets[0],q=x=>encodeURIComponent(x);let cards=video(`${api}/asset?kind=source&case=${q(c.case)}`,'Source GT',null);for(const v of t.variants){const src=v.video_ready?`${api}/asset?kind=generated&case=${q(c.case)}&target=${q(t.name)}&variant=${q(v.name)}`:'';cards+=video(src,v.label,v)}$('gallery').innerHTML=`<div class="case-title"><h2>${E(c.case)} · ${E(t.name)}</h2><span class="mono">seed ${D.seed}</span></div><div class="grid">${cards}</div>`;lazy()}
function reps(){$('representatives').innerHTML=(D.representatives||[]).map(x=>`<tr><td><b>${E(x.mode)}</b></td><td>${E(x.category)}</td><td>${E(x.case)}<br><span class="mono">${E(x.target)}</span></td><td>${F(x.delta_ade_d0)}</td><td>${F(x.delta_track_loss,1)}</td><td><button class="jump" data-case="${E(x.case)}" data-target="${E(x.target)}">跳转</button></td></tr>`).join('')||'<tr><td colspan="6" class="pending">轨迹指标完成后自动生成。</td></tr>';document.querySelectorAll('.jump').forEach(b=>b.onclick=()=>{$('case').value=b.dataset.case;$('target').dataset.case='';render();$('target').value=b.dataset.target;render();$('paired').scrollIntoView({behavior:'smooth'})})}
function summary(){const p=D.guided_total?Math.round(100*D.guided_complete/D.guided_total):0;$('summary').innerHTML=`<div class="stat"><span class="mono">Source audit</span><b>${D.case_count}/20</b><small>完成 tube 的 case</small></div><div class="stat"><span class="mono">Frozen screen</span><b>${D.eligible_target_count}</b><small>${D.eligible_case_count} cases 的 eligible targets</small></div><div class="stat"><span class="mono">Guided generation</span><b>${D.guided_complete}/${D.guided_total}</b><small>${p}% · Region/Point/Combined</small></div><div class="stat"><span class="mono">Trajectory metrics</span><b>${D.guided_metrics}/${D.guided_total}</b><small>CoTracker future-only</small></div>`;$('defs').innerHTML=D.definitions.map(x=>`<tr><td><b>${E(x.metric)}</b></td><td>${E(x.calculation)}</td><td>${E(x.direction)}</td></tr>`).join('');$('aggregate').innerHTML=D.final_report_ready?D.final_aggregate.filter(x=>x.lambda===.1).map(x=>`<div class="agg"><b>${E(x.mode)} · λ0.1</b><span>ΔADE/D0 ${F(x.case_balanced_mean_delta_ade_d0)} · ΔTrack Loss ${F(x.case_balanced_mean_delta_track_loss,1)} · 改善 cases ${x.improved_case_count}</span></div>`).join(''):'<p class="pending">完整三模式指标尚未齐全；汇总将在最后一个评估完成后自动出现。</p>';reps()}
async function load(){D=await fetch(api+'/catalog?x='+Date.now()).then(r=>r.json());const old=$('case').value;$('case').innerHTML=D.cases.map(c=>`<option>${E(c.case)}</option>`).join('');if(D.cases.some(c=>c.case===old))$('case').value=old;summary();render();$('updated').textContent=new Date().toLocaleTimeString()}
$('case').onchange=render;$('target').onchange=render;$('refresh').onclick=load;$('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});load();setInterval(load,30000);
</script></body></html>'''
