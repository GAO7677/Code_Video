#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


BASE_SERVER = Path(__file__).with_name("serve_latent_block_head_viewer_alltoken.py")
BENCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme_benchmark_test5_ready"
)
STATIC30_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme30_all720_head_zero_ablation_test5"
)
STATIC100_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme100_all720_head_zero_ablation_test5"
)
ADAPTIVE30_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "pck_step_adaptive_top30_bottom30_all720_head_zero_ablation_test5"
)
QK_NOISE30_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "pck_step_adaptive_qk_logit_noise_sigma030_all720_test5"
)
QK_ATTENTION_COMPARE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_additive_noise_alpha030_pilot"
)
ATTENTION_LORA_CASE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5/lora"
)
ATTENTION_LORA_CASE = "0613pybullet_sample_001460_w002"
ATTENTION_TEST_LIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
)
FULL_SA_ATTENTION_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5/full_sa"
)
BASELINE_ATTENTION_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5/baseline"
)
ATTENTION_REPLACEMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_replacement_steps40_frames49_test5"
)

VIDEO_CONDITIONS = (
    ("original", "Original", STATIC30_ROOT, "original.mp4"),
    ("static_top30_s00_09", "Static Top30 · S00-09", STATIC30_ROOT, "top30_steps_00_10.mp4"),
    ("static_top30_s10_19", "Static Top30 · S10-19", STATIC30_ROOT, "top30_steps_10_20.mp4"),
    ("static_top30_s20_29", "Static Top30 · S20-29", STATIC30_ROOT, "top30_steps_20_30.mp4"),
    ("static_top30_s30_39", "Static Top30 · S30-39", STATIC30_ROOT, "top30_steps_30_40.mp4"),
    ("static_top30", "Static Top30 · S00-39", STATIC30_ROOT, "top30_steps_00_40.mp4"),
    ("static_bottom30_s00_09", "Static Bottom30 · S00-09", STATIC30_ROOT, "bottom30_steps_00_10.mp4"),
    ("static_bottom30_s10_19", "Static Bottom30 · S10-19", STATIC30_ROOT, "bottom30_steps_10_20.mp4"),
    ("static_bottom30_s20_29", "Static Bottom30 · S20-29", STATIC30_ROOT, "bottom30_steps_20_30.mp4"),
    ("static_bottom30_s30_39", "Static Bottom30 · S30-39", STATIC30_ROOT, "bottom30_steps_30_40.mp4"),
    ("static_bottom30", "Static Bottom30 · S00-39", STATIC30_ROOT, "bottom30_steps_00_40.mp4"),
    ("static_top100_s00_09", "Static Top100 · S00-09", STATIC100_ROOT, "top100_steps_00_10.mp4"),
    ("static_top100_s10_19", "Static Top100 · S10-19", STATIC100_ROOT, "top100_steps_10_20.mp4"),
    ("static_top100_s20_29", "Static Top100 · S20-29", STATIC100_ROOT, "top100_steps_20_30.mp4"),
    ("static_top100_s30_39", "Static Top100 · S30-39", STATIC100_ROOT, "top100_steps_30_40.mp4"),
    ("static_top100", "Static Top100 · S00-39", STATIC100_ROOT, "top100_steps_00_40.mp4"),
    ("static_bottom100_s00_09", "Static Bottom100 · S00-09", STATIC100_ROOT, "bottom100_steps_00_10.mp4"),
    ("static_bottom100_s10_19", "Static Bottom100 · S10-19", STATIC100_ROOT, "bottom100_steps_10_20.mp4"),
    ("static_bottom100_s20_29", "Static Bottom100 · S20-29", STATIC100_ROOT, "bottom100_steps_20_30.mp4"),
    ("static_bottom100_s30_39", "Static Bottom100 · S30-39", STATIC100_ROOT, "bottom100_steps_30_40.mp4"),
    ("static_bottom100", "Static Bottom100 · S00-39", STATIC100_ROOT, "bottom100_steps_00_40.mp4"),
    ("adaptive_top30", "Step-adaptive Top30", ADAPTIVE30_ROOT, "top30_steps_00_40.mp4"),
    ("adaptive_bottom30", "Step-adaptive Bottom30", ADAPTIVE30_ROOT, "bottom30_steps_00_40.mp4"),
    ("qk_noise_top30_sigma030", "Q@K Noise σ0.30 · Adaptive Top30", QK_NOISE30_ROOT, "top30_steps_00_40.mp4"),
    ("qk_noise_bottom30_sigma030", "Q@K Noise σ0.30 · Adaptive Bottom30", QK_NOISE30_ROOT, "bottom30_steps_00_40.mp4"),
)

spec = importlib.util.spec_from_file_location("difftrack_base_viewer", BASE_SERVER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load visualization server: {BASE_SERVER}")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


METRICS = (
    ("physics_iq_with_context", "Physics-IQ +Ctx", "score", "max"),
    ("physics_iq_without_context", "Physics-IQ -Ctx", "score", "max"),
    ("pmf_with_context", "PMF +Ctx", "score", "max"),
    ("pmf_without_context", "PMF -Ctx", "score", "max"),
    ("wmreward", "WMReward Surprise", "surprise", "min"),
    ("vbench_subject_consistency", "Subject", "score", "max"),
    ("vbench_background_consistency", "Background", "score", "max"),
    ("vbench_temporal_flickering", "Flicker", "score", "max"),
    ("vbench_motion_smoothness", "Smoothness", "score", "max"),
    ("vbench_dynamic_degree", "Dynamic", "score", "max"),
    ("vbench_aesthetic_quality", "Aesthetic", "score", "max"),
    ("vbench_imaging_quality", "Imaging", "score", "max"),
    ("videophy2", "VideoPhy2 Joint", "joint_pass", "max"),
    ("cosmos_reason1", "Cosmos Reason1", "score", "max"),
)

METHOD_ORDER = (
    "original",
    "top30_steps_00_10",
    "top30_steps_10_20",
    "top30_steps_20_30",
    "top30_steps_30_40",
    "top30_steps_00_40",
    "bottom30_steps_00_10",
    "bottom30_steps_10_20",
    "bottom30_steps_20_30",
    "bottom30_steps_30_40",
    "bottom30_steps_00_40",
    "top100_steps_00_10",
    "top100_steps_10_20",
    "top100_steps_20_30",
    "top100_steps_30_40",
    "top100_steps_00_40",
    "bottom100_steps_00_10",
    "bottom100_steps_10_20",
    "bottom100_steps_20_30",
    "bottom100_steps_30_40",
    "bottom100_steps_00_40",
    "adaptive_top30_steps_00_40",
    "adaptive_bottom30_steps_00_40",
)

METHOD_LABELS = {
    "original": "Baseline / Original",
    "top30_steps_00_10": "Static Top30 · S00-09",
    "top30_steps_10_20": "Static Top30 · S10-19",
    "top30_steps_20_30": "Static Top30 · S20-29",
    "top30_steps_30_40": "Static Top30 · S30-39",
    "top30_steps_00_40": "Static Top30 · S00-39",
    "bottom30_steps_00_10": "Static Bottom30 · S00-09",
    "bottom30_steps_10_20": "Static Bottom30 · S10-19",
    "bottom30_steps_20_30": "Static Bottom30 · S20-29",
    "bottom30_steps_30_40": "Static Bottom30 · S30-39",
    "bottom30_steps_00_40": "Static Bottom30 · S00-39",
    "top100_steps_00_10": "Static Top100 · S00-09",
    "top100_steps_10_20": "Static Top100 · S10-19",
    "top100_steps_20_30": "Static Top100 · S20-29",
    "top100_steps_30_40": "Static Top100 · S30-39",
    "top100_steps_00_40": "Static Top100 · S00-39",
    "bottom100_steps_00_10": "Static Bottom100 · S00-09",
    "bottom100_steps_10_20": "Static Bottom100 · S10-19",
    "bottom100_steps_20_30": "Static Bottom100 · S20-29",
    "bottom100_steps_30_40": "Static Bottom100 · S30-39",
    "bottom100_steps_00_40": "Static Bottom100 · S00-39",
    "adaptive_top30_steps_00_40": "Step-adaptive Top30 · S00-39",
    "adaptive_bottom30_steps_00_40": "Step-adaptive Bottom30 · S00-39",
}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def load_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def expected_cases() -> dict[str, int]:
    payload = load_payload(BENCH_ROOT / "prepared_manifest.json") or {}
    raw = payload.get("cases_by_model")
    if not isinstance(raw, dict):
        return {"baseline": 0, "lora": 0}
    return {
        model: len(cases) if isinstance(cases, list) else 0
        for model, cases in raw.items()
    }


def method_dir(model: str, variant: str) -> Path:
    prefix = "wan22_baseline" if model == "baseline" else "wan_lora"
    return BENCH_ROOT / "methods" / f"{prefix}_{variant}"


def summarize_method(model: str, variant: str, expected: int) -> dict[str, Any]:
    result_root = method_dir(model, variant)
    payloads = [
        payload
        for path in sorted(result_root.glob("*.json"))
        if path.name != "batch_manifest.json"
        if (payload := load_payload(path)) is not None
        and isinstance(payload.get("input_json"), str)
    ]
    metric_values: dict[str, dict[str, Any]] = {}
    for field, _label, value_key, _direction in METRICS:
        values: list[float] = []
        for payload in payloads:
            metric_payload = payload.get(field)
            if not isinstance(metric_payload, dict):
                continue
            value = finite_number(metric_payload.get(value_key))
            if value is not None:
                values.append(value)
        metric_values[field] = {
            "count": len(values),
            "expected": expected,
            "complete": expected > 0 and len(values) == expected,
            "mean": None if not values else round(sum(values) / len(values), 4),
        }
    return {
        "variant": variant,
        "label": METHOD_LABELS[variant],
        "result_root": str(result_root),
        "num_cases": len(payloads),
        "expected_cases": expected,
        "metrics": metric_values,
    }


def benchmark_summary() -> dict[str, Any]:
    expected = expected_cases()
    models = []
    for model, label in (("baseline", "Wan2.2 Baseline"), ("lora", "Wan + LoRA")):
        rows = [summarize_method(model, variant, expected.get(model, 0)) for variant in METHOD_ORDER]
        models.append({"model": model, "label": label, "rows": rows})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": str(BENCH_ROOT),
        "comparison_note": "Best values are selected within each model because current model snapshots use different cases.",
        "metrics": [
            {"field": field, "label": label, "direction": direction}
            for field, label, _value_key, direction in METRICS
        ],
        "models": models,
    }


def ablation_video_path(model: str, case_key: str, condition: str) -> Path | None:
    if model not in {"baseline", "lora"}:
        return None
    for key, _label, root, filename in VIDEO_CONDITIONS:
        if key == condition:
            return root / model / "cases" / case_key / filename
    return None


def ablation_video_catalog() -> dict[str, Any]:
    case_keys: set[str] = set()
    for root in (STATIC30_ROOT, STATIC100_ROOT, ADAPTIVE30_ROOT, QK_NOISE30_ROOT):
        for model in ("baseline", "lora"):
            cases_root = root / model / "cases"
            if cases_root.is_dir():
                case_keys.update(path.name for path in cases_root.iterdir() if path.is_dir())
    cases = []
    ready_videos = 0
    for case_key in sorted(case_keys):
        model_rows = []
        for model, model_label in (("baseline", "Wan2.2 Baseline"), ("lora", "Wan + LoRA")):
            videos = []
            for key, label, _root, _filename in VIDEO_CONDITIONS:
                path = ablation_video_path(model, case_key, key)
                ready = bool(path and path.is_file() and path.stat().st_size > 0)
                ready_videos += int(ready)
                videos.append({"condition": key, "label": label, "ready": ready})
            model_rows.append({"model": model, "label": model_label, "videos": videos})
        cases.append({"case": case_key, "models": model_rows})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready_videos": ready_videos,
        "expected_videos": len(cases) * 2 * len(VIDEO_CONDITIONS),
        "num_cases": len(cases),
        "cases": cases,
    }


METRICS_PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCK Head 全条件消融指标</title><style>
:root{--paper:#f1ecdf;--ink:#17211e;--card:#fffdf8;--line:#b8b09f;--rust:#b64a31;--green:#176654;--pale:#e8f2ea;--wait:#eee7d8}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 0 0,#d7764b2b,transparent 34rem),radial-gradient(circle at 100% 5%,#4b9a8030,transparent 36rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1900px,calc(100% - 24px));margin:auto;padding:26px 0 70px}a{color:var(--green)}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(38px,6vw,78px);line-height:.94;letter-spacing:-.045em;margin:12px 0}.eyebrow{color:var(--rust);font-size:12px;font-weight:900;letter-spacing:.16em}.lead{max-width:1100px;line-height:1.65}.status{display:flex;gap:9px;flex-wrap:wrap;margin:20px 0}.pill{background:var(--card);border:1px solid var(--line);padding:8px 12px;border-radius:99px;font-size:13px}.model{margin:28px 0 45px}.model-head{display:flex;align-items:end;justify-content:space-between;gap:15px}.model h2{font-size:30px;margin:0 0 10px}.scroll{overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:4px 26px 4px 4px;box-shadow:0 15px 45px #53472f12}table{border-collapse:separate;border-spacing:0;width:100%;min-width:1750px;font-variant-numeric:tabular-nums}th,td{padding:11px 10px;border-right:1px solid #d7d0c0;border-bottom:1px solid #d7d0c0;text-align:center;font-size:12px}thead th{position:sticky;top:0;background:#20342d;color:white;z-index:2;white-space:nowrap}th:first-child,td:first-child{position:sticky;left:0;text-align:left;min-width:205px;z-index:1}thead th:first-child{z-index:3}tbody td:first-child{background:var(--card);font-weight:900}.value{font-size:15px;font-weight:900}.count{display:block;color:#746f64;font-size:10px;margin-top:3px}.best{background:var(--pale)}.best .value{color:var(--green)}.badge{display:inline-block;margin-left:5px;padding:2px 5px;border-radius:99px;background:var(--green);color:white;font-size:8px;vertical-align:2px}.pending{background:var(--wait);color:#877d69}.legend{font-size:12px;color:#665f52}.error{padding:25px;background:#fff2ec;border:1px solid #d68b72}.footer-note{line-height:1.6;font-size:12px;color:#665f52;margin-top:14px}@media(max-width:700px){main{width:min(100% - 12px,1900px)}h1{font-size:39px}.model-head{display:block}}
</style></head><body><main><a href="/">返回可视化总览</a><div class="eyebrow">PCK EXTREME HEAD ZERO · BENCHMARK</div><h1>消融指标<br>实时对照表</h1><p class="lead">比较 Original、PCK Top/Bottom 30 和 Top/Bottom 100 Head 在全部 40 个去噪步置零后的生成质量。BEST 仅在同一模型的五个条件全部完成该指标后标注。</p><div class="status" id="status"><span class="pill">正在读取结果...</span></div><div id="tables"></div><p class="footer-note">比较原则：Baseline 与 LoRA 当前使用不同的已完成 case 分片，因此最佳值按模型内部计算。WMReward Surprise 越低越好，其余指标越高越好。页面每 10 秒自动刷新。</p></main><script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function fmt(v){if(v===null||v===undefined)return '...';return Number(v).toFixed(4).replace(/0+$/,'').replace(/\.$/,'')}
function render(data){
  const metrics=data.metrics;
  let completed=0,total=0;
  for(const model of data.models)for(const row of model.rows)for(const metric of metrics){total++;if(row.metrics[metric.field].complete)completed++}
  document.getElementById('status').innerHTML=`<span class="pill"><b>${completed}/${total}</b> 方法-指标单元完成</span><span class="pill">更新 ${new Date(data.generated_at).toLocaleTimeString()}</span><span class="pill">14 项默认指标</span>`;
  document.getElementById('tables').innerHTML=data.models.map(model=>{
    const best={};
    for(const metric of metrics){
      const cells=model.rows.map(r=>r.metrics[metric.field]);
      if(cells.every(c=>c.complete&&c.mean!==null))best[metric.field]=metric.direction==='min'?Math.min(...cells.map(c=>c.mean)):Math.max(...cells.map(c=>c.mean));
    }
    const head=metrics.map(m=>`<th>${esc(m.label)}<span class="count">${m.direction==='min'?'LOWER':'HIGHER'}</span></th>`).join('');
    const body=model.rows.map(row=>`<tr><td>${esc(row.label)}<span class="count">${row.num_cases}/${row.expected_cases} cases</span></td>${metrics.map(m=>{const c=row.metrics[m.field];const isBest=best[m.field]!==undefined&&Math.abs(c.mean-best[m.field])<1e-9;return `<td class="${isBest?'best':c.complete?'':'pending'}"><span class="value">${fmt(c.mean)}</span>${isBest?'<span class="badge">BEST</span>':''}<span class="count">${c.count}/${c.expected}</span></td>`}).join('')}</tr>`).join('');
    return `<section class="model"><div class="model-head"><h2>${esc(model.label)}</h2><span class="legend">五条件内部比较</span></div><div class="scroll"><table><thead><tr><th>Condition</th>${head}</tr></thead><tbody>${body}</tbody></table></div></section>`;
  }).join('');
}
async function refresh(){try{const r=await fetch('/api/pck-extreme-benchmark/summary',{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);render(await r.json())}catch(e){document.getElementById('tables').innerHTML=`<div class="error">读取失败：${esc(e)}</div>`}}
refresh();setInterval(refresh,10000);
</script></body></html>'''


ABLATION_VIDEOS_PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCK Head 消融视频矩阵</title><style>
:root{--sand:#ebe5d6;--ink:#18221f;--card:#fffdf8;--line:#b9af9a;--rust:#b64a31;--green:#176654;--dark:#142820}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 3% 0,#d7764b30,transparent 34rem),radial-gradient(circle at 97% 4%,#4b9a8035,transparent 36rem),var(--sand);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1900px,calc(100% - 22px));margin:auto;padding:24px 0 90px}a{color:var(--green)}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}.eyebrow{margin-top:16px;color:var(--rust);font-size:12px;font-weight:900;letter-spacing:.16em}h1{font-size:clamp(38px,6vw,76px);line-height:.94;letter-spacing:-.045em;margin:10px 0}.lead{max-width:1050px;line-height:1.6}.toolbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:20px 0;padding:12px;background:#f8f3e8e8;border:1px solid var(--line);backdrop-filter:blur(12px);box-shadow:0 9px 30px #5448331a}.toolbar select{min-width:min(650px,100%);flex:1;padding:10px 12px;background:white;border:1px solid var(--line);font-weight:800}.status{font-size:12px;color:#6d6557}.model{margin:32px 0}.model h2{font-size:30px;margin:0 0 10px}.grid{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);padding:10px;border-radius:3px 20px 3px 3px}.card h3{margin:2px 0 9px;font-size:13px}.card.adaptive{border-color:#4c8e78;background:#f5fbf7}.card video{display:block;width:100%;aspect-ratio:16/9;background:#161b19;object-fit:contain}.pending{display:grid;place-items:center;width:100%;aspect-ratio:16/9;background:repeating-linear-gradient(135deg,#e8e1d3,#e8e1d3 10px,#f5f0e6 10px,#f5f0e6 20px);color:#756d5f;font-size:12px}.replay{position:fixed;right:18px;bottom:18px;z-index:30;border:0;border-radius:99px;padding:14px 20px;background:var(--dark);color:white;font-weight:900;box-shadow:0 12px 35px #10251d55;cursor:pointer}.empty{padding:35px;background:var(--card);border:1px solid var(--line)}@media(max-width:1250px){.grid{grid-template-columns:repeat(3,minmax(220px,1fr))}}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(190px,1fr))}}@media(max-width:560px){main{width:calc(100% - 10px)}.grid{grid-template-columns:1fr}.toolbar select{min-width:100%}.replay{right:10px;bottom:10px}}
.timeline-wrap{overflow:auto;border:1px solid var(--line);background:#f8f3e8a8;box-shadow:0 12px 34px #5448331a}.timeline{display:grid;grid-template-columns:220px repeat(6,minmax(285px,1fr));min-width:1930px}.corner,.time-head,.model-head,.time-cell{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px}.corner,.time-head{position:sticky;top:0;z-index:8;background:#ded6c4;font-weight:900}.corner,.model-head{position:sticky;left:0;z-index:9}.corner{z-index:12}.model-head{background:#17352c;color:white;display:flex;align-items:flex-start;justify-content:center;flex-direction:column}.model-head h2{font-size:23px;margin:0 0 5px}.model-head small{opacity:.75}.time-head{text-align:center;font-size:15px}.time-cell{background:#eee8da;display:flex;flex-direction:column;gap:9px;min-height:210px}.cell-empty{display:grid;place-items:center;min-height:170px;border:1px dashed var(--line);color:#756d5f}@media(max-width:900px){.timeline{grid-template-columns:160px repeat(6,minmax(250px,1fr));min-width:1660px}.model-head h2{font-size:19px}}
</style></head><body><main><a href="/">返回可视化总览</a><div class="eyebrow">CASE-CENTRIC ABLATION ATLAS</div><h1>一个 Case<br>看完所有消融</h1><p class="lead">按 case 聚合两个模型的 Original、静态 PCK Top/Bottom30、静态 Top/Bottom100，以及每个去噪 step 动态选择 Head 的 Top/Bottom30。视频不循环播放。</p><div class="toolbar"><select id="caseSelect"></select><span class="status" id="status">正在读取...</span></div><div id="content"></div></main><button class="replay" id="replay">重新全部播放</button><script>
let catalog=null,currentCase='';const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function videoUrl(model,caseKey,condition){const q=new URLSearchParams({model,case:caseKey,condition});return `/api/pck-ablation-case-videos/video?${q}`}
const TIMES=[['original','Original'],['s00_09','S00–S09'],['s10_19','S10–S19'],['s20_29','S20–S29'],['s30_39','S30–S39'],['s00_39','S00–S39']];
function timeKey(v){const c=String(v.condition||'').toLowerCase();if(c==='original')return'original';if(c.includes('s00_09')||c.includes('steps_00_10'))return's00_09';if(c.includes('s10_19')||c.includes('steps_10_20'))return's10_19';if(c.includes('s20_29')||c.includes('steps_20_30'))return's20_29';if(c.includes('s30_39')||c.includes('steps_30_40'))return's30_39';return's00_39'}
function videoCard(row,item,v){return `<article class="card ${v.condition.startsWith('adaptive_')?'adaptive':''}"><h3>${esc(v.label)}</h3>${v.ready?`<video controls preload="metadata" playsinline src="${videoUrl(row.model,item.case,v.condition)}"></video>`:'<div class="pending">生成中 / 暂不可用</div>'}</article>`}
function render(){const item=catalog.cases.find(x=>x.case===currentCase);if(!item){document.getElementById('content').innerHTML='<div class="empty">暂无 case</div>';return}const head=`<div class="corner">模型 / 时间</div>${TIMES.map(x=>`<div class="time-head">${x[1]}</div>`).join('')}`;const body=item.models.map(row=>{const grouped=Object.fromEntries(TIMES.map(x=>[x[0],[]]));row.videos.forEach(v=>grouped[timeKey(v)].push(v));return `<div class="model-head"><h2>${esc(row.label)}</h2><small>每列一个去噪时间区间</small></div>${TIMES.map(([key])=>`<div class="time-cell">${grouped[key].length?grouped[key].map(v=>videoCard(row,item,v)).join(''):'<div class="cell-empty">该时间无实验</div>'}</div>`).join('')}`}).join('');document.getElementById('content').innerHTML=`<div class="timeline-wrap"><div class="timeline">${head}${body}</div></div>`;document.title=`${item.case} · PCK 消融视频`}
function setCase(key){currentCase=key;document.getElementById('caseSelect').value=key;const u=new URL(location.href);u.searchParams.set('case',key);history.replaceState(null,'',u);render()}
async function refresh(){const r=await fetch('/api/pck-ablation-case-videos/catalog',{cache:'no-store'});catalog=await r.json();const select=document.getElementById('caseSelect');const wanted=new URL(location.href).searchParams.get('case');const previous=currentCase;if(!select.options.length||select.options.length!==catalog.cases.length){select.innerHTML=catalog.cases.map(x=>`<option value="${esc(x.case)}">${esc(x.case)}</option>`).join('')}const keys=new Set(catalog.cases.map(x=>x.case));currentCase=keys.has(previous)?previous:keys.has(wanted)?wanted:(catalog.cases[0]?.case||'');document.getElementById('status').textContent=`${catalog.ready_videos}/${catalog.expected_videos} videos · ${catalog.num_cases} cases · 自动刷新`;if(currentCase){select.value=currentCase;render()}}
document.getElementById('caseSelect').addEventListener('change',e=>setCase(e.target.value));document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));refresh();setInterval(refresh,10000);
</script></body></html>'''


PORTAL_CARD = r'''
<a class="card new" href="/pck-extreme-benchmark?v=1"><div><span>10 / 消融指标</span><h2>Top / Bottom Head 指标表</h2><p>汇总 Baseline 与 LoRA 的 Original、Top/Bottom 30 和 Top/Bottom 100 全步置零结果，实时标注模型内最佳值。</p></div><span class="go">打开消融指标表</span></a>
'''
VIDEOS_PORTAL_CARD = r'''
<a class="card new" href="/pck-ablation-case-videos?v=1"><div><span>11 / CASE 视频矩阵</span><h2>每个 Case 的全部消融</h2><p>按 case 汇总两个模型的 Original、静态 Top/Bottom30、Top/Bottom100 和 step-adaptive Top/Bottom30。</p></div><span class="go">打开 CASE 消融视频矩阵</span></a>
'''
QK_ATTENTION_PORTAL_CARD = r'''
<a class="card new" href="/qk-noise-attention-compare?v=2"><div><span>12 / Attention 扰动对比</span><h2>Probability Noise Before / After</h2><p>对比 α=0.30 概率空间加噪前后的全 token 与潜变量帧注意力，Q/K/V 保持不变。</p></div><span class="go">打开注意力对比</span></a>
'''
ATTENTION_LORA_PORTAL_CARD = r'''
<a class="card new" href="/attention-additive-lora-case?v=3"><div><span>13 / LORA ATTENTION 消融</span><h2>Top/Bottom 30/100 × α</h2><p>单页汇总 001460 case 的 Baseline、LoRA Original、16 组概率空间 attention 扰动视频与热力图。</p></div><span class="go">打开 LoRA Attention 对比</span></a>
'''
viewer.PORTAL = viewer.PORTAL.replace(
    "</section>", PORTAL_CARD + VIDEOS_PORTAL_CARD + QK_ATTENTION_PORTAL_CARD + ATTENTION_LORA_PORTAL_CARD + "</section>", 1
)


from AAA_my_test import serve_attention_noise_metrics as combined_metrics


class MetricsHandler(viewer.Handler):
    def do_GET(self) -> None:
        if urlparse(self.path).path == "/pck-extreme-benchmark":
            self.send_payload(
                combined_metrics.PAGE.encode("utf-8"), "text/html; charset=utf-8"
            )
            return
        path = urlparse(self.path).path
        if path == "/pck-extreme-benchmark":
            self.send_payload(METRICS_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/pck-extreme-benchmark/summary":
            payload = json.dumps(
                combined_metrics.build_summary(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/pck-ablation-case-videos":
            self.send_payload(ABLATION_VIDEOS_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/pck-ablation-case-videos/catalog":
            payload = json.dumps(ablation_video_catalog(), ensure_ascii=False).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/pck-ablation-case-videos/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            model = params.get("model", [""])[0]
            case_key = params.get("case", [""])[0]
            condition = params.get("condition", [""])[0]
            video_path = ablation_video_path(model, case_key, condition)
            if video_path is None or not video_path.is_file():
                raise FileNotFoundError("unknown ablation video")
            viewer.send_file_with_range(self, video_path, "video/mp4")
            return
        if path == "/qk-noise-attention-compare":
            self.send_payload(
                qk_noise_attention_compare_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/qk-noise-attention-compare/catalog":
            payload = json.dumps(
                qk_noise_attention_compare_catalog(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/qk-noise-attention-compare/file":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            image_path = qk_noise_attention_compare_file(
                params.get("name", [""])[0]
            )
            if image_path is None:
                raise FileNotFoundError("unknown Q@K attention image")
            viewer.send_file_with_range(self, image_path, "image/png")
            return
        if path == "/attention-additive-lora-case":
            self.send_payload(
                attention_lora_case_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/attention-additive-lora-case/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                attention_lora_case_catalog(params.get("case", [""])[0]),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path in {
            "/api/attention-additive-lora-case/video",
            "/api/attention-additive-lora-case/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = attention_lora_case_asset(
                params.get("id", [""])[0], params.get("case", [""])[0]
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown LoRA attention asset")
            content_type = "video/mp4" if path.endswith("/video") else "image/png"
            viewer.send_file_with_range(self, asset, content_type)
            return
        super().do_GET()


viewer.Handler = MetricsHandler


def qk_noise_attention_compare_catalog():
    records = []
    if QK_ATTENTION_COMPARE_ROOT.exists():
        for path in sorted(QK_ATTENTION_COMPARE_ROOT.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return {"records": records}


def qk_noise_attention_compare_file(requested_name: str):
    name = Path(requested_name).name
    path = QK_ATTENTION_COMPARE_ROOT / name
    if not name or not path.is_file():
        return None
    return path


def qk_noise_attention_compare_page():
    return r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attention Probability Noise Compare</title>
<style>
:root{--ink:#17221d;--muted:#607068;--paper:#f3efe3;--card:#fffdf7;--line:#c9c1ae;--top:#b83f2f;--bottom:#146b64}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 12% 8%,#f8d9ad 0,transparent 28%),linear-gradient(135deg,#ece5d4,#dce9df);font-family:"Noto Serif SC","Source Han Serif SC",serif}
header{position:sticky;top:0;z-index:3;padding:18px 26px;background:rgba(243,239,227,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(9px)}
h1{margin:0;font-size:clamp(22px,3vw,38px);letter-spacing:.02em}header p{margin:6px 0 0;color:var(--muted)}main{padding:22px;max-width:1700px;margin:auto}.status{padding:18px;border:1px dashed var(--line);background:var(--card)}
.row{margin:22px 0;padding:20px;border:1px solid var(--line);border-radius:18px;background:rgba(255,253,247,.92);box-shadow:0 12px 34px rgba(45,52,43,.09)}.row.top{border-left:8px solid var(--top)}.row.bottom{border-left:8px solid var(--bottom)}
.title{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}.title h2{margin:0;font-size:25px}.meta{color:var(--muted);font-family:ui-monospace,SFMono-Regular,monospace;font-size:13px}
.images{display:grid;grid-template-columns:1.25fr 1fr;gap:16px;margin-top:14px}.images figure{margin:0}.images img{width:100%;display:block;border:1px solid var(--line);background:#fff}.images figcaption{margin-top:7px;color:var(--muted)}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}.pill{padding:7px 10px;border-radius:999px;background:#ebe5d6;font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}@media(max-width:900px){.images{grid-template-columns:1fr}header{position:static}main{padding:12px}}
</style></head><body><header><h1>Attention Probability Noise α=0.30</h1><p>A′ = normalize(clamp(A + α/K · ε, 0))；Q/K/V 不变；S039，Baseline pilot case。</p></header><main id="main"><div class="status">正在等待注意力捕获结果，页面会自动刷新。</div></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>Number(x).toExponential(4);
async function load(){const data=await fetch('/api/qk-noise-attention-compare/catalog',{cache:'no-store'}).then(r=>r.json());const root=document.getElementById('main');if(!data.records.length)return;data.records.sort((a,b)=>a.group==='top30'?-1:b.group==='top30'?1:0);root.innerHTML=data.records.map(r=>`<section class="row ${r.group==='top30'?'top':'bottom'}"><div class="title"><h2>${esc(r.group.toUpperCase())}</h2><span class="meta">${esc(r.case)} · S${String(r.step).padStart(3,'0')} · ${r.unique_block_heads} block-heads</span></div><div class="images"><figure><img src="/api/qk-noise-attention-compare/file?name=${encodeURIComponent(r.all_token_image)}"><figcaption>全 token：Before / After / 逐 head 平均 |ΔA|</figcaption></figure><figure><img src="/api/qk-noise-attention-compare/file?name=${encodeURIComponent(r.frame_image)}"><figcaption>7×7 潜变量帧注意力质量</figcaption></figure></div><div class="stats"><span class="pill">mean per-head |ΔA| ${fmt(r.mean_abs_attention_delta)}</span><span class="pill">clipped ${(100*r.clipped_fraction).toFixed(2)}%</span><span class="pill">row-sum error ${fmt(r.max_row_sum_error)}</span><span class="pill">entropy ${r.before_mean_row_entropy.toFixed(4)} → ${r.after_mean_row_entropy.toFixed(4)}</span><span class="pill">Q/K/V modified: ${r.qkv_modified}</span></div></section>`).join('');}
load().catch(e=>document.getElementById('main').textContent=e);setInterval(load,5000);
</script></body></html>'''


def _attention_lora_runs():
    for alpha_tag, alpha in (
        ("030", 0.3),
        ("060", 0.6),
        ("090", 0.9),
        ("150", 1.5),
    ):
        for count in (30, 100):
            key = f"alpha{alpha_tag}_count{count}"
            yield key, alpha, count, ATTENTION_LORA_CASE_ROOT / key


def _attention_replacement_run_root(model_slug: str, intervention: str):
    if model_slug not in {"baseline", "lora", "full_sa"}:
        return None
    if intervention not in {"zero", "uniform", "temporal_causal"}:
        return None
    return ATTENTION_REPLACEMENT_ROOT / model_slug / f"{intervention}_count100"


def _attention_replacement_asset(
    model_slug: str,
    intervention: str,
    group: str,
    kind: str,
    name: str,
    case_key: str,
):
    if group not in {"top100", "bottom100"} or Path(name).name != name:
        return None
    run_root = _attention_replacement_run_root(model_slug, intervention)
    if run_root is None:
        return None
    if kind == "image":
        if model_slug == "full_sa":
            return run_root / "_attention_heatmaps" / f"{group}_{intervention}" / name
        return run_root / "heatmaps" / name
    if kind != "video":
        return None
    if model_slug != "full_sa":
        return run_root / "videos" / model_slug / "cases" / case_key / name
    for variant_dir in reversed(sorted(run_root.glob(f"*_{group}_{intervention}"))):
        for video in sorted(variant_dir.rglob(name)):
            if case_key in str(video):
                return video
    return None


def _attention_replacement_metadata(
    model_slug: str, intervention: str, group: str, case_key: str
):
    run_root = _attention_replacement_run_root(model_slug, intervention)
    if run_root is None:
        return None
    if model_slug == "full_sa":
        capture_root = run_root / "_attention_heatmaps" / f"{group}_{intervention}"
        pattern = f"full_sa__{case_key}__{group}__step39.json"
    else:
        capture_root = run_root / "heatmaps"
        pattern = f"*__{case_key}__{group}__step39.json"
    return next(iter(sorted(capture_root.glob(pattern))), None)


def _attention_baseline_runs():
    for alpha_tag, alpha in (("090", 0.9), ("150", 1.5)):
        for count in (30, 100):
            key = f"baseline_alpha{alpha_tag}_count{count}"
            yield key, alpha, count, BASELINE_ATTENTION_ROOT / f"alpha{alpha_tag}_count{count}"


def _attention_case_keys():
    if not ATTENTION_TEST_LIST.is_file():
        return [ATTENTION_LORA_CASE]
    keys = []
    for line in ATTENTION_TEST_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            key = Path(line).stem
            if key not in keys:
                keys.append(key)
    return keys or [ATTENTION_LORA_CASE]


def attention_lora_case_asset(asset_id: str, case_key: str = ""):
    cases = _attention_case_keys()
    if case_key not in cases:
        case_key = ATTENTION_LORA_CASE if ATTENTION_LORA_CASE in cases else cases[0]
    if asset_id == "control_baseline":
        return STATIC30_ROOT / "baseline" / "cases" / case_key / "original.mp4"
    if asset_id == "control_lora":
        static = STATIC30_ROOT / "lora" / "cases" / case_key / "original.mp4"
        if static.is_file():
            return static
        for _key, _alpha, _count, run_root in _attention_lora_runs():
            generated = run_root / "videos" / "lora" / "cases" / case_key / "original.mp4"
            if generated.is_file():
                return generated
        return static
    if asset_id == "control_full_sa":
        return _full_sa_attention_video("baseline", case_key)
    if asset_id.startswith("fullsa_image::"):
        try:
            _prefix, label, name = asset_id.split("::", 2)
        except ValueError:
            return None
        if Path(label).name != label or Path(name).name != name:
            return None
        return FULL_SA_ATTENTION_ROOT / "_attention_heatmaps" / label / name
    if asset_id.startswith("fullsa::"):
        return _full_sa_attention_video(asset_id.split("::", 1)[1], case_key)
    if asset_id.startswith("replacement::"):
        try:
            _prefix, model_slug, intervention, group, kind, name = asset_id.split("::", 5)
        except ValueError:
            return None
        return _attention_replacement_asset(
            model_slug, intervention, group, kind, name, case_key
        )
    try:
        run_key, kind, name = asset_id.split("::", 2)
    except ValueError:
        return None
    run = next(
        (
            (*item, "lora")
            for item in _attention_lora_runs()
            if item[0] == run_key
        ),
        None,
    )
    if run is None:
        run = next(
            (
                (*item, "baseline")
                for item in _attention_baseline_runs()
                if item[0] == run_key
            ),
            None,
        )
    if run is None or Path(name).name != name:
        return None
    run_root = run[3]
    model_slug = run[4]
    if kind == "image":
        return run_root / name
    if kind == "video":
        return run_root / "videos" / model_slug / "cases" / case_key / name
    return None


def _full_sa_attention_video(label: str, case_key: str):
    if Path(label).name != label or not FULL_SA_ATTENTION_ROOT.exists():
        return None
    variant_dirs = sorted(FULL_SA_ATTENTION_ROOT.glob(f"*_{label}"))
    for variant_dir in reversed(variant_dirs):
        for video in sorted(variant_dir.rglob("*.mp4")):
            if case_key in str(video):
                return video
    return None


def _full_sa_attention_metadata(label: str, case_key: str, group: str):
    capture_root = FULL_SA_ATTENTION_ROOT / "_attention_heatmaps" / label
    return next(
        iter(sorted(capture_root.glob(f"full_sa__{case_key}__{group}__step39.json"))),
        None,
    )


def attention_lora_case_catalog(requested_case: str = ""):
    cases = _attention_case_keys()
    case_key = requested_case if requested_case in cases else (
        ATTENTION_LORA_CASE if ATTENTION_LORA_CASE in cases else cases[0]
    )
    controls = []
    for asset_id, label in (
        ("control_baseline", "参考模型：Wan2.2 Baseline · Original"),
        ("control_lora", "消融模型：Wan+LoRA · Original"),
        ("control_full_sa", "消融模型：Full-SA no-object step-002500 · Original"),
    ):
        path = attention_lora_case_asset(asset_id, case_key)
        controls.append({"id": asset_id, "label": label, "ready": bool(path and path.is_file())})
    records = []
    for run_key, alpha, count, run_root in _attention_baseline_runs():
        for group in (f"top{count}", f"bottom{count}"):
            metadata_path = next(
                iter(sorted(run_root.glob(f"*__{case_key}__{group}__step39.json"))),
                None,
            )
            metadata = {}
            if metadata_path is not None:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
            video_name = f"{group}_steps_00_40.mp4"
            video_id = f"{run_key}::video::{video_name}"
            video_path = attention_lora_case_asset(video_id, case_key)
            all_token = metadata.get("all_token_image", "")
            frame = metadata.get("frame_image", "")
            records.append(
                {
                    "run_key": run_key,
                    "model": "Wan2.2 Baseline",
                    "alpha": alpha,
                    "count": count,
                    "group": group,
                    "video_id": video_id,
                    "video_ready": bool(video_path and video_path.is_file()),
                    "all_token_id": f"{run_key}::image::{all_token}" if all_token else "",
                    "frame_id": f"{run_key}::image::{frame}" if frame else "",
                    "heatmap_ready": bool(all_token and frame),
                    "heatmap_expected": True,
                    "metrics": metadata,
                }
            )
    for run_key, alpha, count, run_root in _attention_lora_runs():
        for group in (f"top{count}", f"bottom{count}"):
            metadata_path = next(
                iter(sorted(run_root.glob(f"*__{case_key}__{group}__step39.json"))),
                None,
            )
            metadata = {}
            if metadata_path is not None:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
            video_name = f"{group}_steps_00_40.mp4"
            video_id = f"{run_key}::video::{video_name}"
            video_path = attention_lora_case_asset(video_id, case_key)
            all_token = metadata.get("all_token_image", "")
            frame = metadata.get("frame_image", "")
            records.append(
                {
                    "run_key": run_key,
                    "model": "Wan+LoRA",
                    "alpha": alpha,
                    "count": count,
                    "group": group,
                    "video_id": video_id,
                    "video_ready": bool(video_path and video_path.is_file()),
                    "all_token_id": f"{run_key}::image::{all_token}" if all_token else "",
                    "frame_id": f"{run_key}::image::{frame}" if frame else "",
                    "heatmap_ready": bool(
                        all_token
                        and frame
                        and attention_lora_case_asset(
                            f"{run_key}::image::{all_token}", case_key
                        ).is_file()
                        and attention_lora_case_asset(
                            f"{run_key}::image::{frame}", case_key
                        ).is_file()
                    ),
                    "metrics": metadata,
                    "heatmap_expected": True,
                }
            )
    for alpha in (0.9, 1.5):
        alpha_tag = str(alpha).replace(".", "p")
        for count in (30, 100):
            for direction in ("top", "bottom"):
                group = f"{direction}{count}"
                label = f"{group}_alpha{alpha_tag}"
                video_id = f"fullsa::{label}"
                video_path = attention_lora_case_asset(video_id, case_key)
                metadata_path = _full_sa_attention_metadata(label, case_key, group)
                metadata = {}
                if metadata_path is not None:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                all_token = metadata.get("all_token_image", "")
                frame = metadata.get("frame_image", "")
                records.append(
                    {
                        "run_key": label,
                        "model": "Full-SA no-object · step-002500",
                        "alpha": alpha,
                        "count": count,
                        "group": group,
                        "video_id": video_id,
                        "video_ready": bool(video_path and video_path.is_file()),
                        "all_token_id": f"fullsa_image::{label}::{all_token}" if all_token else "",
                        "frame_id": f"fullsa_image::{label}::{frame}" if frame else "",
                        "heatmap_ready": bool(all_token and frame),
                        "heatmap_expected": True,
                        "metrics": metadata,
                    }
                )
    replacement_models = (
        ("baseline", "Wan2.2 Baseline"),
        ("lora", "Wan+LoRA"),
        ("full_sa", "Full-SA no-object · step-002500"),
    )
    for model_slug, model_label in replacement_models:
        for intervention in ("zero", "uniform", "temporal_causal"):
            for group in ("top100", "bottom100"):
                metadata_path = _attention_replacement_metadata(
                    model_slug, intervention, group, case_key
                )
                metadata = {}
                if metadata_path is not None:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                video_name = f"{group}_steps_00_40.mp4"
                video_id = (
                    f"replacement::{model_slug}::{intervention}::{group}::video::{video_name}"
                )
                video_path = attention_lora_case_asset(video_id, case_key)
                all_token = metadata.get("all_token_image", "")
                frame = metadata.get("frame_image", "")
                all_token_id = (
                    f"replacement::{model_slug}::{intervention}::{group}::image::{all_token}"
                    if all_token
                    else ""
                )
                frame_id = (
                    f"replacement::{model_slug}::{intervention}::{group}::image::{frame}"
                    if frame
                    else ""
                )
                all_token_path = attention_lora_case_asset(all_token_id, case_key) if all_token_id else None
                frame_path = attention_lora_case_asset(frame_id, case_key) if frame_id else None
                records.append(
                    {
                        "run_key": f"{model_slug}_{intervention}_count100",
                        "model": model_label,
                        "experiment": "replacement",
                        "intervention": intervention,
                        "count": 100,
                        "group": group,
                        "video_id": video_id,
                        "video_ready": bool(video_path and video_path.is_file()),
                        "all_token_id": all_token_id,
                        "frame_id": frame_id,
                        "heatmap_ready": bool(
                            all_token_path
                            and frame_path
                            and all_token_path.is_file()
                            and frame_path.is_file()
                        ),
                        "heatmap_expected": True,
                        "metrics": metadata,
                    }
                )
    return {
        "case": case_key,
        "cases": cases,
        "controls": controls,
        "records": records,
        "ready_records": sum(
            r["video_ready"] and (r["heatmap_ready"] or not r["heatmap_expected"])
            for r in records
        ),
        "expected_records": len(records),
    }


def attention_lora_case_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan+LoRA Attention Probability Noise</title><style>
:root{--ink:#18211e;--paper:#eee9dc;--card:#fffdf7;--line:#c8bda7;--red:#b94332;--green:#176b61}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 8% 4%,#f6cf9c,transparent 25%),linear-gradient(145deg,#ece5d4,#d9e8df);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:5;padding:17px 24px;background:rgba(238,233,220,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(9px)}h1{margin:0;font-size:clamp(23px,3vw,39px)}header p{margin:5px 0;color:#5e6c65}.status{font-family:ui-monospace,monospace;font-size:13px}main{max-width:1800px;margin:auto;padding:20px}.controls,.control-grid,.matrix-head,.row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.control,.cell,.row-head,.alpha-title,.card{background:rgba(255,253,247,.94);border:1px solid var(--line);border-radius:16px;padding:15px;box-shadow:0 10px 28px rgba(39,48,42,.08)}h2{margin:7px 0 12px}.card.top{border-left:7px solid var(--red)}.card.bottom{border-left:7px solid var(--green)}video,img{display:block;width:100%;background:#151816;border:1px solid var(--line)}.heatmaps{display:grid;grid-template-columns:1.25fr 1fr;gap:10px;margin-top:11px}.pending{min-height:170px;display:grid;place-items:center;border:1px dashed var(--line);color:#68736d}.meta{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}.pill{padding:5px 8px;border-radius:999px;background:#e9e2d2;font:12px ui-monospace,monospace}.section-title{margin:28px 0 13px}
.matrix{display:flex;flex-direction:column;gap:10px;min-width:4020px}.matrix-head{display:grid;grid-template-columns:260px repeat(6,minmax(620px,1fr));gap:10px;min-width:4020px;font-weight:700;color:#5e6c65}
.row{display:grid;grid-template-columns:260px repeat(6,minmax(620px,1fr));gap:10px;min-width:4020px}.row-head{padding:12px}.row-head .title{margin:4px 0;font-size:19px;font-family:"Trebuchet MS","Noto Serif CJK SC",sans-serif;font-weight:900}.row-head .sub{color:#5e6c65;font-size:12px}.alpha-title,.cell-inner{text-align:center}.alpha-title{padding:10px;font-size:17px}.cell{padding:8px}.cell.paired{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;align-items:start}.cell.paired>.card{margin:0}
.replay-all{position:fixed;right:22px;bottom:22px;z-index:20;border:0;border-radius:999px;padding:13px 20px;background:#172e27;color:#fff;font:700 14px ui-monospace,monospace;box-shadow:0 10px 28px rgba(20,35,29,.3);cursor:pointer}.replay-all:hover{background:#b94332}.replay-all:active{transform:translateY(1px)}.manual-refresh{margin-left:10px;border:1px solid #748078;border-radius:999px;padding:8px 14px;background:#fff;color:#172e27;font-weight:800;cursor:pointer}.manual-refresh:hover{border-color:#b94332;color:#b94332}
@media(max-width:980px){.controls,.control-grid,.heatmaps,.matrix-head,.row,.cell.paired{grid-template-columns:1fr}header{position:static}main{padding:11px}.row,.matrix-head{grid-template-columns:1fr}}
</style></head><body><button id="replayAll" class="replay-all" type="button">重新播放全部</button><header><h1>Attention Probability Noise Ablation</h1><p><strong>消融模型：Wan+LoRA、Full-SA no-object step-002500</strong> · 参考模型：Wan2.2 Baseline</p><p><label for="caseSelect">Case：</label><select id="caseSelect"></select><button id="manualRefresh" class="manual-refresh" type="button">手动刷新结果</button></p><p id="case"></p><div id="status" class="status">加载中</div></header><main><h2 class="section-title">Original controls（参考模型 vs 消融模型）</h2><section id="controls" class="controls"></section><h2 class="section-title">Adaptive Top/Bottom 30/100 · Additive Noise 与 Attention Replacement</h2><p>A=0：选中 head 输出归零；A=1：按行归一化为均匀注意力 A=1/N<sub>K</sub>。</p><section id="matrix-head" class="matrix-head"></section><section id="grid" class="matrix"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let currentCase=new URL(location.href).searchParams.get('case')||'';const v=id=>`/api/attention-additive-lora-case/video?id=${encodeURIComponent(id)}&case=${encodeURIComponent(currentCase)}`;const im=id=>`/api/attention-additive-lora-case/image?id=${encodeURIComponent(id)}&case=${encodeURIComponent(currentCase)}`;const f=x=>Number(x).toExponential(3);
function renderCell(r,expectedGroup){const m=r?r.metrics||{}:{ },intervention=r&&r.experiment==='replacement'?(r.intervention==='zero'?'A=0':r.intervention==='uniform'?'A=1/N_K':'Temporal Causal Mask'):r?'α='+r.alpha.toFixed(1):'';const label=r?(r.group||'').toUpperCase()+' · '+intervention:expectedGroup.toUpperCase()+' · 未完成';if(!r){return `<article class="card ${expectedGroup.startsWith('top')?'top':'bottom'}"><div class="alpha-title"><strong>${e(label)}</strong></div><div class="pending">该实验的新结果尚未生成</div></article>`;}const heatmaps=r.heatmap_ready?`<div class="heatmaps"><img loading="lazy" src="${im(r.all_token_id)}"><img loading="lazy" src="${im(r.frame_id)}"></div>`:r.heatmap_expected?`<div class="heatmaps"><div class="pending">S039 全 token 热力图生成中</div><div class="pending">帧级热力图生成中</div></div>`:`<div class="pending">新统一推理结果尚未采集热力图</div>`;return `<article class="card ${r.group&&r.group.startsWith('top')?'top':'bottom'}"><div class="alpha-title"><strong>${e(label)}</strong></div><div class="meta"><span class="pill">消融模型：${e(r.model)}</span><span class="pill">统一配置：40步 · 49帧 · 热力图S039</span></div>${r.video_ready?`<video controls preload="metadata" playsinline src="${v(r.video_id)}"></video>`:`<div class="pending">${e(r.model)} 新视频生成中</div>`}${heatmaps}${r.heatmap_ready?`<div class="meta"><span class="pill">mean |ΔA| ${f(m.mean_abs_attention_delta)}</span><span class="pill">row error ${f(m.max_row_sum_error)}</span></div>`:''}</article>`}
function controlIdForModel(model){if(model.includes('Wan+LoRA'))return'control_lora';if(model.includes('Full-SA'))return'control_full_sa';return'control_baseline'}
function renderBaselineCell(model,controlsById){const control=controlsById[controlIdForModel(model)];if(!control||!control.ready){return `<article class="card"><div class="alpha-title"><strong>Baseline Original</strong></div><div class="pending">基线视频尚未就绪</div></article>`;}return `<article class="card"><div class="alpha-title"><strong>Baseline Original</strong></div><div class="meta"><span class="pill">基线：${e(control.label)}</span></div><video controls preload="metadata" playsinline src="${v(control.id)}"></video></article>`}
function makeRowMeta(r){return {name:`${r.model.startsWith('Wan+LoRA')?'Wan+LoRA':r.model.includes('Full-SA')?'Full-SA':'消融模型'} · Top/Bottom${r.count}`,sub:'同数量 Top 与 Bottom 并排 · 统一配置：40步 · 49帧'}}
function renderRows(records,columns,headId,gridId,controlsById){document.getElementById(headId).innerHTML=`<div class="row-head"><div class="title">模型 × Top/Bottom 数量</div></div><div class="alpha-title">Baseline Original</div>`+columns.map(c=>`<div class="alpha-title">${e(c.label)}</div>`).join('');const rows=new Map();for(const r of records){const count=String(r.count),key=`${r.model}::${count}`,column=r.experiment==='replacement'?r.intervention:Number(r.alpha).toFixed(1),direction=r.group.toLowerCase().startsWith('top')?'top':'bottom';if(!rows.has(key)){rows.set(key,{model:r.model,count,items:{}})}if(!rows.get(key).items[column])rows.get(key).items[column]={};rows.get(key).items[column][direction]=r}const ordered=Array.from(rows.values()).sort((a,b)=>a.model.localeCompare(b.model)||Number(a.count)-Number(b.count));document.getElementById(gridId).innerHTML=ordered.map(row=>{const meta=makeRowMeta(row);return `<article class="row"><div class="row-head"><div class="title">${e(meta.name)}</div><div class="sub">${e(meta.sub)}</div></div><div class="cell">${renderBaselineCell(row.model,controlsById)}</div>`+columns.map(c=>{const pair=row.items[c.key]||{};return `<div class="cell paired">${renderCell(pair.top,`top${row.count}`)}${renderCell(pair.bottom,`bottom${row.count}`)}</div>`}).join('')+`</article>`}).join('')}
async function load(){const d=await fetch(`/api/attention-additive-lora-case/catalog?case=${encodeURIComponent(currentCase)}`,{cache:'no-store'}).then(r=>r.json());currentCase=d.case;const select=document.getElementById('caseSelect');if(select.options.length!==d.cases.length){select.innerHTML=d.cases.map(x=>`<option value="${e(x)}">${e(x)}</option>`).join('')}select.value=currentCase;document.getElementById('case').textContent=`Case: ${d.case} · additive noise、normalized replacement 与 temporal causal mask`;const controlsById=Object.fromEntries(d.controls.map(x=>[x.id,x])),alphaKeys=new Set(['0.9','1.5']),additive=d.records.filter(r=>r.experiment!=='replacement'&&alphaKeys.has(Number(r.alpha).toFixed(1))),replacement=d.records.filter(r=>r.experiment==='replacement'),visible=[...additive,...replacement],columns=[{key:'0.9',label:'α = 0.9'},{key:'1.5',label:'α = 1.5'},{key:'zero',label:'A = 0'},{key:'uniform',label:'A = 1（归一化为 1/N_K）'},{key:'temporal_causal',label:'Temporal Causal Mask'}];document.getElementById('status').textContent=`${visible.filter(r=>r.video_ready).length}/${visible.length} visible videos ready · 点击按钮手动刷新`;document.getElementById('controls').innerHTML=d.controls.map(x=>`<article class="control"><h2>${e(x.label)}</h2>${x.ready?`<video controls preload="metadata" playsinline src="${v(x.id)}"></video>`:'<div class="pending">等待 Original 视频</div>'}</article>`).join('');renderRows(visible,columns,'matrix-head','grid',controlsById)}
document.getElementById('caseSelect').addEventListener('change',event=>{currentCase=event.target.value;const url=new URL(location.href);url.searchParams.set('case',currentCase);history.replaceState(null,'',url);load()});document.getElementById('manualRefresh').addEventListener('click',()=>load());document.getElementById('replayAll').addEventListener('click',()=>{document.querySelectorAll('video').forEach(video=>{video.pause();video.currentTime=0;video.loop=false;video.play().catch(()=>{})})});load();
</script></body></html>'''


if __name__ == "__main__":
    viewer.main()
