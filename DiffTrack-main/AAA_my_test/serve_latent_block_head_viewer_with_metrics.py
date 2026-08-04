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
QK_ATTENTION_MONO_SCALE_COMPARE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_test5"
)
QK_ATTENTION_MONO_SCALE_HEAD_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460"
)
ATTENTION_LORA_CASE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5/lora"
)
ATTENTION_LORA_STEPS00_09_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_wan_lora_steps00_09_case001460/lora"
)
ATTENTION_LORA_SEED_SWEEP_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
)
OBJECT_QUERY_OVERLAY_PILOT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_attention_overlay_case001460_seed090094"
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
<a class="card new" href="/qk-noise-attention-compare?v=2"><div><span>12 / Attention 扰动对比</span><h2>Attention Before / After</h2><p>对比不同 Q@K 干预（概率加噪 / 幂次放大）前后的全 token 与潜变量帧注意力，Q/K/V 保持不变。</p></div><span class="go">打开注意力对比</span></a>
'''
ATTENTION_LORA_PORTAL_CARD = r'''
<a class="card new" href="/attention-additive-lora-case?v=3"><div><span>13 / LORA ATTENTION 消融</span><h2>Top/Bottom 30/100 × α</h2><p>单页汇总 001460 case 的 Baseline、LoRA Original、16 组概率空间 attention 扰动视频与热力图。</p></div><span class="go">打开 LoRA Attention 对比</span></a>
'''
MONO_SCALE_HEAD_PORTAL_CARD = r'''
<a class="card new" href="/qk-mono-scale-heads?case=0613pybullet_sample_001460_w002&alpha=0.9"><div><span>14 / MONO-SCALE 逐 HEAD</span><h2>Attention 幂次拉伸逐 Head 热力图</h2><p>按模型和 Top/Bottom 组查看每个 block/head 在 S039 调整前、调整后与差值热力图。</p></div><span class="go">打开逐 Head 对比</span></a>
'''
MONO_SCALE_LORA_VIDEO_PORTAL_CARD = r'''
<a class="card new" href="/qk-mono-scale-lora-videos?case=0613pybullet_sample_001460_w002"><div><span>15 / MONO-SCALE LORA 视频</span><h2>Top/Bottom100 × α&lt;0.9</h2><p>统一对比 Wan2.2 Original、Wan+LoRA Original 与 α=0.3/0.6 的 Top100、Bottom100 生成视频。</p></div><span class="go">打开视频对比</span></a>
'''
ATTENTION_LORA_SEED_SWEEP_PORTAL_CARD = r'''
<a class="card new" href="/attention-additive-lora-seed-sweep?v=1"><div><span>16 / LORA 50-SEED SWEEP</span><h2>001460 · 50 Seeds × 全实验</h2><p>查看 Wan+LoRA Top/Bottom100 在全时间步与 S000-S009 的六类干预结果，seed 可下拉选择。</p></div><span class="go">打开 50-Seed 对比</span></a>
<a class="card new" href="/object-query-attention-overlay?v=1"><div><span>17 / OBJECT QUERY OVERLAY</span><h2>Top/Bottom10 · 13×13 Latent 时间轴</h2><p>将动态物体 Query tokens 的 Before、After 与 |Delta| attention overlay 到对应视频帧。</p></div><span class="go">打开 Object Query 对比</span></a>
'''
viewer.PORTAL = viewer.PORTAL.replace(
    "</section>", PORTAL_CARD + VIDEOS_PORTAL_CARD + QK_ATTENTION_PORTAL_CARD + ATTENTION_LORA_PORTAL_CARD + MONO_SCALE_HEAD_PORTAL_CARD + MONO_SCALE_LORA_VIDEO_PORTAL_CARD + ATTENTION_LORA_SEED_SWEEP_PORTAL_CARD + "</section>", 1
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
        if path == "/qk-mono-scale-heads":
            self.send_payload(
                qk_mono_scale_heads_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/qk-mono-scale-heads/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                qk_mono_scale_heads_catalog(
                    params.get("case", [""])[0], params.get("alpha", ["0.9"])[0]
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/qk-mono-scale-heads/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            image_path = qk_mono_scale_head_asset(params.get("name", [""])[0])
            if image_path is None:
                raise FileNotFoundError("unknown mono-scale per-head image")
            viewer.send_file_with_range(self, image_path, "image/png")
            return
        if path == "/qk-mono-scale-lora-videos":
            self.send_payload(
                qk_mono_scale_lora_videos_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/qk-mono-scale-lora-videos/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                qk_mono_scale_lora_video_catalog(params.get("case", [""])[0]),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/qk-mono-scale-lora-videos/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            video_path = qk_mono_scale_lora_video_asset(
                params.get("id", [""])[0], params.get("case", [""])[0]
            )
            if video_path is None or not video_path.is_file():
                raise FileNotFoundError("unknown mono-scale LoRA video")
            viewer.send_file_with_range(self, video_path, "video/mp4")
            return
        if path == "/api/qk-mono-scale-lora-videos/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            image_path = qk_mono_scale_head_asset(params.get("name", [""])[0])
            if image_path is None:
                raise FileNotFoundError("unknown mono-scale LoRA heatmap")
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
        if path == "/attention-additive-lora-seed-sweep":
            self.send_payload(
                attention_lora_seed_sweep_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/attention-additive-lora-seed-sweep/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                attention_lora_seed_sweep_catalog(params.get("seed", [""])[0]),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path in {
            "/api/attention-additive-lora-seed-sweep/video",
            "/api/attention-additive-lora-seed-sweep/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = attention_lora_seed_sweep_asset(
                params.get("seed", [""])[0],
                params.get("stage", [""])[0],
                params.get("profile", [""])[0],
                params.get("group", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown seed-sweep asset")
            content_type = "video/mp4" if path.endswith("/video") else "image/png"
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/object-query-attention-overlay":
            self.send_payload(
                object_query_attention_overlay_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/object-query-attention-overlay/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                object_query_attention_overlay_catalog(
                    params.get("stage", ["all_steps"])[0],
                    params.get("profile", ["alpha090"])[0],
                    params.get("group", ["top100"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-attention-overlay/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_attention_overlay_asset(
                params.get("stage", [""])[0],
                params.get("profile", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown object-query overlay")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        super().do_GET()


viewer.Handler = MetricsHandler


def qk_noise_attention_compare_catalog():
    records = []
    for root in (QK_ATTENTION_COMPARE_ROOT, QK_ATTENTION_MONO_SCALE_COMPARE_ROOT):
        if root.exists():
            for path in sorted(root.glob("*.json")):
                try:
                    records.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
    return {"records": records}


def qk_noise_attention_compare_file(requested_name: str):
    name = Path(requested_name).name
    if not name:
        return None
    path = None
    for root in (QK_ATTENTION_COMPARE_ROOT, QK_ATTENTION_MONO_SCALE_COMPARE_ROOT):
        candidate = root / name
        if candidate.is_file():
            path = candidate
            break
    if path is None:
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
</style></head><body><header><h1>Attention Noise Compare</h1><p>对比不同 Q@K 干预前后（含逐组 α/指数参数）在 Baseline / LoRA 的 attention 汇总热力图；Q/K/V 不变。</p></header><main id="main"><div class="status">正在等待注意力捕获结果，页面会自动刷新。</div></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>Number(x).toExponential(4);
async function load(){const data=await fetch('/api/qk-noise-attention-compare/catalog',{cache:'no-store'}).then(r=>r.json());const root=document.getElementById('main');if(!data.records.length){return;}data.records.sort((a,b)=>a.group===b.group?(String(a.intervention||'').localeCompare(String(b.intervention||''))):(a.group==='top30'?-1:b.group==='top30'?1:0));root.innerHTML=data.records.map(r=>`<section class=\"row ${r.group==='top30'?'top':'bottom'}\"><div class=\"title\"><h2>${esc(r.group.toUpperCase())}</h2><span class=\"meta\">${esc(r.case)} · ${esc(r.intervention || '')} · S${String(r.step).padStart(3,'0')} · ${r.unique_block_heads} block-heads</span></div><div class=\"images\"><figure><img src=\"/api/qk-noise-attention-compare/file?name=${encodeURIComponent(r.all_token_image)}\"><figcaption>全 token：Before / After / 逐 head 平均 |ΔA|</figcaption></figure><figure><img src=\"/api/qk-noise-attention-compare/file?name=${encodeURIComponent(r.frame_image)}\"><figcaption>7×7 潜变量帧注意力质量</figcaption></figure></div><div class=\"stats\"><span class=\"pill\">mean per-head |ΔA| ${fmt(r.mean_abs_attention_delta)}</span><span class=\"pill\">intervention ${esc(r.intervention || '-')}</span><span class=\"pill\">clipped ${r.clipped_fraction===null?'-':(100*r.clipped_fraction).toFixed(2)+'%'}</span><span class=\"pill\">row-sum error ${fmt(r.max_row_sum_error)}</span><span class=\"pill\">entropy ${r.before_mean_row_entropy.toFixed(4)} → ${r.after_mean_row_entropy.toFixed(4)}</span><span class=\"pill\">Q/K/V modified: ${r.qkv_modified}</span></div></section>`).join('');}
load().catch(e=>document.getElementById('main').textContent=e);setInterval(load,5000);
</script></body></html>'''


def qk_mono_scale_heads_catalog(case_key: str, alpha_text: str):
    try:
        requested_alpha = float(alpha_text)
    except ValueError:
        requested_alpha = 0.9
    records = []
    root = QK_ATTENTION_MONO_SCALE_HEAD_ROOT
    if root.exists():
        for metadata_path in sorted(root.rglob("*.json")):
            try:
                record = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            selected = record.get("selected_block_heads") or []
            recorded_case = str(record.get("case", ""))
            if (
                record.get("intervention") != "probability_mono_scale"
                or len(selected) != 1
                or (case_key and recorded_case not in {case_key, "case"})
                or not math.isclose(float(record.get("alpha", -1)), requested_alpha)
            ):
                continue
            group = str(record.get("group", ""))
            if "_b" not in group or "_h" not in group:
                continue
            base_group = group.split("_b", 1)[0]
            block_head = selected[0]
            all_token_path = metadata_path.parent / str(record.get("all_token_image", ""))
            frame_path = metadata_path.parent / str(record.get("frame_image", ""))
            records.append(
                {
                    **record,
                    "case": case_key if recorded_case == "case" else recorded_case,
                    "base_group": base_group,
                    "block": int(block_head["block"]),
                    "head": int(block_head["head"]),
                    "all_token_asset": str(all_token_path.relative_to(root)),
                    "frame_asset": str(frame_path.relative_to(root)),
                    "ready": all_token_path.is_file() and frame_path.is_file(),
                }
            )
    records.sort(
        key=lambda r: (
            str(r.get("model", "")),
            0 if str(r["base_group"]).startswith("top") else 1,
            int(str(r["base_group"]).replace("top", "").replace("bottom", "")),
            r["block"],
            r["head"],
        )
    )
    return {
        "case": case_key or "0613pybullet_sample_001460_w002",
        "alpha": requested_alpha,
        "records": records,
    }


def qk_mono_scale_head_asset(requested_name: str):
    if not requested_name:
        return None
    root = QK_ATTENTION_MONO_SCALE_HEAD_ROOT.resolve()
    candidate = (root / requested_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() and candidate.suffix.lower() == ".png" else None


def qk_mono_scale_heads_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mono-scale 逐 Head Attention</title><style>
:root{--ink:#1d2924;--paper:#ece6d8;--card:#fffdf8;--line:#c4b9a4;--red:#b74631;--green:#17685d}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 8% 0,#efb77f55,transparent 34rem),radial-gradient(circle at 95% 5%,#6aa18b55,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:5;padding:16px 22px;background:#ece6d8ee;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}h1{margin:0 0 6px;font-size:clamp(25px,4vw,45px)}p{margin:5px 0}.tools{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:11px}select,button{padding:9px 13px;border:1px solid var(--line);background:#fff;font-weight:800}main{max-width:1900px;margin:auto;padding:18px}.status{font-family:ui-monospace,monospace;color:#5c6962}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:4px 19px 4px 4px;padding:12px;box-shadow:0 10px 27px #4a453920}.card.top{border-left:7px solid var(--red)}.card.bottom{border-left:7px solid var(--green)}.title{display:flex;justify-content:space-between;gap:12px;align-items:center}.title h2{margin:0;font:900 18px "Trebuchet MS",sans-serif}.pill{padding:5px 8px;border-radius:99px;background:#e9e2d4;font:12px ui-monospace,monospace}.images{display:grid;grid-template-columns:1.3fr 1fr;gap:9px;margin-top:10px}.images img{display:block;width:100%;border:1px solid var(--line);background:#161916}.stats{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}.empty{padding:40px;border:1px dashed var(--line);background:var(--card)}@media(max-width:1050px){.grid,.images{grid-template-columns:1fr}header{position:static}main{padding:10px}}
</style></head><body><header><h1>Q@K Probability Mono-scale · 逐 Head</h1><p id="subtitle">A′ = A<sup>1+α</sup> / ΣA<sup>1+α</sup>；Q/K/V 不变，S039 捕获。</p><div class="tools"><label>模型 <select id="model"></select></label><label>组 <select id="group"></select></label><button id="refresh">手动刷新</button><span id="status" class="status">等待结果</span></div></header><main><section id="grid" class="grid"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const q=new URL(location.href).searchParams,caseKey=q.get('case')||'0613pybullet_sample_001460_w002',alpha=q.get('alpha')||'0.9',image=n=>`/api/qk-mono-scale-heads/image?name=${encodeURIComponent(n)}`,fmt=x=>Number(x).toExponential(3);let records=[];
function options(id,values){const node=document.getElementById(id),old=node.value;node.innerHTML=values.map(x=>`<option value="${e(x)}">${e(x)}</option>`).join('');if(values.includes(old))node.value=old}
function render(){const model=document.getElementById('model').value,group=document.getElementById('group').value,shown=records.filter(r=>r.model===model&&r.base_group===group);document.getElementById('status').textContent=`${shown.length} heads ready · Case ${caseKey} · α=${alpha}`;document.getElementById('grid').innerHTML=shown.length?shown.map(r=>`<article class="card ${r.base_group.startsWith('top')?'top':'bottom'}"><div class="title"><h2>Block ${String(r.block).padStart(2,'0')} / Head ${String(r.head).padStart(2,'0')}</h2><span class="pill">${e(r.model)} · ${e(r.base_group.toUpperCase())}</span></div><div class="images"><img loading="lazy" src="${image(r.all_token_asset)}" alt="all-token before after delta"><img loading="lazy" src="${image(r.frame_asset)}" alt="frame attention before after delta"></div><div class="stats"><span class="pill">mean |ΔA| ${fmt(r.mean_abs_attention_delta)}</span><span class="pill">entropy ${Number(r.before_mean_row_entropy).toFixed(4)} → ${Number(r.after_mean_row_entropy).toFixed(4)}</span><span class="pill">row error ${fmt(r.max_row_sum_error)}</span></div></article>`).join(''):`<div class="empty">该模型/组的逐-head 热力图尚未生成。</div>`}
async function load(){const data=await fetch(`/api/qk-mono-scale-heads/catalog?case=${encodeURIComponent(caseKey)}&alpha=${encodeURIComponent(alpha)}`,{cache:'no-store'}).then(r=>r.json());records=data.records;options('model',[...new Set(records.map(r=>r.model))]);options('group',[...new Set(records.filter(r=>r.model===document.getElementById('model').value).map(r=>r.base_group))]);render()}
document.getElementById('model').addEventListener('change',()=>{options('group',[...new Set(records.filter(r=>r.model===document.getElementById('model').value).map(r=>r.base_group))]);render()});document.getElementById('group').addEventListener('change',render);document.getElementById('refresh').addEventListener('click',load);load().catch(err=>document.getElementById('status').textContent=err);
</script></body></html>'''


def qk_mono_scale_lora_video_asset(asset_id: str, case_key: str):
    if Path(case_key).name != case_key or not case_key:
        return None
    root = QK_ATTENTION_MONO_SCALE_HEAD_ROOT
    if asset_id == "wan_original":
        return (
            root
            / "baseline/alpha090_count30/videos/baseline/cases"
            / case_key
            / "original.mp4"
        )
    if asset_id == "lora_original":
        return (
            root
            / "lora/alpha090_count30/videos/lora/cases"
            / case_key
            / "original.mp4"
        )
    variants = {
        "alpha030_top100": ("alpha030_count100", "top100_steps_00_40.mp4"),
        "alpha030_bottom100": ("alpha030_count100", "bottom100_steps_00_40.mp4"),
        "alpha060_top100": ("alpha060_count100", "top100_steps_00_40.mp4"),
        "alpha060_bottom100": ("alpha060_count100", "bottom100_steps_00_40.mp4"),
        "alpha090_top100": ("alpha090_count100", "top100_steps_00_40.mp4"),
        "alpha090_bottom100": ("alpha090_count100", "bottom100_steps_00_40.mp4"),
    }
    variant = variants.get(asset_id)
    if variant is None:
        return None
    run_key, video_name = variant
    return root / "lora" / run_key / "videos/lora/cases" / case_key / video_name


def qk_mono_scale_lora_video_catalog(case_key: str):
    if not case_key or Path(case_key).name != case_key:
        case_key = "0613pybullet_sample_001460_w002"
    controls = []
    for asset_id, label in (
        ("wan_original", "Wan2.2 Original"),
        ("lora_original", "Wan+LoRA Original"),
    ):
        path = qk_mono_scale_lora_video_asset(asset_id, case_key)
        controls.append({"id": asset_id, "label": label, "ready": bool(path and path.is_file())})
    records = []
    for alpha_tag, alpha in (("030", 0.3), ("060", 0.6), ("090", 0.9)):
        run_root = QK_ATTENTION_MONO_SCALE_HEAD_ROOT / "lora" / f"alpha{alpha_tag}_count100"
        for direction in ("top", "bottom"):
            asset_id = f"alpha{alpha_tag}_{direction}100"
            path = qk_mono_scale_lora_video_asset(asset_id, case_key)
            group = f"{direction}100"
            heatmaps = []
            if run_root.exists():
                for metadata_path in sorted(run_root.glob("*.json")):
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    metadata_group = str(metadata.get("group", ""))
                    if (
                        metadata.get("intervention") != "probability_mono_scale"
                        or not (
                            metadata_group == group
                            or metadata_group.startswith(f"{group}_b")
                        )
                    ):
                        continue
                    selected = metadata.get("selected_block_heads") or []
                    head = selected[0] if len(selected) == 1 else None
                    all_token_path = metadata_path.parent / str(
                        metadata.get("all_token_image", "")
                    )
                    frame_path = metadata_path.parent / str(metadata.get("frame_image", ""))
                    heatmaps.append(
                        {
                            "label": (
                                f"Block {int(head['block']):02d} / Head {int(head['head']):02d}"
                                if head
                                else f"{group.upper()} aggregate ({len(selected)} heads)"
                            ),
                            "block": int(head["block"]) if head else None,
                            "head": int(head["head"]) if head else None,
                            "all_token_asset": str(
                                all_token_path.relative_to(QK_ATTENTION_MONO_SCALE_HEAD_ROOT)
                            ),
                            "frame_asset": str(
                                frame_path.relative_to(QK_ATTENTION_MONO_SCALE_HEAD_ROOT)
                            ),
                            "ready": all_token_path.is_file() and frame_path.is_file(),
                            "mean_abs_attention_delta": metadata.get(
                                "mean_abs_attention_delta"
                            ),
                            "before_mean_row_entropy": metadata.get(
                                "before_mean_row_entropy"
                            ),
                            "after_mean_row_entropy": metadata.get(
                                "after_mean_row_entropy"
                            ),
                            "max_row_sum_error": metadata.get("max_row_sum_error"),
                        }
                    )
            heatmaps.sort(
                key=lambda item: (
                    item["block"] is None,
                    item["block"] if item["block"] is not None else -1,
                    item["head"] if item["head"] is not None else -1,
                )
            )
            records.append(
                {
                    "id": asset_id,
                    "model": "Wan+LoRA",
                    "alpha": alpha,
                    "group": group,
                    "ready": bool(path and path.is_file()),
                    "heatmaps": heatmaps,
                }
            )
    return {"case": case_key, "controls": controls, "records": records}


def qk_mono_scale_lora_videos_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoRA Mono-scale Top/Bottom100</title><style>
:root{--ink:#17251f;--paper:#e9e2d3;--card:#fffdf7;--line:#bdb19b;--red:#b94731;--green:#17695d;--dark:#16352b}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 6% 0,#e6a66455,transparent 34rem),radial-gradient(circle at 96% 3%,#51957c55,transparent 36rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:5;padding:17px 23px;background:#e9e2d3ed;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}h1{margin:0;font-size:clamp(27px,4vw,48px)}header p{margin:6px 0}.status{font:13px ui-monospace,monospace;color:#5c6962}button{padding:9px 14px;border:1px solid var(--line);background:#fff;font-weight:900;cursor:pointer}main{max-width:1900px;margin:auto;padding:20px}.matrix{display:grid;grid-template-columns:190px repeat(5,minmax(260px,1fr));gap:10px;min-width:1550px}.wrap{overflow:auto}.head,.label,.cell{background:var(--card);border:1px solid var(--line);padding:11px}.head{font-weight:900;text-align:center;background:#ddd3c0}.label{display:flex;flex-direction:column;justify-content:center;font:900 20px "Trebuchet MS",sans-serif}.label.top{border-left:8px solid var(--red)}.label.bottom{border-left:8px solid var(--green)}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#141816}.pending{display:grid;place-items:center;aspect-ratio:16/9;background:repeating-linear-gradient(135deg,#e5dece,#e5dece 10px,#f5f0e6 10px,#f5f0e6 20px);color:#746b5d}.meta{margin:8px 0 0;font:12px ui-monospace,monospace}.replay{position:fixed;right:20px;bottom:20px;z-index:10;border:0;border-radius:99px;padding:14px 20px;background:var(--dark);color:white;box-shadow:0 12px 30px #17352b55}.heat-section{margin:30px 0;padding:16px;border:1px solid var(--line);background:#f6f1e7cc}.heat-section h2{margin:0 0 12px}.heat-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.heat-card{padding:11px;background:var(--card);border:1px solid var(--line)}.heat-card h3{margin:0 0 8px;font:900 16px "Trebuchet MS",sans-serif}.heat-images{display:grid;grid-template-columns:1.35fr 1fr;gap:8px}.heat-images img{display:block;width:100%;background:#151816;border:1px solid var(--line)}.heat-meta{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.pill{padding:4px 7px;border-radius:99px;background:#e7dfcf;font:11px ui-monospace,monospace}@media(max-width:1000px){.heat-grid,.heat-images{grid-template-columns:1fr}}@media(max-width:800px){header{position:static}main{padding:9px}}
</style></head><body><button id="replay" class="replay">重新播放全部</button><header><h1>Wan+LoRA · Top/Bottom100 Mono-scale</h1><p>A′ = Normalize(A<sup>1+α</sup>)，统一配置：40 denoising steps、49 frames、全时间步应用。</p><button id="refresh">手动刷新结果</button> <span id="status" class="status">加载中</span></header><main><div class="wrap"><section id="matrix" class="matrix"></section></div><section id="heatmaps"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const caseKey=new URL(location.href).searchParams.get('case')||'0613pybullet_sample_001460_w002',src=id=>`/api/qk-mono-scale-lora-videos/video?id=${encodeURIComponent(id)}&case=${encodeURIComponent(caseKey)}`,image=name=>`/api/qk-mono-scale-lora-videos/image?name=${encodeURIComponent(name)}`,video=x=>x&&x.ready?`<video controls preload="metadata" playsinline src="${src(x.id)}"></video>`:`<div class="pending">视频生成中</div>`,fmt=x=>x==null?'-':Number(x).toExponential(3);
function renderHeatmaps(records){document.getElementById('heatmaps').innerHTML='<h1>调整前后 Attention 热力图</h1><p>全 token 图包含 Before / After / |ΔA|；帧级图包含 13×13 Before / After / Delta。</p>'+records.map(r=>`<section class="heat-section"><h2>${e(r.group.toUpperCase())} · α=${r.alpha.toFixed(1)} · ${r.heatmaps.length} heatmaps</h2><div class="heat-grid">${r.heatmaps.length?r.heatmaps.map(h=>`<article class="heat-card"><h3>${e(h.label)}</h3><div class="heat-images"><img loading="lazy" src="${image(h.all_token_asset)}" alt="all-token before after delta"><img loading="lazy" src="${image(h.frame_asset)}" alt="frame before after delta"></div><div class="heat-meta"><span class="pill">mean |ΔA| ${fmt(h.mean_abs_attention_delta)}</span><span class="pill">entropy ${h.before_mean_row_entropy==null?'-':Number(h.before_mean_row_entropy).toFixed(4)} → ${h.after_mean_row_entropy==null?'-':Number(h.after_mean_row_entropy).toFixed(4)}</span><span class="pill">row error ${fmt(h.max_row_sum_error)}</span></div></article>`).join(''):'<div class="pending">热力图生成中</div>'}</div></section>`).join('')}
async function load(){const d=await fetch(`/api/qk-mono-scale-lora-videos/catalog?case=${encodeURIComponent(caseKey)}`,{cache:'no-store'}).then(r=>r.json()),controls=Object.fromEntries(d.controls.map(x=>[x.id,x])),by=Object.fromEntries(d.records.map(x=>[x.group+'_'+x.alpha.toFixed(1),x]));document.getElementById('status').textContent=`${d.records.filter(x=>x.ready).length}/${d.records.length} intervention videos ready · ${d.records.reduce((n,x)=>n+x.heatmaps.length,0)} heatmaps · Case ${d.case}`;document.getElementById('matrix').innerHTML=`<div class="head">组</div><div class="head">Wan2.2 Original</div><div class="head">Wan+LoRA Original</div><div class="head">α = 0.3</div><div class="head">α = 0.6</div><div class="head">α = 0.9</div>`+['top100','bottom100'].map(g=>`<div class="label ${g.startsWith('top')?'top':'bottom'}">${g.toUpperCase()}<small>全时间步</small></div><div class="cell">${video(controls.wan_original)}<div class="meta">参考模型</div></div><div class="cell">${video(controls.lora_original)}<div class="meta">同模型未干预</div></div><div class="cell">${video(by[g+'_0.3'])}<div class="meta">Wan+LoRA · α=0.3</div></div><div class="cell">${video(by[g+'_0.6'])}<div class="meta">Wan+LoRA · α=0.6</div></div><div class="cell">${video(by[g+'_0.9'])}<div class="meta">Wan+LoRA · α=0.9</div></div>`).join('');renderHeatmaps(d.records)}
document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load().catch(err=>document.getElementById('status').textContent=err);
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


def _attention_lora_steps00_09_runs():
    specs = (
        ("alpha090_count100", "alpha", 0.9, None),
        ("alpha150_count100", "alpha", 1.5, None),
        ("zero_count100", "replacement", None, "zero"),
        ("uniform_count100", "replacement", None, "uniform"),
        ("temporal_causal_count100", "replacement", None, "temporal_causal"),
    )
    for run_key, experiment, alpha, intervention in specs:
        yield (
            run_key,
            experiment,
            alpha,
            intervention,
            ATTENTION_LORA_STEPS00_09_ROOT / run_key,
        )


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
    if asset_id.startswith("mono::"):
        try:
            _prefix, kind, name = asset_id.split("::", 2)
        except ValueError:
            return None
        if kind == "video":
            return qk_mono_scale_lora_video_asset(name, case_key)
        if kind == "image":
            return qk_mono_scale_head_asset(name)
        return None
    if asset_id.startswith("case_heatmap::"):
        try:
            _prefix, model_slug, run_dir, name = asset_id.split("::", 3)
        except ValueError:
            return None
        if (
            model_slug not in {"baseline", "lora"}
            or Path(run_dir).name != run_dir
            or Path(name).name != name
        ):
            return None
        return QK_ATTENTION_MONO_SCALE_HEAD_ROOT / model_slug / run_dir / name
    if asset_id.startswith("steps00_09::"):
        try:
            _prefix, run_key, kind, name = asset_id.split("::", 3)
        except ValueError:
            return None
        run = next(
            (item for item in _attention_lora_steps00_09_runs() if item[0] == run_key),
            None,
        )
        if run is None or Path(name).name != name:
            return None
        run_root = run[4]
        if kind == "image":
            return run_root / "heatmaps" / name
        if kind == "video":
            return run_root / "videos" / "lora" / "cases" / case_key / name
        return None
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
    baseline_clean_heatmaps = []
    for count in (30, 100):
        run_dir = f"alpha000_count{count}"
        run_root = QK_ATTENTION_MONO_SCALE_HEAD_ROOT / "baseline" / run_dir
        for group in (f"top{count}", f"bottom{count}"):
            metadata_path = next(
                iter(
                    sorted(
                        run_root.glob(
                            f"*__{case_key}__{group}__probability_mono_scale__step39.json"
                        )
                    )
                ),
                None,
            )
            metadata = {}
            if metadata_path is not None:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
            all_token = str(metadata.get("all_token_image", ""))
            frame = str(metadata.get("frame_image", ""))
            all_token_id = (
                f"case_heatmap::baseline::{run_dir}::{all_token}" if all_token else ""
            )
            frame_id = f"case_heatmap::baseline::{run_dir}::{frame}" if frame else ""
            all_token_path = (
                attention_lora_case_asset(all_token_id, case_key) if all_token_id else None
            )
            frame_path = attention_lora_case_asset(frame_id, case_key) if frame_id else None
            baseline_clean_heatmaps.append(
                {
                    "model": "baseline",
                    "group": group,
                    "all_token_id": all_token_id,
                    "frame_id": frame_id,
                    "ready": bool(
                        all_token_path
                        and frame_path
                        and all_token_path.is_file()
                        and frame_path.is_file()
                    ),
                    "metrics": metadata,
                }
            )
    for count in (30, 100):
        run_dir = f"alpha000_count{count}"
        run_root = QK_ATTENTION_MONO_SCALE_HEAD_ROOT / "lora" / run_dir
        for group in (f"top{count}", f"bottom{count}"):
            metadata_path = next(
                iter(
                    sorted(
                        run_root.glob(
                            f"*__{case_key}__{group}__probability_mono_scale__step39.json"
                        )
                    )
                ),
                None,
            )
            metadata = {}
            if metadata_path is not None:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}
            all_token = str(metadata.get("all_token_image", ""))
            frame = str(metadata.get("frame_image", ""))
            all_token_id = (
                f"case_heatmap::lora::{run_dir}::{all_token}" if all_token else ""
            )
            frame_id = f"case_heatmap::lora::{run_dir}::{frame}" if frame else ""
            all_token_path = (
                attention_lora_case_asset(all_token_id, case_key) if all_token_id else None
            )
            frame_path = attention_lora_case_asset(frame_id, case_key) if frame_id else None
            baseline_clean_heatmaps.append(
                {
                    "model": "lora",
                    "group": group,
                    "all_token_id": all_token_id,
                    "frame_id": frame_id,
                    "ready": bool(
                        all_token_path
                        and frame_path
                        and all_token_path.is_file()
                        and frame_path.is_file()
                    ),
                    "metrics": metadata,
                }
            )
    records = []
    for run_key, alpha, count, run_root in _attention_baseline_runs():
        for group in (f"top{count}", f"bottom{count}"):
            case_run_root = (
                QK_ATTENTION_MONO_SCALE_HEAD_ROOT / "baseline" / run_root.name
            )
            metadata_path = next(
                iter(sorted(case_run_root.glob(f"*__{case_key}__{group}__*__step39.json"))),
                None,
            )
            case_heatmap = metadata_path is not None
            if metadata_path is None:
                metadata_path = next(
                    iter(sorted(run_root.glob(f"*__{case_key}__{group}__*__step39.json"))),
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
            all_token_id = (
                f"case_heatmap::baseline::{run_root.name}::{all_token}"
                if case_heatmap and all_token
                else f"{run_key}::image::{all_token}" if all_token
                else ""
            )
            frame_id = (
                f"case_heatmap::baseline::{run_root.name}::{frame}"
                if case_heatmap and frame
                else f"{run_key}::image::{frame}" if frame
                else ""
            )
            all_token_path = attention_lora_case_asset(all_token_id, case_key) if all_token_id else None
            frame_path = attention_lora_case_asset(frame_id, case_key) if frame_id else None
            records.append(
                {
                    "run_key": run_key,
                    "model": "Wan2.2 Baseline",
                    "alpha": alpha,
                    "count": count,
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
    for run_key, alpha, count, run_root in _attention_lora_runs():
        for group in (f"top{count}", f"bottom{count}"):
            case_run_root = QK_ATTENTION_MONO_SCALE_HEAD_ROOT / "lora" / run_root.name
            metadata_path = next(
                iter(sorted(case_run_root.glob(f"*__{case_key}__{group}__*__step39.json"))),
                None,
            )
            case_heatmap = metadata_path is not None
            if metadata_path is None:
                metadata_path = next(
                    iter(sorted(run_root.glob(f"*__{case_key}__{group}__*__step39.json"))),
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
            all_token_id = (
                f"case_heatmap::lora::{run_root.name}::{all_token}"
                if case_heatmap and all_token
                else f"{run_key}::image::{all_token}" if all_token
                else ""
            )
            frame_id = (
                f"case_heatmap::lora::{run_root.name}::{frame}"
                if case_heatmap and frame
                else f"{run_key}::image::{frame}" if frame
                else ""
            )
            records.append(
                {
                    "run_key": run_key,
                    "model": "Wan+LoRA",
                    "alpha": alpha,
                    "count": count,
                    "group": group,
                    "video_id": video_id,
                    "video_ready": bool(video_path and video_path.is_file()),
                    "all_token_id": all_token_id,
                    "frame_id": frame_id,
                    "heatmap_ready": bool(
                        all_token
                        and frame
                        and attention_lora_case_asset(
                            all_token_id, case_key
                        ).is_file()
                        and attention_lora_case_asset(
                            frame_id, case_key
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
    mono_catalog = qk_mono_scale_lora_video_catalog(case_key)
    for mono_record in mono_catalog["records"]:
        mono_heatmaps = [
            {
                **heatmap,
                "all_token_id": f"mono::image::{heatmap['all_token_asset']}",
                "frame_id": f"mono::image::{heatmap['frame_asset']}",
            }
            for heatmap in mono_record.get("heatmaps", [])
        ]
        records.append(
            {
                "run_key": mono_record["id"],
                "model": "Wan+LoRA",
                "experiment": "mono_scale",
                "intervention": "probability_mono_scale",
                "alpha": mono_record["alpha"],
                "count": 100,
                "group": mono_record["group"],
                "video_id": f"mono::video::{mono_record['id']}",
                "video_ready": mono_record["ready"],
                "heatmaps": mono_heatmaps,
                "heatmap_ready": any(item["ready"] for item in mono_heatmaps),
                "heatmap_expected": True,
                "metrics": {},
            }
        )
    steps00_09_records = []
    if case_key == ATTENTION_LORA_CASE:
        for run_key, experiment, alpha, intervention, run_root in _attention_lora_steps00_09_runs():
            for group in ("top100", "bottom100"):
                metadata_pattern = (
                    f"*__{case_key}__{group}__probability_mono_scale__step09.json"
                    if experiment == "alpha"
                    else f"*__{case_key}__{group}__step09.json"
                )
                metadata_path = next(
                    iter(sorted((run_root / "heatmaps").glob(metadata_pattern))),
                    None,
                )
                metadata = {}
                if metadata_path is not None:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                video_name = f"{group}_steps_00_10.mp4"
                video_id = f"steps00_09::{run_key}::video::{video_name}"
                all_token = str(metadata.get("all_token_image", ""))
                frame = str(metadata.get("frame_image", ""))
                all_token_id = (
                    f"steps00_09::{run_key}::image::{all_token}" if all_token else ""
                )
                frame_id = f"steps00_09::{run_key}::image::{frame}" if frame else ""
                video_path = attention_lora_case_asset(video_id, case_key)
                all_token_path = (
                    attention_lora_case_asset(all_token_id, case_key)
                    if all_token_id
                    else None
                )
                frame_path = (
                    attention_lora_case_asset(frame_id, case_key) if frame_id else None
                )
                record = {
                    "run_key": run_key,
                    "model": "Wan+LoRA",
                    "count": 100,
                    "group": group,
                    "stage": "steps00_09",
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
                if experiment == "alpha":
                    record["alpha"] = alpha
                else:
                    record["experiment"] = "replacement"
                    record["intervention"] = intervention
                steps00_09_records.append(record)
    return {
        "case": case_key,
        "cases": cases,
        "controls": controls,
        "baseline_clean_heatmaps": baseline_clean_heatmaps,
        "records": records,
        "steps00_09_records": steps00_09_records,
        "ready_records": sum(
            r["video_ready"] and (r["heatmap_ready"] or not r["heatmap_expected"])
            for r in records
        ),
        "expected_records": len(records),
    }


def attention_lora_case_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan+LoRA Attention Probability Noise</title><style>
:root{--ink:#1e2924;--paper:#e9e3d6;--card:#fffdf8;--line:#c4b8a2;--red:#b94731;--green:#17695d;--gold:#b17b25;--blue:#38627a;--dark:#17342b}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 7% 0,#e7a65f45,transparent 33rem),radial-gradient(circle at 96% 2%,#4f92784a,transparent 38rem),linear-gradient(145deg,#eee8dc,#dce7df);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:18px 26px;background:#ebe5d9ed;border-bottom:1px solid var(--line);backdrop-filter:blur(12px);box-shadow:0 8px 28px #3d463f18}h1{margin:0;font-size:clamp(25px,3vw,42px);letter-spacing:-.025em}header p{margin:6px 0;color:#59675f}.status{font-family:ui-monospace,monospace;font-size:13px}main{width:min(2200px,calc(100% - 24px));margin:auto;padding:22px 0 80px}.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.control,.card{background:rgba(255,253,248,.96);border:1px solid var(--line);border-radius:14px;padding:13px;box-shadow:0 9px 25px #34403912}.control h2{font-size:18px}.control video{aspect-ratio:16/9;object-fit:contain}.card.top{border-left:6px solid var(--red)}.card.bottom{border-left:6px solid var(--green)}video,img{display:block;width:100%;background:#151916;border:1px solid #a99e8b;border-radius:7px}video{aspect-ratio:16/9;object-fit:contain}.heatmaps{display:grid;grid-template-columns:1fr;gap:9px;margin-top:11px}.baseline-heatmaps{margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}.baseline-heatmaps h4{margin:0 0 8px}.baseline-heatmap-images{display:grid;grid-template-columns:1fr;gap:7px}.pending{min-height:170px;display:grid;place-items:center;border:1px dashed var(--line);border-radius:8px;color:#68736d;background:#f4efe5}.meta{display:flex;gap:6px;flex-wrap:wrap;margin:9px 0}.pill{padding:5px 8px;border-radius:999px;background:#e9e2d2;font:11px ui-monospace,monospace}.section-title{margin:30px 0 12px;font-size:26px}.matrix-shell{overflow:auto;border:1px solid var(--line);border-radius:16px;background:#d8d0c1;box-shadow:0 16px 45px #33403820;padding:10px;scrollbar-color:#6f7d75 #d8d0c1}.matrix{display:flex;flex-direction:column;gap:8px;width:max-content}.matrix-head,.row{display:grid;grid-template-columns:220px repeat(9,340px);gap:8px;width:max-content;min-width:3352px}.matrix-head{margin-bottom:8px}.row-head,.alpha-title,.cell{border:1px solid var(--line);border-radius:10px}.row-head{position:sticky;left:10px;z-index:5;padding:14px;background:var(--dark);color:#fff;box-shadow:8px 0 20px #20352c33}.matrix-head .row-head{z-index:8;background:#102a22}.row-head .title{margin:4px 0;font-size:18px;font-family:"Trebuchet MS","Noto Serif CJK SC",sans-serif;font-weight:900}.row-head .sub{color:#c8d6cf;font-size:12px}.alpha-title{text-align:center;padding:11px 8px;font-size:15px;font-weight:900;background:#f8f3e9}.alpha-title small{display:block;margin-bottom:4px;font:10px ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase;opacity:.68}.alpha-title.family-baseline{border-top:5px solid var(--dark)}.alpha-title.family-additive{border-top:5px solid var(--gold)}.alpha-title.family-replacement{border-top:5px solid var(--blue)}.alpha-title.family-mono{border-top:5px solid var(--green)}.cell{padding:8px;background:#f5f0e6;min-width:0}.cell.family-additive{background:#f4eddf}.cell.family-replacement{background:#e9f0f2}.cell.family-mono{background:#e5f0eb}.cell .card{height:100%;box-shadow:none}.not-applicable{min-height:170px;display:grid;place-items:center;color:#8a8274;background:#eee8dc;border:1px dashed var(--line);border-radius:8px}
.replay-all{position:fixed;right:22px;bottom:22px;z-index:20;border:0;border-radius:999px;padding:13px 20px;background:#172e27;color:#fff;font:700 14px ui-monospace,monospace;box-shadow:0 10px 28px rgba(20,35,29,.3);cursor:pointer}.replay-all:hover{background:#b94332}.replay-all:active{transform:translateY(1px)}.manual-refresh{margin-left:10px;border:1px solid #748078;border-radius:999px;padding:8px 14px;background:#fff;color:#172e27;font-weight:800;cursor:pointer}.manual-refresh:hover{border-color:#b94332;color:#b94332}
.mono-heads{display:grid;grid-template-columns:1fr;gap:9px;margin-top:10px}.mono-head{border:1px solid var(--line);border-radius:8px;padding:9px;background:#f8f4eb}.mono-head h4{margin:0 0 7px}.mono-head .heatmaps{grid-template-columns:1fr}details summary{cursor:pointer;font-weight:900;padding:10px;background:#e4ede8;border-radius:8px;margin-top:10px}details[open] summary{border-radius:8px 8px 0 0}.row.steps00-09{margin-top:-2px}.row.steps00-09 .row-head{background:#8b3d2d}.row.steps00-09 .cell{border-top:3px solid #c26a4d}.replay-all{position:fixed;right:22px;bottom:22px;z-index:30;border:0;border-radius:999px;padding:13px 20px;background:var(--dark);color:#fff;font:700 14px ui-monospace,monospace;box-shadow:0 10px 28px rgba(20,35,29,.3);cursor:pointer}.replay-all:hover{background:var(--red)}.manual-refresh{margin-left:10px;border:1px solid #748078;border-radius:999px;padding:8px 14px;background:#fff;color:#172e27;font-weight:800;cursor:pointer}.manual-refresh:hover{border-color:var(--red);color:var(--red)}@media(max-width:980px){header{position:static;padding:14px}main{width:calc(100% - 12px)}.controls{grid-template-columns:1fr}.matrix-head,.row{grid-template-columns:170px repeat(9,290px);min-width:2842px}.row-head{left:10px}.replay-all{right:10px;bottom:10px}}
</style></head><body><button id="replayAll" class="replay-all" type="button">重新播放全部</button><header><h1>Attention Probability Noise Ablation</h1><p><strong>消融模型：Wan+LoRA、Full-SA no-object step-002500</strong> · 参考模型：Wan2.2 Baseline</p><p><label for="caseSelect">Case：</label><select id="caseSelect"></select><button id="manualRefresh" class="manual-refresh" type="button">手动刷新结果</button></p><p id="case"></p><div id="status" class="status">加载中</div></header><main><h2 class="section-title">Original controls</h2><section id="controls" class="controls"></section><h2 class="section-title">Head Intervention Workbench</h2><p>横向滑动查看完整实验矩阵。Baseline、Additive、Replacement、Mono-scale 使用不同色带；热力图在对应实验卡片内展开。</p><div class="matrix-shell"><section id="matrix-head" class="matrix-head"></section><section id="grid" class="matrix"></section></div></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let currentCase=new URL(location.href).searchParams.get('case')||'';const v=id=>`/api/attention-additive-lora-case/video?id=${encodeURIComponent(id)}&case=${encodeURIComponent(currentCase)}`;const im=id=>`/api/attention-additive-lora-case/image?id=${encodeURIComponent(id)}&case=${encodeURIComponent(currentCase)}`;const f=x=>Number(x).toExponential(3);
function renderMonoCell(r){const maps=r.heatmaps||[];return `<article class="card ${r.group.startsWith('top')?'top':'bottom'}"><div class="alpha-title"><strong>Mono α=${Number(r.alpha).toFixed(1)}</strong></div><div class="meta"><span class="pill">Wan+LoRA</span><span class="pill">40步 · 49帧 · 全时间步</span></div>${r.video_ready?`<video controls preload="metadata" playsinline src="${v(r.video_id)}"></video>`:'<div class="pending">新视频生成中</div>'}<details ${maps.length<=2?'open':''}><summary>Before / After / Delta（${maps.length}）</summary><div class="mono-heads">${maps.length?maps.map(h=>`<article class="mono-head"><h4>${e(h.label)}</h4><div class="heatmaps"><img loading="lazy" src="${im(h.all_token_id)}"><img loading="lazy" src="${im(h.frame_id)}"></div><div class="meta"><span class="pill">mean |ΔA| ${h.mean_abs_attention_delta==null?'-':f(h.mean_abs_attention_delta)}</span><span class="pill">row error ${h.max_row_sum_error==null?'-':f(h.max_row_sum_error)}</span></div></article>`).join(''):'<div class="pending">热力图生成中</div>'}</div></details></article>`}
function renderCell(r,expectedGroup){if(r&&r.experiment==='mono_scale')return renderMonoCell(r);const m=r?r.metrics||{}:{ },intervention=r&&r.experiment==='replacement'?(r.intervention==='zero'?'A=0':r.intervention==='uniform'?'A=1/N_K':'Temporal Causal Mask'):r?'α='+r.alpha.toFixed(1):'';const label=r?(r.group||'').toUpperCase()+' · '+intervention:expectedGroup.toUpperCase()+' · 未完成';if(!r){return `<article class="card ${expectedGroup.startsWith('top')?'top':'bottom'}"><div class="alpha-title"><strong>${e(label)}</strong></div><div class="pending">该实验的新结果尚未生成</div></article>`;}const staged=r.stage==='steps00_09',capture=staged?'S009':'S039',phase=staged?'仅 S000-S009 干预 · S010-S039 clean':'全时间步干预';const heatmaps=r.heatmap_ready?`<div class="heatmaps"><img loading="lazy" src="${im(r.all_token_id)}"><img loading="lazy" src="${im(r.frame_id)}"></div>`:r.heatmap_expected?`<div class="heatmaps"><div class="pending">${capture} 全 token 热力图生成中</div><div class="pending">帧级热力图生成中</div></div>`:`<div class="pending">新统一推理结果尚未采集热力图</div>`;return `<article class="card ${r.group&&r.group.startsWith('top')?'top':'bottom'}"><div class="alpha-title"><strong>${e(label)}</strong></div><div class="meta"><span class="pill">消融模型：${e(r.model)}</span><span class="pill">统一配置：40步 · 49帧 · 热力图${capture}</span><span class="pill">${phase}</span></div>${r.video_ready?`<video controls preload="metadata" playsinline src="${v(r.video_id)}"></video>`:`<div class="pending">${e(r.model)} 新视频生成中</div>`}${heatmaps}${r.heatmap_ready?`<div class="meta"><span class="pill">mean |ΔA| ${f(m.mean_abs_attention_delta)}</span><span class="pill">row error ${f(m.max_row_sum_error)}</span></div>`:''}</article>`}
function controlIdForModel(model){if(model.includes('Wan+LoRA'))return'control_lora';if(model.includes('Full-SA'))return'control_full_sa';return'control_baseline'}
function renderBaselineCell(row,controlsById,cleanByGroup){const control=controlsById[controlIdForModel(row.model)],modelSlug=row.model.includes('Wan2.2 Baseline')?'baseline':row.model.includes('Wan+LoRA')?'lora':'',clean=modelSlug?cleanByGroup[`${modelSlug}:${row.direction}${row.count}`]:null;if(!control||!control.ready){return `<article class="card"><div class="alpha-title"><strong>Baseline Original</strong></div><div class="pending">基线视频尚未就绪</div></article>`;}const heatmap=clean&&clean.ready?`<section class="baseline-heatmaps"><h4>Clean Baseline Attention · ${e(clean.group.toUpperCase())} · S039</h4><div class="baseline-heatmap-images"><img loading="lazy" src="${im(clean.all_token_id)}" alt="clean baseline all-token attention"><img loading="lazy" src="${im(clean.frame_id)}" alt="clean baseline frame attention"></div><div class="meta"><span class="pill">模型：${e(row.model)}</span><span class="pill">无干预轨迹</span><span class="pill">entropy ${Number(clean.metrics.before_mean_row_entropy).toFixed(4)}</span></div></section>`:clean?`<div class="pending">${e(row.model)} Clean Baseline 热力图生成中</div>`:'';return `<article class="card"><div class="alpha-title"><strong>Baseline Original</strong></div><div class="meta"><span class="pill">基线：${e(control.label)}</span></div><video controls preload="metadata" playsinline src="${v(control.id)}"></video>${heatmap}</article>`}
function makeRowMeta(r){return {name:`${r.model.startsWith('Wan+LoRA')?'Wan+LoRA':r.model.includes('Full-SA')?'Full-SA':'消融模型'} · ${r.direction==='top'?'Top':'Bottom'}${r.count}`,sub:`统一配置：40步 · 49帧`}}
function renderMono(records){document.getElementById('monoGrid').innerHTML=records.map(r=>{const maps=r.heatmaps||[],direction=r.group.startsWith('top')?'top':'bottom';return `<article class="mono-run ${direction}"><h3>${e(r.group.toUpperCase())} · α=${Number(r.alpha).toFixed(1)}</h3><div class="meta"><span class="pill">模型：Wan+LoRA</span><span class="pill">40步 · 49帧 · 全时间步</span><span class="pill">${maps.length} heatmaps</span></div>${r.video_ready?`<video controls preload="metadata" playsinline src="${v(r.video_id)}"></video>`:'<div class="pending">新视频生成中</div>'}<details ${maps.length<=2?'open':''}><summary>调整前后热力图（${maps.length}）</summary><div class="mono-heads">${maps.length?maps.map(h=>`<article class="mono-head"><h4>${e(h.label)}</h4><div class="heatmaps"><img loading="lazy" src="${im(h.all_token_id)}" alt="all-token before after delta"><img loading="lazy" src="${im(h.frame_id)}" alt="frame before after delta"></div><div class="meta"><span class="pill">mean |ΔA| ${h.mean_abs_attention_delta==null?'-':f(h.mean_abs_attention_delta)}</span><span class="pill">row error ${h.max_row_sum_error==null?'-':f(h.max_row_sum_error)}</span></div></article>`).join(''):'<div class="pending">热力图生成中</div>'}</div></details></article>`}).join('')}
function columnKey(r){return r.experiment==='replacement'?r.intervention:r.experiment==='mono_scale'?`mono_${Number(r.alpha).toFixed(1)}`:`additive_${Number(r.alpha).toFixed(1)}`}
function collectRows(records){const rows=new Map();for(const r of records){const count=String(r.count),direction=r.group.toLowerCase().startsWith('top')?'top':'bottom',key=`${r.model}::${direction}::${count}`;if(!rows.has(key))rows.set(key,{model:r.model,count,direction,items:{}});rows.get(key).items[columnKey(r)]=r}return rows}
function renderRows(records,columns,headId,gridId,controlsById,cleanByGroup,stageRecords=[]){document.getElementById(headId).innerHTML=`<div class="row-head"><div class="title">模型 × Head 组</div><div class="sub">横向滑动浏览</div></div><div class="alpha-title family-baseline"><small>Control</small>Baseline</div>`+columns.map(c=>`<div class="alpha-title family-${c.family}"><small>${e(c.familyLabel)}</small>${e(c.label)}</div>`).join('');const rows=collectRows(records),stageRows=collectRows(stageRecords),ordered=Array.from(rows.values()).sort((a,b)=>a.model.localeCompare(b.model)||Number(a.count)-Number(b.count)||b.direction.localeCompare(a.direction));const rowHtml=(row,staged=false)=>`<article class="row${staged?' steps00-09':''}"><div class="row-head"><div class="title">${e(makeRowMeta(row).name)}${staged?' · steps00_09':''}</div><div class="sub">${staged?'仅 S000-S009 干预 · S010-S039 clean · 热力图 S009':e(makeRowMeta(row).sub)}</div></div><div class="cell family-baseline">${renderBaselineCell(row,controlsById,cleanByGroup)}</div>`+columns.map(c=>{const unavailable=staged?c.family==='mono':c.key.startsWith('mono_')&&!(row.model.startsWith('Wan+LoRA')&&Number(row.count)===100);return `<div class="cell family-${c.family}">${unavailable?'<div class="not-applicable">本阶段未运行</div>':renderCell(row.items[c.key],`${row.direction}${row.count}`)}</div>`}).join('')+`</article>`;document.getElementById(gridId).innerHTML=ordered.map(row=>{const key=`${row.model}::${row.direction}::${row.count}`,stageRow=stageRows.get(key);return rowHtml(row)+(stageRow?rowHtml(stageRow,true):'')}).join('')}
async function load(){const d=await fetch(`/api/attention-additive-lora-case/catalog?case=${encodeURIComponent(currentCase)}`,{cache:'no-store'}).then(r=>r.json());currentCase=d.case;const select=document.getElementById('caseSelect');if(select.options.length!==d.cases.length){select.innerHTML=d.cases.map(x=>`<option value="${e(x)}">${e(x)}</option>`).join('')}select.value=currentCase;document.getElementById('case').textContent=`Case: ${d.case} · additive noise、normalized replacement、temporal causal mask 与 probability mono-scale`;const controlsById=Object.fromEntries(d.controls.map(x=>[x.id,x])),cleanHeatmaps=d.baseline_clean_heatmaps||[],cleanByGroup=Object.fromEntries(cleanHeatmaps.map(x=>[`${x.model}:${x.group}`,x])),alphaKeys=new Set(['0.9','1.5']),additive=d.records.filter(r=>!r.experiment&&alphaKeys.has(Number(r.alpha).toFixed(1))),replacement=d.records.filter(r=>r.experiment==='replacement'),mono=d.records.filter(r=>r.experiment==='mono_scale'),visible=[...additive,...replacement,...mono],stageRecords=d.steps00_09_records||[],columns=[{key:'additive_0.9',label:'α = 0.9',family:'additive',familyLabel:'Additive Noise'},{key:'additive_1.5',label:'α = 1.5',family:'additive',familyLabel:'Additive Noise'},{key:'zero',label:'A = 0',family:'replacement',familyLabel:'Replacement'},{key:'uniform',label:'A = 1/N_K',family:'replacement',familyLabel:'Replacement'},{key:'temporal_causal',label:'Temporal Causal',family:'replacement',familyLabel:'Mask'},{key:'mono_0.3',label:'α = 0.3',family:'mono',familyLabel:'Mono-scale'},{key:'mono_0.6',label:'α = 0.6',family:'mono',familyLabel:'Mono-scale'},{key:'mono_0.9',label:'α = 0.9',family:'mono',familyLabel:'Mono-scale'}];document.getElementById('status').textContent=`${visible.filter(r=>r.video_ready).length}/${visible.length} visible videos ready · ${stageRecords.filter(r=>r.video_ready&&r.heatmap_ready).length}/${stageRecords.length} steps00_09 ready · ${cleanHeatmaps.filter(x=>x.ready).length}/${cleanHeatmaps.length} clean Baseline heatmaps · ${mono.reduce((n,r)=>n+(r.heatmaps||[]).length,0)} mono-scale heatmaps · 点击按钮手动刷新`;document.getElementById('controls').innerHTML=d.controls.map(x=>`<article class="control"><h2>${e(x.label)}</h2>${x.ready?`<video controls preload="metadata" playsinline src="${v(x.id)}"></video>`:'<div class="pending">等待 Original 视频</div>'}</article>`).join('');renderRows(visible,columns,'matrix-head','grid',controlsById,cleanByGroup,stageRecords)}
document.getElementById('caseSelect').addEventListener('change',event=>{currentCase=event.target.value;const url=new URL(location.href);url.searchParams.set('case',currentCase);history.replaceState(null,'',url);load()});document.getElementById('manualRefresh').addEventListener('click',()=>load());document.getElementById('replayAll').addEventListener('click',()=>{document.querySelectorAll('video').forEach(video=>{video.pause();video.currentTime=0;video.loop=false;video.play().catch(()=>{})})});load();
</script></body></html>'''


SEED_SWEEP_PROFILES = (
    ("alpha090", "α = 0.9"),
    ("alpha150", "α = 1.5"),
    ("zero", "A = 0"),
    ("uniform", "A = 1/N_K"),
    ("temporal_causal", "Temporal Causal"),
    ("strict_past", "Strict Past Only"),
    ("strict_future", "Strict Future Only"),
    ("head_output_zero", "Head Output Zero"),
)


def _attention_lora_seed_sweep_seeds():
    path = ATTENTION_LORA_SEED_SWEEP_ROOT / "seeds.txt"
    if not path.is_file():
        return []
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            seed = int(line.strip())
        except ValueError:
            continue
        if 0 <= seed <= 100000 and seed not in seeds:
            seeds.append(seed)
    return seeds


def attention_lora_seed_sweep_asset(
    seed_text: str, stage: str, profile: str, group: str, name: str
):
    try:
        seed = int(seed_text)
    except ValueError:
        return None
    if seed not in _attention_lora_seed_sweep_seeds():
        return None
    seed_root = ATTENTION_LORA_SEED_SWEEP_ROOT / "seeds" / f"seed_{seed:06d}"
    if profile == "original" and stage == "original" and group == "original":
        return seed_root / "original.mp4"
    allowed_profiles = {item[0] for item in SEED_SWEEP_PROFILES}
    if (
        stage not in {"all_steps", "steps00_09"}
        or profile not in allowed_profiles
        or group not in {"top100", "bottom100"}
    ):
        return None
    run_root = seed_root / stage / profile
    if name:
        if Path(name).name != name or not name.endswith(".png"):
            return None
        return run_root / "heatmaps" / name
    suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
    return (
        run_root
        / "videos"
        / "lora"
        / "cases"
        / ATTENTION_LORA_CASE
        / f"{group}_{suffix}.mp4"
    )


def attention_lora_seed_sweep_catalog(requested_seed: str = ""):
    seeds = _attention_lora_seed_sweep_seeds()
    try:
        selected = int(requested_seed)
    except ValueError:
        selected = seeds[0] if seeds else 0
    if selected not in seeds and seeds:
        selected = seeds[0]
    original = attention_lora_seed_sweep_asset(
        str(selected), "original", "original", "original", ""
    )
    records = []
    for stage in ("all_steps", "steps00_09"):
        capture_step = 39 if stage == "all_steps" else 9
        for profile, label in SEED_SWEEP_PROFILES:
            run_root = (
                ATTENTION_LORA_SEED_SWEEP_ROOT
                / "seeds"
                / f"seed_{selected:06d}"
                / stage
                / profile
            )
            for group in ("top100", "bottom100"):
                video = attention_lora_seed_sweep_asset(
                    str(selected), stage, profile, group, ""
                )
                metadata_path = next(
                    iter(
                        sorted(
                            (run_root / "heatmaps").glob(
                                f"*__{ATTENTION_LORA_CASE}__{group}__*step{capture_step:02d}.json"
                            )
                        )
                    ),
                    None,
                )
                metadata = {}
                if metadata_path is not None:
                    try:
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        metadata = {}
                all_token = str(metadata.get("all_token_image", ""))
                frame = str(metadata.get("frame_image", ""))
                all_token_path = attention_lora_seed_sweep_asset(
                    str(selected), stage, profile, group, all_token
                ) if all_token else None
                frame_path = attention_lora_seed_sweep_asset(
                    str(selected), stage, profile, group, frame
                ) if frame else None
                records.append(
                    {
                        "stage": stage,
                        "profile": profile,
                        "label": label,
                        "group": group,
                        "video_ready": bool(video and video.is_file()),
                        "all_token": all_token,
                        "frame": frame,
                        "heatmap_ready": bool(
                            all_token_path
                            and frame_path
                            and all_token_path.is_file()
                            and frame_path.is_file()
                        ),
                        "heatmap_expected": profile != "head_output_zero",
                        "metrics": metadata,
                    }
                )
    completed_seeds = 0
    for seed in seeds:
        seed_root = ATTENTION_LORA_SEED_SWEEP_ROOT / "seeds" / f"seed_{seed:06d}"
        if sum(1 for _ in seed_root.glob("*/*/complete")) == 16:
            completed_seeds += 1
    return {
        "case": ATTENTION_LORA_CASE,
        "seeds": seeds,
        "selected_seed": selected,
        "completed_seeds": completed_seeds,
        "total_seeds": len(seeds),
        "original_ready": bool(original and original.is_file()),
        "ready_records": sum(record["video_ready"] for record in records),
        "expected_records": len(records),
        "records": records,
    }


def attention_lora_seed_sweep_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wan+LoRA 50-Seed Sweep</title><style>
:root{--ink:#1a2822;--paper:#eee8dc;--card:#fffdf8;--line:#bdb3a0;--red:#ae432f;--green:#17695d;--dark:#19362d;--gold:#bb7b28}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 8% 0,#eaa76755,transparent 34rem),radial-gradient(circle at 96% 3%,#52977c55,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:17px 24px;background:#eee8dced;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 13px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#53635b}main{width:min(2300px,calc(100% - 20px));margin:auto;padding:20px 0 80px}.original{max-width:700px;background:var(--card);border:1px solid var(--line);border-radius:15px;padding:13px}.original video,.card video,.card img{display:block;width:100%;background:#151916;border:1px solid var(--line);border-radius:7px}.matrix-shell{overflow:auto;margin-top:20px;border:1px solid var(--line);border-radius:16px;padding:9px;background:#d8d0c1}.matrix{display:grid;grid-template-columns:230px repeat(8,330px);gap:8px;width:max-content}.head,.row-head,.cell{border:1px solid var(--line);border-radius:10px}.head{padding:12px;text-align:center;background:#f9f4e9;font-weight:900;border-top:5px solid var(--gold)}.row-head{position:sticky;left:9px;z-index:4;padding:15px;background:var(--dark);color:#fff}.row-head.top{border-left:7px solid var(--red)}.row-head.bottom{border-left:7px solid var(--green)}.row-head small{display:block;color:#cbd8d1;margin-top:8px}.cell{padding:8px;background:#f6f1e7}.card{height:100%;padding:10px;background:var(--card);border-radius:10px}.card.top{border-left:5px solid var(--red)}.card.bottom{border-left:5px solid var(--green)}.card h3{margin:0 0 7px;font-size:16px}.pill{display:inline-block;margin:3px;padding:4px 7px;border-radius:99px;background:#e9e2d4;font:10px ui-monospace,monospace}.pending{min-height:170px;display:grid;place-items:center;border:1px dashed var(--line);border-radius:7px;color:#746e62}.maps{display:grid;gap:7px;margin-top:8px}.note{font-size:11px;color:#665f55}.replay{position:fixed;right:20px;bottom:20px;z-index:30;border:0;border-radius:99px;background:var(--dark);color:#fff;padding:13px 19px}@media(max-width:800px){header{position:static}.matrix{grid-template-columns:170px repeat(6,280px)}main{width:calc(100% - 10px)}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a><h1>Wan+LoRA · 50-Seed Sweep</h1><p>0613pybullet_sample_001460_w002 · 40步 · 49帧 · 仅 seed 改变</p><div class="tools"><label>Seed <select id="seed"></select></label><button id="refresh">手动刷新</button><span class="status" id="status">读取中</span></div></header><main><h2>Original</h2><section id="original" class="original"></section><div class="matrix-shell"><section id="matrix" class="matrix"></section></div></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let current=q.get('seed')||'';const profiles=[['alpha090','α = 0.9'],['alpha150','α = 1.5'],['zero','A = 0'],['uniform','A = 1/N_K'],['temporal_causal','Temporal Causal'],['strict_past','Strict Past Only'],['strict_future','Strict Future Only'],['head_output_zero','Head Output Zero']];
const video=(stage,profile,group)=>`/api/attention-additive-lora-seed-sweep/video?seed=${encodeURIComponent(current)}&stage=${stage}&profile=${profile}&group=${group}`;const image=(r,name)=>`/api/attention-additive-lora-seed-sweep/image?seed=${encodeURIComponent(current)}&stage=${r.stage}&profile=${r.profile}&group=${r.group}&name=${encodeURIComponent(name)}`;
function card(r){const maps=r.heatmap_ready?`<div class="maps"><img loading="lazy" src="${image(r,r.all_token)}"><img loading="lazy" src="${image(r,r.frame)}"></div>`:r.heatmap_expected?'<div class="note">热力图等待生成</div>':'<div class="note">Attention 不变，仅 Head 输出置零</div>';return `<article class="card ${r.group.startsWith('top')?'top':'bottom'}"><h3>${e(r.label)}</h3><span class="pill">${e(r.group.toUpperCase())}</span><span class="pill">${r.stage==='all_steps'?'S000-S039':'S000-S009'}</span>${r.video_ready?`<video controls preload="metadata" playsinline src="${video(r.stage,r.profile,r.group)}"></video>`:'<div class="pending">视频生成中</div>'}${maps}</article>`}
function render(d){current=String(d.selected_seed);const select=document.getElementById('seed');if(select.options.length!==d.seeds.length)select.innerHTML=d.seeds.map(x=>`<option value="${x}">${x}</option>`).join('');select.value=current;document.getElementById('status').textContent=`${d.completed_seeds}/${d.total_seeds} seeds complete · 当前 ${d.ready_records}/${d.expected_records} experiments ready`;document.getElementById('original').innerHTML=d.original_ready?`<video controls preload="metadata" playsinline src="${video('original','original','original')}"></video>`:'<div class="pending">Original 生成中</div>';let html='<div class="head">Head组 × 阶段</div>'+profiles.map(x=>`<div class="head">${e(x[1])}</div>`).join('');for(const stage of ['all_steps','steps00_09'])for(const group of ['top100','bottom100']){html+=`<div class="row-head ${group.startsWith('top')?'top':'bottom'}"><strong>${group.toUpperCase()}</strong><small>${stage==='all_steps'?'全时间步 S000-S039':'仅前10步 S000-S009'}</small></div>`;for(const [profile] of profiles){const r=d.records.find(x=>x.stage===stage&&x.group===group&&x.profile===profile);html+=`<div class="cell">${card(r)}</div>`}}document.getElementById('matrix').innerHTML=html}
async function load(){const d=await fetch(`/api/attention-additive-lora-seed-sweep/catalog?seed=${encodeURIComponent(current)}`,{cache:'no-store'}).then(r=>r.json());render(d)}document.getElementById('seed').addEventListener('change',ev=>{current=ev.target.value;const u=new URL(location.href);u.searchParams.set('seed',current);history.replaceState(null,'',u);load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


def object_query_attention_overlay_asset(stage: str, profile: str, name: str):
    if (
        stage not in {"all_steps", "steps00_09"}
        or profile not in {item[0] for item in SEED_SWEEP_PROFILES}
        or Path(name).name != name
        or not name.endswith(".jpg")
    ):
        return None
    return OBJECT_QUERY_OVERLAY_PILOT_ROOT / stage / profile / "overlays" / name


def object_query_attention_overlay_catalog(
    stage: str, profile: str, group: str
):
    allowed_profiles = {item[0] for item in SEED_SWEEP_PROFILES}
    if stage not in {"all_steps", "steps00_09"}:
        stage = "all_steps"
    if profile not in allowed_profiles:
        profile = "alpha090"
    if group not in {"top100", "bottom100"}:
        group = "top100"
    root = OBJECT_QUERY_OVERLAY_PILOT_ROOT / stage / profile / "overlays"
    payload = load_payload(root / "manifest.json") or {}
    records = []
    for record in payload.get("records", []):
        if record.get("group") != group:
            continue
        images = record.get("images", {})
        ready = all(
            bool(
                (asset := object_query_attention_overlay_asset(stage, profile, str(name)))
                and asset.is_file()
            )
            for name in images.values()
        )
        records.append({**record, "ready": ready})
    records.sort(key=lambda item: (int(item.get("block", 0)), int(item.get("head", 0))))
    return {
        "case": ATTENTION_LORA_CASE,
        "seed": 90094,
        "stage": stage,
        "profile": profile,
        "group": group,
        "profiles": [
            {"id": profile_id, "label": label}
            for profile_id, label in SEED_SWEEP_PROFILES
        ],
        "records": records,
    }


def object_query_attention_overlay_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object Query Attention Overlay</title><style>
:root{--paper:#eee8dc;--ink:#17251f;--line:#bcb19d;--card:#fffdf8;--red:#ad422e;--green:#17685c}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 5% 0,#e99e5555,transparent 32rem),radial-gradient(circle at 98% 4%,#4b937655,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:10;padding:17px 24px;background:#eee8dced;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(27px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;flex-wrap:wrap;align-items:center}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#58675f}main{width:min(2280px,calc(100% - 18px));margin:auto;padding:20px 0 70px}.video-panel,.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:15px}.video-panel{max-width:760px}.video-panel video{display:block;width:100%;background:#131714}.panel.top{border-left:7px solid var(--red)}.panel.bottom{border-left:7px solid var(--green)}.panel h2{margin:0 0 8px}.meta{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:9px}.pill{padding:5px 8px;background:#e8e1d3;border-radius:99px;font:11px ui-monospace,monospace}.images{display:grid;gap:10px}.images figure{margin:0}.images img{display:block;width:100%;min-width:1900px;border:1px solid var(--line);background:#111}.images figcaption{font-weight:900;margin:4px 0}.scroll{overflow:auto}.pending{padding:50px;border:1px dashed var(--line);background:var(--card)}@media(max-width:800px){header{position:static}}
</style></head><body><header><a href="/attention-additive-lora-seed-sweep?v=1&seed=90094">返回 Seed Sweep</a><h1>Object Query Attention Overlay</h1><p>Seed 90094 · drop_ball GT Query tokens · 13 Query latent frames × 13 Key latent frames · RGB F000,F004,...,F048</p><div class="tools"><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><label>Experiment <select id="profile"></select></label><label>Group <select id="group"><option value="top100">Top10 Heads</option><option value="bottom100">Bottom10 Heads</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="video" class="video-panel"></section><section id="records"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),params=new URL(location.href).searchParams;let stage=params.get('stage')||'all_steps',profile=params.get('profile')||'alpha090',group=params.get('group')||'top100';const image=(name)=>`/api/object-query-attention-overlay/image?stage=${stage}&profile=${profile}&name=${encodeURIComponent(name)}`,video=()=>`/api/attention-additive-lora-seed-sweep/video?seed=90094&stage=${stage}&profile=${profile}&group=${group}`;
function syncUrl(){const u=new URL(location.href);u.searchParams.set('stage',stage);u.searchParams.set('profile',profile);u.searchParams.set('group',group);history.replaceState(null,'',u)}function render(d){const profileSelect=document.getElementById('profile');if(profileSelect.options.length!==d.profiles.length)profileSelect.innerHTML=d.profiles.map(x=>`<option value="${e(x.id)}">${e(x.label)}</option>`).join('');profileSelect.value=profile;document.getElementById('stage').value=stage;document.getElementById('group').value=group;document.getElementById('status').textContent=`${d.records.filter(x=>x.ready).length}/${d.records.length||10} heads ready`;document.getElementById('video').innerHTML=`<h2>对应生成视频</h2><video controls preload="metadata" playsinline src="${video()}"></video>`;document.getElementById('records').innerHTML=d.records.length?d.records.map(r=>`<article class="panel ${group.startsWith('top')?'top':'bottom'}"><h2>L${String(r.block).padStart(2,'0')} / H${String(r.head).padStart(2,'0')}</h2><div class="meta"><span class="pill">PCK@32 ${Number(r.pck32).toFixed(3)}</span><span class="pill">S${String(r.step).padStart(3,'0')}</span><span class="pill">${group==='top100'?'Top10':'Bottom10'}</span></div>${r.ready?`<div class="images"><figure><figcaption>Before</figcaption><div class="scroll"><img loading="lazy" src="${image(r.images.before)}"></div></figure><figure><figcaption>After</figcaption><div class="scroll"><img loading="lazy" src="${image(r.images.after)}"></div></figure><figure><figcaption>|Delta|</figcaption><div class="scroll"><img loading="lazy" src="${image(r.images.abs_delta)}"></div></figure></div>`:'<div class="pending">Capture / overlay 生成中</div>'}</article>`).join('):'<div class="pending">该组合正在捕获，点击手动刷新查看。</div>'}
async function load(){const d=await fetch(`/api/object-query-attention-overlay/catalog?stage=${stage}&profile=${profile}&group=${group}`,{cache:'no-store'}).then(r=>r.json());render(d)}for(const id of ['stage','profile','group'])document.getElementById(id).addEventListener('change',ev=>{if(id==='stage')stage=ev.target.value;if(id==='profile')profile=ev.target.value;if(id==='group')group=ev.target.value;syncUrl();load()});document.getElementById('refresh').addEventListener('click',load);load();
</script></body></html>'''


if __name__ == "__main__":
    viewer.main()
