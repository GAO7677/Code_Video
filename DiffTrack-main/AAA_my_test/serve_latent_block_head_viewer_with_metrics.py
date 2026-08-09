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
ATTENTION_LORA_TEST5_10SEED_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_test5_20case_10seed"
)
ATTENTION_LORA_TEST5_10SEED_METRIC_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "attention_lora_seed_sweep_metrics_test5_20case_10seed"
)
ATTENTION_LORA_TEST5_2SEED_10STEP_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "attention_lora_seed_sweep_test5_20case_2seed_steps10"
)
ATTENTION_LORA_TEST5_2SEED_10STEP_SEEDS = (90094, 35075)

ATTENTION_NEIGHBOR_SEED90094_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed090094_case001460"
)
OBJECT_QUERY_OVERLAY_PILOT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "object_query_attention_overlay_headwise_pck_case001460_seed090094"
)
ATTENTION_LORA_SEED_METRICS_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_metrics_case001460"
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
<a class="card new" href="/attention-additive-lora-test5-10seed?v=1"><div><span>16B / TEST5 10-SEED SWEEP</span><h2>20 Cases × 10 Seeds × 全实验</h2><p>固定同一组 10 seeds，逐 case 比较 32 个 Attention 干预配置、Original、Q@K 热力图及自动指标进度。</p></div><span class="go">打开多 Case Sweep</span></a>
<a class="card new" href="/object-query-attention-overlay?v=1&stage=all_steps&profile=alpha150&group=top100"><div><span>17 / OBJECT QUERY OVERLAY</span><h2>α=1.5 Top100 · 13×13 Latent 时间轴</h2><p>直达 all-steps、alpha150、Top100 视图，将 Object Query 的 Before、After 与 |Delta| overlay 到对应视频帧。</p></div><span class="go">打开 α=1.5 Object Query 对比</span></a>
<a class="card new" href="/object-query-frozen-trajectory?v=21&seed=47326&step=9&branch=conditional&viz=reverse"><div><span>17B / FROZEN TRAJECTORY</span><h2>Seed 47326 · S009 · Conditional</h2><p>直达 reverse 视图，对比 10-step/40-step Object Query 对齐、Attention 替换、固定掩码与移除区域。</p></div><span class="go">打开 Frozen Trajectory</span></a>
<a class="card new" href="/attention-neighbor-ranking-seed90094?v=1"><div><span>18 / NEIGHBOR RANKING</span><h2>Seed 90094 · 8种 Head 排名</h2><p>比较 Wan+LoRA 在八种 Neighbor/Diagonal 排名 Top100、Bottom100 下的完整注意力干预矩阵。</p></div><span class="go">打开 Neighbor Ranking</span></a>
<a class="card new" href="/attention-lora-pck32-seed90094?v=1"><div><span>19 / LORA PCK@32</span><h2>Seed 90094 · PCK@32 全实验</h2><p>固定 LoRA PCK@32 排名，在单页展示全部 Attention 实验、CFG 分支、阶段及 Top/Bottom100。</p></div><span class="go">打开 PCK@32 全实验</span></a>
<a class="card new" href="/attention-lora-pck32-temporal-test5?v=1"><div><span>20 / TEMPORAL TEST5</span><h2>LoRA PCK@32 Top100 · 20 Cases</h2><p>逐 case 对比 Wan+LoRA Original 与 Temporal Causal 在 S000-S009、S000-S039 两个阶段的结果。</p></div><span class="go">打开 Temporal Causal 对比</span></a>
<a class="card new" href="/object-query-continuity-overlay?v=1"><div><span>21 / OBJECT CONTINUITY</span><h2>Object Query 跨帧空间连续性</h2><p>LoRA PCK@32 Top100，分别约束 Object A/B query，使生成帧高响应仅保留与上一帧空间相邻的区域。</p></div><span class="go">打开 Object Continuity 对比</span></a>
'''
STEP_ALIGNMENT_PORTAL_CARD = r'''
<a class="card new" href="/object-query-step-alignment?v=1"><div><span>22 / DENOISING ALIGNMENT</span><h2>10-Step × 40-Step Attention 对齐</h2><p>比较 Wan+LoRA 与标准 Wan2.2-TI2V Baseline 的逐 Head、Top30/50/100 Object Query Attention 相似度。</p></div><span class="go">打开去噪步对齐图谱</span></a>
<a class="card new" href="/wan22-ti2v-baseline-seeds?v=1"><div><span>23 / OFFICIAL TI2V SEEDS</span><h2>Wan2.2-TI2V Baseline 全 Seed 视频</h2><p>单页展示 prompt + 首帧官方 TI2V 推理的全部六个 seed，并排比较 40-step 与 10-step。</p></div><span class="go">打开 Baseline Seed 视频墙</span></a>
<a class="card new" href="/wan22-ti2v-legacy-test5?v=1"><div><span>24 / LEGACY TI2V TEST5</span><h2>Wan2.2-TI2V 旧批次视频墙</h2><p>同页展示 legacy DiffSynth、seed 42、40-step、704×1280、49帧生成的全部 test_5 case。</p></div><span class="go">打开 Legacy Test5 视频墙</span></a>
'''
UNLISTED_PORTAL_CARD = r'''
<a class="card new" href="/top5-head-zero-ablation?v=1"><div><span>25 / LEGACY TOP5 消融</span><h2>Top5 PCK Head 分阶段消融</h2><p>旧版 Top5 PCK Head 输出置零页面，按 case、模型和去噪时间段检查生成视频。</p></div><span class="go">打开 TOP5 消融页面</span></a>
<a class="card new" href="/representative-ranking-heatmaps?v=1"><div><span>26 / REPRESENTATIVE HEATMAPS</span><h2>代表排名 Head 热力图</h2><p>按排名组查看代表 Head 的 attention 热力图，快速对比不同策略的 Top / Bottom 表现。</p></div><span class="go">打开代表热力图</span></a>
<a class="card new" href="/attention-additive-lora-seed-sweep-metrics?v=1"><div><span>27 / 50-SEED METRICS</span><h2>50-Seed 全指标均值</h2><p>汇总 50 个 seed 的共同完成指标，对所有 Attention 干预方法计算均值并标注最佳项。</p></div><span class="go">打开 Seed 指标表</span></a>
<a class="card new" href="/object-query-continuity-comparison?v=4"><div><span>28 / CONTINUITY OLD VS NEW</span><h2>Object Query 新旧连续性对比</h2><p>同页比较旧版与新版 Object Query continuity 方案的生成视频、轨迹和 attention overlay。</p></div><span class="go">打开新旧对比</span></a>
<a class="card new" href="/physiq025-object-query-continuity-comparison?v=1"><div><span>29 / PHYSIQ025 CONTINUITY</span><h2>PhysicIQ025 Object Query 对比</h2><p>针对 PhysicIQ025 case 检查 Object Query continuity 的旧方案、新方案和原始视频。</p></div><span class="go">打开 PhysicIQ025 对比</span></a>
<a class="card new" href="/object-query-top100-mean-overlay?v=1"><div><span>30 / TOP100 MEAN OVERLAY</span><h2>Top100 Head Mean Attention</h2><p>展示 Top100 Head 平均后的 Object Query Attention，对比 baseline、旧方案和新方案。</p></div><span class="go">打开 Top100 Mean 页面</span></a>
<a class="card new" href="/object-query-group-mean-continuity?v=3"><div><span>31 / GROUP-MEAN CONTINUITY</span><h2>Top100 Group-Mean Continuity</h2><p>查看 Top100 组均值的 continuity mask、after attention 和 removed 区域。</p></div><span class="go">打开 Group-Mean 页面</span></a>
<a class="card new" href="/physiq025-object-query-frozen-trajectory?v=1"><div><span>32 / PHYSIQ025 FROZEN</span><h2>PhysIQ025 Frozen Trajectory</h2><p>PhysIQ025 case 的 frozen Object Query trajectory 页面，集中展示视频、对象轨迹和注意力图。</p></div><span class="go">打开 PhysIQ025 Frozen</span></a>
<a class="card new" href="/wan22-ti2v-legacy-pck50?v=2"><div><span>33 / LEGACY PCK50</span><h2>五组 PCK Head 排名与重合</h2><p>首个 latent frame 固定为 object query；查看 Legacy、GT、LoRA、Baseline、三模型综合在 S039 与全步平均下的 720 Head 排名、30 × 24 矩阵、Top-K 重合和相关性。</p></div><span class="go">打开 PCK Head 对比</span></a>
<a class="card new" href="/wan22-ti2v-legacy-physiciq67-samples?v=1"><div><span>34 / PHYSICIQ67 SAMPLES</span><h2>新 Legacy Object Query 样例</h2><p>固定随机抽取已完成的 PhysicIQ67 runs，展示生成视频、SAM2 object query、单 run PCK 矩阵和 S039 Top10 attention。</p></div><span class="go">打开 PhysicIQ67 样例</span></a>
<a class="card new" href="/object-query-ablation-metrics?v=1"><div><span>35 / OBJECT QUERY METRICS</span><h2>Fixed × Tube 消融量化诊断</h2><p>001460 / seed 47326 的 49 个 Top100 视频；同时对比未消融 Baseline 与 simulator/source GT，并展示轨迹、mask、RAFT、DINO 和 LPIPS 的真实计算量。</p></div><span class="go">打开消融指标页</span></a>
'''
viewer.PORTAL = viewer.PORTAL.replace(
    "</section>", PORTAL_CARD + VIDEOS_PORTAL_CARD + QK_ATTENTION_PORTAL_CARD + ATTENTION_LORA_PORTAL_CARD + MONO_SCALE_HEAD_PORTAL_CARD + MONO_SCALE_LORA_VIDEO_PORTAL_CARD + ATTENTION_LORA_SEED_SWEEP_PORTAL_CARD + STEP_ALIGNMENT_PORTAL_CARD + UNLISTED_PORTAL_CARD + "</section>", 1
)


from AAA_my_test import serve_attention_noise_metrics as combined_metrics
from AAA_my_test.object_query_ablation_metrics import dashboard as object_query_metrics_dashboard


class MetricsHandler(viewer.Handler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/object-query-ablation-metrics":
            self.send_payload(
                object_query_metrics_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/object-query-ablation-metrics/catalog":
            payload = json.dumps(
                object_query_metrics_dashboard.catalog(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-ablation-metrics/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_metrics_dashboard.asset(params.get("path", [""])[0])
            if asset is None:
                raise FileNotFoundError("unknown object-query metric asset")
            content_type = {
                ".mp4": "video/mp4",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".json": "application/json; charset=utf-8",
                ".csv": "text/csv; charset=utf-8",
            }.get(asset.suffix.lower(), "application/octet-stream")
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/api/object-query-ablation-metrics/input-video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_metrics_dashboard.input_video(params.get("id", [""])[0])
            if asset is None:
                raise FileNotFoundError("unknown object-query input video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
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
                (
                    attention_neighbor_seed90094_page()
                    if getattr(self.server, "server_port", 0) == 61882
                    else attention_lora_seed_sweep_page()
                ).encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/attention-additive-lora-test5-10seed":
            self.send_payload(
                attention_lora_test5_10seed_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/representative-ranking-heatmaps":
            self.send_payload(
                representative_ranking_heatmaps_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/attention-neighbor-ranking-seed90094":
            self.send_payload(
                attention_neighbor_seed90094_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/attention-lora-pck32-seed90094":
            self.send_payload(
                attention_lora_pck32_seed90094_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/attention-lora-pck32-temporal-test5":
            self.send_payload(
                attention_lora_pck32_temporal_test5_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/attention-lora-pck32-temporal-test5/catalog":
            payload = json.dumps(
                attention_lora_pck32_temporal_test5_catalog(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/attention-lora-pck32-temporal-test5/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = attention_lora_pck32_temporal_test5_asset(
                params.get("case", [""])[0], params.get("kind", [""])[0]
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown temporal test5 video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/object-query-continuity-overlay":
            self.send_payload(
                object_query_continuity_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-continuity-comparison":
            self.send_payload(
                object_query_continuity_comparison_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/physiq025-object-query-continuity-comparison":
            self.send_payload(
                physiq025_object_query_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-top100-mean-overlay":
            self.send_payload(
                object_query_top100_mean_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-group-mean-continuity":
            self.send_payload(
                object_query_group_mean_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-frozen-trajectory":
            self.send_payload(
                object_query_frozen_trajectory_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/physiq025-object-query-frozen-trajectory":
            self.send_payload(
                physiq025_object_query_frozen_trajectory_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/object-query-frozen-trajectory/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                object_query_frozen_trajectory_catalog(
                    params.get("seed", ["47326"])[0],
                    params.get("stage", ["all_steps"])[0],
                    params.get("step", ["39"])[0],
                    params.get("branch", ["conditional"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/physiq025-object-query-frozen-trajectory/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                physiq025_object_query_frozen_trajectory_catalog(
                    params.get("seed", [""])[0],
                    params.get("stage", ["all_steps"])[0],
                    params.get("step", ["9"])[0],
                    params.get("branch", ["conditional"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path in {
            "/api/physiq025-object-query-frozen-trajectory/video",
            "/api/physiq025-object-query-frozen-trajectory/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = physiq025_object_query_frozen_trajectory_asset(
                params.get("seed", [""])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("label", [""])[0],
                params.get("kind", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown PhysIQ025 frozen-trajectory asset")
            content_type = "video/mp4" if path.endswith("/video") else "image/jpeg"
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path in {
            "/api/object-query-frozen-trajectory/image",
            "/api/object-query-frozen-trajectory/video",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_frozen_trajectory_asset(
                params.get("seed", ["47326"])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("kind", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown frozen-trajectory asset")
            viewer.send_file_with_range(
                self, asset, "video/mp4" if path.endswith("/video") else "image/jpeg"
            )
            return
        if path == "/api/object-query-group-mean-continuity/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                object_query_group_mean_catalog(
                    params.get("seed", ["90094"])[0],
                    params.get("stage", ["all_steps"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-group-mean-continuity/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_group_mean_asset(
                params.get("seed", ["90094"])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("kind", ["group"])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown group-mean object-query image")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-group-mean-continuity/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_group_mean_video(
                params.get("seed", ["90094"])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("kind", ["group"])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown group-mean object-query video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-top100-mean-overlay/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                object_query_top100_mean_catalog(
                    params.get("seed", ["90094"])[0],
                    params.get("stage", ["all_steps"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-top100-mean-overlay/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_top100_mean_video_asset(
                params.get("seed", ["90094"])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("kind", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown Top100 mean object-query video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-top100-mean-overlay/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_top100_mean_asset(
                params.get("seed", ["90094"])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("method", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown Top100 mean object-query overlay")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/physiq025-object-query-continuity-comparison/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                physiq025_object_query_catalog(
                    params.get("seed", [""])[0],
                    params.get("stage", ["all_steps"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path in {
            "/api/physiq025-object-query-continuity-comparison/video",
            "/api/physiq025-object-query-continuity-comparison/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = physiq025_object_query_asset(
                params.get("seed", [""])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("method", [""])[0],
                params.get("kind", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown PhysicIQ025 object-query asset")
            content_type = "video/mp4" if path.endswith("/video") else "image/jpeg"
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/api/object-query-continuity-overlay/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                object_query_continuity_catalog(
                    params.get("seed", ["90094"])[0],
                    params.get("stage", ["all_steps"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path in {
            "/api/object-query-continuity-overlay/video",
            "/api/object-query-continuity-overlay/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_continuity_asset(
                params.get("seed", ["90094"])[0],
                params.get("stage", ["all_steps"])[0],
                params.get("kind", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown object continuity asset")
            content_type = "video/mp4" if path.endswith("/video") else "image/jpeg"
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/api/attention-neighbor-ranking-seed90094/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                attention_neighbor_seed90094_catalog(
                    params.get("criterion", ["strict_score"])[0],
                    params.get("branch", ["both"])[0],
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path in {
            "/api/attention-neighbor-ranking-seed90094/video",
            "/api/attention-neighbor-ranking-seed90094/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = attention_neighbor_seed90094_asset(
                params.get("criterion", ["strict_score"])[0],
                params.get("stage", [""])[0],
                params.get("profile", [""])[0],
                params.get("group", [""])[0],
                params.get("name", [""])[0],
                params.get("branch", ["both"])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown neighbor-ranking asset")
            content_type = "video/mp4" if path.endswith("/video") else "image/png"
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/api/attention-additive-lora-seed-sweep/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            if getattr(self.server, "server_port", 0) == 61882:
                catalog = attention_neighbor_seed90094_catalog(
                    params.get("criterion", ["strict_score"])[0]
                )
            else:
                catalog = attention_lora_seed_sweep_catalog(
                    params.get("seed", [""])[0]
                )
            payload = json.dumps(catalog, ensure_ascii=False).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/attention-additive-lora-test5-10seed/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = json.dumps(
                attention_lora_test5_10seed_catalog(
                    params.get("case", [""])[0]
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/attention-additive-lora-test5-10seed/video10":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = attention_lora_test5_10step_asset(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("profile", [""])[0],
                params.get("group", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown Test5 10-step video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path in {
            "/api/attention-additive-lora-test5-10seed/video",
            "/api/attention-additive-lora-test5-10seed/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = attention_lora_test5_10seed_asset(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("stage", [""])[0],
                params.get("profile", [""])[0],
                params.get("group", [""])[0],
                params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown test5 10-seed sweep asset")
            content_type = "video/mp4" if path.endswith("/video") else "image/png"
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path in {
            "/api/attention-additive-lora-seed-sweep/video",
            "/api/attention-additive-lora-seed-sweep/image",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            if getattr(self.server, "server_port", 0) == 61882:
                asset = attention_neighbor_seed90094_asset(
                    params.get("criterion", ["strict_score"])[0],
                    params.get("stage", [""])[0],
                    params.get("profile", [""])[0],
                    params.get("group", [""])[0],
                )
            else:
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
        if path == "/api/object-query-attention-overlay/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_attention_overlay_video_asset(
                params.get("stage", [""])[0],
                params.get("profile", [""])[0],
                params.get("group", [""])[0],
                params.get("kind", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown object-query overlay video")
            viewer.send_file_with_range(self, asset, "video/mp4")
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
        if path == "/attention-additive-lora-seed-sweep-metrics":
            self.send_payload(
                attention_lora_seed_sweep_metrics_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/attention-additive-lora-seed-sweep-metrics/summary":
            payload = json.dumps(
                attention_lora_seed_sweep_metrics_summary(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/object-query-step-alignment":
            self.send_payload(
                object_query_step_alignment_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/wan22-ti2v-baseline-seeds":
            self.send_payload(
                wan22_ti2v_baseline_seeds_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/wan22-ti2v-legacy-test5":
            self.send_payload(
                wan22_ti2v_legacy_test5_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/wan22-ti2v-legacy-pck50":
            self.send_payload(
                wan22_ti2v_legacy_pck50_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/wan22-ti2v-legacy-physiciq67-samples":
            self.send_payload(
                wan22_ti2v_legacy_physiciq67_visual_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/catalog":
            payload = json.dumps(
                wan22_ti2v_legacy_physiciq67_visual_catalog(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/wan22-ti2v-legacy-pck50/catalog":
            payload = json.dumps(
                wan22_ti2v_legacy_pck50_catalog(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/wan22-ti2v-legacy-pck50/comparison":
            payload = json.dumps(
                wan22_ti2v_legacy_pck50_comparison(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/pck-head-rankings":
            payload = json.dumps(
                pck_head_rankings_payload(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/downloads/pck-head-rankings.json":
            if not PCK_HEAD_RANKINGS_JSON.is_file():
                raise FileNotFoundError("PCK head rankings JSON is unavailable")
            viewer.send_file_with_range(
                self, PCK_HEAD_RANKINGS_JSON, "application/json; charset=utf-8"
            )
            return
        if path == "/downloads/pck-head-rankings.md":
            if not PCK_HEAD_RANKINGS_MD.is_file():
                raise FileNotFoundError("PCK head rankings Markdown is unavailable")
            viewer.send_file_with_range(
                self, PCK_HEAD_RANKINGS_MD, "text/markdown; charset=utf-8"
            )
            return
        if path == "/api/wan22-ti2v-legacy-pck50/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_pck50_video(
                params.get("case", [""])[0], params.get("seed", [""])[0]
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown legacy TI2V PCK50 video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/wan22-ti2v-legacy-pck50/heatmap":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = wan22_ti2v_legacy_pck50_heatmap(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("rank", ["0"])[0],
                params.get("region", ["object_A"])[0],
            )
            if payload is None:
                raise FileNotFoundError("legacy TI2V PCK50 heatmap is not ready")
            self.send_payload(payload, "image/jpeg")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_visual_video(
                params.get("case", [""])[0], params.get("seed", [""])[0]
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown PhysicIQ67 visual sample video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/query-image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_query_image(
                params.get("case", [""])[0],
                params.get("region", ["all"])[0],
                params.get("seed", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown PhysicIQ67 object-query image")
            viewer.send_file_with_range(self, asset, "image/png")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/heatmap":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = wan22_ti2v_legacy_physiciq67_visual_heatmap(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("rank", ["0"])[0],
                params.get("region", ["object_A"])[0],
            )
            if payload is None:
                raise FileNotFoundError("PhysicIQ67 visual sample heatmap is not ready")
            self.send_payload(payload, "image/jpeg")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/mean-heatmap":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            payload = wan22_ti2v_legacy_physiciq67_mean_heatmap(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("top_n", ["30"])[0],
                params.get("region", ["object_A"])[0],
            )
            if payload is None:
                raise FileNotFoundError("PhysicIQ67 Top-N mean heatmap is not ready")
            self.send_payload(payload, "image/jpeg")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/ablation-video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_ablation_video(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("target_scope", [""])[0],
                params.get("mask_mode", [""])[0],
                params.get("top_n", [""])[0],
                params.get("region", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("PhysicIQ67 attention-matrix video is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/temporal-tube-ablation-video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_temporal_tube_video(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("target_scope", [""])[0],
                params.get("mask_mode", [""])[0],
                params.get("top_n", [""])[0],
                params.get("region", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("PhysicIQ67 temporal-tube ablation video is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/raft-flow-video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_raft_flow_video(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("video_id", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("PhysicIQ67 RAFT flow video is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/wan22-ti2v-legacy-test5/catalog":
            payload = json.dumps(
                wan22_ti2v_legacy_test5_catalog(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/wan22-ti2v-legacy-test5/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_test5_asset(
                params.get("name", [""])[0]
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown legacy TI2V test5 video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-step-alignment/catalog":
            payload = json.dumps(
                object_query_step_alignment_catalog(), ensure_ascii=False
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path in {
            "/api/object-query-step-alignment/image",
            "/api/object-query-step-alignment/download",
        }:
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_step_alignment_asset(
                params.get("model", [""])[0], params.get("name", [""])[0]
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown object-query step-alignment asset")
            content_type = (
                "image/png"
                if path.endswith("/image")
                else "text/csv; charset=utf-8"
            )
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/api/object-query-step-alignment/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_step_alignment_video(
                params.get("model", [""])[0],
                params.get("seed", [""])[0],
                params.get("steps", [""])[0],
                params.get("kind", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("unknown object-query step-alignment video")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-s09-fixed-mask-multiseed/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            kernel = params.get("kernel", [""])[0]
            seed = params.get("seed", [""])[0]
            safe_name = Path(name).name
            allowed_seeds = {"90094", "35075", "21890", "49530", "32466"}
            if seed not in allowed_seeds or kernel not in {"1", "2", "3"}:
                raise FileNotFoundError("unknown S09 multiseed mask asset")
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_s09_fixed_delta_mask_multiseed_case001460/"
                f"seed_{int(seed):06d}/mask_{kernel}x{kernel}/removal_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("S09 multiseed mask overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-s09-fixed-mask-multiseed/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            kernel = params.get("kernel", [""])[0]
            seed = params.get("seed", [""])[0]
            allowed_seeds = {"90094", "35075", "21890", "49530", "32466"}
            if seed not in allowed_seeds or kernel not in {"1", "2", "3"}:
                raise FileNotFoundError("unknown S09 multiseed mask video")
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_s09_fixed_delta_mask_multiseed_case001460/"
                f"seed_{int(seed):06d}/mask_{kernel}x{kernel}/removal_run/lora/cases/"
                "0613pybullet_sample_001460_w002/top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("S09 multiseed mask video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-s09-fixed-mask-no-renorm/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            kernel = params.get("kernel", [""])[0]
            safe_name = Path(name).name
            if kernel not in {"1", "2", "3"}:
                raise FileNotFoundError("unknown S09-fixed no-renorm mask kernel")
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_s09_fixed_delta_mask_no_renorm_case001460/"
                f"seed_047326/mask_{kernel}x{kernel}/removal_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("S09-fixed no-renorm mask overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-s09-fixed-mask-no-renorm/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            kernel = params.get("kernel", [""])[0]
            if kernel not in {"1", "2", "3"}:
                raise FileNotFoundError("unknown S09-fixed no-renorm mask kernel")
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_s09_fixed_delta_mask_no_renorm_case001460/"
                f"seed_047326/mask_{kernel}x{kernel}/removal_run/lora/cases/"
                "0613pybullet_sample_001460_w002/top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("S09-fixed no-renorm mask video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-s09-fixed-mask/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            kernel = params.get("kernel", [""])[0]
            safe_name = Path(name).name
            if kernel not in {"1", "2", "3"}:
                raise FileNotFoundError("unknown S09-fixed mask kernel")
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_s09_fixed_delta_mask_case001460/"
                f"seed_047326/mask_{kernel}x{kernel}/removal_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("S09-fixed mask overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-s09-fixed-mask/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            kernel = params.get("kernel", [""])[0]
            if kernel not in {"1", "2", "3"}:
                raise FileNotFoundError("unknown S09-fixed mask kernel")
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_s09_fixed_delta_mask_case001460/"
                f"seed_047326/mask_{kernel}x{kernel}/removal_run/lora/cases/"
                "0613pybullet_sample_001460_w002/top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("S09-fixed mask video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-similarity-delta-mask/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            safe_name = Path(name).name
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_similarity_delta_mask1x1_case001460/"
                "seed_047326/removal_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("similarity-delta mask overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-similarity-delta-mask/video":
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_similarity_delta_mask1x1_case001460/"
                "seed_047326/removal_run/lora/cases/0613pybullet_sample_001460_w002/"
                "top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("similarity-delta mask removal video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-positive-delta-mask/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            safe_name = Path(name).name
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_positive_delta_mask1x1_case001460/"
                "seed_047326/removal_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("positive-delta mask overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-positive-delta-mask/video":
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_positive_delta_mask1x1_case001460/"
                "seed_047326/removal_run/lora/cases/0613pybullet_sample_001460_w002/"
                "top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("positive-delta mask removal video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-same-index-transplant/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            safe_name = Path(name).name
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_same_index_attention_transplant_case001460/"
                "seed_047326/replacement_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("same-index transplant overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-same-index-transplant/video":
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_same_index_attention_transplant_case001460/"
                "seed_047326/replacement_run/lora/cases/0613pybullet_sample_001460_w002/"
                "top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("same-index attention transplant video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-sigma-transplant/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            safe_name = Path(name).name
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_sigma_attention_transplant_case001460/"
                "seed_047326/replacement_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("sigma transplant Top100 Mean overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-sigma-transplant/video":
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_sigma_attention_transplant_case001460/"
                "seed_047326/replacement_run/lora/cases/0613pybullet_sample_001460_w002/"
                "top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("sigma matched attention transplant video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-reverse-transplant/image":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            name = params.get("name", [""])[0]
            safe_name = Path(name).name
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_reverse_attention_transplant_case001460/"
                "seed_047326/replacement_overlays"
            ) / safe_name
            if safe_name != name or not asset.is_file():
                raise FileNotFoundError("reverse transplant Top100 Mean overlay is pending")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/object-query-reverse-transplant/video":
            asset = Path(
                "/data/gaoya/agent-data/outputs/object_query_reverse_attention_transplant_case001460/"
                "seed_047326/replacement_run/lora/cases/0613pybullet_sample_001460_w002/"
                "top100_steps_00_40.mp4"
            )
            if not asset.is_file():
                raise FileNotFoundError("reverse matched attention transplant video is pending")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        super().do_GET()


viewer.Handler = MetricsHandler


OBJECT_QUERY_STEP_ALIGNMENT_MODELS = {
    "lora": {
        "label": "Wan+LoRA",
        "detail": "LoRA PCK@32 固定物理 Head · context-video pipeline",
        "attention_capture": True,
        "root": Path(
            "/data/gaoya/agent-data/outputs/object_query_attention_step10_vs_step40"
        ),
    },
    "baseline": {
        "label": "Wan2.2-TI2V-5B Baseline",
        "detail": "官方 DiffSynth 示例批处理 · prompt + 首帧 · 无 context_video · 无 attention hook",
        "attention_capture": False,
        "video_root": Path(
            "/data/gaoya/agent-data/outputs/wan22_ti2v_official_first_frame_seed_sweep"
        ),
        "root": Path(
            "/data/gaoya/agent-data/outputs/"
            "object_query_attention_step10_vs_step40_baseline_official_ti2v"
        ),
    },
}
WAN22_TI2V_LEGACY_TEST5_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/basemodel/wan2p2_ti2v5B_frame49"
)
WAN22_TI2V_LEGACY_PCK50_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50"
)
WAN22_TI2V_LEGACY_PCK50_CACHE = Path(
    "/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_regions_704x1280"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_ROOT / "visual_samples"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_MANIFEST = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT / "samples.json"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_ZERO_ROOT = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT / "attention_zero_ablations"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_ROOT = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT / "attention_matrix_ablations_v2"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT / "attention_zero_seed47326"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_MANIFEST = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT / "cases.json"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_MULTISEED_MANIFEST = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT / "cases_001460_5seeds.json"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT
    / "attention_matrix_ablations_temporal_tube_v1"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_VBENCH_METRICS = (
    ("vbench_subject_consistency", "Subject"),
    ("vbench_background_consistency", "Background"),
    ("vbench_temporal_flickering", "Flicker"),
    ("vbench_motion_smoothness", "Smoothness"),
    ("vbench_dynamic_degree", "Dynamic"),
    ("vbench_aesthetic_quality", "Aesthetic"),
    ("vbench_imaging_quality", "Imaging"),
)
WAN22_TI2V_LEGACY_PCK50_SEEDS = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds.txt"
)
PCK_HEAD_RANKINGS_JSON = Path(
    "/data/gaoya/agent-data/outputs/pck_head_rankings/pck_head_rankings.json"
)
PCK_HEAD_RANKINGS_MD = Path(
    "/data/gaoya/agent-data/outputs/pck_head_rankings/pck_head_rankings.md"
)
WAN22_TI2V_LEGACY_PCK50_CASES = (
    "0613pybullet_sample_000301_w000",
    "0613pybullet_sample_000331_w001",
    "0613pybullet_sample_001455_w000",
    "0613pybullet_sample_000336_w001",
    "0613pybullet_sample_001460_w002",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed",
)


def pck_head_rankings_payload() -> dict:
    """Load the stable two-view PCK ranking export used by the comparison UI."""
    if not PCK_HEAD_RANKINGS_JSON.is_file():
        return {"ready": False, "reason": f"missing {PCK_HEAD_RANKINGS_JSON}"}
    try:
        payload = json.loads(PCK_HEAD_RANKINGS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ready": False, "reason": f"cannot read PCK ranking export: {exc}"}
    if not isinstance(payload, dict) or not isinstance(payload.get("views"), dict):
        return {"ready": False, "reason": "PCK ranking export has an invalid schema"}
    return payload


def wan22_ti2v_legacy_pck50_performance():
    """Return PCK/error matrices for S039 and the all-step aggregate."""
    counts_path = WAN22_TI2V_LEGACY_PCK50_ROOT / "aggregate" / "combined_counts.npz"
    if not counts_path.is_file():
        return {"ready": False, "matrices": {}, "step_count": 40}
    try:
        import numpy as np

        with np.load(counts_path) as arrays:
            correct = np.asarray(arrays["correct32"], dtype=np.float64)
            comparisons = np.asarray(arrays["comparisons"], dtype=np.float64)
            error_sum = np.asarray(arrays["error_sum"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return {"ready": False, "matrices": {}, "step_count": 40}
    if correct.shape != (40, 30, 24) or comparisons.shape != correct.shape:
        return {"ready": False, "matrices": {}, "step_count": 40}

    def records(correct_matrix, comparisons_matrix, error_matrix):
        with np.errstate(divide="ignore", invalid="ignore"):
            pck = np.divide(
                100.0 * correct_matrix,
                comparisons_matrix,
                out=np.full(correct_matrix.shape, np.nan),
                where=comparisons_matrix > 0,
            )
            mean_error = np.divide(
                error_matrix,
                comparisons_matrix,
                out=np.full(correct_matrix.shape, np.nan),
                where=comparisons_matrix > 0,
            )
        rows = []
        for block in range(30):
            for head in range(24):
                pck_value = pck[block, head]
                error_value = mean_error[block, head]
                rows.append(
                    {
                        "block": block,
                        "head": head,
                        "pck32": float(pck_value) if np.isfinite(pck_value) else None,
                        "mean_error_px": float(error_value)
                        if np.isfinite(error_value)
                        else None,
                        "comparisons": int(comparisons_matrix[block, head]),
                    }
                )
        return rows

    all_correct = correct.sum(axis=0)
    all_comparisons = comparisons.sum(axis=0)
    all_error_sum = error_sum.sum(axis=0)
    matrices = {
        "s039": records(correct[39], comparisons[39], error_sum[39]),
        "all_steps_mean": records(all_correct, all_comparisons, all_error_sum),
    }
    return {
        "ready": True,
        "step_count": int(correct.shape[0]),
        "matrices": matrices,
        "aggregation": "micro aggregate over 6 cases × 50 seeds × 40 steps",
    }


def _wan22_ti2v_legacy_pck50_comparison_from_catalog():
    performance = wan22_ti2v_legacy_pck50_performance()
    neighbor = viewer.neighbor_diagonal_catalog()
    neighbor_rows = neighbor.get("records", [])
    if not performance.get("ready") or len(neighbor_rows) != 720:
        return {
            "ready": False,
            "reason": "Legacy or neighbor-diagonal 720-head summary is unavailable",
        }

    import numpy as np

    top_ks = (10, 30, 50, 100)
    scope_specs = (
        ("s039", "Legacy S039"),
        ("all_steps_mean", "Legacy S000-S039 average"),
    )
    reference_specs = (
        ("gt", "GT teacher-forced", "gt_pck32", "#d49a25"),
        ("lora", "LoRA", "lora_pck32", "#197d72"),
        ("baseline", "Wan2.2 Baseline", "baseline_pck32", "#bd4d36"),
        ("combined", "Three-model combined", "pck32", "#3d568f"),
    )

    def dataset(rows, label, color, value_key="pck32"):
        normalized = [
            {
                "block": int(row["block"]),
                "head": int(row["head"]),
                "pck32": float(row[value_key]),
            }
            for row in rows
            if row.get(value_key) is not None
        ]
        ordered = sorted(
            normalized,
            key=lambda row: (-row["pck32"], row["block"], row["head"]),
        )
        ranks = {
            (row["block"], row["head"]): index
            for index, row in enumerate(ordered, start=1)
        }
        for row in normalized:
            row["rank"] = ranks[(row["block"], row["head"])]
        normalized.sort(key=lambda row: (row["block"], row["head"]))
        values = np.asarray([row["pck32"] for row in normalized], dtype=np.float64)
        percentiles = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
        return {
            "label": label,
            "color": color,
            "rows": normalized,
            "distribution": {
                "count": int(values.size),
                "min": float(percentiles[0]),
                "p10": float(percentiles[1]),
                "p25": float(percentiles[2]),
                "median": float(percentiles[3]),
                "mean": float(values.mean()),
                "p75": float(percentiles[4]),
                "p90": float(percentiles[5]),
                "max": float(percentiles[6]),
                "std": float(values.std()),
            },
        }

    def average_ranks(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            average = (start + 1 + end) / 2.0
            for position in range(start, end):
                ranks[order[position]] = average
            start = end
        return ranks

    def pearson(left, right):
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        left_centered = left_array - left_array.mean()
        right_centered = right_array - right_array.mean()
        denominator = np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
        if denominator <= 0:
            return None
        return float(np.dot(left_centered, right_centered) / denominator)

    def compare(left, right):
        left_rows = {
            (row["block"], row["head"]): row for row in left["rows"]
        }
        right_rows = {
            (row["block"], row["head"]): row for row in right["rows"]
        }
        keys = sorted(left_rows.keys() & right_rows.keys())
        left_values = [left_rows[key]["pck32"] for key in keys]
        right_values = [right_rows[key]["pck32"] for key in keys]
        differences = np.asarray(left_values) - np.asarray(right_values)
        overlaps = {}
        for top_k in top_ks:
            left_top = {
                key for key in keys if left_rows[key]["rank"] <= top_k
            }
            right_top = {
                key for key in keys if right_rows[key]["rank"] <= top_k
            }
            common = sorted(
                left_top & right_top,
                key=lambda key: (
                    left_rows[key]["rank"] + right_rows[key]["rank"],
                    left_rows[key]["rank"],
                    right_rows[key]["rank"],
                    key,
                ),
            )
            common_count = len(common)
            union_count = len(left_top | right_top)
            overlaps[str(top_k)] = {
                "common_count": common_count,
                "coverage_pct": 100.0 * common_count / top_k,
                "jaccard": common_count / union_count if union_count else 0.0,
                "common_heads": [
                    {
                        "block": key[0],
                        "head": key[1],
                        "legacy_rank": left_rows[key]["rank"],
                        "reference_rank": right_rows[key]["rank"],
                        "legacy_pck32": left_rows[key]["pck32"],
                        "reference_pck32": right_rows[key]["pck32"],
                    }
                    for key in common
                ],
            }
        return {
            "pair_count": len(keys),
            "pearson": pearson(left_values, right_values),
            "spearman": pearson(
                average_ranks(left_values), average_ranks(right_values)
            ),
            "mean_delta": float(differences.mean()),
            "mean_abs_delta": float(np.abs(differences).mean()),
            "overlaps": overlaps,
        }

    references = {
        key: dataset(neighbor_rows, label, color, value_key)
        for key, label, value_key, color in reference_specs
    }
    scope_colors = {"s039": "#202d29", "all_steps_mean": "#8e5a2d"}
    scopes = {}
    for key, label in scope_specs:
        legacy = dataset(
            performance["matrices"][key], label, scope_colors[key]
        )
        scopes[key] = {
            **legacy,
            "comparisons": {
                reference_key: compare(legacy, reference)
                for reference_key, reference in references.items()
            },
        }
    result = {
        "ready": True,
        "top_ks": list(top_ks),
        "scopes": scopes,
        "references": references,
        "protocol_note": (
            "Legacy uses 6 cases x 50 seeds; neighbor-diagonal uses S039 over "
            "GT, LoRA, and Baseline on 5 cases. Physical Block/Head IDs are aligned."
        ),
    }
    ranking = pck_head_rankings_payload()
    if ranking.get("ready", True) is not False:
        series_labels = {
            "legacy_s039": "Legacy S039",
            "gt": "GT teacher-forced",
            "lora": "LoRA",
            "baseline": "Wan2.2 Baseline",
            "combined": "Three-model combined",
        }
        result["ranking_export"] = {
            "generated_at_utc": ranking.get("generated_at_utc"),
            "json_path": str(PCK_HEAD_RANKINGS_JSON),
            "md_path": str(PCK_HEAD_RANKINGS_MD),
            "views": {
                view_id: {
                    "label": view.get("label", view_id),
                    "source_steps": view.get("source_steps", []),
                    "series": list(view.get("series", {}).keys()),
                }
                for view_id, view in ranking.get("views", {}).items()
            },
        }
        result["pairwise_by_view"] = {}
        for view_id, pairs in ranking.get(
            "pairwise_comparisons_by_view", {}
        ).items():
            enriched = {}
            for pair_id, pair in pairs.items():
                left_id = pair.get("left_series")
                right_id = pair.get("right_series")
                enriched[pair_id] = {
                    **pair,
                    "left_label": series_labels.get(left_id, left_id),
                    "right_label": series_labels.get(right_id, right_id),
                }
            result["pairwise_by_view"][view_id] = enriched
    else:
        result["ranking_export"] = ranking
        result["pairwise_by_view"] = {}
    return result


def wan22_ti2v_legacy_pck50_comparison():
    """Build both comparison views from the exported, two-view ranking JSON."""
    ranking = pck_head_rankings_payload()
    if ranking.get("ready", True) is False:
        return ranking

    import numpy as np

    labels = {
        "legacy_s039": "Legacy S039",
        "gt": "GT teacher-forced",
        "lora": "LoRA",
        "baseline": "Wan2.2 Baseline",
        "combined": "Three-model combined",
    }
    colors = {
        "legacy_s039": "#202d29",
        "gt": "#d49a25",
        "lora": "#197d72",
        "baseline": "#bd4d36",
        "combined": "#3d568f",
    }
    top_ks = (10, 30, 50, 100)

    def dataset(series_id, series, view_id):
        rows = [
            {
                "block": int(row["block"]),
                "head": int(row["head"]),
                "pck32": float(row["pck32"]),
                "rank": int(row["rank"]),
            }
            for row in series.get("ranked_heads", [])
            if row.get("pck32") is not None
        ]
        rows.sort(key=lambda row: (row["block"], row["head"]))
        values = np.asarray([row["pck32"] for row in rows], dtype=np.float64)
        percentiles = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
        return {
            "id": series_id,
            "label": labels[series_id],
            "color": colors[series_id],
            "view_id": view_id,
            "rows": rows,
            "distribution": {
                "count": int(values.size),
                "min": float(percentiles[0]),
                "p10": float(percentiles[1]),
                "p25": float(percentiles[2]),
                "median": float(percentiles[3]),
                "mean": float(values.mean()),
                "p75": float(percentiles[4]),
                "p90": float(percentiles[5]),
                "max": float(percentiles[6]),
                "std": float(values.std()),
            },
        }

    def add_common_heads(raw_pair, left, right):
        left_rows = {(row["block"], row["head"]): row for row in left["rows"]}
        right_rows = {(row["block"], row["head"]): row for row in right["rows"]}
        result = {
            "pair_count": raw_pair.get(
                "pair_count", len(left_rows.keys() & right_rows.keys())
            ),
            "pearson": raw_pair.get("pearson_pck32"),
            "spearman": raw_pair.get("spearman_pck32"),
            "mean_delta": raw_pair.get("mean_delta_left_minus_right"),
            "mean_abs_delta": raw_pair.get("mean_abs_delta_pck32"),
            "overlaps": {},
        }
        for top_k in top_ks:
            left_top = {
                key for key, row in left_rows.items() if row["rank"] <= top_k
            }
            right_top = {
                key for key, row in right_rows.items() if row["rank"] <= top_k
            }
            common = sorted(
                left_top & right_top,
                key=lambda key: (
                    left_rows[key]["rank"] + right_rows[key]["rank"],
                    left_rows[key]["rank"],
                    right_rows[key]["rank"],
                    key,
                ),
            )
            raw_overlap = raw_pair["overlaps"][f"Top{top_k}"]
            result["overlaps"][str(top_k)] = {
                **raw_overlap,
                "common_heads": [
                    {
                        "block": key[0],
                        "head": key[1],
                        "legacy_rank": left_rows[key]["rank"],
                        "reference_rank": right_rows[key]["rank"],
                        "legacy_pck32": left_rows[key]["pck32"],
                        "reference_pck32": right_rows[key]["pck32"],
                    }
                    for key in common
                ],
            }
        return result

    scopes = {}
    pairwise_by_view = {}
    for view_id, view in ranking.get("views", {}).items():
        series = view.get("series", {})
        datasets = {
            series_id: dataset(series_id, series_data, view_id)
            for series_id, series_data in series.items()
            if series_id in labels
        }
        legacy = datasets.get("legacy_s039")
        if legacy is None or len(legacy["rows"]) != 720:
            continue
        references = {
            series_id: datasets[series_id]
            for series_id in ("gt", "lora", "baseline", "combined")
            if series_id in datasets
        }
        raw_pairs = ranking.get("pairwise_comparisons_by_view", {}).get(view_id, {})
        comparisons = {}
        for reference_id, reference in references.items():
            pair_id = f"legacy_s039__{reference_id}"
            raw_pair = raw_pairs.get(pair_id)
            if raw_pair is not None:
                comparisons[reference_id] = add_common_heads(
                    raw_pair, legacy, reference
                )
        scopes[view_id] = {
            **legacy,
            "label": labels["legacy_s039"]
            if view_id == "s039"
            else "Legacy S000-S039 average",
            "view_label": view.get("label", view_id),
            "references": references,
            "comparisons": comparisons,
        }
        enriched_pairs = {}
        for pair_id, raw_pair in raw_pairs.items():
            left_id = raw_pair.get("left_series")
            right_id = raw_pair.get("right_series")
            enriched_pairs[pair_id] = {
                **raw_pair,
                "left_label": labels.get(left_id, left_id),
                "right_label": labels.get(right_id, right_id),
            }
        pairwise_by_view[view_id] = enriched_pairs

    return {
        "ready": bool(scopes),
        "top_ks": list(top_ks),
        "scopes": scopes,
        "references": scopes.get("s039", {}).get("references", {}),
        "pairwise_by_view": pairwise_by_view,
        "ranking_export": {
            "generated_at_utc": ranking.get("generated_at_utc"),
            "json_path": str(PCK_HEAD_RANKINGS_JSON),
            "md_path": str(PCK_HEAD_RANKINGS_MD),
            "views": {
                view_id: {
                    "label": view.get("label", view_id),
                    "source_steps": view.get("source_steps", []),
                    "series": list(view.get("series", {}).keys()),
                }
                for view_id, view in ranking.get("views", {}).items()
            },
        },
    }


def wan22_ti2v_legacy_pck50_catalog():
    seeds = [
        int(value)
        for value in WAN22_TI2V_LEGACY_PCK50_SEEDS.read_text().splitlines()
        if value.strip()
    ] if WAN22_TI2V_LEGACY_PCK50_SEEDS.is_file() else []
    summary_path = WAN22_TI2V_LEGACY_PCK50_ROOT / "aggregate" / "summary.json"
    summary = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    progress = []
    availability = {}
    objects = {}
    for case in WAN22_TI2V_LEGACY_PCK50_CASES:
        pck_count = 0
        heatmap_count = 0
        availability[case] = {}
        for seed in seeds:
            run = WAN22_TI2V_LEGACY_PCK50_ROOT / "runs" / case / f"seed_{seed:05d}"
            heatmap = WAN22_TI2V_LEGACY_PCK50_ROOT / "heatmaps" / case / f"seed_{seed:05d}"
            pck_ready = (run / "complete.json").is_file()
            heatmap_ready = (heatmap / "complete.json").is_file()
            pck_count += int(pck_ready)
            heatmap_count += int(heatmap_ready)
            availability[case][str(seed)] = {
                "pck": pck_ready,
                "video": (run / "generated.mp4").is_file(),
                "heatmap": heatmap_ready,
            }
        region_path = WAN22_TI2V_LEGACY_PCK50_CACHE / case / "regions.json"
        region_ready = region_path.is_file()
        names = []
        if region_ready:
            try:
                region_payload = json.loads(region_path.read_text(encoding="utf-8"))
                names = [
                    region["region_name"]
                    for region in region_payload.get("regions", [])
                    if region.get("region_type") == "object"
                ]
            except (OSError, json.JSONDecodeError):
                names = []
        objects[case] = names or ["object_A"]
        progress.append(
            {
                "case": case,
                "region": region_ready,
                "pck": pck_count,
                "heatmap": heatmap_count,
            }
        )
    totals = {
        "regions_done": sum(int(item["region"]) for item in progress),
        "regions_total": len(WAN22_TI2V_LEGACY_PCK50_CASES),
        "pck_done": sum(int(item["pck"]) for item in progress),
        "pck_total": len(WAN22_TI2V_LEGACY_PCK50_CASES) * len(seeds),
        "heatmap_done": sum(int(item["heatmap"]) for item in progress),
        "heatmap_total": len(WAN22_TI2V_LEGACY_PCK50_CASES) * len(seeds),
        "ranking_final": bool(summary.get("final", False)),
    }
    totals["work_done"] = (
        totals["regions_done"] + totals["pck_done"] + totals["heatmap_done"]
    )
    totals["work_total"] = (
        totals["regions_total"] + totals["pck_total"] + totals["heatmap_total"]
    )
    return {
        "protocol": "Legacy DiffSynth Wan2.2-TI2V-5B · prompt + first frame · first latent query",
        "matrix": "6 cases x 50 seeds x 40 steps x 30 blocks x 24 heads",
        "cases": list(WAN22_TI2V_LEGACY_PCK50_CASES),
        "seeds": seeds,
        "progress": progress,
        "availability": availability,
        "objects": objects,
        "summary": summary,
        "totals": totals,
        "performance": wan22_ti2v_legacy_pck50_performance(),
    }


def wan22_ti2v_legacy_pck50_video(case: str, seed: str):
    if case not in WAN22_TI2V_LEGACY_PCK50_CASES:
        return None
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    return WAN22_TI2V_LEGACY_PCK50_ROOT / "runs" / case / f"seed_{seed_value:05d}" / "generated.mp4"


def _wan22_ti2v_legacy_attention_montage(
    video_path: Path, selected, anchors, panel_label: str, title_label: str
):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        return None
    panels = []
    for latent_index, probability in enumerate(selected):
        normalized = probability - float(np.nanmin(probability))
        normalized /= max(float(np.nanmax(normalized)), 1.0e-12)
        color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        pixel_index = min(int(anchors[latent_index]), len(frames) - 1)
        frame = cv2.resize(frames[pixel_index], (480, 264), interpolation=cv2.INTER_AREA)
        color = cv2.resize(color, (480, 264), interpolation=cv2.INTER_NEAREST)
        panel = cv2.addWeighted(frame, 0.55, color, 0.45, 0)
        label = f"K{latent_index:02d}/F{pixel_index:02d} | {panel_label}"
        cv2.putText(panel, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(panel, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        panels.append(panel)
    columns = 4
    blank = np.full_like(panels[0], 245)
    while len(panels) % columns:
        panels.append(blank.copy())
    rows = [
        np.concatenate(panels[index : index + columns], axis=1)
        for index in range(0, len(panels), columns)
    ]
    montage = np.concatenate(rows, axis=0)
    title = np.full((64, montage.shape[1], 3), (232, 224, 210), dtype=np.uint8)
    cv2.putText(title, title_label, (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (23, 68, 58), 2)
    ok, encoded = cv2.imencode(
        ".jpg", np.concatenate([title, montage], axis=0), [cv2.IMWRITE_JPEG_QUALITY, 91]
    )
    return None if not ok else encoded.tobytes()


def _wan22_ti2v_legacy_heatmap_payload(
    heatmap_dir: Path, video_path: Path, rank_value: int, region: str
):
    import numpy as np

    metadata_path = heatmap_dir / "metadata.json"
    maps_path = heatmap_dir / "attention_maps.npy"
    if not metadata_path.is_file() or not maps_path.is_file() or not video_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    regions = metadata.get("regions", [])
    if region not in regions or not 0 <= rank_value < len(metadata.get("entries", [])):
        return None
    maps = np.load(maps_path, mmap_mode="r")
    selected = np.asarray(maps[rank_value, regions.index(region)], dtype=np.float32)
    anchors = metadata.get("latent_anchor_pixel_frames", list(range(len(selected))))
    entry = metadata["entries"][rank_value]
    return _wan22_ti2v_legacy_attention_montage(
        video_path,
        selected,
        anchors,
        f"S{entry['step']:02d} L{entry['block']:02d} H{entry['head']:02d}",
        (
            f"Rank {rank_value + 1} | {region} | per-frame color scale | "
            f"PCK@32 {entry.get('pck32', 0):.2f}%"
        ),
    )


def wan22_ti2v_legacy_pck50_heatmap(case: str, seed: str, rank: str, region: str):
    if case not in WAN22_TI2V_LEGACY_PCK50_CASES:
        return None
    try:
        seed_value, rank_value = int(seed), int(rank)
    except ValueError:
        return None
    video_path = wan22_ti2v_legacy_pck50_video(case, seed)
    if video_path is None:
        return None
    heatmap_dir = WAN22_TI2V_LEGACY_PCK50_ROOT / "heatmaps" / case / f"seed_{seed_value:05d}"
    return _wan22_ti2v_legacy_heatmap_payload(
        heatmap_dir, video_path, rank_value, region
    )


def wan22_ti2v_legacy_physiciq67_visual_manifest():
    if not WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_MANIFEST.is_file():
        return {
            "ready": False,
            "reason": "visual sample manifest has not been generated",
            "samples": [],
            "entries": [],
        }
    try:
        payload = json.loads(
            WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_MANIFEST.read_text(encoding="utf-8")
        )
        for requested_path in (
            WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_MANIFEST,
            WAN22_TI2V_LEGACY_PHYSICIQ67_MULTISEED_MANIFEST,
        ):
            if not requested_path.is_file():
                continue
            requested = json.loads(
                requested_path.read_text(encoding="utf-8")
            )
            if requested.get("entries") != payload.get("entries"):
                raise ValueError(
                    f"requested ranking snapshot does not match: {requested_path}"
                )
            seen = {
                (str(row.get("case")), int(row.get("seed", -1)))
                for row in payload.get("samples", [])
            }
            payload["samples"].extend(
                row
                for row in requested.get("samples", [])
                if (str(row.get("case")), int(row.get("seed", -1))) not in seen
            )
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ready": False, "reason": str(exc), "samples": [], "entries": []}


WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_MASKS = (
    "self_only",
    "incoming_only",
    "outgoing_only",
    "query_row",
    "key_value_column",
    "cross_boundary",
    "row_and_column",
)
WAN22_TI2V_LEGACY_PHYSICIQ67_ALL_TOKEN_CONTROLS = (
    "qk_logits_zero",
    "full_head_output",
)
WAN22_TI2V_LEGACY_PHYSICIQ67_PROTOCOL = "attention_matrix_ablation_v2"
WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_PROTOCOL = (
    "attention_matrix_ablation_temporal_object_tube_v1"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE = "0613pybullet_sample_001460_w002"
WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_SEEDS = frozenset(
    (47326, 90094, 68613, 35075, 32466, 13248)
)
WAN22_TI2V_LEGACY_PHYSICIQ67_RAFT_DIR = "raft_motion_top100_v1"


def _wan22_ti2v_legacy_physiciq67_similarity(case: str, seed: int) -> dict[str, Any]:
    path = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT
        / case
        / f"seed_{seed:05d}"
        / "video_similarity_top100.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("case") != case
            or int(payload.get("seed", -1)) != seed
            or int(payload.get("video_count", -1)) != 49
            or int(payload.get("ablation_video_count", -1)) != 48
        ):
            raise ValueError("video similarity payload does not match the pilot")
        return {**payload, "ready": True}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ready": False, "reason": str(exc)}


def _wan22_ti2v_legacy_physiciq67_raft_motion(
    case: str, seed: int
) -> dict[str, Any]:
    path = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT
        / case
        / f"seed_{seed:05d}"
        / WAN22_TI2V_LEGACY_PHYSICIQ67_RAFT_DIR
        / "raft_motion_similarity_top100.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("case") != case
            or int(payload.get("seed", -1)) != seed
            or int(payload.get("video_count", -1)) != 49
            or int(payload.get("comparison_count", -1)) != 240
        ):
            raise ValueError("RAFT motion payload does not match the pilot")
        return {**payload, "ready": True}
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ready": False, "reason": str(exc)}


def _wan22_ti2v_legacy_physiciq67_vbench_scores(payload: dict) -> dict[str, float]:
    scores = {}
    for field, _label in WAN22_TI2V_LEGACY_PHYSICIQ67_VBENCH_METRICS:
        metric = payload.get(field)
        value = finite_number(metric.get("score")) if isinstance(metric, dict) else None
        if value is not None:
            scores[field] = value
    return scores


def _wan22_ti2v_legacy_physiciq67_vbench_report(
    payload: dict, baseline_scores: dict[str, float]
) -> dict[str, Any]:
    scores = _wan22_ti2v_legacy_physiciq67_vbench_scores(payload)
    metrics = []
    for field, label in WAN22_TI2V_LEGACY_PHYSICIQ67_VBENCH_METRICS:
        score, baseline = scores.get(field), baseline_scores.get(field)
        metrics.append(
            {
                "field": field,
                "label": label,
                "score": score,
                "baseline": baseline,
                "delta": None
                if score is None or baseline is None
                else round(score - baseline, 6),
            }
        )
    return {
        "completed": len(scores),
        "expected": len(WAN22_TI2V_LEGACY_PHYSICIQ67_VBENCH_METRICS),
        "metrics": metrics,
    }


def _wan22_ti2v_legacy_physiciq67_ablation_variant(
    target_scope: str, mask_mode: str, top_n: int, region: str = ""
):
    if top_n not in {30, 50, 100}:
        return None
    if mask_mode in WAN22_TI2V_LEGACY_PHYSICIQ67_ALL_TOKEN_CONTROLS:
        if target_scope != "all_tokens":
            return None
        return f"{mask_mode}__all_tokens__top{top_n:03d}"
    if mask_mode not in WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_MASKS + (
        "literal_kv_zero",
    ):
        return None
    if target_scope == "single_object":
        if not region:
            return None
        target = region
    elif target_scope == "all_objects":
        target = "all_objects"
    else:
        return None
    return f"{target_scope}__{target}__{mask_mode}__top{top_n:03d}"


def _wan22_ti2v_legacy_physiciq67_ablation_records(
    sample: dict, entries: list[dict], baseline_scores: dict[str, float]
):
    case, seed = str(sample["case"]), int(sample["seed"])
    ablation_root = Path(
        str(
            sample.get("matrix_ablation_root")
            or WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_ROOT
        )
    )
    regions = [
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    ]
    targets = [("single_object", region) for region in regions]
    targets.append(("all_objects", ""))
    specs = [
        (target_scope, region, mask_mode)
        for target_scope, region in targets
        for mask_mode in WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_MASKS
        + ("literal_kv_zero",)
    ]
    specs.extend(
        ("all_tokens", "", mask_mode)
        for mask_mode in WAN22_TI2V_LEGACY_PHYSICIQ67_ALL_TOKEN_CONTROLS
    )
    records = []
    for target_scope, region, mask_mode in specs:
        for top_n in (30, 50, 100):
            variant = _wan22_ti2v_legacy_physiciq67_ablation_variant(
                target_scope, mask_mode, top_n, region
            )
            root = (
                ablation_root
                / case
                / f"seed_{seed:05d}"
                / str(variant)
            )
            video_path = root / "generated.mp4"
            metadata_path = root / "manifest.json"
            complete_path = root / "complete.json"
            ready = video_path.is_file() and metadata_path.is_file() and complete_path.is_file()
            selected_token_count = None
            vbench = _wan22_ti2v_legacy_physiciq67_vbench_report({}, baseline_scores)
            if ready:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    ready = (
                        metadata.get("case") == case
                        and int(metadata.get("seed", -1)) == seed
                        and metadata.get("target_scope") == target_scope
                        and metadata.get("mask_mode") == mask_mode
                        and int(metadata.get("top_n", -1)) == top_n
                        and str(metadata.get("region") or "") == region
                        and metadata.get("selected_entries") == entries[:top_n]
                        and metadata.get("protocol")
                        == WAN22_TI2V_LEGACY_PHYSICIQ67_PROTOCOL
                    )
                    if ready and target_scope != "all_tokens":
                        selected_token_count = len(
                            metadata.get("audit", {}).get("query_token_indices") or []
                        )
                    if ready:
                        vbench = _wan22_ti2v_legacy_physiciq67_vbench_report(
                            metadata, baseline_scores
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    ready = False
            records.append(
                {
                    "target_scope": target_scope,
                    "mask_mode": mask_mode,
                    "region": region or None,
                    "top_n": top_n,
                    "variant_id": variant,
                    "ready": ready,
                    "error": (root / "error.txt").is_file(),
                    "selected_token_count": selected_token_count,
                    "vbench": vbench,
                }
            )
    return records


def _wan22_ti2v_legacy_physiciq67_temporal_tube_records(
    sample: dict, entries: list[dict], baseline_scores: dict[str, float]
):
    case, seed = str(sample["case"]), int(sample["seed"])
    regions = [
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    ]
    targets = [("single_object", region) for region in regions]
    targets.append(("all_objects", ""))
    records = []
    for target_scope, region in targets:
        for mask_mode in WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_MASKS + (
            "literal_kv_zero",
        ):
            variant = _wan22_ti2v_legacy_physiciq67_ablation_variant(
                target_scope, mask_mode, 100, region
            )
            root = (
                WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT
                / case
                / f"seed_{seed:05d}"
                / str(variant)
            )
            metadata_path = root / "manifest.json"
            ready = all(
                (root / name).is_file()
                for name in ("complete.json", "manifest.json", "generated.mp4")
            )
            selected_token_count = None
            latent_frame_token_counts = None
            vbench = _wan22_ti2v_legacy_physiciq67_vbench_report({}, baseline_scores)
            if ready:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    ready = (
                        metadata.get("case") == case
                        and int(metadata.get("seed", -1)) == seed
                        and metadata.get("target_scope") == target_scope
                        and metadata.get("mask_mode") == mask_mode
                        and int(metadata.get("top_n", -1)) == 100
                        and str(metadata.get("region") or "") == region
                        and metadata.get("selected_entries") == entries[:100]
                        and metadata.get("protocol")
                        == WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_PROTOCOL
                    )
                    if ready:
                        audit = metadata.get("audit", {})
                        selected_token_count = len(audit.get("query_token_indices") or [])
                        latent_frame_token_counts = audit.get("latent_frame_token_counts")
                        vbench = _wan22_ti2v_legacy_physiciq67_vbench_report(
                            metadata, baseline_scores
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    ready = False
            records.append(
                {
                    "target_scope": target_scope,
                    "mask_mode": mask_mode,
                    "region": region or None,
                    "top_n": 100,
                    "variant_id": variant,
                    "ready": ready,
                    "error": (root / "error.txt").is_file(),
                    "selected_token_count": selected_token_count,
                    "latent_frame_token_counts": latent_frame_token_counts,
                    "vbench": vbench,
                }
            )
    return records


def wan22_ti2v_legacy_physiciq67_visual_catalog():
    payload = wan22_ti2v_legacy_physiciq67_visual_manifest()
    if payload.get("ready") is False:
        return payload
    summary_path = WAN22_TI2V_LEGACY_PHYSICIQ67_ROOT / "aggregate" / "summary.json"
    try:
        aggregate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        aggregate_summary = {}
    samples = []
    for row in payload.get("samples", []):
        case, seed = str(row["case"]), int(row["seed"])
        heatmap_root = (
            WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT
            / "heatmaps" / case / f"seed_{seed:05d}"
        )
        sample = dict(row)
        video_path = Path(
            str(
                sample.get("baseline_video")
                or WAN22_TI2V_LEGACY_PHYSICIQ67_ROOT
                / "runs"
                / case
                / f"seed_{seed:05d}"
                / "generated.mp4"
            )
        )
        query_root = Path(
            str(
                sample.get("query_cache_dir")
                or WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT / "regions" / case
            )
        )
        sample["video_ready"] = video_path.is_file()
        try:
            baseline_payload = json.loads(
                (video_path.parent / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            baseline_payload = {}
        baseline_scores = _wan22_ti2v_legacy_physiciq67_vbench_scores(
            baseline_payload
        )
        sample["vbench"] = _wan22_ti2v_legacy_physiciq67_vbench_report(
            baseline_payload, baseline_scores
        )
        sample["query_visual_ready"] = (query_root / "sam2_regions_points.png").is_file()
        attention_files_ready = (
            (heatmap_root / "complete.json").is_file()
            and (heatmap_root / "metadata.json").is_file()
            and (heatmap_root / "attention_maps.npy").is_file()
        )
        attention_matches_snapshot = False
        if attention_files_ready:
            try:
                heatmap_metadata = json.loads(
                    (heatmap_root / "metadata.json").read_text(encoding="utf-8")
                )
                attention_matches_snapshot = (
                    heatmap_metadata.get("entries") == payload.get("entries")
                )
            except (OSError, json.JSONDecodeError):
                attention_matches_snapshot = False
        sample["attention_ready"] = attention_files_ready and attention_matches_snapshot
        sample["attention_matrix_ablations"] = (
            _wan22_ti2v_legacy_physiciq67_ablation_records(
                sample, payload.get("entries", []), baseline_scores
            )
        )
        if (
            case == WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE
            and seed in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_SEEDS
        ):
            sample["temporal_tube_attention_matrix_ablations"] = (
                _wan22_ti2v_legacy_physiciq67_temporal_tube_records(
                    sample, payload.get("entries", []), baseline_scores
                )
            )
            sample["ablation_video_similarity"] = (
                _wan22_ti2v_legacy_physiciq67_similarity(case, seed)
            )
            sample["ablation_raft_motion"] = (
                _wan22_ti2v_legacy_physiciq67_raft_motion(case, seed)
            )
        samples.append(sample)
    ablation_records = [
        record
        for sample in samples
        for record in sample["attention_matrix_ablations"]
    ]
    return {
        **payload,
        "ready": True,
        "samples": samples,
        "aggregate_progress": {
            "completed_runs": int(aggregate_summary.get("completed_runs", 0)),
            "expected_runs": int(aggregate_summary.get("expected_runs", 3350)),
            "final": bool(aggregate_summary.get("final", False)),
        },
        "ablation_progress": {
            "ready": sum(int(record["ready"]) for record in ablation_records),
            "failed": sum(int(record["error"]) for record in ablation_records),
            "expected": len(ablation_records),
        },
    }


def _wan22_ti2v_legacy_physiciq67_sample(case: str, seed: str):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    manifest = wan22_ti2v_legacy_physiciq67_visual_manifest()
    for row in manifest.get("samples", []):
        if str(row.get("case")) == case and int(row.get("seed", -1)) == seed_value:
            return row
    return None


def wan22_ti2v_legacy_physiciq67_visual_video(case: str, seed: str):
    sample = _wan22_ti2v_legacy_physiciq67_sample(case, seed)
    if sample is None:
        return None
    return Path(
        str(
            sample.get("baseline_video")
            or (
                WAN22_TI2V_LEGACY_PHYSICIQ67_ROOT
                / "runs"
                / case
                / f"seed_{int(seed):05d}"
                / "generated.mp4"
            )
        )
    )


def wan22_ti2v_legacy_physiciq67_ablation_video(
    case: str,
    seed: str,
    target_scope: str,
    mask_mode: str,
    top_n: str,
    region: str,
):
    sample = _wan22_ti2v_legacy_physiciq67_sample(case, seed)
    if sample is None:
        return None
    try:
        seed_value, count = int(seed), int(top_n)
    except ValueError:
        return None
    object_regions = {
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    }
    if target_scope == "single_object" and region not in object_regions:
        return None
    if target_scope != "single_object":
        region = ""
    variant = _wan22_ti2v_legacy_physiciq67_ablation_variant(
        target_scope, mask_mode, count, region
    )
    if variant is None:
        return None
    return (
        Path(
            str(
                sample.get("matrix_ablation_root")
                or WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_ROOT
            )
        )
        / case
        / f"seed_{seed_value:05d}"
        / variant
        / "generated.mp4"
    )


def wan22_ti2v_legacy_physiciq67_temporal_tube_video(
    case: str,
    seed: str,
    target_scope: str,
    mask_mode: str,
    top_n: str,
    region: str,
):
    sample = _wan22_ti2v_legacy_physiciq67_sample(case, seed)
    if sample is None or case != WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE:
        return None
    try:
        seed_value, count = int(seed), int(top_n)
    except ValueError:
        return None
    if (
        seed_value not in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_SEEDS
        or count != 100
    ):
        return None
    object_regions = {
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    }
    if target_scope == "single_object" and region not in object_regions:
        return None
    if target_scope != "single_object":
        region = ""
    variant = _wan22_ti2v_legacy_physiciq67_ablation_variant(
        target_scope, mask_mode, count, region
    )
    if variant is None or mask_mode in WAN22_TI2V_LEGACY_PHYSICIQ67_ALL_TOKEN_CONTROLS:
        return None
    return (
        WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT
        / case
        / f"seed_{seed_value:05d}"
        / variant
        / "generated.mp4"
    )


def wan22_ti2v_legacy_physiciq67_raft_flow_video(
    case: str, seed: str, video_id: str
):
    if case != WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE:
        return None
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if (
        seed_value not in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_SEEDS
        or not video_id
    ):
        return None
    payload = _wan22_ti2v_legacy_physiciq67_raft_motion(case, seed_value)
    if not payload.get("ready"):
        return None
    record = next(
        (row for row in payload.get("videos", []) if row.get("id") == video_id),
        None,
    )
    if record is None or not record.get("raft_flow_video"):
        return None
    allowed_root = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT
        / case
        / f"seed_{seed_value:05d}"
        / WAN22_TI2V_LEGACY_PHYSICIQ67_RAFT_DIR
        / "flow_videos"
    ).resolve()
    asset = Path(str(record["raft_flow_video"])).resolve()
    try:
        asset.relative_to(allowed_root)
    except ValueError:
        return None
    return asset if asset.suffix.lower() == ".mp4" else None


def wan22_ti2v_legacy_physiciq67_query_image(
    case: str, region: str, seed: str = ""
):
    manifest = wan22_ti2v_legacy_physiciq67_visual_manifest()
    sample = _wan22_ti2v_legacy_physiciq67_sample(case, seed) if seed else None
    if sample is None:
        sample = next(
            (row for row in manifest.get("samples", []) if str(row.get("case")) == case),
            None,
        )
    if sample is None:
        return None
    names = {str(row.get("region_name")) for row in sample.get("regions", [])}
    root = Path(
        str(
            sample.get("query_cache_dir")
            or WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT / "regions" / case
        )
    )
    if region in {"", "all"}:
        return root / "sam2_regions_points.png"
    if region not in names:
        return None
    return root / "regions" / region / "mask_points.png"


def wan22_ti2v_legacy_physiciq67_visual_heatmap(
    case: str, seed: str, rank: str, region: str
):
    if _wan22_ti2v_legacy_physiciq67_sample(case, seed) is None:
        return None
    try:
        seed_value, rank_value = int(seed), int(rank)
    except ValueError:
        return None
    heatmap_dir = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT
        / "heatmaps" / case / f"seed_{seed_value:05d}"
    )
    video_path = wan22_ti2v_legacy_physiciq67_visual_video(case, seed)
    if video_path is None:
        return None
    return _wan22_ti2v_legacy_heatmap_payload(
        heatmap_dir, video_path, rank_value, region
    )


def wan22_ti2v_legacy_physiciq67_mean_heatmap(
    case: str, seed: str, top_n: str, region: str
):
    import numpy as np

    if _wan22_ti2v_legacy_physiciq67_sample(case, seed) is None:
        return None
    try:
        seed_value, count = int(seed), int(top_n)
    except ValueError:
        return None
    if count not in {30, 50, 100}:
        return None
    heatmap_dir = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_VISUAL_ROOT
        / "heatmaps" / case / f"seed_{seed_value:05d}"
    )
    metadata_path = heatmap_dir / "metadata.json"
    maps_path = heatmap_dir / "attention_maps.npy"
    video_path = wan22_ti2v_legacy_physiciq67_visual_video(case, seed)
    if (
        video_path is None
        or not video_path.is_file()
        or not metadata_path.is_file()
        or not maps_path.is_file()
    ):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    regions = metadata.get("regions", [])
    entries = metadata.get("entries", [])
    if region not in regions or len(entries) < count:
        return None
    maps = np.load(maps_path, mmap_mode="r")
    if maps.shape[0] < count:
        return None
    selected = np.asarray(
        np.nanmean(maps[:count, regions.index(region)], axis=0), dtype=np.float32
    )
    anchors = metadata.get("latent_anchor_pixel_frames", list(range(len(selected))))
    mean_pck = float(np.mean([float(row.get("pck32", 0.0)) for row in entries[:count]]))
    return _wan22_ti2v_legacy_attention_montage(
        video_path,
        selected,
        anchors,
        f"S39 Top{count} Head Mean",
        (
            f"Top{count} Head Mean | {region} | per-frame color scale | "
            f"mean ranking PCK@32 {mean_pck:.2f}%"
        ),
    )


def wan22_ti2v_legacy_physiciq67_visual_page():
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Legacy PhysicIQ67 Visual Samples</title><link rel="icon" href="data:,"><style>
:root{--paper:#ece4d5;--ink:#17261f;--deep:#17443a;--line:#baad98;--card:#fffaf0;--gold:#d29c35}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#d65f3c44,transparent 34rem),radial-gradient(circle at 100% 0,#27897942,transparent 40rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:15px 22px;background:#ece4d5f2;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:4px 0;font-size:clamp(29px,4.4vw,58px);line-height:1}.lead{max-width:1200px;margin:7px 0}.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}button,select{padding:8px 11px;border:1px solid var(--line);background:var(--card);font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(100% - 18px,2200px);margin:auto;padding:18px 0 70px}.note,.media,.performance,.ranking{padding:14px;margin:14px 0;border:1px solid var(--line);border-radius:16px;background:#fffaf0e8;box-shadow:0 13px 34px #58442b16}.note{border-left:7px solid var(--gold)}.note h2,.performance h2,.ranking h2{margin:0 0 7px}.caption{line-height:1.5;color:#5f5a51}.media-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px}figure{margin:0;border:1px solid #d4c8b5;background:#fff;padding:8px;min-width:0}video,img{display:block;width:100%;background:#111}video{aspect-ratio:1280/704}figcaption{padding:8px 3px 2px;font-weight:900}.pending{display:grid;place-items:center;min-height:260px;background:#f0eadf;color:#766d60;text-align:center}.heat-scroll,.table-scroll{overflow:auto;border:1px solid var(--line);background:#fff}.performance-heat{display:grid;grid-template-columns:48px repeat(24,minmax(34px,1fr));gap:3px;min-width:980px;padding:12px}.axis,.cell{display:flex;align-items:center;justify-content:center;height:29px;border-radius:4px;font:9px ui-monospace,monospace}.axis{color:#6d756f}.cell{border:0;color:#fff;text-shadow:0 1px 2px #000}.cell:hover{outline:2px solid var(--ink)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;border-bottom:1px solid #ddd2c0;text-align:center}th{background:var(--deep);color:#fff}.ranking-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:1100px){header{position:static}.media-grid,.ranking-grid{grid-template-columns:1fr}}</style></head><body><header><a href="/">返回总览</a> · <a href="/wan22-ti2v-legacy-pck50?v=2">旧 Legacy 排名页</a><h1>Legacy PhysicIQ67<br>Object Query Samples</h1><p class="lead">固定随机抽取已完成 runs；F00/K00 object query，704×1280，S039 provisional Top10 attention。样例不覆盖旧 6-case 稳定排名。</p><div class="tools"><label>Sample <select id="sample"></select></label><label>Object <select id="region"></select></label><label>S039 Head <select id="rank"></select></label><label>Matrix <select id="view"><option value="s039">S039</option><option value="all_steps_mean">S000–S039 Mean</option></select></label><label>Metric <select id="metric"><option value="pck32">PCK@32</option><option value="mean_error_px">Mean error</option></select></label><button id="refresh">手动刷新</button><button id="replay">重新播放</button><span id="status" class="status">读取中</span></div></header><main><section class="note"><h2 id="title">读取样例</h2><div id="caption" class="caption"></div></section><section class="media"><div id="media" class="media-grid"></div></section><section class="performance"><h2>单 Run · 30 × 24 Head Matrix</h2><p id="matrixMeta" class="status"></p><div class="heat-scroll"><div id="matrix" class="performance-heat"></div></div></section><section class="ranking"><h2>捕获时的 provisional S039 Top10</h2><div class="ranking-grid"><div id="table" class="table-scroll"></div><div><p>这些 Head 来自样例清单生成时的增量 aggregate。3350 runs 完成前排名仍可能变化；attention capture 只用于定性检查。</p><p id="protocol" class="status"></p></div></div></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),api='/api/wan22-ti2v-legacy-physiciq67-samples',q=new URL(location.href).searchParams;let data=null;const $=id=>document.getElementById(id),key=x=>`${x.case}::${x.seed}`,current=()=>data.samples.find(x=>key(x)===$('sample').value)||data.samples[0];function options(el,rows,value,label){const old=el.value;el.innerHTML=rows.map(x=>`<option value="${e(value(x))}">${e(label(x))}</option>`).join('');if([...el.options].some(x=>x.value===old))el.value=old}function color(v,lo,hi,inverse){let t=hi>lo?Math.max(0,Math.min(1,(v-lo)/(hi-lo))):.5;if(inverse)t=1-t;return `hsl(${12+120*t} 58% ${34+7*t}%)`}function sync(){const s=current(),u=new URL(location.href);u.searchParams.set('case',s.case);u.searchParams.set('seed',s.seed);u.searchParams.set('region',$('region').value);u.searchParams.set('rank',$('rank').value);history.replaceState(null,'',u)}function renderMatrix(){const s=current(),view=$('view').value,metric=$('metric').value,rows=s.matrices?.[view]||[],vals=rows.map(x=>x[metric]).filter(Number.isFinite),lo=Math.min(...vals),hi=Math.max(...vals),map=new Map(rows.map(x=>[`${x.block}-${x.head}`,x])),box=$('matrix');box.innerHTML='<span class="axis">L/H</span>'+[...Array(24)].map((_,h)=>`<span class="axis">H${String(h).padStart(2,'0')}</span>`).join('');for(let b=0;b<30;b++){box.insertAdjacentHTML('beforeend',`<span class="axis">L${String(b).padStart(2,'0')}</span>`);for(let h=0;h<24;h++){const r=map.get(`${b}-${h}`),v=r?.[metric],c=document.createElement('button');c.className='cell';if(Number.isFinite(v)){c.style.background=color(v,lo,hi,metric==='mean_error_px');c.textContent=Number(v).toFixed(metric==='pck32'?1:0);c.title=`L${b} H${h} · PCK@32 ${Number(r.pck32).toFixed(3)}% · error ${Number(r.mean_error_px).toFixed(3)} px · N ${r.comparisons}`}else{c.disabled=true;c.textContent='—'}box.append(c)}}$('matrixMeta').textContent=`${view} · ${metric} · range ${lo.toFixed(3)}–${hi.toFixed(3)}`}function render(){const s=current(),region=$('region').value,rank=Number($('rank').value||0),entry=data.entries[rank],base=`case=${encodeURIComponent(s.case)}&seed=${s.seed}`;$('title').textContent=`${s.category} · ${s.case} · seed ${s.seed}`;$('caption').textContent=s.caption;const video=s.video_ready?`<video controls muted playsinline preload="metadata" src="${api}/video?${base}"></video>`:'<div class="pending">Generated video unavailable</div>',query=s.query_visual_ready?`<img src="${api}/query-image?case=${encodeURIComponent(s.case)}&region=${encodeURIComponent(region)}&v=1">`:'<div class="pending">Query visualization unavailable</div>',heat=s.attention_ready&&entry?`<img src="${api}/heatmap?${base}&rank=${rank}&region=${encodeURIComponent(region)}&v=${Date.now()}">`:'<div class="pending">Attention capture 待生成<br>视频、SAM2 query 和 PCK 矩阵已可查看</div>';$('media').innerHTML=`<figure>${video}<figcaption>Legacy generated video · 40 steps / 49 frames</figcaption></figure><figure>${query}<figcaption>SAM2 mask + 8 object query points · F00/K00</figcaption></figure><figure>${heat}<figcaption>${entry?`#${rank+1} · S39 L${entry.block} H${entry.head} · ${region}`:'S039 attention'}</figcaption></figure>`;renderMatrix();$('table').innerHTML='<table><thead><tr><th>#</th><th>Block</th><th>Head</th><th>PCK@32</th><th>N</th></tr></thead><tbody>'+data.entries.map((x,i)=>`<tr><td>${i+1}</td><td>L${String(x.block).padStart(2,'0')}</td><td>H${String(x.head).padStart(2,'0')}</td><td>${Number(x.pck32).toFixed(3)}%</td><td>${Number(x.comparisons).toLocaleString()}</td></tr>`).join('')+'</tbody></table>';$('protocol').textContent=`Selection seed ${data.selection_seed} · aggregate ${data.completed_runs_at_selection}/3350 · ${data.ranking_status}`;sync()}function sampleChanged(){const s=current();options($('region'),s.regions.filter(x=>x.region_type==='object'),x=>x.region_name,x=>`${x.region_name} · ${x.region_phrase||''}`);render()}async function load(){data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());if(!data.ready){$('status').textContent=data.reason||'样例清单未生成';return}const wantedCase=q.get('case'),wantedSeed=q.get('seed');options($('sample'),data.samples,key,x=>`${x.category} · ${x.case} · seed ${x.seed}`);const wanted=data.samples.find(x=>x.case===wantedCase&&String(x.seed)===wantedSeed);if(wanted)$('sample').value=key(wanted);options($('rank'),data.entries,(x,i)=>i,(x,i)=>`#${i+1} S39 L${String(x.block).padStart(2,'0')} H${String(x.head).padStart(2,'0')}`);if(q.get('rank'))$('rank').value=q.get('rank');sampleChanged();if(q.get('region')&&[...$('region').options].some(x=>x.value===q.get('region')))$('region').value=q.get('region');$('status').textContent=`${data.aggregate_progress.completed_runs}/${data.aggregate_progress.expected_runs} runs · ${data.samples.filter(x=>x.attention_ready).length}/${data.samples.length} attention samples ready`;render()}$('sample').addEventListener('change',sampleChanged);for(const id of ['region','rank'])$(id).addEventListener('change',render);for(const id of ['view','metric'])$(id).addEventListener('change',renderMatrix);$('refresh').addEventListener('click',load);$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load();
</script></body></html>'''
    page = page.replace(
        "S039 provisional Top10 attention",
        "S039 provisional Top30/50/100 Head mean attention",
    )
    page = page.replace(
        "固定随机抽取已完成 runs；F00/K00 object query，704×1280，S039 provisional Top30/50/100 Head mean attention。样例不覆盖旧 6-case 稳定排名。",
        "704×1280 / 49 帧 / 13 个 latent 时刻。Top heads 只由 S039 positive-conditional 的 Q00→未来 K PCK 排名选出；页面同时区分固定 Q00 token 集与冻结轨迹形成的 Q00–Q12 时空 tube。",
    )
    page = page.replace('<label>S039 Head <select id="rank"></select></label>', "")
    page = page.replace(
        '<section class="media"><div id="media" class="media-grid"></div></section><section class="performance"><h2>单 Run · 30 × 24 Head Matrix</h2>',
        '<section class="media"><h2>Generated Video & Object Query</h2><div id="media" class="media-grid base-media"></div></section><section class="media"><h2>S039 Top-N Head Mean Attention Overlay</h2><div id="means" class="media-grid"></div></section><section class="ablation"><h2>完整 Attention Matrix Ablation</h2><p>R 是 F00 稀疏 object-query token，C 是其余 token。每个对象及所有对象并集分别展示 7 种 S/I/O 矩阵区域消融、literal K/V=0 对照和整-head 输出置零控制；冻结 S039 Top30/50/100 heads，应用全部 40 个去噪步及两个 CFG 分支。</p><details class="ablation-definitions" open><summary>消融矩阵定义（M1–M7、C1–C3）</summary><p class="definition-formula"><code>A = softmax(QK<sup>T</sup>/√d)，Y = AV</code></p><div class="definition-symbols"><span><strong>R</strong>：F00 稀疏 query points 映射并去重后的 latent tokens；不是完整 object mask，也不是跨时间 object tube</span><span><strong>C</strong>：当前 attention 序列中除 R 外的全部 tokens</span><span><strong>S=A[R,R]</strong>：selected token 内部连接</span><span><strong>I=A[R,C]</strong>：C K/V → R Query</span><span><strong>O=A[C,R]</strong>：R K/V → C Query</span></div><div class="definition-table"><table><thead><tr><th>ID</th><th>实现名</th><th>置零矩阵块</th><th>精确计算含义</th><th>理论诊断目标</th><th>可能观察（非结果保证）</th></tr></thead><tbody><tr><td>M1</td><td><code>self_only</code></td><td>S</td><td>R Query 不读取 R K/V</td><td>稀疏对象 token 的内部自支持</td><td>局部身份或形状维持变弱</td></tr><tr><td>M2</td><td><code>incoming_only</code></td><td>I</td><td>R Query 不读取 C K/V</td><td>外部场景向选中 Query 的输入</td><td>对环境、其他对象或运动背景响应变弱</td></tr><tr><td>M3</td><td><code>outgoing_only</code></td><td>O</td><td>C Query 不读取 R K/V</td><td>选中 token 向其他 token 的输出</td><td>其他区域受该对象影响减弱</td></tr><tr><td>M4</td><td><code>query_row</code></td><td>S+I</td><td><code>A[R,:]=0</code>，所以 <code>Y[R]=0</code></td><td>删除选中 Query 在该 head 的全部更新</td><td>选中位置的 head 信息通路消失</td></tr><tr><td>M5</td><td><code>key_value_column</code></td><td>S+O</td><td><code>A[:,R]=0</code>，不重新归一化</td><td>删除选中 token 的全部 Value 贡献</td><td>全局不再接收该稀疏对象信息</td></tr><tr><td>M6</td><td><code>cross_boundary</code></td><td>I+O</td><td>双向跨边界连接置零，保留 <code>A[R,R]</code></td><td>隔离 R 与 C，同时保留内部连接</td><td>对象内部可能保持但交互减弱</td></tr><tr><td>M7</td><td><code>row_and_column</code></td><td>S+I+O</td><td>R 不读取任何 token，C 也不读取 R</td><td>删除所有涉及 R 的连接</td><td>比单行或单列更强的联合效应</td></tr></tbody></table></div><h3>必须分开展示的算子对照</h3><div class="definition-table"><table><thead><tr><th>ID</th><th>实现名</th><th>实际操作</th><th>与 M5 等价</th><th>含义</th></tr></thead><tbody><tr><td>C1</td><td><code>literal_kv_zero</code></td><td>选中 head 上令 <code>K<sub>R</sub>=V<sub>R</sub>=0</code>，重新计算 attention</td><td>否</td><td>对应列仍进入 softmax；logits 变为 0，并占用概率质量</td></tr><tr><td>C2</td><td><code>qk_logits_zero</code></td><td>选中 head 的全部 token 上令 <code>q<sub>h</sub>=0</code>，重新计算 softmax</td><td>否</td><td><code>softmax(0)=1/N</code>，每行输出相同的 <code>mean(V<sub>h</sub>)</code>，不是零输出</td></tr><tr><td>C3</td><td><code>full_head_output</code></td><td>令选中 head 的整个 <code>Y<sub>h</sub>=A<sub>h</sub>V<sub>h</sub>=0</code></td><td>否</td><td>删除整个 head 输出；与 QK logits 置零不同</td></tr><tr><td>Baseline</td><td>无</td><td>不干预</td><td>—</td><td>同 seed 基线视频</td></tr></tbody></table></div><p class="definition-warning"><strong>关键区别：</strong>M1–M7 是 post-softmax attention entries 置零且不重新归一化。M5 与保持 K 不变、仅令 <code>V<sub>R</sub>=0</code> 严格等价；C1 同时修改 K 和 V，因此会改变 softmax。Top heads 来自 S039 positive-conditional 的冻结 provisional PCK ranking，但干预覆盖 S000–S039 全部 40 步及 conditional/unconditional 两个 CFG 分支。</p></details><div id="ablation"></div></section><details class="performance"><summary>单 Run · 30 × 24 Head Matrix（点击展开）</summary>',
    )
    page = page.replace(
        "<h2>完整 Attention Matrix Ablation</h2><p>R 是 F00 稀疏 object-query token，C 是其余 token。每个对象及所有对象并集分别展示 7 种 S/I/O 矩阵区域消融、literal K/V=0 对照和整-head 输出置零控制；冻结 S039 Top30/50/100 heads，应用全部 40 个去噪步及两个 CFG 分支。</p>",
        "<h2>Object Query Attention 因果消融</h2><p>先选定 token 集合 R，再令 C=N\\R。M1–M7/C1 的计算公式固定；固定 Q00 与全时序 Tube 的区别仅在 R 的构造。视频/latent 时间 t=0…12 决定 R 覆盖哪些位置，去噪时间 s=0…39 决定何时应用干预；当前干预覆盖全部 40 步及 conditional/unconditional 两个 CFG 分支。</p><p><b>VBench 读法：</b>每项先显示消融视频绝对分数，再显示 <code>Δ = 消融分数 − 同 case、同 seed 未扰动分数</code>。7 项均为越高越好；绿色正值表示该 VBench 维度上升，红色负值表示下降。Dynamic 上升只表示运动幅度判定增加，不等价于物理正确性提高。</p>",
    )
    page = page.replace(
        "消融矩阵定义（M1–M7、C1–C3）",
        "共同算子定义：先确定 R，再解释 M1–M7 / C1–C3",
    )
    page = page.replace(
        "<strong>R</strong>：F00 稀疏 query points 映射并去重后的 latent tokens；不是完整 object mask，也不是跨时间 object tube",
        "<strong>R</strong>：协议选中的 token 集；固定版为 R_fixed（仅 t=0），Tube pilot 为 R_tube（冻结轨迹在 t=0…12 的并集）",
    )
    page = page.replace(
        "<strong>C</strong>：当前 attention 序列中除 R 外的全部 tokens",
        "<strong>C=N\\R</strong>：完整 self-attention 序列中不属于当前 R 的全部 tokens；左右两侧的 C 也随 R 改变",
    )
    page = page.replace(
        "<strong>S=A[R,R]</strong>：selected token 内部连接",
        "<strong>S=A[R,R]</strong>：R 内部读取；当 R=R_tube 时同时包含帧内连接和跨帧连接",
    )
    page = page.replace(
        "<strong>关键区别：</strong>M1–M7 是 post-softmax attention entries 置零且不重新归一化。M5 与保持 K 不变、仅令 <code>V<sub>R</sub>=0</code> 严格等价；C1 同时修改 K 和 V，因此会改变 softmax。Top heads 来自 S039 positive-conditional 的冻结 provisional PCK ranking，但干预覆盖 S000–S039 全部 40 步及 conditional/unconditional 两个 CFG 分支。",
        "<strong>阅读顺序：</strong>先确认本列使用 R_fixed 还是 R_tube，再读被置零的 S/I/O 分块，最后确认 softmax 是否重算。M1–M7 是 post-softmax A@V 分块置零且不重归一化；M5 严格等价于保持 K 不变、仅令 V_R=0。C1 同时修改 K、V 并重算 softmax。Top heads 只由 S039 positive-conditional 的 Q00→未来 K PCK 排名选出；把它们用于 R_tube 是新的因果干预，不是逐 query 帧 tracking accuracy 的证明。",
    )
    page = page.replace(
        "<th>置零矩阵块</th><th>精确计算含义</th><th>理论诊断目标</th>",
        "<th>置零矩阵块</th><th>被切断的信息流</th><th>精确计算含义</th><th>理论诊断目标（这个实验在问什么）</th>",
    )
    definition_rows = (
        (
            "<tr><td>M1</td><td><code>self_only</code></td><td>S</td><td>R Query 不读取 R K/V</td><td>稀疏对象 token 的内部自支持</td><td>局部身份或形状维持变弱</td></tr>",
            "<tr><td>M1</td><td><code>self_only</code></td><td>S</td><td><code>R K/V ──X──&gt; R Query</code></td><td>令 <code>A[R,R]=0</code>；R Query 仍可读取 C，但不能读取 R 自身的 Value</td><td>R 内部 Value 是否为 R Query 提供维持局部身份、形状和内部一致性的自支持</td><td>局部身份或形状维持可能变弱</td></tr>",
        ),
        (
            "<tr><td>M2</td><td><code>incoming_only</code></td><td>I</td><td>R Query 不读取 C K/V</td><td>外部场景向选中 Query 的输入</td><td>对环境、其他对象或运动背景响应变弱</td></tr>",
            "<tr><td>M2</td><td><code>incoming_only</code></td><td>I</td><td><code>C K/V ──X──&gt; R Query</code></td><td>令 <code>A[R,C]=0</code>；R Query 只保留对 R 的读取</td><td>环境、背景和其他对象的 Value 是否向 R Query 输入交互或运动上下文</td><td>R 对环境、其他对象或运动背景的响应可能变弱</td></tr>",
        ),
        (
            "<tr><td>M3</td><td><code>outgoing_only</code></td><td>O</td><td>C Query 不读取 R K/V</td><td>选中 token 向其他 token 的输出</td><td>其他区域受该对象影响减弱</td></tr>",
            "<tr><td>M3</td><td><code>outgoing_only</code></td><td>O</td><td><code>R K/V ──X──&gt; C Query</code></td><td>令 <code>A[C,R]=0</code>；R Query 自身仍可读取 R 与 C</td><td>R 的 Value 是否作为信息源向背景、其他对象和其他位置广播影响</td><td>其他区域受该对象影响可能减弱</td></tr>",
        ),
        (
            "<tr><td>M4</td><td><code>query_row</code></td><td>S+I</td><td><code>A[R,:]=0</code>，所以 <code>Y[R]=0</code></td><td>删除选中 Query 在该 head 的全部更新</td><td>选中位置的 head 信息通路消失</td></tr>",
            "<tr><td>M4</td><td><code>query_row</code></td><td>S+I</td><td><code>全部 K/V ──X──&gt; R Query</code></td><td>令 <code>A[R,:]=0</code>，因此该 head 的 <code>Y[R]=0</code>；R 的 K/V 仍可被 C Query 读取</td><td>该 head 写回 R 位置的完整 Query 更新是否必要；只删除接收端，不删除 R 作为发送端</td><td>R 位置来自该 head 的更新完全消失</td></tr>",
        ),
        (
            "<tr><td>M5</td><td><code>key_value_column</code></td><td>S+O</td><td><code>A[:,R]=0</code>，不重新归一化</td><td>删除选中 token 的全部 Value 贡献</td><td>全局不再接收该稀疏对象信息</td></tr>",
            "<tr><td>M5</td><td><code>key_value_column</code></td><td>S+O</td><td><code>R Value ──X──&gt; 全部 Query</code></td><td>保持 softmax 权重 A 不变，令 <code>A[:,R]=0</code> 且不重新归一化；严格等价于 K 不变、仅令 <code>V_R=0</code></td><td>R 的 Value 是否是供全局 Query 读取的信息源；R Query 仍能读取 C，只删除发送端贡献</td><td>全局不再接收该稀疏对象的 Value 信息</td></tr>",
        ),
        (
            "<tr><td>M6</td><td><code>cross_boundary</code></td><td>I+O</td><td>双向跨边界连接置零，保留 <code>A[R,R]</code></td><td>隔离 R 与 C，同时保留内部连接</td><td>对象内部可能保持但交互减弱</td></tr>",
            "<tr><td>M6</td><td><code>cross_boundary</code></td><td>I+O</td><td><code>C K/V ──X──&gt; R Query</code><br><code>R K/V ──X──&gt; C Query</code></td><td>令 <code>A[R,C]=A[C,R]=0</code>；保留 R→R 和 C→C 的内部读取</td><td>把 R 与 C 隔离后，内部自支持能否保留，以及跨对象/场景双向交互是否是变化来源</td><td>R 内部可能保持，但与环境及其他对象的交互可能减弱</td></tr>",
        ),
        (
            "<tr><td>M7</td><td><code>row_and_column</code></td><td>S+I+O</td><td>R 不读取任何 token，C 也不读取 R</td><td>删除所有涉及 R 的连接</td><td>比单行或单列更强的联合效应</td></tr>",
            "<tr><td>M7</td><td><code>row_and_column</code></td><td>S+I+O</td><td><code>全部 K/V ──X──&gt; R Query</code><br><code>R K/V ──X──&gt; C Query</code></td><td>令 <code>A[R,:]=0</code> 且 <code>A[C,R]=0</code>；R 的 head 更新为零，也不能向 C 提供 Value，C→C 保持</td><td>同时删除 R 的接收端和发送端，诊断该 head 中所有涉及 R 的通信总效应</td><td>通常比仅行消融或仅列消融产生更强的联合效应</td></tr>",
        ),
    )
    for old_row, new_row in definition_rows:
        if old_row not in page:
            raise RuntimeError("PhysicIQ67 ablation definition row changed")
        page = page.replace(old_row, new_row)
    page = page.replace(
        '<div id="matrix" class="performance-heat"></div></div></section><section class="ranking"><h2>捕获时的 provisional S039 Top10</h2>',
        '<div id="matrix" class="performance-heat"></div></div></details><section class="ranking"><h2>捕获时的 provisional S039 Top100</h2>',
    )
    page = page.replace(
        ".media-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px}",
        ".media-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px}.base-media{grid-template-columns:repeat(2,minmax(0,1fr))}.ablation{overflow:hidden}.ablation-baseline{max-width:520px;margin:12px 0 18px}.ablation-table{overflow-x:auto}.ablation-row{display:grid;grid-template-columns:minmax(190px,.7fr) repeat(3,minmax(300px,1fr));gap:10px;min-width:1150px;padding:10px 0;border-top:1px solid var(--line)}.ablation-label{display:flex;flex-direction:column;justify-content:center;padding:12px;background:#17443a;color:#fff}.ablation-label small{margin-top:7px;opacity:.8}.ablation-row video{aspect-ratio:1280/704}.ablation-definitions{margin:14px 0 18px;padding:12px;border:1px solid var(--line);border-radius:12px;background:#f8f1e5}.definition-formula{font-size:18px}.definition-symbols{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:10px 0}.definition-symbols span{padding:9px 11px;border-left:5px solid var(--gold);background:#fff}.definition-table{overflow-x:auto;margin:10px 0 15px;border:1px solid var(--line);background:#fff}.definition-table table{min-width:1050px}.definition-table td{text-align:left;vertical-align:top;line-height:1.45}.definition-table td:first-child,.definition-table td:nth-child(2){white-space:nowrap}.definition-warning{padding:11px 13px;border-left:6px solid var(--gold);background:#fff;line-height:1.55}.requested-top100{overflow:visible}.requested-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.requested-grid figure{min-width:0}.requested-grid figcaption{overflow-wrap:anywhere}.requested-note{margin:8px 0 14px;padding:10px 12px;background:#e8dfcc;border-left:6px solid var(--gold)}summary{cursor:pointer;font-size:20px;font-weight:900}details[open] summary{margin-bottom:12px}@media(max-width:900px){.requested-grid,.definition-symbols{grid-template-columns:1fr}}",
    )
    page = page.replace(
        ".requested-grid figcaption{overflow-wrap:anywhere}",
        ".requested-grid figcaption{overflow-wrap:anywhere;line-height:1.45}.caption-matrix,.caption-flow,.caption-exact,.caption-protocol{display:block;margin-top:5px}.caption-matrix{font:12px ui-monospace,monospace;color:var(--deep)}.caption-flow,.caption-exact{font-weight:500}.caption-protocol{font:11px ui-monospace,monospace;color:#6b665c}",
    )
    page = page.replace(
        "</style>",
        ".tube-compare{display:grid;gap:12px}.tube-compare-row{display:grid;grid-template-columns:minmax(210px,.65fr) repeat(2,minmax(0,1fr));gap:10px;align-items:stretch}.tube-compare-row>div,.tube-column-head{padding:12px;background:#17443a;color:#fff;border-radius:10px}.tube-compare-row figure{min-width:0}.tube-compare-row video{width:100%;aspect-ratio:1280/704;background:#111}.tube-column-head{text-align:center;font-weight:900;background:#a35f1d}.tube-row-label{display:flex;flex-direction:column;justify-content:center}.tube-row-label small{margin-top:7px;opacity:.82;line-height:1.4}.tube-compare .pending{height:100%;min-height:180px}.tube-baseline{max-width:640px;margin-bottom:12px}.tube-protocol-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:10px 0}.tube-protocol-grid span{padding:10px 12px;background:#fff;border-left:5px solid var(--deep);line-height:1.5}.tube-dose-warning{padding:10px 12px;background:#fff1d8;border-left:6px solid #b95031;line-height:1.55}.tube-scope-title{margin:22px 0 2px;padding:10px 12px;background:#d8cab4;border-left:7px solid var(--gold)}@media(max-width:900px){.tube-compare-row,.tube-protocol-grid{grid-template-columns:1fr}.tube-column-head:first-child{display:none}}</style>",
        1,
    )
    page = page.replace(
        ".note,.media,.performance,.ranking{",
        ".note,.media,.ablation,.performance,.ranking{",
    )
    page = page.replace("u.searchParams.set('rank',$('rank').value);", "")
    page = page.replace(
        "const s=current(),region=$('region').value,rank=Number($('rank').value||0),entry=data.entries[rank],base=",
        "const s=current(),region=$('region').value,base=",
    )
    old_media = """const video=s.video_ready?`<video controls muted playsinline preload="metadata" src="${api}/video?${base}"></video>`:'<div class="pending">Generated video unavailable</div>',query=s.query_visual_ready?`<img src="${api}/query-image?case=${encodeURIComponent(s.case)}&region=${encodeURIComponent(region)}&v=1">`:'<div class="pending">Query visualization unavailable</div>',heat=s.attention_ready&&entry?`<img src="${api}/heatmap?${base}&rank=${rank}&region=${encodeURIComponent(region)}&v=${Date.now()}">`:'<div class="pending">Attention capture 待生成<br>视频、SAM2 query 和 PCK 矩阵已可查看</div>';$('media').innerHTML=`<figure>${video}<figcaption>Legacy generated video · 40 steps / 49 frames</figcaption></figure><figure>${query}<figcaption>SAM2 mask + 8 object query points · F00/K00</figcaption></figure><figure>${heat}<figcaption>${entry?`#${rank+1} · S39 L${entry.block} H${entry.head} · ${region}`:'S039 attention'}</figcaption></figure>`;renderMatrix();"""
    new_media = """const video=s.video_ready?`<video controls muted playsinline preload="metadata" src="${api}/video?${base}"></video>`:'<div class="pending">Generated video unavailable</div>',query=s.query_visual_ready?`<img src="${api}/query-image?case=${encodeURIComponent(s.case)}&region=${encodeURIComponent(region)}&v=1">`:'<div class="pending">Query visualization unavailable</div>',isTubePilot=Array.isArray(s.temporal_tube_attention_matrix_ablations);$('media').innerHTML=isTubePilot?`<figure>${query}<figcaption>SAM2 mask + 8 object query points · F00/K00</figcaption></figure>`:`<figure>${video}<figcaption>Legacy generated video · 40 steps / 49 frames</figcaption></figure><figure>${query}<figcaption>SAM2 mask + 8 object query points · F00/K00</figcaption></figure>`;$('means').innerHTML=(isTubePilot?[100]:[30,50,100]).map(n=>`<figure>${s.attention_ready?`<img loading="lazy" src="${api}/mean-heatmap?${base}&top_n=${n}&region=${encodeURIComponent(region)}&v=${Date.now()}">`:'<div class="pending">Top'+n+' capture 待生成</div>'}<figcaption>S039 Top${n} Head Mean · ${region}</figcaption></figure>`).join('');renderAblations(s,base);renderMatrix();"""
    if old_media not in page:
        raise RuntimeError("PhysicIQ67 visual page media template changed")
    page = page.replace(old_media, new_media)
    page = page.replace(
        "query-image?case=${encodeURIComponent(s.case)}&region=",
        "query-image?${base}&region=",
    )
    matrix_prefix = """function renderMatrix(){const s=current(),view=$('view').value,metric=$('metric').value,rows=s.matrices?.[view]||[],vals=rows.map(x=>x[metric]).filter(Number.isFinite),lo=Math.min(...vals),hi=Math.max(...vals),map=new Map(rows.map(x=>[`${x.block}-${x.head}`,x])),box=$('matrix');"""
    guarded_matrix_prefix = """function renderMatrix(){const s=current(),view=$('view').value,metric=$('metric').value,rows=s.matrices?.[view]||[],vals=rows.map(x=>x[metric]).filter(Number.isFinite),box=$('matrix');if(!vals.length){box.innerHTML='<div class="pending">该 case 暂无单-run head matrix</div>';$('matrixMeta').textContent='matrix unavailable';return}const lo=Math.min(...vals),hi=Math.max(...vals),map=new Map(rows.map(x=>[`${x.block}-${x.head}`,x]));"""
    if matrix_prefix not in page:
        raise RuntimeError("PhysicIQ67 visual page matrix template changed")
    page = page.replace(matrix_prefix, guarded_matrix_prefix)
    page = page.replace(
        "function render(){",
        """function renderAblations(s,base){const records=s.attention_matrix_ablations||[],region=$('region').value,masks=[['self_only','S · A[R,R]','selected queries 不读取 selected K/V'],['incoming_only','I · A[R,C]','只切断 unselected K/V → selected queries'],['outgoing_only','O · A[C,R]','只切断 selected K/V → unselected queries'],['query_row','S+I · A[R,:]','selected query rows 全部置零'],['key_value_column','S+O · A[:,R]','selected K/V columns 贡献置零，不重新归一化'],['cross_boundary','I+O · Cross boundary','切断双向跨边界连接，保留 A[R,R]'],['row_and_column','S+I+O · Row + column','selected rows 与 columns 联合消融'],['literal_kv_zero','Literal K_R=V_R=0','重新计算 softmax；与 post-softmax column zero 不等价']],targets=[['single_object',region,`Selected object · ${region}`],['all_objects','', 'All object-query tokens union']],rows=targets.flatMap(([target_scope,targetRegion,targetLabel])=>masks.map(([mask_mode,label,detail])=>({target_scope,mask_mode,region:targetRegion,label:`${targetLabel} · ${label}`,detail})));rows.push({target_scope:'all_tokens',mask_mode:'full_head_output',region:'',label:'Control · Full Head Output Zero',detail:'Y_h=A_hV_h 整个 head 输出置零；不是 QK logits=0'});const card=(spec,n)=>{const r=records.find(x=>x.target_scope===spec.target_scope&&x.mask_mode===spec.mask_mode&&String(x.region||'')===spec.region&&Number(x.top_n)===n),suffix=spec.region?`&region=${encodeURIComponent(spec.region)}`:'',src=`${api}/ablation-video?${base}&target_scope=${encodeURIComponent(spec.target_scope)}&mask_mode=${encodeURIComponent(spec.mask_mode)}&top_n=${n}${suffix}`,media=r?.ready?`<video controls muted playsinline preload="metadata" src="${src}"></video>`:`<div class="pending">${r?.error?'生成失败':'待生成'}</div>`,protocol=spec.mask_mode==='literal_kv_zero'?'literal K/V zero + softmax recompute':spec.mask_mode==='full_head_output'?'full A@V head output zero':'post-softmax block zero · no renorm';return `<figure>${media}<figcaption>Top${n} · ${protocol}</figcaption></figure>`},baseline=s.video_ready?`<video controls muted playsinline preload="metadata" src="${api}/video?${base}"></video>`:'<div class="pending">Baseline unavailable</div>';$('ablation').innerHTML=`<div class="ablation-baseline"><figure>${baseline}<figcaption>Baseline · No Attention Intervention</figcaption></figure></div><div class="ablation-table">${rows.map(spec=>`<div class="ablation-row"><div class="ablation-label"><strong>${e(spec.label)}</strong><small>${e(spec.detail)}</small></div>${[30,50,100].map(n=>card(spec,n)).join('')}</div>`).join('')}</div>`}function render(){""",
    )
    page = page.replace(
        "masks=[['self_only','S · A[R,R]','selected queries 不读取 selected K/V'],['incoming_only','I · A[R,C]','只切断 unselected K/V → selected queries'],['outgoing_only','O · A[C,R]','只切断 selected K/V → unselected queries'],['query_row','S+I · A[R,:]','selected query rows 全部置零'],['key_value_column','S+O · A[:,R]','selected K/V columns 贡献置零，不重新归一化'],['cross_boundary','I+O · Cross boundary','切断双向跨边界连接，保留 A[R,R]'],['row_and_column','S+I+O · Row + column','selected rows 与 columns 联合消融'],['literal_kv_zero','Literal K_R=V_R=0','重新计算 softmax；与 post-softmax column zero 不等价']]",
        "masks=[['self_only','M1 · S=A[R,R]=0','R Query 不读取 R K/V；仍可读取 C'],['incoming_only','M2 · I=A[R,C]=0','R Query 不读取 C K/V；只保留对 R 的读取'],['outgoing_only','M3 · O=A[C,R]=0','C Query 不读取 R K/V；R Query 读取保持'],['query_row','M4 · A[R,:]=0','R Query 整行 attention 置零，因此 Y[R]=0'],['key_value_column','M5 · A[:,R]=0','softmax A 保持不变，删除 R 的全部 Value 贡献且不重新归一化'],['cross_boundary','M6 · A[R,C]=A[C,R]=0','双向跨边界连接置零，保留 R→R 与 C→C'],['row_and_column','M7 · A[R,:]=0 且 A[C,R]=0','R 的接收端更新为零，同时 R 不再向 C 提供 Value'],['literal_kv_zero','C1 · K_R=V_R=0','修改 K_R 与 V_R 后重新计算 softmax；R 列 logits=0 且仍占概率质量']]",
    )
    page = page.replace(
        "rows.push({target_scope:'all_tokens',mask_mode:'full_head_output',region:'',label:'Control · Full Head Output Zero',detail:'Y_h=A_hV_h 整个 head 输出置零；不是 QK logits=0'});",
        "rows.push({target_scope:'all_tokens',mask_mode:'qk_logits_zero',region:'',label:'C2 · Control · QK Logits Zero',detail:'令 Q_h=0，所以完整 QK^T=0；重新计算 softmax 后 A_h=1/N，Y_h=mean(V_h)'});rows.push({target_scope:'all_tokens',mask_mode:'full_head_output',region:'',label:'C3 · Control · Full Head Output Zero',detail:'令整个 Y_h=A_hV_h=0；直接删除 head 输出，与 QK logits=0 不同'});",
    )
    page = page.replace(
        "protocol=spec.mask_mode==='literal_kv_zero'?'literal K/V zero + softmax recompute':spec.mask_mode==='full_head_output'?'full A@V head output zero':'post-softmax block zero · no renorm';",
        "protocol=spec.mask_mode==='literal_kv_zero'?'literal K/V zero + softmax recompute':spec.mask_mode==='qk_logits_zero'?'QK logits zero · uniform softmax · mean(V)':spec.mask_mode==='full_head_output'?'full A@V head output zero':'post-softmax block zero · no renorm';",
    )
    page = page.replace(
        "rows.push({target_scope:'all_tokens',mask_mode:'full_head_output',region:'',label:'C3 · Control · Full Head Output Zero',detail:'令整个 Y_h=A_hV_h=0；直接删除 head 输出，与 QK logits=0 不同'});const card=",
        "rows.push({target_scope:'all_tokens',mask_mode:'full_head_output',region:'',label:'C3 · Control · Full Head Output Zero',detail:'令整个 Y_h=A_hV_h=0；直接删除 head 输出，与 QK logits=0 不同'});if(Array.isArray(s.temporal_tube_attention_matrix_ablations))return renderRequestedTop100(s);const card=",
    )
    page = page.replace(
        "return `<figure>${media}<figcaption>Top${n} · ${protocol}</figcaption></figure>`",
        "return `<figure>${media}<figcaption><strong>${e(spec.label)} · Top${n}</strong><span class=\"caption-exact\"><b>精确计算：</b>${e(spec.detail)}</span><span class=\"caption-protocol\">${e(protocol)}</span></figcaption></figure>`",
    )
    page = page.replace(
        "function render(){",
        """function renderRequestedTop100(s){const base=`case=${encodeURIComponent(s.case)}&seed=${s.seed}`,records=(s.attention_matrix_ablations||[]).filter(r=>Number(r.top_n)===100),ready=records.filter(r=>r.ready),definitions={self_only:{id:'M1',matrix:'S=A[R,R]=0',flow:'R K/V ──X──> R Query',exact:'R Query 不读取 R K/V；仍可读取 C'},incoming_only:{id:'M2',matrix:'I=A[R,C]=0',flow:'C K/V ──X──> R Query',exact:'R Query 不读取 C K/V；只保留对 R 的读取'},outgoing_only:{id:'M3',matrix:'O=A[C,R]=0',flow:'R K/V ──X──> C Query',exact:'C Query 不读取 R K/V；R Query 的读取保持不变'},query_row:{id:'M4',matrix:'A[R,:]=0',flow:'全部 K/V ──X──> R Query',exact:'R Query 整行 attention 置零，因此该 head 的 Y[R]=0；R K/V 仍可被 C Query 读取'},key_value_column:{id:'M5',matrix:'A[:,R]=0 · no renorm',flow:'R Value ──X──> 全部 Query',exact:'保持 softmax A 不变，删除 R 的全部 Value 贡献且不重新归一化；等价于 K 不变、仅 V_R=0'},cross_boundary:{id:'M6',matrix:'A[R,C]=A[C,R]=0',flow:'C K/V ──X──> R Query；R K/V ──X──> C Query',exact:'双向跨边界连接置零，保留 R→R 与 C→C 的内部读取'},row_and_column:{id:'M7',matrix:'A[R,:]=0 且 A[C,R]=0',flow:'全部 K/V ──X──> R Query；R K/V ──X──> C Query',exact:'R 的接收端 head 更新为零，同时 R 不再向 C 提供 Value；C→C 保持'},literal_kv_zero:{id:'C1',matrix:'K_R=V_R=0 · recompute softmax',flow:'R Value 贡献为零，但 R 列仍参与路由概率分配',exact:'同时修改 K_R 与 V_R 后重新计算 attention；R 列 logits 变为 0，并继续占用 softmax 概率质量'},qk_logits_zero:{id:'C2',matrix:'Q_h=0 ⇒ QK^T=0 ⇒ A_h=1/N',flow:'所有 Query 均匀读取全部 Value',exact:'重新计算 softmax 后每一行均匀，Y_h=mean(V_h)，不是零输出'},full_head_output:{id:'C3',matrix:'Y_h=A_hV_h=0',flow:'该 head ──X──> 后续输出',exact:'直接令选中 head 的整个输出为零；与 QK logits=0 的均匀注意力不同'}},targetLabel=r=>r.target_scope==='single_object'?`Selected object · ${r.region}`:r.target_scope==='all_objects'?'All object-query tokens union':'All tokens',card=r=>{const definition=definitions[r.mask_mode]||{id:r.mask_mode,matrix:r.mask_mode,flow:'未定义',exact:'未定义'},suffix=r.region?`&region=${encodeURIComponent(r.region)}`:'',src=`${api}/ablation-video?${base}&target_scope=${encodeURIComponent(r.target_scope)}&mask_mode=${encodeURIComponent(r.mask_mode)}&top_n=100${suffix}`;return `<figure><video controls muted playsinline preload="metadata" src="${src}"></video><figcaption><strong>${e(definition.id)} · ${e(r.mask_mode)} · ${e(targetLabel(r))} · Top100</strong><span class="caption-matrix">${e(definition.matrix)}</span><span class="caption-flow">信息流：${e(definition.flow)}</span><span class="caption-exact">精确计算：${e(definition.exact)}</span></figcaption></figure>`},baseline=s.video_ready?`<figure><video controls muted playsinline preload="metadata" src="${api}/video?${base}"></video><figcaption><strong>Baseline · No intervention</strong><span class="caption-flow">信息流：全部保留</span><span class="caption-exact">精确计算：保持原始 Q/K/V、softmax attention 与所有 head 输出不变</span></figcaption></figure>`:'';$('ablation').innerHTML=`<div class="requested-note">${e(s.case)} · seed=47326 · 原视频 + 已生成 Top100 消融 ${ready.length}/${records.length}；未生成项不占位，Top30/Top50 不显示</div><div class="requested-grid requested-top100">${baseline}${ready.map(card).join('')}</div>`}function render(){""",
        1,
    )
    page = page.replace(
        "function renderRequestedTop100(s){",
        """function renderTemporalTubeComparison(s){const base=`case=${encodeURIComponent(s.case)}&seed=${s.seed}`,fixed=(s.attention_matrix_ablations||[]).filter(r=>Number(r.top_n)===100),tube=s.temporal_tube_attention_matrix_ablations||[],defs={self_only:{id:'M1',matrix:'S=A[R,R]=0',flow:'R K/V ──X──> R Query',exact:'R Query 不读取 R K/V；仍可读取 C'},incoming_only:{id:'M2',matrix:'I=A[R,C]=0',flow:'C K/V ──X──> R Query',exact:'R Query 不读取 C K/V；只保留对 R 的读取'},outgoing_only:{id:'M3',matrix:'O=A[C,R]=0',flow:'R K/V ──X──> C Query',exact:'C Query 不读取 R K/V；R Query 的读取保持'},query_row:{id:'M4',matrix:'A[R,:]=0',flow:'全部 K/V ──X──> R Query',exact:'该 head 的 Y[R]=0；R K/V 仍可被 C Query 读取'},key_value_column:{id:'M5',matrix:'A[:,R]=0 · no renorm',flow:'R Value ──X──> 全部 Query',exact:'保持 A 不变并删除 R 的全部 Value 贡献；等价于仅 V_R=0'},cross_boundary:{id:'M6',matrix:'A[R,C]=A[C,R]=0',flow:'C K/V ──X──> R Query；R K/V ──X──> C Query',exact:'切断双向跨边界读取，保留 R→R 与 C→C'},row_and_column:{id:'M7',matrix:'A[R,:]=0 且 A[C,R]=0',flow:'全部 K/V ──X──> R Query；R K/V ──X──> C Query',exact:'R 接收端更新为零，同时 R 不再向 C 提供 Value'},literal_kv_zero:{id:'C1',matrix:'K_R=V_R=0 · recompute softmax',flow:'R Value 为零；R 列仍参与 softmax 路由',exact:'K_R、V_R 同时置零后重算 attention；R 列 logits=0 且仍占概率质量'}},modes=['self_only','incoming_only','outgoing_only','query_row','key_value_column','cross_boundary','row_and_column','literal_kv_zero'],objectTargets=[...new Set(tube.filter(r=>r.target_scope==='single_object').map(r=>String(r.region)))].map(region=>['single_object',region,`Selected object · ${region}`]),targets=[...objectTargets,['all_objects','', 'All objects union']],specs=targets.flatMap(([target_scope,targetRegion,targetLabel])=>modes.map(mask_mode=>({target_scope,region:targetRegion,targetLabel,mask_mode}))),find=(rows,x)=>rows.find(r=>r.target_scope===x.target_scope&&r.mask_mode===x.mask_mode&&String(r.region||'')===x.region),src=(kind,x)=>`${api}/${kind==='fixed'?'ablation-video':'temporal-tube-ablation-video'}?${base}&target_scope=${encodeURIComponent(x.target_scope)}&mask_mode=${encodeURIComponent(x.mask_mode)}&top_n=100${x.region?`&region=${encodeURIComponent(x.region)}`:''}`,card=(kind,x,r)=>{const d=defs[x.mask_mode],label=kind==='fixed'?'左 · Fixed Query (Q00 only)':'右 · All-time Query Tube (Q00–Q12)',rdef=kind==='fixed'?'R_fixed：F00 稀疏点映射出的唯一 t=0 tokens':'R_tube：baseline CoTracker 轨迹在 F00,F04,…,F48 映射出的 13 帧 token 并集',media=r?.ready?`<video controls muted playsinline preload="metadata" src="${src(kind,x)}"></video>`:`<div class="pending">${r?.error?'生成失败':'待生成'}</div>`;return `<figure>${media}<figcaption><strong>${e(label)}</strong><span class="caption-matrix">${e(rdef)}</span><span class="caption-exact"><b>算子：</b>${e(d.exact)}</span></figcaption></figure>`},rows=specs.map(x=>{const d=defs[x.mask_mode];return `<div class="tube-compare-row"><div class="tube-row-label"><strong>${e(d.id)} · ${e(x.mask_mode)}<br>${e(x.targetLabel)}</strong><small>${e(d.matrix)}</small><small>信息流：${e(d.flow)}</small></div>${card('fixed',x,find(fixed,x))}${card('tube',x,find(tube,x))}</div>`}).join(''),done=tube.filter(r=>r.ready).length,baseline=s.video_ready?`<figure class="tube-baseline"><video controls muted playsinline preload="metadata" src="${api}/video?${base}"></video><figcaption><strong>Baseline · seed=47326 · No intervention</strong></figcaption></figure>`:'';$('ablation').innerHTML=`<div class="requested-note"><b>固定 Query vs 全时序 Query Tube</b><br>唯一自变量是 R：左侧只含 F00/t=0 token；右侧使用同一 baseline 上冻结的 CoTracker pseudo-GT，将每个 object point 映射到 Q00–Q12（对应视频 F00,F04,…,F48）。两侧使用完全相同的 Top100 heads、seed、40 denoising steps、两个 CFG 分支和 M1–M7/C1 计算。页面同时展示 object_A、object_B 与 all_objects；动态进度 ${done}/${tube.length}。C2/C3 与 R 无关，因此不重复生成。</div>${baseline}<div class="tube-compare"><div class="tube-compare-row"><div class="tube-column-head">实验 ID / 精确信息流</div><div class="tube-column-head">左：Fixed Q00</div><div class="tube-column-head">右：All-time Q00–Q12</div></div>${rows}</div>`}function renderRequestedTop100(s){if(s.temporal_tube_attention_matrix_ablations)return renderTemporalTubeComparison(s);""",
        1,
    )
    page = page.replace(
        "label=kind==='fixed'?'左 · Fixed Query (Q00 only)':'右 · All-time Query Tube (Q00–Q12)',rdef=kind==='fixed'?'R_fixed：F00 稀疏点映射出的唯一 t=0 tokens':'R_tube：baseline CoTracker 轨迹在 F00,F04,…,F48 映射出的 13 帧 token 并集'",
        "isControl=x.target_scope==='all_tokens',label=isControl?`${d.id} · ${x.mask_mode} · Global control`:`${d.id} · ${x.mask_mode} · ${kind==='fixed'?'Fixed R_fixed':'Tube R_tube'}`,tokenCount=Number.isInteger(r?.selected_token_count)?r.selected_token_count:'—',perFrame=Array.isArray(r?.latent_frame_token_counts)?r.latent_frame_token_counts.join(','):'—',rdef=isControl?'All tokens · operator 与 object token 集 R 无关':kind==='fixed'?`R_fixed：仅 F00 / latent t=0；|R|=${tokenCount}`:`R_tube：冻结 baseline 轨迹在 Q00–Q12 的联合集合；|R|=${tokenCount}；逐 latent=[${perFrame}]`",
    )
    page = page.replace(
        "rows=specs.map(x=>{const d=defs[x.mask_mode];return `<div class=\"tube-compare-row\"><div class=\"tube-row-label\"><strong>${e(d.id)} · ${e(x.mask_mode)}<br>${e(x.targetLabel)}</strong><small>${e(d.matrix)}</small><small>信息流：${e(d.flow)}</small></div>${card('fixed',x,find(fixed,x))}${card('tube',x,find(tube,x))}</div>`}).join('')",
        "rows=targets.map(([targetScope,targetRegion,targetLabel])=>{const group=specs.filter(x=>x.target_scope===targetScope&&x.region===targetRegion);return `<h3 class=\"tube-scope-title\">Target：${e(targetLabel)} · 8 个 R-dependent operators</h3>`+group.map(x=>{const d=defs[x.mask_mode];return `<div class=\"tube-compare-row\"><div class=\"tube-row-label\"><strong>${e(d.id)} · ${e(x.mask_mode)}</strong><small>${e(d.matrix)}</small><small>被切断：${e(d.flow)}</small></div>${card('fixed',x,find(fixed,x))}${card('tube',x,find(tube,x))}</div>`}).join('')}).join('')",
    )
    page = page.replace(
        '<div class="requested-note"><b>固定 Query vs 全时序 Query Tube</b><br>唯一自变量是 R：左侧只含 F00/t=0 token；右侧使用同一 baseline 上冻结的 CoTracker pseudo-GT，将每个 object point 映射到 Q00–Q12（对应视频 F00,F04,…,F48）。两侧使用完全相同的 Top100 heads、seed、40 denoising steps、两个 CFG 分支和 M1–M7/C1 计算。页面同时展示 object_A、object_B 与 all_objects；动态进度 ${done}/${tube.length}。C2/C3 与 R 无关，因此不重复生成。</div>',
        '<div class="requested-note"><h3>实验问题：扩大 R 的视频时间范围后，同一算子的生成效应如何变化？</h3><p>左右两侧保持 Top100 heads、seed=47326、采样参数、40 个去噪步、conditional/unconditional 两个 CFG 分支和 M1–M7/C1 公式完全一致。</p><div class="tube-protocol-grid"><span><b>左 · R_fixed</b><br>F00 稀疏 object points 只映射到 latent t=0；不是完整 mask，也不复制到未来帧。</span><span><b>右 · R_tube</b><br>同一批点在 baseline 上由 CoTracker 冻结，映射 Q00–Q12 / F00,F04,…,F48 后取联合集合。</span><span><b>Tube 内部含义</b><br>A[R_tube,R_tube] 同时包含帧内和跨帧对象连接；右侧不是 13 次彼此独立的逐帧实验。</span><span><b>两条时间轴</b><br>t=0…12 是视频 latent 时间；s=0…39 是扩散去噪时间。右侧覆盖全部 t，两侧均覆盖全部 s。</span></div><p class="tube-dose-warning"><b>解释限制：</b>R_tube 的 token 数显著多于 R_fixed，因此右侧更强变化可能来自更大的干预剂量，不能直接证明每一帧的 PCK head 都准确。当前 Tube 消融也不等于 Q_t×K_s 的 13×13 响应/PCK 验证；轨迹标签为 CoTracker pseudo-GT，不是真实 GT。</p><p>页面展示 object_A、object_B、all_objects 三组；R-dependent 生成进度：Fixed <b>${fixedDone}/24</b> · Tube <b>${done}/${tube.length}</b>。C2/C3 不依赖 R，不重复生成 Tube，并在 Global all-token controls 行展示已有 Fixed 控制视频。</p></div>',
    )
    page = page.replace(
        '<div class="tube-column-head">实验 ID / 精确信息流</div><div class="tube-column-head">左：Fixed Q00</div><div class="tube-column-head">右：All-time Q00–Q12</div>',
        '<div class="tube-column-head">算子 ID / 被切断的信息流</div><div class="tube-column-head">左：R_fixed · 仅 latent t=0</div><div class="tube-column-head">右：R_tube · latent t=0…12 联合集合</div>',
    )
    page = page.replace(
        '<span class="caption-flow">信息流：${e(definition.flow)}</span><span class="caption-exact">精确计算：${e(definition.exact)}</span>',
        '<span class="caption-flow"><b>信息流：</b>${e(definition.flow)}</span><span class="caption-exact"><b>精确计算：</b>${e(definition.exact)}</span>',
    )
    page = page.replace(
        '<span class="caption-flow">信息流：全部保留</span><span class="caption-exact">精确计算：保持原始 Q/K/V、softmax attention 与所有 head 输出不变</span>',
        '<span class="caption-flow"><b>信息流：</b>全部保留</span><span class="caption-exact"><b>精确计算：</b>保持原始 Q/K/V、softmax attention 与所有 head 输出不变</span>',
    )
    page = page.replace(
        "options($('rank'),data.entries,(x,i)=>i,(x,i)=>`#${i+1} S39 L${String(x.block).padStart(2,'0')} H${String(x.head).padStart(2,'0')}`);if(q.get('rank'))$('rank').value=q.get('rank');",
        "",
    )
    page = page.replace(
        "for(const id of ['region','rank'])$(id).addEventListener('change',render);",
        "$('region').addEventListener('change',render);",
    )
    page = page.replace(
        " attention samples ready`;render()",
        " attention samples ready · ${data.ablation_progress.ready}/${data.ablation_progress.expected} ablation videos ready`;render()",
    )
    page = page.replace(
        "</style>",
        ".vbench-summary{margin:8px 0 0;padding:8px;border-top:1px solid #ddd2c0;background:#f8f4eb}.vbench-summary>strong{display:block;margin-bottom:6px;font:12px ui-monospace,monospace}.vbench-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}.vbench-chip{display:grid;grid-template-columns:1fr auto auto;gap:5px;padding:5px 6px;background:#fff;border:1px solid #e1d8c9;font:10px ui-monospace,monospace}.vbench-chip b{overflow:hidden;text-overflow:ellipsis}.vbench-chip em,.vbench-chip i{font-style:normal;font-variant-numeric:tabular-nums}.vbench-chip .positive{color:#08704e}.vbench-chip .negative{color:#b33b28}.vbench-chip .neutral{color:#6c665d}.vbench-pending{padding:9px;background:#f1ece2;color:#756d61;font:11px ui-monospace,monospace}@media(max-width:900px){.vbench-grid{grid-template-columns:1fr}}</style>",
        1,
    )
    page = page.replace(
        "function renderTemporalTubeComparison(s){",
        """function vbenchSummary(x,isBaseline=false){const report=x?.vbench,metrics=(report?.metrics||[]).filter(m=>Number.isFinite(m.score));if(!metrics.length)return `<div class="vbench-pending">VBench 待评测 · 0/${report?.expected||7}</div>`;const chips=metrics.map(m=>{const score=Number(m.score),delta=Number(m.delta),hasDelta=Number.isFinite(delta),deltaClass=!hasDelta||Math.abs(delta)<5e-7?'neutral':delta>0?'positive':'negative',deltaText=isBaseline?'baseline':hasDelta?`Δ ${delta>=0?'+':''}${delta.toFixed(4)}`:'Δ —';return `<span class="vbench-chip" title="${e(m.field)} · baseline ${Number.isFinite(m.baseline)?Number(m.baseline).toFixed(6):'—'}"><b>${e(m.label)}</b><em>${score.toFixed(4)}</em><i class="${deltaClass}">${deltaText}</i></span>`}).join('');return `<div class="vbench-summary"><strong>VBench ${isBaseline?'未扰动基线':'相比未扰动 Δ'} · ${report.completed}/${report.expected}</strong><div class="vbench-grid">${chips}</div></div>`}function renderTemporalTubeComparison(s){""",
        1,
    )
    page = page.replace(
        '<span class="caption-protocol">${e(protocol)}</span></figcaption></figure>',
        '<span class="caption-protocol">${e(protocol)}</span></figcaption>${vbenchSummary(r)}</figure>',
    )
    page = page.replace(
        '<div class="ablation-baseline"><figure>${baseline}<figcaption>Baseline · No Attention Intervention</figcaption></figure></div>',
        '<div class="ablation-baseline"><figure>${baseline}<figcaption>Baseline · No Attention Intervention</figcaption>${vbenchSummary(s,true)}</figure></div>',
    )
    page = page.replace(
        '<span class="caption-exact"><b>精确计算：</b>${e(definition.exact)}</span></figcaption></figure>',
        '<span class="caption-exact"><b>精确计算：</b>${e(definition.exact)}</span></figcaption>${vbenchSummary(r)}</figure>',
    )
    page = page.replace(
        '<span class="caption-exact"><b>精确计算：</b>保持原始 Q/K/V、softmax attention 与所有 head 输出不变</span></figcaption></figure>',
        '<span class="caption-exact"><b>精确计算：</b>保持原始 Q/K/V、softmax attention 与所有 head 输出不变</span></figcaption>${vbenchSummary(s,true)}</figure>',
    )
    page = page.replace(
        '<span class="caption-exact"><b>算子：</b>${e(d.exact)}</span></figcaption></figure>',
        '<span class="caption-exact"><b>算子：</b>${e(d.exact)}</span></figcaption>${vbenchSummary(r)}</figure>',
    )
    page = page.replace(
        '<figcaption><strong>Baseline · seed=47326 · No intervention</strong></figcaption></figure>',
        '<figcaption><strong>Baseline · seed=47326 · No intervention</strong></figcaption>${vbenchSummary(s,true)}</figure>',
    )
    temporal_defs_tail = "literal_kv_zero:{id:'C1',matrix:'K_R=V_R=0 · recompute softmax',flow:'R Value 为零；R 列仍参与 softmax 路由',exact:'K_R、V_R 同时置零后重算 attention；R 列 logits=0 且仍占概率质量'}},modes="
    temporal_defs_with_logic = """literal_kv_zero:{id:'C1',matrix:'K_R=V_R=0 · recompute softmax',flow:'R Value 为零；R 列仍参与 softmax 路由',exact:'K_R、V_R 同时置零后重算 attention；R 列 logits=0 且仍占概率质量'},qk_logits_zero:{id:'C2',matrix:'Q_h=0 ⇒ A_h=1/N',flow:'所有 Query 改为均匀读取全部 Value',exact:'令 Q_h=0，完整 QK^T logits 为 0；重算 softmax 后 A_h=1/N，Y_h=mean(V_h)'},full_head_output:{id:'C3',matrix:'Y_h=A_hV_h=0',flow:'该 head ──X──> 后续输出',exact:'直接令选中 head 的完整输出为 0；Q/K/V 与 softmax A 本身不变'}},logic={self_only:{calc:\"Y'_R=I V_C；Y'_C=O V_R+B V_C\",theory:'只删除 S V_R。若结果变化，说明 R 内部 Value 对 R 接收端有因果贡献；不代表对象 token 被删除。'},incoming_only:{calc:\"Y'_R=S V_R；Y'_C=O V_R+B V_C\",theory:'只删除 I V_C。R 仍能内部读取并继续向 C 输出；变化诊断 tube 外上下文进入 R 的作用。'},outgoing_only:{calc:\"Y'_R=S V_R+I V_C；Y'_C=B V_C\",theory:'只删除 O V_R。R 自身更新不变，但 C 不再接收 R Value；接近 baseline 表明该实验下 O 的可见边际效应较弱。'},query_row:{calc:\"Y'_R=0；Y'_C=O V_R+B V_C\",theory:'删除该 head 对 R 的全部接收端更新，但残差、其他 heads 和 FFN 仍保留，所以 R token 本身不会被置零。'},key_value_column:{calc:\"Y'_R=I V_C；Y'_C=B V_C\",theory:'保持 A 不变，删除所有 S/O 中的 R Value 且不重归一化；等价于只令 V_R=0，不等价于 K_R=V_R=0。'},cross_boundary:{calc:\"Y'_R=S V_R；Y'_C=B V_C\",theory:'同时删除 I 与 O，保留 R 内部和 C 内部通信；变化诊断 R–C 双向耦合的联合效应。'},row_and_column:{calc:\"Y'_R=0；Y'_C=B V_C\",theory:'删除 S、I、O，所有涉及 R 的该 head 通信均消失；残差和未选 heads 仍可补偿。'},literal_kv_zero:{calc:\"K'_R=V'_R=0；A'=softmax(QK'^T/√d)；Y'=A'V'\",theory:'R Value 为零，但 R 列 logits 变为 0 后仍进入 softmax。与 M5 的差异测到的是重新路由/概率质量效应。'},qk_logits_zero:{calc:'Q_h=0；A_h=softmax(0)=1/N；Y_h=mean(V_h)',theory:'把该 head 改为全 token 均匀平均，不是删除输出；用于区分均匀路由与真正 head 消融。'},full_head_output:{calc:'Y_h=A_hV_h=0',theory:'直接删除该 head 对残差分支的全部贡献；它不依赖 object token 集 R。'}},modes="""
    if page.count(temporal_defs_tail) != 1:
        raise RuntimeError("PhysicIQ67 temporal definitions changed")
    page = page.replace(temporal_defs_tail, temporal_defs_with_logic, 1)

    temporal_find = "find=(rows,x)=>rows.find(r=>r.target_scope===x.target_scope&&r.mask_mode===x.mask_mode&&String(r.region||'')===x.region),src="
    temporal_similarity_helpers = """find=(rows,x)=>rows.find(r=>r.target_scope===x.target_scope&&r.mask_mode===x.mask_mode&&String(r.region||'')===x.region),similarity=s.ablation_video_similarity||{},comparisons=similarity.comparisons||[],simId=(kind,x,mode=x.mask_mode)=>`${kind}:${x.target_scope}:${x.region||'all_objects'}:${mode}`,pair=(left,right)=>comparisons.find(c=>(c.left_id===left&&c.right_id===right)||(c.left_id===right&&c.right_id===left)),metricText=m=>m?`SSIM ${Number(m.ssim_mean).toFixed(4)} · PSNR ${m.psnr_db===null?'∞':Number(m.psnr_db).toFixed(2)+' dB'} · MAE ${Number(m.mae_0_1).toFixed(4)} · Δt-MAE ${Number(m.temporal_delta_mae_0_1).toFixed(4)}${m.decoded_equal?' · 逐像素相同':' · 非逐像素相同'}`:'待计算',similarityCard=(kind,x)=>{const own=simId(kind,x),baseMetric=pair('baseline',own),fixedTube=pair(simId('fixed',x),simId('tube',x));return `<span class=\"similarity-card\"><b>vs Baseline：</b>${e(metricText(baseMetric))}<br><b>Fixed ↔ Tube：</b>${e(metricText(fixedTube))}</span>`},prettyId=id=>String(id).replace('single_object:','').replace('all_objects:all_objects','all_objects').replace('key_value_column','M5').replace('literal_kv_zero','C1').replace('cross_boundary','M6').replace('incoming_only','M2').replace('outgoing_only','M3').replace('query_row','M4').replace('row_and_column','M7').replace('self_only','M1'),marginalPairs={S:[['baseline','self_only'],['incoming_only','query_row'],['outgoing_only','key_value_column'],['cross_boundary','row_and_column']],I:[['baseline','incoming_only'],['self_only','query_row'],['outgoing_only','cross_boundary'],['key_value_column','row_and_column']],O:[['baseline','outgoing_only'],['self_only','key_value_column'],['incoming_only','cross_boundary'],['query_row','row_and_column']]},marginalScore=(kind,x,block)=>{const values=marginalPairs[block].map(([a,b])=>pair(a==='baseline'?'baseline':simId(kind,x,a),simId(kind,x,b))?.ssim_mean).filter(Number.isFinite);return values.length?values.reduce((u,v)=>u+v,0)/values.length:null},similarityPanel=()=>{if(!similarity.ready)return `<div class=\"similarity-panel pending\">视频相似度待计算：${e(similarity.reason||'metrics unavailable')}</div>`;const closest=(similarity.closest_same_target_pairs||[]).slice(0,6).map(c=>`<li><code>${e(prettyId(c.left_id))}</code> ↔ <code>${e(prettyId(c.right_id))}</code> · SSIM ${Number(c.ssim_mean).toFixed(4)} · MAE ${Number(c.mae_0_1).toFixed(4)}</li>`).join(''),diagnostics=targets.flatMap(([target_scope,region,label])=>['fixed','tube'].map(kind=>{const x={target_scope,region};return `<tr><td>${e(kind)}</td><td>${e(label)}</td>${['S','I','O'].map(block=>{const score=marginalScore(kind,x,block);return `<td>${Number.isFinite(score)?score.toFixed(4):'—'}</td>`}).join('')}</tr>`})).join(''),exactCount=(similarity.exact_decoded_groups||[]).length,res=similarity.comparison_resolution_hwc||[];return `<section class=\"similarity-panel\"><h3>视频相似度与“看起来一致”的诊断</h3><p><b>${similarity.video_count} 个视频 / ${similarity.comparison_count} 组比较</b>；全部 49 帧在 ${res[1]||'—'}×${res[0]||'—'} 上计算 SSIM、PSNR、MAE 和时间差分 MAE。原始尺寸解码帧哈希重复组：<b>${exactCount}</b>，因此当前没有两个结果逐像素完全相同。</p><div class=\"similarity-grid\"><div><h4>最相似的同 target 算子对</h4><ol>${closest}</ol></div><div><h4>如何从相似算子反推弱信息流</h4><p>M1↔M5、M2↔M6、M4↔M7 只多切一个 O；M1↔M4、M3↔M6、M5↔M7 只多切一个 I；M2↔M4、M3↔M5、M6↔M7 只多切一个 S。对应 pair 越相似，只能说明该上下文中新增矩阵块的<b>最终视频边际效应较弱</b>，不能说明内部 attention 数值为零。</p></div></div><div class=\"similarity-table\"><table><thead><tr><th>协议</th><th>Target</th><th>仅差 S 的平均 SSIM</th><th>仅差 I 的平均 SSIM</th><th>仅差 O 的平均 SSIM</th></tr></thead><tbody>${diagnostics}</tbody></table></div><p class=\"similarity-warning\"><b>为什么仍会很像：</b>R_fixed/R_tube 分别只占 6–14 / 79–179 个 token；只干预 Top100/720 个物理 heads。残差连接、其余 heads、FFN、cross-attention 与后续 40 步扩散可补偿或压低差异。高 SSIM 也可能来自变化集中在小物体区域；它不等于实验未执行。所有 manifest 的干预审计仍需与相似度一起判断。</p></section>`},src="""
    if page.count(temporal_find) != 1:
        raise RuntimeError("PhysicIQ67 temporal record finder changed")
    page = page.replace(temporal_find, temporal_similarity_helpers, 1)

    temporal_raft_anchor = "</p></section>`},src="
    temporal_raft_helpers = """</p></section>`},raft=s.ablation_raft_motion||{},raftComparisons=raft.comparisons||[],raftVideos=raft.videos||[],raftPair=(left,right)=>raftComparisons.find(c=>(c.left_id===left&&c.right_id===right)||(c.left_id===right&&c.right_id===left)),raftScope=x=>x.target_scope==='all_objects'?'all_objects_roi':`${x.region}_roi`,raftMetrics=(kind,x)=>raftPair('baseline',simId(kind,x))?.scopes?.[raftScope(x)],raftFixedTube=(x)=>raftPair(simId('fixed',x),simId('tube',x))?.scopes?.[raftScope(x)],fmt=(v,n=3)=>Number.isFinite(Number(v))?Number(v).toFixed(n):'—',raftMetricText=m=>m?`EPE/ref ${fmt(m.flow_epe_over_reference_magnitude)} · vector cos ${fmt(m.flow_vector_cosine)} · motion ×${fmt(m.motion_magnitude_ratio)} · profile r ${fmt(m.motion_profile_correlation)}`:'待计算',raftCard=(kind,x)=>{const own=raftMetrics(kind,x),fixedTube=raftFixedTube(x);return `<span class="raft-card"><b>RAFT vs Baseline · ${e(raftScope(x))}：</b>${e(raftMetricText(own))}<br><b>Fixed → Tube：</b>${e(raftMetricText(fixedTube))}</span>`},raftFlowSrc=id=>`${api}/raft-flow-video?${base}&video_id=${encodeURIComponent(id)}`,raftFlowDetails=id=>raft.ready&&raftVideos.some(v=>v.id===id)?`<details class="raft-flow"><summary>查看 RAFT 光流场 · HSV 方向/幅值（全视频同尺度）</summary><video controls muted playsinline preload="none" src="${raftFlowSrc(id)}"></video></details>`:'',raftPanel=()=>{if(!raft.ready)return `<div class="raft-panel pending">RAFT 运动相似度待计算：${e(raft.reason||'metrics unavailable')}</div>`;const baselineVideo=raftVideos.find(v=>v.id==='baseline'),baseRows=[['object_A_roi','object_A'],['object_B_roi','object_B'],['all_objects_roi','all_objects']].map(([scope,label])=>`<tr><td>${e(label)}</td><td>${fmt(baselineVideo?.motion?.[scope]?.mean_magnitude_px,4)}</td><td>${fmt(baselineVideo?.motion?.[scope]?.p95_magnitude_px,4)}</td></tr>`).join(''),metricRows=targets.flatMap(([target_scope,region,label])=>modes.map(mask_mode=>{const x={target_scope,region,mask_mode},d=defs[mask_mode],fixedMetric=raftMetrics('fixed',x),tubeMetric=raftMetrics('tube',x);return `<tr><td>${e(label)}</td><td>${e(d.id)}</td><td>${fmt(fixedMetric?.motion_magnitude_ratio)}</td><td>${fmt(fixedMetric?.flow_epe_over_reference_magnitude)}</td><td>${fmt(tubeMetric?.motion_magnitude_ratio)}</td><td>${fmt(tubeMetric?.flow_epe_over_reference_magnitude)}</td></tr>`})).join(''),corr=raft.analysis?.pixel_motion_correlation||{},epeCorr=corr.pixel_ssim_vs_flow_epe||{},cosCorr=corr.pixel_ssim_vs_flow_vector_cosine||{},profileCorr=corr.pixel_ssim_vs_motion_profile_correlation||{},settings=raft.settings||{},scale=raft.flow_visualization?.shared_max_magnitude_px;return `<section class="raft-panel"><h3>RAFT 运动相似度 · 生成运动是否真的保持</h3><p><b>${raft.video_count} 个视频 / ${raft.comparison_count} 组光流比较</b>；RAFT Large ${e(settings.weights||'')}，${settings.width||'—'}×${settings.height||'—'}，相邻 48 帧对。EPE 是两个生成视频的估计光流差，不是对 GT 光流的误差。每个实验默认读取自己的 baseline-frozen target ROI。</p><div class="raft-grid"><div><h4>Baseline ROI 运动量（px/frame）</h4><table><thead><tr><th>ROI</th><th>Mean</th><th>P95</th></tr></thead><tbody>${baseRows}</tbody></table><p>object_B 接近静止，方向 cosine 与倍率容易被小分母放大，应优先看绝对 EPE。</p></div><div><h4>像素 SSIM 与运动指标并不等价</h4><p>${corr.sample_count||0} 个 vs-baseline 实验：SSIM↔EPE Pearson ${fmt(epeCorr.pearson)} / Spearman ${fmt(epeCorr.spearman)}；SSIM↔flow cosine ${fmt(cosCorr.pearson)} / ${fmt(cosCorr.spearman)}；SSIM↔运动时序 profile ${fmt(profileCorr.pearson)} / ${fmt(profileCorr.spearman)}。</p><p>负的 SSIM↔EPE 符合“画面越像，光流差通常越小”，但相关性不是 ±1，因此 RAFT 提供了额外的运动诊断。</p>${raftFlowDetails('baseline')}</div></div><div class="raft-table"><table><thead><tr><th>Target</th><th>ID</th><th>Fixed motion/base</th><th>Fixed EPE/ref</th><th>Tube motion/base</th><th>Tube EPE/ref</th></tr></thead><tbody>${metricRows}</tbody></table></div><p class="raft-warning"><b>读取方法：</b>motion/base≈1 表示目标 ROI 的平均运动幅值接近基线，≈0 表示运动几乎消失；EPE/ref 越小，说明方向、幅值和空间分布的联合光流越接近。两者必须一起看。HSV 光流视频统一使用 baseline 全图 99.5% 分位 ${fmt(scale)} px 的幅值上限，所以不同结果可直接目测比较。</p></section>`},src="""
    if page.count(temporal_raft_anchor) != 1:
        raise RuntimeError("PhysicIQ67 temporal RAFT helper anchor changed")
    page = page.replace(temporal_raft_anchor, temporal_raft_helpers, 1)

    temporal_card_caption = '<span class="caption-exact"><b>算子：</b>${e(d.exact)}</span></figcaption>${vbenchSummary(r)}</figure>'
    temporal_card_with_similarity = '<span class="caption-exact"><b>算子：</b>${e(d.exact)}</span>${x.target_scope===\'all_tokens\'?\'\':similarityCard(kind,x)}${x.target_scope===\'all_tokens\'?\'\':raftCard(kind,x)}</figcaption>${vbenchSummary(r)}${x.target_scope===\'all_tokens\'?\'\':raftFlowDetails(simId(kind,x))}</figure>'
    if page.count(temporal_card_caption) != 1:
        raise RuntimeError("PhysicIQ67 temporal card caption changed")
    page = page.replace(temporal_card_caption, temporal_card_with_similarity, 1)

    temporal_baseline_caption = '<figcaption><strong>Baseline · seed=47326 · No intervention</strong></figcaption>${vbenchSummary(s,true)}</figure>'
    temporal_baseline_with_raft = '<figcaption><strong>Baseline · seed=47326 · No intervention</strong></figcaption>${vbenchSummary(s,true)}${raftFlowDetails(\'baseline\')}</figure>'
    if page.count(temporal_baseline_caption) != 1:
        raise RuntimeError("PhysicIQ67 temporal baseline caption changed")
    page = page.replace(temporal_baseline_caption, temporal_baseline_with_raft, 1)

    temporal_row_logic = '<small>被切断：${e(d.flow)}</small></div>${card(\'fixed\',x,find(fixed,x))}'
    temporal_row_with_logic = '<small>被切断：${e(d.flow)}</small><small><b>精确计算：</b>${e(logic[x.mask_mode].calc)}</small><small><b>理论后果：</b>${e(logic[x.mask_mode].theory)}</small></div>${card(\'fixed\',x,find(fixed,x))}'
    if page.count(temporal_row_logic) != 1:
        raise RuntimeError("PhysicIQ67 temporal row logic changed")
    page = page.replace(temporal_row_logic, temporal_row_with_logic, 1)

    temporal_panel_anchor = '</p></div>${baseline}<div class="tube-compare"><div class="tube-compare-row"><div class="tube-column-head">算子 ID / 被切断的信息流</div>'
    temporal_panel_insert = '</p></div>${similarityPanel()}${raftPanel()}${baseline}<div class="tube-compare"><div class="tube-compare-row"><div class="tube-column-head">算子 ID / 被切断的信息流</div>'
    if page.count(temporal_panel_anchor) != 1:
        raise RuntimeError("PhysicIQ67 temporal similarity panel anchor changed")
    page = page.replace(temporal_panel_anchor, temporal_panel_insert, 1)

    page = page.replace(
        "</style>",
        ".similarity-card{display:block;margin-top:8px;padding:7px 8px;border-left:4px solid var(--gold);background:#f8f1e5;font:11px/1.5 ui-monospace,monospace;color:#3f463f}.similarity-panel{margin:14px 0;padding:13px;border:1px solid var(--line);border-radius:12px;background:#f7f3e9}.similarity-panel h3,.similarity-panel h4{margin:0 0 8px}.similarity-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.similarity-grid>div{padding:11px;background:#fff}.similarity-grid ol{margin:0;padding-left:22px}.similarity-grid li{margin:5px 0;overflow-wrap:anywhere}.similarity-table{overflow-x:auto;margin-top:10px}.similarity-table table{min-width:760px}.similarity-warning{padding:10px 12px;border-left:6px solid #b95031;background:#fff1d8;line-height:1.55}@media(max-width:900px){.similarity-grid{grid-template-columns:1fr}}</style>",
        1,
    )
    page = page.replace(
        "</style>",
        ".raft-card{display:block;margin-top:8px;padding:7px 8px;border-left:4px solid #238171;background:#e7f4ef;font:11px/1.5 ui-monospace,monospace;color:#183f37}.raft-panel{margin:14px 0;padding:13px;border:1px solid #76a99d;border-radius:12px;background:#eaf5f1}.raft-panel h3,.raft-panel h4{margin:0 0 8px}.raft-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.raft-grid>div{padding:11px;background:#fff}.raft-table{overflow-x:auto;margin-top:10px}.raft-table table{min-width:820px}.raft-warning{padding:10px 12px;border-left:6px solid #238171;background:#fff;line-height:1.55}.raft-flow{margin:8px;padding:7px;border:1px solid #76a99d;background:#eef8f4}.raft-flow summary{cursor:pointer;font-weight:900}.raft-flow video{margin-top:8px;aspect-ratio:640/352;image-rendering:auto}@media(max-width:900px){.raft-grid{grid-template-columns:1fr}}</style>",
        1,
    )
    page = page.replace(
        "Baseline · seed=47326 · No intervention",
        "Baseline · seed=${s.seed} · No intervention",
        1,
    )
    page = page.replace("seed=47326、采样参数", "seed=${s.seed}、采样参数", 1)
    page = page.replace(
        "R_fixed/R_tube 分别只占 6–14 / 79–179 个 token",
        "R 只占 11440 个 token 中的一小部分；本 seed 的实际 |R| 见每张视频卡片",
        1,
    )

    temporal_card_start = "card=(kind,x,r)=>{const d=defs[x.mask_mode],isControl="
    if page.count(temporal_card_start) != 1:
        raise RuntimeError("PhysicIQ67 temporal card builder changed")
    page = page.replace(
        temporal_card_start,
        "card=(kind,x,r)=>{if(!r?.ready)return '';const d=defs[x.mask_mode],isControl=",
        1,
    )
    temporal_card_logic = '<span class="caption-exact"><b>算子：</b>${e(d.exact)}</span>'
    temporal_card_logic_expanded = '<span class="caption-exact"><b>矩阵块：</b>${e(d.matrix)}</span><span class="caption-flow"><b>切断信息流：</b>${e(d.flow)}</span><span class="caption-exact"><b>精确计算：</b>${e(logic[x.mask_mode].calc)}</span><span class="caption-exact"><b>理论后果：</b>${e(logic[x.mask_mode].theory)}</span>'
    if page.count(temporal_card_logic) != 1:
        raise RuntimeError("PhysicIQ67 temporal card logic changed")
    page = page.replace(temporal_card_logic, temporal_card_logic_expanded, 1)

    temporal_modes_anchor = "modes=['self_only','incoming_only','outgoing_only','query_row','key_value_column','cross_boundary','row_and_column','literal_kv_zero'],objectTargets="
    temporal_modes_with_controls = "modes=['self_only','incoming_only','outgoing_only','query_row','key_value_column','cross_boundary','row_and_column','literal_kv_zero'],controlModes=['qk_logits_zero','full_head_output'],objectTargets="
    if page.count(temporal_modes_anchor) != 1:
        raise RuntimeError("PhysicIQ67 temporal modes changed")
    page = page.replace(temporal_modes_anchor, temporal_modes_with_controls, 1)

    temporal_rows_start = page.index("rows=targets.map(([targetScope")
    temporal_rows_end = page.index(",done=tube.filter", temporal_rows_start)
    temporal_rows = """rows=(()=>{const objectRows=targets.map(([targetScope,targetRegion,targetLabel])=>{const group=specs.filter(x=>x.target_scope===targetScope&&x.region===targetRegion),fixedCards=group.map(x=>card('fixed',x,find(fixed,x))).filter(Boolean),tubeCards=group.map(x=>card('tube',x,find(tube,x))).filter(Boolean),readyCount=fixedCards.length+tubeCards.length;if(!readyCount)return '';const protocolRow=(kind,cards)=>cards.length?`<div class="object-protocol-row ${kind}"><div class="object-protocol-heading"><strong>${kind==='fixed'?'Fixed · R_fixed · 仅 latent t=0':'Tube · R_tube · latent t=0…12 联合集合'}</strong><span>${cards.length} 个已生成视频 · M1→C1</span></div><div class="object-ablation-strip">${cards.join('')}</div></div>`:'';return `<section class="object-ablation-row"><div class="object-ablation-heading"><h3>${e(targetLabel)}</h3><span>${readyCount} 个已生成 Top100 消融</span></div>${protocolRow('fixed',fixedCards)}${protocolRow('tube',tubeCards)}</section>`}).join(''),controlSpecs=controlModes.map(mask_mode=>({target_scope:'all_tokens',region:'',targetLabel:'Global all-token controls',mask_mode})),controlCards=controlSpecs.map(x=>card('fixed',x,find(fixed,x))).filter(Boolean),controlRow=controlCards.length?`<section class="object-ablation-row control-row"><div class="object-ablation-heading"><h3>Global all-token controls</h3><span>${controlCards.length} 个已生成 Top100 控制 · C2/C3 不依赖 R，无 Tube 重复项</span></div><div class="object-ablation-strip">${controlCards.join('')}</div></section>`:'';return objectRows+controlRow})()"""
    page = page[:temporal_rows_start] + temporal_rows + page[temporal_rows_end:]

    temporal_progress_anchor = ",done=tube.filter(r=>r.ready).length,baseline="
    temporal_progress_with_fixed = ",fixedDone=fixed.filter(r=>r.ready&&r.target_scope!=='all_tokens'&&modes.includes(r.mask_mode)).length,done=tube.filter(r=>r.ready).length,baseline="
    if page.count(temporal_progress_anchor) != 1:
        raise RuntimeError("PhysicIQ67 temporal progress changed")
    page = page.replace(temporal_progress_anchor, temporal_progress_with_fixed, 1)

    temporal_column_header = '<div class="tube-compare-row"><div class="tube-column-head">算子 ID / 被切断的信息流</div><div class="tube-column-head">左：R_fixed · 仅 latent t=0</div><div class="tube-column-head">右：R_tube · latent t=0…12 联合集合</div></div>'
    if page.count(temporal_column_header) != 1:
        raise RuntimeError("PhysicIQ67 temporal column header changed")
    page = page.replace(
        temporal_column_header,
        '<div class="object-layout-note">每个 object/target 分成两条独立横向视频行：Fixed R_fixed 一行、Tube R_tube 一行，均按 M1→C1 排列；C2/C3 放在独立的 Global all-token controls 行。只显示已经生成的 Top100 视频，未生成项不占位；拖动各行底部的水平滑动条查看全部视频。</div>',
        1,
    )
    page = page.replace(
        "</style>",
        ".object-layout-note{padding:10px 12px;border-left:6px solid var(--gold);background:#f8f1e5;line-height:1.5}.tube-compare,.object-ablation-row,.object-protocol-row,.object-ablation-strip{min-width:0;max-width:100%;width:100%}.object-ablation-row{margin:16px 0 28px}.object-ablation-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;padding:10px 12px;background:#17443a;color:#fff}.control-row .object-ablation-heading{background:#7a4b18}.object-ablation-heading h3{margin:0}.object-ablation-heading span,.object-protocol-heading span{font:11px ui-monospace,monospace}.object-protocol-row{margin-top:10px;padding:8px 9px 0;border:1px solid #d4c8b5;background:#f8f4eb}.object-protocol-row.tube{border-color:#86aa9f;background:#edf6f2}.object-protocol-heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;padding:4px 3px}.object-protocol-row.fixed .object-protocol-heading strong{color:#9a5318}.object-protocol-row.tube .object-protocol-heading strong{color:#176654}.object-ablation-strip{display:flex;flex-wrap:nowrap;gap:10px;overflow-x:scroll;overflow-y:hidden;padding:8px 2px 18px;scrollbar-gutter:stable;scrollbar-width:auto;scrollbar-color:#a35f1d #e8dfcc;overscroll-behavior-inline:contain;touch-action:pan-x;-webkit-overflow-scrolling:touch}.object-protocol-row.tube .object-ablation-strip{scrollbar-color:#176654 #d9e8e2}.object-ablation-strip::-webkit-scrollbar{height:16px}.object-ablation-strip::-webkit-scrollbar-track{background:#e8dfcc;border-radius:9px}.object-ablation-strip::-webkit-scrollbar-thumb{background:#a35f1d;border:3px solid #e8dfcc;border-radius:9px}.object-protocol-row.tube .object-ablation-strip::-webkit-scrollbar-track{background:#d9e8e2}.object-protocol-row.tube .object-ablation-strip::-webkit-scrollbar-thumb{background:#176654;border-color:#d9e8e2}.object-ablation-strip::-webkit-scrollbar-thumb:hover{background:#17443a}.object-ablation-strip>figure{flex:0 0 clamp(300px,27vw,430px);min-width:0}.object-ablation-strip video{width:100%;aspect-ratio:1280/704}@media(max-width:700px){.object-ablation-heading,.object-protocol-heading{align-items:flex-start;flex-direction:column}.object-ablation-strip>figure{flex-basis:82vw}}</style>",
        1,
    )
    page = page.replace(
        "</style>",
        ".ablation figure .similarity-card,.ablation figure .raft-card,.ablation figure .vbench-summary,.ablation figure .vbench-pending,.ablation figure .raft-flow{display:none!important}</style>",
        1,
    )
    return page


def wan22_ti2v_legacy_test5_asset(name: str):
    safe_name = Path(name).name
    if safe_name != name or not safe_name.endswith(".mp4"):
        return None
    return WAN22_TI2V_LEGACY_TEST5_ROOT / safe_name


def wan22_ti2v_legacy_test5_catalog():
    manifest_path = WAN22_TI2V_LEGACY_TEST5_ROOT / "batch_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    videos = []
    for path in sorted(WAN22_TI2V_LEGACY_TEST5_ROOT.glob("*.mp4")):
        metadata_path = path.with_suffix(".json")
        metadata = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
        videos.append(
            {
                "name": path.name,
                "case": path.stem,
                "bytes": path.stat().st_size,
                "prompt": metadata.get("input_caption", ""),
                "seed": metadata.get("seed", manifest.get("seed")),
                "steps": metadata.get("step", manifest.get("sampling_steps")),
                "backend": metadata.get("backend", manifest.get("backend")),
            }
        )
    return {"manifest": manifest, "videos": videos}
OBJECT_QUERY_STEP_ALIGNMENT_SEEDS = (47326, 90094, 32466, 35075, 21890, 49530)
OBJECT_QUERY_STEP_ALIGNMENT_IMAGES = {
    "best_match_curves.png",
    "top30_mean_per_head_cosine.png",
    "top50_mean_per_head_cosine.png",
    "top100_mean_per_head_cosine.png",
    "top30_cosine_of_mean_map.png",
    "top50_cosine_of_mean_map.png",
    "top100_cosine_of_mean_map.png",
}
OBJECT_QUERY_STEP_ALIGNMENT_DOWNLOADS = {
    "topn_best_matches.csv",
    "per_head_best_matches.csv",
    "per_head_best_matches_by_sample.csv",
}


def object_query_step_alignment_asset(model: str, name: str):
    config = OBJECT_QUERY_STEP_ALIGNMENT_MODELS.get(model)
    safe_name = Path(name).name
    if config is None or safe_name != name:
        return None
    if safe_name not in OBJECT_QUERY_STEP_ALIGNMENT_IMAGES | OBJECT_QUERY_STEP_ALIGNMENT_DOWNLOADS:
        return None
    return config["root"] / "analysis" / safe_name


def object_query_step_alignment_video(model: str, seed: str, steps: str, kind: str):
    config = OBJECT_QUERY_STEP_ALIGNMENT_MODELS.get(model)
    if config is None or not seed.isdigit() or int(seed) not in OBJECT_QUERY_STEP_ALIGNMENT_SEEDS:
        return None
    if steps not in {"10", "40"} or kind not in {"original", "probe"}:
        return None
    if model == "baseline":
        if kind != "original":
            return None
        return (
            config["video_root"]
            / "seeds"
            / f"seed_{int(seed):06d}"
            / f"steps{steps}"
            / "original.mp4"
        )
    model_dir = "baseline" if model == "baseline" else "lora"
    filename = "original.mp4" if kind == "original" else "top100_steps_00_40.mp4"
    return (
        config["root"]
        / "seeds"
        / f"seed_{int(seed):06d}"
        / f"steps{steps}"
        / "videos"
        / model_dir
        / "cases"
        / "0613pybullet_sample_001460_w002"
        / filename
    )


def object_query_step_alignment_video_exists(
    model: str, seed: int, steps: int, kind: str
) -> bool:
    asset = object_query_step_alignment_video(
        model, str(seed), str(steps), kind
    )
    return asset is not None and asset.is_file()


def object_query_step_alignment_catalog():
    import csv

    models = []
    for model, config in OBJECT_QUERY_STEP_ALIGNMENT_MODELS.items():
        root = config["root"]
        analysis = root / "analysis"
        progress = []
        for seed in OBJECT_QUERY_STEP_ALIGNMENT_SEEDS:
            seed_root = root / "seeds" / f"seed_{seed:06d}"
            progress.append(
                {
                    "seed": seed,
                    "steps40": len(list((seed_root / "steps40" / "captures").glob("*.npz"))),
                    "steps10": len(list((seed_root / "steps10" / "captures").glob("*.npz"))),
                    "complete": (seed_root / "complete").is_file(),
                }
            )
        summary_path = analysis / "summary.json"
        summary = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = None
        matches = []
        matches_path = analysis / "topn_best_matches.csv"
        if matches_path.is_file():
            try:
                with matches_path.open("r", encoding="utf-8", newline="") as handle:
                    matches = list(csv.DictReader(handle))
            except OSError:
                matches = []
        models.append(
            {
                "id": model,
                "label": config["label"],
                "detail": config["detail"],
                "attention_capture": config["attention_capture"],
                "ready": summary is not None,
                "summary": summary,
                "matches": matches,
                "progress": progress,
                "videos": {
                    str(seed): {
                        str(steps): {
                            kind: object_query_step_alignment_video_exists(
                                model, seed, steps, kind
                            )
                            for kind in ("original", "probe")
                        }
                        for steps in (40, 10)
                    }
                    for seed in OBJECT_QUERY_STEP_ALIGNMENT_SEEDS
                },
                "images": sorted(
                    name for name in OBJECT_QUERY_STEP_ALIGNMENT_IMAGES
                    if (analysis / name).is_file()
                ),
                "downloads": sorted(
                    name for name in OBJECT_QUERY_STEP_ALIGNMENT_DOWNLOADS
                    if (analysis / name).is_file()
                ),
            }
        )
    return {
        "case": "0613pybullet_sample_001460_w002",
        "seeds": list(OBJECT_QUERY_STEP_ALIGNMENT_SEEDS),
        "models": models,
    }


def wan22_ti2v_legacy_test5_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wan2.2 Legacy TI2V Test5</title><style>
:root{--paper:#ece4d6;--ink:#182720;--deep:#17443a;--line:#bcae99;--card:#fffaf0;--orange:#bd4c31}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 2% 0,#d7633e46,transparent 34rem),radial-gradient(circle at 98% 2%,#26887543,transparent 39rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:16px 22px;background:#ece4d6ef;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:5px 0;font-size:clamp(31px,4.6vw,62px);line-height:1}.lead{max-width:1150px;margin:8px 0}.tools,.meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap}button,.pill{padding:7px 11px;border:1px solid var(--line);background:var(--card);font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(100% - 18px,2100px);margin:auto;padding:18px 0 70px}.grid{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:12px}.card{overflow:hidden;border:1px solid var(--line);border-radius:14px;background:#fffaf0;box-shadow:0 13px 35px #53412817}.card h2{margin:0;padding:11px 13px;background:var(--deep);color:#fff;font:900 15px ui-monospace,monospace}.card video{display:block;width:100%;aspect-ratio:1280/704;background:#111}.caption{padding:10px 12px;min-height:72px;font-size:12px;line-height:1.45}.case-meta{padding:0 12px 12px;font:11px ui-monospace,monospace;color:#6d675d}.empty{padding:50px;text-align:center}@media(max-width:1150px){.grid{grid-template-columns:repeat(2,minmax(300px,1fr))}}@media(max-width:720px){header{position:static}.grid{grid-template-columns:1fr}}
</style></head><body><header><a href="/">返回总览</a><h1>Wan2.2-TI2V<br>Legacy Test5 Wall</h1><p class="lead">该页展示历史目录中的全部现有视频。生成入口为 AAAinfer/wanti2v.py，backend 为 legacy_diffsynth.WanVideoPipeline；输入是 JSON prompt 与 input_image 首帧，不使用 context_video。</p><div class="tools"><button id="replay">重新播放全部</button><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div><div id="meta" class="meta"></div></header><main><section id="grid" class="grid"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));async function load(){const d=await fetch('/api/wan22-ti2v-legacy-test5/catalog',{cache:'no-store'}).then(r=>r.json()),m=d.manifest||{};document.getElementById('status').textContent=`${d.videos.length} videos ready`;document.getElementById('meta').innerHTML=[`backend ${m.backend_impl||m.backend||'unknown'}`,`${m.height||'?'}×${m.width||'?'}`,`${m.frame_num||'?'} frames`,`${m.sampling_steps||'?'} steps`,`seed ${m.seed??'?'}`,`CFG ${m.cfg_scale??'?'}`].map(x=>`<span class="pill">${e(x)}</span>`).join('');document.getElementById('grid').innerHTML=d.videos.length?d.videos.map(v=>`<article class="card"><h2>${e(v.case)}</h2><video controls muted playsinline preload="metadata" src="/api/wan22-ti2v-legacy-test5/video?name=${encodeURIComponent(v.name)}"></video><div class="caption">${e(v.prompt||'No prompt metadata')}</div><div class="case-meta">seed ${e(v.seed)} · ${e(v.steps)} steps · ${e(v.backend)} backend · ${(Number(v.bytes)/1048576).toFixed(1)} MiB</div></article>`).join(''):'<div class="empty">目录中暂无 MP4</div>'}document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load();
</script></body></html>'''


WAN22_TI2V_LEGACY_COMPARISON_SCRIPT = r'''
let comparison = null;
let comparisonScope = "s039";
let comparisonReference = "combined";
let comparisonTopK = 30;

function comparisonDatasets() {
  const scope = comparison.scopes[comparisonScope];
  return [scope, ...Object.values(scope.references)];
}

function comparisonGrid(width, height, xTicks, xMin, xMax) {
  const left = 58, right = 18, top = 18, bottom = 42;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const x = value => left + (value - xMin) / (xMax - xMin) * plotWidth;
  const y = value => top + (100 - value) / 100 * plotHeight;
  const horizontal = [0, 20, 40, 60, 80, 100].map(value =>
    `<line x1="${left}" y1="${y(value)}" x2="${width-right}" y2="${y(value)}" stroke="#ded5c5"/>` +
    `<text x="${left-8}" y="${y(value)+4}" text-anchor="end">${value}</text>`
  ).join("");
  const vertical = xTicks.map(value =>
    `<line x1="${x(value)}" y1="${top}" x2="${x(value)}" y2="${height-bottom}" stroke="#eee7db"/>` +
    `<text x="${x(value)}" y="${height-bottom+18}" text-anchor="middle">${value}</text>`
  ).join("");
  return {left, right, top, bottom, plotWidth, plotHeight, x, y, grid: horizontal + vertical};
}

function renderRankChart() {
  const width = 980, height = 360;
  const grid = comparisonGrid(width, height, [1, 180, 360, 540, 720], 1, 720);
  const sets = comparisonDatasets();
  const paths = sets.map(set => {
    const rows = [...set.rows].sort((left, right) => left.rank - right.rank);
    const points = rows.map(row =>
      `${grid.x(row.rank).toFixed(2)},${grid.y(row.pck32).toFixed(2)}`
    ).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${set.color}" stroke-width="2.2" opacity=".9"><title>${e(set.label)}</title></polyline>`;
  }).join("");
  const legend = sets.map(set =>
    `<span><i class="legend-swatch" style="background:${set.color}"></i>${e(set.label)}</span>`
  ).join("");
  $("rankChart").innerHTML =
    `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="PCK ranked distributions">` +
    `<g font-family="ui-monospace,monospace" font-size="10" fill="#59645e">${grid.grid}</g>` +
    `<line x1="${grid.left}" y1="${grid.top}" x2="${grid.left}" y2="${height-grid.bottom}" stroke="#17261f"/>` +
    `<line x1="${grid.left}" y1="${height-grid.bottom}" x2="${width-grid.right}" y2="${height-grid.bottom}" stroke="#17261f"/>` +
    paths +
    `<text x="${width/2}" y="${height-7}" text-anchor="middle" font-size="11">PCK rank among 720 Heads</text>` +
    `<text x="14" y="${height/2}" text-anchor="middle" font-size="11" transform="rotate(-90 14 ${height/2})">PCK@32 (%)</text></svg>` +
    `<div class="chart-legend">${legend}</div>`;
}

function renderScatter(scope, reference, selected) {
  const width = 760, height = 360;
  const grid = comparisonGrid(width, height, [0, 20, 40, 60, 80, 100], 0, 100);
  const referenceMap = new Map(reference.rows.map(row => [`${row.block}-${row.head}`, row]));
  const common = new Set(selected.common_heads.map(row => `${row.block}-${row.head}`));
  const circles = scope.rows.map(row => {
    const key = `${row.block}-${row.head}`, ref = referenceMap.get(key);
    const isCommon = common.has(key);
    return `<circle cx="${grid.x(row.pck32).toFixed(2)}" cy="${grid.y(ref.pck32).toFixed(2)}" r="${isCommon ? 3.1 : 1.8}" fill="${isCommon ? "#17261f" : reference.color}" opacity="${isCommon ? .9 : .35}"><title>L${String(row.block).padStart(2,"0")}/H${String(row.head).padStart(2,"0")} · Legacy ${row.pck32.toFixed(3)}% · ${e(reference.label)} ${ref.pck32.toFixed(3)}%</title></circle>`;
  }).join("");
  $("scatterTitle").textContent = `${scope.label} × ${reference.label}`;
  $("scatterChart").innerHTML =
    `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Paired Head PCK scatter">` +
    `<g font-family="ui-monospace,monospace" font-size="10" fill="#59645e">${grid.grid}</g>` +
    `<line x1="${grid.left}" y1="${height-grid.bottom}" x2="${width-grid.right}" y2="${grid.top}" stroke="#8c897f" stroke-dasharray="5 5"/>` +
    `<line x1="${grid.left}" y1="${grid.top}" x2="${grid.left}" y2="${height-grid.bottom}" stroke="#17261f"/>` +
    `<line x1="${grid.left}" y1="${height-grid.bottom}" x2="${width-grid.right}" y2="${height-grid.bottom}" stroke="#17261f"/>` +
    circles +
    `<text x="${width/2}" y="${height-7}" text-anchor="middle" font-size="11">Legacy PCK@32 (%)</text>` +
    `<text x="14" y="${height/2}" text-anchor="middle" font-size="11" transform="rotate(-90 14 ${height/2})">Reference PCK@32 (%)</text></svg>` +
    `<div class="chart-legend"><span><i class="legend-swatch" style="background:${reference.color}"></i>全部 720 Head</span><span><i class="legend-swatch" style="background:#17261f"></i>当前 Top-K 交集</span></div>`;
}

function renderPairwise() {
  const box = $("pairwiseBody");
  const exportMeta = comparison?.ranking_export;
  if (!box || !comparison?.pairwise_by_view?.[comparisonScope]) {
    if (box) box.innerHTML = `<tr><td colspan="7" class="common-empty">PCK 排名导出尚未生成</td></tr>`;
    return;
  }
  const pairs = Object.values(comparison.pairwise_by_view[comparisonScope]);
  const view = exportMeta?.views?.[comparisonScope];
  $("pairwiseStatus").textContent =
    `${view?.label || comparisonScope} · ${pairs.length} 个 pair · 每组 720 个物理 Head`;
  box.innerHTML = pairs.map(pair => {
    const overlapCell = topK => {
      const item = pair.overlaps[`Top${topK}`];
      return `<td class="pair-cell"><b>${item.common_count}/${topK}</b><span>J ${item.jaccard.toFixed(3)} · ${item.coverage_pct.toFixed(1)}%</span></td>`;
    };
    return `<tr><td>${e(pair.left_label)} × ${e(pair.right_label)}</td>` +
      [10, 30, 50, 100].map(overlapCell).join("") +
      `<td class="pair-cell"><b>${pair.pearson_pck32.toFixed(4)}</b><span>全 720 Head</span></td>` +
      `<td class="pair-cell"><b>${pair.spearman_pck32.toFixed(4)}</b><span>平均秩</span></td></tr>`;
  }).join("");
}

function renderComparison() {
  if (!comparison?.ready) {
    $("comparisonStatus").textContent = comparison?.reason || "比较数据读取中";
    return;
  }
  const scope = comparison.scopes[comparisonScope];
  const reference = scope.references[comparisonReference];
  const result = scope.comparisons[comparisonReference];
  const selected = result.overlaps[String(comparisonTopK)];
  const references = Object.entries(scope.references);
  document.querySelectorAll("#scopeTabs button").forEach(button => {
    const active = button.dataset.scope === comparisonScope;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  document.querySelectorAll("#referenceTabs button").forEach(button =>
    button.classList.toggle("active", button.dataset.reference === comparisonReference)
  );
  document.querySelectorAll("#topKTabs button").forEach(button =>
    button.classList.toggle("active", Number(button.dataset.topK) === comparisonTopK)
  );
  $("comparisonStatus").textContent = `${scope.view_label || scope.label} · Legacy 与五组导出排名对齐 · 720/720 个物理 Head`;
  $("overlapBody").innerHTML = references.map(([key, ref]) =>
    `<tr><td><i class="legend-swatch" style="background:${ref.color}"></i>${e(ref.label)}</td>` +
    comparison.top_ks.map(topK => {
      const overlap = scope.comparisons[key].overlaps[String(topK)];
      const active = key === comparisonReference && topK === comparisonTopK;
      return `<td><button class="overlap-button ${active ? "active" : ""}" data-overlap-reference="${key}" data-overlap-k="${topK}"><b>${overlap.common_count}/${topK}</b><span>覆盖 ${overlap.coverage_pct.toFixed(1)}% · J ${overlap.jaccard.toFixed(3)}</span></button></td>`;
    }).join("") + "</tr>"
  ).join("");
  document.querySelectorAll(".overlap-button").forEach(button =>
    button.addEventListener("click", () => selectComparison(null, button.dataset.overlapReference, Number(button.dataset.overlapK)))
  );
  $("comparisonSummary").innerHTML =
    `<span>Top-K 交集<b>${selected.common_count} / ${comparisonTopK}</b></span>` +
    `<span>双向覆盖率<b>${selected.coverage_pct.toFixed(1)}%</b></span>` +
    `<span>Jaccard<b>${selected.jaccard.toFixed(3)}</b></span>` +
    `<span>Pearson · 720 Head<b>${result.pearson.toFixed(3)}</b></span>` +
    `<span>Spearman · 720 Head<b>${result.spearman.toFixed(3)}</b></span>`;
  renderRankChart();
  renderScatter(scope, reference, selected);
  renderPairwise();
  $("distributionBody").innerHTML = comparisonDatasets().map(set => {
    const dist = set.distribution;
    return `<tr><td><i class="legend-swatch" style="background:${set.color}"></i>${e(set.label)}</td>` +
      ["min", "p10", "p25", "median", "mean", "p75", "p90", "max", "std"].map(key => `<td>${dist[key].toFixed(2)}</td>`).join("") +
      "</tr>";
  }).join("");
  $("commonHeadsTitle").textContent = `${scope.label} × ${reference.label} · Top${comparisonTopK} 公共 Head (${selected.common_count})`;
  $("commonHeadsBody").innerHTML = selected.common_heads.length ? selected.common_heads.map((row, index) => {
    const delta = row.reference_rank - row.legacy_rank;
    return `<tr><td>${index+1}</td><td>L${String(row.block).padStart(2,"0")} / H${String(row.head).padStart(2,"0")}</td><td>#${row.legacy_rank}</td><td>${row.legacy_pck32.toFixed(3)}%</td><td>#${row.reference_rank}</td><td>${row.reference_pck32.toFixed(3)}%</td><td>${delta > 0 ? "+" : ""}${delta}</td></tr>`;
  }).join("") : `<tr><td colspan="7" class="common-empty">当前两组 Top${comparisonTopK} 没有公共 Head</td></tr>`;
}

function selectComparison(scope, reference, topK) {
  if (scope) comparisonScope = scope;
  if (reference) comparisonReference = reference;
  if (topK) comparisonTopK = topK;
  renderComparison();
}

document.querySelectorAll("#scopeTabs button").forEach(button =>
  button.addEventListener("click", () => selectComparison(button.dataset.scope, null, null))
);
document.querySelectorAll("#referenceTabs button").forEach(button =>
  button.addEventListener("click", () => selectComparison(null, button.dataset.reference, null))
);
document.querySelectorAll("#topKTabs button").forEach(button =>
  button.addEventListener("click", () => selectComparison(null, null, Number(button.dataset.topK)))
);
'''


def wan22_ti2v_legacy_pck50_page():
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Legacy TI2V First-Latent PCK50</title><style>
:root{--paper:#ece4d5;--ink:#17261f;--deep:#17443a;--line:#baad98;--card:#fffaf0;--rust:#b7482f;--gold:#d29c35}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#d65f3c44,transparent 34rem),radial-gradient(circle at 100% 0,#27897942,transparent 40rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:15px 22px;background:#ece4d5f2;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:4px 0;font-size:clamp(29px,4.4vw,58px);line-height:1}.lead{max-width:1200px;margin:7px 0}.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.tools label,.tools select{max-width:100%}button,select{padding:8px 11px;border:1px solid var(--line);background:var(--card);font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(100% - 18px,2100px);margin:auto;padding:18px 0 70px}.overall{padding:18px;margin-bottom:14px;border:1px solid #0f3d35;border-radius:18px;background:linear-gradient(115deg,#153f35,#21675a);color:#fff;box-shadow:0 16px 38px #183d342b}.overall-head{display:flex;justify-content:space-between;gap:14px;align-items:end;flex-wrap:wrap}.overall-number{font:900 clamp(30px,5vw,66px)/.9 ui-monospace,monospace}.overall-label{opacity:.8}.overall-grid{display:grid;grid-template-columns:repeat(4,minmax(190px,1fr));gap:10px;margin-top:17px}.phase{padding:11px;border:1px solid #ffffff38;background:#ffffff10;border-radius:11px}.phase-title{display:flex;justify-content:space-between;gap:8px;font:800 12px ui-monospace,monospace}.bar{height:9px;margin-top:9px;border-radius:99px;background:#061c1780;overflow:hidden}.bar span{display:block;height:100%;background:linear-gradient(90deg,var(--gold),#f0cc75);border-radius:inherit}.phase-note{margin-top:7px;font-size:11px;opacity:.78}.progress{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:9px;margin-bottom:15px}.case-progress{padding:10px 12px;border:1px solid var(--line);background:#fff9ee;border-radius:10px;font:12px ui-monospace,monospace}.workspace,.ranking,.performance,.comparison{padding:15px;margin:14px 0;border:1px solid var(--line);border-radius:16px;background:#fffaf0e8;box-shadow:0 13px 34px #58442b16}.performance-head,.comparison-head{display:flex;justify-content:space-between;gap:12px;align-items:end;flex-wrap:wrap}.performance-head h2,.comparison-head h2{margin:0}.performance-note,.comparison-note{max-width:1050px;margin:7px 0 0;color:#6d675d;line-height:1.45}.performance-matrices{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:13px}.matrix-panel{min-width:0;border:1px solid var(--line);background:#fff;padding:10px}.matrix-panel h3{margin:0 0 4px;font-size:16px}.matrix-meta{min-height:18px;margin:0 0 7px;color:#6d675d;font:11px ui-monospace,monospace}.heat-scroll{overflow:auto;border:1px solid #d4c8b5}.performance-heat{display:grid;grid-template-columns:48px repeat(24,minmax(34px,1fr));gap:3px;min-width:980px;padding:12px}.heat-axis,.heat-cell{display:flex;align-items:center;justify-content:center;height:29px;border-radius:4px;font:9px ui-monospace,monospace}.heat-axis{color:#6d756f}.heat-cell{border:0;min-width:0;padding:1px;cursor:default;color:#fff;text-shadow:0 1px 2px #000}.heat-cell:hover{outline:2px solid var(--ink);outline-offset:1px}.segmented{display:flex;gap:2px;padding:3px;border:1px solid var(--line);background:#e7decf}.segmented button{border:0;background:transparent;color:var(--ink);padding:7px 10px;cursor:pointer}.segmented button.active{background:var(--deep);color:#fff}.comparison-status{display:block;margin:10px 0;color:#6d675d}.overlap-wrap{overflow:auto;border:1px solid var(--line);background:#fff}.overlap-table{min-width:760px}.overlap-table th:first-child,.overlap-table td:first-child{text-align:left;white-space:nowrap}.overlap-button{display:block;width:100%;border:0;background:transparent;padding:5px;color:inherit;cursor:pointer}.overlap-button b{display:block;font:900 18px ui-monospace,monospace}.overlap-button span{display:block;margin-top:2px;font:10px ui-monospace,monospace;color:#6d675d}.overlap-button.active{background:#fff0c8;outline:2px solid var(--gold);outline-offset:-2px}.comparison-controls{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin:13px 0}.comparison-summary{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:7px;margin:10px 0}.comparison-summary span{padding:9px 10px;border-left:3px solid var(--deep);background:#eee6d8;font:11px ui-monospace,monospace}.comparison-summary b{display:block;margin-top:3px;font-size:18px}.chart-grid{display:grid;grid-template-columns:1.15fr 1fr;gap:14px;margin-top:16px}.chart-panel{min-width:0;border-top:2px solid var(--deep);padding-top:9px}.chart-panel h3{margin:0 0 3px}.chart-caption{margin:0 0 8px;color:#6d675d;font-size:11px}.chart-panel svg{display:block;width:100%;height:auto;aspect-ratio:980/360;background:#fff;border:1px solid var(--line)}.chart-legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:7px;font:10px ui-monospace,monospace}.legend-swatch{display:inline-block;width:13px;height:3px;margin-right:5px;vertical-align:middle}.distribution{margin-top:16px}.distribution h3,.common-heads h3{margin:0 0 7px}.distribution .scroll,.common-heads .scroll{max-height:470px}.common-heads{margin-top:16px}.common-empty{padding:24px;text-align:center;color:#756d61}.viewer{display:grid;grid-template-columns:minmax(340px,.72fr) minmax(650px,1.6fr);gap:12px;margin-top:13px}figure{margin:0;border:1px solid #d4c8b5;background:#fff;padding:8px}video,img{display:block;width:100%;background:#111}video{aspect-ratio:1280/704}figcaption{padding:8px 3px 2px;font-weight:900}.pending{display:grid;place-items:center;min-height:250px;background:#f0eadf;color:#766d60}.tables{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.tables>div{min-width:0}.scroll{overflow:auto;border:1px solid var(--line);background:#fff}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;border-bottom:1px solid #ddd2c0;text-align:center}th{background:var(--deep);color:#fff}tr:first-child td{background:#fff0c8;font-weight:900}@media(max-width:1050px){header{position:static}.overall-grid{grid-template-columns:repeat(2,1fr)}.progress,.tables,.viewer,.performance-matrices,.chart-grid{grid-template-columns:minmax(0,1fr)}.comparison-summary{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:620px){.overall-grid,.comparison-summary{grid-template-columns:1fr}.segmented{width:100%;overflow:auto}.segmented button{flex:1;white-space:nowrap}}</style></head><body><header><a href="/">返回总览</a><h1>Legacy TI2V<br>First-Latent PCK50</h1><p class="lead">首个 latent frame 固定为 object query；逐个验证 6 case × 50 seed × 40 step × 30 block × 24 head。Conditional 分支、不跨 Head 平均，PCK@32 对可见 object-query token/latent 比较做 micro aggregation。</p><div class="tools"><label>Case <select id="case"></select></label><label>Seed <select id="seed"></select></label><label>Global Top10 <select id="rank"></select></label><label>Object <select id="region"></select></label><button id="refresh">手动刷新</button><button id="replay">重新播放</button><span id="status" class="status">读取中</span></div></header><main><section id="overall" class="overall"></section><section id="progress" class="progress"></section><section class="performance"><div class="performance-head"><div><h2>Block × Head PCK@32 性能矩阵</h2><p class="performance-note">同一组 6 case × 50 seed 的 micro aggregation；左图固定最终时间步 S039，右图汇总 S000–S039 全部 step。悬停格子可查看具体 Block、Head、PCK@32、平均误差和比较次数。</p></div><label>指标 <select id="performanceMetric"><option value="pck32">PCK@32（越高越好）</option><option value="mean_error_px">平均误差 px（越低越好）</option></select></label></div><div class="performance-matrices"><article class="matrix-panel"><h3>当前时间步 S039 · 30 × 24 性能图</h3><p class="matrix-meta" id="s039Meta">等待矩阵数据</p><div class="heat-scroll"><div class="performance-heat" id="s039Heat"><span class="pending">计算中</span></div></div></article><article class="matrix-panel"><h3>所有 step 平均 · 30 × 24 性能图</h3><p class="matrix-meta" id="allStepsMeta">等待矩阵数据</p><div class="heat-scroll"><div class="performance-heat" id="allStepsHeat"><span class="pending">计算中</span></div></div></article></div></section><section class="comparison"><div class="comparison-head"><div><h2>Legacy 与三模型 PCK Head 重合分析</h2><p class="comparison-note">按相同物理 Block/Head 对齐。重合矩阵同时报告 Top10、Top30、Top50 和 Top100；分布与相关性使用全部 720 个 Head 的实际 PCK@32。</p></div><div class="segmented" id="scopeTabs" role="group" aria-label="Legacy aggregation"><button class="active" data-scope="s039" aria-pressed="true">Legacy S039</button><button data-scope="all_steps_mean" aria-pressed="false">所有 Step 平均</button></div></div><span id="comparisonStatus" class="comparison-status status">读取比较数据</span><div class="overlap-wrap"><table class="overlap-table"><thead><tr><th>参考 PCK 排名</th><th>Top10</th><th>Top30</th><th>Top50</th><th>Top100</th></tr></thead><tbody id="overlapBody"></tbody></table></div><div class="comparison-controls"><div class="segmented" id="referenceTabs" role="tablist" aria-label="Reference ranking"><button data-reference="gt">GT</button><button data-reference="lora">LoRA</button><button data-reference="baseline">Baseline</button><button class="active" data-reference="combined">三模型综合</button></div><div class="segmented" id="topKTabs" role="group" aria-label="Top K"><button data-top-k="10">Top10</button><button class="active" data-top-k="30">Top30</button><button data-top-k="50">Top50</button><button data-top-k="100">Top100</button></div></div><div id="comparisonSummary" class="comparison-summary"></div><div class="chart-grid"><div class="chart-panel"><h3>720 Head PCK@32 排名分布</h3><p class="chart-caption">五组实际 PCK@32 由高到低排列；横轴为排名，纵轴固定为 0–100%。</p><div id="rankChart"></div></div><div class="chart-panel"><h3 id="scatterTitle">成对 PCK@32</h3><p class="chart-caption">每个点是同一个 Block/Head；深色点属于当前 Top-K 交集。</p><div id="scatterChart"></div></div></div><div class="distribution"><h3>实际 PCK@32 分布统计</h3><div class="scroll"><table><thead><tr><th>数据集</th><th>Min</th><th>P10</th><th>P25</th><th>Median</th><th>Mean</th><th>P75</th><th>P90</th><th>Max</th><th>Std</th></tr></thead><tbody id="distributionBody"></tbody></table></div></div><div class="common-heads"><h3 id="commonHeadsTitle">公共 Head</h3><div class="scroll"><table><thead><tr><th>#</th><th>Block / Head</th><th>Legacy Rank</th><th>Legacy PCK@32</th><th>Reference Rank</th><th>Reference PCK@32</th><th>Rank Delta</th></tr></thead><tbody id="commonHeadsBody"></tbody></table></div></div></section><section class="workspace"><h2>Object-query attention overlay</h2><div id="viewer" class="viewer"></div></section><section class="ranking"><h2>PCK@32 Ranking</h2><div id="tables" class="tables"></div></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null;const $=id=>document.getElementById(id);const performanceLabels={pck32:'PCK@32',mean_error_px:'平均误差 px'};function options(node,values,label){const old=node.value;node.innerHTML=values.map((v,i)=>`<option value="${e(v.value??v)}">${e(label?label(v,i):v.label??v)}</option>`).join('');if([...node.options].some(o=>o.value===old))node.value=old}function table(rows,kind){return `<div class="scroll"><table><thead><tr><th>#</th>${kind==='combo'?'<th>Step</th>':''}<th>Block</th><th>Head</th><th>PCK@32</th><th>N</th></tr></thead><tbody>${(rows||[]).map((r,i)=>`<tr><td>${i+1}</td>${kind==='combo'?`<td>S${String(r.step).padStart(2,'0')}</td>`:''}<td>L${String(r.block).padStart(2,'0')}</td><td>H${String(r.head).padStart(2,'0')}</td><td>${r.pck32==null?'—':Number(r.pck32).toFixed(2)+'%'}</td><td>${r.comparisons||0}</td></tr>`).join('')}</tbody></table></div>`}function syncRegions(){const c=$('case').value;options($('region'),data.objects[c]||['object_A'])}function phase(label,done,total,note){const pct=total?100*done/total:0;return `<div class="phase"><div class="phase-title"><span>${e(label)}</span><span>${done}/${total}</span></div><div class="bar"><span style="width:${Math.min(100,pct).toFixed(2)}%"></span></div><div class="phase-note">${e(note)} · ${pct.toFixed(1)}%</div></div>`}function performanceColor(value,low,high,inverse){let t=high>low?Math.max(0,Math.min(1,(value-low)/(high-low))):.5;if(inverse)t=1-t;return `hsl(${12+120*t} 58% ${34+7*t}%)`}function renderPerformanceMatrix(kind,boxId,metaId,label,low,high){const rows=data.performance?.matrices?.[kind]||[],metric=$('performanceMetric').value,box=$(boxId),map=new Map(rows.map(r=>[`${r.block}-${r.head}`,r]));if(!rows.length){box.innerHTML='<span class="pending">矩阵数据尚未生成</span>';$(metaId).textContent='等待 combined_counts.npz';return}box.innerHTML='<span class="heat-axis">L/H</span>'+[...Array(24)].map((_,h)=>`<span class="heat-axis">H${String(h).padStart(2,'0')}</span>`).join('');for(let block=0;block<30;block++){box.insertAdjacentHTML('beforeend',`<span class="heat-axis">L${String(block).padStart(2,'0')}</span>`);for(let head=0;head<24;head++){const row=map.get(`${block}-${head}`),value=row?.[metric],cell=document.createElement('button');cell.type='button';cell.className='heat-cell';if(Number.isFinite(value)){cell.style.background=performanceColor(value,low,high,metric==='mean_error_px');cell.textContent=Number(value).toFixed(metric==='pck32'?1:0);cell.title=`${label} · L${String(block).padStart(2,'0')} H${String(head).padStart(2,'0')} · PCK@32 ${Number(row.pck32).toFixed(3)}% · 平均误差 ${Number(row.mean_error_px).toFixed(3)} px · N ${Number(row.comparisons).toLocaleString()}`}else{cell.disabled=true;cell.textContent='—'}box.append(cell)}}const better=metric==='pck32'?'绿色表示 PCK 更高':'绿色表示误差更低';$(metaId).textContent=`${performanceLabels[metric]} · 共用色标 ${low.toFixed(2)}–${high.toFixed(2)} · ${better}`}function renderPerformance(){const performance=data.performance,metric=$('performanceMetric').value,rows=[...(performance?.matrices?.s039||[]),...(performance?.matrices?.all_steps_mean||[])],values=rows.map(r=>r[metric]).filter(Number.isFinite);if(!performance?.ready||!values.length){renderPerformanceMatrix('s039','s039Heat','s039Meta','S039',0,1);renderPerformanceMatrix('all_steps_mean','allStepsHeat','allStepsMeta','S000–S039 平均',0,1);return}const low=Math.min(...values),high=Math.max(...values);renderPerformanceMatrix('s039','s039Heat','s039Meta','S039',low,high);renderPerformanceMatrix('all_steps_mean','allStepsHeat','allStepsMeta','S000–S039 平均',low,high)}function render(){const c=$('case').value,s=$('seed').value,r=Number($('rank').value||0),region=$('region').value,a=data.availability?.[c]?.[s]||{},top=data.summary?.top_step_block_head||[],entry=top[r],t=data.totals||{},overall=t.work_total?100*t.work_done/t.work_total:0;$('overall').innerHTML=`<div class="overall-head"><div><div class="overall-label">TOTAL COMPUTE PROGRESS</div><div class="overall-number">${overall.toFixed(1)}%</div></div><div class="overall-label">${t.work_done||0} / ${t.work_total||606} work units · Ranking ${t.ranking_final?'FROZEN':'WAITING'}</div></div><div class="overall-grid">${phase('First-frame SAM2',t.regions_done||0,t.regions_total||6,'query-frame 0 region cache')}${phase('PCK@32 Matrix',t.pck_done||0,t.pck_total||300,'40×30×24 per run')}${phase('Global Top10',t.ranking_final?1:0,1,t.ranking_final?'final ranking frozen':'unlocks at PCK 300/300')}${phase('Top10 Heatmaps',t.heatmap_done||0,t.heatmap_total||300,'case × seed object overlays')}</div>`;$('progress').innerHTML=data.progress.map(p=>`<div class="case-progress"><b>${e(p.case)}</b><br>SAM2 ${p.region?'READY':'WAIT'} · PCK ${p.pck}/50 · Top10 heatmap ${p.heatmap}/50</div>`).join('');$('status').textContent=`${data.summary?.completed_runs||0}/${data.summary?.expected_runs||300} PCK runs · ${data.summary?.final?'FINAL RANKING':'incremental ranking'}`;renderPerformance();const video=a.video?`<figure><video controls muted playsinline preload="metadata" src="/api/wan22-ti2v-legacy-pck50/video?case=${encodeURIComponent(c)}&seed=${s}"></video><figcaption>${e(c)} · seed ${s} · 40-step / 49-frame</figcaption></figure>`:`<figure><div class="pending">该 seed 视频尚未生成</div><figcaption>Generated video</figcaption></figure>`;const heat=a.heatmap&&entry?`<figure><img src="/api/wan22-ti2v-legacy-pck50/heatmap?case=${encodeURIComponent(c)}&seed=${s}&rank=${r}&region=${encodeURIComponent(region)}&v=${Date.now()}"><figcaption>Rank ${r+1} · S${String(entry.step).padStart(2,'0')} / L${String(entry.block).padStart(2,'0')} / H${String(entry.head).padStart(2,'0')} · ${e(region)} · 每帧独立色标</figcaption></figure>`:`<figure><div class="pending">等待最终 Top10 与该 seed 热力图重跑</div><figcaption>Object-query heatmap</figcaption></figure>`;$('viewer').innerHTML=video+heat;$('tables').innerHTML=`<div><h3>Step × Block × Head · Global Top10</h3>${table(top,'combo')}</div><div><h3>Block × Head · 跨全部 40 Steps</h3>${table(data.summary?.top_block_head_across_steps||[],'head')}</div>`}async function load(){data=await fetch('/api/wan22-ti2v-legacy-pck50/catalog',{cache:'no-store'}).then(r=>r.json());options($('case'),data.cases);options($('seed'),data.seeds);options($('rank'),(data.summary?.top_step_block_head||[]).map((x,i)=>({value:i,label:`#${i+1} S${String(x.step).padStart(2,'0')} L${String(x.block).padStart(2,'0')} H${String(x.head).padStart(2,'0')}`})));syncRegions();render()}$('case').addEventListener('change',()=>{syncRegions();render()});['seed','rank','region'].forEach(id=>$(id).addEventListener('change',render));$('performanceMetric').addEventListener('change',renderPerformance);$('refresh').addEventListener('click',load);$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load();
</script></body></html>'''
    page = page.replace(
        "<title>Legacy TI2V First-Latent PCK50</title>",
        '<title>Legacy TI2V First-Latent PCK50</title><link rel="icon" href="data:,">',
        1,
    )
    page = page.replace(
        "</style>",
        ".pairwise{margin-top:16px}.pairwise-note{margin:6px 0 10px;color:#6d675d;line-height:1.45}.pairwise .scroll{max-height:540px}.pairwise-table{min-width:1100px}.pairwise-table th:first-child,.pairwise-table td:first-child{text-align:left;white-space:nowrap}.pair-cell{font:11px ui-monospace,monospace;line-height:1.25}.pair-cell b{display:block;font-size:15px}.pair-cell span{display:block;color:#6d675d}.pairwise-table tr:hover td{background:#fff3d3}.download-row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}.download-row a{display:inline-block;padding:7px 10px;border:1px solid var(--line);background:#fff;color:var(--deep);font:700 11px ui-monospace,monospace;text-decoration:none}</style>",
        1,
    )
    page = page.replace(
        ".pairwise{margin-top:16px}",
        ".pairwise{padding:15px;margin:14px 0;border:1px solid var(--line);border-radius:16px;background:#fffaf0e8;box-shadow:0 13px 34px #58442b16}",
        1,
    )
    page = page.replace(
        '<section class="workspace"><h2>Object-query attention overlay</h2>',
        '<section class="pairwise"><h2>五组 PCK 排序的完整重合度与相关性</h2><p class="pairwise-note">只展示两个口径：Legacy S039 与所有 Step 平均。每个单元格为“公共 Head / Top-K”，下一行是 Jaccard 与覆盖率；Pearson/Spearman 使用同一组 720 个物理 Head 的实际 PCK@32，Spearman 对并列值使用平均秩。</p><span id="pairwiseStatus" class="comparison-status status">读取排名导出</span><div class="download-row"><a href="/downloads/pck-head-rankings.json">下载完整 JSON</a><a href="/downloads/pck-head-rankings.md">下载说明 Markdown</a></div><div class="scroll"><table class="pairwise-table"><thead><tr><th>排序组合</th><th>Top10</th><th>Top30</th><th>Top50</th><th>Top100</th><th>Pearson</th><th>Spearman</th></tr></thead><tbody id="pairwiseBody"></tbody></table></div></section><section class="workspace"><h2>Object-query attention overlay</h2>',
        1,
    )
    page = page.replace(
        "<script>",
        "<script>" + WAN22_TI2V_LEGACY_COMPARISON_SCRIPT + "</script><script>",
        1,
    )
    page = page.replace(
        "async function load(){data=await fetch('/api/wan22-ti2v-legacy-pck50/catalog',{cache:'no-store'}).then(r=>r.json());",
        "async function load(){[data,comparison]=await Promise.all([fetch('/api/wan22-ti2v-legacy-pck50/catalog',{cache:'no-store'}).then(r=>r.json()),fetch('/api/wan22-ti2v-legacy-pck50/comparison',{cache:'no-store'}).then(r=>r.json())]);",
        1,
    )
    page = page.replace(
        "syncRegions();render()}$('case')",
        "syncRegions();render();renderComparison()}$('case')",
        1,
    )
    page = page.replace(
        "tr:first-child td{background:#fff0c8;font-weight:900}",
        ".tables tr:first-child td{background:#fff0c8;font-weight:900}",
        1,
    )
    page = page.replace("<th>Rank Delta</th>", "<th>Ref - Legacy Rank</th>", 1)
    return page


def wan22_ti2v_baseline_seeds_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wan2.2-TI2V Baseline Seed Wall</title><style>
:root{--ink:#17261f;--paper:#e9e1d3;--line:#baad98;--deep:#17443a;--card:#fffaf0;--rust:#b7472f}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#d45e3745,transparent 34rem),radial-gradient(circle at 100% 2%,#288a7c42,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:16px 22px;background:#e9e1d3ef;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:5px 0;font-size:clamp(30px,4.5vw,60px);line-height:1}.lead{max-width:1100px;margin:8px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button{padding:8px 13px;border:1px solid var(--line);background:var(--card);font-weight:900;cursor:pointer}.status{font:12px ui-monospace,monospace}main{width:min(100% - 18px,1900px);margin:auto;padding:18px 0 70px}.matrix-head,.seed-row{display:grid;grid-template-columns:150px repeat(2,minmax(360px,1fr));gap:10px}.matrix-head{position:sticky;top:174px;z-index:5;padding:9px;background:var(--deep);color:#fff;border-radius:13px 13px 0 0}.matrix-head div{padding:6px;font-weight:900;text-align:center}.seed-row{padding:10px;margin-top:10px;border:1px solid var(--line);border-radius:14px;background:#fff9eddb;box-shadow:0 12px 34px #57442915}.seed-label{display:flex;align-items:center;justify-content:center;font:900 18px ui-monospace,monospace;color:var(--deep)}figure{margin:0;border:1px solid #d4c8b5;background:#fff;padding:8px}video{display:block;width:100%;aspect-ratio:16/9;background:#111}figcaption{padding:7px 3px 2px;font-weight:900}.pending{display:grid;place-items:center;min-height:210px;color:#746c60;background:#f2ede4}.badge{display:inline-block;margin-left:8px;padding:3px 7px;border-radius:99px;background:#327b63;color:#fff;font-size:10px}@media(max-width:850px){header{position:static}.matrix-head{display:none}.seed-row{grid-template-columns:1fr}.seed-label{justify-content:flex-start;padding:5px}.seed-label:before{content:'Seed ';margin-right:5px}}
</style></head><body><header><a href="/">返回总览</a><h1>Wan2.2-TI2V<br>Baseline Seed Wall</h1><p class="lead">标准 DiffSynth Wan2.2-TI2V-5B pipeline；每个样本仅输入 JSON prompt 与首帧 input_image，不传 context_video，不加载 LoRA，不捕获 attention。</p><div class="tools"><button id="replay">重新播放全部</button><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><div class="matrix-head"><div>Seed</div><div>40-Step · 49 Frames</div><div>10-Step · 49 Frames</div></div><div id="rows"></div></main><script>
function card(seed,steps,ready){return ready?`<figure><video controls muted playsinline preload="metadata" src="/api/object-query-step-alignment/video?model=baseline&seed=${seed}&steps=${steps}&kind=original"></video><figcaption>${steps}-Step Official TI2V <span class="badge">READY</span></figcaption></figure>`:`<figure><div class="pending">${steps}-Step 尚未生成</div><figcaption>${steps}-Step Official TI2V</figcaption></figure>`}async function load(){const d=await fetch('/api/object-query-step-alignment/catalog',{cache:'no-store'}).then(r=>r.json()),m=d.models.find(x=>x.id==='baseline');let ready=0,total=0;document.getElementById('rows').innerHTML=d.seeds.map(seed=>{const v=m?.videos?.[seed]||{};ready+=Number(!!v?.[40]?.original)+Number(!!v?.[10]?.original);total+=2;return `<section class="seed-row"><div class="seed-label">${seed}</div>${card(seed,40,!!v?.[40]?.original)}${card(seed,10,!!v?.[10]?.original)}</section>`}).join('');document.getElementById('status').textContent=`${d.case} · ${ready}/${total} videos ready`}document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load();
</script></body></html>'''


def object_query_step_alignment_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>10-Step × 40-Step Attention Alignment</title><style>
:root{--paper:#e8e0d2;--ink:#172720;--deep:#153f35;--line:#b9ad98;--card:#fffaf0;--rust:#b9472f;--gold:#d5a237;--sea:#247d82}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#d96b3d42,transparent 32rem),radial-gradient(circle at 98% 2%,#24898240,transparent 38rem),linear-gradient(135deg,#eee7da,#ded4c3);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:9;padding:16px 24px;background:#e8e0d2ed;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:5px 0;font-size:clamp(29px,4.4vw,58px);line-height:1}header p{margin:7px 0;max-width:1100px}.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,.download,select{border:1px solid var(--line);background:#fffaf0;padding:8px 13px;color:var(--ink);font-weight:900;text-decoration:none;cursor:pointer}.status{font:12px ui-monospace,monospace}main{width:min(100% - 20px,2300px);margin:auto;padding:20px 0 70px}.model{margin:0 0 24px;border:1px solid var(--line);border-radius:18px;background:#fdf8eddd;overflow:hidden;box-shadow:0 16px 40px #58472b17}.model-head{padding:17px 20px;background:linear-gradient(100deg,var(--deep),#276d61);color:#fff}.model-head h2{margin:0;font-size:28px}.model-head p{margin:5px 0 0}.progress{display:grid;grid-template-columns:repeat(6,minmax(140px,1fr));gap:8px;padding:12px}.seed{padding:8px;border:1px solid #d0c5b3;background:#fff;border-radius:9px;font:11px ui-monospace,monospace}.seed.done{border-color:#3d8b70;background:#e4f2e8}.section-title{padding:9px 15px;margin:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:#eee4d3;font-size:15px;text-transform:uppercase;letter-spacing:.08em}.grid{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:11px;padding:12px}.video-grid{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:10px;padding:12px;overflow:auto}.video-card{margin:0;border:1px solid #d6cab7;background:#fff;padding:8px}.video-card video{display:block;width:100%;background:#111;aspect-ratio:16/9}.video-card figcaption{padding:7px 3px 2px;font-weight:900}.plot{margin:0;border:1px solid #d6cab7;background:#fff;padding:8px}.plot img{width:100%;display:block}.plot figcaption{padding:7px 4px 2px;font-weight:900}.curve{grid-column:1/-1}.curve img{max-height:620px;object-fit:contain}.pending{padding:35px;text-align:center;color:#766e61}.scroll{overflow:auto;margin:12px;border:1px solid var(--line);background:#fff}table{border-collapse:collapse;width:100%;min-width:680px;font-variant-numeric:tabular-nums}th,td{padding:8px 10px;border-bottom:1px solid #ddd4c5;text-align:center}th{background:var(--deep);color:#fff}td:first-child{font-weight:900}.downloads{display:flex;gap:8px;flex-wrap:wrap;padding:4px 12px 15px}@media(max-width:1100px){.video-grid{grid-template-columns:repeat(2,minmax(260px,1fr))}}@media(max-width:900px){header{position:static}.grid,.video-grid{grid-template-columns:1fr}.progress{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><a href="/">返回总览</a><h1>10-Step × 40-Step<br>Object-Query Alignment</h1><p>同一物理 Head 在两种去噪日程间进行 cosine 对齐。主结果先逐 Head 计算相似度再对 TopN、Object A/B、CFG 分支与六个 seed 宏平均；辅助结果先平均 TopN attention map 再计算 cosine。</p><div class="tools"><label>Seed <select id="seed"></select></label><button id="replay">重新播放本页视频</button><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main id="main"></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),ns=[30,50,100],labels={mean_per_head_cosine:'逐 Head cosine 后平均',cosine_of_mean_map:'TopN Mean Map cosine'};let data=null;function img(model,name,title,wide=false){return `<figure class="plot ${wide?'curve':''}"><img loading="lazy" src="/api/object-query-step-alignment/image?model=${model}&name=${name}&v=${Date.now()}"><figcaption>${e(title)}</figcaption></figure>`}function table(model){const rows=(model.matches||[]).filter(r=>r.aggregation==='mean_per_head_cosine'),map=new Map(rows.map(r=>[`${r.step10}-${r.top_n}`,r]));return `<div class="scroll"><table><thead><tr><th>10-Step</th>${ns.map(n=>`<th>Top${n} 最佳 40-Step</th>`).join('')}</tr></thead><tbody>${[...Array(10)].map((_,s)=>`<tr><td>S${String(s).padStart(2,'0')}</td>${ns.map(n=>{const r=map.get(`${s}-${n}`);return `<td>${r?`S${String(r.best_step40).padStart(2,'0')} · ${Number(r.cosine).toFixed(4)}`:'—'}</td>`}).join('')}</tr>`).join('')}</tbody></table></div>`}function videos(m){const seed=document.getElementById('seed').value,availability=m.videos?.[seed]||{},specs=[[40,'original','40-Step Original'],[40,'probe','40-Step Top100 No-op Probe'],[10,'original','10-Step Original'],[10,'probe','10-Step Top100 No-op Probe']];return `<h3 class="section-title">Generated Videos · Seed ${seed}</h3><div class="video-grid">${specs.map(([steps,kind,label])=>availability?.[steps]?.[kind]?`<figure class="video-card"><video controls muted playsinline preload="metadata" src="/api/object-query-step-alignment/video?model=${m.id}&seed=${seed}&steps=${steps}&kind=${kind}"></video><figcaption>${label}</figcaption></figure>`:`<figure class="video-card"><div class="pending">${label}<br>${kind==='probe'&&!m.attention_capture?'官方纯推理不执行 Attention Probe':'尚未生成'}</div></figure>`).join('')}</div>`}function renderModel(m){const progress=m.progress.map(p=>`<div class="seed ${p.complete?'done':''}">seed ${p.seed}<br>40-step ${p.steps40}/80<br>10-step ${p.steps10}/20</div>`).join(''),videoSection=videos(m),head=`<div class="model-head"><h2>${e(m.label)}</h2><p>${e(m.detail)}</p></div>${m.attention_capture?`<div class="progress">${progress}</div>`:''}${videoSection}`;if(!m.ready)return `<section class="model">${head}<div class="pending">${m.attention_capture?'Attention 对齐统计生成中。视频完成后会先于统计结果显示。':'本模型仅运行官方首帧 TI2V 推理，不捕获或统计 attention。'}</div></section>`;const primary=ns.map(n=>img(m.id,`top${n}_mean_per_head_cosine.png`,`Top${n} · ${labels.mean_per_head_cosine}`)).join(''),aux=ns.map(n=>img(m.id,`top${n}_cosine_of_mean_map.png`,`Top${n} · ${labels.cosine_of_mean_map}`)).join(''),downloads=m.downloads.map(name=>`<a class="download" href="/api/object-query-step-alignment/download?model=${m.id}&name=${name}">${e(name)}</a>`).join('');return `<section class="model">${head}<h3 class="section-title">Primary · Mean of per-head cosine</h3><div class="grid">${primary}</div><h3 class="section-title">Auxiliary · Cosine after TopN map averaging</h3><div class="grid">${aux}</div><h3 class="section-title">Best 40-step mapping curve</h3><div class="grid">${img(m.id,'best_match_curves.png','Top30 / Top50 / Top100 最佳匹配路径',true)}</div><h3 class="section-title">Primary best matches</h3>${table(m)}<div class="downloads">${downloads}</div></section>`}function render(){document.getElementById('main').innerHTML=data.models.map(renderModel).join('')}async function load(){data=await fetch('/api/object-query-step-alignment/catalog',{cache:'no-store'}).then(r=>r.json());const select=document.getElementById('seed'),current=select.value;if(!select.options.length)select.innerHTML=data.seeds.map(seed=>`<option value="${seed}">${seed}</option>`).join('');if(current&&data.seeds.map(String).includes(current))select.value=current;document.getElementById('status').textContent=`${data.case} · ${data.models.filter(m=>m.ready).length}/${data.models.length} attention models ready`;render()}document.getElementById('seed').addEventListener('change',render);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));document.getElementById('refresh').addEventListener('click',load);load();
</script></body></html>'''


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


def _attention_lora_test5_10seed_manifest():
    path = ATTENTION_LORA_TEST5_10SEED_ROOT / "experiment_manifest.json"
    if not path.is_file():
        return {"cases": [], "seeds": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"cases": [], "seeds": []}
    return payload if isinstance(payload, dict) else {"cases": [], "seeds": []}


def _attention_lora_test5_10seed_cases():
    manifest = _attention_lora_test5_10seed_manifest()
    return [
        str(record["case_key"])
        for record in manifest.get("cases", [])
        if isinstance(record, dict) and record.get("case_key")
    ]


def _attention_lora_test5_10seed_seeds():
    manifest = _attention_lora_test5_10seed_manifest()
    return [int(seed) for seed in manifest.get("seeds", [])]


def attention_lora_test5_10step_asset(
    case_key: str, requested_seed: str, profile: str, group: str
):
    if case_key not in set(_attention_lora_test5_10seed_cases()):
        return None
    try:
        seed = int(requested_seed)
    except ValueError:
        return None
    if seed not in ATTENTION_LORA_TEST5_2SEED_10STEP_SEEDS:
        return None
    seed_root = (
        ATTENTION_LORA_TEST5_2SEED_10STEP_ROOT
        / "cases" / case_key / "seeds" / f"seed_{seed:06d}"
    )
    if profile == "original" and group == "original":
        return seed_root / "original.mp4"
    if profile not in {item[0] for item in SEED_SWEEP_PROFILES}:
        return None
    if group not in {"top100", "bottom100"}:
        return None
    return (
        seed_root / "all_steps" / profile / "videos" / "lora" / "cases"
        / case_key / f"{group}_steps_00_40.mp4"
    )


def attention_lora_test5_10seed_asset(
    case_key: str,
    seed_text: str,
    stage: str,
    profile: str,
    group: str,
    name: str,
):
    if case_key not in set(_attention_lora_test5_10seed_cases()):
        return None
    try:
        seed = int(seed_text)
    except ValueError:
        return None
    if seed not in _attention_lora_test5_10seed_seeds():
        return None
    seed_root = (
        ATTENTION_LORA_TEST5_10SEED_ROOT
        / "cases"
        / case_key
        / "seeds"
        / f"seed_{seed:06d}"
    )
    if profile == "original" and stage == "original" and group == "original":
        return seed_root / "original.mp4"
    if (
        stage not in {"all_steps", "steps00_09"}
        or profile not in {item[0] for item in SEED_SWEEP_PROFILES}
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
        / case_key
        / f"{group}_{suffix}.mp4"
    )


def attention_lora_test5_10seed_catalog(requested_case: str = ""):
    cases = _attention_lora_test5_10seed_cases()
    seeds = _attention_lora_test5_10seed_seeds()
    case_key = requested_case if requested_case in cases else (cases[0] if cases else "")
    seed_catalogs = []
    for seed in seeds:
        original = attention_lora_test5_10seed_asset(
            case_key, str(seed), "original", "original", "original", ""
        )
        records = []
        for stage in ("all_steps", "steps00_09"):
            capture_step = 39 if stage == "all_steps" else 9
            for profile, label in SEED_SWEEP_PROFILES:
                run_root = (
                    ATTENTION_LORA_TEST5_10SEED_ROOT
                    / "cases"
                    / case_key
                    / "seeds"
                    / f"seed_{seed:06d}"
                    / stage
                    / profile
                )
                for group in ("top100", "bottom100"):
                    video = attention_lora_test5_10seed_asset(
                        case_key, str(seed), stage, profile, group, ""
                    )
                    metadata_path = next(
                        iter(
                            sorted(
                                (run_root / "heatmaps").glob(
                                    f"*__{case_key}__{group}__*step{capture_step:02d}.json"
                                )
                            )
                        ),
                        None,
                    )
                    metadata = {}
                    if metadata_path is not None:
                        try:
                            metadata = json.loads(
                                metadata_path.read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError):
                            metadata = {}
                    all_token = str(metadata.get("all_token_image", ""))
                    frame = str(metadata.get("frame_image", ""))
                    all_token_path = (
                        attention_lora_test5_10seed_asset(
                            case_key, str(seed), stage, profile, group, all_token
                        )
                        if all_token
                        else None
                    )
                    frame_path = (
                        attention_lora_test5_10seed_asset(
                            case_key, str(seed), stage, profile, group, frame
                        )
                        if frame
                        else None
                    )
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
                        }
                    )
        seed_catalogs.append(
            {
                "seed": seed,
                "original_ready": bool(original and original.is_file()),
                "ready_records": sum(record["video_ready"] for record in records),
                "records": records,
            }
        )

    case_progress = []
    complete_profiles = 0
    expected_profiles = len(cases) * len(seeds) * 16
    for candidate in cases:
        candidate_complete = 0
        for seed in seeds:
            seed_root = (
                ATTENTION_LORA_TEST5_10SEED_ROOT
                / "cases"
                / candidate
                / "seeds"
                / f"seed_{seed:06d}"
            )
            candidate_complete += sum(1 for _ in seed_root.glob("*/*/complete"))
        complete_profiles += candidate_complete
        case_progress.append(
            {
                "case": candidate,
                "complete_profiles": candidate_complete,
                "expected_profiles": len(seeds) * 16,
                "metrics_complete": (
                    ATTENTION_LORA_TEST5_10SEED_METRIC_ROOT
                    / "cases"
                    / candidate
                    / "METRICS_COMPLETE"
                ).is_file(),
            }
        )
    return {
        "case": case_key,
        "cases": cases,
        "seeds": seeds,
        "profiles": [
            {"id": profile, "label": label}
            for profile, label in SEED_SWEEP_PROFILES
        ],
        "seed_catalogs": seed_catalogs,
        "case_progress": case_progress,
        "complete_profiles": complete_profiles,
        "expected_profiles": expected_profiles,
        "complete_intervention_videos": complete_profiles * 2,
        "expected_intervention_videos": expected_profiles * 2,
        "metrics_complete_cases": sum(
            record["metrics_complete"] for record in case_progress
        ),
    }


def attention_lora_test5_10seed_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wan+LoRA Test5 · 20 Cases × 2 Seeds</title><style>
:root{--ink:#1e2824;--muted:#626b66;--paper:#f2f1ed;--card:#fff;--line:#cfd4d1;--green:#166a5d;--red:#b24432;--blue:#315f78;--gold:#a87522}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Arial,"Noto Sans SC",sans-serif}header{position:sticky;top:0;z-index:10;padding:12px 18px;background:#fffffff2;border-bottom:1px solid var(--line)}header a{color:var(--green);font-weight:700}h1{margin:4px 0;font-size:24px;letter-spacing:0}header p{margin:3px 0;color:var(--muted)}.tools,.families,.conditions,.summary{display:flex;gap:7px;align-items:center;flex-wrap:wrap}select,button{padding:6px 9px;border:1px solid #aeb7b2;border-radius:4px;background:#fff;color:var(--ink);font-weight:700}.status{font:12px ui-monospace,monospace;color:var(--muted)}.summary{margin-top:7px}.pill{padding:4px 7px;border:1px solid var(--line);background:#f7f8f7;font:11px ui-monospace,monospace}.filters,main{width:min(2300px,calc(100% - 16px));margin:auto}.filters{padding:10px 0;border-bottom:1px solid var(--line)}.filter-row{display:flex;align-items:flex-start;gap:10px;margin:5px 0}.filter-row b{min-width:86px;font-size:12px;padding-top:7px}.experiment.active{background:#213c34;color:#fff;border-color:#213c34}.condition.active{background:var(--blue);color:#fff;border-color:var(--blue)}.condition.top.active{background:var(--red);border-color:var(--red)}.condition.bottom.active{background:var(--green);border-color:var(--green)}main{padding:10px 0 64px}.selection{display:flex;justify-content:space-between;align-items:end;gap:10px;margin-bottom:8px}.selection h2{margin:0;font-size:19px}.selection p{margin:3px 0}.seed-grid{display:grid;grid-template-columns:repeat(2,minmax(420px,1fr));gap:8px}.card{padding:9px;border:1px solid var(--line);border-radius:6px;background:var(--card)}.card.top{border-top:4px solid var(--red)}.card.bottom{border-top:4px solid var(--green)}.card h3{display:flex;justify-content:space-between;margin:0 0 7px;font:700 13px ui-monospace,monospace}.video-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.video-pair figure{margin:0;min-width:0}.video-pair figcaption{margin-bottom:4px;font-size:11px;font-weight:700}.card video,.card img{display:block;width:100%;max-height:220px;object-fit:contain;background:#111}.pending{min-height:132px;display:grid;place-items:center;border:1px dashed var(--line);color:var(--muted);font-size:12px}.experiment-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:8px}.experiment-block{padding:8px;border:1px solid var(--line);border-left:4px solid var(--blue);background:#f8faf9}.experiment-block h4{margin:0 0 6px;font-size:13px;line-height:1.2}.seed-original{margin:0;width:min(430px,32vw)}.seed-original figcaption{margin-bottom:4px;font-size:11px;font-weight:700}details{margin-top:5px}details summary{cursor:pointer;font-size:12px;font-weight:700}.maps{display:grid;gap:5px;margin-top:5px}.replay{position:fixed;right:18px;bottom:18px;z-index:20;background:#213c34;color:#fff;border:0;padding:10px 14px}@media(max-width:1200px){.experiment-grid{grid-template-columns:1fr}.seed-original{width:100%}}@media(max-width:1000px){header{position:static}.seed-grid,.video-pair{grid-template-columns:1fr}.filter-row{display:block}.filter-row b{display:block}}
</style></head><body><button class="replay" id="replay">重新播放当前页</button><header><a href="/">返回总览</a><h1>Wan+LoRA · Test5 20 Cases × 2 Seeds</h1><p>仅展示实际运行的 seed 90094/35075 · 40-step/10-step 对照 · 8 profiles × Top/Bottom100 · 49 frames</p><div class="tools"><label>Case <select id="case"></select></label><button id="refresh">刷新进度</button><span id="status" class="status">读取中</span></div><div id="summary" class="summary"></div></header><section class="filters"><div class="filter-row"><b>实验</b><div id="experiments" class="families"></div></div><div class="filter-row"><b>阶段 / Head</b><div id="conditions" class="conditions"></div></div></section><main><div class="selection"><div><h2 id="title"></h2><p id="subtitle"></p></div><span id="count" class="status"></span></div><section id="grid" class="seed-grid"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let caseKey=q.get('case')||'',profile=q.get('experiment')||'alpha090',stage=q.get('stage')||'all_steps',group=q.get('group')||'top100',data=null,generation=Date.now();
const labels={alpha090:'α = 0.9',alpha150:'α = 1.5',zero:'A = 0',uniform:'A = 1/N_K',temporal_causal:'Temporal Causal',strict_past:'Strict Past Only',strict_future:'Strict Future Only',head_output_zero:'Head Output Zero'};
function sync(){const u=new URL(location.href);u.searchParams.set('case',caseKey);u.searchParams.set('experiment',profile);u.searchParams.set('stage',stage);u.searchParams.set('group',group);history.replaceState(null,'',u)}
function video(seed,s,p,g){return `/api/attention-additive-lora-test5-10seed/video?case=${encodeURIComponent(caseKey)}&seed=${seed}&stage=${s}&profile=${p}&group=${g}&v=${generation}`}
function video10(seed,p,g){return '/api/attention-additive-lora-test5-10seed/video10?case='+encodeURIComponent(caseKey)+'&seed='+seed+'&profile='+p+'&group='+g+'&v='+generation}
function image(seed,r,name){return `/api/attention-additive-lora-test5-10seed/image?case=${encodeURIComponent(caseKey)}&seed=${seed}&stage=${r.stage}&profile=${r.profile}&group=${r.group}&name=${encodeURIComponent(name)}&v=${generation}`}
function buttons(){document.getElementById('experiments').innerHTML=Object.entries(labels).map(([id,label])=>`<button class="experiment ${id===profile?'active':''}" data-profile="${id}">${e(label)}</button>`).join('');document.getElementById('conditions').innerHTML=[['all_steps','S000-S039'],['steps00_09','S000-S009']].map(([id,label])=>`<button class="condition ${id===stage?'active':''}" data-stage="${id}">${label}</button>`).join('')+[['top100','Top100'],['bottom100','Bottom100']].map(([id,label])=>`<button class="condition ${id} ${id===group?'active':''}" data-group="${id}">${label}</button>`).join('');document.querySelectorAll('[data-profile]').forEach(b=>b.onclick=()=>{profile=b.dataset.profile;sync();buttons();render()});document.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>{stage=b.dataset.stage;sync();buttons();render()});document.querySelectorAll('[data-group]').forEach(b=>b.onclick=()=>{group=b.dataset.group;sync();buttons();render()})}
function render(){if(!data)return;const select=document.getElementById('case');select.innerHTML=data.cases.map(value=>`<option value="${e(value)}">${e(value)}</option>`).join('');select.value=caseKey;const label=labels[profile],stageLabel=stage==='all_steps'?'S000-S039':'S000-S009';document.getElementById('title').textContent=`${label} · ${group.toUpperCase()} · ${stageLabel}`;document.getElementById('subtitle').textContent=caseKey;let ready=0,originals=0;document.getElementById('grid').innerHTML=data.seed_catalogs.map(seedData=>{const r=seedData.records.find(item=>item.stage===stage&&item.profile===profile&&item.group===group),experimentReady=Boolean(r&&r.video_ready);if(experimentReady)ready++;if(seedData.original_ready)originals++;const tenstepScheduled=[90094,35075].includes(Number(seedData.seed)),tenstep=tenstepScheduled?'<div class="tenstep-media"><video controls preload="none" playsinline src="'+video10(seedData.seed,profile,group)+'" onerror="this.hidden=true;this.nextElementSibling.hidden=false"></video><div class="pending" hidden>10-step 生成中</div></div>':'<div class="pending">该历史 seed 未安排 10-step</div>',original=seedData.original_ready?`<video controls preload="none" playsinline src="${video(seedData.seed,'original','original','original')}"></video>`:'<div class="pending">Original 等待生成</div>',experiment=experimentReady?`<video controls preload="none" playsinline src="${video(seedData.seed,stage,profile,group)}"></video>`:'<div class="pending">干预视频等待生成</div>',maps=r&&r.heatmap_ready?`<details><summary>Q@K 干预前后热力图</summary><div class="maps"><img loading="lazy" src="${image(seedData.seed,r,r.all_token)}"><img loading="lazy" src="${image(seedData.seed,r,r.frame)}"></div></details>`:`<div class="status">${r&&r.heatmap_expected?'热力图等待生成':'该配置不改变 Attention'}</div>`;return `<article class="card ${group==='top100'?'top':'bottom'}"><h3><span>Seed ${seedData.seed}</span><span>${experimentReady?'READY':'PENDING'}</span></h3><div class="video-pair"><figure><figcaption>Original · 同 Seed</figcaption>${original}</figure><figure><figcaption>${e(label)} · ${group.toUpperCase()} · ${stageLabel}</figcaption>${experiment}</figure><figure><figcaption>10-step · ${e(label)} · ${group.toUpperCase()}</figcaption>${tenstep}</figure></div>${maps}</article>`}).join('');document.getElementById('count').textContent=`干预 ${ready}/${data.seeds.length} · Original ${originals}/${data.seeds.length}`;document.getElementById('status').textContent=`${data.complete_intervention_videos}/${data.expected_intervention_videos} intervention videos`;document.getElementById('summary').innerHTML=`<span class="pill">Profiles ${data.complete_profiles}/${data.expected_profiles}</span><span class="pill">Metrics cases ${data.metrics_complete_cases}/${data.cases.length}</span><span class="pill">Cases ${data.cases.length}</span><span class="pill">Seeds ${data.seeds.length}</span>`;buttons()}
function render(){if(!data)return;const select=document.getElementById('case');select.innerHTML=data.cases.map(value=>`<option value="${e(value)}">${e(value)}</option>`).join('');select.value=caseKey;const stageLabel=stage==='all_steps'?'S000-S039':'S000-S009',profileEntries=Object.entries(labels),shownSeeds=data.seed_catalogs;let ready40=0;document.getElementById('title').textContent=`全部实验 · ${group.toUpperCase()} · ${stageLabel}`;document.getElementById('subtitle').textContent=caseKey;document.getElementById('grid').style.gridTemplateColumns='1fr';document.getElementById('grid').innerHTML=shownSeeds.map(seedData=>{const original=seedData.original_ready?`<video controls preload="none" playsinline src="${video(seedData.seed,'original','original','original')}"></video>`:'<div class="pending">Original 等待生成</div>',rows=profileEntries.map(([id,label])=>{const r=seedData.records.find(item=>item.stage===stage&&item.profile===id&&item.group===group),experimentReady=Boolean(r&&r.video_ready);if(experimentReady)ready40++;const forty=experimentReady?`<video controls preload="none" playsinline src="${video(seedData.seed,stage,id,group)}"></video>`:'<div class="pending">40-step 等待生成</div>',ten='<div class="tenstep-media"><video controls preload="none" playsinline src="'+video10(seedData.seed,id,group)+'" onerror="this.hidden=true;this.nextElementSibling.hidden=false"></video><div class="pending" hidden>10-step 等待生成</div></div>',maps=r&&r.heatmap_ready?`<details><summary>40-step Q@K 干预前后热力图</summary><div class="maps"><img loading="lazy" src="${image(seedData.seed,r,r.all_token)}"><img loading="lazy" src="${image(seedData.seed,r,r.frame)}"></div></details>`:`<div class="status">${r&&r.heatmap_expected?'热力图等待生成':'该配置不改变 Attention'}</div>`;return `<section class="experiment-block"><h4>${e(label)} · ${group.toUpperCase()} · ${stageLabel}</h4><div class="video-pair"><figure><figcaption>40-step</figcaption>${forty}</figure><figure><figcaption>10-step</figcaption>${ten}</figure></div>${maps}</section>`}).join('');return `<article class="card ${group==='top100'?'top':'bottom'}"><h3><span>Seed ${seedData.seed}</span><span>全部 ${profileEntries.length} 种实验</span></h3><figure class="seed-original"><figcaption>Wan+LoRA Original · 同 Seed</figcaption>${original}</figure><div class="experiment-grid">${rows}</div></article>`}).join('');document.getElementById('count').textContent=`40-step ${ready40}/${shownSeeds.length*profileEntries.length} · Seeds ${shownSeeds.length}`;document.getElementById('status').textContent=`${data.complete_intervention_videos}/${data.expected_intervention_videos} intervention videos`;document.getElementById('summary').innerHTML=`<span class="pill">同页实验 ${profileEntries.length}</span><span class="pill">Metrics cases ${data.metrics_complete_cases}/${data.cases.length}</span><span class="pill">Cases ${data.cases.length}</span><span class="pill">Seeds ${shownSeeds.length}</span>`;buttons();document.getElementById('experiments').closest('.filter-row').hidden=true}
async function load(){generation=Date.now();document.getElementById('status').textContent='正在扫描产物';const url=`/api/attention-additive-lora-test5-10seed/catalog?case=${encodeURIComponent(caseKey)}&v=${generation}`;data=await fetch(url,{cache:'no-store'}).then(r=>r.json());caseKey=data.case;data.seed_catalogs=data.seed_catalogs.filter(x=>[90094,35075].includes(Number(x.seed)));data.seeds=data.seeds.filter(x=>[90094,35075].includes(Number(x)));sync();render()}
document.getElementById('case').addEventListener('change',ev=>{caseKey=ev.target.value;sync();load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));buttons();load();
</script></body></html>'''


def attention_lora_seed_sweep_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wan+LoRA 50-Seed Sweep</title><style>
:root{--ink:#1a2822;--paper:#eee8dc;--card:#fffdf8;--line:#bdb3a0;--red:#ae432f;--green:#17695d;--dark:#19362d;--gold:#bb7b28;--blue:#315f78}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 8% 0,#eaa76755,transparent 34rem),radial-gradient(circle at 96% 3%,#52977c55,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:14px 24px;background:#eee8df2;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools,.families,.conditions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}button{padding:8px 12px;border:1px solid var(--line);border-radius:7px;background:#fffdf8;color:var(--ink);font-weight:900;cursor:pointer}.status{font:12px ui-monospace,monospace;color:#53635b}.filters{width:min(2300px,calc(100% - 20px));margin:16px auto 0;display:grid;gap:10px}.filter-row{padding:12px;border:1px solid var(--line);border-radius:13px;background:#f8f3e9}.filter-row h3{margin:0 0 8px;font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#696054}.family{display:flex;align-items:center;gap:7px;padding:4px 8px 4px 4px;border-right:1px solid var(--line)}.family b{font-size:11px;color:#6e675e}.experiment.active{background:var(--dark);border-color:var(--dark);color:#fff}.condition.active{background:var(--blue);border-color:var(--blue);color:#fff}.condition.top.active{background:var(--red);border-color:var(--red)}.condition.bottom.active{background:var(--green);border-color:var(--green)}main{width:min(2300px,calc(100% - 20px));margin:auto;padding:18px 0 80px}.selection{display:flex;justify-content:space-between;align-items:end;gap:15px;margin-bottom:12px}.selection h2{margin:0;font-size:clamp(22px,3vw,35px)}.selection p{margin:4px 0 0;color:#685f54}.seed-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:11px}.card{padding:10px;background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 8px 25px #4d3f2820}.card.top{border-top:5px solid var(--red)}.card.bottom{border-top:5px solid var(--green)}.card.focus{outline:4px solid var(--gold);outline-offset:2px}.card h3{display:flex;justify-content:space-between;gap:8px;margin:0 0 8px;font:900 15px ui-monospace,monospace}.card video,.card img{display:block;width:100%;background:#151916;border:1px solid var(--line);border-radius:7px}.video-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:7px}.video-pair figure{margin:0;min-width:0}.video-pair figcaption{margin-bottom:6px;font-size:12px;font-weight:900}.pill{display:inline-block;margin:3px;padding:4px 7px;border-radius:99px;background:#e9e2d4;font:10px ui-monospace,monospace}.pending{min-height:170px;display:grid;place-items:center;border:1px dashed var(--line);border-radius:7px;color:#746e62}.maps{display:grid;gap:7px;margin-top:8px}.maps summary{cursor:pointer;font-weight:900;color:var(--blue)}.maps img{margin-top:7px}.note{font-size:11px;color:#665f55;margin-top:7px}.replay{position:fixed;right:20px;bottom:20px;z-index:30;border:0;border-radius:99px;background:var(--dark);color:#fff;padding:13px 19px;box-shadow:0 8px 22px #162b25aa}@media(max-width:800px){header{position:static}.family{border:0}.seed-grid,.video-pair{grid-template-columns:1fr}main,.filters{width:calc(100% - 10px)}.selection{display:block}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/attention-additive-lora-seed-sweep-metrics?v=1">指标表</a><h1>Wan+LoRA · 50-Seed Sweep</h1><p>0613pybullet_sample_001460_w002 · 40步 · 49帧 · 同一实验的不同 seed 聚合展示</p><div class="tools"><button id="refresh">手动刷新当前实验</button><span class="status" id="status">读取中</span></div></header><section class="filters"><div class="filter-row"><h3>按实验种类选择</h3><div id="experiments" class="families"></div></div><div id="conditionRow" class="filter-row"><h3>实验条件</h3><div id="conditions" class="conditions"></div></div></section><main><div class="selection"><div><h2 id="title">实验结果</h2><p id="subtitle"></p></div><span class="status" id="count"></span></div><section id="grid" class="seed-grid"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),params=new URL(location.href).searchParams,focusSeed=params.get('seed')||'';
const families=[['Baseline',[['original','Original / No-op']] ],['Attention 加性扰动',[['alpha090','α = 0.9'],['alpha150','α = 1.5']]],['概率替换',[['zero','A = 0'],['uniform','A = 1/N_K']]],['时序 Mask',[['temporal_causal','Temporal Causal'],['strict_past','Strict Past Only'],['strict_future','Strict Future Only']]],['输出消融',[['head_output_zero','Head Output Zero']]]];
const profileLabels=Object.fromEntries(families.flatMap(x=>x[1]));let profile=params.get('experiment')||'alpha090',stage=params.get('stage')||'all_steps',group=params.get('group')||'top100',catalogs=[],generation=0;
const video=(seed,s,p,g)=>`/api/attention-additive-lora-seed-sweep/video?seed=${encodeURIComponent(seed)}&stage=${s}&profile=${p}&group=${g}`;const image=(seed,r,name)=>`/api/attention-additive-lora-seed-sweep/image?seed=${encodeURIComponent(seed)}&stage=${r.stage}&profile=${r.profile}&group=${r.group}&name=${encodeURIComponent(name)}`;
function syncUrl(){const u=new URL(location.href);u.searchParams.set('experiment',profile);if(profile!=='original'){u.searchParams.set('stage',stage);u.searchParams.set('group',group)}history.replaceState(null,'',u)}
function renderButtons(){document.getElementById('experiments').innerHTML=families.map(([family,items])=>`<div class="family"><b>${e(family)}</b>${items.map(([id,label])=>`<button class="experiment ${id===profile?'active':''}" data-profile="${id}">${e(label)}</button>`).join('')}</div>`).join('');document.getElementById('conditions').innerHTML=[['all_steps','S000-S039'],['steps00_09','S000-S009']].map(([id,label])=>`<button class="condition ${id===stage?'active':''}" data-stage="${id}">${label}</button>`).join('')+[['top100','Top100'],['bottom100','Bottom100']].map(([id,label])=>`<button class="condition ${id} ${id===group?'active':''}" data-group="${id}">${label}</button>`).join('');document.getElementById('conditionRow').hidden=profile==='original';document.querySelectorAll('[data-profile]').forEach(b=>b.onclick=()=>{profile=b.dataset.profile;syncUrl();renderButtons();render()});document.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>{stage=b.dataset.stage;syncUrl();renderButtons();render()});document.querySelectorAll('[data-group]').forEach(b=>b.onclick=()=>{group=b.dataset.group;syncUrl();renderButtons();render()})}
function heatmaps(seed,r){if(profile==='original')return '';if(r.heatmap_ready)return `<details class="maps"><summary>扰动前后热力图</summary><img loading="lazy" src="${image(seed,r,r.all_token)}"><img loading="lazy" src="${image(seed,r,r.frame)}"></details>`;return r.heatmap_expected?'<div class="note">热力图等待生成</div>':'<div class="note">Attention 不变，仅 Head 输出置零</div>'}
function render(){renderButtons();const label=profileLabels[profile]||profile,stageLabel=stage==='all_steps'?'S000-S039':'S000-S009',groupLabel=group.toUpperCase();document.getElementById('title').textContent=profile==='original'?label:`${label} · ${groupLabel} · ${stageLabel}`;document.getElementById('subtitle').textContent=`Case 0613pybullet_sample_001460_w002 · ${catalogs.length} seeds 放在同一实验分组内`;let ready=0,originalReady=0;document.getElementById('grid').innerHTML=catalogs.map(d=>{const seed=String(d.selected_seed),r=profile==='original'?null:d.records.find(x=>x.stage===stage&&x.group===group&&x.profile===profile),isReady=profile==='original'?d.original_ready:Boolean(r&&r.video_ready),baseReady=Boolean(d.original_ready);if(isReady)ready++;if(baseReady)originalReady++;const originalVideo=baseReady?`<video controls preload="none" playsinline src="${video(seed,'original','original','original')}"></video>`:'<div class="pending">Original 生成中</div>',experimentVideo=isReady?`<video controls preload="none" playsinline src="${video(seed,stage,profile,group)}"></video>`:'<div class="pending">实验视频生成中</div>',media=profile==='original'?originalVideo:`<div class="video-pair"><figure><figcaption>Original / No-op · 同 Seed</figcaption>${originalVideo}</figure><figure><figcaption>${e(label)} · ${groupLabel} · ${stageLabel}</figcaption>${experimentVideo}</figure></div>`;return `<article class="card ${profile==='original'?'':group.startsWith('top')?'top':'bottom'} ${seed===focusSeed?'focus':''}" id="seed-${seed}"><h3><span>Seed ${e(seed)}</span><span>${isReady?'READY':'PENDING'}</span></h3><span class="pill">${e(label)}</span>${profile==='original'?'':`<span class="pill">${groupLabel}</span><span class="pill">${stageLabel}</span>`}${media}${r?heatmaps(seed,r):''}</article>`}).join('');document.getElementById('count').textContent=profile==='original'?`${ready}/${catalogs.length} videos ready`:`Experiment ${ready}/${catalogs.length} · Original ${originalReady}/${catalogs.length}`;document.getElementById('status').textContent=`已汇总 ${catalogs.length} seeds · 当前实验 ${ready}/${catalogs.length} 完成`}
async function fetchCatalog(seed,token){const d=await fetch(`/api/attention-additive-lora-seed-sweep/catalog?seed=${encodeURIComponent(seed)}`,{cache:'no-store'}).then(r=>r.json());if(token!==generation)throw new Error('stale');return d}
async function load(){const token=++generation;document.getElementById('status').textContent='正在读取 seed 清单';try{const first=await fetchCatalog(focusSeed,token),seeds=first.seeds||[];const bySeed=new Map([[String(first.selected_seed),first]]),queue=seeds.filter(x=>String(x)!==String(first.selected_seed));let done=1;async function worker(){while(queue.length){const seed=queue.shift(),d=await fetchCatalog(seed,token);bySeed.set(String(d.selected_seed),d);done++;document.getElementById('status').textContent=`正在汇总 ${done}/${seeds.length} seeds`}}await Promise.all(Array.from({length:Math.min(8,queue.length)},worker));catalogs=seeds.map(x=>bySeed.get(String(x))).filter(Boolean);render();if(focusSeed){const target=document.getElementById(`seed-${focusSeed}`);if(target)target.scrollIntoView({block:'center'})}}catch(err){if(err.message!=='stale')document.getElementById('status').textContent=`读取失败: ${err.message}`}}
document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));renderButtons();load();
</script></body></html>'''


ATTENTION_NEIGHBOR_CRITERIA = (
    ("strict_score", "Strict Combined"),
    ("allblock_purity", "All-block Diagonal Purity"),
    ("allblock_min_purity", "Weakest-block Purity"),
    ("balanced", "Neighbor Balanced"),
    ("uniformity", "Uniformity"),
    ("joint", "Quality x Uniformity"),
    ("mass", "Three-frame Mass"),
    ("pck32", "LoRA PCK@32"),
)
ATTENTION_NEIGHBOR_PROFILES = (
    ("alpha090", "alpha = 0.9"),
    ("alpha150", "alpha = 1.5"),
    ("zero", "A = 0"),
    ("uniform", "A = 1/N_K"),
    ("temporal_causal", "Temporal Causal"),
    ("strict_past", "Strict Past"),
    ("strict_future", "Strict Future"),
    ("exclude_current", "Exclude Current Frame"),
    ("context_only", "Context Frames Only"),
    ("head_output_zero", "Head Output Zero"),
)


ATTENTION_NEIGHBOR_CFG_BRANCHES = (
    ("both", "Both CFG Branches"),
    ("conditional", "Conditional Only"),
    ("unconditional", "Unconditional Only"),
)


def attention_neighbor_seed90094_run_root(branch, criterion, stage, profile):
    seed_root = ATTENTION_NEIGHBOR_SEED90094_ROOT / "seeds" / "seed_090094"
    if branch == "both" or profile == "identity":
        return seed_root / criterion / stage / profile
    return seed_root / "branches" / branch / criterion / stage / profile


def attention_neighbor_seed90094_asset(
    criterion: str, stage: str, profile: str, group: str, name: str = "", branch: str = "both"):
    if branch not in dict(ATTENTION_NEIGHBOR_CFG_BRANCHES):
        return None
    valid_criteria = {item[0] for item in ATTENTION_NEIGHBOR_CRITERIA}
    if stage == "original" and profile == "original" and group == "original":
        return (
            ATTENTION_LORA_SEED_SWEEP_ROOT / "seeds" / "seed_090094" / "original.mp4"
        )
    if criterion not in valid_criteria:
        return None
    if stage not in {"all_steps", "steps00_09"}:
        return None
    if profile not in {item[0] for item in ATTENTION_NEIGHBOR_PROFILES} | {"identity"}:
        return None
    if group not in {"top100", "bottom100"}:
        return None
    run_root = attention_neighbor_seed90094_run_root(branch, criterion, stage, profile)
    if name:
        if Path(name).name != name or not name.endswith(".png"):
            return None
        return run_root / "heatmaps" / name
    suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
    return (
        run_root / "videos" / "lora" / "cases" / ATTENTION_LORA_CASE
        / f"{group}_{suffix}.mp4"
    )


def attention_neighbor_seed90094_catalog(criterion: str, branch: str = "both"):
    if branch not in dict(ATTENTION_NEIGHBOR_CFG_BRANCHES):
        branch = "both"
    criteria = dict(ATTENTION_NEIGHBOR_CRITERIA)
    selected_criteria = (
        ATTENTION_NEIGHBOR_CRITERIA
        if criterion == "all"
        else tuple(item for item in ATTENTION_NEIGHBOR_CRITERIA if item[0] == criterion)
    )
    if not selected_criteria:
        criterion = "strict_score"
        selected_criteria = (ATTENTION_NEIGHBOR_CRITERIA[0],)
    records = []
    for criterion_id, criterion_label in selected_criteria:
        for stage in ("all_steps", "steps00_09"):
            capture_step = 39 if stage == "all_steps" else 9
            for group in ("top100", "bottom100"):
                for profile, label in ATTENTION_NEIGHBOR_PROFILES:
                    run_root = attention_neighbor_seed90094_run_root(branch, criterion_id, stage, profile)
                    video = attention_neighbor_seed90094_asset(
                        criterion_id, stage, profile, group, branch=branch
                    )
                    metadata_path = next(
                        iter(sorted((run_root / "heatmaps").glob(
                            f"*__{ATTENTION_LORA_CASE}__{group}__*step{capture_step:02d}.json"
                        ))),
                        None,
                    )
                    metadata = load_payload(metadata_path) if metadata_path else {}
                    all_token = str(metadata.get("all_token_image", ""))
                    frame = str(metadata.get("frame_image", ""))
                    all_token_path = attention_neighbor_seed90094_asset(
                        criterion_id, stage, profile, group, all_token, branch
                    ) if all_token else None
                    frame_path = attention_neighbor_seed90094_asset(
                        criterion_id, stage, profile, group, frame, branch
                    ) if frame else None
                    baseline_root = attention_neighbor_seed90094_run_root("both", criterion_id, stage, "identity")
                    baseline_metadata_path = next(
                        iter(sorted((baseline_root / "heatmaps").glob(
                            f"*__{ATTENTION_LORA_CASE}__{group}__*step{capture_step:02d}.json"
                        ))),
                        None,
                    )
                    baseline_metadata = (
                        load_payload(baseline_metadata_path)
                        if baseline_metadata_path else {}
                    )
                    baseline_all_token = str(
                        baseline_metadata.get("all_token_image", "")
                    )
                    baseline_frame = str(baseline_metadata.get("frame_image", ""))
                    baseline_all_token_path = attention_neighbor_seed90094_asset(
                        criterion_id, stage, "identity", group, baseline_all_token, "both"
                    ) if baseline_all_token else None
                    baseline_frame_path = attention_neighbor_seed90094_asset(
                        criterion_id, stage, "identity", group, baseline_frame, "both"
                    ) if baseline_frame else None
                    records.append({
                        "criterion": criterion_id,
                        "criterion_label": criterion_label,
                        "stage": stage,
                        "profile": profile,
                        "label": label,
                        "group": group,
                        "video_ready": bool(video and video.is_file() and video.stat().st_size),
                        "heatmap_expected": profile != "head_output_zero",
                        "heatmap_ready": bool(
                            all_token_path and all_token_path.is_file()
                            and frame_path and frame_path.is_file()
                        ),
                        "all_token": all_token,
                        "frame": frame,
                        "baseline_heatmap_ready": bool(
                            baseline_all_token_path and baseline_all_token_path.is_file()
                            and baseline_frame_path and baseline_frame_path.is_file()
                        ),
                        "baseline_all_token": baseline_all_token,
                        "baseline_frame": baseline_frame,
                    })
    completed = 0
    seed_root = ATTENTION_NEIGHBOR_SEED90094_ROOT / "seeds" / "seed_090094"
    for criterion_id, _label in ATTENTION_NEIGHBOR_CRITERIA:
        if all(
            attention_neighbor_seed90094_run_root(branch, criterion_id, stage, profile).joinpath("complete").is_file()
            for stage in ("all_steps", "steps00_09")
            for profile, _ in ATTENTION_NEIGHBOR_PROFILES
        ):
            completed += 1
    original = attention_neighbor_seed90094_asset(
        "strict_score", "original", "original", "original"
    )
    return {
        "seed": 90094,
        "criterion": criterion,
        "branch": branch,
        "branch_label": dict(ATTENTION_NEIGHBOR_CFG_BRANCHES)[branch],
        "criterion_label": "All Rankings" if criterion == "all" else criteria[criterion],
        "criteria": [
            {"id": criterion_id, "label": label}
            for criterion_id, label in ATTENTION_NEIGHBOR_CRITERIA
        ],
        "completed_criteria": completed,
        "total_criteria": len(ATTENTION_NEIGHBOR_CRITERIA),
        "ready_records": sum(record["video_ready"] for record in records),
        "expected_records": len(records),
        "original_ready": bool(original and original.is_file() and original.stat().st_size),
        "records": records,
    }


def representative_ranking_heatmaps_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Representative Ranking Heatmaps</title><style>
:root{--ink:#182720;--paper:#eee7d9;--card:#fffdf7;--line:#bdb19d;--red:#ad432f;--green:#17695d;--gold:#b87927}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 5% 0,#e8a45b55,transparent 34rem),radial-gradient(circle at 96% 2%,#4d947755,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:10;padding:16px 23px;background:#eee7d9ed;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:4px 0;font-size:clamp(27px,4vw,47px)}header p{margin:5px 0}.tools{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-top:9px}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(1900px,calc(100% - 18px));margin:auto;padding:18px 0 70px}.notes{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-bottom:14px}.note{padding:12px;background:var(--card);border:1px solid var(--line);border-top:5px solid var(--gold);border-radius:10px}.note h2{margin:0 0 5px;font-size:17px}.note p{margin:3px 0;font-size:12px}.matrix{display:grid;grid-template-columns:150px repeat(4,minmax(280px,1fr));gap:9px}.head,.row,.cell{border:1px solid var(--line);border-radius:10px}.head{padding:11px;background:#f9f3e7;text-align:center;font-weight:900}.row{padding:15px;color:#fff;background:#18372d;position:sticky;left:0;z-index:2}.row.top{border-left:7px solid var(--red)}.row.bottom{border-left:7px solid var(--green)}.cell{padding:9px;background:var(--card)}.cell img{display:block;width:100%;background:#131714;border:1px solid var(--line);margin-top:8px}.pill{display:inline-block;padding:4px 7px;margin:2px;background:#e9e1d3;border-radius:99px;font:10px ui-monospace,monospace}.pending{min-height:180px;display:grid;place-items:center;border:1px dashed var(--line);color:#746d62}.baseline{margin-top:10px;padding-top:8px;border-top:1px dashed var(--line)}@media(max-width:900px){header{position:static}.notes{grid-template-columns:1fr 1fr}.matrix{overflow:auto;grid-template-columns:120px repeat(4,300px);width:max-content}}
</style></head><body><header><a href="/attention-neighbor-ranking-seed90094?v=1">返回完整 Ranking 矩阵</a><h1>四种代表性 Ranking · Q@K Heatmaps</h1><p>Wan+LoRA · Seed 90094 · Top/Bottom100 组平均 · 每张图为 Before / After / |Delta|</p><div class="tools"><label>CFG Branch <select id="branch"><option value="both">Both</option><option value="conditional">Conditional Only</option><option value="unconditional">Unconditional Only</option></select></label><label>Experiment <select id="profile"></select></label><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="notes" class="notes"></section><section id="matrix" class="matrix"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams,criteria=[{id:'strict_score',label:'Strict Combined',why:'质量主锚点',overlap:'替代 Balanced / Quality × Uniformity'},{id:'allblock_min_purity',label:'Weakest-block Purity',why:'空间结构短板',overlap:'替代 All-block Diagonal Purity'},{id:'uniformity',label:'Uniformity',why:'最独立的 Top Head 集合',overlap:'与其他 Top100 仅重合 0–11'},{id:'mass',label:'Three-frame Mass',why:'局部时间注意力',overlap:'补充质量指标未覆盖的时间信息'}],profiles=[['alpha090','alpha = 0.9'],['alpha150','alpha = 1.5'],['zero','A = 0'],['uniform','A = 1/N_K'],['temporal_causal','Temporal Causal'],['strict_past','Strict Past'],['strict_future','Strict Future'],['exclude_current','Exclude Current Frame'],['context_only','Context Frames Only']],branch=q.get('branch')||'both',profile=q.get('profile')||'temporal_causal',stage=q.get('stage')||'all_steps';const image=(r,name,sourceProfile=r.profile,sourceBranch=branch)=>`/api/attention-neighbor-ranking-seed90094/image?criterion=${r.criterion}&stage=${r.stage}&profile=${sourceProfile}&group=${r.group}&name=${encodeURIComponent(name)}&branch=${sourceBranch}`;
function card(r){if(!r)return'<div class="pending">记录尚未建立</div>';const intervention=r.heatmap_ready?`<img loading="lazy" src="${image(r,r.all_token)}">`:'<div class="pending">该组热力图生成中</div>',baseline=r.baseline_heatmap_ready?`<div class="baseline"><strong>No-op reference</strong><img loading="lazy" src="${image(r,r.baseline_all_token,'identity','both')}"></div>`:'';return `<span class="pill">${e(r.group.toUpperCase())}</span><span class="pill">${branch}</span>${intervention}${baseline}`}
function render(d){document.getElementById('notes').innerHTML=criteria.map(c=>`<article class="note"><h2>${e(c.label)}</h2><p><strong>${e(c.why)}</strong></p><p>${e(c.overlap)}</p></article>`).join('');let html='<div class="head">Head Group</div>'+criteria.map(c=>`<div class="head">${e(c.label)}</div>`).join('');let ready=0;for(const group of ['top100','bottom100']){html+=`<div class="row ${group.startsWith('top')?'top':'bottom'}"><strong>${group.toUpperCase()}</strong><br><small>${stage==='all_steps'?'Capture S039':'Capture S009'}</small></div>`;for(const c of criteria){const r=d.records.find(x=>x.criterion===c.id&&x.stage===stage&&x.profile===profile&&x.group===group);if(r?.heatmap_ready)ready++;html+=`<div class="cell">${card(r)}</div>`}}document.getElementById('matrix').innerHTML=html;document.getElementById('status').textContent=`${ready}/8 representative heatmaps ready`}
async function load(){const d=await fetch(`/api/attention-neighbor-ranking-seed90094/catalog?criterion=all&branch=${encodeURIComponent(branch)}`,{cache:'no-store'}).then(r=>r.json());render(d)}const ps=document.getElementById('profile');ps.innerHTML=profiles.map(x=>`<option value="${x[0]}">${e(x[1])}</option>`).join('');ps.value=profile;document.getElementById('branch').value=branch;document.getElementById('viz').value=viz;document.getElementById('stage').value=stage;for(const id of ['branch','profile','stage'])document.getElementById(id).addEventListener('change',ev=>{if(id==='branch')branch=ev.target.value;if(id==='profile')profile=ev.target.value;if(id==='stage')stage=ev.target.value;const u=new URL(location.href);u.searchParams.set(id,ev.target.value);history.replaceState(null,'',u);load()});document.getElementById('refresh').addEventListener('click',load);load();
</script></body></html>'''


def attention_neighbor_seed90094_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neighbor Ranking Seed 90094</title><style>
:root{--ink:#192821;--paper:#ece6d9;--card:#fffdf7;--line:#bdb19d;--red:#ae432f;--green:#17695d;--dark:#18372d;--gold:#bc7e29}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 7% 0,#eca55d55,transparent 34rem),radial-gradient(circle at 96% 3%,#4f967955,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:16px 23px;background:#ece6d9ed;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(27px,4vw,47px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 13px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(2300px,calc(100% - 18px));margin:auto;padding:20px 0 80px}.original{max-width:700px;background:var(--card);padding:12px;border:1px solid var(--line);border-radius:14px}.original video,.card video,.card img{display:block;width:100%;background:#141815;border:1px solid var(--line);border-radius:7px}.shell{overflow:auto;margin-top:18px;border:1px solid var(--line);border-radius:15px;padding:9px;background:#d6cebf}.matrix{display:grid;grid-template-columns:220px repeat(8,330px);gap:8px;width:max-content}.head,.row,.cell{border:1px solid var(--line);border-radius:9px}.head{padding:11px;text-align:center;background:#faf5e9;font-weight:900;border-top:5px solid var(--gold)}.row{position:sticky;left:9px;z-index:3;padding:14px;background:var(--dark);color:#fff}.row.top{border-left:7px solid var(--red)}.row.bottom{border-left:7px solid var(--green)}.row small{display:block;margin-top:7px;color:#cbd8d1}.cell{padding:8px;background:#f7f1e7}.card{padding:9px;background:var(--card);height:100%;border-radius:8px}.card h3{margin:0 0 7px}.pill{display:inline-block;margin:2px;padding:4px 7px;border-radius:99px;background:#e8e0d2;font:10px ui-monospace,monospace}.maps{display:grid;gap:7px;margin-top:8px}.note{font-size:11px;color:#685f54;margin-top:7px}.pending{min-height:170px;display:grid;place-items:center;border:1px dashed var(--line);color:#766f63}.replay{position:fixed;right:20px;bottom:20px;z-index:30;border:0;border-radius:99px;background:var(--dark);color:#fff;padding:13px 19px}@media(max-width:800px){header{position:static}.matrix{grid-template-columns:170px repeat(8,280px)}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a><h1>Neighbor Ranking · Seed 90094</h1><p>同一实验横向比较 8 种 Ranking · Wan+LoRA · 001460_w002 · 40 steps / 49 frames</p><div class="tools"><label>CFG Branch <select id="branch"><option value="both">Both CFG Branches</option><option value="conditional">Conditional Only</option><option value="unconditional">Unconditional Only</option></select></label><label>Experiment <select id="profile"></select></label><button id="refresh">手动刷新</button><span class="status" id="status">读取中</span></div></header><main><h2>Original</h2><section id="original" class="original"></section><div class="shell"><section id="matrix" class="matrix"></section></div></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let profile=q.get('profile')||'alpha090',branch=q.get('branch')||'both';const branchLabels={both:'Both CFG Branches',conditional:'Conditional Only',unconditional:'Unconditional Only'};const profiles=[['alpha090','alpha = 0.9'],['alpha150','alpha = 1.5'],['zero','A = 0'],['uniform','A = 1/N_K'],['temporal_causal','Temporal Causal'],['strict_past','Strict Past'],['strict_future','Strict Future'],['exclude_current','Exclude Current Frame'],['context_only','Context Frames Only'],['head_output_zero','Head Output Zero']];const src=(criterion,stage,group)=>`/api/attention-neighbor-ranking-seed90094/video?criterion=${criterion}&stage=${stage}&profile=${profile}&group=${group}&branch=${branch}`,image=(r,name,sourceProfile=r.profile,sourceBranch=branch)=>`/api/attention-neighbor-ranking-seed90094/image?criterion=${r.criterion}&stage=${r.stage}&profile=${sourceProfile}&group=${r.group}&name=${encodeURIComponent(name)}&branch=${sourceBranch}`;function card(r){const baseline=r.baseline_heatmap_ready?`<div class="maps"><strong>No Intervention Attention · Both CFG</strong><img loading="lazy" src="${image(r,r.baseline_all_token,'identity')}" alt="No intervention all-token attention"><img loading="lazy" src="${image(r,r.baseline_frame,'identity')}" alt="No intervention frame attention"></div>`:'<div class="note">No Intervention Attention 热力图生成中</div>';const maps=r.heatmap_ready?`<div class="maps"><strong>Intervention Before / After · ${e(branchLabels[branch])}</strong><img loading="lazy" src="${image(r,r.all_token)}" alt="All-token Before/After"><img loading="lazy" src="${image(r,r.frame)}" alt="Frame Before/After"></div>`:r.heatmap_expected?'<div class="note">扰动前后热力图生成中</div>':'<div class="note">Head Output Zero 不改变 Attention</div>';return `<article class="card"><h3>${e(r.criterion_label)}</h3><span class="pill">${e(branchLabels[branch])}</span><span class="pill">${e(r.group.toUpperCase())}</span><span class="pill">${r.stage==='all_steps'?'S000-S039':'S000-S009'}</span>${r.video_ready?`<video controls preload="metadata" playsinline src="${src(r.criterion,r.stage,r.group)}"></video>`:'<div class="pending">视频生成中</div>'}${baseline}${maps}</article>`}function render(d){document.getElementById('branch').value=branch;document.getElementById('viz').value=viz;const s=document.getElementById('profile');if(!s.options.length)s.innerHTML=profiles.map(x=>`<option value="${e(x[0])}">${e(x[1])}</option>`).join('');s.value=profile;document.getElementById('status').textContent=`${d.completed_criteria}/${d.total_criteria} rankings complete · ${d.ready_records}/${d.expected_records} total videos ready`;document.getElementById('original').innerHTML=d.original_ready?`<video controls preload="metadata" playsinline src="${src('strict_score','original','original')}"></video>`:'<div class="pending">Original missing</div>';let html='<div class="head">Experiment x Stage</div>'+d.criteria.map(x=>`<div class="head">${e(x.label)}</div>`).join('');for(const stage of ['all_steps','steps00_09'])for(const group of ['top100','bottom100']){html+=`<div class="row ${group.startsWith('top')?'top':'bottom'}"><strong>${e(profiles.find(x=>x[0]===profile)?.[1]||profile)}</strong><small>${group.toUpperCase()} · ${stage==='all_steps'?'S000-S039':'S000-S009'}</small></div>`;for(const criterion of d.criteria){const r=d.records.find(x=>x.criterion===criterion.id&&x.stage===stage&&x.group===group&&x.profile===profile);html+=`<div class="cell">${card(r)}</div>`}}document.getElementById('matrix').innerHTML=html}async function load(){const d=await fetch(`/api/attention-neighbor-ranking-seed90094/catalog?criterion=all&branch=${encodeURIComponent(branch)}`,{cache:'no-store'}).then(r=>r.json());render(d)}document.getElementById('branch').addEventListener('change',ev=>{branch=ev.target.value;const u=new URL(location.href);u.searchParams.set('branch',branch);history.replaceState(null,'',u);load()});document.getElementById('profile').addEventListener('change',ev=>{profile=ev.target.value;const u=new URL(location.href);u.searchParams.set('profile',profile);history.replaceState(null,'',u);load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


def attention_lora_pck32_seed90094_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LoRA PCK@32 All Experiments · Seed 90094</title><style>
:root{--ink:#17251f;--paper:#eee8dc;--card:#fffdf8;--line:#b9ad98;--top:#b1432f;--bottom:#14695b;--dark:#17382e;--gold:#bc7b28;--cond:#245d7a;--uncond:#7a4c24}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99d5355,transparent 32rem),radial-gradient(circle at 98% 2%,#4d967855,transparent 38rem),linear-gradient(135deg,#eee8dc,#e5dece);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:16px 24px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button{padding:9px 14px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#506159}main{width:min(2700px,calc(100% - 18px));margin:auto;padding:18px 0 90px}.original{max-width:680px;padding:12px;border:1px solid var(--line);border-radius:14px;background:var(--card)}.original video,.card video,.card img{display:block;width:100%;border:1px solid var(--line);border-radius:7px;background:#131714}.branch{margin-top:25px}.branch-title{position:sticky;left:0;width:max-content;margin:0 0 9px;padding:9px 16px;border-radius:99px;background:var(--dark);color:#fff}.branch.conditional .branch-title{background:var(--cond)}.branch.unconditional .branch-title{background:var(--uncond)}.shell{overflow:auto;border:1px solid var(--line);border-radius:15px;padding:9px;background:#d6cebf}.matrix{display:grid;grid-template-columns:220px repeat(10,315px);gap:8px;width:max-content}.head,.row,.cell{border:1px solid var(--line);border-radius:9px}.head{padding:11px;text-align:center;background:#faf5e9;font-weight:900;border-top:5px solid var(--gold)}.row{position:sticky;left:9px;z-index:4;padding:14px;background:var(--dark);color:#fff}.row.top{border-left:7px solid var(--top)}.row.bottom{border-left:7px solid var(--bottom)}.row small{display:block;margin-top:7px;color:#cad8d1}.cell{padding:8px;background:#f7f1e7}.card{height:100%;padding:9px;border-radius:8px;background:var(--card)}.card h3{margin:0 0 7px;font-size:15px}.pill{display:inline-block;margin:2px;padding:4px 7px;border-radius:99px;background:#e8e0d2;font:10px ui-monospace,monospace}.maps{display:grid;gap:7px;margin-top:8px}.maps strong{font-size:12px}.pending{min-height:165px;display:grid;place-items:center;border:1px dashed var(--line);color:#756d61;text-align:center}.note{margin-top:7px;font-size:11px;color:#685f54}.replay{position:fixed;right:20px;bottom:20px;z-index:30;border:0;border-radius:99px;padding:13px 19px;background:var(--dark);color:#fff;box-shadow:0 8px 24px #0003}@media(max-width:800px){header{position:static}.matrix{grid-template-columns:170px repeat(10,275px)}main{width:calc(100% - 10px)}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/attention-neighbor-ranking-seed90094?v=1">返回全部 Ranking</a><h1>LoRA PCK@32 · 全实验矩阵</h1><p>Seed 90094 · Wan+LoRA · 001460_w002 · 固定 PCK@32 Head 排名 · 40 steps / 49 frames</p><div class="tools"><button id="refresh">手动刷新</button><span class="status" id="status">读取中</span></div></header><main><h2>Original</h2><section id="original" class="original"></section><section id="branches"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const branches=[['both','Both CFG Branches'],['conditional','Conditional Only'],['unconditional','Unconditional Only']];const profiles=[['alpha090','alpha = 0.9'],['alpha150','alpha = 1.5'],['zero','A = 0'],['uniform','A = 1/N_K'],['temporal_causal','Temporal Causal'],['strict_past','Strict Past'],['strict_future','Strict Future'],['exclude_current','Exclude Current Frame'],['context_only','Context Frames Only'],['head_output_zero','Head Output Zero']];const video=(stage,profile,group,branch)=>`/api/attention-neighbor-ranking-seed90094/video?criterion=pck32&stage=${stage}&profile=${profile}&group=${group}&branch=${branch}`,image=(r,name,branch)=>`/api/attention-neighbor-ranking-seed90094/image?criterion=pck32&stage=${r.stage}&profile=${r.profile}&group=${r.group}&name=${encodeURIComponent(name)}&branch=${branch}`;
function card(r,branch){if(!r)return '<div class="pending">等待任务记录</div>';const maps=r.heatmap_ready?`<div class="maps"><strong>Before / After Attention</strong><img loading="lazy" src="${image(r,r.all_token,branch)}" alt="All-token QK"><img loading="lazy" src="${image(r,r.frame,branch)}" alt="Frame attention"></div>`:r.heatmap_expected?'<div class="note">扰动前后热力图生成中</div>':'<div class="note">Head Output Zero 不改变 Attention</div>';return `<article class="card"><h3>${e(r.profile_label||r.profile)}</h3><span class="pill">LoRA PCK@32</span><span class="pill">${e(r.group.toUpperCase())}</span><span class="pill">${r.stage==='all_steps'?'S000-S039':'S000-S009'}</span>${r.video_ready?`<video controls preload="none" playsinline src="${video(r.stage,r.profile,r.group,branch)}"></video>`:'<div class="pending">视频生成中</div>'}${maps}</article>`}
function matrix(d,branch,label){let html='<div class="head">Group × Stage</div>'+profiles.map(x=>`<div class="head">${e(x[1])}</div>`).join('');for(const stage of ['all_steps','steps00_09'])for(const group of ['top100','bottom100']){html+=`<div class="row ${group.startsWith('top')?'top':'bottom'}"><strong>${group.toUpperCase()}</strong><small>${stage==='all_steps'?'全时间步 S000-S039':'仅前10步 S000-S009'}</small></div>`;for(const [profile] of profiles){const r=d.records.find(x=>x.criterion==='pck32'&&x.stage===stage&&x.group===group&&x.profile===profile);html+=`<div class="cell">${card(r,branch)}</div>`}}return `<section class="branch ${branch}"><h2 class="branch-title">${e(label)}</h2><div class="shell"><div class="matrix">${html}</div></div></section>`}
async function load(){document.getElementById('status').textContent='读取三个 CFG 分支...';const data=await Promise.all(branches.map(([branch])=>fetch(`/api/attention-neighbor-ranking-seed90094/catalog?criterion=pck32&branch=${branch}`,{cache:'no-store'}).then(r=>r.json())));const ready=data.reduce((n,d)=>n+d.ready_records,0),expected=data.reduce((n,d)=>n+d.expected_records,0);document.getElementById('status').textContent=`${ready}/${expected} videos ready · 3 CFG branches · 10 experiments`;document.getElementById('original').innerHTML=data[0].original_ready?`<video controls preload="metadata" playsinline src="${video('original','original','original','both')}"></video>`:'<div class="pending">Original missing</div>';document.getElementById('branches').innerHTML=branches.map(([branch,label],i)=>matrix(data[i],branch,label)).join('')}
document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


ATTENTION_LORA_PCK32_TEMPORAL_TEST5_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_pck32_temporal_test5_seed000851"
)
ATTENTION_LORA_PCK32_TEMPORAL_TEST5_LIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
)
ATTENTION_LORA_PCK32_TEMPORAL_TEST5_BASELINE = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5/"
    "lora/alpha090_count100/videos/lora/cases"
)


def attention_lora_pck32_temporal_test5_cases():
    cases = []
    seen = set()
    for line in ATTENTION_LORA_PCK32_TEMPORAL_TEST5_LIST.read_text(
        encoding="utf-8"
    ).splitlines():
        case = Path(line.strip()).stem
        if case and case not in seen:
            seen.add(case)
            cases.append(case)
    return cases


def attention_lora_pck32_temporal_test5_asset(case: str, kind: str):
    if case not in set(attention_lora_pck32_temporal_test5_cases()):
        return None
    if kind == "original":
        return ATTENTION_LORA_PCK32_TEMPORAL_TEST5_BASELINE / case / "original.mp4"
    filenames = {
        "steps00_09": "top100_steps_00_10.mp4",
        "all_steps": "top100_steps_00_40.mp4",
    }
    if kind not in filenames:
        return None
    return (
        ATTENTION_LORA_PCK32_TEMPORAL_TEST5_ROOT
        / kind
        / "temporal_causal"
        / "videos"
        / "lora"
        / "cases"
        / case
        / filenames[kind]
    )


def attention_lora_pck32_temporal_test5_catalog():
    records = []
    for case in attention_lora_pck32_temporal_test5_cases():
        ready = {}
        for kind in ("original", "steps00_09", "all_steps"):
            asset = attention_lora_pck32_temporal_test5_asset(case, kind)
            ready[kind] = bool(asset and asset.is_file() and asset.stat().st_size)
        records.append({"case": case, "ready": ready})
    return {
        "seed": 851,
        "model": "Wan+LoRA",
        "criterion": "LoRA PCK@32 Top100",
        "records": records,
        "ready_interventions": sum(
            record["ready"][kind]
            for record in records
            for kind in ("steps00_09", "all_steps")
        ),
        "expected_interventions": len(records) * 2,
    }


def attention_lora_pck32_temporal_test5_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LoRA PCK@32 Temporal Causal · Test5</title><style>
:root{--ink:#15241e;--paper:#eee8dc;--card:#fffdf8;--line:#b8ad98;--green:#17685b;--rust:#af472f;--gold:#bd7d29}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 5% 0,#e99d5555,transparent 34rem),radial-gradient(circle at 97% 4%,#4b947755,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:17px 24px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 13px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#506159}main{width:min(1900px,calc(100% - 20px));margin:auto;padding:24px 0 90px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{padding:13px;border:1px solid var(--line);border-radius:15px;background:var(--card);box-shadow:0 10px 28px #2b251515}.card.original{border-top:7px solid var(--gold)}.card.short{border-top:7px solid var(--rust)}.card.full{border-top:7px solid var(--green)}.card h2{margin:0 0 5px}.meta{margin-bottom:10px;font:11px ui-monospace,monospace;color:#625c52}.card video{display:block;width:100%;background:#131714;border-radius:8px}.pending{min-height:300px;display:grid;place-items:center;border:1px dashed var(--line);border-radius:8px;color:#746d61}.replay{position:fixed;right:20px;bottom:20px;z-index:30;border:0;border-radius:99px;padding:13px 19px;background:var(--green);color:#fff;box-shadow:0 8px 24px #0003}@media(max-width:950px){header{position:static}.grid{grid-template-columns:1fr}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/attention-lora-pck32-seed90094?v=1">PCK@32 矩阵</a><h1>Temporal Causal · Test5</h1><p>Wan+LoRA · Seed 851 · LoRA PCK@32 Top100 · 40 steps / 49 frames</p><div class="tools"><label>Case <select id="case"></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="grid" class="grid"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let current=q.get('case')||'';const src=kind=>`/api/attention-lora-pck32-temporal-test5/video?case=${encodeURIComponent(current)}&kind=${kind}`;function card(kind,title,meta,ready,cls){return `<article class="card ${cls}"><h2>${e(title)}</h2><div class="meta">${e(meta)}</div>${ready?`<video controls preload="metadata" playsinline src="${src(kind)}"></video>`:'<div class="pending">视频生成中</div>'}</article>`}function render(d){const cases=d.records.map(x=>x.case);if(!current||!cases.includes(current))current=cases[0]||'';const select=document.getElementById('case');if(select.options.length!==cases.length)select.innerHTML=cases.map(x=>`<option value="${e(x)}">${e(x)}</option>`).join('');select.value=current;const r=d.records.find(x=>x.case===current);document.getElementById('status').textContent=`${d.ready_interventions}/${d.expected_interventions} intervention videos ready · ${cases.length} unique cases`;document.getElementById('grid').innerHTML=card('original','Wan+LoRA Original','No Attention Intervention · Seed 851',r?.ready.original,'original')+card('steps00_09','Temporal Causal · S000-S009','LoRA PCK@32 Top100 · only first 10 denoising steps',r?.ready.steps00_09,'short')+card('all_steps','Temporal Causal · S000-S039','LoRA PCK@32 Top100 · all 40 denoising steps',r?.ready.all_steps,'full')}
async function load(){const d=await fetch('/api/attention-lora-pck32-temporal-test5/catalog',{cache:'no-store'}).then(r=>r.json());render(d)}document.getElementById('case').addEventListener('change',ev=>{current=ev.target.value;const u=new URL(location.href);u.searchParams.set('case',current);history.replaceState(null,'',u);load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


OBJECT_QUERY_CONTINUITY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_continuity_case001460"
)
OBJECT_QUERY_MAIN_COMPONENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_main_component_case001460"
)
OBJECT_QUERY_CONTINUITY_SEEDS = (90094, 35075, 21890, 49530, 47326, 32466)
OBJECT_QUERY_CONTINUITY_STAGES = ("all_steps", "steps00_09")
OBJECT_QUERY_CONTINUITY_CASE = "0613pybullet_sample_001460_w002"


def object_query_continuity_asset(seed: str, stage: str, kind: str, name: str = ""):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if seed_value not in OBJECT_QUERY_CONTINUITY_SEEDS or stage not in OBJECT_QUERY_CONTINUITY_STAGES:
        return None
    seed_root = OBJECT_QUERY_CONTINUITY_ROOT / "seeds" / f"seed_{seed_value:06d}" / stage
    if kind == "baseline":
        return (
            Path("/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460")
            / "seeds" / f"seed_{seed_value:06d}" / "original.mp4"
        )
    if kind == "intervention":
        suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
        return (
            seed_root / "videos" / "lora" / "cases" / OBJECT_QUERY_CONTINUITY_CASE
            / f"top100_{suffix}.mp4"
        )
    if kind == "new_intervention":
        suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
        return (
            OBJECT_QUERY_MAIN_COMPONENT_ROOT / "seeds" / f"seed_{seed_value:06d}"
            / stage / "videos" / "lora" / "cases" / OBJECT_QUERY_CONTINUITY_CASE
            / f"top100_{suffix}.mp4"
        )
    if kind == "overlay" and Path(name).name == name and name.endswith(".jpg"):
        return seed_root / "overlays" / name
    if kind == "identity_overlay" and Path(name).name == name and name.endswith(".jpg"):
        step_tag = "step39" if stage == "all_steps" else "step09"
        return (
            OBJECT_QUERY_CONTINUITY_ROOT / "seeds" / f"seed_{seed_value:06d}"
            / "identity" / step_tag / "overlays" / name
        )
    if kind == "new_overlay" and Path(name).name == name and name.endswith(".jpg"):
        return (
            OBJECT_QUERY_MAIN_COMPONENT_ROOT / "seeds" / f"seed_{seed_value:06d}"
            / stage / "overlays" / name
        )
    return None


def object_query_continuity_catalog(seed: str, stage: str):
    try:
        seed_value = int(seed)
    except ValueError:
        seed_value = OBJECT_QUERY_CONTINUITY_SEEDS[0]
    if seed_value not in OBJECT_QUERY_CONTINUITY_SEEDS:
        seed_value = OBJECT_QUERY_CONTINUITY_SEEDS[0]
    if stage not in OBJECT_QUERY_CONTINUITY_STAGES:
        stage = "all_steps"
    manifest = (
        OBJECT_QUERY_CONTINUITY_ROOT / "seeds" / f"seed_{seed_value:06d}"
        / stage / "overlays" / "manifest.json"
    )
    records = []
    if manifest.is_file():
        records = json.loads(manifest.read_text(encoding="utf-8")).get("records", [])
    identity_tag = "step39" if stage == "all_steps" else "step09"
    identity_manifest = (
        OBJECT_QUERY_CONTINUITY_ROOT / "seeds" / f"seed_{seed_value:06d}"
        / "identity" / identity_tag / "overlays" / "manifest.json"
    )
    identity_records = []
    if identity_manifest.is_file():
        identity_records = json.loads(identity_manifest.read_text(encoding="utf-8")).get("records", [])
    new_manifest = (
        OBJECT_QUERY_MAIN_COMPONENT_ROOT / "seeds" / f"seed_{seed_value:06d}"
        / stage / "overlays" / "manifest.json"
    )
    new_records = []
    if new_manifest.is_file():
        new_records = json.loads(new_manifest.read_text(encoding="utf-8")).get("records", [])
    intervention = {(int(row["block"]), int(row["head"]), row["region_name"]): row for row in records}
    identity = {(int(row["block"]), int(row["head"]), row["region_name"]): row for row in identity_records}
    new_intervention = {
        (int(row["block"]), int(row["head"]), row["region_name"]): row
        for row in new_records
    }
    merged = []
    for key in sorted(set(intervention) | set(identity) | set(new_intervention)):
        intervention_row = intervention.get(key, {})
        identity_row = identity.get(key, {})
        row = dict(intervention_row or identity_row)
        intervention_images = intervention_row.get("images", {})
        identity_images = identity_row.get("images", {})
        row["identity_image"] = (
            identity_images.get("identity") or identity_row.get("image")
        )
        row["before_image"] = intervention_images.get("before")
        row["p90_mask_image"] = intervention_images.get("p90_mask")
        row["main_component_image"] = intervention_images.get("main_component")
        row["after_image"] = intervention_images.get("after")
        row["removed_image"] = intervention_images.get("removed")
        row["image"] = intervention_images.get("combined") or intervention_row.get("image")
        new_row = new_intervention.get(key, {})
        new_images = new_row.get("images", {})
        row["new_before_image"] = new_images.get("before")
        row["new_p90_mask_image"] = new_images.get("p90_mask")
        row["new_main_component_image"] = new_images.get("main_component")
        row["new_after_image"] = new_images.get("after")
        row["new_removed_image"] = new_images.get("removed")
        row["new_p90_mask_source"] = new_row.get("p90_mask_source")
        row["new_main_component_source"] = new_row.get("main_component_source")
        merged.append(row)
    records = merged
    records.sort(key=lambda row: (-float(row.get("pck32", 0)), int(row["block"]), int(row["head"]), row["region_name"]))
    return {
        "seed": seed_value,
        "stage": stage,
        "seeds": list(OBJECT_QUERY_CONTINUITY_SEEDS),
        "stages": list(OBJECT_QUERY_CONTINUITY_STAGES),
        "baseline_ready": bool((object_query_continuity_asset(str(seed_value), stage, "baseline") or Path()).is_file()),
        "video_ready": bool((object_query_continuity_asset(str(seed_value), stage, "intervention") or Path()).is_file()),
        "new_video_ready": bool((object_query_continuity_asset(str(seed_value), stage, "new_intervention") or Path()).is_file()),
        "records": records,
        "identity_ready": len(identity_records),
        "intervention_ready": len(intervention),
        "new_intervention_ready": len(new_intervention),
    }


def object_query_continuity_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object Query Spatial Continuity</title><style>
:root{--paper:#eee8dc;--ink:#17251f;--line:#b9ad98;--card:#fffdf8;--orange:#b54a2e;--green:#176a5c;--blue:#285f7a}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 5% 0,#ea9f5655,transparent 34rem),radial-gradient(circle at 97% 3%,#4d957855,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:16px 24px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 13px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#526159}main{width:min(2450px,calc(100% - 18px));margin:auto;padding:20px 0 90px}.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:18px}.video-card,.head-card{padding:12px;border:1px solid var(--line);border-radius:15px;background:var(--card)}.video-card:first-child{border-top:7px solid var(--blue)}.video-card:last-child{border-top:7px solid var(--green)}video{display:block;width:100%;background:#121714;border-radius:8px}.head-card{margin:14px 0;border-left:7px solid var(--orange)}.head-card h2{margin:0 0 8px}.meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}.pill{padding:5px 8px;border-radius:99px;background:#e8e0d2;font:11px ui-monospace,monospace}.objects{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.object{overflow:auto;border:1px solid var(--line);border-radius:9px;padding:7px}.object h3{position:sticky;left:0;width:max-content;margin:0 0 6px;padding:4px 9px;background:var(--ink);color:#fff;border-radius:5px}.attention-row{margin:8px 0 14px}.attention-row h4{position:sticky;left:0;width:max-content;margin:0 0 5px;padding:3px 7px;background:#ece4d6;border-left:4px solid var(--blue)}.attention-row.intervention h4{border-left-color:var(--orange)}.attention-row.removed h4{border-left-color:var(--green)}.object img{display:block;min-width:2240px;width:100%;border:1px solid #d8cfbf}.pending{padding:70px 20px;text-align:center;border:1px dashed var(--line);border-radius:10px}.replay{position:fixed;right:20px;bottom:20px;z-index:30;border:0;border-radius:99px;padding:13px 19px;background:var(--green);color:#fff;box-shadow:0 8px 24px #0003}@media(max-width:900px){header{position:static}.videos,.objects{grid-template-columns:1fr}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/object-query-attention-overlay?v=1">原 Object Query Overlay</a><h1>Object Query 跨帧空间连续性</h1><p>0613pybullet_sample_001460_w002 · Wan+LoRA · PCK@32 Top100 · Object A/B 各 8 query tokens · anchor K01/F04</p><div class="tools"><label>Seed <select id="seed"><option>90094</option><option>35075</option><option>21890</option><option>49530</option><option>47326</option><option>32466</option></select></label><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="videos" class="videos"></section><section id="records"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let seed=q.get('seed')||'90094',stage=q.get('stage')||'all_steps';const video=kind=>`/api/object-query-continuity-overlay/video?seed=${seed}&stage=${stage}&kind=${kind}`,image=(name,kind='overlay')=>`/api/object-query-continuity-overlay/image?seed=${seed}&stage=${stage}&kind=${kind}&name=${encodeURIComponent(name)}`,panel=(title,name,kind='overlay',cls='')=>`<div class="attention-row ${cls}"><h4>${title}</h4>${name?`<img loading="lazy" src="${image(name,kind)}">`:'<div class="pending">capture 生成中</div>'}</div>`;function sync(){const u=new URL(location.href);u.searchParams.set('seed',seed);u.searchParams.set('stage',stage);history.replaceState(null,'',u)}function render(d){document.getElementById('seed').value=String(seed);document.getElementById('stage').value=stage;document.getElementById('status').textContent=`Identity ${d.identity_ready}/20 · Intervention ${d.intervention_ready}/20`;document.getElementById('videos').innerHTML=`<article class="video-card"><h2>Wan+LoRA Original</h2><h3>40-step inference</h3>${d.baseline_ready?`<video controls preload="metadata" playsinline src="${video('baseline')}"></video>`:'<div class="pending">Baseline missing</div>'}</article><article class="video-card"><h2>Spatial Continuity · ${stage==='all_steps'?'S000-S039':'S000-S009'}</h2>${d.video_ready?`<video controls preload="metadata" playsinline src="${video('intervention')}"></video>`:'<div class="pending">视频生成中</div>'}</article>`;const groups=new Map;for(const r of d.records){const key=`${r.block}:${r.head}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r)}document.getElementById('records').innerHTML=groups.size?[...groups.values()].map(rows=>{const first=rows[0];return `<article class="head-card"><h2>L${String(first.block).padStart(2,'0')} / H${String(first.head).padStart(2,'0')}</h2><div class="meta"><span class="pill">LoRA PCK@32 ${Number(first.pck32).toFixed(3)}</span><span class="pill">8 queries preserved then SUM</span><span class="pill">P90 + P99/Top-5 masks captured</span><span class="pill">Main component is visualization-only</span><span class="pill">Only K02-K12 constrained</span></div><div class="objects">${rows.map(r=>`<section class="object"><h3>${e(r.region_name)} · ${e(r.region_phrase)}</h3>${panel('No Intervention · Original Object Query Attention',r.identity_image,'identity_overlay')}${panel('Continuity Intervention · Before',r.before_image,'overlay','intervention')}${panel(`Per-frame P90 Mask · ${e(r.p90_mask_source||'pending')}`,r.p90_mask_image,'overlay','removed')}${panel(`P99 / Top-${e(r.main_component_topk||5)} Main Connected Component · ${e(r.main_component_source||'pending')}`,r.main_component_image,'overlay','removed')}${panel('Continuity Intervention · After',r.after_image,'overlay','intervention')}${panel('Continuity Intervention · Removed Attention Mass',r.removed_image,'overlay','removed')}</section>`).join('')}</div></article>`}).join(''):'<div class="pending">Identity 与 intervention capture 生成中，点击手动刷新。</div>'}
async function load(){const d=await fetch(`/api/object-query-continuity-overlay/catalog?seed=${seed}&stage=${stage}`,{cache:'no-store'}).then(r=>r.json());render(d);renderAlignment(d);renderReverseAlignment(d)}function renderAlignment(d){if(String(seed)!=="47326"||String(step)!=="9"||d.branch!=="conditional"||(viz!=="common"&&viz!=="all"))return;document.querySelectorAll("#content .object").forEach((el,i)=>{const region=d.records[i]?.region_name;if(!region)return;const name=`seed047326__step09__conditional__${region}__best_head_matches.jpg`;el.insertAdjacentHTML("beforeend",row("1c. Common Highest-Similarity Head Pairs","","p95_trajectory",name,"Wan+LoRA sample-specific cosine Top-5 per object. Each pair shows 10-step attention, its best matching 40-step attention, and absolute delta; 10/40 share a per-latent-frame color scale."))})}for(const id of ['seed','stage'])document.getElementById(id).addEventListener('change',ev=>{if(id==='seed')seed=ev.target.value;else stage=ev.target.value;sync();load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


def object_query_continuity_comparison_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object Query Continuity · Old vs New</title><style>
:root{--paper:#eee8dc;--ink:#15241e;--line:#b8ad98;--card:#fffdf8;--old:#b44d31;--new:#176a5c;--blue:#285f7a}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99d5550,transparent 34rem),radial-gradient(circle at 97% 3%,#4b947750,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:15px 22px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(2500px,calc(100% - 16px));margin:auto;padding:18px 0 90px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.video,.head,.object,.method{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px}.video.original{border-top:7px solid var(--blue)}.video.old{border-top:7px solid var(--old)}.video.new{border-top:7px solid var(--new)}video{display:block;width:100%;background:#111714}.head{margin:14px 0}.head>h2{margin:0 0 7px}.objects{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.object{overflow:auto}.object>h3{position:sticky;left:0;width:max-content;margin:0 0 7px;padding:4px 8px;background:var(--ink);color:#fff}.identity{border-left:6px solid var(--blue);margin-bottom:9px}.methods{display:grid;grid-template-columns:1fr;gap:12px}.pair{padding:8px;border:1px solid var(--line);border-radius:10px;background:#f7f1e7}.pair>h3{position:sticky;left:0;width:max-content;margin:0 0 7px;padding:4px 9px;background:var(--ink);color:#fff}.row{margin:6px 0 12px;padding-left:7px}.row.old{border-left:6px solid var(--old)}.row.new{border-left:6px solid var(--new)}.row h4{position:sticky;left:0;width:max-content;margin:0 0 4px;padding:3px 7px;background:#ece4d6}.row img{display:block;min-width:2240px;width:100%;border:1px solid #d8cfbf}.pending{padding:48px 15px;text-align:center;border:1px dashed var(--line)}.replay{position:fixed;right:18px;bottom:18px;z-index:30;border:0;border-radius:99px;padding:13px 18px;background:var(--new);color:white}@media(max-width:1000px){header{position:static}.videos,.objects,.methods{grid-template-columns:1fr}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/object-query-continuity-overlay?v=5">旧方案页面</a><h1>Object Query Continuity · Old vs New</h1><p>旧：P90 链式邻接。新：上一帧 P99/Top-5 主连通分量约束当前帧 P90；主连通分量逐帧独立计算。</p><div class="tools"><label>Seed <select id="seed"><option>90094</option><option>35075</option><option>21890</option><option>49530</option><option>47326</option><option>32466</option></select></label><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="videos" class="videos"></section><section id="records"></section></main><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let seed=q.get('seed')||'90094',stage=q.get('stage')||'all_steps';const asset=(kind,name='')=>`/api/object-query-continuity-overlay/${kind.includes('intervention')||kind==='baseline'?'video':'image'}?seed=${seed}&stage=${stage}&kind=${kind}${name?`&name=${encodeURIComponent(name)}`:''}`,video=(cls,title,kind,ready)=>`<article class="video ${cls}"><h2>${title}</h2>${ready?`<video controls preload="metadata" playsinline src="${asset(kind)}"></video>`:'<div class="pending">生成中</div>'}</article>`,row=(title,name,kind,cls='')=>`<div class="row ${cls}"><h4>${title}</h4>${name?`<img loading="lazy" src="${asset(kind,name)}">`:'<div class="pending">capture 生成中</div>'}</div>`,scheme=(title,cls,identity,before,after,mask,removed,kind,maskTitle)=>`<section class="pair"><h3>${title}</h3>${row('No Intervention · Original Attention',identity,'identity_overlay',cls)}${row('Before',before,kind,cls)}${row('After',after,kind,cls)}${row(maskTitle,mask,kind,cls)}${row('Removed Attention Mass',removed,kind,cls)}</section>`:'';function sync(){const u=new URL(location.href);u.searchParams.set('seed',seed);u.searchParams.set('stage',stage);history.replaceState(null,'',u)}function render(d){document.getElementById('seed').value=String(seed);document.getElementById('stage').value=stage;document.getElementById('status').textContent=`Identity ${d.identity_ready}/20 · Old ${d.intervention_ready}/20 · New ${d.new_intervention_ready}/20`;document.getElementById('videos').innerHTML=video('original','Wan+LoRA Original','baseline',d.baseline_ready)+video('old','旧方案 · P90 Chain','intervention',d.video_ready)+video('new','新方案 · Top-5 Main Component','new_intervention',d.new_video_ready);const groups=new Map;for(const r of d.records){const key=`${r.block}:${r.head}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r)}document.getElementById('records').innerHTML=groups.size?[...groups.values()].map(rows=>{const f=rows[0];return `<article class="head"><h2>L${String(f.block).padStart(2,'0')} / H${String(f.head).padStart(2,'0')} · PCK@32 ${Number(f.pck32).toFixed(3)}</h2><div class="objects">${rows.map(r=>`<section class="object"><h3>${esc(r.region_name)} · ${esc(r.region_phrase)}</h3><div class="methods">${scheme('旧方案 · P90 Chain','old',r.identity_image,r.before_image,r.after_image,r.p90_mask_image,r.removed_image,'overlay','Mask · Per-frame P90')}${scheme('新方案 · Previous Top-5 Main Component','new',r.identity_image,r.new_before_image,r.new_after_image,r.new_main_component_image,r.new_removed_image,'new_overlay','Mask · P99 / Top-5 Main Connected Component')}</div></section>`).join('')}</div></article>`}).join(''):'<div class="pending">等待 capture</div>'}async function load(){const d=await fetch(`/api/object-query-continuity-overlay/catalog?seed=${seed}&stage=${stage}`,{cache:'no-store'}).then(r=>r.json());render(d)}for(const id of ['seed','stage'])document.getElementById(id).addEventListener('change',ev=>{if(id==='seed')seed=ev.target.value;else stage=ev.target.value;sync();load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


PHYSIQ025_OBJECT_QUERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_physiq025_10seed"
)
PHYSIQ025_OBJECT_QUERY_SEEDS_FILE = Path(
    "/data/gaoya/agent-data/inputs/object_query_physiq025/seeds.txt"
)
PHYSIQ025_OBJECT_QUERY_CASE = (
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed"
)


def physiq025_object_query_seeds():
    if not PHYSIQ025_OBJECT_QUERY_SEEDS_FILE.is_file():
        return []
    return [
        int(line.strip())
        for line in PHYSIQ025_OBJECT_QUERY_SEEDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def physiq025_object_query_asset(
    seed: str, stage: str, method: str, kind: str, name: str = ""
):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if seed_value not in physiq025_object_query_seeds():
        return None
    seed_root = PHYSIQ025_OBJECT_QUERY_ROOT / "seeds" / f"seed_{seed_value:06d}"
    if kind == "baseline":
        return seed_root / "original.mp4"
    if stage not in {"all_steps", "steps00_09"}:
        return None
    if kind == "intervention" and method in {"old", "new"}:
        suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
        return (
            seed_root / method / stage / "videos" / "lora" / "cases"
            / PHYSIQ025_OBJECT_QUERY_CASE / f"top100_{suffix}.mp4"
        )
    if kind == "overlay" and method in {"old", "new"}:
        if Path(name).name == name and name.endswith(".jpg"):
            return seed_root / method / stage / "overlays" / name
    if kind == "identity_overlay" and method == "identity":
        if Path(name).name == name and name.endswith(".jpg"):
            step_tag = "step39" if stage == "all_steps" else "step09"
            return seed_root / "identity" / step_tag / "overlays" / name
    return None


def physiq025_object_query_catalog(seed: str, stage: str):
    seeds = physiq025_object_query_seeds()
    try:
        seed_value = int(seed)
    except ValueError:
        seed_value = seeds[0] if seeds else 0
    if seed_value not in seeds and seeds:
        seed_value = seeds[0]
    if stage not in {"all_steps", "steps00_09"}:
        stage = "all_steps"
    seed_root = PHYSIQ025_OBJECT_QUERY_ROOT / "seeds" / f"seed_{seed_value:06d}"

    def records(path):
        if not path.is_file():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("records", [])

    old_rows = records(seed_root / "old" / stage / "overlays" / "manifest.json")
    new_rows = records(seed_root / "new" / stage / "overlays" / "manifest.json")
    step_tag = "step39" if stage == "all_steps" else "step09"
    identity_rows = records(
        seed_root / "identity" / step_tag / "overlays" / "manifest.json"
    )

    def indexed(rows):
        return {
            (int(row["block"]), int(row["head"]), row["region_name"]): row
            for row in rows
        }

    old, new, identity = indexed(old_rows), indexed(new_rows), indexed(identity_rows)
    merged = []
    for key in sorted(set(old) | set(new) | set(identity)):
        old_row, new_row, identity_row = old.get(key, {}), new.get(key, {}), identity.get(key, {})
        row = dict(old_row or new_row or identity_row)
        identity_images = identity_row.get("images", {})
        row["identity_image"] = identity_images.get("identity") or identity_row.get("image")
        for prefix, source in (("old_", old_row), ("new_", new_row)):
            images = source.get("images", {})
            row[prefix + "before_image"] = images.get("before")
            row[prefix + "after_image"] = images.get("after")
            row[prefix + "mask_image"] = images.get(
                "p90_mask" if prefix == "old_" else "main_component"
            )
            row[prefix + "removed_image"] = images.get("removed")
        merged.append(row)
    merged.sort(
        key=lambda row: (
            -float(row.get("pck32", 0)), int(row["block"]),
            int(row["head"]), row["region_name"],
        )
    )
    return {
        "seed": seed_value,
        "stage": stage,
        "seeds": seeds,
        "records": merged,
        "identity_ready": len(identity_rows),
        "old_ready": len(old_rows),
        "new_ready": len(new_rows),
        "baseline_ready": bool(
            (physiq025_object_query_asset(str(seed_value), stage, "identity", "baseline") or Path()).is_file()
        ),
        "old_video_ready": bool(
            (physiq025_object_query_asset(str(seed_value), stage, "old", "intervention") or Path()).is_file()
        ),
        "new_video_ready": bool(
            (physiq025_object_query_asset(str(seed_value), stage, "new", "intervention") or Path()).is_file()
        ),
    }


def physiq025_object_query_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysicIQ025 Object Query · Old vs New</title><style>
:root{--paper:#eee8dc;--ink:#15241e;--line:#b8ad98;--card:#fffdf8;--old:#b44d31;--new:#176a5c;--blue:#285f7a}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99d5550,transparent 34rem),radial-gradient(circle at 97% 3%,#4b947750,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:15px 22px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(2500px,calc(100% - 16px));margin:auto;padding:18px 0 90px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.video,.head,.object,.scheme{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px}.video.original{border-top:7px solid var(--blue)}.video.old,.scheme.old{border-top:7px solid var(--old)}.video.new,.scheme.new{border-top:7px solid var(--new)}video{display:block;width:100%;background:#111714}.head{margin:14px 0}.head>h2{margin:0 0 7px}.objects{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.object{overflow:auto}.object>h3{position:sticky;left:0;width:max-content;margin:0 0 7px;padding:4px 8px;background:var(--ink);color:#fff}.schemes{display:grid;grid-template-columns:1fr;gap:12px}.scheme>h3{position:sticky;left:0;width:max-content;margin:0 0 7px}.row{margin:6px 0 12px}.row h4{position:sticky;left:0;width:max-content;margin:0 0 4px;padding:3px 7px;background:#ece4d6}.row img{display:block;min-width:2240px;width:100%;border:1px solid #d8cfbf}.pending{padding:48px 15px;text-align:center;border:1px dashed var(--line)}.replay{position:fixed;right:18px;bottom:18px;z-index:30;border:0;border-radius:99px;padding:13px 18px;background:var(--new);color:white}@media(max-width:1000px){header{position:static}.videos,.objects{grid-template-columns:1fr}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a><h1>PhysicIQ025 · Object Query Continuity</h1><p>Wan+LoRA · Object A=brown tennis ball · Object B=orange block · Random 10 seeds · PCK@32 Top100</p><div class="tools"><label>Seed <select id="seed"></select></label><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="videos" class="videos"></section><section id="records"></section></main><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let seed=q.get('seed')||'',stage=q.get('stage')||'all_steps',generation=Date.now();const api='/api/physiq025-object-query-continuity-comparison',asset=(type,method,kind,name='')=>`${api}/${type}?seed=${seed}&stage=${stage}&method=${method}&kind=${kind}${name?`&name=${encodeURIComponent(name)}`:''}&v=${generation}`,video=(cls,title,method,kind,ready)=>`<article class="video ${cls}"><h2>${title}</h2>${ready?`<video controls preload="metadata" playsinline src="${asset('video',method,kind)}"></video>`:'<div class="pending">生成中</div>'}</article>`,row=(title,name,method,kind)=>`<div class="row"><h4>${title}</h4>${name?`<img loading="lazy" src="${asset('image',method,kind,name)}">`:'<div class="pending">capture 生成中</div>'}</div>`,scheme=(title,cls,r,prefix,method,maskTitle)=>`<section class="scheme ${cls}"><h3>${title}</h3>${row('No Intervention · Original Attention',r.identity_image,'identity','identity_overlay')}${row('Before',r[`${prefix}before_image`],method,'overlay')}${row('After',r[`${prefix}after_image`],method,'overlay')}${row(maskTitle,r[`${prefix}mask_image`],method,'overlay')}${row('Removed Attention Mass',r[`${prefix}removed_image`],method,'overlay')}</section>`;function sync(){const u=new URL(location.href);u.searchParams.set('seed',seed);u.searchParams.set('stage',stage);history.replaceState(null,'',u)}function render(d){if(!seed)seed=String(d.seed);const sel=document.getElementById('seed');if(sel.options.length!==d.seeds.length)sel.innerHTML=d.seeds.map(x=>`<option>${x}</option>`).join('');sel.value=String(seed);document.getElementById('stage').value=stage;document.getElementById('status').textContent=`Identity ${d.identity_ready}/20 · Old ${d.old_ready}/20 · New ${d.new_ready}/20`;document.getElementById('videos').innerHTML=video('original','Wan+LoRA Original','identity','baseline',d.baseline_ready)+video('old','旧方案 · P90 Chain','old','intervention',d.old_video_ready)+video('new','新方案 · Top-5 Main Component','new','intervention',d.new_video_ready);const groups=new Map;for(const r of d.records){const key=`${r.block}:${r.head}`;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r)}document.getElementById('records').innerHTML=groups.size?[...groups.values()].map(rows=>{const f=rows[0];return `<article class="head"><h2>L${String(f.block).padStart(2,'0')} / H${String(f.head).padStart(2,'0')} · PCK@32 ${Number(f.pck32).toFixed(3)}</h2><div class="objects">${rows.map(r=>`<section class="object"><h3>${esc(r.region_name)} · ${esc(r.region_phrase)}</h3><div class="schemes">${scheme('旧方案 · P90 Chain','old',r,'old_','old','Mask · Per-frame P90')}${scheme('新方案 · Previous Top-5 Main Component','new',r,'new_','new','Mask · P99 / Top-5 Main Connected Component')}</div></section>`).join('')}</div></article>`}).join(''):'<div class="pending">等待该 seed 的 capture</div>'}async function load(){generation=Date.now();const d=await fetch(`${api}/catalog?seed=${seed}&stage=${stage}&v=${generation}`,{cache:'no-store'}).then(r=>r.json());render(d)}for(const id of ['seed','stage'])document.getElementById(id).addEventListener('change',ev=>{if(id==='seed')seed=ev.target.value;else stage=ev.target.value;sync();load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


PHYSIQ025_FROZEN_TRAJECTORY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_physiq025_2seed"
)
PHYSIQ025_FROZEN_TRAJECTORY_SEEDS = [13161, 16342]
PHYSIQ025_FROZEN_TRAJECTORY_VARIANTS = [
    ("p95", "P95 · Multi Component", "Top100 mean", "No dilation"),
    ("p99", "P99 · Multi Component", "Top100 mean", "No dilation"),
    ("p95_single", "P95 · Single Component", "Top100 mean", "No dilation"),
    ("p99_single", "P99 · Single Component", "Top100 mean", "No dilation"),
    ("p95_single_d1", "P95 · Single + Dilate1", "Top100 mean", "Removal dilate 1"),
    ("p99_single_d1", "P99 · Single + Dilate1", "Top100 mean", "Removal dilate 1"),
    ("p95_single_bt3_d1", "P95 · Backtrack3 + Dilate1", "Top30 mean", "Backtrack 3 frames"),
    ("p99_single_bt3_d1", "P99 · Backtrack3 + Dilate1", "Top30 mean", "Backtrack 3 frames"),
]


def _physiq025_frozen_manifest(path: Path):
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return records if isinstance(records, list) else []


def _physiq025_frozen_record(records, step_value: int, branch: str, region: str):
    for record in records:
        if (
            int(record.get("step", -1)) == step_value
            and record.get("cfg_branch") == branch
            and record.get("region_name") == region
        ):
            return record
    return {}


def physiq025_object_query_frozen_trajectory_asset(
    seed: str, stage: str, label: str, kind: str, name: str
):
    try:
        seed_value = int(seed)
    except (TypeError, ValueError):
        return None
    if seed_value not in PHYSIQ025_FROZEN_TRAJECTORY_SEEDS:
        return None
    if stage not in {"all_steps", "steps00_09"}:
        return None
    labels = {item[0] for item in PHYSIQ025_FROZEN_TRAJECTORY_VARIANTS}
    seed_root = PHYSIQ025_FROZEN_TRAJECTORY_ROOT / "seeds" / f"seed_{seed_value:06d}"
    if kind == "original":
        return (
            PHYSIQ025_OBJECT_QUERY_ROOT
            / "seeds"
            / f"seed_{seed_value:06d}"
            / "original.mp4"
        )
    if label not in labels:
        return None
    if kind == "video":
        suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
        return (
            seed_root
            / "apply"
            / label
            / stage
            / "videos"
            / "lora"
            / "cases"
            / PHYSIQ025_OBJECT_QUERY_CASE
            / f"top100_{suffix}.mp4"
        )
    if not name or Path(name).name != name:
        return None
    if kind == "trajectory":
        return seed_root / "trajectory" / label / "overlays" / name
    if kind == "apply":
        return seed_root / "apply" / label / stage / "overlays" / name
    return None


def physiq025_object_query_frozen_trajectory_catalog(
    seed: str, stage: str, step: str, branch: str
):
    try:
        seed_value = int(seed)
    except (TypeError, ValueError):
        seed_value = PHYSIQ025_FROZEN_TRAJECTORY_SEEDS[0]
    if seed_value not in PHYSIQ025_FROZEN_TRAJECTORY_SEEDS:
        seed_value = PHYSIQ025_FROZEN_TRAJECTORY_SEEDS[0]
    stage = stage if stage in {"all_steps", "steps00_09"} else "all_steps"
    branch = branch if branch in {"conditional", "unconditional"} else "conditional"
    max_step = 39 if stage == "all_steps" else 9
    try:
        step_value = max(0, min(max_step, int(step)))
    except (TypeError, ValueError):
        step_value = min(9, max_step)
    seed_root = PHYSIQ025_FROZEN_TRAJECTORY_ROOT / "seeds" / f"seed_{seed_value:06d}"
    objects = {
        "object_A": {"region_name": "object_A", "region_phrase": "brown tennis ball", "variants": {}},
        "object_B": {"region_name": "object_B", "region_phrase": "orange block", "variants": {}},
    }
    variants = []
    ready_videos = 0
    for label, title, probe, note in PHYSIQ025_FROZEN_TRAJECTORY_VARIANTS:
        trajectory_records = _physiq025_frozen_manifest(
            seed_root / "trajectory" / label / "overlays" / "manifest.json"
        )
        apply_records = _physiq025_frozen_manifest(
            seed_root / "apply" / label / stage / "overlays" / "manifest.json"
        )
        video_asset = physiq025_object_query_frozen_trajectory_asset(
            str(seed_value), stage, label, "video", ""
        )
        video_ready = bool(video_asset and video_asset.is_file())
        ready_videos += int(video_ready)
        variants.append(
            {"label": label, "title": title, "probe": probe, "note": note, "video_ready": video_ready}
        )
        for region in objects:
            trajectory = _physiq025_frozen_record(
                trajectory_records, step_value, branch, region
            )
            applied = _physiq025_frozen_record(apply_records, step_value, branch, region)
            objects[region]["variants"][label] = {
                "trajectory_images": trajectory.get("images", {}),
                "apply_images": applied.get("images", {}),
                "trajectory_ready": bool(trajectory),
                "apply_ready": bool(applied),
                "num_probe_heads": trajectory.get("num_heads"),
                "num_apply_heads": applied.get("num_heads"),
            }
    original = physiq025_object_query_frozen_trajectory_asset(
        str(seed_value), stage, "", "original", ""
    )
    return {
        "case": PHYSIQ025_OBJECT_QUERY_CASE,
        "seed": seed_value,
        "seeds": PHYSIQ025_FROZEN_TRAJECTORY_SEEDS,
        "stage": stage,
        "step": step_value,
        "branch": branch,
        "variants": variants,
        "records": list(objects.values()),
        "original_video_ready": bool(original and original.is_file()),
        "ready_videos": ready_videos,
        "expected_videos": len(PHYSIQ025_FROZEN_TRAJECTORY_VARIANTS),
    }


def physiq025_object_query_frozen_trajectory_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysIQ025 · Frozen Object Query Trajectory</title><style>
:root{--paper:#ebe8df;--ink:#16241f;--muted:#647069;--card:#fffdf7;--line:#b9b3a7;--orange:#b95632;--green:#176859;--blue:#2b627b;--gold:#b07b20}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 6% 0,#e79d5748,transparent 35rem),radial-gradient(circle at 96% 4%,#42927645,transparent 40rem),var(--paper);color:var(--ink);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:14px 22px;background:#ebe8dff2;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--green);font-weight:900}h1{margin:4px 0;font-size:clamp(27px,4vw,48px)}header p{margin:5px 0;color:var(--muted)}.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}select,button{padding:8px 10px;border:1px solid var(--line);background:#fff;font-weight:900;color:var(--ink)}.status{font:12px ui-monospace,monospace}.replay{position:fixed;right:18px;bottom:18px;z-index:40;border:0;border-radius:99px;background:var(--green);color:#fff;padding:13px 18px;box-shadow:0 8px 22px #173a31aa}main{width:min(2500px,calc(100% - 16px));margin:auto;padding:18px 0 90px}.videos{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:10px}.video,.variant,.object{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px}.video{border-top:6px solid var(--blue)}.video.variant-video{border-top-color:var(--orange)}.video h2{margin:4px 0;font-size:16px}.video p{min-height:32px;margin:5px 0;color:var(--muted);font-size:11px}.video video{display:block;width:100%;background:#111}.pending{min-height:170px;display:grid;place-items:center;border:1px dashed var(--line);color:var(--muted)}.variant{margin-top:15px;border-top:7px solid var(--green)}.variant>h2{margin:3px 0}.variant>.meta{margin:4px 0 10px;color:var(--muted)}.objects{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.object{overflow:auto}.object>h3{position:sticky;left:0;width:max-content;margin:0 0 8px;padding:5px 9px;background:var(--ink);color:#fff;z-index:2}.row{margin:8px 0 14px}.row h4{position:sticky;left:0;width:max-content;margin:0 0 5px;padding:4px 8px;background:#e8e1d4;z-index:2}.row img{display:block;min-width:2200px;width:100%;border:1px solid #d6cdbd}.row .pending{min-width:700px;min-height:90px}.legend{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0}.pill{padding:5px 8px;border:1px solid var(--line);background:#f7f4ec;font:11px ui-monospace,monospace}@media(max-width:900px){header{position:static}.objects{grid-template-columns:1fr}.videos{grid-template-columns:1fr}main{width:calc(100% - 8px)}}
</style></head><body><button class="replay" id="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/object-query-frozen-trajectory?v=23">001460 Frozen Trajectory</a><h1>PhysIQ025 · Frozen Object Query Trajectory</h1><p>Wan+LoRA · PCK@32 Top100 apply · Object A/B · 40-step · 49 frames · P95/P99 trajectory masks</p><div class="tools"><label>Seed <select id="seed"></select></label><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><label>Step <select id="step"></select></label><label>CFG <select id="branch"><option value="conditional">Conditional</option><option value="unconditional">Unconditional</option></select></label><label>Scheme <select id="variant"><option value="all">All schemes</option></select></label><button id="refresh">手动刷新</button><span class="status" id="status">读取中</span></div><div class="legend"><span class="pill">Top100 mean: standard masks</span><span class="pill">Top30 mean: Backtrack3 masks</span><span class="pill">Top100 heads: intervention applied</span></div></header><main><section id="videos" class="videos"></section><section id="content"></section></main><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let seed=q.get('seed')||'13161',stage=q.get('stage')||'all_steps',step=q.get('step')||'9',branch=q.get('branch')||'conditional',variant=q.get('variant')||'all',generation=Date.now(),data=null;const api='/api/physiq025-object-query-frozen-trajectory';
function sync(){const u=new URL(location.href);for(const [k,v] of Object.entries({seed,stage,step,branch,variant}))u.searchParams.set(k,v);history.replaceState(null,'',u)}
function asset(type,label,kind,name=''){return `${api}/${type}?seed=${seed}&stage=${stage}&label=${encodeURIComponent(label)}&kind=${kind}${name?`&name=${encodeURIComponent(name)}`:''}&v=${generation}`}
function media(ready,title,label,note){return `<article class="video ${label?'variant-video':''}"><h2>${esc(title)}</h2><p>${esc(note||'')}</p>${ready?`<video controls preload="metadata" playsinline src="${asset('video',label,label?'video':'original')}"></video>`:'<div class="pending">PENDING · 生成中</div>'}</article>`}
function row(title,label,kind,name){return `<section class="row"><h4>${esc(title)}</h4>${name?`<img loading="lazy" src="${asset('image',label,kind,name)}">`:'<div class="pending">PENDING · capture 尚未生成</div>'}</section>`}
function render(d){data=d;seed=String(d.seed);stage=d.stage;step=String(d.step);branch=d.branch;const seedSel=document.getElementById('seed');seedSel.innerHTML=d.seeds.map(x=>`<option value="${x}">${x}</option>`).join('');seedSel.value=seed;document.getElementById('stage').value=stage;const max=stage==='all_steps'?39:9,stepSel=document.getElementById('step');stepSel.innerHTML=Array.from({length:max+1},(_,i)=>`<option value="${i}">S${String(i).padStart(3,'0')}</option>`).join('');stepSel.value=step;document.getElementById('branch').value=branch;const variantSel=document.getElementById('variant');variantSel.innerHTML='<option value="all">All schemes</option>'+d.variants.map(v=>`<option value="${v.label}">${esc(v.title)}</option>`).join('');if(!d.variants.some(v=>v.label===variant))variant='all';variantSel.value=variant;const selected=d.variants.filter(v=>variant==='all'||v.label===variant);document.getElementById('status').textContent=`Videos ${d.ready_videos}/${d.expected_videos} · S${String(d.step).padStart(3,'0')} · ${d.branch}`;document.getElementById('videos').innerHTML=media(d.original_video_ready,'Wan+LoRA Original','','Same seed · no intervention')+selected.map(v=>media(v.video_ready,v.title,v.label,`${v.probe} · ${v.note}`)).join('');document.getElementById('content').innerHTML=selected.map(v=>`<article class="variant"><h2>${esc(v.title)}</h2><p class="meta">Mask source: ${esc(v.probe)} · ${esc(v.note)} · Applied to PCK@32 Top100 object-query rows</p><div class="objects">${d.records.map(r=>{const x=r.variants[v.label]||{},t=x.trajectory_images||{},a=x.apply_images||{};return `<section class="object"><h3>${esc(r.region_name)} · ${esc(r.region_phrase)}</h3>${row('1. No Intervention · Mean Attention',v.label,'trajectory',t.mean)}${row('2. All High-Response Candidates',v.label,'trajectory',t.candidate)}${t.forward_trajectory?row('3. Forward Trajectory',v.label,'trajectory',t.forward_trajectory):''}${t.rejected_events?row('4. Rejected Events',v.label,'trajectory',t.rejected_events):''}${t.backward_removed?row('5. Backward-Traced Removal',v.label,'trajectory',t.backward_removed):''}${row('6. Frozen Accepted Trajectory',v.label,'trajectory',t.trajectory)}${row('7. Frozen Removal Mask',v.label,'trajectory',t.forbidden)}${row('8. Intervention · Before',v.label,'apply',a.before)}${row('9. Intervention · After',v.label,'apply',a.after)}${row('10. Removed Attention Mass',v.label,'apply',a.removed)}</section>`}).join('')}</div></article>`).join('');sync()}
async function load(){generation=Date.now();document.getElementById('status').textContent='扫描产物中';const url=`${api}/catalog?seed=${seed}&stage=${stage}&step=${step}&branch=${branch}&v=${generation}`;const d=await fetch(url,{cache:'no-store'}).then(r=>r.json());render(d)}
for(const id of ['seed','stage','step','branch','variant'])document.getElementById(id).addEventListener('change',ev=>{if(id==='seed')seed=ev.target.value;if(id==='stage'){stage=ev.target.value;if(stage==='steps00_09'&&Number(step)>9)step='9'}if(id==='step')step=ev.target.value;if(id==='branch')branch=ev.target.value;if(id==='variant'){variant=ev.target.value;render(data);return}sync();load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


OBJECT_QUERY_TOP100_MEAN_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_top100_mean_case001460"
)
OBJECT_QUERY_GROUP_MEAN_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_group_mean_case001460"
)
OBJECT_QUERY_FROZEN_TRAJECTORY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_case001460"
)
OBJECT_QUERY_FROZEN_TRAJECTORY_10STEP_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_10step_case001460"
)
OBJECT_QUERY_DYNAMIC_COMMON_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_dynamic_common_case001460"
)
OBJECT_QUERY_TOP100_MEAN_SEEDS = (90094, 35075, 21890, 49530, 47326, 32466)


def object_query_top100_mean_asset(
    seed: str, stage: str, method: str, name: str
):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if (
        seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS
        or stage not in {"all_steps", "steps00_09"}
        or method not in {"identity", "old", "new"}
        or Path(name).name != name
        or not name.endswith(".jpg")
    ):
        return None
    seed_root = OBJECT_QUERY_TOP100_MEAN_ROOT / "seeds" / f"seed_{seed_value:06d}"
    method_stage = ("step39" if stage == "all_steps" else "step09") if method == "identity" else stage
    return seed_root / method / method_stage / "mean_overlays" / name


def object_query_top100_mean_video_asset(seed: str, stage: str, kind: str):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if (
        seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS
        or stage not in {"all_steps", "steps00_09"}
        or kind not in {"baseline", "old", "new"}
    ):
        return None
    seed_tag = f"seed_{seed_value:06d}"
    if kind == "baseline":
        return ATTENTION_LORA_SEED_SWEEP_ROOT / "seeds" / seed_tag / "original.mp4"
    suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
    method_root = (
        Path("/data/gaoya/agent-data/outputs/attention_lora_object_query_continuity_case001460")
        if kind == "old"
        else Path("/data/gaoya/agent-data/outputs/attention_lora_object_query_main_component_case001460")
    )
    return (
        method_root / "seeds" / seed_tag / stage / "videos" / "lora" / "cases"
        / ATTENTION_LORA_CASE / f"top100_{suffix}.mp4"
    )


def object_query_top100_mean_catalog(seed: str, stage: str):
    try:
        seed_value = int(seed)
    except ValueError:
        seed_value = OBJECT_QUERY_TOP100_MEAN_SEEDS[0]
    if seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS:
        seed_value = OBJECT_QUERY_TOP100_MEAN_SEEDS[0]
    if stage not in {"all_steps", "steps00_09"}:
        stage = "all_steps"
    seed_root = OBJECT_QUERY_TOP100_MEAN_ROOT / "seeds" / f"seed_{seed_value:06d}"

    def rows(method):
        method_stage = ("step39" if stage == "all_steps" else "step09") if method == "identity" else stage
        manifest = seed_root / method / method_stage / "mean_overlays" / "manifest.json"
        if not manifest.is_file():
            return []
        return json.loads(manifest.read_text(encoding="utf-8")).get("records", [])

    identity_rows, old_rows, new_rows = rows("identity"), rows("old"), rows("new")
    by_method = {
        method: {row["region_name"]: row for row in method_rows}
        for method, method_rows in (
            ("identity", identity_rows), ("old", old_rows), ("new", new_rows)
        )
    }
    records = []
    regions = sorted(set().union(*(set(value) for value in by_method.values())))
    for region in regions:
        identity = by_method["identity"].get(region, {})
        old = by_method["old"].get(region, {})
        new = by_method["new"].get(region, {})
        row = {
            "region_name": region,
            "region_phrase": (identity or old or new).get("region_phrase", region),
            "num_heads": max(
                int(identity.get("num_heads", 0)), int(old.get("num_heads", 0)),
                int(new.get("num_heads", 0)),
            ),
            "original_image": identity.get("images", {}).get("original"),
            "old_before_image": old.get("images", {}).get("before"),
            "old_after_image": old.get("images", {}).get("after"),
            "new_before_image": new.get("images", {}).get("before"),
            "new_after_image": new.get("images", {}).get("after"),
            "original_vmax": identity.get("shared_vmax"),
            "old_vmax": old.get("shared_vmax"),
            "new_vmax": new.get("shared_vmax"),
        }
        records.append(row)
    return {
        "seed": seed_value,
        "stage": stage,
        "seeds": list(OBJECT_QUERY_TOP100_MEAN_SEEDS),
        "records": records,
        "identity_ready": len(identity_rows),
        "old_ready": len(old_rows),
        "new_ready": len(new_rows),
        "baseline_ready": bool(
            (asset := object_query_top100_mean_video_asset(str(seed_value), stage, "baseline"))
            and asset.is_file() and asset.stat().st_size
        ),
        "old_video_ready": bool(
            (asset := object_query_top100_mean_video_asset(str(seed_value), stage, "old"))
            and asset.is_file() and asset.stat().st_size
        ),
        "new_video_ready": bool(
            (asset := object_query_top100_mean_video_asset(str(seed_value), stage, "new"))
            and asset.is_file() and asset.stat().st_size
        ),
    }


def object_query_top100_mean_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top100 Head Mean Object Query Attention</title><style>
:root{--paper:#eee8dc;--ink:#15241e;--line:#b8ad98;--card:#fffdf8;--old:#b44d31;--new:#176a5c;--blue:#285f7a}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99d5550,transparent 34rem),radial-gradient(circle at 97% 3%,#4b947750,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:15px 22px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(2450px,calc(100% - 16px));margin:auto;padding:18px 0 90px}.videos{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:18px}.video-card{padding:11px;background:var(--card);border:1px solid var(--line);border-radius:13px}.video-card.baseline{border-top:7px solid var(--blue)}.video-card.old{border-top:7px solid var(--old)}.video-card.new{border-top:7px solid var(--new)}.video-card h2{margin:0 0 5px}.video-card p{margin:0 0 8px;color:#5e685f;font:11px ui-monospace,monospace}.video-card video{display:block;width:100%;background:#111714;border-radius:7px}.object{overflow:auto;margin:14px 0;padding:10px;background:var(--card);border:1px solid var(--line);border-radius:13px}.object>h2{position:sticky;left:0;width:max-content;margin:0 0 9px}.row{margin:8px 0 15px}.row h3{position:sticky;left:0;width:max-content;margin:0 0 5px;padding:4px 9px;background:#ece4d6;border-left:6px solid var(--blue)}.row.old h3{border-left-color:var(--old)}.row.new h3{border-left-color:var(--new)}.row img{display:block;min-width:2260px;width:100%;border:1px solid #d8cfbf}.pending{padding:52px 15px;text-align:center;border:1px dashed var(--line)}@media(max-width:900px){header{position:static}.videos{grid-template-columns:1fr}}
</style></head><body><header><a href="/">返回总览</a> · <a href="/object-query-continuity-comparison?v=4">逐 Head 对比</a><h1>Top100 Head Mean · Object Query Attention</h1><p>0613pybullet_sample_001460_w002 · 每个 Head 内 SUM 8 Object Queries，再对完整 PCK@32 Top100 Head 求平均。</p><div class="tools"><label>Seed <select id="seed"><option>90094</option><option>35075</option><option>21890</option><option>49530</option><option>47326</option><option>32466</option></select></label><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="videos" class="videos"></section><section id="content"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let seed=q.get('seed')||'90094',stage=q.get('stage')||'all_steps';const api='/api/object-query-top100-mean-overlay',src=(method,name)=>`${api}/image?seed=${seed}&stage=${stage}&method=${method}&name=${encodeURIComponent(name)}`,fmt=v=>v==null?'pending':Number(v).toExponential(3),row=(title,cls,method,name,vmax)=>`<section class="row ${cls}"><h3>${title} · vmax=${fmt(vmax)}</h3>${name?`<img loading="lazy" src="${src(method,name)}">`:'<div class="pending">生成/聚合中</div>'}</section>`;function videoUrl(kind){return api+'/video?seed='+encodeURIComponent(seed)+'&stage='+encodeURIComponent(stage)+'&kind='+encodeURIComponent(kind)}function videoCard(kind,title,note,ready,cls){return '<article class="video-card '+cls+'"><h2>'+title+'</h2><p>'+note+'</p>'+(ready?'<video controls preload="metadata" playsinline src="'+videoUrl(kind)+'"></video>':'<div class="pending">视频尚未生成</div>')+'</article>'}function render(d){document.getElementById('videos').innerHTML=videoCard('baseline','Wan+LoRA Original','No Attention Intervention · 同一 seed 与推理配置',d.baseline_ready,'baseline')+videoCard('old','旧方案生成视频','P90 Multi-Component Continuity',d.old_video_ready,'old')+videoCard('new','新方案生成视频','P99 / Top-5 Main Connected Component',d.new_video_ready,'new');document.getElementById('seed').value=String(seed);document.getElementById('stage').value=stage;document.getElementById('status').textContent=`Original ${d.identity_ready}/2 · Old ${d.old_ready}/2 · New ${d.new_ready}/2`;document.getElementById('content').innerHTML=d.records.length?d.records.map(r=>`<article class="object"><h2>${e(r.region_name)} · ${e(r.region_phrase)} · ${r.num_heads}/100 heads</h2>${row('No Intervention · Top100 Mean','', 'identity',r.original_image,r.original_vmax)}${row('旧方案 Before · Top100 Mean','old','old',r.old_before_image,r.old_vmax)}${row('旧方案 After · Top100 Mean','old','old',r.old_after_image,r.old_vmax)}${row('新方案 Before · Top100 Mean','new','new',r.new_before_image,r.new_vmax)}${row('新方案 After · Top100 Mean','new','new',r.new_after_image,r.new_vmax)}</article>`).join(''):'<div class="pending">等待该 seed/stage 的 Top100 capture</div>'}async function load(){const d=await fetch(`${api}/catalog?seed=${seed}&stage=${stage}`,{cache:'no-store'}).then(r=>r.json());render(d)}for(const id of ['seed','stage'])document.getElementById(id).addEventListener('change',ev=>{if(id==='seed')seed=ev.target.value;else stage=ev.target.value;const u=new URL(location.href);u.searchParams.set('seed',seed);u.searchParams.set('stage',stage);history.replaceState(null,'',u);load()});document.getElementById('refresh').addEventListener('click',load);load();
</script></body></html>'''


def object_query_group_mean_asset(seed: str, stage: str, kind: str, name: str):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if (
        seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS
        or stage not in {"all_steps", "steps00_09"}
        or kind not in {"original", "group"}
        or Path(name).name != name
        or not name.endswith(".jpg")
    ):
        return None
    if kind == "original":
        return object_query_top100_mean_asset(seed, stage, "identity", name)
    return (
        OBJECT_QUERY_GROUP_MEAN_ROOT
        / "seeds"
        / f"seed_{seed_value:06d}"
        / stage
        / "overlays"
        / name
    )


def object_query_group_mean_video(seed: str, stage: str, kind: str):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if (
        seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS
        or stage not in {"all_steps", "steps00_09"}
        or kind not in {"original", "group"}
    ):
        return None
    case_root = (
        OBJECT_QUERY_GROUP_MEAN_ROOT
        / "seeds"
        / f"seed_{seed_value:06d}"
        / stage
        / "videos"
        / "lora"
        / "cases"
        / "0613pybullet_sample_001460_w002"
    )
    if kind == "original":
        return case_root / "original.mp4"
    suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
    return case_root / f"top100_{suffix}.mp4"


def object_query_group_mean_catalog(seed: str, stage: str):
    try:
        seed_value = int(seed)
    except ValueError:
        seed_value = OBJECT_QUERY_TOP100_MEAN_SEEDS[0]
    if seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS:
        seed_value = OBJECT_QUERY_TOP100_MEAN_SEEDS[0]
    if stage not in {"all_steps", "steps00_09"}:
        stage = "all_steps"
    group_root = (
        OBJECT_QUERY_GROUP_MEAN_ROOT
        / "seeds"
        / f"seed_{seed_value:06d}"
        / stage
    )
    group_manifest = load_payload(group_root / "overlays" / "manifest.json") or {}
    group_rows = {
        row["region_name"]: row for row in group_manifest.get("records", [])
    }
    identity_catalog = object_query_top100_mean_catalog(str(seed_value), stage)
    identity_rows = {
        row["region_name"]: row for row in identity_catalog.get("records", [])
    }
    records = []
    for region_name in sorted(set(group_rows) | set(identity_rows)):
        group_row = group_rows.get(region_name, {})
        identity_row = identity_rows.get(region_name, {})
        records.append(
            {
                "region_name": region_name,
                "region_phrase": (group_row or identity_row).get("region_phrase", region_name),
                "num_heads": int(group_row.get("num_heads", identity_row.get("num_heads", 0))),
                "original_image": identity_row.get("original_image"),
                "images": group_row.get("images", {}),
                "scale_mode": group_row.get("scale_mode", "per_frame_before_after_shared"),
                "frame_vmax": group_row.get("frame_vmax", []),
                "removed_frame_vmax": group_row.get("removed_frame_vmax", []),
            }
        )
    return {
        "seed": seed_value,
        "stage": stage,
        "seeds": list(OBJECT_QUERY_TOP100_MEAN_SEEDS),
        "records": records,
        "identity_ready": identity_catalog.get("identity_ready", 0),
        "group_ready": len(group_rows),
        "original_video_ready": bool(
            (asset := object_query_group_mean_video(str(seed_value), stage, "original"))
            and asset.is_file()
        ),
        "group_video_ready": bool(
            (asset := object_query_group_mean_video(str(seed_value), stage, "group"))
            and asset.is_file()
        ),
    }


def object_query_group_mean_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top100 Group-Mean Continuity</title><style>
:root{--paper:#eee8dc;--ink:#15241e;--line:#b8ad98;--card:#fffdf8;--red:#b1432d;--green:#176a5c;--gold:#b78024;--blue:#285f7a}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99d5550,transparent 34rem),radial-gradient(circle at 97% 3%,#4b947750,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:15px 22px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(2450px,calc(100% - 16px));margin:auto;padding:18px 0 90px}.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:11px}.video,.object{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:10px}.video video{display:block;width:100%;background:#111}.object{overflow:auto;margin:14px 0}.object>h2{position:sticky;left:0;width:max-content;margin:0 0 9px}.row{margin:8px 0 18px}.row h3{position:sticky;left:0;width:max-content;margin:0 0 4px;padding:4px 9px;background:#ece4d6;border-left:6px solid var(--blue)}.row.mask h3{border-left-color:var(--gold)}.row.after h3{border-left-color:var(--green)}.row.removed h3{border-left-color:var(--red)}.explain{position:sticky;left:0;width:min(1180px,calc(100vw - 52px));margin:0 0 8px;padding:8px 11px;background:#f5f0e6;border:1px solid #d7cdbb;border-radius:7px;font:13px/1.55 "Noto Sans SC","Source Han Sans SC",sans-serif;color:#405048}.explain b{color:var(--ink)}.row img{display:block;min-width:2260px;width:100%;border:1px solid #d8cfbf}.pending{padding:52px 15px;text-align:center;border:1px dashed var(--line)}.replay{position:fixed;right:18px;bottom:18px;z-index:30;border:0;border-radius:99px;padding:13px 18px;background:var(--green);color:#fff}@media(max-width:900px){header{position:static}.videos{grid-template-columns:1fr}}
</style></head><body><button id="replay" class="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/object-query-top100-mean-overlay?v=1">Top100 Mean 对照</a><h1>Top100 Group-Mean Continuity</h1><p>Probe: SUM 8 object queries → MEAN Top100 heads；P90 候选中移除不邻接上一帧 Top-5 主连通分量的 key，再把同一 forbidden mask 回写全部 Top100 heads。</p><div class="tools"><label>Seed <select id="seed"></select></label><label>Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="videos" class="videos"></section><section id="content"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let seed=q.get('seed')||'90094',stage=q.get('stage')||'all_steps';const api='/api/object-query-group-mean-continuity',img=(kind,name)=>`${api}/image?seed=${seed}&stage=${stage}&kind=${kind}&name=${encodeURIComponent(name)}`,vid=kind=>`${api}/video?seed=${seed}&stage=${stage}&kind=${kind}`,row=(title,cls,kind,name,description,scale='')=>`<section class="row ${cls}"><h3>${title}${scale?` · ${scale}`:''}</h3><p class="explain">${description}</p>${name?`<img loading="lazy" src="${img(kind,name)}">`:'<div class="pending">生成/聚合中</div>'}</section>`:'';function sync(){const u=new URL(location.href);u.searchParams.set('seed',seed);u.searchParams.set('stage',stage);history.replaceState(null,'',u)}function render(d){const sel=document.getElementById('seed');if(sel.options.length!==d.seeds.length)sel.innerHTML=d.seeds.map(x=>`<option>${x}</option>`).join('');sel.value=String(seed);document.getElementById('stage').value=stage;document.getElementById('status').textContent=`Original ${d.identity_ready}/2 · Group ${d.group_ready}/2 · color scale: per frame`;document.getElementById('videos').innerHTML=`<article class="video"><h2>Wan+LoRA Original</h2><h3>40-step inference</h3><p>同一 seed、40 个去噪步、49 帧，不修改任何 attention，用作视频质量与运动变化基线。</p>${d.original_video_ready?`<video controls preload="metadata" playsinline src="${vid('original')}"></video>`:'<div class="pending">等待 baseline</div>'}<h3>10-step inference</h3><video controls preload="metadata" playsinline onerror="this.style.visibility='hidden'" onloadeddata="this.style.visibility='visible'" src="${video('tenstep_original_video')}"></video></article><article class="video"><h2>Top100 Group-Mean Intervention · ${stage==='all_steps'?'S000-S039':'S000-S009'}</h2><p>对 PCK@32 Top100 heads 使用组级 forbidden mask；右侧视频是该 mask 实际回写 attention 后的生成结果。</p>${d.group_video_ready?`<video controls preload="metadata" playsinline src="${vid('group')}"></video>`:'<div class="pending">等待双遍推理</div>'}</article>`;document.getElementById('content').innerHTML=d.records.length?d.records.map(r=>`<article class="object"><h2>${e(r.region_name)} · ${e(r.region_phrase)} · ${r.num_heads}/100 heads</h2>${row('1. No Intervention · Top100 Mean','', 'original',r.original_image,'<b>无干预参照。</b>每个 latent key 帧独立计算色标，不再让强帧压暗其他帧。')}${row('2. Group Before · Top100 Mean','', 'group',r.images.before,'<b>干预前的真实 input attention。</b>每个 K 帧独立缩放，但同一个 K 帧的 Before/After 共用局部 vmax，因此帧内可公平比较。','per-frame shared scale')}${row('3. Current Candidate · P90','mask','group',r.images.p90,'<b>当前帧待检查的高响应候选。</b>它是逐帧计算的二值 mask，不受连续值热力图色标影响。')}${row('4. Previous Anchor · Top-5 Main Connected Component','mask','group',r.images.main_component,'<b>上一帧连续性锚点。</b>逐帧 Top-5 后保留包含峰值的主连通分量；二值显示不做色标归一化。')}${row('5. Forbidden Mask · P90 outside radius-1 neighborhood','removed','group',r.images.forbidden,'<b>最终施加的二值 key mask。</b>当前帧 P90 中不邻接上一帧主连通分量的位置；K00/K01 不修改。')}${row('6. Group After · Top100 Mean','after','group',r.images.after,'<b>干预后的 attention。</b>逐帧缩放，并与同一帧 Before 共用 vmax；不同 K 帧之间不共享 vmax。','per-frame shared scale')}${row('7. Removed Attention Mass','removed','group',r.images.removed,'<b>被移除的注意力质量。</b>每个 K 帧独立缩放，以显示弱响应帧中实际发生的删除；颜色不能直接用于比较不同 K 帧的绝对大小。','per-frame scale')}</article>`).join(''):'<div class="pending">等待该 seed/stage 的 group-mean capture</div>'}async function load(){const d=await fetch(`${api}/catalog?seed=${seed}&stage=${stage}`,{cache:'no-store'}).then(r=>r.json());render(d)}for(const id of ['seed','stage'])document.getElementById(id).addEventListener('change',ev=>{if(id==='seed')seed=ev.target.value;else stage=ev.target.value;sync();load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script></body></html>'''


def object_query_frozen_trajectory_asset(seed: str, stage: str, kind: str, name: str):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS or stage not in {"all_steps", "steps00_09"}:
        return None
    seed_root = OBJECT_QUERY_FROZEN_TRAJECTORY_ROOT / "seeds" / f"seed_{seed_value:06d}"
    if kind in {
        "dynamic40_track", "dynamic10_track",
        "dynamic40_mean", "dynamic10_mean",
    }:
        if Path(name).name != name or not name.endswith(".jpg"):
            return None
        schedule = "steps40" if kind.startswith("dynamic40_") else "steps10"
        dynamic_root = (
            OBJECT_QUERY_DYNAMIC_COMMON_ROOT / "seeds" / f"seed_{seed_value:06d}" / schedule
        )
        if kind.endswith("_track"):
            return dynamic_root / "query_token_overlays" / name
        return dynamic_root / "attention_overlays" / name
    if kind.startswith("tenstep_"):
        tenstep_kind = kind.removeprefix("tenstep_")
        tenstep_root = (
            OBJECT_QUERY_FROZEN_TRAJECTORY_10STEP_ROOT
            / "seeds"
            / f"seed_{seed_value:06d}"
        )
        if tenstep_kind == "original_video":
            return (
                tenstep_root / "probe_top100" / "videos" / "lora" / "cases"
                / "0613pybullet_sample_001460_w002" / "original.mp4"
            )
        tenstep_labels = {
            "common_top100",
            "p95", "p99", "p95_single", "p99_single",
            "p95_single_d1", "p99_single_d1",
            "p95_single_bt3_d1", "p99_single_bt3_d1",
        }
        if tenstep_kind.endswith("_video"):
            label = tenstep_kind.rsplit("_video", 1)[0]
            if label in tenstep_labels:
                return (
                    tenstep_root / "apply" / label / "videos" / "lora" / "cases"
                    / "0613pybullet_sample_001460_w002" / "top100_steps_00_40.mp4"
                )
        if tenstep_kind.endswith("_trajectory") and Path(name).name == name and name.endswith(".jpg"):
            label = tenstep_kind.rsplit("_trajectory", 1)[0]
            if label in tenstep_labels:
                if label == "common_top100":
                    return tenstep_root / "common_top100" / "overlays" / name
                return tenstep_root / "trajectory" / label / "overlays" / name
        if tenstep_kind.endswith("_apply") and Path(name).name == name and name.endswith(".jpg"):
            label = tenstep_kind.rsplit("_apply", 1)[0]
            if label in tenstep_labels:
                return tenstep_root / "apply" / label / "overlays" / name
        return None
    if kind in {"p95_trajectory", "p99_trajectory", "p95_single_trajectory", "p99_single_trajectory", "p95_single_d1_trajectory", "p99_single_d1_trajectory", "p95_single_bt3_d1_trajectory", "p99_single_bt3_d1_trajectory"} and Path(name).name == name and name.endswith(".jpg"):
        label = kind.rsplit("_trajectory", 1)[0]
        return seed_root / "trajectory" / label / "overlays" / name
    if kind in {"p95_apply", "p99_apply", "p95_single_apply", "p99_single_apply", "p95_single_d1_apply", "p99_single_d1_apply", "p95_single_bt3_d1_apply", "p99_single_bt3_d1_apply"} and Path(name).name == name and name.endswith(".jpg"):
        label = kind.rsplit("_apply", 1)[0]
        return seed_root / "apply" / label / stage / "overlays" / name
    if kind == "original_video":
        return (
            Path("/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460")
            / "seeds" / f"seed_{seed_value:06d}" / "original.mp4"
        )
    if kind in {"p95_video", "p99_video", "p95_single_video", "p99_single_video", "p95_single_d1_video", "p99_single_d1_video", "p95_single_bt3_d1_video", "p99_single_bt3_d1_video"}:
        label = kind.rsplit("_video", 1)[0]
        suffix = "steps_00_40" if stage == "all_steps" else "steps_00_10"
        return (
            seed_root / "apply" / label / stage / "videos" / "lora" / "cases"
            / "0613pybullet_sample_001460_w002" / f"top100_{suffix}.mp4"
        )
    return None


def object_query_frozen_trajectory_catalog(seed: str, stage: str, step: str, branch: str):
    try:
        seed_value = int(seed)
    except ValueError:
        seed_value = 47326
    if seed_value not in OBJECT_QUERY_TOP100_MEAN_SEEDS:
        seed_value = 47326
    if stage not in {"all_steps", "steps00_09"}:
        stage = "all_steps"
    try:
        step_value = min(39, max(0, int(step)))
    except ValueError:
        step_value = 39
    if branch not in {"conditional", "unconditional"}:
        branch = "conditional"
    seed_root = OBJECT_QUERY_FROZEN_TRAJECTORY_ROOT / "seeds" / f"seed_{seed_value:06d}"
    variants = {}
    labels = (
        "p95", "p99", "p95_single", "p99_single",
        "p95_single_d1", "p99_single_d1",
        "p95_single_bt3_d1", "p99_single_bt3_d1",
    )
    for label in labels:
        trajectory_payload = load_payload(seed_root / "trajectory" / label / "overlays" / "manifest.json") or {}
        apply_payload = load_payload(seed_root / "apply" / label / stage / "overlays" / "manifest.json") or {}
        trajectory = {
            row["region_name"]: row
            for row in trajectory_payload.get("records", [])
            if int(row.get("step", -1)) == step_value and row.get("cfg_branch") == branch
        }
        applied = {
            row["region_name"]: row
            for row in apply_payload.get("records", [])
            if int(row.get("step", -1)) == step_value and row.get("cfg_branch") == branch
        }
        variants[label] = {"trajectory": trajectory, "applied": applied}
    records = []
    regions = set()
    for payload in variants.values():
        regions.update(payload["trajectory"])
        regions.update(payload["applied"])
    for region in sorted(regions):
        sources = {
            label: payload["trajectory"].get(region, {})
            for label, payload in variants.items()
        }
        results = {
            label: payload["applied"].get(region, {})
            for label, payload in variants.items()
        }
        representative = sources["p95"] or sources["p99"] or results["p95"] or results["p99"]
        records.append({
            "region_name": region,
            "region_phrase": representative.get("region_phrase", region),
            "num_heads": max(
                [int(row.get("num_heads", 0)) for row in [*sources.values(), *results.values()]]
            ),
            "variants": {
                label: {
                    "quantile": sources[label].get("quantile", 0.95 if label.startswith("p95") else 0.99),
                    "radius": sources[label].get("radius", 2),
                    "trajectory_images": sources[label].get("images", {}),
                    "apply_images": results[label].get("images", {}),
                }
                for label in labels
            },
        })
    return {
        "seed": seed_value, "stage": stage, "step": step_value, "branch": branch,
        "seeds": list(OBJECT_QUERY_TOP100_MEAN_SEEDS), "records": records,
        "ready": {
            label: {
                "trajectory": len(variants[label]["trajectory"]),
                "apply": len(variants[label]["applied"]),
                "video": bool((asset := object_query_frozen_trajectory_asset(str(seed_value), stage, f"{label}_video", "")) and asset.is_file()),
            }
            for label in labels
        },
        "original_video_ready": bool((asset := object_query_frozen_trajectory_asset(str(seed_value), stage, "original_video", "")) and asset.is_file()),
    }


def object_query_frozen_trajectory_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen Object Trajectory Mask</title><style>
:root{--paper:#eee8dc;--ink:#15241e;--line:#b8ad98;--card:#fffdf8;--red:#b1432d;--green:#176a5c;--gold:#b78024;--blue:#285f7a;--p95:#176a5c;--p99:#b1432d}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99d5550,transparent 34rem),radial-gradient(circle at 97% 3%,#4b947750,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:20;padding:15px 22px;background:#eee8dcee;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}h1{margin:3px 0;font-size:clamp(28px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace}main{width:min(2450px,calc(100% - 16px));margin:auto;padding:18px 0 90px}.videos{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:16px;align-items:start}.experiment-matrix{grid-column:1/-1;display:grid;gap:18px;margin-bottom:8px}.matrix-hero{display:flex;justify-content:space-between;gap:18px;align-items:end;padding:18px 20px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(115deg,#173f35,#285f52);color:#fff;box-shadow:0 14px 36px #173f3522}.matrix-hero h2{margin:0;font-size:clamp(22px,3vw,34px)}.matrix-hero p{max-width:760px;margin:4px 0 0;color:#e6f1eb}.experiment-group{padding:15px;border:1px solid var(--line);border-radius:16px;background:#f8f3e9cc;box-shadow:0 8px 28px #59492e12}.group-heading{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:0 2px 11px}.group-heading h2{margin:0;font:900 20px/1.2 "Noto Sans SC","Source Han Sans SC",sans-serif}.group-heading p{margin:0;color:#665d50;font:12px/1.4 ui-monospace,monospace}.experiment-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}.legacy-heading{grid-column:1/-1;margin:8px 0 -2px;padding:10px 3px;border-bottom:1px solid var(--line);font:900 17px/1.2 "Noto Sans SC","Source Han Sans SC",sans-serif;color:var(--green)}.video,.object{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px}.video{min-width:0;overflow:hidden;box-shadow:0 7px 22px #57482e12}.video h2{margin:2px 0 5px;font-size:18px}.video h3{min-height:32px;margin:0 0 8px;color:#6d6252;font:700 12px/1.35 ui-monospace,monospace}.video p{margin:9px 1px 2px;color:#5f574b;font:12px/1.5 "Noto Sans SC","Source Han Sans SC",sans-serif}.experiment-note{margin-top:10px!important;padding:9px 10px;border-left:4px solid var(--gold);border-radius:4px;background:#f5ecdc!important;color:#403a31!important;font:13px/1.55 "Noto Sans SC","Source Han Sans SC",sans-serif!important}.experiment-note strong{color:#8a5c12}.eyebrow{display:inline-block;margin-bottom:8px;padding:4px 8px;border-radius:99px;background:#e9dfcd;color:#735d31;font:900 10px/1 ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase}.video.p95{border-top:7px solid var(--p95)}.video.p99{border-top:7px solid var(--p99)}.video.single{box-shadow:inset 0 0 0 3px #d9a441}.video.dilate{outline:3px dashed #285f7a;outline-offset:-6px}.video video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111;border-radius:8px}.media-frame{position:relative;width:100%}.media-pending[hidden]{display:none!important}.media-pending{display:grid;place-items:center;width:100%;min-height:180px;padding:22px;border:1px dashed #a99b82;border-radius:8px;background:repeating-linear-gradient(135deg,#f3ecdf,#f3ecdf 12px,#ebe2d2 12px,#ebe2d2 24px);color:#6d6252;text-align:center;font:800 12px/1.5 ui-monospace,monospace}.video .media-pending{aspect-ratio:16/9;min-height:0}.row .media-frame{min-width:2260px}.row .media-pending{height:320px;font-size:16px}.object{overflow:auto;margin:14px 0}.object>h2{position:sticky;left:0;width:max-content;margin:0 0 9px}.row{margin:8px 0 18px}.row h3{position:sticky;left:0;width:max-content;margin:0 0 4px;padding:4px 9px;background:#ece4d6;border-left:6px solid var(--blue)}.row.p95 h3{border-left-color:var(--p95)}.row.p99 h3{border-left-color:var(--p99)}.row.single h3{box-shadow:inset 0 -3px #d9a441}.row.dilate h3{outline:2px dashed #285f7a}.explain{position:sticky;left:0;width:min(1180px,calc(100vw - 52px));margin:0 0 8px;padding:8px 11px;background:#f5f0e6;border:1px solid #d7cdbb;border-radius:7px;font:13px/1.55 "Noto Sans SC","Source Han Sans SC",sans-serif}.row img{display:block;min-width:2260px;width:100%;border:1px solid #d8cfbf}.pending{display:grid;place-items:center;min-height:180px;padding:34px 15px;text-align:center;border:1px dashed var(--line);border-radius:8px;background:#f4eee2;color:#756a59;font:800 12px/1.5 ui-monospace,monospace}.replay{position:fixed;right:18px;bottom:18px;z-index:30;border:0;border-radius:99px;padding:13px 18px;background:var(--green);color:#fff}@media(max-width:1500px){.videos{grid-template-columns:repeat(2,minmax(280px,1fr))}.experiment-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:900px){header{position:static}.videos,.experiment-grid{grid-template-columns:1fr}.matrix-hero,.group-heading{align-items:flex-start;flex-direction:column}.row .media-frame{min-width:1280px}}
.heatmap-video-compare{margin:0 0 18px;padding:13px;border:1px solid var(--line);border-radius:15px;background:var(--card)}.heatmap-video-compare>h2{margin:0 0 4px}.heatmap-video-compare>.explain{margin:0 0 11px}.heatmap-video-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.heatmap-video-card{padding:10px;border:1px solid var(--line);border-radius:11px;background:#fffdf8}.heatmap-video-card.baseline{border-top:7px solid #285f7a}.heatmap-video-card.generated{border-top:7px solid #176a5c}.heatmap-video-card h3{margin:0 0 4px}.heatmap-video-card p{margin:0 0 8px;color:#5b675f;font:11px ui-monospace,monospace}.heatmap-video-card video{display:block;width:100%;background:#111714;border-radius:7px}@media(max-width:900px){.heatmap-video-grid{grid-template-columns:1fr}}
</style></head><body><button id="replay" class="replay">重新播放全部</button><header><a href="/">返回总览</a> · <a href="/object-query-group-mean-continuity?v=3">上一版 Group Mean</a><h1>Old Multi-Component vs New Single-Component</h1><p>旧方案共用 No-Intervention Top100 Mean；BT3 方案改用 LoRA PCK@32 Top30 Mean 生成 mask，但仍作用于 Top100 heads。旧方案允许多个连续 components；新方案每帧只保留一个空间连续 component，其余高响应全部进入 frozen F_t。</p><div class="tools"><label>Seed <select id="seed"></select></label><label>Apply Stage <select id="stage"><option value="all_steps">S000-S039</option><option value="steps00_09">S000-S009</option></select></label><label>Denoise Step <select id="step"></select></label><label>CFG Branch <select id="branch"><option value="conditional">Conditional</option><option value="unconditional">Unconditional</option></select></label><label>Visualization <select id="viz"><option value="common">Common Top100 Mean · 40 vs 10</option><option value="reverse">Reverse 40→10 Head Match</option><option value="p95_old">P95 · Old Multi</option><option value="p99_old">P99 · Old Multi</option><option value="p95_single">P95 · Single</option><option value="p99_single">P99 · Single</option><option value="p95_dilate">P95 · Single + Dilate1</option><option value="p99_dilate">P99 · Single + Dilate1</option><option value="p95_bt3">P95 · Top30Mean + BT3</option><option value="p99_bt3">P99 · Top30Mean + BT3</option><option value="all">All Experiments</option></select></label><label>热力图实验 <select id="heatmap"><option value="similarity_transplant">Similarity 匹配替换</option><option value="sigma_transplant">Sigma 对齐替换</option><option value="same_index_transplant">Same-Index 前十步替换</option><option value="same_index_delta">Same-Index 正差 Mask 1×1</option><option value="similarity_delta">Similarity 正差 Mask 1×1</option><option value="s09_fixed_1">S09 固定 Mask 1×1</option><option value="s09_fixed_2">S09 固定 Mask 2×2</option><option value="s09_fixed_3">S09 固定 Mask 3×3</option><option value="s09_no_renorm_1">S09 No-Renorm Mask 1×1</option><option value="s09_no_renorm_2">S09 No-Renorm Mask 2×2</option><option value="s09_no_renorm_3">S09 No-Renorm Mask 3×3</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="videos" class="videos"></section><section id="heatmapVideoCompare" class="heatmap-video-compare" hidden></section><section id="content"></section></main><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams;let seed=q.get('seed')||'47326',stage=q.get('stage')||'all_steps',step=q.get('step')||'9',branch=q.get('branch')||'conditional',viz=q.get('viz')||'common',heatmap=q.get('heatmap')||'similarity_transplant';const api='/api/object-query-frozen-trajectory',image=(kind,name)=>`${api}/image?seed=${seed}&stage=${stage}&kind=${kind}&name=${encodeURIComponent(name)}`,video=kind=>`${api}/video?seed=${seed}&stage=${stage}&kind=${kind}`,groupFor=title=>{if(title.includes('Reverse'))return 'reverse';if(title.includes('Common'))return 'common';const p=title.includes('P99')?'p99':'p95';if(title.includes('BT3'))return p+'_bt3';if(title.includes('Dilate'))return p+'_dilate';if(title.includes('Single'))return p+'_single';return p+'_old'},row=(title,cls,kind,name,text)=>(viz==='all'||groupFor(title)===viz)?`<section class="row ${cls}"><h3>${title}</h3><p class="explain">${text}</p>${name?`<img loading="lazy" src="${image(kind,name)}">`:'<div class="pending">等待该步骤 capture</div>'}</section>`:'';const heatmapVideoSpecs={similarity_transplant:["Similarity-Matched A10 Transplant","Cosine-matched 10-step attention replaces selected 40-step object-query rows.","/api/object-query-reverse-transplant/video?v=2"],sigma_transplant:["Sigma-Matched A10 Transplant","Nearest log-sigma 10-step attention replaces selected 40-step object-query rows.","/api/object-query-sigma-transplant/video?v=1"],same_index_transplant:["Same-Index S00-S09 A10 Transplant","S00→S00 through S09→S09; later denoising steps remain unchanged.","/api/object-query-same-index-transplant/video?v=1"],same_index_delta:["Same-Index Positive-Delta P95 Mask 1×1","Remove same-index positive-delta P95 regions and renormalize each attention row.","/api/object-query-positive-delta-mask/video?v=1"],similarity_delta:["Similarity-Matched Positive-Delta P95 Mask 1×1","Build removal masks from cosine-matched 10-step attention across S000-S039.","/api/object-query-similarity-delta-mask/video?v=1"],s09_fixed_1:["S09-Frozen Mask 1×1","Apply the fixed S09 positive-delta mask without spatial expansion.","/api/object-query-s09-fixed-mask/video?kernel=1&v=1"],s09_fixed_2:["S09-Frozen Mask 2×2","Apply the fixed S09 positive-delta mask with 2×2 expansion.","/api/object-query-s09-fixed-mask/video?kernel=2&v=1"],s09_fixed_3:["S09-Frozen Mask 3×3","Apply the fixed S09 positive-delta mask with symmetric 3×3 expansion.","/api/object-query-s09-fixed-mask/video?kernel=3&v=1"],s09_no_renorm_1:["S09-Frozen No-Renorm 1×1","Zero the fixed S09 mask and use A_masked @ V without row renormalization.","/api/object-query-s09-fixed-mask-no-renorm/video?kernel=1&v=1"],s09_no_renorm_2:["S09-Frozen No-Renorm 2×2","Use the expanded 2×2 mask without row renormalization.","/api/object-query-s09-fixed-mask-no-renorm/video?kernel=2&v=1"],s09_no_renorm_3:["S09-Frozen No-Renorm 3×3","Use the expanded 3×3 mask without row renormalization.","/api/object-query-s09-fixed-mask-no-renorm/video?kernel=3&v=1"]};function heatmapVideoCard(cls,title,note,src){return'<article class="heatmap-video-card '+cls+'"><h3>'+title+'</h3><p>'+note+'</p><video controls preload="metadata" playsinline src="'+src+'"></video></article>'}function renderHeatmapVideos(){const box=document.getElementById("heatmapVideoCompare");if(viz!=="reverse"&&viz!=="all"){box.hidden=true;box.innerHTML="";return}const spec=heatmapVideoSpecs[heatmap]||heatmapVideoSpecs.similarity_transplant;box.hidden=false;box.innerHTML="<h2>当前热力图对应视频</h2><p class=\"explain\">左侧为同 seed、40-step、49-frame 的无干预 Baseline；右侧为当前热力图实验实际回写 attention 后的生成结果。</p><div class=\"heatmap-video-grid\">"+heatmapVideoCard("baseline","Wan+LoRA Original Baseline","No Attention Intervention · seed "+seed,video("original_video"))+heatmapVideoCard("generated",spec[0],spec[1],spec[2])+"</div>";decoratePendingMedia(box)}function sync(){const u=new URL(location.href);for(const [k,v] of Object.entries({seed,stage,step,branch,viz,heatmap}))u.searchParams.set(k,v);history.replaceState(null,'',u)}function render(d){const ss=document.getElementById('seed');if(ss.options.length!==d.seeds.length)ss.innerHTML=d.seeds.map(x=>`<option>${x}</option>`).join('');ss.value=String(seed);const ds=document.getElementById('step');if(!ds.options.length)ds.innerHTML=Array.from({length:40},(_,i)=>`<option value="${i}">S${String(i).padStart(3,'0')}</option>`).join('');ds.value=String(step);document.getElementById('stage').value=stage;document.getElementById('branch').value=branch;document.getElementById('viz').value=viz;document.getElementById('heatmap').value=heatmap;document.getElementById('status').textContent=`S${String(d.step).padStart(3,'0')} ${d.branch} · old ${d.ready.p95.apply}/2/${d.ready.p99.apply}/2 · single ${d.ready.p95_single.apply}/2/${d.ready.p99_single.apply}/2 · dilate1 ${d.ready.p95_single_d1.apply}/2/${d.ready.p99_single_d1.apply}/2 · top30mean-bt3 ${d.ready.p95_single_bt3_d1.apply}/2/${d.ready.p99_single_bt3_d1.apply}/2`;document.getElementById('videos').innerHTML=`<article class="video"><h2>Wan+LoRA Original</h2><h3>40-step inference</h3>${d.original_video_ready?`<video controls preload="metadata" playsinline src="${video('original_video')}"></video>`:'<div class="pending">等待 baseline</div>'}<h3>10-step inference</h3><video controls preload="metadata" playsinline onerror="this.style.visibility='hidden'" onloadeddata="this.style.visibility='visible'" src="${video('tenstep_original_video')}"></video></article><article class="video p95"><h2>Old P95 · Multi</h2><h3>40-step inference</h3>${d.ready.p95.video?`<video controls preload="metadata" playsinline src="${video('p95_video')}"></video>`:'<div class="pending">等待旧 P95</div>'}</article><article class="video p99"><h2>Old P99 · Multi</h2><h3>40-step inference</h3>${d.ready.p99.video?`<video controls preload="metadata" playsinline src="${video('p99_video')}"></video>`:'<div class="pending">等待旧 P99</div>'}</article><article class="video p95 single"><h2>New P95 · Single</h2><h3>40-step inference</h3>${d.ready.p95_single.video?`<video controls preload="metadata" playsinline src="${video('p95_single_video')}"></video>`:'<div class="pending">等待新 P95</div>'}</article><article class="video p99 single"><h2>New P99 · Single</h2><h3>40-step inference</h3>${d.ready.p99_single.video?`<video controls preload="metadata" playsinline src="${video('p99_single_video')}"></video>`:'<div class="pending">等待新 P99</div>'}</article><article class="video p95 single dilate"><h2>P95 · Single + Dilate1</h2><h3>40-step inference</h3>${d.ready.p95_single_d1.video?`<video controls preload="metadata" playsinline src="${video('p95_single_d1_video')}"></video>`:'<div class="pending">等待膨胀 P95</div>'}</article><article class="video p99 single dilate"><h2>P99 · Single + Dilate1</h2><h3>40-step inference</h3>${d.ready.p99_single_d1.video?`<video controls preload="metadata" playsinline src="${video('p99_single_d1_video')}"></video>`:'<div class="pending">等待膨胀 P99</div>'}</article><article class="video p95 single dilate"><h2>P95 · Top30 Mean + Backtrack3 + Dilate1</h2><h3>40-step inference</h3>${d.ready.p95_single_bt3_d1.video?`<video controls preload="metadata" playsinline src="${video('p95_single_bt3_d1_video')}"></video>`:'<div class="pending">等待回溯 P95</div>'}</article><article class="video p99 single dilate"><h2>P99 · Top30 Mean + Backtrack3 + Dilate1</h2><h3>40-step inference</h3>${d.ready.p99_single_bt3_d1.video?`<video controls preload="metadata" playsinline src="${video('p99_single_bt3_d1_video')}"></video>`:'<div class="pending">等待回溯 P99</div>'}</article>`;document.getElementById('content').innerHTML=d.records.length?d.records.map(r=>{const a=r.variants.p95,b=r.variants.p99,c=r.variants.p95_single,dv=r.variants.p99_single,e=r.variants.p95_single_d1,f=r.variants.p99_single_d1,g=r.variants.p95_single_bt3_d1,h=r.variants.p99_single_bt3_d1,tenMean=`seed${String(seed).padStart(6,'0')}__step${String(d.step).padStart(2,'0')}__${d.branch}__${r.region_name}__mean.jpg`;return `<article class="object"><h2>${esc(r.region_name)} · ${esc(r.region_phrase)} · ${r.num_heads}/100 heads</h2>${row('1a. 40-Step Common No Intervention · Top100 Mean','', 'p95_trajectory',a.trajectory_images.mean,'40 个去噪步下的无干预 Top100 Mean；每个 latent frame 独立色标。')}${row('1b. 10-Step Common No Intervention · Top100 Mean','', 'tenstep_common_top100_trajectory',tenMean,'10 个去噪步下重新捕获的无干预 Top100 Mean，不复用 40-step attention；仅 S000-S009 有对应结果。')}${row('2. P95 · Raw High-Response Candidates','p95','p95_trajectory',a.trajectory_images.candidate,'共同的原始 P95 候选。')}${row('3. Old P95 · Multi-Component Trajectory','p95','p95_trajectory',a.trajectory_images.trajectory,'旧方案允许多个连续 components。')}${row('4. P95 · Single-Component Trajectory','p95 single','p95_single_trajectory',c.trajectory_images.trajectory,'每帧严格一个 component。')}${row('5. P95 · Single Frozen F_t','p95 single','p95_single_trajectory',c.trajectory_images.forbidden,'未膨胀的单 component removal mask。')}${row('6. P95 · Dilated Removal Mask','p95 single dilate','p95_single_d1_trajectory',e.trajectory_images.forbidden,'对 F_t 做 3×3、radius=1 空间膨胀，并减去有效轨迹 T_t。')}${row('7. Old P95 · Apply After','p95','p95_apply',a.apply_images.after,'旧 P95 回写结果。')}${row('8. Single P95 · Apply After','p95 single','p95_single_apply',c.apply_images.after,'未膨胀 P95 回写结果。')}${row('9. Dilate1 P95 · Apply After','p95 single dilate','p95_single_d1_apply',e.apply_images.after,'膨胀 P95 mask 回写结果。')}${row('10. Single P95 · Removed Mass','p95 single','p95_single_apply',c.apply_images.removed,'未膨胀实际删除量。')}${row('11. Dilate1 P95 · Removed Mass','p95 single dilate','p95_single_d1_apply',e.apply_images.removed,'膨胀后实际删除量。')}${row('12. P99 · Raw High-Response Candidates','p99','p99_trajectory',b.trajectory_images.candidate,'共同的原始 P99 候选。')}${row('13. Old P99 · Multi-Component Trajectory','p99','p99_trajectory',b.trajectory_images.trajectory,'旧方案允许多个连续 components。')}${row('14. P99 · Single-Component Trajectory','p99 single','p99_single_trajectory',dv.trajectory_images.trajectory,'每帧严格一个 component。')}${row('15. P99 · Single Frozen F_t','p99 single','p99_single_trajectory',dv.trajectory_images.forbidden,'未膨胀 removal mask。')}${row('16. P99 · Dilated Removal Mask','p99 single dilate','p99_single_d1_trajectory',f.trajectory_images.forbidden,'3×3、radius=1 膨胀并保护有效轨迹。')}${row('17. Old P99 · Apply After','p99','p99_apply',b.apply_images.after,'旧 P99 回写结果。')}${row('18. Single P99 · Apply After','p99 single','p99_single_apply',dv.apply_images.after,'未膨胀 P99 回写结果。')}${row('19. Dilate1 P99 · Apply After','p99 single dilate','p99_single_d1_apply',f.apply_images.after,'膨胀 P99 mask 回写结果。')}${row('20. Single P99 · Removed Mass','p99 single','p99_single_apply',dv.apply_images.removed,'未膨胀实际删除量。')}${row('21. Dilate1 P99 · Removed Mass','p99 single dilate','p99_single_d1_apply',f.apply_images.removed,'膨胀后实际删除量。')}${row('22a. P95 BT3 · No Intervention Top30 Mean','p95 single dilate','p95_single_bt3_d1_trajectory',g.trajectory_images.mean,'只对 LoRA PCK@32 Top30 heads 求平均，用于生成轨迹和 removal mask。')}${row('22. P95 BT3 · Forward Trajectory','p95 single dilate','p95_single_bt3_d1_trajectory',g.trajectory_images.forward_trajectory,'仅执行前向筛选、尚未进行反向修正的 Single-Component Trajectory。')}${row('23. P95 BT3 · Rejected Component','p95 single dilate','p95_single_bt3_d1_trajectory',g.trajectory_images.rejected_events,'当前帧触发分支切换的被拒绝 component，是向前追溯的起点。')}${row('24. P95 BT3 · Backward-Traced Components','p95 single dilate','p95_single_bt3_d1_trajectory',g.trajectory_images.backward_removed,'从拒绝事件向前最多追溯 3 个 latent 帧后删除的同源历史 component。')}${row('25. P95 BT3 · Corrected Trajectory','p95 single dilate','p95_single_bt3_d1_trajectory',g.trajectory_images.trajectory,'完成向前回溯清理后的 Single-Component Trajectory。')}${row('26. P95 BT3 · Corrected Dilated Removal Mask','p95 single dilate','p95_single_bt3_d1_trajectory',g.trajectory_images.forbidden,'由修正轨迹计算 F_t，再做 3×3 膨胀并保护有效轨迹。')}${row('27. P95 BT3 · Apply After','p95 single dilate','p95_single_bt3_d1_apply',g.apply_images.after,'回溯修正后的 frozen mask 实际回写 attention。')}${row('28. P95 BT3 · Removed Mass','p95 single dilate','p95_single_bt3_d1_apply',g.apply_images.removed,'回溯修正方案实际删除的 attention mass。')}${row('29a. P99 BT3 · No Intervention Top30 Mean','p99 single dilate','p99_single_bt3_d1_trajectory',h.trajectory_images.mean,'只对 LoRA PCK@32 Top30 heads 求平均，用于生成轨迹和 removal mask。')}${row('29. P99 BT3 · Forward Trajectory','p99 single dilate','p99_single_bt3_d1_trajectory',h.trajectory_images.forward_trajectory,'仅执行前向筛选、尚未进行反向修正的 Single-Component Trajectory。')}${row('30. P99 BT3 · Rejected Component','p99 single dilate','p99_single_bt3_d1_trajectory',h.trajectory_images.rejected_events,'当前帧触发分支切换的被拒绝 component。')}${row('31. P99 BT3 · Backward-Traced Components','p99 single dilate','p99_single_bt3_d1_trajectory',h.trajectory_images.backward_removed,'从拒绝事件向前最多追溯 3 个 latent 帧后删除的历史 component。')}${row('32. P99 BT3 · Corrected Trajectory','p99 single dilate','p99_single_bt3_d1_trajectory',h.trajectory_images.trajectory,'完成回溯清理后的 Single-Component Trajectory。')}${row('33. P99 BT3 · Corrected Dilated Removal Mask','p99 single dilate','p99_single_bt3_d1_trajectory',h.trajectory_images.forbidden,'修正轨迹对应的 3×3 膨胀 removal mask。')}${row('34. P99 BT3 · Apply After','p99 single dilate','p99_single_bt3_d1_apply',h.apply_images.after,'回溯修正后的 frozen mask 实际回写 attention。')}${row('35. P99 BT3 · Removed Mass','p99 single dilate','p99_single_bt3_d1_apply',h.apply_images.removed,'回溯修正方案实际删除的 attention mass。')}</article>`}).join(''):'<div class="pending">等待该 seed/step/branch 数据</div>'}function renderS09Multiseed(d){const allowed=new Set(["90094","35075","21890","49530","32466"]);if(!allowed.has(String(seed)))return;if(!heatmap.startsWith("s09_fixed_")){heatmap="s09_fixed_1";document.getElementById("heatmap").value=heatmap;sync()}const videos=document.getElementById("videos");if(videos&&!document.getElementById("s09FixedMask1Card"))videos.insertAdjacentHTML("afterbegin",`<section class="experiment-matrix"><div class="matrix-hero"><div><span class="eyebrow">多 Seed · S09 固定掩码</span><h2>S09 固定掩码核对比</h2><p>每个 seed 独立捕获 A40-S09 与 A10-S09，生成逐 head/query/latent-frame 正差 P95 mask，并固定应用于 S000-S009。</p></div><p>Wan+LoRA · seed ${seed} · Top100 heads<br>Renorm · Object A/B · Conditional/Unconditional</p></div><section class="experiment-group"><div class="group-heading"><h2>S09 Frozen Mask · Renorm</h2><p>仅空间核大小不同，其余推理配置一致</p></div><div class="experiment-grid"><article id="s09FixedMask1Card" class="video"><span class="eyebrow">掩码核 1×1</span><h2>S09-Frozen Mask</h2><h3>S000-S009 · 不扩张</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask-multiseed/video?seed=${seed}&kernel=1&v=1"></video><p>S09 正差 P95 原始 mask，置零后逐行重新归一化。</p></article><article id="s09FixedMask2Card" class="video"><span class="eyebrow">掩码核 2×2</span><h2>S09-Frozen Mask</h2><h3>S000-S009 · 2×2 扩张</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask-multiseed/video?seed=${seed}&kernel=2&v=1"></video><p>S09 正差 P95 mask 经 2×2 形态学扩张，置零后归一化。</p></article><article id="s09FixedMask3Card" class="video"><span class="eyebrow">掩码核 3×3</span><h2>S09-Frozen Mask</h2><h3>S000-S009 · 3×3 对称扩张</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask-multiseed/video?seed=${seed}&kernel=3&v=1"></video><p>S09 正差 P95 mask 经 3×3 对称扩张，置零后归一化。</p></article></div></section></section><div class="legacy-heading">基准视频与已有轨迹消融</div>`);const stepNumber=Number(step);if(stepNumber<0||stepNumber>9||(viz!=="reverse"&&viz!=="all"))return;const stepTag=String(stepNumber).padStart(2,"0");document.querySelectorAll("#content .object").forEach((el,i)=>{const region=d.records[i]?.region_name;if(!region)return;for(const kernel of [1,2,3]){const name=`step${stepTag}__${d.branch}__${region}__top100_mean_before_after_mask.jpg`;el.insertAdjacentHTML("beforeend",`<section class="row"><h3>S09-Frozen Positive-Delta · Mask ${kernel}x${kernel}</h3><p class="explain">Seed ${seed}：mask 固定来自 A40-S09 − A10-S09 的正差 P95，应用于当前 S${stepTag}；Mask 内置零后逐行重新归一化。</p><img loading="lazy" src="/api/object-query-s09-fixed-mask-multiseed/image?seed=${seed}&kernel=${kernel}&name=${encodeURIComponent(name)}&v=1"></section>`)}})}function renderNoRenormS09(d){if(String(seed)!=="47326")return;const matrix=document.querySelector(".experiment-matrix");if(matrix&&!document.getElementById("s09NoRenorm1Card"))matrix.insertAdjacentHTML("beforeend",`<section class="experiment-group"><div class="group-heading"><h2>04 · S09 固定掩码 · 不重新归一化</h2><p>Mask 内概率置零后直接计算 A_masked @ V</p></div><div class="experiment-grid"><article id="s09NoRenorm1Card" class="video"><span class="eyebrow">No-Renorm · 1×1</span><h2>S09-Frozen Mask</h2><h3>S000-S009 · 不做除法归一化</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask-no-renorm/video?kernel=1&v=1"></video><p>由 A40-S09 − A10-S09 生成 P95 mask；1×1 原始范围，置零后直接乘 V。</p></article><article id="s09NoRenorm2Card" class="video"><span class="eyebrow">No-Renorm · 2×2</span><h2>S09-Frozen Mask</h2><h3>S000-S009 · 不做除法归一化</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask-no-renorm/video?kernel=2&v=1"></video><p>S09 P95 mask 经 2×2 扩张，置零后保留剩余 attention 的原始总质量。</p></article><article id="s09NoRenorm3Card" class="video"><span class="eyebrow">No-Renorm · 3×3</span><h2>S09-Frozen Mask</h2><h3>S000-S009 · 不做除法归一化</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask-no-renorm/video?kernel=3&v=1"></video><p>S09 P95 mask 经 3×3 扩张，置零后直接使用 A_masked @ V。</p></article></div></section>`);const stepNumber=Number(step);if(stepNumber<0||stepNumber>9||(viz!=="reverse"&&viz!=="all"))return;const stepTag=String(stepNumber).padStart(2,"0");document.querySelectorAll("#content .object").forEach((el,i)=>{const region=d.records[i]?.region_name;if(!region)return;for(const kernel of [1,2,3]){const name=`step${stepTag}__${d.branch}__${region}__top100_mean_before_after_mask.jpg`;el.insertAdjacentHTML("beforeend",`<section class="row"><h3>S09-Frozen No-Renorm · Mask ${kernel}x${kernel}</h3><p class="explain">Mask 固定来自 A40-S09 − A10-S09 的逐 head/query/latent-frame 正差 P95；应用于当前 S${stepTag}，Mask 内置零后不重新归一化，直接计算 A_masked @ V。</p><img loading="lazy" src="/api/object-query-s09-fixed-mask-no-renorm/image?kernel=${kernel}&name=${encodeURIComponent(name)}&v=1"></section>`)}})}function reverseHeatmapKey(title){if(title.startsWith("Similarity-Matched · Top100 Mean")||title.startsWith("Reverse 40-Step"))return"similarity_transplant";if(title.startsWith("Sigma-Matched · Top100 Mean"))return"sigma_transplant";if(title.startsWith("Same-Index S00-S09 · Top100 Mean"))return"same_index_transplant";if(title.startsWith("Positive-Delta P95 + Mask1x1"))return"same_index_delta";if(title.startsWith("Similarity-Matched Positive-Delta"))return"similarity_delta";if(title.includes("S09-Frozen Positive-Delta")&&title.includes("Mask 1x1"))return"s09_fixed_1";if(title.includes("S09-Frozen Positive-Delta")&&title.includes("Mask 2x2"))return"s09_fixed_2";if(title.includes("S09-Frozen Positive-Delta")&&title.includes("Mask 3x3"))return"s09_fixed_3";if(title.includes("S09-Frozen No-Renorm")&&title.includes("Mask 1x1"))return"s09_no_renorm_1";if(title.includes("S09-Frozen No-Renorm")&&title.includes("Mask 2x2"))return"s09_no_renorm_2";if(title.includes("S09-Frozen No-Renorm")&&title.includes("Mask 3x3"))return"s09_no_renorm_3";return""}function filterReverseHeatmaps(){if(viz!=="reverse"&&viz!=="all")return;document.getElementById("heatmap").value=heatmap;document.querySelectorAll("#content .object").forEach(object=>{object.querySelectorAll(":scope > .heatmap-empty").forEach(node=>node.remove());let kept=0;object.querySelectorAll(":scope > .row").forEach(row=>{const title=row.querySelector("h3")?.textContent.trim()||"",key=reverseHeatmapKey(title);if(!key)return;row.dataset.heatmap=key;row.hidden=key!==heatmap;if(!row.hidden)kept+=1});if(!kept){const empty=document.createElement("div");empty.className="pending heatmap-empty";empty.innerHTML="该实验在当前 step / branch 下尚无热力图<br>请选择适用时间步，或等待产物生成后点击「手动刷新」";object.appendChild(empty)}})}function decorateExperimentNotes(){const notes={s09NoRenorm1Card:"使用 S09 正差 P95 固定 mask，1×1 不扩张；S000-S009 置零后不重新归一化。",s09NoRenorm2Card:"使用 S09 正差 P95 固定 mask，经 2×2 扩张；S000-S009 置零后直接乘 V。",s09NoRenorm3Card:"使用 S09 正差 P95 固定 mask，经 3×3 扩张；S000-S009 置零后直接乘 V。",reverseTransplantCard:"对每个 40-step 的 step、head、Object 和 CFG 分支，从 10-step 的 S00-S09 中选择 cosine 最相似的 attention，替换对应 object-query 行。",sigmaTransplantCard:"对每个 40-step step，按最小 log-sigma 距离选择 10-step donor，替换 Top100 heads 的对应 object-query attention 行。",sameIndexTransplantCard:"在 40-step 的 S000-S009，按 S00→S00 至 S09→S09 替换 10-step attention；S010-S039 保持原始 attention。",positiveDeltaMaskCard:"在 S000-S009 逐步计算 max(A40-Sx − A10-Sx, 0)，逐 head/query/latent frame 取 P95 mask；1×1 不扩张，置零后逐行归一化。",similarityDeltaMaskCard:"每个 40-step step/head 使用 cosine-matched A10 计算正差 P95 mask；1×1 不扩张，应用于 S000-S039 后逐行归一化。",s09FixedMask1Card:"只用 max(A40-S09 − A10-S09, 0) 的 P95 生成固定 mask，以 1×1 原始范围应用于 S000-S009。",s09FixedMask2Card:"只用 S09 正差 P95 生成固定 mask，经 2×2 形态学扩张后应用于 S000-S009。",s09FixedMask3Card:"只用 S09 正差 P95 生成固定 mask，经 3×3 对称空间扩张后应用于 S000-S009。"};for(const [id,note] of Object.entries(notes)){const card=document.getElementById(id);if(!card||card.querySelector(".experiment-note"))continue;card.insertAdjacentHTML("beforeend",`<p class="experiment-note"><strong>实验设置：</strong>${note}</p>`)}const legacy=[...document.querySelectorAll("#videos > article.video")],legacyNotes=["分别使用统一配置执行 Wan+LoRA 40-step 与 10-step 推理，不修改 attention。","按 P95 提取高响应候选，允许每帧保留多个空间连续 component，并据此生成 removal mask。","按 P99 提取高响应候选，允许每帧保留多个空间连续 component，并据此生成 removal mask。","按 P95 提取候选，每个 latent frame 仅保留一个空间连续 component。","按 P99 提取候选，每个 latent frame 仅保留一个空间连续 component。","对 P95 单 component removal mask 做 3×3、radius=1 空间膨胀后回写 attention。","对 P99 单 component removal mask 做 3×3、radius=1 空间膨胀后回写 attention。","使用 LoRA PCK@32 Top30 heads 的 mean attention 建轨迹；被拒绝 component 最多向前回溯三帧，再膨胀 mask。","使用 P99、Top30 mean 和最多三帧反向回溯生成单 component 轨迹及膨胀 removal mask。"] ;legacy.forEach((card,index)=>{if(card.querySelector(".experiment-note")||!legacyNotes[index])return;card.insertAdjacentHTML("beforeend",`<p class="experiment-note"><strong>实验设置：</strong>${legacyNotes[index]}</p>`)})}function decoratePendingMedia(root){root.querySelectorAll("video,img").forEach(media=>{if(media.dataset.pendingReady)return;media.dataset.pendingReady="1";const frame=document.createElement("div");frame.className="media-frame";media.parentNode.insertBefore(frame,media);frame.appendChild(media);const pending=document.createElement("div");pending.className="media-pending";pending.hidden=true;pending.innerHTML="待生成 / 待补充<br><small>产物完成后点击顶部「手动刷新」</small>";frame.appendChild(pending);const showPending=()=>{media.style.display="none";pending.hidden=false};const showMedia=()=>{media.style.display="block";pending.hidden=true};if(media.tagName==="VIDEO"){media.addEventListener("error",showPending,{once:true});media.addEventListener("loadeddata",showMedia,{once:true});if(media.error)showPending()}else{media.addEventListener("error",showPending,{once:true});media.addEventListener("load",showMedia,{once:true});if(media.complete){media.naturalWidth?showMedia():showPending()}}})}async function load(){const d=await fetch(`${api}/catalog?seed=${seed}&stage=${stage}&step=${step}&branch=${branch}`,{cache:'no-store'}).then(r=>r.json());render(d);renderAlignment(d);renderReverseAlignment(d);renderS09Multiseed(d);renderNoRenormS09(d);renderHeatmapVideos();filterReverseHeatmaps();decorateExperimentNotes();decoratePendingMedia(document)}function renderAlignment(d){const stepNumber=Number(step);if(String(seed)!="47326"||stepNumber<0||stepNumber>9||d.branch!=="conditional"||(viz!=="common"&&viz!=="all"))return;const stepTag=String(stepNumber).padStart(2,"0");document.querySelectorAll("#content .object").forEach((el,i)=>{const region=d.records[i]?.region_name;if(!region)return;const name=`seed047326__step${stepTag}__conditional__${region}__best_head_matches.jpg`;el.insertAdjacentHTML("beforeend",row("1c. Common Highest-Similarity Head Pairs","","p95_trajectory",name,"Wan+LoRA sample-specific cosine Top-5 per object. Rows show 10-step S09, each Head's best matching 40-step Sxx, and absolute delta. The matched Sxx is labeled above every strip; 10/40 share a per-latent-frame color scale."))})}function renderReverseAlignment(d){if(String(seed)==="47326"){const videos=document.getElementById("videos");if(videos&&!document.getElementById("reverseTransplantCard"))videos.insertAdjacentHTML("afterbegin",`<section class="experiment-matrix"><div class="matrix-hero"><div><span class="eyebrow">跨步注意力实验</span><h2>Object Query 注意力干预实验</h2><p>页面按 Attention 替换、正差区域移除和 S09 固定掩码三类设置排列。未完成结果保留固定卡位，点击“手动刷新”后加载新产物。</p></div><p>Wan+LoRA · seed 47326 · Top100 heads<br>Q/K/V unchanged · post-softmax intervention</p></div><section class="experiment-group"><div class="group-heading"><h2>01 · 跨步 Attention 替换</h2><p>用 10-step 注意力替换 40-step 的 object-query 行</p></div><div class="experiment-grid"><article id="reverseTransplantCard" class="video single dilate"><span class="eyebrow">余弦相似匹配</span><h2>Similarity-Matched A10 Transplant</h2><h3>40-step · Top100 Object Query Rows</h3><video controls preload="metadata" playsinline src="/api/object-query-reverse-transplant/video?v=2"></video><p>Cosine-similarity matching · Q/K/V unchanged · selected post-softmax object-query rows only · output=A10@V40</p></article><article id="sigmaTransplantCard" class="video single dilate"><span class="eyebrow">噪声强度对齐</span><h2>Sigma-Matched A10 Transplant</h2><h3>40-step · Top100 Object Query Rows</h3><video controls preload="metadata" playsinline src="/api/object-query-sigma-transplant/video?v=1"></video><p>Nearest log-sigma alignment · Q/K/V unchanged · selected post-softmax object-query rows only · output=A10@V40</p></article><article id="sameIndexTransplantCard" class="video single dilate"><span class="eyebrow">同编号替换</span><h2>Same-Index S00-S09 A10 Transplant</h2><h3>40-step · Top100 Object Query Rows</h3><video controls preload="metadata" playsinline src="/api/object-query-same-index-transplant/video?v=1"></video><p>S00→S00 through S09→S09 · S10-S39 unchanged · Q/K/V unchanged · output=A10@V40 only in the first ten steps</p></article></div></section><section class="experiment-group"><div class="group-heading"><h2>02 · 正差 Attention 区域移除</h2><p>P95 掩码 · 区域置零 · 逐行重新归一化</p></div><div class="experiment-grid"><article id="positiveDeltaMaskCard" class="video single dilate"><span class="eyebrow">同编号 · 1×1</span><h2>Same-Index Positive-Delta P95 Mask 1x1</h2><h3>40-step · S000-S009 · Top100 Object Query Rows</h3><video controls preload="metadata" playsinline src="/api/object-query-positive-delta-mask/video?v=1"></video><p>Frozen mask from same-index max(A40-A10,0) · per-query/per-latent-frame P95 · 1x1 mask (no expansion) · renormalized</p></article><article id="similarityDeltaMaskCard" class="video single dilate"><span class="eyebrow">相似匹配 · 1×1</span><h2>Similarity-Matched Positive-Delta P95 Mask 1x1</h2><h3>40-step · S000-S039 · Top100 Object Query Rows</h3><video controls preload="metadata" playsinline src="/api/object-query-similarity-delta-mask/video?v=1"></video><p>Each head uses its cosine-matched A10 step · frozen max(A40-A10,0) P95 mask · 1x1 mask (no expansion) · masked A40 set to zero and renormalized</p></article><article class="video"><span class="eyebrow">预留位置</span><h2>待补充对照</h2><h3>Reserved comparison slot</h3><div class="media-pending">待补充实验结果<br>位置已保留</div><p>待补充具体阈值、分支、head 范围和生成结果。</p></article></div></section><section class="experiment-group"><div class="group-heading"><h2>03 · S09 固定掩码核对比</h2><p>由 S09 差异生成一次掩码，并固定应用于 S000-S009</p></div><div class="experiment-grid"><article id="s09FixedMask1Card" class="video single dilate"><span class="eyebrow">掩码核 1×1</span><h2>S09-Frozen Mask 1x1</h2><h3>A40-S09 minus A10-S09 · Apply S000-S009</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask/video?kernel=1&v=1"></video><p>Per-head frozen P95 mask · no expansion · zero and renormalize</p></article><article id="s09FixedMask2Card" class="video single dilate"><span class="eyebrow">掩码核 2×2</span><h2>S09-Frozen Mask 2x2</h2><h3>A40-S09 minus A10-S09 · Apply S000-S009</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask/video?kernel=2&v=1"></video><p>Per-head frozen P95 mask · 2x2 morphological expansion · zero and renormalize</p></article><article id="s09FixedMask3Card" class="video single dilate"><span class="eyebrow">掩码核 3×3</span><h2>S09-Frozen Mask 3x3</h2><h3>A40-S09 minus A10-S09 · Apply S000-S009</h3><video controls preload="metadata" playsinline src="/api/object-query-s09-fixed-mask/video?kernel=3&v=1"></video><p>Per-head frozen P95 mask · symmetric one-token expansion · zero and renormalize</p></article></div></section></section><div class="legacy-heading">基准视频与已有轨迹消融</div>`)}const stepNumber=Number(step);if(String(seed)!="47326"||stepNumber<0||stepNumber>39||(viz!=="reverse"&&viz!=="all"))return;const stepTag=String(stepNumber).padStart(2,"0");document.querySelectorAll("#content .object").forEach((el,i)=>{const region=d.records[i]?.region_name;if(!region)return;const meanName=`step${stepTag}__${d.branch}__${region}__top100_mean_before_after.jpg`;el.insertAdjacentHTML("beforeend",`<section class="row"><h3>Similarity-Matched · Top100 Mean Before / After / |Delta|</h3><p class="explain">Before is live A40. After is cosine-matched post-softmax A10 multiplied by V40. Before/After share one color scale per latent frame.</p><img loading="lazy" src="/api/object-query-reverse-transplant/image?name=${encodeURIComponent(meanName)}"></section><section class="row"><h3>Sigma-Matched · Top100 Mean Before / After / |Delta|</h3><p class="explain">Nearest log-sigma matching, independent of attention similarity. The image header labels target S40/sigma40, donor S10/sigma10, and log-sigma distance.</p><img loading="lazy" src="/api/object-query-sigma-transplant/image?name=${encodeURIComponent(meanName)}&v=1"></section>`);if(stepNumber<10)el.insertAdjacentHTML("beforeend",`<section class="row"><h3>Same-Index S00-S09 · Top100 Mean Before / After / |Delta|</h3><p class="explain">Direct S40 index to S10 index mapping: S00→S00 through S09→S09. The header labels target step/sigma and donor step/sigma. S10-S39 are not intervened.</p><img loading="lazy" src="/api/object-query-same-index-transplant/image?name=${encodeURIComponent(meanName)}&v=1"></section>`);if(stepNumber<10){const deltaName=`step${stepTag}__${d.branch}__${region}__top100_mean_before_after_mask.jpg`;el.insertAdjacentHTML("beforeend",`<section class="row"><h3>Positive-Delta P95 + Mask1x1 · Before / Mask / After</h3><p class="explain">Frozen no-intervention A40 minus same-index A10; positive P95 is computed separately for every head, object-query token, and latent key frame. The original P95 mask is applied with a 1x1 kernel (no spatial expansion), followed by row renormalization.</p><img loading="lazy" src="/api/object-query-positive-delta-mask/image?name=${encodeURIComponent(deltaName)}&v=1"></section>`)}const similarityDeltaName=`step${stepTag}__${d.branch}__${region}__top100_mean_before_after_mask.jpg`;el.insertAdjacentHTML("beforeend",`<section class="row"><h3>Similarity-Matched Positive-Delta P95 + Mask1x1</h3><p class="explain">Every target step/head/object/CFG branch uses its own cosine-matched 10-step donor. Frozen positive delta P95 masks use a 1x1 kernel with no expansion and are applied to live A40 across S000-S039; the image header reports matched-step distribution and sigma range.</p><img loading="lazy" src="/api/object-query-similarity-delta-mask/image?name=${encodeURIComponent(similarityDeltaName)}&v=1"></section>`);if(stepNumber<10){for(const kernel of [1,2,3]){const fixedName=`step${stepTag}__${d.branch}__${region}__top100_mean_before_after_mask.jpg`;el.insertAdjacentHTML("beforeend",`<section class="row"><h3>S09-Frozen Positive-Delta P95 · Mask ${kernel}x${kernel}</h3><p class="explain">Each head/object-query uses one frozen mask from no-intervention A40-S09 minus A10-S09. The same mask is applied at live A40 S${stepTag}; masked probabilities are zeroed and row-renormalized. S010-S039 remain untouched.</p><img loading="lazy" src="/api/object-query-s09-fixed-mask/image?kernel=${kernel}&name=${encodeURIComponent(fixedName)}&v=1"></section>`)}}if(d.branch==="conditional"){const name=`seed047326__step${stepTag}__conditional__${region}__reverse_best_head_matches.jpg`;el.insertAdjacentHTML("beforeend",row("Reverse 40-Step → Best 10-Step Head Pairs","","p95_trajectory",name,"For every physical Head at the selected 40-step Sxx, search all 10-step S00-S09 and retain its nearest step. The displayed Top-5 are ranked by sample-specific cosine for this seed, branch, and object."))}})}for(const id of ['seed','stage','step','branch','viz','heatmap'])document.getElementById(id).addEventListener('change',ev=>{if(id==='seed')seed=ev.target.value;if(id==='stage')stage=ev.target.value;if(id==='step')step=ev.target.value;if(id==='branch')branch=ev.target.value;if(id==='viz')viz=ev.target.value;if(id==='heatmap')heatmap=ev.target.value;sync();if(id==='heatmap'){renderHeatmapVideos();filterReverseHeatmaps();return}load()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=0;v.loop=false;v.play().catch(()=>{})}));load();
</script><script>
function appendDynamicObjectQueryRows(){
  if(viz!=="common"&&viz!=="all")return;
  document.querySelectorAll("#content .object").forEach(object=>{
    if(object.querySelector(".dynamic-object-query-row"))return;
    const anchor=[...object.querySelectorAll(":scope > .row")].find(section=>section.querySelector("h3")?.textContent.startsWith("1b."));
    if(!anchor)return;
    const region=object.querySelector(":scope > h2")?.textContent.split("·")[0].trim();
    if(!region)return;
    const seedTag=String(seed).padStart(6,"0"),stepTag=String(step).padStart(2,"0");
    const track40=`seed${seedTag}__steps40__${region}__query_tokens.jpg`;
    const track10=`seed${seedTag}__steps10__${region}__query_tokens.jpg`;
    const dynamic=`seed${seedTag}__step${stepTag}__${branch}__${region}__dynamic_same_frame_mean.jpg`;
    const section=(title,kind,name,description)=>`<section class="row dynamic-object-query-row"><h3>${title}</h3><p class="explain">${description}</p><img loading="lazy" src="${image(kind,name)}"></section>`;
    let html=section("1c. 40-Step CoTracker · Dynamic Query Token Positions","dynamic40_track",track40,"圆点是 CoTracker 在 F00/F04/.../F48 的实际像素位置；方框和十字是映射后的 16×28 latent query token，Q(y,x) 为 token 坐标。")+section("1d. 40-Step Dynamic Object Query · Top100 Mean","dynamic40_mean",dynamic,"每个 latent 帧使用该帧实际 object tokens 作为 Q_t，仅展示同帧 K_t；softmax 仍在全部 5824 keys 上计算，Top100 heads 求平均，每帧独立色标。")+section("1e. 10-Step CoTracker · Dynamic Query Token Positions","dynamic10_track",track10,"单独追踪 10-step Original 视频，因此其动态 query 轨迹不复用 40-step 位置。圆点为像素轨迹，方框/十字为实际 latent query token。 ");
    if(Number(step)<10)html+=section("1f. 10-Step Dynamic Object Query · Top100 Mean","dynamic10_mean",dynamic,"10-step 无干预重捕获；Q_t 来自对应 10-step 视频的 CoTracker 轨迹，仅展示 Q_t→K_t，且每个 latent 帧单独归一化。 ");
    anchor.insertAdjacentHTML("afterend",html);
    decoratePendingMedia(object);
  });
}
const dynamicObjectQueryObserver=new MutationObserver(()=>appendDynamicObjectQueryRows());
dynamicObjectQueryObserver.observe(document.getElementById("content"),{childList:true});
appendDynamicObjectQueryRows();
</script></body></html>'''


OBJECT_QUERY_OVERLAY_PROFILES = tuple(
    item for item in SEED_SWEEP_PROFILES if item[0] != "head_output_zero"
) + (
    ("exclude_current", "Exclude Current Frame"),
    ("context_only", "Context Frames Only"),
)


def object_query_attention_overlay_asset(stage: str, profile: str, name: str):
    if (
        stage not in {"all_steps", "steps00_09"}
        or profile not in {item[0] for item in OBJECT_QUERY_OVERLAY_PROFILES}
        or Path(name).name != name
        or not name.endswith(".jpg")
    ):
        return None
    return OBJECT_QUERY_OVERLAY_PILOT_ROOT / stage / profile / "overlays" / name


def object_query_attention_overlay_catalog(
    stage: str, profile: str, group: str
):
    allowed_profiles = {item[0] for item in OBJECT_QUERY_OVERLAY_PROFILES}
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
    records.sort(
        key=lambda item: (
            int(item.get("block", 0)),
            int(item.get("head", 0)),
            str(item.get("region_name", "")),
        )
    )
    return {
        "case": ATTENTION_LORA_CASE,
        "seed": 90094,
        "stage": stage,
        "profile": profile,
        "group": group,
        "profiles": [
            {"id": profile_id, "label": label}
            for profile_id, label in OBJECT_QUERY_OVERLAY_PROFILES
        ],
        "records": records,
    }


def object_query_attention_overlay_video_asset(
    stage: str, profile: str, group: str, kind: str
):
    allowed_profiles = {item[0] for item in OBJECT_QUERY_OVERLAY_PROFILES}
    if stage != "all_steps" or profile not in allowed_profiles:
        return None
    if group not in {"top100", "bottom100"}:
        return None
    case_root = (
        OBJECT_QUERY_OVERLAY_PILOT_ROOT
        / stage
        / profile
        / "videos"
        / "lora"
        / "cases"
        / "0613pybullet_sample_001460_w002"
    )
    if kind == "baseline":
        return case_root / "original.mp4"
    if kind == "intervention":
        return case_root / f"{group}_steps_00_40.mp4"
    return None


def object_query_attention_overlay_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object Query Attention Overlay</title><style>
:root{--paper:#eee8dc;--ink:#17251f;--line:#bcb19d;--card:#fffdf8;--red:#ad422e;--green:#17685c}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 5% 0,#e99e5555,transparent 32rem),radial-gradient(circle at 98% 4%,#4b937655,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:10;padding:17px 24px;background:#eee8dced;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(27px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;flex-wrap:wrap;align-items:center}select,button{padding:9px 12px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#58675f}main{width:min(2280px,calc(100% - 18px));margin:auto;padding:20px 0 70px}.video-panel,.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:15px}.video-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.video-grid figure{margin:0}.video-grid figcaption{font-weight:900;margin:0 0 6px}.video-panel video{display:block;width:100%;background:#131714}.panel.top{border-left:7px solid var(--red)}.panel.bottom{border-left:7px solid var(--green)}.panel h2{margin:0 0 8px}.meta{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:9px}.pill{padding:5px 8px;background:#e8e1d3;border-radius:99px;font:11px ui-monospace,monospace}.images{display:grid;gap:14px}.images figure{margin:0}.images img{display:block;width:100%;min-width:1900px;border:1px solid var(--line);background:#111}.images figcaption{font-weight:900;margin:4px 0}.compare-stack{display:grid;gap:8px;padding-bottom:5px}.compare-stack figure{min-width:1900px}.compare-stack figcaption{position:sticky;left:0;width:max-content;padding:3px 9px;background:var(--ink);color:#fff;border-radius:4px;z-index:1}.delta-row{border-top:1px dashed var(--line);padding-top:8px}.scroll{overflow:auto}.pending{padding:50px;border:1px dashed var(--line);background:var(--card)}@media(max-width:800px){header{position:static}.video-grid{grid-template-columns:1fr}}
</style></head><body><header><a href="/attention-additive-lora-seed-sweep?v=1&seed=90094">返回 Seed Sweep</a><h1>Head-wise PCK Object Query Overlay</h1><p>Seed 90094 · SAM2 context F04 · Q latent 1 · Object A sphere / Object B box · 每个对象的 8 个 query 热力图直接求和叠加 × 13 key latent frames</p><div class="tools"><label>Stage <select id="stage"><option value="all_steps">S000-S039</option></select></label><label>Experiment <select id="profile"></select></label><label>Group <select id="group"><option value="top100">Top10 Heads</option><option value="bottom100">Bottom10 Heads</option></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="video" class="video-panel"></section><section id="records"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),params=new URL(location.href).searchParams;let stage=params.get('stage')||'all_steps',profile=params.get('profile')||'alpha090',group=params.get('group')||'top100';const image=(name)=>`/api/object-query-attention-overlay/image?stage=${stage}&profile=${profile}&name=${encodeURIComponent(name)}`,video=(kind)=>`/api/object-query-attention-overlay/video?stage=${stage}&profile=${profile}&group=${group}&kind=${kind}`;
function syncUrl(){const u=new URL(location.href);u.searchParams.set('stage',stage);u.searchParams.set('profile',profile);u.searchParams.set('group',group);history.replaceState(null,'',u)}function render(d){const profileSelect=document.getElementById('profile');if(profileSelect.options.length!==d.profiles.length)profileSelect.innerHTML=d.profiles.map(x=>`<option value="${e(x.id)}">${e(x.label)}</option>`).join('');profileSelect.value=profile;document.getElementById('stage').value=stage;document.getElementById('group').value=group;document.getElementById('status').textContent=`${d.records.filter(x=>x.ready).length}/${d.records.length||20} object-head records ready`;document.getElementById('video').innerHTML=`<h2>对应生成视频</h2><div class="video-grid"><figure><figcaption>Wan+LoRA Baseline · No Attention Intervention</figcaption><video controls preload="metadata" playsinline src="${video('baseline')}"></video></figure><figure><figcaption>${group==='top100'?'Top10':'Bottom10'} · ${e(profile)} Attention Intervention</figcaption><video controls preload="metadata" playsinline src="${video('intervention')}"></video></figure></div>`;document.getElementById('records').innerHTML=d.records.length?d.records.map(r=>`<article class="panel ${group.startsWith('top')?'top':'bottom'}"><h2>L${String(r.block).padStart(2,'0')} / H${String(r.head).padStart(2,'0')} · ${e(r.region_name)} (${e(r.region_phrase)})</h2><div class="meta"><span class="pill">LoRA PCK@32 ${Number(r.pck32).toFixed(3)}</span><span class="pill">Q=F${String(r.query_pixel_frame).padStart(2,'0')} / latent ${r.query_latent_frame}</span><span class="pill">Σ ${r.query_count} SAM2 queries</span><span class="pill">S${String(r.step).padStart(3,'0')}</span><span class="pill">${group==='top100'?'Top10':'Bottom10'}</span></div>${r.ready?`<div class="images"><figure><figcaption>8 Query Attention 求和叠加 · Before / After 相邻</figcaption><div class="scroll"><img loading="lazy" src="${image(r.images.before_after)}"></div></figure><figure><figcaption>该 Head 全 token Q@K · Before / After / |Delta|</figcaption><div class="scroll"><img loading="lazy" src="${image(r.images.all_token_qk)}"></div></figure><figure class="delta-row"><figcaption>Σ |Delta|</figcaption><div class="scroll"><img loading="lazy" src="${image(r.images.abs_delta)}"></div></figure></div>`:'<div class="pending">Head-wise PCK capture / overlay 生成中</div>'}</article>`).join(''):'<div class="pending">该组合正在捕获，点击手动刷新查看。</div>'}
async function load(){const d=await fetch(`/api/object-query-attention-overlay/catalog?stage=${stage}&profile=${profile}&group=${group}`,{cache:'no-store'}).then(r=>r.json());render(d)}for(const id of ['stage','profile','group'])document.getElementById(id).addEventListener('change',ev=>{if(id==='stage')stage=ev.target.value;if(id==='profile')profile=ev.target.value;if(id==='group')group=ev.target.value;syncUrl();load()});document.getElementById('refresh').addEventListener('click',load);load();
</script></body></html>'''


SEED_SWEEP_METRICS = (
    ("physics_iq_with_context", "PhysicsIQ +Ctx", ("physics_iq_with_context", "score"), "max"),
    ("physics_iq_without_context", "PhysicsIQ -Ctx", ("physics_iq_without_context", "score"), "max"),
    ("pmf_with_context", "PMF +Ctx", ("pmf_with_context", "score"), "max"),
    ("pmf_without_context", "PMF -Ctx", ("pmf_without_context", "score"), "max"),
    ("wmreward", "WMReward Surprise", ("wmreward", "surprise"), "min"),
    ("vbench_subject_consistency", "Subject Consistency", ("vbench_subject_consistency", "score"), "max"),
    ("vbench_background_consistency", "Background Consistency", ("vbench_background_consistency", "score"), "max"),
    ("vbench_temporal_flickering", "Temporal Flickering", ("vbench_temporal_flickering", "score"), "max"),
    ("vbench_motion_smoothness", "Motion Smoothness", ("vbench_motion_smoothness", "score"), "max"),
    ("vbench_dynamic_degree", "Dynamic Degree", ("vbench_dynamic_degree", "score"), "max"),
    ("vbench_aesthetic_quality", "Aesthetic Quality", ("vbench_aesthetic_quality", "score"), "max"),
    ("vbench_imaging_quality", "Imaging Quality", ("vbench_imaging_quality", "score"), "max"),
    ("videophy2", "VideoPhy2", ("videophy2", "score"), "max"),
    ("cosmos_reason1", "Cosmos Reason1", ("cosmos_reason1", "score"), "max"),
)


def _seed_metric_method_specs():
    yield "original", "Original", "original", "original", "original"
    profile_labels = dict(SEED_SWEEP_PROFILES)
    for stage in ("all_steps", "steps00_09"):
        stage_label = "S000-S039" if stage == "all_steps" else "S000-S009"
        for profile, profile_label in SEED_SWEEP_PROFILES:
            for group in ("top100", "bottom100"):
                method = f"{stage}__{profile}__{group}"
                label = f"{stage_label} · {profile_label} · {group.upper()}"
                yield method, label, stage, profile, group


def _nested_metric(payload, keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def attention_lora_seed_sweep_metrics_summary():
    method_specs = list(_seed_metric_method_specs())
    values = {}
    source_counts = {}
    for method, _label, _stage, _profile, _group in method_specs:
        method_root = ATTENTION_LORA_SEED_METRICS_ROOT / "methods" / method
        method_values = {metric[0]: {} for metric in SEED_SWEEP_METRICS}
        source_count = 0
        if method_root.is_dir():
            for path in sorted(method_root.glob("seed_*.json")):
                payload = load_payload(path)
                if not isinstance(payload, dict):
                    continue
                try:
                    seed = int(payload.get("seed"))
                except (TypeError, ValueError):
                    continue
                source_count += 1
                for metric_id, _metric_label, keys, _direction in SEED_SWEEP_METRICS:
                    metric_value = _nested_metric(payload, keys)
                    if metric_value is not None:
                        method_values[metric_id][seed] = metric_value
        values[method] = method_values
        source_counts[method] = source_count
    seed_sets = [
        set(values[method][metric_id])
        for method, *_ in method_specs
        for metric_id, _label, _keys, _direction in SEED_SWEEP_METRICS
    ]
    fully_complete_seeds = set.intersection(*seed_sets) if seed_sets else set()
    common_seeds = {
        metric_id: fully_complete_seeds
        for metric_id, _label, _keys, _direction in SEED_SWEEP_METRICS
    }
    rows = []
    for method, label, stage, profile, group in method_specs:
        cells = {}
        for metric_id, _metric_label, _keys, _direction in SEED_SWEEP_METRICS:
            available = list(values[method][metric_id].values())
            shared = [
                values[method][metric_id][seed]
                for seed in sorted(common_seeds[metric_id])
            ]
            available_mean = sum(available) / len(available) if available else None
            available_std = (
                math.sqrt(
                    sum((value - available_mean) ** 2 for value in available)
                    / len(available)
                )
                if available
                else None
            )
            cells[metric_id] = {
                "available_mean": available_mean,
                "available_std": available_std,
                "available_n": len(available),
                "common_mean": sum(shared) / len(shared) if shared else None,
                "common_n": len(shared),
                "best": False,
            }
        rows.append(
            {
                "method": method,
                "label": label,
                "stage": stage,
                "profile": profile,
                "group": group,
                "source_n": source_counts[method],
                "metrics": cells,
            }
        )
    for metric_id, _label, _keys, direction in SEED_SWEEP_METRICS:
        candidates = [
            row["metrics"][metric_id]["common_mean"]
            for row in rows
            if row["metrics"][metric_id]["common_mean"] is not None
        ]
        if not candidates:
            continue
        target = min(candidates) if direction == "min" else max(candidates)
        for row in rows:
            value = row["metrics"][metric_id]["common_mean"]
            row["metrics"][metric_id]["best"] = (
                value is not None and math.isclose(value, target, rel_tol=1e-10, abs_tol=1e-12)
            )
    prepared = load_payload(ATTENTION_LORA_SEED_METRICS_ROOT / "prepared_status.json") or {}
    return {
        "case": ATTENTION_LORA_CASE,
        "expected_seeds": 50,
        "metrics": [
            {"id": metric_id, "label": label, "direction": direction}
            for metric_id, label, _keys, direction in SEED_SWEEP_METRICS
        ],
        "rows": rows,
        "prepared": prepared,
        "common_seed_counts": {
            metric_id: len(seeds) for metric_id, seeds in common_seeds.items()
        },
    }


def attention_lora_seed_sweep_metrics_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>50-Seed 全指标均值</title><style>
:root{--paper:#eee8dc;--ink:#18251f;--line:#c4b9a5;--card:#fffdf8;--green:#17685a;--gold:#d9a441;--rust:#ad432f}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 3% 0,#e99b5550,transparent 34rem),radial-gradient(circle at 98% 3%,#4d947752,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:17px 24px;background:#eee8dcef;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:4px 0;font-size:clamp(28px,4vw,49px)}header p{margin:5px 0}.status{font:12px ui-monospace,monospace}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}button{padding:8px 13px;border:1px solid var(--line);background:#fff;font-weight:900}main{width:min(100% - 18px,2900px);margin:auto;padding:18px 0 70px}.scroll{overflow:auto;border:1px solid var(--line);border-radius:15px;background:var(--card)}table{border-collapse:separate;border-spacing:0;min-width:5200px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-right:1px solid #d6cebf;border-bottom:1px solid #d6cebf;text-align:center}thead th{position:sticky;top:0;z-index:4;background:#1b3a30;color:#fff}th:first-child,td:first-child{position:sticky;left:0;z-index:3;text-align:left;min-width:285px}thead th:first-child{z-index:6}tbody td:first-child{background:#faf6ed}.method{font-weight:900}.method small{display:block;color:#6e6a61;margin-top:4px}.value{font-size:15px;font-weight:900}.available,.count{display:block;font:10px ui-monospace,monospace;color:#6c6b64;margin-top:3px}.best{background:#e1f0e6}.best .value{color:var(--green)}.badge{display:inline-block;margin-left:4px;padding:2px 5px;border-radius:99px;background:var(--green);color:#fff;font-size:8px}.pending{background:#eee8dc;color:#817969}.legend{margin:10px 0;font-size:12px;color:#625f57}.pill{display:inline-block;padding:6px 9px;border-radius:99px;background:#fff;border:1px solid var(--line);margin:3px}@media(max-width:800px){header{position:static}}
</style></head><body><header><a href="/attention-additive-lora-seed-sweep?v=1&seed=35075">返回 Seed Sweep</a><h1>50-Seed · 全部14项指标</h1><p>仅统计所有 33 个方法共同完成的 seed；BEST 也只依据该共同 seed 子集。</p><div class="tools"><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><div id="summary"></div><p class="legend">单元格仅展示 Common-Seed Mean；WMReward Surprise 越低越好，其余指标越高越好。</p><div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt=v=>v==null?'—':Number(v).toFixed(4);function render(d){const counts=Object.values(d.common_seed_counts||{}),commonN=counts.length?Math.min(...counts):0;document.getElementById('status').textContent=`${commonN}/50 common seeds evaluated`;document.getElementById('summary').innerHTML=`<span class="pill">${d.rows.length} methods</span>`+d.metrics.map(m=>`<span class="pill">${e(m.label)} common n=${d.common_seed_counts[m.id]||0}</span>`).join('');document.getElementById('head').innerHTML='<tr><th>Method</th>'+d.metrics.map(m=>`<th>${e(m.label)}<br><small>${m.direction==='min'?'↓ lower':'↑ higher'}</small></th>`).join('')+'</tr>';document.getElementById('body').innerHTML=d.rows.map(r=>`<tr><td><span class="method">${e(r.label)}<small>${e(r.method)}</small></span></td>`+d.metrics.map(m=>{const c=r.metrics[m.id],pending=c.common_mean==null;return `<td class="${pending?'pending':c.best?'best':''}"><span class="value">${fmt(c.common_mean)}${c.best?`<span class="badge">BEST ${m.direction==='min'?'↓':'↑'}</span>`:''}</span><span class="count">common n=${c.common_n}</span></td>`}).join('')+'</tr>').join('')}async function load(){const d=await fetch('/api/attention-additive-lora-seed-sweep-metrics/summary',{cache:'no-store'}).then(r=>r.json());render(d)}document.getElementById('refresh').addEventListener('click',load);load();setInterval(load,30000);
</script></body></html>'''


if __name__ == "__main__":
    viewer.main()
