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
<a class="card new" href="/object-query-ablation-metrics?v=4"><div><span>35 / OBJECT QUERY METRICS</span><h2>Fixed × Tube 消融量化诊断</h2><p>001460 / seed 47326 的 49 个 Top100 视频；四张实验指标表逐列标注定义与数值方向，并展示全部审计量。</p></div><span class="go">打开消融指标页</span></a>
<a class="card new" href="/object-query-m1-temporal-gallery?v=1"><div><span>36 / M1 TEMPORAL GALLERY</span><h2>M1 Same / Future / Past</h2><p>只展示 001460 / seed 47326；每条时间方向行汇总 Object A、Object B、all_objects 与 Top100、Bottom100、All720。</p></div><span class="go">打开 M1 时间消融页</span></a>
<a class="card new" href="/object-query-all720-ablation-gallery?v=1"><div><span>37 / ALL720 ABLATION GALLERY</span><h2>All720 全部时序消融</h2><p>汇总现有全部 All720 视频：M1/M2/M3 × All-time/Same/Future/Past × Object A/Object B/all_objects。</p></div><span class="go">打开 All720 消融页</span></a>
<a class="card new" href="/object-query-m123-temporal-batch?v=1"><div><span>38 / M1–M3 TEMPORAL BATCH</span><h2>M1/M2/M3 · All-time/Same/Future/Past</h2><p>当前多 case、多 seed 的 Top100/Bottom100 批次独立入口；按 case、seed、target 查看已生成视频。</p></div><span class="go">打开 M1–M3 批次</span></a>
<a class="card new" href="/object-query-m123-temporal-batch?single=1&amp;case=0613pybullet_sample_001460_w002&amp;seed=47326&amp;target=single_object%3A%3Aobject_A&amp;v=4"><div><span>39 / 001460 · SEED 47326 METRICS</span><h2>M1/M2/M3 · 完整指标轻量页</h2><p>0613pybullet_sample_001460_w002 / seed 47326；All-time/Same/Future/Past × Top100/Bottom100/All720，顶部含指标定义表，每个视频下方可展开完整指标。</p></div><span class="go">打开 Seed 47326 完整指标页</span></a>
<a class="card new" href="/object-query-m123-s039-top100-mean-overlays?v=5"><div><span>40 / S039 KEY + QUERY · 108</span><h2>M1/M2/M3 Head-Scope Overlay</h2><p>001460 / seed 47326 的独立 Overlay 入口；同一 M/Time 下 Object A/B 成对排列，同时展示固定 Query 的 Key-side 三行图，以及完整 Query 空间的 S(q)/E(q) receiver 图。</p></div><span class="go">打开独立 Overlay 页面</span></a>
<a class="card new" href="/object-query-head-scope-comparison?v=1"><div><span>41 / HEAD-SCOPE STRICT PAIRS</span><h2>Top100 / Bottom100 / All720 严格对比</h2><p>按 case×seed×Object×M/时间严格配对三种 Head scope；逐指标给出影响排序，并用三列代表视频对比生成、轨迹与对象存活结果。</p></div><span class="go">打开 Head-Scope 对比</span></a>
'''
INFORMATION_FLOW_VALIDATION_PORTAL_CARD = r'''
<a class="card new" href="/object-query-information-flow-validation?v=1"><div><span>42 / LATEST3350 VALIDATION</span><h2>Object Query 信息流验证</h2><p>执行计划 Stage 1–3 的实时入口：Query-time 排名稳定性、实现审计，以及 10 cases × 3 seeds 的 M1/M2/M3 × Top/Bottom/Random/All720 生成与 dose。</p></div><span class="go">打开信息流验证页</span></a>
'''
STAGE4_TEMPORAL_PORTAL_CARD = r'''
<a class="card new" href="/object-query-information-flow-stage4?v=1"><div><span>45 / STAGE 4 TEMPORAL</span><h2>信息流时间方向验证</h2><p>3 cases × 3 seeds；按 target 展示 M1/M2/M3 × All-time/Same/Future/Past，并排比较 latest3350 Top100、Bottom100、Random100 与 All720。</p></div><span class="go">打开 Stage 4 子页面</span></a>
'''
STAGE4_REPRESENTATIVES_PORTAL_CARD = r'''
<a class="card new" href="/object-query-information-flow-stage4-representatives?v=2"><div><span>46 / STAGE 4 CONTROLLED VIDEOS</span><h2>控制变量代表视频</h2><p>固定 case、seed、object 与 Baseline，分别只改变 Head group、Same/Future/Past 或 M1/M2/M3；展示 6 组支持性案例与反例，并附轨迹、身份、存活和背景指标。</p></div><span class="go">打开控制变量视频页</span></a>
'''
TOP100_M1_GUIDANCE_PORTAL_CARD = r'''
<a class="card new" href="/top100-m1-guidance-pilot?v=3"><div><span>47 / TRAINING-FREE M1/M2/M3 GUIDANCE</span><h2>Baseline × Top100 M1/M2/M3</h2><p>3 cases × seeds {47326,42} × λ {0.5,1}；固定 CFG=5、40 步和 Top100，只替换 guidance 强度与 R→R、C→R、R→C 信息流。</p></div><span class="go">打开受控 Guidance 网格</span></a>
'''
TOP100_M1_TOKEN_COMMUNICATION_PORTAL_CARD = r'''
<a class="card new" href="/top100-m1-token-communication?v=1"><div><span>48 / M1 TOKEN COMMUNICATION</span><h2>13×13 R→R Token 通信位置</h2><p>逐 Query 时刻展示实际 22×40 latent 网格单元；橙色为 Query，青色为全部 13 帧被删除的 K/V，并提供 169 条通信矩阵。</p></div><span class="go">打开 Token 通信页面</span></a>
'''
TRAINING_FREE_M1_CONTROL_PORTAL_CARD = r'''
<a class="card new" href="/training-free-m1-control?v=1"><div><span>51 / TRAINING-FREE M1 CONTROL</span><h2>M1 Soft Scaling × Contrast Guidance</h2><p>独立展示 3 cases × seeds {47326,42}：α∈{−1,−0.5,0,+0.5,+1} 直接缩放 R→R contribution，并与同刻度 conditional contrast guidance 严格对比。</p></div><span class="go">打开 M1 双向控制台</span></a>
'''
TRAINING_FREE_M1_PHASE_BD_PORTAL_CARD = r'''
<a class="card new" href="/training-free-m1-phase-bd?v=2"><div><span>56 / M1 DIRECT ENHANCEMENT</span><h2>Phase B/D · Sparse vs SAM2 Full-mask</h2><p>20 cases × 5 seeds；每行并排展示同 seed Baseline、原 8 点稀疏 tube 与逐帧 SAM2 完整 object_A token 的 Phase B/D 结果。</p></div><span class="go">打开 Full-mask 对照实时页</span></a>
'''
TRAINING_FREE_M1_MULTI_OBJECT_PORTAL_CARD = r'''
<a class="card new" href="/training-free-m1-multi-object-search?v=3"><div><span>55 / TOP100 GUIDANCE FLOW CONTROL</span><h2>M1 / M2 / M3 vs Full-Head Zero</h2><p>同 case、seed、guidance window 与 λ 下，对照对象内部 R→R、对象接收 C→R、对象广播 R→C 和整颗 head 输出置零；共同 Baseline 与已生成结果按行展示。</p></div><span class="go">打开四信息流受控对照页</span></a>
'''
GT_STC_PREFLIGHT_PORTAL_CARD = r'''
<a class="card new" href="/gt-stc-guidance-preflight?v=1"><div><span>43 / GT-STC PREFLIGHT</span><h2>GT Tube 引导错误预检</h2><p>逐 case 审计 13 个 latent 时刻的 SAM2 region 与 CoTracker 可见性，区分已复现报错、未来必报错、运行中和待运行。</p></div><span class="go">打开 GT Tube 诊断页</span></a>
'''
GT_STC_RESULTS_PORTAL_CARD = r'''
<a class="card new" href="/gt-stc-guidance-results?v=6"><div><span>44 / GT-STC + ATTENTION MICROSCOPE</span><h2>GT 轨迹潜变量引导结果</h2><p>实时展示 context-Query → future-Key 双协议矩阵，并在最终生成 RGB 帧上逐 step 对比 PRE/POST attention 静态 overlay；未生成项固定为 Pending。</p></div><span class="go">打开双协议实时结果</span></a>
'''
GT_STC_METHOD_COMPARISON_PORTAL_CARD = r'''
<a class="card new" href="/gt-stc-guidance-method-comparison?v=5"><div><span>49 / LATENT × DIRECT ATTENTION</span><h2>两种轨迹干预机制对比</h2><p>固定 001460、object A、seed 47326，用全部视频、全帧拼图、GT correspondence loss、轨迹指标和 PRE/POST attention 对比两种干预。</p></div><span class="go">打开机制对比台</span></a>
'''
GT_STC_FIRST10_COMPARISON_PORTAL_CARD = r'''
<a class="card new" href="/gt-stc-first10-vs-full40?v=1"><div><span>50 / DENOISING WINDOW CONTROL</span><h2>前 10 step vs 全 40 step</h2><p>3 个 0613 case，固定 seed 47326、latest3350 Top100、λ=0.1；Region / Point / Combined 逐组左右对比完整引导与仅高噪声前 10 步引导。</p></div><span class="go">打开去噪窗口对比</span></a>
'''
GT_STC_DIRECT_MULTICASE_PORTAL_CARD = r'''
<a class="card new" href="/gt-stc-direct-attention-multicase?v=1"><div><span>52 / DIRECT ATTENTION MULTICASE</span><h2>5 Case × 3 Seed 配置选择</h2><p>对比 Top100、Bottom100、Random100 的 Context→Future、Future→Context 与双向控制；展示轨迹排行、七项 VBench、全部视频及 Pending 状态。</p></div><span class="go">打开批量配置选择台</span></a>
'''
GT_STC_HYPERPARAM_SEARCH_PORTAL_CARD = r'''
<a class="card new" href="/gt-stc-hyperparam-search?v=1"><div><span>53 / GT-STC HYPERPARAMETER SEARCH</span><h2>First10 小 λ 轨迹–像素权衡</h2><p>001460 / object A / seed 47326；Region、Point、Combined × λ {0.005,0.01,0.02,0.05}，展示全部视频、CoTracker 轨迹门控和 GT 对象/背景 MSE。</p></div><span class="go">打开超参搜索页</span></a>
'''
STAGE5_TOKEN_OVERLAP_PORTAL_CARD = r'''
<a class="card new" href="/object-query-stage5-token-overlap?v=1"><div><span>54 / STAGE 5 TOKEN OVERLAP AUDIT</span><h2>多对象 Latent Token 重叠</h2><p>逐帧并排 Baseline 与 22×40 token overlay；红框精确标出两个对象共享的 latent cell，并列出 latent 时刻、网格坐标和全局 token ID。</p></div><span class="go">打开重叠 Token 显微镜</span></a>
'''
FULL_MASK_SIGNATURE_PORTAL_CARD = r'''
<a class="card new" href="/object-query-full-mask-signature?v=1"><div><span>55 / FULL OBJECT MASK ABLATION</span><h2>完整对象区域消融</h2><p>固定 case、seed、head group 与 M1/M2/M3，对比 Baseline、旧 sparse tube 和完整 SAM2 mask signature；共享 R_AB/R_ABC token 单独成块。</p></div><span class="go">打开 Full-mask 对比页</span></a>
'''
viewer.PORTAL = viewer.PORTAL.replace(
    "</section>", PORTAL_CARD + VIDEOS_PORTAL_CARD + QK_ATTENTION_PORTAL_CARD + ATTENTION_LORA_PORTAL_CARD + MONO_SCALE_HEAD_PORTAL_CARD + MONO_SCALE_LORA_VIDEO_PORTAL_CARD + ATTENTION_LORA_SEED_SWEEP_PORTAL_CARD + STEP_ALIGNMENT_PORTAL_CARD + UNLISTED_PORTAL_CARD + INFORMATION_FLOW_VALIDATION_PORTAL_CARD + STAGE4_TEMPORAL_PORTAL_CARD + STAGE4_REPRESENTATIVES_PORTAL_CARD + TOP100_M1_GUIDANCE_PORTAL_CARD + TOP100_M1_TOKEN_COMMUNICATION_PORTAL_CARD + TRAINING_FREE_M1_CONTROL_PORTAL_CARD + TRAINING_FREE_M1_PHASE_BD_PORTAL_CARD + TRAINING_FREE_M1_MULTI_OBJECT_PORTAL_CARD + GT_STC_PREFLIGHT_PORTAL_CARD + GT_STC_RESULTS_PORTAL_CARD + GT_STC_METHOD_COMPARISON_PORTAL_CARD + GT_STC_FIRST10_COMPARISON_PORTAL_CARD + GT_STC_DIRECT_MULTICASE_PORTAL_CARD + GT_STC_HYPERPARAM_SEARCH_PORTAL_CARD + STAGE5_TOKEN_OVERLAP_PORTAL_CARD + FULL_MASK_SIGNATURE_PORTAL_CARD + "</section>", 1
)
OBJECT_QUERY_ANTI_DUPLICATION_PORTAL_CARD = r'''
<a class="card new" href="/object-query-anti-duplication?v=3"><div><span>57 / OBJECT QUERY ANTI-DUPLICATION</span><h2>检测门控的 R→F 去重探索</h2><p>按 case 汇总全部已生成实验；每个 case 的所有 seed、Baseline、Detector-gated 与 Broad Q@K 扫描在同一页按组对比。</p></div><span class="go">打开完整 case 实验桌</span></a>
'''
PHASE_B_ATTENTION_OVERLAY_PORTAL_CARD = r'''
<a class="card new" href="/training-free-m1-phase-b-attention?v=1&amp;window=all40"><div><span>58 / PHASE-B ATTENTION OVERLAY</span><h2>Seed 90094 · 5 × 13 Latent 热力图</h2><p>Baseline、Sparse 8-point 与 SAM2 Full-mask 的 α=0.1/0.25 五行对照；默认 40-step mean，并切换 First10 / First20 / Last20。</p></div><span class="go">打开 13-Latent Attention 页</span></a>
'''
viewer.PORTAL = viewer.PORTAL.replace(
    "</section>",
    OBJECT_QUERY_ANTI_DUPLICATION_PORTAL_CARD
    + PHASE_B_ATTENTION_OVERLAY_PORTAL_CARD
    + "</section>",
    1,
)


from AAA_my_test import serve_attention_noise_metrics as combined_metrics
from AAA_my_test.object_query_ablation_metrics import dashboard as object_query_metrics_dashboard
from AAA_my_test.object_query_ablation_metrics import head_scope_comparison
from AAA_my_test.object_query_ablation_metrics import information_flow_validation_dashboard
from AAA_my_test.object_query_ablation_metrics import stage4_representatives_dashboard
from AAA_my_test.object_query_ablation_metrics import stage4_temporal_dashboard
from AAA_my_test.object_query_ablation_metrics import stage5_token_overlap_dashboard
from AAA_my_test.object_query_ablation_metrics import full_mask_signature_dashboard
from AAA_my_test.object_query_ablation_metrics import top100_m1_guidance_dashboard
from AAA_my_test.object_query_ablation_metrics import top100_m1_token_communication_dashboard
from AAA_my_test.object_query_ablation_metrics import object_query_anti_duplication_dashboard
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control import dashboard as training_free_m1_control_dashboard
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control import phase_bd_dashboard
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control import phase_b_attention_overlay_dashboard
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control import multi_object_search_dashboard
from AAA_my_test import gt_stc_guidance_dashboard
from AAA_my_test import gt_stc_direct_attention_multicase_dashboard
from AAA_my_test import gt_stc_first10_comparison_dashboard
from AAA_my_test import gt_stc_hyperparam_search_dashboard
from AAA_my_test import gt_stc_guidance_method_comparison_dashboard
from AAA_my_test import gt_stc_guidance_results_dashboard


class MetricsHandler(viewer.Handler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/object-query-anti-duplication":
            self.send_payload(
                object_query_anti_duplication_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/object-query-anti-duplication/catalog":
            payload = json.dumps(
                object_query_anti_duplication_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-anti-duplication/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = object_query_anti_duplication_dashboard.asset_path(
                params.get("key", [""])[0]
            )
            if asset is None:
                raise FileNotFoundError("anti-duplication asset is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/object-query-ablation-metrics":
            self.send_payload(
                object_query_metrics_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-information-flow-validation":
            self.send_payload(
                information_flow_validation_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-information-flow-stage4":
            self.send_payload(
                stage4_temporal_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-information-flow-stage4-representatives":
            self.send_payload(
                stage4_representatives_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-stage5-token-overlap":
            self.send_payload(
                stage5_token_overlap_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-full-mask-signature":
            self.send_payload(
                full_mask_signature_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/top100-m1-guidance-pilot":
            self.send_payload(
                top100_m1_guidance_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/top100-m1-token-communication":
            self.send_payload(
                top100_m1_token_communication_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/training-free-m1-control":
            self.send_payload(
                training_free_m1_control_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/training-free-m1-phase-bd":
            self.send_payload(
                phase_bd_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/training-free-m1-phase-b-attention":
            self.send_payload(
                phase_b_attention_overlay_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/training-free-m1-multi-object-search":
            self.send_payload(
                multi_object_search_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/gt-stc-guidance-preflight":
            self.send_payload(
                gt_stc_guidance_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/gt-stc-guidance-results":
            self.send_payload(
                gt_stc_guidance_results_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/gt-stc-first10-vs-full40":
            self.send_payload(
                gt_stc_first10_comparison_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/gt-stc-hyperparam-search":
            self.send_payload(
                gt_stc_hyperparam_search_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/gt-stc-guidance-method-comparison":
            self.send_payload(
                gt_stc_guidance_method_comparison_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/gt-stc-direct-attention-multicase":
            self.send_payload(
                gt_stc_direct_attention_multicase_dashboard.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/gt-stc-direct-attention-multicase/catalog":
            payload = json.dumps(
                gt_stc_direct_attention_multicase_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/gt-stc-hyperparam-search/catalog":
            payload = json.dumps(
                gt_stc_hyperparam_search_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/gt-stc-hyperparam-search/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            result = gt_stc_hyperparam_search_dashboard.asset(
                params.get("variant", [""])[0]
            )
            if result is None or not result.is_file():
                self.send_error(404, "GT-STC hyperparameter-search asset is not ready")
                return
            viewer.send_file_with_range(self, result, "video/mp4")
            return
        if path == "/api/gt-stc-direct-attention-multicase/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            result = gt_stc_direct_attention_multicase_dashboard.asset(
                params.get("kind", [""])[0],
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("config", [""])[0],
            )
            if result is None or not result.is_file():
                self.send_error(404, "direct-attention multicase asset is not ready")
                return
            viewer.send_file_with_range(self, result, "video/mp4")
            return
        if path == "/api/gt-stc-guidance-method-comparison/catalog":
            payload = json.dumps(
                gt_stc_guidance_method_comparison_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/gt-stc-guidance-method-comparison/source-video":
            result = gt_stc_guidance_method_comparison_dashboard.source_video_asset()
            if result is None or not result.is_file():
                self.send_error(404, "guidance comparison source video is not ready")
                return
            viewer.send_file_with_range(self, result, "video/mp4")
            return
        if path == "/api/gt-stc-guidance-method-comparison/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            result = gt_stc_guidance_method_comparison_dashboard.asset(
                params.get("kind", [""])[0],
                params.get("method", [""])[0],
                params.get("group", [""])[0],
                params.get("direction", [""])[0],
                params.get("latent", [""])[0],
                params.get("step", [""])[0],
            )
            if result is None or not result.is_file():
                self.send_error(404, "guidance comparison asset is not ready")
                return
            content_type = (
                "video/mp4"
                if params.get("kind", [""])[0] == "video"
                else "image/jpeg"
            )
            viewer.send_file_with_range(self, result, content_type)
            return
        if path == "/api/gt-stc-guidance-results/catalog":
            payload = json.dumps(
                gt_stc_guidance_results_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/gt-stc-first10-vs-full40/catalog":
            payload = json.dumps(
                gt_stc_first10_comparison_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/gt-stc-first10-vs-full40/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = gt_stc_first10_comparison_dashboard.asset(
                params.get("kind", [""])[0],
                case=params.get("case", [""])[0],
                target=params.get("target", [""])[0],
                variant=params.get("variant", [""])[0],
            )
            if asset is None or not asset.is_file():
                self.send_error(404, "first10/full40 comparison asset is not ready")
                return
            content_type = (
                "image/jpeg"
                if asset.suffix.lower() in {".jpg", ".jpeg"}
                else "video/mp4"
            )
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/api/gt-stc-guidance-results/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = gt_stc_guidance_results_dashboard.asset(
                params.get("kind", [""])[0],
                case=params.get("case", [""])[0],
                target=params.get("target", [""])[0],
                variant=params.get("variant", [""])[0],
                backend=params.get("backend", [""])[0],
                step=params.get("step", [""])[0],
                latent=params.get("latent", [""])[0],
            )
            if asset is None or not asset.is_file():
                self.send_error(404, "GT-STC validation asset is not ready")
                return
            content_type = "image/jpeg" if asset.suffix.lower() in {".jpg", ".jpeg"} else "video/mp4"
            viewer.send_file_with_range(self, asset, content_type)
            return
        if path == "/api/gt-stc-guidance-preflight/catalog":
            payload = json.dumps(
                gt_stc_guidance_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/gt-stc-guidance-preflight/log":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            log_path = gt_stc_guidance_dashboard.log_file(
                params.get("name", [""])[0]
            )
            if log_path is None:
                raise FileNotFoundError("unknown GT-STC guidance log")
            viewer.send_file_with_range(self, log_path, "text/plain; charset=utf-8")
            return
        if path == "/api/gt-stc-guidance-preflight/filmstrip":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            filmstrip = gt_stc_guidance_dashboard.region_filmstrip(
                params.get("case", [""])[0]
            )
            if filmstrip is None or not filmstrip.is_file():
                raise FileNotFoundError("unknown GT-STC source-region filmstrip")
            viewer.send_file_with_range(self, filmstrip, "image/jpeg")
            return
        if path == "/api/gt-stc-guidance-preflight/cotracker-comparison":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            comparison = gt_stc_guidance_dashboard.cotracker_comparison(
                params.get("case", [""])[0]
            )
            if comparison is None or not comparison.is_file():
                raise FileNotFoundError("unknown GT-STC SAM2/CoTracker comparison")
            viewer.send_file_with_range(self, comparison, "image/jpeg")
            return
        if path == "/api/gt-stc-guidance-preflight/hybrid-comparison":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            comparison = gt_stc_guidance_dashboard.hybrid_comparison(
                params.get("case", [""])[0]
            )
            if comparison is None or not comparison.is_file():
                raise FileNotFoundError("unknown GT-STC CoTracker/SAM2 hybrid comparison")
            viewer.send_file_with_range(self, comparison, "image/jpeg")
            return
        if path == "/api/object-query-information-flow-validation/catalog":
            payload = json.dumps(
                information_flow_validation_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-information-flow-stage4/catalog":
            payload = json.dumps(
                stage4_temporal_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-information-flow-stage4-representatives/catalog":
            payload = json.dumps(
                stage4_representatives_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-stage5-token-overlap/catalog":
            payload = json.dumps(
                stage5_token_overlap_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-full-mask-signature/catalog":
            payload = json.dumps(
                full_mask_signature_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/top100-m1-guidance-pilot/catalog":
            payload = json.dumps(
                top100_m1_guidance_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/top100-m1-guidance-pilot/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            result = top100_m1_guidance_dashboard.asset(
                params.get("kind", [""])[0],
                params.get("case", [""])[0],
                int(params.get("seed", ["47326"])[0]),
                float(params.get("pag_scale", ["0.5"])[0]),
            )
            if result is None or not result.is_file():
                raise FileNotFoundError("Top100-M1 guidance pilot asset is not ready")
            viewer.send_file_with_range(self, result, "video/mp4")
            return
        if path == "/api/top100-m1-token-communication/catalog":
            payload = json.dumps(
                top100_m1_token_communication_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/training-free-m1-control/catalog":
            payload = json.dumps(
                training_free_m1_control_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/training-free-m1-control/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["47326"])[0])
                value = float(params.get("value", ["0"])[0])
            except ValueError:
                raise FileNotFoundError("invalid Training-Free M1 asset coordinates")
            result = training_free_m1_control_dashboard.asset(
                params.get("family", [""])[0],
                params.get("case", [""])[0],
                seed,
                value,
            )
            if result is None or not result.is_file():
                raise FileNotFoundError("Training-Free M1 asset is not ready")
            viewer.send_file_with_range(self, result, "video/mp4")
            return
        if path == "/api/training-free-m1-phase-bd/catalog":
            payload = json.dumps(
                phase_bd_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/training-free-m1-phase-bd/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["-1"])[0])
            except ValueError:
                raise FileNotFoundError("invalid Phase-B/D seed")
            result = phase_bd_dashboard.asset(
                params.get("case", [""])[0],
                seed,
                params.get("asset_id", [""])[0],
            )
            if result is None or not result.is_file():
                raise FileNotFoundError("Phase-B/D asset is not ready")
            viewer.send_file_with_range(self, result, "video/mp4")
            return
        if path == "/api/training-free-m1-phase-b-attention/catalog":
            payload = json.dumps(
                phase_b_attention_overlay_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/training-free-m1-phase-b-attention/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                latent = int(params.get("latent", ["-1"])[0])
            except ValueError:
                raise FileNotFoundError("invalid Phase-B attention latent index")
            result = phase_b_attention_overlay_dashboard.asset(
                params.get("kind", [""])[0],
                params.get("variant_id", [""])[0],
                params.get("window", [""])[0],
                latent,
            )
            if result is None or not result.is_file():
                raise FileNotFoundError("Phase-B attention asset is not ready")
            content_type = "video/mp4" if result.suffix == ".mp4" else "image/jpeg"
            viewer.send_file_with_range(self, result, content_type)
            return
        if path == "/api/training-free-m1-multi-object-search/catalog":
            payload = json.dumps(
                multi_object_search_dashboard.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/training-free-m1-multi-object-search/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["-1"])[0])
                scale = float(params.get("scale", ["0"])[0])
                start = int(params.get("start", ["0"])[0])
                end = int(params.get("end", ["0"])[0])
            except ValueError:
                raise FileNotFoundError("invalid multi-object M1 asset coordinates")
            result = multi_object_search_dashboard.asset(
                params.get("kind", [""])[0],
                params.get("case", [""])[0],
                seed,
                scale,
                start,
                end,
            )
            if result is None or not result.is_file():
                raise FileNotFoundError("multi-object M1 asset is not ready")
            viewer.send_file_with_range(self, result, "video/mp4")
            return
        if path == "/api/top100-m1-token-communication/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                anchor = int(params.get("anchor", ["-1"])[0])
            except ValueError:
                anchor = -1
            kind = params.get("kind", [""])[0]
            result = top100_m1_token_communication_dashboard.asset(
                kind,
                params.get("case", [""])[0],
                anchor,
            )
            if result is None or not result.is_file():
                raise FileNotFoundError("Top100-M1 token communication asset is not ready")
            content_type = "image/jpeg" if kind in {"query", "key"} else "video/mp4"
            viewer.send_file_with_range(self, result, content_type)
            return
        if path == "/api/object-query-information-flow-stage4/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["-1"])[0])
            except ValueError:
                seed = -1
            asset = stage4_temporal_dashboard.asset(
                params.get("kind", [""])[0],
                params.get("case", [""])[0],
                seed,
                params.get("variant", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("Stage-4 information-flow asset is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-stage5-token-overlap/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["-1"])[0])
            except ValueError:
                seed = -1
            asset = stage5_token_overlap_dashboard.asset(
                params.get("case", [""])[0],
                seed,
                params.get("kind", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("Stage-5 token-overlap asset is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-full-mask-signature/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = full_mask_signature_dashboard.asset(
                params.get("kind", [""])[0],
                params.get("case", [""])[0],
                params.get("scope", [""])[0],
                params.get("mode", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("Full-mask signature asset is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-information-flow-stage4/dose":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["-1"])[0])
            except ValueError:
                seed = -1
            result = stage4_temporal_dashboard.dose(
                params.get("case", [""])[0],
                seed,
                params.get("variant", [""])[0],
            )
            if result is None:
                raise FileNotFoundError("Stage-4 dose is not ready")
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-information-flow-validation/asset":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["-1"])[0])
            except ValueError:
                seed = -1
            asset = information_flow_validation_dashboard.asset(
                params.get("kind", [""])[0],
                case=params.get("case", [""])[0],
                seed=seed,
                variant=params.get("variant", [""])[0],
                name=params.get("name", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("information-flow validation asset is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/object-query-information-flow-validation/dose":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", ["-1"])[0])
            except ValueError:
                seed = -1
            result = information_flow_validation_dashboard.dose(
                params.get("stage", [""])[0],
                params.get("case", [""])[0],
                seed,
                params.get("variant", [""])[0],
            )
            if result is None:
                raise FileNotFoundError("information-flow dose is not ready")
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/object-query-m123-s039-top100-mean-overlays":
            self.send_payload(
                wan22_ti2v_legacy_m123_s039_top100_mean_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-m1-temporal-gallery":
            self.send_payload(
                wan22_ti2v_legacy_m1_temporal_gallery_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-m123-temporal-batch":
            self.send_payload(
                wan22_ti2v_legacy_m123_temporal_batch_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-head-scope-comparison":
            self.send_payload(
                head_scope_comparison.page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/object-query-all720-ablation-gallery":
            self.send_payload(
                wan22_ti2v_legacy_all720_ablation_gallery_page().encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/object-query-m123-temporal-batch/catalog":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            case = params.get("case", [None])[0]
            seed_text = params.get("seed", [None])[0]
            try:
                seed = int(seed_text) if seed_text is not None else None
            except ValueError:
                seed = None
            selected_only = params.get("single", ["0"])[0] == "1"
            payload = json.dumps(
                wan22_ti2v_legacy_m123_temporal_batch_catalog(
                    case, seed, selected_only=selected_only
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-head-scope-comparison/catalog":
            payload = json.dumps(
                head_scope_comparison.catalog(),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        if path == "/api/object-query-head-scope-comparison/video":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            try:
                seed = int(params.get("seed", [""])[0])
            except ValueError:
                seed = -1
            asset = head_scope_comparison.asset(
                params.get("case", [""])[0],
                seed,
                params.get("variant_id", [""])[0],
                params.get("view", ["generated"])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("Head-Scope comparison video is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
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
                params.get("head_scope", ["top100"])[0],
                params.get("ranking_tag", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("PhysicIQ67 temporal-tube ablation video is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/head-scope-trajectory-overlay":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_head_scope_trajectory_overlay(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("variant_id", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("Head-Scope trajectory overlay is not ready")
            viewer.send_file_with_range(self, asset, "video/mp4")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/head-scope-s039-top100-mean-overlay":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_m123_s039_top100_mean_overlay(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("variant_id", [""])[0],
                params.get("region", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("S039 Top100 mean overlay is not ready")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/head-scope-s039-query-receiver-overlay":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_m123_s039_query_receiver_overlay(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("variant_id", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("S039 query-side receiver overlay is not ready")
            viewer.send_file_with_range(self, asset, "image/jpeg")
            return
        if path == "/api/wan22-ti2v-legacy-physiciq67-samples/head-scope-survival-overlay":
            from urllib.parse import parse_qs

            params = parse_qs(urlparse(self.path).query)
            asset = wan22_ti2v_legacy_physiciq67_head_scope_survival_overlay(
                params.get("case", [""])[0],
                params.get("seed", [""])[0],
                params.get("variant_id", [""])[0],
            )
            if asset is None or not asset.is_file():
                raise FileNotFoundError("Head-Scope survival overlay is not ready")
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
WAN22_TI2V_LEGACY_PHYSICIQ67_LATEST_OTHER10_MANIFEST = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT
    / "cases_other10_6seeds_latest.json"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_LATEST_OTHER10_HEAD_RANKING = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT
    / "pck_head_scopes_s039_latest2735.json"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_LATEST3350_HEAD_RANKING = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT
    / "pck_head_scopes_s039_latest3350.json"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT
    / "attention_matrix_ablations_temporal_tube_v1"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPE_METRICS_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics/"
    "head_scope_baseline_fast"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPE_TRAJECTORY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics/"
    "head_scope_trajectory"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_M123_S039_TOP100_MEAN_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_attention_overlays/"
    "m123_head_scope_s039_top100_mean_v1"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_M123_S039_QUERY_RECEIVER_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_attention_overlays/"
    "m123_head_scope_s039_query_receiver_v1"
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
        if WAN22_TI2V_LEGACY_PHYSICIQ67_LATEST_OTHER10_MANIFEST.is_file():
            latest = json.loads(
                WAN22_TI2V_LEGACY_PHYSICIQ67_LATEST_OTHER10_MANIFEST.read_text(
                    encoding="utf-8"
                )
            )
            sample_by_key = {
                (str(row.get("case")), int(row.get("seed", -1))): row
                for row in payload.get("samples", [])
            }
            for latest_row in latest.get("samples", []):
                key = (
                    str(latest_row.get("case")),
                    int(latest_row.get("seed", -1)),
                )
                experiment = {
                    "m123_head_ranking_path": str(
                        WAN22_TI2V_LEGACY_PHYSICIQ67_LATEST_OTHER10_HEAD_RANKING
                    ),
                    "m123_head_ranking_tag": "s039r2735",
                    "m123_head_scopes": ["top100", "bottom100", "all720"],
                }
                if key in sample_by_key:
                    sample_by_key[key].update(experiment)
                else:
                    appended = {**latest_row, **experiment}
                    payload["samples"].append(appended)
                    sample_by_key[key] = appended
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
WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS = (
    "self_same",
    "self_future",
    "self_past",
    "incoming_same",
    "incoming_future",
    "incoming_past",
    "outgoing_same",
    "outgoing_future",
    "outgoing_past",
)
WAN22_TI2V_LEGACY_PHYSICIQ67_M123_GALLERY_MASKS = (
    "self_only",
    "incoming_only",
    "outgoing_only",
) + WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS
WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_PROTOCOL = (
    "attention_matrix_ablation_temporal_direction_v1"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_RANKING = (
    WAN22_TI2V_LEGACY_PHYSICIQ67_REQUESTED_ROOT
    / "pck_head_scopes_s039_frozen134.json"
)
WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPES = ("top100", "bottom100", "all720")
WAN22_TI2V_LEGACY_PHYSICIQ67_M123_MASKS = (
    "self_only",
    "self_same",
    "self_future",
    "self_past",
    "incoming_only",
    "incoming_same",
    "incoming_future",
    "incoming_past",
    "outgoing_only",
    "outgoing_same",
    "outgoing_future",
    "outgoing_past",
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
    if mask_mode not in (
        WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_MASKS
        + WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS
        + ("literal_kv_zero",)
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
    directional_masks = WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS
    regions = [
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    ]
    targets = [("single_object", region) for region in regions]
    targets.append(("all_objects", ""))
    records = []
    for target_scope, region in targets:
        for mask_mode in (
            WAN22_TI2V_LEGACY_PHYSICIQ67_MATRIX_MASKS
            + directional_masks
            + ("literal_kv_zero",)
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
                    complete = json.loads(
                        (root / "complete.json").read_text(encoding="utf-8")
                    )
                    is_directional = (
                        mask_mode
                        in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS
                    )
                    expected_protocol = (
                        WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_PROTOCOL
                        if is_directional
                        else WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_PROTOCOL
                    )
                    ready = (
                        metadata.get("case") == case
                        and int(metadata.get("seed", -1)) == seed
                        and metadata.get("target_scope") == target_scope
                        and metadata.get("mask_mode") == mask_mode
                        and int(metadata.get("top_n", -1)) == 100
                        and str(metadata.get("region") or "") == region
                        and metadata.get("selected_entries") == entries[:100]
                        and metadata.get("protocol") == expected_protocol
                    )
                    if ready:
                        audit = metadata.get("audit", {})
                        selected_token_count = len(audit.get("query_token_indices") or [])
                        latent_frame_token_counts = audit.get("latent_frame_token_counts")
                        if is_directional:
                            directional_spec = audit.get(
                                "temporal_directional_spec", {}
                            )
                            expected_block = {
                                "self": "S",
                                "incoming": "I",
                                "outgoing": "O",
                            }[mask_mode.rsplit("_", 1)[0]]
                            expected_direction = mask_mode.rsplit("_", 1)[1]
                            expected_events = int(
                                audit.get("expected_head_events", -1)
                            )
                            modified_events = int(
                                audit.get("modified_head_events", -2)
                            )
                            ready = (
                                directional_spec.get("base_block")
                                == expected_block
                                and directional_spec.get("direction")
                                == expected_direction
                                and expected_events > 0
                                and modified_events == expected_events
                                and int(
                                    audit.get(
                                        "temporal_zeroed_entries_per_head", 0
                                    )
                                )
                                > 0
                                and complete.get("case") == case
                                and int(complete.get("seed", -1)) == seed
                                and complete.get("variant_id") == variant
                                and complete.get("protocol") == expected_protocol
                                and int(
                                    complete.get("selected_temporal_tokens", -1)
                                )
                                == selected_token_count
                                and int(complete.get("modified_head_events", -1))
                                == modified_events
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
                    "top_n": 100,
                    "variant_id": variant,
                    "ready": ready,
                    "error": (root / "error.txt").is_file(),
                    "selected_token_count": selected_token_count,
                    "latent_frame_token_counts": latent_frame_token_counts,
                    "temporal_directional": mask_mode
                    in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS,
                    "vbench": vbench,
                }
            )
    return records


def _wan22_ti2v_legacy_physiciq67_head_scope_variant(
    target_scope: str,
    region: str,
    mask_mode: str,
    head_scope: str,
    ranking_tag: str = "",
) -> str | None:
    if head_scope not in WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPES:
        return None
    if mask_mode not in WAN22_TI2V_LEGACY_PHYSICIQ67_M123_MASKS:
        return None
    if target_scope == "single_object":
        if not region:
            return None
        target = region
    elif target_scope == "all_objects":
        target = "all_objects"
    else:
        return None
    suffix = head_scope if not ranking_tag else f"{head_scope}_{ranking_tag}"
    return f"{target_scope}__{target}__{mask_mode}__{suffix}"


def _wan22_ti2v_legacy_physiciq67_m123_targets(regions: list[str]):
    targets = [("single_object", region) for region in regions]
    if len(regions) > 1:
        targets.append(("all_objects", ""))
    return targets


def _wan22_ti2v_legacy_physiciq67_m123_s039_top100_mean_record(
    case: str, seed: int, variant_id: str
):
    if not variant_id or Path(variant_id).name != variant_id:
        return {"ready": False}
    root = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_M123_S039_TOP100_MEAN_ROOT
        / case
        / f"seed_{seed:05d}"
        / variant_id
    )
    if not all(
        (root / name).is_file()
        for name in ("complete.json", "manifest.json", "overlay_manifest.json")
    ):
        return {"ready": False}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        overlay = json.loads(
            (root / "overlay_manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("case") != case
            or int(manifest.get("seed", -1)) != seed
            or manifest.get("variant_id") != variant_id
            or int(manifest.get("capture_step", -1)) != 39
        ):
            raise ValueError("S039 Top100 mean manifest identity mismatch")
        records = []
        for row in overlay.get("records", []):
            region = str(row.get("region_name") or "")
            image = str(row.get("images", {}).get("comparison") or "")
            if region not in {"object_A", "object_B"} or Path(image).name != image:
                raise ValueError("invalid S039 Top100 mean overlay record")
            if not (root / image).is_file():
                raise FileNotFoundError(root / image)
            records.append(
                {
                    "region_name": region,
                    "region_phrase": row.get("region_phrase"),
                    "locally_ablated_top100_heads": row.get(
                        "locally_ablated_top100_heads"
                    ),
                    "locally_ablated_query_rows": row.get(
                        "locally_ablated_query_rows"
                    ),
                    "image": image,
                }
            )
        if [row["region_name"] for row in records] != ["object_A", "object_B"]:
            raise ValueError("expected Object A/B S039 Top100 mean records")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"ready": False}
    return {
        "ready": True,
        "step": 39,
        "query_pixel_frame": 4,
        "query_latent_frame": 1,
        "observation_heads": "frozen S039 PCK Top100",
        "intervention_head_scope": manifest.get("head_scope"),
        "records": records,
    }


def _wan22_ti2v_legacy_physiciq67_m123_s039_query_receiver_record(
    case: str, seed: int, variant_id: str
):
    if not variant_id or Path(variant_id).name != variant_id:
        return {"ready": False}
    root = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_M123_S039_QUERY_RECEIVER_ROOT
        / case
        / f"seed_{seed:05d}"
        / variant_id
    )
    required = (
        "complete.json",
        "manifest.json",
        "overlay_manifest.json",
        "receiver.npz",
        "receiver__s039_query_side_comparison.jpg",
    )
    if not all((root / name).is_file() for name in required):
        return {"ready": False}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        overlay = json.loads(
            (root / "overlay_manifest.json").read_text(encoding="utf-8")
        )
        image = str(overlay.get("images", {}).get("comparison") or "")
        if (
            manifest.get("case") != case
            or int(manifest.get("seed", -1)) != seed
            or manifest.get("variant_id") != variant_id
            or int(manifest.get("capture_step", -1)) != 39
            or image != "receiver__s039_query_side_comparison.jpg"
            or not (root / image).is_file()
        ):
            raise ValueError("invalid S039 query receiver manifest")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"ready": False}
    return {
        "ready": True,
        "step": 39,
        "head_scope": manifest.get("head_scope"),
        "head_count": manifest.get("top_n"),
        "operator_id": overlay.get("operator_id"),
        "temporal_scope": overlay.get("temporal_scope"),
        "target_partition": overlay.get("target_partition"),
        "source_partition": overlay.get("source_partition"),
        "time_predicate": overlay.get("time_predicate"),
        "coefficient_definition": overlay.get("coefficient_definition"),
        "value_definition": overlay.get("value_definition"),
        "scale_mode": overlay.get("scale_mode"),
        "global_vmax": overlay.get("global_vmax", {}),
        "image": image,
    }


def _wan22_ti2v_legacy_physiciq67_m123_head_scope_records(sample: dict):
    case, seed = str(sample["case"]), int(sample["seed"])
    ranking_path = Path(
        str(sample.get("m123_head_ranking_path") or WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_RANKING)
    )
    ranking_tag = str(sample.get("m123_head_ranking_tag") or "")
    head_scopes = tuple(
        str(value)
        for value in sample.get(
            "m123_head_scopes", WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPES
        )
    )
    if not head_scopes or any(
        value not in WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPES
        for value in head_scopes
    ):
        return {"ready": False, "reason": "invalid head scope configuration", "records": []}
    try:
        ranking = json.loads(
            ranking_path.read_text(encoding="utf-8")
        )
        all_entries = list(ranking["entries"])
        if len(all_entries) != 720:
            raise ValueError("frozen head ranking does not contain 720 entries")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"ready": False, "reason": str(exc), "records": []}

    scope_entries = {
        "top100": all_entries[:100],
        "bottom100": all_entries[-100:],
        "all720": all_entries,
    }
    rank_ranges = {
        "top100": [1, 100],
        "bottom100": [621, 720],
        "all720": [1, 720],
    }
    regions = [
        str(row["region_name"])
        for row in sample.get("regions", [])
        if row.get("region_type") == "object"
    ]
    targets = _wan22_ti2v_legacy_physiciq67_m123_targets(regions)
    metric_report_path = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPE_METRICS_ROOT
        / case
        / f"seed_{seed:05d}"
        / "report.json"
    )
    metric_report = {}
    metric_records = {}
    try:
        metric_report = json.loads(metric_report_path.read_text(encoding="utf-8"))
        if (
            metric_report.get("case") != case
            or int(metric_report.get("seed", -1)) != seed
        ):
            raise ValueError("head-scope metric report identity mismatch")
        metric_records = {
            str(row["variant_id"]): row
            for row in metric_report.get("records", [])
            if isinstance(row, dict) and row.get("variant_id")
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        metric_report = {}
        metric_records = {}
    trajectory_report_path = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPE_TRAJECTORY_ROOT
        / case
        / f"seed_{seed:05d}"
        / "report.json"
    )
    trajectory_report = {}
    trajectory_records = {}
    try:
        trajectory_report = json.loads(
            trajectory_report_path.read_text(encoding="utf-8")
        )
        if (
            trajectory_report.get("case") != case
            or int(trajectory_report.get("seed", -1)) != seed
        ):
            raise ValueError("head-scope trajectory report identity mismatch")
        trajectory_records = {
            str(row["variant_id"]): row
            for row in trajectory_report.get("records", [])
            if isinstance(row, dict) and row.get("variant_id")
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        trajectory_report = {}
        trajectory_records = {}
    survival_report_path = trajectory_report_path.parent / "object_survival_report.json"
    survival_report = {}
    survival_records = {}
    try:
        survival_report = json.loads(
            survival_report_path.read_text(encoding="utf-8")
        )
        if (
            survival_report.get("case") != case
            or int(survival_report.get("seed", -1)) != seed
        ):
            raise ValueError("head-scope survival report identity mismatch")
        survival_records = {
            str(row["variant_id"]): row
            for row in survival_report.get("records", [])
            if isinstance(row, dict) and row.get("variant_id")
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        survival_report = {}
        survival_records = {}
    records = []
    for head_scope in head_scopes:
        expected_entries = scope_entries[head_scope]
        for target_scope, region in targets:
            for mask_mode in WAN22_TI2V_LEGACY_PHYSICIQ67_M123_MASKS:
                variant = _wan22_ti2v_legacy_physiciq67_head_scope_variant(
                    target_scope, region, mask_mode, head_scope, ranking_tag
                )
                root = (
                    WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT
                    / case
                    / f"seed_{seed:05d}"
                    / str(variant)
                )
                ready = all(
                    (root / name).is_file()
                    for name in ("complete.json", "manifest.json", "generated.mp4")
                )
                error = (root / "error.txt").is_file()
                if ready:
                    try:
                        metadata = json.loads(
                            (root / "manifest.json").read_text(encoding="utf-8")
                        )
                        expected_protocol = (
                            WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_PROTOCOL
                            if mask_mode
                            in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS
                            else WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_PROTOCOL
                        )
                        ready = (
                            metadata.get("case") == case
                            and int(metadata.get("seed", -1)) == seed
                            and metadata.get("target_scope") == target_scope
                            and str(metadata.get("region") or "") == region
                            and metadata.get("mask_mode") == mask_mode
                            and metadata.get("selected_entries") == expected_entries
                            and metadata.get("protocol") == expected_protocol
                            and (
                                metadata.get("head_scope", "top100") == head_scope
                            )
                            and str(metadata.get("ranking_tag") or "") == ranking_tag
                            and int(
                                metadata.get(
                                    "selected_head_count", len(expected_entries)
                                )
                            )
                            == len(expected_entries)
                        )
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        ready = False
                records.append(
                    {
                        "target_scope": target_scope,
                        "region": region or None,
                        "mask_mode": mask_mode,
                        "head_scope": head_scope,
                        "head_count": len(expected_entries),
                        "rank_start": rank_ranges[head_scope][0],
                        "rank_end": rank_ranges[head_scope][1],
                        "variant_id": variant,
                        "ready": ready,
                        "error": error,
                        "baseline_metrics": metric_records.get(str(variant)),
                        "trajectory_metrics": trajectory_records.get(str(variant)),
                        "object_survival_metrics": survival_records.get(str(variant)),
                        "s039_top100_mean": (
                            _wan22_ti2v_legacy_physiciq67_m123_s039_top100_mean_record(
                                case, seed, str(variant)
                            )
                        ),
                        "s039_query_receiver": (
                            _wan22_ti2v_legacy_physiciq67_m123_s039_query_receiver_record(
                                case, seed, str(variant)
                            )
                        ),
                        "temporal_directional": mask_mode
                        in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_DIRECTIONAL_MASKS,
                    }
                )
    visible_variants = {str(record["variant_id"]) for record in records}
    metric_ranking_rows = [
        {
            "variant_id": row["variant_id"],
            "target_scope": row.get("target_scope"),
            "region": row.get("region"),
            "mask_mode": row.get("mask_mode"),
            "head_scope": row.get("head_scope"),
            "impact_rank_within_case_seed": row.get(
                "impact_rank_within_case_seed"
            ),
            "impact_score_0_100": row.get("metrics", {}).get(
                "impact_score_0_100"
            ),
            "spillover_score_0_100": row.get("metrics", {}).get(
                "spillover_score_0_100"
            ),
            "category_scores_0_100": row.get("metrics", {}).get(
                "category_scores_0_100", {}
            ),
            "category_ranks_within_case_seed": row.get(
                "category_ranks_within_case_seed", {}
            ),
        }
        for row in metric_records.values()
        if row.get("variant_id") in visible_variants
    ]
    category_definitions = metric_report.get("category_definitions", {})
    trajectory_ranking = sorted(
        [
            row
            for variant, row in trajectory_records.items()
            if variant in visible_variants
            and row.get("metrics", {}).get("trajectory_impact_percent_d0")
            is not None
        ],
        key=lambda row: (
            -float(row["metrics"]["trajectory_impact_percent_d0"]),
            str(row.get("variant_id") or ""),
        ),
    )
    track_loss_ranking = sorted(
        [
            row
            for variant, row in trajectory_records.items()
            if variant in visible_variants
            and row.get("metrics", {}).get(
                "target_worst_track_loss_score_0_100"
            )
            is not None
        ],
        key=lambda row: (
            -float(row["metrics"]["target_worst_track_loss_score_0_100"]),
            str(row.get("variant_id") or ""),
        ),
    )
    disappearance_ranking = sorted(
        [
            row
            for variant, row in survival_records.items()
            if variant in visible_variants
            and row.get("metrics", {}).get(
                "target_worst_disappearance_score_0_100"
            )
            is not None
        ],
        key=lambda row: (
            -float(
                row["metrics"]["target_worst_disappearance_score_0_100"]
            ),
            str(row.get("variant_id") or ""),
        ),
    )
    mask_absence_ranking = sorted(
        [
            row
            for variant, row in survival_records.items()
            if variant in visible_variants
            and row.get("metrics", {}).get(
                "target_worst_mask_absence_score_0_100"
            )
            is not None
        ],
        key=lambda row: (
            -float(row["metrics"]["target_worst_mask_absence_score_0_100"]),
            str(row.get("variant_id") or ""),
        ),
    )
    return {
        "ready": True,
        "ranking_step": int(ranking.get("ranking_step", 39)),
        "completed_runs_at_selection": int(
            ranking.get("completed_runs_at_selection", 0)
        ),
        "ranking_tag": ranking_tag,
        "head_scopes": list(head_scopes),
        "head_ranking_path": str(ranking_path),
        "metric_report_ready": bool(metric_records),
        "metric_report_path": str(metric_report_path),
        "metric_definitions": metric_report.get("metric_definitions", {}),
        "trajectory_report_ready": bool(trajectory_records),
        "trajectory_report_path": str(trajectory_report_path),
        "trajectory_definition": trajectory_report.get(
            "trajectory_definition", {}
        ),
        "trajectory_progress": {
            "tracked": int(trajectory_report.get("tracked_ablation_count", 0)),
            "ranked": int(trajectory_report.get("ranked_ablation_count", 0)),
            "expected": int(trajectory_report.get("expected_ablation_count", 0)),
        },
        "trajectory_ranking": trajectory_ranking,
        "track_loss_definition": trajectory_report.get(
            "track_loss_definition", {}
        ),
        "track_loss_ranking": track_loss_ranking,
        "object_survival_report_ready": bool(survival_records),
        "object_survival_report_path": str(survival_report_path),
        "object_survival_definition": survival_report.get(
            "object_survival_definition", {}
        ),
        "object_survival_progress": {
            "measured": int(survival_report.get("measured_ablation_count", 0)),
            "ranked": int(survival_report.get("ranked_ablation_count", 0)),
            "expected": int(survival_report.get("expected_ablation_count", 0)),
        },
        "s039_top100_mean_progress": {
            "ready": sum(
                int(bool(row.get("s039_top100_mean", {}).get("ready")))
                for row in records
            ),
            "expected": len(records),
        },
        "s039_query_receiver_progress": {
            "ready": sum(
                int(bool(row.get("s039_query_receiver", {}).get("ready")))
                for row in records
            ),
            "expected": len(records),
        },
        "disappearance_ranking": disappearance_ranking,
        "mask_absence_ranking": mask_absence_ranking,
        "category_definitions": category_definitions,
        "category_rankings": {
            category_id: sorted(
                metric_ranking_rows,
                key=lambda row: (
                    -float(
                        row.get("category_scores_0_100", {}).get(
                            category_id, -1
                        )
                    ),
                    str(row.get("variant_id") or ""),
                ),
            )
            for category_id in category_definitions
        },
        "impact_ranking": sorted(
            metric_ranking_rows,
            key=lambda row: (
                -float(row.get("impact_score_0_100") or -1),
                str(row.get("variant_id") or ""),
            ),
        ),
        "records": records,
    }


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
        if (
            (case == WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE and seed == 47326)
            or sample.get("m123_head_ranking_path")
        ):
            sample["m123_head_scope_ablations"] = (
                _wan22_ti2v_legacy_physiciq67_m123_head_scope_records(sample)
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


def _wan22_ti2v_legacy_m123_batch_metric_aggregate(samples, summaries):
    mode_order = (
        "self_only",
        "self_same",
        "self_future",
        "self_past",
        "incoming_only",
        "incoming_same",
        "incoming_future",
        "incoming_past",
        "outgoing_only",
        "outgoing_same",
        "outgoing_future",
        "outgoing_past",
    )
    appearance_metric_names = (
        "impact",
        "global_appearance",
        "target_local",
        "temporal_appearance",
        "outside_spillover",
        "global_ssim",
        "global_mae",
        "global_delta_mae",
        "target_roi_mae",
        "target_roi_delta_mae",
        "outside_mae",
        "outside_delta_mae",
    )

    def mean(values):
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        return round(math.fsum(finite) / len(finite), 8) if finite else None

    def balanced_metrics(rows):
        by_case_seed = {}
        for row in rows:
            by_case_seed.setdefault((row["case"], row["seed"]), []).append(row)
        case_seed_means = {
            key: {
                metric: mean(item.get(metric) for item in values)
                for metric in appearance_metric_names
            }
            for key, values in by_case_seed.items()
        }
        by_case = {}
        for (case, _seed), values in case_seed_means.items():
            by_case.setdefault(case, []).append(values)
        case_means = {
            case: {
                metric: mean(item.get(metric) for item in values)
                for metric in appearance_metric_names
            }
            for case, values in by_case.items()
        }
        return {
            metric: mean(values.get(metric) for values in case_means.values())
            for metric in appearance_metric_names
        }

    summary_lookup = {
        (str(row["case"]), int(row["seed"])): row for row in summaries
    }
    case_rows = {}
    appearance_rows = []
    trajectory_rows = []
    for sample in samples:
        case, seed = str(sample["case"]), int(sample["seed"])
        case_row = case_rows.setdefault(
            case,
            {
                "case": case,
                "configured_seeds": set(),
                "generated_seeds": set(),
                "appearance_seeds": set(),
                "trajectory_seeds": set(),
                "survival_seeds": set(),
                "expected_videos": 0,
                "generated_videos": 0,
                "appearance_records": 0,
                "trajectory_records": 0,
                "survival_records": 0,
            },
        )
        case_row["configured_seeds"].add(seed)
        summary = summary_lookup.get((case, seed), {})
        case_row["expected_videos"] += int(summary.get("expected", 0))
        payload = _wan22_ti2v_legacy_physiciq67_m123_head_scope_records(sample)
        for record in payload.get("records", []):
            if (
                not record.get("ready")
                or record.get("mask_mode") not in mode_order
            ):
                continue
            case_row["generated_videos"] += 1
            case_row["generated_seeds"].add(seed)
            target = (
                f"single_object::{record.get('region')}"
                if record.get("target_scope") == "single_object"
                else "all_objects::"
            )
            baseline_row = record.get("baseline_metrics") or {}
            baseline = baseline_row.get("metrics", {})
            if baseline:
                global_metrics = baseline.get("global", {})
                target_roi = baseline.get("target_roi", {})
                outside = baseline.get("outside_objects", {})
                categories = baseline.get("category_scores_0_100", {})
                appearance_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "target": target,
                        "head_scope": record.get("head_scope"),
                        "mask_mode": record.get("mask_mode"),
                        "impact": baseline.get("impact_score_0_100"),
                        "global_appearance": categories.get("global_appearance"),
                        "target_local": categories.get("target_local"),
                        "temporal_appearance": categories.get("temporal_appearance"),
                        "outside_spillover": categories.get("outside_spillover"),
                        "global_ssim": global_metrics.get("ssim_mean"),
                        "global_mae": global_metrics.get("mae_0_1"),
                        "global_delta_mae": global_metrics.get(
                            "temporal_delta_mae_0_1"
                        ),
                        "target_roi_mae": target_roi.get("mae_0_1"),
                        "target_roi_delta_mae": target_roi.get(
                            "temporal_delta_mae_0_1"
                        ),
                        "outside_mae": outside.get("mae_0_1"),
                        "outside_delta_mae": outside.get(
                            "temporal_delta_mae_0_1"
                        ),
                    }
                )
                case_row["appearance_records"] += 1
                case_row["appearance_seeds"].add(seed)
            trajectory = (record.get("trajectory_metrics") or {}).get(
                "metrics", {}
            )
            survival = (record.get("object_survival_metrics") or {}).get(
                "metrics", {}
            )
            if trajectory:
                case_row["trajectory_records"] += 1
                case_row["trajectory_seeds"].add(seed)
            if survival:
                case_row["survival_records"] += 1
                case_row["survival_seeds"].add(seed)
            if trajectory or survival:
                trajectory_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "target": target,
                        "head_scope": record.get("head_scope"),
                        "mask_mode": record.get("mask_mode"),
                        "trajectory_impact_percent_d0": trajectory.get(
                            "trajectory_impact_percent_d0"
                        ),
                        "track_loss": trajectory.get(
                            "target_worst_track_loss_score_0_100"
                        ),
                        "retention_failure": survival.get(
                            "target_worst_disappearance_score_0_100"
                        ),
                        "mask_absence": survival.get(
                            "target_worst_mask_absence_score_0_100"
                        ),
                    }
                )

    appearance_index = {
        (
            row["case"],
            row["seed"],
            row["target"],
            row["head_scope"],
            row["mask_mode"],
        ): row
        for row in appearance_rows
    }
    candidate_units = {
        (case, seed, target, head_scope)
        for case, seed, target, head_scope, _mode in appearance_index
    }
    matched_units = sorted(
        unit
        for unit in candidate_units
        if all((*unit, mode) in appearance_index for mode in mode_order)
    )
    appearance_by_mode = []
    for mode in mode_order:
        selected = [appearance_index[(*unit, mode)] for unit in matched_units]
        appearance_by_mode.append(
            {
                "mask_mode": mode,
                "matched_units": len(selected),
                "cases": len({row["case"] for row in selected}),
                "case_seeds": len(
                    {(row["case"], row["seed"]) for row in selected}
                ),
                "metrics": balanced_metrics(selected),
            }
        )

    scope_order = ("top100", "bottom100", "all720")
    scope_index = {
        (
            row["case"],
            row["seed"],
            row["target"],
            row["mask_mode"],
            row["head_scope"],
        ): row
        for row in appearance_rows
    }
    scope_candidates = {
        (case, seed, target, mode)
        for case, seed, target, mode, _scope in scope_index
    }
    matched_scope_units = sorted(
        unit
        for unit in scope_candidates
        if all((*unit, scope) in scope_index for scope in scope_order)
    )
    appearance_by_scope = []
    for scope in scope_order:
        selected = [scope_index[(*unit, scope)] for unit in matched_scope_units]
        appearance_by_scope.append(
            {
                "head_scope": scope,
                "matched_units": len(selected),
                "cases": len({row["case"] for row in selected}),
                "case_seeds": len(
                    {(row["case"], row["seed"]) for row in selected}
                ),
                "metrics": balanced_metrics(selected),
            }
        )

    trajectory_by_mode = []
    for mode in mode_order:
        selected = [row for row in trajectory_rows if row["mask_mode"] == mode]
        valid_ade = [
            row["trajectory_impact_percent_d0"]
            for row in selected
            if row.get("trajectory_impact_percent_d0") is not None
        ]
        trajectory_by_mode.append(
            {
                "mask_mode": mode,
                "records": len(selected),
                "cases": len({row["case"] for row in selected}),
                "case_seeds": len(
                    {(row["case"], row["seed"]) for row in selected}
                ),
                "ade_valid_records": len(valid_ade),
                "ade_pass_rate": (
                    round(len(valid_ade) / len(selected), 8) if selected else None
                ),
                "trajectory_impact_percent_d0": mean(valid_ade),
                "track_loss": mean(row.get("track_loss") for row in selected),
                "retention_failure": mean(
                    row.get("retention_failure") for row in selected
                ),
                "mask_absence": mean(
                    row.get("mask_absence") for row in selected
                ),
            }
        )

    trajectory_by_scope = []
    for scope in scope_order:
        selected = [
            row for row in trajectory_rows if row["head_scope"] == scope
        ]
        valid_ade = [
            row["trajectory_impact_percent_d0"]
            for row in selected
            if row.get("trajectory_impact_percent_d0") is not None
        ]
        trajectory_by_scope.append(
            {
                "head_scope": scope,
                "records": len(selected),
                "cases": len({row["case"] for row in selected}),
                "case_seeds": len(
                    {(row["case"], row["seed"]) for row in selected}
                ),
                "ade_valid_records": len(valid_ade),
                "ade_pass_rate": (
                    round(len(valid_ade) / len(selected), 8) if selected else None
                ),
                "trajectory_impact_percent_d0": mean(valid_ade),
                "track_loss": mean(row.get("track_loss") for row in selected),
                "retention_failure": mean(
                    row.get("retention_failure") for row in selected
                ),
                "mask_absence": mean(
                    row.get("mask_absence") for row in selected
                ),
            }
        )

    serializable_cases = []
    for case in sorted(case_rows):
        row = case_rows[case]
        serializable_cases.append(
            {
                **{
                    key: value
                    for key, value in row.items()
                    if not key.endswith("_seeds")
                },
                **{
                    key: sorted(value)
                    for key, value in row.items()
                    if key.endswith("_seeds")
                },
            }
        )
    return {
        "method": (
            "Only complete case×seed×target×head-scope cells containing all "
            "12 M1/M2/M3 time variants enter the ablation comparison. Means are "
            "hierarchical: targets/head scopes within seed, seeds within case, then "
            "equal-weight cases. Missing values are excluded, never replaced by zero."
        ),
        "configured": {
            "cases": len(case_rows),
            "case_seeds": len(samples),
            "expected_videos": sum(
                int(row.get("expected", 0)) for row in summaries
            ),
        },
        "coverage": {
            "generated_videos": sum(row["generated_videos"] for row in case_rows.values()),
            "appearance_records": len(appearance_rows),
            "appearance_cases": len({row["case"] for row in appearance_rows}),
            "appearance_case_seeds": len(
                {(row["case"], row["seed"]) for row in appearance_rows}
            ),
            "trajectory_records": sum(
                row["trajectory_records"] for row in case_rows.values()
            ),
            "trajectory_cases": len(
                {row["case"] for row in trajectory_rows if row.get("track_loss") is not None}
            ),
            "trajectory_case_seeds": len(
                {
                    (row["case"], row["seed"])
                    for row in trajectory_rows
                    if row.get("track_loss") is not None
                }
            ),
            "survival_records": sum(
                row["survival_records"] for row in case_rows.values()
            ),
            "matched_appearance_units": len(matched_units),
            "matched_appearance_cases": len({unit[0] for unit in matched_units}),
            "matched_appearance_case_seeds": len(
                {(unit[0], unit[1]) for unit in matched_units}
            ),
        },
        "cases": serializable_cases,
        "appearance_by_mode": appearance_by_mode,
        "appearance_by_scope": appearance_by_scope,
        "trajectory_by_mode": trajectory_by_mode,
        "trajectory_by_scope": trajectory_by_scope,
    }


def wan22_ti2v_legacy_m123_temporal_batch_catalog(
    selected_case: str | None = None,
    selected_seed: int | None = None,
    selected_only: bool = False,
):
    """Small catalog for the M1/M2/M3 All-time/Same/Future/Past video gallery.

    The full PhysicIQ67 catalog embeds every metric report for every sample and
    is intentionally not reused here.  This endpoint scans completion markers
    for the batch, then loads strict records and metrics for only the selected
    case/seed.
    """
    manifest = wan22_ti2v_legacy_physiciq67_visual_manifest()
    if manifest.get("ready") is False:
        return manifest
    samples = [
        dict(row)
        for row in manifest.get("samples", [])
        if row.get("m123_head_ranking_path")
        or (
            str(row.get("case"))
            == WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE
            and int(row.get("seed", -1)) == 47326
        )
    ]
    samples.sort(
        key=lambda row: (
            0
            if (
                str(row.get("case"))
                == WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE
                and int(row.get("seed", -1)) == 47326
            )
            else 1,
            str(row.get("case")),
            int(row.get("seed", -1)),
        )
    )
    if not samples:
        return {"ready": False, "reason": "M1/M2/M3 batch is empty", "samples": []}

    selected_key = (str(selected_case or ""), int(selected_seed or -1))
    available_keys = {
        (str(row["case"]), int(row["seed"])) for row in samples
    }
    if selected_key not in available_keys:
        selected_key = (str(samples[0]["case"]), int(samples[0]["seed"]))
    summary_samples = (
        [
            row
            for row in samples
            if (str(row["case"]), int(row["seed"])) == selected_key
        ]
        if selected_only
        else samples
    )

    summaries = []
    total_ready = total_expected = total_errors = 0
    for sample in summary_samples:
        case, seed = str(sample["case"]), int(sample["seed"])
        ranking_tag = str(sample.get("m123_head_ranking_tag") or "")
        head_scopes = tuple(
            str(value)
            for value in sample.get(
                "m123_head_scopes", WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPES
            )
        )
        regions = [
            str(row["region_name"])
            for row in sample.get("regions", [])
            if row.get("region_type") == "object"
        ]
        targets = _wan22_ti2v_legacy_physiciq67_m123_targets(regions)
        ready = errors = 0
        for head_scope in head_scopes:
            for target_scope, region in targets:
                for mask_mode in WAN22_TI2V_LEGACY_PHYSICIQ67_M123_GALLERY_MASKS:
                    variant = _wan22_ti2v_legacy_physiciq67_head_scope_variant(
                        target_scope, region, mask_mode, head_scope, ranking_tag
                    )
                    root = (
                        WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_ROOT
                        / case
                        / f"seed_{seed:05d}"
                        / variant
                    )
                    ready += int(
                        all(
                            (root / name).is_file()
                            for name in ("complete.json", "manifest.json", "generated.mp4")
                        )
                    )
                    errors += int((root / "error.txt").is_file())
        expected = len(head_scopes) * len(targets) * len(
            WAN22_TI2V_LEGACY_PHYSICIQ67_M123_GALLERY_MASKS
        )
        total_ready += ready
        total_expected += expected
        total_errors += errors
        summaries.append(
            {
                "case": case,
                "seed": seed,
                "category": sample.get("category", ""),
                "regions": regions,
                "head_scopes": list(head_scopes),
                "ready": ready,
                "expected": expected,
                "errors": errors,
                "complete": bool(expected and ready == expected),
            }
        )

    selected_sample = next(
        row
        for row in samples
        if (str(row["case"]), int(row["seed"])) == selected_key
    )
    full_selected_payload = _wan22_ti2v_legacy_physiciq67_m123_head_scope_records(
        selected_sample
    )
    primary_tag = str(full_selected_payload.get("ranking_tag") or "")
    payload_groups = [
        (
            (
                f"S039 latest2735 · {primary_tag}"
                if primary_tag == "s039r2735"
                else "原快照 · frozen134"
            ),
            full_selected_payload,
        )
    ]
    if selected_key == (WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE, 47326):
        comparison_sample = dict(selected_sample)
        comparison_sample.update(
            {
                "m123_head_ranking_path": str(
                    WAN22_TI2V_LEGACY_PHYSICIQ67_LATEST3350_HEAD_RANKING
                ),
                "m123_head_ranking_tag": "s039r3350",
                "m123_head_scopes": ["top100", "bottom100"],
            }
        )
        payload_groups.append(
            (
                "新快照 · latest3350",
                _wan22_ti2v_legacy_physiciq67_m123_head_scope_records(
                    comparison_sample
                ),
            )
        )
    m123_records = []
    ranking_snapshots = []
    for ranking_label, group_payload in payload_groups:
        group_records = [
            row
            for row in group_payload.get("records", [])
            if row.get("mask_mode")
            in WAN22_TI2V_LEGACY_PHYSICIQ67_M123_GALLERY_MASKS
        ]
        ranking_snapshots.append(
            {
                "ranking_tag": str(group_payload.get("ranking_tag") or ""),
                "ranking_label": ranking_label,
                "completed_runs_at_selection": group_payload.get(
                    "completed_runs_at_selection"
                ),
                "head_scopes": group_payload.get("head_scopes", []),
                "ready": sum(int(bool(row.get("ready"))) for row in group_records),
                "expected": len(group_records),
            }
        )
        for row in group_records:
            tagged_row = dict(row)
            tagged_row["ranking_tag"] = str(
                group_payload.get("ranking_tag") or ""
            )
            tagged_row["ranking_label"] = ranking_label
            m123_records.append(tagged_row)
    compact_records = []
    for row in m123_records:
        baseline_row = row.get("baseline_metrics") or {}
        baseline = baseline_row.get("metrics", {})
        global_metrics = baseline.get("global", {})
        target_roi = baseline.get("target_roi", {})
        outside_objects = baseline.get("outside_objects", {})
        trajectory_row = row.get("trajectory_metrics") or {}
        trajectory = trajectory_row.get("metrics", {})
        trajectory_objects = {
            name: {
                key: values.get(key)
                for key in (
                    "quality_pass",
                    "baseline_center_valid_frames",
                    "common_center_valid_frames",
                    "common_center_coverage",
                    "track_retention_score_0_100",
                    "track_loss_score_0_100",
                    "last_common_visible_frame",
                    "center_ade_px",
                    "center_ade_norm",
                    "center_fde_px",
                    "center_fde_norm",
                    "velocity_vector_error_px_per_frame",
                    "velocity_vector_error_norm_per_frame",
                    "velocity_valid_count",
                    "point_ade_px",
                    "point_ade_norm",
                    "point_valid_count",
                    "pck_normalized",
                )
            }
            for name, values in trajectory.get("objects", {}).items()
        }
        survival_row = row.get("object_survival_metrics") or {}
        survival = survival_row.get("metrics", {})
        survival_objects = {
            name: {
                key: values.get(key)
                for key in (
                    "quality_pass",
                    "f00_prompt_iou",
                    "survival_rate",
                    "retention_score_0_100",
                    "disappearance_score_0_100",
                    "identity_similarity_mean",
                    "identity_failure_rate",
                    "area_failure_rate",
                    "empty_mask_rate",
                    "first_sustained_loss_frame",
                    "terminal_missing_rate",
                    "alive_frame_count",
                    "frame_count",
                )
            }
            for name, values in survival.get("objects", {}).items()
        }
        compact_records.append(
            {
                key: row.get(key)
                for key in (
                    "target_scope",
                    "region",
                    "mask_mode",
                    "head_scope",
                    "head_count",
                    "variant_id",
                    "ranking_tag",
                    "ranking_label",
                    "ready",
                    "error",
                    "s039_top100_mean",
                )
            }
            | {
                "metrics": {
                    "impact_score_0_100": baseline.get("impact_score_0_100"),
                    "trajectory_impact_percent_d0": trajectory.get(
                        "trajectory_impact_percent_d0"
                    ),
                    "trajectory_rank": trajectory_row.get(
                        "trajectory_rank_within_case_seed"
                    ),
                    "track_loss_score_0_100": trajectory.get(
                        "target_worst_track_loss_score_0_100"
                    ),
                    "track_loss_rank": trajectory_row.get(
                        "track_loss_rank_within_case_seed"
                    ),
                    "retention_failure_score_0_100": survival.get(
                        "target_worst_disappearance_score_0_100"
                    ),
                    "retention_failure_rank": survival_row.get(
                        "disappearance_rank_within_case_seed"
                    ),
                    "mask_absence_score_0_100": survival.get(
                        "target_worst_mask_absence_score_0_100"
                    ),
                    "mask_absence_rank": survival_row.get(
                        "mask_absence_rank_within_case_seed"
                    ),
                    "appearance": {
                        "impact_score_0_100": baseline.get("impact_score_0_100"),
                        "impact_rank": baseline_row.get(
                            "impact_rank_within_case_seed"
                        ),
                        "spillover_score_0_100": baseline.get(
                            "spillover_score_0_100"
                        ),
                        "category_scores_0_100": baseline.get(
                            "category_scores_0_100", {}
                        ),
                        "category_ranks": baseline_row.get(
                            "category_ranks_within_case_seed", {}
                        ),
                        "global_ssim": global_metrics.get("ssim_mean"),
                        "global_psnr_db": global_metrics.get("psnr_db"),
                        "global_mae_0_1": global_metrics.get("mae_0_1"),
                        "global_temporal_delta_mae_0_1": global_metrics.get(
                            "temporal_delta_mae_0_1"
                        ),
                        "target_roi_key": baseline_row.get("target_roi_key"),
                        "target_roi_mae_0_1": target_roi.get("mae_0_1"),
                        "target_roi_temporal_delta_mae_0_1": target_roi.get(
                            "temporal_delta_mae_0_1"
                        ),
                        "target_roi_mean_area_fraction": target_roi.get(
                            "mean_area_fraction"
                        ),
                        "outside_objects_mae_0_1": outside_objects.get(
                            "mae_0_1"
                        ),
                        "outside_objects_temporal_delta_mae_0_1": (
                            outside_objects.get("temporal_delta_mae_0_1")
                        ),
                        "outside_objects_mean_area_fraction": outside_objects.get(
                            "mean_area_fraction"
                        ),
                    },
                    "trajectory": {
                        "quality_pass": trajectory.get("quality_pass"),
                        "selected_objects": trajectory.get("selected_objects", []),
                        "target_center_ade_norm": trajectory.get(
                            "target_center_ade_norm"
                        ),
                        "trajectory_impact_percent_d0": trajectory.get(
                            "trajectory_impact_percent_d0"
                        ),
                        "trajectory_rank": trajectory_row.get(
                            "trajectory_rank_within_case_seed"
                        ),
                        "target_mean_track_loss_score_0_100": trajectory.get(
                            "target_mean_track_loss_score_0_100"
                        ),
                        "target_worst_track_loss_score_0_100": trajectory.get(
                            "target_worst_track_loss_score_0_100"
                        ),
                        "track_loss_rank": trajectory_row.get(
                            "track_loss_rank_within_case_seed"
                        ),
                        "objects": trajectory_objects,
                    },
                    "survival": {
                        "quality_pass": survival.get("quality_pass"),
                        "selected_objects": survival.get("selected_objects", []),
                        "target_mean_disappearance_score_0_100": survival.get(
                            "target_mean_disappearance_score_0_100"
                        ),
                        "target_worst_disappearance_score_0_100": survival.get(
                            "target_worst_disappearance_score_0_100"
                        ),
                        "disappearance_rank": survival_row.get(
                            "disappearance_rank_within_case_seed"
                        ),
                        "target_mean_mask_absence_score_0_100": survival.get(
                            "target_mean_mask_absence_score_0_100"
                        ),
                        "target_worst_mask_absence_score_0_100": survival.get(
                            "target_worst_mask_absence_score_0_100"
                        ),
                        "mask_absence_rank": survival_row.get(
                            "mask_absence_rank_within_case_seed"
                        ),
                        "objects": survival_objects,
                    },
                }
            }
        )
    selected_payload = {
            "case": selected_key[0],
            "seed": selected_key[1],
            "category": selected_sample.get("category", ""),
            "regions": [
                str(row["region_name"])
                for row in selected_sample.get("regions", [])
                if row.get("region_type") == "object"
            ],
            "baseline_ready": bool(
                wan22_ti2v_legacy_physiciq67_visual_video(
                    selected_key[0], str(selected_key[1])
                )
            ),
            "ranking_snapshots": ranking_snapshots,
            "records": compact_records,
        }
    response = {
        "ready": True,
        "samples": summaries,
        "selected": selected_payload,
        "progress": {
            "ready": total_ready,
            "expected": total_expected,
            "errors": total_errors,
            "complete_samples": sum(int(row["complete"]) for row in summaries),
            "sample_count": len(summaries),
            "case_count": len({row["case"] for row in summaries}),
        },
    }
    if not selected_only:
        response["aggregate"] = _wan22_ti2v_legacy_m123_batch_metric_aggregate(
            samples, summaries
        )
    return response


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
    head_scope: str = "top100",
    ranking_tag: str = "",
):
    sample = _wan22_ti2v_legacy_physiciq67_sample(case, seed)
    if sample is None:
        return None
    try:
        seed_value, count = int(seed), int(top_n)
    except ValueError:
        return None
    configured_scopes = tuple(
        str(value) for value in sample.get("m123_head_scopes", ())
    )
    configured_ranking_tag = str(sample.get("m123_head_ranking_tag") or "")
    is_pilot = (
        case == WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_CASE
        and seed_value in WAN22_TI2V_LEGACY_PHYSICIQ67_TEMPORAL_TUBE_SEEDS
    )
    allowed_scopes = configured_scopes or (("top100",) if is_pilot else ())
    if is_pilot and seed_value == 47326:
        allowed_scopes = WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPES
    allowed_ranking_tags = {configured_ranking_tag}
    if is_pilot and seed_value == 47326:
        allowed_ranking_tags.add("s039r3350")
    if ranking_tag not in allowed_ranking_tags:
        return None
    if ranking_tag == "s039r3350":
        allowed_scopes = ("top100", "bottom100")
    expected_count = (
        {"top100": 100, "bottom100": 100, "all720": 720}.get(head_scope)
        if head_scope in allowed_scopes
        else None
    )
    if (
        expected_count is None
        or count != expected_count
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
    if is_pilot and head_scope == "top100" and not ranking_tag:
        variant = _wan22_ti2v_legacy_physiciq67_ablation_variant(
            target_scope, mask_mode, count, region
        )
    else:
        variant = _wan22_ti2v_legacy_physiciq67_head_scope_variant(
            target_scope, region, mask_mode, head_scope, ranking_tag
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


def wan22_ti2v_legacy_physiciq67_head_scope_trajectory_overlay(
    case: str, seed: str, variant_id: str
):
    sample = _wan22_ti2v_legacy_physiciq67_sample(case, seed)
    if sample is None or Path(variant_id).name != variant_id or not variant_id:
        return None
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    root = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPE_TRAJECTORY_ROOT
        / case
        / f"seed_{seed_value:05d}"
    )
    report_path = root / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        allowed = {
            str(row["variant_id"])
            for row in report.get("records", [])
            if isinstance(row, dict) and row.get("variant_id")
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if variant_id not in allowed:
        return None
    return root / "overlays" / f"{variant_id}.mp4"


def wan22_ti2v_legacy_physiciq67_m123_s039_top100_mean_overlay(
    case: str, seed: str, variant_id: str, region: str
):
    if region not in {"object_A", "object_B"}:
        return None
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    record = _wan22_ti2v_legacy_physiciq67_m123_s039_top100_mean_record(
        case, seed_value, variant_id
    )
    if not record.get("ready"):
        return None
    region_row = next(
        (row for row in record.get("records", []) if row.get("region_name") == region),
        None,
    )
    if region_row is None:
        return None
    return (
        WAN22_TI2V_LEGACY_PHYSICIQ67_M123_S039_TOP100_MEAN_ROOT
        / case
        / f"seed_{seed_value:05d}"
        / variant_id
        / str(region_row["image"])
    )


def wan22_ti2v_legacy_physiciq67_m123_s039_query_receiver_overlay(
    case: str, seed: str, variant_id: str
):
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    record = _wan22_ti2v_legacy_physiciq67_m123_s039_query_receiver_record(
        case, seed_value, variant_id
    )
    if not record.get("ready"):
        return None
    return (
        WAN22_TI2V_LEGACY_PHYSICIQ67_M123_S039_QUERY_RECEIVER_ROOT
        / case
        / f"seed_{seed_value:05d}"
        / variant_id
        / str(record["image"])
    )


def wan22_ti2v_legacy_physiciq67_head_scope_survival_overlay(
    case: str, seed: str, variant_id: str
):
    sample = _wan22_ti2v_legacy_physiciq67_sample(case, seed)
    if sample is None or Path(variant_id).name != variant_id or not variant_id:
        return None
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    root = (
        WAN22_TI2V_LEGACY_PHYSICIQ67_HEAD_SCOPE_TRAJECTORY_ROOT
        / case
        / f"seed_{seed_value:05d}"
    )
    report_path = root / "object_survival_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        allowed = {
            str(row["variant_id"])
            for row in report.get("records", [])
            if isinstance(row, dict) and row.get("variant_id")
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if variant_id not in allowed:
        return None
    return root / "object_survival" / "overlays" / f"{variant_id}.mp4"


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

    directional_defs_anchor = "full_head_output:{id:'C3',matrix:'Y_h=A_hV_h=0',flow:'该 head ──X──> 后续输出',exact:'直接令选中 head 的完整输出为 0；Q/K/V 与 softmax A 本身不变'}},logic={"
    directional_defs_insert = """full_head_output:{id:'C3',matrix:'Y_h=A_hV_h=0',flow:'该 head ──X──> 后续输出',exact:'直接令选中 head 的完整输出为 0；Q/K/V 与 softmax A 本身不变'},self_same:{id:'M1-same',matrix:'A[R_tq,R_tk]=0 · tq=tk',flow:'R_t K/V ──X──> R_t Query',exact:'删除同一 latent 时刻内全部 R_t Query×R_t K/V 配对；不是只删除 q=k 主对角线，全部跨时刻 R→R 保留'},self_future:{id:'M1-future',matrix:'A[R_tq,R_tk]=0 · tq>tk',flow:'R_t K/V ──X──> R_t′ Query · t′>t',exact:'只删除 object tube 内从较早 key 时刻流向较晚 query 时刻的 post-softmax A@V 项；同帧与反向时间项保留'},self_past:{id:'M1-past',matrix:'A[R_tq,R_tk]=0 · tq<tk',flow:'R_t K/V ──X──> R_t′ Query · t′<t',exact:'只删除 object tube 内从较晚 key 时刻流向较早 query 时刻的 post-softmax A@V 项；同帧与正向时间项保留'},incoming_same:{id:'M2-same',matrix:'A[R_tq,C_tk]=0 · tq=tk',flow:'C_t K/V ──X──> R_t Query',exact:'删除同一 latent 时刻内全部 C_t Value 对 R_t Query 的贡献；全部跨时刻 C→R 保留'},incoming_future:{id:'M2-future',matrix:'A[R_tq,C_tk]=0 · tq>tk',flow:'C_t K/V ──X──> R_t′ Query · t′>t',exact:'只删除较早时刻的非 R Value 对较晚 object-query 的贡献；R→R 与同帧/反向 C→R 保留'},incoming_past:{id:'M2-past',matrix:'A[R_tq,C_tk]=0 · tq<tk',flow:'C_t K/V ──X──> R_t′ Query · t′<t',exact:'只删除较晚时刻的非 R Value 对较早 object-query 的贡献；R→R 与同帧/正向 C→R 保留'},outgoing_same:{id:'M3-same',matrix:'A[C_tq,R_tk]=0 · tq=tk',flow:'R_t K/V ──X──> C_t Query',exact:'删除同一 latent 时刻内全部 R_t Value 对 C_t Query 的贡献；R Query 更新与全部跨时刻 R→C 保留'},outgoing_future:{id:'M3-future',matrix:'A[C_tq,R_tk]=0 · tq>tk',flow:'R_t K/V ──X──> C_t′ Query · t′>t',exact:'只删除较早 object Value 对较晚非 R query 的贡献；R query 的读取与同帧/反向 R→C 保留'},outgoing_past:{id:'M3-past',matrix:'A[C_tq,R_tk]=0 · tq<tk',flow:'R_t K/V ──X──> C_t′ Query · t′<t',exact:'只删除较晚 object Value 对较早非 R query 的贡献；R query 的读取与同帧/正向 R→C 保留'}},logic={"""
    if page.count(directional_defs_anchor) != 1:
        raise RuntimeError("PhysicIQ67 temporal directional definitions changed")
    page = page.replace(directional_defs_anchor, directional_defs_insert, 1)

    directional_logic_anchor = "full_head_output:{calc:'Y_h=A_hV_h=0',theory:'直接删除该 head 对残差分支的全部贡献；它不依赖 object token 集 R。'}},modes="
    directional_logic_insert = """full_head_output:{calc:'Y_h=A_hV_h=0',theory:'直接删除该 head 对残差分支的全部贡献；它不依赖 object token 集 R。'},self_same:{calc:"Y'_{R,tq}=Y_{R,tq}−Σ_{tk=tq} A[R_tq,R_tk]V_{R,tk}",theory:'隔离对象 tube 内的同帧自交互，保留过去→未来与未来→过去的全部 R→R 跨帧项。'},self_future:{calc:"Y'_{R,tq}=Y_{R,tq}−Σ_{tk<tq} A[R_tq,R_tk]V_{R,tk}",theory:'隔离 object tube 内由过去对象状态写入未来对象 query 的前向时间贡献；不删除同帧自作用，也不删除未来→过去项。'},self_past:{calc:"Y'_{R,tq}=Y_{R,tq}−Σ_{tk>tq} A[R_tq,R_tk]V_{R,tk}",theory:'隔离 object tube 内未来对象状态写回过去 object query 的反向时间贡献；用于和 M1-future 检查时间方向不对称。'},incoming_same:{calc:"Y'_{R,tq}=Y_{R,tq}−Σ_{tk=tq} A[R_tq,C_tk]V_{C,tk}",theory:'检验对象 Query 的同帧环境输入；全部跨帧 C→R 输入及 R→R 自作用保留。'},incoming_future:{calc:"Y'_{R,tq}=Y_{R,tq}−Σ_{tk<tq} A[R_tq,C_tk]V_{C,tk}",theory:'检验过去环境/其他 token 是否把状态传入未来 object query；同帧上下文和 object 自身 tube 仍保留。'},incoming_past:{calc:"Y'_{R,tq}=Y_{R,tq}−Σ_{tk>tq} A[R_tq,C_tk]V_{C,tk}",theory:'检验较晚环境 token 对较早 object query 的反向时间读取效应；与 future 差异反映方向性而非总剂量。'},outgoing_same:{calc:"Y'_{C,tq}=Y_{C,tq}−Σ_{tk=tq} A[C_tq,R_tk]V_{R,tk}",theory:'检验对象 Value 对同帧环境/其他对象的广播；object Query 更新和跨帧 R→C 全部保留。'},outgoing_future:{calc:"Y'_{C,tq}=Y_{C,tq}−Σ_{tk<tq} A[C_tq,R_tk]V_{R,tk}",theory:'检验过去对象状态向未来环境/其他对象传播的可见作用；object query 自身更新完全保留。'},outgoing_past:{calc:"Y'_{C,tq}=Y_{C,tq}−Σ_{tk>tq} A[C_tq,R_tk]V_{R,tk}",theory:'检验较晚对象状态向较早非 R query 的反向传播；与 M3-future 对照时间方向性。'}},modes="""
    if page.count(directional_logic_anchor) != 1:
        raise RuntimeError("PhysicIQ67 temporal directional logic changed")
    page = page.replace(directional_logic_anchor, directional_logic_insert, 1)

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
    temporal_rows = """rows=(()=>{const directionalAfter={self_only:['self_same','self_future','self_past'],incoming_only:['incoming_same','incoming_future','incoming_past'],outgoing_only:['outgoing_same','outgoing_future','outgoing_past']},objectRows=targets.map(([targetScope,targetRegion,targetLabel])=>{const group=specs.filter(x=>x.target_scope===targetScope&&x.region===targetRegion),tubeGroup=group.flatMap(x=>[x,...(directionalAfter[x.mask_mode]||[]).map(mask_mode=>({...x,mask_mode}))]),fixedCards=group.map(x=>card('fixed',x,find(fixed,x))).filter(Boolean),tubeCards=tubeGroup.map(x=>card('tube',x,find(tube,x))).filter(Boolean),readyCount=fixedCards.length+tubeCards.length;if(!readyCount)return '';const protocolRow=(kind,cards)=>cards.length?`<div class="object-protocol-row ${kind}"><div class="object-protocol-heading"><strong>${kind==='fixed'?'Fixed · R_fixed · 仅 latent t=0':'Tube · R_tube · latent t=0…12 联合集合'}</strong><span>${cards.length} 个已生成视频 · ${kind==='fixed'?'M1→C1':'M1/Base→Same→Future→Past · M2/Base→Same→Future→Past · M3/Base→Same→Future→Past · M4→C1'}</span></div><div class="object-ablation-strip">${cards.join('')}</div></div>`:'';return `<section class="object-ablation-row"><div class="object-ablation-heading"><h3>${e(targetLabel)}</h3><span>${readyCount} 个已生成 Top100 消融</span></div>${protocolRow('fixed',fixedCards)}${protocolRow('tube',tubeCards)}</section>`}).join(''),controlSpecs=controlModes.map(mask_mode=>({target_scope:'all_tokens',region:'',targetLabel:'Global all-token controls',mask_mode})),controlCards=controlSpecs.map(x=>card('fixed',x,find(fixed,x))).filter(Boolean),controlRow=controlCards.length?`<section class="object-ablation-row control-row"><div class="object-ablation-heading"><h3>Global all-token controls</h3><span>${controlCards.length} 个已生成 Top100 控制 · C2/C3 不依赖 R，无 Tube 重复项</span></div><div class="object-ablation-strip">${controlCards.join('')}</div></section>`:'';return objectRows+controlRow})()"""
    page = page[:temporal_rows_start] + temporal_rows + page[temporal_rows_end:]

    temporal_progress_anchor = ",done=tube.filter(r=>r.ready).length,baseline="
    temporal_progress_with_fixed = ",fixedDone=fixed.filter(r=>r.ready&&r.target_scope!=='all_tokens'&&modes.includes(r.mask_mode)).length,baseTubeDone=tube.filter(r=>r.ready&&!r.temporal_directional).length,baseTubeExpected=tube.filter(r=>!r.temporal_directional).length,directionalDone=tube.filter(r=>r.ready&&r.temporal_directional).length,directionalExpected=tube.filter(r=>r.temporal_directional).length,done=tube.filter(r=>r.ready).length,baseline="
    if page.count(temporal_progress_anchor) != 1:
        raise RuntimeError("PhysicIQ67 temporal progress changed")
    page = page.replace(temporal_progress_anchor, temporal_progress_with_fixed, 1)

    temporal_column_header = '<div class="tube-compare-row"><div class="tube-column-head">算子 ID / 被切断的信息流</div><div class="tube-column-head">左：R_fixed · 仅 latent t=0</div><div class="tube-column-head">右：R_tube · latent t=0…12 联合集合</div></div>'
    if page.count(temporal_column_header) != 1:
        raise RuntimeError("PhysicIQ67 temporal column header changed")
    page = page.replace(
        temporal_column_header,
        '<div class="object-layout-note">每个 object/target 分成两条独立横向视频行。Tube 行在每个基础算子旁按 <b>M1 → M1-same → M1-future → M1-past</b>、<b>M2 → M2-same → M2-future → M2-past</b>、<b>M3 → M3-same → M3-future → M3-past</b> 排列，随后为 M4→C1。Same 表示同一 latent 帧的全部配对，不是仅 q=k 主对角线。只显示已经生成的 Top100 视频，未生成项不占位；27/27 完成后页面会自动重新读取 catalog。</div>',
        1,
    )
    page = page.replace(
        "页面展示 object_A、object_B、all_objects 三组；R-dependent 生成进度：Fixed <b>${fixedDone}/24</b> · Tube <b>${done}/${tube.length}</b>。C2/C3 不依赖 R，不重复生成 Tube，并在 Global all-token controls 行展示已有 Fixed 控制视频。",
        "页面展示 object_A、object_B、all_objects 三组；基础消融进度：Fixed <b>${fixedDone}/24</b> · Tube <b>${baseTubeDone}/${baseTubeExpected}</b>；时间分解对照：<b>${directionalDone}/${directionalExpected}</b>。Same/Future/Past 分别删除 tq=tk、tq>tk、tq<tk，三者互斥且并集等于对应完整 M1/M2/M3 块。C2/C3 不依赖 R，不重复生成 Tube。",
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
    auto_refresh_anchor = "$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load();\n</script>"
    auto_refresh_insert = """$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));let directionalPoll=null;const temporalProgress=catalog=>{const samples=(catalog.samples||[]).filter(x=>x.case==='0613pybullet_sample_001460_w002'&&Array.isArray(x.temporal_tube_attention_matrix_ablations)),perSeed=samples.map(x=>{const rows=x.temporal_tube_attention_matrix_ablations.filter(r=>r.temporal_directional),ready=rows.filter(r=>r.ready).length;return {seed:x.seed,ready,expected:rows.length,complete:rows.length>0&&ready===rows.length}});return {ready:perSeed.reduce((n,x)=>n+x.ready,0),expected:perSeed.reduce((n,x)=>n+x.expected,0),completeSeeds:perSeed.filter(x=>x.complete).length,seedCount:perSeed.length}};async function pollDirectionalCompletion(){if(!data)return;const old=temporalProgress(data);if(old.expected&&old.ready===old.expected){if(directionalPoll)clearInterval(directionalPoll);return}try{const fresh=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),next=temporalProgress(fresh);$('status').textContent=`Temporal M1/M2/M3 Same/Future/Past ${next.ready}/${next.expected} · seeds ${next.completeSeeds}/${next.seedCount} · 完成后自动刷新`;if(next.completeSeeds>old.completeSeeds||next.ready===next.expected){if(next.ready===next.expected&&directionalPoll)clearInterval(directionalPoll);await load();$('status').textContent+=` · 已自动发布 ${next.completeSeeds}/${next.seedCount} seeds`}}catch(err){console.warn('temporal auto-refresh failed',err)}}load().then(()=>{directionalPoll=setInterval(pollDirectionalCompletion,20000);pollDirectionalCompletion()});
</script>"""
    if page.count(auto_refresh_anchor) != 1:
        raise RuntimeError("PhysicIQ67 directional auto-refresh anchor changed")
    page = page.replace(auto_refresh_anchor, auto_refresh_insert, 1)

    head_scope_section_anchor = '<div id="ablation"></div></section>'
    if page.count(head_scope_section_anchor) != 1:
        raise RuntimeError("PhysicIQ67 ablation section anchor changed")
    page = page.replace(
        head_scope_section_anchor,
        '<div id="m123HeadScopePanel" class="m123-head-panel" hidden></div><div id="ablation"></div></section>',
        1,
    )

    head_scope_renderer = r'''function renderM123HeadScopes(s,base){
const panel=$('m123HeadScopePanel'),payload=s.m123_head_scope_ablations;
if(!payload?.ready||!Array.isArray(payload.records)){panel.hidden=true;panel.innerHTML='';return}
panel.hidden=false;
const records=payload.records,
scopeCatalog=[['top100','Top100 PCK Heads','S039 frozen ranks 1–100'],['bottom100','Bottom100 PCK Heads','S039 frozen ranks 621–720'],['all720','All Heads · 720 layer-heads','30 layers × 24 heads']],
availableScopes=new Set(records.map(r=>r.head_scope)),
scopes=scopeCatalog.filter(([scope])=>availableScopes.has(scope)),
ops=[{id:'M1',base:'self',flow:'R_tk K/V ──X──> R_tq Query',target:'R',source:'R',theory:'诊断对象 tube 内部 Value 对对象 Query 的自支持与跨时传播。'},{id:'M2',base:'incoming',flow:'C_tk K/V ──X──> R_tq Query',target:'R',source:'C',theory:'诊断环境、背景和其他对象 Value 向对象 Query 输入上下文。'},{id:'M3',base:'outgoing',flow:'R_tk K/V ──X──> C_tq Query',target:'C',source:'R',theory:'诊断对象 Value 向环境和其他对象 Query 的广播。'}],
times=[['only','All-time','所有 tk'],['same','Same','tk=tq'],['future','Future','tk<tq'],['past','Past','tk>tq']],
targets=[...new Set(records.map(r=>r.target_scope==='single_object'?`single_object::${r.region}`:'all_objects::'))].map(key=>{const [target_scope,region='']=key.split('::');return {target_scope,region,label:target_scope==='single_object'?`Selected object · ${region}`:'All objects union'}}),
find=(target,scope,mode)=>records.find(r=>r.target_scope===target.target_scope&&String(r.region||'')===target.region&&r.head_scope===scope&&r.mask_mode===mode),
formula=(op,predicate)=>`Y′_${op.target}(tq)=Y_${op.target}(tq)−Σ_{${predicate}} A[${op.target}_tq,${op.source}_tk]V_${op.source}(tk)`,
num=(value,digits=4)=>Number.isFinite(Number(value))?Number(value).toFixed(digits):'—',
modeLabel=mode=>{const op=mode.startsWith('self_')?'M1':mode.startsWith('incoming_')?'M2':'M3',time=mode.endsWith('_only')?'All-time':mode.endsWith('_same')?'Same':mode.endsWith('_future')?'Future':'Past';return `${op}-${time}`},
categoryOrder=['global_appearance','target_local','temporal_appearance','outside_spillover'],
categoryDefs=payload.category_definitions||{},
categoryName=id=>categoryDefs[id]?.name||id,
metricDetails=r=>{const row=r.baseline_metrics;if(!row)return `<details class="m123-metrics pending"><summary>vs Baseline 指标待计算</summary><p>视频已生成，但增量 bench 尚未写入该 variant。刷新后会自动出现。</p></details>`;const m=row.metrics,g=m.global,t=m.target_roi,o=m.outside_objects,c=m.category_scores_0_100||{},cr=row.category_ranks_within_case_seed||{},rank=row.impact_rank_within_case_seed,categoryRows=categoryOrder.map(id=>`<tr class="m123-category-score"><th>${e(categoryName(id))} ↑</th><td>${num(c[id])} · #${num(cr[id],0)}</td></tr>`).join('');return `<details class="m123-metrics"><summary>vs Baseline · 四类影响与排名 · 综合 ${num(m.impact_score_0_100)} (#${num(rank,0)})</summary><div class="m123-metric-body"><table><tbody><tr><th>综合影响分 ↑</th><td>${num(m.impact_score_0_100)} · #${num(rank,0)}</td></tr>${categoryRows}<tr class="m123-raw-start"><th>Global SSIM ↓</th><td>${num(g.ssim_mean)}</td></tr><tr><th>Global MAE ↑</th><td>${num(g.mae_0_1)}</td></tr><tr><th>Global Δ-MAE ↑</th><td>${num(g.temporal_delta_mae_0_1)}</td></tr><tr><th>${e(row.target_roi_key)} ROI MAE ↑</th><td>${num(t.mae_0_1)}</td></tr><tr><th>${e(row.target_roi_key)} ROI Δ-MAE ↑</th><td>${num(t.temporal_delta_mae_0_1)}</td></tr><tr><th>Outside-object MAE ↑</th><td>${num(o.mae_0_1)}</td></tr><tr><th>Outside-object Δ-MAE ↑</th><td>${num(o.temporal_delta_mae_0_1)}</td></tr></tbody></table><p>四类分数均只衡量相对同 seed Baseline 的可见干预强度，越大表示该类影响越强，不代表生成质量或物理正确性。“时序外观”依然是像素变化指标，不用于判断轨迹差异。</p></div></details>`},
trajectoryDetails=r=>{const row=r.trajectory_metrics;if(!row)return `<details class="m123-trajectory pending"><summary>真实轨迹指标待 CoTracker</summary><p>尚未对该视频跟踪；不使用 Δ-MAE 代替。</p></details>`;const m=row.metrics||{},objects=m.objects||{},rank=row.trajectory_rank_within_case_seed,trackRank=row.track_loss_rank_within_case_seed,objectRows=Object.entries(objects).map(([name,o])=>`<tr><th>${e(name)} Center-ADE / FDE</th><td>${num(o.center_ade_norm)} / ${num(o.center_fde_norm)}</td></tr><tr><th>${e(name)} Velocity vector error</th><td>${num(o.velocity_vector_error_norm_per_frame)}</td></tr><tr><th>${e(name)} PCK@5/10/20%</th><td>${num(o.pck_normalized?.['0.05'])} / ${num(o.pck_normalized?.['0.1'])} / ${num(o.pck_normalized?.['0.2'])}</td></tr><tr><th>${e(name)} common coverage / Track Loss</th><td>${num(o.common_center_coverage)} (${o.common_center_valid_frames}/${o.baseline_center_valid_frames}) / ${num(o.track_loss_score_0_100)} ${o.quality_pass?'✓':'ADE N/A'}</td></tr>`).join(''),score=m.trajectory_impact_percent_d0,trackLoss=m.target_worst_track_loss_score_0_100,overlay=`${api}/head-scope-trajectory-overlay?${base}&variant_id=${encodeURIComponent(r.variant_id)}`;return `<details class="m123-trajectory"><summary>轨迹/可跟踪性 vs Baseline · ADE ${score==null?'N/A':`${num(score)}% D0 (#${num(rank,0)})`} · Track Loss ${num(trackLoss)} (#${num(trackRank,0)})</summary><div class="m123-trajectory-body"><table><tbody><tr><th>Worst Track Loss /100 ↑</th><td>${num(trackLoss)} · #${num(trackRank,0)}</td></tr><tr><th>Mean Track Loss /100 ↑</th><td>${num(m.target_mean_track_loss_score_0_100)}</td></tr><tr><th>轨迹影响（% D0）↑</th><td>${num(score)}</td></tr><tr><th>Target Center-ADE norm ↑</th><td>${num(m.target_center_ade_norm)}</td></tr>${objectRows}</tbody></table><video controls muted playsinline preload="none" data-src="${overlay}"></video><p>Track Loss = 100 × (1−共同中心覆盖率)，全部视频都可比较，但只表示 CoTracker 可观测性损失。ADE 覆盖质量门未通过时仍保持 N/A，绝不以 Track Loss 填充。</p></div></details>`},
survivalDetails=r=>{const row=r.object_survival_metrics;if(!row)return `<details class="m123-survival pending"><summary>对象存活指标待 SAM2+DINOv2</summary><p>模型指标正在增量计算；Track Loss 不能替代对象存活判断。</p></details>`;const m=row.metrics||{},objects=m.objects||{},rank=row.disappearance_rank_within_case_seed,maskRank=row.mask_absence_rank_within_case_seed,objectRows=Object.entries(objects).map(([name,o])=>`<tr><th>${e(name)} Retention / Failure (/100)</th><td>${num(o.retention_score_0_100)} / ${num(o.disappearance_score_0_100)}</td></tr><tr><th>${e(name)} DINO cosine / identity-fail / area-fail</th><td>${num(o.identity_similarity_mean)} / ${num(o.identity_failure_rate)} / ${num(o.area_failure_rate)}</td></tr><tr><th>${e(name)} empty-mask / terminal-missing rate</th><td>${num(o.empty_mask_rate)} / ${num(o.terminal_missing_rate)}</td></tr><tr><th>${e(name)} first 3-frame sustained loss</th><td>${o.first_sustained_loss_frame==null?'—':`F${o.first_sustained_loss_frame}`}</td></tr>`).join(''),score=m.target_worst_disappearance_score_0_100,maskAbsence=m.target_worst_mask_absence_score_0_100,overlay=`${api}/head-scope-survival-overlay?${base}&variant_id=${encodeURIComponent(r.variant_id)}`;return `<details class="m123-survival"><summary>对象保留 vs Baseline · Retention Failure ${num(score)} (#${num(rank,0)}) · Mask Absence ${num(maskAbsence)} (#${num(maskRank,0)})</summary><div class="m123-survival-body"><table><tbody><tr><th>Worst retention failure /100 ↑</th><td>${num(score)} · #${num(rank,0)}</td></tr><tr><th>Worst SAM2 mask absence /100 ↑</th><td>${num(maskAbsence)} · #${num(maskRank,0)}</td></tr><tr><th>Mean target retention failure /100 ↑</th><td>${num(m.target_mean_disappearance_score_0_100)}</td></tr><tr><th>Mean target mask absence /100 ↑</th><td>${num(m.target_mean_mask_absence_score_0_100)}</td></tr>${objectRows}</tbody></table><video controls muted playsinline preload="none" data-src="${overlay}"></video><p>Retention Failure 是 mask、DINO identity、面积条件的 composite；Mask Absence = 100 × empty-mask rate，更接近纯消失，但仍可能包含 SAM2 跟踪失败。failure rate 为 0–1；绿色=保留，红色=至少一个条件失败。三种失败原因保留为独立字段，必须结合 overlay 审计。</p></div></details>`},
card=(target,scope,scopeLabel,scopeDetail,op,time)=>{const suffix=time[0]==='only'?'only':time[0],mode=`${op.base}_${suffix}`,r=find(target,scope,mode);if(!r?.ready)return '';const src=`${api}/temporal-tube-ablation-video?${base}&target_scope=${encodeURIComponent(target.target_scope)}&mask_mode=${encodeURIComponent(mode)}&top_n=${r.head_count}&head_scope=${encodeURIComponent(scope)}${target.region?`&region=${encodeURIComponent(target.region)}`:''}`;return `<figure><video controls muted playsinline preload="none" src="${src}"></video><figcaption><strong>${e(op.id)}-${e(time[1])} · ${e(scopeLabel)}</strong><span class="caption-protocol">${e(scopeDetail)} · ${r.head_count} layer-heads</span><span class="caption-flow"><b>切断：</b>${e(op.flow)} · ${e(time[2])}</span><span class="caption-exact"><b>精确计算：</b>${e(formula(op,time[2]))}</span><span class="caption-exact"><b>理论诊断：</b>${e(op.theory)}</span></figcaption>${trajectoryDetails(r)}${survivalDetails(r)}${metricDetails(r)}</figure>`},
targetSections=targets.map(target=>{const opSections=ops.map(op=>{const scopeRows=scopes.map(([scope,scopeLabel,scopeDetail])=>{const cards=times.map(time=>card(target,scope,scopeLabel,scopeDetail,op,time)).filter(Boolean);if(!cards.length)return '';return `<div class="m123-scope-row"><div class="m123-scope-heading"><strong>${e(scopeLabel)}</strong><span>${e(scopeDetail)} · ${cards.length}/4 已生成</span></div><div class="m123-video-grid">${cards.join('')}</div></div>`}).join('');return scopeRows?`<section class="m123-op"><h4>${e(op.id)} · ${e(op.flow)}</h4>${scopeRows}</section>`:''}).join('');return opSections?`<section class="m123-target"><h3>${e(target.label)}</h3>${opSections}</section>`:''}).join(''),
ranking=(payload.impact_ranking||[]).filter(r=>Number.isFinite(Number(r.impact_score_0_100))),
trajectoryRanking=(payload.trajectory_ranking||[]).filter(r=>Number.isFinite(Number(r.metrics?.trajectory_impact_percent_d0))),
trackLossRanking=(payload.track_loss_ranking||[]).filter(r=>Number.isFinite(Number(r.metrics?.target_worst_track_loss_score_0_100))),
disappearanceRanking=(payload.disappearance_ranking||[]).filter(r=>Number.isFinite(Number(r.metrics?.target_worst_disappearance_score_0_100))),
maskAbsenceRanking=(payload.mask_absence_ranking||[]).filter(r=>Number.isFinite(Number(r.metrics?.target_worst_mask_absence_score_0_100))),
trajectoryProgress=payload.trajectory_progress||{},
survivalProgress=payload.object_survival_progress||{},
trajectoryRankingPanel=`<section class="m123-trajectory-ranking"><h3>真实对象轨迹影响排名 · CoTracker Center-ADE</h3><p>已跟踪 ${trajectoryProgress.tracked||0}/${trajectoryProgress.expected||records.length}，通过覆盖质量门并参与排名 ${trajectoryProgress.ranked||0}。数值 = 100 × 选中对象相对 Baseline 的 bbox-diagonal-normalized Center-ADE；100 表示平均偏移等于一个 D0，允许超过 100。</p>${trajectoryRanking.length?`<ol>${trajectoryRanking.slice(0,15).map((r,index)=>`<li><b>#${index+1} · ${e(r.target_scope==='single_object'?(r.region||'object'):'all_objects')} · ${e(modeLabel(r.mask_mode))} · ${e(r.head_scope)}</b><span>轨迹影响 ${num(r.metrics.trajectory_impact_percent_d0)}% D0 · Center-ADE ${num(r.metrics.target_center_ade_norm)}</span></li>`).join('')}</ol>`:'<div class="pending">等待首个通过质量门的 CoTracker 结果；不会用像素 MAE 补位。</div>'}</section>`,
trackLossRankingPanel=`<section class="m123-loss-ranking track-loss"><h3>CoTracker Track Loss 排名 · 108 条均可比较</h3><p><code>100 × (1−common center coverage)</code>；越大越不可跟踪，但不单独证明对象真的消失。</p><ol>${trackLossRanking.slice(0,15).map((r,index)=>`<li><b>#${index+1} · ${e(r.target_scope==='single_object'?(r.region||'object'):'all_objects')} · ${e(modeLabel(r.mask_mode))} · ${e(r.head_scope)}</b><span>Worst Track Loss ${num(r.metrics.target_worst_track_loss_score_0_100)} /100 · ADE ${num(r.metrics.trajectory_impact_percent_d0)}</span></li>`).join('')}</ol></section>`,
disappearanceRankingPanel=`<section class="m123-loss-ranking disappearance"><h3>对象保留失败排名 · SAM2+DINOv2 composite</h3><p>已计算 ${survivalProgress.measured||0}/${survivalProgress.expected||records.length}；每帧同时要求 mask、identity、area 三重条件，越大表示对象保留越差。</p>${disappearanceRanking.length?`<ol>${disappearanceRanking.slice(0,15).map((r,index)=>`<li><b>#${index+1} · ${e(r.target_scope==='single_object'?(r.region||'object'):'all_objects')} · ${e(modeLabel(r.mask_mode))} · ${e(r.head_scope)}</b><span>Worst Retention Failure ${num(r.metrics.target_worst_disappearance_score_0_100)} /100</span></li>`).join('')}</ol>`:'<div class="pending">等待 SAM2+DINOv2 首个结果。</div>'}</section>`,
maskAbsenceRankingPanel=`<section class="m123-loss-ranking mask-absence"><h3>SAM2 Mask Absence 排名 · 纯消失代理</h3><p><code>100 × selected-object empty-mask rate</code>；越大表示无对象 mask 的帧越多。它比 composite 更接近纯消失，但仍需用 overlay 排除 SAM2 跟踪失败。</p>${maskAbsenceRanking.length?`<ol>${maskAbsenceRanking.slice(0,15).map((r,index)=>`<li><b>#${index+1} · ${e(r.target_scope==='single_object'?(r.region||'object'):'all_objects')} · ${e(modeLabel(r.mask_mode))} · ${e(r.head_scope)}</b><span>Worst Mask Absence ${num(r.metrics.target_worst_mask_absence_score_0_100)} /100</span></li>`).join('')}</ol>`:'<div class="pending">等待 SAM2 mask-absence 结果。</div>'}</section>`,
categoryPanels=categoryOrder.map(id=>{const rows=(payload.category_rankings?.[id]||[]).filter(r=>Number.isFinite(Number(r.category_scores_0_100?.[id]))),def=categoryDefs[id]||{};if(!rows.length)return '';return `<section class="m123-category-ranking ${e(id)}"><h3>${e(categoryName(id))} ↑</h3><p>${e(def.direction||'越大表示该类影响越强')}</p><code>${e(def.formula||'')}</code><ol>${rows.slice(0,12).map((r,index)=>`<li><b>#${index+1} · ${e(r.target_scope==='single_object'?(r.region||'object'):'all_objects')} · ${e(modeLabel(r.mask_mode))} · ${e(r.head_scope)}</b><span>${e(categoryName(id))} ${num(r.category_scores_0_100[id])}</span></li>`).join('')}</ol></section>`}).join(''),
categoryRankingPanel=categoryPanels?`<div class="m123-category-rankings"><div class="m123-ranking-intro"><h3>vs Baseline · 分类影响独立排序</h3><p>同一 case / seed 中按每类分数单独降序；未计算视频不参与排名。</p></div><div class="m123-category-ranking-grid">${categoryPanels}</div></div>`:`<section class="m123-impact-ranking pending"><h3>vs Baseline 分类排名待计算</h3><p>增量 bench 会在视频生成后自动写入。</p></section>`,
overallRankingPanel=ranking.length?`<details class="m123-overall-ranking"><summary>展开查看旧版综合影响排序 · ${ranking.length} 个视频</summary><ol>${ranking.slice(0,15).map(r=>`<li><b>${e(r.target_scope==='single_object'?(r.region||'object'):'all_objects')} · ${e(modeLabel(r.mask_mode))} · ${e(r.head_scope)}</b><span>综合影响 ${num(r.impact_score_0_100)}</span></li>`).join('')}</ol></details>`:'',
done=records.filter(r=>r.ready).length,expected=records.length,scopeText=scopes.map(x=>x[1]).join(' / '),baseline=s.video_ready?`<figure class="m123-baseline"><video controls muted playsinline preload="none" src="${api}/video?${base}"></video><figcaption><strong>Baseline · seed=${s.seed} · No intervention</strong></figcaption></figure>`:'';
panel.innerHTML=`<div class="m123-title"><h2>M1 / M2 / M3 · Head Scope 因果对照</h2><p>当前比较 ${e(scopeText)}；ranking=${e(payload.ranking_tag||'frozen134')}。每个集合均比较 All-time / Same / Future / Past；干预覆盖 40 个去噪步和 conditional/unconditional 两个 CFG 分支。只展示已生成视频，当前 ${done}/${expected}。</p></div>${trackLossRankingPanel}${disappearanceRankingPanel}${maskAbsenceRankingPanel}${trajectoryRankingPanel}${categoryRankingPanel}${overallRankingPanel}${baseline}${targetSections}`;
if(location.hash==='#m123HeadScopePanel'&&!panel.dataset.hashScrolled){panel.dataset.hashScrolled='1';requestAnimationFrame(()=>panel.scrollIntoView({block:'start'}))}
}
document.addEventListener('toggle',event=>{const details=event.target;if(!details.open||!details.matches('.m123-trajectory,.m123-survival'))return;details.querySelectorAll('video[data-src]').forEach(video=>{video.src=video.dataset.src;delete video.dataset.src;video.load()})},true)
let m123AttentionLast=null,m123AttentionPoll=null;
async function pollM123Attention(){try{const fresh=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),sample=(fresh.samples||[]).find(row=>row.case==='0613pybullet_sample_001460_w002'&&Number(row.seed)===47326),progress=sample?.m123_head_scope_ablations?.s039_top100_mean_progress||{},ready=Number(progress.ready||0),expected=Number(progress.expected||108);if(m123AttentionLast===null)m123AttentionLast=ready;else if(ready>m123AttentionLast){m123AttentionLast=ready;await load()}if(expected&&ready>=expected&&m123AttentionPoll){clearInterval(m123AttentionPoll);m123AttentionPoll=null}}catch(error){console.warn('S039 Top100 mean auto-refresh failed',error)}}
m123AttentionPoll=setInterval(pollM123Attention,20000);pollM123Attention();
'''
    attention_card_anchor = "card=(target,scope,scopeLabel,scopeDetail,op,time)=>"
    if head_scope_renderer.count(attention_card_anchor) != 1:
        raise RuntimeError("PhysicIQ67 Head-Scope card anchor changed")
    attention_details = r'''attentionDetails=r=>{const capture=r.s039_top100_mean;if(!capture?.ready)return `<details class="m123-attention pending"><summary>S039 Fixed F04 Query · Top100 Mean 待提取</summary><p>完成后自动出现 Object A/B 的 F00/F04/…/F48 横向 overlay。</p></details>`;const rows=(capture.records||[]).map(row=>{const src=`${api}/head-scope-s039-top100-mean-overlay?${base}&variant_id=${encodeURIComponent(r.variant_id)}&region=${encodeURIComponent(row.region_name)}`;return `<section><h5>${e(row.region_name)} · ${e(row.region_phrase||'')}</h5><p>本实验在观察用 Top100 中直接置零 ${num(row.locally_ablated_top100_heads,0)} heads；固定 F04 query 中有 ${num(row.locally_ablated_query_rows,0)}/8 行落入该算子的 target partition。</p><img loading="lazy" data-src="${src}" alt="${e(row.region_name)} S039 Top100 mean overlay"></section>`}).join('');return `<details class="m123-attention"><summary>S039 Fixed F04 Query · Top100 Mean · Object A/B</summary><div class="m123-attention-body"><p><b>三行依次：</b>该消融重放中进入算子前的 softmax；按精确 M1/M2/M3 entries 置零后的有效系数（不重归一化）；被删除系数质量。每行横向为 F00/F04/…/F48。</p>${r.head_scope==='bottom100'?'<p class="m123-attention-note">Bottom100 与观察用 Top100 不相交，所以局部 Effective After = Before；但 Before 已包含早期层/时间步 Bottom100 干预产生的上游变化。</p>':''}${rows}</div></details>`},
'''
    head_scope_renderer = head_scope_renderer.replace(
        attention_card_anchor, attention_details + attention_card_anchor, 1
    )
    metric_order_anchor = "${trajectoryDetails(r)}${survivalDetails(r)}${metricDetails(r)}"
    if head_scope_renderer.count(metric_order_anchor) != 1:
        raise RuntimeError("PhysicIQ67 Head-Scope metric details order changed")
    head_scope_renderer = head_scope_renderer.replace(
        metric_order_anchor,
        "${attentionDetails(r)}${trajectoryDetails(r)}${survivalDetails(r)}${metricDetails(r)}",
        1,
    )
    head_scope_renderer = head_scope_renderer.replace(
        "只展示已生成视频，当前 ${done}/${expected}。",
        "视频 ${done}/${expected}；S039 Fixed-F04 Top100 mean overlay ${Number(payload.s039_top100_mean_progress?.ready||0)}/${expected}。",
        1,
    )
    head_scope_renderer = head_scope_renderer.replace(
        ".m123-trajectory,.m123-survival'))return;",
        ".m123-trajectory,.m123-survival,.m123-attention'))return;",
        1,
    )
    head_scope_renderer = head_scope_renderer.replace(
        "video.load()})},true)",
        "video.load()});details.querySelectorAll('img[data-src]').forEach(image=>{image.src=image.dataset.src;delete image.dataset.src})},true)",
        1,
    )
    renderer_anchor = "function renderAblations(s,base){"
    if page.count(renderer_anchor) != 1:
        raise RuntimeError("PhysicIQ67 ablation renderer anchor changed")
    page = page.replace(renderer_anchor, head_scope_renderer + renderer_anchor, 1)
    render_hook = "renderAblations(s,base);renderMatrix();"
    if page.count(render_hook) != 1:
        raise RuntimeError("PhysicIQ67 render hook changed")
    page = page.replace(
        render_hook,
        "renderM123HeadScopes(s,base);renderAblations(s,base);renderMatrix();",
        1,
    )

    main_group = (
        "group=specs.filter(x=>x.target_scope===targetScope&&x.region===targetRegion),"
        "tubeGroup="
    )
    if page.count(main_group) != 1:
        raise RuntimeError("PhysicIQ67 object group anchor changed")
    page = page.replace(
        main_group,
        "group=specs.filter(x=>x.target_scope===targetScope&&x.region===targetRegion&&!['self_only','incoming_only','outgoing_only'].includes(x.mask_mode)),tubeGroup=",
        1,
    )
    page = page.replace(
        "kind==='fixed'?'M1→C1':'M1/Base→Same→Future→Past · M2/Base→Same→Future→Past · M3/Base→Same→Future→Past · M4→C1'",
        "kind==='fixed'?'M4→C1':'M4→C1 · M1/M2/M3 见独立 Head Scope 板块'",
        1,
    )

    page = page.replace(
        "</style>",
        ".m123-head-panel{margin:18px 0 32px;padding:14px;border:2px solid #176654;border-radius:16px;background:#eaf5f1}.m123-title{padding:12px;border-left:7px solid #176654;background:#fff}.m123-title h2{margin:0 0 7px}.m123-baseline{width:min(520px,100%);margin:14px 0}.m123-target{margin:22px 0;padding:10px;background:#fff}.m123-target>h3{margin:0 -10px 12px;padding:9px 12px;background:#17443a;color:#fff}.m123-op{margin:14px 0 24px;border:1px solid #9fbdb4}.m123-op>h4{margin:0;padding:9px 11px;background:#d9ebe5}.m123-scope-row{padding:9px;border-top:1px solid #c9ddd6}.m123-scope-heading{display:flex;justify-content:space-between;gap:12px;align-items:baseline;padding:4px 2px 8px}.m123-scope-heading span{font:11px ui-monospace,monospace}.m123-video-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.m123-video-grid figure{min-width:0}.m123-video-grid figcaption{line-height:1.4}@media(max-width:1400px){.m123-video-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:720px){.m123-video-grid{grid-template-columns:1fr}.m123-scope-heading{align-items:flex-start;flex-direction:column}}</style>",
        1,
    )
    page = page.replace(
        "</style>",
        ".m123-metrics{margin:0 8px 9px;border:1px solid #8caaa1;background:#f6fbf9}.m123-metrics summary{cursor:pointer;padding:8px 9px;font:700 11px/1.4 ui-monospace,monospace;color:#17443a}.m123-metrics[open] summary{border-bottom:1px solid #b9cec7}.m123-metric-body{padding:7px}.m123-metric-body table{width:100%;font:11px/1.35 ui-monospace,monospace}.m123-metric-body th{text-align:left}.m123-metric-body td{text-align:right;font-weight:800}.m123-metric-body p{margin:8px 2px 2px;font-size:11px;line-height:1.45}.m123-category-score{background:#e6f2ee}.m123-raw-start th,.m123-raw-start td{border-top:2px solid #8caaa1;padding-top:6px}.m123-category-rankings{margin:13px 0;padding:12px;background:#fff}.m123-ranking-intro h3{margin:0 0 5px}.m123-category-ranking-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:12px}.m123-category-ranking{min-width:0;padding:11px;border-left:7px solid #b95031;background:#f8f4eb}.m123-category-ranking.target_local{border-left-color:#176654}.m123-category-ranking.temporal_appearance{border-left-color:#315f9a}.m123-category-ranking.outside_spillover{border-left-color:#8452a5}.m123-category-ranking h3{margin:0 0 5px}.m123-category-ranking p{min-height:2.8em;margin:0 0 7px;font-size:12px}.m123-category-ranking code{display:block;overflow-wrap:anywhere;font-size:11px}.m123-category-ranking ol,.m123-overall-ranking ol{margin:10px 0 0;padding-left:27px}.m123-category-ranking li,.m123-overall-ranking li{margin:0 0 7px;padding:5px 7px;background:#fff}.m123-category-ranking li b,.m123-category-ranking li span,.m123-overall-ranking li b,.m123-overall-ranking li span{display:block}.m123-category-ranking li span,.m123-overall-ranking li span{font:11px ui-monospace,monospace;color:#695948}.m123-overall-ranking{margin:10px 0;padding:10px 12px;background:#fff}.m123-overall-ranking summary{cursor:pointer;font-weight:800}@media(max-width:900px){.m123-category-ranking-grid{grid-template-columns:1fr}}</style>",
        1,
    )
    page = page.replace(
        "</style>",
        ".m123-trajectory{margin:0 8px 9px;border:1px solid #315f9a;background:#f2f6fb}.m123-trajectory summary{cursor:pointer;padding:8px 9px;font:800 11px/1.4 ui-monospace,monospace;color:#244d80}.m123-trajectory-body{padding:7px}.m123-trajectory-body table{width:100%;font:11px/1.35 ui-monospace,monospace}.m123-trajectory-body th{text-align:left}.m123-trajectory-body td{text-align:right;font-weight:800}.m123-trajectory-body video{width:100%;margin-top:8px}.m123-trajectory-body p{font-size:11px;line-height:1.45}.m123-trajectory-ranking{margin:13px 0;padding:12px;border-left:8px solid #315f9a;background:#fff}.m123-trajectory-ranking h3{margin:0 0 6px}.m123-trajectory-ranking ol{margin:10px 0 0;padding-left:27px;columns:2;column-gap:28px}.m123-trajectory-ranking li{break-inside:avoid;margin:0 0 7px;padding:5px 7px;background:#edf3fa}.m123-trajectory-ranking li b,.m123-trajectory-ranking li span{display:block}.m123-trajectory-ranking li span{font:11px ui-monospace,monospace;color:#49627e}@media(max-width:900px){.m123-trajectory-ranking ol{columns:1}}</style>",
        1,
    )
    page = page.replace(
        "</style>",
        ".m123-survival{margin:0 8px 9px;border:1px solid #a64c35;background:#fff6f1}.m123-survival summary{cursor:pointer;padding:8px 9px;font:800 11px/1.4 ui-monospace,monospace;color:#8c3625}.m123-survival-body{padding:7px}.m123-survival-body table{width:100%;font:11px/1.35 ui-monospace,monospace}.m123-survival-body th{text-align:left}.m123-survival-body td{text-align:right;font-weight:800}.m123-survival-body video{width:100%;margin-top:8px;aspect-ratio:1280/400}.m123-survival-body p{font-size:11px;line-height:1.45}.m123-loss-ranking{margin:13px 0;padding:12px;border-left:8px solid #8052a5;background:#fff}.m123-loss-ranking.disappearance{border-left-color:#a64c35}.m123-loss-ranking.mask-absence{border-left-color:#111;background:#f6f6f3}.m123-loss-ranking h3{margin:0 0 6px}.m123-loss-ranking ol{margin:10px 0 0;padding-left:27px;columns:2;column-gap:28px}.m123-loss-ranking li{break-inside:avoid;margin:0 0 7px;padding:5px 7px;background:#f4eff8}.m123-loss-ranking.disappearance li{background:#fff1eb}.m123-loss-ranking.mask-absence li{background:#ecece7}.m123-loss-ranking li b,.m123-loss-ranking li span{display:block}.m123-loss-ranking li span{font:11px ui-monospace,monospace;color:#695578}@media(max-width:900px){.m123-loss-ranking ol{columns:1}}</style>",
        1,
    )
    page = page.replace(
        "</style>",
        ".m123-attention{margin:0 8px 9px;border:1px solid #b17a19;background:#fffaf0}.m123-attention summary{cursor:pointer;padding:8px 9px;font:800 11px/1.4 ui-monospace,monospace;color:#79500c}.m123-attention-body{padding:8px;overflow-x:auto}.m123-attention-body>p{font-size:11px;line-height:1.5}.m123-attention-body section{margin:10px 0;padding:7px;background:#fff}.m123-attention-body h5{margin:0 0 4px}.m123-attention-body section p{margin:0 0 7px;font-size:10px;line-height:1.45}.m123-attention-body img{display:block;width:2260px;max-width:none;height:auto}.m123-attention-note{padding:7px;border-left:5px solid #b17a19;background:#fff3cf}</style>",
        1,
    )

    progress_start = page.index("const temporalProgress=catalog=>")
    progress_end = page.index(";async function pollDirectionalCompletion", progress_start)
    head_scope_progress = r'''const temporalProgress=catalog=>{const samples=(catalog.samples||[]).filter(x=>x.case==='0613pybullet_sample_001460_w002'&&Array.isArray(x.temporal_tube_attention_matrix_ablations)),directional=samples.flatMap(x=>x.temporal_tube_attention_matrix_ablations.filter(r=>r.temporal_directional)),headSamples=(catalog.samples||[]).filter(x=>Array.isArray(x.m123_head_scope_ablations?.records)),headRowsBySample=headSamples.map(x=>{const rows=x.m123_head_scope_ablations.records,isPilot=x.case==='0613pybullet_sample_001460_w002'&&Number(x.seed)===47326;return {rows:isPilot?rows.filter(r=>r.head_scope!=='top100'):rows,complete:false}}),headRows=headRowsBySample.flatMap(x=>x.rows),trajectoryProgress=headSamples.map(x=>x.m123_head_scope_ablations.trajectory_progress||{}).filter(x=>Number(x.expected)>0),trajectoryTracked=trajectoryProgress.reduce((n,x)=>n+Number(x.tracked||0),0),trajectoryExpected=trajectoryProgress.reduce((n,x)=>n+Number(x.expected||0),0),survivalProgress=headSamples.map(x=>x.m123_head_scope_ablations.object_survival_progress||{}).filter(x=>Number(x.expected)>0),survivalMeasured=survivalProgress.reduce((n,x)=>n+Number(x.measured||0),0),survivalExpected=survivalProgress.reduce((n,x)=>n+Number(x.expected||0),0),ready=directional.filter(r=>r.ready).length+headRows.filter(r=>r.ready).length,measured=headRows.filter(r=>r.ready&&r.baseline_metrics).length,expected=directional.length+headRows.length,completeSeeds=headRowsBySample.filter(x=>x.rows.length>0&&x.rows.every(r=>r.ready)).length;return {ready,measured,expected,trajectoryTracked,trajectoryExpected,survivalMeasured,survivalExpected,completeSeeds,seedCount:headRowsBySample.length}}'''
    page = page[:progress_start] + head_scope_progress + page[progress_end:]
    page = page.replace(
        "if(old.expected&&old.ready===old.expected){if(directionalPoll)clearInterval(directionalPoll);return}",
        "if(old.expected&&old.ready===old.expected&&old.measured>=old.ready&&(!old.trajectoryExpected||old.trajectoryTracked>=old.trajectoryExpected)&&(!old.survivalExpected||old.survivalMeasured>=old.survivalExpected)){if(directionalPoll)clearInterval(directionalPoll);return}",
        1,
    )
    page = page.replace(
        "Temporal M1/M2/M3 Same/Future/Past ${next.ready}/${next.expected} · seeds ${next.completeSeeds}/${next.seedCount} · 完成后自动刷新",
        "Temporal/Head Scope ${next.ready}/${next.expected} · vs Baseline ${next.measured}/${next.ready} · CoTracker ${next.trajectoryTracked}/${next.trajectoryExpected||0} · Object Survival ${next.survivalMeasured}/${next.survivalExpected||0} · seeds ${next.completeSeeds}/${next.seedCount} · 自动刷新",
        1,
    )
    page = page.replace(
        "if(next.completeSeeds>old.completeSeeds||next.ready===next.expected){if(next.ready===next.expected&&directionalPoll)clearInterval(directionalPoll);await load();",
        "if(next.completeSeeds>old.completeSeeds||next.measured>old.measured||next.trajectoryTracked>old.trajectoryTracked||next.survivalMeasured>old.survivalMeasured||next.ready===next.expected){if(next.ready===next.expected&&next.measured>=next.ready&&(!next.trajectoryExpected||next.trajectoryTracked>=next.trajectoryExpected)&&(!next.survivalExpected||next.survivalMeasured>=next.survivalExpected)&&directionalPoll)clearInterval(directionalPoll);await load();",
        1,
    )
    global_replay_button = '<button id="replay">重新播放</button>'
    if page.count(global_replay_button) != 1:
        raise RuntimeError("PhysicIQ67 global replay button changed")
    page = page.replace(global_replay_button, "", 1)

    section_replay_css = r'''
.section-replay-toolbar{display:flex;justify-content:flex-end;align-items:center;gap:8px;margin:0 0 8px}.section-replay{flex:0 0 auto;padding:6px 10px;border:1px solid #8c7b63;border-radius:7px;background:#fffaf0;color:#17443a;cursor:pointer;font:800 12px/1.2 ui-monospace,monospace}.section-replay:hover{background:#17443a;color:#fff}.section-replay:focus-visible{outline:3px solid #d29c35;outline-offset:2px}.object-protocol-heading .section-replay,.object-ablation-heading .section-replay,.m123-scope-heading .section-replay{margin-left:auto}.object-ablation-heading .section-replay{color:#17443a}.requested-top100>.section-replay-toolbar{grid-column:1/-1}.tube-baseline>.section-replay-toolbar,.m123-baseline>.section-replay-toolbar{margin:0 0 7px}
'''
    if page.count("</style>") != 1:
        raise RuntimeError("PhysicIQ67 style boundary changed")
    page = page.replace("</style>", section_replay_css + "</style>", 1)

    global_replay_listener = "$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));"
    if page.count(global_replay_listener) != 1:
        raise RuntimeError("PhysicIQ67 global replay listener changed")
    section_replay_script = r'''const replaySectionSelectors=['.media','.ablation-baseline','.tube-baseline','.requested-top100','.object-protocol-row','.control-row','.m123-baseline','.m123-scope-row'];function decorateReplaySections(root=document){root.querySelectorAll(replaySectionSelectors.join(',')).forEach(section=>{if(section.dataset.replaySection||!section.querySelector('video'))return;section.dataset.replaySection='1';const button=document.createElement('button');button.type='button';button.className='section-replay';button.textContent='重播本板块';button.setAttribute('aria-label','从头播放本板块全部视频');let heading=null;if(section.matches('.object-protocol-row'))heading=section.querySelector(':scope > .object-protocol-heading');else if(section.matches('.control-row'))heading=section.querySelector(':scope > .object-ablation-heading');else if(section.matches('.m123-scope-row'))heading=section.querySelector(':scope > .m123-scope-heading');if(heading)heading.append(button);else{const toolbar=document.createElement('div');toolbar.className='section-replay-toolbar';toolbar.append(button);section.prepend(toolbar)}})}function replaySection(button){const section=button.closest('[data-replay-section]');if(!section)return;const videos=[...section.querySelectorAll('video')];videos.forEach(video=>{const play=()=>{video.currentTime=0;video.play().catch(()=>{})};video.pause();if(video.readyState>=1)play();else video.addEventListener('loadedmetadata',play,{once:true})});const label=button.textContent;button.textContent=`已重播 ${videos.length} 个视频`;setTimeout(()=>{button.textContent=label},1200)}document.addEventListener('click',event=>{const button=event.target.closest('.section-replay');if(button)replaySection(button)});const replayObserver=new MutationObserver(()=>decorateReplaySections());replayObserver.observe(document.querySelector('main'),{childList:true,subtree:true});decorateReplaySections();'''
    page = page.replace(global_replay_listener, section_replay_script, 1)
    return page


def wan22_ti2v_legacy_m123_s039_top100_mean_page():
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1/M2/M3 S039 Top100 Mean Overlays</title><style>
:root{--paper:#ece4d5;--ink:#17261f;--deep:#17443a;--line:#baad98;--card:#fffaf0;--gold:#b87811;--blue:#315f9a;--red:#a64c35}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#d65f3c35,transparent 34rem),radial-gradient(circle at 100% 0,#27897935,transparent 40rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:5;padding:14px 20px;border-bottom:1px solid var(--line);background:#ece4d5f2;backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:5px 0;font-size:clamp(28px,4vw,54px);line-height:1}.lead{margin:6px 0;max-width:1200px}.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.tools label{font-weight:800}.tools select,.tools button{padding:7px 9px;border:1px solid var(--line);background:var(--card);font-weight:800}.status{font:12px ui-monospace,monospace}.progress{height:5px;margin-top:9px;background:#d6cbb9}.progress>i{display:block;height:100%;width:0;background:var(--deep);transition:width .25s}main{width:min(100% - 18px,2300px);margin:auto;padding:17px 0 70px}.definition{padding:15px;border:1px solid var(--line);border-radius:16px;background:#fffaf0e8}.definition h2{margin:0 0 7px}.math{padding:8px 10px;overflow-wrap:anywhere;border-left:5px solid var(--deep);background:#edf6f2;font:12px/1.55 ui-monospace,monospace}.row-defs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px}.row-def{padding:12px;border-top:6px solid var(--deep);background:#fff}.row-def.after{border-color:var(--blue)}.row-def.removed{border-color:var(--red)}.row-def h3{margin:0 0 6px}.row-def p{margin:4px 0;line-height:1.5}.note{margin:11px 0 0;padding:10px;border-left:6px solid var(--gold);background:#fff3cf;line-height:1.55}.results-head{display:flex;justify-content:space-between;gap:12px;align-items:end;margin:20px 0 9px}.results-head h2{margin:0}.grid{display:grid;grid-template-columns:1fr;gap:14px}.card{padding:12px;border:1px solid var(--line);border-radius:14px;background:#fff}.card-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}.card h3{margin:0}.badges{display:flex;gap:5px;flex-wrap:wrap}.badge{padding:3px 7px;border-radius:999px;background:#dcebe6;font:10px ui-monospace,monospace}.exact{margin:8px 0;padding:8px;background:#f5f0e7;font:11px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}.videos,.objects{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.videos{margin:10px 0}.video-panel{min-width:0;padding:8px;background:#eef4f1}.video-panel h4{margin:0 0 6px}.video-panel video{display:block;width:100%;aspect-ratio:1280/704;background:#111}.object{min-width:0;padding:8px;background:#f8f6ef}.object h4{margin:0 0 4px}.object p{margin:0 0 7px;font-size:11px;line-height:1.45}.object a{display:block;overflow:auto}.object img{display:block;width:100%;height:auto;background:#222}.empty{padding:40px;text-align:center;border:1px dashed var(--line);background:#fff}.bottom-note{margin:8px 0 0;color:#79500c;font-size:11px}@media(max-width:900px){header{position:static}.row-defs,.videos,.objects{grid-template-columns:1fr}.tools label{width:100%}.tools select{width:100%}}
</style></head><body><header><a href="/">返回总入口</a> · <a href="/object-query-m123-temporal-batch?single=1&amp;case=0613pybullet_sample_001460_w002&amp;seed=47326&amp;v=5">消融视频与指标页</a><h1>S039 Fixed-F04<br>Top100 Mean Overlays</h1><p class="lead">0613pybullet_sample_001460_w002 · seed 47326 · 108 项 M1/M2/M3 Head-Scope 实验</p><div class="tools"><label>Target <select id="target"><option value="single_object::object_A">Object A</option><option value="single_object::object_B">Object B</option><option value="all_objects::">All objects</option></select></label><label>Head Scope <select id="scope"><option value="top100">Top100</option><option value="bottom100">Bottom100</option><option value="all720">All720</option></select></label><label>Operator <select id="operator"><option value="all">M1 + M2 + M3</option><option value="self">M1</option><option value="incoming">M2</option><option value="outgoing">M3</option></select></label><label>Time <select id="time"><option value="all">全部四类</option><option value="only">All-time</option><option value="same">Same</option><option value="future">Future</option><option value="past">Past</option></select></label><button id="refresh" type="button">刷新</button><span id="status" class="status">读取中</span></div><div class="progress"><i id="progress"></i></div></header><main><section class="definition"><h2>三行 Overlay 的精确含义</h2><p>固定 Object A/B 在像素帧 F04 的 8 个 SAM2 query points，映射到 latent <code>tq=1</code>。观察集合始终是冻结 S039 PCK 排名的 Top100 heads，与本实验实际干预的是 Top100、Bottom100 还是 All720 分开定义。</p><div class="math">A = softmax(QKᵀ/√d)<br>M(tk,y,x) = (1 / (100 heads × 2 CFG)) Σ_head Σ_CFG Σ_i=1..8 A[q_i, k(tk,y,x)]</div><div class="row-defs"><article class="row-def"><h3>第 1 行 · Pre-mask / Before</h3><p>当前消融重放在 S039、进入该 attention 算子之前的实时 softmax 概率，再按上式聚合。</p><p>它已经包含更早去噪步和更早 block 的上游消融影响，因此不是未消融 Baseline attention。</p></article><article class="row-def after"><h3>第 2 行 · Effective After</h3><p>在 Before 上，仅把当前 M1/M2/M3 算子精确删除的 <code>A[target query, source K/V]</code> entries 设为 0，再按同一方式聚合。</p><p>不重新 softmax、不重归一化，也不改 Q/K/V；它是原实现“从 A@V 中减去对应贡献”的等价有效系数图。</p></article><article class="row-def removed"><h3>第 3 行 · Removed</h3><p><code>Removed = max(Before − Effective After, 0)</code>，表示固定 query 在观察 Top100 heads 中，被当前算子直接删除的 attention coefficient mass 落在哪些 K token。</p><p>它不是消融视频相对 Baseline 的 attention 差，也不直接等于视频运动或质量变化。</p></article></div><div class="note"><b>颜色刻度：</b>Before 与 Effective After 在每个 Fxx 内共享 P99.5 色标；Removed 使用自己的逐帧 P99.5 色标。因此同一帧可比较 Before/After，但不能仅按颜色深浅跨行或跨帧比较绝对大小。<br><b>Bottom100：</b>干预 heads 与观察 Top100 不相交，直接的 Effective After 通常等于 Before、Removed 为 0；但 Before 仍可能因前面层和时间步的 Bottom100 干预而改变。黄色框标出固定 query 所在的 F04。</div></section><div class="results-head"><h2>已生成实验</h2><span id="visible" class="status"></span></div><div id="grid" class="grid"><div class="empty">读取 overlay…</div></div></main><script>
const api='/api/wan22-ti2v-legacy-physiciq67-samples',caseName='0613pybullet_sample_001460_w002',seed=47326,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),query=new URL(location.href).searchParams;let sample=null,lastReady=null,initialized=false;
const ops={self:{id:'M1',target:'R',source:'R',flow:'R_tk K/V → R_tq Query'},incoming:{id:'M2',target:'R',source:'C',flow:'C_tk K/V → R_tq Query'},outgoing:{id:'M3',target:'C',source:'R',flow:'R_tk K/V → C_tq Query'}},times={only:{label:'All-time',predicate:'all tk'},same:{label:'Same',predicate:'tk=tq'},future:{label:'Future',predicate:'tk<tq'},past:{label:'Past',predicate:'tk>tq'}};
function parts(mode){const index=mode.lastIndexOf('_');return {base:mode.slice(0,index),time:mode.slice(index+1)}}
function targetKey(r){return r.target_scope==='single_object'?`single_object::${r.region}`:'all_objects::'}
function imageUrl(r,region){const p=new URLSearchParams({case:caseName,seed:String(seed),variant_id:r.variant_id,region,v:'2'});return `${api}/head-scope-s039-top100-mean-overlay?${p}`}
function baselineUrl(){return `${api}/video?${new URLSearchParams({case:caseName,seed:String(seed)})}`}
function ablationUrl(r){const p=new URLSearchParams({case:caseName,seed:String(seed),target_scope:r.target_scope,mask_mode:r.mask_mode,top_n:String(r.head_count),head_scope:r.head_scope});if(r.region)p.set('region',r.region);return `${api}/temporal-tube-ablation-video?${p}`}
function selectedRows(){const records=sample?.m123_head_scope_ablations?.records||[],target=$('target').value,scope=$('scope').value,operator=$('operator').value,time=$('time').value;return records.filter(r=>{const p=parts(r.mask_mode);return r.s039_top100_mean?.ready&&targetKey(r)===target&&r.head_scope===scope&&(operator==='all'||p.base===operator)&&(time==='all'||p.time===time)})}
function sync(){const u=new URL(location.href);for(const id of ['target','scope','operator','time'])u.searchParams.set(id,$(id).value);history.replaceState(null,'',u)}
function card(r){const p=parts(r.mask_mode),op=ops[p.base],time=times[p.time],target=r.target_scope==='single_object'?r.region:'all_objects',formula=`Y′_${op.target}(tq)=Y_${op.target}(tq)−Σ_{${time.predicate}} A[${op.target}_tq,${op.source}_tk]V_${op.source}(tk)`,baseVideo=baselineUrl(),ablatedVideo=ablationUrl(r),objects=(r.s039_top100_mean.records||[]).map(row=>{const src=imageUrl(r,row.region_name);return `<section class="object"><h4>${esc(row.region_name)} · ${esc(row.region_phrase||'')}</h4><p>直接置零 observation Top100：${Number(row.locally_ablated_top100_heads||0)}/100 heads；命中固定 query-point rows：${Number(row.locally_ablated_query_rows||0)}/8。</p><a href="${src}" target="_blank" rel="noopener"><img loading="lazy" src="${src}" alt="${esc(r.variant_id)} ${esc(row.region_name)} overlay"></a></section>`}).join(''),bottom=r.head_scope==='bottom100'?'<p class="bottom-note">Bottom100 不与 observation Top100 重合：第二行与第一行相同、第三行为 0 是预期结果；第一行仍保留上游干预效应。</p>':'';return `<article class="card"><div class="card-head"><h3>${esc(target)} · ${op.id}-${time.label}</h3><div class="badges"><span class="badge">${esc(r.head_scope)}</span><span class="badge">${r.head_count} layer-heads</span><span class="badge">S039</span></div></div><div class="exact"><b>切断：</b>${esc(op.flow)} · ${esc(time.predicate)}<br><b>有效计算：</b>${esc(formula)}<br><b>ID：</b>${esc(r.variant_id)}</div>${bottom}<div class="videos"><section class="video-panel"><h4>Baseline · seed 47326 · No intervention</h4><video controls muted playsinline preload="none" data-src="${esc(baseVideo)}"></video></section><section class="video-panel"><h4>Ablation · ${op.id}-${time.label} · ${esc(r.head_scope)}</h4><video controls muted playsinline preload="none" data-src="${esc(ablatedVideo)}"></video></section></div><div class="objects">${objects}</div></article>`}
let videoObserver=null;function observeVideos(){if(videoObserver)videoObserver.disconnect();const videos=[...document.querySelectorAll('video[data-src]')];if(!('IntersectionObserver'in window)){videos.forEach(video=>{video.src=video.dataset.src;delete video.dataset.src});return}videoObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{if(!entry.isIntersecting)return;const video=entry.target;video.src=video.dataset.src;delete video.dataset.src;video.load();videoObserver.unobserve(video)}),{rootMargin:'360px'});videos.forEach(video=>videoObserver.observe(video))}
function render(){const rows=selectedRows(),progress=sample?.m123_head_scope_ablations?.s039_top100_mean_progress||{},ready=Number(progress.ready||0),expected=Number(progress.expected||108);$('status').textContent=`Overlay ${ready}/${expected} · 每 20 秒自动刷新`;$('progress').style.width=`${expected?100*ready/expected:0}%`;$('visible').textContent=`当前筛选显示 ${rows.length} 项`;$('grid').innerHTML=rows.length?rows.map(card).join(''):'<div class="empty">当前筛选下尚无完成的 overlay。</div>';observeVideos();sync()}
async function load(){const catalog=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`${r.status} ${r.statusText}`);return r.json()});sample=(catalog.samples||[]).find(x=>x.case===caseName&&Number(x.seed)===seed);if(!sample)throw new Error('catalog 中找不到目标 case/seed');if(!initialized){for(const id of ['target','scope','operator','time']){const value=query.get(id);if(value&&[...$(id).options].some(x=>x.value===value))$(id).value=value}initialized=true}render();lastReady=Number(sample.m123_head_scope_ablations?.s039_top100_mean_progress?.ready||0)}
async function poll(){try{const catalog=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),next=(catalog.samples||[]).find(x=>x.case===caseName&&Number(x.seed)===seed),ready=Number(next?.m123_head_scope_ablations?.s039_top100_mean_progress?.ready||0);if(ready>Number(lastReady||0)){sample=next;lastReady=ready;render()}if(ready>=108)clearInterval(timer)}catch(error){$('status').textContent=`自动刷新失败：${error.message}`}}
for(const id of ['target','scope','operator','time'])$(id).addEventListener('change',render);$('refresh').addEventListener('click',()=>load().catch(error=>$('status').textContent=`加载失败：${error.message}`));load().catch(error=>$('status').textContent=`加载失败：${error.message}`);const timer=setInterval(poll,20000);
</script></body></html>'''
    page = page.replace(
        '<label>Target <select id="target"><option value="single_object::object_A">Object A</option><option value="single_object::object_B">Object B</option><option value="all_objects::">All objects</option></select></label>',
        '<label>Target <select id="target"><option value="single_objects">Object A + Object B</option><option value="single_object::object_A">Object A only</option><option value="single_object::object_B">Object B only</option><option value="all_objects::">All objects union</option></select></label>',
        1,
    )
    page = page.replace(
        "const p=parts(r.mask_mode);return r.s039_top100_mean?.ready&&targetKey(r)===target&&r.head_scope===scope",
        "const p=parts(r.mask_mode),targetMatch=target==='single_objects'?r.target_scope==='single_object'&&['object_A','object_B'].includes(r.region):targetKey(r)===target;return r.s039_top100_mean?.ready&&targetMatch&&r.head_scope===scope",
        1,
    )
    page = page.replace("function selectedRows(){", "function selectedRowsUnsorted(){", 1)
    page = page.replace(
        "function sync(){",
        "function selectedRows(){const rows=selectedRowsUnsorted();if($('target').value!=='single_objects')return rows;const order=['self_only','self_same','self_future','self_past','incoming_only','incoming_same','incoming_future','incoming_past','outgoing_only','outgoing_same','outgoing_future','outgoing_past'];return rows.sort((a,b)=>order.indexOf(a.mask_mode)-order.indexOf(b.mask_mode)||(a.region==='object_A'?-1:1))}\nfunction sync(){",
        1,
    )
    page = page.replace(
        "target=r.target_scope==='single_object'?r.region:'all_objects'",
        "target=r.target_scope==='single_object'?(r.region==='object_A'?'Object A target':'Object B target'):'All objects union'",
        1,
    )
    receiver_css = r'''.receiver{margin-top:10px;padding:10px;border:1px solid #8eb9ad;background:#eef8f4}.receiver h4{margin:0 0 5px}.receiver p{margin:4px 0;font-size:11px;line-height:1.5}.receiver a{display:block;overflow:auto;margin-top:7px}.receiver img{display:block;width:100%;height:auto;background:#222}.receiver.pending{border-style:dashed;color:#6e685d;background:#f4f0e8}.receiver-definition{margin-top:12px;border-left:7px solid #287d6c}.receiver-definition .row-defs{grid-template-columns:repeat(2,minmax(0,1fr))}.receiver-definition .coefficient{border-color:#287d6c}.receiver-definition .value{border-color:#7b4fa0}@media(max-width:900px){.receiver-definition .row-defs{grid-template-columns:1fr}}'''
    if page.count("</style>") != 1:
        raise RuntimeError("S039 standalone style boundary changed")
    page = page.replace("</style>", receiver_css + "</style>", 1)
    page = page.replace(
        "<h1>S039 Fixed-F04<br>Top100 Mean Overlays</h1>",
        "<h1>S039 Key-side + Query-side<br>Attention Overlays</h1>",
        1,
    )
    receiver_definition = r'''<section class="definition receiver-definition"><h2>Query-side Receiver Overlay · 精确含义</h2><p>与上面的 Key-side 固定 F04 图不同，这两行覆盖全部 13 个 Query 时刻和完整 Query 空间，并只在本实验实际干预的 Top100、Bottom100 或 All720 heads 上平均。青色轮廓标出 sender/target 的 R tube。</p><div class="math">S(q) = mean_head,CFG Σ_{k∈source,time predicate} A[q,k]<br>E(q) = mean_head,CFG ‖Σ_{k∈source,time predicate} A[q,k]V[k]‖₂</div><div class="row-defs"><article class="row-def coefficient"><h3>第 1 行 · Coefficient Receiver S(q)</h3><p>每个 Query 从被切断 source partition 读取的 softmax coefficient mass。M3 中非零位置位于 C Query 空间，因此可观察 target R 向其他对象和背景的广播接收方。</p></article><article class="row-def value"><h3>第 2 行 · Value Contribution E(q)</h3><p>在每个 physical head 内先计算被删除的 A@V 向量 L2，再跨 head/CFG 平均；不会因不同 head 的向量方向相反而相互抵消。</p></article></div><div class="note"><b>横向坐标：</b>Q00/F00…Q12/F48 是 Query 时间，不是 Key 时间。<br><b>颜色刻度：</b>每一行、每个实验使用一个全 13 帧共享的 P99.5 色标，保留该实验内部的时间强弱；跨实验比较绝对值应读取 raw NPZ/标注的 vmax，而不能只看颜色。</div></section>'''
    results_anchor = '<div class="results-head"><h2>已生成实验</h2>'
    if page.count(results_anchor) != 1:
        raise RuntimeError("S039 standalone results anchor changed")
    page = page.replace(results_anchor, receiver_definition + results_anchor, 1)
    page = page.replace(
        "function baselineUrl(){",
        "function receiverUrl(r){const p=new URLSearchParams({case:caseName,seed:String(seed),variant_id:r.variant_id,v:'1'});return `${api}/head-scope-s039-query-receiver-overlay?${p}`}\nfunction baselineUrl(){",
        1,
    )
    page = page.replace(
        ",bottom=r.head_scope==='bottom100'?",
        ",receiver=r.s039_query_receiver?.ready?`<section class=\"receiver\"><h4>Query-side Receiver · S(q) coefficient mass + E(q) value contribution</h4><p>Q=${esc(r.s039_query_receiver.target_partition)}；source K/V=${esc(r.s039_query_receiver.source_partition)}；${esc(r.s039_query_receiver.time_predicate)}；使用本实验实际 ${esc(r.head_scope)} heads。青色轮廓为 target R tube。</p><a href=\"${receiverUrl(r)}\" target=\"_blank\" rel=\"noopener\"><img loading=\"lazy\" src=\"${receiverUrl(r)}\" alt=\"${esc(r.variant_id)} query receiver overlay\"></a></section>`:`<section class=\"receiver pending\"><h4>Query-side Receiver 待提取</h4><p>S(q) / E(q) 完成后自动出现在这里。</p></section>`,bottom=r.head_scope==='bottom100'?",
        1,
    )
    page = page.replace(
        '<div class="objects">${objects}</div></article>',
        '<div class="objects">${objects}</div>${receiver}</article>',
        1,
    )
    page = page.replace(
        "$('status').textContent=`Overlay ${ready}/${expected} · 每 20 秒自动刷新`",
        "const receiverProgress=sample?.m123_head_scope_ablations?.s039_query_receiver_progress||{};$('status').textContent=`Key-side ${ready}/${expected} · Query-side ${Number(receiverProgress.ready||0)}/${Number(receiverProgress.expected||108)} · 每 20 秒自动刷新`",
        1,
    )
    page = page.replace(
        "lastReady=Number(sample.m123_head_scope_ablations?.s039_top100_mean_progress?.ready||0)",
        "lastReady=Number(sample.m123_head_scope_ablations?.s039_top100_mean_progress?.ready||0)+Number(sample.m123_head_scope_ablations?.s039_query_receiver_progress?.ready||0)",
        1,
    )
    page = page.replace(
        "ready=Number(next?.m123_head_scope_ablations?.s039_top100_mean_progress?.ready||0);if(ready>Number(lastReady||0)){sample=next;lastReady=ready;render()}if(ready>=108)clearInterval(timer)",
        "ready=Number(next?.m123_head_scope_ablations?.s039_top100_mean_progress?.ready||0),receiverReady=Number(next?.m123_head_scope_ablations?.s039_query_receiver_progress?.ready||0),combinedReady=ready+receiverReady;if(combinedReady>Number(lastReady||0)){sample=next;lastReady=combinedReady;render()}if(ready>=108&&receiverReady>=108)clearInterval(timer)",
        1,
    )
    return page


def wan22_ti2v_legacy_m123_temporal_batch_page():
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 M2 M3 Temporal Batch</title><link rel="icon" href="data:"><style>
:root{--paper:#ece4d5;--ink:#17261f;--deep:#17443a;--line:#baad98;--card:#fffaf0;--gold:#d29c35;--blue:#315f9a}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#d65f3c3b,transparent 34rem),radial-gradient(circle at 100% 0,#27897938,transparent 40rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:14px 22px;background:#ece4d5f4;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:5px 0;font-size:clamp(29px,4vw,54px);line-height:1}.lead{max-width:1500px;margin:6px 0;line-height:1.5}.tools{display:flex;align-items:end;gap:9px;flex-wrap:wrap}label{display:grid;gap:4px;font:700 11px ui-monospace,monospace}select,button{max-width:min(72vw,700px);padding:8px 10px;border:1px solid var(--line);background:var(--card);color:var(--ink);font-weight:800}button{cursor:pointer}.status{font:12px ui-monospace,monospace}main{width:min(100% - 18px,2200px);margin:auto;padding:18px 0 70px}.note,.definitions,.baseline,.operator{margin:14px 0;padding:13px;border:1px solid var(--line);background:#fffaf0e8;box-shadow:0 12px 30px #58442b16}.note{border-left:7px solid var(--gold);line-height:1.55}.definitions{border-left:7px solid var(--blue)}.definitions h2{margin:0 0 7px}.definitions>p{margin:0 0 10px;line-height:1.5}.table-scroll{overflow:auto;max-height:62vh;border:1px solid var(--line)}.definitions table,.metric table{width:100%;border-collapse:collapse;background:#fff}.definitions th,.definitions td,.metric th,.metric td{padding:7px 8px;border:1px solid #d8cebd;vertical-align:top;text-align:left}.definitions thead th{position:sticky;top:0;background:var(--deep);color:#fff}.definitions .group th{background:#dcece6;color:var(--deep)}.definitions td:nth-child(3),.metric td{font-family:ui-monospace,monospace}.baseline{width:min(560px,100%)}.baseline video,.card video{display:block;width:100%;aspect-ratio:1280/704;background:#111}.operator{border-radius:14px}.operator>h2{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:-13px -13px 12px;padding:11px 13px;background:var(--deep);color:#fff}.operator-replay{flex:0 0 auto;border-color:#d7cdbd;background:#fff;color:var(--deep)}.scope{margin:14px 0;border:1px solid #9fbdb4;background:#f7fbf9}.scope-head{display:flex;justify-content:space-between;gap:12px;padding:9px 11px;background:#dcece6}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;padding:9px}.card{min-width:0;margin:0;padding:8px;border:1px solid #d4c8b5;background:#fff}.card figcaption{padding:8px 2px 2px;line-height:1.45}.card figcaption strong,.card figcaption span{display:block}.formula{margin-top:5px;font:11px/1.45 ui-monospace,monospace;color:#4f5e58}.metric{margin-top:7px;border:1px solid #b9cec7;background:#f5faf8}.metric summary{cursor:pointer;padding:7px;font:700 11px/1.45 ui-monospace,monospace}.metric-body{padding:0 7px 8px;font:11px/1.5 ui-monospace,monospace}.metric-body h4{margin:9px 0 4px;padding:5px 7px;background:#dcece6;color:var(--deep)}.metric-body h5{margin:8px 0 3px;color:var(--blue)}.metric th{width:58%;font-family:inherit}.metric .metric-na{padding:8px;color:#776e60}.empty{padding:24px;text-align:center;color:#776e60;background:#f0eadf}.lazy-note{color:#315f9a;font-weight:800}@media(max-width:1250px){.grid{grid-template-columns:1fr 1fr}}@media(max-width:680px){header{position:static}.grid{grid-template-columns:1fr}.scope-head,.operator>h2{align-items:flex-start;flex-direction:column}.definitions{font-size:12px}}
</style></head><body><header><a href="/">返回总览</a> · <a href="/wan22-ti2v-legacy-physiciq67-samples?v=20">返回完整页面</a><h1>M1 / M2 / M3<br>All-time · Same · Future · Past</h1><p class="lead">当前多 case、多 seed 批次的独立入口。只显示已生成视频；每次仅渲染选中的 target，视频滚动到视口附近时才加载。</p><div class="tools"><label>Case / Seed<select id="sample"></select></label><label>Target<select id="target"></select></label><button id="refresh" type="button">刷新</button><span id="status" class="status">读取中</span></div></header><main><section class="note"><b>共同设置：</b>R 为 Baseline 冻结轨迹在 latent t=0…12 的 object token tube；干预覆盖 S000–S039 全部 40 个去噪步和 conditional/unconditional 两个 CFG 分支。<br><b>Ranking 快照对比：</b>对 001460 / seed 47326，同时展示原 frozen134 与 latest3350；Top100 共有 50/100 heads，Bottom100 共有 94/100 heads。seed、Baseline、target、消融算子和时间谓词均保持不变。<br><b>Target 去重：</b>单对象 case 只保留该 `single_object`；多对象 case 才额外加入 `all_objects` 并集。<br><b>时间方向：</b>All-time 删除所有 tk；Same 删除 tk=tq；Future 删除 tk&lt;tq（历史 K/V → 未来 Query）；Past 删除 tk&gt;tq（未来 K/V → 过去 Query）。后三者互斥，合并后等于 All-time 块。<br><span class="lazy-note">页面不会在首屏同时请求全部 MP4。</span></section><section class="definitions"><h2>指标定义与精确计算</h2><p>本页指标均与同 seed 未消融 Baseline 比较。“影响更大”不等于“质量更差”；轨迹质量门未通过时 ADE/FDE 保持 N/A。未计算的 RAFT、LPIPS、VBench 不在表中伪造数值。</p><div class="table-scroll"><table><thead><tr><th>指标</th><th>定义</th><th>计算形式</th><th>数值方向</th></tr></thead><tbody id="metricDefinitions"></tbody></table></div></section><div id="content"><div class="empty">读取批次记录…</div></div></main><script>
const api='/api/object-query-m123-temporal-batch/catalog',videoApi='/api/wan22-ti2v-legacy-physiciq67-samples',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null;
const ops=[{id:'M1',base:'self',target:'R',source:'R',flow:'R_tk K/V ──X──> R_tq Query',meaning:'对象 tube 内部状态对对象 Query 的自支持与时序传播。'},{id:'M2',base:'incoming',target:'R',source:'C',flow:'C_tk K/V ──X──> R_tq Query',meaning:'环境、背景和其他对象向对象 Query 输入上下文。'},{id:'M3',base:'outgoing',target:'C',source:'R',flow:'R_tk K/V ──X──> C_tq Query',meaning:'对象状态向环境和其他对象 Query 广播。'}],times=[{key:'only',label:'All-time',predicate:'all tk'},{key:'same',label:'Same',predicate:'tk=tq'},{key:'future',label:'Future',predicate:'tk<tq'},{key:'past',label:'Past',predicate:'tk>tq'}],scopeLabel={top100:'Top100 PCK Heads',bottom100:'Bottom100 PCK Heads',all720:'All Heads · 720 layer-heads'};
const scopeKey=r=>`${r.ranking_tag||'original'}::${r.head_scope}`,scopeDisplay=r=>`${r.ranking_label||'原快照'} · ${scopeLabel[r.head_scope]||r.head_scope}`,scopeOrderValue=r=>r.head_scope==='all720'?4:(r.head_scope==='bottom100'?2:0)+(r.ranking_tag==='s039r3350'?1:0);
const metricDefs=[
['视觉与区域影响','Composite Impact /100','像素与目标 ROI 的综合可见干预强度','100×[0.20(1−SSIM)+0.15 Global MAE+0.15 Global Δ-MAE+0.30 ROI MAE+0.20 ROI Δ-MAE]','↑ 影响更强；不是质量分'],
['视觉与区域影响','Global SSIM','49 帧全画面结构相似性','mean_t SSIM(I_abl(t), I_base(t))','↓ 整体结构差异更大'],
['视觉与区域影响','Global PSNR','49 帧全画面峰值信噪比','mean_t PSNR(I_abl(t), I_base(t))','↓ 全局像素差异更大'],
['视觉与区域影响','Global MAE','全画面 RGB 绝对误差','mean |I_abl−I_base| / 255','↑ 全局像素影响更强'],
['视觉与区域影响','Global Δ-MAE','两视频相邻帧差分之差','mean |ΔI_abl−ΔI_base| / 255','↑ 逐帧像素变化差异更强；不是纯轨迹'],
['视觉与区域影响','Target ROI MAE','Baseline 冻结对象 tube ROI 内 RGB 误差','mean_(x∈ROI_base) |I_abl−I_base| / 255','↑ 目标所在区域的位置/外观影响更强'],
['视觉与区域影响','Target ROI Δ-MAE','冻结 ROI 内相邻帧差分误差','mean_(x∈ROI_t∪ROI_t+1) |ΔI_abl−ΔI_base| / 255','↑ 目标区域时序外观差异更强'],
['视觉与区域影响','Outside-object MAE','排除全部 Baseline 对象 ROI 后的 RGB 误差','mean_(x outside object ROIs) |I_abl−I_base| / 255','↑ 背景/其他区域 spillover 更强'],
['视觉与区域影响','Outside-object Δ-MAE','对象 ROI 外的相邻帧差分误差','mean_(x outside ROI union) |ΔI_abl−ΔI_base| / 255','↑ 对象外动态 spillover 更强'],
['视觉与区域影响','Spillover /100','对象外静态与动态像素影响','100×mean(Outside MAE, Outside Δ-MAE)','↑ 向背景/其他区域传播更强'],
['视觉与区域影响','Global appearance /100','全局结构与像素外观类别分','100×[0.5(1−SSIM)+0.5 Global MAE]','↑ 全局外观影响更强'],
['视觉与区域影响','Target local /100','目标对象冻结 ROI 局部影响','100×Target ROI MAE','↑ 目标局部影响更强'],
['视觉与区域影响','Temporal appearance /100','全局与 ROI 的时序像素变化','100×[0.4 Global Δ-MAE+0.6 ROI Δ-MAE]','↑ 时序外观/形变/闪烁等综合差异更强'],
['视觉与区域影响','ROI mean area fraction','计算 ROI 或 outside 区域占全帧比例','mean_t |region(t)| / |frame|','用于审计指标支持域，非效果分'],
['轨迹与可跟踪性','Trajectory impact (% D0)','选中对象中心平均轨迹偏移，以 F00 bbox 对角线 D0 归一化','100×mean_selected_objects(Center-ADE / D0)','↑ 真实对象轨迹改变更强；质量门失败为 N/A'],
['轨迹与可跟踪性','Center-ADE','共同可见帧上对象中心距离的时间均值','mean_t ||c_abl(t)−c_base(t)||₂；并同时报告 /D0','↑ 整段轨迹差异更大'],
['轨迹与可跟踪性','Center-FDE','最后共同有效帧的对象中心距离','||c_abl(t*)−c_base(t*)||₂；并同时报告 /D0','↑ 最终运动结果偏移更大'],
['轨迹与可跟踪性','Velocity vector error','4 帧差分的中心速度向量误差','mean_t ||(c_abl(t+4)−c_abl(t))/4−(c_base(t+4)−c_base(t))/4||₂','↑ 运动速度/方向差异更大'],
['轨迹与可跟踪性','Point-ADE','共同可见 CoTracker 表面点的平均距离','mean_(t,p) ||p_abl(t)−p_base(t)||₂；并同时报告 /D0','↑ 表面点轨迹差异更大'],
['轨迹与可跟踪性','PCK@5/10/20% D0','点距离落在对象尺度阈值内的比例','mean_(t,p) 1[||p_abl−p_base||<αD0], α∈{.05,.10,.20}','↑ 与 Baseline 点轨迹更接近'],
['轨迹与可跟踪性','Common center coverage','Baseline 可跟踪中心帧中，消融后仍共同可跟踪的比例','N_common_center / N_baseline_center','↑ 可跟踪保留更好'],
['轨迹与可跟踪性','Track Loss /100','CoTracker 中心可观测性损失','100×(1−common center coverage)','↑ 更不可跟踪；不单独证明对象消失'],
['轨迹与可跟踪性','Trajectory quality gate','ADE/FDE 是否有足够共同中心帧与覆盖率','common frames≥min_valid_frames AND coverage≥min_coverage','未通过时 ADE/FDE 不参与排名'],
['对象存活与身份','F00 prompt IoU','SAM2 在首帧初始化的 mask 质量','IoU(mask_SAM2(F00), prompt/reference mask(F00))','↑ 存活评估初始化更可靠；<0.5 失败'],
['对象存活与身份','Survival rate','同时满足非空 mask、DINO 身份阈值、面积比在 [0.25,4] 的帧比例','mean_t 1[mask≠∅ AND cosine≥τ AND area ratio∈[.25,4]]','↑ 对象保留更好'],
['对象存活与身份','Retention / Disappearance /100','对象存活率及其补集','Retention=100×survival rate; Disappearance=100×(1−survival rate)','Retention ↑ 更好；Disappearance ↑ 保留失败更强'],
['对象存活与身份','DINO identity similarity','当前帧对象与 Baseline 同帧对象的 mask-pooled DINOv2 cosine','mean_t cosine(f_DINO(mask_abl), f_DINO(mask_base))','↑ 对象身份/语义外观更接近'],
['对象存活与身份','Identity failure rate','DINO cosine 低于校准身份阈值的帧比例','mean_t 1[cosine_t<τ_object]','↑ 身份/外观损坏更多'],
['对象存活与身份','Area failure rate','mask 面积比超出 [0.25,4] 的帧比例','mean_t 1[area_abl/area_base ∉ [.25,4]]','↑ 极端缩放/分割异常更多'],
['对象存活与身份','Mask absence /100','SAM2 输出空 mask 的帧比例','100×mean_t 1[mask_t=∅]','↑ 纯消失代理更强；仍可包含 SAM2 失败'],
['对象存活与身份','First sustained loss frame','首次连续至少 3 帧 not-alive 的起始帧','min t such that alive[t:t+3]=false','越早表示对象保留更早崩溃'],
['对象存活与身份','Terminal missing rate','最后 8 帧中 not-alive 的比例','mean_(t in last 8) 1[not alive_t]','↑ 视频末段对象保留更差'],
['对象存活与身份','Target mean / worst score','多对象 target 上存活失败或 mask absence 的聚合','mean_selected_objects(score), max_selected_objects(score)','↑ 平均/最差对象受影响更强']
];
function renderMetricDefinitions(){let group='';$('metricDefinitions').innerHTML=metricDefs.map(row=>{const section=row[0]!==group?(group=row[0],`<tr class="group"><th colspan="4">${esc(group)}</th></tr>`):'';return `${section}<tr><th>${esc(row[1])}</th><td>${esc(row[2])}</td><td>${esc(row[3])}</td><td>${esc(row[4])}</td></tr>`}).join('')}
const sampleKey=x=>`${x.case}::${x.seed}`,targetKey=r=>r.target_scope==='single_object'?`single_object::${r.region}`:'all_objects::',targetLabel=k=>k.startsWith('single_object::')?`Selected object · ${k.split('::')[1]}`:'All objects union',num=(v,d=3)=>Number.isFinite(Number(v))?Number(v).toFixed(d):'—';
function catalogUrl(caseName,seed){const u=new URL(api,location.origin);if(caseName){u.searchParams.set('case',caseName);u.searchParams.set('seed',seed)}if(q.get('single')==='1')u.searchParams.set('single','1');u.searchParams.set('v',Date.now());return u}
function videoTag(src){return `<video controls muted playsinline preload="none" data-src="${esc(src)}"></video>`}
function ensureVideo(video){if(video.dataset.src){video.src=video.dataset.src;delete video.dataset.src;video.load()}}
const lazyObserver='IntersectionObserver'in window?new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){ensureVideo(entry.target);lazyObserver.unobserve(entry.target)}}),{rootMargin:'360px 0px'}):null;
function armVideos(root=document){root.querySelectorAll('video[data-src]').forEach(video=>lazyObserver?lazyObserver.observe(video):ensureVideo(video))}
const rank=v=>Number.isFinite(Number(v))?`#${num(v,0)}`:'—',flag=v=>v==null?'—':v?'✓ 通过':'✗ 未通过',frame=v=>v==null?'—':`F${num(v,0)}`,metricTable=(title,rows)=>`<h4>${esc(title)}</h4><table><tbody>${rows.map(([label,value])=>`<tr><th>${esc(label)}</th><td>${esc(value)}</td></tr>`).join('')}</tbody></table>`;
function metricDetails(r){
 const m=r.metrics||{},a=m.appearance||{},t=m.trajectory||{},s=m.survival||{},has=a.impact_score_0_100!=null||Object.keys(t.objects||{}).length||Object.keys(s.objects||{}).length;
 if(!has)return '<details class="metric"><summary>指标待计算</summary><div class="metric-na">该 variant 暂无完整指标报告。</div></details>';
 const category=a.category_scores_0_100||{},categoryRanks=a.category_ranks||{},appearanceRows=[
  ['Composite Impact /100',`${num(a.impact_score_0_100)} (${rank(a.impact_rank)})`],
  ['Spillover /100',num(a.spillover_score_0_100)],
  ['Global appearance /100',`${num(category.global_appearance)} (${rank(categoryRanks.global_appearance)})`],
  ['Target local /100',`${num(category.target_local)} (${rank(categoryRanks.target_local)})`],
  ['Temporal appearance /100',`${num(category.temporal_appearance)} (${rank(categoryRanks.temporal_appearance)})`],
  ['Outside spillover /100',`${num(category.outside_spillover)} (${rank(categoryRanks.outside_spillover)})`],
  ['Global SSIM',num(a.global_ssim)],['Global PSNR (dB)',num(a.global_psnr_db)],
  ['Global MAE [0,1]',num(a.global_mae_0_1)],['Global Δ-MAE [0,1]',num(a.global_temporal_delta_mae_0_1)],
  [`${a.target_roi_key||'Target'} ROI MAE [0,1]`,num(a.target_roi_mae_0_1)],
  [`${a.target_roi_key||'Target'} ROI Δ-MAE [0,1]`,num(a.target_roi_temporal_delta_mae_0_1)],
  ['Target ROI mean area fraction',num(a.target_roi_mean_area_fraction)],
  ['Outside-object MAE [0,1]',num(a.outside_objects_mae_0_1)],
  ['Outside-object Δ-MAE [0,1]',num(a.outside_objects_temporal_delta_mae_0_1)],
  ['Outside mean area fraction',num(a.outside_objects_mean_area_fraction)]
 ];
 const trajectoryRows=[['Quality gate',flag(t.quality_pass)],['Selected objects',(t.selected_objects||[]).join(', ')||'—'],['Trajectory impact (% D0)',`${num(t.trajectory_impact_percent_d0)} (${rank(t.trajectory_rank)})`],['Target Center-ADE / D0',num(t.target_center_ade_norm)],['Mean Track Loss /100',num(t.target_mean_track_loss_score_0_100)],['Worst Track Loss /100',`${num(t.target_worst_track_loss_score_0_100)} (${rank(t.track_loss_rank)})`]];
 const trajectoryObjectTables=Object.entries(t.objects||{}).map(([name,o])=>{const p=o.pck_normalized||{},gated=value=>o.quality_pass?value:`N/A (raw ${value}; gate failed)`;return metricTable(`轨迹 · ${name}`,[['Quality gate',flag(o.quality_pass)],['Valid center frames',`${num(o.common_center_valid_frames,0)} / ${num(o.baseline_center_valid_frames,0)}`],['Common center coverage',num(o.common_center_coverage)],['Track retention /100',num(o.track_retention_score_0_100)],['Track Loss /100',num(o.track_loss_score_0_100)],['Last common visible frame',frame(o.last_common_visible_frame)],['Center-ADE (px / D0)',gated(`${num(o.center_ade_px)} / ${num(o.center_ade_norm)}`)],['Center-FDE (px / D0)',gated(`${num(o.center_fde_px)} / ${num(o.center_fde_norm)}`)],['Velocity vector error (px/frame / D0)',gated(`${num(o.velocity_vector_error_px_per_frame)} / ${num(o.velocity_vector_error_norm_per_frame)}`)],['Velocity valid count',num(o.velocity_valid_count,0)],['Point-ADE (px / D0)',gated(`${num(o.point_ade_px)} / ${num(o.point_ade_norm)}`)],['Point valid count',num(o.point_valid_count,0)],['PCK@5/10/20% D0',gated(`${num(p['0.05'])} / ${num(p['0.1'])} / ${num(p['0.2'])}`)]])}).join('');
 const survivalRows=[['Quality gate',flag(s.quality_pass)],['Selected objects',(s.selected_objects||[]).join(', ')||'—'],['Mean retention failure /100',num(s.target_mean_disappearance_score_0_100)],['Worst retention failure /100',`${num(s.target_worst_disappearance_score_0_100)} (${rank(s.disappearance_rank)})`],['Mean mask absence /100',num(s.target_mean_mask_absence_score_0_100)],['Worst mask absence /100',`${num(s.target_worst_mask_absence_score_0_100)} (${rank(s.mask_absence_rank)})`]];
 const survivalObjectTables=Object.entries(s.objects||{}).map(([name,o])=>metricTable(`存活/身份 · ${name}`,[['Quality gate',flag(o.quality_pass)],['F00 prompt IoU',num(o.f00_prompt_iou)],['Survival rate',num(o.survival_rate)],['Retention /100',num(o.retention_score_0_100)],['Disappearance /100',num(o.disappearance_score_0_100)],['DINO identity cosine',num(o.identity_similarity_mean)],['Identity failure rate',num(o.identity_failure_rate)],['Area failure rate',num(o.area_failure_rate)],['Empty-mask rate',num(o.empty_mask_rate)],['First sustained loss frame',frame(o.first_sustained_loss_frame)],['Terminal missing rate',num(o.terminal_missing_rate)],['Alive frames',`${num(o.alive_frame_count,0)} / ${num(o.frame_count,0)}`]])).join('');
 const summary=[`Impact ${num(a.impact_score_0_100)}`,`Trajectory ${num(t.trajectory_impact_percent_d0)}% D0`,`Track Loss ${num(t.target_worst_track_loss_score_0_100)}`,`Retention Failure ${num(s.target_worst_disappearance_score_0_100)}`,`Mask Absence ${num(s.target_worst_mask_absence_score_0_100)}`].join(' · ');
 return `<details class="metric"><summary>展开完整指标 · ${esc(summary)}</summary><div class="metric-body">${metricTable('视觉与区域影响 vs Baseline',appearanceRows)}${metricTable('轨迹与可跟踪性 · target 聚合',trajectoryRows)}${trajectoryObjectTables}${metricTable('对象存活与身份 · target 聚合',survivalRows)}${survivalObjectTables}</div></details>`
}
const comparisonMetrics=[['Trajectory impact %D0',m=>m.trajectory?.trajectory_impact_percent_d0],['Track Loss /100',m=>m.trajectory?.target_worst_track_loss_score_0_100],['Target local /100',m=>m.appearance?.category_scores_0_100?.target_local],['Global appearance /100',m=>m.appearance?.category_scores_0_100?.global_appearance],['Outside spillover /100',m=>m.appearance?.category_scores_0_100?.outside_spillover],['Retention Failure /100',m=>m.survival?.target_worst_disappearance_score_0_100]];
function snapshotDelta(r){if(r.ranking_tag!=='s039r3350')return '';const old=(data.selected.records||[]).find(x=>!x.ranking_tag&&x.target_scope===r.target_scope&&String(x.region||'')===String(r.region||'')&&x.mask_mode===r.mask_mode&&x.head_scope===r.head_scope);if(!old)return '<details class="metric"><summary>vs 原快照差异待配对</summary></details>';const rows=comparisonMetrics.map(([label,get])=>{const newer=Number(get(r.metrics||{})),older=Number(get(old.metrics||{}));if(!Number.isFinite(newer)||!Number.isFinite(older))return [label,'—'];const delta=newer-older,fold=Math.abs(older)>1e-12?newer/older:(Math.abs(newer)>1e-12?'∞':null);return [label,`${delta>=0?'+':''}${num(delta)} · ${fold===null?'—':fold==='∞'?'×∞':`×${num(fold,2)}`}`]});return `<details class="metric"><summary>latest3350 − 原快照 · 指标差值与倍数</summary><div class="metric-body">${metricTable('严格同配置配对',rows)}<p>正差值表示相对 Baseline 的该类影响更强；倍数 = latest3350 / 原快照。它不自动表示物理质量更差。</p></div></details>`}
function videoUrl(r){const s=data.selected,u=new URL(`${videoApi}/temporal-tube-ablation-video`,location.origin);u.searchParams.set('case',s.case);u.searchParams.set('seed',s.seed);u.searchParams.set('target_scope',r.target_scope);u.searchParams.set('mask_mode',r.mask_mode);u.searchParams.set('top_n',r.head_count);u.searchParams.set('head_scope',r.head_scope);if(r.ranking_tag)u.searchParams.set('ranking_tag',r.ranking_tag);if(r.region)u.searchParams.set('region',r.region);return u}
function card(r,op,time){const formula=`Y′_${op.target}(tq)=Y_${op.target}(tq)−Σ_{${time.predicate}} A[${op.target}_tq,${op.source}_tk]V_${op.source}(tk)`;return `<figure class="card">${videoTag(videoUrl(r))}<figcaption><strong>${esc(op.id)}-${esc(time.label)} · ${esc(scopeDisplay(r))}</strong><span>${r.head_count} layer-heads · ${esc(time.predicate)}</span><span class="formula"><b>切断：</b>${esc(op.flow)}<br><b>精确计算：</b>${esc(formula)}<br><b>诊断：</b>${esc(op.meaning)}</span></figcaption>${snapshotDelta(r)}${metricDetails(r)}</figure>`}
function render(){const s=data.selected,records=s.records||[],target=$('target').value,filtered=records.filter(r=>targetKey(r)===target),scopes=[...new Map(filtered.map(r=>[scopeKey(r),r])).values()].sort((a,b)=>scopeOrderValue(a)-scopeOrderValue(b)),base=new URL(`${videoApi}/video`,location.origin);base.searchParams.set('case',s.case);base.searchParams.set('seed',s.seed);const operators=ops.map(op=>{const scopeRows=scopes.map(scope=>{const cards=times.map(time=>{const r=filtered.find(x=>scopeKey(x)===scopeKey(scope)&&x.mask_mode===`${op.base}_${time.key}`);return r?.ready?card(r,op,time):''}).filter(Boolean);return cards.length?`<section class="scope"><div class="scope-head"><b>${esc(scopeDisplay(scope))}</b><span>${cards.length}/4 已生成</span></div><div class="grid">${cards.join('')}</div></section>`:''}).join('');return scopeRows?`<section class="operator"><h2><span>${esc(op.id)} · ${esc(op.flow)}</span><button type="button" class="operator-replay">重播 ${esc(op.id)} 板块</button></h2>${scopeRows}</section>`:''}).join('');const snapshots=(s.ranking_snapshots||[]).map(x=>`${x.ranking_label} ${x.ready}/${x.expected}`).join(' · '),baseline=s.baseline_ready?`<figure class="baseline">${videoTag(base)}<figcaption><b>Baseline · ${esc(s.case)} · seed ${s.seed} · No intervention</b><br>${esc(snapshots)}</figcaption></figure>`:'';$('content').innerHTML=`${baseline}${operators||'<div class="empty">当前 target 暂无已生成的 All-time/Same/Future/Past 视频</div>'}`;armVideos($('content'));syncUrl()}
function fillTargets(){const records=data.selected.records||[],keys=[...new Set(records.map(targetKey))];$('target').innerHTML=keys.map(k=>`<option value="${esc(k)}">${esc(targetLabel(k))}</option>`).join('');const wanted=q.get('target');if(keys.includes(wanted))$('target').value=wanted}
function syncUrl(){const s=data.selected,u=new URL(location.href);u.searchParams.set('case',s.case);u.searchParams.set('seed',s.seed);u.searchParams.set('target',$('target').value);history.replaceState(null,'',u)}
async function load(caseName,seed){$('status').textContent='读取批次…';const d=await fetch(catalogUrl(caseName,seed),{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`${r.status} ${r.statusText}`);return r.json()});if(!d.ready)throw new Error(d.reason||'catalog unavailable');data=d;$('sample').innerHTML=d.samples.map(x=>`<option value="${esc(sampleKey(x))}">${esc(x.case)} · seed ${x.seed} · ${x.ready}/${x.expected}</option>`).join('');$('sample').value=sampleKey(d.selected);fillTargets();render();const p=d.progress,selectedReady=(d.selected.records||[]).filter(r=>r.ready).length;$('status').textContent=`${q.get('single')==='1'?'独立入口':'全批次'} ${p.ready}/${p.expected} · 当前 ${selectedReady}/${(d.selected.records||[]).length} · errors ${p.errors}`}
function replayOperator(button){const section=button.closest('.operator'),videos=[...section.querySelectorAll('video')];videos.forEach(video=>{ensureVideo(video);const play=()=>{video.currentTime=0;video.play().catch(()=>{})};video.pause();if(video.readyState>=1)play();else video.addEventListener('loadedmetadata',play,{once:true})});const label=button.textContent;button.textContent=`已重播 ${videos.length} 个视频`;setTimeout(()=>{button.textContent=label},1200)}
$('sample').addEventListener('change',()=>{const value=$('sample').value,index=value.lastIndexOf('::');load(value.slice(0,index),value.slice(index+2)).catch(showError)});$('target').addEventListener('change',render);$('refresh').addEventListener('click',()=>load(data?.selected.case,data?.selected.seed).catch(showError));document.addEventListener('click',event=>{const button=event.target.closest('.operator-replay');if(button)replayOperator(button)});function showError(error){$('status').textContent=`加载失败：${error.message}`}
if(q.get('single')==='1'){document.title='001460 · Seed 47326 M1/M2/M3';document.querySelector('h1').innerHTML='001460 · Seed 47326<br>M1 / M2 / M3 · All-time / Same / Future / Past';document.querySelector('.lead').textContent='独立轻量入口：接口只返回当前 case/seed，每次只渲染一个 target，视频进入视口附近才加载。';$('sample').closest('label').hidden=true}renderMetricDefinitions();const initialCase=q.get('case')||'',initialSeed=q.get('seed')||'';load(initialCase,initialSeed).catch(showError);
</script></body></html>'''
    aggregate_style = r'''
.aggregate{margin:14px 0;padding:13px;border:1px solid var(--line);border-left:7px solid #a3452f;background:#fffaf0e8;box-shadow:0 12px 30px #58442b16}.aggregate h2,.aggregate h3{margin:0 0 8px}.aggregate h3{margin-top:18px}.aggregate p{line-height:1.55}.coverage-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:10px 0}.coverage-card{padding:10px;border:1px solid var(--line);background:#fff}.coverage-card b,.coverage-card span{display:block}.coverage-card b{font:900 24px ui-monospace,monospace;color:var(--deep)}.aggregate .table-scroll{max-height:52vh}.aggregate table{width:100%;border-collapse:collapse;background:#fff;font:12px/1.45 ui-monospace,monospace}.aggregate th,.aggregate td{padding:7px 8px;border:1px solid #d8cebd;text-align:right;white-space:nowrap}.aggregate th:first-child,.aggregate td:first-child{text-align:left}.aggregate thead th{position:sticky;top:0;background:#603327;color:#fff}.aggregate details{margin-top:10px}.aggregate details>summary{cursor:pointer;font-weight:900}.analysis-list{margin:8px 0;padding-left:22px;line-height:1.55}.caveat{padding:9px;border-left:5px solid var(--gold);background:#fff3d9}@media(max-width:1250px){.coverage-grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:680px){.coverage-grid{grid-template-columns:1fr 1fr}}
'''
    page = page.replace("</style>", aggregate_style + "</style>", 1)
    page = page.replace(
        '<main><section class="note">',
        '<main><section id="batchAggregate" class="aggregate" hidden></section><section class="note">',
        1,
    )
    aggregate_script = r'''
const aggregateModeLabel={self_only:'M1-All-time',self_same:'M1-Same',self_future:'M1-Future',self_past:'M1-Past',incoming_only:'M2-All-time',incoming_same:'M2-Same',incoming_future:'M2-Future',incoming_past:'M2-Past',outgoing_only:'M3-All-time',outgoing_same:'M3-Same',outgoing_future:'M3-Future',outgoing_past:'M3-Past'};
function renderBatchAggregate(a){
 const box=$('batchAggregate');if(q.get('single')==='1'||!a){box.hidden=true;return}box.hidden=false;
 const cfg=a.configured||{},cov=a.coverage||{},rows=a.appearance_by_mode||[],scopes=a.appearance_by_scope||[],trajectory=a.trajectory_by_mode||[],trajectoryScopes=a.trajectory_by_scope||[],mode=r=>aggregateModeLabel[r.mask_mode]||r.mask_mode,table=(headers,body)=>`<div class="table-scroll"><table><thead><tr>${headers.map(x=>`<th>${esc(x)}</th>`).join('')}</tr></thead><tbody>${body.map(row=>`<tr>${row.map(value=>`<td>${esc(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`,maxBy=(items,key)=>items.filter(x=>Number.isFinite(Number(key(x)))).sort((x,y)=>Number(key(y))-Number(key(x)))[0],pct=value=>Number.isFinite(Number(value))?`${num(100*Number(value),1)}%`:'—';
 const caseTable=table(['Case','配置 seeds','已有指标 seeds','生成视频','像素/ROI','轨迹','存活'],(a.cases||[]).map(r=>[r.case,`${r.configured_seeds.length} · ${r.configured_seeds.join(',')}`,`${r.appearance_seeds.length} · ${r.appearance_seeds.join(',')||'—'}`,`${r.generated_videos}/${r.expected_videos}`,r.appearance_records,r.trajectory_records,r.survival_records]));
 const appearanceTable=table(['消融','匹配单元','Case / Case-seed','Impact /100','Global appearance','Target local','Temporal appearance','Outside spillover'],rows.map(r=>[mode(r),r.matched_units,`${r.cases} / ${r.case_seeds}`,num(r.metrics.impact),num(r.metrics.global_appearance),num(r.metrics.target_local),num(r.metrics.temporal_appearance),num(r.metrics.outside_spillover)]));
 const rawTable=table(['消融','SSIM','Global MAE','Global Δ-MAE','ROI MAE','ROI Δ-MAE','Outside MAE','Outside Δ-MAE'],rows.map(r=>[mode(r),num(r.metrics.global_ssim),num(r.metrics.global_mae),num(r.metrics.global_delta_mae),num(r.metrics.target_roi_mae),num(r.metrics.target_roi_delta_mae),num(r.metrics.outside_mae),num(r.metrics.outside_delta_mae)]));
 const scopeTable=table(['Head scope','匹配单元','Case / Case-seed','Impact /100','Global appearance','Target local','Temporal appearance','Outside spillover'],scopes.map(r=>[scopeLabel[r.head_scope]||r.head_scope,r.matched_units,`${r.cases} / ${r.case_seeds}`,num(r.metrics.impact),num(r.metrics.global_appearance),num(r.metrics.target_local),num(r.metrics.temporal_appearance),num(r.metrics.outside_spillover)]));
 const trajectoryTable=table(['消融','N','ADE 通过','ADE %D0 (仅通过)','Track Loss /100','Retention Failure /100','Mask Absence /100'],trajectory.map(r=>[mode(r),r.records,`${r.ade_valid_records}/${r.records} · ${pct(r.ade_pass_rate)}`,num(r.trajectory_impact_percent_d0),num(r.track_loss),num(r.retention_failure),num(r.mask_absence)]));
 const trajectoryScopeTable=table(['Head scope','N','ADE 通过','ADE %D0 (仅通过)','Track Loss /100','Retention Failure /100','Mask Absence /100'],trajectoryScopes.map(r=>[scopeLabel[r.head_scope]||r.head_scope,r.records,`${r.ade_valid_records}/${r.records} · ${pct(r.ade_pass_rate)}`,num(r.trajectory_impact_percent_d0),num(r.track_loss),num(r.retention_failure),num(r.mask_absence)]));
 const impact=maxBy(rows,r=>r.metrics.impact),target=maxBy(rows,r=>r.metrics.target_local),temporal=maxBy(rows,r=>r.metrics.temporal_appearance),spill=maxBy(rows,r=>r.metrics.outside_spillover),scopeImpact=maxBy(scopes,r=>r.metrics.impact),track=maxBy(trajectory,r=>r.track_loss),retention=maxBy(trajectory,r=>r.retention_failure),mask=maxBy(trajectory,r=>r.mask_absence),lowestPass=trajectory.filter(r=>r.records).sort((x,y)=>Number(x.ade_pass_rate)-Number(y.ade_pass_rate))[0];
 const analysis=[impact?`综合可见影响最大：${mode(impact)} (${num(impact.metrics.impact)}/100)。`:'',target?`目标 ROI 局部影响最大：${mode(target)} (${num(target.metrics.target_local)}/100)。`:'',temporal?`时序外观影响最大：${mode(temporal)} (${num(temporal.metrics.temporal_appearance)}/100)；这是像素变化指标，不是纯轨迹。`:'',spill?`对象外 spillover 最大：${mode(spill)} (${num(spill.metrics.outside_spillover)}/100)。`:'',scopeImpact?`在三种 head scope 都齐全的匹配子集中，${scopeLabel[scopeImpact.head_scope]||scopeImpact.head_scope} 综合影响最大 (${num(scopeImpact.metrics.impact)}/100)。`:'',track?`单 case 轨迹报告中 Track Loss 最大：${mode(track)} (${num(track.track_loss)}/100)；Retention Failure 最大：${mode(retention)} (${num(retention.retention_failure)}/100)；Mask Absence 最大：${mode(mask)} (${num(mask.mask_absence)}/100)。`:'',lowestPass?`${mode(lowestPass)} 的 ADE 质量门通过率最低 (${pct(lowestPass.ade_pass_rate)})，其“仅通过样本 ADE 均值”存在选择偏差，不能单独判定影响小。`:'' ].filter(Boolean);
 box.innerHTML=`<h2>全批次平均指标 · 样本量先行</h2><div class="coverage-grid"><div class="coverage-card"><b>${cfg.cases||0}</b><span>配置 cases</span></div><div class="coverage-card"><b>${cfg.case_seeds||0}</b><span>配置 case-seed</span></div><div class="coverage-card"><b>${cov.generated_videos||0}/${cfg.expected_videos||0}</b><span>已生成视频</span></div><div class="coverage-card"><b>${cov.appearance_records||0}</b><span>像素/ROI 指标 · ${cov.appearance_cases||0} cases / ${cov.appearance_case_seeds||0} case-seeds</span></div><div class="coverage-card"><b>${cov.trajectory_records||0}</b><span>轨迹/存活 · ${cov.trajectory_cases||0} case / ${cov.trajectory_case_seeds||0} case-seed</span></div></div><p class="caveat"><b>口径：</b>只有 12 种消融都齐全的 ${cov.matched_appearance_units||0} 个 <code>case×seed×target×head-scope</code> 单元进入横向比较，覆盖 ${cov.matched_appearance_cases||0} cases / ${cov.matched_appearance_case_seeds||0} case-seeds。先在 seed 内平均，再在 case 内平均，最后 case 等权平均；缺失值不作 0。</p><h3>每个 case 的 seed 与指标覆盖</h3>${caseTable}<h3>12 种消融 · 匹配单元的 case-balanced 均值</h3>${appearanceTable}<details><summary>展开原始像素/ROI 均值</summary>${rawTable}</details><h3>Head scope 对照 · 只用三种 scope 都齐全的单元</h3>${scopeTable}<h3>轨迹、可跟踪性与对象存活</h3><p class="caveat">目前仅覆盖 <b>${cov.trajectory_cases||0} case / ${cov.trajectory_case_seeds||0} seed</b>（${cov.trajectory_records||0} 条），不是“所有 case 平均”。ADE 均值只统计通过质量门的记录；Track Loss、Retention Failure、Mask Absence 使用全部记录。</p>${trajectoryTable}<details><summary>展开轨迹/存活的 Head scope 均值</summary>${trajectoryScopeTable}</details><h3>当前结论</h3><ul class="analysis-list">${analysis.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`;
}
'''
    page = page.replace("function videoTag(src){", aggregate_script + "function videoTag(src){", 1)
    page = page.replace(
        "data=d;$('sample').innerHTML=",
        "data=d;renderBatchAggregate(d.aggregate);$('sample').innerHTML=",
        1,
    )
    return page


def wan22_ti2v_legacy_m1_temporal_gallery_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 Temporal Ablation Gallery</title><link rel="icon" href="data:"><style>
:root{--paper:#ece4d5;--ink:#17261f;--deep:#17443a;--line:#baad98;--card:#fffaf0;--gold:#d29c35;--rust:#a84f32}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#d65f3c3d,transparent 34rem),radial-gradient(circle at 100% 0,#27897935,transparent 40rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:15px 22px;background:#ece4d5f3;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:5px 0;font-size:clamp(30px,4.2vw,56px);line-height:1}.lead{max-width:1500px;margin:7px 0;line-height:1.5}.tools{display:flex;align-items:center;gap:9px;flex-wrap:wrap}button{padding:8px 11px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--deep);cursor:pointer;font-weight:900}.status,.mono{font:12px ui-monospace,SFMono-Regular,monospace}main{width:min(100% - 18px,2500px);margin:auto;padding:18px 0 70px}.scope-note,.mode-row{margin:14px 0;padding:14px;border:1px solid var(--line);border-radius:16px;background:#fffaf0e8;box-shadow:0 13px 34px #58442b16}.scope-note{border-left:7px solid var(--gold);line-height:1.55}.scope-note h2{margin:0 0 7px}.mode-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin:-14px -14px 12px;padding:12px 14px;background:var(--deep);color:#fff;border-radius:15px 15px 0 0}.mode-heading h2{margin:0}.mode-heading p{margin:5px 0 0;max-width:1500px;line-height:1.45}.mode-heading button{flex:0 0 auto}.video-strip{display:flex;flex-wrap:nowrap;gap:10px;overflow-x:auto;padding:2px 2px 17px;scrollbar-gutter:stable;scrollbar-color:var(--rust) #e8dfcc;overscroll-behavior-inline:contain}.video-strip figure{flex:0 0 clamp(310px,27vw,440px);margin:0;padding:8px;border:1px solid #d4c8b5;background:#fff;min-width:0}.video-strip video{display:block;width:100%;aspect-ratio:1280/704;background:#111}.video-strip figcaption{padding:8px 3px 2px;line-height:1.45}.video-strip figcaption strong,.video-strip figcaption span{display:block}.target{margin-top:5px;color:var(--rust);font-weight:900}.scope{font:12px ui-monospace,monospace;color:var(--deep)}.formula{margin-top:5px;font:11px/1.5 ui-monospace,monospace;color:#5e594f}.empty{display:grid;place-items:center;min-height:260px;background:#f0eadf;color:#766d60}.mode-heading button:focus-visible{outline:3px solid var(--gold);outline-offset:2px}@media(max-width:750px){header{position:static}.mode-heading{flex-direction:column}.video-strip figure{flex-basis:84vw}}
</style></head><body><header><a href="/">返回总览</a> · <a href="/wan22-ti2v-legacy-physiciq67-samples?v=15&case=0613pybullet_sample_001460_w002&seed=47326&region=object_A">返回完整消融页</a><h1>M1 Temporal Ablation<br>Same / Future / Past</h1><p class="lead"><span class="mono">case=0613pybullet_sample_001460_w002 · seed=47326</span><br>每行固定一个 M1 时间谓词，并在同一横向视频带中放入全部 target × head scope。</p><div class="tools"><button id="refresh" type="button">刷新</button><span id="status" class="status">读取中</span></div></header><main><section class="scope-note"><h2>固定比较范围</h2><div><b>Target：</b>Object A、Object B、all_objects　·　<b>Head scope：</b>Top100、Bottom100、All720　·　<b>共同设置：</b>Tube R、40 个去噪步、conditional/unconditional 两个 CFG 分支。</div></section><div id="rows"></div></main><script>
const api='/api/wan22-ti2v-legacy-physiciq67-samples',caseName='0613pybullet_sample_001460_w002',seed=47326,$=id=>document.getElementById(id),e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const modes=[{key:'self_same',id:'M1-Same',predicate:'tk=tq',flow:'同一 latent 时刻的 R K/V ──X──> R Query',formula:"Y′_R(tq)=Y_R(tq)−Σ_{tk=tq} A[R_tq,R_tk]V_R(tk)"},{key:'self_future',id:'M1-Future',predicate:'tk<tq',flow:'较早时刻的 R K/V ──X──> 较晚 R Query',formula:"Y′_R(tq)=Y_R(tq)−Σ_{tk<tq} A[R_tq,R_tk]V_R(tk)"},{key:'self_past',id:'M1-Past',predicate:'tk>tq',flow:'较晚时刻的 R K/V ──X──> 较早 R Query',formula:"Y′_R(tq)=Y_R(tq)−Σ_{tk>tq} A[R_tq,R_tk]V_R(tk)"}],targetOrder={'single_object:object_A':0,'single_object:object_B':1,'all_objects:':2},scopeOrder={top100:0,bottom100:1,all720:2},targetLabel=r=>r.target_scope==='single_object'?(r.region==='object_A'?'Object A':'Object B'):'all_objects',scopeLabel={top100:'Top100 PCK Heads',bottom100:'Bottom100 PCK Heads',all720:'All720 Heads'};
function videoUrl(r){const q=new URLSearchParams({case:caseName,seed:String(seed),target_scope:r.target_scope,mask_mode:r.mask_mode,top_n:String(r.head_count),head_scope:r.head_scope});if(r.region)q.set('region',r.region);return `${api}/temporal-tube-ablation-video?${q}`}
function card(r,mode){return `<figure><video controls muted playsinline preload="metadata" src="${videoUrl(r)}"></video><figcaption><strong>${e(mode.id)} · ${e(targetLabel(r))} · ${e(r.head_scope)}</strong><span class="target">Target：${e(targetLabel(r))}</span><span class="scope">${e(scopeLabel[r.head_scope]||r.head_scope)} · ${r.head_count} layer-heads</span><span class="formula">${e(mode.formula)}</span></figcaption></figure>`}
function replayRow(button){const row=button.closest('.mode-row'),videos=[...row.querySelectorAll('video')];videos.forEach(video=>{const play=()=>{video.currentTime=0;video.play().catch(()=>{})};video.pause();if(video.readyState>=1)play();else video.addEventListener('loadedmetadata',play,{once:true})});const label=button.textContent;button.textContent=`已重播 ${videos.length} 个视频`;setTimeout(()=>button.textContent=label,1200)}
function render(records){let readyTotal=0;$('rows').innerHTML=modes.map(mode=>{const rows=records.filter(r=>r.mask_mode===mode.key&&r.ready).sort((a,b)=>(targetOrder[`${a.target_scope}:${a.region||''}`]??99)-(targetOrder[`${b.target_scope}:${b.region||''}`]??99)||(scopeOrder[a.head_scope]??99)-(scopeOrder[b.head_scope]??99));readyTotal+=rows.length;return `<section class="mode-row"><div class="mode-heading"><div><h2>${e(mode.id)} · ${e(mode.predicate)}</h2><p><b>切断信息流：</b>${e(mode.flow)}<br><span class="mono">${e(mode.formula)}</span></p></div><button type="button" class="replay-row">重播本行</button></div><div class="video-strip">${rows.length?rows.map(r=>card(r,mode)).join(''):'<div class="empty">该行暂无已生成视频</div>'}</div></section>`}).join('');$('status').textContent=`${readyTotal}/27 videos ready · 3 rows × 9 target/scope combinations`}
async function load(){const data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),sample=(data.samples||[]).find(x=>x.case===caseName&&Number(x.seed)===seed),records=sample?.m123_head_scope_ablations?.records||[];render(records)}
document.addEventListener('click',event=>{const button=event.target.closest('.replay-row');if(button)replayRow(button)});$('refresh').addEventListener('click',load);load().catch(error=>{$('status').textContent=`加载失败：${error.message}`});
</script></body></html>'''


def wan22_ti2v_legacy_all720_ablation_gallery_page():
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All720 Object Query Ablation Gallery</title><link rel="icon" href="data:"><style>
:root{--paper:#e9e4d8;--ink:#17251f;--deep:#173f36;--line:#b9ad99;--card:#fffdf7;--gold:#cf9833;--rust:#a64c35;--green:#277565}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 2% 0,#d7613f3d,transparent 35rem),radial-gradient(circle at 98% 0,#2b8a7838,transparent 42rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:8;padding:15px 22px;background:#e9e4d8f3;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header a{color:var(--deep);font-weight:900}h1{margin:5px 0;font-size:clamp(30px,4.2vw,56px);line-height:1}.lead{max-width:1550px;margin:7px 0;line-height:1.5}.tools{display:flex;align-items:center;gap:9px;flex-wrap:wrap}button{padding:8px 11px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--deep);cursor:pointer;font-weight:900}.status,.mono{font:12px ui-monospace,SFMono-Regular,monospace}main{width:min(100% - 18px,1900px);margin:auto;padding:18px 0 70px}.scope-note,.operator{margin:14px 0;padding:14px;border:1px solid var(--line);border-radius:16px;background:#fffaf0e8;box-shadow:0 13px 34px #58442b16}.scope-note{border-left:7px solid var(--gold);line-height:1.55}.scope-note h2{margin:0 0 7px}.operator{padding-top:0;overflow:hidden}.operator-title{margin:0 -14px 14px;padding:13px 15px;background:var(--deep);color:#fff}.operator-title h2{margin:0}.operator-title p{margin:5px 0 0;line-height:1.45}.mode-row{margin:13px 0 21px;border:1px solid #d1c5b2;background:#f8f3e9}.mode-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;padding:10px 11px;background:#e2d7c5}.mode-heading h3{margin:0}.mode-heading p{margin:4px 0 0;line-height:1.45}.mode-heading button{flex:0 0 auto}.video-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;padding:10px}.video-grid figure{margin:0;padding:8px;border:1px solid #d4c8b5;background:#fff;min-width:0}.video-grid video{display:block;width:100%;aspect-ratio:1280/704;background:#111}.video-grid figcaption{padding:8px 3px 2px;line-height:1.45}.video-grid figcaption strong,.video-grid figcaption span{display:block}.target{margin-top:5px;color:var(--rust);font-weight:900}.scope{font:12px ui-monospace,monospace;color:var(--green)}.formula{margin-top:5px;font:11px/1.5 ui-monospace,monospace;color:#5e594f}.empty{display:grid;place-items:center;min-height:260px;background:#f0eadf;color:#766d60}.mode-heading button:focus-visible{outline:3px solid var(--gold);outline-offset:2px}@media(max-width:1050px){.video-grid{grid-template-columns:1fr}}@media(max-width:750px){header{position:static}.mode-heading{flex-direction:column}}
</style></head><body><header><a href="/">返回总览</a> · <a href="/object-query-m1-temporal-gallery?v=1">M1 时间消融页</a> · <a href="/wan22-ti2v-legacy-physiciq67-samples?v=15&case=0613pybullet_sample_001460_w002&seed=47326&region=object_A">完整消融页</a><h1>All720 Object Query<br>Temporal Ablations</h1><p class="lead"><span class="mono">case=0613pybullet_sample_001460_w002 · seed=47326 · head_scope=all720</span><br>完整展示 M1/M2/M3 的 All-time、Same、Future、Past；每行并排比较 Object A、Object B 和 all_objects。</p><div class="tools"><button id="refresh" type="button">刷新</button><span id="status" class="status">读取中</span></div></header><main><section class="scope-note"><h2>固定比较范围</h2><div><b>All720：</b>30 layers × 24 heads 全部参与消融。<b>Tube R：</b>由 Baseline 冻结轨迹在 latent t=0…12 的 token 联合集合构成。所有干预覆盖 40 个去噪步以及 conditional/unconditional 两个 CFG 分支。</div></section><div id="operators"></div></main><script>
const api='/api/wan22-ti2v-legacy-physiciq67-samples',caseName='0613pybullet_sample_001460_w002',seed=47326,$=id=>document.getElementById(id),e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const operators=[{id:'M1',base:'self',target:'R',source:'R',flow:'R K/V ──X──> R Query',meaning:'对象 tube 内部 Value 对对象 Query 的自支持与跨时传播'},{id:'M2',base:'incoming',target:'R',source:'C',flow:'C K/V ──X──> R Query',meaning:'环境、背景和其他对象 Value 向对象 Query 输入上下文'},{id:'M3',base:'outgoing',target:'C',source:'R',flow:'R K/V ──X──> C Query',meaning:'对象 Value 向环境及其他对象 Query 广播'}],times=[{suffix:'only',label:'All-time',predicate:'所有 tk'},{suffix:'same',label:'Same',predicate:'tk=tq'},{suffix:'future',label:'Future',predicate:'tk<tq'},{suffix:'past',label:'Past',predicate:'tk>tq'}],targetOrder={'single_object:object_A':0,'single_object:object_B':1,'all_objects:':2},targetLabel=r=>r.target_scope==='single_object'?(r.region==='object_A'?'Object A':'Object B'):'all_objects';
function modeName(op,time){return `${op.base}_${time.suffix}`}
function formula(op,time){return `Y′_${op.target}(tq)=Y_${op.target}(tq)−Σ_{${time.predicate}} A[${op.target}_tq,${op.source}_tk]V_${op.source}(tk)`}
function videoUrl(r){const q=new URLSearchParams({case:caseName,seed:String(seed),target_scope:r.target_scope,mask_mode:r.mask_mode,top_n:String(r.head_count),head_scope:'all720'});if(r.region)q.set('region',r.region);return `${api}/temporal-tube-ablation-video?${q}`}
function card(r,op,time){return `<figure><video controls muted playsinline preload="metadata" src="${videoUrl(r)}"></video><figcaption><strong>${e(op.id)}-${e(time.label)} · ${e(targetLabel(r))}</strong><span class="target">Target：${e(targetLabel(r))}</span><span class="scope">All720 Heads · 30 layers × 24 heads</span><span class="formula">${e(formula(op,time))}</span></figcaption></figure>`}
function replayRow(button){const row=button.closest('.mode-row'),videos=[...row.querySelectorAll('video')];videos.forEach(video=>{const play=()=>{video.currentTime=0;video.play().catch(()=>{})};video.pause();if(video.readyState>=1)play();else video.addEventListener('loadedmetadata',play,{once:true})});const label=button.textContent;button.textContent=`已重播 ${videos.length} 个视频`;setTimeout(()=>button.textContent=label,1200)}
function render(records){let readyTotal=0;$('operators').innerHTML=operators.map(op=>{const modeRows=times.map(time=>{const mode=modeName(op,time),rows=records.filter(r=>r.mask_mode===mode&&r.ready).sort((a,b)=>(targetOrder[`${a.target_scope}:${a.region||''}`]??99)-(targetOrder[`${b.target_scope}:${b.region||''}`]??99));readyTotal+=rows.length;return `<section class="mode-row"><div class="mode-heading"><div><h3>${e(op.id)}-${e(time.label)} · ${e(time.predicate)}</h3><p><b>切断：</b>${e(op.flow)}　<span class="mono">${e(formula(op,time))}</span></p></div><button type="button" class="replay-row">重播本行</button></div><div class="video-grid">${rows.length?rows.map(r=>card(r,op,time)).join(''):'<div class="empty">该行暂无已生成视频</div>'}</div></section>`}).join('');return `<section class="operator"><div class="operator-title"><h2>${e(op.id)} · ${e(op.flow)}</h2><p><b>诊断目标：</b>${e(op.meaning)}</p></div>${modeRows}</section>`}).join('');$('status').textContent=`${readyTotal}/36 All720 videos ready · 3 operators × 4 temporal modes × 3 targets`}
async function load(){const data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),sample=(data.samples||[]).find(x=>x.case===caseName&&Number(x.seed)===seed),records=(sample?.m123_head_scope_ablations?.records||[]).filter(r=>r.head_scope==='all720');render(records)}
document.addEventListener('click',event=>{const button=event.target.closest('.replay-row');if(button)replayRow(button)});$('refresh').addEventListener('click',load);load().catch(error=>{$('status').textContent=`加载失败：${error.message}`});
</script></body></html>'''


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
