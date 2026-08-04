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
    "top30_steps_00_40",
    "bottom30_steps_00_40",
    "top100_steps_00_40",
    "bottom100_steps_00_40",
)

METHOD_LABELS = {
    "original": "Baseline / Original",
    "top30_steps_00_40": "Top30 zero, S00-39",
    "bottom30_steps_00_40": "Bottom30 zero, S00-39",
    "top100_steps_00_40": "Top100 zero, S00-39",
    "bottom100_steps_00_40": "Bottom100 zero, S00-39",
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


METRICS_PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCK Head 消融指标</title><style>
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


PORTAL_CARD = r'''
<a class="card new" href="/pck-extreme-benchmark?v=1"><div><span>10 / 消融指标</span><h2>Top / Bottom Head 指标表</h2><p>汇总 Baseline 与 LoRA 的 Original、Top/Bottom 30 和 Top/Bottom 100 全步置零结果，实时标注模型内最佳值。</p></div><span class="go">打开消融指标表</span></a>
'''
viewer.PORTAL = viewer.PORTAL.replace("</section>", PORTAL_CARD + "</section>", 1)


class MetricsHandler(viewer.Handler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/pck-extreme-benchmark":
            self.send_payload(METRICS_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/pck-extreme-benchmark/summary":
            payload = json.dumps(benchmark_summary(), ensure_ascii=False).encode("utf-8")
            self.send_payload(payload, "application/json; charset=utf-8")
            return
        super().do_GET()


viewer.Handler = MetricsHandler


if __name__ == "__main__":
    viewer.main()
