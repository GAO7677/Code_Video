#!/usr/bin/env python3
"""Combined dashboard for head-zero and attention-noise benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


BENCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_metrics_test5"
)
UNIFIED_BENCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_metrics_test5"
)
UNIFIED_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_unified_steps40_frames49_test5"
)
HEAD_ZERO_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme_benchmark_test5_ready"
)
GEN_LOG_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_complete_logs"
)
METRICS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
    "wmreward",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "videophy2",
    "cosmos_reason1",
)
METRIC_LABELS = {
    "physics_iq_with_context": "PhysicsIQ +ctx",
    "physics_iq_without_context": "PhysicsIQ -ctx",
    "pmf_with_context": "PMF +ctx",
    "pmf_without_context": "PMF -ctx",
    "wmreward": "WMReward",
    "vbench_subject_consistency": "Subject",
    "vbench_background_consistency": "Background",
    "vbench_temporal_flickering": "Flicker",
    "vbench_motion_smoothness": "Motion",
    "vbench_dynamic_degree": "Dynamic",
    "vbench_aesthetic_quality": "Aesthetic",
    "vbench_imaging_quality": "Imaging",
    "videophy2": "VideoPhy2",
    "cosmos_reason1": "Cosmos",
}
MODEL_LABELS = {
    "wan22_baseline": "Wan2.2 Baseline",
    "wan_lora": "Wan+LoRA",
    "full_sa_no_object_step2500": "Full-SA no-object step-002500",
}


def expected_methods() -> list[dict[str, object]]:
    methods: list[dict[str, object]] = []
    for model in ("wan22_baseline", "wan_lora"):
        methods.append(
            {
                "name": f"{model}_original",
                "model": model,
                "variant": "Original",
                "alpha": None,
                "experiment": "Head Zero Ablation",
                "application_steps": "No ablation",
                "selection": "Control",
                "benchmark_root": str(HEAD_ZERO_ROOT),
            }
        )
        for count in (30, 100):
            for rank in ("top", "bottom"):
                methods.append(
                    {
                        "name": f"{model}_{rank}{count}_steps_00_40",
                        "model": model,
                        "variant": f"{rank.title()}{count}",
                        "alpha": None,
                        "experiment": "Head Zero Ablation",
                        "application_steps": "S00-S39 (all steps)",
                        "selection": "Static fixed heads",
                        "benchmark_root": str(HEAD_ZERO_ROOT),
                    }
                )
        for count in (30, 100):
            for rank in ("top", "bottom"):
                for start, end in ((0, 10), (10, 20), (20, 30), (30, 40)):
                    methods.append(
                        {
                            "name": f"{model}_{rank}{count}_steps_{start:02d}_{end:02d}",
                            "model": model,
                            "variant": f"{rank.title()}{count}",
                            "alpha": None,
                            "experiment": "Head Zero Ablation",
                            "application_steps": f"S{start:02d}-S{end - 1:02d}",
                            "selection": "Static fixed heads",
                            "benchmark_root": str(HEAD_ZERO_ROOT),
                        }
                    )
        for rank in ("top", "bottom"):
            methods.append(
                {
                    "name": f"{model}_adaptive_{rank}30_steps_00_40",
                    "model": model,
                    "variant": f"Adaptive {rank.title()}30",
                    "alpha": None,
                    "experiment": "Head Zero Ablation",
                    "application_steps": "S00-S39 (all steps)",
                    "selection": "Adaptive per-step heads",
                    "benchmark_root": str(HEAD_ZERO_ROOT),
                }
            )
    for model in ("wan22_baseline", "wan_lora"):
        methods.append(
            {
                "name": f"{model}_original",
                "model": model,
                "variant": "Original",
                "alpha": None,
                "experiment": "Legacy Attention Noise (40-step/25-frame)",
                "application_steps": "No perturbation",
                "selection": "Control",
                "benchmark_root": str(BENCH_ROOT),
            }
        )
        for count in (30, 100):
            for rank in ("top", "bottom"):
                for alpha in (0.9, 1.5):
                    methods.append(
                        {
                            "name": f"{model}_{rank}{count}_alpha{str(alpha).replace('.', 'p')}",
                            "model": model,
                            "variant": f"{rank.title()}{count}",
                            "alpha": alpha,
                            "experiment": "Legacy Attention Noise (40-step/25-frame)",
                            "application_steps": "S00-S39 (all steps)",
                            "selection": "Adaptive per-step heads",
                            "benchmark_root": str(BENCH_ROOT),
                        }
                    )
    for model in MODEL_LABELS:
        methods.append(
            {
                "name": f"{model}_original",
                "model": model,
                "variant": "Original",
                "alpha": None,
                "experiment": "Unified Attention Noise (40-step/49-frame)",
                "application_steps": "No perturbation",
                "selection": "Control",
                "benchmark_root": str(UNIFIED_BENCH_ROOT),
            }
        )
        for count in (30, 100):
            for rank in ("top", "bottom"):
                for alpha in (0.9, 1.5):
                    methods.append(
                        {
                            "name": f"{model}_{rank}{count}_alpha{str(alpha).replace('.', 'p')}",
                            "model": model,
                            "variant": f"{rank.title()}{count}",
                            "alpha": alpha,
                            "experiment": "Unified Attention Noise (40-step/49-frame)",
                            "application_steps": "S00-S39 (all steps)",
                            "selection": "Adaptive per-step heads",
                            "benchmark_root": str(UNIFIED_BENCH_ROOT),
                        }
                    )
    return methods


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def metric_value(metric: str, payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    preferred = {
        "wmreward": ("surprise", "score", "reward"),
        "videophy2": ("score", "joint_rate", "joint_score"),
    }.get(metric, ("score", f"{metric}_score", "overall_score", "final_score"))
    for key in preferred:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def metric_status(metric: str) -> str:
    status_dir = BENCH_ROOT / "status"
    for suffix in ("done", "failed", "running"):
        if (status_dir / f"{metric}.{suffix}").is_file():
            return suffix
    return "pending"


def generation_status() -> dict[str, object]:
    queues = []
    marker_names = (
        "gpu0.complete",
        "gpu1.complete",
        "gpu2.complete",
        "gpu3.complete",
        "gpu5.complete",
        "fullsa_alpha09_count30_gpu0.complete",
        "fullsa_alpha09_count100_gpu1.complete",
    )
    for marker_name in marker_names:
        queues.append(
            {
                "gpu": marker_name,
                "complete": (UNIFIED_OUTPUT_ROOT / "logs" / marker_name).is_file(),
            }
        )
    return {
        "queues": queues,
        "complete": all(bool(item["complete"]) for item in queues),
    }


def build_summary() -> dict[str, object]:
    rows = []
    for spec in expected_methods():
        method_dir = Path(str(spec["benchmark_root"])) / "methods" / str(spec["name"])
        case_jsons = sorted(
            path
            for path in method_dir.glob("*.json")
            if not path.name.startswith("eval_summary_")
            and path.name != "batch_manifest.json"
        )
        values: dict[str, dict[str, object]] = {}
        payloads = [read_json(path) for path in case_jsons]
        for metric in METRICS:
            numbers = [
                number
                for payload in payloads
                if (number := metric_value(metric, payload.get(metric))) is not None
            ]
            values[metric] = {
                "mean": sum(numbers) / len(numbers) if numbers else None,
                "count": len(numbers),
            }
        rows.append(
            {
                **spec,
                "model_label": MODEL_LABELS[str(spec["model"])],
                "case_count": len(case_jsons),
                "metrics": values,
            }
        )

    best: dict[str, float | None] = {}
    for metric in METRICS:
        candidates = [
            float(row["metrics"][metric]["mean"])
            for row in rows
            if row["metrics"][metric]["mean"] is not None
        ]
        best[metric] = max(candidates) if candidates else None

    best_by_experiment: dict[str, dict[str, float | None]] = {}
    for experiment in (
        "Head Zero Ablation",
        "Legacy Attention Noise (40-step/25-frame)",
        "Unified Attention Noise (40-step/49-frame)",
    ):
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        best_by_experiment[experiment] = {}
        for metric in METRICS:
            candidates = [
                float(row["metrics"][metric]["mean"])
                for row in experiment_rows
                if row["metrics"][metric]["mean"] is not None
            ]
            best_by_experiment[experiment][metric] = max(candidates) if candidates else None

    statuses = {metric: metric_status(metric) for metric in METRICS}
    return {
        "root": str(BENCH_ROOT),
        "generation": generation_status(),
        "prepared": (BENCH_ROOT / "PREPARED").is_file(),
        "prepare_failed": (BENCH_ROOT / "PREPARE_FAILED").is_file(),
        "metrics": list(METRICS),
        "metric_labels": METRIC_LABELS,
        "metric_status": statuses,
        "completed_metrics": sum(status == "done" for status in statuses.values()),
        "rows": rows,
        "best": best,
        "best_by_experiment": best_by_experiment,
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head Ablation Benchmark</title>
<style>
:root{--ink:#17251f;--muted:#66736d;--paper:#f4f0e6;--card:#fffdf7;--line:#d8d2c3;--teal:#126e65;--gold:#d99d27;--red:#b74b3f}
*{box-sizing:border-box}body{margin:0;color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif;background:radial-gradient(circle at 8% 0,#d7eadf 0,transparent 34rem),linear-gradient(135deg,#f6f2e8,#e9eee7);min-height:100vh}
header{padding:32px clamp(18px,4vw,64px) 18px}h1{font-family:"Iowan Old Style","Palatino Linotype",serif;font-size:clamp(30px,4vw,54px);line-height:1;margin:0 0 10px;letter-spacing:-.035em}.sub{color:var(--muted);max-width:900px}
.status-grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;padding:0 clamp(18px,4vw,64px) 22px}.status{background:rgba(255,253,247,.85);border:1px solid var(--line);border-radius:14px;padding:15px;box-shadow:0 10px 30px rgba(30,55,43,.06)}.status b{display:block;font-size:23px;margin-top:4px}.eyebrow{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.toolbar{position:sticky;top:0;z-index:8;display:flex;gap:12px;align-items:center;padding:12px clamp(18px,4vw,64px);background:rgba(244,240,230,.9);backdrop-filter:blur(12px);border-block:1px solid var(--line)}select,button{font:inherit;border:1px solid var(--line);border-radius:9px;background:var(--card);padding:8px 12px;color:var(--ink)}button{cursor:pointer;background:var(--teal);color:white;border-color:var(--teal)}#stamp{margin-left:auto;color:var(--muted);font-size:13px}
.table-wrap{margin:22px clamp(12px,2vw,32px) 60px;border:1px solid var(--line);border-radius:16px;overflow:auto;max-height:72vh;background:var(--card);box-shadow:0 16px 45px rgba(30,55,43,.09)}table{border-collapse:separate;border-spacing:0;min-width:2100px;width:100%;font-size:13px}th,td{padding:10px 9px;border-right:1px solid #e5dfd1;border-bottom:1px solid #e5dfd1;text-align:right;white-space:nowrap}thead th{position:sticky;top:0;z-index:4;background:#e5eee8;font-size:11px;letter-spacing:.035em}.left{text-align:left}.sticky{position:sticky;left:0;z-index:3;background:var(--card)}thead .sticky{z-index:6;background:#e5eee8}.model{font-weight:700}.metric small{display:block;color:var(--muted);font-size:10px}.best{background:#fff0bb!important;color:#654508;font-weight:800;box-shadow:inset 0 0 0 1px var(--gold)}.group-best{background:#dff1e9;color:#145a50;font-weight:700}.experiment-tag{display:inline-block;padding:4px 7px;border-radius:999px;background:#e8eee8;font-size:11px;font-weight:700}.pending{color:#999}.running{color:var(--teal)}.failed{color:var(--red)}.complete{color:var(--teal)}.legend{padding:0 clamp(18px,4vw,64px) 18px;color:var(--muted);font-size:13px}
@media(max-width:760px){.status-grid{grid-template-columns:repeat(2,1fr)}header{padding-top:24px}.toolbar{flex-wrap:wrap}#stamp{width:100%;margin:0}.table-wrap{max-height:68vh}}
</style>
</head>
<body>
<header><div class="eyebrow">20 cases / controlled head ablation</div><h1>PCK Extreme Ablation Benchmark</h1><div class="sub">Head Zero includes static S00-09, S10-19, S20-29, S30-39 and all-step interventions, plus Adaptive Top/Bottom30 with heads reselected at every denoising step. Legacy and Unified Noise use adaptive per-step head selection across S00-S39. Results refresh every 30 seconds.</div></header>
<section class="status-grid">
  <div class="status"><span class="eyebrow">Generation queues</span><b id="generation">-</b></div>
  <div class="status"><span class="eyebrow">Benchmark tree</span><b id="prepared">-</b></div>
  <div class="status"><span class="eyebrow">Metrics complete</span><b id="metricProgress">-</b></div>
  <div class="status"><span class="eyebrow">Method-case rows</span><b id="caseProgress">-</b></div>
</section>
<div class="toolbar"><label>Experiment <select id="experiment"><option value="all">All experiments</option><option value="Head Zero Ablation">Head Zero Ablation</option><option value="Legacy Attention Noise (40-step/25-frame)">Legacy Attention Noise (40-step/25-frame)</option><option value="Unified Attention Noise (40-step/49-frame)">Unified Attention Noise (40-step/49-frame)</option></select></label><label>Model <select id="model"><option value="all">All models</option><option value="wan22_baseline">Wan2.2 Baseline</option><option value="wan_lora">Wan+LoRA</option><option value="full_sa_no_object_step2500">Full-SA no-object</option></select></label><button id="refresh">Refresh now</button><span id="stamp"></span></div>
<div class="legend">Gold marks the global highest mean across both experiments; pale green marks the best result within one experiment group. WMReward displays mean surprise.</div>
<div class="table-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
<script>
let DATA=null;
const fmt=v=>v==null?'-':(Math.abs(v)>=10?v.toFixed(3):v.toFixed(4));
function render(){
  if(!DATA)return;
  const filter=document.querySelector('#model').value;
  const experiment=document.querySelector('#experiment').value;
  const done=DATA.generation.queues.filter(x=>x.complete).length;
  document.querySelector('#generation').textContent=`${done}/${DATA.generation.queues.length} complete`;
  document.querySelector('#prepared').textContent='91-method matrix';
  document.querySelector('#metricProgress').textContent=`${DATA.completed_metrics}/14`;
  const total=DATA.rows.reduce((n,r)=>n+r.case_count,0);
  document.querySelector('#caseProgress').textContent=`${total}/1820`;
  const status=m=>DATA.metric_status[m];
  document.querySelector('#thead').innerHTML='<tr><th class="left sticky">Experiment</th><th class="left">Model / condition</th><th class="left">Application steps / selection</th><th>Cases</th>'+DATA.metrics.map(m=>`<th title="${status(m)}">${DATA.metric_labels[m]}<br><span class="${status(m)}">${status(m)}</span></th>`).join('')+'</tr>';
  const rows=DATA.rows.filter(r=>(filter==='all'||r.model===filter)&&(experiment==='all'||r.experiment===experiment));
  document.querySelector('#tbody').innerHTML=rows.map(r=>{
    const condition=r.alpha==null?r.variant:`${r.variant} / a=${r.alpha}`;
    const cells=DATA.metrics.map(m=>{const x=r.metrics[m];const best=DATA.best[m];const groupBest=DATA.best_by_experiment[r.experiment][m];const isBest=x.mean!=null&&best!=null&&Math.abs(x.mean-best)<1e-10;const isGroupBest=x.mean!=null&&groupBest!=null&&Math.abs(x.mean-groupBest)<1e-10;return `<td class="metric ${isBest?'best':(isGroupBest?'group-best':'')}">${fmt(x.mean)}<small>n=${x.count}</small></td>`}).join('');
    return `<tr><td class="left sticky"><span class="experiment-tag">${r.experiment}</span></td><td class="left"><span class="model">${r.model_label}</span><br>${condition}</td><td class="left"><strong>${r.application_steps}</strong><br><small>${r.selection}</small></td><td>${r.case_count}</td>${cells}</tr>`;
  }).join('');
  document.querySelector('#stamp').textContent='Updated '+new Date().toLocaleTimeString();
}
async function load(){try{const response=await fetch('/api/pck-extreme-benchmark/summary?ts='+Date.now());DATA=await response.json();render()}catch(error){document.querySelector('#stamp').textContent='Refresh failed: '+error}}
document.querySelector('#refresh').onclick=load;document.querySelector('#model').onchange=render;document.querySelector('#experiment').onchange=render;load();setInterval(load,30000);
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, payload: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/pck-extreme-benchmark"}:
            self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/pck-extreme-benchmark/summary":
            payload = json.dumps(build_summary(), ensure_ascii=True).encode()
            self.send_bytes(payload, "application/json; charset=utf-8")
            return
        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving benchmark dashboard at http://127.0.0.1:{args.port}/pck-extreme-benchmark?v=1")
    server.serve_forever()


if __name__ == "__main__":
    main()
