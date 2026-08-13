#!/usr/bin/env python3
"""Live batch dashboard for the direct-attention multicase/multiseed pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/"
    "direct_attention_tv_v1"
)
GEN_ROOT = ROOT / "firstframe_ti2v" / "generations"
VBENCH_ROOT = ROOT / "vbench_multicase"
STATE_ROOT = ROOT / "multicase_pilot_state"
CASES = (
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_001455_w000",
    "0613pybullet_sample_000336_w001",
    "phyco_kubric_ball_wall_collision_2025-08-08_00ac15",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px",
)
CASE_LABELS = {
    "0613pybullet_sample_001460_w002": "PyBullet 001460",
    "0613pybullet_sample_001455_w000": "PyBullet 001455",
    "0613pybullet_sample_000336_w001": "PyBullet 000336",
    "phyco_kubric_ball_wall_collision_2025-08-08_00ac15": "Kubric wall collision",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px": "Physics-IQ solid mechanics",
}
TARGETS = {
    "0613pybullet_sample_001460_w002": "object_A",
    "0613pybullet_sample_001455_w000": "object_A",
    "0613pybullet_sample_000336_w001": "object_B",
    "phyco_kubric_ball_wall_collision_2025-08-08_00ac15": "object_B",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px": "object_B",
}
SEEDS = (13248, 47326, 90094)
GROUPS = ("top100", "bottom100", "random100")
DIRECTIONS = ("context_to_future", "future_to_context", "bidirectional")
GROUP_LABELS = {
    "top100": "Top100",
    "bottom100": "Bottom100",
    "random100": "Random100",
}
DIRECTION_LABELS = {
    "context_to_future": "Context Q → Future K",
    "future_to_context": "Future Q → Context K",
    "bidirectional": "Bidirectional",
}
VBENCH = (
    ("vbench_subject_consistency", "Subject"),
    ("vbench_background_consistency", "Background"),
    ("vbench_temporal_flickering", "Flicker"),
    ("vbench_motion_smoothness", "Smoothness"),
    ("vbench_dynamic_degree", "Dynamic"),
    ("vbench_aesthetic_quality", "Aesthetic"),
    ("vbench_imaging_quality", "Imaging"),
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        if number == number and abs(number) != float("inf"):
            return number
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _variant(group: str, direction: str, target: str) -> str:
    return "baseline" if group == "baseline" else f"{group}__{direction}__{target}"


def _config_id(group: str, direction: str) -> str:
    return "baseline" if group == "baseline" else f"{group}__{direction}"


def _configs() -> list[dict[str, str]]:
    values = [{"id": "baseline", "group": "baseline", "direction": "baseline", "label": "Baseline"}]
    values.extend(
        {
            "id": _config_id(group, direction),
            "group": group,
            "direction": direction,
            "label": f"{GROUP_LABELS[group]} · {DIRECTION_LABELS[direction]}",
        }
        for group in GROUPS
        for direction in DIRECTIONS
    )
    return values


def _vbench_result(case: str, seed: int, variant: str) -> dict[str, Any]:
    identity = f"{case}:{seed}:{variant}"
    name = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
    return _json(VBENCH_ROOT / "index" / name / "result.json")


def _trajectory(run_dir: Path, target: str) -> dict[str, Any] | None:
    for row in _json(run_dir / "trajectory_metrics.json").get("metrics", []):
        if isinstance(row, dict) and str(row.get("target")) == target:
            return row
    return None


def _record(case: str, seed: int, config: dict[str, str]) -> dict[str, Any]:
    target = TARGETS[case]
    variant = _variant(config["group"], config["direction"], target)
    run_dir = GEN_ROOT / case / f"seed_{seed:05d}" / variant
    video = run_dir / "generated.mp4"
    complete = (run_dir / "complete.json").is_file() and video.is_file()
    metric = _trajectory(run_dir, target)
    vbench_payload = _vbench_result(case, seed, variant)
    scores = {}
    for field, _label in VBENCH:
        value = vbench_payload.get(field)
        scores[field] = _finite(value.get("score") if isinstance(value, dict) else None)
    source_video = _json(run_dir / "manifest.json").get("source_video")
    return {
        "case": case,
        "case_label": CASE_LABELS[case],
        "seed": seed,
        "target": target,
        "config": config["id"],
        "variant": variant,
        "complete": complete,
        "metric_ready": metric is not None,
        "metric": metric,
        "vbench": scores,
        "vbench_ready": sum(value is not None for value in scores.values()),
        "source_ready": isinstance(source_video, str) and Path(source_video).is_file(),
    }


def _case_balanced(records: list[dict[str, Any]], getter: Any) -> tuple[float | None, int, int]:
    case_values: list[float] = []
    unit_count = 0
    for case in CASES:
        values = []
        for row in records:
            if row["case"] != case:
                continue
            value = getter(row)
            if value is not None:
                values.append(value)
        unit_count += len(values)
        if values:
            case_values.append(sum(values) / len(values))
    return _mean(case_values), len(case_values), unit_count


def _aggregate(config: dict[str, str], records: list[dict[str, Any]], all_records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_getters = {
        "ade_d0": lambda row: _finite((row.get("metric") or {}).get("ade_d0")),
        "fde_d0": lambda row: _finite((row.get("metric") or {}).get("fde_d0")),
        "pck_10pct_d0": lambda row: _finite((row.get("metric") or {}).get("pck_10pct_d0")),
        "track_loss": lambda row: _finite((row.get("metric") or {}).get("future_track_loss_score_0_100")),
        "gate_pass": lambda row: 1.0 if (row.get("metric") or {}).get("quality_pass") is True else (0.0 if row.get("metric_ready") else None),
    }
    values: dict[str, Any] = {}
    for key, getter in metric_getters.items():
        mean, cases, units = _case_balanced(records, getter)
        values[key] = mean
        values[f"{key}_cases"] = cases
        values[f"{key}_units"] = units
    for field, _label in VBENCH:
        mean, cases, units = _case_balanced(records, lambda row, field=field: row["vbench"].get(field))
        values[field] = mean
        values[f"{field}_cases"] = cases
        values[f"{field}_units"] = units

    paired_by_case: list[float] = []
    paired_units = 0
    if config["id"] != "baseline":
        baseline = {(row["case"], row["seed"]): row for row in all_records if row["config"] == "baseline"}
        for case in CASES:
            differences: list[float] = []
            for row in records:
                if row["case"] != case:
                    continue
                base = baseline.get((case, row["seed"]))
                left = metric_getters["ade_d0"](row)
                right = metric_getters["ade_d0"](base) if base is not None else None
                if left is not None and right is not None:
                    differences.append(left - right)
            paired_units += len(differences)
            if differences:
                paired_by_case.append(sum(differences) / len(differences))
    values["delta_ade_d0"] = _mean(paired_by_case)
    values["delta_ade_cases"] = len(paired_by_case)
    values["delta_ade_units"] = paired_units
    return {
        **config,
        "planned": len(CASES) * len(SEEDS),
        "generated": sum(int(row["complete"]) for row in records),
        "trajectory_ready": sum(int(row["metric_ready"]) for row in records),
        "vbench_ready": sum(row["vbench_ready"] for row in records),
        "vbench_total": len(records) * len(VBENCH),
        **values,
    }


def _winner(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in aggregates if row["id"] == "baseline")
    all_ready = all(
        row["trajectory_ready"] == row["planned"] and row["vbench_ready"] == row["vbench_total"]
        for row in aggregates
    )
    if not all_ready:
        return {"ready": False, "label": "Pending", "reason": "等待 150 个轨迹结果与 1050 个 VBench 分数全部完成"}
    candidates = []
    for row in aggregates:
        if row["id"] == "baseline" or row["delta_ade_d0"] is None:
            continue
        gate_ok = row["gate_pass"] is not None and baseline["gate_pass"] is not None and row["gate_pass"] >= baseline["gate_pass"] - 0.05
        quality_ok = all(
            row[field] is not None and baseline[field] is not None and row[field] >= baseline[field] - 0.02
            for field in (
                "vbench_subject_consistency",
                "vbench_background_consistency",
                "vbench_imaging_quality",
            )
        )
        if gate_ok and quality_ok:
            candidates.append(row)
    if not candidates:
        return {"ready": True, "label": "No eligible winner", "reason": "没有配置同时通过轨迹 gate 与 VBench 质量 guardrail"}
    winner = min(candidates, key=lambda row: row["delta_ade_d0"])
    return {
        "ready": True,
        "id": winner["id"],
        "label": winner["label"],
        "reason": "在不明显降低轨迹 gate、Subject、Background、Imaging 的配置中，paired ΔADE/D0 最低",
    }


def catalog() -> dict[str, Any]:
    configs = _configs()
    records = [
        _record(case, seed, config)
        for case in CASES
        for seed in SEEDS
        for config in configs
    ]
    aggregates = [
        _aggregate(config, [row for row in records if row["config"] == config["id"]], records)
        for config in configs
    ]
    states = {
        name: (STATE_ROOT / f"{name}.done").is_file()
        for name in ("gpu1", "gpu2", "vbench")
    }
    states.update({
        f"{name}_failed": (STATE_ROOT / f"{name}.failed").is_file()
        for name in ("gpu1", "gpu2", "vbench")
    })
    return {
        "cases": [{"id": case, "label": CASE_LABELS[case], "target": TARGETS[case]} for case in CASES],
        "seeds": list(SEEDS),
        "configs": configs,
        "records": records,
        "aggregates": aggregates,
        "winner": _winner(aggregates),
        "vbench": [{"id": field, "label": label} for field, label in VBENCH],
        "summary": {
            "planned_videos": len(records),
            "generated_videos": sum(int(row["complete"]) for row in records),
            "trajectory_records": sum(int(row["metric_ready"]) for row in records),
            "planned_vbench_scores": len(records) * len(VBENCH),
            "vbench_scores": sum(row["vbench_ready"] for row in records),
        },
        "state": states,
        "selection_rule": (
            "Case 为最高独立单位；seed 先在 case 内平均。主指标为 paired ΔGT Center-ADE/D0。"
            "配置必须满足 trajectory gate pass rate 不低于 Baseline−5pp，且 VBench Subject、"
            "Background、Imaging 均不低于 Baseline−0.02，才参与最优配置选择。"
        ),
    }


def asset(kind: str, case: str, seed: str, config_id: str) -> Path | None:
    configs = _configs()
    if case not in CASES or config_id not in {config["id"] for config in configs}:
        return None
    try:
        seed_value = int(seed)
    except ValueError:
        return None
    if seed_value not in SEEDS:
        return None
    config = next(item for item in configs if item["id"] == config_id)
    variant = _variant(config["group"], config["direction"], TARGETS[case])
    run_dir = GEN_ROOT / case / f"seed_{seed_value:05d}" / variant
    if kind == "video":
        path = run_dir / "generated.mp4"
    elif kind == "source":
        value = _json(run_dir / "manifest.json").get("source_video")
        path = Path(value) if isinstance(value, str) else Path("/nonexistent")
    else:
        return None
    return path if path.is_file() else None


def page() -> str:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Direct Attention · Multicase Pilot</title><style>
:root{--ink:#18241f;--paper:#eee9dc;--card:#fffdf8;--line:#bcb29f;--green:#176a58;--rust:#b74c32;--blue:#285f87;--pending:#eee4d0}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 0 0,#e58d4d33,transparent 34rem),radial-gradient(circle at 100% 0,#39836b33,transparent 40rem),var(--paper);font:15px/1.5 "Avenir Next","Segoe UI",sans-serif}a{color:var(--blue)}header{padding:28px clamp(18px,5vw,72px);border-bottom:1px solid var(--line);background:#f5f1e8e8}h1{margin:8px 0;font:800 clamp(38px,6vw,76px)/.94 "Arial Narrow",sans-serif;letter-spacing:-.04em}.eyebrow{font:800 11px monospace;letter-spacing:.14em;color:var(--rust)}main{max-width:2100px;margin:auto;padding:22px clamp(12px,3vw,48px) 70px}.summary{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:9px}.stat,.panel{background:var(--card);border:1px solid var(--line);box-shadow:0 14px 36px #463b2b12}.stat{padding:13px;border-top:6px solid var(--green)}.stat b{display:block;font:800 28px "Arial Narrow",sans-serif}.panel{margin-top:16px;padding:17px}.panel h2{margin:0 0 8px;font:750 28px "Arial Narrow",sans-serif}.winner{border-left:8px solid var(--rust);padding:16px;background:#fff7e9}.tools{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin:12px 0}.tools select,.tools button{display:block;margin-top:4px;padding:8px 10px;border:1px solid var(--line);background:#fff;font:inherit}.tools button{background:var(--ink);color:#fff}.table{overflow:auto}table{border-collapse:collapse;min-width:1850px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;border-bottom:1px solid #ddd5c6;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--card)}thead th{font:700 10px monospace;text-transform:uppercase;color:#5e6b65}.best{background:#e4f2e9}.pending{background:var(--pending);color:#826f50}.videos{display:grid;grid-template-columns:repeat(2,minmax(340px,1fr));gap:10px}.video{border:1px solid var(--line);background:#fff}.video video{display:block;width:100%;aspect-ratio:16/9;background:#15201c}.video .copy{padding:10px}.video .copy b{display:block}.placeholder{display:grid;place-items:center;width:100%;aspect-ratio:16/9;background:#18251f;color:#f0c680;font:800 12px monospace}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:7px;font:11px monospace;color:#50625a}.note{padding:12px 15px;border-left:6px solid #d5a436;background:#fff7de}.state{display:flex;gap:7px;flex-wrap:wrap}.pill{padding:6px 9px;border-radius:99px;background:#fff;border:1px solid var(--line);font:11px monospace}.ok{color:var(--green)}.bad{color:#b53e32}@media(max-width:900px){.summary{grid-template-columns:1fr 1fr}.videos{grid-template-columns:1fr}}@media(max-width:560px){.summary{grid-template-columns:1fr}}
</style></head><body><header><a href="/">返回总入口</a><div class="eyebrow">51 / DIRECT ATTENTION MULTICASE PILOT</div><h1>5 CASE × 3 SEED<br>配置选择台</h1><p>Baseline + Top100 / Bottom100 / Random100 × 三种信息流方向；轨迹以 source GT/pseudo-GT CoTracker 为参考，VBench 为生成质量 guardrail。</p><div id="state" class="state"></div></header><main><div id="summary" class="summary"></div><section class="panel"><div id="winner" class="winner"></div><p id="rule" class="note"></p></section><section class="panel"><h2>Case-balanced 配置排行</h2><p>所有指标均先在同一 case 内平均 seed，再让 case 等权。ADE/FDE/TrackLoss 越低越好；PCK 与七项 VBench 越高越好。</p><div class="table"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div></section><section class="panel"><h2>逐 case / seed 视频</h2><div class="tools"><label>Case<select id="case"></select></label><label>Seed<select id="seed"></select></label><button id="refresh">刷新落盘状态</button><span id="updated"></span></div><div id="videos" class="videos"></div></section></main><script>
const api='/api/gt-stc-direct-attention-multicase',E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),F=(v,n=3)=>v==null?'—':Number(v).toFixed(n);let D;const $=id=>document.getElementById(id);
function asset(kind,row){return `${api}/asset?kind=${kind}&case=${encodeURIComponent(row.case)}&seed=${row.seed}&config=${encodeURIComponent(row.config)}`}
function render(){const s=D.summary;$('summary').innerHTML=[['Planned videos',s.planned_videos],['Generated',s.generated_videos],['Trajectory',s.trajectory_records],['VBench scores',`${s.vbench_scores}/${s.planned_vbench_scores}`],['Independent cases',D.cases.length]].map(x=>`<div class="stat"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('');$('state').innerHTML=Object.entries(D.state).map(([k,v])=>`<span class="pill ${v?(k.includes('failed')?'bad':'ok'):''}">${E(k)}: ${v?'YES':'NO'}</span>`).join('');$('winner').innerHTML=`<span class="eyebrow">PRE-REGISTERED WINNER</span><h2>${E(D.winner.label)}</h2><p>${E(D.winner.reason)}</p>`;$('rule').textContent=D.selection_rule;renderTable();renderVideos()}
function renderTable(){const heads=['Config','Generated','Gate pass','ΔADE/D0','ADE/D0','FDE/D0','PCK10','TrackLoss',...D.vbench.map(x=>x.label)];$('thead').innerHTML='<tr>'+heads.map(x=>`<th>${E(x)}</th>`).join('')+'</tr>';const readyWinner=D.winner.id;$('tbody').innerHTML=D.aggregates.map(r=>`<tr class="${r.id===readyWinner?'best':''}"><td><b>${E(r.label)}</b><br><small>${r.trajectory_ready}/${r.planned} trajectory · ${r.vbench_ready}/${r.vbench_total} VBench</small></td><td>${r.generated}/${r.planned}</td><td>${r.gate_pass==null?'—':F(100*r.gate_pass,1)+'%'}</td><td>${F(r.delta_ade_d0)}</td><td>${F(r.ade_d0)}</td><td>${F(r.fde_d0)}</td><td>${F(r.pck_10pct_d0)}</td><td>${F(r.track_loss,1)}</td>${D.vbench.map(m=>`<td>${F(r[m.id],4)}</td>`).join('')}</tr>`).join('')}
function metric(row){const m=row.metric||{};return `<div class="metrics"><span>Gate ${m.quality_pass===true?'PASS':row.metric_ready?'FAIL':'Pending'}</span><span>ADE ${F(m.ade_d0)}</span><span>FDE ${F(m.fde_d0)}</span><span>PCK10 ${F(m.pck_10pct_d0)}</span><span>TrackLoss ${F(m.future_track_loss_score_0_100,1)}</span><span>VBench ${row.vbench_ready}/7</span></div>`}
function renderVideos(){if(!D)return;const caseId=$('case').value||D.cases[0].id,seed=Number($('seed').value||D.seeds[0]);const rows=D.records.filter(r=>r.case===caseId&&r.seed===seed);$('videos').innerHTML=rows.map(r=>`<article class="video">${r.complete?`<video controls preload="metadata" playsinline loop src="${asset('video',r)}"></video>`:`<div class="placeholder">PENDING · GENERATION</div>`}<div class="copy"><b>${E(D.configs.find(c=>c.id===r.config)?.label)}</b><span>${E(r.case_label)} · seed ${r.seed} · ${E(r.target)}</span>${metric(r)}</div></article>`).join('')}
async function load(){D=await fetch(`${api}/catalog`,{cache:'no-store'}).then(r=>r.json());if(!$('case').options.length){$('case').innerHTML=D.cases.map(x=>`<option value="${E(x.id)}">${E(x.label)} · ${E(x.target)}</option>`).join('');$('seed').innerHTML=D.seeds.map(x=>`<option value="${x}">${x}</option>`).join('')}render();$('updated').textContent=new Date().toLocaleTimeString()}$('case').addEventListener('change',renderVideos);$('seed').addEventListener('change',renderVideos);$('refresh').addEventListener('click',load);load();setInterval(load,30000);
</script></body></html>'''
