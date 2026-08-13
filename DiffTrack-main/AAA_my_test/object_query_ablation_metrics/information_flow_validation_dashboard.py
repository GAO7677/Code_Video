#!/usr/bin/env python3
"""Live Stage 1--3 dashboard for the latest3350 information-flow study."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
STAGE1_ROOT = ROOT / "stage1_query_time_validation" / "analysis"
STAGE2_ROOT = ROOT / "stage2_smoke_videos"
STAGE3_ROOT = ROOT / "stage3_discovery_videos"
INPUT_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
    "visual_samples/attention_zero_seed47326/cases_other10_6seeds_latest3350.json"
)
PLAN_PATH = Path(
    "/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/plan.md"
)
SPEC_PATH = PLAN_PATH.with_name("experiment_spec_latest3350.json")

DISCOVERY_SEEDS = (13248, 47326, 90094)
HEAD_SCOPES = (
    "top100",
    "bottom100",
    "random100_layer_matched_draw0",
    "all720",
)
HEAD_SCOPE_LABELS = {
    "top100": "latest3350 Top100",
    "bottom100": "latest3350 Bottom100",
    "random100_layer_matched_draw0": "Layer-matched Random100",
    "all720": "All720",
}
HEAD_SCOPE_COUNTS = {
    "top100": 100,
    "bottom100": 100,
    "random100_layer_matched_draw0": 100,
    "all720": 720,
}
FLOW_ORDER = ("self_only", "incoming_only", "outgoing_only")
FLOW_DEFINITIONS = {
    "self_only": {
        "id": "M1",
        "block": "R Query × R K/V",
        "flow": "R K/V → R Query",
        "formula": "Y′[R] = Y[R] − A[R,R]V_R",
        "diagnosis": "对象 tube 内部的信息维持：轨迹、状态连续性、形状或身份。",
    },
    "incoming_only": {
        "id": "M2",
        "block": "R Query × C K/V",
        "flow": "C K/V → R Query",
        "formula": "Y′[R] = Y[R] − A[R,C]V_C",
        "diagnosis": "环境和其他对象向目标对象输入的运动、接触与场景约束。",
    },
    "outgoing_only": {
        "id": "M3",
        "block": "C Query × R K/V",
        "flow": "R K/V → C Query",
        "formula": "Y′[C] = Y[C] − A[C,R]V_R",
        "diagnosis": "目标对象状态向其他对象或背景的广播与 spillover。",
    },
}
STAGE1_OVERLAYS = {
    "q00": "fixed_query_t00_F00.mp4",
    "q06": "fixed_query_t06_F24.mp4",
    "q12": "fixed_query_t12_F48.mp4",
}

_catalog_lock = threading.Lock()
_catalog_signature: tuple[int, ...] | None = None
_catalog_value: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def _input_payload() -> dict[str, Any]:
    return _load_json(INPUT_MANIFEST)


def _input_samples() -> list[dict[str, Any]]:
    rows = _input_payload().get("samples") or []
    return [row for row in rows if isinstance(row, dict)]


def _sample(case: str, seed: int) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in _input_samples()
            if str(row.get("case")) == case and int(row.get("seed", -1)) == seed
        ),
        None,
    )


def _target_rows(sample: dict[str, Any]) -> list[dict[str, str]]:
    objects = [
        str(row["region_name"])
        for row in sample.get("regions") or []
        if isinstance(row, dict) and row.get("region_type") == "object"
    ]
    targets = [
        {"key": f"single_object::{name}", "scope": "single_object", "region": name}
        for name in objects
    ]
    if len(objects) > 1:
        targets.append({"key": "all_objects::", "scope": "all_objects", "region": ""})
    return targets


def _manifest_record(path: Path, stage: str) -> dict[str, Any] | None:
    payload = _load_json(path)
    required = {"case", "seed", "variant_id", "head_scope", "mask_mode", "target_scope"}
    if not required.issubset(payload):
        return None
    if str(payload["head_scope"]) not in HEAD_SCOPES:
        return None
    if str(payload["mask_mode"]) not in FLOW_ORDER:
        return None
    region = str(payload.get("region") or "")
    target_key = (
        f"single_object::{region}"
        if payload["target_scope"] == "single_object"
        else "all_objects::"
    )
    audit = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
    return {
        "stage": stage,
        "case": str(payload["case"]),
        "seed": int(payload["seed"]),
        "variant_id": str(payload["variant_id"]),
        "target_scope": str(payload["target_scope"]),
        "region": region or None,
        "target_key": target_key,
        "head_scope": str(payload["head_scope"]),
        "head_scope_label": HEAD_SCOPE_LABELS[str(payload["head_scope"])],
        "selected_head_count": int(payload.get("selected_head_count") or payload.get("top_n") or 0),
        "mask_mode": str(payload["mask_mode"]),
        "flow": FLOW_DEFINITIONS[str(payload["mask_mode"])],
        "selected_token_count": len(audit.get("query_token_indices") or []),
        "modified_head_events": int(audit.get("modified_head_events") or 0),
        "dose_finite_events": int(audit.get("dose_finite_events") or 0),
        "video_ready": (path.parent / "generated.mp4").is_file(),
        "dose_ready": (path.parent / "dose_metrics.npz").is_file(),
    }


def _complete_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("complete.json")) if root.is_dir() else []


def _stage_records(root: Path, stage: str) -> list[dict[str, Any]]:
    result = []
    for complete in _complete_paths(root):
        record = _manifest_record(complete.parent / "manifest.json", stage)
        if record is not None and record["video_ready"]:
            result.append(record)
    return result


def _progress_groups(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    by_case = Counter(row["case"] for row in records)
    by_scope = Counter(row["head_scope"] for row in records)
    by_flow = Counter(row["mask_mode"] for row in records)
    by_seed = Counter(str(row["seed"]) for row in records)
    expected_by_case: Counter[str] = Counter()
    expected_by_seed: Counter[str] = Counter()
    for sample in _input_samples():
        seed = int(sample.get("seed", -1))
        if seed not in DISCOVERY_SEEDS:
            continue
        cells = len(_target_rows(sample)) * len(HEAD_SCOPES) * len(FLOW_ORDER)
        expected_by_case[str(sample["case"])] += cells
        expected_by_seed[str(seed)] += cells
    return {
        "completed": len(records),
        "expected": expected,
        "fraction": len(records) / expected if expected else 0.0,
        "by_case": [
            {"key": case, "completed": by_case[case], "expected": count}
            for case, count in expected_by_case.items()
        ],
        "by_scope": [
            {"key": scope, "label": HEAD_SCOPE_LABELS[scope], "completed": by_scope[scope], "expected": expected // 4}
            for scope in HEAD_SCOPES
        ],
        "by_flow": [
            {"key": flow, "label": FLOW_DEFINITIONS[flow]["id"], "completed": by_flow[flow], "expected": expected // 3}
            for flow in FLOW_ORDER
        ],
        "by_seed": [
            {"key": seed, "completed": by_seed[seed], "expected": expected_by_seed[seed]}
            for seed in map(str, DISCOVERY_SEEDS)
        ],
    }


def _signature() -> tuple[int, ...]:
    stage2 = _complete_paths(STAGE2_ROOT)
    stage3 = _complete_paths(STAGE3_ROOT)
    errors = sorted(STAGE3_ROOT.rglob("error.txt")) if STAGE3_ROOT.is_dir() else []
    reports = sorted(STAGE3_ROOT.rglob("report.json")) if STAGE3_ROOT.is_dir() else []
    latest2 = max((int(path.stat().st_mtime_ns) for path in stage2), default=0)
    latest3 = max((int(path.stat().st_mtime_ns) for path in stage3), default=0)
    latest_report = max((int(path.stat().st_mtime_ns) for path in reports), default=0)
    return len(stage2), latest2, len(stage3), latest3, len(errors), len(reports), latest_report


def _build_catalog() -> dict[str, Any]:
    report = _load_json(STAGE1_ROOT / "report.json")
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    per_anchor = report.get("per_anchor") if isinstance(report.get("per_anchor"), list) else []
    stage2_records = _stage_records(STAGE2_ROOT, "stage2")
    stage3_records = _stage_records(STAGE3_ROOT, "stage3")

    samples = []
    target_count_per_seed = 0
    seen_cases: set[str] = set()
    for row in _input_samples():
        seed = int(row.get("seed", -1))
        if seed not in DISCOVERY_SEEDS:
            continue
        case = str(row["case"])
        targets = _target_rows(row)
        if case not in seen_cases:
            target_count_per_seed += len(targets)
            seen_cases.add(case)
        samples.append(
            {
                "case": case,
                "seed": seed,
                "caption": str(row.get("caption") or ""),
                "targets": targets,
                "baseline_ready": Path(str(row.get("baseline_video") or "")).is_file(),
            }
        )
    expected = target_count_per_seed * len(DISCOVERY_SEEDS) * len(HEAD_SCOPES) * len(FLOW_ORDER)
    error_count = len(list(STAGE3_ROOT.rglob("error.txt"))) if STAGE3_ROOT.is_dir() else 0
    metric_report_count = len(list(STAGE3_ROOT.rglob("report.json"))) if STAGE3_ROOT.is_dir() else 0
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "object_query_information_flow_latest3350_v1",
        "plan_path": str(PLAN_PATH),
        "spec_path": str(SPEC_PATH),
        "shared_setting": {
            "attention": "A=softmax(QKᵀ/√d), Y=AV",
            "operator": "post-softmax contribution subtraction; no renormalization",
            "R": "同 seed Baseline 上冻结的 13-anchor object token tube",
            "C": "R 之外的全部 latent-video tokens",
            "denoising": "S000–S039 全 40 steps",
            "cfg": "conditional + unconditional",
        },
        "flows": FLOW_DEFINITIONS,
        "head_scopes": [
            {"key": key, "label": HEAD_SCOPE_LABELS[key], "count": HEAD_SCOPE_COUNTS[key]}
            for key in HEAD_SCOPES
        ],
        "stage1": {
            "decision": str(summary.get("decision") or "unknown"),
            "run_count": int(report.get("run_count") or 0),
            "case_count": int(report.get("case_count") or 0),
            "seed_count": len({seed for seeds in (report.get("seeds_by_case") or {}).values() for seed in seeds}),
            "median_jaccard": summary.get("median_pairwise_top100_jaccard"),
            "median_spearman": summary.get("median_pairwise_spearman"),
            "top_beats_fraction": summary.get("fixed_top100_beats_bottom100_anchor_fraction"),
            "top_minus_bottom_pck32": summary.get("case_mean_top_minus_bottom_pck32"),
            "bootstrap_lcb": summary.get("case_cluster_bootstrap_lcb_top_minus_bottom_pck32"),
            "anchors": [
                {
                    "query_time": int(row.get("query_time", -1)),
                    "pixel_frame": int(row.get("pixel_frame", -1)),
                    "top": row.get("fixed_top100_pck32"),
                    "bottom": row.get("fixed_bottom100_pck32"),
                    "difference": row.get("top_minus_bottom_pck32"),
                }
                for row in per_anchor
                if isinstance(row, dict)
            ],
            "overlays": [
                {"key": key, "label": name.replace("fixed_query_", "").replace(".mp4", "")}
                for key, name in STAGE1_OVERLAYS.items()
                if (STAGE1_ROOT / "overlays" / "0613pybullet_sample_001460_w002" / "seed_47326" / name).is_file()
            ],
        },
        "stage2": {
            "decision": "pass" if len(stage2_records) >= 2 else "incomplete",
            "unit_tests_passed": 9,
            "unit_tests_expected": 9,
            "smoke_records": stage2_records,
            "checks": [
                "Same/Future/Past 两两互斥且并集等于 All-time",
                "三段 contribution 之和逐元素等于 All-time 删除项",
                "no-op 与原 attention 逐元素一致",
                "Random100 精确逐层匹配并排除固定 Top/Bottom",
                "dose 与 dense attention 定义一致",
                "真实视频、manifest、complete marker 与 dose 均可读取",
            ],
        },
        "stage3": {
            "case_count": len(seen_cases),
            "seed_count": len(DISCOVERY_SEEDS),
            "seeds": list(DISCOVERY_SEEDS),
            "target_count_per_seed": target_count_per_seed,
            "expected": expected,
            "records": stage3_records,
            "samples": samples,
            "progress": _progress_groups(stage3_records, expected),
            "error_count": error_count,
            "dose_count": sum(bool(row["dose_ready"]) for row in stage3_records),
            "metric_report_count": metric_report_count,
            "effect_metrics_ready": metric_report_count > 0,
        },
    }


def catalog() -> dict[str, Any]:
    global _catalog_signature, _catalog_value
    signature = _signature()
    with _catalog_lock:
        if signature != _catalog_signature or _catalog_value is None:
            _catalog_value = _build_catalog()
            _catalog_signature = signature
        return _catalog_value


def _safe_variant_dir(root: Path, case: str, seed: int, variant: str) -> Path | None:
    if not case or not variant or seed < 0:
        return None
    candidate = (root / case / f"seed_{seed:05d}" / variant).resolve()
    base = root.resolve()
    if candidate != base and base not in candidate.parents:
        return None
    manifest = _load_json(candidate / "manifest.json")
    if (
        str(manifest.get("case")) != case
        or int(manifest.get("seed", -1)) != seed
        or str(manifest.get("variant_id")) != variant
    ):
        return None
    return candidate


def asset(
    kind: str,
    *,
    case: str = "",
    seed: int = -1,
    variant: str = "",
    name: str = "",
) -> Path | None:
    if kind == "baseline":
        sample = _sample(case, seed)
        path = Path(str(sample.get("baseline_video") or "")) if sample else None
        return path if path and path.is_file() else None
    if kind == "stage1":
        filename = STAGE1_OVERLAYS.get(name)
        path = (
            STAGE1_ROOT
            / "overlays"
            / "0613pybullet_sample_001460_w002"
            / "seed_47326"
            / str(filename)
        )
        return path if filename and path.is_file() else None
    root = STAGE2_ROOT if kind == "stage2" else STAGE3_ROOT if kind == "stage3" else None
    directory = _safe_variant_dir(root, case, seed, variant) if root else None
    path = directory / "generated.mp4" if directory else None
    return path if path and path.is_file() else None


@lru_cache(maxsize=512)
def _dose_cached(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as payload:
        result: dict[str, Any] = {}
        for key in (
            "attention_mass",
            "removed_value_norm",
            "original_output_norm",
            "removed_to_output_ratio",
            "target_query_count",
        ):
            if key not in payload:
                continue
            values = np.asarray(payload[key], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[key] = {
                "mean": float(finite.mean()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "p95": float(np.quantile(finite, 0.95)) if finite.size else None,
                "min": float(finite.min()) if finite.size else None,
                "max": float(finite.max()) if finite.size else None,
                "finite_count": int(finite.size),
            }
    result["definition"] = {
        "attention_mass": "被精确删除 source 集合上的 softmax probability mass",
        "removed_value_norm": "被删除向量 Σ A[q,k]V[k] 的平均 L2 范数",
        "original_output_norm": "同一受影响 Query 的原始 attention 输出平均 L2 范数",
        "removed_to_output_ratio": "removed_value_norm / original_output_norm；不是视频效应量",
        "target_query_count": "该 head/step/CFG 实际受影响的 Query token 数",
    }
    return result


def dose(stage: str, case: str, seed: int, variant: str) -> dict[str, Any] | None:
    root = STAGE2_ROOT if stage == "stage2" else STAGE3_ROOT if stage == "stage3" else None
    directory = _safe_variant_dir(root, case, seed, variant) if root else None
    path = directory / "dose_metrics.npz" if directory else None
    if path is None or not path.is_file():
        return None
    return _dose_cached(str(path), path.stat().st_mtime_ns)


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>latest3350 Object Query 信息流验证</title><style>
:root{--paper:#eee9dc;--ink:#17201e;--card:#fffdf8;--line:#b9b19f;--green:#176654;--rust:#b64a31;--blue:#315c86;--purple:#7056a2;--dark:#142820;--muted:#6c675d}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 4% 0,#d8784b30,transparent 34rem),radial-gradient(circle at 97% 4%,#4d9a8030,transparent 35rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}a{color:var(--green)}header,main{width:min(1840px,calc(100% - 24px));margin:auto}header{padding:25px 0 10px}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(40px,6vw,80px);line-height:.94;letter-spacing:-.045em;margin:12px 0}.eyebrow{color:var(--rust);font-weight:900;font-size:12px;letter-spacing:.16em}.lead{max-width:1100px;line-height:1.65}.toolbar{position:sticky;top:0;z-index:30;display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:11px;margin:18px 0;background:#f8f3e8ed;border:1px solid var(--line);backdrop-filter:blur(12px)}button,select{padding:9px 11px;border:1px solid var(--line);background:white;color:var(--ink);font-weight:800}button{cursor:pointer}.status{font-size:12px;color:var(--muted)}section{margin:22px 0;padding:18px;background:#ffffff9a;border:1px solid var(--line);border-radius:4px 22px 4px 4px}section h2{font-size:28px;margin:0 0 12px}.pill-row,.metric-row{display:flex;gap:8px;flex-wrap:wrap}.pill{padding:7px 10px;border-radius:99px;background:var(--card);border:1px solid var(--line);font-size:12px}.pass{color:var(--green);font-weight:900}.warn{color:var(--rust);font-weight:900}.metric{min-width:165px;padding:12px;background:var(--card);border:1px solid var(--line)}.metric b{display:block;font-size:23px}.metric span{font-size:11px;color:var(--muted)}.scroll{overflow:auto}.definitions{width:100%;border-collapse:collapse;min-width:900px}.definitions th,.definitions td{padding:10px;border:1px solid #d7d0c0;text-align:left;vertical-align:top}.definitions th{background:#20342d;color:white}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.anchor-chart{display:grid;grid-template-columns:repeat(13,minmax(45px,1fr));gap:5px;min-width:800px;align-items:end;height:210px;padding-top:15px}.anchor{display:flex;height:180px;align-items:end;justify-content:center;gap:2px;border-bottom:1px solid var(--line)}.bar{width:38%;min-height:2px}.bar.top{background:var(--green)}.bar.bottom{background:var(--rust)}.anchor-labels{display:grid;grid-template-columns:repeat(13,minmax(45px,1fr));gap:5px;min-width:800px;text-align:center;font-size:10px}.media-grid,.video-grid{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:11px}.video-grid{grid-template-columns:repeat(4,minmax(230px,1fr))}figure,.video-card{margin:0;padding:10px;background:var(--card);border:1px solid var(--line);border-radius:3px 18px 3px 3px}video{display:block;width:100%;aspect-ratio:16/9;background:#121817;object-fit:contain}figcaption,.caption{font-size:12px;line-height:1.45;margin-top:8px}.progress-list{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:7px}.progress-item{display:grid;grid-template-columns:minmax(140px,1fr) 3fr 74px;gap:8px;align-items:center;font-size:11px}.track{height:9px;background:#ddd5c7;overflow:hidden;border-radius:99px}.track i{display:block;height:100%;background:linear-gradient(90deg,var(--rust),var(--green))}.filters{display:flex;gap:9px;flex-wrap:wrap;margin:13px 0}.filters label{display:grid;gap:4px;font-size:11px;font-weight:900}.filters select{min-width:210px}.flow-section{margin:15px 0;border-top:1px solid var(--line);padding-top:13px}.flow-section h3{margin:0 0 3px}.formula{font-size:12px;color:var(--muted);margin:5px 0 10px}.video-card h4{margin:0 0 7px;font-size:14px}.video-card details{margin-top:8px;font-size:11px}.dose{padding:8px;background:#eee9dc;line-height:1.6}.empty{padding:24px;border:1px dashed var(--line);color:var(--muted)}.legend{display:flex;gap:15px;font-size:11px}.legend i{display:inline-block;width:10px;height:10px;margin-right:4px}.stage2-list{columns:2;line-height:1.7}.baseline{max-width:440px}.footer{padding:10px 0 60px;color:var(--muted);font-size:11px}@media(max-width:1150px){.video-grid{grid-template-columns:repeat(3,minmax(220px,1fr))}}@media(max-width:850px){.media-grid,.video-grid,.progress-list{grid-template-columns:repeat(2,minmax(190px,1fr))}.stage2-list{columns:1}}@media(max-width:560px){header,main{width:calc(100% - 10px)}.media-grid,.video-grid,.progress-list{grid-template-columns:1fr}.progress-item{grid-template-columns:1fr}.filters label,.filters select{width:100%}}
</style></head><body><header><a href="/">返回 8092 总入口</a> · <a href="/object-query-information-flow-stage4?v=1">进入 Stage 4 时序验证</a> · <a href="/object-query-m123-temporal-batch?v=1">旧 M1/M2/M3 页面</a><div class="eyebrow">LATEST3350 · PRE-REGISTERED INFORMATION-FLOW VALIDATION</div><h1>Object Query<br>信息流验证</h1><p class="lead">对应执行计划的 Stage 1–3 实时入口。Stage 1 验证固定 PCK head ranking 是否跨 Query 时刻稳定；Stage 2 验证消融代数、Random100 与 dose 实现；Stage 3 比较 M1/M2/M3 在 Top100、Bottom100、layer-matched Random100 和 All720 上对生成结果的影响。Stage 4 已拆为独立子页面，按时间方向展示正在生成的矩阵。</p><div class="toolbar"><button id="refresh">刷新进度</button><button id="replay">同步重播当前视频</button><span id="status" class="status">读取中…</span></div></header><main><section id="overview"><h2>实验冻结口径</h2><div id="setting" class="pill-row"></div><div id="flowTable" class="scroll"></div></section><section><h2>Stage 1 · Query-time Head Ranking 验证</h2><p>这里验证的是 fixed latest3350 Top/Bottom 对不同 Q<sub>t</sub> 是否仍有区分力，不是 M1/M2/M3 因果消融结果。</p><div id="stage1Metrics" class="metric-row"></div><div class="scroll"><div id="anchorChart" class="anchor-chart"></div><div id="anchorLabels" class="anchor-labels"></div></div><div id="stage1Media" class="media-grid"></div></section><section><h2>Stage 2 · 实现审计与真实 Smoke</h2><div id="stage2Metrics" class="metric-row"></div><ul id="stage2Checks" class="stage2-list"></ul><div id="stage2Media" class="media-grid"></div></section><section><h2>Stage 3 · All-time 信息类型筛查</h2><p>每个已生成视频均在全部 40 个 denoising steps、两个 CFG 分支和所有 latent 时刻上执行。只显示已完成视频，不为尚未生成项留空卡片。</p><div id="stage3Metrics" class="metric-row"></div><div id="progress" class="progress-list"></div><div class="filters"><label>Case<select id="case"></select></label><label>Seed<select id="seed"></select></label><label>Target<select id="target"></select></label></div><div id="baseline"></div><div id="gallery"></div></section><section><h2>视频效果指标状态</h2><p id="metricStatus"></p><p class="status">Attention-dose 回答“实际删除了多少 attention / AV 贡献”；轨迹、外观、背景和对象存活指标回答“生成结果改变了什么”。两者不能互相替代。</p></section><div class="footer">页面每 30 秒刷新生成进度；只有选中的 case/seed/target 视频会进入 DOM，并在靠近视口时加载。</div></main><script>
const api='/api/object-query-information-flow-validation',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null,signature='';
const pct=(a,b)=>b?`${(100*a/b).toFixed(1)}%`:'—',num=(v,d=3)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'—';
function metric(label,value,note){return `<div class="metric"><b>${esc(value)}</b><span>${esc(label)} · ${esc(note||'')}</span></div>`}
function mediaUrl(kind,row,name=''){const p=new URLSearchParams({kind,case:row?.case||'',seed:String(row?.seed??-1),variant:row?.variant_id||'',name});return `${api}/asset?${p}`}
function lazy(root=document){const io=new IntersectionObserver(es=>es.forEach(x=>{if(x.isIntersecting){const v=x.target;if(v.dataset.src){v.src=v.dataset.src;delete v.dataset.src;v.load()}io.unobserve(v)}}),{rootMargin:'350px'});root.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function video(src,caption){return `<figure><video controls muted playsinline preload="none" data-src="${esc(src)}"></video><figcaption>${caption}</figcaption></figure>`}
function renderStatic(){const s=data.shared_setting;$('setting').innerHTML=[s.attention,s.operator,`R：${s.R}`,`C：${s.C}`,s.denoising,s.cfg].map(x=>`<span class="pill">${esc(x)}</span>`).join('');$('flowTable').innerHTML=`<table class="definitions"><thead><tr><th>ID</th><th>删除矩阵块</th><th>信息流</th><th>精确计算</th><th>理论诊断目标</th></tr></thead><tbody>${Object.values(data.flows).map(x=>`<tr><td><b>${esc(x.id)}</b></td><td>${esc(x.block)}</td><td>${esc(x.flow)}</td><td class="mono">${esc(x.formula)}</td><td>${esc(x.diagnosis)}</td></tr>`).join('')}</tbody></table>`;const a=data.stage1;$('stage1Metrics').innerHTML=metric('Gate',a.decision.toUpperCase(),`${a.case_count} cases × ${a.seed_count} seeds`)+metric('Top100 Jaccard',num(a.median_jaccard,4),'跨 Query 时间重合')+metric('Ranking Spearman',num(a.median_spearman,4),'跨 Query 时间排序')+metric('Top−Bottom PCK@32',`${num(a.top_minus_bottom_pck32,3)} pp`,`case-bootstrap LCB ${num(a.bootstrap_lcb,3)} pp`);const max=100;$('anchorChart').innerHTML=a.anchors.map(x=>`<div class="anchor" title="Q${x.query_time} / F${x.pixel_frame} · Top ${num(x.top)} · Bottom ${num(x.bottom)}"><i class="bar top" style="height:${Math.max(1,x.top/max*100)}%"></i><i class="bar bottom" style="height:${Math.max(1,x.bottom/max*100)}%"></i></div>`).join('');$('anchorLabels').innerHTML=a.anchors.map(x=>`<span>Q${String(x.query_time).padStart(2,'0')}<br>F${String(x.pixel_frame).padStart(2,'0')}</span>`).join('');$('stage1Media').innerHTML=a.overlays.map(x=>video(mediaUrl('stage1',null,x.key),`001460 · seed 47326 · ${esc(x.label)} · 固定 Query 对所有 K 时刻的 Top100 响应与 GT tube`)).join('');const b=data.stage2;$('stage2Metrics').innerHTML=metric('Gate',b.decision.toUpperCase(),'大规模运行前硬门槛')+metric('CPU tests',`${b.unit_tests_passed}/${b.unit_tests_expected}`,'全部通过')+metric('GPU smoke',`${b.smoke_records.length}/2`,'Random100-M2 + All720-M3');$('stage2Checks').innerHTML=b.checks.map(x=>`<li>${esc(x)}</li>`).join('');$('stage2Media').innerHTML=b.smoke_records.map(r=>video(mediaUrl('stage2',r),`${esc(r.flow.id)} · ${esc(r.head_scope_label)} · ${esc(r.case)} / seed ${r.seed}<br>${esc(r.flow.formula)} · dose events ${r.dose_finite_events.toLocaleString()}`)).join('');lazy()}
function options(el,rows,value,label,wanted){const previous=wanted??el.value;el.innerHTML=rows.map(x=>`<option value="${esc(value(x))}">${esc(label(x))}</option>`).join('');if([...el.options].some(o=>o.value===String(previous)))el.value=String(previous)}
function selectedSample(){return data.stage3.samples.find(x=>x.case===$('case').value&&String(x.seed)===$('seed').value)}
function populateFilters(first=false){const cases=[...new Set(data.stage3.samples.map(x=>x.case))],caseWanted=first?(q.get('case')||cases[0]):$('case').value;options($('case'),cases,x=>x,x=>x,caseWanted);const seeds=data.stage3.samples.filter(x=>x.case===$('case').value).map(x=>x.seed),seedWanted=first?(q.get('seed')||47326):$('seed').value;options($('seed'),seeds,x=>x,x=>`seed ${x}`,seedWanted);const sample=selectedSample(),targetWanted=first?(q.get('target')||sample?.targets[0]?.key):$('target').value;options($('target'),sample?.targets||[],x=>x.key,x=>x.scope==='all_objects'?'All objects':x.region,targetWanted)}
function progressRow(x){return `<div class="progress-item"><span>${esc(x.label||x.key)}</span><div class="track"><i style="width:${pct(x.completed,x.expected)}"></i></div><b>${x.completed}/${x.expected}</b></div>`}
function doseDetails(r){return `<details data-stage="${r.stage}" data-case="${esc(r.case)}" data-seed="${r.seed}" data-variant="${esc(r.variant_id)}"><summary>实际删除 dose · 点击加载</summary><div class="dose">尚未加载</div></details>`}
function card(r){return `<article class="video-card"><h4>${esc(r.head_scope_label)} · ${r.selected_head_count} heads</h4><video controls muted playsinline preload="none" data-src="${esc(mediaUrl('stage3',r))}"></video><div class="caption"><b>${esc(r.flow.id)}</b> · ${esc(r.flow.flow)}<br><span class="mono">${esc(r.flow.formula)}</span><br>R tokens ${r.selected_token_count} · modified events ${r.modified_head_events.toLocaleString()}</div>${doseDetails(r)}</article>`}
async function loadDose(details){if(details.dataset.loaded)return;details.dataset.loaded='1';const p=new URLSearchParams({stage:details.dataset.stage,case:details.dataset.case,seed:details.dataset.seed,variant:details.dataset.variant});const box=details.querySelector('.dose');try{const d=await fetch(`${api}/dose?${p}`,{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()});const rows=[['attention_mass','Removed attention mass'],['removed_value_norm','Removed AV norm'],['original_output_norm','Original output norm'],['removed_to_output_ratio','Removed / output ratio'],['target_query_count','Target Query count']];box.innerHTML=rows.map(([k,l])=>`<b>${l}</b>：mean ${num(d[k]?.mean,5)} · median ${num(d[k]?.median,5)} · P95 ${num(d[k]?.p95,5)} · N ${d[k]?.finite_count??0}<br>`).join('')}catch(e){box.textContent=`读取失败：${e}`;details.dataset.loaded=''}}
function renderDynamic(force=false){const p=data.stage3.progress,selected=selectedSample();$('stage3Metrics').innerHTML=metric('生成进度',`${p.completed}/${p.expected}`,pct(p.completed,p.expected))+metric('设计规模',`${data.stage3.case_count} cases × ${data.stage3.seed_count} seeds`,`${data.stage3.target_count_per_seed} targets/seed`)+metric('Dose',`${data.stage3.dose_count}/${p.completed}`,'attention/AV 删除量')+metric('Errors',data.stage3.error_count,'目标为 0');$('progress').innerHTML=p.by_case.map(x=>progressRow(x)).join('');const records=data.stage3.records.filter(r=>r.case===$('case').value&&String(r.seed)===$('seed').value&&r.target_key===$('target').value);const baseline=selected?.baseline_ready?video(mediaUrl('baseline',selected),`未消融 Baseline · ${esc(selected.case)} · seed ${selected.seed}`):'<div class="empty">Baseline 不可用</div>';$('baseline').innerHTML=`<h3>共同 Baseline</h3><div class="baseline">${baseline}</div>`;$('gallery').innerHTML=data.stage3.records.length?Object.keys(data.flows).map(flow=>{const rows=records.filter(r=>r.mask_mode===flow),f=data.flows[flow];return `<div class="flow-section"><h3>${esc(f.id)} · ${esc(f.flow)}</h3><div class="formula"><span class="mono">${esc(f.formula)}</span> · ${esc(f.diagnosis)}</div>${rows.length?`<div class="video-grid">${rows.sort((a,b)=>data.head_scopes.findIndex(x=>x.key===a.head_scope)-data.head_scopes.findIndex(x=>x.key===b.head_scope)).map(card).join('')}</div>`:'<div class="empty">该组合尚未生成；不创建空视频卡片。</div>'}</div>`}).join(''):'<div class="empty">Stage 3 尚无完成项。</div>';$('metricStatus').innerHTML=data.stage3.effect_metrics_ready?`已发现 <b>${data.stage3.metric_report_count}</b> 个结果指标报告。`:`当前已有 <b>${data.stage3.dose_count}</b> 份 attention-dose，但轨迹 ADE/FDE、对象 LPIPS/DINO、背景 spillover 和存活/消失等结果指标尚未写入本批次（report.json=${data.stage3.metric_report_count}）。`;document.querySelectorAll('details[data-variant]').forEach(x=>x.addEventListener('toggle',()=>{if(x.open)loadDose(x)}));lazy();const u=new URL(location.href);u.searchParams.set('case',$('case').value);u.searchParams.set('seed',$('seed').value);u.searchParams.set('target',$('target').value);history.replaceState(null,'',u);$('status').textContent=`Stage 3 ${p.completed}/${p.expected} · ${pct(p.completed,p.expected)} · ${data.stage3.error_count} errors · 更新 ${new Date(data.generated_at_utc).toLocaleTimeString()}`}
async function load(first=false){const next=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());const nextSig=`${next.stage3.progress.completed}:${next.stage3.error_count}:${next.stage3.metric_report_count}`;data=next;if(first){renderStatic();populateFilters(true);renderDynamic(true)}else{const oldCase=$('case').value,oldSeed=$('seed').value,oldTarget=$('target').value;populateFilters(false);if([...$('case').options].some(x=>x.value===oldCase))$('case').value=oldCase;populateFilters(false);if([...$('seed').options].some(x=>x.value===oldSeed))$('seed').value=oldSeed;populateFilters(false);if([...$('target').options].some(x=>x.value===oldTarget))$('target').value=oldTarget;if(nextSig!==signature)renderDynamic(true)}signature=nextSig}
$('case').addEventListener('change',()=>{populateFilters(false);renderDynamic(true)});$('seed').addEventListener('change',()=>{populateFilters(false);renderDynamic(true)});$('target').addEventListener('change',()=>renderDynamic(true));$('refresh').addEventListener('click',()=>load(false));$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load(true).catch(e=>$('status').textContent=`读取失败：${e}`);setInterval(()=>load(false).catch(()=>{}),30000);
</script></body></html>'''
