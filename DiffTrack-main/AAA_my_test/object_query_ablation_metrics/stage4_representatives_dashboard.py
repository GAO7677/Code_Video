#!/usr/bin/env python3
"""Controlled-variable Stage-4 representative-video dashboard."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from AAA_my_test.object_query_ablation_metrics import stage4_temporal_dashboard


EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
METRICS_ROOT = EXPERIMENT_ROOT / "stage4_metrics"
FAST_RANKING = METRICS_ROOT / "head_scope_baseline_fast" / "ranking.json"
TRAJECTORY_ROOT = METRICS_ROOT / "head_scope_trajectory"
THREE_AXIS_REPORT = EXPERIMENT_ROOT / "stage4_current_analysis" / "three_axis_report.json"

BALL_CASE = "".join(
    (
        "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-",
        "ball-and-block-fall_motion_to_end",
    )
)
PYBULLET_CASE = "0613pybullet_sample_001460_w002"


REPRESENTATIVES: tuple[dict[str, Any], ...] = (
    {
        "axis": "head",
        "id": "head-m1-future",
        "label": "HEAD / M1-FUTURE",
        "case": "0613pybullet_sample_000331_w001",
        "seed": 13248,
        "target_scope": "single_object",
        "region": "object_A",
        "title": "固定 M1 + Future，只换 Top100 / Bottom100",
        "fixed": "固定 case、seed、object_A、M1、Future；唯一变化是被消融的 head 集合。",
        "claim": (
            "当前 3-case case-balanced pilot 中，Top100 相比 Bottom100："
            "Center-ADE 高 0.072 D0、Track Loss 高 27.97 pp、Identity Failure 高 23.33 pp、"
            "Disappearance 高 28.27 pp；四项均为 3/3 case 同向。"
        ),
        "evidence": (
            "本例是最清楚的可视正例：Top100 的 Disappearance=91.84%，Bottom100=0%；"
            "Top100 已无法通过轨迹门控，因此不能只比较 ADE。"
        ),
        "caveat": (
            "这是 3 个独立 case 的探索性结果，且 Top100 删除 AV dose 是 Bottom100 的 7.32×；"
            "它支持 Top100-M1-Future contribution 更强，但不能证明单位 dose 的 head 更关键。"
        ),
        "primary_label": "Disappearance %",
        "primary_path": ("survival", "disappearance_percent"),
        "rows": (
            {"label": "Top100", "head_scope": "top100", "mask_mode": "self_future", "tone": "top"},
            {"label": "Bottom100", "head_scope": "bottom100", "mask_mode": "self_future", "tone": "bottom"},
        ),
    },
    {
        "axis": "head",
        "id": "head-m2-same",
        "label": "HEAD / M2-SAME",
        "case": BALL_CASE,
        "seed": 13248,
        "target_scope": "single_object",
        "region": "object_C",
        "title": "固定 M2 + Same，Bottom100 反而更强",
        "fixed": "固定 case、seed、object_C、M2、Same；唯一变化是 Top100 / Bottom100。",
        "claim": (
            "当前 3-case pilot 中，相比 Top100，Bottom100 的 Track Loss 高 10.35 pp、"
            "Identity Failure 高 16.96 pp、Mask Absence 高 14.35 pp、Disappearance 高 15.12 pp；"
            "四项均为 3/3 case 同向。"
        ),
        "evidence": (
            "本例 Bottom100 的 Disappearance=69.39%，Top100=2.04%。"
            "这说明 latest3350 的 PCK 排名不等价于所有信息流上的统一重要性排名。"
        ),
        "caveat": (
            "Bottom100 的 M2-Same 删除 AV dose 在总体上约为 Top100 的 3.70×；"
            "因此这里首先说明 C→R Same contribution 在 Bottom100 更大，不是单位 dose 因果效率。"
        ),
        "primary_label": "Disappearance %",
        "primary_path": ("survival", "disappearance_percent"),
        "rows": (
            {"label": "Top100", "head_scope": "top100", "mask_mode": "incoming_same", "tone": "top"},
            {"label": "Bottom100", "head_scope": "bottom100", "mask_mode": "incoming_same", "tone": "bottom"},
        ),
    },
    {
        "axis": "time",
        "id": "time-m2-top",
        "label": "TIME / TOP100-M2",
        "case": BALL_CASE,
        "seed": 13248,
        "target_scope": "single_object",
        "region": "object_C",
        "title": "固定 Top100 + M2，只换 Same / Future / Past",
        "fixed": "固定 case、seed、object_C、Top100、M2；唯一变化是 t_k 与 t_q 的关系。",
        "claim": (
            "当前 3-case pilot 中，Future 相比 Past 的 Identity Failure 高 5.82 pp、"
            "Disappearance 高 3.64 pp，3/3 case 均为非负；但 Center-ADE 和 Track Loss 方向混合。"
        ),
        "evidence": (
            "本例 Future 的 Disappearance=89.80%，Same=2.04%，Past=0%。"
            "它直观展示同一 M2 中不同时间切片可以产生完全不同的对象存活结果。"
        ),
        "caveat": (
            "Future/Past 的删除 dose 本身不同；当前证据只支持对象身份/存活差异，"
            "不支持“Future 一定更改轨迹”的普遍结论。"
        ),
        "primary_label": "Disappearance %",
        "primary_path": ("survival", "disappearance_percent"),
        "rows": (
            {"label": "Same · t_k=t_q", "head_scope": "top100", "mask_mode": "incoming_same", "tone": "same"},
            {"label": "Future · t_k<t_q", "head_scope": "top100", "mask_mode": "incoming_future", "tone": "future"},
            {"label": "Past · t_k>t_q", "head_scope": "top100", "mask_mode": "incoming_past", "tone": "past"},
        ),
    },
    {
        "axis": "time",
        "id": "time-m1-counterexample",
        "label": "TIME / COUNTEREXAMPLE",
        "case": PYBULLET_CASE,
        "seed": 47326,
        "target_scope": "single_object",
        "region": "object_A",
        "title": "M1 时间方向的反例：Past 比 Future 更破坏对象",
        "fixed": "固定 case、seed、object_A、Top100、M1；唯一变化是 Same / Future / Past。",
        "claim": (
            "3-case 汇总中 M1 Future−Past 的 Disappearance、Identity Failure 和 Center-ADE 均为 case 方向混合；"
            "因此没有足够证据给出 M1 的统一时间顺序。"
        ),
        "evidence": (
            "本例 Future 的 Disappearance=0%，Past=36.73%，与另一个 case 中 Future=91.84%、Past=0% 的方向相反。"
            "它是防止把单一 seed 现象误写成机制结论的关键反例。"
        ),
        "caveat": "该组的作用是证伪过强结论，不代表 Past 总体更重要；需要更多独立 case 才能估计时间方向效应。",
        "primary_label": "Disappearance %",
        "primary_path": ("survival", "disappearance_percent"),
        "rows": (
            {"label": "Same · t_k=t_q", "head_scope": "top100", "mask_mode": "self_same", "tone": "same"},
            {"label": "Future · t_k<t_q", "head_scope": "top100", "mask_mode": "self_future", "tone": "future"},
            {"label": "Past · t_k>t_q", "head_scope": "top100", "mask_mode": "self_past", "tone": "past"},
        ),
    },
    {
        "axis": "flow",
        "id": "flow-top-same",
        "label": "FLOW / TOP100-SAME",
        "case": BALL_CASE,
        "seed": 90094,
        "target_scope": "single_object",
        "region": "object_D",
        "title": "固定 Top100 + Same，只换 M1 / M2 / M3",
        "fixed": "固定 case、seed、object_D、Top100、Same；唯一变化是被切断的信息流。",
        "claim": (
            "当前 3-case pilot 中，M1 相比 M2：Track Loss 高 13.93 pp、Identity Failure 高 19.32 pp、"
            "Disappearance 高 16.99 pp；相比 M3 分别高 14.03、11.91、11.53 pp，均为 3/3 case 同向。"
        ),
        "evidence": (
            "本例 M1/M2/M3 的 Disappearance 分别为 73.47%/4.08%/10.20%。"
            "在相同 heads 和时间切片下，对象内部 R→R 对身份维持和存活的影响最大。"
        ),
        "caveat": (
            "Center-ADE 的 M1−M2/M3 case 方向仍混合，所以这里能较稳地讨论身份/存活，"
            "不能据此断言 M1 在所有 case 中造成最大轨迹偏移。"
        ),
        "primary_label": "Disappearance %",
        "primary_path": ("survival", "disappearance_percent"),
        "rows": (
            {"label": "M1 · R→R", "head_scope": "top100", "mask_mode": "self_same", "tone": "m1"},
            {"label": "M2 · C→R", "head_scope": "top100", "mask_mode": "incoming_same", "tone": "m2"},
            {"label": "M3 · R→C", "head_scope": "top100", "mask_mode": "outgoing_same", "tone": "m3"},
        ),
    },
    {
        "axis": "flow",
        "id": "flow-bottom-future",
        "label": "FLOW / BOTTOM100-FUTURE",
        "case": BALL_CASE,
        "seed": 13248,
        "target_scope": "single_object",
        "region": "object_C",
        "title": "固定 Bottom100 + Future，M2 比 M1 更影响轨迹与存活",
        "fixed": "固定 case、seed、object_C、Bottom100、Future；唯一变化是 M1 / M2 / M3。",
        "claim": (
            "当前 3-case pilot 中，相比 M1，M2 的 Center-ADE 高 0.021 D0、Velocity Error 高 0.009 D0/frame、"
            "Track Loss 高 10.64 pp、Identity Failure 高 7.63 pp、Disappearance 高 6.21 pp；"
            "各 case 均同向或含零同向。"
        ),
        "evidence": (
            "本例 M1/M2/M3 的 Center-ADE=0.082/0.409/0.152 D0，"
            "Disappearance=0%/71.43%/14.29%；M2 同时改变轨迹和对象存活。"
        ),
        "caveat": (
            "Bottom100-M2-Future 的删除 AV dose 总体约为 M1 的 5.72×；"
            "因此结论是该通道总 contribution 更大，而非证明 C→R 的单位 dose 更有效。"
        ),
        "primary_label": "Center-ADE / D0",
        "primary_path": ("trajectory", "center_ade_d0"),
        "rows": (
            {"label": "M1 · R→R", "head_scope": "bottom100", "mask_mode": "self_future", "tone": "m1"},
            {"label": "M2 · C→R", "head_scope": "bottom100", "mask_mode": "incoming_future", "tone": "m2"},
            {"label": "M3 · R→C", "head_scope": "bottom100", "mask_mode": "outgoing_future", "tone": "m3"},
        ),
    },
)


_lock = threading.Lock()
_cache_signature: tuple[int, ...] | None = None
_cache_value: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _records(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    return {
        str(row.get("variant_id") or row.get("id")): row
        for row in payload.get("records", [])
        if isinstance(row, dict) and (row.get("variant_id") or row.get("id"))
    }


def _all_report_records(name: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    if not TRAJECTORY_ROOT.is_dir():
        return result
    for path in sorted(TRAJECTORY_ROOT.rglob(name)):
        payload = _load_json(path)
        case = str(payload.get("case") or path.parent.parent.name)
        seed = int(payload.get("seed", str(path.parent.name).replace("seed_", "") or -1))
        for variant, row in _records(path).items():
            result[(case, seed, variant)] = row
    return result


def _signature() -> tuple[int, ...]:
    paths = [FAST_RANKING, THREE_AXIS_REPORT]
    if TRAJECTORY_ROOT.is_dir():
        paths.extend(sorted(TRAJECTORY_ROOT.rglob("report.json")))
        paths.extend(sorted(TRAJECTORY_ROOT.rglob("object_survival_report.json")))
    stage4 = stage4_temporal_dashboard.catalog()
    return (
        int(stage4["progress"]["completed"]),
        *(path.stat().st_mtime_ns if path.is_file() else 0 for path in paths),
    )


def _trajectory_summary(row: dict[str, Any], region: str | None) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    objects = metrics.get("objects") if isinstance(metrics.get("objects"), dict) else {}
    target = objects.get(region) if region and isinstance(objects.get(region), dict) else {}
    pck = target.get("pck_normalized") if isinstance(target.get("pck_normalized"), dict) else {}
    return {
        "quality_pass": metrics.get("quality_pass"),
        "center_ade_d0": metrics.get("target_center_ade_norm"),
        "center_fde_d0": target.get("center_fde_norm"),
        "velocity_error_d0_per_frame": target.get("velocity_vector_error_norm_per_frame"),
        "pck10_error_percent": 100.0 * (1.0 - float(pck["0.1"])) if "0.1" in pck else None,
        "track_loss_percent": metrics.get("target_worst_track_loss_score_0_100"),
    }


def _survival_summary(row: dict[str, Any], region: str | None) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    objects = metrics.get("objects") if isinstance(metrics.get("objects"), dict) else {}
    target = objects.get(region) if region and isinstance(objects.get(region), dict) else {}
    return {
        "disappearance_percent": metrics.get("target_worst_disappearance_score_0_100"),
        "mask_absence_percent": metrics.get("target_worst_mask_absence_score_0_100"),
        "identity_failure_percent": 100.0 * float(target["identity_failure_rate"]) if "identity_failure_rate" in target else None,
        "area_failure_percent": 100.0 * float(target["area_failure_rate"]) if "area_failure_rate" in target else None,
        "terminal_missing_percent": 100.0 * float(target["terminal_missing_rate"]) if "terminal_missing_rate" in target else None,
    }


def _calculation(mask_mode: str) -> str:
    flow, direction = mask_mode.rsplit("_", 1)
    predicate = {"same": "t_k=t_q", "future": "t_k<t_q", "past": "t_k>t_q"}[direction]
    term = {
        "self": "A[R_tq,R_tk] · V[R_tk]",
        "incoming": "A[R_tq,C_tk] · V[C_tk]",
        "outgoing": "A[C_tq,R_tk] · V[R_tk]",
    }[flow]
    return f"删除 Σ_{{{predicate}}} {term}；post-softmax subtraction，不重归一化"


def _nested(row: dict[str, Any], path: tuple[str, str]) -> Any:
    parent = row.get(path[0])
    return parent.get(path[1]) if isinstance(parent, dict) else None


def _build_catalog() -> dict[str, Any]:
    stage4 = stage4_temporal_dashboard.catalog()
    generated = list(stage4.get("records", []))
    fast_payload = _load_json(FAST_RANKING)
    fast = {
        (str(row.get("case")), int(row.get("seed", -1)), str(row.get("variant_id"))): row
        for row in fast_payload.get("records", [])
        if isinstance(row, dict) and row.get("variant_id")
    }
    trajectory = _all_report_records("report.json")
    survival = _all_report_records("object_survival_report.json")
    analysis = _load_json(THREE_AXIS_REPORT)
    groups: list[dict[str, Any]] = []
    for definition in REPRESENTATIVES:
        rows: list[dict[str, Any]] = []
        for row_definition in definition["rows"]:
            matched = next(
                (
                    row
                    for row in generated
                    if row.get("case") == definition["case"]
                    and int(row.get("seed", -1)) == definition["seed"]
                    and row.get("target_scope") == definition["target_scope"]
                    and row.get("region") == definition["region"]
                    and row.get("mask_mode") == row_definition["mask_mode"]
                    and row.get("head_scope") == row_definition["head_scope"]
                ),
                None,
            )
            card = dict(row_definition)
            card["calculation"] = _calculation(row_definition["mask_mode"])
            if matched is None:
                card["ready"] = False
                rows.append(card)
                continue
            variant = str(matched["variant_id"])
            key = (definition["case"], definition["seed"], variant)
            fast_row = fast.get(key, {})
            categories = fast_row.get("category_scores_0_100") or {}
            card.update(
                {
                    "ready": True,
                    "variant_id": variant,
                    "fast": {
                        "impact": fast_row.get("impact_score_0_100"),
                        "target_local": categories.get("target_local"),
                        "outside_spillover": categories.get("outside_spillover"),
                    },
                    "trajectory": _trajectory_summary(trajectory[key], definition["region"]) if key in trajectory else None,
                    "survival": _survival_summary(survival[key], definition["region"]) if key in survival else None,
                }
            )
            card["primary_value"] = _nested(card, definition["primary_path"])
            rows.append(card)
        group = {key: value for key, value in definition.items() if key not in {"rows", "primary_path"}}
        group["rows"] = rows
        group["baseline_ready"] = stage4_temporal_dashboard.asset("baseline", definition["case"], definition["seed"]) is not None
        groups.append(group)

    coverage = analysis.get("coverage") if isinstance(analysis.get("coverage"), dict) else {}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_snapshot": datetime.fromtimestamp(THREE_AXIS_REPORT.stat().st_mtime, timezone.utc).isoformat() if THREE_AXIS_REPORT.is_file() else None,
        "groups": groups,
        "status": {
            "generated": stage4["progress"]["completed"],
            "expected": stage4["progress"]["expected"],
            "analysis_records": coverage.get("records"),
            "case_count": len(coverage.get("cases", [])),
            "case_seed_count": len(coverage.get("case_seeds", [])),
            "representative_groups": len(groups),
        },
    }


def catalog() -> dict[str, Any]:
    global _cache_signature, _cache_value
    signature = _signature()
    with _lock:
        if _cache_value is None or signature != _cache_signature:
            _cache_value = _build_catalog()
            _cache_signature = signature
        return _cache_value


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 4 · 控制变量代表视频</title><style>
:root{--bg:#091115;--panel:#111e24;--panel2:#182a31;--ink:#eef7f5;--muted:#9aafb1;--line:#2e4850;--head:#4fe0bf;--time:#ffba67;--flow:#b99aff;--warn:#ffe080;--top:#4fe0bf;--bottom:#ff8d68;--same:#d7dfe1;--future:#63b5ff;--past:#ffcc67;--m1:#4fe0bf;--m2:#ff8d68;--m3:#b99aff}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 12% 0,#153039 0,transparent 34%),linear-gradient(180deg,#081014,#0f2026 55%,#081216);color:var(--ink);font-family:"Aptos","Noto Sans CJK SC",sans-serif}a{color:var(--head)}header,main{width:min(1720px,calc(100% - 28px));margin:auto}header{padding:28px 0 18px}.nav,.axis-nav{display:flex;gap:14px;flex-wrap:wrap;font:800 12px ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{margin-top:44px;color:var(--head);letter-spacing:.16em;font:900 12px ui-monospace,SFMono-Regular,Menlo,monospace}h1,h2,h3{font-family:"Iowan Old Style","Noto Serif CJK SC",Georgia,serif}.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:38px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:26px}h1{font-size:clamp(46px,6vw,92px);line-height:.9;letter-spacing:-.055em;margin:12px 0}.hero p{font-size:16px;line-height:1.75;color:var(--muted);margin:0}.status{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:8px;margin:18px 0}.stat{padding:13px 15px;background:var(--panel);border-top:3px solid var(--line)}.stat b{display:block;font:800 25px ui-monospace,SFMono-Regular,Menlo,monospace}.stat span{font-size:10px;color:var(--muted)}.method{margin:26px 0;padding:18px;background:var(--panel);border:1px solid var(--line)}.method h2{margin:0 0 12px}.control-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.control{padding:14px;background:var(--bg);border-top:4px solid var(--axis)}.control b{font-size:14px}.control p{font-size:11px;color:var(--muted);line-height:1.6;margin:6px 0 0}.warning{margin-top:12px;padding:11px 14px;background:#ffe08012;border-left:4px solid var(--warn);font-size:12px;line-height:1.6}.axis-nav{margin-top:16px}.axis-nav a{padding:8px 11px;background:var(--panel2);text-decoration:none}.axis-block{margin:42px 0 8px;scroll-margin-top:15px}.axis-title{display:flex;align-items:end;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:10px}.axis-title h2{font-size:36px;margin:0}.axis-title p{margin:0;color:var(--muted);font-size:12px}.case{margin:26px 0 54px}.case-head{display:grid;grid-template-columns:170px minmax(280px,.9fr) minmax(420px,1.2fr);gap:22px;padding:18px 0}.case-label{font:900 11px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--axis);letter-spacing:.12em}.case h3{font-size:29px;line-height:1.05;margin:8px 0}.identity{font:11px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);overflow-wrap:anywhere}.fixed{font-size:12px;line-height:1.65;padding:10px 12px;background:var(--panel2);border-left:4px solid var(--axis)}.claim{font-size:14px;line-height:1.7}.evidence{padding:11px 13px;background:#4fe0bf10;border-left:4px solid var(--head);font:750 12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}.caveat{padding-left:13px;border-left:4px solid var(--warn);font-size:12px;line-height:1.65;color:var(--muted)}.actions{display:flex;gap:12px;align-items:center;margin-top:10px}.actions button{border:1px solid var(--line);background:var(--panel2);color:var(--ink);padding:8px 11px;font-weight:800;cursor:pointer}.ruler{display:grid;grid-template-columns:repeat(var(--cards),1fr);gap:8px;margin:12px 0}.ruler-item{position:relative;min-height:58px;padding:10px;background:var(--panel);overflow:hidden}.ruler-item:after{content:"";position:absolute;left:0;bottom:0;width:var(--width);height:5px;background:var(--tone)}.ruler-item b{display:block;font:800 20px ui-monospace,SFMono-Regular,Menlo,monospace}.ruler-item span{font-size:10px;color:var(--muted)}.video-grid{display:grid;grid-template-columns:repeat(var(--cards),minmax(245px,1fr));gap:9px}.video-card{--tone:var(--line);padding:9px;background:var(--panel);border:1px solid var(--line);border-top:5px solid var(--tone)}.video-card.baseline{--tone:#73878d}.tone-top{--tone:var(--top)}.tone-bottom{--tone:var(--bottom)}.tone-same{--tone:var(--same)}.tone-future{--tone:var(--future)}.tone-past{--tone:var(--past)}.tone-m1{--tone:var(--m1)}.tone-m2{--tone:var(--m2)}.tone-m3{--tone:var(--m3)}.video-card h4{margin:2px 0 7px;font:900 13px ui-monospace,SFMono-Regular,Menlo,monospace}.calc{min-height:47px;color:var(--muted);font:10px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}.video-card video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#020608}.primary{display:flex;justify-content:space-between;align-items:center;margin-top:8px;padding:8px 10px;background:var(--panel2)}.primary b{font:900 18px ui-monospace,SFMono-Regular,Menlo,monospace}.primary span{font-size:9px;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--line);margin-top:7px}.metric{padding:7px;background:var(--panel2)}.metric b{display:block;font:800 13px ui-monospace,SFMono-Regular,Menlo,monospace}.metric span{font-size:9px;color:var(--muted)}details{margin-top:8px;font-size:11px;color:var(--muted)}summary{cursor:pointer;color:var(--ink);font-weight:800}.detail-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin-top:7px}.detail-grid div{padding:7px;background:var(--bg)}.pending{padding:15px;color:var(--muted);border:1px dashed var(--line)}footer{border-top:1px solid var(--line);padding:24px 0 50px;color:var(--muted);font-size:11px}@media(max-width:1180px){.case-head{grid-template-columns:150px 1fr}.case-analysis{grid-column:1/-1}.video-grid,.ruler{grid-template-columns:repeat(2,1fr)}.control-grid{grid-template-columns:1fr}}@media(max-width:700px){header,main{width:calc(100% - 14px)}.hero,.case-head,.video-grid,.ruler,.status{grid-template-columns:1fr}.case-analysis{grid-column:auto}h1{font-size:50px}.axis-title{display:block}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body><header><div class="nav"><a href="/">8092 总入口</a><a href="/object-query-information-flow-stage4?v=1">Stage 4 全矩阵</a><a href="/object-query-information-flow-validation?v=2">Stage 1–3</a></div><div class="eyebrow">LATEST3350 / CONTROLLED REPRESENTATIVE VIDEOS</div><div class="hero"><h1>固定两维，<br>只换一维。</h1><p>每组都使用同一个 case、seed、object 和 Baseline，只改变 Head group、时间方向或 M1/M2/M3 中的一项。页面同时展示支持性案例与反例；单个视频只负责说明“现象长什么样”，结论强度以 3-case 汇总证据为准。</p></div><div id="status" class="status"></div></header><main><section class="method"><h2>三种比较如何控制变量</h2><div class="control-grid"><article class="control" style="--axis:var(--head)"><b>① Head group</b><p>固定 M 和 Same/Future/Past，只比较 Top100 与 Bottom100。回答排名组是否改变干预效应。</p></article><article class="control" style="--axis:var(--time)"><b>② Time direction</b><p>固定 Head group 和 M，只比较 Same、Future、Past。回答通信时间方向是否改变结果。</p></article><article class="control" style="--axis:var(--flow)"><b>③ Information flow</b><p>固定 Head group 和时间方向，只比较 M1 R→R、M2 C→R、M3 R→C。回答不同流向携带的信息是否不同。</p></article></div><div class="warning"><b>读数规则：</b>所有指标均相对同 seed Baseline，越大表示干预效应越强，不等于相对 GT 更差。Center-ADE 只在轨迹门控通过时有效；门控失败必须改看 Track Loss、Identity Failure 和 Disappearance。当前只有 3 个独立 case，均属于探索性证据。</div><nav class="axis-nav"><a href="#axis-head">Head 代表例</a><a href="#axis-time">Time 代表例</a><a href="#axis-flow">M1/M2/M3 代表例</a></nav></section><div id="content"></div><footer>视频使用 preload=none，并在接近视口时才加载。每组的 Baseline、对照视频和指标均来自同一 case/seed/object。</footer></main><script>
const api='/api/object-query-information-flow-stage4-representatives',videoApi='/api/object-query-information-flow-stage4/asset',$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const F=(v,d=3)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'N/A';
const AXES={head:['HEAD GROUP','只换 Top100 / Bottom100'],time:['TIME DIRECTION','只换 Same / Future / Past'],flow:['INFORMATION FLOW','只换 M1 / M2 / M3']};
function media(kind,g,r={}){return `${videoApi}?${new URLSearchParams({kind,case:g.case,seed:String(g.seed),variant:r.variant_id||''})}`}
function lazy(root=document){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){const v=e.target;if(v.dataset.src){v.src=v.dataset.src;delete v.dataset.src;v.load()}io.unobserve(v)}}),{rootMargin:'420px'});root.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function details(r){const t=r.trajectory||{},s=r.survival||{},rows=[['Center-ADE / D0',t.center_ade_d0],['Center-FDE / D0',t.center_fde_d0],['Velocity / D0/frame',t.velocity_error_d0_per_frame],['100×(1−PCK@10%)',t.pck10_error_percent],['Track Loss %',t.track_loss_percent],['Identity Failure %',s.identity_failure_percent],['Disappearance %',s.disappearance_percent],['Mask Absence %',s.mask_absence_percent],['Area Failure %',s.area_failure_percent],['Terminal Missing %',s.terminal_missing_percent]];return `<details><summary>展开完整轨迹 / 身份 / 存活指标</summary><div class="detail-grid">${rows.map(([k,v])=>`<div><b>${F(v)}</b><br>${esc(k)}</div>`).join('')}</div></details>`}
function card(g,r){if(!r.ready)return `<article class="video-card tone-${esc(r.tone)}"><h4>${esc(r.label)}</h4><div class="pending">尚未生成</div></article>`;const t=r.trajectory||{},s=r.survival||{},main=[['Center-ADE',t.center_ade_d0],['Track Loss %',t.track_loss_percent],['Identity Fail %',s.identity_failure_percent],['Disappear %',s.disappearance_percent]];return `<article class="video-card tone-${esc(r.tone)}"><h4>${esc(r.label)}</h4><div class="calc">${esc(r.calculation)}</div><video controls muted loop playsinline preload="none" data-src="${esc(media('ablation',g,r))}"></video><div class="primary"><span>${esc(g.primary_label)}</span><b>${F(r.primary_value)}</b></div><div class="metrics">${main.map(([k,v])=>`<div class="metric"><b>${F(v)}</b><span>${esc(k)}</span></div>`).join('')}</div>${details(r)}</article>`}
function ruler(g){const values=g.rows.map(r=>Number(r.primary_value)).filter(Number.isFinite),max=Math.max(...values,1e-9);return g.rows.map(r=>{const v=Number(r.primary_value),w=Number.isFinite(v)?Math.max(2,100*v/max):0;return `<div class="ruler-item" style="--width:${w}%;--tone:var(--${esc(r.tone)})"><b>${F(v)}</b><span>${esc(r.label)} · ${esc(g.primary_label)}</span></div>`}).join('')}
function group(g,i){const cards=g.rows.length+1,full=`/object-query-information-flow-stage4?${new URLSearchParams({v:'1',case:g.case,seed:String(g.seed),target:`${g.target_scope}::${g.region||''}`})}`;return `<section class="case" style="--axis:var(--${esc(g.axis)});--cards:${cards}" id="${esc(g.id)}"><div class="case-head"><div><div class="case-label">${String(i+1).padStart(2,'0')} / ${esc(g.label)}</div><h3>${esc(g.title)}</h3><div class="identity">${esc(g.case)}<br>seed ${g.seed} · ${esc(g.region)}</div></div><div><div class="fixed"><b>控制变量：</b>${esc(g.fixed)}</div><p class="claim"><b>跨 case 证据：</b>${esc(g.claim)}</p><div class="actions"><button data-replay="${esc(g.id)}">同步重播</button><a href="${esc(full)}">打开完整矩阵</a></div></div><div class="case-analysis"><div class="evidence">${esc(g.evidence)}</div><p class="caveat"><b>证据边界：</b>${esc(g.caveat)}</p></div></div><div class="ruler">${ruler(g)}</div><div class="video-grid"><article class="video-card baseline"><h4>Baseline · no intervention</h4><div class="calc">共同 reference；同 seed、同初始噪声，不做 attention contribution subtraction</div><video controls muted loop playsinline preload="none" data-src="${esc(media('baseline',g))}"></video><div class="primary"><span>Reference</span><b>0</b></div></article>${g.rows.map(r=>card(g,r)).join('')}</div></section>`}
function axisBlock(axis,groups){const [title,note]=AXES[axis];return `<section class="axis-block" id="axis-${axis}"><div class="axis-title"><h2>${title}</h2><p>${note}</p></div>${groups.map(group).join('')}</section>`}
async function load(){const d=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),s=d.status;$('status').innerHTML=[['已生成',`${s.generated}/${s.expected}`],['分析记录',s.analysis_records],['独立 case',s.case_count],['case-seed',s.case_seed_count],['代表组',s.representative_groups]].map(([k,v])=>`<div class="stat"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join('');$('content').innerHTML=['head','time','flow'].map(a=>axisBlock(a,d.groups.filter(g=>g.axis===a))).join('');document.querySelectorAll('[data-replay]').forEach(b=>b.onclick=()=>document.querySelectorAll(`#${CSS.escape(b.dataset.replay)} video`).forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));lazy()}
load().catch(e=>$('content').innerHTML=`<div class="pending">读取失败：${esc(e)}</div>`);
</script></body></html>'''
