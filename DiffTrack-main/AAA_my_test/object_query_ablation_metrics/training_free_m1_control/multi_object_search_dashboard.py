#!/usr/bin/env python3
"""Live dashboard for the multi-object M1 contrast-guidance search."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_multi_object_search_v1"
)
MANIFEST_PATH = ROOT / "search_manifest.json"
GUIDED_ROOT = ROOT / "guided"
HEAD_ZERO_ROOT = ROOT.parent / "training_free_top100_full_head_output_zero_search_v1"
HEAD_ZERO_GUIDED_ROOT = HEAD_ZERO_ROOT / "guided"
M2_ROOT = ROOT.parent / "training_free_m2_multi_object_search_v1"
M3_ROOT = ROOT.parent / "training_free_m3_multi_object_search_v1"
HEAD_ZERO_CASES = (
    "0613pybullet_sample_000301_w000",
    "0613pybullet_sample_000331_w001",
    "0613pybullet_sample_000336_w001",
    "0613pybullet_sample_001455_w000",
    "0613pybullet_sample_001460_w002",
)
M2_M3_CASES = ("0613pybullet_sample_001460_w002",)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _variant(
    scale: float, start: int, end: int, family: str = "m1_multi"
) -> str:
    if family == "m1_multi":
        prefix = "multi_object_blockdiag__m1_all_time__top100"
    elif family == "m2_multi":
        prefix = "multi_object_independent__m2_all_time__top100"
    elif family == "m3_multi":
        prefix = "multi_object_independent__m3_all_time__top100"
    elif family == "head_zero":
        prefix = "all_token__full_head_output_zero__top100"
    else:
        raise ValueError(f"unknown guidance family: {family}")
    return f"{prefix}__pag{_tag(scale)}__denoise_{start:02d}_{end:02d}"


def _directory(
    case: str,
    seed: int,
    scale: float,
    start: int,
    end: int,
    family: str = "m1_multi",
) -> Path:
    roots = {
        "m1_multi": GUIDED_ROOT,
        "m2_multi": M2_ROOT / "guided",
        "m3_multi": M3_ROOT / "guided",
        "head_zero": HEAD_ZERO_GUIDED_ROOT,
    }
    try:
        root = roots[family]
    except KeyError as exc:
        raise ValueError(f"unknown guidance family: {family}") from exc
    return root / case / f"seed_{seed:05d}" / _variant(
        scale, start, end, family
    )


def _manifest() -> dict[str, Any]:
    return _read_json(MANIFEST_PATH)


def _sample_map(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.get("samples") or []:
        if not isinstance(row, dict):
            continue
        try:
            result[(str(row["case"]), int(row["seed"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _record(
    case: str,
    seed: int,
    scale: float,
    start: int,
    end: int,
    family: str = "m1_multi",
) -> dict[str, Any]:
    directory = _directory(case, seed, scale, start, end, family)
    video = directory / "generated.mp4"
    complete = directory / "complete.json"
    run_manifest = _read_json(directory / "manifest.json")
    audit = run_manifest.get("audit") if isinstance(run_manifest.get("audit"), dict) else {}
    block = audit.get("block_diagonal") if isinstance(audit.get("block_diagonal"), dict) else {}
    head_zero = (
        audit.get("all_token_head_output_zero")
        if isinstance(audit.get("all_token_head_output_zero"), dict)
        else {}
    )
    multi_flow = (
        audit.get("multi_object_flow")
        if isinstance(audit.get("multi_object_flow"), dict)
        else block
    )
    deltas = audit.get("perturbation_delta_l2_by_step") or {}
    finite_deltas: list[float] = []
    if isinstance(deltas, dict):
        for value in deltas.values():
            try:
                finite_deltas.append(float(value))
            except (TypeError, ValueError):
                pass
    error_path = directory / "error.json"
    if not error_path.is_file():
        error_path = directory / "error.txt"
    if complete.is_file() and video.is_file():
        state = "complete"
    elif error_path.is_file():
        state = "error"
    elif directory.is_dir():
        state = "running"
    else:
        state = "pending"
    return {
        "family": family,
        "family_label": (
            {
                "m1_multi": "M1 multi-object R→R",
                "m2_multi": "M2 multi-object C→R",
                "m3_multi": "M3 multi-object R→C",
                "head_zero": "Top100 full-head output zero",
            }[family]
        ),
        "scale": scale,
        "window": [start, end],
        "window_key": f"{start:02d}_{end:02d}",
        "variant": _variant(scale, start, end, family),
        "state": state,
        "ready": state == "complete",
        "object_count": int(run_manifest.get("object_count") or 0),
        "object_regions": list(run_manifest.get("object_regions") or []),
        "modified_head_events": int(audit.get("modified_head_events") or 0),
        "expected_modified_head_events": int(audit.get("expected_modified_head_events") or 0),
        "flow_id": str(multi_flow.get("flow_id") or ("M1" if family == "m1_multi" else "")),
        "deleted_pairs_per_head": int(multi_flow.get("deleted_pair_count_per_head") or 0),
        "overlap_token_count": int(multi_flow.get("overlap_token_count") or 0),
        "duplicate_pair_subtractions_prevented": int(
            multi_flow.get("duplicate_pair_subtractions_prevented") or 0
        ),
        "object_specific_complements": bool(
            multi_flow.get("object_specific_complements")
        ),
        "all_query_tokens": bool(head_zero.get("all_query_tokens")),
        "removed_flows": list(head_zero.get("removed_flows") or []),
        "mean_prediction_delta_l2": (
            sum(finite_deltas) / len(finite_deltas) if finite_deltas else None
        ),
        "error": (
            error_path.read_text(encoding="utf-8", errors="replace")[-1000:]
            if error_path.is_file()
            else ""
        ),
    }


def catalog() -> dict[str, Any]:
    manifest = _manifest()
    samples = _sample_map(manifest)
    grid = manifest.get("search_grid") if isinstance(manifest.get("search_grid"), dict) else {}
    scales = [float(value) for value in grid.get("pag_scales") or []]
    windows = [
        [int(value[0]), int(value[1])]
        for value in grid.get("guidance_windows_inclusive") or []
        if isinstance(value, list) and len(value) == 2
    ]
    cases = list(dict.fromkeys(case for case, _ in samples))
    seeds = [int(value) for value in manifest.get("seeds") or []]
    rows: list[dict[str, Any]] = []
    case_complete = {
        case: {family: 0 for family in ("m1_multi", "m2_multi", "m3_multi", "head_zero")}
        for case in cases
    }
    family_complete = {family: 0 for family in case_complete[cases[0]]}
    family_error = {family: 0 for family in case_complete[cases[0]]}
    for case in cases:
        for seed in seeds:
            sample = samples.get((case, seed))
            if sample is None:
                continue
            records = [
                _record(case, seed, scale, start, end, "m1_multi")
                for start, end in windows
                for scale in scales
            ]
            m2_records = (
                [
                    _record(case, seed, scale, start, end, "m2_multi")
                    for start, end in windows
                    for scale in scales
                ]
                if case in M2_M3_CASES
                else []
            )
            m3_records = (
                [
                    _record(case, seed, scale, start, end, "m3_multi")
                    for start, end in windows
                    for scale in scales
                ]
                if case in M2_M3_CASES
                else []
            )
            head_zero_records = (
                [
                    _record(case, seed, scale, start, end, "head_zero")
                    for start, end in windows
                    for scale in scales
                ]
                if case in HEAD_ZERO_CASES
                else []
            )
            family_records = {
                "m1_multi": records,
                "m2_multi": m2_records,
                "m3_multi": m3_records,
                "head_zero": head_zero_records,
            }
            completed = {
                family: sum(record["ready"] for record in items)
                for family, items in family_records.items()
            }
            errors = {
                family: sum(record["state"] == "error" for record in items)
                for family, items in family_records.items()
            }
            for family in family_records:
                family_complete[family] += completed[family]
                family_error[family] += errors[family]
                case_complete[case][family] += completed[family]
            baseline = Path(str(sample.get("baseline_video") or ""))
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "caption": str(sample.get("caption") or ""),
                    "objects": [
                        str(region.get("region_name") or "")
                        for region in sample.get("regions") or []
                        if isinstance(region, dict)
                    ],
                    "baseline_ready": baseline.is_file(),
                    "records": records,
                    "m2_records": m2_records,
                    "m3_records": m3_records,
                    "head_zero_records": head_zero_records,
                    "progress": {
                        "complete": completed["m1_multi"],
                        "expected": len(scales) * len(windows),
                        "errors": errors["m1_multi"],
                        "m2_complete": completed["m2_multi"],
                        "m2_expected": len(m2_records),
                        "m2_errors": errors["m2_multi"],
                        "m3_complete": completed["m3_multi"],
                        "m3_expected": len(m3_records),
                        "m3_errors": errors["m3_multi"],
                        "head_zero_complete": completed["head_zero"],
                        "head_zero_expected": len(head_zero_records),
                        "head_zero_errors": errors["head_zero"],
                    },
                }
            )
    expected = len(rows) * len(scales) * len(windows)
    head_zero_expected = sum(
        len(scales) * len(windows)
        for row in rows
        if row["case"] in HEAD_ZERO_CASES
    )
    m2_m3_expected = len(M2_M3_CASES) * len(seeds) * len(scales) * len(windows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest.get("experiment_id"),
        "cases": cases,
        "seeds": seeds,
        "scales": scales,
        "windows": windows,
        "rows": rows,
        "controlled": manifest.get("controlled") or {},
        "head_zero_cases": list(HEAD_ZERO_CASES),
        "m2_m3_cases": list(M2_M3_CASES),
        "families": {
            "m1_multi": {
                "label": "M1 multi-object R→R",
                "definition": "delete union_i A[R_i,R_i]V[R_i]; preserve cross-object and C flows",
            },
            "m2_multi": {
                "label": "M2 multi-object C→R",
                "definition": "delete union_i A[R_i,C_i]V[C_i], C_i=Omega\\R_i; other objects are part of C_i",
            },
            "m3_multi": {
                "label": "M3 multi-object R→C",
                "definition": "delete union_i A[C_i,R_i]V[R_i], C_i=Omega\\R_i; broadcast to other objects is removed",
            },
            "head_zero": {
                "label": "Top100 full-head output zero",
                "definition": "set O_h=A_hV_h=0 at every query token; remove R→R, C→R, R→C, C→C together",
            },
        },
        "progress": {
            "guided_complete": family_complete["m1_multi"],
            "guided_expected": expected,
            "baseline_complete": sum(row["baseline_ready"] for row in rows),
            "baseline_expected": len(rows),
            "errors": family_error["m1_multi"],
            "m2_complete": family_complete["m2_multi"],
            "m2_expected": m2_m3_expected,
            "m2_errors": family_error["m2_multi"],
            "m3_complete": family_complete["m3_multi"],
            "m3_expected": m2_m3_expected,
            "m3_errors": family_error["m3_multi"],
            "head_zero_complete": family_complete["head_zero"],
            "head_zero_expected": head_zero_expected,
            "head_zero_errors": family_error["head_zero"],
            "by_case": [
                {
                    "case": case,
                    "complete": case_complete[case]["m1_multi"],
                    "expected": len(seeds) * len(scales) * len(windows),
                    "m2_complete": case_complete[case]["m2_multi"],
                    "m2_expected": (
                        len(seeds) * len(scales) * len(windows)
                        if case in M2_M3_CASES
                        else 0
                    ),
                    "m3_complete": case_complete[case]["m3_multi"],
                    "m3_expected": (
                        len(seeds) * len(scales) * len(windows)
                        if case in M2_M3_CASES
                        else 0
                    ),
                    "head_zero_complete": case_complete[case]["head_zero"],
                    "head_zero_expected": (
                        len(seeds) * len(scales) * len(windows)
                        if case in HEAD_ZERO_CASES
                        else 0
                    ),
                }
                for case in cases
            ],
        },
    }


def asset(
    kind: str,
    case: str,
    seed: int,
    scale: float = 0.0,
    start: int = 0,
    end: int = 0,
) -> Path | None:
    manifest = _manifest()
    samples = _sample_map(manifest)
    sample = samples.get((case, seed))
    if sample is None:
        return None
    if kind == "baseline":
        path = Path(str(sample.get("baseline_video") or ""))
        return path if path.is_file() else None
    grid = manifest.get("search_grid") if isinstance(manifest.get("search_grid"), dict) else {}
    allowed_scales = [float(value) for value in grid.get("pag_scales") or []]
    allowed_windows = [
        (int(value[0]), int(value[1]))
        for value in grid.get("guidance_windows_inclusive") or []
        if isinstance(value, list) and len(value) == 2
    ]
    family = {
        "guided": "m1_multi",
        "m1_multi": "m1_multi",
        "m2_multi": "m2_multi",
        "m3_multi": "m3_multi",
        "head_zero": "head_zero",
    }.get(kind)
    if (
        family is None
        or scale not in allowed_scales
        or (start, end) not in allowed_windows
        or (family == "head_zero" and case not in HEAD_ZERO_CASES)
        or (family in {"m2_multi", "m3_multi"} and case not in M2_M3_CASES)
    ):
        return None
    directory = _directory(case, seed, scale, start, end, family)
    path = directory / "generated.mp4"
    return path if (directory / "complete.json").is_file() and path.is_file() else None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top100 M1/M2/M3 Guidance Flow Comparison</title><style>
:root{--bg:#dce7e9;--paper:#f5f9f8;--ink:#142a31;--muted:#62747a;--line:#93a8ad;--deep:#173d48;--cyan:#117f88;--red:#b1455b;--amber:#c2901e;--violet:#6746a5;--complete:#247457}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(90deg,#173d480c 1px,transparent 1px),linear-gradient(#173d480c 1px,transparent 1px),var(--bg);background-size:22px 22px;color:var(--ink);font-family:"Aptos Narrow","Noto Sans CJK SC",Arial,sans-serif}a{color:var(--cyan);font-weight:800}header,main,footer{width:min(1800px,calc(100% - 28px));margin:auto}header{padding:24px 0 18px}.eyebrow{margin-top:20px;color:var(--red);font:900 11px ui-monospace,monospace;letter-spacing:.18em}.hero{display:grid;grid-template-columns:minmax(380px,.82fr) minmax(560px,1.18fr);gap:34px;align-items:end}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{margin:8px 0 0;font-size:clamp(46px,7vw,96px);line-height:.85;letter-spacing:-.07em}.lead{margin:0;max-width:940px;font-size:17px;line-height:1.65}.equation{margin-top:18px;padding:13px 16px;background:var(--deep);color:#edf8f8;font:13px/1.65 ui-monospace,monospace}.equation b{color:#73d5d5}.equation em{color:#cdb9ff;font-style:normal}.window-ruler{display:grid;grid-template-columns:5fr 5fr 10fr 20fr;margin-top:16px;border:1px solid var(--line);background:var(--paper)}.window-ruler span{position:relative;padding:10px;text-align:center;border-right:1px solid var(--line);font:800 10px ui-monospace,monospace}.window-ruler span:last-child{border:0}.window-ruler span:after{content:"";position:absolute;left:0;bottom:0;height:4px;width:100%;background:var(--cyan);opacity:calc(.35 + var(--i)*.15)}.toolbar{position:sticky;top:0;z-index:6;display:flex;gap:9px;align-items:end;flex-wrap:wrap;padding:11px;margin-top:16px;border:1px solid var(--line);background:#dce7e9ef;backdrop-filter:blur(9px)}label{display:grid;gap:4px;color:var(--muted);font:900 10px ui-monospace,monospace;letter-spacing:.08em}select,button{min-height:37px;padding:7px 10px;border:1px solid var(--line);background:white;color:var(--ink);font-weight:800}select#case{max-width:min(650px,75vw)}button{cursor:pointer}.status{margin-left:auto;font:800 11px ui-monospace,monospace}.summary{display:grid;grid-template-columns:1.35fr repeat(5,1fr);gap:9px;margin:18px 0}.summary article{min-height:100px;padding:13px 15px;border:1px solid var(--line);background:var(--paper)}.summary strong{display:block;font:900 31px "Arial Black",sans-serif}.summary span{color:var(--muted);font:11px/1.5 ui-monospace,monospace}.summary .case-summary strong{font-size:19px;line-height:1.25}.baseline{margin:20px 0 24px;padding:13px;border:1px solid var(--line);background:var(--paper)}.section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;padding-bottom:8px;border-bottom:3px solid var(--ink)}.section-head h2{margin:0;font-size:27px}.section-head p{margin:0;color:var(--muted);font:11px ui-monospace,monospace}.baseline-grid{display:grid;grid-template-columns:minmax(300px,580px) 1fr;gap:14px;margin-top:11px}.baseline-copy{padding:8px 5px}.baseline-copy h3{margin:0 0 8px;font-size:20px}.baseline-copy p{margin:5px 0;color:var(--muted);line-height:1.55}.window{margin:26px 0 38px}.window-label{display:flex;align-items:end;gap:18px;padding-bottom:9px;border-bottom:3px solid var(--deep)}.window-label .step{font:900 45px/.9 "Arial Black",sans-serif;color:var(--cyan)}.window-label h2{margin:0;font-size:26px}.window-label p{margin:3px 0 0;color:var(--muted)}.compare-head{display:grid;grid-template-columns:120px repeat(2,minmax(280px,1fr));gap:10px;margin-top:11px}.compare-head span{padding:8px 10px;border-bottom:4px solid var(--cyan);font:900 11px ui-monospace,monospace}.compare-head span:last-child{border-color:var(--violet)}.compare-row{display:grid;grid-template-columns:120px 1fr;gap:10px;margin-top:10px}.compare-label{display:grid;place-content:center;padding:10px;border:1px solid var(--line);background:var(--deep);color:white;text-align:center;font:900 12px/1.55 ui-monospace,monospace}.compare-cards{display:grid;grid-template-columns:repeat(4,minmax(250px,1fr));gap:10px}.compare-cards.count-1{grid-template-columns:minmax(280px,1fr)}.compare-cards.count-2{grid-template-columns:repeat(2,minmax(280px,1fr))}.compare-cards.count-3{grid-template-columns:repeat(3,minmax(260px,1fr))}.card{padding:9px;border:1px solid var(--line);border-top:6px solid var(--cyan);background:var(--paper)}.card.m2-multi{border-top-color:var(--amber)}.card.m3-multi{border-top-color:var(--red)}.card.head-zero{border-top-color:var(--violet)}.card h3{display:flex;justify-content:space-between;gap:8px;margin:1px 0 8px;font-size:16px}.family{display:block;margin-bottom:6px;color:var(--cyan);font:900 10px ui-monospace,monospace;letter-spacing:.08em}.m2-multi .family{color:var(--amber)}.m3-multi .family{color:var(--red)}.head-zero .family{color:var(--violet)}.badge{color:var(--complete);font:900 10px ui-monospace,monospace}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#071318}.facts{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.facts span{padding:4px 6px;background:#e3ebec;font:10px ui-monospace,monospace}.card details{margin-top:7px;color:var(--muted);font:11px/1.5 ui-monospace,monospace}.empty{margin-top:11px;padding:24px;border:1px dashed var(--line);background:#f5f9f880;color:var(--muted);text-align:center}.case-progress{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin:14px 0 25px}.case-progress a{display:grid;gap:5px;padding:8px;border:1px solid var(--line);background:var(--paper);color:var(--ink);text-decoration:none;font:10px ui-monospace,monospace;overflow:hidden}.case-progress a.active{outline:3px solid var(--cyan);outline-offset:-3px}.case-progress b{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.meter{height:4px;background:#d1dcde}.meter i{display:block;height:100%;background:var(--cyan)}.meter.m2 i{background:var(--amber)}.meter.m3 i{background:var(--red)}.meter.zero i{background:var(--violet)}footer{padding:0 0 45px;color:var(--muted);font:11px/1.6 ui-monospace,monospace}@media(max-width:1450px){.compare-cards{grid-template-columns:repeat(2,minmax(280px,1fr))}}@media(max-width:1100px){.hero,.summary,.baseline-grid{grid-template-columns:1fr}.case-progress{grid-template-columns:repeat(3,1fr)}.status{width:100%;margin-left:0}}@media(max-width:760px){header,main,footer{width:calc(100% - 12px)}.case-progress{grid-template-columns:1fr}.window-label{align-items:start}.summary{grid-template-columns:1fr 1fr}.summary .case-summary{grid-column:1/-1}.compare-head{display:none}.compare-row{grid-template-columns:1fr}.compare-cards{grid-template-columns:1fr}.compare-label{place-content:start;text-align:left}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body><header><a href="/">← 返回 8092 总入口</a> · <a href="/training-free-m1-control?v=2">单对象 M1 控制</a><div class="eyebrow">LATEST3350 TOP100 · SAME CASE / SEED / λ / WINDOW · FOUR-FLOW LIVE CONTROL</div><div class="hero"><h1>对象通信，<br>四种切法</h1><p class="lead">严格固定首帧、prompt、seed、初始噪声、CFG=5 和 40 个采样步。同一行只比较相同 guidance window 与 λ：M1 删除对象内部通信，M2 删除每个对象接收的外部输入，M3 删除每个对象向外广播的输出，Full-head zero 关闭同一 Top100 的整颗 head。</p></div><div class="equation"><b>M1 · Rᵢ→Rᵢ</b>: εᵤ + 5(εc−εᵤ) + λ(εc−εM1)　|　⋃ᵢ A[Rᵢ,Rᵢ]V[Rᵢ]=0<br><strong style="color:var(--amber)">M2 · Cᵢ→Rᵢ</strong>: εᵤ + 5(εc−εᵤ) + λ(εc−εM2)　|　⋃ᵢ A[Rᵢ,Cᵢ]V[Cᵢ]=0，Cᵢ=Ω∖Rᵢ<br><strong style="color:#ef93a6">M3 · Rᵢ→Cᵢ</strong>: εᵤ + 5(εc−εᵤ) + λ(εc−εM3)　|　⋃ᵢ A[Cᵢ,Rᵢ]V[Rᵢ]=0<br><em>Full-head zero</em>: εᵤ + 5(εc−εᵤ) + λ(εc−εhead-zero)　|　Oₕ=AₕVₕ=0，四类流同时删除</div><div class="window-ruler"><span style="--i:0">STEP 0–4</span><span style="--i:1">STEP 0–9</span><span style="--i:2">STEP 0–19</span><span style="--i:3">STEP 0–39</span></div><div class="toolbar"><label>CASE<select id="case"></select></label><label>SEED<select id="seed"></select></label><button id="refresh">刷新已生成结果</button><button id="replay">同步重播</button><span id="status" class="status">读取中…</span></div></header><main><section id="summary" class="summary"></section><nav id="caseProgress" class="case-progress"></nav><section id="baseline" class="baseline"></section><div id="windows"></div></main><footer>页面每 20 秒扫描 M1、M2、M3 与 Full-head-zero 的 complete.json。Baseline 只显示一次；比较区仅加载当前 case×seed 已完成的视频，不为未生成项保留空卡。</footer><script>
const api='/api/training-free-m1-multi-object-search',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null,initialized=false;
const fmt=v=>(Number(v)>0?'+':'')+Number(v).toFixed(Math.abs(Number(v))===1?0:1);
function opts(node,items){const old=node.value;node.innerHTML=items.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');if([...node.options].some(o=>o.value===old))node.value=old}
function url(kind,row,r=null){const p={kind,case:row.case,seed:row.seed};if(r)Object.assign(p,{scale:r.scale,start:r.window[0],end:r.window[1]});return `${api}/asset?${new URLSearchParams(p)}`}
function lazy(){const io=new IntersectionObserver(xs=>xs.forEach(x=>{if(x.isIntersecting){const v=x.target;v.src=v.dataset.src;v.load();io.unobserve(v)}}),{rootMargin:'650px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function syncUrl(){const u=new URL(location.href);u.searchParams.set('case',$('case').value);u.searchParams.set('seed',$('seed').value);u.searchParams.set('v','3');history.replaceState(null,'',u)}
function card(family,row,r){const meta={m1_multi:{cls:'',label:'M1 MULTI · Rᵢ→Rᵢ'},m2_multi:{cls:'m2-multi',label:'M2 MULTI · Cᵢ→Rᵢ'},m3_multi:{cls:'m3-multi',label:'M3 MULTI · Rᵢ→Cᵢ'},head_zero:{cls:'head-zero',label:'FULL-HEAD ZERO · ALL FLOWS'}}[family],zero=family==='head_zero',direction=r.scale<0?'prediction 靠近对应扰动':'prediction 远离对应扰动',audit=zero?`all query tokens: ${r.all_query_tokens?'YES':'N/A'}<br>removed flows: ${esc((r.removed_flows||[]).join(' / ')||'N/A')}`:`flow: ${esc(r.flow_id||'M1')}<br>deleted pairs/head: ${r.deleted_pairs_per_head||'N/A'}<br>object-specific complements: ${r.object_specific_complements?'YES':'NO'}<br>overlap tokens: ${r.overlap_token_count}<br>duplicate subtraction prevented: ${r.duplicate_pair_subtractions_prevented}`;return `<article class="card ${meta.cls}"><span class="family">${meta.label}</span><h3>λ ${fmt(r.scale)}<span class="badge">COMPLETE</span></h3><video controls muted loop playsinline preload="none" data-src="${esc(url(family,row,r))}"></video><div class="facts"><span>${esc(direction)}</span><span>step ${r.window[0]}–${r.window[1]}</span><span>Top100</span>${r.modified_head_events?`<span>${r.modified_head_events} head-events</span>`:''}</div><details><summary>精确运行审计</summary><div>${audit}<br>mean ‖εc−εperturbed‖₂: ${r.mean_prediction_delta_l2==null?'N/A':Number(r.mean_prediction_delta_l2).toFixed(3)}</div></details></article>`}
function render(){
const caseName=$('case').value,seed=Number($('seed').value),row=data.rows.find(x=>x.case===caseName&&x.seed===seed);if(!row)return;syncUrl();
const allErrors=data.progress.errors+data.progress.m2_errors+data.progress.m3_errors+data.progress.head_zero_errors;
$('status').textContent=`M1 ${data.progress.guided_complete}/${data.progress.guided_expected} · M2 ${data.progress.m2_complete}/${data.progress.m2_expected} · M3 ${data.progress.m3_complete}/${data.progress.m3_expected} · Head-zero ${data.progress.head_zero_complete}/${data.progress.head_zero_expected} · errors ${allErrors} · ${new Date(data.generated_at_utc).toLocaleTimeString()}`;
$('summary').innerHTML=`<article class="case-summary"><strong>${esc(row.case)}</strong><span>${esc(row.objects.join(' + '))}<br>${esc(row.caption)}</span></article><article><strong>${row.progress.complete}/${row.progress.expected}</strong><span>当前 case×seed · M1</span></article><article><strong>${row.progress.m2_complete}/${row.progress.m2_expected||0} · ${row.progress.m3_complete}/${row.progress.m3_expected||0}</strong><span>当前 case×seed · M2 / M3</span></article><article><strong>${row.progress.head_zero_complete}/${row.progress.head_zero_expected||0}</strong><span>当前 case×seed · Full-head zero</span></article><article><strong>${data.progress.m2_complete}/${data.progress.m2_expected} · ${data.progress.m3_complete}/${data.progress.m3_expected}</strong><span>M2 / M3 全局进度</span></article><article><strong>${allErrors}</strong><span>四组错误记录</span></article>`;
$('caseProgress').innerHTML=data.progress.by_case.map(x=>`<a class="${x.case===row.case?'active':''}" href="?case=${encodeURIComponent(x.case)}&seed=${row.seed}&v=3"><b>${esc(x.case)}</b><span>M1 ${x.complete}/${x.expected}${x.m2_expected?` · M2 ${x.m2_complete}/${x.m2_expected} · M3 ${x.m3_complete}/${x.m3_expected}`:''}${x.head_zero_expected?` · Zero ${x.head_zero_complete}/${x.head_zero_expected}`:''}</span><div class="meter"><i style="width:${100*x.complete/Math.max(1,x.expected)}%"></i></div>${x.m2_expected?`<div class="meter m2"><i style="width:${100*x.m2_complete/x.m2_expected}%"></i></div><div class="meter m3"><i style="width:${100*x.m3_complete/x.m3_expected}%"></i></div>`:''}${x.head_zero_expected?`<div class="meter zero"><i style="width:${100*x.head_zero_complete/x.head_zero_expected}%"></i></div>`:''}</a>`).join('');
$('baseline').innerHTML=`<div class="section-head"><h2>共同 Baseline · seed ${row.seed}</h2><p>λ=0；同 seed、同首帧、同 prompt、同初始噪声</p></div><div class="baseline-grid">${row.baseline_ready?`<video controls muted loop playsinline preload="metadata" src="${esc(url('baseline',row))}"></video>`:'<div class="empty">Baseline 尚未落盘</div>'}<div class="baseline-copy"><h3>一个 Baseline，四种通信切口</h3><p>青色 M1 删除对象内部 Rᵢ→Rᵢ；琥珀色 M2 删除每个对象从 Cᵢ 接收的输入；红色 M3 删除每个对象向 Cᵢ 的广播；紫色把相同 Top100 heads 的完整 Oₕ=AₕVₕ 置零。</p><p>M2/M3 中 Cᵢ=Ω∖Rᵢ，因此其他对象属于当前对象的 Cᵢ。MSE 与 CoTracker trajectory loss 只用于生成后评价，不进入 guidance。</p></div></div>`;
const specs=[{family:'m1_multi',key:'records',label:'M1'},{family:'m2_multi',key:'m2_records',label:'M2'},{family:'m3_multi',key:'m3_records',label:'M3'},{family:'head_zero',key:'head_zero_records',label:'Zero'}].filter(s=>(row[s.key]||[]).length);
const blocks=data.windows.map((w,i)=>{const rows=data.scales.map(scale=>{const cards=specs.map(s=>{const r=row[s.key].find(x=>x.window[0]===w[0]&&x.window[1]===w[1]&&x.scale===scale&&x.ready);return r?card(s.family,row,r):''}).filter(Boolean);if(!cards.length)return '';return `<section class="compare-row"><div class="compare-label">λ ${fmt(scale)}<br>STEP ${w[0]}–${w[1]}</div><div class="compare-cards count-${cards.length}">${cards.join('')}</div></section>`}).join('');if(!rows)return '';const counts=specs.map(s=>`${s.label} ${row[s.key].filter(r=>r.window[0]===w[0]&&r.window[1]===w[1]&&r.ready).length}/${data.scales.length}`).join(' · ');return `<section class="window"><div class="window-label"><span class="step">${String(i+1).padStart(2,'0')}</span><div><h2>Guidance step ${w[0]}–${w[1]}</h2><p>${counts} · 其余去噪步保持 clean CFG</p></div></div>${rows}</section>`}).join('');
$('windows').innerHTML=blocks||'<div class="empty">该 case×seed 尚未生成 guidance 视频；Baseline 仍可查看。</div>';lazy()
}
async function load(){data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());const oldCase=$('case').value,oldSeed=$('seed').value;opts($('case'),data.cases);opts($('seed'),data.seeds.map(String));if(!initialized){if(q.get('case')&&data.cases.includes(q.get('case')))$('case').value=q.get('case');if(q.get('seed')&&data.seeds.map(String).includes(q.get('seed')))$('seed').value=q.get('seed');initialized=true}else{if(data.cases.includes(oldCase))$('case').value=oldCase;if(data.seeds.map(String).includes(oldSeed))$('seed').value=oldSeed}render()}
$('case').addEventListener('change',render);$('seed').addEventListener('change',render);$('refresh').addEventListener('click',load);$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load().catch(e=>$('status').textContent=`读取失败：${e}`);setInterval(()=>load().catch(()=>{}),20000);
</script></body></html>'''
