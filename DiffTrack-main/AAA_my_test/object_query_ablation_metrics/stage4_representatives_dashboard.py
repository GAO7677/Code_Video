#!/usr/bin/env python3
"""Focused Stage-4 representative-case dashboard."""

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

BALL_CASE = (
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-"
    "ball-and-block-fall_motion_to_end"
)
PYBULLET_CASE = "0613pybullet_sample_001460_w002"
SCOPE_ORDER = ("top100", "bottom100", "random100_layer_matched_draw0")
SCOPE_LABELS = {
    "top100": "latest3350 Top100",
    "bottom100": "latest3350 Bottom100",
    "random100_layer_matched_draw0": "Layer-matched Random100",
}

REPRESENTATIVES: tuple[dict[str, Any], ...] = (
    {
        "id": "m1-future-top-specific",
        "label": "最强正例",
        "case": BALL_CASE,
        "seed": 47326,
        "target_scope": "single_object",
        "region": "object_C",
        "mask_mode": "self_future",
        "title": "M1-future · Top100 的 R→R 跨帧贡献",
        "flow": "删除 t_k < t_q 的 R K/V → R Query，即对象历史状态到未来对象状态的贡献。",
        "claim": (
            "在相同 case、seed=47326、object_C 和 M1-future 下，相比消融 Bottom100，"
            "消融 Top100 的 Target-local 高 20.588（16.7×）；相比 Random100 高 20.528（16.0×）。"
            "因此，这个代表例支持 Top100 的 R→R Future contribution 更强。"
        ),
        "evidence": (
            "Target-local：Top100 21.895，Bottom100 1.307，Random100 1.367；"
            "Top100 约为 Bottom100 的 16.7 倍。"
        ),
        "caveat": (
            "支持 latest3350 Top100 含强 R→R contribution；不能单独证明它是因果时间专属，"
            "也不能排除 Top100 实际删除 dose 更大的解释。"
        ),
        "primary_metric": "target_local",
    },
    {
        "id": "m1-past-reverse-control",
        "label": "反向时间控制",
        "case": BALL_CASE,
        "seed": 47326,
        "target_scope": "single_object",
        "region": "object_C",
        "mask_mode": "self_past",
        "title": "M1-past · Future→Past 也产生强效应",
        "flow": "删除 t_k > t_q 的 R K/V → R Query，作为 M1-future 的反向时间控制。",
        "claim": (
            "在相同 case、seed=47326、object_C 和 M1-past 下，相比消融 Bottom100，"
            "消融 Top100 的 Target-local 高 20.851（20.3×）；相比 Random100 高 20.066（11.8×）。"
            "同时，Top100-M1-past 与 Top100-M1-future 仅相差 0.035，因此该 seed 不支持 Future 明显强于 Past。"
        ),
        "evidence": (
            "Target-local：Top100 21.929，Bottom100 1.078，Random100 1.864；"
            "Top100 约为 Bottom100 的 20.3 倍。"
        ),
        "caveat": "Future 与 Past 的差异需要严格配对统计；只看两条高分不能判断方向性。",
        "primary_metric": "target_local",
    },
    {
        "id": "m1-same-ranking-boundary",
        "label": "时间方向反例",
        "case": BALL_CASE,
        "seed": 47326,
        "target_scope": "single_object",
        "region": "object_C",
        "mask_mode": "self_same",
        "title": "M1-same · 同帧 R→R 并非 Top100 独占",
        "flow": "只删除 t_k = t_q 的 R K/V → R Query；跨帧 R→R 保留。",
        "claim": (
            "在相同 case、seed=47326、object_C 和 M1-same 下，相比 Bottom100，"
            "Top100 的 Target-local 只高 0.047（约 0.2%）；但两者分别是 Random100 的 5.37× 和 5.35×。"
            "因此，只能说这个 seed 的同帧 M1 中 Top100 与 Bottom100 接近，不能推广为跨 seed 结论。"
        ),
        "evidence": (
            "Target-local：Top100 21.203，Bottom100 21.156，Random100 3.951；"
            "Top/Bottom 比仅 1.00 倍。"
        ),
        "caveat": "该结果说明同帧对象内部贡献分布更广；不意味着 Bottom100 在整体 PCK 定义上更好。",
        "primary_metric": "target_local",
    },
    {
        "id": "m3-same-random-exception",
        "label": "Random100 异常反例",
        "case": BALL_CASE,
        "seed": 47326,
        "target_scope": "single_object",
        "region": "object_C",
        "mask_mode": "outgoing_same",
        "title": "M3-same · 单个 Random100 draw 反而最强",
        "flow": "只删除 t_k = t_q 的 R K/V → C Query，即对象状态向同帧其余 token 的广播。",
        "claim": (
            "在相同 case、seed=47326、object_C 和 M3-same 下，相比消融 Top100，"
            "消融这个 Random100 draw 的 Target-local 高 18.560（8.0×）；相比 Bottom100 高 18.524（7.9×）。"
            "因此，这个 draw 是 Top/Bottom ranking 的明确反例，但不能代表 Random100 总体。"
        ),
        "evidence": (
            "Target-local：Random100 21.209，Top100 2.650，Bottom100 2.686；"
            "Random100 约为 Top100 的 8.0 倍。"
        ),
        "caveat": "只有一个 layer-matched Random100 draw；必须增加随机 draws 才能判断是否稳定。",
        "primary_metric": "target_local",
    },
    {
        "id": "m3-future-spillover",
        "label": "对外传播候选",
        "case": BALL_CASE,
        "seed": 90094,
        "target_scope": "single_object",
        "region": "object_C",
        "mask_mode": "outgoing_future",
        "title": "M3-future · R→C 的时序与对象外变化",
        "flow": "删除 t_k < t_q 的 R K/V → C Query，即历史对象状态到未来环境/其他 token 的广播。",
        "claim": (
            "在相同 case、seed=90094、object_C 和 M3-future 下，相比 Bottom100，"
            "Top100 的 Outside 高 0.284（1.36×）、Temporal pixel 高 2.886（1.69×）；"
            "相比 Random100，分别高 0.561（2.11×）和 3.829（2.19×）。"
            "因此，它只是该 seed 下较强的 R→C spillover 候选。"
        ),
        "evidence": (
            "Temporal：Top/Bottom/Random = 7.057/4.171/3.228；"
            "Outside = 1.066/0.782/0.505。"
        ),
        "caveat": (
            "Temporal 与 Outside 当前仍是像素代理；在 Other-object trajectory 完成前，"
            "只能称为传播候选，不能称为物理状态传播证明。"
        ),
        "primary_metric": "outside_spillover",
    },
    {
        "id": "m2-future-object-change",
        "label": "C→R 对象变化",
        "case": PYBULLET_CASE,
        "seed": 13248,
        "target_scope": "single_object",
        "region": "object_A",
        "mask_mode": "incoming_future",
        "title": "M2-future · 环境输入缺失后的对象变化",
        "flow": "删除 t_k < t_q 的 C K/V → R Query，即历史环境/其他 token 到未来对象状态的输入。",
        "claim": (
            "在相同 case、seed=13248、object_A 和 M2-future 下，相比消融 Top100，"
            "消融 Bottom100 的 Target-local 高 3.293（1.40×）；相比 Random100 高 3.321（1.41×）。"
            "因此，该代表例中 Bottom100 对目标区域的综合改变最大，但不能仅凭该指标判为外观变化。"
        ),
        "evidence": (
            "Target-local：Top100 8.190，Bottom100 11.483，Random100 8.162；"
            "Bottom100 是 Top100 的 1.40 倍。"
        ),
        "caveat": "冻结 ROI 同时混合位置、形状和外观；需结合轨迹与 DINO/LPIPS 才能拆开解释。",
        "primary_metric": "target_local",
    },
    {
        "id": "m2-same-low-effect-control",
        "label": "低影响负对照",
        "case": BALL_CASE,
        "seed": 13248,
        "target_scope": "single_object",
        "region": "object_A",
        "mask_mode": "incoming_same",
        "title": "M2-same · 已执行消融但输出响应很弱",
        "flow": "只删除 t_k = t_q 的 C K/V → R Query。",
        "claim": (
            "在相同 case、seed=13248、object_A 和 M2-same 下，Top100、Bottom100、Random100 的综合影响"
            "最大只相差 0.013（最大/最小仅 1.02×），且三者都低于 0.60。"
            "因此，相比本页的强响应代表例，它是该 case/seed 内的低响应对照。"
        ),
        "evidence": "综合影响 Top/Bottom/Random = 0.586/0.590/0.577；最大值仅 0.590。",
        "caveat": "低像素效应不等于 attention dose 为零；应展开 Stage 4 dose 检查实际删除量。",
        "primary_metric": "impact",
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
    paths = [FAST_RANKING]
    if TRAJECTORY_ROOT.is_dir():
        paths.extend(TRAJECTORY_ROOT.rglob("report.json"))
        paths.extend(TRAJECTORY_ROOT.rglob("object_survival_report.json"))
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
        "pck10_error_percent": (
            100.0 * (1.0 - float(pck["0.1"])) if "0.1" in pck else None
        ),
        "track_loss_percent": metrics.get("target_worst_track_loss_score_0_100"),
        "overlay_path": row.get("overlay_path"),
    }


def _survival_summary(row: dict[str, Any], region: str | None) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    objects = metrics.get("objects") if isinstance(metrics.get("objects"), dict) else {}
    target = objects.get(region) if region and isinstance(objects.get(region), dict) else {}
    return {
        "quality_pass": metrics.get("quality_pass"),
        "disappearance_percent": metrics.get("target_worst_disappearance_score_0_100"),
        "mask_absence_percent": metrics.get("target_worst_mask_absence_score_0_100"),
        "identity_failure_percent": (
            100.0 * float(target["identity_failure_rate"])
            if "identity_failure_rate" in target
            else None
        ),
        "area_failure_percent": (
            100.0 * float(target["area_failure_rate"])
            if "area_failure_rate" in target
            else None
        ),
        "terminal_missing_percent": (
            100.0 * float(target["terminal_missing_rate"])
            if "terminal_missing_rate" in target
            else None
        ),
        "overlay_path": row.get("overlay_path"),
    }


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
    groups = []
    for definition in REPRESENTATIVES:
        rows = []
        for scope in SCOPE_ORDER:
            matched = next(
                (
                    row
                    for row in generated
                    if row.get("case") == definition["case"]
                    and int(row.get("seed", -1)) == definition["seed"]
                    and row.get("target_scope") == definition["target_scope"]
                    and row.get("region") == definition["region"]
                    and row.get("mask_mode") == definition["mask_mode"]
                    and row.get("head_scope") == scope
                ),
                None,
            )
            if matched is None:
                rows.append({"head_scope": scope, "label": SCOPE_LABELS[scope], "ready": False})
                continue
            variant = str(matched["variant_id"])
            record_key = (definition["case"], definition["seed"], variant)
            fast_row = fast.get(record_key, {})
            categories = fast_row.get("category_scores_0_100") or {}
            rows.append(
                {
                    "head_scope": scope,
                    "label": SCOPE_LABELS[scope],
                    "ready": True,
                    "variant_id": variant,
                    "fast": {
                        "impact": fast_row.get("impact_score_0_100"),
                        "global_appearance": categories.get("global_appearance"),
                        "target_local": categories.get("target_local"),
                        "temporal_appearance": categories.get("temporal_appearance"),
                        "outside_spillover": categories.get("outside_spillover"),
                    },
                    "trajectory": (
                        _trajectory_summary(trajectory[record_key], definition["region"])
                        if record_key in trajectory
                        else None
                    ),
                    "survival": (
                        _survival_summary(survival[record_key], definition["region"])
                        if record_key in survival
                        else None
                    ),
                }
            )
        group = dict(definition)
        group["rows"] = rows
        group["baseline_ready"] = (
            stage4_temporal_dashboard.asset(
                "baseline", definition["case"], definition["seed"]
            )
            is not None
        )
        groups.append(group)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "Stage 4 representative evidence",
        "groups": groups,
        "status": {
            "generated": stage4["progress"]["completed"],
            "expected": stage4["progress"]["expected"],
            "fast_records": int(fast_payload.get("sample_record_count") or len(fast)),
            "trajectory_records": len(trajectory),
            "survival_records": len(survival),
            "represented_cases": len({row["case"] for row in REPRESENTATIVES}),
            "unavailable_case": "0613pybullet_sample_000331_w001",
        },
        "definitions": fast_payload.get("category_definitions") or {},
        "ranking_definition": fast_payload.get("ranking_definition") or {},
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
<title>Stage 4 · 代表性证据</title><style>
:root{--night:#111a22;--panel:#182630;--panel2:#20333d;--ink:#edf5f4;--muted:#9cb0b3;--line:#36505a;--top:#56e0c4;--bottom:#ff9f62;--random:#be96ff;--warn:#ffd66b;--bad:#ff7777;--shadow:#071014}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(180deg,#0b141b,#13232b 48%,#0c171d);color:var(--ink);font-family:"Aptos","Noto Sans CJK SC","Trebuchet MS",sans-serif}a{color:var(--top)}a:focus-visible,button:focus-visible{outline:3px solid var(--warn);outline-offset:3px}header,main{width:min(1780px,calc(100% - 28px));margin:auto}header{padding:28px 0 16px}.nav{font:700 12px ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;gap:16px;flex-wrap:wrap}.eyebrow{margin-top:42px;color:var(--top);font:800 12px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.16em}.hero{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(300px,.7fr);gap:34px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:28px}h1,h2,h3{font-family:"Iowan Old Style","Noto Serif CJK SC",Georgia,serif}h1{font-size:clamp(47px,7vw,102px);letter-spacing:-.065em;line-height:.86;margin:12px 0 4px;max-width:1050px}.hero p{font-size:16px;line-height:1.7;color:var(--muted);margin:0}.status{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px;margin:20px 0}.stat{padding:13px 15px;background:var(--panel);border-top:3px solid var(--line)}.stat b{display:block;font:700 27px ui-monospace,SFMono-Regular,Menlo,monospace}.stat span{color:var(--muted);font-size:11px}.reading{margin:30px 0;padding:18px;background:var(--panel);border:1px solid var(--line)}.reading h2{margin:0 0 14px}.definition-grid{display:grid;grid-template-columns:repeat(5,minmax(170px,1fr));gap:8px}.definition{padding:12px;background:var(--night);border:1px solid var(--line)}.definition b{font-size:13px}.definition p{font-size:11px;color:var(--muted);line-height:1.55;margin:7px 0 0}.warning{margin-top:12px;padding:10px 13px;border-left:4px solid var(--warn);background:#ffd66b12;font-size:12px}.case{margin:34px 0 52px;scroll-margin-top:15px}.case-head{display:grid;grid-template-columns:180px minmax(260px,.8fr) minmax(390px,1.2fr);gap:20px;padding:20px 0;border-top:1px solid var(--line)}.case-label{font:800 11px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--warn);letter-spacing:.12em}.case h2{font-size:31px;line-height:1.05;margin:8px 0}.identity{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--muted);line-height:1.6;overflow-wrap:anywhere}.claim{font-size:15px;line-height:1.65;margin:0}.flow,.caveat{font-size:12px;line-height:1.65;color:var(--muted)}.evidence{padding:11px 13px;background:#56e0c410;border-left:4px solid var(--top);font:700 12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}.caveat{padding-left:13px;border-left:4px solid var(--warn)}.ruler{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:15px 0}.ruler-item{position:relative;min-height:66px;padding:11px;background:var(--panel);overflow:hidden}.ruler-item::after{content:"";position:absolute;left:0;bottom:0;height:5px;width:var(--width);background:var(--scope)}.ruler-item b{display:block;font:700 21px ui-monospace,SFMono-Regular,Menlo,monospace}.ruler-item span{font-size:10px;color:var(--muted)}.video-grid{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:9px}.video-card{background:var(--panel);border:1px solid var(--line);border-top:5px solid var(--scope,var(--line));padding:9px}.video-card.baseline{--scope:#80949a}.video-card[data-scope=top100],.ruler-item[data-scope=top100]{--scope:var(--top)}.video-card[data-scope=bottom100],.ruler-item[data-scope=bottom100]{--scope:var(--bottom)}.video-card[data-scope=random100_layer_matched_draw0],.ruler-item[data-scope=random100_layer_matched_draw0]{--scope:var(--random)}.video-card h3{font:800 13px ui-monospace,SFMono-Regular,Menlo,monospace;margin:2px 0 8px}.video-card video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#030708}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--line);margin-top:8px}.metric{padding:7px;background:var(--panel2)}.metric b{display:block;font:700 14px ui-monospace,SFMono-Regular,Menlo,monospace}.metric span{font-size:9px;color:var(--muted)}details{margin-top:8px;font-size:11px;color:var(--muted)}summary{cursor:pointer;color:var(--ink);font-weight:800}.detail-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin-top:7px}.detail-grid div{padding:7px;background:var(--night)}.pending{padding:16px;color:var(--muted);border:1px dashed var(--line)}.actions{display:flex;gap:10px;align-items:center;margin-top:10px}.actions button{border:1px solid var(--line);background:var(--panel2);color:var(--ink);padding:8px 11px;font-weight:800;cursor:pointer}.footer{border-top:1px solid var(--line);padding:22px 0 50px;color:var(--muted);font-size:11px}@media(max-width:1200px){.video-grid{grid-template-columns:repeat(2,1fr)}.definition-grid{grid-template-columns:repeat(3,1fr)}.case-head{grid-template-columns:140px 1fr}.case-analysis{grid-column:1/-1}}@media(max-width:700px){header,main{width:calc(100% - 14px)}.hero,.case-head{grid-template-columns:1fr}.status,.definition-grid,.ruler,.video-grid{grid-template-columns:1fr}h1{font-size:52px}.case-analysis{grid-column:auto}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body><header><div class="nav"><a href="/">8092 总入口</a><a href="/object-query-information-flow-stage4?v=1">Stage 4 全矩阵</a><a href="/object-query-information-flow-validation?v=2">Stage 1–3</a></div><div class="eyebrow">LATEST3350 / REPRESENTATIVE EVIDENCE REVIEW</div><div class="hero"><h1>先看反例，<br>再谈结论。</h1><p>这个页面只保留能区分假设的案例。每组固定同一个 case、seed、object 和消融算子，并排 Baseline、Top100、Bottom100、Random100；数值均相对同 seed Baseline，表示影响强度，不表示物理质量。</p></div><div id="status" class="status"></div></header><main><section class="reading"><h2>指标怎么读</h2><div id="definitions" class="definition-grid"></div><div class="warning">时序像素差异不是轨迹。只有 Center-ADE/FDE、velocity 与 PCK 可直接讨论轨迹；轨迹门控失败时必须同时查看 Track Loss 与 Disappearance。</div></section><div id="content"></div><div class="footer">视频按接近视口时加载。指标文件更新后刷新页面即可看到 trajectory / survival 补算结果。</div></main><script>
const api='/api/object-query-information-flow-stage4-representatives',videoApi='/api/object-query-information-flow-stage4/asset',$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const F=(v,d=3)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'—';
function media(kind,g,r={}){return `${videoApi}?${new URLSearchParams({kind,case:g.case,seed:String(g.seed),variant:r.variant_id||''})}`}
function lazy(root=document){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){const v=e.target;if(v.dataset.src){v.src=v.dataset.src;delete v.dataset.src;v.load()}io.unobserve(v)}}),{rootMargin:'500px'});root.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function fastMetrics(f){return [['Impact',f?.impact],['Target local',f?.target_local],['Temporal pixel',f?.temporal_appearance],['Outside',f?.outside_spillover]].map(([k,v])=>`<div class="metric"><b>${F(v)}</b><span>${k}</span></div>`).join('')}
function details(r){const t=r.trajectory,s=r.survival;if(!t&&!s)return '<div class="pending">Trajectory / survival 指标补算中</div>';let rows=[];if(t)rows.push(['Center-ADE / D0',t.center_ade_d0],['Center-FDE / D0',t.center_fde_d0],['Velocity / D0/frame',t.velocity_error_d0_per_frame],['100×(1−PCK@10%)',t.pck10_error_percent],['Track Loss %',t.track_loss_percent]);if(s)rows.push(['Disappearance %',s.disappearance_percent],['Mask Absence %',s.mask_absence_percent],['Identity Failure %',s.identity_failure_percent],['Area Failure %',s.area_failure_percent],['Terminal Missing %',s.terminal_missing_percent]);return `<details><summary>轨迹与对象存活指标</summary><div class="detail-grid">${rows.map(([k,v])=>`<div><b>${F(v)}</b><br>${k}</div>`).join('')}</div></details>`}
function videoCard(g,r){if(!r.ready)return `<article class="video-card" data-scope="${esc(r.head_scope)}"><h3>${esc(r.label)}</h3><div class="pending">尚未生成</div></article>`;return `<article class="video-card" data-scope="${esc(r.head_scope)}"><h3>${esc(r.label)}</h3><video controls muted loop playsinline preload="none" data-src="${esc(media('ablation',g,r))}"></video><div class="metrics">${fastMetrics(r.fast)}</div>${details(r)}</article>`}
function ruler(g){const vals=g.rows.map(r=>Number(r.fast?.[g.primary_metric])).filter(Number.isFinite),max=Math.max(...vals,1e-9);return g.rows.map(r=>{const v=Number(r.fast?.[g.primary_metric]),w=Number.isFinite(v)?Math.max(2,100*v/max):0;return `<div class="ruler-item" data-scope="${esc(r.head_scope)}" style="--width:${w}%"><b>${F(v)}</b><span>${esc(r.label)} · ${esc(g.primary_metric)}</span></div>`}).join('')}
function group(g,i){const full=`/object-query-information-flow-stage4?${new URLSearchParams({v:'1',case:g.case,seed:String(g.seed),target:`${g.target_scope}::${g.region||''}`})}`;return `<section class="case" id="${esc(g.id)}"><div class="case-head"><div><div class="case-label">${String(i+1).padStart(2,'0')} / ${esc(g.label)}</div><h2>${esc(g.title)}</h2><div class="identity">${esc(g.case)}<br>seed ${g.seed} · ${esc(g.region||'all_objects')}<br>${esc(g.mask_mode)}</div></div><div><p class="flow">${esc(g.flow)}</p><p class="claim"><b>当前结论（明确比较）：</b>${esc(g.claim)}</p><div class="actions"><button data-replay="${esc(g.id)}">同步重播</button><a href="${esc(full)}">打开完整矩阵</a></div></div><div class="case-analysis"><div class="evidence">${esc(g.evidence)}</div><p class="caveat"><b>证据边界：</b>${esc(g.caveat)}</p></div></div><div class="ruler">${ruler(g)}</div><div class="video-grid"><article class="video-card baseline"><h3>Baseline · no intervention</h3><video controls muted loop playsinline preload="none" data-src="${esc(media('baseline',g))}"></video><div class="pending">共同 reference；所有指标均与它比较</div></article>${g.rows.map(r=>videoCard(g,r)).join('')}</div></section>`}
async function load(){const d=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),s=d.status;$('status').innerHTML=[['生成',`${s.generated}/${s.expected}`],['Fast records',s.fast_records],['Trajectory records',s.trajectory_records],['Survival records',s.survival_records]].map(([k,v])=>`<div class="stat"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join('');const defs=[['Impact',d.ranking_definition?.direction],['Target local',d.definitions?.target_local?.direction],['Temporal pixel',d.definitions?.temporal_appearance?.direction],['Outside spillover',d.definitions?.outside_spillover?.direction],['Trajectory / survival','ADE/FDE 越大表示轨迹偏移更强；Disappearance 越大表示对象存活更差。']];$('definitions').innerHTML=defs.map(([k,v])=>`<div class="definition"><b>${esc(k)}</b><p>${esc(v)}</p></div>`).join('');$('content').innerHTML=d.groups.map(group).join('');document.querySelectorAll('[data-replay]').forEach(b=>b.onclick=()=>document.querySelectorAll(`#${CSS.escape(b.dataset.replay)} video`).forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));lazy()}
load().catch(e=>$('content').innerHTML=`<div class="pending">读取失败：${esc(e)}</div>`);
</script></body></html>'''
