#!/usr/bin/env python3
"""Dashboard data for the detector-gated object-query anti-duplication pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_anti_duplication/latest3350_v1"
)
GATE_CASE = "0613pybullet_sample_000331_w001"
PHYSIQ_CASE = (
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-"
    "ball-and-block-fall_motion_to_end"
)
PHYSIQ_DISK_CASE = PHYSIQ_CASE
GATE_BASELINE = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50/"
    "runs/0613pybullet_sample_000331_w001/seed_90094/generated.mp4"
)
PHYSIQ_BASE = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/runs"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def records(path: Path) -> dict[str, dict[str, Any]]:
    return {row["label"]: row for row in load_json(path).get("records", [])}


def metric_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "extra_frames": round(49 * float(row["extra_instance_frame_rate"])),
        "missing_frames": round(49 * float(row["missing_target_frame_rate"])),
        "center_ade_d0": row["baseline_matched_center_ade_over_d0"],
        "full_frame_mae": row["full_frame_mae"],
    }


def catalog() -> dict[str, Any]:
    gate_eval = ROOT / "evaluation" / GATE_CASE / "seed_90094"
    pareto = records(gate_eval / "rgbmask_pareto_audit_thr065" / "report.json")
    broad = records(gate_eval / "threshold_sensitivity_0p65" / "report.json")
    late = records(gate_eval / "rgbmask_late_audit_partial_thr065" / "report.json")
    guided = ROOT / "guided" / GATE_CASE / "seed_90094"
    gate_specs = [
        ("Baseline", GATE_BASELINE, pareto.get("Baseline"), "control"),
        (
            "Broad Q@K P95 · lambda=-1 · S00-09",
            guided / "secondary_branch__outgoing__top100_s039r3350__p95d1__pagm1__denoise_00_09/generated.mp4",
            broad.get("Outgoing_P95_L1"),
            "rejected-broad",
        ),
        (
            "RGB-confirmed 25 tokens · lambda=-0.25 · S00-09",
            guided / "secondary_branch__outgoing__top100_s039r3350__rgb_duplicate__pagm0p25__denoise_00_09/generated.mp4",
            pareto.get("RGB_L025_S00_09"),
            "targeted",
        ),
        (
            "RGB-confirmed 25 tokens · lambda=-0.50 · S00-09",
            guided / "secondary_branch__outgoing__top100_s039r3350__rgb_duplicate__pagm0p5__denoise_00_09/generated.mp4",
            pareto.get("RGB_L050_S00_09"),
            "targeted",
        ),
        (
            "RGB-confirmed 25 tokens · lambda=-1.00 · S00-09",
            guided / "secondary_branch__outgoing__top100_s039r3350__rgb_duplicate__pagm1__denoise_00_09/generated.mp4",
            pareto.get("RGB_L100_S00_09"),
            "targeted",
        ),
        (
            "RGB-confirmed 25 tokens · lambda=-0.50 · S10-19",
            guided / "secondary_branch__outgoing__top100_s039r3350__rgb_duplicate__pagm0p5__denoise_10_19/generated.mp4",
            late.get("RGB_L050_S10_19"),
            "targeted",
        ),
        (
            "RGB-confirmed 25 tokens · lambda=-0.50 · S20-29",
            guided / "secondary_branch__outgoing__top100_s039r3350__rgb_duplicate__pagm0p5__denoise_20_29/generated.mp4",
            late.get("RGB_L050_S20_29"),
            "targeted",
        ),
    ]
    assets: dict[str, str] = {}
    gate_rows = []
    for index, (label, path, metric, kind) in enumerate(gate_specs):
        key = f"gate-{index}"
        if path.is_file():
            assets[key] = str(path.resolve())
        gate_rows.append(
            {
                "label": label,
                "asset": key if key in assets else None,
                "kind": kind,
                "metrics": metric_payload(metric),
            }
        )

    physiq_rows = []
    for seed in (47326, 13248, 32466):
        eval_rows = records(
            ROOT / "evaluation" / PHYSIQ_DISK_CASE / f"seed_{seed}"
            / "brown_ball_duplicate_audit" / "report.json"
        )
        base = PHYSIQ_BASE / PHYSIQ_DISK_CASE / f"seed_{seed}" / "generated.mp4"
        broad_video = (
            ROOT / "guided" / PHYSIQ_DISK_CASE / f"seed_{seed}"
            / "secondary_branch__outgoing__top100_s039r3350__p95d1__pagm1__denoise_00_09/generated.mp4"
        )
        noop = (
            ROOT / "guided" / PHYSIQ_DISK_CASE / f"seed_{seed}"
            / "secondary_branch__outgoing__top100_s039r3350__rgb_duplicate__pagm0p25__denoise_00_09/generated.mp4"
        )
        items = []
        for suffix, label, path, metric, kind in (
            ("base", "Baseline", base, eval_rows.get("Baseline"), "control"),
            (
                "broad",
                "Broad Q@K P95（反例）",
                broad_video,
                eval_rows.get("Outgoing_P95_L1"),
                "rejected-broad",
            ),
            ("noop", "RGB gate · 0-token exact no-op", noop, eval_rows.get("Baseline"), "noop"),
        ):
            key = f"physiq-{seed}-{suffix}"
            if path.is_file():
                assets[key] = str(path.resolve())
            items.append(
                {
                    "label": label,
                    "asset": key if key in assets else None,
                    "kind": kind,
                    "metrics": metric_payload(metric),
                }
            )
        physiq_rows.append({"seed": seed, "items": items})
    return {
        "gate_case": GATE_CASE,
        "gate_seed": 90094,
        "gate": gate_rows,
        "physiq_case": PHYSIQ_CASE,
        "physiq": physiq_rows,
        "assets": assets,
        "report": str(ROOT / "REPORT.md"),
        "literature": str(ROOT / "LITERATURE_REVIEW.md"),
    }


def asset_path(key: str) -> Path | None:
    value = catalog()["assets"].get(key)
    path = Path(value) if value else None
    return path if path is not None and path.is_file() else None


def page() -> str:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object Query 去重探索</title><style>
:root{--bg:#ece9e1;--ink:#17201d;--card:#fffdf8;--line:#bbb4a7;--red:#bb3e32;--blue:#155f76;--green:#19745d;--muted:#68716d}*{box-sizing:border-box}body{margin:0;background:linear-gradient(120deg,#b4473020,transparent 32rem),var(--bg);color:var(--ink);font-family:Inter,"Noto Sans SC",system-ui,sans-serif}header{background:#17201d;color:white;padding:24px clamp(16px,4vw,64px)}header a{color:#9bd9d0}.eyebrow{font:12px ui-monospace,monospace;color:#ef9e79;letter-spacing:.14em}h1{font-size:clamp(35px,6vw,72px);line-height:.95;letter-spacing:-.05em;margin:12px 0}.lead{max-width:1100px;line-height:1.65;color:#d8e0dc}main{width:min(1900px,calc(100% - 24px));margin:auto;padding:20px 0 80px}.defs{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.def,.finding{background:var(--card);border:1px solid var(--line);padding:15px}.def h2,.finding h2{margin:0 0 8px;font-size:18px}.formula{font:12px/1.55 ui-monospace,monospace;background:#e3dfd5;padding:9px}.warn{border-left:6px solid var(--red);background:#fff2e9;padding:13px 16px;margin:14px 0;line-height:1.55}.section-title{margin:34px 0 10px;font-size:29px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);min-width:0}.card.targeted{border-top:5px solid var(--blue)}.card.rejected-broad{border-top:5px solid var(--red)}.card.noop{border-top:5px solid var(--green)}.card h3{font-size:14px;margin:0;padding:11px 12px;min-height:52px}.card video{display:block;width:100%;aspect-ratio:1280/704;background:#111;object-fit:contain}.pending{display:grid;place-items:center;aspect-ratio:1280/704;background:#ddd8cd;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line)}.metric{padding:9px;border-right:1px solid var(--line);font:11px ui-monospace,monospace}.metric:last-child{border:0}.metric b{display:block;font-size:16px;margin-bottom:4px}.seed{margin:18px 0}.seed h3{font:18px ui-monospace,monospace}.finding{margin-top:16px}.finding li{margin:8px 0;line-height:1.55}@media(max-width:900px){.defs{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}</style></head><body><header><a href="/">← 返回 8092 总入口</a><div class="eyebrow">DETECT → CONFIRM → TARGETED R(K/V)→F(QUERY)</div><h1>多实例不是一个峰值问题。</h1><p class="lead">先确认同类额外实例，再只改写额外区域接收的对象信息；干净 seed 严格 no-op。所有视频按需加载，不自动播放。</p></header><main><section class="defs"><article class="def"><h2>Q@K 只作候选定位</h2><div class="formula">S(k)=mean_h softmax(Q_R0 K_k^T / sqrt(d))</div><p>多峰不等于多个因果实例；它可能来自合法对象、部件或背景相关性。</p></article><article class="def"><h2>精确修改</h2><div class="formula">O_F^pert = O_F − sum_{k∈R} A[F,k]V[k]</div><p>只切断主对象 R 对已确认额外区域 F 的写入，R→R 保留。</p></article><article class="def"><h2>安全门控</h2><div class="formula">F=∅ ⇒ intervention_triggered=false</div><p>没有同类额外检测时直接复用 Baseline，不运行 guidance。</p></article></section><div class="warn"><b>当前判断：</b>目标 case 的额外球可以被清零，但已测设置仍带来 0.25–0.37 D0 的轨迹偏移；因此是“部分解决”，不是无损修复。</div><h2 class="section-title">0613pybullet_sample_000331_w001 · seed 90094</h2><div id="gate" class="grid"></div><h2 class="section-title">PhysicIQ 随机 seed 安全性验证</h2><div id="physiq"></div><section class="finding"><h2>如何读指标</h2><ul><li><b>Extra</b>：49 帧中同类实例数大于 1 的帧数，越小越好。</li><li><b>Missing</b>：检测不到主目标的帧数，越小越好。</li><li><b>ADE/D0</b>：候选与同 seed Baseline 的检测中心平均距离，以首帧对象尺寸归一化；越小越保真。</li><li><b>MAE</b>：全帧像素差，只作全局变化 sanity check。</li></ul></section></main><script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function card(x){const m=x.metrics;const metrics=m?`<div class="metrics"><div class="metric"><b>${m.extra_frames}/49</b>Extra</div><div class="metric"><b>${m.missing_frames}/49</b>Missing</div><div class="metric"><b>${Number(m.center_ade_d0).toFixed(3)}</b>ADE/D0</div><div class="metric"><b>${Number(m.full_frame_mae).toFixed(5)}</b>MAE</div></div>`:'';const media=x.asset?`<video controls muted playsinline preload="metadata" src="/api/object-query-anti-duplication/asset?key=${encodeURIComponent(x.asset)}"></video>`:'<div class="pending">Pending</div>';return `<article class="card ${esc(x.kind)}"><h3>${esc(x.label)}</h3>${media}${metrics}</article>`}
async function init(){const r=await fetch('/api/object-query-anti-duplication/catalog',{cache:'no-store'});const d=await r.json();document.querySelector('#gate').innerHTML=d.gate.map(card).join('');document.querySelector('#physiq').innerHTML=d.physiq.map(row=>`<section class="seed"><h3>seed ${row.seed}</h3><div class="grid">${row.items.map(card).join('')}</div></section>`).join('')}
init();</script></body></html>'''
