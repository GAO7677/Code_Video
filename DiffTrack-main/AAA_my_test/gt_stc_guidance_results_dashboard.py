#!/usr/bin/env python3
"""Read-only results dashboard for the frozen GT-STC validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)
DUAL_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/attention_audit_v3"
)
SEED = 47326
ATTENTION_STEPS = tuple(range(5, 41, 5))
MODES = ("region", "point", "combined")
MODE_LABELS = {
    "region": "Region · tube mass",
    "point": "Point · tracked correspondence",
    "combined": "Combined · region + point",
}
DUAL_BACKENDS = {
    "firstframe_ti2v": {
        "label": "Protocol A · First-frame TI2V",
        "flow": "observed R0 Query -> future R1:12* Keys",
        "description": "只消费首帧条件；R0 同 ID 对象点的响应对齐到 12 个未来 latent anchor。",
    },
    "context8_v2v": {
        "label": "Protocol B · 8-frame V2V",
        "flow": "observed R0:1 Queries -> future R2:12* Keys",
        "description": "真正消费 8 帧 context；R0/R1 同 ID 对象点的响应对齐到 11 个未来 latent anchor。",
    },
}
DUAL_GROUPS = {
    "baseline": "Baseline · no guidance",
    "top100": "Top100 · high-PCK",
    "bottom100": "Bottom100 · low-PCK",
    "random100": "Random100 · layer-matched",
}
DUAL_DIAGNOSTICS = {
    "gt_trajectory": {
        "label": "13-anchor GT / pseudo-GT point trajectory",
        "filename": "gt_13_anchor_trajectory.mp4",
        "note": "青色同 ID CoTracker 点；R0...R12 使用 frozen anchor_source_frames（49 帧 clip 通常是 F00,F04,...,F48；短 clip 使用已存的非均匀锚点）。",
    },
    "current_constraint": {
        "label": "修正版实际约束方向",
        "filename": "current_forward_constraint.mp4",
        "note": "Q(Rctx,p_ctx^i) → K(Rt,p_t^i)：attention 的 future Key 响应表示未来落点。",
    },
    "previous_constraint": {
        "label": "旧版 reverse retrieval（仅作对照）",
        "filename": "previous_reverse_constraint.mp4",
        "note": "旧版 Q(Rt,p_t^i) → K(Rctx,p_ctx^i) 已停止，不用于 forward-v2 重跑。",
    },
    "baseline_before": {
        "label": "约束前 Baseline 生成轨迹",
        "filename": "baseline_before_guidance_trajectory.mp4",
        "note": "青色为 source pseudo-GT；橙色为同 backend 无 guidance 输出的 CoTracker 点。",
    },
}


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _target_metric(path: Path, target: str) -> dict[str, Any] | None:
    for row in _json(path).get("metrics", []):
        if str(row.get("target")) == target:
            return row
    return None


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _dual_variant(
    backend: str, case: str, target: str, group: str
) -> dict[str, Any]:
    variant = "baseline" if group == "baseline" else f"{group}__{target}"
    directory = (
        DUAL_ROOT
        / backend
        / "generations"
        / case
        / f"seed_{SEED:05d}"
        / variant
    )
    video = directory / "generated.mp4"
    complete = _ready(directory / "complete.json") and _ready(video)
    metric = _target_metric(directory / "trajectory_metrics.json", target)
    attention_audits = []
    if group != "baseline":
        for step in ATTENTION_STEPS:
            step_directory = directory / "attention_audit" / f"step_{step:02d}"
            report = _json(step_directory / "metrics.json")
            attention_audits.append(
                {
                    "step": step,
                    "ready": _ready(step_directory / "complete.json")
                    and _ready(step_directory / "attention_comparison.mp4"),
                    "metrics_ready": bool(report),
                    "summary": report.get("summary", {}),
                }
            )
    return {
        "name": variant,
        "group": group,
        "label": DUAL_GROUPS[group],
        "complete": complete,
        "video_ready": complete,
        "metric_ready": metric is not None,
        "metric": metric,
        "attention_audits": attention_audits,
    }


def _dual_diagnostics(backend: str, case: str, target: str) -> list[dict[str, Any]]:
    directory = DUAL_ROOT / "diagnostics" / backend / case / target
    return [
        {
            "name": name,
            "label": definition["label"],
            "note": definition["note"],
            "ready": _ready(directory / definition["filename"]),
        }
        for name, definition in DUAL_DIAGNOSTICS.items()
    ]


def _dual_catalog() -> dict[str, Any]:
    """Return all planned slots, including explicit pending entries."""
    backend_rows: list[dict[str, Any]] = []
    case_targets: dict[str, list[str]] = {}
    total_slots = complete_slots = metric_slots = 0
    attention_total = attention_ready = 0
    for backend, definition in DUAL_BACKENDS.items():
        manifest = _json(DUAL_ROOT / backend / "task_manifest.json")
        planned = int(manifest.get("total_video_count", 0))
        complete = 0
        metrics = 0
        for row in manifest.get("cases", []):
            case = str(row.get("case", ""))
            targets = [str(value) for value in row.get("targets", [])]
            if case:
                case_targets.setdefault(case, [])
                case_targets[case].extend(
                    target for target in targets if target not in case_targets[case]
                )
            baseline = _dual_variant(backend, case, targets[0] if targets else "", "baseline")
            complete += int(baseline["complete"])
            metrics += int(baseline["metric_ready"])
            for target in targets:
                for group in ("top100", "bottom100", "random100"):
                    variant = _dual_variant(backend, case, target, group)
                    complete += int(variant["complete"])
                    metrics += int(variant["metric_ready"])
                    attention_total += len(variant["attention_audits"])
                    attention_ready += sum(
                        int(row["ready"]) for row in variant["attention_audits"]
                    )
        total_slots += planned
        complete_slots += complete
        metric_slots += metrics
        backend_rows.append(
            {
                "name": backend,
                **definition,
                "manifest_ready": bool(manifest),
                "planned": planned,
                "complete": complete,
                "metrics": metrics,
                "context_rgb_frames": int(
                    (manifest.get("backend") or {}).get("context_rgb_frames", 0)
                ),
                "context_latent_frames": int(
                    (manifest.get("backend") or {}).get("context_latent_frames", 0)
                ),
                "query_times": list(
                    (manifest.get("loss") or {}).get("context_query_times", [])
                ),
                "key_times": list(
                    (manifest.get("loss") or {}).get("future_key_times", [])
                ),
            }
        )
    cases: list[dict[str, Any]] = []
    for case, targets in case_targets.items():
        tube_manifest = _json(ROOT / "gt_tubes" / case / "manifest.json")
        target_rows = []
        for target in targets:
            protocols = []
            for backend in DUAL_BACKENDS:
                protocols.append(
                    {
                        "backend": backend,
                        "diagnostics": _dual_diagnostics(backend, case, target),
                        "variants": [
                            _dual_variant(backend, case, target, group)
                            for group in ("baseline", "top100", "bottom100", "random100")
                        ],
                    }
                )
            target_rows.append({"name": target, "protocols": protocols})
        cases.append(
            {
                "case": case,
                "source_video_ready": _ready(
                    Path(str(tube_manifest.get("source_video", "")))
                ),
                "targets": target_rows,
            }
        )
    return {
        "root": str(DUAL_ROOT),
        "seed": SEED,
        "case_count": len(cases),
        "target_count": sum(len(row["targets"]) for row in cases),
        "planned": total_slots,
        "complete": complete_slots,
        "metrics": metric_slots,
        "attention_audits_ready": attention_ready,
        "attention_audits_total": attention_total,
        "attention_steps": list(ATTENTION_STEPS),
        "backends": backend_rows,
        "cases": cases,
        "equal_budget_rms": 0.01,
        "definitions": [
            {
                "metric": "Point correspondence loss",
                "calculation": "固定 observed-context Query Q(Rctx,p_ctx^i)，对 Wan 全部 T×H×W Keys 做一次全局 softmax；target 为同 ID 点在所有可见 future latent 的 Gaussian 等权混合。",
                "direction": "修正版为 Q(Rctx,p_ctx^i) → K(Rt,p_t^i)，future attention 响应位置直接表示对象点未来落点。",
            },
            {
                "metric": "Equal latent budget",
                "calculation": "Top100 / Bottom100 / Random100 的 mutable future latent 更新 RMS 均固定为 0.01；context latent 更新严格为 0。",
                "direction": "三组用力大小相同，比较的是 head group 给出的梯度方向。",
            },
            {
                "metric": "PRE / POST global attention overlay",
                "calculation": "同一次 guided run、同一个 denoising step：PRE 用 x_s，POST 用归一化梯度更新后的 x_s'；两者均对 Wan 全部 13×H×W Keys 做原生全局 softmax，再按 selected heads 与可见 context point Queries 做 pair-weighted 平均。",
                "direction": "同一 latent 帧的 PRE/POST 共用 p99.5 色标；红色差分表示约束后响应增加，蓝色表示减少。frame mass 保留该 head 是否真正读取这个时刻的信息。",
            },
            {
                "metric": "Localized mass / peak distance / hit rate",
                "calculation": "Localized mass 为 GT 同 ID 点半径 2σ 内的全局 attention 概率；peak distance 为该帧 attention 峰到 GT 点的 token 距离；hit rate 为峰落在 2σ 内的比例。",
                "direction": "希望 Δlocalized mass > 0、Δpeak distance < 0、Δhit rate > 0；必须结合 loss 与最终轨迹判断。",
            },
            {
                "metric": "Post-guidance predicted x0",
                "calculation": "FlowMatch 当前步估计 x̂0 = x_s' − σ_s v_CFG(x_s')，随后仅用于 VAE 解码展示，不参与下一步更新。",
                "direction": "观察中间去噪状态是否逐步形成目标运动；它不是最终生成帧。",
            },
            {
                "metric": "GT Center-ADE / D0",
                "calculation": "生成对象 CoTracker 中心与 source GT/pseudo-GT 中心逐未来 anchor 的平均距离，除以首帧对象 bbox 对角线 D0。",
                "direction": "越小越接近 GT；需同时检查 Track Loss，避免对象消失被轨迹门控丢弃。",
            },
        ],
    }


def _variant(
    case: str, target: str, mode: str, guidance_lambda: float = 0.1
) -> dict[str, Any]:
    name = (
        "baseline"
        if mode == "baseline"
        else f"{mode}__{target}__lambda{_float_tag(guidance_lambda)}"
    )
    directory = ROOT / "generations" / case / f"seed_{SEED:05d}" / name
    video = directory / "generated.mp4"
    metric_path = directory / "trajectory_metrics.json"
    metric = _target_metric(metric_path, target)
    trajectory_overlay = (
        ROOT
        / "trajectory_overlays"
        / case
        / f"seed_{SEED:05d}"
        / f"{name}__{target}.mp4"
    )
    complete = (
        (directory / "complete.json").is_file()
        and video.is_file()
        and video.stat().st_size > 0
    )
    return {
        "name": name,
        "mode": mode,
        "lambda": None if mode == "baseline" else guidance_lambda,
        "label": (
            "Baseline"
            if mode == "baseline"
            else f"{MODE_LABELS[mode]} · λ{guidance_lambda:g}"
        ),
        "complete": complete,
        "video_ready": complete,
        "trajectory_overlay_ready": (
            trajectory_overlay.is_file() and trajectory_overlay.stat().st_size > 0
        ),
        "metric_ready": metric is not None,
        "metric": metric,
    }


def _representatives(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select auditable best/worst examples from the frozen eligible cohort."""
    rows: list[dict[str, Any]] = []
    for case_row in cases:
        for target_row in case_row["targets"]:
            for variant in target_row["variants"]:
                if (
                    variant["mode"] not in MODES
                    or variant["lambda"] != 0.1
                    or not variant["metric_ready"]
                ):
                    continue
                rows.append(
                    {
                        "case": case_row["case"],
                        "target": target_row["name"],
                        "mode": variant["mode"],
                        "delta_ade_d0": variant.get("delta_ade_d0"),
                        "delta_track_loss": variant.get("delta_track_loss"),
                        "quality_pass": bool((variant.get("metric") or {}).get("quality_pass")),
                    }
                )
    selected: list[dict[str, Any]] = []
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        gated = [
            row
            for row in mode_rows
            if row["quality_pass"] and row["delta_ade_d0"] is not None
        ]
        trackable = [row for row in mode_rows if row["delta_track_loss"] is not None]
        trajectory_choices = (
            [("唯一可评估轨迹", gated[0])]
            if len(gated) == 1
            else [
                (
                    "最大轨迹改善",
                    min(gated, key=lambda row: row["delta_ade_d0"]) if gated else None,
                ),
                (
                    "最大轨迹恶化",
                    max(gated, key=lambda row: row["delta_ade_d0"]) if gated else None,
                ),
            ]
        )
        choices = trajectory_choices + [
            (
                "最大可追踪性损失",
                max(trackable, key=lambda row: row["delta_track_loss"]) if trackable else None,
            )
        ]
        for category, row in choices:
            if row is not None:
                selected.append({"category": category, **row})
    return selected


def catalog() -> dict[str, Any]:
    screening_path = ROOT / "screening" / f"seed_{SEED:05d}" / "baseline_eligibility.json"
    screening = _json(screening_path)
    final_report = _json(
        ROOT / "final_analysis" / f"seed_{SEED:05d}" / "frozen_validation_report.json"
    )
    trigger_modes = {
        str(mode) for mode in final_report.get("trigger_modes", []) if mode in MODES
    }
    cases = []
    guided_complete = 0
    guided_metrics = 0
    sensitivity_complete = 0
    sensitivity_metrics = 0
    trajectory_overlays_ready = 0
    for job in screening.get("eligible_jobs", []):
        case = str(job["case"])
        tube_manifest = _json(ROOT / "gt_tubes" / case / "manifest.json")
        targets = []
        for target in job["targets"]:
            target = str(target)
            variants = [_variant(case, target, "baseline")]
            for mode in MODES:
                if mode in trigger_modes:
                    variants.append(_variant(case, target, mode, 0.05))
                variants.append(_variant(case, target, mode, 0.1))
                if mode in trigger_modes:
                    variants.append(_variant(case, target, mode, 0.2))
            primary = [row for row in variants[1:] if row["lambda"] == 0.1]
            guided_complete += sum(row["complete"] for row in primary)
            guided_metrics += sum(row["metric_ready"] for row in primary)
            sensitivity = [row for row in variants[1:] if row["lambda"] != 0.1]
            sensitivity_complete += sum(row["complete"] for row in sensitivity)
            sensitivity_metrics += sum(row["metric_ready"] for row in sensitivity)
            baseline_metric = variants[0]["metric"] or {}
            for variant in variants[1:]:
                metric = variant["metric"] or {}
                variant["delta_ade_d0"] = (
                    float(metric["ade_d0"]) - float(baseline_metric["ade_d0"])
                    if metric.get("ade_d0") is not None
                    and baseline_metric.get("ade_d0") is not None
                    else None
                )
                variant["delta_track_loss"] = (
                    float(metric["future_track_loss_score_0_100"])
                    - float(baseline_metric["future_track_loss_score_0_100"])
                    if metric.get("future_track_loss_score_0_100") is not None
                    and baseline_metric.get("future_track_loss_score_0_100") is not None
                    else None
                )
            source_trajectory = (
                ROOT / "trajectory_overlays" / case / f"source__{target}.mp4"
            )
            source_trajectory_ready = (
                source_trajectory.is_file() and source_trajectory.stat().st_size > 0
            )
            trajectory_overlays_ready += int(source_trajectory_ready)
            trajectory_overlays_ready += sum(
                int(row["trajectory_overlay_ready"]) for row in variants
            )
            targets.append(
                {
                    "name": target,
                    "source_trajectory_overlay_ready": source_trajectory_ready,
                    "variants": variants,
                }
            )
        cases.append(
            {
                "case": case,
                "source_video_ready": Path(str(tube_manifest.get("source_video", ""))).is_file(),
                "targets": targets,
            }
        )
    total = int(screening.get("eligible_target_count", 0)) * len(MODES)
    sensitivity_total = (
        int(screening.get("eligible_target_count", 0)) * len(trigger_modes) * 2
    )
    eligible_target_count = int(screening.get("eligible_target_count", 0))
    overlays_per_target = 2 + len(MODES) + 2 * len(trigger_modes)
    return {
        "protocol": "wan_gt_guidance_frozen_validation_v1",
        "dual_protocol": _dual_catalog(),
        "seed": SEED,
        "case_count": int(screening.get("case_count", 0)),
        "eligible_case_count": int(screening.get("eligible_case_count", 0)),
        "eligible_target_count": eligible_target_count,
        "missing_case_count": int(screening.get("missing_case_count", 0)),
        "guided_total": total,
        "guided_complete": guided_complete,
        "guided_metrics": guided_metrics,
        "sensitivity_total": sensitivity_total,
        "sensitivity_complete": sensitivity_complete,
        "sensitivity_metrics": sensitivity_metrics,
        "trajectory_overlays_ready": trajectory_overlays_ready,
        "trajectory_overlays_total": eligible_target_count * overlays_per_target,
        "cases": cases,
        "representatives": _representatives(cases),
        "final_report_ready": bool(final_report),
        "final_aggregate": final_report.get("aggregate", []),
        "trigger_modes": [mode for mode in MODES if mode in trigger_modes],
        "definitions": [
            {
                "metric": "Future ADE / D0",
                "calculation": "F04–F48 共同可见 CoTracker 点到 source GT tube 的逐帧平均距离 ÷ F00 bbox 对角线",
                "direction": "越小越接近 GT；Δ<0 表示优于同 seed Baseline",
            },
            {
                "metric": "Future FDE / D0",
                "calculation": "最后共同有效未来 anchor 的点距离 ÷ D0",
                "direction": "越小越接近 GT；只在轨迹门控通过时报告",
            },
            {
                "metric": "PCK@10% D0",
                "calculation": "共同可见未来点中误差 ≤ 0.1D0 的比例",
                "direction": "越大越接近 GT",
            },
            {
                "metric": "Future Track Loss",
                "calculation": "100 × (1 − 共同有效未来 anchor / source 有效未来 anchor)",
                "direction": "越小越好；消失/不可追踪也保留，不能被 ADE 均值丢掉",
            },
            {
                "metric": "Trajectory quality gate",
                "calculation": "未来共同 anchor ≥4 且覆盖率 ≥0.8；F00 condition frame 完全排除",
                "direction": "未通过时 ADE/FDE/PCK 记 N/A",
            },
            {
                "metric": "Trajectory overlay",
                "calculation": "每帧重新运行 CoTracker；青色绘制 source GT 对应点/质心/历史路径，橙色绘制生成视频对应点/质心/历史路径",
                "direction": "两条路径越重合越接近 GT；红色 TRACK LOST 表示当前帧不足 4 个对应点可见",
            },
        ],
    }


def asset(
    kind: str,
    case: str,
    target: str = "",
    variant: str = "",
    backend: str = "",
    step: int | str = "",
) -> Path | None:
    if kind in {
        "dual_source",
        "dual_generated",
        "dual_diagnostic",
        "dual_attention_audit",
    }:
        if backend not in DUAL_BACKENDS:
            return None
        manifest = _json(DUAL_ROOT / backend / "task_manifest.json")
        jobs = {
            str(row.get("case", "")): {
                str(value) for value in row.get("targets", [])
            }
            for row in manifest.get("cases", [])
        }
        if case not in jobs:
            return None
        if kind == "dual_source":
            source = Path(
                str(_json(ROOT / "gt_tubes" / case / "manifest.json").get("source_video", ""))
            )
            return source if _ready(source) else None
        if target not in jobs[case]:
            return None
        if kind == "dual_diagnostic":
            definition = DUAL_DIAGNOSTICS.get(variant)
            if definition is None:
                return None
            video = (
                DUAL_ROOT
                / "diagnostics"
                / backend
                / case
                / target
                / definition["filename"]
            )
            return video if _ready(video) else None
        allowed = {"baseline"} | {
            f"{group}__{target}" for group in ("top100", "bottom100", "random100")
        }
        if variant not in allowed:
            return None
        if kind == "dual_attention_audit":
            if variant == "baseline":
                return None
            try:
                capture_step = int(step)
            except (TypeError, ValueError):
                return None
            if capture_step not in ATTENTION_STEPS:
                return None
            video = (
                DUAL_ROOT
                / backend
                / "generations"
                / case
                / f"seed_{SEED:05d}"
                / variant
                / "attention_audit"
                / f"step_{capture_step:02d}"
                / "attention_comparison.mp4"
            )
            return video if _ready(video) else None
        video = (
            DUAL_ROOT
            / backend
            / "generations"
            / case
            / f"seed_{SEED:05d}"
            / variant
            / "generated.mp4"
        )
        return video if _ready(video) else None
    screening = _json(
        ROOT / "screening" / f"seed_{SEED:05d}" / "baseline_eligibility.json"
    )
    jobs = {
        str(row["case"]): {str(value) for value in row["targets"]}
        for row in screening.get("eligible_jobs", [])
    }
    if case not in jobs:
        return None
    if kind == "source":
        source = Path(str(_json(ROOT / "gt_tubes" / case / "manifest.json").get("source_video", "")))
        return source if source.is_file() else None
    if kind == "trajectory_source":
        if target not in jobs[case]:
            return None
        overlay = ROOT / "trajectory_overlays" / case / f"source__{target}.mp4"
        return overlay if overlay.is_file() else None
    if kind not in {"generated", "trajectory_generated"} or target not in jobs[case]:
        return None
    final_report = _json(
        ROOT / "final_analysis" / f"seed_{SEED:05d}" / "frozen_validation_report.json"
    )
    trigger_modes = {
        str(mode) for mode in final_report.get("trigger_modes", []) if mode in MODES
    }
    allowed = {"baseline"} | {f"{mode}__{target}__lambda0p1" for mode in MODES}
    allowed |= {
        f"{mode}__{target}__lambda{_float_tag(guidance_lambda)}"
        for mode in trigger_modes
        for guidance_lambda in (0.05, 0.2)
    }
    if variant not in allowed:
        return None
    video = (
        ROOT / "generations" / case / f"seed_{SEED:05d}" / variant / "generated.mp4"
        if kind == "generated"
        else ROOT
        / "trajectory_overlays"
        / case
        / f"seed_{SEED:05d}"
        / f"{variant}__{target}.mp4"
    )
    return video if video.is_file() else None


def page() -> str:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GT-STC Guidance Validation</title><style>
:root{--ink:#152238;--paper:#edf3f7;--panel:#f8fbfd;--line:#b7c7d5;--cobalt:#175c91;--cyan:#1d91a8;--amber:#d88a24;--red:#b64d50;--muted:#60748a;--shadow:0 14px 40px #17345018}*{box-sizing:border-box}body{margin:0;background:linear-gradient(90deg,#dbe7ee 1px,transparent 1px),linear-gradient(#dbe7ee 1px,transparent 1px),var(--paper);background-size:28px 28px;color:var(--ink);font:15px/1.55 "Avenir Next","Segoe UI",sans-serif}header{padding:34px clamp(20px,5vw,72px) 28px;background:#eef5f9eF;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}a{color:var(--cobalt)}.eyebrow,.mono{font:700 11px/1.3 ui-monospace,SFMono-Regular,monospace;letter-spacing:.12em;text-transform:uppercase}.eyebrow{color:var(--cyan);margin-top:18px}h1{max-width:1100px;margin:8px 0 10px;font:700 clamp(34px,6vw,76px)/.94 "Arial Narrow","Avenir Next Condensed",sans-serif;letter-spacing:-.045em}.lead{max-width:970px;color:#3c536b;font-size:17px}.anchor-strip{display:grid;grid-template-columns:repeat(13,1fr);max-width:780px;margin-top:24px;border:1px solid var(--line);background:var(--panel)}.anchor-strip i{height:13px;border-right:1px solid var(--line);background:linear-gradient(90deg,var(--cobalt),var(--cyan));opacity:calc(.25 + var(--n)*.055)}.anchor-strip i:last-child{border:0}main{padding:26px clamp(16px,4vw,64px) 80px;max-width:1900px;margin:auto}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.stat,.section{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow)}.stat{padding:18px}.stat b{display:block;font:700 30px/1 "Arial Narrow",sans-serif;margin-top:7px}.section{margin-top:18px;padding:20px}.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:end;margin:16px 0}.toolbar label{font-weight:700}.toolbar select,.toolbar button{display:block;margin-top:5px;padding:9px 12px;border:1px solid #8da5b7;background:#fff;color:var(--ink)}.toolbar button{cursor:pointer;background:var(--cobalt);color:#fff}.track-legend{display:flex;flex-wrap:wrap;gap:18px;margin:-4px 0 14px;padding:10px 12px;border-left:4px solid var(--cyan);background:#eaf3f7;color:#3d5368}.swatch{display:inline-block;width:22px;height:4px;margin:0 7px 3px 0;border-radius:4px}.swatch.gt{background:#29e2ee}.swatch.candidate{background:#ffa73d}.swatch.lost{background:#f34a58}.definitions{overflow:auto}table{width:100%;border-collapse:collapse;min-width:800px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d6e1e8;vertical-align:top}th{font:700 11px ui-monospace,monospace;text-transform:uppercase;color:var(--muted)}.case-title{display:flex;justify-content:space-between;gap:16px;align-items:center}.case-title h2{margin:0;font:700 25px "Arial Narrow",sans-serif}.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:11px;overflow-x:auto;padding-bottom:8px}.card{min-width:220px;border:1px solid var(--line);background:#fff}.card video,.empty{width:100%;aspect-ratio:16/9;background:#102033;display:block}.empty{display:grid;place-items:center;color:#b9c8d4;font:700 12px ui-monospace,monospace;text-align:center;padding:20px}.caption{padding:12px}.caption b{display:block}.view-tag{display:block;margin-top:3px;color:var(--muted);font:700 10px ui-monospace,monospace;text-transform:uppercase}.bad{color:var(--red)}.good{color:#16785f}.pending{color:var(--amber)}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:5px;margin-top:9px;font:12px ui-monospace,monospace;color:#40576d}.aggregate{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}.agg{padding:14px;border-left:5px solid var(--cyan);background:#eef6f8}.agg b{display:block}.jump{padding:6px 9px;border:1px solid var(--cobalt);background:#fff;color:var(--cobalt);cursor:pointer}.footer{color:var(--muted);margin-top:28px}.dual-board{border-top:7px solid var(--cobalt)}.dual-intro{display:grid;grid-template-columns:minmax(240px,1.2fr) minmax(280px,2fr);gap:18px;align-items:start}.dual-intro h2{margin:2px 0 8px;font:700 clamp(27px,4vw,46px)/1 "Arial Narrow",sans-serif}.protocol-row{margin-top:16px;border:1px solid var(--line);background:#eef5f8}.protocol-head{display:grid;grid-template-columns:minmax(240px,1fr) minmax(280px,2fr);gap:16px;padding:14px 16px;border-bottom:1px solid var(--line);background:#e4eef3}.protocol-head h3{margin:0;font:700 22px "Arial Narrow",sans-serif}.protocol-head p{margin:4px 0 0;color:var(--muted)}.flow-rail{display:grid;grid-template-columns:repeat(13,1fr);gap:3px;align-self:center}.flow-rail i{height:22px;border:1px solid #a8bdca;background:#c9d7df;position:relative}.flow-rail i.context{background:var(--cobalt);border-color:var(--cobalt)}.flow-rail i.future{background:linear-gradient(135deg,#d4eef1,var(--cyan));border-color:#69aeba}.flow-rail i::after{content:attr(data-t);position:absolute;inset:0;display:grid;place-items:center;color:#fff;font:700 8px ui-monospace,monospace}.dual-grid{display:grid;grid-template-columns:repeat(4,minmax(225px,1fr));gap:11px;padding:12px;overflow-x:auto}.source-grid{display:grid;grid-template-columns:minmax(240px,380px) 1fr;gap:16px;align-items:start;margin-top:14px}.source-note{padding:14px 16px;border-left:5px solid var(--amber);background:#fff5e7}.source-note p{margin:5px 0;color:#53687a}.pending-slot{background:repeating-linear-gradient(135deg,#13263a,#13263a 12px,#193149 12px,#193149 24px);grid-template-rows:auto auto;align-content:center;gap:7px}.pending-slot strong{color:#ffd18c;letter-spacing:.16em}.pending-slot span{font-weight:500;color:#aebfcc}.dual-progress{height:8px;background:#d7e2e8;margin-top:8px;overflow:hidden}.dual-progress i{display:block;height:100%;background:linear-gradient(90deg,var(--cobalt),var(--cyan))}.constraint-audit{margin-top:18px;padding-top:16px;border-top:2px dashed #8aa6b8}.constraint-audit>h3{font:700 26px "Arial Narrow",sans-serif;margin:0 0 5px}.direction-warning{padding:14px 16px;border-left:6px solid var(--red);background:#fff0ef;color:#6c3034}.direction-warning b{display:block;margin-bottom:4px}.audit-row .protocol-head{background:#f4e8e7}.audit-note{display:block;margin-top:8px;color:#516879;font-size:12px}.audit-grid{display:grid;grid-template-columns:repeat(4,minmax(245px,1fr));gap:11px;padding:12px;overflow-x:auto}.attention-microscope{margin-top:22px;padding:18px;background:#10253a;color:#eaf7fb;border:1px solid #163e5c;box-shadow:0 18px 44px #10253a32}.attention-head{display:grid;grid-template-columns:minmax(280px,1fr) minmax(420px,1.5fr);gap:18px;align-items:end}.attention-head h3{margin:0;font:700 clamp(26px,3vw,42px)/1 "Arial Narrow",sans-serif}.attention-head p{margin:7px 0 0;color:#a9c4d3}.step-rail{display:grid;grid-template-columns:repeat(8,1fr);gap:5px}.step-rail button{border:1px solid #47738e;background:#17344c;color:#b9d2df;padding:10px 4px;font:700 11px ui-monospace,monospace;cursor:pointer}.step-rail button.active{background:var(--amber);border-color:#ffd08b;color:#17263a;box-shadow:0 0 0 2px #ffd08b33}.attention-protocol{margin-top:15px;border-top:1px solid #31526a;padding-top:12px}.attention-protocol h4{margin:0 0 8px;font:700 19px "Arial Narrow",sans-serif}.attention-grid{display:grid;grid-template-columns:repeat(3,minmax(410px,1fr));gap:10px;overflow-x:auto}.attention-card{background:#f8fbfd;color:var(--ink)}.attention-card video,.attention-card .empty{aspect-ratio:6.7/1}.attention-card .metrics{grid-template-columns:repeat(4,1fr)}.attention-legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;color:#bdd2dd;font:12px ui-monospace,monospace}.attention-legend b{color:#fff}@media(max-width:900px){.dual-intro,.protocol-head,.source-grid,.attention-head{grid-template-columns:1fr}.dual-grid,.audit-grid{grid-template-columns:repeat(4,78vw)}.attention-grid{grid-template-columns:repeat(3,86vw)}}@media(max-width:760px){h1{font-size:43px}.section{padding:13px}.grid{grid-template-columns:repeat(5,82vw)}.step-rail{grid-template-columns:repeat(4,1fr)}}@media(prefers-reduced-motion:no-preference){.stat,.section{animation:up .35s ease both}@keyframes up{from{opacity:0;transform:translateY(8px)}}}</style></head><body>
<header><a href="/">← 返回 8092 总入口</a> · <a href="/gt-stc-guidance-preflight?v=2">Tube 预检</a><div class="eyebrow">Frozen latest3350 · source-oracle intervention</div><h1>同一条 GT 轨迹，<br>哪组 heads 拉得更准？</h1><p class="lead">First-frame TI2V 与 8-frame V2V 使用相同 latent RMS 预算；除最终视频外，页面逐步展示 step 5/10/…/40 的原始 PRE attention、约束后 POST attention、差分和 predicted-x0。未落盘项保留 Pending。</p><div class="anchor-strip" aria-label="13 latent anchors"><i style="--n:0"></i><i style="--n:1"></i><i style="--n:2"></i><i style="--n:3"></i><i style="--n:4"></i><i style="--n:5"></i><i style="--n:6"></i><i style="--n:7"></i><i style="--n:8"></i><i style="--n:9"></i><i style="--n:10"></i><i style="--n:11"></i><i style="--n:12"></i></div></header>
<main><section class="section dual-board" id="equalBudget"><div class="dual-intro"><div><span class="eyebrow">Live GPU1 matrix · equal-budget head direction</span><h2>双协议 Head Guidance</h2><p>所有 planned slot 始终占位；页面每 30 秒读取落盘状态，不等待整批跑完。</p></div><div id="dualSummary" class="summary"></div></div><div class="toolbar"><label>Case<select id="dualCase"></select></label><label>Target<select id="dualTarget"></select></label><button id="dualRefresh">刷新现场</button><button id="dualReplay">双协议同步重播</button><span id="dualUpdated" class="mono">读取中</span></div><div class="definitions"><table><thead><tr><th>约束 / 指标</th><th>精确计算</th><th>判读</th></tr></thead><tbody id="dualDefs"></tbody></table></div><div id="dualGallery"></div></section><div id="summary" class="summary" style="margin-top:18px"></div><section class="section"><h2>原冻结验证 · 计算与判读</h2><div class="definitions"><table><thead><tr><th>指标</th><th>精确计算</th><th>方向</th></tr></thead><tbody id="defs"></tbody></table></div></section><section class="section" id="paired"><div class="case-title"><h2>原冻结配对结果 · Region / Point / Combined</h2><span id="updated" class="mono">读取中</span></div><div class="toolbar"><label>Case<select id="case"></select></label><label>Target<select id="target"></select></label><label>画面<select id="view"><option value="trajectory">对象轨迹叠加</option><option value="raw">原始视频</option></select></label><button id="refresh">刷新现场</button><button id="replay">同步重播</button></div><div id="trackLegend" class="track-legend"><span><i class="swatch gt"></i>青色：Source GT 对应点、质心与历史路径</span><span><i class="swatch candidate"></i>橙色：生成视频 CoTracker 结果</span><span><i class="swatch lost"></i>红色：当前帧 TRACK LOST</span></div><div id="gallery"></div></section><section class="section"><h2>Case-balanced 汇总</h2><div id="aggregate" class="aggregate"></div></section><section class="section"><h2>代表性样本</h2><p>仅在冻结 eligible cohort 内选择；ADE 排序要求 guided trajectory gate 通过，Track Loss 排序保留破坏性失败。</p><div class="definitions"><table><thead><tr><th>Mode</th><th>选择理由</th><th>Case / Target</th><th>ΔADE/D0</th><th>ΔTrack Loss</th><th>查看</th></tr></thead><tbody id="representatives"></tbody></table></div></section><p class="footer">视频使用懒加载；未生成项保留明确 Pending 卡位，刷新无需改变当前 case/target。</p></main>
<script>
const api='/api/gt-stc-guidance-results',E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),F=(v,d=3)=>v==null?'N/A':Number(v).toFixed(d);let D;const $=id=>document.getElementById(id);
function lazy(){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting&&e.target.dataset.src){e.target.src=e.target.dataset.src;delete e.target.dataset.src;e.target.load();io.unobserve(e.target)}}),{rootMargin:'500px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function metric(v){const m=v.metric||{},gate=m.quality_pass===true;return `<div class="metrics"><span class="${gate?'good':'bad'}">Gate ${gate?'PASS':'FAIL/N.A.'}</span><span>ADE/D0 ${F(m.ade_d0)}</span><span>FDE/D0 ${F(m.fde_d0)}</span><span>PCK10 ${F(m.pck_10pct_d0)}</span><span>TrackLoss ${F(m.future_track_loss_score_0_100,1)}</span>${v.mode==='baseline'?'':`<span>ΔADE ${F(v.delta_ade_d0)}</span><span>ΔLoss ${F(v.delta_track_loss,1)}</span>`}</div>`}
function video(src,label,v,view){const waiting=view==='trajectory'?'轨迹叠加生成中':'视频未生成';return `<article class="card">${src?`<video controls muted playsinline preload="none" data-src="${src}"></video>`:`<div class="empty">${waiting}</div>`}<div class="caption"><b>${E(label)}</b><span class="view-tag">${view==='trajectory'?'GT / CoTracker trajectory overlay':'raw video'}</span>${v?metric(v):'<div class="metrics"><span>Source GT tube</span></div>'}</div></article>`}
function dualMetric(v){const m=v.metric||{};if(!v.metric_ready)return `<div class="metrics"><span class="pending">指标 Pending</span><span>${v.complete?'等待 CoTracker':'等待视频'}</span></div>`;return `<div class="metrics"><span class="${m.quality_pass?'good':'bad'}">Gate ${m.quality_pass?'PASS':'FAIL'}</span><span>ADE/D0 ${F(m.ade_d0)}</span><span>FDE/D0 ${F(m.fde_d0)}</span><span>PCK10 ${F(m.pck_10pct_d0)}</span><span>TrackLoss ${F(m.future_track_loss_score_0_100,1)}</span></div>`}
function dualCard(src,v,backend,caseName,targetName){return `<article class="card">${src?`<video controls muted playsinline preload="none" data-src="${src}"></video>`:`<div class="empty pending-slot"><strong>PENDING</strong><span>GPU1 尚未生成该 slot</span></div>`}<div class="caption"><b>${E(v.label)}</b><span class="view-tag">${E(backend)} · ${E(targetName)}</span>${dualMetric(v)}</div></article>`}
function diagnosticCard(src,v,backend){return `<article class="card">${src?`<video controls muted playsinline loop preload="none" data-src="${src}"></video>`:`<div class="empty pending-slot"><strong>PENDING</strong><span>${v.name==='baseline_before'?'同 backend Baseline 尚未生成':'诊断视频尚未渲染'}</span></div>`}<div class="caption"><b>${E(v.label)}</b><span class="view-tag">${E(backend)} · 13 latent anchors</span><span class="audit-note">${E(v.note)}</span></div></article>`}
function rail(contextCount){return `<div class="flow-rail" aria-label="13 latent anchors">${Array.from({length:13},(_,i)=>`<i class="${i<contextCount?'context':'future'}" data-t="${i}"></i>`).join('')}</div>`}
let attentionStep=5;
function attentionMetrics(a){const s=a.summary||{};if(!a.metrics_ready)return `<div class="metrics"><span class="pending">统计 Pending</span><span>等待 step ${a.step}</span></div>`;const lossGood=s.loss_change!=null&&s.loss_change<0,localGood=s.delta_future_localized_mass!=null&&s.delta_future_localized_mass>0,peakGood=s.delta_future_peak_distance_tokens!=null&&s.delta_future_peak_distance_tokens<0,hitGood=s.delta_future_peak_hit_rate_2sigma!=null&&s.delta_future_peak_hit_rate_2sigma>0;return `<div class="metrics"><span class="${lossGood?'good':'bad'}">L ${F(s.pre_loss,4)}→${F(s.post_loss,4)}</span><span class="${localGood?'good':'bad'}">Δlocal ${F(s.delta_future_localized_mass,5)}</span><span class="${peakGood?'good':'bad'}">Δpeak d ${F(s.delta_future_peak_distance_tokens,3)}</span><span class="${hitGood?'good':'bad'}">Δhit ${F(s.delta_future_peak_hit_rate_2sigma,3)}</span></div>`}
function attentionCard(src,v,a,backend){return `<article class="card attention-card">${src?`<video controls muted playsinline loop preload="none" data-src="${src}"></video>`:`<div class="empty pending-slot"><strong>PENDING · STEP ${a.step}</strong><span>PRE / POST attention audit 尚未落盘</span></div>`}<div class="caption"><b>${E(v.label)}</b><span class="view-tag">${E(backend)} · Source | PRE | POST | Δ | predicted-x0</span>${attentionMetrics(a)}</div></article>`}
function attentionMicroscope(t,d,c,q){const steps=d.attention_steps||[5,10,15,20,25,30,35,40];if(!steps.includes(attentionStep))attentionStep=steps[0];let html=`<section class="attention-microscope"><div class="attention-head"><div><span class="eyebrow">Denoising attention microscope</span><h3>同一 step，约束前后究竟改了哪里？</h3><p>每个视频播放 13 个 latent anchor；PRE/POST 在同一源帧上共用色标，差分红增蓝减，第五栏是 post-guidance predicted-x0。</p></div><div class="step-rail">${steps.map(s=>`<button class="${s===attentionStep?'active':''}" data-attention-step="${s}">STEP ${s}</button>`).join('')}</div></div><div class="attention-legend"><span><b>frame mass</b>：全局 softmax 真正分配给该时刻的概率</span><span><b>local</b>：GT 点 2σ 邻域内概率</span><span><b>peak d</b>：峰值到 GT 点距离</span></div>`;for(const p of t.protocols){const meta=d.backends.find(x=>x.name===p.backend)||{};const cards=p.variants.filter(v=>v.group!=='baseline').map(v=>{const a=(v.attention_audits||[]).find(x=>x.step===attentionStep)||{step:attentionStep,ready:false,metrics_ready:false,summary:{}};const src=a.ready?`${api}/asset?kind=dual_attention_audit&backend=${q(p.backend)}&case=${q(c.case)}&target=${q(t.name)}&variant=${q(v.name)}&step=${attentionStep}`:'';return attentionCard(src,v,a,p.backend)}).join('');html+=`<div class="attention-protocol"><h4>${E(meta.label)} · step ${attentionStep}</h4><div class="attention-grid">${cards}</div></div>`}return html+'</section>'}
function bindAttentionSteps(){document.querySelectorAll('[data-attention-step]').forEach(button=>button.onclick=()=>{attentionStep=Number(button.dataset.attentionStep);renderDual()})}
function renderDual(){const d=D.dual_protocol,c=d.cases.find(x=>x.case===$('dualCase').value)||d.cases[0];if(!c){$('dualGallery').innerHTML='<p class="pending">双协议 task manifest 尚未建立。</p>';return}if($('dualTarget').dataset.case!==c.case){$('dualTarget').innerHTML=c.targets.map(t=>`<option>${E(t.name)}</option>`).join('');$('dualTarget').dataset.case=c.case}const t=c.targets.find(x=>x.name===$('dualTarget').value)||c.targets[0],q=x=>encodeURIComponent(x);const source=`${api}/asset?kind=dual_source&backend=firstframe_ti2v&case=${q(c.case)}&target=${q(t.name)}`;let html=`<div class="case-title"><h2>${E(c.case)} · ${E(t.name)}</h2><span class="mono">seed ${d.seed}</span></div><div class="source-grid"><article class="card">${c.source_video_ready?`<video controls muted playsinline preload="none" data-src="${source}"></video>`:'<div class="empty pending-slot"><strong>PENDING</strong><span>Source video 不可用</span></div>'}<div class="caption"><b>Source GT / pseudo-GT</b><span class="view-tag">CoTracker point trajectory 的来源</span></div></article><div class="source-note"><b>比较原则</b><p>Top100 / Bottom100 / Random100 每个 guided step 的 future latent 更新 RMS 都是 ${F(d.equal_budget_rms,2)}，context latent 更新为 0；因此主要比较梯度方向，而不是扰动强度。</p><p class="mono">${d.complete}/${d.planned} videos · ${d.metrics}/${d.planned} metrics ready · ${d.attention_audits_ready}/${d.attention_audits_total} attention steps</p></div></div><div class="constraint-audit"><h3>13 个 latent 时刻 · 约束几何审计</h3><div class="direction-warning"><b>方向已修正：当前 attention-audit-v3 从 observed context 查询未来落点。</b>实际 loss 为 Q(Rctx,p_ctx^i) → K(Rt,p_t^i)，对全部 future point-time Gaussian 做等权目标；旧版 Q(Rt,p_t^i) → K(Rctx,p_ctx^i) 仅保留为粉色反向对照。下方 microscope 的 PRE 是同一次 guided run 在 latent 更新前的原始 attention。</div>`;for(const p of t.protocols){const meta=d.backends.find(x=>x.name===p.backend)||{},cards=p.diagnostics.map(v=>{const src=v.ready?`${api}/asset?kind=dual_diagnostic&backend=${q(p.backend)}&case=${q(c.case)}&target=${q(t.name)}&variant=${q(v.name)}`:'';return diagnosticCard(src,v,p.backend)}).join('');html+=`<section class="protocol-row audit-row"><div class="protocol-head"><div><h3>${E(meta.label)} · Geometry</h3><p>${E(meta.flow)}</p></div>${rail(meta.context_latent_frames||0)}</div><div class="audit-grid">${cards}</div></section>`}html+='</div>'+attentionMicroscope(t,d,c,q);for(const p of t.protocols){const meta=d.backends.find(x=>x.name===p.backend)||{},cards=p.variants.map(v=>{const src=v.video_ready?`${api}/asset?kind=dual_generated&backend=${q(p.backend)}&case=${q(c.case)}&target=${q(t.name)}&variant=${q(v.name)}`:'';return dualCard(src,v,p.backend,c.case,t.name)}).join('');html+=`<section class="protocol-row"><div class="protocol-head"><div><h3>${E(meta.label)}</h3><p>${E(meta.flow)} · ${E(meta.description)}</p></div>${rail(meta.context_latent_frames||0)}</div><div class="dual-grid">${cards}</div></section>`}$('dualGallery').innerHTML=html;bindAttentionSteps();lazy()}
function dualSummary(){const d=D.dual_protocol,p=d.planned?Math.round(100*d.complete/d.planned):0,a=d.attention_audits_total?Math.round(100*d.attention_audits_ready/d.attention_audits_total):0;$('dualSummary').innerHTML=`<div class="stat"><span class="mono">Matrix</span><b>${d.case_count} / ${d.target_count}</b><small>cases / targets · seed ${d.seed}</small></div><div class="stat"><span class="mono">Generated</span><b>${d.complete}/${d.planned}</b><small>${p}% 已落盘<div class="dual-progress"><i style="width:${p}%"></i></div></small></div><div class="stat"><span class="mono">Attention steps</span><b>${d.attention_audits_ready}/${d.attention_audits_total}</b><small>${a}% PRE/POST overlays<div class="dual-progress"><i style="width:${a}%"></i></div></small></div><div class="stat"><span class="mono">Metrics</span><b>${d.metrics}/${d.planned}</b><small>CoTracker vs source GT</small></div>`;$('dualDefs').innerHTML=d.definitions.map(x=>`<tr><td><b>${E(x.metric)}</b></td><td>${E(x.calculation)}</td><td>${E(x.direction)}</td></tr>`).join('')}
function render(){const c=D.cases.find(x=>x.case===$('case').value)||D.cases[0];if(!c){$('gallery').innerHTML='<p>Baseline screen 尚未完成。</p>';return}const opts=c.targets.map(t=>`<option>${E(t.name)}</option>`).join('');if($('target').dataset.case!==c.case){$('target').innerHTML=opts;$('target').dataset.case=c.case}const t=c.targets.find(x=>x.name===$('target').value)||c.targets[0],q=x=>encodeURIComponent(x),view=$('view').value;$('trackLegend').hidden=view!=='trajectory';const sourceKind=view==='trajectory'?'trajectory_source':'source',sourceReady=view==='trajectory'?t.source_trajectory_overlay_ready:c.source_video_ready;let cards=video(sourceReady?`${api}/asset?kind=${sourceKind}&case=${q(c.case)}&target=${q(t.name)}`:'','Source GT',null,view);for(const v of t.variants){const ready=view==='trajectory'?v.trajectory_overlay_ready:v.video_ready,kind=view==='trajectory'?'trajectory_generated':'generated',src=ready?`${api}/asset?kind=${kind}&case=${q(c.case)}&target=${q(t.name)}&variant=${q(v.name)}`:'';cards+=video(src,v.label,v,view)}$('gallery').innerHTML=`<div class="case-title"><h2>${E(c.case)} · ${E(t.name)}</h2><span class="mono">seed ${D.seed}</span></div><div class="grid">${cards}</div>`;lazy()}
function reps(){$('representatives').innerHTML=(D.representatives||[]).map(x=>`<tr><td><b>${E(x.mode)}</b></td><td>${E(x.category)}</td><td>${E(x.case)}<br><span class="mono">${E(x.target)}</span></td><td>${F(x.delta_ade_d0)}</td><td>${F(x.delta_track_loss,1)}</td><td><button class="jump" data-case="${E(x.case)}" data-target="${E(x.target)}">跳转</button></td></tr>`).join('')||'<tr><td colspan="6" class="pending">轨迹指标完成后自动生成。</td></tr>';document.querySelectorAll('.jump').forEach(b=>b.onclick=()=>{$('case').value=b.dataset.case;$('target').dataset.case='';render();$('target').value=b.dataset.target;render();$('paired').scrollIntoView({behavior:'smooth'})})}
function summary(){const p=D.guided_total?Math.round(100*D.guided_complete/D.guided_total):0,s=D.sensitivity_total?`${D.sensitivity_complete}/${D.sensitivity_total}`:'未触发';$('summary').innerHTML=`<div class="stat"><span class="mono">Source audit</span><b>${D.case_count}/20</b><small>完成 tube 的 case</small></div><div class="stat"><span class="mono">Frozen screen</span><b>${D.eligible_target_count}</b><small>${D.eligible_case_count} cases 的 eligible targets</small></div><div class="stat"><span class="mono">Primary λ0.1</span><b>${D.guided_complete}/${D.guided_total}</b><small>${p}% · Region/Point/Combined</small></div><div class="stat"><span class="mono">Primary metrics</span><b>${D.guided_metrics}/${D.guided_total}</b><small>CoTracker future-only</small></div><div class="stat"><span class="mono">Trajectory overlays</span><b>${D.trajectory_overlays_ready}/${D.trajectory_overlays_total}</b><small>GT 青色 / generated 橙色</small></div><div class="stat"><span class="mono">Conditional sensitivity</span><b>${s}</b><small>${D.trigger_modes.length?E(D.trigger_modes.join(' + '))+' · λ0.05/0.2':'冻结触发尚未满足/判定'}</small></div>`;$('defs').innerHTML=D.definitions.map(x=>`<tr><td><b>${E(x.metric)}</b></td><td>${E(x.calculation)}</td><td>${E(x.direction)}</td></tr>`).join('');$('aggregate').innerHTML=D.final_report_ready?D.final_aggregate.map(x=>`<div class="agg"><b>${E(x.mode)} · λ${x.lambda}</b><span>完成 ${x.completed_target_count}/${x.eligible_target_count} · gated ${x.guided_target_gate_pass_count} · ΔADE/D0 ${F(x.case_balanced_mean_delta_ade_d0)} · ΔTrack Loss ${F(x.case_balanced_mean_delta_track_loss,1)} · 改善 cases ${x.improved_case_count}</span></div>`).join(''):'<p class="pending">完整三模式指标尚未齐全；汇总将在最后一个评估完成后自动出现。</p>';reps()}
async function load(){D=await fetch(api+'/catalog?x='+Date.now()).then(r=>r.json());const old=$('case').value,dualOld=$('dualCase').value;$('case').innerHTML=D.cases.map(c=>`<option>${E(c.case)}</option>`).join('');if(D.cases.some(c=>c.case===old))$('case').value=old;$('dualCase').innerHTML=D.dual_protocol.cases.map(c=>`<option>${E(c.case)}</option>`).join('');if(D.dual_protocol.cases.some(c=>c.case===dualOld))$('dualCase').value=dualOld;dualSummary();renderDual();summary();render();const now=new Date().toLocaleTimeString();$('updated').textContent=now;$('dualUpdated').textContent=`last scan ${now}`}
$('dualCase').onchange=renderDual;$('dualTarget').onchange=renderDual;$('dualRefresh').onclick=load;$('dualReplay').onclick=()=>document.querySelectorAll('#dualGallery video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});$('case').onchange=render;$('target').onchange=render;$('view').onchange=render;$('refresh').onclick=load;$('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});load();setInterval(load,30000);
</script></body></html>'''
