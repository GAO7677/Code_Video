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

PILOT_CASES = (
    {
        "case": "0613pybullet_sample_000331_w001",
        "seed": 90094,
        "target": "orange sphere",
        "cohort": "positive",
        "baseline": GATE_BASELINE,
        "gate_tokens": 25,
    },
    {
        "case": (
            "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_"
            "crop_top60px"
        ),
        "seed": 13248,
        "target": "brown tennis ball",
        "cohort": "positive",
        "baseline": (
            Path(
                "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
                "visual_samples/attention_zero_seed47326/multicase_multiseed_baselines"
            )
            / "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px"
            / "seed_13248/generated.mp4"
        ),
        "gate_tokens": 105,
    },
    {
        "case": "0613pybullet_sample_000301_w000",
        "seed": 13248,
        "target": "orange sphere",
        "cohort": "clean-control",
        "baseline": Path(
            "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50/"
            "runs/0613pybullet_sample_000301_w000/seed_13248/generated.mp4"
        ),
        "gate_tokens": 0,
    },
    {
        "case": PHYSIQ_DISK_CASE,
        "seed": 47326,
        "target": "brown tennis ball",
        "cohort": "clean-control",
        "baseline": PHYSIQ_BASE / PHYSIQ_DISK_CASE / "seed_47326/generated.mp4",
        "gate_tokens": 0,
    },
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


def _evaluation_index(
    case: str, seed: int, preferred_box_threshold: float
) -> dict[str, dict[str, Any]]:
    """Index the closest primary detector audit for every evaluated video."""

    root = ROOT / "evaluation" / case / f"seed_{seed:05d}"
    indexed: dict[str, tuple[tuple[float, int, str], dict[str, Any]]] = {}
    for report_path in sorted(root.glob("*/report.json")):
        report = load_json(report_path)
        detector = report.get("detector", {})
        threshold = float(detector.get("box_threshold", preferred_box_threshold))
        report_name = report_path.parent.name
        # Prefer the registered detector threshold, then the consolidated/pilot audit.
        report_priority = 0 if report_name == "pilot_three_way" else 1
        score = (abs(threshold - preferred_box_threshold), report_priority, report_name)
        for row in report.get("records", []):
            video_value = row.get("video")
            label = str(row.get("label", ""))
            if not video_value or not label:
                continue
            overlay = report_path.parent / "overlays" / f"{label}.mp4"
            payload = {
                "metrics": metric_payload(row),
                "overlay_path": str(overlay.resolve()) if overlay.is_file() else None,
                "detector_box_threshold": threshold,
                "audit": report_name,
            }
            raw_key = str(Path(video_value).absolute())
            old = indexed.get(raw_key)
            if old is None or score < old[0]:
                indexed[raw_key] = (score, payload)
    return {key: value for key, (_, value) in indexed.items()}


def _experiment_label(manifest: dict[str, Any]) -> tuple[str, str, str]:
    direct_mode = manifest.get("direct_mode")
    if direct_mode:
        mode_label = {
            "measured": "Q(R0) → K(F) 删除",
            "reverse": "Q(F) → K(R) 删除",
            "bidirectional": "R0 ↔ F 双向删除",
        }.get(str(direct_mode), str(direct_mode))
        start, end = manifest.get("guidance_step_range_inclusive", [None, None])
        step_label = (
            f"S{int(start):02d}–S{int(end):02d}"
            if start is not None and end is not None
            else "step N/A"
        )
        triggered = bool(manifest.get("intervention_triggered", True))
        if not triggered:
            return (
                f"Direct softmax · 0-token exact no-op · {mode_label}",
                "noop",
                "Direct softmax (new)",
            )
        return (
            f"Direct softmax · {mode_label} · {step_label}",
            "direct-softmax",
            "Direct softmax (new)",
        )
    mode = str(manifest.get("branch_mode", "unknown"))
    mode_label = {
        "outgoing": "R K/V → F Query",
        "incoming": "F K/V → R Query",
        "bidirectional": "R ↔ F 双向",
    }.get(mode, mode)
    start, end = manifest.get("guidance_step_range_inclusive", [None, None])
    step_label = (
        f"S{int(start):02d}–S{int(end):02d}"
        if start is not None and end is not None
        else "step N/A"
    )
    scale = manifest.get("guidance_scale")
    scale_label = f"lambda={float(scale):g}" if scale is not None else "lambda=N/A"
    external = bool(manifest.get("external_secondary_mask_npz"))
    triggered = bool(manifest.get("intervention_triggered", True))
    if external and not triggered:
        return (
            f"Detector gate · 0-token exact no-op · {mode_label}",
            "noop",
            "Detector-gated",
        )
    if external:
        return (
            f"Detector-confirmed · {mode_label} · {scale_label} · {step_label}",
            "targeted",
            "Detector-gated",
        )
    quantile = manifest.get("quantile")
    percentile = f"P{round(100 * float(quantile))}" if quantile is not None else "P?"
    return (
        f"Broad Q@K {percentile} · {mode_label} · {scale_label} · {step_label}",
        "rejected-broad",
        "Broad Q@K scan",
    )


def _all_case_rows(assets: dict[str, str]) -> list[dict[str, Any]]:
    """Collect every generated experiment under its case and seed."""

    pilot_specs = {
        (str(spec["case"]), int(spec["seed"])): spec for spec in PILOT_CASES
    }
    case_rows: dict[str, dict[str, Any]] = {}
    experiment_roots = [ROOT / "guided", ROOT / "direct_softmax"]
    case_names = sorted(
        {
            path.name
            for root in experiment_roots
            if root.is_dir()
            for path in root.iterdir()
            if path.is_dir()
        }
    )
    for case in case_names:
        case_row = case_rows.setdefault(case, {"case": case, "seeds": []})
        seed_names = sorted(
            {
                path.name
                for root in experiment_roots
                for path in (root / case).glob("seed_*")
                if path.is_dir()
            }
        )
        for seed_name in seed_names:
            try:
                seed = int(seed_name.removeprefix("seed_"))
            except ValueError:
                continue
            result_dirs = sorted(
                {
                    path.parent
                    for root in experiment_roots
                    for path in (root / case / seed_name).glob("*/generated.mp4")
                    if path.exists()
                }
            )
            if not result_dirs:
                continue
            manifests = [(path, load_json(path / "manifest.json")) for path in result_dirs]
            manifests = [(path, row) for path, row in manifests if row]
            if not manifests:
                continue
            first_manifest = manifests[0][1]
            spec = pilot_specs.get((case, seed), {})
            threshold = 0.65 if case.startswith("0613") else 0.55
            evaluation = _evaluation_index(case, seed, threshold)
            baseline_value = next(
                (
                    row.get("baseline_video")
                    for _, row in manifests
                    if row.get("baseline_video")
                ),
                spec.get("baseline"),
            )
            if baseline_value is None:
                baseline_value = (manifests[0][0] / "generated.mp4").resolve()
            baseline = Path(baseline_value)
            items: list[dict[str, Any]] = []

            def append_item(
                *,
                label: str,
                path: Path,
                kind: str,
                group: str,
                manifest: dict[str, Any] | None = None,
            ) -> None:
                if not path.exists():
                    return
                item_index = len(items)
                key = f"all-{len(case_rows)}-{seed}-{item_index}"
                assets[key] = str(path.resolve())
                audit = evaluation.get(str(path.absolute()), {})
                if not audit and manifest and manifest.get("intervention_triggered") is False:
                    # Exact no-op outputs are symlinks to Baseline; their audit may record
                    # the resolved Baseline path instead of the experiment symlink.
                    audit = evaluation.get(str(baseline.absolute()), {})
                overlay_key = None
                overlay_value = audit.get("overlay_path")
                if overlay_value and Path(overlay_value).is_file():
                    overlay_key = f"{key}-overlay"
                    assets[overlay_key] = overlay_value
                entry = {
                    "label": label,
                    "asset": key,
                    "overlay": overlay_key,
                    "kind": kind,
                    "group": group,
                    "metrics": audit.get("metrics"),
                    "detector_box_threshold": audit.get("detector_box_threshold"),
                    "audit": audit.get("audit"),
                }
                if manifest:
                    direct_directions = (
                        manifest.get("audit", {})
                        .get("attention", {})
                        .get("directions", {})
                    )
                    entry.update(
                        {
                            "branch_mode": manifest.get("branch_mode"),
                            "direct_mode": manifest.get("direct_mode"),
                            "guidance_scale": manifest.get("guidance_scale"),
                            "guidance_steps": manifest.get(
                                "guidance_step_range_inclusive"
                            ),
                            "intervention_triggered": manifest.get(
                                "intervention_triggered", True
                            ),
                            "no_op_reason": manifest.get("no_op_reason"),
                            "direct_audit": [
                                {
                                    "direction": direction,
                                    "events": values.get("events"),
                                    "pre_mass": values.get(
                                        "pre_attention_mass_mean"
                                    ),
                                    "post_mass": values.get(
                                        "post_attention_mass_mean"
                                    ),
                                    "removed_av": values.get(
                                        "removed_av_norm_mean"
                                    ),
                                }
                                for direction, values in direct_directions.items()
                            ],
                        }
                    )
                items.append(entry)

            append_item(
                label="Baseline · 同 seed 未干预控制",
                path=baseline,
                kind="control",
                group="Control",
            )
            experiment_rows = []
            for result_dir, manifest in manifests:
                label, kind, group = _experiment_label(manifest)
                group_order = {
                    "Direct softmax (new)": 0,
                    "Detector-gated": 1,
                    "Broad Q@K scan": 2,
                }.get(group, 9)
                experiment_rows.append(
                    (group_order, label, result_dir, manifest, kind, group)
                )
            for _, label, result_dir, manifest, kind, group in sorted(experiment_rows):
                append_item(
                    label=label,
                    path=result_dir / "generated.mp4",
                    kind=kind,
                    group=group,
                    manifest=manifest,
                )
            case_row["seeds"].append(
                {
                    "seed": seed,
                    "target": spec.get("target", first_manifest.get("target_phrase", "N/A")),
                    "cohort": spec.get("cohort", "clean-control"),
                    "gate_tokens": int(spec.get("gate_tokens", 0)),
                    "items": items,
                    "experiment_count": len(items),
                }
            )
    ordered_cases = [str(spec["case"]) for spec in PILOT_CASES]
    order = {case: index for index, case in enumerate(dict.fromkeys(ordered_cases))}
    return sorted(
        case_rows.values(), key=lambda row: (order.get(row["case"], 999), row["case"])
    )


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

    pilot_rows = []
    for row_index, spec in enumerate(PILOT_CASES):
        case = str(spec["case"])
        seed = int(spec["seed"])
        guided = ROOT / "guided" / case / f"seed_{seed:05d}"
        audit_root = ROOT / "evaluation" / case / f"seed_{seed:05d}" / "pilot_three_way"
        audit_rows = records(audit_root / "report.json")
        candidates = (
            (
                "Baseline",
                "Baseline · 同 seed 控制",
                Path(spec["baseline"]),
                "control",
            ),
            (
                "Broad_P95",
                "旧方案 · Q@K P95 全局区域 · lambda=-1",
                guided
                / "secondary_branch__outgoing__top100_s039r3350__p95d1__pagm1__denoise_00_09/generated.mp4",
                "rejected-broad",
            ),
            (
                "Detector_Gated",
                (
                    "新方案 · 检测确认后定向抑制 · lambda=-0.25"
                    if int(spec["gate_tokens"])
                    else "新方案 · 0-token exact no-op"
                ),
                guided
                / "secondary_branch__outgoing__top100_s039r3350__rgb_duplicate__pagm0p25__denoise_00_09/generated.mp4",
                "targeted" if int(spec["gate_tokens"]) else "noop",
            ),
        )
        items = []
        for item_index, (label_key, label, path, kind) in enumerate(candidates):
            key = f"pilot-{row_index}-{item_index}"
            if path.is_file():
                assets[key] = str(path.resolve())
            overlay = audit_root / "overlays" / f"{label_key}.mp4"
            overlay_key = f"{key}-overlay"
            if overlay.is_file():
                assets[overlay_key] = str(overlay.resolve())
            items.append(
                {
                    "label": label,
                    "asset": key if key in assets else None,
                    "overlay": overlay_key if overlay_key in assets else None,
                    "kind": kind,
                    "metrics": metric_payload(audit_rows.get(label_key)),
                }
            )
        pilot_rows.append(
            {
                "case": case,
                "seed": seed,
                "target": spec["target"],
                "cohort": spec["cohort"],
                "gate_tokens": int(spec["gate_tokens"]),
                "items": items,
            }
        )
    return {
        "gate_case": GATE_CASE,
        "gate_seed": 90094,
        "gate": gate_rows,
        "physiq_case": PHYSIQ_CASE,
        "physiq": physiq_rows,
        "pilot": pilot_rows,
        "cases": _all_case_rows(assets),
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
:root{--bg:#ece9e1;--ink:#17201d;--card:#fffdf8;--line:#bbb4a7;--red:#bb3e32;--blue:#155f76;--green:#19745d;--amber:#c57812;--muted:#68716d;--dark:#17201d}*{box-sizing:border-box}body{margin:0;background:linear-gradient(120deg,#b4473020,transparent 32rem),var(--bg);color:var(--ink);font-family:Inter,"Noto Sans SC",system-ui,sans-serif}header{background:var(--dark);color:#fff;padding:24px clamp(16px,4vw,64px) 28px}header a{color:#9bd9d0}.eyebrow{font:12px ui-monospace,monospace;color:#ef9e79;letter-spacing:.14em}h1{font-size:clamp(35px,6vw,72px);line-height:.95;letter-spacing:-.05em;margin:12px 0}.lead{max-width:1100px;line-height:1.65;color:#d8e0dc}.case-tool{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:end;max-width:1250px;margin-top:22px}.case-tool label{display:grid;gap:6px;font:11px ui-monospace,monospace;color:#9bd9d0;text-transform:uppercase;letter-spacing:.08em}.case-tool select{width:100%;padding:12px 42px 12px 13px;background:#fff;color:var(--ink);border:0;border-radius:0;font:14px ui-monospace,monospace}.case-count{font:12px ui-monospace,monospace;color:#d8e0dc;padding-bottom:12px;white-space:nowrap}main{width:min(1900px,calc(100% - 24px));margin:auto;padding:20px 0 80px}.defs{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.def,.finding,.seed-block{background:var(--card);border:1px solid var(--line)}.def{padding:15px}.def h2,.finding h2{margin:0 0 8px;font-size:18px}.def p{margin-bottom:0;line-height:1.55}.formula{font:12px/1.55 ui-monospace,monospace;background:#e3dfd5;padding:9px}.warn{border-left:6px solid var(--red);background:#fff2e9;padding:13px 16px;margin:14px 0;line-height:1.55}.case-title{margin:34px 0 5px;font:clamp(21px,3vw,34px)/1.1 ui-monospace,monospace;overflow-wrap:anywhere}.case-summary{color:var(--muted);margin:0 0 14px}.seed-block{margin:14px 0 24px;padding:15px}.seed-head{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding-bottom:12px;border-bottom:1px solid var(--line)}.seed-head h2{font:21px ui-monospace,monospace;margin:0 auto 0 0}.badge{font:11px ui-monospace,monospace;padding:4px 7px;border-radius:20px;background:#e3dfd5}.badge.positive{background:#ffe0d4;color:#8b291d}.badge.clean-control{background:#dceee8;color:#145c49}.experiment-group{margin-top:18px}.group-head{display:flex;align-items:baseline;gap:10px;margin-bottom:9px}.group-head h3{margin:0;font-size:17px}.group-head span{font:11px ui-monospace,monospace;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);min-width:0;overflow:hidden}.card.control{border-top:5px solid var(--dark)}.card.direct-softmax{border-top:5px solid var(--amber)}.card.targeted{border-top:5px solid var(--blue)}.card.rejected-broad{border-top:5px solid var(--red)}.card.noop{border-top:5px solid var(--green)}.card h4{font-size:14px;line-height:1.4;margin:0;padding:11px 12px;min-height:59px}.card video{display:block;width:100%;aspect-ratio:1280/704;background:#111;object-fit:contain}.facts{display:flex;gap:5px;flex-wrap:wrap;padding:9px 10px;border-top:1px solid var(--line)}.fact{font:10px ui-monospace,monospace;background:#e9e5dc;padding:4px 6px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line)}.metric{padding:9px;border-right:1px solid var(--line);font:10px ui-monospace,monospace}.metric:last-child{border:0}.metric b{display:block;font-size:15px;margin-bottom:4px}.metric-note{padding:7px 10px;border-top:1px solid var(--line);font:10px ui-monospace,monospace;color:var(--muted)}.overlay{padding:9px 12px;border-top:1px solid var(--line)}.overlay summary{cursor:pointer;color:var(--blue);font-size:13px}.overlay video{margin-top:9px}.finding{margin-top:16px;padding:15px}.finding li{margin:8px 0;line-height:1.55}.empty{padding:60px 20px;text-align:center;color:var(--muted);background:var(--card);border:1px solid var(--line)}@media(max-width:900px){.defs{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.case-tool{grid-template-columns:1fr}.case-count{padding:0}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}</style></head><body><header><a href="/">← 返回 8092 总入口</a><div class="eyebrow">ONE CASE · ALL GENERATED EXPERIMENTS</div><h1>一个 case，一张完整实验桌。</h1><p class="lead">选择 case 后，在同一页查看它的全部 seed、Baseline、直接 softmax 删除、旧检测门控实验、宽泛 Q@K 扫描和已计算 overlay。未生成的组合不占位；视频进入视口附近才加载。</p><div class="case-tool"><label>选择 Case<select id="case-select" aria-label="选择 case"></select></label><span id="case-count" class="case-count">读取中</span></div></header><main><section class="defs"><article class="def"><h2>Q@K 只作候选定位</h2><div class="formula">S(k)=mean_h softmax(Q_R0 K_k^T / sqrt(d))</div><p>多峰不等于多个因果实例；额外区域 F 仍由同类检测器确认。</p></article><article class="def"><h2>新方案：真实 softmax 删除</h2><div class="formula">A[q,F]=0; A[q,:] ← A[q,:] / sum A[q,:]</div><p>分别测试 Q(R0)→K(F)、Q(F)→K(R)，以及双向；直接作用于条件前向的 A@V，不再混合 PAG 分支。</p></article><article class="def"><h2>安全门控</h2><div class="formula">F=∅ ⇒ intervention_triggered=false</div><p>没有检测到额外同类实例时，三组实验均精确复用 Baseline。</p></article></section><div class="warn"><b>结论边界：</b>当前共有 4 个 case、6 个 case-seed 样本；2 个重复正例各只有 1 个 seed。Extra 下降必须与 Missing、ADE/D0 一起判断，不能只凭“额外实例消失”宣布修复成功。</div><div id="content"><div class="empty">正在整理各 case 的全部实验…</div></div><section class="finding"><h2>如何读指标</h2><ul><li><b>Extra</b>：49 帧中同类实例数超过期望数量的帧数，越小越好。</li><li><b>Missing</b>：49 帧中检测不到主目标的帧数，越小越好。</li><li><b>ADE/D0</b>：候选与同 seed Baseline 的目标中心平均距离，以首帧目标尺寸归一化；越小越保真。</li><li><b>MAE</b>：全帧像素差，只作全局变化 sanity check。</li></ul></section></main><script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n)=>v===null||v===undefined?'N/A':Number(v).toFixed(n);
const media=key=>`/api/object-query-anti-duplication/asset?key=${encodeURIComponent(key)}`;
let observer;
function activateLazy(root=document){if(observer)observer.disconnect();observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(!entry.isIntersecting)return;const video=entry.target;if(video.dataset.src&&!video.src)video.src=video.dataset.src;observer.unobserve(video)}),{rootMargin:'500px 0px'});root.querySelectorAll('video[data-src]').forEach(video=>observer.observe(video))}
function card(x){const m=x.metrics;const metrics=m?`<div class="metrics"><div class="metric"><b>${m.extra_frames}/49</b>Extra ↓</div><div class="metric"><b>${m.missing_frames}/49</b>Missing ↓</div><div class="metric"><b>${fmt(m.center_ade_d0,3)}</b>ADE/D0 ↓</div><div class="metric"><b>${fmt(m.full_frame_mae,5)}</b>MAE ↓</div></div><div class="metric-note">检测阈值 ${fmt(x.detector_box_threshold,2)} · audit=${esc(x.audit||'N/A')}</div>`:'<div class="metric-note">该结果尚无匹配的检测指标记录</div>';const facts=[];if(x.direct_mode)facts.push(`direct=${x.direct_mode}`);if(x.branch_mode)facts.push(x.branch_mode);if(x.guidance_scale!==undefined&&x.guidance_scale!==null)facts.push(`lambda=${Number(x.guidance_scale)}`);if(x.guidance_steps)facts.push(`S${String(x.guidance_steps[0]).padStart(2,'0')}–S${String(x.guidance_steps[1]).padStart(2,'0')}`);facts.push(x.intervention_triggered===false?'not triggered':'generated');const directAudit=(x.direct_audit||[]).length?`<div class="metric-note">${x.direct_audit.map(a=>`${esc(a.direction)} · ${a.events} events · QK ${fmt(a.pre_mass,6)}→${fmt(a.post_mass,6)} · mean |ΔAV|=${fmt(a.removed_av,5)}`).join('<br>')}</div>`:'';const overlay=x.overlay?`<details class="overlay"><summary>查看实际检测点 / 框 overlay</summary><video controls muted playsinline preload="none" data-src="${media(x.overlay)}"></video></details>`:'';return `<article class="card ${esc(x.kind)}"><h4>${esc(x.label)}</h4><video controls muted playsinline preload="none" data-src="${media(x.asset)}"></video><div class="facts">${facts.map(value=>`<span class="fact">${esc(value)}</span>`).join('')}</div>${directAudit}${metrics}${overlay}</article>`}
function renderCase(row){const groups=['Control','Direct softmax (new)','Detector-gated','Broad Q@K scan'];const seeds=row.seeds.map(seed=>{const sections=groups.map(group=>{const items=seed.items.filter(item=>item.group===group);return items.length?`<section class="experiment-group"><div class="group-head"><h3>${esc(group)}</h3><span>${items.length} 个已生成结果</span></div><div class="grid">${items.map(card).join('')}</div></section>`:''}).join('');return `<section class="seed-block"><div class="seed-head"><h2>seed ${seed.seed}</h2><span class="badge ${esc(seed.cohort)}">${seed.cohort==='positive'?'重复正例':'干净对照'}</span><span class="badge">target=${esc(seed.target)}</span><span class="badge">gate=${seed.gate_tokens} tokens</span><span class="badge">${seed.experiment_count} videos</span></div>${sections}</section>`}).join('');document.querySelector('#content').innerHTML=`<h2 class="case-title">${esc(row.case)}</h2><p class="case-summary">${row.seeds.length} 个 seed；本页汇总 ${row.seeds.reduce((n,seed)=>n+seed.experiment_count,0)} 个 Baseline / 实验结果。</p>${seeds}`;activateLazy(document.querySelector('#content'))}
async function init(){const response=await fetch('/api/object-query-anti-duplication/catalog',{cache:'no-store'});const data=await response.json();const rows=data.cases||[];const select=document.querySelector('#case-select');select.innerHTML=rows.map(row=>`<option value="${esc(row.case)}">${esc(row.case)} (${row.seeds.reduce((n,seed)=>n+seed.experiment_count,0)})</option>`).join('');document.querySelector('#case-count').textContent=`${rows.length} cases · ${rows.reduce((n,row)=>n+row.seeds.length,0)} case-seed samples`;const requested=new URLSearchParams(location.search).get('case');if(requested&&rows.some(row=>row.case===requested))select.value=requested;const render=()=>{const row=rows.find(item=>item.case===select.value)||rows[0];if(!row){document.querySelector('#content').innerHTML='<div class="empty">当前没有已生成实验</div>';return}select.value=row.case;history.replaceState(null,'',`${location.pathname}?case=${encodeURIComponent(row.case)}&v=4`);renderCase(row)};select.addEventListener('change',render);render()}
init().catch(error=>{document.querySelector('#content').innerHTML=`<div class="empty">目录读取失败：${esc(error.message)}</div>`});</script></body></html>'''
