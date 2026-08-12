#!/usr/bin/env python3
"""Case-balanced paired analysis of all currently available Stage-3 metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1"
)
SCOPES = (
    "top100",
    "bottom100",
    "random100_layer_matched_draw0",
    "all720",
)
FLOWS = ("self_only", "incoming_only", "outgoing_only")
SCOPE_LABELS = {
    "top100": "Top100",
    "bottom100": "Bottom100",
    "random100_layer_matched_draw0": "Random100",
    "all720": "All720",
}
FLOW_LABELS = {
    "self_only": "M1 · R→R",
    "incoming_only": "M2 · C→R",
    "outgoing_only": "M3 · R→C",
}
METRICS = {
    "attention_mass": {
        "family": "dose",
        "label": "Removed attention mass / head",
        "unit": "probability mass/head",
        "meaning": "每个 selected head 被删除的信息流概率质量；不是输出效果。",
    },
    "removed_value_norm": {
        "family": "dose",
        "label": "Removed AV norm / head",
        "unit": "L2/head",
        "meaning": "精确删除的 ΣA·V 项的每-head L2 范数；不是语义指标。",
    },
    "removed_to_output_ratio": {
        "family": "dose",
        "label": "Removed/output norm ratio",
        "unit": "ratio",
        "meaning": "删除贡献相对原 attention head 输出的比例。",
    },
    "target_local": {
        "family": "fast",
        "label": "Target frozen-ROI MAE ×100",
        "unit": "0–100",
        "meaning": "Baseline 对象 tube ROI 内的像素变化；混合位置、外观、形状和消失。",
    },
    "global_appearance": {
        "family": "fast",
        "label": "Global pixel/structure effect ×100",
        "unit": "0–100",
        "meaning": "全帧 SSIM 与 MAE 的诊断组合；不是中心对齐对象外观。",
    },
    "outside_static": {
        "family": "fast",
        "label": "Outside-object MAE ×100",
        "unit": "0–100",
        "meaning": "Baseline 对象 ROI 外的像素变化，是背景/其他对象 spillover 粗代理。",
    },
    "center_ade": {
        "family": "trajectory",
        "label": "Target Center-ADE / D0",
        "unit": "D0",
        "meaning": "通过轨迹门控后，目标中心相对 Baseline 的逐帧平均距离。",
    },
    "center_fde": {
        "family": "trajectory",
        "label": "Target Center-FDE / D0",
        "unit": "D0",
        "meaning": "通过轨迹门控后，最后共同有效帧的目标中心距离。",
    },
    "velocity": {
        "family": "trajectory",
        "label": "Velocity vector error / D0/frame",
        "unit": "D0/frame",
        "meaning": "通过门控且有足够帧时，四帧差分速度向量相对 Baseline 的误差。",
    },
    "pck10_failure": {
        "family": "trajectory",
        "label": "100 × (1 − PCK@10%D0)",
        "unit": "percentage points",
        "meaning": "共同可见点中未落入 10%D0 阈值的比例；越大轨迹偏差越强。",
    },
    "track_loss": {
        "family": "trajectory",
        "label": "Worst target Track Loss",
        "unit": "0–100",
        "meaning": "selected objects 中最差的 CoTracker 可观测性损失；门控失败时仍保留。",
    },
    "other_ade": {
        "family": "trajectory",
        "label": "Other-object Center-ADE / D0",
        "unit": "D0",
        "meaning": "单对象消融后，未选中对象相对 Baseline 的平均中心轨迹变化。",
    },
    "disappearance": {
        "family": "survival",
        "label": "Worst target Disappearance",
        "unit": "0–100",
        "meaning": "selected objects 中最差的非存活帧比例；综合身份、面积和空 mask。",
    },
    "mask_absence": {
        "family": "survival",
        "label": "Worst target Mask Absence",
        "unit": "0–100",
        "meaning": "selected objects 中最差的 SAM2 空 mask 帧比例；只表示字面消失。",
    },
    "identity_failure": {
        "family": "survival",
        "label": "Worst target Identity Failure",
        "unit": "0–100",
        "meaning": "selected objects 中最差的 DINO 身份阈值失败帧比例。",
    },
    "area_failure": {
        "family": "survival",
        "label": "Worst target Area Failure",
        "unit": "0–100",
        "meaning": "selected objects 中最差的面积比例越界帧比例。",
    },
    "terminal_missing": {
        "family": "survival",
        "label": "Worst terminal missing rate",
        "unit": "0–100",
        "meaning": "selected objects 在最后八帧中的最差非存活比例。",
    },
}
# Keep the report exhaustive: every metric currently emitted by the Stage-3
# collectors must appear in the human-readable tables, not only in report.json.
REPORT_METRICS = tuple(METRICS)
HEAD_CONTRASTS = (
    ("top100", "bottom100", "Top100 − Bottom100"),
    ("top100", "random100_layer_matched_draw0", "Top100 − Random100"),
    ("all720", "top100", "All720 − Top100"),
)
FLOW_CONTRASTS = (
    ("self_only", "incoming_only", "M1 − M2"),
    ("self_only", "outgoing_only", "M1 − M3"),
    ("incoming_only", "outgoing_only", "M2 − M3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def finite_mean(values: Iterable[Any]) -> float | None:
    selected = [float(value) for value in values if finite(value)]
    return float(np.mean(selected)) if selected else None


def target_id(source: dict[str, Any]) -> str:
    if source.get("target_scope") == "single_object":
        return f"single_object::{source.get('region')}"
    return "all_objects::all_objects"


def base_record(
    report: dict[str, Any], source: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "case": str(report["case"]),
        "seed": int(report["seed"]),
        "target_scope": str(source["target_scope"]),
        "target": target_id(source),
        "flow": str(source["mask_mode"]),
        "scope": str(source["head_scope"]),
        "variant_id": str(source["variant_id"]),
        "metrics": metrics,
    }


def collect_dose(root: Path) -> list[dict[str, Any]]:
    path = root / "stage3_interim_analysis" / "report.json"
    report = read_json(path)
    output = []
    for source in report.get("records", []):
        target_scope = str(source["target"]).split("::", 1)[0]
        region = str(source["target"]).split("::", 1)[1]
        synthetic = {
            "target_scope": target_scope,
            "region": region,
            "mask_mode": source["flow"],
            "head_scope": source["scope"],
            "variant_id": source["variant_id"],
        }
        output.append(
            base_record(
                {"case": source["case"], "seed": source["seed"]},
                synthetic,
                {
                    key: source.get(key)
                    for key in (
                        "attention_mass",
                        "removed_value_norm",
                        "removed_to_output_ratio",
                    )
                },
            )
        )
    return output


def collect_fast(root: Path) -> list[dict[str, Any]]:
    output = []
    report_root = root / "stage3_metrics" / "head_scope_baseline_fast"
    for path in sorted(report_root.glob("*/seed_*/report.json")):
        report = read_json(path)
        for source in report.get("records", []):
            metrics = source["metrics"]
            categories = metrics["category_scores_0_100"]
            output.append(
                base_record(
                    report,
                    source,
                    {
                        "target_local": categories["target_local"],
                        "global_appearance": categories["global_appearance"],
                        "outside_static": 100.0
                        * metrics["outside_objects"]["mae_0_1"],
                    },
                )
            )
    return output


def collect_trajectory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    quality = defaultdict(lambda: {"total": 0, "failed": 0})
    report_root = root / "stage3_metrics" / "head_scope_trajectory"
    for path in sorted(report_root.glob("*/seed_*/report.json")):
        report = read_json(path)
        for source in report.get("records", []):
            metrics = source["metrics"]
            selected = list(metrics["selected_objects"])
            selected_set = set(selected)
            objects = metrics["objects"]
            passed = metrics.get("quality_pass") is True
            quality[(source["mask_mode"], source["head_scope"])]["total"] += 1
            quality[(source["mask_mode"], source["head_scope"])]["failed"] += int(
                not passed
            )
            pck10 = (
                finite_mean(
                    objects[name].get("pck_normalized", {}).get("0.1")
                    for name in selected
                )
                if passed
                else None
            )
            other_objects = [
                value for name, value in objects.items() if name not in selected_set
            ]
            output.append(
                base_record(
                    report,
                    source,
                    {
                        "center_ade": metrics.get("target_center_ade_norm")
                        if passed
                        else None,
                        "center_fde": finite_mean(
                            objects[name].get("center_fde_norm") for name in selected
                        )
                        if passed
                        else None,
                        "velocity": finite_mean(
                            objects[name].get("velocity_vector_error_norm_per_frame")
                            for name in selected
                        )
                        if passed
                        else None,
                        "pck10_failure": 100.0 * (1.0 - pck10)
                        if pck10 is not None
                        else None,
                        "track_loss": metrics[
                            "target_worst_track_loss_score_0_100"
                        ],
                        "other_ade": finite_mean(
                            value.get("center_ade_norm") for value in other_objects
                        )
                        if source["target_scope"] == "single_object"
                        else None,
                    },
                )
            )
    quality_rows = []
    for (flow, scope), values in sorted(quality.items()):
        quality_rows.append(
            {
                "flow": flow,
                "scope": scope,
                "total": values["total"],
                "failed": values["failed"],
                "failure_rate": values["failed"] / values["total"],
            }
        )
    return output, {"rows": quality_rows}


def collect_survival(root: Path) -> list[dict[str, Any]]:
    output = []
    report_root = root / "stage3_metrics" / "head_scope_trajectory"
    for path in sorted(report_root.glob("*/seed_*/object_survival_report.json")):
        report = read_json(path)
        for source in report.get("records", []):
            metrics = source["metrics"]
            selected = list(metrics["selected_objects"])
            objects = metrics["objects"]
            output.append(
                base_record(
                    report,
                    source,
                    {
                        "disappearance": metrics[
                            "target_worst_disappearance_score_0_100"
                        ],
                        "mask_absence": metrics[
                            "target_worst_mask_absence_score_0_100"
                        ],
                        "identity_failure": 100.0
                        * max(objects[name]["identity_failure_rate"] for name in selected),
                        "area_failure": 100.0
                        * max(objects[name]["area_failure_rate"] for name in selected),
                        "terminal_missing": 100.0
                        * max(objects[name]["terminal_missing_rate"] for name in selected),
                    },
                )
            )
    return output


def group_records(
    records: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    output: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        output[tuple(record[field] for field in fields)].append(record)
    return output


def scope_units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for key, rows in group_records(
        records, ("case", "seed", "target", "flow")
    ).items():
        scopes = {row["scope"]: row for row in rows}
        if set(scopes) != set(SCOPES):
            continue
        case, seed, target, flow = key
        output.append(
            {
                "case": case,
                "seed": seed,
                "target": target,
                "flow": flow,
                "scopes": scopes,
            }
        )
    return output


def flow_units(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for key, rows in group_records(
        records, ("case", "seed", "target", "scope")
    ).items():
        flows = {row["flow"]: row for row in rows}
        if set(flows) != set(FLOWS):
            continue
        case, seed, target, scope = key
        output.append(
            {
                "case": case,
                "seed": seed,
                "target": target,
                "scope": scope,
                "flows": flows,
            }
        )
    return output


def case_aggregate(
    rows: Iterable[dict[str, Any]], value: Callable[[dict[str, Any]], Any]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        current = value(row)
        if finite(current):
            grouped[str(row["case"])].append(float(current))
    return {
        case: float(np.mean(values))
        for case, values in grouped.items()
        if values
    }


def average(case_values: dict[str, float]) -> float | None:
    return (
        float(np.mean(list(case_values.values()))) if case_values else None
    )


def bootstrap_ci(
    case_values: dict[str, float], repetitions: int, rng: np.random.Generator
) -> list[float | None]:
    values = np.asarray(list(case_values.values()), dtype=np.float64)
    if values.size == 0:
        return [None, None]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    samples = rng.integers(0, values.size, size=(repetitions, values.size))
    distribution = values[samples].mean(axis=1)
    return [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


def summarize_scopes(
    units: list[dict[str, Any]],
    metric: str,
    flow: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = []
    for unit in units:
        if unit["flow"] != flow:
            continue
        values = {
            scope: unit["scopes"][scope]["metrics"].get(metric)
            for scope in SCOPES
        }
        if all(finite(value) for value in values.values()):
            selected.append({**unit, "values": values})
    summaries = {}
    for scope in SCOPES:
        cases = case_aggregate(selected, lambda row, s=scope: row["values"][s])
        summaries[scope] = {
            "mean": average(cases),
            "ci95": bootstrap_ci(cases, repetitions, rng),
            "case_count": len(cases),
            "case_means": cases,
        }
    return {
        "metric": metric,
        "flow": flow,
        "paired_unit_count": len(selected),
        "case_count": len({row["case"] for row in selected}),
        "scopes": summaries,
    }


def head_contrast(
    units: list[dict[str, Any]],
    metric: str,
    flow: str,
    left: str,
    right: str,
    label: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = []
    for unit in units:
        if unit["flow"] != flow:
            continue
        a = unit["scopes"][left]["metrics"].get(metric)
        b = unit["scopes"][right]["metrics"].get(metric)
        if finite(a) and finite(b):
            selected.append({**unit, "left_value": float(a), "right_value": float(b)})
    left_cases = case_aggregate(selected, lambda row: row["left_value"])
    right_cases = case_aggregate(selected, lambda row: row["right_value"])
    differences = case_aggregate(
        selected, lambda row: row["left_value"] - row["right_value"]
    )
    left_mean, right_mean = average(left_cases), average(right_cases)
    ratio = (
        left_mean / right_mean
        if left_mean is not None and right_mean is not None and abs(right_mean) > 1e-12
        else None
    )
    return {
        "metric": metric,
        "flow": flow,
        "left": left,
        "right": right,
        "label": label,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "difference": average(differences),
        "ratio": ratio,
        "ci95": bootstrap_ci(differences, repetitions, rng),
        "case_positive_fraction": (
            sum(value > 0 for value in differences.values()) / len(differences)
            if differences
            else None
        ),
        "case_count": len(differences),
        "paired_unit_count": len(selected),
        "case_differences": differences,
    }


def summarize_flows(
    units: list[dict[str, Any]],
    metric: str,
    scope: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = []
    for unit in units:
        if unit["scope"] != scope:
            continue
        values = {
            flow: unit["flows"][flow]["metrics"].get(metric) for flow in FLOWS
        }
        if all(finite(value) for value in values.values()):
            selected.append({**unit, "values": values})
    summaries = {}
    for flow in FLOWS:
        cases = case_aggregate(selected, lambda row, f=flow: row["values"][f])
        summaries[flow] = {
            "mean": average(cases),
            "ci95": bootstrap_ci(cases, repetitions, rng),
            "case_count": len(cases),
            "case_means": cases,
        }
    return {
        "metric": metric,
        "scope": scope,
        "paired_unit_count": len(selected),
        "case_count": len({row["case"] for row in selected}),
        "flows": summaries,
    }


def flow_contrast(
    units: list[dict[str, Any]],
    metric: str,
    scope: str,
    left: str,
    right: str,
    label: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = []
    for unit in units:
        if unit["scope"] != scope:
            continue
        a = unit["flows"][left]["metrics"].get(metric)
        b = unit["flows"][right]["metrics"].get(metric)
        if finite(a) and finite(b):
            selected.append({**unit, "left_value": float(a), "right_value": float(b)})
    left_cases = case_aggregate(selected, lambda row: row["left_value"])
    right_cases = case_aggregate(selected, lambda row: row["right_value"])
    differences = case_aggregate(
        selected, lambda row: row["left_value"] - row["right_value"]
    )
    left_mean, right_mean = average(left_cases), average(right_cases)
    return {
        "metric": metric,
        "scope": scope,
        "left": left,
        "right": right,
        "label": label,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "difference": average(differences),
        "ratio": (
            left_mean / right_mean
            if left_mean is not None
            and right_mean is not None
            and abs(right_mean) > 1e-12
            else None
        ),
        "ci95": bootstrap_ci(differences, repetitions, rng),
        "case_positive_fraction": (
            sum(value > 0 for value in differences.values()) / len(differences)
            if differences
            else None
        ),
        "case_count": len(differences),
        "paired_unit_count": len(selected),
        "case_differences": differences,
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "—" if not finite(value) else f"{float(value):.{digits}f}"


def markdown(report: dict[str, Any]) -> str:
    scope_lookup = {
        (row["metric"], row["flow"]): row for row in report["scope_summaries"]
    }
    head_lookup = {
        (row["metric"], row["flow"], row["label"]): row
        for row in report["head_contrasts"]
    }
    flow_lookup = {
        (row["metric"], row["scope"]): row for row in report["flow_summaries"]
    }
    lines = [
        "# Stage 3 · Current Metrics Statistical Analysis",
        "",
        f"Snapshot: `{report['generated_at_utc']}`",
        "",
        f"- Generated videos with dose: **{report['coverage']['dose_records']} / 1188**.",
        f"- Fast records: **{report['coverage']['fast_records']}**; trajectory records: **{report['coverage']['trajectory_records']}**; survival records: **{report['coverage']['survival_records']}**.",
        f"- Strict four-scope trajectory/survival units: **{report['coverage']['trajectory_scope_units']}**, from **{report['coverage']['trajectory_scope_cases']} cases**.",
        "",
        "> Interim discovery analysis. Case is the highest independent unit. Seeds and objects are averaged within case before cases receive equal weight. CIs are case-bootstrap descriptive intervals; no interim significance decision or BH-FDR claim is made.",
        "",
        "## Metric interpretation",
        "",
        "All outcome metrics below are Baseline-relative intervention effects: larger means a stronger change, not better/worse physical correctness. Dose metrics describe how much was removed. Center-ADE/FDE use only trackable pairs; Track Loss and Disappearance retain destructive failures.",
        "",
        "## Four-head-scope case-balanced means",
        "",
    ]
    for metric in REPORT_METRICS:
        definition = METRICS[metric]
        lines += [f"### {definition['label']}", "", definition["meaning"], ""]
        lines += [
            "| Flow | Top100 | Bottom100 | Random100 | All720 | Cases | Paired units |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for flow in FLOWS:
            row = scope_lookup[(metric, flow)]
            lines.append(
                f"| {FLOW_LABELS[flow]} | "
                + " | ".join(fmt(row["scopes"][scope]["mean"]) for scope in SCOPES)
                + f" | {row['case_count']} | {row['paired_unit_count']} |"
            )
        lines.append("")
    lines += ["## Paired head-group contrasts", ""]
    lines += [
        "| Metric | Flow | Contrast | Left | Right | Difference | Ratio | Case direction | 95% case-bootstrap CI | Cases / units |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in REPORT_METRICS:
        for flow in FLOWS:
            for label in ("Top100 − Bottom100", "Top100 − Random100"):
                row = head_lookup[(metric, flow, label)]
                ci = row["ci95"]
                direction = (
                    100.0 * row["case_positive_fraction"]
                    if row["case_positive_fraction"] is not None
                    else None
                )
                lines.append(
                    f"| {METRICS[metric]['label']} | {FLOW_LABELS[flow]} | {label} | "
                    f"{fmt(row['left_mean'])} | {fmt(row['right_mean'])} | {fmt(row['difference'])} | "
                    f"{fmt(row['ratio'])}× | {fmt(direction, 1)}% | "
                    f"[{fmt(ci[0])}, {fmt(ci[1])}] | {row['case_count']} / {row['paired_unit_count']} |"
                )
    lines += ["", "## M1/M2/M3 comparison within each head group", ""]
    for metric in REPORT_METRICS:
        lines += [f"### {METRICS[metric]['label']}", ""]
        lines += [
            "| Head group | M1 R→R | M2 C→R | M3 R→C | Cases | Paired units |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for scope in SCOPES:
            row = flow_lookup[(metric, scope)]
            lines.append(
                f"| {SCOPE_LABELS[scope]} | "
                + " | ".join(fmt(row["flows"][flow]["mean"]) for flow in FLOWS)
                + f" | {row['case_count']} | {row['paired_unit_count']} |"
            )
        lines.append("")
    lines += [
        "## Trajectory quality-gate audit",
        "",
        "| Flow | Head group | Failed | Total | Failure rate |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["trajectory_quality"]["rows"]:
        lines.append(
            f"| {FLOW_LABELS[row['flow']]} | {SCOPE_LABELS[row['scope']]} | "
            f"{row['failed']} | {row['total']} | {100.0 * row['failure_rate']:.1f}% |"
        )
    lines += [
        "",
        "## Guardrails",
        "",
        "- Four-scope means use only units where all four values are finite. Pairwise Top−Bottom/Top−Random contrasts use their exact common finite cohort, so their means may differ from the four-scope table.",
        "- All720 outcome values are total 720-head intervention effects. All720 dose values remain per-head means and must not be read as total dose.",
        "- Mask Absence only detects an empty SAM2 mask. Disappearance additionally includes identity and implausible-area failures, so the two should not be conflated.",
        "- Outside-object MAE is a pixel proxy. Other-object Center-ADE is the direct cross-object trajectory metric.",
        "- Current cases overlap exploratory/ranking inspection and the 1188 matrix is incomplete; conclusions require the complete matrix and held-out confirmation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = root / "stage3_interim_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        "dose": collect_dose(root),
        "fast": collect_fast(root),
    }
    trajectory, quality = collect_trajectory(root)
    records["trajectory"] = trajectory
    records["survival"] = collect_survival(root)
    family_metrics: dict[str, list[str]] = defaultdict(list)
    for metric, definition in METRICS.items():
        family_metrics[str(definition["family"])].append(metric)
    rng = np.random.default_rng(args.bootstrap_seed)
    scope_summaries = []
    head_contrasts = []
    flow_summaries = []
    flow_contrasts = []
    structural_scope_units = {}
    structural_flow_units = {}
    for family, current_records in records.items():
        scopes = scope_units(current_records)
        flows = flow_units(current_records)
        structural_scope_units[family] = scopes
        structural_flow_units[family] = flows
        for metric in family_metrics[family]:
            for flow in FLOWS:
                scope_summaries.append(
                    summarize_scopes(
                        scopes, metric, flow, args.bootstrap_repetitions, rng
                    )
                )
                for left, right, label in HEAD_CONTRASTS:
                    head_contrasts.append(
                        head_contrast(
                            scopes,
                            metric,
                            flow,
                            left,
                            right,
                            label,
                            args.bootstrap_repetitions,
                            rng,
                        )
                    )
            for scope in SCOPES:
                flow_summaries.append(
                    summarize_flows(
                        flows, metric, scope, args.bootstrap_repetitions, rng
                    )
                )
                for left, right, label in FLOW_CONTRASTS:
                    flow_contrasts.append(
                        flow_contrast(
                            flows,
                            metric,
                            scope,
                            left,
                            right,
                            label,
                            args.bootstrap_repetitions,
                            rng,
                        )
                    )
    report = {
        "schema_version": 1,
        "status": "interim_descriptive_incomplete_matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {
            "highest_independent_unit": "case",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
        },
        "coverage": {
            "dose_records": len(records["dose"]),
            "fast_records": len(records["fast"]),
            "trajectory_records": len(records["trajectory"]),
            "survival_records": len(records["survival"]),
            "dose_scope_units": len(structural_scope_units["dose"]),
            "fast_scope_units": len(structural_scope_units["fast"]),
            "trajectory_scope_units": len(structural_scope_units["trajectory"]),
            "survival_scope_units": len(structural_scope_units["survival"]),
            "trajectory_scope_cases": len(
                {row["case"] for row in structural_scope_units["trajectory"]}
            ),
        },
        "metric_definitions": METRICS,
        "scope_summaries": scope_summaries,
        "head_contrasts": head_contrasts,
        "flow_summaries": flow_summaries,
        "flow_contrasts": flow_contrasts,
        "trajectory_quality": quality,
    }
    atomic_json(output_dir / "current_metrics_report.json", report)
    (output_dir / "STAGE3_CURRENT_METRICS_ANALYSIS.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["coverage"], indent=2))


if __name__ == "__main__":
    main()
