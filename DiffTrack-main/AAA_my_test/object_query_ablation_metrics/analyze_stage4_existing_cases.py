#!/usr/bin/env python3
"""Case-balanced pilot analysis of the currently completed Stage-4 matrix."""

from __future__ import annotations

import argparse
import json
import math
import numbers
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1"
)
FLOWS = ("M1", "M2", "M3")
DIRECTIONS = ("same", "future", "past")
SCOPES = ("top100", "bottom100", "random100_layer_matched_draw0")
FLOW_LABELS = {"M1": "M1 · R→R", "M2": "M2 · C→R", "M3": "M3 · R→C"}
SCOPE_LABELS = {
    "top100": "Top100",
    "bottom100": "Bottom100",
    "random100_layer_matched_draw0": "Random100",
}
METRIC_LABELS = {
    "attention_mass": "Removed attention mass / head",
    "attention_mass_query_sum": "Removed attention mass query-sum / head",
    "removed_value_norm": "Removed AV norm / head",
    "removed_value_norm_query_sum": "Removed AV query-sum / head",
    "removed_to_output_ratio": "Removed/output norm ratio",
    "target_local": "Target frozen-ROI MAE ×100",
    "outside_static": "Outside-object MAE ×100",
    "center_ade": "Target Center-ADE / D0",
    "center_fde": "Target Center-FDE / D0",
    "velocity": "Velocity vector error / D0/frame",
    "pck10_failure": "100 × (1 − PCK@10%D0)",
    "track_loss": "Worst target Track Loss",
    "other_ade": "Other-object Center-ADE / D0",
    "disappearance": "Worst target Disappearance",
    "mask_absence": "Worst target Mask Absence",
    "identity_failure": "Worst target Identity Failure",
    "area_failure": "Worst target Area Failure",
    "terminal_missing": "Worst terminal missing rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def finite(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def finite_mean(values: Iterable[Any]) -> float | None:
    if isinstance(values, np.ndarray):
        selected = values[np.isfinite(values)]
        return float(selected.mean()) if selected.size else None
    selected = [float(value) for value in values if finite(value)]
    return float(np.mean(selected)) if selected else None


def parse_mode(mask_mode: str) -> tuple[str, str]:
    prefix, direction = mask_mode.rsplit("_", 1)
    flow = {"self": "M1", "incoming": "M2", "outgoing": "M3"}.get(prefix)
    if flow is None or direction not in DIRECTIONS:
        raise ValueError(f"unexpected Stage-4 mask_mode: {mask_mode}")
    return flow, direction


def target_id(source: dict[str, Any]) -> str:
    if source.get("target_scope") == "single_object":
        return f"single_object::{source.get('region')}"
    return "all_objects::all_objects"


def record_key(case: str, seed: int, source: dict[str, Any]) -> tuple[Any, ...]:
    flow, direction = parse_mode(str(source["mask_mode"]))
    return (
        case,
        seed,
        target_id(source),
        flow,
        direction,
        str(source["head_scope"]),
    )


def collect(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_root = root / "stage4_metrics"
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    coverage: dict[str, Any] = {"fast_reports": 0, "trajectory_reports": 0, "survival_reports": 0}

    fast_root = metric_root / "head_scope_baseline_fast"
    for path in sorted(fast_root.glob("*/seed_*/report.json")):
        report = read_json(path)
        coverage["fast_reports"] += 1
        for source in report.get("records", []):
            key = record_key(str(report["case"]), int(report["seed"]), source)
            flow, direction = key[3], key[4]
            manifest_path = Path(source["manifest_path"])
            metrics = source["metrics"]
            categories = metrics["category_scores_0_100"]
            merged[key] = {
                "case": key[0],
                "seed": key[1],
                "target": key[2],
                "flow": flow,
                "direction": direction,
                "scope": key[5],
                "variant_id": str(source["variant_id"]),
                "manifest_path": str(manifest_path),
                "video_path": str(manifest_path.parent / "generated.mp4"),
                "metrics": {
                    "target_local": categories["target_local"],
                    "outside_static": 100.0 * metrics["outside_objects"]["mae_0_1"],
                },
            }

            manifest = read_json(manifest_path)
            dose_path = Path(manifest["dose_metrics"]["path"])
            with np.load(dose_path) as dose:
                for name in (
                    "attention_mass",
                    "attention_mass_query_sum",
                    "removed_value_norm",
                    "removed_value_norm_query_sum",
                    "removed_to_output_ratio",
                ):
                    merged[key]["metrics"][name] = finite_mean(dose[name].ravel())

    trajectory_root = metric_root / "head_scope_trajectory"
    quality = Counter()
    for path in sorted(trajectory_root.glob("*/seed_*/report.json")):
        report = read_json(path)
        coverage["trajectory_reports"] += 1
        for source in report.get("records", []):
            key = record_key(str(report["case"]), int(report["seed"]), source)
            if key not in merged:
                continue
            metrics = source["metrics"]
            selected = list(metrics["selected_objects"])
            selected_set = set(selected)
            objects = metrics["objects"]
            passed = metrics.get("quality_pass") is True
            quality[(key[3], key[4], key[5], "total")] += 1
            quality[(key[3], key[4], key[5], "failed")] += int(not passed)
            pck10 = (
                finite_mean(objects[name].get("pck_normalized", {}).get("0.1") for name in selected)
                if passed
                else None
            )
            other_objects = [value for name, value in objects.items() if name not in selected_set]
            merged[key]["metrics"].update(
                {
                    "center_ade": metrics.get("target_center_ade_norm") if passed else None,
                    "center_fde": (
                        finite_mean(objects[name].get("center_fde_norm") for name in selected)
                        if passed
                        else None
                    ),
                    "velocity": (
                        finite_mean(
                            objects[name].get("velocity_vector_error_norm_per_frame")
                            for name in selected
                        )
                        if passed
                        else None
                    ),
                    "pck10_failure": 100.0 * (1.0 - pck10) if pck10 is not None else None,
                    "track_loss": metrics["target_worst_track_loss_score_0_100"],
                    "other_ade": (
                        finite_mean(value.get("center_ade_norm") for value in other_objects)
                        if source["target_scope"] == "single_object"
                        else None
                    ),
                    "quality_pass": passed,
                }
            )

    for path in sorted(trajectory_root.glob("*/seed_*/object_survival_report.json")):
        report = read_json(path)
        coverage["survival_reports"] += 1
        for source in report.get("records", []):
            key = record_key(str(report["case"]), int(report["seed"]), source)
            if key not in merged:
                continue
            metrics = source["metrics"]
            selected = list(metrics["selected_objects"])
            objects = metrics["objects"]
            merged[key]["metrics"].update(
                {
                    "disappearance": metrics["target_worst_disappearance_score_0_100"],
                    "mask_absence": metrics["target_worst_mask_absence_score_0_100"],
                    "identity_failure": 100.0 * max(objects[name]["identity_failure_rate"] for name in selected),
                    "area_failure": 100.0 * max(objects[name]["area_failure_rate"] for name in selected),
                    "terminal_missing": 100.0 * max(objects[name]["terminal_missing_rate"] for name in selected),
                }
            )

    records = list(merged.values())
    coverage["records"] = len(records)
    coverage["cases"] = sorted({record["case"] for record in records})
    coverage["case_seeds"] = sorted({(record["case"], record["seed"]) for record in records})
    coverage["records_by_case_seed"] = [
        {"case": case, "seed": seed, "records": count}
        for (case, seed), count in sorted(Counter((r["case"], r["seed"]) for r in records).items())
    ]
    coverage["scope_counts"] = dict(sorted(Counter(r["scope"] for r in records).items()))
    coverage["quality_gate"] = [
        {
            "flow": flow,
            "direction": direction,
            "scope": scope,
            "failed": quality[(flow, direction, scope, "failed")],
            "total": quality[(flow, direction, scope, "total")],
            "failure_rate": quality[(flow, direction, scope, "failed")]
            / quality[(flow, direction, scope, "total")],
        }
        for flow in FLOWS
        for direction in DIRECTIONS
        for scope in SCOPES
        if quality[(flow, direction, scope, "total")]
    ]
    return records, coverage


def bootstrap_ci(values: list[float], repetitions: int, rng: np.random.Generator) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [None, None]
    if array.size == 1:
        return [float(array[0]), float(array[0])]
    indices = rng.integers(0, array.size, size=(repetitions, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_contrast(
    records: list[dict[str, Any]],
    metric: str,
    varying: str,
    left: str,
    right: str,
    filters: dict[str, str],
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = [r for r in records if all(r.get(k) == v for k, v in filters.items())]
    pair_fields = ("case", "seed", "target", "flow", "direction", "scope")
    pair_fields = tuple(field for field in pair_fields if field != varying)
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in selected:
        grouped[tuple(record[field] for field in pair_fields)][record[varying]] = record

    units = []
    for key, sides in grouped.items():
        if left not in sides or right not in sides:
            continue
        left_value = sides[left]["metrics"].get(metric)
        right_value = sides[right]["metrics"].get(metric)
        if not finite(left_value) or not finite(right_value):
            continue
        units.append(
            {
                "case": sides[left]["case"],
                "seed": sides[left]["seed"],
                "target": sides[left]["target"],
                "left": float(left_value),
                "right": float(right_value),
                "difference": float(left_value) - float(right_value),
                "left_video": sides[left]["video_path"],
                "right_video": sides[right]["video_path"],
            }
        )
    case_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        case_groups[unit["case"]].append(unit)
    case_rows = []
    for case, rows in sorted(case_groups.items()):
        case_rows.append(
            {
                "case": case,
                "left": float(np.mean([row["left"] for row in rows])),
                "right": float(np.mean([row["right"] for row in rows])),
                "difference": float(np.mean([row["difference"] for row in rows])),
                "units": len(rows),
            }
        )
    differences = [row["difference"] for row in case_rows]
    representatives = sorted(units, key=lambda row: abs(row["difference"]), reverse=True)
    return {
        "metric": metric,
        "varying": varying,
        "left": left,
        "right": right,
        "filters": filters,
        "left_mean": finite_mean(row["left"] for row in case_rows),
        "right_mean": finite_mean(row["right"] for row in case_rows),
        "difference": finite_mean(differences),
        "ratio": (
            finite_mean(row["left"] for row in case_rows)
            / finite_mean(row["right"] for row in case_rows)
            if finite_mean(row["right"] for row in case_rows) not in (None, 0.0)
            else None
        ),
        "case_direction_positive": (
            sum(value > 0 for value in differences) / len(differences)
            if differences
            else None
        ),
        "bootstrap_95_ci": bootstrap_ci(differences, repetitions, rng),
        "case_count": len(case_rows),
        "unit_count": len(units),
        "case_rows": case_rows,
        "largest_absolute_units": representatives[:3],
    }


def same_vs_cross(
    records: list[dict[str, Any]],
    metric: str,
    filters: dict[str, str],
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = [r for r in records if all(r.get(k) == v for k, v in filters.items())]
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in selected:
        key = (record["case"], record["seed"], record["target"], record["flow"], record["scope"])
        grouped[key][record["direction"]] = record
    synthetic = []
    for sides in grouped.values():
        if set(DIRECTIONS) - set(sides):
            continue
        values = {direction: sides[direction]["metrics"].get(metric) for direction in DIRECTIONS}
        if not all(finite(value) for value in values.values()):
            continue
        synthetic.extend(
            [
                {**sides["same"], "direction": "same_value"},
                {
                    **sides["same"],
                    "direction": "cross_mean",
                    "metrics": {metric: 0.5 * (float(values["future"]) + float(values["past"]))},
                    "video_path": f"{sides['future']['video_path']} | {sides['past']['video_path']}",
                },
            ]
        )
    return paired_contrast(
        synthetic,
        metric,
        "direction",
        "same_value",
        "cross_mean",
        {},
        repetitions,
        rng,
    )


def strict_direction_means(
    records: list[dict[str, Any]], metric: str, filters: dict[str, str]
) -> dict[str, dict[str, Any]]:
    selected = [r for r in records if all(r.get(k) == v for k, v in filters.items())]
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in selected:
        key = (record["case"], record["seed"], record["target"], record["flow"], record["scope"])
        grouped[key][record["direction"]] = record
    complete = []
    for sides in grouped.values():
        if set(DIRECTIONS) - set(sides):
            continue
        if not all(finite(sides[d]["metrics"].get(metric)) for d in DIRECTIONS):
            continue
        complete.append(sides)

    output = {}
    for direction in DIRECTIONS:
        by_case: dict[str, list[float]] = defaultdict(list)
        for sides in complete:
            by_case[sides[direction]["case"]].append(
                float(sides[direction]["metrics"][metric])
            )
        case_values = {case: float(np.mean(values)) for case, values in by_case.items()}
        output[direction] = {
            "mean": finite_mean(case_values.values()),
            "case_count": len(case_values),
            "unit_count": len(complete),
            "case_values": case_values,
        }
    return output


def fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if not finite(value) else f"{float(value):.{digits}f}"


def contrast_row(label: str, row: dict[str, Any]) -> str:
    ci = row["bootstrap_95_ci"]
    positive_cases = row["case_direction_positive"]
    positive_text = (
        f"{100.0 * float(positive_cases):.1f}%"
        if finite(positive_cases)
        else "N/A"
    )
    return (
        f"| {label} | {fmt(row['left_mean'])} | {fmt(row['right_mean'])} | "
        f"{fmt(row['difference'])} | {fmt(row['ratio'])}× | "
        f"{positive_text} | "
        f"[{fmt(ci[0])}, {fmt(ci[1])}] | {row['case_count']} / {row['unit_count']} |"
    )


def markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Stage 4 · Existing-case Pilot Analysis",
        "",
        f"Snapshot: `{report['snapshot']}`",
        "",
        f"- Generated/metric-complete variants used: **{coverage['records']}**.",
        f"- Independent cases: **{len(coverage['cases'])}**; case-seed units: **{len(coverage['case_seeds'])}**.",
        "- Aggregation: average targets and seeds within each case, then give each case equal weight.",
        "- All effects are relative to the same-seed Baseline; larger means a stronger intervention effect, not worse GT correctness.",
        "",
        "> Pilot boundary: only three independent cases are available. Bootstrap intervals are descriptive; no p-value, BH-FDR, or population-level mechanism claim is made.",
        "",
        "## Coverage",
        "",
        "| Case | Seed | Variants |",
        "|---|---:|---:|",
    ]
    for row in coverage["records_by_case_seed"]:
        lines.append(f"| {row['case']} | {row['seed']} | {row['records']} |")
    lines.extend(
        [
            "",
            "Scope counts: "
            + ", ".join(f"{SCOPE_LABELS.get(key, key)}={value}" for key, value in coverage["scope_counts"].items())
            + ".",
            "",
            "## Preregistered primary contrasts: Future − Past",
            "",
            "Positive means Future has the larger Baseline-relative effect; negative means Past has the larger effect.",
            "",
            "| Test / metric | Future | Past | Difference | Ratio | Positive cases | Descriptive 95% case-bootstrap CI | Cases / paired units |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in report["primary_contrasts"]:
        lines.append(contrast_row(label, row))

    lines.extend(
        [
            "",
            "## Same-frame contribution versus cross-time mean",
            "",
            "Difference = Same − 0.5 × (Future + Past).",
            "",
            "| Flow / metric | Same | Cross-time mean | Difference | Ratio | Positive cases | Descriptive 95% case-bootstrap CI | Cases / paired units |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in report["same_contrasts"]:
        lines.append(contrast_row(label, row))

    lines.extend(
        [
            "",
            "## Top100 head specificity",
            "",
            "Only exact paired finite cohorts are compared. Positive means Top100 has the larger effect.",
            "",
            "| Flow / direction / metric / contrast | Top100 | Comparator | Difference | Ratio | Positive cases | Descriptive 95% case-bootstrap CI | Cases / paired units |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in report["head_contrasts"]:
        lines.append(contrast_row(label, row))

    lines.extend(
        [
            "",
            "## Top100 dose specificity",
            "",
            "This table checks whether a larger output effect is accompanied by a larger exact removed contribution.",
            "",
            "| Flow / direction / contrast | Top100 | Comparator | Difference | Ratio | Positive cases | Descriptive 95% case-bootstrap CI | Cases / paired units |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in report["dose_head_contrasts"]:
        lines.append(contrast_row(label, row))

    lines.extend(
        [
            "",
            "## Top100 directional means",
            "",
            "| Flow | Metric | Same | Future | Past |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["directional_means"]:
        lines.append(
            f"| {FLOW_LABELS[row['flow']]} | {METRIC_LABELS[row['metric']]} | "
            f"{fmt(row['same']['mean'])} | {fmt(row['future']['mean'])} | {fmt(row['past']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Trajectory quality-gate audit",
            "",
            "Center-ADE/FDE and velocity exclude failed rows; Track Loss and Disappearance retain them.",
            "",
            "| Flow | Direction | Scope | Failed | Total | Failure rate |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in coverage["quality_gate"]:
        lines.append(
            f"| {FLOW_LABELS[row['flow']]} | {row['direction'].title()} | "
            f"{SCOPE_LABELS[row['scope']]} | {row['failed']} | {row['total']} | "
            f"{100.0 * row['failure_rate']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- T1 can currently diagnose whether temporal R→R removal changes Baseline-relative trajectory, identity, or survival; it cannot yet identify a population effect.",
            "- T2's preregistered GT contact/post-contact test is unavailable in the complete current cohort. Velocity here is a Baseline-relative fallback, not a physics-correctness score.",
            "- T3 uses Other-object Center-ADE only for single-object interventions. It is the direct cross-object trajectory outcome; outside-object MAE remains a pixel proxy.",
            "- Center-ADE/FDE describe only trackable survivors. A smaller survivor-only ADE can coexist with more disappearance; always read Track Loss and Disappearance beside it.",
            "- Frozen-ROI MAE mixes position, appearance, shape, and disappearance. The two available complete25 reports are insufficient for a case-balanced appearance conclusion.",
            "- Future and Past delete different query/source counts. Directional output effects must be interpreted together with query-sum dose, not only per-query dose.",
            "- R is the frozen sparse tracked-token tube, not a dense whole-object mask. Knockout establishes necessity of the deleted contribution, not exclusive semantic encoding.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records, coverage = collect(args.root)
    rng = np.random.default_rng(args.bootstrap_seed)

    primary_specs = [
        ("T1 · M1 Top100 · Center-ADE", "center_ade", {"flow": "M1", "scope": "top100"}),
        ("T1 guardrail · Track Loss", "track_loss", {"flow": "M1", "scope": "top100"}),
        ("T1 guardrail · Disappearance", "disappearance", {"flow": "M1", "scope": "top100"}),
        ("T1 dose · Removed AV query-sum", "removed_value_norm_query_sum", {"flow": "M1", "scope": "top100"}),
        ("T2 fallback · M2 Top100 · Velocity", "velocity", {"flow": "M2", "scope": "top100"}),
        ("T2 guardrail · Track Loss", "track_loss", {"flow": "M2", "scope": "top100"}),
        ("T2 guardrail · Disappearance", "disappearance", {"flow": "M2", "scope": "top100"}),
        ("T2 dose · Removed AV query-sum", "removed_value_norm_query_sum", {"flow": "M2", "scope": "top100"}),
        ("T3 · M3 Top100 · Other-object ADE", "other_ade", {"flow": "M3", "scope": "top100"}),
        ("T3 pixel proxy · Outside-object MAE", "outside_static", {"flow": "M3", "scope": "top100"}),
        ("T3 dose · Removed AV query-sum", "removed_value_norm_query_sum", {"flow": "M3", "scope": "top100"}),
    ]
    primary = [
        (
            label,
            paired_contrast(records, metric, "direction", "future", "past", filters, args.bootstrap_repetitions, rng),
        )
        for label, metric, filters in primary_specs
    ]

    same_specs = [
        ("M1 · Center-ADE", "center_ade", {"flow": "M1", "scope": "top100"}),
        ("M1 · Disappearance", "disappearance", {"flow": "M1", "scope": "top100"}),
        ("M2 · Velocity", "velocity", {"flow": "M2", "scope": "top100"}),
        ("M2 · Disappearance", "disappearance", {"flow": "M2", "scope": "top100"}),
        ("M3 · Other-object ADE", "other_ade", {"flow": "M3", "scope": "top100"}),
        ("M3 · Outside-object MAE", "outside_static", {"flow": "M3", "scope": "top100"}),
    ]
    same = [
        (label, same_vs_cross(records, metric, filters, args.bootstrap_repetitions, rng))
        for label, metric, filters in same_specs
    ]

    head_specs = []
    for flow, metric in (("M1", "center_ade"), ("M1", "identity_failure"), ("M1", "disappearance"), ("M2", "velocity"), ("M2", "identity_failure"), ("M2", "disappearance"), ("M3", "other_ade"), ("M3", "outside_static"), ("M3", "identity_failure")):
        for direction in DIRECTIONS:
            for comparator, comparison_label in (("bottom100", "Top−Bottom"), ("random100_layer_matched_draw0", "Top−Random")):
                head_specs.append((f"{flow} {direction.title()} · {METRIC_LABELS[metric]} · {comparison_label}", metric, comparator, {"flow": flow, "direction": direction}))
    head = [
        (
            label,
            paired_contrast(records, metric, "scope", "top100", comparator, filters, args.bootstrap_repetitions, rng),
        )
        for label, metric, comparator, filters in head_specs
    ]

    dose_head = []
    for flow in FLOWS:
        for direction in DIRECTIONS:
            for comparator, comparison_label in (("bottom100", "Top−Bottom"), ("random100_layer_matched_draw0", "Top−Random")):
                dose_head.append(
                    (
                        f"{flow} {direction.title()} · Removed AV query-sum · {comparison_label}",
                        paired_contrast(
                            records,
                            "removed_value_norm_query_sum",
                            "scope",
                            "top100",
                            comparator,
                            {"flow": flow, "direction": direction},
                            args.bootstrap_repetitions,
                            rng,
                        ),
                    )
                )

    directional_metrics = {
        "M1": ("removed_value_norm", "removed_value_norm_query_sum", "center_ade", "track_loss", "identity_failure", "disappearance"),
        "M2": ("removed_value_norm", "removed_value_norm_query_sum", "velocity", "track_loss", "identity_failure", "disappearance"),
        "M3": ("removed_value_norm", "removed_value_norm_query_sum", "other_ade", "outside_static", "identity_failure", "disappearance"),
    }
    directional = []
    for flow, metrics in directional_metrics.items():
        for metric in metrics:
            strict_means = strict_direction_means(
                records, metric, {"flow": flow, "scope": "top100"}
            )
            directional.append(
                {
                    "flow": flow,
                    "metric": metric,
                    **strict_means,
                }
            )

    report = {
        "snapshot": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage,
        "primary_contrasts": primary,
        "same_contrasts": same,
        "head_contrasts": head,
        "dose_head_contrasts": dose_head,
        "directional_means": directional,
        "limitations": {
            "independent_cases": len(coverage["cases"]),
            "formal_inference": False,
            "reason": "Stage 4A pilot has fewer than 8 independent cases and an incomplete head-scope matrix.",
        },
    }
    output_dir = args.root / "stage4_current_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "STAGE4_EXISTING_CASES_ANALYSIS.md").write_text(markdown(report), encoding="utf-8")
    print(f"records={len(records)} cases={len(coverage['cases'])} case_seeds={len(coverage['case_seeds'])}")
    print(output_dir / "STAGE4_EXISTING_CASES_ANALYSIS.md")


if __name__ == "__main__":
    main()
