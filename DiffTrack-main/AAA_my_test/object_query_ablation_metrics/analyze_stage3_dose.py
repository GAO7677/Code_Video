#!/usr/bin/env python3
"""Interim/final paired analysis of Stage-3 attention-dose artifacts.

This analyzes what the intervention actually removed.  It intentionally does
not treat dose as a trajectory, appearance, spillover, or generation-quality
outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_INPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage3_discovery_videos"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage3_interim_analysis"
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
    "random100_layer_matched_draw0": "Random100-layer-matched",
    "all720": "All720",
}
FLOW_LABELS = {
    "self_only": "M1 · R→R",
    "incoming_only": "M2 · C→R",
    "outgoing_only": "M3 · R→C",
}
METRICS = (
    "attention_mass",
    "removed_value_norm",
    "removed_to_output_ratio",
)
METRIC_LABELS = {
    "attention_mass": "removed attention mass / selected head",
    "removed_value_norm": "removed AV norm / selected head",
    "removed_to_output_ratio": "removed AV norm / original head-output norm",
}
CONTRASTS = (
    ("top100", "bottom100", "Top100 − Bottom100"),
    ("top100", "random100_layer_matched_draw0", "Top100 − Random100"),
    ("all720", "top100", "All720 − Top100 (per-head dose)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(finite.mean()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p95": float(np.quantile(finite, 0.95)) if finite.size else None,
        "finite_count": int(finite.size),
    }


def collect_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for complete in sorted(root.rglob("complete.json")):
        directory = complete.parent
        try:
            manifest = read_json(directory / "manifest.json")
            scope = str(manifest.get("head_scope") or "")
            flow = str(manifest.get("mask_mode") or "")
            if scope not in SCOPES or flow not in FLOWS:
                continue
            dose_path = directory / "dose_metrics.npz"
            if not dose_path.is_file():
                raise FileNotFoundError(dose_path)
            with np.load(dose_path, allow_pickle=False) as payload:
                summaries = {
                    metric: finite_summary(payload[metric]) for metric in METRICS
                }
                query_count = finite_summary(payload["target_query_count"])
            head_count = int(
                manifest.get("selected_head_count")
                or manifest.get("top_n")
                or 0
            )
            expected_events = 40 * 2 * head_count
            actual_events = int(summaries["attention_mass"]["finite_count"] or 0)
            if actual_events != expected_events:
                raise RuntimeError(
                    f"dose events {actual_events} != expected {expected_events}"
                )
            target = (
                f"single_object::{manifest.get('region')}"
                if manifest.get("target_scope") == "single_object"
                else "all_objects::"
            )
            records.append(
                {
                    "case": str(manifest["case"]),
                    "seed": int(manifest["seed"]),
                    "target": target,
                    "flow": flow,
                    "scope": scope,
                    "head_count": head_count,
                    "variant_id": str(manifest["variant_id"]),
                    "target_query_count_mean": query_count["mean"],
                    **{
                        metric: summaries[metric]["mean"] for metric in METRICS
                    },
                    "dose_finite_events": actual_events,
                    "manifest_path": str((directory / "manifest.json").resolve()),
                    "video_path": str((directory / "generated.mp4").resolve()),
                    "dose_path": str(dose_path.resolve()),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            failures.append({"directory": str(directory), "reason": str(exc)})
    return records, failures


def group_records(
    records: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in fields)].append(record)
    return grouped


def strict_scope_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key, rows in group_records(
        records, ("case", "seed", "target", "flow")
    ).items():
        mapping = {row["scope"]: row for row in rows}
        if set(mapping) != set(SCOPES):
            continue
        case, seed, target, flow = key
        result.append(
            {
                "case": case,
                "seed": seed,
                "target": target,
                "flow": flow,
                "scopes": mapping,
            }
        )
    return result


def strict_flow_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for key, rows in group_records(
        records, ("case", "seed", "target", "scope")
    ).items():
        mapping = {row["flow"]: row for row in rows}
        if set(mapping) != set(FLOWS):
            continue
        case, seed, target, scope = key
        result.append(
            {
                "case": case,
                "seed": seed,
                "target": target,
                "scope": scope,
                "flows": mapping,
            }
        )
    return result


def case_balanced(
    rows: Iterable[dict[str, Any]], value
) -> tuple[dict[str, float], float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        current = value(row)
        if current is not None and np.isfinite(float(current)):
            values[str(row["case"])].append(float(current))
    means = {
        case: float(np.mean(current)) for case, current in values.items() if current
    }
    return means, float(np.mean(list(means.values()))) if means else float("nan")


def bootstrap_cases(
    case_values: dict[str, float], repetitions: int, rng: np.random.Generator
) -> list[float | None]:
    values = np.asarray(list(case_values.values()), dtype=np.float64)
    if not values.size:
        return [None, None]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    distribution = values[indices].mean(axis=1)
    return [
        float(np.quantile(distribution, 0.025)),
        float(np.quantile(distribution, 0.975)),
    ]


def summarize_scopes(
    pairs: list[dict[str, Any]], repetitions: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    summaries = []
    for flow in FLOWS:
        subset = [row for row in pairs if row["flow"] == flow]
        for metric in METRICS:
            for scope in SCOPES:
                cases, mean = case_balanced(
                    subset, lambda row, s=scope, m=metric: row["scopes"][s][m]
                )
                summaries.append(
                    {
                        "flow": flow,
                        "scope": scope,
                        "metric": metric,
                        "case_balanced_mean": mean,
                        "case_bootstrap_ci95": bootstrap_cases(cases, repetitions, rng),
                        "case_count": len(cases),
                        "paired_unit_count": len(subset),
                        "case_means": cases,
                    }
                )
    return summaries


def summarize_scope_contrasts(
    pairs: list[dict[str, Any]], repetitions: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    summaries = []
    for flow in FLOWS:
        subset = [row for row in pairs if row["flow"] == flow]
        for metric in METRICS:
            for left, right, label in CONTRASTS:
                cases, mean = case_balanced(
                    subset,
                    lambda row, a=left, b=right, m=metric: (
                        row["scopes"][a][m] - row["scopes"][b][m]
                    ),
                )
                left_cases, left_mean = case_balanced(
                    subset, lambda row, s=left, m=metric: row["scopes"][s][m]
                )
                right_cases, right_mean = case_balanced(
                    subset, lambda row, s=right, m=metric: row["scopes"][s][m]
                )
                ratio = (
                    left_mean / right_mean
                    if np.isfinite(left_mean)
                    and np.isfinite(right_mean)
                    and abs(right_mean) > 1e-12
                    else None
                )
                summaries.append(
                    {
                        "flow": flow,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "label": label,
                        "case_balanced_difference": mean,
                        "case_bootstrap_ci95": bootstrap_cases(cases, repetitions, rng),
                        "ratio_of_case_balanced_means": ratio,
                        "case_count": len(cases),
                        "paired_unit_count": len(subset),
                        "case_differences": cases,
                        "left_case_count": len(left_cases),
                        "right_case_count": len(right_cases),
                    }
                )
    return summaries


def summarize_flows(
    pairs: list[dict[str, Any]], repetitions: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    comparisons = (
        ("self_only", "incoming_only", "M1 − M2"),
        ("self_only", "outgoing_only", "M1 − M3"),
        ("incoming_only", "outgoing_only", "M2 − M3"),
    )
    summaries = []
    for scope in SCOPES:
        subset = [row for row in pairs if row["scope"] == scope]
        for metric in METRICS:
            for left, right, label in comparisons:
                cases, mean = case_balanced(
                    subset,
                    lambda row, a=left, b=right, m=metric: (
                        row["flows"][a][m] - row["flows"][b][m]
                    ),
                )
                summaries.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "label": label,
                        "case_balanced_difference": mean,
                        "case_bootstrap_ci95": bootstrap_cases(cases, repetitions, rng),
                        "case_count": len(cases),
                        "paired_unit_count": len(subset),
                        "case_differences": cases,
                    }
                )
    return summaries


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(records)


def f(value: Any, digits: int = 5) -> str:
    return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def markdown(report: dict[str, Any]) -> str:
    complete = report["record_count"] == report["expected_record_count"] and not report["failures"]
    lines = [
        "# Stage 3 Attention-Dose Analysis",
        "",
        f"Snapshot: `{report['generated_at_utc']}`.",
        "",
        f"- Completed records: **{report['record_count']} / 1188**.",
        f"- Strict four-scope paired units: **{report['strict_scope_pair_count']}**; cases: **{report['strict_scope_case_count']}**.",
        f"- Strict three-flow paired units: **{report['strict_flow_pair_count']}**; cases: **{report['strict_flow_case_count']}**.",
        f"- Invalid/missing dose artifacts: **{len(report['failures'])}**.",
        "",
        (
            "> The 1188-record dose matrix is complete. Dose is an intervention exposure, not an outcome; outcome tests and BH-FDR are in `../stage3_final_analysis/STAGE3_FINAL_REPORT.md`."
            if complete
            else "> Interim descriptive analysis only. Generation is incomplete, so no significance decision, BH-FDR result, early stopping, or confirmatory claim is made."
        ),
        "",
        "## What dose means",
        "",
        "`attention_mass` is the selected source probability mass per selected head. `removed_value_norm` is the per-head L2 norm of the exact removed `Σ A[q,k]V[k]` term. `removed_to_output_ratio` divides that norm by the same head's original attention-output norm. These are on-intervention-path exposure measures, not trajectory, appearance, background, survival, or quality outcomes.",
        "",
        "All720 values below are **per-head** averages. They do not include the 7.2× head-count dose and must not be read as total intervention strength.",
        "",
        "## Case-balanced scope summaries",
        "",
    ]
    lookup = {
        (row["flow"], row["metric"], row["scope"]): row
        for row in report["scope_summaries"]
    }
    for metric in METRICS:
        lines += [f"### {METRIC_LABELS[metric]}", ""]
        lines.append("| Flow | Top100 | Bottom100 | Random100 | All720 | Cases | Paired units |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for flow in FLOWS:
            rows = [lookup[(flow, metric, scope)] for scope in SCOPES]
            lines.append(
                f"| {FLOW_LABELS[flow]} | "
                + " | ".join(f(row["case_balanced_mean"]) for row in rows)
                + f" | {rows[0]['case_count']} | {rows[0]['paired_unit_count']} |"
            )
        lines.append("")
    lines += ["## Strict paired head-scope contrasts", ""]
    lines.append("| Metric | Flow | Contrast | Difference | 95% case-bootstrap CI | Ratio | Cases |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in report["scope_contrasts"]:
        ci = row["case_bootstrap_ci95"]
        lines.append(
            f"| {METRIC_LABELS[row['metric']]} | {FLOW_LABELS[row['flow']]} | {row['label']} | "
            f"{f(row['case_balanced_difference'])} | [{f(ci[0])}, {f(ci[1])}] | "
            f"{f(row['ratio_of_case_balanced_means'], 3)}× | {row['case_count']} |"
        )
    lines += [
        "",
        "## Required outcome analysis after metric extraction",
        "",
        "1. Target trajectory: Center-ADE/FDE, PCK, velocity; survival failures remain separate rather than imputed as zero.",
        "2. Center-aligned object appearance: LPIPS/DINO/shape, so displacement is not mislabeled as appearance change.",
        "3. Propagation: other-object trajectory plus outside-object appearance/background metrics, especially for M3.",
        "4. Sanity: object retention/disappearance, full-frame corruption, flicker/smoothness.",
        "5. Statistics: same strict units; first average seeds/targets inside case, then case-cluster bootstrap; report Top−Bottom and Top−Random with absolute effects and CI, and apply BH-FDR only after the frozen complete matrix is analyzed.",
        "",
        "## Interpretation limits",
        "",
        "- Dose is recorded along each ablated generation path. Later steps already contain upstream effects of that intervention, so M1-dose and M2-dose from different videos are not complementary Baseline quantities.",
        "- Knockout establishes necessity under this intervention, not semantic sufficiency. Assigning 'motion' or 'appearance' to a flow requires outcome metrics and, later, Baseline-message probes/rescue.",
        "- These data remain exploratory because cases overlap ranking/inspection; completeness does not make them held-out confirmation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap-repetitions must be at least 100")
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if str(args.output_dir).startswith("/home/gaoya/"):
        raise ValueError("analysis artifacts must be written under /data/gaoya")
    records, failures = collect_records(args.input_root)
    scope_pairs = strict_scope_pairs(records)
    flow_pairs = strict_flow_pairs(records)
    rng = np.random.default_rng(args.bootstrap_seed)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete_stage3_discovery_dose_matrix"
            if len(records) == 1188 and not failures
            else "interim_descriptive_incomplete_matrix"
        ),
        "input_root": str(args.input_root),
        "record_count": len(records),
        "expected_record_count": 1188,
        "record_fraction": len(records) / 1188,
        "case_count_any_record": len({row["case"] for row in records}),
        "strict_scope_pair_count": len(scope_pairs),
        "strict_scope_case_count": len({row["case"] for row in scope_pairs}),
        "strict_flow_pair_count": len(flow_pairs),
        "strict_flow_case_count": len({row["case"] for row in flow_pairs}),
        "bootstrap": {
            "highest_independent_unit": "case",
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
        },
        "dose_is_outcome_metric": False,
        "all720_is_per_head_not_total_dose": True,
        "failures": failures,
        "scope_summaries": summarize_scopes(
            scope_pairs, args.bootstrap_repetitions, rng
        ),
        "scope_contrasts": summarize_scope_contrasts(
            scope_pairs, args.bootstrap_repetitions, rng
        ),
        "flow_contrasts": summarize_flows(
            flow_pairs, args.bootstrap_repetitions, rng
        ),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "report.json", report)
    write_records(args.output_dir / "records.csv", records)
    (args.output_dir / "README.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "strict_scope_pairs": len(scope_pairs),
                "strict_scope_cases": report["strict_scope_case_count"],
                "strict_flow_pairs": len(flow_pairs),
                "strict_flow_cases": report["strict_flow_case_count"],
                "failures": len(failures),
                "output": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
