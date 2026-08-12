#!/usr/bin/env python3
"""Strictly paired interim analysis of fast Stage-3 Baseline-relative outcomes.

The current fast metrics are pixel/ROI effects.  They are useful for target-local
change, global appearance, and outside-object spillover, but they are deliberately
not labelled as trajectory, identity, shape, or physical-correctness metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


DEFAULT_INPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage3_metrics/head_scope_baseline_fast"
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
    "random100_layer_matched_draw0": "Random100",
    "all720": "All720",
}
FLOW_LABELS = {
    "self_only": "M1 · R→R",
    "incoming_only": "M2 · C→R",
    "outgoing_only": "M3 · R→C",
}
METRICS = (
    "target_local",
    "outside_static",
    "outside_spillover",
    "global_appearance",
    "temporal_appearance",
    "impact_composite",
)
METRIC_LABELS = {
    "target_local": "Target frozen-ROI MAE ×100",
    "outside_static": "Outside-object MAE ×100",
    "outside_spillover": "Outside static/delta composite ×100",
    "global_appearance": "Global appearance composite ×100",
    "temporal_appearance": "Temporal pixel-change composite ×100",
    "impact_composite": "Legacy fast impact composite (0–100)",
}
PRIMARY_METRICS = ("target_local", "outside_static", "global_appearance")
CONTRASTS = (
    ("top100", "bottom100", "Top100 − Bottom100"),
    ("top100", "random100_layer_matched_draw0", "Top100 − Random100"),
    ("all720", "top100", "All720 − Top100"),
)
STRATA = ("single_object", "all_objects", "all_targets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def collect_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(root.glob("*/seed_*/report.json")):
        report = read_json(path)
        for source in report.get("records", []):
            scope = str(source.get("head_scope") or "")
            flow = str(source.get("mask_mode") or "")
            if scope not in SCOPES or flow not in FLOWS:
                continue
            key = (str(source["case"]), int(source["seed"]), str(source["variant_id"]))
            if key in seen:
                continue
            seen.add(key)
            metrics = source["metrics"]
            categories = metrics["category_scores_0_100"]
            target = (
                f"single_object::{source['region']}"
                if source["target_scope"] == "single_object"
                else "all_objects::all_objects"
            )
            records.append(
                {
                    "case": str(source["case"]),
                    "seed": int(source["seed"]),
                    "target_scope": str(source["target_scope"]),
                    "region": str(source["region"]),
                    "target": target,
                    "flow": flow,
                    "scope": scope,
                    "variant_id": str(source["variant_id"]),
                    "video_path": str(source["path"]),
                    "target_local": float(categories["target_local"]),
                    "outside_static": 100.0 * float(metrics["outside_objects"]["mae_0_1"]),
                    "outside_spillover": float(categories["outside_spillover"]),
                    "global_appearance": float(categories["global_appearance"]),
                    "temporal_appearance": float(categories["temporal_appearance"]),
                    "impact_composite": float(metrics["impact_score_0_100"]),
                }
            )
    return records


def grouped(
    rows: Iterable[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def strict_scope_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for key, current in grouped(rows, ("case", "seed", "target", "flow")).items():
        mapping = {row["scope"]: row for row in current}
        if set(mapping) != set(SCOPES):
            continue
        case, seed, target, flow = key
        pairs.append(
            {
                "case": case,
                "seed": seed,
                "target": target,
                "target_scope": mapping[SCOPES[0]]["target_scope"],
                "region": mapping[SCOPES[0]]["region"],
                "flow": flow,
                "scopes": mapping,
            }
        )
    return pairs


def strict_flow_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for key, current in grouped(rows, ("case", "seed", "target", "scope")).items():
        mapping = {row["flow"]: row for row in current}
        if set(mapping) != set(FLOWS):
            continue
        case, seed, target, scope = key
        pairs.append(
            {
                "case": case,
                "seed": seed,
                "target": target,
                "target_scope": mapping[FLOWS[0]]["target_scope"],
                "region": mapping[FLOWS[0]]["region"],
                "scope": scope,
                "flows": mapping,
            }
        )
    return pairs


def in_stratum(row: dict[str, Any], stratum: str) -> bool:
    return stratum == "all_targets" or row["target_scope"] == stratum


def case_values(
    rows: Iterable[dict[str, Any]], value: Callable[[dict[str, Any]], float]
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        current = float(value(row))
        if np.isfinite(current):
            values[str(row["case"])].append(current)
    return {case: float(np.mean(current)) for case, current in values.items() if current}


def bootstrap(
    values: dict[str, float], repetitions: int, rng: np.random.Generator
) -> list[float | None]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    if not array.size:
        return [None, None]
    if array.size == 1:
        return [float(array[0]), float(array[0])]
    samples = rng.integers(0, array.size, size=(repetitions, array.size))
    distribution = array[samples].mean(axis=1)
    return [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))]


def mean(values: dict[str, float]) -> float | None:
    return float(np.mean(list(values.values()))) if values else None


def scope_summaries(
    pairs: list[dict[str, Any]], repetitions: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    output = []
    for stratum in STRATA:
        for flow in FLOWS:
            subset = [row for row in pairs if in_stratum(row, stratum) and row["flow"] == flow]
            for metric in METRICS:
                for scope in SCOPES:
                    cases = case_values(subset, lambda row, s=scope, m=metric: row["scopes"][s][m])
                    output.append(
                        {
                            "stratum": stratum,
                            "flow": flow,
                            "scope": scope,
                            "metric": metric,
                            "case_balanced_mean": mean(cases),
                            "case_bootstrap_ci95": bootstrap(cases, repetitions, rng),
                            "case_count": len(cases),
                            "paired_unit_count": len(subset),
                            "case_means": cases,
                        }
                    )
    return output


def scope_contrasts(
    pairs: list[dict[str, Any]], repetitions: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    output = []
    for stratum in STRATA:
        for flow in FLOWS:
            subset = [row for row in pairs if in_stratum(row, stratum) and row["flow"] == flow]
            for metric in METRICS:
                for left, right, label in CONTRASTS:
                    differences = case_values(
                        subset,
                        lambda row, a=left, b=right, m=metric: row["scopes"][a][m] - row["scopes"][b][m],
                    )
                    left_values = case_values(subset, lambda row, s=left, m=metric: row["scopes"][s][m])
                    right_values = case_values(subset, lambda row, s=right, m=metric: row["scopes"][s][m])
                    left_mean, right_mean = mean(left_values), mean(right_values)
                    ratio = (
                        left_mean / right_mean
                        if left_mean is not None and right_mean is not None and abs(right_mean) > 1e-12
                        else None
                    )
                    case_wins = [
                        case for case in differences if differences[case] > 0
                    ]
                    output.append(
                        {
                            "stratum": stratum,
                            "flow": flow,
                            "metric": metric,
                            "left": left,
                            "right": right,
                            "label": label,
                            "case_balanced_difference": mean(differences),
                            "case_bootstrap_ci95": bootstrap(differences, repetitions, rng),
                            "ratio_of_case_balanced_means": ratio,
                            "case_positive_fraction": len(case_wins) / len(differences) if differences else None,
                            "case_count": len(differences),
                            "paired_unit_count": len(subset),
                            "case_differences": differences,
                        }
                    )
    return output


def flow_contrasts(
    pairs: list[dict[str, Any]], repetitions: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    comparisons = (
        ("self_only", "incoming_only", "M1 − M2"),
        ("self_only", "outgoing_only", "M1 − M3"),
        ("incoming_only", "outgoing_only", "M2 − M3"),
    )
    output = []
    for stratum in STRATA:
        for scope in SCOPES:
            subset = [row for row in pairs if in_stratum(row, stratum) and row["scope"] == scope]
            for metric in METRICS:
                for left, right, label in comparisons:
                    differences = case_values(
                        subset,
                        lambda row, a=left, b=right, m=metric: row["flows"][a][m] - row["flows"][b][m],
                    )
                    output.append(
                        {
                            "stratum": stratum,
                            "scope": scope,
                            "metric": metric,
                            "left": left,
                            "right": right,
                            "label": label,
                            "case_balanced_difference": mean(differences),
                            "case_bootstrap_ci95": bootstrap(differences, repetitions, rng),
                            "case_positive_fraction": sum(v > 0 for v in differences.values()) / len(differences) if differences else None,
                            "case_count": len(differences),
                            "paired_unit_count": len(subset),
                            "case_differences": differences,
                        }
                    )
    return output


def representative_rankings(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for metric in PRIMARY_METRICS:
        for flow in FLOWS:
            for contrast_left, contrast_right, label in CONTRASTS:
                ranked = sorted(
                    (
                        {
                            "metric": metric,
                            "flow": flow,
                            "contrast": label,
                            "case": row["case"],
                            "seed": row["seed"],
                            "target": row["target"],
                            "difference": row["scopes"][contrast_left][metric] - row["scopes"][contrast_right][metric],
                            "left_value": row["scopes"][contrast_left][metric],
                            "right_value": row["scopes"][contrast_right][metric],
                            "left_video": row["scopes"][contrast_left]["video_path"],
                            "right_video": row["scopes"][contrast_right]["video_path"],
                        }
                        for row in pairs
                        if row["flow"] == flow
                    ),
                    key=lambda row: abs(float(row["difference"])),
                    reverse=True,
                )
                output.extend(ranked[:5])
    return output


def f(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def markdown(report: dict[str, Any]) -> str:
    summaries = {
        (row["stratum"], row["flow"], row["metric"], row["scope"]): row
        for row in report["scope_summaries"]
    }
    contrasts = {
        (row["stratum"], row["flow"], row["metric"], row["label"]): row
        for row in report["scope_contrasts"]
    }
    lines = [
        "# Stage 3 Fast-Outcome Interim Analysis",
        "",
        f"Snapshot: `{report['generated_at_utc']}`",
        "",
        f"- Fast-metric records: **{report['record_count']} / 1188**.",
        f"- Reports: **{report['report_count']} case-seed**, from **{report['case_count_any_record']} cases**.",
        f"- Strict four-scope pairs: **{report['strict_scope_pair_count']}**, from **{report['strict_scope_case_count']} cases**.",
        f"- Strict three-flow pairs: **{report['strict_flow_pair_count']}**, from **{report['strict_flow_case_count']} cases**.",
        "",
        "> Interim descriptive analysis only. The matrix is incomplete. Case is the highest independent unit; seeds and object targets are averaged inside case before cases receive equal weight.",
        "",
        "## What these fast metrics can and cannot answer",
        "",
        "- `Target frozen-ROI MAE` detects change near the Baseline object tube, but mixes displacement, deformation, identity/texture change, disappearance, and occlusion. It is **not trajectory ADE**.",
        "- `Outside-object MAE` is a fast background/other-region spillover proxy. It does not separate another object's motion from background appearance.",
        "- `Temporal pixel-change` mixes motion, appearance, deformation, and flicker and is **not used as a trajectory conclusion**.",
        "- Definitive motion/appearance claims require CoTracker center-ADE/FDE/PCK/velocity, retention, center-aligned LPIPS/DINO/shape, and other-object/outside-background metrics.",
        "",
        "## Strict-pair case-balanced means — all targets",
        "",
    ]
    for metric in PRIMARY_METRICS:
        lines += [f"### {METRIC_LABELS[metric]}", ""]
        lines += [
            "| Flow | Top100 | Bottom100 | Random100 | All720 | Cases | Paired units |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for flow in FLOWS:
            rows = [summaries[("all_targets", flow, metric, scope)] for scope in SCOPES]
            lines.append(
                f"| {FLOW_LABELS[flow]} | "
                + " | ".join(f(row["case_balanced_mean"]) for row in rows)
                + f" | {rows[0]['case_count']} | {rows[0]['paired_unit_count']} |"
            )
        lines.append("")
    lines += ["## Primary head-group contrasts — all targets", ""]
    lines += [
        "| Metric | Flow | Contrast | Difference | Ratio | Case direction consistency | 95% case-bootstrap CI |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for metric in PRIMARY_METRICS:
        for flow in FLOWS:
            for label in ("Top100 − Bottom100", "Top100 − Random100", "All720 − Top100"):
                row = contrasts[("all_targets", flow, metric, label)]
                ci = row["case_bootstrap_ci95"]
                lines.append(
                    f"| {METRIC_LABELS[metric]} | {FLOW_LABELS[flow]} | {label} | "
                    f"{f(row['case_balanced_difference'])} | {f(row['ratio_of_case_balanced_means'])}× | "
                    f"{f(100.0 * row['case_positive_fraction'], 1)}% | [{f(ci[0])}, {f(ci[1])}] |"
                )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Ratios compare generated-video effects and therefore may compare All720 with 100-head groups; unlike the dose table, these are total intervention outcomes.",
        "- A larger Baseline-relative effect means stronger necessity under the knockout, not better or worse physical quality.",
        "- Incomplete generation can bias an interim snapshot even after strict pairing; confidence intervals are descriptive and no BH-FDR/significance decision is made yet.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if str(args.output_dir).startswith("/home/gaoya/"):
        raise ValueError("analysis artifacts must be written under /data/gaoya")
    records = collect_records(args.input_root)
    scope_pairs = strict_scope_pairs(records)
    flow_pairs = strict_flow_pairs(records)
    rng = np.random.default_rng(args.bootstrap_seed)
    report = {
        "schema_version": 1,
        "status": "interim_descriptive_incomplete_matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "expected_record_count": 1188,
        "report_count": len(list(args.input_root.glob("*/seed_*/report.json"))),
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
        "metric_labels": METRIC_LABELS,
        "scope_summaries": scope_summaries(scope_pairs, args.bootstrap_repetitions, rng),
        "scope_contrasts": scope_contrasts(scope_pairs, args.bootstrap_repetitions, rng),
        "flow_contrasts": flow_contrasts(flow_pairs, args.bootstrap_repetitions, rng),
        "representatives": representative_rankings(scope_pairs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "fast_outcomes_report.json", report)
    write_csv(args.output_dir / "fast_outcomes_records.csv", records)
    (args.output_dir / "FAST_OUTCOMES.md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "records": len(records),
                "reports": report["report_count"],
                "strict_scope_pairs": len(scope_pairs),
                "strict_scope_cases": report["strict_scope_case_count"],
                "strict_flow_pairs": len(flow_pairs),
                "strict_flow_cases": report["strict_flow_case_count"],
                "output": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
