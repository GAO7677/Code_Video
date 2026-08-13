#!/usr/bin/env python3
"""Aggregate the preregistered GT correspondence-guidance validation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)
MODES = ("region", "point", "combined")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def metric_for_target(path: Path, target: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = load_json(path).get("metrics", [])
    return next((row for row in rows if str(row.get("target")) == target), None)


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def bootstrap_ci(values: list[float], seed: int = 47326) -> list[float] | None:
    if len(values) < 3:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"
    return str(value)


def analyze(root: Path, seed: int, lambdas: tuple[float, ...]) -> dict[str, Any]:
    screen_path = root / "screening" / f"seed_{seed:05d}" / "baseline_eligibility.json"
    screen = load_json(screen_path)
    eligible = [row for row in screen["targets"] if row["eligible"]]
    requested_lambdas = tuple(float(value) for value in lambdas)
    primary_trigger_modes: list[str] | None = None
    if any(not math.isclose(value, 0.1) for value in requested_lambdas):
        # Sensitivity modes are determined exclusively from the frozen primary
        # analysis.  This prevents never-registered variants from appearing as
        # misleading 0/N rows in the final report.
        primary_trigger_modes = list(analyze(root, seed, (0.1,))["trigger_modes"])
    sensitivity_modes = set(primary_trigger_modes or ())
    records: list[dict[str, Any]] = []
    for row in eligible:
        case, target = str(row["case"]), str(row["target"])
        generation_root = root / "generations" / case / f"seed_{seed:05d}"
        baseline = metric_for_target(
            generation_root / "baseline" / "trajectory_metrics.json", target
        )
        if baseline is None:
            raise RuntimeError(f"eligible Baseline metric disappeared: {case}/{target}")
        for guidance_lambda in requested_lambdas:
            modes = (
                MODES
                if math.isclose(guidance_lambda, 0.1)
                else tuple(mode for mode in MODES if mode in sensitivity_modes)
            )
            for mode in modes:
                variant = f"{mode}__{target}__lambda{float_tag(guidance_lambda)}"
                metric_path = generation_root / variant / "trajectory_metrics.json"
                guided = metric_for_target(metric_path, target)
                record: dict[str, Any] = {
                    "case": case,
                    "target": target,
                    "mode": mode,
                    "lambda": guidance_lambda,
                    "variant": variant,
                    "complete": guided is not None,
                    "baseline_quality_pass": bool(baseline["quality_pass"]),
                    "baseline_ade_d0": baseline.get("ade_d0"),
                    "baseline_fde_d0": baseline.get("fde_d0"),
                    "baseline_pck_10pct_d0": baseline.get("pck_10pct_d0"),
                    "baseline_track_loss": baseline["future_track_loss_score_0_100"],
                }
                if guided is not None:
                    guided_pass = bool(guided["quality_pass"])
                    record.update(
                        {
                            "guided_quality_pass": guided_pass,
                            "guided_ade_d0": guided.get("ade_d0"),
                            "guided_fde_d0": guided.get("fde_d0"),
                            "guided_pck_10pct_d0": guided.get("pck_10pct_d0"),
                            "guided_track_loss": guided[
                                "future_track_loss_score_0_100"
                            ],
                            "delta_track_loss": float(
                                guided["future_track_loss_score_0_100"]
                            )
                            - float(baseline["future_track_loss_score_0_100"]),
                            "delta_ade_d0": (
                                float(guided["ade_d0"]) - float(baseline["ade_d0"])
                                if guided_pass
                                and guided.get("ade_d0") is not None
                                and baseline.get("ade_d0") is not None
                                else None
                            ),
                            "delta_fde_d0": (
                                float(guided["fde_d0"]) - float(baseline["fde_d0"])
                                if guided_pass
                                and guided.get("fde_d0") is not None
                                and baseline.get("fde_d0") is not None
                                else None
                            ),
                            "delta_pck_10pct_d0": (
                                float(guided["pck_10pct_d0"])
                                - float(baseline["pck_10pct_d0"])
                                if guided_pass
                                and guided.get("pck_10pct_d0") is not None
                                and baseline.get("pck_10pct_d0") is not None
                                else None
                            ),
                        }
                    )
                records.append(record)

    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(float(record["lambda"]), str(record["mode"]), str(record["case"]))].append(
            record
        )
    case_rows: list[dict[str, Any]] = []
    for (guidance_lambda, mode, case), rows in sorted(grouped.items()):
        complete = all(row["complete"] for row in rows)
        all_gate_pass = complete and all(row.get("guided_quality_pass", False) for row in rows)
        delta_track_loss = mean(
            [float(row["delta_track_loss"]) for row in rows if row.get("complete")]
        )
        delta_ade = (
            mean([float(row["delta_ade_d0"]) for row in rows])
            if all_gate_pass
            else None
        )
        case_rows.append(
            {
                "case": case,
                "mode": mode,
                "lambda": guidance_lambda,
                "registered_target_count": len(rows),
                "completed_target_count": sum(bool(row["complete"]) for row in rows),
                "all_guided_quality_pass": all_gate_pass,
                "mean_delta_ade_d0": delta_ade,
                "mean_delta_track_loss": delta_track_loss,
                "sensitivity_success": bool(
                    all_gate_pass
                    and delta_ade is not None
                    and delta_ade < 0
                    and delta_track_loss is not None
                    and delta_track_loss <= 0
                ),
            }
        )

    aggregate_rows: list[dict[str, Any]] = []
    registered_pairs = sorted(
        {(float(record["lambda"]), str(record["mode"])) for record in records}
    )
    for guidance_lambda, mode in registered_pairs:
        rows = [
            row
            for row in case_rows
            if row["lambda"] == guidance_lambda and row["mode"] == mode
        ]
        completed = [
            record
            for record in records
            if record["lambda"] == guidance_lambda
            and record["mode"] == mode
            and record["complete"]
        ]
        evaluable = [row for row in rows if row["mean_delta_ade_d0"] is not None]
        ade_values = [float(row["mean_delta_ade_d0"]) for row in evaluable]
        track_values = [
            float(row["mean_delta_track_loss"])
            for row in rows
            if row["mean_delta_track_loss"] is not None
        ]
        aggregate_rows.append(
            {
                "mode": mode,
                "lambda": guidance_lambda,
                "eligible_case_count": len(rows),
                "eligible_target_count": len(
                    [
                        record
                        for record in records
                        if record["lambda"] == guidance_lambda
                        and record["mode"] == mode
                    ]
                ),
                "completed_target_count": len(completed),
                "guided_target_gate_pass_count": sum(
                    bool(record.get("guided_quality_pass")) for record in completed
                ),
                "fully_evaluable_case_count": len(evaluable),
                "case_balanced_mean_delta_ade_d0": mean(ade_values),
                "case_bootstrap_95pct_ci_delta_ade_d0": bootstrap_ci(ade_values),
                "case_balanced_mean_delta_track_loss": mean(track_values),
                "improved_case_count": sum(row["sensitivity_success"] for row in rows),
            }
        )

    primary = [row for row in aggregate_rows if math.isclose(row["lambda"], 0.1)]
    trigger_modes = (
        [
            mode
            for mode in MODES
            if any(
                row["mode"] == mode and row["improved_case_count"] >= 2
                for row in primary
            )
        ]
        if primary
        else list(primary_trigger_modes or ())
    )
    return {
        "protocol": "wan_gt_guidance_frozen_validation_v1",
        "seed": seed,
        "screening_report": str(screen_path),
        "eligible_case_count": int(screen["eligible_case_count"]),
        "eligible_target_count": int(screen["eligible_target_count"]),
        "lambdas": list(requested_lambdas),
        "case_is_highest_independent_unit": True,
        "missing_ADE_policy": (
            "A case is not ADE-evaluable if any preregistered target fails the guided "
            "trajectory gate; Track Loss remains included."
        ),
        "sensitivity_trigger": (
            "At lambda=0.1, at least two independent cases must have all targets "
            "gated, mean delta ADE<0, and mean delta Track Loss<=0."
        ),
        "trigger_modes": trigger_modes,
        "records": records,
        "case_results": case_rows,
        "aggregate": aggregate_rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Frozen GT Correspondence-Guidance Validation",
        "",
        f"- Seed: **{report['seed']}**",
        f"- Eligible cases/targets: **{report['eligible_case_count']} / {report['eligible_target_count']}**",
        f"- Sensitivity-trigger modes: **{', '.join(report['trigger_modes']) or 'none'}**",
        "- Primary metric: future-only ADE/D0 versus source GT; negative delta is improvement.",
        "- Guardrail: Future Track Loss; positive delta is worse observability.",
        "- Case is the highest independent unit; missing guided tracks are never dropped as successes.",
        "",
        "## Aggregate",
        "",
        "| λ | Mode | Complete targets | Gate-pass targets | Evaluable cases | ΔADE/D0 mean [95% CI] | ΔTrack Loss | Improvement cases |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        ci = row["case_bootstrap_95pct_ci_delta_ade_d0"]
        ade = fmt(row["case_balanced_mean_delta_ade_d0"])
        ade_ci = ade if ci is None else f"{ade} [{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(
            f"| {fmt(row['lambda'], 2)} | {row['mode']} | "
            f"{row['completed_target_count']}/{row['eligible_target_count']} | "
            f"{row['guided_target_gate_pass_count']} | {row['fully_evaluable_case_count']} | "
            f"{ade_ci} | {fmt(row['case_balanced_mean_delta_track_loss'])} | "
            f"{row['improved_case_count']} |"
        )
    lines.extend(
        [
            "",
            "## Case-level primary results",
            "",
            "A case counts as an improvement only when all of its preregistered targets pass the guided gate, case-mean ΔADE/D0 < 0, and case-mean ΔTrack Loss <= 0.",
            "",
            "| λ | Mode | Case | Targets complete | All gated | Mean ΔADE/D0 | Mean ΔTrack Loss | Trigger success |",
            "|---:|---|---|---:|---|---:|---:|---|",
        ]
    )
    for row in report["case_results"]:
        lines.append(
            f"| {fmt(row['lambda'], 2)} | {row['mode']} | {row['case']} | "
            f"{row['completed_target_count']}/{row['registered_target_count']} | "
            f"{fmt(row['all_guided_quality_pass'])} | {fmt(row['mean_delta_ade_d0'])} | "
            f"{fmt(row['mean_delta_track_loss'])} | {fmt(row['sensitivity_success'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- ΔADE/FDE/PCK are defined only when Baseline and guided variants pass the future-track gate.",
            "- A failed guided gate is a destructive outcome, not missing-at-random data; it remains visible through Track Loss.",
            "- Bootstrap intervals are descriptive case-resampling intervals; no significance or FDR claim is made.",
            "- Sensitivity is conditional on the frozen trigger above and is not an unrestricted hyperparameter search.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--lambdas", type=float, nargs="+", default=(0.1,))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    report = analyze(root, args.seed, tuple(args.lambdas))
    output = root / "final_analysis" / f"seed_{args.seed:05d}"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "frozen_validation_report.json"
    md_path = output / "FROZEN_VALIDATION_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "trigger_modes": report["trigger_modes"],
            },
            indent=2,
        )
    )
    if args.strict:
        primary = [row for row in report["aggregate"] if row["lambda"] == 0.1]
        incomplete = [
            row
            for row in primary
            if row["completed_target_count"] != row["eligible_target_count"]
        ]
        if incomplete:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
