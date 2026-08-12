#!/usr/bin/env python3
"""Audit numerical completeness of frozen GT-STC guided-generation manifests."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def manifest_row(
    root: Path,
    case: str,
    target: str,
    mode: str,
    guidance_lambda: float,
    seed: int,
) -> dict[str, Any]:
    variant = f"{mode}__{target}__lambda{float_tag(guidance_lambda)}"
    directory = root / "generations" / case / f"seed_{seed:05d}" / variant
    path = directory / "manifest.json"
    complete_marker = directory / "complete.json"
    row: dict[str, Any] = {
        "case": case,
        "target": target,
        "mode": mode,
        "lambda": guidance_lambda,
        "variant": variant,
        "complete": complete_marker.is_file() and path.is_file(),
        "manifest": str(path),
    }
    if not row["complete"]:
        return row
    payload = load_json(path)
    audit = [step for step in payload.get("audit", []) if step.get("guided")]
    losses = [float(step["loss"]) for step in audit if step.get("loss") is not None]
    gradient_rms = [
        float(step["raw_gradient_rms"])
        for step in audit
        if step.get("raw_gradient_rms") is not None
    ]
    expected_steps = int(payload.get("denoising_steps", 40))
    finite = all(math.isfinite(value) for value in losses + gradient_rms)
    row.update(
        {
            "expected_steps": expected_steps,
            "audited_steps": len(audit),
            "all_steps_present": len(audit) == expected_steps,
            "all_loss_and_gradient_values_finite": finite,
            "first_loss": losses[0] if losses else None,
            "minimum_loss": min(losses) if losses else None,
            "final_loss": losses[-1] if losses else None,
            "mean_raw_gradient_rms": mean(gradient_rms),
            "component_loss_keys": sorted(
                {
                    key
                    for step in audit
                    for key in (step.get("component_losses") or {}).keys()
                }
            ),
        }
    )
    return row


def audit(root: Path, seed: int, lambdas: tuple[float, ...]) -> dict[str, Any]:
    screening = load_json(
        root / "screening" / f"seed_{seed:05d}" / "baseline_eligibility.json"
    )
    final_report_path = (
        root / "final_analysis" / f"seed_{seed:05d}" / "frozen_validation_report.json"
    )
    final_report = load_json(final_report_path) if final_report_path.is_file() else {}
    trigger_modes = set(final_report.get("trigger_modes", []))
    rows = []
    for job in screening.get("eligible_jobs", []):
        for target in job["targets"]:
            for guidance_lambda in lambdas:
                modes = (
                    ("region", "point", "combined")
                    if guidance_lambda == 0.1
                    else tuple(mode for mode in ("region", "point") if mode in trigger_modes)
                )
                rows.extend(
                    manifest_row(
                        root,
                        str(job["case"]),
                        str(target),
                        mode,
                        guidance_lambda,
                        seed,
                    )
                    for mode in modes
                )
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["lambda"]), str(row["mode"]))].append(row)
    aggregate = []
    for (guidance_lambda, mode), group in sorted(groups.items()):
        complete = [row for row in group if row["complete"]]
        aggregate.append(
            {
                "lambda": guidance_lambda,
                "mode": mode,
                "expected_variants": len(group),
                "complete_variants": len(complete),
                "all_complete_variants_have_40_steps": bool(complete)
                and all(row["all_steps_present"] for row in complete),
                "all_complete_variants_finite": bool(complete)
                and all(row["all_loss_and_gradient_values_finite"] for row in complete),
                "mean_first_loss": mean(
                    [float(row["first_loss"]) for row in complete if row["first_loss"] is not None]
                ),
                "mean_minimum_loss": mean(
                    [float(row["minimum_loss"]) for row in complete if row["minimum_loss"] is not None]
                ),
                "mean_final_loss": mean(
                    [float(row["final_loss"]) for row in complete if row["final_loss"] is not None]
                ),
                "mean_raw_gradient_rms": mean(
                    [
                        float(row["mean_raw_gradient_rms"])
                        for row in complete
                        if row["mean_raw_gradient_rms"] is not None
                    ]
                ),
            }
        )
    return {
        "protocol": "wan_gt_guidance_optimization_audit_v1",
        "seed": seed,
        "eligible_case_count": screening.get("eligible_case_count"),
        "eligible_target_count": screening.get("eligible_target_count"),
        "lambdas": list(lambdas),
        "trigger_modes": sorted(trigger_modes),
        "interpretation": (
            "Loss and gradient summaries are numerical optimization diagnostics, not "
            "trajectory-quality or physical-correctness outcomes."
        ),
        "aggregate": aggregate,
        "records": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    def fmt(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.6g}"

    lines = [
        "# GT-STC Guidance Optimization Audit",
        "",
        f"- Eligible cases / targets: **{report['eligible_case_count']} / {report['eligible_target_count']}**",
        f"- Sensitivity-trigger modes: **{', '.join(report['trigger_modes']) or 'none'}**",
        "- These are numerical diagnostics only; they do not measure trajectory quality.",
        "",
        "| λ | Mode | Complete | 40-step complete | Finite | Mean first loss | Mean min loss | Mean final loss | Mean raw grad RMS |",
        "|---:|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"| {row['lambda']:.2f} | {row['mode']} | "
            f"{row['complete_variants']}/{row['expected_variants']} | "
            f"{row['all_complete_variants_have_40_steps']} | "
            f"{row['all_complete_variants_finite']} | "
            f"{fmt(row['mean_first_loss'])} | {fmt(row['mean_minimum_loss'])} | "
            f"{fmt(row['mean_final_loss'])} | {fmt(row['mean_raw_gradient_rms'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--lambdas", type=float, nargs="+", default=(0.05, 0.1, 0.2))
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    report = audit(root, args.seed, tuple(args.lambdas))
    output = root / "final_analysis" / f"seed_{args.seed:05d}"
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "optimization_audit.json"
    md_path = output / "OPTIMIZATION_AUDIT.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
