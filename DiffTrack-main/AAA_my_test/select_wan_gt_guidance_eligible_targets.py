#!/usr/bin/env python3
"""Freeze guidance targets using only source-tube and Baseline trajectory data."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)


def read_cases(path: Path) -> list[str]:
    unique: OrderedDict[str, None] = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            unique.setdefault(Path(value).stem, None)
    return list(unique)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def eligibility_row(
    case: str,
    object_row: dict[str, Any],
    metric_row: dict[str, Any] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    moving = bool(object_row.get("moving", False))
    if not moving:
        reasons.append("source motion_score_d0 < 0.05")
    if metric_row is None:
        reasons.append("Baseline metric missing")
    else:
        if int(metric_row.get("future_reference_anchor_count") or 0) < 4:
            reasons.append("source has <4 valid future anchors")
        if not bool(metric_row.get("quality_pass", False)):
            reasons.append("Baseline future trajectory quality gate failed")
    return {
        "case": case,
        "target": str(object_row["name"]),
        "phrase": str(object_row.get("phrase") or ""),
        "motion_score_d0": object_row.get("motion_score_d0"),
        "moving": moving,
        "baseline_quality_pass": (
            bool(metric_row.get("quality_pass", False)) if metric_row else False
        ),
        "future_reference_anchor_count": (
            metric_row.get("future_reference_anchor_count") if metric_row else None
        ),
        "future_common_anchor_count": (
            metric_row.get("future_common_anchor_count") if metric_row else None
        ),
        "future_common_anchor_coverage": (
            metric_row.get("future_common_anchor_coverage") if metric_row else None
        ),
        "future_track_loss_score_0_100": (
            metric_row.get("future_track_loss_score_0_100") if metric_row else None
        ),
        "eligible": not reasons,
        "exclusion_reasons": reasons,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    cases = read_cases(args.input_list.expanduser().resolve())
    root = args.output_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    case_status: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    for case in cases:
        tube_root = root / "gt_tubes" / case
        baseline_root = root / "generations" / case / f"seed_{args.seed:05d}" / "baseline"
        tube_manifest = tube_root / "manifest.json"
        metrics_path = baseline_root / "trajectory_metrics.json"
        status = {
            "case": case,
            "tube_complete": (tube_root / "complete.json").is_file(),
            "baseline_complete": (baseline_root / "complete.json").is_file(),
            "baseline_metrics_complete": metrics_path.is_file(),
        }
        case_status.append(status)
        if not status["tube_complete"] or not tube_manifest.is_file() or not metrics_path.is_file():
            missing_cases.append(case)
            continue
        tube = load_json(tube_manifest)
        metrics = load_json(metrics_path)
        by_target = {str(row["target"]): row for row in metrics.get("metrics", [])}
        for object_row in tube.get("objects", []):
            rows.append(
                eligibility_row(case, object_row, by_target.get(str(object_row["name"])))
            )

    eligible = [row for row in rows if row["eligible"]]
    jobs: dict[str, list[str]] = {}
    for row in eligible:
        jobs.setdefault(str(row["case"]), []).append(str(row["target"]))
    jobs = {case: sorted(targets) for case, targets in sorted(jobs.items())}
    report = {
        "protocol": "wan_gt_guidance_baseline_eligibility_v1",
        "input_list": str(args.input_list),
        "seed": int(args.seed),
        "case_count": len(cases),
        "case_status": case_status,
        "target_count": len(rows),
        "eligible_target_count": len(eligible),
        "eligible_case_count": len(jobs),
        "missing_case_count": len(missing_cases),
        "missing_cases": missing_cases,
        "gate": {
            "source_future_anchor_minimum": 4,
            "moving_threshold_d0": 0.05,
            "baseline_common_future_anchor_minimum": 4,
            "baseline_common_source_anchor_coverage_minimum": 0.8,
            "uses_guided_outcomes": False,
        },
        "eligible_jobs": [
            {"case": case, "targets": targets} for case, targets in jobs.items()
        ],
        "targets": rows,
    }
    screen_root = root / "screening" / f"seed_{args.seed:05d}"
    screen_root.mkdir(parents=True, exist_ok=True)
    json_path = screen_root / "baseline_eligibility.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Baseline Eligibility for GT Correspondence Guidance",
        "",
        f"- Cases: **{len(cases)}**",
        f"- Audited object targets: **{len(rows)}**",
        f"- Eligible targets/cases: **{len(eligible)} / {len(jobs)}**",
        f"- Missing cases: **{len(missing_cases)}**",
        "- Gate uses only source GT and same-seed Baseline; guided outcomes are not inspected.",
        "",
        "| Case | Target | Phrase | Motion/D0 | Baseline anchors | Coverage | Track Loss | Eligible | Reason |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        anchor_pair = (
            f"{fmt(row['future_common_anchor_count'], 0)}/"
            f"{fmt(row['future_reference_anchor_count'], 0)}"
        )
        lines.append(
            "| {case} | {target} | {phrase} | {motion} | {anchors} | {coverage} | "
            "{loss} | {eligible} | {reason} |".format(
                case=row["case"],
                target=row["target"],
                phrase=row["phrase"].replace("|", "/"),
                motion=fmt(row["motion_score_d0"]),
                anchors=anchor_pair,
                coverage=fmt(row["future_common_anchor_coverage"]),
                loss=fmt(row["future_track_loss_score_0_100"]),
                eligible="yes" if row["eligible"] else "no",
                reason="; ".join(row["exclusion_reasons"]) or "—",
            )
        )
    if missing_cases:
        lines.extend(["", "## Missing cases", ""] + [f"- `{case}`" for case in missing_cases])
    (screen_root / "BASELINE_ELIGIBILITY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "json": str(json_path),
        "eligible_target_count": len(eligible),
        "eligible_case_count": len(jobs),
        "missing_case_count": len(missing_cases),
    }, indent=2))
    if args.strict and missing_cases:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

