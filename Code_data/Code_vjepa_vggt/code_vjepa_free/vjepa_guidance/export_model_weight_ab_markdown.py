#!/usr/bin/env python3
from __future__ import annotations

"""
Export model-weight A/B score summaries into a local Markdown report.

Example:
  python export_model_weight_ab_markdown.py \
    --scores-dir /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/scores \
    --output-md /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/ab_report/model_weight_ab_report.md
"""

import argparse
import json
from pathlib import Path
from typing import Any


FAMILY_TITLES = {
    "train0705_step002500": "train0705 step-002500",
    "train0705_step005000": "train0705 step-005000",
    "wan22_official_ti2v5b": "Wan2.2 official TI2V-5B",
    "wan22_early_lora_step000500": "Wan2.2 early LoRA step-000500",
}

SUMMARY_ORDER = [
    "wan22_official_ti2v5b",
    "wan22_early_lora_step000500",
    "train0705_step002500",
    "train0705_step005000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model-weight A/B results to Markdown.")
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Model-Weight A/B Report")
    return parser.parse_args()


def family_sort_key(family_id: str) -> tuple[int, str]:
    try:
        return (SUMMARY_ORDER.index(family_id), family_id)
    except ValueError:
        return (len(SUMMARY_ORDER), family_id)


def safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_link(path_str: str | None, label: str | None = None) -> str:
    if not path_str:
        return "NA"
    label = label or Path(path_str).name
    return f"[{label}]({path_str})"


def collect_payload(scores_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_files = sorted(
        [
            path
            for path in scores_dir.glob("*_summary.json")
            if path.name != "combined_summary.json"
        ],
        key=lambda path: family_sort_key(path.stem.removesuffix("_summary")),
    )

    families: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}

    for summary_path in family_files:
        family_id = summary_path.stem.removesuffix("_summary")
        payload = safe_read_json(summary_path)
        family_title = FAMILY_TITLES.get(family_id, family_id)
        families.append(
            {
                "family_id": family_id,
                "title": family_title,
                "summary": payload.get("summary_by_method", {}),
                "method_dirs": payload.get("method_dirs", {}),
                "baseline_label": payload.get("baseline_label", "baseline"),
            }
        )

        for row in payload.get("rows", []):
            case_id = row["case_id"]
            method = row["method"]
            case_bucket = cases.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "prompt": row.get("prompt"),
                    "source_video": row.get("source_video"),
                    "input_json": row.get("input_json"),
                    "families": {},
                },
            )
            case_bucket["prompt"] = case_bucket["prompt"] or row.get("prompt")
            case_bucket["source_video"] = case_bucket["source_video"] or row.get("source_video")
            case_bucket["input_json"] = case_bucket["input_json"] or row.get("input_json")
            family_bucket = case_bucket["families"].setdefault(
                family_id,
                {
                    "family_id": family_id,
                    "title": family_title,
                    "baseline": None,
                    "guided": None,
                },
            )
            family_bucket[method] = row

    case_list = [cases[key] for key in sorted(cases)]
    return families, case_list


def build_markdown(title: str, families: list[dict[str, Any]], cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Families: {len(families)}")
    lines.append(f"- Cases: {len(cases)}")
    lines.append("- Note: video links are absolute local file paths. Open them directly from the Markdown preview or file explorer if inline playback is unavailable.")
    lines.append("")

    lines.append("## Family Summary")
    lines.append("")
    lines.append("| Family | Baseline surprise | Guided surprise | Δ surprise | Baseline Physics-IQ | Guided Physics-IQ | Δ Physics-IQ | Baseline VideoPhy2 | Guided VideoPhy2 | Δ VideoPhy2 | Baseline Cosmos-R1 | Guided Cosmos-R1 | Δ Cosmos-R1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for family in sorted(families, key=lambda item: family_sort_key(item["family_id"])):
        baseline = family["summary"].get("baseline", {})
        guided = family["summary"].get("guided", {})
        lines.append(
            "| {title} | {bs} | {gs} | {ds} | {bp} | {gp} | {dp} | {bv} | {gv} | {dv} | {bc} | {gc} | {dc} |".format(
                title=family["title"],
                bs=fmt(baseline.get("mean_surprise"), 6),
                gs=fmt(guided.get("mean_surprise"), 6),
                ds=fmt(guided.get("mean_delta_surprise_vs_baseline"), 6),
                bp=fmt(baseline.get("mean_physics_iq"), 4),
                gp=fmt(guided.get("mean_physics_iq"), 4),
                dp=fmt(guided.get("mean_delta_physics_iq_vs_baseline"), 4),
                bv=fmt(baseline.get("mean_videophy2"), 4),
                gv=fmt(guided.get("mean_videophy2"), 4),
                dv=fmt(guided.get("mean_delta_videophy2_vs_baseline"), 4),
                bc=fmt(baseline.get("mean_cosmos_reason1"), 4),
                gc=fmt(guided.get("mean_cosmos_reason1"), 4),
                dc=fmt(guided.get("mean_delta_cosmos_reason1_vs_baseline"), 4),
            )
        )
    lines.append("")

    lines.append("## Case Index")
    lines.append("")
    for case in cases:
        lines.append(f"- [{case['case_id']}](#{case['case_id'].lower()})")
    lines.append("")

    lines.append("## Per-Case Comparison")
    lines.append("")
    for case in cases:
        lines.append(f"### {case['case_id']}")
        lines.append("")
        lines.append(f"- Prompt: {case.get('prompt') or 'NA'}")
        lines.append(f"- Input JSON: {md_link(case.get('input_json'))}")
        lines.append(f"- Source Video: {md_link(case.get('source_video'))}")
        lines.append("")
        lines.append("| Family | Baseline video | Guided video | Baseline surprise | Guided surprise | Δ surprise | Baseline Physics-IQ | Guided Physics-IQ | Δ Physics-IQ | Baseline VideoPhy2 | Guided VideoPhy2 | Δ VideoPhy2 | Baseline Cosmos-R1 | Guided Cosmos-R1 | Δ Cosmos-R1 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for family_id in sorted(case["families"], key=family_sort_key):
            family = case["families"][family_id]
            baseline = family.get("baseline") or {}
            guided = family.get("guided") or {}
            lines.append(
                "| {title} | {bvideo} | {gvideo} | {bs} | {gs} | {ds} | {bp} | {gp} | {dp} | {bv} | {gv} | {dv} | {bc} | {gc} | {dc} |".format(
                    title=family["title"],
                    bvideo=md_link(baseline.get("video_path"), "baseline.mp4"),
                    gvideo=md_link(guided.get("video_path"), "guided.mp4"),
                    bs=fmt(baseline.get("surprise"), 6),
                    gs=fmt(guided.get("surprise"), 6),
                    ds=fmt(guided.get("delta_surprise_vs_baseline"), 6),
                    bp=fmt(baseline.get("physics_iq_score"), 4),
                    gp=fmt(guided.get("physics_iq_score"), 4),
                    dp=fmt(guided.get("delta_physics_iq_vs_baseline"), 4),
                    bv=fmt(baseline.get("videophy2_score"), 4),
                    gv=fmt(guided.get("videophy2_score"), 4),
                    dv=fmt(guided.get("delta_videophy2_vs_baseline"), 4),
                    bc=fmt(baseline.get("cosmos_reason1_score"), 4),
                    gc=fmt(guided.get("cosmos_reason1_score"), 4),
                    dc=fmt(guided.get("delta_cosmos_reason1_vs_baseline"), 4),
                )
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    families, cases = collect_payload(args.scores_dir)
    md_text = build_markdown(args.title, families, cases)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_text, encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
