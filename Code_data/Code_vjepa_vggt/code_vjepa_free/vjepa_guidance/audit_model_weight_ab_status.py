#!/usr/bin/env python3
from __future__ import annotations

"""
Audit the current completion state of model-weight A/B output roots.

Example:
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/audit_model_weight_ab_status.py \
    --output-root /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705 \
    --out-json /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/status_audit.json \
    --out-md /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/status_audit.md
"""

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CASES = 17
EXPECTED_TARGET_FAMILIES = [
    "wan22_official_ti2v5b",
    "wan22_early_lora_step000500",
    "train0705_step002500",
    "train0705_step007000",
]
REQUIRED_SCORE_FIELDS = [
    "official_pdi",
    "wmreward_surprise",
    "proxy_score",
    "videophy2_score",
    "phyground_general_avg",
    "cosmos_reason1_score",
    "physics_iq_score",
    "pmf_score",
]
KNOWN_FAMILY_TITLES = {
    "wan22_official_ti2v5b": "Wan2.2 official TI2V-5B",
    "wan22_early_lora_step000500": "Wan2.2 early LoRA step-000500",
    "train0705_step002500": "train0705 step-002500",
    "train0705_step005000": "train0705 step-005000",
    "train0705_step007000": "train0705 step-007000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit model-weight A/B output completeness.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser.parse_args()


def is_family_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.startswith(("wan22_", "train0705_")):
        return True
    return (path / "baseline").is_dir() or (path / "guided").is_dir()


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_unique_case_videos(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    names: set[str] = set()
    for path in directory.rglob("*.mp4"):
        if path.name.endswith(".browser.mp4"):
            continue
        names.add(path.stem)
    return len(names)


def detect_primary_leaf(directory: Path) -> Path:
    current = directory
    while current.is_dir():
        mp4_count = len([p for p in current.glob("*.mp4") if not p.name.endswith(".browser.mp4")])
        if mp4_count > 0:
            return current
        subdirs = sorted([p for p in current.iterdir() if p.is_dir()])
        if len(subdirs) != 1:
            return current
        current = subdirs[0]
    return directory


def audit_family(output_root: Path, family_id: str) -> dict[str, Any]:
    family_root = output_root / family_id
    baseline_root = family_root / "baseline"
    guided_root = family_root / "guided"
    baseline_leaf = detect_primary_leaf(baseline_root) if baseline_root.exists() else baseline_root
    guided_leaf = detect_primary_leaf(guided_root) if guided_root.exists() else guided_root
    score_json = output_root / "scores" / f"{family_id}_summary.json"
    score_payload = safe_read_json(score_json) if score_json.exists() else None
    sample_row = ((score_payload or {}).get("rows") or [None])[0] or {}
    present_score_fields = [field for field in REQUIRED_SCORE_FIELDS if field in sample_row]
    missing_score_fields = [field for field in REQUIRED_SCORE_FIELDS if field not in sample_row]
    baseline_count = count_unique_case_videos(baseline_leaf)
    guided_count = count_unique_case_videos(guided_leaf)

    return {
        "family_id": family_id,
        "title": KNOWN_FAMILY_TITLES.get(family_id, family_id),
        "family_root": str(family_root),
        "baseline_leaf": str(baseline_leaf),
        "guided_leaf": str(guided_leaf),
        "baseline_exists": baseline_root.exists(),
        "guided_exists": guided_root.exists(),
        "baseline_video_count": baseline_count,
        "guided_video_count": guided_count,
        "baseline_complete": baseline_count >= EXPECTED_CASES,
        "guided_complete": guided_count >= EXPECTED_CASES,
        "score_json": str(score_json),
        "score_exists": score_json.exists(),
        "score_method_dirs": (score_payload or {}).get("method_dirs", {}),
        "score_row_count": len((score_payload or {}).get("rows", [])),
        "score_summary_methods": sorted(((score_payload or {}).get("summary_by_method") or {}).keys()),
        "score_fields_present": present_score_fields,
        "score_fields_missing": missing_score_fields,
        "score_has_full_metric_coverage": score_json.exists() and not missing_score_fields,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Model-Weight A/B Status Audit")
    lines.append("")
    lines.append(f"- Output root: `{report['output_root']}`")
    lines.append(f"- Expected cases per family: {report['expected_cases']}")
    lines.append("")

    lines.append("## Target Family Status")
    lines.append("")
    lines.append("| Family | Baseline videos | Guided videos | Score file | Full metrics | Complete for A/B |")
    lines.append("|---|---:|---:|---|---|---|")
    for family in report["target_families"]:
        lines.append(
            "| {title} | {b} | {g} | {s} | {m} | {done} |".format(
                title=family["title"],
                b=family["baseline_video_count"],
                g=family["guided_video_count"],
                s="yes" if family["score_exists"] else "no",
                m="yes" if family["score_has_full_metric_coverage"] else "no",
                done="yes"
                if family["baseline_complete"] and family["guided_complete"] and family["score_has_full_metric_coverage"]
                else "no",
            )
        )
    lines.append("")

    if report["extra_families"]:
        lines.append("## Extra Families Present In This Root")
        lines.append("")
        lines.append("| Family | Baseline videos | Guided videos | Score file | Full metrics |")
        lines.append("|---|---:|---:|---|---|")
        for family in report["extra_families"]:
            lines.append(
                "| {title} | {b} | {g} | {s} | {m} |".format(
                    title=family["title"],
                    b=family["baseline_video_count"],
                    g=family["guided_video_count"],
                    s="yes" if family["score_exists"] else "no",
                    m="yes" if family["score_has_full_metric_coverage"] else "no",
                )
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    present_families = sorted([p.name for p in output_root.iterdir() if is_family_dir(p)]) if output_root.is_dir() else []
    target_families = [audit_family(output_root, family_id) for family_id in EXPECTED_TARGET_FAMILIES]
    extra_ids = [family_id for family_id in present_families if family_id not in EXPECTED_TARGET_FAMILIES]
    extra_families = [audit_family(output_root, family_id) for family_id in extra_ids]

    notes: list[str] = []
    missing_targets = [family["family_id"] for family in target_families if not (family["baseline_exists"] or family["guided_exists"] or family["score_exists"])]
    if missing_targets:
        notes.append(f"Missing target families in this root: {', '.join(missing_targets)}")
    partial_targets = [
        family["family_id"]
        for family in target_families
        if (family["baseline_exists"] or family["guided_exists"] or family["score_exists"])
        and not (family["baseline_complete"] and family["guided_complete"] and family["score_has_full_metric_coverage"])
    ]
    if partial_targets:
        notes.append(f"Partially populated target families: {', '.join(partial_targets)}")
    incomplete_score_targets = [
        family["family_id"] for family in target_families if family["score_exists"] and not family["score_has_full_metric_coverage"]
    ]
    if incomplete_score_targets:
        notes.append(
            "Score files exist but do not yet cover the full README metric set for: "
            + ", ".join(incomplete_score_targets)
        )
    if extra_ids:
        notes.append(f"Extra families present beyond the current 4-family target: {', '.join(extra_ids)}")
    if not notes:
        notes.append("All target families appear complete for baseline, guided, and score coverage.")

    report = {
        "output_root": str(output_root),
        "expected_cases": EXPECTED_CASES,
        "expected_target_families": EXPECTED_TARGET_FAMILIES,
        "present_families": present_families,
        "target_families": target_families,
        "extra_families": extra_families,
        "notes": notes,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(build_markdown(report), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
