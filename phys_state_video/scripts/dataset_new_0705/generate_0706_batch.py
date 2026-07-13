#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render_sim_0705 import render_generated_case
from .scene_generators_0705 import build_scenario_family_catalog, preview_diversity_report


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0706pybullet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 0706 pybullet dataset batch with richer families and motion modes.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed-base", type=int, default=20260706)
    parser.add_argument("--family-pattern", default="balanced")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new cases after the largest existing batch index under output-root/cases.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Override the starting global case index. If omitted, starts from 0 unless --append is set.",
    )
    return parser.parse_args()


def _family_plan(num_cases: int, pattern: str) -> list[str]:
    families = list(build_scenario_family_catalog().keys())
    if pattern == "balanced":
        counts = {k: num_cases // len(families) for k in families}
        remainder = num_cases - sum(counts.values())
        for key in families[:remainder]:
            counts[key] += 1
        plan: list[str] = []
        for key in families:
            plan.extend([key] * counts[key])
        return plan
    if pattern == "motion_heavy":
        weights = {"F1": 12, "F2": 12, "F3": 12, "F4": 10, "F5": 10, "F6": 10, "F7": 10, "F8": 10, "F9": 7, "F10": 7}
        expanded: list[str] = []
        for key in families:
            expanded.extend([key] * weights[key])
        return [expanded[i % len(expanded)] for i in range(num_cases)]
    raise ValueError(f"unsupported family pattern: {pattern}")


def _detect_next_index(cases_root: Path) -> int:
    max_idx = -1
    for case_dir in cases_root.glob("F*/*"):
        if not case_dir.is_dir():
            continue
        suffix = case_dir.name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def _load_existing_manifest(output_root: Path) -> list[dict[str, object]]:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    cases_root = output_root / "cases"
    preview_root = output_root / "qa_preview"
    logs_root = output_root / "logs"
    reports_root = output_root / "reports"
    for path in [cases_root, preview_root, logs_root, reports_root]:
        path.mkdir(parents=True, exist_ok=True)

    if args.start_index is not None:
        start_index = int(args.start_index)
    elif args.append:
        start_index = _detect_next_index(cases_root)
    else:
        start_index = 0

    full_family_plan = _family_plan(start_index + args.num_cases, args.family_pattern)
    family_plan = full_family_plan[start_index : start_index + args.num_cases]
    manifest: list[dict[str, object]] = _load_existing_manifest(output_root) if (args.append or start_index > 0) else []
    failures: list[dict[str, object]] = []

    for offset, family_key in enumerate(family_plan):
        global_idx = start_index + offset
        case_id = f"0706_{family_key.lower()}_{global_idx:03d}"
        seed = args.seed_base + global_idx * 1009
        case_root = cases_root / family_key / case_id
        try:
            record = render_generated_case(
                family_key=family_key,
                sample_key=case_id,
                seed=seed,
                output_root=case_root,
                width=args.width,
                height=args.height,
            )
            manifest.append(
                {
                    "case_id": case_id,
                    "family_key": family_key,
                    "seed": seed,
                    "output_root": str(case_root),
                    "video": record["video"],
                    "meta": record["meta"],
                    "caption": record.get("caption", ""),
                    "short_caption": record.get("short_caption", ""),
                    "negative_prompt": record.get("negative_prompt", ""),
                }
            )
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append(
                {
                    "case_id": case_id,
                    "family_key": family_key,
                    "seed": seed,
                    "error": repr(exc),
                }
            )

    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_root / "failure_report.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_root / "diversity_report.json").write_text(
        json.dumps(preview_diversity_report(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (logs_root / "batch_summary.txt").write_text(
        (
            f"cases={len(manifest)} failures={len(failures)} pattern={args.family_pattern} "
            f"start_index={start_index} append={bool(args.append)} width={args.width} height={args.height}\n"
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(manifest),
                "new_cases": len(family_plan),
                "failures": len(failures),
                "start_index": start_index,
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
