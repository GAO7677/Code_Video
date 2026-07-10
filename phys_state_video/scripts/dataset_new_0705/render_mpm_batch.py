#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .render_mpm_preview_case import CASE_LIBRARY, build_family_case_catalog, render_case
except ImportError:
    from render_mpm_preview_case import CASE_LIBRARY, build_family_case_catalog, render_case


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_preview_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-render the current MPM preview family library.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--families", default="", help="Optional comma-separated family filter, e.g. F1,F4,F7.")
    parser.add_argument("--case-keys", default="", help="Optional comma-separated explicit case keys.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Video playback speed relative to real simulation time. Values below 1.0 are slower.",
    )
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--mpm-vis-mode", choices=["visual", "particle"], default="")
    parser.add_argument("--save-every-frame", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_csv(raw: str) -> set[str]:
    return {token.strip() for token in raw.split(",") if token.strip()}


def _select_cases(family_filter: set[str], case_filter: set[str]) -> list[tuple[str, str]]:
    catalog = build_family_case_catalog()
    selected: list[tuple[str, str]] = []
    for family, cases in sorted(catalog.items()):
        if family_filter and family not in family_filter:
            continue
        for case in cases:
            if case_filter and case.key not in case_filter:
                continue
            selected.append((family, case.key))
    return selected


def main() -> None:
    args = parse_args()
    family_filter = _parse_csv(args.families)
    case_filter = _parse_csv(args.case_keys)
    selected = _select_cases(family_filter, case_filter)
    if not selected:
        raise SystemExit("No MPM cases selected. Check --families or --case-keys.")

    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for family, case_key in selected:
        case_root = args.output_root / case_key
        manifest_path = case_root / "manifest.json"
        if args.skip_existing and manifest_path.exists():
            skipped.append({"family": family, "case_key": case_key, "output_root": str(case_root)})
            continue
        if args.dry_run:
            manifest.append(
                {
                    "family": family,
                    "case_key": case_key,
                    "title": CASE_LIBRARY[case_key].title,
                    "output_root": str(case_root),
                    "dry_run": True,
                }
            )
            continue
        try:
            result = render_case(
                case_key=case_key,
                output_root=args.output_root,
                run_name=case_key,
                fps=args.fps,
                playback_speed=args.playback_speed,
                width=args.width,
                height=args.height,
                mpm_vis_mode_override=args.mpm_vis_mode,
                save_every_frame=args.save_every_frame,
            )
            manifest.append(result)
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append(
                {
                    "family": family,
                    "case_key": case_key,
                    "output_root": str(case_root),
                    "error": repr(exc),
                }
            )

    batch_manifest_path = args.output_root / "batch_manifest.json"
    batch_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    failure_report_path = args.output_root / "failure_report.json"
    failure_report_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    skip_report_path = args.output_root / "skip_report.json"
    skip_report_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "selected_cases": len(selected),
        "rendered_cases": len([item for item in manifest if not item.get("dry_run")]),
        "dry_run_cases": len([item for item in manifest if item.get("dry_run")]),
        "failures": len(failures),
        "skipped": len(skipped),
        "families": sorted({family for family, _ in selected}),
        "output_root": str(args.output_root),
        "filters": {
            "families": sorted(family_filter),
            "case_keys": sorted(case_filter),
        },
        "render_settings": {
            "fps": args.fps,
            "playback_speed": args.playback_speed,
            "width": args.width,
            "height": args.height,
            "mpm_vis_mode_override": args.mpm_vis_mode,
            "save_every_frame": args.save_every_frame,
            "skip_existing": args.skip_existing,
            "dry_run": args.dry_run,
        },
        "artifacts": {
            "batch_manifest": str(batch_manifest_path),
            "failure_report": str(failure_report_path),
            "skip_report": str(skip_report_path),
        },
    }
    summary_path = args.output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
