#!/usr/bin/env python3
"""Validate the three-model matched-subset generation preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from matched_head_subset_targets import load_matched_subset
from run_head_role_dose_control_pilot_worker import _input_cases, _validate_job


MODELS = ("wan_lora", "xssc", "physrvg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--subset-id", required=True)
    parser.add_argument("--seed", type=int, default=851)
    parser.add_argument("--step-start", type=int, default=0)
    parser.add_argument("--step-end", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    cases = _input_cases(args.input_list.expanduser().resolve())
    _, targets, _ = load_matched_subset(manifest, args.subset_id)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    records = {}
    for model in MODELS:
        variant = (
            f"{args.subset_id}_steps{args.step_start:02d}_{args.step_end:02d}"
        )
        job_root = (
            root
            / "generation"
            / model
            / f"seed-{args.seed:06d}"
            / variant
        )
        videos = _validate_job(
            job_root,
            cases=cases,
            subset_id=args.subset_id,
            manifest_sha256=manifest_sha256,
            k=len(targets),
            start=args.step_start,
            end=args.step_end,
        )
        records[model] = videos
    payload = {
        "status": "passed",
        "models": list(MODELS),
        "cases": sorted(cases),
        "subset_id": args.subset_id,
        "k": len(targets),
        "step_range": [args.step_start, args.step_end],
        "manifest_sha256": manifest_sha256,
        "videos": records,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
