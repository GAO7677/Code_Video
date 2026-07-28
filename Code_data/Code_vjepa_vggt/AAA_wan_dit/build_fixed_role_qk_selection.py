#!/usr/bin/env python3
"""Build a case-complete selected-QK map from aggregate head roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROLES = ("S", "T", "P", "C", "G")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-role-report", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.head_role_report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = [
        Path(line.strip()).stem
        for line in args.input_list.expanduser().resolve().read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    samples = {}
    selected_heads = {}
    for model, model_payload in report["models"].items():
        role_pairs = {}
        for role in ROLES:
            candidates = [
                row
                for row in model_payload["aggregate_heads"]
                if row["role"] == role
            ]
            if not candidates:
                raise RuntimeError(f"{model}: no aggregate head classified as {role}")
            best = max(
                candidates,
                key=lambda row: (
                    float(row[f"score_{role}"]),
                    float(row["support"]),
                    float(row["margin"]),
                ),
            )
            role_pairs[role] = {
                "block": int(best["block"]),
                "head": int(best["head"]),
                "aggregate_score": float(best[f"score_{role}"]),
                "support": float(best["support"]),
                "margin": float(best["margin"]),
            }
        selected_heads[model] = role_pairs
        samples[model] = {
            case: {"roles": role_pairs}
            for case in cases
        }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": str(report_path),
                "policy": (
                    "One fixed aggregate top head per S/T/P/C/G role and model; "
                    "the same five heads are captured for every case and pending seed."
                ),
                "selected_heads": selected_heads,
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[fixed-role-selection] wrote {output}")


if __name__ == "__main__":
    main()
