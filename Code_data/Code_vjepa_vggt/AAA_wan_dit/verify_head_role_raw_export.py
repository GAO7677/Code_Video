#!/usr/bin/env python3
"""Verify that exported sample-level scores reconstruct the frozen report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROLES = ("S", "T", "P", "C", "G")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--frozen-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    return parser.parse_args()


def _frozen_rows(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        key = (row["model"], int(row["block"]), int(row["head"]))
        output[key] = row
    return output


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.expanduser().resolve()
    frozen_path = args.frozen_report.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest = json.loads(
        (export_dir / "classification_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "complete":
        raise RuntimeError("Raw export manifest is not complete")
    files = sorted((export_dir / "sample_roles").glob("**/*.parquet"))
    if len(files) != len(manifest["partitions"]):
        raise RuntimeError(
            f"Found {len(files)} sample-role files for "
            f"{len(manifest['partitions'])} partitions"
        )
    frame = pd.concat(
        [pd.read_parquet(path) for path in files],
        ignore_index=True,
    )
    primary_key = ["model", "source_case", "seed", "block", "head"]
    if frame.duplicated(primary_key).any():
        raise RuntimeError("Duplicate sample-role primary keys")
    frozen = _frozen_rows(frozen_path)
    observed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for (model, block, head), group in frame.groupby(
        ["model", "block", "head"],
        sort=True,
    ):
        mean_scores = np.asarray(
            [
                np.nanmean(
                    np.where(
                        np.isfinite(group[f"mean_score_{role}"].to_numpy()),
                        group[f"mean_score_{role}"].to_numpy(),
                        np.nan,
                    )
                )
                for role in ROLES
            ],
            dtype=np.float64,
        )
        winner_index = int(np.argmax(mean_scores))
        candidate_role = ROLES[winner_index]
        sorted_scores = np.sort(mean_scores)
        margin = float(sorted_scores[-1] - sorted_scores[-2])
        agreement = group["role"].to_numpy() == candidate_role
        if candidate_role in {"T", "P"}:
            valid = group["trajectory_valid"].to_numpy(dtype=bool)
            if not valid.any():
                raise RuntimeError(f"No valid trajectories for {(model, block, head)}")
            support = float(agreement[valid].mean())
        else:
            support = float(agreement.mean())
        role = candidate_role if margin >= 0.08 and support >= 0.50 else "M"
        observed[(str(model), int(block), int(head))] = {
            "role": role,
            "margin": margin,
            "support": support,
            **{
                f"score_{name}": float(mean_scores[index])
                for index, name in enumerate(ROLES)
            },
        }

    if set(observed) != set(frozen):
        raise RuntimeError(
            f"Aggregate key mismatch: observed={len(observed)} frozen={len(frozen)}"
        )
    mismatches: list[dict[str, Any]] = []
    maximum_absolute_difference = 0.0
    for key in sorted(frozen):
        expected = frozen[key]
        actual = observed[key]
        if actual["role"] != expected["role"]:
            mismatches.append(
                {
                    "key": key,
                    "field": "role",
                    "expected": expected["role"],
                    "actual": actual["role"],
                }
            )
        for field in (
            "margin",
            "support",
            "score_S",
            "score_T",
            "score_P",
            "score_C",
            "score_G",
        ):
            difference = abs(float(actual[field]) - float(expected[field]))
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                difference,
            )
            if difference > float(args.atol):
                mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "expected": float(expected[field]),
                        "actual": float(actual[field]),
                        "absolute_difference": difference,
                    }
                )
    report = {
        "status": "passed" if not mismatches else "failed",
        "export_manifest": str(export_dir / "classification_manifest.json"),
        "frozen_report": str(frozen_path),
        "sample_rows": len(frame),
        "aggregate_rows": len(observed),
        "absolute_tolerance": float(args.atol),
        "maximum_absolute_difference": maximum_absolute_difference,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
