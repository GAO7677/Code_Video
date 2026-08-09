#!/usr/bin/env python3
"""Build a strict common-cohort macro mean over single-seed metric reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


CASE = "0613pybullet_sample_001460_w002"
DEFAULT_SEEDS = (13248, 32466, 35075, 47326, 68613, 90094)
DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics"
) / CASE
METRIC_FIELDS = (
    "objects",
    "other_object",
    "interaction",
    "raft",
    "pixel",
    "outside_object_lpips",
    "vbench",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--representative-seed", type=int, default=47326)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_report(root: Path, seed: int) -> dict[str, Any]:
    path = root / f"seed_{seed:05d}" / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing report for seed {seed}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case") != CASE or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"report identity mismatch: {path}")
    if int(payload.get("ablation_count", -1)) != 48:
        raise RuntimeError(f"expected 48 ablations: {path}")
    return payload


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def strict_mean_tree(
    values: list[Any],
    path: str,
    expected_count: int,
    completeness: dict[str, int],
) -> Any:
    """Average scalar leaves only when every cohort sample has a finite value."""
    first = values[0]
    if all(isinstance(value, dict) for value in values):
        keys = list(first)
        return {
            key: strict_mean_tree(
                [value.get(key) for value in values],
                f"{path}.{key}" if path else key,
                expected_count,
                completeness,
            )
            for key in keys
        }
    finite_count = sum(is_number(value) for value in values)
    if finite_count:
        completeness[path] = finite_count
        if finite_count == expected_count:
            return float(fmean(float(value) for value in values))
        return None
    if any(value is None for value in values):
        completeness[path] = 0
        return None
    # Per-frame arrays are kept from the representative seed for audit/overlay only;
    # no dashboard metric cell reads a list-valued leaf.
    if isinstance(first, list):
        return deepcopy(first)
    return deepcopy(first)


def main() -> None:
    args = parse_args()
    seeds = tuple(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.representative_seed not in seeds:
        raise ValueError("representative seed must belong to the common cohort")

    reports = {seed: load_report(args.root, seed) for seed in seeds}
    representative = reports[args.representative_seed]
    expected_ids = [record["id"] for record in representative["records"]]
    for seed, report in reports.items():
        actual_ids = [record["id"] for record in report["records"]]
        if actual_ids != expected_ids:
            raise RuntimeError(f"record cohort/order mismatch for seed {seed}")

    records_by_seed = {
        seed: {record["id"]: record for record in report["records"]}
        for seed, report in reports.items()
    }
    completeness: dict[str, int] = {}
    aggregate_records = []
    for representative_record in representative["records"]:
        record_id = representative_record["id"]
        cohort_records = [records_by_seed[seed][record_id] for seed in seeds]
        aggregate_record = {
            key: deepcopy(representative_record[key])
            for key in ("id", "protocol", "target_scope", "region", "mask_mode", "operator_id")
        }
        for field in METRIC_FIELDS:
            aggregate_record[field] = strict_mean_tree(
                [record.get(field) for record in cohort_records],
                f"records.{record_id}.{field}",
                len(seeds),
                completeness,
            )
        aggregate_record["assets"] = deepcopy(representative_record["assets"])
        aggregate_records.append(aggregate_record)

    incomplete = sorted(path for path, count in completeness.items() if count != len(seeds))
    output = args.output or args.root / "aggregate" / "report.json"
    representative_root = args.root / f"seed_{args.representative_seed:05d}"
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": CASE,
        "seed": None,
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "sample_unit": "case-seed",
        "sample_count": len(seeds),
        "distinct_case_count": 1,
        "video_count": len(seeds) * 49,
        "ablation_count": len(seeds) * 48,
        "experiment_row_count": 48,
        "aggregation": {
            "method": "case-seed macro mean",
            "cohort_policy": "strict common cohort",
            "rule": (
                "Every displayed scalar is averaged over the same six case-seed samples. "
                "If any sample is missing/non-finite, the aggregate scalar is N/A."
            ),
            "seeds": list(seeds),
            "expected_count_per_scalar": len(seeds),
            "incomplete_scalar_count": len(incomplete),
            "incomplete_scalar_paths": incomplete,
            "representative_seed_for_assets": args.representative_seed,
            "per_frame_arrays": "representative seed only; never used in aggregate table cells",
        },
        "representative_output_root": str(representative_root),
        "source_reports": {
            str(seed): str(args.root / f"seed_{seed:05d}" / "report.json")
            for seed in seeds
        },
        "references": deepcopy(representative["references"]),
        "metric_definitions": deepcopy(representative["metric_definitions"]),
        "protocol": deepcopy(representative["protocol"]),
        "records": aggregate_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)

    audit_csv = output.with_name("scalar_completeness.csv")
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric_path", "finite_sample_count", "expected_sample_count", "displayable"),
        )
        writer.writeheader()
        for metric_path, count in sorted(completeness.items()):
            writer.writerow(
                {
                    "metric_path": metric_path,
                    "finite_sample_count": count,
                    "expected_sample_count": len(seeds),
                    "displayable": count == len(seeds),
                }
            )
    print(
        json.dumps(
            {
                "output": str(output),
                "seeds": seeds,
                "records": len(aggregate_records),
                "scalar_leaves": len(completeness),
                "incomplete_scalar_leaves": len(incomplete),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
