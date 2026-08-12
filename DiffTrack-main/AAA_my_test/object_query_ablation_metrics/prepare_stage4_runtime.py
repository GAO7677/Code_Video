#!/usr/bin/env python3
"""Build the exact frozen Stage-4 runtime manifests from existing sample manifests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326"
)
HEAD_RANKING = EXPERIMENT_ROOT / "head_scopes_latest3350_with_random100.json"
SPEC = Path(__file__).resolve().parent / "experiment_spec_stage4_temporal_v1.json"
DISCOVERY_SEEDS = (13248, 47326, 90094)
SOURCE_MANIFESTS = (
    SOURCE_ROOT / "cases.json",
    SOURCE_ROOT / "cases_001460_5seeds.json",
    SOURCE_ROOT / "cases_other10_6seeds_latest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "stage4_runtime",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    spec = read_json(SPEC)
    ranking = read_json(HEAD_RANKING)
    expected_cases = {
        str(row["case"]): int(row["object_count"])
        for row in spec["stage4a_pilot"]["cases"]
    }

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for source_path in SOURCE_MANIFESTS:
        source = read_json(source_path)
        for sample in source.get("samples", []):
            key = (str(sample.get("case")), int(sample.get("seed", -1)))
            if key[0] in expected_cases and key[1] in DISCOVERY_SEEDS:
                indexed[key] = sample

    required = {
        (case, seed) for case in expected_cases for seed in DISCOVERY_SEEDS
    }
    missing = sorted(required - set(indexed))
    if missing:
        raise RuntimeError(f"missing Stage-4 case/seed samples: {missing}")

    for key in sorted(required):
        sample = indexed[key]
        object_count = sum(
            row.get("region_type") == "object" for row in sample.get("regions", [])
        )
        if object_count != expected_cases[key[0]]:
            raise RuntimeError(
                f"{key}: manifest has {object_count} object regions, "
                f"spec requires {expected_cases[key[0]]}"
            )
        baseline = Path(str(sample.get("baseline_video") or ""))
        if not baseline.is_file():
            raise FileNotFoundError(f"missing Baseline for {key}: {baseline}")

    ball_case = next(case for case in expected_cases if "ball-and-block-fall" in case)
    short_cases = [case for case in expected_cases if case != ball_case]
    # With sample-level [worker_id::2] sharding this order gives 17 vs 16 targets.
    ordered_keys = [
        (ball_case, 47326),
        (ball_case, 13248),
        (short_cases[0], 13248),
        (ball_case, 90094),
        (short_cases[0], 47326),
        (short_cases[0], 90094),
        (short_cases[1], 13248),
        (short_cases[1], 47326),
        (short_cases[1], 90094),
    ]
    if set(ordered_keys) != required:
        raise RuntimeError("internal Stage-4 sharding order does not cover exact cohort")

    base_payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": spec["experiment_id"],
        "ranking_tag": spec["head_scopes"]["ranking_tag"],
        "head_ranking_path": str(HEAD_RANKING),
        "entries": ranking["entries"][:100],
        "seeds": list(DISCOVERY_SEEDS),
        "source_manifests": [str(path) for path in SOURCE_MANIFESTS],
    }
    full = {**base_payload, "samples": [indexed[key] for key in ordered_keys]}
    only_001460 = {
        **base_payload,
        "samples": [
            indexed[("0613pybullet_sample_001460_w002", seed)]
            for seed in DISCOVERY_SEEDS
        ],
    }
    atomic_json(args.output_dir / "stage4_manifest.json", full)
    atomic_json(args.output_dir / "stage4_manifest_001460.json", only_001460)
    atomic_json(
        args.output_dir / "runtime_summary.json",
        {
            "case_seed_count": len(full["samples"]),
            "target_count_per_seed_across_cases": 11,
            "directional_cells": 891,
            "missing_001460_all_time_cells": 81,
            "all720_sentinel_cells": 27,
            "total_generation_cells": 999,
            "worker_target_loads": [17, 16],
        },
    )
    print(args.output_dir / "stage4_manifest.json")


if __name__ == "__main__":
    main()
