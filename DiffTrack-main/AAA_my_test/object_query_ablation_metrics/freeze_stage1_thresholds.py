#!/usr/bin/env python3
"""Freeze query-time ranking stability thresholds before reading results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage1_query_time_validation/frozen_thresholds.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--draws", type=int, default=100_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    population, selected = 720, 100
    intersection = rng.hypergeometric(
        ngood=selected,
        nbad=population - selected,
        nsample=selected,
        size=args.draws,
    )
    jaccard = intersection / (2 * selected - intersection)

    # Under independent rankings of n items, Spearman rho has asymptotic
    # standard deviation 1/sqrt(n-1).  Sampling from that null avoids storing
    # 100k full 720-element permutations while remaining conservative here.
    rho_null = rng.normal(0.0, 1.0 / np.sqrt(population - 1), size=args.draws)
    null_jaccard_q99 = float(np.quantile(jaccard, 0.99))
    null_spearman_q99 = float(np.quantile(rho_null, 0.99))

    payload = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_seed": int(args.seed),
        "draws": int(args.draws),
        "population_heads": population,
        "top_n": selected,
        "null": {
            "top100_jaccard_q99": null_jaccard_q99,
            "spearman_q99": null_spearman_q99,
        },
        "practical_thresholds": {
            "median_pairwise_top100_jaccard": max(0.25, null_jaccard_q99),
            "median_pairwise_spearman": max(0.30, null_spearman_q99),
            "fixed_top100_beats_fixed_bottom100_anchor_fraction": 0.80,
            "case_cluster_bootstrap_lcb_top_minus_bottom_pck32": 0.0,
        },
        "bootstrap": {
            "highest_cluster": "case",
            "confidence": 0.95,
            "resamples": 10_000,
            "seed": 20260812,
        },
        "decision_rule": {
            "pass": "all practical thresholds pass",
            "conditional": (
                "metrics exceed permutation null but any practical threshold fails; "
                "retain fixed latest3350 scopes and add TubeTop100/TubeBottom100"
            ),
            "fail": (
                "ranking stability does not exceed permutation null or fixed Top100 "
                "does not beat Bottom100; do not interpret fixed ranking as tube-wide"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
