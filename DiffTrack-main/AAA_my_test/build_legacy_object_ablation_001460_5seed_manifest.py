#!/usr/bin/env python3
"""Build the five-seed manifest for the fixed-query and temporal-tube pilot."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path


CASE = "0613pybullet_sample_001460_w002"
SEEDS = (90094, 68613, 35075, 32466, 13248)
ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326"
)
SOURCE = ROOT / "cases.json"
OUTPUT = ROOT / "cases_001460_5seeds.json"
BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_pck50/runs"
)


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    pilot = next(
        row
        for row in source["samples"]
        if row["case"] == CASE and int(row["seed"]) == 47326
    )
    samples = []
    for seed in SEEDS:
        baseline = BASELINE_ROOT / CASE / f"seed_{seed:05d}" / "generated.mp4"
        if not baseline.is_file():
            raise FileNotFoundError(f"missing seed-matched baseline: {baseline}")
        sample = copy.deepcopy(pilot)
        sample.update(
            {
                "seed": seed,
                "category": "Legacy object-query five-seed replication",
                "baseline_video": str(baseline),
                "sample_group": "object_query_001460_5seed",
                "matrices": {"s039": [], "all_steps_mean": []},
            }
        )
        samples.append(sample)

    payload = {
        **source,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": None,
        "seeds": list(SEEDS),
        "samples": samples,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT}")
    print(f"samples={len(samples)} entries={len(payload['entries'])}")


if __name__ == "__main__":
    main()
