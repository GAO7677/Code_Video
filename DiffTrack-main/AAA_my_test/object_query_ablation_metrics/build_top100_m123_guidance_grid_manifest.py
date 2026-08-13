#!/usr/bin/env python3
"""Extend the frozen three-case Stage-4 manifest with a controlled seed-42 arm."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
SOURCE = ROOT / "stage4_runtime/stage4_manifest.json"
OUTPUT_ROOT = ROOT / "training_free_top100_m23_guidance_v1"
OUTPUT = OUTPUT_ROOT / "guidance_grid_manifest.json"
CASES = (
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_000331_w001",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end",
)
SOURCE_SEED = 47326
NEW_SEED = 42


def main() -> None:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_rows = {
        str(row["case"]): row
        for row in manifest.get("samples", [])
        if int(row.get("seed", -1)) == SOURCE_SEED and str(row.get("case")) in CASES
    }
    if set(source_rows) != set(CASES):
        missing = sorted(set(CASES) - set(source_rows))
        raise RuntimeError(f"missing seed-{SOURCE_SEED} source rows: {missing}")

    samples = []
    for case in CASES:
        source = copy.deepcopy(source_rows[case])
        samples.append(source)
        seed42 = copy.deepcopy(source)
        seed42["seed"] = NEW_SEED
        seed42["baseline_video"] = str(
            OUTPUT_ROOT / "baselines" / case / f"seed_{NEW_SEED:05d}" / "generated.mp4"
        )
        seed42["sample_group"] = "training_free_guidance_seed42_control"
        seed42["matrix_ablation_root"] = None
        seed42["temporal_tube_ablation_root"] = None
        samples.append(seed42)

    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["experiment_id"] = "top100_m123_guidance_seed42_lambda_grid_v1"
    manifest["seeds"] = [SOURCE_SEED, NEW_SEED]
    manifest["samples"] = samples
    manifest["guidance_grid"] = {
        "cases": list(CASES),
        "seeds": [SOURCE_SEED, NEW_SEED],
        "flows": ["m1", "m2", "m3"],
        "pag_scales": [0.5, 1.0],
        "controlled": {
            "region": "object_A",
            "head_scope": "latest3350 Top100",
            "cfg_scale": 5.0,
            "sampling_steps": 40,
            "time_scope": "all_time",
            "num_frames": 49,
            "height": 704,
            "width": 1280,
            "fps": 30,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
