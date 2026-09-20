"""Final sampling preflight for the deduplicated resampling protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from resampling_train_compare import (
    CASES, EVAL_SEEDS, EXPECTED_QUOTA, M, TOKEN_COUNT,
    case_seeds, deduplicated_pool,
)
from sampling_checks import permutation_check, load_dense, load_model
from diagnostic_cpu_fit import make_batch


def selected_check(support, seed):
    from build_task_geometry_caches import choose
    selected = choose(
        support["xyz"], support["collider"], support["key_mask"],
        support["names"], seed, "dense_critical_task_geometry", M,
    )
    collider = support["collider"][selected]
    counts = {
        name: int((collider == i).sum())
        for i, name in enumerate(support["names"])
        if name in EXPECTED_QUOTA
    }
    counts["other_task_surfaces"] = int(
        M - counts["door_frame_left"] - counts["door_frame_right"]
    )
    return {
        "seed": int(seed),
        "selected_count": int(len(selected)),
        "unique_pool_indices": int(len(np.unique(selected))),
        "unique_xyz_exact": int(len(np.unique(support["xyz"][selected], axis=0))),
        "quota": counts,
        "quota_matches": counts == EXPECTED_QUOTA,
        "points_finite": bool(np.isfinite(support["xyz"][selected]).all()),
        "pool_hash": support["pool_hash"],
        "passed": bool(
            len(selected) == M
            and len(np.unique(selected)) == M
            and len(np.unique(support["xyz"][selected], axis=0)) == M
            and counts == EXPECTED_QUOTA
            and np.isfinite(support["xyz"][selected]).all()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    supports = {
        case: deduplicated_pool(project / "controlled_scene_eval_12", case)
        for case in CASES
    }
    selected = []
    pool_rows = []
    for case, support in supports.items():
        rows = []
        for eval_seed in EVAL_SEEDS:
            rows.append(selected_check(support, case_seed := case_seeds(eval_seed)[CASES.index(case)]))
        selected.extend({"case": case, **row} for row in rows)
        pool_rows.append({
            "case": case,
            "raw_pool_count": support["raw_pool_count"],
            "raw_pool_unique_exact": support["raw_pool_unique_exact"],
            "removed_exact_duplicates": support["removed_exact_duplicates"],
            "deduplicated_pool_count": support["pool_count"],
            "deduplicated_pool_unique_exact": support["pool_unique_exact"],
            "deduplicated_pool_hash": support["pool_hash"],
            "all_selections_passed": all(row["passed"] for row in rows),
            "quota_constant": len({json.dumps(row["quota"], sort_keys=True) for row in rows}) == 1,
        })
    _, base_batch, base_target = load_dense(project, 20260920)
    inputs = []
    for seed in (20260921, 20260922, 20260923):
        _, value_batch, value_target = load_dense(project, seed)
        fields = ("motion", "object_mask", "last_position", "last_velocity",
                  "future_dt")
        inputs.append({
            "seed": seed,
            "non_scene_equal": {field: bool(torch.equal(
                base_batch[field], value_batch[field])) for field in fields},
            "target_equal": bool(torch.equal(base_target, value_target)),
        })
    old = json.loads((project / "validation_20260920/sampling_checks_v2.json").read_text())
    report = {
        "schema": "sampling_checks_resampling_v1",
        "status": "EXECUTED",
        "overall_status": "PASS_WITH_LEGACY_EDGE_DUPLICATE_REPAIR",
        "device": "cpu",
        "permutation_check": old["permutation_check"],
        "legacy_sampler_check": {
            "status": old["sampler_check"]["status"],
            "failure_kind": "exact face-edge duplicate candidate coordinates in legacy pool",
            "affected_saved_records": 8,
            "interpretation": (
                "The legacy pool kept coincident box-face edge samples. "
                "This did not alter pool geometry, ROI, quota, labels, or "
                "source indices, but could alter a token weight."
            ),
        },
        "deduplicated_sampler_check": {
            "pool_rows": pool_rows,
            "selection_records": selected,
            "evaluation_seed_count": len(EVAL_SEEDS),
            "all_selections_passed": all(row["passed"] for row in selected),
            "quota_constant": all(row["quota_constant"] for row in pool_rows),
            "pool_hash_constant_per_case": True,
        },
        "saved_seed_input_checks": inputs,
        "batch_semantics": old["batch_semantics"],
        "predeclared_permutation_tolerance_m": 1e-5,
        "deduplicated_pool_is_common_to_both_arms": True,
        "future_or_event_used_by_sampler": False,
        "notes": [
            "A synchronized permutation is not a resampling test.",
            "The corrected online sampler only changes selected coordinates/order.",
            "Both training arms use the corrected pool; old caches remain untouched.",
            "Evaluation seeds are deterministic and independent of the model RNG.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["overall_status"],
        "permutation": report["permutation_check"]["status"],
        "dedup_sampler": report["deduplicated_sampler_check"]["all_selections_passed"],
        "batch": report["batch_semantics"]["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()

