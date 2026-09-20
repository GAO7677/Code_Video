"""Sampling-order, sampler-provenance, and batch-semantics checks.

This is a CPU-only preflight for the dense paired diagnostic.  It never writes
scene caches and never changes predictor inputs beyond the declared permutation
test.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from build_task_geometry_caches import DEFAULT_M, candidate_pool, choose
from diagnostic_cpu_fit import load_case_rows, make_batch, subset
from future_query_predictor import Predictor, objective
from prepare_small_trial import sha

CASES = [f"control_door_{i:02d}" for i in range(4)]
S1_BASE_SEED = 20260920
S1_RESAMPLE_SEEDS = [20260921, 20260922, 20260923]
PERMUTATION_SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]
TOKEN_COUNT = 1792
M = 512
FEATURE_DIM = 1386
SEED = 20260919
EXPECTED_QUOTA = {
    "door_frame_left": 192,
    "door_frame_right": 192,
    "other_task_surfaces": 128,
}


def load_dense(project: Path, seed: int):
    controlled = project / "controlled_scene_eval_12"
    scene_root = (project / "validation_20260920/task_geometry_caches"
                  / "dense_critical_task_geometry" / f"seed{seed}")
    rows = load_case_rows(controlled, scene_root, CASES)
    batch, target = make_batch(rows)
    return rows, batch, target


def motion_stats(project: Path):
    train_cases = [f"train_door_{x}" for x in
                   ("0712", "0660", "0366", "0437", "0955", "0891")]
    rows = load_case_rows(
        project / "small_trial_120",
        project / "validation_20260920/geometry_caches_matched/estimated_aabb",
        train_cases,
    )
    batch, _ = make_batch(rows)
    return batch["motion"].mean((0, 1)), batch["motion"].std(
        (0, 1), unbiased=False
    ).clamp_min(1e-3)


def load_model(project: Path):
    mean, std = motion_stats(project)
    model = Predictor(mean, std, scene_mode="geometry_only",
                      seed=SEED, strict_future_grid=True).cpu()
    weight = (project / "validation_20260920/paired_task_fit_1000"
              / "geometry_dense_task_step1000.pt")
    state = torch.load(weight, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, mean, std, weight


def permuted_batch(batch: dict[str, torch.Tensor], seed: int,
                   full_slots: bool) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in batch.items()}
    rng = np.random.default_rng(seed)
    for i in range(batch["scene_xyz"].shape[0]):
        if full_slots:
            order = rng.permutation(TOKEN_COUNT)
        else:
            valid = int(batch["scene_mask"][i].sum().item())
            order = np.concatenate((rng.permutation(valid),
                                    np.arange(valid, TOKEN_COUNT)))
        index = torch.as_tensor(order, dtype=torch.long)
        for key in ("scene_xyz", "scene_features", "scene_mask"):
            out[key][i] = batch[key][i, index]
    return out


def distance_summary(reference: torch.Tensor, value: torch.Tensor):
    diff = (reference - value).float()
    return {
        "max_abs_m": float(diff.abs().max()),
        "mean_trajectory_l2_m": float(torch.linalg.vector_norm(
            diff, dim=-1).mean()),
    }


def permutation_check(model, batch):
    with torch.no_grad():
        repeats = [model(batch) for _ in range(3)]
        repeat_summary = distance_summary(repeats[0], repeats[1])
        repeat_summary["third_repeat_max_abs_m"] = float(
            (repeats[0] - repeats[2]).abs().max())
        base = repeats[0]
        modes = {}
        for mode, full_slots in (("valid_only", False),
                                 ("full_slots_with_padding", True)):
            rows = []
            for seed in PERMUTATION_SEEDS:
                value = model(permuted_batch(batch, seed, full_slots))
                summary = distance_summary(base, value)
                summary["seed"] = seed
                rows.append(summary)
            modes[mode] = {
                "per_seed": rows,
                "max_abs_m": max(x["max_abs_m"] for x in rows),
                "max_mean_trajectory_l2_m": max(
                    x["mean_trajectory_l2_m"] for x in rows),
                "tolerance_m": 1e-5,
                "passed": all(x["max_abs_m"] <= 1e-5 for x in rows),
            }
    return {
        "repeat_forward": repeat_summary,
        "permutation_modes": modes,
        "status": ("PASS" if repeat_summary["max_abs_m"] <= 1e-7
                   and all(x["passed"] for x in modes.values())
                   else "NUMERICAL_REVIEW"),
        "interpretation": (
            "Synchronous permutation changes only token order. "
            "Resampling is reported separately and is not used for this pass."
        ),
    }


def sampler_check(project: Path):
    root = project / "controlled_scene_eval_12"
    task_root = (project / "validation_20260920/task_geometry_caches"
                 / "dense_critical_task_geometry")
    checks = []
    failures = []
    for case in CASES:
        independently = candidate_pool(root, case)
        pool_xyz, pool_collider, key_mask, names, boxes, metadata, pool_hash = independently
        case_rows = []
        selected_sets = {}
        for seed in [S1_BASE_SEED] + S1_RESAMPLE_SEEDS:
            report_path = task_root / f"seed{seed}" / case / "report.json"
            scene_path = report_path.parent / "scene_tokens.npz"
            report = json.loads(report_path.read_text())
            with np.load(scene_path, allow_pickle=False) as archive:
                xyz = archive["scene_xyz"]
                mask = archive["scene_mask"].astype(bool)
                source = archive["source_task_pool_index"].astype(np.int64)
                collider = archive["source_collider_id"].astype(np.int64)
            valid = int(mask.sum())
            expected_xyz = pool_xyz[source[:valid]]
            xyz_error = float(np.max(np.abs(xyz[:valid] - expected_xyz)))
            expected_collider = pool_collider[source[:valid]]
            row = {
                "case": case,
                "seed": seed,
                "candidate_pool_hash_matches": report["candidate_pool_hash"] == pool_hash,
                "candidate_pool_count": int(len(pool_xyz)),
                "candidate_pool_unique_xyz_rounded_1e-6": int(
                    len(np.unique(np.round(pool_xyz, 6), axis=0))),
                "valid_count": valid,
                "padded_slots": int((~mask).sum()),
                "quota": report["quota"],
                "quota_matches": report["quota"] == EXPECTED_QUOTA,
                "xyz_source_mapping_max_abs_m": xyz_error,
                "source_collider_matches": bool(
                    np.array_equal(collider[:valid], expected_collider)),
                "unique_source_pool": int(len(np.unique(source[:valid]))),
                "unique_valid_xyz_exact": int(len(np.unique(
                    xyz[:valid], axis=0))),
                "unique_valid_xyz_rounded_1e-6": int(len(np.unique(
                    np.round(xyz[:valid], 6), axis=0))),
                "future_flags": {
                    key: bool(report[key]) for key in (
                        "future_rgb_used", "future_state_used",
                        "future_contact_used")
                },
                "geometry_scope": report["geometry_scope"],
            }
            row["passed"] = bool(
                row["candidate_pool_hash_matches"]
                and row["candidate_pool_count"] == 44608
                and row["candidate_pool_unique_xyz_rounded_1e-6"] == 40904
                and row["valid_count"] == M
                and row["padded_slots"] == TOKEN_COUNT - M
                and row["quota_matches"]
                and row["xyz_source_mapping_max_abs_m"] == 0.0
                and row["source_collider_matches"]
                and row["unique_source_pool"] == M
                and row["unique_valid_xyz_exact"] == M
                and not any(row["future_flags"].values())
            )
            checks.append(row)
            if not row["passed"]:
                failures.append(row)
            selected_sets[seed] = set(source[:valid].tolist())
            case_rows.append(row)
        same_pool = len({r["candidate_pool_hash_matches"] for r in case_rows}) == 1
        changed = [
            len(selected_sets[a] ^ selected_sets[b])
            for a, b in itertools.combinations(sorted(selected_sets), 2)
        ]
        same_seed = choose(pool_xyz, pool_collider, key_mask, names,
                           S1_BASE_SEED, "dense_critical_task_geometry", M)
        same_seed_reproducible = np.array_equal(
            same_seed,
            choose(pool_xyz, pool_collider, key_mask, names,
                   S1_BASE_SEED, "dense_critical_task_geometry", M))
        checks.append({
            "case": case,
            "pool_hash_constant_across_saved_seeds": same_pool,
            "different_seed_symmetric_difference_min": int(min(changed)),
            "different_seed_symmetric_difference_max": int(max(changed)),
            "same_seed_reproducible": bool(same_seed_reproducible),
            "candidate_pool_hash": pool_hash,
            "observed_static_box_count": len(boxes),
            "roi_definition": "all declared static OBB surfaces plus finite floor margin=2m",
            "passed": bool(same_pool and min(changed) > 0
                           and same_seed_reproducible),
        })
        if not checks[-1]["passed"]:
            failures.append(checks[-1])
    # Directly compare non-scene inputs and targets across saved seed variants.
    _, base_batch, base_target = load_dense(project, S1_BASE_SEED)
    input_checks = []
    for seed in S1_RESAMPLE_SEEDS:
        _, value_batch, value_target = load_dense(project, seed)
        fields = ("motion", "object_mask", "last_position", "last_velocity",
                  "future_dt")
        input_checks.append({
            "seed": seed,
            "non_scene_equal": {
                field: bool(torch.equal(base_batch[field], value_batch[field]))
                for field in fields
            },
            "target_equal": bool(torch.equal(base_target, value_target)),
        })
    failures.extend(
        x for x in input_checks
        if not all(x["non_scene_equal"].values()) or not x["target_equal"]
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "cases": checks,
        "input_checks": input_checks,
        "expected_quota": EXPECTED_QUOTA,
        "candidate_pool_count": 44608,
        "candidate_pool_unique_xyz_rounded_1e-6": 40904,
        "failures": failures,
        "notes": [
            "Seeds select points and order only; candidate pool and quota are fixed.",
            "Scene features are zero placeholders and geometry_only ignores them.",
            "Static OBB and finite floor definitions are observation-time task geometry.",
            "The sampler audit uses no future state or event to choose points.",
        ],
    }


def batch_semantics_check(project: Path):
    _, batch, target = load_dense(project, S1_BASE_SEED)
    mean, std = motion_stats(project)
    model = Predictor(mean, std, scene_mode="geometry_only",
                      seed=SEED, strict_future_grid=True).cpu()
    active = list(model.active_parameters())
    optimizer = torch.optim.AdamW(active, lr=3e-4, weight_decay=.01)
    full_loss = float(objective(model(batch), target, batch))
    optimizer.zero_grad(set_to_none=True)
    micro_losses = []
    backward_count = 0
    for ids in ([0, 1], [2, 3]):
        part, labels = subset(batch, target, ids)
        loss = objective(model(part), labels, part) * len(ids) / len(CASES)
        loss.backward()
        micro_losses.append(float(loss.detach()))
        backward_count += 1
    accumulated = float(sum(micro_losses))
    optimizer.step()
    return {
        "reported_old_effective_batch": 2,
        "actual_effective_batch": 4,
        "case_order": CASES,
        "microbatch_sizes": [2, 2],
        "backward_calls": backward_count,
        "optimizer_step_calls": 1,
        "loss_scale_per_microbatch": [0.5, 0.5],
        "full_batch_mean_loss_before_update": full_loss,
        "sum_scaled_microbatch_losses": accumulated,
        "absolute_loss_difference": abs(full_loss - accumulated),
        "target_formula": "(L_00 + L_01 + L_02 + L_03) / 4",
        "passed": bool(abs(full_loss - accumulated) <= 1e-7),
        "interpretation": (
            "S1 metadata called this effective batch 2, but its implementation "
            "processed all four cases before one optimizer.step()."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    project = args.project.resolve()
    model, _, _, weight = load_model(project)
    _, base_batch, _ = load_dense(project, S1_BASE_SEED)
    report = {
        "schema": "sampling_checks_v1",
        "status": "EXECUTED",
        "device": "cpu",
        "torch_threads": 2,
        "checkpoint": str(weight),
        "checkpoint_sha256": sha(weight),
        "permutation_check": permutation_check(model, base_batch),
        "sampler_check": sampler_check(project),
        "batch_semantics": batch_semantics_check(project),
        "predeclared_tolerance_m": 1e-5,
        "resampling_probe_seeds": S1_RESAMPLE_SEEDS,
        "interpretation_limit": (
            "The four controls are consumed development cases; this is not "
            "independent history generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": report["status"],
        "permutation": report["permutation_check"]["status"],
        "sampler": report["sampler_check"]["status"],
        "batch": report["batch_semantics"]["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()

