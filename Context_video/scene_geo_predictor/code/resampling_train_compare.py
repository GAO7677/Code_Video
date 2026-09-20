"""Dense fixed-vs-online-resampling CPU comparison.

The two arms share the same deduplicated candidate surface pool, quotas,
initialization, loss, optimizer, update schedule, and evaluation point sets.
The only training factor is whether the dense 512-point scene is fixed or
redrawn between optimizer steps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from build_task_geometry_caches import (
    FEATURE_DIM, TOKEN_COUNT, candidate_pool, choose,
)
from diagnostic_cpu_fit import load_case_rows, make_batch
from future_query_predictor import Predictor, objective
from prepare_small_trial import sha
from train_paired_task import evaluate_predictions, grad_norms, group_parameters
from train_paired_task import pair_delta, update_norms

CASES = [f"control_door_{i:02d}" for i in range(4)]
SEED = 20260919
M = 512
STEPS = 1000
TRAIN_SAMPLER_SEED = 20261101
EVAL_SEEDS = [20260920, 20260921, 20260922, 20260923] + list(
    range(20260924, 20260940)
)
LEGACY_EVAL_SEEDS = {20260920, 20260921, 20260922, 20260923}
CHECKPOINT_STEPS = (0, 100, 250, 500, 1000)
EXPECTED_QUOTA = {
    "door_frame_left": 192,
    "door_frame_right": 192,
    "other_task_surfaces": 128,
}


def pool_hash(xyz: np.ndarray, collider: np.ndarray) -> str:
    return hashlib.sha256(xyz.tobytes() + collider.tobytes()).hexdigest()


def deduplicated_pool(root: Path, case: str):
    raw = candidate_pool(root, case)
    xyz, collider, key_mask, names, boxes, metadata, raw_hash = raw
    # The legacy builder leaves coincident face-edge points in the candidate
    # pool.  Keep the first exact coordinate and its collider provenance.  This
    # common repair is applied to both arms, so it is not an arm factor.
    _, keep = np.unique(xyz, axis=0, return_index=True)
    keep = np.sort(keep)
    xyz = xyz[keep]
    collider = collider[keep]
    key_mask = key_mask[keep]
    return {
        "xyz": xyz,
        "collider": collider,
        "key_mask": key_mask,
        "names": names,
        "boxes": boxes,
        "metadata": metadata,
        "raw_pool_hash": raw_hash,
        "raw_pool_count": int(len(raw[0])),
        "raw_pool_unique_exact": int(len(np.unique(raw[0], axis=0))),
        "pool_hash": pool_hash(xyz, collider),
        "pool_count": int(len(xyz)),
        "pool_unique_exact": int(len(np.unique(xyz, axis=0))),
        "removed_exact_duplicates": int(len(raw[0]) - len(xyz)),
    }


def case_seeds(seed: int) -> list[int]:
    """Derive four streams from one eval seed, never from labels or geometry."""
    sequence = np.random.SeedSequence([int(seed), 0x53414D50])
    return [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in sequence.spawn(len(CASES))
    ]


def sample_row(template: dict, support: dict, seed: int) -> dict:
    selected = choose(
        support["xyz"], support["collider"], support["key_mask"],
        support["names"], int(seed), "dense_critical_task_geometry", M,
    )
    row = dict(template)
    xyz = np.zeros((TOKEN_COUNT, 3), dtype=np.float32)
    mask = np.zeros(TOKEN_COUNT, dtype=bool)
    confidence = np.zeros(TOKEN_COUNT, dtype=np.float32)
    source = np.full(TOKEN_COUNT, -1, dtype=np.int64)
    collider = np.full(TOKEN_COUNT, -1, dtype=np.int16)
    xyz[:M] = support["xyz"][selected]
    mask[:M] = True
    confidence[:M] = 1.0
    source[:M] = selected
    collider[:M] = support["collider"][selected]
    row["scene_xyz"] = xyz
    row["scene_features"] = np.zeros(
        (TOKEN_COUNT, FEATURE_DIM), dtype=np.float32
    )
    row["scene_mask"] = mask
    row["scene_confidence"] = confidence
    row["source_task_pool_index"] = source
    row["source_collider_id"] = collider
    row["sampling_seed"] = int(seed)
    return row


def sampled_rows(templates: list[dict], supports: list[dict],
                 seeds: list[int]) -> list[dict]:
    return [
        sample_row(template, support, seed)
        for template, support, seed in zip(templates, supports, seeds)
    ]


def load_templates(project: Path):
    controlled = project / "controlled_scene_eval_12"
    root = (project / "validation_20260920/task_geometry_caches"
            / "dense_critical_task_geometry" / "seed20260920")
    rows = load_case_rows(controlled, root, CASES)
    _, target = make_batch(rows)
    return rows, target


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


def initial_state(mean: torch.Tensor, std: torch.Tensor):
    model = Predictor(mean, std, scene_mode="geometry_only",
                      seed=SEED, strict_future_grid=True).cpu()
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def sample_batch(templates, supports, seeds):
    rows = sampled_rows(templates, supports, seeds)
    batch, target = make_batch(rows)
    return rows, batch, target


def run_accumulated_update(model, batch, target, order):
    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    for begin in (0, 2):
        ids = order[begin:begin + 2].tolist()
        index = torch.as_tensor(ids, dtype=torch.long)
        part = {key: value[index] for key, value in batch.items()}
        labels = target[index]
        loss = objective(model(part), labels, part) * len(ids) / len(CASES)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        losses.append(float(loss.detach()))
    return sum(losses)


def train_arm(name: str, resample: bool, templates, supports,
              target_ref, mean, std, init):
    model = Predictor(mean, std, scene_mode="geometry_only",
                      seed=SEED, strict_future_grid=True).cpu()
    model.load_state_dict(init, strict=True)
    active = list(model.active_parameters())
    optimizer = torch.optim.AdamW(active, lr=3e-4, weight_decay=.01)
    groups = group_parameters(model)
    fixed_seeds = case_seeds(20260920)
    seed_rng = np.random.default_rng(TRAIN_SAMPLER_SEED)
    seed_stream = hashlib.sha256()
    seed_preview = []
    with torch.no_grad():
        _, initial_batch, initial_target = sample_batch(
            templates, supports, fixed_seeds
        )
        initial = float(objective(model(initial_batch), initial_target,
                                  initial_batch))
    diagnostics = {}
    # Record a gradient diagnostic without changing parameters.
    order0 = torch.arange(len(CASES), dtype=torch.long)
    run_accumulated_update(model, initial_batch, initial_target, order0)
    diagnostics[0] = {
        "loss": initial,
        "gradient_norm": grad_norms(groups),
        "update_norm": {key: 0.0 for key in groups},
    }
    model.zero_grad(set_to_none=True)
    losses = [initial]
    started = time.monotonic()
    for step in range(1, STEPS + 1):
        if resample:
            seeds = [int(x) for x in seed_rng.integers(
                0, np.iinfo(np.uint32).max, size=len(CASES),
                dtype=np.uint64
            )]
        else:
            seeds = fixed_seeds
        if len(seed_preview) < 3:
            seed_preview.append({"step": step, "case_seeds": seeds})
        if resample:
            seed_stream.update(np.asarray(seeds, dtype=np.uint64).tobytes())
        order = torch.randperm(
            len(CASES),
            generator=torch.Generator().manual_seed(SEED + step),
        )
        _, batch, target = sample_batch(templates, supports, seeds)
        before = {id(parameter): parameter.detach().float().clone()
                  for parameter in active}
        total = run_accumulated_update(model, batch, target, order)
        gradients = grad_norms(groups)
        torch.nn.utils.clip_grad_norm_(
            active, 1., error_if_nonfinite=True
        )
        optimizer.step()
        losses.append(total)
        if step in CHECKPOINT_STEPS:
            with torch.no_grad():
                checkpoint_loss = float(objective(model(batch), target, batch))
            diagnostics[step] = {
                "loss": checkpoint_loss,
                "gradient_norm": gradients,
                "update_norm": update_norms(before, groups),
                "case_seeds": seeds,
            }
        if resample and step > STEPS - 3:
            seed_preview.append({"step": step, "case_seeds": seeds})
    model.eval()
    with torch.no_grad():
        _, final_batch, final_target = sample_batch(
            templates, supports, fixed_seeds
        )
        final_loss = float(objective(model(final_batch), final_target,
                                     final_batch))
    result = {
        "name": name,
        "resample_between_optimizer_steps": resample,
        "steps": STEPS,
        "microbatch_size": 2,
        "gradient_accumulation_microbatches": 2,
        "effective_batch": 4,
        "initial_loss": initial,
        "final_loss": final_loss,
        "loss_reduction_fraction": float(1 - final_loss / initial),
        "elapsed_seconds": time.monotonic() - started,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "active_parameters": int(sum(p.numel() for p in active)),
        "loss_curve": losses,
        "diagnostics": diagnostics,
        "fixed_case_seeds": fixed_seeds,
        "training_seed_rule": (
            "np.random.default_rng(20261101), four uint32 seeds drawn "
            "sequentially per optimizer step; independent from torch order RNG"
        ),
        "training_seed_stream_sha256": seed_stream.hexdigest(),
        "training_seed_preview": seed_preview,
        "geometry_pool_hashes": [support["pool_hash"] for support in supports],
    }
    return model, result


def eval_one(model, templates, supports, seed, root, target_ref):
    case_streams = case_seeds(seed)
    rows, batch, target = sample_batch(templates, supports, case_streams)
    if not torch.equal(target, target_ref):
        raise ValueError(f"target changed for eval seed {seed}")
    with torch.no_grad():
        prediction = model(batch).detach().numpy()[:, 0]
    evaluation = evaluate_predictions(
        root, rows, batch, target, prediction
    )
    evaluation["case_sampling_seeds"] = case_streams
    evaluation["point_pool_hashes"] = [s["pool_hash"] for s in supports]
    return evaluation, prediction


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def stability_rows(predictions: dict[str, dict[int, np.ndarray]]):
    result = []
    for arm, by_seed in predictions.items():
        ordered = sorted(by_seed)
        stack = np.stack([by_seed[seed] for seed in ordered], axis=0)
        for case_index, case in enumerate(CASES):
            pair_values = [
                float(np.linalg.norm(
                    stack[i, case_index] - stack[j, case_index], axis=-1
                ).mean())
                for i, j in itertools.combinations(range(len(ordered)), 2)
            ]
            result.append({
                "arm": arm,
                "case": case,
                "seed_count": len(ordered),
                "S_case_mean_m": float(np.mean(pair_values)),
                "S_case_p95_m": float(np.percentile(pair_values, 95)),
                "S_case_max_m": float(np.max(pair_values)),
                "threshold_m": .05 * .18,
                "passed_mean_threshold": bool(np.mean(pair_values) <= .05 * .18),
            })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True)
    root = project / "controlled_scene_eval_12"
    templates, target_ref = load_templates(project)
    supports = [
        deduplicated_pool(root, case) for case in CASES
    ]
    mean, std = motion_stats(project)
    init = initial_state(mean, std)
    fixed_model, fixed_fit = train_arm(
        "dense_fixed", False, templates, supports, target_ref,
        mean, std, init
    )
    resample_model, resample_fit = train_arm(
        "dense_resample_train", True, templates, supports, target_ref,
        mean, std, init
    )
    models = {"dense_fixed": fixed_model, "dense_resample_train": resample_model}
    predictions = {name: {} for name in models}
    evaluations = {name: {} for name in models}
    eval_manifest = []
    for seed in EVAL_SEEDS:
        streams = case_seeds(seed)
        eval_manifest.append({
            "seed": seed,
            "role": "legacy_development_probe" if seed in LEGACY_EVAL_SEEDS
                     else "new_unseen_sampling_seed",
            "case_sampling_seeds": streams,
        })
        for name, model in models.items():
            evaluation, prediction = eval_one(
                model, templates, supports, seed, root, target_ref
            )
            evaluations[name][str(seed)] = evaluation
            predictions[name][seed] = prediction
    stability = stability_rows(predictions)
    case_rows = []
    pair_rows = []
    negative_control = []
    for name, by_seed in evaluations.items():
        for seed, evaluation in by_seed.items():
            role = next(row["role"] for row in eval_manifest
                        if str(row["seed"]) == seed)
            for row in evaluation["case_rows"]:
                case_rows.append({
                    "arm": name, "seed": int(seed), "seed_role": role, **row
                })
            for row in evaluation["pairs"]:
                pair_rows.append({
                    "arm": name, "seed": int(seed), "seed_role": role, **row
                })
                if row["case_a"] == "control_door_02" and row["case_b"] == "control_door_03":
                    negative_control.append({
                        "arm": name, "seed": int(seed), "seed_role": role,
                        "D_gt_m": row["D_gt_m"], "D_pred_m": row["D_pred_m"],
                        "E_delta_m": row["E_delta_m"],
                        "threshold_m": .05 * .18,
                        "passed_threshold": row["D_pred_m"] <= .05 * .18,
                    })
    # Save the raw predictions before producing aggregate summaries.
    arrays = {"target": target_ref.numpy()}
    for arm, by_seed in predictions.items():
        for seed, value in by_seed.items():
            arrays[f"{arm}_seed{seed}"] = value
    np.savez_compressed(output / "predictions.npz", **arrays)
    write_csv(
        output / "resampling_case_metrics.csv", case_rows,
        ["arm", "seed", "seed_role", "case", "ADE_m", "FDE_m",
         "radius_normalized_ADE", "penetration_max_depth_m",
         "penetration_primary_frame_count", "penetration_primary_frame_rate",
         "penetration_primary_longest_continuous_frames"],
    )
    write_csv(
        output / "resampling_pair_metrics.csv", pair_rows,
        ["arm", "seed", "seed_role", "case_a", "case_b", "D_gt_m",
         "D_pred_m", "E_delta_m", "R_delta", "nondegenerate"],
    )
    write_csv(
        output / "sampling_stability.csv", stability,
        ["arm", "case", "seed_count", "S_case_mean_m", "S_case_p95_m",
         "S_case_max_m", "threshold_m", "passed_mean_threshold"],
    )
    gradient_rows = []
    for fit in (fixed_fit, resample_fit):
        for step, value in fit["diagnostics"].items():
            for module in value["gradient_norm"]:
                gradient_rows.append({
                    "arm": fit["name"], "step": step, "module": module,
                    "gradient_norm": value["gradient_norm"][module],
                    "update_norm": value["update_norm"][module],
                })
    write_csv(
        output / "resampling_gradient_update_metrics.csv", gradient_rows,
        ["arm", "step", "module", "gradient_norm", "update_norm"],
    )
    protocol = {
        "schema": "resampling_train_protocol_v1",
        "status": "EXECUTED",
        "device": "cpu",
        "threads": 2,
        "steps": STEPS,
        "microbatch_size": 2,
        "gradient_accumulation_microbatches": 2,
        "effective_batch": 4,
        "loss": "SmoothL1(position)+0.2*SmoothL1(interval_velocity)",
        "query_reference": "legacy_last_position",
        "M": M,
        "padded_tokens": TOKEN_COUNT,
        "seed_model": SEED,
        "training_sampler_seed": TRAIN_SAMPLER_SEED,
        "evaluation_seeds": eval_manifest,
        "same_forward_scene_for_all_41_queries": True,
        "resampling_between_optimizer_steps_only": True,
        "fixed_branch": {
            "mode": "trained",
            "reuse_s1": False,
            "reason": (
                "S1 candidate pool retained exact face-edge duplicates; "
                "this run uses a deduplicated common pool, so both arms "
                "were retrained from the shared initialization."
            ),
        },
        "old_s1_weight": str(
            project / "validation_20260920/paired_task_fit_1000"
            / "geometry_dense_task_step1000.pt"
        ),
        "initial_state_sha256": sha_state(init),
        "source_hashes": {
            "resampling_script": sha(Path(__file__)),
            "legacy_builder": sha(project / "build_task_geometry_caches.py"),
        },
        "motion_statistics_source": str(
            project / "small_trial_120"
        ),
        "future_used_for_geometry": False,
        "dit_loaded": False,
        "utonia_extracted": False,
        "deduplicated_pool": {
            case: {
                key: support[key] for key in (
                    "raw_pool_hash", "raw_pool_count", "raw_pool_unique_exact",
                    "pool_hash", "pool_count", "pool_unique_exact",
                    "removed_exact_duplicates",
                )
            } for case, support in zip(CASES, supports)
        },
        "old_s1_batch_erratum": (
            "S1 config said effective_batch=2. Its actual loop processed "
            "four cases (two microbatches of two), scaled each by 1/2, and "
            "called optimizer.step once; the corrected protocol records 4."
        ),
    }
    (output / "resampling_train_protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    report = {
        "schema": "resampling_fit_report_v1",
        "status": "EXECUTED",
        "formal_training": False,
        "development_group_consumed": True,
        "cases": CASES,
        "device": "cpu",
        "protocol": protocol,
        "training": {
            "dense_fixed": fixed_fit,
            "dense_resample_train": resample_fit,
        },
        "evaluation": {
            "case_metrics_csv": str(output / "resampling_case_metrics.csv"),
            "pair_metrics_csv": str(output / "resampling_pair_metrics.csv"),
            "stability_csv": str(output / "sampling_stability.csv"),
            "stability": stability,
            "negative_control_02_03": negative_control,
            "evaluations": evaluations,
        },
        "interpretation_limit": (
            "All evaluation seeds reuse four consumed physics cases; no "
            "independent history or OOD claim is supported."
        ),
    }
    (output / "resampling_fit_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for model in models.values():
        del model
    print(json.dumps({
        "status": report["status"],
        "steps": STEPS,
        "arms": {
            "dense_fixed": fixed_fit["final_loss"],
            "dense_resample_train": resample_fit["final_loss"],
        },
        "eval_seeds": len(EVAL_SEEDS),
    }, indent=2))


def sha_state(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()

