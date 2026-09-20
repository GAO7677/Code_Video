"""Bounded CPU paired-fit for privileged sparse/dense task geometry."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from diagnostic_cpu_fit import attach_events, load_case_rows, make_batch, metrics, subset
from future_query_predictor import Predictor, objective
from penetration_decomposition import colliders_for, depth_stats, distances
from prepare_small_trial import sha

SEED = 20260919
CASES = [f"control_door_{i:02d}" for i in range(4)]
ARMS = {
    "motion_only": ("motion_only", "sparse_critical_task_geometry"),
    "geometry_sparse_task": ("geometry_only", "sparse_critical_task_geometry"),
    "geometry_dense_task": ("geometry_only", "dense_critical_task_geometry"),
}
CHECKPOINT_STEPS = (0, 100, 250, 500, 1000)


def group_parameters(model):
    groups = {"motion_encoder": list(model.motion_encoder.parameters()),
              "geometry_attention": list(model.reader.parameters()),
              "position_head": list(model.position.parameters()),
              "time_encoder": list(model.time_encoder.parameters())}
    return groups


def norm_parameters(parameters):
    values = [p.detach().float().norm() ** 2 for p in parameters]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.0


def grad_norms(groups):
    output = {}
    for name, parameters in groups.items():
        values = [p.grad.detach().float().norm() ** 2 for p in parameters if p.grad is not None]
        output[name] = float(torch.sqrt(torch.stack(values).sum())) if values else 0.0
    return output


def update_norms(before, groups):
    output = {}
    for name, parameters in groups.items():
        values = []
        for p in parameters:
            if id(p) in before:
                values.append((p.detach().float() - before[id(p)]).norm() ** 2)
        output[name] = float(torch.sqrt(torch.stack(values).sum())) if values else 0.0
    return output


def run_accumulated_step(model, batch, target, order):
    model.train()
    optimizer_loss = 0.0
    model.zero_grad(set_to_none=True)
    for begin in (0, 2):
        ids = order[begin:begin + 2].tolist()
        part, labels = subset(batch, target, ids)
        loss = objective(model(part), labels, part) * len(ids) / len(CASES)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite paired-fit loss")
        loss.backward()
        optimizer_loss += float(loss.detach())
    return optimizer_loss


def train_arm(name, scene_mode, batch, target, mean, std, init_state, steps):
    model = Predictor(mean, std, scene_mode=scene_mode, seed=SEED, strict_future_grid=True).cpu()
    model.load_state_dict(init_state, strict=True)
    active = list(model.active_parameters())
    optimizer = torch.optim.AdamW(active, lr=3e-4, weight_decay=.01)
    groups = group_parameters(model)
    snapshots = {}
    with torch.no_grad():
        initial = float(objective(model(batch), target, batch))
    # Step 0 gradient evidence is collected without changing the model.
    order0 = torch.arange(len(CASES), dtype=torch.long)
    run_accumulated_step(model, batch, target, order0)
    snapshots[0] = {"loss": initial, "gradient_norm": grad_norms(groups),
                    "update_norm": {k: 0.0 for k in groups}}
    model.zero_grad(set_to_none=True)
    losses = [initial]
    start = time.monotonic()
    for step in range(1, steps + 1):
        order = torch.randperm(len(CASES), generator=torch.Generator().manual_seed(SEED + step))
        before = {id(p): p.detach().float().clone() for p in active}
        total = run_accumulated_step(model, batch, target, order)
        gradients = grad_norms(groups)
        torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(total)
        if step in CHECKPOINT_STEPS:
            snapshots[step] = {"loss": float(objective(model(batch), target, batch)),
                               "gradient_norm": gradients,
                               "update_norm": update_norms(before, groups)}
    model.eval()
    with torch.no_grad():
        prediction = model(batch); final_loss = float(objective(prediction, target, batch))
    return model, {"name": name, "scene_mode": scene_mode, "steps": steps,
                   "effective_batch": 2, "gradient_accumulation_microbatches": 2,
                   "initial_loss": initial, "final_loss": final_loss,
                   "loss_reduction_fraction": float(1 - final_loss / initial),
                   "elapsed_seconds": time.monotonic() - start,
                   "parameters": int(sum(p.numel() for p in model.parameters())),
                   "active_parameters": int(sum(p.numel() for p in active)),
                   "loss_curve": losses, "diagnostics": snapshots}, prediction


def pair_delta(pred_a, pred_b, gt_a, gt_b):
    gt = np.linalg.norm(gt_a - gt_b, axis=-1)
    pred = np.linalg.norm(pred_a - pred_b, axis=-1)
    err = np.linalg.norm((pred_a - pred_b) - (gt_a - gt_b), axis=-1)
    d = float(gt.mean())
    return {"D_gt_m": d, "D_pred_m": float(pred.mean()), "E_delta_m": float(err.mean()),
            "R_delta": float(err.mean() / d) if d > 1e-8 else None,
            "D_gt_max_m": float(gt.max()), "D_pred_max_m": float(pred.max()),
            "nondegenerate": bool(d > 1e-8)}


def evaluate_predictions(root, rows, batch, target, predictions):
    attach_events(root, rows)
    batch_metrics = metrics(torch.from_numpy(predictions[:, None]), target, batch, rows)
    case_rows = []
    target_np = target.numpy()[:, 0]
    for i, case in enumerate(CASES):
        pred = predictions[i]
        err = np.linalg.norm(pred - target_np[i], axis=-1)
        case_row = {"case": case, "ADE_m": float(err.mean()), "FDE_m": float(err[-1]),
                    "radius_normalized_ADE": float(err.mean() / rows[i]["radius_m"]),
                    "radius_m": rows[i]["radius_m"]}
        colliders, radius = colliders_for(root, case)
        signed = distances(pred, radius, colliders)
        case_row.update({f"penetration_{k}": v for k, v in depth_stats(signed.min(axis=1)).items()})
        case_rows.append(case_row)
    pairs = []
    for i, j in itertools.combinations(range(len(CASES)), 2):
        row = {"case_a": CASES[i], "case_b": CASES[j]}
        row.update(pair_delta(predictions[i], predictions[j], target_np[i], target_np[j]))
        pairs.append(row)
    return {"aggregate": batch_metrics, "case_rows": case_rows, "pairs": pairs}


def load_variant(root, mode, seed, controlled_root):
    scene_root = root / mode / f"seed{seed}"
    rows = load_case_rows(controlled_root, scene_root, CASES)
    batch, target = make_batch(rows)
    return rows, batch, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.steps > 1000:
        raise ValueError("paired task budget is 1..1000 steps")
    torch.set_num_threads(2)
    project = args.project.resolve(); task_root = args.task_root.resolve(); controlled_root = project / "controlled_scene_eval_12"
    args.output.mkdir(parents=True)
    train_cases = [f"train_door_{x}" for x in ("0712", "0660", "0366", "0437", "0955", "0891")]
    train_batch, _ = make_batch(load_case_rows(project / "small_trial_120", args.train_cache / "estimated_aabb", train_cases))
    mean = train_batch["motion"].mean((0, 1)); std = train_batch["motion"].std((0, 1), unbiased=False).clamp_min(1e-3)
    # Ensure all three arms receive the same observed context and target.
    base_rows, base_batch, base_target = load_variant(task_root, "sparse_critical_task_geometry", 20260920, controlled_root)
    dense_rows, dense_batch, dense_target = load_variant(task_root, "dense_critical_task_geometry", 20260920, controlled_root)
    for field in ("motion", "object_mask", "last_position", "last_velocity", "future_dt"):
        if field == "object_mask":
            if not torch.equal(base_batch[field], dense_batch[field]): raise ValueError(f"paired input mismatch: {field}")
        elif not torch.equal(base_batch[field], dense_batch[field]):
            raise ValueError(f"paired input mismatch: {field}")
    if not torch.equal(base_target, dense_target): raise ValueError("paired target mismatch")
    base_model = Predictor(mean, std, scene_mode="geometry_only", seed=SEED, strict_future_grid=True).cpu()
    init_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    output = args.output.resolve(); results = {}; predictions_npz = {"target": base_target.numpy()}
    models = {}
    for name, (scene_mode, mode) in ARMS.items():
        rows, batch, target = (base_rows, base_batch, base_target) if mode == "sparse_critical_task_geometry" else (dense_rows, dense_batch, dense_target)
        model, fit, _ = train_arm(name, scene_mode, batch, target, mean, std, init_state, args.steps)
        pred = model(batch).detach().numpy()[:, 0]
        fit["base_eval"] = evaluate_predictions(controlled_root, rows, batch, target, pred)
        weight_path = output / f"{name}_step{args.steps:04d}.pt"; torch.save(model.state_dict(), weight_path)
        fit["weight_path"] = str(weight_path); fit["weight_sha256"] = sha(weight_path)
        fit["geometry_mode"] = mode; fit["base_seed"] = 20260920
        fit["paired_fit_status"] = "FIT_WITHIN_BUDGET" if (
            all(r["radius_normalized_ADE"] <= .1 for r in fit["base_eval"]["case_rows"])
            and all((r["R_delta"] is None or r["R_delta"] <= .25) for r in fit["base_eval"]["pairs"] if r["nondegenerate"])
        ) else "NOT_FIT_WITHIN_BUDGET"
        # Evaluate three unseen deterministic sampling seeds with the same
        # model and targets; no fitting or seed selection is performed.
        fit["resampling"] = {}
        for seed in (20260921, 20260922, 20260923):
            res_rows, res_batch, res_target = load_variant(task_root, mode, seed, controlled_root)
            if not torch.equal(res_target, target): raise ValueError(f"resample target mismatch: {name}/{seed}")
            with torch.no_grad(): res_pred = model(res_batch).numpy()[:, 0]
            ev = evaluate_predictions(controlled_root, res_rows, res_batch, res_target, res_pred)
            diff = np.linalg.norm(res_pred - pred, axis=-1)
            ev["prediction_vs_base_mean_l2_m"] = float(diff.mean())
            ev["prediction_vs_base_max_l2_m"] = float(diff.max())
            fit["resampling"][str(seed)] = ev
            predictions_npz[f"{name}_seed{seed}"] = res_pred
        predictions_npz[f"{name}_base"] = pred
        results[name] = fit; models[name] = model
        del model
        print("PAIRED_ARM_COMPLETE", name, fit["paired_fit_status"], "final_loss", fit["final_loss"], flush=True)
    config = {"schema": "paired_task_fit_config_v1", "steps": args.steps, "seed": SEED,
              "threads": 2, "effective_batch": 2, "gradient_accumulation_microbatches": 2,
              "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": .01,
              "loss": "SmoothL1(position)+0.2*SmoothL1(interval_velocity)",
              "query_reference": "legacy_last_position", "M": 512, "padded_tokens": 1792,
              "train_motion_statistics_source": str((project / "small_trial_120").resolve()),
              "base_task_cache_manifest": str((task_root / "manifest.json").resolve()),
              "resample_seeds": [20260921, 20260922, 20260923], "dit_loaded": False,
              "utonia_extracted": False, "future_used_for_geometry": False,
              "shared_initial_state": True, "initial_state_sha256": sha_state(init_state)}
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "predictions.npz", **predictions_npz)
    report = {"schema": "paired_task_fit_report_v1", "status": "EXECUTED",
              "protocol": "paired_fit_only/development_group_consumed", "cases": CASES,
              "results": results, "config": config,
              "interpretation_limit": "Four same-history cases can be memorized; no independent history generalization claim.",
              "engineering_targets": {"per_case_ADE_le_0.1_radius": True,
                                      "pair_R_delta_le_0.25_for_D_gt_ge_0.05_radius": True,
                                      "radius_m": 0.18},
              "source_weight_hashes": {}}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "arms": {k: {"status": v["paired_fit_status"], "final_loss": v["final_loss"], "base_ADE": v["base_eval"]["aggregate"]["ADE_m"]} for k,v in results.items()}}, indent=2))


def sha_state(state):
    import hashlib
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
