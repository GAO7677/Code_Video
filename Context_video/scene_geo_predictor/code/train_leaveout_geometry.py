"""Matched train/development-val CPU comparison for four predictor arms."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from diagnostic_cpu_fit import attach_events, load_case_rows, make_batch, metrics, subset
from future_query_predictor import Predictor, objective
from prepare_small_trial import sha

SEED = 20260919
ARMS = {
    "motion_only": ("motion_only", "estimated_aabb"),
    "geometry_estimated_aabb": ("geometry_only", "estimated_aabb"),
    "geometry_estimated_sphere": ("geometry_only", "estimated_sphere"),
    "geometry_partial_oracle": ("geometry_only", "oracle_partial_visible_geometry"),
}


def rows_for(root, cache_root, mode, cases):
    rows = load_case_rows(root, cache_root / mode, cases)
    for row in rows:
        report = row["scene_report"]
        actual = report.get("alignment_mode")
        if actual != mode:
            raise ValueError(f"cache alignment mode mismatch for {row['case']}: {actual} != {mode}")
        if not report.get("scene_features_placeholder"):
            raise ValueError(f"non-placeholder scene features in geometry cache: {row['case']}")
    return rows


def validate_matched(rows_by_mode):
    modes = list(rows_by_mode)
    base = rows_by_mode[modes[0]]
    for mode in modes[1:]:
        other = rows_by_mode[mode]
        for a, b in zip(base, other):
            if a["case"] != b["case"]:
                raise ValueError("case order differs across geometry arms")
            # Target and observed motion must be bitwise/near-bitwise shared.
            if not np.allclose(a["positions"], b["positions"], atol=1e-7, rtol=0):
                raise ValueError(f"motion differs across cache arms: {a['case']}")
            if not np.allclose(a["target"], b["target"], atol=1e-7, rtol=0):
                raise ValueError(f"target differs across cache arms: {a['case']}")
            if not np.array_equal(a["scene_mask"], b["scene_mask"]):
                raise ValueError(f"matched support mask differs: {a['case']}")
            if a.get("source_flat_indices") is None or b.get("source_flat_indices") is None:
                raise ValueError(f"matched cache lacks source_flat_indices: {a['case']}")
            if not np.array_equal(a["source_flat_indices"], b["source_flat_indices"]):
                raise ValueError(f"matched support source ids differ: {a['case']}")
            if a.get("source_frame_y_x") is None or b.get("source_frame_y_x") is None:
                raise ValueError(f"matched cache lacks source_frame_y_x: {a['case']}")
            if not np.array_equal(a["source_frame_y_x"], b["source_frame_y_x"]):
                raise ValueError(f"matched support pixel/frame ids differ: {a['case']}")
            # Geometry is expected to differ; only source ids and validity are
            # held fixed across the diagnostic arms.
    for i in range(len(base)):
        ids = [rows_by_mode[m][i]["scene_report"].get("common_support_count") for m in modes]
        if len(set(ids)) != 1:
            raise ValueError(f"common support count differs for {base[i]['case']}: {ids}")


@torch.no_grad()
def evaluate_arm(model, batch, target, rows):
    model.eval()
    prediction = model(batch)
    return metrics(prediction, target, batch, rows), prediction


def train_one(name, mode, batch, target, rows, mean, std, init_state, steps, batch_size=2):
    model = Predictor(mean, std, scene_mode=mode, seed=SEED, strict_future_grid=True).cpu()
    model.load_state_dict(init_state, strict=True)
    optimizer = torch.optim.AdamW(model.active_parameters(), lr=3e-4, weight_decay=.01)
    with torch.no_grad():
        initial_loss = float(objective(model(batch), target, batch))
    losses = [initial_loss]
    start = time.monotonic()
    batches_per_epoch = len(rows) // batch_size
    for step in range(1, steps + 1):
        generator = torch.Generator().manual_seed(SEED + (step - 1) // batches_per_epoch)
        order = torch.randperm(len(rows), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for begin in range(0, len(rows), batch_size):
            ids = order[begin:begin + batch_size].tolist()
            part, labels = subset(batch, target, ids)
            loss = objective(model(part), labels, part) * len(ids) / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} step {step}: nonfinite loss")
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(list(model.active_parameters()), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(total)
    final_loss = float(objective(model(batch), target, batch))
    return model, {"name": name, "mode": mode, "steps": steps, "effective_batch": batch_size,
                   "initial_loss": initial_loss, "final_loss": final_loss,
                   "loss_reduction_fraction": float(1 - final_loss / initial_loss),
                   "elapsed_seconds": time.monotonic() - start,
                   "active_parameters": int(sum(p.numel() for p in model.active_parameters())),
                   "parameters": int(sum(p.numel() for p in model.parameters())),
                   "loss_curve": losses}


def add_case_rows(summary, metric_obj, split):
    summary["metrics"] = metric_obj
    summary["split"] = split


def pairwise_delta(pred_a, pred_b, gt_a, gt_b, threshold=1e-8):
    gt_delta = np.linalg.norm(gt_a - gt_b, axis=-1)
    pred_delta = np.linalg.norm(pred_a - pred_b, axis=-1)
    e_delta = np.linalg.norm((pred_a - pred_b) - (gt_a - gt_b), axis=-1)
    denom = float(gt_delta.mean())
    return {"E_delta_m": float(e_delta.mean()), "D_gt_m": denom,
            "E_delta_over_D_gt": float(e_delta.mean() / denom) if denom > threshold else None,
            "pred_delta_mean_m": float(pred_delta.mean()),
            "gt_delta_nonzero": bool(denom > threshold),
            "denominator_threshold_m": threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 500:
        raise ValueError("leaveout CPU budget is 1..500 optimizer steps")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    train_keys = [r["key"] for r in manifest["records"] if r.get("family") == "door" and r.get("split") == "train"]
    val_keys = [r["key"] for r in manifest["records"] if r.get("family") == "door" and r.get("split") == "val"]
    rows_by_mode = {}
    for mode in ("estimated_aabb", "estimated_sphere", "oracle_partial_visible_geometry"):
        rows_by_mode[mode] = rows_for(args.root, args.cache_root, mode, train_keys)
    validate_matched(rows_by_mode)
    rows_by_mode_val = {mode: rows_for(args.root, args.cache_root, mode, val_keys)
                        for mode in rows_by_mode}
    validate_matched(rows_by_mode_val)
    attach_events(args.root, rows_by_mode["estimated_aabb"])
    attach_events(args.root, rows_by_mode["estimated_sphere"])
    attach_events(args.root, rows_by_mode["oracle_partial_visible_geometry"])
    attach_events(args.root, rows_by_mode_val["estimated_aabb"])
    attach_events(args.root, rows_by_mode_val["estimated_sphere"])
    attach_events(args.root, rows_by_mode_val["oracle_partial_visible_geometry"])
    train_batches = {m: make_batch(rows)[0:2] for m, rows in rows_by_mode.items()}
    val_batches = {m: make_batch(rows)[0:2] for m, rows in rows_by_mode_val.items()}
    # Same motion statistics and exact initial state for all arms.
    mean = train_batches["estimated_aabb"][0]["motion"].mean((0, 1))
    std = train_batches["estimated_aabb"][0]["motion"].std((0, 1), unbiased=False).clamp_min(1e-3)
    base = Predictor(mean, std, scene_mode="geometry_only", seed=SEED, strict_future_grid=True).cpu()
    init_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    output = args.output.resolve(); output.mkdir(parents=True)
    results, models, predictions = {}, {}, {}
    # Analytic CV is recorded as a fixed reference, not one of the four learned arms.
    with torch.no_grad():
        cv_train = train_batches["estimated_aabb"][0]["last_position"][:, :, None] + train_batches["estimated_aabb"][0]["last_velocity"][:, :, None] * train_batches["estimated_aabb"][0]["future_dt"][:, None, :, None]
        cv_val = val_batches["estimated_aabb"][0]["last_position"][:, :, None] + val_batches["estimated_aabb"][0]["last_velocity"][:, :, None] * val_batches["estimated_aabb"][0]["future_dt"][:, None, :, None]
    cv_summary = {"name": "analytic_cv", "steps": 0, "split": "train", "metrics": metrics(cv_train, train_batches["estimated_aabb"][1], train_batches["estimated_aabb"][0], rows_by_mode["estimated_aabb"])}
    cv_val_summary = {"name": "analytic_cv", "steps": 0, "split": "development_val", "metrics": metrics(cv_val, val_batches["estimated_aabb"][1], val_batches["estimated_aabb"][0], rows_by_mode_val["estimated_aabb"])}
    results["analytic_cv"] = {"train": cv_summary, "development_val": cv_val_summary}
    predictions["analytic_cv"] = {"train": cv_train.numpy(), "development_val": cv_val.numpy()}
    for name, (model_mode, cache_mode) in ARMS.items():
        model, fit = train_one(name, model_mode, train_batches[cache_mode][0], train_batches[cache_mode][1], rows_by_mode[cache_mode], mean, std, init_state, args.steps)
        train_metrics, train_pred = evaluate_arm(model, train_batches[cache_mode][0], train_batches[cache_mode][1], rows_by_mode[cache_mode])
        val_metrics, val_pred = evaluate_arm(model, val_batches[cache_mode][0], val_batches[cache_mode][1], rows_by_mode_val[cache_mode])
        fit["train_metrics"] = train_metrics; fit["development_val_metrics"] = val_metrics
        state_path = output / f"{name}_step{args.steps:04d}.pt"
        torch.save(model.state_dict(), state_path)
        fit["final_weight_path"] = str(state_path)
        fit["final_weight_sha256"] = sha(state_path)
        fit["cache_mode"] = cache_mode
        fit["cache_report_hashes_train"] = {r["case"]: sha(args.cache_root / cache_mode / r["case"] / "report.json") for r in rows_by_mode[cache_mode]}
        fit["cache_report_hashes_val"] = {r["case"]: sha(args.cache_root / cache_mode / r["case"] / "report.json") for r in rows_by_mode_val[cache_mode]}
        results[name] = {"fit": fit, "train": {"metrics": train_metrics}, "development_val": {"metrics": val_metrics}}
        predictions[name] = {"train": train_pred.numpy(), "development_val": val_pred.numpy()}
        models[name] = model
        print("ARM_COMPLETE", name, "train_ADE", train_metrics["ADE_m"], "val_ADE", val_metrics["ADE_m"], flush=True)
        del model
    np.savez_compressed(output / "leaveout_predictions.npz",
                        **{f"{arm}_{split}": values for arm, splits in predictions.items() for split, values in splits.items()},
                        train_target=train_batches["estimated_aabb"][1].numpy(),
                        val_target=val_batches["estimated_aabb"][1].numpy())
    report = {"schema": "future_query_geometry_leaveout_v1", "status": "EXECUTED",
              "protocol": "matched_geometry_diagnostic", "privileged_support": True,
              "train_keys": train_keys, "development_val_keys": val_keys,
              "split_source": str(args.manifest.resolve()), "steps": args.steps,
              "effective_batch": 2, "threads": 2, "seed": SEED, "optimizer": "AdamW",
              "learning_rate": 3e-4, "weight_decay": .01,
              "loss": "SmoothL1(position)+0.2*SmoothL1(interval_velocity)",
              "legacy_query_reference": True, "dit_loaded": False, "utonia_extracted": False,
              "same_initial_state": True, "initial_state_sha256": hashlib_sha_state(init_state),
              "cache_root": str(args.cache_root.resolve()), "arms": results,
              "interpretation": "Development-val is not blind; metrics use one fixed final step and no val-based checkpoint selection.",
              "evidence_limit": "No OOD claim; same-history controlled pairs are a separate development probe."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "steps": args.steps,
                      "val_ADE": {k: v["development_val"]["metrics"]["ADE_m"] for k, v in results.items()}}, indent=2))


def hashlib_sha_state(state):
    import hashlib
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


if __name__ == "__main__":
    main()
