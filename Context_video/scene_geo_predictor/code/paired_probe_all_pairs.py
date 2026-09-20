"""Re-evaluate every unordered control-door pair with actual Predictor batches.

This is a zero-shot audit of the S1 checkpoints.  The control group is marked
consumed development data; no value here is used for fitting or selection.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_cpu_fit import load_case_rows, make_batch
from future_query_predictor import Predictor
from prepare_small_trial import sha

CONTROL_KEYS = [f"control_door_{i:02d}" for i in range(4)]
SEED = 20260919
ARMS = {
    "motion_only": ("motion_only", "estimated_aabb", "motion_only_step0500.pt"),
    "geometry_estimated_aabb": ("geometry_only", "estimated_aabb", "geometry_estimated_aabb_step0500.pt"),
    "geometry_estimated_sphere": ("geometry_only", "estimated_sphere", "geometry_estimated_sphere_step0500.pt"),
    "geometry_partial_oracle": ("geometry_only", "oracle_partial_visible_geometry", "geometry_partial_oracle_step0500.pt"),
}
NON_SCENE_INPUTS = ("motion", "object_mask", "last_position", "last_velocity", "future_dt")


def cache_path(root: Path, mode: str, case: str) -> Path:
    base = root / mode / case
    if (base / "scene_tokens.npz").exists():
        return base
    nested = root / mode / mode / case
    if (nested / "scene_tokens.npz").exists():
        return nested
    raise FileNotFoundError(f"no control cache for {mode}/{case}")


def pair_delta(pred_a, pred_b, gt_a, gt_b):
    gt_diff = np.linalg.norm(gt_a - gt_b, axis=-1)
    pred_diff = np.linalg.norm(pred_a - pred_b, axis=-1)
    err = np.linalg.norm((pred_a - pred_b) - (gt_a - gt_b), axis=-1)
    d_gt = float(gt_diff.mean())
    return {"D_gt_m": d_gt, "D_pred_m": float(pred_diff.mean()),
            "E_delta_m": float(err.mean()),
            "R_delta": float(err.mean() / d_gt) if d_gt > 1e-8 else None,
            "D_gt_max_m": float(gt_diff.max()), "D_pred_max_m": float(pred_diff.max()),
            "nondegenerate": bool(d_gt > 1e-8)}


def width_for(root: Path, case: str) -> float:
    meta = json.loads((root / "samples" / case / "metadata.json").read_text())
    return float(meta["scenario_spec"]["door_opening_width_m"])


def write_csv(path: Path, rows):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    train_keys = [f"train_door_{x}" for x in ("0712", "0660", "0366", "0437", "0955", "0891")]
    train_rows = load_case_rows(args.train_root, args.train_cache / "estimated_aabb", train_keys)
    train_batch, _ = make_batch(train_rows)
    mean = train_batch["motion"].mean((0, 1))
    std = train_batch["motion"].std((0, 1), unbiased=False).clamp_min(1e-3)

    # Door-width range is computed from all 30 train metadata records, not from
    # the previous S2 summary or from the controlled probe.
    manifest = json.loads((args.train_root / "manifest.json").read_text())
    train_manifest_keys = [r["key"] for r in manifest["records"]
                           if r.get("family") == "door" and r.get("split") == "train"]
    train_widths = [width_for(args.train_root, k) for k in train_manifest_keys]
    train_min, train_max = min(train_widths), max(train_widths)
    widths = {case: width_for(args.controlled_root, case) for case in CONTROL_KEYS}

    predictions, batches, targets = {}, {}, {}
    input_checks = {}
    weight_hashes = {}
    for arm, (scene_mode, mode, weight_name) in ARMS.items():
        rows = load_case_rows(args.controlled_root, args.pair_cache, CONTROL_KEYS) if mode == "task" else None
        # The control pair cache root is mode-specific; load rows directly.
        scene_root = args.pair_cache / mode
        if not (scene_root / CONTROL_KEYS[0] / "report.json").exists():
            scene_root = args.pair_cache / mode / mode
        rows = load_case_rows(args.controlled_root, scene_root, CONTROL_KEYS)
        batch, target = make_batch(rows)
        batches[arm], targets[arm] = batch, target
        checks = {}
        for field in NON_SCENE_INPUTS:
            reference = batch[field][0]
            if batch[field].dtype == torch.bool:
                diffs = [float((batch[field][i] ^ reference).sum()) for i in range(1, len(CONTROL_KEYS))]
            else:
                diffs = [float((batch[field][i] - reference).abs().max()) for i in range(1, len(CONTROL_KEYS))]
            checks[field] = {"max_abs_diff": max(diffs or [0.0]), "per_case_abs_diff": diffs,
                             "exact": bool(max(diffs or [0.0]) == 0.0)}
        input_checks[arm] = checks
        model = Predictor(mean, std, scene_mode=scene_mode, seed=SEED, strict_future_grid=True).cpu()
        weight_path = args.weights / weight_name
        model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
        weight_hashes[arm] = {"path": str(weight_path.resolve()), "sha256": sha(weight_path)}
        with torch.no_grad():
            predictions[arm] = model(batch).numpy()[:, 0]

    pair_rows = []
    for arm in ARMS:
        pred = predictions[arm]
        target = targets[arm].numpy()[:, 0]
        for i, j in itertools.combinations(range(len(CONTROL_KEYS)), 2):
            row = {"arm": arm, "case_a": CONTROL_KEYS[i], "case_b": CONTROL_KEYS[j],
                   "width_a_m": widths[CONTROL_KEYS[i]], "width_b_m": widths[CONTROL_KEYS[j]],
                   "width_a_in_train_range": train_min <= widths[CONTROL_KEYS[i]] <= train_max,
                   "width_b_in_train_range": train_min <= widths[CONTROL_KEYS[j]] <= train_max,
                   "both_widths_in_train_range": (train_min <= widths[CONTROL_KEYS[i]] <= train_max and
                                                   train_min <= widths[CONTROL_KEYS[j]] <= train_max),
                   "train_width_min_m": train_min, "train_width_max_m": train_max}
            row.update(pair_delta(pred[i], pred[j], target[i], target[j]))
            pair_rows.append(row)

    report = {
        "schema": "paired_probe_all_pairs_v1", "status": "EXECUTED",
        "role": "zero-shot development audit; group consumed and not held out",
        "development_group_consumed": True, "cases": CONTROL_KEYS,
        "pair_count": 6, "train_case_count_for_width_range": len(train_manifest_keys),
        "train_widths_m": train_widths, "train_width_min_m": train_min,
        "train_width_max_m": train_max, "control_widths_m": widths,
        "input_checks": input_checks, "weight_hashes": weight_hashes,
        "pairs": pair_rows,
        "same_history_expected": True,
        "interpretation_limit": "No independent pair group; no generalization claim.",
        "motion_only_expected": "D_pred=0 and R_delta=1 for exact shared inputs",
    }
    (args.output / "paired_probe_all_pairs.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "paired_probe_all_pairs.csv", pair_rows)
    print(json.dumps({"status": report["status"], "pair_count": 6,
                      "train_width_range_m": [train_min, train_max],
                      "input_checks": input_checks,
                      "pair_summary": {arm: [r for r in pair_rows if r["arm"] == arm] for arm in ARMS}}, indent=2))


if __name__ == "__main__":
    main()
