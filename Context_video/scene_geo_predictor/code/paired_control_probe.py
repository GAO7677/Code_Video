"""Check same-history/different-door-geometry behaviour on a development probe.

The four controlled door cases were generated with one identical observed
motion and re-simulated futures.  They are deliberately reported as a
development probe, not an independent held-out group.  Predictor inputs use
observation-only caches; future states are read only for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_cpu_fit import load_case_rows, make_batch
from future_query_predictor import Predictor
from penetration_decomposition import colliders_for, distances
from prepare_small_trial import sha

SEED = 20260919
CONTROL_KEYS = [f"control_door_{i:02d}" for i in range(4)]
ARMS = {
    "motion_only": ("motion_only", "estimated_aabb"),
    "geometry_estimated_aabb": ("geometry_only", "estimated_aabb"),
    "geometry_estimated_sphere": ("geometry_only", "estimated_sphere"),
    "geometry_partial_oracle": ("geometry_only", "oracle_partial_visible_geometry"),
}


def _history_and_physics(root: Path, case: str):
    sample = root / "samples" / case
    metadata = json.loads((sample / "metadata.json").read_text())
    with np.load(sample / "raw" / "states_xyzw.npz", allow_pickle=False) as data:
        names = data["object_names"].astype(str).tolist()
        dynamic = [i for i, name in enumerate(names) if metadata["actors"][name].get("dynamic")]
        if len(dynamic) != 1:
            raise ValueError(f"expected one dynamic actor for {case}")
        d = dynamic[0]
        actor = metadata["actors"][names[d]]
        history = {"positions": data["positions"][:8, d].copy(),
                   "velocities": data["linear_velocities"][:8, d].copy(),
                   "quats": data["quats"][:8, d].copy(),
                   "frame_times": data["frame_times"][:8].copy()}
        static = {}
        for i, name in enumerate(names):
            actor_data = metadata["actors"][name]
            if actor_data.get("dynamic"):
                continue
            static[name] = {"size_m": actor_data.get("size_m"),
                            "friction": actor_data.get("friction"),
                            "restitution": actor_data.get("restitution"),
                            "position": data["positions"][0, i].astype(float).tolist()}
    physics = {k: actor.get(k) for k in ("mass_kg", "friction", "restitution", "shape", "size_m")}
    return history, physics, static, metadata


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def first_obstacle_contact(root, case, trajectory, threshold=.005):
    colliders, radius = colliders_for(root, case)
    d = distances(trajectory, radius, colliders)[:, 1:]
    hit = (d <= threshold).any(axis=1)
    prior = np.r_[False, hit[:-1]]
    starts = np.flatnonzero(hit & ~prior)
    return int(8 + starts[0]) if len(starts) else None


def pair_delta(pred_a, pred_b, gt_a, gt_b):
    gt_delta = np.linalg.norm(gt_a - gt_b, axis=-1)
    pred_delta = np.linalg.norm(pred_a - pred_b, axis=-1)
    residual = np.linalg.norm((pred_a - pred_b) - (gt_a - gt_b), axis=-1)
    d = float(gt_delta.mean())
    return {"E_delta_m": float(residual.mean()), "D_gt_m": d,
            "E_delta_over_D_gt": float(residual.mean() / d) if d > 1e-8 else None,
            "pred_delta_mean_m": float(pred_delta.mean()),
            "gt_max_delta_m": float(gt_delta.max()), "gt_nonzero": bool(d > 1e-8)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cache_root = args.controlled_root.parent / "validation_20260920" / "controlled_pair_caches"
    # Use the exact training statistic and arm initialization contract from the
    # matched leave-out run, not statistics computed from controlled futures.
    train_keys = [f"train_door_{x}" for x in ("0712", "0660", "0366", "0437", "0955", "0891")]
    train_rows = load_case_rows(args.train_root, args.train_cache / "estimated_aabb", train_keys)
    train_batch, _ = make_batch(train_rows)
    mean = train_batch["motion"].mean((0, 1))
    std = train_batch["motion"].std((0, 1), unbiased=False).clamp_min(1e-3)
    reference_history, reference_physics, reference_static, metadata = _history_and_physics(args.controlled_root, CONTROL_KEYS[0])
    protocol = {"pair_id": "controlled_door_same_history_v1", "blind": False,
                "role": "development_probe_already used in prior geometry analysis",
                "future_re_simulated": True, "observation_frames": list(range(8)),
                "future_frames": list(range(8, 49)), "history_tolerances": {"exact": 0.0}}
    pair_rows = []
    control_rows = {}
    for case in CONTROL_KEYS:
        history, physics, static, meta = _history_and_physics(args.controlled_root, case)
        with np.load(args.controlled_root / "samples" / case / "raw" / "states_xyzw.npz", allow_pickle=False) as labels:
            names = labels["object_names"].astype(str).tolist(); d = names.index("door_ball")
            future = labels["positions"][8:49, d].astype(np.float32)
        pair_rows.append({"case": case, "history_max_position_diff_m": max_abs(history["positions"], reference_history["positions"]),
                          "history_max_velocity_diff_mps": max_abs(history["velocities"], reference_history["velocities"]),
                          "history_max_quaternion_diff": max_abs(history["quats"], reference_history["quats"]),
                          "history_max_time_diff_s": max_abs(history["frame_times"], reference_history["frame_times"]),
                          "dynamic_physics_equal": physics == reference_physics,
                          "future_vs_reference_D_gt_m": None if case == CONTROL_KEYS[0] else None,
                          "opening_width_m": meta["scenario_spec"]["door_opening_width_m"],
                          "future": future})
        control_rows[case] = (history, physics, static, future)
    # Exact pair data checks and static-geometry change evidence.
    pair_rows[0]["future_vs_reference_D_gt_m"] = 0.0
    for row in pair_rows[1:]:
        row["future_vs_reference_D_gt_m"] = float(np.linalg.norm(row["future"] - pair_rows[0]["future"], axis=-1).mean())
    static_diffs = []
    for case in CONTROL_KEYS[1:]:
        _, _, static, _ = control_rows[case]
        pos_diffs = []
        size_diffs = []
        for name in reference_static:
            a, b = reference_static[name], static[name]
            if a.get("position") is not None and b.get("position") is not None:
                pos_diffs.append(max_abs(a["position"], b["position"]))
            if a.get("size_m") is not None and b.get("size_m") is not None:
                size_diffs.append(max_abs(list(a["size_m"].values()), list(b["size_m"].values())))
        static_diffs.append({"case": case, "max_static_position_diff_m": max(pos_diffs or [0.0]),
                             "max_static_size_diff_m": max(size_diffs or [0.0])})
    for row, diff in zip(pair_rows[1:], static_diffs): row.update(diff)
    for row in pair_rows: row.pop("future", None)
    if any(r["history_max_position_diff_m"] != 0 or r["history_max_velocity_diff_mps"] != 0 or
           r["history_max_quaternion_diff"] != 0 for r in pair_rows):
        raise ValueError("controlled cases do not have identical observed history")
    # Load controlled observation-only rows per geometry arm and run the fixed
    # final-step weights without any adaptation on the probe.
    predictions = {}
    pair_metrics = {}
    for arm, (scene_mode, cache_mode) in ARMS.items():
        cache_path = cache_root / cache_mode
        # The first controlled-cache build used a mode-specific output root
        # for the two estimated variants, leaving one extra directory level.
        # Accept that recorded diagnostic location without copying or changing
        # the cache contents; the oracle cache uses the normal layout.
        if not (cache_path / CONTROL_KEYS[0] / "report.json").exists():
            nested = cache_path / cache_mode
            if (nested / CONTROL_KEYS[0] / "report.json").exists():
                cache_path = nested
        rows = load_case_rows(args.controlled_root, cache_path, CONTROL_KEYS)
        batch, target = make_batch(rows)
        state_path = args.weights / ({"motion_only": "motion_only", "geometry_estimated_aabb": "geometry_estimated_aabb",
                                      "geometry_estimated_sphere": "geometry_estimated_sphere",
                                      "geometry_partial_oracle": "geometry_partial_oracle"}[arm] + "_step0500.pt")
        model = Predictor(mean, std, scene_mode=scene_mode, seed=SEED, strict_future_grid=True).cpu()
        model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad(): pred = model(batch).numpy()[:, 0]
        predictions[arm] = pred
        rows_for_arm = []
        for i in range(1, len(CONTROL_KEYS)):
            metric = pair_delta(pred[0], pred[i], target[0, 0].numpy(), target[i, 0].numpy())
            metric.update({"pair_id": protocol["pair_id"], "arm": arm,
                           "case_a": CONTROL_KEYS[0], "case_b": CONTROL_KEYS[i],
                           "width_a_m": pair_rows[0]["opening_width_m"],
                           "width_b_m": pair_rows[i]["opening_width_m"],
                           "contact_a_frame": first_obstacle_contact(args.controlled_root, CONTROL_KEYS[0], target[0, 0].numpy()),
                           "contact_b_frame": first_obstacle_contact(args.controlled_root, CONTROL_KEYS[i], target[i, 0].numpy()),
                           "pred_contact_a_frame": first_obstacle_contact(args.controlled_root, CONTROL_KEYS[0], pred[0]),
                           "pred_contact_b_frame": first_obstacle_contact(args.controlled_root, CONTROL_KEYS[i], pred[i])})
            rows_for_arm.append(metric)
        pair_metrics[arm] = rows_for_arm
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    report = {"schema": "controlled_pair_probe_v1", "status": "EXECUTED",
              "protocol": protocol, "pair_rows": pair_rows, "static_geometry_diffs": static_diffs,
              "pair_metrics": pair_metrics, "weights_root": str(args.weights.resolve()),
              "weight_sha256": {p.name: sha(p) for p in args.weights.glob("*_step0500.pt")},
              "controlled_manifest_sha256": sha(args.controlled_root / "manifest.json"),
              "interpretation_limit": "One controlled history group, previously analyzed; no held-out scene generalization claim.",
              "heldout_pair_groups": 0, "new_pair_generation": "not executed; GPU/rendering authorization not available"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "heldout_pair_groups": 0,
                      "arms": list(pair_metrics), "history_position_max": max(r["history_max_position_diff_m"] for r in pair_rows),
                      "gt_delta": {k: [x["D_gt_m"] for x in v] for k, v in pair_metrics.items()}}, indent=2))


if __name__ == "__main__":
    main()
