"""Recompute Phase 2 seed-42 train/dev metrics with the original vector pair R.

This command never optimizes a model and never emits locked-test predictions.
It replays eval-mode forwards from the already-frozen epoch 50/100/200
checkpoints because the earlier Phase 2 scalar report did not archive tracks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from evaluate_physvideo_phase2 import (
    analytic_prediction,
    build_obbs,
    enrich_records,
    event_for_trajectory,
    penetration_series,
    response_label,
)
from finite_surface_geometry import FiniteSurfacePredictor
from phase2_response_metrics import (
    EQUAL_D_PRED_THRESHOLD_M,
    PAIR_R_THRESHOLD,
    RADIUS_M,
    synthetic_selftests,
    vector_pair_metrics,
)
from run_physvideo_phase1_triarm_profile import (
    build_motion_stats,
    load_records,
    make_batch,
    make_models,
)


ARMS = ("motion_only", "point_geometry_resample", "finite_surface_geometry")
EPOCHS = (50, 100, 200)
SPLITS = ("train", "dev")
POINT_EVAL_SEEDS = tuple(range(16))
FPS = 30.0
STEPS = 41
PENETRATION_THRESHOLD_M = 0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def record_order(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda row: (row["family"], row["group_id"], float(row["geometry_value"])))


def epoch_seed_keys(epoch: str | int, arm: str) -> tuple[int, ...]:
    if arm == "point_geometry_resample":
        return POINT_EVAL_SEEDS
    return (0,)


def load_checkpoint_model(training_root: Path, arm: str, epoch: int, mean: torch.Tensor, std: torch.Tensor):
    models, _ = make_models(mean, std, seed=42)
    checkpoint = training_root / "seed_42" / f"{arm}_seed42_epoch{epoch}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    models[arm].load_state_dict(payload["state_dict"])
    models[arm].eval()
    return models[arm], checkpoint


def forward_predictions(records: list[dict], mean: torch.Tensor, std: torch.Tensor,
                        training_root: Path, output: Path) -> tuple[dict, dict, dict]:
    """Forward exactly the prior eval seeds, returning (predictions, hashes, stats)."""
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    checkpoint_hashes = {}
    forward_stats = []
    for epoch in EPOCHS:
        for arm in ARMS:
            model, checkpoint = load_checkpoint_model(training_root, arm, epoch, mean, std)
            checkpoint_hashes[f"{arm}/epoch{epoch}"] = {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
            seeds = epoch_seed_keys(epoch, arm)
            for eval_seed in seeds:
                values = np.empty((len(records), STEPS, 3), dtype=np.float32)
                actual_seed_values = []
                for index, record in enumerate(records):
                    actual_seed = (int(record["point_seed"]) + 70000 + int(eval_seed)
                                   if arm == "point_geometry_resample" else int(record["point_seed"]))
                    batch, _ = make_batch([record], arm, mean, std, seeds=[actual_seed])
                    with torch.no_grad():
                        values[index] = model(batch)[0, 0].detach().cpu().numpy().astype(np.float32)
                    actual_seed_values.append(actual_seed)
                predictions[(str(epoch), arm, int(eval_seed))] = values
                forward_stats.append({"epoch": epoch, "arm": arm, "eval_seed": eval_seed,
                                      "records": len(records), "actual_seed_min": min(actual_seed_values),
                                      "actual_seed_max": max(actual_seed_values), "device": "cpu", "eval_mode": True})
    targets = np.stack([record["target"] for record in records]).astype(np.float32)
    archive = {"record_keys": np.asarray([record["key"] for record in records]), "targets": targets}
    for (epoch, arm, seed), values in predictions.items():
        archive[f"pred_{arm}_e{epoch}_s{seed}"] = values
    np.savez_compressed(output / "predictions_train_dev_epoch50_100_200.npz", **archive)
    return predictions, checkpoint_hashes, {"forwards": forward_stats, "archive_sha256": sha256_file(output / "predictions_train_dev_epoch50_100_200.npz")}


def add_analytic_prediction(predictions: dict, records: list[dict]) -> None:
    predictions[("analytic_cv", "analytic_cv", 0)] = np.stack([analytic_prediction(record) for record in records]).astype(np.float32)


def velocity_values(positions: np.ndarray, last: np.ndarray) -> np.ndarray:
    before = np.concatenate((last[None], positions[:-1]), axis=0)
    return (positions - before) * FPS


def episode_metric(prediction: np.ndarray, target: np.ndarray, record: dict) -> dict:
    diff = np.linalg.norm(prediction - target, axis=-1)
    pv = velocity_values(prediction, record["positions"][7])
    tv = velocity_values(target, record["positions"][7])
    return {"ADE_m": float(diff.mean()), "FDE_m": float(diff[-1]),
            "ADE_over_r": float(diff.mean() / RADIUS_M),
            "interval_velocity_MAE_mps": float(np.abs(pv - tv).mean())}


def pair_infos(manifest: dict, records: list[dict]) -> list[dict]:
    by_group = {}
    for record in records:
        by_group.setdefault(record["group_id"], []).append(record)
    result = []
    for pair in manifest["pair_rows"]:
        if pair["split"] not in SPLITS:
            continue
        variants = by_group[pair["group_id"]]
        left = next(row for row in variants if float(row["geometry_value"]) == float(pair["geometry_a"]))
        right = next(row for row in variants if float(row["geometry_value"]) == float(pair["geometry_b"]))
        result.append({**pair, "response_layer": response_label(pair["response_layer"]),
                       "left_key": left["key"], "right_key": right["key"],
                       "history_index": left["history_index"],
                       "parent_history_id": f"parent_h{int(left['history_index']):02d}",
                       "pair_id": f"{pair['group_id']}|{pair['geometry_a']}|{pair['geometry_b']}"})
    return result


def build_episode_and_pair_rows(records: list[dict], predictions: dict, pairs: list[dict], output: Path):
    index = {record["key"]: i for i, record in enumerate(records)}
    targets = {record["key"]: record["target"].astype(np.float32) for record in records}
    episode_rows, pair_rows = [], []
    prediction_items = sorted(predictions.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2]))
    for (epoch, arm, eval_seed), array in prediction_items:
        if epoch == "analytic_cv":
            epoch_label = "analytic_cv"
        else:
            epoch_label = int(epoch)
        for record, prediction in zip(records, array):
            row = {"epoch": epoch_label, "arm": arm, "eval_seed": eval_seed,
                   "key": record["key"], "family": record["family"], "split": record["split"],
                   "group_id": record["group_id"], "history_index": record["history_index"],
                   "parent_history_id": f"parent_h{int(record['history_index']):02d}",
                   **episode_metric(prediction, targets[record["key"]], record)}
            episode_rows.append(row)
        for pair in pairs:
            left = array[index[pair["left_key"]]]
            right = array[index[pair["right_key"]]]
            metrics = vector_pair_metrics(left, right, targets[pair["left_key"]], targets[pair["right_key"]])
            pair_rows.append({"epoch": epoch_label, "arm": arm, "eval_seed": eval_seed,
                              "family": pair["family"], "split": pair["split"], "group_id": pair["group_id"],
                              "history_index": pair["history_index"], "parent_history_id": pair["parent_history_id"],
                              "pair_id": pair["pair_id"], "geometry_a": pair["geometry_a"], "geometry_b": pair["geometry_b"],
                              "response_layer": pair["response_layer"], "D_gt_manifest_m": pair["D_gt_m"],
                              "point_seed_a": next(r["point_seed"] for r in records if r["key"] == pair["left_key"]),
                              "point_seed_b": next(r["point_seed"] for r in records if r["key"] == pair["right_key"]),
                              **metrics})
    write_csv(output / "per_episode_seed_vector.csv", episode_rows)
    write_csv(output / "per_pair_seed_vector.csv", pair_rows)
    return episode_rows, pair_rows


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_response(episode_rows: list[dict], pair_rows: list[dict], output: Path) -> list[dict]:
    keys = sorted({(str(row["epoch"]), row["arm"], row["family"], row["split"]) for row in episode_rows},
                  key=lambda value: (value[0], value[1], value[2], value[3]))
    summary = []
    history_rows = []
    for epoch, arm, family, split in keys:
        cases = [row for row in episode_rows if str(row["epoch"]) == epoch and row["arm"] == arm and row["family"] == family and row["split"] == split]
        pairs = [row for row in pair_rows if str(row["epoch"]) == epoch and row["arm"] == arm and row["family"] == family and row["split"] == split]
        strong = [row for row in pairs if row["response_layer"] == "strong"]
        weak = [row for row in pairs if row["response_layer"] == "weak"]
        equal = [row for row in pairs if row["response_layer"] == "equal_future"]
        def unique_count(rows, field): return len({row[field] for row in rows})
        row = {"epoch": epoch, "arm": arm, "family": family, "split": split,
               "episode_row_count": len(cases), "episode_history_count": unique_count(cases, "group_id"),
               "eval_seed_count": unique_count(cases, "eval_seed"),
               "ADE_m_mean": mean_or_none([x["ADE_m"] for x in cases]),
               "FDE_m_mean": mean_or_none([x["FDE_m"] for x in cases]),
               "ADE_over_r_mean": mean_or_none([x["ADE_over_r"] for x in cases]),
               "interval_velocity_MAE_mps_mean": mean_or_none([x["interval_velocity_MAE_mps"] for x in cases]),
               "strong_pair_count": unique_count(strong, "pair_id"),
               "strong_pair_seed_count": len(strong),
               "strong_history_count": unique_count(strong, "group_id"),
               "strong_history_seed_count": len({(x["group_id"], x["eval_seed"]) for x in strong}),
               "strong_E_delta_mm_mean": mean_or_none([x["E_delta_m"] * 1000 for x in strong]),
               "strong_R_delta_mean": mean_or_none([x["R_delta"] for x in strong]),
               "strong_R_delta_p95": float(np.percentile([x["R_delta"] for x in strong], 95)) if strong else None,
               "strong_magnitude_ratio_mean": mean_or_none([x["response_magnitude_ratio"] for x in strong]),
               "strong_magnitude_error_mean": mean_or_none([x["response_magnitude_error"] for x in strong]),
               "strong_pass_fraction": mean_or_none([x["strong_response_pass"] for x in strong]),
               "weak_pair_count": unique_count(weak, "pair_id"), "weak_pair_seed_count": len(weak),
               "weak_history_count": unique_count(weak, "group_id"),
               "weak_E_delta_mm_mean": mean_or_none([x["E_delta_m"] * 1000 for x in weak]),
               "weak_R_delta_mean": mean_or_none([x["R_delta"] for x in weak]),
               "weak_magnitude_ratio_mean": mean_or_none([x["response_magnitude_ratio"] for x in weak]),
               "weak_magnitude_error_mean": mean_or_none([x["response_magnitude_error"] for x in weak]),
               "negative_pair_count": unique_count(equal, "pair_id"), "negative_pair_seed_count": len(equal),
               "negative_history_count": unique_count(equal, "group_id"),
               "equal_D_pred_mm_mean": mean_or_none([x["D_pred_m"] * 1000 for x in equal]),
               "equal_pass_fraction": mean_or_none([x["equal_future_pass"] for x in equal]),
               "pair_R_denominator_definition": "D_gt mean_t ||gt_A[t]-gt_B[t]||; vector E_delta/D_gt"}
        summary.append(row)
        for group_id in sorted({x["group_id"] for x in pairs}):
            group_pairs = [x for x in pairs if x["group_id"] == group_id]
            for layer in ("strong", "weak", "equal_future"):
                layer_rows = [x for x in group_pairs if x["response_layer"] == layer]
                if not layer_rows:
                    continue
                history_rows.append({"epoch": epoch, "arm": arm, "family": family, "split": split,
                                     "group_id": group_id, "history_index": layer_rows[0]["history_index"],
                                     "parent_history_id": layer_rows[0]["parent_history_id"],
                                     "response_layer": layer, "pair_count": len({x["pair_id"] for x in layer_rows}),
                                     "pair_seed_count": len(layer_rows),
                                     "E_delta_mm_mean": mean_or_none([x["E_delta_m"] * 1000 for x in layer_rows]),
                                     "R_delta_mean": mean_or_none([x["R_delta"] for x in layer_rows]),
                                     "magnitude_ratio_mean": mean_or_none([x["response_magnitude_ratio"] for x in layer_rows]),
                                     "magnitude_error_mean": mean_or_none([x["response_magnitude_error"] for x in layer_rows]),
                                     "pass_fraction": mean_or_none([x["strong_response_pass"] if layer == "strong" else x["equal_future_pass"] if layer == "equal_future" else (x["R_delta"] is not None and x["R_delta"] <= PAIR_R_THRESHOLD) for x in layer_rows])})
    write_csv(output / "train_dev_response_summary.csv", summary)
    write_csv(output / "response_by_history.csv", history_rows)
    write_csv(output / "checkpoint_curves_vector.csv", [row for row in summary if row["epoch"] in {"50", "100", "200"}])
    return summary


def sampling_stability(predictions: dict, records: list[dict], output: Path) -> None:
    rows, summary = [], []
    for epoch in EPOCHS:
        values_by_seed = [predictions[(str(epoch), "point_geometry_resample", seed)] for seed in POINT_EVAL_SEEDS]
        for index, record in enumerate(records):
            values = np.stack([array[index] for array in values_by_seed], axis=0)
            center = values.mean(axis=0)
            center_values = np.linalg.norm(values - center[None], axis=-1)
            pairwise = [float(np.linalg.norm(values[i] - values[j], axis=-1).mean()) for i in range(16) for j in range(i)]
            seed0 = [float(np.linalg.norm(values[i] - values[0], axis=-1).mean()) for i in range(1, 16)]
            rows.append({"epoch": epoch, "key": record["key"], "family": record["family"], "split": record["split"],
                         "group_id": record["group_id"], "history_index": record["history_index"], "seed_count": 16,
                         "sampling_pair_count": len(pairwise), "S_center_m": float(center_values.mean()),
                         "S_pairwise_m": float(np.mean(pairwise)), "S_max_pair_m": float(np.max(pairwise)),
                         "S_seed0_m": float(np.mean(seed0)), "S_seed0_max_m": float(np.max(seed0)),
                         "threshold_5p5mm_comparison": "descriptive_only_UNRESOLVED"})
        for family in sorted({r["family"] for r in records}):
            for split in SPLITS:
                subset = [r for r in rows if r["epoch"] == epoch and r["family"] == family and r["split"] == split]
                summary.append({"epoch": epoch, "family": family, "split": split, "episode_count": len(subset),
                                "S_center_m_mean": mean_or_none([r["S_center_m"] for r in subset]),
                                "S_pairwise_m_mean": mean_or_none([r["S_pairwise_m"] for r in subset]),
                                "S_pairwise_m_p95": float(np.percentile([r["S_pairwise_m"] for r in subset], 95)) if subset else None,
                                "S_max_pair_m_max": max((r["S_max_pair_m"] for r in subset), default=None),
                                "S_seed0_m_mean": mean_or_none([r["S_seed0_m"] for r in subset]),
                                "threshold_status": "UNRESOLVED_for_S_center_and_pairwise"})
    write_csv(output / "sampling_stability_recomputed.csv", rows)
    write_csv(output / "sampling_stability_summary.csv", summary)


def event_category(frame):
    if frame is None:
        return "no_event"
    frame = int(frame)
    relative = (frame - 7) / FPS
    if frame <= 7:
        return "observation"
    if relative < 0.25:
        return "before_0p25"
    if relative <= 0.90:
        return "recommended_0p25_0p90"
    if frame <= 48:
        return "after_0p90_in_prediction_window"
    return "outside_prediction_window"


def event_coverage(manifest: dict, records: list[dict], episode_rows: list[dict], pair_rows: list[dict], output: Path) -> None:
    def group_layers(group_id):
        layers = [response_label(p["response_layer"]) for p in manifest["pair_rows"] if p["group_id"] == group_id and p["split"] in SPLITS]
        return sorted(set(layers)), layers.count("strong"), layers.count("weak")
    pair_lookup = {}
    for row in pair_rows:
        key = (str(row["epoch"]), row["arm"], int(row["eval_seed"]), row["group_id"])
        pair_lookup.setdefault(key, []).append(row)
    detailed = []
    deflector_records = [r for r in records if r["family"] == "deflector"]
    for record in deflector_records:
        obb_event = event_for_trajectory(record, record["target"], build_obbs(record))
        interaction = record["interaction"]
        layers, strong_count, weak_count = group_layers(record["group_id"])
        for row in episode_rows:
            if row["key"] != record["key"] or row["family"] != "deflector":
                continue
            group_pair_rows = pair_lookup.get((str(row["epoch"]), row["arm"], int(row["eval_seed"]), record["group_id"]), [])
            strong_pairs = [x for x in group_pair_rows if x["response_layer"] == "strong"]
            weak_pairs = [x for x in group_pair_rows if x["response_layer"] == "weak"]
            frame = interaction.get("first_future_frame")
            detailed.append({"epoch": row["epoch"], "arm": row["arm"], "eval_seed": row["eval_seed"], "key": record["key"],
                             "split": record["split"], "group_id": record["group_id"], "history_index": record["history_index"],
                             "parent_history_id": f"parent_h{int(record['history_index']):02d}",
                             "response_layers_present": json.dumps(layers), "strong_pair_count_in_group": strong_count,
                             "weak_pair_count_in_group": weak_count, "manifest_event_type": "target_collision_reference_x_proxy",
                             "target_collider": "puck_barrier", "manifest_event_frame": frame,
                             "manifest_event_relative_rgb7_s": None if frame is None else (int(frame) - 7) / FPS,
                             "manifest_event_category": event_category(frame),
                             "manifest_frames_after_event": None if frame is None else max(0, 49 - int(frame)),
                             "finite_obb_event_type": obb_event["kind"], "finite_obb_event_frame": obb_event.get("first_future_frame"),
                             "finite_obb_event_relative_rgb7_s": obb_event.get("first_future_time_after_rgb7_s"),
                             "finite_obb_event_category": event_category(obb_event.get("first_future_frame")),
                             "ADE_m": row["ADE_m"], "FDE_m": row["FDE_m"],
                             "strong_E_delta_mm_mean_in_group": mean_or_none([x["E_delta_m"] * 1000 for x in strong_pairs]),
                             "weak_E_delta_mm_mean_in_group": mean_or_none([x["E_delta_m"] * 1000 for x in weak_pairs])})
    write_csv(output / "deflector_event_coverage.csv", detailed)
    summary = []
    for category_field in ("manifest_event_category", "finite_obb_event_category"):
        for key in sorted({(r[category_field], str(r["epoch"]), r["arm"], r["split"]) for r in detailed}):
            category, epoch, arm, split = key
            subset = [r for r in detailed if r[category_field] == category and str(r["epoch"]) == epoch and r["arm"] == arm and r["split"] == split]
            summary.append({"event_source": category_field, "event_category": category, "epoch": epoch, "arm": arm, "split": split,
                            "episode_seed_count": len(subset), "unique_episode_count": len({r["key"] for r in subset}),
                            "history_count": len({r["group_id"] for r in subset}),
                            "ADE_m_mean": mean_or_none([r["ADE_m"] for r in subset]), "FDE_m_mean": mean_or_none([r["FDE_m"] for r in subset]),
                            "strong_E_delta_mm_mean": mean_or_none([r["strong_E_delta_mm_mean_in_group"] for r in subset]),
                            "weak_E_delta_mm_mean": mean_or_none([r["weak_E_delta_mm_mean_in_group"] for r in subset])})
    write_csv(output / "deflector_event_coverage_summary.csv", summary)
    histories = []
    for record in deflector_records:
        interaction = record["interaction"]
        layers, strong_count, weak_count = group_layers(record["group_id"])
        histories.append({"group_id": record["group_id"], "history_index": record["history_index"],
                          "parent_history_id": f"parent_h{int(record['history_index']):02d}", "split": record["split"],
                          "manifest_event_category": event_category(interaction.get("first_future_frame")),
                          "manifest_event_relative_rgb7_s": interaction.get("first_future_time_after_rgb7_s"),
                          "strong_pair_count": strong_count, "weak_pair_count": weak_count,
                          "response_layers": json.dumps(layers)})
    write_csv(output / "deflector_response_histories.csv", histories)


def longest_run(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def penetration_breakdown(predictions: dict, records: list[dict], output: Path) -> None:
    rows = []
    # Ground truth is emitted once; model rows retain every point seed.
    items = [("ground_truth", "ground_truth", 0, {r["key"]: r["target"].astype(np.float32) for r in records})]
    for key, values in predictions.items():
        epoch, arm, seed = key
        items.append((epoch, arm, seed, {record["key"]: values[i] for i, record in enumerate(records)}))
    for epoch, arm, seed, trajectories in items:
        for record in records:
            obbs = build_obbs(record)
            by_collider = penetration_series(trajectories[record["key"]], obbs)
            for collider, depth in by_collider.items():
                positive = depth[depth > 0]
                rows.append({"epoch": epoch, "arm": arm, "eval_seed": seed, "truth": "gt" if arm == "ground_truth" else "pred",
                             "key": record["key"], "family": record["family"], "split": record["split"],
                             "group_id": record["group_id"], "history_index": record["history_index"], "collider": collider,
                             "frame_denominator": len(depth), "max_depth_m": float(depth.max(initial=0.0)),
                             "mean_positive_depth_m": float(positive.mean()) if len(positive) else 0.0,
                             "p95_positive_depth_m": float(np.percentile(positive, 95)) if len(positive) else 0.0,
                             "positive_frame_count": int((depth > 0).sum()), "frames_gt5mm": int((depth > PENETRATION_THRESHOLD_M).sum()),
                             "frame_gt5mm_rate": float(np.mean(depth > PENETRATION_THRESHOLD_M)),
                             "case_gt5mm": bool(np.any(depth > PENETRATION_THRESHOLD_M)),
                             "longest_gt5mm_run_frames": longest_run(depth > PENETRATION_THRESHOLD_M),
                             "first_gt5mm_future_index": None if not np.any(depth > PENETRATION_THRESHOLD_M) else int(np.flatnonzero(depth > PENETRATION_THRESHOLD_M)[0])})
    write_csv(output / "penetration_case_breakdown.csv", rows)
    summary = []
    for key in sorted({(str(r["epoch"]), r["arm"], r["truth"], r["family"], r["split"], r["collider"]) for r in rows}):
        epoch, arm, truth, family, split, collider = key
        subset = [r for r in rows if (str(r["epoch"]), r["arm"], r["truth"], r["family"], r["split"], r["collider"]) == key]
        summary.append({"epoch": epoch, "arm": arm, "truth": truth, "family": family, "split": split, "collider": collider,
                        "unique_episode_denominator": len({r["key"] for r in subset}), "episode_seed_denominator": len(subset),
                        "frame_denominator": sum(int(r["frame_denominator"]) for r in subset),
                        "cases_gt5mm": sum(bool(r["case_gt5mm"]) for r in subset),
                        "frames_gt5mm": sum(int(r["frames_gt5mm"]) for r in subset),
                        "case_gt5mm_rate": float(np.mean([r["case_gt5mm"] for r in subset])),
                        "frame_gt5mm_rate": float(sum(int(r["frames_gt5mm"]) for r in subset) / sum(int(r["frame_denominator"]) for r in subset)),
                        "mean_max_depth_m": float(np.mean([r["max_depth_m"] for r in subset])),
                        "p95_case_max_depth_m": float(np.percentile([r["max_depth_m"] for r in subset], 95)),
                        "max_depth_m": float(max(r["max_depth_m"] for r in subset)),
                        "mean_longest_gt5mm_run_frames": float(np.mean([r["longest_gt5mm_run_frames"] for r in subset]))})
    write_csv(output / "penetration_summary.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--old-evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)

    manifest, all_records = load_records(args.data_root.resolve())
    enrich_records(all_records)
    records = record_order([record for record in all_records if record["split"] in SPLITS])
    train_records = [record for record in records if record["split"] == "train"]
    mean, std = build_motion_stats(train_records)
    pairs = pair_infos(manifest, records)
    predictions, checkpoint_hashes, forward_stats = forward_predictions(records, mean, std, args.training_root.resolve(), args.output)
    add_analytic_prediction(predictions, records)
    episode_rows, pair_rows = build_episode_and_pair_rows(records, predictions, pairs, args.output)
    summary = summarize_response(episode_rows, pair_rows, args.output)
    sampling_stability({key: value for key, value in predictions.items() if key[1] == "point_geometry_resample"}, records, args.output)
    event_coverage(manifest, records, episode_rows, pair_rows, args.output)
    penetration_breakdown(predictions, records, args.output)

    old_csv = args.old_evaluation_root / "per_pair_seed.csv"
    old_sampling = args.old_evaluation_root / "sampling_stability.csv"
    metric_definition = {"schema": "physvideo_phase2_response_metric_recheck_v1", "status": "EXECUTED",
                         "vector_definition": {"true_delta_t": "gt_A[t]-gt_B[t]", "pred_delta_t": "pred_A[t]-pred_B[t]",
                                                "D_gt": "mean_t ||true_delta_t||_2", "D_pred": "mean_t ||pred_delta_t||_2",
                                                "E_delta": "mean_t ||pred_delta_t-true_delta_t||_2", "R_delta": "E_delta/D_gt"},
                         "magnitude_definition": {"response_magnitude_ratio": "D_pred/D_gt",
                                                    "response_magnitude_error": "abs(D_pred-D_gt)/D_gt"},
                         "equal_future": {"R_delta": "N/A", "response_magnitude_ratio": "N/A",
                                           "response_magnitude_error": "N/A", "equal_future_absolute_prediction_delta": "D_pred"},
                         "thresholds": {"radius_m": RADIUS_M, "ADE_over_r": 0.1, "strong_R_delta": PAIR_R_THRESHOLD,
                                        "equal_D_pred_m": EQUAL_D_PRED_THRESHOLD_M},
                         "alignment": "fixed RGB8-RGB48 time index; manifest geometry_a then geometry_b; no swap/time shift/selection"}
    write_json(args.output / "metric_definition.json", metric_definition)
    write_json(args.output / "metric_old_new_mapping.json", {
        "status": "EXECUTED", "old_report": str(args.old_evaluation_root.resolve()),
        "old_columns": {"D_pred_m": "response_magnitude_D_pred", "E_delta_m": "abs(D_pred-D_gt) magnitude error",
                         "R_delta": "response_magnitude_ratio (old Phase2 table; not original vector R)"},
        "new_columns": {"D_pred_m": "kept amplitude field", "E_delta_m": "time-aligned vector E_delta",
                         "R_delta": "time-aligned vector E_delta/D_gt", "response_magnitude_ratio": "D_pred/D_gt",
                         "response_magnitude_error": "abs(D_pred-D_gt)/D_gt"},
        "old_pair_csv_sha256": sha256_file(old_csv), "old_sampling_csv_sha256": sha256_file(old_sampling),
        "old_full_prediction_archive": "NOT_AVAILABLE: v4 stored scalar CSV only; current run re-forwarded frozen checkpoints"})
    write_json(args.output / "metric_numeric_selftests.json", synthetic_selftests())
    write_json(args.output / "sampling_metric_definitions.json", {
        "status": "EXECUTED", "seed_count": 16, "pair_count": 120,
        "formulas": {"S_center": "mean_{s,t} ||pred[s,t]-mean_s pred[s,t]||_2",
                     "S_pairwise": "mean_{s<s',t} ||pred[s,t]-pred[s',t]||_2",
                     "S_max_pair": "max_{s<s'} mean_t ||pred[s,t]-pred[s',t]||_2",
                     "S_seed0": "mean_{s>0,t} ||pred[s,t]-pred[0,t]||_2"},
        "legacy_phase1_profile": {"file": "code/run_physvideo_phase1_triarm_profile.py", "field": "mean_pairwise_to_seed0_m",
                                   "formula": "mean_{s>0,t} distance to seed0; not S_center or all-pair S_pairwise"},
        "old_phase2_v4": {"file": "sampling_stability.csv", "mean_pairwise_m": "all 16 choose 2 pairs",
                          "mean_to_seed0_m": "seed0 reference"},
        "threshold_5p5mm": "UNRESOLVED for S_center/S_pairwise; values reported continuously and not called pass/fail"})
    write_json(args.output / "prediction_source.json", {"status": "EXECUTED", "source": "frozen seed42 checkpoints, eval mode, CPU",
                                                          "data_manifest_sha256": sha256_file(args.data_root / "pilot_manifest.json"),
                                                          "training_receipt_sha256": sha256_file(args.training_root / "training_receipt.json"),
                                                          "checkpoints": checkpoint_hashes, "forward_stats": forward_stats,
                                                          "locked_test_model_metrics": "NOT_RUN",
                                                          "prediction_archive_sha256": forward_stats["archive_sha256"]})
    write_json(args.output / "recheck_status.json", {"status": "EXECUTED", "seed": 42, "epochs": list(EPOCHS),
                                                       "train_records": len(train_records), "dev_records": len(records) - len(train_records),
                                                       "pair_rows_physical": len(pairs), "point_eval_seeds": list(POINT_EVAL_SEEDS),
                                                       "optimizer_updates": 0, "data_modified": False,
                                                       "locked_test_model_metrics": "NOT_RUN", "seed43_44": "NOT_RUN"})
    command = ("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
               f"/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B {Path(__file__).resolve()} "
               f"--data-root {args.data_root.resolve()} --training-root {args.training_root.resolve()} "
               f"--old-evaluation-root {args.old_evaluation_root.resolve()} --output {args.output.resolve()}\n")
    (args.output / "recheck_command.txt").write_text(command)
    print(json.dumps({"status": "EXECUTED", "output": str(args.output), "records": len(records),
                      "pairs": len(pairs), "epochs": list(EPOCHS), "locked_test_model_metrics": "NOT_RUN",
                      "optimizer_updates": 0}, indent=2))


if __name__ == "__main__":
    main()
