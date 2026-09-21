"""Phase 2 admission, finite-geometry evaluation, and seed-42 reporting.

The evaluator deliberately keeps the physics replay and the predictor protocol
separate.  It reconstructs static boxes from each episode's saved actor state,
uses a sphere--finite-OBB signed gap for penetration/event diagnostics, and
never evaluates model predictions on locked-test records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from finite_surface_geometry import SURFACE_TOKEN_COUNT, POINT_VALID_COUNT
from run_physvideo_phase1_triarm_profile import (
    analytic_cv,
    build_motion_stats,
    interval_velocity,
    load_records,
    make_batch,
    make_models,
)


FPS = 30.0
STEPS = 41
BALL_RADIUS_M = 0.11
EVENT_TOL_M = 0.005
EVENT_NEAR_TOL_M = 0.02
PENETRATION_REPORT_TOL_M = 0.005
ARMS = ("motion_only", "point_geometry_resample", "finite_surface_geometry")
EVAL_SPLITS = ("train", "dev")
CONTROL_FAMILIES = ("aperture", "deflector", "support_edge")


def response_label(value: str) -> str:
    """Normalize manifest's *_response labels to the reporting vocabulary."""
    value = str(value)
    if value in {"strong", "strong_response"}:
        return "strong"
    if value in {"weak", "weak_response"}:
        return "weak"
    if value == "equal_future":
        return value
    raise ValueError(f"unknown response layer: {value}")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def enrich_records(records: list[dict]) -> None:
    for record in records:
        raw = np.load(record["sample"] / "raw/states_xyzw.npz", allow_pickle=False)
        record["all_positions"] = raw["positions"].astype(np.float32)
        record["quats"] = raw["quats"].astype(np.float32)
        record["object_names"] = [str(value) for value in raw["object_names"].tolist()]
        record["metadata"] = json.loads((record["sample"] / "metadata.json").read_text())


def quat_matrix_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(value) for value in quat]
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def build_obbs(record: dict) -> list[dict]:
    """Rebuild finite static boxes from the saved replay, never an infinite floor."""
    positions = record["all_positions"]
    names = record["object_names"]
    actors = record["metadata"]["actors"]
    obbs = []
    for index, name in enumerate(names):
        actor = actors.get(name, {})
        if actor.get("dynamic", False) or actor.get("shape") != "box":
            continue
        size = actor.get("size_m", {})
        half = np.asarray([size["hx"], size["hy"], size["hz"]], dtype=np.float32)
        obbs.append({"name": name, "center": positions[0, index].astype(np.float32),
                     "rotation": quat_matrix_xyzw(record["quats"][0, index]),
                     "half": half})
    return obbs


def signed_distance_obb(points: np.ndarray, obb: dict) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float32) - obb["center"][None]
    local = delta @ obb["rotation"]
    q = np.abs(local) - obb["half"][None]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def penetration_series(points: np.ndarray, obbs: list[dict]) -> dict[str, np.ndarray]:
    return {obb["name"]: np.maximum(0.0, BALL_RADIUS_M - signed_distance_obb(points, obb))
            for obb in obbs}


def target_names(record: dict) -> set[str]:
    if record["family"] == "aperture":
        return {name for name in record["object_names"] if name.startswith("door_frame")}
    if record["family"] == "deflector":
        return {"puck_barrier"}
    return set()


def support_state(points: np.ndarray, obbs: list[dict]) -> np.ndarray:
    platforms = [obb for obb in obbs if obb["name"] in {"left_platform", "right_platform"}]
    result = np.zeros((len(points),), dtype=bool)
    for obb in platforms:
        local = (points - obb["center"][None]) @ obb["rotation"]
        inside_xy = ((np.abs(local[:, 0]) <= obb["half"][0] + BALL_RADIUS_M) &
                     (np.abs(local[:, 1]) <= obb["half"][1] + BALL_RADIUS_M))
        # The pilot platforms are horizontal finite boxes.  Keep this test
        # explicit so a support edge is not silently replaced by z=0 ground.
        top_z = float(obb["center"][2] + obb["half"][2])
        bottom = points[:, 2] - BALL_RADIUS_M
        near_top = (bottom <= top_z + EVENT_NEAR_TOL_M) & (bottom >= top_z - 0.08)
        result |= inside_xy & near_top
    return result


def event_for_trajectory(record: dict, trajectory: np.ndarray, obbs: list[dict]) -> dict:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    family = record["family"]
    if family in {"aperture", "deflector"}:
        wanted = target_names(record)
        by_name = penetration_series(trajectory, obbs)
        target_obbs = [obb for obb in obbs if obb["name"] in wanted]
        target_pen = np.stack([by_name[name] for name in sorted(wanted)], axis=0) if wanted else np.zeros((0, len(trajectory)))
        minimum_pen = target_pen.max(axis=0) if len(target_pen) else np.zeros((len(trajectory),), dtype=np.float32)
        # Position-derived event proxy: a sphere is considered in contact when
        # its finite-OBB signed gap is within 5 mm.  This catches solver
        # contacts whose recorded penetration remains below a millimetre.
        gaps = [signed_distance_obb(trajectory, obb) - BALL_RADIUS_M for obb in target_obbs]
        minimum_gap = np.min(np.stack(gaps, axis=0), axis=0) if gaps else np.full((len(trajectory),), np.inf)
        candidates = np.flatnonzero(minimum_gap <= EVENT_TOL_M)
        index = None if len(candidates) == 0 else int(candidates[0])
        return {"kind": "target_collision", "first_future_index": index,
                "first_future_frame": None if index is None else index + 8,
                "first_future_time_after_rgb7_s": None if index is None else float((index + 1) / FPS),
                "target_penetration_max_m": float(minimum_pen.max()) if len(minimum_pen) else 0.0,
                "target_min_gap_m": float(minimum_gap.min()) if len(minimum_gap) else None}
    observed = record["positions"][7][None]
    states = support_state(np.concatenate((observed, trajectory), axis=0), obbs)
    losses = np.flatnonzero(states[:-1] & ~states[1:])
    landings = np.flatnonzero(~states[:-1] & states[1:])
    loss = None if len(losses) == 0 else int(losses[0])
    landing = None if len(landings) == 0 else int(landings[0])
    index = loss if loss is not None else landing
    return {"kind": "support_loss" if loss is not None else "landing",
            "first_future_index": index, "first_future_frame": None if index is None else index + 8,
            "first_future_time_after_rgb7_s": None if index is None else float((index + 1) / FPS),
            "support_loss_index": loss, "landing_index": landing}


def velocity_from_positions(positions: np.ndarray, last_position: np.ndarray) -> np.ndarray:
    full = np.concatenate((last_position[None], positions), axis=0)
    return np.diff(full, axis=0) * FPS


def penetration_summary(record: dict, trajectory: np.ndarray, obbs: list[dict]) -> tuple[dict, list[dict]]:
    values = penetration_series(trajectory, obbs)
    details = []
    all_values = []
    for name, series in values.items():
        all_values.extend(series.tolist())
        details.append({"key": record["key"], "family": record["family"], "split": record["split"],
                        "collider": name, "mean_penetration_m": float(np.mean(series)),
                        "max_penetration_m": float(np.max(series)),
                        "p95_penetration_m": float(np.percentile(series, 95)),
                        "penetration_gt5mm_frame_rate": float(np.mean(series > PENETRATION_REPORT_TOL_M)),
                        "frames": int(len(series))})
    all_values = np.asarray(all_values, dtype=np.float32)
    overall = {"mean_penetration_m": float(np.mean(all_values)) if len(all_values) else 0.0,
               "max_penetration_m": float(np.max(all_values)) if len(all_values) else 0.0,
               "p95_penetration_m": float(np.percentile(all_values, 95)) if len(all_values) else 0.0,
               "penetration_gt5mm_frame_rate": float(np.mean(all_values > PENETRATION_REPORT_TOL_M)) if len(all_values) else 0.0}
    return overall, details


def trajectory_metrics(record: dict, prediction: np.ndarray, target: np.ndarray,
                       mean: torch.Tensor, std: torch.Tensor, arm: str,
                       eval_seed: int, obbs: list[dict]) -> tuple[dict, list[dict], dict]:
    last = record["positions"][7].astype(np.float32)
    diff = np.linalg.norm(prediction - target, axis=-1)
    true_v = velocity_from_positions(target, last)
    pred_v = velocity_from_positions(prediction, last)
    pen, details = penetration_summary(record, prediction, obbs)
    gt_event = event_for_trajectory(record, target, obbs)
    pred_event = event_for_trajectory(record, prediction, obbs)
    gt_index = gt_event.get("first_future_index")
    pred_index = pred_event.get("first_future_index")
    event_error = None if gt_index is None or pred_index is None else float((pred_index - gt_index) / FPS)
    if gt_index is None:
        before = near = after = None
    else:
        windows = {"before": range(max(0, gt_index - 3), gt_index),
                   "near": range(max(0, gt_index - 1), min(STEPS, gt_index + 2)),
                   "after": range(min(STEPS, gt_index + 2), min(STEPS, gt_index + 5))}
        values = {name: [float(diff[i]) for i in indices] for name, indices in windows.items()}
        before = float(np.mean(values["before"])) if values["before"] else None
        near = float(np.mean(values["near"])) if values["near"] else None
        after = float(np.mean(values["after"])) if values["after"] else None
    row = {"key": record["key"], "family": record["family"], "split": record["split"],
           "group_id": record["group_id"], "history_index": record["history_index"],
           "parent_history_id": f"parent_h{int(record['history_index']):02d}",
           "arm": arm, "eval_seed": eval_seed,
           "ADE_m": float(np.mean(diff)), "FDE_m": float(diff[-1]),
           "ADE_over_radius": float(np.mean(diff) / BALL_RADIUS_M),
           "interval_velocity_MAE_mps": float(np.mean(np.abs(pred_v - true_v))),
           "gt_event_kind": gt_event["kind"], "gt_event_index": gt_index,
           "pred_event_index": pred_index, "event_time_error_s": event_error,
           "event_missing": bool(gt_index is not None and pred_index is None),
           "event_extra": bool(gt_index is None and pred_index is not None),
           "error_before_event_m": before, "error_near_event_m": near,
           "error_after_event_m": after, **pen}
    event_row = {"key": record["key"], "family": record["family"], "split": record["split"],
                 "group_id": record["group_id"], "history_index": record["history_index"],
                 "parent_history_id": f"parent_h{int(record['history_index']):02d}",
                 "arm": arm, "eval_seed": eval_seed,
                 "event_kind": gt_event["kind"], "gt_event_index": gt_index,
                 "pred_event_index": pred_index, "event_time_error_s": event_error,
                 "missing": bool(gt_index is not None and pred_index is None),
                 "extra": bool(gt_index is None and pred_index is not None),
                 "error_before_event_m": before, "error_near_event_m": near,
                 "error_after_event_m": after}
    return row, details, event_row


def model_prediction(model, record: dict, arm: str, mean: torch.Tensor,
                     std: torch.Tensor, seed: int) -> np.ndarray:
    batch, _ = make_batch([record], arm, mean, std, seeds=[seed])
    with torch.no_grad():
        return model(batch)[0, 0].detach().cpu().numpy().astype(np.float32)


def analytic_prediction(record: dict) -> np.ndarray:
    return (record["positions"][7][None] +
            record["velocities"][7][None] * (np.arange(1, STEPS + 1, dtype=np.float32)[:, None] / FPS)).astype(np.float32)


def control_rows(control_path: Path) -> dict[str, list[dict]]:
    config = json.loads(control_path.read_text())
    families = config.get("families", {})
    if set(families) != set(CONTROL_FAMILIES):
        raise ValueError("control table must contain exactly the three Phase 2 families")
    result = {}
    for family in CONTROL_FAMILIES:
        if len(families[family]) != 48:
            raise ValueError(f"{family} control table has {len(families[family])}, expected 48")
        result[family] = []
        for index, control in enumerate(families[family]):
            split = "train" if index < 32 else "dev" if index < 40 else "locked_test"
            result[family].append({"parent_history_id": f"parent_h{index:02d}",
                                   "history_index": index, "split": split, **control})
    return result


def response_and_event_admission(manifest: dict, records: list[dict], controls: dict[str, list[dict]], output: Path) -> dict:
    by_family_split_layer = {}
    for pair in manifest["pair_rows"]:
        key = (pair["family"], pair["split"], response_label(pair["response_layer"]))
        by_family_split_layer.setdefault(key, []).append(pair)
    response_rows = []
    for family in CONTROL_FAMILIES:
        for split in ("train", "dev", "locked_test"):
            for layer in ("strong", "weak", "equal_future"):
                values = by_family_split_layer.get((family, split, layer), [])
                response_rows.append({"family": family, "split": split, "response_layer": layer,
                                      "pair_count": len(values),
                                      "group_count": len({row["group_id"] for row in values}),
                                      "positive_response": layer in {"strong", "weak"} and bool(values),
                                      "equal_future_negative_control": layer == "equal_future" and bool(values)})
    write_csv(output / "response_strata.csv", response_rows,
              ["family", "split", "response_layer", "pair_count", "group_count",
               "positive_response", "equal_future_negative_control"])

    timing_rows = []
    for family in CONTROL_FAMILIES:
        for split in ("train", "dev", "locked_test"):
            subset = [row for row in records if row["family"] == family and row["split"] == split]
            categories = {"observation": 0, "recommended_window_0p25_0p90": 0,
                          "prediction_window_outside_recommended": 0, "no_event": 0}
            for row in subset:
                interaction = row["interaction"]
                frame = interaction.get("first_future_frame")
                if frame is None:
                    categories["no_event"] += 1
                elif int(frame) <= 7:
                    categories["observation"] += 1
                elif interaction.get("within_recommended_window_0p25_0p90_s", False):
                    categories["recommended_window_0p25_0p90"] += 1
                else:
                    categories["prediction_window_outside_recommended"] += 1
            for category, count in categories.items():
                timing_rows.append({"family": family, "split": split, "event_category": category,
                                    "episode_count": count, "episode_total": len(subset)})
    write_csv(output / "event_timing.csv", timing_rows,
              ["family", "split", "event_category", "episode_count", "episode_total"])

    group_rows = []
    for family in CONTROL_FAMILIES:
        family_records = [row for row in records if row["family"] == family]
        by_group = {}
        for row in family_records:
            by_group.setdefault(row["group_id"], []).append(row)
        for group_id, variants in sorted(by_group.items()):
            first = variants[0]
            index = int(first["history_index"])
            controls_for_family = controls[family][index]
            group_rows.append({"family": family, "group_id": group_id,
                               "parent_history_id": controls_for_family["parent_history_id"],
                               "history_index": index, "split": first["split"],
                               "variant_count": len(variants),
                               "geometry_values": json.dumps(sorted(float(x["geometry_value"]) for x in variants)),
                               "history_hashes": json.dumps(sorted({x["history_hash"] for x in variants})),
                               "motion_hashes": json.dumps(sorted({x["motion_hash"] for x in variants})),
                               "history_equal_within_group": len({x["history_hash"] for x in variants}) == 1,
                               "motion_equal_within_group": len({x["motion_hash"] for x in variants}) == 1})
    write_json(output / "group_lineage_manifest.json", {"status": "EXECUTED", "rows": group_rows,
                                                         "parent_table_reuse": "same parent_hXX index across families; family-specific controls are versioned rows"})
    return {"response_rows": response_rows, "timing_rows": timing_rows, "group_rows": group_rows}


def admission(manifest: dict, records: list[dict], data_root: Path, controls_path: Path, output: Path) -> None:
    controls = control_rows(controls_path)
    if len(records) != 432 or int(manifest.get("episodes_total", -1)) != 432:
        raise ValueError("Phase 2 admission requires exactly 432 replay records")
    by_family = {family: [row for row in records if row["family"] == family] for family in CONTROL_FAMILIES}
    checks = {"manifest_records_432": len(records) == 432, "groups_144": len({r["group_id"] for r in records}) == 144,
              "three_variants_per_group": all(sum(x["group_id"] == g for x in records) == 3 for g in {r["group_id"] for r in records}),
              "split_counts": {}, "family_split_positive_and_equal": {}, "all_group_history_equal": bool(manifest["history_input_checks"]["all_group_history_equal"]),
              "all_group_motion_equal": bool(manifest["history_input_checks"]["all_group_motion_equal"]),
              "future_not_used": not bool(manifest.get("future_used_for_geometry")) and not bool(manifest.get("future_used_for_input")),
              "zero_rebound_corrections": all(int(r.get("rebound_corrections", 0)) == 0 for r in records),
              "surface_tokens_within_limit": all(int(r["surface_report"]["padded_surface_count"]) == SURFACE_TOKEN_COUNT and int(r["surface_report"]["valid_surface_count"]) <= SURFACE_TOKEN_COUNT for r in records),
              "point_count_512": all(int(r["surface_report"]["point_valid_count"]) == POINT_VALID_COUNT for r in records),
              "cross_split_parent_disjoint": True}
    for family in CONTROL_FAMILIES:
        checks["split_counts"][family] = {}
        checks["family_split_positive_and_equal"][family] = {}
        for split in ("train", "dev", "locked_test"):
            subset = [r for r in by_family[family] if r["split"] == split]
            checks["split_counts"][family][split] = {"episodes": len(subset), "groups": len({r["group_id"] for r in subset})}
            layers = {response_label(p["response_layer"]) for p in manifest["pair_rows"] if p["family"] == family and p["split"] == split}
            checks["family_split_positive_and_equal"][family][split] = {
                "positive_response": bool(layers & {"strong", "weak"}), "equal_future": "equal_future" in layers,
                "layers_present": sorted(layers)}
    parent_split = {}
    for family, rows in controls.items():
        for row in rows:
            previous = parent_split.setdefault(row["parent_history_id"], row["split"])
            checks["cross_split_parent_disjoint"] &= previous == row["split"]
    result = {"status": "EXECUTED", "purpose": "full_data_admission_before_model_training",
              "data_root": str(data_root.resolve()), "manifest_protocol": manifest["protocol_version"],
              "manifest_source_sha256": manifest.get("source_sha256"), "records": len(records),
              "groups": len({r["group_id"] for r in records}), "splits": {s: sum(r["split"] == s for r in records) for s in ("train", "dev", "locked_test")},
              "checks": checks, "locked_test_model_metrics": "NOT_RUN",
              "response_layer_counts": manifest.get("response_layer_counts"),
              "event_timing_source": "manifest interaction metadata; no model predictions"}
    response_and_event_admission(manifest, records, controls, output)
    files = [data_root / "pilot_manifest.json", data_root / "physics_protocol.json", data_root / "full_432_data_config.json", controls_path,
             Path(__file__), Path(__file__).with_name("run_physvideo_phase1_triarm_profile.py"), Path(__file__).with_name("finite_surface_geometry.py")]
    write_json(output / "data_freeze.json", {"status": "EXECUTED", "protocol": manifest["protocol_version"],
                                             "files": {str(path): sha256_file(path) for path in files},
                                             "split_policy": "train/dev/locked_test; model metrics only train+dev"})
    write_json(output / "full_data_admission.json", result)


def finite_geometry_selftest() -> dict:
    obb = {"name": "test_box", "center": np.zeros(3, dtype=np.float32),
           "rotation": np.eye(3, dtype=np.float32), "half": np.ones(3, dtype=np.float32)}
    points = np.asarray([[0.0, 0.0, 1.11], [0.0, 0.0, 1.05], [2.0, 0.0, 0.0]], dtype=np.float32)
    values = np.maximum(0.0, BALL_RADIUS_M - signed_distance_obb(points, obb))
    tests = {"surface_clearance_zero": abs(float(values[0])) < 1e-7,
             "known_inside_penetration": abs(float(values[1]) - 0.06) < 1e-6,
             "finite_edge_distance": abs(float(values[2])) < 1e-7}
    return {"status": "EXECUTED", "tests": tests, "all_pass": all(tests.values()),
            "geometry": "sphere radius 0.11 vs finite OBB signed distance; no infinite plane"}


def gt_selftest(manifest: dict, records: list[dict], output: Path) -> None:
    train_dev = [row for row in records if row["split"] in EVAL_SPLITS]
    rows = []
    gt_predictions = {}
    for record in train_dev:
        obbs = build_obbs(record)
        target = record["target"].astype(np.float32)
        gt_predictions[record["key"]] = target
        pen, details = penetration_summary(record, target, obbs)
        event = event_for_trajectory(record, target, obbs)
        rows.append({"key": record["key"], "family": record["family"], "split": record["split"],
                     "history_index": record["history_index"], "parent_history_id": f"parent_h{int(record['history_index']):02d}",
                     "ADE_m": 0.0, "FDE_m": 0.0, "interval_velocity_MAE_mps": 0.0,
                     "gt_mean_penetration_m": pen["mean_penetration_m"], "gt_max_penetration_m": pen["max_penetration_m"],
                     "gt_p95_penetration_m": pen["p95_penetration_m"], "gt_event_kind": event["kind"],
                     "gt_event_index": event["first_future_index"]})
    write_csv(output / "gt_selftest.csv", rows)
    pair_rows = []
    by_group = {}
    for row in train_dev:
        by_group.setdefault(row["group_id"], []).append(row)
    for pair in manifest["pair_rows"]:
        if pair["split"] not in EVAL_SPLITS:
            continue
        group = by_group[pair["group_id"]]
        left = next(row for row in group if float(row["geometry_value"]) == float(pair["geometry_a"]))
        right = next(row for row in group if float(row["geometry_value"]) == float(pair["geometry_b"]))
        delta = float(np.linalg.norm(gt_predictions[left["key"]] - gt_predictions[right["key"]], axis=-1).mean())
        pair_rows.append({"family": pair["family"], "split": pair["split"], "group_id": pair["group_id"],
                          "response_layer": response_label(pair["response_layer"]), "D_gt_m": pair["D_gt_m"], "D_pred_gt_m": delta,
                          "E_delta_m": abs(delta - float(pair["D_gt_m"])),
                          "R_delta": None if float(pair["D_gt_m"]) <= 1e-6 else delta / float(pair["D_gt_m"]),
                          "R_error": None if float(pair["D_gt_m"]) <= 1e-6 else abs(delta - float(pair["D_gt_m"])) / float(pair["D_gt_m"]),
                          "equal_future_absolute_prediction_delta_m": delta if float(pair["D_gt_m"]) <= 1e-6 else None})
    write_csv(output / "gt_pair_selftest.csv", pair_rows)
    write_json(output / "evaluator_selftest.json", {"status": "EXECUTED", "finite_geometry": finite_geometry_selftest(),
                                                     "prediction_equal_gt": {"ADE_m": 0.0, "FDE_m": 0.0, "interval_velocity_MAE_mps": 0.0},
                                                     "gt_pair_rows": len(pair_rows),
                                                     "gt_pair_max_abs_E_delta_m": max((x["E_delta_m"] for x in pair_rows), default=0.0),
                                                     "gt_equal_future_max_absolute_delta_m": max((x["equal_future_absolute_prediction_delta_m"] for x in pair_rows if x["equal_future_absolute_prediction_delta_m"] is not None), default=0.0),
                                                     "gt_penetration_policy": "report solver geometry proxy; do not force GT to zero"})


def analytic_report(records: list[dict], output: Path, manifest: dict | None = None) -> None:
    rows = []
    predictions = {}
    for record in records:
        if record["split"] not in EVAL_SPLITS:
            continue
        prediction = analytic_prediction(record)
        predictions[record["key"]] = prediction
        target = record["target"].astype(np.float32)
        diff = np.linalg.norm(prediction - target, axis=-1)
        vel_error = np.mean(np.abs(velocity_from_positions(prediction, record["positions"][7]) - velocity_from_positions(target, record["positions"][7])))
        rows.append({"key": record["key"], "family": record["family"], "split": record["split"], "arm": "analytic_cv", "eval_seed": 0,
                     "ADE_m": float(diff.mean()), "FDE_m": float(diff[-1]), "ADE_over_radius": float(diff.mean() / BALL_RADIUS_M),
                     "interval_velocity_MAE_mps": float(vel_error)})
    write_csv(output / "analytic_cv_train_dev.csv", rows)
    if manifest is not None:
        pair_rows = []
        by_group = {}
        for record in records:
            if record["split"] in EVAL_SPLITS:
                by_group.setdefault(record["group_id"], []).append(record)
        for pair in manifest["pair_rows"]:
            if pair["split"] not in EVAL_SPLITS:
                continue
            group = by_group[pair["group_id"]]
            left = next(row for row in group if float(row["geometry_value"]) == float(pair["geometry_a"]))
            right = next(row for row in group if float(row["geometry_value"]) == float(pair["geometry_b"]))
            delta = float(np.linalg.norm(predictions[left["key"]] - predictions[right["key"]], axis=-1).mean())
            dgt = float(pair["D_gt_m"])
            pair_rows.append({"family": pair["family"], "split": pair["split"], "group_id": pair["group_id"],
                              "response_layer": response_label(pair["response_layer"]), "D_gt_m": dgt,
                              "D_pred_m": delta, "E_delta_m": abs(delta - dgt),
                              "R_delta": None if dgt <= 1e-6 else delta / dgt,
                              "R_error": None if dgt <= 1e-6 else abs(delta - dgt) / dgt,
                              "equal_future_absolute_prediction_delta_m": delta if dgt <= 1e-6 else None})
        write_csv(output / "analytic_cv_pairs.csv", pair_rows)


def load_checkpoint_model(training_root: Path, arm: str, epoch: int, mean: torch.Tensor, std: torch.Tensor):
    models, _ = make_models(mean, std, seed=42)
    checkpoint = training_root / "seed_42" / f"{arm}_seed42_epoch{epoch}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    models[arm].load_state_dict(payload["state_dict"])
    models[arm].eval()
    return models[arm], sha256_file(checkpoint)


def aggregate_family(rows: list[dict], pair_rows: list[dict], event_rows: list[dict], output: Path) -> None:
    result = []
    keys = sorted({(r["arm"], r["family"], r["split"], int(r["eval_seed"])) for r in rows})
    for arm, family, split, seed in keys:
        subset = [r for r in rows if r["arm"] == arm and r["family"] == family and r["split"] == split and int(r["eval_seed"]) == seed]
        pairs = [r for r in pair_rows if r["arm"] == arm and r["family"] == family and r["split"] == split and int(r["eval_seed"]) == seed]
        events = [r for r in event_rows if r["arm"] == arm and r["family"] == family and r["split"] == split and int(r["eval_seed"]) == seed]
        strong = [r for r in pairs if r["response_layer"] == "strong"]
        weak = [r for r in pairs if r["response_layer"] == "weak"]
        equal = [r for r in pairs if r["response_layer"] == "equal_future"]
        def avg(values, key):
            values = [float(x[key]) for x in values if x.get(key) is not None]
            return float(np.mean(values)) if values else None
        result.append({"arm": arm, "family": family, "split": split, "eval_seed": seed, "episodes": len(subset), "pairs": len(pairs),
                       "ADE_m": avg(subset, "ADE_m"), "FDE_m": avg(subset, "FDE_m"), "ADE_over_radius": avg(subset, "ADE_over_radius"),
                       "interval_velocity_MAE_mps": avg(subset, "interval_velocity_MAE_mps"),
                       "strong_R_delta": avg(strong, "R_delta"),
                       "strong_R_error": avg(strong, "R_error"),
                       "strong_R_delta_le_0p25_rate": float(np.mean([float(x["R_delta"]) <= 0.25 for x in strong])) if strong else None,
                       "strong_response_error_le_0p25_rate": float(np.mean([float(x["R_error"]) <= 0.25 for x in strong])) if strong else None,
                       "weak_E_delta_m": avg(weak, "E_delta_m"), "equal_D_pred_m": avg(equal, "D_pred_m"),
                       "equal_absolute_delta_m": avg(equal, "D_pred_m"),
                       "mean_penetration_m": avg(subset, "mean_penetration_m"), "max_penetration_m": max((float(x["max_penetration_m"]) for x in subset), default=None),
                       "penetration_gt5mm_frame_rate": avg(subset, "penetration_gt5mm_frame_rate"),
                       "event_missing_rate": avg(events, "missing"), "event_extra_rate": avg(events, "extra"),
                       "event_time_error_s": avg(events, "event_time_error_s"), "error_before_event_m": avg(events, "error_before_event_m"),
                       "error_near_event_m": avg(events, "error_near_event_m"), "error_after_event_m": avg(events, "error_after_event_m")})
    write_csv(output / "per_family_summary.csv", result)


def pair_evaluation(manifest: dict, records: list[dict], predictions: dict[tuple[str, int, str], np.ndarray], output: Path) -> list[dict]:
    by_group = {}
    for row in records:
        if row["split"] in EVAL_SPLITS:
            by_group.setdefault(row["group_id"], []).append(row)
    rows = []
    for pair in manifest["pair_rows"]:
        if pair["split"] not in EVAL_SPLITS:
            continue
        group = by_group[pair["group_id"]]
        left = next(row for row in group if float(row["geometry_value"]) == float(pair["geometry_a"]))
        right = next(row for row in group if float(row["geometry_value"]) == float(pair["geometry_b"]))
        for arm in ARMS:
            seeds = sorted(seed for (a, seed, key) in predictions if a == arm and key == left["key"])
            for seed in seeds:
                delta = float(np.linalg.norm(predictions[(arm, seed, left["key"])] - predictions[(arm, seed, right["key"])], axis=-1).mean())
                dgt = float(pair["D_gt_m"])
                rows.append({"family": pair["family"], "split": pair["split"], "group_id": pair["group_id"], "arm": arm, "eval_seed": seed,
                             "history_index": left["history_index"], "parent_history_id": f"parent_h{int(left['history_index']):02d}",
                             "geometry_a": pair["geometry_a"], "geometry_b": pair["geometry_b"], "response_layer": response_label(pair["response_layer"]),
                             "D_gt_m": dgt, "D_pred_m": delta, "E_delta_m": abs(delta - dgt),
                             "R_delta": None if dgt <= 1e-6 else delta / dgt,
                             "R_error": None if dgt <= 1e-6 else abs(delta - dgt) / dgt,
                             "response_recovery": None if dgt <= 1e-6 else delta / dgt,
                             "equal_future_absolute_prediction_delta_m": delta if dgt <= 1e-6 else None})
    write_csv(output / "per_pair_seed.csv", rows)
    return rows


def sampling_report(records: list[dict], prediction_map: dict, output: Path) -> None:
    rows = []
    for record in records:
        if record["split"] not in EVAL_SPLITS:
            continue
        values = [prediction_map[("point_geometry_resample", seed, record["key"])] for seed in range(16)]
        distances = [float(np.linalg.norm(values[index] - values[0], axis=-1).mean()) for index in range(1, 16)]
        pairwise = [float(np.linalg.norm(values[i] - values[j], axis=-1).mean()) for i in range(16) for j in range(i)]
        rows.append({"key": record["key"], "family": record["family"], "split": record["split"], "seeds": 16,
                     "mean_to_seed0_m": float(np.mean(distances)), "max_to_seed0_m": float(np.max(distances)),
                     "mean_pairwise_m": float(np.mean(pairwise)) if pairwise else 0.0,
                     "max_pairwise_m": float(np.max(pairwise)) if pairwise else 0.0})
    write_csv(output / "sampling_stability.csv", rows)


def permutation_report(model, records: list[dict], mean: torch.Tensor, std: torch.Tensor, output: Path) -> None:
    rows = []
    rng = np.random.default_rng(9021)
    for record in records:
        if record["split"] not in EVAL_SPLITS:
            continue
        batch, _ = make_batch([record], "finite_surface_geometry", mean, std, seeds=[0])
        with torch.no_grad():
            original = model(batch)[0, 0].detach().cpu().numpy()
        permutation = rng.permutation(batch["surface_center"].shape[1])
        permuted = {key: value.clone() for key, value in batch.items()}
        for key in ("surface_center", "surface_normal", "surface_tangent_u", "surface_half_extents", "surface_mask"):
            permuted[key] = permuted[key][:, permutation]
        with torch.no_grad():
            changed = model(permuted)[0, 0].detach().cpu().numpy()
        delta = np.linalg.norm(original - changed, axis=-1)
        rows.append({"key": record["key"], "family": record["family"], "split": record["split"],
                     "mean_permutation_delta_m": float(delta.mean()), "max_permutation_delta_m": float(delta.max())})
    write_csv(output / "finite_surface_permutation.csv", rows)


def checkpoint_curves(manifest: dict, records: list[dict], mean: torch.Tensor, std: torch.Tensor, training_root: Path, output: Path) -> None:
    rows = []
    for epoch in (50, 100, 200):
        for arm in ARMS:
            model, _ = load_checkpoint_model(training_root, arm, epoch, mean, std)
            predictions = {}
            episode_rows = []
            for record in records:
                if record["split"] not in EVAL_SPLITS:
                    continue
                prediction = model_prediction(model, record, arm, mean, std, int(record["point_seed"]))
                predictions[record["key"]] = prediction
                target = record["target"].astype(np.float32)
                diff = np.linalg.norm(prediction - target, axis=-1)
                episode_rows.append((record, diff))
            for split in EVAL_SPLITS:
                for family in CONTROL_FAMILIES:
                    subset = [(r, d) for r, d in episode_rows if r["split"] == split and r["family"] == family]
                    pair_subset = [r for r in records if r["split"] == split and r["family"] == family]
                    pair_deltas, response_errors = [], []
                    for pair in manifest["pair_rows"]:
                        if pair["family"] != family or pair["split"] != split:
                            continue
                        variants = [r for r in pair_subset if r["group_id"] == pair["group_id"]]
                        left = next(r for r in variants if float(r["geometry_value"]) == float(pair["geometry_a"]))
                        right = next(r for r in variants if float(r["geometry_value"]) == float(pair["geometry_b"]))
                        dgt = float(pair["D_gt_m"])
                        dpred = float(np.linalg.norm(predictions[left["key"]] - predictions[right["key"]], axis=-1).mean())
                        if dgt > 1e-6 and response_label(pair["response_layer"]) == "strong":
                            pair_deltas.append(dpred / dgt)
                            response_errors.append(abs(dpred - dgt) / dgt)
                    rows.append({"epoch": epoch, "arm": arm, "split": split, "family": family, "episodes": len(subset),
                                 "ADE_m": float(np.mean([d.mean() for _, d in subset])),
                                 "FDE_m": float(np.mean([d[-1] for _, d in subset])),
                                 "strong_R_delta": float(np.mean(pair_deltas)) if pair_deltas else None,
                                 "strong_R_error": float(np.mean(response_errors)) if response_errors else None})
    write_csv(output / "checkpoint_curves.csv", rows)


def predictions(manifest: dict, records: list[dict], training_root: Path, output: Path) -> None:
    train_records = [row for row in records if row["split"] == "train"]
    mean, std = build_motion_stats(train_records)
    write_json(output / "evaluation_config.json", {"status": "EXECUTED", "seed": 42, "epochs_main": 200,
                                                    "evaluated_splits": ["train", "dev"], "locked_test_model_metrics": "NOT_RUN",
                                                    "point_eval_seeds": list(range(16)), "event_tolerance_m": EVENT_TOL_M,
                                                    "penetration_report_tolerance_m": PENETRATION_REPORT_TOL_M,
                                                    "normalization": "train observation motion only"})
    prediction_map = {}
    all_rows, all_event_rows, all_penetration_rows = [], [], []
    checkpoint_hashes = {}
    models = {}
    for arm in ARMS:
        model, checkpoint_hash = load_checkpoint_model(training_root, arm, 200, mean, std)
        models[arm] = model
        checkpoint_hashes[arm] = checkpoint_hash
        eval_records = [row for row in records if row["split"] in EVAL_SPLITS]
        seeds_by_record = (list(range(16)) if arm == "point_geometry_resample" else [0])
        for record in eval_records:
            for seed in seeds_by_record:
                actual_seed = int(record["point_seed"]) + 70000 + seed if arm == "point_geometry_resample" else int(record["point_seed"])
                prediction = model_prediction(model, record, arm, mean, std, actual_seed)
                prediction_map[(arm, seed, record["key"])] = prediction
                metric, penetration_rows, event_row = trajectory_metrics(record, prediction, record["target"].astype(np.float32), mean, std, arm, seed, build_obbs(record))
                all_rows.append(metric)
                for penetration_row in penetration_rows:
                    penetration_row["arm"] = arm
                    penetration_row["eval_seed"] = seed
                all_penetration_rows.extend(penetration_rows)
                all_event_rows.append(event_row)
    write_csv(output / "per_episode_seed.csv", all_rows)
    write_csv(output / "penetration_by_collider.csv", all_penetration_rows)
    write_csv(output / "event_by_episode.csv", all_event_rows)
    pairs = pair_evaluation(manifest, records, prediction_map, output)
    aggregate_family(all_rows, pairs, all_event_rows, output)
    sampling_report([row for row in records if row["split"] in EVAL_SPLITS], prediction_map, output)
    permutation_report(models["finite_surface_geometry"], [row for row in records if row["split"] in EVAL_SPLITS], mean, std, output)
    checkpoint_curves(manifest, records, mean, std, training_root, output)
    write_json(output / "evaluation_status.json", {"status": "EXECUTED", "checkpoint_epoch": 200, "seed": 42,
                                                   "arms": list(ARMS), "checkpoint_sha256": checkpoint_hashes,
                                                   "train_records": sum(row["split"] == "train" for row in records),
                                                   "dev_records": sum(row["split"] == "dev" for row in records),
                                                   "locked_test_model_metrics": "NOT_RUN",
                                                   "analytic_cv": "EVALUATED_IN_ADMISSION_OUTPUT",
                                                   "sampling_seeds": 16, "finite_surface_permutation": "EXECUTED"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("admission", "predictions"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--controls-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest, records = load_records(args.data_root.resolve())
    enrich_records(records)
    args.output.mkdir(parents=True)
    if args.mode == "admission":
        admission(manifest, records, args.data_root, args.controls_json, args.output)
        gt_selftest(manifest, records, args.output)
        analytic_report(records, args.output, manifest)
        command = ("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
                   f"/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B {Path(__file__).resolve()} "
                   f"--mode admission --data-root {args.data_root.resolve()} --controls-json {args.controls_json.resolve()} --output {args.output.resolve()}\n")
        (args.output / "admission_command.txt").write_text(command)
        print(json.dumps({"status": "EXECUTED", "mode": "admission", "output": str(args.output),
                          "records": len(records), "locked_test_model_metrics": "NOT_RUN"}, indent=2))
    else:
        if args.training_root is None:
            raise ValueError("--training-root is required for predictions mode")
        predictions(manifest, records, args.training_root.resolve(), args.output)
        command = ("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
                   f"/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B {Path(__file__).resolve()} "
                   f"--mode predictions --data-root {args.data_root.resolve()} --controls-json {args.controls_json.resolve()} "
                   f"--training-root {args.training_root.resolve()} --output {args.output.resolve()}\n")
        (args.output / "evaluation_command.txt").write_text(command)
        print(json.dumps({"status": "EXECUTED", "mode": "predictions", "output": str(args.output),
                          "seed": 42, "epoch": 200, "locked_test_model_metrics": "NOT_RUN"}, indent=2))


if __name__ == "__main__":
    main()
