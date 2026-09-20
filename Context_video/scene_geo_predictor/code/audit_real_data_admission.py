"""Read-only real-data admission audit for the Future Query Predictor.

This script deliberately consumes the existing small_trial_120 artifacts.  It
does not run the visual front end, rewrite caches, train a model, or require a
GPU.  The predictor smoke is a data-chain check, not a quality experiment.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


SMOKE_CASES = (
    "train_barrier_0057", "train_door_0024", "train_gap_0008",
    "val_barrier_0009", "val_door_0005", "val_gap_0025",
)
REQUIRED = (
    ("samples", "metadata.json"),
    ("samples", "replay.json"),
    ("samples", "raw/states_xyzw.npz"),
    ("observed_context", "context_geometry.npz"),
    ("observed_context", "report.json"),
    ("aligned_scene", "coarse_static_points.npz"),
    ("aligned_scene", "report.json"),
    ("scene_features", "scene_tokens.npz"),
    ("scene_features", "report.json"),
)


def _json(path: Path):
    return json.loads(path.read_text())


def inventory(root: Path, rows: list[dict]) -> dict:
    keys = [r["key"] for r in rows]
    result = {"expected_records": len(rows), "by_split": dict(collections.Counter(r["split"] for r in rows))}
    files = {}
    for directory, suffix in REQUIRED:
        ok = missing = corrupt = 0
        for key in keys:
            path = root / directory / key / suffix
            if not path.exists():
                missing += 1
                continue
            try:
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as archive:
                        _ = archive.files
                else:
                    _json(path)
                ok += 1
            except Exception:
                corrupt += 1
        files[f"{directory}/{suffix}"] = {"readable": ok, "missing": missing, "corrupt": corrupt}
    result["files"] = files

    schemas = collections.Counter()
    source_flags = collections.Counter()
    shapes = collections.Counter()
    traceable = 0
    cache_status = collections.Counter()
    for key in keys:
        report = _json(root / "scene_features" / key / "report.json")
        schema = report.get("cache_schema")
        schemas[schema] += 1
        cache_status["v2" if schema == "observed_utonia_tokens_v2" else "legacy"] += 1
        source_flags[(report.get("static_scene_gt_used"), report.get("future_rgb_used"),
                     report.get("gt_mask_used"))] += 1
        if all(report.get(name) for name in ("aligned_points_sha256", "alignment_report_sha256",
                                              "raw_sha256", "input_pixels_sha256", "output_sha256")):
            traceable += 1
        with np.load(root / "scene_features" / key / "scene_tokens.npz", allow_pickle=False) as archive:
            shapes[(tuple(archive["scene_xyz"].shape), tuple(archive["scene_features"].shape),
                    tuple(archive["scene_mask"].shape))] += 1
    result.update(cache_schema=dict(schemas), cache_status=dict(cache_status),
                  source_flags={str(k): v for k, v in source_flags.items()},
                  file_level_traceable=traceable, shapes={str(k): v for k, v in shapes.items()},
                  formal_cache_gate="require_versioned_scene_cache=true; legacy raises ValueError")
    return result


def physics_inventory(root: Path, rows: list[dict]) -> dict:
    dynamic_counts = collections.Counter()
    dynamic_shapes = collections.Counter()
    static_shapes = collections.Counter()
    dynamic_sizes = collections.defaultdict(list)
    mass = collections.defaultdict(list)
    friction = collections.defaultdict(list)
    restitution = collections.defaultdict(list)
    controlled = collections.defaultdict(list)
    source_corrections = collections.Counter()
    cameras = collections.Counter()
    object_count_by_family = collections.defaultdict(list)
    external_fields = collections.Counter()
    for row in rows:
        key = row["key"]
        metadata = _json(root / "samples" / key / "metadata.json")
        actors = metadata.get("actors", {})
        dynamics = [a for a in actors.values() if a.get("dynamic") is True]
        dynamic_counts[len(dynamics)] += 1
        object_count_by_family[row["family"]].append(len(dynamics))
        for actor in actors.values():
            shape = actor.get("shape", "UNKNOWN")
            (dynamic_shapes if actor.get("dynamic") else static_shapes)[shape] += 1
            if actor.get("dynamic"):
                mass[row["family"]].append(actor.get("mass_kg"))
                friction[row["family"]].append(actor.get("friction"))
                restitution[row["family"]].append(actor.get("restitution"))
                dynamic_sizes[row["family"]].append(actor.get("size_m", {}))
        spec = metadata.get("scenario_spec", {})
        name = spec.get("controlled_variable", "UNKNOWN")
        value = None
        for candidate in (name, "barrier_normal_angle_deg", "door_opening_width_m", "gap_width_m"):
            if candidate in spec:
                value = spec[candidate]
                break
        controlled[row["family"]].append({"variable": name, "value": value, "split": row["split"]})
        source_corrections[metadata.get("rebound_corrections")] += 1
        camera = metadata.get("camera", {}).get("extrinsics", {})
        cameras[str(camera)] += 1
        for key_name in ("external_force", "external_forces", "actuation", "control", "impulses"):
            if key_name in spec or key_name in metadata:
                external_fields[key_name] += 1

    def ranges(values):
        numeric = [float(x) for x in values if isinstance(x, (int, float))]
        return {"min": min(numeric), "max": max(numeric), "unique": sorted(set(numeric))} if numeric else {"status": "UNKNOWN"}

    return {
        "records": len(rows),
        "dynamic_object_count": dict(dynamic_counts),
        "dynamic_object_count_by_family": {k: dict(collections.Counter(v)) for k, v in object_count_by_family.items()},
        "dynamic_shapes": dict(dynamic_shapes),
        "static_shapes": dict(static_shapes),
        "dynamic_mass_kg_by_family": {k: ranges(v) for k, v in mass.items()},
        "dynamic_friction_by_family": {k: ranges(v) for k, v in friction.items()},
        "dynamic_restitution_by_family": {k: ranges(v) for k, v in restitution.items()},
        "dynamic_size_by_family": {k: {name: ranges([item.get(name) for item in values])
                                       for name in sorted({name for item in values for name in item})}
                                   for k, values in dynamic_sizes.items()},
        "controlled_variable_by_family": {k: {"variables": sorted(set(x["variable"] for x in v)),
                                               "values": ranges([x["value"] for x in v]),
                                               "by_split": dict(collections.Counter(x["split"] for x in v))}
                                             for k, v in controlled.items()},
        "rebound_corrections": dict(source_corrections),
        "camera_variants": len(cameras),
        "external_action_metadata": ("PRESENT" if external_fields else "UNKNOWN_NOT_RECORDED"),
        "external_action_fields": dict(external_fields),
        "collision_labels": "UNKNOWN_NOT_IN_states_xyzw_or_metadata",
    }


def appearance_inventory(root: Path, rows: list[dict]) -> dict:
    # These IDs are derived from recorded render configuration, not assigned
    # from the sample key.  They are therefore useful for detecting leakage,
    # but do not create unseen appearances that the renderer never produced.
    fields = ("texture_sources", "visible_refine_texture", "natural_texture_assets",
              "hdri", "room_scene", "lighting_preset", "material_assignments",
              "material_overrides", "basketball_texture")
    rows_out = []
    missing = []
    for row in rows:
        path = root / "render_context8" / row["key"] / "cycles_frames" / "render_metadata.json"
        if not path.exists():
            missing.append(row["key"])
            continue
        metadata = _json(path)
        payload = {name: metadata.get(name) for name in fields}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        rows_out.append((row["key"], row["split"], row["family"], hashlib.sha256(raw.encode()).hexdigest()))
    all_ids = {x[3] for x in rows_out}
    train = {x[3] for x in rows_out if x[1] == "train"}
    val = {x[3] for x in rows_out if x[1] == "val"}
    by_family = {}
    for family in sorted({x[2] for x in rows_out}):
        ftrain = {x[3] for x in rows_out if x[2] == family and x[1] == "train"}
        fval = {x[3] for x in rows_out if x[2] == family and x[1] == "val"}
        by_family[family] = {"all_unique": len(ftrain | fval), "train_unique": len(ftrain),
                             "val_unique": len(fval), "shared_train_val": len(ftrain & fval)}
    return {"derived_from": list(fields), "records_with_render_metadata": len(rows_out),
            "missing_render_metadata": missing, "unique_derived_ids": len(all_ids),
            "train_unique": len(train), "val_unique": len(val),
            "shared_train_val": len(train & val), "by_family": by_family,
            "interpretation": "three family-level appearances are repeated across train/val; no OOD appearance split"}


def predictor_smoke(root: Path, cases: list[str]) -> dict:
    import torch
    from future_query_predictor import Predictor, interval_velocity, objective
    from predictor_pilot import build_batch

    batch, target, records = build_batch(root, root / "scene_features", cases)
    motion_mean = batch["motion"].mean((0, 1))
    motion_std = batch["motion"].std((0, 1), unbiased=False).clamp_min(1e-3)
    result = {"cases": cases, "batch_shapes": {k: list(v.shape) for k, v in batch.items()},
              "target_shape": list(target.shape), "modes": {}}
    dt = batch["future_dt"][:, None, :, None]
    cv = batch["last_position"][:, :, None] + batch["last_velocity"][:, :, None] * dt
    cv_error = (cv - target).norm(dim=-1)
    result["analytic_cv"] = {"shape": list(cv.shape), "finite": bool(torch.isfinite(cv).all()),
                              "ADE_m": float(cv_error.mean()), "FDE_m": float(cv_error[:, :, -1].mean()),
                              "interval_velocity_MAE_mps": float(interval_velocity(cv, batch["last_position"], batch["future_dt"]).sub(
                                  interval_velocity(target, batch["last_position"], batch["future_dt"])).abs().mean())}
    for mode in ("motion_only", "constant", "geometry_only", "visual"):
        model = Predictor(motion_mean, motion_std, scene_mode=mode)
        with torch.no_grad():
            prediction = model(batch)
            loss = objective(prediction, target, batch)
        error = (prediction - target).norm(dim=-1)
        pv = interval_velocity(prediction, batch["last_position"], batch["future_dt"])
        tv = interval_velocity(target, batch["last_position"], batch["future_dt"])
        position_loss = torch.nn.functional.smooth_l1_loss(prediction[batch["object_mask"]], target[batch["object_mask"]])
        velocity_loss = torch.nn.functional.smooth_l1_loss(pv[batch["object_mask"]], tv[batch["object_mask"]])
        result["modes"][mode] = {"shape": list(prediction.shape), "finite": bool(torch.isfinite(prediction).all()),
                                  "loss_finite": bool(torch.isfinite(loss)), "loss": float(loss),
                                  "position_smooth_l1": float(position_loss),
                                  "interval_velocity_smooth_l1": float(velocity_loss),
                                  "valid_object_count": int(batch["object_mask"].sum()),
                                  "ADE_m": float(error.mean()), "FDE_m": float(error[:, :, -1].mean()),
                                  "interval_velocity_MAE_mps": float((pv - tv).abs().mean()),
                                  "active_parameter_count": sum(p.numel() for p in model.active_parameters())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "small_trial_120")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _json(args.root / "manifest.json")
    rows = manifest["records"]
    result = {
        "schema": "future_query_real_data_admission_v1",
        "status": "read_only_cpu_audit",
        "root": str(args.root.resolve()),
        "inventory": inventory(args.root, rows),
        "physics": physics_inventory(args.root, rows),
        "appearance": appearance_inventory(args.root, rows),
        "predictor_cpu_smoke": predictor_smoke(args.root, list(SMOKE_CASES)),
        "collision_split_counts": "N/A: no contact/event labels in current sample artifacts",
        "appearance_ids": "BLOCKED: no appearance/style IDs in manifest or metadata",
        "oracle_visible_geometry": "DATA_BLOCKED: no visible-GT-depth cache; only estimated depth and legacy Utonia caches exist",
        "future_replay": "PARTIAL_CPU_ONLY: target swap test is not implemented in this audit; full rebuild would re-run front end",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
