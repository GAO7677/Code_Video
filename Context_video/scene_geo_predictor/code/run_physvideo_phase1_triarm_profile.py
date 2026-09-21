"""Run the phase-one three-arm CPU-only forward/backward profile.

This is intentionally a small runtime validation, not a model-selection run:
each arm receives 20 optimizer updates, with matched groups, observations,
normalization, loss, and optimizer settings.  The script reads only the
versioned physics-only pilot manifest and refuses to overwrite its output.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from finite_surface_geometry import (
    FiniteSurfacePredictor,
    POINT_TOKEN_COUNT,
    POINT_VALID_COUNT,
    canonicalize_surfaces,
    sample_surface_points,
)
from future_query_predictor import Predictor, interval_velocity, motion_features, objective


FPS = 30
STEPS = 41
PROFILE_STEPS = 20
SEED = 42
GROUP_BATCH = 2
CLIP_NORM = 1.0
POINT_FEATURE_DIM = 1386


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha_state(value) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; keep the conversion explicit in the
    # receipt so the value is not mistaken for an allocated CUDA byte count.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def load_records(data_root: Path):
    manifest = json.loads((data_root / "pilot_manifest.json").read_text())
    if manifest.get("status") != "EXECUTED" or int(manifest.get("episodes_total", 0)) < 36:
        raise ValueError("profile/training requires an EXECUTED manifest with at least the 36-episode pilot")
    records = []
    for item in manifest["records"]:
        sample = Path(item["sample_dir"])
        raw = np.load(sample / "raw/states_xyzw.npz", allow_pickle=False)
        names = [str(x) for x in raw["object_names"].tolist()]
        if names.count("pilot_ball") != 1:
            raise ValueError(f"expected exactly one pilot_ball in {sample}")
        dynamic_index = names.index("pilot_ball")
        positions = raw["positions"][:, dynamic_index].astype(np.float32)
        velocities = raw["linear_velocities"][:, dynamic_index].astype(np.float32)
        geometry = np.load(sample / "geometry/finite_surfaces.npz", allow_pickle=False)
        surfaces = canonicalize_surfaces(
            geometry["surface_center"], geometry["surface_normal"],
            geometry["surface_tangent_u"], geometry["surface_half_extents"],
            geometry["surface_mask"],
        )
        records.append({
            **item,
            "sample": sample,
            "positions": positions,
            "velocities": velocities,
            "target": positions[8:49],
            "surfaces": surfaces,
        })
    records.sort(key=lambda row: (row["family"], row["group_id"], row["geometry_value"]))
    return manifest, records


def motion_item(record):
    positions = torch.from_numpy(record["positions"][:8])[None, :, None]
    size = torch.full((1, 1, 3), 0.22, dtype=torch.float32)
    times = torch.arange(8, dtype=torch.float32)[None] / FPS
    return motion_features(positions, size, times)


def build_motion_stats(records):
    features = torch.cat([motion_item(row)[0][:, 0] for row in records], dim=0)
    mean = features.mean(0)
    std = features.std(0, unbiased=False).clamp_min(1e-3)
    return mean, std


def point_tokens(record, seed: int):
    sample = sample_surface_points(record["surfaces"], int(seed), count=POINT_VALID_COUNT, resolution=10)
    xyz = np.zeros((POINT_TOKEN_COUNT, 3), dtype=np.float32)
    xyz[:POINT_VALID_COUNT] = sample["point_xyz"]
    mask = np.zeros((POINT_TOKEN_COUNT,), dtype=bool)
    mask[:POINT_VALID_COUNT] = True
    return xyz, mask


def make_batch(records, arm: str, mean: torch.Tensor, std: torch.Tensor, seeds=None):
    motion, last, velocity = [], [], []
    targets = []
    for record in records:
        item_motion, item_last, item_velocity = motion_item(record)
        motion.append(item_motion[0, 0])
        last.append(item_last[0, 0])
        velocity.append(item_velocity[0, 0])
        targets.append(torch.from_numpy(record["target"]))
    b = len(records)
    motion = torch.stack(motion)[:, None]
    last = torch.stack(last)[:, None]
    velocity = torch.stack(velocity)[:, None]
    target = torch.stack(targets)[:, None]
    future_dt = torch.arange(1, STEPS + 1, dtype=torch.float32)[None].expand(b, -1) / FPS
    batch = {
        "motion": motion,
        "object_mask": torch.ones((b, 1), dtype=torch.bool),
        "last_position": last,
        "last_velocity": velocity,
        "future_dt": future_dt,
    }
    if arm in {"motion_only", "point_geometry_resample"}:
        scene_xyz = torch.zeros((b, POINT_TOKEN_COUNT, 3), dtype=torch.float32)
        scene_mask = torch.zeros((b, POINT_TOKEN_COUNT), dtype=torch.bool)
        if arm == "point_geometry_resample":
            if seeds is None:
                raise ValueError("point arm requires explicit per-record seeds")
            for i, (record, seed) in enumerate(zip(records, seeds)):
                xyz, mask = point_tokens(record, seed)
                scene_xyz[i] = torch.from_numpy(xyz)
                scene_mask[i] = torch.from_numpy(mask)
        batch.update({
            "scene_xyz": scene_xyz,
            "scene_features": torch.zeros((b, POINT_TOKEN_COUNT, POINT_FEATURE_DIM), dtype=torch.float32),
            "scene_mask": scene_mask,
        })
    else:
        fields = ("surface_center", "surface_normal", "surface_tangent_u", "surface_half_extents")
        for field in fields:
            batch[field] = torch.from_numpy(np.stack([row["surfaces"][field] for row in records])).float()
        batch["surface_mask"] = torch.from_numpy(np.stack([row["surfaces"]["surface_mask"] for row in records])).bool()
    return batch, target


def analytic_cv(batch):
    return batch["last_position"][:, :, None] + batch["last_velocity"][:, :, None] * batch["future_dt"][:, None, :, None]


def make_models(mean: torch.Tensor, std: torch.Tensor, seed: int = SEED):
    base = Predictor(mean, std, scene_mode="geometry_only", seed=seed)
    base_state = base.state_dict()
    motion_only = Predictor(mean, std, scene_mode="motion_only", seed=seed)
    point = Predictor(mean, std, scene_mode="geometry_only", seed=seed)
    motion_only.load_state_dict(base_state)
    point.load_state_dict(base_state)
    finite = FiniteSurfacePredictor(mean, std, seed=seed)
    for name in ("motion_encoder", "time_encoder", "position"):
        getattr(finite, name).load_state_dict(getattr(base, name).state_dict())
    shared = {name: tensor.detach().cpu() for name, tensor in base_state.items()
              if name.startswith(("motion_encoder.", "time_encoder.", "position."))}
    shared_hash = sha_state(shared)
    return {
        "motion_only": motion_only,
        "point_geometry_resample": point,
        "finite_surface_geometry": finite,
    }, shared_hash


def parameter_report(models):
    result = {}
    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        geometry = 0
        for key, value in model.named_parameters():
            if any(token in key for token in ("reader", "scene_projection", "surface_encoder")):
                geometry += value.numel() if value.requires_grad else 0
        result[name] = {"total_parameters": total, "trainable_parameters": trainable,
                        "active_geometry_parameters": geometry}
    return result


def profile_arm(name, model, groups, mean, std, output: Path):
    model.train()
    active = list(model.active_parameters())
    if not active:
        raise ValueError(f"{name} has no active parameters")
    optimizer = torch.optim.AdamW(active, lr=3e-4, weight_decay=0.01)
    steps = []
    group_trace = []
    initial_rss = rss_bytes()
    for step in range(PROFILE_STEPS):
        selected = [groups[(step * GROUP_BATCH + offset) % len(groups)] for offset in range(GROUP_BATCH)]
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        losses = []
        exposure = 0
        for group in selected:
            seeds = [int(row["point_seed"]) + 1000003 * step for row in group]
            batch, target = make_batch(group, name, mean, std, seeds=seeds)
            prediction = model(batch)
            loss = objective(prediction, target, batch) / GROUP_BATCH
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {name} step {step}")
            loss.backward()
            losses.append(float(loss.detach()))
            exposure += len(group)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(active, CLIP_NORM))
        optimizer.step()
        elapsed = time.perf_counter() - step_start
        steps.append({"step": step + 1, "loss_accumulated": float(sum(losses)),
                      "loss_mean_group": float(np.mean(losses)), "grad_norm_preclip": grad_norm,
                      "step_seconds": elapsed, "rss_bytes": rss_bytes(), "exposure": exposure})
        group_trace.append([group[0]["group_id"] for group in selected])
    checkpoint = output / f"{name}_after_20_steps.pt"
    torch.save({"arm": name, "status": "EXECUTED", "steps": PROFILE_STEPS,
                "state_dict": model.state_dict()}, checkpoint)
    return {
        "status": "EXECUTED", "optimizer_steps": PROFILE_STEPS,
        "effective_batch": GROUP_BATCH * 3, "microbatch": 3,
        "gradient_accumulation_groups": GROUP_BATCH,
        "gradient_accumulation_scale": 1.0 / GROUP_BATCH,
        "exposure_per_step": GROUP_BATCH * 3, "total_episode_exposures": PROFILE_STEPS * GROUP_BATCH * 3,
        "initial_rss_bytes": initial_rss, "peak_rss_bytes": max(row["rss_bytes"] for row in steps),
        "mean_step_seconds": float(np.mean([row["step_seconds"] for row in steps])),
        "p50_step_seconds": float(np.percentile([row["step_seconds"] for row in steps], 50)),
        "p95_step_seconds": float(np.percentile([row["step_seconds"] for row in steps], 95)),
        "steps": steps, "group_trace": group_trace,
        "checkpoint": str(checkpoint),
    }


@torch.no_grad()
def predict_one(model, record, arm, mean, std, seed):
    batch, target = make_batch([record], arm, mean, std, seeds=[seed])
    return model(batch)[0, 0].detach().cpu(), target[0, 0]


@torch.no_grad()
def evaluate_arm(model, arm, records, mean, std):
    per_episode = []
    predictions = {}
    for record in records:
        pred, target = predict_one(model, record, arm, mean, std, int(record["point_seed"]))
        batch, _ = make_batch([record], arm, mean, std, seeds=[int(record["point_seed"])])
        pred_v = interval_velocity(pred[None, None], batch["last_position"], batch["future_dt"])[0, 0]
        true_v = interval_velocity(target[None, None], batch["last_position"], batch["future_dt"])[0, 0]
        per_episode.append({"key": record["key"], "family": record["family"],
                            "ADE_m": float(torch.linalg.vector_norm(pred - target, dim=-1).mean()),
                            "FDE_m": float(torch.linalg.vector_norm(pred[-1] - target[-1])),
                            "interval_velocity_MAE_mps": float((pred_v - true_v).abs().mean())})
        predictions[record["key"]] = pred.numpy()
    return {
        "status": "EXECUTED", "per_episode": per_episode,
        "aggregate": {key: float(np.mean([row[key] for row in per_episode]))
                       for key in ("ADE_m", "FDE_m", "interval_velocity_MAE_mps")},
        "predictions": predictions,
    }


@torch.no_grad()
def evaluate_analytic(records, mean, std):
    values = []
    for record in records:
        batch, target = make_batch([record], "motion_only", mean, std, seeds=[0])
        pred = analytic_cv(batch)
        values.append({"key": record["key"],
                       "ADE_m": float(torch.linalg.vector_norm(pred - target[:, :, :], dim=-1).mean()),
                       "FDE_m": float(torch.linalg.vector_norm(pred[0, 0, -1] - target[0, 0, -1])),
                       "interval_velocity_MAE_mps": float((interval_velocity(pred, batch["last_position"], batch["future_dt"]) - interval_velocity(target, batch["last_position"], batch["future_dt"])).abs().mean())})
    return {"status": "EXECUTED", "per_episode": values,
            "aggregate": {key: float(np.mean([row[key] for row in values]))
                           for key in ("ADE_m", "FDE_m", "interval_velocity_MAE_mps")}}


def pair_metrics(records, pair_rows, predictions):
    by_key = {row["key"]: row for row in records}
    by_group = {}
    for row in records:
        by_group.setdefault(row["group_id"], []).append(row)
    output = []
    for pair in pair_rows:
        group = sorted(by_group[pair["group_id"]], key=lambda row: row["geometry_value"])
        left = next(row for row in group if row["geometry_value"] == pair["geometry_a"])
        right = next(row for row in group if row["geometry_value"] == pair["geometry_b"])
        result = {"family": pair["family"], "group_id": pair["group_id"],
                  "geometry_a": pair["geometry_a"], "geometry_b": pair["geometry_b"],
                  "D_gt_m": pair["D_gt_m"], "response_layer": pair["response_layer"]}
        for arm, values in predictions.items():
            delta = float(np.linalg.norm(values[left["key"]] - values[right["key"]], axis=-1).mean())
            result[arm] = {"D_pred_m": delta, "E_delta_m": abs(delta - pair["D_gt_m"]),
                           "R_delta": None if pair["D_gt_m"] <= 1e-6 else delta / pair["D_gt_m"],
                           "equal_future_absolute_prediction_delta": delta if pair["D_gt_m"] <= 1e-6 else None}
        output.append(result)
    return output


@torch.no_grad()
def sampling_stability(model, records, mean, std):
    values = []
    for record in records:
        seeds = [int(record["point_seed"]) + 70000 + i for i in range(8)]
        preds = [predict_one(model, record, "point_geometry_resample", mean, std, seed)[0].numpy() for seed in seeds]
        base = preds[0]
        values.append({"key": record["key"], "mean_pairwise_to_seed0_m": float(np.mean([np.linalg.norm(p - base, axis=-1).mean() for p in preds[1:]])),
                       "max_pairwise_to_seed0_m": float(np.max([np.linalg.norm(p - base, axis=-1).mean() for p in preds[1:]]))})
    return {"status": "EXECUTED", "seeds_per_episode": 8, "episodes": values,
            "mean_m": float(np.mean([x["mean_pairwise_to_seed0_m"] for x in values])),
            "max_m": float(np.max([x["max_pairwise_to_seed0_m"] for x in values]))}


@torch.no_grad()
def surface_permutation_check(model, records, mean, std):
    values = []
    rng = np.random.default_rng(9021)
    for record in records:
        batch, _ = make_batch([record], "finite_surface_geometry", mean, std, seeds=[0])
        original = model(batch)[0, 0]
        perm = rng.permutation(batch["surface_center"].shape[1])
        permuted = {key: value.clone() for key, value in batch.items()}
        for key in ("surface_center", "surface_normal", "surface_tangent_u", "surface_half_extents", "surface_mask"):
            permuted[key] = permuted[key][:, perm]
        changed = model(permuted)[0, 0]
        values.append({"key": record["key"], "mean_m": float(torch.linalg.vector_norm(original - changed, dim=-1).mean()),
                       "max_m": float(torch.linalg.vector_norm(original - changed, dim=-1).max())})
    return {"status": "EXECUTED", "episodes": values,
            "mean_m": float(np.mean([x["mean_m"] for x in values])),
            "max_m": float(np.max([x["max_m"] for x in values]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output or data_root / "triarm_profile").resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    if device.type != "cpu":
        raise AssertionError("phase-one profile must use CPU")
    output.mkdir(parents=True)
    manifest, records = load_records(data_root)
    mean, std = build_motion_stats(records)
    dump(output / "motion_normalization.json", {"status": "EXECUTED", "source": "pilot observation only",
        "mean": mean.tolist(), "std": std.tolist()})
    models, shared_hash = make_models(mean, std)
    dump(output / "input_contract.json", {"status": "EXECUTED", "device": "cpu", "threads": 2,
        "observation_fields": {"motion": ["B", "K", 65], "object_mask": ["B", "K"],
        "last_position": ["B", "K", 3], "last_velocity": ["B", "K", 3], "future_dt": ["B", 41]},
        "point_geometry_resample": {"scene_xyz": ["B", 1792, 3], "scene_features": ["B", 1792, 1386], "scene_mask": ["B", 1792], "valid_points": 512},
        "finite_surface_geometry": {"surface_center": ["B", 48, 3], "surface_normal": ["B", 48, 3], "surface_tangent_u": ["B", 48, 3], "surface_half_extents": ["B", 48, 2], "surface_mask": ["B", 48], "center_reference": "world coordinates; predictor forms last-position-relative relation"},
        "future_oracle_fields": [], "family_case_collider_fields": []})
    profile = {"status": "EXECUTED", "purpose": "runtime_and_gradient_validation_only",
               "data_root": str(data_root), "manifest_protocol": manifest["protocol_version"],
               "device": "cpu", "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
               "shared_backbone_initialization_sha256": shared_hash,
               "parameter_counts": parameter_report(models), "arms": {},
               "analytic_cv": {"status": "IMPLEMENTED_AND_EVALUATED_NOT_TRAINED"},
               "selection_warning": "20-step metrics are not evidence that one representation is better"}
    groups = []
    for group_id in sorted({row["group_id"] for row in records}):
        groups.append([row for row in records if row["group_id"] == group_id])
    for name, model in models.items():
        profile["arms"][name] = profile_arm(name, model, groups, mean, std, output)
    evals = {}
    prediction_map = {}
    for name, model in models.items():
        evaluation = evaluate_arm(model, name, records, mean, std)
        prediction_map[name] = evaluation.pop("predictions")
        evals[name] = evaluation
    evals["analytic_cv"] = evaluate_analytic(records, mean, std)
    dump(output / "metrics_not_for_selection.json", evals)
    dump(output / "pair_metrics.json", pair_metrics(records, manifest["pair_rows"], prediction_map))
    dump(output / "sampling_stability.json", sampling_stability(models["point_geometry_resample"], records, mean, std))
    dump(output / "finite_surface_permutation_check.json", surface_permutation_check(models["finite_surface_geometry"], records, mean, std))
    dump(output / "evaluation_schema.json", {"status": "EXECUTED_PARTIAL_PROFILE", "executed": ["ADE_m", "FDE_m", "interval_velocity_MAE_mps", "D_gt_m", "D_pred_m", "E_delta_m", "R_delta", "equal_future_absolute_prediction_delta", "point_resampling_stability", "finite_surface_permutation_stability"], "not_run": {"penetration_depth": "NOT_RUN: full collision evaluator is reserved for the complete evaluation command", "penetration_frame_rate": "NOT_RUN: full collision evaluator is reserved for the complete evaluation command", "event_before_after_error": "NOT_RUN: requires event-aligned evaluation split"}, "metric_policy": "N/A_or_NOT_RUN_not_zero"})
    profile["evaluation_files"] = ["metrics_not_for_selection.json", "pair_metrics.json", "sampling_stability.json", "finite_surface_permutation_check.json", "evaluation_schema.json"]
    dump(output / "profile.json", profile)
    (output / "profile_command.txt").write_text("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B " + str(Path(__file__).resolve()) + " --data-root " + str(data_root) + " --output " + str(output) + "\n")
    print(json.dumps({"status": "EXECUTED", "output": str(output), "arms": list(models), "optimizer_steps_each": PROFILE_STEPS, "device": "cpu"}, indent=2))


if __name__ == "__main__":
    main()
