"""Bounded CPU-only door fit for geometry separation diagnostics.

This is deliberately a separate entrypoint from the GPU/formal trainer.  It
uses six fixed train samples, at most 200 steps and batch size two.  The
oracle arm is privileged and is never presented as an RGB deployment result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from future_query_predictor import Predictor, motion_features, objective, interval_velocity
from predictor_config_guard import validate
from prepare_small_trial import sha
from trial_metrics import behaviour

CASES = ["train_door_0712", "train_door_0660", "train_door_0366",
         "train_door_0437", "train_door_0955", "train_door_0891"]
SEED = 20260919


def load_case_rows(root: Path, scene_root: Path, cases):
    rows = []
    for case in cases:
        context = root / "observed_context" / case
        context_report = json.loads((context / "report.json").read_text())
        with np.load(context / "context_geometry.npz", allow_pickle=False) as archive:
            geometry = {k: archive[k] for k in archive.files}
        scene_path = scene_root / case / "scene_tokens.npz"
        scene_report = json.loads((scene_path.parent / "report.json").read_text())
        if sha(scene_path) != scene_report["output_sha256"]:
            raise ValueError(f"scene hash mismatch: {case}")
        with np.load(scene_path, allow_pickle=False) as archive:
            # Keep provenance ids in the row for matched-support diagnostics.
            # They are not part of the Predictor batch and therefore cannot
            # leak into the model input.
            scene = {k: archive[k] for k in (
                "scene_xyz", "scene_features", "scene_mask",
                "source_flat_indices", "source_frame_y_x")
                     if k in archive.files}
        label_path = Path(context_report["sample"]) / "raw/states_xyzw.npz"
        with np.load(label_path, allow_pickle=False) as labels:
            names = labels["object_names"].astype(str).tolist()
            ids = [names.index(n) for n in context_report["dynamic_names"]]
            if not np.allclose(labels["positions"][:8, ids], geometry["positions_world"], atol=1e-6, rtol=0):
                raise ValueError(f"context/supervision mismatch: {case}")
            times = labels["frame_times"][:49]
            if times.shape != (49,) or not np.allclose(times, np.arange(49) / 30, atol=1e-6, rtol=0):
                raise ValueError(f"future timeline mismatch: {case}")
            positions = labels["positions"][:, ids].astype(np.float32)
            target = positions[8:49].transpose(1, 0, 2)
        rows.append(dict(case=case, positions=positions[:8], size=geometry["size_m"].astype(np.float32),
                         times=geometry["frame_times"].astype(np.float32), target=target,
                         scene_xyz=scene["scene_xyz"].astype(np.float32),
                         scene_features=scene["scene_features"].astype(np.float32),
                         scene_mask=scene["scene_mask"].astype(bool),
                         source_flat_indices=scene.get("source_flat_indices"),
                         source_frame_y_x=scene.get("source_frame_y_x"),
                         scene_schema=scene_report.get("cache_schema", scene_report.get("schema")),
                         scene_report=scene_report, supervision_sha256=sha(label_path)))
    return rows


def make_batch(rows):
    positions = torch.from_numpy(np.stack([r["positions"] for r in rows]))
    sizes = torch.from_numpy(np.stack([r["size"] for r in rows]))
    times = torch.from_numpy(np.stack([r["times"] for r in rows]))
    motion, last, velocity = motion_features(positions, sizes, times)
    b = len(rows); n = max(len(r["scene_xyz"]) for r in rows)
    batch = {
        "motion": motion, "object_mask": torch.ones(b, 1, dtype=torch.bool),
        "last_position": last, "last_velocity": velocity,
        "future_dt": torch.arange(1, 42, dtype=torch.float32)[None].expand(b, -1) / 30,
        "scene_xyz": torch.zeros(b, n, 3), "scene_features": torch.zeros(b, n, 1386),
        "scene_mask": torch.zeros(b, n, dtype=torch.bool),
    }
    for i, row in enumerate(rows):
        count = len(row["scene_xyz"])
        batch["scene_xyz"][i, :count] = torch.from_numpy(row["scene_xyz"])
        batch["scene_features"][i, :count] = torch.from_numpy(row["scene_features"])
        batch["scene_mask"][i, :count] = torch.from_numpy(row["scene_mask"])
    target = torch.from_numpy(np.stack([r["target"] for r in rows]))
    if not torch.isfinite(target).all() or not all(torch.isfinite(v).all() for v in batch.values()):
        raise ValueError("nonfinite diagnostic batch")
    return batch, target


def subset(batch, target, indices):
    ids = torch.as_tensor(indices, dtype=torch.long)
    return {k: v[ids] for k, v in batch.items()}, target[ids]


@torch.no_grad()
def metrics(prediction, target, batch, rows):
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    pv = interval_velocity(prediction, batch["last_position"], batch["future_dt"])
    tv = interval_velocity(target, batch["last_position"], batch["future_dt"])
    position_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    velocity_loss = torch.nn.functional.smooth_l1_loss(pv, tv)
    event_rows = []
    for i, row in enumerate(rows):
        event_rows.append(row["event"](prediction[i, 0].cpu().numpy(), target[i, 0].cpu().numpy()))
    numeric = {}
    keys = sorted({k for row in event_rows for k, v in row.items() if isinstance(v, (int, float)) and v is not None})
    for key in keys:
        values = [r[key] for r in event_rows if isinstance(r.get(key), (int, float))]
        numeric[key] = float(np.mean(values)) if values else None
    return {"ADE_m": float(error.mean()), "FDE_m": float(error[:, :, -1].mean()),
            "interval_velocity_MAE_mps": float((pv - tv).abs().mean()),
            "position_loss": float(position_loss), "velocity_loss": float(velocity_loss),
            "physical_loss": float(position_loss + .2 * velocity_loss),
            "event_mean": numeric, "event_rows": event_rows}


def attach_events(root: Path, rows):
    from audit_scene_geometry import boxes_for
    for row in rows:
        sample = root / "samples" / row["case"]
        metadata = json.loads((sample / "metadata.json").read_text())
        with np.load(sample / "raw/states_xyzw.npz", allow_pickle=False) as states:
            names = states["object_names"].astype(str).tolist()
            dynamic = [i for i, name in enumerate(names) if metadata["actors"][name].get("dynamic")]
            radius = float(metadata["actors"][names[dynamic[0]]]["size_m"]["radius"])
            static = []
            for i, name in enumerate(names):
                actor = metadata["actors"][name]
                if actor.get("dynamic"):
                    continue
                if actor.get("shape") != "box":
                    continue
                static.append((states["positions"][0, i].astype(float),
                               np.array([actor["size_m"][k] for k in ("hx", "hy", "hz")], dtype=float),
                               states["quats"][0, i].astype(float)))
            history = row["positions"][:, 0].copy()
        row["event"] = lambda pred, truth, h=history, r=radius, b=static: behaviour(pred, h, truth, r, b)
        row["radius_m"] = radius


def run_arm(name, mode, batch, target, rows, mean, std, steps=200, batch_size=2):
    torch.manual_seed(SEED)
    if mode == "analytic_cv":
        with torch.no_grad():
            prediction = batch["last_position"][:, :, None] + batch["last_velocity"][:, :, None] * batch["future_dt"][:, None, :, None]
        return {"name": name, "mode": mode, "steps": 0, "initial_loss": float(objective(prediction, target, batch)),
                "final_loss": float(objective(prediction, target, batch)),
                "metrics": metrics(prediction, target, batch, rows), "state_saved": False,
                "note": "deterministic constant-velocity reference; no fitting"}, prediction
    model = Predictor(mean, std, scene_mode=mode, seed=SEED, strict_future_grid=True).cpu()
    optimizer = torch.optim.AdamW(model.active_parameters(), lr=3e-4, weight_decay=.01)
    with torch.no_grad():
        initial_prediction = model(batch)
        initial_loss = float(objective(initial_prediction, target, batch))
    losses = [initial_loss]
    start = time.monotonic()
    for step in range(1, steps + 1):
        generator = torch.Generator().manual_seed(SEED + (step - 1) // (len(rows) // batch_size))
        order = torch.randperm(len(rows), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for begin in range(0, len(rows), batch_size):
            ids = order[begin:begin + batch_size].tolist()
            part, labels = subset(batch, target, ids)
            loss = objective(model(part), labels, part) * len(ids) / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} step {step}: nonfinite loss")
            loss.backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(list(model.active_parameters()), 1., error_if_nonfinite=True)
        optimizer.step(); losses.append(total)
    model.eval()
    with torch.no_grad():
        prediction = model(batch); final_loss = float(objective(prediction, target, batch))
    return {"name": name, "mode": mode, "steps": steps, "effective_batch": batch_size,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "loss_reduction_fraction": float(1 - final_loss / initial_loss),
            "elapsed_seconds": time.monotonic() - start, "metrics": metrics(prediction, target, batch, rows),
            "loss_curve": losses, "parameters": int(sum(p.numel() for p in model.parameters())),
            "active_parameters": int(sum(p.numel() for p in model.active_parameters())),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--estimated", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 200:
        raise ValueError("CPU diagnostic budget is 1..200 steps")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    cases = list(CASES)
    est_reports = [args.estimated / c / "report.json" for c in cases]
    oracle_reports = [args.oracle / c / "report.json" for c in cases]
    guard_est = validate(args.config, protocol="diagnostic", variant="geometry_only",
                         scene_reports=est_reports, allow_legacy=True)
    guard_oracle = validate(args.config, protocol="diagnostic", variant="oracle_partial_visible_geometry",
                            scene_reports=oracle_reports)
    est_rows = load_case_rows(args.root, args.estimated, cases)
    oracle_rows = load_case_rows(args.root, args.oracle, cases)
    if [r["case"] for r in est_rows] != [r["case"] for r in oracle_rows]:
        raise ValueError("estimated/oracle case order differs")
    attach_events(args.root, est_rows); attach_events(args.root, oracle_rows)
    est_batch, target = make_batch(est_rows)
    oracle_batch, oracle_target = make_batch(oracle_rows)
    if not torch.equal(target, oracle_target):
        raise ValueError("oracle/estimated supervision changed")
    values = est_batch["motion"]
    mean, std = values.mean((0, 1)), values.std((0, 1), unbiased=False).clamp_min(1e-3)
    output = args.output.resolve(); output.mkdir(parents=True)
    results, predictions = {}, {}
    analytic, pred = run_arm("analytic_cv", "analytic_cv", est_batch, target, est_rows, mean, std, steps=args.steps)
    results["analytic_cv"] = analytic; predictions["analytic_cv"] = pred.numpy()
    motion, pred = run_arm("motion_only", "motion_only", est_batch, target, est_rows, mean, std, steps=args.steps)
    results["motion_only"] = motion; predictions["motion_only"] = pred.detach().numpy()
    geom_est, pred = run_arm("geometry_only_estimated", "geometry_only", est_batch, target, est_rows, mean, std, steps=args.steps)
    results["geometry_only_estimated"] = geom_est; predictions["geometry_only_estimated"] = pred.detach().numpy()
    geom_oracle, pred = run_arm("geometry_only_oracle_partial", "geometry_only", oracle_batch, target, oracle_rows, mean, std, steps=args.steps)
    results["geometry_only_oracle_partial"] = geom_oracle; predictions["geometry_only_oracle_partial"] = pred.detach().numpy()
    for key, result in results.items():
        state = result.pop("state_dict", None)
        if state is not None:
            torch.save(state, output / f"{key}_step{args.steps:04d}.pt")
    np.savez_compressed(output / "predictions.npz", target=target.numpy(),
                        future_dt=est_batch["future_dt"].numpy(), **predictions)
    config = json.loads(args.config.read_text())
    report = {"status": "EXECUTED", "diagnostic_only": True, "formal_training": False,
              "dit_loaded": False, "device": "cpu", "torch_threads": 2,
              "steps": args.steps, "effective_batch": 2, "seed": SEED,
              "cases": cases, "split": "train", "held_out_evaluation": False,
              "target_source": "future states only in loss/metrics; observed RGB0-7 context in inputs",
              "estimated_scene_provenance": [str(x) for x in est_reports],
              "oracle_scene_provenance": [str(x) for x in oracle_reports],
              "guard_estimated": guard_est, "guard_oracle": guard_oracle,
              "shared_motion_statistics": True, "shared_initialization_seed": SEED,
              "same_supervision": True, "results": results,
              "interpretation_limit": "Six train cases can be memorized; no generalization or scene-use claim is supported without paired/held-out data.",
              "config": config}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # A compact plot is useful for checking that all arms produced finite paths.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for key, result in results.items():
            if result.get("loss_curve"):
                ax[0].plot(result["loss_curve"], label=key)
        ax[0].set(xlabel="CPU optimizer step", ylabel="physical loss", title="6-door diagnostic fit")
        ax[0].legend(fontsize=7)
        for key, pred in predictions.items():
            if key == "analytic_cv":
                continue
            ax[1].plot(pred[:, 0, :, 0].mean(0), pred[:, 0, :, 2].mean(0), label=key)
        ax[1].set(xlabel="mean X (m)", ylabel="mean Z (m)"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(output / "fit_diagnostic.png", dpi=140); plt.close(fig)
    except Exception as exc:
        (output / "plot_status.json").write_text(json.dumps({"status": "BLOCKED", "reason": str(exc)}) + "\n")
    print(json.dumps({"status": report["status"], "arms": list(results),
                      "final_loss": {k: v["final_loss"] for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
