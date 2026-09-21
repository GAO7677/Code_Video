"""Full-budget training entry point prepared for the phase-one protocol.

This command is deliberately not invoked by the phase-one execution request:
the 432-episode replay and 200-epoch budget require a separate authorization.
It consumes the same manifest/adapter/loss as the executed 36-episode profile,
keeps group order matched across the three arms, and writes checkpoints at
50/100/200 epochs when authorized.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from future_query_predictor import objective
from run_physvideo_phase1_triarm_profile import (
    GROUP_BATCH,
    build_motion_stats,
    load_records,
    make_batch,
    make_models,
)


ARMS = ("motion_only", "point_geometry_resample", "finite_surface_geometry")
CHECKPOINT_EPOCHS = (50, 100, 200)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train_arm(name, model, groups, mean, std, epochs, group_batch, seed, output):
    active = list(model.active_parameters())
    optimizer = torch.optim.AdamW(active, lr=3e-4, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(groups) / group_batch)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = np.arange(len(groups), dtype=np.int64)
        np.random.default_rng(seed * 100003 + epoch).shuffle(order)
        epoch_losses = []
        for update in range(updates_per_epoch):
            selected = [groups[int(order[(update * group_batch + j) % len(groups)])]
                        for j in range(group_batch)]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for group in selected:
                point_seeds = [int(row["point_seed"]) + seed * 1000003 + epoch * 1009 + update for row in group]
                batch, target = make_batch(group, name, mean, std, seeds=point_seeds)
                loss = objective(model(batch), target, batch) / len(selected)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {name} loss at epoch {epoch}, update {update}")
                loss.backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(active, 1.0)
            optimizer.step()
            epoch_losses.extend(losses)
        history.append({"epoch": epoch, "mean_group_loss": float(np.mean(epoch_losses)),
                        "updates": updates_per_epoch})
        if epoch in CHECKPOINT_EPOCHS or epoch == epochs:
            torch.save({"status": "EXECUTED", "arm": name, "seed": seed, "epoch": epoch,
                        "state_dict": model.state_dict()}, output / f"{name}_seed{seed}_epoch{epoch}.pt")
    return {"status": "EXECUTED", "epochs": epochs, "updates_per_epoch": updates_per_epoch,
            "optimizer_steps": epochs * updates_per_epoch, "seconds": time.perf_counter() - started,
            "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--group-batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.epochs <= 0 or args.group_batch <= 0:
        raise ValueError("epochs and group-batch must be positive")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest, records = load_records(args.data_root.resolve())
    train_records = [row for row in records if row.get("split", "train_pilot") in ("train", "train_pilot")]
    if not train_records:
        raise ValueError("manifest has no train/train_pilot records")
    groups_by_id = {}
    for row in train_records:
        groups_by_id.setdefault(row["group_id"], []).append(row)
    groups = [sorted(rows, key=lambda row: row["geometry_value"]) for _, rows in sorted(groups_by_id.items())]
    if any(len(group) != 3 for group in groups):
        raise ValueError("every training history group must contain exactly three geometry variants")
    mean, std = build_motion_stats(train_records)
    args.output.mkdir(parents=True)
    receipt = {"status": "EXECUTED", "data_root": str(args.data_root.resolve()),
               "manifest_protocol": manifest["protocol_version"], "records": len(train_records),
               "history_groups": len(groups), "episodes_per_update": args.group_batch * 3,
               "epochs": args.epochs, "group_batch": args.group_batch,
               "optimizer_steps_per_epoch": math.ceil(len(groups) / args.group_batch),
               "loss": "SmoothL1(position)+0.2*SmoothL1(interval_velocity)",
               "normalization": "training observation motion only", "device": "cpu", "threads": 2,
               "arms": {}, "seeds": args.seeds}
    for seed in args.seeds:
        models, shared_hash = make_models(mean, std, seed=seed)
        seed_output = args.output / f"seed_{seed}"
        seed_output.mkdir()
        for name in ARMS:
            receipt["arms"][f"{name}/seed_{seed}"] = train_arm(
                name, models[name], groups, mean, std, args.epochs,
                args.group_batch, seed, seed_output,
            )
        receipt.setdefault("shared_backbone_initialization_sha256", {})[str(seed)] = shared_hash
    write_json(args.output / "training_receipt.json", receipt)
    print(json.dumps({"status": "EXECUTED", "output": str(args.output),
                      "arms": list(ARMS), "seeds": args.seeds,
                      "optimizer_steps_per_arm_seed": args.epochs * math.ceil(len(groups) / args.group_batch)}, indent=2))


if __name__ == "__main__":
    main()
