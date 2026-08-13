#!/usr/bin/env python3
"""Train one cell of the Stage-1 history/context factorial matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1_causal_state_probe.data import PredictionWindowDataset  # noqa: E402
from stage1_causal_state_probe.io_utils import (  # noqa: E402
    atomic_torch_save,
    atomic_write_json,
    read_yaml,
)
from stage1_causal_state_probe.models import (  # noqa: E402
    CONTEXT_MODES,
    REPRESENTATIONS,
    SlotNormalizer,
    StatePredictor,
    normalized_prediction_loss,
    representation_target,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/stage1_movic.yaml",
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--representation", choices=sorted(REPRESENTATIONS), required=True)
    parser.add_argument("--history", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--context", choices=sorted(CONTEXT_MODES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--wandb-project")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, config):
    params = config["predictor"]
    return StatePredictor(
        representation=args.representation,
        history=args.history,
        context_mode=args.context,
        model_dim=int(params["model_dim"]),
        num_heads=int(params["num_heads"]),
        feedforward_dim=int(params["feedforward_dim"]),
        temporal_layers=int(params["temporal_layers"]),
        context_layers=int(params["context_layers"]),
        dropout=float(params["dropout"]),
    )


def run_epoch(model, loader, normalizer, representation, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        history = normalizer.normalize(batch["history"].to(device))
        target_full = normalizer.normalize(batch["target"].to(device))
        target = representation_target(target_full, representation)
        slot_valid = batch["slot_valid"].to(device)
        with torch.set_grad_enabled(training):
            prediction = model(history, slot_valid=slot_valid)
            loss = normalized_prediction_loss(
                prediction, target, representation, slot_valid=slot_valid
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), loader.gradient_clip_norm
            )
            optimizer.step()
        batch_size = history.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def main():
    args = parse_args()
    if args.device == "cuda:4":
        raise ValueError("GPU 4 is prohibited by workspace policy")
    config = read_yaml(args.stage_config.resolve())
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    stats = torch.load(args.stats.resolve(), map_location="cpu", weights_only=True)
    normalizer = SlotNormalizer.from_state_dict(stats)
    cache_root = Path(config["paths"]["cache_root"])
    params = config["predictor"]
    train_data = PredictionWindowDataset(
        cache_root, "train", history=args.history
    )
    val_data = PredictionWindowDataset(
        cache_root, "validation", history=args.history
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader.gradient_clip_norm = float(params["gradient_clip_norm"])
    val_loader.gradient_clip_norm = float(params["gradient_clip_norm"])

    model = build_model(args, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(config["paths"]["checkpoint_root"])
            / "predictors"
            / args.representation
            / f"h{args.history}_{args.context}"
            / f"seed_{args.seed}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{args.representation}-h{args.history}-{args.context}-s{args.seed}",
            config={
                "representation": args.representation,
                "history": args.history,
                "context": args.context,
                "seed": args.seed,
                **params,
            },
        )

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history_log = []
    for epoch in range(1, int(params["max_epochs"]) + 1):
        train_loss = run_epoch(
            model, train_loader, normalizer, args.representation, device, optimizer
        )
        with torch.inference_mode():
            val_loss = run_epoch(
                model, val_loader, normalizer, args.representation, device
            )
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history_log.append(row)
        print(json.dumps(row), flush=True)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "format": "xssc_stage1_state_predictor_v1",
                    "representation": args.representation,
                    "history": args.history,
                    "context": args.context,
                    "seed": args.seed,
                    "model_config": params,
                    "model": model.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= int(params["patience"]):
            break

    summary = {
        "representation": args.representation,
        "history": args.history,
        "context": args.context,
        "seed": args.seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs": history_log,
    }
    atomic_write_json(summary, output_dir / "summary.json")
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()

