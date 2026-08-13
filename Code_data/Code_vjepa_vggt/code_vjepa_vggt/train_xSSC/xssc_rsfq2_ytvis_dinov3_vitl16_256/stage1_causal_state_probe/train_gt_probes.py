#!/usr/bin/env python3
"""Train simple GT readouts on real causal slots, then freeze them."""

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

from stage1_causal_state_probe.data import TrajectoryDataset  # noqa: E402
from stage1_causal_state_probe.io_utils import (  # noqa: E402
    atomic_torch_save,
    atomic_write_json,
    read_yaml,
)
from stage1_causal_state_probe.models import (  # noqa: E402
    REPRESENTATIONS,
    FrozenGTProbes,
    SlotNormalizer,
    probe_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/stage1_movic.yaml",
    )
    parser.add_argument("--slot-stats", type=Path, required=True)
    parser.add_argument("--representation", choices=sorted(REPRESENTATIONS), required=True)
    parser.add_argument("--mapping", choices=("prefix", "boundary"), default="prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mapped_targets(batch, mapping_key, device):
    mapping = batch[mapping_key].to(device).long()
    clamped = mapping.clamp_min(0)
    object_valid = batch["object_valid"].to(device).bool()
    mapped_valid = (mapping >= 0) & object_valid.gather(1, clamped)
    time_steps = batch["slots"].shape[1]

    def gather(key):
        value = batch[key].to(device)
        index = clamped[:, None, :, None].expand(
            -1, time_steps, -1, value.shape[-1]
        )
        return value.gather(2, index)

    visibility = batch["gt_visibility"].to(device).gather(
        2, clamped[:, None].expand(-1, time_steps, -1)
    )
    target = {
        "position": gather("gt_position"),
        "velocity": gather("gt_velocity"),
        "image_position": gather("gt_image_position"),
        "bbox": gather("gt_bbox"),
        "presence": visibility > 0,
    }
    valid = mapped_valid[:, None].expand(-1, time_steps, -1)
    return target, valid


def fit_target_stats(loader, mapping_key):
    values = {key: [] for key in ("position", "velocity", "image_position")}
    for batch in loader:
        target, valid = mapped_targets(batch, mapping_key, torch.device("cpu"))
        for key in values:
            values[key].append(target[key][valid].double())
    stats = {}
    for key, chunks in values.items():
        value = torch.cat(chunks)
        stats[key] = {
            "mean": value.mean(dim=0).float(),
            "std": value.std(dim=0, unbiased=False).clamp_min(1e-6).float(),
        }
    return stats


def normalize_targets(target, stats):
    target = dict(target)
    for key, value_stats in stats.items():
        target[key] = (
            target[key] - value_stats["mean"].to(target[key])
        ) / value_stats["std"].to(target[key])
    return target


def run_epoch(
    model,
    loader,
    slot_normalizer,
    target_stats,
    mapping_key,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    examples = 0
    components = {}
    for batch in loader:
        slots = slot_normalizer.normalize(batch["slots"].to(device).float())
        target, valid = mapped_targets(batch, mapping_key, device)
        target = normalize_targets(target, target_stats)
        with torch.set_grad_enabled(training):
            output = model(slots)
            loss, losses = probe_loss(output, target, valid)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = slots.shape[0]
        total += float(loss.detach()) * batch_size
        examples += batch_size
        for key, value in losses.items():
            components[key] = components.get(key, 0.0) + float(value.detach()) * batch_size
    return total / max(examples, 1), {
        key: value / max(examples, 1) for key, value in components.items()
    }


def main():
    args = parse_args()
    if args.device == "cuda:4":
        raise ValueError("GPU 4 is prohibited by workspace policy")
    config = read_yaml(args.stage_config.resolve())
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cache_root = Path(config["paths"]["cache_root"])
    params = config["probe"]
    mapping_key = f"{args.mapping}_slot_to_object"
    slot_stats = torch.load(
        args.slot_stats.resolve(), map_location="cpu", weights_only=True
    )
    slot_normalizer = SlotNormalizer.from_state_dict(slot_stats)

    train_data = TrajectoryDataset(cache_root, "train")
    val_data = TrajectoryDataset(cache_root, "validation")
    train_loader = DataLoader(
        train_data,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    stats_loader = DataLoader(
        train_data,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(params["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    target_stats = fit_target_stats(stats_loader, mapping_key)

    model = FrozenGTProbes(args.representation).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(config["paths"]["checkpoint_root"])
            / "gt_probes"
            / args.representation
            / args.mapping
            / f"seed_{args.seed}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, int(params["max_epochs"]) + 1):
        train_loss, train_parts = run_epoch(
            model,
            train_loader,
            slot_normalizer,
            target_stats,
            mapping_key,
            device,
            optimizer,
        )
        with torch.inference_mode():
            val_loss, val_parts = run_epoch(
                model,
                val_loader,
                slot_normalizer,
                target_stats,
                mapping_key,
                device,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_components": train_parts,
            "val_components": val_parts,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                {
                    "format": "xssc_stage1_gt_probes_v1",
                    "representation": args.representation,
                    "mapping": args.mapping,
                    "seed": args.seed,
                    "model": model.state_dict(),
                    "slot_normalizer": slot_normalizer.state_dict(),
                    "target_stats": target_stats,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= int(params["patience"]):
            break

    atomic_write_json(
        {
            "representation": args.representation,
            "mapping": args.mapping,
            "seed": args.seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "epochs": history,
        },
        output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()

