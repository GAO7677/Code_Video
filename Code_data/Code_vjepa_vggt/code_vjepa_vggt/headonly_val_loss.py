from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset


@dataclass
class HeadOnlyValConfig:
    enabled: bool
    split: str
    every_steps: int | None
    num_batches: int


def build_headonly_val_config(args) -> HeadOnlyValConfig:
    enabled = (
        bool(getattr(args, "enable_object_branch", False))
        and str(getattr(args, "dataset_type", "")) == "phys_state_episode"
        and getattr(args, "headonly_val_loss_every_steps", None) is not None
    )
    return HeadOnlyValConfig(
        enabled=enabled,
        split=str(getattr(args, "headonly_val_loss_split", "val")),
        every_steps=getattr(args, "headonly_val_loss_every_steps", None),
        num_batches=max(1, int(getattr(args, "headonly_val_loss_num_batches", 8))),
    )


def should_run_headonly_val_loss(config: HeadOnlyValConfig, global_step: int) -> bool:
    return (
        config.enabled
        and config.every_steps is not None
        and global_step > 0
        and global_step % int(config.every_steps) == 0
    )


def build_headonly_val_dataset(args, config: HeadOnlyValConfig):
    if not config.enabled:
        return None
    return PhysStateEpisodeDataset(
        root=args.phys_state_root,
        split=config.split,
        resolution=(args.height, args.width),
        num_context_frames=args.fixed_num_context_frames,
        context_fraction=0.5,
        random_context_frames=False,
        seed=42,
    )


def build_headonly_val_dataloader(dataset, args):
    if dataset is None:
        return None
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
        num_workers=args.dataset_num_workers,
    )


def _rename_train_metric_key(key: str) -> str:
    if key.startswith("train/"):
        return "val/" + key[len("train/") :]
    return "val/" + key


def _mean_metrics(across_batches: Iterable[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in across_batches:
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / max(counts[key], 1) for key in sums}


def run_headonly_val_loss(
    *,
    accelerator,
    model,
    val_dataloader,
    global_step: int,
    num_batches: int,
) -> dict[str, float]:
    if val_dataloader is None:
        return {}

    wrapped_model = model
    unwrapped_model = accelerator.unwrap_model(model)
    was_training = wrapped_model.training
    wrapped_model.eval()

    local_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(val_dataloader):
            if batch_index >= int(num_batches):
                break
            loss = wrapped_model(batch)
            batch_metrics = dict(getattr(unwrapped_model, "last_train_metrics", {}))
            batch_metrics["train/loss_total"] = float(loss.detach().item())
            local_metrics.append(batch_metrics)

    if was_training:
        wrapped_model.train()

    mean_local_metrics = _mean_metrics(local_metrics)
    if not mean_local_metrics:
        return {}

    metric_keys = sorted(mean_local_metrics.keys())
    values = torch.tensor(
        [mean_local_metrics[key] for key in metric_keys],
        device=accelerator.device,
        dtype=torch.float64,
    )
    counts = torch.tensor(
        [1.0 if key in mean_local_metrics else 0.0 for key in metric_keys],
        device=accelerator.device,
        dtype=torch.float64,
    )
    reduced_values = accelerator.reduce(values, reduction="mean")
    reduced_counts = accelerator.reduce(counts, reduction="sum")

    metrics = {}
    for index, key in enumerate(metric_keys):
        if float(reduced_counts[index].item()) <= 0.0:
            continue
        metrics[_rename_train_metric_key(key)] = float(reduced_values[index].item())

    if metrics:
        metrics["val/num_batches"] = float(num_batches)
        metrics["val/failed"] = 0.0
        accelerator.log(metrics, step=global_step)
    return metrics
