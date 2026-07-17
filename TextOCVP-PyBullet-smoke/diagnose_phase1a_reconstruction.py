#!/usr/bin/env python3
"""Quantify reconstruction collapse and slot quality on fixed Kubric validation clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


PROJECT = Path(__file__).resolve().parent
TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
os.chdir(TEXTOCVP_ROOT)

from data.Stage1Indexed import Stage1Indexed  # noqa: E402
from lib.setup_model import load_checkpoint, setup_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--num-validation", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def masked_rgb_mean(video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None].float()
    return (video.float() * weights).sum(dim=(0, 2, 3)) / weights.sum().clamp_min(1.0)


def contrast_retention(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    dynamic: torch.Tensor,
    background: torch.Tensor,
) -> tuple[float, float, float]:
    target_contrast = torch.linalg.vector_norm(
        masked_rgb_mean(target, dynamic) - masked_rgb_mean(target, background)
    )
    reconstruction_contrast = torch.linalg.vector_norm(
        masked_rgb_mean(reconstruction, dynamic)
        - masked_rgb_mean(reconstruction, background)
    )
    retention = reconstruction_contrast / target_contrast.clamp_min(1e-8)
    return (
        float(target_contrast.item()),
        float(reconstruction_contrast.item()),
        float(retention.item()),
    )


def variance_ratios(
    target: torch.Tensor, reconstruction: torch.Tensor
) -> tuple[float, float, float, float]:
    target_spatial = target.float().var(dim=(-2, -1), unbiased=False).mean()
    reconstruction_spatial = reconstruction.float().var(dim=(-2, -1), unbiased=False).mean()
    target_temporal = (target[1:].float() - target[:-1].float()).square().mean()
    reconstruction_temporal = (
        reconstruction[1:].float() - reconstruction[:-1].float()
    ).square().mean()
    return (
        float((reconstruction_spatial / target_spatial.clamp_min(1e-8)).item()),
        float((reconstruction_temporal / target_temporal.clamp_min(1e-8)).item()),
        float(target_temporal.item()),
        float(reconstruction_temporal.item()),
    )


def slot_reconstruction_diversity(reconstructions: torch.Tensor) -> tuple[float, float]:
    """Measure whether slots decode distinct RGB layers before alpha compositing."""
    slots = reconstructions.float().permute(1, 0, 2, 3, 4)
    pairwise_l1 = []
    for left in range(slots.shape[0]):
        for right in range(left + 1, slots.shape[0]):
            pairwise_l1.append((slots[left] - slots[right]).abs().mean())
    between_slot_variance = slots.var(dim=0, unbiased=False).mean()
    return (
        float(torch.stack(pairwise_l1).mean().item()),
        float(between_slot_variance.item()),
    )


def instance_dice(masks: torch.Tensor, instances: torch.Tensor) -> list[float]:
    slots = masks.permute(1, 0, 2, 3).reshape(masks.shape[1], -1).float()
    targets = instances.permute(1, 0, 2, 3).reshape(instances.shape[1], -1).float()
    intersection = slots @ targets.transpose(0, 1)
    denominator = slots.sum(dim=1, keepdim=True) + targets.sum(dim=1)[None]
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    rows, columns = linear_sum_assignment((-dice).cpu().numpy())
    assignment = {column: row for row, column in zip(rows.tolist(), columns.tolist())}
    return [float(dice[assignment[index], index].item()) for index in range(targets.shape[0])]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    resolution = tuple(config["dataset"]["img_size"])
    dataset = Stage1Indexed(
        index_root=args.index_root,
        dataset_mode="kubric",
        split="valid",
        num_frames=10,
        img_size=resolution,
        frame_stride=1,
        random_start=False,
        preprocess_mode="resize",
        load_masks=True,
        max_mask_instances=6,
        mask_temporal_stride=1,
        mask_spatial_stride=1,
    )
    indices = np.random.default_rng(args.seed).choice(
        len(dataset), size=args.num_validation, replace=False
    ).tolist()
    device = torch.device(f"cuda:{args.gpu}")
    model = setup_model(config["model"])
    model = load_checkpoint(
        checkpoint_path=str(args.checkpoint), model=model, only_model=True, map_cpu=True
    ).eval().to(device)

    reports = []
    for index in indices:
        video, metadata = dataset[index]
        targets = metadata.pop("_mask_targets")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            output = model(x=video[None].to(device), num_imgs=10, decode=True)
        reconstruction = output["recons_imgs"][0].float().cpu().clamp(0, 1)
        slot_reconstructions = output["recons_objs"][0].float().cpu()
        masks = output["masks"][0].float().cpu().squeeze(2)
        soft_usage = masks.mean(dim=(0, 2, 3))
        hard_labels = masks.argmax(dim=1)
        hard_usage = torch.stack(
            [(hard_labels == slot).float().mean() for slot in range(masks.shape[1])]
        )
        entropy = -(masks.clamp_min(1e-8) * masks.clamp_min(1e-8).log()).sum(dim=1)
        normalized_entropy = entropy.mean() / math.log(masks.shape[1])
        adjacent_mask_change = (masks[1:] - masks[:-1]).abs().mean()

        dynamic = targets["dynamic_union_mask"][:, 0] > 0.5
        static = targets["static_geometry_mask"][:, 0] > 0.5
        background = ~(dynamic | static)
        gt_contrast, reconstruction_contrast, retention = contrast_retention(
            video, reconstruction, dynamic, background
        )
        spatial_ratio, temporal_ratio, target_temporal, reconstruction_temporal = (
            variance_ratios(video, reconstruction)
        )
        slot_pairwise_l1, slot_between_variance = slot_reconstruction_diversity(
            slot_reconstructions
        )
        valid_instances = targets["dynamic_instance_valid"].bool()
        dice_values = instance_dice(
            masks, targets["dynamic_instance_masks"][:, valid_instances]
        ) if bool(valid_instances.any()) else []
        reports.append(
            {
                "dataset_index": index,
                "sample_id": metadata["sample_id"],
                "soft_slot_usage": soft_usage.tolist(),
                "hard_slot_usage": hard_usage.tolist(),
                "soft_usage_max": float(soft_usage.max().item()),
                "hard_usage_max": float(hard_usage.max().item()),
                "normalized_mask_entropy": float(normalized_entropy.item()),
                "adjacent_mask_l1": float(adjacent_mask_change.item()),
                "instance_dice": dice_values,
                "mean_instance_dice": finite_mean(dice_values),
                "dynamic_area": float(dynamic.float().mean().item()),
                "static_area": float(static.float().mean().item()),
                "background_area": float(background.float().mean().item()),
                "gt_dynamic_background_contrast": gt_contrast,
                "reconstruction_dynamic_background_contrast": reconstruction_contrast,
                "dynamic_contrast_retention": retention,
                "spatial_variance_retention": spatial_ratio,
                "temporal_variance_retention": temporal_ratio,
                "target_temporal_mse": target_temporal,
                "reconstruction_temporal_mse": reconstruction_temporal,
                "slot_reconstruction_pairwise_l1": slot_pairwise_l1,
                "slot_reconstruction_between_variance": slot_between_variance,
            }
        )

    all_dice = [value for report in reports for value in report["instance_dice"]]
    aggregate = {
        "validation_indices": indices,
        "samples": len(reports),
        "mean_soft_slot_usage": np.mean(
            [report["soft_slot_usage"] for report in reports], axis=0
        ).tolist(),
        "mean_hard_slot_usage": np.mean(
            [report["hard_slot_usage"] for report in reports], axis=0
        ).tolist(),
        "mean_soft_usage_max": finite_mean([r["soft_usage_max"] for r in reports]),
        "mean_hard_usage_max": finite_mean([r["hard_usage_max"] for r in reports]),
        "mean_normalized_mask_entropy": finite_mean(
            [r["normalized_mask_entropy"] for r in reports]
        ),
        "mean_adjacent_mask_l1": finite_mean([r["adjacent_mask_l1"] for r in reports]),
        "mean_instance_dice": finite_mean(all_dice),
        "mean_dynamic_area": finite_mean([r["dynamic_area"] for r in reports]),
        "mean_static_area": finite_mean([r["static_area"] for r in reports]),
        "mean_background_area": finite_mean([r["background_area"] for r in reports]),
        "mean_dynamic_contrast_retention": finite_mean(
            [r["dynamic_contrast_retention"] for r in reports]
        ),
        "mean_spatial_variance_retention": finite_mean(
            [r["spatial_variance_retention"] for r in reports]
        ),
        "mean_temporal_variance_retention": finite_mean(
            [r["temporal_variance_retention"] for r in reports]
        ),
        "median_temporal_variance_retention": float(
            np.median([r["temporal_variance_retention"] for r in reports])
        ),
        "mean_target_temporal_mse": finite_mean(
            [r["target_temporal_mse"] for r in reports]
        ),
        "mean_reconstruction_temporal_mse": finite_mean(
            [r["reconstruction_temporal_mse"] for r in reports]
        ),
        "mean_slot_reconstruction_pairwise_l1": finite_mean(
            [r["slot_reconstruction_pairwise_l1"] for r in reports]
        ),
        "mean_slot_reconstruction_between_variance": finite_mean(
            [r["slot_reconstruction_between_variance"] for r in reports]
        ),
    }
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "aggregate": aggregate,
        "samples": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
