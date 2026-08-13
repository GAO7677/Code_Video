#!/usr/bin/env python3
"""Compute train-only channel statistics for latent predictor normalization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1_causal_state_probe import SLOT_DIM  # noqa: E402
from stage1_causal_state_probe.data import TrajectoryDataset  # noqa: E402
from stage1_causal_state_probe.io_utils import atomic_torch_save  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/gaoya/agent-data/cache/xssc_stage1_causal_state"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = TrajectoryDataset(args.cache_root, "train")
    total = torch.zeros(SLOT_DIM, dtype=torch.float64)
    total_square = torch.zeros(SLOT_DIM, dtype=torch.float64)
    count = 0
    for index in range(len(dataset)):
        value = dataset[index]["slots"].double().reshape(-1, SLOT_DIM)
        total += value.sum(dim=0)
        total_square += value.square().sum(dim=0)
        count += value.shape[0]
        if (index + 1) % 500 == 0:
            print(f"[stats] {index + 1}/{len(dataset)}", flush=True)
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(1e-12)
    payload = {
        "format": "xssc_stage1_slot_stats_v1",
        "split": "train",
        "records": len(dataset),
        "slot_vectors": count,
        "mean": mean.float(),
        "std": variance.sqrt().float(),
    }
    atomic_torch_save(payload, args.output.resolve())
    print(f"[stats] wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

