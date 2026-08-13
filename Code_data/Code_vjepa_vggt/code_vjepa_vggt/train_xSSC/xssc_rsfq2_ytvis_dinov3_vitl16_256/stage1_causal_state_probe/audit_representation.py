#!/usr/bin/env python3
"""Measure causal slot stability without training a future predictor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1_causal_state_probe import CALIBRATION_STATES, STATIC_DIM  # noqa: E402
from stage1_causal_state_probe.alignment import (  # noqa: E402
    assignment_switch_rate,
    hard_slot_masks,
    pairwise_prefix_iou,
    per_frame_oracle_assignments,
)
from stage1_causal_state_probe.data import TrajectoryDataset  # noqa: E402
from stage1_causal_state_probe.io_utils import atomic_write_json, read_yaml  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/stage1_movic.yaml",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fixed_future_iou(attention, masks, mapping, object_valid):
    values = []
    for time_index in range(CALIBRATION_STATES, attention.shape[0]):
        pairwise = pairwise_prefix_iou(
            hard_slot_masks(attention[time_index : time_index + 1]),
            masks[time_index : time_index + 1],
            object_valid,
        )
        valid_slots = mapping >= 0
        if bool(valid_slots.any()):
            slot_indices = torch.arange(len(mapping))[valid_slots]
            values.append(pairwise[slot_indices, mapping[valid_slots]])
    if not values:
        return 0.0
    return float(torch.cat(values).mean())


def oracle_future_iou(attention, masks, assignments, object_valid):
    values = []
    for time_index in range(CALIBRATION_STATES, attention.shape[0]):
        pairwise = pairwise_prefix_iou(
            hard_slot_masks(attention[time_index : time_index + 1]),
            masks[time_index : time_index + 1],
            object_valid,
        )
        mapping = assignments[time_index]
        valid_slots = mapping >= 0
        if bool(valid_slots.any()):
            slots = torch.arange(len(mapping))[valid_slots]
            values.append(pairwise[slots, mapping[valid_slots]])
    if not values:
        return 0.0
    return float(torch.cat(values).mean())


def audit_case(record):
    slots = record["slots"].float()
    static = slots[..., :STATIC_DIM]
    dynamic = slots[..., STATIC_DIM:]
    static_drift = (static[1:] - static[:-1]).square().mean(dim=-1).sqrt()
    dynamic_drift = (dynamic[1:] - dynamic[:-1]).square().mean(dim=-1).sqrt()
    adjacent_cosine = functional.cosine_similarity(slots[1:], slots[:-1], dim=-1)
    attention = record["slot_attention"].float()
    masks = record["gt_mask"].bool()
    object_valid = record["object_valid"].bool()
    oracle = per_frame_oracle_assignments(attention, masks, object_valid)
    oracle_iou = oracle_future_iou(attention, masks, oracle, object_valid)

    result = {
        "video_index": int(record["source"]["index"]),
        "video_name": record["source"]["video_name"],
        "static_drift": float(static_drift.mean()),
        "dynamic_drift": float(dynamic_drift.mean()),
        "adjacent_slot_cosine": float(adjacent_cosine.mean()),
        "per_frame_oracle_switch_rate": assignment_switch_rate(oracle),
        "per_frame_oracle_future_iou": oracle_iou,
    }
    for name, key in (
        ("prefix", "prefix_slot_to_object"),
        ("boundary", "boundary_slot_to_object"),
    ):
        mapping = record[key].long()
        matched = int((mapping >= 0).sum())
        coverage = matched / max(int(object_valid.sum()), 1)
        fixed_iou = fixed_future_iou(attention, masks, mapping, object_valid)
        result[f"{name}_coverage"] = coverage
        result[f"{name}_future_iou"] = fixed_iou
        result[f"{name}_oracle_iou_gap"] = oracle_iou - fixed_iou
    return result


def main():
    args = parse_args()
    config = read_yaml(args.stage_config.resolve())
    dataset = TrajectoryDataset(Path(config["paths"]["cache_root"]), args.split)
    stop = len(dataset) if args.max_cases is None else min(len(dataset), args.max_cases)
    cases = [audit_case(dataset[index]) for index in range(stop)]
    metric_keys = [key for key in cases[0] if key not in {"video_index", "video_name"}]
    summary = {
        "format": "xssc_stage1_representation_audit_v1",
        "split": args.split,
        "videos": len(cases),
        "metrics": {
            key: {
                "mean": float(np.mean([case[key] for case in cases])),
                "std_between_videos": float(np.std([case[key] for case in cases])),
            }
            for key in metric_keys
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, output_dir / "summary.json")
    atomic_write_json(cases, output_dir / "cases.json")
    with (output_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

