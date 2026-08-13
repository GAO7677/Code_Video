#!/usr/bin/env python3
"""Report the physical readout ceiling on real causal xSSC slots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1_causal_state_probe.data import (  # noqa: E402
    TrajectoryDataset,
    gather_object_targets,
)
from stage1_causal_state_probe.evaluate_stage1 import (  # noqa: E402
    MetricAccumulator,
    denormalize_probe_output,
)
from stage1_causal_state_probe.io_utils import atomic_write_json, read_yaml  # noqa: E402
from stage1_causal_state_probe.models import (  # noqa: E402
    FrozenGTProbes,
    SlotNormalizer,
    bbox_iou,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/stage1_movic.yaml",
    )
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--mapping", choices=("prefix", "boundary"), default="prefix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def evaluate_case(record, probe, normalizer, target_stats, mapping_key, device):
    slots = normalizer.normalize(record["slots"].float().to(device))
    target, valid = gather_object_targets(record, mapping_key)
    target = {key: value.to(device) for key, value in target.items()}
    valid = valid.to(device)
    visible = valid & target["presence"]
    output = denormalize_probe_output(probe(slots), target_stats)
    metrics = MetricAccumulator()
    position_rmse = (output["position"] - target["position"]).square().mean(-1).sqrt()
    velocity_normalized = (
        output["velocity"] - target["velocity"]
    ) / target_stats["velocity"]["std"].to(output["velocity"])
    center_error = (output["image_position"] - target["image_position"]).norm(dim=-1)
    metrics.add("position_rmse", position_rmse[valid])
    metrics.add(
        "velocity_nrmse", velocity_normalized.square().mean(-1).sqrt()[valid]
    )
    metrics.add("center_error", center_error[valid])
    if bool(visible.any()):
        metrics.add("bbox_iou", bbox_iou(output["bbox"], target["bbox"])[visible])
    prediction = output["presence_logit"] >= 0
    truth = target["presence"]
    metrics.add("presence_accuracy", (prediction[valid] == truth[valid]).float())
    tp = float((prediction[valid] & truth[valid]).sum())
    fp = float((prediction[valid] & ~truth[valid]).sum())
    fn = float((~prediction[valid] & truth[valid]).sum())
    tn = float((~prediction[valid] & ~truth[valid]).sum())
    result = metrics.result()
    result["presence_f1"] = 2 * tp / max(2 * tp + fp + fn, 1e-8)
    result["presence_balanced_accuracy"] = 0.5 * (
        tp / max(tp + fn, 1e-8) + tn / max(tn + fp, 1e-8)
    )
    return result


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device == "cuda:4":
        raise ValueError("GPU 4 is prohibited by workspace policy")
    config = read_yaml(args.stage_config.resolve())
    payload = torch.load(args.probe.resolve(), map_location="cpu", weights_only=True)
    if payload["mapping"] != args.mapping:
        raise ValueError("Probe mapping does not match requested evaluation mapping")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    probe = FrozenGTProbes(payload["representation"])
    probe.load_state_dict(payload["model"], strict=True)
    probe = probe.to(device).eval()
    normalizer = SlotNormalizer.from_state_dict(payload["slot_normalizer"])
    dataset = TrajectoryDataset(Path(config["paths"]["cache_root"]), args.split)
    stop = len(dataset) if args.max_cases is None else min(len(dataset), args.max_cases)
    mapping_key = f"{args.mapping}_slot_to_object"
    rows = []
    for index in range(stop):
        record = dataset[index]
        rows.append(
            {
                "video_index": int(record["source"]["index"]),
                "video_name": record["source"]["video_name"],
                **evaluate_case(
                    record,
                    probe,
                    normalizer,
                    payload["target_stats"],
                    mapping_key,
                    device,
                ),
            }
        )
    metric_keys = [key for key in rows[0] if key not in {"video_index", "video_name"}]
    summary = {
        "format": "xssc_stage1_gt_probe_ceiling_v1",
        "split": args.split,
        "mapping": args.mapping,
        "representation": payload["representation"],
        "videos": len(rows),
        "metrics": {
            key: float(np.mean([row[key] for row in rows])) for key in metric_keys
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary, output_dir / "summary.json")
    with (output_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
