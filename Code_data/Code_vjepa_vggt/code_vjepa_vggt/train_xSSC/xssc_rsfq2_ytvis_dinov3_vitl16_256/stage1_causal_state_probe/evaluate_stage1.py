#!/usr/bin/env python3
"""Evaluate one predictor/probe pair on fixed-identity open-loop rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1_causal_state_probe.data import (  # noqa: E402
    TrajectoryDataset,
    gather_object_targets,
)
from stage1_causal_state_probe.io_utils import atomic_write_json, read_yaml  # noqa: E402
from stage1_causal_state_probe.models import (  # noqa: E402
    FrozenGTProbes,
    SlotNormalizer,
    StatePredictor,
    bbox_iou,
    compose_full_state,
    representation_target,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/stage1_movic.yaml",
    )
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--mapping", choices=("prefix", "boundary"), default="prefix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def build_predictor(payload, device):
    params = payload["model_config"]
    model = StatePredictor(
        representation=payload["representation"],
        history=int(payload["history"]),
        context_mode=payload["context"],
        model_dim=int(params["model_dim"]),
        num_heads=int(params["num_heads"]),
        feedforward_dim=int(params["feedforward_dim"]),
        temporal_layers=int(params["temporal_layers"]),
        context_layers=int(params["context_layers"]),
        dropout=float(params["dropout"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def denormalize_probe_output(output, target_stats):
    output = dict(output)
    for key in ("position", "velocity", "image_position"):
        output[key] = (
            output[key] * target_stats[key]["std"].to(output[key])
            + target_stats[key]["mean"].to(output[key])
        )
    return output


class MetricAccumulator:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def add(self, key, values):
        values = torch.as_tensor(values).detach().float().flatten()
        values = values[torch.isfinite(values)]
        if not values.numel():
            return
        self.sums[key] = self.sums.get(key, 0.0) + float(values.sum())
        self.counts[key] = self.counts.get(key, 0) + int(values.numel())

    def mean(self, key):
        return self.sums.get(key, 0.0) / max(self.counts.get(key, 0), 1)

    def result(self):
        return {key: self.mean(key) for key in sorted(self.sums)}


def rollout(model, history, horizon, representation, slot_valid):
    values = [history[:, index] for index in range(history.shape[1])]
    predictions = []
    for _ in range(horizon):
        model_history = torch.stack(values[-model.history :], dim=1)
        prediction = model(model_history, slot_valid=slot_valid)
        next_full = compose_full_state(values[-1], prediction, representation)
        values.append(next_full)
        predictions.append(next_full)
    return torch.stack(predictions, dim=1)


def evaluate_case(
    record,
    predictor,
    probe,
    normalizer,
    target_stats,
    representation,
    history,
    horizons,
    mapping_key,
    device,
):
    raw_slots = record["slots"].float().to(device)
    slots = normalizer.normalize(raw_slots)[None]
    slot_valid = record["slot_valid"].bool().to(device)[None]
    targets, mapped_valid = gather_object_targets(record, mapping_key)
    targets = {key: value.to(device) for key, value in targets.items()}
    mapped_valid = mapped_valid.to(device)
    results = {}
    for horizon in horizons:
        metrics = MetricAccumulator()
        origins = range(max(3, history - 1), raw_slots.shape[0] - horizon)
        for origin in origins:
            history_value = slots[:, origin - history + 1 : origin + 1]
            predicted = rollout(
                predictor, history_value, horizon, representation, slot_valid
            )[0]
            actual_norm = slots[0, origin + 1 : origin + horizon + 1]
            actual_raw = raw_slots[origin + 1 : origin + horizon + 1]
            predicted_raw = normalizer.denormalize(predicted)
            predicted_repr = representation_target(predicted, representation)
            actual_repr = representation_target(actual_norm, representation)
            predicted_raw_repr = representation_target(predicted_raw, representation)
            actual_raw_repr = representation_target(actual_raw, representation)
            latent_error = (predicted_repr - actual_repr).square().mean(dim=-1)
            latent_cosine = 1 - functional.cosine_similarity(
                predicted_raw_repr, actual_raw_repr, dim=-1
            )
            metrics.add("latent_nmse", latent_error)
            metrics.add("latent_cosine_distance", latent_cosine)

            output = denormalize_probe_output(probe(predicted), target_stats)
            ceiling = denormalize_probe_output(probe(actual_norm), target_stats)
            time_slice = slice(origin + 1, origin + horizon + 1)
            valid = mapped_valid[time_slice]
            visible = valid & targets["presence"][time_slice]

            for name in ("position", "velocity", "image_position"):
                target = targets[name][time_slice]
                error = (output[name] - target).norm(dim=-1)
                ceiling_error = (ceiling[name] - target).norm(dim=-1)
                metrics.add(f"{name}_error", error[valid])
                metrics.add(f"ceiling_{name}_error", ceiling_error[valid])
            position_rmse = (
                (output["position"] - targets["position"][time_slice])
                .square()
                .mean(dim=-1)
                .sqrt()
            )
            velocity_normalized_error = (
                output["velocity"] - targets["velocity"][time_slice]
            ) / target_stats["velocity"]["std"].to(output["velocity"])
            velocity_nrmse = velocity_normalized_error.square().mean(dim=-1).sqrt()
            metrics.add("position_rmse", position_rmse[valid])
            metrics.add("velocity_nrmse", velocity_nrmse[valid])
            image_error = (
                output["image_position"] - targets["image_position"][time_slice]
            ).norm(dim=-1)
            normalized_image_error = image_error / math.sqrt(2.0)
            metrics.add("center_ade", normalized_image_error[valid])
            metrics.add("center_fde", normalized_image_error[-1][valid[-1]])
            if bool(visible.any()):
                metrics.add(
                    "bbox_iou",
                    bbox_iou(output["bbox"], targets["bbox"][time_slice])[visible],
                )
                metrics.add(
                    "ceiling_bbox_iou",
                    bbox_iou(ceiling["bbox"], targets["bbox"][time_slice])[visible],
                )
            presence_prediction = output["presence_logit"] >= 0
            presence_target = targets["presence"][time_slice]
            metrics.add(
                "presence_accuracy",
                (presence_prediction[valid] == presence_target[valid]).float(),
            )
            metrics.add(
                "presence_tp",
                (presence_prediction[valid] & presence_target[valid]).float().sum()[None],
            )
            metrics.add(
                "presence_fp",
                (presence_prediction[valid] & ~presence_target[valid]).float().sum()[None],
            )
            metrics.add(
                "presence_fn",
                (~presence_prediction[valid] & presence_target[valid]).float().sum()[None],
            )
            metrics.add(
                "presence_tn",
                (~presence_prediction[valid] & ~presence_target[valid]).float().sum()[None],
            )
        values = metrics.result()
        tp = metrics.sums.get("presence_tp", 0.0)
        fp = metrics.sums.get("presence_fp", 0.0)
        fn = metrics.sums.get("presence_fn", 0.0)
        tn = metrics.sums.get("presence_tn", 0.0)
        values["presence_f1"] = 2 * tp / max(2 * tp + fp + fn, 1e-8)
        tpr = tp / max(tp + fn, 1e-8)
        tnr = tn / max(tn + fp, 1e-8)
        values["presence_balanced_accuracy"] = 0.5 * (tpr + tnr)
        values["origins"] = len(tuple(origins))
        results[str(horizon)] = values
    return results


def aggregate_cases(cases, horizons):
    summary = {}
    for horizon in horizons:
        rows = [case["metrics"][str(horizon)] for case in cases]
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        summary[str(horizon)] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in keys
            if key != "origins"
        }
        summary[str(horizon)]["videos"] = len(rows)
    return summary


def write_case_csv(cases, path):
    rows = []
    for case in cases:
        for horizon, metrics in case["metrics"].items():
            rows.append(
                {
                    "video_index": case["video_index"],
                    "video_name": case["video_name"],
                    "horizon": int(horizon),
                    **metrics,
                }
            )
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device == "cuda:4":
        raise ValueError("GPU 4 is prohibited by workspace policy")
    config = read_yaml(args.stage_config.resolve())
    predictor_payload = torch.load(
        args.predictor.resolve(), map_location="cpu", weights_only=True
    )
    probe_payload = torch.load(
        args.probe.resolve(), map_location="cpu", weights_only=True
    )
    representation = predictor_payload["representation"]
    if probe_payload["representation"] != representation:
        raise ValueError("Predictor and probe representations do not match")
    if probe_payload["mapping"] != args.mapping:
        raise ValueError("Probe mapping does not match evaluation mapping")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    predictor = build_predictor(predictor_payload, device)
    probe = FrozenGTProbes(representation)
    probe.load_state_dict(probe_payload["model"], strict=True)
    probe = probe.to(device).eval()
    normalizer = SlotNormalizer.from_state_dict(
        predictor_payload["normalizer"]
    )
    probe_normalizer = SlotNormalizer.from_state_dict(
        probe_payload["slot_normalizer"]
    )
    if not (
        torch.equal(normalizer.mean, probe_normalizer.mean)
        and torch.equal(normalizer.std, probe_normalizer.std)
    ):
        raise ValueError("Predictor and probe use different slot normalization")

    horizons = list(config["evaluation"]["horizons"])
    horizons.append(int(config["evaluation"]["stress_horizon"]))
    dataset = TrajectoryDataset(Path(config["paths"]["cache_root"]), args.split)
    stop = len(dataset) if args.max_cases is None else min(len(dataset), args.max_cases)
    mapping_key = f"{args.mapping}_slot_to_object"
    cases = []
    for index in range(stop):
        record = dataset[index]
        metrics = evaluate_case(
            record,
            predictor,
            probe,
            normalizer,
            probe_payload["target_stats"],
            representation,
            int(predictor_payload["history"]),
            horizons,
            mapping_key,
            device,
        )
        cases.append(
            {
                "video_index": int(record["source"]["index"]),
                "video_name": record["source"]["video_name"],
                "metrics": metrics,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"[evaluate] {index + 1}/{stop}", flush=True)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "xssc_stage1_evaluation_v1",
        "split": args.split,
        "mapping": args.mapping,
        "predictor": str(args.predictor.resolve()),
        "probe": str(args.probe.resolve()),
        "representation": representation,
        "history": int(predictor_payload["history"]),
        "context": predictor_payload["context"],
        "seed": int(predictor_payload["seed"]),
        "horizons": horizons,
        "metrics": aggregate_cases(cases, horizons),
    }
    atomic_write_json(summary, output_dir / "summary.json")
    atomic_write_json(cases, output_dir / "cases.json")
    write_case_csv(cases, output_dir / "cases.csv")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
