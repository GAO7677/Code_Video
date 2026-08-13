#!/usr/bin/env python3
"""Executable future-perturbation causality gate for real checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional


STAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "upstream"))
sys.path.insert(0, "/home/gaoya/Code_Video/vjepa2-main")

from stage1_causal_state_probe import NUM_OBJECTS, NUM_SLOTS  # noqa: E402
from stage1_causal_state_probe.cache_causal_slots import (  # noqa: E402
    build_dataset,
    load_checkpoint,
)
from stage1_causal_state_probe.io_utils import atomic_write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--substitute-index", type=int, default=1)
    parser.add_argument("--cut-states", type=int, nargs="+", default=[0, 3, 7, 10])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def initial_bbox(sample):
    num_objects = min(int(sample["bbox"].shape[1]), NUM_OBJECTS)
    value = torch.zeros(NUM_SLOTS, 4, dtype=torch.float32)
    value[:num_objects] = sample["bbox"][0, :num_objects]
    return value


def difference(reference, candidate):
    delta = (candidate.float() - reference.float()).abs()
    cosine = 1 - functional.cosine_similarity(
        candidate.float().flatten(1), reference.float().flatten(1), dim=-1
    )
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "max_cosine_distance": float(cosine.max()),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device == "cuda:4":
        raise ValueError("GPU 4 is prohibited by workspace policy")
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(args.config_file.resolve())
    if cfg.model.encode_backbone.temporal_mode != "prefix_causal":
        raise RuntimeError("Future perturbation requires prefix_causal mode")
    dataset = build_dataset(cfg, args.split, args.data_root.resolve())
    sample = dataset[args.case_index]
    substitute = dataset[args.substitute_index]
    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    load_checkpoint(model, args.checkpoint.resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device).eval()
    dtype = getattr(torch, args.amp_dtype)
    video = sample["video"].to(device)
    condition = initial_bbox(sample).to(device)

    def extract(value):
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            feature, slots, _ = model.m.extract_slot_trajectory(
                value[None], initial_condit=condition[None]
            )
        return feature[0].cpu(), slots[0].cpu()

    reference_feature, reference_slots = extract(video)
    records = []
    generator = torch.Generator(device=device).manual_seed(20260813)
    for cut_state in args.cut_states:
        if not 0 <= cut_state < reference_slots.shape[0] - 1:
            raise ValueError(f"Invalid cut state: {cut_state}")
        future_start = 2 * (cut_state + 1)
        candidates = {}
        zeroed = video.clone()
        zeroed[future_start:] = 0
        candidates["zero"] = zeroed
        shuffled = video.clone()
        order = torch.randperm(
            len(video) - future_start, generator=generator, device=device
        )
        shuffled[future_start:] = video[future_start:].index_select(0, order)
        candidates["shuffle"] = shuffled
        replaced = video.clone()
        replaced[future_start:] = substitute["video"][future_start:].to(device)
        candidates["substitute"] = replaced

        for perturbation, candidate in candidates.items():
            feature, slots = extract(candidate)
            feature_diff = difference(
                reference_feature[: cut_state + 1], feature[: cut_state + 1]
            )
            slot_diff = difference(
                reference_slots[: cut_state + 1], slots[: cut_state + 1]
            )
            passed = max(feature_diff["max_abs"], slot_diff["max_abs"]) <= args.atol
            records.append(
                {
                    "cut_state": cut_state,
                    "future_raw_start": future_start,
                    "perturbation": perturbation,
                    "feature": feature_diff,
                    "slot": slot_diff,
                    "passed": passed,
                }
            )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config_file.resolve()),
        "split": args.split,
        "case_index": args.case_index,
        "substitute_index": args.substitute_index,
        "atol": args.atol,
        "passed": all(record["passed"] for record in records),
        "records": records,
    }
    atomic_write_json(payload, args.output.resolve())
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

