#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))

from data.Stage1Indexed import Stage1Indexed  # noqa: E402
from feature_space_stage1.backbones import FrozenVJEPA2Extractor  # noqa: E402
from feature_space_stage1.model import FeatureSlotDecomposer, feature_space_losses  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-root",
        default="/data/gaoya/AAA_test_video/0623_savi/indices_pybullet1200_kubric9600_full_pool",
    )
    parser.add_argument(
        "--checkpoint",
        default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8,12,16,24,32,40,48,64")
    parser.add_argument("--slot-dim", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    device = torch.device("cuda:0")
    dataset = Stage1Indexed(
        index_root=args.index_root,
        dataset_mode="mixed",
        split="train",
        num_frames=10,
        preprocess_mode="vjepa",
        random_start=False,
        max_samples=max(batch_sizes),
    )
    videos = torch.stack([dataset[index][0] for index in range(max(batch_sizes))])
    extractor = FrozenVJEPA2Extractor(Path(args.checkpoint), device, num_frames=10)
    core = FeatureSlotDecomposer(
        feature_dim=extractor.feature_dim,
        num_slots=8,
        slot_dim=args.slot_dim,
    ).to(device)
    results = []
    for batch_size in batch_sizes:
        core.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch = videos[:batch_size]
            features = extractor(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = core(features)
            losses = feature_space_losses(output, features, "vjepa")
            losses["total"].mean().backward()
            torch.cuda.synchronize(device)
            result = {
                "batch_size": batch_size,
                "status": "passed",
                "loss": float(losses["total"].mean().detach()),
                "feature_shape": list(features.shape),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            del batch, features, output, losses
        except torch.OutOfMemoryError as error:
            result = {
                "batch_size": batch_size,
                "status": "oom",
                "error": str(error),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            core.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            break
        results.append(result)
        print(json.dumps(result), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
