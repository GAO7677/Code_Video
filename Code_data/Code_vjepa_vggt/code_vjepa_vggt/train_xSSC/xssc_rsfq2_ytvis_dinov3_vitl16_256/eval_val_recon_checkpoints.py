#!/usr/bin/env python3
"""Evaluate reconstruction MSE for multiple DINOv3 xSSC checkpoints."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = args.config_file.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoints = sorted(checkpoint_dir.glob("step-*.pth"))
    if not checkpoints:
        raise RuntimeError(f"No step checkpoints found in {checkpoint_dir}")

    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = args.data_dir.resolve()
    dataset = build_from_config(cfg.dataset_v)
    num_cases = len(dataset) if args.max_cases <= 0 else min(args.max_cases, len(dataset))
    collate_fn = build_from_config(cfg.collate_fn_v)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = ModelWrap(
        build_from_config(cfg.model), cfg.model_imap, cfg.model_omap
    ).to(device).eval()
    model.freez(cfg.freez, verbose=False)
    amp_dtype = getattr(torch, args.amp_dtype)

    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(config_file),
        "dataset": str(args.data_dir.resolve() / cfg.dataset_v.data_file),
        "dataset_cases": len(dataset),
        "evaluated_cases_per_checkpoint": num_cases,
        "amp_dtype": args.amp_dtype,
        "metric": "per-video mean of (recon - detached_feature)^2.mean()",
        "checkpoints": [],
    }

    for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
        print(
            f"[checkpoint {checkpoint_index}/{len(checkpoints)}] loading {checkpoint.name}",
            flush=True,
        )
        state_dict = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        started = time.time()
        losses = []
        for index in range(num_cases):
            batch = collate_fn([dataset[index]])
            video = batch["video"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(batch={"video": video})
                loss = (output["recon"] - output["feature"].detach()).square().mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    f"Non-finite reconstruction loss for {checkpoint.name}, case {index}"
                )
            losses.append(float(loss.float().item()))
            if (index + 1) % 40 == 0 or index + 1 == num_cases:
                print(
                    f"[checkpoint {checkpoint_index}/{len(checkpoints)}] "
                    f"{checkpoint.name}: {index + 1}/{num_cases}",
                    flush=True,
                )

        loss_tensor = torch.tensor(losses, dtype=torch.float64)
        summary = {
            "step": int(checkpoint.stem.split("-")[-1]),
            "checkpoint": str(checkpoint),
            "mean_reconstruction_mse": float(loss_tensor.mean().item()),
            "std_reconstruction_mse": float(loss_tensor.std(unbiased=False).item()),
            "min_reconstruction_mse": float(loss_tensor.min().item()),
            "max_reconstruction_mse": float(loss_tensor.max().item()),
            "seconds": time.time() - started,
            "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
        payload["checkpoints"].append(summary)
        output_file.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
