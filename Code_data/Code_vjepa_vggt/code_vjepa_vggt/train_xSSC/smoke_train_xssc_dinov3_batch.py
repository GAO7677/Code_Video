#!/usr/bin/env python3
"""Run one real DINOv3-xSSC/Wan training step for per-GPU batch probing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = REPO_ROOT.parent
sys.path.insert(0, str(PACKAGE_PARENT))
sys.path.insert(0, str(TRAIN_XSSC_ROOT))

from code_vjepa_vggt.train_xSSC import train_xssc_context_slots_dinov3 as train_xssc  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = train_xssc.build_parser()
    group = parser.add_argument_group("dinov3_xssc_smoke")
    group.add_argument("--smoke_batch_size", type=int, default=1)
    group.add_argument("--smoke_start_index", type=int, default=0)
    return parser


def _sample_indices(dataset, start: int, count: int) -> list[int]:
    length = len(dataset)
    if length <= 0:
        raise RuntimeError("cannot smoke-test an empty dataset")
    return [(int(start) + offset) % length for offset in range(int(count))]


def main() -> None:
    parser = _build_parser()
    args = train_xssc.tvn.prepare_args(parser.parse_args())
    if args.smoke_batch_size < 1:
        parser.error("--smoke_batch_size must be positive")
    if int(args.fixed_num_context_frames) != train_xssc.base.XSSC_NUM_CONTEXT_FRAMES:
        parser.error("DINOv3 xSSC smoke requires --fixed_num_context_frames 8")
    args.no_context_ratio = 0.0

    accelerator = train_xssc.tvn.build_accelerator(args)
    if accelerator.num_processes != 1:
        raise RuntimeError("Run this memory smoke on one GPU without accelerate launch")

    try:
        dataset = train_xssc.base.build_dataset(args)
        model = train_xssc.build_model(args, accelerator)
        if getattr(args, "xssc_filter_empty_amg", False):
            model.set_empty_amg_resample_dataset(dataset)
        model.to(accelerator.device)
        model.train()
        optimizer = train_xssc.tvn._build_optimizer(
            args.optimizer_type,
            model.trainable_modules(),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=train_xssc.tvn.DEFAULT_OPTIMIZER_BETAS,
            eps=train_xssc.tvn.DEFAULT_OPTIMIZER_EPS,
        )

        indices = _sample_indices(dataset, args.smoke_start_index, args.smoke_batch_size)
        samples = [dataset[index] for index in indices]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        baseline_allocated = torch.cuda.memory_allocated(accelerator.device)

        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = model(samples)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(
            model.trainable_modules(),
            args.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(accelerator.device)
        elapsed = time.perf_counter() - started

        metrics = dict(model.last_train_metrics)
        gib = 1024**3
        payload = {
            "status": "ok",
            "batch_size_per_gpu": int(args.smoke_batch_size),
            "sample_indices": indices,
            "loss": float(loss.detach().item()),
            "grad_l2_norm_before_clip": float(grad_norm),
            "baseline_allocated_gib": baseline_allocated / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(accelerator.device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(accelerator.device) / gib,
            "step_seconds": elapsed,
            "xssc_checkpoint": str(args.xssc_checkpoint),
            "xssc_box_source": str(args.xssc_box_source),
            "metrics": metrics,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        payload = {
            "status": "oom",
            "batch_size_per_gpu": int(args.smoke_batch_size),
            "error": str(error).splitlines()[0],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
