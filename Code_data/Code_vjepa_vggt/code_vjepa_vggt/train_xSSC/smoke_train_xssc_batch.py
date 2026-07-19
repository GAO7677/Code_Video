"""Run one real batched xSSC/Wan training step for memory validation."""
from __future__ import annotations

import argparse
import time

import torch

from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as train_xssc


def _build_parser() -> argparse.ArgumentParser:
    parser = train_xssc.build_parser()
    group = parser.add_argument_group("xssc_smoke")
    group.add_argument("--smoke_batch_size", type=int, default=2)
    return parser


def main() -> None:
    parser = _build_parser()
    args = train_xssc.tvn.prepare_args(parser.parse_args())
    if args.smoke_batch_size < 1:
        parser.error("--smoke_batch_size must be positive")
    if int(args.fixed_num_context_frames) != train_xssc.XSSC_NUM_CONTEXT_FRAMES:
        parser.error("xSSC smoke requires --fixed_num_context_frames 8")
    args.no_context_ratio = 0.0
    accelerator = train_xssc.tvn.build_accelerator(args)
    if accelerator.num_processes != 1:
        raise RuntimeError("Run this memory smoke on one GPU without accelerate launch")

    dataset = train_xssc.tvn.build_dataset(args)
    model = train_xssc.build_model(args, accelerator)
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

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    baseline_allocated = torch.cuda.memory_allocated(accelerator.device)
    started = time.perf_counter()
    loss = model._forward_sample_batch(
        [dataset[index] for index in range(args.smoke_batch_size)]
    )
    metrics = model.last_train_metrics
    accelerator.backward(loss)
    accelerator.clip_grad_norm_(model.trainable_modules(), args.max_grad_norm)
    optimizer.step()
    torch.cuda.synchronize(accelerator.device)
    elapsed = time.perf_counter() - started

    gib = 1024**3
    print("xSSC batch smoke succeeded")
    print(f"batch_size_per_gpu={args.smoke_batch_size}")
    print(f"loss={float(loss.detach().item()):.6f}")
    print(f"object_tokens={int(metrics['train/xssc_token_count'])}")
    print(f"baseline_allocated_gib={baseline_allocated / gib:.3f}")
    print(f"peak_allocated_gib={torch.cuda.max_memory_allocated(accelerator.device) / gib:.3f}")
    print(f"peak_reserved_gib={torch.cuda.max_memory_reserved(accelerator.device) / gib:.3f}")
    print(f"step_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
