#!/usr/bin/env python3
"""Run one realistic train microbatch and reject unsafe GPU-memory peaks."""

from argparse import ArgumentParser
import gc
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))
sys.path.insert(0, "/home/gaoya/Code_Video/vjepa2-main")

from object_centric_bench.model import ModelWrap  # noqa: E402
from object_centric_bench.util import Config, build_from_config  # noqa: E402
from train_ddp_ytvis_hq import load_matching_checkpoint  # noqa: E402


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cfg-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-reserved-gib", type=float, default=44.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    cfg = Config.fromfile(args.cfg_file.resolve())
    cfg.dataset_t.base_dir = args.data_dir.resolve()
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.use_deterministic_algorithms(
        cfg.use_deterministic_algorithms,
        warn_only=bool(getattr(cfg, "deterministic_warn_only", True)),
    )
    if bool(getattr(cfg, "deterministic_sdp_math", False)):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    dataset = build_from_config(cfg.dataset_t)
    # Use distinct records to retain realistic bbox/presence patterns.
    samples = [dataset[index] for index in range(args.batch_size)]
    batch = build_from_config(cfg.collate_fn_t)(samples)
    batch = {
        key: value.to(device, non_blocking=False)
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }

    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    load_report = load_matching_checkpoint(
        model,
        args.checkpoint.resolve(),
        cfg.transfer_load_exclude,
        allowed_missing_patterns=cfg.transfer_allowed_missing,
        expected_source_variant=cfg.transfer_expected_source_variant,
        expected_source_step=cfg.transfer_expected_source_step,
        prefix_map=getattr(cfg, "transfer_prefix_map", ()),
        partial_row_patterns=getattr(cfg, "transfer_partial_row_patterns", ()),
    )
    model.freez(cfg.freez, verbose=False)
    model = model.to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(cfg.lr))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=getattr(torch, cfg.amp_dtype)):
        output = model(batch=batch)
        loss = torch.nn.functional.mse_loss(
            output["recon"], output["feature"].detach()
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, float(cfg.gradient_clip_norm))
    optimizer.step()  # Materialize Adam moments as in the real first step.
    torch.cuda.synchronize(device)
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    result = {
        "status": "pass" if peak_reserved_gib <= args.max_reserved_gib else "unsafe",
        "batch_size": args.batch_size,
        "loss": float(loss.detach().float().item()),
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
        "max_reserved_gib": args.max_reserved_gib,
        "matched_key_count": load_report["matched_key_count"],
    }
    print(json.dumps(result, sort_keys=True), flush=True)

    del optimizer, model, batch, samples, output, loss
    gc.collect()
    torch.cuda.empty_cache()
    if peak_reserved_gib > args.max_reserved_gib:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
