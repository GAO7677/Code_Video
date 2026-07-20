#!/usr/bin/env python3
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
        ),
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/checkpoints/xssc_slot512_ddp_smoke/"
            "slot512_smoke_nonbackbone.pth"
        ),
    )
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    return parser.parse_args()


def setup_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def distributed_mean(value, device):
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / dist.get_world_size()).item())


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    set_seed(args.seed + rank)

    from object_centric_bench.datum import DataLoader
    from object_centric_bench.learn import MetricWrap
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = args.config_file
    if not config_file.is_absolute():
        config_file = (ROOT / config_file).resolve()
    cfg = Config.fromfile(config_file)
    cfg.dataset_t.base_dir = args.data_dir
    # A bounded smoke run does not need the official full-dataset rebalancing scan.
    cfg.dataset_t.ts = None
    dataset = build_from_config(cfg.dataset_t)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_t,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=build_from_config(cfg.collate_fn_t),
        pin_memory=True,
        drop_last=True,
    )

    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    model = model.to(device).train()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.lr)
    amp_dtype = getattr(torch, args.amp_dtype)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn_t))
    acc_fn = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_t))
    callback = build_from_config(cfg.callback_t[0])

    torch.cuda.reset_peak_memory_stats(device)
    step_records = []
    iterator = iter(loader)
    for step in range(args.steps):
        batch = next(iterator)
        batch = {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        pack = {"batch": batch}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            output = ddp_model(**pack)
            pack["output"] = output
            callback.after_forward(**pack)
            losses = loss_fn(**pack)
        accuracies = acc_fn(**pack)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = sum(value[0][value[1]].mean() for value in losses.values())

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, cfg.gclip.max_norm
        )
        gradients_finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item())
            for p in trainable_parameters
        )
        if use_scaler:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_step = scaler.get_scale() >= scale_before
        else:
            optimizer.step()
            optimizer_step = True
        if not gradients_finite or not optimizer_step:
            raise RuntimeError(
                f"rank {rank}: non-finite gradients or skipped optimizer step"
            )

        step_records.append(
            {
                "step": step,
                "loss": distributed_mean(loss.detach().float().item(), device),
                "mbo": distributed_mean(
                    accuracies["mbo"][0][accuracies["mbo"][1]]
                    .mean()
                    .detach()
                    .float()
                    .item(),
                    device,
                ),
                "gradient_l2_norm_before_clip": distributed_mean(
                    gradient_norm.detach().float().item(), device
                ),
            }
        )

    torch.cuda.synchronize(device)
    peak_allocated = torch.tensor(
        torch.cuda.max_memory_allocated(device) / 1024**3, device=device
    )
    peak_reserved = torch.tensor(
        torch.cuda.max_memory_reserved(device) / 1024**3, device=device
    )
    dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)

    checkpoint_file = args.checkpoint_file.resolve()
    if rank == 0:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        nonbackbone_state = {
            key: value.detach().cpu()
            for key, value in ddp_model.module.state_dict().items()
            if not key.startswith("m.encode_backbone.")
        }
        torch.save(nonbackbone_state, checkpoint_file)
        summary_file = (
            args.summary_file.resolve()
            if args.summary_file is not None
            else checkpoint_file.with_suffix(".json")
        )
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_file),
            "checkpoint": str(checkpoint_file),
            "world_size": world_size,
            "batch_size_per_gpu": cfg.batch_size_t,
            "global_batch_size": cfg.batch_size_t * world_size,
            "steps": args.steps,
            "amp_dtype": args.amp_dtype,
            "dataset_samples_without_rebalancing": len(dataset),
            "video_shape_per_gpu": list(batch["video"].shape),
            "feature_shape_per_gpu": list(output["feature"].shape),
            "slot_shape_per_gpu": list(output["slotz"].shape),
            "recon_shape_per_gpu": list(output["recon"].shape),
            "gradient_clip_max_norm": cfg.gclip.max_norm,
            "optimizer_step": True,
            "peak_allocated_gib_max_rank": float(peak_allocated.item()),
            "peak_reserved_gib_max_rank": float(peak_reserved.item()),
            "checkpoint_tensors": len(nonbackbone_state),
            "steps_detail": step_records,
        }
        summary_file.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
