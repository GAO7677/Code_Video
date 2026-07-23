#!/usr/bin/env python3
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


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
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/xssc_slot512_ddp_smoke/"
            "ytvis_hq_val_all_loss.json"
        ),
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    return parser.parse_args()


def setup_distributed():
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


@torch.inference_mode()
def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_file = args.config_file
    if not config_file.is_absolute():
        config_file = (ROOT / config_file).resolve()
    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = args.data_dir
    dataset = build_from_config(cfg.dataset_v)
    num_cases = len(dataset) if args.max_cases <= 0 else min(args.max_cases, len(dataset))
    indices = list(range(num_cases))
    local_indices = indices[rank::world_size]
    collate_fn = build_from_config(cfg.collate_fn_v)

    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    checkpoint_file = args.checkpoint_file.resolve()
    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    nonbackbone_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("m.encode_backbone.")
    ]
    if nonbackbone_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint mismatch: "
            f"missing={nonbackbone_missing}, unexpected={incompatible.unexpected_keys}"
        )
    model.freez(cfg.freez, verbose=False)
    model = model.to(device).eval()
    amp_dtype = getattr(torch, args.amp_dtype)
    torch.cuda.reset_peak_memory_stats(device)

    local_records = []
    for position, index in enumerate(local_indices):
        batch = collate_fn([dataset[index]])
        video = batch["video"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            output = model(batch={"video": video})
            loss = (
                output["recon"] - output["feature"].detach()
            ).square().mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"rank {rank}: non-finite loss for val index {index}")
        local_records.append(
            {
                "index": index,
                "frames": int(video.shape[1]),
                "loss": float(loss.detach().float().item()),
                "feature_shape": list(output["feature"].shape),
                "slot_shape": list(output["slotz"].shape),
                "recon_shape": list(output["recon"].shape),
            }
        )
        if rank == 0 and (position + 1) % 10 == 0:
            print(
                f"[val] rank0 {position + 1}/{len(local_indices)} local cases",
                flush=True,
            )

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    peak_reserved = torch.tensor(
        torch.cuda.max_memory_reserved(device) / 1024**3, device=device
    )
    dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)

    if rank == 0:
        records = sorted(
            [record for rank_records in gathered for record in rank_records],
            key=lambda record: record["index"],
        )
        losses = [record["loss"] for record in records]
        output_file = args.output_file.resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_file),
            "checkpoint": str(checkpoint_file),
            "dataset": str(args.data_dir / cfg.dataset_v.data_file),
            "dataset_cases": len(dataset),
            "evaluated_cases": len(records),
            "world_size": world_size,
            "amp_dtype": args.amp_dtype,
            "mean_reconstruction_mse": sum(losses) / len(losses),
            "min_reconstruction_mse": min(losses),
            "max_reconstruction_mse": max(losses),
            "mean_frames": sum(record["frames"] for record in records) / len(records),
            "peak_reserved_gib_max_rank": float(peak_reserved.item()),
            "cases": records,
        }
        output_file.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({key: value for key, value in payload.items() if key != "cases"}, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
