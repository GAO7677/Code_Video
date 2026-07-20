#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    from object_centric_bench.datum import DataLoader
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    torch.manual_seed(args.seed)
    config_file = (
        ROOT
        / "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256.py"
    )
    cfg = Config.fromfile(config_file)
    cfg.dataset_t.base_dir = args.data_dir
    # A one-batch smoke does not need the full-dataset scan used for temporal rebalancing.
    cfg.dataset_t.ts = None
    dataset = build_from_config(cfg.dataset_t)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=build_from_config(cfg.collate_fn_t),
    )
    batch = next(iter(loader))

    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)
    device = torch.device(args.device)
    model = model.to(device).train()

    video = batch["video"].to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.lr)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        outputs = model(batch={"video": video})
        loss = (outputs["recon"] - outputs["feature"].detach()).square().mean()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_parameters, cfg.gclip.max_norm
    )
    gradients_finite = all(
        parameter.grad is None
        or bool(torch.isfinite(parameter.grad.detach()).all().item())
        for parameter in trainable_parameters
    )
    optimizer.step()

    payload = {
        "config": str(config_file),
        "dataset": str(args.data_dir / cfg.dataset_t.data_file),
        "dataset_samples": len(dataset),
        "temporal_rebalancing_scan": False,
        "batch_size": args.batch_size,
        "video_shape": list(video.shape),
        "segment_shape": list(batch["segment"].shape),
        "video_dtype": str(video.dtype),
        "loss": float(loss.detach().float().item()),
        "gradient_l2_norm_before_clip": float(gradient_norm.detach().float().item()),
        "gradient_clip_max_norm": cfg.gclip.max_norm,
        "gradients_finite": gradients_finite,
        "optimizer_step": True,
        "outputs": {
            key: {
                "shape": list(value.shape),
                "finite": bool(torch.isfinite(value.detach().float()).all().item()),
            }
            for key, value in outputs.items()
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
