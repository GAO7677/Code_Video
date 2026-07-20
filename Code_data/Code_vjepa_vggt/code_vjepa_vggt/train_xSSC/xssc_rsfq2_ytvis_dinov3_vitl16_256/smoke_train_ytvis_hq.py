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
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256.py"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    from object_centric_bench.datum import DataLoader
    from object_centric_bench.learn import MetricWrap
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    torch.manual_seed(args.seed)
    config_file = args.config_file
    if not config_file.is_absolute():
        config_file = ROOT / config_file
    config_file = config_file.resolve()
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

    callback = build_from_config(cfg.callback_t[0])
    callback.before_step[0](batch=batch)
    video = batch["video"]
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.lr)
    amp_dtype = getattr(torch, args.amp_dtype)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn_t))
    acc_fn = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_t))
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pack = {"batch": batch}
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        outputs = model(**pack)
        pack["output"] = outputs
        callback.after_forward(**pack)
        losses = loss_fn(**pack)
    accuracies = acc_fn(**pack)
    with torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
    ):
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
        parameter.grad is None
        or bool(torch.isfinite(parameter.grad.detach()).all().item())
        for parameter in trainable_parameters
    )
    if use_scaler:
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        optimizer_step = scale_after >= scale_before
    else:
        optimizer.step()
        optimizer_step = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    cuda_memory = None
    if device.type == "cuda":
        gib = 1024**3
        cuda_memory = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
            "total_gib": torch.cuda.get_device_properties(device).total_memory / gib,
        }

    payload = {
        "config": str(config_file),
        "dataset": str(args.data_dir / cfg.dataset_t.data_file),
        "dataset_samples": len(dataset),
        "temporal_rebalancing_scan": False,
        "num_slots": cfg.max_num,
        "slot_dim": cfg.emb_dim,
        "backbone_feature_dim": cfg.vfm_dim,
        "batch_size": args.batch_size,
        "video_shape": list(video.shape),
        "segment_shape": list(batch["segment"].shape),
        "video_dtype": str(video.dtype),
        "amp_dtype": args.amp_dtype,
        "loss": float(loss.detach().float().item()),
        "gradient_l2_norm_before_clip": float(gradient_norm.detach().float().item()),
        "gradient_clip_max_norm": cfg.gclip.max_norm,
        "gradients_finite": gradients_finite,
        "optimizer_step": optimizer_step,
        "train_metrics": {
            key: float(value[0][value[1]].mean().detach().float().item())
            for key, value in accuracies.items()
        },
        "cuda_memory": cuda_memory,
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
