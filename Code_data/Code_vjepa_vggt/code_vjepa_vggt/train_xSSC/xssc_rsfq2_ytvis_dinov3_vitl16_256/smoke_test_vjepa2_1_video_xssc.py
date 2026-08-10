#!/usr/bin/env python3
"""Real-checkpoint forward/backward smoke test for V-JEPA2.1 video xSSC."""

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_path = ROOT / (
        "upstream/config-randsfq/"
        "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
    )
    cfg = Config.fromfile(config_path)
    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    model.freez(cfg.freez, verbose=False)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    device = torch.device(args.device)
    model = model.to(device)
    model.train(args.backward)
    torch.manual_seed(42)
    video = torch.randn(
        args.batch_size,
        cfg.raw_clip_frames,
        3,
        cfg.resolut0[0],
        cfg.resolut0[1],
        device=device,
    )

    context = torch.enable_grad() if args.backward else torch.inference_mode()
    with context, torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        outputs = model(batch={"video": video})
        loss = (outputs["recon"] - outputs["feature"].detach()).square().mean()
    if args.backward:
        loss.backward()

    expected = {
        "feature": [args.batch_size, cfg.xssc_steps, 1024, 16, 16],
        "slotz": [args.batch_size, cfg.xssc_steps, cfg.num_slots, cfg.slot_dim],
        "attenta": [args.batch_size, cfg.xssc_steps, cfg.num_slots, 16, 16],
        "recon": [args.batch_size, cfg.xssc_steps, 1024, 16, 16],
        "attentd": [args.batch_size, cfg.xssc_steps, cfg.num_slots, 16, 16],
    }
    actual = {key: list(value.shape) for key, value in outputs.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected output shapes: {actual} != {expected}")

    payload = {
        "config": str(config_path),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "raw_frames": cfg.raw_clip_frames,
        "xssc_steps": cfg.xssc_steps,
        "label_frame_indices_zero_based": cfg.label_frame_indices,
        "loss": float(loss.detach().float().item()),
        "outputs": {
            key: {
                "shape": list(value.shape),
                "finite": bool(torch.isfinite(value.detach().float()).all().item()),
            }
            for key, value in outputs.items()
        },
        "backward": bool(args.backward),
    }
    if args.backward:
        gradients = [
            parameter.grad.detach().float()
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        payload["gradient_tensors"] = len(gradients)
        payload["gradients_finite"] = bool(
            gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients)
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
