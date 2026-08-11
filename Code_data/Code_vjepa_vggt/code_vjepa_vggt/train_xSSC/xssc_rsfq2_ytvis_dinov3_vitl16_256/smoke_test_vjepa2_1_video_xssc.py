#!/usr/bin/env python3
"""Real-checkpoint forward/backward smoke test for V-JEPA2.1 video xSSC."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/"
            "rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config

    config_path = args.cfg_file
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = Config.fromfile(config_path)
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
    height = cfg.resolut0[0] if args.height is None else args.height
    width = cfg.resolut0[1] if args.width is None else args.width
    video = torch.randn(
        args.batch_size,
        cfg.raw_clip_frames,
        3,
        height,
        width,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    grid_height = height // 16
    grid_width = width // 16
    expected = {
        "feature": [args.batch_size, cfg.xssc_steps, 1024, grid_height, grid_width],
        "slotz": [args.batch_size, cfg.xssc_steps, cfg.num_slots, cfg.slot_dim],
        "attenta": [
            args.batch_size,
            cfg.xssc_steps,
            cfg.num_slots,
            grid_height,
            grid_width,
        ],
        "recon": [
            args.batch_size,
            cfg.xssc_steps,
            1024,
            grid_height,
            grid_width,
        ],
        "attentd": [
            args.batch_size,
            cfg.xssc_steps,
            cfg.num_slots,
            grid_height,
            grid_width,
        ],
    }
    actual = {key: list(value.shape) for key, value in outputs.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected output shapes: {actual} != {expected}")

    payload = {
        "config": str(config_path),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "raw_frames": cfg.raw_clip_frames,
        "input_height": height,
        "input_width": width,
        "xssc_steps": cfg.xssc_steps,
        "label_frame_indices_zero_based": cfg.label_frame_indices,
        "temporal_mode": cfg.temporal_mode,
        "loss": float(loss.detach().float().item()),
        "outputs": {
            key: {
                "shape": list(value.shape),
                "finite": bool(torch.isfinite(value.detach().float()).all().item()),
            }
            for key, value in outputs.items()
        },
        "backward": bool(args.backward),
        "elapsed_seconds": elapsed_seconds,
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / 1024**3
            if device.type == "cuda"
            else None
        ),
        "deterministic_sdp_math": bool(
            getattr(cfg, "deterministic_sdp_math", False)
        ),
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
