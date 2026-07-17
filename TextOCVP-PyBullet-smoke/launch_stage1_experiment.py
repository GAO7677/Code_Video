#!/usr/bin/env python3
"""Create and launch one fixed-index SAVi comparison experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
TRAINER = Path(__file__).resolve().parent / "train_stage1_stepval.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-mode", choices=("pybullet", "kubric", "mixed"), required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--micro-global-batch-size", type=int, required=True)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--validation-frequency-steps", type=int, default=1000)
    parser.add_argument("--dataset-max-samples", type=int, default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--mask-loss-weight", type=float, default=0.0)
    parser.add_argument("--mask-loss-warmup-steps", type=int, default=500)
    parser.add_argument("--mask-max-instances", type=int, default=6)
    parser.add_argument("--mask-union-weight", type=float, default=0.20)
    parser.add_argument("--mask-instance-weight", type=float, default=0.10)
    parser.add_argument("--mask-static-weight", type=float, default=0.02)
    parser.add_argument("--mask-background-weight", type=float, default=0.01)
    parser.add_argument("--mask-unused-weight", type=float, default=0.01)
    parser.add_argument("--mask-focal-bce-weight", type=float, default=0.25)
    parser.add_argument("--wandb-project", default="textocvp_savi_stage1")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def prepare_config(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
    try:
        os.chdir(TEXTOCVP_ROOT)
        from lib.config import Config

        config = Config(exp_path=str(args.output_dir))
        config.create_exp_config_file(model_name="SAVi", dataset_name="Stage1_Indexed")
        params = config.load_exp_config_file()
    finally:
        os.chdir(previous_cwd)

    params["dataset"].update(
        {
            "index_root": str(args.index_root.resolve()),
            "dataset_mode": args.dataset_mode,
            "num_frames": 10,
            "img_size": [216, 384],
            "frame_stride": 1,
            "random_start": True,
            "max_samples": args.dataset_max_samples,
            "load_masks": args.mask_loss_weight > 0,
            "max_mask_instances": args.mask_max_instances,
            "mask_temporal_stride": 1,
            "mask_spatial_stride": 1,
        }
    )
    model = params["model"]["model_params"]
    model["num_slots"] = 8
    model["slot_dim"] = 256
    model["encoder"]["encoder_params"]["resolution"] = [216, 384]
    model["decoder"]["decoder_params"]["resolution"] = [216, 384]
    train_index = args.index_root.resolve() / args.dataset_mode / "train.jsonl"
    if not train_index.is_file():
        raise FileNotFoundError(f"Training index not found: {train_index}")
    dataset_size = sum(1 for line in train_index.read_text(encoding="utf-8").splitlines() if line)
    if args.dataset_max_samples is not None:
        dataset_size = min(dataset_size, int(args.dataset_max_samples))
    steps_per_epoch = (
        dataset_size + args.effective_batch_size - 1
    ) // args.effective_batch_size
    total_steps = steps_per_epoch * args.epochs
    # Keep warmup at 10% of optimizer steps when effective batch size changes.
    warmup_steps = max(100, total_steps // 10)
    params["training"].update(
        {
            "num_epochs": args.epochs,
            "batch_size": args.micro_global_batch_size,
            "effective_batch_size": args.effective_batch_size,
            "validation_frequency_steps": args.validation_frequency_steps,
            "scheduler_steps": max(1, total_steps - warmup_steps),
            "save_frequency": args.epochs + 1,
            "warmup_steps": warmup_steps,
            "dataset_size": dataset_size,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "overfit_patience_validations": 3,
            "overfit_relative_degradation": 0.02,
            "mask_loss_weight": args.mask_loss_weight,
            "mask_loss_warmup_steps": args.mask_loss_warmup_steps,
            "mask_union_weight": args.mask_union_weight,
            "mask_instance_weight": args.mask_instance_weight,
            "mask_static_weight": args.mask_static_weight,
            "mask_background_weight": args.mask_background_weight,
            "mask_unused_weight": args.mask_unused_weight,
            "mask_focal_bce_weight": args.mask_focal_bce_weight,
        }
    )
    params["wandb"] = {
        "enabled": not args.disable_wandb,
        "project": args.wandb_project,
        "group": args.wandb_group,
        "run_name": f"{args.output_dir.parent.name}-{args.dataset_mode}",
    }
    config_path = args.output_dir / "experiment_params.json"
    config_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return config_path


def main():
    args = parse_args()
    config_path = prepare_config(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("WANDB_MODE", "online")
    env.setdefault("WANDB_SILENT", "false")
    command = [
        sys.executable,
        str(TRAINER),
        "--exp-directory",
        str(args.output_dir),
    ]
    if args.max_optimizer_steps is not None:
        command.extend(["--max-optimizer-steps", str(args.max_optimizer_steps)])
    print(f"config={config_path}", flush=True)
    print(f"command={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=TEXTOCVP_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
