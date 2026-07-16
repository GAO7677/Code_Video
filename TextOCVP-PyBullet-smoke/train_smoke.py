#!/usr/bin/env python3
"""Prepare and launch an official TextOCVP Stage 1 smoke experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-limit", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--sampling-first-frame", type=int, default=0)
    parser.add_argument("--sampling-last-frame", type=int, default=49)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--slot-dim", type=int, default=128)
    return parser.parse_args()


def prepare_experiment(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
    previous_cwd = Path.cwd()
    try:
        os.chdir(TEXTOCVP_ROOT)
        from lib.config import Config

        config = Config(exp_path=str(args.output_dir))
        config.create_exp_config_file(
            model_name="SAVi",
            dataset_name="PyBullet_Raw",
        )
        params = config.load_exp_config_file()
    finally:
        os.chdir(previous_cwd)

    params["dataset"].update(
        {
            "root": str(args.dataset_root.expanduser().resolve()),
            "num_frames": args.num_frames,
            "img_size": [args.height, args.width],
            "frame_stride": args.frame_stride,
            "sampling_frame_range": [
                args.sampling_first_frame,
                args.sampling_last_frame,
            ],
            "max_samples": args.dataset_limit,
        }
    )
    model_params = params["model"]["model_params"]
    model_params["num_slots"] = args.num_slots
    model_params["slot_dim"] = args.slot_dim
    model_params["encoder"]["encoder_params"]["resolution"] = [args.height, args.width]
    model_params["decoder"]["decoder_params"]["resolution"] = [args.height, args.width]
    params["training"].update(
        {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "save_frequency": 1,
        }
    )
    config_path = args.output_dir / "experiment_params.json"
    config_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return config_path


def main():
    args = parse_args()
    config_path = prepare_experiment(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONNOUSERSITE", "1")
    command = [
        sys.executable,
        str(TEXTOCVP_ROOT / "src" / "02_train_savi.py"),
        "-d",
        str(args.output_dir),
    ]
    print(f"[stage1] config={config_path}", flush=True)
    print(f"[stage1] command={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=TEXTOCVP_ROOT, env=env, check=True)
    final_checkpoint = args.output_dir / "models" / "checkpoint_epoch_final.pth"
    if not final_checkpoint.is_file():
        raise RuntimeError(
            "Official TextOCVP trainer exited without producing the expected "
            f"Stage 1 checkpoint: {final_checkpoint}"
        )
    print(f"[stage1] checkpoint={final_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
