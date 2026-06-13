from __future__ import annotations

import argparse
import os

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_ball_block_ti2v_vjepa_vggt.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    trainer = ContextVideoTrainer(cfg, device=_resolve_launch_device())
    trainer.train()


if __name__ == "__main__":
    main()
