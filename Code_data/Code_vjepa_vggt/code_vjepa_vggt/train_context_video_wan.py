from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.infer_context_video_wan import _infer_object_pooler_latent_dim, _load_trainable_state
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
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="optional step_*.pt checkpoint containing trainable weights to resume from",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.resume_checkpoint is not None:
        checkpoint_state = _load_trainable_state(Path(args.resume_checkpoint))
        object_pooler_latent_dim = _infer_object_pooler_latent_dim(
            checkpoint_state,
            int(cfg["model"].get("object_pooler_latent_dim", 16)),
        )
        cfg["model"]["object_pooler_latent_dim"] = int(object_pooler_latent_dim)
    trainer = ContextVideoTrainer(cfg, device=_resolve_launch_device())
    trainer.train(resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
