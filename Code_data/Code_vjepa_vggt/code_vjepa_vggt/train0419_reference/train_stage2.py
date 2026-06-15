from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from code_vjepa_vggt.infer_context_video_wan import _infer_object_pooler_latent_dim, _load_trainable_state
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _apply_stage1_checkpoint(cfg: dict, checkpoint_path: Path) -> None:
    checkpoint_state = _load_trainable_state(checkpoint_path)
    object_pooler_latent_dim = _infer_object_pooler_latent_dim(
        checkpoint_state,
        int(cfg["model"].get("object_pooler_latent_dim", 16)),
    )
    cfg["model"]["object_pooler_latent_dim"] = int(object_pooler_latent_dim)
    cfg["model"]["init_wan_lora_from_checkpoint"] = str(checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_stage2_adapters_gpu67.yaml",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        required=True,
        help="stage1 Wan LoRA checkpoint, typically checkpoint.safetensors or a step_*.pt file",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="optional stage2 step_*.pt checkpoint to resume from",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    _apply_stage1_checkpoint(cfg, Path(args.stage1_checkpoint))
    trainer = ContextVideoTrainer(cfg, device=_resolve_launch_device())
    trainer.train(resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
