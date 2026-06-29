from __future__ import annotations

import argparse
import os

from code_vjepa_vggt.utils.config import load_yaml_config

from .runtime_stage1c_joint import Stage1CJointTrainer


def _resolve_launch_device() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage1C joint finetune branch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--init-from", default=None, help="load prior-stage trainable weights (strict=False) without restoring step")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    trainer = Stage1CJointTrainer(cfg, device=_resolve_launch_device())
    trainer.train(resume_checkpoint=args.resume_checkpoint, init_from=args.init_from)


if __name__ == "__main__":
    main()
