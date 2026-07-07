from __future__ import annotations

import argparse
import os

from code_vjepa_vggt.utils.config import load_yaml_config

from .runtime_stage1a_full_token_structure_ablation import (
    StructureAblationFullTokenTeacherTrainer,
)


def _resolve_launch_device() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage1A full-token teacher branch with structure ablations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--init-from", default=None, help="load prior-stage trainable weights (strict=False) without restoring step")
    parser.add_argument(
        "--structure_ablation_type",
        required=True,
        choices=("wo_jepa", "wo_vggt"),
        help="Full-pipeline Stage1A structure ablation type.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    if args.output_dir:
        cfg.setdefault("experiment", {})["output_dir"] = str(args.output_dir)
    if args.experiment_name:
        cfg.setdefault("experiment", {})["name"] = str(args.experiment_name)
    if args.wandb_project:
        cfg.setdefault("logging", {})["wandb_project"] = str(args.wandb_project)
    if args.wandb_run_name:
        cfg.setdefault("logging", {})["wandb_run_name"] = str(args.wandb_run_name)
    trainer = StructureAblationFullTokenTeacherTrainer(
        cfg,
        structure_ablation_type=args.structure_ablation_type,
        device=_resolve_launch_device(),
    )
    trainer.train(resume_checkpoint=args.resume_checkpoint, init_from=args.init_from)


if __name__ == "__main__":
    main()
