#!/usr/bin/env python3
"""Validation-loss evaluator for the teacher-student pipeline.

Builds the trainer for a given stage ONCE, then loads each checkpoint (strict=False),
runs the trainer.forward over the `val` split under no_grad, and logs mean loss
components to wandb (one point per checkpoint step).

Usage:
  python3 ts_eval.py --stage 1a --config <cfg.yaml> \
      --ckpt <file.pt | dir-of-step_*.pt> --max-batches 50 --device cuda:5 [--no-wandb]
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
from pathlib import Path

# stage -> (module path, class name)
STAGE_MAP = {
    "1a": ("code_vjepa_vggt.object_token_teacher_student.runtime_stage1a_full_token", "FullTokenTeacherTrainer"),
    "1b": ("code_vjepa_vggt.object_token_teacher_student.runtime_stage1", "OracleInjectionTrainer"),
    "1c": ("code_vjepa_vggt.object_token_teacher_student.runtime_stage1c_joint", "Stage1CJointTrainer"),
    "2": ("code_vjepa_vggt.object_token_teacher_student.runtime", "TeacherStudentPredictorTrainer"),
}


def _import_trainer(stage: str):
    import importlib

    mod_path, cls_name = STAGE_MAP[stage]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def build_trainer(stage: str, cfg: dict, device: str):
    cls = _import_trainer(stage)
    # eval needs no optimizer; try to skip building it, fall back if signature differs
    try:
        return cls(cfg, build_optimizer=False, device=device)
    except TypeError:
        return cls(cfg, device=device)


def list_ckpts(ckpt_arg: str):
    p = Path(ckpt_arg)
    if p.is_dir():
        items = sorted(p.glob("step_*.pt"))
    elif p.is_file():
        items = [p]
    else:
        raise FileNotFoundError(f"ckpt path not found: {ckpt_arg}")
    out = []
    for f in items:
        m = re.search(r"step_(\d+)\.pt", f.name)
        out.append((int(m.group(1)) if m else 0, f))
    return out


def load_ckpt_into(trainer, ckpt_path: Path):
    import torch

    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    loaded = trainer.load_state_dict(model_state, strict=False)
    matched = len(model_state) - len(loaded.unexpected_keys)
    return matched, len(model_state), len(loaded.missing_keys), len(loaded.unexpected_keys)


def eval_one(trainer, device: str, max_batches: int, seed: int):
    import torch

    trainer.eval()
    # deterministic batch sampling + (for flow-matching) reproducible timestep noise
    torch.manual_seed(seed)
    loader = trainer.build_dataloader(num_workers=2)
    agg: dict[str, list[float]] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            torch.manual_seed(seed + n)  # same noise draw across checkpoints at batch n
            loss = trainer.forward(batch)
            metrics = getattr(trainer, "last_train_metrics", {}) or {}
            for k, v in metrics.items():
                agg.setdefault(k, []).append(float(v))
            agg.setdefault("train/loss", []).append(float(loss.detach().item()))
            n += 1
            if n >= max_batches:
                break
    # mean over batches; rename train/ -> val/
    means = {}
    for k, vals in agg.items():
        vk = k.replace("train/", "val/")
        means[vk] = statistics.fmean(vals) if vals else float("nan")
    means["val/num_batches"] = float(n)
    return means


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation-loss evaluator (teacher-student stages)")
    ap.add_argument("--stage", required=True, choices=list(STAGE_MAP))
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="a step_*.pt file or a dir containing them")
    ap.add_argument("--max-batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="vjepa_vggt_wan")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    import torch  # noqa
    from code_vjepa_vggt.utils.config import load_yaml_config

    cfg = load_yaml_config(args.config)
    cfg["data"]["split"] = args.split          # evaluate on val (or test)
    cfg["data"]["random_context_frames"] = False
    cfg.setdefault("logging", {})["use_wandb"] = False  # trainer must not init its own wandb

    print(f"[ts_eval] stage={args.stage} split={args.split} device={args.device} "
          f"building trainer (loads full model, ~minutes)...", flush=True)
    trainer = build_trainer(args.stage, cfg, args.device)

    ckpts = list_ckpts(args.ckpt)
    print(f"[ts_eval] {len(ckpts)} checkpoint(s) to evaluate", flush=True)

    wandb_run = None
    if not args.no_wandb:
        import wandb

        run_name = args.wandb_run_name or f"valeval_stage{args.stage}_{Path(args.config).stem}"
        wandb_run = wandb.init(project=args.wandb_project, name=run_name,
                               dir=cfg["logging"].get("wandb_dir", "/data/gaoya/agent-data/outputs/ts_smoke"))

    for step, fpath in ckpts:
        matched, total, miss, unexp = load_ckpt_into(trainer, fpath)
        if matched <= 0:
            print(f"[ts_eval] WARNING step={step}: 0 tensors matched ({fpath.name}); skipping", flush=True)
            continue
        means = eval_one(trainer, args.device, args.max_batches, args.seed)
        means["val/ckpt_step"] = float(step)
        means["val/init_matched"] = float(matched)
        line = ", ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in sorted(means.items()) if k.startswith("val/loss"))
        print(f"[ts_eval] step={step} matched={matched}/{total} miss={miss} unexp={unexp} | {line}", flush=True)
        if wandb_run is not None:
            wandb_run.log(means, step=step)

    if wandb_run is not None:
        wandb_run.finish()
    print("[ts_eval] done", flush=True)


if __name__ == "__main__":
    main()

