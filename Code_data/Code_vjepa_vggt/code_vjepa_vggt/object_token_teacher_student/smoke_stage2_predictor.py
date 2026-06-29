from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.utils.config import load_yaml_config

from .runtime import TeacherStudentPredictorTrainer


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _grad_report(model: torch.nn.Module) -> dict[str, Any]:
    num_trainable = 0
    with_grad = 0
    nonfinite_grad: list[str] = []
    grad_abs_max = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num_trainable += 1
        if param.grad is None:
            continue
        with_grad += 1
        grad = param.grad.detach().float()
        if not torch.isfinite(grad).all():
            nonfinite_grad.append(name)
        if grad.numel() > 0:
            grad_abs_max = max(grad_abs_max, float(grad.abs().max().item()))
    return {
        "num_trainable": num_trainable,
        "with_grad": with_grad,
        "nonfinite_grad": nonfinite_grad,
        "grad_abs_max": grad_abs_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step smoke test for Stage2 predictor training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml_config(args.config)
    trainer = TeacherStudentPredictorTrainer(cfg, build_optimizer=False, device=_resolve_launch_device())

    sample = trainer.dataset[int(args.index)]
    dataloader = trainer.build_dataloader(num_workers=0)
    collate_fn = getattr(dataloader, "collate_fn", None)
    if collate_fn is None:
        raise RuntimeError("trainer dataloader does not expose a collate_fn for smoke batching")
    collated = collate_fn([sample])

    optimizer = torch.optim.AdamW(
        trainer.trainable_parameters(),
        lr=float(cfg["optimization"]["lr"]),
        weight_decay=float(cfg["optimization"]["weight_decay"]),
    )

    optimizer.zero_grad(set_to_none=True)
    loss = trainer(collated)
    loss.backward()
    grad_before = _grad_report(trainer)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    prepared = trainer._prepare_predictor_batch(collated)
    report = {
        "sample_index": int(args.index),
        "seed": int(args.seed),
        "loss": float(loss.detach().item()),
        "metrics": dict(trainer.last_train_metrics),
        "grad_summary": grad_before,
        "shapes": {
            "teacher_context_tokens": list(prepared.teacher_context_tokens.shape),
            "teacher_future_tokens": list(prepared.teacher_future_tokens.shape),
            "pred_future_tokens": list(prepared.pred_future_tokens.shape),
            "pred_future_track_summary": list(prepared.pred_future_track_summary.shape),
            "pred_future_box_xyxy": list(prepared.pred_future_box_xyxy.shape),
            "gt_future_track_summary": list(prepared.gt_future_track_summary.shape),
            "gt_future_box_xyxy": list(prepared.gt_future_box_xyxy.shape),
            "object_valid_mask": list(prepared.object_valid_mask.shape),
        },
    }
    out_path = output_dir / "smoke_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
