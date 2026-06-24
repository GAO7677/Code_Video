from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from code_vjepa_vggt.train_v_newtrain import build_accelerator, build_dataset, build_model, prepare_args, wan_parser


def _clone_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            out[name] = param.detach().cpu().clone()
    return out


def _compare(before: dict[str, torch.Tensor], after_model: torch.nn.Module) -> dict[str, object]:
    changed = []
    unchanged = []
    for name, param in after_model.named_parameters():
        if not param.requires_grad or name not in before:
            continue
        now = param.detach().cpu()
        if torch.equal(before[name], now):
            unchanged.append(name)
        else:
            changed.append(name)
    return {
        "changed_count": len(changed),
        "unchanged_count": len(unchanged),
        "changed_preview": changed[:40],
        "unchanged_preview": unchanged[:40],
    }


def main() -> None:
    parser = wan_parser()
    parser.add_argument("--smoke-output-dir", required=True)
    args = prepare_args(parser.parse_args())
    output_dir = Path(args.smoke_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "smoke_report.json"

    accelerator = build_accelerator(args)
    dataset = build_dataset(args)
    model = build_model(args, accelerator)
    model.to(accelerator.device)
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)

    sample = dataset[0]
    before = _clone_trainable(model)
    loss = model(sample)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {float(torch.nan_to_num(loss).item())}")
    accelerator.backward(loss)

    grad_summary = {
        "num_trainable": 0,
        "with_grad": 0,
        "without_grad": 0,
        "nonfinite_grad": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad_summary["num_trainable"] += 1
        if param.grad is None:
            grad_summary["without_grad"] += 1
            continue
        grad_summary["with_grad"] += 1
        if not torch.isfinite(param.grad).all():
            grad_summary["nonfinite_grad"].append(name)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    delta = _compare(before, model)

    report = {
        "loss": float(loss.detach().item()),
        "last_train_metrics": getattr(model, "last_train_metrics", {}),
        "grad_summary": grad_summary,
        "param_delta": delta,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
