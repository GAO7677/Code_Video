from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from code_vjepa_vggt.smoke_train_forward_backward import _grad_report
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import collate_video_batch


def _flush(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _summarize_modules(grad_report: dict[str, Any], module_names: list[str]) -> dict[str, Any]:
    summary = grad_report.get("module_summary", {})
    out: dict[str, Any] = {}
    for name in module_names:
        if name in summary:
            out[name] = summary[name]
    return out


def _run_backward_case(
    trainer: ContextVideoTrainer,
    batch: dict[str, Any],
    *,
    label: str,
    loss_tensor: torch.Tensor,
) -> dict[str, Any]:
    trainer.zero_grad(set_to_none=True)
    loss_tensor.backward()
    grad_report = _grad_report(trainer)
    return {
        "label": label,
        "loss_value": float(loss_tensor.detach().item()),
        "module_summary_focus": _summarize_modules(
            grad_report,
            [
                "bundle.dit",
                "object_pooler",
                "object_aux_heads",
                "object_adapter",
                "jepa_adapter",
                "vggt_adapter",
            ],
        ),
        "top_grad_params": grad_report.get("top_grad_params", []),
        "num_trainable_tensors": grad_report.get("num_trainable_tensors", 0),
        "trainable_missing_grad_count": len(grad_report.get("trainable_missing_grad", [])),
        "nonfinite_grad_count": len(grad_report.get("nonfinite_grad_tensors", [])),
        "frozen_with_grad_count": len(grad_report.get("frozen_with_grad", [])),
        "trainable_missing_grad_head": grad_report.get("trainable_missing_grad", [])[:50],
        "nonfinite_grad_tensors": grad_report.get("nonfinite_grad_tensors", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="/data/gaoya/AAA_test_video/0623/train/smoke_test_main_loss_diag",
    )
    parser.add_argument("--init-scan-limit", type=int, default=1)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "main_loss_grad_report.json"

    cfg = load_yaml_config(args.config)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1
    if args.init_scan_limit is not None:
        cfg["data"]["init_scan_limit"] = int(args.init_scan_limit)
    if args.resolution is not None:
        cfg["data"]["resolution"] = [int(args.resolution[0]), int(args.resolution[1])]

    report: dict[str, Any] = {
        "status": "started",
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "sample_index": int(args.index),
        "effective_resolution": cfg["data"]["resolution"],
        "init_scan_limit": cfg["data"].get("init_scan_limit"),
    }
    _flush(report_path, report)

    try:
        trainer = ContextVideoTrainer(cfg, build_optimizer=True)
        torch.nn.Module.train(trainer, True)
        sample = trainer.dataset[int(args.index)]
        batch = collate_video_batch([sample])

        report["phase"] = "forward_total"
        _flush(report_path, report)
        total_loss = trainer.forward(batch)
        loss_breakdown = trainer.last_loss_breakdown
        report["forward_metrics"] = {
            "loss_total": float(total_loss.detach().item()),
            **trainer.last_train_metrics,
        }
        report["prepare_debug"] = trainer._prepare_batch(batch)["debug"]
        report["loss_breakdown_values"] = {
            key: float(value.detach().item())
            for key, value in loss_breakdown.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        _flush(report_path, report)

        main_case = _run_backward_case(
            trainer,
            batch,
            label="loss_main_only",
            loss_tensor=loss_breakdown["loss_main"],
        )
        report["main_case"] = main_case
        _flush(report_path, report)

        total_loss = trainer.forward(batch)
        loss_breakdown = trainer.last_loss_breakdown
        aux_only = (
            loss_breakdown["lambda_track_aux"] * loss_breakdown["track_aux_loss"]
            + loss_breakdown["lambda_box_aux"] * loss_breakdown["box_aux_loss"]
            + loss_breakdown["lambda_depth_aux"] * loss_breakdown["depth_aux_loss"]
        )
        report["phase"] = "backward_aux_only"
        _flush(report_path, report)
        aux_case = _run_backward_case(
            trainer,
            batch,
            label="aux_only",
            loss_tensor=aux_only,
        )
        report["aux_case"] = aux_case
        _flush(report_path, report)

        total_loss = trainer.forward(batch)
        loss_breakdown = trainer.last_loss_breakdown
        report["phase"] = "backward_total"
        _flush(report_path, report)
        total_case = _run_backward_case(
            trainer,
            batch,
            label="loss_total",
            loss_tensor=loss_breakdown["loss_total"],
        )
        report["total_case"] = total_case
        report["status"] = "ok"
        report["phase"] = "done"
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()

    _flush(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
