from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import collate_video_batch


def _module_bucket(name: str) -> str:
    if name.startswith("bundle.dit."):
        return "bundle.dit"
    if name.startswith("object_pooler."):
        return "object_pooler"
    if name.startswith("object_aux_heads."):
        return "object_aux_heads"
    if name.startswith("object_adapter."):
        return "object_adapter"
    if name.startswith("jepa_adapter."):
        return "jepa_adapter"
    if name.startswith("vggt_adapter."):
        return "vggt_adapter"
    return name.split(".", 1)[0]


def _grad_report(model: torch.nn.Module) -> dict[str, Any]:
    trainable_params = []
    frozen_with_grad = []
    module_summary: dict[str, dict[str, Any]] = {}
    top_params = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad is not None and not param.requires_grad:
            frozen_with_grad.append(name)
        if not param.requires_grad:
            continue

        entry = {
            "name": name,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "grad_present": grad is not None,
        }
        if grad is not None:
            grad_f = grad.detach().float()
            entry.update(
                {
                    "grad_finite": bool(torch.isfinite(grad_f).all().item()),
                    "grad_norm": float(grad_f.norm().item()),
                    "grad_abs_max": float(grad_f.abs().max().item()),
                    "grad_abs_mean": float(grad_f.abs().mean().item()),
                }
            )
            top_params.append(entry)
        trainable_params.append(entry)

        bucket = _module_bucket(name)
        bucket_entry = module_summary.setdefault(
            bucket,
            {
                "num_params": 0,
                "numel": 0,
                "params_with_grad": 0,
                "params_without_grad": 0,
                "finite_grad_params": 0,
                "grad_norm_sum": 0.0,
                "grad_abs_max": 0.0,
            },
        )
        bucket_entry["num_params"] += 1
        bucket_entry["numel"] += int(param.numel())
        if grad is None:
            bucket_entry["params_without_grad"] += 1
        else:
            bucket_entry["params_with_grad"] += 1
            grad_f = grad.detach().float()
            finite = bool(torch.isfinite(grad_f).all().item())
            bucket_entry["finite_grad_params"] += int(finite)
            bucket_entry["grad_norm_sum"] += float(grad_f.norm().item())
            bucket_entry["grad_abs_max"] = max(bucket_entry["grad_abs_max"], float(grad_f.abs().max().item()))

    top_params = sorted(top_params, key=lambda x: x["grad_norm"], reverse=True)[:20]
    trainable_missing_grad = [entry["name"] for entry in trainable_params if not entry["grad_present"]]
    nonfinite_grad = [entry["name"] for entry in trainable_params if entry["grad_present"] and not entry["grad_finite"]]
    return {
        "num_trainable_tensors": len(trainable_params),
        "trainable_missing_grad": trainable_missing_grad,
        "nonfinite_grad_tensors": nonfinite_grad,
        "frozen_with_grad": frozen_with_grad,
        "module_summary": module_summary,
        "top_grad_params": top_params,
    }


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    status = report["status"]
    issues = report.get("issues", [])
    metrics = report.get("forward_metrics", {})
    grad = report.get("grad_report", {})
    prepare_debug = report.get("prepare_debug", {})
    lines = [
        f"# Train Smoke Test",
        "",
        f"- status: `{status}`",
        f"- config: `{report.get('config_path', '')}`",
        f"- output_dir: `{report.get('output_dir', '')}`",
        f"- device: `{report.get('device', '')}`",
        f"- loss_total: `{metrics.get('train/loss_total', 'n/a')}`",
        f"- loss_main: `{metrics.get('train/loss_main', 'n/a')}`",
        f"- loss_track_aux: `{metrics.get('train/loss_track_aux', 'n/a')}`",
        f"- loss_box_aux: `{metrics.get('train/loss_box_aux', 'n/a')}`",
        f"- loss_depth_aux: `{metrics.get('train/loss_depth_aux', 'n/a')}`",
        "",
        "## Shapes",
        "",
        f"- text_context: `{prepare_debug.get('text_context')}`",
        f"- context_latents: `{prepare_debug.get('context_latents')}`",
        f"- active_tracks: `{prepare_debug.get('active_tracks')}`",
        f"- object_latent_tokens: `{prepare_debug.get('object_latent_tokens')}`",
        f"- object_context: `{prepare_debug.get('object_context')}`",
        f"- object_aux_pred_track_summary: `{prepare_debug.get('object_aux_pred_track_summary')}`",
        f"- object_aux_pred_box_xyxy: `{prepare_debug.get('object_aux_pred_box_xyxy')}`",
        f"- object_aux_pred_depth: `{prepare_debug.get('object_aux_pred_depth')}`",
        "",
        "## Gradients",
        "",
        f"- num_trainable_tensors: `{grad.get('num_trainable_tensors', 0)}`",
        f"- trainable_missing_grad: `{len(grad.get('trainable_missing_grad', []))}`",
        f"- nonfinite_grad_tensors: `{len(grad.get('nonfinite_grad_tensors', []))}`",
        f"- frozen_with_grad: `{len(grad.get('frozen_with_grad', []))}`",
        "",
        "## Issues",
        "",
    ]
    if issues:
        lines.extend([f"- {issue}" for issue in issues])
    else:
        lines.append("- none detected")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flush_report(report_path: Path, summary_path: Path, report: dict[str, Any]) -> None:
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_summary(summary_path, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="/data/gaoya/AAA_test_video/0623/train/smoke_test",
    )
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--disable-sam2-priors", action="store_true")
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--init-scan-limit", type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "smoke_report.json"
    summary_path = output_dir / "smoke_summary.md"

    cfg = load_yaml_config(args.config)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1
    if args.resolution is not None:
        cfg["data"]["resolution"] = [int(args.resolution[0]), int(args.resolution[1])]
    if args.disable_sam2_priors:
        cfg["model"]["enable_sam2_priors"] = False
    if args.init_scan_limit is not None:
        cfg["data"]["init_scan_limit"] = int(args.init_scan_limit)

    report: dict[str, Any] = {
        "status": "started",
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "sample_index": int(args.index),
        "detect_anomaly": bool(args.detect_anomaly),
        "effective_resolution": cfg["data"]["resolution"],
        "enable_sam2_priors": bool(cfg["model"].get("enable_sam2_priors", False)),
        "init_scan_limit": cfg["data"].get("init_scan_limit"),
    }
    _flush_report(report_path, summary_path, report)

    try:
        report["phase"] = "construct_trainer"
        _flush_report(report_path, summary_path, report)
        trainer = ContextVideoTrainer(cfg, build_optimizer=True)
        torch.nn.Module.train(trainer, True)
        report["device"] = str(trainer.device_obj)
        sample = trainer.dataset[int(args.index)]
        batch = collate_video_batch([sample])
        report["batch_keys"] = sorted(batch.keys())
        _flush_report(report_path, summary_path, report)

        if args.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        report["phase"] = "prepare_batch"
        _flush_report(report_path, summary_path, report)
        with torch.no_grad():
            prepared = trainer._prepare_batch(batch)
        report["prepare_debug"] = prepared["debug"]
        _flush_report(report_path, summary_path, report)

        report["phase"] = "forward"
        _flush_report(report_path, summary_path, report)
        trainer.zero_grad(set_to_none=True)
        loss = trainer.forward(batch)
        report["forward_metrics"] = {
            "loss": float(loss.detach().item()),
            **trainer.last_train_metrics,
        }
        _flush_report(report_path, summary_path, report)

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"non-finite loss detected: {float(torch.nan_to_num(loss).item())}")

        report["phase"] = "backward"
        _flush_report(report_path, summary_path, report)
        loss.backward()
        report["grad_report"] = _grad_report(trainer)

        issues = []
        grad_report = report["grad_report"]
        if grad_report["trainable_missing_grad"]:
            issues.append(f"trainable tensors without grad: {len(grad_report['trainable_missing_grad'])}")
        if grad_report["nonfinite_grad_tensors"]:
            issues.append(f"non-finite gradients: {len(grad_report['nonfinite_grad_tensors'])}")
        if grad_report["frozen_with_grad"]:
            issues.append(f"frozen tensors received gradients: {len(grad_report['frozen_with_grad'])}")

        metrics = report["forward_metrics"]
        for key in ("train/loss_total", "train/loss_main", "train/loss_track_aux", "train/loss_box_aux", "train/loss_depth_aux"):
            value = metrics.get(key)
            if value is not None and (not isinstance(value, (int, float)) or value != value):
                issues.append(f"metric {key} is NaN or invalid")

        object_context_shape = prepared["debug"].get("object_context")
        object_latent_shape = prepared["debug"].get("object_latent_tokens")
        if isinstance(object_context_shape, list) and isinstance(object_latent_shape, list) and len(object_latent_shape) == 4:
            expected_object_tokens = int(object_latent_shape[1]) * int(object_latent_shape[2])
            if int(object_context_shape[1]) != expected_object_tokens:
                issues.append(
                    f"object_context token count mismatch: got {object_context_shape[1]}, expected {expected_object_tokens}"
                )

        report["issues"] = issues
        report["status"] = "ok" if not issues else "warning"
        report["phase"] = "done"
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    _flush_report(report_path, summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
