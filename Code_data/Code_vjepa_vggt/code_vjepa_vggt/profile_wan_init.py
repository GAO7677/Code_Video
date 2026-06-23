from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.wan_like.bootstrap import load_wan_model


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _meta_param_names(module: torch.nn.Module) -> list[str]:
    out = []
    for name, param in module.named_parameters():
        if getattr(param, "is_meta", False):
            out.append(name)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/gaoya/AAA_test_video/0623/train/smoke_test",
    )
    parser.add_argument("--skip-text-vae", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "wan_init_profile.json"

    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    device = _resolve_launch_device()
    device_obj = torch.device(device)

    report: dict[str, Any] = {
        "status": "started",
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "device": str(device_obj),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "skip_text_vae": bool(args.skip_text_vae),
        "steps": [],
    }

    def mark(step: str, **extra: Any) -> None:
        report["steps"].append({"step": step, **extra})
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        t0 = time.perf_counter()
        mark("load_wan_model_class_start")
        WanModel = load_wan_model()
        mark("load_wan_model_class_done", elapsed_sec=time.perf_counter() - t0, wan_model=str(WanModel))

        t1 = time.perf_counter()
        mark("construct_bundle_start")
        bundle = WanContextVideoModel(
            ckpt_dir=model_cfg["wan_ckpt_dir"],
            task=model_cfg["wan_task"],
            device=str(device_obj),
            load_dit=not bool(args.skip_text_vae),
            lora_rank=int(model_cfg.get("wan_lora_rank", 0)),
            lora_alpha=int(model_cfg.get("wan_lora_alpha", 0)),
            lora_dropout=float(model_cfg.get("wan_lora_dropout", 0.0)),
            lora_init=str(model_cfg.get("wan_lora_init", "gaussian")),
        )
        mark("construct_bundle_done", elapsed_sec=time.perf_counter() - t1)

        if args.skip_text_vae:
            t2 = time.perf_counter()
            mark("ensure_dit_loaded_start")
            bundle.ensure_dit_loaded()
            mark("ensure_dit_loaded_done", elapsed_sec=time.perf_counter() - t2)

        if bundle.dit is None:
            raise RuntimeError("bundle.dit is None after initialization")

        meta_names = _meta_param_names(bundle.dit)
        mark(
            "inspect_dit",
            num_named_parameters=sum(1 for _ in bundle.dit.named_parameters()),
            num_meta_parameters=len(meta_names),
            first_meta_parameter=(meta_names[0] if meta_names else None),
            dtype=str(next(bundle.dit.parameters()).dtype),
            device=str(next(bundle.dit.parameters()).device),
        )

        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
