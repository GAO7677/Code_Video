#!/usr/bin/env python3
"""Create a watcher-compatible, adapter-free Wan2.2 + OpenVid baseline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


DEFAULT_TEMPLATE = Path(
    "/data/gaoya/agent-data/checkpoints/xssc_feature_loss/"
    "t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_"
    "dinov3_movic_step50000/formal_gpu01/resolved_experiment_config.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/checkpoints/inference_baselines/"
    "wan22_openvid_lora_no_additional_adapter"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_manifest(template: dict) -> dict:
    manifest = deepcopy(template)
    config = manifest["resolved_config"]
    config["experiment"] = {
        "name": "wan22_openvid_lora_baseline",
        "description": (
            "Wan2.2-TI2V-5B with the OpenVid rank-32 LoRA merged; "
            "no additional Object, Full-SA, or selected-head adapter"
        ),
        "expected_trainable_params": 0,
    }
    config["initialization"] = {"type": "openvid_lora"}
    config["adaptation"]["mode"] = "object_only"
    config["adaptation"]["enable_object_branch"] = False
    config["logging"]["wandb_name"] = "wan22_openvid_lora_baseline"
    if "xssc_loss" in config:
        config["xssc_loss"]["enabled"] = False
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["config_sources"] = [str(DEFAULT_TEMPLATE)]
    manifest["head_selection_snapshot"] = None
    manifest["launch_summary"] = {
        "purpose": "inference-only baseline",
        "trainable_params": 0,
        "additional_adapter_checkpoint_loaded": False,
    }
    return manifest


def main() -> None:
    args = parse_args()
    template_path = args.template.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    manifest = build_manifest(template)
    config = manifest["resolved_config"]
    pretrained_lora = Path(config["paths"]["pretrained_lora_checkpoint"])
    if not pretrained_lora.is_file():
        raise FileNotFoundError(f"OpenVid LoRA checkpoint not found: {pretrained_lora}")

    checkpoint_dir = output_root / "checkpoints" / "step-000000"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_experiment_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_file(
        {"baseline_inference_marker": torch.tensor([1], dtype=torch.uint8)},
        checkpoint_dir / "checkpoint.safetensors",
        metadata={
            "model": "Wan2.2-TI2V-5B + merged OpenVid LoRA",
            "additional_adapter": "none",
        },
    )
    torch.save(
        {
            "global_step": 0,
            "model_logger_num_steps": 0,
            "inference_only": True,
            "additional_adapter_checkpoint_loaded": False,
        },
        checkpoint_dir / "training_state.pt",
    )
    print(checkpoint_dir)


if __name__ == "__main__":
    main()
