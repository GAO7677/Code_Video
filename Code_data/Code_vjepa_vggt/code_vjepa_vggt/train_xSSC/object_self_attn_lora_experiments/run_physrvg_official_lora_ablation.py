#!/usr/bin/env python3
"""Run and register the official PhysRVG LoRA OFF/ON reference baselines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from xssc_lora_checkpoint_watch import (  # noqa: E402
    atomic_write_json,
    load_json,
    method_config,
    read_inputs,
    register_manifest,
    state_paths,
    timestamp,
    validate_result_root,
)
from xssc_lora_physiciq_watch import (  # noqa: E402
    append_leaf_folder,
    phys_manifest_path,
    phys_state_root,
)


PYTHON = Path("/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python")
INFER_SCRIPT = Path("/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/infer_full_sa_lora_json_list.py")
MODEL_ID = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers")
DIT_CHECKPOINT = Path("/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors")
OFFICIAL_LORA = Path("/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method-key", choices=["physrvg_test5_lora_off", "physrvg_test5_lora_on"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run_inference(
    *,
    input_list: Path,
    output_root: Path,
    num_inference_steps: int,
    gpu: int,
    lora_checkpoint: Path | None,
) -> None:
    command = [
        str(PYTHON),
        str(INFER_SCRIPT),
        "--input-json-list",
        str(input_list),
        "--output-root",
        str(output_root),
        "--model-id",
        str(MODEL_ID),
        "--physrvg-dit-checkpoint",
        str(DIT_CHECKPOINT),
        "--device",
        "cuda:0",
        "--height",
        "512",
        "--width",
        "896",
        "--num-frames",
        "49",
        "--fps",
        "30",
        "--num-inference-steps",
        str(num_inference_steps),
        "--seed",
        "42",
        "--flat-output",
        "--force",
    ]
    if lora_checkpoint is not None:
        command.extend(["--lora-checkpoint", str(lora_checkpoint)])
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/home/gaoya/code_V2V_baselines/PhysRVG-main",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root.parent.parent / "logs" / f"{output_root.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def register_physiciq_manifest(
    config: dict[str, Any],
    *,
    method_key: str,
    step: int,
    checkpoint_dir: Path,
    result_root: Path,
    input_list: Path,
    gpu: int,
) -> None:
    phys = config["physiciq"]
    validation = validate_result_root(
        config,
        result_root,
        input_list=input_list,
        expected_cases=int(phys["expected_cases"]),
    )
    method = method_config(config, method_key)
    payload = {
        "method_key": method_key,
        "method_label": method["label"],
        "step": step,
        "checkpoint_dir": str(checkpoint_dir),
        "result_root": str(result_root),
        "input_list": str(input_list),
        "num_inference_steps": int(phys["num_inference_steps"]),
        "gpu_id": gpu,
        "completed_utc": timestamp(),
        "validation": validation,
        "origin": "official_lora_ablation",
    }
    manifest_path = phys_manifest_path(config, method_key, step)
    atomic_write_json(manifest_path, payload)
    meta_root = Path(phys["output_root"]).resolve() / "_run_meta" / (
        f"xssc_lora_{method_key}_step-{step:06d}_steps40_512x896_ctx08_49f"
    )
    atomic_write_json(meta_root / "batch_manifest.json", payload)
    append_leaf_folder(config, result_root)


def run_baseline(config: dict[str, Any], method_key: str, gpu: int) -> None:
    method = method_config(config, method_key)
    step = 0
    checkpoint_dir = DIT_CHECKPOINT if method_key.endswith("lora_off") else OFFICIAL_LORA
    lora_checkpoint = None if method_key.endswith("lora_off") else OFFICIAL_LORA
    paths = state_paths(config)

    test5_input = Path(config["paths"]["input_list"]).resolve()
    test5_root = (
        paths["results"]
        / method_key
        / "step-000000_steps8_512x896_ctx08_49f"
    )
    test5_manifest = paths["checkpoints"] / method_key / "step-000000.json"
    if not test5_manifest.is_file():
        run_inference(
            input_list=test5_input,
            output_root=test5_root,
            num_inference_steps=int(config["runtime"]["num_inference_steps"]),
            gpu=gpu,
            lora_checkpoint=lora_checkpoint,
        )
        task = {
            "method_key": method_key,
            "method_label": method["label"],
            "method_index": next(i for i, item in enumerate(config["methods"]) if item["key"] == method_key),
            "step": step,
            "checkpoint_dir": str(checkpoint_dir),
            "source": "static",
        }
        register_manifest(config, task, test5_root, "official_lora_ablation")

    phys_input = Path(config["physiciq"]["input_list"]).resolve()
    phys_name = config["physiciq"]["method_name_template"].format(
        method_key=method_key,
        step=step,
    )
    phys_root = Path(config["physiciq"]["output_root"]).resolve() / phys_name
    phys_manifest = phys_manifest_path(config, method_key, step)
    if not phys_manifest.is_file():
        run_inference(
            input_list=phys_input,
            output_root=phys_root,
            num_inference_steps=int(config["physiciq"]["num_inference_steps"]),
            gpu=gpu,
            lora_checkpoint=lora_checkpoint,
        )
        register_physiciq_manifest(
            config,
            method_key=method_key,
            step=step,
            checkpoint_dir=checkpoint_dir,
            result_root=phys_root,
            input_list=phys_input,
            gpu=gpu,
        )

    subprocess.run(
        [
            config["paths"]["python"],
            config["paths"]["dashboard_builder"],
            "--config",
            config["_config_path"],
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.gpu == 4:
        raise SystemExit("GPU4 is forbidden")
    if not PYTHON.is_file() or not INFER_SCRIPT.is_file():
        raise FileNotFoundError("PhysRVG inference environment or script is missing")
    if not MODEL_ID.is_dir() or not DIT_CHECKPOINT.is_file():
        raise FileNotFoundError("official PhysRVG base or DiT checkpoint is missing")
    if args.method_key.endswith("lora_on") and not OFFICIAL_LORA.is_dir():
        raise FileNotFoundError(f"official LoRA checkpoint is missing: {OFFICIAL_LORA}")
    config = load_json(args.config.resolve())
    config["_config_path"] = str(args.config.resolve())
    run_baseline(config, args.method_key, args.gpu)


if __name__ == "__main__":
    main()
