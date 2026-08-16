#!/usr/bin/env python3
"""Validate and launch the formal GT latent-mask CE experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import launch_from_config as shared


TRAIN_SCRIPT = (
    HERE / "train_xssc_object_self_attn_lora_frozen_motion_probe_latent_mask.py"
)


def require_latent_mask(config: dict, key: str):
    return shared.require(config, f"latent_mask_loss.{key}")


def _fairness_payload(config: dict, *, allow_gpu_set_override: bool = False) -> dict:
    payload = deepcopy(config)
    payload.pop("experiment", None)
    payload.pop("trajectory_loss", None)
    payload.pop("latent_mask_loss", None)
    payload.pop("comparison", None)
    payload.get("launch", {}).pop("main_process_port", None)
    if allow_gpu_set_override:
        payload.get("launch", {}).pop("gpu_set", None)
    payload.get("paths", {}).pop("output_root", None)
    payload.get("logging", {}).pop("wandb_name", None)
    return payload


def _collect_differences(left, right, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(f"{path}: {left.get(key)!r} != {right.get(key)!r}")
            else:
                differences.extend(_collect_differences(left[key], right[key], path))
        return differences
    if left != right:
        return [f"{prefix}: {left!r} != {right!r}"]
    return []


def validate_fairness(
    config: dict, reference_path: Path, *, allow_gpu_set_override: bool = False
) -> None:
    manifest = json.loads(reference_path.read_text(encoding="utf-8"))
    reference = manifest.get("resolved_config", manifest)
    differences = _collect_differences(
        _fairness_payload(config, allow_gpu_set_override=allow_gpu_set_override),
        _fairness_payload(reference, allow_gpu_set_override=allow_gpu_set_override),
    )
    if differences:
        raise ValueError(
            "formal latent-mask config differs from the CoTracker reference outside "
            "the allowed loss/run fields:\n" + "\n".join(differences[:30])
        )


def validate_config(raw: dict, config_dir: Path) -> dict:
    config = shared.validate_config(raw, config_dir)
    latent_mask = config.get("latent_mask_loss", {})
    if not bool(latent_mask.get("enabled", False)):
        raise ValueError("latent_mask_loss.enabled must be true")
    for loss_name in ("xssc_loss", "vjepa_loss", "trajectory_loss"):
        if bool(config.get(loss_name, {}).get("enabled", False)):
            raise ValueError(f"{loss_name} must be disabled for latent-mask training")
    if config["adaptation"]["mode"] != "full_sa":
        raise ValueError("latent-mask loss requires adaptation.mode=full_sa")
    if config["adaptation"]["enable_object_branch"]:
        raise ValueError("latent-mask loss requires the object branch to be disabled")
    if config["checkpointing"].get("resume_from"):
        raise ValueError("formal latent-mask training must start at step 0")
    if float(require_latent_mask(config, "weight")) <= 0.0:
        raise ValueError("latent_mask_loss.weight must be positive")
    if int(require_latent_mask(config, "query_latent_frame")) != 1:
        raise ValueError("latent_mask_loss.query_latent_frame must be 1 (pixel F04)")
    if int(require_latent_mask(config, "object_index")) != 0:
        raise ValueError("formal comparison fixes latent_mask_loss.object_index=0")
    if int(require_latent_mask(config, "expected_latent_frames")) != 13:
        raise ValueError("latent_mask_loss.expected_latent_frames must be 13")
    if int(require_latent_mask(config, "expected_pixel_frames")) != 49:
        raise ValueError("latent_mask_loss.expected_pixel_frames must be 49")
    if int(require_latent_mask(config, "gradient_diagnostics_every_n_forwards")) <= 0:
        raise ValueError("gradient diagnostics interval must be positive")

    for key in ("cache_dir", "head_config"):
        path = Path(
            shared.resolve_config_path(
                str(require_latent_mask(config, key)), config_dir
            )
        )
        if not path.exists():
            raise FileNotFoundError(f"latent_mask_loss.{key} does not exist: {path}")
        config["latent_mask_loss"][key] = str(path)
    reference_path = Path(
        shared.resolve_config_path(
            str(shared.require(config, "comparison.reference_resolved_config")),
            config_dir,
        )
    )
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"comparison reference does not exist: {reference_path}"
        )
    config["comparison"]["reference_resolved_config"] = str(reference_path)
    enforce_fairness = bool(config["comparison"].get("enforce_fairness", True))
    allow_gpu_set_override = bool(
        config["comparison"].get("allow_gpu_set_override", False)
    )
    if enforce_fairness:
        validate_fairness(
            config,
            reference_path,
            allow_gpu_set_override=allow_gpu_set_override,
        )
    elif not str(config["experiment"]["name"]).startswith("smoke_"):
        raise ValueError("only smoke_* experiments may disable the fairness check")
    return config


def add_option(command: list[str], name: str, value) -> None:
    command.extend((name, str(value)))


def build_command(config: dict, output_dir: Path) -> list[str]:
    command = shared.build_command(config, output_dir)
    script_index = command.index(str(shared.TRAIN_SCRIPT))
    command[script_index] = str(TRAIN_SCRIPT)
    loss = config["latent_mask_loss"]
    options = {
        "--motion_probe_wan_root": config["paths"]["wan_root"],
        "--motion_probe_head_config": loss["head_config"],
        "--motion_probe_head_subset_id": loss["head_subset_id"],
        "--motion_probe_head_feature_subtype": loss["head_feature_subtype"],
        "--probe_timestep": loss["probe_timestep"],
        "--probe_noise_level": loss["probe_noise_level"],
        "--motion_probe_pck_weight_power": loss["pck_weight_power"],
        "--motion_probe_latent_mask_weight": loss["weight"],
        "--motion_probe_mask_cache_root": loss["cache_dir"],
        "--motion_probe_tracking_mask_key": loss["tracking_mask_key"],
        "--motion_probe_query_latent_frame": loss["query_latent_frame"],
        "--motion_probe_query_object_index": loss["object_index"],
        "--motion_probe_expected_latent_frames": loss["expected_latent_frames"],
        "--motion_probe_expected_pixel_frames": loss["expected_pixel_frames"],
        "--motion_probe_gradient_diagnostics_every_n_forwards": loss[
            "gradient_diagnostics_every_n_forwards"
        ],
    }
    for name, value in options.items():
        add_option(command, name, value)
    if bool(loss.get("main_gradient_checkpointing_offload", True)):
        command.append("--use_gradient_checkpointing_offload")
    if not bool(loss.get("probe_gradient_checkpointing_offload", True)):
        command.append("--disable_motion_probe_gradient_checkpointing_offload")
    init_scan_limit = config["data"].get("pybullet_init_scan_limit")
    if init_scan_limit is not None:
        add_option(command, "--pybullet0713_init_scan_limit", init_scan_limit)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, sources = shared.load_config(args.config)
    config = validate_config(raw, args.config.expanduser().resolve().parent)
    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(config["paths"]["output_root"]) / config["experiment"]["name"] / run_tag
    )
    command = build_command(config, output_dir)
    effective_batch = (
        int(config["launch"]["num_processes"])
        * int(config["optimization"]["train_batch_size_per_gpu"])
        * int(config["optimization"]["gradient_accumulation_steps"])
    )
    allow_gpu_set_override = bool(
        config["comparison"].get("allow_gpu_set_override", False)
    )
    reference_manifest = json.loads(
        Path(config["comparison"]["reference_resolved_config"]).read_text(
            encoding="utf-8"
        )
    )
    reference_config = reference_manifest.get("resolved_config", reference_manifest)
    summary = {
        "experiment": config["experiment"]["name"],
        "mode": config["adaptation"]["mode"],
        "enable_object_branch": config["adaptation"]["enable_object_branch"],
        "initialization": config["initialization"]["type"],
        "latent_mask_loss_enabled": True,
        "fairness_reference": config["comparison"]["reference_resolved_config"],
        "fairness_check": (
            (
                "passed_with_gpu_resource_override"
                if allow_gpu_set_override
                else "passed"
            )
            if bool(config["comparison"].get("enforce_fairness", True))
            else "smoke_override"
        ),
        "gpu_resource_override": (
            {
                "reference": reference_config["launch"]["gpu_set"],
                "actual": config["launch"]["gpu_set"],
            }
            if allow_gpu_set_override
            else None
        ),
        "start_step": 0,
        "config_sources": sources,
        "gpu_set": config["launch"]["gpu_set"],
        "effective_batch": effective_batch,
        "output_dir": str(output_dir),
        "command": shlex.join(command),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    if args.validate_only or args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    cache_root = Path(config["paths"]["cache_root"])
    cache_dirs = {
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for path in cache_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sources": sources,
        "resolved_config": config,
        "head_selection_snapshot": None,
        "launch_summary": summary,
    }
    (output_dir / "resolved_experiment_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "launch_command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env.update({key: str(value) for key, value in cache_dirs.items()})
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(config["launch"]["gpu_set"])
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if config["logging"].get("wandb_run_id"):
        env["WANDB_RUN_ID"] = str(config["logging"]["wandb_run_id"])
        env["WANDB_RESUME"] = str(config["logging"]["wandb_resume"])
    env["PYTHONPATH"] = os.pathsep.join(
        [
            config["paths"]["project_root"],
            config["paths"]["diffsynth_root"],
            str(EXPERIMENT_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
