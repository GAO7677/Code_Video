#!/usr/bin/env python3
"""Validate and launch the CoTracker trajectory-loss experiment config."""

from __future__ import annotations

import argparse
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


TRAIN_SCRIPT = HERE / "train_xssc_object_self_attn_lora_trajectory_loss.py"
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")


def require_trajectory(config: dict, key: str):
    return shared.require(config, f"trajectory_loss.{key}")


def validate_config(raw: dict, config_dir: Path) -> dict:
    config = shared.validate_config(raw, config_dir)
    trajectory = config.get("trajectory_loss", {})
    if not bool(trajectory.get("enabled", False)):
        raise ValueError("trajectory_loss.enabled must be true")
    if bool(config.get("xssc_loss", {}).get("enabled", False)):
        raise ValueError("xssc_loss must be disabled for trajectory-loss training")
    if bool(config.get("vjepa_loss", {}).get("enabled", False)):
        raise ValueError("vjepa_loss must be disabled for trajectory-loss training")
    if config["adaptation"]["mode"] != "full_sa":
        raise ValueError("trajectory loss requires adaptation.mode=full_sa")
    if config["adaptation"]["enable_object_branch"]:
        raise ValueError("trajectory loss requires the object branch to be disabled")
    if config["checkpointing"].get("resume_from"):
        raise ValueError("trajectory-loss runs must start at step 0, not resume")
    if float(require_trajectory(config, "weight")) <= 0.0:
        raise ValueError("trajectory_loss.weight must be positive")
    if int(require_trajectory(config, "anchor_frame")) != 4:
        raise ValueError("trajectory_loss.anchor_frame must be 4")
    if int(require_trajectory(config, "future_start_frame")) != 8:
        raise ValueError("trajectory_loss.future_start_frame must be 8")
    if int(require_trajectory(config, "points_per_object")) <= 0:
        raise ValueError("trajectory_loss.points_per_object must be positive")
    if float(require_trajectory(config, "huber_delta")) <= 0.0:
        raise ValueError("trajectory_loss.huber_delta must be positive")
    threshold = float(require_trajectory(config, "visibility_threshold"))
    if not 0.0 < threshold < 1.0:
        raise ValueError("trajectory_loss.visibility_threshold must be in (0,1)")
    if float(require_trajectory(config, "visibility_loss_weight")) < 0.0:
        raise ValueError("trajectory_loss.visibility_loss_weight must be non-negative")
    if int(require_trajectory(config, "gradient_diagnostics_every_n_forwards")) <= 0:
        raise ValueError(
            "trajectory_loss.gradient_diagnostics_every_n_forwards must be positive"
        )
    for key in (
        "cache_dir",
        "cotracker_checkpoint",
        "tiny_vae_root",
        "tiny_vae_checkpoint",
    ):
        path = Path(
            shared.resolve_config_path(str(require_trajectory(config, key)), config_dir)
        )
        if not path.exists():
            raise FileNotFoundError(f"trajectory_loss.{key} does not exist: {path}")
        config["trajectory_loss"][key] = str(path)
    return config


def add_option(command: list[str], name: str, value) -> None:
    command.extend((name, str(value)))


def build_command(config: dict, output_dir: Path) -> list[str]:
    command = shared.build_command(config, output_dir)
    script_index = command.index(str(shared.TRAIN_SCRIPT))
    command[script_index] = str(TRAIN_SCRIPT)
    trajectory = config["trajectory_loss"]
    options = {
        "--trajectory_cache_dir": trajectory["cache_dir"],
        "--trajectory_loss_weight": trajectory["weight"],
        "--trajectory_anchor_frame": trajectory["anchor_frame"],
        "--trajectory_future_start_frame": trajectory["future_start_frame"],
        "--trajectory_points_per_object": trajectory["points_per_object"],
        "--trajectory_huber_delta": trajectory["huber_delta"],
        "--trajectory_visibility_threshold": trajectory["visibility_threshold"],
        "--trajectory_visibility_loss_weight": trajectory["visibility_loss_weight"],
        "--trajectory_gradient_diagnostics_every_n_forwards": trajectory[
            "gradient_diagnostics_every_n_forwards"
        ],
        "--trajectory_cotracker_checkpoint": trajectory["cotracker_checkpoint"],
        "--tiny_vae_root": trajectory["tiny_vae_root"],
        "--tiny_vae_checkpoint": trajectory["tiny_vae_checkpoint"],
    }
    for name, value in options.items():
        add_option(command, name, value)
    if trajectory.get("tiny_vae_parallel", False):
        command.append("--tiny_vae_parallel")
    if trajectory.get("gradient_checkpointing_offload", False):
        command.append("--use_gradient_checkpointing_offload")
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
        Path(config["paths"]["output_root"])
        / config["experiment"]["name"]
        / run_tag
    )
    command = build_command(config, output_dir)
    effective_batch = (
        int(config["launch"]["num_processes"])
        * int(config["optimization"]["train_batch_size_per_gpu"])
        * int(config["optimization"]["gradient_accumulation_steps"])
    )
    summary = {
        "experiment": config["experiment"]["name"],
        "mode": config["adaptation"]["mode"],
        "enable_object_branch": config["adaptation"]["enable_object_branch"],
        "initialization": config["initialization"]["type"],
        "trajectory_loss_enabled": True,
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
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
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
            str(COTRACKER_ROOT),
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
