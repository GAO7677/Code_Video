#!/usr/bin/env python3
"""Prepare and launch a config-only all-block/all-head sweep in tmux."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from configured_head_ablation import (
    METRIC_KINDS,
    config_fingerprint,
    generation_jobs,
    load_config,
    metric_count,
    metric_worker_count,
    output_base,
    read_unique_inputs,
    result_config_count,
    run_root,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
GEN_WORKER = SCRIPT_DIR / "run_configured_head_ablation_generation_worker.py"
COORDINATOR = SCRIPT_DIR / "run_configured_head_ablation_coordinator.py"
METRIC_WORKER = SCRIPT_DIR / "run_test5_ablation_metric_wait_worker.sh"


def _shell(command: list[str]) -> str:
    return shlex.join(command) + "; exec bash"


def _check_checkpoints(config: dict) -> None:
    checkpoints = config["checkpoints"]
    required = [
        Path(checkpoints["wan_root"]),
        Path(checkpoints["wan_lora_root"]) / "checkpoint.safetensors",
        Path(checkpoints["xssc_weights_root"]),
        Path(checkpoints["xssc_config"]),
        Path(checkpoints["xssc_checkpoint"]),
        Path(checkpoints["physrvg_root"]),
        Path(checkpoints["physrvg_model_id"]),
        Path(checkpoints["physrvg_dit_checkpoint"]),
        Path(checkpoints["physrvg_lora_checkpoint"])
        / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints/configs: {missing}")


def _prepare(config_path: Path, config: dict) -> None:
    root = run_root(config)
    generation = root / "generation"
    metrics = root / "metrics"
    for path in (
        generation / "logs",
        generation / "state",
        generation / "validations",
        metrics / "state",
    ):
        path.mkdir(parents=True, exist_ok=True)
    inputs = read_unique_inputs(config)
    (root / "input_unique.txt").write_text(
        "\n".join(str(path) for path in inputs) + "\n",
        encoding="utf-8",
    )
    jobs = generation_jobs(config)
    (generation / "queue.tsv").write_text(
        "\n".join(
            f"{task_id}\t{model}\t{block}\t{head}"
            for task_id, model, block, head in jobs
        )
        + "\n",
        encoding="utf-8",
    )
    (generation / "cursor").write_text("1\n", encoding="utf-8")
    (generation / "completed.tsv").write_text("", encoding="utf-8")
    (generation / "failed.tsv").write_text("", encoding="utf-8")
    for state in (generation / "state").glob("*"):
        state.unlink()
    for state in (metrics / "state").glob("*"):
        state.unlink()
    for marker in (
        "generation.failed",
        "metrics.ready",
        "metrics.failed",
        "pipeline.complete",
    ):
        (root / marker).unlink(missing_ok=True)

    fingerprint = config_fingerprint(config_path)
    snapshot = root / "config.snapshot.json"
    snapshot.write_text(
        json.dumps(
            json.loads(config_path.read_text(encoding="utf-8")),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "config_sha256": fingerprint,
                "num_configs": len(jobs),
                "num_cases_per_config": len(inputs),
                "num_videos": len(jobs) * len(inputs),
                "metrics_per_config": metric_count(config),
                "metric_tasks": len(jobs) * metric_count(config),
                "gpus": config["experiment"]["gpus"],
                "tmux_session": config["experiment"]["tmux_session"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _tmux(*args: str) -> None:
    subprocess.run(["tmux", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    _check_checkpoints(config)
    inputs = read_unique_inputs(config)
    configs = result_config_count(config)
    videos = configs * len(inputs)
    metric_tasks = configs * metric_count(config)
    free_gb = shutil.disk_usage(output_base(config).parent).free / 1024**3
    summary = {
        "session": config["experiment"]["tmux_session"],
        "output_base": str(output_base(config)),
        "run_root": str(run_root(config)),
        "gpus": config["experiment"]["gpus"],
        "unique_cases": len(inputs),
        "configs": configs,
        "videos": videos,
        "metrics_per_config": metric_count(config),
        "metric_tasks": metric_tasks,
        "generation_workers": len(config["experiment"]["gpus"])
        * int(config["experiment"]["generation_workers_per_gpu"]),
        "metric_workers": metric_worker_count(config),
        "free_disk_gb": round(free_gb, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    session = str(config["experiment"]["tmux_session"])
    existing = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    _prepare(config_path, config)
    runtime_config = run_root(config) / "config.snapshot.json"

    coordinator_command = [
        str(PYTHON),
        str(COORDINATOR),
        "--config",
        str(runtime_config),
    ]
    _tmux(
        "new-session",
        "-d",
        "-s",
        session,
        "-n",
        "coordinator",
        _shell(coordinator_command),
    )

    for gpu in config["experiment"]["gpus"]:
        for index in range(
            int(config["experiment"]["generation_workers_per_gpu"])
        ):
            name = f"gen_g{gpu}_{index}"
            command = [
                str(PYTHON),
                str(GEN_WORKER),
                "--config",
                str(runtime_config),
                "--gpu",
                str(gpu),
                "--worker-name",
                name,
            ]
            _tmux("new-window", "-t", session, "-n", name, _shell(command))

    ready = run_root(config) / "metrics.ready"
    for gpu in config["experiment"]["gpus"]:
        for kind in METRIC_KINDS:
            count = int(config["metrics"]["workers_per_gpu"][kind])
            for index in range(count):
                name = f"g{gpu}_{kind[:3]}{index}"
                command = [
                    "bash",
                    str(METRIC_WORKER),
                    str(gpu),
                    kind,
                    name,
                    str(run_root(config)),
                    str(run_root(config) / "input_unique.txt"),
                    str(ready),
                ]
                _tmux("new-window", "-t", session, "-n", name, _shell(command))

    _tmux("select-window", "-t", f"{session}:coordinator")
    print(f"tmux session started: {session}")


if __name__ == "__main__":
    main()
