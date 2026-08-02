#!/usr/bin/env python3
"""Stop No-Object after step 3000 and expand the unified evaluation pool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Any

import torch


DEFAULT_EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/"
    "full_sa_no_object_gpu67_resume_step6/resume_step000006_20260801T165700Z"
)
DEFAULT_SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_STATE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/"
    "automation/no_object_step3000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--target-step", type=int, default=3000)
    parser.add_argument(
        "--train-tmux-target",
        default="wan_train_full_sa_no_object_gpu27:fullsa_noobj_resume6_g67",
    )
    parser.add_argument(
        "--watch-config",
        type=Path,
        default=DEFAULT_SCRIPT_ROOT / "xssc_lora_three_train_watch_full_sa_no_object_gpu27.json",
    )
    parser.add_argument("--watch-session", default="wan_train_full_sa_no_object_gpu27")
    parser.add_argument("--eval-gpus", default="2,3,5,6,7")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--checkpoint-stability-seconds", type=int, default=30)
    parser.add_argument("--gpu-ready-max-used-mib", type=int, default=8000)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Invalid GPU list: {value}")
    if 4 in gpu_ids:
        raise ValueError("GPU4 is forbidden by workspace policy")
    return gpu_ids


def checkpoint_files(checkpoint_dir: Path) -> tuple[Path, Path]:
    return checkpoint_dir / "checkpoint.safetensors", checkpoint_dir / "training_state.pt"


def checkpoint_signature(checkpoint_dir: Path) -> tuple[tuple[int, int], tuple[int, int]] | None:
    files = checkpoint_files(checkpoint_dir)
    if not all(path.is_file() and path.stat().st_size > 0 for path in files):
        return None
    return tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in files)  # type: ignore[return-value]


def validate_training_state(checkpoint_dir: Path, expected_step: int) -> None:
    training_state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    actual_step = int(training_state.get("global_step", -1))
    if actual_step != expected_step:
        raise RuntimeError(
            f"Checkpoint directory says step {expected_step}, but training_state has {actual_step}"
        )


def latest_checkpoint_step(checkpoints_root: Path) -> int | None:
    steps = []
    for path in checkpoints_root.glob("step-*"):
        if path.is_dir() and path.name[5:].isdigit() and checkpoint_signature(path):
            steps.append(int(path.name[5:]))
    return max(steps) if steps else None


def matching_training_processes(experiment_root: Path) -> list[dict[str, Any]]:
    process = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    token = f"--output_path {experiment_root.resolve()}"
    matches: list[dict[str, Any]] = []
    for line in process.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or token not in fields[2]:
            continue
        if "train_xssc_object_self_attn_lora.py" not in fields[2]:
            continue
        matches.append({"pid": int(fields[0]), "ppid": int(fields[1]), "args": fields[2]})
    return matches


def tmux_target_exists(target: str) -> bool:
    return subprocess.run(
        ["tmux", "display-message", "-p", "-t", target, "#{pane_id}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def stop_training(target: str, experiment_root: Path, dry_run: bool) -> None:
    matches = matching_training_processes(experiment_root)
    if not matches:
        log("No matching No-Object training process remains; treating it as already stopped")
        return
    if not tmux_target_exists(target):
        raise RuntimeError(f"Training processes exist but tmux target is missing: {target}")
    log(f"Stopping {len(matches)} exact No-Object training processes via {target}")
    if dry_run:
        return
    subprocess.run(["tmux", "send-keys", "-t", target, "C-c"], check=True)
    deadline = time.time() + 180
    second_interrupt_sent = False
    while time.time() < deadline:
        remaining = matching_training_processes(experiment_root)
        if not remaining:
            log("No-Object training processes exited cleanly")
            return
        if not second_interrupt_sent and time.time() > deadline - 120:
            subprocess.run(["tmux", "send-keys", "-t", target, "C-c"], check=True)
            second_interrupt_sent = True
        time.sleep(5)
    remaining = matching_training_processes(experiment_root)
    log(f"Graceful stop timed out; sending SIGTERM to exact PIDs {[row['pid'] for row in remaining]}")
    for row in sorted(remaining, key=lambda item: item["pid"], reverse=True):
        try:
            os.kill(int(row["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(10)
    if matching_training_processes(experiment_root):
        raise RuntimeError("No-Object training did not stop after SIGTERM")


def gpu_memory_used(gpu_id: int) -> int:
    process = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(process.stdout.strip())


def wait_for_released_gpus(gpu_ids: list[int], threshold: int, poll_seconds: int) -> None:
    release_targets = [gpu_id for gpu_id in gpu_ids if gpu_id in {6, 7}]
    while release_targets:
        usage = {gpu_id: gpu_memory_used(gpu_id) for gpu_id in release_targets}
        if all(value <= threshold for value in usage.values()):
            log(f"Released GPU check passed: {usage}")
            return
        log(f"Waiting for No-Object GPUs to release: {usage}")
        time.sleep(poll_seconds)


def write_eval_config(source: Path, destination: Path, gpu_ids: list[int], threshold: int) -> None:
    config = json.loads(source.read_text(encoding="utf-8"))
    config["runtime"]["gpu_ids"] = gpu_ids
    config["runtime"]["gpu_ready_max_used_mib"] = threshold
    config["automation"] = {
        "created_utc": timestamp(),
        "purpose": "Drain all checkpoint inference and metrics after No-Object step 3000",
        "source_config": str(source.resolve()),
        "excluded_gpu_ids": [0, 1, 4],
    }
    atomic_write_json(destination, config)


def tmux_window_exists(session: str, window: str) -> bool:
    process = subprocess.run(
        ["tmux", "list-windows", "-t", session, "-F", "#{window_name}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return window in process.stdout.splitlines()


def launch_eval_workers(session: str, config: Path, dry_run: bool) -> list[str]:
    python = "/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
    root = Path(__file__).resolve().parent
    checkpoint_watcher = root / "xssc_lora_checkpoint_watch.py"
    phys_watcher = root / "xssc_lora_physiciq_watch.py"
    refresher = root / "xssc_lora_dashboard_refresh_loop.py"
    worker_specs: list[tuple[str, list[str]]] = []
    for index in range(2):
        worker_specs.append(
            (f"drain3k_t5inf{index + 1}", [python, str(checkpoint_watcher), "--config", str(config), "--mode", "inference"])
        )
        worker_specs.append(
            (f"drain3k_t5gpu{index + 1}", [python, str(checkpoint_watcher), "--config", str(config), "--mode", "metrics", "--kind", "gpu"])
        )
        worker_specs.append(
            (f"drain3k_phyinf{index + 1}", [python, str(phys_watcher), "--config", str(config), "--mode", "inference"])
        )
        worker_specs.append(
            (f"drain3k_phygpu{index + 1}", [python, str(phys_watcher), "--config", str(config), "--mode", "metrics", "--kind", "gpu"])
        )
    for index in range(4):
        worker_specs.append(
            (f"drain3k_t5cpu{index + 1}", [python, str(checkpoint_watcher), "--config", str(config), "--mode", "metrics", "--kind", "cpu"])
        )
        worker_specs.append(
            (f"drain3k_phycpu{index + 1}", [python, str(phys_watcher), "--config", str(config), "--mode", "metrics", "--kind", "cpu"])
        )
    worker_specs.append(
        ("drain3k_refresh", [python, str(refresher), "--config", str(config), "--interval", "60"])
    )
    launched: list[str] = []
    for window, command in worker_specs:
        if tmux_window_exists(session, window):
            log(f"tmux worker already exists: {session}:{window}")
            continue
        shell_command = "env PYTHONNOUSERSITE=1 " + shlex.join(command) + "; exec bash"
        log(f"Launching {session}:{window}")
        if not dry_run:
            subprocess.run(
                ["tmux", "new-window", "-t", session, "-n", window, shell_command],
                check=True,
            )
        launched.append(window)
    return launched


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for path in root.glob(pattern) if path.is_file())


def progress_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    watch_root = Path(config["paths"]["watch_root"])
    methods = [method["key"] for method in config["methods"]]
    phys_root = watch_root / "state" / "physiciq"
    return {
        "updated_utc": timestamp(),
        "methods": methods,
        "test5": {
            "completed_inference": count_files(watch_root / "state" / "checkpoints", "*/step-*.json"),
            "completed_metrics": count_files(watch_root / "state" / "metrics", "*/step-*/*.json"),
        },
        "physiciq": {
            "completed_inference": count_files(phys_root / "inference", "*/step-*.json"),
            "completed_metrics": count_files(phys_root / "metrics", "*/step-*/*.json"),
        },
    }


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    checkpoint_dir = experiment_root / "checkpoints" / f"step-{args.target_step:06d}"
    config_source = args.watch_config.resolve()
    gpu_ids = parse_gpu_ids(args.eval_gpus)
    state_root = args.state_root.resolve()
    status_path = state_root / "status.json"
    eval_config = state_root / "resolved_eval_drain_config.json"
    state_root.mkdir(parents=True, exist_ok=True)

    if not experiment_root.is_dir():
        raise FileNotFoundError(experiment_root)
    if not config_source.is_file():
        raise FileNotFoundError(config_source)
    watch_config = json.loads(config_source.read_text(encoding="utf-8"))
    method_keys = [method["key"] for method in watch_config["methods"]]
    if len(method_keys) != len(set(method_keys)):
        raise ValueError(f"Duplicate watcher method keys: {method_keys}")
    if not tmux_target_exists(args.train_tmux_target):
        raise RuntimeError(f"Missing training tmux target: {args.train_tmux_target}")
    process_matches = matching_training_processes(experiment_root)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "experiment_root": str(experiment_root),
                    "target_checkpoint": str(checkpoint_dir),
                    "latest_complete_step": latest_checkpoint_step(experiment_root / "checkpoints"),
                    "matching_training_pids": [row["pid"] for row in process_matches],
                    "train_tmux_target": args.train_tmux_target,
                    "watch_methods": method_keys,
                    "eval_gpu_ids": gpu_ids,
                    "excluded_gpu_ids": [0, 1, 4],
                    "valid": bool(process_matches),
                },
                indent=2,
            )
        )
        if not process_matches:
            raise RuntimeError("No exact No-Object training process matched")
        return 0

    base_status: dict[str, Any] = {
        "experiment_root": str(experiment_root),
        "target_step": args.target_step,
        "target_checkpoint": str(checkpoint_dir),
        "train_tmux_target": args.train_tmux_target,
        "eval_gpu_ids": gpu_ids,
        "watch_session": args.watch_session,
        "dry_run": args.dry_run,
    }
    log(f"Waiting for complete checkpoint: {checkpoint_dir}")
    stable_since: float | None = None
    previous_signature = None
    while True:
        signature = checkpoint_signature(checkpoint_dir)
        latest = latest_checkpoint_step(experiment_root / "checkpoints")
        if signature is not None and signature == previous_signature:
            stable_since = stable_since or time.time()
        else:
            stable_since = None
        previous_signature = signature
        stable_for = 0 if stable_since is None else int(time.time() - stable_since)
        atomic_write_json(
            status_path,
            {
                **base_status,
                "phase": "waiting_for_step3000",
                "updated_utc": timestamp(),
                "latest_complete_step": latest,
                "target_files_complete": signature is not None,
                "target_stable_seconds": stable_for,
            },
        )
        if signature is not None and stable_for >= args.checkpoint_stability_seconds:
            break
        time.sleep(max(5, args.poll_seconds))

    log(f"Checkpoint step {args.target_step} is complete and stable")
    validate_training_state(checkpoint_dir, args.target_step)
    atomic_write_json(status_path, {**base_status, "phase": "stopping_training", "updated_utc": timestamp()})
    stop_training(args.train_tmux_target, experiment_root, args.dry_run)
    if not args.dry_run:
        wait_for_released_gpus(gpu_ids, args.gpu_ready_max_used_mib, args.poll_seconds)

    write_eval_config(config_source, eval_config, gpu_ids, args.gpu_ready_max_used_mib)
    launched = launch_eval_workers(args.watch_session, eval_config, args.dry_run)
    config = json.loads(eval_config.read_text(encoding="utf-8"))
    atomic_write_json(
        status_path,
        {
            **base_status,
            "phase": "evaluation_workers_running" if not args.dry_run else "dry_run_complete",
            "updated_utc": timestamp(),
            "eval_config": str(eval_config),
            "launched_windows": launched,
            "progress": progress_snapshot(config),
        },
    )
    if args.dry_run:
        return 0

    log("Expanded inference/metric workers are running; monitoring progress")
    while True:
        atomic_write_json(
            status_path,
            {
                **base_status,
                "phase": "evaluation_workers_running",
                "updated_utc": timestamp(),
                "eval_config": str(eval_config),
                "launched_windows": launched,
                "progress": progress_snapshot(config),
            },
        )
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
