#!/usr/bin/env python3
"""Run the formal watcher evaluation for the PyBullet-only step-500 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

from xssc_lora_checkpoint_watch import (
    discover_checkpoints,
    exclusive_lock,
    load_json,
    load_manifests,
    log,
    manifest_path,
    metric_marker_path,
    refresh_site,
    run_inference_task,
    run_metric_task,
    state_paths,
    write_discovery,
)
from xssc_lora_physiciq_watch import (
    load_phys_manifests,
    phys_manifest_path,
    phys_metric_marker_path,
    phys_state_root,
    refresh_dashboard,
    run_phys_inference,
    run_metric as run_phys_metric,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "xssc_lora_three_train_watch_config_with_t_head.json"
METHOD_KEY = "full_sa_no_object_pybullet100"
STEP = 500
CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/"
    "full_sa_no_object_pybullet100_gpu67_1000steps/"
    "serial_20260804T115337Z/checkpoints/step-000500"
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("test5", "physiciq"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-used-mib", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def gpu_used_mib(gpu: int) -> int:
    process = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(process.stdout.strip().splitlines()[0])


def wait_for_gpu(gpu: int, max_used_mib: int, poll_seconds: int) -> None:
    while True:
        used = gpu_used_mib(gpu)
        if used <= max_used_mib:
            log(f"GPU{gpu} ready used={used} MiB")
            return
        log(f"GPU{gpu} used={used} MiB; waiting for <= {max_used_mib} MiB")
        time.sleep(poll_seconds)


def selected_checkpoint_task(config: dict) -> dict:
    matches = [
        task
        for task in discover_checkpoints(config)
        if task["method_key"] == METHOD_KEY
        and int(task["step"]) == STEP
        and Path(task["checkpoint_dir"]).resolve() == CHECKPOINT
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint task, found {len(matches)}")
    return matches[0]


def selected_manifest(config: dict, *, physiciq: bool) -> dict:
    manifests = load_phys_manifests(config) if physiciq else load_manifests(config)
    matches = [
        manifest
        for manifest in manifests
        if manifest["method_key"] == METHOD_KEY
        and int(manifest["step"]) == STEP
        and Path(manifest["checkpoint_dir"]).resolve() == CHECKPOINT
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {'PhysicIQ ' if physiciq else ''}manifest, found {len(matches)}")
    return matches[0]


def run_test5(config: dict, gpu: int, max_used_mib: int, poll_seconds: int) -> None:
    task = selected_checkpoint_task(config)
    manifest = manifest_path(config, METHOD_KEY, STEP)
    inference_lock = (
        state_paths(config)["state"]
        / "inference_locks"
        / METHOD_KEY
        / f"step-{STEP:06d}.lock"
    )
    with exclusive_lock(inference_lock):
        if not manifest.is_file():
            wait_for_gpu(gpu, max_used_mib, poll_seconds)
            run_inference_task(config, task, gpu)
        else:
            log(f"test_5 manifest already exists: {manifest}")
    refresh_site(config)

    result_manifest = selected_manifest(config, physiciq=False)
    for metric in config["metrics"]["cpu"]:
        marker = metric_marker_path(config, METHOD_KEY, STEP, metric)
        lock = (
            state_paths(config)["state"]
            / "metric_locks"
            / "cpu"
            / METHOD_KEY
            / f"step-{STEP:06d}"
            / f"{metric}.lock"
        )
        with exclusive_lock(lock):
            if not marker.is_file():
                run_metric_task(
                    config,
                    "cpu",
                    {"manifest": result_manifest, "metric": metric},
                )
                refresh_site(config)

    for metric in config["metrics"]["gpu"]:
        marker = metric_marker_path(config, METHOD_KEY, STEP, metric)
        lock = (
            state_paths(config)["state"]
            / "metric_locks"
            / "gpu"
            / METHOD_KEY
            / f"step-{STEP:06d}"
            / f"{metric}.lock"
        )
        with exclusive_lock(lock):
            if not marker.is_file():
                wait_for_gpu(gpu, max_used_mib, poll_seconds)
                run_metric_task(
                    config,
                    "gpu",
                    {"manifest": result_manifest, "metric": metric},
                    gpu,
                )
                refresh_site(config)
    log("test_5 inference and metrics complete")


def run_physiciq(config: dict, gpu: int, max_used_mib: int, poll_seconds: int) -> None:
    # PhysicIQ uses the same checkpoint directly and can run in parallel with test_5.
    task = selected_checkpoint_task(config)
    manifest = phys_manifest_path(config, METHOD_KEY, STEP)
    inference_lock = (
        phys_state_root(config)
        / "inference_locks"
        / METHOD_KEY
        / f"step-{STEP:06d}.lock"
    )
    with exclusive_lock(inference_lock):
        if not manifest.is_file():
            wait_for_gpu(gpu, max_used_mib, poll_seconds)
            run_phys_inference(config, task, gpu)
        else:
            log(f"PhysicIQ manifest already exists: {manifest}")
    refresh_dashboard(config)

    result_manifest = selected_manifest(config, physiciq=True)
    for kind in ("cpu", "gpu"):
        for metric in config["metrics"][kind]:
            marker = phys_metric_marker_path(config, METHOD_KEY, STEP, metric)
            lock = (
                phys_state_root(config)
                / "metric_locks"
                / kind
                / METHOD_KEY
                / f"step-{STEP:06d}"
                / f"{metric}.lock"
            )
            with exclusive_lock(lock):
                if marker.is_file():
                    continue
                if kind == "gpu":
                    wait_for_gpu(gpu, max_used_mib, poll_seconds)
                    try:
                        run_phys_metric(
                            config,
                            kind,
                            {"manifest": result_manifest, "metric": metric},
                            gpu,
                        )
                    except subprocess.CalledProcessError:
                        if not marker.is_file():
                            raise
                        log(f"metric committed despite plot refresh failure: {metric}")
                else:
                    try:
                        run_phys_metric(
                            config,
                            kind,
                            {"manifest": result_manifest, "metric": metric},
                        )
                    except subprocess.CalledProcessError:
                        if not marker.is_file():
                            raise
                        log(f"metric committed despite plot refresh failure: {metric}")
                refresh_dashboard(config)
    log("PhysicIQ inference and metrics complete")


def main() -> None:
    args = parse_args()
    if args.gpu not in (6, 7):
        raise SystemExit("This targeted runner is restricted to GPU6/7")
    config_path = args.config.resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    write_discovery(config, discover_checkpoints(config))
    if args.phase == "test5":
        run_test5(config, args.gpu, args.max_used_mib, args.poll_seconds)
    else:
        run_physiciq(config, args.gpu, args.max_used_mib, args.poll_seconds)


if __name__ == "__main__":
    main()
