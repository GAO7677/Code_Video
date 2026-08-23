#!/usr/bin/env python3
"""Serially fill the missing Utonia no-scene test_5 and PhysicIQ runs.

The existing watcher/metric processes are deliberately left in place.  This
coordinator only submits one missing checkpoint at a time, which avoids
loading several copies of the 5B pipeline while the data volume is busy.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "xssc_lora_three_train_watch_config_with_t_head.json"
METHOD = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_b2gacc2"
CHECKPOINT_WATCH = ROOT / "run_missing_checkpoint_generation.py"
PHYSICIQ_RUNNER = ROOT / "xssc_lora_physiciq_parallel_infer.py"
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")

CACHE_ROOT = Path("/dev/shm/physrvg_utonia_eval_cache")
READY = CACHE_ROOT / "READY"
MODEL_ID = CACHE_ROOT / "Wan-AI-Wan2.2-TI2V-5B-Diffusers"
DIT = CACHE_ROOT / "physrvg-dit" / "diffusion_pytorch_model.safetensors"

WATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_object_self_attn_lora_three_run_watch"
)
TEST5_STATE = WATCH_ROOT / "state" / "checkpoints" / METHOD
PHYSICIQ_STATE = WATCH_ROOT / "state" / "physiciq" / "inference" / METHOD
GPU_LOCK_ROOT = WATCH_ROOT / "state" / "gpu_locks"
LOG_PATH = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_utonia_no_scene_test70/"
    "logs/test5_physiciq_serial/coordinator.log"
)

# GPU4 is intentionally absent.  The list is also used by the existing
# watcher so its inter-process GPU locks remain authoritative.
GPUS = (0, 1, 2, 3, 5, 6, 7)
EXPECTED_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["_config_path"] = str(CONFIG_PATH)
    return config


def wait_for_cache() -> None:
    while not (
        READY.is_file()
        and MODEL_ID.is_dir()
        and DIT.is_file()
        and DIT.stat().st_size > 0
    ):
        log("waiting for /dev/shm model cache to finish: " + str(CACHE_ROOT))
        time.sleep(30)
    log(f"model cache ready base={MODEL_ID} dit={DIT}")


def checkpoint_tasks(config: dict) -> dict[int, dict]:
    # Import only after the configured project root is available.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import xssc_lora_checkpoint_watch as watcher

    tasks = {}
    method_cfg = watcher.method_config(config, METHOD)
    for task in watcher.discover_checkpoints(config):
        if task["method_key"] != METHOD:
            continue
        step = int(task["step"])
        if step not in EXPECTED_STEPS:
            continue
        if watcher.checkpoint_complete(Path(task["checkpoint_dir"]), method_cfg):
            tasks[step] = task
    return tasks


def gpu_memory_used(gpu: int) -> int:
    result = subprocess.run(
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
    return int(result.stdout.strip().splitlines()[0])


def gpu_lock_available(gpu: int) -> bool:
    GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    path = GPU_LOCK_ROOT / f"gpu-{gpu}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return False
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return True


def choose_gpu() -> int:
    # Keep the selection conservative: metric watchers/training may change
    # usage between the probe and the child acquiring its official lock.
    while True:
        candidates = []
        for gpu in GPUS:
            try:
                used = gpu_memory_used(gpu)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                log(f"cannot query GPU{gpu}: {exc}")
                continue
            if used <= 12000 and gpu_lock_available(gpu):
                candidates.append((used, gpu))
        if candidates:
            used, gpu = min(candidates)
            log(f"selected GPU{gpu} (memory.used={used} MiB)")
            return gpu
        usage = ", ".join(
            f"GPU{gpu}={gpu_memory_used(gpu)}MiB" for gpu in GPUS
        )
        log(f"all permitted GPUs busy; waiting ({usage})")
        time.sleep(60)


def child_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT),
            "MODEL_ID": str(MODEL_ID),
            "PHYSRVG_DIT_CHECKPOINT": str(DIT),
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def run_child(command: list[str], label: str) -> None:
    log("start " + label + ": " + " ".join(command))
    result = subprocess.run(command, env=child_env())
    if result.returncode:
        raise RuntimeError(f"{label} exited with code {result.returncode}")
    log("finished " + label)


def test5_manifest(step: int) -> Path:
    return TEST5_STATE / f"step-{step:06d}.json"


def physiciq_manifest(step: int) -> Path:
    return PHYSICIQ_STATE / f"step-{step:06d}.json"


def run_test5(config: dict, tasks: dict[int, dict]) -> None:
    for step in EXPECTED_STEPS:
        manifest = test5_manifest(step)
        if manifest.is_file():
            log(f"test_5 already complete step={step}; skip")
            continue
        while step not in tasks:
            log(f"checkpoint step={step} is not complete yet; rediscovering")
            time.sleep(60)
            tasks = checkpoint_tasks(config)
        gpu = choose_gpu()
        command = [
            str(PYTHON),
            "-u",
            str(CHECKPOINT_WATCH),
            "--config",
            str(CONFIG_PATH),
            "--gpus",
            str(gpu),
            "--methods",
            METHOD,
            "--steps",
            str(step),
            "--test5-only",
        ]
        try:
            run_child(command, f"test_5 step={step} gpu={gpu}")
        except Exception as exc:
            log(f"test_5 step={step} failed: {exc}; retrying after 60s")
            time.sleep(60)
            continue
        if not manifest.is_file():
            log(f"test_5 step={step} returned without manifest; retrying")
            time.sleep(60)
            continue
        log(f"test_5 manifest registered step={step}: {manifest}")


def run_physiciq(config: dict, tasks: dict[int, dict]) -> None:
    for step in EXPECTED_STEPS:
        manifest = physiciq_manifest(step)
        if manifest.is_file():
            log(f"PhysicIQ already complete step={step}; skip")
            continue
        while step not in tasks:
            log(f"checkpoint step={step} is not complete yet; rediscovering")
            time.sleep(60)
            tasks = checkpoint_tasks(config)
        gpu = choose_gpu()
        command = [
            str(PYTHON),
            "-u",
            str(PHYSICIQ_RUNNER),
            "--config",
            str(CONFIG_PATH),
            "--gpus",
            str(gpu),
            "--methods",
            METHOD,
            "--steps",
            str(step),
            "--poll-seconds",
            "60",
            "--adopt-existing",
        ]
        try:
            run_child(command, f"PhysicIQ step={step} gpu={gpu}")
        except Exception as exc:
            log(f"PhysicIQ step={step} failed: {exc}; retrying after 60s")
            time.sleep(60)
            continue
        if not manifest.is_file():
            log(f"PhysicIQ step={step} returned without manifest; retrying")
            time.sleep(60)
            continue
        log(f"PhysicIQ manifest registered step={step}: {manifest}")


def main() -> None:
    config = load_config()
    log("coordinator started; scope=test_5 + PhysicIQ only")
    wait_for_cache()
    tasks = checkpoint_tasks(config)
    log(
        "complete checkpoints discovered: "
        + ",".join(str(step) for step in sorted(tasks))
    )
    run_test5(config, tasks)
    run_physiciq(config, tasks)
    tasks = checkpoint_tasks(config)
    marker = WATCH_ROOT / "state" / "utonia_test5_physiciq_generation_complete.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "method_key": METHOD,
                "test5_steps": [
                    step for step in EXPECTED_STEPS if test5_manifest(step).is_file()
                ],
                "physiciq_steps": [
                    step
                    for step in EXPECTED_STEPS
                    if physiciq_manifest(step).is_file()
                ],
                "checkpoint_steps": sorted(tasks),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"all requested generation gaps filled; marker={marker}")


if __name__ == "__main__":
    main()
