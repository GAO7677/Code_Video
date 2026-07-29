#!/usr/bin/env python3
"""Run and verify the largest S-depth ablation on one case per model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
RUNNER = SCRIPT_DIR / "run_matched_head_subset_ablation_job.sh"
VERIFY = SCRIPT_DIR / "verify_head_role_dose_control_preflight.py"
PREFLIGHT_INPUT = SCRIPT_DIR / "common22_public_head_ablation_preflight_case.txt"
SUBSET_ID = "S_depth_late_B20_29_all"
MODEL_GPUS = {
    "wan_lora": 0,
    "xssc": 1,
    "physrvg": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _gpu_memory_used(gpu: int) -> int:
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


def _wait_for_file(path: Path, poll_seconds: int) -> None:
    while not path.is_file():
        print(f"[s-depth-preflight] waiting for {path}", flush=True)
        time.sleep(poll_seconds)


def _wait_for_gpus(threshold: int, poll_seconds: int) -> None:
    while True:
        used = {gpu: _gpu_memory_used(gpu) for gpu in MODEL_GPUS.values()}
        busy = {gpu: value for gpu, value in used.items() if value > threshold}
        if not busy:
            return
        print(f"[s-depth-preflight] GPUs busy: {busy}", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    manifest = Path(config["matched_subset_manifest"]).expanduser().resolve()
    execution = config["execution"]
    start_after = Path(execution["start_after_file"]).expanduser().resolve()
    complete = Path(execution["preflight_complete_file"]).expanduser().resolve()
    failed = complete.with_name("preflight.failed")
    preflight_root = root / "preflight"
    verification = preflight_root / "preflight_verification.json"
    poll_seconds = int(execution["poll_seconds"])
    threshold = int(execution["gpu_start_memory_threshold_mib"])

    complete.unlink(missing_ok=True)
    failed.unlink(missing_ok=True)
    _wait_for_file(start_after, poll_seconds)
    _wait_for_gpus(threshold, poll_seconds)

    processes: list[tuple[str, subprocess.Popen[bytes], Any]] = []
    try:
        for model, gpu in MODEL_GPUS.items():
            log_path = preflight_root / "logs" / f"{model}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("ab")
            env = os.environ.copy()
            env.update(
                {
                    "MODEL": model,
                    "SEED": "851",
                    "SUBSET_ID": SUBSET_ID,
                    "GPU": str(gpu),
                    "STEP_START": "0",
                    "STEP_END": "10",
                    "INPUT_LIST": str(PREFLIGHT_INPUT),
                    "OUTPUT_ROOT": str(preflight_root),
                    "MANIFEST": str(manifest),
                }
            )
            process = subprocess.Popen(
                ["bash", str(RUNNER)],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((model, process, handle))
            print(
                f"[s-depth-preflight] started {model} on GPU{gpu}, pid={process.pid}",
                flush=True,
            )

        return_codes = {}
        for model, process, handle in processes:
            return_codes[model] = process.wait()
            handle.close()
        failures = {
            model: code for model, code in return_codes.items() if code != 0
        }
        if failures:
            raise RuntimeError(f"preflight inference failed: {failures}")

        subprocess.run(
            [
                str(PYTHON),
                str(VERIFY),
                "--root",
                str(preflight_root),
                "--manifest",
                str(manifest),
                "--input-list",
                str(PREFLIGHT_INPUT),
                "--subset-id",
                SUBSET_ID,
                "--seed",
                "851",
                "--step-start",
                "0",
                "--step-end",
                "10",
                "--output",
                str(verification),
            ],
            check=True,
        )
        complete.parent.mkdir(parents=True, exist_ok=True)
        complete.touch()
        print(f"[s-depth-preflight] passed: {verification}", flush=True)
    except Exception as error:
        for _, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            if not handle.closed:
                handle.close()
        _atomic_json(
            preflight_root / "preflight_failure.json",
            {
                "status": "failed",
                "error": repr(error),
                "failed_at_unix": time.time(),
            },
        )
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.touch()
        raise


if __name__ == "__main__":
    main()
