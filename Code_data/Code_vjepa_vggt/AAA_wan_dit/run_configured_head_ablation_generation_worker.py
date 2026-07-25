#!/usr/bin/env python3
"""Queue worker for one configured block/head generation sweep."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

from configured_head_ablation import (
    ablation_tag,
    config_root,
    load_config,
    output_base,
    read_unique_inputs,
    run_root,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
VERIFY = SCRIPT_DIR / "verify_test5_ablation_outputs.py"


def _append_locked(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def _claim_task(queue: Path, cursor: Path, lock: Path) -> str | None:
    with lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        line_number = int(cursor.read_text(encoding="utf-8").strip())
        lines = queue.read_text(encoding="utf-8").splitlines()
        task = lines[line_number - 1] if line_number <= len(lines) else None
        if task is not None:
            cursor.write_text(f"{line_number + 1}\n", encoding="utf-8")
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
    return task


def _verification_command(
    *,
    config: dict,
    model: str,
    block: int,
    head: int,
    input_list: Path,
    validation: Path,
) -> list[str]:
    return [
        str(PYTHON),
        str(VERIFY),
        "--config-root",
        str(config_root(config, model, block, head)),
        "--input-list",
        str(input_list),
        "--model",
        model,
        "--mode",
        "self_attn_head_zero",
        "--block",
        str(block),
        "--head",
        str(head),
        "--expected-cases",
        str(config["input"]["expected_unique_cases"]),
        "--output",
        str(validation),
    ]


def _generation_command(
    config: dict, model: str, block: int, head: int, gpu: int
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    inference = config["inference"]
    checkpoints = config["checkpoints"]
    env.update(
        {
            "INPUT_LIST": str(run_root(config) / "input_unique.txt"),
            "HEIGHT": str(inference["height"]),
            "WIDTH": str(inference["width"]),
            "NUM_FRAMES": str(inference["num_frames"]),
            "CONTEXT_FRAMES": str(inference["context_frames"]),
            "NUM_INFERENCE_STEPS": str(inference["num_inference_steps"]),
            "CFG_SCALE": str(inference["cfg_scale"]),
            "GUIDANCE_SCALE": str(inference["guidance_scale"]),
            "DO_CFG": str(inference["physrvg_do_cfg"]),
            "FPS": str(inference["fps"]),
            "SEED": str(inference["seed"]),
            "NEGATIVE_PROMPT": str(inference["negative_prompt"]),
            "WAN_ROOT": str(checkpoints["wan_root"]),
            "WAN_LORA_ROOT": str(checkpoints["wan_lora_root"]),
            "XSSC_WEIGHTS_ROOT": str(checkpoints["xssc_weights_root"]),
            "XSSC_ROOT": str(checkpoints["xssc_root"]),
            "XSSC_CONFIG": str(checkpoints["xssc_config"]),
            "XSSC_CHECKPOINT": str(checkpoints["xssc_checkpoint"]),
            "PHYSRVG_ROOT": str(checkpoints["physrvg_root"]),
            "MODEL_ID": str(checkpoints["physrvg_model_id"]),
            "DIT_CHECKPOINT": str(checkpoints["physrvg_dit_checkpoint"]),
            "LORA_CHECKPOINT": str(checkpoints["physrvg_lora_checkpoint"]),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if model == "physrvg":
        env["OUTPUT_BASE"] = str(output_base(config) / "PhyRVG")
        command = [
            "bash",
            str(SCRIPT_DIR / "run_physrvg_physiciq_one.sh"),
            "self_attn_head_zero",
            str(block),
            str(gpu),
            str(head),
        ]
    else:
        env["OUTPUT_BASE"] = str(output_base(config))
        command = [
            "bash",
            str(SCRIPT_DIR / "run_physiciq_one.sh"),
            model,
            "self_attn_head_zero",
            str(block),
            str(gpu),
            str(head),
        ]
    return command, env


def _run_logged(
    command: list[str],
    *,
    env: dict[str, str] | None,
    log: TextIO,
) -> int:
    log.write(f"$ {' '.join(command)}\n")
    log.flush()
    completed = subprocess.run(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return int(completed.returncode)


def _clean_incomplete_config(
    config: dict, model: str, block: int, head: int, log: TextIO
) -> None:
    path = config_root(config, model, block, head)
    if not path.exists():
        return
    if not config["experiment"].get(
        "clean_incomplete_config_before_retry", True
    ):
        raise RuntimeError(
            f"incomplete config exists and cleanup is disabled: {path}"
        )
    expected_parent = (
        output_base(config) / ("PhyRVG" if model == "physrvg" else model)
    )
    if path.parent != expected_parent or path.name != ablation_tag(block, head):
        raise RuntimeError(f"refusing to clean unexpected config path: {path}")
    log.write(f"[worker] removing incomplete config before retry: {path}\n")
    log.flush()
    shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-name", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    root = run_root(config)
    expected_cases = len(read_unique_inputs(config))
    generation_root = root / "generation"
    queue = generation_root / "queue.tsv"
    cursor = generation_root / "cursor"
    lock = generation_root / "queue.lock"
    completed_path = generation_root / "completed.tsv"
    failed_path = generation_root / "failed.tsv"
    state_path = generation_root / "state" / f"{args.worker_name}.json"
    logs = generation_root / "logs"
    validations = generation_root / "validations"
    logs.mkdir(parents=True, exist_ok=True)
    validations.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    failed = 0
    skipped = 0
    started = time.time()
    worker_log_path = logs / f"{args.worker_name}.log"
    with worker_log_path.open("a", encoding="utf-8") as worker_log:
        worker_log.write(
            f"[worker] start name={args.worker_name} gpu={args.gpu} "
            f"expected_cases={expected_cases}\n"
        )
        while True:
            free_gb = shutil.disk_usage(output_base(config)).free / 1024**3
            minimum = float(config["experiment"]["minimum_free_disk_gb"])
            if free_gb < minimum:
                worker_log.write(
                    f"[worker] stopping: free_gb={free_gb:.1f} < {minimum:.1f}\n"
                )
                failed += 1
                break

            task = _claim_task(queue, cursor, lock)
            if task is None:
                break
            task_id, model, block_text, head_text = task.split("\t")
            block = int(block_text)
            head = int(head_text)
            validation = validations / f"{task_id}.json"
            task_log_path = logs / f"{task_id}.log"
            verify = _verification_command(
                config=config,
                model=model,
                block=block,
                head=head,
                input_list=root / "input_unique.txt",
                validation=validation,
            )
            with task_log_path.open("a", encoding="utf-8") as task_log:
                task_log.write(
                    f"task={task_id} model={model} block={block} head={head} "
                    f"tag={ablation_tag(block, head)} gpu={args.gpu}\n"
                )
                if (
                    config["experiment"].get("skip_verified_outputs", True)
                    and _run_logged(verify, env=None, log=task_log) == 0
                ):
                    status = 0
                    skipped += 1
                    task_log.write("[worker] existing output verified; skipped\n")
                else:
                    validation.unlink(missing_ok=True)
                    _clean_incomplete_config(
                        config, model, block, head, task_log
                    )
                    command, env = _generation_command(
                        config, model, block, head, args.gpu
                    )
                    retries = int(
                        config["experiment"].get("retry_failed_generation", 0)
                    )
                    status = 1
                    for attempt in range(retries + 1):
                        if attempt:
                            _clean_incomplete_config(
                                config, model, block, head, task_log
                            )
                        task_log.write(f"[worker] generation attempt={attempt + 1}\n")
                        status = _run_logged(command, env=env, log=task_log)
                        if status == 0:
                            status = _run_logged(verify, env=None, log=task_log)
                        if status == 0:
                            break
                    if status != 0:
                        validation.unlink(missing_ok=True)

            if status == 0:
                done += 1
                _append_locked(
                    completed_path,
                    f"{task_id}\t{model}\t{block}\t{head}\t{args.worker_name}",
                )
            else:
                failed += 1
                _append_locked(
                    failed_path,
                    f"{task_id}\t{model}\t{block}\t{head}\t"
                    f"{args.worker_name}\t{status}",
                )
            worker_log.write(
                f"[worker] task={task_id} status={status} done={done} "
                f"failed={failed} skipped={skipped}\n"
            )
            worker_log.flush()

    state_path.write_text(
        json.dumps(
            {
                "worker": args.worker_name,
                "gpu": args.gpu,
                "done": done,
                "failed": failed,
                "skipped": skipped,
                "elapsed_seconds": round(time.time() - started, 3),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
