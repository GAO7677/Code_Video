#!/usr/bin/env python3
"""Shard missing PhysicIQ cases for already-discovered xSSC LoRA checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from xssc_lora_checkpoint_watch import (
    atomic_write_json,
    checkpoint_complete,
    discover_checkpoints,
    load_json,
    log,
    read_inputs,
    timestamp,
    validate_result_root,
)
from xssc_lora_physiciq_watch import (
    append_leaf_folder,
    method_name,
    phys_manifest_path,
)


def parse_csv_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if 4 in items:
        raise argparse.ArgumentTypeError("GPU4 is forbidden by workspace rules")
    return items


def parse_csv_strings(value: str) -> set[str]:
    items = {item.strip() for item in value.split(",") if item.strip()}
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_csv_ints, required=True)
    parser.add_argument("--steps", type=parse_csv_ints, required=True)
    parser.add_argument("--methods", type=parse_csv_strings, required=True)
    parser.add_argument("--force-missing", action="store_true")
    return parser.parse_args()


def load_case_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def input_stem(path: Path) -> str:
    return path.resolve().stem


def result_complete(result_root: Path, stem: str) -> bool:
    return (
        (result_root / f"{stem}.mp4").is_file()
        and (result_root / f"{stem}.mp4").stat().st_size > 0
        and (result_root / f"{stem}.json").is_file()
        and (result_root / f"{stem}.json").stat().st_size > 0
    )


def selected_tasks(
    config: dict[str, Any],
    steps: set[int],
    methods: set[str],
) -> list[dict[str, Any]]:
    phys_methods = set(config["physiciq"]["method_keys"])
    tasks: list[dict[str, Any]] = []
    for task in discover_checkpoints(config):
        step = int(task["step"])
        method_key = str(task["method_key"])
        checkpoint_dir = Path(task["checkpoint_dir"]).resolve()
        if step not in steps or method_key not in methods or method_key not in phys_methods:
            continue
        if not checkpoint_complete(checkpoint_dir):
            continue
        tasks.append({**task, "checkpoint_dir": str(checkpoint_dir)})
    return sorted(tasks, key=lambda row: (int(row["step"]), str(row["method_key"])))


def split_round_robin(items: list[Path], parts: int) -> list[list[Path]]:
    shards = [[] for _ in range(parts)]
    for index, item in enumerate(items):
        shards[index % parts].append(item)
    return [shard for shard in shards if shard]


def run_shard(
    config: dict[str, Any],
    task: dict[str, Any],
    shard_inputs: list[Path],
    shard_index: int,
    gpu_id: int,
) -> None:
    phys = config["physiciq"]
    output_root = Path(phys["output_root"]).resolve()
    name = method_name(config, task)
    meta_root = output_root / "_run_meta" / name
    shard_root = meta_root / "shards"
    log_root = meta_root / "logs"
    trace_root = meta_root / "numeric_traces" / f"catchup_shard_{shard_index:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    shard_file = shard_root / f"catchup_shard_{shard_index:02d}.txt"
    shard_file.write_text(
        "".join(f"{path}\n" for path in shard_inputs),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "TEST_LIST": str(shard_file),
            "NUM_INFERENCE_STEPS": str(phys["num_inference_steps"]),
            "STEP_OUTPUT_DIR_NAME": name,
            "SHARD_TAG": f"catchup_shard_{shard_index:02d}",
            "TRACE_ROOT": str(trace_root),
        }
    )
    log_path = log_root / f"catchup_shard_{shard_index:02d}_gpu{gpu_id}.log"
    log(
        f"PhysicIQ catchup shard start method={task['method_key']} "
        f"step={task['step']} shard={shard_index} cases={len(shard_inputs)} gpu={gpu_id}"
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        subprocess.run(
            [
                "bash",
                config["paths"]["run_infer_script"],
                task["checkpoint_dir"],
                str(gpu_id),
                str(output_root),
            ],
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    log(
        f"PhysicIQ catchup shard complete method={task['method_key']} "
        f"step={task['step']} shard={shard_index}"
    )


def register_task(config: dict[str, Any], task: dict[str, Any]) -> None:
    output_root = Path(config["physiciq"]["output_root"]).resolve()
    input_list = Path(config["physiciq"]["input_list"]).resolve()
    result_root = output_root / method_name(config, task)
    validation = validate_result_root(
        config,
        result_root,
        input_list=input_list,
        expected_cases=int(config["physiciq"]["expected_cases"]),
    )
    payload = {
        "method_key": task["method_key"],
        "method_label": task["method_label"],
        "step": int(task["step"]),
        "checkpoint_dir": task["checkpoint_dir"],
        "result_root": str(result_root.resolve()),
        "input_list": str(input_list),
        "num_inference_steps": int(config["physiciq"]["num_inference_steps"]),
        "completed_utc": timestamp(),
        "validation": validation,
        "runner": "parallel_sharded_catchup",
    }
    atomic_write_json(
        phys_manifest_path(config, task["method_key"], int(task["step"])),
        payload,
    )
    atomic_write_json(
        output_root / "_run_meta" / method_name(config, task) / "batch_manifest.json",
        payload,
    )
    append_leaf_folder(config, result_root)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    input_paths = read_inputs(Path(config["physiciq"]["input_list"]).resolve())
    tasks = selected_tasks(config, set(args.steps), args.methods)
    if not tasks:
        log("PhysicIQ catchup: no matching checkpoint tasks")
        return

    for task in tasks:
        result_root = (
            Path(config["physiciq"]["output_root"]).resolve()
            / method_name(config, task)
        )
        missing = [
            path
            for path in input_paths
            if args.force_missing or not result_complete(result_root, input_stem(path))
        ]
        if not missing:
            register_task(config, task)
            log(
                f"PhysicIQ catchup registered complete result method={task['method_key']} "
                f"step={task['step']}"
            )
            continue
        shards = split_round_robin(missing, len(args.gpus))
        with ThreadPoolExecutor(max_workers=len(shards)) as executor:
            futures = [
                executor.submit(
                    run_shard,
                    config,
                    task,
                    shard,
                    shard_index,
                    args.gpus[shard_index % len(args.gpus)],
                )
                for shard_index, shard in enumerate(shards)
            ]
            for future in as_completed(futures):
                future.result()
        register_task(config, task)
        log(
            f"PhysicIQ catchup complete method={task['method_key']} "
            f"step={task['step']} total_cases={len(input_paths)}"
        )

    subprocess.run(
        [
            config["paths"]["python"],
            config["paths"]["dashboard_builder"],
            "--config",
            str(config_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
