#!/usr/bin/env python3
"""Run step-triggered PhysicIQ inference and metrics for xSSC LoRA models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from xssc_lora_checkpoint_watch import (
    atomic_write_json,
    candidate_gpu_ids,
    exclusive_lock,
    gpu_metric_can_share,
    gpu_metric_worker_count,
    try_exclusive_lock,
    load_json,
    log,
    method_config,
    read_inputs,
    reserve_metric_gpu,
    state_paths,
    timestamp,
    validate_result_root,
    wait_for_gpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["discover", "inference", "metrics"], required=True)
    parser.add_argument("--kind", choices=["cpu", "gpu"])
    parser.add_argument("--methods", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--gpu-ready-max-used-mib", type=int, default=None)
    parser.add_argument("--gpu-metric-workers-per-gpu", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def phys_state_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["watch_root"]).resolve() / "state" / "physiciq"


def phys_manifest_path(config: dict[str, Any], method_key: str, step: int) -> Path:
    return phys_state_root(config) / "inference" / method_key / f"step-{step:06d}.json"


def phys_metric_marker_path(
    config: dict[str, Any],
    method_key: str,
    step: int,
    metric: str,
) -> Path:
    return (
        phys_state_root(config)
        / "metrics"
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.json"
    )


def main_checkpoint_manifests(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = state_paths(config)["checkpoints"]
    manifests = [
        load_json(path)
        for path in sorted(root.glob("*/step-*.json"))
        if path.is_file()
    ]
    return manifests


def discover_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    phys = config["physiciq"]
    configured_steps = phys.get("trigger_steps", "all")
    trigger_steps = (
        None
        if configured_steps == "all"
        else {int(step) for step in configured_steps}
    )
    method_keys = set(phys["method_keys"])
    tasks = [
        manifest
        for manifest in main_checkpoint_manifests(config)
        if manifest["method_key"] in method_keys
        and (trigger_steps is None or int(manifest["step"]) in trigger_steps)
    ]
    return sorted(tasks, key=lambda row: (int(row["step"]), row["method_key"]))


def load_phys_manifests(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = phys_state_root(config) / "inference"
    method_keys = set(config["physiciq"]["method_keys"])
    manifests = [
        load_json(path)
        for path in sorted(root.glob("*/step-*.json"))
        if path.is_file()
    ]
    manifests = [
        manifest for manifest in manifests if manifest["method_key"] in method_keys
    ]
    return sorted(manifests, key=lambda row: (int(row["step"]), row["method_key"]))


def append_leaf_folder(config: dict[str, Any], result_root: Path) -> None:
    leaf_values = [config["physiciq"]["leaf_folders"]]
    leaf_values.extend(config["physiciq"].get("additional_leaf_folders", []))
    for leaf_value in leaf_values:
        leaf_path = Path(leaf_value).resolve()
        leaf_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = leaf_path.with_suffix(leaf_path.suffix + ".lock")
        with exclusive_lock(lock_path):
            lines = []
            if leaf_path.is_file():
                lines = [
                    line.strip()
                    for line in leaf_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            normalized = str(result_root.resolve())
            if normalized not in lines:
                lines.append(normalized)
            temporary = leaf_path.with_name(f".{leaf_path.name}.tmp.{os.getpid()}")
            temporary.write_text(
                "".join(f"{line}\n" for line in lines),
                encoding="utf-8",
            )
            os.replace(temporary, leaf_path)


def method_name(config: dict[str, Any], task: dict[str, Any]) -> str:
    return config["physiciq"]["method_name_template"].format(
        method_key=task["method_key"],
        step=int(task["step"]),
    )


def run_phys_inference(
    config: dict[str, Any],
    task: dict[str, Any],
    gpu_id: int,
) -> None:
    phys = config["physiciq"]
    runtime = config["runtime"]
    method = method_config(config, task["method_key"])
    output_root = Path(phys["output_root"]).resolve()
    name = method_name(config, task)
    result_root = output_root / name
    meta_root = output_root / "_run_meta" / name
    shard_root = meta_root / "shards"
    log_root = meta_root / "logs"
    trace_root = meta_root / "numeric_traces" / "shard_00"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    input_list = Path(phys["input_list"]).resolve()
    shard_file = shard_root / "shard_00.txt"
    shard_file.write_text(input_list.read_text(encoding="utf-8"), encoding="utf-8")
    log_path = log_root / "shard_00.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "TEST_LIST": str(shard_file),
            "NUM_INFERENCE_STEPS": str(phys["num_inference_steps"]),
            "STEP_OUTPUT_DIR_NAME": name,
            "SHARD_TAG": "shard_00",
            "TRACE_ROOT": str(trace_root),
        }
    )
    log(
        f"PhysicIQ inference start method={task['method_key']} "
        f"step={task['step']} gpu={gpu_id}"
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        subprocess.run(
            [
                "bash",
                method.get("run_infer_script", config["paths"]["run_infer_script"]),
                task["checkpoint_dir"],
                str(gpu_id),
                str(output_root),
            ],
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    validation = validate_result_root(
        config,
        result_root,
        input_list=input_list,
        expected_cases=int(phys["expected_cases"]),
    )
    payload = {
        "method_key": task["method_key"],
        "method_label": task["method_label"],
        "step": int(task["step"]),
        "checkpoint_dir": task["checkpoint_dir"],
        "result_root": str(result_root),
        "input_list": str(input_list),
        "num_inference_steps": int(phys["num_inference_steps"]),
        "gpu_id": gpu_id,
        "completed_utc": timestamp(),
        "validation": validation,
    }
    atomic_write_json(
        phys_manifest_path(config, task["method_key"], int(task["step"])),
        payload,
    )
    atomic_write_json(meta_root / "batch_manifest.json", payload)
    append_leaf_folder(config, result_root)
    log(
        f"PhysicIQ inference complete method={task['method_key']} "
        f"step={task['step']} cases={validation['num_cases']}"
    )


def inference_loop(config: dict[str, Any], once: bool) -> None:
    paths = state_paths(config)
    phys_pending = phys_state_root(config) / "inference.pending"
    while True:
        tasks = discover_tasks(config)
        pending = [
            task
            for task in tasks
            if not phys_manifest_path(
                config, task["method_key"], int(task["step"])
            ).is_file()
        ]
        if not pending:
            phys_pending.unlink(missing_ok=True)
            if once:
                return
            time.sleep(int(config["runtime"]["poll_seconds"]))
            continue
        atomic_write_json(
            phys_pending,
            {
                "updated_utc": timestamp(),
                "num_pending": len(pending),
                "next": pending[0],
            },
        )
        handled = False
        for task in pending:
            task_lock = (
                phys_state_root(config)
                / "inference_locks"
                / task["method_key"]
                / f"step-{int(task['step']):06d}.lock"
            )
            with try_exclusive_lock(task_lock) as acquired:
                if not acquired:
                    continue
                if phys_manifest_path(
                    config, task["method_key"], int(task["step"])
                ).is_file():
                    continue
                handled = True
                try:
                    with reserve_available_gpu(config) as gpu_id:
                        run_phys_inference(config, task, gpu_id)
                    subprocess.run(
                        [
                            config["paths"]["python"],
                            config["paths"]["dashboard_builder"],
                            "--config",
                            config["_config_path"],
                        ],
                        check=True,
                    )
                except Exception as exc:
                    log(
                        f"PhysicIQ inference failed method={task['method_key']} "
                        f"step={task['step']}: {exc}"
                    )
                    time.sleep(int(config["runtime"]["retry_seconds"]))
                break
        if not handled:
            time.sleep(int(config["runtime"]["gpu_poll_seconds"]))
        if once:
            return


def metric_tasks(config: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for manifest in load_phys_manifests(config):
        for metric in config["metrics"][kind]:
            if phys_metric_marker_path(
                config,
                manifest["method_key"],
                int(manifest["step"]),
                metric,
            ).is_file():
                continue
            tasks.append({"manifest": manifest, "metric": metric})
    return tasks


def run_metric(
    config: dict[str, Any],
    kind: str,
    task: dict[str, Any],
    gpu_id: int | None = None,
) -> None:
    manifest = task["manifest"]
    metric = task["metric"]
    method_key = manifest["method_key"]
    step = int(manifest["step"])
    root = Path(config["paths"]["watch_root"]).resolve()
    summary_path = (
        root
        / "physiciq_metric_task_summaries"
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.json"
    )
    log_path = (
        root
        / "logs"
        / "physiciq_metrics"
        / kind
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.log"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config["paths"]["python"],
        config["paths"]["bench_script"],
        "--metric",
        metric,
        "--result-root",
        manifest["result_root"],
        "--input-json-allowlist",
        config["physiciq"]["input_list"],
        "--output-summary",
        str(summary_path),
    ]
    if metric == "wmreward":
        command.extend(["--wmreward-reset-interval", "1000000"])
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = (
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
        "/home/gaoya/Code_Video/Code_data/Code_try0526"
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if kind == "gpu" and gpu_id is None:
        raise ValueError("gpu_id is required for a GPU metric task")
    environment["CUDA_VISIBLE_DEVICES"] = "" if kind == "cpu" else str(gpu_id)
    gpu_text = "cpu" if gpu_id is None else f"gpu={gpu_id}"
    log(
        f"PhysicIQ metric start kind={kind} {gpu_text} method={method_key} "
        f"step={step} metric={metric}"
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        subprocess.run(
            command,
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            [
                config["paths"]["python"],
                config["paths"]["metric_validator_script"],
                str(summary_path),
                "--expected-cases",
                str(config["physiciq"]["expected_cases"]),
            ],
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    atomic_write_json(
        phys_metric_marker_path(config, method_key, step, metric),
        {
            "completed_utc": timestamp(),
            "kind": kind,
            "method_key": method_key,
            "step": step,
            "metric": metric,
            "result_root": manifest["result_root"],
            "summary_path": str(summary_path),
            "gpu_id": gpu_id,
        },
    )
    log(f"PhysicIQ metric complete method={method_key} step={step} metric={metric}")
    refresh_plots_if_complete(config, manifest)


def run_gpu_metric_concurrent(
    config: dict[str, Any], task: dict[str, Any]
) -> bool:
    """Run one PhysicIQ GPU metric while preserving watcher task locking."""
    manifest = task["manifest"]
    method_key = manifest["method_key"]
    step = int(manifest["step"])
    metric = task["metric"]
    task_lock = (
        phys_state_root(config)
        / "metric_locks"
        / "gpu"
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.lock"
    )
    with try_exclusive_lock(task_lock) as acquired:
        if not acquired or phys_metric_marker_path(config, method_key, step, metric).is_file():
            return False
        with reserve_metric_gpu(
            config,
            allow_parallel=gpu_metric_can_share(config, metric),
        ) as gpu_id:
            run_metric(config, "gpu", task, gpu_id)
        return True


def refresh_plots(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    plot_log = (
        Path(config["paths"]["watch_root"]).resolve()
        / "logs"
        / "physiciq_plot_refresh.log"
    )
    leaf_values = [config["physiciq"]["leaf_folders"]]
    leaf_values.extend(config["physiciq"].get("additional_leaf_folders", []))
    plot_failures = []
    with exclusive_lock(phys_state_root(config) / "plot.lock"):
        with plot_log.open("a", encoding="utf-8") as log_handle:
            for leaf_value in leaf_values:
                leaf_path = Path(leaf_value).resolve()
                if not leaf_path.is_file():
                    continue
                try:
                    subprocess.run(
                        [
                            "bash",
                            config["physiciq"]["plot_script"],
                            str(leaf_path),
                        ],
                        check=True,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                except subprocess.CalledProcessError as exc:
                    plot_failures.append(f"{leaf_path}: exit {exc.returncode}")
    refresh_dashboard(config)
    if plot_failures:
        log("Legacy PhysicIQ plot refresh skipped failures: " + "; ".join(plot_failures))
    log(
        "PhysicIQ plots refreshed with "
        + ", ".join(str(Path(leaf).resolve()) for leaf in leaf_values)
    )


def refresh_plots_if_complete(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> bool:
    """Refresh curves only after every configured metric marker is committed."""
    method_key = str(manifest["method_key"])
    step = int(manifest["step"])
    metrics = list(config["metrics"]["cpu"]) + list(config["metrics"]["gpu"])
    complete = all(
        phys_metric_marker_path(config, method_key, step, metric).is_file()
        for metric in metrics
    )
    if complete:
        refresh_plots(config, manifest)
    return complete


def refresh_dashboard(config: dict[str, Any]) -> None:
    """Rebuild the dashboard after a complete PhysicIQ metric set is committed."""
    lock_path = phys_state_root(config) / "dashboard_refresh.lock"
    with exclusive_lock(lock_path):
        subprocess.run(
            [
                config["paths"]["python"],
                config["paths"]["dashboard_builder"],
                "--config",
                config["_config_path"],
            ],
            check=True,
        )
    log("PhysicIQ dashboard refreshed after metric completion")


def metrics_loop(config: dict[str, Any], kind: str, once: bool) -> None:
    while True:
        tasks = metric_tasks(config, kind)
        if not tasks:
            if once:
                return
            time.sleep(int(config["runtime"]["poll_seconds"]))
            continue

        if kind == "gpu":
            # Keep one metric model per worker and spread independent metrics
            # over all configured, dynamically reserved GPUs.
            parallelism = (
                len(candidate_gpu_ids(config))
                * gpu_metric_worker_count(config)
            )
            batch = tasks[:parallelism]
            handled = False
            with ThreadPoolExecutor(
                max_workers=parallelism,
                thread_name_prefix="physiciq-gpu-metric",
            ) as executor:
                future_tasks = {
                    executor.submit(run_gpu_metric_concurrent, config, task): task
                    for task in batch
                }
                for future in as_completed(future_tasks):
                    try:
                        handled = future.result() or handled
                    except Exception as exc:
                        task = future_tasks[future]
                        manifest = task["manifest"]
                        log(
                            f"PhysicIQ metric failed kind=gpu "
                            f"method={manifest['method_key']} "
                            f"step={manifest['step']} metric={task['metric']}: {exc}"
                        )
            if not handled:
                time.sleep(int(config["runtime"]["gpu_poll_seconds"]))
            if once:
                return
            continue

        handled = False
        for task in tasks:
            manifest = task["manifest"]
            task_lock = (
                phys_state_root(config)
                / "metric_locks"
                / kind
                / manifest["method_key"]
                / f"step-{int(manifest['step']):06d}"
                / f"{task['metric']}.lock"
            )
            with try_exclusive_lock(task_lock) as acquired:
                if not acquired:
                    continue
                if phys_metric_marker_path(
                    config,
                    manifest["method_key"],
                    int(manifest["step"]),
                    task["metric"],
                ).is_file():
                    continue
                try:
                    if kind == "gpu":
                        with reserve_available_gpu(config) as gpu_id:
                            run_metric(config, kind, task, gpu_id)
                    else:
                        run_metric(config, kind, task)
                    handled = True
                except Exception as exc:
                    log(
                        f"PhysicIQ metric failed kind={kind} "
                        f"method={manifest['method_key']} step={manifest['step']} "
                        f"metric={task['metric']}: {exc}"
                        )
                    time.sleep(int(config["runtime"]["retry_seconds"]))
                break
        if not handled:
            time.sleep(int(config["runtime"]["gpu_poll_seconds"]))
        if once:
            return


def main() -> None:
    args = parse_args()
    config = load_json(args.config.resolve())
    config["_config_path"] = str(args.config.resolve())
    if args.methods:
        requested = {item.strip() for item in args.methods.split(",") if item.strip()}
        known = set(config["physiciq"]["method_keys"])
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown PhysicIQ watcher methods: {unknown}")
        config["physiciq"]["method_keys"] = [
            method for method in config["physiciq"]["method_keys"]
            if method in requested
        ]
    if args.gpus:
        gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
        if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"Invalid --gpus value: {args.gpus!r}")
        if 4 in gpu_ids:
            raise ValueError("GPU 4 is prohibited by workspace rules")
        config["runtime"]["gpu_ids"] = gpu_ids
    if args.gpu_ready_max_used_mib is not None:
        if args.gpu_ready_max_used_mib < 0:
            raise ValueError("--gpu-ready-max-used-mib must be non-negative")
        config["runtime"]["gpu_ready_max_used_mib"] = args.gpu_ready_max_used_mib
    if args.gpu_metric_workers_per_gpu is not None:
        if args.gpu_metric_workers_per_gpu < 1:
            raise ValueError("--gpu-metric-workers-per-gpu must be positive")
        config["runtime"]["gpu_metric_workers_per_gpu"] = args.gpu_metric_workers_per_gpu
    if not config.get("physiciq", {}).get("enabled"):
        raise SystemExit("PhysicIQ watcher is disabled in config")
    phys_state_root(config).mkdir(parents=True, exist_ok=True)
    if args.mode == "discover":
        print(json.dumps(discover_tasks(config), indent=2, ensure_ascii=False))
        return
    if args.mode == "inference":
        inference_loop(config, args.once)
        return
    if args.kind is None:
        raise SystemExit("--kind is required for metrics mode")
    metrics_loop(config, args.kind, args.once)


if __name__ == "__main__":
    main()
