#!/usr/bin/env python3
"""Continuously infer and evaluate xSSC LoRA checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator


STEP_PATTERN = re.compile(r"^step-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "inference", "metrics", "refresh", "discover"],
        required=True,
    )
    parser.add_argument("--kind", choices=["cpu", "gpu"])
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def read_inputs(path: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = Path(line.strip()).resolve()
        if candidate not in seen:
            paths.append(candidate)
            seen.add(candidate)
    return paths


def checkpoint_complete(path: Path) -> bool:
    return (
        (path / "checkpoint.safetensors").is_file()
        and (path / "checkpoint.safetensors").stat().st_size > 0
        and (path / "training_state.pt").is_file()
        and (path / "training_state.pt").stat().st_size > 0
    )


def checkpoint_signature(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for name in ("checkpoint.safetensors", "training_state.pt"):
        stat = (path / name).stat()
        result.append((stat.st_size, stat.st_mtime_ns))
    return result[0], result[1]


def discover_checkpoints(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: dict[tuple[str, int], dict[str, Any]] = {}
    for method_index, method in enumerate(config["methods"]):
        key = method["key"]
        for item in method.get("static_checkpoints", []):
            step = int(item["step"])
            path = Path(item["path"]).resolve()
            tasks[(key, step)] = {
                "method_key": key,
                "method_label": method["label"],
                "method_index": method_index,
                "step": step,
                "checkpoint_dir": str(path),
                "source": "static",
            }
        for root_value in method.get("watch_roots", []):
            root = Path(root_value).resolve()
            if not root.is_dir():
                continue
            for path in root.iterdir():
                match = STEP_PATTERN.match(path.name)
                if not match or not path.is_dir():
                    continue
                step = int(match.group(1))
                tasks[(key, step)] = {
                    "method_key": key,
                    "method_label": method["label"],
                    "method_index": method_index,
                    "step": step,
                    "checkpoint_dir": str(path.resolve()),
                    "source": "watch",
                }
    return sorted(tasks.values(), key=lambda row: (row["step"], row["method_index"]))


def state_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["paths"]["watch_root"]).resolve()
    return {
        "root": root,
        "state": root / "state",
        "checkpoints": root / "state" / "checkpoints",
        "metrics": root / "state" / "metrics",
        "summaries": root / "metric_task_summaries",
        "logs": root / "logs",
        "results": root / "results",
        "site": root / "site",
        "pending": root / "state" / "inference.pending",
        "gpu_lock": root / "state" / "gpu.lock",
        "refresh_lock": root / "state" / "refresh.lock",
    }


def prepare_directories(config: dict[str, Any]) -> None:
    for path in state_paths(config).values():
        if path.suffix in {".lock", ".pending"}:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def manifest_path(config: dict[str, Any], method_key: str, step: int) -> Path:
    return (
        state_paths(config)["checkpoints"]
        / method_key
        / f"step-{step:06d}.json"
    )


def metric_marker_path(
    config: dict[str, Any],
    method_key: str,
    step: int,
    metric: str,
) -> Path:
    return (
        state_paths(config)["metrics"]
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.json"
    )


def load_manifests(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = state_paths(config)["checkpoints"]
    manifests = [
        load_json(path)
        for path in sorted(root.glob("*/step-*.json"))
        if path.is_file()
    ]
    return sorted(
        manifests,
        key=lambda row: (int(row["step"]), str(row["method_key"])),
    )


def probe_video(ffprobe: Path, video: Path) -> tuple[int, int, str, int]:
    process = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "csv=p=0",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height, fps, frames = process.stdout.strip().split(",")
    return int(width), int(height), fps, int(frames)


def validate_result_root(
    config: dict[str, Any],
    result_root: Path,
    input_list: Path | None = None,
    expected_cases: int | None = None,
) -> dict[str, Any]:
    runtime = config["runtime"]
    input_paths = read_inputs(
        input_list if input_list is not None else Path(config["paths"]["input_list"])
    )
    expected = (
        int(expected_cases)
        if expected_cases is not None
        else int(runtime["expected_cases"])
    )
    if len(input_paths) != expected:
        raise ValueError(f"Expected {expected} inputs, found {len(input_paths)}")
    ffprobe = Path(config["paths"]["ffprobe"])
    errors: list[str] = []
    for input_path in input_paths:
        stem = input_path.stem
        video = result_root / f"{stem}.mp4"
        result_json = result_root / f"{stem}.json"
        if not video.is_file() or video.stat().st_size == 0:
            errors.append(f"missing video: {video}")
            continue
        if not result_json.is_file() or result_json.stat().st_size == 0:
            errors.append(f"missing result json: {result_json}")
            continue
        payload = load_json(result_json)
        if Path(payload.get("input_json", "")).resolve() != input_path:
            errors.append(f"input_json mismatch: {result_json}")
        try:
            width, height, fps, frames = probe_video(ffprobe, video)
        except Exception as exc:
            errors.append(f"ffprobe failed: {video}: {exc}")
            continue
        if (
            width != int(runtime["width"])
            or height != int(runtime["height"])
            or fps != f"{int(runtime['fps'])}/1"
            or frames != int(runtime["num_frames"])
        ):
            errors.append(
                f"bad media: {video}: {width}x{height} {fps} frames={frames}"
            )
    if errors:
        raise RuntimeError("\n".join(errors[:20]))
    return {
        "num_cases": expected,
        "width": int(runtime["width"]),
        "height": int(runtime["height"]),
        "fps": int(runtime["fps"]),
        "num_frames": int(runtime["num_frames"]),
    }


def method_config(config: dict[str, Any], method_key: str) -> dict[str, Any]:
    for method in config["methods"]:
        if method["key"] == method_key:
            return method
    raise KeyError(method_key)


def register_manifest(
    config: dict[str, Any],
    task: dict[str, Any],
    result_root: Path,
    origin: str,
) -> dict[str, Any]:
    validation = validate_result_root(config, result_root)
    payload = {
        **task,
        "checkpoint_dir": str(Path(task["checkpoint_dir"]).resolve()),
        "result_root": str(result_root.resolve()),
        "origin": origin,
        "inference_completed_utc": timestamp(),
        "validation": validation,
    }
    atomic_write_json(
        manifest_path(config, task["method_key"], int(task["step"])),
        payload,
    )
    log(
        "registered inference "
        f"method={task['method_key']} step={task['step']} origin={origin}"
    )
    return payload


def register_bootstraps(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> int:
    registered = 0
    for task in tasks:
        path = manifest_path(config, task["method_key"], int(task["step"]))
        if path.is_file():
            continue
        method = method_config(config, task["method_key"])
        bootstrap = method.get("bootstrap_result_roots", {}).get(str(task["step"]))
        if not bootstrap:
            continue
        result_root = Path(bootstrap).resolve()
        register_manifest(config, task, result_root, "bootstrap")
        registered += 1
    return registered


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def wait_for_gpu(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    gpu_id = int(runtime["gpu_id"])
    threshold = int(runtime["gpu_ready_max_used_mib"])
    while True:
        memory_used = gpu_memory_used(gpu_id)
        if memory_used <= threshold:
            return
        log(f"GPU{gpu_id} used={memory_used} MiB; waiting for <= {threshold} MiB")
        time.sleep(int(runtime["gpu_poll_seconds"]))


def refresh_site(config: dict[str, Any]) -> None:
    paths = state_paths(config)
    with exclusive_lock(paths["refresh_lock"]):
        manifests = load_manifests(config)
        leaf_list = paths["root"] / "leaf_folders.txt"
        leaf_list.write_text(
            "".join(f"{manifest['result_root']}\n" for manifest in manifests),
            encoding="utf-8",
        )
        if manifests:
            subprocess.run(
                [
                    config["paths"]["python"],
                    config["paths"]["metric_summary_script"],
                    "--input-txt",
                    str(leaf_list),
                    "--output-csv",
                    str(paths["root"] / "metric_summary.csv"),
                    "--input-json-allowlist",
                    config["paths"]["input_list"],
                ],
                check=True,
            )
        subprocess.run(
            [
                config["paths"]["python"],
                config["paths"]["dashboard_builder"],
                "--config",
                str(config["_config_path"]),
            ],
            check=True,
        )


def write_discovery(config: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    atomic_write_json(
        state_paths(config)["state"] / "discovery.json",
        {
            "updated_utc": timestamp(),
            "num_checkpoints": len(tasks),
            "checkpoints": tasks,
        },
    )


def checkpoint_is_stable(config: dict[str, Any], checkpoint: Path) -> bool:
    if not checkpoint_complete(checkpoint):
        return False
    before = checkpoint_signature(checkpoint)
    delay = int(config["runtime"]["checkpoint_stability_seconds"])
    log(f"checking checkpoint stability for {delay}s: {checkpoint}")
    time.sleep(delay)
    return checkpoint_complete(checkpoint) and before == checkpoint_signature(checkpoint)


def run_inference_task(config: dict[str, Any], task: dict[str, Any]) -> None:
    paths = state_paths(config)
    runtime = config["runtime"]
    checkpoint = Path(task["checkpoint_dir"]).resolve()
    if not checkpoint_is_stable(config, checkpoint):
        raise RuntimeError(f"Checkpoint is incomplete or still changing: {checkpoint}")
    method_key = task["method_key"]
    step = int(task["step"])
    method_output_root = paths["results"] / method_key
    output_name = (
        f"step-{step:06d}_steps{int(runtime['num_inference_steps'])}"
        f"_{int(runtime['height'])}x{int(runtime['width'])}"
        f"_ctx{int(runtime['context_frames']):02d}_{int(runtime['num_frames'])}f"
    )
    result_root = method_output_root / output_name
    trace_root = paths["root"] / "numeric_traces" / method_key / output_name
    log_path = paths["logs"] / "inference" / method_key / f"step-{step:06d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "TEST_LIST": config["paths"]["input_list"],
            "NUM_INFERENCE_STEPS": str(runtime["num_inference_steps"]),
            "STEP_OUTPUT_DIR_NAME": output_name,
            "TRACE_ROOT": str(trace_root),
        }
    )
    gpu_id = int(runtime["gpu_id"])
    log(f"inference start method={method_key} step={step} gpu={gpu_id}")
    with log_path.open("a", encoding="utf-8") as log_handle:
        subprocess.run(
            [
                "bash",
                config["paths"]["run_infer_script"],
                str(checkpoint),
                str(gpu_id),
                str(method_output_root),
            ],
            check=True,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    register_manifest(config, task, result_root, "watcher")
    log(f"inference complete method={method_key} step={step}")


def inference_loop(config: dict[str, Any], once: bool) -> None:
    paths = state_paths(config)
    while True:
        tasks = discover_checkpoints(config)
        write_discovery(config, tasks)
        register_bootstraps(config, tasks)
        pending = [
            task
            for task in tasks
            if not manifest_path(
                config, task["method_key"], int(task["step"])
            ).is_file()
        ]
        if pending:
            paths["pending"].write_text(
                json.dumps(
                    {
                        "updated_utc": timestamp(),
                        "num_pending": len(pending),
                        "next": pending[0],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            task = pending[0]
            try:
                with exclusive_lock(paths["gpu_lock"]):
                    wait_for_gpu(config)
                    run_inference_task(config, task)
                refresh_site(config)
            except Exception as exc:
                log(
                    f"inference failed method={task['method_key']} "
                    f"step={task['step']}: {exc}"
                )
                time.sleep(int(config["runtime"]["retry_seconds"]))
        else:
            paths["pending"].unlink(missing_ok=True)
            refresh_site(config)
            if once:
                return
            time.sleep(int(config["runtime"]["poll_seconds"]))
        if once:
            return


def metric_tasks(config: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    metrics = config["metrics"][kind]
    for manifest in load_manifests(config):
        for metric in metrics:
            if metric_marker_path(
                config,
                manifest["method_key"],
                int(manifest["step"]),
                metric,
            ).is_file():
                continue
            tasks.append({"metric": metric, "manifest": manifest})
    return tasks


def run_metric_task(
    config: dict[str, Any],
    kind: str,
    task: dict[str, Any],
) -> None:
    metric = task["metric"]
    manifest = task["manifest"]
    method_key = manifest["method_key"]
    step = int(manifest["step"])
    paths = state_paths(config)
    summary_path = (
        paths["summaries"]
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = (
        paths["logs"]
        / "metrics"
        / kind
        / method_key
        / f"step-{step:06d}"
        / f"{metric}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config["paths"]["python"],
        config["paths"]["bench_script"],
        "--metric",
        metric,
        "--result-root",
        manifest["result_root"],
        "--input-json-allowlist",
        config["paths"]["input_list"],
        "--output-summary",
        str(summary_path),
    ]
    if metric == "wmreward":
        command.extend(["--wmreward-reset-interval", "1000000"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
        "/home/gaoya/Code_Video/Code_data/Code_try0526"
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["CUDA_VISIBLE_DEVICES"] = (
        "" if kind == "cpu" else str(config["runtime"]["gpu_id"])
    )
    log(f"metric start kind={kind} method={method_key} step={step} metric={metric}")
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
                str(config["runtime"]["expected_cases"]),
            ],
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    marker = {
        "completed_utc": timestamp(),
        "kind": kind,
        "method_key": method_key,
        "step": step,
        "metric": metric,
        "result_root": manifest["result_root"],
        "summary_path": str(summary_path),
    }
    atomic_write_json(metric_marker_path(config, method_key, step, metric), marker)
    log(f"metric complete method={method_key} step={step} metric={metric}")


def metrics_loop(config: dict[str, Any], kind: str, once: bool) -> None:
    paths = state_paths(config)
    while True:
        tasks = metric_tasks(config, kind)
        if not tasks:
            refresh_site(config)
            if once:
                return
            time.sleep(int(config["runtime"]["poll_seconds"]))
            continue
        task = tasks[0]
        try:
            if kind == "gpu":
                if paths["pending"].is_file():
                    time.sleep(int(config["runtime"]["gpu_poll_seconds"]))
                    continue
                with exclusive_lock(paths["gpu_lock"]):
                    if paths["pending"].is_file():
                        continue
                    wait_for_gpu(config)
                    run_metric_task(config, kind, task)
            else:
                run_metric_task(config, kind, task)
            refresh_site(config)
        except Exception as exc:
            manifest = task["manifest"]
            log(
                f"metric failed kind={kind} method={manifest['method_key']} "
                f"step={manifest['step']} metric={task['metric']}: {exc}"
            )
            time.sleep(int(config["runtime"]["retry_seconds"]))
        if once:
            return


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    config["_config_path"] = str(config_path)
    prepare_directories(config)
    tasks = discover_checkpoints(config)
    write_discovery(config, tasks)
    if args.mode == "discover":
        print(json.dumps(tasks, indent=2, ensure_ascii=False))
        return
    if args.mode == "bootstrap":
        registered = register_bootstraps(config, tasks)
        refresh_site(config)
        log(f"bootstrap complete registered={registered}")
        return
    if args.mode == "refresh":
        refresh_site(config)
        return
    if args.mode == "inference":
        inference_loop(config, args.once)
        return
    if args.kind is None:
        raise SystemExit("--kind is required for --mode metrics")
    metrics_loop(config, args.kind, args.once)


if __name__ == "__main__":
    main()
