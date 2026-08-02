#!/usr/bin/env python3
"""Score the currently generated subset of PhysicIQ outputs.

This runner intentionally writes separate "partial" summaries/markers so the
normal 67-case PhysicIQ watcher can still run full metrics later.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from xssc_lora_physiciq_watch import refresh_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--steps", default="2000,2500")
    parser.add_argument("--methods", default="full_sa,s_head59,t_head70")
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric list. Defaults to all configured PhysicIQ metrics.",
    )
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def metric_list(config: dict[str, Any], requested: str | None) -> list[str]:
    if requested:
        return split_csv(requested)
    return list(config["metrics"]["cpu"]) + list(config["metrics"]["gpu"])


def result_dir_name(config: dict[str, Any], method: str, step: int) -> str:
    return config["physiciq"]["method_name_template"].format(method_key=method, step=step)


def resolve_input_json(payload: dict[str, Any]) -> Path | None:
    value = payload.get("input_json")
    if not isinstance(value, str) or not value.strip():
        value = payload.get("case_json")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_file() else None


def collect_generated_inputs(result_root: Path) -> list[Path]:
    inputs: list[Path] = []
    for result_json in sorted(result_root.glob("*.json")):
        if result_json.name.startswith("eval_summary_"):
            continue
        if result_json.name in {"summary.json", "batch_manifest.json", "eval_summary.json"}:
            continue
        if not result_json.with_suffix(".mp4").is_file():
            continue
        try:
            input_json = resolve_input_json(load_json(result_json))
        except Exception:
            input_json = None
        if input_json is not None:
            inputs.append(input_json)
    return sorted(set(inputs))


def marker_path(config: dict[str, Any], method: str, step: int, metric: str) -> Path:
    return (
        Path(config["paths"]["watch_root"]).resolve()
        / "state"
        / "physiciq_partial_metrics"
        / method
        / f"step-{step:06d}"
        / f"{metric}.json"
    )


def summary_path(config: dict[str, Any], method: str, step: int, metric: str) -> Path:
    return (
        Path(config["paths"]["watch_root"]).resolve()
        / "physiciq_partial_metric_task_summaries"
        / method
        / f"step-{step:06d}"
        / f"{metric}.json"
    )


def allowlist_path(config: dict[str, Any], method: str, step: int) -> Path:
    return (
        Path(config["paths"]["watch_root"]).resolve()
        / "state"
        / "physiciq_partial_metrics"
        / "allowlists"
        / method
        / f"step-{step:06d}.txt"
    )


def log_path(config: dict[str, Any], method: str, step: int, metric: str) -> Path:
    return (
        Path(config["paths"]["watch_root"]).resolve()
        / "logs"
        / "physiciq_partial_metrics"
        / f"gpu{config['_partial_gpu']}"
        / method
        / f"step-{step:06d}"
        / f"{metric}.log"
    )


def marker_covers_current(path: Path, num_cases: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    return bool(payload.get("ok")) and int(payload.get("num_cases", -1)) >= int(num_cases)


def run_metric(
    config: dict[str, Any],
    *,
    method: str,
    step: int,
    metric: str,
    result_root: Path,
    allowlist: Path,
    num_cases: int,
    force: bool,
) -> None:
    marker = marker_path(config, method, step, metric)
    if not force and marker_covers_current(marker, num_cases):
        print(f"[partial-metric:skip] {method} step={step} metric={metric} cases={num_cases}")
        return

    summary = summary_path(config, method, step, metric)
    log_file = log_path(config, method, step, metric)
    summary.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        config["paths"]["python"],
        config["paths"]["bench_script"],
        "--metric",
        metric,
        "--result-root",
        str(result_root),
        "--input-json-allowlist",
        str(allowlist),
        "--output-summary",
        str(summary),
    ]
    if metric == "wmreward":
        command.extend(["--wmreward-reset-interval", "1000000"])

    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
        "/home/gaoya/Code_Video/Code_data/Code_try0526"
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["CUDA_VISIBLE_DEVICES"] = str(config["_partial_gpu"])

    print(f"[partial-metric:start] {method} step={step} metric={metric} cases={num_cases}")
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{timestamp()}] command={' '.join(command)} "
            f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']}\n"
        )
        handle.flush()
        subprocess.run(command, check=True, env=environment, stdout=handle, stderr=subprocess.STDOUT)

    payload = load_json(summary)
    status = payload.get("metric_status", {})
    ok = (
        isinstance(status, dict)
        and int(status.get("num_cases", -1)) == int(num_cases)
        and int(status.get("num_failed", -1)) == 0
        and int(status.get("num_success", -1)) == int(num_cases)
    )
    atomic_write_json(
        marker,
        {
            "ok": ok,
            "completed_utc": timestamp(),
            "method_key": method,
            "step": step,
            "metric": metric,
            "num_cases": num_cases,
            "result_root": str(result_root),
            "allowlist": str(allowlist),
            "summary_path": str(summary),
            "metric_status": status,
            "errors": payload.get("errors", []),
        },
    )
    if not ok:
        raise RuntimeError(f"Metric summary did not cleanly cover partial set: {summary}")
    print(f"[partial-metric:done] {method} step={step} metric={metric} cases={num_cases}")
    refresh_dashboard(config)


def main() -> None:
    args = parse_args()
    config = load_json(args.config.expanduser().resolve())
    config["_config_path"] = str(args.config.expanduser().resolve())
    config["_partial_gpu"] = str(args.gpu)
    steps = [int(step) for step in split_csv(args.steps)]
    methods = split_csv(args.methods)
    metrics = metric_list(config, args.metrics)
    output_root = Path(config["physiciq"]["output_root"]).resolve()

    for step in steps:
        for method in methods:
            result_root = output_root / result_dir_name(config, method, step)
            inputs = collect_generated_inputs(result_root) if result_root.exists() else []
            if len(inputs) < int(args.min_cases):
                print(f"[partial-metric:no-cases] {method} step={step} cases={len(inputs)}")
                continue
            allowlist = allowlist_path(config, method, step)
            allowlist.parent.mkdir(parents=True, exist_ok=True)
            allowlist.write_text("".join(f"{path}\n" for path in inputs), encoding="utf-8")
            print(f"[partial-metric:set] {method} step={step} cases={len(inputs)} root={result_root}")
            for metric in metrics:
                run_metric(
                    config,
                    method=method,
                    step=step,
                    metric=metric,
                    result_root=result_root,
                    allowlist=allowlist,
                    num_cases=len(inputs),
                    force=bool(args.force),
                )


if __name__ == "__main__":
    main()
