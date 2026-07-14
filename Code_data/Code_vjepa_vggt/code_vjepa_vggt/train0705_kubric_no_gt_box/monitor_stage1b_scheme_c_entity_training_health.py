"""Continuously audit Stage1B object/entity training metrics from its log."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path


LINE_MARKER = "[object-reg]"
VALUE_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-stale-seconds", type=int, default=900)
    return parser.parse_args()


def parse_metric_line(line: str) -> dict[str, float]:
    payload = line.split(LINE_MARKER, 1)[1]
    metrics: dict[str, float] = {}
    for key, raw_value in VALUE_PATTERN.findall(payload):
        if key == "objects" or "/" in raw_value or "->" in raw_value:
            continue
        try:
            metrics[key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def audit(metrics: dict[str, float]) -> list[str]:
    issues: list[str] = []
    for key, value in metrics.items():
        if not math.isfinite(value):
            issues.append(f"{key} is non-finite: {value}")
    checks = (
        ("loss_main", 0.0, 10.0),
        ("grad_norm", 0.0, 1.05),
        ("grad_absmax", 0.0, 10.0),
        ("gate_abs_max", 0.0, 0.30),
        ("max_ratio", 0.0, 0.3001),
        ("entity_ratio_max", 0.0, 0.1001),
    )
    for key, lower, upper in checks:
        value = metrics.get(key)
        if value is None:
            issues.append(f"missing {key}")
        elif value < lower or value > upper:
            issues.append(f"{key}={value} outside [{lower}, {upper}]")
    if metrics.get("grad_norm", 0.0) <= 0.0:
        issues.append(f"grad_norm must be positive: {metrics.get('grad_norm')}")
    if metrics.get("grad_params", 0.0) <= 0.0:
        issues.append(f"grad_params must be positive: {metrics.get('grad_params')}")
    return issues


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    health_path = args.output_dir / "health.json"
    history_path = args.output_dir / "health_history.jsonl"
    alerts_path = args.output_dir / "alerts.jsonl"
    last_recorded_step = -1

    while True:
        now = time.time()
        if not args.log.is_file():
            record = {
                "checked_at": utc_now(),
                "status": "waiting_for_log",
                "log": str(args.log),
            }
            health_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)
            time.sleep(args.poll_seconds)
            continue

        lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
        metric_lines = [line for line in lines if LINE_MARKER in line]
        if not metric_lines:
            status = "waiting_for_first_step"
            metrics: dict[str, float] = {}
            issues: list[str] = []
        else:
            metrics = parse_metric_line(metric_lines[-1])
            issues = audit(metrics)
            status = "healthy" if not issues else "anomaly"
        stale_seconds = max(0.0, now - args.log.stat().st_mtime)
        if stale_seconds > args.max_stale_seconds:
            status = "stale"
            issues.append(
                f"training log has not changed for {stale_seconds:.0f}s "
                f"(limit={args.max_stale_seconds}s)"
            )
        record = {
            "checked_at": utc_now(),
            "status": status,
            "issues": issues,
            "latest": metrics,
            "log": str(args.log),
            "log_stale_seconds": stale_seconds,
            "num_metric_steps": len(metric_lines),
        }
        health_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        step = int(metrics.get("step", -1))
        if step > last_recorded_step:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            last_recorded_step = step
        if issues:
            with alerts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
