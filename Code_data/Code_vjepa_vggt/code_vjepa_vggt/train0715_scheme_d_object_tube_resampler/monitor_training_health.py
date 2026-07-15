#!/usr/bin/env python3
"""Continuously summarize Scheme-D object-reg training diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path


NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
METRIC_KEYS = (
    "step",
    "loss_main",
    "loss_obj_ctx_reg",
    "loss_obj_gate_reg",
    "loss_obj_adapter_mlp_reg",
    "adapter_mlp_cap",
    "gate_abs_max",
    "max_ratio",
    "pre_guard_max_ratio",
    "guard_layers",
    "grad_norm",
    "grad_absmax",
    "grad_params",
    "entity_active",
    "entity_matched",
    "entity_ratio_max",
    "entity_cap",
    "entity_cap_scale_min",
    "entity_drop",
)


def parse_object_reg(line: str) -> dict[str, float] | None:
    if "[object-reg]" not in line:
        return None
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        match = re.search(rf"\b{re.escape(key)}=({NUMBER})", line)
        if match:
            metrics[key] = float(match.group(1))
    objects = re.search(r"objects=(\d+)->(\d+)", line)
    if objects:
        metrics["objects_before"] = float(objects.group(1))
        metrics["objects_after"] = float(objects.group(2))
    if "step" not in metrics:
        return None
    return metrics


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def summarize(
    rows: deque[dict[str, float]],
    *,
    log_path: Path,
    last_progress_time: float,
    stale_seconds: float,
) -> dict[str, object]:
    latest = rows[-1]
    warnings: list[str] = []
    critical: list[str] = []
    finite_values = [
        value
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    ]
    if not all(math.isfinite(value) for value in finite_values):
        critical.append("non-finite metric detected")

    rolling: dict[str, dict[str, float]] = {}
    for key in (
        "loss_main",
        "grad_norm",
        "grad_absmax",
        "max_ratio",
        "gate_abs_max",
        "entity_ratio_max",
    ):
        values = [row[key] for row in rows if key in row]
        if values:
            rolling[key] = {
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
            }

    def rolling_max(key: str) -> float:
        return float(rolling.get(key, {}).get("max", 0.0))

    trends: dict[str, dict[str, float]] = {}
    midpoint = len(rows) // 2
    if midpoint > 0:
        first_half = list(rows)[:midpoint]
        second_half = list(rows)[midpoint:]
        for key in ("loss_main", "grad_norm", "max_ratio"):
            first_values = [row[key] for row in first_half if key in row]
            second_values = [row[key] for row in second_half if key in row]
            if first_values and second_values:
                first_mean = sum(first_values) / len(first_values)
                second_mean = sum(second_values) / len(second_values)
                trends[key] = {
                    "first_half_mean": first_mean,
                    "second_half_mean": second_mean,
                    "mean_ratio": second_mean / max(first_mean, 1.0e-12),
                    "mean_delta": second_mean - first_mean,
                }

    if rolling_max("loss_main") > 10.0:
        critical.append("loss_main exceeded 10")
    loss_trend = trends.get("loss_main", {})
    if (
        loss_trend.get("second_half_mean", 0.0) > 1.0
        and loss_trend.get("mean_ratio", 0.0) > 3.0
    ):
        warnings.append("rolling loss mean increased by more than 3x")
    zero_loss_steps = sum(row.get("loss_main", 0.0) <= 0.0 for row in rows)
    if zero_loss_steps > 1:
        warnings.append(
            f"{zero_loss_steps} zero-weight main-loss steps in rolling window"
        )
    if rolling_max("grad_norm") > 1.05:
        critical.append("post-clip grad_norm exceeded 1.05")
    grad_trend = trends.get("grad_norm", {})
    if (
        grad_trend.get("second_half_mean", 0.0) > 0.01
        and grad_trend.get("mean_ratio", 0.0) > 2.0
    ):
        warnings.append("rolling gradient mean increased by more than 2x")
    if latest.get("grad_norm", 0.0) <= 0.0:
        warnings.append("latest grad_norm is zero")
    if rolling_max("max_ratio") >= 0.15:
        warnings.append("object residual reached half of the 0.30 guard")
    residual_trend = trends.get("max_ratio", {})
    if (
        residual_trend.get("second_half_mean", 0.0) > 0.03
        and residual_trend.get("mean_ratio", 0.0) > 2.0
    ):
        warnings.append("rolling object residual mean increased by more than 2x")
    if any(row.get("guard_layers", 0.0) > 0.0 for row in rows):
        warnings.append("object residual guard activated")
    if any(row.get("adapter_mlp_cap", 0.0) > 0.0 for row in rows):
        warnings.append("object adapter MLP cap activated")
    if rolling_max("entity_ratio_max") >= 0.25:
        warnings.append("entity residual reached half of the 0.50 cap")
    if any(row.get("entity_cap", 0.0) > 0.0 for row in rows):
        warnings.append("entity residual cap activated")
    seconds_since_progress = max(time.time() - last_progress_time, 0.0)
    if seconds_since_progress > stale_seconds:
        critical.append(
            f"no new optimizer step for {seconds_since_progress:.0f} seconds"
        )

    status = "critical" if critical else "warning" if warnings else "healthy"
    return {
        "status": status,
        "updated_at_unix": time.time(),
        "log_path": str(log_path),
        "latest_step": int(latest["step"]),
        "latest": latest,
        "rolling_window_steps": len(rows),
        "rolling": rolling,
        "trends": trends,
        "seconds_since_progress": seconds_since_progress,
        "zero_loss_steps": zero_loss_steps,
        "warnings": warnings,
        "critical": critical,
    }


def read_new_rows(
    log_path: Path,
    *,
    offset: int,
) -> tuple[int, list[dict[str, float]]]:
    if not log_path.is_file():
        return offset, []
    size = log_path.stat().st_size
    if size < offset:
        offset = 0
    rows: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        while True:
            line = handle.readline()
            if not line:
                break
            metrics = parse_object_reg(line)
            if metrics is not None:
                rows.append(metrics)
        return handle.tell(), rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.window <= 0 or args.poll_seconds <= 0.0:
        raise ValueError("window and poll-seconds must be positive")
    rows: deque[dict[str, float]] = deque(maxlen=args.window)
    offset = 0
    last_step = -1
    last_progress_time = time.time()
    while True:
        offset, new_rows = read_new_rows(args.log, offset=offset)
        for row in new_rows:
            rows.append(row)
            step = int(row["step"])
            if step > last_step:
                last_step = step
                last_progress_time = time.time()
        if rows:
            payload = summarize(
                rows,
                log_path=args.log,
                last_progress_time=last_progress_time,
                stale_seconds=args.stale_seconds,
            )
            atomic_write_json(args.output, payload)
            if args.history is not None and new_rows:
                args.history.parent.mkdir(parents=True, exist_ok=True)
                with args.history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(
                f"step={payload['latest_step']} status={payload['status']} "
                f"warnings={payload['warnings']} critical={payload['critical']}",
                flush=True,
            )
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
