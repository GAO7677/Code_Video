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
SUMMARY_KEYS = (
    "loss_main",
    "grad_norm",
    "grad_absmax",
    "adapter_mlp_ratio_max",
    "gate_abs_max",
    "max_ratio",
    "pre_guard_max_ratio",
    "entity_ratio_max",
)


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
        if key == "adapter_mlp_ratio" and "/" in raw_value:
            mean_value, max_value = raw_value.split("/", 1)
            try:
                metrics["adapter_mlp_ratio_mean"] = float(mean_value)
                metrics["adapter_mlp_ratio_max"] = float(max_value)
            except ValueError:
                continue
            continue
        if key == "objects" and "->" in raw_value:
            before_value, after_value = raw_value.split("->", 1)
            try:
                metrics["objects_before_dropout"] = float(before_value)
                metrics["objects_after_dropout"] = float(after_value)
            except ValueError:
                continue
            continue
        if "/" in raw_value or "->" in raw_value:
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
        ("adapter_mlp_ratio_max", 0.0, 3.0001),
        ("adapter_mlp_cap", 0.0, 1.0),
        ("max_ratio", 0.0, 0.3001),
        ("guard_scale_min", 0.0, 1.0),
        # This log metric is measured before the configured 0.10 hard cap.
        # Saturation is a warning; only a 10x pre-cap excursion is anomalous.
        ("entity_ratio_max", 0.0, 1.0),
        ("entity_drop", 0.0, 1.0),
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


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in SUMMARY_KEYS:
        values = [row[key] for row in rows if key in row and math.isfinite(row[key])]
        if values:
            summary[key] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "last": values[-1],
            }
    return summary


def saturation_warnings(rows: list[dict[str, float]]) -> list[str]:
    warnings: list[str] = []
    checks = (
        (
            "entity_ratio_max",
            0.09,
            "pre-cap entity residual reached the configured 0.10 hard-cap region",
        ),
        (
            "adapter_mlp_ratio_max",
            2.70,
            "adapter residual is above 90% of its 3.0 cap",
        ),
        (
            "pre_guard_max_ratio",
            0.27,
            "object residual is above 90% of its 0.30 guard",
        ),
    )
    for key, threshold, message in checks:
        hits = [row for row in rows if row.get(key, 0.0) >= threshold]
        if hits:
            peak = max(row[key] for row in hits)
            warnings.append(
                f"{message}: first_step={int(hits[0].get('step', -1))}, peak={peak}"
            )
    cap_hits = [row for row in rows if row.get("adapter_mlp_cap", 0.0) > 0.0]
    if cap_hits:
        warnings.append(
            "adapter hard cap was applied: "
            f"first_step={int(cap_hits[0].get('step', -1))}"
        )
    guard_hits = [row for row in rows if row.get("guard_layers", 0.0) > 0.0]
    if guard_hits:
        warnings.append(
            "object ratio guard was applied: "
            f"first_step={int(guard_hits[0].get('step', -1))}"
        )
    return warnings


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
        parsed_rows = [parse_metric_line(line) for line in metric_lines]
        if not parsed_rows:
            status = "waiting_for_first_step"
            metrics: dict[str, float] = {}
            issues: list[str] = []
            warnings: list[str] = []
            violations: list[dict[str, object]] = []
        else:
            metrics = parsed_rows[-1]
            violations = []
            for row in parsed_rows:
                for issue in audit(row):
                    violations.append({"step": int(row.get("step", -1)), "issue": issue})
            issues = [
                f"step={item['step']}: {item['issue']}" for item in violations[-20:]
            ]
            warnings = saturation_warnings(parsed_rows)
            status = "anomaly" if violations else "warning" if warnings else "healthy"
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
            "warnings": warnings,
            "latest": metrics,
            "recent_100_summary": summarize(parsed_rows[-100:]),
            "overall_summary": summarize(parsed_rows),
            "num_historical_violations": len(violations),
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
