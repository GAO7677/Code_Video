#!/usr/bin/env python3
"""Continuously summarize xSSC loss and gradient diagnostics from a train log."""
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
KEYS = (
    "step",
    "loss_main",
    "loss_obj_ctx_reg",
    "gate_abs_max",
    "slot_drop",
    "grad_norm",
    "grad_absmax",
    "grad_params",
)


def parse_line(line: str) -> dict[str, float] | None:
    if "[object-reg]" not in line:
        return None
    row: dict[str, float] = {}
    for key in KEYS:
        match = re.search(rf"\b{key}=({NUMBER})", line)
        if match:
            row[key] = float(match.group(1))
    objects = re.search(r"objects=(\d+)->(\d+)", line)
    if objects:
        row["objects_before"] = float(objects.group(1))
        row["objects_after"] = float(objects.group(2))
    return row if "step" in row else None


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    rows: deque[dict[str, float]] = deque(maxlen=args.window)
    offset = 0
    last_step = -1
    last_progress = time.time()
    while True:
        new_rows: list[dict[str, float]] = []
        if args.log.is_file():
            if args.log.stat().st_size < offset:
                offset = 0
            with args.log.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                for line in handle:
                    row = parse_line(line)
                    if row is not None:
                        rows.append(row)
                        new_rows.append(row)
                        if int(row["step"]) > last_step:
                            last_step = int(row["step"])
                            last_progress = time.time()
                offset = handle.tell()
        if rows:
            warnings: list[str] = []
            critical: list[str] = []
            all_values = [value for row in rows for value in row.values()]
            if not all(math.isfinite(value) for value in all_values):
                critical.append("non-finite metric detected")
            latest = rows[-1]
            if latest.get("grad_norm", 0.0) <= 0.0:
                warnings.append("latest grad_norm is zero")
            if max(row.get("grad_norm", 0.0) for row in rows) > 1.05:
                critical.append("post-clip grad_norm exceeded 1.05")
            stale = max(time.time() - last_progress, 0.0)
            if stale > args.stale_seconds:
                critical.append(f"no optimizer step for {stale:.0f} seconds")
            rolling = {}
            for key in (
                "loss_main",
                "loss_obj_ctx_reg",
                "grad_norm",
                "grad_absmax",
                "gate_abs_max",
                "slot_drop",
                "objects_after",
            ):
                values = [row[key] for row in rows if key in row]
                if values:
                    rolling[key] = {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
            payload = {
                "status": "critical" if critical else "warning" if warnings else "healthy",
                "updated_at_unix": time.time(),
                "latest_step": int(latest["step"]),
                "latest": latest,
                "rolling_window_steps": len(rows),
                "rolling": rolling,
                "seconds_since_progress": stale,
                "warnings": warnings,
                "critical": critical,
            }
            atomic_write(args.output, payload)
            if args.history is not None and new_rows:
                args.history.parent.mkdir(parents=True, exist_ok=True)
                with args.history.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")
            print(f"step={payload['latest_step']} status={payload['status']} warnings={warnings} critical={critical}", flush=True)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
