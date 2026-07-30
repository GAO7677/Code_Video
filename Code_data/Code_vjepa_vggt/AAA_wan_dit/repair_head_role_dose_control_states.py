#!/usr/bin/env python3
"""Repair task states after an infrastructure-only validation failure."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

from matched_head_subset_targets import load_matched_subset
from run_head_role_dose_control_pilot_worker import (
    _atomic_json,
    _claim_is_live,
    _job_root,
    _load_config,
    _sha256,
    _task_id,
    _tasks,
    _validate_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--reset-ffprobe-failures",
        action="store_true",
        help="Reset attempts when old state failed with FileNotFoundError.",
    )
    parser.add_argument(
        "--reset-orphaned-running",
        action="store_true",
        help="Reset an unclaimed running state interrupted for worker repair.",
    )
    return parser.parse_args()


def write_report(root: Path, report: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = root / "repair_reports" / f"state_repair_{stamp}.json"
    _atomic_json(path, report)
    return path


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config, root, manifest, cases, subset_ids = _load_config(config_path)
    manifest_sha256 = _sha256(manifest)
    repaired: list[str] = []
    reset: list[dict[str, str]] = []
    skipped_live: list[str] = []
    unchanged: list[dict[str, str]] = []

    for model, seed, subset_id, start, end in _tasks(config, subset_ids):
        task_id = _task_id(model, seed, subset_id, start, end)
        state_path = root / "state" / f"{task_id}.json"
        claim_path = root / "claims" / f"{task_id}.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            continue
        if claim_path.is_file() and _claim_is_live(claim_path):
            skipped_live.append(task_id)
            continue

        claim_path.unlink(missing_ok=True)
        _, targets, _ = load_matched_subset(manifest, subset_id)
        try:
            videos = _validate_job(
                _job_root(root, model, seed, subset_id, start, end),
                cases=cases,
                subset_id=subset_id,
                manifest_sha256=manifest_sha256,
                k=len(targets),
                start=start,
                end=end,
            )
        except Exception as error:
            old_error = str(state.get("error", ""))
            reset_reason = None
            if args.reset_ffprobe_failures and "FileNotFoundError" in old_error:
                reset_reason = "ffprobe_file_not_found"
            elif args.reset_orphaned_running and state.get("status") == "running":
                reset_reason = "orphaned_running_worker"
            if reset_reason is not None:
                state.update(
                    {
                        "status": "failed",
                        "attempt": 0,
                        "repair_reset_reason": reset_reason,
                        "repair_reset_host": socket.gethostname(),
                        "repair_reset_at_unix": time.time(),
                        "repair_validation_error": repr(error),
                    }
                )
                _atomic_json(state_path, state)
                reset.append(
                    {"task_id": task_id, "validation_error": repr(error)}
                )
            else:
                unchanged.append(
                    {"task_id": task_id, "validation_error": repr(error)}
                )
            continue

        state.update(
            {
                "status": "complete",
                "completed_at_unix": time.time(),
                "videos": videos,
                "repaired_after_validation_failure": True,
                "repair_host": socket.gethostname(),
            }
        )
        _atomic_json(state_path, state)
        repaired.append(task_id)

    report = {
        "schema_version": 1,
        "config": str(config_path),
        "root": str(root),
        "repaired_complete": repaired,
        "reset_for_retry": reset,
        "skipped_live": skipped_live,
        "unchanged": unchanged,
    }
    report_path = write_report(root, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "repaired_complete": len(repaired),
                "reset_for_retry": len(reset),
                "skipped_live": len(skipped_live),
                "unchanged": len(unchanged),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
