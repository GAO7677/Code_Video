#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every row in a Wan2.2 TI2V manifest sequentially.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--runner_script",
        type=Path,
        default=Path(__file__).resolve().parent / "run_experiment_case.py",
    )
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--vjepa_device_id", type=int, default=None)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def collect_exp_ids(manifest_path: Path, limit: int | None) -> list[str]:
    exp_ids: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exp_ids.append(row["exp_id"])
            if limit is not None and len(exp_ids) >= limit:
                break
    return exp_ids


def main() -> None:
    args = parse_args()
    exp_ids = collect_exp_ids(args.manifest, args.limit)
    if not exp_ids:
        raise RuntimeError(f"No rows found in {args.manifest}")

    failures: list[str] = []
    total = len(exp_ids)
    for idx, exp_id in enumerate(exp_ids, start=1):
        cmd = [
            "python3",
            str(args.runner_script),
            "--manifest",
            str(args.manifest),
            "--exp_id",
            exp_id,
        ]
        if args.device_id is not None:
            cmd.extend(["--device_id", str(args.device_id)])
        if args.vjepa_device_id is not None:
            cmd.extend(["--vjepa_device_id", str(args.vjepa_device_id)])
        if args.offload_model:
            cmd.append("--offload_model")
        if args.t5_cpu:
            cmd.append("--t5_cpu")
        if args.convert_model_dtype:
            cmd.append("--convert_model_dtype")
        if args.dry_run:
            cmd.append("--dry_run")
        print(f"[{idx}/{total}] [RUN] {exp_id}", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append(exp_id)
            print(f"[{idx}/{total}] [FAIL] {exp_id} returncode={result.returncode}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        print("FAILED_EXP_IDS:", flush=True)
        for exp_id in failures:
            print(exp_id, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
