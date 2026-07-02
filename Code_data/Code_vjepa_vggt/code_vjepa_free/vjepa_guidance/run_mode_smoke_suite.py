#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one smoke-test case across all experiment modes.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=None)
    parser.add_argument("--vjepa_device_id", type=int, default=None)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")
    parser.add_argument("--mode_ids", type=str, nargs="*", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--runner_script",
        type=Path,
        default=Path(__file__).resolve().parent / "run_experiment_case.py",
    )
    return parser.parse_args()


def collect_exp_ids(manifest_path: Path, prompt_id: str, seed: int, mode_ids: set[str] | None) -> list[str]:
    exp_ids: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["prompt_id"] != prompt_id:
                continue
            if int(row["seed"]) != seed:
                continue
            if mode_ids is not None and row["mode_id"] not in mode_ids:
                continue
            exp_ids.append(row["exp_id"])
    return exp_ids


def main() -> None:
    args = parse_args()
    mode_ids = set(args.mode_ids) if args.mode_ids else None
    exp_ids = collect_exp_ids(args.manifest, args.prompt_id, args.seed, mode_ids)
    if not exp_ids:
        raise RuntimeError(
            f"No matching rows found for prompt_id={args.prompt_id}, seed={args.seed} in {args.manifest}"
        )

    for exp_id in exp_ids:
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
        print(f"[RUN] {exp_id}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
