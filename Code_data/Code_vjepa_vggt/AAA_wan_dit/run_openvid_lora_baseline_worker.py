#!/usr/bin/env python3
"""Run and validate the one baseline task that accompanies 33 Head ablations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

from run_common22_public_head_ablation_worker import (
    _atomic_json,
    _input_cases,
    _probe,
    _video_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _iter_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _validate(job_root: Path, cases: set[str]) -> dict[str, str]:
    videos = _video_map(job_root, cases)
    if set(videos) != cases:
        raise RuntimeError(f"Found {len(videos)}/{len(cases)} baseline videos")
    for case, video in videos.items():
        if _probe(video) != "896,512,49":
            raise RuntimeError(f"Unexpected baseline video shape: {video}")
        sidecar = video.with_suffix(".json")
        if not sidecar.is_file():
            raise RuntimeError(f"Missing baseline sidecar: {sidecar}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        matches = [
            item
            for item in _iter_dicts(payload)
            if item.get("mode") == "baseline"
            and item.get("disabled_module") is None
            and item.get("attention_semantics") is None
        ]
        if not matches:
            raise RuntimeError(f"Invalid baseline metadata for {case}: {sidecar}")
    return {case: str(path) for case, path in sorted(videos.items())}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    if config["models"] != ["openvid_lora_step10000"]:
        raise ValueError("Baseline worker requires only openvid_lora_step10000")
    if args.gpu not in {6, 7} or args.gpu not in config["execution"]["gpus"]:
        raise ValueError("This SSH118 experiment is frozen to GPU6/7")
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    input_list = Path(config["input_list"]).expanduser().resolve()
    cases = _input_cases(input_list)
    if len(cases) != int(config["expected_cases"]):
        raise ValueError(f"Expected {config['expected_cases']} cases, found {len(cases)}")
    seed = int(config["seeds"][0])
    job_root = (
        root
        / "generation"
        / "openvid_lora_step10000"
        / f"seed-{seed:06d}"
        / "baseline"
    )
    state_path = root / "baseline_state" / "baseline.json"
    try:
        videos = _validate(job_root, cases)
        print(f"[openvid-baseline] already complete: {len(videos)} videos", flush=True)
        return
    except Exception as error:
        if args.validate_only:
            raise
        print(f"[openvid-baseline] generation required: {error}", flush=True)

    log_path = root / "logs" / "openvid_lora_step10000__seed-000851__baseline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _atomic_json(
        state_path,
        {
            "status": "running",
            "model": "openvid_lora_step10000",
            "seed": seed,
            "gpu": args.gpu,
            "started_at_unix": started,
        },
    )
    env = os.environ.copy()
    env.update(
        {
            "GPU": str(args.gpu),
            "SEED": str(seed),
            "INPUT_LIST": str(input_list),
            "JOB_ROOT": str(job_root),
        }
    )
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            subprocess.run(
                ["bash", str(args.runner.expanduser().resolve())],
                check=True,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        videos = _validate(job_root, cases)
        completed = time.time()
        _atomic_json(
            state_path,
            {
                "status": "complete",
                "model": "openvid_lora_step10000",
                "seed": seed,
                "gpu": args.gpu,
                "started_at_unix": started,
                "completed_at_unix": completed,
                "elapsed_seconds": completed - started,
                "videos": videos,
            },
        )
        print(f"[openvid-baseline] complete: {len(videos)} videos", flush=True)
    except Exception as error:
        _atomic_json(
            state_path,
            {
                "status": "failed",
                "model": "openvid_lora_step10000",
                "seed": seed,
                "gpu": args.gpu,
                "failed_at_unix": time.time(),
                "error": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
