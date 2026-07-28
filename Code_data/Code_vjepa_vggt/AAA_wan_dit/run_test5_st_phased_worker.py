#!/usr/bin/env python3
"""Run resumable test_5 S/T/ST phased-ablation tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from common22_public_head_targets import targets_for_role
from run_common22_public_head_ablation_worker import (
    RUNNER,
    _attempts,
    _atomic_json,
    _input_cases,
    _probe,
    _try_claim,
    _video_map,
    _wait_for_gpu,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker-id")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def variant_name(role: str, start: int, end: int) -> str:
    return f"{role}_steps{start:02d}_{end:02d}"


def iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def validate_baseline(
    source: Path,
    cases: set[str],
) -> dict[str, str]:
    videos = _video_map(source, cases)
    if set(videos) != cases:
        raise RuntimeError(
            f"Baseline {source} has {len(videos)}/{len(cases)} expected videos"
        )
    for video in videos.values():
        if _probe(video) != "896,512,49":
            raise RuntimeError(f"Unexpected baseline video shape: {video}")
    return {case: str(path) for case, path in sorted(videos.items())}


def prepare_baselines(
    config: dict[str, Any],
    root: Path,
    cases: set[str],
) -> None:
    seed = int(config["seed"])
    for model in config["models"]:
        source = Path(config["baseline_sources"][model]).expanduser().resolve()
        videos = validate_baseline(source, cases)
        link = (
            root
            / "generated"
            / model
            / f"seed-{seed:06d}"
            / "role-baseline"
        )
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != source:
                raise RuntimeError(f"Baseline link points elsewhere: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace baseline path: {link}")
        else:
            link.symlink_to(source, target_is_directory=True)
        _atomic_json(
            root / "state" / model / f"seed-{seed:06d}" / "role-baseline.json",
            {
                "status": "complete",
                "kind": "reused_baseline",
                "model": model,
                "seed": seed,
                "source": str(source),
                "videos": videos,
                "validated_at_unix": time.time(),
            },
        )


def validate_phased_job(
    job_root: Path,
    *,
    cases: set[str],
    role: str,
    start: int,
    end: int,
    target_count: int,
    report_sha256: str,
) -> dict[str, str]:
    videos = _video_map(job_root, cases)
    if set(videos) != cases:
        raise RuntimeError(
            f"Found {len(videos)}/{len(cases)} expected videos under {job_root}"
        )
    expected_range = [start, end]
    expected_category = variant_name(role, start, end).upper()
    for video in videos.values():
        if _probe(video) != "896,512,49":
            raise RuntimeError(f"Unexpected video shape/frame count: {video}")
        payload = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
        matches = [
            record
            for record in iter_dicts(payload)
            if record.get("mode") == "self_attn_grouped_head_zero"
            and record.get("category") == expected_category
            and int(record.get("num_targets", -1)) == target_count
            and record.get("target_selection", {}).get("sha256") == report_sha256
            and record.get("active_denoise_step_range") == expected_range
            and record.get("target_forward_call_count_ok") is True
            and record.get("observed_target_forward_calls")
            == record.get("expected_target_forward_calls")
        ]
        if not matches:
            raise RuntimeError(
                f"Missing validated {expected_category} metadata: {video}"
            )
    return {case: str(path) for case, path in sorted(videos.items())}


def load_and_validate_config(
    path: Path,
) -> tuple[dict[str, Any], Path, Path, set[str], dict[str, int]]:
    config = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    input_list = Path(config["input_list"]).expanduser().resolve()
    report = Path(config["public_head_report"]).expanduser().resolve()
    cases = _input_cases(input_list)
    if len(cases) != int(config["expected_cases"]):
        raise ValueError(f"Expected {config['expected_cases']} cases, found {len(cases)}")
    if int(config["seed"]) != 851:
        raise ValueError("This frozen experiment must use seed 851")
    roles = [str(value) for value in config["roles"]]
    if roles != ["S", "T", "ST"]:
        raise ValueError(f"Expected roles S/T/ST, found {roles}")
    ranges = [tuple(map(int, value)) for value in config["step_ranges"]]
    expected_ranges = [(0, 5), (5, 10), (0, 10), (10, 20), (20, 30)]
    if ranges != expected_ranges:
        raise ValueError(f"Unexpected denoising ranges: {ranges}")
    target_counts: dict[str, int] = {}
    for role in roles:
        targets, source = targets_for_role(report, role)
        if source["sha256"] != config["public_head_report_sha256"]:
            raise ValueError("Public Head report SHA256 differs from frozen config")
        expected_count = int(config["target_counts"][role])
        if len(targets) != expected_count:
            raise ValueError(
                f"Role {role} expected {expected_count} targets, found {len(targets)}"
            )
        target_counts[role] = len(targets)
    return config, root, report, cases, target_counts


def main() -> None:
    args = parse_args()
    config, root, report, cases, target_counts = load_and_validate_config(args.config)
    prepare_baselines(config, root, cases)
    tasks = [
        (str(model), str(role), int(start), int(end))
        for start, end in config["step_ranges"]
        for role in config["roles"]
        for model in config["models"]
    ]
    if len(tasks) != 45:
        raise RuntimeError(f"Expected 45 phased tasks, found {len(tasks)}")
    if args.preflight:
        print(
            "[test5-phased-preflight] "
            f"cases={len(cases)} tasks={len(tasks)} "
            f"generated_videos={len(tasks) * len(cases)} "
            f"reused_baselines={len(config['models']) * len(cases)}"
        )
        return
    if args.gpu is None or not args.worker_id:
        raise ValueError("--gpu and --worker-id are required unless --preflight is used")

    threshold = int(config["execution"]["gpu_start_memory_threshold_mib"])
    poll_seconds = int(config["execution"]["poll_seconds"])
    max_attempts = int(config["execution"]["max_attempts_per_job"])
    seed = int(config["seed"])
    worker_log = root / "worker_logs" / f"{args.worker_id}.events.jsonl"
    worker_log.parent.mkdir(parents=True, exist_ok=True)

    for model, role, start, end in tasks:
        variant = variant_name(role, start, end)
        label = f"{model}_seed-{seed:06d}_{variant}"
        job_root = (
            root
            / "generated"
            / model
            / f"seed-{seed:06d}"
            / f"role-{variant}"
        )
        state_path = (
            root
            / "state"
            / model
            / f"seed-{seed:06d}"
            / f"role-{variant}.json"
        )
        claim_path = root / "claims" / f"{label}.json"
        try:
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("status") == "complete":
                    validate_phased_job(
                        job_root,
                        cases=cases,
                        role=role,
                        start=start,
                        end=end,
                        target_count=target_counts[role],
                        report_sha256=config["public_head_report_sha256"],
                    )
                    continue
        except Exception as error:
            print(f"[test5-phased] revalidate failed {label}: {error}", flush=True)

        attempt = _attempts(state_path) + 1
        if attempt > max_attempts:
            print(f"[test5-phased] max attempts reached {label}", flush=True)
            continue
        if not _try_claim(
            claim_path,
            {
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "worker_id": args.worker_id,
                "gpu": args.gpu,
                "job": label,
                "claimed_at_unix": time.time(),
            },
        ):
            continue
        started = time.time()
        try:
            _wait_for_gpu(args.gpu, threshold, poll_seconds)
            if job_root.exists():
                shutil.rmtree(job_root)
            _atomic_json(
                state_path,
                {
                    "status": "running",
                    "model": model,
                    "seed": seed,
                    "role": role,
                    "variant": variant,
                    "step_range": [start, end],
                    "target_count": target_counts[role],
                    "gpu": args.gpu,
                    "worker_id": args.worker_id,
                    "attempt": attempt,
                    "started_at_unix": started,
                },
            )
            log_path = root / "logs" / model / f"seed-{seed:06d}" / f"{variant}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "MODEL": model,
                    "SEED": str(seed),
                    "ROLE": role,
                    "GPU": str(args.gpu),
                    "STEP_START": str(start),
                    "STEP_END": str(end),
                    "INPUT_LIST": str(config["input_list"]),
                    "OUTPUT_ROOT": str(root),
                    "PUBLIC_HEAD_REPORT": str(report),
                }
            )
            with log_path.open("a", encoding="utf-8") as handle:
                subprocess.run(
                    ["bash", str(RUNNER)],
                    check=True,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            videos = validate_phased_job(
                job_root,
                cases=cases,
                role=role,
                start=start,
                end=end,
                target_count=target_counts[role],
                report_sha256=config["public_head_report_sha256"],
            )
            completed = time.time()
            _atomic_json(
                state_path,
                {
                    "status": "complete",
                    "model": model,
                    "seed": seed,
                    "role": role,
                    "variant": variant,
                    "step_range": [start, end],
                    "target_count": target_counts[role],
                    "gpu": args.gpu,
                    "worker_id": args.worker_id,
                    "attempt": attempt,
                    "started_at_unix": started,
                    "completed_at_unix": completed,
                    "elapsed_seconds": completed - started,
                    "videos": videos,
                },
            )
            print(f"[test5-phased] complete {label}", flush=True)
        except Exception as error:
            _atomic_json(
                state_path,
                {
                    "status": "failed",
                    "model": model,
                    "seed": seed,
                    "role": role,
                    "variant": variant,
                    "step_range": [start, end],
                    "target_count": target_counts[role],
                    "gpu": args.gpu,
                    "worker_id": args.worker_id,
                    "attempt": attempt,
                    "error": repr(error),
                    "failed_at_unix": time.time(),
                },
            )
            print(f"[test5-phased] failed {label}: {error!r}", flush=True)
        finally:
            claim_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
