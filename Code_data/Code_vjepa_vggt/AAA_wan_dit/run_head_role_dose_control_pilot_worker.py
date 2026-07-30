#!/usr/bin/env python3
"""Run resumable matched S/T/C subset ablations for the pilot experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

from matched_head_subset_targets import load_matched_subset
from run_common22_public_head_ablation_worker import _probe, _video_map


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_matched_head_subset_ablation_job.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=RUNNER)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--worker-id")
    parser.add_argument("--gpu-start-memory-threshold-mib", type=int)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_cases(path: Path) -> set[str]:
    return {
        Path(line.strip()).expanduser().resolve().stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _iter_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], Path, Path, set[str], list[str]]:
    config = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Config schema_version must be 1")
    gpus = [int(value) for value in config["execution"]["gpus"]]
    if 4 in gpus and not config["execution"].get("allow_gpu4", False):
        raise ValueError("GPU4 is prohibited by workspace policy")
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    manifest = Path(config["matched_subset_manifest"]).expanduser().resolve()
    input_list = Path(config["input_list"]).expanduser().resolve()
    cases = _input_cases(input_list)
    if len(cases) != int(config["expected_cases"]):
        raise ValueError(f"Expected {config['expected_cases']} cases, found {len(cases)}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    accepted = set(config["subset_matching_modes"])
    subset_ids = [
        subset_id
        for subset_id, record in payload["subsets"].items()
        if record["matching"] in accepted
    ]
    if not subset_ids:
        raise ValueError("No matched subsets selected")
    for subset_id in subset_ids:
        load_matched_subset(manifest, subset_id)
    return config, root, manifest, cases, subset_ids


def _tasks(
    config: dict[str, Any],
    subset_ids: list[str],
) -> list[tuple[str, int, str, int, int]]:
    task_specs = config.get("task_specs")
    if task_specs is None:
        tasks = [
            (str(model), int(seed), subset_id, int(start), int(end))
            for seed in config["seeds"]
            for start, end in config["step_ranges"]
            for subset_id in subset_ids
            for model in config["models"]
        ]
    else:
        tasks = [
            (str(model), int(spec["seed"]), subset_id, int(start), int(end))
            for spec in task_specs
            for model in spec.get("models", config["models"])
            for start, end in spec["step_ranges"]
            for subset_id in subset_ids
        ]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Task matrix contains duplicate tasks")
    return tasks


def _task_id(
    model: str,
    seed: int,
    subset_id: str,
    start: int,
    end: int,
) -> str:
    return (
        f"{model}__seed-{seed:06d}__{subset_id}"
        f"__steps{start:02d}_{end:02d}"
    )


def _job_root(
    root: Path,
    model: str,
    seed: int,
    subset_id: str,
    start: int,
    end: int,
) -> Path:
    variant = f"{subset_id}_steps{start:02d}_{end:02d}"
    return root / "generation" / model / f"seed-{seed:06d}" / variant


def _validate_job(
    job_root: Path,
    *,
    cases: set[str],
    subset_id: str,
    manifest_sha256: str,
    k: int,
    start: int,
    end: int,
) -> dict[str, str]:
    videos = _video_map(job_root, cases)
    if set(videos) != cases:
        raise RuntimeError(f"Found {len(videos)}/{len(cases)} videos under {job_root}")
    expected_category = f"{subset_id}_steps{start:02d}_{end:02d}".upper()
    for case, video in videos.items():
        if _probe(video) != "896,512,49":
            raise RuntimeError(f"Unexpected video shape: {video}")
        sidecar = video.with_suffix(".json")
        if not sidecar.is_file():
            raise RuntimeError(f"Missing sidecar JSON: {video}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        matches = [
            item
            for item in _iter_dicts(payload)
            if item.get("mode") == "self_attn_grouped_head_zero"
            and item.get("category") == expected_category
            and int(item.get("num_targets", -1)) == k
            and item.get("active_denoise_step_range") == [start, end]
            and item.get("target_forward_call_count_ok") is True
            and item.get("target_selection", {}).get("sha256") == manifest_sha256
            and item.get("target_selection", {}).get("subset_id") == subset_id
        ]
        if not matches:
            raise RuntimeError(f"Invalid ablation metadata for {case}: {sidecar}")
    return {case: str(path) for case, path in sorted(videos.items())}


def _claim_is_live(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("host") != socket.gethostname():
            return True
        os.kill(int(payload["pid"]), 0)
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
        return False
    except PermissionError:
        return True


def _try_claim(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if _claim_is_live(path):
            return False
        path.unlink(missing_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return True


def _gpu_memory_used(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def _wait_for_gpu(
    gpu: int,
    threshold: int,
    poll_seconds: int,
    stable_polls: int = 1,
) -> None:
    free_streak = 0
    while free_streak < stable_polls:
        used = _gpu_memory_used(gpu)
        if used <= threshold:
            free_streak += 1
            if free_streak < stable_polls:
                print(
                    f"[dose-worker] GPU{gpu} free check "
                    f"{free_streak}/{stable_polls}; confirming stability",
                    flush=True,
                )
        else:
            free_streak = 0
            print(f"[dose-worker] GPU{gpu} busy: {used} MiB; waiting", flush=True)
        if free_streak < stable_polls:
            time.sleep(poll_seconds)


def _attempts(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempt", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"Task runner does not exist: {runner}")
    config, root, manifest, cases, subset_ids = _load_config(config_path)
    tasks = _tasks(config, subset_ids)
    manifest_sha256 = _sha256(manifest)
    if args.preflight:
        summary = {
            "tasks": len(tasks),
            "videos": len(tasks) * len(cases),
            "models": config["models"],
            "seeds": config["seeds"],
            "step_ranges": config["step_ranges"],
            "subsets": len(subset_ids),
            "cases": len(cases),
            "gpus": config["execution"]["gpus"],
            "manifest_sha256": manifest_sha256,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.gpu is None or args.worker_id is None:
        raise ValueError("--gpu and --worker-id are required")
    if int(args.gpu) not in config["execution"]["gpus"]:
        raise ValueError(f"GPU{args.gpu} is not in the frozen config")

    threshold = (
        int(args.gpu_start_memory_threshold_mib)
        if args.gpu_start_memory_threshold_mib is not None
        else int(config["execution"]["gpu_start_memory_threshold_mib"])
    )
    poll_seconds = int(config["execution"]["poll_seconds"])
    stable_polls = int(config["execution"].get("gpu_free_stability_polls", 1))
    if stable_polls < 1:
        raise ValueError("gpu_free_stability_polls must be >= 1")
    max_attempts = int(config["execution"]["max_attempts_per_task"])
    for model, seed, subset_id, start, end in tasks:
        task_id = _task_id(model, seed, subset_id, start, end)
        state_path = root / "state" / f"{task_id}.json"
        claim_path = root / "claims" / f"{task_id}.json"
        _, targets, _ = load_matched_subset(manifest, subset_id)
        try:
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("status") == "complete":
                    _validate_job(
                        _job_root(root, model, seed, subset_id, start, end),
                        cases=cases,
                        subset_id=subset_id,
                        manifest_sha256=manifest_sha256,
                        k=len(targets),
                        start=start,
                        end=end,
                    )
                    continue
        except Exception as error:
            print(f"[dose-worker] revalidation failed {task_id}: {error}", flush=True)

        attempt = _attempts(state_path) + 1
        if attempt > max_attempts:
            continue
        if not _try_claim(
            claim_path,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "worker_id": args.worker_id,
                "gpu": int(args.gpu),
                "task_id": task_id,
                "claimed_at_unix": time.time(),
            },
        ):
            continue
        started = time.time()
        job_root = _job_root(root, model, seed, subset_id, start, end)
        try:
            _wait_for_gpu(
                int(args.gpu),
                threshold,
                poll_seconds,
                stable_polls,
            )
            if job_root.exists():
                shutil.rmtree(job_root)
            _atomic_json(
                state_path,
                {
                    "status": "running",
                    "task_id": task_id,
                    "model": model,
                    "seed": seed,
                    "subset_id": subset_id,
                    "step_range": [start, end],
                    "k": len(targets),
                    "gpu": int(args.gpu),
                    "worker_id": args.worker_id,
                    "attempt": attempt,
                    "started_at_unix": started,
                },
            )
            log = root / "logs" / f"{task_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "MODEL": model,
                    "SEED": str(seed),
                    "SUBSET_ID": subset_id,
                    "GPU": str(args.gpu),
                    "STEP_START": str(start),
                    "STEP_END": str(end),
                    "INPUT_LIST": str(config["input_list"]),
                    "OUTPUT_ROOT": str(root),
                    "MANIFEST": str(manifest),
                    "ALLOW_GPU4": (
                        "1" if config["execution"].get("allow_gpu4", False) else "0"
                    ),
                }
            )
            with log.open("a", encoding="utf-8") as handle:
                subprocess.run(
                    ["bash", str(runner)],
                    check=True,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            videos = _validate_job(
                job_root,
                cases=cases,
                subset_id=subset_id,
                manifest_sha256=manifest_sha256,
                k=len(targets),
                start=start,
                end=end,
            )
            completed = time.time()
            _atomic_json(
                state_path,
                {
                    "status": "complete",
                    "task_id": task_id,
                    "model": model,
                    "seed": seed,
                    "subset_id": subset_id,
                    "step_range": [start, end],
                    "k": len(targets),
                    "gpu": int(args.gpu),
                    "worker_id": args.worker_id,
                    "attempt": attempt,
                    "started_at_unix": started,
                    "completed_at_unix": completed,
                    "elapsed_seconds": completed - started,
                    "videos": videos,
                },
            )
            print(f"[dose-worker] complete {task_id}", flush=True)
        except Exception as error:
            _atomic_json(
                state_path,
                {
                    "status": "failed",
                    "task_id": task_id,
                    "model": model,
                    "seed": seed,
                    "subset_id": subset_id,
                    "step_range": [start, end],
                    "k": len(targets),
                    "gpu": int(args.gpu),
                    "worker_id": args.worker_id,
                    "attempt": attempt,
                    "error": repr(error),
                    "failed_at_unix": time.time(),
                },
            )
            print(f"[dose-worker] failed {task_id}: {error!r}", flush=True)
        finally:
            claim_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
