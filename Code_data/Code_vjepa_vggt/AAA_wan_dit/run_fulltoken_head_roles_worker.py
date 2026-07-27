#!/usr/bin/env python3
"""Run resumable model-seed jobs for the 50-seed full-token head-role study."""

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
from typing import Any

import numpy as np

from run_paired_query_50seeds_worker import (
    SEED42_QUERY_ROOT,
    _clean_localization_masks,
    _deduplicated_paths,
    _run_inference_batch,
    _run_locator,
    _video_map,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_RUNNER = SCRIPT_DIR / "run_fulltoken_moving_capture.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _wait_for_gpu(gpu: int, threshold: int, poll_seconds: int) -> None:
    while True:
        used = _gpu_memory_used(gpu)
        if used <= threshold:
            print(f"[fulltoken-worker] GPU{gpu} available: {used} MiB", flush=True)
            return
        print(
            f"[fulltoken-worker] GPU{gpu} busy: {used} MiB; waiting",
            flush=True,
        )
        time.sleep(poll_seconds)


def _query_case_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")).get("cases", {}))
    except json.JSONDecodeError:
        return 0


def _capture_files(root: Path, model: str) -> list[Path]:
    return sorted(
        root.glob(
            f"block*/matrices/{model}/*/block*_fulltoken_moving.npz"
        )
    )


def _validate_capture_packages(
    root: Path,
    *,
    model: str,
    expected_cases: int,
    config: dict[str, Any],
) -> int:
    blocks = tuple(int(value) for value in config["capture"]["blocks"].split(","))
    steps = tuple(
        int(value) for value in config["capture"]["steps_one_based"].split(",")
    )
    heads = int(config["capture"]["expected_heads"])
    times = int(config["capture"]["latent_grid"][0])
    expected = expected_cases * len(blocks)
    files = _capture_files(root, model)
    if len(files) != expected:
        raise RuntimeError(f"{model}: found {len(files)}/{expected} capture packages")
    observed_pairs: set[tuple[str, int]] = set()
    for path in files:
        block = int(path.name.split("_", 1)[0].replace("block", ""))
        case = path.parent.name
        if block not in blocks:
            raise RuntimeError(f"unexpected block {block} in {path}")
        pair = (case, block)
        if pair in observed_pairs:
            raise RuntimeError(f"duplicate case/block package: {pair}")
        observed_pairs.add(pair)
        with np.load(path, allow_pickle=False) as data:
            if tuple(data["steps_one_based"].astype(int)) != steps:
                raise RuntimeError(f"step mismatch in {path}")
            expected_shapes = {
                "trajectory_valid_times": (len(steps), times),
                "full_features": (len(steps), heads, 8),
                "object_features_by_query_time": (
                    len(steps),
                    heads,
                    times,
                    10,
                ),
            }
            for key, shape in expected_shapes.items():
                if key not in data or data[key].shape != shape:
                    actual = None if key not in data else data[key].shape
                    raise RuntimeError(
                        f"{path}: {key} shape {actual}, expected {shape}"
                    )
    return len(files)


def _run_capture(
    *,
    model: str,
    seed: int,
    gpu: int,
    input_list: Path,
    query_map: Path,
    output_root: Path,
    config: dict[str, Any],
    log_path: Path,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MODEL": model,
            "GPU": str(gpu),
            "SEED": str(seed),
            "INPUT_LIST": str(input_list),
            "QUERY_MAP": str(query_map),
            "OUTPUT_ROOT": str(output_root),
            "BLOCKS": config["capture"]["blocks"],
            "STEPS": config["capture"]["steps_one_based"],
            "QUERY_CHUNK": str(config["capture"]["query_chunk"]),
            "COMPACT_STORAGE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(CAPTURE_RUNNER)]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[fulltoken-worker] command: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


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


def _task_paths(
    *,
    prerequisite_root: Path,
    output_root: Path,
    model: str,
    seed: int,
) -> dict[str, Path]:
    seed_name = f"seed-{seed:06d}"
    return {
        "pass1": prerequisite_root / "pass1" / model / seed_name,
        "query": prerequisite_root / "query_maps" / model / seed_name,
        "capture": output_root / "capture" / model / seed_name,
        "log": output_root / "logs" / model / seed_name,
        "state": output_root / "state" / model / f"{seed_name}.json",
        "claim": output_root / "claims" / f"{model}_{seed_name}.json",
    }


def _task_priority(
    paths: dict[str, Path],
    *,
    model: str,
    cases: set[str],
    expected_cases: int,
) -> int:
    videos = _video_map(paths["pass1"] / "generated" / model, cases)
    query_count = _query_case_count(paths["query"] / "query_map.json")
    if len(videos) == expected_cases and query_count == expected_cases:
        return 0
    if len(videos) == expected_cases:
        return 1
    return 2


def _attempts(state_path: Path) -> int:
    if not state_path.is_file():
        return 0
    try:
        return int(json.loads(state_path.read_text(encoding="utf-8")).get("attempt", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _is_complete(state_path: Path) -> bool:
    if not state_path.is_file():
        return False
    try:
        return json.loads(state_path.read_text(encoding="utf-8")).get("status") == "complete"
    except json.JSONDecodeError:
        return False


def _run_task(
    *,
    model: str,
    seed: int,
    gpu: int,
    input_list: Path,
    cases: set[str],
    expected_cases: int,
    config: dict[str, Any],
    paths: dict[str, Path],
    attempt: int,
) -> None:
    state = {
        "model": model,
        "seed": seed,
        "status": "running",
        "attempt": attempt,
        "gpu": gpu,
        "worker_pid": os.getpid(),
        "started_at_unix": time.time(),
    }
    _atomic_json(paths["state"], state)
    threshold = int(config["execution"]["gpu_start_memory_threshold_mib"])
    poll_seconds = int(config["execution"]["poll_seconds"])
    pass1_videos = _video_map(paths["pass1"] / "generated" / model, cases)
    if len(pass1_videos) != expected_cases:
        _wait_for_gpu(gpu, threshold, poll_seconds)
        _run_inference_batch(
            model=model,
            seed=seed,
            gpu=gpu,
            input_list=input_list,
            query_map=SEED42_QUERY_ROOT / model / "query_map.json",
            output_root=paths["pass1"],
            query_mode="moving",
            log_path=paths["log"] / "pass1.log",
            blocks="0",
            steps="5",
        )
        pass1_videos = _video_map(paths["pass1"] / "generated" / model, cases)
    if len(pass1_videos) != expected_cases:
        raise RuntimeError(
            f"{model}/seed-{seed:06d}: pass1 has "
            f"{len(pass1_videos)}/{expected_cases} videos"
        )

    query_map = paths["query"] / "query_map.json"
    if _query_case_count(query_map) != expected_cases:
        _wait_for_gpu(gpu, threshold, poll_seconds)
        _run_locator(
            model=model,
            gpu=gpu,
            input_list=input_list,
            video_root=paths["pass1"] / "generated" / model,
            output_root=paths["query"],
            log_path=paths["log"] / "locator.log",
        )
    if _query_case_count(query_map) != expected_cases:
        raise RuntimeError(f"{model}/seed-{seed:06d}: query map is incomplete")

    expected_packages = expected_cases * 30
    if len(_capture_files(paths["capture"], model)) != expected_packages:
        _wait_for_gpu(gpu, threshold, poll_seconds)
        _run_capture(
            model=model,
            seed=seed,
            gpu=gpu,
            input_list=input_list,
            query_map=query_map,
            output_root=paths["capture"],
            config=config,
            log_path=paths["log"] / "capture.log",
        )
    package_count = _validate_capture_packages(
        paths["capture"],
        model=model,
        expected_cases=expected_cases,
        config=config,
    )
    replay_videos = _video_map(paths["capture"] / "generated" / model, cases)
    if len(replay_videos) != expected_cases:
        raise RuntimeError(
            f"{model}/seed-{seed:06d}: replay has "
            f"{len(replay_videos)}/{expected_cases} videos"
        )
    mismatches = [
        case
        for case in sorted(cases)
        if _md5(pass1_videos[case]) != _md5(replay_videos[case])
    ]
    if mismatches:
        raise RuntimeError(
            f"{model}/seed-{seed:06d}: deterministic replay mismatch {mismatches}"
        )
    if not config["storage"]["keep_duplicate_replay_videos"]:
        for path in replay_videos.values():
            path.unlink()
    if not config["storage"].get("keep_all_masks", False):
        _clean_localization_masks(query_map)
    attention_junk = paths["pass1"] / "block00"
    if attention_junk.is_dir():
        shutil.rmtree(attention_junk)
    state.update(
        {
            "status": "complete",
            "cases": expected_cases,
            "feature_packages": package_count,
            "md5_verified": True,
            "completed_at_unix": time.time(),
        }
    )
    _atomic_json(paths["state"], state)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    output_root = Path(config["storage"]["output_root"]).expanduser().resolve()
    prerequisite_root = (
        Path(config["storage"]["prerequisite_root"]).expanduser().resolve()
    )
    source_paths = _deduplicated_paths(
        Path(config["input"]["json_list"]).expanduser().resolve()
    )
    expected_cases = int(config["input"]["expected_unique_cases"])
    if len(source_paths) != expected_cases:
        raise ValueError(
            f"input has {len(source_paths)} unique cases, expected {expected_cases}"
        )
    input_list = output_root / "input_lists" / "test5_unique20.txt"
    input_list.parent.mkdir(parents=True, exist_ok=True)
    input_list.write_text(
        "\n".join(str(path) for path in source_paths) + "\n",
        encoding="utf-8",
    )
    cases = {path.stem for path in source_paths}
    max_attempts = int(config["execution"]["max_attempts_per_job"])
    tasks = []
    for model in config["models"]:
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            paths = _task_paths(
                prerequisite_root=prerequisite_root,
                output_root=output_root,
                model=model,
                seed=seed,
            )
            tasks.append(
                (
                    _task_priority(
                        paths,
                        model=model,
                        cases=cases,
                        expected_cases=expected_cases,
                    ),
                    model,
                    seed,
                    paths,
                )
            )
    tasks.sort(key=lambda item: (item[0], item[2], item[1]))
    while True:
        if all(_is_complete(paths["state"]) for _, _, _, paths in tasks):
            print(f"[fulltoken-worker] {args.worker_id}: all jobs complete", flush=True)
            return
        did_work = False
        exhausted = 0
        for _, model, seed, paths in tasks:
            if _is_complete(paths["state"]):
                continue
            previous_attempts = _attempts(paths["state"])
            if previous_attempts >= max_attempts:
                exhausted += 1
                continue
            claim = {
                "model": model,
                "seed": seed,
                "gpu": int(args.gpu),
                "worker_id": args.worker_id,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "claimed_at_unix": time.time(),
            }
            if not _try_claim(paths["claim"], claim):
                continue
            did_work = True
            attempt = previous_attempts + 1
            print(
                f"[fulltoken-worker] claimed {model}/seed-{seed:06d} "
                f"attempt={attempt} GPU{args.gpu}",
                flush=True,
            )
            try:
                _run_task(
                    model=model,
                    seed=seed,
                    gpu=int(args.gpu),
                    input_list=input_list,
                    cases=cases,
                    expected_cases=expected_cases,
                    config=config,
                    paths=paths,
                    attempt=attempt,
                )
                print(
                    f"[fulltoken-worker] complete {model}/seed-{seed:06d}",
                    flush=True,
                )
            except KeyboardInterrupt:
                raise
            except Exception as error:
                _atomic_json(
                    paths["state"],
                    {
                        "model": model,
                        "seed": seed,
                        "status": "failed",
                        "attempt": attempt,
                        "gpu": int(args.gpu),
                        "error": repr(error),
                        "failed_at_unix": time.time(),
                    },
                )
                print(
                    f"[fulltoken-worker] failed {model}/seed-{seed:06d}: {error!r}",
                    flush=True,
                )
            finally:
                paths["claim"].unlink(missing_ok=True)
        if exhausted and exhausted == sum(
            not _is_complete(paths["state"]) for _, _, _, paths in tasks
        ):
            print(
                f"[fulltoken-worker] {args.worker_id}: "
                f"{exhausted} jobs exhausted retry budget",
                flush=True,
            )
            return
        if not did_work:
            time.sleep(int(config["execution"]["poll_seconds"]))


if __name__ == "__main__":
    main()
