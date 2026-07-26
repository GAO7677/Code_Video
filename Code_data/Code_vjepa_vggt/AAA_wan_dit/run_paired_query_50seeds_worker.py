#!/usr/bin/env python3
"""Run one resumable GPU worker for the paired-query 50-seed experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNNER = SCRIPT_DIR / "run_allblock_ball_query_test5.sh"
LOCATOR = SCRIPT_DIR / "build_model_output_sam2_query_map.py"
WAN_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
SEED42_QUERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_model_specific_query_maps/test5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _deduplicated_paths(input_list: Path) -> list[Path]:
    output = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip()).expanduser().resolve()
        if path not in output:
            output.append(path)
    return output


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


def _wait_for_gpu(gpu: int, threshold: int) -> None:
    while True:
        used = _gpu_memory_used(gpu)
        if used <= threshold:
            print(f"[paired-worker] GPU{gpu} available: {used} MiB", flush=True)
            return
        print(
            f"[paired-worker] GPU{gpu} busy: {used} MiB; waiting",
            flush=True,
        )
        time.sleep(60)


def _run(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[paired-worker] run: {' '.join(command)}; log={log_path}",
        flush=True,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[paired-worker] command: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def _video_map(root: Path, cases: set[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in root.rglob("*.mp4"):
        if path.stem not in cases:
            continue
        if path.stem in output:
            raise RuntimeError(f"duplicate video for {path.stem} under {root}")
        output[path.stem] = path
    return output


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_case_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")).get("cases", {}))


def _capture_count(root: Path, model: str) -> int:
    return len(
        list(
            root.glob(
                f"block*/matrices/{model}/*/"
                "block*_paired_query_features.npz"
            )
        )
    )


def _base_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _run_inference_batch(
    *,
    model: str,
    seed: int,
    gpu: int,
    input_list: Path,
    query_map: Path,
    output_root: Path,
    query_mode: str,
    log_path: Path,
    blocks: str,
    steps: str,
) -> None:
    env = _base_env(gpu)
    env.update(
        {
            "MODEL": model,
            "GPU": str(gpu),
            "SEED": str(seed),
            "INPUT_LIST": str(input_list),
            "QUERY_MAP": str(query_map),
            "QUERY_MODE": query_mode,
            "OUTPUT_ROOT": str(output_root),
            "BLOCKS": blocks,
            "STEPS": steps,
        }
    )
    _run(["bash", str(RUNNER)], log_path=log_path, env=env)


def _run_locator(
    *,
    model: str,
    gpu: int,
    input_list: Path,
    video_root: Path,
    output_root: Path,
    log_path: Path,
) -> None:
    env = _base_env(gpu)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    _run(
        [
            str(WAN_PYTHON),
            str(LOCATOR),
            "--input-list",
            str(input_list),
            "--model",
            model,
            "--video-root",
            str(video_root),
            "--output-dir",
            str(output_root),
        ],
        log_path=log_path,
        env=env,
    )


def _clean_capture_videos(root: Path, cases: set[str]) -> None:
    for path in _video_map(root, cases).values():
        path.unlink()


def _clean_localization_masks(query_map_path: Path) -> None:
    payload = json.loads(query_map_path.read_text(encoding="utf-8"))
    changed = False
    for item in payload["cases"].values():
        mask_value = item.get("masks")
        if mask_value:
            mask_path = Path(mask_value)
            if mask_path.is_file():
                mask_path.unlink()
            item["masks"] = None
            item["masks_retained"] = False
            changed = True
    if changed:
        _atomic_json(query_map_path, payload)


def _run_task(
    *,
    model: str,
    seed: int,
    gpu: int,
    input_list: Path,
    root: Path,
    expected_cases: int,
    config: dict[str, Any],
    label: str,
) -> None:
    case_paths = _deduplicated_paths(input_list)
    cases = {path.stem for path in case_paths}
    if len(cases) != expected_cases:
        raise ValueError(
            f"{label}: got {len(cases)} unique cases, expected {expected_cases}"
        )
    seed_name = f"seed-{seed:06d}"
    pass1_root = root / "pass1" / model / seed_name
    query_root = root / "query_maps" / model / seed_name
    capture_root = root / "capture" / model / seed_name
    log_root = root / "logs" / model / seed_name
    state_path = root / "state" / model / f"{seed_name}.json"
    if state_path.is_file():
        existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        if existing_state.get("status") == "complete":
            print(f"[paired-worker] skip complete {label}", flush=True)
            return
    state = {
        "model": model,
        "seed": seed,
        "label": label,
        "status": "running",
        "updated_at_unix": time.time(),
    }
    _atomic_json(state_path, state)

    pass1_videos = _video_map(pass1_root / "generated" / model, cases)
    if len(pass1_videos) != expected_cases:
        _run_inference_batch(
            model=model,
            seed=seed,
            gpu=gpu,
            input_list=input_list,
            query_map=SEED42_QUERY_ROOT / model / "query_map.json",
            output_root=pass1_root,
            query_mode="moving",
            log_path=log_root / "pass1.log",
            blocks="0",
            steps="5",
        )
        pass1_videos = _video_map(pass1_root / "generated" / model, cases)
    if len(pass1_videos) != expected_cases:
        raise RuntimeError(
            f"{label}: pass1 produced {len(pass1_videos)}/{expected_cases} videos"
        )

    query_map_path = query_root / "query_map.json"
    if _query_case_count(query_map_path) != expected_cases:
        _run_locator(
            model=model,
            gpu=gpu,
            input_list=input_list,
            video_root=pass1_root / "generated" / model,
            output_root=query_root,
            log_path=log_root / "locator.log",
        )
    if _query_case_count(query_map_path) != expected_cases:
        raise RuntimeError(f"{label}: query map is incomplete")

    expected_packages = expected_cases * 30
    if _capture_count(capture_root, model) != expected_packages:
        _run_inference_batch(
            model=model,
            seed=seed,
            gpu=gpu,
            input_list=input_list,
            query_map=query_map_path,
            output_root=capture_root,
            query_mode="paired",
            log_path=log_root / "capture.log",
            blocks=",".join(str(index) for index in range(30)),
            steps="5,15,25,35",
        )
    package_count = _capture_count(capture_root, model)
    if package_count != expected_packages:
        raise RuntimeError(
            f"{label}: capture has {package_count}/{expected_packages} packages"
        )

    replay_videos = _video_map(capture_root / "generated" / model, cases)
    if len(replay_videos) != expected_cases:
        raise RuntimeError(
            f"{label}: replay produced {len(replay_videos)}/{expected_cases} videos"
        )
    mismatches = [
        case
        for case in sorted(cases)
        if _md5(pass1_videos[case]) != _md5(replay_videos[case])
    ]
    if mismatches:
        raise RuntimeError(f"{label}: replay MD5 mismatch: {mismatches}")

    if not config["storage"]["keep_duplicate_replay_videos"]:
        _clean_capture_videos(capture_root / "generated" / model, cases)
    if not config["storage"]["keep_all_masks"]:
        _clean_localization_masks(query_map_path)
    attention_junk = pass1_root / "block00"
    if attention_junk.is_dir():
        shutil.rmtree(attention_junk)

    state.update(
        {
            "status": "complete",
            "feature_packages": package_count,
            "cases": expected_cases,
            "md5_verified": True,
            "updated_at_unix": time.time(),
        }
    )
    _atomic_json(state_path, state)
    print(f"[paired-worker] complete {label}", flush=True)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_paths = _deduplicated_paths(Path(config["input"]["json_list"]))
    if len(source_paths) != int(config["input"]["expected_unique_cases"]):
        raise ValueError("configured input case count does not match")
    input_root = root / "input_lists"
    input_root.mkdir(parents=True, exist_ok=True)
    full_input = input_root / "test5_unique20.txt"
    full_input.write_text(
        "\n".join(str(path) for path in source_paths) + "\n",
        encoding="utf-8",
    )
    smoke_input = input_root / "smoke_one.txt"
    smoke_input.write_text(str(source_paths[0]) + "\n", encoding="utf-8")

    models = tuple(config["models"])
    seeds = tuple(int(seed) for seed in config["seed_sampling"]["seeds"])
    tasks = [
        (model, seed)
        for seed in seeds
        for model in models
    ]
    assigned = tasks[args.worker_index :: args.worker_count]
    if not assigned:
        raise ValueError("worker has no assigned tasks")
    worker_model = assigned[0][0]
    if any(model != worker_model for model, _ in assigned):
        raise ValueError("task ordering no longer assigns one model per worker")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "worker_index": args.worker_index,
                    "worker_count": args.worker_count,
                    "gpu": args.gpu,
                    "model": worker_model,
                    "task_count": len(assigned),
                    "first_task": assigned[0],
                    "last_task": assigned[-1],
                    "unique_cases": len(source_paths),
                }
            )
        )
        return

    _wait_for_gpu(args.gpu, args.gpu_memory_threshold_mib)
    _run_task(
        model=worker_model,
        seed=seeds[0],
        gpu=args.gpu,
        input_list=smoke_input,
        root=root / "preflight",
        expected_cases=1,
        config=config,
        label=f"preflight-{worker_model}-seed-{seeds[0]:06d}",
    )

    for model, seed in assigned:
        _wait_for_gpu(args.gpu, args.gpu_memory_threshold_mib)
        try:
            _run_task(
                model=model,
                seed=seed,
                gpu=args.gpu,
                input_list=full_input,
                root=root,
                expected_cases=20,
                config=config,
                label=f"{model}-seed-{seed:06d}",
            )
        except Exception as error:
            state_path = root / "state" / model / f"seed-{seed:06d}.json"
            _atomic_json(
                state_path,
                {
                    "model": model,
                    "seed": seed,
                    "status": "failed",
                    "error": repr(error),
                    "updated_at_unix": time.time(),
                },
            )
            raise

    print(
        f"[paired-worker] worker {args.worker_index}/{args.worker_count} complete",
        flush=True,
    )


if __name__ == "__main__":
    main()
