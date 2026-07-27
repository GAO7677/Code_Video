#!/usr/bin/env python3
"""Wait for all compact captures, aggregate roles, and capture selected QK maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from run_paired_query_50seeds_worker import _deduplicated_paths, _video_map


SCRIPT_DIR = Path(__file__).resolve().parent
WAN_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def _state_counts(root: Path, models: list[str], seeds: list[int]) -> dict[str, int]:
    counts = {"complete": 0, "failed": 0, "running": 0, "missing": 0}
    for model in models:
        for seed in seeds:
            path = root / "state" / model / f"seed-{seed:06d}.json"
            if not path.is_file():
                counts["missing"] += 1
                continue
            try:
                status = json.loads(path.read_text(encoding="utf-8")).get(
                    "status", "missing"
                )
            except json.JSONDecodeError:
                status = "missing"
            counts[status if status in counts else "missing"] += 1
    return counts


def _run_logged(command: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[finalizer] command: {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_for_gpu(gpu: int, threshold: int, poll_seconds: int) -> None:
    while True:
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
        used = int(result.stdout.strip())
        if used <= threshold:
            return
        print(f"[finalizer] GPU{gpu} busy: {used} MiB; waiting for QK", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    prerequisite = (
        Path(config["storage"]["prerequisite_root"]).expanduser().resolve()
    )
    models = list(config["models"])
    seeds = [int(value) for value in config["seeds"]]
    expected = len(models) * len(seeds)
    while True:
        counts = _state_counts(root, models, seeds)
        print(
            f"[finalizer] states {counts}; expected={expected}",
            flush=True,
        )
        if counts["complete"] == expected:
            break
        time.sleep(int(args.poll_seconds))

    analysis_log = root / "finalizer_logs" / "analysis.log"
    _run_logged(
        [
            str(WAN_PYTHON),
            str(SCRIPT_DIR / "analyze_fulltoken_head_roles_batch.py"),
            "--config",
            str(config_path),
        ],
        analysis_log,
    )
    qk_root = root / "alltoken_qk"
    selection = root / "analysis" / "selected_qk_selection.json"
    input_list = root / "input_lists" / "test5_unique20.txt"
    representative_seed = int(config["seeds"][0])
    processes = []
    for gpu, model in zip((0, 1, 2), models):
        _wait_for_gpu(
            gpu,
            int(config["execution"]["gpu_start_memory_threshold_mib"]),
            int(config["execution"]["poll_seconds"]),
        )
        env = os.environ.copy()
        env.update(
            {
                "MODEL": model,
                "GPU": str(gpu),
                "SEED": str(representative_seed),
                "ROOT": str(qk_root),
                "OUTPUT_ROOT": str(qk_root / "capture" / model),
                "SELECTION": str(selection),
                "INPUT_LIST": str(input_list),
                "STEPS": config["capture"]["steps_one_based"],
                "OUTPUT_BINS": "512",
                "QUERY_CHUNK": str(config["capture"]["query_chunk"]),
            }
        )
        log = qk_root / "logs" / f"{model}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        command = ["bash", str(SCRIPT_DIR / "run_selected_qk_capture.sh")]
        handle.write(f"\n[finalizer] command: {' '.join(command)}\n")
        handle.flush()
        processes.append(
            (
                model,
                handle,
                subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                ),
            )
        )
    failures = []
    for model, handle, process in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((model, return_code))
    if failures:
        raise RuntimeError(f"selected QK captures failed: {failures}")

    cases = {
        path.stem
        for path in _deduplicated_paths(
            Path(config["input"]["json_list"]).expanduser().resolve()
        )
    }
    for model in models:
        pass1 = _video_map(
            prerequisite
            / "pass1"
            / model
            / f"seed-{representative_seed:06d}"
            / "generated"
            / model,
            cases,
        )
        replay = _video_map(
            qk_root / "capture" / model / "generated" / model,
            cases,
        )
        if set(pass1) != cases or set(replay) != cases:
            raise RuntimeError(f"{model}: selected-QK replay video set is incomplete")
        mismatches = [
            case for case in sorted(cases) if _md5(pass1[case]) != _md5(replay[case])
        ]
        if mismatches:
            raise RuntimeError(f"{model}: selected-QK replay MD5 mismatch {mismatches}")
        for path in replay.values():
            path.unlink()

    _run_logged(
        [
            str(WAN_PYTHON),
            str(SCRIPT_DIR / "render_selected_qk_batch.py"),
            "--selection",
            str(selection),
            "--capture-root",
            str(qk_root / "capture"),
            "--output-dir",
            str(qk_root / "heatmaps"),
        ],
        root / "finalizer_logs" / "render_qk.log",
    )
    (root / "FINALIZED").write_text(
        json.dumps(
            {
                "completed_at_unix": time.time(),
                "compact_jobs": expected,
                "selected_qk_seed": representative_seed,
                "selected_qk_heads_per_role": int(
                    config["analysis"]["selected_raw_qk_heads_per_role"]
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[finalizer] complete: {root}", flush=True)


if __name__ == "__main__":
    main()
