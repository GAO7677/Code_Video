#!/usr/bin/env python3
"""Run resumable selected-head Q@K replays for frozen pending model-seed jobs."""

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

from run_paired_query_50seeds_worker import _deduplicated_paths, _video_map


SCRIPT_DIR = Path(__file__).resolve().parent
QK_RUNNER = SCRIPT_DIR / "run_selected_qk_capture.sh"
RENDERER = SCRIPT_DIR / "render_pending_selected_qk.py"
WAN_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
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
            print(f"[pending-qk] GPU{gpu} available: {used} MiB", flush=True)
            return
        print(f"[pending-qk] GPU{gpu} busy: {used} MiB; waiting", flush=True)
        time.sleep(poll_seconds)


def _claim_is_live(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("host") != socket.gethostname():
            return True
        os.kill(int(payload["pid"]), 0)
        return True
    except (
        FileNotFoundError,
        ProcessLookupError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ):
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


def _state_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "missing"))
    except json.JSONDecodeError:
        return "invalid"


def _validate_qk(
    *,
    capture_seed_root: Path,
    selection: dict,
    model: str,
    steps: tuple[int, ...],
) -> int:
    count = 0
    for case, item in selection["samples"][model].items():
        by_block: dict[int, set[int]] = {}
        for pair in item["roles"].values():
            by_block.setdefault(int(pair["block"]), set()).add(int(pair["head"]))
        for block, expected_heads in by_block.items():
            path = (
                capture_seed_root
                / model
                / f"block{block:02d}"
                / "matrices"
                / model
                / case
                / f"block{block:02d}_selected_qk.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            import numpy as np

            with np.load(path, allow_pickle=False) as data:
                actual_steps = tuple(data["steps_one_based"].astype(int))
                actual_heads = set(data["selected_heads"].astype(int).tolist())
                attention = data["softmax_attention_mass"]
                temporal = data["temporal_matrix"]
                if actual_steps != steps:
                    raise RuntimeError(f"{path}: steps {actual_steps}, expected {steps}")
                if actual_heads != expected_heads:
                    raise RuntimeError(
                        f"{path}: heads {actual_heads}, expected {expected_heads}"
                    )
                expected_shape = (len(steps), len(expected_heads), 512, 512)
                if attention.shape != expected_shape:
                    raise RuntimeError(
                        f"{path}: softmax shape {attention.shape}, "
                        f"expected {expected_shape}"
                    )
                expected_temporal_shape = (
                    len(steps),
                    len(expected_heads),
                    13,
                    13,
                )
                if temporal.shape != expected_temporal_shape:
                    raise RuntimeError(
                        f"{path}: temporal shape {temporal.shape}, "
                        f"expected {expected_temporal_shape}"
                    )
                if not np.allclose(
                    temporal.astype(np.float32).sum(axis=-1),
                    1.0,
                    atol=1.0e-4,
                ):
                    raise RuntimeError(f"{path}: temporal rows do not sum to one")
            count += 1
    return count


def _run_job(
    *,
    model: str,
    seed: int,
    gpu: int,
    config: dict,
    selection_path: Path,
    selection: dict,
    input_list: Path,
    pending_root: Path,
    prerequisite_root: Path,
    cases: set[str],
) -> dict[str, Any]:
    seed_name = f"seed-{seed:06d}"
    capture_seed_root = pending_root / "capture" / seed_name
    output_root = capture_seed_root / model
    log = pending_root / "logs" / model / f"{seed_name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    threshold = int(config["execution"]["gpu_start_memory_threshold_mib"])
    poll_seconds = int(config["execution"]["poll_seconds"])
    _wait_for_gpu(gpu, threshold, poll_seconds)
    env = os.environ.copy()
    env.update(
        {
            "MODEL": model,
            "GPU": str(gpu),
            "SEED": str(seed),
            "ROOT": str(pending_root),
            "OUTPUT_ROOT": str(output_root),
            "SELECTION": str(selection_path),
            "INPUT_LIST": str(input_list),
            "STEPS": config["capture"]["steps_one_based"],
            "OUTPUT_BINS": "512",
            "QUERY_CHUNK": str(config["capture"]["query_chunk"]),
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[pending-qk] seed={seed} model={model} GPU={gpu}\n")
        handle.flush()
        subprocess.run(
            ["bash", str(QK_RUNNER)],
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

    steps = tuple(
        int(value) for value in config["capture"]["steps_one_based"].split(",")
    )
    package_count = _validate_qk(
        capture_seed_root=capture_seed_root,
        selection=selection,
        model=model,
        steps=steps,
    )
    source = _video_map(
        prerequisite_root / "pass1" / model / seed_name / "generated" / model,
        cases,
    )
    replay = _video_map(output_root / "generated" / model, cases)
    if set(source) != cases or set(replay) != cases:
        raise RuntimeError(
            f"{model}/{seed_name}: source={len(source)}, replay={len(replay)}, "
            f"expected={len(cases)}"
        )
    mismatches = [
        case for case in sorted(cases) if _md5(source[case]) != _md5(replay[case])
    ]
    if mismatches:
        raise RuntimeError(f"{model}/{seed_name}: replay MD5 mismatch {mismatches}")
    shutil.rmtree(output_root / "generated")

    heatmap_root = pending_root / "heatmaps" / seed_name
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            [
                str(WAN_PYTHON),
                str(RENDERER),
                "--selection",
                str(selection_path),
                "--capture-root",
                str(capture_seed_root),
                "--output-dir",
                str(heatmap_root),
                "--model",
                model,
            ],
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    contacts = list((heatmap_root / model).glob("*/all_roles_softmax_qk.png"))
    temporal_contacts = list(
        (heatmap_root / model).glob("*/all_roles_temporal_13x13.png")
    )
    if len(contacts) != len(cases) or len(temporal_contacts) != len(cases):
        raise RuntimeError(
            f"{model}/{seed_name}: rendered QK={len(contacts)} and "
            f"temporal={len(temporal_contacts)}, expected {len(cases)} each"
        )
    return {
        "qk_packages": package_count,
        "contact_sheets": len(contacts),
        "temporal_contact_sheets": len(temporal_contacts),
        "md5_verified": True,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    snapshot = json.loads(
        args.snapshot.expanduser().resolve().read_text(encoding="utf-8")
    )
    output_root = Path(config["storage"]["output_root"]).expanduser().resolve()
    pending_root = output_root / "pending_selected_qk"
    ready_path = pending_root / "READY.json"
    while not ready_path.is_file():
        print(f"[pending-qk] {args.worker_id}: waiting for {ready_path}", flush=True)
        time.sleep(int(config["execution"]["poll_seconds"]))
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    selection_path = Path(ready["selection"]).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    prerequisite_root = Path(config["storage"]["prerequisite_root"]).expanduser().resolve()
    input_list = output_root / "input_lists" / "test5_unique20.txt"
    cases = {
        path.stem
        for path in _deduplicated_paths(
            Path(config["input"]["json_list"]).expanduser().resolve()
        )
    }
    tasks = sorted(
        (
            (str(model), int(item["seed"]))
            for model, rows in snapshot["pending"].items()
            for item in rows
        ),
        key=lambda item: (item[1], item[0]),
    )
    state_root = pending_root / "state"
    claim_root = pending_root / "claims"
    max_attempts = int(config["execution"]["max_attempts_per_job"])
    while True:
        if all(
            _state_status(state_root / model / f"seed-{seed:06d}.json") == "complete"
            for model, seed in tasks
        ):
            print(f"[pending-qk] {args.worker_id}: all jobs complete", flush=True)
            return
        did_work = False
        for model, seed in tasks:
            state_path = state_root / model / f"seed-{seed:06d}.json"
            if _state_status(state_path) == "complete":
                continue
            compact_state = (
                output_root / "state" / model / f"seed-{seed:06d}.json"
            )
            if _state_status(compact_state) != "complete":
                continue
            previous = {}
            if state_path.is_file():
                try:
                    previous = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    previous = {}
            attempt = int(previous.get("attempt", 0)) + 1
            if attempt > max_attempts:
                continue
            claim_path = claim_root / f"{model}_seed-{seed:06d}.json"
            claim = {
                "model": model,
                "seed": seed,
                "gpu": int(args.gpu),
                "worker_id": args.worker_id,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "claimed_at_unix": time.time(),
            }
            if not _try_claim(claim_path, claim):
                continue
            did_work = True
            _atomic_json(
                state_path,
                {
                    **claim,
                    "status": "running",
                    "attempt": attempt,
                    "started_at_unix": time.time(),
                },
            )
            print(
                f"[pending-qk] claimed {model}/seed-{seed:06d} "
                f"attempt={attempt} GPU{args.gpu}",
                flush=True,
            )
            try:
                result = _run_job(
                    model=model,
                    seed=seed,
                    gpu=int(args.gpu),
                    config=config,
                    selection_path=selection_path,
                    selection=selection,
                    input_list=input_list,
                    pending_root=pending_root,
                    prerequisite_root=prerequisite_root,
                    cases=cases,
                )
                _atomic_json(
                    state_path,
                    {
                        **claim,
                        **result,
                        "status": "complete",
                        "attempt": attempt,
                        "completed_at_unix": time.time(),
                    },
                )
            except KeyboardInterrupt:
                raise
            except Exception as error:
                _atomic_json(
                    state_path,
                    {
                        **claim,
                        "status": "failed",
                        "attempt": attempt,
                        "error": repr(error),
                        "failed_at_unix": time.time(),
                    },
                )
                print(
                    f"[pending-qk] failed {model}/seed-{seed:06d}: {error!r}",
                    flush=True,
                )
            finally:
                claim_path.unlink(missing_ok=True)
        if not did_work:
            time.sleep(int(config["execution"]["poll_seconds"]))


if __name__ == "__main__":
    main()
