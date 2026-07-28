#!/usr/bin/env python3
"""Claim and run resumable common22 public Head ablation jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from common22_public_head_targets import load_public_head_targets


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_common22_public_head_ablation_job.sh"
FFPROBE = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffprobe")


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


def _input_cases(path: Path) -> set[str]:
    return {
        Path(line.strip()).expanduser().resolve().stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _video_map(root: Path, cases: set[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in root.rglob("*.mp4"):
        if path.stem not in cases:
            continue
        if path.stem in output:
            raise RuntimeError(f"Duplicate video for {path.stem} under {root}")
        output[path.stem] = path
    return output


def _probe(path: Path) -> str:
    return subprocess.check_output(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()


def _validate_job(
    root: Path,
    *,
    cases: set[str],
    role: str,
    target_count: int,
    report_sha256: str,
) -> dict[str, str]:
    videos = _video_map(root, cases)
    if set(videos) != cases:
        raise RuntimeError(f"Found {len(videos)}/{len(cases)} expected videos")
    for case, video in videos.items():
        if _probe(video) != "896,512,49":
            raise RuntimeError(f"Unexpected video shape/frame count: {video}")
        sidecar = video.with_suffix(".json")
        if not sidecar.is_file():
            raise RuntimeError(f"Missing sidecar for {video}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata = payload.get("dit_ablation") or payload.get("physrvg_ablation")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Missing ablation metadata in {sidecar}")
        selection = metadata.get("target_selection", {})
        if (
            metadata.get("mode") != "self_attn_grouped_head_zero"
            or metadata.get("category") != role
            or int(metadata.get("num_targets", -1)) != target_count
            or selection.get("sha256") != report_sha256
            or metadata.get("target_forward_call_count_ok") is not True
        ):
            raise RuntimeError(f"Invalid ablation metadata in {sidecar}")
    return {case: str(path) for case, path in sorted(videos.items())}


def _memory_used(gpu: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def _wait_for_gpu(gpu: int, threshold: int, poll_seconds: int) -> None:
    while True:
        used = _memory_used(gpu)
        if used <= threshold:
            print(f"[common22-worker] GPU{gpu} available: {used} MiB", flush=True)
            return
        print(f"[common22-worker] GPU{gpu} busy: {used} MiB; waiting", flush=True)
        time.sleep(poll_seconds)


def _claim_live(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("host") != socket.gethostname():
            return True
        os.kill(int(payload["pid"]), 0)
        return True
    except (FileNotFoundError, ProcessLookupError, KeyError, ValueError, json.JSONDecodeError):
        return False
    except PermissionError:
        return True


def _try_claim(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if _claim_live(path):
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


def _attempts(path: Path) -> int:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("attempt", 0))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return 0


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    input_list = Path(config["input_list"]).expanduser().resolve()
    report = Path(config["public_head_report"]).expanduser().resolve()
    cases = _input_cases(input_list)
    if len(cases) != int(config["expected_cases"]):
        raise ValueError(f"Expected {config['expected_cases']} cases, found {len(cases)}")
    targets, source = load_public_head_targets(report)
    if source["sha256"] != config["public_head_report_sha256"]:
        raise ValueError("Public Head report SHA256 differs from frozen config")

    roles = [str(value) for value in config["roles"]]
    models = [str(value) for value in config["models"]]
    seeds = [int(value) for value in config["seeds"]]
    jobs = [
        (model, seed, role)
        for seed in seeds
        for role in ("T", "C", "P", "G", "S")
        for model in models
        if role in roles
    ]
    threshold = int(config["execution"]["gpu_start_memory_threshold_mib"])
    poll_seconds = int(config["execution"]["poll_seconds"])
    max_attempts = int(config["execution"]["max_attempts_per_job"])

    for model, seed, role in jobs:
        label = f"{model}_seed-{seed:06d}_role-{role}"
        job_root = root / "generated" / model / f"seed-{seed:06d}" / f"role-{role}"
        state_path = root / "state" / model / f"seed-{seed:06d}" / f"role-{role}.json"
        claim_path = root / "claims" / f"{label}.json"
        try:
            if state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("status") == "complete":
                    _validate_job(
                        job_root,
                        cases=cases,
                        role=role,
                        target_count=len(targets[role]),
                        report_sha256=str(source["sha256"]),
                    )
                    continue
        except Exception as error:
            print(f"[common22-worker] revalidate failed {label}: {error}", flush=True)

        attempt = _attempts(state_path) + 1
        if attempt > max_attempts:
            continue
        if not _try_claim(
            claim_path,
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "worker_id": args.worker_id,
                "gpu": args.gpu,
                "job": label,
                "claimed_at_unix": time.time(),
            },
        ):
            continue
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
                    "target_count": len(targets[role]),
                    "gpu": args.gpu,
                    "attempt": attempt,
                    "started_at_unix": time.time(),
                },
            )
            log_path = root / "logs" / model / f"seed-{seed:06d}" / f"role-{role}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "MODEL": model,
                    "SEED": str(seed),
                    "ROLE": role,
                    "GPU": str(args.gpu),
                    "INPUT_LIST": str(input_list),
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
            videos = _validate_job(
                job_root,
                cases=cases,
                role=role,
                target_count=len(targets[role]),
                report_sha256=str(source["sha256"]),
            )
            _atomic_json(
                state_path,
                {
                    "status": "complete",
                    "model": model,
                    "seed": seed,
                    "role": role,
                    "target_count": len(targets[role]),
                    "gpu": args.gpu,
                    "attempt": attempt,
                    "completed_at_unix": time.time(),
                    "videos": videos,
                },
            )
            print(f"[common22-worker] complete {label}", flush=True)
        except Exception as error:
            _atomic_json(
                state_path,
                {
                    "status": "failed",
                    "model": model,
                    "seed": seed,
                    "role": role,
                    "target_count": len(targets[role]),
                    "gpu": args.gpu,
                    "attempt": attempt,
                    "error": repr(error),
                    "failed_at_unix": time.time(),
                },
            )
            print(f"[common22-worker] failed {label}: {error!r}", flush=True)
        finally:
            claim_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
