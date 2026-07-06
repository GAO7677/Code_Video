#!/usr/bin/env python3
"""
Probe OpenVid smoke-train capacity on Wan2.1-1.3B using GPUs 5 and 6.

The script runs 1-step training jobs under two sweeps:
1. minimal_context: keeps context close to 1 frame to isolate full-video length.
2. max_context: uses the largest allowed prefix context under max_context_ratio <= 0.5.

Results are written to a JSON summary and per-case logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ACCELERATE = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate")
DEFAULT_TRAIN_SCRIPT = THIS_DIR / "train_v_newtrain.py"
DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train")
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B")
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/train0706_wan21_13b_capacity_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Wan2.1-1.3B smoke-train frame capacity on OpenVid.")
    parser.add_argument("--gpu-set", type=str, default="5,6")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--accelerate-bin", type=Path, default=DEFAULT_ACCELERATE)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--candidate-num-frames", type=str, default="25,49,73,97,121,145,169")
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--keep-run-artifacts", action="store_true")
    return parser.parse_args()


def parse_candidates(raw: str) -> list[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"num_frames must be positive, got {value}")
        if (value - 1) % 4 != 0:
            raise ValueError(f"num_frames must satisfy 4n+1 for Wan2.1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("candidate-num-frames must contain at least one value")
    return values


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def exact_ratio_for_context(num_frames: int, target_context_frames: int) -> float:
    if target_context_frames < 1:
        raise ValueError("target_context_frames must be >= 1")
    upper = min(0.5, (target_context_frames + 0.49) / float(num_frames))
    lower = target_context_frames / float(num_frames)
    ratio = min(upper, max(lower + 1e-4, lower + (upper - lower) * 0.5))
    if int(num_frames * ratio) != target_context_frames:
        ratio = upper
    if int(num_frames * ratio) != target_context_frames:
        raise ValueError(
            f"Could not build exact ratio for num_frames={num_frames}, target_context_frames={target_context_frames}"
        )
    return float(ratio)


def build_cases(candidates: list[int]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for num_frames in candidates:
        max_context = min(num_frames - 1, int(num_frames * 0.5))
        cases.append(
            {
                "sweep": "minimal_context",
                "num_frames": int(num_frames),
                "target_context_frames": 1,
                "max_context_ratio": exact_ratio_for_context(num_frames, 1),
                "stop_after_failure_in_sweep": True,
            }
        )
        cases.append(
            {
                "sweep": "max_context",
                "num_frames": int(num_frames),
                "target_context_frames": int(max_context),
                "max_context_ratio": 0.5,
                "stop_after_failure_in_sweep": True,
            }
        )
    return cases


def run_case(args: argparse.Namespace, case: dict[str, Any], work_root: Path) -> dict[str, Any]:
    tag = f"{case['sweep']}__nf{int(case['num_frames']):03d}__ctx{int(case['target_context_frames']):03d}"
    run_dir = work_root / tag
    log_path = work_root / f"{tag}.log"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "env",
        f"PYTHONPATH={args.project_root}:{args.diffsynth_root}",
        f"CUDA_VISIBLE_DEVICES={args.gpu_set}",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        str(args.accelerate_bin),
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(int(args.num_processes)),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        str(args.train_script),
        "--diffsynth_root",
        str(args.diffsynth_root),
        "--wan_root",
        str(args.wan_root),
        "--dataset_base_path",
        str(args.dataset_root),
        "--dataset_metadata_path",
        "",
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num_frames",
        str(int(case["num_frames"])),
        "--max_train_steps",
        str(int(args.max_train_steps)),
        "--context_sampling_profile",
        "legacy_prefix",
        "--min_context_frames",
        str(int(case["target_context_frames"])),
        "--max_context_ratio",
        f"{float(case['max_context_ratio']):.8f}",
        "--dataset_repeat",
        "1",
        "--dataset_num_workers",
        "0",
        "--learning_rate",
        "1e-4",
        "--weight_decay",
        "0.01",
        "--num_epochs",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--save_steps",
        "1000",
        "--remove_prefix_in_ckpt",
        "pipe.dit.",
        "--output_path",
        str(run_dir),
        "--lora_base_model",
        "dit",
        "--lora_target_modules",
        "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank",
        "32",
        "--report_to",
        "none",
    ]

    started_at = time.time()
    status = "unknown"
    failure_kind = None
    tail_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log_handle.flush()
        process = subprocess.run(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(args.timeout_seconds),
        )
    elapsed = round(time.time() - started_at, 3)

    try:
        tail_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    except FileNotFoundError:
        tail_lines = []
    joined_tail = "\n".join(tail_lines).lower()

    if process.returncode == 0:
        status = "success"
    else:
        status = "failed"
        if "out of memory" in joined_tail or "cuda error: out of memory" in joined_tail:
            failure_kind = "oom"
        elif "signal 9" in joined_tail or "sigkill" in joined_tail or "killed" in joined_tail:
            failure_kind = "killed"
        else:
            failure_kind = "other"

    result = {
        **case,
        "tag": tag,
        "status": status,
        "failure_kind": failure_kind,
        "returncode": int(process.returncode),
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "tail_lines": tail_lines,
    }

    if status == "success" and not args.keep_run_artifacts:
        shutil.rmtree(run_dir, ignore_errors=True)
        result["run_dir_removed"] = True
    else:
        result["run_dir_removed"] = False
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"results": results, "by_sweep": {}}
    by_sweep: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_sweep.setdefault(str(item["sweep"]), []).append(item)

    for sweep_name, sweep_results in by_sweep.items():
        successes = [item for item in sweep_results if item["status"] == "success"]
        failures = [item for item in sweep_results if item["status"] != "success"]
        best = None
        if successes:
            best = max(successes, key=lambda item: int(item["num_frames"]))
        summary["by_sweep"][sweep_name] = {
            "num_cases": len(sweep_results),
            "num_success": len(successes),
            "num_failure": len(failures),
            "best_success": best,
            "first_failure": failures[0] if failures else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    if ",4," in f",{args.gpu_set},":
        raise ValueError(f"gpu4 is faulty and cannot be used, got gpu-set={args.gpu_set}")
    candidates = parse_candidates(args.candidate_num_frames)

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "probe_plan.json",
        {
            "gpu_set": args.gpu_set,
            "num_processes": int(args.num_processes),
            "dataset_root": str(args.dataset_root),
            "candidate_num_frames": candidates,
        },
    )

    results: list[dict[str, Any]] = []
    stopped_sweeps: set[str] = set()
    for case in build_cases(candidates):
        if case["sweep"] in stopped_sweeps:
            continue
        print(
            f"[probe] sweep={case['sweep']} num_frames={case['num_frames']} "
            f"target_context_frames={case['target_context_frames']} ratio={case['max_context_ratio']:.6f}"
        )
        result = run_case(args=args, case=case, work_root=output_root / "runs")
        results.append(result)
        write_json(output_root / "latest_result.json", result)
        write_json(output_root / "results_partial.json", results)
        if result["status"] != "success" and case.get("stop_after_failure_in_sweep", False):
            stopped_sweeps.add(str(case["sweep"]))

    summary = summarize(results)
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
