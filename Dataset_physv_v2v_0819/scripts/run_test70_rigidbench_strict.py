#!/usr/bin/env python3
"""Evaluate existing PhysV test70 videos with strict CYCLES RigidBench-style GT.

This adapter deliberately evaluates the available 49-frame/30-FPS test70
outputs without running ``prepare.py``: the official RigidBench preparation
requires a full 2-second clip at 24 FPS, while these outputs are the local
test70 protocol (about 1.6 seconds at native CYCLES 30 FPS).  The report is
therefore explicitly marked non-official/partial.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
STAGE_ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/staging")
REPORT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/runs")
RESULT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_test70_no_event_timing_40step/results")
RIGIDBENCH_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench")
SINGLE_RUNNER = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_cycles_gt_rigidbench_tracker_eval.py")
FPS = 30


def load_single_runner():
    spec = importlib.util.spec_from_file_location("cycles_rigidbench_single", SINGLE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SINGLE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Hydra's sam2 config module is rooted at the installed package; keep the
    # same config name used by Meta's checkpoint mapping.
    module.SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    return module


def strict_case_ids() -> list[str]:
    root = STRICT_ROOT / "truth" / "cases"
    return sorted(p.name for p in root.iterdir() if (p / "rigidbench" / "metadata.json").is_file())


def ensure_staging(ids: list[str]) -> Path:
    dataset = STAGE_ROOT / "rigidbench_dataset"
    samples = dataset / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for sample_id in ids:
        destination = samples / sample_id
        source = STRICT_ROOT / "truth" / "cases" / sample_id / "rigidbench"
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination.is_symlink() or destination.is_file():
            if destination.resolve() != source.resolve():
                destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=True)
    return dataset


def output_video(task_id: str, sample_id: str) -> Path | None:
    directory = RESULT_ROOT / task_id
    candidates = sorted(directory.glob(f"{sample_id}.mp4"))
    if candidates:
        return candidates[0]
    # The test70 writer has also used a nested sample directory in older runs.
    candidates = sorted(directory.glob(f"**/{sample_id}.mp4"))
    return candidates[0] if candidates else None


def link_or_copy_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.is_file():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        destination.unlink()
    destination.symlink_to(source)


def write_metadata(report_dir: Path, task_id: str, evaluated: list[str], missing: list[str], aggregate: dict) -> None:
    strict_ids = set(strict_case_ids())
    re_rendered = sorted(strict_ids & {
        "scene_door_frame_w038", "scene_door_frame_w046", "scene_door_frame_w054",
        "scene_door_frame_w062", "scene_door_frame_w074", "scene_puck_barrier_n030",
        "scene_puck_barrier_n045", "scene_puck_barrier_n060", "scene_puck_barrier_n075",
        "scene_puck_barrier_n090",
    })
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "rigidbench-style-local-test70-strict-cycles",
        "official": False,
        "official_protocol_note": "Not an official RigidBench score: local test70 outputs are 49 frames at native 30 FPS (~1.6 s), while official RigidBench uses 49 reference frames at 24 FPS from a 2.0 s clip.",
        "gt_dataset": str(STRICT_ROOT),
        "gt_video": "rgb_cycles.mp4 aligned strict CYCLES GT; evaluator adapter uses rigidbench/video.mp4 symlink",
        "fps": FPS,
        "window_frames": 49,
        "window_seconds": 49 / FPS,
        "resolution": [896, 512],
        "task_id": task_id,
        "evaluated_case_count": len(evaluated),
        "missing_case_count": len(missing),
        "evaluated_cases": evaluated,
        "missing_cases": missing,
        "strict_reference_exact_cases": sorted(set(evaluated) - set(re_rendered)),
        "strict_reference_rerendered_cases": sorted(set(evaluated) & set(re_rendered)),
        "aggregated": aggregate,
    }
    (report_dir / "strict_cycles_test70.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def evaluate(args: argparse.Namespace) -> int:
    if args.task_id == "":
        raise ValueError("--task-id cannot be empty")
    ids = strict_case_ids()
    dataset = ensure_staging(ids)
    report_dir = REPORT_ROOT / args.task_id
    generated = report_dir / "generated"
    evaluated: list[str] = []
    missing: list[str] = []
    for sample_id in ids:
        source = output_video(args.task_id, sample_id)
        if source is None:
            missing.append(sample_id)
            continue
        link_or_copy_video(source, generated / f"{sample_id}.mp4")
        evaluated.append(sample_id)
    if args.max_samples:
        evaluated = evaluated[: args.max_samples]
    if not evaluated:
        raise RuntimeError(f"No generated videos found for {args.task_id} under {RESULT_ROOT / args.task_id}")

    # The tracker/scorer implementation is imported only after the adapter is ready.
    sys.path.insert(0, str(RIGIDBENCH_ROOT / "src"))
    sys.path.insert(0, str(RIGIDBENCH_ROOT / "vendor" / "Video-Depth-Anything"))
    runner = load_single_runner()
    # RigidBench trackers consume prepared frame directories rather than MP4
    # files.  Keep the native CYCLES decode (30 FPS, no resize or resampling).
    for sample_id in evaluated:
        video = generated / f"{sample_id}.mp4"
        frame_dir = generated / sample_id
        frame_count = runner.extract_native_frames(video, frame_dir)
        if frame_count < 49:
            raise RuntimeError(f"{video} decoded to only {frame_count} frames; expected at least 49")
    runner.patch_local_trackers()
    import rigidbench.eval.score.context as score_context
    score_context.GT_FPS = FPS
    from rigidbench.eval.run import run_eval

    print(json.dumps({
        "task_id": args.task_id,
        "dataset": str(dataset),
        "generated": len(evaluated),
        "missing": len(missing),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
        "fps": FPS,
        "window_frames": 49,
    }, ensure_ascii=False), flush=True)
    aggregate = run_eval(
        args.task_id,
        str(dataset),
        str(REPORT_ROOT),
        split="eval",
        sample_ids=evaluated,
        force=args.force,
        generated_fps=FPS,
    )
    write_metadata(REPORT_ROOT / args.task_id, args.task_id, evaluated, missing, aggregate)
    print("STRICT_RIGIDBENCH_RESULT=" + json.dumps(aggregate, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
