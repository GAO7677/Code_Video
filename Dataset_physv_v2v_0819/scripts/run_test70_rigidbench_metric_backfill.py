#!/usr/bin/env python3
"""Backfill missing strict-test70 metrics one metric at a time.

The runner owns directory scanning and JSON updates.  The single-case modules
under ``physv_eval.single_case_rigidbench`` only receive metric inputs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PHYSV_EVAL_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
RIGIDBENCH_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench")
DEFAULT_INPUT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70")
DEFAULT_STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
BUILDER = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_metrics.py")
METRICS = ("iou", "l2", "chamfer", "ate", "si_mse", "lpips", "ssim", "ate3d", "iddrift", "bgdrift")

sys.path.insert(0, str(PHYSV_EVAL_ROOT))
sys.path.insert(0, str(RIGIDBENCH_ROOT / "src"))
sys.path.insert(0, str(RIGIDBENCH_ROOT / "vendor" / "Video-Depth-Anything"))

from physv_eval.single_case_rigidbench.common import load_npz_array, load_video_rgb
from physv_eval.single_case_rigidbench.prediction import (
    active_actor_indices,
    concatenate_gt_tracks,
    load_cotracker_model,
    load_dinov2_model,
    load_sam2_model,
    load_vda_model,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **payload)
        generated = temporary.with_suffix(temporary.suffix + ".npz")
        generated.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(temporary.suffix + ".npz").unlink(missing_ok=True)


def strict_ids(strict_root: Path) -> list[str]:
    root = strict_root / "truth" / "cases"
    return sorted(p.name for p in root.iterdir() if (p / "rigidbench" / "metadata.json").is_file())


def task_dirs(input_root: Path, requested: list[str] | None) -> list[Path]:
    runs = input_root / "runs" if (input_root / "runs").is_dir() else input_root
    candidates = sorted(p for p in runs.iterdir() if p.is_dir() and (p / "generated").is_dir())
    if requested:
        wanted = set(requested)
        candidates = [p for p in candidates if p.name in wanted]
    return candidates


def generated_frames(path: Path) -> np.ndarray:
    files = sorted(path.glob("*.jpg")) or sorted(path.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No generated frame jpg/png under {path}")
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in files])


def task_sample_json(task: Path, sample_id: str) -> Path:
    return task / "metrics" / f"{sample_id}.json"


def metric_is_missing(path: Path, metric: str) -> bool:
    payload = read_json(path)
    value = payload.get(metric)
    return value is None or (isinstance(value, float) and not np.isfinite(value))


def metric_inputs_ready(task: Path, sample_id: str, metric: str, strict_root: Path) -> bool:
    generated = task / "generated" / sample_id
    frames = bool(list(generated.glob("*.jpg")) or list(generated.glob("*.png")))
    case = sample_dir(strict_root, sample_id)
    mask = (case / "masks.npz").is_file() and (case / "metadata.json").is_file()
    tracks = mask and (case / "depth.npz").is_file() and (case / "trajectories.npz").is_file()
    depth = (case / "depth.npz").is_file()
    if metric in {"iou", "l2", "chamfer"}:
        return frames and mask
    if metric == "ate":
        return frames and tracks
    if metric == "si_mse":
        return frames and depth
    if metric in {"ssim", "lpips"}:
        return frames
    if metric == "ate3d":
        return frames and tracks
    if metric == "iddrift":
        return frames and tracks
    if metric == "bgdrift":
        return frames and mask
    return False


def sample_dir(strict_root: Path, sample_id: str) -> Path:
    return strict_root / "truth" / "cases" / sample_id / "rigidbench"


def load_gt_track_bundle(strict_root: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    case = sample_dir(strict_root, sample_id)
    metadata = read_json(case / "metadata.json")
    return concatenate_gt_tracks(case, metadata)


def load_frames(task: Path, sample_id: str, strict_root: Path) -> tuple[np.ndarray, np.ndarray]:
    gt = load_video_rgb(sample_dir(strict_root, sample_id) / "video.mp4")
    pred = load_video_rgb(task / "generated" / sample_id)
    return gt[: min(len(gt), len(pred))], pred[: min(len(gt), len(pred))]


def load_shared_model(metric: str, device: str):
    if metric == "lpips":
        import lpips as lpips_pkg

        return lpips_pkg.LPIPS(net="alex").to(device).eval()
    if metric in {"iou", "l2", "chamfer"}:
        return load_sam2_model(device)
    if metric == "ate":
        return load_cotracker_model(device)
    if metric == "si_mse":
        return load_vda_model(device)
    if metric == "ate3d":
        return {"vda": load_vda_model(device), "cotracker": load_cotracker_model(device)}
    if metric == "iddrift":
        return {"dino": load_dinov2_model(device), "cotracker": load_cotracker_model(device)}
    if metric == "bgdrift":
        return {"sam2": load_sam2_model(device), "cotracker": load_cotracker_model(device)}
    return None


def compute(metric: str, task: Path, sample_id: str, strict_root: Path, shared_model, device: str) -> dict:
    case = sample_dir(strict_root, sample_id)
    if metric in {"iou", "l2", "chamfer"}:
        from physv_eval.single_case_rigidbench import chamfer, iou, l2

        gt = load_npz_array(case / "masks.npz", "masks", "mask")
        metadata = read_json(case / "metadata.json")
        active = active_actor_indices(case / "masks.npz", metadata)
        pred_video = task / "generated" / sample_id
        if metric == "iou":
            return iou.score_case(gt, pred_video, shared_model, active)
        if metric == "l2":
            return l2.score_case(gt, pred_video, shared_model, active)
        return chamfer.score_case(gt, pred_video, shared_model, active)
    if metric == "ate":
        from physv_eval.single_case_rigidbench import ate

        gt, visibility, _offsets, _actors = load_gt_track_bundle(strict_root, sample_id)
        height = int(read_json(case / "metadata.json")["camera"]["intrinsics"]["height"])
        return ate.score_case(gt, task / "generated" / sample_id, height, shared_model, visibility)
    if metric == "si_mse":
        from physv_eval.single_case_rigidbench import si_mse

        gt = load_npz_array(case / "depth.npz", "depth")
        return si_mse.score_case(gt, task / "generated" / sample_id, shared_model, device)
    if metric in {"ssim", "lpips"}:
        from physv_eval.single_case_rigidbench import lpips, ssim

        gt_frames, pred_frames = load_frames(task, sample_id, strict_root)
        if metric == "ssim":
            return ssim.score_case(gt_frames, pred_frames, device)
        return lpips.score_case(gt_frames, pred_frames, shared_model, device)
    if metric == "ate3d":
        from physv_eval.single_case_rigidbench import ate3d

        metadata = read_json(case / "metadata.json")
        gt_tracks, _visibility, offsets, actors = load_gt_track_bundle(strict_root, sample_id)
        gt_depth = load_npz_array(case / "depth.npz", "depth")
        with np.load(case / "trajectories.npz", allow_pickle=False) as data:
            gt_trajectories = {key: data[key] for key in data.files}
        return ate3d.score_case(
            task / "generated" / sample_id,
            gt_tracks,
            gt_depth,
            gt_trajectories,
            actors,
            metadata["camera"],
            offsets,
            shared_model["vda"],
            shared_model["cotracker"],
            device,
        )
    if metric == "iddrift":
        from physv_eval.single_case_rigidbench import iddrift

        gt_frames, _pred_frames = load_frames(task, sample_id, strict_root)
        gt_tracks, visibility, offsets, _actors = load_gt_track_bundle(strict_root, sample_id)
        return iddrift.score_case(
            gt_frames,
            task / "generated" / sample_id,
            gt_tracks,
            visibility,
            offsets,
            shared_model["dino"],
            shared_model["cotracker"],
            device,
        )
    if metric == "bgdrift":
        from physv_eval.single_case_rigidbench import bgdrift

        gt_mask = load_npz_array(case / "masks.npz", "masks", "mask")
        metadata = read_json(case / "metadata.json")
        active = active_actor_indices(case / "masks.npz", metadata)
        return bgdrift.score_case(
            task / "generated" / sample_id,
            gt_mask,
            shared_model["sam2"],
            shared_model["cotracker"],
            active,
            device,
        )
    raise KeyError(metric)


def update_case(task: Path, sample_id: str, result: dict) -> None:
    path = task_sample_json(task, sample_id)
    lock_path = task / "metrics" / ".locks" / f"{sample_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        payload = read_json(path)
        payload.setdefault("sample_id", sample_id)
        for key, value in result.items():
            if key == "per_frame" or isinstance(value, (np.ndarray, list, dict)):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            payload[key] = value
        atomic_json(path, payload)
        per_frame = result.get("per_frame")
        if per_frame is not None:
            frame_path = task / "metrics_per_frame" / f"{sample_id}.npz"
            existing: dict[str, np.ndarray] = {}
            if frame_path.is_file():
                with np.load(frame_path, allow_pickle=False) as data:
                    existing = {key: data[key] for key in data.files}
            metric_key = next(key for key in result if key != "per_frame")
            existing[metric_key] = np.asarray(per_frame)
            atomic_npz(frame_path, existing)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def build_snapshot() -> None:
    subprocess.run([sys.executable, str(BUILDER)], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill missing strict-test70 RigidBench metrics")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--task-id", action="append", help="Restrict to one or more task directory names")
    parser.add_argument("--exclude-task-id", action="append", default=[], help="Exclude task directory names, e.g. active generation tasks")
    parser.add_argument("--case-id", action="append", help="Restrict to one or more case IDs; useful for disjoint workers")
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--metric", choices=METRICS, help="Run exactly one metric in this process")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    metrics = [args.metric] if args.metric else [name.strip() for name in args.metrics.split(",") if name.strip()]
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise SystemExit(f"Unknown metrics: {unknown}; choose from {METRICS}")
    ids = strict_ids(args.strict_root)
    if args.case_id:
        requested_cases = set(args.case_id)
        unknown_cases = sorted(requested_cases - set(ids))
        if unknown_cases:
            raise SystemExit(f"Unknown case IDs: {unknown_cases[:10]}")
        ids = [sample_id for sample_id in ids if sample_id in requested_cases]
    tasks = task_dirs(args.input_root, args.task_id)
    excluded = set(args.exclude_task_id)
    tasks = [task for task in tasks if task.name not in excluded]
    print(json.dumps({"tasks": len(tasks), "cases": len(ids), "metrics": metrics, "device": args.device}, ensure_ascii=False), flush=True)
    for metric in metrics:
        pending = [
            (task, sample_id)
            for task in tasks
            for sample_id in ids
            if metric_is_missing(task_sample_json(task, sample_id), metric)
            and metric_inputs_ready(task, sample_id, metric, args.strict_root)
        ]
        print(f"[backfill] metric={metric} pending={len(pending)}", flush=True)
        if args.dry_run or not pending:
            continue
        shared_model = load_shared_model(metric, args.device)
        for task, sample_id in pending:
            try:
                result = compute(metric, task, sample_id, args.strict_root, shared_model, args.device)
                update_case(task, sample_id, result)
                print(f"[backfill] metric={metric} task={task.name} sample={sample_id} done", flush=True)
            except Exception as exc:
                print(f"[backfill] metric={metric} task={task.name} sample={sample_id} failed: {exc}", flush=True)
        if not args.no_build:
            build_snapshot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
