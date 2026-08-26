#!/usr/bin/env python3
"""Backfill missing strict-test70 metrics one metric at a time.

The runner owns directory scanning and JSON updates.  The single-case modules
under ``physv_eval.single_case_rigidbench`` only receive metric inputs.
"""

from __future__ import annotations

import argparse
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


def metric_inputs_ready(task: Path, sample_id: str, metric: str) -> bool:
    generated = task / "generated" / sample_id
    frames = bool(list(generated.glob("*.jpg")) or list(generated.glob("*.png")))
    mask = (task / "masks" / sample_id / "mask.npz").is_file()
    tracks = (task / "tracks" / sample_id / "tracks.npz").is_file() and (task / "tracks" / sample_id / "gt_tracks.npz").is_file()
    depth = (task / "depth" / sample_id / "depth.npz").is_file()
    if metric in {"iou", "l2", "chamfer"}:
        return mask
    if metric == "ate":
        return tracks
    if metric == "si_mse":
        return depth
    if metric in {"ssim", "lpips"}:
        return frames
    if metric == "ate3d":
        return tracks and depth
    if metric == "iddrift":
        return frames and tracks
    if metric == "bgdrift":
        return frames and mask
    return False


def sample_dir(strict_root: Path, sample_id: str) -> Path:
    return strict_root / "truth" / "cases" / sample_id / "rigidbench"


def load_tracks(task: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_path = task / "tracks" / sample_id / "tracks.npz"
    gt_path = task / "tracks" / sample_id / "gt_tracks.npz"
    with np.load(pred_path, allow_pickle=False) as pred, np.load(gt_path, allow_pickle=False) as gt:
        pred_tracks = pred["tracks"]
        gt_tracks = gt["tracks"]
        pred_vis = pred["visibility"] if "visibility" in pred.files else np.ones(pred_tracks.shape[:2], dtype=bool)
        gt_vis = gt["visibility"] if "visibility" in gt.files else np.ones(gt_tracks.shape[:2], dtype=bool)
        offsets = pred["actor_offsets"]
    T = min(gt_tracks.shape[1], pred_tracks.shape[1])
    return gt_tracks[:, :T], pred_tracks[:, :T], pred_vis[:, :T] & gt_vis[:, :T], offsets


def load_frames(task: Path, sample_id: str, strict_root: Path) -> tuple[np.ndarray, np.ndarray]:
    gt = load_video_rgb(sample_dir(strict_root, sample_id) / "video.mp4")
    pred = generated_frames(task / "generated" / sample_id)
    return gt[: min(len(gt), len(pred))], pred[: min(len(gt), len(pred))]


def load_ate3d_inputs(task: Path, sample_id: str, strict_root: Path):
    from rigidbench.eval.score.depth import affine_align_disparity
    from rigidbench.eval.score.trajectory import quat_wxyz_to_rotmat, reconstruct_centroids

    case = sample_dir(strict_root, sample_id)
    metadata = read_json(case / "metadata.json")
    gt_tracks, pred_tracks, visibility, offsets = load_tracks(task, sample_id)
    gt_depth = load_npz_array(case / "depth.npz", "depth")
    pred_depth = load_npz_array(task / "depth" / sample_id / "depth.npz", "depth")
    T = min(len(gt_depth), len(pred_depth), pred_tracks.shape[1])
    gt_depth, pred_depth = gt_depth[:T], pred_depth[:T]
    aligned, _, _ = affine_align_disparity(pred_depth, gt_depth)
    camera = metadata["camera"]
    intrinsics = camera["intrinsics"]
    extrinsics = camera["extrinsics"]
    pred_centroids = reconstruct_centroids(
        pred_tracks[:, :T], visibility[:, :T], aligned,
        intrinsics,
        np.asarray(extrinsics["location"], dtype=np.float64),
        quat_wxyz_to_rotmat(np.asarray(extrinsics["rotation"], dtype=np.float64)),
        offsets,
    )
    with np.load(case / "trajectories.npz", allow_pickle=False) as data:
        gt_trajectories = {key: data[key] for key in data.files}
    actors = [name for name, info in metadata.get("actors", {}).items() if info.get("role") == "active"]
    return pred_centroids, gt_trajectories, actors


def load_shared_model(metric: str, device: str):
    if metric == "lpips":
        import lpips as lpips_pkg

        return lpips_pkg.LPIPS(net="alex").to(device).eval()
    if metric == "iddrift":
        import torch

        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()
    if metric == "bgdrift":
        import torch

        return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    return None


def compute(metric: str, task: Path, sample_id: str, strict_root: Path, shared_model, device: str) -> dict:
    case = sample_dir(strict_root, sample_id)
    if metric in {"iou", "l2", "chamfer"}:
        from physv_eval.single_case_rigidbench import chamfer, iou, l2

        gt = load_npz_array(case / "masks.npz", "masks", "mask")
        pred = load_npz_array(task / "masks" / sample_id / "mask.npz", "masks", "mask")
        T = min(len(gt), len(pred))
        gt, pred = gt[:T], pred[:T]
        if metric == "iou":
            return iou.score_case(gt, pred)
        if metric == "l2":
            return l2.score_case(gt, pred)
        return chamfer.score_case(gt, pred)
    if metric == "ate":
        from physv_eval.single_case_rigidbench import ate

        gt, pred, visibility, _ = load_tracks(task, sample_id)
        height = int(read_json(case / "metadata.json")["camera"]["intrinsics"]["height"])
        return ate.score_case(gt, pred, height, visibility)
    if metric == "si_mse":
        from physv_eval.single_case_rigidbench import si_mse

        gt = load_npz_array(case / "depth.npz", "depth")
        pred = load_npz_array(task / "depth" / sample_id / "depth.npz", "depth")
        T = min(len(gt), len(pred))
        return si_mse.score_case(
            gt[:T], pred[:T],
        )
    if metric in {"ssim", "lpips"}:
        from physv_eval.single_case_rigidbench import lpips, ssim

        gt_frames, pred_frames = load_frames(task, sample_id, strict_root)
        if metric == "ssim":
            return ssim.score_case(gt_frames, pred_frames, device)
        return lpips.score_case(gt_frames, pred_frames, shared_model, device)
    if metric == "ate3d":
        from physv_eval.single_case_rigidbench import ate3d

        return ate3d.score_case(*load_ate3d_inputs(task, sample_id, strict_root))
    if metric == "iddrift":
        from physv_eval.single_case_rigidbench import iddrift

        gt_frames, pred_frames = load_frames(task, sample_id, strict_root)
        gt_tracks, pred_tracks, visibility, offsets = load_tracks(task, sample_id)
        return iddrift.score_case(gt_frames, pred_frames, gt_tracks, pred_tracks, visibility, offsets, shared_model, device)
    if metric == "bgdrift":
        from physv_eval.single_case_rigidbench import bgdrift

        pred_frames = generated_frames(task / "generated" / sample_id)
        pred_mask = load_npz_array(task / "masks" / sample_id / "mask.npz", "masks", "mask")
        return bgdrift.score_case(pred_frames, pred_mask, shared_model, device)
    raise KeyError(metric)


def update_case(task: Path, sample_id: str, result: dict) -> None:
    path = task_sample_json(task, sample_id)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
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
            and metric_inputs_ready(task, sample_id, metric)
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
