#!/usr/bin/env python3
"""Evaluate every generated method on the test70 strict-CYCLES benchmark.

The source of prediction videos is the existing test70 visualization page;
the prediction side passed to the metric modules is therefore only a video.
All GT-side inputs are read from the strict CYCLES package.  Work is grouped
by expensive prediction model (SAM2, VDA, CoTracker/DINO, LPIPS), so one
worker computes all metrics that share the same extraction pass.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts")
EVAL_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
RIGIDBENCH_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench")
DEFAULT_DASHBOARD = Path(
    "/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/"
    "physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_all_methods"
)
DEFAULT_STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
METRICS = ("iou", "l2", "chamfer", "ate", "si_mse", "lpips", "ssim", "ate3d", "iddrift", "bgdrift")
GROUP_METRICS = {
    "mask": ("iou", "l2", "chamfer", "bgdrift"),
    "depth": ("ate3d", "si_mse"),
    "track": ("ate", "iddrift"),
    "image": ("lpips", "ssim"),
}

sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(RIGIDBENCH_ROOT / "src"))
sys.path.insert(0, str(RIGIDBENCH_ROOT / "vendor" / "Video-Depth-Anything"))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
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


def metric_missing(payload: dict[str, Any], metric: str) -> bool:
    value = payload.get(metric)
    if value is None:
        return True
    try:
        return not np.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def case_dir(strict_root: Path, case_id: str) -> Path:
    return strict_root / "truth" / "cases" / case_id / "rigidbench"


def metric_path(output_root: Path, task_id: str, case_id: str) -> Path:
    return output_root / "methods" / task_id / "metrics" / f"{case_id}.json"


def prediction_ready(path: str | Path) -> bool:
    """Return whether a prediction file is plausibly complete.

    The test70 dashboard can expose a path while inference is still writing it
    (or after a failed encode).  Treating ``Path.is_file()`` as completion lets
    one truncated MP4 abort a whole metric-family worker.  A cheap size check
    avoids opening thousands of videos during worker startup; the metric
    implementation performs the real decode when it evaluates a case.
    """
    path = Path(path)
    if not path.exists():
        return False
    if path.is_dir():
        return any(path.glob("*.jpg")) or any(path.glob("*.png"))
    try:
        if path.stat().st_size < 1024:
            return False
    except OSError:
        return False
    return True


def load_registry(
    dashboard_path: Path,
    strict_root: Path,
    output_root: Path,
    write: bool = False,
) -> dict[str, Any]:
    registry_path = output_root / "registry.json"
    if registry_path.is_file() and not write:
        registry = read_json(registry_path)
        # A running inference job may add videos after initialization. Refresh
        # only this cheap existence bit so --resume can pick them up later.
        for model in registry.get("models", []):
            for case in model.get("cases", []):
                case["prediction_exists"] = prediction_ready(case["video_path"])
        return registry

    dashboard = read_json(dashboard_path)
    records = dashboard.get("records", [])
    page_root = dashboard_path.parent
    record_map: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("task_id", "")), str(record.get("case_id", "")))
        if not key[0] or not key[1]:
            continue
        if key in record_map:
            raise ValueError(f"Duplicate dashboard record: {key}")
        record_map[key] = record

    models = []
    errors = []
    for model in dashboard.get("models", []):
        task_id = str(model.get("task_id", ""))
        cases = []
        for case_id in sorted({case for task, case in record_map if task == task_id}):
            record = record_map[(task_id, case_id)]
            video_url = str(record.get("video_url", ""))
            page_video_path = page_root / video_url
            result_root = model.get("result_root")
            canonical_video_path = (
                Path(str(result_root)) / Path(video_url).name
                if result_root
                else page_video_path
            )
            # Prefer the same overlay path served by the dashboard. If the
            # overlay link is absent, use the model's canonical output root.
            video_path = page_video_path if page_video_path.is_file() else canonical_video_path
            gt_video = case_dir(strict_root, case_id) / "video.mp4"
            if not gt_video.is_file():
                errors.append(f"missing strict GT video: {gt_video}")
            cases.append(
                {
                    "case_id": case_id,
                    "family_key": record.get("family_key"),
                    "video_path": str(video_path),
                    "video_url": video_url,
                    "prediction_exists": prediction_ready(video_path),
                    "gt_video": str(gt_video),
                }
            )
        if not cases:
            errors.append(f"dashboard model has no records: {task_id}")
        models.append(
            {
                "task_id": task_id,
                "model_key": model.get("model_key"),
                "label": model.get("label", task_id),
                "color": model.get("color"),
                "step": model.get("step"),
                "checkpoint_format": model.get("checkpoint_format"),
                "source_checkpoint": model.get("source_checkpoint"),
                "result_root": model.get("result_root"),
                "inference_steps": model.get("inference_steps"),
                "cases": cases,
            }
        )
    if errors:
        raise FileNotFoundError("Registry validation failed:\n" + "\n".join(errors[:30]))
    if len(models) != len(dashboard.get("models", [])):
        raise ValueError("Registry model count does not match dashboard")
    case_ids = sorted({case["case_id"] for model in models for case in model["cases"]})
    expected_model_count = len(dashboard.get("models", []))
    expected_case_count = int(dashboard.get("case_count", 70))
    if expected_model_count <= 0:
        raise ValueError("Dashboard has no models")
    if len(models) != expected_model_count or len(case_ids) != expected_case_count:
        raise ValueError(
            f"Expected {expected_model_count} models × {expected_case_count} cases, "
            f"got {len(models)} × {len(case_ids)}"
        )
    registry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dashboard": str(dashboard_path),
        "source_page_root": str(page_root),
        "strict_root": str(strict_root),
        "protocol": "rigidbench-style-local-test70-strict-cycles",
        "fps": 30,
        "resolution": [896, 512],
        "window_frames": 49,
        "metrics": list(METRICS),
        "models": models,
        "case_ids": case_ids,
    }
    atomic_json(registry_path, registry)
    return registry


def load_gt_track_bundle(strict_root: Path, case_id: str):
    from physv_eval.single_case_rigidbench.prediction import concatenate_gt_tracks

    case = case_dir(strict_root, case_id)
    metadata = read_json(case / "metadata.json")
    return concatenate_gt_tracks(case, metadata)


def load_gt_array(path: Path, *names: str) -> np.ndarray:
    from physv_eval.single_case_rigidbench.common import load_npz_array

    return load_npz_array(path, *names)


def compute_mask(case: dict[str, Any], needed: set[str], models, strict_root: Path, device: str) -> dict[str, Any]:
    from rigidbench.eval.score.mask import chamfer_per_frame, iou_per_frame, l2_per_frame
    from physv_eval.single_case_rigidbench import bgdrift
    from physv_eval.single_case_rigidbench.common import load_video_rgb
    from physv_eval.single_case_rigidbench.mask_metric_common import score_mask_metric
    from physv_eval.single_case_rigidbench.prediction import active_actor_indices, extract_masks

    gt_case = case_dir(strict_root, case["case_id"])
    pred_frames = load_video_rgb(case["video_path"])
    gt_mask = load_gt_array(gt_case / "masks.npz", "masks", "mask")
    metadata = read_json(gt_case / "metadata.json")
    active = active_actor_indices(gt_case / "masks.npz", metadata)
    pred_mask = extract_masks(case["video_path"], gt_mask, models["sam2"], active, frames=pred_frames)
    # The test70 predictions are 49 frames at the same 30 FPS as strict GT,
    # while strict CYCLES stores the complete 90-frame rollout. Match the
    # official evaluator's common-prefix behavior before calling the low-level
    # mask functions, which intentionally require identical shapes.
    T = min(gt_mask.shape[0], pred_mask.shape[0])
    gt_active = gt_mask[:T, active] if active else gt_mask[:T]
    pred_mask = pred_mask[:T]
    functions = {"iou": iou_per_frame, "l2": l2_per_frame, "chamfer": chamfer_per_frame}
    result: dict[str, Any] = {}
    per_frame: dict[str, np.ndarray] = {}
    for metric in ("iou", "l2", "chamfer"):
        if metric not in needed:
            continue
        scored = score_mask_metric(gt_active, pred_mask, functions[metric])
        result[metric] = scored["value"]
        per_frame[metric] = np.asarray(scored["per_frame"])
    if "bgdrift" in needed:
        result.update(bgdrift.score_from_frames_and_masks(pred_frames, pred_mask, models["cotracker"], device))
    if per_frame:
        result["_per_frame"] = per_frame
    return result


def compute_depth(case: dict[str, Any], needed: set[str], models, strict_root: Path, device: str) -> dict[str, Any]:
    from physv_eval.single_case_rigidbench import ate3d, si_mse
    from physv_eval.single_case_rigidbench.common import load_video_rgb
    from physv_eval.single_case_rigidbench.prediction import extract_disparity, extract_tracks

    gt_case = case_dir(strict_root, case["case_id"])
    pred_frames = load_video_rgb(case["video_path"])
    pred_disparity = extract_disparity(case["video_path"], models["vda"], device, frames=pred_frames)
    result: dict[str, Any] = {}
    if "si_mse" in needed:
        result.update(si_mse.score_from_disparity(load_gt_array(gt_case / "depth.npz", "depth"), pred_disparity))
    if "ate3d" in needed:
        metadata = read_json(gt_case / "metadata.json")
        gt_tracks, gt_visibility, offsets, actors = load_gt_track_bundle(strict_root, case["case_id"])
        gt_depth = load_gt_array(gt_case / "depth.npz", "depth")
        with np.load(gt_case / "trajectories.npz", allow_pickle=False) as data:
            gt_trajectories = {key: data[key] for key in data.files}
        pred_tracks, pred_visibility = extract_tracks(
            case["video_path"], gt_tracks, models["cotracker"], frames=pred_frames
        )
        result.update(
            ate3d.score_from_predictions(
                gt_tracks,
                gt_visibility,
                gt_depth,
                gt_trajectories,
                actors,
                metadata["camera"],
                offsets,
                pred_tracks,
                pred_visibility,
                pred_disparity,
            )
        )
    return result


def compute_track(case: dict[str, Any], needed: set[str], models, strict_root: Path, device: str) -> dict[str, Any]:
    from physv_eval.single_case_rigidbench import ate, iddrift
    from physv_eval.single_case_rigidbench.common import load_video_rgb
    from physv_eval.single_case_rigidbench.prediction import extract_tracks

    gt_case = case_dir(strict_root, case["case_id"])
    pred_frames = load_video_rgb(case["video_path"])
    gt_tracks, gt_visibility, offsets, _actors = load_gt_track_bundle(strict_root, case["case_id"])
    pred_tracks, pred_visibility = extract_tracks(
        case["video_path"], gt_tracks, models["cotracker"], frames=pred_frames
    )
    result: dict[str, Any] = {}
    if "ate" in needed:
        metadata = read_json(gt_case / "metadata.json")
        scored = ate.score_from_predictions(
            gt_tracks,
            pred_tracks,
            pred_visibility,
            int(metadata["camera"]["intrinsics"]["height"]),
            gt_visibility,
        )
        result.update({key: value for key, value in scored.items() if key != "per_frame"})
        result["_per_frame"] = {"ate": np.asarray(scored["per_frame"])}
    if "iddrift" in needed:
        gt_frames = load_video_rgb(case["gt_video"])
        result.update(
            iddrift.score_from_predictions(
                gt_frames,
                pred_frames,
                gt_tracks,
                pred_tracks,
                pred_visibility,
                gt_visibility,
                offsets,
                models["dino"],
                device,
            )
        )
    return result


def compute_image(case: dict[str, Any], needed: set[str], model, strict_root: Path, device: str) -> dict[str, Any]:
    from physv_eval.single_case_rigidbench import lpips, ssim
    from physv_eval.single_case_rigidbench.common import load_video_rgb

    del strict_root
    gt_frames = load_video_rgb(case["gt_video"])
    pred_frames = load_video_rgb(case["video_path"])
    result: dict[str, Any] = {}
    per_frame: dict[str, np.ndarray] = {}
    if "lpips" in needed:
        scored = lpips.score_case(gt_frames, pred_frames, model, device)
        result["lpips"] = scored["lpips"]
        per_frame["lpips"] = np.asarray(scored["per_frame"])
    if "ssim" in needed:
        scored = ssim.score_case(gt_frames, pred_frames, device)
        result["ssim"] = scored["ssim"]
        per_frame["ssim"] = np.asarray(scored["per_frame"])
    if per_frame:
        result["_per_frame"] = per_frame
    return result


def load_models(group: str, device: str):
    from physv_eval.single_case_rigidbench.prediction import (
        load_cotracker_model,
        load_dinov2_model,
        load_sam2_model,
        load_vda_model,
    )

    if group == "mask":
        return {"sam2": load_sam2_model(device), "cotracker": load_cotracker_model(device)}
    if group == "depth":
        return {"vda": load_vda_model(device), "cotracker": load_cotracker_model(device)}
    if group == "track":
        return {"dino": load_dinov2_model(device), "cotracker": load_cotracker_model(device)}
    if group == "image":
        import lpips as lpips_pkg

        return lpips_pkg.LPIPS(net="alex").to(device).eval()
    raise KeyError(group)


def write_result(output_root: Path, task_id: str, case_id: str, result: dict[str, Any]) -> None:
    path = metric_path(output_root, task_id, case_id)
    lock_path = path.parent / ".locks" / f"{case_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        payload = read_json(path)
        payload.setdefault("task_id", task_id)
        payload.setdefault("case_id", case_id)
        for key, value in result.items():
            if key == "_per_frame" or isinstance(value, (np.ndarray, list, dict)):
                continue
            if isinstance(value, np.generic):
                value = value.item()
            payload[key] = value
        atomic_json(path, payload)
        per_frame = result.get("_per_frame", {})
        if per_frame:
            frame_path = output_root / "methods" / task_id / "metrics_per_frame" / f"{case_id}.npz"
            existing: dict[str, np.ndarray] = {}
            if frame_path.is_file():
                with np.load(frame_path, allow_pickle=False) as data:
                    existing = {key: data[key] for key in data.files}
            existing.update({key: np.asarray(value) for key, value in per_frame.items()})
            atomic_npz(frame_path, existing)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def compute_case(group: str, case: dict[str, Any], needed: set[str], models, strict_root: Path, device: str):
    if group == "mask":
        return compute_mask(case, needed, models, strict_root, device)
    if group == "depth":
        return compute_depth(case, needed, models, strict_root, device)
    if group == "track":
        return compute_track(case, needed, models, strict_root, device)
    if group == "image":
        return compute_image(case, needed, models, strict_root, device)
    raise KeyError(group)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grouped full-method test70 RigidBench evaluator")
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--group", choices=tuple(GROUP_METRICS))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--gpu-label", default="unknown")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    registry = load_registry(args.dashboard, args.strict_root, args.output_root, write=args.initialize)
    if args.initialize:
        print(json.dumps({"registry": str(args.output_root / "registry.json"), "models": len(registry["models"]), "cases": len(registry["case_ids"])}, ensure_ascii=False))
        if args.group is None:
            return 0
    if args.group is None:
        raise SystemExit("--group is required unless only --initialize is requested")

    models = registry["models"][args.shard_index :: args.shard_count]
    metrics = GROUP_METRICS[args.group]
    progress_dir = args.output_root / "logs" / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"{args.group}_gpu{args.gpu_label}_shard{args.shard_index}.jsonl"
    done_keys: set[tuple[str, str, str]] = set()
    if args.resume and progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "done":
                    done_keys.add((row["task_id"], row["case_id"], row["metric"]))
    else:
        progress_path.write_text("", encoding="utf-8")

    targets = []
    for model in models:
        for case in model["cases"]:
            if not case.get("prediction_exists"):
                continue
            payload = read_json(metric_path(args.output_root, model["task_id"], case["case_id"]))
            needed = {
                metric
                for metric in metrics
                if metric_missing(payload, metric)
                and (model["task_id"], case["case_id"], metric) not in done_keys
            }
            if needed:
                targets.append((model, case, needed))
    target_rows = sum(len(needed) for _model, _case, needed in targets)
    print(
        json.dumps(
            {
                "group": args.group,
                "gpu": args.gpu_label,
                "shard": f"{args.shard_index}/{args.shard_count}",
                "models": len(models),
                "target_rows": target_rows,
                "metrics": metrics,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if target_rows == 0:
        return 0

    models_loaded = load_models(args.group, args.device)
    completed = len(done_keys)
    failures = 0
    try:
        with progress_path.open("a", encoding="utf-8") as progress:
            for model, case, needed in targets:
                try:
                    result = compute_case(args.group, case, needed, models_loaded, args.strict_root, args.device)
                    write_result(args.output_root, model["task_id"], case["case_id"], result)
                    for metric in sorted(needed):
                        row = {
                            "task_id": model["task_id"],
                            "case_id": case["case_id"],
                            "metric": metric,
                            "status": "done",
                        }
                        progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done_keys.add((model["task_id"], case["case_id"], metric))
                        completed += 1
                    progress.flush()
                    atomic_json(
                        progress_dir / f"{args.group}_gpu{args.gpu_label}_shard{args.shard_index}.state.json",
                        {
                            "group": args.group,
                            "gpu": args.gpu_label,
                            "shard_index": args.shard_index,
                            "shard_count": args.shard_count,
                            "target_rows": target_rows,
                            "completed_rows": completed,
                            "last_task_id": model["task_id"],
                            "last_case_id": case["case_id"],
                        },
                    )
                    print(
                        f"[all-methods] group={args.group} gpu={args.gpu_label} "
                        f"done={completed}/{target_rows} task={model['task_id']} case={case['case_id']} "
                        f"metrics={sorted(needed)}",
                        flush=True,
                    )
                except Exception as exc:
                    failures += len(needed)
                    print(
                        f"[all-methods] group={args.group} gpu={args.gpu_label} "
                        f"task={model['task_id']} case={case['case_id']} failed: {exc!r}",
                        flush=True,
                    )
    finally:
        del models_loaded
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print(json.dumps({"group": args.group, "gpu": args.gpu_label, "completed": completed, "target": target_rows, "failures": failures}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
