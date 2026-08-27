#!/usr/bin/env python3
"""Grouped RigidBench regression evaluation.

This keeps the official metric functions and tolerances unchanged while
sharing prediction extraction within one task/case:

* ``mask``: one SAM2 propagation for IoU, L2, Chamfer and BG-Drift;
* ``depth``: one VDA pass for SI-MSE and ATE-3D;
* ``identity``: one CoTracker pass and one DINO pass for ID-Drift.

Only the generated video and strict CYCLES GT are used.  The process writes a
small JSONL progress stream after every metric/case, but the final comparison
reports retain the same schema as the original regression script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNNER_DIR = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts")
sys.path.insert(0, str(RUNNER_DIR))
import run_test70_rigidbench_metric_backfill as backfill  # noqa: E402

EVAL_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
sys.path.insert(0, str(EVAL_ROOT))

GROUP_METRICS = {
    "mask": ("iou", "l2", "chamfer", "bgdrift"),
    "depth": ("ate3d", "si_mse"),
    "identity": ("iddrift",),
}


def close_value(old, new, atol: float, rtol: float) -> bool:
    if old is None or new is None:
        return old == new
    try:
        old_f, new_f = float(old), float(new)
    except (TypeError, ValueError):
        return old == new
    if math.isnan(old_f) and math.isnan(new_f):
        return True
    return bool(np.isclose(old_f, new_f, atol=atol, rtol=rtol, equal_nan=True))


def target_cases(task: Path, metric: str, case_filter: set[str], all_complete: bool) -> list[str]:
    if case_filter:
        return sorted(case_filter)
    if not all_complete:
        return []
    result = []
    for path in sorted((task / "metrics").glob("*.json")):
        payload = backfill.read_json(path)
        if payload.get(metric) is not None and (task / "generated" / path.stem).is_dir():
            result.append(path.stem)
    return result


def build_targets(tasks: list[Path], metrics: tuple[str, ...], args) -> dict[tuple[Path, str], set[str]]:
    targets: dict[tuple[Path, str], set[str]] = defaultdict(set)
    case_filter = set(args.case_id or [])
    for metric in metrics:
        for task in tasks:
            for sample_id in target_cases(task, metric, case_filter, args.all_complete):
                targets[(task, sample_id)].add(metric)
    return targets


def compare_row(metric: str, task: Path, sample_id: str, result: dict, atol: float, rtol: float) -> dict:
    old_payload = backfill.read_json(backfill.task_sample_json(task, sample_id))
    old_value = old_payload.get(metric)
    new_value = result.get(metric)
    ok = close_value(old_value, new_value, atol, rtol)
    try:
        abs_diff = abs(float(old_value) - float(new_value)) if old_value is not None and new_value is not None else None
    except (TypeError, ValueError):
        abs_diff = None
    return {
        "task": task.name,
        "metric": metric,
        "case": sample_id,
        "old": old_value,
        "new": new_value,
        "abs_diff": abs_diff,
        "status": "match" if ok else "mismatch",
    }


def mask_result(gt_mask: np.ndarray, pred_mask: np.ndarray, metric: str, include_per_frame: bool = False) -> dict:
    from rigidbench.eval.score.mask import chamfer_per_frame, iou_per_frame, l2_per_frame
    from physv_eval.single_case_rigidbench.mask_metric_common import score_mask_metric

    functions = {
        "iou": iou_per_frame,
        "l2": l2_per_frame,
        "chamfer": chamfer_per_frame,
    }
    T = min(len(gt_mask), len(pred_mask))
    scored = score_mask_metric(gt_mask[:T], pred_mask[:T], functions[metric])
    result = {metric: scored["value"]}
    if include_per_frame:
        result.setdefault("_per_frame", {})[metric] = scored["per_frame"]
    return result


def compute_mask_case(
    task: Path,
    sample_id: str,
    needed: set[str],
    models,
    strict_root: Path,
    device: str,
    include_per_frame: bool = False,
) -> dict:
    from physv_eval.single_case_rigidbench import bgdrift
    from physv_eval.single_case_rigidbench.common import load_npz_array, load_video_rgb
    from physv_eval.single_case_rigidbench.prediction import active_actor_indices, extract_masks

    case = backfill.sample_dir(strict_root, sample_id)
    pred_video = task / "generated" / sample_id
    frames = load_video_rgb(pred_video)
    gt_mask = load_npz_array(case / "masks.npz", "masks", "mask")
    metadata = backfill.read_json(case / "metadata.json")
    active = active_actor_indices(case / "masks.npz", metadata)
    pred_mask = extract_masks(pred_video, gt_mask, models["sam2"], active, frames=frames)
    gt_active = gt_mask[:, active] if active else gt_mask

    result = {}
    for metric in ("iou", "l2", "chamfer"):
        if metric in needed:
            scored = mask_result(gt_active, pred_mask, metric, include_per_frame)
            per_frame = scored.pop("_per_frame", None)
            result.update(scored)
            if per_frame is not None:
                result.setdefault("_per_frame", {}).update(per_frame)
    if "bgdrift" in needed:
        result.update(bgdrift.score_from_frames_and_masks(frames, pred_mask, models["cotracker"], device))
    return result


def compute_depth_case(task: Path, sample_id: str, needed: set[str], models, strict_root: Path, device: str) -> dict:
    from physv_eval.single_case_rigidbench import ate3d, si_mse
    from physv_eval.single_case_rigidbench.common import load_npz_array, load_video_rgb
    from physv_eval.single_case_rigidbench.prediction import extract_disparity, extract_tracks

    case = backfill.sample_dir(strict_root, sample_id)
    pred_video = task / "generated" / sample_id
    frames = load_video_rgb(pred_video)
    pred_disparity = extract_disparity(pred_video, models["vda"], device, frames=frames)
    gt_depth = load_npz_array(case / "depth.npz", "depth")
    result = {}
    if "si_mse" in needed:
        result.update(si_mse.score_from_disparity(gt_depth, pred_disparity))

    if "ate3d" in needed:
        metadata = backfill.read_json(case / "metadata.json")
        gt_tracks, gt_visibility, offsets, actors = backfill.load_gt_track_bundle(strict_root, sample_id)
        with np.load(case / "trajectories.npz", allow_pickle=False) as data:
            gt_trajectories = {key: data[key] for key in data.files}
        pred_tracks, pred_visibility = extract_tracks(
            pred_video,
            gt_tracks,
            models["cotracker"],
            frames=frames,
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


def compute_identity_case(task: Path, sample_id: str, models, strict_root: Path, device: str) -> dict:
    from physv_eval.single_case_rigidbench import iddrift
    from physv_eval.single_case_rigidbench.common import load_video_rgb
    from physv_eval.single_case_rigidbench.prediction import extract_tracks

    case = backfill.sample_dir(strict_root, sample_id)
    pred_video = task / "generated" / sample_id
    gt_frames = load_video_rgb(case / "video.mp4")
    pred_frames = load_video_rgb(pred_video)
    gt_tracks, gt_visibility, offsets, _actors = backfill.load_gt_track_bundle(strict_root, sample_id)
    pred_tracks, pred_visibility = extract_tracks(
        pred_video,
        gt_tracks,
        models["cotracker"],
        frames=pred_frames,
    )
    return iddrift.score_from_predictions(
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


def load_group_models(group: str, device: str):
    if group == "mask":
        return backfill.load_shared_model("bgdrift", device)
    if group == "depth":
        return backfill.load_shared_model("ate3d", device)
    if group == "identity":
        return backfill.load_shared_model("iddrift", device)
    raise KeyError(group)


def write_final_reports(report_dir: Path, metrics: tuple[str, ...], rows_by_metric: dict[str, list[dict]], task_ids, args) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        rows = rows_by_metric[metric]
        ok = all(row.get("status") == "match" for row in rows)
        output = {
            "tasks": task_ids,
            "cases": args.case_id or [],
            "all_complete": args.all_complete,
            "ok": ok,
            "rows": rows,
        }
        backfill.atomic_json(report_dir / f"gpu2_{metric}.json", output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grouped video-only RigidBench regression")
    parser.add_argument("--group", choices=tuple(GROUP_METRICS), required=True)
    parser.add_argument("--input-root", type=Path, default=backfill.DEFAULT_INPUT_ROOT)
    parser.add_argument("--strict-root", type=Path, default=backfill.DEFAULT_STRICT_ROOT)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--all-complete", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    metrics = GROUP_METRICS[args.group]
    tasks = backfill.task_dirs(args.input_root, args.task_id)
    if not tasks:
        raise SystemExit(f"No requested task found: {args.task_id}")
    targets = build_targets(tasks, metrics, args)
    total = sum(len(values) for values in targets.values())
    print(json.dumps({"group": args.group, "tasks": len(tasks), "target_rows": total, "metrics": metrics}, ensure_ascii=False), flush=True)

    args.progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_dir / f"{args.group}.jsonl"
    completed: dict[tuple[str, str, str], dict] = {}
    if args.resume and progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") in {"match", "mismatch"}:
                completed[(row["task"], row["case"], row["metric"])] = row
    else:
        progress_path.write_text("", encoding="utf-8")

    rows_by_metric: dict[str, list[dict]] = defaultdict(list)
    for row in completed.values():
        rows_by_metric[row["metric"]].append(row)

    models = load_group_models(args.group, args.device)
    try:
        with progress_path.open("a", encoding="utf-8") as progress:
            done = len(completed)
            for (task, sample_id), needed in sorted(targets.items(), key=lambda item: (item[0][0].name, item[0][1])):
                pending = {
                    metric
                    for metric in needed
                    if (task.name, sample_id, metric) not in completed
                }
                if not pending:
                    continue

                ready = {metric for metric in pending if backfill.metric_inputs_ready(task, sample_id, metric, args.strict_root)}
                for metric in sorted(pending - ready):
                    row = {
                        "task": task.name,
                        "metric": metric,
                        "case": sample_id,
                        "status": "not_ready",
                    }
                    rows_by_metric[metric].append(row)
                    progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.flush()
                    done += 1
                if not ready:
                    continue

                try:
                    if args.group == "mask":
                        result = compute_mask_case(task, sample_id, ready, models, args.strict_root, args.device)
                    elif args.group == "depth":
                        result = compute_depth_case(task, sample_id, ready, models, args.strict_root, args.device)
                    else:
                        result = compute_identity_case(task, sample_id, models, args.strict_root, args.device)
                except Exception as exc:
                    for metric in sorted(ready):
                        row = {
                            "task": task.name,
                            "metric": metric,
                            "case": sample_id,
                            "status": "error",
                            "error": repr(exc),
                        }
                        rows_by_metric[metric].append(row)
                        progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.flush()
                        done += 1
                    print(f"[grouped] {args.group} {task.name}/{sample_id} failed: {exc!r}", flush=True)
                    continue

                for metric in sorted(ready):
                    row = compare_row(metric, task, sample_id, result, args.atol, args.rtol)
                    rows_by_metric[metric].append(row)
                    progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.flush()
                    done += 1
                print(f"[grouped] group={args.group} done={done}/{total} task={task.name} case={sample_id} metrics={sorted(ready)}", flush=True)
                backfill.atomic_json(
                    args.progress_dir / f"{args.group}.state.json",
                    {
                        "group": args.group,
                        "target_rows": total,
                        "completed_rows": done,
                        "metrics": {metric: len(rows_by_metric[metric]) for metric in metrics},
                        "last_task": task.name,
                        "last_case": sample_id,
                    },
                )
    finally:
        del models
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    write_final_reports(args.report_dir, metrics, rows_by_metric, args.task_id, args)
    overall_ok = all(row.get("status") == "match" for rows in rows_by_metric.values() for row in rows)
    print(json.dumps({"group": args.group, "ok": overall_ok, "rows": sum(len(v) for v in rows_by_metric.values())}, ensure_ascii=False), flush=True)
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
