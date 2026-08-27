#!/usr/bin/env python3
"""Backfill missing test70 RigidBench metrics with shared extraction.

The metric formulas and GT protocol are delegated to the existing single-case
modules.  This runner only changes scheduling: a prediction video is decoded
once per case and shared within a model family, so the same SAM2/VDA/CoTracker
forward pass is not repeated for every scalar metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts")
sys.path.insert(0, str(SCRIPT_ROOT))
import run_test70_rigidbench_metric_backfill as backfill  # noqa: E402
import compare_test70_rigidbench_grouped as grouped  # noqa: E402

GROUP_METRICS = {
    "mask": ("iou", "l2", "chamfer", "bgdrift"),
    "depth": ("ate3d", "si_mse"),
    "track": ("ate", "iddrift"),
    "image": ("lpips", "ssim"),
}


def pending_targets(tasks: list[Path], metrics: tuple[str, ...], strict_root: Path):
    targets: dict[tuple[Path, str], set[str]] = defaultdict(set)
    not_ready = defaultdict(int)
    for task in tasks:
        for metric_path in sorted((task / "metrics").glob("*.json")):
            sample_id = metric_path.stem
            generated = task / "generated" / sample_id
            if not generated.is_dir():
                continue
            payload = backfill.read_json(metric_path)
            for metric in metrics:
                if not backfill.metric_is_missing(metric_path, metric):
                    continue
                if backfill.metric_inputs_ready(task, sample_id, metric, strict_root):
                    targets[(task, sample_id)].add(metric)
                else:
                    not_ready[metric] += 1
    return targets, not_ready


def compute_track_case(task: Path, sample_id: str, needed: set[str], models, strict_root: Path, device: str) -> dict:
    from physv_eval.single_case_rigidbench import ate, iddrift
    from physv_eval.single_case_rigidbench.common import load_video_rgb
    from physv_eval.single_case_rigidbench.prediction import extract_tracks

    case = backfill.sample_dir(strict_root, sample_id)
    pred_video = task / "generated" / sample_id
    pred_frames = load_video_rgb(pred_video)
    gt_tracks, gt_visibility, offsets, _actors = backfill.load_gt_track_bundle(strict_root, sample_id)
    pred_tracks, pred_visibility = extract_tracks(
        pred_video,
        gt_tracks,
        models["cotracker"],
        frames=pred_frames,
    )
    result = {}
    if "ate" in needed:
        metadata = backfill.read_json(case / "metadata.json")
        scored = ate.score_from_predictions(
            gt_tracks,
            pred_tracks,
            pred_visibility,
            int(metadata["camera"]["intrinsics"]["height"]),
            gt_visibility,
        )
        result.update({key: value for key, value in scored.items() if key != "per_frame"})
        if "per_frame" in scored:
            result.setdefault("_per_frame", {})["ate"] = scored["per_frame"]
    if "iddrift" in needed:
        gt_frames = load_video_rgb(case / "video.mp4")
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


def compute_image_case(task: Path, sample_id: str, needed: set[str], model, strict_root: Path, device: str) -> dict:
    from physv_eval.single_case_rigidbench import lpips, ssim

    gt_frames, pred_frames = backfill.load_frames(task, sample_id, strict_root)
    result = {}
    if "lpips" in needed:
        scored = lpips.score_case(gt_frames, pred_frames, model, device)
        result.update({key: value for key, value in scored.items() if key != "per_frame"})
        if "per_frame" in scored:
            result.setdefault("_per_frame", {})["lpips"] = scored["per_frame"]
    if "ssim" in needed:
        scored = ssim.score_case(gt_frames, pred_frames, device)
        result.update({key: value for key, value in scored.items() if key != "per_frame"})
        if "per_frame" in scored:
            result.setdefault("_per_frame", {})["ssim"] = scored["per_frame"]
    return result


def load_models(group: str, device: str):
    if group == "mask":
        return grouped.load_group_models("mask", device)
    if group == "depth":
        return grouped.load_group_models("depth", device)
    if group == "track":
        return backfill.load_shared_model("iddrift", device)
    if group == "image":
        return backfill.load_shared_model("lpips", device)
    raise KeyError(group)


def update_metrics(task: Path, sample_id: str, needed: set[str], result: dict) -> None:
    per_frame = result.get("_per_frame", {})
    scalar_result = {
        key: value
        for key, value in result.items()
        if key != "_per_frame" and not isinstance(value, (np.ndarray, list, dict))
    }
    if scalar_result:
        backfill.update_case(task, sample_id, scalar_result)
    for metric in needed:
        if metric not in per_frame:
            continue
        backfill.update_case(
            task,
            sample_id,
            {metric: result[metric], "per_frame": per_frame[metric]},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Grouped missing-metric backfill for test70")
    parser.add_argument("--group", choices=tuple(GROUP_METRICS), required=True)
    parser.add_argument("--input-root", type=Path, default=backfill.DEFAULT_INPUT_ROOT)
    parser.add_argument("--strict-root", type=Path, default=backfill.DEFAULT_STRICT_ROOT)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--exclude-task-id", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    metrics = GROUP_METRICS[args.group]
    tasks = backfill.task_dirs(args.input_root, args.task_id or None)
    excluded = set(args.exclude_task_id)
    tasks = [task for task in tasks if task.name not in excluded]
    if args.case_id:
        wanted = set(args.case_id)
        for task in tasks:
            # The filter is applied below without changing the task scan.
            pass
    targets, not_ready = pending_targets(tasks, metrics, args.strict_root)
    if args.case_id:
        wanted = set(args.case_id)
        targets = {(task, sample): values for (task, sample), values in targets.items() if sample in wanted}
    total = sum(len(values) for values in targets.values())
    print(
        json.dumps(
            {
                "group": args.group,
                "tasks": len(tasks),
                "target_rows": total,
                "metrics": metrics,
                "not_ready": dict(not_ready),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if total == 0:
        return 0

    args.progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_dir / f"{args.group}.jsonl"
    done_keys = set()
    if args.resume and progress_path.is_file():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "done":
                    done_keys.add((row["task"], row["case"], row["metric"]))
    else:
        progress_path.write_text("", encoding="utf-8")

    models = load_models(args.group, args.device)
    failures = 0
    completed = len(done_keys)
    try:
        with progress_path.open("a", encoding="utf-8") as progress:
            for (task, sample_id), needed_all in sorted(targets.items(), key=lambda item: (item[0][0].name, item[0][1])):
                needed = {
                    metric
                    for metric in needed_all
                    if (task.name, sample_id, metric) not in done_keys
                }
                if not needed:
                    continue
                try:
                    if args.group == "mask":
                        result = grouped.compute_mask_case(
                            task,
                            sample_id,
                            needed,
                            models,
                            args.strict_root,
                            args.device,
                            include_per_frame=True,
                        )
                    elif args.group == "depth":
                        result = grouped.compute_depth_case(task, sample_id, needed, models, args.strict_root, args.device)
                    elif args.group == "track":
                        result = compute_track_case(task, sample_id, needed, models, args.strict_root, args.device)
                    else:
                        result = compute_image_case(task, sample_id, needed, models, args.strict_root, args.device)
                    update_metrics(task, sample_id, needed, result)
                    for metric in sorted(needed):
                        row = {"task": task.name, "case": sample_id, "metric": metric, "status": "done"}
                        progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done_keys.add((task.name, sample_id, metric))
                        completed += 1
                    progress.flush()
                    print(
                        f"[backfill-grouped] group={args.group} done={completed}/{total} "
                        f"task={task.name} case={sample_id} metrics={sorted(needed)}",
                        flush=True,
                    )
                    backfill.atomic_json(
                        args.progress_dir / f"{args.group}.state.json",
                        {
                            "group": args.group,
                            "target_rows": total,
                            "completed_rows": completed,
                            "metrics": metrics,
                            "last_task": task.name,
                            "last_case": sample_id,
                        },
                    )
                except Exception as exc:
                    failures += len(needed)
                    print(f"[backfill-grouped] group={args.group} {task.name}/{sample_id} failed: {exc!r}", flush=True)
    finally:
        del models
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print(json.dumps({"group": args.group, "completed": completed, "target": total, "failures": failures}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
