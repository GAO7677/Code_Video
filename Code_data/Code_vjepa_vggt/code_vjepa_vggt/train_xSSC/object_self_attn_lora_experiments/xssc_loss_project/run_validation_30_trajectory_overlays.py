#!/usr/bin/env python3
"""Track full-inference validation videos and render trajectory overlays."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("validation_30cases_config.json")
DEFAULT_CACHE = Path(
    "/data/gaoya/agent-data/cache/pybullet0713_object_cotracker_trajectory_v1"
)
DEFAULT_COTRACKER = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)


def find_generated_video(root: Path, entry_id: str, case_id: str) -> Path:
    video_root = root / "videos" / entry_id
    candidates = list(video_root.glob(f"{case_id}*.mp4"))
    candidates.extend(video_root.glob(f"*/{case_id}*.mp4"))
    valid = sorted(
        path.resolve()
        for path in candidates
        if path.is_file() and path.stat().st_size
    )
    if len(valid) != 1:
        raise RuntimeError(
            f"expected one generated video for {entry_id}/{case_id}, "
            f"found {len(valid)}"
        )
    return valid[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--entry-id", action="append", required=True)
    parser.add_argument("--case-count", type=int, default=5)
    parser.add_argument("--trajectory-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu == 4:
        raise SystemExit("GPU4 prohibited")
    if args.case_count <= 0:
        raise SystemExit("--case-count must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import torch

    import trajectory_validation_preview as preview
    from cotracker.predictor import CoTrackerPredictor

    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["output_root"]).expanduser().resolve()
    manifest = json.loads(
        Path(config["cases_manifest"]).read_text(encoding="utf-8")
    )
    cases = manifest["cases"][: args.case_count]
    entries_by_id = {entry["entry_id"]: entry for entry in config["entries"]}
    missing_entries = sorted(set(args.entry_id) - entries_by_id.keys())
    if missing_entries:
        raise ValueError(f"unknown entry ids: {missing_entries}")

    cache_root = args.trajectory_cache.expanduser().resolve()
    cache = preview.PyBulletTrajectoryCache(
        cache_root,
        num_frames=49,
        anchor_frame=preview.ANCHOR_FRAME,
        points_per_object=preview.POINTS_PER_OBJECT,
        track_height=preview.TRACK_HEIGHT,
        track_width=preview.TRACK_WIDTH,
    )
    device = torch.device("cuda:0")
    print(f"[tracker] loading CoTracker3 on physical GPU {args.gpu}", flush=True)
    predictor = (
        CoTrackerPredictor(
            checkpoint=str(args.cotracker_checkpoint.expanduser().resolve()),
            offline=True,
            v2=False,
            window_len=60,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    status_path = root / "trajectory_overlay_status.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {"state": "running", "entries": {}}
    )
    try:
        for entry_id in args.entry_id:
            entry = entries_by_id[entry_id]
            entry_root = root / "trajectory_overlays" / entry_id
            case_rows = []
            status["state"] = "running"
            status["current_entry"] = entry_id
            status["entries"][entry_id] = {
                "state": "running",
                "completed_cases": 0,
                "total_cases": len(cases),
                "manifest_total_cases": len(manifest["cases"]),
            }
            preview.atomic_json(status_path, status)
            for position, case in enumerate(cases, start=1):
                case_dir = entry_root / case["case_id"]
                metrics_path = case_dir / "metrics.json"
                overlay_path = case_dir / "trajectory_overlay.mp4"
                if (
                    metrics_path.is_file()
                    and overlay_path.is_file()
                    and not args.overwrite
                ):
                    case_rows.append(
                        json.loads(metrics_path.read_text(encoding="utf-8"))
                    )
                    continue
                generated_path = find_generated_video(
                    root, entry_id, case["case_id"]
                )
                print(
                    f"[{entry_id} {position}/{len(cases)}] {case['case_id']}",
                    flush=True,
                )
                source_frames = preview.read_video(Path(case["gt_video"]))
                pred_frames = preview.read_video(generated_path)
                if pred_frames.shape != source_frames.shape:
                    raise RuntimeError(
                        f"generated/GT shape mismatch for {case['case_id']}: "
                        f"{pred_frames.shape} != {source_frames.shape}"
                    )
                pred_raw = (
                    torch.from_numpy(pred_frames)
                    .to(device=device, dtype=torch.float32)
                    .permute(0, 3, 1, 2)
                    .unsqueeze(0)
                    .div_(255.0)
                )
                trajectory_metrics, arrays = preview.evaluate_trajectory(
                    predictor,
                    pred_raw,
                    source_frames,
                    cache.load(case["sample_key"]),
                    device,
                    cache_root,
                )
                case_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(case_dir / "trajectories.npz", **arrays)
                preview.render_overlay(
                    case_dir,
                    source_frames,
                    pred_frames,
                    arrays,
                    trajectory_metrics,
                    prediction_label="40-step generated + predicted tracks",
                )
                full_metrics = {
                    **case,
                    **trajectory_metrics,
                    "entry_id": entry_id,
                    "checkpoint": entry["checkpoint"],
                    "generated_video": str(generated_path),
                    "inference_steps": int(
                        config["inference"]["num_inference_steps"]
                    ),
                }
                preview.atomic_json(metrics_path, full_metrics)
                case_rows.append(full_metrics)
                status["entries"][entry_id]["completed_cases"] = len(case_rows)
                status["updated_utc"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                preview.atomic_json(status_path, status)
                del pred_raw, arrays
                torch.cuda.empty_cache()
            report = {
                "schema_version": 1,
                "state": "complete",
                "entry": entry,
                "cases": case_rows,
                "mean_trajectory_loss": float(
                    np.mean([row["trajectory_loss"] for row in case_rows])
                ),
                "mean_trajectory_coordinate_loss": float(
                    np.mean(
                        [row["trajectory_coordinate_loss"] for row in case_rows]
                    )
                ),
                "mean_trajectory_visibility_penalty": float(
                    np.mean(
                        [row["trajectory_visibility_penalty"] for row in case_rows]
                    )
                ),
            }
            preview.atomic_json(entry_root / "report.json", report)
            status["entries"][entry_id] = {
                "state": "complete",
                "completed_cases": len(case_rows),
                "total_cases": len(cases),
                "manifest_total_cases": len(manifest["cases"]),
                "mean_trajectory_loss": report["mean_trajectory_loss"],
            }
            preview.atomic_json(status_path, status)
    finally:
        del predictor
        torch.cuda.empty_cache()
    status.pop("current_entry", None)
    status["state"] = "complete"
    status["updated_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    preview.atomic_json(status_path, status)


if __name__ == "__main__":
    main()
