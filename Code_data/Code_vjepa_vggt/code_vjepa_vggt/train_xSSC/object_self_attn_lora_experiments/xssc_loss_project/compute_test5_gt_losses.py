#!/usr/bin/env python3
"""Compute future-frame RGB MSE and CoTracker trajectory loss against GT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from run_test5_all_checkpoints_train_cases import (
    DEFAULT_CONFIG,
    PYTHON,
    atomic_json,
    completed_cases,
    discover_inventory,
)


ROOT = Path(__file__).resolve().parent
BUILD_PAGE = ROOT / "build_test5_all_checkpoints_train_case_gallery.py"
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
COTRACKER_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--watch-seconds", type=int, default=30)
    parser.add_argument("--no-watch", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_video(path: Path, num_frames: int, height: int, width: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != num_frames:
        raise RuntimeError(f"{path} has {len(frames)} frames; expected {num_frames}")
    return np.stack(frames)


def rgb_future_mse(prediction: Path, gt: Path, context_frames: int) -> float:
    pred_frames = read_video(prediction, 49, 512, 896)[context_frames:].astype(
        np.float32
    )
    gt_frames = read_video(gt, 49, 512, 896)[context_frames:].astype(np.float32)
    difference = (pred_frames - gt_frames) / 255.0
    return float(np.mean(np.square(difference), dtype=np.float64))


def load_cotracker(device: str):
    if str(COTRACKER_ROOT) not in sys.path:
        sys.path.insert(0, str(COTRACKER_ROOT))
    from cotracker.predictor import CoTrackerPredictor

    return (
        CoTrackerPredictor(
            checkpoint=str(COTRACKER_CHECKPOINT),
            offline=True,
            v2=False,
            window_len=60,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )


def query_grid(size: int, height: int, width: int, query_frame: int) -> torch.Tensor:
    if str(COTRACKER_ROOT) not in sys.path:
        sys.path.insert(0, str(COTRACKER_ROOT))
    from cotracker.models.core.model_utils import get_points_on_a_grid

    points = get_points_on_a_grid(size, (height, width))
    times = torch.full_like(points[:, :, :1], float(query_frame))
    return torch.cat((times, points), dim=2)


@torch.inference_mode()
def track_video(
    model,
    path: Path,
    device: str,
    queries: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    frames = read_video(path, 49, 256, 448)
    video = (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)
        .float()
        .unsqueeze(0)
        .to(device)
    )
    tracks, visibility = model(
        video,
        queries=queries.to(device),
        backward_tracking=False,
    )
    tracks = tracks[0].float().cpu().numpy()
    tracks[..., 0] /= 447.0
    tracks[..., 1] /= 255.0
    visible = visibility[0].cpu().numpy().astype(bool)
    return tracks.astype(np.float32), visible


def gt_tracks(
    model,
    case: dict,
    cache_root: Path,
    device: str,
    queries: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    gt_path = Path(case["gt_video"]).resolve()
    cache_path = cache_root / f"{case['case_id']}.npz"
    gt_signature = signature(gt_path)
    if cache_path.is_file():
        with np.load(cache_path) as archive:
            cached_signature = {
                "size": int(archive["source_size"]),
                "mtime_ns": int(archive["source_mtime_ns"]),
            }
            if cached_signature == gt_signature:
                return archive["tracks_norm"], archive["visibility"]
    tracks, visibility = track_video(model, gt_path, device, queries)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        tracks_norm=tracks,
        visibility=visibility,
        source_size=np.int64(gt_signature["size"]),
        source_mtime_ns=np.int64(gt_signature["mtime_ns"]),
    )
    os.replace(temporary, cache_path)
    return tracks, visibility


def trajectory_loss(
    pred_tracks: np.ndarray,
    pred_visibility: np.ndarray,
    gt_track: np.ndarray,
    gt_visibility: np.ndarray,
    query_frame: int,
    context_frames: int,
) -> tuple[float | None, float]:
    pred_displacement = pred_tracks - pred_tracks[query_frame : query_frame + 1]
    gt_displacement = gt_track - gt_track[query_frame : query_frame + 1]
    difference = pred_displacement[context_frames:] - gt_displacement[context_frames:]
    visible = (
        pred_visibility[context_frames:] & gt_visibility[context_frames:]
    )
    valid_ratio = float(visible.mean())
    if not visible.any():
        return None, valid_ratio
    squared_distance = np.square(difference).sum(axis=-1)
    return float(np.sqrt(squared_distance[visible].mean())), valid_ratio


def load_payload(path: Path, settings: dict) -> dict:
    if not path.is_file():
        return {
            "schema_version": 1,
            "updated_utc": timestamp(),
            "settings": settings,
            "results": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("settings") != settings:
        raise ValueError(f"Metric settings changed; refusing to mix results in {path}")
    return payload


def build_page(config_path: Path) -> None:
    subprocess.run(
        [str(PYTHON), str(BUILD_PAGE), "--config", str(config_path)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"]).expanduser().resolve()
    inventory_path = output_root / "all_checkpoint_inventory.json"
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    else:
        inventory = discover_inventory(config)
        atomic_json(inventory_path, inventory)
    cases_payload = json.loads(
        Path(config["cases_manifest"]).read_text(encoding="utf-8")
    )
    cases = cases_payload["cases"]
    cases_by_id = {str(case["case_id"]): case for case in cases}
    case_names = list(cases_by_id)
    settings = {
        "mse": "mean((pred_rgb_gt_future / 255)^2), frames [8,49)",
        "trajectory_loss": (
            "sqrt(mean((pred_displacement_norm - gt_displacement_norm)^2_xy)), "
            "CoTracker3 20x20 grid queried at frame 7, jointly visible points, "
            "frames [8,49)"
        ),
        "num_frames": 49,
        "context_frames": 8,
        "track_query_frame": 7,
        "track_grid_size": 20,
        "track_height": 256,
        "track_width": 448,
        "cotracker_checkpoint": str(COTRACKER_CHECKPOINT),
    }
    metrics_path = output_root / "gt_losses.json"
    status_path = output_root / "gt_losses_runtime_status.json"
    payload = load_payload(metrics_path, settings)
    queries = query_grid(20, 256, 448, 7)
    model = None
    processed_this_run = 0

    while True:
        ready = []
        total_ready = 0
        total_complete = 0
        for entry in inventory["entries"]:
            entry_results = payload["results"].setdefault(entry["entry_id"], {})
            root = Path(entry["result_root"])
            for case_id in case_names:
                video = root / f"{case_id}.mp4"
                metadata = root / f"{case_id}.json"
                if not video.is_file() or not metadata.is_file():
                    continue
                total_ready += 1
                current_signature = signature(video)
                record = entry_results.get(case_id, {})
                if (
                    record.get("video_signature") == current_signature
                    and record.get("mse_loss") is not None
                    and "trajectory_loss" in record
                ):
                    total_complete += 1
                    continue
                ready.append((entry, case_id, video, current_signature, record))

        if args.limit is not None:
            ready = ready[: max(0, args.limit - processed_this_run)]
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "updated_utc": timestamp(),
                "state": "running" if ready or total_ready < 360 else "complete",
                "ready_videos": total_ready,
                "complete_metrics": total_complete,
                "expected_videos": inventory["num_checkpoints"] * len(case_names),
                "pending_ready_metrics": len(ready),
            },
        )

        for entry, case_id, video, current_signature, previous in ready:
            case = cases_by_id[case_id]
            record = (
                previous
                if previous.get("video_signature") == current_signature
                else {}
            )
            record.update(
                {
                    "method_key": entry["method_key"],
                    "step": entry["step"],
                    "case_id": case_id,
                    "video_path": str(video),
                    "gt_path": str(Path(case["gt_video"]).resolve()),
                    "video_signature": current_signature,
                }
            )
            if record.get("mse_loss") is None:
                record["mse_loss"] = rgb_future_mse(
                    video,
                    Path(case["gt_video"]).resolve(),
                    context_frames=8,
                )
                record["updated_utc"] = timestamp()
                payload["results"][entry["entry_id"]][case_id] = record
                payload["updated_utc"] = timestamp()
                atomic_json(metrics_path, payload)

            if model is None:
                print(f"[{timestamp()}] loading CoTracker on {args.device}", flush=True)
                model = load_cotracker(args.device)
            gt_track, gt_visibility = gt_tracks(
                model,
                case,
                output_root / "gt_trajectory_cache",
                args.device,
                queries,
            )
            pred_track, pred_visibility = track_video(
                model,
                video,
                args.device,
                queries,
            )
            loss, valid_ratio = trajectory_loss(
                pred_track,
                pred_visibility,
                gt_track,
                gt_visibility,
                query_frame=7,
                context_frames=8,
            )
            record["trajectory_loss"] = loss
            record["trajectory_joint_visibility"] = valid_ratio
            record["updated_utc"] = timestamp()
            payload["results"][entry["entry_id"]][case_id] = record
            payload["updated_utc"] = timestamp()
            atomic_json(metrics_path, payload)
            processed_this_run += 1
            print(
                f"[{timestamp()}] {entry['entry_id']} {case_id} "
                f"mse={record['mse_loss']:.6f} trajectory={loss}",
                flush=True,
            )
            if processed_this_run % 3 == 0:
                build_page(config_path)
            if args.limit is not None and processed_this_run >= args.limit:
                build_page(config_path)
                return

        build_page(config_path)
        if args.no_watch or total_ready >= 360 and not ready:
            break
        time.sleep(args.watch_seconds)

    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "updated_utc": timestamp(),
            "state": "complete",
            "ready_videos": 360,
            "complete_metrics": 360,
            "expected_videos": 360,
            "pending_ready_metrics": 0,
        },
    )
    build_page(config_path)


if __name__ == "__main__":
    main()
