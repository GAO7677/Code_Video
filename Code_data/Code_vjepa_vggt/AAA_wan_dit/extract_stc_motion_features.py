#!/usr/bin/env python3
"""Extract normalized RAFT flow and CoTracker trajectories for ablation videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_motion_analysis"
)
DEFAULT_VBENCH_ROOT = Path(
    "/home/gaoya/Code_Video/DreamWorld-main/evaluation/VBench"
)
DEFAULT_RAFT_CHECKPOINT = Path(
    "/data/gaoya/ckpt/RAFT-Things/models/raft-things.pth"
)
DEFAULT_COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
DEFAULT_COTRACKER_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)
DEFAULT_REGION_CACHE = Path(
    "/data/gaoya/agent-data/cache/physiciq_selected_sam2_regions/"
    "case_physiciq_01_physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed-ball-and-block-fall_motion_to_end"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_OUTPUT_ROOT / "inventory.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--flow-store-height", type=int, default=64)
    parser.add_argument("--flow-store-width", type=int, default=112)
    parser.add_argument("--raft-batch-size", type=int, default=4)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--cotracker-grid-size", type=int, default=20)
    parser.add_argument("--region-cache", type=Path, default=DEFAULT_REGION_CACHE)
    parser.add_argument(
        "--track-query-frame",
        type=int,
        help="Defaults to the final context frame. Region points come from the cached context mask.",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--entry-id", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def center_crop_to_aspect(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = width / height
    if source_aspect > target_aspect:
        crop_width = max(1, round(height * target_aspect))
        left = (width - crop_width) // 2
        return frame[:, left : left + crop_width]
    crop_height = max(1, round(width / target_aspect))
    top = (height - crop_height) // 2
    return frame[top : top + crop_height]


def read_video(path: Path, num_frames: int, height: int, width: int) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    source_frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_frame_count += 1
        if len(frames) >= num_frames:
            continue
        frame = center_crop_to_aspect(frame, width, height)
        interpolation = (
            cv2.INTER_AREA
            if frame.shape[0] >= height and frame.shape[1] >= width
            else cv2.INTER_LINEAR
        )
        frame = cv2.resize(frame, (width, height), interpolation=interpolation)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < num_frames:
        raise RuntimeError(
            f"{path} has {len(frames)} readable frames; expected at least {num_frames}"
        )
    return np.stack(frames), fps, source_frame_count


def load_raft(device: str):
    if str(DEFAULT_VBENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(DEFAULT_VBENCH_ROOT))
    from vbench.third_party.RAFT.core.raft import RAFT

    model_args = Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        dropout=0.0,
    )
    model = RAFT(model_args)
    checkpoint = torch.load(DEFAULT_RAFT_CHECKPOINT, map_location="cpu")
    checkpoint = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_cotracker(device: str):
    if str(DEFAULT_COTRACKER_ROOT) not in sys.path:
        sys.path.insert(0, str(DEFAULT_COTRACKER_ROOT))
    from cotracker.predictor import CoTrackerPredictor

    return (
        CoTrackerPredictor(
            checkpoint=str(DEFAULT_COTRACKER_CHECKPOINT),
            offline=True,
            v2=False,
            window_len=60,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )


@torch.inference_mode()
def extract_raft(
    model,
    frames: np.ndarray,
    device: str,
    batch_size: int,
    iters: int,
    store_height: int,
    store_width: int,
) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    height, width = frames.shape[1:3]
    outputs: list[np.ndarray] = []
    for start in range(0, len(frames) - 1, batch_size):
        end = min(len(frames) - 1, start + batch_size)
        image1 = tensor[start:end].to(device, non_blocking=True)
        image2 = tensor[start + 1 : end + 1].to(device, non_blocking=True)
        _, flow = model(image1, image2, iters=iters, test_mode=True)
        flow[:, 0] /= float(width)
        flow[:, 1] /= float(height)
        flow = F.interpolate(
            flow,
            size=(store_height, store_width),
            mode="bilinear",
            align_corners=True,
        )
        outputs.append(flow.float().cpu().numpy())
    flow_norm = np.concatenate(outputs, axis=0)
    magnitude = np.linalg.norm(flow_norm, axis=1)
    return {
        "flow_norm": flow_norm.astype(np.float16),
        "flow_mean": magnitude.mean(axis=(1, 2)).astype(np.float32),
        "flow_p90": np.quantile(magnitude, 0.90, axis=(1, 2)).astype(np.float32),
        "flow_top05": np.mean(
            np.sort(magnitude.reshape(len(magnitude), -1), axis=1)[
                :, -max(1, magnitude.shape[1] * magnitude.shape[2] // 20) :
            ],
            axis=1,
        ).astype(np.float32),
    }


@torch.inference_mode()
def extract_cotracker(
    model,
    frames: np.ndarray,
    device: str,
    grid_size: int,
    query_frame: int,
    query_points: np.ndarray | None,
    region_ids: np.ndarray | None,
) -> dict[str, np.ndarray]:
    height, width = frames.shape[1:3]
    video = (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)
        .float()
        .unsqueeze(0)
        .to(device)
    )
    if query_points is None:
        tracks, visibility = model(
            video,
            grid_size=grid_size,
            grid_query_frame=query_frame,
            backward_tracking=False,
        )
        query_points_norm = tracks[0, query_frame].float().cpu().numpy()
        query_points_norm[..., 0] /= max(width - 1, 1)
        query_points_norm[..., 1] /= max(height - 1, 1)
        region_ids = np.full(len(query_points_norm), -1, dtype=np.int16)
    else:
        query_tensor = torch.from_numpy(query_points).float()
        query_times = torch.full((len(query_tensor), 1), float(query_frame))
        queries = torch.cat((query_times, query_tensor), dim=1).unsqueeze(0).to(device)
        tracks, visibility = model(
            video,
            queries=queries,
            backward_tracking=False,
        )
        query_points_norm = query_points.copy()
        query_points_norm[..., 0] /= max(width - 1, 1)
        query_points_norm[..., 1] /= max(height - 1, 1)
    tracks = tracks[0].float().cpu().numpy()
    tracks[..., 0] /= max(width - 1, 1)
    tracks[..., 1] /= max(height - 1, 1)
    return {
        "tracks_norm": tracks.astype(np.float32),
        "track_visibility": visibility[0].cpu().numpy().astype(np.bool_),
        "track_query_points_norm": query_points_norm.astype(np.float32),
        "track_region_ids": np.asarray(region_ids, dtype=np.int16),
    }


def load_region_queries(
    cache_dir: Path | None,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int, list[dict[str, Any]]]:
    if cache_dir is None:
        return None, None, 0, []
    arrays_path = cache_dir / "regions.npz"
    metadata_path = cache_dir / "regions.json"
    if not arrays_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete SAM2 region cache: {cache_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as arrays:
        points = arrays["query_points"].astype(np.float32)
    source_height = int(metadata["height"])
    source_width = int(metadata["width"])
    points[:, 0] *= max(target_width - 1, 1) / max(source_width - 1, 1)
    points[:, 1] *= max(target_height - 1, 1) / max(source_height - 1, 1)
    regions = list(metadata["regions"])
    region_ids = np.full(len(points), -1, dtype=np.int16)
    for region_index, region in enumerate(regions):
        region_ids[int(region["point_start"]) : int(region["point_end"])] = region_index
    if np.any(region_ids < 0):
        raise ValueError("SAM2 region slices do not cover every query point")
    return (
        points,
        region_ids,
        int(metadata["query_context_frame"]),
        regions,
    )


def feature_is_current(metadata_path: Path, entry: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.overwrite or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("status") == "complete"
        and metadata.get("source", {}).get("cache_key") == entry["source"]["cache_key"]
        and metadata.get("settings", {}).get("num_frames") == args.num_frames
        and metadata.get("settings", {}).get("context_frames") == args.context_frames
        and metadata.get("settings", {}).get("height") == args.height
        and metadata.get("settings", {}).get("width") == args.width
        and metadata.get("settings", {}).get("cotracker_grid_size")
        == args.cotracker_grid_size
        and metadata.get("settings", {}).get("track_query_mode")
        == ("sam2_regions" if args.region_cache is not None else "uniform_grid")
        and metadata.get("settings", {}).get("query_frame")
        == (
            args.track_query_frame
            if args.track_query_frame is not None
            else args.context_frames - 1
        )
        and metadata.get("settings", {}).get("raft_iters") == args.raft_iters
        and metadata_path.with_name("features.npz").is_file()
    )


def select_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    for entry in entries:
        if args.entry_id and entry["entry_id"] not in args.entry_id:
            continue
        if args.model and entry["model"] not in args.model:
            continue
        if args.seed and entry["seed"] not in args.seed:
            continue
        if args.variant and entry["variant"] not in args.variant:
            continue
        selected.append(entry)
    selected = [
        entry
        for index, entry in enumerate(selected)
        if index % args.num_shards == args.shard_id
    ]
    return selected[: args.limit] if args.limit is not None else selected


def main() -> None:
    args = parse_args()
    if args.context_frames < 1 or args.context_frames >= args.num_frames:
        raise ValueError("context_frames must be in [1, num_frames)")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    entries = select_entries(inventory["entries"], args)
    if not entries:
        raise RuntimeError("No inventory entries matched the filters")

    raft = load_raft(args.device)
    cotracker = load_cotracker(args.device)
    query_points, region_ids, region_source_frame, track_regions = load_region_queries(
        args.region_cache,
        args.height,
        args.width,
    )
    track_query_frame = (
        args.track_query_frame
        if args.track_query_frame is not None
        else args.context_frames - 1
    )
    print(
        f"[extract] entries={len(entries)} device={args.device} "
        f"shard={args.shard_id}/{args.num_shards}",
        flush=True,
    )
    completed = 0
    reused = 0
    for index, entry in enumerate(entries, start=1):
        feature_dir = args.output_root / "features" / entry["source"]["cache_key"]
        metadata_path = feature_dir / "metadata.json"
        if feature_is_current(metadata_path, entry, args):
            reused += 1
            print(f"[{index}/{len(entries)}] reuse {entry['entry_id']}", flush=True)
            continue
        started = time.time()
        frames, fps, source_frame_count = read_video(
            Path(entry["source"]["path"]),
            args.num_frames,
            args.height,
            args.width,
        )
        arrays = extract_raft(
            raft,
            frames,
            args.device,
            args.raft_batch_size,
            args.raft_iters,
            args.flow_store_height,
            args.flow_store_width,
        )
        arrays.update(
            extract_cotracker(
                cotracker,
                frames,
                args.device,
                args.cotracker_grid_size,
                track_query_frame,
                query_points,
                region_ids,
            )
        )
        atomic_save_npz(feature_dir / "features.npz", **arrays)
        metadata = {
            "schema_version": 2,
            "status": "complete",
            "entry": entry,
            "source": entry["source"],
            "fps": fps,
            "source_frame_count": source_frame_count,
            "elapsed_seconds": time.time() - started,
            "settings": {
                "num_frames": args.num_frames,
                "context_frames": args.context_frames,
                "query_frame": track_query_frame,
                "height": args.height,
                "width": args.width,
                "flow_store_height": args.flow_store_height,
                "flow_store_width": args.flow_store_width,
                "raft_iters": args.raft_iters,
                "raft_batch_size": args.raft_batch_size,
                "cotracker_grid_size": args.cotracker_grid_size,
                "track_query_mode": (
                    "sam2_regions" if query_points is not None else "uniform_grid"
                ),
                "track_regions": track_regions,
                "region_source_frame": region_source_frame,
                "spatial_preprocess": "center_crop_to_7:4_then_resize",
                "flow_units": "normalized_image_width_and_height_per_frame",
                "track_units": "normalized_xy",
            },
        }
        atomic_write_json(metadata_path, metadata)
        completed += 1
        print(
            f"[{index}/{len(entries)}] complete {entry['entry_id']} "
            f"{metadata['elapsed_seconds']:.1f}s",
            flush=True,
        )
    print(f"[extract] complete={completed} reused={reused}", flush=True)


if __name__ == "__main__":
    main()
