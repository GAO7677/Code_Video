#!/usr/bin/env python3
"""Precompute compact object-level CoTracker supervision for PyBullet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import itertools
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
DIFFTRACK_ROOT = Path("/home/gaoya/Code_Video/DiffTrack-main")
for _path in (HERE, EXPERIMENT_ROOT, PACKAGE_ROOT, COTRACKER_ROOT, DIFFTRACK_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from AAA_my_test.precompute_toydataset_sam2_regions import build_provider
from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
    PyBullet0713NoGTBoxDataset,
)
from code_vjepa_vggt.data.pybullet_vae_cache import sample_uid
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    DetectedObjectTrack,
)
from code_vjepa_vggt.utils.object_priors import sample_points_from_mask
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix
from trajectory_cache import CACHE_SCHEMA_VERSION, trajectory_relative_path


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/pybullet0713_object_cotracker_trajectory_v1"
)
DEFAULT_COTRACKER_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)
TRACK_HEIGHT = 256
TRACK_WIDTH = 448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER_CHECKPOINT
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--anchor-frame", type=int, default=4)
    parser.add_argument("--points-per-object", type=int, default=24)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--logical-key", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_dataset(args: argparse.Namespace) -> PyBullet0713NoGTBoxDataset:
    return PyBullet0713NoGTBoxDataset(
        root=args.dataset_root,
        split=args.split,
        resolution=(int(args.height), int(args.width)),
        num_frames=int(args.num_frames),
        num_context_frames=8,
        sampling_strategy="prefix",
    )


def selected_records(args: argparse.Namespace, dataset) -> list[Any]:
    records = list(dataset.samples)
    selected = set(args.logical_key or [])
    if selected:
        records = [record for record in records if record.key in selected]
        missing = sorted(selected - {record.key for record in records})
        if missing:
            raise KeyError(f"logical keys not present in {args.split} split: {missing}")
    if args.limit is not None:
        if int(args.limit) <= 0:
            raise ValueError("--limit must be positive")
        records = records[: int(args.limit)]
    if not records:
        raise RuntimeError("no selected PyBullet records")
    return records


def cache_config(args: argparse.Namespace, selected_count: int, status: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "split": str(args.split),
        "selected_count": int(selected_count),
        "num_frames": int(args.num_frames),
        "anchor_frame": int(args.anchor_frame),
        "points_per_object": int(args.points_per_object),
        "track_height": TRACK_HEIGHT,
        "track_width": TRACK_WIDTH,
        "native_height": int(args.height),
        "native_width": int(args.width),
        "query_source": "dynamic_object_phrases -> GroundingDINO F04 -> SAM2 masks",
        "trajectory_source": "frozen CoTracker3 scaled_offline",
        "cotracker_checkpoint": str(args.cotracker_checkpoint.expanduser().resolve()),
    }


def load_frames(record, args: argparse.Namespace) -> np.ndarray:
    frames, _ = read_video_prefix(Path(record.video_path), int(args.num_frames))
    video = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
        value_range="minus_one_to_one",
        resize_mode="stretch",
    )
    return (
        ((video.permute(1, 0, 2, 3).float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )


def dynamic_object_phrases(record) -> list[str]:
    manifest = json.loads(Path(record.manifest_path).read_text(encoding="utf-8"))
    phrases = [
        str(value).strip()
        for value in manifest.get("dynamic_object_phrases", [])
        if str(value).strip()
    ]
    if not phrases:
        raise RuntimeError(f"{record.key}: no dynamic_object_phrases")
    return phrases


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1.0e-6)


def containment(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return max(intersection / max(area_a, 1.0e-6), intersection / max(area_b, 1.0e-6))


def detect_and_track_objects(provider, frames_tchw_01, phrases, anchor_frame: int):
    detections = []
    for phrase in phrases:
        output = provider.detector.detect(
            frames_tchw_01[int(anchor_frame)], phrase, guidance_box_xyxy=None
        )
        candidates = [
            {
                "box": np.asarray(box, dtype=np.float32),
                "score": float(score),
                "detected_phrase": str(detected_phrase),
            }
            for box, score, detected_phrase in zip(
                output.boxes_xyxy, output.scores, output.phrases
            )
        ]
        if not candidates:
            raise RuntimeError(f"GroundingDINO found no F04 candidate for {phrase!r}")
        detections.append(candidates)

    best = None
    for indices in itertools.product(*(range(len(items)) for items in detections)):
        selected = [detections[index][candidate] for index, candidate in enumerate(indices)]
        conflict = any(
            box_iou(selected[i]["box"], selected[j]["box"]) >= 0.50
            or containment(selected[i]["box"], selected[j]["box"]) >= 0.85
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        )
        if conflict:
            continue
        score = sum(math.log(max(item["score"], 1.0e-8)) for item in selected)
        if best is None or score > best[0]:
            best = (score, indices, selected)
    if best is None:
        raise RuntimeError(
            "could not assign distinct GroundingDINO boxes; "
            f"candidates={[len(items) for items in detections]}"
        )

    _, selected_indices, selected = best
    tracks = []
    for phrase, candidate in zip(phrases, selected):
        output = provider.tracker.track(
            frames_tchw_01,
            prompt_frame_idx=int(anchor_frame),
            prompt_box_xyxy=candidate["box"],
            caption="",
        )
        masks = np.asarray(output.masks_thw, dtype=np.uint8)
        if not bool(masks.any()) or not bool(masks[int(anchor_frame)].any()):
            raise RuntimeError(f"SAM2 produced an empty F04 track for {phrase!r}")
        tracks.append(
            DetectedObjectTrack(
                box_prompt_xyxy=candidate["box"],
                masks_thw=masks,
                boxes_t4=np.asarray(output.boxes_t4, dtype=np.float32),
                score=float(candidate["score"]),
                phrase=phrase,
                source_phrase=phrase,
            )
        )
    return SimpleNamespace(
        object_tracks=tracks,
        debug={
            "mode": "per_phrase_global_distinct_box_assignment_at_f04",
            "selected_candidate_indices": list(selected_indices),
            "selected_scores": [float(item["score"]) for item in selected],
        },
    )


def prepare_tracker_inputs(
    frames: np.ndarray,
    points_xy_native: np.ndarray,
    *,
    anchor_frame: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    native_height, native_width = frames.shape[1:3]
    video = torch.from_numpy(frames).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    video = F.interpolate(
        video, size=(TRACK_HEIGHT, TRACK_WIDTH), mode="bilinear", align_corners=True
    ).unsqueeze(0)
    points = torch.from_numpy(points_xy_native).to(device=device, dtype=torch.float32)
    points[:, 0] *= float(TRACK_WIDTH - 1) / float(native_width - 1)
    points[:, 1] *= float(TRACK_HEIGHT - 1) / float(native_height - 1)
    frame_ids = torch.full(
        (points.shape[0], 1), float(anchor_frame), device=device, dtype=points.dtype
    )
    return video, torch.cat((frame_ids, points), dim=-1).unsqueeze(0)


def replace_query_predictions(tracks, visibility, confidence, scaled_queries):
    anchor_mask = torch.zeros_like(visibility, dtype=torch.bool)
    anchor_tracks = torch.zeros_like(tracks)
    for batch_index in range(tracks.shape[0]):
        query_frames = scaled_queries[batch_index, :, 0].long()
        point_ids = torch.arange(tracks.shape[2], device=tracks.device)
        anchor_mask[batch_index, query_frames, point_ids] = True
        anchor_tracks[batch_index, query_frames, point_ids] = scaled_queries[
            batch_index, :, 1:
        ]
    return (
        torch.where(anchor_mask.unsqueeze(-1), anchor_tracks, tracks),
        torch.where(anchor_mask, torch.ones_like(visibility), visibility),
        torch.where(anchor_mask, torch.ones_like(confidence), confidence),
    )


def track_video_with_scores(predictor, video, queries):
    from cotracker.models.core.model_utils import get_points_on_a_grid

    batch, frames, channels, height, width = video.shape
    resized = F.interpolate(
        video.reshape(batch * frames, channels, height, width),
        tuple(predictor.interp_shape),
        mode="bilinear",
        align_corners=True,
    ).reshape(batch, frames, channels, *predictor.interp_shape)
    scaled_queries = queries.clone()
    scaled_queries[:, :, 1:] *= scaled_queries.new_tensor(
        (
            (predictor.interp_shape[1] - 1) / (width - 1),
            (predictor.interp_shape[0] - 1) / (height - 1),
        )
    )
    support = get_points_on_a_grid(
        predictor.support_grid_size, predictor.interp_shape, device=video.device
    )
    support = torch.cat((torch.zeros_like(support[:, :, :1]), support), dim=-1)
    model_queries = torch.cat((scaled_queries, support.repeat(batch, 1, 1)), dim=1)
    tracks, visibility, confidence, _ = predictor.model.forward(
        video=resized, queries=model_queries, iters=6
    )
    query_count = int(queries.shape[1])
    tracks, visibility, confidence = replace_query_predictions(
        tracks[:, :, :query_count],
        visibility[:, :, :query_count],
        confidence[:, :, :query_count],
        scaled_queries,
    )
    tracks = tracks * tracks.new_tensor(
        (
            (width - 1) / (predictor.interp_shape[1] - 1),
            (height - 1) / (predictor.interp_shape[0] - 1),
        )
    )
    return tracks, visibility, confidence


def tracks_inside_object_masks(tracks, object_masks, points_per_object: int):
    object_count, frames, mask_height, mask_width = object_masks.shape
    xy = tracks.detach().float().clone()
    xy[..., 0] *= float(mask_width - 1) / float(TRACK_WIDTH - 1)
    xy[..., 1] *= float(mask_height - 1) / float(TRACK_HEIGHT - 1)
    x = xy[..., 0].round().long()
    y = xy[..., 1].round().long()
    in_bounds = (x >= 0) & (x < mask_width) & (y >= 0) & (y < mask_height)
    result = torch.zeros(
        (tracks.shape[0], frames, tracks.shape[2]), dtype=torch.bool, device=tracks.device
    )
    mask_tensor = torch.from_numpy(object_masks).to(device=tracks.device)
    frame_index = torch.arange(frames, device=tracks.device)[None, :, None]
    for object_index in range(object_count):
        start = object_index * int(points_per_object)
        stop = start + int(points_per_object)
        sampled = mask_tensor[object_index][
            frame_index,
            y[:, :, start:stop].clamp(0, mask_height - 1),
            x[:, :, start:stop].clamp(0, mask_width - 1),
        ]
        result[:, :, start:stop] = sampled.bool() & in_bounds[:, :, start:stop]
    return result


def write_entry(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.safetensors")
    save_file({key: value.contiguous().cpu() for key, value in tensors.items()}, temporary, metadata)
    temporary.replace(path)


def entry_is_valid(cache_root: Path, record) -> bool:
    path = cache_root / trajectory_relative_path(record.key)
    if not path.is_file():
        return False
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            return (handle.metadata() or {}).get("logical_key") == record.key
    except Exception:  # noqa: BLE001
        return False


def finalize(args: argparse.Namespace, records: list[Any]) -> None:
    cache_root = args.cache_root.expanduser().resolve()
    missing = [record.key for record in records if not entry_is_valid(cache_root, record)]
    if missing:
        atomic_json(
            cache_root / "cache_config.json",
            {
                **cache_config(args, len(records), "building"),
                "complete_count": len(records) - len(missing),
                "missing_count": len(missing),
                "missing_examples": missing[:20],
            },
        )
        raise RuntimeError(
            f"trajectory cache incomplete: {len(missing)}/{len(records)} missing; "
            f"examples={missing[:3]}"
        )
    atomic_json(
        cache_root / "cache_config.json",
        {**cache_config(args, len(records), "complete"), "complete_count": len(records)},
    )
    print(f"[finalize] complete entries={len(records)} cache={cache_root}", flush=True)


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda:4"):
        raise ValueError("GPU 4 is prohibited by workspace rules")
    if not 0 <= int(args.worker_id) < int(args.num_workers):
        raise ValueError("worker-id must be in [0, num-workers)")
    if int(args.num_frames) != 49 or int(args.anchor_frame) != 4:
        raise ValueError("trajectory cache is fixed to 49 frames with F04 anchor")
    if int(args.points_per_object) <= 0:
        raise ValueError("points-per-object must be positive")
    dataset = build_dataset(args)
    records = selected_records(args, dataset)
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        finalize(args, records)
        return
    atomic_json(cache_root / "cache_config.json", cache_config(args, len(records), "building"))

    worker_records = [
        record
        for index, record in enumerate(records)
        if index % int(args.num_workers) == int(args.worker_id)
    ]
    pending = [
        record
        for record in worker_records
        if args.overwrite or not entry_is_valid(cache_root, record)
    ]
    print(
        f"[worker {args.worker_id}/{args.num_workers}] selected={len(worker_records)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        if int(args.num_workers) == 1:
            finalize(args, records)
        return

    from cotracker.predictor import CoTrackerPredictor

    device = torch.device(args.device)
    print("[models] loading GroundingDINO + SAM2 + frozen CoTracker3", flush=True)
    provider = build_provider(str(device), int(args.points_per_object))
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
    completed = []
    failures = []
    try:
        for position, record in enumerate(pending, start=1):
            print(f"[{position}/{len(pending)}] {record.key}", flush=True)
            try:
                frames = load_frames(record, args)
                phrases = dynamic_object_phrases(record)
                frames_tchw_01 = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
                objects = detect_and_track_objects(
                    provider, frames_tchw_01, phrases, int(args.anchor_frame)
                )
                masks_othw = np.stack(
                    [np.asarray(track.masks_thw, dtype=np.uint8) for track in objects.object_tracks]
                )
                points_on2 = np.stack(
                    [
                        sample_points_from_mask(
                            mask[int(args.anchor_frame)],
                            int(args.points_per_object),
                            avoid_edges=True,
                        )
                        for mask in masks_othw
                    ]
                ).astype(np.float32)
                object_count = len(phrases)
                if points_on2.shape != (
                    object_count,
                    int(args.points_per_object),
                    2,
                ):
                    raise RuntimeError(f"invalid sampled query shape: {points_on2.shape}")
                video, queries = prepare_tracker_inputs(
                    frames,
                    points_on2.reshape(-1, 2),
                    anchor_frame=int(args.anchor_frame),
                    device=device,
                )
                with torch.inference_mode():
                    tracks, visibility, confidence = track_video_with_scores(
                        predictor, video, queries
                    )
                    geometric = tracks_inside_object_masks(
                        tracks, masks_othw, int(args.points_per_object)
                    )
                time_steps = int(tracks.shape[1])
                tensors = {
                    "query_points": queries[0, :, 1:].reshape(
                        object_count, int(args.points_per_object), 2
                    ).float(),
                    "gt_tracks": tracks[0].reshape(
                        time_steps, object_count, int(args.points_per_object), 2
                    ).float(),
                    "gt_visibility_probability": visibility[0].reshape(
                        time_steps, object_count, int(args.points_per_object)
                    ).float(),
                    "gt_confidence_probability": confidence[0].reshape(
                        time_steps, object_count, int(args.points_per_object)
                    ).float(),
                    "gt_geometric_visibility": geometric[0].reshape(
                        time_steps, object_count, int(args.points_per_object)
                    ).to(torch.uint8),
                }
                source_stat = Path(record.video_path).stat()
                metadata = {
                    "logical_key": record.key,
                    "sample_uid": sample_uid(record.key),
                    "object_count": str(object_count),
                    "points_per_object": str(int(args.points_per_object)),
                    "object_phrases": json.dumps(phrases, ensure_ascii=True),
                    "source_size": str(source_stat.st_size),
                    "source_mtime_ns": str(source_stat.st_mtime_ns),
                    "assignment": json.dumps(objects.debug, ensure_ascii=True),
                }
                output_path = cache_root / trajectory_relative_path(record.key)
                write_entry(output_path, tensors, metadata)
                completed.append(record.key)
                del frames, objects, masks_othw, video, queries, tracks
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "logical_key": record.key,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[error] {record.key}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        del predictor, provider
        gc.collect()
        torch.cuda.empty_cache()

    status = {
        "worker_id": int(args.worker_id),
        "num_workers": int(args.num_workers),
        "selected": len(worker_records),
        "completed_this_run": completed,
        "failures": failures,
    }
    atomic_json(cache_root / f"worker_status_{args.worker_id:02d}.json", status)
    if failures:
        raise RuntimeError(f"trajectory cache worker had {len(failures)} failures")
    if int(args.num_workers) == 1:
        finalize(args, records)


if __name__ == "__main__":
    main()
