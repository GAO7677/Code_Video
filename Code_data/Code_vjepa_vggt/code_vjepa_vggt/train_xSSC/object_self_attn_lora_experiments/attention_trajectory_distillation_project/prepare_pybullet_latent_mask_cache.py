#!/usr/bin/env python3
"""Precompute GroundingDINO + SAM2 object masks for formal PyBullet training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
TRAJECTORY_ROOT = EXPERIMENT_ROOT / "object_cotracker_trajectory_project"
DIFFTRACK_ROOT = Path("/home/gaoya/Code_Video/DiffTrack-main")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for _path in (
    HERE,
    EXPERIMENT_ROOT,
    PACKAGE_ROOT,
    TRAJECTORY_ROOT,
    DIFFTRACK_ROOT,
    COTRACKER_ROOT,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from AAA_my_test.precompute_toydataset_sam2_regions import build_provider
from attention_trajectory_distillation_project.latent_mask_cache import (
    CACHE_SCHEMA_VERSION,
    mask_case_root,
)
from object_cotracker_trajectory_project.prepare_pybullet_trajectory_cache import (
    build_dataset,
    dynamic_object_phrases,
    load_frames,
    select_maximum_distinct_detections,
    selected_records,
)
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    DetectedObjectTrack,
)


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_CACHE_ROOT = Path("/data/gaoya/agent-data/cache/pybullet0713_gt_latent_mask_v1")


def audit_latent_mapping(
    masks_thw: np.ndarray,
) -> tuple[torch.Tensor, dict[str, float]]:
    masks = torch.from_numpy(np.asarray(masks_thw, dtype=np.float32))
    if tuple(masks.shape) != (49, 512, 896):
        raise ValueError(f"expected [49,512,896] object masks, got {masks.shape}")
    aligned = masks[torch.arange(13) * 4].unsqueeze(1)
    occupancy = F.adaptive_avg_pool2d(aligned, (32, 56)).squeeze(1)
    support = occupancy > 0
    reverse_support = (
        F.interpolate(support[:, None].float(), size=(512, 896), mode="nearest")
        .squeeze(1)
        .bool()
    )
    gt_support = aligned.squeeze(1).bool()
    missed = gt_support & ~reverse_support
    intersection = (gt_support & reverse_support).sum().double()
    reverse_area = reverse_support.sum().double()
    union = (gt_support | reverse_support).sum().double()
    gt_area = gt_support.sum().double()
    if bool(missed.any()):
        raise RuntimeError(
            f"latent-mask reverse mapping missed {int(missed.sum().item())} GT pixels"
        )
    occupancy_sum = occupancy.flatten(1).sum(-1)
    if float(occupancy_sum[1].item()) <= 0.0:
        raise RuntimeError("GT-role object0 mask is empty at source latent frame F04")
    valid = (occupancy_sum > 0) & (torch.arange(13) > 1)
    return valid, {
        "reverse_recall": float(intersection / gt_area.clamp_min(1)),
        "reverse_precision": float(intersection / reverse_area.clamp_min(1)),
        "reverse_iou": float(intersection / union.clamp_min(1)),
        "missed_gt_pixels": float(missed.sum().item()),
    }


def detect_and_track_gt_role_object(
    provider, frames_tchw_01, phrases, anchor_frame: int
):
    detections = []
    for phrase in phrases:
        output = provider.detector.detect(
            frames_tchw_01[int(anchor_frame)], phrase, guidance_box_xyxy=None
        )
        detections.append(
            [
                {
                    "box": np.asarray(box, dtype=np.float32),
                    "score": float(score),
                    "detected_phrase": str(detected_phrase),
                }
                for box, score, detected_phrase in zip(
                    output.boxes_xyxy, output.scores, output.phrases
                )
            ]
        )
    selected_phrase_indices, selected_indices, selected = (
        select_maximum_distinct_detections(detections)
    )
    role_phrase = phrases[selected_phrase_indices[0]]
    role_candidate = selected[0]
    output = provider.tracker.track(
        frames_tchw_01,
        prompt_frame_idx=int(anchor_frame),
        prompt_box_xyxy=role_candidate["box"],
        caption="",
    )
    masks = np.asarray(output.masks_thw, dtype=np.uint8)
    if not bool(masks.any()) or not bool(masks[int(anchor_frame)].any()):
        raise RuntimeError(f"SAM2 produced an empty F04 track for {role_phrase!r}")
    track = DetectedObjectTrack(
        box_prompt_xyxy=role_candidate["box"],
        masks_thw=masks,
        boxes_t4=np.asarray(output.boxes_t4, dtype=np.float32),
        score=float(role_candidate["score"]),
        phrase=role_phrase,
        source_phrase=role_phrase,
    )
    return SimpleNamespace(
        object_tracks=[track],
        selected_phrases=[role_phrase],
        debug={
            "mode": "maximum_distinct_visible_box_assignment_at_f04_then_object0",
            "selected_phrase_indices": list(selected_phrase_indices),
            "selected_candidate_indices": list(selected_indices),
            "selected_scores": [float(item["score"]) for item in selected],
            "supervised_phrase_index": int(selected_phrase_indices[0]),
            "supervised_candidate_index": int(selected_indices[0]),
            "dropped_phrases": [
                phrase
                for index, phrase in enumerate(phrases)
                if index not in selected_phrase_indices
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--split", default="train", choices=("train", "val", "test", "all")
    )
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


def cache_config(
    args: argparse.Namespace, selected_count: int, status: str
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "split": str(args.split),
        "selected_count": int(selected_count),
        "num_frames": int(args.num_frames),
        "anchor_frame": int(args.anchor_frame),
        "native_height": int(args.height),
        "native_width": int(args.width),
        "latent_grid": [13, 32, 56],
        "supervised_object_index": 0,
        "query_source": (
            "dynamic_object_phrases -> maximum distinct visible GroundingDINO "
            "boxes at F04 -> SAM2 masks"
        ),
    }


def write_npz(path: Path, *, masks_othw: np.ndarray, frame_indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            masks_othw=np.asarray(masks_othw, dtype=np.uint8),
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def entry_is_valid(cache_root: Path, record) -> bool:
    case_root = mask_case_root(cache_root, record.key)
    arrays_path = case_root / "object_masks.npz"
    metadata_path = case_root / "entry.json"
    if not arrays_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("logical_key") == record.key
            and int(metadata.get("object_count", 0)) > 0
            and float(metadata.get("reverse_recall", 0.0)) >= 1.0
        )
    except Exception:  # noqa: BLE001
        return False


def finalize(args: argparse.Namespace, records: list[Any]) -> None:
    cache_root = args.cache_root.expanduser().resolve()
    missing = [
        record.key for record in records if not entry_is_valid(cache_root, record)
    ]
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
            f"latent-mask cache incomplete: {len(missing)}/{len(records)} missing; "
            f"examples={missing[:3]}"
        )
    atomic_json(
        cache_root / "cache_config.json",
        {
            **cache_config(args, len(records), "complete"),
            "complete_count": len(records),
        },
    )
    print(f"[finalize] complete entries={len(records)} cache={cache_root}", flush=True)


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda:4"):
        raise ValueError("GPU 4 is prohibited by workspace rules")
    if not 0 <= int(args.worker_id) < int(args.num_workers):
        raise ValueError("worker-id must be in [0, num-workers)")
    if (
        int(args.height),
        int(args.width),
        int(args.num_frames),
        int(args.anchor_frame),
    ) != (
        512,
        896,
        49,
        4,
    ):
        raise ValueError(
            "formal latent-mask cache is fixed to 49x512x896 with F04 anchor"
        )

    dataset = build_dataset(args)
    records = selected_records(args, dataset)
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        finalize(args, records)
        return
    atomic_json(
        cache_root / "cache_config.json", cache_config(args, len(records), "building")
    )

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

    device = torch.device(args.device)
    print("[models] loading GroundingDINO + SAM2", flush=True)
    provider = build_provider(str(device), int(args.points_per_object))
    completed: list[str] = []
    failures: list[dict[str, str]] = []
    try:
        for position, record in enumerate(pending, start=1):
            print(f"[{position}/{len(pending)}] {record.key}", flush=True)
            try:
                frames = load_frames(record, args)
                requested_phrases = dynamic_object_phrases(record)
                frames_tchw_01 = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
                objects = detect_and_track_gt_role_object(
                    provider,
                    frames_tchw_01,
                    requested_phrases,
                    int(args.anchor_frame),
                )
                masks_othw = np.stack(
                    [
                        np.asarray(track.masks_thw, dtype=np.uint8)
                        for track in objects.object_tracks
                    ]
                )
                if tuple(masks_othw.shape[1:]) != (49, 512, 896):
                    raise RuntimeError(f"invalid SAM2 mask shape: {masks_othw.shape}")
                valid, mapping_audit = audit_latent_mapping(masks_othw[0])
                source_stat = Path(record.video_path).stat()
                metadata = {
                    "logical_key": record.key,
                    "object_count": int(masks_othw.shape[0]),
                    "object_phrases": list(objects.selected_phrases),
                    "mask_shape": list(map(int, masks_othw.shape)),
                    "frame_indices": list(range(49)),
                    "valid_future_latent_frames_object0": int(valid.sum().item()),
                    "reverse_recall": mapping_audit["reverse_recall"],
                    "reverse_precision": mapping_audit["reverse_precision"],
                    "reverse_iou": mapping_audit["reverse_iou"],
                    "missed_gt_pixels": mapping_audit["missed_gt_pixels"],
                    "source_size": int(source_stat.st_size),
                    "source_mtime_ns": int(source_stat.st_mtime_ns),
                    "assignment": objects.debug,
                }
                case_root = mask_case_root(cache_root, record.key)
                write_npz(
                    case_root / "object_masks.npz",
                    masks_othw=masks_othw,
                    frame_indices=np.arange(49, dtype=np.int64),
                )
                atomic_json(case_root / "entry.json", metadata)
                completed.append(record.key)
                del frames, frames_tchw_01, objects, masks_othw
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
        del provider
        gc.collect()
        torch.cuda.empty_cache()

    atomic_json(
        cache_root / f"worker_status_{int(args.worker_id):02d}.json",
        {
            "worker_id": int(args.worker_id),
            "num_workers": int(args.num_workers),
            "selected": len(worker_records),
            "completed_this_run": completed,
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError(f"latent-mask cache worker had {len(failures)} failures")
    if int(args.num_workers) == 1:
        finalize(args, records)


if __name__ == "__main__":
    main()
