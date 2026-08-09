#!/usr/bin/env python3
"""Propagate the two F00 object prompts through all videos with SAM2.1 Large."""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.object_query_ablation_metrics.common import (  # noqa: E402
    FRAME_COUNT,
    HEIGHT,
    OUTPUT_ROOT,
    WIDTH,
    atomic_json,
    atomic_npz,
    load_inventory,
    load_query_data,
    load_video_frames,
    safe_id,
    sha256_file,
)


SAM2_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt"
)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", default="", help="optional exact video id")
    return parser.parse_args()


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    y, x = np.where(mask)
    if not len(x):
        raise RuntimeError("empty F00 prompt mask")
    return np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=np.float32)


def cache_valid(path: Path, digest: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as arrays:
            return (
                tuple(arrays["masks"].shape) == (FRAME_COUNT, 2, HEIGHT, WIDTH)
                and str(arrays["video_sha256"].item()) == digest
            )
    except (OSError, KeyError, ValueError):
        return False


def save_jpegs(frames: np.ndarray, directory: Path) -> None:
    for index, frame in enumerate(frames):
        path = directory / f"{index:05d}.jpg"
        ok = cv2.imwrite(
            str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not ok:
            raise RuntimeError(f"failed to write {path}")


def track_masks(
    predictor,
    frames: np.ndarray,
    points: np.ndarray,
    prompt_masks: np.ndarray,
    temporary_root: Path,
) -> np.ndarray:
    masks = np.zeros((FRAME_COUNT, 2, HEIGHT, WIDTH), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="sam2_001460_", dir=temporary_root) as tmp:
        frame_dir = Path(tmp)
        save_jpegs(frames, frame_dir)
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for object_index in range(2):
                part = slice(object_index * 8, (object_index + 1) * 8)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=object_index + 1,
                    points=points[part],
                    labels=np.ones(8, dtype=np.int32),
                    box=bbox_from_mask(prompt_masks[object_index]),
                )
            for frame_index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=0
            ):
                for local_index, object_id in enumerate(object_ids):
                    masks[int(frame_index), int(object_id) - 1] = (
                        logits[local_index, 0] > 0
                    ).detach().cpu().numpy().astype(np.uint8)
        predictor.reset_state(state)
        del state
    return masks


def main() -> None:
    args = parse_args()
    videos = load_inventory(include_source=True)
    if args.only:
        videos = [row for row in videos if row["id"] == args.only]
        if not videos:
            raise ValueError(f"unknown video id: {args.only}")
    output = OUTPUT_ROOT / "masks"
    temporary_root = OUTPUT_ROOT / "tmp"
    output.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    points, _slices, prompt_masks = load_query_data()

    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        SAM2_CONFIG, str(SAM2_CHECKPOINT), device=args.device
    )
    predictor.fill_hole_area = 0
    records = []
    try:
        for index, video in enumerate(videos, start=1):
            video_id = str(video["id"])
            video_path = Path(video["path"])
            digest = sha256_file(video_path)
            mask_path = output / f"{safe_id(video_id)}.npz"
            if cache_valid(mask_path, digest) and not args.overwrite:
                print(f"[{index:02d}/{len(videos):02d}] reuse {video_id}", flush=True)
            else:
                frames, _fps = load_video_frames(video_path)
                masks = track_masks(
                    predictor, frames, points, prompt_masks, temporary_root
                )
                atomic_npz(
                    mask_path,
                    masks=masks,
                    video_id=np.asarray(video_id),
                    video_path=np.asarray(str(video_path)),
                    video_sha256=np.asarray(digest),
                    segmenter=np.asarray("SAM2.1 Hiera Large video predictor"),
                    prompt_frame=np.int32(0),
                    prompt_points=points,
                    prompt_masks=prompt_masks.astype(np.uint8),
                )
                del frames, masks
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[{index:02d}/{len(videos):02d}] segmented {video_id}", flush=True)
            with np.load(mask_path, allow_pickle=False) as arrays:
                masks = arrays["masks"]
                areas = masks.sum(axis=(2, 3))
                f00_iou = []
                for object_index in range(2):
                    left, right = masks[0, object_index] > 0, prompt_masks[object_index]
                    f00_iou.append(float(np.logical_and(left, right).sum() / np.logical_or(left, right).sum()))
            records.append(
                {
                    "id": video_id,
                    "mask_file": str(mask_path.relative_to(OUTPUT_ROOT)),
                    "mean_area_px": [round(float(value), 3) for value in areas.mean(axis=0)],
                    "nonempty_rate": [
                        round(float(value), 6) for value in (areas > 0).mean(axis=0)
                    ],
                    "f00_prompt_iou": [round(value, 6) for value in f00_iou],
                    "video_sha256": digest,
                }
            )
    finally:
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "segmenter": "SAM2.1 Hiera Large video predictor",
            "prompt_frame": 0,
            "video_count": len(records),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
