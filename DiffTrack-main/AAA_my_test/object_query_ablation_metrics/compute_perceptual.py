#!/usr/bin/env python3
"""Compute mask-aligned DINOv2/LPIPS metrics and their exact-input visualizations."""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.object_query_ablation_metrics.common import (  # noqa: E402
    FRAME_COUNT,
    OBJECTS,
    OUTPUT_ROOT,
    atomic_json,
    load_inventory,
    load_video_frames,
    safe_id,
)


DINO_ROOT = Path("/home/gaoya/dinov2-main")
DINO_WEIGHT = Path("/home/gaoya/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth")
ANCHORS = np.arange(0, FRAME_COUNT, 4, dtype=np.int64)
MONTAGE_FRAMES = np.asarray([0, 12, 24, 36, 48], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", default="", help="optional exact ablation id")
    return parser.parse_args()


def load_masks(video_id: str) -> np.ndarray:
    path = OUTPUT_ROOT / "masks" / f"{safe_id(video_id)}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        masks = arrays["masks"].astype(bool)
    if masks.shape[:2] != (FRAME_COUNT, 2):
        raise RuntimeError(f"invalid masks: {path}")
    return masks


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.where(mask)
    if not len(x):
        raise RuntimeError("empty object mask")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def load_track_centers(video_id: str) -> np.ndarray:
    path = OUTPUT_ROOT / "tracks" / f"{safe_id(video_id)}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        visibility = arrays["visibility"].astype(bool)
    centers = np.full((FRAME_COUNT, 2, 2), np.nan, dtype=np.float32)
    for object_index in range(2):
        part = slice(object_index * 8, (object_index + 1) * 8)
        for frame_index in range(FRAME_COUNT):
            valid = visibility[frame_index, part] & np.isfinite(tracks[frame_index, part]).all(axis=1)
            if valid.any():
                centers[frame_index, object_index] = np.median(tracks[frame_index, part][valid], axis=0)
    for object_index in range(2):
        for frame_index in range(FRAME_COUNT):
            if not np.isfinite(centers[frame_index, object_index]).all():
                centers[frame_index, object_index] = centers[max(0, frame_index - 1), object_index]
    return centers


def crop_sides(video_ids: list[str]) -> list[int]:
    extents = [[], []]
    for video_id in video_ids:
        masks = load_masks(video_id)
        for object_index in range(2):
            for mask in masks[:, object_index]:
                if mask.any():
                    x0, y0, x1, y1 = mask_box(mask)
                    extents[object_index].append(max(x1 - x0, y1 - y0))
    return [
        int(np.clip(np.ceil(np.percentile(values, 99) * 1.5 / 14) * 14, 98, 448))
        for values in extents
    ]


def centered_crop(
    frame: np.ndarray,
    mask: np.ndarray,
    side: int,
    fallback_center: np.ndarray,
    output_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.where(mask)
    if len(x):
        cx, cy = float(x.mean()), float(y.mean())
    else:
        cx, cy = (float(fallback_center[0]), float(fallback_center[1]))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    image = np.full((side, side, 3), 127, dtype=np.uint8)
    local_mask = np.zeros((side, side), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(frame.shape[1], x1), min(frame.shape[0], y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    if sx1 > sx0 and sy1 > sy0:
        image[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = frame[sy0:sy1, sx0:sx1]
        local_mask[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = mask[sy0:sy1, sx0:sx1]
    image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    local_mask = cv2.resize(local_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST) > 0
    image = np.where(local_mask[..., None], image, 127).astype(np.uint8)
    return image, local_mask


def object_crops(
    frames: np.ndarray,
    masks: np.ndarray,
    centers: np.ndarray,
    object_index: int,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        centered_crop(frame, mask, side, center)
        for frame, mask, center in zip(
            frames, masks[:, object_index], centers[:, object_index], strict=True
        )
    ]
    return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def load_dino(device: torch.device):
    model = torch.hub.load(str(DINO_ROOT), "dinov2_vitl14", source="local", pretrained=False)
    state = torch.load(DINO_WEIGHT, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


def dino_tokens(
    model, crops: np.ndarray, masks: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    pooled, tokens = [], []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = torch.from_numpy(crops[start : start + batch_size]).to(device).float()
            batch = batch.permute(0, 3, 1, 2).div(255.0)
            features = model.forward_features((batch - mean) / std)["x_norm_patchtokens"]
            patch_mask = torch.from_numpy(masks[start : start + batch_size]).to(device).float()[:, None]
            patch_mask = F.interpolate(patch_mask, size=(16, 16), mode="area").flatten(1)
            pooled_batch = (features * patch_mask[..., None]).sum(1) / patch_mask.sum(1, keepdim=True).clamp_min(1e-6)
            pooled.append(F.normalize(pooled_batch, dim=-1).cpu().numpy().astype(np.float32))
            tokens.append(F.normalize(features, dim=-1).cpu().numpy().astype(np.float16))
    return np.concatenate(pooled), np.concatenate(tokens)


def lpips_maps(model, left: np.ndarray, right: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(left), batch_size):
            a = torch.from_numpy(left[start : start + batch_size]).to(device).float().permute(0, 3, 1, 2)
            b = torch.from_numpy(right[start : start + batch_size]).to(device).float().permute(0, 3, 1, 2)
            outputs.append(model(a.div(127.5).sub(1), b.div(127.5).sub(1)).float().cpu().numpy()[:, 0])
    return np.concatenate(outputs)


def outside_frames(
    left: np.ndarray,
    right: np.ndarray,
    left_masks: np.ndarray,
    right_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    left_out, right_out = [], []
    for a, b, ma, mb in zip(left, right, left_masks, right_masks, strict=True):
        union = np.logical_or(ma.any(axis=0), mb.any(axis=0)).astype(np.uint8)
        outside = cv2.dilate(union, kernel, iterations=1) == 0
        a_masked = np.where(outside[..., None], a, 127).astype(np.uint8)
        b_masked = np.where(outside[..., None], b, 127).astype(np.uint8)
        left_out.append(cv2.resize(a_masked, (320, 176), interpolation=cv2.INTER_AREA))
        right_out.append(cv2.resize(b_masked, (320, 176), interpolation=cv2.INTER_AREA))
    return np.stack(left_out), np.stack(right_out)


def heatmap(value: np.ndarray, size: int = 224) -> np.ndarray:
    value = value.astype(np.float32)
    scale = float(np.percentile(value, 99))
    normalized = np.clip(value / max(scale, 1e-8), 0, 1)
    image = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)


def montage(
    path: Path,
    candidate_id: str,
    reference_id: str,
    object_name: str,
    candidate_crops: np.ndarray,
    reference_crops: np.ndarray,
    candidate_tokens: np.ndarray,
    reference_tokens: np.ndarray,
    lpips: np.ndarray,
) -> None:
    tile = 180
    header_height = 56
    rows = []
    for frame_index in MONTAGE_FRAMES:
        anchor_index = int(np.where(ANCHORS == frame_index)[0][0])
        dino_delta = 1.0 - np.sum(
            candidate_tokens[anchor_index].astype(np.float32)
            * reference_tokens[anchor_index].astype(np.float32), axis=-1
        ).reshape(16, 16)
        images = [
            cv2.resize(candidate_crops[frame_index], (tile, tile), interpolation=cv2.INTER_AREA),
            cv2.resize(reference_crops[frame_index], (tile, tile), interpolation=cv2.INTER_AREA),
            heatmap(dino_delta, tile),
            heatmap(lpips[frame_index], tile),
        ]
        row = np.concatenate(images, axis=1)
        cv2.putText(row, f"F{frame_index:02d}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(row)
    body = np.concatenate(rows, axis=0)
    header = np.full((header_height, body.shape[1], 3), (238, 233, 221), np.uint8)
    title = f"{candidate_id} | {object_name} | ref={reference_id}"
    cv2.putText(header, title[:105], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 38, 31), 1, cv2.LINE_AA)
    cv2.putText(header, "candidate crop | reference crop | DINO 1-cos patch map | LPIPS spatial map", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 38, 31), 1, cv2.LINE_AA)
    output = np.concatenate([header, body], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise RuntimeError(f"failed to write {path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    videos = load_inventory(include_source=True)
    video_map = {row["id"]: row for row in videos}
    candidates = [row for row in videos if row["id"] not in {"baseline", "source_gt_video"}]
    if args.only:
        candidates = [row for row in candidates if row["id"] == args.only]
        if not candidates:
            raise ValueError(f"unknown ablation id: {args.only}")
    output = OUTPUT_ROOT / "perceptual"
    feature_root = output / "features"
    montage_root = output / "montages"
    feature_root.mkdir(parents=True, exist_ok=True)
    montage_root.mkdir(parents=True, exist_ok=True)
    ids = [row["id"] for row in videos]
    sides = crop_sides(ids)

    dino = load_dino(device)
    from torchmetrics.functional.image.lpips import _NoTrainLpips
    lpips_model = _NoTrainLpips(net="alex", spatial=True).to(device).eval().requires_grad_(False)

    frame_cache: dict[str, np.ndarray] = {}
    mask_cache: dict[str, np.ndarray] = {}
    crop_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    feature_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    center_cache: dict[str, np.ndarray] = {}

    def frames(video_id: str) -> np.ndarray:
        if video_id not in frame_cache:
            frame_cache[video_id] = load_video_frames(Path(video_map[video_id]["path"]))[0]
        return frame_cache[video_id]

    def masks(video_id: str) -> np.ndarray:
        if video_id not in mask_cache:
            mask_cache[video_id] = load_masks(video_id)
        return mask_cache[video_id]

    def crops(video_id: str, object_index: int) -> tuple[np.ndarray, np.ndarray]:
        key = (video_id, object_index)
        if key not in crop_cache:
            if video_id not in center_cache:
                center_cache[video_id] = load_track_centers(video_id)
            crop_cache[key] = object_crops(
                frames(video_id), masks(video_id), center_cache[video_id],
                object_index, sides[object_index]
            )
        return crop_cache[key]

    def features(video_id: str, object_index: int) -> tuple[np.ndarray, np.ndarray]:
        key = (video_id, object_index)
        if key not in feature_cache:
            image, mask = crops(video_id, object_index)
            feature_cache[key] = dino_tokens(dino, image[ANCHORS], mask[ANCHORS], device, args.batch_size)
        return feature_cache[key]

    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate["id"])
        record: dict[str, Any] = {"id": candidate_id, "objects": {}, "references": {}}
        for object_index, object_name in enumerate(OBJECTS):
            candidate_crops, _candidate_crop_masks = crops(candidate_id, object_index)
            candidate_pooled, candidate_tokens = features(candidate_id, object_index)
            record["objects"][object_name] = {}
            for reference_id in ("baseline", "source_gt_video"):
                reference_crops, _reference_crop_masks = crops(reference_id, object_index)
                reference_pooled, reference_tokens = features(reference_id, object_index)
                dino_cosine = np.sum(candidate_pooled * reference_pooled, axis=-1)
                lpips = lpips_maps(
                    lpips_model, candidate_crops, reference_crops, device, args.batch_size
                )
                montage_path = montage_root / f"{safe_id(candidate_id)}__{object_name}__{reference_id}.jpg"
                montage(
                    montage_path, candidate_id, reference_id, object_name,
                    candidate_crops, reference_crops, candidate_tokens,
                    reference_tokens, lpips,
                )
                record["objects"][object_name][reference_id] = {
                    "dino_cosine_mean": round(float(dino_cosine.mean()), 8),
                    "dino_cosine_by_anchor": [round(float(value), 8) for value in dino_cosine],
                    "dino_anchor_frames": ANCHORS.tolist(),
                    "lpips_mean": round(float(lpips.mean()), 8),
                    "lpips_by_frame": [round(float(value), 8) for value in lpips.mean(axis=(1, 2))],
                    "montage": str(montage_path.relative_to(OUTPUT_ROOT)),
                    "crop_side_px": sides[object_index],
                }
        for reference_id in ("baseline", "source_gt_video"):
            left, right = outside_frames(
                frames(candidate_id), frames(reference_id), masks(candidate_id), masks(reference_id)
            )
            outside_lpips = lpips_maps(lpips_model, left, right, device, args.batch_size)
            record["references"][reference_id] = {
                "outside_object_lpips_mean": round(float(outside_lpips.mean()), 8),
                "outside_object_lpips_by_frame": [
                    round(float(value), 8) for value in outside_lpips.mean(axis=(1, 2))
                ],
            }
        records.append(record)
        print(f"[{index:02d}/{len(candidates):02d}] perceptual {candidate_id}", flush=True)
        if len(frame_cache) > 6:
            for key in list(frame_cache):
                if key not in {candidate_id, "baseline", "source_gt_video"}:
                    frame_cache.pop(key, None)
                    mask_cache.pop(key, None)
                    center_cache.pop(key, None)
            for key in list(crop_cache):
                if key[0] not in {candidate_id, "baseline", "source_gt_video"}:
                    crop_cache.pop(key, None)
                    feature_cache.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()

    atomic_json(
        output / "perceptual_metrics.json",
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dino": "official DINOv2 ViT-L/14, mask-pooled normalized patch tokens",
            "dino_frames": ANCHORS.tolist(),
            "lpips": "LPIPS v0.1 AlexNet through torchmetrics, spatial=True",
            "lpips_frames": list(range(FRAME_COUNT)),
            "alignment": "per-video mask centroid translation; fixed native crop side per object; no scale alignment",
            "crop_sides_px": dict(zip(OBJECTS, sides, strict=True)),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
