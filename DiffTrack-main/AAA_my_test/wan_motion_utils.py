#!/usr/bin/env python3
"""Shared data and metric utilities for the Wan motion Q/K experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")
OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/wan22_motion_qk")
WAN_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
WAN_CHECKPOINT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
COTRACKER_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
TARGET_HEIGHT = 704
TARGET_WIDTH = 1280
NUM_FRAMES = 49
TOKEN_STRIDE = 32
LATENT_ANCHOR_FRAMES = np.arange(0, NUM_FRAMES, 4, dtype=np.int64)


def atomic_write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_manifest(dataset_root: Path = DATASET_ROOT) -> dict:
    return json.loads((dataset_root / "dataset_manifest.json").read_text())


def enumerate_samples(manifest: dict, sample_types: list[str]) -> list[dict]:
    samples = []
    for case in manifest["cases"]:
        if "base" in sample_types:
            samples.append({**case["base"], "case_key": case["case_key"], "sample_type": "base"})
        pairs = case.get("pairs", {})
        if isinstance(pairs, list):
            variants = {item["attribute"]: item.get("variant", item) for item in pairs}
        else:
            variants = {key: value.get("variant", value) for key, value in pairs.items()}
        for sample_type in sample_types:
            if sample_type == "base":
                continue
            if sample_type in variants:
                samples.append({**variants[sample_type], "case_key": case["case_key"], "sample_type": sample_type})
    return samples


def find_sample(manifest: dict, case_key: str, sample_type: str) -> dict:
    matches = [sample for sample in enumerate_samples(manifest, [sample_type]) if sample["case_key"] == case_key]
    if len(matches) != 1:
        raise KeyError(f"Expected one sample for {case_key}/{sample_type}, found {len(matches)}")
    return matches[0]


def read_video(path: Path, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != num_frames:
        raise ValueError(f"{path} contains {len(frames)} frames, expected at least {num_frames}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    return resize_and_center_crop(video, TARGET_HEIGHT, TARGET_WIDTH, mode="bilinear")


def read_instance_ids(path: Path, num_frames: int = NUM_FRAMES) -> tuple[np.ndarray, list[str], list[int]]:
    with np.load(path) as data:
        ids = torch.from_numpy(data["instance_ids"][:num_frames]).unsqueeze(1).float()
        names = [str(value) for value in data["object_names"]]
        object_ids = [int(value) for value in data["object_ids"]]
    if ids.shape[0] != num_frames:
        raise ValueError(f"{path} contains {ids.shape[0]} masks, expected {num_frames}")
    ids = resize_and_center_crop(ids, TARGET_HEIGHT, TARGET_WIDTH, mode="nearest")
    return ids[:, 0].byte().numpy(), names, object_ids


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int, mode: str) -> torch.Tensor:
    source_height, source_width = tensor.shape[-2:]
    scale = max(height / source_height, width / source_width)
    resized_height = max(height, round(source_height * scale))
    resized_width = max(width, round(source_width * scale))
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    resized = F.interpolate(tensor, size=(resized_height, resized_width), mode=mode, **kwargs)
    top = (resized_height - height) // 2
    left = (resized_width - width) // 2
    return resized[..., top : top + height, left : left + width]


def erode_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return mask.astype(bool)
    if size % 2 == 0:
        size += 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def build_regions(
    instance_ids: np.ndarray,
    object_names: list[str],
    object_ids: list[int],
    object_erode_px: int = 15,
    background_erode_px: int = 31,
) -> list[dict]:
    frame_ids = instance_ids[0]
    regions = []
    object_masks = []
    for object_name, object_id in zip(object_names, object_ids):
        mask = frame_ids == object_id
        object_masks.append(mask)
        if not mask.any():
            continue
        regions.append(
            {
                "region_name": object_name,
                "region_type": "object",
                "object_id": object_id,
                "mask": erode_mask(mask, object_erode_px),
            }
        )
    union = np.logical_or.reduce(object_masks) if object_masks else np.zeros_like(frame_ids, dtype=bool)
    background = erode_mask(~union, background_erode_px)
    border = max(background_erode_px, TOKEN_STRIDE)
    background[:border] = False
    background[-border:] = False
    background[:, :border] = False
    background[:, -border:] = False
    regions.append(
        {
            "region_name": "background",
            "region_type": "background",
            "object_id": 0,
            "mask": background,
        }
    )
    return regions


def farthest_point_sample(mask: np.ndarray, count: int) -> np.ndarray:
    yx = np.argwhere(mask)
    if len(yx) < count:
        raise ValueError(f"Mask has {len(yx)} valid pixels, cannot sample {count}")
    center = yx.mean(axis=0)
    selected = [int(np.argmin(np.square(yx - center).sum(axis=1)))]
    min_distance = np.square(yx - yx[selected[0]]).sum(axis=1).astype(np.float64)
    for _ in range(1, count):
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(min_distance, np.square(yx - yx[next_index]).sum(axis=1))
    return yx[np.asarray(selected)][:, ::-1].astype(np.float32)


def free_space_gib(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / (1024**3)


def classify_region_tracks(tracks: np.ndarray, visibility: np.ndarray, region_type: str) -> str:
    if region_type == "background":
        return "background"
    visible_from_query = visibility & visibility[0:1]
    if not visible_from_query.any():
        return "low_motion_object"
    displacement = np.linalg.norm(tracks - tracks[0:1], axis=-1)
    displacement[~visible_from_query] = -np.inf
    max_displacement = displacement.max(axis=0)
    valid_points = np.isfinite(max_displacement)
    return "moving_object" if float(np.median(max_displacement[valid_points])) > 16.0 else "low_motion_object"


def compute_track_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    query: np.ndarray,
) -> dict:
    if predicted.shape != target.shape or visibility.shape != target.shape[:2]:
        raise ValueError(
            f"Metric shape mismatch: predicted={predicted.shape}, target={target.shape}, visibility={visibility.shape}"
        )
    valid = visibility.astype(bool).copy()
    valid[0] = False
    distance = np.linalg.norm(predicted - target, axis=-1)
    values = distance[valid]
    if not values.size:
        raise ValueError("No visible target points after the query frame")

    query_track = np.broadcast_to(query[None], target.shape)
    static_error = np.linalg.norm(query_track - target, axis=-1)[valid]
    predicted_motion = predicted - query[None]
    target_motion = target - query[None]
    pred_norm = np.linalg.norm(predicted_motion, axis=-1)
    target_norm = np.linalg.norm(target_motion, axis=-1)
    dot = (predicted_motion * target_motion).sum(axis=-1)
    direction_valid = valid & (target_norm > 2.0) & (pred_norm > 1e-6)
    direction = dot[direction_valid] / (pred_norm[direction_valid] * target_norm[direction_valid] + 1e-8)
    motion_error = np.linalg.norm(predicted_motion - target_motion, axis=-1)[valid]
    ratio_valid = valid & (target_norm > 2.0)
    ratio = pred_norm[ratio_valid] / (target_norm[ratio_valid] + 1e-8)

    return {
        "comparisons": int(values.size),
        "mean_error_px": float(values.mean()),
        "median_error_px": float(np.median(values)),
        "normalized_mean_error_tokens": float(values.mean() / TOKEN_STRIDE),
        "pck8": float((values < 8).mean() * 100),
        "pck16": float((values < 16).mean() * 100),
        "pck32": float((values < 32).mean() * 100),
        "pck64": float((values < 64).mean() * 100),
        "static_mean_error_px": float(static_error.mean()),
        "static_pck32": float((static_error < 32).mean() * 100),
        "mean_motion_error_px": float(motion_error.mean()),
        "mean_motion_magnitude_ratio": float(ratio.mean()) if ratio.size else None,
        "mean_direction_cosine": float(direction.mean()) if direction.size else None,
        "predicted_mean_displacement_px": float(pred_norm[valid].mean()),
        "target_mean_displacement_px": float(target_norm[valid].mean()),
        "relative_mean_error_improvement": float((static_error.mean() - values.mean()) / static_error.mean())
        if static_error.mean() > 0
        else 0.0,
    }


def compute_rigidity_error(
    predicted: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
) -> float | None:
    """Compare within-region pairwise-distance changes against CoTracker."""
    errors = []
    for frame_index in range(1, len(predicted)):
        valid_indices = np.flatnonzero(visibility[0] & visibility[frame_index])
        if len(valid_indices) < 2:
            continue
        pred_points = predicted[frame_index, valid_indices]
        target_points = target[frame_index, valid_indices]
        pred_dist = np.linalg.norm(pred_points[:, None] - pred_points[None, :], axis=-1)
        target_dist = np.linalg.norm(target_points[:, None] - target_points[None, :], axis=-1)
        upper = np.triu_indices(len(valid_indices), k=1)
        errors.extend(np.abs(pred_dist[upper] - target_dist[upper]).tolist())
    return float(np.mean(errors)) if errors else None
