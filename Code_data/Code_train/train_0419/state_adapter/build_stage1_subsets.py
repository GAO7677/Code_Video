#!/usr/bin/env python3
# 用途：提供 stage1 window 切分和状态读写的通用 helper。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


WINDOW_STRIDE = 4
STAGE1A_CONTEXT_LEN = 8
STAGE1A_SAFETY_MARGIN = 0
STAGE1B_CONTEXT_LEN = 8
STAGE1B_SAFETY_MARGIN = 0
STAGE1A_FUTURE_CANDIDATES = (8, 16, 24)
STAGE1B_FUTURE_CANDIDATES = (8, 16, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 subset helper.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--count_buckets", type=str, default="count_01,count_02")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_samples(dataset_root: Path) -> list[Path]:
    train_root = dataset_root / "train" / "rigid"
    if not train_root.exists():
        train_root = dataset_root / "train"
    sample_dirs: set[Path] = set()
    for meta_name in ("metadata.json", "meta.json"):
        for meta_path in train_root.rglob(meta_name):
            sample_dirs.add(meta_path.parent)
    return sorted(sample_dirs)


def load_raw_state(sample_dir: Path, fps: float) -> dict[str, Any]:
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        meta_path = sample_dir / "metadata.json"
    metadata = load_json(meta_path)

    anchor = np.load(sample_dir / "physics" / "anchor_targets.npz")
    kin = np.load(sample_dir / "physics" / "rigid_kinematics.npz")

    object_ids = np.asarray(anchor["object_ids"], dtype=np.int32)
    seg_ids = np.asarray(anchor["seg_ids"], dtype=np.int32)
    com_uv = np.asarray(anchor["com_uv"], dtype=np.float32)
    bbox_xyxy = np.asarray(anchor["bbox_xyxy"], dtype=np.float32)
    visibility_mask = np.asarray(anchor["visibility_mask"], dtype=np.uint8)
    center_depth = np.asarray(anchor["center_depth"], dtype=np.float32)
    linear_vel = np.asarray(kin["linear_vel"], dtype=np.float32)
    angular_vel = np.asarray(kin["angular_vel"], dtype=np.float32)
    com_pos = np.asarray(kin["com_pos"], dtype=np.float32)

    x1 = bbox_xyxy[..., 0]
    y1 = bbox_xyxy[..., 1]
    x2 = bbox_xyxy[..., 2]
    y2 = bbox_xyxy[..., 3]
    width_px = np.maximum(0.0, x2 - x1).astype(np.float32)
    height_px = np.maximum(0.0, y2 - y1).astype(np.float32)
    u = com_uv[..., 0].astype(np.float32)
    v = com_uv[..., 1].astype(np.float32)
    d = center_depth.astype(np.float32)
    dt = 1.0 / max(float(fps), 1e-6)
    du = np.zeros_like(u, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    dd = np.zeros_like(d, dtype=np.float32)
    if u.shape[0] > 1:
        du[1:] = (u[1:] - u[:-1]) / dt
        dv[1:] = (v[1:] - v[:-1]) / dt
        dd[1:] = (d[1:] - d[:-1]) / dt
    vis = visibility_mask.astype(np.float32)
    state_raw = np.stack([u, v, d, width_px, height_px, du, dv, dd, vis], axis=-1).astype(np.float32)

    return {
        "metadata": metadata,
        "state_raw": state_raw,
        "visibility_mask": visibility_mask.astype(np.uint8),
        "object_ids": object_ids,
        "seg_ids": seg_ids,
        "com_pos": com_pos,
        "linear_vel": linear_vel,
        "angular_vel": angular_vel,
        "bbox_xyxy": bbox_xyxy,
        "dt": float(dt),
    }


def normalize_state(state_raw: np.ndarray, *, width: float, height: float, depth_near: float, depth_far: float) -> np.ndarray:
    state_raw = np.asarray(state_raw, dtype=np.float32)
    out = state_raw.copy()
    width = max(float(width), 1e-6)
    height = max(float(height), 1e-6)
    depth_range = max(float(depth_far) - float(depth_near), 1e-6)
    out[..., 0] = out[..., 0] / width
    out[..., 1] = out[..., 1] / height
    out[..., 2] = (out[..., 2] - float(depth_near)) / depth_range
    out[..., 3] = out[..., 3] / width
    out[..., 4] = out[..., 4] / height
    out[..., 5] = out[..., 5] / width
    out[..., 6] = out[..., 6] / height
    out[..., 7] = out[..., 7] / depth_range
    out[..., 8] = np.clip(out[..., 8], 0.0, 1.0)
    return out.astype(np.float32)


def resolve_main_object_index(metadata: dict[str, Any], object_ids: np.ndarray) -> int:
    objects = metadata.get("objects") if isinstance(metadata.get("objects"), list) else []
    target_ids = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        role = str(obj.get("role", "")).strip().lower()
        source_tag = str(obj.get("source_tag", "")).strip().lower()
        if role == "target" or source_tag == "physxnet_main":
            try:
                target_ids.append(int(obj.get("object_id")))
            except Exception:
                continue
    if not target_ids and len(object_ids) > 0:
        return 0
    for idx, oid in enumerate(np.asarray(object_ids, dtype=np.int32).tolist()):
        if int(oid) in target_ids:
            return int(idx)
    return 0


def rgb_frame_paths(sample_dir: Path, indices: np.ndarray) -> list[Path]:
    rgb_dir = sample_dir / "rgb"
    return [rgb_dir / f"frame_{int(idx):03d}.png" for idx in np.asarray(indices).tolist()]


def window_has_visible_object_every_frame(visibility_mask: np.ndarray, start: int, end: int) -> bool:
    vis = np.asarray(visibility_mask, dtype=np.float32)
    if vis.ndim != 2:
        return False
    window = vis[int(start):int(end)]
    if window.size == 0:
        return False
    return bool(np.all(np.any(window > 0.5, axis=1)))


def future_main_object_visibility_ok(
    visibility_mask: np.ndarray,
    start: int,
    end: int,
    main_object_index: int,
    threshold: float,
) -> tuple[bool, float]:
    vis = np.asarray(visibility_mask, dtype=np.float32)
    if vis.ndim != 2 or vis.shape[1] <= int(main_object_index):
        return False, 0.0
    window = vis[int(start):int(end), int(main_object_index)]
    if window.size == 0:
        return False, 0.0
    ratio = float(np.mean(window > 0.5))
    return ratio >= float(threshold), ratio


if __name__ == "__main__":
    args = parse_args()
    print(find_samples(args.dataset_root))
