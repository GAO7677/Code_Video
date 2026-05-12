#!/usr/bin/env python3
"""Repair vertically inverted depth artifacts in existing TDW Genesis-style exports."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


EXPORT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_genesis_format_exports")


def depth_norm(depth_metric: np.ndarray, near: float, far: float) -> np.ndarray:
    arr = np.asarray(depth_metric, dtype=np.float32)
    denom = max(float(far) - float(near), 1e-6)
    out = np.zeros(arr.shape + (1,), dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    out[..., 0][valid] = np.clip((arr[valid] - float(near)) / denom, 0.0, 1.0)
    return out


def depth_to_uint8(depth_normalized: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth_normalized, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def depth_to_vis(depth_metric: np.ndarray, near: float, far: float) -> np.ndarray:
    vis = depth_to_uint8(depth_norm(depth_metric, near=near, far=far))
    return np.repeat(vis[..., None], 3, axis=2)


def recompute_center_depth(seg_arr: np.ndarray, depth_metric_arr: np.ndarray, seg_ids: np.ndarray) -> np.ndarray:
    num_frames = int(seg_arr.shape[0])
    num_objects = int(seg_ids.shape[0])
    center_depth = np.zeros((num_frames, num_objects), dtype=np.float32)
    for frame_idx in range(num_frames):
        frame_seg = seg_arr[frame_idx]
        frame_depth = depth_metric_arr[frame_idx]
        for obj_idx, seg_id in enumerate(seg_ids.tolist()):
            mask = frame_seg == int(seg_id)
            if np.any(mask):
                center_depth[frame_idx, obj_idx] = float(np.median(frame_depth[mask]))
    return center_depth


def repair_case(case_dir: Path) -> None:
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    camera_intrinsics = meta.get("camera_intrinsics") or {}
    near = float(camera_intrinsics.get("near", 0.1))
    far = float(camera_intrinsics.get("far", 100.0))

    physics_dir = case_dir / "physics"
    depth_metric_path = physics_dir / "depth_metric.npy"
    seg_path = physics_dir / "seg.npy"
    anchor_targets_path = physics_dir / "anchor_targets.npz"
    rigid_kinematics_path = physics_dir / "rigid_kinematics.npz"

    depth_metric_arr = np.asarray(np.load(depth_metric_path), dtype=np.float32)
    depth_metric_arr = np.flip(depth_metric_arr, axis=1).copy()
    np.save(depth_metric_path, depth_metric_arr)

    depth_dir = case_dir / "depth"
    for frame_idx, depth_metric in enumerate(depth_metric_arr):
        imageio.imwrite(depth_dir / f"frame_{frame_idx:03d}.png",
                        depth_to_uint8(depth_norm(depth_metric, near=near, far=far)))

    imageio.mimwrite(case_dir / "videos" / "depth.mp4",
                     [depth_to_uint8(depth_norm(d, near=near, far=far)) for d in depth_metric_arr],
                     fps=24,
                     quality=8)
    imageio.mimwrite(case_dir / "visualizations" / "depth_vis.mp4",
                     [depth_to_vis(d, near=near, far=far) for d in depth_metric_arr],
                     fps=24,
                     quality=8)

    if seg_path.exists() and anchor_targets_path.exists():
        seg_arr = np.asarray(np.load(seg_path), dtype=np.int32)
        anchor_targets_payload = dict(np.load(anchor_targets_path, allow_pickle=True))
        seg_ids = np.asarray(anchor_targets_payload["seg_ids"], dtype=np.int32)
        anchor_targets_payload["center_depth"] = recompute_center_depth(seg_arr=seg_arr,
                                                                         depth_metric_arr=depth_metric_arr,
                                                                         seg_ids=seg_ids)
        np.savez_compressed(anchor_targets_path, **anchor_targets_payload)

        if rigid_kinematics_path.exists():
            rigid_payload = dict(np.load(rigid_kinematics_path, allow_pickle=True))
            rigid_payload["center_depth"] = anchor_targets_payload["center_depth"].astype(np.float32)
            np.savez_compressed(rigid_kinematics_path, **rigid_payload)

    print(f"REPAIRED {case_dir}", flush=True)


def main() -> None:
    meta_paths = sorted(EXPORT_ROOT.glob("train/rigid/*/*/*/meta.json"))
    for meta_path in meta_paths:
        repair_case(meta_path.parent)


if __name__ == "__main__":
    main()
