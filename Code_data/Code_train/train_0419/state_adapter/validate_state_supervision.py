#!/usr/bin/env python3
"""Validate 9D state supervision against dense observations for Genesis and MOVI-D."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_movi_d_physics import (
    compute_state_9d,
    decode_float_tensor,
    decode_image_sequence,
    decode_rgb_frames,
    iter_serialized_records,
    parse_example,
    uint16_to_metric,
)


STATE_NAMES = ["u", "v", "d", "w", "h", "du", "dv", "dd", "vis"]
STATE_COLORS = [
    "#ff6b6b",
    "#4dabf7",
    "#51cf66",
    "#f59f00",
    "#845ef7",
    "#e64980",
    "#12b886",
    "#fd7e14",
]


def fmt_num(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate state supervision for Genesis and MOVI-D samples.",
    )
    parser.add_argument(
        "--movi_root",
        type=Path,
        default=Path("/data/gaoya/dataset/kubric_tfds_movi-d/mytrain/movi_d_physics/train/rigid/movi_d"),
    )
    parser.add_argument(
        "--genesis_root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/gaoya/dataset/kubric_tfds_movi-d/mytrain/state_validation"),
    )
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=128,
        help="0 means use all discovered samples.",
    )
    parser.add_argument(
        "--visualize_count",
        type=int,
        default=6,
        help="Representative cases to render per dataset.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--write_state_9d_missing",
        action="store_true",
        help="Write physics/state_9d.npy back into source samples when missing.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_sample_dirs(root: Path) -> List[Path]:
    samples: List[Path] = []
    for meta_path in sorted(root.rglob("metadata.json")):
        sample_dir = meta_path.parent
        if (sample_dir / "physics" / "anchor_targets.npz").exists():
            samples.append(sample_dir)
    return samples


def sample_id_from_dir(sample_dir: Path) -> str:
    return sample_dir.name


def choose_sample_dirs(sample_dirs: Sequence[Path], max_samples: int, seed: int) -> List[Path]:
    sample_dirs = list(sample_dirs)
    if max_samples <= 0 or len(sample_dirs) <= max_samples:
        return sample_dirs
    rng = random.Random(seed)
    rng.shuffle(sample_dirs)
    return sorted(sample_dirs[:max_samples])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_anchor_targets(sample_dir: Path) -> dict:
    payload = np.load(sample_dir / "physics" / "anchor_targets.npz")
    return {key: payload[key] for key in payload.files}


def visibility_ratio_from_seg(seg: np.ndarray, seg_ids: np.ndarray) -> np.ndarray:
    num_frames, height, width = seg.shape
    num_objects = int(seg_ids.shape[0])
    ratio = np.zeros((num_frames, num_objects), dtype=np.float32)
    norm = float(height * width)
    for obj_idx, seg_id in enumerate(seg_ids.astype(np.int32).tolist()):
        ratio[:, obj_idx] = (seg == int(seg_id)).reshape(num_frames, -1).sum(axis=1).astype(np.float32) / norm
    return ratio


def compute_observation_tables(seg: np.ndarray, depth_metric: np.ndarray, seg_ids: np.ndarray) -> dict:
    num_frames, height, width = seg.shape
    num_objects = int(seg_ids.shape[0])
    bbox_xyxy = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    centroid_uv = np.zeros((num_frames, num_objects, 2), dtype=np.float32)
    visibility_mask = np.zeros((num_frames, num_objects), dtype=np.uint8)
    visibility_ratio = np.zeros((num_frames, num_objects), dtype=np.float32)
    center_depth = np.zeros((num_frames, num_objects), dtype=np.float32)
    pixel_norm = float(height * width)

    for obj_idx, seg_id in enumerate(seg_ids.astype(np.int32).tolist()):
        target = int(seg_id)
        for frame_idx in range(num_frames):
            mask = seg[frame_idx] == target
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            visibility_mask[frame_idx, obj_idx] = 1
            visibility_ratio[frame_idx, obj_idx] = float(xs.size) / pixel_norm
            centroid_uv[frame_idx, obj_idx, 0] = float(xs.mean())
            centroid_uv[frame_idx, obj_idx, 1] = float(ys.mean())
            bbox_xyxy[frame_idx, obj_idx] = np.asarray(
                [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                dtype=np.float32,
            )
            center_depth[frame_idx, obj_idx] = float(depth_metric[frame_idx][mask].mean())

    return {
        "bbox_xyxy": bbox_xyxy,
        "centroid_uv": centroid_uv,
        "visibility_mask": visibility_mask,
        "visibility_ratio": visibility_ratio,
        "center_depth": center_depth,
    }


def load_rgb_frames_from_dir(rgb_dir: Path) -> List[Image.Image]:
    frame_paths = sorted(rgb_dir.glob("*.png"))
    return [Image.open(path).convert("RGB") for path in frame_paths]


def load_genesis_dense(sample_dir: Path, include_rgb: bool) -> dict:
    meta = load_json(sample_dir / "metadata.json")
    physics_dir = sample_dir / "physics"
    anchor = load_anchor_targets(sample_dir)
    seg = np.load(physics_dir / "seg.npy")
    depth_metric = np.load(physics_dir / "depth_metric.npy")
    obs = compute_observation_tables(seg, depth_metric, anchor["seg_ids"])
    fps = float(meta.get("fps", 12.0))
    state_recomputed = compute_state_9d(
        com_uv=anchor["com_uv"],
        center_depth=anchor["center_depth"],
        bbox_xyxy=anchor["bbox_xyxy"],
        visibility_pixels=obs["visibility_ratio"],
        fps=fps,
    )
    state_path = physics_dir / "state_9d.npy"
    state_saved = np.load(state_path) if state_path.exists() else None
    rigid_path = physics_dir / "rigid_kinematics.npz"
    reference_com_uv = anchor["com_uv"]
    if rigid_path.exists():
        rigid = np.load(rigid_path)
        reference_com_uv = rigid["com_uv"]
    energy_total = None
    kinetic_proxy = np.linalg.norm(state_recomputed[..., 5:8], axis=-1).astype(np.float32)
    energy_path = physics_dir / "energy.npz"
    if energy_path.exists():
        energy_npz = np.load(energy_path)
        if "mechanical_total" in energy_npz.files:
            energy_total = energy_npz["mechanical_total"].astype(np.float32)
    frames = load_rgb_frames_from_dir(sample_dir / "rgb") if include_rgb else []
    return {
        "dataset_slug": "genesis",
        "dataset_label": "Genesis",
        "sample_dir": sample_dir,
        "sample_id": sample_id_from_dir(sample_dir),
        "meta": meta,
        "anchor": anchor,
        "seg": seg,
        "depth_metric": depth_metric,
        "obs": obs,
        "fps": fps,
        "height": int(seg.shape[1]),
        "width": int(seg.shape[2]),
        "frames": frames,
        "state_recomputed": state_recomputed,
        "state_saved": state_saved,
        "reference_com_uv": reference_com_uv.astype(np.float32),
        "kinetic_proxy": kinetic_proxy,
        "energy_total": energy_total,
    }


def collect_movi_feature_map(sample_dirs: Sequence[Path]) -> Dict[str, object]:
    lookup: Dict[Tuple[str, int], str] = {}
    per_shard: Dict[str, set[int]] = {}
    for sample_dir in sample_dirs:
        meta = load_json(sample_dir / "metadata.json")
        source_paths = meta.get("source_paths", {})
        shard_path = str(source_paths.get("tfrecord_path", ""))
        record_index = int(source_paths.get("tfrecord_record_index", -1))
        if not shard_path or record_index < 0:
            continue
        lookup[(shard_path, record_index)] = str(sample_dir)
        per_shard.setdefault(shard_path, set()).add(record_index)

    feature_map: Dict[str, object] = {}
    for shard_path, target_indices in sorted(per_shard.items()):
        targets = set(int(x) for x in target_indices)
        for record_index, payload in enumerate(iter_serialized_records(Path(shard_path))):
            if record_index not in targets:
                continue
            feature_map[lookup[(shard_path, record_index)]] = parse_example(payload)
            targets.remove(record_index)
            if not targets:
                break
    return feature_map


def load_movi_dense(sample_dir: Path, feature_map: Dict[str, object], include_rgb: bool) -> dict:
    meta = load_json(sample_dir / "metadata.json")
    anchor = load_anchor_targets(sample_dir)
    features = feature_map[str(sample_dir)]
    num_frames = int(features["metadata/num_frames"].int64_list.value[0])
    height = int(features["metadata/height"].int64_list.value[0])
    width = int(features["metadata/width"].int64_list.value[0])
    num_objects = int(features["metadata/num_instances"].int64_list.value[0])

    seg = decode_image_sequence(features["segmentations"].bytes_list.value).reshape(num_frames, height, width).astype(
        np.uint8
    )
    depth_raw = decode_image_sequence(features["depth"].bytes_list.value).astype(np.uint16)
    depth_metric = uint16_to_metric(
        depth_raw.reshape(num_frames, height, width),
        np.asarray(features["metadata/depth_range"].float_list.value, dtype=np.float32),
    )
    obs = compute_observation_tables(seg, depth_metric, anchor["seg_ids"])
    fps = float(meta.get("fps", 12.0))
    state_saved = np.load(sample_dir / "physics" / "state_9d.npy")
    state_recomputed = compute_state_9d(
        com_uv=anchor["com_uv"],
        center_depth=anchor["center_depth"],
        bbox_xyxy=anchor["bbox_xyxy"],
        visibility_pixels=obs["visibility_ratio"],
        fps=fps,
    )
    image_positions = decode_float_tensor(features["instances/image_positions"], (num_objects, num_frames, 2))
    reference_com_uv = np.transpose(image_positions, (1, 0, 2)).astype(np.float32)
    reference_com_uv[..., 0] *= float(width)
    reference_com_uv[..., 1] *= float(height)
    velocities = decode_float_tensor(features["instances/velocities"], (num_objects, num_frames, 3))
    velocities = np.transpose(velocities, (1, 0, 2)).astype(np.float32)
    mass = np.asarray(features["instances/mass"].float_list.value, dtype=np.float32)
    kinetic_proxy = 0.5 * np.square(np.linalg.norm(velocities, axis=-1)).astype(np.float32)
    if mass.shape[0] == kinetic_proxy.shape[1]:
        kinetic_proxy = kinetic_proxy * mass[None, :]
    frames = []
    if include_rgb:
        rgb_frames = decode_rgb_frames(features["video"].bytes_list.value)
        frames = [Image.fromarray(frame).convert("RGB") for frame in rgb_frames]
    return {
        "dataset_slug": "movi_d",
        "dataset_label": "MOVI-D",
        "sample_dir": sample_dir,
        "sample_id": sample_id_from_dir(sample_dir),
        "meta": meta,
        "anchor": anchor,
        "seg": seg,
        "depth_metric": depth_metric,
        "obs": obs,
        "fps": fps,
        "height": height,
        "width": width,
        "frames": frames,
        "state_recomputed": state_recomputed,
        "state_saved": state_saved,
        "reference_com_uv": reference_com_uv,
        "kinetic_proxy": kinetic_proxy,
        "energy_total": None,
    }


def bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bx2, by2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return (inter / union).astype(np.float32)


def summarize_metric(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "p95": float(np.percentile(flat, 95)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.shape != mask.shape:
        raise ValueError(f"shape mismatch: values={values.shape}, mask={mask.shape}")
    return values[mask.astype(bool)]


def compute_velocity_smoothness(state_9d: np.ndarray, width: int, height: int, depth_scale: float) -> float:
    velocity = state_9d[..., 5:8].astype(np.float32).copy()
    velocity[..., 0] /= max(float(width), 1.0)
    velocity[..., 1] /= max(float(height), 1.0)
    velocity[..., 2] /= max(float(depth_scale), 1e-6)
    if velocity.shape[0] <= 1:
        return 0.0
    accel = np.diff(velocity, axis=0)
    return float(np.mean(np.linalg.norm(accel, axis=-1)))


def compute_sample_metrics(sample: dict) -> dict:
    anchor = sample["anchor"]
    obs = sample["obs"]
    state_recomputed = sample["state_recomputed"]
    state_saved = sample["state_saved"] if sample["state_saved"] is not None else state_recomputed
    visible = obs["visibility_mask"].astype(bool)
    depth_visible = visible & (obs["center_depth"] > 0.0)

    center_projection_error = np.linalg.norm(anchor["com_uv"] - sample["reference_com_uv"], axis=-1).astype(np.float32)
    bbox_iou = bbox_iou_matrix(anchor["bbox_xyxy"], obs["bbox_xyxy"])
    depth_abs_error = np.abs(anchor["center_depth"] - obs["center_depth"]).astype(np.float32)
    depth_rel_error = depth_abs_error / np.maximum(obs["center_depth"], 1e-6)
    state_abs_error = np.abs(state_saved - state_recomputed).astype(np.float32)
    vis_abs_error = np.abs(state_saved[..., 8] - obs["visibility_ratio"]).astype(np.float32)

    depth_scale = float(np.nanmax(obs["center_depth"][depth_visible])) if np.any(depth_visible) else 1.0
    metrics = {
        "visible_frame_ratio": float(np.mean(visible.astype(np.float32))) if visible.size else 0.0,
        "center_projection_error_px": summarize_metric(masked_values(center_projection_error, visible)),
        "bbox_iou": summarize_metric(masked_values(bbox_iou, visible)),
        "depth_abs_error": summarize_metric(masked_values(depth_abs_error, depth_visible)),
        "depth_rel_error": summarize_metric(masked_values(depth_rel_error, depth_visible)),
        "state_abs_error": summarize_metric(state_abs_error.reshape(-1)),
        "vis_abs_error": summarize_metric(masked_values(vis_abs_error, visible)),
        "velocity_smoothness": compute_velocity_smoothness(
            state_recomputed,
            width=sample["width"],
            height=sample["height"],
            depth_scale=depth_scale,
        ),
    }

    anomaly_reasons: List[str] = []
    bbox_mean = metrics["bbox_iou"]["mean"]
    depth_rel_mean = metrics["depth_rel_error"]["mean"]
    state_abs_mean = metrics["state_abs_error"]["mean"]
    vis_abs_mean = metrics["vis_abs_error"]["mean"]
    center_mean = metrics["center_projection_error_px"]["mean"]
    if bbox_mean is not None and bbox_mean < 0.90:
        anomaly_reasons.append("low_bbox_iou")
    if depth_rel_mean is not None and depth_rel_mean > 0.05:
        anomaly_reasons.append("high_depth_rel_error")
    if state_abs_mean is not None and state_abs_mean > 1e-4:
        anomaly_reasons.append("state_recompute_mismatch")
    if vis_abs_mean is not None and vis_abs_mean > 0.02:
        anomaly_reasons.append("vis_ratio_mismatch")
    if center_mean is not None and center_mean > 1e-3:
        anomaly_reasons.append("projection_mismatch")
    metrics["anomaly"] = bool(anomaly_reasons)
    metrics["anomaly_reasons"] = anomaly_reasons
    return metrics


def aggregate_dataset_metrics(rows: Sequence[dict]) -> dict:
    def pull(metric_name: str, field: str) -> List[float]:
        values: List[float] = []
        for row in rows:
            value = row["metrics"].get(metric_name, {}).get(field)
            if value is not None:
                values.append(float(value))
        return values

    def stat(values: Sequence[float]) -> dict:
        array = np.asarray(list(values), dtype=np.float32)
        return summarize_metric(array) if array.size else summarize_metric(np.asarray([], dtype=np.float32))

    summary = {
        "num_samples": int(len(rows)),
        "anomaly_ratio": float(sum(1 for row in rows if row["metrics"]["anomaly"]) / max(len(rows), 1)),
        "visible_frame_ratio": stat([row["metrics"]["visible_frame_ratio"] for row in rows]),
        "center_projection_error_px": stat(pull("center_projection_error_px", "mean")),
        "bbox_iou": stat(pull("bbox_iou", "mean")),
        "depth_abs_error": stat(pull("depth_abs_error", "mean")),
        "depth_rel_error": stat(pull("depth_rel_error", "mean")),
        "state_abs_error": stat(pull("state_abs_error", "mean")),
        "vis_abs_error": stat(pull("vis_abs_error", "mean")),
        "velocity_smoothness": stat([row["metrics"]["velocity_smoothness"] for row in rows]),
    }
    anomaly_hist: Dict[str, int] = {}
    for row in rows:
        for reason in row["metrics"]["anomaly_reasons"]:
            anomaly_hist[reason] = int(anomaly_hist.get(reason, 0)) + 1
    summary["anomaly_reason_histogram"] = dict(sorted(anomaly_hist.items()))
    return summary


def compute_risk_score(metrics: dict) -> float:
    bbox_penalty = 1.0 - float(metrics["bbox_iou"]["mean"] or 1.0)
    depth_penalty = float(metrics["depth_rel_error"]["mean"] or 0.0)
    state_penalty = min(float(metrics["state_abs_error"]["mean"] or 0.0) * 1000.0, 5.0)
    vis_penalty = min(float(metrics["vis_abs_error"]["mean"] or 0.0) * 100.0, 5.0)
    return float(bbox_penalty * 10.0 + depth_penalty * 4.0 + state_penalty + vis_penalty)


def select_visual_rows(rows: Sequence[dict], count: int) -> List[dict]:
    if not rows or count <= 0:
        return []
    ranked = sorted(rows, key=lambda row: (-row["risk_score"], row["sample_id"]))
    if len(ranked) <= count:
        return ranked
    picks: List[dict] = []
    picks.extend(ranked[: max(1, count // 2)])
    picks.append(ranked[len(ranked) // 2])
    picks.append(ranked[-1])
    dedup: List[dict] = []
    seen = set()
    for row in picks + ranked:
        if row["sample_id"] in seen:
            continue
        seen.add(row["sample_id"])
        dedup.append(row)
        if len(dedup) >= count:
            break
    return dedup


def draw_overlay_frames(sample: dict) -> List[Image.Image]:
    frames: List[Image.Image] = []
    anchor = sample["anchor"]
    obs = sample["obs"]
    rgb_frames = sample["frames"]
    seg = sample["seg"]
    num_objects = anchor["seg_ids"].shape[0]

    for frame_idx, frame in enumerate(rgb_frames):
        image = frame.copy().convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for obj_idx in range(num_objects):
            mask = seg[frame_idx] == int(anchor["seg_ids"][obj_idx])
            if not np.any(mask):
                continue
            color = STATE_COLORS[obj_idx % len(STATE_COLORS)]
            fill_rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
            ys, xs = np.nonzero(mask)
            min_x = int(xs.min())
            min_y = int(ys.min())
            max_x = int(xs.max()) + 1
            max_y = int(ys.max()) + 1
            draw.rectangle((min_x, min_y, max_x, max_y), outline=fill_rgb + (180,), width=2)

            x1, y1, x2, y2 = [float(x) for x in anchor["bbox_xyxy"][frame_idx, obj_idx]]
            u, v = [float(x) for x in anchor["com_uv"][frame_idx, obj_idx]]
            draw.rectangle((x1, y1, x2, y2), outline=fill_rgb + (255,), width=4)
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=fill_rgb + (255,))
            label_y = max(4, int(y1) - 22)
            draw.rounded_rectangle((int(x1), label_y, int(x1) + 84, label_y + 18), radius=4, fill=fill_rgb + (220,))
            draw.text((int(x1) + 6, label_y + 3), f"obj {obj_idx}", fill=(255, 255, 255, 255))
        draw.rounded_rectangle((8, 8, 136, 32), radius=6, fill=(0, 0, 0, 180))
        draw.text((14, 13), f"frame {frame_idx:02d}", fill=(255, 255, 255, 255))
        frames.append(Image.alpha_composite(image, overlay).convert("RGB"))
    return frames


def save_gif(frames: Sequence[Image.Image], dst: Path, duration_ms: int = 120, max_side: int = 720) -> None:
    if not frames:
        return
    resized: List[Image.Image] = []
    for frame in frames:
        scale = min(max_side / float(frame.width), max_side / float(frame.height), 1.0)
        size = (max(1, int(round(frame.width * scale))), max(1, int(round(frame.height * scale))))
        resized.append(frame.resize(size, Image.Resampling.BILINEAR))
    resized[0].save(dst, save_all=True, append_images=resized[1:], duration=duration_ms, loop=0)


def save_strip(frames: Sequence[Image.Image], dst: Path, frame_count: int = 6, thumb_height: int = 180) -> None:
    if not frames:
        return
    if len(frames) <= frame_count:
        indices = list(range(len(frames)))
    else:
        indices = np.linspace(0, len(frames) - 1, frame_count).round().astype(int).tolist()
    thumbs: List[Image.Image] = []
    for index in indices:
        frame = frames[index]
        scale = thumb_height / float(frame.height)
        thumbs.append(
            frame.resize((max(1, int(round(frame.width * scale))), thumb_height), Image.Resampling.BILINEAR)
        )
    total_width = sum(frame.width for frame in thumbs) + 8 * max(0, len(thumbs) - 1)
    canvas = Image.new("RGB", (total_width, thumb_height), color=(18, 18, 20))
    cursor = 0
    for thumb in thumbs:
        canvas.paste(thumb, (cursor, 0))
        cursor += thumb.width + 8
    canvas.save(dst)


def save_curve_plot(sample: dict, out_path: Path) -> None:
    state = sample["state_saved"] if sample["state_saved"] is not None else sample["state_recomputed"]
    anchor = sample["anchor"]
    obs = sample["obs"]
    num_frames, num_objects = state.shape[:2]
    t = np.arange(num_frames, dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    for obj_idx in range(num_objects):
        color = STATE_COLORS[obj_idx % len(STATE_COLORS)]
        speed = np.linalg.norm(state[:, obj_idx, 5:8], axis=-1)
        depth = state[:, obj_idx, 2]
        bbox_area = np.maximum(state[:, obj_idx, 3], 0.0) * np.maximum(state[:, obj_idx, 4], 0.0)
        vis = obs["visibility_ratio"][:, obj_idx]
        axes[0, 0].plot(t, speed, color=color, linewidth=2, alpha=0.9, label=f"obj {obj_idx}")
        axes[0, 1].plot(t, depth, color=color, linewidth=2, alpha=0.9)
        axes[1, 0].plot(t, bbox_area, color=color, linewidth=2, alpha=0.9)
        axes[1, 1].plot(t, vis, color=color, linewidth=2, alpha=0.9)

    if sample["energy_total"] is not None:
        ax_energy = axes[0, 0].twinx()
        ax_energy.plot(t, sample["energy_total"], color="#111111", linewidth=2, linestyle="--", label="total energy")
        ax_energy.set_ylabel("total energy")
    elif sample["kinetic_proxy"] is not None:
        proxy = np.sum(sample["kinetic_proxy"], axis=1) if sample["kinetic_proxy"].ndim == 2 else sample["kinetic_proxy"]
        ax_energy = axes[0, 0].twinx()
        ax_energy.plot(t, proxy, color="#111111", linewidth=2, linestyle="--", label="kinetic proxy")
        ax_energy.set_ylabel("kinetic proxy")

    axes[0, 0].set_title("Velocity magnitude")
    axes[0, 1].set_title("Center depth")
    axes[1, 0].set_title("BBox area")
    axes[1, 1].set_title("Visibility ratio")
    for ax in axes.reshape(-1):
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(loc="upper left", ncol=min(3, num_objects))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_metric_compare_plot(sample: dict, out_path: Path) -> None:
    anchor = sample["anchor"]
    obs = sample["obs"]
    num_frames, num_objects = anchor["com_uv"].shape[:2]
    t = np.arange(num_frames, dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    for obj_idx in range(num_objects):
        color = STATE_COLORS[obj_idx % len(STATE_COLORS)]
        center_error = np.linalg.norm(anchor["com_uv"][:, obj_idx] - sample["reference_com_uv"][:, obj_idx], axis=-1)
        bbox_iou = bbox_iou_matrix(anchor["bbox_xyxy"][:, obj_idx], obs["bbox_xyxy"][:, obj_idx])
        depth_error = np.abs(anchor["center_depth"][:, obj_idx] - obs["center_depth"][:, obj_idx])
        velocity_error = np.linalg.norm(
            (sample["state_saved"] if sample["state_saved"] is not None else sample["state_recomputed"])[:, obj_idx, 5:8]
            - sample["state_recomputed"][:, obj_idx, 5:8],
            axis=-1,
        )
        axes[0, 0].plot(t, center_error, color=color, linewidth=2, alpha=0.9, label=f"obj {obj_idx}")
        axes[0, 1].plot(t, bbox_iou, color=color, linewidth=2, alpha=0.9)
        axes[1, 0].plot(t, depth_error, color=color, linewidth=2, alpha=0.9)
        axes[1, 1].plot(t, velocity_error, color=color, linewidth=2, alpha=0.9)

    axes[0, 0].set_title("Projection reconstruction error (px)")
    axes[0, 1].set_title("BBox IoU vs seg")
    axes[1, 0].set_title("Depth error vs depth map")
    axes[1, 1].set_title("Velocity error vs finite diff")
    axes[0, 1].set_ylim(-0.02, 1.02)
    for ax in axes.reshape(-1):
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper left", ncol=min(3, num_objects))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def metrics_table_html(metrics: dict) -> str:
    rows = []
    for name in [
        "center_projection_error_px",
        "bbox_iou",
        "depth_abs_error",
        "depth_rel_error",
        "state_abs_error",
        "vis_abs_error",
    ]:
        values = metrics.get(name, {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{fmt_num(values.get('mean'))}</td>"
            f"<td>{fmt_num(values.get('median'))}</td>"
            f"<td>{fmt_num(values.get('p95'))}</td>"
            "</tr>"
        )
    rows.append(
        "<tr>"
        "<td>velocity_smoothness</td>"
        f"<td>{metrics['velocity_smoothness']:.6f}</td>"
        "<td>n/a</td><td>n/a</td>"
        "</tr>"
    )
    rows.append(
        "<tr>"
        "<td>anomaly</td>"
        f"<td>{str(metrics['anomaly'])}</td>"
        f"<td colspan='2'>{html.escape(', '.join(metrics['anomaly_reasons']) or 'none')}</td>"
        "</tr>"
    )
    return (
        "<table><thead><tr><th>metric</th><th>mean</th><th>median</th><th>p95</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_case_page(sample: dict, metrics: dict, out_dir: Path, rank_label: str) -> dict:
    ensure_dir(out_dir)
    overlay_frames = draw_overlay_frames(sample)
    overlay_gif = "overlay.gif"
    strip_png = "trajectory_strip.png"
    curve_png = "curves.png"
    compare_png = "comparisons.png"
    save_gif(overlay_frames, out_dir / overlay_gif)
    save_strip(overlay_frames, out_dir / strip_png)
    save_curve_plot(sample, out_dir / curve_png)
    save_metric_compare_plot(sample, out_dir / compare_png)
    np.save(out_dir / "state_9d.npy", sample["state_recomputed"].astype(np.float32))
    summary_payload = {
        "sample_id": sample["sample_id"],
        "sample_dir": str(sample["sample_dir"]),
        "dataset": sample["dataset_label"],
        "rank_label": rank_label,
        "metrics": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(sample["dataset_label"])} | {html.escape(sample["sample_id"])}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: linear-gradient(180deg, #f6f3eb 0%, #efe9db 100%);
      color: #1f2933;
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; }}
    .hero {{
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(17,24,39,0.08);
      border-radius: 18px;
      padding: 22px 24px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.08);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(17,24,39,0.08);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 8px 22px rgba(15,23,42,0.07);
    }}
    img {{ width: 100%; border-radius: 12px; border: 1px solid rgba(17,24,39,0.08); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid rgba(17,24,39,0.08); padding: 8px 10px; text-align: left; }}
    code {{ background: rgba(17,24,39,0.06); padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div><a href="../index.html">Back to dataset summary</a></div>
      <h1>{html.escape(sample["dataset_label"])} | {html.escape(sample["sample_id"])}</h1>
      <p>{html.escape(rank_label)} sample. Source: <code>{html.escape(str(sample["sample_dir"]))}</code></p>
    </div>
    <div class="grid">
      <section class="card">
        <h2>Trajectory Overlay</h2>
        <img src="{overlay_gif}" alt="overlay gif">
      </section>
      <section class="card">
        <h2>Trajectory Strip</h2>
        <img src="{strip_png}" alt="trajectory strip">
      </section>
      <section class="card">
        <h2>Curves</h2>
        <img src="{curve_png}" alt="curves">
      </section>
      <section class="card">
        <h2>Comparisons</h2>
        <img src="{compare_png}" alt="comparison curves">
      </section>
      <section class="card" style="grid-column: 1 / -1;">
        <h2>Metrics</h2>
        {metrics_table_html(metrics)}
      </section>
    </div>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
    return {
        "sample_id": sample["sample_id"],
        "sample_dir": str(sample["sample_dir"]),
        "rank_label": rank_label,
        "page_rel": f"cases/{out_dir.name}/index.html",
        "metrics": metrics,
    }


def render_dataset_portal(dataset_label: str, dataset_slug: str, dataset_summary: dict, case_cards: Sequence[dict], out_dir: Path) -> None:
    ensure_dir(out_dir)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for card in case_cards:
        metrics = card["metrics"]
        rows.append(
            f"""
<article class="case-card">
  <div class="case-top">
    <div>
      <h3>{html.escape(card['sample_id'])}</h3>
      <p>{html.escape(card['rank_label'])}</p>
    </div>
    <a href="{html.escape(card['page_rel'])}">Open</a>
  </div>
  <p><code>{html.escape(card['sample_dir'])}</code></p>
  <ul>
    <li>bbox IoU: {fmt_num(metrics['bbox_iou']['mean'])}</li>
    <li>depth rel err: {fmt_num(metrics['depth_rel_error']['mean'])}</li>
    <li>state abs err: {fmt_num(metrics['state_abs_error']['mean'])}</li>
    <li>anomaly: {str(metrics['anomaly'])} ({html.escape(', '.join(metrics['anomaly_reasons']) or 'none')})</li>
  </ul>
</article>
"""
        )

    summary = dataset_summary["aggregate"]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(dataset_label)} State Validation</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: radial-gradient(circle at top left, #e0f2fe 0%, #f8fafc 45%, #f4f1ea 100%);
      color: #14213d;
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; }}
    .hero, .card, .case-card {{
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.07);
    }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .card {{ padding: 18px; }}
    .cases {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .case-card {{ padding: 18px; }}
    .case-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    a {{
      color: #0f766e;
      font-weight: 700;
      text-decoration: none;
    }}
    code {{
      background: rgba(17,24,39,0.06);
      padding: 2px 6px;
      border-radius: 6px;
      word-break: break-all;
    }}
    ul {{ padding-left: 18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div><a href="../index.html">Back to all datasets</a></div>
      <h1>{html.escape(dataset_label)} State Validation</h1>
      <p>Samples: {dataset_summary['num_samples']} | anomaly ratio: {summary['anomaly_ratio']:.6f}</p>
      <p>Summary file: <code>{html.escape(str(summary_path))}</code></p>
    </section>
    <section class="summary-grid">
      <div class="card"><strong>Center projection err</strong><br>{fmt_num(summary['center_projection_error_px']['mean'])}</div>
      <div class="card"><strong>BBox IoU</strong><br>{fmt_num(summary['bbox_iou']['mean'])}</div>
      <div class="card"><strong>Depth abs err</strong><br>{fmt_num(summary['depth_abs_error']['mean'])}</div>
      <div class="card"><strong>State abs err</strong><br>{fmt_num(summary['state_abs_error']['mean'])}</div>
      <div class="card"><strong>Velocity smoothness</strong><br>{fmt_num(summary['velocity_smoothness']['mean'])}</div>
    </section>
    <section class="cases">
      {''.join(rows)}
    </section>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def render_root_index(dataset_entries: Sequence[dict], out_root: Path) -> None:
    cards = []
    for entry in dataset_entries:
        aggregate = entry["summary"]["aggregate"]
        cards.append(
            f"""
<article class="card">
  <h2>{html.escape(entry['dataset_label'])}</h2>
  <p>samples: {entry['summary']['num_samples']}</p>
  <p>bbox IoU: {fmt_num(aggregate['bbox_iou']['mean'])}</p>
  <p>depth abs err: {fmt_num(aggregate['depth_abs_error']['mean'])}</p>
  <p>state abs err: {fmt_num(aggregate['state_abs_error']['mean'])}</p>
  <p>anomaly ratio: {aggregate['anomaly_ratio']:.6f}</p>
  <a href="{html.escape(entry['dataset_slug'])}/index.html">Open dataset report</a>
</article>
"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>State Validation Portal</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 45%, #f5efe0 100%);
      color: #14213d;
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; }}
    .hero {{
      background: rgba(255,255,255,0.88);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 18px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.07);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 10px 24px rgba(15,23,42,0.06);
    }}
    a {{
      color: #0f766e;
      font-weight: 700;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>State Validation Portal</h1>
      <p>Dense validation for Genesis and MOVI-D 9D state supervision.</p>
      <p>Output root: <code>{html.escape(str(out_root))}</code></p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""
    (out_root / "index.html").write_text(html_text, encoding="utf-8")


def write_sample_csv(rows: Sequence[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fieldnames = [
        "sample_id",
        "sample_dir",
        "risk_score",
        "visible_frame_ratio",
        "center_projection_error_px_mean",
        "bbox_iou_mean",
        "depth_abs_error_mean",
        "depth_rel_error_mean",
        "state_abs_error_mean",
        "vis_abs_error_mean",
        "velocity_smoothness",
        "anomaly",
        "anomaly_reasons",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "sample_dir": row["sample_dir"],
                    "risk_score": f"{row['risk_score']:.6f}",
                    "visible_frame_ratio": f"{metrics['visible_frame_ratio']:.6f}",
                    "center_projection_error_px_mean": "" if metrics["center_projection_error_px"]["mean"] is None else f"{metrics['center_projection_error_px']['mean']:.6f}",
                    "bbox_iou_mean": "" if metrics["bbox_iou"]["mean"] is None else f"{metrics['bbox_iou']['mean']:.6f}",
                    "depth_abs_error_mean": "" if metrics["depth_abs_error"]["mean"] is None else f"{metrics['depth_abs_error']['mean']:.6f}",
                    "depth_rel_error_mean": "" if metrics["depth_rel_error"]["mean"] is None else f"{metrics['depth_rel_error']['mean']:.6f}",
                    "state_abs_error_mean": "" if metrics["state_abs_error"]["mean"] is None else f"{metrics['state_abs_error']['mean']:.6f}",
                    "vis_abs_error_mean": "" if metrics["vis_abs_error"]["mean"] is None else f"{metrics['vis_abs_error']['mean']:.6f}",
                    "velocity_smoothness": f"{metrics['velocity_smoothness']:.6f}",
                    "anomaly": str(metrics["anomaly"]),
                    "anomaly_reasons": ",".join(metrics["anomaly_reasons"]),
                }
            )


def build_conclusion(dataset_label: str, aggregate: dict) -> str:
    bbox = aggregate["bbox_iou"]["mean"]
    depth_abs = aggregate["depth_abs_error"]["mean"]
    depth_rel = aggregate["depth_rel_error"]["mean"]
    state_err = aggregate["state_abs_error"]["mean"]
    vis_err = aggregate["vis_abs_error"]["mean"]
    anomaly_ratio = aggregate["anomaly_ratio"]
    if bbox is None or depth_abs is None or depth_rel is None or state_err is None or vis_err is None:
        return f"{dataset_label}: insufficient data."
    if bbox >= 0.95 and depth_rel <= 0.02 and state_err <= 1e-5 and vis_err <= 0.01 and anomaly_ratio <= 0.05:
        return (
            f"{dataset_label}: 9D state supervision is reliable. Projection reconstruction is numerically exact, "
            f"bbox-depth alignment with dense observations is tight, and anomaly rate stays low."
        )
    return (
        f"{dataset_label}: supervision is usable but needs inspection for the flagged outliers. "
        f"Check low-IoU or high-depth-error samples in the portal."
    )


def maybe_write_state_9d(sample_dir: Path, state_9d: np.ndarray) -> None:
    physics_dir = sample_dir / "physics"
    state_path = physics_dir / "state_9d.npy"
    if not state_path.exists():
        np.save(state_path, state_9d.astype(np.float32))


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    rng_seed = int(args.seed)

    dataset_specs = [
        ("genesis", "Genesis", args.genesis_root),
        ("movi_d", "MOVI-D", args.movi_root),
    ]
    dataset_entries: List[dict] = []

    selected_by_dataset: Dict[str, List[Path]] = {}
    for dataset_slug, _dataset_label, root in dataset_specs:
        selected_by_dataset[dataset_slug] = choose_sample_dirs(
            find_sample_dirs(root),
            args.max_samples_per_dataset,
            rng_seed + (13 if dataset_slug == "genesis" else 29),
        )

    movi_feature_map = collect_movi_feature_map(selected_by_dataset["movi_d"])

    for dataset_slug, dataset_label, root in dataset_specs:
        selected = selected_by_dataset[dataset_slug]
        rows: List[dict] = []
        for sample_dir in selected:
            if dataset_slug == "genesis":
                sample = load_genesis_dense(sample_dir, include_rgb=False)
            else:
                sample = load_movi_dense(sample_dir, movi_feature_map, include_rgb=False)
            if args.write_state_9d_missing:
                maybe_write_state_9d(sample_dir, sample["state_recomputed"])
            metrics = compute_sample_metrics(sample)
            row = {
                "sample_id": sample["sample_id"],
                "sample_dir": str(sample["sample_dir"]),
                "metrics": metrics,
                "risk_score": compute_risk_score(metrics),
            }
            rows.append(row)

        aggregate = aggregate_dataset_metrics(rows)
        conclusion = build_conclusion(dataset_label, aggregate)
        dataset_summary = {
            "dataset_slug": dataset_slug,
            "dataset_label": dataset_label,
            "source_root": str(root),
            "num_samples": int(len(rows)),
            "aggregate": aggregate,
            "conclusion": conclusion,
        }
        dataset_out_dir = args.output_root / dataset_slug
        ensure_dir(dataset_out_dir)
        write_sample_csv(rows, dataset_out_dir / "sample_metrics.csv")

        visual_rows = select_visual_rows(rows, args.visualize_count)
        case_cards: List[dict] = []
        for rank_idx, row in enumerate(visual_rows):
            sample_dir = Path(row["sample_dir"])
            if dataset_slug == "genesis":
                sample = load_genesis_dense(sample_dir, include_rgb=True)
            else:
                sample = load_movi_dense(sample_dir, movi_feature_map, include_rgb=True)
            rank_label = "worst-ranked" if rank_idx == 0 else ("best-ranked" if rank_idx == len(visual_rows) - 1 else "representative")
            case_dir = dataset_out_dir / "cases" / f"{rank_idx:02d}_{sample['sample_id']}"
            case_cards.append(render_case_page(sample, row["metrics"], case_dir, rank_label))

        render_dataset_portal(dataset_label, dataset_slug, dataset_summary, case_cards, dataset_out_dir)
        dataset_entries.append(
            {
                "dataset_slug": dataset_slug,
                "dataset_label": dataset_label,
                "summary": dataset_summary,
            }
        )

    render_root_index(dataset_entries, args.output_root)
    summary_payload = {
        "datasets": [entry["summary"] for entry in dataset_entries],
        "portal_url": "http://127.0.0.1:8150/state_validation/index.html",
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
