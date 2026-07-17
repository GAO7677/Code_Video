#!/usr/bin/env python3
"""Visualize four object/event-aware loss weighting schemes on PyBullet data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import decord
import numpy as np


CASES = (
    ("F1", "F1_single_object", "sample_000001"),
    ("F2", "F2_two_object", "sample_000076"),
    ("F3", "F3_chain_reaction", "sample_000166"),
    ("F4", "F4_occlusion", "sample_000226"),
    ("F5", "F5_drop_support", "sample_000271"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reconstruction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / max(norm, 1.0e-8)


def object_radius(obj: dict) -> float:
    size = obj.get("size", {})
    if "radius" in size and "height" in size:
        return float(max(size["radius"], 0.5 * size["height"]))
    if "radius" in size:
        return float(size["radius"])
    values = [float(size.get(key, 0.0)) for key in ("hx", "hy", "hz")]
    return max(values + [0.12])


def project_object_masks(
    meta: dict,
    states: dict[str, np.ndarray],
    frame_indices: list[int],
    grid_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = states["positions"]
    velocities = states["linear_velocities"]
    names = [str(value) for value in states["object_names"]]
    objects_by_name = {str(obj["name"]): obj for obj in meta["objects"]}
    height = int(np.asarray(states["frame_height"]).reshape(-1)[0])
    width = int(np.asarray(states["frame_width"]).reshape(-1)[0])
    eye = np.asarray(states["camera_eye"], dtype=np.float32)
    target = np.asarray(states["camera_target"], dtype=np.float32)
    up_hint = np.asarray(states["camera_up"], dtype=np.float32)
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    fov_y = math.radians(float(meta.get("camera", {}).get("yfov_deg", 50.0)))
    focal_y = height / (2.0 * math.tan(0.5 * fov_y))
    focal_x = focal_y
    grid_h, grid_w = grid_hw
    masks = np.zeros((len(frame_indices) // 2, len(names), grid_h, grid_w), dtype=bool)
    visible = np.zeros((len(frame_indices) // 2, len(names)), dtype=bool)
    centers_world = np.zeros((len(frame_indices) // 2, len(names), 3), dtype=np.float32)

    for token_t in range(masks.shape[0]):
        source_index = frame_indices[min(2 * token_t + 1, len(frame_indices) - 1)]
        source_index = min(source_index, positions.shape[0] - 1)
        centers_world[token_t] = positions[source_index]
        for object_id, name in enumerate(names):
            obj = objects_by_name.get(name, {})
            relative = positions[source_index, object_id] - eye
            cam_x = float(np.dot(relative, right))
            cam_y = float(np.dot(relative, up))
            cam_z = float(np.dot(relative, forward))
            if cam_z <= 1.0e-4:
                continue
            u = width * 0.5 + focal_x * cam_x / cam_z
            v = height * 0.5 - focal_y * cam_y / cam_z
            radius = object_radius(obj)
            radius_x = max(1.0, focal_x * radius / cam_z * grid_w / width)
            radius_y = max(1.0, focal_y * radius / cam_z * grid_h / height)
            center_x = u * grid_w / width
            center_y = v * grid_h / height
            if center_x < -radius_x or center_x >= grid_w + radius_x:
                continue
            if center_y < -radius_y or center_y >= grid_h + radius_y:
                continue
            yy, xx = np.ogrid[:grid_h, :grid_w]
            masks[token_t, object_id] = (
                ((xx - center_x) / (radius_x + 1.0)) ** 2
                + ((yy - center_y) / (radius_y + 1.0)) ** 2
                <= 1.0
            )
            visible[token_t, object_id] = True

    # Event score: acceleration, close object pairs and visibility transitions.
    scores = np.zeros(visible.shape, dtype=np.float32)
    sampled_velocities = np.zeros((masks.shape[0], len(names), 3), dtype=np.float32)
    for token_t in range(masks.shape[0]):
        idx = frame_indices[min(2 * token_t + 1, len(frame_indices) - 1)]
        sampled_velocities[token_t] = velocities[min(idx, velocities.shape[0] - 1)]
    acceleration = np.linalg.norm(np.diff(sampled_velocities, axis=0, prepend=sampled_velocities[:1]), axis=-1)
    dynamic_ids = [
        index for index, name in enumerate(names)
        if str(objects_by_name.get(name, {}).get("role", "dynamic")) == "dynamic"
    ]
    if dynamic_ids:
        values = acceleration[:, dynamic_ids].reshape(-1)
        low, high = np.quantile(values, [0.70, 0.95])
        scores[:, dynamic_ids] += np.clip((acceleration[:, dynamic_ids] - low) / max(high - low, 1.0e-6), 0.0, 1.0)
    radii = np.asarray([object_radius(objects_by_name.get(name, {})) for name in names])
    for token_t in range(masks.shape[0]):
        for left in range(len(names)):
            for right_id in range(left + 1, len(names)):
                distance = float(np.linalg.norm(centers_world[token_t, left] - centers_world[token_t, right_id]))
                threshold = 1.5 * float(radii[left] + radii[right_id])
                if distance < threshold:
                    scores[token_t, left] = max(scores[token_t, left], 1.0)
                    scores[token_t, right_id] = max(scores[token_t, right_id], 1.0)
    changes = np.zeros_like(visible)
    changes[1:] = visible[1:] != visible[:-1]
    scores[changes] = 1.0
    event_objects = scores >= 0.65
    expanded = event_objects.copy()
    expanded[1:] |= event_objects[:-1]
    expanded[:-1] |= event_objects[1:]
    event_mask = np.any(masks & expanded[:, :, None, None], axis=1)
    object_mask = np.any(masks, axis=1)
    return object_mask, event_mask, scores


def normalized_weight(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.ones_like(raw, dtype=np.float32)
    clipped = np.clip(raw, 0.5, 3.0)
    output[valid] = clipped[valid] / max(float(clipped[valid].mean()), 1.0e-6)
    return output


def add_header(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(frame, 46, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(canvas, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def heat_overlay(frame: np.ndarray, values: np.ndarray, scale: float) -> np.ndarray:
    encoded = np.clip(values / max(scale, 1.0e-8), 0.0, 1.0)
    heat = cv2.applyColorMap((encoded * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(cv2.resize(heat, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame, 0.48, heat, 0.52, 0.0)


def write_h264(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        intermediate = Path(temp_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        ffmpeg = shutil.which("ffmpeg") or "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(intermediate), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
            check=True,
        )


def process_case(family: str, family_dir: str, sample: str, args: argparse.Namespace) -> dict:
    case_dir = args.dataset_root / family_dir / sample
    reconstruction_dir = args.reconstruction_root / sample
    output_dir = args.output_dir / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((case_dir / "meta.json").read_text())
    metrics = json.loads((reconstruction_dir / "metrics.json").read_text())
    archive = np.load(reconstruction_dir / "patch_maps.npz")
    surprise = archive["surprise"].astype(np.float32)
    valid = np.isfinite(surprise)
    with np.load(case_dir / "states.npz", allow_pickle=True) as states_file:
        states = {key: states_file[key] for key in states_file.files}
    object_mask, event_mask, event_scores = project_object_masks(
        meta, states, metrics["frame_indices"], surprise.shape[1:]
    )
    object_valid = object_mask & valid
    hard_mask = np.zeros_like(valid)
    if object_valid.any():
        hard_threshold = float(np.quantile(surprise[object_valid], 0.80))
        hard_mask = object_valid & (surprise >= hard_threshold)
    else:
        hard_threshold = float("nan")

    raw_weights = {
        "uniform": np.ones_like(surprise, dtype=np.float32),
        "object-only": 1.0 + 0.50 * object_mask,
        "object+event": 1.0 + 0.50 * object_mask + 1.00 * event_mask,
        "object+event+hard": 1.0 + 0.50 * object_mask + 1.00 * event_mask + 0.50 * hard_mask,
    }
    weights = {name: normalized_weight(value, valid) for name, value in raw_weights.items()}
    weighted_losses = {name: weight * np.nan_to_num(surprise, nan=0.0) for name, weight in weights.items()}
    shared_scale = float(np.quantile(np.concatenate([value[valid] for value in weighted_losses.values()]), 0.99))

    input_video_path = case_dir / "video.mp4"
    if not input_video_path.exists():
        input_video_path = case_dir / "source_video.mp4"
    reader = decord.VideoReader(str(input_video_path), ctx=decord.cpu(0))
    frame_indices = np.asarray(metrics["frame_indices"], dtype=np.int64)
    frames = reader.get_batch(frame_indices).asnumpy()
    frames = np.stack([cv2.resize(frame, (384, 384), interpolation=cv2.INTER_AREA) for frame in frames])
    rendered = []
    selected_rows = []
    selected = {8, 16, 24, 32, 40, 48}
    names = tuple(weights)
    for frame_id, frame in enumerate(frames):
        token_t = min(frame_id // 2, surprise.shape[0] - 1)
        panels = [add_header(frame, f"{family} {sample} | frame {frame_id:02d}")]
        for name in names:
            view = heat_overlay(frame, weighted_losses[name][token_t], shared_scale)
            mean_weight = float(weights[name][valid].mean())
            mean_loss = float(weighted_losses[name][valid].mean())
            panels.append(add_header(view, f"{name} | mean w={mean_weight:.2f} weighted loss={mean_loss:.4f}"))
        row = np.concatenate(panels, axis=1)
        rendered.append(row)
        if frame_id in selected:
            selected_rows.append(row)
    output_video_path = output_dir / "loss_weight_schemes_overlay_h264.mp4"
    write_h264(output_video_path, rendered)
    contact_path = output_dir / "loss_weight_schemes_contact.jpg"
    cv2.imwrite(str(contact_path), cv2.cvtColor(np.concatenate(selected_rows, axis=0), cv2.COLOR_RGB2BGR))
    np.savez_compressed(
        output_dir / "weight_maps.npz",
        object_mask=object_mask.astype(np.uint8),
        event_mask=event_mask.astype(np.uint8),
        hard_mask=hard_mask.astype(np.uint8),
        surprise=surprise.astype(np.float16),
        **{name.replace("+", "_").replace("-", "_"): value.astype(np.float16) for name, value in weights.items()},
    )
    report = {
        "sample": sample,
        "family": family,
        "template": meta.get("template_key"),
        "source_video": str(input_video_path),
        "object_mask_source": "oracle projection from PyBullet states for weighting-design visualization",
        "event_definition": "acceleration OR object proximity OR visibility transition, temporal radius 1 tubelet",
        "hard_definition": "top 20% V-JEPA error within object support",
        "hard_threshold": hard_threshold,
        "areas": {
            "object": float(object_valid.sum() / max(valid.sum(), 1)),
            "event": float((event_mask & valid).sum() / max(valid.sum(), 1)),
            "hard": float(hard_mask.sum() / max(valid.sum(), 1)),
        },
        "weighted_loss_means": {
            name: float(value[valid].mean()) for name, value in weighted_losses.items()
        },
        "weight_ranges": {
            name: [float(value[valid].min()), float(value[valid].max())] for name, value in weights.items()
        },
        "video": str(output_video_path),
        "contact": str(contact_path),
        "event_score_max": float(event_scores.max()),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_gallery(reports: list[dict], output_dir: Path) -> None:
    cards = []
    for report in reports:
        rel = Path(report["video"]).relative_to(output_dir)
        cards.append(
            f"<article><h2>{report['family']} / {report['sample']}</h2>"
            f"<p>{report['template']} | areas: object={report['areas']['object']:.1%}, "
            f"event={report['areas']['event']:.1%}, hard={report['areas']['hard']:.1%}</p>"
            f"<video controls loop muted preload='metadata' src='{rel.as_posix()}'></video>"
            f"<p><code>{rel.as_posix()}</code></p></article>"
        )
    html = """<!doctype html><html><head><meta charset='utf-8'><title>Loss weighting heatmaps</title>
<style>body{font-family:Georgia,serif;background:#eee7d8;color:#17231d;margin:0}main{max-width:1500px;margin:auto;padding:28px}article{background:#fffaf0;border:1px solid #b9aa8c;padding:18px;margin:22px 0}video{width:100%;background:#111}code{font-size:13px}h1,h2{font-family:Verdana,sans-serif}</style></head><body><main>
<h1>Training-data loss weighting overlays</h1><p>Columns: source, uniform, object-only, object+event, object+event+V-JEPA-hard. Shared color scale within each case.</p>""" + "".join(cards) + "</main></body></html>"
    (output_dir / "index.html").write_text(html)
    (output_dir / "gallery_manifest.json").write_text(json.dumps({"cases": reports}, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = [process_case(*case, args) for case in CASES]
    build_gallery(reports, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "num_cases": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
