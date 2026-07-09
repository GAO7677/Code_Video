"""
Mask-based query-point repair for Kubric / PhyCo raw samples.

This script is a standalone validation-first tool for the query-point failure
mode we observed in stage1 object conditioning:

1. Sample an oversubscribed candidate pool from a GT instance mask.
2. Run CoTracker from the chosen anchor frame.
3. Measure whether each tracked point stays inside the same GT instance mask.
4. Keep the best points and drop points that drift to background / other objects.
5. Export repaired query points plus visualization artifacts.

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0710querypoints/querypoint_mask_tracking_repair.py \
  --sample-dir /data/gaoya/dataset/nnsriram97-phyco_kubric/ball_wall_collision/2025-08-13/5cf30a \
  --object-ids 3 \
  --num-queries 8 \
  --oversample-factor 4 \
  --device cuda:0 \
  --output-dir /data/gaoya/agent-data/outputs/querypoint_mask_tracking_repair_demo

Or resolve raw assets directly from a Kubric/PhyCo rgba path:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0710querypoints/querypoint_mask_tracking_repair.py \
  --video-path /data/gaoya/dataset/nnsriram97-phyco_kubric/pool_table_force/2025-09-26/9e1c8f/rgba.mp4 \
  --object-ids 2,3 \
  --num-queries 8 \
  --oversample-factor 4 \
  --device cuda:0 \
  --output-dir /data/gaoya/agent-data/outputs/querypoint_mask_tracking_repair_demo_pool_table
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.data.phyco_dataset import _motion_scores
from code_vjepa_vggt.utils.object_priors import _allocate_queries_per_component
from code_vjepa_vggt.utils.object_priors import _extract_mask_components
from code_vjepa_vggt.utils.object_priors import sample_points_from_box
from code_vjepa_vggt.utils.object_priors import sample_points_from_mask


DEFAULT_COTRACKER_CKPT = "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
DEFAULT_OUTPUT_DIR = "/data/gaoya/agent-data/outputs/querypoint_mask_tracking_repair"
POINT_COLORS = [
    (214, 40, 40),
    (247, 127, 0),
    (252, 191, 73),
    (42, 157, 143),
    (39, 125, 161),
    (106, 76, 147),
    (180, 90, 120),
    (70, 70, 70),
]


@dataclass(slots=True)
class PointScore:
    point_id: int
    query_x: float
    query_y: float
    visible_frames: int
    object_visible_frames: int
    same_mask_frames: int
    visible_ratio: float
    in_mask_ratio: float
    retained_given_visible: float
    other_object_frames: int
    background_frames: int
    mean_mask_margin_px: float
    score: float
    selected: bool


@dataclass(slots=True)
class ObjectRepairResult:
    object_id: int
    object_type: str
    anchor_frame: int
    num_candidates: int
    num_selected: int
    selected_point_ids: list[int]
    selected_query_points_xy: list[list[float]]
    point_scores: list[PointScore]
    prompt_preview_png: str
    overlay_video_mp4: str
    selected_overlay_video_mp4: str
    summary_json: str


def _read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return np.stack(frames, axis=0)


def _load_sample(sample_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rgba_path = sample_dir / "rgba.mp4"
    segmentation_path = sample_dir / "segmentation.mp4"
    metadata_path = sample_dir / "metadata.json"
    if not rgba_path.is_file():
        raise FileNotFoundError(f"missing rgba.mp4 under {sample_dir}")
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"missing segmentation.mp4 under {sample_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata.json under {sample_dir}")
    rgba_rgb = _read_video_rgb(rgba_path)
    seg_rgb = _read_video_rgb(segmentation_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if rgba_rgb.shape[:3] != seg_rgb.shape[:3]:
        raise RuntimeError(
            f"rgba/segmentation shape mismatch: rgba={list(rgba_rgb.shape)} seg={list(seg_rgb.shape)}"
        )
    return rgba_rgb, seg_rgb, metadata


def _resolve_sample_dir(*, sample_dir: Path | None, video_path: Path | None) -> Path:
    if sample_dir is not None:
        resolved = sample_dir.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"sample-dir not found: {resolved}")
        return resolved
    if video_path is None:
        raise ValueError("one of --sample-dir or --video-path is required")
    resolved_video = video_path.expanduser().resolve()
    if not resolved_video.is_file():
        raise FileNotFoundError(f"video-path not found: {resolved_video}")
    if resolved_video.name != "rgba.mp4":
        raise ValueError(
            f"video-path must point to raw Kubric/PhyCo rgba.mp4 so we can find GT masks, got: {resolved_video}"
        )
    sample_root = resolved_video.parent
    for required_name in ("rgba.mp4", "segmentation.mp4", "metadata.json"):
        if not (sample_root / required_name).is_file():
            raise FileNotFoundError(
                f"cannot resolve raw GT mask assets from {resolved_video}; missing {required_name} under {sample_root}"
            )
    return sample_root


def _decode_instance_masks(
    segmentation_rgb_thwc: np.ndarray,
    metadata: dict[str, Any],
    *,
    color_tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    object_data = metadata.get("object_data", {})
    object_types = [str(item) for item in object_data.get("type", [])]
    color_map = metadata.get("segmentation_color_map", {})
    object_colors: list[np.ndarray] = []
    for object_idx in range(len(object_types)):
        color = color_map.get(str(object_idx + 1))
        if color is None:
            seg_ids = object_data.get("segmentation_id", [])
            seg_id = int(seg_ids[object_idx]) if object_idx < len(seg_ids) else object_idx + 1
            color = color_map.get(str(seg_id))
        if color is None:
            raise KeyError(f"missing segmentation color for object_idx={object_idx}")
        object_colors.append(np.asarray(color, dtype=np.int16))

    num_frames, height, width, _ = segmentation_rgb_thwc.shape
    num_objects = len(object_colors)
    masks = np.zeros((num_frames, num_objects, height, width), dtype=np.uint8)
    boxes = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    areas = np.zeros((num_frames, num_objects), dtype=np.float32)
    frame_i16 = segmentation_rgb_thwc.astype(np.int16)

    for obj_idx, color in enumerate(object_colors):
        diff = np.abs(frame_i16 - color[None, None, None, :])
        mask = (diff.max(axis=-1) <= int(color_tolerance)).astype(np.uint8)
        masks[:, obj_idx] = mask
        for frame_idx in range(num_frames):
            ys, xs = np.where(mask[frame_idx] > 0)
            if xs.size == 0 or ys.size == 0:
                continue
            boxes[frame_idx, obj_idx] = np.asarray(
                [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                dtype=np.float32,
            )
            areas[frame_idx, obj_idx] = float(xs.size)
    return masks, boxes, areas, object_types


def _parse_object_ids(raw_value: str | None) -> list[int] | None:
    if raw_value is None or not str(raw_value).strip():
        return None
    values: list[int] = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or None


def _auto_select_object_ids(
    *,
    object_types: list[str],
    motion_scores: np.ndarray,
    max_objects: int,
) -> list[int]:
    ranked = sorted(
        range(len(object_types)),
        key=lambda idx: (float(motion_scores[idx]), object_types[idx] not in {"dome", "wall", "cube_platform"}),
        reverse=True,
    )
    selected = [idx for idx in ranked if float(motion_scores[idx]) > 1.0e-3]
    if not selected:
        selected = ranked
    return selected[: max(int(max_objects), 1)]


def _choose_anchor_frame(mask_thw: np.ndarray) -> int:
    areas = mask_thw.reshape(mask_thw.shape[0], -1).sum(axis=1)
    if int((areas > 0).sum()) == 0:
        return 0
    return int(np.argmax(areas))


def _sample_candidate_queries(
    mask_thw: np.ndarray,
    boxes_t4: np.ndarray,
    *,
    anchor_frame: int,
    num_candidates: int,
) -> np.ndarray:
    anchor_mask = mask_thw[int(anchor_frame)]
    if int(anchor_mask.sum()) <= 0:
        box = boxes_t4[int(anchor_frame)]
        return sample_points_from_box(box.astype(np.float32), int(num_candidates)).astype(np.float32)

    components = _extract_mask_components(anchor_mask)
    if components:
        alloc = _allocate_queries_per_component(components, int(num_candidates))
        sampled: list[np.ndarray] = []
        for component, count in zip(components, alloc):
            if int(count) <= 0:
                continue
            points = sample_points_from_mask(component["mask"], int(count), avoid_edges=True)
            if points.shape[0] > 0:
                sampled.append(points.astype(np.float32))
        if sampled:
            merged = np.concatenate(sampled, axis=0)
            if merged.shape[0] >= int(num_candidates):
                return merged[: int(num_candidates)].astype(np.float32)
            pad = sample_points_from_mask(anchor_mask, int(num_candidates - merged.shape[0]), avoid_edges=True)
            if pad.shape[0] > 0:
                merged = np.concatenate([merged, pad.astype(np.float32)], axis=0)
            if merged.shape[0] >= int(num_candidates):
                return merged[: int(num_candidates)].astype(np.float32)

    sampled = sample_points_from_mask(anchor_mask, int(num_candidates), avoid_edges=True)
    if sampled.shape[0] == int(num_candidates):
        return sampled.astype(np.float32)

    box = boxes_t4[int(anchor_frame)]
    return sample_points_from_box(box.astype(np.float32), int(num_candidates)).astype(np.float32)


def _prepare_cotracker_input(rgba_rgb_thwc: np.ndarray) -> torch.Tensor:
    frames_01 = torch.from_numpy(rgba_rgb_thwc).float() / 255.0
    return frames_01.unsqueeze(0).contiguous()


def _point_inside_mask(mask_hw: np.ndarray, xy: np.ndarray) -> bool:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if y < 0 or y >= int(mask_hw.shape[0]) or x < 0 or x >= int(mask_hw.shape[1]):
        return False
    return bool(mask_hw[y, x] > 0)


def _mask_margin(distance_hw: np.ndarray, xy: np.ndarray) -> float:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if y < 0 or y >= int(distance_hw.shape[0]) or x < 0 or x >= int(distance_hw.shape[1]):
        return 0.0
    return float(distance_hw[y, x])


def _score_points(
    *,
    object_id: int,
    query_points_xy: np.ndarray,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    masks_tnhw: np.ndarray,
) -> list[PointScore]:
    target_masks = masks_tnhw[:, int(object_id)]
    object_visible_frames = int((target_masks.reshape(target_masks.shape[0], -1).sum(axis=1) > 0).sum())
    visible_areas = target_masks.reshape(target_masks.shape[0], -1).sum(axis=1)
    visible_areas = visible_areas[visible_areas > 0]
    object_scale = float(math.sqrt(float(np.median(visible_areas)))) if visible_areas.size > 0 else 1.0
    object_scale = max(object_scale, 1.0)
    distance_maps = [
        cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) if int(mask.sum()) > 0 else np.zeros_like(mask, dtype=np.float32)
        for mask in target_masks
    ]
    scores: list[PointScore] = []

    for point_idx in range(int(query_points_xy.shape[0])):
        visible_frames = 0
        same_mask_frames = 0
        other_object_frames = 0
        background_frames = 0
        margins: list[float] = []
        for frame_idx in range(int(tracks_tk2.shape[0])):
            target_visible = bool(target_masks[frame_idx].sum() > 0)
            if not target_visible:
                continue
            if float(visibility_tk[frame_idx, point_idx]) <= 0.5:
                continue
            visible_frames += 1
            xy = tracks_tk2[frame_idx, point_idx]
            if _point_inside_mask(target_masks[frame_idx], xy):
                same_mask_frames += 1
                margins.append(_mask_margin(distance_maps[frame_idx], xy))
                continue
            hit_other = False
            for other_idx in range(int(masks_tnhw.shape[1])):
                if int(other_idx) == int(object_id):
                    continue
                if _point_inside_mask(masks_tnhw[frame_idx, other_idx], xy):
                    hit_other = True
                    break
            if hit_other:
                other_object_frames += 1
            else:
                background_frames += 1

        visible_ratio = float(visible_frames) / max(float(object_visible_frames), 1.0)
        in_mask_ratio = float(same_mask_frames) / max(float(object_visible_frames), 1.0)
        retained_given_visible = float(same_mask_frames) / max(float(visible_frames), 1.0)
        mean_mask_margin = float(np.mean(margins)) if margins else 0.0
        # Keep the ranking dominated by temporal instance consistency.
        # Margin only acts as a weak tie-breaker after normalizing by object scale,
        # so large support/background masks cannot win purely because they are wide.
        norm_margin = min(mean_mask_margin / object_scale, 1.0)
        score_value = (
            4.0 * in_mask_ratio
            + 2.5 * retained_given_visible
            + 1.0 * visible_ratio
            + 0.25 * norm_margin
            - 0.75 * (float(other_object_frames) / max(float(object_visible_frames), 1.0))
            - 0.35 * (float(background_frames) / max(float(object_visible_frames), 1.0))
        )
        scores.append(
            PointScore(
                point_id=int(point_idx),
                query_x=float(query_points_xy[point_idx, 0]),
                query_y=float(query_points_xy[point_idx, 1]),
                visible_frames=int(visible_frames),
                object_visible_frames=int(object_visible_frames),
                same_mask_frames=int(same_mask_frames),
                visible_ratio=float(visible_ratio),
                in_mask_ratio=float(in_mask_ratio),
                retained_given_visible=float(retained_given_visible),
                other_object_frames=int(other_object_frames),
                background_frames=int(background_frames),
                mean_mask_margin_px=float(mean_mask_margin),
                score=float(score_value),
                selected=False,
            )
        )
    return scores


def _select_repaired_points(
    point_scores: list[PointScore],
    *,
    num_queries: int,
    min_visible_ratio: float,
    min_in_mask_ratio: float,
) -> list[int]:
    eligible = [
        item for item in point_scores
        if float(item.visible_ratio) >= float(min_visible_ratio) and float(item.in_mask_ratio) >= float(min_in_mask_ratio)
    ]
    eligible.sort(key=lambda item: (item.score, item.in_mask_ratio, item.retained_given_visible), reverse=True)
    chosen = eligible[: int(num_queries)]
    if len(chosen) < int(num_queries):
        remaining = [item for item in point_scores if item.point_id not in {x.point_id for x in chosen}]
        remaining.sort(key=lambda item: (item.score, item.in_mask_ratio, item.retained_given_visible), reverse=True)
        chosen.extend(remaining[: int(num_queries - len(chosen))])
    chosen_ids = [int(item.point_id) for item in chosen[: int(num_queries)]]
    selected_set = set(chosen_ids)
    for item in point_scores:
        item.selected = int(item.point_id) in selected_set
    return chosen_ids


def _draw_box_rgb(frame_rgb: np.ndarray, box_xyxy: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box_xyxy.tolist()]
    cv2.rectangle(frame_rgb, (x0, y0), (x1, y1), color_rgb, 2)
    cv2.putText(
        frame_rgb,
        label,
        (x0 + 2, max(12, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color_rgb,
        1,
        cv2.LINE_AA,
    )


def _draw_point_rgb(frame_rgb: np.ndarray, xy: np.ndarray, color_rgb: tuple[int, int, int], label: str, *, radius: int) -> None:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    cv2.circle(frame_rgb, (x, y), int(radius), color_rgb, -1, cv2.LINE_AA)
    cv2.circle(frame_rgb, (x, y), int(radius + 2), (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(
            frame_rgb,
            label,
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color_rgb,
            1,
            cv2.LINE_AA,
        )


def _render_prompt_preview(
    *,
    anchor_rgb: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_box: np.ndarray,
    candidate_points_xy: np.ndarray,
    selected_ids: set[int],
) -> np.ndarray:
    frame = anchor_rgb.copy()
    overlay = frame.copy()
    overlay[anchor_mask > 0] = (0.75 * overlay[anchor_mask > 0] + 0.25 * np.asarray([255, 210, 80])).astype(np.uint8)
    frame = overlay
    _draw_box_rgb(frame, anchor_box.astype(np.float32), (255, 180, 0), "anchor_box")
    for point_idx in range(int(candidate_points_xy.shape[0])):
        color = POINT_COLORS[point_idx % len(POINT_COLORS)]
        radius = 5 if point_idx in selected_ids else 3
        label = f"q{point_idx}" if point_idx in selected_ids else ""
        _draw_point_rgb(frame, candidate_points_xy[point_idx], color, label, radius=radius)
    return frame


def _render_overlay_video(
    *,
    rgba_rgb_thwc: np.ndarray,
    boxes_t4: np.ndarray,
    target_masks_thw: np.ndarray,
    candidate_points_xy: np.ndarray,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    selected_ids: set[int],
    point_scores: dict[int, PointScore],
    selected_only: bool,
) -> np.ndarray:
    rendered: list[np.ndarray] = []
    for frame_idx in range(int(rgba_rgb_thwc.shape[0])):
        frame = rgba_rgb_thwc[frame_idx].copy()
        overlay = frame.copy()
        mask = target_masks_thw[frame_idx] > 0
        overlay[mask] = (0.7 * overlay[mask] + 0.3 * np.asarray([255, 210, 80])).astype(np.uint8)
        frame = overlay
        if float(boxes_t4[frame_idx, 2] - boxes_t4[frame_idx, 0]) > 1.0 and float(boxes_t4[frame_idx, 3] - boxes_t4[frame_idx, 1]) > 1.0:
            _draw_box_rgb(frame, boxes_t4[frame_idx].astype(np.float32), (255, 180, 0), f"obj@{frame_idx}")

        for point_idx in range(int(candidate_points_xy.shape[0])):
            if selected_only and point_idx not in selected_ids:
                continue
            color = POINT_COLORS[point_idx % len(POINT_COLORS)]
            score = point_scores[int(point_idx)]
            xy = tracks_tk2[frame_idx, point_idx]
            visible = float(visibility_tk[frame_idx, point_idx]) > 0.5
            same_mask = _point_inside_mask(target_masks_thw[frame_idx], xy) if visible else False
            radius = 5 if point_idx in selected_ids else 3
            label = f"q{point_idx}"
            draw_color = color if same_mask else (220, 220, 220)
            if visible:
                _draw_point_rgb(frame, xy, draw_color, label, radius=radius)
            prompt_xy = candidate_points_xy[point_idx]
            _draw_point_rgb(frame, prompt_xy, color, "" if frame_idx > 0 else f"p{point_idx}", radius=2)
            if frame_idx > 0 and visible and float(visibility_tk[frame_idx - 1, point_idx]) > 0.5:
                prev_xy = tracks_tk2[frame_idx - 1, point_idx]
                cv2.line(
                    frame,
                    (int(round(float(prev_xy[0]))), int(round(float(prev_xy[1])))),
                    (int(round(float(xy[0]))), int(round(float(xy[1])))),
                    draw_color,
                    2,
                    cv2.LINE_AA,
                )
            if frame_idx == 0:
                cv2.putText(
                    frame,
                    f"r={score.in_mask_ratio:.2f}",
                    (int(round(float(prompt_xy[0]))) + 4, int(round(float(prompt_xy[1]))) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"frame={frame_idx:02d} {'selected' if selected_only else 'candidates'}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered.append(frame)
    return np.stack(rendered, axis=0)


def _write_mp4(path: Path, frames_rgb_thwc: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_rgb_thwc.shape[1]), int(frames_rgb_thwc.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    for frame_rgb in frames_rgb_thwc:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def _write_png(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _build_html(output_dir: Path, sample_dir: Path, object_results: list[ObjectRepairResult], summary: dict[str, Any]) -> None:
    cards: list[str] = []
    for item in object_results:
        cards.append(
            f"""
<section class=\"card\">
  <h2>object {item.object_id} ({html.escape(item.object_type)})</h2>
  <p><b>anchor_frame:</b> {item.anchor_frame} &nbsp; <b>candidates:</b> {item.num_candidates} &nbsp; <b>selected:</b> {item.num_selected}</p>
  <p><b>selected_point_ids:</b> {html.escape(str(item.selected_point_ids))}</p>
  <div class=\"grid\">
    <figure>
      <img src=\"{html.escape(item.prompt_preview_png)}\" />
      <figcaption>Prompt-frame mask, anchor box, candidate points, selected points</figcaption>
    </figure>
    <figure>
      <video controls preload=\"none\" playsinline src=\"{html.escape(item.overlay_video_mp4)}\"></video>
      <figcaption>All candidate tracks. Grey means drifted outside the same instance mask.</figcaption>
    </figure>
    <figure>
      <video controls preload=\"none\" playsinline src=\"{html.escape(item.selected_overlay_video_mp4)}\"></video>
      <figcaption>Repaired query points only.</figcaption>
    </figure>
  </div>
  <details>
    <summary>Summary JSON</summary>
    <pre>{html.escape(item.summary_json)}</pre>
  </details>
</section>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Query Point Mask Tracking Repair</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f1e7;
      --panel: #fffdf8;
      --line: #d9d0c2;
      --text: #1f1f1f;
      --muted: #5f5a53;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top right, #efe6d3 0, transparent 24%), linear-gradient(180deg, var(--bg) 0%, #f2ede3 100%);
    }}
    .page {{ max-width: 1700px; margin: 0 auto; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      margin-top: 18px;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #fff; }}
    img, video {{ display: block; width: 100%; background: #000; }}
    figcaption {{ padding: 10px 12px; font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); }}
    pre {{ white-space: pre-wrap; overflow-x: auto; background: #faf7f0; border: 1px solid var(--line); border-radius: 10px; padding: 14px; }}
    @media (max-width: 1300px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class=\"page\">
    <h1>Query Point Mask Tracking Repair</h1>
    <p><b>sample_dir:</b> {html.escape(str(sample_dir))}</p>
    <p>This page compares the raw oversampled candidate queries against the repaired subset that actually stays on the same GT instance over time.</p>
    <details>
      <summary>Run Summary</summary>
      <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
    </details>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair object query points with GT masks plus CoTracker consistency checks.")
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--object-ids", type=str, default=None, help="Comma-separated object ids. Default: auto-pick dynamic objects.")
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--oversample-factor", type=int, default=4)
    parser.add_argument("--color-tolerance", type=int, default=18)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cotracker-checkpoint", type=str, default=DEFAULT_COTRACKER_CKPT)
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--min-visible-ratio", type=float, default=0.60)
    parser.add_argument("--min-in-mask-ratio", type=float, default=0.60)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = _resolve_sample_dir(sample_dir=args.sample_dir, video_path=args.video_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgba_rgb_thwc, segmentation_rgb_thwc, metadata = _load_sample(sample_dir)
    masks_tnhw, boxes_tn4, areas_tn, object_types = _decode_instance_masks(
        segmentation_rgb_thwc,
        metadata,
        color_tolerance=int(args.color_tolerance),
    )
    motion_scores = _motion_scores(boxes_tn4, areas_tn)
    requested_ids = _parse_object_ids(args.object_ids)
    object_ids = requested_ids or _auto_select_object_ids(
        object_types=object_types,
        motion_scores=motion_scores,
        max_objects=int(args.max_objects),
    )

    cotracker = CoTrackerAdapter(
        checkpoint_path=str(args.cotracker_checkpoint),
        num_queries=int(args.num_queries * args.oversample_factor),
        device=str(args.device),
        input_hw=(int(args.cotracker_input_h), int(args.cotracker_input_w)),
        window_len=int(args.cotracker_window_len),
    )
    video_bthwc_01 = _prepare_cotracker_input(rgba_rgb_thwc)
    object_results: list[ObjectRepairResult] = []

    for object_id in object_ids:
        if not (0 <= int(object_id) < int(masks_tnhw.shape[1])):
            raise ValueError(f"object_id out of range: {object_id}")
        object_dir = output_dir / f"object_{int(object_id):02d}_{object_types[int(object_id)]}"
        object_dir.mkdir(parents=True, exist_ok=True)
        target_masks_thw = masks_tnhw[:, int(object_id)]
        target_boxes_t4 = boxes_tn4[:, int(object_id)]
        anchor_frame = _choose_anchor_frame(target_masks_thw)
        num_candidates = int(args.num_queries) * int(args.oversample_factor)
        candidate_points_xy = _sample_candidate_queries(
            target_masks_thw,
            target_boxes_t4,
            anchor_frame=int(anchor_frame),
            num_candidates=int(num_candidates),
        )
        query_points_prior = torch.from_numpy(candidate_points_xy).unsqueeze(0).float()
        query_frame_ids = torch.full((1, int(num_candidates), 1), float(anchor_frame), dtype=torch.float32)
        with torch.no_grad():
            cot_out = cotracker(
                video_bthwc_01,
                query_points_prior=query_points_prior,
                query_frame_ids=query_frame_ids,
                query_image_hw=(int(rgba_rgb_thwc.shape[1]), int(rgba_rgb_thwc.shape[2])),
            )
        tracks_tk2 = cot_out.tracks[0].detach().cpu().numpy()
        visibility_tk = cot_out.visibility[0].detach().cpu().numpy()
        point_scores = _score_points(
            object_id=int(object_id),
            query_points_xy=candidate_points_xy,
            tracks_tk2=tracks_tk2,
            visibility_tk=visibility_tk,
            masks_tnhw=masks_tnhw,
        )
        selected_ids = _select_repaired_points(
            point_scores,
            num_queries=int(args.num_queries),
            min_visible_ratio=float(args.min_visible_ratio),
            min_in_mask_ratio=float(args.min_in_mask_ratio),
        )
        selected_set = set(int(item) for item in selected_ids)
        point_score_map = {int(item.point_id): item for item in point_scores}

        prompt_preview = _render_prompt_preview(
            anchor_rgb=rgba_rgb_thwc[int(anchor_frame)],
            anchor_mask=target_masks_thw[int(anchor_frame)],
            anchor_box=target_boxes_t4[int(anchor_frame)],
            candidate_points_xy=candidate_points_xy,
            selected_ids=selected_set,
        )
        prompt_preview_path = object_dir / "prompt_preview.png"
        _write_png(prompt_preview_path, prompt_preview)

        overlay_all = _render_overlay_video(
            rgba_rgb_thwc=rgba_rgb_thwc,
            boxes_t4=target_boxes_t4,
            target_masks_thw=target_masks_thw,
            candidate_points_xy=candidate_points_xy,
            tracks_tk2=tracks_tk2,
            visibility_tk=visibility_tk,
            selected_ids=selected_set,
            point_scores=point_score_map,
            selected_only=False,
        )
        overlay_all_path = object_dir / "candidate_overlay.mp4"
        _write_mp4(overlay_all_path, overlay_all, fps=int(args.fps))

        overlay_selected = _render_overlay_video(
            rgba_rgb_thwc=rgba_rgb_thwc,
            boxes_t4=target_boxes_t4,
            target_masks_thw=target_masks_thw,
            candidate_points_xy=candidate_points_xy,
            tracks_tk2=tracks_tk2,
            visibility_tk=visibility_tk,
            selected_ids=selected_set,
            point_scores=point_score_map,
            selected_only=True,
        )
        overlay_selected_path = object_dir / "selected_overlay.mp4"
        _write_mp4(overlay_selected_path, overlay_selected, fps=int(args.fps))

        object_summary = {
            "object_id": int(object_id),
            "object_type": object_types[int(object_id)],
            "anchor_frame": int(anchor_frame),
            "num_candidates": int(num_candidates),
            "num_selected": int(len(selected_ids)),
            "selected_point_ids": [int(item) for item in selected_ids],
            "selected_query_points_xy": candidate_points_xy[selected_ids].round(3).tolist(),
            "point_scores": [asdict(item) for item in point_scores],
        }
        summary_path = object_dir / "summary.json"
        summary_path.write_text(json.dumps(object_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        object_results.append(
            ObjectRepairResult(
                object_id=int(object_id),
                object_type=object_types[int(object_id)],
                anchor_frame=int(anchor_frame),
                num_candidates=int(num_candidates),
                num_selected=int(len(selected_ids)),
                selected_point_ids=[int(item) for item in selected_ids],
                selected_query_points_xy=candidate_points_xy[selected_ids].round(3).tolist(),
                point_scores=point_scores,
                prompt_preview_png=str(prompt_preview_path.relative_to(output_dir)),
                overlay_video_mp4=str(overlay_all_path.relative_to(output_dir)),
                selected_overlay_video_mp4=str(overlay_selected_path.relative_to(output_dir)),
                summary_json=json.dumps(object_summary, ensure_ascii=False, indent=2),
            )
        )

    summary = {
        "sample_dir": str(sample_dir),
        "rgba_shape": list(rgba_rgb_thwc.shape),
        "num_objects": int(masks_tnhw.shape[1]),
        "object_types": object_types,
        "motion_scores": [float(item) for item in motion_scores.tolist()],
        "object_ids": [int(item) for item in object_ids],
        "num_queries": int(args.num_queries),
        "oversample_factor": int(args.oversample_factor),
        "min_visible_ratio": float(args.min_visible_ratio),
        "min_in_mask_ratio": float(args.min_in_mask_ratio),
        "device": str(args.device),
    }
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "objects": [
                    {
                        "object_id": int(item.object_id),
                        "object_type": item.object_type,
                        "anchor_frame": int(item.anchor_frame),
                        "num_candidates": int(item.num_candidates),
                        "num_selected": int(item.num_selected),
                        "selected_point_ids": item.selected_point_ids,
                        "selected_query_points_xy": item.selected_query_points_xy,
                    }
                    for item in object_results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    _build_html(output_dir, sample_dir, object_results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
