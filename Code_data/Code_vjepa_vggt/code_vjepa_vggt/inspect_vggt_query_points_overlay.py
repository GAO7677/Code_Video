from __future__ import annotations

import argparse
import base64
import html
import http.server
import io
import json
import math
import shutil
import socketserver
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker, build_motion_prompt_box
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior
from code_vjepa_vggt.utils.object_priors import _extract_mask_components
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes


GT_PALETTE = ["#d62828", "#f77f00", "#fcbf49", "#2a9d8f", "#277da1", "#6a4c93"]
QUERY_PALETTE = ["#00b4d8", "#0077b6", "#8338ec", "#3a86ff", "#ff006e", "#fb5607", "#2ec4b6", "#8ac926"]
COTRACKER_PALETTE = ["#2b8a3e", "#40916c", "#52b788", "#74c69d", "#1b4332", "#95d5b2", "#007f5f", "#55a630"]
SAM_PROMPT_COLOR = "#ff8c00"
SAM_TRACK_COLOR = "#2ca25f"
MOTION_BUCKET_ORDER = ["horizontal", "vertical", "diagonal-mixed", "scale-change", "low-motion"]
OBJECT_TOKENS = ("sphere", "capsule", "cube", "cylinder", "cone", "torus", "pyramid", "box", "object")


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def resize_frame_chw(frame_chw: torch.Tensor, dst_hw: tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        frame_chw.unsqueeze(0),
        size=dst_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0]


def scale_points_xy(points_k2: np.ndarray, *, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(points_k2, dtype=np.float32).copy()
    if out.size == 0:
        return out.reshape(-1, 2)
    scale_x = float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    scale_y = float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    out[..., 0] *= scale_x
    out[..., 1] *= scale_y
    return out


def scale_tracks_xy_tensor(
    tracks_xy: torch.Tensor,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    out = tracks_xy.clone()
    out[..., 0] *= float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    out[..., 1] *= float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    return out


def scale_box_xyxy(box_xyxy: np.ndarray, *, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(box_xyxy, dtype=np.float32).copy()
    if out.shape != (4,):
        return out.astype(np.float32, copy=False)
    scale_x = float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    scale_y = float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    out[[0, 2]] *= scale_x
    out[[1, 3]] *= scale_y
    return out


def resize_mask_hw(mask_hw: np.ndarray, dst_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask_hw, dtype=np.uint8)
    if mask.shape[:2] == dst_hw:
        return mask
    return cv2.resize(mask, (int(dst_hw[1]), int(dst_hw[0])), interpolation=cv2.INTER_NEAREST)


def point_stats(points_k2: np.ndarray) -> dict[str, object]:
    pts = np.asarray(points_k2, dtype=np.float32)
    if pts.size == 0:
        return {
            "count": 0,
            "min_xy": [0.0, 0.0],
            "max_xy": [0.0, 0.0],
            "mean_xy": [0.0, 0.0],
        }
    return {
        "count": int(pts.shape[0]),
        "min_xy": [float(pts[:, 0].min()), float(pts[:, 1].min())],
        "max_xy": [float(pts[:, 0].max()), float(pts[:, 1].max())],
        "mean_xy": [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
    }


def extract_object_token(caption: str) -> str:
    lower = caption.lower()
    for token in OBJECT_TOKENS:
        if token in lower:
            return token
    words = [word for word in lower.replace("-", " ").split() if word]
    return words[-1] if words else "object"


def summarize_motion_from_boxes(
    context_boxes: torch.Tensor | np.ndarray,
    *,
    caption: str,
    metadata: dict[str, object] | None = None,
    dataset_index: int | None = None,
) -> dict[str, object]:
    boxes = np.asarray(context_boxes, dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"expected boxes with shape [T, K, 4], got {boxes.shape}")

    widths = np.clip(boxes[..., 2] - boxes[..., 0], 0.0, None)
    heights = np.clip(boxes[..., 3] - boxes[..., 1], 0.0, None)
    valid = (widths > 1e-6) & (heights > 1e-6)
    centers = np.stack(((boxes[..., 0] + boxes[..., 2]) * 0.5, (boxes[..., 1] + boxes[..., 3]) * 0.5), axis=-1)
    areas = widths * heights

    object_summaries = []
    for obj_idx in range(boxes.shape[1]):
        valid_idx = np.nonzero(valid[:, obj_idx])[0]
        if valid_idx.size < 2:
            continue
        obj_centers = centers[valid_idx, obj_idx]
        obj_areas = areas[valid_idx, obj_idx]
        deltas = np.diff(obj_centers, axis=0)
        mean_step_motion = float(np.linalg.norm(deltas, axis=-1).mean()) if deltas.size > 0 else 0.0
        total_dx = float(obj_centers[-1, 0] - obj_centers[0, 0])
        total_dy = float(obj_centers[-1, 1] - obj_centers[0, 1])
        total_disp = float(math.hypot(total_dx, total_dy))
        size_change = float(np.mean(np.abs(np.diff(obj_areas)))) if obj_areas.size > 1 else 0.0
        area_span = float(obj_areas.max() - obj_areas.min()) if obj_areas.size > 0 else 0.0
        aspect = widths[valid_idx, obj_idx] / np.clip(heights[valid_idx, obj_idx], 1e-6, None)
        aspect_change = float(np.mean(np.abs(np.diff(aspect)))) if aspect.size > 1 else 0.0
        object_summaries.append(
            {
                "obj_idx": int(obj_idx),
                "mean_step_motion": mean_step_motion,
                "total_dx": total_dx,
                "total_dy": total_dy,
                "total_disp": total_disp,
                "size_change": size_change,
                "area_span": area_span,
                "aspect_change": aspect_change,
            }
        )

    if not object_summaries:
        primary = {
            "obj_idx": -1,
            "mean_step_motion": 0.0,
            "total_dx": 0.0,
            "total_dy": 0.0,
            "total_disp": 0.0,
            "size_change": 0.0,
            "area_span": 0.0,
            "aspect_change": 0.0,
        }
    else:
        primary = max(object_summaries, key=lambda item: (item["total_disp"] + item["area_span"] * 0.5, item["mean_step_motion"]))

    metadata = metadata or {}
    return {
        "dataset_index": int(dataset_index) if dataset_index is not None else None,
        "caption": caption,
        "object_token": extract_object_token(caption),
        "sample_id": str(metadata.get("sample_id") or Path(str(metadata.get("sample_dir") or "")).name or "unknown"),
        "window_index": int(metadata.get("window_index", -1)),
        "template_key": str(metadata.get("template_key", "")),
        "primary_object_index": int(primary["obj_idx"]),
        "motion_score": float(primary["mean_step_motion"]),
        "total_disp": float(primary["total_disp"]),
        "total_dx": float(primary["total_dx"]),
        "total_dy": float(primary["total_dy"]),
        "size_change": float(primary["size_change"]),
        "area_span": float(primary["area_span"]),
        "aspect_change": float(primary["aspect_change"]),
        "motion_bucket": "unclassified",
    }


def classify_motion_bucket(
    summary: dict[str, object],
    *,
    motion_q25: float,
    motion_q75: float,
    size_q75: float,
) -> str:
    motion_score = float(summary["motion_score"])
    total_dx = float(summary["total_dx"])
    total_dy = float(summary["total_dy"])
    size_change = float(summary["size_change"])
    disp_abs_x = abs(total_dx)
    disp_abs_y = abs(total_dy)

    low_motion_cutoff = max(motion_q25, 1e-4)
    scale_cutoff = max(size_q75, 1e-4)
    moving_cutoff = max(motion_q75 * 0.5, low_motion_cutoff * 1.5)
    direction_ratio = 1.6

    if motion_score <= low_motion_cutoff and size_change <= scale_cutoff * 0.6:
        return "low-motion"
    if size_change >= scale_cutoff and size_change >= motion_score * 0.75:
        return "scale-change"
    if motion_score >= moving_cutoff and disp_abs_x >= disp_abs_y * direction_ratio:
        return "horizontal"
    if motion_score >= moving_cutoff and disp_abs_y >= disp_abs_x * direction_ratio:
        return "vertical"
    return "diagonal-mixed"


def load_motion_candidate_summary(dataset: PhysStateEpisodeDataset, idx: int) -> dict[str, object]:
    meta_path = dataset.samples[idx]
    npz_path = meta_path.with_suffix(".npz")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with np.load(npz_path) as data:
        context_boxes = data["context_boxes"].astype(np.float32)
        future_boxes = data["future_boxes"].astype(np.float32)
    all_boxes = np.concatenate([context_boxes, future_boxes], axis=0)
    total_frames = int(all_boxes.shape[0])
    context_indices = dataset._select_context_indices(total_frames, idx).cpu().numpy()
    selected_context_boxes = all_boxes[context_indices]
    return summarize_motion_from_boxes(
        selected_context_boxes,
        caption=str(meta["prompt"]),
        metadata=meta,
        dataset_index=idx,
    )


def select_case_summaries(
    dataset: PhysStateEpisodeDataset,
    *,
    start_index: int,
    num_cases: int,
    sample_mode: str,
) -> list[dict[str, object]]:
    if sample_mode == "sequential":
        return [{"dataset_index": idx} for idx in range(start_index, min(len(dataset), start_index + num_cases))]

    candidate_summaries = [load_motion_candidate_summary(dataset, idx) for idx in range(start_index, len(dataset))]
    if not candidate_summaries:
        return []

    motion_scores = np.asarray([float(item["motion_score"]) for item in candidate_summaries], dtype=np.float32)
    size_changes = np.asarray([float(item["size_change"]) for item in candidate_summaries], dtype=np.float32)
    motion_q25 = float(np.quantile(motion_scores, 0.25))
    motion_q75 = float(np.quantile(motion_scores, 0.75))
    size_q75 = float(np.quantile(size_changes, 0.75))
    for item in candidate_summaries:
        item["motion_bucket"] = classify_motion_bucket(
            item,
            motion_q25=motion_q25,
            motion_q75=motion_q75,
            size_q75=size_q75,
        )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in sorted(
        candidate_summaries,
        key=lambda entry: (
            MOTION_BUCKET_ORDER.index(entry["motion_bucket"]) if entry["motion_bucket"] in MOTION_BUCKET_ORDER else len(MOTION_BUCKET_ORDER),
            str(entry["object_token"]),
            -float(entry["motion_score"]),
            -float(entry["size_change"]),
            int(entry["dataset_index"]),
        ),
    ):
        grouped[(str(item["motion_bucket"]), str(item["object_token"]))].append(item)

    ordered_keys = sorted(
        grouped.keys(),
        key=lambda key: (
            MOTION_BUCKET_ORDER.index(key[0]) if key[0] in MOTION_BUCKET_ORDER else len(MOTION_BUCKET_ORDER),
            key[1],
        ),
    )

    selected: list[dict[str, object]] = []
    used_sample_ids: set[str] = set()
    while len(selected) < num_cases:
        made_progress = False
        for key in ordered_keys:
            queue = grouped[key]
            while queue:
                candidate = queue.pop(0)
                sample_id = str(candidate.get("sample_id", ""))
                if sample_id and sample_id in used_sample_ids:
                    continue
                selected.append(candidate)
                if sample_id:
                    used_sample_ids.add(sample_id)
                made_progress = True
                break
            if len(selected) >= num_cases:
                break
        if not made_progress:
            break

    if len(selected) < num_cases:
        leftovers = sorted(
            (item for queue in grouped.values() for item in queue),
            key=lambda entry: (-float(entry["motion_score"]), -float(entry["size_change"]), int(entry["dataset_index"])),
        )
        for candidate in leftovers:
            selected.append(candidate)
            if len(selected) >= num_cases:
                break

    return sorted(selected[:num_cases], key=lambda item: int(item["dataset_index"]))


def pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def init_canvas(frame_chw: torch.Tensor) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    out = Image.fromarray(tensor_frame_to_uint8_hwc(frame_chw))
    return out, ImageDraw.Draw(out)


def draw_gt_boxes(draw: ImageDraw.ImageDraw, gt_boxes_k4: torch.Tensor, *, width: int, height: int) -> None:
    for obj_idx, box in enumerate(gt_boxes_k4.tolist()):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        color = GT_PALETTE[obj_idx % len(GT_PALETTE)]
        draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline=color, width=2)
        draw.text((x0 * width + 2, y0 * height + 2), f"gt{obj_idx}", fill=color)


def draw_query_points(
    draw: ImageDraw.ImageDraw,
    query_points_k2: np.ndarray,
    *,
    width: int,
    height: int,
    image_hw: tuple[int, int],
) -> None:
    scale_x = width / max(float(image_hw[1]), 1.0)
    scale_y = height / max(float(image_hw[0]), 1.0)
    for query_idx, point in enumerate(query_points_k2.tolist()):
        x, y = float(point[0]) * scale_x, float(point[1]) * scale_y
        color = "#111111"
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        draw.text((x + 7, y + 4), f"q{query_idx}", fill=color)


def draw_track_points(
    draw: ImageDraw.ImageDraw,
    tracks_xy_k2: torch.Tensor,
    vis_k: torch.Tensor,
    matched_gt_idx_k: torch.Tensor,
    *,
    width: int,
    height: int,
    image_hw: tuple[int, int],
    label_prefix: str = "q",
    palette: list[str] | None = None,
) -> None:
    colors = palette if palette is not None else QUERY_PALETTE
    scale_x = width / max(float(image_hw[1]), 1.0)
    scale_y = height / max(float(image_hw[0]), 1.0)
    for query_idx, point in enumerate(tracks_xy_k2.tolist()):
        x, y = float(point[0]) * scale_x, float(point[1]) * scale_y
        color = colors[query_idx % len(colors)]
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        gt_idx = int(matched_gt_idx_k[query_idx].item())
        label = f"{label_prefix}{query_idx}->gt{gt_idx}"
        if float(vis_k[query_idx].item()) < 0.5:
            label += "(inv)"
        draw.text((x + 6, y - 6), label, fill=color)


def draw_sam_prompt_box(draw: ImageDraw.ImageDraw, sam_prompt_box_xyxy: np.ndarray) -> None:
    if np.any(sam_prompt_box_xyxy > 0):
        x0, y0, x1, y1 = [float(v) for v in sam_prompt_box_xyxy.tolist()]
        draw.rectangle([x0, y0, x1, y1], outline=SAM_PROMPT_COLOR, width=4)
        draw.text((x0 + 2, max(y0 + 2, 2)), "sam_prompt", fill=SAM_PROMPT_COLOR)


def draw_sam_track_box(draw: ImageDraw.ImageDraw, sam_track_box_xyxy: np.ndarray) -> None:
    if np.any(sam_track_box_xyxy > 0):
        x0, y0, x1, y1 = [float(v) for v in sam_track_box_xyxy.tolist()]
        draw.rectangle([x0, y0, x1, y1], outline=SAM_TRACK_COLOR, width=4)
        draw.text((x0 + 2, max(y0 + 2, 2)), "sam_track", fill=SAM_TRACK_COLOR)


def draw_sam_mask(draw: ImageDraw.ImageDraw, sam_mask_hw: np.ndarray) -> None:
    ys, xs = np.where(sam_mask_hw > 0)
    if xs.size > 0 and ys.size > 0:
        step = max(1, xs.size // 800)
        for x, y in zip(xs[::step], ys[::step]):
            draw.point((float(x), float(y)), fill=SAM_TRACK_COLOR)


def draw_component_boxes(draw: ImageDraw.ImageDraw, components: list[dict[str, object]]) -> None:
    for idx, component in enumerate(components):
        box = np.asarray(component["box"], dtype=np.float32)
        if box.shape != (4,):
            continue
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        color = GT_PALETTE[idx % len(GT_PALETTE)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 2, max(y0 + 2, 2)), f"comp{idx}", fill=color)


def draw_overlay_frame(
    frame_chw: torch.Tensor,
    gt_boxes_k4: torch.Tensor,
    query_points_k2: np.ndarray | None = None,
    tracks_xy_k2: torch.Tensor | None = None,
    vis_k: torch.Tensor | None = None,
    matched_gt_idx_k: torch.Tensor | None = None,
    sam_prompt_box_xyxy: np.ndarray | None = None,
    sam_track_box_xyxy: np.ndarray | None = None,
    sam_mask_hw: np.ndarray | None = None,
    *,
    image_hw: tuple[int, int],
    show_gt: bool = False,
    show_query: bool = False,
    show_tracks: bool = False,
    show_sam_prompt: bool = False,
    show_sam_track: bool = False,
    show_sam_mask: bool = False,
    component_boxes: list[dict[str, object]] | None = None,
    track_label_prefix: str = "q",
    track_palette: list[str] | None = None,
) -> Image.Image:
    out, draw = init_canvas(frame_chw)
    draw = ImageDraw.Draw(out)
    width, height = out.size

    if show_gt:
        draw_gt_boxes(draw, gt_boxes_k4, width=width, height=height)
    if show_query and query_points_k2 is not None:
        draw_query_points(draw, query_points_k2, width=width, height=height, image_hw=image_hw)
    if show_tracks and tracks_xy_k2 is not None and vis_k is not None and matched_gt_idx_k is not None:
        draw_track_points(
            draw,
            tracks_xy_k2,
            vis_k,
            matched_gt_idx_k,
            width=width,
            height=height,
            image_hw=image_hw,
            label_prefix=track_label_prefix,
            palette=track_palette,
        )
    if show_sam_prompt and sam_prompt_box_xyxy is not None:
        draw_sam_prompt_box(draw, sam_prompt_box_xyxy)
    if show_sam_track and sam_track_box_xyxy is not None:
        draw_sam_track_box(draw, sam_track_box_xyxy)
    if show_sam_mask and sam_mask_hw is not None:
        draw_sam_mask(draw, sam_mask_hw)
    if component_boxes is not None:
        draw_component_boxes(draw, component_boxes)

    return out


def draw_sam_debug_frame(
    frame_chw: torch.Tensor,
    gt_boxes_k4: torch.Tensor,
    sam_prompt_box_xyxy: np.ndarray,
    sam_track_box_xyxy: np.ndarray,
    sam_mask_hw: np.ndarray,
    *,
    prompt_frame: bool,
) -> Image.Image:
    return draw_overlay_frame(
        frame_chw=frame_chw,
        gt_boxes_k4=gt_boxes_k4,
        sam_prompt_box_xyxy=sam_prompt_box_xyxy if prompt_frame else None,
        sam_track_box_xyxy=sam_track_box_xyxy,
        sam_mask_hw=sam_mask_hw,
        image_hw=(frame_chw.shape[-2], frame_chw.shape[-1]),
        show_gt=True,
        show_sam_prompt=prompt_frame,
        show_sam_track=True,
        show_sam_mask=True,
    )


def ensure_browser_video(source_path: Path) -> Path:
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return source_path
    if out_path.exists() and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns and out_path.stat().st_size > 0:
        return out_path
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def build_stage_rows_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_order = [
        ("stage1_context", "1. Context Video", "原始 context_video，保持 native 分辨率。"),
        ("stage2_sam2", "2. SAM2 Frame0 Sampling Basis", "展示第0帧 connected components、整张 mask 和 box，这三者就是 query priors 的采样依据。"),
        ("stage3_query_priors", "3. Query Priors From SAM2", "从 SAM2 mask / box 采样 query priors，仍画在 native 分辨率视频上。"),
        ("stage4_model_inputs", "4. Resized Inputs For VGGT / CoTracker", "分别缩放到 VGGT 和 CoTracker 的实际输入分辨率，再把 priors overlay 上去。"),
        ("stage5_tracks", "5. Output Tracks", "分别在 VGGT / CoTracker 的实际执行分辨率下展示最终 track。"),
    ]

    row_blocks = []
    for stage_key, stage_title, stage_desc in stage_order:
        case_cards = []
        for idx, result in enumerate(results):
            entries = result["stage_rows"].get(stage_key, [])
            if not entries:
                continue
            entry_cards = []
            for entry in entries:
                entry_cards.append(
                    f"""
        <figure class="stage-video-card">
          <video controls preload="none" playsinline src="{entry['path']}"></video>
          <figcaption><b>{entry['title']}</b><br>{entry['source']}</figcaption>
        </figure>
"""
                )
            case_cards.append(
                f"""
    <article class="case-card">
      <h3>Case {idx}</h3>
      <p><b>Caption:</b> {html.escape(result['caption'])}</p>
      <p><b>Motion:</b> {html.escape(result['motion_bucket'])} | <b>Shape:</b> {html.escape(result['object_token'])} | <b>Sample:</b> {html.escape(result['sample_id'])} / window {result['window_index']}</p>
      <div class="stage-video-stack">
        {''.join(entry_cards)}
      </div>
    </article>
"""
            )
        row_blocks.append(
            f"""
  <section class="stage-row">
    <h2>{stage_title}</h2>
    <p>{stage_desc}</p>
    <div class="case-grid">
      {''.join(case_cards)}
    </div>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Track Pipeline Viewer</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .stage-row {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .case-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 16px; align-items: start; }}
    .case-card {{ background: #fff; border: 1px solid #ddd; padding: 12px; }}
    .case-card h3 {{ margin: 0 0 8px 0; }}
    .case-card p {{ font-size: 13px; color: #444; }}
    .stage-video-stack {{ display: grid; gap: 12px; }}
    .stage-video-card {{ margin: 0; border: 1px solid #e5e1d8; padding: 8px; background: #fcfbf8; }}
    video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Track Pipeline Viewer</h1>
  <p>页面按 5 个阶段横向展示多个 case。每个阶段里的 overlay 都画在该模块真实执行时使用的分辨率视频上：native 阶段画在 native 视频上，VGGT 阶段画在 `vggt_input_hw` 上，CoTracker 阶段画在 `cotracker_input_hw` 上。</p>
  {''.join(row_blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def build_track_source_compare_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_keys = [
        "raw_context",
        "sam_prompt_only",
        "sam_mask_only",
        "sam_track_only",
        "query_priors_native",
        "raw_context_vggt_input",
        "sam2_priors_vggt_input",
        "vggt_tracks_only",
        "cotracker_tracks_only",
    ]
    card_titles = {
        "raw_context": ("1. Raw Context Video", "来源: 原始 context video"),
        "sam_prompt_only": ("2. SAM2 Prompt Box", "来源: SAM2 motion prompt box"),
        "sam_mask_only": ("3. SAM2 Mask Track", "来源: SAM2 传播得到的 mask"),
        "sam_track_only": ("4. SAM2 Track Box", "来源: SAM2 传播得到的 track box"),
        "query_priors_native": ("5. Native Query Points", "来源: native 分辨率下采样得到的 query priors"),
        "raw_context_vggt_input": ("6. Raw Context Video (VGGT Input)", "来源: 缩放到 VGGT 实际输入分辨率的 context"),
        "sam2_priors_vggt_input": ("7. SAM2 Priors Overlay (VGGT Input)", "来源: VGGT 输入分辨率下的 priors overlay"),
        "vggt_tracks_only": ("8. VGGT Tracked Points", "来源: track_source=vggt"),
        "cotracker_tracks_only": ("9. CoTracker Tracked Points", "来源: track_source=cotracker"),
    }

    case_blocks = []
    for idx, result in enumerate(results):
        video_lookup = {entry["key"]: entry for entry in result["videos"]}
        cards = []
        for key in ordered_keys:
            entry = video_lookup.get(key)
            if entry is None:
                continue
            title, source = card_titles[key]
            cards.append(
                f"""
    <figure class="video-card">
      <video autoplay muted loop playsinline preload="metadata" src="{entry['path']}" onclick="this.paused ? this.play() : this.pause();" title="点击播放或暂停"></video>
      <figcaption><b>{title}</b><br>{source}</figcaption>
    </figure>
"""
            )
        case_blocks.append(
            f"""
  <section class="case">
    <div class="case-head">
      <div>
        <h2>Case {idx}</h2>
        <p><b>Caption:</b> {html.escape(result['caption'])}</p>
      </div>
      <div class="chips">
        <span class="chip">{html.escape(result['motion_bucket'])}</span>
        <span class="chip">{html.escape(result['object_token'])}</span>
        <span class="chip">sample {html.escape(result['sample_id'])}</span>
        <span class="chip">window {result['window_index']}</span>
      </div>
    </div>
    <p class="metrics">mean_step_motion={result['motion_score']:.4f} | total_disp={result['total_disp']:.4f} | size_change={result['size_change']:.4f}</p>
    <div class="video-grid">
      {''.join(cards)}
    </div>
  </section>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Track Source Compare</title>
  <style>
    :root {{
      --bg: #f6f4ee;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #5b6873;
      --line: #ddd6c8;
      --chip: #efe7da;
      --accent: #8c4a2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: sans-serif; margin: 20px; background: var(--bg); color: var(--ink); }}
    h1, h2, p {{ margin: 0; }}
    .hero {{ background: var(--panel); border: 1px solid var(--line); padding: 18px; margin-bottom: 18px; }}
    .hero p {{ margin-top: 10px; color: var(--muted); line-height: 1.6; }}
    .case {{ margin-bottom: 28px; padding: 16px; background: var(--panel); border: 1px solid var(--line); }}
    .case-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .case-head p {{ margin-top: 8px; color: var(--muted); line-height: 1.5; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{ background: var(--chip); color: var(--accent); padding: 6px 10px; font-size: 12px; border-radius: 999px; white-space: nowrap; }}
    .metrics {{ margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(9, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 16px;
      align-items: start;
    }}
    .video-card {{
      margin: 0;
      background: #fffdf9;
      border: 1px solid var(--line);
      padding: 10px;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(31, 41, 51, 0.05);
    }}
    video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      border: 1px solid #ccc;
      border-radius: 8px;
      background: #000;
      cursor: pointer;
    }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 6px; line-height: 1.5; }}
    .hint {{ margin-top: 12px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 1800px) {{
      .video-grid {{ grid-template-columns: repeat(5, minmax(220px, 1fr)); }}
    }}
    @media (max-width: 1320px) {{
      .video-grid {{ grid-template-columns: repeat(3, minmax(220px, 1fr)); }}
    }}
    @media (max-width: 860px) {{
      .video-grid {{ grid-template-columns: repeat(2, minmax(220px, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .video-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 900px) {{
      .case-head {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <section class="hero">
    <h1>Track Source Compare</h1>
    <p>这里把训练集中不同运动模式的 case 放在同一页。每个 case 共用同一段 context video 和同一批 query priors，按训练里的实际处理顺序展示从 SAM2 motion prior 到最终 active tracks 的链路，最后并排对比 <code>track_source=vggt</code> 和 <code>track_source=cotracker</code>。</p>
    <p class="hint">每个 case 内部改为横向卡片布局；视频默认静音循环播放，点击单个视频可播放/暂停，不再显示原生进度滑块。</p>
  </section>
  {''.join(case_blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return html_path


def build_report(results: list[dict], output_dir: Path, *, report_style: str) -> Path:
    if report_style == "track_source_compare":
        return build_track_source_compare_report(results, output_dir)
    return build_stage_rows_report(results, output_dir)


def evaluate_sample(
    sample: dict,
    adapter: VGGTTrackAdapter,
    cotracker_adapter: CoTrackerAdapter | None,
    device: torch.device,
    output_dir: Path,
) -> dict:
    context_video = sample["context_video"].unsqueeze(0).to(device)
    context_boxes = sample["context_boxes"].unsqueeze(0).to(device)

    native_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))

    frames_tchw_01 = ((sample["context_video"].float() + 1.0) / 2.0).permute(1, 0, 2, 3).cpu().numpy()
    prompt_frame_idx = max(int(context_video.shape[2]) - 1, 0)
    motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
    sam_tracker = SAM2MotionTracker(device=str(device), segment_len=8, enable_text_prompt=False)
    sam_out = sam_tracker.track(
        frames_tchw_01,
        prompt_frame_idx=prompt_frame_idx,
        prompt_box_xyxy=motion_prompt_box_xyxy,
        caption=sample["caption"],
    )
    frame0_components = _extract_mask_components(sam_out.masks_thw[0])
    query_points_prior_px, sam_prior_source = build_vggt_query_prior(
        sam_out.masks_thw,
        sam_out.boxes_t4,
        num_queries=adapter.num_queries,
    )
    query_points_prior = torch.from_numpy(query_points_prior_px).unsqueeze(0).to(device=device, dtype=context_video.dtype)

    frames_bthwc = context_video.permute(0, 2, 3, 4, 1).float()
    frames_bthwc = (frames_bthwc + 1.0) / 2.0
    with torch.no_grad():
        vggt_out = adapter(
            frames_bthwc,
            query_points_prior=query_points_prior,
            query_image_hw=native_hw,
        )

    tracks = vggt_out.tracks
    track_image_hw = vggt_out.image_hw
    query_points_vggt_input_px = vggt_out.query_points[0].detach().cpu().numpy().astype(np.float32)
    query_points_roundtrip_px = scale_points_xy(
        query_points_vggt_input_px,
        src_hw=track_image_hw,
        dst_hw=native_hw,
    )
    query_roundtrip_abs_err = np.abs(query_points_roundtrip_px - query_points_prior_px) if query_points_prior_px.size > 0 else np.zeros((0, 2), dtype=np.float32)
    query_roundtrip_max_abs_err_px = float(query_roundtrip_abs_err.max()) if query_roundtrip_abs_err.size > 0 else 0.0
    query_roundtrip_mean_abs_err_px = float(query_roundtrip_abs_err.mean()) if query_roundtrip_abs_err.size > 0 else 0.0

    scale_x = float(context_video.shape[-1]) / float(track_image_hw[1])
    scale_y = float(context_video.shape[-2]) / float(track_image_hw[0])
    tracks_native = tracks.clone()
    tracks_native[..., 0] *= scale_x
    tracks_native[..., 1] *= scale_y
    tracks_native_px = tracks_native[0].detach().cpu().numpy().astype(np.float32)

    alignment = align_tracks_to_boxes(
        tracks=tracks_native,
        gt_boxes=context_boxes,
        image_hw=(context_video.shape[-2], context_video.shape[-1]),
    )
    matched_gt_indices = [int(x) for x in alignment.matched_gt_indices[0].tolist()]
    valid_gt_mask = ((context_boxes[..., 2] - context_boxes[..., 0] > 1e-6) & (context_boxes[..., 3] - context_boxes[..., 1] > 1e-6)).any(dim=1)[0]
    valid_gt_indices = [int(i) for i in torch.nonzero(valid_gt_mask, as_tuple=False).flatten().tolist()]
    unmatched_gt_indices = [i for i in valid_gt_indices if i not in set(matched_gt_indices)]

    cotracker_out = None
    cotracker_alignment = None
    cotracker_tracks_native_px = np.zeros((0, 2), dtype=np.float32)
    cotracker_mean_center_l1_px = None
    cotracker_valid_track_points = None
    if cotracker_adapter is not None:
        with torch.no_grad():
            cotracker_out = cotracker_adapter(
                frames_bthwc,
                query_points_prior=query_points_prior,
                query_image_hw=native_hw,
            )
        cotracker_alignment = align_tracks_to_boxes(
            tracks=cotracker_out.tracks,
            gt_boxes=context_boxes,
            image_hw=native_hw,
        )
        cotracker_tracks_native_px = cotracker_out.tracks[0].detach().cpu().numpy().astype(np.float32)
        cot_valid_mask = cotracker_alignment.matched_gt_valid > 0.5
        if cot_valid_mask.any():
            cot_l1 = (cotracker_out.tracks - cotracker_alignment.matched_gt_centers).abs().sum(dim=-1)
            cotracker_mean_center_l1_px = float(cot_l1[cot_valid_mask].mean().item())
        else:
            cotracker_mean_center_l1_px = 0.0
        cotracker_valid_track_points = int(cot_valid_mask.sum().item())
    cotracker_input_hw = tuple(cotracker_out.input_hw) if cotracker_out is not None else None
    cotracker_query_points_input_px = (
        scale_points_xy(query_points_prior_px, src_hw=native_hw, dst_hw=cotracker_input_hw) if cotracker_input_hw is not None else None
    )

    video_buffers: dict[str, list[np.ndarray]] = {
        "raw_context": [],
        "raw_context_vggt_input": [],
        "sam_prompt_only": [],
        "sam_mask_only": [],
        "sam_track_only": [],
        "query_priors_native": [],
        "vggt_tracks_only": [],
        "cotracker_tracks_only": [],
        "sam2_frame0_components": [],
        "sam2_priors_vggt_input": [],
        "sam2_priors_cotracker_input": [],
    }
    context_frames = sample["context_video"].permute(1, 0, 2, 3)
    for t in range(context_frames.shape[0]):
        raw_img = Image.fromarray(tensor_frame_to_uint8_hwc(context_frames[t]))
        video_buffers["raw_context"].append(np.array(raw_img))

        resized_frame = resize_frame_chw(context_frames[t], track_image_hw)
        resized_raw_img = Image.fromarray(tensor_frame_to_uint8_hwc(resized_frame))
        video_buffers["raw_context_vggt_input"].append(np.array(resized_raw_img))
        cotracker_resized_frame = resize_frame_chw(context_frames[t], cotracker_input_hw) if cotracker_input_hw is not None else None

        prompt_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            image_hw=native_hw,
            sam_prompt_box_xyxy=sam_out.prompt_box_xyxy if t == prompt_frame_idx else None,
            show_sam_prompt=(t == prompt_frame_idx),
        )
        video_buffers["sam_prompt_only"].append(np.array(prompt_img))

        query_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            query_points_k2=query_points_prior_px if t == 0 else None,
            image_hw=native_hw,
            show_query=(t == 0),
        )
        video_buffers["query_priors_native"].append(np.array(query_img))

        comp_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            image_hw=native_hw,
            sam_mask_hw=sam_out.masks_thw[0] if t == 0 else None,
            show_sam_mask=(t == 0),
            component_boxes=frame0_components if t == 0 else None,
        )
        video_buffers["sam2_frame0_components"].append(np.array(comp_img))

        mask_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            image_hw=native_hw,
            sam_mask_hw=sam_out.masks_thw[t],
            show_sam_mask=True,
        )
        video_buffers["sam_mask_only"].append(np.array(mask_img))

        box_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            image_hw=native_hw,
            sam_track_box_xyxy=sam_out.boxes_t4[t],
            show_sam_track=True,
        )
        video_buffers["sam_track_only"].append(np.array(box_img))

        query_vggt_input_img = draw_overlay_frame(
            frame_chw=resized_frame,
            gt_boxes_k4=sample["context_boxes"][t],
            query_points_k2=query_points_vggt_input_px if t == 0 else None,
            image_hw=track_image_hw,
            sam_prompt_box_xyxy=scale_box_xyxy(sam_out.prompt_box_xyxy, src_hw=native_hw, dst_hw=track_image_hw) if t == prompt_frame_idx else None,
            sam_track_box_xyxy=scale_box_xyxy(sam_out.boxes_t4[t], src_hw=native_hw, dst_hw=track_image_hw),
            sam_mask_hw=resize_mask_hw(sam_out.masks_thw[t], track_image_hw),
            show_query=(t == 0),
            show_sam_prompt=(t == prompt_frame_idx),
            show_sam_track=True,
            show_sam_mask=True,
        )
        video_buffers["sam2_priors_vggt_input"].append(np.array(query_vggt_input_img))

        if cotracker_resized_frame is not None and cotracker_input_hw is not None and cotracker_query_points_input_px is not None:
            cot_prompt = scale_box_xyxy(sam_out.prompt_box_xyxy, src_hw=native_hw, dst_hw=cotracker_input_hw) if t == prompt_frame_idx else None
            cot_track_box = scale_box_xyxy(sam_out.boxes_t4[t], src_hw=native_hw, dst_hw=cotracker_input_hw)
            cot_mask = resize_mask_hw(sam_out.masks_thw[t], cotracker_input_hw)
            query_cot_input_img = draw_overlay_frame(
                frame_chw=cotracker_resized_frame,
                gt_boxes_k4=sample["context_boxes"][t],
                query_points_k2=cotracker_query_points_input_px if t == 0 else None,
                image_hw=cotracker_input_hw,
                sam_prompt_box_xyxy=cot_prompt,
                sam_track_box_xyxy=cot_track_box,
                sam_mask_hw=cot_mask,
                show_query=(t == 0),
                show_sam_prompt=(t == prompt_frame_idx),
                show_sam_track=True,
                show_sam_mask=True,
            )
            video_buffers["sam2_priors_cotracker_input"].append(np.array(query_cot_input_img))

        track_img = draw_overlay_frame(
            frame_chw=resized_frame,
            gt_boxes_k4=sample["context_boxes"][t],
            tracks_xy_k2=tracks[0, t].detach().cpu(),
            vis_k=vggt_out.visibility[0, t].detach().cpu(),
            matched_gt_idx_k=alignment.matched_gt_indices[0].detach().cpu(),
            image_hw=track_image_hw,
            show_tracks=True,
            track_label_prefix="v",
            track_palette=QUERY_PALETTE,
        )
        video_buffers["vggt_tracks_only"].append(np.array(track_img))

        if cotracker_out is not None and cotracker_alignment is not None and cotracker_resized_frame is not None and cotracker_input_hw is not None:
            cotracker_tracks_input = scale_tracks_xy_tensor(
                cotracker_out.tracks[0, t].detach().cpu(),
                src_hw=native_hw,
                dst_hw=cotracker_input_hw,
            )
            cotrack_img = draw_overlay_frame(
                frame_chw=cotracker_resized_frame,
                gt_boxes_k4=sample["context_boxes"][t],
                tracks_xy_k2=cotracker_tracks_input,
                vis_k=cotracker_out.visibility[0, t].detach().cpu(),
                matched_gt_idx_k=cotracker_alignment.matched_gt_indices[0].detach().cpu(),
                image_hw=cotracker_input_hw,
                show_tracks=True,
                track_label_prefix="c",
                track_palette=COTRACKER_PALETTE,
            )
            video_buffers["cotracker_tracks_only"].append(np.array(cotrack_img))

    video_specs = {
        "raw_context": ("Context Video", "native"),
        "raw_context_vggt_input": ("Raw Context Video (VGGT Input)", f"VGGT @ {track_image_hw[0]}x{track_image_hw[1]}"),
        "sam_prompt_only": ("SAM2 Prompt Box", "native"),
        "sam_mask_only": ("SAM2 Mask Track", "native"),
        "sam_track_only": ("SAM2 Track Box", "native"),
        "sam2_frame0_components": ("Frame0 Connected Components", "native"),
        "query_priors_native": ("Query Priors", "native"),
        "sam2_priors_vggt_input": ("VGGT Input Overlay", f"VGGT @ {track_image_hw[0]}x{track_image_hw[1]}"),
        "sam2_priors_cotracker_input": (
            "CoTracker Input Overlay",
            f"CoTracker @ {cotracker_input_hw[0]}x{cotracker_input_hw[1]}" if cotracker_input_hw is not None else "CoTracker",
        ),
        "vggt_tracks_only": ("VGGT Tracks", f"VGGT @ {track_image_hw[0]}x{track_image_hw[1]}"),
        "cotracker_tracks_only": (
            "CoTracker Tracks",
            f"CoTracker @ {cotracker_input_hw[0]}x{cotracker_input_hw[1]}" if cotracker_input_hw is not None else "CoTracker",
        ),
    }
    browser_videos = {}
    for key, (title, source) in video_specs.items():
        if len(video_buffers[key]) == 0:
            continue
        raw_path = output_dir / f"{Path(sample['video_path']).stem}__{key}.mp4"
        write_mp4(raw_path, np.stack(video_buffers[key], axis=0), fps=int(sample.get("_fps", 8)))
        browser_path = ensure_browser_video(raw_path)
        browser_videos[key] = {
            "key": key,
            "title": title,
            "source": source,
            "path": str(browser_path.relative_to(output_dir.parent)),
        }

    stage_rows = {
        "stage1_context": [browser_videos["raw_context"]] if "raw_context" in browser_videos else [],
        "stage2_sam2": [
            browser_videos[key]
            for key in ("sam_prompt_only", "sam2_frame0_components", "sam_mask_only", "sam_track_only")
            if key in browser_videos
        ],
        "stage3_query_priors": [browser_videos["query_priors_native"]] if "query_priors_native" in browser_videos else [],
        "stage4_model_inputs": [
            browser_videos[key]
            for key in ("raw_context_vggt_input", "sam2_priors_vggt_input", "sam2_priors_cotracker_input")
            if key in browser_videos
        ],
        "stage5_tracks": [
            browser_videos[key]
            for key in ("vggt_tracks_only", "cotracker_tracks_only")
            if key in browser_videos
        ],
    }

    return {
        "caption": sample["caption"],
        "video_path": sample["video_path"],
        "stage_rows": stage_rows,
        "videos": list(browser_videos.values()),
        "motion_bucket": str(sample["_selection_summary"]["motion_bucket"]),
        "object_token": str(sample["_selection_summary"]["object_token"]),
        "sample_id": str(sample["_selection_summary"]["sample_id"]),
        "window_index": int(sample["_selection_summary"]["window_index"]),
        "motion_score": float(sample["_selection_summary"]["motion_score"]),
        "total_disp": float(sample["_selection_summary"]["total_disp"]),
        "size_change": float(sample["_selection_summary"]["size_change"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--sample-mode", choices=["sequential", "diverse"], default="sequential")
    parser.add_argument("--report-style", choices=["stage_rows", "track_source_compare"], default="stage_rows")
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/vggt_query_points_overlay",
    )
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--disable-cotracker", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PhysStateEpisodeDataset(
        root=data_cfg["root"],
        split=args.split,
        resolution=tuple(data_cfg["resolution"]),
        num_context_frames=int(data_cfg["num_context_frames"]),
        context_fraction=float(data_cfg.get("context_fraction", 0.5)),
        random_context_frames=bool(data_cfg.get("random_context_frames", True)),
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    adapter = VGGTTrackAdapter(
        model_path=model_cfg.get("vggt_model_path"),
        num_queries=int(model_cfg["object_num_queries"]),
        device=str(device),
        input_hw=tuple(model_cfg["vggt_input_hw"]),
    ).to(device)
    cotracker_checkpoint = model_cfg.get("cotracker_checkpoint", "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    cotracker_adapter = None
    if not args.disable_cotracker and cotracker_checkpoint and Path(cotracker_checkpoint).exists():
        cotracker_adapter = CoTrackerAdapter(
            checkpoint_path=str(cotracker_checkpoint),
            num_queries=int(model_cfg["object_num_queries"]),
            device=str(device),
            input_hw=tuple(model_cfg.get("cotracker_input_hw", [384, 512])),
            window_len=int(model_cfg.get("cotracker_window_len", 60)),
        ).to(device)

    output_dir = Path(args.output_dir)
    results = []
    selected_summaries = select_case_summaries(
        dataset,
        start_index=args.start_index,
        num_cases=args.num_cases,
        sample_mode=args.sample_mode,
    )
    for selected in selected_summaries:
        idx = int(selected["dataset_index"])
        sample = dataset[idx]
        sample["_output_dir"] = str(output_dir / "assets")
        sample["_case_name"] = f"case_{idx:03d}"
        sample["_fps"] = int(data_cfg.get("fps", 8))
        sample["_selection_summary"] = (
            selected
            if "motion_bucket" in selected
            else summarize_motion_from_boxes(
                sample["context_boxes"],
                caption=sample["caption"],
                metadata=sample.get("metadata", {}),
                dataset_index=idx,
            )
        )
        results.append(evaluate_sample(sample, adapter, cotracker_adapter, device, output_dir / "assets"))

    html_path = build_report(results, output_dir, report_style=args.report_style)
    print(f"eval report: {html_path}")
    if args.no_serve:
        return

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{args.port}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
