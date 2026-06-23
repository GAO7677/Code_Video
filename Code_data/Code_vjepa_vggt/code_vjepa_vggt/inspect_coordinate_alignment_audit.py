from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from code_vjepa_vggt.train0419_reference.AAAinfer.training_flow_notebook_helper import TrainingFlowInspector
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes


RAW_COLOR = (240, 240, 240)
PROMPT_COLOR = (255, 140, 0)
SAM_MASK_COLORS = [
    (214, 40, 40),
    (247, 127, 0),
    (252, 191, 73),
    (42, 157, 143),
]
QUERY_COLORS = [
    (0, 180, 216),
    (0, 119, 182),
    (131, 56, 236),
    (58, 134, 255),
    (255, 0, 110),
    (251, 86, 7),
    (46, 196, 182),
    (138, 201, 38),
]
VGGT_COLOR = (0, 180, 216)
ACTIVE_COLOR = (43, 138, 62)
GT_COLOR = (255, 255, 255)
HEADER_BG = (18, 21, 27)
MASK_ALPHA = 0.26


@dataclass
class ModuleCard:
    slug: str
    title: str
    coord_note: str
    mapping_note: str
    verdict: str
    metrics: list[str]
    media_relpath: str


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def draw_box_rgb(image: np.ndarray, box_xyxy: np.ndarray, color_rgb: tuple[int, int, int], label: str, width: int = 2) -> None:
    box = np.asarray(box_xyxy, dtype=np.float32)
    if box.shape != (4,):
        return
    x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, int(width))
    if label:
        cv2.putText(image, label, (x0 + 4, max(16, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color_bgr, 1, cv2.LINE_AA)


def draw_point_rgb(
    image: np.ndarray,
    point_xy: np.ndarray,
    color_rgb: tuple[int, int, int],
    label: str,
    *,
    radius: int = 5,
    filled: bool = False,
) -> None:
    point = np.asarray(point_xy, dtype=np.float32)
    if point.shape != (2,):
        return
    x, y = [int(round(v)) for v in point.tolist()]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    thickness = -1 if filled else 2
    cv2.circle(image, (x, y), int(radius), color_bgr, thickness)
    if label:
        cv2.putText(image, label, (x + 6, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)


def overlay_mask_rgb(frame_hwc: np.ndarray, mask_hw: np.ndarray, color_rgb: tuple[int, int, int], alpha: float = MASK_ALPHA) -> np.ndarray:
    frame = frame_hwc.astype(np.float32).copy()
    mask = np.asarray(mask_hw) > 0
    if np.any(mask):
        color = np.asarray(color_rgb, dtype=np.float32)
        frame[mask] = (1.0 - alpha) * frame[mask] + alpha * color[None, :]
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def annotate_frame(frame_hwc: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = frame_hwc.copy()
    height, width = canvas.shape[:2]
    line_height = 21
    block_h = min(height, 12 + line_height * max(len(lines), 1))
    cv2.rectangle(canvas, (0, 0), (width, block_h), (HEADER_BG[2], HEADER_BG[1], HEADER_BG[0]), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            str(line),
            (12, 24 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_gif(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames_thwc_uint8]
    duration_ms = max(int(round(1000.0 / max(float(fps), 1.0))), 40)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def ensure_browser_video(source_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return source_path
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
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


def scale_tracks_xy(tracks: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    out = tracks.clone()
    out[..., 0] *= float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    out[..., 1] *= float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    return out


def box_norm_to_px(box_xyxy_norm: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(image_hw[0]), int(image_hw[1])
    box = np.asarray(box_xyxy_norm, dtype=np.float32)
    if box.shape != (4,):
        return np.zeros((4,), dtype=np.float32)
    return np.asarray([box[0] * w, box[1] * h, box[2] * w, box[3] * h], dtype=np.float32)


def gt_boxes_frame_px(context_boxes: torch.Tensor, frame_idx: int, image_hw: tuple[int, int]) -> np.ndarray:
    boxes = context_boxes[0, frame_idx].detach().cpu().numpy()
    return np.stack([box_norm_to_px(box, image_hw) for box in boxes], axis=0)


def box_valid(box_xyxy: np.ndarray, eps: float = 1.0e-6) -> bool:
    box = np.asarray(box_xyxy, dtype=np.float32)
    return bool(box.shape == (4,) and float(box[2] - box[0]) > eps and float(box[3] - box[1]) > eps)


def point_inside_box(point_xy: np.ndarray, box_xyxy: np.ndarray) -> bool:
    point = np.asarray(point_xy, dtype=np.float32)
    box = np.asarray(box_xyxy, dtype=np.float32)
    return bool(
        box_valid(box)
        and float(box[0]) <= float(point[0]) <= float(box[2])
        and float(box[1]) <= float(point[1]) <= float(box[3])
    )


def box_center_xy(box_xyxy: np.ndarray) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32)
    return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    if not box_valid(box_a) or not box_valid(box_b):
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in np.asarray(box_a, dtype=np.float32)]
    bx0, by0, bx1, by1 = [float(v) for v in np.asarray(box_b, dtype=np.float32)]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(inter_x1 - inter_x0, 0.0)
    inter_h = max(inter_y1 - inter_y0, 0.0)
    inter = inter_w * inter_h
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = max(area_a + area_b - inter, 1.0e-6)
    return float(inter / union)


def best_gt_iou(box_xyxy: np.ndarray, gt_boxes_px_k4: np.ndarray) -> float:
    if gt_boxes_px_k4.size == 0:
        return 0.0
    return max(box_iou_xyxy(box_xyxy, gt_box) for gt_box in gt_boxes_px_k4)


def point_region_metrics(
    points_k2: np.ndarray,
    gt_boxes_px_k4: np.ndarray,
    *,
    sam_boxes_px_m4: np.ndarray | None = None,
) -> dict[str, float]:
    gt_valid = [box for box in gt_boxes_px_k4 if box_valid(box)]
    gt_centers = [box_center_xy(box) for box in gt_valid]
    inside_any_gt = 0
    inside_sam = 0
    min_center_dists: list[float] = []
    for point in np.asarray(points_k2, dtype=np.float32):
        if any(point_inside_box(point, box) for box in gt_valid):
            inside_any_gt += 1
        if sam_boxes_px_m4 is not None and any(point_inside_box(point, box) for box in sam_boxes_px_m4 if box_valid(box)):
            inside_sam += 1
        if gt_centers:
            min_center_dists.append(min(float(np.linalg.norm(point - center)) for center in gt_centers))
    count = max(int(points_k2.shape[0]), 1)
    return {
        "inside_any_gt_rate": float(inside_any_gt / count),
        "inside_sam_box_rate": float(inside_sam / count) if sam_boxes_px_m4 is not None else float("nan"),
        "mean_min_gt_center_dist_px": float(np.mean(min_center_dists)) if min_center_dists else float("nan"),
    }


def track_alignment_metrics(
    tracks_tk2: np.ndarray,
    gt_boxes_tk4: np.ndarray,
    matched_gt_indices_k: np.ndarray,
) -> dict[str, float]:
    center_distances: list[float] = []
    inside_hits = 0
    inside_total = 0
    for t in range(tracks_tk2.shape[0]):
        for q_idx in range(tracks_tk2.shape[1]):
            gt_idx = int(matched_gt_indices_k[q_idx])
            if gt_idx < 0 or gt_idx >= gt_boxes_tk4.shape[1]:
                continue
            gt_box = gt_boxes_tk4[t, gt_idx]
            if not box_valid(gt_box):
                continue
            point_xy = tracks_tk2[t, q_idx]
            center_distances.append(float(np.linalg.norm(point_xy - box_center_xy(gt_box))))
            inside_total += 1
            if point_inside_box(point_xy, gt_box):
                inside_hits += 1
    if not center_distances:
        return {
            "mean_center_dist_px": float("nan"),
            "max_center_dist_px": float("nan"),
            "inside_rate": float("nan"),
        }
    return {
        "mean_center_dist_px": float(np.mean(center_distances)),
        "max_center_dist_px": float(np.max(center_distances)),
        "inside_rate": float(inside_hits / max(inside_total, 1)),
    }


def prompt_box_metrics(prompt_boxes: list[np.ndarray], gt_boxes_px_k4: np.ndarray) -> dict[str, float]:
    best_ious = [best_gt_iou(box, gt_boxes_px_k4) for box in prompt_boxes if box_valid(box)]
    if not best_ious:
        return {"mean_best_iou": float("nan"), "max_best_iou": float("nan")}
    return {
        "mean_best_iou": float(np.mean(best_ious)),
        "max_best_iou": float(np.max(best_ious)),
    }


def sam_track_box_metrics(sam_boxes_tmk4: np.ndarray, gt_boxes_tk4: np.ndarray) -> dict[str, float]:
    best_ious: list[float] = []
    for t in range(sam_boxes_tmk4.shape[0]):
        for m_idx in range(sam_boxes_tmk4.shape[1]):
            box = sam_boxes_tmk4[t, m_idx]
            if not box_valid(box):
                continue
            best_ious.append(best_gt_iou(box, gt_boxes_tk4[t]))
    if not best_ious:
        return {"mean_best_iou": float("nan"), "max_best_iou": float("nan")}
    return {
        "mean_best_iou": float(np.mean(best_ious)),
        "max_best_iou": float(np.max(best_ious)),
    }


def fmt_float(value: float, suffix: str = "", digits: int = 2) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}{suffix}"


def verdict_from_inside_rate(inside_rate: float, mean_dist_px: float, *, strong_px: float = 28.0, weak_px: float = 48.0) -> str:
    if inside_rate == inside_rate and mean_dist_px == mean_dist_px:
        if inside_rate >= 0.80 and mean_dist_px <= strong_px:
            return "大体锁定同一物体区域。"
        if inside_rate >= 0.60 and mean_dist_px <= weak_px:
            return "基本在同一区域，但边界附近有可见漂移。"
    return "和 GT / 其他模块不够一致，需要重点检查。"


def render_raw_context_video(context_video: torch.Tensor, caption: str) -> np.ndarray:
    frames = []
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t])
        frame = annotate_frame(frame, [f"Raw Context | native frame t={t}", caption])
        frames.append(frame)
    return np.stack(frames, axis=0)


def render_prompt_boxes_video(context_video: torch.Tensor, prompt_frame_idx: int, prompt_boxes: list[np.ndarray], caption: str) -> np.ndarray:
    frames = []
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t])
        if t == int(prompt_frame_idx):
            for idx, box in enumerate(prompt_boxes):
                draw_box_rgb(frame, box, PROMPT_COLOR, f"prompt{idx}")
        frame = annotate_frame(
            frame,
            [
                f"SAM2 Prompt Boxes | native xyxy | prompt_frame={prompt_frame_idx} | t={t}",
                caption,
            ],
        )
        frames.append(frame)
    return np.stack(frames, axis=0)


def render_sam_tracks_video(
    context_video: torch.Tensor,
    prompt_frame_idx: int,
    prompt_boxes: list[np.ndarray],
    sam_masks_tmh: np.ndarray,
    sam_boxes_tmk4: np.ndarray,
    caption: str,
) -> np.ndarray:
    frames = []
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t])
        for obj_idx in range(sam_masks_tmh.shape[1]):
            color = SAM_MASK_COLORS[obj_idx % len(SAM_MASK_COLORS)]
            frame = overlay_mask_rgb(frame, sam_masks_tmh[t, obj_idx], color)
            draw_box_rgb(frame, sam_boxes_tmk4[t, obj_idx], color, f"sam{obj_idx}")
            if t == int(prompt_frame_idx):
                draw_box_rgb(frame, prompt_boxes[obj_idx], PROMPT_COLOR, f"prompt{obj_idx}")
        frame = annotate_frame(
            frame,
            [
                f"SAM2 Masks + Boxes | native masks/xyxy | t={t}",
                caption,
            ],
        )
        frames.append(frame)
    return np.stack(frames, axis=0)


def render_points_video(
    context_video: torch.Tensor,
    points_k2: np.ndarray,
    caption: str,
    *,
    title: str,
    color_mode: str,
) -> np.ndarray:
    frames = []
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t])
        for q_idx, point in enumerate(points_k2):
            color = QUERY_COLORS[q_idx % len(QUERY_COLORS)] if color_mode == "query" else VGGT_COLOR
            draw_point_rgb(frame, point, color, f"q{q_idx}", radius=5)
        frame = annotate_frame(frame, [f"{title} | native xy | t={t}", caption])
        frames.append(frame)
    return np.stack(frames, axis=0)


def render_tracks_video(
    context_video: torch.Tensor,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    caption: str,
    *,
    title: str,
    color_rgb: tuple[int, int, int],
    label_prefix: str,
) -> np.ndarray:
    frames = []
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t])
        for q_idx in range(tracks_tk2.shape[1]):
            label = f"{label_prefix}{q_idx}"
            if visibility_tk[t, q_idx] < 0.5:
                label += "(inv)"
            draw_point_rgb(frame, tracks_tk2[t, q_idx], color_rgb, label, radius=5)
        frame = annotate_frame(frame, [f"{title} | native xy | t={t}", caption])
        frames.append(frame)
    return np.stack(frames, axis=0)


def render_gt_vs_tracks_video(
    context_video: torch.Tensor,
    gt_boxes_tk4: np.ndarray,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    matched_gt_indices_k: np.ndarray,
    caption: str,
    *,
    title: str,
    track_color_rgb: tuple[int, int, int],
    label_prefix: str,
) -> np.ndarray:
    frames = []
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t])
        for gt_idx in range(gt_boxes_tk4.shape[1]):
            draw_box_rgb(frame, gt_boxes_tk4[t, gt_idx], GT_COLOR, f"gt{gt_idx}", width=2)
        for q_idx in range(tracks_tk2.shape[1]):
            gt_idx = int(matched_gt_indices_k[q_idx])
            label = f"{label_prefix}{q_idx}->gt{gt_idx}"
            if visibility_tk[t, q_idx] < 0.5:
                label += "(inv)"
            draw_point_rgb(frame, tracks_tk2[t, q_idx], track_color_rgb, label, radius=5)
        frame = annotate_frame(frame, [f"{title} | GT xyxy + track xy | t={t}", caption])
        frames.append(frame)
    return np.stack(frames, axis=0)


def save_video_asset(asset_dir: Path, name: str, frames: np.ndarray, fps: int) -> str:
    raw_path = asset_dir / f"{name}.mp4"
    write_mp4(raw_path, frames, fps=fps)
    gif_path = asset_dir / f"{name}.gif"
    write_gif(gif_path, frames, fps=fps)
    return str(gif_path.relative_to(asset_dir.parent))


def build_report(
    *,
    output_dir: Path,
    summary: dict[str, object],
    cards: list[ModuleCard],
) -> Path:
    summary_pretty = html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
    card_html = []
    for card in cards:
        metrics_html = "".join(f"<li>{html.escape(line)}</li>" for line in card.metrics)
        card_html.append(
            f"""
      <article class="card">
        <h2>{html.escape(card.title)}</h2>
        <p class="coord"><b>坐标/顺序:</b> {html.escape(card.coord_note)}</p>
        <p class="coord"><b>映射回 native:</b> {html.escape(card.mapping_note)}</p>
        <p class="verdict"><b>结论:</b> {html.escape(card.verdict)}</p>
        <ul>{metrics_html}</ul>
        <img src="{html.escape(card.media_relpath)}" alt="{html.escape(card.title)}">
      </article>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Coordinate Alignment Audit</title>
  <style>
    :root {{
      --bg: #f5f0e7;
      --card: #fffdf8;
      --text: #1d1d1d;
      --muted: #5b5b5b;
      --line: #d9cfbf;
      --accent: #8f3b2e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Source Sans 3", "Helvetica Neue", Helvetica, Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(143,59,46,0.08), transparent 34%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px 28px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 32px;
      line-height: 1.1;
    }}
    .lead {{
      margin: 0 0 16px;
      max-width: 1100px;
      color: var(--muted);
      font-size: 16px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      margin: 18px 0 26px;
    }}
    .panel {{
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--line);
      padding: 18px;
      backdrop-filter: blur(2px);
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .panel ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .panel li {{
      margin: 0 0 8px;
    }}
    pre {{
      margin: 0;
      background: #f8f5ef;
      border: 1px solid var(--line);
      padding: 14px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.45;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      padding: 16px;
    }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 20px;
    }}
    .card p {{
      margin: 0 0 8px;
      color: var(--muted);
    }}
    .card .verdict {{
      color: var(--text);
    }}
    .card ul {{
      margin: 10px 0 14px;
      padding-left: 18px;
      color: var(--muted);
    }}
    .card img {{
      width: 100%;
      border: 1px solid var(--line);
      background: #000;
      aspect-ratio: 7 / 4;
      object-fit: contain;
    }}
    @media (max-width: 1180px) {{
      .summary {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Coordinate Alignment Audit</h1>
    <p class="lead">这页按真实训练链路跑了一次 sample 前向。所有可视化都统一 overlay 到原始 native context video 上，专门核查各模块的坐标系、xy 顺序，以及它们是否在圈定同一片区域。</p>
    <section class="summary">
      <div class="panel">
        <h2>关键结论</h2>
        <ul>
          <li>训练当前 `track_source=cotracker`，所以进入 object pooler 的 active tracks 已经是 native `(x, y)` 像素坐标。</li>
          <li>VGGT 的 query points / tracks / depth / world_points 都在 `vggt_input_hw` 空间里，回 native 时必须按 `x` 对应宽、`y` 对应高做缩放。</li>
          <li>CoTracker 内部先把 query points 缩到 `cotracker_input_hw`，但 adapter 输出时已经缩回 native，所以训练里下游看到的是 native 坐标。</li>
          <li>GT boxes 在数据集里是归一化 `xyxy`，可视化时转成 native 像素框；用它来判定各模块是不是盯着同一目标区域。</li>
          <li>VGGT geometry 采样前，active tracks 现在会先从 native remap 到 VGGT geometry image，避免不同 resize 语义下的坐标错位。</li>
        </ul>
      </div>
      <div class="panel">
        <h2>运行摘要</h2>
        <pre>{summary_pretty}</pre>
      </div>
    </section>
    <section class="grid">
      {''.join(card_html)}
    </section>
  </div>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/coordinate_alignment_audit",
    )
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    inspector = TrainingFlowInspector(Path(args.config), sample_index=int(args.sample_index))
    describe = inspector.describe()
    artifacts = inspector.collect_artifacts()
    forward_metrics = inspector.run_forward_dry_run()

    context_video = artifacts.context_videos[0].detach().cpu()
    native_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    caption = str(artifacts.captions[0])
    qvis = artifacts.query_prior_visual
    if qvis is None:
        raise RuntimeError("query_prior_visual is unavailable; current config likely disabled SAM2 priors")
    if "context_boxes" not in artifacts.batch:
        raise RuntimeError("context_boxes are required for GT-based coordinate audit")

    gt_boxes_tk4 = np.stack(
        [gt_boxes_frame_px(artifacts.batch["context_boxes"], t, native_hw) for t in range(context_video.shape[1])],
        axis=0,
    )
    prompt_frame_idx = int(qvis.prompt_frame_idx)
    prompt_boxes = [np.asarray(record.prompt_box_xyxy, dtype=np.float32) for record in qvis.object_records]
    sam_masks_tmh = np.stack([record.masks_thw for record in qvis.object_records], axis=1).astype(np.uint8)
    sam_boxes_tmk4 = np.stack([record.boxes_t4 for record in qvis.object_records], axis=1).astype(np.float32)
    query_points_native = np.asarray(qvis.query_points_px, dtype=np.float32)

    vggt_query_native = scale_tracks_xy(
        artifacts.vggt_out.query_points.detach().cpu(),
        tuple(int(v) for v in artifacts.vggt_out.image_hw),
        native_hw,
    )[0].numpy()
    vggt_tracks_native = inspector._vggt_tracks_native()[0].detach().cpu().numpy().astype(np.float32)
    active_tracks_native = inspector._active_tracks_native()[0].detach().cpu().numpy().astype(np.float32)
    vggt_visibility = artifacts.vggt_out.visibility[0].detach().cpu().numpy().astype(np.float32)
    active_visibility = artifacts.active_visibility[0].detach().cpu().numpy().astype(np.float32)
    active_alignment = artifacts.track_alignment
    if active_alignment is None:
        raise RuntimeError("active track alignment is unavailable")
    active_gt_match = active_alignment.matched_gt_indices[0].detach().cpu().numpy().astype(np.int64)
    vggt_alignment = align_tracks_to_boxes(
        torch.from_numpy(vggt_tracks_native).unsqueeze(0).to(dtype=artifacts.batch["context_boxes"].dtype),
        artifacts.batch["context_boxes"],
        image_hw=native_hw,
    )
    vggt_gt_match = vggt_alignment.matched_gt_indices[0].detach().cpu().numpy().astype(np.int64)

    raw_rel = save_video_asset(asset_dir, "01_raw_context", render_raw_context_video(context_video, caption), fps=int(args.fps))
    prompt_rel = save_video_asset(
        asset_dir,
        "02_sam_prompt_boxes",
        render_prompt_boxes_video(context_video, prompt_frame_idx, prompt_boxes, caption),
        fps=int(args.fps),
    )
    sam_rel = save_video_asset(
        asset_dir,
        "03_sam_masks_boxes",
        render_sam_tracks_video(context_video, prompt_frame_idx, prompt_boxes, sam_masks_tmh, sam_boxes_tmk4, caption),
        fps=int(args.fps),
    )
    query_rel = save_video_asset(
        asset_dir,
        "04_query_priors_native",
        render_points_video(context_video, query_points_native, caption, title="Query Priors From SAM2", color_mode="query"),
        fps=int(args.fps),
    )
    vggt_query_rel = save_video_asset(
        asset_dir,
        "05_vggt_query_back_to_native",
        render_points_video(context_video, vggt_query_native, caption, title="VGGT Query Points Back To Native", color_mode="vggt"),
        fps=int(args.fps),
    )
    vggt_tracks_rel = save_video_asset(
        asset_dir,
        "06_vggt_tracks_native",
        render_tracks_video(
            context_video,
            vggt_tracks_native,
            vggt_visibility,
            caption,
            title="VGGT Tracks Rescaled To Native",
            color_rgb=VGGT_COLOR,
            label_prefix="v",
        ),
        fps=int(args.fps),
    )
    active_tracks_rel = save_video_asset(
        asset_dir,
        "07_active_tracks_native",
        render_tracks_video(
            context_video,
            active_tracks_native,
            active_visibility,
            caption,
            title="Active Tracks Used By Trainer",
            color_rgb=ACTIVE_COLOR,
            label_prefix="a",
        ),
        fps=int(args.fps),
    )
    active_gt_rel = save_video_asset(
        asset_dir,
        "08_gt_vs_active_tracks",
        render_gt_vs_tracks_video(
            context_video,
            gt_boxes_tk4,
            active_tracks_native,
            active_visibility,
            active_gt_match,
            caption,
            title="GT Boxes vs Active Tracks",
            track_color_rgb=ACTIVE_COLOR,
            label_prefix="a",
        ),
        fps=int(args.fps),
    )
    vggt_gt_rel = save_video_asset(
        asset_dir,
        "09_gt_vs_vggt_tracks",
        render_gt_vs_tracks_video(
            context_video,
            gt_boxes_tk4,
            vggt_tracks_native,
            vggt_visibility,
            vggt_gt_match,
            caption,
            title="GT Boxes vs VGGT Tracks",
            track_color_rgb=VGGT_COLOR,
            label_prefix="v",
        ),
        fps=int(args.fps),
    )

    prompt_metrics = prompt_box_metrics(prompt_boxes, gt_boxes_tk4[prompt_frame_idx])
    sam_metrics = sam_track_box_metrics(sam_boxes_tmk4, gt_boxes_tk4)
    query_metrics = point_region_metrics(query_points_native, gt_boxes_tk4[0], sam_boxes_px_m4=sam_boxes_tmk4[0])
    vggt_query_diff = np.linalg.norm(vggt_query_native - query_points_native, axis=-1)
    vggt_query_metrics = point_region_metrics(vggt_query_native, gt_boxes_tk4[0], sam_boxes_px_m4=sam_boxes_tmk4[0])

    active_track_metrics = track_alignment_metrics(active_tracks_native, gt_boxes_tk4, active_gt_match)
    vggt_track_metrics = track_alignment_metrics(vggt_tracks_native, gt_boxes_tk4, vggt_gt_match)
    vggt_active_dist = np.linalg.norm(vggt_tracks_native - active_tracks_native, axis=-1)

    cards = [
        ModuleCard(
            slug="raw-context",
            title="1. Raw Context Video",
            coord_note=f"原始 native 图像，尺寸 HxW={native_hw[0]}x{native_hw[1]}。",
            mapping_note="无变换，作为所有 overlay 的参照底图。",
            verdict="这是参考底图，不参与区域一致性判定。",
            metrics=[
                f"context frames = {context_video.shape[1]}",
                f"caption = {caption}",
            ],
            media_relpath=raw_rel,
        ),
        ModuleCard(
            slug="sam-prompt",
            title="2. SAM2 Prompt Boxes",
            coord_note="prompt box 使用 native 像素 xyxy，顺序是 (x0, y0, x1, y1)。",
            mapping_note="无需 remap，直接画回 native prompt frame。",
            verdict=(
                "prompt 框已经大体落在目标上。"
                if prompt_metrics["mean_best_iou"] == prompt_metrics["mean_best_iou"] and prompt_metrics["mean_best_iou"] >= 0.35
                else "prompt 框和 GT 有明显偏差，需要回看 prompt 生成逻辑。"
            ),
            metrics=[
                f"prompt_frame = {prompt_frame_idx}",
                f"mean best IoU vs GT@prompt = {fmt_float(prompt_metrics['mean_best_iou'])}",
                f"max best IoU vs GT@prompt = {fmt_float(prompt_metrics['max_best_iou'])}",
            ],
            media_relpath=prompt_rel,
        ),
        ModuleCard(
            slug="sam-track",
            title="3. SAM2 Masks And Boxes",
            coord_note="SAM2 mask 是 native HxW 二值图；SAM2 box 是 native 像素 xyxy。",
            mapping_note="无需 remap，直接 overlay 到原视频每一帧。",
            verdict=(
                "SAM2 跟踪盒整体和 GT 属于同一块区域。"
                if sam_metrics["mean_best_iou"] == sam_metrics["mean_best_iou"] and sam_metrics["mean_best_iou"] >= 0.35
                else "SAM2 跟踪盒和 GT 的重叠偏低，需要检查检测/传播。"
            ),
            metrics=[
                f"tracked objects = {sam_boxes_tmk4.shape[1]}",
                f"mean best IoU vs GT = {fmt_float(sam_metrics['mean_best_iou'])}",
                f"max best IoU vs GT = {fmt_float(sam_metrics['max_best_iou'])}",
            ],
            media_relpath=sam_rel,
        ),
        ModuleCard(
            slug="query-priors",
            title="4. Native Query Priors",
            coord_note="query priors 存的是 native 像素点，顺序是 (x, y)。",
            mapping_note="无需 remap；这些点就是送进 VGGT / CoTracker 之前的公共对象条件。",
            verdict=(
                "采样点和 SAM2 目标区域一致；是否覆盖 GT 取决于 SAM2 本身框住的目标。"
                if query_metrics["inside_sam_box_rate"] == query_metrics["inside_sam_box_rate"] and query_metrics["inside_sam_box_rate"] >= 0.95
                else "采样点和 SAM2 目标区域都不够一致，需要先检查 query prior 来源。"
            ),
            metrics=[
                f"query count = {query_points_native.shape[0]}",
                f"inside SAM box@frame0 = {fmt_float(query_metrics['inside_sam_box_rate'] * 100.0, '%')}",
                f"inside any GT box@frame0 = {fmt_float(query_metrics['inside_any_gt_rate'] * 100.0, '%')}",
                f"mean min dist to any GT center@frame0 = {fmt_float(query_metrics['mean_min_gt_center_dist_px'], ' px')}",
            ],
            media_relpath=query_rel,
        ),
        ModuleCard(
            slug="vggt-query",
            title="5. VGGT Query Points Back To Native",
            coord_note=f"VGGT 内部 query points 在 resized 空间 HxW={artifacts.vggt_out.image_hw[0]}x{artifacts.vggt_out.image_hw[1]}，顺序仍是 (x, y)。",
            mapping_note="这里先把 native priors 缩进 VGGT，再把 `vggt_out.query_points` 缩回 native 做核查。",
            verdict=(
                "native -> VGGT -> native 的 query 变换基本一致。"
                if float(vggt_query_diff.mean()) <= 1.5
                else "VGGT query remap 存在明显偏移。"
            ),
            metrics=[
                f"mean native-vs-back diff = {fmt_float(float(vggt_query_diff.mean()), ' px')}",
                f"max native-vs-back diff = {fmt_float(float(vggt_query_diff.max()), ' px')}",
                f"inside SAM box@frame0 = {fmt_float(vggt_query_metrics['inside_sam_box_rate'] * 100.0, '%')}",
                f"inside any GT box@frame0 = {fmt_float(vggt_query_metrics['inside_any_gt_rate'] * 100.0, '%')}",
            ],
            media_relpath=vggt_query_rel,
        ),
        ModuleCard(
            slug="vggt-tracks",
            title="6. VGGT Tracks Rescaled To Native",
            coord_note="VGGT tracks 原始输出在 VGGT resized 图上，存的是 (x, y) 像素点。",
            mapping_note="可视化前按 `x*W_native/W_vggt`, `y*H_native/H_vggt` 缩回 native。",
            verdict=verdict_from_inside_rate(
                vggt_track_metrics["inside_rate"],
                vggt_track_metrics["mean_center_dist_px"],
            ),
            metrics=[
                f"inside matched GT = {fmt_float(vggt_track_metrics['inside_rate'] * 100.0, '%')}",
                f"mean dist to matched GT center = {fmt_float(vggt_track_metrics['mean_center_dist_px'], ' px')}",
                f"max dist to matched GT center = {fmt_float(vggt_track_metrics['max_center_dist_px'], ' px')}",
            ],
            media_relpath=vggt_tracks_rel,
        ),
        ModuleCard(
            slug="active-tracks",
            title="7. Active Tracks Used By Trainer",
            coord_note="当前 `track_source=cotracker`，所以 active tracks 已经是 native 像素 (x, y)。",
            mapping_note=f"CoTracker 内部只在输入阶段缩到 {tuple(int(v) for v in artifacts.cotracker_out.input_hw)}，adapter 输出时已缩回 native。",
            verdict=verdict_from_inside_rate(
                active_track_metrics["inside_rate"],
                active_track_metrics["mean_center_dist_px"],
            ),
            metrics=[
                f"inside matched GT = {fmt_float(active_track_metrics['inside_rate'] * 100.0, '%')}",
                f"mean dist to matched GT center = {fmt_float(active_track_metrics['mean_center_dist_px'], ' px')}",
                f"max dist to matched GT center = {fmt_float(active_track_metrics['max_center_dist_px'], ' px')}",
            ],
            media_relpath=active_tracks_rel,
        ),
        ModuleCard(
            slug="gt-vs-active",
            title="8. GT Boxes Vs Active Tracks",
            coord_note="GT boxes 在数据里是归一化 xyxy；这里先转成 native 像素框，再和 active native tracks 对照。",
            mapping_note="这一步对应训练里的 track supervision，可直接看 active tracks 是否圈在同一个 GT 区域内。",
            verdict=verdict_from_inside_rate(
                active_track_metrics["inside_rate"],
                active_track_metrics["mean_center_dist_px"],
                strong_px=20.0,
                weak_px=38.0,
            ),
            metrics=[
                f"matched GT ids = {active_gt_match.tolist()}",
                f"track_box_l1_loss = {fmt_float(float(forward_metrics['train/track_box_loss']))}",
                f"track_iou_loss = {fmt_float(float(forward_metrics['train/track_iou_loss']))}",
            ],
            media_relpath=active_gt_rel,
        ),
        ModuleCard(
            slug="gt-vs-vggt",
            title="9. GT Boxes Vs VGGT Tracks",
            coord_note="同样把 GT 转成 native，再和缩回 native 的 VGGT tracks 对照。",
            mapping_note="这能直接看出 VGGT 和 active tracks 的空间差异是不是来自坐标变换，还是来自跟踪本身。",
            verdict=(
                "VGGT 与 CoTracker 大体盯着同一块区域。"
                if float(vggt_active_dist.mean()) <= 32.0
                else "VGGT 与 CoTracker 的空间落点差异明显。"
            ),
            metrics=[
                f"VGGT matched GT ids = {vggt_gt_match.tolist()}",
                f"mean VGGT-vs-active track dist = {fmt_float(float(vggt_active_dist.mean()), ' px')}",
                f"max VGGT-vs-active track dist = {fmt_float(float(vggt_active_dist.max()), ' px')}",
            ],
            media_relpath=vggt_gt_rel,
        ),
    ]

    summary = {
        "config_path": str(Path(args.config)),
        "sample_index": int(args.sample_index),
        "track_source": describe["track_source"],
        "caption": caption,
        "video_path": describe["video_path"],
        "native_hw": list(native_hw),
        "vggt_hw": [int(v) for v in artifacts.vggt_out.image_hw],
        "cotracker_input_hw": [int(v) for v in artifacts.cotracker_out.input_hw],
        "prompt_frame_idx": prompt_frame_idx,
        "sam_objects": int(sam_boxes_tmk4.shape[1]),
        "num_queries": int(query_points_native.shape[0]),
        "forward_loss": float(forward_metrics["loss"]),
        "track_box_l1_loss": float(forward_metrics["train/track_box_loss"]),
        "track_iou_loss": float(forward_metrics["train/track_iou_loss"]),
        "query_native_vs_vggt_back_mean_px": float(vggt_query_diff.mean()),
        "query_native_vs_vggt_back_max_px": float(vggt_query_diff.max()),
        "active_inside_rate": float(active_track_metrics["inside_rate"]),
        "active_mean_center_dist_px": float(active_track_metrics["mean_center_dist_px"]),
        "vggt_inside_rate": float(vggt_track_metrics["inside_rate"]),
        "vggt_mean_center_dist_px": float(vggt_track_metrics["mean_center_dist_px"]),
        "vggt_vs_active_mean_px": float(vggt_active_dist.mean()),
        "vggt_vs_active_max_px": float(vggt_active_dist.max()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path = build_report(output_dir=output_dir, summary=summary, cards=cards)
    print(f"report: {html_path}")


if __name__ == "__main__":
    main()
