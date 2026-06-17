from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import math
import shutil
import socketserver
import subprocess
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
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes


GT_PALETTE = ["#d62828", "#f77f00", "#fcbf49", "#2a9d8f", "#277da1", "#6a4c93"]
QUERY_PALETTE = ["#00b4d8", "#0077b6", "#8338ec", "#3a86ff", "#ff006e", "#fb5607", "#2ec4b6", "#8ac926"]
COTRACKER_PALETTE = ["#2b8a3e", "#40916c", "#52b788", "#74c69d", "#1b4332", "#95d5b2", "#007f5f", "#55a630"]
SAM_PROMPT_COLOR = "#ff8c00"
SAM_TRACK_COLOR = "#2ca25f"


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


def build_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for idx, result in enumerate(results):
        video_cards = []
        for video in result["videos"]:
            video_cards.append(
                f"""
    <figure class="video-card">
      <video controls preload="none" playsinline src="{video['path']}"></video>
      <figcaption><b>{video['title']}</b><br>来源: {video['source']}</figcaption>
    </figure>
"""
            )
        blocks.append(
            f"""
  <section class="case">
    <h2>Case {idx}</h2>
    <p><b>Caption:</b> {result['caption']}</p>
    <p><b>Pipeline:</b> 从 context video 开始，依次看 SAM2 的 prompt / mask / track box，随后看 native query priors、VGGT 输入分辨率下的 priors overlay，最后对比 VGGT 和 CoTracker 的 track 输出。</p>
    <div class="video-grid">
      {''.join(video_cards)}
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
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .video-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 16px; }}
    .video-card {{ margin: 0; background: #fff; border: 1px solid #ddd; padding: 12px; }}
    video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Track Pipeline Viewer</h1>
  <p>每个 case 都使用同一段 context video 和同一批由 SAM2 采样得到的 query priors。页面按流程展示从视频输入到最终得到 track 的中间产物，并在最后并排给出 VGGT 和 CoTracker 两种 track_source 的结果。</p>
  {''.join(blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


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

    video_buffers: dict[str, list[np.ndarray]] = {
        "raw_context": [],
        "gt_only": [],
        "vggt_query_only": [],
        "vggt_tracks_only": [],
        "cotracker_tracks_only": [],
        "sam_prompt_only": [],
        "sam_mask_only": [],
        "sam_track_only": [],
        "raw_context_vggt_input": [],
        "sam2_priors_vggt_input": [],
    }
    context_frames = sample["context_video"].permute(1, 0, 2, 3)
    for t in range(context_frames.shape[0]):
        raw_img = Image.fromarray(tensor_frame_to_uint8_hwc(context_frames[t]))
        video_buffers["raw_context"].append(np.array(raw_img))

        resized_frame = resize_frame_chw(context_frames[t], track_image_hw)
        resized_raw_img = Image.fromarray(tensor_frame_to_uint8_hwc(resized_frame))
        video_buffers["raw_context_vggt_input"].append(np.array(resized_raw_img))

        gt_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            image_hw=track_image_hw,
            show_gt=True,
        )
        video_buffers["gt_only"].append(np.array(gt_img))

        query_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            query_points_k2=query_points_prior_px if t == 0 else None,
            image_hw=native_hw,
            show_query=(t == 0),
        )
        video_buffers["vggt_query_only"].append(np.array(query_img))

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

        track_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            tracks_xy_k2=tracks_native[0, t].detach().cpu(),
            vis_k=vggt_out.visibility[0, t].detach().cpu(),
            matched_gt_idx_k=alignment.matched_gt_indices[0].detach().cpu(),
            image_hw=track_image_hw,
            show_tracks=True,
            track_label_prefix="v",
            track_palette=QUERY_PALETTE,
        )
        video_buffers["vggt_tracks_only"].append(np.array(track_img))

        if cotracker_out is not None and cotracker_alignment is not None:
            cotrack_img = draw_overlay_frame(
                frame_chw=context_frames[t],
                gt_boxes_k4=sample["context_boxes"][t],
                tracks_xy_k2=cotracker_out.tracks[0, t].detach().cpu(),
                vis_k=cotracker_out.visibility[0, t].detach().cpu(),
                matched_gt_idx_k=cotracker_alignment.matched_gt_indices[0].detach().cpu(),
                image_hw=native_hw,
                show_tracks=True,
                track_label_prefix="c",
                track_palette=COTRACKER_PALETTE,
            )
            video_buffers["cotracker_tracks_only"].append(np.array(cotrack_img))

        sam_prompt_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            sam_prompt_box_xyxy=sam_out.prompt_box_xyxy if t == prompt_frame_idx else None,
            image_hw=(context_frames.shape[-2], context_frames.shape[-1]),
            show_sam_prompt=(t == prompt_frame_idx),
        )
        video_buffers["sam_prompt_only"].append(np.array(sam_prompt_img))

        sam_mask_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            sam_mask_hw=sam_out.masks_thw[t],
            image_hw=(context_frames.shape[-2], context_frames.shape[-1]),
            show_sam_mask=True,
        )
        video_buffers["sam_mask_only"].append(np.array(sam_mask_img))

        sam_track_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            sam_track_box_xyxy=sam_out.boxes_t4[t],
            image_hw=(context_frames.shape[-2], context_frames.shape[-1]),
            show_sam_track=True,
        )
        video_buffers["sam_track_only"].append(np.array(sam_track_img))

    video_specs = [
        ("raw_context", "1. Raw Context Video", "原始 context video"),
        ("sam_prompt_only", "2. SAM2 Prompt Box", "SAM2 的 prompt box"),
        ("sam_mask_only", "3. SAM2 Mask", "SAM2 传播得到的 mask"),
        ("sam_track_only", "4. SAM2 Track Box", "SAM2 传播得到的 track box"),
        ("vggt_query_only", "5. Native Query Points", "native 分辨率下采样得到的 query priors"),
        ("raw_context_vggt_input", "6. Raw Context Video (VGGT Input)", "缩放到 VGGT 实际输入分辨率的 context"),
        ("sam2_priors_vggt_input", "7. SAM2 Priors Overlay (VGGT Input)", "VGGT 输入分辨率下的 priors overlay"),
        ("vggt_tracks_only", "8. VGGT Tracked Points", "track_source=vggt"),
        ("cotracker_tracks_only", "9. CoTracker Tracked Points", "track_source=cotracker"),
    ]
    browser_videos = []
    for key, title, source in video_specs:
        if len(video_buffers[key]) == 0:
            continue
        raw_path = output_dir / f"{Path(sample['video_path']).stem}__{key}.mp4"
        write_mp4(raw_path, np.stack(video_buffers[key], axis=0), fps=int(sample.get("_fps", 8)))
        browser_path = ensure_browser_video(raw_path)
        browser_videos.append(
            {
                "key": key,
                "title": title,
                "source": source,
                "path": str(browser_path.relative_to(output_dir.parent)),
            }
        )

    return {
        "caption": sample["caption"],
        "video_path": sample["video_path"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "matched_gt_indices": matched_gt_indices,
        "unmatched_gt_indices": unmatched_gt_indices,
        "coord_trace": {
            "summary": "SAM2 先在 native context 上生成 mask/track box，再从 frame0 的 SAM2 mask/box 采样 query points。VGGT 接收到的是这些 prior 缩放到固定 vggt_input_hw 后的像素坐标，输出轨迹也在这个坐标系里，最后再缩回 native 像素做对齐和评估。",
            "native_hw": list(native_hw),
            "vggt_input_hw": [int(track_image_hw[0]), int(track_image_hw[1])],
            "sam_prior_source": sam_prior_source,
            "query_points_prior_px": point_stats(query_points_prior_px),
            "query_points_vggt_input_px": point_stats(query_points_vggt_input_px),
            "query_points_roundtrip_px": point_stats(query_points_roundtrip_px),
            "query_roundtrip_max_abs_err_px": query_roundtrip_max_abs_err_px,
            "query_roundtrip_mean_abs_err_px": query_roundtrip_mean_abs_err_px,
            "tracks_native_scale": [scale_x, scale_y],
            "tracks_vggt_input_px": point_stats(tracks[0].detach().cpu().numpy().astype(np.float32).reshape(-1, 2)),
            "tracks_native_px": point_stats(tracks_native_px.reshape(-1, 2)),
            "cotracker_input_hw": list(cotracker_out.input_hw) if cotracker_out is not None else None,
            "cotracker_tracks_native_px": point_stats(cotracker_tracks_native_px.reshape(-1, 2)) if cotracker_tracks_native_px.size > 0 else point_stats(cotracker_tracks_native_px),
            "cotracker_mean_center_l1_px": cotracker_mean_center_l1_px,
            "cotracker_valid_track_points": cotracker_valid_track_points,
        },
        "shapes": {
            "context_video": list(context_video.shape),
            "context_boxes": list(context_boxes.shape),
            "sam_prompt_box_xyxy": list(np.asarray(sam_out.prompt_box_xyxy).shape),
            "sam_masks_thw": list(np.asarray(sam_out.masks_thw).shape),
            "sam_boxes_t4": list(np.asarray(sam_out.boxes_t4).shape),
            "sam_motion_box_xyxy": list(np.asarray(motion_prompt_box_xyxy).shape),
            "sam_query_points": list(query_points_prior_px.shape),
            "vggt_query_points": list(vggt_out.query_points.shape),
            "vggt_tracks": list(vggt_out.tracks.shape),
            "vggt_visibility": list(vggt_out.visibility.shape),
            "vggt_confidence": list(vggt_out.confidence.shape),
            "cotracker_query_points": list(cotracker_out.query_points.shape) if cotracker_out is not None else None,
            "cotracker_tracks": list(cotracker_out.tracks.shape) if cotracker_out is not None else None,
            "cotracker_visibility": list(cotracker_out.visibility.shape) if cotracker_out is not None else None,
            "cotracker_confidence": list(cotracker_out.confidence.shape) if cotracker_out is not None else None,
        },
        "videos": browser_videos,
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
    for idx in range(args.start_index, min(len(dataset), args.start_index + args.num_cases)):
        sample = dataset[idx]
        sample["_output_dir"] = str(output_dir / "assets")
        sample["_case_name"] = f"case_{idx:03d}"
        sample["_fps"] = int(data_cfg.get("fps", 8))
        results.append(evaluate_sample(sample, adapter, cotracker_adapter, device, output_dir / "assets"))

    html_path = build_report(results, output_dir)
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
