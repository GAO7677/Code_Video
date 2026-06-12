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
from PIL import Image, ImageDraw

from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker, build_motion_prompt_box
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes


GT_PALETTE = ["#d62828", "#f77f00", "#fcbf49", "#2a9d8f", "#277da1", "#6a4c93"]
QUERY_PALETTE = ["#00b4d8", "#0077b6", "#8338ec", "#3a86ff", "#ff006e", "#fb5607", "#2ec4b6", "#8ac926"]
SAM_PROMPT_COLOR = "#ff8c00"
SAM_TRACK_COLOR = "#2ca25f"


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def scale_points_xy(points_k2: np.ndarray, *, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(points_k2, dtype=np.float32).copy()
    if out.size == 0:
        return out.reshape(-1, 2)
    scale_x = float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    scale_y = float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    out[..., 0] *= scale_x
    out[..., 1] *= scale_y
    return out


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
) -> None:
    scale_x = width / max(float(image_hw[1]), 1.0)
    scale_y = height / max(float(image_hw[0]), 1.0)
    for query_idx, point in enumerate(tracks_xy_k2.tolist()):
        x, y = float(point[0]) * scale_x, float(point[1]) * scale_y
        color = QUERY_PALETTE[query_idx % len(QUERY_PALETTE)]
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        gt_idx = int(matched_gt_idx_k[query_idx].item())
        label = f"q{query_idx}->gt{gt_idx}"
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
    <p><b>Context frames:</b> {result['context_frame_indices']}</p>
    <p><b>Matched GT indices:</b> {result['matched_gt_indices']}</p>
    <p><b>Unmatched GT indices:</b> {result['unmatched_gt_indices']}</p>
    <p><b>Coordinate trace:</b> {result['coord_trace']['summary']}</p>
    <p><b>Shapes:</b></p>
    <pre>{json.dumps(result['shapes'], indent=2, ensure_ascii=False)}</pre>
    <p><b>Coordinate stats:</b></p>
    <pre>{json.dumps(result['coord_trace'], indent=2, ensure_ascii=False)}</pre>
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
  <title>VGGT Query Points Overlay</title>
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
  <h1>VGGT Query Points Overlay</h1>
  <p>同一个 case 现在按来源拆成多个独立视频。黑色圆点是喂给 VGGT 的 query points，彩色圆点是 VGGT 跟踪结果，彩色框是数据集 GT box。每个视频下方都会标注来源，方便单独检查是哪一路信号出了问题。</p>
  {''.join(blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def evaluate_sample(sample: dict, adapter: VGGTTrackAdapter, device: torch.device, output_dir: Path) -> dict:
    context_video = sample["context_video"].unsqueeze(0).to(device)
    context_boxes = sample["context_boxes"].unsqueeze(0).to(device)

    frames_bthwc = context_video.permute(0, 2, 3, 4, 1).float()
    frames_bthwc = (frames_bthwc + 1.0) / 2.0
    with torch.no_grad():
        vggt_out = adapter(frames_bthwc)

    tracks = vggt_out.tracks
    track_image_hw = vggt_out.image_hw
    scale_x = float(context_video.shape[-1]) / float(track_image_hw[1])
    scale_y = float(context_video.shape[-2]) / float(track_image_hw[0])
    tracks_native = tracks.clone()
    tracks_native[..., 0] *= scale_x
    tracks_native[..., 1] *= scale_y

    native_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    query_points_prior_px = vggt_out.query_points[0].detach().cpu().numpy().astype(np.float32)
    query_points_vggt_input_px = vggt_out.query_points[0].detach().cpu().numpy().astype(np.float32)
    query_points_roundtrip_px = scale_points_xy(
        query_points_vggt_input_px,
        src_hw=track_image_hw,
        dst_hw=native_hw,
    )
    query_roundtrip_abs_err = np.abs(query_points_roundtrip_px - query_points_prior_px) if query_points_prior_px.size > 0 else np.zeros((0, 2), dtype=np.float32)
    query_roundtrip_max_abs_err_px = float(query_roundtrip_abs_err.max()) if query_roundtrip_abs_err.size > 0 else 0.0
    query_roundtrip_mean_abs_err_px = float(query_roundtrip_abs_err.mean()) if query_roundtrip_abs_err.size > 0 else 0.0
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

    video_buffers: dict[str, list[np.ndarray]] = {
        "raw_context": [],
        "gt_only": [],
        "vggt_query_only": [],
        "vggt_tracks_only": [],
        "sam_prompt_only": [],
        "sam_mask_only": [],
        "sam_track_only": [],
    }
    context_frames = sample["context_video"].permute(1, 0, 2, 3)
    for t in range(context_frames.shape[0]):
        raw_img = Image.fromarray(tensor_frame_to_uint8_hwc(context_frames[t]))
        video_buffers["raw_context"].append(np.array(raw_img))

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
            query_points_k2=vggt_out.query_points[0].detach().cpu().numpy(),
            image_hw=track_image_hw,
            show_query=True,
        )
        video_buffers["vggt_query_only"].append(np.array(query_img))

        track_img = draw_overlay_frame(
            frame_chw=context_frames[t],
            gt_boxes_k4=sample["context_boxes"][t],
            tracks_xy_k2=tracks_native[0, t].detach().cpu(),
            vis_k=vggt_out.visibility[0, t].detach().cpu(),
            matched_gt_idx_k=alignment.matched_gt_indices[0].detach().cpu(),
            image_hw=track_image_hw,
            show_tracks=True,
        )
        video_buffers["vggt_tracks_only"].append(np.array(track_img))

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
        ("raw_context", "Raw Context Video", "原始 context 帧"),
        ("gt_only", "GT Boxes", "数据集 GT boxes"),
        ("vggt_query_only", "VGGT Query Points", "VGGT 输入 query points"),
        ("vggt_tracks_only", "VGGT Tracked Points", "VGGT 输出 tracked points"),
        ("sam_prompt_only", "SAM2 Prompt Box", "SAM2 motion prompt box"),
        ("sam_mask_only", "SAM2 Mask", "SAM2 输出 mask"),
        ("sam_track_only", "SAM2 Track Box", "SAM2 输出 tracked box"),
    ]
    browser_videos = []
    for key, title, source in video_specs:
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
            "summary": "SAM2/object priors are sampled in native pixels. VGGT receives those points after resizing to its fixed input size, returns tracks in that resized pixel space, and we scale the tracks back to native pixels for overlay/eval.",
            "native_hw": list(native_hw),
            "vggt_input_hw": [int(track_image_hw[0]), int(track_image_hw[1])],
            "query_points_prior_px": point_stats(query_points_prior_px),
            "query_points_vggt_input_px": point_stats(query_points_vggt_input_px),
            "query_points_roundtrip_px": point_stats(query_points_roundtrip_px),
            "query_roundtrip_max_abs_err_px": query_roundtrip_max_abs_err_px,
            "query_roundtrip_mean_abs_err_px": query_roundtrip_mean_abs_err_px,
            "tracks_native_scale": [scale_x, scale_y],
            "tracks_vggt_input_px": point_stats(tracks[0].detach().cpu().numpy().astype(np.float32).reshape(-1, 2)),
            "tracks_native_px": point_stats(tracks_native_px.reshape(-1, 2)),
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

    output_dir = Path(args.output_dir)
    results = []
    for idx in range(args.start_index, min(len(dataset), args.start_index + args.num_cases)):
        sample = dataset[idx]
        sample["_output_dir"] = str(output_dir / "assets")
        sample["_case_name"] = f"case_{idx:03d}"
        sample["_fps"] = int(data_cfg.get("fps", 8))
        results.append(evaluate_sample(sample, adapter, device, output_dir / "assets"))

    html_path = build_report(results, output_dir)
    print(f"eval report: {html_path}")

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
