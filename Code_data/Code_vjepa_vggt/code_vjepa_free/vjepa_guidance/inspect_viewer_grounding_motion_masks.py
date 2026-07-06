#!/usr/bin/env python3
"""
Run command example:

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/inspect_viewer_grounding_motion_masks.py \
  --video /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500/baseline/step-002500/0613pybullet_sample_001460_w002.mp4 \
  --caption "f5 sample 001460 industrial rigid body simulation sphere box" \
  --out-dir /data/gaoya/agent-data/outputs/viewer_grounding_motion_masks/0613pybullet_sample_001460_w002 \
  --motion-mask-mode both \
  --serve
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from code_vjepa_free.vjepa_guidance.analyze_local_vjepa import load_video_rgb
from code_vjepa_free.vjepa_guidance.mask_video_viz import (
    ensure_browser_video,
    render_background_overlay_video,
    render_binary_mask_video,
    render_motion_overlay_video,
    write_mp4_h264,
)
from code_vjepa_free.vjepa_guidance.motion_masks import MotionMaskResult, compute_all_motion_masks
from code_vjepa_free.vjepa_guidance.viewer_grounding_motion_masks import (
    compute_viewer_grounding_object_motion_masks,
    summarize_debug_payload,
)
from code_vjepa_vggt.infer_context_video_wan import _draw_box_rgb, _draw_point_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect viewer-grounding-derived motion masks on a single video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--caption", type=str, default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grounding-device", type=str, default="cuda:0")
    parser.add_argument("--segment-len", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--points-per-object", type=int, default=8)
    parser.add_argument("--proposal-source", type=str, default="gdino_only")
    parser.add_argument("--motion-score-ratio", type=float, default=0.15)
    parser.add_argument("--text-prompt", type=str, default="box . cube . block . cylinder . capsule . sphere . ball .")
    parser.add_argument("--extra-prompt-terms", type=str, default="")
    parser.add_argument("--include-caption-terms", action="store_true")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.15)
    parser.add_argument("--prompt-frame-mode", type=str, default="first")
    parser.add_argument("--track-dedupe-iou-threshold", type=float, default=0.75)
    parser.add_argument("--container-suppress-ratio-threshold", type=float, default=0.95)
    parser.add_argument("--container-suppress-min-contained", type=int, default=2)
    parser.add_argument("--container-suppress-min-area-ratio", type=float, default=1.5)
    parser.add_argument("--container-suppress-small-iou-threshold", type=float, default=0.7)
    parser.add_argument("--motion-dilate-px", type=int, default=10)
    parser.add_argument("--support-dilate-px", type=int, default=20)
    parser.add_argument(
        "--motion-mask-mode",
        choices=["per_frame", "temporal_union", "both"],
        default="per_frame",
    )
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


def _overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    alpha_map = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    base = frame_rgb.astype(np.float32)
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1.0 - alpha * alpha_map) + tint * (alpha * alpha_map)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _overlay_background_region(frame_rgb: np.ndarray, motion_mask: np.ndarray, *, color: tuple[int, int, int] = (70, 140, 245)) -> np.ndarray:
    motion = np.clip(motion_mask.astype(np.float32), 0.0, 1.0)[..., None]
    background = 1.0 - motion
    base = frame_rgb.astype(np.float32)
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1.0 - 0.40 * background) + tint * (0.40 * background)
    out = out * (0.85 + 0.15 * motion)
    return np.clip(out, 0, 255).astype(np.uint8)


def _binary_tile(mask: np.ndarray, *, fg_color: tuple[int, int, int], bg_color: tuple[int, int, int] = (244, 241, 234)) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[...] = np.asarray(bg_color, dtype=np.uint8)
    out[binary > 0] = np.asarray(fg_color, dtype=np.uint8)
    return out


def _find_font(size: int) -> ImageFont.FreeTypeFont | None:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return None


def _draw_viewer_boxes_and_queries(
    frame_rgb: np.ndarray,
    *,
    context_boxes_norm: np.ndarray,
    grouped_queries_px: np.ndarray,
    object_valid_mask: np.ndarray,
    frame_idx: int,
) -> np.ndarray:
    canvas = frame_rgb.copy()
    height, width = frame_rgb.shape[:2]
    colors = [
        (230, 57, 70),
        (29, 78, 216),
        (46, 125, 50),
        (245, 158, 11),
    ]
    for object_idx in range(int(min(len(object_valid_mask), grouped_queries_px.shape[0]))):
        if float(object_valid_mask[object_idx]) <= 0.5:
            continue
        color = colors[object_idx % len(colors)]
        if context_boxes_norm.ndim == 3 and frame_idx < context_boxes_norm.shape[0]:
            box_norm = context_boxes_norm[frame_idx, object_idx]
            box_px = np.asarray(
                [
                    box_norm[0] * width,
                    box_norm[1] * height,
                    box_norm[2] * width,
                    box_norm[3] * height,
                ],
                dtype=np.float32,
            )
            _draw_box_rgb(canvas, box_px, color, f"obj{object_idx}")
        for point_idx in range(int(grouped_queries_px.shape[1])):
            _draw_point_rgb(canvas, grouped_queries_px[object_idx, point_idx].astype(np.float32), color, "", radius=4)
    return canvas


def build_panel(
    video_thwc_u8: np.ndarray,
    legacy_masks: dict[str, MotionMaskResult],
    viewer_masks: dict[str, MotionMaskResult],
    viewer_debug: dict,
    out_path: Path,
) -> None:
    frame_idx = min(video_thwc_u8.shape[0] - 1, max(0, int(viewer_debug["prompt_frame_idx"])))
    frame_rgb = video_thwc_u8[frame_idx]
    motion_color = (70, 220, 120)
    background_color = (70, 140, 245)

    legacy_motion = legacy_masks["background_residual"].mask[frame_idx]
    viewer_motion = viewer_masks["viewer_guidance_support"].mask[frame_idx]

    boxes_queries = _draw_viewer_boxes_and_queries(
        frame_rgb,
        context_boxes_norm=np.asarray(viewer_debug["context_boxes_norm"], dtype=np.float32),
        grouped_queries_px=np.asarray(viewer_debug["grouped_queries_px"], dtype=np.float32),
        object_valid_mask=np.asarray(viewer_debug["object_valid_mask"], dtype=np.float32),
        frame_idx=frame_idx,
    )

    tiles = [
        Image.fromarray(frame_rgb),
        Image.fromarray(_binary_tile(legacy_motion, fg_color=motion_color)),
        Image.fromarray(_overlay_mask(frame_rgb, legacy_motion, color=motion_color)),
        Image.fromarray(_overlay_background_region(frame_rgb, legacy_motion, color=background_color)),
        Image.fromarray(boxes_queries),
        Image.fromarray(_binary_tile(viewer_motion, fg_color=motion_color)),
        Image.fromarray(_overlay_mask(frame_rgb, viewer_motion, color=motion_color)),
        Image.fromarray(_overlay_background_region(frame_rgb, viewer_motion, color=background_color)),
    ]
    labels = [
        "Legacy | Raw Frame",
        "Legacy | Motion Region Binary",
        "Legacy | Motion Region Overlay",
        "Legacy | Background Region Overlay",
        "Viewer | Boxes + Query Points",
        "Viewer | Motion Region Binary",
        "Viewer | Motion Region Overlay",
        "Viewer | Background Region Overlay",
    ]

    tile_w = frame_rgb.shape[1]
    tile_h = frame_rgb.shape[0]
    pad = 12
    caption_h = 54
    cols = 4
    rows = 2
    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * pad, rows * (tile_h + caption_h) + (rows + 1) * pad),
        (248, 244, 236),
    )
    draw = ImageDraw.Draw(canvas)
    font = _find_font(24)
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row = index // cols
        col = index % cols
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_h + caption_h + pad)
        canvas.paste(tile.resize((tile_w, tile_h)), (x, y))
        draw.text((x, y + tile_h + 10), label, fill=(20, 20, 20), font=font)
    canvas.save(out_path)


def _select_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "temporal_union":
        union_hw = (mask > 0.5).any(axis=0).astype(np.float32)
        return np.repeat(union_hw[None, ...], mask.shape[0], axis=0).astype(np.float32)
    return mask.astype(np.float32)


def _render_mode_videos(
    video_thwc_u8: np.ndarray,
    motion_mask: np.ndarray,
    *,
    out_dir: Path,
    prefix: str,
) -> dict[str, str]:
    binary_video = out_dir / f"{prefix}_binary.mp4"
    motion_overlay_video = out_dir / f"{prefix}_motion_overlay.mp4"
    background_overlay_video = out_dir / f"{prefix}_background_overlay.mp4"
    write_mp4_h264(binary_video, render_binary_mask_video(motion_mask), fps=30)
    write_mp4_h264(motion_overlay_video, render_motion_overlay_video(video_thwc_u8, motion_mask), fps=30)
    write_mp4_h264(background_overlay_video, render_background_overlay_video(video_thwc_u8, motion_mask), fps=30)
    return {
        "binary_video": str(binary_video),
        "motion_overlay_video": str(motion_overlay_video),
        "background_overlay_video": str(background_overlay_video),
    }


def build_html(summary: dict, *, out_html: Path) -> None:
    title = f"Viewer Grounding Motion Masks: {Path(summary['video']).name}"
    mode_titles = summary.get("mode_titles", {})
    mode_badge = summary.get("motion_mask_mode", "per_frame")

    def _render_video_cards_for_mode(mode: str) -> str:
        mode_data = summary.get("modes", {}).get(mode, {})
        if not mode_data:
            return ""
        cards = []
        for key in (
            "legacy_binary_video",
            "legacy_motion_overlay_video",
            "legacy_background_overlay_video",
            "viewer_binary_video",
            "viewer_motion_overlay_video",
            "viewer_background_overlay_video",
        ):
            src = mode_data.get(key, "")
            if not src:
                continue
            cards.append(
                f"""
                <div class="video-card">
                  <div class="video-title">{html.escape(mode_titles.get(f'{mode}_{key}', key))}</div>
                  <video controls loop muted preload="metadata">
                    <source src="{html.escape(Path(src).name)}" type="video/mp4">
                  </video>
                </div>
                """
            )
        return "".join(cards)

    if mode_badge == "both":
        video_cards_html = (
            _render_video_cards_for_mode("per_frame")
            + _render_video_cards_for_mode("temporal_union")
        )
    else:
        video_cards_html = "".join(
            f"""
            <div class="video-card">
              <div class="video-title">{html.escape(mode_titles.get(key, key))}</div>
              <video controls loop muted preload="metadata">
                <source src="{html.escape(Path(summary[key]).name)}" type="video/mp4">
              </video>
            </div>
            """
            for key in (
                "legacy_binary_video",
                "legacy_motion_overlay_video",
                "legacy_background_overlay_video",
                "viewer_binary_video",
                "viewer_motion_overlay_video",
                "viewer_background_overlay_video",
            )
        )

    html_text = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #f6f2ea; color: #1e1e1e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .lead {{ color: #5f5648; line-height: 1.6; margin-bottom: 16px; }}
    .note {{ background: #fff9e9; border: 1px solid #ead9a2; border-radius: 10px; padding: 12px 14px; margin-bottom: 20px; color: #5d512e; }}
    .card {{ background: #fff; border: 1px solid #e0d7ca; border-radius: 14px; padding: 16px; }}
    video {{ width: 100%; max-width: 640px; border-radius: 10px; background: #000; display: block; margin-bottom: 12px; }}
    img {{ width: 100%; border-radius: 10px; border: 1px solid #ddd2c2; display: block; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #faf6ef; border: 1px solid #e0d7ca; border-radius: 10px; padding: 12px; }}
    .video-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; margin-bottom: 16px; }}
    .video-card {{ background: #faf6ef; border: 1px solid #e0d7ca; border-radius: 12px; padding: 12px; }}
    .video-title {{ font-weight: 700; margin-bottom: 8px; }}
    .mode-note {{ margin: 8px 0 14px; color: #6b6256; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="lead">
      这页直接用 train0705 的 <code>ViewerGroundingBoxProvider</code> 做 proposal + SAM2 track，再构造 object-motion masks。
    </div>
    <div class="note">
      绿色 = 运动区域，蓝色 = 背景区域。左行是旧版 legacy <code>background_residual</code>，右行是新的 viewer-grounding <code>viewer_guidance_support</code>。
    </div>
    <div class="mode-note">当前运动 mask 模式：<b>{html.escape(str(mode_badge))}</b></div>
    <div class="card">
      <video controls loop muted preload="metadata">
        <source src="{Path(summary['browser_video']).name}" type="video/mp4">
      </video>
      <img src="{Path(summary['panel']).name}" />
      <div class="video-grid">{video_cards_html}</div>
      <pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
    </div>
  </div>
</body>
</html>
"""
    out_html.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_thwc_u8 = load_video_rgb(args.video.expanduser().resolve())
    legacy_masks = compute_all_motion_masks(video_thwc_u8)
    viewer_masks, viewer_debug = compute_viewer_grounding_object_motion_masks(
        video_thwc_u8,
        caption=str(args.caption),
        provider_kwargs={
            "device": args.grounding_device,
            "segment_len": args.segment_len,
            "max_objects": args.max_objects,
            "points_per_object": args.points_per_object,
            "proposal_source": args.proposal_source,
            "motion_score_ratio": args.motion_score_ratio,
            "text_prompt": args.text_prompt,
            "extra_prompt_terms": args.extra_prompt_terms,
            "include_caption_terms": bool(args.include_caption_terms),
            "gdino_box_threshold": args.gdino_box_threshold,
            "gdino_text_threshold": args.gdino_text_threshold,
            "prompt_frame_mode": args.prompt_frame_mode,
            "track_dedupe_iou_threshold": args.track_dedupe_iou_threshold,
            "container_suppress_ratio_threshold": args.container_suppress_ratio_threshold,
            "container_suppress_min_contained": args.container_suppress_min_contained,
            "container_suppress_min_area_ratio": args.container_suppress_min_area_ratio,
            "container_suppress_small_iou_threshold": args.container_suppress_small_iou_threshold,
        },
        motion_dilate_px=args.motion_dilate_px,
        support_dilate_px=args.support_dilate_px,
    )

    legacy_raw_motion = legacy_masks["background_residual"].mask.astype(np.float32)
    viewer_raw_motion = viewer_masks["viewer_guidance_support"].mask.astype(np.float32)

    mode_list = ["per_frame", "temporal_union"] if args.motion_mask_mode == "both" else [args.motion_mask_mode]
    mode_outputs: dict[str, dict[str, str]] = {}
    mode_titles: dict[str, str] = {}
    for mode in mode_list:
        legacy_motion = _select_mask(legacy_raw_motion, mode)
        viewer_motion = _select_mask(viewer_raw_motion, mode)
        suffix = mode
        legacy_paths = _render_mode_videos(video_thwc_u8, legacy_motion, out_dir=out_dir, prefix=f"legacy_{suffix}")
        viewer_paths = _render_mode_videos(video_thwc_u8, viewer_motion, out_dir=out_dir, prefix=f"viewer_{suffix}")
        mode_outputs[f"legacy_{suffix}"] = legacy_paths
        mode_outputs[f"viewer_{suffix}"] = viewer_paths
        mode_titles[f"legacy_{suffix}_binary_video"] = f"Legacy ({mode}) | Binary Mask Video"
        mode_titles[f"legacy_{suffix}_motion_overlay_video"] = f"Legacy ({mode}) | Motion Overlay Video"
        mode_titles[f"legacy_{suffix}_background_overlay_video"] = f"Legacy ({mode}) | Background Overlay Video"
        mode_titles[f"viewer_{suffix}_binary_video"] = f"Viewer ({mode}) | Binary Mask Video"
        mode_titles[f"viewer_{suffix}_motion_overlay_video"] = f"Viewer ({mode}) | Motion Overlay Video"
        mode_titles[f"viewer_{suffix}_background_overlay_video"] = f"Viewer ({mode}) | Background Overlay Video"

    if args.motion_mask_mode == "both":
        panel_path = out_dir / "viewer_grounding_motion_panel_both.png"
        debug_payload = summarize_debug_payload(viewer_debug)
        build_panel(video_thwc_u8, legacy_masks, viewer_masks, debug_payload, panel_path)
        browser_video = ensure_browser_video(args.video.expanduser().resolve())
        summary = {
            "video": str(args.video.expanduser().resolve()),
            "browser_video": str(browser_video),
            "panel": str(panel_path),
            "caption": str(args.caption),
            "motion_mask_mode": args.motion_mask_mode,
            "mode_titles": mode_titles,
            "viewer_debug": debug_payload,
            "legacy_coverage": float(legacy_masks["background_residual"].coverage),
            "viewer_coverage": float(viewer_masks["viewer_guidance_support"].coverage),
            "legacy_mask_shape": list(legacy_raw_motion.shape),
            "viewer_mask_shape": list(viewer_raw_motion.shape),
            "modes": mode_outputs,
        }
    else:
        mode = mode_list[0]
        legacy_paths = mode_outputs[f"legacy_{mode}"]
        viewer_paths = mode_outputs[f"viewer_{mode}"]
        panel_path = out_dir / f"viewer_grounding_motion_panel_{mode}.png"
        debug_payload = summarize_debug_payload(viewer_debug)
        build_panel(video_thwc_u8, legacy_masks, viewer_masks, debug_payload, panel_path)
        browser_video = ensure_browser_video(args.video.expanduser().resolve())
        summary = {
            "video": str(args.video.expanduser().resolve()),
            "browser_video": str(browser_video),
            "panel": str(panel_path),
            "caption": str(args.caption),
            "motion_mask_mode": mode,
            "mode_titles": mode_titles,
            "viewer_debug": debug_payload,
            "legacy_coverage": float(legacy_masks["background_residual"].coverage),
            "viewer_coverage": float(viewer_masks["viewer_guidance_support"].coverage),
            "legacy_mask_shape": list(legacy_raw_motion.shape),
            "viewer_mask_shape": list(viewer_raw_motion.shape),
            "legacy_binary_video": legacy_paths["binary_video"],
            "legacy_motion_overlay_video": legacy_paths["motion_overlay_video"],
            "legacy_background_overlay_video": legacy_paths["background_overlay_video"],
            "viewer_binary_video": viewer_paths["binary_video"],
            "viewer_motion_overlay_video": viewer_paths["motion_overlay_video"],
            "viewer_background_overlay_video": viewer_paths["background_overlay_video"],
        }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_path = out_dir / "index.html"
    build_html(summary, out_html=html_path)
    print(f"[summary] html: {html_path}", flush=True)
    print(f"[summary] json: {summary_path}", flush=True)

    if args.serve:
        import http.server
        import socketserver

        os.chdir(out_dir)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            print(f"[serve] http://localhost:{args.port}/index.html", flush=True)
            httpd.serve_forever()


if __name__ == "__main__":
    main()
