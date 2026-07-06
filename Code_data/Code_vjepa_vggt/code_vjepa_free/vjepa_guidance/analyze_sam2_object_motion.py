#!/usr/bin/env python3
"""
Run command example:

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/analyze_sam2_object_motion.py \
  --device cuda:0 \
  --sam2-device cuda:0 \
  --limit 2
"""
from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from code_vjepa_free.vjepa_guidance.analyze_local_vjepa import (
    DEFAULT_VIDEOS,
    DEFAULT_VJEPA_CKPT,
    build_raw_strip,
    compute_local_scores,
    ensure_media_link,
    load_video_rgb,
    video_to_btchw,
    _heat_to_rgb,
    _resize_heat,
    _safe_stem,
)
from code_vjepa_free.vjepa_guidance.motion_masks import MotionMaskResult, compute_all_motion_masks
from code_vjepa_free.vjepa_guidance.sam2_object_motion_masks import (
    SAM2TrackDebug,
    compute_sam2_object_motion_masks,
    summarize_debug_payload,
)
from code_vjepa_free.vjepa_guidance.vjepa_surprise import VJEPASurpriseEnergy


DEFAULT_OUT_DIR = Path("/data/gaoya/agent-data/outputs/sam2_object_motion_compare/0613pybullet_sample_001460_w002")
LEGACY_SCHEMES = ("frame_diff", "background_residual", "hybrid", "optical_flow")
SAM2_SCHEMES = ("sam2_object_union", "sam2_motion_xor", "sam2_trajectory_envelope", "sam2_guidance_support")
DISPLAY_LEGACY_SCHEME = "background_residual"
DISPLAY_SAM2_SCHEME = "sam2_guidance_support"


@dataclass
class SchemeScore:
    local_score: float
    coverage: float
    threshold: float


@dataclass
class CompareDiagnostic:
    label: str
    family: str
    variant: str
    video_path: str
    served_video_path: str
    raw_strip_path: str
    panel_path: str
    global_score: float
    schemes: dict[str, SchemeScore]
    sam2_debug: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy local masks against SAM2 object-motion masks.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sam2-device", type=str, default="cuda:0")
    parser.add_argument("--vjepa-model", type=str, default="vith")
    parser.add_argument("--vjepa-ckpt", type=Path, default=DEFAULT_VJEPA_CKPT)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--diff-quantile", type=float, default=0.85)
    parser.add_argument("--flow-quantile", type=float, default=0.80)
    parser.add_argument("--dilate-px", type=int, default=14)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--sam2-max-objects", type=int, default=4)
    parser.add_argument("--sam2-top-frames", type=int, default=3)
    parser.add_argument("--sam2-motion-dilate-px", type=int, default=10)
    parser.add_argument("--sam2-support-dilate-px", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


def _overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    alpha_map = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    base = frame_rgb.astype(np.float32)
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1.0 - alpha * alpha_map) + tint * (alpha * alpha_map)
    return np.clip(out, 0, 255).astype(np.uint8)


def _binary_tile(mask: np.ndarray, *, fg_color: tuple[int, int, int], bg_color: tuple[int, int, int] = (244, 241, 234)) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[...] = np.asarray(bg_color, dtype=np.uint8)
    out[binary > 0] = np.asarray(fg_color, dtype=np.uint8)
    return out


def _overlay_background_region(frame_rgb: np.ndarray, motion_mask: np.ndarray, *, color: tuple[int, int, int] = (70, 140, 245)) -> np.ndarray:
    motion = np.clip(motion_mask.astype(np.float32), 0.0, 1.0)[..., None]
    background = 1.0 - motion
    base = frame_rgb.astype(np.float32)
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out = base * (1.0 - 0.40 * background) + tint * (0.40 * background)
    out = out * (0.85 + 0.15 * motion)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_boxes(frame_rgb: np.ndarray, *, prompt_boxes_xyxy: np.ndarray, track_boxes_xyxy: np.ndarray, frame_idx: int) -> np.ndarray:
    canvas = frame_rgb.copy()
    for box in np.asarray(prompt_boxes_xyxy, dtype=np.float32):
        if box.shape != (4,):
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (220, 70, 70), 2)
    if track_boxes_xyxy.ndim == 3:
        for track_id in range(track_boxes_xyxy.shape[0]):
            box = track_boxes_xyxy[track_id, min(frame_idx, track_boxes_xyxy.shape[1] - 1)]
            x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
            if x1 <= x0 or y1 <= y0:
                continue
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 210, 110), 2)
    return canvas


def _blend_heat(frame_rgb: np.ndarray, heat_hw: np.ndarray) -> np.ndarray:
    heat = _resize_heat(heat_hw, frame_rgb.shape[:2])
    heat_rgb = _heat_to_rgb(heat)
    return np.clip(0.55 * frame_rgb.astype(np.float32) + 0.45 * heat_rgb.astype(np.float32), 0, 255).astype(np.uint8)


def build_compare_panel(
    video_thwc_u8: np.ndarray,
    masks: dict[str, MotionMaskResult],
    heatmaps: dict[str, np.ndarray],
    sam2_debug: SAM2TrackDebug,
    out_path: Path,
) -> None:
    frame_idx = min(video_thwc_u8.shape[0] - 1, max(0, int(sam2_debug.prompt_frame_idx)))
    frame_rgb = video_thwc_u8[frame_idx]
    motion_color = (70, 220, 120)
    background_color = (70, 140, 245)

    legacy_binary = _binary_tile(masks[DISPLAY_LEGACY_SCHEME].mask[frame_idx], fg_color=motion_color)
    legacy_motion_overlay = _overlay_mask(frame_rgb, masks[DISPLAY_LEGACY_SCHEME].mask[frame_idx], color=motion_color)
    legacy_background_overlay = _overlay_background_region(frame_rgb, masks[DISPLAY_LEGACY_SCHEME].mask[frame_idx], color=background_color)

    sam2_motion_binary = _binary_tile(masks[DISPLAY_SAM2_SCHEME].mask[frame_idx], fg_color=motion_color)
    sam2_motion_overlay = _overlay_mask(frame_rgb, masks[DISPLAY_SAM2_SCHEME].mask[frame_idx], color=motion_color)
    sam2_background_overlay = _overlay_background_region(frame_rgb, masks[DISPLAY_SAM2_SCHEME].mask[frame_idx], color=background_color)
    boxes_overlay = _draw_boxes(
        frame_rgb,
        prompt_boxes_xyxy=sam2_debug.prompt_boxes_xyxy,
        track_boxes_xyxy=sam2_debug.boxes_xyxy,
        frame_idx=frame_idx,
    )
    legacy_heat = _blend_heat(frame_rgb, heatmaps[DISPLAY_LEGACY_SCHEME])
    sam2_heat = _blend_heat(frame_rgb, heatmaps[DISPLAY_SAM2_SCHEME])

    tiles = [
        Image.fromarray(frame_rgb),
        Image.fromarray(legacy_binary),
        Image.fromarray(legacy_motion_overlay),
        Image.fromarray(legacy_background_overlay),
        Image.fromarray(legacy_heat),
        Image.fromarray(boxes_overlay),
        Image.fromarray(sam2_motion_binary),
        Image.fromarray(sam2_motion_overlay),
        Image.fromarray(sam2_background_overlay),
        Image.fromarray(sam2_heat),
    ]
    labels = [
        "Legacy | Raw Frame",
        "Legacy | Motion Region Binary",
        "Legacy | Motion Region Overlay",
        "Legacy | Background Region Overlay",
        "Legacy | Token Surprise Heat",
        "SAM2 | Prompt Boxes Red / Tracks Green",
        "SAM2 | Motion Region Binary",
        "SAM2 | Motion Region Overlay",
        "SAM2 | Background Region Overlay",
        "SAM2 | Token Surprise Heat",
    ]

    tile_w = frame_rgb.shape[1]
    tile_h = frame_rgb.shape[0]
    pad = 12
    caption_h = 54
    cols = 5
    rows = int(np.ceil(len(tiles) / cols))
    canvas = Image.new(
        "RGB",
        (cols * tile_w + (cols + 1) * pad, rows * (tile_h + caption_h) + (rows + 1) * pad),
        (248, 244, 236),
    )
    draw = ImageDraw.Draw(canvas)
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            try:
                font = ImageFont.truetype(candidate, 24)
                break
            except Exception:
                font = None
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row = index // cols
        col = index % cols
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_h + caption_h + pad)
        canvas.paste(tile.resize((tile_w, tile_h)), (x, y))
        draw.text((x, y + tile_h + 10), label, fill=(20, 20, 20), font=font)
    canvas.save(out_path)


def summarize_pairwise(rows: list[CompareDiagnostic]) -> dict[str, Any]:
    grouped: dict[str, dict[str, CompareDiagnostic]] = {}
    for row in rows:
        grouped.setdefault(row.family, {})[row.variant] = row

    scheme_names = LEGACY_SCHEMES + SAM2_SCHEMES
    aggregate: dict[str, list[float]] = {name: [] for name in scheme_names}
    global_deltas: list[float] = []
    pair_rows: list[dict[str, Any]] = []

    for family, variants in sorted(grouped.items()):
        baseline = variants.get("baseline")
        guided = variants.get("guided")
        if baseline is None or guided is None:
            continue
        global_delta = float(guided.global_score - baseline.global_score)
        global_deltas.append(abs(global_delta))
        row: dict[str, Any] = {"family": family, "global_delta": global_delta}
        for name in scheme_names:
            local_delta = float(guided.schemes[name].local_score - baseline.schemes[name].local_score)
            aggregate[name].append(abs(local_delta))
            row[name] = local_delta
        pair_rows.append(row)

    global_abs_mean = float(np.mean(global_deltas)) if global_deltas else 0.0
    ranked = []
    for name in scheme_names:
        local_abs_mean = float(np.mean(aggregate[name])) if aggregate[name] else 0.0
        ranked.append(
            {
                "name": name,
                "family": "legacy" if name in LEGACY_SCHEMES else "sam2",
                "mean_abs_local_delta": local_abs_mean,
                "mean_abs_global_delta": global_abs_mean,
                "amplification_vs_global": local_abs_mean / max(global_abs_mean, 1.0e-6),
            }
        )
    ranked.sort(key=lambda item: item["amplification_vs_global"], reverse=True)
    return {"pair_rows": pair_rows, "scheme_rank": ranked}


def build_html(rows: list[CompareDiagnostic], pairwise: dict[str, Any], out_html: Path) -> None:
    rank_cards = "".join(
        f"""
        <div class="rank-card">
          <div class="rank-family">{html.escape(item['family'])}</div>
          <div class="rank-name">{html.escape(item['name'])}</div>
          <div class="rank-main">|Δlocal| / |Δglobal| = {item['amplification_vs_global']:.3f}</div>
          <div class="rank-sub">mean |Δlocal| = {item['mean_abs_local_delta']:.5f}</div>
        </div>
        """
        for item in pairwise["scheme_rank"]
    )

    table_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(row['family'])}</td>
          <td>{row['global_delta']:+.5f}</td>
          <td>{row['background_residual']:+.5f}</td>
          <td>{row['sam2_motion_xor']:+.5f}</td>
          <td>{row['sam2_guidance_support']:+.5f}</td>
        </tr>
        """
        for row in pairwise["pair_rows"]
    )

    cards = []
    for row in rows:
        pills = "".join(
            f"<div class='pill'>{html.escape(name)} {score.local_score:.4f} | cov {score.coverage * 100:.1f}%</div>"
            for name, score in row.schemes.items()
        )
        cards.append(
            f"""
            <div class="card">
              <div class="card-head">
                <div>
                  <div class="label">{html.escape(row.label)}</div>
                  <div class="sub">{html.escape(row.video_path)}</div>
                </div>
                <div class="global">global {row.global_score:.4f}</div>
              </div>
              <div class="note-line">
                legacy display = <b>{html.escape(DISPLAY_LEGACY_SCHEME)}</b>,
                sam2 display = <b>{html.escape(DISPLAY_SAM2_SCHEME)}</b>,
                prompt_frame = <b>{int(row.sam2_debug['prompt_frame_idx'])}</b>,
                sam2 tracks = <b>{int(row.sam2_debug['track_count'])}</b>
              </div>
              <div class="pill-row">{pills}</div>
              <div class="media-grid">
                <div>
                  <video controls loop muted preload="metadata" width="560">
                    <source src="{html.escape(Path(row.served_video_path).name)}" type="video/mp4">
                  </video>
                  <img class="strip" src="{html.escape(Path(row.raw_strip_path).name)}" />
                </div>
                <div>
                  <img class="panel" src="{html.escape(Path(row.panel_path).name)}" />
                </div>
              </div>
            </div>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>SAM2 Object-Motion vs Legacy Motion Mask</title>
  <style>
    body {{ margin: 0; background: #f6f2ea; color: #1e1e1e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1560px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .lead {{ color: #5f5648; line-height: 1.6; margin-bottom: 16px; }}
    .note {{ background: #fff9e9; border: 1px solid #ead9a2; border-radius: 10px; padding: 12px 14px; margin-bottom: 20px; color: #5d512e; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 14px; margin-bottom: 20px; }}
    .rank-card {{ background: #fff; border: 1px solid #e0d7ca; border-radius: 12px; padding: 14px 16px; }}
    .rank-family {{ color: #756d60; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .rank-name {{ font-weight: 700; font-size: 16px; margin: 4px 0 6px; }}
    .rank-main {{ font-size: 18px; }}
    .rank-sub {{ color: #6c6558; margin-top: 4px; }}
    .table-wrap {{ background: #fff; border: 1px solid #e0d7ca; border-radius: 12px; padding: 16px; margin-bottom: 22px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #ece5d8; }}
    th {{ background: #faf6ef; }}
    .card {{ background: #fff; border: 1px solid #e0d7ca; border-radius: 14px; padding: 16px; margin-bottom: 18px; }}
    .card-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 10px; }}
    .label {{ font-size: 18px; font-weight: 700; }}
    .sub {{ color: #756d60; font-size: 12px; margin-top: 4px; word-break: break-all; }}
    .global {{ font-size: 20px; font-weight: 700; white-space: nowrap; }}
    .note-line {{ color: #5d512e; background: #fff8e4; border: 1px solid #ecd8a6; border-radius: 10px; padding: 8px 10px; margin-bottom: 10px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }}
    .pill {{ background: #f4eee3; border: 1px solid #ded3c3; border-radius: 999px; padding: 6px 10px; font-size: 12px; }}
    .media-grid {{ display: grid; grid-template-columns: 580px minmax(720px, 1fr); gap: 18px; align-items: start; }}
    video {{ width: 100%; border-radius: 10px; background: #000; margin-bottom: 10px; }}
    img.panel, img.strip {{ width: 100%; border-radius: 10px; border: 1px solid #ddd2c2; display: block; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>SAM2 Object-Motion vs Legacy Motion Mask</h1>
    <div class="lead">
      对同一个 case 的已有 A/B 视频，比较旧版局部区域方案和新版基于 SAM2 tracked object 的局部区域方案。
      旧版重点看 <code>{html.escape(DISPLAY_LEGACY_SCHEME)}</code>，新版重点看 <code>{html.escape(DISPLAY_SAM2_SCHEME)}</code>。
    </div>
    <div class="note">
      图例：绿色 = legacy background residual 区域，橙色 = sam2 motion xor，蓝色 = sam2 guidance support，
      橙红 = sam2 object union，紫色 = trajectory envelope，红框 = prompt boxes，绿框 = tracked boxes。
    </div>
    <div class="summary">{rank_cards}</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>family</th>
            <th>Δglobal</th>
            <th>Δlegacy background_residual</th>
            <th>Δsam2 motion_xor</th>
            <th>Δsam2 guidance_support</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    out_html.write_text(html_text, encoding="utf-8")


def analyze_video(
    label: str,
    video_path: Path,
    *,
    energy: VJEPASurpriseEnergy,
    args: argparse.Namespace,
    out_dir: Path,
) -> CompareDiagnostic:
    family, variant = label.split("::", 1)
    video_thwc_u8 = load_video_rgb(video_path)
    video_btchw = video_to_btchw(video_thwc_u8)

    legacy_masks = compute_all_motion_masks(
        video_thwc_u8,
        diff_quantile=args.diff_quantile,
        flow_quantile=args.flow_quantile,
        dilate_px=args.dilate_px,
        blur_ksize=args.blur_ksize,
    )
    sam2_masks, sam2_debug = compute_sam2_object_motion_masks(
        video_thwc_u8,
        sam2_device=args.sam2_device,
        segment_len=max(args.window_size, args.context_frames),
        max_objects=args.sam2_max_objects,
        top_frames=args.sam2_top_frames,
        motion_dilate_px=args.sam2_motion_dilate_px,
        support_dilate_px=args.sam2_support_dilate_px,
    )
    all_masks = {**legacy_masks, **sam2_masks}

    scheme_scores: dict[str, SchemeScore] = {}
    heatmaps: dict[str, np.ndarray] = {}
    global_score = None
    for name, motion in all_masks.items():
        global_value, local_value, heat_map = compute_local_scores(
            video_btchw,
            motion.mask,
            energy=energy,
            window_size=args.window_size,
            context_frames=args.context_frames,
            stride=args.stride,
        )
        if global_score is None:
            global_score = global_value
        scheme_scores[name] = SchemeScore(
            local_score=float(local_value),
            coverage=float(motion.coverage),
            threshold=float(motion.threshold),
        )
        heatmaps[name] = heat_map

    raw_strip_path = out_dir / f"{family}_{variant}_strip.png"
    panel_path = out_dir / f"{family}_{variant}_compare.png"
    build_raw_strip(video_thwc_u8, raw_strip_path)
    build_compare_panel(video_thwc_u8, all_masks, heatmaps, sam2_debug, panel_path)

    assert global_score is not None
    return CompareDiagnostic(
        label=label,
        family=family,
        variant=variant,
        video_path=str(video_path),
        served_video_path=str(out_dir / f"{_safe_stem(label)}.mp4"),
        raw_strip_path=str(raw_strip_path),
        panel_path=str(panel_path),
        global_score=float(global_score),
        schemes=scheme_scores,
        sam2_debug=summarize_debug_payload(sam2_debug),
    )


def save_summary(rows: list[CompareDiagnostic], pairwise: dict[str, Any], out_path: Path) -> None:
    payload = {
        "rows": [
            {
                "label": row.label,
                "family": row.family,
                "variant": row.variant,
                "video_path": row.video_path,
                "served_video_path": row.served_video_path,
                "raw_strip_path": row.raw_strip_path,
                "panel_path": row.panel_path,
                "global_score": row.global_score,
                "schemes": {
                    name: {
                        "local_score": score.local_score,
                        "coverage": score.coverage,
                        "threshold": score.threshold,
                    }
                    for name, score in row.schemes.items()
                },
                "sam2_debug": row.sam2_debug,
            }
            for row in rows
        ],
        "pairwise": pairwise,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    energy = VJEPASurpriseEnergy(
        model_name=str(args.vjepa_model),
        device=str(args.device),
        local_torchhub=True,
        checkpoint_path=Path(args.vjepa_ckpt).expanduser().resolve(),
    )

    videos = list(DEFAULT_VIDEOS)
    if args.limit > 0:
        videos = videos[: args.limit]

    rows: list[CompareDiagnostic] = []
    for label, video_path in videos:
        if not video_path.is_file():
            raise FileNotFoundError(f"missing input video: {video_path}")
        ensure_media_link(video_path, out_dir, dst_name=f"{_safe_stem(label)}.mp4")
        print(f"[analyze] {label}", flush=True)
        rows.append(analyze_video(label, video_path, energy=energy, args=args, out_dir=out_dir))

    pairwise = summarize_pairwise(rows)
    summary_path = out_dir / "summary.json"
    html_path = out_dir / "index.html"
    save_summary(rows, pairwise, summary_path)
    build_html(rows, pairwise, html_path)
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
