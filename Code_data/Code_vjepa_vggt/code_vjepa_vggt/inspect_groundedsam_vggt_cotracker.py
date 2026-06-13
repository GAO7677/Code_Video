from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from code_vjepa_vggt.eval_vggt_sam_multi_object_viewer import (
    OBJECT_COLORS,
    QUERY_COLORS,
    build_query_prior_from_tracks_with_minimum,
    detect_and_track_objects,
)


COTRACKER_REPO_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
if str(COTRACKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_REPO_ROOT))

from cotracker.predictor import CoTrackerPredictor  # type: ignore


GT_COLOR = (255, 255, 255)
PROMPT_COLOR = (255, 140, 0)
MASK_ALPHA = 0.36
MASK_BASE = np.array([32, 160, 96], dtype=np.float32)


def read_video_cv2(video_path: Path, *, max_frames: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames > 0 and len(frames) >= int(max_frames):
                break
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"decoded zero frames from video: {video_path}")
    return np.stack(frames, axis=0)


def write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def find_ffmpeg() -> str | None:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/home/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/home/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
    ]
    for item in candidates:
        if item and Path(item).is_file():
            return str(item)
    return None


def ensure_browser_video(source_path: Path) -> Path:
    ffmpeg = find_ffmpeg()
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


def draw_box_rgb(image: np.ndarray, box_xyxy: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 3, max(16, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def draw_point_rgb(image: np.ndarray, point_xy: np.ndarray, color_rgb: tuple[int, int, int], label: str = "", radius: int = 5) -> None:
    x, y = [int(round(v)) for v in point_xy.tolist()]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.circle(image, (x, y), radius, color_bgr, 2)
    if label:
        cv2.putText(image, label, (x + 5, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)


def overlay_mask(frame_hwc: np.ndarray, mask_hw: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    frame = frame_hwc.astype(np.float32).copy()
    mask = mask_hw > 0
    if np.any(mask):
        color = np.asarray(color_rgb, dtype=np.float32)
        frame[mask] = (1.0 - MASK_ALPHA) * frame[mask] + MASK_ALPHA * color[None, :]
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


def save_prompt_preview(
    *,
    frame_hwc: np.ndarray,
    object_tracks: list,
    prompt_frame_idx: int,
    caption: str,
    output_path: Path,
) -> None:
    frame = frame_hwc.copy()
    for obj_idx, track in enumerate(object_tracks):
        color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
        draw_box_rgb(frame, track.box_prompt_xyxy.astype(np.float32), color, f"gdino{obj_idx}:{track.phrase}")
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle([4, 4, min(image.width - 4, 920), 72], fill=(0, 0, 0))
    draw.text((10, 10), f"prompt_frame={prompt_frame_idx}", fill=(255, 255, 255))
    draw.text((10, 34), f"caption: {caption[:110]}", fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_query_preview(
    *,
    frame_hwc: np.ndarray,
    object_tracks: list,
    query_points_px: np.ndarray,
    query_owner: list[int],
    output_path: Path,
) -> None:
    frame = frame_hwc.copy()
    for obj_idx, track in enumerate(object_tracks):
        color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
        frame = overlay_mask(frame, track.masks_thw[0], color)
        draw_box_rgb(frame, track.boxes_t4[0].astype(np.float32), color, f"sam{obj_idx}")
    for q_idx, point in enumerate(query_points_px):
        owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
        color = OBJECT_COLORS[owner % len(OBJECT_COLORS)] if owner >= 0 else GT_COLOR
        draw_point_rgb(frame, point.astype(np.float32), color, f"q{q_idx}", radius=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)


def render_sam2_overlay(
    *,
    frames_thwc: np.ndarray,
    object_tracks: list,
    query_points_px: np.ndarray,
    query_owner: list[int],
    prompt_frame_idx: int,
) -> np.ndarray:
    rendered = []
    num_frames = frames_thwc.shape[0]
    for t in range(num_frames):
        frame = frames_thwc[t].copy()
        for obj_idx, track in enumerate(object_tracks):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            frame = overlay_mask(frame, track.masks_thw[t], color)
            draw_box_rgb(frame, track.boxes_t4[t].astype(np.float32), color, f"sam{obj_idx}")
            if t == prompt_frame_idx:
                draw_box_rgb(frame, track.box_prompt_xyxy.astype(np.float32), PROMPT_COLOR, f"prompt{obj_idx}")
        if t == 0:
            for q_idx, point in enumerate(query_points_px):
                owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
                color = OBJECT_COLORS[owner % len(OBJECT_COLORS)] if owner >= 0 else GT_COLOR
                draw_point_rgb(frame, point.astype(np.float32), color, f"q{q_idx}", radius=6)
        rendered.append(frame)
    return np.stack(rendered, axis=0)


def render_track_overlay(
    *,
    frames_thwc: np.ndarray,
    object_tracks: list,
    query_points_px: np.ndarray,
    query_owner: list[int],
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    title_prefix: str,
) -> np.ndarray:
    rendered = []
    num_frames = frames_thwc.shape[0]
    for t in range(num_frames):
        frame = frames_thwc[t].copy()
        for obj_idx, track in enumerate(object_tracks):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            draw_box_rgb(frame, track.boxes_t4[t].astype(np.float32), color, f"sam{obj_idx}")
        if t == 0:
            for q_idx, point in enumerate(query_points_px):
                owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
                color = OBJECT_COLORS[owner % len(OBJECT_COLORS)] if owner >= 0 else GT_COLOR
                draw_point_rgb(frame, point.astype(np.float32), color, f"q{q_idx}", radius=6)
        for q_idx in range(tracks_tk2.shape[1]):
            owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
            color = OBJECT_COLORS[owner % len(OBJECT_COLORS)] if owner >= 0 else QUERY_COLORS[q_idx % len(QUERY_COLORS)]
            label = f"{title_prefix}{q_idx}"
            if visibility_tk is not None and float(visibility_tk[t, q_idx]) < 0.5:
                label += "(inv)"
            draw_point_rgb(frame, tracks_tk2[t, q_idx].astype(np.float32), color, label, radius=5)
        rendered.append(frame)
    return np.stack(rendered, axis=0)


def build_report(results: dict, output_dir: Path) -> Path:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Grounded-SAM Query Points to CoTracker</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    figure {{ margin: 0; }}
    img, video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 5px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>GroundingDINO -> SAM2 -> Query Points -> CoTracker</h1>
  <p>同一条视频先做 GroundingDINO 文本检测，再逐物体跑 SAM2 得到 mask 和 box，然后从 frame0 的 SAM2 mask 里采样 query points，最后把这些点送入 CoTracker 做整段视频跟踪。下方分别展示 prompt、query points、公共前处理、以及 CoTracker 跟踪结果。</p>
  <div class="grid">
    <figure>
      <img src="{results['prompt_preview']}" alt="prompt preview">
      <figcaption>Prompt Frame: GroundingDINO 检测框和短文本 prompt</figcaption>
    </figure>
    <figure>
      <img src="{results['query_preview']}" alt="query preview">
      <figcaption>Query Frame: SAM2 mask / SAM2 box / query points</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{results['sam2_overlay_video']}"></video>
      <figcaption>公共前处理: GroundingDINO + SAM2 + query points</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{results['cotracker_overlay_video']}"></video>
      <figcaption>CoTracker tracks overlay</figcaption>
    </figure>
  </div>
  <pre>{json.dumps(results['meta'], indent=2, ensure_ascii=False)}</pre>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-path",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block/e03_mu05_m1.mp4",
    )
    parser.add_argument("--caption", default="Ball colliding with a wooden block")
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--min-queries-per-object", type=int, default=4)
    parser.add_argument("--prompt-frame-mode", choices=["first", "last"], default="first")
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--port", type=int, default=8803)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/groundedsam_vggt_cotracker_viewer",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames_thwc = read_video_cv2(Path(args.video_path), max_frames=int(args.max_frames))
    frames_tchw_01 = np.transpose(frames_thwc.astype(np.float32) / 255.0, (0, 3, 1, 2))
    context_video = torch.from_numpy(frames_thwc).permute(3, 0, 1, 2).float() / 127.5 - 1.0

    object_tracks, prompt_frame_idx = detect_and_track_objects(
        frames_tchw_01,
        args.caption,
        sam2_device=str(device),
        gdino_device=str(device),
        max_objects=4,
        prompt_frame_mode=str(args.prompt_frame_mode),
    )
    if not object_tracks:
        raise RuntimeError("GroundingDINO + SAM2 produced zero usable object tracks")
    query_points_px, query_owner, query_alloc, prior_source = build_query_prior_from_tracks_with_minimum(
        object_tracks,
        int(args.num_queries),
        int(args.min_queries_per_object),
    )
    if query_points_px.shape[0] <= 0:
        raise RuntimeError("failed to build query points from SAM2 masks")

    prompt_preview_path = assets_dir / "prompt_preview.png"
    query_preview_path = assets_dir / "query_preview.png"
    save_prompt_preview(
        frame_hwc=frames_thwc[prompt_frame_idx],
        object_tracks=object_tracks,
        prompt_frame_idx=int(prompt_frame_idx),
        caption=args.caption,
        output_path=prompt_preview_path,
    )
    save_query_preview(
        frame_hwc=frames_thwc[0],
        object_tracks=object_tracks,
        query_points_px=query_points_px,
        query_owner=query_owner,
        output_path=query_preview_path,
    )

    sam2_overlay = render_sam2_overlay(
        frames_thwc=frames_thwc,
        object_tracks=object_tracks,
        query_points_px=query_points_px,
        query_owner=query_owner,
        prompt_frame_idx=int(prompt_frame_idx),
    )
    sam2_overlay_raw = assets_dir / "sam2_overlay.mp4"
    write_mp4(sam2_overlay_raw, sam2_overlay, fps=int(args.fps))
    sam2_overlay_video = ensure_browser_video(sam2_overlay_raw)

    queries = np.concatenate(
        [
            np.zeros((query_points_px.shape[0], 1), dtype=np.float32),
            query_points_px.astype(np.float32),
        ],
        axis=1,
    )
    queries_torch = torch.from_numpy(queries).unsqueeze(0).to(device=device)
    video_torch = torch.from_numpy(frames_thwc).permute(0, 3, 1, 2)[None].float().to(device)
    cotracker = CoTrackerPredictor(
        checkpoint=str(args.cotracker_checkpoint),
        offline=True,
        v2=False,
        window_len=60,
    ).to(device)
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(
            video_torch,
            queries=queries_torch,
            backward_tracking=False,
        )
    cotracker_tracks = pred_tracks[0].detach().cpu().numpy().astype(np.float32)
    cotracker_visibility = pred_visibility[0].detach().cpu().numpy().astype(np.float32)
    cotracker_overlay = render_track_overlay(
        frames_thwc=frames_thwc,
        object_tracks=object_tracks,
        query_points_px=query_points_px,
        query_owner=query_owner,
        tracks_tk2=cotracker_tracks,
        visibility_tk=cotracker_visibility,
        title_prefix="c",
    )
    cotracker_overlay_raw = assets_dir / "cotracker_overlay.mp4"
    write_mp4(cotracker_overlay_raw, cotracker_overlay, fps=int(args.fps))
    cotracker_overlay_video = ensure_browser_video(cotracker_overlay_raw)

    meta = {
        "video_path": str(args.video_path),
        "caption": args.caption,
        "device": str(device),
        "prompt_frame_idx": int(prompt_frame_idx),
        "detected_objects": [
            {
                "phrase": track.phrase,
                "score": float(track.score),
                "prompt_box_xyxy": track.box_prompt_xyxy.astype(np.float32).tolist(),
            }
            for track in object_tracks
        ],
        "query_alloc": query_alloc,
        "query_owner": query_owner,
        "prior_source": prior_source,
        "query_points_shape": list(query_points_px.shape),
        "cotracker_tracks_shape": list(pred_tracks.shape),
        "cotracker_visibility_shape": list(pred_visibility.shape),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    report = {
        "prompt_preview": str(prompt_preview_path.relative_to(output_dir)),
        "query_preview": str(query_preview_path.relative_to(output_dir)),
        "sam2_overlay_video": str(sam2_overlay_video.relative_to(output_dir)),
        "cotracker_overlay_video": str(cotracker_overlay_video.relative_to(output_dir)),
        "meta": meta,
    }
    html_path = build_report(report, output_dir)
    print(f"report: {html_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", int(args.port)), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{int(args.port)}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
