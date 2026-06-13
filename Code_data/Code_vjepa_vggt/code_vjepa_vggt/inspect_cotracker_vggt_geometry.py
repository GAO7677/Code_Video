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
import torch.nn.functional as F

from code_vjepa_vggt.adapters.sam2_motion import (
    GroundingDINOTextDetector,
    SAM2MotionTracker,
    build_motion_prompt_box,
    build_motion_prompt_boxes,
)
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.ball_block_dataset import BallBlockVideoDataset
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior


COTRACKER_REPO_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
if str(COTRACKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_REPO_ROOT))

from cotracker.predictor import CoTrackerPredictor  # type: ignore


OBJECT_COLORS = [
    (214, 40, 40),
    (247, 127, 0),
    (252, 191, 73),
    (42, 157, 143),
    (39, 125, 161),
    (106, 76, 147),
]
VGGT_COLOR = (0, 180, 216)
COTRACKER_COLOR = (0, 119, 182)
QUERY_COLOR = (17, 17, 17)


class ObjectTrack:
    def __init__(self, box_prompt_xyxy: np.ndarray, masks_thw: np.ndarray, boxes_t4: np.ndarray, score: float, phrase: str) -> None:
        self.box_prompt_xyxy = box_prompt_xyxy
        self.masks_thw = masks_thw
        self.boxes_t4 = boxes_t4
        self.score = score
        self.phrase = phrase


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


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


def draw_box_rgb(image: np.ndarray, box_xyxy_px: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def draw_point_rgb(image: np.ndarray, point_xy: np.ndarray, color_rgb: tuple[int, int, int], label: str, radius: int = 5) -> None:
    x, y = [int(round(v)) for v in point_xy.tolist()]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.circle(image, (x, y), radius, color_bgr, 2)
    if label:
        cv2.putText(image, label, (x + 6, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)


def build_multi_object_prompt(caption: str) -> str:
    caption_lower = str(caption).lower()
    ordered = ["sphere", "ball", "block", "box", "cube", "cylinder", "capsule"]
    found = []
    for token in ordered:
        if token in caption_lower and token not in found:
            found.append(token)
    if not found:
        return caption
    return " . ".join(found) + " ."


def colorize_scalar_video(video_thw: np.ndarray) -> np.ndarray:
    values = np.asarray(video_thw, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        return np.zeros(values.shape + (3,), dtype=np.uint8)
    lo = float(np.nanpercentile(values[valid], 5.0))
    hi = float(np.nanpercentile(values[valid], 95.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((values - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    frames = []
    for frame_hw in norm:
        heat_bgr = cv2.applyColorMap((frame_hw * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        frames.append(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def colorize_world_points_video(world_points_thwc: np.ndarray) -> np.ndarray:
    pts = np.asarray(world_points_thwc, dtype=np.float32)
    rgb = np.zeros_like(pts, dtype=np.float32)
    for channel in range(min(3, pts.shape[-1])):
        values = pts[..., channel]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        lo = float(np.nanpercentile(values[valid], 5.0))
        hi = float(np.nanpercentile(values[valid], 95.0))
        if hi <= lo:
            hi = lo + 1.0
        rgb[..., channel] = np.clip((values - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def colorize_point_feature_video(features_tkc: np.ndarray, *, cell_w: int = 48, cell_h: int = 48) -> np.ndarray:
    values = np.asarray(features_tkc, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [T,K,C], got {values.shape}")
    t, k, c = values.shape
    rgb = np.zeros((t, k, 3), dtype=np.float32)
    for ch in range(min(3, c)):
        v = values[..., ch]
        valid = np.isfinite(v)
        if not np.any(valid):
            continue
        lo = float(np.nanpercentile(v[valid], 5.0))
        hi = float(np.nanpercentile(v[valid], 95.0))
        if hi <= lo:
            hi = lo + 1.0
        rgb[..., ch] = np.clip((v - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    frames = []
    for frame_k3 in rgb_u8:
        tile = np.repeat(np.repeat(frame_k3[:, None, :], cell_h, axis=1), cell_w, axis=0)
        frames.append(tile)
    return np.stack(frames, axis=0)


def scale_tracks_to_native(tracks: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    out = tracks.clone()
    out[..., 0] *= float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    out[..., 1] *= float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    return out


def resize_video_bthwc(frames_bthwc_01: torch.Tensor, dst_hw: tuple[int, int]) -> torch.Tensor:
    b, t, h, w, c = frames_bthwc_01.shape
    frames_bchw = frames_bthwc_01.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    resized = F.interpolate(frames_bchw, size=dst_hw, mode="bilinear", align_corners=True)
    return resized.reshape(b, t, c, dst_hw[0], dst_hw[1]).permute(0, 1, 3, 4, 2).contiguous()


def sample_feature_grid(
    features: torch.Tensor,
    tracks: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    batch, frames, grid_h, grid_w, dim = features.shape
    _, _, objects, _ = tracks.shape
    height, width = image_hw
    feature_map = features.permute(0, 1, 4, 2, 3).reshape(batch * frames, dim, grid_h, grid_w)
    x = tracks[..., 0] / max(float(width - 1), 1.0)
    y = tracks[..., 1] / max(float(height - 1), 1.0)
    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(batch * frames, objects, 1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1).reshape(batch, frames, objects, dim)


def render_track_overlay(
    *,
    context_video: torch.Tensor,
    object_tracks: list[ObjectTrack],
    prompt_frame_idx: int,
    query_points_px_k2: np.ndarray,
    query_owner: list[int],
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    color_rgb: tuple[int, int, int],
    prefix: str,
) -> np.ndarray:
    frames = []
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t]).copy()
        for obj_idx, obj_track in enumerate(object_tracks):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            draw_box_rgb(frame, obj_track.boxes_t4[t].astype(np.float32), color, f"sam{obj_idx}")
            if t == prompt_frame_idx:
                draw_box_rgb(frame, obj_track.box_prompt_xyxy.astype(np.float32), color, f"prompt{obj_idx}")
        if t == 0:
            for q_idx, point in enumerate(query_points_px_k2):
                owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
                color = OBJECT_COLORS[owner % len(OBJECT_COLORS)] if owner >= 0 else QUERY_COLOR
                draw_point_rgb(frame, point.astype(np.float32), color, f"q{q_idx}", radius=6)
        for q_idx in range(tracks_tk2.shape[1]):
            label = f"{prefix}{q_idx}"
            if float(visibility_tk[t, q_idx]) < 0.5:
                label += "(inv)"
            draw_point_rgb(frame, tracks_tk2[t, q_idx].astype(np.float32), color_rgb, label, radius=5)
        frames.append(frame)
    return np.stack(frames, axis=0)


def detect_and_track_objects(
    frames_tchw_01: np.ndarray,
    caption: str,
    *,
    sam2_device: str,
    gdino_device: str,
    max_objects: int,
    prompt_frame_mode: str,
) -> tuple[list[ObjectTrack], int]:
    prompt_frame_idx = 0 if prompt_frame_mode == "first" else max(frames_tchw_01.shape[0] - 1, 0)
    text_prompt = build_multi_object_prompt(caption)
    detector = GroundingDINOTextDetector(device=gdino_device, max_boxes=max_objects)
    try:
        detection = detector.detect(frames_tchw_01[prompt_frame_idx], text_prompt, guidance_box_xyxy=None)
        track_boxes = detection.boxes_xyxy[:max_objects]
        track_scores = detection.scores[:max_objects]
        track_phrases = detection.phrases[:max_objects]
    except Exception:
        track_boxes = np.zeros((0, 4), dtype=np.float32)
        track_scores = np.zeros((0,), dtype=np.float32)
        track_phrases = []
    if track_boxes.shape[0] == 0:
        motion_multi = build_motion_prompt_boxes(frames_tchw_01, max_boxes=max_objects)
        track_boxes = motion_multi.boxes_xyxy[:max_objects]
        track_scores = motion_multi.scores[:max_objects]
        track_phrases = [f"motion_component_{i}" for i in range(track_boxes.shape[0])]
        if track_boxes.shape[0] == 0:
            track_boxes = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)[None, :]
            track_scores = np.asarray([1.0], dtype=np.float32)
            track_phrases = ["motion_proxy"]

    sam_tracker = SAM2MotionTracker(device=sam2_device, enable_text_prompt=False)
    outputs: list[ObjectTrack] = []
    for box_xyxy, score, phrase in zip(track_boxes, track_scores, track_phrases):
        sam_out = sam_tracker.track(
            frames_tchw_01,
            prompt_frame_idx=prompt_frame_idx,
            prompt_box_xyxy=np.asarray(box_xyxy, dtype=np.float32),
            caption="",
        )
        if int(sam_out.masks_thw[0].sum()) <= 0:
            continue
        outputs.append(
            ObjectTrack(
                box_prompt_xyxy=np.asarray(box_xyxy, dtype=np.float32),
                masks_thw=sam_out.masks_thw.astype(np.uint8),
                boxes_t4=sam_out.boxes_t4.astype(np.float32),
                score=float(score),
                phrase=str(phrase),
            )
        )
    if not outputs:
        raise RuntimeError("no usable SAM2 object tracks")
    return outputs, prompt_frame_idx


def build_queries_from_tracks(object_tracks: list[ObjectTrack], num_queries: int, min_queries_per_object: int) -> tuple[np.ndarray, list[int], list[int]]:
    alloc = np.zeros((len(object_tracks),), dtype=np.int64)
    alloc[:] = max(min_queries_per_object, num_queries // max(len(object_tracks), 1))
    while alloc.sum() > num_queries:
        alloc[np.argmax(alloc)] -= 1
    query_sets = []
    query_owner = []
    for obj_idx, (track, nq) in enumerate(zip(object_tracks, alloc)):
        pts, _ = build_vggt_query_prior(track.masks_thw, track.boxes_t4, num_queries=int(max(nq, 1)))
        if pts.shape[0] == 0:
            continue
        query_sets.append(pts.astype(np.float32))
        query_owner.extend([obj_idx] * int(pts.shape[0]))
    if not query_sets:
        raise RuntimeError("failed to sample query points from SAM2 masks")
    query_points_px = np.concatenate(query_sets, axis=0)[:num_queries]
    query_owner = query_owner[: query_points_px.shape[0]]
    return query_points_px, query_owner, alloc.tolist()


def build_report(results: list[dict], output_dir: Path) -> Path:
    blocks = []
    for idx, result in enumerate(results):
        blocks.append(
            f"""
  <section class="case">
    <h2>Case {idx} | {result['group']}</h2>
    <p><b>Caption:</b> {result['caption']}</p>
    <p><b>Video:</b> {result['video_path']}</p>
    <p><b>Prompt frame:</b> {result['prompt_frame_idx']}</p>
    <p><b>Query alloc per object:</b> {result['query_alloc']}</p>
    <p><b>Shapes:</b></p>
    <pre>{json.dumps(result['shapes'], indent=2, ensure_ascii=False)}</pre>
    <p><b>Geometry stats:</b></p>
    <pre>{json.dumps(result['geometry_stats'], indent=2, ensure_ascii=False)}</pre>
    <div class="grid">
      <figure><video controls preload="none" playsinline src="{result['vggt_track_video']}"></video><figcaption>VGGT tracks</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['cotracker_track_video']}"></video><figcaption>CoTracker tracks</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['vggt_track_depth_video']}"></video><figcaption>VGGT depth sampled on VGGT tracks</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['cotracker_track_depth_video']}"></video><figcaption>VGGT depth sampled on CoTracker tracks</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['vggt_track_world_video']}"></video><figcaption>VGGT world_points sampled on VGGT tracks</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['cotracker_track_world_video']}"></video><figcaption>VGGT world_points sampled on CoTracker tracks</figcaption></figure>
    </div>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CoTracker Tracks Sample VGGT Geometry</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #ddd; padding: 12px; }}
    video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>CoTracker Tracks Sample VGGT Geometry</h1>
  <p>同一份 VGGT dense geometry，分别沿 VGGT 自己的 tracks 和 CoTracker 的 tracks 做采样，比较几何读数是否更稳定。</p>
  {''.join(blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return html_path


def evaluate_case(
    sample: dict,
    *,
    case_group: str,
    vggt_adapter: VGGTTrackAdapter,
    device: torch.device,
    sam2_device: str,
    gdino_device: str,
    output_dir: Path,
    num_queries: int,
    min_queries_per_object: int,
    prompt_frame_mode: str,
    cotracker_checkpoint: str,
) -> dict:
    context_video = sample["context_video"]
    frames_tchw_01 = ((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy()
    object_tracks, prompt_frame_idx = detect_and_track_objects(
        frames_tchw_01,
        sample["caption"],
        sam2_device=sam2_device,
        gdino_device=gdino_device,
        max_objects=4,
        prompt_frame_mode=prompt_frame_mode,
    )
    query_points_px, query_owner, query_alloc = build_queries_from_tracks(object_tracks, num_queries, min_queries_per_object)

    query_points_prior = torch.from_numpy(query_points_px).unsqueeze(0).to(device=device, dtype=context_video.dtype)
    frames_bthwc = context_video.unsqueeze(0).permute(0, 2, 3, 4, 1).float()
    frames_bthwc = (frames_bthwc + 1.0) / 2.0
    with torch.no_grad():
        vggt_out = vggt_adapter(
            frames_bthwc.to(device),
            query_points_prior=query_points_prior,
            query_image_hw=(context_video.shape[-2], context_video.shape[-1]),
        )
    vggt_tracks_native = scale_tracks_to_native(vggt_out.tracks, vggt_out.image_hw, (context_video.shape[-2], context_video.shape[-1]))

    cotracker_input_hw = (384, 512)
    cotracker_video = resize_video_bthwc(frames_bthwc, cotracker_input_hw).permute(0, 1, 4, 2, 3)
    cotracker_query_points = query_points_px.astype(np.float32).copy()
    cotracker_query_points[:, 0] *= float(cotracker_input_hw[1]) / max(float(context_video.shape[-1]), 1.0)
    cotracker_query_points[:, 1] *= float(cotracker_input_hw[0]) / max(float(context_video.shape[-2]), 1.0)
    queries_torch = torch.from_numpy(
        np.concatenate(
            [
                np.zeros((cotracker_query_points.shape[0], 1), dtype=np.float32),
                cotracker_query_points,
            ],
            axis=1,
        )
    ).unsqueeze(0).to(device)
    cotracker = CoTrackerPredictor(checkpoint=cotracker_checkpoint, offline=True, v2=False, window_len=60).to(device)
    with torch.no_grad():
        cot_tracks, cot_vis = cotracker(cotracker_video.to(device), queries=queries_torch, backward_tracking=False)
    cot_tracks_native = cot_tracks.clone()
    cot_tracks_native[..., 0] *= float(context_video.shape[-1]) / float(cotracker_input_hw[1])
    cot_tracks_native[..., 1] *= float(context_video.shape[-2]) / float(cotracker_input_hw[0])

    if vggt_out.depth is None or vggt_out.world_points is None:
        raise RuntimeError("VGGT output does not contain dense geometry")

    vggt_depth_on_vggt = sample_feature_grid(vggt_out.depth, vggt_out.tracks, vggt_out.image_hw)
    vggt_depth_on_cotracker = sample_feature_grid(vggt_out.depth, scale_tracks_to_native(cot_tracks_native, (context_video.shape[-2], context_video.shape[-1]), vggt_out.image_hw), vggt_out.image_hw)
    vggt_world_on_vggt = sample_feature_grid(vggt_out.world_points, vggt_out.tracks, vggt_out.image_hw)
    vggt_world_on_cotracker = sample_feature_grid(vggt_out.world_points, scale_tracks_to_native(cot_tracks_native, (context_video.shape[-2], context_video.shape[-1]), vggt_out.image_hw), vggt_out.image_hw)

    vggt_track_video = render_track_overlay(
        context_video=context_video,
        object_tracks=object_tracks,
        prompt_frame_idx=prompt_frame_idx,
        query_points_px_k2=query_points_px,
        query_owner=query_owner,
        tracks_tk2=vggt_tracks_native[0].detach().cpu().numpy().astype(np.float32),
        visibility_tk=vggt_out.visibility[0].detach().cpu().numpy().astype(np.float32),
        color_rgb=VGGT_COLOR,
        prefix="v",
    )
    cotracker_track_video = render_track_overlay(
        context_video=context_video,
        object_tracks=object_tracks,
        prompt_frame_idx=prompt_frame_idx,
        query_points_px_k2=query_points_px,
        query_owner=query_owner,
        tracks_tk2=cot_tracks_native[0].detach().cpu().numpy().astype(np.float32),
        visibility_tk=cot_vis[0].detach().cpu().numpy().astype(np.float32),
        color_rgb=COTRACKER_COLOR,
        prefix="c",
    )
    depth_vggt_frames = colorize_point_feature_video(vggt_depth_on_vggt[0].detach().cpu().numpy())
    depth_cot_frames = colorize_point_feature_video(vggt_depth_on_cotracker[0].detach().cpu().numpy())
    world_vggt_frames = colorize_point_feature_video(vggt_world_on_vggt[0].detach().cpu().numpy())
    world_cot_frames = colorize_point_feature_video(vggt_world_on_cotracker[0].detach().cpu().numpy())

    stem = f"{case_group}__{Path(sample['video_path']).stem}"
    paths = {
        "vggt_track_video": output_dir / f"{stem}__vggt_tracks.mp4",
        "cotracker_track_video": output_dir / f"{stem}__cotracker_tracks.mp4",
        "vggt_track_depth_video": output_dir / f"{stem}__depth_on_vggt_tracks.mp4",
        "cotracker_track_depth_video": output_dir / f"{stem}__depth_on_cotracker_tracks.mp4",
        "vggt_track_world_video": output_dir / f"{stem}__world_on_vggt_tracks.mp4",
        "cotracker_track_world_video": output_dir / f"{stem}__world_on_cotracker_tracks.mp4",
    }
    fps = int(sample.get("_fps", 8))
    write_mp4(paths["vggt_track_video"], vggt_track_video, fps=fps)
    write_mp4(paths["cotracker_track_video"], cotracker_track_video, fps=fps)
    write_mp4(paths["vggt_track_depth_video"], depth_vggt_frames, fps=fps)
    write_mp4(paths["cotracker_track_depth_video"], depth_cot_frames, fps=fps)
    write_mp4(paths["vggt_track_world_video"], world_vggt_frames, fps=fps)
    write_mp4(paths["cotracker_track_world_video"], world_cot_frames, fps=fps)

    geometry_stats = {
        "depth_on_vggt_tracks_mean": float(vggt_depth_on_vggt.mean().item()),
        "depth_on_cotracker_tracks_mean": float(vggt_depth_on_cotracker.mean().item()),
        "depth_on_vggt_tracks_std": float(vggt_depth_on_vggt.std().item()),
        "depth_on_cotracker_tracks_std": float(vggt_depth_on_cotracker.std().item()),
        "world_on_vggt_tracks_std": float(vggt_world_on_vggt.std().item()),
        "world_on_cotracker_tracks_std": float(vggt_world_on_cotracker.std().item()),
    }

    return {
        "group": case_group,
        "caption": sample["caption"],
        "video_path": sample["video_path"],
        "prompt_frame_idx": prompt_frame_idx,
        "query_alloc": query_alloc,
        "geometry_stats": geometry_stats,
        "shapes": {
            "context_video": list(context_video.unsqueeze(0).shape),
            "query_points": list(query_points_px.shape),
            "vggt_tracks": list(vggt_out.tracks.shape),
            "cotracker_tracks": list(cot_tracks.shape),
            "vggt_depth": list(vggt_out.depth.shape) if vggt_out.depth is not None else None,
            "vggt_world_points": list(vggt_out.world_points.shape) if vggt_out.world_points is not None else None,
            "depth_on_vggt_tracks": list(vggt_depth_on_vggt.shape),
            "depth_on_cotracker_tracks": list(vggt_depth_on_cotracker.shape),
            "world_on_vggt_tracks": list(vggt_world_on_vggt.shape),
            "world_on_cotracker_tracks": list(vggt_world_on_cotracker.shape),
        },
        "vggt_track_video": str(ensure_browser_video(paths["vggt_track_video"]).relative_to(output_dir.parent)),
        "cotracker_track_video": str(ensure_browser_video(paths["cotracker_track_video"]).relative_to(output_dir.parent)),
        "vggt_track_depth_video": str(ensure_browser_video(paths["vggt_track_depth_video"]).relative_to(output_dir.parent)),
        "cotracker_track_depth_video": str(ensure_browser_video(paths["cotracker_track_depth_video"]).relative_to(output_dir.parent)),
        "vggt_track_world_video": str(ensure_browser_video(paths["vggt_track_world_video"]).relative_to(output_dir.parent)),
        "cotracker_track_world_video": str(ensure_browser_video(paths["cotracker_track_world_video"]).relative_to(output_dir.parent)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-single", type=int, default=1)
    parser.add_argument("--num-multi", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--min-queries-per-object", type=int, default=4)
    parser.add_argument("--prompt-frame-mode", choices=["first", "last"], default="first")
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--output-dir", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/cotracker_vggt_geometry")
    parser.add_argument("--port", type=int, default=8807)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    device = torch.device(args.device if not str(args.device).startswith("cuda") else args.device)

    phys_ds = PhysStateEpisodeDataset(
        root=data_cfg["root"],
        split=data_cfg["split"],
        resolution=tuple(data_cfg["resolution"]),
        num_context_frames=int(data_cfg["num_context_frames"]),
        context_fraction=float(data_cfg.get("context_fraction", 0.5)),
        random_context_frames=False,
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    ball_ds = BallBlockVideoDataset(
        root="/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block",
        num_frames=16,
        num_context_frames=16,
        resolution=tuple(data_cfg["resolution"]),
    )
    vggt_adapter = VGGTTrackAdapter(
        model_path=model_cfg.get("vggt_model_path"),
        num_queries=int(args.num_queries),
        device=str(device),
        input_hw=tuple(model_cfg["vggt_input_hw"]),
    ).to(device)

    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx in range(min(args.num_single, len(phys_ds))):
        sample = phys_ds[idx]
        sample["_fps"] = int(data_cfg.get("fps", 8))
        results.append(
            evaluate_case(
                sample,
                case_group="single_phys_state",
                vggt_adapter=vggt_adapter,
                device=device,
                sam2_device=str(args.device),
                gdino_device=str(args.device),
                output_dir=assets_dir,
                num_queries=int(args.num_queries),
                min_queries_per_object=int(args.min_queries_per_object),
                prompt_frame_mode=str(args.prompt_frame_mode),
                cotracker_checkpoint=str(args.cotracker_checkpoint),
            )
        )
    for idx in range(min(args.num_multi, len(ball_ds))):
        sample = ball_ds[idx]
        sample["_fps"] = int(data_cfg.get("fps", 8))
        results.append(
            evaluate_case(
                sample,
                case_group="multi_ball_block",
                vggt_adapter=vggt_adapter,
                device=device,
                sam2_device=str(args.device),
                gdino_device=str(args.device),
                output_dir=assets_dir,
                num_queries=int(args.num_queries),
                min_queries_per_object=int(args.min_queries_per_object),
                prompt_frame_mode=str(args.prompt_frame_mode),
                cotracker_checkpoint=str(args.cotracker_checkpoint),
            )
        )

    html_path = build_report(results, output_dir)
    print(f"eval report: {html_path}")

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
