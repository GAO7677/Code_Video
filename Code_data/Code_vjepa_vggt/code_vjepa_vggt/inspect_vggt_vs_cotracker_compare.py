from __future__ import annotations

import argparse
import base64
import http.server
import io
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

from code_vjepa_vggt.adapters.sam2_motion import GroundingDINOTextDetector, SAM2MotionTracker, build_motion_prompt_box, build_motion_prompt_boxes
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.ball_block_dataset import BallBlockVideoDataset
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes


COTRACKER_REPO_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
if str(COTRACKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_REPO_ROOT))


OBJECT_COLORS = [
    (214, 40, 40),
    (247, 127, 0),
    (252, 191, 73),
    (42, 157, 143),
    (39, 125, 161),
    (106, 76, 147),
]
GT_COLOR = (255, 255, 255)
VGGT_COLOR = (0, 180, 216)
COTRACKER_COLOR = (0, 119, 182)
QUERY_COLOR = (17, 17, 17)


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


def overlay_mask(frame_hwc: np.ndarray, mask_hw: np.ndarray, color_rgb: tuple[int, int, int], alpha: float = 0.32) -> np.ndarray:
    frame = frame_hwc.astype(np.float32).copy()
    mask = mask_hw > 0
    if np.any(mask):
        color = np.asarray(color_rgb, dtype=np.float32)
        frame[mask] = (1.0 - alpha) * frame[mask] + alpha * color[None, :]
    return np.clip(frame, 0.0, 255.0).astype(np.uint8)


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


def scale_tracks_to_native(tracks: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    out = tracks.clone()
    out[..., 0] *= float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    out[..., 1] *= float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    return out


def resize_video_bthwc(frames_bthwc_01: torch.Tensor, dst_hw: tuple[int, int]) -> torch.Tensor:
    b, t, h, w, c = frames_bthwc_01.shape
    frames_bchw = frames_bthwc_01.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    resized = torch.nn.functional.interpolate(frames_bchw, size=dst_hw, mode="bilinear", align_corners=True)
    return resized.reshape(b, t, c, dst_hw[0], dst_hw[1]).permute(0, 1, 3, 4, 2).contiguous()


def render_overlay_video(
    *,
    context_video: torch.Tensor,
    object_tracks: list,
    prompt_frame_idx: int,
    query_points_px_k2: np.ndarray,
    query_owner: list[int],
    vggt_tracks_native_tk2: torch.Tensor,
    vggt_visibility_tk: torch.Tensor,
    cotracker_tracks_tk2: np.ndarray,
    cotracker_visibility_tk: np.ndarray,
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

        for q_idx in range(vggt_tracks_native_tk2.shape[1]):
            color = VGGT_COLOR
            label = f"v{q_idx}"
            if float(vggt_visibility_tk[t, q_idx].item()) < 0.5:
                label += "(inv)"
            draw_point_rgb(frame, vggt_tracks_native_tk2[t, q_idx].cpu().numpy().astype(np.float32), color, label, radius=5)

        for q_idx in range(cotracker_tracks_tk2.shape[1]):
            label = f"c{q_idx}"
            if float(cotracker_visibility_tk[t, q_idx]) < 0.5:
                label += "(inv)"
            draw_point_rgb(frame, cotracker_tracks_tk2[t, q_idx].astype(np.float32), COTRACKER_COLOR, label, radius=4)

        frames.append(frame)
    return np.stack(frames, axis=0)


def build_split_views(
    *,
    context_video: torch.Tensor,
    object_tracks: list,
    query_points_px_k2: np.ndarray,
    query_owner: list[int],
    vggt_tracks_native_tk2: torch.Tensor,
    vggt_visibility_tk: torch.Tensor,
    cotracker_tracks_tk2: np.ndarray,
    cotracker_visibility_tk: np.ndarray,
    prompt_frame_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vggt_frames = []
    cot_frames = []
    query_frames = []
    for t in range(context_video.shape[1]):
        base = tensor_frame_to_uint8_hwc(context_video[:, t]).copy()
        for obj_idx, obj_track in enumerate(object_tracks):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            draw_box_rgb(base, obj_track.boxes_t4[t].astype(np.float32), color, f"sam{obj_idx}")
            if t == prompt_frame_idx:
                draw_box_rgb(base, obj_track.box_prompt_xyxy.astype(np.float32), color, f"prompt{obj_idx}")
        query_base = base.copy()
        if t == 0:
            for q_idx, point in enumerate(query_points_px_k2):
                owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
                color = OBJECT_COLORS[owner % len(OBJECT_COLORS)] if owner >= 0 else QUERY_COLOR
                draw_point_rgb(query_base, point.astype(np.float32), color, f"q{q_idx}", radius=6)
        query_frames.append(query_base)

        vggt_frame = base.copy()
        for q_idx in range(vggt_tracks_native_tk2.shape[1]):
            label = f"v{q_idx}"
            if float(vggt_visibility_tk[t, q_idx].item()) < 0.5:
                label += "(inv)"
            draw_point_rgb(vggt_frame, vggt_tracks_native_tk2[t, q_idx].cpu().numpy().astype(np.float32), VGGT_COLOR, label, radius=5)
        vggt_frames.append(vggt_frame)

        cot_frame = base.copy()
        for q_idx in range(cotracker_tracks_tk2.shape[1]):
            label = f"c{q_idx}"
            if float(cotracker_visibility_tk[t, q_idx]) < 0.5:
                label += "(inv)"
            draw_point_rgb(cot_frame, cotracker_tracks_tk2[t, q_idx].astype(np.float32), COTRACKER_COLOR, label, radius=4)
        cot_frames.append(cot_frame)

    return np.stack(query_frames, axis=0), np.stack(vggt_frames, axis=0), np.stack(cot_frames, axis=0)


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
    <p><b>Metrics:</b></p>
    <pre>{json.dumps(result['metrics'], indent=2, ensure_ascii=False)}</pre>
    <div class="grid">
      <figure><video controls preload="none" playsinline src="{result['overlay_video']}"></video><figcaption>Overlay: VGGT + CoTracker</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['query_video']}"></video><figcaption>Query points + SAM2 prompt</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['vggt_video']}"></video><figcaption>VGGT tracks</figcaption></figure>
      <figure><video controls preload="none" playsinline src="{result['cotracker_video']}"></video><figcaption>CoTracker tracks</figcaption></figure>
    </div>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VGGT vs CoTracker Compare</title>
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
  <h1>VGGT vs CoTracker Compare</h1>
  <p>同一套 SAM2 产生的 query points，同时送入 VGGT 和 CoTracker，按 case 并排比较轨迹结果。</p>
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
    prompt_frame_idx = 0 if prompt_frame_mode == "first" else max(frames_tchw_01.shape[0] - 1, 0)

    text_prompt = build_multi_object_prompt(sample["caption"])
    detector = GroundingDINOTextDetector(device=gdino_device, max_boxes=4)
    try:
        detection = detector.detect(frames_tchw_01[prompt_frame_idx], text_prompt, guidance_box_xyxy=None)
        track_boxes = detection.boxes_xyxy[:4]
        track_scores = detection.scores[:4]
        track_phrases = detection.phrases[:4]
    except Exception:
        track_boxes = np.zeros((0, 4), dtype=np.float32)
        track_scores = np.zeros((0,), dtype=np.float32)
        track_phrases = []
    if track_boxes.shape[0] == 0:
        motion_multi = build_motion_prompt_boxes(frames_tchw_01, max_boxes=4)
        track_boxes = motion_multi.boxes_xyxy[:4]
        track_scores = motion_multi.scores[:4]
        track_phrases = [f"motion_component_{i}" for i in range(track_boxes.shape[0])]
        if track_boxes.shape[0] == 0:
            track_boxes = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)[None, :]
            track_scores = np.asarray([1.0], dtype=np.float32)
            track_phrases = ["motion_proxy"]

    sam_tracker = SAM2MotionTracker(device=sam2_device, enable_text_prompt=False)
    object_tracks = []
    for box_xyxy, score, phrase in zip(track_boxes, track_scores, track_phrases):
        sam_out = sam_tracker.track(
            frames_tchw_01,
            prompt_frame_idx=prompt_frame_idx,
            prompt_box_xyxy=np.asarray(box_xyxy, dtype=np.float32),
            caption="",
        )
        if int(sam_out.masks_thw[0].sum()) <= 0:
            continue
        object_tracks.append(
            type("ObjectTrack", (), {
                "box_prompt_xyxy": np.asarray(box_xyxy, dtype=np.float32),
                "masks_thw": sam_out.masks_thw.astype(np.uint8),
                "boxes_t4": sam_out.boxes_t4.astype(np.float32),
                "score": float(score),
                "phrase": str(phrase),
            })()
        )
    if not object_tracks:
        raise RuntimeError("no usable SAM2 object tracks")

    if len(object_tracks) > num_queries:
        object_tracks = object_tracks[:num_queries]

    alloc = np.zeros((len(object_tracks),), dtype=np.int64)
    if len(object_tracks) <= num_queries:
        alloc[:] = max(min_queries_per_object, num_queries // max(len(object_tracks), 1))
        while alloc.sum() > num_queries:
            alloc[np.argmax(alloc)] -= 1
    else:
        alloc[:] = 1

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
    video_torch = cotracker_video.to(device)
    queries_torch = torch.from_numpy(
        np.concatenate(
            [
                np.zeros((cotracker_query_points.shape[0], 1), dtype=np.float32),
                cotracker_query_points,
            ],
            axis=1,
        )
    ).unsqueeze(0).to(device)
    from cotracker.predictor import CoTrackerPredictor  # type: ignore

    cotracker = CoTrackerPredictor(checkpoint=cotracker_checkpoint, offline=True, v2=False, window_len=60).to(device)
    with torch.no_grad():
        cot_tracks, cot_vis = cotracker(video_torch, queries=queries_torch, backward_tracking=False)
    cot_tracks_np = cot_tracks[0].detach().cpu().numpy().astype(np.float32)
    cot_tracks_native = cot_tracks_np.copy()
    cot_tracks_native[..., 0] *= float(context_video.shape[-1]) / float(cotracker_input_hw[1])
    cot_tracks_native[..., 1] *= float(context_video.shape[-2]) / float(cotracker_input_hw[0])
    cot_vis_np = cot_vis[0].detach().cpu().numpy().astype(np.float32)

    overlay = render_overlay_video(
        context_video=context_video,
        object_tracks=object_tracks,
        prompt_frame_idx=prompt_frame_idx,
        query_points_px_k2=query_points_px,
        query_owner=query_owner,
        vggt_tracks_native_tk2=vggt_tracks_native[0].cpu(),
        vggt_visibility_tk=vggt_out.visibility[0].cpu(),
        cotracker_tracks_tk2=cot_tracks_native,
        cotracker_visibility_tk=cot_vis_np,
    )
    query_video, vggt_video, cot_video = build_split_views(
        context_video=context_video,
        object_tracks=object_tracks,
        query_points_px_k2=query_points_px,
        query_owner=query_owner,
        vggt_tracks_native_tk2=vggt_tracks_native[0].cpu(),
        vggt_visibility_tk=vggt_out.visibility[0].cpu(),
        cotracker_tracks_tk2=cot_tracks_native,
        cotracker_visibility_tk=cot_vis_np,
        prompt_frame_idx=prompt_frame_idx,
    )

    overlay_raw = output_dir / f"{case_group}__{Path(sample['video_path']).stem}__overlay.mp4"
    query_raw = output_dir / f"{case_group}__{Path(sample['video_path']).stem}__query.mp4"
    vggt_raw = output_dir / f"{case_group}__{Path(sample['video_path']).stem}__vggt.mp4"
    cot_raw = output_dir / f"{case_group}__{Path(sample['video_path']).stem}__cotracker.mp4"
    fps = int(sample.get("_fps", 8))
    write_mp4(overlay_raw, overlay, fps=fps)
    write_mp4(query_raw, query_video, fps=fps)
    write_mp4(vggt_raw, vggt_video, fps=fps)
    write_mp4(cot_raw, cot_video, fps=fps)

    gt_boxes = sample.get("context_boxes")
    metrics = {}
    if gt_boxes is not None:
        alignment_v = align_tracks_to_boxes(vggt_tracks_native, gt_boxes.unsqueeze(0).to(device), image_hw=(context_video.shape[-2], context_video.shape[-1]))
        alignment_c = align_tracks_to_boxes(torch.from_numpy(cot_tracks_native).unsqueeze(0).to(device), gt_boxes.unsqueeze(0).to(device), image_hw=(context_video.shape[-2], context_video.shape[-1]))
        metrics = {
            "vggt_mean_center_l1_px": float((vggt_tracks_native - alignment_v.matched_gt_centers).abs().sum(dim=-1)[alignment_v.matched_gt_valid > 0.5].mean().item()) if (alignment_v.matched_gt_valid > 0.5).any() else 0.0,
            "cotracker_mean_center_l1_px": float((torch.from_numpy(cot_tracks_native).unsqueeze(0).to(device) - alignment_c.matched_gt_centers).abs().sum(dim=-1)[alignment_c.matched_gt_valid > 0.5].mean().item()) if (alignment_c.matched_gt_valid > 0.5).any() else 0.0,
            "vggt_valid_track_points": int((alignment_v.matched_gt_valid > 0.5).sum().item()),
            "cotracker_valid_track_points": int((alignment_c.matched_gt_valid > 0.5).sum().item()),
        }

    return {
        "group": case_group,
        "caption": sample["caption"],
        "video_path": sample["video_path"],
        "prompt_frame_idx": prompt_frame_idx,
        "query_alloc": alloc.tolist(),
        "query_owner": query_owner,
        "overlay_video": str(ensure_browser_video(overlay_raw).relative_to(output_dir.parent)),
        "query_video": str(ensure_browser_video(query_raw).relative_to(output_dir.parent)),
        "vggt_video": str(ensure_browser_video(vggt_raw).relative_to(output_dir.parent)),
        "cotracker_video": str(ensure_browser_video(cot_raw).relative_to(output_dir.parent)),
        "metrics": metrics,
        "shapes": {
            "context_video": list(context_video.unsqueeze(0).shape),
            "cotracker_input_hw": list(cotracker_input_hw),
            "query_points": list(query_points_px.shape),
            "vggt_tracks": list(vggt_out.tracks.shape),
            "cotracker_tracks": list(cot_tracks.shape),
            "vggt_visibility": list(vggt_out.visibility.shape),
            "cotracker_visibility": list(cot_vis.shape),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--num-single", type=int, default=2)
    parser.add_argument("--num-multi", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--min-queries-per-object", type=int, default=4)
    parser.add_argument("--prompt-frame-mode", choices=["first", "last"], default="first")
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--output-dir", default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/vggt_vs_cotracker_compare")
    parser.add_argument("--port", type=int, default=8805)
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
