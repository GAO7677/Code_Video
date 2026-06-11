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
VGGT_QUERY_COLOR = "#111111"
UNMATCHED_GT_COLOR_RGB = (160, 160, 160)


def tensor_frame_to_pil(frame_chw: torch.Tensor) -> Image.Image:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return Image.fromarray(x.numpy())


def pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("ffmpeg not found")


def ensure_browser_video(source_path: Path) -> Path:
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if out_path.exists() and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns and out_path.stat().st_size > 0:
        return out_path
    ffmpeg = find_ffmpeg()
    cmd = [
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
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def box_valid(box_xyxy: torch.Tensor, eps: float = 1e-6) -> bool:
    return bool((box_xyxy[2] - box_xyxy[0] > eps).item() and (box_xyxy[3] - box_xyxy[1] > eps).item())


def track_inside_box(point_xy: torch.Tensor, box_xyxy: torch.Tensor, image_hw: tuple[int, int]) -> bool:
    width = image_hw[1]
    height = image_hw[0]
    x0 = float(box_xyxy[0].item()) * width
    y0 = float(box_xyxy[1].item()) * height
    x1 = float(box_xyxy[2].item()) * width
    y1 = float(box_xyxy[3].item()) * height
    x = float(point_xy[0].item())
    y = float(point_xy[1].item())
    return x0 <= x <= x1 and y0 <= y <= y1


def sample_points_from_box(box_xyxy_px: np.ndarray, num_points: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return np.zeros((num_points, 2), dtype=np.float32)
    cols = max(1, int(math.ceil(math.sqrt(float(num_points)))))
    rows = max(1, int(math.ceil(float(num_points) / float(cols))))
    xs = np.linspace(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), cols, dtype=np.float32)
    ys = np.linspace(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if grid.shape[0] >= num_points:
        return grid[:num_points].astype(np.float32)
    repeat = np.repeat(grid[-1:], num_points - grid.shape[0], axis=0)
    return np.concatenate([grid, repeat], axis=0).astype(np.float32)


def sample_points_from_mask(mask_hw: np.ndarray, num_points: int) -> np.ndarray:
    ys, xs = np.where(mask_hw > 0)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)
    if coords.shape[0] >= num_points:
        index = np.linspace(0, coords.shape[0] - 1, num_points, dtype=np.int64)
        return coords[index].astype(np.float32)
    repeat = np.repeat(coords[-1:], num_points - coords.shape[0], axis=0)
    return np.concatenate([coords, repeat], axis=0).astype(np.float32)


def build_vggt_query_prior(
    sam_masks_thw: np.ndarray,
    sam_boxes_t4: np.ndarray,
    *,
    num_queries: int,
) -> tuple[np.ndarray, str]:
    frame0_mask = sam_masks_thw[0]
    query_points = sample_points_from_mask(frame0_mask, num_queries)
    if query_points.shape[0] == num_queries and np.any(frame0_mask > 0):
        return query_points.astype(np.float32), "sam_mask_frame0"
    box0 = sam_boxes_t4[0]
    return sample_points_from_box(box0, num_queries), "sam_box_frame0"


def color_hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def draw_box_px(draw: ImageDraw.ImageDraw, box_xyxy_px: np.ndarray, color: str, label: str) -> None:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    draw.text((x0 + 2, max(y0 + 2, 2)), label, fill=color)


def draw_case_frame(
    *,
    frame_chw: torch.Tensor,
    gt_boxes_k4: torch.Tensor,
    matched_gt_idx_k: torch.Tensor,
    vggt_tracks_xy_k2: torch.Tensor,
    vggt_vis_k: torch.Tensor,
    sam_box_xyxy_px: np.ndarray,
    sam_prompt_box_xyxy_px: np.ndarray,
    prompt_frame_idx: int,
    frame_idx: int,
    query_points_px_k2: np.ndarray,
) -> Image.Image:
    out = tensor_frame_to_pil(frame_chw).copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size

    for obj_idx, box in enumerate(gt_boxes_k4.tolist()):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline=GT_PALETTE[obj_idx % len(GT_PALETTE)], width=2)
        draw.text((x0 * width + 2, y0 * height + 2), f"gt{obj_idx}", fill=GT_PALETTE[obj_idx % len(GT_PALETTE)])

    draw_box_px(draw, sam_box_xyxy_px.astype(np.float32), SAM_TRACK_COLOR, "sam_track")
    if frame_idx == prompt_frame_idx:
        draw_box_px(draw, sam_prompt_box_xyxy_px.astype(np.float32), SAM_PROMPT_COLOR, "sam_prompt")

    if frame_idx == 0:
        for query_idx, point in enumerate(query_points_px_k2.tolist()):
            x, y = float(point[0]), float(point[1])
            r = 6
            draw.ellipse([x - r, y - r, x + r, y + r], outline=VGGT_QUERY_COLOR, width=3)
            draw.text((x + 7, y + 4), f"prior_q{query_idx}", fill=VGGT_QUERY_COLOR)

    for query_idx, point in enumerate(vggt_tracks_xy_k2.tolist()):
        color = QUERY_PALETTE[query_idx % len(QUERY_PALETTE)]
        x, y = float(point[0]), float(point[1])
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
        gt_idx = int(matched_gt_idx_k[query_idx].item())
        label = f"q{query_idx}->gt{gt_idx}"
        if float(vggt_vis_k[query_idx].item()) < 0.5:
            label += "(inv)"
        draw.text((x + 6, y - 6), label, fill=color)
    return out


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


def render_overlay_video(
    *,
    context_video: torch.Tensor,
    context_boxes: torch.Tensor,
    matched_gt_indices: list[int],
    unmatched_gt_indices: list[int],
    sam_boxes_t4: np.ndarray,
    sam_prompt_box_xyxy: np.ndarray,
    prompt_frame_idx: int,
    query_points_px_k2: np.ndarray,
    tracks_native_tk2: torch.Tensor,
    visibility_tk: torch.Tensor,
) -> np.ndarray:
    frames = []
    image_hw = (context_video.shape[-2], context_video.shape[-1])
    matched_set = set(int(x) for x in matched_gt_indices)
    unmatched_set = set(int(x) for x in unmatched_gt_indices)
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t]).copy()

        for obj_idx in range(context_boxes.shape[1]):
            gt_box = context_boxes[t, obj_idx]
            if not box_valid(gt_box):
                continue
            box_px = (gt_box.cpu() * torch.tensor([image_hw[1], image_hw[0], image_hw[1], image_hw[0]], dtype=gt_box.dtype)).numpy().astype(np.float32)
            if obj_idx in matched_set:
                color_rgb = color_hex_to_rgb(GT_PALETTE[obj_idx % len(GT_PALETTE)])
                label = f"gt{obj_idx}"
            elif obj_idx in unmatched_set:
                color_rgb = UNMATCHED_GT_COLOR_RGB
                label = f"gt{obj_idx}_unmatched"
            else:
                color_rgb = UNMATCHED_GT_COLOR_RGB
                label = f"gt{obj_idx}"
            draw_box_rgb(frame, box_px, color_rgb, label)

        draw_box_rgb(frame, sam_boxes_t4[t].astype(np.float32), (44, 162, 95), "sam_track")
        if t == int(prompt_frame_idx):
            draw_box_rgb(frame, sam_prompt_box_xyxy.astype(np.float32), (255, 140, 0), "sam_prompt")

        if t == 0:
            for query_idx, point in enumerate(query_points_px_k2):
                draw_point_rgb(frame, point.astype(np.float32), (17, 17, 17), f"prior_q{query_idx}", radius=6)

        for query_idx in range(tracks_native_tk2.shape[1]):
            color_rgb = color_hex_to_rgb(QUERY_PALETTE[query_idx % len(QUERY_PALETTE)])
            point = tracks_native_tk2[t, query_idx].cpu().numpy().astype(np.float32)
            gt_idx = matched_gt_indices[query_idx] if query_idx < len(matched_gt_indices) else -1
            label = f"q{query_idx}->gt{gt_idx}"
            if float(visibility_tk[t, query_idx].item()) < 0.5:
                label += "(inv)"
            draw_point_rgb(frame, point, color_rgb, label)
        frames.append(frame)
    return np.stack(frames, axis=0)


def evaluate_sample(
    sample: dict,
    *,
    sam_tracker: SAM2MotionTracker,
    vggt_adapter: VGGTTrackAdapter,
    device: torch.device,
) -> dict:
    context_video = sample["context_video"].unsqueeze(0).to(device)
    context_boxes = sample["context_boxes"].unsqueeze(0).to(device)
    caption = sample["caption"]

    frames_tchw_01 = ((context_video[0].permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy()
    prompt_frame_idx = max(int(context_video.shape[2]) - 1, 0)
    motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
    sam_out = sam_tracker.track(
        frames_tchw_01,
        prompt_frame_idx=prompt_frame_idx,
        prompt_box_xyxy=motion_prompt_box_xyxy,
        caption=caption,
    )

    query_points_px, prior_source = build_vggt_query_prior(
        sam_out.masks_thw,
        sam_out.boxes_t4,
        num_queries=vggt_adapter.num_queries,
    )
    query_points_prior = torch.from_numpy(query_points_px).unsqueeze(0).to(device=device, dtype=context_video.dtype)

    frames_bthwc = context_video.permute(0, 2, 3, 4, 1).float()
    frames_bthwc = (frames_bthwc + 1.0) / 2.0
    with torch.no_grad():
        vggt_out = vggt_adapter(
            frames_bthwc,
            query_points_prior=query_points_prior,
            query_image_hw=(context_video.shape[-2], context_video.shape[-1]),
        )

    tracks = vggt_out.tracks
    vis = vggt_out.visibility
    conf = vggt_out.confidence
    track_image_hw = vggt_out.image_hw

    scale_x = float(context_video.shape[-1]) / float(track_image_hw[1])
    scale_y = float(context_video.shape[-2]) / float(track_image_hw[0])
    tracks_native = tracks.clone()
    tracks_native[..., 0] *= scale_x
    tracks_native[..., 1] *= scale_y

    alignment = align_tracks_to_boxes(
        tracks=tracks_native,
        gt_boxes=context_boxes,
        image_hw=(context_video.shape[-2], context_video.shape[-1]),
    )
    gt_valid_any = ((context_boxes[..., 2] - context_boxes[..., 0] > 1e-6) & (context_boxes[..., 3] - context_boxes[..., 1] > 1e-6)).any(dim=1)[0]
    valid_gt_indices = [int(idx) for idx in torch.nonzero(gt_valid_any, as_tuple=False).flatten().tolist()]
    matched_gt_indices = [int(x) for x in alignment.matched_gt_indices[0].tolist()]
    unmatched_gt_indices = [idx for idx in valid_gt_indices if idx not in set(matched_gt_indices)]
    valid_mask = alignment.matched_gt_valid > 0.5
    l1 = (tracks_native - alignment.matched_gt_centers).abs().sum(dim=-1)
    mean_center_l1 = float(l1[valid_mask].mean().item()) if valid_mask.any() else 0.0

    inside_hits = []
    for t in range(tracks_native.shape[1]):
        for q in range(tracks_native.shape[2]):
            if not bool(valid_mask[0, t, q].item()):
                continue
            gt_idx = int(alignment.matched_gt_indices[0, q].item())
            hit = track_inside_box(
                point_xy=tracks_native[0, t, q],
                box_xyxy=context_boxes[0, t, gt_idx],
                image_hw=(context_video.shape[-2], context_video.shape[-1]),
            )
            inside_hits.append(float(hit))
    inside_rate = float(sum(inside_hits) / len(inside_hits)) if inside_hits else 0.0

    overlay_video = render_overlay_video(
        context_video=sample["context_video"],
        context_boxes=sample["context_boxes"],
        matched_gt_indices=matched_gt_indices,
        unmatched_gt_indices=unmatched_gt_indices,
        sam_boxes_t4=sam_out.boxes_t4,
        sam_prompt_box_xyxy=sam_out.prompt_box_xyxy.astype(np.float32),
        prompt_frame_idx=prompt_frame_idx,
        query_points_px_k2=query_points_px.astype(np.float32),
        tracks_native_tk2=tracks_native[0].cpu(),
        visibility_tk=vggt_out.visibility[0].cpu(),
    )
    assets_dir = Path(sample.get("_output_dir", "."))
    assets_dir.mkdir(parents=True, exist_ok=True)
    raw_video_path = assets_dir / f"{sample.get('_case_name', 'case')}__overlay.mp4"
    write_mp4(raw_video_path, overlay_video, fps=int(sample.get("_fps", 8)))
    browser_video_path = ensure_browser_video(raw_video_path)

    return {
        "caption": caption,
        "video_path": sample["video_path"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "prompt_frame_idx": prompt_frame_idx,
        "sam_prompt_mode": sam_out.prompt_mode,
        "sam_prompt_text": sam_out.prompt_text,
        "sam_prompt_box_xyxy": sam_out.prompt_box_xyxy.tolist(),
        "sam_motion_box_xyxy": motion_prompt_box_xyxy.tolist(),
        "sam_prior_source": prior_source,
        "track_image_hw": list(track_image_hw),
        "vggt_used_model": bool(vggt_out.used_model),
        "matched_gt_indices": matched_gt_indices,
        "unmatched_gt_indices": unmatched_gt_indices,
        "shapes": {
            "context_video": list(context_video.shape),
            "context_boxes": list(context_boxes.shape),
            "sam_masks_thw": list(sam_out.masks_thw.shape),
            "sam_boxes_t4": list(np.asarray(sam_out.boxes_t4).shape),
            "sam_query_points": list(query_points_prior.shape),
            "vggt_query_points": list(vggt_out.query_points.shape),
            "vggt_tracks": list(tracks.shape),
            "vggt_tracks_native_xy": list(tracks_native.shape),
            "vggt_visibility": list(vis.shape),
            "vggt_confidence": list(conf.shape),
            "matched_gt_centers": list(alignment.matched_gt_centers.shape),
            "matched_gt_valid": list(alignment.matched_gt_valid.shape),
        },
        "metrics": {
            "mean_center_l1_px": mean_center_l1,
            "inside_box_rate": inside_rate,
            "valid_track_points": int(valid_mask.sum().item()),
        },
        "overlay_video": str(browser_video_path.relative_to(assets_dir.parent)),
    }


def build_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_cases": len(results),
        "avg_mean_center_l1_px": sum(r["metrics"]["mean_center_l1_px"] for r in results) / max(len(results), 1),
        "avg_inside_box_rate": sum(r["metrics"]["inside_box_rate"] for r in results) / max(len(results), 1),
        "cases": [
            {
                "case_id": idx,
                "video_path": r["video_path"],
                "caption": r["caption"],
                "sam_prompt_mode": r["sam_prompt_mode"],
                "sam_prompt_text": r["sam_prompt_text"],
                "sam_prior_source": r["sam_prior_source"],
                "unmatched_gt_indices": r["unmatched_gt_indices"],
                "metrics": r["metrics"],
            }
            for idx, r in enumerate(results)
        ],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    blocks = []
    for idx, result in enumerate(results):
        blocks.append(
            f"""
  <section class="case">
    <h2>Case {idx}</h2>
    <p><b>Caption:</b> {result['caption']}</p>
    <p><b>Context frames:</b> {result['context_frame_indices']}</p>
    <p><b>SAM prompt:</b> mode={result['sam_prompt_mode']} | text={result['sam_prompt_text']} | prior_source={result['sam_prior_source']} | prompt_frame={result['prompt_frame_idx']}</p>
    <p><b>Metrics:</b> mean_center_l1_px={result['metrics']['mean_center_l1_px']:.2f}, inside_box_rate={result['metrics']['inside_box_rate']:.3f}, valid_track_points={result['metrics']['valid_track_points']}</p>
    <p><b>Matched gt indices:</b> {result['matched_gt_indices']}</p>
    <p><b>Unmatched gt indices:</b> {result['unmatched_gt_indices']}</p>
    <figure>
      <video controls preload="none" playsinline src="{result['overlay_video']}"></video>
      <figcaption>overlay video</figcaption>
    </figure>
    <pre>{json.dumps({'metrics': result['metrics'], 'shapes': result['shapes'], 'sam_prompt_mode': result['sam_prompt_mode'], 'sam_prompt_text': result['sam_prompt_text'], 'sam_prior_source': result['sam_prior_source'], 'sam_motion_box_xyxy': result['sam_motion_box_xyxy'], 'sam_prompt_box_xyxy': result['sam_prompt_box_xyxy'], 'matched_gt_indices': result['matched_gt_indices'], 'unmatched_gt_indices': result['unmatched_gt_indices'], 'track_image_hw': result['track_image_hw'], 'vggt_used_model': result['vggt_used_model']}, indent=2, ensure_ascii=False)}</pre>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VGGT with SAM Prior</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>VGGT with SAM Prior</h1>
  <p>当前页面验证的是：先用 SAM2 依据 caption 和视频内容找到目标物体，再从 SAM2 在 frame0 的 mask/box 里生成 VGGT query priors，随后仍由 VGGT 输出整段轨迹。彩色 gt 框是匹配到 query 的 GT；灰框是当前没有对应 query 的 GT；橙框是送进 SAM2 的 prompt；绿框是 SAM2 跟踪框；黑圈是 frame0 上喂给 VGGT 的 prior 点；彩色圆点是 VGGT 的 tracks。</p>
  <p><b>Overall:</b> avg_mean_center_l1_px={summary['avg_mean_center_l1_px']:.2f}, avg_inside_box_rate={summary['avg_inside_box_rate']:.3f}, num_cases={summary['num_cases']}</p>
  {''.join(blocks)}
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
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/vggt_sam_prior_viewer",
    )
    parser.add_argument("--port", type=int, default=8771)
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

    vggt_adapter = VGGTTrackAdapter(
        model_path=model_cfg.get("vggt_model_path"),
        num_queries=int(model_cfg["object_num_queries"]),
        device=str(device),
        input_hw=tuple(model_cfg["vggt_input_hw"]),
    ).to(device)
    sam_tracker = SAM2MotionTracker(device="cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    results = []
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(args.start_index, min(len(dataset), args.start_index + args.num_cases)):
        sample = dataset[idx]
        sample["_output_dir"] = assets_dir
        sample["_case_name"] = f"case_{idx:03d}"
        sample["_fps"] = 8
        results.append(
            evaluate_sample(
                sample,
                sam_tracker=sam_tracker,
                vggt_adapter=vggt_adapter,
                device=device,
            )
        )

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
