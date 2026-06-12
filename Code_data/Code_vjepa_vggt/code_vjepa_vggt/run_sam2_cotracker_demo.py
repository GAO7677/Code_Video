from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker, build_motion_prompt_box
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior


COTRACKER_REPO_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
if str(COTRACKER_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_REPO_ROOT))

from cotracker.predictor import CoTrackerPredictor  # type: ignore
from cotracker.utils.visualizer import Visualizer  # type: ignore


def infer_caption_from_path(video_path: Path) -> str:
    stem = video_path.stem.replace("_", " ").strip()
    return stem if stem else "object motion"


def tensor_video_from_frames(frames_thwc: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames_thwc).permute(0, 3, 1, 2)[None].float().to(device)


def read_video_cv2(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video with cv2: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"decoded zero frames from video: {video_path}")
    return np.stack(frames, axis=0)


def draw_query_preview(
    frame_hwc: np.ndarray,
    *,
    mask_hw: np.ndarray,
    prompt_box_xyxy: np.ndarray,
    query_points_xy: np.ndarray,
    output_path: Path,
) -> None:
    rgb = Image.fromarray(frame_hwc.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    overlay_arr[mask_hw > 0] = np.array([44, 162, 95, 90], dtype=np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    out = Image.alpha_composite(rgb, overlay)
    draw = ImageDraw.Draw(out)

    if np.any(prompt_box_xyxy > 0):
        x0, y0, x1, y1 = [float(v) for v in prompt_box_xyxy.tolist()]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 140, 0, 255), width=3)
        draw.text((x0 + 2, max(2, y0 + 2)), "sam_prompt", fill=(255, 140, 0, 255))

    for idx, point in enumerate(query_points_xy.tolist()):
        x, y = float(point[0]), float(point[1])
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(17, 17, 17, 255), width=3)
        draw.text((x + 6, y - 6), f"q{idx}", fill=(17, 17, 17, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument(
        "--checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-frame", choices=("first", "last"), default="last")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    device = torch.device(args.device if not args.device.startswith("cuda") else "cuda")

    frames_thwc = read_video_cv2(video_path)
    frames_tchw_01 = np.transpose(frames_thwc.astype(np.float32) / 255.0, (0, 3, 1, 2))

    prompt_frame_idx = 0 if args.prompt_frame == "first" else max(int(frames_tchw_01.shape[0]) - 1, 0)
    prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)

    sam_tracker = SAM2MotionTracker(device=str(device), enable_text_prompt=False)
    caption = infer_caption_from_path(video_path)
    sam_out = sam_tracker.track(
        frames_tchw_01,
        prompt_frame_idx=prompt_frame_idx,
        prompt_box_xyxy=prompt_box_xyxy,
        caption=caption,
    )

    query_points_xy, prior_source = build_vggt_query_prior(
        sam_out.masks_thw,
        sam_out.boxes_t4,
        num_queries=int(args.num_queries),
    )
    if query_points_xy.shape[0] <= 0:
        raise RuntimeError("no query points sampled from SAM2 output")

    queries = np.concatenate(
        [
            np.zeros((query_points_xy.shape[0], 1), dtype=np.float32),
            query_points_xy.astype(np.float32),
        ],
        axis=1,
    )
    queries_torch = torch.from_numpy(queries).unsqueeze(0).to(device=device)

    video_torch = tensor_video_from_frames(frames_thwc, device=device)
    cotracker = CoTrackerPredictor(
        checkpoint=str(args.checkpoint),
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

    base_name = video_path.stem
    vis = Visualizer(save_dir=str(output_dir), pad_value=120, linewidth=3, fps=int(args.fps))
    vis.visualize(
        video_torch,
        pred_tracks,
        pred_visibility,
        filename=f"{base_name}__sam2_mask_points__cotracker3_offline",
        query_frame=0,
    )

    preview_path = output_dir / f"{base_name}__sam2_query_preview.png"
    draw_query_preview(
        frames_thwc[0],
        mask_hw=sam_out.masks_thw[0],
        prompt_box_xyxy=sam_out.prompt_box_xyxy.astype(np.float32),
        query_points_xy=query_points_xy.astype(np.float32),
        output_path=preview_path,
    )

    metadata = {
        "video_path": str(video_path),
        "caption": caption,
        "device": str(device),
        "prompt_frame_idx": int(prompt_frame_idx),
        "prompt_box_xyxy": sam_out.prompt_box_xyxy.astype(np.float32).tolist(),
        "sam_prompt_mode": sam_out.prompt_mode,
        "sam_prior_source": prior_source,
        "num_queries": int(query_points_xy.shape[0]),
        "query_frame": 0,
        "query_points_xy": query_points_xy.astype(np.float32).tolist(),
        "tracks_shape": list(pred_tracks.shape),
        "visibility_shape": list(pred_visibility.shape),
        "preview_image": str(preview_path),
        "overlay_video": str(output_dir / f"{base_name}__sam2_mask_points__cotracker3_offline.mp4"),
    }
    with open(output_dir / f"{base_name}__sam2_cotracker_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
