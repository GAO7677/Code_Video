#!/usr/bin/env python3
"""Track the extra yellow block in the final predicted-x0 video with SAM2."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import decord
import numpy as np

from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker


DEFAULT_VIDEO = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "text_noun_attention_x0_every5_step1000_physiq025_20260714/"
    "train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_"
    "init3500_save500_keepall_20260713T090024Z_step-001000_steps40_512x896_ctx08_"
    "49f_defaultnegprompt/physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed_text_noun_attention/predicted_x0/pred_x0_remaining_01_h264.mp4"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "yellow_block_sam_surprise_localization_20260715/sam2_track"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-frame", type=int, default=36)
    parser.add_argument("--prompt-box", default="630,65,735,165")
    parser.add_argument("--title-height", type=int, default=60)
    parser.add_argument("--mode", choices=["image", "video"], default="image")
    return parser.parse_args()


def yellow_candidate_box(frame_rgb: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.asarray([10, 70, 55]), np.asarray([42, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 100:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        # Reject the thin ceiling stripe and prefer compact object components.
        if height < 12 or width < 12:
            continue
        candidates.append((area, x, y, width, height))
    if not candidates:
        return None
    _, x, y, width, height = max(candidates)
    pad_x = max(6, int(round(width * 0.18)))
    pad_y = max(6, int(round(height * 0.18)))
    return np.asarray(
        [
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(frame_rgb.shape[1], x + width + pad_x),
            min(frame_rgb.shape[0], y + height + pad_y),
        ],
        dtype=np.float32,
    )


def write_h264(path: Path, frames: list[np.ndarray], fps: float = 15.0) -> None:
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        intermediate = Path(temp_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open {intermediate}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        ffmpeg = shutil.which("ffmpeg") or "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(intermediate), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
            check=True,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = decord.VideoReader(str(args.video), ctx=decord.cpu(0))
    frames = reader.get_batch(np.arange(len(reader))).asnumpy()
    if args.title_height:
        frames = frames[:, args.title_height:]
    box = np.asarray([float(value) for value in args.prompt_box.split(",")], dtype=np.float32)
    tracker = SAM2MotionTracker(
        device=args.device,
        segment_len=16,
        enable_text_prompt=False,
    )
    frames_tchw = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    if args.mode == "video":
        result = tracker.track(frames_tchw, args.prompt_frame, box)
        masks = result.masks_thw.astype(np.uint8)
        masks[: args.prompt_frame] = 0
    else:
        masks = np.zeros(frames.shape[:3], dtype=np.uint8)
        for frame_index in range(args.prompt_frame, len(frames)):
            candidate = yellow_candidate_box(frames[frame_index])
            if candidate is None:
                continue
            mask = tracker._refine_box_to_mask(frames_tchw[frame_index], candidate)
            if mask is not None:
                masks[frame_index] = mask.astype(np.uint8)

    boxes = []
    overlays = []
    for frame_index, (frame, mask) in enumerate(zip(frames, masks)):
        rows, cols = np.where(mask > 0)
        if rows.size:
            tracked_box = [int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)]
        else:
            tracked_box = [0, 0, 0, 0]
        boxes.append(
            {
                "frame": frame_index,
                "area": int(mask.sum()),
                "box_xyxy": tracked_box,
            }
        )
        overlay = frame.copy()
        if mask.any():
            yellow = np.zeros_like(frame)
            yellow[..., :2] = 255
            mixed = cv2.addWeighted(frame, 0.45, yellow, 0.55, 0)
            overlay[mask > 0] = mixed[mask > 0]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        canvas = cv2.copyMakeBorder(
            overlay, 46, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            canvas,
            f"SAM2 extra yellow block | frame {frame_index:02d} | area {int(mask.sum())}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        overlays.append(canvas)

    np.savez_compressed(args.output_dir / "yellow_block_masks.npz", masks=masks)
    (args.output_dir / "track.json").write_text(
        json.dumps(
            {
                "video": str(args.video),
                "prompt_frame": args.prompt_frame,
                "prompt_box_xyxy": box.tolist(),
                "pre_prompt_masks_forced_zero": True,
                "mode": args.mode,
                "frames": boxes,
            },
            indent=2,
        )
        + "\n"
    )
    write_h264(args.output_dir / "yellow_block_sam2_overlay_h264.mp4", overlays, fps=15.0)
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
