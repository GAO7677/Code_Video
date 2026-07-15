#!/usr/bin/env python3
"""Overlay self-surprise motion proposals on the complete remaining-01 video."""

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


DEFAULT_VIDEO = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "text_noun_attention_x0_every5_step1000_physiq025_20260714/"
    "train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_"
    "init3500_save500_keepall_20260713T090024Z_step-001000_steps40_512x896_ctx08_"
    "49f_defaultnegprompt/physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed_text_noun_attention/predicted_x0/pred_x0_remaining_01_h264.mp4"
)
DEFAULT_SURPRISE = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "yellow_block_sam_surprise_localization_20260715/remaining49f_analysis/"
    "remaining49f_patch_surprise_fp16.npz"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "remaining01_self_surprise_motion_overlay_20260715"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--surprise-archive", type=Path, default=DEFAULT_SURPRISE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--motion-quantile", type=float, default=0.80)
    parser.add_argument("--title-height", type=int, default=60)
    return parser.parse_args()


def header(image: np.ndarray, text: str, height: int = 46) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def write_h264(path: Path, frames: list[np.ndarray], fps: float) -> None:
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
    fps = float(reader.get_avg_fps())
    frames = reader.get_batch(np.arange(len(reader))).asnumpy()
    if args.title_height:
        frames = frames[:, args.title_height:]
    surprise = np.load(args.surprise_archive)["remaining_01"].astype(np.float32)
    valid = np.isfinite(surprise)
    threshold = float(np.quantile(surprise[valid], args.motion_quantile))
    patch_masks = np.zeros_like(valid)
    kernel = np.ones((3, 3), np.uint8)
    for token_t in range(len(surprise)):
        if not valid[token_t].any():
            continue
        raw = surprise[token_t] >= threshold
        patch_masks[token_t] = cv2.morphologyEx(
            raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

    full_masks = np.zeros((len(frames), frames.shape[1], frames.shape[2]), dtype=np.uint8)
    overlay_frames = []
    diagnostic_frames = []
    heat_scale = float(np.quantile(surprise[valid], 0.99))
    for frame_index, frame in enumerate(frames):
        token_t = min(frame_index // 2, len(surprise) - 1)
        scored = bool(valid[token_t].any())
        if scored:
            mask = cv2.resize(
                patch_masks[token_t].astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            heat_encoded = np.clip(surprise[token_t] / heat_scale, 0, 1)
            heat = cv2.applyColorMap(
                (cv2.resize(heat_encoded, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8),
                cv2.COLORMAP_TURBO,
            )
            heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        else:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            heat = np.full_like(frame, 225)
        full_masks[frame_index] = mask
        red = np.zeros_like(frame)
        red[..., 0] = 255
        mixed = cv2.addWeighted(frame, 0.35, red, 0.65, 0)
        overlay = frame.copy()
        overlay[mask > 0] = mixed[mask > 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        status = (
            f"tubelet {token_t:02d} | threshold {threshold:.4f} | area {100.0 * mask.mean():.1f}%"
            if scored
            else "unscored context/tail"
        )
        overlay_frames.append(
            header(overlay, f"remaining-01 self-surprise motion proposal | frame {frame_index:02d} | {status}")
        )
        diagnostic_frames.append(
            np.concatenate(
                [
                    header(frame, f"remaining-01 | frame {frame_index:02d}"),
                    header(heat, f"self-surprise | scale [0,{heat_scale:.4f}]"),
                    header(overlay, f"top {100 * (1 - args.motion_quantile):.0f}% proposal overlay"),
                ],
                axis=1,
            )
        )

    overlay_path = args.output_dir / "remaining01_self_surprise_motion_overlay_h264.mp4"
    diagnostic_path = args.output_dir / "remaining01_self_surprise_motion_diagnostic_h264.mp4"
    write_h264(overlay_path, overlay_frames, fps)
    write_h264(diagnostic_path, diagnostic_frames, fps)
    np.savez_compressed(
        args.output_dir / "remaining01_self_surprise_motion_masks.npz",
        masks=full_masks,
        patch_masks=patch_masks.astype(np.uint8),
        surprise=surprise.astype(np.float16),
    )

    selected = [8, 16, 24, 32, 36, 40, 44, 47]
    contact = np.concatenate(
        [
            np.concatenate([overlay_frames[index] for index in selected[:4]], axis=1),
            np.concatenate([overlay_frames[index] for index in selected[4:]], axis=1),
        ],
        axis=0,
    )
    contact_path = args.output_dir / "remaining01_self_surprise_motion_contact.jpg"
    cv2.imwrite(str(contact_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    (args.output_dir / "result.json").write_text(
        json.dumps(
            {
                "video": str(args.video),
                "uses_gt": False,
                "uses_optical_flow": False,
                "uses_sam": False,
                "definition": "global top-20% remaining-01 self-surprise patches, 3x3 patch close",
                "motion_quantile": args.motion_quantile,
                "threshold": threshold,
                "heat_scale_q99": heat_scale,
                "scored_frame_range": [8, 47],
                "unscored_frames": list(range(0, 8)) + [48],
                "overlay_video": str(overlay_path),
                "diagnostic_video": str(diagnostic_path),
                "contact_sheet": str(contact_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
