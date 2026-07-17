#!/usr/bin/env python3
"""Measure V-JEPA latent prediction error in motion and static regions."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import decord
import numpy as np
import torch

from visualize_wmreward_patch_surprise import (
    WMREWARD_ROOT,
    compute_patch_surprise,
    install_optional_diffusers_stub,
    install_upstream_paths,
    prepare_official_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--motion-quantile", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_video(path: Path, count: int) -> tuple[torch.Tensor, np.ndarray, list[int]]:
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    indices = np.linspace(0, len(reader) - 1, count).round().astype(np.int64)
    frames = reader.get_batch(indices).asnumpy()
    tensor, visual = prepare_official_input(frames, 384)
    return tensor, visual, indices.tolist()


def patch_motion(frames: np.ndarray, tubelet_count: int) -> np.ndarray:
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    transitions = []
    for index in range(1, len(gray)):
        flow = cv2.calcOpticalFlowFarneback(
            gray[index - 1], gray[index], None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        transitions.append(
            cv2.resize(
                np.linalg.norm(flow, axis=-1).astype(np.float32),
                (24, 24),
                interpolation=cv2.INTER_AREA,
            )
        )
    transitions = np.stack(transitions)
    output = []
    for token_t in range(tubelet_count):
        selected = [
            transitions[i]
            for i in (2 * token_t - 1, 2 * token_t)
            if 0 <= i < len(transitions)
        ]
        output.append(np.maximum.reduce(selected) if selected else np.zeros((24, 24)))
    return np.stack(output)


def add_header(frame: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        frame, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (0, 0, 0), 2, cv2.LINE_AA,
    )
    return canvas


def write_h264(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        intermediate = Path(temp_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {intermediate}")
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        ffmpeg = shutil.which("ffmpeg") or "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(intermediate),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
            check=True,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    install_upstream_paths()
    install_optional_diffusers_stub()
    from utils import load_vjepa_model_source

    video_tensor, frames, frame_indices = load_video(args.input_video, args.num_frames)
    cwd = Path.cwd()
    os.chdir(WMREWARD_ROOT)
    try:
        encoder, target_encoder, predictor, img_size = load_vjepa_model_source("vitg384")
    finally:
        os.chdir(cwd)
    device = torch.device(args.device)
    surprise, model_info = compute_patch_surprise(
        video_tensor,
        encoder.to(device).eval(),
        target_encoder.to(device).eval(),
        predictor.to(device).eval(),
        img_size=img_size,
        window_size=16,
        context_frames=8,
        stride=8,
        seed=args.seed,
        device=device,
    )

    valid = np.isfinite(surprise)
    magnitude = patch_motion(frames, surprise.shape[0])
    threshold = float(np.quantile(magnitude[valid], args.motion_quantile))
    motion = (magnitude >= threshold) & valid
    kernel = np.ones((3, 3), np.uint8)
    for token_t in range(motion.shape[0]):
        motion[token_t] = cv2.morphologyEx(
            motion[token_t].astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    static = valid & ~motion

    global_loss = float(surprise[valid].mean())
    motion_loss = float(surprise[motion].mean())
    static_loss = float(surprise[static].mean())
    motion_area = float(motion.sum() / valid.sum())
    motion_contribution = float(surprise[motion].sum() / surprise[valid].sum())
    weighted_reconstruction = motion_area * motion_loss + (1.0 - motion_area) * static_loss
    metrics = {
        "input_video": str(args.input_video),
        "vjepa_model": "facebook/vjepa2-vitg-fpc64-384 original predictor checkpoint",
        "weight_path": "/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
        "metric": "1 - cosine similarity between predictor and target-encoder latent tokens",
        "frame_indices": frame_indices,
        "patch_grid_t_h_w": list(surprise.shape),
        "motion_quantile": args.motion_quantile,
        "motion_flow_threshold": threshold,
        "valid_patch_count": int(valid.sum()),
        "motion_patch_count": int(motion.sum()),
        "static_patch_count": int(static.sum()),
        "motion_area_ratio": motion_area,
        "global_patch_loss": global_loss,
        "motion_region_loss": motion_loss,
        "static_region_loss": static_loss,
        "motion_minus_static": motion_loss - static_loss,
        "motion_to_static_ratio": motion_loss / max(static_loss, 1.0e-12),
        "motion_loss_contribution_ratio": motion_contribution,
        "global_minus_static": global_loss - static_loss,
        "weighted_reconstruction_check": weighted_reconstruction,
        "official_window_loss": model_info["official_surprise_mean"],
        "note": "V-JEPA reconstructs/predicts latent targets; it does not decode RGB pixels.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "patch_maps.npz",
        surprise=surprise.astype(np.float16),
        motion_magnitude=magnitude.astype(np.float16),
        motion_mask=motion.astype(np.uint8),
        valid_mask=valid.astype(np.uint8),
    )

    scale = float(np.quantile(surprise[valid], 0.99))
    visual_frames = []
    tubelet = int(model_info["tubelet_size"])
    for frame_index, frame in enumerate(frames):
        token_t = min(frame_index // tubelet, surprise.shape[0] - 1)
        score = np.nan_to_num(surprise[token_t], nan=0.0)
        heat = cv2.applyColorMap(
            (np.clip(score / max(scale, 1.0e-8), 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        heat = cv2.cvtColor(
            cv2.resize(heat, (384, 384), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2RGB,
        )
        mask = cv2.resize(
            motion[token_t].astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        overlay = cv2.addWeighted(frame, 0.55, heat, 0.45, 0.0)
        outlined = frame.copy()
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(outlined, contours, -1, (255, 64, 0), 2)
        row = np.concatenate(
            [
                add_header(frame, f"input | frame {frame_index:02d}"),
                add_header(overlay, f"V-JEPA latent error | mean={np.nanmean(score):.4f}"),
                add_header(outlined, "motion support (orange contour)"),
            ],
            axis=1,
        )
        visual_frames.append(row)
    write_h264(args.output_dir / "vjepa_reconstruction_motion_overlay_h264.mp4", visual_frames, 15.0)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
