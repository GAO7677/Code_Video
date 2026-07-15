#!/usr/bin/env python3
"""Overlay self-surprise motion candidates at several thresholds on a full video."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import decord
import numpy as np


VIDEO = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "text_noun_attention_x0_every5_step1000_physiq025_20260714/"
    "train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_"
    "init3500_save500_keepall_20260713T090024Z_step-001000_steps40_512x896_ctx08_"
    "49f_defaultnegprompt/physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed_text_noun_attention/predicted_x0/pred_x0_remaining_01_h264.mp4"
)
SURPRISE_ARCHIVE = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "yellow_block_sam_surprise_localization_20260715/remaining49f_analysis/"
    "remaining49f_patch_surprise_fp16.npz"
)
MOTION_ARCHIVE = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "motion_vs_wmreward_surprise_49f_20260715/motion_masks_and_gt_surprise_fp16.npz"
)
OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "remaining01_self_surprise_motion_thresholds_20260715"
)
QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def header(frame: np.ndarray, text: str, height: int = 46) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        frame, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def write_h264(path: Path, frames: list[np.ndarray], fps: float = 15.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def overlay_candidate(
    frame: np.ndarray,
    patch_mask: np.ndarray | None,
    patch_surprise: np.ndarray | None,
    scale: float,
) -> np.ndarray:
    if patch_mask is None or patch_surprise is None:
        return frame.copy()
    mask = cv2.resize(
        patch_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    normalized = np.clip(patch_surprise / scale, 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(
        cv2.resize(heat, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_BGR2RGB,
    )
    mixed = cv2.addWeighted(frame, 0.38, heat, 0.62, 0)
    output = frame.copy()
    output[mask] = mixed[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 255, 255), 2)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reader = decord.VideoReader(str(VIDEO), ctx=decord.cpu(0))
    frames = reader.get_batch(np.arange(len(reader))).asnumpy()[:, 60:]
    surprise = np.load(SURPRISE_ARCHIVE)["remaining_01"].astype(np.float32)
    own_motion = np.load(MOTION_ARCHIVE)["motion_remaining_01"].astype(bool)
    valid = np.isfinite(surprise)
    scale = float(np.quantile(surprise[valid], 0.99))
    records = []
    rendered: dict[float, list[np.ndarray]] = {}

    for quantile in QUANTILES:
        threshold = float(np.quantile(surprise[valid], quantile))
        predicted = (surprise >= threshold) & valid
        intersection = int((predicted & own_motion).sum())
        union = int((predicted | own_motion).sum())
        precision = intersection / max(int(predicted.sum()), 1)
        recall = intersection / max(int(own_motion.sum()), 1)
        iou = intersection / max(union, 1)
        frames_out = []
        for frame_index, frame in enumerate(frames):
            token_t = frame_index // 2
            if token_t < surprise.shape[0] and valid[token_t].any():
                candidate = predicted[token_t]
                patch_map = surprise[token_t]
                status = (
                    f"q{int(quantile * 100)} | threshold {threshold:.4f} | "
                    f"selected {candidate.mean() * 100:.1f}% | frame {frame_index:02d}"
                )
            else:
                candidate = None
                patch_map = None
                status = f"q{int(quantile * 100)} | context/unscored | frame {frame_index:02d}"
            visual = overlay_candidate(frame, candidate, patch_map, scale)
            frames_out.append(header(visual, status))
        output_path = OUTPUT_DIR / f"remaining01_self_surprise_q{int(quantile * 100)}_h264.mp4"
        write_h264(output_path, frames_out)
        rendered[quantile] = frames_out
        records.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "selected_patch_ratio": float(predicted.sum() / valid.sum()),
                "motion_precision": precision,
                "motion_recall": recall,
                "motion_iou": iou,
                "video_path": str(output_path),
            }
        )

    comparison_paths = []
    for group_name, group_quantiles in (
        ("q10_to_q60_3x2", QUANTILES[:6]),
        ("q70_to_q95_2x2", QUANTILES[6:]),
    ):
        comparison = []
        for frame_index in range(len(frames)):
            panels = []
            for quantile in group_quantiles:
                panel = rendered[quantile][frame_index]
                panel = cv2.resize(panel, (448, 279), interpolation=cv2.INTER_AREA)
                panels.append(panel)
            rows = [
                np.concatenate(panels[start : start + 2], axis=1)
                for start in range(0, len(panels), 2)
            ]
            comparison.append(np.concatenate(rows, axis=0))
        comparison_path = OUTPUT_DIR / f"remaining01_self_surprise_comparison_{group_name}_h264.mp4"
        write_h264(comparison_path, comparison)
        comparison_paths.append(str(comparison_path))

    with (OUTPUT_DIR / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(
            {
                "video": str(VIDEO),
                "surprise": "remaining-01 49-frame WMReward patch self-surprise",
                "motion_reference": "remaining-01 top-20% Farneback flow support",
                "shared_heat_scale": [0.0, scale],
                "thresholds": records,
                "comparison_videos": comparison_paths,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
