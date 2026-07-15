#!/usr/bin/env python3
"""Compare WMReward surprise in motion, static, and out-of-GT-motion regions."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import cv2
import decord
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualize_wmreward_patch_surprise import (
    WMREWARD_ROOT,
    compute_patch_surprise,
    install_optional_diffusers_stub,
    install_upstream_paths,
    prepare_official_input,
)


X0_ROOT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "text_noun_attention_x0_every5_step1000_physiq025_20260714/"
    "train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_"
    "init3500_save500_keepall_20260713T090024Z_step-001000_steps40_512x896_ctx08_"
    "49f_defaultnegprompt/physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed_text_noun_attention/predicted_x0"
)
GT_PATH = Path(
    "/data/gaoya/AAA_test_video/0623/testdataset/"
    "025_Solid_Mechanics_0002_perspective-center_trimmed/"
    "physicIQ_0002_clip_2p5s_3p5s.mp4"
)
REMAINING_ARCHIVE = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "yellow_block_sam_surprise_localization_20260715/remaining49f_analysis/"
    "remaining49f_patch_surprise_fp16.npz"
)
OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "motion_vs_wmreward_surprise_49f_20260715"
)
STEPS = [40, 35, 30, 25, 20, 15, 10, 5, 1]


def load_video(path: Path, *, crop_top: int, frame_count: int = 49) -> tuple[torch.Tensor, np.ndarray]:
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    indices = np.linspace(0, len(reader) - 1, frame_count).round().astype(np.int64)
    frames = reader.get_batch(indices).asnumpy()
    if crop_top:
        frames = frames[:, crop_top:]
    tensor, visual = prepare_official_input(frames, 384)
    return tensor, visual


def patch_motion(frames: np.ndarray) -> np.ndarray:
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    transition = []
    for index in range(1, len(gray)):
        flow = cv2.calcOpticalFlowFarneback(
            gray[index - 1], gray[index], None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        magnitude = np.linalg.norm(flow, axis=-1).astype(np.float32)
        transition.append(cv2.resize(magnitude, (24, 24), interpolation=cv2.INTER_AREA))
    transition = np.stack(transition)
    tubelets = []
    for token_t in range(24):
        relevant = []
        for transition_index in (2 * token_t - 1, 2 * token_t):
            if 0 <= transition_index < len(transition):
                relevant.append(transition[transition_index])
        tubelets.append(np.maximum.reduce(relevant))
    return np.stack(tubelets)


def motion_support(magnitude: np.ndarray, valid: np.ndarray, quantile: float = 0.80) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(magnitude[valid], quantile))
    raw = (magnitude >= threshold) & valid
    support = np.zeros_like(raw)
    kernel = np.ones((3, 3), np.uint8)
    for token_t in range(raw.shape[0]):
        support[token_t] = cv2.morphologyEx(
            raw[token_t].astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    return support, threshold


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def zone_stats(surprise: np.ndarray, own_motion: np.ndarray, gt_motion: np.ndarray) -> dict:
    valid = np.isfinite(surprise)
    own_motion &= valid
    own_static = valid & ~own_motion
    gt_motion = gt_motion & valid
    outside_gt = valid & ~gt_motion
    extra_motion = own_motion & outside_gt
    outside_static = outside_gt & ~own_motion
    intersection = own_motion & gt_motion
    union = own_motion | gt_motion

    def mean(mask: np.ndarray) -> float:
        return float(surprise[mask].mean()) if mask.any() else float("nan")

    extra_selector = extra_motion | outside_static
    return {
        "motion_area_ratio": float(own_motion.sum() / valid.sum()),
        "gt_motion_overlap_iou": float(intersection.sum() / max(int(union.sum()), 1)),
        "own_motion_surprise": mean(own_motion),
        "own_static_surprise": mean(own_static),
        "motion_minus_static": mean(own_motion) - mean(own_static),
        "motion_patch_auc": auc(surprise[valid], own_motion[valid]),
        "gt_motion_region_surprise": mean(gt_motion),
        "outside_gt_region_surprise": mean(outside_gt),
        "extra_motion_area_ratio": float(extra_motion.sum() / valid.sum()),
        "extra_motion_surprise": mean(extra_motion),
        "outside_gt_static_surprise": mean(outside_static),
        "extra_minus_outside_static": mean(extra_motion) - mean(outside_static),
        "extra_motion_auc_vs_outside_static": auc(
            surprise[extra_selector], extra_motion[extra_selector]
        ),
    }


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def mask_overlay(frame: np.ndarray, gt_mask: np.ndarray, extra_mask: np.ndarray) -> np.ndarray:
    gt = cv2.resize(gt_mask.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST).astype(bool)
    extra = cv2.resize(extra_mask.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST).astype(bool)
    colors = np.zeros_like(frame)
    colors[gt] = (0, 180, 255)
    colors[extra] = (255, 0, 0)
    mixed = cv2.addWeighted(frame, 0.4, colors, 0.6, 0)
    output = frame.copy()
    output[gt | extra] = mixed[gt | extra]
    return output


def render_contact(
    name: str,
    frames: np.ndarray,
    surprise: np.ndarray,
    own_motion: np.ndarray,
    gt_motion: np.ndarray,
    scale: float,
) -> Path:
    rows = []
    for frame_index in (16, 28, 36, 44):
        token_t = frame_index // 2
        extra = own_motion[token_t] & ~gt_motion[token_t]
        motion_view = mask_overlay(frames[frame_index], gt_motion[token_t], extra)
        encoded = np.clip(surprise[token_t] / scale, 0, 1)
        heat = cv2.applyColorMap((encoded * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(
            cv2.resize(heat, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB
        )
        rows.append(
            np.concatenate(
                [
                    add_header(frames[frame_index], f"{name} | frame {frame_index:02d}"),
                    add_header(motion_view, "cyan=GT motion | red=extra generated motion"),
                    add_header(heat, "WMReward patch surprise"),
                ],
                axis=1,
            )
        )
    image = np.concatenate(rows, axis=0)
    path = OUTPUT_DIR / f"{name}_motion_surprise_contact.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    install_upstream_paths()
    install_optional_diffusers_stub()
    from utils import load_vjepa_model_source

    gt_tensor, gt_frames = load_video(GT_PATH, crop_top=0)
    cwd = Path.cwd()
    os.chdir(WMREWARD_ROOT)
    try:
        encoder, target_encoder, predictor, img_size = load_vjepa_model_source("vitg384")
    finally:
        os.chdir(cwd)
    device = torch.device("cuda:0")
    gt_surprise, gt_info = compute_patch_surprise(
        gt_tensor,
        encoder.to(device).eval(),
        target_encoder.to(device).eval(),
        predictor.to(device).eval(),
        img_size=img_size,
        window_size=16,
        context_frames=8,
        stride=8,
        seed=42,
        device=device,
    )
    archive = np.load(REMAINING_ARCHIVE)
    surprise_maps = {"ground_truth": gt_surprise}
    frame_sets = {"ground_truth": gt_frames}
    for step in STEPS:
        surprise_maps[f"remaining_{step:02d}"] = archive[f"remaining_{step:02d}"].astype(np.float32)
        _, frame_sets[f"remaining_{step:02d}"] = load_video(
            X0_ROOT / f"pred_x0_remaining_{step:02d}_h264.mp4", crop_top=60
        )

    valid = np.isfinite(gt_surprise)
    motion_magnitudes = {name: patch_motion(frames) for name, frames in frame_sets.items()}
    motion_masks = {}
    thresholds = {}
    for name, magnitude in motion_magnitudes.items():
        motion_masks[name], thresholds[name] = motion_support(magnitude, valid)
    gt_motion = motion_masks["ground_truth"]
    shared_scale = float(
        np.quantile(
            np.concatenate([value[np.isfinite(value)] for value in surprise_maps.values()]), 0.99
        )
    )

    rows = []
    images = {}
    for name, surprise in surprise_maps.items():
        stats = zone_stats(surprise, motion_masks[name].copy(), gt_motion.copy())
        row = {
            "video": name,
            "motion_flow_q80_threshold": thresholds[name],
            "official_surprise": gt_info["official_surprise_mean"] if name == "ground_truth" else "",
            **stats,
        }
        rows.append(row)
        images[name] = str(
            render_contact(
                name,
                frame_sets[name],
                surprise,
                motion_masks[name],
                gt_motion,
                shared_scale,
            )
        )

    with (OUTPUT_DIR / "motion_surprise_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        OUTPUT_DIR / "motion_masks_and_gt_surprise_fp16.npz",
        gt_surprise=gt_surprise.astype(np.float16),
        **{f"motion_{name}": mask.astype(np.uint8) for name, mask in motion_masks.items()},
    )

    generated = [row for row in rows if row["video"] != "ground_truth"]
    labels = [row["video"].replace("remaining_", "") for row in generated]
    x = np.arange(len(generated))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(x, [row["motion_minus_static"] for row in generated])
    axes[0].set_title("Own motion minus static surprise")
    axes[1].bar(x, [row["extra_minus_outside_static"] for row in generated])
    axes[1].set_title("Extra motion outside GT minus static")
    axes[2].bar(x, [row["extra_motion_auc_vs_outside_static"] for row in generated])
    axes[2].set_title("Extra-motion AUROC vs outside static")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Denoising steps remaining")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generated_motion_surprise_summary.png", dpi=160)
    plt.close(fig)

    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(
            {
                "gt_video": str(GT_PATH),
                "gt_resampling": "30 source frames uniformly resampled to 49",
                "motion_definition": "per-video top 20% Farneback flow magnitude on 24x24 tubelet grid, 3x3 close",
                "extra_motion_definition": "generated own-motion mask outside GT motion mask",
                "shared_surprise_scale": [0.0, shared_scale],
                "metrics": rows,
                "visualizations": images,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
