#!/usr/bin/env python3
"""Evaluate which predicted-x0 denoising snapshot localizes the extra yellow block."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
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


PREDICTED_X0_ROOT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "text_noun_attention_x0_every5_step1000_physiq025_20260714/"
    "train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_"
    "init3500_save500_keepall_20260713T090024Z_step-001000_steps40_512x896_ctx08_"
    "49f_defaultnegprompt/physicIQ_025_Solid_Mechanics_0002_perspective-center_"
    "trimmed_text_noun_attention/predicted_x0"
)
MASK_PATH = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "yellow_block_sam_surprise_localization_20260715/sam2_track/yellow_block_masks.npz"
)
OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "yellow_block_sam_surprise_localization_20260715/remaining49f_analysis"
)
REMAINING_STEPS = [40, 35, 30, 25, 20, 15, 10, 5, 1]


def load_video(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    frames = reader.get_batch(np.arange(len(reader))).asnumpy()
    frames = frames[:, 60:]
    tensor, visual = prepare_official_input(frames, 384)
    return tensor, visual


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


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def mask_to_patch_occupancy(masks: np.ndarray) -> np.ndarray:
    per_frame = np.stack(
        [cv2.resize(mask.astype(np.float32), (24, 24), interpolation=cv2.INTER_AREA) for mask in masks]
    )
    return np.stack(
        [np.maximum(per_frame[2 * index], per_frame[2 * index + 1]) for index in range(24)]
    )


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas, text, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas


def heatmap(values: np.ndarray, scale: float) -> np.ndarray:
    encoded = np.clip(values / scale, 0.0, 1.0)
    heat = cv2.applyColorMap((encoded * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(
        cv2.resize(heat, (384, 384), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_BGR2RGB,
    )


def sam_contour_for_input(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    resized = cv2.resize(mask.astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return resized.astype(bool), contours


def render_snapshot(
    remaining: int,
    visual: np.ndarray,
    surprise: np.ndarray,
    masks: np.ndarray,
    scale: float,
    output_dir: Path,
) -> Path:
    rows = []
    for frame_index in (36, 40, 44, 46):
        token_t = frame_index // 2
        original = visual[frame_index].copy()
        heat = heatmap(surprise[token_t], scale)
        overlay = cv2.addWeighted(original, 0.55, heat, 0.45, 0)
        _, contours = sam_contour_for_input(masks[frame_index])
        for panel in (original, heat, overlay):
            cv2.drawContours(panel, contours, -1, (255, 255, 255), 3)
        rows.append(
            np.concatenate(
                [
                    add_header(original, f"remaining-{remaining:02d} | x0 frame {frame_index:02d} + SAM"),
                    add_header(heat, f"WMReward surprise | tubelet {token_t:02d}"),
                    add_header(overlay, "surprise overlay + final yellow-block SAM mask"),
                ],
                axis=1,
            )
        )
    image = np.concatenate(rows, axis=0)
    path = output_dir / f"remaining_{remaining:02d}_yellow_block_surprise_contact.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    install_upstream_paths()
    install_optional_diffusers_stub()
    from utils import load_vjepa_model_source

    cwd = Path.cwd()
    os.chdir(WMREWARD_ROOT)
    try:
        encoder, target_encoder, predictor, img_size = load_vjepa_model_source("vitg384")
    finally:
        os.chdir(cwd)
    device = torch.device("cuda:0")
    encoder = encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    masks = np.load(MASK_PATH)["masks"].astype(np.uint8)
    occupancy = mask_to_patch_occupancy(masks)
    object_patch = occupancy >= 0.10
    object_t = np.where(object_patch.reshape(24, -1).any(axis=1))[0]
    maps = {}
    visuals = {}
    metadata = {}

    for remaining in REMAINING_STEPS:
        path = PREDICTED_X0_ROOT / f"pred_x0_remaining_{remaining:02d}_h264.mp4"
        tensor, visual = load_video(path)
        patch_map, info = compute_patch_surprise(
            tensor,
            encoder,
            target_encoder,
            predictor,
            img_size=img_size,
            window_size=16,
            context_frames=8,
            stride=8,
            seed=42,
            device=device,
        )
        maps[remaining] = patch_map
        visuals[remaining] = visual
        metadata[remaining] = info
        print(f"[computed] remaining-{remaining:02d}: official={info['official_surprise_mean']:.6f}")

    finite_all = np.concatenate([value[np.isfinite(value)] for value in maps.values()])
    shared_scale = float(np.quantile(finite_all, 0.99))
    rows = []
    image_paths = {}
    eval_selector = np.zeros_like(object_patch, dtype=bool)
    eval_selector[object_t] = True
    labels = object_patch[eval_selector]

    for remaining in REMAINING_STEPS:
        surprise = maps[remaining]
        scores = surprise[eval_selector]
        inside = scores[labels]
        outside = scores[~labels]
        threshold = float(np.quantile(surprise[np.isfinite(surprise)], 0.90))
        predicted = scores >= threshold
        intersection = int(np.logical_and(predicted, labels).sum())
        union = int(np.logical_or(predicted, labels).sum())
        row = {
            "remaining": remaining,
            "official_surprise": metadata[remaining]["official_surprise_mean"],
            "yellow_inside_mean": float(inside.mean()),
            "same_frames_outside_mean": float(outside.mean()),
            "inside_minus_outside": float(inside.mean() - outside.mean()),
            "inside_over_outside": float(inside.mean() / max(outside.mean(), 1.0e-8)),
            "yellow_patch_auc": binary_auc(scores, labels),
            "top10_threshold": threshold,
            "yellow_recall_at_top10": float(intersection / max(int(labels.sum()), 1)),
            "top10_precision_on_yellow": float(intersection / max(int(predicted.sum()), 1)),
            "yellow_top10_iou": float(intersection / max(union, 1)),
        }
        rows.append(row)
        image_paths[str(remaining)] = str(
            render_snapshot(remaining, visuals[remaining], surprise, masks, shared_scale, OUTPUT_DIR)
        )

    rows.sort(key=lambda item: item["yellow_patch_auc"], reverse=True)
    with (OUTPUT_DIR / "remaining_yellow_block_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        OUTPUT_DIR / "remaining49f_patch_surprise_fp16.npz",
        **{f"remaining_{key:02d}": value.astype(np.float16) for key, value in maps.items()},
        yellow_block_patch_occupancy=occupancy.astype(np.float16),
    )

    ordered = sorted(rows, key=lambda item: item["remaining"], reverse=True)
    x = np.arange(len(ordered))
    labels_x = [str(item["remaining"]) for item in ordered]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(x, [item["yellow_patch_auc"] for item in ordered], marker="o")
    axes[0].set_title("Yellow-mask patch AUROC")
    axes[1].plot(x, [item["inside_minus_outside"] for item in ordered], marker="o")
    axes[1].set_title("Inside minus outside surprise")
    axes[2].plot(x, [item["yellow_recall_at_top10"] for item in ordered], marker="o")
    axes[2].set_title("Yellow-mask recall at global top 10%")
    for axis in axes:
        axis.set_xticks(x, labels_x)
        axis.set_xlabel("Denoising steps remaining")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "remaining_metric_curves.png", dpi=160)
    plt.close(fig)

    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(
            {
                "mask_path": str(MASK_PATH),
                "evaluated_object_tubelets": object_t.tolist(),
                "object_patch_definition": "SAM occupancy >= 0.10 after 24x24 area resize and 2-frame max",
                "shared_surprise_scale": [0.0, shared_scale],
                "ranking": rows,
                "visualizations": image_paths,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
