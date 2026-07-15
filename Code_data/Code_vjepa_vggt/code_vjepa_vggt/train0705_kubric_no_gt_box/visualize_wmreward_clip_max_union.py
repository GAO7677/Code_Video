#!/usr/bin/env python3
"""Visualize the spatial union of each scored clip's maximum-surprise patch."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visualize_wmreward_patch_surprise import prepare_official_input, sample_video


RESULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "wmreward_patch_surprise_30f_physiq025_x0_remaining35_vs01_vs_gt_20260714"
)


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        canvas,
        text,
        (9, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def maximum_in_clip(
    surprise: np.ndarray,
    frame_start: int,
    frame_end: int,
    tubelet_size: int,
) -> dict:
    token_start = frame_start // tubelet_size
    token_end = (frame_end + 1) // tubelet_size
    local = surprise[token_start:token_end]
    flat_index = int(np.nanargmax(local))
    local_t, y, x = np.unravel_index(flat_index, local.shape)
    token_t = token_start + local_t
    return {
        "surprise": float(surprise[token_t, y, x]),
        "token_t_y_x": [int(token_t), int(y), int(x)],
        "sampled_frame_indices": [int(token_t * tubelet_size), int(token_t * tubelet_size + 1)],
    }


def render_video_union(
    name: str,
    visual: np.ndarray,
    surprise: np.ndarray,
    windows: list[dict],
    tubelet_size: int,
    sampled_source_indices: list[int],
    output_dir: Path,
) -> dict:
    grid_h, grid_w = surprise.shape[1:]
    height, width = visual.shape[1:3]
    patch_h, patch_w = height // grid_h, width // grid_w
    union = np.zeros((grid_h, grid_w), dtype=bool)
    clip_results = []
    clip_panels = []
    colors = [(255, 64, 64), (64, 220, 255)]

    for clip_index, window in enumerate(windows):
        start, end = window["target_frame_range"]
        maximum = maximum_in_clip(surprise, start, end, tubelet_size)
        token_t, y, x = maximum["token_t_y_x"]
        union[y, x] = True
        frame_index = maximum["sampled_frame_indices"][0]
        source_frames = [sampled_source_indices[index] for index in maximum["sampled_frame_indices"]]
        maximum["source_frame_indices"] = source_frames
        maximum["input_pixel_box_xyxy"] = [
            x * patch_w,
            y * patch_h,
            (x + 1) * patch_w,
            (y + 1) * patch_h,
        ]
        maximum["target_frame_range"] = [start, end]
        maximum["official_window_surprise"] = window["official_chunk_surprise"]
        clip_results.append(maximum)

        panel = visual[frame_index].copy()
        x0, y0, x1, y1 = maximum["input_pixel_box_xyxy"]
        overlay = panel.copy()
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), colors[clip_index], -1)
        panel = cv2.addWeighted(panel, 0.62, overlay, 0.38, 0)
        cv2.rectangle(panel, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 3)
        clip_panels.append(
            add_header(
                panel,
                f"c{clip_index} f{start:02d}-{end:02d} | S={maximum['surprise']:.3f} | t{token_t} y{y} x{x}",
            )
        )

    reference = np.median(visual.astype(np.float32), axis=0).astype(np.uint8)
    union_large = cv2.resize(
        union.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    union_mask = np.full_like(reference, 245)
    union_mask[union_large] = (255, 80, 80)
    union_overlay = reference.copy()
    tint = np.zeros_like(reference)
    tint[:] = (255, 64, 64)
    union_overlay[union_large] = cv2.addWeighted(
        reference, 0.45, tint, 0.55, 0
    )[union_large]
    contours, _ = cv2.findContours(
        union_large.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(union_overlay, contours, -1, (255, 255, 255), 3)
    clip_panels.extend(
        [
            add_header(union_mask, f"union mask | {int(union.sum())} patches"),
            add_header(union_overlay, "union overlay | temporal median"),
        ]
    )
    contact = np.concatenate(clip_panels, axis=1)
    output_path = output_dir / f"{name}_clip_max_surprise_union.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    return {
        "clips": clip_results,
        "union_patch_count": int(union.sum()),
        "union_patch_yx": np.argwhere(union).tolist(),
        "image_path": str(output_path),
    }


def main() -> None:
    payload = json.loads((RESULT_ROOT / "result.json").read_text())
    arrays = np.load(RESULT_ROOT / "patch_surprise_maps_fp16.npz")
    output_dir = RESULT_ROOT / "clip_max_surprise_union"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, metadata in payload["videos"].items():
        crop_top = int(metadata["crop_top"])
        frames, sampled_indices, _ = sample_video(
            Path(metadata["path"]),
            int(payload["method"]["num_frames"]),
            crop_top,
        )
        _, visual = prepare_official_input(frames, 384)
        surprise = arrays[name].astype(np.float32)
        surprise[surprise < 0] = np.nan
        report[name] = render_video_union(
            name,
            visual,
            surprise,
            metadata["windows"],
            int(metadata["tubelet_size"]),
            sampled_indices,
            output_dir,
        )
        print(f"[rendered] {name}: {report[name]['image_path']}")

    (output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
