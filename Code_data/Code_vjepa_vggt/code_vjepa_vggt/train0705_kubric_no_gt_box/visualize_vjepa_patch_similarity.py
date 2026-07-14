#!/usr/bin/env python3
"""Overlay same-position V-JEPA patch cosine similarity on source videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader, cpu


DEFAULT_INPUT_DIR = (
    "/data/gaoya/agent-data/outputs/AAA_physv/"
    "vjepa_similarity_physiq025_x0_remaining35_vs01_vs_gt_20260714"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.48)
    parser.add_argument("--cosine-min", type=float, default=0.2)
    parser.add_argument("--cosine-max", type=float, default=1.0)
    parser.add_argument("--panel-width", type=int, default=896)
    parser.add_argument("--panel-height", type=int, default=512)
    parser.add_argument(
        "--ffmpeg", default="/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg"
    )
    return parser.parse_args()


def load_sampled_content_frames(
    metadata: dict,
    *,
    panel_hw: tuple[int, int],
) -> list[np.ndarray]:
    vr = VideoReader(metadata["path"], ctx=cpu(0))
    indices = np.asarray(metadata["sampled_frame_indices"], dtype=np.int64)
    frames = vr.get_batch(indices).asnumpy()
    crop_top = int(metadata["crop_top"])
    if crop_top:
        frames = frames[:, crop_top:, :, :]
    panel_h, panel_w = panel_hw
    return [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (panel_w, panel_h))
        for frame in frames
    ]


def similarity_color(similarity: np.ndarray, cosine_min: float, cosine_max: float) -> np.ndarray:
    clipped = np.clip(similarity, cosine_min, cosine_max)
    dissimilarity = 1.0 - (clipped - cosine_min) / (cosine_max - cosine_min)
    return cv2.applyColorMap(np.round(dissimilarity * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def add_header(image: np.ndarray, lines: list[str], height: int = 76) -> np.ndarray:
    output = cv2.copyMakeBorder(image, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (12, 27 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return output


def patch_rect(
    patch_y: int,
    patch_x: int,
    *,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    grid_h, grid_w = grid_hw
    image_h, image_w = image_hw
    x0 = round(patch_x * image_w / grid_w)
    x1 = round((patch_x + 1) * image_w / grid_w) - 1
    y0 = round(patch_y * image_h / grid_h)
    y1 = round((patch_y + 1) * image_h / grid_h) - 1
    return x0, y0, x1, y1


def mark_patch(
    image: np.ndarray,
    patch_y: int,
    patch_x: int,
    *,
    grid_hw: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x0, y0, x1, y1 = patch_rect(
        patch_y,
        patch_x,
        grid_hw=grid_hw,
        image_hw=image.shape[:2],
    )
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


def write_h264(path: Path, frames: list[np.ndarray], fps: float, ffmpeg: str) -> None:
    if not frames:
        raise ValueError("cannot encode an empty frame list")
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("video frame dimensions changed")
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with code {return_code}: {path}")


def main() -> None:
    args = parse_args()
    if not args.cosine_min < args.cosine_max:
        raise ValueError("--cosine-min must be smaller than --cosine-max")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "patch_similarity_overlay"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((input_dir / "result.json").read_text(encoding="utf-8"))
    feature_archive = np.load(input_dir / "vjepa_features_fp16.npz")
    names = list(report["feature_shapes"])
    source_frames = {
        name: load_sampled_content_frames(
            report["inputs"][name],
            panel_hw=(args.panel_height, args.panel_width),
        )
        for name in names
    }

    pair_results = []
    for name_a, name_b in combinations(names, 2):
        feature_a = feature_archive[name_a].astype(np.float32)
        feature_b = feature_archive[name_b].astype(np.float32)
        norm_a = np.linalg.norm(feature_a, axis=-1)
        norm_b = np.linalg.norm(feature_b, axis=-1)
        similarity = np.sum(feature_a * feature_b, axis=-1) / np.maximum(norm_a * norm_b, 1e-8)
        token_t, grid_h, grid_w = similarity.shape
        minimum_flat = int(np.argmin(similarity))
        minimum_t, minimum_y, minimum_x = np.unravel_index(minimum_flat, similarity.shape)
        minimum_value = float(similarity[minimum_t, minimum_y, minimum_x])
        pair_name = f"{name_a}_vs_{name_b}"
        pair_dir = output_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        np.save(pair_dir / "patch_cosine_t24_h24_w24.npy", similarity.astype(np.float32))

        overlay_frames = []
        minimum_frame = None
        for temporal_index in range(token_t):
            # Tubelet size is 2; use the second frame as the visual representative.
            sampled_index = min(temporal_index * 2 + 1, len(source_frames[name_a]) - 1)
            heat = cv2.resize(
                similarity_color(
                    similarity[temporal_index], args.cosine_min, args.cosine_max
                ),
                (args.panel_width, args.panel_height),
                interpolation=cv2.INTER_NEAREST,
            )
            panels = []
            local_y, local_x = np.unravel_index(
                int(np.argmin(similarity[temporal_index])), (grid_h, grid_w)
            )
            for name in (name_a, name_b):
                base = source_frames[name][sampled_index].copy()
                panel = cv2.addWeighted(base, 1.0 - args.overlay_alpha, heat, args.overlay_alpha, 0)
                mark_patch(
                    panel,
                    int(local_y),
                    int(local_x),
                    grid_hw=(grid_h, grid_w),
                    color=(0, 255, 255),
                    thickness=2,
                )
                if temporal_index == minimum_t:
                    mark_patch(
                        panel,
                        int(minimum_y),
                        int(minimum_x),
                        grid_hw=(grid_h, grid_w),
                        color=(255, 255, 255),
                        thickness=4,
                    )
                panels.append(add_header(panel, [name, f"sampled frame {sampled_index:02d}"]))
            combined = np.concatenate(panels, axis=1)
            combined = add_header(
                combined,
                [
                    f"same-position V-JEPA patch cosine | tubelet {temporal_index:02d}/{token_t - 1:02d}",
                    f"yellow: frame minimum | white: global minimum | frame min={similarity[temporal_index].min():.4f}",
                ],
            )
            overlay_frames.append(combined)
            if temporal_index == minimum_t:
                minimum_frame = combined.copy()

        video_path = pair_dir / "patch_cosine_overlay_h264.mp4"
        write_h264(video_path, overlay_frames, args.fps, args.ffmpeg)
        assert minimum_frame is not None
        minimum_path = pair_dir / "global_minimum_patch.jpg"
        cv2.imwrite(str(minimum_path), minimum_frame)

        x0, y0, x1, y1 = patch_rect(
            int(minimum_y),
            int(minimum_x),
            grid_hw=(grid_h, grid_w),
            image_hw=(args.panel_height, args.panel_width),
        )
        pair_result = {
            "pair": pair_name,
            "similarity_shape": list(similarity.shape),
            "global_minimum": {
                "cosine": minimum_value,
                "temporal_token": int(minimum_t),
                "sampled_frame_pair": [int(minimum_t * 2), int(minimum_t * 2 + 1)],
                "patch_y": int(minimum_y),
                "patch_x": int(minimum_x),
                "normalized_center_xy": [
                    float((minimum_x + 0.5) / grid_w),
                    float((minimum_y + 0.5) / grid_h),
                ],
                "panel_pixel_rect_xyxy": [x0, y0, x1, y1],
            },
            "overlay_video": str(video_path),
            "minimum_image": str(minimum_path),
            "raw_patch_cosine": str(pair_dir / "patch_cosine_t24_h24_w24.npy"),
        }
        pair_results.append(pair_result)
        print(f"[pair] {pair_name}: min={minimum_value:.6f} at t={minimum_t}, y={minimum_y}, x={minimum_x}")

    output_report = {
        "description": "Per-token same-position cosine from saved V-JEPA patch features.",
        "color_scale": {
            "cosine_min": args.cosine_min,
            "cosine_max": args.cosine_max,
            "meaning": "warmer colors indicate lower cosine similarity",
        },
        "marker_legend": {
            "yellow": "minimum-similarity patch in the current tubelet",
            "white": "global minimum-similarity patch over all tubelets",
        },
        "pairs": pair_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output_report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
