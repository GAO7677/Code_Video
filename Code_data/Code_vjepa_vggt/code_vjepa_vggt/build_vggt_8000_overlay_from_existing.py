#!/usr/bin/env python3
"""Build 8,000-point RGB overlays from an existing VGGT world-point render.

This is a CPU-only recovery path for a completed VGGT run.  The existing
``vggt_world_points.mp4`` is already pixel-aligned with the source window and
contains the normalized XYZ colorization produced by the dense world-point
prediction.  We sample source pixels from that render and overlay their colors
on the original RGB frames without loading VGGT again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader, cpu

from code_vjepa_vggt.visualize_vggt_0717_manifest_cases import (
    add_header,
    read_prefix,
    safe_name,
    write_json,
    write_mp4,
)


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/vggt_0717_train10_context8_prefix49"
)


def overlay_from_colorized_world_video(
    frames: np.ndarray,
    colorized_world: np.ndarray,
    *,
    sample_count: int,
    context_frames: int,
    seed: int,
) -> np.ndarray:
    raw = np.asarray(frames, dtype=np.uint8)
    world = np.asarray(colorized_world, dtype=np.uint8)
    frame_count = min(int(raw.shape[0]), int(world.shape[0]))
    output: list[np.ndarray] = []
    for frame_id in range(frame_count):
        image = raw[frame_id].copy()
        colors = world[frame_id]
        if colors.shape[:2] != image.shape[:2]:
            colors = cv2.resize(colors, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        valid = np.any(colors > 3, axis=-1)
        flat_valid = np.flatnonzero(valid.reshape(-1))
        if not flat_valid.size:
            flat_valid = np.arange(image.shape[0] * image.shape[1], dtype=np.int64)
        rng = np.random.default_rng(int(seed) + frame_id)
        count = min(int(sample_count), int(flat_valid.size))
        chosen = rng.choice(flat_valid, size=count, replace=False)
        ys, xs = np.unravel_index(chosen, image.shape[:2])
        sampled_colors = colors[ys, xs]
        for x, y, color_rgb in zip(xs.tolist(), ys.tolist(), sampled_colors.tolist()):
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            cv2.circle(image, (int(x), int(y)), 1, color_bgr, -1, cv2.LINE_AA)
        output.append(image)
    return add_header(
        np.stack(output, axis=0),
        f"VGGT world points · {int(sample_count):,} samples/frame",
        context_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--mode", choices=("context8", "prefix49"), default="prefix49")
    parser.add_argument("--sample-count", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    rows = selection["rows"]
    frame_count = 8 if args.mode == "context8" else 49
    completed = 0
    for index, row in enumerate(rows):
        case_id = str(row["case_id"])
        case_dir = root / args.mode / "cases" / f"{index:02d}_{safe_name(case_id)}"
        world_path = case_dir / "vggt_world_points.mp4"
        if not world_path.is_file():
            raise FileNotFoundError(f"missing existing VGGT world-point video: {world_path}")
        frames, _, fps, source_count = read_prefix(Path(str(row["video"])), frame_count)
        world_reader = VideoReader(str(world_path), ctx=cpu(0))
        world_indices = np.arange(min(frame_count, len(world_reader)), dtype=np.int64)
        world_frames = world_reader.get_batch(world_indices).asnumpy()
        output_path = case_dir / f"vggt_world_points_overlay_{int(args.sample_count)}.mp4"
        overlay = overlay_from_colorized_world_video(
            frames,
            world_frames,
            sample_count=int(args.sample_count),
            context_frames=8,
            seed=int(args.seed) + index,
        )
        write_mp4(output_path, overlay, fps)

        result_path = case_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        result.update(
            {
                "status": "ok",
                "mode": args.mode,
                "frame_count": int(frame_count),
                "context_frames": 8,
                "source_video": str(row["video"]),
                "case_id": case_id,
                "family_key": str(row.get("family_key", "")),
                "caption": str(row.get("caption", "")),
                "source_frames": int(source_count),
                "fps": float(fps),
                "world_points_sample_count_per_frame": int(args.sample_count),
                "world_points_overlay_source": "existing pixel-aligned vggt_world_points.mp4",
            }
        )
        videos = dict(result.get("videos", {}))
        videos.update(
            {
                "input_window": str((case_dir / "input_window.mp4").relative_to(root)),
                "vggt_tracks": str((case_dir / "vggt_tracks.mp4").relative_to(root)),
                "vggt_depth": str((case_dir / "vggt_depth.mp4").relative_to(root)),
                "vggt_world_points": str(world_path.relative_to(root)),
                "world_points_8000_overlay": str(output_path.relative_to(root)),
            }
        )
        result["videos"] = videos
        write_json(result_path, result)
        completed += 1
        print(f"[{args.mode}] overlay {completed}/{len(rows)} {case_id}", flush=True)
    print(f"wrote {completed} overlays under {root / args.mode}", flush=True)


if __name__ == "__main__":
    main()
