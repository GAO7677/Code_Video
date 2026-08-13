#!/usr/bin/env python3
"""Render the exact geometry of the dual-protocol point-guidance experiment.

For every registered case/target/backend this produces three auditable 13-frame
videos:

1. source GT/pseudo-GT CoTracker point trajectory;
2. the modified forward constraint, Q(R_ctx, p_ctx) -> K(R_t, p_t);
3. the previous reverse constraint, Q(R_t, p_t) -> K(R_ctx, p_ctx).

When the same-backend Baseline exists, the script also runs CoTracker once and
renders its output trajectory against the source reference.  Missing Baselines
are deliberately left absent so the dashboard can retain a Pending slot.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from AAA_my_test import run_wan_gt_spatiotemporal_correspondence_guidance as legacy
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (
    load_cotracker,
    run_cotracker,
)
from AAA_my_test.wan_context_point_guidance.run_dual_protocol import (
    BACKENDS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TUBE_ROOT,
    LATENT_ANCHORS,
    backend_tracks,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


SEED = 47326
FFMPEG = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg")
CYAN = (41, 226, 238)
MAGENTA = (243, 92, 162)
GREEN = (90, 220, 145)
AMBER = (255, 167, 61)
WHITE = (245, 249, 252)
INK = (12, 28, 43)
MUTED = (173, 194, 209)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _target_indices(tube: legacy.FrozenTube, target: str) -> np.ndarray:
    names = [str(value) for value in tube.region_names]
    if target not in names:
        raise KeyError(f"{tube.case}: unknown target {target}; available={names}")
    index = names.index(target)
    return np.arange(
        int(tube.point_starts[index]), int(tube.point_ends[index]), dtype=np.int64
    )


def _select_source_anchors(frames: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    anchors = np.asarray(anchors, dtype=np.int64)
    if anchors.shape != (13,):
        raise RuntimeError(f"expected 13 frozen source anchors, got {anchors.shape}")
    if anchors.min() < 0 or anchors.max() >= len(frames):
        raise RuntimeError(
            f"source anchors {anchors.tolist()} exceed {len(frames)} decoded frames"
        )
    return np.asarray(frames)[anchors]


def _source_frames(tube: legacy.FrozenTube, backend: str) -> np.ndarray:
    spec = BACKENDS[backend]
    frames = np.asarray(iio.imread(tube.source_video))[:49, ..., :3]
    if backend == "firstframe_ti2v":
        resized = legacy.resize_frames(frames, spec.height, spec.width)
        return _select_source_anchors(resized, tube.anchor_source_frames)
    tensor = preprocess_video_rgb_uint8(
        frames,
        (spec.height, spec.width),
        resize_mode="cover_crop",
        cover_crop_hw=(spec.height, spec.width),
    )
    resized = (
        ((tensor.permute(1, 2, 3, 0).float() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .byte()
        .cpu()
        .numpy()
    )
    return _select_source_anchors(resized, tube.anchor_source_frames)


def _centers(
    tracks: np.ndarray, visibility: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = tracks[:, indices]
    valid_points = visibility[:, indices].astype(bool) & np.isfinite(points).all(-1)
    centers = np.full((len(points), 2), np.nan, dtype=np.float32)
    valid = valid_points.any(axis=1)
    for frame in np.flatnonzero(valid):
        centers[frame] = points[frame, valid_points[frame]].mean(axis=0)
    return centers, valid


def _draw_path(
    draw: ImageDraw.ImageDraw,
    centers: np.ndarray,
    valid: np.ndarray,
    stop: int,
    color: tuple[int, int, int],
    width: int = 4,
) -> None:
    run = [tuple(map(float, centers[index])) for index in range(stop + 1) if valid[index]]
    if len(run) > 1:
        draw.line(run, fill=color, width=width, joint="curve")


def _draw_points(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    valid: np.ndarray,
    color: tuple[int, int, int],
    prefix: str,
    filled: bool = False,
) -> None:
    for index, ((x, y), keep) in enumerate(zip(points, valid, strict=True)):
        if not keep or not np.isfinite((x, y)).all():
            continue
        radius = 6
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(
            box,
            fill=color if filled else None,
            outline=INK if filled else color,
            width=2,
        )
        draw.text((x + 8, y - 12), f"{prefix}{index}", fill=color, font=_font(13, True))


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=3)
    angle = np.arctan2(ey - sy, ex - sx)
    for delta in (-0.55, 0.55):
        px = ex - 13 * np.cos(angle + delta)
        py = ey - 13 * np.sin(angle + delta)
        draw.line((ex, ey, px, py), fill=color, width=3)


def _banner(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    anchor: int,
    canvas_width: int,
    frame_label: str,
) -> None:
    right = canvas_width - 16
    draw.rounded_rectangle((16, 14, right, 102), radius=12, fill=INK, outline=(74, 105, 128), width=2)
    draw.text((32, 25), title, fill=WHITE, font=_font(22, True))
    draw.text((32, 61), subtitle, fill=MUTED, font=_font(15))
    draw.text(
        (canvas_width - 132, 66),
        f"R{anchor:02d} · {frame_label}",
        fill=WHITE,
        font=_font(16, True),
    )


def _gt_frame(
    frame: np.ndarray,
    anchor: int,
    source_frame: int,
    tracks: np.ndarray,
    visibility: np.ndarray,
    indices: np.ndarray,
    target: str,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    centers, center_valid = _centers(tracks, visibility, indices)
    _draw_path(draw, centers, center_valid, anchor, CYAN)
    points = tracks[anchor, indices]
    valid = visibility[anchor, indices].astype(bool)
    _draw_points(draw, points, valid, CYAN, "i")
    _banner(
        draw,
        f"{target} | 13-anchor GT / pseudo-GT point trajectory",
        "CoTracker identity i is preserved across R0...R12",
        anchor,
        frame.shape[1],
        f"SRC F{source_frame:02d}",
    )
    return np.asarray(image)


def _constraint_frame(
    frame: np.ndarray,
    anchor: int,
    source_frame: int,
    tracks: np.ndarray,
    visibility: np.ndarray,
    indices: np.ndarray,
    key_times: tuple[int, ...],
    target: str,
    reverse: bool,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGB")
    future = tracks[anchor, indices]
    future_valid = visibility[anchor, indices].astype(bool)
    _draw_points(draw, future, future_valid, CYAN if reverse else GREEN, "t")
    for key_time in key_times:
        context = tracks[key_time, indices]
        context_valid = visibility[key_time, indices].astype(bool)
        _draw_points(draw, context, context_valid, MAGENTA, f"{key_time}:", filled=True)
        for point_index in range(len(indices)):
            if not (future_valid[point_index] and context_valid[point_index]):
                continue
            start = future[point_index] if reverse else context[point_index]
            end = context[point_index] if reverse else future[point_index]
            _arrow(draw, tuple(map(float, start)), tuple(map(float, end)), MAGENTA if reverse else GREEN)
    if reverse:
        title = f"{target} | PREVIOUS reverse constraint (archived)"
        subtitle = f"Q(R{anchor}, p{anchor}^i) -> K({','.join(f'R{k}' for k in key_times)}, p_ctx^i); future position is placed on the Query side"
    else:
        title = f"{target} | CURRENT forward motion lookup"
        subtitle = f"Q({','.join(f'R{k}' for k in key_times)}, p_ctx^i) -> K(R{anchor}, p{anchor}^i); attention response itself locates the future point"
    _banner(
        draw, title, subtitle, anchor, frame.shape[1], f"SRC F{source_frame:02d}"
    )
    return np.asarray(image)


def _baseline_frame(
    frame: np.ndarray,
    anchor: int,
    gt_tracks: np.ndarray,
    gt_visibility: np.ndarray,
    baseline_tracks: np.ndarray,
    baseline_visibility: np.ndarray,
    indices: np.ndarray,
    target: str,
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    gt_centers, gt_valid = _centers(gt_tracks, gt_visibility, indices)
    base_centers, base_valid = _centers(
        baseline_tracks[LATENT_ANCHORS], baseline_visibility[LATENT_ANCHORS], indices
    )
    _draw_path(draw, gt_centers, gt_valid, anchor, CYAN, 5)
    _draw_path(draw, base_centers, base_valid, anchor, AMBER, 5)
    _draw_points(
        draw,
        gt_tracks[anchor, indices],
        gt_visibility[anchor, indices].astype(bool),
        CYAN,
        "G",
    )
    _draw_points(
        draw,
        baseline_tracks[LATENT_ANCHORS[anchor], indices],
        baseline_visibility[LATENT_ANCHORS[anchor], indices].astype(bool),
        AMBER,
        "B",
        filled=True,
    )
    _banner(
        draw,
        f"{target} | BEFORE guidance: same-backend Baseline output",
        "cyan = source GT/pseudo-GT; amber = generated Baseline CoTracker trajectory",
        anchor,
        frame.shape[1],
        f"GEN F{int(LATENT_ANCHORS[anchor]):02d}",
    )
    return np.asarray(image)


def _encode(frames: list[np.ndarray], output: Path, overwrite: bool) -> None:
    if output.is_file() and output.stat().st_size > 0 and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    temporary = output.with_name(output.stem + ".tmp.mp4")
    command = [
        str(FFMPEG), "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "2",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    if return_code or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {output}")
    temporary.replace(output)


def _baseline_tracks(
    video: Path,
    cache: Path,
    query_points: np.ndarray,
    model: Any,
    device: str,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache.is_file() and not overwrite:
        with np.load(cache, allow_pickle=False) as payload:
            return payload["tracks_tn2"], payload["visibility_tn"].astype(bool)
    frames = np.asarray(iio.imread(video))[:49, ..., :3]
    tracks, visibility = run_cotracker(model, frames, query_points, device)
    _atomic_npz(
        cache,
        tracks_tn2=np.asarray(tracks, dtype=np.float32),
        visibility_tn=np.asarray(visibility, dtype=np.uint8),
    )
    return np.asarray(tracks), np.asarray(visibility, dtype=bool)


def run(args: argparse.Namespace) -> dict[str, Any]:
    backends = tuple(BACKENDS) if args.backend == "all" else (args.backend,)
    model = None
    rows: list[dict[str, Any]] = []
    try:
        for backend in backends:
            spec = BACKENDS[backend]
            manifest_path = args.output_root / backend / "task_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for case_row in manifest.get("cases", []):
                case = str(case_row["case"])
                if args.case and case != args.case:
                    continue
                tube = legacy.load_frozen_tube(args.tube_root, case)
                source_frames = _source_frames(tube, backend)
                tracks, visibility, geometry = backend_tracks(tube, spec)
                for target in case_row.get("targets", []):
                    target = str(target)
                    indices = _target_indices(tube, target)
                    output = args.output_root / "diagnostics" / backend / case / target
                    _encode(
                        [_gt_frame(source_frames[anchor], anchor, int(tube.anchor_source_frames[anchor]), tracks, visibility, indices, target) for anchor in range(13)],
                        output / "gt_13_anchor_trajectory.mp4",
                        args.overwrite,
                    )
                    _encode(
                        [_constraint_frame(source_frames[anchor], anchor, int(tube.anchor_source_frames[anchor]), tracks, visibility, indices, spec.key_times, target, False) for anchor in range(13)],
                        output / "current_forward_constraint.mp4",
                        args.overwrite,
                    )
                    _encode(
                        [_constraint_frame(source_frames[anchor], anchor, int(tube.anchor_source_frames[anchor]), tracks, visibility, indices, spec.key_times, target, True) for anchor in range(13)],
                        output / "previous_reverse_constraint.mp4",
                        args.overwrite,
                    )
                    baseline = (
                        args.output_root / backend / "generations" / case
                        / f"seed_{args.seed:05d}" / "baseline" / "generated.mp4"
                    )
                    baseline_output = output / "baseline_before_guidance_trajectory.mp4"
                    if baseline.is_file():
                        if model is None:
                            model = load_cotracker(args.device)
                        cache = output / "baseline_cotracker.npz"
                        baseline_tracks, baseline_visibility = _baseline_tracks(
                            baseline, cache, tracks[0], model, args.device, args.overwrite
                        )
                        baseline_frames = np.asarray(iio.imread(baseline))[:49, ..., :3]
                        _encode(
                            [_baseline_frame(baseline_frames[pixel], anchor, tracks, visibility, baseline_tracks, baseline_visibility, indices, target) for anchor, pixel in enumerate(LATENT_ANCHORS)],
                            baseline_output,
                            args.overwrite,
                        )
                    rows.append(
                        {
                            "backend": backend,
                            "case": case,
                            "target": target,
                            "geometry": geometry,
                            "gt_ready": (output / "gt_13_anchor_trajectory.mp4").is_file(),
                            "current_constraint_ready": (output / "current_forward_constraint.mp4").is_file(),
                            "previous_constraint_ready": (output / "previous_reverse_constraint.mp4").is_file(),
                            "baseline_trajectory_ready": baseline_output.is_file(),
                        }
                    )
                    print(f"[diagnostic] {backend}/{case}/{target}", flush=True)
    finally:
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    path = args.output_root / "diagnostics" / "manifest.json"
    if args.case and path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
        except (OSError, json.JSONDecodeError, TypeError):
            existing = []
        current_keys = {
            (row["backend"], row["case"], row["target"]) for row in rows
        }
        rows = [
            row
            for row in existing
            if (row.get("backend"), row.get("case"), row.get("target"))
            not in current_keys
        ] + rows
    rows.sort(key=lambda row: (row["backend"], row["case"], row["target"]))
    report = {"protocol": "wan_context_point_constraint_diagnostics_v1", "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tube-root", type=Path, default=DEFAULT_TUBE_ROOT)
    parser.add_argument("--backend", choices=("all", *BACKENDS), default="all")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not FFMPEG.is_file():
        raise FileNotFoundError(FFMPEG)
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"rows": len(result["rows"])}, indent=2))
