#!/usr/bin/env python3
"""Cache CoTracker trajectories and render GT/candidate overlay videos.

The frozen GT-STC scalar metrics did not persist the raw candidate tracks.
This post-processing step reruns the same CoTracker entry point, stores the
native source horizon (up to 49 frames) and full 49-frame generated output
once per video, and then renders one target-specific overlay for every
registered dashboard card.

Colors are intentionally stable across every case:

* cyan: source-GT CoTracker points/centroid/trail;
* amber: generated-video CoTracker points/centroid/trail;
* red: fewer than the metric's minimum number of candidate points visible.

The source overlay contains only the cyan trajectory.  Generated overlays
draw both coordinate-aligned trajectories on the generated frames so visual
appearance changes cannot be mistaken for motion changes.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from AAA_my_test.gt_stc_guidance_results_dashboard import MODES, _float_tag
from AAA_my_test.run_wan_gt_spatiotemporal_correspondence_guidance import resize_frames
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (
    load_cotracker,
    run_cotracker,
)


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)
DEFAULT_FFMPEG = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg")
SEED = 47326
FRAMES = 49
GT_COLOR = (41, 226, 238)
CANDIDATE_COLOR = (255, 167, 61)
LOST_COLOR = (243, 74, 88)
WHITE = (244, 248, 251)
INK = (15, 28, 43)


@dataclass(frozen=True)
class TrackTask:
    case: str
    name: str
    video: Path
    cache: Path
    resize_to_model: bool = False
    expected_frames: int = FRAMES


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def registered_variants(root: Path, seed: int) -> dict[tuple[str, str], list[str]]:
    screen = read_json(root / "screening" / f"seed_{seed:05d}" / "baseline_eligibility.json")
    final = read_json(
        root / "final_analysis" / f"seed_{seed:05d}" / "frozen_validation_report.json"
    )
    trigger_modes = {str(value) for value in final.get("trigger_modes", [])}
    result: dict[tuple[str, str], list[str]] = {}
    for job in screen.get("eligible_jobs", []):
        case = str(job["case"])
        for raw_target in job.get("targets", []):
            target = str(raw_target)
            variants = ["baseline"]
            for mode in MODES:
                if mode in trigger_modes:
                    variants.append(f"{mode}__{target}__lambda{_float_tag(0.05)}")
                variants.append(f"{mode}__{target}__lambda{_float_tag(0.1)}")
                if mode in trigger_modes:
                    variants.append(f"{mode}__{target}__lambda{_float_tag(0.2)}")
            result[(case, target)] = variants
    return result


def target_point_indices(tube: dict[str, np.ndarray], target: str) -> np.ndarray:
    names = [str(value) for value in tube["region_names"].tolist()]
    if target not in names:
        raise KeyError(f"unknown target {target}; available={names}")
    index = names.index(target)
    start = int(tube["point_starts"][index])
    end = int(tube["point_ends"][index])
    return np.arange(start, end, dtype=np.int64)


def trajectory_centers(
    tracks_tn2: np.ndarray,
    visibility_tn: np.ndarray,
    point_indices: np.ndarray,
    minimum_visible: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks_tn2, dtype=np.float32)[:, point_indices]
    visible = np.asarray(visibility_tn, dtype=bool)[:, point_indices]
    finite = visible & np.isfinite(tracks).all(axis=-1)
    centers = np.full((len(tracks), 2), np.nan, dtype=np.float32)
    valid = finite.sum(axis=1) >= int(minimum_visible)
    for frame in np.flatnonzero(valid):
        centers[frame] = tracks[frame, finite[frame]].mean(axis=0)
    return centers, valid


def load_frames(
    video: Path, resize_to_model: bool = False, expected_frames: int = FRAMES
) -> np.ndarray:
    frames = np.asarray(iio.imread(video))[: int(expected_frames), ..., :3]
    if len(frames) != int(expected_frames):
        raise RuntimeError(
            f"expected {expected_frames} frames, got {len(frames)}: {video}"
        )
    frames = frames.astype(np.uint8, copy=False)
    return resize_frames(frames) if resize_to_model else frames


def track_cache_valid(task: TrackTask) -> bool:
    if not task.cache.is_file():
        return False
    try:
        with np.load(task.cache, allow_pickle=False) as payload:
            tracks = payload["tracks_tn2"]
            height = int(payload["frame_height"])
            width = int(payload["frame_width"])
    except (OSError, KeyError, ValueError):
        return False
    return bool(
        tracks.shape[0] == task.expected_frames
        and (not task.resize_to_model or (height, width) == (704, 1280))
    )


def save_tracks(
    task: TrackTask,
    query_points: np.ndarray,
    model: Any,
    device: str,
    overwrite: bool,
) -> None:
    if task.cache.is_file() and not overwrite:
        return
    frames = load_frames(task.video, task.resize_to_model, task.expected_frames)
    tracks, visibility = run_cotracker(model, frames, query_points, device)
    tracks = np.asarray(tracks, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=bool)
    if tracks.shape[:2] != visibility.shape or tracks.shape[0] != task.expected_frames:
        raise RuntimeError(
            f"invalid CoTracker result for {task.video}: {tracks.shape}, {visibility.shape}"
        )
    atomic_npz(
        task.cache,
        tracks_tn2=tracks,
        visibility_tn=visibility.astype(np.uint8),
        query_points_n2=np.asarray(query_points, dtype=np.float32),
        frame_height=np.int32(frames.shape[1]),
        frame_width=np.int32(frames.shape[2]),
        source_resized_to_model=np.uint8(task.resize_to_model),
    )


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _valid_points(
    tracks: np.ndarray, visibility: np.ndarray, frame: int, indices: np.ndarray
) -> np.ndarray:
    points = tracks[frame, indices]
    valid = visibility[frame, indices].astype(bool) & np.isfinite(points).all(axis=-1)
    return points[valid]


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    centers: np.ndarray,
    valid: np.ndarray,
    frame: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    run: list[tuple[float, float]] = []
    for index in range(frame + 1):
        if valid[index]:
            run.append(tuple(float(value) for value in centers[index]))
        elif len(run) >= 2:
            draw.line(run, fill=color, width=width, joint="curve")
            run = []
        else:
            run = []
    if len(run) >= 2:
        draw.line(run, fill=color, width=width, joint="curve")


def _draw_points(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int,
    filled: bool,
) -> None:
    for x, y in points:
        box = (float(x - radius), float(y - radius), float(x + radius), float(y + radius))
        if filled:
            draw.ellipse(box, fill=color, outline=INK, width=1)
        else:
            draw.ellipse(box, outline=color, width=3)


def overlay_frame(
    frame_rgb: np.ndarray,
    frame_index: int,
    target: str,
    point_indices: np.ndarray,
    gt_tracks: np.ndarray,
    gt_visibility: np.ndarray,
    candidate_tracks: np.ndarray | None,
    candidate_visibility: np.ndarray | None,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    gt_centers, gt_valid = trajectory_centers(
        gt_tracks, gt_visibility, point_indices, minimum_visible=1
    )
    _draw_polyline(draw, gt_centers, gt_valid, frame_index, GT_COLOR, width=4)
    gt_points = _valid_points(gt_tracks, gt_visibility, frame_index, point_indices)
    _draw_points(draw, gt_points, GT_COLOR, radius=6, filled=False)
    if gt_valid[frame_index]:
        x, y = gt_centers[frame_index]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=GT_COLOR, width=4)

    status = "SOURCE GT"
    candidate_lost = False
    if candidate_tracks is not None and candidate_visibility is not None:
        minimum = min(4, len(point_indices))
        candidate_centers, candidate_valid = trajectory_centers(
            candidate_tracks, candidate_visibility, point_indices, minimum_visible=minimum
        )
        _draw_polyline(
            draw, candidate_centers, candidate_valid, frame_index, CANDIDATE_COLOR, width=4
        )
        candidate_points = _valid_points(
            candidate_tracks, candidate_visibility, frame_index, point_indices
        )
        _draw_points(draw, candidate_points, CANDIDATE_COLOR, radius=5, filled=True)
        if candidate_valid[frame_index]:
            x, y = candidate_centers[frame_index]
            draw.ellipse(
                (x - 8, y - 8, x + 8, y + 8),
                fill=CANDIDATE_COLOR,
                outline=INK,
                width=2,
            )
        else:
            candidate_lost = True
        status = "TRACK LOST" if candidate_lost else "TRACKED"

    draw.rounded_rectangle((18, 16, 574, 102), radius=12, fill=(12, 28, 43), outline=(83, 112, 134), width=2)
    draw.text((34, 27), f"{target}  |  F{frame_index:02d}", font=_font(22, True), fill=WHITE)
    draw.text((34, 62), "GT", font=_font(17, True), fill=GT_COLOR)
    if candidate_tracks is not None:
        draw.text((80, 62), "vs GENERATED", font=_font(17, True), fill=CANDIDATE_COLOR)
    draw.text(
        (422, 62),
        status,
        font=_font(17, True),
        fill=LOST_COLOR if candidate_lost else WHITE,
    )
    return np.asarray(image)


def render_overlay(
    video: Path,
    output: Path,
    target: str,
    point_indices: np.ndarray,
    gt_track_path: Path,
    candidate_track_path: Path | None,
    ffmpeg: Path,
    overwrite: bool,
    resize_to_model: bool = False,
    expected_frames: int = FRAMES,
) -> None:
    if output.is_file() and output.stat().st_size > 0 and not overwrite:
        return
    frames = load_frames(video, resize_to_model, expected_frames)
    with np.load(gt_track_path, allow_pickle=False) as gt:
        gt_tracks = np.asarray(gt["tracks_tn2"], dtype=np.float32)
        gt_visibility = np.asarray(gt["visibility_tn"], dtype=bool)
    candidate_tracks = candidate_visibility = None
    if candidate_track_path is not None:
        with np.load(candidate_track_path, allow_pickle=False) as candidate:
            candidate_tracks = np.asarray(candidate["tracks_tn2"], dtype=np.float32)
            candidate_visibility = np.asarray(candidate["visibility_tn"], dtype=bool)
    height, width = frames.shape[1:3]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    command = [
        str(ffmpeg), "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_index, frame in enumerate(frames):
            rendered = overlay_frame(
                frame,
                frame_index,
                target,
                point_indices,
                gt_tracks,
                gt_visibility,
                candidate_tracks,
                candidate_visibility,
            )
            process.stdin.write(np.ascontiguousarray(rendered).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    if return_code != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {output}")
    temporary.replace(output)


def expand_frozen_anchor_tracks(
    tracks_an2: np.ndarray,
    visibility_an: np.ndarray,
    output_frames: int = FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the exact 13 metric anchors onto the 49-frame generated timeline."""
    tracks = np.asarray(tracks_an2, dtype=np.float32)
    visibility = np.asarray(visibility_an, dtype=bool)
    if tracks.shape[0] != 13 or visibility.shape != tracks.shape[:2]:
        raise ValueError(f"expected 13-anchor tube, got {tracks.shape}/{visibility.shape}")
    anchor_frames = np.arange(13, dtype=np.int64) * 4
    frame_axis = np.arange(output_frames, dtype=np.float32)
    expanded = np.full((output_frames, tracks.shape[1], 2), np.nan, dtype=np.float32)
    expanded_visibility = np.zeros((output_frames, tracks.shape[1]), dtype=bool)
    for point in range(tracks.shape[1]):
        valid_anchor = visibility[:, point] & np.isfinite(tracks[:, point]).all(axis=-1)
        valid_indices = np.flatnonzero(valid_anchor)
        if not len(valid_indices):
            continue
        valid_frames = anchor_frames[valid_indices]
        for coordinate in range(2):
            expanded[:, point, coordinate] = np.interp(
                frame_axis,
                valid_frames.astype(np.float32),
                tracks[valid_indices, point, coordinate],
                left=np.nan,
                right=np.nan,
            )
        # In-between frames are visible only when both bracketing metric anchors
        # are visible. Exact anchors retain their recorded visibility.
        for anchor in range(13):
            expanded_visibility[anchor_frames[anchor], point] = valid_anchor[anchor]
        for anchor in range(12):
            if valid_anchor[anchor] and valid_anchor[anchor + 1]:
                start = int(anchor_frames[anchor]) + 1
                end = int(anchor_frames[anchor + 1])
                expanded_visibility[start:end, point] = True
    return expanded, expanded_visibility


def frozen_generated_reference(root: Path, case: str) -> Path:
    output = root / "trajectory_tracks" / case / "frozen_gt_generated_timeline.npz"
    with np.load(root / "gt_tubes" / case / "tube.npz", allow_pickle=False) as tube:
        tracks, visibility = expand_frozen_anchor_tracks(
            tube["tracks_tn2"], tube["visibility_tn"]
        )
    atomic_npz(
        output,
        tracks_tn2=tracks,
        visibility_tn=visibility.astype(np.uint8),
        source="frozen 13-anchor tube linearly mapped to generated F00-F48",
        frame_height=np.int32(704),
        frame_width=np.int32(1280),
    )
    return output


def iter_track_tasks(
    root: Path, seed: int, matrix: dict[tuple[str, str], list[str]]
) -> tuple[list[TrackTask], dict[str, Path]]:
    tasks: dict[tuple[str, str], TrackTask] = {}
    source_videos: dict[str, Path] = {}
    for case, _target in matrix:
        if case not in source_videos:
            manifest = read_json(root / "gt_tubes" / case / "manifest.json")
            source_videos[case] = Path(str(manifest["source_video"]))
            source_frame_count = min(FRAMES, int(manifest["source_frame_count"]))
            tasks[(case, "source")] = TrackTask(
                case,
                "source",
                source_videos[case],
                root / "trajectory_tracks" / case / "source.npz",
                True,
                source_frame_count,
            )
        for variant in matrix[(case, _target)]:
            key = (case, variant)
            tasks[key] = TrackTask(
                case,
                variant,
                root / "generations" / case / f"seed_{seed:05d}" / variant / "generated.mp4",
                root / "trajectory_tracks" / case / f"seed_{seed:05d}" / f"{variant}.npz",
            )
    return list(tasks.values()), source_videos


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    matrix = registered_variants(root, args.seed)
    tasks, source_videos = iter_track_tasks(root, args.seed, matrix)
    if args.case:
        matrix = {key: value for key, value in matrix.items() if key[0] == args.case}
        tasks = [task for task in tasks if task.case == args.case]
        source_videos = {key: value for key, value in source_videos.items() if key == args.case}
    for task in tasks:
        if not task.video.is_file():
            raise FileNotFoundError(task.video)
    missing_tasks = [
        task for task in tasks if args.overwrite_tracks or not track_cache_valid(task)
    ]
    if args.max_track_videos is not None:
        missing_tasks = missing_tasks[: args.max_track_videos]
    model = None
    try:
        if missing_tasks:
            model = load_cotracker(args.device)
            for index, task in enumerate(missing_tasks, 1):
                with np.load(root / "gt_tubes" / task.case / "tube.npz", allow_pickle=False) as tube:
                    queries = np.asarray(tube["query_points_n2"], dtype=np.float32)
                print(f"[track {index}/{len(missing_tasks)}] {task.case}/{task.name}", flush=True)
                save_tracks(task, queries, model, args.device, args.overwrite_tracks)
    finally:
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    task_by_key = {(task.case, task.name): task for task in tasks}
    overlay_count = 0
    expected_overlay_count = 0
    for (case, target), variants in matrix.items():
        with np.load(root / "gt_tubes" / case / "tube.npz", allow_pickle=False) as tube:
            indices = target_point_indices(dict(tube), target)
        source_task = task_by_key[(case, "source")]
        generated_gt_reference = frozen_generated_reference(root, case)
        source_output = root / "trajectory_overlays" / case / f"source__{target}.mp4"
        expected_overlay_count += 1
        if source_task.cache.is_file():
            render_overlay(
                source_videos[case], source_output, target, indices,
                source_task.cache, None, args.ffmpeg, args.overwrite_overlays, True,
                source_task.expected_frames,
            )
        overlay_count += int(source_output.is_file() and source_output.stat().st_size > 0)
        for variant in variants:
            expected_overlay_count += 1
            task = task_by_key[(case, variant)]
            output = (
                root / "trajectory_overlays" / case / f"seed_{args.seed:05d}"
                / f"{variant}__{target}.mp4"
            )
            if source_task.cache.is_file() and task.cache.is_file():
                render_overlay(
                    task.video, output, target, indices, generated_gt_reference,
                    task.cache, args.ffmpeg, args.overwrite_overlays, False,
                )
            overlay_count += int(output.is_file() and output.stat().st_size > 0)
    report = {
        "protocol": "gt_stc_cotracker_trajectory_overlay_v1",
        "seed": int(args.seed),
        "case_count": len({case for case, _target in matrix}),
        "target_count": len(matrix),
        "track_cache_count": sum(track_cache_valid(task) for task in tasks),
        "track_cache_expected": len(tasks),
        "overlay_count": overlay_count,
        "overlay_expected": expected_overlay_count,
        "legend": {"source_gt": "cyan", "generated": "amber", "track_lost": "red"},
    }
    summary = root / "trajectory_overlays" / f"seed_{args.seed:05d}" / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--case")
    parser.add_argument("--max-track-videos", type=int)
    parser.add_argument("--overwrite-tracks", action="store_true")
    parser.add_argument("--overwrite-overlays", action="store_true")
    args = parser.parse_args()
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)
    return args


if __name__ == "__main__":
    run(parse_args())
