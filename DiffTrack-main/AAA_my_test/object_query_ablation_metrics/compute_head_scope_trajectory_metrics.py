#!/usr/bin/env python3
"""Incrementally track Head-Scope ablations and measure real trajectory change.

The primary ranking is object-normalized CoTracker center ADE versus the
same-seed no-intervention Baseline.  Pixel MAE is deliberately not used.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.object_query_ablation_metrics.compute_head_scope_baseline_metrics import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    HEAD_SCOPES,
    MULTICASE_BASELINE_ROOT,
    PHYSICIQ67_BASELINE_ROOT,
    atomic_json,
    collect_candidates,
    file_signature,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (  # noqa: E402
    load_cotracker,
    run_cotracker,
)


FRAME_COUNT = 49
HEIGHT = 704
WIDTH = 1280
DELTA_FRAMES = 4
DEFAULT_OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics/"
    "head_scope_trajectory"
)
TRAJECTORY_DEFINITION = {
    "name": "CoTracker object trajectory impact",
    "reference": "same-seed no-intervention Baseline",
    "primary_metric": "target_center_ade_norm",
    "ranking_score": "100 * mean_selected_objects(center_ADE_norm)",
    "center_definition": (
        "per-frame median of at least four visible CoTracker query points "
        "belonging to the object"
    ),
    "center_ade_formula": "mean_t ||c_abl(t)-c_base(t)||_2 / D0",
    "center_fde_formula": "||c_abl(t*)-c_base(t*)||_2 / D0",
    "velocity_formula": (
        "mean_t ||(c_abl(t+4)-c_abl(t))/4 - "
        "(c_base(t+4)-c_base(t))/4||_2 / D0"
    ),
    "pck_formula": (
        "mean_(t,p) 1[||p_abl(t)-p_base(t)|| < alpha*D0], "
        "alpha in {0.05,0.10,0.20}"
    ),
    "normalizer": "F00 object-mask bounding-box diagonal D0",
    "quality_gate": (
        "at least min_valid_frames common center frames and common/reference "
        "center coverage >= min_coverage; failed records are N/A and unranked"
    ),
    "interpretation": (
        "larger means stronger object-trajectory change versus Baseline; it "
        "does not mean lower generation quality or greater simulator-GT error"
    ),
}
TRACK_LOSS_DEFINITION = {
    "name": "CoTracker target track loss",
    "reference": "same-seed no-intervention Baseline",
    "object_score": "100 * (1 - common_center_coverage)",
    "target_mean_score": "mean_selected_objects(object_track_loss_score_0_100)",
    "target_worst_score": "max_selected_objects(object_track_loss_score_0_100)",
    "ranking_score": "target_worst_track_loss_score_0_100",
    "direction": "larger means more selected-object center frames are not jointly trackable",
    "interpretation": (
        "tracker-specific observability loss; it is available even when trajectory "
        "ADE is N/A, but it does not by itself prove that the rendered object disappeared"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--head-scopes", nargs="+", choices=HEAD_SCOPES, default=HEAD_SCOPES)
    parser.add_argument("--min-valid-frames", type=int, default=4)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="debug: track at most N variants")
    return parser.parse_args()


def rounded(value: float | None, digits: int = 8) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def load_video_frames(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    while len(frames) < FRAME_COUNT:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (HEIGHT, WIDTH):
            frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != FRAME_COUNT:
        raise RuntimeError(f"expected {FRAME_COUNT} frames, got {len(frames)}: {path}")
    return np.stack(frames), fps


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def signature_text(signature: dict[str, int]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def locate_baseline(case: str, seed: int) -> Path:
    candidates = (
        DEFAULT_BASELINE_ROOT / case / f"seed_{seed:05d}" / "generated.mp4",
        PHYSICIQ67_BASELINE_ROOT / case / f"seed_{seed:05d}" / "generated.mp4",
        MULTICASE_BASELINE_ROOT / case / f"seed_{seed:05d}" / "generated.mp4",
    )
    baseline = next((path for path in candidates if path.is_file()), None)
    if baseline is None:
        raise FileNotFoundError(
            "no seed-matched Baseline; checked "
            + ", ".join(str(path) for path in candidates)
        )
    return baseline


def load_track_cache(
    path: Path, signature: dict[str, int], point_count: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            tracks = arrays["tracks"].astype(np.float32)
            visibility = arrays["visibility"].astype(bool)
            cached_signature = str(arrays["video_signature"].item())
        if (
            tracks.shape != (FRAME_COUNT, point_count, 2)
            or visibility.shape != (FRAME_COUNT, point_count)
            or cached_signature != signature_text(signature)
            or not np.isfinite(tracks).all()
        ):
            return None
        return tracks, visibility
    except (OSError, KeyError, ValueError):
        return None


def resolve_frozen_baseline_inputs(seed_dir: Path) -> tuple[Path, Path]:
    """Resolve the frozen track bundle locally or from candidate manifests.

    Newer Stage-3 outputs reuse the immutable track bundle from the earlier
    temporal-tube experiment and record its absolute path in every manifest,
    instead of copying the bundle into each large result directory.
    """
    local_tracks = seed_dir / "frozen_baseline_tracks" / "tracks.npz"
    local_manifest = seed_dir / "frozen_baseline_tracks" / "manifest.json"
    if local_tracks.is_file() and local_manifest.is_file():
        return local_tracks.resolve(), local_manifest.resolve()

    referenced: set[Path] = set()
    for candidate_manifest in sorted(seed_dir.glob("*/manifest.json")):
        try:
            payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            raw_path = str(payload.get("tracks_npz") or "").strip()
            if raw_path:
                referenced.add(Path(raw_path).expanduser().resolve())
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not referenced:
        raise FileNotFoundError("missing local or manifest-referenced frozen Baseline tracks")
    if len(referenced) != 1:
        raise RuntimeError(
            "candidate manifests reference multiple frozen track bundles: "
            + ", ".join(str(path) for path in sorted(referenced))
        )
    tracks_path = next(iter(referenced))
    manifest_path = tracks_path.parent / "manifest.json"
    if not tracks_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing referenced frozen Baseline tracks or manifest: {tracks_path}"
        )
    return tracks_path, manifest_path.resolve()


def validate_frozen_baseline_inputs(seed_dir: Path) -> tuple[bool, str]:
    """Validate the variable-object frozen query/track bundle used downstream."""
    try:
        frozen_path, manifest_path = resolve_frozen_baseline_inputs(seed_dir)
        with np.load(frozen_path, allow_pickle=False) as arrays:
            tracks = arrays["tracks"]
            visibility = arrays["visibility"]
            query_points = arrays["query_points"]
            region_names = [str(value) for value in arrays["region_names"].tolist()]
            starts = arrays["point_starts"].astype(int).tolist()
            ends = arrays["point_ends"].astype(int).tolist()
        point_count = len(query_points)
        if point_count <= 0 or query_points.shape != (point_count, 2):
            return False, f"invalid query_points shape: {query_points.shape}"
        if tracks.shape != (FRAME_COUNT, point_count, 2):
            return False, f"invalid tracks shape: {tracks.shape}"
        if visibility.shape != (FRAME_COUNT, point_count):
            return False, f"invalid visibility shape: {visibility.shape}"
        if not np.isfinite(tracks).all() or not np.isfinite(query_points).all():
            return False, "tracks/query_points contain non-finite values"
        if not region_names or not (
            len(region_names) == len(starts) == len(ends)
        ):
            return False, "invalid region slice metadata"
        if starts[0] != 0 or ends[-1] != point_count:
            return False, "region slices do not span all query points"
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if start < 0 or end <= start or end > point_count:
                return False, f"invalid region slice {index}: [{start}, {end})"
            if index and start != ends[index - 1]:
                return False, f"non-contiguous region slice {index}: starts at {start}"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        region_cache = Path(str(manifest["query_cache_dir"])) / "regions.npz"
        with np.load(region_cache, allow_pickle=False) as arrays:
            masks = arrays["masks_rhw"]
        if masks.ndim != 3 or masks.shape[0] < len(region_names):
            return False, f"invalid region mask shape: {masks.shape}"
        if any(not np.asarray(mask).any() for mask in masks[: len(region_names)]):
            return False, "one or more F00 region masks are empty"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid frozen Baseline inputs: {exc}"
    return True, f"eligible ({len(region_names)} objects, {point_count} query points)"


def object_centers(
    tracks: np.ndarray,
    visibility: np.ndarray,
    object_slices: dict[str, slice],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    centers: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    for name, part in object_slices.items():
        center = np.full((FRAME_COUNT, 2), np.nan, dtype=np.float32)
        use_frame = np.zeros(FRAME_COUNT, dtype=bool)
        for frame_index in range(FRAME_COUNT):
            points = tracks[frame_index, part]
            use = visibility[frame_index, part] & np.isfinite(points).all(axis=1)
            if int(use.sum()) >= 4:
                center[frame_index] = np.median(points[use], axis=0)
                use_frame[frame_index] = True
        centers[name] = center
        valid[name] = use_frame
    return centers, valid


def object_trajectory_metrics(
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    candidate_center: np.ndarray,
    candidate_center_valid: np.ndarray,
    baseline_tracks: np.ndarray,
    baseline_visibility: np.ndarray,
    baseline_center: np.ndarray,
    baseline_center_valid: np.ndarray,
    part: slice,
    diagonal: float,
    min_valid_frames: int,
    min_coverage: float,
) -> dict[str, Any]:
    common_center = candidate_center_valid & baseline_center_valid
    baseline_valid_count = int(baseline_center_valid.sum())
    common_count = int(common_center.sum())
    coverage = common_count / max(baseline_valid_count, 1)
    quality_pass = common_count >= min_valid_frames and coverage >= min_coverage
    track_loss_score = 100.0 * (1.0 - min(max(coverage, 0.0), 1.0))

    center_distance = np.linalg.norm(candidate_center - baseline_center, axis=-1)
    center_values = center_distance[common_center]
    last_frame = int(np.where(common_center)[0][-1]) if common_count else None

    velocity_valid = (
        common_center[:-DELTA_FRAMES] & common_center[DELTA_FRAMES:]
    )
    candidate_velocity = (
        candidate_center[DELTA_FRAMES:] - candidate_center[:-DELTA_FRAMES]
    ) / DELTA_FRAMES
    baseline_velocity = (
        baseline_center[DELTA_FRAMES:] - baseline_center[:-DELTA_FRAMES]
    ) / DELTA_FRAMES
    velocity_error = np.linalg.norm(candidate_velocity - baseline_velocity, axis=-1)
    velocity_values = velocity_error[velocity_valid]

    point_valid = (
        candidate_visibility[:, part]
        & baseline_visibility[:, part]
        & np.isfinite(candidate_tracks[:, part]).all(axis=-1)
        & np.isfinite(baseline_tracks[:, part]).all(axis=-1)
    )
    point_distance = np.linalg.norm(
        candidate_tracks[:, part] - baseline_tracks[:, part], axis=-1
    )
    point_values = point_distance[point_valid]

    return {
        "quality_pass": bool(quality_pass),
        "baseline_center_valid_frames": baseline_valid_count,
        "common_center_valid_frames": common_count,
        "common_center_coverage": rounded(coverage),
        "track_retention_score_0_100": rounded(100.0 - track_loss_score),
        "track_loss_score_0_100": rounded(track_loss_score),
        "last_common_visible_frame": last_frame,
        "center_ade_px": rounded(float(center_values.mean()) if center_values.size else None),
        "center_ade_norm": rounded(
            float(center_values.mean() / diagonal) if center_values.size else None
        ),
        "center_fde_px": rounded(
            float(center_distance[last_frame]) if last_frame is not None else None
        ),
        "center_fde_norm": rounded(
            float(center_distance[last_frame] / diagonal)
            if last_frame is not None
            else None
        ),
        "velocity_vector_error_px_per_frame": rounded(
            float(velocity_values.mean()) if velocity_values.size else None
        ),
        "velocity_vector_error_norm_per_frame": rounded(
            float(velocity_values.mean() / diagonal) if velocity_values.size else None
        ),
        "velocity_valid_count": int(velocity_valid.sum()),
        "point_ade_px": rounded(float(point_values.mean()) if point_values.size else None),
        "point_ade_norm": rounded(
            float(point_values.mean() / diagonal) if point_values.size else None
        ),
        "point_valid_count": int(point_valid.sum()),
        "pck_normalized": {
            str(alpha): rounded(float(np.mean(point_values < alpha * diagonal)))
            if point_values.size
            else None
            for alpha in (0.05, 0.10, 0.20)
        },
        "series": {
            "center_distance_px": [
                rounded(float(value), 5) if is_valid else None
                for value, is_valid in zip(center_distance, common_center, strict=True)
            ],
            "velocity_vector_error_px_per_frame": [
                rounded(float(value), 5) if is_valid else None
                for value, is_valid in zip(velocity_error, velocity_valid, strict=True)
            ],
        },
    }


def bbox_diagonal(mask: np.ndarray) -> float:
    y, x = np.where(mask)
    if not len(x):
        raise RuntimeError("empty F00 object mask")
    return float(math.hypot(x.max() - x.min() + 1, y.max() - y.min() + 1))


def draw_track_panel(
    frame_rgb: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    centers: dict[str, np.ndarray],
    center_valid: dict[str, np.ndarray],
    object_slices: dict[str, slice],
    frame_index: int,
    reference_centers: dict[str, np.ndarray] | None = None,
    reference_valid: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    colors = (
        (36, 170, 245),
        (232, 188, 42),
        (88, 205, 102),
        (205, 95, 210),
        (72, 205, 225),
        (225, 120, 72),
    )
    for object_index, (name, part) in enumerate(object_slices.items()):
        color = colors[object_index % len(colors)]
        for point_index in range(part.start or 0, part.stop or 0):
            for previous in range(max(0, frame_index - 20), frame_index):
                if visibility[previous, point_index] and visibility[previous + 1, point_index]:
                    p0 = tuple(np.rint(tracks[previous, point_index]).astype(int))
                    p1 = tuple(np.rint(tracks[previous + 1, point_index]).astype(int))
                    cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
            if visibility[frame_index, point_index]:
                point = tuple(np.rint(tracks[frame_index, point_index]).astype(int))
                cv2.circle(canvas, point, 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)
        valid_indices = np.where(center_valid[name][: frame_index + 1])[0]
        if len(valid_indices) > 1:
            path = np.rint(centers[name][valid_indices]).astype(np.int32)
            cv2.polylines(canvas, [path], False, color, 4, cv2.LINE_AA)
        if center_valid[name][frame_index]:
            center = tuple(np.rint(centers[name][frame_index]).astype(int))
            cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 16, 3)
        if reference_centers is not None and reference_valid is not None:
            ref_indices = np.where(reference_valid[name][: frame_index + 1])[0]
            if len(ref_indices) > 1:
                ref_path = np.rint(reference_centers[name][ref_indices]).astype(np.int32)
                cv2.polylines(canvas, [ref_path], False, (210, 60, 210), 2, cv2.LINE_AA)
    return canvas


def render_overlay(
    path: Path,
    variant_id: str,
    baseline_frames: np.ndarray,
    candidate_frames: np.ndarray,
    baseline_tracks: np.ndarray,
    baseline_visibility: np.ndarray,
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    object_slices: dict[str, slice],
    target_score: float | None,
    fps: float,
) -> None:
    baseline_centers, baseline_valid = object_centers(
        baseline_tracks, baseline_visibility, object_slices
    )
    candidate_centers, candidate_valid = object_centers(
        candidate_tracks, candidate_visibility, object_slices
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    with imageio.get_writer(
        temporary, fps=fps, codec="libx264", quality=7, macro_block_size=None
    ) as writer:
        for frame_index in range(FRAME_COUNT):
            left = draw_track_panel(
                baseline_frames[frame_index],
                baseline_tracks,
                baseline_visibility,
                baseline_centers,
                baseline_valid,
                object_slices,
                frame_index,
            )
            right = draw_track_panel(
                candidate_frames[frame_index],
                candidate_tracks,
                candidate_visibility,
                candidate_centers,
                candidate_valid,
                object_slices,
                frame_index,
                baseline_centers,
                baseline_valid,
            )
            left = cv2.resize(left, (640, 352), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (640, 352), interpolation=cv2.INTER_AREA)
            cv2.putText(left, f"Baseline | F{frame_index:02d}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right, f"Ablation | F{frame_index:02d} | magenta=Baseline center", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
            body = np.concatenate((left, right), axis=1)
            header = np.full((48, body.shape[1], 3), (30, 62, 55), dtype=np.uint8)
            score = "N/A" if target_score is None else f"{target_score:.4f}"
            cv2.putText(header, f"{variant_id} | trajectory impact={score} | colors=tracked objects", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(np.concatenate((header, body), axis=0), cv2.COLOR_BGR2RGB))
    temporary.replace(path)


def rank_records(records: list[dict[str, Any]]) -> None:
    valid = [
        row
        for row in records
        if row["metrics"].get("trajectory_impact_percent_d0") is not None
    ]
    valid.sort(
        key=lambda row: (
            -float(row["metrics"]["trajectory_impact_percent_d0"]),
            row["variant_id"],
        )
    )
    for row in records:
        row["trajectory_rank_within_case_seed"] = None
        row["track_loss_rank_within_case_seed"] = None
    for rank, row in enumerate(valid, start=1):
        row["trajectory_rank_within_case_seed"] = rank
    track_loss_rows = sorted(
        records,
        key=lambda row: (
            -float(row["metrics"]["target_worst_track_loss_score_0_100"]),
            row["variant_id"],
        ),
    )
    for rank, row in enumerate(track_loss_rows, start=1):
        row["track_loss_rank_within_case_seed"] = rank


def report_payload(
    case: str,
    seed: int,
    baseline_path: Path,
    baseline_signature: dict[str, int],
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rank_records(records)
    records.sort(
        key=lambda row: (
            row["trajectory_rank_within_case_seed"] is None,
            row["trajectory_rank_within_case_seed"] or 10**9,
            row["variant_id"],
        )
    )
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "seed": seed,
        "reference": "same-seed no-intervention Baseline",
        "trajectory_definition": TRAJECTORY_DEFINITION,
        "track_loss_definition": TRACK_LOSS_DEFINITION,
        "quality_gate": {
            "min_valid_frames": args.min_valid_frames,
            "min_coverage": args.min_coverage,
        },
        "baseline_path": str(baseline_path),
        "baseline_signature": baseline_signature,
        "expected_ablation_count": len(candidates),
        "tracked_ablation_count": len(records),
        "ranked_ablation_count": sum(
            row["trajectory_rank_within_case_seed"] is not None for row in records
        ),
        "track_loss_ranked_ablation_count": sum(
            row["track_loss_rank_within_case_seed"] is not None for row in records
        ),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.min_valid_frames < 1 or args.min_valid_frames > FRAME_COUNT:
        raise ValueError("--min-valid-frames must be in [1,49]")
    if not 0.0 < args.min_coverage <= 1.0:
        raise ValueError("--min-coverage must be in (0,1]")
    seed_dir = args.result_dir.expanduser().resolve()
    candidates = collect_candidates(seed_dir, set(args.head_scopes))
    if args.limit > 0:
        candidates = candidates[: args.limit]
    if not candidates:
        raise RuntimeError(f"no completed Head-Scope candidates: {seed_dir}")
    case = str(candidates[0]["case"])
    seed = int(candidates[0]["seed"])
    if any(row["case"] != case or int(row["seed"]) != seed for row in candidates):
        raise RuntimeError("trajectory mode accepts exactly one case/seed directory")

    baseline_path = locate_baseline(case, seed)
    baseline_signature = file_signature(baseline_path)
    frozen_path, frozen_manifest_path = resolve_frozen_baseline_inputs(seed_dir)
    frozen_valid, frozen_reason = validate_frozen_baseline_inputs(seed_dir)
    if not frozen_valid:
        raise RuntimeError(f"invalid frozen baseline inputs: {frozen_reason}")
    with np.load(frozen_path, allow_pickle=False) as arrays:
        baseline_tracks = arrays["tracks"].astype(np.float32)
        baseline_visibility = arrays["visibility"].astype(bool)
        query_points = arrays["query_points"].astype(np.float32)
        region_names = [str(value) for value in arrays["region_names"].tolist()]
        starts = arrays["point_starts"].astype(int).tolist()
        ends = arrays["point_ends"].astype(int).tolist()
    point_count = len(query_points)
    if (
        point_count < 1
        or baseline_tracks.shape != (FRAME_COUNT, point_count, 2)
        or baseline_visibility.shape != (FRAME_COUNT, point_count)
        or not np.isfinite(query_points).all()
    ):
        raise RuntimeError(f"invalid frozen baseline tracks: {frozen_path}")
    object_slices = {
        name: slice(start, end)
        for name, start, end in zip(region_names, starts, ends, strict=True)
    }
    frozen_manifest = json.loads(
        frozen_manifest_path.read_text(encoding="utf-8")
    )
    region_cache = Path(str(frozen_manifest["query_cache_dir"])) / "regions.npz"
    with np.load(region_cache, allow_pickle=False) as arrays:
        masks = arrays["masks_rhw"].astype(bool)[: len(region_names)]
    diagonals = {
        name: bbox_diagonal(mask)
        for name, mask in zip(region_names, masks, strict=True)
    }
    baseline_centers, baseline_center_valid = object_centers(
        baseline_tracks, baseline_visibility, object_slices
    )
    baseline_frames, baseline_fps = load_video_frames(baseline_path)

    output_root = args.output_base.expanduser().resolve() / case / f"seed_{seed:05d}"
    if str(output_root).startswith("/home/gaoya/"):
        raise RuntimeError("large trajectory artifacts may not be stored under /home/gaoya")
    report_path = output_root / "report.json"
    records: list[dict[str, Any]] = []
    model = None
    try:
        for index, candidate in enumerate(candidates, start=1):
            variant = str(candidate["variant_id"])
            signature = candidate["video_signature"]
            track_path = output_root / "tracks" / f"{variant}.npz"
            overlay_path = output_root / "overlays" / f"{variant}.mp4"
            cached_tracks = (
                None
                if args.overwrite
                else load_track_cache(track_path, signature, point_count)
            )
            candidate_frames = None
            if cached_tracks is None:
                if model is None:
                    model = load_cotracker(args.device)
                candidate_frames, fps = load_video_frames(Path(candidate["path"]))
                tracks, visibility = run_cotracker(
                    model, candidate_frames, query_points.copy(), args.device
                )
                atomic_npz(
                    track_path,
                    tracks=tracks.astype(np.float32),
                    visibility=visibility.astype(bool),
                    query_points=query_points,
                    video_signature=np.asarray(signature_text(signature)),
                    video_path=np.asarray(str(candidate["path"])),
                    tracker=np.asarray("CoTracker3 offline scaled checkpoint"),
                )
                state = "tracked"
            else:
                tracks, visibility = cached_tracks
                fps = baseline_fps
                state = "reused"

            candidate_centers, candidate_center_valid = object_centers(
                tracks, visibility, object_slices
            )
            object_metrics = {
                name: object_trajectory_metrics(
                    tracks,
                    visibility,
                    candidate_centers[name],
                    candidate_center_valid[name],
                    baseline_tracks,
                    baseline_visibility,
                    baseline_centers[name],
                    baseline_center_valid[name],
                    object_slices[name],
                    diagonals[name],
                    args.min_valid_frames,
                    args.min_coverage,
                )
                for name in region_names
            }
            selected_objects = (
                [str(candidate["region"])]
                if candidate["target_scope"] == "single_object"
                else region_names
            )
            selected_values = [
                object_metrics[name]["center_ade_norm"]
                for name in selected_objects
                if object_metrics[name]["quality_pass"]
            ]
            quality_pass = len(selected_values) == len(selected_objects)
            target_center_ade = fmean(selected_values) if quality_pass else None
            trajectory_score = (
                100.0 * target_center_ade if target_center_ade is not None else None
            )
            selected_track_loss = [
                float(object_metrics[name]["track_loss_score_0_100"])
                for name in selected_objects
            ]
            metrics = {
                "selected_objects": selected_objects,
                "quality_pass": quality_pass,
                "target_center_ade_norm": rounded(target_center_ade),
                "trajectory_impact_percent_d0": rounded(trajectory_score),
                "target_mean_track_loss_score_0_100": rounded(
                    fmean(selected_track_loss)
                ),
                "target_worst_track_loss_score_0_100": rounded(
                    max(selected_track_loss)
                ),
                "objects": object_metrics,
            }
            if args.overwrite or not overlay_path.is_file():
                if candidate_frames is None:
                    candidate_frames, fps = load_video_frames(Path(candidate["path"]))
                render_overlay(
                    overlay_path,
                    variant,
                    baseline_frames,
                    candidate_frames,
                    baseline_tracks,
                    baseline_visibility,
                    tracks,
                    visibility,
                    object_slices,
                    metrics["trajectory_impact_percent_d0"],
                    fps,
                )
            record = {
                "variant_id": variant,
                "target_scope": candidate["target_scope"],
                "region": candidate.get("region"),
                "mask_mode": candidate["mask_mode"],
                "head_scope": candidate["head_scope"],
                "ranking_tag": candidate.get("ranking_tag"),
                "video_path": str(candidate["path"]),
                "video_signature": signature,
                "track_path": str(track_path),
                "overlay_path": str(overlay_path),
                "metrics": metrics,
            }
            records = [row for row in records if row["variant_id"] != variant]
            records.append(record)
            atomic_json(
                report_path,
                report_payload(
                    case,
                    seed,
                    baseline_path,
                    baseline_signature,
                    candidates,
                    records,
                    args,
                ),
            )
            print(
                f"[{index:03d}/{len(candidates):03d}] {state} {variant} "
                f"trajectory={metrics['trajectory_impact_percent_d0']}",
                flush=True,
            )
            del candidate_frames, tracks, visibility
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
