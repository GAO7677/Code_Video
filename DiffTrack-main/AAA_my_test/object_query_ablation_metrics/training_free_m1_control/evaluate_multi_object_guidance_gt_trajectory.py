#!/usr/bin/env python3
"""Compare guided and Baseline CoTracker trajectories to source-video GT."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.object_query_ablation_metrics.compute_head_scope_trajectory_metrics import (  # noqa: E402
    bbox_diagonal,
    load_video_frames,
    resolve_frozen_baseline_inputs,
    resolve_region_cache_path,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (  # noqa: E402
    load_cotracker,
    run_cotracker,
)


FRAMES = 49


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--pixel-report", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def gt_tracks(
    model: Any,
    source_video: Path,
    points: np.ndarray,
    cache_path: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as arrays:
            tracks = arrays["tracks"].astype(np.float32)
            visibility = arrays["visibility"].astype(bool)
            cached_points = arrays["query_points"].astype(np.float32)
        if tracks.shape == (FRAMES, len(points), 2) and np.allclose(cached_points, points):
            return tracks, visibility
    frames, _fps = load_video_frames(source_video)
    tracks, visibility = run_cotracker(model, frames, points, device)
    atomic_npz(
        cache_path,
        tracks=np.asarray(tracks, dtype=np.float32),
        visibility=np.asarray(visibility, dtype=np.bool_),
        query_points=np.asarray(points, dtype=np.float32),
        source_video=np.asarray(str(source_video)),
    )
    return tracks, visibility


def object_center_ade(
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    reference_tracks: np.ndarray,
    reference_visibility: np.ndarray,
    part: slice,
    d0: float,
) -> tuple[float | None, int]:
    distances = []
    for frame in range(FRAMES):
        use = (
            candidate_visibility[frame, part]
            & reference_visibility[frame, part]
            & np.isfinite(candidate_tracks[frame, part]).all(axis=1)
            & np.isfinite(reference_tracks[frame, part]).all(axis=1)
        )
        if int(use.sum()) < 4:
            continue
        candidate_center = np.median(candidate_tracks[frame, part][use], axis=0)
        reference_center = np.median(reference_tracks[frame, part][use], axis=0)
        distances.append(float(np.linalg.norm(candidate_center - reference_center) / d0))
    if len(distances) < 25:
        return None, len(distances)
    return float(np.mean(distances)), len(distances)


def aggregate_ade(
    tracks: np.ndarray,
    visibility: np.ndarray,
    reference_tracks: np.ndarray,
    reference_visibility: np.ndarray,
    slices: dict[str, slice],
    diagonals: dict[str, float],
) -> tuple[float | None, dict[str, Any]]:
    per_object = {}
    values = []
    for name, part in slices.items():
        ade, valid = object_center_ade(
            tracks,
            visibility,
            reference_tracks,
            reference_visibility,
            part,
            diagonals[name],
        )
        per_object[name] = {"center_ade_d0": ade, "valid_frames": valid}
        if ade is not None:
            values.append(ade)
    return (float(np.mean(values)) if len(values) == len(slices) else None), per_object


def main() -> None:
    args = parse_args()
    experiment = args.experiment_root.expanduser().resolve()
    trajectory_root = args.trajectory_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    pixel = read_json(args.pixel_report)
    strict_units = {
        (str(row["case"]), int(row["seed"]))
        for row in pixel.get("snapshot", [])
        if row.get("full_grid") and row.get("gt_eligible")
    }
    pixel_records = {
        (str(row["case"]), int(row["seed"]), str(row["variant_id"])): row
        for row in pixel.get("records", [])
        if (str(row["case"]), int(row["seed"])) in strict_units
    }
    source_by_case = {
        str(row["case"]): Path(str(row["source_video"])).expanduser().resolve()
        for row in read_json(experiment / "search_manifest.json").get("samples", [])
    }
    records = []
    model = load_cotracker(args.device)
    try:
        for case_index, case in enumerate(sorted({case for case, _seed in strict_units}), start=1):
            case_units = sorted(seed for name, seed in strict_units if name == case)
            first_seed_dir = experiment / "guided" / case / f"seed_{case_units[0]:05d}"
            frozen_path, frozen_manifest_path = resolve_frozen_baseline_inputs(first_seed_dir)
            with np.load(frozen_path, allow_pickle=False) as arrays:
                points = arrays["query_points"].astype(np.float32)
                names = [str(value) for value in arrays["region_names"].tolist()]
                starts = arrays["point_starts"].astype(int).tolist()
                ends = arrays["point_ends"].astype(int).tolist()
            slices = {
                name: slice(start, end)
                for name, start, end in zip(names, starts, ends, strict=True)
            }
            region_cache = resolve_region_cache_path(frozen_manifest_path, case)
            with np.load(region_cache, allow_pickle=False) as arrays:
                masks = arrays["masks_rhw"].astype(bool)[: len(names)]
            diagonals = {
                name: bbox_diagonal(mask)
                for name, mask in zip(names, masks, strict=True)
            }
            source = source_by_case[case]
            gt_track, gt_visibility = gt_tracks(
                model,
                source,
                points,
                cache_root / case / "source_gt_tracks.npz",
                args.device,
            )
            for seed in case_units:
                seed_dir = experiment / "guided" / case / f"seed_{seed:05d}"
                frozen_path, _manifest_path = resolve_frozen_baseline_inputs(seed_dir)
                with np.load(frozen_path, allow_pickle=False) as arrays:
                    baseline_tracks = arrays["tracks"].astype(np.float32)
                    baseline_visibility = arrays["visibility"].astype(bool)
                baseline_ade, baseline_objects = aggregate_ade(
                    baseline_tracks,
                    baseline_visibility,
                    gt_track,
                    gt_visibility,
                    slices,
                    diagonals,
                )
                for key, pixel_row in pixel_records.items():
                    row_case, row_seed, variant = key
                    if row_case != case or row_seed != seed:
                        continue
                    track_path = trajectory_root / case / f"seed_{seed:05d}" / "tracks" / f"{variant}.npz"
                    if not track_path.is_file():
                        continue
                    with np.load(track_path, allow_pickle=False) as arrays:
                        candidate_tracks = arrays["tracks"].astype(np.float32)
                        candidate_visibility = arrays["visibility"].astype(bool)
                    candidate_ade, candidate_objects = aggregate_ade(
                        candidate_tracks,
                        candidate_visibility,
                        gt_track,
                        gt_visibility,
                        slices,
                        diagonals,
                    )
                    delta = (
                        candidate_ade - baseline_ade
                        if candidate_ade is not None and baseline_ade is not None
                        else None
                    )
                    records.append(
                        {
                            "case": case,
                            "seed": seed,
                            "variant_id": variant,
                            "pag_scale": pixel_row["pag_scale"],
                            "guidance_step_range_inclusive": pixel_row[
                                "guidance_step_range_inclusive"
                            ],
                            "candidate_gt_center_ade_d0": candidate_ade,
                            "baseline_gt_center_ade_d0": baseline_ade,
                            "gt_center_ade_delta_vs_baseline": delta,
                            "trajectory_improved": delta is not None and delta < 0.0,
                            "quality_gate_passed": delta is not None,
                            "baseline_objects": baseline_objects,
                            "candidate_objects": candidate_objects,
                            "candidate_track_path": str(track_path),
                        }
                    )
            atomic_json(
                args.output,
                {
                    "schema_version": 1,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "post_generation_only": True,
                    "gt_used_by_guidance": False,
                    "metric": (
                        "mean over objects of CoTracker center ADE/D0 to source-video GT; "
                        "delta=candidate-Baseline, negative is better"
                    ),
                    "case_count": case_index,
                    "record_count": len(records),
                    "records": records,
                },
            )
            print(f"[{case_index}] {case}: cumulative records={len(records)}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
