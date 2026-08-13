#!/usr/bin/env python3
"""Select the fixed Phase-D alpha using post-generation GT metrics only."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_direct_scaling_phase_bd import (
    output_directory,
)
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_test5_phase_bd_batch import (
    phase_args,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (
    load_cotracker,
    run_cotracker,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache


ALPHAS = (0.1, 0.25)
HEIGHT, WIDTH, FRAMES = 704, 1280, 49


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_video(path: Path) -> np.ndarray:
    frames = np.asarray(iio.imread(path))
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise RuntimeError(f"unexpected video shape for {path}: {frames.shape}")
    if len(frames) < FRAMES:
        raise RuntimeError(f"expected at least {FRAMES} frames in {path}, got {len(frames)}")
    frames = frames[:FRAMES, ..., :3].astype(np.uint8, copy=False)
    if frames.shape[1:3] != (HEIGHT, WIDTH):
        frames = np.stack(
            [cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4) for frame in frames]
        )
    return frames


def video_mse(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape:
        raise ValueError(f"MSE shape mismatch: {candidate.shape} != {reference.shape}")
    total, count = 0.0, 0
    for cand, ref in zip(candidate, reference):
        difference = cand.astype(np.float32) / 255.0 - ref.astype(np.float32) / 255.0
        total += float(np.square(difference, dtype=np.float32).sum(dtype=np.float64))
        count += int(difference.size)
    return total / count


def center_ade_d0(
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    gt_tracks: np.ndarray,
    gt_visibility: np.ndarray,
    d0: float,
) -> tuple[float | None, int, float]:
    centers = []
    visible_point_rates = []
    for frame in range(min(len(candidate_tracks), len(gt_tracks))):
        joint = candidate_visibility[frame] & gt_visibility[frame]
        visible_point_rates.append(float(joint.mean()))
        if int(joint.sum()) < 2:
            continue
        candidate_center = candidate_tracks[frame, joint].mean(axis=0)
        gt_center = gt_tracks[frame, joint].mean(axis=0)
        centers.append(float(np.linalg.norm(candidate_center - gt_center) / d0))
    # Frozen selection gate: at least half of the 49 frames need two common points.
    if len(centers) < 25:
        return None, len(centers), float(np.mean(visible_point_rates))
    return float(np.mean(centers)), len(centers), float(np.mean(visible_point_rates))


def rank_two(left: float, right: float, tolerance: float = 1.0e-12) -> tuple[float, float]:
    if abs(left - right) <= tolerance:
        return 1.5, 1.5
    return (1.0, 2.0) if left < right else (2.0, 1.0)


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest.get("samples") or [])
    if len(samples) != 100:
        raise RuntimeError(f"expected 100 samples, got {len(samples)}")
    by_case: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_case[str(sample["case"])].append(sample)
    if len(by_case) != 20:
        raise RuntimeError(f"expected 20 cases, got {len(by_case)}")

    model = load_cotracker(str(args.device))
    records: list[dict] = []
    try:
        for case_index, (case, case_samples) in enumerate(sorted(by_case.items()), start=1):
            sample0 = case_samples[0]
            cache_dir = Path(str(sample0["query_cache_dir"]))
            cache = load_region_cache(cache_dir.parent, cache_dir.name)
            object_a = next(region for region in cache.regions if region.region_name == "object_A")
            points = cache.query_points[object_a.point_start : object_a.point_end].astype(np.float32)
            mask = cache.masks_rhw[cache.regions.index(object_a)].astype(bool)
            ys, xs = np.where(mask)
            d0 = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
            if not d0 > 0:
                raise RuntimeError(f"{case}: invalid object_A D0")
            gt_frames = read_video(Path(str(sample0["source_video"])))
            gt_tracks, gt_visibility = run_cotracker(
                model, gt_frames, points, str(args.device)
            )
            print(f"[{case_index}/{len(by_case)}] GT tracks ready: {case}", flush=True)
            for sample in sorted(case_samples, key=lambda row: int(row["seed"])):
                for alpha in ALPHAS:
                    frozen = phase_args(
                        output_root=args.output_root,
                        sample=sample,
                        phase="phase_b",
                        alpha=alpha,
                        start=0,
                        end=39,
                    )
                    video_path = output_directory(frozen) / "generated.mp4"
                    if not video_path.is_file():
                        raise FileNotFoundError(video_path)
                    candidate = read_video(video_path)
                    mse = video_mse(candidate, gt_frames)
                    tracks, visibility = run_cotracker(
                        model, candidate, points, str(args.device)
                    )
                    ade, valid_frames, visibility_rate = center_ade_d0(
                        tracks,
                        visibility,
                        gt_tracks,
                        gt_visibility,
                        d0,
                    )
                    records.append(
                        {
                            "case": case,
                            "seed": int(sample["seed"]),
                            "alpha": float(alpha),
                            "video": str(video_path),
                            "gt_video": str(sample["source_video"]),
                            "gt_full_frame_mse": float(mse),
                            "gt_cotracker_center_ade_d0": ade,
                            "trajectory_gate_passed": ade is not None,
                            "trajectory_common_valid_frames": int(valid_frames),
                            "trajectory_joint_visibility_rate": float(visibility_rate),
                            "object_a_d0_px": d0,
                        }
                    )
                    del candidate, tracks, visibility
                    torch.cuda.empty_cache()
    finally:
        del model
        torch.cuda.empty_cache()

    case_summary: list[dict] = []
    rank_scores = {alpha: [] for alpha in ALPHAS}
    for case in sorted(by_case):
        rows = [row for row in records if row["case"] == case]
        values: dict[float, dict[str, float | None]] = {}
        for alpha in ALPHAS:
            alpha_rows = [row for row in rows if row["alpha"] == alpha]
            values[alpha] = {
                "gt_full_frame_mse": mean([row["gt_full_frame_mse"] for row in alpha_rows]),
                "gt_cotracker_center_ade_d0": mean(
                    [
                        row["gt_cotracker_center_ade_d0"]
                        for row in alpha_rows
                        if row["gt_cotracker_center_ade_d0"] is not None
                    ]
                ),
                "trajectory_valid_seed_count": sum(
                    row["trajectory_gate_passed"] for row in alpha_rows
                ),
            }
        metric_ranks: dict[str, dict[float, float]] = {}
        mse_ranks = rank_two(
            float(values[0.1]["gt_full_frame_mse"]),
            float(values[0.25]["gt_full_frame_mse"]),
        )
        metric_ranks["gt_full_frame_mse"] = {0.1: mse_ranks[0], 0.25: mse_ranks[1]}
        left_ade = values[0.1]["gt_cotracker_center_ade_d0"]
        right_ade = values[0.25]["gt_cotracker_center_ade_d0"]
        if left_ade is not None and right_ade is not None:
            ade_ranks = rank_two(float(left_ade), float(right_ade))
            metric_ranks["gt_cotracker_center_ade_d0"] = {
                0.1: ade_ranks[0],
                0.25: ade_ranks[1],
            }
        for alpha in ALPHAS:
            case_rank = float(np.mean([ranks[alpha] for ranks in metric_ranks.values()]))
            rank_scores[alpha].append(case_rank)
        case_summary.append(
            {
                "case": case,
                "seed_count": len(by_case[case]),
                "alpha_metrics": {str(alpha): values[alpha] for alpha in ALPHAS},
                "metric_ranks": {
                    metric: {str(alpha): rank for alpha, rank in ranks.items()}
                    for metric, ranks in metric_ranks.items()
                },
                "mean_rank": {
                    str(alpha): float(np.mean([ranks[alpha] for ranks in metric_ranks.values()]))
                    for alpha in ALPHAS
                },
            }
        )

    aggregate_rank = {alpha: float(np.mean(rank_scores[alpha])) for alpha in ALPHAS}
    selected = min(ALPHAS, key=lambda alpha: (aggregate_rank[alpha], alpha))
    selection = {
        "protocol": "phase_b_post_generation_mse_cotracker_selection_v1",
        "manifest": str(args.manifest_path),
        "case_count": len(by_case),
        "seed_count_per_case": 5,
        "record_count": len(records),
        "candidate_alphas": list(ALPHAS),
        "selected_alpha": float(selected),
        "selection_rule": (
            "lower is better; average seeds within each case; rank alphas separately "
            "for GT full-frame MSE and gated GT-relative CoTracker Center-ADE/D0; "
            "average available metric ranks per case; give every case equal weight; "
            "break an exact aggregate tie in favor of the smaller alpha"
        ),
        "trajectory_gate": "at least 25/49 frames with >=2 jointly visible object_A points",
        "metric_role": "post-generation hyperparameter selection only; neither metric enters guidance",
        "aggregate_case_balanced_mean_rank": {
            str(alpha): aggregate_rank[alpha] for alpha in ALPHAS
        },
        "case_summary": case_summary,
        "records": records,
    }
    atomic_json(args.selection_path, selection)
    print(
        f"selected alpha={selected:g}; ranks="
        + ", ".join(f"{alpha:g}:{aggregate_rank[alpha]:.4f}" for alpha in ALPHAS),
        flush=True,
    )
    print(args.selection_path, flush=True)


if __name__ == "__main__":
    main()
