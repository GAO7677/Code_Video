#!/usr/bin/env python3
"""Evaluate completed multi-object M1 guidance videos against Baseline and GT.

This is a post-generation evaluator.  Neither GT frames nor these metrics are
used by guidance.  Pixel metrics are computed at a fixed 320x176 resolution so
the completed hyperparameter grid can be screened incrementally and fairly.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


FRAMES = 49
WIDTH = 320
HEIGHT = 176


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while len(frames) < FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
    capture.release()
    if len(frames) != FRAMES:
        raise RuntimeError(f"expected {FRAMES} frames, got {len(frames)}: {path}")
    return np.stack(frames)


def pair_metrics(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    difference = candidate.astype(np.float32) / 255.0 - reference.astype(np.float32) / 255.0
    mse = float(np.square(difference).mean(dtype=np.float64))
    ssim = float(
        np.mean(
            [
                structural_similarity(left, right, channel_axis=2, data_range=255)
                for left, right in zip(candidate, reference, strict=True)
            ]
        )
    )
    return mse, ssim


def complete_candidates(seed_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for manifest_path in sorted(seed_dir.glob("*/manifest.json")):
        video = manifest_path.parent / "generated.mp4"
        if not video.is_file() or not (manifest_path.parent / "complete.json").is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("protocol") != "wan_top100_m1_multi_object_blockdiag_contrast_guidance_v1":
            continue
        rows.append(
            {
                "variant_id": manifest_path.parent.name,
                "video": video,
                "manifest": manifest,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = args.experiment_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if Path("/home/gaoya") in output.parents:
        raise RuntimeError("metric output must be stored under /data/gaoya")
    search = read_json(root / "search_manifest.json")
    source_by_case = {
        str(row["case"]): Path(str(row["source_video"])).expanduser().resolve()
        for row in search.get("samples", [])
    }

    records: list[dict[str, Any]] = []
    snapshot: list[dict[str, Any]] = []
    seed_dirs = sorted(path for path in (root / "guided").glob("*/*") if path.is_dir())
    for seed_dir in seed_dirs:
        candidates = complete_candidates(seed_dir)
        if not candidates:
            continue
        case = str(candidates[0]["manifest"]["case"])
        seed = int(candidates[0]["manifest"]["seed"])
        baseline_path = Path(str(candidates[0]["manifest"]["baseline_video"]))
        source_path = source_by_case[case]
        if not baseline_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"missing Baseline/GT for {case} seed={seed}")
        baseline = read_video(baseline_path)
        try:
            source = read_video(source_path)
        except RuntimeError as exc:
            snapshot.append(
                {
                    "case": case,
                    "seed": seed,
                    "completed_variants": len(candidates),
                    "full_grid": len(candidates) == 16,
                    "gt_eligible": False,
                    "gt_ineligible_reason": str(exc),
                }
            )
            print(f"[skip GT] {case}/seed_{seed:05d}: {exc}", flush=True)
            continue
        baseline_mse, baseline_ssim = pair_metrics(baseline, source)

        def evaluate(row: dict[str, Any]) -> dict[str, Any]:
            candidate = read_video(Path(row["video"]))
            gt_mse, gt_ssim = pair_metrics(candidate, source)
            manifest = row["manifest"]
            delta_mse = gt_mse - baseline_mse
            delta_ssim = gt_ssim - baseline_ssim
            return {
                "case": case,
                "seed": seed,
                "variant_id": row["variant_id"],
                "video": str(row["video"]),
                "baseline_video": str(baseline_path),
                "gt_video": str(source_path),
                "pag_scale": float(manifest["pag_scale"]),
                "guidance_step_range_inclusive": list(
                    manifest["guidance_step_range_inclusive"]
                ),
                "gt_mse_320x176": gt_mse,
                "baseline_gt_mse_320x176": baseline_mse,
                "gt_mse_delta_vs_baseline": delta_mse,
                "gt_mse_relative_change_percent": (
                    100.0 * delta_mse / baseline_mse if baseline_mse > 0 else None
                ),
                "gt_ssim_320x176": gt_ssim,
                "baseline_gt_ssim_320x176": baseline_ssim,
                "gt_ssim_delta_vs_baseline": delta_ssim,
                "mse_improved": delta_mse < 0.0,
                "ssim_improved": delta_ssim > 0.0,
            }

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            seed_records = list(pool.map(evaluate, candidates))
        records.extend(seed_records)
        snapshot.append(
            {
                "case": case,
                "seed": seed,
                "completed_variants": len(seed_records),
                "full_grid": len(seed_records) == 16,
                "gt_eligible": True,
            }
        )
        print(f"[{case}/seed_{seed:05d}] {len(seed_records)} variants", flush=True)

    atomic_json(
        output,
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_root": str(root),
            "post_generation_only": True,
            "gt_used_by_guidance": False,
            "resolution": [WIDTH, HEIGHT],
            "frame_count": FRAMES,
            "metric_definitions": {
                "gt_mse_delta_vs_baseline": (
                    "MSE(candidate,GT)-MSE(Baseline,GT); negative means candidate is closer to GT"
                ),
                "gt_ssim_delta_vs_baseline": (
                    "SSIM(candidate,GT)-SSIM(Baseline,GT); positive means candidate is closer to GT"
                ),
            },
            "case_seed_count": len(snapshot),
            "completed_video_count": len(records),
            "snapshot": snapshot,
            "records": records,
        },
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
