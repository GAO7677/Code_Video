#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0713pybullet")


def _flow_metrics(previous: np.ndarray, current: np.ndarray, active_quantile: float) -> tuple[float, float, float]:
    flow = cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.linalg.norm(flow, axis=-1).astype(np.float32)
    global_mean = float(magnitude.mean())

    active_threshold = float(np.quantile(magnitude, active_quantile))
    active = magnitude[magnitude >= active_threshold]
    active_mean = float(active.mean()) if active.size else 0.0

    median_flow = np.median(flow.reshape(-1, 2), axis=0)
    residual_magnitude = np.linalg.norm(flow - median_flow[None, None, :], axis=-1)
    residual_mean = float(residual_magnitude.mean())
    return global_mean, active_mean, residual_mean


def compute_motion_from_gray_frames(
    gray_frames: list[np.ndarray],
    *,
    fps: float,
    active_quantile: float = 0.80,
) -> dict[str, object]:
    if len(gray_frames) < 2:
        raise ValueError("motion degree requires at least two frames")
    if not 0.0 < active_quantile < 1.0:
        raise ValueError("active_quantile must be between zero and one")
    height, width = gray_frames[0].shape
    if any(frame.shape != (height, width) for frame in gray_frames):
        raise ValueError("all gray frames must have the same resolution")

    global_steps: list[float] = []
    active_steps: list[float] = []
    residual_steps: list[float] = []
    for previous, current in zip(gray_frames[:-1], gray_frames[1:]):
        global_mean, active_mean, residual_mean = _flow_metrics(previous, current, active_quantile)
        global_steps.append(global_mean)
        active_steps.append(active_mean)
        residual_steps.append(residual_mean)

    diagonal = math.hypot(width, height)
    global_raw = float(np.mean(global_steps))
    active_raw = float(np.mean(active_steps))
    residual_raw = float(np.mean(residual_steps))
    scale_to_diag_pct_per_second = 100.0 * float(fps) / diagonal
    return {
        "method": "farneback_dense_optical_flow",
        "frame_count": len(gray_frames),
        "transition_count": len(global_steps),
        "fps": float(fps),
        "analysis_resolution": [width, height],
        "active_quantile": float(active_quantile),
        "motion_global_px_per_frame": global_raw,
        "motion_degree_diag_pct_per_second": global_raw * scale_to_diag_pct_per_second,
        "motion_active_px_per_frame": active_raw,
        "motion_active_diag_pct_per_second": active_raw * scale_to_diag_pct_per_second,
        "motion_residual_px_per_frame": residual_raw,
        "motion_residual_diag_pct_per_second": residual_raw * scale_to_diag_pct_per_second,
        "motion_temporal_p90_px_per_frame": float(np.quantile(global_steps, 0.90)),
        "motion_temporal_peak_px_per_frame": float(np.max(global_steps)),
        "temporal_global_px_per_frame": global_steps,
    }


def compute_video_motion(
    video_path: Path,
    *,
    analysis_width: int = 320,
    active_quantile: float = 0.80,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analysis_height = max(1, round(original_height * analysis_width / original_width))
    gray_frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
            gray_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()

    metrics = compute_motion_from_gray_frames(gray_frames, fps=fps, active_quantile=active_quantile)
    metrics["video"] = str(video_path)
    metrics["original_resolution"] = [original_width, original_height]
    return metrics


def _relative_level(score: float, low_threshold: float, high_threshold: float) -> str:
    if score <= low_threshold:
        return "low"
    if score >= high_threshold:
        return "high"
    return "medium"


def compute_manifest_motion(
    input_root: Path,
    *,
    analysis_width: int = 320,
    active_quantile: float = 0.80,
) -> dict[str, object]:
    manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for index, item in enumerate(manifest, 1):
        case_id = str(item["case_id"])
        print(f"[{index:03d}/{len(manifest):03d}] {case_id}", flush=True)
        metrics = compute_video_motion(
            Path(item["video"]),
            analysis_width=analysis_width,
            active_quantile=active_quantile,
        )
        records.append(
            {
                "case_id": case_id,
                "family_key": str(item["family_key"]),
                "direction_mode": str(item.get("direction_mode", "legacy_unlabeled")),
                **metrics,
            }
        )

    scores = np.asarray([float(record["motion_degree_diag_pct_per_second"]) for record in records])
    low_threshold, high_threshold = (float(value) for value in np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0]))
    for record in records:
        score = float(record["motion_degree_diag_pct_per_second"])
        record["relative_motion_level"] = _relative_level(score, low_threshold, high_threshold)

    return {
        "definition": {
            "raw": "mean dense optical-flow magnitude over all pixels and adjacent frame pairs (pixels/frame)",
            "motion_degree": "100 * fps * raw / analysis-frame diagonal (% diagonal/second)",
            "active": f"same normalization using the top {100.0 * (1.0 - active_quantile):.0f}% flow magnitudes per transition",
            "residual": "same normalization after subtracting the spatial median flow vector",
        },
        "input_root": str(input_root),
        "analysis_width": analysis_width,
        "case_count": len(records),
        "relative_level_thresholds": {"low_max": low_threshold, "high_min": high_threshold},
        "summary": {
            "motion_degree_mean": float(scores.mean()),
            "motion_degree_median": float(np.median(scores)),
            "motion_degree_min": float(scores.min()),
            "motion_degree_max": float(scores.max()),
        },
        "records": records,
    }


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    excluded = {"temporal_global_px_per_frame"}
    fieldnames = [key for key in records[0] if key not in excluded]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute optical-flow motion degree for a rigid video manifest.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--analysis-width", type=int, default=320)
    parser.add_argument("--active-quantile", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compute_manifest_motion(
        args.input_root,
        analysis_width=args.analysis_width,
        active_quantile=args.active_quantile,
    )
    json_path = args.input_root / "motion_metrics.json"
    csv_path = args.input_root / "motion_metrics.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, payload["records"])
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), **payload["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
