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


def _trimmed_mean(values: np.ndarray, trim_ratio: float = 0.10) -> float:
    if values.size == 0:
        return 0.0
    if values.size < 4 or trim_ratio <= 0.0:
        return float(values.mean())
    sorted_values = np.sort(values.astype(np.float32, copy=False))
    trim = int(sorted_values.size * trim_ratio)
    if trim == 0 or 2 * trim >= sorted_values.size:
        return float(sorted_values.mean())
    return float(sorted_values[trim:-trim].mean())


def _robust_motion_threshold(magnitude: np.ndarray, *, min_motion_px: float, noise_mad_scale: float) -> float:
    median = float(np.median(magnitude))
    mad = float(np.median(np.abs(magnitude - median)))
    robust_sigma = 1.4826 * mad
    return max(float(min_motion_px), median + float(noise_mad_scale) * robust_sigma)


def _top_fraction_mean(magnitude: np.ndarray, top_flow_percent: float) -> float:
    flat = magnitude.reshape(-1).astype(np.float32, copy=False)
    count = max(1, int(round(flat.size * float(top_flow_percent))))
    if count >= flat.size:
        return float(flat.mean())
    top_values = np.partition(flat, flat.size - count)[-count:]
    return float(top_values.mean())


def _flow_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    active_quantile: float,
    *,
    top_flow_percent: float,
    min_motion_px: float,
    noise_mad_scale: float,
) -> dict[str, float]:
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
    object_threshold = _robust_motion_threshold(
        residual_magnitude,
        min_motion_px=min_motion_px,
        noise_mad_scale=noise_mad_scale,
    )
    object_mask = residual_magnitude > object_threshold
    object_values = residual_magnitude[object_mask]
    moving_area_ratio = float(object_values.size / residual_magnitude.size)
    object_mean = _trimmed_mean(object_values)
    object_p90 = float(np.quantile(object_values, 0.90)) if object_values.size else 0.0
    vbench_top_mean = _top_fraction_mean(residual_magnitude, top_flow_percent)
    return {
        "global_mean": global_mean,
        "legacy_active_mean": active_mean,
        "residual_mean": residual_mean,
        "object_mean": object_mean,
        "object_p90": object_p90,
        "moving_area_ratio": moving_area_ratio,
        "object_threshold_px": object_threshold,
        "vbench_top_mean": vbench_top_mean,
    }


def compute_motion_from_gray_frames(
    gray_frames: list[np.ndarray],
    *,
    fps: float,
    active_quantile: float = 0.80,
    top_flow_percent: float = 0.05,
    min_motion_px: float = 0.05,
    noise_mad_scale: float = 3.0,
) -> dict[str, object]:
    if len(gray_frames) < 2:
        raise ValueError("motion degree requires at least two frames")
    if not 0.0 < active_quantile < 1.0:
        raise ValueError("active_quantile must be between zero and one")
    if not 0.0 < top_flow_percent <= 1.0:
        raise ValueError("top_flow_percent must be in (0, 1]")
    if min_motion_px < 0.0:
        raise ValueError("min_motion_px must be non-negative")
    if noise_mad_scale < 0.0:
        raise ValueError("noise_mad_scale must be non-negative")
    height, width = gray_frames[0].shape
    if any(frame.shape != (height, width) for frame in gray_frames):
        raise ValueError("all gray frames must have the same resolution")

    global_steps: list[float] = []
    active_steps: list[float] = []
    residual_steps: list[float] = []
    object_steps: list[float] = []
    object_p90_steps: list[float] = []
    moving_area_steps: list[float] = []
    object_threshold_steps: list[float] = []
    vbench_top_steps: list[float] = []
    for previous, current in zip(gray_frames[:-1], gray_frames[1:]):
        metrics = _flow_metrics(
            previous,
            current,
            active_quantile,
            top_flow_percent=top_flow_percent,
            min_motion_px=min_motion_px,
            noise_mad_scale=noise_mad_scale,
        )
        global_steps.append(metrics["global_mean"])
        active_steps.append(metrics["legacy_active_mean"])
        residual_steps.append(metrics["residual_mean"])
        object_steps.append(metrics["object_mean"])
        object_p90_steps.append(metrics["object_p90"])
        moving_area_steps.append(metrics["moving_area_ratio"])
        object_threshold_steps.append(metrics["object_threshold_px"])
        vbench_top_steps.append(metrics["vbench_top_mean"])

    diagonal = math.hypot(width, height)
    global_raw = float(np.mean(global_steps))
    active_raw = float(np.mean(active_steps))
    residual_raw = float(np.mean(residual_steps))
    object_raw = float(np.mean(object_steps))
    object_p90_raw = float(np.mean(object_p90_steps))
    moving_area_ratio = float(np.mean(moving_area_steps))
    moving_area_p90 = float(np.quantile(moving_area_steps, 0.90))
    motion_presence_ratio = float(np.mean(np.asarray(moving_area_steps) > 0.0))
    object_threshold_raw = float(np.mean(object_threshold_steps))
    vbench_top_raw = float(np.mean(vbench_top_steps))
    scale_to_diag_pct_per_second = 100.0 * float(fps) / diagonal
    object_score = object_raw * scale_to_diag_pct_per_second
    return {
        "method": "farneback_dense_optical_flow",
        "frame_count": len(gray_frames),
        "transition_count": len(global_steps),
        "fps": float(fps),
        "analysis_resolution": [width, height],
        "active_quantile": float(active_quantile),
        "top_flow_percent": float(top_flow_percent),
        "min_motion_px": float(min_motion_px),
        "noise_mad_scale": float(noise_mad_scale),
        "motion_global_px_per_frame": global_raw,
        "motion_degree_diag_pct_per_second": global_raw * scale_to_diag_pct_per_second,
        "motion_active_px_per_frame": active_raw,
        "motion_active_diag_pct_per_second": active_raw * scale_to_diag_pct_per_second,
        "motion_residual_px_per_frame": residual_raw,
        "motion_residual_diag_pct_per_second": residual_raw * scale_to_diag_pct_per_second,
        "motion_object_px_per_frame": object_raw,
        "motion_object_diag_pct_per_second": object_score,
        "motion_object_p90_px_per_frame": object_p90_raw,
        "motion_object_p90_diag_pct_per_second": object_p90_raw * scale_to_diag_pct_per_second,
        "moving_area_ratio": moving_area_ratio,
        "moving_area_ratio_p90": moving_area_p90,
        "motion_presence_ratio": motion_presence_ratio,
        "motion_object_energy": object_score * math.sqrt(max(moving_area_ratio, 0.0)),
        "motion_vbench_top_px_per_frame": vbench_top_raw,
        "motion_vbench_top_diag_pct_per_second": vbench_top_raw * scale_to_diag_pct_per_second,
        "motion_object_threshold_px": object_threshold_raw,
        "motion_temporal_p90_px_per_frame": float(np.quantile(global_steps, 0.90)),
        "motion_temporal_peak_px_per_frame": float(np.max(global_steps)),
        "temporal_global_px_per_frame": global_steps,
        "temporal_object_px_per_frame": object_steps,
        "temporal_moving_area_ratio": moving_area_steps,
    }


def compute_video_motion(
    video_path: Path,
    *,
    analysis_width: int = 320,
    active_quantile: float = 0.80,
    top_flow_percent: float = 0.05,
    min_motion_px: float = 0.05,
    noise_mad_scale: float = 3.0,
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

    metrics = compute_motion_from_gray_frames(
        gray_frames,
        fps=fps,
        active_quantile=active_quantile,
        top_flow_percent=top_flow_percent,
        min_motion_px=min_motion_px,
        noise_mad_scale=noise_mad_scale,
    )
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
    top_flow_percent: float = 0.05,
    min_motion_px: float = 0.05,
    noise_mad_scale: float = 3.0,
    primary_metric: str = "motion_object_diag_pct_per_second",
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
            top_flow_percent=top_flow_percent,
            min_motion_px=min_motion_px,
            noise_mad_scale=noise_mad_scale,
        )
        records.append(
            {
                "case_id": case_id,
                "family_key": str(item["family_key"]),
                "direction_mode": str(item.get("direction_mode", "legacy_unlabeled")),
                **metrics,
            }
        )

    if not records:
        raise ValueError(f"manifest has no records: {input_root / 'manifest.json'}")
    if primary_metric not in records[0]:
        raise KeyError(f"primary metric not found in records: {primary_metric}")
    scores = np.asarray([float(record[primary_metric]) for record in records])
    low_threshold, high_threshold = (float(value) for value in np.quantile(scores, [1.0 / 3.0, 2.0 / 3.0]))
    for record in records:
        score = float(record[primary_metric])
        record["relative_motion_level"] = _relative_level(score, low_threshold, high_threshold)

    return {
        "definition": {
            "motion_degree_diag_pct_per_second": "legacy global mean dense optical-flow magnitude over all pixels, normalized as 100 * fps * px_per_frame / analysis-frame diagonal",
            "motion_object_diag_pct_per_second": "recommended object-centric score: robust trimmed mean residual optical-flow magnitude over moving pixels only, with residual flow subtracting the spatial median vector",
            "motion_object_energy": "object-centric speed multiplied by sqrt(moving_area_ratio), useful when visible moving extent should matter",
            "motion_vbench_top_diag_pct_per_second": f"VBench-style top-flow score using the largest {100.0 * top_flow_percent:.1f}% residual flow magnitudes per transition",
            "moving_area_ratio": "mean fraction of pixels whose residual flow exceeds max(min_motion_px, median + noise_mad_scale * 1.4826 * MAD)",
        },
        "input_root": str(input_root),
        "analysis_width": analysis_width,
        "primary_metric": primary_metric,
        "case_count": len(records),
        "relative_level_thresholds": {"low_max": low_threshold, "high_min": high_threshold},
        "summary": {
            f"{primary_metric}_mean": float(scores.mean()),
            f"{primary_metric}_median": float(np.median(scores)),
            f"{primary_metric}_min": float(scores.min()),
            f"{primary_metric}_max": float(scores.max()),
            "motion_degree_global_mean": float(np.mean([float(record["motion_degree_diag_pct_per_second"]) for record in records])),
            "motion_object_mean": float(np.mean([float(record["motion_object_diag_pct_per_second"]) for record in records])),
            "motion_object_energy_mean": float(np.mean([float(record["motion_object_energy"]) for record in records])),
            "moving_area_ratio_mean": float(np.mean([float(record["moving_area_ratio"]) for record in records])),
        },
        "records": records,
    }


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    excluded = {"temporal_global_px_per_frame", "temporal_object_px_per_frame", "temporal_moving_area_ratio"}
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
    parser.add_argument("--top-flow-percent", type=float, default=0.05)
    parser.add_argument("--min-motion-px", type=float, default=0.05)
    parser.add_argument("--noise-mad-scale", type=float, default=3.0)
    parser.add_argument("--primary-metric", default="motion_object_diag_pct_per_second")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compute_manifest_motion(
        args.input_root,
        analysis_width=args.analysis_width,
        active_quantile=args.active_quantile,
        top_flow_percent=args.top_flow_percent,
        min_motion_px=args.min_motion_px,
        noise_mad_scale=args.noise_mad_scale,
        primary_metric=args.primary_metric,
    )
    json_path = args.input_root / "motion_metrics.json"
    csv_path = args.input_root / "motion_metrics.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, payload["records"])
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "primary_metric": payload["primary_metric"],
                **payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
