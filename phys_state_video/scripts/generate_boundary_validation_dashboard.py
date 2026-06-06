#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cv2
import html
import json
import math
import os
from pathlib import Path

import numpy as np


CENTER_SLICE = slice(0, 2)
LOG_SCALE_INDEX = 3
MOTION_SLICE = slice(4, 7)
VISIBILITY_INDEX = 7
EXISTENCE_INDEX = 8

MODEL_COLORS = {
    "gt": "#1f1f1b",
    "baseline": "#b6422e",
    "control": "#7f6c57",
    "boundary0.1": "#1f6f8b",
    "boundary0.5": "#0f8a5f",
    "boundary1.0": "#7e3af2",
    "boundary_new": "#c06c2b",
}

ORDERED_COMPARE_LABELS = ["control", "boundary0.1", "boundary0.5", "boundary1.0"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a boundary validation dashboard from predictor comparison exports.")
    parser.add_argument(
        "--comparison-root",
        required=True,
        help="Root directory containing scale subdirectories with report.json and assets/*.npz.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write dashboard assets and index.html.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=3,
        help="Number of early future steps used by rollout metrics.",
    )
    parser.add_argument(
        "--rollout-decay",
        type=float,
        default=0.6,
        help="Geometric decay used for early rollout weighting.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = path
    payload["_slug"] = path.parent.name
    payload["_label_b"] = str(payload.get("label_b", path.parent.name))
    return payload


def boundary_mask(context_last: np.ndarray, future_first: np.ndarray) -> np.ndarray:
    context_present = (context_last[..., EXISTENCE_INDEX] > 0.5) | (context_last[..., VISIBILITY_INDEX] > 0.2)
    future_present = (future_first[..., EXISTENCE_INDEX] > 0.5) | (future_first[..., VISIBILITY_INDEX] > 0.2)
    return context_present | future_present


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    mask = mask.astype(bool)
    if values.shape[: mask.ndim] != mask.shape:
        raise ValueError(f"mask shape {mask.shape} incompatible with values {values.shape}")
    if not np.any(mask):
        return float(np.mean(values))
    if values.ndim == mask.ndim:
        return float(values[mask].mean())
    masked = values[mask]
    return float(masked.mean())


def weighted_masked_mean(values: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> float:
    weighted = values * weights
    denom = (mask * weights).sum()
    if float(denom) <= 1e-8:
        return float(weighted.mean())
    return float(weighted.sum() / denom)


def safe_direction_error(pred_delta: np.ndarray, gt_delta: np.ndarray) -> np.ndarray:
    pred_norm = np.linalg.norm(pred_delta, axis=-1)
    gt_norm = np.linalg.norm(gt_delta, axis=-1)
    denom = np.maximum(pred_norm * gt_norm, 1e-8)
    cosine = np.clip((pred_delta * gt_delta).sum(axis=-1) / denom, -1.0, 1.0)
    valid = (pred_norm > 1e-6) & (gt_norm > 1e-6)
    out = np.zeros_like(pred_norm)
    out[valid] = 1.0 - cosine[valid]
    return out


def compute_case_metrics(
    predicted_context: np.ndarray,
    predicted_future: np.ndarray,
    target_context: np.ndarray,
    target_future: np.ndarray,
    *,
    rollout_steps: int,
    rollout_decay: float,
) -> dict[str, float]:
    pred_context_last = predicted_context[-1]
    pred_future_first = predicted_future[0]
    tgt_context_last = target_context[-1]
    tgt_future_first = target_future[0]
    mask = boundary_mask(tgt_context_last, tgt_future_first)

    pred_center_delta = pred_future_first[..., CENTER_SLICE] - pred_context_last[..., CENTER_SLICE]
    tgt_center_delta = tgt_future_first[..., CENTER_SLICE] - tgt_context_last[..., CENTER_SLICE]
    pred_motion_delta = pred_future_first[..., MOTION_SLICE] - pred_context_last[..., MOTION_SLICE]
    tgt_motion_delta = tgt_future_first[..., MOTION_SLICE] - tgt_context_last[..., MOTION_SLICE]
    pred_scale_delta = pred_future_first[..., LOG_SCALE_INDEX] - pred_context_last[..., LOG_SCALE_INDEX]
    tgt_scale_delta = tgt_future_first[..., LOG_SCALE_INDEX] - tgt_context_last[..., LOG_SCALE_INDEX]

    head_center_error = np.linalg.norm(pred_future_first[..., CENTER_SLICE] - tgt_future_first[..., CENTER_SLICE], axis=-1)
    tail_center_error = np.linalg.norm(pred_context_last[..., CENTER_SLICE] - tgt_context_last[..., CENTER_SLICE], axis=-1)
    head_motion_error = np.linalg.norm(pred_future_first[..., MOTION_SLICE] - tgt_future_first[..., MOTION_SLICE], axis=-1)
    tail_motion_error = np.linalg.norm(pred_context_last[..., MOTION_SLICE] - tgt_context_last[..., MOTION_SLICE], axis=-1)
    head_log_scale_error = np.abs(pred_future_first[..., LOG_SCALE_INDEX] - tgt_future_first[..., LOG_SCALE_INDEX])
    tail_log_scale_error = np.abs(pred_context_last[..., LOG_SCALE_INDEX] - tgt_context_last[..., LOG_SCALE_INDEX])

    delta_center_error = np.linalg.norm(pred_center_delta - tgt_center_delta, axis=-1)
    delta_motion_error = np.linalg.norm(pred_motion_delta - tgt_motion_delta, axis=-1)
    delta_log_scale_error = np.abs(pred_scale_delta - tgt_scale_delta)
    jump_mag_error = np.abs(np.linalg.norm(pred_center_delta, axis=-1) - np.linalg.norm(tgt_center_delta, axis=-1))
    jump_direction_error = safe_direction_error(pred_center_delta, tgt_center_delta)

    horizon = min(int(rollout_steps), int(target_future.shape[0]))
    rollout_weights = np.asarray([float(rollout_decay) ** idx for idx in range(horizon)], dtype=np.float32).reshape(horizon, 1)
    rollout_mask = (
        (target_future[:horizon, :, EXISTENCE_INDEX] > 0.5)
        | (target_future[:horizon, :, VISIBILITY_INDEX] > 0.2)
    ).astype(np.float32)
    rollout_center_error = np.linalg.norm(
        predicted_future[:horizon, :, CENTER_SLICE] - target_future[:horizon, :, CENTER_SLICE],
        axis=-1,
    )
    rollout_motion_error = np.linalg.norm(
        predicted_future[:horizon, :, MOTION_SLICE] - target_future[:horizon, :, MOTION_SLICE],
        axis=-1,
    )
    rollout_log_scale_error = np.abs(
        predicted_future[:horizon, :, LOG_SCALE_INDEX] - target_future[:horizon, :, LOG_SCALE_INDEX]
    )

    if target_future.shape[0] >= 2:
        pred_center_curvature = (
            predicted_future[1, :, CENTER_SLICE]
            - 2.0 * predicted_future[0, :, CENTER_SLICE]
            + predicted_context[-1, :, CENTER_SLICE]
        )
        tgt_center_curvature = (
            target_future[1, :, CENTER_SLICE]
            - 2.0 * target_future[0, :, CENTER_SLICE]
            + target_context[-1, :, CENTER_SLICE]
        )
        pred_motion_curvature = (
            predicted_future[1, :, MOTION_SLICE]
            - 2.0 * predicted_future[0, :, MOTION_SLICE]
            + predicted_context[-1, :, MOTION_SLICE]
        )
        tgt_motion_curvature = (
            target_future[1, :, MOTION_SLICE]
            - 2.0 * target_future[0, :, MOTION_SLICE]
            + target_context[-1, :, MOTION_SLICE]
        )
        pred_scale_curvature = (
            predicted_future[1, :, LOG_SCALE_INDEX]
            - 2.0 * predicted_future[0, :, LOG_SCALE_INDEX]
            + predicted_context[-1, :, LOG_SCALE_INDEX]
        )
        tgt_scale_curvature = (
            target_future[1, :, LOG_SCALE_INDEX]
            - 2.0 * target_future[0, :, LOG_SCALE_INDEX]
            + target_context[-1, :, LOG_SCALE_INDEX]
        )
        curvature_center_error = np.linalg.norm(pred_center_curvature - tgt_center_curvature, axis=-1)
        curvature_motion_error = np.linalg.norm(pred_motion_curvature - tgt_motion_curvature, axis=-1)
        curvature_log_scale_error = np.abs(pred_scale_curvature - tgt_scale_curvature)
    else:
        curvature_center_error = np.zeros(mask.shape, dtype=np.float32)
        curvature_motion_error = np.zeros(mask.shape, dtype=np.float32)
        curvature_log_scale_error = np.zeros(mask.shape, dtype=np.float32)

    summary = {
        "tail_center_error": masked_mean(tail_center_error, mask),
        "head_center_error": masked_mean(head_center_error, mask),
        "tail_motion_error": masked_mean(tail_motion_error, mask),
        "head_motion_error": masked_mean(head_motion_error, mask),
        "tail_log_scale_error": masked_mean(tail_log_scale_error, mask),
        "head_log_scale_error": masked_mean(head_log_scale_error, mask),
        "boundary_center_delta_error": masked_mean(delta_center_error, mask),
        "boundary_motion_delta_error": masked_mean(delta_motion_error, mask),
        "boundary_log_scale_delta_error": masked_mean(delta_log_scale_error, mask),
        "boundary_jump_magnitude_error": masked_mean(jump_mag_error, mask),
        "boundary_jump_direction_error": masked_mean(jump_direction_error, mask),
        "rollout_center_error": weighted_masked_mean(rollout_center_error, rollout_mask, rollout_weights),
        "rollout_motion_error": weighted_masked_mean(rollout_motion_error, rollout_mask, rollout_weights),
        "rollout_log_scale_error": weighted_masked_mean(rollout_log_scale_error, rollout_mask, rollout_weights),
        "boundary_curvature_center_error": masked_mean(curvature_center_error, mask),
        "boundary_curvature_motion_error": masked_mean(curvature_motion_error, mask),
        "boundary_curvature_log_scale_error": masked_mean(curvature_log_scale_error, mask),
        "boundary_object_count": float(mask.sum()),
    }
    summary["boundary_validation_score_v1"] = (
        1.00 * summary["head_center_error"]
        + 0.75 * summary["boundary_center_delta_error"]
        + 0.50 * summary["rollout_center_error"]
        + 0.50 * summary["boundary_curvature_center_error"]
        + 0.25 * summary["head_motion_error"]
        + 0.25 * summary["boundary_motion_delta_error"]
        + 0.15 * summary["rollout_motion_error"]
        + 0.15 * summary["boundary_curvature_motion_error"]
        + 0.10 * summary["head_log_scale_error"]
        + 0.10 * summary["boundary_log_scale_delta_error"]
        + 0.05 * summary["rollout_log_scale_error"]
        + 0.05 * summary["boundary_curvature_log_scale_error"]
    )
    summary["boundary_discontinuity_index_v1"] = (
        1.00 * summary["head_center_error"]
        + 1.00 * summary["boundary_center_delta_error"]
        + 0.50 * summary["head_motion_error"]
        + 0.50 * summary["boundary_motion_delta_error"]
        + 0.35 * summary["boundary_curvature_center_error"]
        + 0.20 * summary["boundary_jump_direction_error"]
        + 0.10 * summary["head_log_scale_error"]
        + 0.10 * summary["boundary_log_scale_delta_error"]
    )
    return summary


def choose_primary_object(target_context: np.ndarray, target_future: np.ndarray, rollout_steps: int) -> int:
    num_objects = target_future.shape[1]
    horizon = min(int(rollout_steps), int(target_future.shape[0]))
    best_idx = 0
    best_score = -1.0
    for obj_idx in range(num_objects):
        visible = (
            (target_future[:horizon, obj_idx, EXISTENCE_INDEX] > 0.5)
            | (target_future[:horizon, obj_idx, VISIBILITY_INDEX] > 0.2)
        )
        if not np.any(visible):
            continue
        jump = np.linalg.norm(
            target_future[0, obj_idx, CENTER_SLICE] - target_context[-1, obj_idx, CENTER_SLICE]
        )
        rollout = np.linalg.norm(
            target_future[min(horizon - 1, target_future.shape[0] - 1), obj_idx, CENTER_SLICE]
            - target_context[-1, obj_idx, CENTER_SLICE]
        )
        score = float(jump + 0.5 * rollout + visible.sum() * 1e-3)
        if score > best_score:
            best_score = score
            best_idx = obj_idx
    return best_idx


def collect_reports(comparison_root: Path) -> list[dict]:
    reports = []
    for path in sorted(comparison_root.glob("*/report.json")):
        reports.append(load_report(path))
    if not reports:
        raise FileNotFoundError(f"no report.json found under {comparison_root}")
    return reports


def ensure_cropped_half_video(
    source_path: Path,
    output_path: Path,
    *,
    side: str,
    default_fps: float = 6.0,
) -> None:
    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    if output_path.exists() and output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video for cropping: {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        fps = default_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 1 or height <= 0:
        capture.release()
        raise RuntimeError(f"invalid source video shape for cropping: {source_path} width={width} height={height}")
    half_width = width // 2
    if half_width <= 0:
        capture.release()
        raise RuntimeError(f"source video too narrow to split: {source_path} width={width}")
    x0 = 0 if side == "left" else half_width
    x1 = half_width if side == "left" else width
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (x1 - x0, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open cropped video writer: {output_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame[:, x0:x1])
    finally:
        writer.release()
        capture.release()


def build_svg_series(case_series: dict, width: int = 760, height: int = 240) -> str:
    padding_left = 42
    padding_right = 18
    padding_top = 20
    padding_bottom = 32
    inner_w = width - padding_left - padding_right
    inner_h = height - padding_top - padding_bottom
    labels = case_series["labels"]
    xs = np.asarray(case_series["x_values"], dtype=np.float32)
    ys = np.asarray(case_series["y_values"], dtype=np.float32)
    all_values = np.concatenate([xs.reshape(-1), ys.reshape(-1)], axis=0)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    if y_max - y_min < 1e-6:
        y_max = y_min + 1.0

    def px_x(idx: int) -> float:
        if len(labels) == 1:
            return padding_left + inner_w * 0.5
        return padding_left + inner_w * idx / (len(labels) - 1)

    def px_y(value: float) -> float:
        return padding_top + inner_h * (1.0 - (value - y_min) / (y_max - y_min))

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" class="metric-svg" xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="16" fill="#fffaf3" stroke="#e2d4c0"/>',
    ]
    for tick in range(5):
        ratio = tick / 4.0
        y = padding_top + inner_h * ratio
        value = y_max - (y_max - y_min) * ratio
        svg_parts.append(f'<line x1="{padding_left}" y1="{y:.1f}" x2="{width-padding_right}" y2="{y:.1f}" stroke="#ede1d0" stroke-width="1"/>')
        svg_parts.append(
            f'<text x="{padding_left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#766554">{value:.3f}</text>'
        )
    boundary_x = px_x(case_series["boundary_index"])
    svg_parts.append(
        f'<line x1="{boundary_x:.1f}" y1="{padding_top}" x2="{boundary_x:.1f}" y2="{height-padding_bottom}" stroke="#b8642a" stroke-width="2" stroke-dasharray="6 4"/>'
    )
    for idx, label in enumerate(labels):
        x = px_x(idx)
        svg_parts.append(f'<line x1="{x:.1f}" y1="{height-padding_bottom}" x2="{x:.1f}" y2="{height-padding_bottom+4}" stroke="#8b7a69" stroke-width="1"/>')
        svg_parts.append(
            f'<text x="{x:.1f}" y="{height-padding_bottom+18}" text-anchor="middle" font-size="11" fill="#766554">{html.escape(label)}</text>'
        )

    legend_y = 16
    legend_x = padding_left
    for model_name, series in case_series["series"].items():
        color = MODEL_COLORS.get(model_name, "#333333")
        x_points = [px_x(i) for i in range(len(labels))]
        x_path = " ".join(f"{x:.1f},{px_y(v):.1f}" for x, v in zip(x_points, series["x"]))
        y_path = " ".join(f"{x:.1f},{px_y(v):.1f}" for x, v in zip(x_points, series["y"]))
        svg_parts.append(f'<polyline points="{x_path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        svg_parts.append(f'<polyline points="{y_path}" fill="none" stroke="{color}" stroke-width="2.0" stroke-dasharray="6 4"/>')
        svg_parts.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" fill="{color}"/>')
        svg_parts.append(
            f'<text x="{legend_x+8}" y="{legend_y+4}" font-size="12" fill="#403326">{html.escape(model_name)} (solid=x, dashed=y)</text>'
        )
        legend_x += 140
    svg_parts.append("</svg>")
    return "".join(svg_parts)


def build_dashboard(
    reports: list[dict],
    output_dir: Path,
    *,
    rollout_steps: int,
    rollout_decay: float,
    asset_prefix: str,
) -> dict:
    compare_reports = {report["_label_b"]: report for report in reports}
    known_labels = [label for label in ORDERED_COMPARE_LABELS if label in compare_reports]
    extra_labels = sorted(label for label in compare_reports if label not in ORDERED_COMPARE_LABELS)
    available_labels = known_labels + extra_labels
    if not available_labels:
        raise ValueError("no expected comparison labels found")
    base_report = compare_reports[available_labels[0]]

    case_entries = []
    global_rows: dict[str, list[dict[str, float]]] = {"baseline": []}
    for label in available_labels:
        global_rows[label] = []

    for case_meta in base_report["cases"]:
        case_id = str(case_meta["case_id"])
        base_case_item = next(item for item in base_report["cases"] if str(item["case_id"]) == case_id)
        baseline_npz_path = base_report["_path"].parent / "assets" / f"{case_id}_comparison_outputs.npz"
        with np.load(baseline_npz_path, allow_pickle=False) as payload:
            target_context = payload["target_context_states"].astype(np.float32)
            target_future = payload["target_future_states"].astype(np.float32)
            predicted_context_baseline = payload["predicted_context_states_a"].astype(np.float32)
            predicted_future_baseline = payload["predicted_future_states_a"].astype(np.float32)

        primary_object = choose_primary_object(target_context, target_future, rollout_steps=rollout_steps)
        timeline_context = max(0, target_context.shape[0] - 3)
        future_horizon = min(target_future.shape[0], max(4, rollout_steps + 1))
        labels = [f"c{-offset}" for offset in range(target_context.shape[0] - timeline_context - 1, -1, -1)]
        labels.extend(f"f{idx}" for idx in range(future_horizon))
        boundary_index = target_context.shape[0] - timeline_context - 1

        baseline_metrics = compute_case_metrics(
            predicted_context_baseline,
            predicted_future_baseline,
            target_context,
            target_future,
            rollout_steps=rollout_steps,
            rollout_decay=rollout_decay,
        )
        global_rows["baseline"].append(baseline_metrics)

        series = {
            "gt": {
                "x": np.concatenate(
                    [
                        target_context[timeline_context:, primary_object, 0],
                        target_future[:future_horizon, primary_object, 0],
                    ]
                ).tolist(),
                "y": np.concatenate(
                    [
                        target_context[timeline_context:, primary_object, 1],
                        target_future[:future_horizon, primary_object, 1],
                    ]
                ).tolist(),
            },
            "baseline": {
                "x": np.concatenate(
                    [
                        predicted_context_baseline[timeline_context:, primary_object, 0],
                        predicted_future_baseline[:future_horizon, primary_object, 0],
                    ]
                ).tolist(),
                "y": np.concatenate(
                    [
                        predicted_context_baseline[timeline_context:, primary_object, 1],
                        predicted_future_baseline[:future_horizon, primary_object, 1],
                    ]
                ).tolist(),
            },
        }

        model_rows = [
            {
                "label": "baseline",
                **baseline_metrics,
            }
        ]

        for label in available_labels:
            report = compare_reports[label]
            npz_path = report["_path"].parent / "assets" / f"{case_id}_comparison_outputs.npz"
            with np.load(npz_path, allow_pickle=False) as payload:
                predicted_context = payload["predicted_context_states_b"].astype(np.float32)
                predicted_future = payload["predicted_future_states_b"].astype(np.float32)
            metrics = compute_case_metrics(
                predicted_context,
                predicted_future,
                target_context,
                target_future,
                rollout_steps=rollout_steps,
                rollout_decay=rollout_decay,
            )
            global_rows[label].append(metrics)
            model_rows.append({"label": label, **metrics})
            series[label] = {
                "x": np.concatenate(
                    [
                        predicted_context[timeline_context:, primary_object, 0],
                        predicted_future[:future_horizon, primary_object, 0],
                    ]
                ).tolist(),
                "y": np.concatenate(
                    [
                        predicted_context[timeline_context:, primary_object, 1],
                        predicted_future[:future_horizon, primary_object, 1],
                    ]
                ).tolist(),
            }

        case_entries.append(
            {
                "case_id": case_id,
                "split": case_meta.get("split", ""),
                "template_key": case_meta.get("template_key", ""),
                "prompt": case_meta.get("prompt", ""),
                "primary_object": primary_object,
                "context_video": f"{asset_prefix}/{base_report['_slug']}/{case_meta['context_video']}",
                "gt_video": f"{asset_prefix}/{base_report['_slug']}/{case_meta['gt_video']}",
                "baseline_state_video": f"cropped_assets/{case_id}_baseline_state.mp4",
                "baseline_condition_video": f"cropped_assets/{case_id}_baseline_condition.mp4",
                "compare_videos": [
                    {
                        "label": label,
                        "title": f"{label} overlay",
                        "state_video": f"cropped_assets/{case_id}_{label}_state.mp4",
                        "condition_video": f"cropped_assets/{case_id}_{label}_condition.mp4",
                        "state_compare_video": f"{asset_prefix}/{compare_reports[label]['_slug']}/"
                        f"{next(item for item in compare_reports[label]['cases'] if str(item['case_id']) == case_id)['state_compare_video']}",
                        "condition_compare_video": f"{asset_prefix}/{compare_reports[label]['_slug']}/"
                        f"{next(item for item in compare_reports[label]['cases'] if str(item['case_id']) == case_id)['condition_compare_video']}",
                    }
                    for label in available_labels
                ],
                "series_svg": build_svg_series(
                    {
                        "labels": labels,
                        "x_values": [series_item["x"] for series_item in series.values()],
                        "y_values": [series_item["y"] for series_item in series.values()],
                        "boundary_index": boundary_index,
                        "series": series,
                    }
                ),
                "models": model_rows,
            }
        )
        ensure_cropped_half_video(
            base_report["_path"].parent / str(base_case_item["state_compare_video"]),
            output_dir / case_entries[-1]["baseline_state_video"],
            side="left",
        )
        ensure_cropped_half_video(
            base_report["_path"].parent / str(base_case_item["condition_compare_video"]),
            output_dir / case_entries[-1]["baseline_condition_video"],
            side="left",
        )
        for video in case_entries[-1]["compare_videos"]:
            label = str(video["label"])
            compare_case_item = next(
                item for item in compare_reports[label]["cases"] if str(item["case_id"]) == case_id
            )
            ensure_cropped_half_video(
                compare_reports[label]["_path"].parent / str(compare_case_item["state_compare_video"]),
                output_dir / str(video["state_video"]),
                side="right",
            )
            ensure_cropped_half_video(
                compare_reports[label]["_path"].parent / str(compare_case_item["condition_compare_video"]),
                output_dir / str(video["condition_video"]),
                side="right",
            )
        model_by_label = {str(item["label"]): item for item in model_rows}
        baseline_bdi = float(model_by_label["baseline"]["boundary_discontinuity_index_v1"])
        ranked_labels = sorted(
            model_by_label.keys(),
            key=lambda label: float(model_by_label[label]["boundary_discontinuity_index_v1"]),
        )
        rank_by_label = {label: idx + 1 for idx, label in enumerate(ranked_labels)}
        enriched_compare_videos = []
        for video in case_entries[-1]["compare_videos"]:
            label = str(video["label"])
            metrics = model_by_label[label]
            model_bdi = float(metrics["boundary_discontinuity_index_v1"])
            enriched_compare_videos.append(
                {
                    **video,
                    "model_bdi": model_bdi,
                    "baseline_bdi": baseline_bdi,
                    "improvement_pct": 100.0 * (baseline_bdi - model_bdi) / max(abs(baseline_bdi), 1e-8),
                    "rank": rank_by_label[label],
                    "num_models": len(model_by_label),
                    "head_center_error": float(metrics["head_center_error"]),
                    "boundary_center_delta_error": float(metrics["boundary_center_delta_error"]),
                    "boundary_curvature_center_error": float(metrics["boundary_curvature_center_error"]),
                    "head_motion_error": float(metrics["head_motion_error"]),
                    "boundary_motion_delta_error": float(metrics["boundary_motion_delta_error"]),
                }
            )
        case_entries[-1]["compare_videos"] = enriched_compare_videos

    summary_rows = []
    metric_keys = [
        "boundary_discontinuity_index_v1",
        "head_center_error",
        "boundary_center_delta_error",
        "rollout_center_error",
        "boundary_curvature_center_error",
        "head_motion_error",
        "boundary_motion_delta_error",
        "boundary_validation_score_v1",
    ]
    for label, values in global_rows.items():
        row = {"label": label}
        for key in metric_keys:
            row[key] = float(np.mean([item[key] for item in values])) if values else None
        summary_rows.append(row)

    return {
        "comparison_root": str(output_dir),
        "rollout_steps": rollout_steps,
        "rollout_decay": rollout_decay,
        "summary": summary_rows,
        "cases": case_entries,
    }


def ensure_asset_link(output_dir: Path, comparison_root: Path) -> str:
    link_name = "comparison_assets"
    link_path = output_dir / link_name
    target_path = comparison_root.resolve()
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == target_path:
            return link_name
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            raise FileExistsError(f"{link_path} exists and is not a symlink; refusing to overwrite")
    os.symlink(target_path, link_path, target_is_directory=True)
    return link_name


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def fmt_signed_pct(value: float) -> str:
    return f"{value:+.1f}%"


def render_html(dashboard: dict) -> str:
    summary_rows = []
    for row in dashboard["summary"]:
        summary_rows.append(
            f"""
            <tr>
              <td>{html.escape(str(row['label']))}</td>
              <td>{fmt(row['boundary_discontinuity_index_v1'])}</td>
              <td>{fmt(row['head_center_error'])}</td>
              <td>{fmt(row['boundary_center_delta_error'])}</td>
              <td>{fmt(row['rollout_center_error'])}</td>
              <td>{fmt(row['boundary_curvature_center_error'])}</td>
              <td>{fmt(row['head_motion_error'])}</td>
              <td>{fmt(row['boundary_motion_delta_error'])}</td>
              <td>{fmt(row['boundary_validation_score_v1'])}</td>
            </tr>
            """
        )

    case_blocks = []
    for case in dashboard["cases"]:
        model_rows = []
        for model in case["models"]:
            model_rows.append(
                f"""
                <tr>
                  <td>{html.escape(str(model['label']))}</td>
                  <td>{fmt(model['boundary_discontinuity_index_v1'])}</td>
                  <td>{fmt(model['tail_center_error'])}</td>
                  <td>{fmt(model['head_center_error'])}</td>
                  <td>{fmt(model['boundary_center_delta_error'])}</td>
                  <td>{fmt(model['boundary_jump_magnitude_error'])}</td>
                  <td>{fmt(model['boundary_jump_direction_error'])}</td>
                  <td>{fmt(model['rollout_center_error'])}</td>
                  <td>{fmt(model['boundary_curvature_center_error'])}</td>
                  <td>{fmt(model['boundary_validation_score_v1'])}</td>
                </tr>
                """
            )
        video_cards = [
            f"""
            <article class="video-card">
              <div class="video-eyebrow">Reference</div>
              <h3>context</h3>
              <video controls preload="none" playsinline src="{html.escape(case['context_video'])}"></video>
            </article>
            """,
            f"""
            <article class="video-card">
              <div class="video-eyebrow">Reference</div>
              <h3>future gt</h3>
              <video controls preload="none" playsinline src="{html.escape(case['gt_video'])}"></video>
            </article>
            """,
            f"""
            <article class="video-card">
              <div class="video-eyebrow">Overlay</div>
              <h3>baseline</h3>
              <video controls preload="none" playsinline src="{html.escape(case['baseline_state_video'])}"></video>
              <details>
                <summary>展开 baseline condition overlay</summary>
                <video controls preload="none" playsinline src="{html.escape(case['baseline_condition_video'])}"></video>
              </details>
            </article>
            """,
        ]
        for video in case["compare_videos"]:
            video_cards.append(
                f"""
                <article class="video-card">
                  <div class="video-eyebrow">Overlay</div>
                  <h3>{html.escape(video['label'])}</h3>
                  <video controls preload="none" playsinline src="{html.escape(video['state_video'])}"></video>
                  <div class="video-metric-card">
                    <div class="video-metric-top">
                      <span class="metric-chip metric-primary">BDI-v1 {fmt(video['model_bdi'])}</span>
                      <span class="metric-chip">vs baseline {fmt_signed_pct(video['improvement_pct'])}</span>
                      <span class="metric-chip">rank #{int(video['rank'])}/{int(video['num_models'])}</span>
                    </div>
                    <div class="video-metric-sub">
                      <span>HeadCtr {fmt(video['head_center_error'])}</span>
                      <span>DeltaCtr {fmt(video['boundary_center_delta_error'])}</span>
                      <span>CurvCtr {fmt(video['boundary_curvature_center_error'])}</span>
                    </div>
                    <div class="video-metric-sub">
                      <span>HeadMotion {fmt(video['head_motion_error'])}</span>
                      <span>DeltaMotion {fmt(video['boundary_motion_delta_error'])}</span>
                    </div>
                  </div>
                  <details>
                    <summary>展开 condition overlay</summary>
                    <video controls preload="none" playsinline src="{html.escape(video['condition_video'])}"></video>
                  </details>
                </article>
                """
            )
        case_blocks.append(
            f"""
            <section class="case-card" id="{html.escape(case['case_id'])}">
              <div class="case-head">
                <div>
                  <div class="eyebrow">{html.escape(str(case['split']))} | {html.escape(str(case['template_key']))}</div>
                  <h2>{html.escape(case['case_id'])}</h2>
                  <p class="meta">primary object = {int(case['primary_object'])}</p>
                </div>
                <p class="prompt">{html.escape(str(case['prompt']))}</p>
              </div>
              <div class="chart-wrap">
                {case['series_svg']}
              </div>
              <div class="video-grid">
                {''.join(video_cards)}
              </div>
              <div class="metric-legend">
                <div class="metric-legend-title">这个 case 的指标含义</div>
                <div class="metric-legend-grid">
                  <div><code>Tail Ctr</code>: context 尾帧中心点绝对误差，用来看历史尾帧本身有没有被 predictor 预测偏。</div>
                  <div><code>Head Ctr</code>: future 首帧中心点绝对误差，直接看第一步落点是否对齐 GT。</div>
                  <div><code>Delta Ctr</code>: 从 <code>context[-1]</code> 跳到 <code>future[0]</code> 的中心位移误差，直接看边界跳变量对不对。</div>
                  <div><code>Jump Mag</code>: 边界跳变幅度误差，只比较“跳了多远”，不比较方向。</div>
                  <div><code>Jump Dir</code>: 边界跳变方向误差，比较位移方向是否和 GT 一致，越接近 0 越好。</div>
                  <div><code>Rollout Ctr</code>: future 前 {int(dashboard['rollout_steps'])} 步加权中心误差，越靠近边界的步权重越大，用来看接上之后短轨迹是否稳。</div>
                  <div><code>Curvature Ctr</code>: <code>context[-1], future[0], future[1]</code> 构成的二阶差分误差，用来看边界折点是否自然。</div>
                  <div><code>BDI-v1</code>: 边界不连续指数，只聚焦 <code>context[-1] -> future[0]</code> 这个边界现象。它综合了首步落点误差、边界跳变量误差、边界 motion、边界曲率、方向偏差和少量 scale 误差；越小表示边界越连续。</div>
                  <div><code>BVS-v1</code>: 聚合验证分数，综合首步误差、跳变量误差、短轨迹误差和曲率误差；越小表示边界整体越连续。</div>
                </div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>BDI-v1</th>
                    <th>Tail Ctr</th>
                    <th>Head Ctr</th>
                    <th>Delta Ctr</th>
                    <th>Jump Mag</th>
                    <th>Jump Dir</th>
                    <th>Rollout Ctr</th>
                    <th>Curvature Ctr</th>
                    <th>BVS-v1</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(model_rows)}
                </tbody>
              </table>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Boundary Validation Dashboard</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 252, 246, 0.97);
      --line: #dfd3c4;
      --ink: #201b16;
      --muted: #6f675d;
      --accent: #0f5a52;
      --accent2: #b8642a;
      --shadow: 0 18px 40px rgba(55, 40, 22, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 100, 42, 0.12), transparent 26%),
        radial-gradient(circle at top right, rgba(15, 90, 82, 0.12), transparent 22%),
        linear-gradient(180deg, #f7f3ea 0%, #efe5d8 100%);
    }}
    .page {{
      max-width: 1760px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .panel, .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 16px;
    }}
    .hero p, .meta, .prompt, li {{
      color: var(--muted);
      line-height: 1.7;
    }}
    .panel {{
      padding: 18px 20px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent2);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .case-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}
    .case-nav a {{
      text-decoration: none;
      color: var(--accent);
      border: 1px solid rgba(15, 90, 82, 0.18);
      background: rgba(15, 90, 82, 0.06);
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: #fcf8f2;
      border: 1px solid #eadfce;
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #eadfce;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f1e8db;
      color: #714724;
    }}
    .case-card {{
      padding: 20px;
      margin-bottom: 18px;
    }}
    .case-head {{
      display: grid;
      grid-template-columns: minmax(0, 380px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .case-head h2 {{
      margin: 0;
      font-size: 28px;
    }}
    .prompt {{
      margin: 0;
      background: rgba(255,255,255,0.6);
      border: 1px dashed #dbcdb9;
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .chart-wrap {{
      overflow-x: auto;
      margin-bottom: 14px;
    }}
    .metric-legend {{
      margin-bottom: 14px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid #e6dac8;
      background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(248,241,231,0.96));
    }}
    .metric-legend-title {{
      font-size: 14px;
      font-weight: 700;
      color: #5f4c39;
      margin-bottom: 10px;
    }}
    .metric-legend-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 16px;
      color: var(--muted);
      line-height: 1.7;
      font-size: 13px;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .video-card {{
      background: rgba(255,255,255,0.7);
      border: 1px solid #e6dac8;
      border-radius: 16px;
      padding: 12px;
    }}
    .video-card h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .video-metric-card {{
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 14px;
      background: #fcf6ee;
      border: 1px solid #eadfce;
    }}
    .video-metric-top, .video-metric-sub {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .video-metric-sub:last-child {{
      margin-bottom: 0;
    }}
    .metric-chip {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: #efe4d5;
      color: #5a4a37;
      font-size: 12px;
      font-weight: 700;
    }}
    .metric-primary {{
      background: #dfeee7;
      color: #144b42;
    }}
    .video-metric-sub span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    .video-eyebrow {{
      color: var(--accent2);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .metric-svg {{
      width: 100%;
      min-width: 720px;
      display: block;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #121212;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }}
    code {{
      background: rgba(15, 90, 82, 0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }}
    @media (max-width: 1200px) {{
      .case-head {{
        grid-template-columns: 1fr;
      }}
      .video-grid {{
        grid-template-columns: 1fr;
      }}
      .metric-legend-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Boundary Validation</div>
      <h1>Context Tail vs Future Head Dashboard</h1>
      <p>这个页面不是看最终 Wan 视频，而是专门验证 predictor 在边界处是不是连续。核心思路是把 <code>context[-1]</code> 到 <code>future[0]</code> 的问题拆成四层：首步绝对落点、边界跳变量、前几步短轨迹、跨边界二阶曲率。这样你可以先判断这些指标是否比单纯看整体 future MSE 更贴近你关心的“衔接顺不顺”。</p>
      <ul>
        <li><code>Head Ctr</code>: future 首帧中心点绝对误差，直接对应“第一步落点对不对”。</li>
        <li><code>Delta Ctr</code>: 边界中心跳变量误差，衡量从 context 尾帧跳到 future 首帧的位移是否对。</li>
        <li><code>Rollout Ctr</code>: future 前 {int(dashboard['rollout_steps'])} 步加权中心误差，衡量接上之后短轨迹是否稳。</li>
        <li><code>Curvature Ctr</code>: <code>context[-1], future[0], future[1]</code> 的二阶差分误差，衡量边界折点是否自然。</li>
        <li><code>BDI-v1</code>: 边界不连续指数，专门用来比较同一个 source case 的不同设置在边界处谁更断、谁更顺；越小越好。</li>
        <li><code>BVS-v1</code>: 一个聚合分数，按“首步 > 跳变 > 短轨迹 > 曲率”的优先级加权；如果它和肉眼判断一致，再考虑把它作为验证指标。</li>
      </ul>
      <div class="case-nav">
        {''.join(f'<a href="#{html.escape(case["case_id"])}">{html.escape(case["case_id"])}</a>' for case in dashboard["cases"])}
      </div>
    </section>

    <section class="panel">
      <div class="eyebrow">Global Summary</div>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>BDI-v1</th>
            <th>Head Ctr</th>
            <th>Delta Ctr</th>
            <th>Rollout Ctr</th>
            <th>Curvature Ctr</th>
            <th>Head Motion</th>
            <th>Delta Motion</th>
            <th>BVS-v1</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    </section>

    {''.join(case_blocks)}
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    comparison_root = Path(args.comparison_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_link_name = ensure_asset_link(output_dir, comparison_root)

    reports = collect_reports(comparison_root)
    dashboard = build_dashboard(
        reports,
        output_dir=output_dir,
        rollout_steps=args.rollout_steps,
        rollout_decay=args.rollout_decay,
        asset_prefix=asset_link_name,
    )
    (output_dir / "report.json").write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(dashboard), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
