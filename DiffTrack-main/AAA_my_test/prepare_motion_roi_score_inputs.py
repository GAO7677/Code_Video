#!/usr/bin/env python3
"""Build matched 25-frame motion-ROI score inputs for Stage1b, LoRA, and GT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DASHBOARD = Path(
    "/data/gaoya/agent-data/outputs/sam2_region_generation_comparison"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/sam2_region_motion_roi_scores"
)
DEFAULT_SAM_CACHE = Path(
    "/data/gaoya/agent-data/cache/toydataset_sam2_regions"
)
FFMPEG = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg")
MODEL_ORDER = ("stage1b", "lora", "gt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-dir", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sam-cache", type=Path, default=DEFAULT_SAM_CACHE)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--flow-floor", type=float, default=0.35)
    parser.add_argument("--mad-scale", type=float, default=6.0)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def read_video(path: Path, count: int, width: int, height: int) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while len(frames) < count:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA))
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if len(frames) < count:
        raise RuntimeError(f"{path}: requires {count} frames, decoded {len(frames)}")
    return np.stack(frames), fps, source_count


def write_h264(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_rgb.shape[1:3]
    if width % 2 or height % 2:
        raise ValueError(f"H.264 output dimensions must be even, got {width}x{height}")
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        intermediate = Path(temp_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create temporary video: {intermediate}")
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        subprocess.run(
            [
                str(FFMPEG), "-y", "-loglevel", "error", "-i", str(intermediate),
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ],
            check=True,
        )


def estimate_global_alignment(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(previous, maxCorners=600, qualityLevel=0.01, minDistance=6)
    if points is None or len(points) < 12:
        return np.eye(2, 3, dtype=np.float32)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if tracked is None or status is None:
        return np.eye(2, 3, dtype=np.float32)
    keep = status.reshape(-1) > 0
    if int(keep.sum()) < 12:
        return np.eye(2, 3, dtype=np.float32)
    # Warp the current frame back into the previous-frame coordinates.
    matrix, _ = cv2.estimateAffinePartial2D(
        tracked[keep], points[keep], method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    return matrix.astype(np.float32) if matrix is not None else np.eye(2, 3, dtype=np.float32)


def residual_flow_magnitude(frames: np.ndarray, flow_scale: float) -> np.ndarray:
    height, width = frames.shape[1:3]
    flow_width = max(64, int(round(width * flow_scale)))
    flow_height = max(64, int(round(height * flow_scale)))
    grays = [
        cv2.cvtColor(cv2.resize(frame, (flow_width, flow_height)), cv2.COLOR_RGB2GRAY)
        for frame in frames
    ]
    magnitudes = []
    for previous, current in zip(grays[:-1], grays[1:]):
        matrix = estimate_global_alignment(previous, current)
        aligned = cv2.warpAffine(
            current, matrix, (flow_width, flow_height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        flow = cv2.calcOpticalFlowFarneback(
            previous, aligned, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        magnitudes.append(np.linalg.norm(flow, axis=-1).astype(np.float32))
    return np.stack(magnitudes)


def clean_motion_mask(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    masks = []
    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((7, 7), np.uint8)
    min_area = max(32, int(round(magnitude.shape[1] * magnitude.shape[2] * 0.0005)))
    for frame_magnitude in magnitude:
        mask = (frame_magnitude >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                cleaned[labels == label] = 1
        masks.append(cleaned)
    return np.stack(masks).astype(np.uint8)


def dominant_motion_tube(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the largest temporally connected motion tube for one video."""
    temporal_union = masks.max(axis=0).astype(np.uint8)
    connected = cv2.morphologyEx(
        temporal_union, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    if count <= 1:
        return masks, temporal_union
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    support = (labels == largest).astype(np.uint8)
    support = cv2.dilate(support, np.ones((5, 5), np.uint8), iterations=1)
    restricted = masks * support[None]
    return restricted.astype(np.uint8), support


def object_support(cache_dir: Path, width: int, height: int) -> np.ndarray:
    metadata_path = cache_dir / "regions.json"
    archive_path = cache_dir / "regions.npz"
    if not metadata_path.is_file() or not archive_path.is_file():
        return np.zeros((height, width), dtype=np.uint8)
    metadata = json.loads(metadata_path.read_text())
    masks = np.load(archive_path)["masks_rhw"]
    support = np.zeros(masks.shape[1:], dtype=np.uint8)
    for index, region in enumerate(metadata["regions"]):
        if region.get("region_type") == "object":
            support |= (masks[index] > 0).astype(np.uint8)
    return cv2.resize(support, (width, height), interpolation=cv2.INTER_NEAREST)


def crop_box_from_support(
    support: np.ndarray, *, margin: float, min_width_ratio: float = 0.40,
    min_height_ratio: float = 0.50, aspect: float = 16 / 9,
) -> tuple[int, int, int, int]:
    height, width = support.shape
    rows, cols = np.where(support > 0)
    if rows.size == 0:
        return 0, 0, width, height
    x0, x1 = float(cols.min()), float(cols.max() + 1)
    y0, y1 = float(rows.min()), float(rows.max() + 1)
    box_width, box_height = x1 - x0, y1 - y0
    x0 -= box_width * margin
    x1 += box_width * margin
    y0 -= box_height * margin
    y1 += box_height * margin
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    crop_width = max(x1 - x0, width * min_width_ratio)
    crop_height = max(y1 - y0, height * min_height_ratio)
    if crop_width / crop_height < aspect:
        crop_width = crop_height * aspect
    else:
        crop_height = crop_width / aspect
    crop_width = min(crop_width, float(width))
    crop_height = min(crop_height, float(height))
    x0 = max(0.0, min(center_x - crop_width / 2, width - crop_width))
    y0 = max(0.0, min(center_y - crop_height / 2, height - crop_height))
    x1, y1 = x0 + crop_width, y0 + crop_height
    result = [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]
    result[0] -= result[0] % 2
    result[1] -= result[1] % 2
    result[2] -= result[2] % 2
    result[3] -= result[3] % 2
    if result[2] <= result[0] or result[3] <= result[1]:
        return 0, 0, width, height
    return tuple(result)


def overlay_frames(
    frames: np.ndarray, shared_masks: np.ndarray, crop_box: tuple[int, int, int, int]
) -> np.ndarray:
    output = []
    x0, y0, x1, y1 = crop_box
    for index, frame in enumerate(frames):
        mask = shared_masks[index] > 0
        rendered = frame.copy()
        red = np.zeros_like(rendered)
        red[..., 0] = 255
        mixed = cv2.addWeighted(rendered, 0.45, red, 0.55, 0)
        rendered[mask] = mixed[mask]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, (255, 255, 255), 2)
        cv2.rectangle(rendered, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 3)
        cv2.putText(
            rendered, "red: shared residual-flow motion | yellow: fixed score ROI",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
        )
        output.append(rendered)
    return np.stack(output)


def load_cases(dashboard_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads((dashboard_dir / "dashboard_data.json").read_text())
    models = {model["name"]: model for model in payload["models"]}
    first = models[MODEL_ORDER[0]]
    cases = []
    for case_index, case in enumerate(first["cases"]):
        model_entries = {}
        for model_name in MODEL_ORDER:
            model = models[model_name]
            model_case = model["cases"][case_index]
            if model_case["case_key"] != case["case_key"]:
                raise RuntimeError("Dashboard case order mismatch")
            model_entries[model_name] = {
                "label": model["label"],
                "video": str(
                    (dashboard_dir / model_case["asset_root"] / model["video_file"]).resolve()
                ),
            }
        cases.append(
            {"case_key": case["case_key"], "prompt": case["prompt"], "models": model_entries}
        )
    return cases, models


def process_case(args: argparse.Namespace, case: dict[str, Any]) -> None:
    case_dir = args.output_dir / "cases" / case["case_key"]
    complete_path = case_dir / "complete.json"
    if complete_path.is_file() and not args.force:
        print(f"[skip] {case['case_key']}", flush=True)
        return

    frame_sets: dict[str, np.ndarray] = {}
    source_info: dict[str, Any] = {}
    magnitudes = {}
    for model_name in MODEL_ORDER:
        source = Path(case["models"][model_name]["video"])
        frames, fps, source_count = read_video(
            source, args.num_frames, args.width, args.height
        )
        frame_sets[model_name] = frames
        source_info[model_name] = {
            "source_video": str(source), "source_frame_count": source_count, "fps": fps
        }
        magnitudes[model_name] = residual_flow_magnitude(frames, args.flow_scale)

    pooled = np.concatenate([value.reshape(-1) for value in magnitudes.values()])
    median = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - median)))
    threshold = max(args.flow_floor, median + args.mad_scale * 1.4826 * mad)
    raw_model_masks = {
        name: clean_motion_mask(magnitude, threshold) for name, magnitude in magnitudes.items()
    }
    model_masks = {}
    model_tube_support = {}
    for name, masks in raw_model_masks.items():
        model_masks[name], model_tube_support[name] = dominant_motion_tube(masks)
    shared_transition = np.maximum.reduce(list(model_masks.values()))
    temporal_kernel = np.ones((9, 9), np.uint8)
    shared_transition = np.stack(
        [cv2.dilate(mask, temporal_kernel, iterations=1) for mask in shared_transition]
    ).astype(np.uint8)
    shared_frames_small = np.concatenate(
        [shared_transition[:1], shared_transition], axis=0
    )
    shared_frames = np.stack(
        [cv2.resize(mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST) for mask in shared_frames_small]
    ).astype(np.uint8)

    sam_support = object_support(args.sam_cache / case["case_key"], args.width, args.height)
    spatial_support = np.maximum(shared_frames.max(axis=0), sam_support)
    spatial_support = cv2.dilate(spatial_support, np.ones((21, 21), np.uint8), iterations=1)
    crop_box = crop_box_from_support(spatial_support, margin=args.crop_margin)
    x0, y0, x1, y1 = crop_box

    case_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case_dir / "motion_masks.npz",
        shared_motion_frames=shared_frames,
        shared_motion_transitions_small=shared_transition,
        stage1b_raw_motion=raw_model_masks["stage1b"],
        lora_raw_motion=raw_model_masks["lora"],
        gt_raw_motion=raw_model_masks["gt"],
        stage1b_motion=model_masks["stage1b"],
        lora_motion=model_masks["lora"],
        gt_motion=model_masks["gt"],
        stage1b_tube_support=model_tube_support["stage1b"],
        lora_tube_support=model_tube_support["lora"],
        gt_tube_support=model_tube_support["gt"],
        sam_object_support=sam_support,
    )

    model_outputs = {}
    for model_name in MODEL_ORDER:
        model_dir = case_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        frames = frame_sets[model_name]
        full_path = model_dir / "wm_input_full25.mp4"
        roi_path = model_dir / "motion_roi_input.mp4"
        overlay_path = model_dir / "motion_roi_overlay.mp4"
        write_h264(full_path, frames, source_info[model_name]["fps"])
        write_h264(roi_path, frames[:, y0:y1, x0:x1], source_info[model_name]["fps"])
        write_h264(
            overlay_path, overlay_frames(frames, shared_frames, crop_box),
            source_info[model_name]["fps"],
        )
        model_outputs[model_name] = {
            **source_info[model_name],
            "wm_input_video": str(full_path),
            "motion_roi_input_video": str(roi_path),
            "motion_overlay_video": str(overlay_path),
        }

    metadata = {
        "status": "ok",
        "case_key": case["case_key"],
        "prompt": case["prompt"],
        "protocol": "matched_first_25_frames_shared_residual_flow_motion_roi",
        "canonical_size_wh": [args.width, args.height],
        "frame_count": args.num_frames,
        "global_motion_compensation": "RANSAC partial-affine before Farneback",
        "flow_processing_size_wh": [
            int(round(args.width * args.flow_scale)), int(round(args.height * args.flow_scale))
        ],
        "flow_threshold": threshold,
        "flow_threshold_rule": f"max({args.flow_floor}, pooled_median + {args.mad_scale} * 1.4826 * MAD)",
        "flow_median": median,
        "flow_mad": mad,
        "shared_motion_area_ratio": float(shared_frames.mean()),
        "crop_box_xyxy": list(crop_box),
        "crop_size_wh": [x1 - x0, y1 - y0],
        "crop_area_ratio": float((x1 - x0) * (y1 - y0) / (args.width * args.height)),
        "sam_object_support_area_ratio": float(sam_support.mean()),
        "models": model_outputs,
    }
    atomic_json(case_dir / "metadata.json", metadata)
    atomic_json(complete_path, {"status": "ok", "case_key": case["case_key"]})
    print(
        f"[ok] {case['case_key']} threshold={threshold:.3f} "
        f"motion={metadata['shared_motion_area_ratio']:.3f} crop={metadata['crop_area_ratio']:.3f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.dashboard_dir = args.dashboard_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    cases, _ = load_cases(args.dashboard_dir)
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        process_case(args, case)
    atomic_json(
        args.output_dir / "input_manifest.json",
        {"status": "ok", "case_count": len(cases), "cases": [case["case_key"] for case in cases]},
    )


if __name__ == "__main__":
    main()
