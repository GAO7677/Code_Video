#!/usr/bin/env python3
"""Detect rigid objects on aligned generated frames and report count increases."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

from build_context_guidance_comparisons import (
    case_stems,
    letterbox,
    parse_method,
    read_video,
    resolve_case_video,
)
from code_vjepa_vggt.adapters.sam2_motion import GroundingDINOTextDetector


DEFAULT_PROMPT = "box . cube . block . cylinder . capsule . sphere . ball ."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument("--method", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--frame-indices", default="0,7,8,16,24,32,40,48")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--analysis-score-threshold", type=float, default=0.25)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--dedupe-iou-threshold", type=float, default=0.60)
    parser.add_argument("--min-object-area-ratio", type=float, default=0.0002)
    parser.add_argument("--max-object-area-ratio", type=float, default=0.08)
    parser.add_argument("--tile-width", type=int, default=448)
    parser.add_argument("--tile-height", type=int, default=256)
    return parser.parse_args()


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def dedupe_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    phrases: list[str],
    *,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if len(boxes) == 0:
        return boxes, scores, phrases
    order = np.argsort(-scores)
    kept: list[int] = []
    for index in order.tolist():
        if any(box_iou(boxes[index], boxes[other]) >= iou_threshold for other in kept):
            continue
        kept.append(index)
    return boxes[kept], scores[kept], [phrases[index] for index in kept]


def filter_object_scale(
    boxes: np.ndarray,
    scores: np.ndarray,
    phrases: list[str],
    *,
    image_width: int,
    image_height: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    image_area = max(1.0, float(image_width * image_height))
    kept = []
    for index, box in enumerate(boxes):
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        ratio = area / image_area
        if min_area_ratio <= ratio <= max_area_ratio:
            kept.append(index)
    if not kept:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    return boxes[kept], scores[kept], [phrases[index] for index in kept]


def filter_detection_score(
    boxes: np.ndarray,
    scores: np.ndarray,
    phrases: list[str],
    *,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    kept = [index for index, score in enumerate(scores) if float(score) >= min_score]
    if not kept:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    return boxes[kept], scores[kept], [phrases[index] for index in kept]


def canonical_category(phrase: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", phrase.lower())
    if re.search(r"\b(ball|sphere)\b", normalized):
        return "ball"
    if re.search(r"\b(block|cube)\b", normalized):
        return "block"
    if re.search(r"\bcapsule\b", normalized):
        return "capsule"
    if re.search(r"\bcylinder\b", normalized):
        return "cylinder"
    return normalized.strip() or "unknown"


def draw_detections(
    frame: np.ndarray,
    *,
    label: str,
    frame_index: int,
    boxes: np.ndarray,
    scores: np.ndarray,
    phrases: list[str],
    width: int,
    height: int,
) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    offset_x = (width - round(source_w * scale)) // 2
    offset_y = (height - round(source_h * scale)) // 2
    canvas = letterbox(frame, width, height)
    colors = ((60, 220, 255), (255, 120, 80), (100, 230, 100), (220, 100, 230))
    for index, (box, score, phrase) in enumerate(zip(boxes, scores, phrases)):
        x0 = round(float(box[0]) * scale) + offset_x
        y0 = round(float(box[1]) * scale) + offset_y
        x1 = round(float(box[2]) * scale) + offset_x
        y1 = round(float(box[3]) * scale) + offset_y
        color = colors[index % len(colors)]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            canvas,
            f"{canonical_category(phrase)} {float(score):.2f}",
            (max(2, x0), max(42, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (width, 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        f"{label}  frame {frame_index:02d}  count={len(boxes)}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    args = parse_args()
    methods = [parse_method(value) for value in args.method]
    requested_indices = [int(value.strip()) for value in args.frame_indices.split(",") if value.strip()]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = GroundingDINOTextDetector(
        device=str(args.device),
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        max_boxes=int(args.max_boxes),
    )
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for stem in case_stems(args.case_list.expanduser().resolve()):
        method_rows: list[np.ndarray] = []
        for label, root in methods:
            video_path = resolve_case_video(root, stem)
            frames, _ = read_video(video_path)
            frame_indices = [min(max(0, index), len(frames) - 1) for index in requested_indices]
            tiles: list[np.ndarray] = []
            counts: dict[int, int] = {}
            for frame_index in frame_indices:
                frame = frames[frame_index]
                rgb_chw_01 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
                detected = detector.detect(rgb_chw_01, str(args.prompt))
                boxes, scores, phrases = filter_object_scale(
                    detected.boxes_xyxy,
                    detected.scores,
                    list(detected.phrases),
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
                    min_area_ratio=float(args.min_object_area_ratio),
                    max_area_ratio=float(args.max_object_area_ratio),
                )
                boxes, scores, phrases = filter_detection_score(
                    boxes,
                    scores,
                    phrases,
                    min_score=float(args.analysis_score_threshold),
                )
                boxes, scores, phrases = dedupe_detections(
                    boxes,
                    scores,
                    phrases,
                    iou_threshold=float(args.dedupe_iou_threshold),
                )
                counts[frame_index] = len(boxes)
                records.append(
                    {
                        "case": stem,
                        "method": label,
                        "video": str(video_path),
                        "frame_index": frame_index,
                        "count": len(boxes),
                        "boxes_xyxy": boxes.tolist(),
                        "scores": scores.tolist(),
                        "phrases": phrases,
                        "categories": [canonical_category(phrase) for phrase in phrases],
                    }
                )
                tiles.append(
                    draw_detections(
                        frame,
                        label=label,
                        frame_index=frame_index,
                        boxes=boxes,
                        scores=scores,
                        phrases=phrases,
                        width=int(args.tile_width),
                        height=int(args.tile_height),
                    )
                )
            method_rows.append(np.concatenate(tiles, axis=1))
            context_counts = [count for index, count in counts.items() if index < 8]
            future_counts = [count for index, count in counts.items() if index >= 8]
            context_max = max(context_counts, default=0)
            context_anchor = context_counts[-1] if context_counts else 0
            future_max = max(future_counts, default=0)
            future_min = min(future_counts, default=0)
            future_median = float(np.median(future_counts)) if future_counts else 0.0
            summaries.append(
                {
                    "case": stem,
                    "method": label,
                    "context_max_count": context_max,
                    "context_anchor_count": context_anchor,
                    "future_max_count": future_max,
                    "future_min_count": future_min,
                    "future_median_count": future_median,
                    "future_count_increase": max(0, future_max - context_max),
                    "future_count_increase_from_anchor": max(0, future_max - context_anchor),
                    "future_count_drop_from_context_max": max(0, context_max - future_min),
                    "future_count_drop_from_anchor": max(0, context_anchor - future_min),
                    "context_counts": json.dumps(context_counts),
                    "future_counts": json.dumps(future_counts),
                }
            )
        overlay_path = output_dir / f"{stem}_grounding_count_overlay.jpg"
        if not cv2.imwrite(str(overlay_path), np.concatenate(method_rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write overlay: {overlay_path}")
        print(overlay_path)

    records_path = output_dir / "detection_records.json"
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path = output_dir / "object_count_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(summary_path)


if __name__ == "__main__":
    main()
