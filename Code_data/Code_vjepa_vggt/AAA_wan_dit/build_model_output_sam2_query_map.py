#!/usr/bin/env python3
"""Build model-specific Wan query tokens with GroundingDINO and SAM2.

Example:
CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/build_model_output_sam2_query_map.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model wan_lora \
  --video-root /data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/test5_allblocks_stability/generated/wan_lora \
  --output-dir /data/gaoya/agent-data/outputs/wan_dit_model_specific_query_maps/wan_lora
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.adapters.sam2_motion import (
    GroundingDINOTextDetector,
    SAM2MotionTracker,
)
from code_vjepa_vggt.utils.object_priors import sample_points_from_mask


TARGET_HEIGHT = 512
TARGET_WIDTH = 896
LATENT_TIMES = 13
GRID_HEIGHT = 16
GRID_WIDTH = 28
DEFAULT_PROMPT = (
    "sphere . ball . box . cube . block . cylinder . capsule . brick . "
    "jenga block . pillow . cardstock . paper ."
)
KNOWN_TERMS = (
    "tennis ball",
    "billiard ball",
    "jenga block",
    "cardstock",
    "cylinder",
    "capsule",
    "sphere",
    "brick",
    "pillow",
    "paper",
    "block",
    "cube",
    "box",
    "ball",
)


@dataclass
class Candidate:
    frame_idx: int
    box_xyxy: np.ndarray
    detector_score: float
    phrase: str
    local_motion: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--anchor-frames", default="0,8,24,40,48")
    parser.add_argument("--max-detections-per-anchor", type=int, default=6)
    parser.add_argument("--max-tracked-candidates", type=int, default=2)
    parser.add_argument("--sam2-segment-len", type=int, default=49)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--token-overlap-threshold", type=float, default=0.10)
    parser.add_argument("--max-query-tokens", type=int, default=8)
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--cotracker-num-queries", type=int, default=8)
    parser.add_argument("--case", action="append", default=[])
    return parser.parse_args()


def _read_video(path: Path, max_frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
            frame_rgb = cv2.resize(
                frame_rgb,
                (TARGET_WIDTH, TARGET_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame_rgb)
    capture.release()
    if not frames:
        raise RuntimeError(f"cannot read video: {path}")
    if len(frames) < max_frames:
        frames.extend([frames[-1].copy() for _ in range(max_frames - len(frames))])
    return np.stack(frames, axis=0)


def _case_prompt(caption: str) -> str:
    lower = caption.lower()
    terms = [term for term in KNOWN_TERMS if re.search(rf"\b{re.escape(term)}s?\b", lower)]
    if not terms:
        return DEFAULT_PROMPT
    deduped = list(dict.fromkeys(terms))
    return " . ".join(deduped) + " ."


def _gray_motion(frames_rgb: np.ndarray, frame_idx: int) -> np.ndarray:
    current = cv2.cvtColor(frames_rgb[frame_idx], cv2.COLOR_RGB2GRAY)
    maps = []
    for other in (frame_idx - 4, frame_idx + 4):
        if 0 <= other < len(frames_rgb):
            gray = cv2.cvtColor(frames_rgb[other], cv2.COLOR_RGB2GRAY)
            maps.append(cv2.absdiff(current, gray).astype(np.float32))
    if not maps:
        return np.zeros(current.shape, dtype=np.float32)
    return cv2.GaussianBlur(np.maximum.reduce(maps), (9, 9), 0)


def _box_motion(motion: np.ndarray, box: np.ndarray) -> float:
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0, x1 = sorted((max(0, x0), min(motion.shape[1], x1)))
    y0, y1 = sorted((max(0, y0), min(motion.shape[0], y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = motion[y0:y1, x0:x1]
    return float(np.percentile(crop, 80)) if crop.size else 0.0


def _candidate_priority(candidate: Candidate, motion_scale: float) -> float:
    motion = candidate.local_motion / max(motion_scale, 1.0e-6)
    return 0.65 * float(candidate.detector_score) + 0.35 * min(motion, 1.5)


def _detect_candidates(
    detector: GroundingDINOTextDetector,
    frames_tchw_01: np.ndarray,
    frames_rgb: np.ndarray,
    prompt: str,
    anchors: list[int],
    max_per_anchor: int,
    max_candidates: int,
) -> list[Candidate]:
    candidates = []
    for frame_idx in anchors:
        print(f"[sam2-query-map] detect anchor frame {frame_idx}", flush=True)
        detection = detector.detect(frames_tchw_01[frame_idx], prompt)
        print(
            f"[sam2-query-map] anchor frame {frame_idx}: "
            f"{len(detection.boxes_xyxy)} detections",
            flush=True,
        )
        motion = _gray_motion(frames_rgb, frame_idx)
        for box, score, phrase in zip(
            detection.boxes_xyxy[:max_per_anchor],
            detection.scores[:max_per_anchor],
            detection.phrases[:max_per_anchor],
        ):
            candidates.append(
                Candidate(
                    frame_idx=frame_idx,
                    box_xyxy=np.asarray(box, dtype=np.float32),
                    detector_score=float(score),
                    phrase=str(phrase),
                    local_motion=_box_motion(motion, box),
                )
            )
    if not candidates:
        return []
    motion_scale = float(np.percentile([item.local_motion for item in candidates], 80))
    candidates.sort(
        key=lambda item: _candidate_priority(item, motion_scale), reverse=True
    )
    selected = []
    for candidate in candidates:
        duplicate = False
        for existing in selected:
            if candidate.frame_idx != existing.frame_idx:
                continue
            ax0, ay0, ax1, ay1 = candidate.box_xyxy
            bx0, by0, bx1, by1 = existing.box_xyxy
            inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
                0.0, min(ay1, by1) - max(ay0, by0)
            )
            union = max(
                (ax1 - ax0) * (ay1 - ay0)
                + (bx1 - bx0) * (by1 - by0)
                - inter,
                1.0,
            )
            duplicate = inter / union >= 0.75
            if duplicate:
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return selected


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return TARGET_WIDTH / 2.0, TARGET_HEIGHT / 2.0
    return float(xs.mean()), float(ys.mean())


def _track_quality(masks: np.ndarray, detector_score: float) -> dict[str, float]:
    valid = np.asarray([int(mask.sum()) > 0 for mask in masks], dtype=bool)
    valid_ratio = float(valid.mean())
    if not valid.any():
        return {"score": -1.0, "valid_ratio": 0.0, "path": 0.0, "shape": 0.0}
    centroids = np.asarray([_mask_centroid(mask) for mask in masks], dtype=np.float32)
    sampled = centroids[np.arange(0, len(masks), 4)]
    path = (
        float(np.linalg.norm(np.diff(sampled, axis=0), axis=1).sum())
        / math.hypot(TARGET_HEIGHT, TARGET_WIDTH)
        if len(sampled) > 1
        else 0.0
    )
    shape_changes = []
    for first, second in zip(masks[:-4:4], masks[4::4]):
        union = np.logical_or(first, second).sum()
        if union > 0:
            shape_changes.append(
                1.0 - float(np.logical_and(first, second).sum()) / float(union)
            )
    shape = float(np.mean(shape_changes)) if shape_changes else 0.0
    score = 2.5 * path + 0.7 * shape + 0.5 * detector_score + 0.5 * valid_ratio
    return {
        "score": float(score),
        "valid_ratio": valid_ratio,
        "path": path,
        "shape": shape,
    }


def _valid_track_frames(masks: np.ndarray) -> np.ndarray:
    areas = masks.reshape(len(masks), -1).sum(axis=1).astype(np.float32)
    positive = areas[areas > 0]
    if positive.size == 0:
        return np.zeros((len(masks),), dtype=bool)
    median = float(np.median(positive))
    return (areas >= max(16.0, 0.10 * median)) & (areas <= 10.0 * median)


def _same_phrase(first: str, second: str) -> bool:
    first_terms = set(re.findall(r"[a-z]+", first.lower()))
    second_terms = set(re.findall(r"[a-z]+", second.lower()))
    aliases = {"sphere": "ball", "cube": "box", "brick": "block"}
    first_terms |= {aliases.get(term, term) for term in first_terms}
    second_terms |= {aliases.get(term, term) for term in second_terms}
    return bool(first_terms & second_terms)


def _compose_track_masks(
    tracks: list[tuple[float, Candidate, Any, dict[str, float]]],
    winner_index: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    winner_phrase = tracks[winner_index][1].phrase
    candidate_masks = [item[2].masks_thw.astype(np.uint8) for item in tracks]
    candidate_valid = [_valid_track_frames(masks) for masks in candidate_masks]
    fallback_order = sorted(
        range(len(tracks)),
        key=lambda index: tracks[index][0],
        reverse=True,
    )
    composite = np.zeros_like(candidate_masks[winner_index])
    source_per_frame = []
    unresolved = []
    for frame_idx in range(len(composite)):
        chosen = None
        if candidate_valid[winner_index][frame_idx]:
            chosen = winner_index
        else:
            for candidate_index in fallback_order:
                if not _same_phrase(
                    tracks[candidate_index][1].phrase, winner_phrase
                ):
                    continue
                if candidate_valid[candidate_index][frame_idx]:
                    chosen = candidate_index
                    break
        if chosen is None:
            unresolved.append(frame_idx)
            source_per_frame.append(-1)
            continue
        composite[frame_idx] = candidate_masks[chosen][frame_idx]
        source_per_frame.append(chosen)
    return composite, source_per_frame, unresolved


def _repair_masks_with_cotracker(
    masks: np.ndarray,
    source_per_frame: list[int],
    unresolved_frames: list[int],
    *,
    frames_rgb: np.ndarray,
    device: str,
    checkpoint: str,
    num_queries: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    if not unresolved_frames:
        return masks, source_per_frame, unresolved_frames
    valid_frames = [index for index, mask in enumerate(masks) if int(mask.sum()) > 0]
    if not valid_frames:
        return masks, source_per_frame, unresolved_frames
    anchor_idx = min(valid_frames)
    anchor_mask = masks[anchor_idx]
    query_points = sample_points_from_mask(
        anchor_mask, num_queries, avoid_edges=True
    )
    if query_points.shape[0] == 0:
        return masks, source_per_frame, unresolved_frames

    adapter = CoTrackerAdapter(
        checkpoint_path=checkpoint,
        num_queries=int(query_points.shape[0]),
        device=device,
        input_hw=(384, 512),
        window_len=60,
    )
    frames = (
        torch.from_numpy(frames_rgb)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device=device)
    )
    queries = torch.from_numpy(query_points).float().unsqueeze(0).to(device=device)
    frame_ids = torch.full(
        (1, int(query_points.shape[0]), 1),
        float(anchor_idx),
        dtype=torch.float32,
        device=device,
    )
    output = adapter(
        frames,
        query_points_prior=queries,
        query_frame_ids=frame_ids,
        query_image_hw=(TARGET_HEIGHT, TARGET_WIDTH),
    )
    tracks = output.tracks[0].detach().float().cpu().numpy()
    visibility = output.visibility[0].detach().float().cpu().numpy()
    anchor_center = np.median(query_points, axis=0)
    repaired = masks.copy()
    remaining = []
    for frame_idx in unresolved_frames:
        visible = visibility[frame_idx] > 0.5
        source_code = -2
        if not np.any(visible):
            points = tracks[frame_idx]
            usable = (
                np.isfinite(points).all(axis=1)
                & (points[:, 0] >= 0.0)
                & (points[:, 0] < TARGET_WIDTH)
                & (points[:, 1] >= 0.0)
                & (points[:, 1] < TARGET_HEIGHT)
            )
            if not np.any(usable):
                remaining.append(frame_idx)
                continue
            visible = usable
            source_code = -3
        if not np.any(visible):
            remaining.append(frame_idx)
            continue
        center = np.median(tracks[frame_idx, visible], axis=0)
        dx, dy = (float(value) for value in center - anchor_center)
        transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(
            anchor_mask.astype(np.uint8),
            transform,
            (TARGET_WIDTH, TARGET_HEIGHT),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if int(shifted.sum()) <= 0:
            remaining.append(frame_idx)
            continue
        repaired[frame_idx] = shifted
        source_per_frame[frame_idx] = source_code
    return repaired, source_per_frame, remaining


def _query_tokens(
    masks: np.ndarray,
    overlap_threshold: float,
    max_tokens: int,
) -> tuple[list[list[list[int]]], list[dict[str, float]]]:
    coords_per_time = []
    trajectory = []
    for latent_t in range(LATENT_TIMES):
        frame_idx = min(4 * latent_t, len(masks) - 1)
        mask = masks[frame_idx].astype(np.float32)
        if int(mask.sum()) <= 0:
            coords_per_time.append([])
            trajectory.append(
                {
                    "cx": None,
                    "cy": None,
                    "radius": None,
                    "area": 0.0,
                    "energy": 0.0,
                    "video_frame": frame_idx,
                    "valid": False,
                }
            )
            continue
        pooled = cv2.resize(
            mask,
            (GRID_WIDTH, GRID_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        candidates = np.argwhere(pooled >= overlap_threshold)
        if len(candidates) == 0:
            row, column = np.unravel_index(int(np.argmax(pooled)), pooled.shape)
            candidates = np.asarray([[row, column]], dtype=np.int64)
        candidates = sorted(
            candidates.tolist(),
            key=lambda rc: float(pooled[int(rc[0]), int(rc[1])]),
            reverse=True,
        )[:max_tokens]
        coords_per_time.append(
            [[latent_t, int(row), int(column)] for row, column in candidates]
        )
        cx, cy = _mask_centroid(mask)
        trajectory.append(
            {
                "cx": cx,
                "cy": cy,
                "radius": math.sqrt(max(float(mask.sum()), 1.0) / math.pi),
                "area": float(mask.sum()),
                "energy": float(pooled.max()),
                "video_frame": frame_idx,
                "valid": True,
            }
        )
    return coords_per_time, trajectory


def _render_outputs(
    frames_rgb: np.ndarray,
    masks: np.ndarray,
    coords_per_time: list[list[list[int]]],
    output_video: Path,
    preview: Path,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (TARGET_WIDTH, TARGET_HEIGHT),
    )
    preview_frames = []
    latent_lookup = {min(4 * time, len(frames_rgb) - 1): time for time in range(LATENT_TIMES)}
    for frame_idx, (frame_rgb, mask) in enumerate(zip(frames_rgb, masks)):
        canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        red = np.zeros_like(canvas)
        red[:, :, 2] = 255
        alpha = (mask.astype(np.float32) * 0.38)[:, :, None]
        canvas = (canvas * (1.0 - alpha) + red * alpha).astype(np.uint8)
        if frame_idx in latent_lookup:
            latent_t = latent_lookup[frame_idx]
            for _, row, column in coords_per_time[latent_t]:
                x0 = column * TARGET_WIDTH // GRID_WIDTH
                x1 = (column + 1) * TARGET_WIDTH // GRID_WIDTH
                y0 = row * TARGET_HEIGHT // GRID_HEIGHT
                y1 = (row + 1) * TARGET_HEIGHT // GRID_HEIGHT
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                f"frame {frame_idx:02d} / latent {latent_t:02d}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            preview_frames.append(cv2.resize(canvas, (448, 256)))
        writer.write(canvas)
    writer.release()
    rows = []
    for start in range(0, len(preview_frames), 4):
        chunk = preview_frames[start : start + 4]
        while len(chunk) < 4:
            chunk.append(np.zeros_like(preview_frames[0]))
        rows.append(np.concatenate(chunk, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(preview), sheet)


def _find_video(video_root: Path, case: str) -> Path:
    direct = video_root / f"{case}.mp4"
    if direct.is_file():
        return direct
    matches = list(video_root.rglob(f"{case}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one video for {case} under {video_root}, found {len(matches)}"
        )
    return matches[0]


def _write_gallery(output_dir: Path, model: str, cases: dict[str, Any]) -> None:
    cards = []
    for case, item in cases.items():
        preview = Path(item["preview"]).relative_to(output_dir)
        overlay = Path(item["overlay_video"]).relative_to(output_dir)
        cards.append(
            f"<section><h2>{html.escape(case)}</h2>"
            f"<p>{html.escape(item['target_phrase'])}; "
            f"score={item['track_quality']['score']:.3f}</p>"
            f"<img src='{preview.as_posix()}' loading='lazy'>"
            f"<video controls muted loop src='{overlay.as_posix()}'></video></section>"
        )
    page = f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(model)} model-specific queries</title>
<style>
body{{font:14px system-ui;margin:20px;background:#111;color:#eee}}
section{{margin:24px 0;border-top:1px solid #555;padding-top:12px}}
img,video{{display:block;max-width:100%;margin:8px 0}} h2{{font-size:18px}}
</style><h1>{html.escape(model)} model-specific SAM2 query locations</h1>
{''.join(cards)}
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def _query_map_payload(
    *,
    model: str,
    input_list: Path,
    video_root: Path,
    cases: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "input_list": str(input_list),
        "video_root": str(video_root),
        "target_shape": [TARGET_HEIGHT, TARGET_WIDTH],
        "grid": [LATENT_TIMES, GRID_HEIGHT, GRID_WIDTH],
        "query_method": (
            "model-output GroundingDINO anchors + full-video SAM2 masks "
            "+ CoTracker gap repair"
        ),
        "cases": cases,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_list = args.input_list.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = set(args.case)
    json_paths = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip()).expanduser().resolve()
        if path not in json_paths and (not selected_cases or path.stem in selected_cases):
            json_paths.append(path)

    detector = GroundingDINOTextDetector(
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_boxes=args.max_detections_per_anchor,
    )
    tracker = SAM2MotionTracker(
        device=args.device,
        segment_len=args.sam2_segment_len,
        enable_text_prompt=False,
    )
    raw_anchors = [int(value) for value in args.anchor_frames.split(",") if value]
    partial_path = output_dir / "query_map.partial.json"
    final_path = output_dir / "query_map.json"
    resume_path = final_path if final_path.is_file() else partial_path
    cases: dict[str, Any] = {}
    if resume_path.is_file():
        resume_payload = json.loads(resume_path.read_text(encoding="utf-8"))
        cases = dict(resume_payload.get("cases") or {})
        print(
            f"[sam2-query-map] resume {args.model}: {len(cases)} completed cases",
            flush=True,
        )
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        case = json_path.stem
        if case in cases:
            print(f"[sam2-query-map] {args.model} {case}: skip completed", flush=True)
            continue
        video_path = _find_video(video_root, case)
        frames_rgb = _read_video(video_path, args.max_frames)
        frames_tchw_01 = frames_rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        anchors = sorted({min(max(value, 0), len(frames_rgb) - 1) for value in raw_anchors})
        prompt = _case_prompt(str(payload.get("input_caption", "")))
        candidates = _detect_candidates(
            detector,
            frames_tchw_01,
            frames_rgb,
            prompt,
            anchors,
            args.max_detections_per_anchor,
            args.max_tracked_candidates,
        )
        if not candidates:
            raise RuntimeError(f"GroundingDINO found no candidate for {case}: {prompt}")
        tracks = []
        for candidate_index, candidate in enumerate(candidates):
            print(
                f"[sam2-query-map] {args.model} {case}: track candidate "
                f"{candidate_index + 1}/{len(candidates)} "
                f"{candidate.phrase}@{candidate.frame_idx}",
                flush=True,
            )
            output = tracker.track(
                frames_tchw_01,
                prompt_frame_idx=candidate.frame_idx,
                prompt_box_xyxy=candidate.box_xyxy,
                caption="",
            )
            quality = _track_quality(output.masks_thw, candidate.detector_score)
            tracks.append((quality["score"], candidate, output, quality))
        winner_index = max(range(len(tracks)), key=lambda index: tracks[index][0])
        _, winner, track, quality = tracks[winner_index]
        composite_masks, source_per_frame, unresolved_frames = _compose_track_masks(
            tracks, winner_index
        )
        if unresolved_frames:
            print(
                f"[sam2-query-map] {args.model} {case}: CoTracker repairs "
                f"{len(unresolved_frames)} unresolved SAM2 frames",
                flush=True,
            )
            composite_masks, source_per_frame, unresolved_frames = (
                _repair_masks_with_cotracker(
                    composite_masks,
                    source_per_frame,
                    unresolved_frames,
                    frames_rgb=frames_rgb,
                    device=args.device,
                    checkpoint=args.cotracker_checkpoint,
                    num_queries=args.cotracker_num_queries,
                )
            )
        unresolved_latent_frames = [
            frame_idx
            for frame_idx in (4 * time for time in range(LATENT_TIMES))
            if frame_idx in unresolved_frames
        ]
        if unresolved_latent_frames:
            print(
                f"[sam2-query-map] {args.model} {case}: skip no-visible-query "
                f"frames {unresolved_latent_frames}",
                flush=True,
            )
        coords_per_time, trajectory = _query_tokens(
            composite_masks,
            args.token_overlap_threshold,
            args.max_query_tokens,
        )
        case_dir = output_dir / "cases" / case
        case_dir.mkdir(parents=True, exist_ok=True)
        mask_path = case_dir / "sam2_masks.npz"
        np.savez_compressed(
            mask_path,
            masks=composite_masks.astype(np.uint8),
            candidate_masks=np.stack(
                [item[2].masks_thw.astype(np.uint8) for item in tracks], axis=0
            ),
            source_per_frame=np.asarray(source_per_frame, dtype=np.int16),
        )
        preview = case_dir / "query_preview.jpg"
        overlay_video = case_dir / "query_overlay.mp4"
        _render_outputs(
            frames_rgb,
            composite_masks,
            coords_per_time,
            overlay_video,
            preview,
        )
        cases[case] = {
            "input_json": str(json_path),
            "generated_video": str(video_path),
            "frame_shape": [TARGET_HEIGHT, TARGET_WIDTH],
            "grid": [LATENT_TIMES, GRID_HEIGHT, GRID_WIDTH],
            "prompt": prompt,
            "target_phrase": winner.phrase,
            "prompt_frame_idx": winner.frame_idx,
            "prompt_box_xyxy": [float(value) for value in winner.box_xyxy],
            "track_quality": quality,
            "winner_candidate_index": winner_index,
            "track_source_per_frame": source_per_frame,
            "fallback_frame_count": sum(
                int(index >= 0 and index != winner_index)
                for index in source_per_frame
            ),
            "cotracker_repair_frame_count": sum(
                int(index == -2) for index in source_per_frame
            ),
            "cotracker_low_visibility_repair_frame_count": sum(
                int(index == -3) for index in source_per_frame
            ),
            "unresolved_frames": unresolved_frames,
            "query_coords_per_time": coords_per_time,
            "query_tokens_per_time": [len(items) for items in coords_per_time],
            "trajectory": trajectory,
            "masks": str(mask_path),
            "preview": str(preview),
            "overlay_video": str(overlay_video),
            "candidate_tracks": [
                {
                    "phrase": candidate.phrase,
                    "frame_idx": candidate.frame_idx,
                    "detector_score": candidate.detector_score,
                    "local_motion": candidate.local_motion,
                    "track_quality": candidate_quality,
                }
                for _, candidate, _, candidate_quality in tracks
            ],
        }
        print(
            f"[sam2-query-map] {args.model} {case}: {winner.phrase} "
            f"score={quality['score']:.3f}",
            flush=True,
        )
        _write_json_atomic(
            partial_path,
            _query_map_payload(
                model=args.model,
                input_list=input_list,
                video_root=video_root,
                cases=cases,
            ),
        )

    result = _query_map_payload(
        model=args.model,
        input_list=input_list,
        video_root=video_root,
        cases=cases,
    )
    _write_json_atomic(final_path, result)
    _write_gallery(output_dir, args.model, cases)
    print(json.dumps({"cases": len(cases), "query_map": str(final_path)}))


if __name__ == "__main__":
    main()
