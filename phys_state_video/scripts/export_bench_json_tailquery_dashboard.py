#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.experiment import compute_state_metrics
from phys_state_video.mask_tracking import (
    GroundingDINOTextDetector,
    SAM2VideoMaskTracker,
    build_caption_prompt_boxes,
    build_mask_track_outputs,
    build_proxy_prompt_box,
)
from phys_state_video.predictor_wan_state_v2 import resample_temporal_states
from phys_state_video.proxy_state import extract_primary_track, read_video_frames
from phys_state_video.schemas import StateIndex
from phys_state_video.utils import require_torch
from phys_state_video.wan_predictor_runtime import (
    build_predictor_latent_extractor,
    build_predictor_prompt_context_encoder,
    load_wan_state_predictor,
    resolve_predictor_wan_task,
)
from phys_state_video.wan_state_v2_helpers import (
    compute_future_latent_steps,
    split_context_future_camera,
    resample_camera_to_latent_steps,
)

torch = require_torch()


@dataclass(slots=True)
class CaseSpec:
    source_name: str
    source_index: int
    category: str
    source_video: str
    caption: str
    case_id: str | None = None
    clip_start: int | None = None
    context_steps: int | None = None
    future_steps: int | None = None
    height: int | None = None
    width: int | None = None
    source_total_frames: int | None = None


@dataclass(slots=True)
class ModelSpec:
    label: str
    checkpoint: Path


@dataclass(slots=True)
class LoadedModel:
    label: str
    checkpoint: Path
    predictor: object
    predictor_ckpt: dict
    wan_task: str
    camera_dim: int
    max_context_latent_steps: int
    max_future_latent_steps: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export tailquery_multictx vs tailquery_converge dashboard on random clips from bench jsons."
    )
    parser.add_argument("--bench-json-root", required=True)
    parser.add_argument("--wan-ckpt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", default=None)
    parser.add_argument("--sam2-cache-report", default=None)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Repeated model spec in the form label=/abs/path/to/checkpoint.pt",
    )
    parser.add_argument("--sample-per-json", type=int, default=5)
    parser.add_argument("--json-names", nargs="+", default=["A.json", "B.json", "D.json"])
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--context-min", type=int, default=4)
    parser.add_argument("--context-max", type=int, default=12)
    parser.add_argument("--future-min", type=int, default=8)
    parser.add_argument("--future-max", type=int, default=20)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18879)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latent-device", default=None)
    parser.add_argument("--prompt-device", default=None)
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="ti2v-5B")
    parser.add_argument("--sam2-gt-prompt-mode", choices=["proxy_box", "caption_gdino"], default="caption_gdino")
    parser.add_argument(
        "--sam2-config",
        default="/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--sam2-ckpt",
        default="/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt",
    )
    parser.add_argument("--gdino-repo-root", default="/home/gaoya/Grounded-SAM-2-main")
    parser.add_argument(
        "--gdino-config",
        default="/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/GroundingDINO_SwinT_OGC.cfg.py",
    )
    parser.add_argument(
        "--gdino-ckpt",
        default="/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--gdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.20)
    parser.add_argument("--gdino-max-boxes", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[ModelSpec]:
    parsed: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --model-spec {spec!r}; expected label=/abs/path/to/checkpoint.pt")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        raw_path = raw_path.strip()
        if not label or not raw_path:
            raise ValueError(f"invalid --model-spec {spec!r}")
        if label in seen:
            raise ValueError(f"duplicate model label {label!r}")
        seen.add(label)
        parsed.append(ModelSpec(label=label, checkpoint=Path(raw_path).resolve()))
    if not parsed:
        raise ValueError("at least one --model-spec is required")
    return parsed


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)


def draw_text(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = rgb.copy()
    for row, line in enumerate(lines):
        y = 22 + row * 20
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def write_mp4(path: Path, frames_tchw: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames, _, height, width = frames_tchw.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path}")
    for idx in range(num_frames):
        rgb = to_uint8_rgb(frames_tchw[idx])
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def make_labeled_video(frames_tchw: np.ndarray, lines: list[str]) -> np.ndarray:
    labeled = []
    for frame_idx, frame in enumerate(frames_tchw):
        rgb = to_uint8_rgb(frame)
        overlay = draw_text(rgb, [*lines, f"frame={frame_idx:02d}"])
        labeled.append(np.transpose(overlay, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(labeled, axis=0)


def normalized_box_to_pixels(box_xyxy: np.ndarray, height: int, width: int) -> tuple[int, int, int, int]:
    x0 = int(np.clip(box_xyxy[0] * width, 0, width - 1))
    y0 = int(np.clip(box_xyxy[1] * height, 0, height - 1))
    x1 = int(np.clip(box_xyxy[2] * width, 0, width - 1))
    y1 = int(np.clip(box_xyxy[3] * height, 0, height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def predict_box_from_state(state: np.ndarray, ref_box: np.ndarray | None) -> np.ndarray:
    cx = float(np.clip(state[StateIndex.CENTER_X], 0.0, 1.0))
    cy = float(np.clip(state[StateIndex.CENTER_Y], 0.0, 1.0))
    area = max(float(np.exp(np.clip(state[StateIndex.LOG_SCALE], -12.0, 2.0))), 1e-6)
    if ref_box is not None:
        ref_w = max(float(ref_box[2] - ref_box[0]), 1e-4)
        ref_h = max(float(ref_box[3] - ref_box[1]), 1e-4)
        aspect = np.clip(ref_w / ref_h, 0.2, 5.0)
    else:
        aspect = 1.0
    box_h = np.sqrt(area / aspect)
    box_w = area / max(box_h, 1e-6)
    x0 = np.clip(cx - box_w * 0.5, 0.0, 1.0)
    y0 = np.clip(cy - box_h * 0.5, 0.0, 1.0)
    x1 = np.clip(cx + box_w * 0.5, 0.0, 1.0)
    y1 = np.clip(cy + box_h * 0.5, 0.0, 1.0)
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _draw_gt_boxes(canvas: np.ndarray, gt_boxes: np.ndarray, gt_states: np.ndarray, frame_idx: int) -> None:
    height, width = canvas.shape[:2]
    green = (60, 210, 90)
    num_objects = int(gt_boxes.shape[1])
    for obj_idx in range(num_objects):
        state = gt_states[frame_idx, obj_idx]
        box = gt_boxes[frame_idx, obj_idx]
        if (
            float(state[StateIndex.EXISTENCE]) <= 0.3
            or float(state[StateIndex.VISIBILITY]) <= 0.2
            or float(box[2] - box[0]) <= 1e-4
            or float(box[3] - box[1]) <= 1e-4
        ):
            continue
        x0, y0, x1, y1 = normalized_box_to_pixels(box, height, width)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), green, 2, cv2.LINE_AA)
        cx = int(np.clip(state[StateIndex.CENTER_X] * width, 0, width - 1))
        cy = int(np.clip(state[StateIndex.CENTER_Y] * height, 0, height - 1))
        cv2.circle(canvas, (cx, cy), 3, green, -1, cv2.LINE_AA)


def apply_mask_tint(
    canvas: np.ndarray,
    mask_hw: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (60, 210, 90),
    alpha: float = 0.35,
) -> None:
    if mask_hw is None:
        return
    mask = np.asarray(mask_hw)
    if mask.ndim != 2:
        raise ValueError(f"expected mask with shape [H, W], got {tuple(mask.shape)}")
    active = mask > 0
    if not np.any(active):
        return
    tint = np.zeros_like(canvas, dtype=np.float32)
    tint[..., 0] = float(color[0])
    tint[..., 1] = float(color[1])
    tint[..., 2] = float(color[2])
    mixed = canvas.astype(np.float32)
    mixed[active] = mixed[active] * (1.0 - alpha) + tint[active] * alpha
    canvas[...] = np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def _draw_prediction_track(
    canvas: np.ndarray,
    predicted_future_states: np.ndarray,
    frame_idx: int,
    *,
    color: tuple[int, int, int],
    ref_boxes: list[np.ndarray | None],
    traj_points: list[list[tuple[int, int]]],
) -> None:
    height, width = canvas.shape[:2]
    if frame_idx < 0 or frame_idx >= int(predicted_future_states.shape[0]):
        return
    num_objects = int(predicted_future_states.shape[1])
    for obj_idx in range(num_objects):
        pred = predicted_future_states[frame_idx, obj_idx]
        if float(pred[StateIndex.EXISTENCE]) <= 0.3 or float(pred[StateIndex.VISIBILITY]) <= 0.2:
            continue
        pred_box = predict_box_from_state(pred, ref_boxes[obj_idx])
        px0, py0, px1, py1 = normalized_box_to_pixels(pred_box, height, width)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), color, 2, cv2.LINE_AA)
        px = int(np.clip(pred[StateIndex.CENTER_X] * width, 0, width - 1))
        py = int(np.clip(pred[StateIndex.CENTER_Y] * height, 0, height - 1))
        traj_points[obj_idx].append((px, py))
        if len(traj_points[obj_idx]) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(traj_points[obj_idx], dtype=np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.circle(canvas, (px, py), 3, color, -1, cv2.LINE_AA)


def draw_gt_future_overlay(
    future_frames: np.ndarray,
    gt_future_boxes: np.ndarray,
    gt_future_states: np.ndarray,
    *,
    label: str,
    gt_future_masks: np.ndarray | None = None,
) -> np.ndarray:
    frames = []
    for frame_idx in range(future_frames.shape[0]):
        rgb = to_uint8_rgb(future_frames[frame_idx])
        canvas = rgb.copy()
        if gt_future_masks is not None and frame_idx < int(gt_future_masks.shape[0]):
            apply_mask_tint(canvas, gt_future_masks[frame_idx])
        _draw_gt_boxes(canvas, gt_future_boxes, gt_future_states, frame_idx)
        canvas = draw_text(canvas, [label, "green=GT", f"frame={frame_idx:02d}"])
        frames.append(np.transpose(canvas, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


def choose_primary_object_index(states: np.ndarray, boxes: np.ndarray) -> int:
    if states.ndim != 3 or boxes.ndim != 3:
        raise ValueError("expected states [T,N,D] and boxes [T,N,4]")
    areas = np.maximum((boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]), 0.0)
    visibility = np.clip(states[..., StateIndex.VISIBILITY], 0.0, 1.0)
    existence = np.clip(states[..., StateIndex.EXISTENCE], 0.0, 1.0)
    scores = (areas * visibility * existence).mean(axis=0)
    return int(np.argmax(scores))


def slice_single_object_track(
    states: np.ndarray,
    boxes: np.ndarray,
    obj_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        states[:, obj_idx:obj_idx + 1].astype(np.float32),
        boxes[:, obj_idx:obj_idx + 1].astype(np.float32),
    )


def build_sam2_gt_track(
    frames_tchw: np.ndarray,
    *,
    prompt_frame_idx: int,
    caption: str,
    tracker: SAM2VideoMaskTracker,
    text_detector: GroundingDINOTextDetector | None,
    prompt_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    proxy_guidance_box = build_proxy_prompt_box(frames_tchw, prompt_frame_idx=prompt_frame_idx)
    prompt_boxes_xyxy = proxy_guidance_box[None]
    resolved_prompt_mode = "proxy_box"
    if prompt_mode == "caption_gdino" and text_detector is not None and caption.strip():
        detection = build_caption_prompt_boxes(
            frames_tchw,
            prompt_frame_idx=prompt_frame_idx,
            caption=caption,
            detector=text_detector,
            guidance_box_xyxy=proxy_guidance_box,
        )
        if detection.boxes_xyxy.shape[0] > 0:
            prompt_boxes_xyxy = detection.boxes_xyxy.astype(np.float32)
            resolved_prompt_mode = detection.prompt_mode
        else:
            resolved_prompt_mode = "proxy_box_fallback"
    outputs = build_mask_track_outputs(
        frames_tchw,
        prompt_frame_idx=int(prompt_frame_idx),
        prompt_boxes_xyxy=prompt_boxes_xyxy,
        prompt_mode=resolved_prompt_mode,
        tracker=tracker,
    )
    primary_idx = choose_primary_object_index(outputs.states, outputs.boxes)
    states, boxes = slice_single_object_track(outputs.states, outputs.boxes, primary_idx)
    masks = outputs.masks[:, primary_idx].astype(np.uint8)
    return (
        states,
        boxes,
        masks,
        prompt_boxes_xyxy.astype(np.float32),
        resolved_prompt_mode,
        primary_idx,
    )


def draw_single_future_overlay(
    future_frames: np.ndarray,
    gt_future_boxes: np.ndarray,
    gt_future_states: np.ndarray,
    predicted_future_states: np.ndarray,
    *,
    model_label: str,
    pred_color: tuple[int, int, int],
    context_last_boxes: list[np.ndarray | None],
    gt_future_masks: np.ndarray | None = None,
) -> np.ndarray:
    num_objects = int(predicted_future_states.shape[1])
    ref_boxes = context_last_boxes if context_last_boxes is not None else [None for _ in range(num_objects)]
    traj_points: list[list[tuple[int, int]]] = [[] for _ in range(num_objects)]
    frames = []
    for frame_idx in range(future_frames.shape[0]):
        rgb = to_uint8_rgb(future_frames[frame_idx])
        canvas = rgb.copy()
        if gt_future_masks is not None and frame_idx < int(gt_future_masks.shape[0]):
            apply_mask_tint(canvas, gt_future_masks[frame_idx])
        _draw_gt_boxes(canvas, gt_future_boxes, gt_future_states, frame_idx)
        _draw_prediction_track(
            canvas,
            predicted_future_states,
            frame_idx,
            color=pred_color,
            ref_boxes=ref_boxes,
            traj_points=traj_points,
        )
        lines = [model_label, "green=GT", "pred=red/blue", f"frame={frame_idx:02d}"]
        canvas = draw_text(canvas, lines)
        frames.append(np.transpose(canvas, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


def resample_predicted_states_to_frame_steps(predicted_states: np.ndarray, target_steps: int) -> np.ndarray:
    tensor = torch.from_numpy(predicted_states[None]).float()
    resized = resample_temporal_states(tensor, target_steps=target_steps)[0]
    return resized.cpu().numpy().astype(np.float32)


def choose_bench_cases(bench_json_root: Path, json_names: list[str], sample_per_json: int, seed: int) -> list[CaseSpec]:
    rng = random.Random(seed)
    chosen: list[CaseSpec] = []
    for json_name in json_names:
        payload = json.loads((bench_json_root / json_name).read_text(encoding="utf-8"))
        indices = list(range(len(payload)))
        rng.shuffle(indices)
        for source_index in indices[: min(sample_per_json, len(indices))]:
            item = payload[source_index]
            chosen.append(
                CaseSpec(
                    source_name=Path(json_name).stem,
                    source_index=int(source_index),
                    category=str(item.get("category") or "unknown"),
                    source_video=str(item["source_video"]),
                    caption=str(item.get("caption") or ""),
                )
            )
    return chosen


def load_manifest_cases(manifest_json: Path) -> list[CaseSpec]:
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    cases: list[CaseSpec] = []
    for item in payload.get("cases", []):
        cases.append(
            CaseSpec(
                case_id=str(item["case_id"]),
                source_name=str(item["source_name"]),
                source_index=int(item["source_index"]),
                category=str(item.get("category") or "unknown"),
                source_video=str(item["source_video"]),
                caption=str(item.get("caption") or ""),
                clip_start=int(item["clip_start"]),
                context_steps=int(item["context_steps"]),
                future_steps=int(item["future_steps"]),
                height=int(item["height"]),
                width=int(item["width"]),
                source_total_frames=int(item.get("source_total_frames", 0)),
            )
        )
    return cases


def load_sam2_cache_index(cache_report_path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(cache_report_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, object]] = {}
    for item in payload.get("cases", []):
        records[str(item["case_id"])] = dict(item)
    return records


def load_sam2_case_cache(cache_record: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, int]:
    npz_path = Path(str(cache_record["npz"]))
    payload = np.load(npz_path)
    states_full = payload["states"].astype(np.float32)
    boxes_full = payload["boxes"].astype(np.float32)
    masks_full = payload["masks"].astype(np.uint8)
    prompt_boxes_xyxy = payload["prompt_boxes_xyxy"].astype(np.float32)
    primary_idx = int(payload["primary_idx"][0])
    states = states_full[:, primary_idx:primary_idx + 1].astype(np.float32)
    boxes = boxes_full[:, primary_idx:primary_idx + 1].astype(np.float32)
    masks = masks_full[:, primary_idx].astype(np.uint8)
    prompt_mode = str(cache_record.get("prompt_mode") or "unknown")
    return states, boxes, masks, prompt_boxes_xyxy, prompt_mode, primary_idx


def resolve_clip_bounds(
    total_frames: int,
    *,
    temporal_stride: int,
    max_context_latent_steps: int,
    max_future_latent_steps: int,
    context_min: int,
    context_max: int,
    future_min: int,
    future_max: int,
    rng: random.Random,
) -> tuple[int, int, int]:
    candidates: list[tuple[int, int]] = []
    for context_steps in range(context_min, context_max + 1):
        if context_steps >= total_frames:
            continue
        context_latent_steps = 1 + max(context_steps - 1, 0) // temporal_stride
        if context_latent_steps > max_context_latent_steps:
            continue
        for future_steps in range(future_min, future_max + 1):
            if context_steps + future_steps > total_frames:
                continue
            future_latent_steps = compute_future_latent_steps(context_steps, future_steps, temporal_stride)
            if future_latent_steps > max_future_latent_steps:
                continue
            candidates.append((context_steps, future_steps))
    if not candidates:
        raise RuntimeError(
            "no valid random clip can satisfy the predictor latent-step limits: "
            f"total_frames={total_frames}, context=[{context_min},{context_max}], future=[{future_min},{future_max}]"
        )
    context_steps, future_steps = rng.choice(candidates)
    start_max = total_frames - (context_steps + future_steps)
    start_idx = rng.randint(0, max(start_max, 0))
    return start_idx, context_steps, future_steps


def build_episode_from_clip(frames_tchw: np.ndarray, context_steps: int, *, camera_dim: int) -> dict[str, np.ndarray]:
    track = extract_primary_track(frames_tchw)
    total_steps = int(frames_tchw.shape[0])
    future_steps = int(total_steps - context_steps)
    return {
        "context_frames": frames_tchw[:context_steps].astype(np.float32),
        "future_frames": frames_tchw[context_steps:].astype(np.float32),
        "full_frames": frames_tchw.astype(np.float32),
        "context_states": track.states[:context_steps].astype(np.float32),
        "future_states": track.states[context_steps:].astype(np.float32),
        "full_states": track.states.astype(np.float32),
        "context_boxes": track.boxes[:context_steps].astype(np.float32),
        "future_boxes": track.boxes[context_steps:].astype(np.float32),
        "full_boxes": track.boxes.astype(np.float32),
        "appearance": track.appearance.astype(np.float32),
        "camera_full": np.zeros((total_steps, camera_dim), dtype=np.float32),
        "future_steps": np.asarray([future_steps], dtype=np.int64),
    }


def build_batch(episode: dict[str, np.ndarray], prompt: str) -> tuple[dict[str, object], int]:
    future_steps = int(episode["future_frames"].shape[0])
    batch = {
        "context_frames": torch.from_numpy(episode["context_frames"][None]).float(),
        "future_states": torch.from_numpy(episode["future_states"][None]).float(),
        "camera": torch.from_numpy(episode["camera_full"][None]).float(),
        "prompts": [prompt],
    }
    return batch, future_steps


def load_models(
    model_specs: list[ModelSpec],
    *,
    device: str,
    latent_device: str,
    prompt_device: str,
    default_wan_task: str,
):
    loaded: list[LoadedModel] = []
    for spec in model_specs:
        predictor, predictor_ckpt = load_wan_state_predictor(str(spec.checkpoint), device)
        config = predictor_ckpt.get("config", {})
        loaded.append(
            LoadedModel(
                label=spec.label,
                checkpoint=spec.checkpoint,
                predictor=predictor,
                predictor_ckpt=predictor_ckpt,
                wan_task=resolve_predictor_wan_task(
                    predictor_ckpt,
                    default_wan_task=default_wan_task,
                    predictor_wan_task=None,
                ),
                camera_dim=int(config.get("camera_dim", 8)),
                max_context_latent_steps=int(config.get("max_context_latent_steps", 3)),
                max_future_latent_steps=int(config.get("max_future_latent_steps", 5)),
            )
        )
    wan_tasks = {item.wan_task for item in loaded}
    if len(wan_tasks) != 1:
        raise ValueError(f"all models must resolve to the same wan task, got {sorted(wan_tasks)}")
    shared_wan_task = next(iter(wan_tasks))
    reference_ckpt = loaded[0].predictor_ckpt
    latent_extractor = build_predictor_latent_extractor(
        wan_ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        device=latent_device,
        predictor_ckpt=reference_ckpt,
        default_wan_task=shared_wan_task,
        predictor_wan_task=shared_wan_task,
        context="bench_dashboard shared",
    )
    prompt_context_encoder = build_predictor_prompt_context_encoder(
        wan_ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        device=prompt_device,
        predictor_ckpt=reference_ckpt,
        default_wan_task=shared_wan_task,
        predictor_wan_task=shared_wan_task,
        context="bench_dashboard shared",
    )
    return loaded, latent_extractor, prompt_context_encoder


def run_predictor_for_batch(
    batch: dict[str, object],
    *,
    loaded_model: LoadedModel,
    latent_extractor,
    prompt_context_encoder,
    device: str,
) -> dict[str, object]:
    predictor_version = loaded_model.predictor_ckpt.get("predictor_version", "wan_state_v1")
    if predictor_version != "wan_state_v2_latent_time":
        raise ValueError(
            f"{loaded_model.label} expects wan_state_v2_latent_time checkpoint, got {predictor_version}"
        )
    with torch.no_grad():
        context_frames = batch["context_frames"]
        future_states = batch["future_states"]
        camera = batch["camera"].to(device)
        context_latents = latent_extractor.encode_context_frames_raw(context_frames).to(device)
        context_latent_steps = int(context_latents.shape[1])
        future_latent_steps = compute_future_latent_steps(
            context_steps=int(context_frames.shape[1]),
            future_steps=int(future_states.shape[1]),
            temporal_stride=latent_extractor.temporal_stride,
        )
        context_camera, future_camera = split_context_future_camera(
            camera,
            context_steps=int(context_frames.shape[1]),
            future_steps=int(future_states.shape[1]),
        )
        camera_latent = resample_camera_to_latent_steps(context_camera, context_latent_steps)
        future_camera_latent = resample_camera_to_latent_steps(future_camera, future_latent_steps)
        prompt_context, prompt_mask = prompt_context_encoder.encode_prompts(list(batch["prompts"]))
        outputs = loaded_model.predictor(
            context_latents=context_latents,
            camera=camera_latent,
            prompt_context=prompt_context.to(device),
            prompt_mask=prompt_mask.to(device),
            future_latent_steps=future_latent_steps,
            num_objects=int(future_states.shape[2]),
            future_camera=future_camera_latent,
        )
    return outputs


def last_valid_box(boxes: np.ndarray, obj_idx: int, end_exclusive: int) -> np.ndarray | None:
    for frame_idx in range(end_exclusive - 1, -1, -1):
        box = boxes[frame_idx, obj_idx]
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        if width > 1e-4 and height > 1e-4:
            return box.astype(np.float32)
    return None


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_server(output_dir: Path, port: int) -> int:
    log_path = output_dir / f"http_{port}.log"
    pid_path = output_dir / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            if is_port_open(port):
                return pid
        except Exception:
            pid_path.unlink(missing_ok=True)

    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(output_dir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            for sub in sorted(child.rglob("*"), reverse=True):
                if sub.is_file() or sub.is_symlink():
                    sub.unlink()
                elif sub.is_dir():
                    sub.rmdir()
            child.rmdir()


def render_html(report: dict) -> str:
    summary_rows = []
    for row in report["summary"]:
        summary_rows.append(
            f"""
            <tr>
              <td>{html.escape(row['label'])}</td>
              <td>{row['proxy_center_error_mean']:.4f}</td>
              <td>{row['proxy_log_scale_error_mean']:.4f}</td>
              <td>{row['proxy_visibility_error_mean']:.4f}</td>
              <td>{row['proxy_future_start_head_center_error_mean']:.4f}</td>
              <td>{row['sam2_center_error_mean']:.4f}</td>
              <td>{row['sam2_log_scale_error_mean']:.4f}</td>
              <td>{row['sam2_visibility_error_mean']:.4f}</td>
              <td>{row['sam2_future_start_head_center_error_mean']:.4f}</td>
            </tr>
            """
        )

    case_cards = []
    for case in report["cases"]:
        model_cards = []
        for model in case["models"]:
            model_cards.append(
                f"""
                <article class="video-card">
                  <div class="video-eyebrow">Model | Proxy GT</div>
                  <h3>{html.escape(model['label'])}</h3>
                  <video controls preload="none" playsinline src="{html.escape(model['proxy_video'])}"></video>
                  <div class="metric-box">
                    <div>Center ↓ {model['proxy_metrics']['center_error']:.4f}</div>
                    <div>Scale ↓ {model['proxy_metrics']['log_scale_error']:.4f}</div>
                    <div>Vis ↓ {model['proxy_metrics']['visibility_error']:.4f}</div>
                    <div>Head ↓ {model['proxy_metrics']['future_start_head_center_error']:.4f}</div>
                  </div>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Model | SAM2 GT</div>
                  <h3>{html.escape(model['label'])}</h3>
                  <video controls preload="none" playsinline src="{html.escape(model['sam2_video'])}"></video>
                  <div class="metric-box">
                    <div>Center ↓ {model['sam2_metrics']['center_error']:.4f}</div>
                    <div>Scale ↓ {model['sam2_metrics']['log_scale_error']:.4f}</div>
                    <div>Vis ↓ {model['sam2_metrics']['visibility_error']:.4f}</div>
                    <div>Head ↓ {model['sam2_metrics']['future_start_head_center_error']:.4f}</div>
                  </div>
                </article>
                """
            )
        case_cards.append(
            f"""
            <section class="case-card" id="{html.escape(case['case_id'])}">
              <div class="case-head">
                <div>
                  <div class="eyebrow">{html.escape(case['source_name'])} | {html.escape(case['category'])}</div>
                  <h2>{html.escape(case['case_id'])}</h2>
                  <p class="prompt">{html.escape(case['caption'])}</p>
                  <p class="meta">
                    source index={case['source_index']} |
                    source frames={case['source_total_frames']} |
                    clip start={case['clip_start']} |
                    context={case['context_steps']} |
                    future={case['future_steps']}
                  </p>
                </div>
              </div>
              <div class="video-grid">
                <article class="video-card">
                  <div class="video-eyebrow">Reference</div>
                  <h3>Input Context</h3>
                  <video controls preload="none" playsinline src="{html.escape(case['context_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Reference | Proxy GT</div>
                  <h3>Proxy GT Future</h3>
                  <video controls preload="none" playsinline src="{html.escape(case['proxy_gt_future_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Reference | SAM2 GT</div>
                  <h3>SAM2 GT Future</h3>
                  <video controls preload="none" playsinline src="{html.escape(case['sam2_gt_future_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Prompt</div>
                  <h3>SAM2 Prompt</h3>
                  <div class="metric-box">
                    <div>prompt mode: {html.escape(case['sam2_prompt_mode'])}</div>
                    <div>prompt boxes: {case['sam2_prompt_box_count']}</div>
                    <div>primary object idx: {case['sam2_primary_object_idx']}</div>
                  </div>
                </article>
                {''.join(model_cards)}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>bench json tailquery dashboard</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 252, 246, 0.97);
      --line: #dfd3c4;
      --ink: #201b16;
      --muted: #6f675d;
      --accent: #0f5a52;
      --accent2: #b8642a;
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
      max-width: 1800px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero p, .prompt, .meta {{
      color: var(--muted);
      line-height: 1.7;
    }}
    .eyebrow {{
      color: var(--accent2);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .case-card {{
      padding: 18px;
      margin-bottom: 18px;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .video-card {{
      background: #fcf8f2;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 12px;
    }}
    .video-eyebrow {{
      color: var(--accent2);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    h1, h2, h3 {{
      margin: 0 0 8px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
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
    .metric-box {{
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #f1e8db;
      color: #714724;
      line-height: 1.7;
      font-size: 13px;
    }}
    @media (max-width: 1200px) {{
      .video-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>bench json random clip dashboard</h1>
      <p>这里直接从 <code>A/B/D.json</code> 里抽样，每个 source video 随机切一段 clip，再用 caption 作为 prompt 跑 predictor。输入只展示截出的 context；输出只在 future 视频底图上叠加框。</p>
      <p>这页把同一批 predictor 预测结果分别叠在两套 GT 上做对照：一套是项目内的 <code>proxy GT</code>，一套是 <code>SAM2 GT</code>。当前页面支持两种 SAM2 GT 来源：直接在本环境生成，或从外部缓存读取 <code>caption_gdino -&gt; SAM2</code> 结果。颜色约定：绿色框/掩码是当前卡片对应的 GT，红色框是 <code>tailquery_multictx</code>，蓝色框是 <code>tailquery_multictx_converge</code>。下表指标统一按 <code>↓</code> 理解为越低越好。</p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Proxy Ctr ↓</th>
            <th>Proxy Scale ↓</th>
            <th>Proxy Vis ↓</th>
            <th>Proxy Head ↓</th>
            <th>SAM2 Ctr ↓</th>
            <th>SAM2 Scale ↓</th>
            <th>SAM2 Vis ↓</th>
            <th>SAM2 Head ↓</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    </section>
    {''.join(case_cards)}
  </div>
</body>
</html>"""


def main():
    parsed = parse_args()
    output_dir = Path(parsed.output_dir)
    if parsed.clean:
        clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = parsed.device or ("cuda" if torch.cuda.is_available() else "cpu")
    latent_device = parsed.latent_device or device
    prompt_device = parsed.prompt_device or device
    model_specs = parse_model_specs(parsed.model_spec)
    manifest_path = Path(parsed.manifest_json).resolve() if parsed.manifest_json else None
    sam2_cache_report_path = Path(parsed.sam2_cache_report).resolve() if parsed.sam2_cache_report else None
    if (manifest_path is None) != (sam2_cache_report_path is None):
        raise ValueError("--manifest-json and --sam2-cache-report must be provided together")
    use_external_sam2_cache = manifest_path is not None
    if use_external_sam2_cache:
        bench_cases = load_manifest_cases(manifest_path)
        sam2_cache_index = load_sam2_cache_index(sam2_cache_report_path)
    else:
        bench_cases = choose_bench_cases(
            Path(parsed.bench_json_root),
            json_names=parsed.json_names,
            sample_per_json=parsed.sample_per_json,
            seed=parsed.seed,
        )
        sam2_cache_index = {}

    global args
    args = parsed
    loaded_models, latent_extractor, prompt_context_encoder = load_models(
        model_specs,
        device=device,
        latent_device=latent_device,
        prompt_device=prompt_device,
        default_wan_task=parsed.wan_task,
    )
    max_context_latent_steps = min(item.max_context_latent_steps for item in loaded_models)
    max_future_latent_steps = min(item.max_future_latent_steps for item in loaded_models)
    camera_dim = min(item.camera_dim for item in loaded_models)
    temporal_stride = int(latent_extractor.temporal_stride)

    tracker = None
    text_detector = None
    if not use_external_sam2_cache:
        tracker = SAM2VideoMaskTracker(
            device=device,
            model_cfg=parsed.sam2_config,
            checkpoint_path=parsed.sam2_ckpt,
        )
        if parsed.sam2_gt_prompt_mode == "caption_gdino":
            text_detector = GroundingDINOTextDetector(
                repo_root=parsed.gdino_repo_root,
                config_path=parsed.gdino_config,
                checkpoint_path=parsed.gdino_ckpt,
                device=device,
                box_threshold=parsed.gdino_box_threshold,
                text_threshold=parsed.gdino_text_threshold,
                max_boxes=parsed.gdino_max_boxes,
            )

    model_aggregate: dict[str, dict[str, list[float]]] = {
        model.label: {
            "proxy_center_error": [],
            "proxy_log_scale_error": [],
            "proxy_visibility_error": [],
            "proxy_future_start_head_center_error": [],
            "sam2_center_error": [],
            "sam2_log_scale_error": [],
            "sam2_visibility_error": [],
            "sam2_future_start_head_center_error": [],
        }
        for model in loaded_models
    }

    cases = []
    rng = random.Random(parsed.seed)
    for case_idx, spec in enumerate(bench_cases):
        source_path = Path(spec.source_video)
        if not source_path.exists():
            print(f"skip missing source video: {source_path}")
            continue
        resize_height = int(spec.height) if spec.height is not None else int(parsed.height)
        resize_width = int(spec.width) if spec.width is not None else int(parsed.width)
        full_video = read_video_frames(
            source_path,
            resize_height=resize_height,
            resize_width=resize_width,
        )
        if use_external_sam2_cache:
            start_idx = int(spec.clip_start)
            context_steps = int(spec.context_steps)
            future_steps = int(spec.future_steps)
        else:
            start_idx, context_steps, future_steps = resolve_clip_bounds(
                int(full_video.shape[0]),
                temporal_stride=temporal_stride,
                max_context_latent_steps=max_context_latent_steps,
                max_future_latent_steps=max_future_latent_steps,
                context_min=parsed.context_min,
                context_max=parsed.context_max,
                future_min=parsed.future_min,
                future_max=parsed.future_max,
                rng=rng,
            )
        clip = full_video[start_idx : start_idx + context_steps + future_steps]
        proxy_episode = build_episode_from_clip(clip, context_steps, camera_dim=camera_dim)
        batch, target_future_steps = build_batch(proxy_episode, spec.caption)
        proxy_context_last_boxes = [
            last_valid_box(proxy_episode["full_boxes"], obj_idx, context_steps)
            for obj_idx in range(int(proxy_episode["full_boxes"].shape[1]))
        ]
        if use_external_sam2_cache:
            cache_record = sam2_cache_index.get(str(spec.case_id))
            if cache_record is None:
                print(f"skip missing sam2 cache for case: {spec.case_id}")
                continue
            sam2_states_full, sam2_boxes_full, sam2_masks_full, sam2_prompt_boxes_xyxy, sam2_prompt_mode, sam2_primary_object_idx = load_sam2_case_cache(cache_record)
        else:
            sam2_states_full, sam2_boxes_full, sam2_masks_full, sam2_prompt_boxes_xyxy, sam2_prompt_mode, sam2_primary_object_idx = build_sam2_gt_track(
                clip,
                prompt_frame_idx=context_steps - 1,
                caption=spec.caption,
                tracker=tracker,
                text_detector=text_detector,
                prompt_mode=parsed.sam2_gt_prompt_mode,
            )
        sam2_context_states = sam2_states_full[:context_steps].astype(np.float32)
        sam2_future_states = sam2_states_full[context_steps:].astype(np.float32)
        sam2_context_boxes = sam2_boxes_full[:context_steps].astype(np.float32)
        sam2_future_boxes = sam2_boxes_full[context_steps:].astype(np.float32)
        sam2_future_masks = sam2_masks_full[context_steps:].astype(np.uint8)
        sam2_context_last_boxes = [
            last_valid_box(sam2_boxes_full, obj_idx, context_steps)
            for obj_idx in range(int(sam2_boxes_full.shape[1]))
        ]

        stem = source_path.stem.replace(" ", "_")
        case_id = str(spec.case_id) if spec.case_id else f"{spec.source_name.lower()}_{spec.source_index:03d}_{case_idx:02d}_{stem}"
        context_video_rel = f"assets/{case_id}_context.mp4"
        proxy_gt_video_rel = f"assets/{case_id}_proxy_future_gt.mp4"
        sam2_gt_video_rel = f"assets/{case_id}_sam2_future_gt.mp4"
        write_mp4(
            output_dir / context_video_rel,
            make_labeled_video(
                proxy_episode["context_frames"],
                [
                    f"context | {spec.source_name}:{spec.source_index}",
                    f"category={spec.category}",
                    f"length={context_steps}",
                ],
            ),
            parsed.fps,
        )
        write_mp4(
            output_dir / proxy_gt_video_rel,
            draw_gt_future_overlay(
                proxy_episode["future_frames"],
                proxy_episode["future_boxes"],
                proxy_episode["future_states"],
                label=f"Proxy GT future | length={future_steps}",
            ),
            parsed.fps,
        )
        write_mp4(
            output_dir / sam2_gt_video_rel,
            draw_gt_future_overlay(
                proxy_episode["future_frames"],
                sam2_future_boxes,
                sam2_future_states,
                label=f"SAM2 GT future | length={future_steps}",
                gt_future_masks=sam2_future_masks,
            ),
            parsed.fps,
        )

        case_models = []
        for model in loaded_models:
            outputs = run_predictor_for_batch(
                batch,
                loaded_model=model,
                latent_extractor=latent_extractor,
                prompt_context_encoder=prompt_context_encoder,
                device=device,
            )
            pred_future_latent = outputs["future_state_predictions"][0].detach().cpu().numpy().astype(np.float32)
            pred_future = resample_predicted_states_to_frame_steps(pred_future_latent, target_steps=target_future_steps)
            proxy_metrics = compute_state_metrics(pred_future, proxy_episode["future_states"])
            proxy_metrics["future_start_head_center_error"] = float(
                np.linalg.norm(
                    pred_future[0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
                    - proxy_episode["future_states"][0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1],
                    axis=-1,
                ).mean()
            )
            sam2_metrics = compute_state_metrics(pred_future, sam2_future_states)
            sam2_metrics["future_start_head_center_error"] = float(
                np.linalg.norm(
                    pred_future[0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
                    - sam2_future_states[0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1],
                    axis=-1,
                ).mean()
            )
            for key, value in proxy_metrics.items():
                model_aggregate[model.label][f"proxy_{key}"].append(float(value))
            for key, value in sam2_metrics.items():
                model_aggregate[model.label][f"sam2_{key}"].append(float(value))

            pred_color = (228, 74, 62) if model.label == "tailquery_multictx" else (48, 118, 255)
            proxy_model_video_rel = f"assets/{case_id}_{model.label}_proxy.mp4"
            sam2_model_video_rel = f"assets/{case_id}_{model.label}_sam2.mp4"
            write_mp4(
                output_dir / proxy_model_video_rel,
                draw_single_future_overlay(
                    proxy_episode["future_frames"],
                    proxy_episode["future_boxes"],
                    proxy_episode["future_states"],
                    pred_future,
                    model_label=f"{model.label} | Proxy GT",
                    pred_color=pred_color,
                    context_last_boxes=proxy_context_last_boxes,
                ),
                parsed.fps,
            )
            write_mp4(
                output_dir / sam2_model_video_rel,
                draw_single_future_overlay(
                    proxy_episode["future_frames"],
                    sam2_future_boxes,
                    sam2_future_states,
                    pred_future,
                    model_label=f"{model.label} | SAM2 GT",
                    pred_color=pred_color,
                    context_last_boxes=sam2_context_last_boxes,
                    gt_future_masks=sam2_future_masks,
                ),
                parsed.fps,
            )
            case_models.append(
                {
                    "label": model.label,
                    "proxy_video": proxy_model_video_rel,
                    "sam2_video": sam2_model_video_rel,
                    "proxy_metrics": proxy_metrics,
                    "sam2_metrics": sam2_metrics,
                }
            )

        cases.append(
            {
                "case_id": case_id,
                "source_name": spec.source_name,
                "source_index": int(spec.source_index),
                "source_video": str(source_path),
                "source_total_frames": int(spec.source_total_frames) if spec.source_total_frames is not None else int(full_video.shape[0]),
                "category": spec.category,
                "caption": spec.caption,
                "clip_start": int(start_idx),
                "context_steps": int(context_steps),
                "future_steps": int(future_steps),
                "context_video": context_video_rel,
                "proxy_gt_future_video": proxy_gt_video_rel,
                "sam2_gt_future_video": sam2_gt_video_rel,
                "sam2_prompt_mode": sam2_prompt_mode,
                "sam2_prompt_box_count": int(sam2_prompt_boxes_xyxy.shape[0]),
                "sam2_primary_object_idx": int(sam2_primary_object_idx),
                "models": case_models,
            }
        )

    summary = []
    for model in loaded_models:
        metrics = model_aggregate[model.label]
        summary.append(
            {
                "label": model.label,
                "checkpoint": str(model.checkpoint),
                "proxy_center_error_mean": float(np.mean(metrics["proxy_center_error"])) if metrics["proxy_center_error"] else 0.0,
                "proxy_log_scale_error_mean": float(np.mean(metrics["proxy_log_scale_error"])) if metrics["proxy_log_scale_error"] else 0.0,
                "proxy_visibility_error_mean": float(np.mean(metrics["proxy_visibility_error"])) if metrics["proxy_visibility_error"] else 0.0,
                "proxy_future_start_head_center_error_mean": (
                    float(np.mean(metrics["proxy_future_start_head_center_error"]))
                    if metrics["proxy_future_start_head_center_error"]
                    else 0.0
                ),
                "sam2_center_error_mean": float(np.mean(metrics["sam2_center_error"])) if metrics["sam2_center_error"] else 0.0,
                "sam2_log_scale_error_mean": float(np.mean(metrics["sam2_log_scale_error"])) if metrics["sam2_log_scale_error"] else 0.0,
                "sam2_visibility_error_mean": float(np.mean(metrics["sam2_visibility_error"])) if metrics["sam2_visibility_error"] else 0.0,
                "sam2_future_start_head_center_error_mean": (
                    float(np.mean(metrics["sam2_future_start_head_center_error"]))
                    if metrics["sam2_future_start_head_center_error"]
                    else 0.0
                ),
            }
        )

    report = {
        "bench_json_root": parsed.bench_json_root,
        "output_dir": str(output_dir),
        "case_count": len(cases),
        "model_count": len(loaded_models),
        "sample_per_json": parsed.sample_per_json,
        "seed": parsed.seed,
        "summary": summary,
        "cases": cases,
        "mode": "bench_json_tailquery_dashboard",
        "sam2_source": "external_cache" if use_external_sam2_cache else "inline",
        "manifest_json": str(manifest_path) if manifest_path is not None else None,
        "sam2_cache_report": str(sam2_cache_report_path) if sam2_cache_report_path is not None else None,
        "port": parsed.port,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    if parsed.no_serve:
        print(f"exported: {output_dir}")
        return
    pid = start_server(output_dir, parsed.port)
    print(f"page: {output_dir / 'index.html'}")
    print(f"server: http://127.0.0.1:{parsed.port}")
    print(f"pid: {pid}")


if __name__ == "__main__":
    main()
