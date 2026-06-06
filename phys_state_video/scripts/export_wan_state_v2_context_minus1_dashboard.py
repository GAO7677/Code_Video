#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
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
from phys_state_video.predictor_wan_state_v2 import resample_temporal_states
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
    resample_camera_to_latent_steps,
    split_context_future_camera,
)

torch = require_torch()


@dataclass(slots=True)
class ModelSpec:
    label: str
    checkpoint: Path
    disable_future_camera: bool = False


@dataclass(slots=True)
class LoadedModel:
    label: str
    checkpoint: Path
    predictor: object
    predictor_ckpt: dict
    disable_future_camera: bool
    wan_task: str


TAILQUERY_CONTEXT_RATIOS = (1.0, 0.75, 0.50, 0.25)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a context[:-1] dashboard for wan_state_v2 predictors on the full video timeline."
    )
    parser.add_argument("--episode-root", required=True, help="Episode root containing val/test split folders.")
    parser.add_argument("--wan-ckpt-dir", required=True, help="Wan checkpoint directory used by predictor VAE/T5.")
    parser.add_argument("--output-dir", required=True, help="Directory for html/assets/json.")
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help="Repeated model spec in the form label=/abs/path/to/checkpoint.pt",
    )
    parser.add_argument(
        "--disable-future-camera-label",
        action="append",
        default=[],
        help="Optional repeated labels for which future_camera should be disabled during export.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18878)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="ti2v-5B")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def parse_model_specs(specs: list[str], disable_labels: set[str]) -> list[ModelSpec]:
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
        parsed.append(
            ModelSpec(
                label=label,
                checkpoint=Path(raw_path).resolve(),
                disable_future_camera=label in disable_labels,
            )
        )
    if not parsed:
        raise ValueError("at least one --model-spec is required")
    return parsed


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)


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


def draw_text(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = rgb.copy()
    for row, line in enumerate(lines):
        y = 22 + row * 20
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def choose_case_files(episode_root: Path, splits: list[str], max_cases: int) -> list[Path]:
    selected: list[Path] = []
    seen_templates: set[tuple[str, str]] = set()
    split_files: dict[str, list[Path]] = {}
    for split in splits:
        split_dir = episode_root / split
        if not split_dir.exists():
            continue
        split_files[split] = sorted(split_dir.glob("*.npz"))

    for split in splits:
        for path in split_files.get(split, []):
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            key = (split, meta.get("template_key", "unknown"))
            if key in seen_templates:
                continue
            selected.append(path)
            seen_templates.add(key)
            if len(selected) >= max_cases:
                return selected

    if len(selected) < max_cases:
        for split in splits:
            for path in split_files.get(split, []):
                if path in selected:
                    continue
                selected.append(path)
                if len(selected) >= max_cases:
                    return selected
    return selected


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


def last_valid_box(boxes: np.ndarray, obj_idx: int, end_exclusive: int) -> np.ndarray | None:
    for frame_idx in range(end_exclusive - 1, -1, -1):
        box = boxes[frame_idx, obj_idx]
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        if width > 1e-4 and height > 1e-4:
            return box.astype(np.float32)
    return None


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


def resample_predicted_states_to_frame_steps(predicted_states: np.ndarray, target_steps: int) -> np.ndarray:
    if predicted_states.ndim != 3:
        raise ValueError(f"expected predicted states [T, N, D], got {tuple(predicted_states.shape)}")
    tensor = torch.from_numpy(predicted_states[None]).float()
    resized = resample_temporal_states(tensor, target_steps=target_steps)[0]
    return resized.cpu().numpy().astype(np.float32)


def load_episode_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        context_frames = payload["context_frames"].astype(np.float32)
        future_frames = payload["future_frames"].astype(np.float32)
        context_states = payload["context_states"].astype(np.float32)
        future_states = payload["future_states"].astype(np.float32)
        context_boxes = payload["context_boxes"].astype(np.float32)
        future_boxes = payload["future_boxes"].astype(np.float32)
        appearance = payload["appearance"].astype(np.float32)
        if "camera_full" in payload:
            camera_full = payload["camera_full"].astype(np.float32)
        else:
            camera = payload["camera"].astype(np.float32)
            total_steps = int(context_frames.shape[0] + future_frames.shape[0])
            if camera.shape[0] < total_steps:
                pad = np.repeat(camera[-1:], repeats=total_steps - camera.shape[0], axis=0)
                camera_full = np.concatenate([camera, pad.astype(np.float32)], axis=0)
            else:
                camera_full = camera[:total_steps].astype(np.float32)
    return {
        "context_frames": context_frames,
        "future_frames": future_frames,
        "full_frames": np.concatenate([context_frames, future_frames], axis=0).astype(np.float32),
        "context_states": context_states,
        "future_states": future_states,
        "full_states": np.concatenate([context_states, future_states], axis=0).astype(np.float32),
        "context_boxes": context_boxes,
        "future_boxes": future_boxes,
        "full_boxes": np.concatenate([context_boxes, future_boxes], axis=0).astype(np.float32),
        "appearance": appearance,
        "camera_full": camera_full.astype(np.float32),
    }


def build_batch_with_context_steps(
    episode: dict[str, np.ndarray],
    prompt: str,
    context_steps: int,
) -> tuple[dict[str, object], int]:
    full_frames = episode["full_frames"]
    full_states = episode["full_states"]
    if context_steps <= 0 or context_steps >= int(full_frames.shape[0]):
        raise ValueError(
            f"context_steps must be in [1, {int(full_frames.shape[0]) - 1}], got {context_steps}"
        )
    future_steps = int(full_frames.shape[0] - context_steps)
    batch = {
        "context_frames": torch.from_numpy(full_frames[None, :context_steps]).float(),
        "future_states": torch.from_numpy(full_states[None, context_steps:]).float(),
        "camera": torch.from_numpy(episode["camera_full"][None]).float(),
        "prompts": [prompt],
    }
    return batch, future_steps


def build_trimmed_batch(episode: dict[str, np.ndarray], prompt: str) -> tuple[dict[str, object], int, int]:
    input_context_steps = max(int(episode["context_frames"].shape[0]) - 1, 1)
    batch, future_steps = build_batch_with_context_steps(episode, prompt, input_context_steps)
    return batch, input_context_steps, future_steps


def resolve_ratio_context_steps(total_context_steps: int, ratio: float) -> int:
    if total_context_steps <= 0:
        raise ValueError(f"total_context_steps must be positive, got {total_context_steps}")
    steps = int(round(total_context_steps * ratio))
    return min(total_context_steps, max(1, steps))


def _draw_prediction_track(
    canvas: np.ndarray,
    predicted_future_states: np.ndarray,
    future_start_idx: int,
    frame_idx: int,
    *,
    color: tuple[int, int, int],
    ref_boxes: list[np.ndarray | None],
    traj_points: list[list[tuple[int, int]]],
) -> None:
    height, width = canvas.shape[:2]
    if frame_idx < future_start_idx:
        return
    pred_step = int(frame_idx - future_start_idx)
    if pred_step < 0 or pred_step >= int(predicted_future_states.shape[0]):
        return
    num_objects = int(predicted_future_states.shape[1])
    for obj_idx in range(num_objects):
        pred = predicted_future_states[pred_step, obj_idx]
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
        vx = int(np.clip(px + pred[StateIndex.VEL_X] * width * 0.25, 0, width - 1))
        vy = int(np.clip(py + pred[StateIndex.VEL_Y] * height * 0.25, 0, height - 1))
        cv2.arrowedLine(canvas, (px, py), (vx, vy), color, 2, cv2.LINE_AA, tipLength=0.25)


def _draw_gt_boxes(
    canvas: np.ndarray,
    gt_boxes: np.ndarray,
    gt_states: np.ndarray,
    frame_idx: int,
) -> None:
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


def draw_single_future_overlay(
    future_frames: np.ndarray,
    gt_future_boxes: np.ndarray,
    gt_future_states: np.ndarray,
    predicted_future_states: np.ndarray,
    *,
    model_label: str,
    context_mode_label: str,
    pred_color: tuple[int, int, int],
    context_last_boxes: list[np.ndarray | None] | None = None,
) -> np.ndarray:
    num_objects = int(predicted_future_states.shape[1])
    ref_boxes = context_last_boxes if context_last_boxes is not None else [None for _ in range(num_objects)]
    traj_points: list[list[tuple[int, int]]] = [[] for _ in range(num_objects)]
    frames = []
    for frame_idx in range(future_frames.shape[0]):
        rgb = to_uint8_rgb(future_frames[frame_idx])
        canvas = rgb.copy()
        _draw_gt_boxes(canvas, gt_future_boxes, gt_future_states, frame_idx)
        lines = [
            f"{model_label} | {context_mode_label} | green=GT",
        ]
        lines.append(f"frame={frame_idx:02d}")
        _draw_prediction_track(
            canvas,
            predicted_future_states,
            0,
            frame_idx,
            color=pred_color,
            ref_boxes=ref_boxes,
            traj_points=traj_points,
        )
        canvas = draw_text(canvas, lines)
        frames.append(np.transpose(canvas, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


def make_labeled_video(frames_tchw: np.ndarray, lines: list[str]) -> np.ndarray:
    labeled = []
    for frame_idx, frame in enumerate(frames_tchw):
        rgb = to_uint8_rgb(frame)
        overlay = draw_text(rgb, [*lines, f"frame={frame_idx:02d}"])
        labeled.append(np.transpose(overlay, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(labeled, axis=0)


def load_models(
    model_specs: list[ModelSpec],
    *,
    device: str,
    default_wan_task: str,
) -> tuple[list[LoadedModel], object, object]:
    loaded: list[LoadedModel] = []
    for spec in model_specs:
        predictor, predictor_ckpt = load_wan_state_predictor(str(spec.checkpoint), device)
        loaded.append(
            LoadedModel(
                label=spec.label,
                checkpoint=spec.checkpoint,
                predictor=predictor,
                predictor_ckpt=predictor_ckpt,
                disable_future_camera=spec.disable_future_camera,
                wan_task=resolve_predictor_wan_task(
                    predictor_ckpt,
                    default_wan_task=default_wan_task,
                    predictor_wan_task=None,
                ),
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
        device=device,
        predictor_ckpt=reference_ckpt,
        default_wan_task=shared_wan_task,
        predictor_wan_task=shared_wan_task,
        context="context_minus1_dashboard shared",
    )
    prompt_context_encoder = build_predictor_prompt_context_encoder(
        wan_ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        device=device,
        predictor_ckpt=reference_ckpt,
        default_wan_task=shared_wan_task,
        predictor_wan_task=shared_wan_task,
        context="context_minus1_dashboard shared",
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
        context_frames = batch["context_frames"].to(device)
        future_states = batch["future_states"].to(device)
        camera = batch["camera"].to(device)
        context_latents = latent_extractor.encode_context_frames_raw(context_frames)
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
        future_camera_latent = None
        if not loaded_model.disable_future_camera:
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


def render_html(report: dict) -> str:
    summary_rows = []
    for row in report["summary"]:
        summary_rows.append(
            f"""
            <tr>
              <td>{html.escape(row['label'])}</td>
              <td>{row['center_error_mean']:.4f}</td>
              <td>{row['log_scale_error_mean']:.4f}</td>
              <td>{row['visibility_error_mean']:.4f}</td>
              <td>{row['future_start_head_center_error_mean']:.4f}</td>
            </tr>
            """
        )

    case_cards = []
    for case in report["cases"]:
        tailquery_model = next((model for model in case["models"] if model["label"] == "tailquery"), None)
        other_models = [model for model in case["models"] if model["label"] != "tailquery"]
        tailquery_ratio_variants = case.get("tailquery_ratio_variants", [])

        def _render_model_pair(model: dict, *, primary: bool) -> str:
            pair_class = "compare-grid primary-compare" if primary else "compare-grid"
            return f"""
            <div class="{pair_class}">
              <article class="video-card">
                <div class="video-eyebrow">Model</div>
                <h3>{html.escape(model['label'])} · context</h3>
                <video controls preload="none" playsinline src="{html.escape(model['normal_video'])}"></video>
                <div class="metric-box">
                  <div>Center {model['normal_metrics']['center_error']:.4f}</div>
                  <div>Scale {model['normal_metrics']['log_scale_error']:.4f}</div>
                  <div>Vis {model['normal_metrics']['visibility_error']:.4f}</div>
                </div>
              </article>
              <article class="video-card">
                <div class="video-eyebrow">Model</div>
                <h3>{html.escape(model['label'])} · context[:-1]</h3>
                <video controls preload="none" playsinline src="{html.escape(model['trimmed_video'])}"></video>
                <div class="metric-box">
                  <div>Center {model['trimmed_metrics']['center_error']:.4f}</div>
                  <div>Scale {model['trimmed_metrics']['log_scale_error']:.4f}</div>
                  <div>Vis {model['trimmed_metrics']['visibility_error']:.4f}</div>
                  <div>Head@new boundary {model['trimmed_metrics']['future_start_head_center_error']:.4f}</div>
                </div>
              </article>
            </div>
            """

        collapsed_pairs = "".join(_render_model_pair(model, primary=False) for model in other_models)
        ratio_cards = []
        for variant in tailquery_ratio_variants:
            ratio_cards.append(
                f"""
                <article class="video-card">
                  <div class="video-eyebrow">Tailquery Ratio</div>
                  <h3>{html.escape(variant['ratio_label'])}</h3>
                  <div class="mini-label">Input Context</div>
                  <video controls preload="none" playsinline src="{html.escape(variant['context_video'])}"></video>
                  <div class="mini-label">Future Overlay</div>
                  <video controls preload="none" playsinline src="{html.escape(variant['video'])}"></video>
                  <div class="metric-box">
                    <div>context steps {int(variant['context_steps'])}</div>
                    <div>future steps {int(variant['future_steps'])}</div>
                    <div>Center {variant['metrics']['center_error']:.4f}</div>
                    <div>Scale {variant['metrics']['log_scale_error']:.4f}</div>
                    <div>Vis {variant['metrics']['visibility_error']:.4f}</div>
                    <div>Head {variant['metrics']['future_start_head_center_error']:.4f}</div>
                  </div>
                </article>
                """
            )
        tailquery_block = (
            f"""
            <div class="feature-head">
              <h3>Tailquery 多长度主对比</h3>
              <p class="meta">同一个 case，下方直接比较 `tailquery` 在不同 context 比例 `100% / 75% / 50% / 25%` 下预测出的 future overlay。</p>
            </div>
            <div class="ratio-grid">
              {''.join(ratio_cards)}
            </div>
            """
            if ratio_cards
            else (
                _render_model_pair(tailquery_model, primary=True)
                if tailquery_model is not None
                else '<p class="meta">tailquery result is missing for this case.</p>'
            )
        )
        collapsed_block = (
            f"""
            <details class="collapsed-models">
              <summary>展开其它方法对比</summary>
              {(_render_model_pair(tailquery_model, primary=False) if tailquery_model is not None else '')}
              {collapsed_pairs}
            </details>
            """
            if collapsed_pairs or tailquery_model is not None
            else ""
        )
        case_cards.append(
            f"""
            <section class="case-card" id="{html.escape(case['case_id'])}">
              <div class="case-head">
                <div>
                  <div class="eyebrow">{html.escape(case['split'])} | {html.escape(case['template_key'])}</div>
                  <h2>{html.escape(case['case_id'])}</h2>
                  <p class="prompt">{html.escape(case['prompt'])}</p>
                  <p class="meta">original context={case['original_context_steps']} | trimmed context={case['trimmed_context_steps']} | trimmed future = normal future + 1 frame</p>
                </div>
              </div>
              <div class="video-grid">
                <article class="video-card">
                  <div class="video-eyebrow">Reference</div>
                  <h3>GT Future · context</h3>
                  <video controls preload="none" playsinline src="{html.escape(case['normal_future_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Reference</div>
                  <h3>GT Future · context[:-1]</h3>
                  <video controls preload="none" playsinline src="{html.escape(case['trimmed_future_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Reference</div>
                  <h3>Input Context</h3>
                  <video controls preload="none" playsinline src="{html.escape(case['normal_context_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Reference</div>
                  <h3>Input Context[:-1]</h3>
                  <video controls preload="none" playsinline src="{html.escape(case['trimmed_context_video'])}"></video>
                </article>
              </div>
              {tailquery_block}
              {collapsed_block}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>wan_state_v2 context[:-1] dashboard</title>
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
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .ratio-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 10px;
    }}
    .primary-compare {{
      margin-top: 10px;
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
    .mini-label {{
      margin: 10px 0 6px;
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }}
    .feature-head {{
      margin-top: 18px;
    }}
    .collapsed-models {{
      margin-top: 16px;
      background: #fcf8f2;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 12px;
    }}
    .collapsed-models summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 12px;
    }}
    @media (max-width: 1200px) {{
      .video-grid {{
        grid-template-columns: 1fr;
      }}
      .compare-grid {{
        grid-template-columns: 1fr;
      }}
      .ratio-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>wan_state_v2 context ablation dashboard</h1>
      <p>这里把两条 overlay 分开展示。每个模型都有两条单独视频：一条对应正常 <code>context</code>，一条对应 <code>context[:-1]</code>。两条视频都只在各自对应的 future 底图上画框。</p>
      <p>页面主区域优先展示同一个 case 下 `tailquery` 在不同 context 比例下的 future overlay 对比，其它方法先折叠收起。绿色框表示 GT；正常 <code>context</code> 的 overlay 里预测框是红色；<code>context[:-1]</code> 或其它缩短比例的 overlay 里预测框是蓝色。</p>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Center</th>
            <th>Scale</th>
            <th>Vis</th>
            <th>Head@new boundary</th>
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
    episode_root = Path(parsed.episode_root)
    if parsed.clean and output_dir.exists():
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
    output_dir.mkdir(parents=True, exist_ok=True)

    device = parsed.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = parse_model_specs(parsed.model_spec, set(parsed.disable_future_camera_label))

    global args
    args = parsed
    loaded_models, latent_extractor, prompt_context_encoder = load_models(
        model_specs,
        device=device,
        default_wan_task=parsed.wan_task,
    )

    model_aggregate: dict[str, dict[str, list[float]]] = {
        model.label: {
            "center_error": [],
            "log_scale_error": [],
            "visibility_error": [],
            "future_start_head_center_error": [],
        }
        for model in loaded_models
    }
    cases = []
    for path in choose_case_files(episode_root, parsed.splits, parsed.max_cases):
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        episode = load_episode_arrays(path)
        prompt = meta.get("prompt", "")
        batch_trimmed, trimmed_context_steps, future_steps = build_trimmed_batch(episode, prompt)
        original_context_steps = int(episode["context_frames"].shape[0])
        batch_full, future_steps_full = build_batch_with_context_steps(episode, prompt, original_context_steps)
        if future_steps_full != int(episode["future_frames"].shape[0]):
            raise ValueError(
                f"expected normal-context future steps {int(episode['future_frames'].shape[0])}, got {future_steps_full}"
            )
        full_frames = episode["full_frames"]
        full_states = episode["full_states"]
        full_boxes = episode["full_boxes"]
        future_start_idx = trimmed_context_steps
        future_frames_trimmed = full_frames[future_start_idx:]
        future_states_trimmed_gt = full_states[future_start_idx:]
        future_boxes_trimmed_gt = full_boxes[future_start_idx:]
        future_frames_normal = episode["future_frames"]
        future_states_normal_gt = episode["future_states"]
        future_boxes_normal_gt = episode["future_boxes"]
        blue_context_last_boxes = [
            last_valid_box(full_boxes, obj_idx, future_start_idx) for obj_idx in range(int(full_boxes.shape[1]))
        ]
        red_context_last_boxes = [
            last_valid_box(full_boxes, obj_idx, original_context_steps) for obj_idx in range(int(full_boxes.shape[1]))
        ]

        case_id = path.stem
        normal_future_video_rel = f"assets/{case_id}_future_normal_gt.mp4"
        trimmed_future_video_rel = f"assets/{case_id}_future_trimmed_gt.mp4"
        normal_context_rel = f"assets/{case_id}_normal_context.mp4"
        trimmed_context_rel = f"assets/{case_id}_trimmed_context.mp4"
        write_mp4(
            output_dir / normal_future_video_rel,
            make_labeled_video(
                future_frames_normal,
                [f"future video for context | length={future_frames_normal.shape[0]} | green=GT"],
            ),
            parsed.fps,
        )
        write_mp4(
            output_dir / trimmed_future_video_rel,
            make_labeled_video(
                future_frames_trimmed,
                [f"future video for context[:-1] | length={future_frames_trimmed.shape[0]} | green=GT"],
            ),
            parsed.fps,
        )
        write_mp4(
            output_dir / normal_context_rel,
            make_labeled_video(
                episode["context_frames"],
                [f"normal context input | length={episode['context_frames'].shape[0]}"],
            ),
            parsed.fps,
        )
        write_mp4(
            output_dir / trimmed_context_rel,
            make_labeled_video(
                full_frames[:trimmed_context_steps],
                [f"trimmed context input | using frames [0:{trimmed_context_steps - 1}]"],
            ),
            parsed.fps,
        )

        case_models = []
        tailquery_ratio_variants = []
        npz_payload = {
            "full_frames": full_frames,
            "full_states": full_states,
            "full_boxes": episode["full_boxes"],
            "trimmed_context_steps": np.asarray([trimmed_context_steps], dtype=np.int64),
            "future_steps": np.asarray([future_steps], dtype=np.int64),
        }
        for model in loaded_models:
            outputs_trimmed = run_predictor_for_batch(
                batch_trimmed,
                loaded_model=model,
                latent_extractor=latent_extractor,
                prompt_context_encoder=prompt_context_encoder,
                device=device,
            )
            outputs_full = run_predictor_for_batch(
                batch_full,
                loaded_model=model,
                latent_extractor=latent_extractor,
                prompt_context_encoder=prompt_context_encoder,
                device=device,
            )
            pred_future_latent = (
                outputs_trimmed["future_state_predictions"][0].detach().cpu().numpy().astype(np.float32)
            )
            pred_future = resample_predicted_states_to_frame_steps(pred_future_latent, target_steps=future_steps)
            pred_future_latent_full = (
                outputs_full["future_state_predictions"][0].detach().cpu().numpy().astype(np.float32)
            )
            pred_future_full = resample_predicted_states_to_frame_steps(
                pred_future_latent_full,
                target_steps=int(episode["future_frames"].shape[0]),
            )
            future_target_states = future_states_trimmed_gt
            predictor_metrics_trimmed = compute_state_metrics(pred_future, future_target_states)
            predictor_metrics_trimmed["future_start_head_center_error"] = float(
                np.linalg.norm(
                    pred_future[0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
                    - future_target_states[0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1],
                    axis=-1,
                ).mean()
            )
            predictor_metrics_normal = compute_state_metrics(pred_future_full, future_states_normal_gt)
            for key in model_aggregate[model.label]:
                model_aggregate[model.label][key].append(float(predictor_metrics_trimmed[key]))

            normal_rel_video = f"assets/{case_id}_{model.label}_context.mp4"
            trimmed_rel_video = f"assets/{case_id}_{model.label}_context_minus1.mp4"
            write_mp4(
                output_dir / normal_rel_video,
                draw_single_future_overlay(
                    future_frames_normal,
                    future_boxes_normal_gt,
                    future_states_normal_gt,
                    pred_future_full,
                    model_label=model.label,
                    context_mode_label="context",
                    pred_color=(228, 74, 62),
                    context_last_boxes=red_context_last_boxes,
                ),
                parsed.fps,
            )
            write_mp4(
                output_dir / trimmed_rel_video,
                draw_single_future_overlay(
                    future_frames_trimmed,
                    future_boxes_trimmed_gt,
                    future_states_trimmed_gt,
                    pred_future,
                    model_label=model.label,
                    context_mode_label="context[:-1]",
                    pred_color=(48, 118, 255),
                    context_last_boxes=blue_context_last_boxes,
                ),
                parsed.fps,
            )
            case_models.append(
                {
                    "label": model.label,
                    "checkpoint": str(model.checkpoint),
                    "normal_video": normal_rel_video,
                    "trimmed_video": trimmed_rel_video,
                    "normal_metrics": predictor_metrics_normal,
                    "trimmed_metrics": predictor_metrics_trimmed,
                }
            )
            npz_payload[f"{model.label}_pred_future_latent"] = pred_future_latent
            npz_payload[f"{model.label}_pred_future"] = pred_future
            npz_payload[f"{model.label}_pred_future_latent_fullctx"] = pred_future_latent_full
            npz_payload[f"{model.label}_pred_future_fullctx"] = pred_future_full

            if model.label == "tailquery":
                for ratio in TAILQUERY_CONTEXT_RATIOS:
                    ratio_context_steps = resolve_ratio_context_steps(original_context_steps, ratio)
                    ratio_batch, ratio_future_steps = build_batch_with_context_steps(
                        episode,
                        prompt,
                        ratio_context_steps,
                    )
                    ratio_outputs = run_predictor_for_batch(
                        ratio_batch,
                        loaded_model=model,
                        latent_extractor=latent_extractor,
                        prompt_context_encoder=prompt_context_encoder,
                        device=device,
                    )
                    ratio_pred_future_latent = (
                        ratio_outputs["future_state_predictions"][0].detach().cpu().numpy().astype(np.float32)
                    )
                    ratio_pred_future = resample_predicted_states_to_frame_steps(
                        ratio_pred_future_latent,
                        target_steps=ratio_future_steps,
                    )
                    ratio_future_frames = full_frames[ratio_context_steps:]
                    ratio_future_states_gt = full_states[ratio_context_steps:]
                    ratio_future_boxes_gt = full_boxes[ratio_context_steps:]
                    ratio_context_last_boxes = [
                        last_valid_box(full_boxes, obj_idx, ratio_context_steps)
                        for obj_idx in range(int(full_boxes.shape[1]))
                    ]
                    ratio_metrics = compute_state_metrics(ratio_pred_future, ratio_future_states_gt)
                    ratio_metrics["future_start_head_center_error"] = float(
                        np.linalg.norm(
                            ratio_pred_future[0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
                            - ratio_future_states_gt[0, :, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1],
                            axis=-1,
                        ).mean()
                    )
                    ratio_label = f"{int(round(ratio * 100))}%"
                    ratio_slug = f"r{int(round(ratio * 100)):03d}"
                    ratio_context_video_rel = f"assets/{case_id}_{model.label}_{ratio_slug}_context.mp4"
                    ratio_video_rel = f"assets/{case_id}_{model.label}_{ratio_slug}.mp4"
                    ratio_color = (228, 74, 62) if abs(ratio - 1.0) < 1e-6 else (48, 118, 255)
                    write_mp4(
                        output_dir / ratio_context_video_rel,
                        make_labeled_video(
                            full_frames[:ratio_context_steps],
                            [f"context ratio {ratio_label} | length={ratio_context_steps}"],
                        ),
                        parsed.fps,
                    )
                    write_mp4(
                        output_dir / ratio_video_rel,
                        draw_single_future_overlay(
                            ratio_future_frames,
                            ratio_future_boxes_gt,
                            ratio_future_states_gt,
                            ratio_pred_future,
                            model_label=model.label,
                            context_mode_label=f"context ratio {ratio_label}",
                            pred_color=ratio_color,
                            context_last_boxes=ratio_context_last_boxes,
                        ),
                        parsed.fps,
                    )
                    tailquery_ratio_variants.append(
                        {
                            "ratio": float(ratio),
                            "ratio_label": ratio_label,
                            "context_steps": int(ratio_context_steps),
                            "future_steps": int(ratio_future_steps),
                            "context_video": ratio_context_video_rel,
                            "video": ratio_video_rel,
                            "metrics": ratio_metrics,
                        }
                    )
                    npz_payload[f"{model.label}_{ratio_slug}_pred_future_latent"] = ratio_pred_future_latent
                    npz_payload[f"{model.label}_{ratio_slug}_pred_future"] = ratio_pred_future
                    npz_payload[f"{model.label}_{ratio_slug}_context_steps"] = np.asarray(
                        [ratio_context_steps],
                        dtype=np.int64,
                    )
                    npz_payload[f"{model.label}_{ratio_slug}_future_steps"] = np.asarray(
                        [ratio_future_steps],
                        dtype=np.int64,
                    )

        np.savez_compressed(output_dir / f"assets/{case_id}_context_minus1_outputs.npz", **npz_payload)
        cases.append(
            {
                "case_id": case_id,
                "split": path.parent.name,
                "template_key": meta.get("template_key", "unknown"),
                "prompt": prompt,
                "normal_future_video": normal_future_video_rel,
                "trimmed_future_video": trimmed_future_video_rel,
                "normal_context_video": normal_context_rel,
                "trimmed_context_video": trimmed_context_rel,
                "original_context_steps": int(original_context_steps),
                "trimmed_context_steps": int(trimmed_context_steps),
                "future_start_idx": int(future_start_idx),
                "tailquery_ratio_variants": tailquery_ratio_variants,
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
                "disable_future_camera": bool(model.disable_future_camera),
                "center_error_mean": float(np.mean(metrics["center_error"])),
                "log_scale_error_mean": float(np.mean(metrics["log_scale_error"])),
                "visibility_error_mean": float(np.mean(metrics["visibility_error"])),
                "future_start_head_center_error_mean": float(np.mean(metrics["future_start_head_center_error"])),
            }
        )

    report = {
        "episode_root": str(episode_root),
        "output_dir": str(output_dir),
        "case_count": len(cases),
        "model_count": len(loaded_models),
        "summary": summary,
        "cases": cases,
        "mode": "wan_state_v2_context_minus1_dashboard",
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
