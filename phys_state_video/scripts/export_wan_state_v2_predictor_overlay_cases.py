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
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.dataset import NpzPredictorDataset, collate_predictor_episodes
from phys_state_video.experiment import compute_state_metrics
from phys_state_video.schemas import StateIndex
from phys_state_video.utils import require_torch
from phys_state_video.predictor_wan_state_v2 import resample_temporal_states
from phys_state_video.wan_predictor_runtime import (
    build_predictor_latent_extractor,
    build_predictor_prompt_context_encoder,
    load_wan_state_predictor,
    resolve_predictor_wan_task,
)
from phys_state_video.wan_state_v2_helpers import (
    compute_future_latent_steps,
    resample_camera_to_latent_steps,
)

torch = require_torch()
F = torch.nn.functional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export overlay videos for wan_state_v2 predictor outputs on real future frames."
    )
    parser.add_argument("--episode-root", required=True, help="Episode root containing val/test split folders.")
    parser.add_argument("--predictor", required=True, help="wan_state_v2 predictor checkpoint path.")
    parser.add_argument("--wan-ckpt-dir", required=True, help="Wan checkpoint directory used by the predictor VAE/T5.")
    parser.add_argument("--output-dir", required=True, help="Directory for html/assets/json.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18846)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--wan-task", default="ti2v-5B")
    parser.add_argument("--predictor-wan-task", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return (image * 255.0).round().astype(np.uint8)


def write_mp4(path: Path, frames_tchw: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, _, height, width = frames_tchw.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path}")
    for idx in range(t):
        rgb = to_uint8_rgb(frames_tchw[idx])
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def draw_text(rgb: np.ndarray, text: str, line2: str | None = None) -> np.ndarray:
    canvas = rgb.copy()
    lines = [text] if line2 is None else [text, line2]
    for row, line in enumerate(lines):
        y = 22 + row * 22
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
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


def last_valid_box(context_boxes: np.ndarray, obj_idx: int) -> np.ndarray | None:
    for frame_idx in range(context_boxes.shape[0] - 1, -1, -1):
        box = context_boxes[frame_idx, obj_idx]
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


def draw_state_overlay(
    future_frames: np.ndarray,
    context_boxes: np.ndarray,
    future_boxes: np.ndarray,
    predicted_states: np.ndarray,
    target_states: np.ndarray,
) -> np.ndarray:
    _, _, height, width = future_frames.shape
    ref_boxes = [last_valid_box(context_boxes, obj_idx) for obj_idx in range(predicted_states.shape[1])]
    frames = []
    for idx in range(future_frames.shape[0]):
        rgb = to_uint8_rgb(future_frames[idx])
        canvas = rgb.copy()
        for obj_idx in range(predicted_states.shape[1]):
            pred = predicted_states[idx, obj_idx]
            tgt = target_states[idx, obj_idx]
            gt_box = future_boxes[idx, obj_idx]
            if (
                float(tgt[StateIndex.EXISTENCE]) > 0.3
                and float(tgt[StateIndex.VISIBILITY]) > 0.2
                and float(gt_box[2] - gt_box[0]) > 1e-4
                and float(gt_box[3] - gt_box[1]) > 1e-4
            ):
                gx0, gy0, gx1, gy1 = normalized_box_to_pixels(gt_box, height, width)
                cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (60, 210, 90), 2, cv2.LINE_AA)
                gcx = int(np.clip(tgt[StateIndex.CENTER_X] * width, 0, width - 1))
                gcy = int(np.clip(tgt[StateIndex.CENTER_Y] * height, 0, height - 1))
                cv2.circle(canvas, (gcx, gcy), 3, (60, 210, 90), -1, cv2.LINE_AA)
            if float(pred[StateIndex.EXISTENCE]) > 0.3 and float(pred[StateIndex.VISIBILITY]) > 0.2:
                pred_box = predict_box_from_state(pred, ref_boxes[obj_idx])
                px0, py0, px1, py1 = normalized_box_to_pixels(pred_box, height, width)
                cv2.rectangle(canvas, (px0, py0), (px1, py1), (228, 74, 62), 2, cv2.LINE_AA)
                px = int(np.clip(pred[StateIndex.CENTER_X] * width, 0, width - 1))
                py = int(np.clip(pred[StateIndex.CENTER_Y] * height, 0, height - 1))
                cv2.circle(canvas, (px, py), 3, (228, 74, 62), -1, cv2.LINE_AA)
                vx = int(np.clip(px + pred[StateIndex.VEL_X] * width * 0.25, 0, width - 1))
                vy = int(np.clip(py + pred[StateIndex.VEL_Y] * height * 0.25, 0, height - 1))
                cv2.arrowedLine(canvas, (px, py), (vx, vy), (228, 74, 62), 2, cv2.LINE_AA, tipLength=0.25)
        frames.append(
            draw_text(
                canvas,
                "green=GT bbox/center  red=pred bbox/center",
                f"frame={idx:02d}",
            )
        )
    return np.stack([np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0 for frame in frames], axis=0)


def pca_project_condition_maps(condition_maps: np.ndarray) -> np.ndarray:
    # [T, C, H, W] -> [T, H, W, 3]
    tokens = np.transpose(condition_maps, (0, 2, 3, 1)).reshape(-1, condition_maps.shape[1]).astype(np.float32)
    tokens = tokens - tokens.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(tokens, full_matrices=False)
    basis = vh[:3].T
    projected = tokens @ basis
    proj_min = projected.min(axis=0, keepdims=True)
    proj_max = projected.max(axis=0, keepdims=True)
    projected = (projected - proj_min) / np.maximum(proj_max - proj_min, 1e-6)
    projected = projected.reshape(condition_maps.shape[0], condition_maps.shape[2], condition_maps.shape[3], 3)
    return np.clip(projected, 0.0, 1.0)


def temporal_repeat_indices(source_steps: int, target_steps: int) -> np.ndarray:
    if source_steps <= 0:
        raise ValueError(f"source_steps must be positive, got {source_steps}")
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if source_steps == target_steps:
        return np.arange(target_steps, dtype=np.int64)
    return np.round(np.linspace(0, source_steps - 1, target_steps)).astype(np.int64)


def build_condition_overlay_video(future_frames: np.ndarray, condition_maps: np.ndarray) -> np.ndarray:
    _, _, frame_h, frame_w = future_frames.shape
    rgb_maps = pca_project_condition_maps(condition_maps)
    time_indices = temporal_repeat_indices(rgb_maps.shape[0], future_frames.shape[0])
    frames = []
    for out_idx, map_idx in enumerate(time_indices):
        rgb = to_uint8_rgb(future_frames[out_idx]).astype(np.float32) / 255.0
        cond_rgb = rgb_maps[map_idx]
        cond_rgb = cv2.resize(cond_rgb, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        blended = np.clip(0.55 * rgb + 0.45 * cond_rgb, 0.0, 1.0)
        annotated = draw_text(
            (blended * 255.0).round().astype(np.uint8),
            "condition_maps PCA overlay",
            f"future_frame={out_idx:02d} latent_step={int(map_idx):02d}",
        )
        frames.append(np.transpose(annotated, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


def resample_predicted_states_to_frame_steps(predicted_states: np.ndarray, target_steps: int) -> np.ndarray:
    if predicted_states.ndim != 3:
        raise ValueError(f"expected predicted states [T, N, D], got {tuple(predicted_states.shape)}")
    tensor = torch.from_numpy(predicted_states[None]).float()
    resized = resample_temporal_states(tensor, target_steps=target_steps)[0]
    return resized.cpu().numpy().astype(np.float32)


def render_html(report: dict) -> str:
    case_cards = []
    for case in report["cases"]:
        case_cards.append(
            f"""
            <article class="case-card">
              <div class="case-head">
                <div>
                  <div class="eyebrow">{html.escape(case['split'])} · {html.escape(case['template_key'])}</div>
                  <h2>{html.escape(case['case_id'])}</h2>
                  <p class="prompt">{html.escape(case['prompt'])}</p>
                  <p class="meta">raw sample: {html.escape(case['sample_id'])} | family: {html.escape(case['family'])}</p>
                </div>
                <div class="score-box">
                  <div>center error {case['predictor_metrics']['center_error']:.3f}</div>
                  <div>log-scale error {case['predictor_metrics']['log_scale_error']:.3f}</div>
                  <div>visibility error {case['predictor_metrics']['visibility_error']:.3f}</div>
                </div>
              </div>
              <div class="media-grid">
                <section class="media-card">
                  <div class="media-title">Context</div>
                  <video controls preload="metadata" src="{html.escape(case['context_video'])}"></video>
                </section>
                <section class="media-card">
                  <div class="media-title">GT Future</div>
                  <video controls preload="metadata" src="{html.escape(case['gt_video'])}"></video>
                </section>
                <section class="media-card">
                  <div class="media-title">State Overlay</div>
                  <video controls preload="metadata" src="{html.escape(case['overlay_video'])}"></video>
                </section>
                <section class="media-card">
                  <div class="media-title">Condition Overlay</div>
                  <video controls preload="metadata" src="{html.escape(case['condition_overlay_video'])}"></video>
                </section>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>wan_state_v2 predictor overlay</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: rgba(255, 252, 246, 0.95);
      --line: #ddcfbf;
      --ink: #1f1f1b;
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
        radial-gradient(circle at top left, rgba(184, 100, 42, 0.14), transparent 24%),
        radial-gradient(circle at top right, rgba(15, 90, 82, 0.14), transparent 22%),
        linear-gradient(180deg, #f7f3ea 0%, #efe5d8 100%);
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 26px;
    }}
    .hero, .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero p {{
      color: var(--muted);
      line-height: 1.75;
      margin: 10px 0 0;
    }}
    .case-card {{
      padding: 18px;
      margin-bottom: 18px;
    }}
    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 16px;
    }}
    .eyebrow {{
      color: var(--accent2);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .prompt, .meta {{
      color: var(--muted);
      line-height: 1.7;
      margin: 6px 0 0;
    }}
    .score-box {{
      min-width: 240px;
      border-radius: 14px;
      background: #f1e8db;
      padding: 12px 14px;
      color: #714724;
      line-height: 1.8;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .media-card {{
      background: #fcf8f2;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 12px;
    }}
    .media-title {{
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 10px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    @media (max-width: 1100px) {{
      .media-grid {{
        grid-template-columns: 1fr;
      }}
      .case-head {{
        flex-direction: column;
      }}
      .score-box {{
        min-width: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>wan_state_v2 predictor overlay 页面</h1>
      <p>这页直接把 predictor 的未来状态输出叠加到真实 future video 上。第三列是 GT future 帧上的 state overlay：绿色是 GT box/center，红色是由预测 state 重建出的 proxy box/center，并附带预测速度箭头。第四列是将 predictor 导出的 condition_maps 做 PCA 后，再叠加到 future video 上的 condition overlay，用来直观看空间条件随时间如何变化。</p>
      <p>predictor: {html.escape(report['predictor_checkpoint_name'])} | predictor_wan_task: {html.escape(report['predictor_wan_task'])}</p>
    </section>
    {''.join(case_cards)}
  </div>
</body>
</html>"""


def run_predictor_for_batch(
    batch,
    *,
    predictor,
    predictor_ckpt,
    latent_extractor,
    prompt_context_encoder,
    device: str,
):
    predictor_version = predictor_ckpt.get("predictor_version", "wan_state_v1")
    if predictor_version != "wan_state_v2_latent_time":
        raise ValueError(
            "export_wan_state_v2_predictor_overlay_cases.py currently expects a wan_state_v2_latent_time checkpoint"
        )
    with torch.no_grad():
        context_frames = batch["context_frames"].to(device)
        context_latents = latent_extractor.encode_context_frames_raw(context_frames)
        context_latent_steps = int(context_latents.shape[1])
        future_latent_steps = compute_future_latent_steps(
            context_steps=context_frames.shape[1],
            future_steps=batch["future_states"].shape[1],
            temporal_stride=latent_extractor.temporal_stride,
        )
        camera_latent = resample_camera_to_latent_steps(batch["camera"].to(device), context_latent_steps)
        prompt_context, prompt_mask = prompt_context_encoder.encode_prompts(list(batch["prompts"]))
        outputs = predictor(
            context_latents=context_latents,
            camera=camera_latent,
            prompt_context=prompt_context.to(device),
            prompt_mask=prompt_mask.to(device),
            future_latent_steps=future_latent_steps,
            num_objects=batch["future_states"].shape[2],
        )
    return outputs


def main():
    args = parse_args()
    episode_root = Path(args.episode_root)
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor, predictor_ckpt = load_wan_state_predictor(args.predictor, device)
    latent_extractor = build_predictor_latent_extractor(
        wan_ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        device=device,
        predictor_ckpt=predictor_ckpt,
        default_wan_task=args.wan_task,
        predictor_wan_task=args.predictor_wan_task,
        context="wan_state_v2 predictor overlay export",
    )
    prompt_context_encoder = build_predictor_prompt_context_encoder(
        wan_ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        device=device,
        predictor_ckpt=predictor_ckpt,
        default_wan_task=args.wan_task,
        predictor_wan_task=args.predictor_wan_task,
        context="wan_state_v2 predictor overlay export",
    )

    cases = []
    selected_files = choose_case_files(episode_root, args.splits, args.max_cases)
    center_errors = []
    scale_errors = []
    visibility_errors = []
    for path in selected_files:
        split = path.parent.name
        dataset = NpzPredictorDataset(path)
        batch = collate_predictor_episodes([dataset[0]])
        with np.load(path, allow_pickle=False) as payload:
            context_frames = payload["context_frames"].astype(np.float32)
            future_frames = payload["future_frames"].astype(np.float32)
            context_boxes = payload["context_boxes"].astype(np.float32)
            future_boxes = payload["future_boxes"].astype(np.float32)
            future_states = payload["future_states"].astype(np.float32)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

        outputs = run_predictor_for_batch(
            batch,
            predictor=predictor,
            predictor_ckpt=predictor_ckpt,
            latent_extractor=latent_extractor,
            prompt_context_encoder=prompt_context_encoder,
            device=device,
        )
        predicted_states_latent = outputs["future_state_predictions"][0].detach().cpu().numpy().astype(np.float32)
        predicted_states = resample_predicted_states_to_frame_steps(
            predicted_states_latent,
            target_steps=future_states.shape[0],
        )
        condition_maps = outputs["condition_maps"][0].detach().cpu().numpy().astype(np.float32)
        predictor_metrics = compute_state_metrics(predicted_states, future_states)
        center_errors.append(predictor_metrics["center_error"])
        scale_errors.append(predictor_metrics["log_scale_error"])
        visibility_errors.append(predictor_metrics["visibility_error"])

        case_id = path.stem
        context_video = f"assets/{case_id}_context.mp4"
        gt_video = f"assets/{case_id}_gt_future.mp4"
        overlay_video = f"assets/{case_id}_state_overlay.mp4"
        condition_overlay_video = f"assets/{case_id}_condition_overlay.mp4"

        write_mp4(output_dir / context_video, context_frames, args.fps)
        write_mp4(output_dir / gt_video, future_frames, args.fps)
        write_mp4(
            output_dir / overlay_video,
            draw_state_overlay(future_frames, context_boxes, future_boxes, predicted_states, future_states),
            args.fps,
        )
        write_mp4(
            output_dir / condition_overlay_video,
            build_condition_overlay_video(future_frames, condition_maps),
            args.fps,
        )

        np.savez_compressed(
            output_dir / f"assets/{case_id}_predictor_outputs.npz",
            predicted_future_states_latent=predicted_states_latent,
            predicted_future_states=predicted_states,
            target_future_states=future_states,
            condition_maps=condition_maps,
            future_state_maps=outputs["future_state_maps"][0].detach().cpu().numpy().astype(np.float32),
            memory_tokens=outputs["memory_tokens"][0].detach().cpu().numpy().astype(np.float32),
        )

        cases.append(
            {
                "case_id": case_id,
                "split": split,
                "sample_id": meta.get("sample_id", case_id),
                "template_key": meta.get("template_key", "unknown"),
                "prompt": meta.get("prompt", ""),
                "family": meta.get("family", "unknown"),
                "predictor_metrics": predictor_metrics,
                "context_video": context_video,
                "gt_video": gt_video,
                "overlay_video": overlay_video,
                "condition_overlay_video": condition_overlay_video,
            }
        )

    report = {
        "episode_root": str(episode_root),
        "predictor_checkpoint": str(Path(args.predictor).resolve()),
        "predictor_checkpoint_name": Path(args.predictor).name,
        "predictor_wan_task": resolve_predictor_wan_task(
            predictor_ckpt,
            default_wan_task=args.wan_task,
            predictor_wan_task=args.predictor_wan_task,
        ),
        "port": args.port,
        "case_count": len(cases),
        "aggregate_preview": {
            "center_error_mean": float(np.mean(center_errors)) if center_errors else None,
            "log_scale_error_mean": float(np.mean(scale_errors)) if scale_errors else None,
            "visibility_error_mean": float(np.mean(visibility_errors)) if visibility_errors else None,
        },
        "cases": cases,
        "mode": "wan_state_v2_predictor_overlay",
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    if args.no_serve:
        print(f"exported: {output_dir}")
        return
    pid = start_server(output_dir, args.port)
    print(f"page: {output_dir / 'index.html'}")
    print(f"server: http://127.0.0.1:{args.port}")
    print(f"pid: {pid}")


if __name__ == "__main__":
    main()
