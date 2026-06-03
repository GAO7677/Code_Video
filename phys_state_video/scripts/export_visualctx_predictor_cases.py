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
from phys_state_video.predictor_visual_v3 import (
    VisualContextLatentPredictorV3,
    VisualLatentPredictorConfig,
    predictor_visual_v3_loss,
)
from phys_state_video.schemas import STATE_DIM, StateIndex
from phys_state_video.utils import require_torch

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Export predictor-only visual context cases.")
    parser.add_argument("--episode-root", required=True, help="Episode root containing val/test folders.")
    parser.add_argument("--predictor", required=True, help="Predictor checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for html/assets/json.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18836)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


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
        y = 20 + row * 20
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


def safe_radius(log_scale: float, height: int, width: int) -> int:
    area = max(float(np.exp(np.clip(log_scale, -12.0, 6.0))), 1e-6)
    radius = int(np.sqrt(area) * 0.12 * np.sqrt(height * width))
    return int(np.clip(radius, 4, max(8, min(height, width) // 5)))


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


def predict_box_from_state(
    state: np.ndarray,
    ref_box: np.ndarray | None,
) -> np.ndarray:
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
    t, _, height, width = future_frames.shape
    ref_boxes = [last_valid_box(context_boxes, obj_idx) for obj_idx in range(predicted_states.shape[1])]
    frames = []
    for idx in range(t):
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
                cv2.rectangle(canvas, (gx0, gy0), (gx1, gy1), (46, 180, 80), 2, cv2.LINE_AA)
                gcx = int(np.clip(tgt[StateIndex.CENTER_X] * width, 0, width - 1))
                gcy = int(np.clip(tgt[StateIndex.CENTER_Y] * height, 0, height - 1))
                cv2.circle(canvas, (gcx, gcy), 3, (46, 180, 80), -1, cv2.LINE_AA)
            if float(pred[StateIndex.EXISTENCE]) > 0.3 and float(pred[StateIndex.VISIBILITY]) > 0.2:
                pred_box = predict_box_from_state(pred, ref_boxes[obj_idx])
                px0, py0, px1, py1 = normalized_box_to_pixels(pred_box, height, width)
                cv2.rectangle(canvas, (px0, py0), (px1, py1), (210, 60, 50), 2, cv2.LINE_AA)
                px = int(np.clip(pred[StateIndex.CENTER_X] * width, 0, width - 1))
                py = int(np.clip(pred[StateIndex.CENTER_Y] * height, 0, height - 1))
                cv2.circle(canvas, (px, py), 3, (210, 60, 50), -1, cv2.LINE_AA)
                vx = int(np.clip(px + pred[StateIndex.VEL_X] * width * 0.25, 0, width - 1))
                vy = int(np.clip(py + pred[StateIndex.VEL_Y] * height * 0.25, 0, height - 1))
                cv2.arrowedLine(canvas, (px, py), (vx, vy), (210, 60, 50), 2, cv2.LINE_AA, tipLength=0.25)
        frames.append(
            draw_text(
                canvas,
                "green=GT bbox  red=pred bbox",
                f"frame={idx:02d}",
            )
        )
    return np.stack([np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0 for frame in frames], axis=0)


def normalize_map(channel_thw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = float(channel_thw.min())
    hi = float(channel_thw.max())
    if hi - lo < eps:
        return np.zeros_like(channel_thw, dtype=np.float32)
    return (channel_thw - lo) / (hi - lo)


def gaussian_disk(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    sigma = max(radius, 2.0)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))


def build_condition_video(predicted_states: np.ndarray, height: int, width: int) -> np.ndarray:
    frames = []
    for idx in range(predicted_states.shape[0]):
        heat = np.zeros((height, width), dtype=np.float32)
        depth = np.zeros((height, width), dtype=np.float32)
        scale = np.zeros((height, width), dtype=np.float32)
        visibility = np.zeros((height, width), dtype=np.float32)
        for obj_idx in range(predicted_states.shape[1]):
            state = predicted_states[idx, obj_idx]
            if float(state[StateIndex.EXISTENCE]) <= 0.15:
                continue
            cx = float(np.clip(state[StateIndex.CENTER_X], 0.0, 1.0)) * (width - 1)
            cy = float(np.clip(state[StateIndex.CENTER_Y], 0.0, 1.0)) * (height - 1)
            radius = safe_radius(float(state[StateIndex.LOG_SCALE]), height, width)
            disk = gaussian_disk(height, width, cx, cy, radius)
            heat = np.maximum(heat, disk)
            depth = np.maximum(depth, disk * float(state[StateIndex.DEPTH]))
            scale = np.maximum(scale, disk * float(np.exp(np.clip(state[StateIndex.LOG_SCALE], -8.0, 4.0))))
            visibility = np.maximum(visibility, disk * float(state[StateIndex.VISIBILITY]))

        heat_rgb = np.repeat((normalize_map(heat) * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
        depth_rgb = np.repeat((normalize_map(depth) * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
        scale_rgb = np.repeat((normalize_map(scale) * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
        vis_rgb = np.repeat((normalize_map(visibility) * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)

        heat_rgb = draw_text(heat_rgb, "pred center heat")
        depth_rgb = draw_text(depth_rgb, "pred depth")
        scale_rgb = draw_text(scale_rgb, "pred scale")
        vis_rgb = draw_text(vis_rgb, "pred visibility")
        top = np.concatenate([heat_rgb, depth_rgb], axis=1)
        bottom = np.concatenate([scale_rgb, vis_rgb], axis=1)
        frames.append(np.concatenate([top, bottom], axis=0))
    return np.stack([np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0 for frame in frames], axis=0)


def evaluate_split(model, split_dir: Path, device: str, batch_size: int) -> dict | None:
    if not split_dir.exists():
        return None
    dataset = NpzPredictorDataset(split_dir)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_predictor_episodes,
        num_workers=4,
        pin_memory=True,
    )
    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "center_error": 0.0,
        "log_scale_error": 0.0,
        "visibility_error": 0.0,
    }
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["context_frames"].to(device),
                prompt_token_ids=batch["prompt_token_ids"].to(device),
                prompt_token_mask=batch["prompt_token_mask"].to(device),
                future_steps=batch["future_states"].shape[1],
                num_objects=batch["future_states"].shape[2],
            )
            losses = predictor_visual_v3_loss(outputs, batch["future_states"].to(device))
            metrics = compute_state_metrics(
                outputs["states"].detach().cpu().numpy(),
                batch["future_states"].numpy(),
            )
            totals["loss"] += float(losses["loss"].detach().cpu())
            totals["recon"] += float(losses["mse"].detach().cpu())
            totals["center_error"] += metrics["center_error"]
            totals["log_scale_error"] += metrics["log_scale_error"]
            totals["visibility_error"] += metrics["visibility_error"]
    denom = max(len(loader), 1)
    return {
        "metrics": {key: value / denom for key, value in totals.items()}
    }


def render_html(report: dict) -> str:
    metric_cards = []
    for split, metrics in report["eval_metrics"].items():
        if not metrics:
            continue
        metric_cards.append(
            f"""
            <section class="metric-card">
              <h3>{html.escape(split)}</h3>
              <p>loss {metrics['metrics']['loss']:.4f}</p>
              <p>recon {metrics['metrics']['recon']:.4f}</p>
              <p>center_error {metrics['metrics']['center_error']:.4f}</p>
              <p>log_scale_error {metrics['metrics']['log_scale_error']:.4f}</p>
              <p>visibility_error {metrics['metrics']['visibility_error']:.4f}</p>
            </section>
            """
        )

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
                  <div>predictor center {case['predictor_metrics']['center_error']:.3f}</div>
                  <div>predictor scale {case['predictor_metrics']['log_scale_error']:.3f}</div>
                  <div>predictor vis {case['predictor_metrics']['visibility_error']:.3f}</div>
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
                  <div class="media-title">Predictor Overlay</div>
                  <video controls preload="metadata" src="{html.escape(case['generated_video'])}"></video>
                </section>
                <section class="media-card">
                  <div class="media-title">Predicted Explicit Conditions</div>
                  <video controls preload="metadata" src="{html.escape(case['condition_video'])}"></video>
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
  <title>visual_context_predictor_v3 Cases</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --panel: rgba(255, 252, 246, 0.94);
      --line: #dccfbf;
      --ink: #1d1d1b;
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
      max-width: 1580px;
      margin: 0 auto;
      padding: 26px;
    }}
    .hero, .metric-card, .case-card {{
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
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .metric-card {{
      padding: 16px;
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
      min-width: 220px;
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
      <h1>visual_context_predictor_v3 predictor-only case 页面</h1>
      <p>这页展示的是视觉上下文驱动 predictor 的显式未来状态预测结果。由于这版当前还没有接入新的视频生成 adapter，所以第三列不是生成视频，而是把预测状态重建成代理 bbox 后叠加到 GT future 帧上的 overlay；绿色框是 GT future bbox，红色框是由 pred center + pred log_scale + context 最后一帧宽高比重建出来的 pred proxy bbox，第四列则是显式预测条件的 2x2 可视化。</p>
      <p><a href="../index.html">返回方法总入口页</a></p>
    </section>
    <section class="metric-grid">
      {''.join(metric_cards)}
    </section>
    {''.join(case_cards)}
  </div>
</body>
</html>"""


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
    checkpoint = load_checkpoint(args.predictor, map_location="cpu")
    config = checkpoint["config"]
    model = VisualContextLatentPredictorV3(config=VisualLatentPredictorConfig(**config))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    eval_metrics = {}
    for split in args.splits:
        split_metrics = evaluate_split(model, episode_root / split, device, args.batch_size)
        if split_metrics is not None:
            eval_metrics[split] = split_metrics

    cases = []
    selected_files = choose_case_files(episode_root, args.splits, args.max_cases)
    aggregate_predictor_center = []
    aggregate_predictor_scale = []
    for path in selected_files:
        split = path.parent.name
        payload = np.load(path, allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        context_frames = payload["context_frames"].astype(np.float32)
        future_frames = payload["future_frames"].astype(np.float32)
        context_boxes = payload["context_boxes"].astype(np.float32)
        future_boxes = payload["future_boxes"].astype(np.float32)
        future_states = payload["future_states"].astype(np.float32)

        context_tensor = torch.from_numpy(context_frames[None]).to(device)
        with torch.no_grad():
            outputs = model(
                context_tensor,
                prompts=[meta.get("prompt", "")],
                future_steps=future_states.shape[0],
                num_objects=future_states.shape[1],
            )
        predicted_states = outputs["states"][0].detach().cpu().numpy()
        predictor_metrics = compute_state_metrics(predicted_states, future_states)
        aggregate_predictor_center.append(predictor_metrics["center_error"])
        aggregate_predictor_scale.append(predictor_metrics["log_scale_error"])

        case_id = path.stem
        context_video = f"assets/{case_id}_context.mp4"
        gt_video = f"assets/{case_id}_gt.mp4"
        overlay_video = f"assets/{case_id}_predictor_overlay.mp4"
        condition_video = f"assets/{case_id}_predictor_conditions.mp4"

        write_mp4(output_dir / context_video, context_frames, args.fps)
        write_mp4(output_dir / gt_video, future_frames, args.fps)
        write_mp4(
            output_dir / overlay_video,
            draw_state_overlay(future_frames, context_boxes, future_boxes, predicted_states, future_states),
            args.fps,
        )
        write_mp4(
            output_dir / condition_video,
            build_condition_video(predicted_states, future_frames.shape[-2], future_frames.shape[-1]),
            args.fps,
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
                "video_metrics": predictor_metrics,
                "context_video": context_video,
                "gt_video": gt_video,
                "generated_video": overlay_video,
                "condition_video": condition_video,
                "raw_meta": meta,
            }
        )

    report = {
        "episode_root": str(episode_root),
        "predictor_checkpoint": args.predictor,
        "predictor_checkpoint_name": Path(args.predictor).name,
        "adapter_checkpoint": None,
        "adapter_checkpoint_name": None,
        "predictor_load_info": {"missing": [], "unexpected": []},
        "adapter_load_info": None,
        "port": args.port,
        "case_count": len(cases),
        "eval_metrics": eval_metrics,
        "aggregate_preview": {
            "predictor_center_error_mean": float(np.mean(aggregate_predictor_center)) if aggregate_predictor_center else float("nan"),
            "video_center_error_mean": float(np.mean(aggregate_predictor_center)) if aggregate_predictor_center else float("nan"),
            "predictor_log_scale_error_mean": float(np.mean(aggregate_predictor_scale)) if aggregate_predictor_scale else float("nan"),
        },
        "cases": cases,
        "mode": "predictor_only_visualctx_v3",
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
