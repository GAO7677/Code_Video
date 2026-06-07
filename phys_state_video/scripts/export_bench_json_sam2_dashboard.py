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

from phys_state_video.mask_tracking import (
    GroundingDINOTextDetector,
    SAM2VideoMaskTracker,
    build_caption_prompt_boxes,
    build_mask_track_outputs,
    build_proxy_prompt_box,
)
from phys_state_video.proxy_state import extract_primary_track, read_video_frames
from phys_state_video.schemas import StateIndex


@dataclass(slots=True)
class CaseSpec:
    json_name: str
    source_index: int
    category: str
    source_video: str
    caption: str
    context_video: str | None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run proxy-box -> SAM2 tracking on benchmark json clips and export a local dashboard."
    )
    parser.add_argument("--bench-json-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json-names", nargs="+", default=["A.json", "B.json", "D.json"])
    parser.add_argument("--sample-per-json", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--context-steps", type=int, default=None)
    parser.add_argument("--context-ratio", type=float, default=0.25)
    parser.add_argument("--future-steps", type=int, default=None)
    parser.add_argument("--prompt-frame", choices=["last_context", "first"], default="last_context")
    parser.add_argument("--prompt-mode", choices=["proxy_box", "caption_gdino"], default="caption_gdino")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sam2-model-id", default="")
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
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18879)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            for sub in sorted(child.rglob("*"), reverse=True):
                if sub.is_file() or sub.is_symlink():
                    sub.unlink()
                elif sub.is_dir():
                    sub.rmdir()
            child.rmdir()


def choose_cases(bench_json_root: Path, json_names: list[str], sample_per_json: int, seed: int) -> list[CaseSpec]:
    rng = random.Random(seed)
    chosen: list[CaseSpec] = []
    for json_name in json_names:
        payload = json.loads((bench_json_root / json_name).read_text(encoding="utf-8"))
        indices = list(range(len(payload)))
        if sample_per_json >= len(indices):
            selected_indices = indices
        else:
            rng.shuffle(indices)
            selected_indices = indices[: min(sample_per_json, len(indices))]
        for source_index in selected_indices:
            item = payload[source_index]
            source_index_value = item.get("source_index_override", item.get("source_index", source_index))
            chosen.append(
                CaseSpec(
                    json_name=json_name,
                    source_index=int(source_index_value),
                    category=str(item.get("category") or "unknown"),
                    source_video=str(item["source_video"]),
                    caption=str(item.get("caption") or ""),
                    context_video=str(item["context_video"]) if item.get("context_video") else None,
                )
            )
    return chosen


def resolve_context_steps(item: CaseSpec, full_frames: np.ndarray, args) -> int:
    total_steps = int(full_frames.shape[0])
    if args.context_steps is not None:
        return min(max(int(args.context_steps), 1), total_steps - 1)
    if item.context_video:
        path = Path(item.context_video)
        if path.exists():
            context_frames = read_video_frames(
                path,
                resize_height=args.height,
                resize_width=args.width,
            )[:: args.frame_stride]
            return min(max(int(context_frames.shape[0]), 1), total_steps - 1)
    ratio_steps = int(round(total_steps * float(args.context_ratio)))
    return min(max(ratio_steps, 1), total_steps - 1)


def resolve_prompt_frame_idx(context_steps: int, prompt_frame: str) -> int:
    if prompt_frame == "first":
        return 0
    return max(int(context_steps) - 1, 0)


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


def pixel_box_to_pixels(box_xyxy: np.ndarray, height: int, width: int) -> tuple[int, int, int, int]:
    x0 = int(np.clip(box_xyxy[0], 0, width - 1))
    y0 = int(np.clip(box_xyxy[1], 0, height - 1))
    x1 = int(np.clip(box_xyxy[2], 0, width - 1))
    y1 = int(np.clip(box_xyxy[3], 0, height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def render_raw_video(frames_tchw: np.ndarray) -> np.ndarray:
    return frames_tchw.astype(np.float32)


def draw_box_overlay_video(
    frames_tchw: np.ndarray,
    *,
    boxes_norm: np.ndarray | None = None,
    prompt_boxes_xyxy: np.ndarray | None = None,
    prompt_frame_idx: int | None = None,
    masks_tnhw: np.ndarray | None = None,
    include_centers: bool = False,
) -> np.ndarray:
    rendered: list[np.ndarray] = []
    num_frames = int(frames_tchw.shape[0])
    num_objects = 0 if boxes_norm is None else int(boxes_norm.shape[1])
    for frame_idx in range(num_frames):
        rgb = to_uint8_rgb(frames_tchw[frame_idx])
        canvas = rgb.copy()
        height, width = canvas.shape[:2]
        if masks_tnhw is not None:
            frame_masks = masks_tnhw[frame_idx]
            if frame_masks.ndim == 2:
                frame_masks = frame_masks[None]
            union = np.any(frame_masks > 0, axis=0)
            if np.any(union):
                tint = canvas.copy()
                tint[union] = (0.35 * tint[union] + 0.65 * np.asarray([70, 210, 120], dtype=np.float32)).astype(np.uint8)
                canvas = tint
        if boxes_norm is not None:
            for obj_idx in range(num_objects):
                box = boxes_norm[frame_idx, obj_idx]
                if float(box[2] - box[0]) <= 1e-4 or float(box[3] - box[1]) <= 1e-4:
                    continue
                x0, y0, x1, y1 = normalized_box_to_pixels(box, height, width)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 210, 120), 2, cv2.LINE_AA)
                if include_centers:
                    cx = int(np.clip((box[0] + box[2]) * 0.5 * width, 0, width - 1))
                    cy = int(np.clip((box[1] + box[3]) * 0.5 * height, 0, height - 1))
                    cv2.circle(canvas, (cx, cy), 3, (70, 210, 120), -1, cv2.LINE_AA)
        if prompt_boxes_xyxy is not None and prompt_frame_idx is not None and frame_idx == int(prompt_frame_idx):
            prompt_boxes = np.asarray(prompt_boxes_xyxy, dtype=np.float32)
            if prompt_boxes.ndim == 1:
                prompt_boxes = prompt_boxes[None]
            for prompt_box_xyxy in prompt_boxes:
                x0, y0, x1, y1 = pixel_box_to_pixels(prompt_box_xyxy, height, width)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (250, 210, 40), 2, cv2.LINE_AA)
        rendered.append(np.transpose(canvas, (2, 0, 1)).astype(np.float32) / 255.0)
    return np.stack(rendered, axis=0)


def compute_stats(boxes: np.ndarray, prompt_frame_idx: int) -> dict[str, float]:
    visible = (boxes[..., 2] - boxes[..., 0] > 1e-4) & (boxes[..., 3] - boxes[..., 1] > 1e-4)
    visible_ratio = float(visible.mean())
    centers = np.stack(((boxes[..., 0] + boxes[..., 2]) * 0.5, (boxes[..., 1] + boxes[..., 3]) * 0.5), axis=-1)
    center_delta = np.linalg.norm(centers[1:] - centers[:-1], axis=-1)
    valid_pairs = visible[1:] & visible[:-1]
    mean_step = float(center_delta[valid_pairs].mean()) if np.any(valid_pairs) else 0.0
    prompt_visible = bool(visible[int(prompt_frame_idx), 0]) if boxes.ndim >= 3 and boxes.shape[1] > 0 else False
    return {
        "visible_ratio": visible_ratio,
        "mean_center_step": mean_step,
        "prompt_visible": float(prompt_visible),
    }


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


def render_html(report: dict) -> str:
    summary_rows = []
    for row in report["summary"]:
        summary_rows.append(
            f"""
            <tr>
              <td>{html.escape(row['json_name'])}</td>
              <td>{row['case_count']}</td>
              <td>{row['sam2_visible_ratio_mean']:.4f}</td>
              <td>{row['proxy_visible_ratio_mean']:.4f}</td>
              <td>{row['sam2_center_step_mean']:.4f}</td>
              <td>{row['proxy_center_step_mean']:.4f}</td>
            </tr>
            """
        )

    case_cards = []
    for case in report["cases"]:
        case_cards.append(
            f"""
            <section class="case-card" id="{html.escape(case['case_id'])}">
              <div class="case-head">
                <div>
                  <div class="eyebrow">{html.escape(case['json_name'])} | {html.escape(case['category'])}</div>
                  <h2>{html.escape(case['case_id'])}</h2>
                  <p class="prompt">{html.escape(case['caption'])}</p>
                  <p class="meta">
                    source index={case['source_index']} |
                    total={case['total_frames']} |
                    context={case['context_steps']} |
                    future={case['future_steps']} |
                    prompt frame={case['prompt_frame_idx']}
                  </p>
                  <p class="meta">
                    prompt mode={html.escape(case['prompt_mode'])} |
                    prompt boxes={case['prompt_box_count']} |
                    prompt labels={html.escape(', '.join(case['prompt_phrases'])) if case['prompt_phrases'] else 'n/a'}
                  </p>
                  <p class="meta">
                    green = SAM2 tracked box, yellow = detector / proxy prompt box on prompt frame only
                  </p>
                </div>
              </div>
              <div class="video-grid">
                <article class="video-card">
                  <div class="video-eyebrow">Reference</div>
                  <h3>Input Context</h3>
                  <video controls preload="metadata" playsinline src="{html.escape(case['context_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">Baseline</div>
                  <h3>Proxy Boxes</h3>
                  <video controls preload="metadata" playsinline src="{html.escape(case['proxy_overlay_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">SAM2</div>
                  <h3>Full Video Overlay</h3>
                  <video controls preload="metadata" playsinline src="{html.escape(case['sam2_full_video'])}"></video>
                </article>
                <article class="video-card">
                  <div class="video-eyebrow">SAM2</div>
                  <h3>Future Overlay</h3>
                  <video controls preload="metadata" playsinline src="{html.escape(case['sam2_future_video'])}"></video>
                </article>
              </div>
              <div class="metric-grid">
                <div class="metric-box">
                  <div class="metric-title">SAM2 Stats</div>
                  <div>Visible ratio: {case['sam2_stats']['visible_ratio']:.4f}</div>
                  <div>Mean center step: {case['sam2_stats']['mean_center_step']:.4f}</div>
                  <div>Prompt visible: {int(case['sam2_stats']['prompt_visible'])}</div>
                </div>
                <div class="metric-box">
                  <div class="metric-title">Proxy Stats</div>
                  <div>Visible ratio: {case['proxy_stats']['visible_ratio']:.4f}</div>
                  <div>Mean center step: {case['proxy_stats']['mean_center_step']:.4f}</div>
                  <div>Prompt visible: {int(case['proxy_stats']['prompt_visible'])}</div>
                </div>
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SAM2 Bench Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.97);
      --line: #ddd2c2;
      --ink: #1c1814;
      --muted: #6f675d;
      --accent: #0e5d53;
      --accent2: #bc6a2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(188, 106, 47, 0.12), transparent 22%),
        radial-gradient(circle at top right, rgba(14, 93, 83, 0.12), transparent 26%),
        linear-gradient(180deg, #f7f3ea 0%, #eee3d6 100%);
    }}
    .page {{
      max-width: 1760px;
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
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .video-card {{
      background: #fcf8f2;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 12px;
    }}
    .metric-box {{
      background: #f1e8db;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 12px;
      color: #714724;
      line-height: 1.8;
      font-size: 14px;
    }}
    .metric-title {{
      font-weight: 700;
      margin-bottom: 4px;
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
    code {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
    }}
    @media (max-width: 1200px) {{
      .video-grid, .metric-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>SAM2 Benchmark Box Dashboard</h1>
      <p>这页直接对 <code>A/B/D.json</code> 中抽取的 source case 运行 <code>caption / proxy prompt -&gt; SAM2 双向时序传播</code>。页面里的 <code>Proxy Boxes</code> 是旧运动粗框基线，<code>SAM2</code> 两列则是新 pipeline 提取出的时序框结果。</p>
      <p>颜色约定：绿色是 SAM2 tracked box，黄色只在 prompt frame 上画出初始化 prompt box。视频上不额外叠文字，元信息统一放在卡片文字区。</p>
      <table>
        <thead>
          <tr>
            <th>JSON</th>
            <th>Cases</th>
            <th>SAM2 Visible</th>
            <th>Proxy Visible</th>
            <th>SAM2 Step</th>
            <th>Proxy Step</th>
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
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.clean:
        clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    tracker = SAM2VideoMaskTracker(
        device=device,
        model_id=args.sam2_model_id or None,
        model_cfg=args.sam2_config,
        checkpoint_path=args.sam2_ckpt,
    )
    text_detector = None
    if args.prompt_mode == "caption_gdino":
        text_detector = GroundingDINOTextDetector(
            repo_root=args.gdino_repo_root,
            config_path=args.gdino_config,
            checkpoint_path=args.gdino_ckpt,
            device=device,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
            max_boxes=args.gdino_max_boxes,
        )

    cases = choose_cases(
        Path(args.bench_json_root),
        json_names=args.json_names,
        sample_per_json=args.sample_per_json,
        seed=args.seed,
    )

    summary_by_json: dict[str, dict[str, list[float] | int]] = {}
    report_cases: list[dict[str, object]] = []

    for case_idx, spec in enumerate(cases):
        source_path = Path(spec.source_video)
        if not source_path.exists():
            print(f"skip missing source video: {source_path}")
            continue
        frames = read_video_frames(
            source_path,
            resize_height=args.height,
            resize_width=args.width,
        )[:: args.frame_stride]
        if args.future_steps is not None:
            total_keep = min(
                int(frames.shape[0]),
                resolve_context_steps(spec, frames, args) + int(args.future_steps),
            )
            frames = frames[:total_keep]
        if int(frames.shape[0]) < 2:
            print(f"skip too-short video: {source_path}")
            continue

        context_steps = resolve_context_steps(spec, frames, args)
        if context_steps >= int(frames.shape[0]):
            context_steps = int(frames.shape[0]) - 1
        future_steps = int(frames.shape[0] - context_steps)
        prompt_frame_idx = resolve_prompt_frame_idx(context_steps, args.prompt_frame)

        prompt_phrases: list[str] = []
        prompt_scores = np.zeros((0,), dtype=np.float32)
        if args.prompt_mode == "caption_gdino":
            proxy_guidance_box = build_proxy_prompt_box(frames, prompt_frame_idx=prompt_frame_idx)
            detection = build_caption_prompt_boxes(
                frames,
                prompt_frame_idx=prompt_frame_idx,
                caption=spec.caption,
                detector=text_detector,
                guidance_box_xyxy=proxy_guidance_box,
            )
            if detection.boxes_xyxy.shape[0] == 0:
                prompt_boxes_xyxy = build_proxy_prompt_box(frames, prompt_frame_idx=prompt_frame_idx)[None]
                resolved_prompt_mode = "proxy_box_fallback"
            else:
                prompt_boxes_xyxy = detection.boxes_xyxy.astype(np.float32)
                prompt_phrases = list(detection.phrases)
                prompt_scores = detection.scores.astype(np.float32)
                resolved_prompt_mode = detection.prompt_mode
        else:
            prompt_boxes_xyxy = build_proxy_prompt_box(frames, prompt_frame_idx=prompt_frame_idx)[None]
            resolved_prompt_mode = "proxy_box"
        sam2_outputs = build_mask_track_outputs(
            frames,
            prompt_frame_idx=prompt_frame_idx,
            prompt_boxes_xyxy=prompt_boxes_xyxy,
            prompt_mode=resolved_prompt_mode,
            tracker=tracker,
        )
        proxy_track = extract_primary_track(frames)

        case_id = f"{Path(spec.json_name).stem.lower()}_{spec.source_index:03d}_{case_idx:02d}_{source_path.stem}"
        context_video_rel = f"assets/{case_id}_context.mp4"
        proxy_overlay_rel = f"assets/{case_id}_proxy_overlay.mp4"
        sam2_full_rel = f"assets/{case_id}_sam2_full.mp4"
        sam2_future_rel = f"assets/{case_id}_sam2_future.mp4"

        write_mp4(output_dir / context_video_rel, render_raw_video(frames[:context_steps]), args.fps)
        write_mp4(
            output_dir / proxy_overlay_rel,
            draw_box_overlay_video(
                frames,
                boxes_norm=proxy_track.boxes,
                prompt_boxes_xyxy=prompt_boxes_xyxy,
                prompt_frame_idx=prompt_frame_idx,
                include_centers=False,
            ),
            args.fps,
        )
        write_mp4(
            output_dir / sam2_full_rel,
            draw_box_overlay_video(
                frames,
                boxes_norm=sam2_outputs.boxes,
                prompt_boxes_xyxy=prompt_boxes_xyxy,
                prompt_frame_idx=prompt_frame_idx,
                masks_tnhw=sam2_outputs.masks,
                include_centers=False,
            ),
            args.fps,
        )
        write_mp4(
            output_dir / sam2_future_rel,
            draw_box_overlay_video(
                frames[context_steps:],
                boxes_norm=sam2_outputs.boxes[context_steps:],
                masks_tnhw=sam2_outputs.masks[context_steps:],
                include_centers=False,
            ),
            args.fps,
        )

        sam2_stats = compute_stats(sam2_outputs.boxes, prompt_frame_idx)
        proxy_stats = compute_stats(proxy_track.boxes, prompt_frame_idx)
        bucket = summary_by_json.setdefault(
            spec.json_name,
            {
                "case_count": 0,
                "sam2_visible_ratio": [],
                "proxy_visible_ratio": [],
                "sam2_center_step": [],
                "proxy_center_step": [],
            },
        )
        bucket["case_count"] = int(bucket["case_count"]) + 1
        bucket["sam2_visible_ratio"].append(sam2_stats["visible_ratio"])
        bucket["proxy_visible_ratio"].append(proxy_stats["visible_ratio"])
        bucket["sam2_center_step"].append(sam2_stats["mean_center_step"])
        bucket["proxy_center_step"].append(proxy_stats["mean_center_step"])

        report_cases.append(
            {
                "case_id": case_id,
                "json_name": spec.json_name,
                "source_index": int(spec.source_index),
                "source_video": str(source_path),
                "category": spec.category,
                "caption": spec.caption,
                "total_frames": int(frames.shape[0]),
                "context_steps": int(context_steps),
                "future_steps": int(future_steps),
                "prompt_frame_idx": int(prompt_frame_idx),
                "prompt_boxes_xyxy": prompt_boxes_xyxy.tolist(),
                "prompt_box_count": int(prompt_boxes_xyxy.shape[0]),
                "prompt_phrases": prompt_phrases,
                "prompt_scores": prompt_scores.tolist(),
                "prompt_mode": resolved_prompt_mode,
                "context_video": context_video_rel,
                "proxy_overlay_video": proxy_overlay_rel,
                "sam2_full_video": sam2_full_rel,
                "sam2_future_video": sam2_future_rel,
                "sam2_stats": sam2_stats,
                "proxy_stats": proxy_stats,
            }
        )
        print(
            f"exported {case_id} "
            f"(json={spec.json_name}, context={context_steps}, future={future_steps}, prompt={prompt_frame_idx})"
        )

    summary = []
    for json_name in args.json_names:
        bucket = summary_by_json.get(json_name)
        if not bucket:
            continue
        summary.append(
            {
                "json_name": json_name,
                "case_count": int(bucket["case_count"]),
                "sam2_visible_ratio_mean": float(np.mean(bucket["sam2_visible_ratio"])),
                "proxy_visible_ratio_mean": float(np.mean(bucket["proxy_visible_ratio"])),
                "sam2_center_step_mean": float(np.mean(bucket["sam2_center_step"])),
                "proxy_center_step_mean": float(np.mean(bucket["proxy_center_step"])),
            }
        )

    report = {
        "bench_json_root": args.bench_json_root,
        "output_dir": str(output_dir),
        "case_count": len(report_cases),
        "sample_per_json": args.sample_per_json,
        "json_names": args.json_names,
        "summary": summary,
        "cases": report_cases,
        "mode": "bench_json_sam2_dashboard",
        "port": args.port,
        "sam2_config": args.sam2_config,
        "sam2_ckpt": args.sam2_ckpt,
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
