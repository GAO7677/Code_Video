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

from phys_state_video.conditioning import build_condition_bundle
from phys_state_video.config import AdapterConfig, ConditioningConfig, PredictorConfig
from phys_state_video.dataset import NpzEpisodeDataset, collate_episodes
from phys_state_video.experiment import compute_state_metrics
from phys_state_video.pipeline import StateConditionedGenerationPipeline
from phys_state_video.predictor import FutureStatePredictor
from phys_state_video.projection import ConfidenceAwareProjector
from phys_state_video.proxy_state import extract_primary_track
from phys_state_video.utils import detach_to_cpu_numpy, require_torch
from phys_state_video.adapter import TinyVideoBackbone

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="Export representative trained-model video cases.")
    parser.add_argument("--episode-root", required=True, help="Episode root containing val/test split folders.")
    parser.add_argument("--predictor", required=True, help="Predictor checkpoint path.")
    parser.add_argument("--adapter", required=True, help="Adapter checkpoint path.")
    parser.add_argument("--output-dir", required=True, help="Directory for html/assets/json.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18832)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_model_state(module, state_dict, checkpoint_label: str) -> dict[str, list[str]]:
    try:
        module.load_state_dict(state_dict)
        return {"missing": [], "unexpected": []}
    except RuntimeError as exc:
        message = str(exc)
        key_mismatch = "Missing key(s) in state_dict" in message or "Unexpected key(s) in state_dict" in message
        if not key_mismatch:
            raise
        incompatible = module.load_state_dict(state_dict, strict=False)
        return {
            "missing": list(incompatible.missing_keys),
            "unexpected": list(incompatible.unexpected_keys),
        }


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


def draw_text(rgb: np.ndarray, text: str) -> np.ndarray:
    canvas = rgb.copy()
    cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def normalize_map(channel_thw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = float(channel_thw.min())
    hi = float(channel_thw.max())
    if hi - lo < eps:
        return np.zeros_like(channel_thw, dtype=np.float32)
    return (channel_thw - lo) / (hi - lo)


def gray_to_rgb(gray_thw: np.ndarray, label: str) -> np.ndarray:
    frames = []
    for frame in gray_thw:
        rgb = np.repeat((np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)[..., None], 3, axis=2)
        frames.append(draw_text(rgb, label))
    return np.stack(frames, axis=0)


def build_condition_video(cond_maps: np.ndarray) -> np.ndarray:
    heat = cond_maps[:, 0]
    bbox = cond_maps[:, 1]
    depth = normalize_map(cond_maps[:, 2])
    vis = cond_maps[:, 3]
    heat_rgb = gray_to_rgb(heat, "pred heatmap")
    bbox_rgb = gray_to_rgb(bbox, "pred bbox")
    depth_rgb = gray_to_rgb(depth, "pred depth")
    vis_rgb = gray_to_rgb(vis, "pred vis")
    rows = []
    for idx in range(cond_maps.shape[0]):
        top = np.concatenate([heat_rgb[idx], bbox_rgb[idx]], axis=1)
        bottom = np.concatenate([depth_rgb[idx], vis_rgb[idx]], axis=1)
        rows.append(np.concatenate([top, bottom], axis=0))
    return np.stack(rows, axis=0)


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
                  <div>video center {case['video_metrics']['center_error']:.3f}</div>
                  <div>video scale {case['video_metrics']['log_scale_error']:.3f}</div>
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
                  <div class="media-title">Generated Future</div>
                  <video controls preload="metadata" src="{html.escape(case['generated_video'])}"></video>
                </section>
                <section class="media-card">
                  <div class="media-title">Predicted Conditions</div>
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
  <title>训练模型 Video Cases</title>
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
      max-width: 1560px;
      margin: 0 auto;
      padding: 26px;
    }}
    .hero, .metric-card, .case-card, .media-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      backdrop-filter: blur(8px);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    .lead {{
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      max-width: 1100px;
    }}
    .hero-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 16px;
      color: var(--accent2);
      font-size: 14px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .metric-card {{
      padding: 16px;
    }}
    .metric-card h3 {{
      margin: 0 0 10px;
      color: var(--accent);
    }}
    .metric-card p {{
      margin: 6px 0;
    }}
    .case-card {{
      padding: 18px;
      margin-bottom: 22px;
      box-shadow: 0 18px 48px rgba(67, 48, 21, 0.08);
    }}
    .case-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      color: var(--accent2);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    h2 {{
      margin: 6px 0 6px;
      font-size: 24px;
    }}
    .prompt, .meta {{
      margin: 0 0 6px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .score-box {{
      min-width: 260px;
      align-self: flex-start;
      color: var(--accent2);
      font-size: 14px;
      line-height: 1.8;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(245, 236, 223, 0.75);
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .media-card {{
      padding: 12px;
    }}
    .media-title {{
      margin-bottom: 10px;
      color: var(--accent);
      font-weight: 700;
    }}
    img, video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
    }}
    @media (max-width: 1000px) {{
      .case-head {{
        flex-direction: column;
      }}
      .media-grid {{
        grid-template-columns: 1fr;
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
      <h1>当前训练模型 Video Cases</h1>
      <p class="lead">
        这页展示的是当前最佳 predictor 与当前最佳 adapter 的端到端生成结果。每个 case 给出 context、真实 future、模型生成 future，以及模型真正使用的 predicted condition maps，
        方便直接判断状态预测是否合理，以及视频生成是否真的跟随 state condition。
      </p>
      <div class="hero-meta">
        <span>cases: {report['case_count']}</span>
        <span>predictor: {html.escape(report['predictor_checkpoint_name'])}</span>
        <span>adapter: {html.escape(report['adapter_checkpoint_name'])}</span>
        <span>url: http://127.0.0.1:{report['port']}</span>
      </div>
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
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    episode_root = Path(args.episode_root)
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    if args.clean and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                for sub in child.rglob("*"):
                    pass
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    predictor_ckpt = load_checkpoint(args.predictor, map_location=args.device)
    adapter_ckpt = load_checkpoint(args.adapter, map_location=args.device)

    predictor = FutureStatePredictor(PredictorConfig(**predictor_ckpt["config"])).to(args.device)
    predictor_info = load_model_state(predictor, predictor_ckpt["model"], args.predictor)
    predictor.eval()

    adapter = TinyVideoBackbone(AdapterConfig(**adapter_ckpt["config"])).to(args.device)
    adapter_info = load_model_state(adapter, adapter_ckpt["model"], args.adapter)
    adapter.eval()

    cond_cfg = ConditioningConfig(**adapter_ckpt["conditioning"])
    pipeline = StateConditionedGenerationPipeline(
        predictor=predictor,
        projector=ConfidenceAwareProjector(),
        video_model=adapter,
        conditioning_config=cond_cfg,
    )

    eval_metrics: dict[str, dict | None] = {}
    run_root = Path(args.adapter).resolve().parents[1]
    eval_dir = run_root / "eval"
    for split in args.splits:
        metrics_path = eval_dir / f"{split}_metrics.json"
        eval_metrics[split] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None

    case_files = choose_case_files(episode_root, args.splits, args.max_cases)
    cases = []
    predictor_center_errors = []
    video_center_errors = []

    with torch.no_grad():
        for case_path in case_files:
            meta = json.loads(case_path.with_suffix(".json").read_text(encoding="utf-8"))
            dataset = NpzEpisodeDataset(case_path)
            episode = dataset[0]
            batch = collate_episodes([episode])

            outputs = pipeline.generate(
                context_frames=batch["context_frames"].to(args.device),
                context_states=batch["context_states"].to(args.device),
                context_boxes=batch["context_boxes"].to(args.device),
                appearance=batch["appearance"].to(args.device),
                camera=batch["camera"].to(args.device),
                prompts=batch["prompts"],
            )

            predicted_states = detach_to_cpu_numpy(outputs["predicted_states"])[0]
            condition_maps = detach_to_cpu_numpy(outputs["condition_maps"])[0]
            generated_frames = detach_to_cpu_numpy(outputs["generated_frames"])[0]
            target_states = batch["future_states"][0].numpy()
            predictor_metrics = compute_state_metrics(predicted_states, target_states)
            proxy = extract_primary_track(generated_frames)
            video_metrics = compute_state_metrics(proxy.states, target_states)
            predictor_center_errors.append(predictor_metrics["center_error"])
            video_center_errors.append(video_metrics["center_error"])

            family = "unknown"
            raw_meta_path = Path(meta["sample_dir"]) / "meta.json"
            if raw_meta_path.exists():
                raw_meta = json.loads(raw_meta_path.read_text(encoding="utf-8"))
                family = str(raw_meta.get("family", raw_meta.get("family_slug", "unknown")))
            else:
                raw_meta = {}

            case_id = case_path.stem
            case_dir = assets_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            context_video_path = case_dir / "context.mp4"
            gt_video_path = case_dir / "gt_future.mp4"
            generated_video_path = case_dir / "generated_future.mp4"
            condition_video_path = case_dir / "predicted_conditions.mp4"

            write_mp4(context_video_path, batch["context_frames"][0].numpy(), args.fps)
            write_mp4(gt_video_path, batch["future_frames"][0].numpy(), args.fps)
            write_mp4(generated_video_path, generated_frames, args.fps)
            condition_video_rgb = build_condition_video(condition_maps)
            write_mp4(
                condition_video_path,
                np.transpose(condition_video_rgb, (0, 3, 1, 2)).astype(np.float32) / 255.0,
                args.fps,
            )

            cases.append(
                {
                    "case_id": case_id,
                    "split": meta.get("split", case_path.parent.name),
                    "sample_id": meta.get("sample_id", ""),
                    "template_key": meta.get("template_key", "unknown"),
                    "prompt": meta.get("prompt", ""),
                    "family": family,
                    "predictor_metrics": predictor_metrics,
                    "video_metrics": video_metrics,
                    "context_video": f"assets/{case_id}/context.mp4",
                    "gt_video": f"assets/{case_id}/gt_future.mp4",
                    "generated_video": f"assets/{case_id}/generated_future.mp4",
                    "condition_video": f"assets/{case_id}/predicted_conditions.mp4",
                    "raw_meta": raw_meta,
                }
            )

    report = {
        "episode_root": str(episode_root),
        "predictor_checkpoint": str(Path(args.predictor).resolve()),
        "adapter_checkpoint": str(Path(args.adapter).resolve()),
        "predictor_checkpoint_name": Path(args.predictor).name,
        "adapter_checkpoint_name": Path(args.adapter).name,
        "predictor_load_info": predictor_info,
        "adapter_load_info": adapter_info,
        "port": args.port,
        "case_count": len(cases),
        "eval_metrics": eval_metrics,
        "aggregate_preview": {
            "predictor_center_error_mean": float(np.mean(predictor_center_errors)) if predictor_center_errors else None,
            "video_center_error_mean": float(np.mean(video_center_errors)) if video_center_errors else None,
        },
        "cases": cases,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")

    print(f"exported cases to {output_dir}")
    if not args.no_serve:
        pid = start_server(output_dir, args.port)
        print(f"server: http://127.0.0.1:{args.port}")
        print(f"pid: {pid}")


if __name__ == "__main__":
    main()
