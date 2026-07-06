#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmf import compute_pmf
from pmf.core import align_pred_to_gt, _ensure_5d_b_t_c_h_w


def read_video(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0), float(fps)


def video_tensor_from_numpy(frames: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return tensor.unsqueeze(0)


def tensor_b_t_c_h_w_to_numpy_t_h_w_c(video: torch.Tensor) -> np.ndarray:
    return (
        video.squeeze(0)
        .permute(0, 2, 3, 1)
        .detach()
        .cpu()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )


def write_video(path: Path, frames_rgb: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, h, w, _ = frames_rgb.shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {path}")

    for i in range(t):
        frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "H.264 output requested, but imageio_ffmpeg is not installed."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def transcode_to_h264_baseline(src_path: Path, dst_path: Path) -> None:
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level:v",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to transcode {src_path} -> {dst_path}\n{proc.stderr}"
        )


def write_video_with_codec(
    path: Path, frames_rgb: np.ndarray, fps: float, video_codec: str
) -> None:
    if video_codec == "mp4v":
        write_video(path, frames_rgb, fps)
        return
    if video_codec != "h264_baseline":
        raise ValueError(f"Unsupported video codec: {video_codec}")

    with tempfile.NamedTemporaryFile(
        prefix="pmf_tmp_", suffix=".mp4", dir=str(path.parent), delete=False
    ) as tmp:
        temp_path = Path(tmp.name)
    try:
        write_video(temp_path, frames_rgb, fps)
        transcode_to_h264_baseline(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def make_side_by_side(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> np.ndarray:
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError("Pred and GT frames must share shape for side-by-side preview.")
    t, h, w, c = pred_rgb.shape
    canvas = np.zeros((t, h, w * 2, c), dtype=np.uint8)
    canvas[:, :, :w, :] = pred_rgb
    canvas[:, :, w:, :] = gt_rgb

    for i in range(t):
        cv2.putText(
            canvas[i],
            "Aligned Prediction",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas[i],
            "Ground Truth",
            (w + 20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def save_frame(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to save image: {path}")


def build_report(
    report_path: Path,
    score: float,
    pred_path: Path,
    gt_path: Path,
    pred_used_path: Path,
    gt_used_path: Path,
    compare_path: Path,
    metadata: dict,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhysInOne PMF Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --card: #fffaf2;
      --ink: #1f1b16;
      --muted: #64584d;
      --accent: #b65c34;
      --line: #e7d9c9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(182, 92, 52, 0.18), transparent 30%),
        linear-gradient(180deg, #f6f0e5 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    .hero, .card {{
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 22px;
      box-shadow: 0 14px 30px rgba(31, 27, 22, 0.08);
      backdrop-filter: blur(8px);
    }}
    .hero {{
      display: grid;
      gap: 10px;
      margin-bottom: 18px;
    }}
    h1, h2 {{ margin: 0; }}
    h1 {{
      font-size: clamp(34px, 4vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }}
    .score {{
      font-size: clamp(40px, 7vw, 82px);
      color: var(--accent);
      font-weight: 700;
      line-height: 1;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .meta div {{
      background: rgba(231, 217, 201, 0.35);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    video, img {{
      width: 100%;
      border-radius: 12px;
      display: block;
      background: #000;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      color: var(--muted);
    }}
    .card p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>PhysInOne PMF Report</h1>
      <div class="score">{score:.6f}</div>
      <p>PMF is higher when the frequency-domain motion energy distribution is more similar to the ground truth.</p>
      <div class="meta">
        <div><span class="label">Original Prediction</span><pre>{html.escape(str(pred_path))}</pre></div>
        <div><span class="label">Ground Truth</span><pre>{html.escape(str(gt_path))}</pre></div>
        <div><span class="label">Aligned Input Used For PMF</span><pre>{html.escape(str(pred_used_path))}</pre></div>
        <div><span class="label">GT Used For PMF</span><pre>{html.escape(str(gt_used_path))}</pre></div>
      </div>
      <div class="meta">
        <div><span class="label">Prediction Original Shape</span><strong>{html.escape(metadata["pred_original"])}</strong></div>
        <div><span class="label">GT Original Shape</span><strong>{html.escape(metadata["gt_original"])}</strong></div>
        <div><span class="label">Shape Used For PMF</span><strong>{html.escape(metadata["used_shape"])}</strong></div>
        <div><span class="label">FPS Used For Saved Preview</span><strong>{metadata["fps_used"]:.3f}</strong></div>
      </div>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Aligned Prediction Used For PMF</h2>
        <video controls loop playsinline src="{html.escape(pred_used_path.name)}"></video>
        <p>This is the prediction after the exact temporal and spatial alignment step used before PMF scoring.</p>
      </article>
      <article class="card">
        <h2>Ground Truth Used For PMF</h2>
        <video controls loop playsinline src="{html.escape(gt_used_path.name)}"></video>
        <p>This matches the PMF reference tensor dimensions. In this pair, it is the decoded GT video.</p>
      </article>
      <article class="card">
        <h2>Side-by-Side Comparison</h2>
        <video controls loop playsinline src="{html.escape(compare_path.name)}"></video>
        <p>Left: aligned prediction. Right: ground truth.</p>
      </article>
      <article class="card">
        <h2>First Frame Preview</h2>
        <img alt="side-by-side first frame" src="compare_first_frame.png">
        <p>Quick static preview for remote access without scrubbing the video.</p>
      </article>
    </section>
  </main>
</body>
</html>
"""
    report_path.write_text(report_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PhysInOne PMF score between two video files.")
    parser.add_argument("--pred", required=True, help="Prediction video path")
    parser.add_argument("--gt", required=True, help="Ground-truth video path")
    parser.add_argument("--out-dir", required=True, help="Directory for report artifacts")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        choices=["mp4v", "h264_baseline"],
        help="Codec used for exported preview videos",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred).resolve()
    gt_path = Path(args.gt).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frames, pred_fps = read_video(pred_path)
    gt_frames, gt_fps = read_video(gt_path)

    pred_tensor = video_tensor_from_numpy(pred_frames)
    gt_tensor = video_tensor_from_numpy(gt_frames)

    gt_for_pmf = _ensure_5d_b_t_c_h_w(gt_tensor).to(args.device)
    pred_for_pmf = _ensure_5d_b_t_c_h_w(pred_tensor).to(args.device)
    pred_aligned = align_pred_to_gt(pred_for_pmf, gt_for_pmf)

    score_tensor = compute_pmf(gt_tensor, pred_tensor, device=args.device)
    score = float(score_tensor.squeeze().detach().cpu().item())

    pred_aligned_np = tensor_b_t_c_h_w_to_numpy_t_h_w_c(pred_aligned)
    gt_used_np = tensor_b_t_c_h_w_to_numpy_t_h_w_c(gt_for_pmf)
    compare_np = make_side_by_side(pred_aligned_np, gt_used_np)

    pred_used_path = out_dir / "pred_used_for_pmf.mp4"
    gt_used_path = out_dir / "gt_used_for_pmf.mp4"
    compare_path = out_dir / "compare_side_by_side.mp4"
    report_path = out_dir / "index.html"
    first_frame_path = out_dir / "compare_first_frame.png"
    json_path = out_dir / "result.json"

    write_video_with_codec(pred_used_path, pred_aligned_np, gt_fps, args.video_codec)
    write_video_with_codec(gt_used_path, gt_used_np, gt_fps, args.video_codec)
    write_video_with_codec(compare_path, compare_np, gt_fps, args.video_codec)
    save_frame(first_frame_path, compare_np[0])

    metadata = {
        "pred_original": f"{pred_frames.shape[0]} x {pred_frames.shape[1]} x {pred_frames.shape[2]} x {pred_frames.shape[3]} @ {pred_fps:.3f} fps",
        "gt_original": f"{gt_frames.shape[0]} x {gt_frames.shape[1]} x {gt_frames.shape[2]} x {gt_frames.shape[3]} @ {gt_fps:.3f} fps",
        "used_shape": f"{gt_used_np.shape[0]} x {gt_used_np.shape[1]} x {gt_used_np.shape[2]} x {gt_used_np.shape[3]}",
        "fps_used": gt_fps,
        "score": score,
        "video_codec": args.video_codec,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "pred_used_path": str(pred_used_path),
        "gt_used_path": str(gt_used_path),
        "compare_path": str(compare_path),
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    build_report(
        report_path=report_path,
        score=score,
        pred_path=pred_path,
        gt_path=gt_path,
        pred_used_path=pred_used_path,
        gt_used_path=gt_used_path,
        compare_path=compare_path,
        metadata=metadata,
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
