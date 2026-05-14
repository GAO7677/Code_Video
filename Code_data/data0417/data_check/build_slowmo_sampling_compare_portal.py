"""Build a side-by-side portal comparing baseline and dense frame sampling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_root", type=Path, required=True)
    parser.add_argument("--dense_root", type=Path, required=True)
    parser.add_argument("--extra_root", type=Path, default=None)
    parser.add_argument("--baseline_label", type=str, default="Baseline")
    parser.add_argument("--dense_label", type=str, default="Dense Sampling")
    parser.add_argument("--extra_label", type=str, default="Dense No-Slowmo")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Sampling Compare")
    return parser.parse_args()


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 = q1 / max(np.linalg.norm(q1), 1e-8)
    q2 = q2 / max(np.linalg.norm(q2), 1e-8)
    dot = abs(float(np.dot(q1, q2)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def summarize_sample(meta_path: Path) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sample_dir = meta_path.parent
    gif_path = sample_dir / "visualizations" / "rgb_preview.gif"
    kin = np.load(sample_dir / "physics" / "rigid_kinematics.npz")
    quat = kin["orientation_quat"]
    deg = [quat_angle_deg(quat[t - 1, 0], quat[t, 0]) for t in range(1, quat.shape[0])]
    deg_arr = np.asarray(deg, dtype=float) if deg else np.zeros((0,), dtype=float)
    sim = meta["simulation"]
    return {
        "scene_id": str(meta["scene_id"]),
        "case_name": str(meta.get("case_name", "")),
        "gif_path": str(gif_path),
        "gif_noslow_path": str(sample_dir / "visualizations" / "rgb_preview_noslow.gif"),
        "meta_path": str(meta_path),
        "frame_dt": float(sim["frame_dt"]),
        "base_video_fps": float(sim["base_video_fps"]),
        "requested_video_fps": float(sim.get("requested_video_fps", sim["base_video_fps"])),
        "render_video_fps": float(sim["video_fps"]),
        "slowdown_factor": float(sim.get("playback_slowdown_factor", 1.0)),
        "sampling_fps_mult": float(sim.get("sampling_fps_mult", 1.0)),
        "frames": int(meta["frames"]),
        "mean_deg_per_frame": float(deg_arr.mean()) if deg_arr.size else 0.0,
        "p95_deg_per_frame": float(np.percentile(deg_arr, 95)) if deg_arr.size else 0.0,
        "max_deg_per_frame": float(deg_arr.max()) if deg_arr.size else 0.0,
    }


def scan(root: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(root.rglob("meta.json")):
        gif_path = meta_path.parent / "visualizations" / "rgb_preview.gif"
        if not gif_path.exists():
            continue
        sample = summarize_sample(meta_path)
        samples[sample["scene_id"]] = sample
    return samples


def render_branch(label: str, sample: dict[str, Any], *, media_key: str = "gif_path") -> str:
    return f"""
    <section class="branch">
      <div class="branch-title">{label}</div>
      <img class="media" src="{sample[media_key]}" loading="lazy" />
      <div class="stat">frame_dt: {sample['frame_dt']:.3f}s</div>
      <div class="stat">sampling_fps: {sample['base_video_fps']:.2f}</div>
      <div class="stat">requested_fps: {sample['requested_video_fps']:.2f}</div>
      <div class="stat">render_fps: {sample['render_video_fps']:.2f}</div>
      <div class="stat">slowdown: x{sample['slowdown_factor']:.2f}</div>
      <div class="stat">sampling_mult: x{sample['sampling_fps_mult']:.2f}</div>
      <div class="stat">frames: {sample['frames']}</div>
      <div class="stat">mean rot/frame: {sample['mean_deg_per_frame']:.2f}°</div>
      <div class="stat">p95 rot/frame: {sample['p95_deg_per_frame']:.2f}°</div>
      <div class="stat">max rot/frame: {sample['max_deg_per_frame']:.2f}°</div>
      <div class="path">{sample['meta_path']}</div>
    </section>
    """


def build_html(title: str, rows: list[str], has_extra: bool) -> str:
    rows_html = "\n".join(rows) if rows else "<p>no compare pairs found</p>"
    cols = "1fr 1fr 1fr" if has_extra else "1fr 1fr"
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin:0; font-family:Arial, 'PingFang SC', sans-serif; background:#f4f1eb; color:#1f1d1a; }}
    header {{ position:sticky; top:0; z-index:10; background:rgba(244,241,235,0.95); border-bottom:1px solid #d7d0c6; padding:16px 20px; }}
    h1 {{ margin:0; font-size:22px; }}
    .sub {{ margin-top:6px; color:#6a645c; font-size:13px; }}
    main {{ padding:18px; display:grid; gap:16px; }}
    .row {{ background:#fffdf8; border:1px solid #d7d0c6; border-radius:14px; padding:14px; }}
    .scene {{ font-size:16px; font-weight:700; margin-bottom:10px; }}
    .pair {{ display:grid; grid-template-columns:{cols}; gap:14px; }}
    .branch {{ border:1px solid #e0d8cc; border-radius:12px; padding:10px; background:#fff; }}
    .branch-title {{ font-size:14px; font-weight:700; margin-bottom:8px; }}
    .media {{ width:100%; aspect-ratio:4/3; object-fit:contain; background:#000; border-radius:8px; display:block; }}
    .stat {{ margin-top:6px; font-size:12px; color:#5f5952; }}
    .path {{ margin-top:8px; font-size:11px; color:#7a736a; word-break:break-all; }}
    @media (max-width: 1100px) {{ .pair {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="sub">对比原始采样、dense 采样，以及可选的 dense 无慢放版本。重点看逐帧角度变化和主观翻滚连续性。</div>
  </header>
  <main>{rows_html}</main>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    baseline = scan(args.baseline_root)
    dense = scan(args.dense_root)
    extra = scan(args.extra_root) if args.extra_root else {}
    rows: list[str] = []
    scene_ids = sorted(set(baseline) & set(dense))
    if extra:
        scene_ids = [scene_id for scene_id in scene_ids if scene_id in extra]
    has_extra = bool(extra)
    if not has_extra:
        has_extra = any(Path(dense[scene_id]["gif_noslow_path"]).exists() for scene_id in scene_ids)
    for scene_id in scene_ids:
        b = baseline[scene_id]
        d = dense[scene_id]
        extra_html = ""
        if extra:
            extra_html = render_branch(args.extra_label, extra[scene_id])
        elif Path(d["gif_noslow_path"]).exists():
            extra_html = render_branch(args.extra_label, d, media_key="gif_noslow_path")
        rows.append(
            f"""
            <article class="row">
              <div class="scene">{scene_id} | {b['case_name']}</div>
              <div class="pair">
                {render_branch(args.baseline_label, b)}
                {render_branch(args.dense_label, d)}
                {extra_html}
              </div>
            </article>
            """
        )
    (args.output_root / "index.html").write_text(build_html(args.title, rows, has_extra), encoding="utf-8")


if __name__ == "__main__":
    main()
