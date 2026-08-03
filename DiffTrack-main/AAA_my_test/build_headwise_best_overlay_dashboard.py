#!/usr/bin/env python3
"""Build browser-playable overlays for the globally best head of each model."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_headwise_50case")
OUTPUT = ROOT / "overlay"
VIDEO_OUTPUT = OUTPUT / "videos"

MODELS = (
    {
        "key": "gt",
        "label": "GT teacher-forced",
        "head": 18,
        "layer": 5,
        "step": 29,
        "color": (38, 195, 255),
    },
    {
        "key": "stage1b",
        "label": "Stage1b step-004000",
        "head": 6,
        "layer": 23,
        "step": 39,
        "color": (91, 221, 145),
    },
    {
        "key": "lora",
        "label": "LoRA step-000500",
        "head": 18,
        "layer": 5,
        "step": 39,
        "color": (255, 164, 73),
    },
    {
        "key": "baseline",
        "label": "Wan2.2 baseline",
        "head": 6,
        "layer": 23,
        "step": 39,
        "color": (238, 102, 132),
    },
)

PANEL_WIDTH = 640
PANEL_HEIGHT = 360
ANALYSIS_WIDTH = 896.0
ANALYSIS_HEIGHT = 512.0
MAX_EVAL_FRAMES = 25


def load_model_case(model: dict, case_name: str) -> dict:
    case_dir = ROOT / model["key"] / "cases" / case_name
    track_key = (
        f"qk_head{model['head']:02d}_layer{model['layer']:02d}_"
        f"step{model['step']:03d}_predictions"
    )
    with np.load(case_dir / "predicted_tracks.npz", allow_pickle=False) as data:
        predictions = data[track_key].astype(np.float32)
    with np.load(case_dir / "cotracker_pseudo_gt.npz", allow_pickle=False) as data:
        gt_tracks = data["tracks"].astype(np.float32)
        visibility = data["visibility"].astype(bool)
        anchors = data["latent_anchor_frames"].astype(np.int32)
    manifest = json.loads((case_dir / "manifest.json").read_text())
    source = manifest.get("gt_video") or manifest.get("context_video")
    if not source:
        raise RuntimeError(f"No source video in {case_dir / 'manifest.json'}")
    return {
        "predictions": predictions,
        "gt_tracks": gt_tracks,
        "visibility": visibility,
        "anchors": anchors,
        "source": Path(source),
    }


def interpolate_predictions(predictions: np.ndarray, anchors: np.ndarray, frames: int) -> np.ndarray:
    timeline = np.arange(frames, dtype=np.float32)
    result = np.empty((frames, predictions.shape[1], 2), dtype=np.float32)
    for point in range(predictions.shape[1]):
        for axis in range(2):
            values = predictions[:, point, axis]
            valid = np.isfinite(values) & (anchors >= 0) & (anchors < frames)
            if not valid.any():
                result[:, point, axis] = np.nan
            elif valid.sum() == 1:
                result[:, point, axis] = values[valid][0]
            else:
                result[:, point, axis] = np.interp(
                    timeline, anchors[valid].astype(np.float32), values[valid]
                )
    return result


def point_xy(point: np.ndarray) -> tuple[int, int]:
    return (
        int(round(float(point[0]) * PANEL_WIDTH / ANALYSIS_WIDTH)),
        int(round(float(point[1]) * PANEL_HEIGHT / ANALYSIS_HEIGHT)),
    )


def draw_panel(frame: np.ndarray, data: dict, model: dict, frame_index: int) -> np.ndarray:
    panel = cv2.resize(frame, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    accent = model["color"]
    gt_tracks = data["gt_tracks"]
    pred_tracks = data["interpolated"]
    visibility = data["visibility"]
    points = min(gt_tracks.shape[1], pred_tracks.shape[1])

    trail_start = max(0, frame_index - 10)
    trail_layer = panel.copy()
    for point in range(points):
        gt_path = []
        pred_path = []
        for t in range(trail_start, frame_index + 1):
            if t < visibility.shape[0] and visibility[t, point] and np.isfinite(gt_tracks[t, point]).all():
                gt_path.append(point_xy(gt_tracks[t, point]))
            if np.isfinite(pred_tracks[t, point]).all():
                pred_path.append(point_xy(pred_tracks[t, point]))
        if len(gt_path) > 1:
            cv2.polylines(trail_layer, [np.asarray(gt_path)], False, (245, 245, 245), 1, cv2.LINE_AA)
        if len(pred_path) > 1:
            cv2.polylines(trail_layer, [np.asarray(pred_path)], False, accent, 2, cv2.LINE_AA)
    panel = cv2.addWeighted(trail_layer, 0.58, panel, 0.42, 0)

    for point in range(points):
        gt_ok = (
            frame_index < gt_tracks.shape[0]
            and frame_index < visibility.shape[0]
            and visibility[frame_index, point]
            and np.isfinite(gt_tracks[frame_index, point]).all()
        )
        pred_ok = np.isfinite(pred_tracks[frame_index, point]).all()
        gt_xy = point_xy(gt_tracks[frame_index, point]) if gt_ok else None
        pred_xy = point_xy(pred_tracks[frame_index, point]) if pred_ok else None
        if gt_xy and pred_xy:
            cv2.line(panel, gt_xy, pred_xy, (92, 98, 105), 1, cv2.LINE_AA)
        if gt_xy:
            cv2.circle(panel, gt_xy, 6, (15, 18, 22), 3, cv2.LINE_AA)
            cv2.circle(panel, gt_xy, 6, (250, 250, 250), 2, cv2.LINE_AA)
        if pred_xy:
            cv2.circle(panel, pred_xy, 5, (12, 15, 18), -1, cv2.LINE_AA)
            cv2.circle(panel, pred_xy, 4, accent, -1, cv2.LINE_AA)

    shade = panel.copy()
    cv2.rectangle(shade, (0, 0), (PANEL_WIDTH, 50), (10, 14, 18), -1)
    panel = cv2.addWeighted(shade, 0.82, panel, 0.18, 0)
    title = f"{model['label']}  H{model['head']:02d}  L{model['layer']:02d}/S{model['step']:03d}"
    cv2.putText(panel, title, (16, 23), cv2.FONT_HERSHEY_DUPLEX, 0.58, (246, 247, 242), 1, cv2.LINE_AA)
    cv2.putText(panel, "white ring: pseudo-GT   color: Q@K", (16, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 197, 201), 1, cv2.LINE_AA)
    return panel


def encode_case(case_name: str) -> None:
    model_data = [load_model_case(model, case_name) for model in MODELS]
    source = model_data[0]["source"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = min(MAX_EVAL_FRAMES, *(item["gt_tracks"].shape[0] for item in model_data))
    for item in model_data:
        item["interpolated"] = interpolate_predictions(item["predictions"], item["anchors"], frames)

    output_path = VIDEO_OUTPUT / f"{case_name}.mp4"
    command = [
        get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{PANEL_WIDTH * 2}x{PANEL_HEIGHT * 2}",
        "-r", f"{fps:.8g}", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_index in range(frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Source ended at frame {frame_index}: {source}")
            panels = [
                draw_panel(frame, item, model, frame_index)
                for item, model in zip(model_data, MODELS)
            ]
            montage = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
            assert process.stdin is not None
            process.stdin.write(np.ascontiguousarray(montage).tobytes())
    finally:
        capture.release()
        if process.stdin:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {case_name}")


def write_dashboard(case_names: list[str]) -> None:
    cards = []
    for index, case_name in enumerate(case_names, start=1):
        label = case_name.removeprefix("case_").replace("_", " ")
        cards.append(
            f'''<article class="case" data-name="{html.escape(label)}">
              <div class="case-head"><span>{index:02d}</span><h2>{html.escape(label)}</h2></div>
              <video controls muted loop playsinline preload="metadata" src="videos/{html.escape(case_name)}.mp4?v=1"></video>
            </article>'''
        )
    config = " · ".join(
        f"{model['label']}: H{model['head']:02d}" for model in MODELS
    )
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>50-case best-head Q@K overlays</title>
<style>
:root{{--ink:#16201e;--paper:#f3f0e7;--card:#fffdf7;--line:#d8d1c2;--accent:#d65336;}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:radial-gradient(circle at 12% 4%,#f8d9b8 0,transparent 28rem),linear-gradient(135deg,#f5f1e7,#e8eee8);font-family:"Avenir Next","Trebuchet MS",sans-serif}}
header{{padding:54px clamp(20px,5vw,76px) 28px;border-bottom:1px solid var(--line)}}
.eyebrow{{font:700 12px/1.2 monospace;letter-spacing:.18em;text-transform:uppercase;color:var(--accent)}}
h1{{margin:10px 0 8px;font-family:Georgia,serif;font-size:clamp(34px,5vw,68px);line-height:.98;font-weight:500}}
.sub{{max-width:1050px;color:#52605b;line-height:1.6}} .toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}}
button,input{{border:1px solid #bdb5a5;background:#fffdf7;color:var(--ink);border-radius:999px;padding:10px 15px;font:600 13px inherit}} input{{min-width:240px}} button{{cursor:pointer}} button:hover{{background:#16201e;color:white}}
main{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;padding:28px clamp(20px,5vw,76px) 70px}}
.case{{background:rgba(255,253,247,.9);border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 15px 40px rgba(35,44,39,.08)}}
.case-head{{display:flex;align-items:center;gap:12px;padding:14px 17px}} .case-head span{{font:700 12px monospace;color:var(--accent)}} h2{{margin:0;font-size:16px;text-transform:capitalize}} video{{display:block;width:100%;aspect-ratio:16/9;background:#101418}}
.hidden{{display:none}} @media(max-width:820px){{main{{grid-template-columns:1fr;padding-inline:12px}}header{{padding-inline:20px}}}}
</style></head><body>
<header><div class="eyebrow">DiffTrack · independent attention heads</div><h1>Best-head trajectory overlays</h1>
<p class="sub">50 evaluated cases, one page. Each 2×2 video overlays pseudo-GT tracks (white rings) and the globally selected Q@K head (colored points and trails). No per-case head selection and no head averaging.<br>{html.escape(config)}</p>
<div class="toolbar"><button id="play">Play all</button><button id="pause">Pause all</button><input id="search" placeholder="Filter cases"></div></header>
<main>{''.join(cards)}</main>
<script>
const videos=()=>[...document.querySelectorAll('.case:not(.hidden) video')];
document.querySelector('#play').onclick=()=>videos().forEach(v=>v.play().catch(()=>{{}}));
document.querySelector('#pause').onclick=()=>videos().forEach(v=>v.pause());
document.querySelector('#search').oninput=e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('.case').forEach(c=>c.classList.toggle('hidden',!c.dataset.name.includes(q)))}};
</script></body></html>'''
    (OUTPUT / "index.html").write_text(page)


def main() -> None:
    VIDEO_OUTPUT.mkdir(parents=True, exist_ok=True)
    case_names = sorted(path.name for path in (ROOT / "gt" / "cases").iterdir() if path.is_dir())
    for index, case_name in enumerate(case_names, start=1):
        encode_case(case_name)
        print(f"[{index:02d}/{len(case_names):02d}] {case_name}", flush=True)
    write_dashboard(case_names)
    print(f"dashboard: {OUTPUT / 'index.html'}")


if __name__ == "__main__":
    main()
