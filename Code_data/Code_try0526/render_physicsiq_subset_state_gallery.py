#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BENCHMARK_ROOT = Path(
    "/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark")
OUTPUT_ROOT = BENCHMARK_ROOT / "output"
REPORT_DIR = BENCHMARK_ROOT / "report_subset"
REPORT_PATH = REPORT_DIR / "index.html"
MANIFEST_PATH = REPORT_DIR / "selected_cases.json"
ASSET_ROOT = REPORT_DIR / "assets"

METHODS = ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
METHOD_LABELS = {
    "GT": "GT",
    "wan22-5B-TI2V": "Wan2.2-5B TI2V",
    "VACE_1p3B_TI2V": "VACE 1.3B TI2V",
    "VACE_1p3B_ctx08": "VACE 1.3B ctx=8",
}
TASK_NAME = "physics-iq-benchmark"
SELECTED_CASES = [
    {
        "clip_name": "0002_perspective-center_trimmed-ball-and-block-fall",
        "theme": "Release and free fall",
        "reason":
        "Suspended objects are released together, which is useful for checking gravity, relative fall speed, and support removal."
    },
    {
        "clip_name": "0032_perspective-center_trimmed-balls-collide",
        "theme": "Collision",
        "reason":
        "Two moving balls interact directly, making momentum transfer and contact timing easy to compare."
    },
    {
        "clip_name": "0035_perspective-center_trimmed-block-domino",
        "theme": "Chain reaction",
        "reason":
        "A domino-like sequence stresses multi-object propagation, delayed effects, and event ordering."
    },
    {
        "clip_name": "0089_perspective-center_trimmed-liquid-overfill",
        "theme": "Fluid overflow",
        "reason":
        "Liquid leaving a container is hard for video models and exposes scale drift, spill timing, and visibility errors."
    },
    {
        "clip_name": "0101_perspective-center_trimmed-magnet-wrench",
        "theme": "Magnetic interaction",
        "reason":
        "Non-contact motion tests whether the model can preserve causal motion without visible pushes."
    },
    {
        "clip_name": "0116_perspective-center_trimmed-mirror-ball-fall",
        "theme": "Reflection",
        "reason":
        "Mirror scenes stress object identity, duplicated appearances, and whether the moving object and reflection stay coupled."
    },
    {
        "clip_name": "0125_perspective-center_trimmed-mug-breaks",
        "theme": "Breakage",
        "reason":
        "Topology change and fragments make this a good stress test for sudden state transitions and motion discontinuities."
    },
    {
        "clip_name": "0197_perspective-center_trimmed-weight-protects-duck",
        "theme": "Shielding / support",
        "reason":
        "This scene depends on spatial protection and contact ordering, so it is useful for checking relational reasoning."
    },
]
COLORS = {
    "bbox": (50, 205, 50),
    "center": (32, 99, 245),
    "trail": (230, 130, 50),
}


@dataclass(slots=True)
class MethodView:
    payload: dict[str, Any]
    overlay_video: str
    chart_svg: str
    summary_json: str
    summary: dict[str, Any]


@dataclass(slots=True)
class CaseRecord:
    clip_name: str
    prompt: str
    first_frame: str | None
    source_video_path: str | None
    theme: str
    reason: str
    methods: dict[str, MethodView]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def href_from_report(target: str | Path) -> str:
    return html.escape(
        os.path.relpath(Path(target).resolve(),
                        REPORT_DIR.resolve()).replace("\\", "/"))


def resolve_video_path(payload: dict[str, Any]) -> str:
    for key in ("video_path", "copied_video_path", "source"):
        value = payload.get(key)
        if value:
            return str(value)
    raise KeyError("No usable video path found in payload")


def _polyline(values: list[float], width: int, height: int,
              padding: int) -> str:
    if not values:
        return ""
    n = len(values)
    xs = np.linspace(padding, width - padding, n)
    ys = padding + (1.0 - np.clip(np.asarray(values, dtype=np.float32), 0.0,
                                  1.0)) * (height - 2 * padding)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def render_series_chart(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 860
    height = 280
    padding = 28
    visibility = summary["visibility"]
    hidden_rects = []
    if visibility:
        xs = np.linspace(padding, width - padding, len(visibility))
        step = ((width - 2 * padding) / max(len(visibility) - 1, 1))
        for idx, vis in enumerate(visibility):
            if int(vis) == 0:
                x0 = max(0.0, xs[idx] - step * 0.5)
                hidden_rects.append(
                    f'<rect x="{x0:.2f}" y="{padding}" width="{max(6.0, step):.2f}" height="{height - 2 * padding}" fill="rgba(220,80,80,0.12)" />'
                )
    legend = """
      <g font-size="12" fill="#6d6157">
        <circle cx="44" cy="20" r="5" fill="#1f77b4"/><text x="56" y="24">center_x</text>
        <circle cx="148" cy="20" r="5" fill="#d62728"/><text x="160" y="24">center_y</text>
        <circle cx="252" cy="20" r="5" fill="#2ca02c"/><text x="264" y="24">area_ratio</text>
      </g>
    """
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" rx="18" fill="#fffdf9" stroke="#dacabb"/>
  {''.join(hidden_rects)}
  <g stroke="#ece3d8" stroke-width="1">
    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" />
    <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" />
    <line x1="{padding}" y1="{padding + (height-2*padding)/2:.2f}" x2="{width-padding}" y2="{padding + (height-2*padding)/2:.2f}" />
  </g>
  <polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{_polyline(summary['center_x_norm'], width, height, padding)}" />
  <polyline fill="none" stroke="#d62728" stroke-width="3" points="{_polyline(summary['center_y_norm'], width, height, padding)}" />
  <polyline fill="none" stroke="#2ca02c" stroke-width="3" points="{_polyline(summary['area_ratio_norm'], width, height, padding)}" />
  {legend}
  <text x="{width-220}" y="24" font-size="12" fill="#6d6157">red bands = invisible / extraction miss</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def extract_motion_proxy(video_path: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        grays.append(
            cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5),
                             0))
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {video_path}")

    height, width = frames[0].shape[:2]
    frame_area = float(height * width)
    ref = grays[0]
    prev = ref
    prev_center: tuple[float, float] | None = None
    prev_bbox: tuple[int, int, int, int] | None = None

    centers: list[tuple[float, float] | None] = []
    boxes: list[tuple[int, int, int, int] | None] = []
    area_ratios: list[float] = []
    visibility: list[int] = []
    speeds: list[float] = []

    min_area = max(60.0, frame_area * 0.00022)
    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_big = np.ones((7, 7), dtype=np.uint8)

    for gray in grays:
        diff_ref = cv2.absdiff(gray, ref)
        diff_prev = cv2.absdiff(gray, prev)
        motion = cv2.max(diff_ref, diff_prev)
        _, mask = cv2.threshold(motion, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w * 0.5
            cy = y + h * 0.5
            score = area
            if prev_center is not None:
                dx = (cx - prev_center[0]) / width
                dy = (cy - prev_center[1]) / height
                score -= 0.35 * frame_area * np.hypot(dx, dy)
            if prev_bbox is not None:
                px, py, pw, ph = prev_bbox
                inter_w = max(0, min(x + w, px + pw) - max(x, px))
                inter_h = max(0, min(y + h, py + ph) - max(y, py))
                inter = inter_w * inter_h
                union = w * h + pw * ph - inter
                iou = inter / union if union > 0 else 0.0
                score += 0.12 * frame_area * iou
            if best_score is None or score > best_score:
                best = (x, y, w, h, cx, cy, area)
                best_score = score

        if best is None:
            centers.append(None)
            boxes.append(None)
            area_ratios.append(0.0)
            visibility.append(0)
            speeds.append(0.0)
            prev = gray
            continue

        x, y, w, h, cx, cy, area = best
        center = (float(cx), float(cy))
        bbox = (int(x), int(y), int(w), int(h))
        centers.append(center)
        boxes.append(bbox)
        area_ratios.append(float(area / frame_area))
        visibility.append(1)
        if prev_center is None:
            speeds.append(0.0)
        else:
            speeds.append(
                float(
                    np.hypot((center[0] - prev_center[0]) / width,
                             (center[1] - prev_center[1]) / height)))
        prev_center = center
        prev_bbox = bbox
        prev = gray

    center_x_norm = [(c[0] / width) if c is not None else 0.0 for c in centers]
    center_y_norm = [(c[1] / height) if c is not None else 0.0 for c in centers]
    max_area = max(area_ratios) if area_ratios else 0.0
    area_ratio_norm = [value / max_area if max_area > 0 else 0.0
                       for value in area_ratios]
    visible_fraction = float(np.mean(visibility)) if visibility else 0.0
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    valid_areas = [value for value, vis in zip(area_ratios, visibility) if vis]
    area_cv = 0.0
    if valid_areas:
        mean_area = float(np.mean(valid_areas))
        if mean_area > 1e-8:
            area_cv = float(np.std(valid_areas) / mean_area)

    summary = {
        "fps": fps,
        "num_frames": len(frames),
        "width": width,
        "height": height,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "area_ratio_norm": area_ratio_norm,
        "area_ratio_raw": area_ratios,
        "visibility": visibility,
        "speed_per_frame": speeds,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "visible_fraction": visible_fraction,
        "area_cv": area_cv,
        "boxes_xywh": [[*box] if box is not None else None for box in boxes],
        "centers_xy": [[*center] if center is not None else None
                        for center in centers],
        "visible_frame_count": int(sum(visibility)),
    }
    return summary, frames


def build_overlay_video(summary: dict[str, Any], frames: list[np.ndarray],
                        out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height = int(summary["height"])
    width = int(summary["width"])
    fps = float(summary["fps"]) if summary["fps"] else 16.0
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    trail: list[tuple[int, int]] = []
    for idx, frame in enumerate(frames):
        canvas = frame.copy()
        box = summary["boxes_xywh"][idx]
        center = summary["centers_xy"][idx]
        vis = int(summary["visibility"][idx])
        if center is not None:
            trail.append((int(center[0]), int(center[1])))
        trail = trail[-18:]
        for t_idx in range(1, len(trail)):
            alpha = t_idx / max(len(trail) - 1, 1)
            color = (
                int(COLORS["trail"][0] * alpha + 255 * (1 - alpha)),
                int(COLORS["trail"][1] * alpha + 255 * (1 - alpha)),
                int(COLORS["trail"][2] * alpha + 255 * (1 - alpha)),
            )
            cv2.line(canvas, trail[t_idx - 1], trail[t_idx], color, 2,
                     cv2.LINE_AA)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), COLORS["bbox"], 2)
        if center is not None:
            cv2.circle(canvas, (int(center[0]), int(center[1])), 4,
                       COLORS["center"], -1)
        label = (f"frame {idx:02d} | vis={vis} | "
                 f"area={summary['area_ratio_raw'][idx]:.4f} | "
                 f"speed={summary['speed_per_frame'][idx]:.4f}")
        cv2.rectangle(canvas, (10, 10), (width - 10, 52), (250, 247, 240), -1)
        cv2.putText(canvas, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (36, 29, 22), 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()


def build_assets_for_video(case: CaseRecord, method: str,
                           payload: dict[str, Any]) -> MethodView:
    asset_dir = ASSET_ROOT / case.clip_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    overlay_video_path = asset_dir / f"{method}.overlay.mp4"
    chart_svg_path = asset_dir / f"{method}.series.svg"
    summary_json_path = asset_dir / f"{method}.summary.json"
    if not (overlay_video_path.exists() and chart_svg_path.exists()
            and summary_json_path.exists()):
        summary, frames = extract_motion_proxy(Path(resolve_video_path(payload)))
        render_series_chart(chart_svg_path, summary)
        build_overlay_video(summary, frames, overlay_video_path)
        write_json(summary_json_path, summary)
    else:
        summary = load_json(summary_json_path)
    return MethodView(
        payload=payload,
        overlay_video=str(overlay_video_path),
        chart_svg=str(chart_svg_path),
        summary_json=str(summary_json_path),
        summary=summary,
    )


def discover_cases() -> list[CaseRecord]:
    gt_root = OUTPUT_ROOT / "GT" / TASK_NAME
    by_name = {path.stem: load_json(path) for path in gt_root.glob("*.json")}
    cases: list[CaseRecord] = []
    for item in SELECTED_CASES:
        clip_name = item["clip_name"]
        gt_payload = by_name.get(clip_name)
        if gt_payload is None:
            continue
        methods = {}
        missing = False
        for method in METHODS:
            json_path = OUTPUT_ROOT / method / TASK_NAME / f"{clip_name}.json"
            if not json_path.exists():
                missing = True
                break
            payload = load_json(json_path)
            methods[method] = build_assets_for_video(
                CaseRecord(
                    clip_name=clip_name,
                    prompt=gt_payload["prompt"],
                    first_frame=gt_payload.get("first_frame"),
                    source_video_path=gt_payload.get("source_video_path"),
                    theme=item["theme"],
                    reason=item["reason"],
                    methods={},
                ), method, payload)
        if missing:
            continue
        cases.append(
            CaseRecord(
                clip_name=clip_name,
                prompt=gt_payload["prompt"],
                first_frame=gt_payload.get("first_frame"),
                source_video_path=gt_payload.get("source_video_path"),
                theme=item["theme"],
                reason=item["reason"],
                methods=methods,
            ))
    return cases


def render_video_card(method: str, view: MethodView) -> str:
    payload = view.payload
    original_rel = href_from_report(resolve_video_path(payload))
    overlay_rel = href_from_report(view.overlay_video)
    chart_rel = href_from_report(view.chart_svg)
    summary = view.summary
    conditioning = html.escape(str(payload.get("conditioning_mode", "-")))
    return f"""
    <article class="video-card">
      <div class="video-card-head">
        <div>
          <div class="method-name">{html.escape(METHOD_LABELS.get(method, method))}</div>
          <div class="meta-line"><strong>Conditioning</strong> {conditioning}</div>
          <div class="meta-line">
            <span><strong>Visible</strong> {summary['visible_frame_count']} / {summary['num_frames']}</span>
            <span><strong>Vis frac</strong> {summary['visible_fraction']:.3f}</span>
            <span><strong>Mean speed</strong> {summary['mean_speed']:.4f}</span>
          </div>
          <div class="meta-line">
            <span><strong>Max speed</strong> {summary['max_speed']:.4f}</span>
            <span><strong>Area CV</strong> {summary['area_cv']:.4f}</span>
          </div>
        </div>
      </div>
      <div class="video-pair">
        <div>
          <div class="mini-title">Original</div>
          <video controls preload="metadata" src="{original_rel}"></video>
        </div>
        <div>
          <div class="mini-title">State Overlay</div>
          <video controls preload="metadata" src="{overlay_rel}"></video>
        </div>
      </div>
      <div class="chart-box">
        <div class="mini-title">Extracted Motion Proxy</div>
        <img src="{chart_rel}" alt="state series chart" />
      </div>
      <div class="link-row">
        <a href="{original_rel}">Open original</a>
        <a href="{overlay_rel}">Open overlay</a>
        <a href="{chart_rel}">Open chart</a>
      </div>
    </article>
    """


def render_method_summary_table(case: CaseRecord) -> str:
    rows = []
    for method in METHODS:
        summary = case.methods[method].summary
        rows.append(f"""
        <tr>
          <td>{html.escape(METHOD_LABELS.get(method, method))}</td>
          <td>{summary['visible_frame_count']} / {summary['num_frames']}</td>
          <td>{summary['visible_fraction']:.3f}</td>
          <td>{summary['mean_speed']:.4f}</td>
          <td>{summary['max_speed']:.4f}</td>
          <td>{summary['area_cv']:.4f}</td>
        </tr>
        """)
    return f"""
    <div class="table-wrap">
      <table class="metric-table">
        <thead>
          <tr>
            <th>Method</th>
            <th>Visible Frames</th>
            <th>Visible Fraction</th>
            <th>Mean Speed</th>
            <th>Max Speed</th>
            <th>Area CV</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def render_case(case: CaseRecord) -> str:
    cards = "".join(
        render_video_card(method, case.methods[method]) for method in METHODS)
    frame_html = ""
    if case.first_frame:
        frame_html = (
            f'<img src="{href_from_report(case.first_frame)}" alt="first frame" />')
    source_line = ""
    if case.source_video_path:
        source_line = (
            f'<div class="source-line"><strong>Source</strong>: '
            f'<code>{html.escape(case.source_video_path)}</code></div>')
    return f"""
    <section class="case-card" id="{html.escape(case.clip_name)}">
      <div class="case-top">
        <div class="case-copy">
          <div class="eyebrow">{html.escape(case.theme)}</div>
          <h2>{html.escape(case.clip_name)}</h2>
          <p class="prompt"><strong>Prompt</strong>: {html.escape(case.prompt)}</p>
          <p class="reason">{html.escape(case.reason)}</p>
          {source_line}
        </div>
        <div class="frame-panel">
          <div class="frame-title">Reference First Frame</div>
          {frame_html}
        </div>
      </div>
      <div class="video-grid">
        {cards}
      </div>
      {render_method_summary_table(case)}
    </section>
    """


def render_navigation(cases: list[CaseRecord]) -> str:
    return "".join(
        f'<a href="#{html.escape(case.clip_name)}">{html.escape(case.theme)} / {html.escape(case.clip_name)}</a>'
        for case in cases)


def render_html(cases: list[CaseRecord]) -> str:
    sections = "".join(render_case(case) for case in cases)
    nav = render_navigation(cases)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Physics-IQ Representative Cases with State Overlay</title>
  <style>
    :root {{
      --bg: #f6f1ea;
      --panel: rgba(255, 252, 246, 0.95);
      --line: #dacabb;
      --text: #231b14;
      --muted: #726457;
      --accent: #8e4d2d;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background:
        radial-gradient(circle at 0% 0%, rgba(180, 132, 92, 0.18), transparent 25%),
        radial-gradient(circle at 100% 0%, rgba(76, 112, 156, 0.12), transparent 22%),
        linear-gradient(180deg, #f9f5ef 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1820px;
      margin: 0 auto;
      padding: 24px 24px 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.8fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }}
    .hero-card, .nav-card, .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 42px rgba(82, 63, 47, 0.08);
    }}
    .hero-card {{ padding: 22px 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 38px; letter-spacing: 0.01em; }}
    .sub {{ margin: 0; color: var(--muted); line-height: 1.7; font-size: 15px; }}
    .nav-card {{
      padding: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-content: start;
    }}
    .nav-card a {{
      color: var(--accent);
      text-decoration: none;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.55);
      font-size: 13px;
    }}
    .case-card {{ padding: 20px; margin-bottom: 22px; }}
    .case-top {{
      display: grid;
      grid-template-columns: 1.5fr 0.8fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    h2 {{ margin: 0 0 8px; font-size: 30px; }}
    .prompt, .reason, .source-line {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.75;
      font-size: 14px;
    }}
    .frame-panel {{
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.5);
    }}
    .frame-title {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .frame-panel img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .video-card {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.58);
    }}
    .method-name {{ font-size: 20px; font-weight: 700; margin-bottom: 6px; }}
    .meta-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .video-pair {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .mini-title {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    video, .chart-box img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #000;
    }}
    .chart-box {{ margin-bottom: 10px; }}
    .chart-box img {{ background: #fffdf9; }}
    .link-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .link-row a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
    }}
    .table-wrap {{ overflow-x: auto; }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: rgba(255,255,255,0.58);
    }}
    .metric-table th, .metric-table td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    .metric-table th {{ background: rgba(245, 236, 225, 0.98); }}
    code {{
      font-family: "SFMono-Regular", "Consolas", monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    @media (max-width: 1300px) {{
      .hero, .case-top, .video-grid, .video-pair {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <section class="hero-card">
        <h1>Physics-IQ Representative Cases with State Overlay</h1>
        <p class="sub">
          This page samples eight representative Physics-IQ scenes spanning release, collision, chain reaction, fluid overflow, magnetic interaction,
          reflection, breakage, and shielding. For each case and method, it shows the original video, a lightweight extracted state overlay,
          and a normalized motion proxy chart built from per-frame foreground box, center, scale, and visibility estimates.
        </p>
      </section>
      <nav class="nav-card">
        {nav}
      </nav>
    </div>
    {sections}
  </div>
</body>
</html>
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    cases = discover_cases()
    REPORT_PATH.write_text(render_html(cases), encoding="utf-8")
    write_json(
        MANIFEST_PATH, {
            "cases": [{
                "clip_name": case.clip_name,
                "theme": case.theme,
                "reason": case.reason,
                "methods": {
                    method: {
                        "overlay_video": case.methods[method].overlay_video,
                        "chart_svg": case.methods[method].chart_svg,
                        "summary_json": case.methods[method].summary_json,
                    }
                    for method in METHODS
                },
            } for case in cases]
        })
    print(
        json.dumps(
            {
                "report_path": str(REPORT_PATH),
                "manifest_path": str(MANIFEST_PATH),
                "num_cases": len(cases),
                "asset_root": str(ASSET_ROOT),
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
