#!/usr/bin/env python3
"""Build a static audit page for all 0613 PyBullet windows of one raw case."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


DEFAULT_RAW_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/"
    "industrial_s1_scale2_merged_h264_batch1500"
)
DEFAULT_EPISODE_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/"
    "industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/0613_window_visualization"
)
OBJECT_COLORS_RGB = (
    (15, 124, 140),
    (202, 70, 67),
    (212, 158, 42),
    (42, 105, 168),
    (118, 83, 148),
    (70, 132, 88),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every precomputed 0613 window for one raw case."
    )
    parser.add_argument("--case-id", default="sample_000227")
    parser.add_argument("--split", default="train")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--query-local-frame", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def find_raw_case(raw_root: Path, split: str, case_id: str) -> Path:
    matches = sorted((raw_root / split).glob(f"F*/{case_id}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one raw case for {split}/{case_id}, found {len(matches)}"
        )
    return matches[0]


def episode_paths(episode_root: Path, split: str, case_id: str) -> list[Path]:
    paths = sorted((episode_root / split).glob(f"{case_id}_w*.json"))
    if not paths:
        raise RuntimeError(f"no episode windows found for {split}/{case_id}")
    for path in paths:
        if not path.with_suffix(".npz").is_file():
            raise FileNotFoundError(path.with_suffix(".npz"))
    return paths


def frame_rgb_uint8(frame_chw: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_chw, dtype=np.float32).transpose(1, 2, 0)
    return np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)


def draw_instance_boxes(
    frame_rgb: np.ndarray,
    boxes_o4: np.ndarray,
    objects: list[dict],
    *,
    scale: int,
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    output = cv2.resize(
        frame_rgb,
        (width * scale, height * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    visible_labels: list[tuple[str, tuple[int, int, int]]] = []
    for object_index, obj in enumerate(objects):
        box = np.asarray(boxes_o4[object_index], dtype=np.float32)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        color = OBJECT_COLORS_RGB[object_index % len(OBJECT_COLORS_RGB)]
        x0 = int(round(float(box[0]) * width * scale))
        y0 = int(round(float(box[1]) * height * scale))
        x1 = int(round(float(box[2]) * width * scale))
        y1 = int(round(float(box[3]) * height * scale))
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
        role = "D" if bool(obj.get("dynamic", False)) else "S"
        label = f"O{object_index + 1} {obj.get('name', 'object')} [{role}]"
        visible_labels.append((label, color))
    if visible_labels:
        legend_height = 13 + 19 * len(visible_labels)
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (280, legend_height), (25, 31, 34), -1)
        cv2.addWeighted(overlay, 0.78, output, 0.22, 0.0, output)
    for label_index, (label, color) in enumerate(visible_labels):
        cv2.putText(
            output,
            label,
            (8, 22 + 19 * label_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def render_video_frame(
    frame_rgb: np.ndarray,
    boxes_o4: np.ndarray,
    objects: list[dict],
    *,
    window_name: str,
    local_step: int,
    raw_frame: int,
    context_steps: int,
    query_local_frame: int,
) -> np.ndarray:
    body = draw_instance_boxes(frame_rgb, boxes_o4, objects, scale=3)
    width = body.shape[1]
    header_height = 58
    footer_height = 30
    output = np.full(
        (header_height + body.shape[0] + footer_height, width, 3),
        247,
        dtype=np.uint8,
    )
    output[header_height : header_height + body.shape[0]] = body
    is_context = local_step < context_steps
    phase = "CONTEXT" if is_context else "FUTURE"
    phase_color = (23, 124, 119) if is_context else (211, 101, 47)
    cv2.rectangle(output, (0, 0), (width, header_height), (31, 38, 42), -1)
    cv2.putText(
        output,
        f"{window_name}  |  step {local_step:02d}/23  |  raw F{raw_frame:02d}",
        (18, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (245, 247, 248),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        phase,
        (18, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        phase_color,
        2,
        cv2.LINE_AA,
    )
    if local_step == query_local_frame:
        cv2.putText(
            output,
            "F04 QUERY",
            (width - 148, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (196, 61, 85),
            2,
            cv2.LINE_AA,
        )
    segment_width = width / 24.0
    for step in range(24):
        x0 = int(round(step * segment_width))
        x1 = int(round((step + 1) * segment_width))
        color = (23, 124, 119) if step < context_steps else (211, 101, 47)
        cv2.rectangle(
            output,
            (x0, header_height + body.shape[0]),
            (x1, output.shape[0]),
            color,
            -1,
        )
    marker_x = int(round((local_step + 0.5) * segment_width))
    cv2.line(
        output,
        (marker_x, header_height + body.shape[0]),
        (marker_x, output.shape[0]),
        (249, 249, 249),
        4,
    )
    return output


def resolve_ffmpeg(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable).resolve()
    candidates = sorted(
        (Path.home() / ".local" / "lib").glob(
            "python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        )
    )
    candidates.extend(
        sorted(
            Path("/data/gaoya/agent-data/cache").glob(
                "**/imageio_ffmpeg/binaries/ffmpeg-*"
            )
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "no ffmpeg executable found in PATH, imageio_ffmpeg, or agent cache"
    )


def write_h264_video(
    path: Path,
    frames_rgb: list[np.ndarray],
    fps: float,
    ffmpeg: Path,
) -> None:
    if not frames_rgb:
        raise ValueError(f"cannot write an empty video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.mp4v{path.suffix}")
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {temporary}")
    try:
        for frame in frames_rgb:
            if frame.shape[:2] != (height, width):
                raise ValueError("video frames have inconsistent dimensions")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(path),
        ],
        check=True,
    )
    temporary.unlink()


def contact_sheet(
    frame_rows: list[tuple[np.ndarray, int, int, bool]],
    boxes_t_o4: np.ndarray,
    objects: list[dict],
    *,
    context_steps: int,
    query_local_frame: int,
) -> np.ndarray:
    columns = 6
    tile_width = 384
    frame_height = 216
    label_height = 42
    rows = int(np.ceil(len(frame_rows) / columns))
    sheet = np.full(
        (rows * (frame_height + label_height), columns * tile_width, 3),
        242,
        dtype=np.uint8,
    )
    for local_step, (frame, _, raw_frame, _) in enumerate(frame_rows):
        boxed = draw_instance_boxes(frame, boxes_t_o4[local_step], objects, scale=2)
        boxed = cv2.resize(boxed, (tile_width, frame_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(local_step, columns)
        x0 = column * tile_width
        y0 = row * (frame_height + label_height)
        sheet[y0 : y0 + frame_height, x0 : x0 + tile_width] = boxed
        phase_color = (23, 124, 119) if local_step < context_steps else (211, 101, 47)
        cv2.rectangle(
            sheet,
            (x0, y0 + frame_height),
            (x0 + tile_width, y0 + frame_height + label_height),
            (31, 38, 42),
            -1,
        )
        text = f"step {local_step:02d}  raw F{raw_frame:02d}"
        cv2.putText(
            sheet,
            text,
            (x0 + 10, y0 + frame_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            phase_color,
            2,
            cv2.LINE_AA,
        )
        if local_step == query_local_frame:
            cv2.putText(
                sheet,
                "F04",
                (x0 + tile_width - 55, y0 + frame_height + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (196, 61, 85),
                2,
                cv2.LINE_AA,
            )
    return sheet


def window_timeline_html(records: list[dict], raw_frame_count: int) -> str:
    rows = []
    denominator = max(raw_frame_count - 1, 1)
    for record in records:
        short_name = str(record["name"]).rsplit("_", 1)[-1]
        start = int(record["raw_indices"][0])
        context_end = int(record["raw_indices"][record["context_steps"] - 1])
        future_start = int(record["raw_indices"][record["context_steps"]])
        end = int(record["raw_indices"][-1])
        query = int(record["raw_indices"][record["query_local_frame"]])
        left = 100.0 * start / denominator
        context_width = 100.0 * (context_end - start + 1) / denominator
        future_left = 100.0 * future_start / denominator
        future_width = 100.0 * (end - future_start + 1) / denominator
        query_left = 100.0 * query / denominator
        rows.append(
            f"""
            <div class="timeline-row">
              <div class="timeline-name"><strong>{html.escape(short_name)}</strong><span>raw {start}-{end}</span></div>
              <div class="track">
                <span class="segment context" style="left:{left:.4f}%;width:{context_width:.4f}%">context</span>
                <span class="segment future" style="left:{future_left:.4f}%;width:{future_width:.4f}%">future</span>
                <span class="query-marker" style="left:{query_left:.4f}%"><i>F04</i></span>
              </div>
            </div>
            """
        )
    ticks = "".join(
        f'<span style="left:{100.0 * frame / denominator:.4f}%">{frame}</span>'
        for frame in (0, 16, 32, 48, 64, 80, raw_frame_count - 1)
    )
    return f"""
      <div class="timeline-ruler"><div></div><div class="ticks">{ticks}</div></div>
      {''.join(rows)}
    """


def frame_index_table(record: dict) -> str:
    cells = []
    for local_step, raw_frame in enumerate(record["raw_indices"]):
        phase = "C" if local_step < record["context_steps"] else "P"
        query = " query" if local_step == record["query_local_frame"] else ""
        cells.append(
            f'<div class="frame-cell {"context" if phase == "C" else "future"}{query}">'
            f'<strong>{local_step:02d}</strong><span>raw {raw_frame:02d}</span></div>'
        )
    return "".join(cells)


def render_html(
    *,
    case_id: str,
    raw_case: Path,
    raw_meta: dict,
    records: list[dict],
    raw_frame_count: int,
    raw_fps: float,
) -> str:
    objects = raw_meta.get("objects", [])
    object_rows = []
    for object_index, obj in enumerate(objects):
        color = OBJECT_COLORS_RGB[object_index % len(OBJECT_COLORS_RGB)]
        color_hex = "#" + "".join(f"{value:02x}" for value in color)
        object_rows.append(
            "<tr>"
            f'<td><span class="swatch" style="background:{color_hex}"></span>O{object_index + 1}</td>'
            f"<td>{html.escape(str(obj.get('name', '')))}</td>"
            f"<td>{html.escape(str(obj.get('shape', '')))}</td>"
            f"<td>{'dynamic' if obj.get('dynamic') else html.escape(str(obj.get('role', 'static')))}</td>"
            "</tr>"
        )
    window_sections = []
    for record in records:
        window_sections.append(
            f"""
            <article class="window" id="{html.escape(record['name'])}">
              <header class="window-heading">
                <div><span class="window-index">{html.escape(record['name'])}</span><h2>Raw F{record['raw_indices'][0]:02d}-F{record['raw_indices'][-1]:02d}</h2></div>
                <dl><div><dt>Context</dt><dd>F{record['raw_indices'][0]:02d}-F{record['raw_indices'][record['context_steps'] - 1]:02d}</dd></div><div><dt>Future</dt><dd>F{record['raw_indices'][record['context_steps']]:02d}-F{record['raw_indices'][-1]:02d}</dd></div><div><dt>F04 query</dt><dd>raw F{record['raw_indices'][record['query_local_frame']]:02d}</dd></div></dl>
              </header>
              <div class="frame-strip">{frame_index_table(record)}</div>
              <div class="media-grid">
                <figure><figcaption>Full 24 steps</figcaption><video controls muted loop preload="metadata" src="{record['name']}/full.mp4"></video></figure>
                <figure><figcaption>Context 8 steps</figcaption><video controls muted loop preload="metadata" src="{record['name']}/context.mp4"></video></figure>
                <figure><figcaption>Future 16 steps</figcaption><video controls muted loop preload="metadata" src="{record['name']}/future.mp4"></video></figure>
              </div>
              <figure class="sheet"><figcaption>All sampled frames · local step / raw frame</figcaption><img loading="lazy" src="{record['name']}/contact_sheet.jpg" alt="{html.escape(record['name'])} all 24 sampled frames"></figure>
            </article>
            """
        )
    prompt = records[0]["prompt"]
    first_used = records[0]["raw_indices"][0]
    last_used = records[-1]["raw_indices"][-1]
    frame_stride = int(records[0]["frame_stride"])
    remaining_sampled = list(
        range(last_used + frame_stride, raw_frame_count, frame_stride)
    )
    remaining_label = ", ".join(f"F{value:02d}" for value in remaining_sampled)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(case_id)} · window audit</title>
<style>
:root{{--ink:#1f262a;--muted:#697378;--line:#d7dee1;--paper:#f4f7f7;--white:#fff;--context:#177c77;--future:#d3652f;--query:#c43d55;--blue:#2a69a8}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"IBM Plex Sans","Noto Sans SC","Segoe UI",sans-serif;letter-spacing:0}}header,main,section,article,figure{{min-width:0}}.top{{padding:28px max(24px,calc((100vw - 1440px)/2));background:#253036;color:#f6f8f8;border-bottom:5px solid var(--context)}}.top .kicker{{font-family:"DIN Alternate","Bahnschrift",sans-serif;font-size:13px;color:#9fd4d1;text-transform:uppercase}}h1{{margin:8px 0 4px;font:700 clamp(26px,3vw,42px)/1.05 "DIN Alternate","Bahnschrift",sans-serif;letter-spacing:0}}.top p{{margin:0;color:#cad2d5;font-size:14px}}main{{max-width:1440px;margin:0 auto;padding:24px}}.overview{{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(320px,.6fr);gap:24px;align-items:start;margin-bottom:28px}}.raw-video,.facts{{background:var(--white);border:1px solid var(--line);border-radius:6px;overflow:hidden}}.raw-video figcaption,.facts h2,.sheet figcaption,.media-grid figcaption{{padding:12px 14px;border-bottom:1px solid var(--line);font:700 14px/1.2 "DIN Alternate","Bahnschrift",sans-serif}}video,img{{display:block;width:100%;height:auto;background:#111}}.facts{{padding-bottom:12px}}.facts h2{{margin:0}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #e6ebed}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}.swatch{{display:inline-block;width:10px;height:10px;margin-right:7px;border-radius:2px}}.prompt{{margin:12px;font:12px/1.5 "IBM Plex Mono","Cascadia Mono",monospace;color:#465157;overflow-wrap:anywhere}}.window-map{{background:#fff;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:22px 24px;margin:0 calc(50% - 50vw) 30px}}.window-map-inner{{max-width:1440px;margin:auto}}.window-map h2{{margin:0 0 18px;font:700 20px/1.2 "DIN Alternate","Bahnschrift",sans-serif}}.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:-8px 0 18px;font-size:12px;color:var(--muted)}}.legend span::before{{content:"";display:inline-block;width:20px;height:7px;margin-right:6px;border-radius:1px}}.legend .lc::before{{background:var(--context)}}.legend .lf::before{{background:var(--future)}}.legend .lq::before{{background:var(--query)}}.timeline-ruler,.timeline-row{{display:grid;grid-template-columns:116px 1fr;gap:14px}}.timeline-ruler{{margin-bottom:7px}}.ticks,.track{{position:relative;height:24px}}.ticks span{{position:absolute;transform:translateX(-50%);font:10px "IBM Plex Mono","Cascadia Mono",monospace;color:var(--muted)}}.timeline-name{{display:flex;justify-content:space-between;align-items:center;font-size:12px}}.timeline-name span{{font:10px "IBM Plex Mono","Cascadia Mono",monospace;color:var(--muted)}}.track{{height:32px;background:repeating-linear-gradient(90deg,#edf1f2 0,#edf1f2 1px,transparent 1px,transparent 11.236%);border:1px solid #dfe5e7;margin-bottom:10px;overflow:visible}}.segment{{position:absolute;top:5px;height:20px;color:#fff;font:10px/20px "IBM Plex Mono","Cascadia Mono",monospace;text-align:center;overflow:hidden}}.segment.context{{background:var(--context)}}.segment.future{{background:var(--future)}}.query-marker{{position:absolute;top:-5px;width:2px;height:40px;background:var(--query);z-index:2}}.query-marker i{{position:absolute;top:-14px;left:4px;color:var(--query);font:700 9px "IBM Plex Mono","Cascadia Mono",monospace;font-style:normal}}.window{{background:#fff;border:1px solid var(--line);border-radius:6px;margin:0 0 26px;overflow:hidden}}.window-heading{{display:flex;justify-content:space-between;gap:20px;padding:18px 20px;border-bottom:1px solid var(--line)}}.window-index{{font:700 12px "IBM Plex Mono","Cascadia Mono",monospace;color:var(--blue)}}.window h2{{margin:4px 0 0;font:700 22px "DIN Alternate","Bahnschrift",sans-serif}}dl{{display:flex;gap:22px;margin:0}}dl div{{min-width:86px}}dt{{font-size:10px;color:var(--muted);text-transform:uppercase}}dd{{margin:5px 0 0;font:700 12px "IBM Plex Mono","Cascadia Mono",monospace}}.frame-strip{{display:grid;grid-template-columns:repeat(24,minmax(36px,1fr));overflow-x:auto;border-bottom:1px solid var(--line)}}.frame-cell{{min-width:42px;padding:7px 4px;text-align:center;color:#fff;font:10px "IBM Plex Mono","Cascadia Mono",monospace;border-right:1px solid rgba(255,255,255,.3)}}.frame-cell.context{{background:var(--context)}}.frame-cell.future{{background:var(--future)}}.frame-cell.query{{box-shadow:inset 0 0 0 4px var(--query)}}.frame-cell strong,.frame-cell span{{display:block}}.frame-cell span{{margin-top:3px;opacity:.8;font-size:8px}}.media-grid{{display:grid;grid-template-columns:2fr 1fr 1fr;gap:1px;background:var(--line)}}figure{{margin:0;background:#fff}}.sheet{{border-top:1px solid var(--line)}}.sheet img{{cursor:zoom-in}}.foot{{max-width:1440px;margin:0 auto;padding:0 24px 30px;color:var(--muted);font-size:12px}}code{{font-family:"IBM Plex Mono","Cascadia Mono",monospace}}a:focus-visible,video:focus-visible{{outline:3px solid var(--query);outline-offset:2px}}
@media(max-width:850px){{main{{padding:14px}}.overview{{grid-template-columns:1fr}}.window-map{{padding:18px 14px}}.timeline-ruler,.timeline-row{{grid-template-columns:76px 1fr;gap:8px}}.timeline-name{{display:block}}.timeline-name span{{display:block}}.window-heading{{display:block}}dl{{margin-top:14px;gap:12px;justify-content:space-between}}.media-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header class="top"><span class="kicker">0613 PyBullet · raw-to-window audit</span><h1>{html.escape(case_id)}</h1><p>{html.escape(str(raw_meta.get('title', '')))} · {raw_frame_count} raw frames at {raw_fps:g} FPS · sampled every 2 frames</p></header>
<main>
  <section class="overview">
    <figure class="raw-video"><figcaption>Original raw video · F00-F{raw_frame_count - 1:02d}</figcaption><video controls muted loop preload="metadata" src="raw_video.mp4"></video></figure>
    <section class="facts"><h2>Object slots</h2><table><thead><tr><th>Slot</th><th>Instance</th><th>Shape</th><th>Role</th></tr></thead><tbody>{''.join(object_rows)}</tbody></table><p class="prompt">{html.escape(prompt)}</p></section>
  </section>
  <section class="window-map"><div class="window-map-inner"><h2>One raw sequence, three overlapping training windows</h2><div class="legend"><span class="lc">8 context steps</span><span class="lf">16 future steps</span><span class="lq">local F04 query</span></div>{window_timeline_html(records, raw_frame_count)}<p class="prompt">Windowed stride-{frame_stride} frames: F{first_used:02d}, F{first_used + frame_stride:02d}, ..., F{last_used:02d}. Remaining eligible samples {remaining_label} cannot form another 24-step window; odd raw frames are skipped by frame_stride={frame_stride}.</p></div></section>
  {''.join(window_sections)}
</main>
<footer class="foot">Projected boxes come from simulator position, orientation, object size, and camera geometry. They identify object slots but do not encode pixel-level occlusion.</footer>
</body></html>"""


def main() -> None:
    args = parse_args()
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    raw_case = find_raw_case(args.raw_root.resolve(), args.split, args.case_id)
    raw_meta = load_json(raw_case / "meta.json")
    objects = raw_meta.get("objects", [])
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"case has no object metadata: {raw_case}")
    source_video = raw_case / "video.mp4"
    capture = cv2.VideoCapture(str(source_video))
    raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if raw_frame_count <= 0 or raw_fps <= 0:
        raise RuntimeError(f"could not inspect raw video: {source_video}")

    output_dir = args.output_root.resolve() / args.case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_video, output_dir / "raw_video.mp4")
    records: list[dict] = []
    for episode_json in episode_paths(
        args.episode_root.resolve(), args.split, args.case_id
    ):
        episode_meta = load_json(episode_json)
        name = episode_json.stem
        window_dir = output_dir / name
        window_dir.mkdir(parents=True, exist_ok=True)
        with np.load(episode_json.with_suffix(".npz"), allow_pickle=False) as arrays:
            frames_chw = np.asarray(arrays["full_frames"], dtype=np.float32)
            boxes_t_o4 = np.asarray(arrays["full_boxes"], dtype=np.float32)
        context_steps = 8
        future_steps = int(frames_chw.shape[0]) - context_steps
        if (context_steps, future_steps) != (8, 16):
            raise ValueError(
                f"unexpected window shape for {name}: {context_steps}/{future_steps}"
            )
        frame_stride = int(episode_meta["frame_stride"])
        window_start = int(episode_meta["window_start"])
        raw_indices = [
            frame_stride * (window_start + local_step)
            for local_step in range(frames_chw.shape[0])
        ]
        source_rows = [
            (
                frame_rgb_uint8(frames_chw[local_step]),
                local_step,
                raw_indices[local_step],
                local_step < context_steps,
            )
            for local_step in range(frames_chw.shape[0])
        ]
        video_frames = [
            render_video_frame(
                frame,
                boxes_t_o4[local_step],
                objects,
                window_name=name,
                local_step=local_step,
                raw_frame=raw_frame,
                context_steps=context_steps,
                query_local_frame=args.query_local_frame,
            )
            for frame, local_step, raw_frame, _ in source_rows
        ]
        targets = {
            "full.mp4": video_frames,
            "context.mp4": video_frames[:context_steps],
            "future.mp4": video_frames[context_steps:],
        }
        for filename, selected_frames in targets.items():
            path = window_dir / filename
            if args.overwrite or not path.is_file():
                write_h264_video(path, selected_frames, args.fps, ffmpeg)
        sheet_path = window_dir / "contact_sheet.jpg"
        if args.overwrite or not sheet_path.is_file():
            sheet = contact_sheet(
                source_rows,
                boxes_t_o4,
                objects,
                context_steps=context_steps,
                query_local_frame=args.query_local_frame,
            )
            if not cv2.imwrite(
                str(sheet_path),
                cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            ):
                raise RuntimeError(f"failed to write {sheet_path}")
        records.append(
            {
                "name": name,
                "prompt": str(episode_meta["prompt"]),
                "window_start": window_start,
                "frame_stride": frame_stride,
                "context_steps": context_steps,
                "future_steps": future_steps,
                "query_local_frame": int(args.query_local_frame),
                "raw_indices": raw_indices,
            }
        )

    manifest = {
        "case_id": args.case_id,
        "raw_case": str(raw_case),
        "raw_frame_count": raw_frame_count,
        "raw_fps": raw_fps,
        "windows": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(
        render_html(
            case_id=args.case_id,
            raw_case=raw_case,
            raw_meta=raw_meta,
            records=records,
            raw_frame_count=raw_frame_count,
            raw_fps=raw_fps,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
