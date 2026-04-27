#!/usr/bin/env python3
"""Visualize mixed training samples and context strategies over a local HTTP page."""

from __future__ import annotations

import argparse
import html
import io
import json
import mimetypes
import os
import random
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

TRAIN_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
OUTPUT_ROOT = TRAIN_ROOT / "visualizations" / "_mixed_train_context_samples"

if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from dataset import (
    GenesisRigidDataset,
    MoviDTFRecordDataset,
    OpenVidParquetDataset,
    _decode_video_bytes,
    _decode_video_path,
)

HEIGHT = 384
WIDTH = 672
NUM_FRAMES = 24
MAX_PIXELS = 1024 * 1024
OPENVID_ROOT = "/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train"
MOVID_ROOT = "/data/gaoya/dataset/kubric_tfds_movi-d"
GENESIS_ROOT = "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"

CONTEXT_REFERENCE_FRAMES = 49
CONTEXT_REFERENCE_PREFIXES = (1, 4, 8, 12, 16)
PORT = 8099


@dataclass
class VisualCase:
    case_id: str
    dataset: str
    strategy: str
    sample_ref: str
    prompt: str
    frame_count: int
    context_indices: list[int]
    source_paths: dict[str, str]
    media_paths: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize mixed-dataset training samples with context-frame strategies."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where the generated HTML and media files are written.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild visualization assets even if they already exist.",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Only build the visualization files without starting the HTTP server.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_video(path: Path, frames: list[Image.Image], fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB"), dtype=np.uint8))


def scaled_reference_counts(total_frames: int) -> list[int]:
    counts: list[int] = []
    for ref_count in CONTEXT_REFERENCE_PREFIXES:
        count = max(1, min((total_frames * ref_count + CONTEXT_REFERENCE_FRAMES - 1) // CONTEXT_REFERENCE_FRAMES, total_frames - 1))
        if count not in counts:
            counts.append(count)
    return counts or [1]


def sparse_indices(total_frames: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    positions: list[int] = []
    for idx in range(count):
        frame_index = round(idx * (total_frames - 1) / (count - 1))
        if not positions or frame_index != positions[-1]:
            positions.append(frame_index)
    if positions[-1] != total_frames - 1:
        positions[-1] = total_frames - 1
    cursor = 1
    while len(positions) < count and cursor < total_frames - 1:
        if cursor not in positions:
            positions.insert(-1, cursor)
        cursor += 1
    return sorted(positions[:count])


def select_context_indices(mode: str, total_frames: int, rng: random.Random) -> list[int]:
    if total_frames < 2:
        return []
    counts = scaled_reference_counts(total_frames)
    multiframe_counts = [count for count in counts if count > 1] or [min(total_frames - 1, 2)]

    if mode == "text_only":
        return []
    if mode == "first_frame":
        return [0]
    if mode == "prefix":
        count = rng.choice(multiframe_counts)
        return list(range(count))
    if mode == "sparse":
        count = rng.choice(multiframe_counts)
        return sparse_indices(total_frames, count)
    if mode == "random":
        count = rng.choice(multiframe_counts)
        if count <= 1:
            return [0]
        extra = sorted(rng.sample(range(1, total_frames), count - 1))
        return [0, *extra]
    raise ValueError(f"Unsupported context mode: {mode}")


def build_strip_image(frames: list[Image.Image], context_indices: list[int], output_path: Path) -> None:
    thumb_w = 112
    thumb_h = 64
    gap = 8
    pad = 12
    width = pad * 2 + len(frames) * thumb_w + max(0, len(frames) - 1) * gap
    height = pad * 2 + thumb_h + 28
    canvas = Image.new("RGB", (width, height), (247, 244, 236))
    draw = ImageDraw.Draw(canvas)
    context_set = set(context_indices)

    for idx, frame in enumerate(frames):
        thumb = frame.convert("RGB").resize((thumb_w, thumb_h))
        x = pad + idx * (thumb_w + gap)
        y = pad
        canvas.paste(thumb, (x, y))
        border_color = (27, 120, 66) if idx in context_set else (90, 90, 90)
        border_width = 4 if idx in context_set else 2
        draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=border_color, width=border_width)
        draw.text((x, y + thumb_h + 6), f"{idx:02d}", fill=border_color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def target_indices_from_context(frame_count: int, context_indices: list[int]) -> list[int]:
    context_set = set(context_indices)
    return [idx for idx in range(frame_count) if idx not in context_set]


def deterministic_decode_openvid(dataset: OpenVidParquetDataset, row_index: int, seed: int) -> tuple[list[Image.Image], str, dict[str, str]]:
    info, raw_video = dataset._read_row(row_index)
    prompt = str(info.get("caption", "")).strip()
    if not prompt:
        raise ValueError("OpenVid sample is missing caption.")
    random_state = random.getstate()
    random.seed(seed)
    try:
        frames = _decode_video_bytes(
            raw_video,
            num_frames=dataset.num_frames,
            frame_processor=dataset.frame_processor,
            require_min_frames=True,
        )
    finally:
        random.setstate(random_state)
    return frames, prompt, {"openvid_row_index": str(row_index)}


def deterministic_decode_movid(dataset: MoviDTFRecordDataset, record_index: int, seed: int) -> tuple[list[Image.Image], str, dict[str, str]]:
    record_ref = dataset.record_refs[record_index]
    example = dataset._parse_example(dataset._read_serialized_record(record_ref))
    features = example.features.feature
    prompt = dataset._build_prompt(features)
    random_state = random.getstate()
    random.seed(seed)
    try:
        frames = dataset._decode_frames(features["video"].bytes_list.value)
    finally:
        random.setstate(random_state)
    return frames, prompt, {
        "split": str(record_ref["split"]),
        "shard_path": str(record_ref["shard_path"]),
        "record_index": str(record_ref["record_index"]),
    }


def deterministic_decode_genesis(dataset: GenesisRigidDataset, entry_index: int) -> tuple[list[Image.Image], str, dict[str, str]]:
    entry = dataset.entries[entry_index]
    prompt = str(entry["prompt"]).strip()
    if not prompt:
        raise ValueError("Genesis sample is missing prompt.")
    frames = _decode_video_path(
        entry["video_path"],
        num_frames=dataset.num_frames,
        frame_processor=dataset.frame_processor,
    )
    return frames, prompt, {
        "sample_dir": str(entry["sample_dir"]),
        "video_path": str(entry["video_path"]),
        "object_id": str(entry.get("object_id", "")),
    }


def pick_openvid_cases() -> list[tuple[str, str, list[Image.Image], str, dict[str, str]]]:
    dataset = OpenVidParquetDataset(
        dataset_base_path=OPENVID_ROOT,
        dataset_repeat=1,
        max_pixels=MAX_PIXELS,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )
    modes = [("openvid_case_1", "prefix"), ("openvid_case_2", "first_frame")]
    indices = list(range(dataset.total_rows))
    random.Random(20260430).shuffle(indices)
    results = []
    cursor = 0
    for case_id, mode in modes:
        while cursor < len(indices):
            row_index = indices[cursor]
            cursor += 1
            try:
                frames, prompt, source = deterministic_decode_openvid(dataset, row_index, seed=7000 + row_index)
                results.append((case_id, mode, frames, prompt, source))
                break
            except Exception:
                continue
    if len(results) != len(modes):
        raise RuntimeError("Failed to pick enough OpenVid cases for visualization.")
    return results


def pick_movid_cases() -> list[tuple[str, str, list[Image.Image], str, dict[str, str]]]:
    dataset = MoviDTFRecordDataset(
        dataset_base_path=MOVID_ROOT,
        split="train",
        dataset_repeat=1,
        max_pixels=MAX_PIXELS,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )
    modes = [("movi_d_case_1", "sparse"), ("movi_d_case_2", "random")]
    indices = list(range(len(dataset.record_refs)))
    random.Random(20260431).shuffle(indices)
    results = []
    cursor = 0
    for case_id, mode in modes:
        while cursor < len(indices):
            record_index = indices[cursor]
            cursor += 1
            try:
                frames, prompt, source = deterministic_decode_movid(dataset, record_index, seed=9000 + record_index)
                results.append((case_id, mode, frames, prompt, source))
                break
            except Exception:
                continue
    if len(results) != len(modes):
        raise RuntimeError("Failed to pick enough MOVI-D cases for visualization.")
    return results


def pick_genesis_cases() -> list[tuple[str, str, list[Image.Image], str, dict[str, str]]]:
    dataset = GenesisRigidDataset(
        dataset_base_path=GENESIS_ROOT,
        split="train",
        dataset_repeat=1,
        max_pixels=MAX_PIXELS,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )
    modes = [("genesis_case_1", "text_only"), ("genesis_case_2", "prefix")]
    indices = list(range(len(dataset.entries)))
    random.Random(20260432).shuffle(indices)
    results = []
    cursor = 0
    for case_id, mode in modes:
        while cursor < len(indices):
            entry_index = indices[cursor]
            cursor += 1
            try:
                frames, prompt, source = deterministic_decode_genesis(dataset, entry_index)
                results.append((case_id, mode, frames, prompt, source))
                break
            except Exception:
                continue
    if len(results) != len(modes):
        raise RuntimeError("Failed to pick enough Genesis cases for visualization.")
    return results


def build_cases(output_root: Path, rebuild: bool) -> list[VisualCase]:
    assets_dir = output_root / "assets"
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not rebuild:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [VisualCase(**item) for item in payload["cases"]]

    raw_cases = [
        ("OpenVid", *item) for item in pick_openvid_cases()
    ] + [
        ("MOVI-D", *item) for item in pick_movid_cases()
    ] + [
        ("GenesisRigid", *item) for item in pick_genesis_cases()
    ]

    cases: list[VisualCase] = []
    for dataset_name, case_id, strategy, frames, prompt, source_paths in raw_cases:
        rng = random.Random(100000 + len(cases))
        context_indices = select_context_indices(strategy, len(frames), rng)
        target_indices = target_indices_from_context(len(frames), context_indices)
        full_video_path = assets_dir / f"{case_id}_full.mp4"
        context_video_path = assets_dir / f"{case_id}_context.mp4"
        gt_target_video_path = assets_dir / f"{case_id}_gt_target.mp4"
        strip_path = assets_dir / f"{case_id}_strip.png"
        first_frame_path = assets_dir / f"{case_id}_first.png"

        write_video(full_video_path, frames, fps=8)
        if context_indices:
            write_video(context_video_path, [frames[idx] for idx in context_indices], fps=4)
        if target_indices:
            write_video(gt_target_video_path, [frames[idx] for idx in target_indices], fps=8)
        build_strip_image(frames, context_indices, strip_path)
        frames[0].save(first_frame_path)

        case = VisualCase(
            case_id=case_id,
            dataset=dataset_name,
            strategy=strategy,
            sample_ref=case_id,
            prompt=prompt,
            frame_count=len(frames),
            context_indices=context_indices,
            source_paths=source_paths,
            media_paths={
                "full_video": str(full_video_path),
                "context_video": str(context_video_path) if context_indices else "",
                "gt_target_video": str(gt_target_video_path) if target_indices else "",
                "frame_strip": str(strip_path),
                "first_frame": str(first_frame_path),
            },
        )
        cases.append(case)

    write_json(
        manifest_path,
        {
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "cases": [case.__dict__ for case in cases],
        },
    )
    return cases


def media_url(path: str) -> str:
    return f"/media?path={quote(path, safe='/')}"


def render_source_paths(source_paths: dict[str, str]) -> str:
    if not source_paths:
        return '<div class="muted">No source-path metadata</div>'
    rows = []
    for key, value in sorted(source_paths.items()):
        rows.append(
            f"<div class='kv'><span>{html.escape(key)}</span><code>{html.escape(str(value))}</code></div>"
        )
    return "\n".join(rows)


def render_media_block(case: VisualCase) -> str:
    full_video = case.media_paths["full_video"]
    strip_path = case.media_paths["frame_strip"]
    first_frame = case.media_paths["first_frame"]
    context_video = case.media_paths["context_video"]
    gt_target_video = case.media_paths.get("gt_target_video", full_video)
    context_html = (
        f"<video controls preload='metadata' src='{html.escape(media_url(context_video))}'></video>"
        if context_video
        else "<div class='empty-card'>text-only, no context frames</div>"
    )
    gt_target_html = (
        f"<video controls preload='metadata' src='{html.escape(media_url(gt_target_video))}'></video>"
        if gt_target_video
        else "<div class='empty-card'>all frames are context, no GT target frames</div>"
    )
    return f"""
<div class="media-grid">
  <article class="media-card">
    <div class="media-label">GT full clip</div>
    <video controls preload="metadata" src="{html.escape(media_url(full_video))}"></video>
  </article>
  <article class="media-card">
    <div class="media-label">Context-only clip</div>
    {context_html}
  </article>
  <article class="media-card">
    <div class="media-label">GT target clip</div>
    {gt_target_html}
  </article>
  <article class="media-card">
    <div class="media-label">Frame timeline</div>
    <img src="{html.escape(media_url(strip_path))}" alt="frame strip">
  </article>
  <article class="media-card">
    <div class="media-label">First frame</div>
    <img src="{html.escape(media_url(first_frame))}" alt="first frame">
  </article>
</div>
"""


def render_page(cases: list[VisualCase]) -> str:
    grouped: dict[str, list[VisualCase]] = {}
    for case in cases:
        grouped.setdefault(case.dataset, []).append(case)

    sections = []
    for dataset_name, items in grouped.items():
        cards = []
        for case in items:
            context_indices = ", ".join(str(idx) for idx in case.context_indices) or "None"
            cards.append(
                f"""
<section class="sample-card">
  <div class="sample-head">
    <div>
      <div class="eyebrow">{html.escape(case.dataset)}</div>
      <h2>{html.escape(case.case_id)}</h2>
    </div>
    <div class="strategy-badge">{html.escape(case.strategy)}</div>
  </div>
  <p class="caption">{html.escape(case.prompt)}</p>
  <div class="stats">
    <span class="tag"><strong>frames</strong>: {case.frame_count}</span>
    <span class="tag"><strong>context indices</strong>: {html.escape(context_indices)}</span>
    <span class="tag"><strong>context count</strong>: {len(case.context_indices)}</span>
  </div>
  {render_media_block(case)}
  <details>
    <summary>Source Paths</summary>
    <div class="source-box">
      {render_source_paths(case.source_paths)}
    </div>
  </details>
</section>
"""
            )
        sections.append(
            f"""
<section class="dataset-section">
  <div class="dataset-header">
    <h1>{html.escape(dataset_name)}</h1>
    <p>2 sampled training cases</p>
  </div>
  {"".join(cards)}
</section>
"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mixed Train Context Visualizer</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffaf1;
      --line: #d7ccb6;
      --text: #2c241a;
      --muted: #736758;
      --accent: #116149;
      --accent-soft: #dff1e9;
      --warm: #8d4b24;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(141, 75, 36, 0.08), transparent 28%),
        linear-gradient(180deg, #f7f4ee, #efe7d9 58%, #ebe1d0);
      color: var(--text);
    }}
    header {{
      padding: 28px 34px 16px;
      border-bottom: 1px solid rgba(44, 36, 26, 0.12);
      background: rgba(255, 250, 241, 0.72);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      letter-spacing: 0.02em;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
    }}
    .summary {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .summary .tag {{
      background: var(--accent-soft);
      border-color: rgba(17, 97, 73, 0.2);
    }}
    main {{
      padding: 24px 24px 48px;
      display: grid;
      gap: 28px;
    }}
    .dataset-section {{
      display: grid;
      gap: 18px;
    }}
    .dataset-header {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      padding: 0 8px;
    }}
    .dataset-header h1 {{
      margin: 0;
      font-size: 28px;
    }}
    .dataset-header p {{
      margin: 0;
      color: var(--warm);
    }}
    .sample-card {{
      background: rgba(255, 250, 241, 0.94);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 16px 34px rgba(99, 79, 55, 0.08);
    }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 16px;
    }}
    .sample-head h2 {{
      margin: 4px 0 0;
      font-size: 24px;
    }}
    .eyebrow {{
      color: var(--warm);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.12em;
    }}
    .strategy-badge {{
      padding: 10px 14px;
      border-radius: 999px;
      background: #1e4e3b;
      color: white;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .caption {{
      font-size: 17px;
      line-height: 1.45;
      color: #34291e;
      margin: 14px 0 14px;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }}
    .tag {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 8px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      font-size: 14px;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .media-card {{
      background: #fff;
      border: 1px solid rgba(44, 36, 26, 0.11);
      border-radius: 18px;
      padding: 14px;
      min-height: 100%;
    }}
    .media-label {{
      font-size: 13px;
      color: var(--warm);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .media-card video,
    .media-card img {{
      width: 100%;
      border-radius: 12px;
      background: #d9d2c5;
      display: block;
    }}
    .empty-card {{
      min-height: 180px;
      display: grid;
      place-items: center;
      border-radius: 12px;
      background: linear-gradient(135deg, #ece6da, #ded6c7);
      color: var(--muted);
      font-style: italic;
    }}
    details {{
      margin-top: 16px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 600;
    }}
    .source-box {{
      margin-top: 12px;
      display: grid;
      gap: 8px;
      background: #fff;
      border: 1px solid rgba(44, 36, 26, 0.09);
      border-radius: 14px;
      padding: 12px;
    }}
    .kv {{
      display: grid;
      gap: 4px;
    }}
    .kv span {{
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      word-break: break-all;
      color: #4b4034;
    }}
    @media (max-width: 980px) {{
      .media-grid {{
        grid-template-columns: 1fr;
      }}
      header {{
        position: static;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Mixed Train Context Visualizer</h1>
    <p>OpenVid / MOVI-D / Genesis 各 2 个训练样本，可视化 5 种 context 策略各至少 1 个示例。</p>
    <div class="summary">
      <span class="tag"><strong>train size</strong>: {HEIGHT}x{WIDTH}</span>
      <span class="tag"><strong>target frames</strong>: {NUM_FRAMES}</span>
      <span class="tag"><strong>strategies</strong>: prefix, first_frame, sparse, random, text_only</span>
    </div>
  </header>
  <main>
    {"".join(sections)}
  </main>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    root_dir: Path

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/media":
            self._serve_media(parsed)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _serve_index(self):
        index_path = self.root_dir / "index.html"
        if not index_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "index.html missing")
            return
        data = index_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_media(self, parsed):
        query = parse_qs(parsed.query)
        raw_path = query.get("path", [None])[0]
        if not raw_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
            return
        target = Path(unquote(raw_path)).expanduser().resolve()
        try:
            target.relative_to(self.root_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Path escapes root")
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        mime_type, _ = mimetypes.guess_type(str(target))
        if mime_type is None:
            mime_type = "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def build_site(output_root: Path, rebuild: bool) -> list[VisualCase]:
    output_root.mkdir(parents=True, exist_ok=True)
    cases = build_cases(output_root, rebuild=rebuild)
    if any("gt_target_video" not in case.media_paths for case in cases):
        cases = build_cases(output_root, rebuild=True)
    index_path = output_root / "index.html"
    index_path.write_text(render_page(cases), encoding="utf-8")
    return cases


def serve(output_root: Path, host: str, port: int) -> None:
    handler_cls = type(
        "MixedTrainContextViewerHandler",
        (ViewerHandler,),
        {"root_dir": output_root.resolve()},
    )
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"Serving mixed training sample viewer on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    cases = build_site(args.output_root, rebuild=args.rebuild)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "num_cases": len(cases),
                "datasets": {
                    dataset: sum(1 for case in cases if case.dataset == dataset)
                    for dataset in sorted({case.dataset for case in cases})
                },
                "strategies": [case.strategy for case in cases],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not args.no_serve:
        serve(args.output_root, args.host, args.port)


if __name__ == "__main__":
    main()
