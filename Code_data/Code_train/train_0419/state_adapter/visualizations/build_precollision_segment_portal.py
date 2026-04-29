#!/usr/bin/env python3
"""Export a few collision-free / pre-collision segments and visualize them locally."""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GENESIS_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
)
MOVI_ROOT = Path(
    "/data/gaoya/dataset/kubric_tfds_movi-d/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter"
)

STATE_NAMES = ["u", "v", "d", "w", "h", "du", "dv", "dd", "vis"]
SCENARIO_SPECS = [
    {
        "key": "genesis_none_static",
        "label": "Genesis / no-collision / static",
        "dataset": "genesis",
        "collision": "none",
        "motion": "static",
        "segment_kind": "full_no_collision_window",
    },
    {
        "key": "genesis_none_simple",
        "label": "Genesis / no-collision / simple",
        "dataset": "genesis",
        "collision": "none",
        "motion": "simple",
        "segment_kind": "full_no_collision_window",
    },
    {
        "key": "movi_env_only_simple",
        "label": "MOVI-D / env-only / simple pre-collision",
        "dataset": "movi_d",
        "collision": "env_only",
        "motion": "simple",
        "segment_kind": "precollision_segment",
    },
    {
        "key": "movi_env_only_static",
        "label": "MOVI-D / env-only / static pre-collision",
        "dataset": "movi_d",
        "collision": "env_only",
        "motion": "static",
        "segment_kind": "precollision_segment",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genesis_root", type=Path, default=GENESIS_ROOT)
    parser.add_argument("--movi_root", type=Path, default=MOVI_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8117)
    parser.add_argument("--max_per_bucket", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def build_video(frames: list[Image.Image], dst: Path, fps: float = 12.0) -> None:
    ensure_dir(dst.parent)
    with imageio.get_writer(
        str(dst),
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))


def build_gif(frames: list[Image.Image], dst: Path, duration_ms: int = 260) -> None:
    if not frames:
        return
    ensure_dir(dst.parent)
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def annotate_frame(frame: Image.Image, label: str) -> Image.Image:
    image = frame.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 10, 220, 46), radius=8, fill=(0, 0, 0))
    draw.text((18, 18), label, fill=(255, 255, 255))
    return image


def load_annotated_frames(sample_dir: Path, frame_indices: list[int]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for frame_idx in frame_indices:
        frame_path = sample_dir / "rgb" / f"frame_{int(frame_idx):03d}.png"
        with Image.open(frame_path) as image:
            frames.append(annotate_frame(image.copy(), f"frame {int(frame_idx):03d}"))
    return frames


def choose_context_frame_count(context_len: int, future_visible_frames: int) -> int:
    if context_len <= 0:
        return 0
    if context_len < 2:
        return context_len
    if future_visible_frames <= 2:
        return min(context_len, 2)
    # Use as much context as possible while keeping context:future within [1:2, 1:1].
    return min(context_len, future_visible_frames)


def dataset_slug(dataset_name: str) -> str:
    if dataset_name == "movi_d":
        return "movi-d"
    return dataset_name


def infer_split_and_rel_source(sample_dir: Path) -> tuple[str, Path]:
    parts = sample_dir.parts
    for idx, part in enumerate(parts):
        if part in {"train", "test"}:
            tail = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path(sample_dir.name)
            return part, tail
    return "train", Path(sample_dir.name)


def sample_output_dir(output_dir: Path, sample: dict[str, Any]) -> tuple[Path, str, str]:
    sample_dir = Path(str(sample["source_sample_dir"]))
    split, rel_source = infer_split_and_rel_source(sample_dir)
    rel_dir = Path(split) / dataset_slug(str(sample["dataset"])) / rel_source
    return output_dir / rel_dir, split, rel_dir.as_posix()


def save_main_state_plot(
    state_raw: np.ndarray,
    main_object_index: int,
    frame_indices: list[int],
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    values = np.asarray(state_raw[:, main_object_index, :], dtype=np.float32)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)
    x = np.asarray(frame_indices, dtype=np.int32)
    for idx, name in enumerate(STATE_NAMES):
        ax = axes[idx // 3][idx % 3]
        ax.plot(x, values[:, idx], linewidth=2.0)
        ax.set_title(name)
        ax.grid(alpha=0.25)
    fig.suptitle(f"main object {main_object_index} state")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def first_main_collision_hit(meta: dict[str, Any]) -> int | None:
    wi = meta.get("window_interactions") or {}
    fw = wi.get("future_window") or {}
    future_start = int(fw.get("frame_start", int(meta.get("start_index", 0)) + int(meta.get("context_len", 0))))
    future_end = int(fw.get("frame_end_exclusive", future_start + int(meta.get("future_len", 0))))
    main_idx = int(meta.get("main_object_index", 0))
    first: int | None = None
    for episode in fw.get("episodes", []):
        obj_indices = [int(x) for x in episode.get("object_indices", []) if int(x) >= 0]
        if main_idx not in obj_indices:
            continue
        start_frame = int(episode.get("start_frame", future_end))
        end_frame = int(episode.get("end_frame", start_frame))
        if end_frame < future_start or start_frame >= future_end:
            continue
        hit = future_start if start_frame < future_start else start_frame
        first = hit if first is None else min(first, hit)
    return first


def collect_candidates(dataset_name: str, dataset_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for pair_meta_path in sorted(dataset_root.rglob("pair_meta.json")):
        meta = load_json(pair_meta_path)
        wi = meta.get("window_interactions") or {}
        fw = wi.get("future_window") or {}
        mc = meta.get("motion_complexity") or {}
        collision = str(fw.get("collision_type_bucket", ""))
        motion = str(mc.get("label", ""))
        if collision not in {"none", "env_only"} or motion not in {"static", "simple"}:
            continue
        frame_paths = list(meta.get("x_frame_paths", [])) + list(meta.get("y_frame_paths", []))
        if not frame_paths or any(not Path(str(path)).exists() for path in frame_paths):
            continue
        start_index = int(meta.get("start_index", 0))
        context_len = int(meta.get("context_len", 0))
        future_len = int(meta.get("future_len", 0))
        future_start = start_index + context_len
        future_end = future_start + future_len
        first_hit = first_main_collision_hit(meta)
        if collision == "none":
            segment_end = future_end
            segment_kind = "full_no_collision_window"
        else:
            if first_hit is None:
                segment_end = future_end
                segment_kind = "main_object_clear_full_future"
            else:
                segment_end = first_hit
                segment_kind = "precollision_segment"
        pre_future_frames = segment_end - future_start
        if collision == "env_only" and pre_future_frames < 2 and segment_kind == "precollision_segment":
            continue
        results.append(
            {
                "dataset": dataset_name,
                "window_dir": str(pair_meta_path.parent),
                "source_sample_dir": str(meta.get("source_sample_dir", "")),
                "start_index": start_index,
                "context_len": context_len,
                "future_len": future_len,
                "future_start": future_start,
                "future_end": future_end,
                "segment_end": segment_end,
                "segment_kind": segment_kind,
                "pre_future_frames": pre_future_frames,
                "collision": collision,
                "motion": motion,
                "main_object_index": int(meta.get("main_object_index", 0)),
                "pair_meta": meta,
            }
        )
    return results


def pick_examples(records: list[dict[str, Any]], max_per_bucket: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for spec in SCENARIO_SPECS:
        bucket_records: list[dict[str, Any]] = []
        for item in records:
            if item["dataset"] != spec["dataset"]:
                continue
            if item["collision"] != spec["collision"]:
                continue
            if item["motion"] != spec["motion"]:
                continue
            if item["segment_kind"] != spec["segment_kind"]:
                continue
            bucket_records.append(item)

        best_by_source: dict[str, dict[str, Any]] = {}
        for item in bucket_records:
            source_key = str(item["source_sample_dir"])
            current = best_by_source.get(source_key)
            if current is None:
                best_by_source[source_key] = item
                continue
            current_score = (
                int(current["pre_future_frames"]),
                int(current["future_len"]),
                int(current["segment_end"]) - int(current["start_index"]),
                -int(current["start_index"]),
            )
            item_score = (
                int(item["pre_future_frames"]),
                int(item["future_len"]),
                int(item["segment_end"]) - int(item["start_index"]),
                -int(item["start_index"]),
            )
            if item_score > current_score:
                best_by_source[source_key] = item

        ranked = sorted(
            best_by_source.values(),
            key=lambda item: (
                -int(item["pre_future_frames"]),
                -int(item["future_len"]),
                -(int(item["segment_end"]) - int(item["start_index"])),
                int(item["start_index"]),
                str(item["source_sample_dir"]),
            ),
        )
        for item in ranked[:max_per_bucket]:
            chosen = dict(item)
            chosen["bucket_key"] = spec["key"]
            chosen["bucket_label"] = spec["label"]
            selected.append(chosen)
    return selected


def export_sample(sample: dict[str, Any], output_dir: Path, index: int) -> dict[str, Any]:
    meta = dict(sample["pair_meta"])
    window_dir = Path(str(sample["window_dir"]))
    sample_dir = Path(str(sample["source_sample_dir"]))
    out_dir, split, rel_dir = sample_output_dir(output_dir, sample)
    ensure_dir(out_dir)

    with np.load(window_dir / "state_pair.npz") as payload:
        state_raw = np.asarray(payload["state_raw"]).astype(np.float32)
        state_norm = np.asarray(payload["state_norm"]).astype(np.float32)
        object_ids = np.asarray(payload["object_ids"]).astype(np.int32)
        seg_ids = np.asarray(payload["seg_ids"]).astype(np.int32)
        visibility_mask = np.asarray(payload["visibility_mask"]).astype(np.uint8)

    original_context_len = int(sample["context_len"])
    future_start = int(sample["future_start"])
    segment_end = int(sample["segment_end"])
    future_frame_indices = list(range(future_start, segment_end))
    future_visible_frames = len(future_frame_indices)
    context_frame_count = choose_context_frame_count(original_context_len, future_visible_frames)
    context_start = future_start - context_frame_count
    context_frame_indices = list(range(context_start, future_start))
    frame_indices = context_frame_indices + future_frame_indices

    full_frames = load_annotated_frames(sample_dir, frame_indices)
    context_frames = full_frames[: len(context_frame_indices)]
    future_frames = full_frames[len(context_frame_indices) :]

    build_video(full_frames, out_dir / "segment.mp4")
    build_gif(full_frames, out_dir / "full.gif")
    build_gif(context_frames, out_dir / "context.gif")
    build_gif(future_frames, out_dir / "future.gif")

    if full_frames:
        full_frames[0].save(out_dir / "full.png")
    if context_frames:
        context_frames[0].save(out_dir / "context.png")
    if future_frames:
        future_frames[0].save(out_dir / "future.png")

    segment_state_raw = state_raw[frame_indices[0] : frame_indices[-1] + 1].astype(np.float32)
    segment_state_norm = state_norm[frame_indices[0] : frame_indices[-1] + 1].astype(np.float32)
    segment_visibility = visibility_mask[frame_indices[0] : frame_indices[-1] + 1].astype(np.uint8)
    np.savez_compressed(
        out_dir / "segment_state.npz",
        object_ids=object_ids,
        seg_ids=seg_ids,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        context_frame_indices=np.asarray(context_frame_indices, dtype=np.int32),
        future_frame_indices=np.asarray(future_frame_indices, dtype=np.int32),
        state_raw=segment_state_raw,
        state_norm=segment_state_norm,
        visibility_mask=segment_visibility,
    )
    save_main_state_plot(
        state_raw=segment_state_raw,
        main_object_index=int(sample["main_object_index"]),
        frame_indices=frame_indices,
        out_path=out_dir / "main_object_state.png",
    )

    info = {
        "dataset": sample["dataset"],
        "split": split,
        "dataset_slug": dataset_slug(str(sample["dataset"])),
        "bucket_key": sample.get("bucket_key", ""),
        "bucket_label": sample.get("bucket_label", ""),
        "source_sample_dir": str(sample_dir),
        "export_dir": str(out_dir),
        "export_rel_dir": rel_dir,
        "window_dir": str(window_dir),
        "segment_kind": str(sample["segment_kind"]),
        "motion_complexity": str(sample["motion"]),
        "future_collision_type_bucket": str(sample["collision"]),
        "start_index": int(sample["start_index"]),
        "context_len": int(sample["context_len"]),
        "future_len": int(sample["future_len"]),
        "future_start": int(sample["future_start"]),
        "future_end": int(sample["future_end"]),
        "segment_end": int(sample["segment_end"]),
        "original_context_len": original_context_len,
        "selected_context_frames": len(context_frame_indices),
        "selected_future_frames": len(future_frame_indices),
        "selected_full_frames": len(frame_indices),
        "context_start": context_start,
        "segment_total_frames": len(frame_indices),
        "pre_collision_future_frames": int(future_visible_frames),
        "main_object_index": int(sample["main_object_index"]),
        "context_frame_indices": context_frame_indices,
        "future_frame_indices": future_frame_indices,
        "full_frame_indices": frame_indices,
        "objects": meta.get("objects", []),
        "window_interactions": meta.get("window_interactions"),
    }
    (out_dir / "segment_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    title = f"{dataset_slug(str(sample['dataset']))} / {sample_dir.name}"
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: #ece5dc;
      color: #1b1713;
    }}
    .card {{
      max-width: 1200px;
      margin: 0 auto;
      background: #fffaf3;
      border: 1px solid #d8cbb9;
      border-radius: 20px;
      padding: 22px;
    }}
    video, img {{ width: 100%; border-radius: 14px; border: 1px solid #d8cbb9; }}
    .grid {{ display:grid; grid-template-columns: 1.1fr .9fr; gap:18px; }}
    .tags {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
    .tag {{ border:1px solid #d8cbb9; border-radius:999px; padding:4px 10px; background:#fff; font-size:12px; }}
    code {{ font-family: ui-monospace, monospace; font-size:12px; }}
    @media (max-width: 1000px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{html_escape(title)}</h1>
    <div class="tags">
      <span class="tag">{html_escape(sample['dataset'])}</span>
      <span class="tag">{html_escape(sample['segment_kind'])}</span>
      <span class="tag">{html_escape(sample['collision'])}</span>
      <span class="tag">{html_escape(sample['motion'])}</span>
      <span class="tag">ctx={len(context_frame_indices)}</span>
      <span class="tag">fut={len(future_frame_indices)}</span>
      <span class="tag">full={len(frame_indices)}</span>
    </div>
    <div class="grid">
      <div>
        <video controls muted preload="metadata" src="segment.mp4"></video>
        <p><a href="segment.mp4">segment.mp4</a> | <a href="full.gif">full.gif</a> | <a href="context.gif">context.gif</a> | <a href="future.gif">future.gif</a></p>
        <p><a href="segment_state.npz">segment_state.npz</a> | <a href="segment_info.json">segment_info.json</a></p>
      </div>
      <div>
        <img src="main_object_state.png" alt="main object state">
      </div>
    </div>
    <div class="grid" style="margin-top:16px;">
      <div>
        <img src="context.gif" alt="context gif">
        <p><strong>context</strong> <code>{len(context_frame_indices)} frames</code></p>
      </div>
      <div>
        <img src="future.gif" alt="future gif">
        <p><strong>future</strong> <code>{len(future_frame_indices)} frames</code></p>
      </div>
    </div>
    <p><strong>source_sample_dir</strong>: <code>{html_escape(str(sample_dir))}</code></p>
    <p><strong>window_dir</strong>: <code>{html_escape(str(window_dir))}</code></p>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")

    return {
        "title": title,
        "split": split,
        "dataset": sample["dataset"],
        "dataset_slug": dataset_slug(str(sample["dataset"])),
        "segment_kind": sample["segment_kind"],
        "bucket_key": sample.get("bucket_key", ""),
        "bucket_label": sample.get("bucket_label", ""),
        "motion": sample["motion"],
        "collision": sample["collision"],
        "rel_dir": rel_dir,
        "frames": len(frame_indices),
        "context_frames": len(context_frame_indices),
        "future_frames": len(future_frame_indices),
        "pre_collision_future_frames": future_visible_frames,
        "source_sample_dir": str(sample_dir),
    }


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_index(cards: list[dict[str, Any]]) -> str:
    groups: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        groups.setdefault(str(card.get("bucket_label", "Other")), []).append(card)

    section_parts: list[str] = []
    for bucket_label, bucket_cards in groups.items():
        card_html = "".join(
            f"""
<article class="sample-card">
  <div class="media-strip">
    <div class="media-panel">
      <div class="media-label">Context</div>
      <img loading="lazy" src="{html_escape(card['rel_dir'])}/context.gif" alt="{html_escape(card['title'])} context">
    </div>
    <div class="media-panel">
      <div class="media-label">Future</div>
      <img loading="lazy" src="{html_escape(card['rel_dir'])}/future.gif" alt="{html_escape(card['title'])} future">
    </div>
    <div class="media-panel">
      <div class="media-label">Full</div>
      <img loading="lazy" src="{html_escape(card['rel_dir'])}/full.gif" alt="{html_escape(card['title'])} full">
    </div>
  </div>
  <div class="sample-body">
    <h3>{html_escape(card['title'])}</h3>
    <div class="badge-row">
      <span class="badge">{html_escape(card['dataset'])}</span>
      <span class="badge">{html_escape(card['segment_kind'])}</span>
      <span class="badge">{html_escape(card['collision'])}</span>
      <span class="badge">{html_escape(card['motion'])}</span>
      <span class="badge">ctx {card['context_frames']}</span>
      <span class="badge">fut {card['future_frames']}</span>
      <span class="badge">full {card['frames']}</span>
    </div>
    <p class="meta-line"><strong>source</strong>: <code>{html_escape(card['source_sample_dir'])}</code></p>
    <p><a class="button" href="{html_escape(card['rel_dir'])}/index.html">详情页</a></p>
  </div>
</article>
"""
            for card in bucket_cards
        )
        section_parts.append(
            f"""
<section class="bucket-section">
  <div class="bucket-head">
    <h2>{html_escape(bucket_label)}</h2>
    <span class="bucket-count">{len(bucket_cards)} cases</span>
  </div>
  <div class="sample-grid">
    {card_html}
  </div>
</section>
"""
        )
    sections_html = "".join(section_parts)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pre-Collision Segment Preview</title>
  <style>
    :root {{
      --bg: #ece5dc;
      --panel: #fffaf3;
      --panel2: #f7efe2;
      --ink: #1b1713;
      --muted: #6a6258;
      --line: #d8cbb9;
      --accent: #7c2d12;
      --accent2: #0f766e;
      --shadow: rgba(43, 30, 20, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(124,45,18,0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(15,118,110,0.10), transparent 24%),
        var(--bg);
    }}
    .page {{ max-width: 1500px; margin: 0 auto; padding: 28px 22px 60px; }}
    .hero, .section {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 42px var(--shadow);
    }}
    .hero {{ padding: 28px 30px; margin-bottom: 22px; }}
    .hero h1, .section h2, .sample-body h3 {{ margin: 0; }}
    .hero p, .muted {{ color: var(--muted); }}
    .bucket-section {{
      margin-top: 16px;
      padding: 14px;
      border-top: 1px solid rgba(216,203,185,0.7);
    }}
    .bucket-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .bucket-count {{
      color: var(--muted);
      font-size: 13px;
    }}
    .sample-grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .sample-card {{
      display: grid;
      grid-template-columns: minmax(740px, 1.3fr) minmax(280px, 0.7fr);
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.46);
    }}
    .media-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      align-items: start;
    }}
    .media-panel {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px;
      background: rgba(255,255,255,0.58);
    }}
    .media-label {{
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .media-panel img {{ width: 100%; display: block; border-radius: 10px; background: #0d0f13; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 12px; }}
    .badge {{
      background: rgba(255,255,255,0.78);
      color: #694d33;
      border: 1px solid #dcc7aa;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
    }}
    .button {{
      display: inline-block;
      text-decoration: none;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 14px;
    }}
    code {{ font-family: ui-monospace, monospace; font-size: 12px; }}
    @media (max-width: 1080px) {{
      .sample-card {{ grid-template-columns: 1fr; }}
      .media-strip {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Pre-Collision / Collision-Free Segment Preview</h1>
      <p>导出目录已经组织成 <code>stage1adapter/train|test/(genesis|movi-d)/raw_like_path</code>。每个 case 同时展示 <code>context.gif</code>、<code>future.gif</code>、<code>full.gif</code>，其中 future 截到主物体第一次碰撞之前。</p>
    </section>
    <section class="section">
      {sections_html}
    </section>
  </div>
</body>
</html>
"""


def start_server(output_dir: Path, host: str, port: int) -> tuple[int, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    process = subprocess.Popen(
        [
            "python3",
            "-m",
            "http.server",
            str(port),
            "--bind",
            host,
            "--directory",
            str(output_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid, f"http://{host}:{port}/index.html"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    for split in ("train", "test"):
        for dataset_name in ("genesis", "movi-d"):
            ensure_dir(output_dir / split / dataset_name)

    candidates = collect_candidates("genesis", args.genesis_root.resolve()) + collect_candidates(
        "movi_d", args.movi_root.resolve()
    )
    selected = pick_examples(candidates, max_per_bucket=int(args.max_per_bucket))
    cards = [export_sample(sample, output_dir, idx) for idx, sample in enumerate(selected)]
    (output_dir / "index.html").write_text(build_index(cards), encoding="utf-8")
    manifest = {"selected": cards}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pid, url = start_server(output_dir, str(args.host), int(args.port))
    print(f"output_dir={output_dir}")
    print(f"pid={pid}")
    print(f"url={url}")


if __name__ == "__main__":
    main()
