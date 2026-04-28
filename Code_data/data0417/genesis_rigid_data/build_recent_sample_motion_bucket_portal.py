#!/usr/bin/env python3
"""Build a compact GIF portal for recently generated rigid samples grouped by motion buckets."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/recent_motion_bucket_preview")
MOTION_LABELS = ("static", "simple", "moderate", "complex")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact GIF preview page for recent rigid samples.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modified_within_hours", type=float, default=72.0)
    parser.add_argument("--max_recent_samples", type=int, default=1200)
    parser.add_argument("--samples_per_bucket", type=int, default=24)
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument("--max_gif_side", type=int, default=320)
    parser.add_argument("--gif_duration_ms", type=int, default=120)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _collision_type_bucket(events: Sequence[Dict[str, Any]]) -> str:
    obj_obj = sum(1 for event in events if -1 not in list(event.get("participants", [])))
    obj_env = sum(1 for event in events if -1 in list(event.get("participants", [])))
    if obj_obj == 0 and obj_env == 0:
        return "none"
    if obj_obj == 0:
        return "env_only"
    if obj_env == 0:
        return "obj_obj_only"
    return "mixed"


def _collision_count_bucket(obj_obj_count: int) -> str:
    if obj_obj_count <= 0:
        return "c0"
    if obj_obj_count == 1:
        return "c1"
    return "c2plus"


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(np.float32), q))


def infer_sample_motion_complexity(
    linear_vel: np.ndarray,
    visibility_mask: np.ndarray,
    obj_obj_event_count: int,
) -> Dict[str, Any]:
    vel = np.asarray(linear_vel, dtype=np.float32)
    vis = np.asarray(visibility_mask).astype(bool)
    speed = np.linalg.norm(vel, axis=-1)
    valid_speed = speed[vis]

    pair_visible = vis[1:] & vis[:-1] if vel.shape[0] > 1 else np.zeros((0,) + vis.shape[1:], dtype=bool)
    accel = np.diff(vel, axis=0) if vel.shape[0] > 1 else np.zeros((0,) + vel.shape[1:], dtype=np.float32)
    accel_mag = np.linalg.norm(accel, axis=-1) if accel.size > 0 else np.zeros(pair_visible.shape, dtype=np.float32)
    valid_accel = accel_mag[pair_visible] if accel_mag.size > 0 else np.zeros((0,), dtype=np.float32)

    moving_threshold = 0.03
    moving_mask = vis & (speed >= moving_threshold)
    moving_frame_ratio = float(moving_mask.sum()) / float(max(int(vis.sum()), 1))
    moving_object_count = int(np.sum(np.any(moving_mask, axis=0)))

    speed_mean = float(valid_speed.mean()) if valid_speed.size else 0.0
    speed_p90 = _safe_percentile(valid_speed, 90.0)
    accel_p90 = _safe_percentile(valid_accel, 90.0)

    speed_score = min(speed_p90 / 4.0, 1.0)
    accel_score = min(accel_p90 / 3.0, 1.0)
    moving_ratio_score = min(moving_frame_ratio / 0.75, 1.0)
    moving_object_score = min(max(moving_object_count - 1, 0) / 2.0, 1.0)
    collision_score = min(float(obj_obj_event_count) / 2.0, 1.0)
    complexity_score = float(
        0.35 * speed_score
        + 0.25 * accel_score
        + 0.15 * moving_ratio_score
        + 0.15 * moving_object_score
        + 0.10 * collision_score
    )

    if speed_p90 < 0.02 and moving_frame_ratio < 0.08:
        label = "static"
    elif complexity_score < 0.28 and moving_object_count <= 1 and obj_obj_event_count == 0:
        label = "simple"
    elif complexity_score < 0.68 and moving_object_count <= 2:
        label = "moderate"
    else:
        label = "complex"

    return {
        "label": label,
        "score": float(complexity_score),
        "metrics": {
            "speed_mean": float(speed_mean),
            "speed_p90": float(speed_p90),
            "accel_p90": float(accel_p90),
            "moving_frame_ratio": float(moving_frame_ratio),
            "moving_object_count": int(moving_object_count),
        },
    }


def iter_recent_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    now = time.time()
    cutoff = now - float(args.modified_within_hours) * 3600.0
    records: List[Dict[str, Any]] = []
    for meta_path in sorted(args.dataset_root.glob("*/*/*/metadata.json")):
        sample_dir = meta_path.parent
        if float(meta_path.stat().st_mtime) < cutoff:
            continue
        if args.sample_filter and args.sample_filter not in str(sample_dir):
            continue
        kin_path = sample_dir / "physics" / "rigid_kinematics.npz"
        anchor_path = sample_dir / "physics" / "anchor_targets.npz"
        event_path = sample_dir / "physics" / "collision_events.json"
        rgb_dir = sample_dir / "rgb"
        video_path = sample_dir / "videos" / "rgb.mp4"
        if not (kin_path.exists() and anchor_path.exists() and event_path.exists() and rgb_dir.is_dir() and video_path.exists()):
            continue

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            scene_input = json.loads((sample_dir / "scene_input.json").read_text(encoding="utf-8")) if (sample_dir / "scene_input.json").exists() else {}
            events = json.loads(event_path.read_text(encoding="utf-8"))
            kin = np.load(kin_path)
            anchor = np.load(anchor_path)
        except Exception:
            continue

        linear_vel = np.asarray(kin["linear_vel"], dtype=np.float32)
        visibility_mask = np.asarray(anchor["visibility_mask"]) > 0.5
        obj_obj_events = [event for event in events if -1 not in list(event.get("participants", []))]
        obj_env_events = [event for event in events if -1 in list(event.get("participants", []))]
        motion = infer_sample_motion_complexity(
            linear_vel=linear_vel,
            visibility_mask=visibility_mask,
            obj_obj_event_count=len(obj_obj_events),
        )
        records.append(
            {
                "sample_dir": sample_dir,
                "metadata": metadata,
                "scene_input": scene_input,
                "mtime": float(meta_path.stat().st_mtime),
                "motion_label": str(motion["label"]),
                "motion_score": float(motion["score"]),
                "motion_metrics": dict(motion["metrics"]),
                "num_objects": int(metadata.get("num_objects", linear_vel.shape[1])),
                "count_bucket": str(metadata.get("object_count_bucket", "")),
                "scene_composition": str(metadata.get("scene_composition", "")),
                "sample_role": str(metadata.get("sample_role", "factual")),
                "motion_category": str(metadata.get("motion_category", "")),
                "collision_type_bucket": _collision_type_bucket(events),
                "obj_obj_event_count": int(len(obj_obj_events)),
                "obj_env_event_count": int(len(obj_env_events)),
                "collision_count_bucket": _collision_count_bucket(len(obj_obj_events)),
                "video_path": video_path,
                "rgb_dir": rgb_dir,
            }
        )
    records.sort(key=lambda item: item["mtime"], reverse=True)
    if int(args.max_recent_samples) > 0:
        records = records[: int(args.max_recent_samples)]
    return records


def make_rgb_gif(rgb_dir: Path, dst: Path, max_side: int, duration_ms: int) -> bool:
    frame_paths = sorted(rgb_dir.glob("*.png"))
    if not frame_paths:
        return False
    frames: List[Image.Image] = []
    for idx, frame_path in enumerate(frame_paths):
        frame = Image.open(frame_path).convert("RGB")
        scale = min(max_side / float(frame.width), max_side / float(frame.height), 1.0)
        size = (max(1, int(round(frame.width * scale))), max(1, int(round(frame.height * scale))))
        thumb = frame.resize(size, Image.Resampling.BILINEAR)
        if idx == 0:
            draw = ImageDraw.Draw(thumb)
            draw.rounded_rectangle((8, 8, 128, 34), radius=8, fill=(0, 0, 0, 180))
            draw.text((16, 14), "frame 0", fill=(255, 255, 255))
        frames.append(thumb)
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
    )
    return True


def _fmt_mtime(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _group_counts(records: Iterable[Dict[str, Any]], field: str) -> Dict[str, int]:
    counter = Counter(str(record.get(field, "")) for record in records)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _select_cards(records: List[Dict[str, Any]], samples_per_bucket: int) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["motion_label"]].append(record)
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for label in MOTION_LABELS:
        items = grouped.get(label, [])
        items.sort(
            key=lambda item: (
                -float(item["mtime"]),
                -int(item["num_objects"]),
                -int(item["obj_obj_event_count"]),
                -float(item["motion_score"]),
            )
        )
        selected[label] = items[: int(samples_per_bucket)]
    return selected


def build_html(
    records: List[Dict[str, Any]],
    selected_cards: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> str:
    motion_counts = _group_counts(records, "motion_label")
    count_bucket_counts = _group_counts(records, "count_bucket")
    collision_type_counts = _group_counts(records, "collision_type_bucket")
    role_counts = _group_counts(records, "sample_role")

    summary_chips = "".join(
        f'<span class="chip"><strong>{html.escape(label)}</strong> {motion_counts.get(label, 0)}</span>'
        for label in MOTION_LABELS
    )
    filter_options_motion = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)} ({motion_counts.get(label, 0)})</option>'
        for label in MOTION_LABELS
    )
    filter_options_count = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)} ({count_bucket_counts[label]})</option>'
        for label in count_bucket_counts
        if label
    )
    filter_options_collision = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)} ({collision_type_counts[label]})</option>'
        for label in collision_type_counts
        if label
    )
    filter_options_role = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)} ({role_counts[label]})</option>'
        for label in role_counts
        if label
    )

    sections: List[str] = []
    for label in MOTION_LABELS:
        cards = selected_cards.get(label, [])
        cards_html = []
        for idx, record in enumerate(cards):
            rel_dir = f"{label}/sample_{idx:02d}"
            cards_html.append(
                f"""
<article class="card"
  data-motion="{html.escape(str(record['motion_label']))}"
  data-count="{html.escape(str(record['count_bucket']))}"
  data-collision="{html.escape(str(record['collision_type_bucket']))}"
  data-role="{html.escape(str(record['sample_role']))}">
  <a class="thumb-link" href="{html.escape(rel_dir)}/rgb.mp4">
    <img loading="lazy" src="{html.escape(rel_dir)}/preview.gif" alt="{html.escape(record['sample_dir'].name)}">
  </a>
  <div class="card-body">
    <div class="title-row">
      <h3>{html.escape(record['sample_dir'].name)}</h3>
      <span class="score">{record['motion_score']:.2f}</span>
    </div>
    <div class="mini-row">{html.escape(record['motion_category'])} | {html.escape(record['count_bucket'])} | {record['num_objects']} obj</div>
    <div class="badge-row">
      <span class="badge motion">{html.escape(record['motion_label'])}</span>
      <span class="badge">{html.escape(record['collision_type_bucket'])}</span>
      <span class="badge">{html.escape(record['collision_count_bucket'])}</span>
      <span class="badge">{html.escape(record['sample_role'])}</span>
    </div>
    <div class="mini-row">obj-obj {record['obj_obj_event_count']} | obj-env {record['obj_env_event_count']}</div>
    <div class="mini-row muted">{html.escape(record['scene_composition'])}</div>
    <div class="mini-row muted">{html.escape(_fmt_mtime(float(record['mtime'])))}</div>
    <div class="link-row">
      <a href="{html.escape(rel_dir)}/rgb.mp4">mp4</a>
      <a href="{html.escape(rel_dir)}/metadata.json">meta</a>
      <a href="{html.escape(rel_dir)}/summary.json">summary</a>
    </div>
  </div>
</article>
"""
            )
        sections.append(
            f"""
<section class="bucket">
  <div class="bucket-head">
    <h2>{html.escape(label)}</h2>
    <span class="bucket-count">recent count {motion_counts.get(label, 0)} | showing {len(cards)}</span>
  </div>
  <div class="grid">
    {''.join(cards_html) if cards_html else '<p class="empty">No recent samples in this bucket.</p>'}
  </div>
</section>
"""
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Recent Motion Bucket Preview</title>
  <style>
    :root {{
      --bg: #efe9de;
      --panel: #fffaf4;
      --ink: #1a1816;
      --muted: #6a645c;
      --line: #dacdbd;
      --shadow: rgba(44, 31, 20, 0.09);
      --accent: #8b3a21;
      --accent2: #1d6f69;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(139,58,33,0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(29,111,105,0.10), transparent 24%),
        var(--bg);
    }}
    .page {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 20px 16px 40px;
    }}
    .hero, .bucket {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,250,244,0.98), rgba(248,240,230,0.96));
      box-shadow: 0 16px 40px var(--shadow);
    }}
    .hero {{
      padding: 18px 20px;
      margin-bottom: 14px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: 30px;
      line-height: 1.08;
      margin-bottom: 8px;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    .chip-row, .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.72);
      font-size: 12px;
    }}
    .control {{
      display: grid;
      gap: 4px;
      min-width: 160px;
    }}
    .control label {{
      font-size: 12px;
      color: var(--muted);
    }}
    select {{
      height: 34px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 0 10px;
    }}
    .bucket {{
      margin-bottom: 12px;
      padding: 12px;
    }}
    .bucket-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .bucket-count {{
      color: var(--muted);
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
    }}
    .card {{
      display: grid;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 8px;
      background: rgba(255,255,255,0.66);
    }}
    .thumb-link {{
      display: block;
      text-decoration: none;
    }}
    .thumb-link img {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      border-radius: 10px;
      background: #101216;
      display: block;
    }}
    .card-body {{
      display: grid;
      gap: 5px;
    }}
    .title-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: start;
    }}
    .title-row h3 {{
      font-size: 13px;
      line-height: 1.22;
      word-break: break-word;
    }}
    .score {{
      font-size: 12px;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 999px;
      padding: 3px 8px;
    }}
    .mini-row {{
      font-size: 11px;
      line-height: 1.35;
      word-break: break-word;
    }}
    .muted {{
      color: var(--muted);
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      background: rgba(255,255,255,0.84);
      font-size: 11px;
    }}
    .badge.motion {{
      border-color: #e2b7a6;
      color: #8b3a21;
    }}
    .link-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 11px;
    }}
    .link-row a {{
      color: var(--accent2);
      text-decoration: none;
    }}
    .link-row a:hover {{
      text-decoration: underline;
    }}
    .empty {{
      color: var(--muted);
      margin: 6px 0;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Recent Motion Bucket Preview</h1>
      <p>按最近样本重新分桶，主桶为 `static / simple / moderate / complex`，同时展示 `count_bucket / collision_type / collision_count / sample_role` 标签。页面默认优先显示最近修改的样本 GIF。</p>
      <p>当前输出目录: <code>{html.escape(output_dir.name)}</code></p>
      <div class="chip-row">{summary_chips}</div>
      <div class="control-row">
        <div class="control">
          <label for="motionFilter">motion</label>
          <select id="motionFilter">
            <option value="">all</option>
            {filter_options_motion}
          </select>
        </div>
        <div class="control">
          <label for="countFilter">count bucket</label>
          <select id="countFilter">
            <option value="">all</option>
            {filter_options_count}
          </select>
        </div>
        <div class="control">
          <label for="collisionFilter">collision type</label>
          <select id="collisionFilter">
            <option value="">all</option>
            {filter_options_collision}
          </select>
        </div>
        <div class="control">
          <label for="roleFilter">sample role</label>
          <select id="roleFilter">
            <option value="">all</option>
            {filter_options_role}
          </select>
        </div>
      </div>
    </section>
    {''.join(sections)}
  </div>
  <script>
    const filters = {{
      motion: document.getElementById('motionFilter'),
      count: document.getElementById('countFilter'),
      collision: document.getElementById('collisionFilter'),
      role: document.getElementById('roleFilter'),
    }};
    const cards = Array.from(document.querySelectorAll('.card'));
    function applyFilters() {{
      const motion = filters.motion.value;
      const count = filters.count.value;
      const collision = filters.collision.value;
      const role = filters.role.value;
      cards.forEach((card) => {{
        const ok = (!motion || card.dataset.motion === motion)
          && (!count || card.dataset.count === count)
          && (!collision || card.dataset.collision === collision)
          && (!role || card.dataset.role === role);
        card.style.display = ok ? '' : 'none';
      }});
    }}
    Object.values(filters).forEach((node) => node.addEventListener('change', applyFilters));
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records = iter_recent_samples(args)
    if not records:
        raise RuntimeError(
            f"No recent samples found under {args.dataset_root} within {args.modified_within_hours} hours."
        )

    selected_cards = _select_cards(records, samples_per_bucket=int(args.samples_per_bucket))

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    ensure_dir(args.output_dir)

    summary_payload: Dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "modified_within_hours": float(args.modified_within_hours),
        "max_recent_samples": int(args.max_recent_samples),
        "samples_per_bucket": int(args.samples_per_bucket),
        "recent_sample_count": int(len(records)),
        "motion_bucket_counts": _group_counts(records, "motion_label"),
        "count_bucket_counts": _group_counts(records, "count_bucket"),
        "collision_type_counts": _group_counts(records, "collision_type_bucket"),
        "sample_role_counts": _group_counts(records, "sample_role"),
        "selected_samples": {},
    }

    for label in MOTION_LABELS:
        bucket_dir = args.output_dir / label
        ensure_dir(bucket_dir)
        chosen = selected_cards.get(label, [])
        selected_meta: List[Dict[str, Any]] = []
        for idx, record in enumerate(chosen):
            dst_dir = bucket_dir / f"sample_{idx:02d}"
            ensure_dir(dst_dir)
            make_rgb_gif(
                rgb_dir=record["rgb_dir"],
                dst=dst_dir / "preview.gif",
                max_side=int(args.max_gif_side),
                duration_ms=int(args.gif_duration_ms),
            )
            copy_or_symlink(record["video_path"], dst_dir / "rgb.mp4")
            (dst_dir / "metadata.json").write_text(
                json.dumps(record["metadata"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sample_summary = {
                "sample_dir": str(record["sample_dir"]),
                "motion_label": str(record["motion_label"]),
                "motion_score": float(record["motion_score"]),
                "motion_metrics": dict(record["motion_metrics"]),
                "num_objects": int(record["num_objects"]),
                "count_bucket": str(record["count_bucket"]),
                "scene_composition": str(record["scene_composition"]),
                "sample_role": str(record["sample_role"]),
                "motion_category": str(record["motion_category"]),
                "collision_type_bucket": str(record["collision_type_bucket"]),
                "collision_count_bucket": str(record["collision_count_bucket"]),
                "obj_obj_event_count": int(record["obj_obj_event_count"]),
                "obj_env_event_count": int(record["obj_env_event_count"]),
                "mtime": float(record["mtime"]),
            }
            (dst_dir / "summary.json").write_text(
                json.dumps(sample_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            selected_meta.append(sample_summary)
        summary_payload["selected_samples"][label] = selected_meta

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(build_html(records, selected_cards, args.output_dir), encoding="utf-8")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
