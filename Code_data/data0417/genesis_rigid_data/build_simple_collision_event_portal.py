#!/usr/bin/env python3
# 用途：生成简化版碰撞事件 GIF portal。
"""Build a compact collision-event portal for a few Genesis samples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from build_main_collision_summary import summarize_sample


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def asset_copy(src: Path, dst_dir: Path, name: str | None = None) -> str:
    ensure_dir(dst_dir)
    filename = name or src.name
    dst = dst_dir / filename
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return os.path.relpath(dst, dst_dir.parent)


def sample_label(sample_dir: Path) -> str:
    rel = sample_dir.as_posix()
    if "/count_02/" in rel:
        return "count_02"
    if "/count_03_04/" in rel:
        return "count_03_04"
    return sample_dir.parent.name


def open_rgb_frames(sample_dir: Path) -> list[Image.Image]:
    frames = []
    for path in sorted((sample_dir / "rgb").glob("frame_*.png")):
        frames.append(Image.open(path).convert("RGB"))
    if not frames:
        raise FileNotFoundError(f"no rgb frames under {sample_dir / 'rgb'}")
    return frames


def frame_triplet(frames: list[Image.Image], onset: int) -> tuple[int, int, int]:
    last = len(frames) - 1
    return max(0, onset - 1), max(0, min(onset, last)), max(0, min(onset + 1, last))


def save_triptych(frames: list[Image.Image], onset: int, title: str, out_path: Path) -> None:
    idxs = frame_triplet(frames, onset)
    imgs = [frames[i] for i in idxs]
    w, h = imgs[0].size
    header_h = 34
    gap = 8
    canvas = Image.new("RGB", (w * 3 + gap * 2, h + header_h), (247, 241, 232))
    draw = ImageDraw.Draw(canvas)
    labels = [f"before f={idxs[0]}", f"onset f={idxs[1]}", f"after f={idxs[2]}"]
    for col, (img, label) in enumerate(zip(imgs, labels)):
        x = col * (w + gap)
        canvas.paste(img, (x, header_h))
        draw.rectangle((x, 0, x + w, header_h - 6), fill=(237, 223, 207))
        draw.text((x + 10, 8), label, fill=(34, 30, 24))
    draw.text((10, header_h - 28), title, fill=(150, 72, 21))
    canvas.save(out_path)


def save_gif_sequence(frames: list[Image.Image], out_path: Path, duration_ms: int = 120) -> None:
    seq = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    seq[0].save(
        out_path,
        save_all=True,
        append_images=seq[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_event_gif(frames: list[Image.Image], onset: int, out_path: Path, radius: int = 2) -> None:
    start = max(0, onset - radius)
    end = min(len(frames), onset + radius + 1)
    clip = frames[start:end]
    if not clip:
        clip = [frames[min(max(onset, 0), len(frames) - 1)]]
    save_gif_sequence(clip, out_path, duration_ms=140)


def save_scene_gif(frames: list[Image.Image], out_path: Path, max_frames: int = 24) -> None:
    if len(frames) <= max_frames:
        clip = frames
    else:
        step = max(1, len(frames) // max_frames)
        clip = frames[::step]
        if clip[-1] is not frames[-1]:
            clip = clip + [frames[-1]]
    save_gif_sequence(clip, out_path, duration_ms=120)


def build_cards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for item in summary.get("pair_records", []):
        for onset in item.get("onset_frames", []):
            cards.append(
                {
                    "kind": "recorded_obj_obj",
                    "title": f"Recorded object-object: {item['names'][0]} <-> {item['names'][1]}",
                    "frame": int(onset),
                }
            )
    for item in summary.get("environment_contact_records", []):
        for onset in item.get("onset_frames", []):
            cards.append(
                {
                    "kind": "recorded_obj_env",
                    "title": f"Recorded object-ground: {item['object_name']} -> {item['environment_name']}",
                    "frame": int(onset),
                }
            )
    if not any(card["kind"] == "recorded_obj_obj" for card in cards) and summary.get("closest_role_pair"):
        hint = summary["closest_role_pair"]
        cards.append(
            {
                "kind": "interaction_hint",
                "title": f"Interaction hint: role pair closest at frame {int(hint['frame'])}",
                "frame": int(hint["frame"]),
                "distance": float(hint["distance"]),
            }
        )
    cards.sort(key=lambda x: (int(x["frame"]), str(x["kind"])))
    return cards


def sample_block(sample_dir: Path, output_dir: Path) -> str:
    label = sample_label(sample_dir)
    summary = summarize_sample(sample_dir)
    frames = open_rgb_frames(sample_dir)

    sample_slug = f"{label}__{sample_dir.name}"
    asset_dir = output_dir / "assets" / sample_slug
    ensure_dir(asset_dir)
    scene_gif_path = asset_dir / "scene.gif"
    save_scene_gif(frames, scene_gif_path)
    scene_gif_rel = os.path.relpath(scene_gif_path, output_dir)

    cards_html = []
    for idx, card in enumerate(build_cards(summary)):
        card_name = f"event_{idx:02d}.gif"
        card_path = asset_dir / card_name
        extra = ""
        if "distance" in card:
            extra = f" | closest distance={card['distance']:.3f}"
        save_event_gif(frames, int(card["frame"]), card_path)
        cards_html.append(
            f"""
<div class="event-card">
  <div class="event-title">{card['title']}</div>
  <div class="event-meta">frame {int(card['frame'])}{extra}</div>
  <img src="{html_escape(os.path.relpath(card_path, output_dir))}" alt="{html_escape(card['title'])}">
</div>
"""
        )

    if not cards_html:
        cards_html.append("<div class='event-card'><div class='event-title'>No event card</div></div>")

    obj_obj_onsets = sum(len(item.get("onset_frames", [])) for item in summary.get("pair_records", []))
    obj_env_onsets = sum(len(item.get("onset_frames", [])) for item in summary.get("environment_contact_records", []))

    return f"""
<section class="sample-card">
  <h2>{label}</h2>
  <div class="chips">
    <span class="chip">bucket = {html_escape(str(summary.get('derived_collision_bucket', '')))}</span>
    <span class="chip">obj-obj onsets = {obj_obj_onsets}</span>
    <span class="chip">obj-ground onsets = {obj_env_onsets}</span>
  </div>
  <img src="{html_escape(scene_gif_rel)}" alt="scene gif">
  <div class="events">
    {''.join(cards_html)}
  </div>
</section>
"""


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_page(sample_dirs: list[Path], output_dir: Path) -> str:
    blocks = [sample_block(path, output_dir) for path in sample_dirs]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simple Collision Events</title>
  <style>
    :root {{
      --bg: #f5f1ea;
      --panel: #fffdf8;
      --ink: #1f1b17;
      --muted: #6b6258;
      --line: #d8ccbe;
      --accent: #a04d15;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Arial, sans-serif; }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0 0 8px; }}
    .sub {{ margin: 0 0 18px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .sample-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .chips {{ margin-bottom: 10px; }}
    .chip {{
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #f7eadc;
      border: 1px solid var(--line);
      color: var(--accent);
      font-size: 13px;
    }}
    img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #111;
    }}
    .events {{ display: grid; gap: 12px; margin-top: 12px; }}
    .event-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px;
      background: #fffaf4;
    }}
    .event-title {{ font-weight: 700; margin-bottom: 4px; }}
    .event-meta {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    @media (max-width: 1000px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Simple Collision Event View</h1>
    <p class="sub">只保留原视频和事件卡片。每张卡片展示 before / onset / after 三帧；如果没有记录到 object-object onset，就只展示 ground onset 和 interaction hint。</p>
    <div class="grid">{''.join(blocks)}</div>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("sample_dirs", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    html = build_page([Path(x) for x in args.sample_dirs], output_dir)
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
