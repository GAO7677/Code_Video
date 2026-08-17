#!/usr/bin/env python3
"""Build a static comparison page for PyBullet0613 raw and 49-frame inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from code_vjepa_vggt.data.pybullet_raw_no_gt_box_dataset import _english_caption


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/"
    "industrial_s1_scale2_merged_h264_batch1500"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pybullet0613_cache_preview"
)
DEFAULT_FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cases-per-family", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dynamic_signature(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(item.get("shape", "unknown"))
            for item in metadata.get("objects", [])
            if isinstance(item, dict) and bool(item.get("dynamic"))
        )
    )


def dynamic_objects(metadata: dict[str, Any]) -> list[str]:
    values = []
    for item in metadata.get("objects", []):
        if not isinstance(item, dict) or not bool(item.get("dynamic")):
            continue
        values.append(str(item.get("shape") or item.get("name") or "object"))
    return values


def select_cases(dataset_root: Path, split: str, per_family: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metadata_path in sorted((dataset_root / split).glob("*/*/meta.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        family = str(metadata.get("family_slug") or metadata_path.parent.parent.name)
        grouped[family].append(
            {
                "metadata": metadata,
                "metadata_path": metadata_path,
                "video_path": metadata_path.with_name("video.mp4"),
                "family": family,
                "signature": dynamic_signature(metadata),
                "caption": _english_caption(metadata, family),
            }
        )

    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        candidates = grouped[family]
        chosen: list[dict[str, Any]] = []
        seen_signatures: set[tuple[str, ...]] = set()
        seen_captions: set[str] = set()
        seen_templates: set[str] = set()
        while candidates and len(chosen) < per_family:
            best = max(
                candidates,
                key=lambda item: (
                    int(item["signature"] not in seen_signatures),
                    int(item["caption"] not in seen_captions),
                    int(item["metadata"].get("template_key") not in seen_templates),
                    -int(str(item["metadata"].get("sample_id", "0")).split("_")[-1]),
                ),
            )
            candidates.remove(best)
            chosen.append(best)
            seen_signatures.add(best["signature"])
            seen_captions.add(best["caption"])
            seen_templates.add(str(best["metadata"].get("template_key", "")))
        selected.extend(chosen)
    return selected


def ensure_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    target.unlink(missing_ok=True)
    os.symlink(source.resolve(), target)


def build_prefix_clip(
    ffmpeg: Path,
    source: Path,
    target: Path,
    *,
    overwrite: bool,
) -> None:
    if target.is_file() and not overwrite:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "49",
        "-vf",
        "scale=896:512:flags=bicubic",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def case_html(case: dict[str, Any], index: int) -> str:
    metadata = case["metadata"]
    family = escape(case["family"])
    sample_id = escape(str(metadata.get("sample_id", "unknown")))
    title = escape(str(metadata.get("title", metadata.get("template_key", sample_id))))
    prompt = escape(case["caption"])
    template = escape(str(metadata.get("template_key", "")))
    objects = " ".join(
        f'<span class="object-chip">{escape(value)}</span>'
        for value in dynamic_objects(metadata)
    )
    prefix_path = escape(case["prefix_relpath"])
    full_path = escape(case["full_relpath"])
    return f"""
      <article class="case-row" data-family="{family}">
        <header class="case-header">
          <div class="case-index">{index:02d}</div>
          <div class="case-identity">
            <div class="case-kicker">{family} / {sample_id}</div>
            <h2>{title}</h2>
          </div>
          <button class="sync-button" type="button">同步播放</button>
        </header>
        <div class="prompt-block">
          <span class="prompt-label">PROMPT</span>
          <p>{prompt}</p>
        </div>
        <div class="case-meta">
          <code>{template}</code>
          <div class="object-list">{objects}</div>
        </div>
        <div class="media-grid">
          <figure class="media-panel prefix-panel">
            <figcaption>
              <span>训练输入</span>
              <strong>F00-F48 · 896×512 · 1.63s</strong>
            </figcaption>
            <video controls muted loop preload="metadata" playsinline src="{prefix_path}"></video>
          </figure>
          <figure class="media-panel full-panel">
            <figcaption>
              <span>完整视频</span>
              <strong>F00-F89 · 960×540 · 3.00s</strong>
            </figcaption>
            <video controls muted loop preload="metadata" playsinline src="{full_path}"></video>
          </figure>
        </div>
      </article>
    """


def build_html(cases: list[dict[str, Any]], split: str) -> str:
    families = sorted({case["family"] for case in cases})
    family_buttons = "".join(
        f'<button type="button" data-filter="{escape(family)}">{escape(family)}</button>'
        for family in families
    )
    rows = "".join(case_html(case, index) for index, case in enumerate(cases, start=1))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PyBullet0613 · 49帧输入核查</title>
  <style>
    :root {{
      --paper: #f7f8fa;
      --surface: #ffffff;
      --ink: #17191e;
      --muted: #626975;
      --line: #d9dde3;
      --teal: #087e87;
      --teal-soft: #dceff0;
      --signal: #c94d3c;
      --video: #101216;
    }}
    * {{ box-sizing: border-box; }}
    html {{ color-scheme: light; background: var(--paper); }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }}
    button, code {{ font: inherit; letter-spacing: 0; }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 248, 250, 0.96);
      backdrop-filter: blur(12px);
    }}
    .topbar-inner, main {{ width: min(1480px, calc(100% - 40px)); margin: 0 auto; }}
    .topbar-inner {{ padding: 18px 0 14px; }}
    .title-line {{ display: flex; align-items: baseline; justify-content: space-between; gap: 24px; }}
    h1 {{ margin: 0; font: 700 22px/1.2 "Arial Narrow", Inter, sans-serif; }}
    .dataset-path {{ color: var(--muted); font: 12px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .toolbar {{ display: flex; align-items: center; gap: 14px; margin-top: 14px; flex-wrap: wrap; }}
    .filters {{ display: inline-flex; max-width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }}
    .filters button {{
      min-height: 34px;
      padding: 0 11px;
      border: 0;
      border-right: 1px solid var(--line);
      color: var(--muted);
      background: transparent;
      cursor: pointer;
      white-space: nowrap;
      font-size: 12px;
    }}
    .filters button:last-child {{ border-right: 0; }}
    .filters button.active {{ color: #fff; background: var(--teal); }}
    .summary {{ margin-left: auto; color: var(--muted); font-size: 12px; }}
    main {{ padding-bottom: 80px; }}
    .case-row {{ padding: 34px 0 40px; border-bottom: 1px solid var(--line); }}
    .case-row[hidden] {{ display: none; }}
    .case-header {{ display: grid; grid-template-columns: 42px minmax(0, 1fr) auto; align-items: center; gap: 12px; }}
    .case-index {{ color: var(--signal); font: 700 15px/1 ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .case-kicker {{ color: var(--teal); font-size: 11px; font-weight: 700; text-transform: uppercase; }}
    h2 {{ margin: 3px 0 0; font-size: 18px; line-height: 1.35; }}
    .sync-button {{
      min-height: 34px;
      padding: 0 13px;
      border: 1px solid var(--ink);
      border-radius: 5px;
      color: var(--ink);
      background: var(--surface);
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }}
    .sync-button:hover {{ color: #fff; background: var(--ink); }}
    .prompt-block {{ display: grid; grid-template-columns: 42px minmax(0, 1fr); gap: 12px; margin-top: 18px; }}
    .prompt-label {{ color: var(--muted); font: 700 10px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; }}
    .prompt-block p {{ margin: 0; max-width: 1050px; font-size: 15px; line-height: 1.55; }}
    .case-meta {{ display: flex; align-items: center; gap: 10px; min-height: 28px; margin: 10px 0 14px 54px; flex-wrap: wrap; }}
    .case-meta code {{ color: var(--muted); font-size: 11px; }}
    .object-list {{ display: flex; gap: 5px; flex-wrap: wrap; }}
    .object-chip {{ padding: 3px 7px; border-radius: 4px; color: var(--teal); background: var(--teal-soft); font-size: 10px; font-weight: 700; }}
    .media-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-left: 54px; }}
    .media-panel {{ margin: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }}
    .media-panel figcaption {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 38px; padding: 0 12px; border-bottom: 1px solid var(--line); font-size: 11px; }}
    .media-panel figcaption span {{ color: var(--muted); }}
    .media-panel figcaption strong {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 10px; }}
    video {{ display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: var(--video); }}
    .prefix-panel video {{ aspect-ratio: 7 / 4; }}
    button:focus-visible, video:focus-visible {{ outline: 3px solid var(--signal); outline-offset: 2px; }}
    @media (max-width: 820px) {{
      .topbar-inner, main {{ width: min(100% - 24px, 1480px); }}
      .title-line {{ align-items: flex-start; flex-direction: column; gap: 6px; }}
      .dataset-path {{ overflow-wrap: anywhere; }}
      .summary {{ width: 100%; margin-left: 0; }}
      .case-header {{ grid-template-columns: 32px minmax(0, 1fr); }}
      .sync-button {{ grid-column: 2; justify-self: start; }}
      .prompt-block {{ grid-template-columns: 32px minmax(0, 1fr); }}
      .case-meta, .media-grid {{ margin-left: 44px; }}
      .media-grid {{ grid-template-columns: 1fr; }}
      .media-panel figcaption {{ align-items: flex-start; flex-direction: column; justify-content: center; gap: 2px; padding: 8px 10px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; }} }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-inner">
      <div class="title-line">
        <h1>PyBullet0613 · 49帧训练输入核查</h1>
        <div class="dataset-path">raw_v1 / {escape(split)} / prefix F00-F48</div>
      </div>
      <div class="toolbar">
        <div class="filters" role="group" aria-label="Family filter">
          <button type="button" data-filter="all" class="active">全部</button>
          {family_buttons}
        </div>
        <div class="summary"><span id="visible-count">{len(cases)}</span> / {len(cases)} cases · 2 videos each</div>
      </div>
    </div>
  </div>
  <main>{rows}</main>
  <script>
    const rows = [...document.querySelectorAll('.case-row')];
    const count = document.querySelector('#visible-count');
    document.querySelectorAll('[data-filter]').forEach((button) => {{
      button.addEventListener('click', () => {{
        document.querySelectorAll('[data-filter]').forEach((item) => item.classList.remove('active'));
        button.classList.add('active');
        const filter = button.dataset.filter;
        let visible = 0;
        rows.forEach((row) => {{
          row.hidden = filter !== 'all' && row.dataset.family !== filter;
          if (!row.hidden) visible += 1;
          if (row.hidden) row.querySelectorAll('video').forEach((video) => video.pause());
        }});
        count.textContent = String(visible);
      }});
    }});
    document.querySelectorAll('.sync-button').forEach((button) => {{
      button.addEventListener('click', async () => {{
        const videos = [...button.closest('.case-row').querySelectorAll('video')];
        videos.forEach((video) => {{ video.pause(); video.currentTime = 0; }});
        await Promise.allSettled(videos.map((video) => video.play()));
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg not found: {ffmpeg}")
    cases = select_cases(dataset_root, args.split, max(1, args.cases_per_family))
    if not cases:
        raise RuntimeError(f"no cases found under {dataset_root / args.split}")

    media_root = output_root / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, start=1):
        sample_id = str(case["metadata"].get("sample_id", f"case{index:03d}"))
        stem = f"{case['family']}_{sample_id}"
        prefix_target = media_root / f"{stem}_prefix49_896x512.mp4"
        full_target = media_root / f"{stem}_full90_960x540.mp4"
        build_prefix_clip(
            ffmpeg,
            case["video_path"],
            prefix_target,
            overwrite=args.overwrite,
        )
        ensure_symlink(case["video_path"], full_target)
        case["prefix_relpath"] = prefix_target.relative_to(output_root).as_posix()
        case["full_relpath"] = full_target.relative_to(output_root).as_posix()
        print(f"[{index}/{len(cases)}] {case['family']} {sample_id}", flush=True)

    manifest = [
        {
            "family": case["family"],
            "sample_id": case["metadata"].get("sample_id"),
            "template_key": case["metadata"].get("template_key"),
            "title": case["metadata"].get("title"),
            "prompt": case["caption"],
            "dynamic_objects": dynamic_objects(case["metadata"]),
            "prefix_video": case["prefix_relpath"],
            "full_video": case["full_relpath"],
        }
        for case in cases
    ]
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "index.html").write_text(
        build_html(cases, args.split),
        encoding="utf-8",
    )
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
