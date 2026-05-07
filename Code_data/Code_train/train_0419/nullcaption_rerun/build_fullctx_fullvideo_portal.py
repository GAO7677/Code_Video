#!/usr/bin/env python3
"""Build a portal for physics-iq full-context/full-video caption vs null-caption comparisons."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any


BENCH_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
LIVE_ROOT = BENCH_ROOT / "tools" / "fullctx_runs" / "physicsiq_fullvideo"
PORTAL_DIR = BENCH_ROOT / "tools" / "visualization" / "physicsiq_fullctx_fullvideo_portal"
ASSETS_DIR = PORTAL_DIR / "assets"
OUTPUT_DIR = BENCH_ROOT / "output" / "VACE_1_3B_V2V" / "context_fullctx_fullvideo"
CAPTION_DIR = LIVE_ROOT / "generated" / "caption_fullctx_fullvideo"
NULL_DIR = LIVE_ROOT / "generated" / "nullcaption_fullctx_fullvideo"
META_DIR = LIVE_ROOT / "meta"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_to_root(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def web_path(path: str | None) -> str | None:
    if not path:
        return None
    return "/" + path.replace(os.sep, "/").lstrip("/")


def ensure_clean_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()


def ensure_symlink(target: Path, link_path: Path) -> Path | None:
    if not target.exists():
        return None
    ensure_clean_parent(link_path)
    link_path.symlink_to(target)
    return link_path


def expose_asset(target: Path | None, assets_dir: Path, link_name: str) -> str | None:
    if target is None or not target.exists():
        return None
    if target.is_relative_to(BENCH_ROOT):
        return relative_to_root(BENCH_ROOT, target)
    linked = ensure_symlink(target, assets_dir / link_name)
    if linked is None:
        return None
    return relative_to_root(BENCH_ROOT, linked)


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    return (
        f"<video controls preload='metadata' muted playsinline>"
        f"<source src='{html.escape(resolved or '')}' type='video/mp4'>"
        "</video>"
    )


def text_html(text: str, empty_label: str = "(empty)") -> str:
    if not text:
        return f"<div class='text-body empty'>{html.escape(empty_label)}</div>"
    return f"<div class='text-body'>{html.escape(text)}</div>"


def render_slot(title: str, body: str, extra_class: str = "") -> str:
    class_attr = "slot"
    if extra_class:
        class_attr += f" {extra_class}"
    return (
        f"<section class='{class_attr}'>"
        f"<div class='slot-head'>{html.escape(title)}</div>"
        f"{body}"
        "</section>"
    )


def load_cases() -> list[dict[str, Any]]:
    manifest = read_json(META_DIR / "manifest.json")
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        return []
    records: list[dict[str, Any]] = []
    for case_name in cases:
        if not isinstance(case_name, str):
            continue
        caption_meta = read_json(META_DIR / case_name / "caption.json")
        null_meta = read_json(META_DIR / case_name / "nullcaption.json")
        paths = caption_meta.get("paths", {})
        if not isinstance(paths, dict):
            continue

        case_assets = ASSETS_DIR / case_name
        case_assets.mkdir(parents=True, exist_ok=True)

        caption_candidates = sorted(CAPTION_DIR.glob(f"{case_name}*.json"))
        null_candidates = sorted(NULL_DIR.glob(f"{case_name}*.json"))
        caption_json = caption_candidates[0] if caption_candidates else None
        null_json = null_candidates[0] if null_candidates else None

        caption_sidecar = read_json(caption_json) if caption_json is not None else {}
        null_sidecar = read_json(null_json) if null_json is not None else {}

        caption_paths = caption_sidecar.get("paths", {}) if isinstance(caption_sidecar, dict) else {}
        null_paths = null_sidecar.get("paths", {}) if isinstance(null_sidecar, dict) else {}

        caption_video_path = caption_paths.get("output_video_path") if isinstance(caption_paths, dict) else None
        null_video_path = null_paths.get("output_video_path") if isinstance(null_paths, dict) else None
        caption_video = Path(caption_video_path) if isinstance(caption_video_path, str) and caption_video_path else None
        null_video = Path(null_video_path) if isinstance(null_video_path, str) and null_video_path else None

        context_range = caption_meta.get("context_frame_range") or [0, 0]
        future_range = caption_meta.get("future_frame_range") or [0, 0]
        context_frames = int(context_range[1]) - int(context_range[0]) + 1
        future_frames = int(future_range[1]) - int(future_range[0]) + 1
        full_frames = context_frames + future_frames
        fps = int(float(caption_meta.get("fps") or 30))

        status_lines = [
            f"context_frames: {context_frames}",
            f"future_frames: {future_frames}",
            f"full_frames: {full_frames}",
            f"fps: {fps}",
            f"caption_generated: {'ready' if caption_video is not None and caption_video.exists() else 'pending'}",
            f"nullcaption_generated: {'ready' if null_video is not None and null_video.exists() else 'pending'}",
        ]

        records.append(
            {
                "case_name": case_name,
                "caption": str(caption_meta.get("caption") or ""),
                "null_caption": str(null_meta.get("caption") or ""),
                "category": str(caption_meta.get("category") or ""),
                "scenario": str(caption_meta.get("scenario") or ""),
                "context_video": expose_asset(Path(paths["context_video_path"]), case_assets, "context_video.mp4"),
                "gt_full_video": expose_asset(Path(paths["full_video_path"]), case_assets, "gt_full_video.mp4"),
                "caption_generated": expose_asset(caption_video, case_assets, "generated_with_caption.mp4"),
                "null_generated": expose_asset(null_video, case_assets, "generated_nullcaption.mp4"),
                "caption_json_rel": (
                    relative_to_root(BENCH_ROOT, OUTPUT_DIR / f"{caption_video.stem}__caption_fullctx_fullvideo.json")
                    if caption_video is not None
                    else None
                ),
                "null_json_rel": (
                    relative_to_root(BENCH_ROOT, OUTPUT_DIR / f"{null_video.stem}__nullcaption_fullctx_fullvideo.json")
                    if null_video is not None
                    else None
                ),
                "live_caption_json_rel": relative_to_root(BENCH_ROOT, caption_json) if caption_json is not None else None,
                "live_null_json_rel": relative_to_root(BENCH_ROOT, null_json) if null_json is not None else None,
                "status_text": "\n".join(status_lines),
            }
        )
    return records


def render_cards(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in records:
        dataset = "physics-iq-benchmark"
        cards.append(
            "<article class='case-card' "
            f"data-case='{html.escape(record['case_name'].lower())}' "
            f"data-caption='{html.escape(record['caption'].lower())}'>"
            "<div class='case-head'>"
            f"<span class='badge'>{dataset}</span>"
            f"<h2>{html.escape(record['case_name'])}</h2>"
            "</div>"
            "<div class='shared-grid'>"
            f"{render_slot('context_video_from_meta', media_html(record['context_video']), 'video-slot')}"
            f"{render_slot('gt_full_video', media_html(record['gt_full_video']), 'video-slot gt-slot')}"
            f"{render_slot('caption_text', text_html(record['caption']), 'meta-slot')}"
            f"{render_slot('null_caption_text', text_html(record['null_caption'], '(empty caption)'), 'meta-slot')}"
            "</div>"
            "<div class='generated-grid'>"
            f"{render_slot('generated_with_caption', media_html(record['caption_generated']), 'video-slot caption-slot')}"
            f"{render_slot('generated_nullcaption', media_html(record['null_generated']), 'video-slot null-slot')}"
            f"{render_slot('run_status', text_html(record['status_text']), 'meta-slot')}"
            "</div>"
            "<div class='path-grid'>"
            f"<p><strong>benchmark caption sidecar</strong><br>{html.escape(record['caption_json_rel'] or 'Missing')}</p>"
            f"<p><strong>benchmark null sidecar</strong><br>{html.escape(record['null_json_rel'] or 'Missing')}</p>"
            f"<p><strong>live caption sidecar</strong><br>{html.escape(record['live_caption_json_rel'] or 'Missing')}</p>"
            f"<p><strong>live null sidecar</strong><br>{html.escape(record['live_null_json_rel'] or 'Missing')}</p>"
            "</div>"
            "</article>"
        )
    return "".join(cards)


def build_html(records: list[dict[str, Any]]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Physics-IQ Full Context Full Video</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --caption-soft: #f3d7c9;
      --caption-ink: #6e2a13;
      --null-soft: #d6ead9;
      --null-ink: #28563c;
      --gt-soft: #e8dfd0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(181,83,45,0.10), transparent 26%),
        linear-gradient(180deg, #f9f6ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .shell {{
      width: min(1960px, calc(100vw - 20px));
      margin: 0 auto;
      padding: 14px 0 24px;
    }}
    .hero {{
      margin-bottom: 12px;
      padding: 14px 18px;
      background: rgba(255,253,248,0.90);
      border: 1px solid var(--line);
      border-radius: 14px;
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .filters {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin: 10px 0 12px;
    }}
    input {{
      width: 100%;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    .record-list {{
      display: grid;
      gap: 12px;
    }}
    .case-card {{
      padding: 10px;
      background: rgba(255,253,248,0.96);
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .case-head {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .case-head h2 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: #efe7da;
      color: #4f4338;
      font-size: 11px;
      font-weight: 700;
    }}
    .shared-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(220px, 1fr));
      gap: 8px;
      margin-bottom: 8px;
    }}
    .generated-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 8px;
      margin-bottom: 8px;
    }}
    .slot {{
      background: #fbf8f2;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      min-height: 156px;
    }}
    .slot-head {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      font-size: 11px;
      font-weight: 700;
      color: #55493d;
      background: rgba(239, 231, 218, 0.65);
    }}
    .caption-slot .slot-head {{
      background: rgba(243, 215, 201, 0.78);
      color: var(--caption-ink);
    }}
    .null-slot .slot-head {{
      background: rgba(214, 234, 217, 0.82);
      color: var(--null-ink);
    }}
    .gt-slot .slot-head {{
      background: rgba(232, 223, 208, 0.82);
      color: #584936;
    }}
    video {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 156px;
      object-fit: contain;
      background: #0d0d0d;
    }}
    .text-body {{
      min-height: 156px;
      padding: 8px 10px;
      background: #fffdf9;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .text-body.empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 156px;
      padding: 12px;
      color: var(--muted);
      background: repeating-linear-gradient(
        45deg,
        rgba(216, 207, 191, 0.35),
        rgba(216, 207, 191, 0.35) 10px,
        rgba(255, 253, 248, 0.75) 10px,
        rgba(255, 253, 248, 0.75) 20px
      );
    }}
    .path-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }}
    .path-grid p {{
      margin: 0;
      padding: 8px 10px;
      background: #fffaf2;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      word-break: break-all;
    }}
    @media (max-width: 1200px) {{
      .shared-grid, .generated-grid, .path-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Physics-IQ Full Context to Full Video</h1>
      <p>Each case uses the complete benchmark-provided <code>context_video.mp4</code> as input and compares the generated result against the benchmark <code>full_video.mp4</code>. Caption and null-caption generations are shown side-by-side under the same case card.</p>
    </section>
    <section class="filters">
      <input id="searchBox" type="search" placeholder="Search case id or caption text">
    </section>
    <section id="recordList" class="record-list">
      {render_cards(records)}
    </section>
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const cards = Array.from(document.querySelectorAll('.case-card'));
    function applyFilters() {{
      const search = searchBox.value.trim().toLowerCase();
      for (const card of cards) {{
        const haystack = `${{card.dataset.case}} ${{card.dataset.caption}}`.toLowerCase();
        card.style.display = !search || haystack.includes(search) ? '' : 'none';
      }}
    }}
    searchBox.addEventListener('input', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    PORTAL_DIR.mkdir(parents=True, exist_ok=True)
    records = load_cases()
    html_path = PORTAL_DIR / "index.html"
    html_path.write_text(build_html(records), encoding="utf-8")
    write_json(
        PORTAL_DIR / "build_summary.json",
        {
            "record_count": len(records),
            "html_path": str(html_path),
            "portal_url_path": f"/{relative_to_root(BENCH_ROOT, html_path)}",
        },
    )
    print(json.dumps({"record_count": len(records), "html_path": str(html_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
