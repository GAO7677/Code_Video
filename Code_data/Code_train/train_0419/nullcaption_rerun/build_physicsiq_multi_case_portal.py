#!/usr/bin/env python3
"""Build a portal for multiple physics-iq cases comparing caption/null-caption context sweeps."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import batch_eval_lora as bel


BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
WORK_ROOT = BENCHMARK_ROOT / "tools" / "multi_case_runs" / "physicsiq_more_cases_ctx_sweep"
PORTAL_DIR = BENCHMARK_ROOT / "tools" / "visualization" / "physicsiq_more_cases_ctx_sweep"
ASSETS_DIR = PORTAL_DIR / "assets"
CONTEXTS = [8, 16, 32, 38]
VARIANTS = ["caption", "nullcaption"]
CASES = [
    "0008_perspective-center_trimmed-ball-hits-duck",
    "0011_perspective-center_trimmed-ball-hits-nothing",
    "0014_perspective-center_trimmed-ball-in-basket",
    "0017_perspective-center_trimmed-ball-in-sand",
]


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
    if target.is_relative_to(BENCHMARK_ROOT):
        return relative_to_root(BENCHMARK_ROOT, target)
    linked = ensure_symlink(target, assets_dir / link_name)
    if linked is None:
        return None
    return relative_to_root(BENCHMARK_ROOT, linked)


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    return (
        f"<video controls preload='metadata' muted playsinline>"
        f"<source src='{html.escape(resolved or '')}' type='video/mp4'>"
        "</video>"
    )


def text_html(text: str) -> str:
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


def save_context_clip(
    *,
    context_path: Path,
    context_frames: int,
    height: int,
    width: int,
    resize_mode: str,
    fps: int,
    output_path: Path,
) -> Path:
    frames = bel.load_context_frames(
        context_path=context_path,
        context_frames=context_frames,
        height=height,
        width=width,
        resize_mode=resize_mode,
    )
    bel.save_video(frames, str(output_path), fps=fps, quality=5)
    return output_path


def build_case_block(case_name: str) -> str:
    meta_path = Path(f"/data/gaoya/dataset/physics-iq-benchmark/mytest/{case_name}/meta.json")
    meta = read_json(meta_path)
    paths = meta["paths"]
    context_path = Path(paths["context_video_path"])
    full_video_path = Path(paths["full_video_path"])
    caption = str(meta.get("caption") or "")
    fps = int(float(meta.get("fps") or 30))
    context_range = meta.get("context_frame_range") or [0, 0]
    source_context_frames = int(context_range[1]) - int(context_range[0]) + 1 if len(context_range) == 2 else 0
    resize_mode = bel.resolve_context_resize_mode("physics-iq-benchmark")
    case_assets = ASSETS_DIR / case_name
    case_assets.mkdir(parents=True, exist_ok=True)

    full_context_clip = case_assets / "source_context_full_clip.mp4"
    save_context_clip(
        context_path=context_path,
        context_frames=source_context_frames or max(CONTEXTS),
        height=544,
        width=720,
        resize_mode=resize_mode,
        fps=fps,
        output_path=full_context_clip,
    )

    rows: list[str] = []
    for ctx in CONTEXTS:
        context_clip = case_assets / f"actual_context_{ctx:02d}f.mp4"
        save_context_clip(
            context_path=context_path,
            context_frames=ctx,
            height=544,
            width=720,
            resize_mode=resize_mode,
            fps=fps,
            output_path=context_clip,
        )

        pair_slots = []
        for variant in VARIANTS:
            gen_dir = WORK_ROOT / "generated" / case_name / f"{variant}_context_{ctx:02d}f"
            mp4s = sorted(gen_dir.glob("*.mp4"))
            gen_path = mp4s[0] if mp4s else None
            label = "caption" if variant == "caption" else "null-caption"
            pair_slots.append(
                render_slot(
                    f"{label}_generated",
                    media_html(expose_asset(gen_path, case_assets, f"{variant}_context_{ctx:02d}f.mp4")),
                    "output-slot",
                )
            )

        summary = (
            f"used_context_frames: {ctx}\n"
            f"source_context_fps: {fps}\n"
            f"approx_context_seconds: {ctx / fps:.3f}\n"
            "generated_output_fps: 16\n"
            "generated_output_frames: 49"
        )
        rows.append(
            "<section class='variant-card'>"
            "<div class='variant-head'>"
            f"<h3>context_{ctx:02d}f</h3>"
            f"{text_html(summary)}"
            "</div>"
            "<div class='variant-row variant-row-3'>"
            f"{render_slot('actual_context_video', media_html(relative_to_root(BENCHMARK_ROOT, context_clip)), 'context-slot')}"
            f"{''.join(pair_slots)}"
            "</div>"
            "</section>"
        )

    return (
        "<article class='case-card'>"
        f"<div class='case-head'><span class='badge'>physics-iq-benchmark</span><h2>{html.escape(case_name)}</h2></div>"
        "<div class='shared-grid'>"
        f"{render_slot('source_context_full_clip', media_html(relative_to_root(BENCHMARK_ROOT, full_context_clip)), 'context-slot')}"
        f"{render_slot('gt_full_video', media_html(expose_asset(full_video_path, case_assets, 'gt_full_video.mp4')), 'context-slot')}"
        f"{render_slot('caption', text_html(caption), 'meta-slot')}"
        "</div>"
        f"<div class='variant-stack'>{''.join(rows)}</div>"
        "</article>"
    )


def build_html(case_blocks: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Physics-IQ Multi-Case Context Sweeps</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --ok-soft: #d6ead9;
      --ok-ink: #28563c;
      --alt-soft: #f3d7c9;
      --alt-ink: #6e2a13;
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
      width: min(2100px, calc(100vw - 20px));
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
    .case-list {{ display: grid; gap: 14px; }}
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
    .case-head h2 {{ margin: 0; font-size: 16px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(214, 234, 217, 0.82);
      color: var(--ok-ink);
      font-size: 11px;
      font-weight: 700;
    }}
    .shared-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .variant-stack {{ display: grid; gap: 10px; }}
    .variant-card {{
      padding: 10px;
      background: #fbf8f2;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .variant-head {{
      display: grid;
      grid-template-columns: minmax(180px, 240px) 1fr;
      gap: 8px;
      align-items: start;
      margin-bottom: 8px;
    }}
    .variant-head h3 {{
      margin: 0;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(214, 234, 217, 0.82);
      color: var(--ok-ink);
      font-size: 15px;
    }}
    .variant-row {{
      display: grid;
      gap: 8px;
    }}
    .variant-row-3 {{
      grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr) minmax(320px, 1fr);
    }}
    .slot {{
      background: #fffdf8;
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
    .context-slot .slot-head {{
      background: rgba(232, 223, 208, 0.82);
    }}
    .output-slot .slot-head {{
      background: rgba(243, 215, 201, 0.78);
      color: var(--alt-ink);
    }}
    .meta-slot .slot-head {{
      background: rgba(214, 234, 217, 0.82);
      color: var(--ok-ink);
    }}
    video {{
      display: block;
      width: 100%;
      min-height: 156px;
      background: #0d0d0d;
    }}
    .text-body {{
      min-height: 156px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 156px;
      padding: 12px;
      color: var(--muted);
      background: repeating-linear-gradient(45deg, rgba(216,207,191,0.35), rgba(216,207,191,0.35) 10px, rgba(255,253,248,0.75) 10px, rgba(255,253,248,0.75) 20px);
    }}
    @media (max-width: 1400px) {{
      .shared-grid, .variant-head, .variant-row-3 {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Physics-IQ Multi-Case Context Sweeps</h1>
      <p>Each case shows GT/context and the corresponding generated outputs for caption and null-caption under context lengths 8/16/32/38.</p>
    </section>
    <section class="case-list">
      {case_blocks}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    PORTAL_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    blocks = [build_case_block(case_name) for case_name in CASES]
    html_path = PORTAL_DIR / "index.html"
    html_path.write_text(build_html("".join(blocks)), encoding="utf-8")
    write_json(
        PORTAL_DIR / "build_summary.json",
        {
            "html_path": str(html_path),
            "portal_url_path": f"/{relative_to_root(BENCHMARK_ROOT, html_path)}",
            "num_cases": len(CASES),
        },
    )
    print(json.dumps({"html_path": str(html_path), "num_cases": len(CASES)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
