#!/usr/bin/env python3
"""Build a portal for one case comparing different context lengths."""

from __future__ import annotations

import argparse
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


DEFAULT_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
DEFAULT_ORIG_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V")
DEFAULT_CASE_META = Path(
    "/data/gaoya/dataset/physics-iq-benchmark/mytest/0005_perspective-center_trimmed-ball-behind-rotating-paper/meta.json"
)
DEFAULT_CONTEXT_LENGTHS = [8, 16, 32, 38]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--orig_benchmark_root", type=Path, default=DEFAULT_ORIG_BENCHMARK_ROOT)
    parser.add_argument("--meta_json_path", type=Path, default=DEFAULT_CASE_META)
    parser.add_argument(
        "--portal_subdir",
        type=Path,
        default=Path("tools/visualization/single_case_context_sweep_0005"),
    )
    parser.add_argument("--context-lengths", default="8,16,32,38")
    return parser.parse_args()


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


def expose_asset(
    *,
    target: Path | None,
    benchmark_root: Path,
    assets_dir: Path,
    link_name: str,
) -> str | None:
    if target is None or not target.exists():
        return None
    if target.is_relative_to(benchmark_root):
        return relative_to_root(benchmark_root, target)
    linked = ensure_symlink(target, assets_dir / link_name)
    if linked is None:
        return None
    return relative_to_root(benchmark_root, linked)


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


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    if path.lower().endswith(".mp4"):
        return (
            f"<video controls preload='metadata' muted playsinline>"
            f"<source src='{html.escape(resolved or '')}' type='video/mp4'>"
            "</video>"
        )
    return f"<img loading='lazy' src='{html.escape(resolved or '')}' alt='asset'>"


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


def build_html(payload: dict[str, Any]) -> str:
    rows = []
    for item in payload["variants"]:
        rows.append(
            "<div class='variant-row'>"
            f"{render_slot(item['label'], text_html(item['summary']), 'meta-slot')}"
            f"{render_slot('actual_context_video', media_html(item['context_clip']), 'context-slot')}"
            f"{render_slot('generated_output', media_html(item['generated_video']), 'output-slot')}"
            "</div>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Single-Case Context Sweep</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --accent: #b5532d;
      --accent-soft: #f3d7c9;
      --ok-soft: #d6ead9;
      --ok-ink: #28563c;
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
      width: min(1900px, calc(100vw - 20px));
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
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .shared-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .variant-stack {{
      display: grid;
      gap: 10px;
    }}
    .variant-row {{
      display: grid;
      grid-template-columns: minmax(220px, 280px) minmax(320px, 1fr) minmax(320px, 1fr);
      gap: 8px;
      padding: 10px;
      background: rgba(255,253,248,0.96);
      border: 1px solid var(--line);
      border-radius: 12px;
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
    .context-slot .slot-head {{
      background: rgba(232, 223, 208, 0.82);
      color: #584936;
    }}
    .output-slot .slot-head {{
      background: rgba(243, 215, 201, 0.78);
      color: #6e2a13;
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
      background: repeating-linear-gradient(
        45deg,
        rgba(216, 207, 191, 0.35),
        rgba(216, 207, 191, 0.35) 10px,
        rgba(255, 253, 248, 0.75) 10px,
        rgba(255, 253, 248, 0.75) 20px
      );
    }}
    @media (max-width: 1100px) {{
      .shared-grid, .variant-row {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>{html.escape(payload['sample_id'])}</h1>
      <p>{html.escape(payload['hero_text'])}</p>
    </section>
    <section class="shared-grid">
      {render_slot('source_context_full_clip', media_html(payload['full_context_clip']), 'context-slot')}
      {render_slot('gt_full_video', media_html(payload['gt_full_video']), 'context-slot')}
      {render_slot('caption', text_html(payload['caption']), 'meta-slot')}
    </section>
    <section class="variant-stack">
      {''.join(rows)}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    orig_benchmark_root = args.orig_benchmark_root.expanduser().resolve()
    meta_json_path = args.meta_json_path.expanduser().resolve()
    portal_dir = (benchmark_root / args.portal_subdir).resolve()
    assets_dir = portal_dir / "assets"
    portal_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    meta = read_json(meta_json_path)
    paths = meta.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"meta.paths must be dict: {meta_json_path}")
    sample_id = str(meta.get("sample_id") or meta_json_path.parent.name)
    dataset = str(meta_json_path.parts[meta_json_path.parts.index("mytest") - 1]) if "mytest" in meta_json_path.parts else "unknown"
    context_path = Path(paths["context_video_path"])
    gt_full_path = Path(paths["full_video_path"])
    caption = str(meta.get("caption") or "")
    source_fps = int(meta.get("fps") or 30)
    context_range = meta.get("context_frame_range") or [0, 0]
    source_context_frames = int(context_range[1]) - int(context_range[0]) + 1 if len(context_range) == 2 else 0
    height = 544
    width = 720
    resize_mode = bel.resolve_context_resize_mode(dataset)

    full_context_asset_path = assets_dir / "source_context_full_clip.mp4"
    save_context_clip(
        context_path=context_path,
        context_frames=source_context_frames,
        height=height,
        width=width,
        resize_mode=resize_mode,
        fps=source_fps,
        output_path=full_context_asset_path,
    )

    variants: list[dict[str, str]] = []
    for context_frames in [int(item.strip()) for item in args.context_lengths.split(",") if item.strip()]:
        context_clip_path = assets_dir / f"actual_context_{context_frames:02d}f.mp4"
        save_context_clip(
            context_path=context_path,
            context_frames=context_frames,
            height=height,
            width=width,
            resize_mode=resize_mode,
            fps=source_fps,
            output_path=context_clip_path,
        )

        if context_frames == 8:
            generated_path = (
                orig_benchmark_root
                / "output"
                / "VACE_1_3B_V2V"
                / "context_08f"
                / f"{dataset}__{sample_id}.mp4"
            )
        else:
            generated_path = (
                benchmark_root
                / "tools"
                / "context_sweep_case0005"
                / "generated"
                / f"context_{context_frames:02d}f"
                / f"{dataset}__{sample_id}.mp4"
            )

        variants.append(
            {
                "label": f"context_{context_frames:02d}f",
                "summary": (
                    f"used_context_frames: {context_frames}\n"
                    f"source_context_fps: {source_fps}\n"
                    f"approx_context_seconds: {context_frames / source_fps:.3f}\n"
                    f"generated_output_fps: 16\n"
                    f"generated_output_frames: 49"
                ),
                "context_clip": relative_to_root(benchmark_root, context_clip_path),
                "generated_video": expose_asset(
                    target=generated_path,
                    benchmark_root=benchmark_root,
                    assets_dir=assets_dir,
                    link_name=f"generated_context_{context_frames:02d}f.mp4",
                ),
            }
        )

    payload = {
        "sample_id": sample_id,
        "caption": caption,
        "hero_text": (
            f"Source context clip in meta.json covers {source_context_frames} frames at {source_fps} fps "
            f"({source_context_frames / source_fps:.3f} seconds), but VACE only consumes the first N frames "
            f"for each sweep setting below. The context videos shown here are the actual N source frames "
            f"that were given to the model, not the full 90-frame context clip."
        ),
        "full_context_clip": relative_to_root(benchmark_root, full_context_asset_path),
        "gt_full_video": expose_asset(
            target=gt_full_path,
            benchmark_root=benchmark_root,
            assets_dir=assets_dir,
            link_name="gt_full_video.mp4",
        ),
        "variants": variants,
    }
    html_path = portal_dir / "index.html"
    html_path.write_text(build_html(payload), encoding="utf-8")
    write_json(
        portal_dir / "build_summary.json",
        {
            "sample_id": sample_id,
            "html_path": str(html_path),
            "portal_url_path": f"/{relative_to_root(benchmark_root, html_path)}",
        },
    )
    print(json.dumps({"html_path": str(html_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
