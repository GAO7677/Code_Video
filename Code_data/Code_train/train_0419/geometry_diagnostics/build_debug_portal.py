#!/usr/bin/env python3
"""Build a tiny local HTML portal for geometry diagnostics debug outputs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML portal for geometry diagnostics.")
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--output_html", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relpath(base: Path, target: Path) -> str:
    return target.relative_to(base).as_posix()


def ensure_local_link(target: Path, case_dir: Path, name: str) -> Path:
    local_path = case_dir / name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() or local_path.is_symlink():
        local_path.unlink()
    local_path.symlink_to(target)
    return local_path


def image_html(path: Path, base: Path, label: str) -> str:
    rel = relpath(base, path)
    return (
        f"<div class='panel'>"
        f"<div class='panel-title'>{html.escape(label)}</div>"
        f"<img src='{html.escape(rel)}' alt='{html.escape(label)}'>"
        f"</div>"
    )


def video_html(path: Path, base: Path, label: str) -> str:
    rel = relpath(base, path)
    return (
        f"<div class='panel'>"
        f"<div class='panel-title'>{html.escape(label)}</div>"
        f"<video controls preload='metadata' muted playsinline>"
        f"<source src='{html.escape(rel)}' type='video/mp4'>"
        f"</video>"
        f"</div>"
    )


def render_video_group(items: list[dict[str, Any]], base: Path, title: str) -> str:
    if not items:
        return ""
    panels: list[str] = []
    for item in items:
        raw = item.get("path")
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            continue
        if not str(path).startswith(str(base)):
            continue
        label_parts = []
        if item.get("object_label"):
            label_parts.append(str(item["object_label"]))
        if item.get("classification"):
            label_parts.append(str(item["classification"]))
        if item.get("track_id") is not None:
            label_parts.append(f"track_{item['track_id']}")
        if not label_parts and item.get("seg_id") is not None:
            label_parts.append(f"seg_{item['seg_id']}")
        label = " | ".join(label_parts) or title
        panels.append(video_html(path, base, label))
    if not panels:
        return ""
    return (
        f"<div class='analysis'><div class='analysis-title'>{html.escape(title)}</div></div>"
        f"<div class='panel-grid'>{''.join(panels)}</div>"
    )


def render_analysis(lines: list[str]) -> str:
    if not lines:
        return ""
    parts = "".join(f"<li>{html.escape(str(line))}</li>" for line in lines)
    return f"<div class='analysis'><div class='analysis-title'>Analysis</div><ul>{parts}</ul></div>"


def render_case(case_dir: Path, run_root: Path) -> str:
    diagnostics = read_json(case_dir / "diagnostics.json")
    summary = diagnostics.get("summary", {})
    target = diagnostics.get("target", {})
    gt_reference = diagnostics.get("gt_reference", {})
    artifacts = diagnostics.get("artifacts", {})
    analysis = diagnostics.get("analysis", [])

    panels = []
    for key, label in [
        ("context_video", "Context Video"),
        ("generated_video", "Generated Video"),
        ("full_gt_video", "Full GT Video"),
        ("context_diagnostic_video", "Context Overlay"),
        ("generated_diagnostic_video", "Generated Diagnostic Overlay"),
        ("gt_diagnostic_video", "GT Diagnostic Overlay"),
    ]:
        raw = artifacts.get(key)
        if raw:
            path = Path(raw)
            if path.exists():
                if not str(path).startswith(str(run_root)):
                    local_name = {
                        "context_video": "context_video.mp4",
                        "generated_video": "generated_video.mp4",
                        "full_gt_video": "full_gt_video.mp4",
                    }.get(key, path.name)
                    path = ensure_local_link(path, case_dir, local_name)
                panels.append(video_html(path, run_root, label))
    curves_png = case_dir / "curves.png"
    gt_curves_png = case_dir / "gt_curves.png"
    comparison_curves_png = case_dir / "comparison_curves.png"
    if comparison_curves_png.exists():
        panels.append(image_html(comparison_curves_png, run_root, "GT vs Generated Curves"))
    if curves_png.exists():
        panels.append(image_html(curves_png, run_root, "Generated Curves"))
    if gt_curves_png.exists():
        panels.append(image_html(gt_curves_png, run_root, "GT Reference Curves"))

    grouped_videos_html = "".join(
        [
            render_video_group(list(artifacts.get("context_single_track_videos") or []), run_root, "Context Single Tracks"),
            render_video_group(list(artifacts.get("generated_single_track_videos") or []), run_root, "Generated Known Tracks"),
            render_video_group(list(artifacts.get("generated_born_single_track_videos") or []), run_root, "Generated Born Tracks"),
        ]
    )

    return f"""
    <section class="case-card">
      <div class="head">
        <div>
          <h2>{html.escape(str(diagnostics.get("dataset") or ""))}</h2>
          <div class="sample">{html.escape(str(diagnostics.get("sample_id") or ""))}</div>
        </div>
        <div class="badge">{html.escape(str(summary.get("root_cause") or ""))}</div>
      </div>
      <div class="meta-grid">
        <div><span>Mode</span><strong>{html.escape(str(diagnostics.get("mode") or ""))}</strong></div>
        <div><span>Target</span><strong>{html.escape(str(target.get("object_label") or ""))}</strong></div>
        <div><span>Selection</span><strong>{html.escape(str(target.get("selection_mode") or ""))}</strong></div>
        <div><span>Target Visible</span><strong>{html.escape(str(summary.get("target_visible_ratio") or ""))}</strong></div>
        <div><span>Max Area Ratio</span><strong>{html.escape(str(summary.get("max_future_mask_area_ratio") or ""))}</strong></div>
        <div><span>Max Invariant Ratio</span><strong>{html.escape(str(summary.get("max_future_area_depth2_ratio") or ""))}</strong></div>
        <div><span>Mean BG Scale</span><strong>{html.escape(str(summary.get("mean_bg_scale") or ""))}</strong></div>
        <div><span>GT Frames</span><strong>{html.escape(str(gt_reference.get("num_frames_compared") or ""))}</strong></div>
      </div>
      {render_analysis(list(analysis))}
      <div class="panel-grid">
        {''.join(panels)}
      </div>
      {grouped_videos_html}
      <details>
        <summary>Diagnostics JSON</summary>
        <pre>{html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2))}</pre>
      </details>
    </section>
    """


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve() if args.output_html else run_root / "index.html"
    case_dirs = sorted(path for path in run_root.iterdir() if path.is_dir() and (path / "diagnostics.json").exists())
    cards = [render_case(case_dir, run_root) for case_dir in case_dirs]
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Geometry Diagnostics Debug Portal</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --card: #fffdf9;
      --ink: #1e1a17;
      --muted: #6e645a;
      --line: #d8ccbf;
      --accent: #b64826;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(182,72,38,0.10), transparent 28%),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1200px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 38px;
      line-height: 1.05;
    }}
    .lead {{
      color: var(--muted);
      margin-bottom: 24px;
      font-size: 16px;
    }}
    .case-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 12px 30px rgba(32, 24, 17, 0.06);
    }}
    .head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .head h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .sample {{
      color: var(--muted);
      margin-top: 4px;
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    .badge {{
      background: var(--accent);
      color: white;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      white-space: nowrap;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .meta-grid div {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fffaf3;
    }}
    .meta-grid span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .meta-grid strong {{
      font-size: 14px;
      overflow-wrap: anywhere;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      margin-bottom: 12px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: white;
    }}
    .panel-title {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 10px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 10px;
      background: #120f0d;
    }}
    .analysis {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fffaf3;
      padding: 12px 14px;
      margin-bottom: 16px;
    }}
    .analysis-title {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .analysis ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .analysis li {{
      margin: 6px 0;
      line-height: 1.45;
    }}
    details {{
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f6f1ea;
      border-radius: 12px;
      padding: 12px;
      font-size: 12px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Geometry Diagnostics</h1>
    <div class="lead">Debug portal for the current local run. Generated curves currently use a target-window foreground proxy, while GT reference curves and GT overlays use synthetic segmentation/depth where available.</div>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    output_html.write_text(page, encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
