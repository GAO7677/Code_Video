#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


CSS = """
:root {
  --bg: #f4efe8;
  --card: rgba(255, 252, 246, 0.92);
  --ink: #181412;
  --muted: #6f655c;
  --line: #d9cfc3;
  --accent: #0d6b63;
  --accent-soft: rgba(13, 107, 99, 0.10);
  --shadow: 0 16px 36px rgba(44, 29, 17, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
  background:
    radial-gradient(circle at top right, rgba(13, 107, 99, 0.14), transparent 24%),
    radial-gradient(circle at left top, rgba(198, 114, 58, 0.10), transparent 18%),
    linear-gradient(180deg, #f9f5ee 0%, var(--bg) 100%);
}
.wrap {
  width: min(1500px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0 56px;
}
h1 {
  margin: 0;
  font-size: clamp(34px, 4vw, 58px);
  line-height: 0.96;
  letter-spacing: -0.04em;
}
.sub {
  margin: 10px 0 24px;
  color: var(--muted);
  font-size: 18px;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 28px;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 14px;
  background: rgba(255,255,255,0.65);
  font-size: 13px;
}
.case-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 18px;
}
.case {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.case-header {
  padding: 18px 18px 12px;
  border-bottom: 1px solid var(--line);
}
.case-title {
  margin: 0 0 6px;
  font-size: 24px;
  line-height: 1.05;
}
.prompt {
  color: var(--muted);
  font-size: 15px;
  line-height: 1.45;
  margin: 0;
}
.case-body {
  padding: 16px 18px 20px;
}
.kv {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 14px;
}
.tag {
  font-size: 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.7);
}
.video {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #151515;
  border-radius: 16px;
}
.steps {
  display: grid;
  gap: 14px;
  margin-top: 16px;
}
.step {
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(250,247,242,0.88));
}
.step-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
}
.step-title {
  margin: 0;
  font-size: 18px;
}
.energy {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--accent);
}
.mini-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}
.mini-card {
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
  background: #fffdf9;
}
.mini-card h4 {
  margin: 0;
  padding: 10px 12px;
  font-size: 14px;
  border-bottom: 1px solid var(--line);
  background: var(--accent-soft);
}
.mini-card img {
  display: block;
  width: 100%;
  height: auto;
}
.mini-card video {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #111;
}
.path {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
  word-break: break-all;
  font-family: "SFMono-Regular", Consolas, monospace;
}
@media (max-width: 720px) {
  .wrap { width: min(100vw - 18px, 1500px); padding-top: 18px; }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML viewer for V-JEPA guidance traces.")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--title", type=str, default="V-JEPA Guidance Trace Viewer")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(from_dir: Path, target: Path) -> str:
    return target.relative_to(from_dir).as_posix()


def collect_cases(trace_root: Path) -> list[dict]:
    cases: list[dict] = []
    for case_dir in sorted(path for path in trace_root.iterdir() if path.is_dir()):
        case_json = case_dir / "case.json"
        if not case_json.exists():
            continue
        case_meta = load_json(case_json)
        steps = []
        for step_dir in sorted(path for path in case_dir.iterdir() if path.is_dir() and path.name.startswith("step_")):
            stats_path = step_dir / "stats.json"
            if not stats_path.exists():
                continue
            stats = load_json(stats_path)
            steps.append(
                {
                    "dir": step_dir,
                    "stats": stats,
                    "preview_video": step_dir / "preview_video.mp4",
                    "preview_strip": step_dir / "preview_strip.png",
                    "latent_video": step_dir / "x0_latent_norm.mp4",
                    "latent_strip": step_dir / "x0_latent_norm_strip.png",
                }
            )
        cases.append(
            {
                "dir": case_dir,
                "meta": case_meta,
                "steps": steps,
                "final_video": case_dir / "final_video.mp4",
            }
        )
    return cases


def build_html(trace_root: Path, output_html: Path, title: str, cases: list[dict]) -> None:
    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{CSS}</style>",
        "</head>",
        "<body>",
        "<div class='wrap'>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='sub'>Per-case guidance trace with final video, low-res preview decode, x0 latent-norm view, and V-JEPA surprise at each guidance step.</p>",
        "<div class='meta'>",
        f"<div class='pill'>trace root: {html.escape(str(trace_root))}</div>",
        f"<div class='pill'>cases: {len(cases)}</div>",
        "</div>",
        "<div class='case-grid'>",
    ]

    for case in cases:
        meta = case["meta"]
        sample_id = str(meta.get("sample_id") or case["dir"].name)
        prompt = str(meta.get("prompt") or "")
        final_video = case["final_video"]
        parts.append("<section class='case'>")
        parts.append("<div class='case-header'>")
        parts.append(f"<h2 class='case-title'>{html.escape(sample_id)}</h2>")
        if prompt:
            parts.append(f"<p class='prompt'>{html.escape(prompt)}</p>")
        parts.append("</div>")
        parts.append("<div class='case-body'>")
        parts.append("<div class='kv'>")
        if "source_json" in meta and meta["source_json"]:
            parts.append(f"<div class='tag'>source: {html.escape(str(meta['source_json']))}</div>")
        if "selected_guidance_steps" in meta:
            parts.append(
                f"<div class='tag'>selected steps: {html.escape(', '.join(str(v) for v in meta['selected_guidance_steps']))}</div>"
            )
        if "fps" in meta:
            parts.append(f"<div class='tag'>fps: {html.escape(str(meta['fps']))}</div>")
        parts.append("</div>")
        if final_video.exists():
            parts.append(
                f"<video class='video' controls preload='metadata' src='{html.escape(rel(trace_root, final_video))}'></video>"
            )
        for step in case["steps"]:
            stats = step["stats"]
            parts.append("<div class='steps'>")
            parts.append("<article class='step'>")
            parts.append("<div class='step-head'>")
            parts.append(
                f"<h3 class='step-title'>Guidance Step {int(stats.get('step_idx', 0))} · t={int(stats.get('timestep', 0))}</h3>"
            )
            parts.append(f"<div class='energy'>{float(stats.get('energy', 0.0)):.4f}</div>")
            parts.append("</div>")
            parts.append("<div class='kv'>")
            parts.append(f"<div class='tag'>raw_grad_norm: {float(stats.get('raw_grad_norm', 0.0)):.4f}</div>")
            parts.append(
                f"<div class='tag'>normalized_grad_rms: {float(stats.get('normalized_grad_rms', 0.0)):.4f}</div>"
            )
            parts.append(
                f"<div class='tag'>preview: {int(stats.get('preview_frames', 0))}f · {int(stats.get('preview_width', 0))}x{int(stats.get('preview_height', 0))}</div>"
            )
            parts.append("</div>")
            parts.append("<div class='mini-grid'>")
            if step["preview_strip"].exists():
                parts.append("<div class='mini-card'>")
                parts.append("<h4>Preview Decode Strip</h4>")
                parts.append(
                    f"<img src='{html.escape(rel(trace_root, step['preview_strip']))}' alt='Preview strip for {html.escape(sample_id)}'>"
                )
                parts.append("</div>")
            if step["preview_video"].exists():
                parts.append("<div class='mini-card'>")
                parts.append("<h4>Preview Decode Video</h4>")
                parts.append(
                    f"<video controls preload='metadata' src='{html.escape(rel(trace_root, step['preview_video']))}'></video>"
                )
                parts.append("</div>")
            if step["latent_strip"].exists():
                parts.append("<div class='mini-card'>")
                parts.append("<h4>x0 Pred Latent-Norm Strip</h4>")
                parts.append(
                    f"<img src='{html.escape(rel(trace_root, step['latent_strip']))}' alt='x0 latent strip for {html.escape(sample_id)}'>"
                )
                parts.append("</div>")
            if step["latent_video"].exists():
                parts.append("<div class='mini-card'>")
                parts.append("<h4>x0 Pred Latent-Norm Video</h4>")
                parts.append(
                    f"<video controls preload='metadata' src='{html.escape(rel(trace_root, step['latent_video']))}'></video>"
                )
                parts.append("</div>")
            parts.append("</div>")
            parts.append(
                f"<div class='path'>{html.escape(str(step['dir']))}</div>"
            )
            parts.append("</article>")
            parts.append("</div>")
        parts.append("</div>")
        parts.append("</section>")

    parts.extend(["</div>", "</div>", "</body>", "</html>"])
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    trace_root = args.trace_root.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve() if args.output_html else trace_root / "index.html"
    cases = collect_cases(trace_root)
    build_html(trace_root, output_html, args.title, cases)
    print(output_html)


if __name__ == "__main__":
    main()
