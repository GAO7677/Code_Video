#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


CSS = """
:root {
  --bg: #f5f1e8;
  --card: rgba(255, 252, 246, 0.95);
  --ink: #171311;
  --muted: #6e655d;
  --line: #ddd2c6;
  --accent: #0d6b63;
  --shadow: 0 16px 36px rgba(44, 29, 17, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top right, rgba(13,107,99,0.12), transparent 24%),
    radial-gradient(circle at left top, rgba(198,114,58,0.08), transparent 20%),
    linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
}
.wrap { width: min(1600px, calc(100vw - 28px)); margin: 0 auto; padding: 28px 0 60px; }
h1 { margin: 0; font-size: clamp(34px, 4vw, 56px); line-height: 0.96; letter-spacing: -0.04em; }
.sub { margin: 8px 0 24px; color: var(--muted); font-size: 18px; }
.meta { display:flex; flex-wrap:wrap; gap:10px; margin-bottom: 28px; }
.pill { border:1px solid var(--line); border-radius:999px; padding:8px 14px; background:rgba(255,255,255,0.7); font-size:13px; }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }
.card { background: var(--card); border:1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow); overflow:hidden; }
.head { padding: 18px 18px 10px; border-bottom:1px solid var(--line); }
.title { margin: 0 0 6px; font-size: 24px; }
.prompt { margin: 0; color: var(--muted); font-size: 14px; line-height: 1.45; }
.body { padding: 16px 18px 20px; }
.tags { display:flex; flex-wrap:wrap; gap: 8px; margin-bottom: 14px; }
.tag { font-size:12px; border:1px solid var(--line); border-radius:999px; padding:6px 10px; background: rgba(255,255,255,0.74); }
.panel { border:1px solid var(--line); border-radius: 16px; overflow:hidden; background:#fffdf9; margin-bottom: 12px; }
.panel h4 { margin:0; padding:10px 12px; font-size:14px; background: rgba(13,107,99,0.10); border-bottom:1px solid var(--line); }
.panel video { display:block; width:100%; aspect-ratio:16 / 9; background:#101010; }
.panel img { display:block; width:100%; height:auto; }
.energy { font-size: 30px; font-weight: 700; letter-spacing: -0.03em; color: var(--accent); margin: 0 0 10px; }
.path { color: var(--muted); font-size: 12px; word-break: break-all; font-family: "SFMono-Regular", Consolas, monospace; margin-top: 8px; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sweep viewer over explicit target scheduler timesteps.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--title", type=str, default="V-JEPA Target Timestep Sweep")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_runs(root: Path) -> list[dict]:
    runs: list[dict] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_root = run_dir / "trace"
        if not trace_root.is_dir():
            continue
        case_dirs = sorted(path for path in trace_root.iterdir() if path.is_dir())
        if not case_dirs:
            continue
        case_dir = case_dirs[0]
        case_meta = load_json(case_dir / "case.json")
        step_dirs = sorted(path for path in case_dir.iterdir() if path.is_dir() and path.name.startswith("step_"))
        if not step_dirs:
            continue
        step_dir = step_dirs[0]
        step_stats = load_json(step_dir / "stats.json")
        runs.append(
            {
                "run_dir": run_dir,
                "case_dir": case_dir,
                "case_meta": case_meta,
                "step_dir": step_dir,
                "stats": step_stats,
                "preview_video": step_dir / "preview_video.mp4",
                "preview_strip": step_dir / "preview_strip.png",
                "latent_video": step_dir / "x0_latent_norm.mp4",
                "latent_strip": step_dir / "x0_latent_norm_strip.png",
                "final_video": case_dir / "final_video.mp4",
            }
        )
    runs.sort(key=lambda item: int(item["stats"].get("timestep", 10**9)), reverse=True)
    return runs


def build_html(root: Path, output_html: Path, title: str, runs: list[dict]) -> None:
    prompt = ""
    if runs:
        prompt = str(runs[0]["case_meta"].get("prompt") or "")
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
        "<p class='sub'>One guidance step per run, explicit target scheduler timestep sweep. Each card shows the intermediate preview decode, x0 latent-norm view, and final generated video.</p>",
        "<div class='meta'>",
        f"<div class='pill'>root: {html.escape(str(root))}</div>",
        f"<div class='pill'>runs: {len(runs)}</div>",
    ]
    if prompt:
        parts.append(f"<div class='pill'>prompt: {html.escape(prompt)}</div>")
    parts.extend(["</div>", "<div class='grid'>"])

    for run in runs:
        stats = run["stats"]
        parts.append("<section class='card'>")
        parts.append("<div class='head'>")
        parts.append(
            f"<h2 class='title'>target timestep {int(stats.get('timestep', 0))}</h2>"
        )
        parts.append(
            f"<p class='prompt'>{html.escape(str(run['case_meta'].get('sample_id') or run['case_dir'].name))}</p>"
        )
        parts.append("</div>")
        parts.append("<div class='body'>")
        parts.append(f"<div class='energy'>E = {float(stats.get('energy', 0.0)):.4f}</div>")
        parts.append("<div class='tags'>")
        parts.append(f"<div class='tag'>step_idx: {int(stats.get('step_idx', 0))}</div>")
        parts.append(f"<div class='tag'>raw_grad_norm: {float(stats.get('raw_grad_norm', 0.0)):.4f}</div>")
        parts.append(f"<div class='tag'>preview: {int(stats.get('preview_frames', 0))}f</div>")
        parts.append("</div>")
        if run["preview_strip"].exists():
            parts.append("<div class='panel'><h4>Preview Decode Strip</h4>")
            parts.append(f"<img src='{html.escape(rel(root, run['preview_strip']))}' alt='preview strip'></div>")
        if run["preview_video"].exists():
            parts.append("<div class='panel'><h4>Preview Decode Video</h4>")
            parts.append(f"<video controls preload='metadata' src='{html.escape(rel(root, run['preview_video']))}'></video></div>")
        if run["latent_strip"].exists():
            parts.append("<div class='panel'><h4>x0 Pred Latent-Norm Strip</h4>")
            parts.append(f"<img src='{html.escape(rel(root, run['latent_strip']))}' alt='latent strip'></div>")
        if run["final_video"].exists():
            parts.append("<div class='panel'><h4>Final Video</h4>")
            parts.append(f"<video controls preload='metadata' src='{html.escape(rel(root, run['final_video']))}'></video></div>")
        parts.append(f"<div class='path'>{html.escape(str(run['run_dir']))}</div>")
        parts.append("</div></section>")

    parts.extend(["</div>", "</div>", "</body>", "</html>"])
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve() if args.output_html else root / "index.html"
    runs = collect_runs(root)
    build_html(root, output_html, args.title, runs)
    print(output_html)


if __name__ == "__main__":
    main()
