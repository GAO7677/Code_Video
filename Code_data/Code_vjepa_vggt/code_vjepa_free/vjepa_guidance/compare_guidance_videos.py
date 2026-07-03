#!/usr/bin/env python3
"""
Build a side-by-side visual comparison of the different guidance schemes.

For each selected video it:
  - extracts N evenly-spaced frames into a horizontal filmstrip PNG
  - embeds the mp4 (for playback) + the filmstrip (for frame-by-frame motion)
into one self-contained HTML page served over HTTP.

The point: energy is only a proxy. This lets us eyeball whether the corrected
small-step / dense anchored guidance actually improves physical motion vs the
baseline and the old over-large-step runs.
"""
from __future__ import annotations

import argparse
import base64
import subprocess
from pathlib import Path

FFMPEG = "/home/gaoya/.local/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

PROBE = Path("/data/gaoya/agent-data/outputs/probe_sweep")

# (label, description, path) -- ordered for the comparison narrative.
VIDEOS = [
    ("baseline", "无引导 (纯扩散)", PROBE / "phase4/videos/baseline.mp4"),
    ("anch_p50_s02", "旧: 单步 step=0.02 (过大, 能量上升)", PROBE / "phase4/videos/anch_p50_s02.mp4"),
    ("anch_single_s005", "修正: 单步 step=0.005 (落在下降basin)", PROBE / "phase4/videos/anch_single_s005.mp4"),
    ("anch_dense6_s005", "连续6步 step=0.005", PROBE / "phase4/videos/anch_dense6_s005.mp4"),
    ("anch_dense12_s005", "连续12步 step=0.005", PROBE / "phase4/videos/anch_dense12_s005.mp4"),
    ("anch_dense20_s003", "连续20步 step=0.003", PROBE / "phase4/videos/anch_dense20_s003.mp4"),
    ("anch_dense12_bt", "连续12步 + 自动步长(backtracking)", PROBE / "phase4/videos/anch_dense12_bt.mp4"),
    ("anch_dense20_bt", "连续20步 + 自动步长(backtracking)", PROBE / "phase4/videos/anch_dense20_bt.mp4"),
]

STRIP_FRAMES = 8


def extract_strip(video: Path, out_png: Path, n: int = STRIP_FRAMES) -> bool:
    """Extract n evenly-spaced frames tiled horizontally into out_png."""
    if not video.is_file():
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # select n frames spread across the clip, tile 1 row.
    vf = (
        f"select='not(mod(n\\,{max(1,49//n)}))',"
        f"scale=208:120,tile={n}x1:padding=4:color=white"
    )
    cmd = [FFMPEG, "-y", "-i", str(video), "-vf", vf, "-frames:v", "1", str(out_png)]
    r = subprocess.run(cmd, capture_output=True)
    return out_png.is_file() and r.returncode == 0


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_html(items: list[dict], out_html: Path) -> None:
    cards = []
    for it in items:
        strip_tag = (
            f'<img class="strip" src="data:image/png;base64,{it["strip_b64"]}"/>'
            if it.get("strip_b64") else '<div class="missing">帧条提取失败</div>'
        )
        video_tag = (
            f'<video controls loop muted preload="metadata" width="520">'
            f'<source src="data:video/mp4;base64,{it["mp4_b64"]}" type="video/mp4"></video>'
            if it.get("mp4_b64") else '<div class="missing">视频缺失(可能仍在生成)</div>'
        )
        cards.append(f"""
        <div class="card">
          <div class="hd"><span class="lbl">{it['label']}</span><span class="desc">{it['desc']}</span></div>
          {video_tag}
          <div class="striprow">{strip_tag}</div>
        </div>""")

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>V-JEPA Guidance 视频对比</title>
<style>
  body {{ background:#f7f4ee; color:#222; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:20px; font-weight:600; }}
  .note {{ color:#555; font-size:13px; line-height:1.6; max-width:900px; margin-bottom:20px; }}
  .card {{ background:#fff; border:1px solid #e3ddd2; border-radius:10px; padding:14px 16px; margin-bottom:18px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }}
  .hd {{ display:flex; align-items:baseline; gap:12px; margin-bottom:10px; }}
  .lbl {{ font-weight:600; font-size:15px; font-family:ui-monospace,monospace; }}
  .desc {{ color:#666; font-size:13px; }}
  video {{ display:block; border-radius:6px; background:#000; }}
  .striprow {{ margin-top:10px; overflow-x:auto; }}
  .strip {{ display:block; border-radius:4px; }}
  .missing {{ color:#b00; font-size:13px; padding:8px 0; }}
</style></head><body>
<h1>V-JEPA Guidance 视频对比 — case 025_Solid_Mechanics_0002</h1>
<div class="note">
上排视频可播放；下排为均匀抽取的 {STRIP_FRAMES} 帧帧条，用于逐帧比对物理运动是否更自然。<br>
重点看: (1) <b>anch_p50_s02</b> 旧的过大步长是否引入 artifact / 崩坏; (2) <b>anch_single_s005</b> 修正后是否与 baseline 接近但更"物理"; (3) <b>dense*</b> 连续引导是否在保持画质的前提下改善运动，而非退化。
</div>
{''.join(cards)}
</body></html>"""
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PROBE / "compare")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for label, desc, path in VIDEOS:
        strip_png = args.out_dir / f"{label}_strip.png"
        entry = {"label": label, "desc": desc}
        if path.is_file():
            ok = extract_strip(path, strip_png)
            entry["strip_b64"] = b64(strip_png) if ok else None
            entry["mp4_b64"] = b64(path)
            print(f"[ok]   {label}: video + strip={'ok' if ok else 'FAIL'}")
        else:
            entry["strip_b64"] = None
            entry["mp4_b64"] = None
            print(f"[skip] {label}: {path} not found (still generating?)")
        items.append(entry)

    out_html = args.out_dir / "compare.html"
    build_html(items, out_html)
    print(f"\nHTML written: {out_html}  ({out_html.stat().st_size//1024} KB)")

    if args.serve:
        import http.server, socketserver, os
        os.chdir(args.out_dir)
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("0.0.0.0", args.port), handler) as httpd:
            print(f"Serving at http://localhost:{args.port}/compare.html  (Ctrl-C to stop)")
            httpd.serve_forever()


if __name__ == "__main__":
    main()
