#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a local web gallery for Genesis liquid H5 datasets.

Features:
- Scan a directory of scene_*.h5 files
- Export RGB previews as mp4
- Export depth previews as grayscale mp4
- Write an interactive single-page gallery
- Optionally serve the gallery on a local HTTP port

Example:

/data/gaoya/miniconda3/envs/wan/bin/python /home/gaoya/Code_Video/Code_data/liquid_h5_gallery.py \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/liquid_dataset_h5_fixcheck2 \
  --output-root /home/gaoya/Code_Video/Code_data/liquid_h5_gallery_fixcheck2 \
  --host 0.0.0.0 \
  --port 8023 \
  --serve
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import posixpath
import urllib.parse
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import h5py
except Exception:
    h5py = None

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    import numpy as np
except Exception:
    np = None


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "liquid_h5_gallery"


@dataclass
class ScenePreview:
    scene_name: str
    h5_path: Path
    rgb_relpath: str
    depth_relpath: str
    metadata: Dict[str, Any]


def ensure_dependencies() -> None:
    missing = []
    if h5py is None:
        missing.append("h5py")
    if imageio is None:
        missing.append("imageio")
    if np is None:
        missing.append("numpy")
    if missing:
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + ". Please install them in the runtime environment first."
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def scan_h5_files(dataset_root: Path) -> List[Path]:
    return sorted(path for path in dataset_root.glob("*.h5") if path.is_file())


def load_scene_data(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        rgb = np.asarray(h5["/rgb"], dtype=np.uint8)
        depth = np.asarray(h5["/depth"], dtype=np.float32)
        attrs = {str(k): to_jsonable(v) for k, v in h5.attrs.items()}
    return {
        "rgb": rgb,
        "depth": depth,
        "attrs": attrs,
    }


def depth_to_rgb_video(depth_frames: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_frames, dtype=np.float32)
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    depth = np.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0)
    depth = np.clip(depth, 0.0, 1.0)
    gray = np.round(depth[..., 0] * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def write_mp4(video_path: Path, frames: np.ndarray, fps: int) -> None:
    ensure_dir(video_path.parent)
    imageio.mimwrite(
        video_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def export_scene_preview(h5_path: Path, output_root: Path, force: bool = False) -> ScenePreview:
    scene_name = h5_path.stem
    scene_dir = output_root / "media" / scene_name
    rgb_path = scene_dir / "rgb.mp4"
    depth_path = scene_dir / "depth.mp4"
    meta_path = scene_dir / "metadata.json"

    data = load_scene_data(h5_path)
    fps = int(data["attrs"].get("fps", 24))

    if force or not rgb_path.exists():
        write_mp4(rgb_path, data["rgb"], fps=fps)

    if force or not depth_path.exists():
        depth_rgb = depth_to_rgb_video(data["depth"])
        write_mp4(depth_path, depth_rgb, fps=fps)

    metadata = {
        "scene_name": scene_name,
        "source_h5": str(h5_path),
        "fps": fps,
        "frames": int(data["rgb"].shape[0]),
        "resolution": [int(data["rgb"].shape[2]), int(data["rgb"].shape[1])],
        "attrs": data["attrs"],
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return ScenePreview(
        scene_name=scene_name,
        h5_path=h5_path,
        rgb_relpath=str(rgb_path.relative_to(output_root).as_posix()),
        depth_relpath=str(depth_path.relative_to(output_root).as_posix()),
        metadata=metadata,
    )


def build_manifest(previews: List[ScenePreview]) -> Dict[str, Any]:
    items = []
    for preview in previews:
        attrs = preview.metadata["attrs"]
        items.append(
            {
                "scene_name": preview.scene_name,
                "rgb_url": preview.rgb_relpath,
                "depth_url": preview.depth_relpath,
                "source_h5": str(preview.h5_path),
                "fps": preview.metadata["fps"],
                "frames": preview.metadata["frames"],
                "resolution": preview.metadata["resolution"],
                "container_label": attrs.get("container_label", ""),
                "container_source_id": attrs.get("container_source_id", ""),
                "solver": attrs.get("solver", ""),
                "viscosity": attrs.get("viscosity", ""),
                "emitter_speed": attrs.get("emitter_speed", ""),
                "metadata": preview.metadata,
            }
        )
    return {"items": items}


def build_index_html(title: str) -> str:
    safe_title = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: rgba(255, 252, 245, 0.88);
      --ink: #17202a;
      --muted: #5f6f7f;
      --line: rgba(23, 32, 42, 0.12);
      --accent: #006d77;
      --accent-soft: #d7efe7;
      --shadow: 0 20px 45px rgba(24, 37, 54, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.80), transparent 30%),
        radial-gradient(circle at top right, rgba(0, 109, 119, 0.10), transparent 28%),
        linear-gradient(135deg, #f6f1e7 0%, #ece5d8 100%);
      font-family: Georgia, "Times New Roman", "Noto Serif SC", serif;
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px 1fr;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      background: rgba(255, 249, 240, 0.84);
      backdrop-filter: blur(12px);
      padding: 22px 18px;
      overflow: auto;
    }}
    .brand {{
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
      font-weight: 700;
    }}
    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .scene-list {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .scene-btn {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      border-radius: 16px;
      padding: 12px 14px;
      cursor: pointer;
      transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
      box-shadow: 0 8px 22px rgba(24, 37, 54, 0.04);
    }}
    .scene-btn:hover {{
      transform: translateY(-1px);
      box-shadow: 0 14px 28px rgba(24, 37, 54, 0.10);
      border-color: rgba(0, 109, 119, 0.22);
    }}
    .scene-btn.active {{
      border-color: rgba(0, 109, 119, 0.35);
      background: linear-gradient(135deg, #f9fffd 0%, var(--accent-soft) 100%);
      box-shadow: 0 14px 30px rgba(0, 109, 119, 0.14);
    }}
    .scene-btn .name {{
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 5px;
    }}
    .scene-btn .meta {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .main {{
      padding: 28px;
    }}
    .hero {{
      display: grid;
      gap: 20px;
      grid-template-columns: minmax(0, 1fr) 340px;
      align-items: start;
    }}
    .player-card, .meta-card {{
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.55);
      border-radius: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .player-card {{
      padding: 18px;
    }}
    .meta-card {{
      padding: 20px;
    }}
    .player-top {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }}
    .player-title {{
      font-size: 24px;
      font-weight: 700;
    }}
    .switches {{
      display: inline-flex;
      background: rgba(0, 0, 0, 0.04);
      border-radius: 999px;
      padding: 4px;
      gap: 4px;
    }}
    .switches button {{
      border: none;
      border-radius: 999px;
      padding: 8px 14px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
    }}
    .switches button.active {{
      background: var(--accent);
      color: #ffffff;
    }}
    video {{
      display: block;
      width: 100%;
      border-radius: 18px;
      background: #000;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.55);
    }}
    .stat .k {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat .v {{
      margin-top: 5px;
      font-size: 15px;
      font-weight: 700;
      word-break: break-word;
    }}
    .meta-card h2 {{
      margin: 0 0 12px 0;
      font-size: 18px;
    }}
    pre {{
      margin: 0;
      max-height: 68vh;
      overflow: auto;
      background: rgba(255,255,255,0.66);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      color: #21313d;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .empty {{
      padding: 40px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px dashed var(--line);
      border-radius: 18px;
      color: var(--muted);
    }}
    @media (max-width: 1100px) {{
      .shell {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        border-right: none;
        border-bottom: 1px solid var(--line);
      }}
      .hero {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="eyebrow">Liquid H5 Gallery</div>
        <h1>{safe_title}</h1>
        <div class="subtitle">左侧点选场景，右侧播放 RGB 或 Depth 预览视频，并查看该场景的 H5 元数据。</div>
      </div>
      <div id="scene-list" class="scene-list"></div>
    </aside>

    <main class="main">
      <div id="empty" class="empty">正在加载视频清单...</div>
      <div id="content" class="hero" style="display:none;">
        <section class="player-card">
          <div class="player-top">
            <div>
              <div id="player-title" class="player-title"></div>
            </div>
            <div class="switches">
              <button id="rgb-btn" type="button" class="active">RGB</button>
              <button id="depth-btn" type="button">Depth</button>
            </div>
          </div>
          <video id="player" controls preload="metadata" playsinline></video>
          <div class="stat-grid">
            <div class="stat"><div class="k">Container</div><div id="stat-container" class="v">-</div></div>
            <div class="stat"><div class="k">Solver</div><div id="stat-solver" class="v">-</div></div>
            <div class="stat"><div class="k">FPS / Frames</div><div id="stat-fps" class="v">-</div></div>
            <div class="stat"><div class="k">Viscosity</div><div id="stat-viscosity" class="v">-</div></div>
            <div class="stat"><div class="k">Emitter Speed</div><div id="stat-emitter" class="v">-</div></div>
            <div class="stat"><div class="k">Source H5</div><div id="stat-source" class="v">-</div></div>
          </div>
        </section>
        <aside class="meta-card">
          <h2>Metadata</h2>
          <pre id="meta-json"></pre>
        </aside>
      </div>
    </main>
  </div>

  <script>
    let manifest = [];
    let selectedIndex = 0;
    let selectedMode = "rgb";

    function qs(id) {{
      return document.getElementById(id);
    }}

    function fmtValue(v) {{
      if (v === null || v === undefined || v === "") return "-";
      if (typeof v === "number") {{
        if (Number.isInteger(v)) return String(v);
        return v.toFixed(6).replace(/0+$/, "").replace(/\\.$/, "");
      }}
      if (Array.isArray(v)) return JSON.stringify(v);
      return String(v);
    }}

    function renderList() {{
      const root = qs("scene-list");
      root.innerHTML = "";
      manifest.forEach((item, idx) => {{
        const btn = document.createElement("button");
        btn.className = "scene-btn" + (idx === selectedIndex ? " active" : "");
        btn.type = "button";
        btn.innerHTML = `
          <div class="name">${{item.scene_name}}</div>
          <div class="meta">container: ${{item.container_label || "-"}}</div>
          <div class="meta">solver: ${{item.solver || "-"}} | fps: ${{item.fps}}</div>
        `;
        btn.addEventListener("click", () => {{
          selectedIndex = idx;
          renderList();
          renderPlayer();
        }});
        root.appendChild(btn);
      }});
    }}

    function updateModeButtons() {{
      qs("rgb-btn").classList.toggle("active", selectedMode === "rgb");
      qs("depth-btn").classList.toggle("active", selectedMode === "depth");
    }}

    function renderPlayer() {{
      if (!manifest.length) return;
      const item = manifest[selectedIndex];
      const player = qs("player");
      const sourceUrl = selectedMode === "rgb" ? item.rgb_url : item.depth_url;
      player.src = encodeURI(sourceUrl);
      player.load();

      qs("player-title").textContent = item.scene_name;
      qs("stat-container").textContent = fmtValue(item.container_label);
      qs("stat-solver").textContent = fmtValue(item.solver);
      qs("stat-fps").textContent = `${{fmtValue(item.fps)}} / ${{fmtValue(item.frames)}}`;
      qs("stat-viscosity").textContent = fmtValue(item.viscosity);
      qs("stat-emitter").textContent = fmtValue(item.emitter_speed);
      qs("stat-source").textContent = fmtValue(item.source_h5);
      qs("meta-json").textContent = JSON.stringify(item.metadata, null, 2);

      qs("empty").style.display = "none";
      qs("content").style.display = "grid";
      updateModeButtons();
    }}

    async function boot() {{
      const resp = await fetch("./manifest.json");
      const data = await resp.json();
      manifest = data.items || [];
      if (!manifest.length) {{
        qs("empty").textContent = "没有找到可展示的场景。";
        return;
      }}
      renderList();
      renderPlayer();
    }}

    qs("rgb-btn").addEventListener("click", () => {{
      selectedMode = "rgb";
      renderPlayer();
    }});
    qs("depth-btn").addEventListener("click", () => {{
      selectedMode = "depth";
      renderPlayer();
    }});

    boot().catch((err) => {{
      qs("empty").textContent = "加载页面失败: " + err;
    }});
  </script>
</body>
</html>
"""


class GalleryHandler(SimpleHTTPRequestHandler):
    server_version = "LiquidH5GalleryHTTP/0.1"

    def __init__(self, *args, directory: Optional[str] = None, **kwargs):
        self._root_dir = Path(directory or ".").resolve()
        super().__init__(*args, directory=str(self._root_dir), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[HTTP]", fmt % args)

    def translate_path(self, path: str) -> str:
        path = urllib.parse.urlparse(path).path
        path = posixpath.normpath(urllib.parse.unquote(path))
        parts = [p for p in path.split("/") if p and p not in (".", "..")]
        local_path = self._root_dir
        for part in parts:
            local_path = local_path / part
        try:
            local_path.resolve().relative_to(self._root_dir)
        except Exception:
            return str(self._root_dir / "__forbidden__")
        return str(local_path)

    def guess_type(self, path: str) -> str:
        ctype, _ = mimetypes.guess_type(path)
        if ctype:
            return ctype
        if path.endswith(".json"):
            return "application/json; charset=utf-8"
        return "application/octet-stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and serve a local gallery for liquid H5 dataset previews.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing scene_*.h5 files.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for gallery assets.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8023, help="HTTP bind port.")
    parser.add_argument("--title", type=str, default="Genesis Liquid Dataset Viewer", help="Gallery page title.")
    parser.add_argument("--force", action="store_true", help="Re-export mp4 files even if they already exist.")
    parser.add_argument("--serve", action="store_true", help="Serve the gallery after exporting assets.")
    return parser.parse_args()


def build_gallery(dataset_root: Path, output_root: Path, title: str, force: bool = False) -> Dict[str, Any]:
    ensure_dependencies()
    ensure_dir(output_root)

    h5_files = scan_h5_files(dataset_root)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in dataset root: {dataset_root}")

    previews = [export_scene_preview(path, output_root, force=force) for path in h5_files]
    manifest = build_manifest(previews)

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "index.html").write_text(build_index_html(title), encoding="utf-8")

    summary = {
        "gallery_root": str(output_root.resolve()),
        "scene_count": len(previews),
        "index_html": str((output_root / "index.html").resolve()),
        "manifest_json": str((output_root / "manifest.json").resolve()),
    }
    return summary


def serve_gallery(output_root: Path, host: str, port: int) -> None:
    handler = lambda *a, **kw: GalleryHandler(*a, directory=str(output_root), **kw)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"[INFO] Gallery root: {output_root}")
    print(f"[INFO] Open in browser: http://127.0.0.1:{port}/")
    print("[INFO] If you are on a remote server, forward the port to your local machine.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Gallery server stopped.")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    if not dataset_root.exists():
        raise SystemExit(f"[ERROR] dataset root does not exist: {dataset_root}")

    summary = build_gallery(
        dataset_root=dataset_root,
        output_root=output_root,
        title=args.title,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.serve:
        serve_gallery(output_root=output_root, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
