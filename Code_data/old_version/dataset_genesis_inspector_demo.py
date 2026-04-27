#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地浏览器演示：读取 dataset_3_rigid_genesis.py / dataset_3_mpm_genesis.py 导出的目录结构。

这两个脚本本身是在线生成仿真数据，不是 PyTorch Dataset；导出后每个样本目录通常包含：
  - scene_input.json       仿真前采样得到的完整场景配置
  - scene_metadata.json    仿真摘要、导出文件索引、材料摘要等
  - trajectories/          frame_index.csv, objects_world.csv, object_pointcloud_index.csv
  - rgb/, depth/, depth_vis/, segmentation/, normal/, pointcloud/, object_pointcloud/
  - camera/intrinsics.npy, extrinsics.npy
  - video/preview.mp4

用法示例：
  python dataset_genesis_inspector_demo.py \\
    --root /data/gaoya/AAA_test_video/Dataset_test/Genesis_rigid:rigid \\
    --root /data/gaoya/AAA_test_video/Dataset_test/Genesis_mpm_sophy_only:mpm \\
    --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import posixpath
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_under_root(root: Path, rel: str) -> Path:
    root = root.resolve()
    rel = rel.lstrip("/").replace("\\", "/")
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ValueError("path escapes root") from e
    return target


def discover_scenes(root: Path, max_meta_bytes: int = 120_000) -> List[Dict[str, Any]]:
    """Find directories containing scene_metadata.json; load a short preview."""
    root = root.resolve()
    out: List[Dict[str, Any]] = []
    for meta_path in sorted(root.rglob("scene_metadata.json")):
        scene_dir = meta_path.parent
        try:
            rel = scene_dir.relative_to(root).as_posix()
        except ValueError:
            continue
        preview: Dict[str, Any] = {
            "rel_path": rel,
            "meta_path": rel + "/scene_metadata.json",
        }
        try:
            raw = meta_path.read_bytes()
            if len(raw) > max_meta_bytes:
                preview["meta_truncated"] = True
                raw = raw[:max_meta_bytes]
            meta = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            preview["meta_error"] = str(exc)
            out.append(preview)
            continue
        preview["scene_id"] = meta.get("scene_id", scene_dir.name)
        preview["family"] = meta.get("family")
        preview["status"] = meta.get("status")
        preview["sim_steps"] = meta.get("sim_steps")
        preview["physics_duration_s"] = meta.get("physics_duration_s")
        preview["num_objects"] = meta.get("num_objects")
        # Rigid vs MPM distinguishing fields
        if "rigid_motion_category" in meta:
            preview["motion_category"] = meta.get("rigid_motion_category")
            preview["motion_label_zh"] = meta.get("rigid_motion_label_zh")
        elif "mpm_motion_category" in meta:
            preview["motion_category"] = meta.get("mpm_motion_category")
            preview["motion_label_zh"] = meta.get("mpm_motion_label_zh")
        else:
            preview["motion_category"] = meta.get("motion_category")
            preview["motion_label_zh"] = meta.get("motion_label_zh")
        preview["has_video"] = (scene_dir / "video" / "preview.mp4").is_file()
        preview["has_rgb"] = (scene_dir / "rgb").is_dir() and any((scene_dir / "rgb").glob("*.png"))
        out.append(preview)
    return out


def build_index_payload(roots: List[Tuple[str, Path]]) -> Dict[str, Any]:
    scenes: List[Dict[str, Any]] = []
    for rid, (label, root) in enumerate(roots):
        for s in discover_scenes(root):
            row = dict(s)
            row["root_id"] = rid
            row["root_label"] = label
            row["root_path"] = str(root)
            scenes.append(row)
    return {
        "roots": [{"id": i, "label": lab, "path": str(p)} for i, (lab, p) in enumerate(roots)],
        "scenes": scenes,
        "schema_notes": {
            "per_scene_files": [
                "scene_input.json",
                "scene_metadata.json",
                "trajectories/frame_index.csv",
                "trajectories/objects_world.csv",
                "trajectories/object_pointcloud_index.csv",
                "rgb/<frame>.png",
                "depth/<frame>.npy",
                "depth_vis/<frame>.png",
                "segmentation/<frame>.npy",
                "normal/<frame>.npy",
                "pointcloud/<frame>.npz (keys: xyz, mask)",
                "object_pointcloud/<frame>_obj<id>.npz",
                "camera/intrinsics.npy",
                "camera/extrinsics.npy",
                "video/preview.mp4",
            ],
            "optional_manifests": [
                "dataset_manifest.json (rigid main 写入)",
                "asset_manifest.json (MPM 资产库索引)",
                "DATASET_FORMAT.md (rigid 写入)",
            ],
        },
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Genesis 数据集导出浏览器</title>
  <style>
    :root {
      --bg: #0c0f14;
      --panel: #141a22;
      --text: #e8edf7;
      --muted: #8b98ad;
      --accent: #5da9ff;
      --border: rgba(255,255,255,0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", "PingFang SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .wrap { max-width: 1480px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 1.65rem; font-weight: 600; margin: 0 0 8px; }
    h2 { font-size: 1.1rem; margin: 24px 0 10px; font-weight: 600; }
    .muted { color: var(--muted); font-size: 0.9rem; }
    #rootSummary { white-space: pre-wrap; word-break: break-all; }
    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 20px; align-items: start; }
    @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
    aside {
      position: sticky; top: 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      max-height: calc(100vh - 32px);
      overflow: auto;
    }
    main {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      min-height: 400px;
    }
    ul.scene-list { list-style: none; padding: 0; margin: 0; }
    ul.scene-list li {
      padding: 8px 10px; margin-bottom: 6px;
      border-radius: 8px; cursor: pointer;
      border: 1px solid transparent;
    }
    ul.scene-list li:hover { background: rgba(93,169,255,0.08); }
    ul.scene-list li.active { border-color: var(--accent); background: rgba(93,169,255,0.12); }
    .pill {
      display: inline-block; font-size: 0.72rem;
      padding: 2px 8px; border-radius: 999px;
      background: rgba(93,169,255,0.15); color: var(--accent);
      margin-right: 6px;
    }
    video { max-width: 100%; border-radius: 8px; background: #000; }
    .row-img { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .row-img img { max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border); }
    pre {
      background: #0a0d12; padding: 12px; border-radius: 8px;
      overflow: auto; font-size: 0.78rem; border: 1px solid var(--border);
      max-height: 420px;
    }
    canvas#traj {
      width: 100%; max-width: 720px; height: 220px;
      background: #0a0d12; border-radius: 8px; border: 1px solid var(--border);
    }
    button.btn {
      background: var(--accent); color: #041018; border: none;
      padding: 8px 14px; border-radius: 8px; font-weight: 600; cursor: pointer;
      margin-top: 8px;
    }
    button.btn:disabled { opacity: 0.45; cursor: not-allowed; }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Genesis 刚体 / MPM 导出数据 · 本地浏览器</h1>
    <p class="muted">
      数据由 <code>dataset_3_rigid_genesis.py</code> 与 <code>dataset_3_mpm_genesis.py</code> 在仿真时写出。
      左侧选择场景；右侧查看元数据摘要、视频、RGB/深度可视化、轨迹折线（由 <code>objects_world.csv</code> 解析）。
    </p>
    <div class="grid">
      <aside>
        <div class="muted" id="rootSummary"></div>
        <h2>场景列表</h2>
        <ul class="scene-list" id="sceneList"></ul>
      </aside>
      <main id="detail">
        <p class="muted">加载中…</p>
      </main>
    </div>
  </div>
  <script>
  const state = { payload: null, selected: null };

  function el(tag, props, children) {
    const n = document.createElement(tag);
    if (props) Object.assign(n, props);
    (children || []).forEach(c => { if (c) n.appendChild(c); });
    return n;
  }

  async function loadPayload() {
    const r = await fetch("/api/index.json");
    state.payload = await r.json();
    document.getElementById("rootSummary").textContent =
      state.payload.roots.map(x => x.label + ": " + x.path).join("\\n");
    const ul = document.getElementById("sceneList");
    ul.innerHTML = "";
    if (!state.payload.scenes.length) {
      ul.appendChild(el("li", null, [document.createTextNode("未发现 scene_metadata.json，请检查 --root 路径。")]));
      document.getElementById("detail").innerHTML = "<p class=muted>无数据。可用示例：<br><code>--root /data/.../Genesis_rigid:rigid</code></p>";
      return;
    }
    state.payload.scenes.forEach((s, idx) => {
      const li = el("li", null, []);
      li.appendChild(el("span", { className: "pill", textContent: s.root_label }));
      li.appendChild(document.createTextNode(s.scene_id || s.rel_path));
      li.title = s.rel_path;
      li.onclick = () => selectScene(idx);
      ul.appendChild(li);
    });
    selectScene(0);
  }

  function mediaUrl(rootId, relPath) {
    return "/media/" + encodeURIComponent(String(rootId)) + "/" +
      relPath.split("/").map(encodeURIComponent).join("/");
  }

  function drawTrajectories(csvText, canvas) {
    const lines = csvText.trim().split(/\\r?\\n/);
    if (lines.length < 2) return;
    const header = lines[0].split(",");
    const idx = {};
    header.forEach((h, i) => { idx[h.trim()] = i; });
    const need = ["frame", "object_id", "cx", "cy", "cz"];
    for (const k of need) if (!(k in idx)) return;
    const series = new Map();
    for (let r = 1; r < lines.length; r++) {
      const p = lines[r].split(",");
      const oid = p[idx.object_id];
      const f = parseFloat(p[idx.frame]);
      const z = parseFloat(p[idx.cz]);
      if (!series.has(oid)) series.set(oid, []);
      series.get(oid).push([f, z]);
    }
    const ctx = canvas.getContext("2d");
    const w = canvas.width = canvas.clientWidth * devicePixelRatio;
    const h = canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0a0d12";
    ctx.fillRect(0, 0, w, h);
    let minF = Infinity, maxF = -Infinity, minZ = Infinity, maxZ = -Infinity;
    series.forEach(pts => {
      pts.forEach(([f, z]) => {
        minF = Math.min(minF, f); maxF = Math.max(maxF, f);
        minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
      });
    });
    if (!isFinite(minF)) return;
    const pad = 28;
    const xf = x => pad + (x - minF) / (maxF - minF || 1) * (w - 2 * pad);
    const yf = z => h - pad - (z - minZ) / ((maxZ - minZ) || 1) * (h - 2 * pad);
    const colors = ["#5da9ff", "#ff7a7a", "#7affb0", "#e4ff7a", "#c77aff"];
    let ci = 0;
    series.forEach((pts, oid) => {
      ctx.strokeStyle = colors[ci++ % colors.length];
      ctx.lineWidth = 2 * devicePixelRatio;
      ctx.beginPath();
      pts.forEach(([f, z], i) => {
        const X = xf(f), Y = yf(z);
        if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
      });
      ctx.stroke();
      ctx.fillStyle = "#8b98ad";
      ctx.font = (12 * devicePixelRatio) + "px sans-serif";
      ctx.fillText("obj " + oid, pad, pad - 8 + (ci % 4) * (14 * devicePixelRatio));
    });
    ctx.fillStyle = "#8b98ad";
    ctx.font = (11 * devicePixelRatio) + "px sans-serif";
    ctx.fillText("frame →", w - 80 * devicePixelRatio, h - 8);
    ctx.save();
    ctx.translate(12 * devicePixelRatio, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("centroid z (world)", 0, 0);
    ctx.restore();
  }

  async function selectScene(idx) {
    const s = state.payload.scenes[idx];
    state.selected = s;
    document.querySelectorAll("#sceneList li").forEach((li, i) => {
      li.classList.toggle("active", i === idx);
    });
    const detail = document.getElementById("detail");
    detail.innerHTML = "加载详情…";
    const r = await fetch("/api/scene.json?" + new URLSearchParams({
      root: s.root_id, path: s.rel_path
    }));
    const full = await r.json();
    detail.innerHTML = "";
    const h3 = el("h2", { textContent: full.scene_id || s.rel_path });
    detail.appendChild(h3);
    detail.appendChild(el("p", { className: "muted", textContent: s.root_label + " · " + s.rel_path }));

    if (full.video_exists) {
      const v = el("video", { controls: true, playsInline: true });
      v.appendChild(el("source", { src: mediaUrl(s.root_id, s.rel_path + "/video/preview.mp4"), type: "video/mp4" }));
      detail.appendChild(v);
    }
    const imgs = el("div", { className: "row-img" });
    if (full.sample_frames) {
      for (const fr of full.sample_frames) {
        const cap = el("div", null, []);
        cap.appendChild(el("div", { className: "muted", textContent: fr.label }));
        cap.appendChild(el("img", { src: mediaUrl(s.root_id, fr.file), alt: fr.label }));
        imgs.appendChild(cap);
      }
    }
    detail.appendChild(imgs);

    const trajBlock = el("div", null, []);
    trajBlock.appendChild(el("h2", { textContent: "轨迹：质心 z 随 frame（objects_world.csv）" }));
    const cvs = el("canvas", { id: "traj" });
    trajBlock.appendChild(cvs);
    detail.appendChild(trajBlock);

    detail.appendChild(el("h2", { textContent: "scene_metadata.json（节选骨架）" }));
    detail.appendChild(el("pre", { textContent: JSON.stringify(full.metadata_excerpt, null, 2) }));

    detail.appendChild(el("h2", { textContent: "可从数据目录读取的字段（概念）" }));
    detail.appendChild(el("pre", { textContent: JSON.stringify(full.available_concepts, null, 2) }));

    if (full.trajectory_csv) {
      fetch(mediaUrl(s.root_id, full.trajectory_csv)).then(r => r.text())
        .then(t => drawTrajectories(t, cvs))
        .catch(() => {});
    }
  }

  loadPayload().catch(e => {
    document.getElementById("detail").textContent = String(e);
  });
  </script>
</body>
</html>
"""


class InspectorHandler(BaseHTTPRequestHandler):
    server_version = "GenesisInspector/1.0"

    def __init__(self, *args, roots: List[Tuple[str, Path]] = None, **kwargs):
        self._roots = roots or []
        self._index_cache: Optional[Dict[str, Any]] = None
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[HTTP]", fmt % args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _get_index(self) -> Dict[str, Any]:
        if self._index_cache is None:
            self._index_cache = build_index_payload(self._roots)
        return self._index_cache

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/index.json":
            self._send_json(self._get_index())
            return

        if path == "/api/scene.json":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                root_id = int(qs.get("root", ["0"])[0])
                rel = qs.get("path", [""])[0]
            except ValueError:
                self._send_json({"error": "bad root"}, 400)
                return
            if root_id < 0 or root_id >= len(self._roots):
                self._send_json({"error": "root out of range"}, 400)
                return
            _, root = self._roots[root_id]
            try:
                scene_dir = _safe_under_root(root, rel)
            except ValueError:
                self._send_json({"error": "invalid path"}, 403)
                return
            meta_path = scene_dir / "scene_metadata.json"
            if not meta_path.is_file():
                self._send_json({"error": "no scene_metadata.json"}, 404)
                return
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
                return

            rgb_dir = scene_dir / "rgb"
            sample_frames: List[Dict[str, str]] = []
            if rgb_dir.is_dir():
                rgbs = sorted(rgb_dir.glob("*.png"))
                if rgbs:
                    sample_frames.append({"label": "rgb first", "file": rgbs[0].relative_to(root).as_posix()})
                    if len(rgbs) > 1:
                        sample_frames.append({"label": "rgb mid", "file": rgbs[len(rgbs) // 2].relative_to(root).as_posix()})
                        sample_frames.append({"label": "rgb last", "file": rgbs[-1].relative_to(root).as_posix()})
            dv = scene_dir / "depth_vis"
            if dv.is_dir():
                dvs = sorted(dv.glob("*.png"))
                if dvs:
                    sample_frames.append({"label": "depth_vis first", "file": dvs[0].relative_to(root).as_posix()})

            excerpt_keys = (
                "scene_id",
                "family",
                "seed",
                "rigid_pattern",
                "rigid_motion_category",
                "rigid_motion_label_zh",
                "mpm_motion_category",
                "mpm_motion_label_zh",
                "scene_builder",
                "object_count_bucket",
                "num_objects",
                "num_static_objects",
                "num_dynamic_objects",
                "motion_modes_present",
                "sim_steps",
                "dt",
                "substeps",
                "physics_duration_s",
                "preview_fps",
                "collision_detected",
                "num_dataset_mesh_objects",
                "num_soft_parts",
                "pattern",
                "exports",
                "material_summary",
                "objects",
                "status",
            )
            metadata_excerpt = {k.strip(): meta.get(k.strip()) for k in excerpt_keys if k.strip() in meta}

            traj_rel = (scene_dir / "trajectories" / "objects_world.csv").relative_to(root).as_posix()
            concepts: Dict[str, Any] = {
                "per_frame_sensor": {
                    "rgb": "rgb/*.png",
                    "depth_m": "depth/*.npy",
                    "depth_vis": "depth_vis/*.png",
                    "segmentation": "segmentation/*.npy",
                    "normal": "normal/*.npy",
                    "scene_pointcloud": "pointcloud/*.npz (xyz, mask)",
                },
                "per_object_pointcloud_index": "trajectories/object_pointcloud_index.csv → object_pointcloud/*.npz",
                "trajectory_csv_columns": "frame, object_id, solver, cx..cz, qx..qw, vx..vz, wx..wz, n_points",
                "camera": ["camera/intrinsics.npy", "camera/extrinsics.npy"],
                "full_scene_config": "scene_input.json（与 scene_metadata 互补）",
            }

            self._send_json({
                "scene_id": meta.get("scene_id"),
                "video_exists": (scene_dir / "video" / "preview.mp4").is_file(),
                "sample_frames": sample_frames,
                "metadata_excerpt": metadata_excerpt,
                "available_concepts": concepts,
                "trajectory_csv": traj_rel,
            })
            return

        if path.startswith("/media/"):
            rest = path[len("/media/") :]
            parts = [urllib.parse.unquote(p) for p in rest.split("/") if p and p not in (".", "..")]
            if len(parts) < 2:
                self._send_json({"error": "media path"}, 400)
                return
            try:
                root_id = int(parts[0])
            except ValueError:
                self._send_json({"error": "bad root"}, 400)
                return
            rel = "/".join(parts[1:])
            if root_id < 0 or root_id >= len(self._roots):
                self._send_json({"error": "root out of range"}, 400)
                return
            _, root = self._roots[root_id]
            try:
                file_path = _safe_under_root(root, rel)
            except ValueError:
                self._send_json({"error": "invalid path"}, 403)
                return
            if not file_path.is_file():
                self.send_error(404, "Not found")
                return
            suf = file_path.suffix.lower()
            ctype = "application/octet-stream"
            if suf == ".png":
                ctype = "image/png"
            elif suf == ".jpg" or suf == ".jpeg":
                ctype = "image/jpeg"
            elif suf == ".mp4":
                ctype = "video/mp4"
            elif suf == ".json":
                ctype = "application/json"
            elif suf == ".csv":
                ctype = "text/csv; charset=utf-8"
            elif suf == ".npz":
                ctype = "application/octet-stream"
            elif suf == ".npy":
                ctype = "application/octet-stream"
            try:
                data = file_path.read_bytes()
            except OSError:
                self.send_error(500, "read error")
                return
            self._send_bytes(data, ctype)
            return

        self.send_error(404, "Not found")


def parse_roots(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    """Parse --root entries.

    Accepted forms:
    - `/abs/path`  (label auto: root0, root1, …)
    - `/abs/path:mytag`  (directory before the last colon must exist)
    - `mytag:/abs/path`  (left side is not an existing directory)
    """
    pairs: List[Tuple[str, Path]] = []
    for i, item in enumerate(args.root):
        item = item.strip()
        if ":" in item:
            left, right = item.split(":", 1)
            left, right = left.strip(), right.strip()
            left_path = Path(left)
            if left_path.is_dir():
                path = left_path.resolve()
                label = right or f"root{i}"
            else:
                label = left or f"root{i}"
                path = Path(right).resolve()
        else:
            label = f"root{i}"
            path = Path(item).resolve()
        if not path.is_dir():
            raise SystemExit(f"路径不存在或不是目录: {path}")
        pairs.append((label, path))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="浏览 rigid / MPM Genesis 导出数据集")
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        help="数据根目录，可为 `标签:路径` 或仅路径（自动 root0,root1…）。可多次指定。",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ns = ap.parse_args()
    if not ns.root:
        ns.root = [
            "/data/gaoya/AAA_test_video/Dataset_test/Genesis_rigid:rigid",
            "/data/gaoya/AAA_test_video/Dataset_test/Genesis_mpm_sophy_only:mpm",
        ]
        print("[INFO] 未指定 --root，使用默认（若不存在请自行传入）：")
        for r in ns.root:
            print("       ", r)

    roots = parse_roots(ns)
    n_scenes = sum(len(discover_scenes(p)) for _, p in roots)
    print(f"[INFO] {len(roots)} 个根目录，共发现 {n_scenes} 个 scene_metadata.json")

    def factory(*a: Any, **kw: Any) -> InspectorHandler:
        return InspectorHandler(*a, roots=roots, **kw)

    server = ThreadingHTTPServer((ns.host, ns.port), factory)
    print(f"[INFO] 打开浏览器: http://{ns.host}:{ns.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] 已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
