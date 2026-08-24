#!/usr/bin/env python3
"""Small local viewer for representative cases from a PyBullet training set."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from pathlib import Path
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse


DEFAULT_MANIFEST = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1/manifest.json"
)


def compact_case(item: dict) -> dict:
    motion = item.get("motion_metrics") or {}
    return {
        "case_id": item.get("case_id", ""),
        "family": item.get("family_key", ""),
        "seed": item.get("seed"),
        "attempt": item.get("attempt_index"),
        "video": item.get("video", ""),
        "caption": item.get("caption", ""),
        "objects": item.get("object_nouns") or [],
        "dynamic_objects": item.get("dynamic_object_phrases") or [],
        "motion": {
            "object_speed": motion.get("motion_object_diag_pct_per_second"),
            "moving_area": motion.get("moving_area_ratio"),
            "presence": motion.get("motion_presence_ratio"),
            "frames": motion.get("frame_count"),
            "fps": motion.get("fps"),
        },
    }


def load_cases(path: Path) -> tuple[dict[str, dict], list[str], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [compact_case(item) for item in raw]
    cases = [item for item in cases if item["case_id"] and Path(item["video"]).is_file()]
    cases.sort(key=lambda item: (item["family"], item["case_id"]))
    by_id = {item["case_id"]: item for item in cases}
    families = sorted({item["family"] for item in cases}, key=lambda value: int(value[1:]))
    reps = [next(item["case_id"] for item in cases if item["family"] == family) for family in families]
    return by_id, families, reps


def json_for_html(value: object) -> str:
    # Prevent a caption from terminating the data script element.
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>0717 PyBullet · Training Set Motion Atlas</title>
  <style>
    :root {
      --ink: #101419;
      --ink-soft: #192027;
      --paper: #e9eef0;
      --muted: #9aa8af;
      --lime: #d9ff63;
      --copper: #e37b5f;
      --sky: #8ed1fc;
      --line: rgba(233, 238, 240, .15);
      --shadow: 0 24px 70px rgba(0,0,0,.32);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--paper);
      background: var(--ink);
      font-family: "Arial", "Noto Sans SC", sans-serif;
      letter-spacing: .01em;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .16;
      background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
      background-size: 36px 36px;
      mask-image: linear-gradient(to bottom, black, transparent 78%);
    }
    .shell { max-width: 1500px; margin: 0 auto; padding: 30px 34px 70px; position: relative; }
    .eyebrow { color: var(--lime); font: 700 11px/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .17em; text-transform: uppercase; }
    header { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 38px; align-items: end; padding: 26px 0 30px; border-bottom: 1px solid var(--line); }
    h1 { margin: 12px 0 12px; max-width: 850px; font: 700 clamp(38px, 6vw, 82px)/.94 Georgia, "Times New Roman", serif; letter-spacing: -.055em; }
    .lede { max-width: 720px; margin: 0; color: #bdc9ce; font-size: 15px; line-height: 1.7; }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; border: 1px solid var(--line); background: var(--line); box-shadow: var(--shadow); }
    .stat { min-height: 105px; padding: 18px; background: rgba(25,32,39,.95); }
    .stat strong { display: block; color: var(--lime); font: 700 31px/1 ui-monospace, SFMono-Regular, monospace; }
    .stat span { display: block; margin-top: 13px; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .12em; }
    .toolbar { display: flex; gap: 12px; align-items: center; justify-content: space-between; flex-wrap: wrap; padding: 24px 0 18px; }
    .families { display: flex; flex-wrap: wrap; gap: 8px; }
    button, select { color: var(--paper); background: var(--ink-soft); border: 1px solid var(--line); border-radius: 0; font: inherit; }
    button { cursor: pointer; padding: 9px 13px; font: 700 11px ui-monospace, SFMono-Regular, monospace; letter-spacing: .08em; }
    button:hover, button.active { color: var(--ink); background: var(--lime); border-color: var(--lime); }
    select { min-width: min(520px, 100%); padding: 11px 14px; outline: none; }
    select:focus, button:focus-visible { outline: 2px solid var(--sky); outline-offset: 3px; }
    .label { color: var(--muted); font-size: 12px; }
    .atlas { display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 18px; }
    .card { overflow: hidden; background: rgba(25,32,39,.9); border: 1px solid var(--line); box-shadow: 0 16px 42px rgba(0,0,0,.2); transition: transform .2s ease, border-color .2s ease; }
    .card:hover { transform: translateY(-3px); border-color: rgba(217,255,99,.65); }
    .card video { display: block; width: 100%; aspect-ratio: 16/9; background: #080b0e; object-fit: cover; }
    .card-body { padding: 16px 17px 18px; }
    .card-top { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
    .family-tag { color: var(--copper); font: 700 12px ui-monospace, SFMono-Regular, monospace; letter-spacing: .12em; }
    .case-id { color: var(--muted); font: 11px ui-monospace, SFMono-Regular, monospace; text-align: right; }
    .caption { min-height: 49px; margin: 12px 0 14px; color: #d7e0e3; font-size: 14px; line-height: 1.55; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip { padding: 5px 7px; color: #aab8bd; background: rgba(255,255,255,.055); font: 10px ui-monospace, SFMono-Regular, monospace; }
    .focus { display: none; grid-template-columns: minmax(0, 1.4fr) minmax(280px, .6fr); gap: 0; margin: 5px 0 25px; border: 1px solid var(--lime); background: var(--ink-soft); box-shadow: var(--shadow); }
    .focus.visible { display: grid; }
    .focus video { width: 100%; height: 100%; min-height: 300px; display: block; background: #080b0e; object-fit: contain; }
    .focus-copy { padding: 25px; }
    .focus-copy h2 { margin: 9px 0 16px; font: 700 28px/1.05 Georgia, serif; letter-spacing: -.03em; }
    .focus-copy p { color: #c4d0d4; line-height: 1.7; font-size: 14px; }
    .metric-list { margin: 22px 0 0; padding: 0; list-style: none; border-top: 1px solid var(--line); }
    .metric-list li { display: flex; justify-content: space-between; gap: 20px; padding: 10px 0; border-bottom: 1px solid var(--line); color: var(--muted); font: 11px ui-monospace, SFMono-Regular, monospace; }
    .metric-list b { color: var(--lime); font-weight: 500; }
    footer { margin-top: 36px; padding-top: 17px; border-top: 1px solid var(--line); color: #718087; font: 11px/1.6 ui-monospace, SFMono-Regular, monospace; }
    @media (max-width: 850px) { header { grid-template-columns: 1fr; } .focus { grid-template-columns: 1fr; } .focus video { min-height: 0; } .shell { padding: 20px 16px 48px; } }
    @media (prefers-reduced-motion: reduce) { .card { transition: none; } }
  </style>
</head>
<body>
<main class="shell">
  <header>
    <div>
      <div class="eyebrow">0717 / PYBULLET / TRAINING SET</div>
      <h1>Training set,<br><em>in motion.</em></h1>
      <p class="lede">A compact atlas of the simulation cases used by the prompt-physics consistency training run. Start with one representative from each family, then jump into the full case index.</p>
    </div>
    <div class="stats">
      <div class="stat"><strong id="case-count">—</strong><span>cases</span></div>
      <div class="stat"><strong id="family-count">—</strong><span>families</span></div>
      <div class="stat"><strong>90f</strong><span>at 30 fps</span></div>
    </div>
  </header>
  <section class="toolbar" aria-label="case controls">
    <div>
      <div class="label">Family sampler</div>
      <div class="families" id="families"></div>
    </div>
    <label>
      <span class="label">All cases</span><br>
      <select id="case-select" aria-label="Select a case"></select>
    </label>
  </section>
  <section class="focus" id="focus" aria-live="polite">
    <video id="focus-video" controls playsinline preload="metadata"></video>
    <div class="focus-copy">
      <div class="eyebrow" id="focus-family"></div>
      <h2 id="focus-title"></h2>
      <p id="focus-caption"></p>
      <ul class="metric-list" id="focus-metrics"></ul>
    </div>
  </section>
  <section class="atlas" id="atlas" aria-label="Representative cases"></section>
  <footer>Source: /data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1 · Original simulation videos are streamed from the source dataset; no large files are copied.</footer>
</main>
<script id="case-data" type="application/json">__DATA__</script>
<script>
  const DATA = JSON.parse(document.getElementById('case-data').textContent);
  const cases = DATA.cases;
  const families = DATA.families;
  const reps = new Set(DATA.representatives);
  const byId = new Map(cases.map(c => [c.case_id, c]));
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const fmt = (value, digits=2) => value == null ? '—' : Number(value).toFixed(digits);
  const media = id => `/media/${encodeURIComponent(id)}`;

  document.getElementById('case-count').textContent = cases.length.toLocaleString();
  document.getElementById('family-count').textContent = families.length;
  const select = document.getElementById('case-select');
  for (const c of cases) {
    const option = document.createElement('option');
    option.value = c.case_id;
    option.textContent = `${c.family} · ${c.case_id} · ${(c.objects || []).join(', ') || 'scene'}`;
    select.appendChild(option);
  }

  function card(c) {
    const motion = c.motion || {};
    return `<article class="card" data-family="${esc(c.family)}">
      <video controls playsinline preload="metadata" src="${media(c.case_id)}"></video>
      <div class="card-body"><div class="card-top"><span class="family-tag">${esc(c.family)}</span><span class="case-id">${esc(c.case_id)}</span></div>
      <p class="caption">${esc(c.caption)}</p><div class="chips"><span class="chip">object speed ${fmt(motion.object_speed)}%</span><span class="chip">moving area ${fmt((motion.moving_area || 0) * 100)}%</span><span class="chip">seed ${esc(c.seed)}</span></div></div>
    </article>`;
  }
  function renderCards(family='ALL') {
    const ids = family === 'ALL' ? DATA.representatives : cases.filter(c => c.family === family).slice(0, 6).map(c => c.case_id);
    document.getElementById('atlas').innerHTML = ids.map(id => card(byId.get(id))).join('');
    document.querySelectorAll('.families button').forEach(b => b.classList.toggle('active', b.dataset.family === family));
  }
  function showFocus(id) {
    const c = byId.get(id); if (!c) return;
    const motion = c.motion || {};
    document.getElementById('focus').classList.add('visible');
    const video = document.getElementById('focus-video');
    video.pause(); video.src = media(c.case_id); video.load();
    document.getElementById('focus-family').textContent = `${c.family} / SELECTED CASE`;
    document.getElementById('focus-title').textContent = c.case_id;
    document.getElementById('focus-caption').textContent = c.caption;
    document.getElementById('focus-metrics').innerHTML = [
      ['dynamic objects', (c.dynamic_objects || []).join(', ') || '—'],
      ['object speed', `${fmt(motion.object_speed)}% diagonal px/s`],
      ['moving area', `${fmt((motion.moving_area || 0) * 100)}%`],
      ['motion presence', `${fmt((motion.presence || 0) * 100)}%`],
      ['seed', c.seed],
    ].map(([k,v]) => `<li><span>${esc(k)}</span><b>${esc(v)}</b></li>`).join('');
    document.getElementById('focus').scrollIntoView({behavior: 'smooth', block: 'start'});
  }
  const familyWrap = document.getElementById('families');
  familyWrap.innerHTML = [`ALL`, ...families].map(f => `<button data-family="${f}">${f}</button>`).join('');
  familyWrap.addEventListener('click', event => { if (event.target.matches('button')) renderCards(event.target.dataset.family); });
  select.addEventListener('change', event => showFocus(event.target.value));
  renderCards();
</script>
</body>
</html>"""


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "TrainingSetViewer/1.0"

    @property
    def viewer(self) -> "DatasetViewer":
        return self.server.viewer  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.address_string()}] {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_bytes(self.viewer.page, "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self.send_bytes(json.dumps({"ok": True, "cases": len(self.viewer.cases)}).encode(), "application/json")
            return
        match = re.fullmatch(r"/media/(.+)", parsed.path)
        if match:
            case_id = unquote(match.group(1))
            item = self.viewer.cases.get(case_id)
            if item is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown case")
                return
            self.send_file(Path(item["video"]))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def send_bytes(self, payload: bytes | str, content_type: str) -> None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Video missing")
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            if match.group(1):
                start = int(match.group(1))
            if match.group(2):
                end = int(match.group(2))
            elif start < size:
                end = min(size - 1, start + 4 * 1024 * 1024 - 1)
            if start >= size or end < start:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            end = min(end, size - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        else:
            status = HTTPStatus.OK
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class DatasetViewer:
    def __init__(self, manifest: Path) -> None:
        self.cases, self.families, representatives = load_cases(manifest)
        payload = {
            "cases": list(self.cases.values()),
            "families": self.families,
            "representatives": representatives,
        }
        self.page = PAGE_TEMPLATE.replace("__DATA__", json_for_html(payload)).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    viewer = DatasetViewer(args.manifest.resolve())
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.viewer = viewer  # type: ignore[attr-defined]
    print(f"Training-set viewer: http://127.0.0.1:{args.port}/")
    print(f"Loaded {len(viewer.cases)} cases across {len(viewer.families)} families")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
