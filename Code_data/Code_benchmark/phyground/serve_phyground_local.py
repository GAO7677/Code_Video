#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/NU-World-Model-Embodied-AI-phyground")
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "phyground_index.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 18701


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the local PhyGround dataset index and videos on one port."
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def build_html(summary: dict, models: list[dict], laws: list[str]) -> str:
    model_options = "".join(
        f'<option value="{html.escape(item["model"])}">{html.escape(item["model"])}</option>'
        for item in models
    )
    law_options = "".join(
        f'<option value="{html.escape(law)}">{html.escape(law)}</option>' for law in laws
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhyGround Local Viewer</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf7;
      --line: #d9d2c4;
      --ink: #1d252c;
      --muted: #5d6a73;
      --accent: #a64b2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(166,75,42,0.10), transparent 28%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
    }}
    .wrap {{
      width: min(1680px, calc(100vw - 16px));
      margin: 0 auto;
      padding: 10px 0 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 26px;
    }}
    .sub {{
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .top {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .panel {{
      background: rgba(255, 253, 247, 0.92);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      box-shadow: 0 8px 22px rgba(52, 44, 32, 0.06);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    .stat {{
      padding: 8px 10px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat-value {{
      margin-top: 4px;
      font-size: 20px;
      font-weight: 700;
    }}
    .filters {{
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr auto;
      gap: 8px;
      align-items: end;
    }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    input, select, button {{
      width: 100%;
      height: 38px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fffefb;
      padding: 0 12px;
      font-size: 13px;
      color: var(--ink);
    }}
    button {{
      cursor: pointer;
      background: linear-gradient(135deg, #b75934, var(--accent));
      color: white;
      border: none;
      font-weight: 700;
      min-width: 120px;
    }}
    #results {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }}
    .card {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      background: rgba(255, 253, 247, 0.94);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      box-shadow: 0 8px 20px rgba(52, 44, 32, 0.05);
    }}
    .meta {{
      display: grid;
      gap: 6px;
      grid-template-columns: minmax(220px, 1.2fr) repeat(4, auto);
      align-items: start;
    }}
    .meta-row {{
      display: grid;
      gap: 2px;
    }}
    .meta-key {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .mono {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      word-break: break-word;
    }}
    .prompt {{
      white-space: pre-wrap;
      line-height: 1.35;
      font-size: 13px;
    }}
    .scores {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 8px;
    }}
    pre {{
      margin: 0;
      padding: 6px 8px;
      border-radius: 10px;
      background: #f6f1e8;
      border: 1px solid #e3dacb;
      overflow: auto;
      font-size: 11px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 10px;
      background: #e9e2d5;
      border: 1px solid #ddd2c2;
      aspect-ratio: 4 / 3;
      object-fit: contain;
    }}
    .hint {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .group-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 8px;
    }}
    @media (max-width: 1100px) {{
      .filters, .stats, .meta, .group-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="panel">
        <h1>PhyGround Local Viewer</h1>
        <p class="sub">Index + local video mapping on one port. Filter annotations and open the underlying dataset videos directly from the browser.</p>
        <div class="stats">
          <div class="stat"><div class="stat-label">Models</div><div class="stat-value">{summary["total_models"]}</div></div>
          <div class="stat"><div class="stat-label">Videos</div><div class="stat-value">{summary["total_videos"]}</div></div>
          <div class="stat"><div class="stat-label">Sources</div><div class="stat-value">{summary["total_groups"]}</div></div>
          <div class="stat"><div class="stat-label">Annotations</div><div class="stat-value">{summary["total_annotations"]}</div></div>
        </div>
      </div>
      <div class="panel">
        <div class="filters">
          <div>
            <label for="query">Search prompt / video</label>
            <input id="query" type="text" placeholder="collision, gravity, video stem...">
          </div>
          <div>
            <label for="model">Model</label>
            <select id="model">
              <option value="">All models</option>
              {model_options}
            </select>
          </div>
          <div>
            <label for="law">Law</label>
            <select id="law">
              <option value="">All laws</option>
              {law_options}
            </select>
          </div>
          <div>
            <button id="apply">Apply</button>
          </div>
        </div>
        <div class="hint">按同一来源分组展示。每组对应一个 prompt/source，里面并排放各模型视频和聚合后的标注统计。</div>
      </div>
    </div>
    <div id="results"></div>
  </div>
  <script>
    const resultsEl = document.getElementById("results");
    const queryEl = document.getElementById("query");
    const modelEl = document.getElementById("model");
    const lawEl = document.getElementById("law");
    const applyEl = document.getElementById("apply");
    let groups = [];

    function esc(text) {{
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function scoreBlock(title, obj) {{
      return `
        <div>
          <div class="meta-key">${{esc(title)}}</div>
          <pre>${{esc(JSON.stringify(obj || {{}}, null, 2))}}</pre>
        </div>
      `;
    }}

    function modelCard(model) {{
      const videoSrc = model.video_relpath ? `/dataset/${{model.video_relpath}}` : "";
      return `
        <article class="panel">
          <div class="meta-row">
            <div class="meta-key">Model</div>
            <div class="mono">${{esc(model.model)}}</div>
          </div>
          <div class="meta-row">
            <div class="meta-key">Video stem</div>
            <div class="mono">${{esc(model.video_stem)}}</div>
          </div>
          <div class="meta-row">
            <div class="meta-key">Annotations</div>
            <div>${{esc(model.annotation_count)}}</div>
          </div>
          <div class="meta-row">
            <div class="meta-key">Annotated laws</div>
            <div>${{esc((model.annotation_physical_laws || []).join(", "))}}</div>
          </div>
          ${{videoSrc ? `<video controls preload="metadata" src="${{esc(videoSrc)}}"></video>` : `<div>missing video path</div>`}}
          <div class="scores">
            ${{scoreBlock("General mean", model.general_score_means)}}
            ${{scoreBlock("Physical mean", model.physical_score_means)}}
          </div>
        </article>
      `;
    }}

    function render(list) {{
      const sliced = list.slice(0, 80);
      resultsEl.innerHTML = sliced.map((group) => {{
        return `
          <section class="card">
            <div class="meta">
              <div class="meta-row">
                <div class="meta-key">Source group</div>
                <div class="mono">${{esc(group.group_id)}} / ${{esc(group.prompt_id_stem)}}</div>
              </div>
              <div class="meta-row">
                <div class="meta-key">Prompt</div>
                <div class="prompt">${{esc(group.prompt)}}</div>
              </div>
              <div class="meta-row">
                <div class="meta-key">Prompt laws</div>
                <div>${{esc((group.prompt_physical_laws || []).join(", "))}}</div>
              </div>
              <div class="meta-row">
                <div class="meta-key">Models in group</div>
                <div>${{esc(group.model_count)}}</div>
              </div>
              <div class="meta-row">
                <div class="meta-key">Annotations in group</div>
                <div>${{esc(group.group_annotation_count)}}</div>
              </div>
            </div>
            <div class="group-grid">
              ${{group.models.map(modelCard).join("")}}
            </div>
          </section>
        `;
      }}).join("");
      if (list.length > 80) {{
        resultsEl.insertAdjacentHTML(
          "beforeend",
          `<div class="panel">Showing first 80 of ${{list.length}} matched source groups.</div>`
        );
      }}
      if (!list.length) {{
        resultsEl.innerHTML = `<div class="panel">No matched source groups.</div>`;
      }}
    }}

    function applyFilters() {{
      const q = queryEl.value.trim().toLowerCase();
      const model = modelEl.value;
      const law = lawEl.value;
      const filtered = groups.filter((group) => {{
        if (model && !(group.models || []).some((m) => m.model === model)) return false;
        if (law) {{
          const inPrompt = (group.prompt_physical_laws || []).includes(law);
          const inModels = (group.models || []).some((m) => (m.annotation_physical_laws || []).includes(law));
          if (!inPrompt && !inModels) return false;
        }}
        if (!q) return true;
        const hay = [
          group.prompt,
          group.prompt_id_stem,
          ...(group.prompt_physical_laws || []),
          ...((group.models || []).map((m) => [m.model, m.video_stem, ...(m.annotation_physical_laws || [])]).flat()),
        ].join(" ").toLowerCase();
        return hay.includes(q);
      }});
      render(filtered);
    }}

    applyEl.addEventListener("click", applyFilters);
    queryEl.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") applyFilters();
    }});

    fetch("/api/index")
      .then((res) => res.json())
      .then((payload) => {{
        groups = payload.groups || [];
        render(groups);
      }})
      .catch((err) => {{
        resultsEl.innerHTML = `<div class="panel">Failed to load index: ${{esc(err.message)}}</div>`;
      }});
  </script>
</body>
</html>
"""


def make_handler(dataset_root: Path, index_path: Path):
    index_blob = json.loads(index_path.read_text(encoding="utf-8"))
    summary = index_blob["summary"]
    models = index_blob["models"]
    laws = sorted(summary.get("annotated_laws", {}).keys())
    html_blob = build_html(summary, models, laws).encode("utf-8")
    api_blob = json.dumps(index_blob, ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_HEAD(self) -> None:
            self._dispatch(send_body=False)

        def do_GET(self) -> None:
            self._dispatch(send_body=True)

        def _dispatch(self, *, send_body: bool) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(html_blob, "text/html; charset=utf-8", send_body=send_body)
                return
            if parsed.path == "/healthz":
                self._send_bytes(b"ok\n", "text/plain; charset=utf-8", send_body=send_body)
                return
            if parsed.path == "/api/index":
                self._send_bytes(api_blob, "application/json; charset=utf-8", send_body=send_body)
                return
            if parsed.path == "/api/summary":
                self._send_bytes(
                    json.dumps(summary, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                    send_body=send_body,
                )
                return
            if parsed.path.startswith("/dataset/"):
                rel = parsed.path[len("/dataset/") :]
                self._serve_dataset_file(rel, send_body=send_body)
                return
            if parsed.path == "/favicon.ico":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Path not found")

        def _serve_dataset_file(self, relative_path: str, *, send_body: bool) -> None:
            safe_rel = os.path.normpath("/" + relative_path).lstrip("/")
            file_path = (dataset_root / safe_rel).resolve()
            dataset_root_resolved = dataset_root.resolve()
            if dataset_root_resolved not in file_path.parents and file_path != dataset_root_resolved:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden path")
                return
            if not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            mime, _ = mimetypes.guess_type(file_path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(file_path.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if send_body:
                with file_path.open("rb") as handle:
                    self.wfile.write(handle.read())

        def _send_bytes(self, payload: bytes, content_type: str, *, send_body: bool) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if send_body:
                self.wfile.write(payload)

    return Handler


def main() -> None:
    args = parse_args()
    handler = make_handler(args.dataset_root.resolve(), args.index.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
