#!/usr/bin/env python3
"""该脚本用于浏览 /data/gaoya/dataset/vLAR-PhysInOne/mytest 与 /data/gaoya/dataset/physics-iq-benchmark/mytest 中 meta.json 引用的媒体；输入为一个或多个数据根目录，输出为本地 HTTP 页面中的可视化索引与媒体预览。"""
import argparse
import html
import json
import mimetypes
import os
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


DEFAULT_ROOTS = [
    Path("/data/gaoya/dataset/vLAR-PhysInOne/mytest"),
    Path("/data/gaoya/dataset/physics-iq-benchmark/mytest"),
]


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class MediaItem:
    key: str
    path: str
    exists: bool
    media_type: str


@dataclass
class SampleItem:
    dataset: str
    sample_id: str
    meta_path: str
    caption: str
    tags: dict[str, Any]
    media: list[MediaItem]
    raw_meta: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize videos referenced by meta.json files from multiple datasets."
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=DEFAULT_ROOTS,
        help="Dataset roots to scan recursively for meta.json files.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of samples to show after sorting.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Meta Video Browser",
        help="Page title shown in the browser.",
    )
    return parser.parse_args()


def normalize_path(value: Any, meta_dir: Path) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (meta_dir / path).resolve()
    return str(path)


def infer_media_type(path: str) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return None


def simplify_tags(meta: dict[str, Any]) -> dict[str, Any]:
    preferred_keys = [
        "group_id",
        "group_name",
        "split",
        "camera_name",
        "physics_types",
        "category",
        "scenario",
        "take",
        "fps",
        "perspective",
        "context_frames",
        "future_frames",
        "context_frame_range",
        "future_frame_range",
    ]
    return {key: meta[key] for key in preferred_keys if key in meta}


def build_media(meta: dict[str, Any], meta_dir: Path) -> list[MediaItem]:
    media_items: list[MediaItem] = []
    paths = meta.get("paths", {})
    if not isinstance(paths, dict):
        return media_items
    for key, value in paths.items():
        resolved = normalize_path(value, meta_dir)
        if resolved is None:
            continue
        media_type = infer_media_type(resolved)
        if media_type is None:
            continue
        media_items.append(
            MediaItem(
                key=key,
                path=resolved,
                exists=os.path.isfile(resolved),
                media_type=media_type,
            )
        )
    media_items.sort(key=lambda item: (item.media_type, item.key))
    return media_items


def collect_samples(roots: list[Path], limit: int | None) -> list[SampleItem]:
    samples: list[SampleItem] = []
    for root in roots:
        root = root.resolve()
        dataset_name = root.parent.name if root.name == "mytest" else root.name
        for meta_path in sorted(root.rglob("meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                meta = {
                    "sample_id": meta_path.parent.name,
                    "caption": f"Failed to parse meta.json: {exc}",
                    "paths": {},
                }
            sample_id = str(meta.get("sample_id") or meta_path.parent.name)
            caption = str(meta.get("caption") or meta.get("description") or "")
            samples.append(
                SampleItem(
                    dataset=dataset_name,
                    sample_id=sample_id,
                    meta_path=str(meta_path),
                    caption=caption,
                    tags=simplify_tags(meta),
                    media=build_media(meta, meta_path.parent),
                    raw_meta=meta,
                )
            )
    samples.sort(key=lambda item: (item.dataset, item.sample_id))
    if limit is not None:
        samples = samples[:limit]
    return samples


def media_url(path: str) -> str:
    return f"/media?path={quote(path, safe='/')}"


def render_tags(tags: dict[str, Any]) -> str:
    if not tags:
        return '<div class="muted">No extra tags</div>'
    parts = []
    for key, value in tags.items():
        pretty_value = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
        parts.append(
            f'<span class="tag"><strong>{html.escape(key)}</strong>: {html.escape(pretty_value)}</span>'
        )
    return "".join(parts)


def render_media(sample: SampleItem) -> str:
    if not sample.media:
        return '<div class="empty">No media paths found in meta["paths"].</div>'
    blocks = []
    for item in sample.media:
        path_html = html.escape(item.path)
        key_html = html.escape(item.key)
        if not item.exists:
            blocks.append(
                f"""
<div class="media-card missing">
  <div class="media-head">{key_html}</div>
  <div class="muted">Missing file</div>
  <code>{path_html}</code>
</div>
"""
            )
            continue
        if item.media_type == "video":
            blocks.append(
                f"""
<div class="media-card">
  <div class="media-head">{key_html}</div>
  <video controls preload="metadata" src="{html.escape(media_url(item.path))}"></video>
  <code>{path_html}</code>
</div>
"""
            )
        else:
            blocks.append(
                f"""
<div class="media-card">
  <div class="media-head">{key_html}</div>
  <img loading="lazy" src="{html.escape(media_url(item.path))}" alt="{key_html}">
  <code>{path_html}</code>
</div>
"""
            )
    return "\n".join(blocks)


def render_page(title: str, roots: list[Path], samples: list[SampleItem]) -> str:
    dataset_names = sorted({sample.dataset for sample in samples})
    dataset_options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
        for name in dataset_names
    )
    cards = []
    for index, sample in enumerate(samples, start=1):
        raw_meta = html.escape(json.dumps(sample.raw_meta, ensure_ascii=False, indent=2))
        search_blob = " ".join(
            [
                str(sample.dataset),
                str(sample.sample_id),
                str(sample.caption),
                json.dumps(sample.tags, ensure_ascii=False),
            ]
        ).lower()
        cards.append(
            f"""
<section class="sample-card" data-dataset="{html.escape(sample.dataset)}" data-search="{html.escape(search_blob)}">
  <div class="sample-head">
    <div>
      <div class="eyebrow">{html.escape(sample.dataset)} · #{index:04d}</div>
      <h2>{html.escape(sample.sample_id)}</h2>
    </div>
    <div class="meta-link">
      <a href="{html.escape(media_url(sample.meta_path))}" target="_blank" rel="noopener">meta.json</a>
    </div>
  </div>
  <p class="caption">{html.escape(sample.caption) if sample.caption else '&lt;No caption&gt;'}</p>
  <div class="tags">{render_tags(sample.tags)}</div>
  <div class="media-grid">
    {render_media(sample)}
  </div>
  <details>
    <summary>Raw Meta</summary>
    <pre>{raw_meta}</pre>
  </details>
</section>
"""
        )

    roots_text = "".join(
        f"<li><code>{html.escape(str(root.resolve()))}</code></li>" for root in roots
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: rgba(255, 252, 246, 0.92);
      --panel-strong: #fffaf0;
      --ink: #1f1e1a;
      --muted: #6d6659;
      --line: #d8cfbf;
      --accent: #8b4513;
      --accent-soft: #ead7bd;
      --missing: #fff0ee;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Noto Serif SC", serif;
      background:
        radial-gradient(circle at top left, rgba(245, 220, 180, 0.55), transparent 24%),
        radial-gradient(circle at bottom right, rgba(190, 220, 210, 0.45), transparent 22%),
        linear-gradient(180deg, #f7f2e8 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px 24px;
      box-shadow: 0 18px 48px rgba(78, 62, 34, 0.08);
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.6;
      max-width: 980px;
    }}
    .hero ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 320px) minmax(220px, 1fr) auto;
      gap: 12px;
      margin-top: 18px;
      align-items: end;
    }}
    .control {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .control label {{
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .control select,
    .control input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 15px;
      background: var(--panel-strong);
      color: var(--ink);
    }}
    .control-status {{
      color: var(--muted);
      font-size: 14px;
      padding-bottom: 10px;
      text-align: right;
    }}
    .sample-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px 18px 20px;
      margin-bottom: 20px;
      box-shadow: 0 12px 34px rgba(66, 53, 31, 0.06);
    }}
    .sample-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }}
    .eyebrow {{
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 12px;
      margin-bottom: 8px;
    }}
    h2 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }}
    .meta-link a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .caption {{
      font-size: 18px;
      line-height: 1.6;
      margin: 14px 0 14px;
    }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 16px;
    }}
    .tag {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      background: var(--accent-soft);
      border: 1px solid #d9bea1;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 13px;
      line-height: 1.2;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
    }}
    .media-card {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }}
    .media-card.missing {{
      background: var(--missing);
      border-color: #e4b2aa;
    }}
    .media-head {{
      font-weight: 700;
      margin-bottom: 10px;
    }}
    video, img {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      display: block;
      max-height: 420px;
      object-fit: contain;
    }}
    code {{
      display: block;
      margin-top: 10px;
      padding: 8px 10px;
      border-radius: 10px;
      background: #f2ebdf;
      color: #4a433a;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.5;
    }}
    details {{
      margin-top: 14px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f5efe3;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      margin: 10px 0 0;
      overflow: auto;
      line-height: 1.45;
    }}
    .muted {{
      color: var(--muted);
    }}
    .empty {{
      color: var(--muted);
      padding: 12px;
      border: 1px dashed var(--line);
      border-radius: 14px;
      background: #faf7f1;
    }}
    .sample-card.hidden {{
      display: none;
    }}
    @media (max-width: 720px) {{
      main {{
        padding: 18px 14px 28px;
      }}
      .hero, .sample-card {{
        border-radius: 16px;
      }}
      .sample-head {{
        flex-direction: column;
      }}
      .caption {{
        font-size: 16px;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
      .control-status {{
        text-align: left;
        padding-bottom: 0;
      }}
      .media-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>Scanned <strong>{len(samples)}</strong> samples by recursively reading every <code>meta.json</code> under the provided roots. Each card shows the caption and every media file referenced under <code>meta["paths"]</code> that looks like a video or image.</p>
      <ul>{roots_text}</ul>
      <div class="controls">
        <div class="control">
          <label for="dataset-filter">Test Set</label>
          <select id="dataset-filter">
            <option value="__all__">All</option>
            {dataset_options}
          </select>
        </div>
        <div class="control">
          <label for="keyword-filter">Keyword</label>
          <input id="keyword-filter" type="text" placeholder="sample id / caption / tag">
        </div>
        <div class="control-status" id="filter-status">Showing {len(samples)} / {len(samples)}</div>
      </div>
    </section>
    {''.join(cards)}
  </main>
  <script>
    (function() {{
      const datasetFilter = document.getElementById('dataset-filter');
      const keywordFilter = document.getElementById('keyword-filter');
      const status = document.getElementById('filter-status');
      const cards = Array.from(document.querySelectorAll('.sample-card'));

      function applyFilters() {{
        const dataset = datasetFilter.value;
        const keyword = keywordFilter.value.trim().toLowerCase();
        let visible = 0;
        for (const card of cards) {{
          const datasetOk = dataset === '__all__' || card.dataset.dataset === dataset;
          const searchText = card.dataset.search || '';
          const keywordOk = keyword === '' || searchText.includes(keyword);
          const show = datasetOk && keywordOk;
          card.classList.toggle('hidden', !show);
          if (show) visible += 1;
        }}
        status.textContent = `Showing ${{visible}} / {len(samples)}`;
      }}

      datasetFilter.addEventListener('change', applyFilters);
      keywordFilter.addEventListener('input', applyFilters);
      applyFilters();
    }})();
  </script>
</body>
</html>
"""


def build_handler(html_text: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                payload = html_text.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if parsed.path == "/media":
                params = parse_qs(parsed.query)
                raw_path = params.get("path", [None])[0]
                if raw_path is None:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing 'path' query parameter.")
                    return
                file_path = Path(unquote(raw_path))
                if not file_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, f"File not found: {file_path}")
                    return
                content_type, _ = mimetypes.guess_type(str(file_path))
                if content_type is None:
                    content_type = "application/octet-stream"
                file_size = file_path.stat().st_size
                range_header = self.headers.get("Range")
                if range_header:
                    self._serve_range(file_path, content_type, file_size, range_header)
                    return
                with file_path.open("rb") as handle:
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(handle.read())
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown route.")

        def _serve_range(
            self,
            file_path: Path,
            content_type: str,
            file_size: int,
            range_header: str,
        ) -> None:
            if not range_header.startswith("bytes="):
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Unsupported range.")
                return
            start_text, _, end_text = range_header[len("bytes="):].partition("-")
            if start_text == "":
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range.")
                return
            start = int(start_text)
            end = int(end_text) if end_text else file_size - 1
            if start < 0 or end >= file_size or start > end:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Range out of bounds.")
                return
            length = end - start + 1
            with file_path.open("rb") as handle:
                handle.seek(start)
                payload = handle.read(length)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[http] {self.address_string()} - {fmt % args}")

    return Handler


def main() -> None:
    args = parse_args()
    samples = collect_samples(list(args.roots), args.limit)
    html_text = render_page(args.title, list(args.roots), samples)
    handler = build_handler(html_text)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {len(samples)} samples on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
