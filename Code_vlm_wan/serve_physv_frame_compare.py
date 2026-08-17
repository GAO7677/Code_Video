#!/usr/bin/env python3
"""Serve prefix-frame and full-video comparison results in a local viewer."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


DEFAULT_RESULTS = Path(
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/0613_phyco_frame_compare.jsonl"
)
DEFAULT_VIEWER_ROOT = Path(
    "/home/gaoya/Code_Video/Code_vlm_wan/physv_frame_compare_viewer"
)


def load_rows(results_path):
    rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CompareServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, results_path, viewer_root):
        super().__init__(address, handler)
        self.results_path = results_path
        self.viewer_root = viewer_root


class CompareHandler(BaseHTTPRequestHandler):
    server_version = "PhysVFrameCompare/1.0"

    def _send_bytes(self, payload, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_file(self, file_path, content_type=None):
        try:
            size = file_path.stat().st_size
        except FileNotFoundError:
            self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return

        content_type = content_type or mimetypes.guess_type(file_path.name)[0]
        content_type = content_type or "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if range_header and range_header.startswith("bytes="):
            value = range_header.removeprefix("bytes=").split(",", 1)[0]
            start_text, end_text = value.split("-", 1)
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            elif end_text:
                length = int(end_text)
                start = max(size - length, 0)
                end = size - 1
            if start < 0 or start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return

        with file_path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _rows(self):
        return load_rows(self.server.results_path)

    def _variant_path(self, case_id, variant_key, field="video"):
        for row in self._rows():
            if row.get("case_id") == case_id:
                variant = row.get("variants", {}).get(variant_key)
                if variant:
                    path = Path(variant.get(field, ""))
                    return path if path.is_file() else None
        return None

    def _parse_media_token(self, path, prefix):
        token = unquote(path.removeprefix(prefix))
        try:
            return token.rsplit("/", 1)
        except ValueError:
            return None

    def _handle(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/results":
            public_rows = []
            for row in self._rows():
                public_row = dict(row)
                public_variants = {}
                for variant_key, variant in row.get("variants", {}).items():
                    public_variant = dict(variant)
                    token = f"{row['case_id']}/{variant_key}"
                    public_variant["video_url"] = "/media/" + quote(token, safe="")
                    if public_variant.get("vlm_input_video"):
                        public_variant["vlm_input_video_url"] = (
                            "/vlm-input/" + quote(token, safe="")
                        )
                    public_variants[variant_key] = public_variant
                public_row["variants"] = public_variants
                public_rows.append(public_row)
            payload = json.dumps(public_rows, ensure_ascii=False).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8")
            return

        if path.startswith("/media/"):
            parsed_token = self._parse_media_token(path, "/media/")
            if parsed_token is None:
                self._send_bytes(b"Video not found", "text/plain; charset=utf-8", 404)
                return
            case_id, variant_key = parsed_token
            video_path = self._variant_path(case_id, variant_key)
            if video_path is None:
                self._send_bytes(b"Video not found", "text/plain; charset=utf-8", 404)
                return
            self._send_file(video_path, "video/mp4")
            return

        if path.startswith("/vlm-input/"):
            parsed_token = self._parse_media_token(path, "/vlm-input/")
            if parsed_token is None:
                self._send_bytes(b"VLM input replay not found", "text/plain; charset=utf-8", 404)
                return
            case_id, variant_key = parsed_token
            video_path = self._variant_path(case_id, variant_key, "vlm_input_video")
            if video_path is None:
                self._send_bytes(b"VLM input replay not found", "text/plain; charset=utf-8", 404)
                return
            self._send_file(video_path, "video/mp4")
            return

        if path == "/download/results.jsonl":
            self._send_file(self.server.results_path, "application/x-ndjson")
            return

        if path in ("/", "/index.html"):
            self._send_file(self.server.viewer_root / "index.html", "text/html; charset=utf-8")
            return

        self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def do_GET(self):
        self._handle()

    def do_HEAD(self):
        self._handle()

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWER_ROOT)
    args = parser.parse_args()

    if not args.results.is_file():
        raise FileNotFoundError(args.results)
    if not (args.viewer_root / "index.html").is_file():
        raise FileNotFoundError(args.viewer_root / "index.html")
    rows = load_rows(args.results)
    server = CompareServer(
        (args.host, args.port),
        CompareHandler,
        args.results,
        args.viewer_root,
    )
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print(f"results={args.results}", flush=True)
    print(f"cases={len(rows)}", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
