#!/usr/bin/env python3
"""Serve the PhysV videos and Qwen3-VL answers in a local browser viewer."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


DEFAULT_RESULTS = Path("/data/gaoya/agent-data/outputs/physv_qwen3vl/cases.jsonl")
DEFAULT_VIEWER_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_qwen3vl/viewer"
)


def load_rows(results_path):
    rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, results_path, viewer_root):
        super().__init__(address, handler)
        self.results_path = results_path
        self.viewer_root = viewer_root


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "PhysVViewer/1.0"

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

    def _row_media_path(self, case_id, field):
        for row in self._rows():
            if row.get("case_id") == case_id:
                raw_path = row.get(field)
                if not raw_path:
                    return None
                path = Path(raw_path)
                if path.is_file():
                    return path
                return None
        return None

    def _video_path(self, case_id):
        return self._row_media_path(case_id, "video")

    def _context_video_path(self, case_id):
        return self._row_media_path(case_id, "context_video")

    def _context16_video_path(self, case_id):
        return self._row_media_path(case_id, "context16_video")

    def _handle(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/results":
            rows = []
            for row in self._rows():
                public_row = dict(row)
                public_row["video_url"] = "/media/" + quote(
                    row["case_id"], safe=""
                )
                if row.get("context_video"):
                    public_row["context_video_url"] = "/media-context/" + quote(
                        row["case_id"], safe=""
                    )
                if row.get("context16_video"):
                    public_row["context16_video_url"] = "/media-context16/" + quote(
                        row["case_id"], safe=""
                    )
                rows.append(public_row)
            payload = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8")
            return

        if path.startswith("/media/"):
            case_id = unquote(path.removeprefix("/media/"))
            video_path = self._video_path(case_id)
            if video_path is None:
                self._send_bytes(b"Video not found", "text/plain; charset=utf-8", 404)
                return
            self._send_file(video_path, "video/mp4")
            return

        if path.startswith("/media-context/"):
            case_id = unquote(path.removeprefix("/media-context/"))
            video_path = self._context_video_path(case_id)
            if video_path is None:
                self._send_bytes(
                    b"Context video not found", "text/plain; charset=utf-8", 404
                )
                return
            self._send_file(video_path, "video/mp4")
            return

        if path.startswith("/media-context16/"):
            case_id = unquote(path.removeprefix("/media-context16/"))
            video_path = self._context16_video_path(case_id)
            if video_path is None:
                self._send_bytes(
                    b"Context16 video not found", "text/plain; charset=utf-8", 404
                )
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
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWER_ROOT)
    args = parser.parse_args()

    if not args.results.is_file():
        raise FileNotFoundError(args.results)
    args.viewer_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results)
    server = ViewerServer(
        (args.host, args.port),
        ViewerHandler,
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
