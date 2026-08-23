#!/usr/bin/env python3
"""Serve paired context8/full-video caption demo results."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
RESULTS_PATH = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_context8_vs_full/demo_results.jsonl"
)
VIEWER_ROOT = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def send_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str | None = None) -> None:
    if not path.is_file():
        handler.send_error(404, "Not found")
        return
    size = path.stat().st_size
    start, end, status = 0, size - 1, 200
    range_header = handler.headers.get("Range")
    if range_header and range_header.startswith("bytes="):
        start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        elif end_text:
            length = int(end_text)
            start = max(size - length, 0)
        end = min(end, size - 1)
        if start < 0 or start >= size or end < start:
            handler.send_error(416, "Invalid range")
            return
        status = 206
    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(length))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "no-store")
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    if handler.command == "HEAD":
        return
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


class ComparisonServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, dataset_root: Path, results_path: Path):
        super().__init__(address, handler)
        self.dataset_root = dataset_root
        self.results_path = results_path


class ComparisonHandler(BaseHTTPRequestHandler):
    server: ComparisonServer

    def do_GET(self) -> None:
        self.handle_request()

    def do_HEAD(self) -> None:
        self.handle_request()

    def handle_request(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            send_file(self, VIEWER_ROOT / "context8_vs_full_0819.html", "text/html; charset=utf-8")
            return
        if path == "/api/cases":
            payload = json.dumps(read_rows(self.server.results_path), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
            return
        if path.startswith("/media/"):
            parts = [unquote(part) for part in path.removeprefix("/media/").split("/") if part]
            if len(parts) != 2 or Path(parts[0]).name != parts[0] or parts[1] not in {"context8", "full"}:
                self.send_error(404, "Invalid media path")
                return
            relative = "context/context8_cycles.mp4" if parts[1] == "context8" else "videos/rgb_cycles.mp4"
            sample_root = (self.server.dataset_root / "samples" / parts[0]).resolve()
            candidate = (sample_root / relative).resolve()
            if sample_root not in candidate.parents:
                self.send_error(404, "Invalid media path")
                return
            send_file(self, candidate, "video/mp4")
            return
        self.send_error(404, "Not found")

    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8921)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()
    server = ComparisonServer((args.host, args.port), ComparisonHandler, args.dataset_root, args.results)
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print(f"results={args.results}", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
