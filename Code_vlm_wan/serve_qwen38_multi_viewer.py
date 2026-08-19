#!/usr/bin/env python3
"""Serve the Qwen3.8 six-case video input/output audit viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_DATA = Path("/data/gaoya/agent-data/outputs/physv_qwen3_8/viewer_six_gpu7_fla")
DEFAULT_VIEWER = Path("/home/gaoya/Code_Video/Code_vlm_wan/qwen38_multi_viewer")


class MultiViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, data_dir: Path, viewer_dir: Path):
        super().__init__(address, handler)
        self.data_dir = data_dir
        self.viewer_dir = viewer_dir

    def cases(self) -> list[dict]:
        with (self.data_dir / "viewer_data.json").open(encoding="utf-8") as handle:
            return json.load(handle)["cases"]


class MultiViewerHandler(BaseHTTPRequestHandler):
    server: MultiViewerServer

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_error(404, "Not found")
            return
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def media_path(self, slug: str, kind: str) -> tuple[Path, str] | None:
        allowed = {
            "source": ("source_video", "video/mp4"),
            "sampled": ("sampled_frames.mp4", "video/mp4"),
            "processor": ("processor_frames.mp4", "video/mp4"),
            "sheet": ("sampled_contact_sheet.jpg", "image/jpeg"),
        }
        if kind not in allowed:
            return None
        name, content_type = allowed[kind]
        for case in self.server.cases():
            if case.get("case_slug") == slug:
                if kind == "source":
                    return Path(case[name]), content_type
                return self.server.data_dir / slug / name, content_type
        return None

    def handle_request(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_file(self.server.viewer_dir / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/data":
            self.send_file(self.server.data_dir / "viewer_data.json", "application/json; charset=utf-8")
            return
        if path.startswith("/media/"):
            parts = unquote(path).split("/")
            if len(parts) == 4:
                item = self.media_path(parts[2], parts[3])
                if item:
                    self.send_file(*item)
                    return
        self.send_error(404, "Not found")

    def do_GET(self) -> None:
        self.handle_request()

    def do_HEAD(self) -> None:
        self.handle_request()

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER)
    args = parser.parse_args()
    if not (args.data_dir / "viewer_data.json").is_file() or not (args.viewer_dir / "index.html").is_file():
        raise FileNotFoundError("Build multi-case viewer data before serving it")
    server = MultiViewerServer((args.host, args.port), MultiViewerHandler, args.data_dir, args.viewer_dir)
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
