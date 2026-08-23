#!/usr/bin/env python3
"""Serve the local Qwen3.8 GPU7 input/output audit viewer."""

from __future__ import annotations

import argparse
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_DATA = Path("/data/gaoya/agent-data/outputs/physv_qwen3_8/viewer_gpu7_fla")
DEFAULT_VIEWER = Path("/home/gaoya/Code_Video/Code_vlm_wan/qwen38_demo_viewer")


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, data_dir: Path, viewer_dir: Path, source_video: Path):
        super().__init__(address, handler)
        self.data_dir = data_dir
        self.viewer_dir = viewer_dir
        self.source_video = source_video


class ViewerHandler(SimpleHTTPRequestHandler):
    server: ViewerServer

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

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        data_files = {
            "/api/data": (self.server.data_dir / "viewer_data.json", "application/json; charset=utf-8"),
            "/media/source-video": (self.server.source_video, "video/mp4"),
            "/media/sampled-frames": (self.server.data_dir / "sampled_frames.mp4", "video/mp4"),
            "/media/processor-frames": (self.server.data_dir / "processor_frames.mp4", "video/mp4"),
            "/media/sampled-sheet": (self.server.data_dir / "sampled_contact_sheet.jpg", "image/jpeg"),
            "/media/processor-sheet": (self.server.data_dir / "processor_contact_sheet.jpg", "image/jpeg"),
        }
        if path in ("/", "/index.html"):
            self.send_file(self.server.viewer_dir / "index.html", "text/html; charset=utf-8")
        elif path in data_files:
            self.send_file(*data_files[path])
        else:
            self.send_error(404, "Not found")

    def do_HEAD(self) -> None:
        self.do_GET()

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER)
    args = parser.parse_args()
    data_path = args.data_dir / "viewer_data.json"
    if not data_path.is_file() or not (args.viewer_dir / "index.html").is_file():
        raise FileNotFoundError("Build viewer data before serving it")
    import json
    source_video = Path(json.loads(data_path.read_text(encoding="utf-8"))["run"]["video"])
    server = ViewerServer((args.host, args.port), ViewerHandler, args.data_dir, args.viewer_dir, source_video)
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
