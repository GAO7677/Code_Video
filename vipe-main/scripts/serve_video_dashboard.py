#!/usr/bin/env python3
"""Foreground static server with HTTP Range support for MP4 playback."""
from __future__ import annotations

import argparse
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _video_file(self) -> Path | None:
        path = Path(self.translate_path(urlparse(self.path).path))
        if path.suffix.lower() != ".mp4" or not path.is_file():
            return None
        return path

    def _serve_video(self, with_body: bool) -> bool:
        path = self._video_file()
        if path is None:
            return False
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range")
        partial_response = False
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return True
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:
                suffix = int(match.group(2))
                start = max(0, size - suffix)
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return True
            end = min(end, size - 1)
            partial_response = True

        length = end - start + 1
        self.send_response(206 if partial_response else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-cache")
        if partial_response:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if with_body:
            with path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        return True

    def do_GET(self) -> None:
        if self.path.lower().split("?", 1)[0].endswith(".mp4") and self._serve_video(True):
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self.path.lower().split("?", 1)[0].endswith(".mp4") and self._serve_video(False):
            return
        super().do_HEAD()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8850)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        partial(RangeHandler, directory=str(args.root)),
    )
    print(f"serving {args.root} at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

