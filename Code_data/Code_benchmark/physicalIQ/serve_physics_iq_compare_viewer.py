#!/usr/bin/env python3
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


DEFAULT_VIEWER_DIR = Path("/data/gaoya/AAA_test_video/Benchmark/physics_IQ_demo/compare_viewer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local Physics-IQ compare viewer.")
    parser.add_argument("--viewer_dir", type=Path, default=DEFAULT_VIEWER_DIR)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18711)
    return parser.parse_args()


def make_handler(viewer_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            raw = unquote(path.split("?", 1)[0])
            if raw.startswith("/files/"):
                return raw[len("/files"):]
            target = viewer_dir / raw.lstrip("/")
            if raw == "/":
                target = viewer_dir / "index.html"
            return str(target)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    return Handler


def main() -> None:
    args = parse_args()
    handler = make_handler(args.viewer_dir.resolve())
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
