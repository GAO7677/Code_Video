#!/usr/bin/env python3
"""Serve all PhysV videos alongside deterministic truth descriptions."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from .serve_side_by_side_api_0819 import send_file
except ImportError:  # Direct ``python viewer/serve_truth_description_viewer.py``.
    from serve_side_by_side_api_0819 import send_file


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
TRUTH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_truth_descriptions"
)
VIEWER_ROOT = Path(__file__).resolve().parent
MEDIA_FILES = {
    "cycles": ("videos/rgb_cycles.mp4", "video/mp4"),
    "context8": ("context/context8_cycles.mp4", "video/mp4"),
    "context16": ("context/context16_cycles.mp4", "video/mp4"),
}


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


class TruthViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, dataset_root: Path, truth_root: Path, viewer_root: Path):
        super().__init__(address, handler)
        self.dataset_root = dataset_root
        self.truth_root = truth_root
        self.viewer_root = viewer_root


class TruthViewerHandler(BaseHTTPRequestHandler):
    server: TruthViewerServer

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def _handle(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            send_file(self, self.server.viewer_root / "truth_description_viewer.html", "text/html; charset=utf-8")
            return
        if path == "/api/cases":
            payload = json.dumps(
                read_rows(self.server.truth_root / "truth_descriptions.jsonl"),
                ensure_ascii=False,
            ).encode("utf-8")
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
            if len(parts) != 2 or Path(parts[0]).name != parts[0]:
                self.send_error(404, "Invalid media path")
                return
            relative, content_type = MEDIA_FILES.get(parts[1], (None, None))
            if relative is None:
                self.send_error(404, "Unknown media kind")
                return
            case_root = (self.server.dataset_root / "samples" / parts[0]).resolve()
            candidate = (case_root / relative).resolve()
            samples_root = (self.server.dataset_root / "samples").resolve()
            if samples_root not in candidate.parents:
                self.send_error(404, "Invalid media path")
                return
            send_file(self, candidate, content_type)
            return
        self.send_error(404, "Not found")

    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8923)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--truth-root", type=Path, default=TRUTH_ROOT)
    parser.add_argument("--viewer-root", type=Path, default=VIEWER_ROOT)
    args = parser.parse_args()
    results = args.truth_root / "truth_descriptions.jsonl"
    if not results.is_file():
        raise FileNotFoundError(results)
    server = TruthViewerServer(
        (args.host, args.port),
        TruthViewerHandler,
        args.dataset_root.resolve(),
        args.truth_root.resolve(),
        args.viewer_root.resolve(),
    )
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print(f"dataset={args.dataset_root.resolve()}", flush=True)
    print(f"truth_results={results}", flush=True)
    print(f"cases={len(read_rows(results))}", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
