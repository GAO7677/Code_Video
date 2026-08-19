#!/usr/bin/env python3
"""Serve the unified PhysV V2V dataset with its video captions."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_VIEWER_ROOT = Path(__file__).with_name("physv_dataset_viewer")
MEDIA_FILES = {
    "full": ("videos/rgb.mp4", "video/mp4"),
    "context8": ("context/context8.mp4", "video/mp4"),
    "context16": ("context/context16.mp4", "video/mp4"),
    "masks": ("videos/masks.mp4", "video/mp4"),
    "depth": ("videos/depth.mp4", "video/mp4"),
    "trajectory": ("videos/trajectory.mp4", "video/mp4"),
    "contacts": ("videos/contacts.mp4", "video/mp4"),
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(dataset_root: Path) -> list[dict]:
    cases = []
    for sample_dir in sorted((dataset_root / "samples").iterdir()):
        if not sample_dir.is_dir():
            continue
        metadata = _read_json(sample_dir / "metadata.json")
        wrapper = _read_json(sample_dir / "meta.json")
        manifest = _read_json(sample_dir / "manifest.json")
        captions = metadata.get("captions", {})
        specific = captions.get("specific", {})
        abstract = captions.get("abstract", {})
        physics = _read_json(sample_dir / "physics_supervision.json")
        cases.append(
            {
                "case_id": sample_dir.name,
                "source_group": metadata.get("source_group", manifest.get("source_group", "")),
                "family_key": metadata.get("family_key", manifest.get("family_key", "")),
                "task_type": metadata.get("task_type", manifest.get("task_type", "")),
                "title": metadata.get("title", sample_dir.name),
                "scene_description": metadata.get("scene_description_simulator_only", ""),
                "scene_style": metadata.get("scene_style", ""),
                "control": metadata.get("control", {}),
                "conditioning": metadata.get("conditioning", {}),
                "simulation": metadata.get("simulation", {}),
                "captions": {
                    "specific": specific.get("text", ""),
                    "abstract": abstract.get("text", ""),
                },
                "physics": {
                    "contact_point_count": physics.get("contact_point_count", 0),
                    "objects": physics.get("objects", {}),
                },
                "dynamic_actors": manifest.get("dynamic_actors", []),
                "video": wrapper.get("video", {}),
            }
        )
    return cases


class DatasetViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, dataset_root: Path, viewer_root: Path):
        super().__init__(address, handler)
        self.dataset_root = dataset_root
        self.viewer_root = viewer_root


class DatasetViewerHandler(BaseHTTPRequestHandler):
    server: DatasetViewerServer
    server_version = "PhysVDatasetViewer/1.0"

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_file(self, file_path: Path, content_type: str | None = None) -> None:
        try:
            size = file_path.stat().st_size
        except FileNotFoundError:
            self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return

        content_type = content_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
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
        self.send_header("Cache-Control", "no-store")
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

    def _case_dir(self, case_id: str) -> Path | None:
        candidate = self.server.dataset_root / "samples" / case_id
        if not candidate.is_dir() or candidate.name != case_id:
            return None
        return candidate

    def _handle(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/cases":
            payload = json.dumps(
                load_cases(self.server.dataset_root), ensure_ascii=False
            ).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8")
            return
        if path == "/download/manifest.json":
            self._send_file(self.server.dataset_root / "manifest.json", "application/json; charset=utf-8")
            return
        if path in ("/", "/index.html"):
            self._send_file(self.server.viewer_root / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/media/"):
            parts = [unquote(part) for part in path.split("/")]
            if len(parts) == 4:
                case_dir = self._case_dir(parts[2])
                media_spec = MEDIA_FILES.get(parts[3])
                if case_dir is not None and media_spec is not None:
                    relative_path, content_type = media_spec
                    self._send_file(case_dir / relative_path, content_type)
                    return
            self._send_bytes(b"Media not found", "text/plain; charset=utf-8", 404)
            return
        self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWER_ROOT)
    args = parser.parse_args()
    if not (args.dataset_root / "manifest.json").is_file():
        raise FileNotFoundError(args.dataset_root / "manifest.json")
    if not (args.viewer_root / "index.html").is_file():
        raise FileNotFoundError(args.viewer_root / "index.html")
    cases = load_cases(args.dataset_root)
    server = DatasetViewerServer(
        (args.host, args.port), DatasetViewerHandler, args.dataset_root, args.viewer_root
    )
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print(f"dataset={args.dataset_root}", flush=True)
    print(f"cases={len(cases)}", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
