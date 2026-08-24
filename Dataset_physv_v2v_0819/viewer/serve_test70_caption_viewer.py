#!/usr/bin/env python3
"""Serve the test70 RGB inputs together with qwen38vl_caption outputs."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
INPUT_LIST = DATASET_ROOT / "testjsons/physv_v2v_0819_all_cycles_test70_ctx8.txt"
PROMPT_PATH = Path(
    "/home/gaoya/Code_Video/Dataset_physv_v2v_0819/prompts/describe_this_video.txt"
)
VIEWER_ROOT = Path(__file__).resolve().parent


def read_cases(input_list: Path, dataset_root: Path) -> list[dict]:
    cases: list[dict] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        json_path = Path(line.strip())
        if not json_path.is_file():
            continue
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        case_id = str(record.get("sample_id") or json_path.stem)
        source = Path(str(record.get("source_video", "")))
        if not source.is_file():
            source = dataset_root / "samples" / case_id / "videos" / "rgb_cycles.mp4"
        source = source.resolve()
        sample_root = (dataset_root / "samples" / case_id).resolve()
        video_ok = source.is_file() and sample_root in source.parents
        caption = record.get("qwen38vl_caption")
        caption_ok = isinstance(caption, str) and bool(caption.strip())
        cases.append(
            {
                "case_id": case_id,
                "json_path": str(json_path),
                "video_url": f"/media/{case_id}",
                "video_available": video_ok,
                "caption": caption if caption_ok else "",
                "status": "ready" if caption_ok else "pending",
            }
        )
    return cases


def send_bytes(handler: BaseHTTPRequestHandler, payload: bytes, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if handler.command != "HEAD":
        handler.wfile.write(payload)


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
            start = max(size - int(end_text), 0)
        end = min(end, size - 1)
        if start < 0 or start >= size or end < start:
            handler.send_error(416, "Invalid range")
            return
        status = 206
    length = end - start + 1
    handler.send_response(status)
    handler.send_header(
        "Content-Type",
        content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )
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


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, input_list: Path, dataset_root: Path, prompt_path: Path):
        super().__init__(address, handler)
        self.input_list = input_list
        self.dataset_root = dataset_root
        self.prompt_path = prompt_path


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def do_GET(self) -> None:
        self.handle_request()

    def do_HEAD(self) -> None:
        self.handle_request()

    def handle_request(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            send_file(self, VIEWER_ROOT / "test70_caption_viewer.html", "text/html; charset=utf-8")
            return
        cases = read_cases(self.server.input_list, self.server.dataset_root)
        if path == "/api/cases":
            prompt = self.server.prompt_path.read_text(encoding="utf-8").strip()
            payload = {
                "cases": cases,
                "target_count": len(cases),
                "ready_count": sum(item["status"] == "ready" for item in cases),
                "pending_count": sum(item["status"] == "pending" for item in cases),
                "model": "qwen3.8-27b",
                "input": "full rgb_cycles.mp4",
                "prompt": prompt,
            }
            send_bytes(
                self,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if path.startswith("/media/"):
            case_id = unquote(path.removeprefix("/media/")).strip("/")
            if not case_id or Path(case_id).name != case_id:
                self.send_error(404, "Invalid case")
                return
            item = next((row for row in cases if row["case_id"] == case_id), None)
            if item is None or not item["video_available"]:
                self.send_error(404, "Video not found")
                return
            video_path = (self.server.dataset_root / "samples" / case_id / "videos" / "rgb_cycles.mp4").resolve()
            sample_root = (self.server.dataset_root / "samples" / case_id).resolve()
            if sample_root not in video_path.parents:
                self.send_error(404, "Invalid media path")
                return
            send_file(self, video_path, "video/mp4")
            return
        self.send_error(404, "Not found")

    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--input-list", type=Path, default=INPUT_LIST)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    args = parser.parse_args()
    if not args.input_list.is_file():
        raise FileNotFoundError(args.input_list)
    if not args.prompt.is_file():
        raise FileNotFoundError(args.prompt)
    cases = read_cases(args.input_list, args.dataset_root)
    server = ViewerServer((args.host, args.port), ViewerHandler, args.input_list, args.dataset_root, args.prompt)
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print(f"input_list={args.input_list}", flush=True)
    print(f"model=qwen3.8-27b", flush=True)
    print(f"cases={len(cases)} ready={sum(item['status'] == 'ready' for item in cases)}", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
