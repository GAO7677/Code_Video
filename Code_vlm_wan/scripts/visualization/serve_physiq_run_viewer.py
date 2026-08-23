#!/usr/bin/env python3
"""Serve a grouped Physics-IQ video generation audit page."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import fmean
from urllib.parse import unquote, urlparse


DEFAULT_RUN_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/"
    "train_stage1b_diffsynth_native0705_0705/"
    "run_gpu0235_20260703_step-002500_steps40_512x896_ctx08_49f_defaultnegprompt"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEWER_ROOT = PROJECT_ROOT / "physiq_run_viewer"

CATEGORY_ORDER = (
    "Solid_Mechanics",
    "Fluid_Dynamics",
    "Optics",
    "Magnetism",
    "Thermodynamics",
)
CATEGORY_LABELS = {
    "Solid_Mechanics": "固体力学 · Solid Mechanics",
    "Fluid_Dynamics": "流体动力学 · Fluid Dynamics",
    "Optics": "光学 · Optics",
    "Magnetism": "磁学 · Magnetism",
    "Thermodynamics": "热力学 · Thermodynamics",
}
CATEGORY_PATTERN = re.compile(
    r"physicIQ_(?:\d+_)?"
    r"(Fluid_Dynamics|Solid_Mechanics|Optics|Magnetism|Thermodynamics)_"
)
METRIC_KEYS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
    "vbench_dynamic_degree",
    "vbench_motion_smoothness",
    "vbench_subject_consistency",
    "videophy2",
    "cosmos_reason1",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(value: object) -> float | None:
    if not isinstance(value, dict):
        return None
    for key in ("score", "physics_iq_score", "pmf_score", "raw_dimension_score"):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return None


def category_from_name(name: str) -> str:
    match = CATEGORY_PATTERN.match(name)
    if match:
        return match.group(1)
    return "Other"


def case_title(stem: str, category: str) -> str:
    marker = f"{category}_"
    tail = stem.split(marker, 1)[-1] if marker in stem else stem
    tail = re.sub(r"^\d+_", "", tail)
    tail = re.sub(r"^perspective-center_trimmed-?", "", tail)
    return tail.replace("_", " ").replace("-", " ").strip() or stem


def path_inside(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (FileNotFoundError, ValueError):
        return None
    return resolved


def public_case(video_path: Path, metadata: dict, root: Path) -> dict:
    stem = video_path.stem
    category = category_from_name(stem)
    context_path = Path(str(metadata.get("input_video", "")))
    context_path = path_inside(context_path, root) if context_path else None
    metrics = {key: score(metadata.get(key)) for key in METRIC_KEYS}
    videophy = metadata.get("videophy2")
    if isinstance(videophy, dict):
        metrics["videophy2_joint_pass"] = (
            float(videophy["joint_pass"])
            if isinstance(videophy.get("joint_pass"), (int, float))
            else None
        )
    else:
        metrics["videophy2_joint_pass"] = None

    model_args = metadata.get("model_args")
    if not isinstance(model_args, dict):
        model_args = {}
    checkpoint = str(metadata.get("ckpt", ""))
    parameters = {
        "context_frames": metadata.get("requested_context_frames"),
        "effective_context_frames": metadata.get("effective_context_frames"),
        "frame_indices": metadata.get("frame_indices", []),
        "seed": metadata.get("seed"),
        "step": metadata.get("step"),
        "guidance": metadata.get("guidance"),
        "sampling_mode": metadata.get("sampling_mode"),
        "model_device": metadata.get("model_device"),
        "checkpoint": Path(checkpoint).name if checkpoint else None,
        "height": model_args.get("height"),
        "width": model_args.get("width"),
        "num_frames": model_args.get("num_frames"),
    }
    return {
        "case_id": stem,
        "file_name": video_path.name,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "title": case_title(stem, category),
        "caption": metadata.get("input_caption", ""),
        "parameters": parameters,
        "metrics": metrics,
        "context_available": context_path is not None,
        "context_file": context_path.name if context_path else None,
        "metadata_file": f"{stem}.json",
        "video_bytes": video_path.stat().st_size,
        "input_json": metadata.get("input_json"),
    }


def mean_metric(cases: list[dict], key: str) -> float | None:
    values = [case["metrics"].get(key) for case in cases]
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return fmean(values) if values else None


def load_run(root: Path) -> dict:
    cases = []
    errors = []
    for video_path in sorted(root.glob("*.mp4")):
        metadata_path = video_path.with_suffix(".json")
        if not metadata_path.is_file():
            errors.append(f"missing metadata: {video_path.name}")
            continue
        try:
            metadata = read_json(metadata_path)
            cases.append(public_case(video_path, metadata, root))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append(f"{video_path.name}: {error}")

    grouped: dict[str, list[dict]] = {}
    for case in cases:
        grouped.setdefault(case["category"], []).append(case)
    groups = []
    group_order = [*CATEGORY_ORDER, *sorted(set(grouped) - set(CATEGORY_ORDER))]
    for category in group_order:
        members = grouped.get(category, [])
        if not members:
            continue
        groups.append(
            {
                "key": category,
                "label": CATEGORY_LABELS.get(category, category),
                "count": len(members),
                "summary": {
                    "physics_iq_context_mean": mean_metric(
                        members, "physics_iq_with_context"
                    ),
                    "pmf_context_mean": mean_metric(members, "pmf_with_context"),
                    "vbench_dynamic_mean": mean_metric(
                        members, "vbench_dynamic_degree"
                    ),
                    "vbench_motion_mean": mean_metric(
                        members, "vbench_motion_smoothness"
                    ),
                    "videophy2_joint_rate": mean_metric(
                        members, "videophy2_joint_pass"
                    ),
                },
                "cases": members,
            }
        )

    first = cases[0] if cases else None
    parameters = first["parameters"] if first else {}
    return {
        "dataset_root": str(root),
        "run_name": root.name,
        "case_count": len(cases),
        "group_count": len(groups),
        "groups": groups,
        "run_parameters": parameters,
        "errors": errors,
    }


class PhysIQViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, run_root: Path, viewer_root: Path):
        super().__init__(address, handler)
        self.run_root = run_root.resolve()
        self.viewer_root = viewer_root.resolve()

    def payload(self) -> dict:
        return load_run(self.run_root)

    def case_path(self, case_id: str, kind: str) -> tuple[Path, str] | None:
        if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
            return None
        video_path = self.run_root / f"{case_id}.mp4"
        metadata_path = self.run_root / f"{case_id}.json"
        if kind == "video" and video_path.is_file():
            return video_path, "video/mp4"
        if kind == "metadata" and metadata_path.is_file():
            return metadata_path, "application/json; charset=utf-8"
        if kind == "context" and metadata_path.is_file():
            metadata = read_json(metadata_path)
            context_path = Path(str(metadata.get("input_video", "")))
            context_path = path_inside(context_path, self.run_root) if context_path else None
            if context_path and context_path.is_file():
                return context_path, mimetypes.guess_type(context_path.name)[0] or "image/jpeg"
        return None


class PhysIQViewerHandler(BaseHTTPRequestHandler):
    server: PhysIQViewerServer
    server_version = "PhysIQRunViewer/1.0"

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)
            return
        content_type = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                start_text, end_text = range_header[6:].split("-", 1)
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else size - 1
                elif end_text:
                    length = int(end_text)
                    start = max(size - length, 0)
                    end = size - 1
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
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
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _handle(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(self.server.viewer_root / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/data":
            payload = json.dumps(self.server.payload(), ensure_ascii=False).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8")
            return
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) == 3 and parts[0] == "media":
            item = self.server.case_path(parts[2], parts[1])
            if item:
                self._send_file(*item)
                return
        self._send_bytes(b"Not found", "text/plain; charset=utf-8", 404)

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWER_ROOT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    viewer_root = args.viewer_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    if not (viewer_root / "index.html").is_file():
        raise FileNotFoundError(f"Viewer page does not exist: {viewer_root / 'index.html'}")
    server = PhysIQViewerServer((args.host, args.port), PhysIQViewerHandler, run_root, viewer_root)
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print(f"run_root={run_root}", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
