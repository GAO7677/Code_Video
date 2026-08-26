#!/usr/bin/env python3
"""Serve a frame-accurate CYCLES RGB and dynamic-GT-mask viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_VIEWER_ROOT = Path(__file__).resolve().parent
ALIGNED_TRUTH_NAME = "physv_v2v_0819_cycles_aligned_truth_v1"

# BGR, chosen to remain legible over the muted CYCLES frames.
PALETTE_BGR = (
    (34, 211, 238),
    (245, 158, 11),
    (168, 85, 247),
    (52, 211, 153),
    (244, 63, 94),
    (96, 165, 250),
    (163, 230, 53),
    (251, 113, 133),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_case_index(dataset_root: Path, truth_root: Path) -> list[dict]:
    cases: list[dict] = []
    samples_root = dataset_root / "samples"
    for sample_dir in sorted(samples_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        case_id = sample_dir.name
        metadata = read_json(sample_dir / "metadata.json")
        manifest = read_json(sample_dir / "manifest.json")
        truth_dir = truth_root / "cases" / case_id
        truth_metadata = read_json(truth_dir / "truth_metadata.json")
        cycles_video = sample_dir / "videos" / "rgb_cycles.mp4"
        resolution = truth_metadata.get("resolution", [0, 0])
        width, height = int(resolution[0]), int(resolution[1])
        simulation = metadata.get("simulation", {})
        control = metadata.get("control", {})
        cases.append(
            {
                "case_id": case_id,
                "family_key": metadata.get("family_key", manifest.get("family_key", "")),
                "source_group": metadata.get("source_group", manifest.get("source_group", "")),
                "taxonomy": metadata.get("taxonomy", manifest.get("taxonomy", "")),
                "title": metadata.get("title", case_id),
                "task_type": metadata.get("task_type", manifest.get("task_type", "")),
                "control": control,
                "dynamic_objects": truth_metadata.get(
                    "dynamic_objects", manifest.get("dynamic_actors", [])
                ),
                "frame_count": int(truth_metadata.get("frame_count", simulation.get("frame_count", 0))),
                "fps": float(truth_metadata.get("fps", 30.0)),
                "width": width,
                "height": height,
                "video": str(cycles_video),
                "truth_dir": str(truth_dir),
                "mask_shape": truth_metadata.get("mask_shape", []),
                "scene": truth_metadata.get("room_scene", {}).get("name", ""),
            }
        )
    return cases


class FrameViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, dataset_root: Path, viewer_root: Path):
        super().__init__(address, handler)
        self.dataset_root = dataset_root.resolve()
        self.viewer_root = viewer_root.resolve()
        self.truth_root = self.dataset_root / ALIGNED_TRUTH_NAME
        self.cases = build_case_index(self.dataset_root, self.truth_root)
        self.case_by_id = {case["case_id"]: case for case in self.cases}
        self.mask_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, list[str]]] = OrderedDict()
        self.mask_cache_lock = threading.RLock()

    def load_masks(self, case_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        with self.mask_cache_lock:
            cached = self.mask_cache.get(case_id)
            if cached is not None:
                self.mask_cache.move_to_end(case_id)
                return cached

        truth_dir = Path(self.case_by_id[case_id]["truth_dir"])
        with np.load(truth_dir / "dynamic_masks.npz", allow_pickle=False) as bundle:
            masks = np.asarray(bundle["masks_thw"], dtype=bool)
            union = np.asarray(bundle["union_thw"], dtype=bool)
            names = [str(value) for value in bundle["object_names"].tolist()]
        # Blender's EXR image buffer is bottom-up, while the saved video and
        # trajectory_pixels use a top-left pixel origin. Keep the source NPZ
        # untouched and normalize only the viewer representation.
        masks = np.flip(masks, axis=-2).copy()
        union = np.flip(union, axis=-2).copy()
        loaded = (masks, union, names)
        with self.mask_cache_lock:
            self.mask_cache[case_id] = loaded
            self.mask_cache.move_to_end(case_id)
            while len(self.mask_cache) > 2:
                self.mask_cache.popitem(last=False)
        return loaded


class FrameViewerHandler(BaseHTTPRequestHandler):
    server: FrameViewerServer
    server_version = "PhysVCyclesGTFrameViewer/1.0"

    def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(body, "application/json; charset=utf-8", status)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            self.send_error_json("File not found", 404)
            return
        self.send_bytes(
            payload,
            content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )

    @staticmethod
    def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        if mask.shape == (height, width):
            return mask
        return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

    def _read_rgb_frame(self, case: dict, frame_index: int) -> np.ndarray:
        capture = cv2.VideoCapture(case["video"])
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to decode {case['case_id']} frame {frame_index}")
        return frame

    def _render_frame(self, case: dict, frame_index: int, view: str, mask_mode: str) -> np.ndarray:
        rgb = self._read_rgb_frame(case, frame_index)
        if view == "rgb":
            return rgb

        masks, union, _ = self.server.load_masks(case["case_id"])
        height, width = rgb.shape[:2]
        if frame_index >= masks.shape[1] or frame_index >= union.shape[0]:
            raise RuntimeError(f"Mask frame {frame_index} is unavailable for {case['case_id']}")
        if mask_mode == "union":
            frame_masks = [self._resize_mask(union[frame_index], height, width)]
        else:
            frame_masks = [self._resize_mask(mask[frame_index], height, width) for mask in masks]

        if view == "mask":
            canvas = np.zeros_like(rgb)
            for mask_index, mask in enumerate(frame_masks):
                color = PALETTE_BGR[mask_index % len(PALETTE_BGR)]
                canvas[mask] = color
            return canvas

        if view != "overlay":
            raise RuntimeError(f"Unknown view: {view}")
        canvas = rgb.copy()
        for mask_index, mask in enumerate(frame_masks):
            color = np.asarray(PALETTE_BGR[mask_index % len(PALETTE_BGR)], dtype=np.float32)
            canvas[mask] = (0.56 * canvas[mask].astype(np.float32) + 0.44 * color).astype(np.uint8)
            contour_mask = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, PALETTE_BGR[mask_index % len(PALETTE_BGR)], 2)
        return canvas

    def handle_request(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.send_file(self.server.viewer_root / "cycles_gt_mask_frame_viewer.html", "text/html; charset=utf-8")
            return
        if path == "/api/cases":
            self.send_json(
                {
                    "schema_version": "physv_cycles_aligned_truth_frame_viewer_v1",
                    "case_count": len(self.server.cases),
                    "mask_semantics": "CYCLES-aligned simulator ground-truth masks for dynamic objects only",
                    "display_transform": "vertical flip from Blender bottom-up EXR buffer to top-left video coordinates; source NPZ unchanged",
                    "truth_root": str(self.server.truth_root),
                    "cases": self.server.cases,
                }
            )
            return
        if path == "/api/health":
            self.send_json({"ok": True, "case_count": len(self.server.cases)})
            return
        if path == "/api/frame":
            query = parse_qs(parsed.query)
            case_id = unquote(query.get("case_id", [""])[0])
            if case_id not in self.server.case_by_id:
                self.send_error_json("Unknown case_id", 404)
                return
            try:
                frame_index = int(query.get("frame", ["0"])[0])
            except ValueError:
                self.send_error_json("frame must be an integer")
                return
            case = self.server.case_by_id[case_id]
            if frame_index < 0 or frame_index >= case["frame_count"]:
                self.send_error_json("frame is outside the case range")
                return
            view = query.get("view", ["overlay"])[0]
            mask_mode = query.get("mask", ["objects"])[0]
            try:
                frame = self._render_frame(case, frame_index, view, mask_mode)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            except (OSError, RuntimeError, KeyError, ValueError) as exc:
                self.send_error_json(str(exc), 500)
                return
            if not ok:
                self.send_error_json("Unable to encode rendered frame", 500)
                return
            self.send_bytes(encoded.tobytes(), "image/jpeg")
            return
        self.send_error_json("Not found", 404)

    def do_GET(self) -> None:
        self.handle_request()

    def do_HEAD(self) -> None:
        self.handle_request()

    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWER_ROOT)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    truth_root = dataset_root / ALIGNED_TRUTH_NAME
    if not (dataset_root / "manifest.json").is_file():
        raise FileNotFoundError(dataset_root / "manifest.json")
    if not (truth_root / "cases").is_dir():
        raise FileNotFoundError(truth_root / "cases")
    if not (args.viewer_root / "cycles_gt_mask_frame_viewer.html").is_file():
        raise FileNotFoundError(args.viewer_root / "cycles_gt_mask_frame_viewer.html")

    server = FrameViewerServer(
        (args.host, args.port), FrameViewerHandler, dataset_root, args.viewer_root
    )
    print(f"viewer=http://{args.host}:{args.port}/", flush=True)
    print(f"dataset={dataset_root}", flush=True)
    print(f"aligned_truth={truth_root}", flush=True)
    print(f"cases={len(server.cases)}", flush=True)
    print("serving_foreground=true", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
