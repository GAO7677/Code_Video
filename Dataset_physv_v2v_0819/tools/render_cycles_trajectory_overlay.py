#!/usr/bin/env python3
"""Overlay ground-truth dynamic-object trajectories on Cycles videos."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np


DEFAULT_INPUT_LIST = Path(
    "/data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/physv_v2v_0819_all_cycles_test70_ctx8.txt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay"
)
DEFAULT_FFMPEG = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg")

PALETTE_BGR = (
    (55, 220, 255),
    (80, 220, 120),
    (255, 170, 50),
    (220, 90, 220),
    (255, 235, 70),
    (70, 180, 255),
    (180, 110, 255),
    (70, 240, 210),
)

CAMERA_FRAMING_PRESETS = {
    "F11": {"target": (1.088, 0.0, 0.604), "yfov_deg": 42.0},
    "F12": {"target": (1.930, 0.0, 0.620), "yfov_deg": 38.5},
    "V2V_BOWL": {"target": (0.005, 0.0, 0.600), "yfov_deg": 34.5},
    "V2V_DOMINO": {"target": (-0.147, 0.0, 0.480), "yfov_deg": 25.0},
    "V2V_GAP": {"target": (0.288, 0.0, 0.620), "yfov_deg": 35.0},
    "V2V_OBSTACLE": {"target": (-0.260, 0.0, 0.480), "yfov_deg": 24.5},
    "V2V_OBSTACLE_SIZE": {"target": (-0.635, 0.0, 0.480), "yfov_deg": 31.5},
    "V2V_PENDULUM": {"target": (-0.450, 0.0, 1.128), "yfov_deg": 41.5},
    "V2V_PENDULUM_CABINET": {"target": (-0.28, 0.0, 1.48), "yfov_deg": 50.0},
    "V2V_SEESAW": {"target": (0.002, 0.0, 0.460), "yfov_deg": 21.5},
    "SCENE_PUCK_BARRIER": {"target": (0.20, -0.40, 0.24), "yfov_deg": 45.0},
    "SCENE_DOOR_FRAME": {"target": (0.25, 0.0, 0.82), "yfov_deg": 46.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_camera(metadata: dict, width: int, height: int) -> dict[str, np.ndarray | float]:
    camera = metadata["camera"]
    eye = np.asarray(camera["extrinsics"]["eye"], dtype=np.float64)
    family_key = str(metadata.get("family_key", ""))
    render_family = family_key
    if render_family == "F12_RAMP_LENGTH":
        render_family = "F12"
    if render_family == "SCENE_DOOR_FRAME_BALL":
        render_family = "SCENE_DOOR_FRAME"
    framing = CAMERA_FRAMING_PRESETS.get(render_family, {})
    target = np.asarray(framing.get("target", camera["extrinsics"]["target"]), dtype=np.float64)
    up_hint = np.asarray(camera["extrinsics"]["up"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, forward)
    yfov = math.radians(float(framing.get("yfov_deg", camera["intrinsics"].get("yfov_deg", 50.0))))
    aspect = float(width) / float(height)
    fx = 0.5 * width / (math.tan(yfov * 0.5) * aspect)
    fy = 0.5 * height / math.tan(yfov * 0.5)
    return {
        "eye": eye,
        "forward": forward,
        "right": right,
        "up": true_up,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5,
        "cy": height * 0.5,
    }


def project_point(point: np.ndarray, camera: dict[str, np.ndarray | float]) -> tuple[int, int] | None:
    delta = point.astype(np.float64) - camera["eye"]
    x_cam = float(delta @ camera["right"])
    y_cam = float(delta @ camera["up"])
    z_cam = float(delta @ camera["forward"])
    if z_cam <= 1e-6:
        return None
    u = float(camera["fx"]) * (x_cam / z_cam) + float(camera["cx"])
    v = float(camera["cy"]) - float(camera["fy"]) * (y_cam / z_cam)
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    return int(round(u)), int(round(v))


class FfmpegWriter:
    def __init__(
        self,
        ffmpeg: Path,
        target: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        self.target = target
        self.tmp = target.with_name(f".{target.stem}.tmp{target.suffix}")
        if self.tmp.exists():
            self.tmp.unlink()
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.tmp),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            detail = self.process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg stopped while encoding {self.target}: {detail}") from exc

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        detail = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.target}: {detail}")
        self.tmp.replace(self.target)


def draw_legend(frame: np.ndarray, names: list[str], indices: list[int]) -> None:
    if not indices:
        return
    width = min(frame.shape[1] - 16, 390)
    line_height = 23
    height = 34 + line_height * len(indices)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (12, 18, 18), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)
    cv2.putText(
        frame,
        "GT trajectory / dynamic objects",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 245, 240),
        1,
        cv2.LINE_AA,
    )
    for row, object_index in enumerate(indices):
        y = 53 + row * line_height
        color = PALETTE_BGR[row % len(PALETTE_BGR)]
        cv2.line(frame, (18, y - 5), (39, y - 5), color, 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            names[object_index],
            (48, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 228, 220),
            1,
            cv2.LINE_AA,
        )


def overlay_frame(
    frame: np.ndarray,
    frame_index: int,
    projected: list[list[tuple[int, int] | None]],
    names: list[str],
    dynamic_indices: list[int],
    history: dict[int, list[tuple[int, int] | None]],
) -> np.ndarray:
    canvas = frame.copy()
    for color_index, object_index in enumerate(dynamic_indices):
        point = projected[frame_index][object_index]
        points = history[object_index]
        points.append(point)
        color = PALETTE_BGR[color_index % len(PALETTE_BGR)]
        for left, right in zip(points[:-1], points[1:]):
            if left is not None and right is not None:
                cv2.line(canvas, left, right, color, 3, cv2.LINE_AA)
        if point is not None:
            cv2.circle(canvas, point, 8, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 7, (245, 245, 245), 1, cv2.LINE_AA)
    draw_legend(canvas, names, dynamic_indices)
    return canvas


def project_trajectory(
    positions: np.ndarray,
    metadata: dict,
    width: int,
    height: int,
    dynamic_mask: np.ndarray,
) -> tuple[list[list[tuple[int, int] | None]], list[int]]:
    camera = make_camera(metadata, width, height)
    dynamic_indices = [int(index) for index, value in enumerate(dynamic_mask) if bool(value)]
    projected: list[list[tuple[int, int] | None]] = []
    for frame_positions in positions:
        projected.append([project_point(point, camera) for point in frame_positions])
    return projected, dynamic_indices


def render_case(payload: dict, output_root: Path, ffmpeg: Path, crf: int, preset: str) -> dict:
    sample_id = str(payload["sample_id"])
    source_path = Path(payload["source_video"])
    context_path = Path(payload["input_video"])
    metadata_path = Path(payload["metadata_json"])
    supervision_path = Path(payload["physics_supervision_npz"])
    metadata = read_json(metadata_path)
    supervision = np.load(supervision_path, allow_pickle=False)
    positions = np.asarray(supervision["positions_m"], dtype=np.float32)
    dynamic_mask = np.asarray(supervision["dynamic_mask"], dtype=bool)
    object_names = [str(value) for value in supervision["object_names"].tolist()]
    supervision.close()

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if positions.shape[0] < source_frame_count:
        source_frame_count = positions.shape[0]
    projected, dynamic_indices = project_trajectory(
        positions[:source_frame_count], metadata, width, height, dynamic_mask
    )
    if not dynamic_indices:
        raise RuntimeError(f"{sample_id}: no dynamic objects in physics supervision")

    output_video_dir = output_root / "videos"
    source_target = output_video_dir / f"{sample_id}__source_overlay.mp4"
    context_target = output_video_dir / f"{sample_id}__ctx8_overlay.mp4"
    source_writer = FfmpegWriter(ffmpeg, source_target, width, height, fps, crf, preset)
    context_writer = FfmpegWriter(ffmpeg, context_target, width, height, fps, crf, preset)
    history: dict[int, list[tuple[int, int] | None]] = {index: [] for index in dynamic_indices}
    context_frame_count = 0
    frame_index = 0
    try:
        while frame_index < source_frame_count:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{sample_id}: source ended at frame {frame_index}")
            rendered = overlay_frame(frame, frame_index, projected, object_names, dynamic_indices, history)
            source_writer.write(rendered)
            if frame_index < 8:
                context_writer.write(rendered)
                context_frame_count += 1
            frame_index += 1
    finally:
        cap.release()
        source_writer.close()
        context_writer.close()
    if context_frame_count != 8:
        raise RuntimeError(f"{sample_id}: expected 8 context frames, wrote {context_frame_count}")
    return {
        "sample_id": sample_id,
        "taxonomy": payload.get("taxonomy", ""),
        "source_group": payload.get("source_group", ""),
        "family_key": payload.get("family_key", ""),
        "title": payload.get("title", sample_id),
        "control": payload.get("control", {}),
        "captions": {
            "specific": payload.get("input_caption_specific", ""),
            "abstract": payload.get("input_caption_abstract", ""),
        },
        "dynamic_objects": [object_names[index] for index in dynamic_indices],
        "source_video": str(source_path),
        "context_video": str(context_path),
        "source_overlay": f"videos/{source_target.name}",
        "context8_overlay": f"videos/{context_target.name}",
        "source_frame_count": source_frame_count,
        "context_frame_count": 8,
        "width": width,
        "height": height,
        "fps": fps,
    }


def main() -> None:
    args = parse_args()
    input_list = args.input_list.resolve()
    output_root = args.output_root.resolve()
    ffmpeg = args.ffmpeg.resolve()
    if not input_list.is_file():
        raise FileNotFoundError(input_list)
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    output_root.mkdir(parents=True, exist_ok=True)
    payloads = [read_json(Path(line.strip())) for line in input_list.read_text().splitlines() if line.strip()]
    if len(payloads) != 70:
        raise RuntimeError(f"Expected 70 input JSONs, got {len(payloads)}")

    cases = []
    for index, payload in enumerate(payloads, start=1):
        result = render_case(payload, output_root, ffmpeg, args.crf, args.preset)
        cases.append(result)
        print(
            f"[{index:02d}/{len(payloads)}] {result['sample_id']} "
            f"{result['width']}x{result['height']} dynamic={len(result['dynamic_objects'])}",
            flush=True,
        )

    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(case["family_key"], []).append(case)
    manifest = {
        "schema_version": "physv_cycles_trajectory_overlay_v1",
        "source_input_list": str(input_list),
        "output_root": str(output_root),
        "case_count": len(cases),
        "group_count": len(groups),
        "overlay_policy": "Ground-truth positions are projected with each case metadata camera; dynamic_mask actors only; history is accumulated up to the current frame.",
        "projection": "Perspective projection using metadata camera eye/target/up and yfov, recomputed at each Cycles video resolution.",
        "cases": cases,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"WROTE {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
