#!/usr/bin/env python3
"""Overlay xSSC attention centroids and their frame-to-frame displacement."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


DEFAULT_INPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_slot_separation_cases_dinov3_latest/"
    "models/dinov3_vitl_movic_transfer_movi_c_transfer15000_b64_acc3_20260721T134713Z_step-050000"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_attention_centroid_overlay_step050000"
)

COLORS_BGR = [
    (52, 73, 239),
    (34, 197, 94),
    (234, 88, 12),
    (168, 85, 247),
    (6, 182, 212),
    (236, 72, 153),
    (250, 204, 21),
    (20, 184, 166),
    (248, 113, 113),
    (96, 165, 250),
    (163, 163, 163),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trail-length", type=int, default=12)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--slots",
        choices=("selected", "active", "all"),
        default="selected",
        help="selected uses the existing object-slot selection in summary.json.",
    )
    parser.add_argument("--min-hard-area", type=float, default=0.005)
    return parser.parse_args()


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    return frames, fps


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def attention_statistics(
    attention: np.ndarray,
    frame_height: int,
    frame_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return soft centroids in pixels, attention mass, and hard-label area."""
    _, _, grid_h, grid_w = attention.shape
    yy, xx = np.meshgrid(
        (np.arange(grid_h, dtype=np.float32) + 0.5) / grid_h,
        (np.arange(grid_w, dtype=np.float32) + 0.5) / grid_w,
        indexing="ij",
    )
    mass = attention.sum(axis=(2, 3))
    denominator = np.maximum(mass, 1.0e-8)
    cx = (attention * xx[None, None]).sum(axis=(2, 3)) / denominator
    cy = (attention * yy[None, None]).sum(axis=(2, 3)) / denominator
    centroids = np.stack([cx * frame_width, cy * frame_height], axis=-1)

    hard_labels = attention.argmax(axis=1)
    hard_area = np.stack(
        [(hard_labels == slot).mean(axis=(1, 2)) for slot in range(attention.shape[1])],
        axis=1,
    )
    return centroids.astype(np.float32), mass.astype(np.float32), hard_area.astype(np.float32)


def choose_slots(summary: dict[str, Any], mode: str, num_slots: int) -> list[int]:
    if mode == "all":
        return list(range(num_slots))
    if mode == "selected":
        selected = [int(slot) for slot in summary.get("selected_slots", [])]
        if selected:
            return selected
    active = [
        int(row["slot"])
        for row in summary.get("slot_summary", [])
        if float(row.get("active_frames", 0.0)) >= 0.35
        and float(row.get("mean_area", 0.0)) >= 0.006
    ]
    return active or list(range(min(num_slots, 4)))


def draw_label(frame: np.ndarray, text: str, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = point
    label_y = max(14, y - 8)
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    label_x = x + 7
    if label_x + text_width >= frame.shape[1] - 2:
        label_x = max(2, x - text_width - 7)
    origin = (label_x, label_y)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def render_overlay(
    frames: list[np.ndarray],
    centroids: np.ndarray,
    hard_area: np.ndarray,
    selected_slots: list[int],
    trail_length: int,
    min_hard_area: float,
    normalized_speed: bool = False,
) -> list[np.ndarray]:
    output = []
    for time_index, source in enumerate(frames):
        frame = source.copy()
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 23), (0, 0, 0), -1)
        cv2.putText(
            frame,
            (
                f"frame {time_index:02d} | trail {trail_length}f | speed %diag/frame"
                if normalized_speed
                else f"frame {time_index:02d} | trail {trail_length}f | speed px/frame"
            ),
            (7, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for slot in selected_slots:
            color = COLORS_BGR[slot % len(COLORS_BGR)]
            start = max(0, time_index - trail_length + 1)
            valid_times = [
                t for t in range(start, time_index + 1) if hard_area[t, slot] >= min_hard_area
            ]
            for left, right in zip(valid_times[:-1], valid_times[1:]):
                if right != left + 1:
                    continue
                p0 = tuple(np.rint(centroids[left, slot]).astype(int))
                p1 = tuple(np.rint(centroids[right, slot]).astype(int))
                cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
            if hard_area[time_index, slot] < min_hard_area:
                continue
            point = tuple(np.rint(centroids[time_index, slot]).astype(int))
            if time_index > 0 and hard_area[time_index - 1, slot] >= min_hard_area:
                previous = tuple(np.rint(centroids[time_index - 1, slot]).astype(int))
                cv2.arrowedLine(frame, previous, point, color, 2, cv2.LINE_AA, tipLength=0.35)
            cv2.circle(frame, point, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
            speed = 0.0 if time_index == 0 else float(
                np.linalg.norm(centroids[time_index, slot] - centroids[time_index - 1, slot])
            )
            if normalized_speed:
                speed = 100.0 * speed / np.hypot(frame.shape[1], frame.shape[0])
                speed_label = f"S{slot} {speed:.2f}%d/f"
            else:
                speed_label = f"S{slot} {speed:.1f}px/f"
            draw_label(frame, speed_label, point, color)
        output.append(frame)
    return output


def write_case_csv(
    path: Path,
    centroids: np.ndarray,
    mass: np.ndarray,
    hard_area: np.ndarray,
    selected_slots: list[int],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame", "slot", "centroid_x_px", "centroid_y_px", "dx_px", "dy_px",
                "speed_px_per_frame", "attention_mass", "hard_area_fraction",
            ),
        )
        writer.writeheader()
        for time_index in range(centroids.shape[0]):
            for slot in selected_slots:
                delta = np.zeros(2, dtype=np.float32) if time_index == 0 else (
                    centroids[time_index, slot] - centroids[time_index - 1, slot]
                )
                writer.writerow(
                    {
                        "frame": time_index,
                        "slot": slot,
                        "centroid_x_px": float(centroids[time_index, slot, 0]),
                        "centroid_y_px": float(centroids[time_index, slot, 1]),
                        "dx_px": float(delta[0]),
                        "dy_px": float(delta[1]),
                        "speed_px_per_frame": float(np.linalg.norm(delta)),
                        "attention_mass": float(mass[time_index, slot]),
                        "hard_area_fraction": float(hard_area[time_index, slot]),
                    }
                )


def build_html(output_dir: Path, records: list[dict[str, Any]]) -> None:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC attention 质心位移</title>
<style>
body{{margin:0;background:#f5f7fa;color:#17202a;font:14px Arial,sans-serif}}header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #dce1e8;padding:12px 18px}}h1{{font-size:19px;margin:0 0 8px}}.controls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}select,button{{height:32px;border:1px solid #aeb7c2;background:#fff;padding:0 10px;border-radius:4px}}main{{max-width:1120px;margin:18px auto;padding:0 16px}}.note{{line-height:1.55;color:#4b5563;margin-bottom:12px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0;background:#fff;border:1px solid #dce1e8;padding:8px;border-radius:6px}}video{{display:block;width:100%;background:#111;aspect-ratio:1/1}}figcaption{{padding-top:7px;font-weight:600}}#details{{margin-top:10px;background:#fff;border:1px solid #dce1e8;padding:10px;border-radius:6px}}@media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>DINOv3 MOVi-C xSSC step-050000 · attention 质心位移</h1><div class="controls"><select id="case"></select><button id="play">播放</button><button id="pause">暂停</button><button id="replay">重新播放</button></div></header>
<main><p class="note">圆点是当前帧的 soft-attention 质心；箭头严格表示上一帧到当前帧的一帧位移；细线保留最近的轨迹。速度单位为 256×256 xSSC 输入上的 px/frame。这里只展示已有分析选出的活动 object slots。</p><div class="grid"><figure><video id="source" muted playsinline preload="metadata"></video><figcaption>原始 xSSC 输入</figcaption></figure><figure><video id="overlay" muted playsinline preload="metadata"></video><figcaption>质心、位移箭头与历史轨迹</figcaption></figure></div><div id="details"></div></main>
<script>const DATA={payload};const pick=document.getElementById('case');const source=document.getElementById('source');const overlay=document.getElementById('overlay');const details=document.getElementById('details');for(const r of DATA){{const o=document.createElement('option');o.value=r.case_id;o.textContent=r.case_id;pick.appendChild(o)}}function current(){{return DATA.find(r=>r.case_id===pick.value)}}function load(){{const r=current();source.src=r.source_video;overlay.src=r.overlay_video;details.textContent=`slots: ${{r.selected_slots.map(x=>'S'+x).join(', ')}} | frames: ${{r.frames}} | CSV: ${{r.csv}}`;}}function both(fn){{[source,overlay].forEach(fn)}}pick.onchange=load;document.getElementById('play').onclick=()=>both(v=>v.play());document.getElementById('pause').onclick=()=>both(v=>v.pause());document.getElementById('replay').onclick=()=>both(v=>{{v.currentTime=0;v.play()}});load();</script></body></html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for arrays_path in sorted(input_root.glob("*/xssc/slot_separation_arrays.npz")):
        case_dir = arrays_path.parent.parent
        case_id = case_dir.name
        summary_path = arrays_path.parent / "summary.json"
        source_path = case_dir / "xssc_input_49f.mp4"
        if not summary_path.is_file() or not source_path.is_file():
            print(f"[skip] incomplete cache: {case_id}", flush=True)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(arrays_path) as data:
            attention = data["attention"].astype(np.float32)
        frames, source_fps = read_video(source_path)
        length = min(len(frames), attention.shape[0])
        frames = frames[:length]
        attention = attention[:length]
        height, width = frames[0].shape[:2]
        centroids, mass, hard_area = attention_statistics(attention, height, width)
        selected_slots = choose_slots(summary, args.slots, attention.shape[1])
        rendered = render_overlay(
            frames, centroids, hard_area, selected_slots, args.trail_length, args.min_hard_area
        )
        case_output = output_dir / "cases" / case_id
        overlay_path = case_output / "attention_centroid_overlay.mp4"
        csv_path = case_output / "attention_centroids.csv"
        write_video(overlay_path, rendered, args.fps or source_fps)
        write_case_csv(csv_path, centroids, mass, hard_area, selected_slots)
        source_link = case_output / "source.mp4"
        if source_link.exists() or source_link.is_symlink():
            source_link.unlink()
        source_link.symlink_to(source_path)
        record = {
            "case_id": case_id,
            "frames": length,
            "selected_slots": selected_slots,
            "source_video": str(source_link.relative_to(output_dir)),
            "overlay_video": str(overlay_path.relative_to(output_dir)),
            "csv": str(csv_path.relative_to(output_dir)),
        }
        records.append(record)
        print(f"[done] {case_id}: slots={selected_slots} frames={length}", flush=True)
    if not records:
        raise RuntimeError(f"No complete cached cases found under {input_root}")
    metadata = {
        "input_root": str(input_root),
        "centroid_definition": "per-slot spatial attention weighted patch-center mean",
        "coordinate_system": "xSSC 256x256 input pixels",
        "trail_length": args.trail_length,
        "min_hard_area": args.min_hard_area,
        "records": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_html(output_dir, records)
    print(f"[complete] {len(records)} cases -> {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
