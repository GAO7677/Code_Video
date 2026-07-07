#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import random
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


FRAME_RE = re.compile(r"^(?P<scene>[^/]+)/(?P<camera>CineCamera_[^/]+)/rgb/(?P<index>\d+)\.(jpg|jpeg|png)$")
ACTIVITIES = ("SinglePhysics", "DoublePhysics", "TriplePhysics")


@dataclass(slots=True)
class CaseInfo:
    zip_path: str
    case_name: str
    activity_type: str
    num_cameras: int
    min_camera_frames: int
    max_camera_frames: int
    median_camera_frames: int
    total_frames: int
    fps: float | None
    duration_seconds: float | None
    selected_camera: str


def _parse_main_metadata(zf: zipfile.ZipFile, case_name: str) -> tuple[int | None, float | None]:
    target = next((name for name in zf.namelist() if name.endswith(f"{case_name}.json")), None)
    if target is None:
        return None, None
    try:
        payload = json.loads(zf.read(target).decode("utf-8"))
    except Exception:
        return None, None
    sequence_info = payload.get("sequence_info")
    if not isinstance(sequence_info, dict):
        return None, None
    total_frames = sequence_info.get("total_frames")
    frame_rate = sequence_info.get("frame_rate")
    total_frames = int(total_frames) if isinstance(total_frames, (int, float)) and total_frames > 0 else None
    frame_rate = float(frame_rate) if isinstance(frame_rate, (int, float)) and frame_rate > 0 else None
    return total_frames, frame_rate


def inspect_case(zip_path: Path) -> CaseInfo:
    camera_frames: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            match = FRAME_RE.match(name)
            if match is None:
                continue
            camera_frames[match.group("camera")].append(name)

        if not camera_frames:
            raise RuntimeError(f"no RGB frames found in {zip_path}")

        frame_counts = {camera: len(paths) for camera, paths in camera_frames.items()}
        static_camera_name = None
        static_camera_member = next((name for name in names if name.endswith("/static_camera_list.txt")), None)
        if static_camera_member is not None:
            try:
                content = zf.read(static_camera_member).decode("utf-8").strip().splitlines()
                for line in content:
                    line = line.strip()
                    if line in frame_counts:
                        static_camera_name = line
                        break
            except Exception:
                static_camera_name = None
        selected_camera = static_camera_name or sorted(frame_counts.keys())[0]
        total_frames, fps = _parse_main_metadata(zf, zip_path.stem)

    sorted_counts = sorted(frame_counts.values())
    if total_frames is None:
        total_frames = int(round(float(np.median(sorted_counts))))
    duration_seconds = None
    if fps is not None and fps > 0:
        duration_seconds = float(total_frames) / float(fps)

    return CaseInfo(
        zip_path=str(zip_path),
        case_name=zip_path.stem,
        activity_type=zip_path.parent.name,
        num_cameras=len(frame_counts),
        min_camera_frames=int(sorted_counts[0]),
        max_camera_frames=int(sorted_counts[-1]),
        median_camera_frames=int(round(float(np.median(sorted_counts)))),
        total_frames=int(total_frames),
        fps=fps,
        duration_seconds=duration_seconds,
        selected_camera=selected_camera,
    )


def _decode_rgb_frame(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError("failed to decode frame bytes")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _resize_if_needed(frame_rgb: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    if width <= max_width:
        return frame_rgb
    scale = float(max_width) / float(width)
    dst_w = max(1, int(round(width * scale)))
    dst_h = max(1, int(round(height * scale)))
    return cv2.resize(frame_rgb, (dst_w, dst_h), interpolation=cv2.INTER_AREA)


def _write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _ensure_browser_video(source_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and imageio_ffmpeg is not None:
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        return source_path
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if (
        out_path.exists()
        and out_path.stat().st_size > 0
        and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    ):
        return out_path
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def _build_storyboard(frames: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not frames:
        raise ValueError("frames must be non-empty")
    height, width = frames[0].shape[:2]
    rows = int(np.ceil(len(frames) / max(columns, 1)))
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        row = idx // columns
        col = idx % columns
        y0 = row * height
        x0 = col * width
        canvas[y0 : y0 + height, x0 : x0 + width] = frame
    return canvas


def _write_png(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def render_case_preview(
    case: CaseInfo,
    *,
    output_dir: Path,
    preview_fps: int,
    max_width: int,
    max_frames: int,
) -> dict[str, Any]:
    case_dir = output_dir / case.activity_type / case.case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(case.zip_path) as zf:
        frame_members = [
            name
            for name in zf.namelist()
            if FRAME_RE.match(name) and FRAME_RE.match(name).group("camera") == case.selected_camera
        ]
        frame_members = sorted(frame_members, key=lambda name: int(FRAME_RE.match(name).group("index")))
        if not frame_members:
            raise RuntimeError(f"no frames for camera {case.selected_camera} in {case.zip_path}")

        if len(frame_members) > max_frames:
            indices = np.linspace(0, len(frame_members) - 1, max_frames).round().astype(np.int64)
            selected_members = [frame_members[int(index)] for index in indices]
        else:
            selected_members = frame_members

        frames = []
        for member in selected_members:
            frames.append(_resize_if_needed(_decode_rgb_frame(zf.read(member)), max_width=max_width))

    effective_fps = int(preview_fps) if int(preview_fps) > 0 else int(round(case.fps)) if case.fps else 30
    video_raw = case_dir / "preview.mp4"
    _write_mp4(video_raw, np.stack(frames, axis=0), fps=effective_fps)
    video_browser = _ensure_browser_video(video_raw)

    sample_ids = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(np.int64)
    storyboard = _build_storyboard([frames[int(index)] for index in sample_ids], columns=4)
    storyboard_path = case_dir / "storyboard.png"
    _write_png(storyboard_path, storyboard)

    return {
        "activity_type": case.activity_type,
        "case_name": case.case_name,
        "zip_path": case.zip_path,
        "selected_camera": case.selected_camera,
        "num_cameras": case.num_cameras,
        "total_frames": case.total_frames,
        "fps": case.fps,
        "preview_fps": effective_fps,
        "duration_seconds": case.duration_seconds,
        "preview_video": str(video_browser.relative_to(output_dir)),
        "storyboard_png": str(storyboard_path.relative_to(output_dir)),
    }


def _format_duration(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _histogram_bins(values: list[int]) -> list[tuple[str, int]]:
    bins = [
        ("<= 60", 0),
        ("61-90", 0),
        ("91-120", 0),
        ("121-150", 0),
        ("151-210", 0),
        ("> 210", 0),
    ]
    for value in values:
        if value <= 60:
            idx = 0
        elif value <= 90:
            idx = 1
        elif value <= 120:
            idx = 2
        elif value <= 150:
            idx = 3
        elif value <= 210:
            idx = 4
        else:
            idx = 5
        label, count = bins[idx]
        bins[idx] = (label, count + 1)
    return bins


def build_summary(cases: list[CaseInfo]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_cases": len(cases),
        "activities": {},
    }
    for activity in ACTIVITIES:
        activity_cases = [case for case in cases if case.activity_type == activity]
        frames = [case.total_frames for case in activity_cases]
        durations = [case.duration_seconds for case in activity_cases if case.duration_seconds is not None]
        if not activity_cases:
            summary["activities"][activity] = {
                "count": 0,
                "frame_histogram": _histogram_bins([]),
            }
            continue
        summary["activities"][activity] = {
            "count": len(activity_cases),
            "min_frames": min(frames),
            "max_frames": max(frames),
            "median_frames": float(np.median(frames)),
            "mean_frames": float(np.mean(frames)),
            "frame_histogram": _histogram_bins(frames),
            "min_duration_seconds": min(durations) if durations else None,
            "max_duration_seconds": max(durations) if durations else None,
            "median_duration_seconds": float(np.median(durations)) if durations else None,
        }
    return summary


def _bar_width(value: int, max_value: int) -> float:
    if max_value <= 0:
        return 0.0
    return (100.0 * float(value) / float(max_value))


def build_html(
    *,
    summary: dict[str, Any],
    rendered_cases: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    max_hist = 1
    for activity_payload in summary["activities"].values():
        for _, count in activity_payload.get("frame_histogram", []):
            max_hist = max(max_hist, int(count))

    sections = []
    grouped_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in rendered_cases:
        grouped_cases[case["activity_type"]].append(case)

    for activity in ACTIVITIES:
        activity_payload = summary["activities"].get(activity, {})
        hist_rows = []
        for label, count in activity_payload.get("frame_histogram", []):
            width = _bar_width(int(count), max_hist)
            hist_rows.append(
                f"""
<div class="hist-row">
  <div class="hist-label">{html.escape(label)}</div>
  <div class="hist-bar-wrap"><div class="hist-bar" style="width:{width:.2f}%"></div></div>
  <div class="hist-count">{int(count)}</div>
</div>
"""
            )

        cards = []
        for case in grouped_cases.get(activity, []):
            cards.append(
                f"""
<article class="case-card">
  <h3>{html.escape(case['case_name'])}</h3>
  <p class="meta"><b>camera:</b> {html.escape(case['selected_camera'])}</p>
  <p class="meta"><b>frames:</b> {int(case['total_frames'])} &nbsp; <b>source fps:</b> {case['fps'] if case['fps'] is not None else 'n/a'} &nbsp; <b>preview fps:</b> {case['preview_fps']} &nbsp; <b>duration:</b> {_format_duration(case['duration_seconds'])}</p>
  <p class="meta path">{html.escape(case['zip_path'])}</p>
  <video controls preload="none" playsinline src="{html.escape(case['preview_video'])}"></video>
  <img src="{html.escape(case['storyboard_png'])}" />
</article>
"""
            )

        sections.append(
            f"""
<section class="activity-section">
  <div class="activity-head">
    <div>
      <h2>{html.escape(activity)}</h2>
      <p class="summary-line">
        count={activity_payload.get('count', 0)},
        min={activity_payload.get('min_frames', 'n/a')} frames,
        median={activity_payload.get('median_frames', 'n/a')} frames,
        max={activity_payload.get('max_frames', 'n/a')} frames
      </p>
      <p class="summary-line">
        duration min/median/max = {_format_duration(activity_payload.get('min_duration_seconds'))} /
        {_format_duration(activity_payload.get('median_duration_seconds'))} /
        {_format_duration(activity_payload.get('max_duration_seconds'))}
      </p>
    </div>
  </div>
  <div class="histogram">
    {''.join(hist_rows)}
  </div>
  <div class="case-grid">
    {''.join(cards)}
  </div>
</section>
"""
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PhysInOne Length Distribution + Case Gallery</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffdf8;
      --line: #d8cfbf;
      --text: #1e1d1a;
      --muted: #5f584f;
      --accent: #0f5b78;
      --accent-soft: #d9edf4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      color: var(--text);
      font-family: sans-serif;
      background:
        radial-gradient(circle at top right, #efe0c7 0, transparent 22%),
        linear-gradient(180deg, #f7f2e9 0%, #f2ede4 100%);
    }}
    .page {{ max-width: 1800px; margin: 0 auto; }}
    h1 {{ margin: 0 0 10px; font-size: 32px; }}
    .intro {{ color: var(--muted); margin: 0 0 24px; line-height: 1.6; }}
    .activity-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.04);
    }}
    .activity-head h2 {{ margin: 0 0 8px; font-size: 24px; }}
    .summary-line {{ margin: 4px 0; color: var(--muted); }}
    .histogram {{
      margin: 16px 0 20px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fbf8f2;
    }}
    .hist-row {{
      display: grid;
      grid-template-columns: 90px 1fr 40px;
      gap: 12px;
      align-items: center;
      margin: 8px 0;
    }}
    .hist-label, .hist-count {{ font-size: 13px; color: var(--muted); }}
    .hist-bar-wrap {{
      height: 14px;
      border-radius: 999px;
      background: #ece6db;
      overflow: hidden;
    }}
    .hist-bar {{
      height: 100%;
      background: linear-gradient(90deg, var(--accent) 0%, #4e8ca1 100%);
      border-radius: 999px;
    }}
    .case-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }}
    .case-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #ffffff;
    }}
    .case-card h3 {{ margin: 0; padding: 14px 14px 8px; font-size: 18px; line-height: 1.35; }}
    .case-card .meta {{ margin: 0; padding: 0 14px 8px; color: var(--muted); font-size: 13px; word-break: break-word; }}
    .case-card .path {{ padding-bottom: 12px; }}
    .case-card video, .case-card img {{ display: block; width: 100%; background: #000; border-top: 1px solid var(--line); }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PhysInOne Length Distribution + Case Gallery</h1>
    <p class="intro">
      This page is built from the PhysInOne zip files currently available on disk.
      Length is reported per case as single-camera total frame count; when metadata exposes
      <code>sequence_info.frame_rate</code>, duration in seconds is shown as well.
      Total visible cases in current local cache: {summary['num_cases']}.
    </p>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def discover_zip_files(roots: list[Path]) -> list[Path]:
    discovered: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.zip"):
            discovered[path.stem] = path
    return sorted(discovered.values(), key=lambda path: (path.parent.name, path.stem))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze current local PhysInOne length distribution and build a lightweight gallery."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            Path("/data/gaoya/dataset/vLAR-PhysInOne/PhysInOneP01-PhysInOneP01/Train"),
            Path("/data/gaoya/dataset/vLAR-PhysInOne/TrainBalanced100G/Train"),
        ],
        help="Train roots that contain PhysInOne *.zip files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/phisinone_length_gallery"),
        help="Output directory for summary JSON, mp4 previews, and HTML gallery.",
    )
    parser.add_argument(
        "--cases-per-activity",
        type=int,
        default=3,
        help="How many preview cases to render per activity type.",
    )
    parser.add_argument(
        "--preview-fps",
        type=int,
        default=0,
        help="FPS used for the preview mp4s. Use 0 to follow per-case source fps; fallback is 30.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=896,
        help="Resize previews down to this width if frames are larger.",
    )
    parser.add_argument(
        "--max-preview-frames",
        type=int,
        default=96,
        help="Cap preview video length by uniformly sampling up to this many frames.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for case sampling within each activity type.",
    )
    args = parser.parse_args()

    zip_files = discover_zip_files([path.expanduser().resolve() for path in args.roots])
    if not zip_files:
        raise RuntimeError("no PhysInOne zip files found under the provided roots")

    cases = [inspect_case(path) for path in zip_files]
    summary = build_summary(cases)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "cases": [asdict(case) for case in cases],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rng = random.Random(int(args.seed))
    rendered_cases: list[dict[str, Any]] = []
    by_activity: dict[str, list[CaseInfo]] = defaultdict(list)
    for case in cases:
        by_activity[case.activity_type].append(case)

    for activity in ACTIVITIES:
        pool = list(by_activity.get(activity, []))
        if not pool:
            continue
        pool.sort(key=lambda case: (case.total_frames, case.case_name))
        rng.shuffle(pool)
        selected = sorted(pool[: max(1, int(args.cases_per_activity))], key=lambda case: (case.total_frames, case.case_name))
        for case in selected:
            rendered_cases.append(
                render_case_preview(
                    case,
                    output_dir=output_dir,
                    preview_fps=int(args.preview_fps),
                    max_width=int(args.max_width),
                    max_frames=int(args.max_preview_frames),
                )
            )

    html_text = build_html(summary=summary, rendered_cases=rendered_cases, output_dir=output_dir)
    html_path = output_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_html": str(html_path),
                "summary_json": str(output_dir / "summary.json"),
                "num_cases_scanned": len(cases),
                "num_preview_cases": len(rendered_cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
