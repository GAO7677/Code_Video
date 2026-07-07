#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


@dataclass(slots=True)
class RenderPlan:
    zip_path: Path | None
    case_dir: Path
    camera_name: str
    fps: int
    total_frames: int | None
    output_mp4: Path


def _resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_case(zip_path: Path, *, overwrite_extract: bool) -> Path:
    case_dir = zip_path.with_suffix("")
    if case_dir.exists() and not overwrite_extract:
        return case_dir
    if overwrite_extract and case_dir.exists():
        shutil.rmtree(case_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    return case_dir


def _select_camera(case_dir: Path) -> str:
    static_list = case_dir / "static_camera_list.txt"
    if static_list.is_file():
        lines = [line.strip() for line in static_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return lines[0]
    cameras = sorted(path.name for path in case_dir.iterdir() if path.is_dir() and path.name.startswith("CineCamera_"))
    if not cameras:
        raise RuntimeError(f"no camera folders found under {case_dir}")
    return cameras[0]


def _discover_cameras(case_dir: Path) -> list[str]:
    static_list = case_dir / "static_camera_list.txt"
    cameras: list[str] = []
    if static_list.is_file():
        cameras.extend(
            [
                line.strip()
                for line in static_list.read_text(encoding="utf-8").splitlines()
                if line.strip() and (case_dir / line.strip() / "rgb").is_dir()
            ]
        )

    discovered = sorted(
        path.name
        for path in case_dir.iterdir()
        if path.is_dir() and path.name.startswith("CineCamera_") and (path / "rgb").is_dir()
    )
    seen = set()
    ordered = []
    for camera_name in [*cameras, *discovered]:
        if camera_name not in seen:
            seen.add(camera_name)
            ordered.append(camera_name)
    if not ordered:
        raise RuntimeError(f"no camera folders found under {case_dir}")
    return ordered


def _resolve_fps(case_dir: Path, camera_name: str) -> tuple[int, int | None]:
    main_json = _load_json_if_exists(case_dir / f"{case_dir.name}.json")
    camera_json = _load_json_if_exists(case_dir / f"blender_{camera_name}.json")
    if isinstance(camera_json, dict):
        fps = camera_json.get("fps") or camera_json.get("frame_rate")
        total_frames = camera_json.get("total_frames")
        if isinstance(fps, (int, float)) and fps > 0:
            total_frames_int = int(total_frames) if isinstance(total_frames, (int, float)) and total_frames > 0 else None
            return int(round(float(fps))), total_frames_int

    if isinstance(main_json, dict):
        sequence_info = main_json.get("sequence_info")
        if isinstance(sequence_info, dict):
            frame_rate = sequence_info.get("frame_rate")
            total_frames = sequence_info.get("total_frames") or sequence_info.get("num_frames")
            if isinstance(frame_rate, (int, float)) and frame_rate > 0:
                total_frames_int = int(total_frames) if isinstance(total_frames, (int, float)) and total_frames > 0 else None
                return int(round(float(frame_rate))), total_frames_int

    return 30, None


def _build_plan(case_dir: Path, camera_name: str, *, zip_path: Path | None = None) -> RenderPlan:
    fps, total_frames = _resolve_fps(case_dir, camera_name)
    output_mp4 = case_dir / camera_name / f"{camera_name}.mp4"
    return RenderPlan(
        zip_path=zip_path,
        case_dir=case_dir,
        camera_name=camera_name,
        fps=int(fps),
        total_frames=total_frames,
        output_mp4=output_mp4,
    )


def _build_plans_from_case_dir(case_dir: Path, *, all_cameras: bool) -> list[RenderPlan]:
    camera_names = _discover_cameras(case_dir) if all_cameras else [_select_camera(case_dir)]
    return [_build_plan(case_dir, camera_name) for camera_name in camera_names]


def _build_plans_from_zip(zip_path: Path, *, overwrite_extract: bool, all_cameras: bool) -> list[RenderPlan]:
    case_dir = _extract_case(zip_path, overwrite_extract=overwrite_extract)
    camera_names = _discover_cameras(case_dir) if all_cameras else [_select_camera(case_dir)]
    return [_build_plan(case_dir, camera_name, zip_path=zip_path) for camera_name in camera_names]


def _frame_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):09d}")
    except ValueError:
        return (1, path.stem)


def _detect_numeric_sequence(frame_paths: list[Path]) -> tuple[Path, int, str] | None:
    suffixes = {path.suffix.lower() for path in frame_paths}
    if len(suffixes) != 1:
        return None
    try:
        indices = [int(path.stem) for path in frame_paths]
    except ValueError:
        return None
    stem_widths = {len(path.stem) for path in frame_paths}
    if len(stem_widths) != 1:
        return None
    start = indices[0]
    expected = list(range(start, start + len(indices)))
    if indices != expected:
        return None
    width = stem_widths.pop()
    pattern = f"%0{width}d{frame_paths[0].suffix.lower()}"
    return frame_paths[0].parent / pattern, start, frame_paths[0].suffix.lower()


def _encode_h264_from_frames(frame_paths: list[Path], output_mp4: Path, fps: int) -> None:
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode mp4, but no ffmpeg binary was found")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    sequence_info = _detect_numeric_sequence(frame_paths)
    if sequence_info is not None:
        input_pattern, start_number, _ = sequence_info
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(int(fps)),
                "-start_number",
                str(start_number),
                "-i",
                str(input_pattern),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-r",
                str(int(fps)),
                str(output_mp4),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False, encoding="utf-8") as handle:
        concat_path = Path(handle.name)
        handle.write("ffconcat version 1.0\n")
        frame_duration = 1.0 / max(int(fps), 1)
        for frame_path in frame_paths:
            handle.write(f"file '{frame_path.as_posix()}'\n")
            handle.write(f"duration {frame_duration:.12f}\n")
        handle.write(f"file '{frame_paths[-1].as_posix()}'\n")

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-r",
                str(int(fps)),
                str(output_mp4),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        concat_path.unlink(missing_ok=True)


def render_plan(plan: RenderPlan, *, overwrite_video: bool) -> None:
    if plan.output_mp4.exists() and not overwrite_video:
        return
    camera_rgb_dir = plan.case_dir / plan.camera_name / "rgb"
    if not camera_rgb_dir.is_dir():
        raise RuntimeError(f"missing RGB directory: {camera_rgb_dir}")
    frame_paths = sorted(
        [
            path
            for path in camera_rgb_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ],
        key=_frame_sort_key,
    )
    if not frame_paths:
        raise RuntimeError(f"no RGB frames found under {camera_rgb_dir}")
    _encode_h264_from_frames(frame_paths, plan.output_mp4, fps=plan.fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract PhysInOne TrainBalanced100G case zips and render one H.264 mp4 "
            "per case using the official fps from metadata when available."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/gaoya/dataset/vLAR-PhysInOne/TrainBalanced100G/Train"),
        help="Root containing activity folders and *.zip cases.",
    )
    parser.add_argument(
        "--case-zip",
        type=Path,
        default=None,
        help="Optional explicit case zip path. If omitted, process zips under --root.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Optional explicit extracted case directory. If set, render videos from this directory directly.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of cases to process.",
    )
    parser.add_argument(
        "--overwrite-extract",
        action="store_true",
        help="Delete and re-extract existing case folders.",
    )
    parser.add_argument(
        "--overwrite-video",
        action="store_true",
        help="Rebuild mp4 even if it already exists.",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Render one mp4 inside every CineCamera_* folder instead of only the primary camera.",
    )
    args = parser.parse_args()

    if args.case_zip is not None and args.case_dir is not None:
        raise RuntimeError("--case-zip and --case-dir are mutually exclusive")

    if args.case_dir is not None:
        case_dirs = [args.case_dir.expanduser().resolve()]
        zip_paths: list[Path] = []
    elif args.case_zip is not None:
        zip_paths = [args.case_zip.expanduser().resolve()]
        case_dirs = []
    else:
        case_dirs = []
        zip_paths = sorted(args.root.expanduser().resolve().rglob("*.zip"))

    if args.limit is not None and zip_paths:
        zip_paths = zip_paths[: max(0, int(args.limit))]

    if not zip_paths and not case_dirs:
        raise RuntimeError("no cases found to process")

    results = []
    for case_dir in case_dirs:
        for plan in _build_plans_from_case_dir(case_dir, all_cameras=bool(args.all_cameras)):
            render_plan(plan, overwrite_video=bool(args.overwrite_video))
            results.append(
                {
                    "zip_path": None,
                    "case_dir": str(plan.case_dir),
                    "camera_name": plan.camera_name,
                    "fps": plan.fps,
                    "total_frames": plan.total_frames,
                    "output_mp4": str(plan.output_mp4),
                }
            )
    for zip_path in zip_paths:
        for plan in _build_plans_from_zip(
            zip_path,
            overwrite_extract=bool(args.overwrite_extract),
            all_cameras=bool(args.all_cameras),
        ):
            render_plan(plan, overwrite_video=bool(args.overwrite_video))
            results.append(
                {
                    "zip_path": str(plan.zip_path) if plan.zip_path is not None else None,
                    "case_dir": str(plan.case_dir),
                    "camera_name": plan.camera_name,
                    "fps": plan.fps,
                    "total_frames": plan.total_frames,
                    "output_mp4": str(plan.output_mp4),
                }
            )

    print(json.dumps({"num_processed": len(results), "cases": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
