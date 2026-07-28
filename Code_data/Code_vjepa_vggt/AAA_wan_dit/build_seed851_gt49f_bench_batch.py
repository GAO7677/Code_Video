#!/usr/bin/env python3
"""Stage seed-851 test5 GT videos as 49-frame, 30 FPS, 896x512 cases."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_seed851_baseline_bench"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_gt49f_896x512_bench"
)
FFMPEG = Path("/data/gaoya/home_miniconda3/envs/wan-cu128/bin/ffmpeg")
FFPROBE = Path("/data/gaoya/home_miniconda3/envs/wan-cu128/bin/ffprobe")
TARGET_FRAMES = 49
TARGET_FPS = 30
TARGET_WIDTH = 896
TARGET_HEIGHT = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def probe_video(path: Path, count_frames: bool = True) -> dict[str, Any]:
    entries = "stream=avg_frame_rate,width,height,duration"
    command = [
        str(FFPROBE),
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
        entries += ",nb_read_frames"
    command.extend(
        [
            "-show_entries",
            entries,
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}")
    stream = streams[0]
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration": float(stream.get("duration") or 0.0),
        "frames": (
            int(stream["nb_read_frames"])
            if stream.get("nb_read_frames") not in (None, "N/A")
            else None
        ),
    }


def normalize_video(source: Path, target: Path) -> dict[str, Any]:
    source_info = probe_video(source)
    estimated_resampled_frames = max(
        1,
        int(math.ceil(source_info["duration"] * TARGET_FPS - 1e-6)),
    )
    padded_frames = max(0, TARGET_FRAMES - estimated_resampled_frames)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.stem}.tmp.{os.getpid()}.mp4")
        filter_chain = (
            f"fps={TARGET_FPS},"
            "tpad=stop_mode=clone:stop_duration=2,"
            f"trim=end_frame={TARGET_FRAMES},"
            f"setpts=N/({TARGET_FPS}*TB),"
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:"
            "force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}"
        )
        subprocess.run(
            [
                str(FFMPEG),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vf",
                filter_chain,
                "-frames:v",
                str(TARGET_FRAMES),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(TARGET_FPS),
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        os.replace(temporary, target)
    output_info = probe_video(target)
    if (
        output_info["frames"] != TARGET_FRAMES
        or not math.isclose(output_info["fps"], TARGET_FPS, abs_tol=1e-6)
        or output_info["width"] != TARGET_WIDTH
        or output_info["height"] != TARGET_HEIGHT
    ):
        raise RuntimeError(
            f"Invalid normalized GT {target}: "
            f"frames={output_info['frames']} fps={output_info['fps']} "
            f"size={output_info['width']}x{output_info['height']}"
        )
    return {
        "source": str(source),
        "source_info": source_info,
        "target_frames": TARGET_FRAMES,
        "target_fps": TARGET_FPS,
        "estimated_resampled_frames_before_padding": (
            estimated_resampled_frames
        ),
        "padded_frames": padded_frames,
        "padding_mode": "clone_last_frame" if padded_frames else "none",
        "spatial_transform": (
            f"scale_cover_then_center_crop_{TARGET_WIDTH}x{TARGET_HEIGHT}"
        ),
        "output_info": output_info,
    }


def source_cases(baseline_root: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    cases_root = baseline_root / "cases"
    records = []
    for path in sorted(cases_root.glob("wan_lora__seed-000851__baseline__*.json")):
        baseline = json.loads(path.read_text(encoding="utf-8"))
        source_json = Path(str(baseline["input_json"])).expanduser().resolve()
        source_payload = json.loads(source_json.read_text(encoding="utf-8"))
        source_video = Path(
            str(source_payload["source_video"])
        ).expanduser().resolve()
        case_id = path.stem.split("__baseline__", 1)[1]
        records.append((case_id, source_json, source_payload))
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
    if len(records) != 20:
        raise RuntimeError(f"Expected 20 unique GT cases, found {len(records)}")
    return records


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for case_id, source_json, source_payload in source_cases(baseline_root):
        source_video = Path(
            str(source_payload["source_video"])
        ).expanduser().resolve()
        entry_id = f"gt__seed-000851__gt49f_896x512__{case_id}"
        target_video = cases_root / f"{entry_id}.mp4"
        target_json = target_video.with_suffix(".json")
        preprocess = normalize_video(source_video, target_video)
        metadata = {
            "entry_id": entry_id,
            "model": "gt",
            "seed": 851,
            "variant": "gt49f_896x512",
            "role": "gt",
            "denoise_step_range": None,
            "source_json": str(source_json),
            "source_video": str(source_video),
            "normalized_video": str(target_video),
        }
        if target_json.is_file():
            current = json.loads(target_json.read_text(encoding="utf-8"))
            if current.get("_stc_bench") != metadata:
                raise RuntimeError(f"Existing GT case changed: {target_json}")
        else:
            payload = dict(source_payload)
            payload["input_json"] = str(source_json)
            payload["source_video"] = str(source_video)
            payload["output_video"] = str(target_video)
            payload["_gt_preprocess"] = preprocess
            payload["_stc_bench"] = metadata
            atomic_json(target_json, payload)
        manifest_entries.append(
            {
                **metadata,
                "preprocess": preprocess,
            }
        )
    atomic_json(
        output_root / "batch_manifest.json",
        {
            "schema_version": 1,
            "num_entries": len(manifest_entries),
            "target_frames": TARGET_FRAMES,
            "target_fps": TARGET_FPS,
            "target_width": TARGET_WIDTH,
            "target_height": TARGET_HEIGHT,
            "entries": manifest_entries,
        },
    )
    (output_root / "result_roots.txt").write_text(
        str(output_root) + "\n",
        encoding="utf-8",
    )
    print(
        "[seed851-gt49f-896x512-batch] "
        f"entries=20 root={output_root}"
    )


if __name__ == "__main__":
    main()
