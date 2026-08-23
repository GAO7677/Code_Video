#!/usr/bin/env python3
"""Concatenate original Cycles videos and trajectory+mask overlays side by side."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_OVERLAY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_side_by_side"
)
DEFAULT_FFMPEG = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--overlay-root", type=Path, default=DEFAULT_OVERLAY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-case")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("cases"), list):
        raise ValueError(f"Invalid overlay manifest: {path}")
    return payload


def concat_pair(
    *,
    ffmpeg: Path,
    left: Path,
    right: Path,
    target: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
    force: bool,
) -> bool:
    if not left.is_file():
        raise FileNotFoundError(left)
    if not right.is_file():
        raise FileNotFoundError(right)
    if target.is_file() and target.stat().st_size > 0 and not force:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if temporary.exists():
        temporary.unlink()
    filter_graph = (
        f"[0:v]setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos[left];"
        f"[1:v]setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos[right];"
        "[left][right]hstack=inputs=2,format=yuv420p[v]"
    )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-an",
        "-r",
        f"{fps:.6f}",
        "-shortest",
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
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    overlay_root = args.overlay_root.resolve()
    output_root = args.output_root.resolve()
    ffmpeg = args.ffmpeg.resolve()
    overlay_manifest = load_manifest(overlay_root / "manifest.json")
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)

    cases = overlay_manifest["cases"]
    if args.only_case:
        cases = [case for case in cases if case.get("sample_id") == args.only_case]
        if not cases:
            raise ValueError(f"Case not found in overlay manifest: {args.only_case}")

    output_manifest: dict[str, Any] = {
        "schema_version": "physv_side_by_side_v1",
        "dataset": "physv_v2v_0819",
        "source_dataset_root": str(dataset_root),
        "source_overlay_root": str(overlay_root),
        "output_root": str(output_root),
        "layout": {
            "left": "original RGB Cycles video",
            "right": "trajectory + dynamic-object GT mask overlay",
            "separator": "none",
        },
        "cases": [],
    }
    created = skipped = 0
    for index, case in enumerate(cases, start=1):
        sample_id = str(case["sample_id"])
        width = int(case.get("width", 896))
        height = int(case.get("height", 512))
        fps = float(case.get("fps", 30.0))
        source_video = Path(str(case["source_video"]))
        context_video = Path(str(case["context_video"]))
        context_overlay = overlay_root / str(case["context8_overlay"])
        source_overlay = overlay_root / str(case["source_overlay"])
        context_target = output_root / "videos" / f"{sample_id}__ctx8_side_by_side.mp4"
        source_target = output_root / "videos" / f"{sample_id}__source_side_by_side.mp4"
        print(f"[{index}/{len(cases)}] {sample_id}", flush=True)
        if concat_pair(
            ffmpeg=ffmpeg,
            left=context_video,
            right=context_overlay,
            target=context_target,
            width=width,
            height=height,
            fps=fps,
            crf=args.crf,
            preset=args.preset,
            force=args.force,
        ):
            created += 1
        else:
            skipped += 1
        if concat_pair(
            ffmpeg=ffmpeg,
            left=source_video,
            right=source_overlay,
            target=source_target,
            width=width,
            height=height,
            fps=fps,
            crf=args.crf,
            preset=args.preset,
            force=args.force,
        ):
            created += 1
        else:
            skipped += 1
        output_manifest["cases"].append(
            {
                "sample_id": sample_id,
                "taxonomy": case.get("taxonomy"),
                "source_group": case.get("source_group"),
                "title": case.get("title"),
                "control": case.get("control"),
                "context8": {
                    "path": str(context_target),
                    "relative_path": context_target.relative_to(output_root).as_posix(),
                    "frame_count": int(case.get("context_frame_count", 8)),
                    "width": width * 2,
                    "height": height,
                    "fps": fps,
                },
                "source": {
                    "path": str(source_target),
                    "relative_path": source_target.relative_to(output_root).as_posix(),
                    "frame_count": int(case.get("source_frame_count", 90)),
                    "width": width * 2,
                    "height": height,
                    "fps": fps,
                },
            }
        )

    output_manifest["case_count"] = len(output_manifest["cases"])
    output_manifest["video_count"] = len(output_manifest["cases"]) * 2
    output_manifest["created_count"] = created
    output_manifest["skipped_count"] = skipped
    write_json(output_root / "manifest.json", output_manifest)
    print(f"created={created} skipped={skipped}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(f"manifest={output_root / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
