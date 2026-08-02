#!/usr/bin/env python3
"""Build the 198 Physics-IQ Verified BPP V2V inputs for the xSSC runner."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


DATASET = Path("/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified")
DESCRIPTIONS = Path(
    "/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main/"
    "descriptions/best_practice/descriptions_base.csv"
)
RESULT_BASE = Path(
    "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified"
)
FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
FFPROBE = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe")
FPS = 24
CONDITION_FRAMES = 72
CONDITION_SECONDS = 3.0


def probe_video(path: Path) -> tuple[int, float, float]:
    command = [
        str(FFPROBE),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_read_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    fps = float(numerator) / float(denominator)
    return int(stream["nb_read_frames"]), fps, float(payload["format"]["duration"])


def is_valid_condition(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        frames, fps, duration = probe_video(path)
    except (KeyError, ValueError, subprocess.SubprocessError):
        return False
    return (
        frames == CONDITION_FRAMES
        and abs(fps - FPS) < 1e-6
        and abs(duration - CONDITION_SECONDS) < 0.001
    )


def convert_condition(source: Path, target: Path) -> None:
    if is_valid_condition(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.mp4")
    command = [
        str(FFMPEG),
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"fps={FPS}",
        "-frames:v",
        str(CONDITION_FRAMES),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    if not is_valid_condition(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"converted conditioning video failed validation: {source}")
    temporary.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-base", type=Path, default=RESULT_BASE)
    parser.add_argument("--limit", type=int, default=198)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_base = args.result_base.expanduser().resolve()
    input_root = result_base / "inputs" / "bpp"
    json_root = input_root / "jsons"
    condition_root = input_root / "conditioning" / f"{FPS}FPS"
    json_root.mkdir(parents=True, exist_ok=True)
    condition_root.mkdir(parents=True, exist_ok=True)

    if not FFMPEG.is_file() or not FFPROBE.is_file():
        raise FileNotFoundError("the wan-cu128 ffmpeg and ffprobe binaries are required")

    with DESCRIPTIONS.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if "_take-1_" in row["scenario"]
        ]
    rows.sort(key=lambda row: int(row["scenario"].split("_", 1)[0]))
    if len(rows) != 198:
        raise RuntimeError(f"expected 198 take-1 descriptions, found {len(rows)}")
    if not 1 <= args.limit <= 198:
        raise ValueError("--limit must be between 1 and 198")
    rows = rows[: args.limit]

    json_paths: list[Path] = []
    manifest_rows: list[dict[str, str | int | float]] = []
    for expected_id, row in enumerate(rows, start=1):
        scenario = row["scenario"]
        generated_name = row["generated_video_name"]
        file_id, scenario_rest = scenario.split("_", 1)
        if file_id != f"{expected_id:04d}":
            raise RuntimeError(
                f"non-contiguous official ID: expected {expected_id:04d}, got {file_id}"
            )
        if not generated_name.startswith(f"{file_id}_"):
            raise RuntimeError(f"generated name has the wrong ID: {generated_name}")

        source_name = (
            f"{file_id}_conditioning-videos_30FPS_{scenario_rest}"
        )
        source = DATASET / "split-videos" / "conditioning" / "30FPS" / source_name
        if not source.is_file():
            raise FileNotFoundError(f"official conditioning video not found: {source}")

        target_name = source_name.replace("_30FPS_", f"_{FPS}FPS_", 1)
        condition = condition_root / target_name
        convert_condition(source, condition)

        json_path = json_root / f"{Path(generated_name).stem}.json"
        payload = {
            "input_caption": row["description"],
            "source_video": str(condition),
            "benchmark_scenario": scenario,
            "generated_video_name": generated_name,
            "prompt_setting": "bpp",
            "input_mode": "v2v",
            "conditioning_fps": FPS,
            "conditioning_frames": CONDITION_FRAMES,
            "conditioning_duration_seconds": CONDITION_SECONDS,
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        json_paths.append(json_path.resolve())
        manifest_rows.append(
            {
                "id": expected_id,
                "scenario": scenario,
                "generated_video_name": generated_name,
                "conditioning_video": str(condition),
                "prompt_setting": "bpp",
            }
        )

    list_path = input_root / f"verified_v2v_bpp_{len(json_paths)}.txt"
    list_path.write_text(
        "".join(f"{path}\n" for path in json_paths),
        encoding="utf-8",
    )
    (input_root / "manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "Physics-IQ Verified",
                "input_mode": "v2v",
                "prompt_setting": "bpp",
                "fps": FPS,
                "num_cases": len(manifest_rows),
                "descriptions_file": str(DESCRIPTIONS),
                "source_dataset": str(DATASET),
                "cases": manifest_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Verified V2V input list: {list_path}")


if __name__ == "__main__":
    main()
