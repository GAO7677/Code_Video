#!/usr/bin/env python3
"""Create a path-mapped copy of the shared P0 inputs without changing them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--descriptions-file", type=Path, required=True)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--from-prefix", default="/data/gaoya")
    parser.add_argument("--to-prefix", default="/home/gaoya/data")
    parser.add_argument("--expected-cases", type=int, default=198)
    return parser.parse_args()


def map_path(path: Path, source_prefix: str, target_prefix: str) -> Path:
    value = str(path)
    if value == source_prefix or value.startswith(source_prefix + "/"):
        value = target_prefix + value[len(source_prefix) :]
    return Path(value)


def probe_video(ffprobe: Path, path: Path) -> tuple[int, float, float]:
    payload = json.loads(
        subprocess.check_output(
            [
                str(ffprobe),
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
            ],
            text=True,
        )
    )
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return (
        int(stream["nb_read_frames"]),
        float(numerator) / float(denominator),
        float(payload["format"]["duration"]),
    )


def main() -> None:
    args = parse_args()
    source_list = args.source_list.expanduser().resolve()
    descriptions_file = args.descriptions_file.expanduser().resolve()
    ffprobe = args.ffprobe.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    json_root = output_root / "jsons"
    json_root.mkdir(parents=True, exist_ok=True)

    declared = [
        Path(line.strip())
        for line in source_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(declared) != args.expected_cases:
        raise RuntimeError(
            f"expected {args.expected_cases} source cases, found {len(declared)}"
        )

    with descriptions_file.open(newline="", encoding="utf-8") as handle:
        official = [
            row for row in csv.DictReader(handle) if "_take-1_" in row["scenario"]
        ]
    official.sort(key=lambda row: int(row["scenario"].split("_", 1)[0]))
    if len(official) != args.expected_cases:
        raise RuntimeError(
            f"expected {args.expected_cases} official BPP rows, found {len(official)}"
        )

    output_paths: list[Path] = []
    manifest_cases: list[dict[str, object]] = []
    for index, (declared_path, official_row) in enumerate(
        zip(declared, official), start=1
    ):
        source_json = map_path(
            declared_path, args.from_prefix, args.to_prefix
        ).resolve()
        if not source_json.is_file():
            raise FileNotFoundError(source_json)
        payload = json.loads(source_json.read_text(encoding="utf-8"))
        source_video = map_path(
            Path(payload["source_video"]), args.from_prefix, args.to_prefix
        ).resolve()
        expected_id = f"{index:04d}"
        checks = {
            "scenario": payload["benchmark_scenario"] == official_row["scenario"],
            "filename": payload["generated_video_name"]
            == official_row["generated_video_name"],
            "prompt": payload["input_caption"] == official_row["description"],
            "prompt_setting": payload.get("prompt_setting") == "bpp",
            "input_mode": payload.get("input_mode") == "v2v",
            "conditioning_fps": payload.get("conditioning_fps") == 24,
            "conditioning_frames": payload.get("conditioning_frames") == 72,
            "conditioning_duration": math.isclose(
                float(payload.get("conditioning_duration_seconds", -1)),
                3.0,
                abs_tol=1e-9,
            ),
            "contiguous_id": payload["generated_video_name"].startswith(
                expected_id + "_"
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"case {index} does not match official P0 metadata: {failed}"
            )
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        frames, fps, duration = probe_video(ffprobe, source_video)
        if frames != 72 or not math.isclose(fps, 24.0, abs_tol=1e-6):
            raise RuntimeError(
                f"invalid condition {source_video}: {frames} frames @ {fps} FPS"
            )
        if not math.isclose(duration, 3.0, abs_tol=0.001):
            raise RuntimeError(
                f"invalid condition duration {source_video}: {duration} seconds"
            )

        mapped_payload = dict(payload)
        mapped_payload["source_video"] = str(source_video)
        output_path = json_root / source_json.name
        output_path.write_text(
            json.dumps(mapped_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_paths.append(output_path.resolve())
        manifest_cases.append(
            {
                "id": index,
                "scenario": payload["benchmark_scenario"],
                "generated_video_name": payload["generated_video_name"],
                "prompt": payload["input_caption"],
                "conditioning_video": str(source_video),
                "conditioning_frames": frames,
                "conditioning_fps": fps,
            }
        )

    output_list = output_root / "verified_v2v_bpp_198.txt"
    output_list.write_text(
        "".join(f"{path}\n" for path in output_paths), encoding="utf-8"
    )
    manifest = {
        "benchmark": "Physics-IQ Verified",
        "protocol": "P0",
        "prompt_setting": "bpp",
        "input_mode": "v2v",
        "condition": {"frames": 72, "fps": 24, "seconds": 3.0},
        "cases": manifest_cases,
        "descriptions_file": str(descriptions_file),
        "descriptions_sha256": hashlib.sha256(
            descriptions_file.read_bytes()
        ).hexdigest(),
        "source_list": str(source_list),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"P0 input validation=PASS cases={len(output_paths)} list={output_list}")


if __name__ == "__main__":
    main()

