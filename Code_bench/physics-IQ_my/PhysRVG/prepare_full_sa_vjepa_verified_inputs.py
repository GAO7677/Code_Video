#!/usr/bin/env python3
"""Adapt the shared Physics-IQ P0 JSONs to the Full-SA inference interface.

The benchmark JSON schema calls the conditioning video ``source_video`` while
the reusable PhysRVG inference entrypoint expects ``input_video``.  This
script preserves the official case order and metadata, adds only the interface
alias, and validates the 72-frame/24-FPS condition before inference starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_CASES = 198
EXPECTED_FPS = 24.0
EXPECTED_FRAMES = 72
EXPECTED_DURATION = 3.0
FFPROBE = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--p0-index-parity",
        choices=("all", "odd", "even"),
        default="all",
        help=(
            "Select all cases or a 1-based P0 list parity.  This preserves the "
            "official source index and is used to resume a single-GPU interleaved run."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> tuple[int, float, float]:
    payload = json.loads(
        subprocess.check_output(
            [
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
    input_list = args.input_list.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_list.is_file():
        raise FileNotFoundError(input_list)
    if not FFPROBE.is_file():
        raise FileNotFoundError(FFPROBE)
    if args.limit is not None and not 1 <= args.limit <= EXPECTED_CASES:
        raise ValueError(f"--limit must be between 1 and {EXPECTED_CASES}")

    declared = [
        Path(line.strip()).expanduser().resolve()
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(declared) != EXPECTED_CASES:
        raise ValueError(
            f"P0 input list must contain {EXPECTED_CASES} cases, found {len(declared)}"
        )
    indexed = list(enumerate(declared, start=1))
    if args.p0_index_parity == "odd":
        indexed = [(index, path) for index, path in indexed if index % 2 == 1]
    elif args.p0_index_parity == "even":
        indexed = [(index, path) for index, path in indexed if index % 2 == 0]
    selected = indexed if args.limit is None else indexed[: args.limit]

    output_root.mkdir(parents=True, exist_ok=True)
    normalized_paths: list[Path] = []
    manifest_cases: list[dict[str, object]] = []
    for selected_position, (source_index, source_json) in enumerate(selected, start=1):
        if not source_json.is_file():
            raise FileNotFoundError(source_json)
        payload = json.loads(source_json.read_text(encoding="utf-8"))
        if payload.get("prompt_setting") != "bpp" or payload.get("input_mode") != "v2v":
            raise ValueError(f"case {source_index} is not BPP V2V: {source_json}")
        if payload.get("conditioning_frames") != EXPECTED_FRAMES:
            raise ValueError(f"case {source_index} does not declare 72 frames: {source_json}")
        if float(payload.get("conditioning_fps", -1)) != EXPECTED_FPS:
            raise ValueError(f"case {source_index} does not declare 24 FPS: {source_json}")

        generated_name = str(payload.get("generated_video_name", ""))
        if not generated_name.startswith(f"{source_index:04d}_"):
            raise ValueError(
                f"case order/name mismatch at {source_index}: {generated_name!r}"
            )
        if source_json.stem != Path(generated_name).stem:
            raise ValueError(
                f"JSON stem does not match generated name: {source_json} / {generated_name}"
            )

        source_video = Path(str(payload["source_video"])).expanduser().resolve()
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        frames, fps, duration = probe(source_video)
        if frames != EXPECTED_FRAMES or abs(fps - EXPECTED_FPS) > 1e-6:
            raise ValueError(
                f"invalid condition {source_video}: {frames} frames @ {fps} FPS"
            )
        if abs(duration - EXPECTED_DURATION) > 0.001:
            raise ValueError(
                f"invalid condition duration {source_video}: {duration} seconds"
            )

        normalized = dict(payload)
        normalized["input_video"] = str(source_video)
        normalized["input_caption"] = str(payload["input_caption"])
        normalized["_p0_adapter"] = {
            "source_json": str(source_json),
            "source_video": str(source_video),
            "note": "Only input_video/input_caption aliases were added for infer_full_sa_lora_json_list.py.",
        }
        normalized_path = output_root / source_json.name
        normalized_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        normalized_paths.append(normalized_path)
        manifest_cases.append(
            {
                "index": source_index,
                "selected_position": selected_position,
                "source_json": str(source_json),
                "normalized_json": str(normalized_path),
                "generated_video_name": generated_name,
                "conditioning_video": str(source_video),
                "frames": frames,
                "fps": fps,
                "duration_seconds": duration,
            }
        )

    output_list = output_root / f"verified_v2v_bpp_{len(normalized_paths)}.txt"
    output_list.write_text(
        "".join(f"{path}\n" for path in normalized_paths), encoding="utf-8"
    )
    manifest = {
        "protocol": "physics-iq-verified-bpp-v2v-strict",
        "source_input_list": str(input_list),
        "source_input_list_sha256": sha256(input_list),
        "p0_index_parity": args.p0_index_parity,
        "normalized_input_list": str(output_list),
        "normalized_input_list_sha256": sha256(output_list),
        "num_cases": len(normalized_paths),
        "cases": manifest_cases,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output_list)
    print(json.dumps({"num_cases": len(normalized_paths), "manifest": str(output_root / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
