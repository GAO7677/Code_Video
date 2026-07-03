#!/usr/bin/env python3
"""
Batch generation wrapper for Wan2.1 T2V 1.3B using the official generate.py script.
Reads input JSONs and generates videos one by one.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate videos using Wan2.1 T2V 1.3B")
    parser.add_argument(
        "--input-list",
        type=Path,
        required=True,
        help="Path to text file containing one input JSON path per line",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to save generated videos and result JSONs",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default="/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B",
        help="Wan2.1 T2V 1.3B checkpoint directory",
    )
    parser.add_argument(
        "--wan21-script",
        type=Path,
        default="/home/gaoya/Code_Video/WAN_2p2/Wan2.1-main/generate.py",
        help="Path to Wan2.1 generate.py",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default="/home/gaoya/miniconda3/envs/wan-cu128/bin/python",
        help="Python binary to use",
    )
    parser.add_argument("--size", default="832*480", help="Video size (832*480 or 480*832)")
    parser.add_argument("--frame-num", type=int, default=81, help="Number of frames (4n+1)")
    parser.add_argument("--sample-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--sample-shift", type=float, default=5.0, help="Sampling shift")
    parser.add_argument("--sample-guide-scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed")
    parser.add_argument("--offload-model", action="store_true", help="Offload model to CPU")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def load_input_list(input_list_path: Path) -> list[Path]:
    """Load list of input JSON paths from text file."""
    with input_list_path.open("r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    return [Path(p).resolve() for p in paths]


def load_input_json(json_path: Path) -> dict:
    """Load input JSON and extract caption."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_wan21_generate(
    *,
    python_bin: Path,
    wan21_script: Path,
    ckpt_dir: Path,
    prompt: str,
    save_file: Path,
    args: argparse.Namespace,
) -> None:
    """Run Wan2.1 generate.py for a single prompt."""
    cmd = [
        str(python_bin),
        str(wan21_script),
        "--task", "t2v-1.3B",
        "--ckpt_dir", str(ckpt_dir),
        "--prompt", prompt,
        "--save_file", str(save_file),
        "--size", args.size,
        "--frame_num", str(args.frame_num),
        "--sample_steps", str(args.sample_steps),
        "--sample_shift", str(args.sample_shift),
        "--sample_guide_scale", str(args.sample_guide_scale),
        "--base_seed", str(args.base_seed),
    ]

    if args.offload_model:
        cmd.extend(["--offload_model", "True"])

    print(f"[generate] {save_file.name}")
    print(f"  prompt: {prompt[:80]}..." if len(prompt) > 80 else f"  prompt: {prompt}")
    subprocess.run(cmd, check=True)


def write_result_json(
    *,
    output_json_path: Path,
    input_json: dict,
    output_video_path: Path,
    prompt: str,
) -> None:
    """Write result JSON matching the expected format."""
    result = {
        "input_caption": prompt,
        "input_video": input_json.get("input_video", ""),
        "input_image": input_json.get("input_image", ""),
        "source_video": input_json.get("source_video", ""),
        "output_video": str(output_video_path),
        "model": "Wan2.1-T2V-1.3B",
    }
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    input_list_path = args.input_list.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    wan21_script = args.wan21_script.expanduser().resolve()
    python_bin = args.python_bin.expanduser().resolve()

    if not input_list_path.is_file():
        print(f"Error: input list not found: {input_list_path}", file=sys.stderr)
        sys.exit(1)

    if not ckpt_dir.is_dir():
        print(f"Error: checkpoint directory not found: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    if not wan21_script.is_file():
        print(f"Error: Wan2.1 generate.py not found: {wan21_script}", file=sys.stderr)
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    input_json_paths = load_input_list(input_list_path)
    print(f"Loaded {len(input_json_paths)} input JSONs from {input_list_path}")

    for idx, input_json_path in enumerate(input_json_paths, start=1):
        print(f"\n[{idx}/{len(input_json_paths)}] Processing {input_json_path.name}")

        if not input_json_path.is_file():
            print(f"  Warning: input JSON not found, skipping: {input_json_path}", file=sys.stderr)
            continue

        input_json = load_input_json(input_json_path)
        prompt = input_json.get("input_caption", "")
        if not prompt:
            print(f"  Warning: no input_caption in JSON, skipping", file=sys.stderr)
            continue

        output_video_path = output_root / f"{input_json_path.stem}.mp4"
        output_json_path = output_root / f"{input_json_path.stem}.json"

        if not args.force and output_video_path.is_file() and output_json_path.is_file():
            print(f"  Skipping (already exists): {output_video_path.name}")
            continue

        run_wan21_generate(
            python_bin=python_bin,
            wan21_script=wan21_script,
            ckpt_dir=ckpt_dir,
            prompt=prompt,
            save_file=output_video_path,
            args=args,
        )

        write_result_json(
            output_json_path=output_json_path,
            input_json=input_json,
            output_video_path=output_video_path,
            prompt=prompt,
        )

        print(f"  Done: {output_video_path.name}")

    print(f"\nBatch generation complete. Output: {output_root}")


if __name__ == "__main__":
    main()
