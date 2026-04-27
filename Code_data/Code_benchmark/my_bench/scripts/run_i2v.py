#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchlib.config import load_config
from benchlib.manifest import load_manifest

DEFAULT_DIMENSIONS = [
    "i2v_subject",
    "i2v_background",
    "camera_motion",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VBench-I2V custom-input benchmark.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--manifest", required=True, help="Path to JSON/JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--run-name", default="vbench_i2v")
    parser.add_argument("--resolution", default="1-1", help="Reference image ratio label, e.g. 1-1, 16-9.")
    parser.add_argument("--dimensions", nargs="*", default=DEFAULT_DIMENSIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from benchlib.vbench_wrappers import run_vbench_i2v

    config = load_config(args.config)
    samples = load_manifest(args.manifest)
    output = run_vbench_i2v(
        config=config,
        samples=samples,
        output_dir=args.output_dir,
        dimensions=args.dimensions,
        run_name=args.run_name,
        resolution=args.resolution,
    )
    print(output)


if __name__ == "__main__":
    main()
