#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render_sim_0705 import render_generated_case


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705_preview_case")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a single dataset_new_0705 preview case.")
    parser.add_argument("--family-key", default="F3", choices=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"])
    parser.add_argument("--sample-key", default="f3_preview_case_000")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_generated_case(
        family_key=args.family_key,
        sample_key=args.sample_key,
        seed=args.seed,
        output_root=args.output_root,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
