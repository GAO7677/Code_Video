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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the recommended benchmark bundle.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--manifest", required=True, help="Path to JSON/JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Output root directory.")
    parser.add_argument("--resolution", default="1-1", help="Reference image ratio for I2V wrapper.")
    parser.add_argument("--skip-short", action="store_true")
    parser.add_argument("--skip-i2v", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--skip-continuation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from benchlib.continuation import run_continuation_metrics
    from benchlib.vbench_wrappers import run_vbench_i2v, run_vbench_long, run_vbench_short

    config = load_config(args.config)
    samples = load_manifest(args.manifest)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_short:
        print(
            run_vbench_short(
                config=config,
                samples=samples,
                output_dir=str(output_root / "short"),
                run_name="vbench_short",
            )
        )
    if not args.skip_i2v:
        print(
            run_vbench_i2v(
                config=config,
                samples=samples,
                output_dir=str(output_root / "i2v"),
                run_name="vbench_i2v",
                resolution=args.resolution,
            )
        )
    if not args.skip_long:
        print(
            run_vbench_long(
                config=config,
                samples=samples,
                output_dir=str(output_root / "long"),
                run_name="vbench_long",
            )
        )
    if not args.skip_continuation:
        print(
            run_continuation_metrics(
                config=config,
                samples=samples,
                output_dir=str(output_root / "continuation"),
                run_name="continuation_metrics",
            )
        )


if __name__ == "__main__":
    main()
