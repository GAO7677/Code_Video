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
    parser = argparse.ArgumentParser(description="Run continuation-specific metrics.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--manifest", required=True, help="Path to JSON/JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--run-name", default="continuation_metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from benchlib.continuation import run_continuation_metrics

    config = load_config(args.config)
    samples = load_manifest(args.manifest)
    output = run_continuation_metrics(
        config=config,
        samples=samples,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(output)


if __name__ == "__main__":
    main()
