#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchlib.config import load_config
from benchlib.weight_linker import link_manual_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symlink externally downloaded weights into VBench cache layout.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    linked = link_manual_weights(config)
    for key, src, dst in linked:
        print(f"{key}: {src} -> {dst}")


if __name__ == "__main__":
    main()

