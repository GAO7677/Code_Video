#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rerank_video.pipeline import run_pipeline
from rerank_video.schemas import load_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple video candidates, score them, and rerank.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = load_run_config(payload)
    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
