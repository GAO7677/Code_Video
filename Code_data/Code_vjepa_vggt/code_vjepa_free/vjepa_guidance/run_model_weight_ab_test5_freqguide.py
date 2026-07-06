#!/usr/bin/env python3
from __future__ import annotations

"""
Run the 4-family frequency-guidance baseline/guided A/B on the deduplicated
test_5 list, with outputs isolated from older model-weight A/B runs.

Generate only:
CUDA_VISIBLE_DEVICES=5,6,7 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5_freqguide.py \
  --stage generate

Score only:
CUDA_VISIBLE_DEVICES=7 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5_freqguide.py \
  --stage score
"""

import argparse
import sys
from pathlib import Path

try:
    from . import run_model_weight_ab_test5 as base
except ImportError:
    import run_model_weight_ab_test5 as base


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_freqguide_20260706")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper for the 4-family frequency-guidance A/B runner with an isolated output root."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated videos, scores, and dashboards.",
    )
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough
    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    sys.argv = [str(Path(base.__file__).resolve()), "--output-root", str(output_root), *args.passthrough]
    base.main()


if __name__ == "__main__":
    main()
