#!/usr/bin/env python3
"""Run one existing baseline recorder with an exact ball-query recorder."""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
from pathlib import Path

from ball_query_attention import BallQuerySelfAttentionRecorder


MODULES = {
    "wan_lora": "visualize_wan_lora_self_attention_matrix",
    "xssc": "visualize_xssc_self_attention_matrix",
    "physrvg": "visualize_physrvg_self_attention_matrix",
}


def parse_query_cells(text: str) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    for item in text.split(","):
        row_text, col_text = item.strip().split(":", 1)
        cells.append((int(row_text), int(col_text)))
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("query cells must be a non-empty unique row:col list")
    return tuple(cells)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ball-query-model", choices=tuple(MODULES), required=True)
    parser.add_argument("--ball-query-time", type=int, required=True)
    parser.add_argument("--ball-query-cells", required=True)
    parser.add_argument("--ball-query-reference-video", type=Path, required=True)
    parser.add_argument("--ball-query-reference-frame", type=int, default=24)
    return parser.parse_known_args()


def main() -> None:
    custom, remaining = parse_args()
    reference = custom.ball_query_reference_video.expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    factory = functools.partial(
        BallQuerySelfAttentionRecorder,
        query_time=int(custom.ball_query_time),
        query_cells=parse_query_cells(custom.ball_query_cells),
        query_reference_video=reference,
        query_reference_frame=int(custom.ball_query_reference_frame),
    )
    module = importlib.import_module(MODULES[custom.ball_query_model])
    module.FullTokenSelfAttentionRecorder = factory
    sys.argv = [sys.argv[0], *remaining]
    module.main()


if __name__ == "__main__":
    main()
