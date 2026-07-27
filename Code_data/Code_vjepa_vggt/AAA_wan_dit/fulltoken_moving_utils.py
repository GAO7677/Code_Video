#!/usr/bin/env python3
"""Construction helpers for the full-token moving-trajectory experiment."""

from __future__ import annotations

from pathlib import Path

from allblock_ball_query_utils import parse_block_ids
from ball_query_attention import BallQueryRecorderGroup
from fulltoken_moving_attention import FullTokenMovingRecorder
from moving_query_attention import explicit_moving_query_coords, moving_query_coords
from self_attention_matrix import MatrixCaptureConfig, parse_step_numbers


def build_fulltoken_moving_group(
    *,
    blocks_text: str,
    steps_text: str,
    model_label: str,
    output_root: Path,
    case_key: str,
    query_map: dict,
    query_chunk: int,
    compact_storage: bool = False,
) -> BallQueryRecorderGroup:
    if case_key not in query_map:
        raise KeyError(f"query map has no entry for case {case_key}")
    item = query_map[case_key]
    if "query_coords_per_time" in item:
        coords = explicit_moving_query_coords(item["query_coords_per_time"])
    else:
        coords = moving_query_coords(
            item["trajectory"],
            frame_shape=tuple(int(value) for value in item["frame_shape"]),
        )
    blocks = parse_block_ids(blocks_text)
    steps = parse_step_numbers(steps_text)
    recorders = [
        FullTokenMovingRecorder(
            config=MatrixCaptureConfig(
                block_id=block,
                step_numbers=steps,
                query_chunk=int(query_chunk),
            ),
            model_label=model_label,
            output_root=output_root / f"block{block:02d}" / "matrices",
            trajectory_coords=coords,
            query_preview=Path(item["preview"]),
            compact_storage=compact_storage,
        )
        for block in blocks
    ]
    return BallQueryRecorderGroup(recorders)
