#!/usr/bin/env python3
"""Shared construction helpers for compact all-block ball-query capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ball_query_attention import (
    BallQueryRecorderGroup,
    BallQuerySelfAttentionRecorder,
    parse_query_coords,
)
from self_attention_matrix import (
    MatrixCaptureConfig,
    install_diffsynth_block_recorder,
    parse_step_numbers,
)
from moving_query_attention import MovingQueryFeatureRecorder, moving_query_coords


def parse_block_ids(text: str) -> tuple[int, ...]:
    blocks = tuple(int(value) for value in text.split(",") if value.strip())
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("attention block list must be non-empty and unique")
    if min(blocks) < 0:
        raise ValueError("attention block ids must be non-negative")
    return blocks


def build_recorder_group(
    *,
    blocks_text: str,
    steps_text: str,
    model_label: str,
    output_root: Path,
    query_coords_text: str,
    query_video_frame: int,
    query_preview: Path,
) -> BallQueryRecorderGroup:
    blocks = parse_block_ids(blocks_text)
    steps = parse_step_numbers(steps_text)
    recorders = [
        BallQuerySelfAttentionRecorder(
            config=MatrixCaptureConfig(block_id=block, step_numbers=steps),
            model_label=model_label,
            output_root=output_root / f"block{block:02d}" / "matrices",
            query_coords=parse_query_coords(query_coords_text),
            query_video_frame=query_video_frame,
            query_preview=query_preview,
            render_images=False,
        )
        for block in blocks
    ]
    return BallQueryRecorderGroup(recorders)


def load_case_query_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise ValueError(f"query map contains no cases: {path}")
    return cases


def build_case_recorder_group(
    *,
    blocks_text: str,
    steps_text: str,
    model_label: str,
    output_root: Path,
    case_key: str,
    query_map: dict[str, dict[str, Any]],
) -> BallQueryRecorderGroup:
    if case_key not in query_map:
        raise KeyError(f"query map has no entry for case {case_key}")
    item = query_map[case_key]
    blocks = parse_block_ids(blocks_text)
    steps = parse_step_numbers(steps_text)
    coords = moving_query_coords(
        item["trajectory"],
        frame_shape=tuple(int(value) for value in item["frame_shape"]),
    )
    return BallQueryRecorderGroup(
        [
            MovingQueryFeatureRecorder(
                config=MatrixCaptureConfig(block_id=block, step_numbers=steps),
                model_label=model_label,
                output_root=output_root / f"block{block:02d}" / "matrices",
                query_coords=coords,
                query_preview=Path(item["preview"]),
            )
            for block in blocks
        ]
    )


def install_diffsynth_group(
    dit: Any, group: BallQueryRecorderGroup
) -> Callable[[], None]:
    restores = [
        install_diffsynth_block_recorder(dit, recorder)
        for recorder in group.recorders
    ]

    def restore() -> None:
        for callback in reversed(restores):
            callback()

    return restore
