#!/usr/bin/env python3
"""Capture two identity-locked moving-query tracks for every Wan block."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from allblock_ball_query_utils import install_diffsynth_group, parse_block_ids
from ball_query_attention import BallQueryRecorderGroup
from self_attention_matrix import (
    DiffSynthAttentionScope,
    MatrixCaptureConfig,
    parse_step_numbers,
)
from two_ball_attention import TwoBallAttentionRecorder


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-blocks", required=True)
    parser.add_argument("--attention-step", type=int, default=25)
    parser.add_argument("--attention-query-map", type=Path, required=True)
    parser.add_argument("--attention-map-heads", required=True)
    return parser.parse_known_args(argv)


def _cli_path(argv: list[str], name: str) -> Path:
    for index, token in enumerate(argv):
        if token == name:
            return Path(argv[index + 1]).expanduser().resolve()
        if token.startswith(f"{name}="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    raise ValueError(f"missing required base option {name}")


def _case_map(input_list: Path) -> dict[Path, str]:
    output = {}
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip()).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        output[Path(payload["input_video"]).expanduser().resolve()] = path.stem
    return output


def _track_coords(track: dict) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        tuple(int(value) for value in coord)
        for group in track["query_coords_per_time"]
        for coord in group
    )


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    mapping = _case_map(_cli_path(remaining, "--input-json-list-path"))
    query_payload = json.loads(
        custom.attention_query_map.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    tracks = query_payload["tracks"]
    if len(tracks) != 2:
        raise ValueError("query map must contain exactly two tracks")
    track_names = tuple(str(track["name"]) for track in tracks)
    track_coords = tuple(_track_coords(track) for track in tracks)
    blocks = parse_block_ids(custom.attention_blocks)
    steps = parse_step_numbers(str(custom.attention_step))
    selected_heads = tuple(
        int(value)
        for value in custom.attention_map_heads.split(",")
        if value.strip()
    )

    from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

    original_generate = base.core.generate_one_video

    def generate_with_attention(*args, **kwargs):
        pipe = kwargs["pipe"]
        context_path = Path(kwargs["context_path"]).expanduser().resolve()
        case_key = mapping.get(context_path, context_path.parent.name)
        if case_key != query_payload["case"]:
            raise KeyError(f"two-ball query map does not contain {case_key}")
        group = BallQueryRecorderGroup(
            [
                TwoBallAttentionRecorder(
                    config=MatrixCaptureConfig(
                        block_id=block, step_numbers=steps
                    ),
                    model_label="wan_lora",
                    output_root=(
                        custom.attention_output_root
                        / f"block{block:02d}"
                        / "matrices"
                    ),
                    track_names=track_names,
                    track_coords=track_coords,
                    selected_heads=selected_heads,
                    query_preview=Path(query_payload["preview"]),
                )
                for block in blocks
            ]
        )
        group.begin_case(
            case_key,
            metadata={
                "input_video": str(context_path),
                "query_map": str(custom.attention_query_map),
            },
        )
        restore_blocks = install_diffsynth_group(pipe.dit, group)
        scope = DiffSynthAttentionScope(
            pipe=pipe,
            recorder=group,
            cfg_scale=float(kwargs["cfg_scale"]),
        )
        scope.install()
        try:
            result = original_generate(*args, **kwargs)
        finally:
            scope.restore()
            restore_blocks()
        group.finalize_case()
        return result

    base.core.generate_one_video = generate_with_attention
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
