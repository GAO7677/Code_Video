#!/usr/bin/env python3
"""Run Wan+LoRA once while capturing compact ball-query attention for all blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from allblock_ball_query_utils import (
    build_case_recorder_group,
    build_recorder_group,
    install_diffsynth_group,
    load_case_query_map,
)
from self_attention_matrix import DiffSynthAttentionScope


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-blocks", required=True)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-query-coords", required=True)
    parser.add_argument("--attention-query-video-frame", type=int, required=True)
    parser.add_argument("--attention-query-preview", type=Path, required=True)
    parser.add_argument("--attention-query-map", type=Path)
    parser.add_argument("--attention-query-mode", default="moving")
    parser.add_argument("--attention-map-heads")
    return parser.parse_known_args(argv)


def _case_map(input_list: Path) -> dict[Path, str]:
    output: dict[Path, str] = {}
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        path = Path(line.strip()).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        output[Path(payload["input_video"]).expanduser().resolve()] = path.stem
    return output


def _cli_path(argv: list[str], name: str) -> Path:
    for index, token in enumerate(argv):
        if token == name:
            return Path(argv[index + 1]).expanduser().resolve()
        if token.startswith(f"{name}="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    raise ValueError(f"missing required base option {name}")


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    mapping = _case_map(_cli_path(remaining, "--input-json-list-path"))
    query_map = (
        load_case_query_map(custom.attention_query_map)
        if custom.attention_query_map is not None
        else None
    )

    from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

    original_generate = base.core.generate_one_video

    def generate_with_attention(*args, **kwargs):
        pipe = kwargs["pipe"]
        context_path = Path(kwargs["context_path"]).expanduser().resolve()
        case_key = mapping.get(context_path, context_path.parent.name)
        group = (
            build_case_recorder_group(
                blocks_text=custom.attention_blocks,
                steps_text=custom.attention_steps,
                model_label="wan_lora",
                output_root=custom.attention_output_root,
                case_key=case_key,
                query_map=query_map,
                map_heads_text=custom.attention_map_heads,
                query_mode=custom.attention_query_mode,
            )
            if query_map is not None
            else build_recorder_group(
                blocks_text=custom.attention_blocks,
                steps_text=custom.attention_steps,
                model_label="wan_lora",
                output_root=custom.attention_output_root,
                query_coords_text=custom.attention_query_coords,
                query_video_frame=int(custom.attention_query_video_frame),
                query_preview=custom.attention_query_preview,
            )
        )
        group.begin_case(case_key, metadata={"input_video": str(context_path)})
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
