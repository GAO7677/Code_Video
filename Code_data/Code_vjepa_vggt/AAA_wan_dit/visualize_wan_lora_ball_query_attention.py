#!/usr/bin/env python3
"""Run Wan+LoRA while recording exact ball-query Block 17 attention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ball_query_attention import (
    BallQuerySelfAttentionRecorder,
    parse_query_coords,
)
from self_attention_matrix import (
    DiffSynthAttentionScope,
    MatrixCaptureConfig,
    install_diffsynth_block_recorder,
    parse_step_numbers,
)


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-block", type=int, default=17)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-query-coords", required=True)
    parser.add_argument("--attention-query-video-frame", type=int, required=True)
    parser.add_argument("--attention-query-preview", type=Path, required=True)
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
    config = MatrixCaptureConfig(
        block_id=int(custom.attention_block),
        step_numbers=parse_step_numbers(custom.attention_steps),
    )
    mapping = _case_map(_cli_path(remaining, "--input-json-list-path"))

    from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

    original_generate = base.core.generate_one_video

    def generate_with_attention(*args, **kwargs):
        pipe = kwargs["pipe"]
        context_path = Path(kwargs["context_path"]).expanduser().resolve()
        case_key = mapping.get(context_path, context_path.parent.name)
        recorder = BallQuerySelfAttentionRecorder(
            config=config,
            model_label="wan_lora",
            output_root=custom.attention_output_root,
            query_coords=parse_query_coords(custom.attention_query_coords),
            query_video_frame=int(custom.attention_query_video_frame),
            query_preview=custom.attention_query_preview,
        )
        recorder.begin_case(case_key, metadata={"input_video": str(context_path)})
        restore_block = install_diffsynth_block_recorder(pipe.dit, recorder)
        scope = DiffSynthAttentionScope(
            pipe=pipe,
            recorder=recorder,
            cfg_scale=float(kwargs["cfg_scale"]),
        )
        scope.install()
        try:
            result = original_generate(*args, **kwargs)
        finally:
            scope.restore()
            restore_block()
        recorder.finalize_case()
        return result

    base.core.generate_one_video = generate_with_attention
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
