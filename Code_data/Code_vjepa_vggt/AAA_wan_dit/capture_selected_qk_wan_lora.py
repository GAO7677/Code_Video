#!/usr/bin/env python3
"""Run Wan+LoRA while capturing selected-head all-token QK matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from allblock_ball_query_utils import install_diffsynth_group
from selected_qk_utils import build_selected_qk_group, load_selection
from self_attention_matrix import DiffSynthAttentionScope


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qk-output-root", type=Path, required=True)
    parser.add_argument("--qk-selection", type=Path, required=True)
    parser.add_argument("--qk-steps", default="5,15,25,35")
    parser.add_argument("--qk-output-bins", type=int, default=512)
    parser.add_argument("--qk-query-chunk", type=int, default=64)
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


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    selection = load_selection(custom.qk_selection)
    mapping = _case_map(_cli_path(remaining, "--input-json-list-path"))

    from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

    original_generate = base.core.generate_one_video

    def generate_with_qk(*args, **kwargs):
        pipe = kwargs["pipe"]
        context_path = Path(kwargs["context_path"]).expanduser().resolve()
        case_key = mapping.get(context_path, context_path.parent.name)
        group = build_selected_qk_group(
            selection=selection,
            model_label="wan_lora",
            case_key=case_key,
            steps_text=custom.qk_steps,
            output_root=custom.qk_output_root,
            output_bins=custom.qk_output_bins,
            query_chunk=custom.qk_query_chunk,
        )
        if group is None:
            return original_generate(*args, **kwargs)
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

    base.core.generate_one_video = generate_with_qk
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
