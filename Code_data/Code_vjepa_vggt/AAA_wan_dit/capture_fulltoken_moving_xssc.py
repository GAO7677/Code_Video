#!/usr/bin/env python3
"""Run Wan+xSSC and capture exact compact full-token trajectory statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from allblock_ball_query_utils import install_diffsynth_group, load_case_query_map
from fulltoken_moving_utils import build_fulltoken_moving_group
from self_attention_matrix import DiffSynthAttentionScope


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-blocks", required=True)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-query-map", type=Path, required=True)
    parser.add_argument("--attention-query-chunk", type=int, default=64)
    parser.add_argument(
        "--attention-case-filter",
        help="Optional comma-separated case stems; other cases run without capture.",
    )
    return parser.parse_known_args(argv)


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    query_map = load_case_query_map(custom.attention_query_map)
    case_filter = (
        {value.strip() for value in custom.attention_case_filter.split(",") if value.strip()}
        if custom.attention_case_filter
        else None
    )

    from code_vjepa_vggt.train0705_kubric_no_gt_box import (
        wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
    )
    from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as base

    original_run = batch_base._run_single_case_in_process

    def run_with_attention(*args, **kwargs):
        model = kwargs["model"]
        output_video = Path(kwargs["output_video"])
        provisional_key = output_video.stem
        if case_filter is not None and provisional_key not in case_filter:
            return original_run(*args, **kwargs)
        group = build_fulltoken_moving_group(
            blocks_text=custom.attention_blocks,
            steps_text=custom.attention_steps,
            model_label="xssc",
            output_root=custom.attention_output_root,
            case_key=provisional_key,
            query_map=query_map,
            query_chunk=custom.attention_query_chunk,
        )
        group.begin_case(
            provisional_key,
            metadata={"generated_video": str(output_video)},
        )
        restore_blocks = install_diffsynth_group(model.pipe.dit, group)
        scope = DiffSynthAttentionScope(
            pipe=model.pipe,
            recorder=group,
            cfg_scale=float(kwargs["cfg_scale"]),
        )
        scope.install()
        try:
            result, logs = original_run(*args, **kwargs)
        finally:
            scope.restore()
            restore_blocks()
        case_key = Path(str(result.get("input_json", provisional_key))).stem
        group.set_case_key(case_key)
        summaries = group.finalize_case()
        result["fulltoken_moving_attention"] = {
            str(block): str(path) for block, path in summaries.items()
        }
        logs.append(
            f"[fulltoken-moving] wrote {len(summaries)} compact block summaries"
        )
        return result, logs

    batch_base._install_kubric_runtime_hooks = base._install_runtime_hooks
    batch_base._run_single_case_in_process = run_with_attention
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
