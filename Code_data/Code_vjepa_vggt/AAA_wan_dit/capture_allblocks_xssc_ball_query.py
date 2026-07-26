#!/usr/bin/env python3
"""Run Wan+xSSC once while capturing compact ball-query attention for all blocks."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--attention-map-heads")
    return parser.parse_known_args(argv)


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    query_map = (
        load_case_query_map(custom.attention_query_map)
        if custom.attention_query_map is not None
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
        case_key = output_video.stem
        group = (
            build_case_recorder_group(
                blocks_text=custom.attention_blocks,
                steps_text=custom.attention_steps,
                model_label="xssc",
                output_root=custom.attention_output_root,
                case_key=case_key,
                query_map=query_map,
                map_heads_text=custom.attention_map_heads,
            )
            if query_map is not None
            else build_recorder_group(
                blocks_text=custom.attention_blocks,
                steps_text=custom.attention_steps,
                model_label="xssc",
                output_root=custom.attention_output_root,
                query_coords_text=custom.attention_query_coords,
                query_video_frame=int(custom.attention_query_video_frame),
                query_preview=custom.attention_query_preview,
            )
        )
        group.begin_case(
            output_video.stem,
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
        case_key = Path(str(result.get("input_json", output_video.stem))).stem
        group.set_case_key(case_key)
        summaries = group.finalize_case()
        result["allblock_ball_query_attention"] = {
            str(block): str(path) for block, path in summaries.items()
        }
        logs.append(
            f"[ball-query-attn] wrote {len(summaries)} compact block summaries"
        )
        return result, logs

    batch_base._install_kubric_runtime_hooks = base._install_runtime_hooks
    batch_base._run_single_case_in_process = run_with_attention
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
