#!/usr/bin/env python3
"""Run Wan+xSSC while capturing selected-head all-token QK matrices."""

from __future__ import annotations

import argparse
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


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    selection = load_selection(custom.qk_selection)

    from code_vjepa_vggt.train0705_kubric_no_gt_box import (
        wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
    )
    from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as base

    original_run = batch_base._run_single_case_in_process

    def run_with_qk(*args, **kwargs):
        model = kwargs["model"]
        output_video = Path(kwargs["output_video"])
        case_key = output_video.stem
        group = build_selected_qk_group(
            selection=selection,
            model_label="xssc",
            case_key=case_key,
            steps_text=custom.qk_steps,
            output_root=custom.qk_output_root,
            output_bins=custom.qk_output_bins,
            query_chunk=custom.qk_query_chunk,
        )
        if group is None:
            return original_run(*args, **kwargs)
        group.begin_case(case_key, metadata={"generated_video": str(output_video)})
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
        final_key = Path(str(result.get("input_json", case_key))).stem
        group.set_case_key(final_key)
        summaries = group.finalize_case()
        result["selected_qk_attention"] = {
            str(block): str(path) for block, path in summaries.items()
        }
        logs.append(f"[selected-qk] wrote {len(summaries)} block summaries")
        return result, logs

    batch_base._install_kubric_runtime_hooks = base._install_runtime_hooks
    batch_base._run_single_case_in_process = run_with_qk
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
