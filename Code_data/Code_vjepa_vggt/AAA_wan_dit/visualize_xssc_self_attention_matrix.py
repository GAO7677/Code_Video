#!/usr/bin/env python3
"""Run the xSSC baseline while recording Block 17 self-attention matrices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from self_attention_matrix import (
    DiffSynthAttentionScope,
    FullTokenSelfAttentionRecorder,
    MatrixCaptureConfig,
    install_diffsynth_block_recorder,
    parse_step_numbers,
)


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-block", type=int, default=17)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-bins", type=int, default=512)
    parser.add_argument("--attention-query-chunk", type=int, default=128)
    return parser.parse_known_args(argv)


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    config = MatrixCaptureConfig(
        block_id=int(custom.attention_block),
        step_numbers=parse_step_numbers(custom.attention_steps),
        output_bins=int(custom.attention_bins),
        query_chunk=int(custom.attention_query_chunk),
    )

    from code_vjepa_vggt.train0705_kubric_no_gt_box import (
        wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
    )
    from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as base

    original_run = batch_base._run_single_case_in_process

    def run_with_attention(*args, **kwargs):
        model = kwargs["model"]
        output_video = Path(kwargs["output_video"])
        recorder = FullTokenSelfAttentionRecorder(
            config=config,
            model_label="xssc",
            output_root=custom.attention_output_root,
        )
        recorder.begin_case(
            output_video.stem,
            metadata={"generated_video": str(output_video)},
        )
        restore_block = install_diffsynth_block_recorder(model.pipe.dit, recorder)
        scope = DiffSynthAttentionScope(
            pipe=model.pipe,
            recorder=recorder,
            cfg_scale=float(kwargs["cfg_scale"]),
        )
        scope.install()
        try:
            result, logs = original_run(*args, **kwargs)
        finally:
            scope.restore()
            restore_block()
        case_key = Path(str(result.get("input_json", output_video.stem))).stem
        if case_key != recorder.case_key:
            recorder.case_key = case_key
        summary = recorder.finalize_case()
        result["block17_self_attention_matrix"] = str(summary)
        logs.append(f"[self-attn-matrix] {summary}")
        return result, logs

    batch_base._install_kubric_runtime_hooks = base._install_runtime_hooks
    batch_base._run_single_case_in_process = run_with_attention
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()

