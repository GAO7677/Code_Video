#!/usr/bin/env python3
"""Run PhysRVG while recording exact ball-query Block 17 attention."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from accelerate.utils import set_seed

from ball_query_attention import (
    BallQuerySelfAttentionRecorder,
    parse_query_coords,
)
from self_attention_matrix import (
    MatrixCaptureConfig,
    PhysRVGAttentionProcessorRecorder,
    parse_step_numbers,
)


DEFAULT_PHYSRVG_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_phys_papers_compare/PhysRVG-main"
)


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-block", type=int, default=17)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-query-coords", required=True)
    parser.add_argument("--attention-query-video-frame", type=int, required=True)
    parser.add_argument("--attention-query-preview", type=Path, required=True)
    parser.add_argument(
        "--physrvg-root",
        type=Path,
        default=Path(os.environ.get("PHYSRVG_ROOT", DEFAULT_PHYSRVG_ROOT)),
    )
    return parser.parse_known_args(argv)


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    physrvg_root = custom.physrvg_root.expanduser().resolve()
    if not physrvg_root.is_dir():
        raise FileNotFoundError(f"PhysRVG root not found: {physrvg_root}")
    sys.path.insert(0, str(physrvg_root))

    import batch_infer_from_input_json_lists as base
    from fastvideo.models.wan_v2v import model_wan_v2v as model_module

    config = MatrixCaptureConfig(
        block_id=int(custom.attention_block),
        step_numbers=parse_step_numbers(custom.attention_steps),
    )
    recorder = BallQuerySelfAttentionRecorder(
        config=config,
        model_label="physrvg",
        output_root=custom.attention_output_root,
        query_coords=parse_query_coords(custom.attention_query_coords),
        query_video_frame=int(custom.attention_query_video_frame),
        query_preview=custom.attention_query_preview,
    )
    processor = PhysRVGAttentionProcessorRecorder(
        recorder=recorder,
        model_module=model_module,
    )
    original_load_pipe = base._load_pipe
    original_run_case = base._run_single_case

    def load_pipe_with_recorder(args):
        pipe = original_load_pipe(args)
        processor.install(pipe.transformer)
        return pipe

    def run_case_with_recorder(**kwargs):
        input_json = Path(kwargs["input_json_path"]).expanduser().resolve()
        pipe = kwargs["pipe"]
        args = kwargs["args"]
        transformer = getattr(
            getattr(pipe.transformer, "base_model", None),
            "model",
            pipe.transformer,
        )
        patch_t, patch_h, patch_w = (
            int(value) for value in transformer.config.patch_size
        )
        latent_frames = (
            (int(args.num_frames) - 1) // int(pipe.vae_scale_factor_temporal)
        ) + 1
        recorder.set_grid(
            (
                latent_frames // patch_t,
                (int(args.height) // int(pipe.vae_scale_factor_spatial)) // patch_h,
                (int(args.width) // int(pipe.vae_scale_factor_spatial)) // patch_w,
            )
        )
        recorder.begin_case(input_json.stem, metadata={"input_json": str(input_json)})
        processor.begin_case()
        set_seed(int(args.seed))
        result = original_run_case(**kwargs)
        if result[0]:
            summary = recorder.finalize_case()
            print(f"[ball-query-attn] {summary}", flush=True)
        return result

    base._load_pipe = load_pipe_with_recorder
    base._run_single_case = run_case_with_recorder
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        processor.restore()


if __name__ == "__main__":
    main()
