#!/usr/bin/env python3
"""Run PhysRVG once while capturing compact ball-query attention for all blocks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from accelerate.utils import set_seed

from allblock_ball_query_utils import (
    build_case_recorder_group,
    build_recorder_group,
    load_case_query_map,
)
from self_attention_matrix import PhysRVGAttentionProcessorRecorder


DEFAULT_PHYSRVG_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_phys_papers_compare/PhysRVG-main"
)


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-blocks", required=True)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-query-coords", required=True)
    parser.add_argument("--attention-query-video-frame", type=int, required=True)
    parser.add_argument("--attention-query-preview", type=Path, required=True)
    parser.add_argument("--attention-query-map", type=Path)
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

    query_map = (
        load_case_query_map(custom.attention_query_map)
        if custom.attention_query_map is not None
        else None
    )
    default_group = None
    if query_map is None:
        default_group = build_recorder_group(
            blocks_text=custom.attention_blocks,
            steps_text=custom.attention_steps,
            model_label="physrvg",
            output_root=custom.attention_output_root,
            query_coords_text=custom.attention_query_coords,
            query_video_frame=int(custom.attention_query_video_frame),
            query_preview=custom.attention_query_preview,
        )
    active_processors = []
    original_load_pipe = base._load_pipe
    original_run_case = base._run_single_case

    def load_pipe_with_recorders(args):
        return original_load_pipe(args)

    def run_case_with_recorders(**kwargs):
        input_json = Path(kwargs["input_json_path"]).expanduser().resolve()
        pipe = kwargs["pipe"]
        args = kwargs["args"]
        group = (
            build_case_recorder_group(
                blocks_text=custom.attention_blocks,
                steps_text=custom.attention_steps,
                model_label="physrvg",
                output_root=custom.attention_output_root,
                case_key=input_json.stem,
                query_map=query_map,
            )
            if query_map is not None
            else default_group
        )
        processors = [
            PhysRVGAttentionProcessorRecorder(
                recorder=recorder,
                model_module=model_module,
            )
            for recorder in group.recorders
        ]
        active_processors[:] = processors
        for processor in processors:
            processor.install(pipe.transformer)
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
        group.set_grid(
            (
                latent_frames // patch_t,
                (int(args.height) // int(pipe.vae_scale_factor_spatial)) // patch_h,
                (int(args.width) // int(pipe.vae_scale_factor_spatial)) // patch_w,
            )
        )
        group.begin_case(input_json.stem, metadata={"input_json": str(input_json)})
        for processor in processors:
            processor.begin_case()
        set_seed(int(args.seed))
        try:
            result = original_run_case(**kwargs)
            if result[0]:
                summaries = group.finalize_case()
                print(
                    f"[ball-query-attn] wrote {len(summaries)} compact block summaries",
                    flush=True,
                )
            return result
        finally:
            for processor in reversed(processors):
                processor.restore()
            active_processors.clear()

    base._load_pipe = load_pipe_with_recorders
    base._run_single_case = run_case_with_recorders
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        for processor in reversed(active_processors):
            processor.restore()


if __name__ == "__main__":
    main()
