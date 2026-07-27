#!/usr/bin/env python3
"""Run PhysRVG and capture exact compact full-token trajectory statistics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from accelerate.utils import set_seed

from allblock_ball_query_utils import load_case_query_map
from fulltoken_moving_utils import build_fulltoken_moving_group
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
    parser.add_argument("--attention-query-map", type=Path, required=True)
    parser.add_argument("--attention-query-chunk", type=int, default=64)
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

    query_map = load_case_query_map(custom.attention_query_map)
    active_processors = []
    original_run_case = base._run_single_case

    def run_case_with_recorders(**kwargs):
        input_json = Path(kwargs["input_json_path"]).expanduser().resolve()
        pipe = kwargs["pipe"]
        args = kwargs["args"]
        group = build_fulltoken_moving_group(
            blocks_text=custom.attention_blocks,
            steps_text=custom.attention_steps,
            model_label="physrvg",
            output_root=custom.attention_output_root,
            case_key=input_json.stem,
            query_map=query_map,
            query_chunk=custom.attention_query_chunk,
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
                    f"[fulltoken-moving] wrote {len(summaries)} block summaries",
                    flush=True,
                )
            return result
        finally:
            for processor in reversed(processors):
                processor.restore()
            active_processors.clear()

    base._run_single_case = run_case_with_recorders
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        for processor in reversed(active_processors):
            processor.restore()


if __name__ == "__main__":
    main()
