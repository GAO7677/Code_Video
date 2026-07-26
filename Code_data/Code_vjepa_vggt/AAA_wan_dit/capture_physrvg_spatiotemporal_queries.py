#!/usr/bin/env python3
"""Run official PhysRVG and record exact per-query same/cross-frame attention."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from accelerate.utils import set_seed

from self_attention_matrix import (
    MatrixCaptureConfig,
    PhysRVGAttentionProcessorRecorder,
    parse_step_numbers,
)
from spatiotemporal_query_attention import ExactSpatiotemporalQueryRecorder


DEFAULT_PHYSRVG_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_phys_papers_compare/PhysRVG-main"
)


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-block", type=int, default=17)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-query-chunk", type=int, default=128)
    parser.add_argument("--seed-map-json", type=Path, required=True)
    parser.add_argument(
        "--physrvg-root",
        type=Path,
        default=Path(os.environ.get("PHYSRVG_ROOT", DEFAULT_PHYSRVG_ROOT)),
    )
    return parser.parse_known_args(argv)


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    physrvg_root = custom.physrvg_root.expanduser().resolve()
    sys.path.insert(0, str(physrvg_root))
    seed_map = json.loads(
        custom.seed_map_json.expanduser().resolve().read_text(encoding="utf-8")
    )

    import batch_infer_from_input_json_lists as base
    from fastvideo.models.wan_v2v import model_wan_v2v as model_module

    config = MatrixCaptureConfig(
        block_id=int(custom.attention_block),
        step_numbers=parse_step_numbers(custom.attention_steps),
        output_bins=1,
        query_chunk=int(custom.attention_query_chunk),
    )
    recorder = ExactSpatiotemporalQueryRecorder(
        config=config,
        model_label="physrvg",
        output_root=custom.attention_output_root,
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
        stem = input_json.stem
        if stem not in seed_map:
            raise KeyError(f"{stem} is missing from {custom.seed_map_json}")
        entry = dict(seed_map[stem])
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
                (int(args.height) // int(pipe.vae_scale_factor_spatial))
                // patch_h,
                (int(args.width) // int(pipe.vae_scale_factor_spatial))
                // patch_w,
            )
        )
        recorder.begin_case(
            str(entry["case_key"]),
            metadata={
                **entry,
                "input_json": str(input_json),
            },
        )
        processor.begin_case()
        original_seed = int(args.seed)
        args.seed = int(entry["seed"])
        set_seed(int(args.seed))
        try:
            result = original_run_case(**kwargs)
        finally:
            args.seed = original_seed
        if result[0]:
            summary = recorder.finalize_case()
            print(f"[same-cross] {summary}", flush=True)
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
