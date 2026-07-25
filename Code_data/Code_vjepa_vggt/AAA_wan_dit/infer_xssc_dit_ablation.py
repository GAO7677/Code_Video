#!/usr/bin/env python3
"""xSSC object-cross-attention inference with runtime Wan DiT ablations."""

from __future__ import annotations

import argparse
import json
import sys

from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as base

from dit_ablation import (
    ABLATION_MODES,
    DiTAblationSpec,
    annotate_result_files,
    cli_path,
    cli_value,
    get_dit_head_ablation_call_count,
    install_dit_ablation,
)


def _extract_ablation_args() -> tuple[DiTAblationSpec, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dit-ablation-mode",
        choices=ABLATION_MODES,
        default="baseline",
    )
    parser.add_argument("--dit-ablation-block", type=int, default=None)
    parser.add_argument("--dit-ablation-head", type=int, default=None)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    spec = DiTAblationSpec(
        mode=str(args.dit_ablation_mode),
        block_id=args.dit_ablation_block,
        head_id=args.dit_ablation_head,
    )
    return spec, remaining


def main() -> None:
    spec, remaining = _extract_ablation_args()
    spec.validate(30)
    output_root = cli_path(remaining, "--output-root")
    negative_prompt = cli_value(remaining, "--negative-prompt")
    original_build_runtime_model = base._build_runtime_model
    installed_metadata: dict[str, object] | None = None
    installed_dit = None

    def build_runtime_model_with_ablation(args):
        nonlocal installed_metadata, installed_dit
        model, model_args, load_info = original_build_runtime_model(args)
        metadata = install_dit_ablation(model.pipe.dit, spec)
        installed_metadata = metadata
        installed_dit = model.pipe.dit
        model._aaa_wan_dit_ablation = metadata
        print(f"[dit_ablation] {json.dumps(metadata, sort_keys=True)}", flush=True)
        return model, model_args, load_info

    base._build_runtime_model = build_runtime_model_with_ablation
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        if installed_metadata is not None:
            observed_calls = get_dit_head_ablation_call_count(installed_dit)
            installed_metadata["observed_target_forward_calls"] = observed_calls
            counts = annotate_result_files(
                [output_root],
                installed_metadata,
                negative_prompt=negative_prompt,
            )
            print(f"[dit_ablation_json] {json.dumps(counts, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
