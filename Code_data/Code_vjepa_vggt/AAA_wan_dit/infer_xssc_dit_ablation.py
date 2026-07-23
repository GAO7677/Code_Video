#!/usr/bin/env python3
"""xSSC object-cross-attention inference with runtime Wan DiT ablations."""

from __future__ import annotations

import argparse
import json
import sys

from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as base

from dit_ablation import ABLATION_MODES, DiTAblationSpec, install_dit_ablation


def _extract_ablation_args() -> tuple[DiTAblationSpec, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dit-ablation-mode",
        choices=ABLATION_MODES,
        default="baseline",
    )
    parser.add_argument("--dit-ablation-block", type=int, default=None)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    spec = DiTAblationSpec(
        mode=str(args.dit_ablation_mode),
        block_id=args.dit_ablation_block,
    )
    return spec, remaining


def main() -> None:
    spec, remaining = _extract_ablation_args()
    spec.validate(30)
    original_build_runtime_model = base._build_runtime_model

    def build_runtime_model_with_ablation(args):
        model, model_args, load_info = original_build_runtime_model(args)
        metadata = install_dit_ablation(model.pipe.dit, spec)
        model._aaa_wan_dit_ablation = metadata
        print(f"[dit_ablation] {json.dumps(metadata, sort_keys=True)}", flush=True)
        return model, model_args, load_info

    base._build_runtime_model = build_runtime_model_with_ablation
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
