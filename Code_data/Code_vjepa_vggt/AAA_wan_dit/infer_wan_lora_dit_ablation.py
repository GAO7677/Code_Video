#!/usr/bin/env python3
"""PhysicIQ-capable Wan+LoRA inference with runtime DiT ablations."""

from __future__ import annotations

import argparse
import json
import sys

from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

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
    original_build_pipeline = base.core.build_pipeline

    def build_pipeline_with_ablation(*args, **kwargs):
        pipe = original_build_pipeline(*args, **kwargs)
        metadata = install_dit_ablation(pipe.dit, spec)
        print(f"[dit_ablation] {json.dumps(metadata, sort_keys=True)}", flush=True)
        return pipe

    base.core.build_pipeline = build_pipeline_with_ablation
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
