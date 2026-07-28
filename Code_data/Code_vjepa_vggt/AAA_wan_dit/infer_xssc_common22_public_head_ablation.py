#!/usr/bin/env python3
"""Wan+xSSC batch inference with one common22 public Head-role ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as base

from common22_public_head_targets import ROLES, targets_for_role
from dit_ablation import (
    annotate_result_files,
    cli_path,
    cli_value,
    get_dit_head_ablation_call_count,
    install_grouped_head_ablation,
)


def _extract_args() -> tuple[str, list[tuple[int, int]], dict, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--public-head-report", type=Path, required=True)
    parser.add_argument("--public-head-role", choices=ROLES, required=True)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    targets, source = targets_for_role(args.public_head_report, args.public_head_role)
    return str(args.public_head_role), targets, source, remaining


def _case_count(input_list: Path) -> int:
    return len(
        {
            Path(line.strip()).expanduser().resolve()
            for line in input_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )


def main() -> None:
    role, targets, source, remaining = _extract_args()
    output_root = cli_path(remaining, "--output-root")
    input_list = cli_path(remaining, "--input-json-list-path")
    if output_root is None or input_list is None:
        raise ValueError("Missing output/input-list option")
    negative_prompt = cli_value(remaining, "--negative-prompt")
    inference_steps = int(cli_value(remaining, "--num-inference-steps") or 40)
    expected_cases = _case_count(input_list)
    original_build_runtime_model = base._build_runtime_model
    installed_metadata: dict[str, object] | None = None
    installed_dit = None

    def build_runtime_model_with_ablation(args):
        nonlocal installed_metadata, installed_dit
        model, model_args, load_info = original_build_runtime_model(args)
        installed_metadata = install_grouped_head_ablation(
            model.pipe.dit,
            category=role,
            targets=targets,
        )
        installed_metadata["target_selection"] = {
            "kind": "common22_cross_model_public_stable_role",
            **source,
        }
        installed_dit = model.pipe.dit
        model._aaa_wan_dit_ablation = installed_metadata
        print(f"[common22-public] {json.dumps(installed_metadata, sort_keys=True)}")
        return model, model_args, load_info

    base._build_runtime_model = build_runtime_model_with_ablation
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        if installed_metadata is not None:
            observed = get_dit_head_ablation_call_count(installed_dit)
            expected = len(targets) * inference_steps * 2 * expected_cases
            installed_metadata.update(
                {
                    "expected_cases": expected_cases,
                    "observed_target_forward_calls": observed,
                    "expected_target_forward_calls": expected,
                    "target_forward_call_count_ok": observed == expected,
                }
            )
            annotate_result_files(
                [output_root],
                installed_metadata,
                negative_prompt=negative_prompt,
            )
            if observed != expected:
                raise SystemExit(
                    f"Target call count mismatch: expected {expected}, observed {observed}"
                )


if __name__ == "__main__":
    main()
