#!/usr/bin/env python3
"""Wan+xSSC batch inference with one common22 public Head-role ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as base

from common22_public_head_targets import ROLES, targets_for_role
from score_extreme_head_targets import GROUPS, targets_for_score_group
from dit_ablation import (
    annotate_result_files,
    cli_path,
    cli_value,
    get_dit_head_ablation_call_count,
    install_grouped_head_ablation,
)


def _extract_args(
) -> tuple[str, list[tuple[int, int]], dict, tuple[int, int] | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--public-head-report", type=Path)
    parser.add_argument("--public-head-role", choices=ROLES)
    parser.add_argument("--score-extreme-selection", type=Path)
    parser.add_argument("--score-extreme-group", choices=GROUPS)
    parser.add_argument("--ablation-step-start", type=int)
    parser.add_argument("--ablation-step-end", type=int)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    using_public = (
        args.public_head_report is not None or args.public_head_role is not None
    )
    using_extreme = (
        args.score_extreme_selection is not None
        or args.score_extreme_group is not None
    )
    if using_public == using_extreme:
        raise ValueError("Specify exactly one complete public-role or score-extreme pair")
    if using_public:
        if args.public_head_report is None or args.public_head_role is None:
            raise ValueError("--public-head-report and --public-head-role are paired")
        targets, source = targets_for_role(
            args.public_head_report, args.public_head_role
        )
        role = str(args.public_head_role)
    else:
        if args.score_extreme_selection is None or args.score_extreme_group is None:
            raise ValueError(
                "--score-extreme-selection and --score-extreme-group are paired"
            )
        role, targets, source = targets_for_score_group(
            args.score_extreme_selection, args.score_extreme_group
        )
    if (args.ablation_step_start is None) != (args.ablation_step_end is None):
        raise ValueError(
            "--ablation-step-start and --ablation-step-end must be paired"
        )
    step_range = (
        None
        if args.ablation_step_start is None
        else (int(args.ablation_step_start), int(args.ablation_step_end))
    )
    return role, targets, source, step_range, remaining


def _case_count(input_list: Path) -> int:
    return len(
        {
            Path(line.strip()).expanduser().resolve()
            for line in input_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    )


def main() -> None:
    role, targets, source, step_range, remaining = _extract_args()
    output_root = cli_path(remaining, "--output-root")
    input_list = cli_path(remaining, "--input-json-list-path")
    if output_root is None or input_list is None:
        raise ValueError("Missing output/input-list option")
    negative_prompt = cli_value(remaining, "--negative-prompt")
    inference_steps = int(cli_value(remaining, "--num-inference-steps") or 40)
    if step_range is not None and not (
        0 <= step_range[0] < step_range[1] <= inference_steps
    ):
        raise ValueError(
            f"Invalid ablation step range {step_range} for {inference_steps} steps"
        )
    category = (
        role
        if step_range is None
        else f"{role}_STEPS{step_range[0]:02d}_{step_range[1]:02d}"
    )
    expected_cases = _case_count(input_list)
    original_build_runtime_model = base._build_runtime_model
    installed_metadata: dict[str, object] | None = None
    installed_dit = None

    def build_runtime_model_with_ablation(args):
        nonlocal installed_metadata, installed_dit
        model, model_args, load_info = original_build_runtime_model(args)
        installed_metadata = install_grouped_head_ablation(
            model.pipe.dit,
            category=category,
            targets=targets,
            active_step_range=step_range,
            total_steps=inference_steps,
            calls_per_step=2,
        )
        installed_metadata["target_selection"] = {
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
            active_steps = (
                inference_steps
                if step_range is None
                else step_range[1] - step_range[0]
            )
            expected = len(targets) * active_steps * 2 * expected_cases
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
