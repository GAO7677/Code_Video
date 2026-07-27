#!/usr/bin/env python3
"""Wan+LoRA inference with one protocol-consistent Head category zeroed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

from consistent_head_targets import CATEGORIES, targets_for_category
from dit_ablation import (
    annotate_result_files,
    cli_path,
    cli_value,
    get_dit_head_ablation_call_count,
    install_grouped_head_ablation,
)


def _extract_args() -> tuple[str, list[tuple[int, int]], dict[str, object], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--head-category", required=True, choices=CATEGORIES)
    parser.add_argument(
        "--classification-metadata",
        required=True,
        type=Path,
    )
    parser.add_argument("--expected-target-count", required=True, type=int)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    category = str(args.head_category).upper()
    targets, source = targets_for_category(
        args.classification_metadata,
        category,
    )
    if len(targets) != args.expected_target_count:
        raise ValueError(
            f"Expected {args.expected_target_count} {category} targets, "
            f"found {len(targets)}"
        )
    return category, targets, source, remaining


def main() -> None:
    category, targets, classification_source, remaining = _extract_args()
    output_root = cli_path(remaining, "--output-root")
    runtime_root = cli_path(remaining, "--runtime-root")
    negative_prompt = cli_value(remaining, "--negative-prompt")
    inference_steps = int(cli_value(remaining, "--num-inference-steps") or 40)
    original_build_pipeline = base.core.build_pipeline
    installed_metadata: dict[str, object] | None = None
    installed_dit = None

    def build_pipeline_with_ablation(*args, **kwargs):
        nonlocal installed_metadata, installed_dit
        pipe = original_build_pipeline(*args, **kwargs)
        installed_metadata = install_grouped_head_ablation(
            pipe.dit,
            category=category,
            targets=targets,
        )
        installed_metadata["target_selection"] = {
            "kind": "protocol_consistent_object_query_category",
            **classification_source,
        }
        installed_dit = pipe.dit
        print(
            "[consistent_category_head_ablation] "
            f"{json.dumps(installed_metadata, sort_keys=True)}",
            flush=True,
        )
        return pipe

    base.core.build_pipeline = build_pipeline_with_ablation
    sys.argv = [sys.argv[0], *remaining]
    try:
        base.main()
    finally:
        if installed_metadata is not None:
            observed = get_dit_head_ablation_call_count(installed_dit)
            # Wan+LoRA evaluates positive and negative CFG branches separately.
            expected = len(targets) * inference_steps * 2
            installed_metadata.update(
                {
                    "observed_target_forward_calls": observed,
                    "expected_target_forward_calls": expected,
                    "target_forward_call_count_ok": observed == expected,
                }
            )
            counts = annotate_result_files(
                [output_root, runtime_root],
                installed_metadata,
                negative_prompt=negative_prompt,
            )
            print(
                "[consistent_category_head_ablation_json] "
                f"{json.dumps(counts, sort_keys=True)}",
                flush=True,
            )
            if observed != expected:
                raise SystemExit(
                    f"Target call count mismatch: expected {expected}, "
                    f"observed {observed}"
                )


if __name__ == "__main__":
    main()
