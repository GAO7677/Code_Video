#!/usr/bin/env python3
"""Wan+LoRA inference with one cross-block Head-category ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

from dit_ablation import (
    annotate_result_files,
    cli_path,
    cli_value,
    get_dit_head_ablation_call_count,
    install_grouped_head_ablation,
)
from grouped_head_targets import CATEGORY_TARGETS, targets_for_category


def _extract_args() -> tuple[str, list[tuple[int, int]], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--grouped-head-category",
        choices=tuple(CATEGORY_TARGETS),
    )
    parser.add_argument("--grouped-head-targets-csv", type=Path)
    parser.add_argument("--grouped-head-object", choices=("A", "B"))
    parser.add_argument("--expected-target-count", type=int)
    args, remaining = parser.parse_known_args(sys.argv[1:])
    using_category = args.grouped_head_category is not None
    using_csv = args.grouped_head_targets_csv is not None
    if using_category == using_csv:
        raise ValueError(
            "Specify exactly one of --grouped-head-category or "
            "--grouped-head-targets-csv"
        )
    if using_category:
        if args.grouped_head_object is not None:
            raise ValueError(
                "--grouped-head-object is only valid with a targets CSV"
            )
        category = str(args.grouped_head_category)
        targets = targets_for_category(category)
    else:
        if args.grouped_head_object is None:
            raise ValueError(
                "--grouped-head-object A|B is required with a targets CSV"
            )
        path = args.grouped_head_targets_csv.expanduser().resolve()
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        accepted_scopes = (
            {"A+B", "A-only"}
            if args.grouped_head_object == "A"
            else {"A+B", "B-only"}
        )
        targets = [
            (int(row["block"]), int(row["head"]))
            for row in rows
            if row["scope"] in accepted_scopes
        ]
        category = f"PREV_{args.grouped_head_object}"
    if (
        args.expected_target_count is not None
        and len(targets) != args.expected_target_count
    ):
        raise ValueError(
            f"Expected {args.expected_target_count} targets, found "
            f"{len(targets)}"
        )
    return category, targets, remaining


def main() -> None:
    category, targets, remaining = _extract_args()
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
        installed_dit = pipe.dit
        print(
            f"[grouped_head_ablation] "
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
            # DiffSynth evaluates conditional and unconditional branches separately.
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
                f"[grouped_head_ablation_json] "
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
