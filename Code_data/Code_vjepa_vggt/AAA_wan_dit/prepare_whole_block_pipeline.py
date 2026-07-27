#!/usr/bin/env python3
"""Prepare and validate the three-model whole-block ablation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from manage_remaining_block_pipeline import (
    ALL_BLOCKS,
    config_root,
    find_leaf,
    read_paths,
    result_payloads,
)


MODELS = ("wan_lora", "xssc", "physrvg")


def inspect_config(
    root: Path,
    allowed: set[Path],
) -> tuple[Path | None, list[dict[str, str]]]:
    try:
        leaf = find_leaf(root, allowed)
    except ValueError as error:
        return None, [{"error": "incomplete_case_set", "detail": str(error)}]

    failures: list[dict[str, str]] = []
    for source, payload in result_payloads(leaf, allowed).items():
        output_video = payload.get("output_video")
        if not isinstance(output_video, str) or not Path(output_video).expanduser().is_file():
            failures.append(
                {
                    "error": "missing_output_video",
                    "input_json": str(source),
                    "output_video": str(output_video),
                }
            )
    return leaf, failures


def all_configs(output_base: Path) -> list[tuple[str, int, Path]]:
    return [
        (model, block, config_root(output_base, model, "whole_block", block))
        for model in MODELS
        for block in ALL_BLOCKS
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_generation(args: argparse.Namespace) -> None:
    allowed = set(read_paths(args.input_list))
    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    rows: list[str] = []

    for model, block, root in all_configs(args.output_base):
        leaf, failures = inspect_config(root, allowed)
        record: dict[str, Any] = {
            "model": model,
            "mode": "whole_block",
            "block": block,
            "config_root": str(root),
        }
        if leaf is not None and not failures:
            complete.append({**record, "leaf_root": str(leaf)})
            continue

        task_id = f"whole-gen-{len(pending):04d}"
        pending.append({**record, "task_id": task_id, "failures": failures})
        rows.append(
            f"{task_id}\t{model}\twhole_block\t{block}\t{root}\n"
        )

    if len(complete) + len(pending) != 90:
        raise RuntimeError("Expected exactly 90 whole_block configurations")

    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.queue.write_text("".join(rows), encoding="utf-8")
    report = {
        "expected_configs": 90,
        "complete_configs": len(complete),
        "pending_configs": len(pending),
        "complete": complete,
        "pending": pending,
    }
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "expected_configs": 90,
                "complete_configs": len(complete),
                "pending_configs": len(pending),
            },
            indent=2,
        )
    )


def collect_roots(args: argparse.Namespace) -> None:
    allowed = set(read_paths(args.input_list))
    whole_roots: list[Path] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for model, block, root in all_configs(args.output_base):
        leaf, config_failures = inspect_config(root, allowed)
        if leaf is None or config_failures:
            failures.append(
                {
                    "model": model,
                    "mode": "whole_block",
                    "block": block,
                    "config_root": str(root),
                    "failures": config_failures,
                }
            )
            continue
        whole_roots.append(leaf)
        records.append(
            {
                "model": model,
                "mode": "whole_block",
                "block": block,
                "config_root": str(root),
                "leaf_root": str(leaf),
            }
        )

    baseline_roots: list[Path] = []
    for model in MODELS:
        root = config_root(args.output_base, model, "baseline", None)
        leaf, config_failures = inspect_config(root, allowed)
        if leaf is None or config_failures:
            failures.append(
                {
                    "model": model,
                    "mode": "baseline",
                    "block": None,
                    "config_root": str(root),
                    "failures": config_failures,
                }
            )
        else:
            baseline_roots.append(leaf)

    report = {
        "complete": not failures and len(whole_roots) == 90 and len(baseline_roots) == 3,
        "num_whole_block_roots": len(whole_roots),
        "num_baseline_roots": len(baseline_roots),
        "records": records,
        "failures": failures,
    }
    write_json(args.report, report)
    if not report["complete"]:
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    args.whole_roots.parent.mkdir(parents=True, exist_ok=True)
    args.whole_roots.write_text(
        "".join(f"{root}\n" for root in whole_roots),
        encoding="utf-8",
    )

    plot_roots: list[Path] = []
    seen: set[Path] = set()
    if args.merge_roots is not None and args.merge_roots.is_file():
        for root in read_paths(args.merge_roots):
            if root.is_dir() and root not in seen:
                plot_roots.append(root)
                seen.add(root)
    for root in (*baseline_roots, *whole_roots):
        if root not in seen:
            plot_roots.append(root)
            seen.add(root)
    args.plot_roots.write_text(
        "".join(f"{root}\n" for root in plot_roots),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "num_whole_block_roots": len(whole_roots),
                "num_baseline_roots": len(baseline_roots),
                "num_merged_plot_roots": len(plot_roots),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-generation")
    build.add_argument("--output-base", type=Path, required=True)
    build.add_argument("--input-list", type=Path, required=True)
    build.add_argument("--queue", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=build_generation)

    collect = subparsers.add_parser("collect-roots")
    collect.add_argument("--output-base", type=Path, required=True)
    collect.add_argument("--input-list", type=Path, required=True)
    collect.add_argument("--whole-roots", type=Path, required=True)
    collect.add_argument("--plot-roots", type=Path, required=True)
    collect.add_argument("--merge-roots", type=Path, default=None)
    collect.add_argument("--report", type=Path, required=True)
    collect.set_defaults(func=collect_roots)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
