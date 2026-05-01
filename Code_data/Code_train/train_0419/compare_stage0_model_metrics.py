#!/usr/bin/env python3
"""Aggregate 300-sample metrics across multiple stage0 benchmark model outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import batch_eval_lora as bel
import run_validation_vbench as rv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stage0 benchmark model metrics.")
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument(
        "--model_names",
        required=True,
        help=(
            "Comma-separated model specs. Each item can be either a model alias "
            "or alias=relative/path/from/benchmark_root."
        ),
    )
    parser.add_argument("--reference_model", default="base-ti2v-5b")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--output_root", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_name": payload["model_name"],
        "num_entries": payload["num_entries"],
        "num_success": payload["num_success"],
    }
    for metric_name, value in sorted(payload.get("aggregate", {}).items()):
        row[f"overall_{metric_name}"] = value
    for dataset_name, block in sorted(payload.get("per_dataset", {}).items()):
        dataset_key = bel.sanitize_filename(dataset_name).lower()
        row[f"{dataset_key}_num_entries"] = block.get("num_entries")
        row[f"{dataset_key}_num_success"] = block.get("num_success")
        for metric_name, value in sorted(block.get("aggregate", {}).items()):
            row[f"{dataset_key}_{metric_name}"] = value
    return row


def parse_model_specs(raw_value: str) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for item in raw_value.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" in token:
            model_name, rel_path = token.split("=", 1)
            model_name = model_name.strip()
            rel_path = rel_path.strip()
            if not model_name or not rel_path:
                raise ValueError(f"Invalid model spec: {token}")
            specs.append((model_name, Path(rel_path)))
        else:
            specs.append((token, Path("generated_videos") / token))
    return specs


def main() -> None:
    args = parse_args()
    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    runtime_root = args.benchmark_root / "tools" / "runtime"
    if not runtime_root.is_dir():
        runtime_root = args.benchmark_root / "runtime"
    model_specs = parse_model_specs(args.model_names)
    model_names = [model_name for model_name, _ in model_specs]
    if not model_specs:
        raise ValueError("model_names must not be empty")

    metric_suite = rv.ValidationMetricSuite()
    payloads: list[dict[str, Any]] = []
    for model_name, rel_model_dir in model_specs:
        payload = rv.build_run_payload(
            model_name=model_name,
            generated_dir=args.benchmark_root / rel_model_dir,
            runtime_root=runtime_root / model_name,
            height=args.height,
            width=args.width,
            metric_suite=metric_suite,
        )
        payloads.append(payload)

    by_model = {payload["model_name"]: payload for payload in payloads}
    bel.write_json(args.output_root / "metrics_by_model.json", by_model)
    write_csv(args.output_root / "metrics_by_model.csv", [flatten_payload(payload) for payload in payloads])

    reference = by_model.get(args.reference_model)
    if reference is not None:
        comparison_rows: list[dict[str, Any]] = []
        comparison_json: dict[str, Any] = {}
        for model_name in model_names:
            if model_name == args.reference_model:
                continue
            ft_payload = by_model[model_name]
            rows = rv.build_comparison_rows(reference, ft_payload)
            comparison_json[model_name] = {
                "reference_model": args.reference_model,
                "target_model": model_name,
                "rows": rows,
            }
            for row in rows:
                merged = {"reference_model": args.reference_model, "target_model": model_name}
                merged.update(row)
                comparison_rows.append(merged)
        bel.write_json(args.output_root / f"comparisons_vs_{args.reference_model}.json", comparison_json)
        write_csv(args.output_root / f"comparisons_vs_{args.reference_model}.csv", comparison_rows)

    print(args.output_root / "metrics_by_model.csv")


if __name__ == "__main__":
    main()
