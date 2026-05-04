#!/usr/bin/env python3
"""Backfill per-sample future metrics into stage0 output sidecars."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import run_validation_vbench as rv


MODEL_SPECS = [
    {
        "model_name": "base-ti2v-5b",
        "generated_dir": "output/wan2_2_5B_baseline_TI2V",
        "runtime_dir": "tools/runtime/base-ti2v-5b",
        "height": 384,
        "width": 672,
    },
    {
        "model_name": "step-008000",
        "generated_dir": "output/wan2.25B_lora_sample300_full49/step-008000",
        "runtime_dir": "tools/runtime/step-008000",
        "height": 384,
        "width": 672,
    },
    {
        "model_name": "step-010000",
        "generated_dir": "output/wan2.25B_lora_sample300_full49/step-010000",
        "runtime_dir": "tools/runtime/step-010000",
        "height": 384,
        "width": 672,
    },
    {
        "model_name": "wan_pure_ti2v_5b",
        "generated_dir": "output/Wan2_2_5B_pure_TI2V",
        "runtime_dir": "tools/runtime/wan_pure_ti2v_5b",
        "height": 384,
        "width": 672,
    },
    {
        "model_name": "vace_ti2v_firstframe",
        "generated_dir": "output/VACE_1_3B_TI2V",
        "runtime_dir": "tools/runtime/vace_ti2v_firstframe",
        "height": 544,
        "width": 720,
    },
    {
        "model_name": "vace_v2v_ctx01f",
        "generated_dir": "output/VACE_1_3B_V2V/context_01f",
        "runtime_dir": "tools/runtime/vace_v2v_ctx01f",
        "height": 544,
        "width": 720,
    },
    {
        "model_name": "vace_v2v_ctx02f",
        "generated_dir": "output/VACE_1_3B_V2V/context_02f",
        "runtime_dir": "tools/runtime/vace_v2v_ctx02f",
        "height": 544,
        "width": 720,
    },
    {
        "model_name": "vace_v2v_ctx04f",
        "generated_dir": "output/VACE_1_3B_V2V/context_04f",
        "runtime_dir": "tools/runtime/vace_v2v_ctx04f",
        "height": 544,
        "width": 720,
    },
    {
        "model_name": "vace_v2v_ctx08f",
        "generated_dir": "output/VACE_1_3B_V2V/context_08f",
        "runtime_dir": "tools/runtime/vace_v2v_ctx08f",
        "height": 544,
        "width": 720,
    },
]

METRIC_KEYS = [
    "future_pair_count",
    "future_psnr",
    "future_ssim",
    "future_lpips",
    "future_dino",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill per-sample future metrics into stage0 sidecars.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V"),
    )
    parser.add_argument(
        "--result_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result/per_sample_future_metrics"),
    )
    return parser.parse_args()


def round4(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_map(per_sample: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    metric_map: dict[tuple[str, str], dict[str, Any]] = {}
    for item in per_sample:
        dataset = str(item.get("dataset") or "")
        sample_id = str(item.get("sample_id") or "")
        metric_map[(dataset, sample_id)] = item
    return metric_map


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    metric_suite = rv.ValidationMetricSuite()

    for spec in MODEL_SPECS:
        model_name = str(spec["model_name"])
        generated_dir = benchmark_root / str(spec["generated_dir"])
        runtime_root = benchmark_root / str(spec["runtime_dir"])
        height = int(spec["height"])
        width = int(spec["width"])

        entries = rv.load_entries_for_compare(model_name, generated_dir, runtime_root)
        metric_payload = rv.compute_future_gt_metrics(
            entries,
            height=height,
            width=width,
            metric_suite=metric_suite,
        )
        per_sample = metric_payload.get("per_sample", [])
        metric_map = build_metric_map(per_sample)

        csv_rows: list[dict[str, Any]] = []
        updated = 0
        missing = 0
        for sidecar_path in sorted(generated_dir.glob("*.json")):
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            key = (str(payload.get("dataset") or ""), str(payload.get("sample_id") or ""))
            metric_item = metric_map.get(key)
            if metric_item is None:
                missing += 1
                continue

            rounded_metrics = {metric_key: round4(metric_item.get(metric_key)) for metric_key in METRIC_KEYS}
            payload["future_metrics"] = {
                **rounded_metrics,
                "evaluation_scope": "future_only",
                "evaluation_resolution": {
                    "height": height,
                    "width": width,
                },
            }
            write_json(sidecar_path, payload)
            updated += 1

            csv_rows.append(
                {
                    "model_name": model_name,
                    "dataset": key[0],
                    "sample_id": key[1],
                    **rounded_metrics,
                    "evaluation_height": height,
                    "evaluation_width": width,
                    "sidecar_path": str(sidecar_path),
                }
            )

        write_csv(result_root / f"{model_name}.csv", csv_rows)
        summary_payload = {
            "model_name": model_name,
            "generated_dir": str(generated_dir),
            "runtime_root": str(runtime_root),
            "evaluation_height": height,
            "evaluation_width": width,
            "num_per_sample_metrics": len(per_sample),
            "num_sidecars_updated": updated,
            "num_sidecars_missing_metrics": missing,
            "aggregate": {metric_key: round4(metric_payload.get("aggregate", {}).get(metric_key)) for metric_key in METRIC_KEYS},
        }
        write_json(result_root / f"{model_name}_summary.json", summary_payload)
        print(json.dumps(summary_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
