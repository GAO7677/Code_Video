#!/usr/bin/env python3
"""Run VBench-short on existing stage0 benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import run_validation_vbench as rv
from benchlib.config import load_config
from benchlib.manifest import load_manifest
from benchlib.staging import stage_custom_vbench_dataset
import torch


MODEL_SPECS = [
    ("base-ti2v-5b", "output/wan2_2_5B_baseline_TI2V"),
    ("step-008000", "output/wan2.25B_lora_sample300_full49/step-008000"),
    ("step-010000", "output/wan2.25B_lora_sample300_full49/step-010000"),
    ("wan_pure_ti2v_5b", "output/Wan2_2_5B_pure_TI2V"),
    ("vace_ti2v_firstframe", "output/VACE_1_3B_TI2V"),
    ("vace_v2v_ctx01f", "output/VACE_1_3B_V2V/context_01f"),
    ("vace_v2v_ctx02f", "output/VACE_1_3B_V2V/context_02f"),
    ("vace_v2v_ctx04f", "output/VACE_1_3B_V2V/context_04f"),
    ("vace_v2v_ctx08f", "output/VACE_1_3B_V2V/context_08f"),
    ("vace_v2v_ctx08f_nullcaption", "output/VACE_1_3B_V2V_nullcaption/context_08f"),
    ("vace_v2v_fullctx_fullvideo_nullcaption", "output/VACE_1_3B_V2V_nullcaption/context_fullctx_fullvideo"),
]

VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
    "overall_consistency",
    "temporal_style",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VBench-short on existing stage0 outputs.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result/model_metrics_vbench_short"),
    )
    parser.add_argument(
        "--runtime_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/tools/runtime"),
    )
    parser.add_argument(
        "--vbench_config_path",
        type=Path,
        default=Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/vbench_paths.yaml"),
    )
    parser.add_argument(
        "--model_name",
        action="append",
        default=[],
        help="Optional model_name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cuda:3.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def round6(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_vbench(config: Any) -> None:
    repo_root = Path(config.paths.vbench_repo_root)
    vbench2_root = repo_root / "VBench-2.0"
    for candidate in [repo_root, vbench2_root]:
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    os.environ["VBENCH_CACHE_DIR"] = config.paths.vbench_cache_dir
    os.environ["VBENCH2_CACHE_DIR"] = config.paths.vbench2_cache_dir
    for key, value in config.extra_env.items():
        os.environ[key] = value


def override_runtime_device(config: Any, device: str | None) -> None:
    if device:
        config.runtime.device = str(device)


def get_vbench_device(config: Any) -> torch.device:
    want = config.runtime.device
    if want.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(want)


def build_manifest_from_current_outputs(
    *,
    model_name: str,
    generated_dir: Path,
    runtime_root: Path,
    manifest_path: Path,
) -> tuple[Path, int]:
    entries = rv.load_entries_for_compare(model_name, generated_dir, runtime_root)
    samples: list[dict[str, str]] = []
    for entry in entries:
        if entry.get("status") not in {"generated", "skipped_existing"}:
            continue
        paths = entry.get("paths", {})
        if not isinstance(paths, dict):
            continue
        output_path = paths.get("output_video_path") or paths.get("output_path")
        caption = entry.get("caption")
        sample_id = entry.get("sample_id")
        if not output_path or sample_id is None:
            continue
        output_file = Path(str(output_path))
        if not output_file.is_file():
            continue
        samples.append(
            {
                "sample_id": str(sample_id),
                "prompt": str(caption or ""),
                "video_path": str(output_file),
            }
        )
    if not samples:
        raise ValueError(f"No valid current output videos found for model={model_name} under {generated_dir}")
    write_json(manifest_path, samples)
    return manifest_path, len(samples)


def collect_per_video_vbench_metrics(eval_payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    metrics_by_video: dict[str, dict[str, float]] = {}
    for dimension in VBENCH_DIMENSIONS:
        raw_block = eval_payload.get(dimension)
        if not isinstance(raw_block, list) or len(raw_block) < 2:
            continue
        per_video = raw_block[1]
        if not isinstance(per_video, list):
            continue
        for item in per_video:
            if not isinstance(item, dict):
                continue
            video_path = item.get("video_path")
            video_result = item.get("video_results")
            if not isinstance(video_path, str) or not isinstance(video_result, (int, float)):
                continue
            metrics_by_video.setdefault(str(Path(video_path).resolve()), {})[dimension] = round6(video_result)
    return metrics_by_video


def backfill_sidecars_with_vbench_metrics(
    *,
    generated_dir: Path,
    metrics_by_video: dict[str, dict[str, float]],
    eval_json: Path,
) -> dict[str, int]:
    updated = 0
    missing = 0
    for sidecar_path in sorted(generated_dir.glob("*.json")):
        payload = load_json(sidecar_path)
        paths = payload.get("paths", {})
        if not isinstance(paths, dict):
            missing += 1
            continue
        output_video_path = paths.get("output_video_path") or paths.get("output_path")
        if not isinstance(output_video_path, str):
            missing += 1
            continue
        per_video_metrics = metrics_by_video.get(str(Path(output_video_path).resolve()))
        if per_video_metrics is None:
            missing += 1
            continue

        existing = payload.get("vbench_metrics", {})
        if not isinstance(existing, dict):
            existing = {}
        metric_values = {
            key: value
            for key, value in existing.items()
            if isinstance(value, (int, float))
        }
        metric_values.update(per_video_metrics)
        payload.pop("vbench_short_metrics", None)
        payload["vbench_metrics"] = {
            **metric_values,
            "dimensions": sorted(metric_values.keys()),
            "evaluation_suite": "vbench_short",
            "source_eval_json": str(eval_json),
        }
        write_json(sidecar_path, payload)
        updated += 1
    return {
        "num_sidecars_updated": updated,
        "num_sidecars_missing_vbench_metrics": missing,
    }


def write_model_summary(
    *,
    summary_path: Path,
    model_name: str,
    generated_dir: Path,
    model_runtime_root: Path,
    manifest_path: Path,
    eval_json: Path,
    num_samples: int,
    completed_dimensions: list[str],
    rounded_metrics: dict[str, Any],
    backfill_stats: dict[str, int],
) -> dict[str, Any]:
    aggregate_mean = (
        round6(sum(float(value) for value in rounded_metrics.values()) / len(rounded_metrics))
        if rounded_metrics
        else 0.0
    )
    payload = {
        "model_name": model_name,
        "generated_dir": str(generated_dir),
        "runtime_root": str(model_runtime_root),
        "manifest_path": str(manifest_path),
        "vbench_short_eval_json": str(eval_json),
        "num_samples": num_samples,
        "completed_dimensions": completed_dimensions,
        "num_completed_dimensions": len(completed_dimensions),
        "vbench_short_metrics": rounded_metrics,
        "vbench_aggregate_mean": aggregate_mean,
        **backfill_stats,
    }
    write_json(summary_path, payload)
    return payload


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runtime_root = args.runtime_root.expanduser().resolve()
    selected_models = {str(name) for name in args.model_name}
    bench_config = load_config(str(args.vbench_config_path.expanduser().resolve()))
    override_runtime_device(bench_config, args.device)
    bootstrap_vbench(bench_config)
    from vbench import VBench
    from vbench.utils import init_submodules

    rows: list[dict[str, Any]] = []
    by_model: dict[str, Any] = {}

    for model_name, rel_dir in MODEL_SPECS:
        if selected_models and model_name not in selected_models:
            continue

        generated_dir = benchmark_root / rel_dir
        model_runtime_root = runtime_root / model_name
        print(
            json.dumps(
                {
                    "event": "start_model",
                    "model_name": model_name,
                    "generated_dir": str(generated_dir),
                    "runtime_root": str(model_runtime_root),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        manifest_path, num_samples = build_manifest_from_current_outputs(
            model_name=model_name,
            generated_dir=generated_dir,
            runtime_root=model_runtime_root,
            manifest_path=output_root / "manifests" / f"{model_name}.json",
        )
        samples = load_manifest(str(manifest_path))
        model_output_dir = output_root / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        eval_json = model_output_dir / f"{model_name}_eval_results.json"
        cumulative_eval_payload: dict[str, Any] = load_json(eval_json) if eval_json.is_file() else {}
        staged = stage_custom_vbench_dataset(
            samples=samples,
            staging_root=str(model_output_dir / "staging"),
            use_symlink=bench_config.runtime.use_symlink,
            with_images=False,
        )
        bench = VBench(
            device=get_vbench_device(bench_config),
            full_info_dir=str(Path(bench_config.paths.vbench_repo_root) / "vbench" / "VBench_full_info.json"),
            output_path=str(model_output_dir),
        )
        summary_path = model_output_dir / "summary.json"
        for dimension in VBENCH_DIMENSIONS:
            if dimension in cumulative_eval_payload:
                rounded_metrics = {key: round6(value) for key, value in rv.parse_vbench_eval(eval_json).items()}
                per_video_metrics = collect_per_video_vbench_metrics(cumulative_eval_payload)
                backfill_stats = backfill_sidecars_with_vbench_metrics(
                    generated_dir=generated_dir,
                    metrics_by_video=per_video_metrics,
                    eval_json=eval_json,
                )
                payload = write_model_summary(
                    summary_path=summary_path,
                    model_name=model_name,
                    generated_dir=generated_dir,
                    model_runtime_root=model_runtime_root,
                    manifest_path=manifest_path,
                    eval_json=eval_json,
                    num_samples=num_samples,
                    completed_dimensions=sorted(cumulative_eval_payload.keys()),
                    rounded_metrics=rounded_metrics,
                    backfill_stats=backfill_stats,
                )
                print(
                    json.dumps(
                        {
                            "event": "skip_completed_dimension",
                            "model_name": model_name,
                            "dimension": dimension,
                            "num_completed_dimensions": len(cumulative_eval_payload),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            print(
                json.dumps(
                    {
                        "event": "start_dimension",
                        "model_name": model_name,
                        "dimension": dimension,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            bench.evaluate(
                videos_path=staged.video_dir,
                name=model_name,
                prompt_list=staged.prompt_map,
                dimension_list=[dimension],
                local=bench_config.runtime.load_ckpt_from_local,
                read_frame=bench_config.runtime.read_frame,
                mode="custom_input",
                imaging_quality_preprocessing_mode=bench_config.runtime.imaging_quality_preprocessing_mode,
            )
            single_dimension_payload = load_json(eval_json)
            cumulative_eval_payload.update(single_dimension_payload)
            write_json(eval_json, cumulative_eval_payload)

            rounded_metrics = {key: round6(value) for key, value in rv.parse_vbench_eval(eval_json).items()}
            per_video_metrics = collect_per_video_vbench_metrics(cumulative_eval_payload)
            backfill_stats = backfill_sidecars_with_vbench_metrics(
                generated_dir=generated_dir,
                metrics_by_video=per_video_metrics,
                eval_json=eval_json,
            )
            payload = write_model_summary(
                summary_path=summary_path,
                model_name=model_name,
                generated_dir=generated_dir,
                model_runtime_root=model_runtime_root,
                manifest_path=manifest_path,
                eval_json=eval_json,
                num_samples=num_samples,
                completed_dimensions=sorted(cumulative_eval_payload.keys()),
                rounded_metrics=rounded_metrics,
                backfill_stats=backfill_stats,
            )
            print(
                json.dumps(
                    {
                        "event": "done_dimension",
                        "model_name": model_name,
                        "dimension": dimension,
                        "num_completed_dimensions": len(cumulative_eval_payload),
                        **backfill_stats,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if bench_config.runtime.cleanup_staging:
            shutil.rmtree(staged.root_dir, ignore_errors=True)
        by_model[model_name] = payload

        row = {
            "model_name": model_name,
            "num_samples": num_samples,
            "vbench_aggregate_mean": payload["vbench_aggregate_mean"],
        }
        row.update(payload["vbench_short_metrics"])
        rows.append(row)

        print(json.dumps({"event": "done_model", **payload}, ensure_ascii=False), flush=True)

    write_json(output_root / "metrics_by_model.json", by_model)
    write_csv(output_root / "metrics_by_model.csv", rows)
    print(output_root / "metrics_by_model.csv")


if __name__ == "__main__":
    main()
