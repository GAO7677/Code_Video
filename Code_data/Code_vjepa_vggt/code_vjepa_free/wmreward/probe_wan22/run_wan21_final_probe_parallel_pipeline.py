#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final Wan2.1 1.3B probe pipeline with manifest sharding across multiple GPUs, "
            "then build the final probe indices and training results."
        )
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated"),
    )
    parser.add_argument("--manifest-name", default="generated_probe_pairs_final")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument(
        "--wan21-model-root",
        type=Path,
        default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers"),
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--visible-gpus",
        default="6,7",
        help="Comma-separated list of physical GPU indices to shard extraction across.",
    )
    parser.add_argument(
        "--extract-presets",
        default="full_default,late_focus,late_dense",
        help="Comma-separated extraction presets to run.",
    )
    parser.add_argument("--extract-output-root", type=Path, default=None)
    parser.add_argument("--index-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=17)
    return parser.parse_args()


def preset_specs() -> dict[str, dict[str, str]]:
    return {
        "full_default": {
            "capture_steps": "10,25,40",
            "capture_layers": "2,8,14,20,29",
            "tag": "full_default",
            "reuse_from": "",
        },
        "late_focus": {
            "capture_steps": "40",
            "capture_layers": "14,20,29",
            "tag": "late_focus",
            "reuse_from": "full_default",
        },
        "late_dense": {
            "capture_steps": "25,40",
            "capture_layers": "14,20,29",
            "tag": "late_dense",
            "reuse_from": "full_default",
        },
    }


def parse_csv_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def ensure_manifest_csv(args: argparse.Namespace) -> Path:
    if args.manifest_csv is not None:
        return args.manifest_csv.expanduser().resolve()

    manifest_csv = args.pipeline_root / "manifests" / f"{args.manifest_name}.csv"
    if manifest_csv.is_file():
        return manifest_csv

    build_manifest_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "build_generation_manifest.py",
        "--pipeline-root",
        args.pipeline_root,
        "--subset-name",
        args.manifest_name,
    ]
    run_subprocess(build_manifest_cmd)
    return manifest_csv


def run_subprocess(cmd: list[Path | str], *, env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    pretty_cmd = " ".join(str(item) for item in cmd)
    print("[run]", pretty_cmd, flush=True)
    if log_path is None:
        subprocess.run([str(item) for item in cmd], check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run([str(item) for item in cmd], check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)


def split_manifest_by_pair_id(manifest_csv: Path, shard_count: int) -> list[Path]:
    shard_root = manifest_csv.parent / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []

    pair_rows: list[list[dict[str, str]]] = []
    current_pair_id: str | None = None
    current_rows: list[dict[str, str]] = []
    for row in rows:
        pair_id = row["pair_id"]
        if current_pair_id is None or pair_id == current_pair_id:
            current_rows.append(row)
        else:
            pair_rows.append(current_rows)
            current_rows = [row]
        current_pair_id = pair_id
    if current_rows:
        pair_rows.append(current_rows)

    shard_rows = [[] for _ in range(shard_count)]
    for idx, rows_for_pair in enumerate(pair_rows):
        shard_rows[idx % shard_count].extend(rows_for_pair)

    shard_paths: list[Path] = []
    for shard_idx, rows_for_shard in enumerate(shard_rows):
        shard_path = shard_root / f"{manifest_csv.stem}_shard{shard_idx}.csv"
        with shard_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_for_shard)
        shard_paths.append(shard_path)
        print(
            json.dumps(
                {
                    "shard_path": str(shard_path),
                    "rows": len(rows_for_shard),
                    "pair_count": len({row["pair_id"] for row in rows_for_shard}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return shard_paths


def count_manifest_rows(manifest_csv: Path) -> int:
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def count_probe_feature_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("probe_features.pt"))


def run_preset_extract_shards(
    *,
    args: argparse.Namespace,
    manifest_csv: Path,
    preset_name: str,
    visible_gpus: list[str],
    extract_output_root: Path,
    results_root: Path,
) -> Path:
    spec = preset_specs()[preset_name]
    preset_extract_root = extract_output_root / spec["tag"]
    shard_paths = split_manifest_by_pair_id(manifest_csv, len(visible_gpus))

    running: list[tuple[subprocess.Popen[str], Path, object]] = []
    for shard_idx, (gpu_idx, shard_csv) in enumerate(zip(visible_gpus, shard_paths)):
        log_path = results_root / "logs" / f"{spec['tag']}_shard{shard_idx}_gpu{gpu_idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin,
            SCRIPT_ROOT.parent / "probe_wan21" / "extract_probe_features.py",
            "--model_root",
            args.wan21_model_root,
            "--manifest_csv",
            shard_csv,
            "--output_root",
            preset_extract_root,
            "--device",
            "cuda:0",
            "--dtype",
            args.dtype,
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--guidance_scale",
            str(args.guidance_scale),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num_frames",
            str(args.num_frames),
            "--capture_steps",
            spec["capture_steps"],
            "--capture_layers",
            spec["capture_layers"],
            "--capture_branches",
            "cond",
            "--no_image_cond",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print(
            json.dumps(
                {
                    "preset": preset_name,
                    "gpu_idx": gpu_idx,
                    "shard_csv": str(shard_csv),
                    "output_root": str(preset_extract_root),
                    "log_path": str(log_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [str(item) for item in cmd],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append((proc, log_path, handle))

    failed_logs: list[Path] = []
    for proc, log_path, handle in running:
        try:
            return_code = proc.wait()
            if return_code != 0:
                failed_logs.append(log_path)
        finally:
            handle.close()

    if failed_logs:
        raise RuntimeError(f"Extraction failed for preset={preset_name}, logs={failed_logs}")

    return preset_extract_root


def resolve_existing_extract_root(
    *,
    preset_name: str,
    manifest_csv: Path,
    extract_output_root: Path,
) -> Path | None:
    spec = preset_specs()[preset_name]
    reuse_from = spec.get("reuse_from", "")
    if not reuse_from:
        preset_root = extract_output_root / spec["tag"]
        if count_probe_feature_files(preset_root) >= count_manifest_rows(manifest_csv):
            return preset_root
        return None
    reuse_root = extract_output_root / reuse_from
    return reuse_root if reuse_root.exists() and count_probe_feature_files(reuse_root) > 0 else None


def run_single_preset_postprocess(
    *,
    args: argparse.Namespace,
    preset_name: str,
    extract_root: Path,
    index_root: Path,
    results_root: Path,
) -> dict[str, str]:
    spec = preset_specs()[preset_name]
    preset_index_root = index_root / preset_name
    preset_results_root = results_root / preset_name
    preset_index_root.mkdir(parents=True, exist_ok=True)
    preset_results_root.mkdir(parents=True, exist_ok=True)

    build_index_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "build_probe_index.py",
        "--feature_root",
        extract_root,
        "--output_csv",
        preset_index_root / "probe_index.csv",
        "--output_jsonl",
        preset_index_root / "probe_index.jsonl",
    ]
    run_subprocess(build_index_cmd)

    ridge_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "train_ridge_probe.py",
        "--index_csv",
        preset_index_root / "probe_index.csv",
        "--output_root",
        preset_results_root / "ridge_single_features_mean",
        "--frame_reduce",
        "mean",
        "--allowed_steps",
        spec["capture_steps"],
        "--allowed_layers",
        spec["capture_layers"],
    ]
    run_subprocess(ridge_cmd)

    grid_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "probe_experiment_grid.py",
        "--index_csv",
        preset_index_root / "probe_index.csv",
        "--output_root",
        preset_results_root / "grid_search",
        "--allowed_steps",
        spec["capture_steps"],
        "--allowed_layers",
        spec["capture_layers"],
    ]
    run_subprocess(grid_cmd)

    return {
        "preset_name": preset_name,
        "extract_root": str(extract_root),
        "index_csv": str(preset_index_root / "probe_index.csv"),
        "ridge_metrics_csv": str(preset_results_root / "ridge_single_features_mean" / "probe_metrics.csv"),
        "grid_metrics_csv": str(preset_results_root / "grid_search" / "probe_grid_metrics.csv"),
        "grid_summary_json": str(preset_results_root / "grid_search" / "probe_grid_summary.json"),
    }


def main() -> None:
    args = parse_args()
    visible_gpus = parse_csv_list(args.visible_gpus)
    known_presets = preset_specs()
    preset_names = parse_csv_list(args.extract_presets)
    unknown_presets = [name for name in preset_names if name not in known_presets]
    if unknown_presets:
        raise ValueError(f"Unknown presets: {unknown_presets}; known={sorted(known_presets)}")

    extract_output_root = (
        args.extract_output_root.expanduser().resolve()
        if args.extract_output_root is not None
        else Path("/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/extracted/wan21_t2v_1p3b_final")
    )
    index_root = (
        args.index_root.expanduser().resolve()
        if args.index_root is not None
        else Path("/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/indices/wan21_t2v_1p3b_final")
    )
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root is not None
        else Path("/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/probe_results/wan21_t2v_1p3b_final")
    )
    results_root.mkdir(parents=True, exist_ok=True)

    manifest_csv = ensure_manifest_csv(args)
    summary_payload = {
        "manifest_csv": str(manifest_csv),
        "visible_gpus": visible_gpus,
        "preset_names": preset_names,
        "preset_runs": [],
    }

    for preset_name in preset_names:
        existing_extract_root = resolve_existing_extract_root(
            preset_name=preset_name,
            manifest_csv=manifest_csv,
            extract_output_root=extract_output_root,
        )
        if existing_extract_root is not None:
            reuse_from = preset_specs()[preset_name].get("reuse_from", "")
            print(
                json.dumps(
                    {
                        "preset": preset_name,
                        "reuse_extract_root": str(existing_extract_root),
                        "reason": "subset_of_full_default" if reuse_from else "existing_complete_extract_root",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            extract_root = existing_extract_root
        else:
            extract_root = run_preset_extract_shards(
                args=args,
                manifest_csv=manifest_csv,
                preset_name=preset_name,
                visible_gpus=visible_gpus,
                extract_output_root=extract_output_root,
                results_root=results_root,
            )
        summary_payload["preset_runs"].append(
            run_single_preset_postprocess(
                args=args,
                preset_name=preset_name,
                extract_root=extract_root,
                index_root=index_root,
                results_root=results_root,
            )
        )

    summarize_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "summarize_probe_suite_results.py",
        "--results_root",
        results_root,
    ]
    run_subprocess(summarize_cmd)

    summary_path = results_root / "parallel_probe_suite_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary_path, flush=True)


if __name__ == "__main__":
    sys.exit(main())
