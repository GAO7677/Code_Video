#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for pybullet_lora WMReward completion, then run the final Wan2.1 1.3B probe pipeline."
        )
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated"),
    )
    parser.add_argument(
        "--wan21-model-root",
        type=Path,
        default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers"),
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=DEFAULT_PYTHON,
    )
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("--extract-device", default="cuda:0")
    parser.add_argument("--extract-output-root", type=Path, default=None)
    parser.add_argument("--index-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--manifest-name", default="generated_probe_pairs_final")
    parser.add_argument("--capture-steps", default="10,25,40")
    parser.add_argument("--capture-layers", default="2,8,14,20,29")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--extract-presets",
        default="full_default,late_focus,late_dense",
        help=(
            "Comma-separated extraction presets to run. "
            "Known presets: full_default, late_focus, late_dense."
        ),
    )
    return parser.parse_args()


def preset_specs() -> dict[str, dict[str, str]]:
    return {
        "full_default": {
            "capture_steps": "10,25,40",
            "capture_layers": "2,8,14,20,29",
            "tag": "full_default",
        },
        "late_focus": {
            "capture_steps": "40",
            "capture_layers": "14,20,29",
            "tag": "late_focus",
        },
        "late_dense": {
            "capture_steps": "25,40",
            "capture_layers": "14,20,29",
            "tag": "late_dense",
        },
    }


def parse_preset_names(raw_value: str) -> list[str]:
    presets = preset_specs()
    names = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = [name for name in names if name not in presets]
    if unknown:
        raise ValueError(f"Unknown extraction presets: {unknown}. Known presets: {sorted(presets)}")
    return names


def read_registry_counts(registry_path: Path) -> tuple[int, int]:
    rows = list(csv.DictReader(registry_path.open(newline="", encoding="utf-8")))
    ready = sum(row.get("output_json_exists") == "True" and row.get("output_video_exists") == "True" for row in rows)
    ok = sum(row.get("wmreward_status") == "ok" for row in rows)
    return ready, ok


def wait_for_pybullet_completion(pipeline_root: Path, wait_seconds: int) -> None:
    registry_path = pipeline_root / "manifests" / "generation_registry_pybullet_lora.csv"
    while True:
        ready, ok = read_registry_counts(registry_path)
        print(f"[wait] pybullet_lora ready={ready} wmreward_ok={ok}", flush=True)
        if ready > 0 and ready == ok:
            return
        time.sleep(wait_seconds)


def run_subprocess(cmd: list[str]) -> None:
    print("[run]", " ".join(str(item) for item in cmd), flush=True)
    subprocess.run([str(item) for item in cmd], check=True)


def rewrite_all_registry(pipeline_root: Path) -> None:
    manifest_root = pipeline_root / "manifests"
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for model in ["base", "openvid_lora", "pybullet_lora"]:
        registry_path = manifest_root / f"generation_registry_{model}.csv"
        with registry_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(list(reader))
    if fieldnames is None:
        raise RuntimeError("No registry rows found to merge.")
    output_path = manifest_root / "generation_registry_all.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path, flush=True)


def run_single_preset(
    *,
    args: argparse.Namespace,
    manifest_csv: Path,
    preset_name: str,
    extract_output_root: Path,
    index_root: Path,
    results_root: Path,
) -> dict[str, str]:
    spec = preset_specs()[preset_name]
    preset_extract_root = extract_output_root / spec["tag"]
    preset_index_root = index_root / spec["tag"]
    preset_results_root = results_root / spec["tag"]

    extract_cmd = [
        args.python_bin,
        SCRIPT_ROOT.parent / "probe_wan21" / "extract_probe_features.py",
        "--model_root",
        args.wan21_model_root,
        "--manifest_csv",
        manifest_csv,
        "--output_root",
        preset_extract_root,
        "--device",
        args.extract_device,
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
    run_subprocess(extract_cmd)

    preset_index_root.mkdir(parents=True, exist_ok=True)
    build_index_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "build_probe_index.py",
        "--feature_root",
        preset_extract_root,
        "--output_csv",
        preset_index_root / "probe_index.csv",
        "--output_jsonl",
        preset_index_root / "probe_index.jsonl",
    ]
    run_subprocess(build_index_cmd)

    preset_results_root.mkdir(parents=True, exist_ok=True)
    ridge_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "train_ridge_probe.py",
        "--index_csv",
        preset_index_root / "probe_index.csv",
        "--output_root",
        preset_results_root / "ridge_single_features_mean",
        "--frame_reduce",
        "mean",
    ]
    run_subprocess(ridge_cmd)

    grid_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "probe_experiment_grid.py",
        "--index_csv",
        preset_index_root / "probe_index.csv",
        "--output_root",
        preset_results_root / "grid_search",
    ]
    run_subprocess(grid_cmd)

    return {
        "preset_name": preset_name,
        "extract_root": str(preset_extract_root),
        "index_csv": str(preset_index_root / "probe_index.csv"),
        "ridge_metrics_csv": str(preset_results_root / "ridge_single_features_mean" / "probe_metrics.csv"),
        "grid_metrics_csv": str(preset_results_root / "grid_search" / "probe_grid_metrics.csv"),
        "grid_summary_json": str(preset_results_root / "grid_search" / "probe_grid_summary.json"),
    }


def main() -> None:
    args = parse_args()
    pipeline_root = args.pipeline_root.expanduser().resolve()
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
    preset_names = parse_preset_names(args.extract_presets)

    wait_for_pybullet_completion(pipeline_root, args.wait_seconds)
    rewrite_all_registry(pipeline_root)

    build_manifest_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "build_generation_manifest.py",
        "--pipeline-root",
        pipeline_root,
        "--subset-name",
        args.manifest_name,
    ]
    run_subprocess(build_manifest_cmd)

    manifest_csv = pipeline_root / "manifests" / f"{args.manifest_name}.csv"
    manifest_summary_path = pipeline_root / "manifests" / f"{args.manifest_name}_summary.json"
    summary_payload = {
        "manifest_csv": str(manifest_csv),
        "manifest_summary_json": str(manifest_summary_path),
        "preset_names": preset_names,
        "preset_runs": [],
    }
    for preset_name in preset_names:
        summary_payload["preset_runs"].append(
            run_single_preset(
                args=args,
                manifest_csv=manifest_csv,
                preset_name=preset_name,
                extract_output_root=extract_output_root,
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

    summary_path = results_root / "final_probe_suite_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary_path, flush=True)


if __name__ == "__main__":
    sys.exit(main())
