#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from pipeline_common import (
    DEFAULT_INPUT_JSON_DIR,
    DEFAULT_PIPELINE_ROOT,
    DEFAULT_SMOKE_PIPELINE_ROOT,
    MODEL_SPECS,
    build_normalized_input_json,
    discover_input_jsons,
    generation_registry_fieldnames,
    parse_model_keys,
    pipeline_run_summary,
    resolve_python_bin,
    scan_generated_records,
    write_csv_rows,
    write_json,
    write_list_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate videos and same-name result JSONs for probe_wan22 by reusing external Wan inference scripts."
        )
    )
    parser.add_argument("--input-json-dir", type=Path, default=DEFAULT_INPUT_JSON_DIR)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument(
        "--smoke-name",
        default=None,
        help="If set, override pipeline-root to tmp/smoke/pipeline_runs/<smoke-name>.",
    )
    parser.add_argument("--models", default="base,openvid_lora,pybullet_lora")
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--frame-num", type=int, default=25)
    parser.add_argument("--size", default="704*1280", choices=["704*1280", "1280*704"])
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override for each subprocess. Avoid gpu4.",
    )
    return parser.parse_args()


def build_generation_command(
    *,
    python_bin: Path,
    model_key: str,
    input_list_path: Path,
    model_output_root: Path,
    args: argparse.Namespace,
) -> list[str]:
    spec = MODEL_SPECS[model_key]
    if spec.kind == "ti2v":
        cmd = [
            str(python_bin),
            str(spec.script_path),
            "--input-list",
            str(input_list_path),
            "--output-root",
            str(model_output_root),
            "--model-name",
            spec.model_name,
            "--wan-root",
            str(spec.wan_root),
            "--size",
            args.size,
            "--frame-num",
            str(args.frame_num),
            "--sampling-steps",
            str(args.sampling_steps),
            "--cfg-scale",
            str(args.cfg_scale),
            "--fps",
            str(args.fps),
            "--seed",
            str(args.seed),
            "--negative-prompt",
            args.negative_prompt,
            "--offload-model",
        ]
        if args.force:
            cmd.append("--force")
        return cmd

    cmd = [
        str(python_bin),
        str(spec.script_path),
        "--weights-root",
        str(spec.weights_root),
        "--input-json-list-path",
        str(input_list_path),
        "--model-name",
        spec.model_name,
        "--wan-root",
        str(spec.wan_root),
        "--output-root",
        str(model_output_root),
        "--runtime-root",
        str(model_output_root.parent / f"{model_output_root.name}_runtime"),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num-frames",
        str(args.num_frames),
        "--context-frames",
        str(args.context_frames),
        "--fps",
        str(args.fps),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--cfg-scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        "--negative-prompt",
        args.negative_prompt,
        "--conditioning-mode",
        "input_image_only",
    ]
    if args.force:
        cmd.append("--overwrite")
    return cmd


def run_subprocess(cmd: list[str], *, cuda_visible_devices: str | None) -> None:
    env = None
    if cuda_visible_devices is not None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    input_json_dir = args.input_json_dir.expanduser().resolve()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    if args.smoke_name:
        pipeline_root = (DEFAULT_SMOKE_PIPELINE_ROOT / args.smoke_name).resolve()
    python_bin = resolve_python_bin(args.python_bin)
    selected_model_keys = parse_model_keys(args.models)
    source_input_json_paths = discover_input_jsons(input_json_dir, limit=args.limit)
    if not source_input_json_paths:
        raise FileNotFoundError(f"No input JSONs found under: {input_json_dir}")

    pipeline_root.mkdir(parents=True, exist_ok=True)
    normalized_input_root = pipeline_root / "manifests" / "normalized_inputs"
    input_json_paths = [
        build_normalized_input_json(source_json_path=path, normalized_root=normalized_input_root)
        for path in source_input_json_paths
    ]
    input_list_path = pipeline_root / "manifests" / "input_jsons.txt"
    write_list_file(input_list_path, input_json_paths)

    summary = pipeline_run_summary(
        input_json_dir=input_json_dir,
        pipeline_root=pipeline_root,
        selected_model_keys=selected_model_keys,
        input_json_count=len(input_json_paths),
    )
    write_json(pipeline_root / "manifests" / "generation_run_config.json", summary)

    all_rows: list[dict[str, object]] = []
    for model_key in selected_model_keys:
        spec = MODEL_SPECS[model_key]
        model_output_root = (pipeline_root / spec.output_subdir).resolve()
        model_output_root.mkdir(parents=True, exist_ok=True)
        cmd = build_generation_command(
            python_bin=python_bin,
            model_key=model_key,
            input_list_path=input_list_path,
            model_output_root=model_output_root,
            args=args,
        )
        print(f"[generate] model={model_key} output={model_output_root}")
        print(" ".join(cmd), flush=True)
        run_subprocess(cmd, cuda_visible_devices=args.cuda_visible_devices)
        rows = scan_generated_records(
            model_spec=spec,
            input_json_paths=input_json_paths,
            pipeline_root=pipeline_root,
        )
        write_csv_rows(
            pipeline_root / "manifests" / f"generation_registry_{model_key}.csv",
            rows,
            generation_registry_fieldnames(),
        )
        all_rows.extend(rows)

    output_csv = pipeline_root / "manifests" / "generation_registry_all.csv"
    write_csv_rows(output_csv, all_rows, generation_registry_fieldnames())
    print(output_csv)


if __name__ == "__main__":
    main()
