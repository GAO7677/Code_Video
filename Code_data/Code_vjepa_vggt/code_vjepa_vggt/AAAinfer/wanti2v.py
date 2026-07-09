"""
Batch Wan2.2 TI2V inference over a txt file that lists one input json per line.

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=0 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wanti2v.py \
    --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --model-name testsssss \
    --size 704*1280 \
    --frame-num 49 \
    --sampling-steps 3 \
    --cfg-scale 5.0 \
    --fps 30 \
    --seed 42 \
    --offload-model \
    --negative-prompt   "" 

自动输出到：
- /data/gaoya/AAA_test_video/0623/test/v2v/basemodel/wan2p2_ti2v5B_negcap_null_frame49
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OFFICIAL_WAN_ROOT,
    WanTI2VArgs,
    build_run_manifest,
    build_wan_ti2v_pipeline,
    cleanup_pipeline,
    ensure_cuda_env,
    ensure_firstframe_image,
    ensure_str_field,
    load_json,
    read_list_file,
    resolve_default_cfg_scale,
    resolve_default_frame_num,
    resolve_default_sample_shift,
    resolve_default_sampling_steps,
    run_single_case,
    write_json,
)

DEFAULT_OUTPUT_BASE_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v/basemodel")
DEFAULT_MODEL_NAME = "wan2p2_ti2v5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run official Wan2.2 TI2V on input jsons listed in a txt file."
    )
    parser.add_argument(
        "--input-list",
        default="/data/gaoya/AAA_test_video/0623/testjsons/test_100.txt",
        help="Text file containing one input json path per line.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for mp4/json files.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--wan-root",
        default=str(DEFAULT_OFFICIAL_WAN_ROOT),
        help="Official Wan2.2 TI2V checkpoint directory.",
    )
    parser.add_argument("--backend", default="legacy", choices=["official", "legacy"])
    parser.add_argument(
        "--size",
        default="704*1280",
        choices=["704*1280", "1280*704", "512*896"],
        help=(
            "Output size in H*W format. "
            "Wan TI2V officially exposes 704*1280 / 1280*704; "
            "512*896 is enabled here for aligned local comparisons."
        ),
    )
    parser.add_argument("--frame-num", type=int, default=25)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default=None, help="Override the default negative prompt. Use '' for empty.")
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--convert-model-dtype", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    ensure_cuda_env()
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root=DEFAULT_OUTPUT_BASE_ROOT,
        model_name=model_name,
    )

    args = WanTI2VArgs(
        input_list=Path(cli_args.input_list).expanduser().resolve(),
        output_root=output_root,
        model_name=model_name,
        wan_root=Path(cli_args.wan_root).expanduser().resolve(),
        backend=str(cli_args.backend),
        size=str(cli_args.size),
        frame_num=resolve_default_frame_num(cli_args.frame_num),
        fps=int(cli_args.fps),
        seed=int(cli_args.seed),
        sample_solver=str(cli_args.sample_solver),
        sampling_steps=resolve_default_sampling_steps(cli_args.sampling_steps),
        sample_shift=resolve_default_sample_shift(cli_args.sample_shift),
        cfg_scale=resolve_default_cfg_scale(cli_args.cfg_scale),
        negative_prompt=(
            cli_args.negative_prompt
            if cli_args.negative_prompt is not None
            else DEFAULT_NEGATIVE_PROMPT
        ),
        offload_model=bool(cli_args.offload_model),
        t5_cpu=bool(cli_args.t5_cpu),
        convert_model_dtype=bool(cli_args.convert_model_dtype),
        force=bool(cli_args.force),
    )

    json_paths = read_list_file(args.input_list)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(args, json_paths)
    with (args.output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    prepared_cases: list[tuple[Path, dict[str, object], Path]] = []
    for input_json_path in json_paths:
        payload = load_json(input_json_path)
        try:
            ensure_str_field(payload, "input_video", input_json_path)
            ensure_str_field(payload, "input_caption", input_json_path)
            payload, firstframe_path = ensure_firstframe_image(input_json_path, payload)
        except Exception as exc:
            print(f"[skip] {input_json_path.stem}: {exc}")
            continue
        prepared_cases.append((input_json_path, payload, firstframe_path))

    pipe = build_wan_ti2v_pipeline(args)
    step_dir = args.output_root
    step_result_json = step_dir / "result.json"
    if step_result_json.exists():
        step_result_json.unlink()

    try:
        for input_json_path, payload, firstframe_path in prepared_cases:
            sample_stem = input_json_path.stem
            output_video = step_dir / f"{sample_stem}.mp4"
            output_json = step_dir / f"{sample_stem}.json"

            if output_video.exists() and output_json.exists() and not args.force:
                print(f"[skip] {sample_stem}")
                continue

            try:
                result, case_logs = run_single_case(
                    pipe=pipe,
                    args=args,
                    input_json_path=input_json_path,
                    payload=payload,
                    firstframe_path=firstframe_path,
                    output_video=output_video,
                )
            except Exception as exc:
                print(f"[error] {sample_stem}: {exc}")
                continue

            write_json(output_json, result)
            for log_line in case_logs:
                print(log_line)
            print(f"[done] {sample_stem}")
    finally:
        cleanup_pipeline(pipe)


if __name__ == "__main__":
    main()
