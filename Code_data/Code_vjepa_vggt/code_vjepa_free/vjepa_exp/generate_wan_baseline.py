from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


CODE_VJEPA_VGGT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")


def add_project_to_path() -> None:
    root = str(CODE_VJEPA_VGGT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Wan TI2V baseline videos using the AAAinfer/wanti2v.py runtime style."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    add_project_to_path()

    from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (
        WanTI2VArgs,
        build_run_manifest,
        build_wan_ti2v_pipeline,
        cleanup_pipeline,
        ensure_cuda_env,
        load_json,
        read_list_file,
        run_single_case,
        write_json,
    )

    ensure_cuda_env()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    input_list = Path(manifest["input_list"]).expanduser().resolve()
    output_root = Path(manifest["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    wan_args = manifest["wan_args"]
    runtime_args = WanTI2VArgs(
        input_list=input_list,
        output_root=output_root,
        wan_root=Path(manifest["wan_root"]).expanduser().resolve(),
        backend=str(manifest["backend"]),
        size=str(wan_args["size"]),
        frame_num=int(wan_args["frame_num"]),
        fps=int(wan_args["fps"]),
        seed=42,
        sample_solver=str(wan_args["sample_solver"]),
        sampling_steps=int(wan_args["sampling_steps"]),
        sample_shift=float(wan_args["sample_shift"]),
        cfg_scale=float(wan_args["cfg_scale"]),
        negative_prompt=str(wan_args["negative_prompt"]),
        offload_model=bool(wan_args["offload_model"]),
        t5_cpu=bool(wan_args["t5_cpu"]),
        convert_model_dtype=bool(wan_args["convert_model_dtype"]),
        force=bool(args.force),
    )

    json_paths = read_list_file(runtime_args.input_list)
    cases = manifest["cases"]
    if args.limit is not None:
        json_paths = json_paths[: args.limit]
        cases = cases[: args.limit]

    batch_manifest = build_run_manifest(runtime_args, json_paths)
    write_json(output_root / "batch_manifest.json", batch_manifest)

    case_map = {Path(case["input_json"]).expanduser().resolve(): case for case in cases}
    pipe = build_wan_ti2v_pipeline(runtime_args)
    written: list[str] = []
    try:
        for input_json_path in json_paths:
            payload = load_json(input_json_path)
            case = case_map[input_json_path]
            output_video = Path(case["output_video"]).expanduser().resolve()
            output_json = output_video.with_suffix(".json")

            if output_video.exists() and output_json.exists() and not runtime_args.force:
                print(f"[skip] {input_json_path.stem}")
                written.append(str(output_video))
                continue

            result, case_logs = run_single_case(
                pipe=pipe,
                args=runtime_args,
                input_json_path=input_json_path,
                payload=payload,
                firstframe_path=Path(payload["input_image"]).expanduser().resolve(),
                output_video=output_video,
            )
            result["seed"] = int(case["seed"])
            write_json(output_json, result)
            for log_line in case_logs:
                print(log_line)
            print(f"[done] {input_json_path.stem}")
            written.append(str(output_video))
    finally:
        cleanup_pipeline(pipe)

    print(json.dumps({"num_cases": len(written), "videos": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
