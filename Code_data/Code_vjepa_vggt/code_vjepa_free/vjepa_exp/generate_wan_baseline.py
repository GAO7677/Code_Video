from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


WAN_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
WANTI2V_SCRIPT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_vjepa_vggt/AAAinfer/wanti2v.py"
)
CODE_VJEPA_VGGT_PYTHONPATH = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
WAN_REPO_PYTHONPATH = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt the existing vjepa_exp manifest into AAAinfer/wanti2v.py inputs and run it."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_runtime_inputs(manifest: dict, limit: int | None) -> tuple[Path, Path]:
    run_root = Path(manifest["output_root"]).expanduser().resolve() / manifest["run_name"]
    run_root.mkdir(parents=True, exist_ok=True)

    cases = manifest["cases"]
    if limit is not None:
        cases = cases[:limit]

    input_list_path = run_root / "wanti2v_input_list.txt"
    with input_list_path.open("w", encoding="utf-8") as list_handle:
        for case in cases:
            case_dir = Path(case["run_dir"]).expanduser().resolve()
            case_dir.mkdir(parents=True, exist_ok=True)
            runtime_json = case_dir / f"{case['case_id']}.runtime.json"
            payload = {
                "source_video": case["source_video"],
                "input_video": case["source_video"],
                "input_image": case["image_path"],
                "input_caption": case["prompt"],
                "seed": int(case["seed"]),
                "case_id": case["case_id"],
            }
            runtime_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            list_handle.write(str(runtime_json) + "\n")
    return run_root, input_list_path


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    run_root, input_list_path = build_runtime_inputs(manifest, args.limit)

    wan_args = manifest["wan_args"]
    cmd = [
        str(WAN_PYTHON),
        str(WANTI2V_SCRIPT),
        "--input-list",
        str(input_list_path),
        "--output-root",
        str(run_root),
        "--wan-root",
        str(Path(manifest["wan_root"]).expanduser().resolve()),
        "--backend",
        str(manifest["backend"]),
        "--size",
        str(wan_args["size"]),
        "--frame-num",
        str(wan_args["frame_num"]),
        "--sampling-steps",
        str(wan_args["sampling_steps"]),
        "--cfg-scale",
        str(wan_args["cfg_scale"]),
        "--fps",
        str(wan_args["fps"]),
        "--seed",
        "42",
        "--sample-shift",
        str(wan_args["sample_shift"]),
        "--sample-solver",
        str(wan_args["sample_solver"]),
        "--negative-prompt",
        str(wan_args["negative_prompt"]),
    ]
    if wan_args["offload_model"]:
        cmd.append("--offload-model")
    if wan_args["t5_cpu"]:
        cmd.append("--t5-cpu")
    if wan_args["convert_model_dtype"]:
        cmd.append("--convert-model-dtype")
    if args.force:
        cmd.append("--force")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    extra_paths = f"{CODE_VJEPA_VGGT_PYTHONPATH}:{WAN_REPO_PYTHONPATH}"
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{extra_paths}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = extra_paths
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
