#!/usr/bin/env python3
from __future__ import annotations

"""
Run command:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6,7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_s15_local_sweep.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --device cuda:0 \
  --vjepa-device cuda:1 \
  --initialize-model-on-cpu
"""

import argparse
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().with_name("run_train0705_current_modes.py")
DEFAULT_PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_INPUT_JSON_LIST = Path(
    "/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation_cases.txt"
)
DEFAULT_OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round6_s15_local_sweep_overlap5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin wrapper for the round6 local sweep around target_w24_s15_ratio_003."
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--input-json-list-path", type=Path, default=DEFAULT_INPUT_JSON_LIST)
    parser.add_argument("--model-name-prefix", type=str, default="train0705_round6_s15_local")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vjepa-device", type=str, default="cuda:1")
    parser.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--diffsynth-root", type=Path, default=Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        str(args.python_bin),
        str(SCRIPT_PATH),
        "--weights-root",
        str(args.weights_root),
        "--input-json-list-path",
        str(args.input_json_list_path),
        "--model-name-prefix",
        str(args.model_name_prefix),
        "--output-base",
        str(args.output_base),
        "--device",
        str(args.device),
        "--vjepa-device",
        str(args.vjepa_device),
        "--wan-root",
        str(args.wan_root),
        "--diffsynth-root",
        str(args.diffsynth_root),
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
        "--sampling-mode",
        str(args.sampling_mode),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--cfg-scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        "--quality",
        str(args.quality),
        "--mode-group",
        "s15_local_sweep",
    ]
    if args.initialize_model_on_cpu:
        cmd.append("--initialize-model-on-cpu")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.force:
        cmd.append("--force")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.dry_run:
        cmd.append("--dry-run")
        print(subprocess.list2cmdline(cmd), flush=True)
        return
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
