#!/usr/bin/env python3
from __future__ import annotations

"""
Run command:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6,7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name-prefix train0705_current \
  --device cuda:0 \
  --vjepa-device cuda:1
"""

import argparse
import subprocess
from pathlib import Path

try:
    from .experiment_presets import TRAIN0705_CURRENT_MODES
except ImportError:
    from experiment_presets import TRAIN0705_CURRENT_MODES


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "code_vjepa_vggt"
    / "train0705"
    / "wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py"
)
DEFAULT_PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_OUTPUT_BASE = Path("/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current train0705 V-JEPA baseline/guided preset family over a JSON list."
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name-prefix", type=str, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode-ids", nargs="*", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _select_modes(mode_ids: list[str] | None):
    modes = TRAIN0705_CURRENT_MODES
    if mode_ids:
        wanted = set(mode_ids)
        modes = [mode for mode in modes if mode.mode_id in wanted]
        missing = wanted.difference(mode.mode_id for mode in modes)
        if missing:
            raise ValueError(f"Unknown mode_ids: {sorted(missing)}")
    return modes


def main() -> None:
    args = parse_args()
    modes = _select_modes(args.mode_ids)
    failures: list[str] = []

    for index, mode in enumerate(modes, start=1):
        model_name = f"{args.model_name_prefix}_{mode.mode_id}"
        output_root = args.output_base / model_name
        cmd = [
            str(args.python_bin),
            str(SCRIPT_PATH),
            "--weights-root",
            str(args.weights_root),
            "--input-json-list-path",
            str(args.input_json_list_path),
            "--model-name",
            model_name,
            "--output-root",
            str(output_root),
            "--device",
            str(args.device),
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
            "--vjepa-device",
            str(args.vjepa_device),
            "--vjepa-preset",
            str(mode.mode_id),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.force:
            cmd.append("--force")
        if args.overwrite:
            cmd.append("--overwrite")

        print(f"[{index}/{len(modes)}] [RUN] {model_name}: {mode.description}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(cmd), flush=True)
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append(model_name)
            print(f"[{index}/{len(modes)}] [FAIL] {model_name} returncode={result.returncode}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        print("FAILED_MODELS:", flush=True)
        for model_name in failures:
            print(model_name, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
