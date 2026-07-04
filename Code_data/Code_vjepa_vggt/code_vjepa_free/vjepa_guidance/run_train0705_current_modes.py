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

Run current guard-ablation group:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6,7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation_cases.txt \
  --model-name-prefix train0705_round3_guard \
  --output-base /data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation \
  --device cuda:0 \
  --vjepa-device cuda:1 \
  --mode-group guard_ablation \
  --initialize-model-on-cpu
"""

import argparse
import subprocess
from pathlib import Path

try:
    from .experiment_presets import (
        TRAIN0705_CURRENT_MODES,
        TRAIN0705_MODE_GROUPS,
        resolve_train0705_mode_group,
        resolve_train0705_preset,
    )
except ImportError:
    from experiment_presets import (
        TRAIN0705_CURRENT_MODES,
        TRAIN0705_MODE_GROUPS,
        resolve_train0705_mode_group,
        resolve_train0705_preset,
    )


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
    parser.add_argument("--weights-root", type=Path, default=None)
    parser.add_argument("--input-json-list-path", type=Path, default=None)
    parser.add_argument("--model-name-prefix", type=str, default=None)
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
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode-ids", nargs="*", default=None)
    parser.add_argument("--mode-group", choices=sorted(TRAIN0705_MODE_GROUPS), default=None)
    parser.add_argument("--list-modes", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _select_modes(mode_ids: list[str] | None, mode_group: str | None):
    if mode_ids:
        modes: list = []
        seen: set[str] = set()
        for mode_id in mode_ids:
            preset = resolve_train0705_preset(str(mode_id))
            if preset.mode_id in seen:
                continue
            seen.add(preset.mode_id)
            modes.append(preset)
        return modes

    if mode_group:
        return list(resolve_train0705_mode_group(mode_group))

    if not mode_ids and not mode_group:
        return TRAIN0705_CURRENT_MODES
    return TRAIN0705_CURRENT_MODES


def _print_available_modes() -> None:
    print("TRAIN0705 mode groups:", flush=True)
    for group_name, mode_ids in sorted(TRAIN0705_MODE_GROUPS.items()):
        print(f"  - {group_name}: {', '.join(mode_ids)}", flush=True)
    print("TRAIN0705 canonical modes:", flush=True)
    seen: set[str] = set()
    for preset in resolve_train0705_mode_group("all"):
        if preset.mode_id in seen:
            continue
        seen.add(preset.mode_id)
        aliases = f" aliases={list(preset.aliases)}" if preset.aliases else ""
        print(f"  - {preset.mode_id}: {preset.description}{aliases}", flush=True)


def main() -> None:
    args = parse_args()
    if args.list_modes:
        _print_available_modes()
        return
    missing = [
        name
        for name in ("weights_root", "input_json_list_path", "model_name_prefix")
        if getattr(args, name) in (None, "")
    ]
    if missing:
        readable = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise SystemExit(f"Missing required arguments: {readable}")
    modes = _select_modes(args.mode_ids, args.mode_group)
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
        if args.initialize_model_on_cpu:
            cmd.append("--initialize-model-on-cpu")
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
