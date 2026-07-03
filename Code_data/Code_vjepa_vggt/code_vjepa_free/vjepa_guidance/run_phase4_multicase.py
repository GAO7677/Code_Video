#!/usr/bin/env python3
from __future__ import annotations

"""
Run command:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=7,6 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase4_multicase.py \
  --weights-root /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500 \
  --input-json-list-path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/phase4_pilot3_cases.txt \
  --model-name-prefix phase4_pilot3 \
  --device cuda:0 \
  --vjepa-device cuda:1
"""

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "wan_openvid_0613pybullet_lorav2v_vjepa.py"
DEFAULT_OUTPUT_BASE = Path("/data/gaoya/agent-data/outputs/vjepa_phase4_multicase")
DEFAULT_PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_INPUT_LIST = Path(__file__).resolve().parent / "phase4_pilot3_cases.txt"


@dataclass(frozen=True)
class Phase4Mode:
    mode_id: str
    description: str
    disable_vjepa_guidance: bool
    target_step_indices: tuple[int, ...] = ()
    latent_step_size: float = 0.0
    inner_k: int = 1
    backtracking: bool = False


def _dense_percents(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1:
        return [round((lo + hi) / 2, 3)]
    return [round(lo + i * (hi - lo) / (n - 1), 3) for i in range(n)]


# Exact probe-time indices from phase5/phase6. Do not recompute these from
# percents at runtime; that drifts due to rounding and stops being the same
# winner config we actually validated.
MID12_INDICES = (8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25)
EARLY12_INDICES = (4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21)

PHASE4_TOP_MODES = [
    Phase4Mode(
        mode_id="baseline",
        description="Wan2.2 LoRA baseline without V-JEPA guidance.",
        disable_vjepa_guidance=True,
    ),
    Phase4Mode(
        mode_id="ladder_s20",
        description="Phase 5 winner: dense mid-band guidance, step size 0.20.",
        disable_vjepa_guidance=False,
        target_step_indices=MID12_INDICES,
        latent_step_size=0.20,
        inner_k=1,
    ),
    Phase4Mode(
        mode_id="knee_mid_s18",
        description="Phase 6 near-winner: dense mid-band guidance, step size 0.18.",
        disable_vjepa_guidance=False,
        target_step_indices=MID12_INDICES,
        latent_step_size=0.18,
        inner_k=1,
    ),
    Phase4Mode(
        mode_id="knee_early_s15",
        description="Phase 6 timing variant: dense early-band guidance, step size 0.15.",
        disable_vjepa_guidance=False,
        target_step_indices=EARLY12_INDICES,
        latent_step_size=0.15,
        inner_k=1,
    ),
    Phase4Mode(
        mode_id="knee_mid_s10_k2",
        description="Phase 6 inner-K variant: dense mid-band guidance, step size 0.10, inner_k=2.",
        disable_vjepa_guidance=False,
        target_step_indices=MID12_INDICES,
        latent_step_size=0.10,
        inner_k=2,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current top V-JEPA guidance configs on a small multi-case v2v pilot.")
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json-list-path", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--model-name-prefix", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vjepa-device", type=str, default="cuda:1")
    parser.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--conditioning-mode", choices=["context_aware", "input_image_only"], default="context_aware")
    parser.add_argument("--context-resize-mode", choices=["auto", "crop", "pad"], default="auto")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode-ids", nargs="*", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _selected_modes(args: argparse.Namespace) -> list[Phase4Mode]:
    modes = PHASE4_TOP_MODES
    if args.skip_baseline:
        modes = [mode for mode in modes if not mode.disable_vjepa_guidance]
    if args.mode_ids:
        wanted = set(args.mode_ids)
        modes = [mode for mode in modes if mode.mode_id in wanted]
        missing = wanted.difference(mode.mode_id for mode in modes)
        if missing:
            raise ValueError(f"Unknown mode_ids: {sorted(missing)}")
    return modes


def main() -> None:
    args = parse_args()
    modes = _selected_modes(args)
    failures: list[str] = []

    for idx, mode in enumerate(modes, start=1):
        model_name = f"{args.model_name_prefix}_{mode.mode_id}"
        output_root = args.output_base / model_name
        runtime_root = args.output_base / f"{model_name}_runtime"

        cmd = [
            str(args.python_bin),
            str(SCRIPT_PATH),
            "--weights-root", str(args.weights_root),
            "--input-json-list-path", str(args.input_json_list_path),
            "--model-name", model_name,
            "--wan-root", str(args.wan_root),
            "--output-root", str(output_root),
            "--runtime-root", str(runtime_root),
            "--device", args.device,
            "--vjepa-device", args.vjepa_device,
            "--height", str(args.height),
            "--width", str(args.width),
            "--num-frames", str(args.num_frames),
            "--context-frames", str(args.context_frames),
            "--num-inference-steps", str(args.num_inference_steps),
            "--cfg-scale", str(args.cfg_scale),
            "--seed", str(args.seed),
            "--quality", str(args.quality),
            "--negative-prompt", args.negative_prompt,
            "--conditioning-mode", args.conditioning_mode,
            "--context-resize-mode", args.context_resize_mode,
            "--log-level", args.log_level,
            "--vjepa-model", "vith",
            "--vjepa-guidance-mode", "context_anchored",
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if mode.disable_vjepa_guidance:
            cmd.append("--disable-vjepa-guidance")
        else:
            cmd.extend([
                "--vjepa-guidance-steps", str(len(mode.target_step_indices)),
                "--vjepa-target-step-indices",
                *[str(value) for value in mode.target_step_indices],
                "--vjepa-latent-step-size", str(mode.latent_step_size),
                "--vjepa-inner-k", str(mode.inner_k),
            ])
            if mode.backtracking:
                cmd.append("--vjepa-backtracking")

        print(f"[{idx}/{len(modes)}] [RUN] {model_name}: {mode.description}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(cmd), flush=True)
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append(model_name)
            print(f"[{idx}/{len(modes)}] [FAIL] {model_name} returncode={result.returncode}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        print("FAILED_MODELS:", flush=True)
        for model_name in failures:
            print(model_name, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
