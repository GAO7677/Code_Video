#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from .experiment_presets import ROUND1_MODES
except ImportError:
    from experiment_presets import ROUND1_MODES


SCRIPT_PATH = Path(__file__).resolve().parent / "wan_openvid_0613pybullet_lorav2v_vjepa.py"
DEFAULT_OUTPUT_BASE = Path("/data/gaoya/AAA_test_video/0623/test/v2v/loramodel")
DEFAULT_PYTHON_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA+V-JEPA batch generation across the 7 preset modes.")
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json-list-path", type=Path, required=True)
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mode-ids", nargs="*", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_modes = ROUND1_MODES
    if args.mode_ids:
        wanted = set(args.mode_ids)
        selected_modes = [mode for mode in ROUND1_MODES if mode.mode_id in wanted]
        missing = wanted.difference(mode.mode_id for mode in selected_modes)
        if missing:
            raise ValueError(f"Unknown mode_ids: {sorted(missing)}")

    failures: list[str] = []
    total = len(selected_modes)
    for idx, mode in enumerate(selected_modes, start=1):
        model_name = f"{args.model_name_prefix}_{mode.mode_id}"
        output_root = args.output_base / model_name
        runtime_root = args.output_base / f"{model_name}_runtime"

        cmd = [
            str(args.python_bin),
            str(SCRIPT_PATH),
            "--weights-root",
            str(args.weights_root),
            "--input-json-list-path",
            str(args.input_json_list_path),
            "--model-name",
            model_name,
            "--wan-root",
            str(args.wan_root),
            "--output-root",
            str(output_root),
            "--runtime-root",
            str(runtime_root),
            "--device",
            args.device,
            "--vjepa-device",
            args.vjepa_device,
            "--num-frames",
            str(args.num_frames),
            "--context-frames",
            str(args.context_frames),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--cfg-scale",
            str(args.cfg_scale),
            "--seed",
            str(args.seed),
            "--quality",
            str(args.quality),
            "--negative-prompt",
            args.negative_prompt,
            "--conditioning-mode",
            args.conditioning_mode,
            "--context-resize-mode",
            args.context_resize_mode,
            "--log-level",
            args.log_level,
            "--vjepa-model",
            mode.vjepa_model,
            "--vjepa-guidance-steps",
            str(mode.vjepa_guidance_steps),
            "--vjepa-min-step-percent",
            str(mode.vjepa_min_step_percent),
            "--vjepa-max-step-percent",
            str(mode.vjepa_max_step_percent),
            "--vjepa-latent-step-size",
            str(mode.vjepa_latent_step_size),
            "--vjepa-preview-downsample-factor",
            str(mode.vjepa_preview_downsample_factor),
            "--vjepa-preview-frame-stride",
            str(mode.vjepa_preview_frame_stride),
            "--vjepa-window-size",
            str(mode.vjepa_window_size),
            "--vjepa-context-frames",
            str(mode.vjepa_context_frames),
            "--vjepa-stride",
            str(mode.vjepa_stride),
            "--vjepa-reduction",
            str(mode.vjepa_reduction),
            "--vjepa-grad-norm-mode",
            str(mode.vjepa_grad_norm_mode),
            "--vjepa-max-grad-norm",
            str(mode.vjepa_max_grad_norm),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.overwrite:
            cmd.append("--overwrite")
        if mode.disable_vjepa_guidance:
            cmd.append("--disable-vjepa-guidance")

        print(f"[{idx}/{total}] [RUN] {model_name}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(cmd), flush=True)
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append(model_name)
            print(f"[{idx}/{total}] [FAIL] {model_name} returncode={result.returncode}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        print("FAILED_MODELS:", flush=True)
        for model_name in failures:
            print(model_name, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
